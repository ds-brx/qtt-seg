import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import time
import random
from torchvision import models
from tqdm import tqdm
import pandas as pd

class CamVidDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, class_dict_csv='/content/drive/MyDrive/Camvid/CamVid/class_dict.csv'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Directories for images and masks
        self.images_dir = os.path.join(root_dir, split)
        self.masks_dir = os.path.join(root_dir, f'{split}_labels')
        
        # List of images and masks
        self.images = []
        self.masks = []
        
        # Populate the lists with image and mask paths
        for img_name in os.listdir(self.images_dir):
            img_path = os.path.join(self.images_dir, img_name)
            mask_name = img_name.replace('.png', '_L.png')
            mask_path = os.path.join(self.masks_dir, mask_name)
            
            self.images.append(img_path)
            self.masks.append(mask_path)
        
        # Load the class mappings from CSV
        self.rgb_to_class = self.load_class_mappings(class_dict_csv)
        
    def load_class_mappings(self, csv_path):
        # Read the CSV file
        df = pd.read_csv(csv_path, sep='\t', header=None, names=['Class', 'R', 'G', 'B'])
        
        # Create a mapping from RGB tuples to class indices
        rgb_to_class = {}
        for idx, row in df.iterrows():
            rgb = (row['R'], row['G'], row['B'])
            rgb_to_class[rgb] = idx  # Use the row index as the class index
        
        return rgb_to_class

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Convert mask to class indices
        mask = np.array(mask)
        h, w, _ = mask.shape
        mask_indices = np.zeros((h, w))
        
        for rgb, class_idx in self.rgb_to_class.items():
            mask_indices[np.all(mask == rgb, axis=-1)] = class_idx
        
        # image = torch.tensor(image)
        # mask = torch.tensor(mask_indices, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask_indices)
        mask = mask.to(torch.long)
        
        return image, mask


class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.masks_dir = os.path.join(root_dir, 'gtFine', split)

        
        self.images = []
        self.masks = []
        
        for city in os.listdir(self.images_dir):
            city_images_dir = os.path.join(self.images_dir, city)
            city_masks_dir = os.path.join(self.masks_dir, city)
            
            for img_name in os.listdir(city_images_dir):
                img_path = os.path.join(city_images_dir, img_name)
                mask_name = img_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                mask_path = os.path.join(city_masks_dir, mask_name)
                
                self.images.append(img_path)
                self.masks.append(mask_path)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        return image, mask


def modify_model_for_camvid(model, num_classes=11):
    if isinstance(model, models.segmentation.DeepLabV3):
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif isinstance(model, models.segmentation.FCN):
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif isinstance(model, models.segmentation.LRASPP):
        model.classifier = nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        raise ValueError("Unsupported model type")

    return model

def train_config(config, budget, num_classes, train_dataset, val_dataset):
    print("Training...")
    start_time = time.time()
    config = config["config"]
    print(config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config["model"] == 0:
        model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    elif config["model"] == 1:
        model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)
    elif config["model"] == 2:
        model = models.segmentation.lraspp_mobilenet_v3_large(weights=models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
    
    model = modify_model_for_camvid(model, num_classes=num_classes)
    model, optimiser = modify_model_and_optimizer(
        model,
        pct_to_freeze=config["pct_to_freeze"],
        layer_decay=config["layer_decay"],
        lr=config["lr"]
    )

    optimiser = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    
    for epoch in range(budget):
        running_loss = 0.0
        
        # Wrap the training loop with tqdm
        for n, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{budget}", unit="batch")):
            images, masks = images.to(device), masks.to(device)
            
            optimiser.zero_grad()
            outputs = model(images)['out']
            
            # masks = masks.argmax(dim=1)  # Convert masks to the correct format
            masks = torch.squeeze(masks,1)
            print(outputs.shape, masks.shape)
            loss = criterion(outputs, masks)
            loss.backward()
            optimiser.step()
            
            running_loss += loss.item()

            # Update tqdm with the current loss
            tqdm.write(f"Epoch {epoch+1} Batch {n+1} Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Final Loss: {running_loss/len(train_loader):.4f}")
        
    avg_loss = validate_config(model, val_loader, device, criterion)

    total_time = time.time() - start_time
    return avg_loss, total_time

def modify_model_and_optimizer(model, pct_to_freeze=0.0, layer_decay=None, lr=1e-4):
    layers = list(model.children())
    num_layers = len(layers)

    # for layer in layers:
    #     for param in layer.parameters():
    #         param.requires_grad = False

    if pct_to_freeze > 0.0:
        print("PCT")
        total_params = sum(p.numel() for p in model.parameters())
        params_to_freeze = int(total_params * pct_to_freeze)

        frozen_params = 0
        for layer in layers:
            for param in layer.parameters():
                if frozen_params >= params_to_freeze:
                    break
                param.requires_grad = False
                frozen_params += param.numel()
            if frozen_params >= params_to_freeze:
                break

    if layer_decay > 0.0:
        print("Layer Decay")
        freeze_layers = int(num_layers * layer_decay)
        
        for i in range(num_layers):
            if i >= freeze_layers:
                for param in layers[i].parameters():
                    param.requires_grad = True

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_train, lr=lr)

    return model, optimizer


def validate_config(model, valid_loader, device, loss_fn):
    model.eval()
    loss_list = []
    
    with torch.no_grad():
        for images, masks in tqdm(valid_loader, desc="Validating", unit="batch"):
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1)
            
            outputs = model(images)['out']
            
            loss = loss_fn(outputs, masks)
            loss_list.append(loss.item())
    
    avg_loss = np.mean(loss_list)
    
    print(f'Validation Loss: {avg_loss:.4f}')
    
    return avg_loss

def compute_metrics(predictions, targets, num_classes):
    """
    Compute evaluation metrics like pixel accuracy and mean IoU.
    """
    predictions = predictions.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    
    conf_matrix = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    
    pixel_accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    
    iou_per_class = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=1) + np.sum(conf_matrix, axis=0) - np.diag(conf_matrix))
    iou_per_class[np.isnan(iou_per_class)] = 0  # Handle division by zero (for classes not present in the batch)
    mean_iou = np.mean(iou_per_class)
    
    return pixel_accuracy, mean_iou

# Dummy Training
# def train_config(config,budget,num_classes,train_dataset,val_dataset):
#     print("Dummy Training")
#     perf = random.random()
#     cost = random.randint(30,40)
#     return perf, cost