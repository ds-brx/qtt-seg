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

def train_config(config,budget,num_classes,train_dataset,val_dataset):
    print("Training...")
    start_time = time.time()
    config = config["config"]
    train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config["model"] == 0:
        model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    if config["model"] == 1:
        model = models.segmentation.fcn_resnet50(pretrained=True)
    if config["model"] == 2:
        model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)
    

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
        for n, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimiser.zero_grad()
            outputs = model(images)['out']
            
            masks = masks.squeeze(1) 
            loss = criterion(outputs, masks)
            loss.backward()
            optimiser.step()
            
            running_loss += loss.item()

            print("Epoch {} Batch {} Loss {}".format(epoch, n,loss.item()))
        print("Epoch {} Final Loss {}".format(epoch,running_loss/len(train_loader)))
        
    avg_mean_iou = validate_config(model,val_loader,device,num_classes)

    total_time = time.time() - start_time()
    return avg_mean_iou, total_time

def modify_model_and_optimizer(model, pct_to_freeze=0.0, layer_decay=None, lr=1e-4):
    layers = list(model.children())
    num_layers = len(layers)

    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False

    if pct_to_freeze > 0.0:
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
        freeze_layers = int(num_layers * layer_decay)
        
        for i in range(num_layers):
            if i >= freeze_layers:
                for param in layers[i].parameters():
                    param.requires_grad = True

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_train, lr=lr)

    return model, optimizer


def validate_config(model,valid_loader,device,num_classes):
    model.eval()
    pixel_accuracy_list = []
    mean_iou_list = []
    
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1) 
            outputs = model(images)['out']
            outputs = torch.argmax(outputs, dim=1)  # Get the predicted class
            
            pixel_accuracy, mean_iou = compute_metrics(outputs, masks, num_classes)
            pixel_accuracy_list.append(pixel_accuracy)
            mean_iou_list.append(mean_iou)
    
    avg_pixel_accuracy = np.mean(pixel_accuracy_list)
    avg_mean_iou = np.mean(mean_iou_list)
    
    print(f'Validation Pixel Accuracy: {avg_pixel_accuracy:.4f}')
    print(f'Validation Mean IoU: {avg_mean_iou:.4f}')
    return avg_mean_iou


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