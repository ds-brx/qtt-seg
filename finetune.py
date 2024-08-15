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
import os

def modify_model_for_camvid(model, num_classes=21):
    if isinstance(model, models.segmentation.DeepLabV3):
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif isinstance(model, models.segmentation.LRASPP):
        model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1))
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
        model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    elif config["model"] == 1:
        model = models.segmentation.lraspp_mobilenet_v3_large(weights=models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
    
    model = modify_model_for_camvid(model, num_classes=num_classes)
    model, optimiser = modify_model_and_optimizer(
        model,
        pct_to_freeze=config["pct_to_freeze"],
        layer_decay=config["layer_decay"],
        lr=config["lr"]
    )
    if os.path.exists("models/{}_checkpoint_{}.pth".format(config["model"],budget)):
        print("Checkpoint found, loading...")
        model.load_state_dict(torch.load("models/{}_checkpoint_{}.pth".format(config["model"],budget)))

    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    
    for epoch in range(budget,budget+1):
        running_loss = 0.0
        
        # Wrap the training loop with tqdm
        for n, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{budget}", unit="batch")):
            images, masks = images.to(device), masks.to(device)
            
            optimiser.zero_grad()
            try:
                outputs = model(images)['out']
            except:
                continue
        
            
            # masks = masks.argmax(dim=1)  # Convert masks to the correct format
            masks = torch.squeeze(masks,1)
            loss = criterion(outputs, masks)
            loss.backward()
            optimiser.step()
            
            running_loss += loss.item()

            # Update tqdm with the current loss
            tqdm.write(f"Epoch {epoch+1} Batch {n+1} Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Final Loss: {running_loss/len(train_loader):.4f}")
        
    avg_loss = validate_config(model, val_loader, device, criterion)
    torch.save(model.state_dict(), "models/{}_checkpoint_{}.pth".format(config["model"],budget+1))
    print("Checkpoint saved...")
    
    total_time = (time.time() - start_time)/3600
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
