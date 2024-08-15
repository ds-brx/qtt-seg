from datasets import CamVidDataset,PascalVOCDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from finetune import validate_config
import torch
from torchvision import models
import torch.nn as nn

train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,256))
    ])
val_dataset = PascalVOCDataset(root_dir='/work/dlclarge2/dasb-Camvid/VOCdevkit/VOC2007',split='val',transform = train_transform)

num_classes = 21

val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model 1
model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)

if isinstance(model, models.segmentation.DeepLabV3):
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
elif isinstance(model, models.segmentation.LRASPP):
    model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1))

model.to(device)
criterion = nn.CrossEntropyLoss()

loss = validate_config(model, val_loader, device, criterion)


