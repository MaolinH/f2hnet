import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
import random
import numpy as np

from models.f2hnet import *

class Config:
    val_path = 'imagenet1k/val'  # overwrite with your validation dataset path
    batch_size = 128     
    num_classes = 1000  
    model_name = 'f2hnet_t'     # overwrite with your model name: 'f2hnet_t', 'f2hnet_s', or 'f2hnet_b'
    checkpoint = 'f2hnet_t.pth'  # overwrite with your checkpoint path
    device = 'cuda:0'         # overwrite with your device, 'cuda:x', 'cpu'
    num_workers = 4

def build_val_loader():
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_dataset = ImageFolder(Config.val_path, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size,
                            num_workers=Config.num_workers, shuffle=False,
                             pin_memory=True)
    return val_loader

def build_model():
    if Config.model_name == 'f2hnet_t':
        model = f2hnet_tiny(num_classes=Config.num_classes)
    elif Config.model_name == 'f2hnet_s':
        model = f2hnet_small(num_classes=Config.num_classes)
    elif Config.model_name == 'f2hnet_b':
        model = f2hnet_base(num_classes=Config.num_classes)
    else:
        return
    checkpoint = torch.load(Config.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    return model

def test():
    model = build_model().to(Config.device)
    val_loader = build_val_loader()
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_samples = 0
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(Config.device, non_blocking=True)
            targets = targets.to(Config.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            
            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(targets).sum().item()

            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct = top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred))
            correct_top5 += top5_correct.any(dim=1).sum().item()
            
            total_samples += images.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Val Batch: {batch_idx}/{len(val_loader)}')
    
    avg_loss = total_loss / total_samples
    top1_acc = 100. * correct_top1 / total_samples
    top5_acc = 100. * correct_top5 / total_samples
    
    print(f'\nValidation Results:')
    print(f'Total Samples: {total_samples}')
    print(f'Avg Loss: {avg_loss:.6f}')
    print(f'Top-1 Accuracy: {top1_acc:.2f}% ({correct_top1}/{total_samples})')
    print(f'Top-5 Accuracy: {top5_acc:.2f}% ({correct_top5}/{total_samples})')
    
    return {
        'loss': avg_loss,
        'top1_acc': top1_acc,
        'top5_acc': top5_acc
    }

if __name__ == '__main__':
    results = test()
