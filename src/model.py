import torch
import torch.nn as nn
from torchvision import models, transforms


def build_model(num_classes: int = 2, model_name: str = 'resnet18', freeze_backbone: bool = True):
    """
    Build a transfer learning model with a pretrained CNN backbone.
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        model_name: Name of the pretrained model architecture (default: 'resnet18')
                   Options: 'resnet18', 'resnet50', 'mobilenet_v2'
        freeze_backbone: Whether to freeze backbone weights (default: True)
    
    Returns:
        model: PyTorch model ready for training
    """
    # Load pretrained model
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = model.fc.in_features
        # Replace final fully connected layer
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        num_features = model.classifier[1].in_features
        # Replace final classifier
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
    
    else:
        raise ValueError(f"Model {model_name} not supported. Use 'resnet18', 'resnet50', or 'mobilenet_v2'.")
    
    # Freeze backbone layers if specified
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:  # Don't freeze the classifier head
                param.requires_grad = False
    
    return model


def get_transforms(image_size: int = 224, augment: bool = False):
    """
    Get image preprocessing transforms.
    
    Args:
        image_size: Target image size for resizing (default: 224)
        augment: Whether to apply data augmentation (default: False)
    
    Returns:
        transform: torchvision transforms composition
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform
