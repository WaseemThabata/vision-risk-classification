import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from datetime import datetime
from model import build_model, get_transforms


def load_config(config_path='configs/config.yaml'):
    """
    Load hyperparameters from YAML config file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        config: Dictionary of hyperparameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train(config_path='configs/config.yaml', data_path='data/processed'):
    """
    Main training function.
    
    Args:
        config_path: Path to config file
        data_path: Path to processed data directory
    """
    # Load config
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_transform = get_transforms(config['image_size'], augment=True)
    val_transform = get_transforms(config['image_size'], augment=False)
    
    # Assuming ImageFolder structure: data/processed/train/ and data/processed/val/
    # Or use random_split if you have a single folder
    full_dataset = datasets.ImageFolder(data_path, transform=train_transform)
    train_size = int(config['train_val_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply different transforms
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Build model
    model = build_model(num_classes=2, model_name=config['model_name'])
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            momentum=0.9
        )
    
    # Training loop
    best_val_acc = 0.0
    checkpoint_dir = config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(checkpoint_dir, f"best_model_{timestamp}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_path)
            print(f"  Saved best model to {save_path}")
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    train()
