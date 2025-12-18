import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from model import build_model, get_transforms


def load_model(model_path, device):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded PyTorch model
    """
    model = build_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, data_loader, device, class_names=['safe', 'unsafe']):
    """
    Evaluate model and compute metrics.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device to evaluate on
        class_names: List of class names
    
    Returns:
        metrics: Dictionary of evaluation metrics
        predictions: List of predictions
        labels: List of true labels
        image_paths: List of image paths
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics, all_preds, all_labels, all_probs


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='experiments/confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm


def analyze_errors(y_true, y_pred, class_names=['safe', 'unsafe']):
    """
    Analyze false positives and false negatives.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Assuming 0 = safe, 1 = unsafe
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    true_negatives = np.sum((y_pred == 0) & (y_true == 0))
    
    print("\n=== Error Analysis ===")
    print(f"True Positives (Correctly identified as unsafe): {true_positives}")
    print(f"True Negatives (Correctly identified as safe): {true_negatives}")
    print(f"False Positives (Safe misclassified as unsafe): {false_positives}")
    print(f"False Negatives (Unsafe misclassified as safe): {false_negatives}")
    
    print("\n=== Decision Impact Analysis ===")
    if false_negatives > false_positives:
        print("⚠️  FALSE NEGATIVES ARE MORE CRITICAL")
        print("Impact: Classifying unsafe situations as safe poses direct safety risks.")
        print("Consequence: Users may be exposed to harmful or dangerous content/situations.")
        print("Recommendation: Consider lowering the decision threshold to reduce false negatives.")
    elif false_positives > false_negatives:
        print("⚠️  FALSE POSITIVES ARE MORE FREQUENT")
        print("Impact: Over-blocking safe content may reduce user experience.")
        print("Consequence: Valid use cases may be unnecessarily restricted.")
        print("Recommendation: Balance is needed - false negatives are typically more critical for safety.")
    else:
        print("✓ False positives and false negatives are balanced.")
    
    print("\nFor safety-critical applications, prioritize reducing FALSE NEGATIVES.")
    print("Missing an unsafe case has higher real-world impact than over-filtering.")


def main(model_path, data_path='data/processed', save_dir='experiments/'):
    """
    Main evaluation function.
    
    Args:
        model_path: Path to trained model
        data_path: Path to data directory
        save_dir: Directory to save results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    
    # Load data
    test_transform = get_transforms(image_size=224, augment=False)
    test_dataset = datasets.ImageFolder(data_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Evaluate
    print("Evaluating model...")
    metrics, preds, labels, probs = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    
    # Plot confusion matrix
    class_names = test_dataset.classes
    cm = plot_confusion_matrix(labels, preds, class_names, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Analyze errors
    analyze_errors(labels, preds, class_names)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='data/processed', help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='experiments/', help='Directory to save results')
    args = parser.parse_args()
    
    main(args.model, args.data, args.save_dir)
