import os
import argparse
import torch
from PIL import Image
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


def predict_image(image_path, model, device, threshold=0.5, class_names=['safe', 'unsafe']):
    """
    Predict class for a single image.
    
    Args:
        image_path: Path to image file
        model: Trained PyTorch model
        device: Device for inference
        threshold: Confidence threshold for unsafe class (default: 0.5)
        class_names: List of class names
    
    Returns:
        predicted_class: Predicted class name
        confidence: Confidence score
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(image_size=224, augment=False)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
    
    confidence = confidence.item()
    predicted_idx = predicted_idx.item()
    
    # Apply threshold logic for safety
    # If unsafe probability > threshold, classify as unsafe
    unsafe_prob = probs[0][1].item()
    
    if unsafe_prob > threshold:
        predicted_class = class_names[1]  # unsafe
        confidence = unsafe_prob
    else:
        predicted_class = class_names[0]  # safe
        confidence = 1 - unsafe_prob
    
    return predicted_class, confidence, unsafe_prob


def main():
    parser = argparse.ArgumentParser(description='Run inference on a single image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, required=True, help='Path to model checkpoint (.pth file)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Threshold for unsafe classification (default: 0.5)')
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not os.path.exists(args.weights):
        print(f"Error: Model weights not found: {args.weights}")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.weights}...")
    model = load_model(args.weights, device)
    
    # Run inference
    print(f"Running inference on {args.image}...")
    predicted_class, confidence, unsafe_prob = predict_image(
        args.image, model, device, threshold=args.threshold
    )
    
    # Print results
    print("\n=== Prediction Results ===")
    print(f"Image: {args.image}")
    print(f"Predicted Class: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"Unsafe Probability: {unsafe_prob:.4f}")
    print(f"Threshold: {args.threshold}")
    
    if predicted_class == 'unsafe':
        print("\nWARNING: This image is classified as UNSAFE")
    else:
        print("\nThis image is classified as SAFE")


if __name__ == '__main__':
    main()
