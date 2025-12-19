"""
Interactive demo for vision risk classification.
Usage: python app.py
"""

import gradio as gr
import torch
from PIL import Image
from src.model import build_model, get_transforms

def load_model(model_path='experiments/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(num_classes=2, model_name='resnet50')
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print("⚠️  No trained model found. Using untrained model for demo.")
    
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()
transform = get_transforms(image_size=224, augment=False)

def classify_image(image):
    """Classify a single image"""
    if image is None:
        return "Please upload an image"
    
    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        unsafe_prob = probs[0][1].item()
        safe_prob = probs[0][0].item()
    
    # Format output
    if unsafe_prob > 0.5:
        result = f"🚨 **UNSAFE** ({unsafe_prob*100:.2f}% confidence)"
        recommendation = "⚠️  Flag for human review"
    else:
        result = f"✅ **SAFE** ({safe_prob*100:.2f}% confidence)"
        recommendation = "✓ Approved"
    
    return f"{result}\n\n{recommendation}\n\nSafe: {safe_prob*100:.1f}% | Unsafe: {unsafe_prob*100:.1f}%"

# Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Classification Result"),
    title="🔍 Vision Risk Classifier",
    description="Binary classifier for safe vs unsafe content using fine-tuned ResNet50. Achieved 98.5% accuracy on validation set.",
    examples=[],  # Add example image paths if available
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
