import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.cuda.amp import autocast
import torchvision.transforms.functional as TF
from model import create_model
from dataset import IMAGE_SIZE, DEFAULT_CLASSES

def load_model(model_path, model_type='vit_b16', num_classes=len(DEFAULT_CLASSES), device='cuda'):
    """
    Load a trained model from disk
    
    Args:
        model_path (str): Path to model weights file
        model_type (str): Type of model to load
        num_classes (int): Number of output classes
        device (str): Device to load model on (cuda/cpu)
    
    Returns:
        model: Loaded PyTorch model
    """
    model = create_model(model_type=model_type, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, img_size=IMAGE_SIZE, model_type='vit_b16'):
    """
    Preprocess an image for model inference
    
    Args:
        image_path (str): Path to input image
        img_size (int): Size to resize image to
        model_type (str): Model type to determine optimal preprocessing
    
    Returns:
        tensor: Preprocessed image tensor
    """
    # Transformations for inference - custom based on model type
    if 'vit' in model_type:
        # ViT models typically use center crop for validation/inference
        transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.1)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # For CNN models like EfficientNet/MobileNet
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Open and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def test_time_augmentation(model, image_tensor, device, num_augmentations=10, classes=DEFAULT_CLASSES, use_amp=False):
    """
    Apply Test Time Augmentation (TTA) for more accurate prediction
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor
        device: Device for inference
        num_augmentations: Number of augmentations to apply
        classes: List of class names
        use_amp: Use automatic mixed precision
    
    Returns:
        tuple: (predicted_class, confidence, all_scores)
    """
    # Set model to eval mode
    model.eval()
    
    # Make sure image tensor is on the right device
    image_tensor = image_tensor.to(device)
    
    # Create augmentations for TTA - these work for both CNN and ViT models
    tta_transforms = [
        # Original image (no augmentation)
        lambda x: x,
        
        # Horizontal flip
        lambda x: TF.hflip(x),
        
        # Vertical flip
        lambda x: TF.vflip(x),
        
        # 90 degree rotation
        lambda x: TF.rotate(x, 90),
        
        # 180 degree rotation
        lambda x: TF.rotate(x, 180),
        
        # 270 degree rotation
        lambda x: TF.rotate(x, 270),
        
        # Brightness adjustment +10%
        lambda x: TF.adjust_brightness(x, 1.1),
        
        # Brightness adjustment -10%
        lambda x: TF.adjust_brightness(x, 0.9),
        
        # Contrast adjustment +10%
        lambda x: TF.adjust_contrast(x, 1.1),
        
        # Contrast adjustment -10%
        lambda x: TF.adjust_contrast(x, 0.9),
    ]
    
    # Use a subset of transforms if num_augmentations is less than available transforms
    if num_augmentations < len(tta_transforms):
        tta_transforms = tta_transforms[:num_augmentations]
    
    # List to store all predictions
    all_probs = []
    
    # Apply each augmentation and get prediction
    with torch.no_grad():
        for transform_fn in tta_transforms:
            # Apply the transform to the input image
            augmented = transform_fn(image_tensor)
            
            # Predict with mixed precision if enabled
            if use_amp:
                with autocast():
                    outputs = model(augmented)
            else:
                outputs = model(augmented)
            
            # Store softmax probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            all_probs.append(probs)
    
    # Average the predictions from all augmentations
    avg_probs = torch.stack(all_probs).mean(dim=0)
    
    # Get final prediction
    predicted_class_idx = torch.argmax(avg_probs).item()
    class_name = classes[predicted_class_idx]
    confidence = avg_probs[predicted_class_idx].item() * 100
    
    # Get all confidence scores
    all_scores = [(classes[i], prob.item() * 100) for i, prob in enumerate(avg_probs)]
    all_scores.sort(key=lambda x: x[1], reverse=True)
    
    return class_name, confidence, all_scores

def predict(model, image_tensor, device='cuda', classes=DEFAULT_CLASSES, use_amp=False, use_tta=False):
    """
    Make prediction on an image
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        classes: List of class names
        use_amp: Whether to use automatic mixed precision
        use_tta: Whether to use Test Time Augmentation
    
    Returns:
        tuple: (predicted_class, confidence, all_scores)
    """
    # Use TTA if specified for improved accuracy
    if use_tta:
        return test_time_augmentation(model, image_tensor, device, 10, classes, use_amp)
    
    # Move tensor to appropriate device
    image_tensor = image_tensor.to(device)
    
    # Get model prediction
    with torch.no_grad():
        if use_amp and device == 'cuda':
            with autocast():
                outputs = model(image_tensor)
        else:
            outputs = model(image_tensor)
            
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # Get class name and confidence
    class_name = classes[predicted_class]
    confidence = probabilities[predicted_class].item() * 100
    
    # Get all confidence scores
    all_scores = [(classes[i], prob.item() * 100) for i, prob in enumerate(probabilities)]
    all_scores.sort(key=lambda x: x[1], reverse=True)
    
    return class_name, confidence, all_scores

def visualize_prediction(image_path, class_name, confidence, all_scores, output_path=None):
    """
    Visualize the prediction with the original image
    
    Args:
        image_path: Path to original image
        class_name: Predicted class name
        confidence: Confidence score for prediction
        all_scores: All confidence scores for all classes
        output_path: Path to save visualization (optional)
    """
    # Open the original image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Display image with improved styling
    ax1.imshow(image)
    ax1.set_title(f"Prediction: {class_name}\nConfidence: {confidence:.2f}%", 
                 fontsize=14, color='darkblue', pad=10)
    ax1.axis('off')
    
    # Display confidence scores for all classes
    class_names = [x[0].replace('Chicken_', '') for x in all_scores]  # Remove "Chicken_" prefix for clarity
    scores = [x[1] for x in all_scores]
    
    # Create horizontal bar chart with improved styling
    bars = ax2.barh(class_names, scores, color='lightblue', height=0.6)
    
    # Highlight the predicted class
    for i, (cls, score) in enumerate(all_scores):
        if cls == class_name:
            bars[i].set_color('darkblue')
    
    # Add percentage labels to bars
    for i, v in enumerate(scores):
        ax2.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=12)
    
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=14, pad=10)
    ax2.set_xlim(0, 105)  # Add some room for the text
    ax2.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def batch_predict(model, image_folder, device='cuda', classes=DEFAULT_CLASSES, use_amp=False, use_tta=False, output_folder=None, model_type='vit_b16'):
    """
    Predict on a batch of images in a folder
    
    Args:
        model: PyTorch model
        image_folder: Folder containing images
        device: Device to run inference on
        classes: List of class names
        use_amp: Whether to use automatic mixed precision
        use_tta: Whether to use Test Time Augmentation
        output_folder: Folder to save visualizations (optional)
        model_type: Model type to determine preprocessing
    
    Returns:
        dict: Predictions for each image
    """
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.isfile(os.path.join(image_folder, f)) and 
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = {}
    
    print(f"\nProcessing {len(image_files)} images with{'out' if not use_tta else ''} Test Time Augmentation...")
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        
        # Preprocess and predict
        image_tensor = preprocess_image(img_path, model_type=model_type)
        class_name, confidence, all_scores = predict(model, image_tensor, device, classes, use_amp, use_tta)
        
        # Save results
        results[img_file] = {
            'class': class_name,
            'confidence': confidence,
            'all_scores': all_scores
        }
        
        # Visualize if output folder is provided
        if output_folder:
            output_path = os.path.join(output_folder, f"pred_{os.path.splitext(img_file)[0]}.png")
            visualize_prediction(img_path, class_name, confidence, all_scores, output_path)
        
        print(f"Prediction for {img_file}: {class_name} ({confidence:.2f}%)")
    
    # Print summary
    class_counts = {}
    for result in results.values():
        class_name = result['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\nSummary of predictions:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images ({count/len(results)*100:.1f}%)")
    
    return results

def main(args):
    """
    Main function for inference
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Check for mixed precision
    use_amp = args.mixed_precision and device.type == 'cuda'
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for inference")
    
    try:
        # Load model
        print(f"Loading model from {args.model_path} (type: {args.model_type})")
        model = load_model(args.model_path, args.model_type, device=device)
        
        if args.batch_mode:
            # Batch prediction on a folder
            if not os.path.isdir(args.image_path):
                print(f"Error: {args.image_path} is not a directory")
                return
                
            results = batch_predict(
                model, 
                args.image_path, 
                device, 
                DEFAULT_CLASSES, 
                use_amp, 
                args.use_tta, 
                args.output_path,
                args.model_type
            )
            
            print(f"Batch prediction completed. Processed {len(results)} images.")
            
        else:
            # Single image prediction
            if not os.path.isfile(args.image_path):
                print(f"Error: Image {args.image_path} not found")
                return
                
            # Preprocess image
            image_tensor = preprocess_image(args.image_path, model_type=args.model_type)
            
            # Make prediction
            class_name, confidence, all_scores = predict(
                model, 
                image_tensor, 
                device, 
                DEFAULT_CLASSES, 
                use_amp,
                args.use_tta
            )
            
            # Print results
            print(f"\nPrediction: {class_name}")
            print(f"Confidence: {confidence:.2f}%\n")
            print("All class probabilities:")
            for cls, score in all_scores:
                print(f"  {cls}: {score:.2f}%")
            
            # Visualize if requested
            if args.visualize:
                output_path = args.output_path if args.output_path else None
                visualize_prediction(args.image_path, class_name, confidence, all_scores, output_path)
            
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict chicken health condition from fecal image')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image or directory of images')
    parser.add_argument('--model_path', type=str, default='models/best_vit_chicken_classifier.pth',
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='vit_b16',
                        choices=['vit_b16', 'vit_b32', 'vit_l16', 'efficient', 'efficient_b1', 
                                'efficient_b2', 'efficient_b3', 'mobile', 'ensemble'],
                        help='Type of model to load')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the prediction')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save visualization or output directory for batch mode')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision for inference')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process all images in the specified directory')
    parser.add_argument('--use_tta', action='store_true',
                        help='Use Test Time Augmentation for more accurate predictions')
    
    args = parser.parse_args()
    main(args)