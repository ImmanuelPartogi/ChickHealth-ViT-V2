"""
Chicken Health Classification - Utilities
----------------------------------------------------------
Optimized utility functions for training, evaluating, and visualizing
models for chicken feces classification.
"""
import os
import time
import random
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('chicken_classifier')

# Constants
DEFAULT_CLASSES = ["Chicken_Coccidiosis", "Chicken_Healthy", "Chicken_NewCastleDisease", "Chicken_Salmonella"]
COLORMAP = 'Blues'
FIGURE_SIZE = (10, 8)

# ======================================================
# MODEL PREPARATION FUNCTIONS
# ======================================================
def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # Enable benchmark for faster training when input sizes don't change
        torch.backends.cudnn.benchmark = True
    
    logger.info(f"Random seed set to {seed} for reproducibility")

def setup_finetuning(model: nn.Module, freeze_backbone: bool = True, model_type: str = 'vit_b16') -> nn.Module:
    """
    Configure model for fine-tuning by selectively freezing parameters.
    
    Args:
        model: The PyTorch model
        freeze_backbone: Whether to freeze backbone layers
        model_type: Type of model ('vit_b16', 'efficient_b2', etc.)
    
    Returns:
        model: The model with modified parameters for fine-tuning
    """
    if freeze_backbone:
        # Freeze backbone layers based on model type
        logger.info(f"Freezing backbone layers for {model_type} transfer learning")
        
        # Handle different model architectures
        if 'vit' in model_type:
            if hasattr(model, 'vit'):
                for name, param in model.vit.named_parameters():
                    if 'head' not in name:  # Freeze all layers except head
                        param.requires_grad = False
        elif hasattr(model, 'model'):  # EfficientNet or MobileNet
            for name, param in model.named_parameters():
                if 'classifier' not in name and 'head' not in name:
                    param.requires_grad = False
    
    # Report trainable parameter statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0
    
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)')
    
    return model

def unfreeze_layers(model: nn.Module, num_layers: int = 2, model_type: str = 'vit_b16') -> nn.Module:
    """
    Unfreeze specific layers for further fine-tuning.
    Works with different model architectures.
    
    Args:
        model: The PyTorch model
        num_layers: Number of layers to unfreeze from the end
        model_type: Type of model ('vit_b16', 'efficient_b2', etc.)
    
    Returns:
        model: The model with modified parameters
    """
    logger.info(f"Unfreezing the last {num_layers} layers of {model_type}")
    
    # Handle different model architectures
    if 'vit' in model_type:
        # For ViT models, unfreeze the last N transformer blocks
        if hasattr(model, 'vit'):
            # Base ViT has 12 blocks, Large has 24
            total_blocks = 12  # Default for ViT-Base
            if 'l16' in model_type:
                total_blocks = 24  # ViT-Large
            
            blocks_to_unfreeze = [f'blocks.{total_blocks-i-1}' for i in range(num_layers)]
            logger.info(f"Unfreezing ViT blocks: {blocks_to_unfreeze}")
            
            for name, param in model.vit.named_parameters():
                if any(block in name for block in blocks_to_unfreeze) or 'head' in name:
                    param.requires_grad = True
    
    elif hasattr(model, 'model'):  # EfficientNet or MobileNet
        # For EfficientNet, unfreeze the last stage and classifier
        if hasattr(model.model, 'blocks'):  # EfficientNet
            layers = list(model.model.blocks.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        elif hasattr(model.model, 'stages'):  # MobileNetV3
            stages = list(model.model.stages.children())
            for stage in stages[-num_layers:]:
                for param in stage.parameters():
                    param.requires_grad = True
        
        # Always unfreeze classifier
        if hasattr(model.model, 'classifier'):
            for param in model.model.classifier.parameters():
                param.requires_grad = True
    
    # Report updated trainable parameter statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0
    
    logger.info(f'Trainable parameters after unfreezing: {trainable_params:,} ({trainable_pct:.2f}%)')
    
    return model

# ======================================================
# VISUALIZATION FUNCTIONS
# ======================================================
def plot_confusion_matrix(cm: np.ndarray, 
                         classes: List[str] = DEFAULT_CLASSES, 
                         save_path: str = 'confusion_matrix.png',
                         figsize: Tuple[int, int] = FIGURE_SIZE) -> None:
    """
    Plot and save confusion matrix with enhanced styling.
    
    Args:
        cm: Confusion matrix
        classes: List of class names
        save_path: Path to save the plot
        figsize: Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)
    
    # Calculate accuracy and set title
    accuracy = np.trace(cm) / float(np.sum(cm))
    title = f'Confusion Matrix\nAccuracy: {accuracy:.2%}'
    
    # Display class names without the "Chicken_" prefix for better readability
    display_classes = [cls.replace('Chicken_', '') for cls in classes]
    
    # Create heatmap with improved styling
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=COLORMAP, 
        xticklabels=display_classes, 
        yticklabels=display_classes,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    # Improve label placement and styling
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Ensure tight layout and save figure
    plt.tight_layout()
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving confusion matrix: {str(e)}")
    
    plt.close()

def plot_training_history(history: Dict[str, List[float]], 
                         save_path: str = 'training_history.png',
                         figsize: Tuple[int, int] = (14, 6),
                         model_type: str = 'vit_b16') -> None:
    """
    Plot and save training history with enhanced styling.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        model_type: Type of model for title customization
    """
    plt.figure(figsize=figsize)
    
    # Create a plot with a two-column layout
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'o-', color='#4285F4', label='Training', linewidth=2, markersize=6)
    plt.plot(history['val_loss'], 'o-', color='#DB4437', label='Validation', linewidth=2, markersize=6)
    plt.title('Loss Curves', fontsize=14, pad=10)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    
    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'o-', color='#4285F4', label='Training', linewidth=2, markersize=6)
    plt.plot(history['val_acc'], 'o-', color='#DB4437', label='Validation', linewidth=2, markersize=6)
    plt.title('Accuracy Curves', fontsize=14, pad=10)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    
    # Add overall title with model type
    model_name = model_type.upper().replace('_', '-')
    plt.suptitle(f'Training and Validation Metrics for {model_name}', fontsize=16, y=0.98)
    
    # Ensure layout is optimal
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving training history plot: {str(e)}")
    
    plt.close()

def visualize_model_outputs(image_path: str, model: nn.Module, device: torch.device, 
                           classes: List[str] = DEFAULT_CLASSES, 
                           save_path: Optional[str] = None,
                           model_type: str = 'vit_b16') -> None:
    """
    Visualize model predictions on a single image.
    
    Args:
        image_path: Path to input image
        model: The trained model
        device: Device to run inference on
        classes: List of class names
        save_path: Path to save visualization (optional)
        model_type: Type of model to determine preprocessing
    """
    from PIL import Image
    from torchvision import transforms
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Determine preprocessing based on model type
    if 'vit' in model_type:
        transform = transforms.Compose([
            transforms.Resize(248),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Preprocess image
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get predictions
    probs = probabilities.cpu().numpy()
    class_idx = np.argmax(probs)
    predicted_class = classes[class_idx]
    confidence = probs[class_idx] * 100
    
    # Display class names without the "Chicken_" prefix for better readability
    display_classes = [cls.replace('Chicken_', '') for cls in classes]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(img)
    ax1.set_title(f"Prediction: {predicted_class.replace('Chicken_', '')}\nConfidence: {confidence:.2f}%", 
                 fontsize=14, color='darkblue')
    ax1.axis('off')
    
    # Create bar chart with probabilities
    y_pos = np.arange(len(display_classes))
    ax2.barh(y_pos, probs * 100, color='skyblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(display_classes)
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Class Probabilities')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add values to the bars
    for i, v in enumerate(probs * 100):
        ax2.text(v + 1, i, f"{v:.1f}%", va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

# ======================================================
# UTILITY CLASSES
# ======================================================
class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    Stops training when validation loss doesn't improve for a given number of epochs.
    """
    def __init__(self, patience=5, min_delta=0, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait before stopping after loss stops improving
            min_delta (float): Minimum change in loss to qualify as an improvement
            verbose (bool): If True, prints a message when training is stopped early
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        """
        Check if training should be stopped
        
        Args:
            val_loss (float): Current validation loss
        """
        if val_loss < self.best_loss - self.min_delta:
            # Loss improved
            self.best_loss = val_loss
            self.counter = 0
        else:
            # Loss didn't improve
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info('Early stopping triggered')

# ======================================================
# DATA ANALYSIS FUNCTIONS
# ======================================================
def analyze_dataset(data_dir: str, classes: List[str] = DEFAULT_CLASSES) -> Dict[str, Any]:
    """
    Analyze the dataset structure and provide statistics.
    
    Args:
        data_dir: Path to dataset directory
        classes: List of class names
    
    Returns:
        dict: Dataset statistics
    """
    import glob
    from PIL import Image
    
    stats = {
        'total_images': 0,
        'class_distribution': {},
        'image_sizes': [],
        'aspect_ratios': [],
        'classes': classes
    }
    
    # Check if train/test structure exists
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Process split dataset
        stats['dataset_structure'] = 'train/test split'
        stats['train'] = {'total': 0, 'per_class': {}}
        stats['test'] = {'total': 0, 'per_class': {}}
        
        # Process training set
        for cls in classes:
            class_path = os.path.join(train_dir, cls)
            if os.path.exists(class_path):
                files = glob.glob(os.path.join(class_path, '*.*'))
                count = len(files)
                stats['train']['per_class'][cls] = count
                stats['train']['total'] += count
                stats['total_images'] += count
                
                # Sample image sizes
                for i, file in enumerate(files[:10]):  # Sample 10 images per class
                    try:
                        with Image.open(file) as img:
                            width, height = img.size
                            stats['image_sizes'].append((width, height))
                            stats['aspect_ratios'].append(width/height)
                    except Exception as e:
                        logger.warning(f"Could not analyze image {file}: {e}")
        
        # Process test set
        for cls in classes:
            class_path = os.path.join(test_dir, cls)
            if os.path.exists(class_path):
                files = glob.glob(os.path.join(class_path, '*.*'))
                count = len(files)
                stats['test']['per_class'][cls] = count
                stats['test']['total'] += count
                stats['total_images'] += count
    else:
        # Process flat dataset
        stats['dataset_structure'] = 'flat'
        
        for cls in classes:
            class_path = os.path.join(data_dir, cls)
            if os.path.exists(class_path):
                files = glob.glob(os.path.join(class_path, '*.*'))
                count = len(files)
                stats['class_distribution'][cls] = count
                stats['total_images'] += count
                
                # Sample image sizes
                for i, file in enumerate(files[:10]):  # Sample 10 images per class
                    try:
                        with Image.open(file) as img:
                            width, height = img.size
                            stats['image_sizes'].append((width, height))
                            stats['aspect_ratios'].append(width/height)
                    except Exception as e:
                        logger.warning(f"Could not analyze image {file}: {e}")
    
    # Calculate average image size
    if stats['image_sizes']:
        avg_width = sum(w for w, h in stats['image_sizes']) / len(stats['image_sizes'])
        avg_height = sum(h for w, h in stats['image_sizes']) / len(stats['image_sizes'])
        stats['average_image_size'] = (avg_width, avg_height)
        stats['average_aspect_ratio'] = sum(stats['aspect_ratios']) / len(stats['aspect_ratios'])
    
    return stats

def print_dataset_stats(stats: Dict[str, Any]) -> None:
    """
    Print formatted dataset statistics.
    
    Args:
        stats: Dataset statistics from analyze_dataset()
    """
    print("\n===== DATASET STATISTICS =====")
    print(f"Total images: {stats['total_images']}")
    print(f"Dataset structure: {stats['dataset_structure']}")
    
    if stats['dataset_structure'] == 'train/test split':
        print("\nTraining set:")
        print(f"  Total: {stats['train']['total']} images")
        for cls, count in stats['train']['per_class'].items():
            percentage = (count / stats['train']['total']) * 100
            print(f"  {cls}: {count} images ({percentage:.1f}%)")
        
        print("\nTest set:")
        print(f"  Total: {stats['test']['total']} images")
        for cls, count in stats['test']['per_class'].items():
            percentage = (count / stats['test']['total']) * 100
            print(f"  {cls}: {count} images ({percentage:.1f}%)")
    else:
        print("\nClass distribution:")
        for cls, count in stats['class_distribution'].items():
            percentage = (count / stats['total_images']) * 100
            print(f"  {cls}: {count} images ({percentage:.1f}%)")
    
    if 'average_image_size' in stats:
        print(f"\nAverage image size: {stats['average_image_size'][0]:.1f} x {stats['average_image_size'][1]:.1f}")
        print(f"Average aspect ratio: {stats['average_aspect_ratio']:.2f}")
    
    print("==============================\n")

def plot_class_distribution(stats: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot class distribution as a bar chart.
    
    Args:
        stats: Dataset statistics from analyze_dataset()
        save_path: Path to save visualization (optional)
    """
    plt.figure(figsize=(10, 6))
    
    if stats['dataset_structure'] == 'train/test split':
        # Create bar chart for train/test split
        train_counts = []
        test_counts = []
        classes = []
        
        # Use sorted order for better visualization
        for cls in sorted(stats['classes']):
            if cls in stats['train']['per_class']:
                train_counts.append(stats['train']['per_class'].get(cls, 0))
                test_counts.append(stats['test']['per_class'].get(cls, 0))
                classes.append(cls.replace('Chicken_', ''))  # Remove prefix for display
        
        x = np.arange(len(classes))
        width = 0.35
        
        # Create grouped bar chart
        plt.bar(x - width/2, train_counts, width, label='Training Set', color='#4285F4')
        plt.bar(x + width/2, test_counts, width, label='Test Set', color='#DB4437')
        
        plt.xlabel('Class')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution in Training and Test Sets')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        
    else:
        # Create bar chart for flat dataset
        counts = []
        classes = []
        
        # Use sorted order for better visualization
        for cls in sorted(stats['classes']):
            if cls in stats['class_distribution']:
                counts.append(stats['class_distribution'].get(cls, 0))
                classes.append(cls.replace('Chicken_', ''))  # Remove prefix for display
        
        # Create bar chart
        plt.bar(classes, counts, color='#4285F4')
        plt.xlabel('Class')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution in Dataset')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()