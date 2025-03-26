import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import numpy as np
import random
from tqdm import tqdm
from model import create_model
from dataset import create_dataloaders, create_simple_dataloaders, DEFAULT_CLASSES, calculate_class_weights
import utils

# ======================================================
# TRAINING AND EVALUATION FUNCTIONS
# ======================================================
def train_epoch(model, dataloader, criterion, optimizer, device, use_amp=False, scaler=None):
    """Train for one epoch with mixed precision support"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress tracking
    batch_time = utils.AverageMeter('Batch Time', ':6.3f')
    data_time = utils.AverageMeter('Data Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    accuracy = utils.AverageMeter('Accuracy', ':6.2f')
    
    end = time.time()
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc="Training", unit="batch")
    
    for inputs, labels in pbar:
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Get inputs and labels
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Handle mixup/cutmix labels
        if isinstance(labels, torch.Tensor) and labels.dim() > 1 and labels.shape[1] > 1:
            # One-hot encoded labels from mixup/cutmix
            batch_size = inputs.size(0)
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = torch.sum(-labels * torch.log_softmax(outputs, dim=1), dim=1).mean()
                
                # Backward pass with scaled gradients
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = torch.sum(-labels * torch.log_softmax(outputs, dim=1), dim=1).mean()
                loss.backward()
                optimizer.step()
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)
            _, labels_idx = torch.max(labels, 1)
            correct_samples = (predicted == labels_idx).sum().item()
            
            # Update statistics
            running_loss += loss.item() * batch_size
            correct += correct_samples
            total += batch_size
            
            # Update meters
            losses.update(loss.item(), batch_size)
            accuracy.update(100.0 * correct_samples / batch_size, batch_size)
            
        else:
            # Standard labels
            batch_size = inputs.size(0)
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Backward pass with scaled gradients
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)
            correct_samples = (predicted == labels).sum().item()
            
            # Update statistics
            running_loss += loss.item() * batch_size
            correct += correct_samples
            total += batch_size
            
            # Update meters
            losses.update(loss.item(), batch_size)
            accuracy.update(100.0 * correct_samples / batch_size, batch_size)
        
        # Update batch time meter
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracy.avg:.2f}%',
            'batch_time': f'{batch_time.avg:.3f}s'
        })
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100.0 * correct / total
    
    return epoch_loss, epoch_accuracy

def validate_epoch(model, dataloader, criterion, device, classes=DEFAULT_CLASSES, use_amp=False):
    """Validate the model with mixed precision support"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Progress tracking
    batch_time = utils.AverageMeter('Batch Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    accuracy = utils.AverageMeter('Accuracy', ':6.2f')
    
    end = time.time()
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc="Validating", unit="batch")
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Handle mixup/cutmix labels (although they shouldn't be in val set)
            if isinstance(labels, torch.Tensor) and labels.dim() > 1 and labels.shape[1] > 1:
                # Convert to class indices
                _, labels = torch.max(labels, 1)
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Statistics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            correct_samples = (predicted == labels).sum().item()
            correct += correct_samples
            
            # Save predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update meters
            losses.update(loss.item(), batch_size)
            accuracy.update(100.0 * correct_samples / batch_size, batch_size)
            batch_time.update(time.time() - end)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracy.avg:.2f}%'
            })
            
            end = time.time()
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100.0 * correct / total
    
    return epoch_loss, epoch_accuracy, all_preds, all_labels

def evaluate_model(all_labels, all_preds, classes=DEFAULT_CLASSES):
    """Evaluate model performance with detailed metrics"""
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate per-class metrics
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Calculate average metrics
    avg_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    avg_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    avg_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    print(report)
    
    print(f"Average Metrics - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    
    # Log per-class metrics
    for i, class_name in enumerate(classes):
        print(f"{class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")
    
    return cm, report

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=20, device='cuda', classes=DEFAULT_CLASSES, save_dir='.',
                use_amp=True, early_stopping_patience=5, model_type='vit_b16'):
    """Train and validate model with mixed precision and early stopping"""
    best_val_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Initialize early stopping
    early_stopping = utils.EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if use_amp and device == 'cuda' else None
    
    # Create models directory within save_dir
    models_dir = os.path.join(save_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nTraining {model_type} model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} starting")
        epoch_start_time = time.time()
        
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            use_amp=use_amp, scaler=scaler
        )
        
        # Update scheduler if provided
        if scheduler:
            if isinstance(scheduler, OneCycleLR):
                # OneCycleLR steps every batch, not every epoch
                pass
            else:
                scheduler.step()
        
        # Validation phase
        val_loss, val_acc, all_preds, all_labels = validate_epoch(
            model, val_loader, criterion, device, classes, use_amp=use_amp
        )
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        
        # Save best model in the models directory
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            if 'vit' in model_type:
                model_path = os.path.join(models_dir, 'best_vit_chicken_classifier.pth')
            else:
                model_path = os.path.join(models_dir, 'best_chicken_classifier.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Model saved with validation accuracy: {best_val_accuracy:.2f}%')
            
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Load best model for final evaluation
    if 'vit' in model_type:
        best_model_path = os.path.join(models_dir, 'best_vit_chicken_classifier.pth')
    else:
        best_model_path = os.path.join(models_dir, 'best_chicken_classifier.pth')
    
    model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation on validation set
    print("\nFinal Model Evaluation:")
    _, val_acc, all_preds, all_labels = validate_epoch(
        model, val_loader, criterion, device, classes, use_amp=use_amp
    )
    cm, _ = evaluate_model(all_labels, all_preds, classes)
    
    # Plot confusion matrix
    cm_path = os.path.join(save_dir, f'confusion_matrix_{model_type}.png')
    utils.plot_confusion_matrix(cm, classes, cm_path)
    
    # Plot training history
    history_path = os.path.join(save_dir, f'training_history_{model_type}.png')
    utils.plot_training_history(history, history_path, model_type=model_type)
    
    return history, best_val_accuracy

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    print(f"Random seed set to {seed} for reproducibility")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train chicken feces classifier with ViT')
    parser.add_argument('--data_dir', type=str, default='chicken_feces_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save model and results')
    parser.add_argument('--model_type', type=str, default='vit_b16',
                        choices=['vit_b16', 'vit_b32', 'vit_l16', 'efficient', 'efficient_b1', 
                                'efficient_b2', 'efficient_b3', 'mobile', 'ensemble'],
                        help='Model type to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (reduce for ViT if needed)')
    parser.add_argument('--head_epochs', type=int, default=10,
                        help='Number of epochs for head classifier training')
    parser.add_argument('--finetune_epochs', type=int, default=20,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Initial learning rate (use smaller LR for ViT)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--simple_dataset', action='store_true',
                        help='Use simplified dataset without mixup/cutmix (good for CPU)')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Use simple dataset by default on CPU unless specified otherwise
    use_simple_dataset = args.simple_dataset or device.type == 'cpu'
    
    # Adjust batch size for CPU if needed
    if device.type == 'cpu' and args.batch_size > 8 and 'vit' in args.model_type:
        original_batch_size = args.batch_size
        args.batch_size = 8
        print(f"Reduced batch size from {original_batch_size} to {args.batch_size} for CPU training with ViT")
    
    # Reduce worker count for CPU
    if device.type == 'cpu':
        args.num_workers = min(args.num_workers, 1)
    
    # Check if mixed precision is available
    use_amp = args.mixed_precision and device.type == 'cuda' and torch.cuda.is_available()
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) training")
    
    # Create output directory and models subdirectory
    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n=== STARTING TRAINING WITH {args.model_type.upper()} ===")
    
    # Create data loaders with appropriate augmentation based on model type
    try:
        if use_simple_dataset:
            print("Using simplified dataset without mixup/cutmix (recommended for CPU)")
            train_loader, val_loader = create_simple_dataloaders(
                args.data_dir, 
                batch_size=args.batch_size,
                train_ratio=args.train_ratio,
                num_workers=args.num_workers,
                strong_aug=True,
                model_type=args.model_type,
                oversampling=True
            )
        else:
            train_loader, val_loader = create_dataloaders(
                args.data_dir, 
                batch_size=args.batch_size,
                train_ratio=args.train_ratio,
                num_workers=args.num_workers,
                strong_aug=True,
                use_mixup=True,
                use_cutmix=True,
                oversampling=True,
                model_type=args.model_type
            )
    except Exception as e:
        print(f"Fatal error creating DataLoader: {e}")
        return
    
    # Create model
    try:
        model = create_model(model_type=args.model_type, num_classes=len(DEFAULT_CLASSES))
        model = utils.setup_finetuning(model, freeze_backbone=True, model_type=args.model_type)
        model = model.to(device)
    except Exception as e:
        print(f"Fatal error preparing model: {e}")
        return
    
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(args.data_dir, device)
    print(f"Class weights: {class_weights}")
    
    # Loss function with class weights and label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # PHASE 1: Train only the classifier head - use lower learning rate for ViT
    if 'vit' in args.model_type:
        base_lr = args.learning_rate / 2  # Lower learning rate for ViT
    else:
        base_lr = args.learning_rate
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=1e-4  # Increased weight decay for better regularization
    )
    
    # OneCycleLR scheduler for faster convergence
    total_steps = len(train_loader) * args.head_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of time warming up
        div_factor=25,  # Start with lr/25
        final_div_factor=10000,  # End with lr/10000
    )
    
    print(f"=== PHASE 1: Training head classifier for {args.model_type} ===")
    history, best_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=args.head_epochs, device=device, save_dir=args.output_dir,
        use_amp=use_amp, early_stopping_patience=7, model_type=args.model_type
    )
    
    # PHASE 2: Unfreeze more layers and continue training
    # Use smaller learning rate for Vision Transformer fine-tuning
    model = utils.unfreeze_layers(model, num_layers=4, model_type=args.model_type)
    
    # Use differential learning rates for different parts of the model
    if 'vit' in args.model_type:
        # For ViT, use lower learning rates
        optimizer = optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if 'head' in n], 
             'lr': base_lr/10},
            {'params': [p for n, p in model.named_parameters() 
                        if 'head' not in n and p.requires_grad], 
             'lr': base_lr/30}
        ], weight_decay=1e-5)
    else:
        # For CNN-based models
        optimizer = optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if 'classifier' in n or 'head' in n], 
             'lr': base_lr/5},
            {'params': [p for n, p in model.named_parameters() 
                        if 'classifier' not in n and 'head' not in n and p.requires_grad], 
             'lr': base_lr/20}
        ], weight_decay=1e-4)
    
    # Cosine annealing scheduler with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  # Restart every 5 epochs
        T_mult=1,  # Don't increase restart period
        eta_min=1e-6  # Minimum learning rate
    )
    
    print(f"\n=== PHASE 2: Fine-tuning {args.model_type} with more layers unlocked ===")
    history, best_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=args.finetune_epochs, device=device, save_dir=args.output_dir,
        use_amp=use_amp, early_stopping_patience=5, model_type=args.model_type
    )
    
    print("Training completed!")
    if 'vit' in args.model_type:
        print(f"Best model saved to {os.path.join(models_dir, 'best_vit_chicken_classifier.pth')}")
    else:
        print(f"Best model saved to {os.path.join(models_dir, 'best_chicken_classifier.pth')}")
    
    print(f"Final accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()