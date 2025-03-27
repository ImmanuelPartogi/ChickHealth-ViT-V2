import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Tuple, List, Optional, Dict, Any, Union

# Default class names
DEFAULT_CLASSES = ["Chicken_Coccidiosis", "Chicken_Healthy", "Chicken_NewCastleDisease", "Chicken_Salmonella"]

# Image size - for Vision Transformer we need at least 224x224
IMAGE_SIZE = 224

class AdvancedAugmentation:
    """Class for advanced augmentation with CutMix and MixUp"""
    
    @staticmethod
    def mixup(img1, img2, label1, label2, num_classes, alpha=0.2):
        """
        Implement MixUp augmentation
        https://arxiv.org/abs/1710.09412
        """
        lam = np.random.beta(alpha, alpha)
        mixed_img = lam * img1 + (1 - lam) * img2
        
        # Create one-hot encoded labels
        label1_onehot = torch.zeros(num_classes)
        label1_onehot[label1] = 1
        label2_onehot = torch.zeros(num_classes)
        label2_onehot[label2] = 1
        
        mixed_label = lam * label1_onehot + (1 - lam) * label2_onehot
        return mixed_img, mixed_label
    
    @staticmethod
    def cutmix(img1, img2, label1, label2, num_classes, alpha=0.2):
        """
        Implement CutMix augmentation
        https://arxiv.org/abs/1905.04899
        """
        lam = np.random.beta(alpha, alpha)
        
        # Get image dimensions
        _, H, W = img1.shape
        
        # Generate random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        # Random center position
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Boundary
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)
        
        # Create mixed image
        mixed_img = img1.clone()
        mixed_img[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]
        
        # Adjust lambda based on area ratio
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        
        # Create one-hot encoded labels
        label1_onehot = torch.zeros(num_classes)
        label1_onehot[label1] = 1
        label2_onehot = torch.zeros(num_classes)
        label2_onehot[label2] = 1
        
        mixed_label = lam * label1_onehot + (1 - lam) * label2_onehot
        return mixed_img, mixed_label

class ChickenFecesDataset(Dataset):
    def __init__(self, data_dir, transform=None, classes=DEFAULT_CLASSES, 
                use_mixup=False, use_cutmix=False, mixup_prob=0.5, 
                oversampling=False):
        """
        Custom dataset for chicken feces classification with advanced features
        
        Args:
            data_dir (str): Path to dataset directory
            transform: Transformations to apply to images
            classes (list): List of class names
            use_mixup (bool): Whether to use mixup augmentation
            use_cutmix (bool): Whether to use cutmix augmentation
            mixup_prob (float): Probability of applying mixup/cutmix
            oversampling (bool): Whether to use oversampling for minority classes
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_prob = mixup_prob
        self.num_classes = len(classes)
        
        self.samples = []
        self.class_samples = {cls: [] for cls in self.classes}
        
        # Search for all image files in data directory
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                class_files = [f for f in os.listdir(class_dir) 
                              if os.path.isfile(os.path.join(class_dir, f)) and 
                              f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"Class {class_name}: {len(class_files)} images found")
                
                for img_name in class_files:
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                    self.class_samples[class_name].append((img_path, self.class_to_idx[class_name]))
            else:
                print(f"WARNING: Directory {class_dir} not found!")
        
        if not self.samples:
            raise RuntimeError(f"No images found in {data_dir}. Please check the folder structure.")
        
        # Apply oversampling to balance classes if enabled
        if oversampling:
            self.apply_oversampling()
            
        print(f"Total dataset: {len(self.samples)} images after processing")
    
    def apply_oversampling(self):
        """Apply oversampling to balance minority classes"""
        # Find class with maximum samples
        max_samples = max(len(samples) for samples in self.class_samples.values())
        
        # Oversample classes with fewer samples
        for class_name, samples in self.class_samples.items():
            if len(samples) < max_samples:
                # Calculate how many additional samples needed
                additional_needed = max_samples - len(samples)
                # Sample with replacement
                additional_samples = random.choices(samples, k=additional_needed)
                self.samples.extend(additional_samples)
                print(f"Added {additional_needed} oversampled images to {class_name}")
        
        # Shuffle samples
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Open image and convert to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations if any
            if self.transform:
                image = self.transform(image)
            
            # Convert integer label to tensor immediately to maintain consistency
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            # Apply mixup or cutmix if enabled and if not the last batch
            if (self.use_mixup or self.use_cutmix) and random.random() < self.mixup_prob:
                # Randomly select another sample
                idx2 = random.randint(0, len(self.samples) - 1)
                img_path2, label2 = self.samples[idx2]
                
                # Load and transform second image
                image2 = Image.open(img_path2).convert('RGB')
                if self.transform:
                    image2 = self.transform(image2)
                
                # Apply either mixup or cutmix
                if self.use_mixup and (not self.use_cutmix or random.random() < 0.5):
                    # Apply mixup
                    image, label_vector = AdvancedAugmentation.mixup(
                        image, image2, label, label2, self.num_classes)
                    return image, label_vector
                else:
                    # Apply cutmix
                    image, label_vector = AdvancedAugmentation.cutmix(
                        image, image2, label, label2, self.num_classes)
                    return image, label_vector
            
            # Return image and consistent tensor label    
            return image, label_tensor
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return placeholder image if error
            dummy_img = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
            # Also return a tensor label, not an int
            return dummy_img, torch.tensor(label, dtype=torch.long)

class SimpleChickenFecesDataset(Dataset):
    """
    Simplified dataset without mixup/cutmix to avoid batching issues on CPU
    """
    def __init__(self, data_dir, transform=None, classes=DEFAULT_CLASSES, oversampling=False):
        """
        Simplified dataset for CPU training (no mixup/cutmix)
        
        Args:
            data_dir (str): Path to dataset directory
            transform: Transformations to apply to images
            classes (list): List of class names
            oversampling (bool): Whether to use oversampling for minority classes
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        self.class_samples = {cls: [] for cls in self.classes}
        
        # Search for all image files in data directory
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                class_files = [f for f in os.listdir(class_dir) 
                              if os.path.isfile(os.path.join(class_dir, f)) and 
                              f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"Class {class_name}: {len(class_files)} images found")
                
                for img_name in class_files:
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                    self.class_samples[class_name].append((img_path, self.class_to_idx[class_name]))
            else:
                print(f"WARNING: Directory {class_dir} not found!")
        
        if not self.samples:
            raise RuntimeError(f"No images found in {data_dir}. Please check the folder structure.")
        
        # Apply oversampling to balance classes if enabled
        if oversampling:
            self.apply_oversampling()
            
        print(f"Total dataset: {len(self.samples)} images after processing")
    
    def apply_oversampling(self):
        """Apply oversampling to balance minority classes"""
        # Find class with maximum samples
        max_samples = max(len(samples) for samples in self.class_samples.values())
        
        # Oversample classes with fewer samples
        for class_name, samples in self.class_samples.items():
            if len(samples) < max_samples:
                # Calculate how many additional samples needed
                additional_needed = max_samples - len(samples)
                # Sample with replacement
                additional_samples = random.choices(samples, k=additional_needed)
                self.samples.extend(additional_samples)
                print(f"Added {additional_needed} oversampled images to {class_name}")
        
        # Shuffle samples
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Open image and convert to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations if any
            if self.transform:
                image = self.transform(image)
            
            # Convert to tensor label
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return image, label_tensor
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return placeholder image if error
            dummy_img = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
            return dummy_img, torch.tensor(label, dtype=torch.long)

def get_transforms(train=True, img_size=IMAGE_SIZE, strong_aug=False, model_type='vit_b16'):
    """
    Get image transforms for training or validation
    Enhanced with stronger augmentations and ViT-specific transforms
    
    Args:
        train (bool): Whether to return transforms for training or validation
        img_size (int): Size of the image
        strong_aug (bool): Whether to use stronger augmentation
        model_type (str): Model type to determine specific transformations
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    # Set appropriate normalization values (ImageNet stats)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # For ViT models
    if 'vit' in model_type:
        if train:
            if strong_aug:
                # Strong augmentation strategy for ViT
                return transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.05),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(20),
                    # No heavy affine - ViT is sensitive to severe geometric distortions
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.RandomErasing(p=0.1),
                ])
            else:
                # Standard training augmentation for ViT
                return transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
        else:
            # Validation transforms for ViT - just resize and center crop
            return transforms.Compose([
                transforms.Resize(int(img_size * 1.1)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    # For CNN models (EfficientNet, MobileNet)
    else:
        if train:
            if strong_aug:
                # Strong augmentation strategy
                return transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.05),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(30),
                    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
                ])
            else:
                # Standard training augmentation
                return transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(20),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
        else:
            # Validation transforms (no augmentation)
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

def custom_collate(batch):
    """
    Custom collate function to handle mixed label formats from regular and mixup/cutmix samples.
    
    Args:
        batch: List of tuples (image, label)
        
    Returns:
        images_batch: Tensor of images
        labels_batch: Either tensor of class indices or tensor of one-hot vectors
    """
    images = []
    labels = []
    
    # Check if we have mixed label formats
    label_is_onehot = False
    
    # First pass to determine label format
    for img, lbl in batch:
        if lbl.dim() > 0 and lbl.size(0) > 1:  # One-hot encoded label
            label_is_onehot = True
            break
            
    # Second pass to collect and format data
    for img, lbl in batch:
        images.append(img)
        
        if label_is_onehot:
            # Convert class indices to one-hot if needed
            if lbl.dim() == 0 or lbl.size(0) == 1:
                num_classes = batch[0][1].size(0)  # Get num_classes from the first one-hot label
                onehot = torch.zeros(num_classes)
                onehot[lbl.item() if lbl.dim() == 0 else lbl.item()] = 1.0
                labels.append(onehot)
            else:
                labels.append(lbl)
        else:
            # Convert one-hot to class indices if needed
            if lbl.dim() > 0 and lbl.size(0) > 1:
                labels.append(torch.argmax(lbl).unsqueeze(0))
            else:
                labels.append(lbl)
    
    # Stack images
    images_batch = torch.stack(images, 0)
    
    # Stack labels
    labels_batch = torch.stack(labels, 0)
    
    return images_batch, labels_batch

def create_dataloaders(data_dir, batch_size=32, train_ratio=0.8, classes=DEFAULT_CLASSES, 
                      num_workers=4, strong_aug=False, use_mixup=True, use_cutmix=True,
                      oversampling=True, model_type='vit_b16'):
    """
    Create and return DataLoaders for training and validation
    With advanced augmentation options
    
    Args:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size for DataLoader
        train_ratio (float): Ratio of data to use for training (ignored if train/test folders exist)
        classes (list): List of class names
        num_workers (int): Number of worker processes for data loading
        strong_aug (bool): Whether to use stronger augmentation
        use_mixup (bool): Whether to use mixup augmentation
        use_cutmix (bool): Whether to use cutmix augmentation
        oversampling (bool): Whether to use oversampling for minority classes
        model_type (str): Model type to determine specific transformations
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    print(f"Creating dataset from directory: {data_dir}")
    
    # Verify dataset directory
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # Check if train/test structure exists
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Found train/test directories structure. Using separate datasets.")
        
        # Create train dataset with advanced features
        train_dataset = ChickenFecesDataset(
            train_dir,
            transform=get_transforms(train=True, strong_aug=strong_aug, model_type=model_type),
            classes=classes,
            use_mixup=use_mixup,
            use_cutmix=use_cutmix,
            oversampling=oversampling
        )
        
        # Create test/validation dataset
        val_dataset = ChickenFecesDataset(
            test_dir,
            transform=get_transforms(train=False, model_type=model_type),
            classes=classes
        )
    else:
        # Use the original method with manual split
        print(f"No train/test structure found. Using train_ratio={train_ratio} to split dataset.")
        try:
            full_dataset = ChickenFecesDataset(data_dir, transform=None, classes=classes)
        except Exception as e:
            print(f"Error creating dataset: {e}")
            raise
        
        # Calculate split sizes
        dataset_size = len(full_dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = dataset_size - train_size
        
        print(f"Splitting dataset: {train_size} for training, {val_size} for validation")
        
        # Split dataset
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset samples
        train_samples = [full_dataset.samples[i] for i in train_indices]
        val_samples = [full_dataset.samples[i] for i in val_indices]
        
        # Create datasets with appropriate transforms
        train_dataset = ChickenFecesDataset(
            data_dir, 
            transform=get_transforms(train=True, strong_aug=strong_aug, model_type=model_type),
            classes=classes,
            use_mixup=use_mixup,
            use_cutmix=use_cutmix,
            oversampling=oversampling
        )
        train_dataset.samples = train_samples
        
        val_dataset = ChickenFecesDataset(
            data_dir, 
            transform=get_transforms(train=False, model_type=model_type),
            classes=classes
        )
        val_dataset.samples = val_samples
    
    # Determine optimal number of workers
    if num_workers is None:
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count(), 8)
    
    # For CPU training, reduce worker count to save memory
    if not torch.cuda.is_available():
        num_workers = min(num_workers, 2)
        print(f"Training on CPU: Reduced number of workers to {num_workers}")
    
    # Use custom collate function if mixup/cutmix is enabled
    collate_fn = custom_collate if (use_mixup or use_cutmix) else None
    
    # Create DataLoaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_fn  # Use the custom collate function
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
        # No custom collate for validation as we don't use mixup/cutmix there
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader

def create_simple_dataloaders(data_dir, batch_size=16, train_ratio=0.8, classes=DEFAULT_CLASSES, 
                             num_workers=1, strong_aug=False, model_type='vit_b16', oversampling=True):
    """
    Create and return simpler DataLoaders for CPU training (no mixup/cutmix)
    
    Args:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size for DataLoader
        train_ratio (float): Ratio of data to use for training (ignored if train/test folders exist)
        classes (list): List of class names
        num_workers (int): Number of worker processes for data loading
        strong_aug (bool): Whether to use stronger augmentation
        model_type (str): Model type to determine specific transformations
        oversampling (bool): Whether to use oversampling for minority classes
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    print(f"Creating simple dataset from directory: {data_dir}")
    
    # Verify dataset directory
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # Check if train/test structure exists
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Found train/test directories structure. Using separate datasets.")
        
        # Create train dataset
        train_dataset = SimpleChickenFecesDataset(
            train_dir,
            transform=get_transforms(train=True, strong_aug=strong_aug, model_type=model_type),
            classes=classes,
            oversampling=oversampling
        )
        
        # Create test/validation dataset
        val_dataset = SimpleChickenFecesDataset(
            test_dir,
            transform=get_transforms(train=False, model_type=model_type),
            classes=classes
        )
    else:
        # Use the original method with manual split
        print(f"No train/test structure found. Using train_ratio={train_ratio} to split dataset.")
        try:
            full_dataset = SimpleChickenFecesDataset(data_dir, transform=None, classes=classes)
        except Exception as e:
            print(f"Error creating dataset: {e}")
            raise
        
        # Calculate split sizes
        dataset_size = len(full_dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = dataset_size - train_size
        
        print(f"Splitting dataset: {train_size} for training, {val_size} for validation")
        
        # Split dataset
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset samples
        train_samples = [full_dataset.samples[i] for i in train_indices]
        val_samples = [full_dataset.samples[i] for i in val_indices]
        
        # Create datasets with appropriate transforms
        train_dataset = SimpleChickenFecesDataset(
            data_dir, 
            transform=get_transforms(train=True, strong_aug=strong_aug, model_type=model_type),
            classes=classes,
            oversampling=oversampling
        )
        train_dataset.samples = train_samples
        
        val_dataset = SimpleChickenFecesDataset(
            data_dir, 
            transform=get_transforms(train=False, model_type=model_type),
            classes=classes
        )
        val_dataset.samples = val_samples
    
    # Minimize worker count for CPU training
    if not torch.cuda.is_available():
        num_workers = min(num_workers, 1)
        print(f"Training on CPU: Reduced number of workers to {num_workers}")
    
    # Create DataLoaders with optimized settings for CPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False,  # No need for pin_memory on CPU
        drop_last=True,
        persistent_workers=False  # No need for persistent workers with few workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,  # Don't double batch size on CPU
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader

def calculate_class_weights(data_dir, device):
    """Calculate class weights to handle imbalanced classes"""
    import glob
    
    class_counts = []
    for cls in DEFAULT_CLASSES:
        # Check if train folder exists
        train_path = os.path.join(data_dir, 'train', cls)
        if os.path.exists(train_path):
            count = len(glob.glob(os.path.join(train_path, '*.*')))
        else:
            # Try direct structure
            path = os.path.join(data_dir, cls)
            if os.path.exists(path):
                count = len(glob.glob(os.path.join(path, '*.*')))
            else:
                count = 1  # To avoid division by zero
                print(f"Warning: No images found for class {cls}")
        
        class_counts.append(count)
        print(f"Class {cls}: {count} images")
    
    # Calculate weights with inverse frequency
    class_weights = 1.0 / np.array(class_counts)
    class_weights = class_weights / np.sum(class_weights) * len(class_counts)
    # Adjust weights to make them more balanced, using square root to reduce extremes
    class_weights = np.sqrt(class_weights) * 2
    
    return torch.tensor(class_weights, dtype=torch.float).to(device)