import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Any

class VisionTransformerModel(nn.Module):
    """
    Vision Transformer (ViT) model for chicken feces classification
    """
    def __init__(self, num_classes=4, model_variant='vit_base_patch16_224', dropout=0.1):
        """
        Initialize Vision Transformer model for chicken feces classification
        
        Args:
            num_classes (int): Number of output classes
            model_variant (str): ViT model variant ('vit_base_patch16_224', 'vit_large_patch16_224', etc.)
            dropout (float): Dropout rate for regularization
        """
        super(VisionTransformerModel, self).__init__()
        
        # Load pre-trained ViT model
        print(f"Loading pre-trained {model_variant} model...")
        try:
            self.vit = timm.create_model(model_variant, pretrained=True)
            print(f"{model_variant} model successfully loaded")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Get the in_features for head replacement
        if hasattr(self.vit, 'head'):
            in_features = self.vit.head.in_features
            
            # Replace head with custom classifier
            self.vit.head = nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.LayerNorm(in_features // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_features // 2, num_classes)
            )
        else:
            raise AttributeError(f"Model {model_variant} structure not supported")
            
        print(f"Classification head replaced with enhanced architecture")
    
    def forward(self, x):
        """Forward pass through the model"""
        return self.vit(x)

class EfficientNetVariants(nn.Module):
    """
    EfficientNet model variants for chicken feces classification
    """
    def __init__(self, num_classes=4, model_variant='b2', dropout=0.3):
        """
        EfficientNet model for chicken feces classification
        
        Args:
            num_classes (int): Number of output classes
            model_variant (str): EfficientNet variant (b0, b1, b2, b3, etc)
            dropout (float): Dropout rate for regularization
        """
        super(EfficientNetVariants, self).__init__()
        
        # Model name based on variant
        model_name = f'efficientnet_{model_variant}'
        
        # Load pre-trained model
        print(f"Loading pre-trained {model_name} model...")
        try:
            self.model = timm.create_model(model_name, pretrained=True)
            print(f"{model_name} model successfully loaded")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Get the in_features for classifier replacement
        if hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            # Replace classifier with custom head
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, in_features // 2),
                nn.SiLU(),  # SiLU (Swish) activation
                nn.BatchNorm1d(in_features // 2),
                nn.Dropout(dropout / 2),
                nn.Linear(in_features // 2, num_classes)
            )
        else:
            raise AttributeError(f"Model {model_name} structure not supported")
            
        print(f"Classification head modified with more complex architecture")
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)

class MobileNetWithAttention(nn.Module):
    """
    MobileNetV3 with attention mechanism for better accuracy
    """
    def __init__(self, num_classes=4, dropout=0.3):
        """
        MobileNetV3 model with attention mechanism
        
        Args:
            num_classes (int): Number of output classes
            dropout (float): Dropout rate for regularization
        """
        super(MobileNetWithAttention, self).__init__()
        
        # Load pre-trained MobileNetV3
        print("Loading pre-trained MobileNetV3 model...")
        try:
            self.base_model = timm.create_model('mobilenetv3_large_100', pretrained=True)
            print("MobileNetV3 model successfully loaded")
        except Exception as e:
            print(f"Error loading MobileNetV3 model: {e}")
            raise
        
        # Get features from the base model (everything except the classifier)
        if hasattr(self.base_model, 'classifier'):
            self.features = nn.Sequential(*list(self.base_model.children())[:-1])
            in_features = self.base_model.classifier.in_features
            
            # Channel Attention mechanism
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_features, in_features // 16, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features // 16, in_features, 1, bias=False),
                nn.Sigmoid()
            )
            
            # Classifier with more complex architecture
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(in_features // 2),
                nn.Dropout(dropout / 2),
                nn.Linear(in_features // 2, num_classes)
            )
            
        else:
            raise AttributeError("MobileNetV3 structure not supported")
            
        print(f"MobileNet with attention mechanism created")
    
    def forward(self, x):
        """Forward pass with attention mechanism"""
        x = self.base_model.features(x)
        
        # Apply channel attention
        attention = self.channel_attention(x)
        x = x * attention
        
        # Global pooling and flatten
        x = self.base_model.global_pool(x)
        x = x.flatten(1)
        
        # Classifier
        x = self.classifier(x)
        return x

class EnsembleModel(nn.Module):
    """
    Ensemble model combining multiple models for better results
    """
    def __init__(self, models, num_classes=4):
        """
        Initialize ensemble with list of models
        
        Args:
            models (list): List of PyTorch models for ensemble
            num_classes (int): Number of output classes
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        
        # Optional: add weights for each model
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
    
    def forward(self, x):
        """Forward pass with weighted voting"""
        outputs = []
        
        # Get output from each model
        for i, model in enumerate(self.models):
            output = model(x)
            outputs.append(output)
        
        # Stack outputs and apply softmax to each model's output
        stacked = torch.stack([torch.softmax(output, dim=1) for output in outputs], dim=0)
        
        # Apply softmax to weights
        weights = torch.softmax(self.weights, dim=0).view(-1, 1, 1)
        
        # Weighted average
        weighted_output = (stacked * weights).sum(dim=0)
        
        # Convert back to logits
        # Add a small epsilon to avoid log(0)
        weighted_output = torch.log(weighted_output + 1e-7)
        
        return weighted_output

# Factory function to create a model based on needs
def create_model(model_type='vit_b16', num_classes=4, dropout=0.1) -> nn.Module:
    """
    Factory function to create appropriate model based on needs
    
    Args:
        model_type (str): Type of model to create
                          'vit_b16': Vision Transformer Base/16
                          'vit_b32': Vision Transformer Base/32
                          'vit_l16': Vision Transformer Large/16
                          'efficient': base EfficientNet-B0
                          'efficient_b1': EfficientNet-B1
                          'efficient_b2': EfficientNet-B2
                          'efficient_b3': EfficientNet-B3
                          'mobile': MobileNetV3 with attention
                          'ensemble': Ensemble of models
        num_classes (int): Number of output classes
        dropout (float): Dropout rate for regularization
    
    Returns:
        model: PyTorch model
    """
    if model_type == 'vit_b16':
        return VisionTransformerModel(num_classes=num_classes, model_variant='vit_base_patch16_224', dropout=dropout)
    elif model_type == 'vit_b32':
        return VisionTransformerModel(num_classes=num_classes, model_variant='vit_base_patch32_224', dropout=dropout)
    elif model_type == 'vit_l16':
        return VisionTransformerModel(num_classes=num_classes, model_variant='vit_large_patch16_224', dropout=dropout)
    elif model_type == 'efficient':
        return EfficientNetVariants(num_classes=num_classes, model_variant='b0', dropout=dropout)
    elif model_type == 'efficient_b1':
        return EfficientNetVariants(num_classes=num_classes, model_variant='b1', dropout=dropout)
    elif model_type == 'efficient_b2':
        return EfficientNetVariants(num_classes=num_classes, model_variant='b2', dropout=dropout)
    elif model_type == 'efficient_b3':
        return EfficientNetVariants(num_classes=num_classes, model_variant='b3', dropout=dropout)
    elif model_type == 'mobile':
        return MobileNetWithAttention(num_classes=num_classes, dropout=dropout)
    elif model_type == 'ensemble':
        # Create ensemble with multiple models, including ViT
        models = [
            VisionTransformerModel(num_classes=num_classes, model_variant='vit_base_patch16_224'),
            EfficientNetVariants(num_classes=num_classes, model_variant='b2'),
            MobileNetWithAttention(num_classes=num_classes)
        ]
        return EnsembleModel(models, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")