"""
GradCAM (Gradient-weighted Class Activation Mapping) for CropShield AI
=======================================================================

Provides explainability visualization by highlighting regions of the input image
that most influence the CNN's prediction.

Features:
- Hooks into last convolutional layer
- Computes class-specific gradients
- Generates weighted heatmap overlay
- Supports CPU and GPU execution
- Compatible with Custom CNN and transfer learning models

Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (ICCV 2017)
    https://arxiv.org/abs/1610.02391

Usage:
    from utils.gradcam import GradCAM, generate_gradcam_visualization
    
    # Method 1: Using GradCAM class
    gradcam = GradCAM(model, target_layer=model.block4)
    heatmap = gradcam(image_tensor, class_idx=5)
    
    # Method 2: Using convenience function
    overlay = generate_gradcam_visualization(
        model, image_path, target_layer, device='cuda'
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not installed. Install with: pip install opencv-python")
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Tuple, List
import warnings


class GradCAM:
    """
    GradCAM implementation for PyTorch models.
    
    Computes gradient-weighted class activation maps to visualize which regions
    of the input image contribute most to the model's prediction.
    
    Attributes:
        model: PyTorch model in eval mode
        target_layer: Layer to hook (typically last conv layer)
        device: Device (cuda/cpu)
        gradients: Stored gradients during backward pass
        activations: Stored activations during forward pass
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model (will be set to eval mode)
            target_layer: Layer to hook (e.g., model.block4 for Custom CNN)
            device: Device to use (auto-detected if None)
            
        Example:
            >>> from model_custom_cnn import CropShieldCNN
            >>> model = CropShieldCNN(num_classes=22)
            >>> gradcam = GradCAM(model, target_layer=model.block4)
        """
        self.model = model
        self.target_layer = target_layer
        
        # Auto-detect device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
        
        # Set model to eval mode
        self.model.eval()
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register forward and backward hooks on target layer.
        """
        def forward_hook(module, input, output):
            """Store activations during forward pass."""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Store gradients during backward pass."""
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        retain_graph: bool = False
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap.
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W) or (C, H, W)
            class_idx: Target class index (uses predicted class if None)
            retain_graph: Retain computation graph for multiple backward passes
            
        Returns:
            Heatmap as numpy array of shape (H, W) with values in [0, 1]
            
        Example:
            >>> heatmap = gradcam(image_tensor, class_idx=5)
            >>> # heatmap.shape: (224, 224), values in [0, 1]
        """
        # Ensure batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        # Ensure gradients are enabled
        input_tensor.requires_grad = True
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Use predicted class if not specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Get score for target class
        target_score = output[0, class_idx]
        
        # Backward pass
        target_score.backward(retain_graph=retain_graph)
        
        # Get gradients and activations
        gradients = self.gradients  # Shape: (1, C, H, W)
        activations = self.activations  # Shape: (1, C, H, W)
        
        # Global average pooling of gradients (weights)
        # Shape: (1, C, 1, 1)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        # Shape: (1, H, W)
        weighted_activations = torch.sum(weights * activations, dim=1)
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(weighted_activations)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Clear stored gradients and activations to prevent memory leaks
        self.gradients = None
        self.activations = None
        
        # Clear CUDA cache if using GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return cam
    
    def generate_heatmap_overlay(
        self,
        original_image: Union[np.ndarray, Image.Image],
        cam: np.ndarray,
        colormap: int = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create heatmap overlay on original image.
        
        Args:
            original_image: Original image (RGB numpy array or PIL Image)
            cam: GradCAM heatmap of shape (H, W) with values in [0, 1]
            colormap: OpenCV colormap (default: COLORMAP_JET if cv2 available)
            alpha: Overlay transparency (0=original, 1=heatmap)
            
        Returns:
            RGB image with heatmap overlay as numpy array (H, W, 3)
            
        Example:
            >>> overlay = gradcam.generate_heatmap_overlay(image, cam, alpha=0.5)
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for heatmap overlay. Install with: pip install opencv-python")
        
        if colormap is None:
            colormap = cv2.COLORMAP_JET
        
        # Convert PIL to numpy
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Ensure RGB
        if original_image.shape[-1] != 3:
            raise ValueError(f"Expected RGB image, got shape {original_image.shape}")
        
        # Resize CAM to match original image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert CAM to heatmap
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR
        
        # Overlay heatmap on original image
        overlay = (1 - alpha) * original_image + alpha * heatmap
        overlay = np.uint8(overlay)
        
        return overlay


def get_target_layer(model: nn.Module, model_type: str = 'custom') -> nn.Module:
    """
    Get the appropriate target layer for GradCAM based on model type.
    
    Args:
        model: PyTorch model
        model_type: Model architecture ('custom', 'efficientnet_b0', 'resnet18')
        
    Returns:
        Target layer (last convolutional layer)
        
    Raises:
        ValueError: If model type is not supported
        
    Example:
        >>> target_layer = get_target_layer(model, 'custom')
        >>> # Returns model.block4 for Custom CNN
    """
    if model_type == 'custom':
        # Custom CNN: last conv block (block4)
        if hasattr(model, 'block4'):
            return model.block4
        else:
            raise ValueError("Custom CNN should have 'block4' attribute")
    
    elif model_type == 'efficientnet_b0' or 'efficientnet' in str(type(model)).lower():
        # EfficientNet: last conv layer in features
        if hasattr(model, 'features'):
            # Find last Conv2d in features
            for module in reversed(list(model.features.modules())):
                if isinstance(module, nn.Conv2d):
                    return module
        raise ValueError("Cannot find last conv layer in EfficientNet")
    
    elif model_type == 'resnet18' or 'resnet' in str(type(model)).lower():
        # ResNet: layer4 (last residual block)
        if hasattr(model, 'layer4'):
            return model.layer4
        raise ValueError("ResNet should have 'layer4' attribute")
    
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}\n"
            "Supported: 'custom', 'efficientnet_b0', 'resnet18'"
        )


def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: nn.Module,
    class_idx: Optional[int] = None,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Generate GradCAM heatmap (convenience function).
    
    Args:
        model: PyTorch model
        image_tensor: Input tensor (1, C, H, W) or (C, H, W)
        target_layer: Layer to hook (e.g., model.block4)
        class_idx: Target class (uses predicted if None)
        device: Device (auto-detected if None)
        
    Returns:
        Heatmap as numpy array (H, W) with values in [0, 1]
        
    Example:
        >>> heatmap = generate_gradcam(model, image_tensor, model.block4, class_idx=5)
    """
    gradcam = GradCAM(model, target_layer, device)
    heatmap = gradcam(image_tensor, class_idx)
    return heatmap


def generate_gradcam_visualization(
    model: nn.Module,
    image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
    target_layer: nn.Module,
    class_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
    transform=None,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
    return_heatmap: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate GradCAM visualization from various input types.
    
    This is the main convenience function for generating GradCAM overlays.
    
    Args:
        model: PyTorch model
        image_input: Image path, PIL Image, numpy array, or tensor
        target_layer: Layer to hook (e.g., model.block4)
        class_idx: Target class (uses predicted if None)
        device: Device (auto-detected if None)
        transform: Transform to apply (if None, uses validation transforms)
        colormap: OpenCV colormap for heatmap
        alpha: Overlay transparency (0=original, 1=heatmap)
        return_heatmap: If True, return (overlay, heatmap) tuple
        
    Returns:
        If return_heatmap=False: RGB overlay image (numpy array)
        If return_heatmap=True: (overlay, heatmap) tuple
        
    Example:
        >>> overlay = generate_gradcam_visualization(
        ...     model, 'test.jpg', model.block4, device='cuda'
        ... )
        >>> Image.fromarray(overlay).save('gradcam_overlay.jpg')
    """
    # Auto-detect device
    if device is None:
        device = next(model.parameters()).device
    
    # Load and preprocess image
    if isinstance(image_input, torch.Tensor):
        # Already a tensor
        image_tensor = image_input
        # For overlay, we need original image
        # Denormalize if needed (assume ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        image_denorm = image_tensor.squeeze(0) * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        original_image = (image_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    elif isinstance(image_input, np.ndarray):
        # Numpy array
        original_image = image_input
        if transform is None:
            from transforms import get_validation_transforms, IMAGENET_MEAN, IMAGENET_STD
            transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
        # Convert to PIL for transform
        pil_image = Image.fromarray(original_image)
        image_tensor = transform(pil_image).unsqueeze(0)
    
    elif isinstance(image_input, Image.Image):
        # PIL Image
        original_image = np.array(image_input)
        if transform is None:
            from transforms import get_validation_transforms, IMAGENET_MEAN, IMAGENET_STD
            transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
        image_tensor = transform(image_input).unsqueeze(0)
    
    else:
        # File path
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        pil_image = Image.open(image_path).convert('RGB')
        original_image = np.array(pil_image)
        
        if transform is None:
            from transforms import get_validation_transforms, IMAGENET_MEAN, IMAGENET_STD
            transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
        
        image_tensor = transform(pil_image).unsqueeze(0)
    
    # Generate GradCAM
    gradcam = GradCAM(model, target_layer, device)
    heatmap = gradcam(image_tensor, class_idx)
    
    # Generate overlay
    overlay = gradcam.generate_heatmap_overlay(
        original_image, heatmap, colormap, alpha
    )
    
    if return_heatmap:
        return overlay, heatmap
    else:
        return overlay


def visualize_gradcam_grid(
    model: nn.Module,
    image_paths: List[Union[str, Path]],
    target_layer: nn.Module,
    class_indices: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Generate a grid of GradCAM visualizations for multiple images.
    
    Args:
        model: PyTorch model
        image_paths: List of image paths
        target_layer: Layer to hook
        class_indices: Target classes for each image (uses predicted if None)
        device: Device
        save_path: Path to save grid image (optional)
        colormap: OpenCV colormap
        alpha: Overlay transparency
        
    Returns:
        Grid image as numpy array
        
    Example:
        >>> image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
        >>> grid = visualize_gradcam_grid(model, image_paths, model.block4)
        >>> Image.fromarray(grid).save('gradcam_grid.jpg')
    """
    import math
    
    # Generate overlays
    overlays = []
    for i, image_path in enumerate(image_paths):
        class_idx = class_indices[i] if class_indices else None
        overlay = generate_gradcam_visualization(
            model, image_path, target_layer, class_idx, device, 
            colormap=colormap, alpha=alpha
        )
        overlays.append(overlay)
    
    # Create grid
    n_images = len(overlays)
    n_cols = int(math.ceil(math.sqrt(n_images)))
    n_rows = int(math.ceil(n_images / n_cols))
    
    # Get image size
    h, w = overlays[0].shape[:2]
    
    # Create grid canvas
    grid = np.zeros((h * n_rows, w * n_cols, 3), dtype=np.uint8)
    
    # Fill grid
    for idx, overlay in enumerate(overlays):
        row = idx // n_cols
        col = idx % n_cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = overlay
    
    # Save if requested
    if save_path:
        Image.fromarray(grid).save(save_path)
        print(f"✅ Saved GradCAM grid: {save_path}")
    
    return grid


def compare_gradcam_predictions(
    model: nn.Module,
    image_path: Union[str, Path],
    target_layer: nn.Module,
    top_k: int = 3,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None
) -> List[Tuple[int, str, float, np.ndarray]]:
    """
    Generate GradCAM for top-k predicted classes.
    
    Useful for comparing which regions activate for different predictions.
    
    Args:
        model: PyTorch model
        image_path: Path to image
        target_layer: Layer to hook
        top_k: Number of top predictions to visualize
        device: Device
        save_path: Path to save comparison image (optional)
        
    Returns:
        List of (class_idx, class_name, confidence, overlay) tuples
        
    Example:
        >>> results = compare_gradcam_predictions(
        ...     model, 'test.jpg', model.block4, top_k=3
        ... )
        >>> for idx, name, conf, overlay in results:
        ...     print(f"{name}: {conf:.2%}")
    """
    # Load image and get predictions
    from predict import load_image, predict_disease
    from transforms import get_validation_transforms, IMAGENET_MEAN, IMAGENET_STD
    
    if device is None:
        device = next(model.parameters()).device
    
    # Load class names (you'll need to pass these or load from file)
    # For now, use placeholder
    class_names = [f"Class_{i}" for i in range(model.classifier[-1].out_features if hasattr(model, 'classifier') else 22)]
    
    # Load and transform image
    pil_image = load_image(image_path)
    transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
    
    # Generate GradCAM for each top prediction
    results = []
    for i in range(top_k):
        class_idx = top_indices[0, i].item()
        confidence = top_probs[0, i].item()
        class_name = class_names[class_idx]
        
        overlay = generate_gradcam_visualization(
            model, image_path, target_layer, class_idx, device
        )
        
        results.append((class_idx, class_name, confidence, overlay))
    
    # Save comparison if requested
    if save_path:
        # Create side-by-side comparison
        h, w = results[0][3].shape[:2]
        comparison = np.zeros((h, w * (top_k + 1), 3), dtype=np.uint8)
        
        # Original image
        original = np.array(Image.open(image_path).resize((w, h)))
        comparison[:, :w] = original
        
        # GradCAM overlays
        for i, (_, _, _, overlay) in enumerate(results):
            comparison[:, (i+1)*w:(i+2)*w] = overlay
        
        Image.fromarray(comparison).save(save_path)
        print(f"✅ Saved GradCAM comparison: {save_path}")
    
    return results


# ============================================================================
# Utility Functions
# ============================================================================

def save_gradcam(
    overlay: np.ndarray,
    save_path: Union[str, Path],
    title: Optional[str] = None
):
    """
    Save GradCAM overlay with optional title.
    
    Args:
        overlay: GradCAM overlay (numpy array)
        save_path: Output path
        title: Optional title text
        
    Raises:
        ImportError: If OpenCV is not installed (required for title)
    """
    if title:
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV is required for adding titles to GradCAM. "
                "Install with: pip install opencv-python"
            )
        import cv2
        # Add title
        overlay_with_title = overlay.copy()
        cv2.putText(
            overlay_with_title, title, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        Image.fromarray(overlay_with_title).save(save_path)
    else:
        Image.fromarray(overlay).save(save_path)
    
    print(f"✅ Saved GradCAM: {save_path}")


def get_colormap_options() -> dict:
    """
    Get available OpenCV colormaps for GradCAM visualization.
    
    Returns:
        Dictionary of colormap names and their OpenCV constants
        
    Raises:
        ImportError: If OpenCV is not installed
    """
    if not CV2_AVAILABLE:
        raise ImportError(
            "OpenCV is required for colormap options. "
            "Install with: pip install opencv-python"
        )
    
    return {
        'jet': cv2.COLORMAP_JET,          # Blue -> Green -> Red (default)
        'hot': cv2.COLORMAP_HOT,          # Black -> Red -> Yellow -> White
        'cool': cv2.COLORMAP_COOL,        # Cyan -> Magenta
        'rainbow': cv2.COLORMAP_RAINBOW,  # Full spectrum
        'viridis': cv2.COLORMAP_VIRIDIS,  # Purple -> Green -> Yellow
        'plasma': cv2.COLORMAP_PLASMA,    # Purple -> Pink -> Yellow
        'inferno': cv2.COLORMAP_INFERNO,  # Black -> Purple -> Yellow
        'bone': cv2.COLORMAP_BONE,        # Black -> White (grayscale-ish)
        'spring': cv2.COLORMAP_SPRING,    # Magenta -> Yellow
        'summer': cv2.COLORMAP_SUMMER,    # Green -> Yellow
        'autumn': cv2.COLORMAP_AUTUMN,    # Red -> Yellow
        'winter': cv2.COLORMAP_WINTER,    # Blue -> Green
    }
