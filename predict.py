"""
CropShield AI - Production Inference Script
============================================

GPU-accelerated inference with model caching for efficient deployment.

Features:
- Model caching (load once, reuse)
- Mixed precision inference (torch.cuda.amp.autocast)
- GPU/CPU automatic device selection
- Top-k predictions with confidence scores
- Compatible with Custom CNN and transfer learning models
- Loads from class_to_idx.json mapping

Usage:
    # Command line
    python predict.py --image path/to/image.jpg --model models/cropshield_cnn.pth --top_k 3
    
    # Python API
    from predict import load_model_once, predict_disease
    
    model, class_names, device = load_model_once('models/cropshield_cnn.pth')
    predictions = predict_disease('test.jpg', model, class_names, device, top_k=3)
    
    for class_name, confidence in predictions:
        print(f"{class_name}: {confidence:.2%}")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from PIL import Image
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional
import time
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import transforms
from transforms import get_validation_transforms, IMAGENET_MEAN, IMAGENET_STD

# Import custom CNN
from model_custom_cnn import CropShieldCNN

# Try importing transfer learning models
try:
    from torchvision import models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn("torchvision not available. Transfer learning models disabled.")


# ============================================================================
# Image Preprocessing (Same as Training)
# ============================================================================

def get_device() -> torch.device:
    """
    Get the best available device for inference.
    
    Returns:
        torch.device: CUDA if available, else CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Inference: {gpu_name}")
    else:
        device = torch.device('cpu')
        print("⚠️  CPU Inference (GPU not available)")
    
    return device


def load_class_mapping(model_dir: Path) -> List[str]:
    """
    Load class_to_idx mapping and create class names list.
    
    Args:
        model_dir: Directory containing class_to_idx.json
        
    Returns:
        List of class names indexed by class index
        
    Raises:
        FileNotFoundError: If class_to_idx.json not found
    """
    class_mapping_path = model_dir / 'class_to_idx.json'
    
    if not class_mapping_path.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {class_mapping_path}\n"
            "Expected format: {'Potato__early_blight': 0, 'Potato__healthy': 1, ...}"
        )
    
    with open(class_mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    
    # Create reverse mapping: idx -> class_name
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    
    # Validate: ensure indices are contiguous from 0 to num_classes-1
    num_classes = len(idx_to_class)
    expected_indices = set(range(num_classes))
    actual_indices = set(idx_to_class.keys())
    
    if expected_indices != actual_indices:
        raise ValueError(
            f"Invalid class mapping: Expected indices {expected_indices}, got {actual_indices}"
        )
    
    # Convert to list for fast indexing
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    print(f"✅ Loaded {num_classes} classes from {class_mapping_path.name}")
    
    return class_names


# ============================================================================
# Model Loading and Caching
# ============================================================================

# Global cache for loaded model (avoid reloading)
_MODEL_CACHE = {}


def load_model(
    model_path: str,
    device: torch.device,
    verbose: bool = True
) -> Tuple[nn.Module, List[str]]:
    """
    Load trained model and class names from checkpoint and class_to_idx.json.
    
    Args:
        model_path: Path to saved model (.pth file)
        device: Target device (cuda/cpu)
        verbose: Print loading details
        
    Returns:
        (model, class_names) tuple
        
    Raises:
        FileNotFoundError: If model file or class_to_idx.json not found
        RuntimeError: If model architecture detection fails
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"LOADING MODEL: {model_path.name}")
        print(f"{'='*60}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model metadata
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_type = checkpoint.get('model_type', 'unknown')
        num_classes = checkpoint.get('num_classes', None)
    else:
        state_dict = checkpoint
        model_type = 'unknown'
        num_classes = None
    
    # Load class names from class_to_idx.json
    class_names = load_class_mapping(model_path.parent)
    
    # Auto-detect num_classes if not in checkpoint
    if num_classes is None:
        num_classes = len(class_names)
        if verbose:
            print(f"⚠️  num_classes not in checkpoint. Auto-detected: {num_classes}")
    
    # Initialize model based on type
    if model_type == 'custom' or 'block1' in state_dict:
        # Custom CNN
        model = CropShieldCNN(num_classes=num_classes)
        model_name = 'CropShieldCNN'
        
    elif model_type == 'efficientnet_b0' or 'efficientnet' in model_path.name.lower():
        # EfficientNet-B0
        if not TORCHVISION_AVAILABLE:
            raise RuntimeError("torchvision required for EfficientNet. Install with: pip install torchvision")
        
        model = models.efficientnet_b0(weights=None)
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        model_name = 'EfficientNet-B0'
        
    elif model_type == 'resnet18' or 'resnet18' in model_path.name.lower():
        # ResNet-18
        if not TORCHVISION_AVAILABLE:
            raise RuntimeError("torchvision required for ResNet. Install with: pip install torchvision")
        
        model = models.resnet18(weights=None)
        # Replace classifier
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model_name = 'ResNet-18'
        
    else:
        raise RuntimeError(
            f"Unknown model type: {model_type}\n"
            f"Supported: 'custom', 'efficientnet_b0', 'resnet18'\n"
            f"Checkpoint keys: {list(state_dict.keys())[:5]}..."
        )
    
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model: {model_name}")
        print(f"   Classes: {num_classes}")
        print(f"   Parameters: {total_params:,}")
        print(f"   Device: {device}")
        print(f"{'='*60}\n")
    
    return model, class_names


def load_model_once(
    model_path: str,
    device: Optional[torch.device] = None,
    force_reload: bool = False
) -> Tuple[nn.Module, List[str], torch.device]:
    """
    Load model with caching to avoid repeated loading.
    
    This function maintains a global cache of loaded models. Subsequent calls
    with the same model_path will return the cached model instantly.
    
    Args:
        model_path: Path to saved model (.pth file)
        device: Target device (auto-detected if None)
        force_reload: Force reload even if cached
        
    Returns:
        (model, class_names, device) tuple
        
    Example:
        >>> # First call: loads model (slow)
        >>> model, class_names, device = load_model_once('models/cropshield_cnn.pth')
        >>> 
        >>> # Subsequent calls: returns cached model (instant)
        >>> model, class_names, device = load_model_once('models/cropshield_cnn.pth')
    """
    global _MODEL_CACHE
    
    model_path = str(Path(model_path).resolve())
    
    # Check cache
    if model_path in _MODEL_CACHE and not force_reload:
        print(f"✅ Using cached model: {Path(model_path).name}")
        return _MODEL_CACHE[model_path]
    
    # Auto-detect device
    if device is None:
        device = get_device()
    
    # Load model
    model, class_names = load_model(model_path, device, verbose=True)
    
    # Cache for future use
    _MODEL_CACHE[model_path] = (model, class_names, device)
    
    return model, class_names, device


def clear_model_cache() -> None:
    """
    Clear the model cache to free memory.
    
    This function clears all cached models from memory. Useful when you need
    to free up GPU/CPU memory or when switching between different models.
    """
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.info("Model cache cleared")


# ============================================================================
# Image Loading and Preprocessing
# ============================================================================

def load_image(
    image_input: Union[str, Path, Image.Image, np.ndarray]
) -> Image.Image:
    """
    Load image from various input types.
    
    Args:
        image_input: Image path (str/Path), PIL Image, or numpy array
        
    Returns:
        PIL Image in RGB mode
        
    Example:
        >>> img = load_image('test.jpg')
        >>> img = load_image(pil_image)
        >>> img = load_image(numpy_array)
    """
    # PIL Image
    if isinstance(image_input, Image.Image):
        image = image_input
    
    # Numpy array
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    
    # File path
    elif isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path)
    
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")
    
    # Convert to RGB (handle RGBA, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


# ============================================================================
# Fast Inference Function
# ============================================================================

@torch.no_grad()
def predict_disease(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    model: nn.Module,
    class_names: List[str],
    top_k: int = 3,
    device: Optional[torch.device] = None,
    return_raw_logits: bool = False,
    use_amp: bool = True
) -> Union[List[Tuple[str, float]], Tuple[List[Tuple[str, float]], torch.Tensor]]:
    """
    Predict disease class from image with top-k results.
    
    Args:
        image_input: Image path, PIL Image, or numpy array
        model: Trained PyTorch model (in eval mode)
        class_names: List of class names
        top_k: Number of top predictions to return (default: 3)
        device: Device to run inference on (auto-detect if None)
        return_raw_logits: Return raw logits for GradCAM (default: False)
        use_amp: Use mixed precision inference for speed (default: True)
        
    Returns:
        If return_raw_logits=False:
            List of (class_name, probability) tuples, ordered by probability
        If return_raw_logits=True:
            Tuple of (predictions, logits) where logits is tensor for GradCAM
            
    Example:
        >>> predictions = predict_disease('test.jpg', model, class_names, top_k=3)
        >>> for class_name, prob in predictions:
        ...     print(f"{class_name}: {prob:.2%}")
        
        >>> # For GradCAM
        >>> predictions, logits = predict_disease(
        ...     'test.jpg', model, class_names, 
        ...     return_raw_logits=True
        ... )
    """
    start_time = time.time()
    
    # Auto-detect device
    if device is None:
        device = next(model.parameters()).device
    
    # Disable AMP if not on CUDA
    use_amp = use_amp and device.type == 'cuda'
    
    # 1. Load and preprocess image
    image = load_image(image_input)
    transform = get_validation_transforms(
        image_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension [1, 3, 224, 224]
    input_tensor = input_tensor.to(device, non_blocking=True)
    
    # 2. Run inference
    model.eval()
    
    if use_amp:
        with autocast():
            logits = model(input_tensor)
    else:
        logits = model(input_tensor)
    
    # 3. Compute probabilities
    probabilities = F.softmax(logits, dim=1)
    
    # 4. Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(class_names)), dim=1)
    
    # Convert to CPU and numpy
    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()
    
    # Handle single prediction (top_k=1)
    if top_k == 1:
        top_probs = [top_probs]
        top_indices = [top_indices]
    
    # 5. Build results
    predictions = [
        (class_names[idx], float(prob))
        for idx, prob in zip(top_indices, top_probs)
    ]
    
    # Inference time
    elapsed = (time.time() - start_time) * 1000  # Convert to ms
    
    # Optional: print inference time (can be removed for production)
    if elapsed < 100:
        # Fast inference (< 100ms target)
        pass  # Silent for speed
    else:
        print(f"⚠️  Inference time: {elapsed:.1f}ms (target: <100ms)")
    
    # Return results
    if return_raw_logits:
        return predictions, logits
    else:
        return predictions


# ============================================================================
# Batch Inference (for processing multiple images)
# ============================================================================

@torch.no_grad()
def predict_batch(
    image_list: List[Union[str, Path, Image.Image, np.ndarray]],
    model: nn.Module,
    class_names: List[str],
    top_k: int = 3,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    use_amp: bool = True
) -> List[List[Tuple[str, float]]]:
    """
    Predict disease classes for multiple images in batches.
    More efficient than calling predict_disease() repeatedly.
    
    Args:
        image_list: List of images (paths, PIL Images, or numpy arrays)
        model: Trained PyTorch model
        class_names: List of class names
        top_k: Number of top predictions per image
        batch_size: Batch size for inference
        device: Device to run inference on
        use_amp: Use mixed precision inference
        
    Returns:
        List of predictions for each image
        
    Example:
        >>> images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        >>> results = predict_batch(images, model, class_names, top_k=3)
        >>> for i, preds in enumerate(results):
        ...     print(f"Image {i}:")
        ...     for class_name, prob in preds:
        ...         print(f"  {class_name}: {prob:.2%}")
    """
    # Auto-detect device
    if device is None:
        device = next(model.parameters()).device
    
    use_amp = use_amp and device.type == 'cuda'
    
    # Preprocess all images
    transform = get_validation_transforms(
        image_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
    tensors = []
    
    for img_input in image_list:
        image = load_image(img_input)
        tensor = transform(image)
        tensors.append(tensor)
    
    # Stack into batch
    batch_tensor = torch.stack(tensors).to(device, non_blocking=True)
    
    # Run inference in batches
    all_predictions = []
    
    for i in range(0, len(batch_tensor), batch_size):
        batch = batch_tensor[i:i+batch_size]
        
        if use_amp:
            with autocast():
                logits = model(batch)
        else:
            logits = model(batch)
        
        # Compute probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Get top-k for each image in batch
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(class_names)), dim=1)
        
        # Convert to CPU
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        # Build predictions for each image
        for probs, indices in zip(top_probs, top_indices):
            predictions = [
                (class_names[idx], float(prob))
                for idx, prob in zip(indices, probs)
            ]
            all_predictions.append(predictions)
    
    return all_predictions


# ============================================================================
# Streamlit Integration Example
# ============================================================================

def streamlit_example():
    """
    Example code for Streamlit integration.
    Copy this into your Streamlit app (app.py or streamlit_app.py).
    """
    example_code = '''
import streamlit as st
from PIL import Image
from predict import load_model_once, predict_disease

# Page configuration
st.set_page_config(
    page_title="CropShield AI",
    page_icon="🌾",
    layout="wide"
)

# Title
st.title("🌾 CropShield AI - Plant Disease Detection")
st.write("Upload a plant image to detect diseases")

# Load model once (cached across reruns)
@st.cache_resource
def load_model():
    """Load model (cached by Streamlit)"""
    model, class_names, device = load_model_once(
        checkpoint_path='checkpoints/best.pth',
        model_type='custom'
    )
    return model, class_names, device

# Load model
with st.spinner("Loading model..."):
    model, class_names, device = load_model()
    st.success("✅ Model loaded!")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image of the plant leaf"
)

if uploaded_file is not None:
    # Display image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🔍 Predictions")
        
        # Run prediction
        with st.spinner("Analyzing..."):
            predictions = predict_disease(
                image_input=image,
                model=model,
                class_names=class_names,
                device=device,
                top_k=3
            )
        
        # Display results
        for i, (class_name, prob) in enumerate(predictions, 1):
            # Format class name (replace underscores)
            display_name = class_name.replace('_', ' ').title()
            
            # Color based on confidence
            if prob > 0.7:
                color = "🟢"  # High confidence
            elif prob > 0.4:
                color = "🟡"  # Medium confidence
            else:
                color = "🔴"  # Low confidence
            
            # Display prediction
            st.write(f"{color} **{i}. {display_name}**")
            st.progress(prob)
            st.write(f"Confidence: {prob:.1%}")
            st.write("---")
        
        # Show top prediction prominently
        top_class, top_prob = predictions[0]
        
        if top_prob > 0.8:
            st.success(f"✅ **Detected: {top_class.replace('_', ' ').title()}** ({top_prob:.1%})")
        elif top_prob > 0.5:
            st.warning(f"⚠️ **Likely: {top_class.replace('_', ' ').title()}** ({top_prob:.1%})")
        else:
            st.info(f"ℹ️ **Uncertain** - Please provide a clearer image")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    CropShield AI detects diseases in:
    - 🥔 Potato
    - 🌿 Sugarcane  
    - 🍅 Tomato
    - 🌾 Wheat
    
    **How to use:**
    1. Upload a clear leaf image
    2. Wait for analysis (~50ms)
    3. View top 3 predictions
    
    **Model:** Custom CNN (4.7M params)
    **Accuracy:** ~85% on test set
    """)
    
    st.header("📊 Model Stats")
    st.metric("Classes", len(class_names))
    st.metric("Device", str(device).upper())
    st.metric("Inference Speed", "~50ms")

# Run command:
# streamlit run app.py
    '''
    
    return example_code


# ============================================================================
# Utility Functions
# ============================================================================

def format_predictions(predictions: List[Tuple[str, float]], show_all: bool = False) -> str:
    """
    Format predictions for display.
    
    Args:
        predictions: List of (class_name, confidence) tuples
        show_all: Show all predictions (else only top prediction)
        
    Returns:
        Formatted string
    """
    if not predictions:
        return "No predictions"
    
    lines = []
    
    if show_all:
        lines.append(f"\n{'='*60}")
        lines.append(f"TOP {len(predictions)} PREDICTIONS")
        lines.append(f"{'='*60}")
        for i, (class_name, confidence) in enumerate(predictions, 1):
            # Format class name: Replace underscores, capitalize
            display_name = class_name.replace('__', ' - ').replace('_', ' ').title()
            # Confidence indicator
            if confidence > 0.7:
                icon = "🟢"
            elif confidence > 0.4:
                icon = "🟡"
            else:
                icon = "🔴"
            lines.append(f"{i}. {icon} {display_name:40s} {confidence:6.2%}")
        lines.append(f"{'='*60}")
    else:
        class_name, confidence = predictions[0]
        display_name = class_name.replace('__', ' - ').replace('_', ' ').title()
        lines.append(f"\n🌿 Predicted Disease: {display_name}")
        lines.append(f"   Confidence: {confidence:.2%}")
    
    return '\n'.join(lines)


# ============================================================================
# CLI Testing Interface
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test CropShield AI inference on single image'
    )
    
    parser.add_argument(
        'image',
        type=str,
        help='Path to image file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best.pth',
        help='Path to model checkpoint (default: checkpoints/best.pth)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='custom',
        choices=['custom', 'efficientnet', 'efficientnet_b0'],
        help='Model architecture (default: custom)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Number of top predictions to show (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Test inference
    print(f"\n{'='*60}")
    print("🌾 CROPSHIELD AI - INFERENCE TEST")
    print(f"{'='*60}")
    print(f"Image:       {args.image}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Model:       {args.model}")
    print(f"Top-K:       {args.top_k}")
    print(f"{'='*60}\n")
    
    # Load model
    model, class_names, device = load_model_once(
        model_path=args.checkpoint
    )
    
    # Run prediction
    print(f"\n🔍 Running inference...")
    start = time.time()
    
    predictions = predict_disease(
        image_input=args.image,
        model=model,
        class_names=class_names,
        device=device,
        top_k=args.top_k
    )
    
    elapsed = (time.time() - start) * 1000
    
    # Display results
    print(f"\n{'='*60}")
    print(f"📊 PREDICTIONS (inference: {elapsed:.1f}ms)")
    print(f"{'='*60}")
    
    for i, (class_name, prob) in enumerate(predictions, 1):
        # Format class name
        display_name = class_name.replace('_', ' ').title()
        
        # Confidence indicator
        if prob > 0.7:
            confidence = "HIGH"
            icon = "🟢"
        elif prob > 0.4:
            confidence = "MEDIUM"
            icon = "🟡"
        else:
            confidence = "LOW"
            icon = "🔴"
        
        print(f"{i}. {icon} {display_name:45s} {prob:6.2%}  [{confidence}]")
    
    print(f"{'='*60}\n")
    
    # Performance check
    if elapsed < 50:
        print(f"✅ Excellent latency: {elapsed:.1f}ms (target: <100ms)")
    elif elapsed < 100:
        print(f"✅ Good latency: {elapsed:.1f}ms (target: <100ms)")
    else:
        print(f"⚠️  High latency: {elapsed:.1f}ms (target: <100ms)")
        print(f"   Consider using GPU or reducing image size")
    
    print()
