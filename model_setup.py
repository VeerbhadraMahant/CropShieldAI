"""
Model Setup and Initialization for CropShield AI
================================================

Provides utilities for:
- Model initialization and device placement
- Loss function and optimizer setup
- Model summary and parameter analysis
- Recommended hyperparameters for training

Supports both custom CNN and transfer learning models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional

try:
    from torchsummary import summary
    TORCHSUMMARY_AVAILABLE = True
except ImportError:
    TORCHSUMMARY_AVAILABLE = False
    print("⚠️  torchsummary not installed. Install with: pip install torchsummary")


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device: CUDA device if available, else CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (GPU not available)")
    
    return device


def count_parameters(model: nn.Module, verbose: bool = True) -> Dict[str, int]:
    """
    Count model parameters (total, trainable, non-trainable).
    
    Args:
        model: PyTorch model
        verbose: If True, print parameter counts
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    param_dict = {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }
    
    if verbose:
        print("\n" + "="*60)
        print("MODEL PARAMETERS")
        print("="*60)
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable:        {non_trainable_params:,}")
        print("="*60)
    
    return param_dict


def get_model_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Get comprehensive model summary including architecture and memory usage.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        device: Device to use (auto-detected if None)
        verbose: If True, print detailed summary
        
    Returns:
        Dictionary with model statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Get parameter counts
    param_info = count_parameters(model, verbose=False)
    
    # Calculate memory usage
    param_memory_mb = param_info['total'] * 4 / (1024**2)  # 4 bytes per float32
    
    # Estimate activation memory (forward pass)
    # This is approximate - actual memory depends on batch size
    with torch.no_grad():
        model.eval()
        dummy_input = torch.randn(1, *input_size).to(device)
        
        # Hook to capture intermediate activations
        activation_memory = 0
        def hook_fn(module, input, output):
            nonlocal activation_memory
            if isinstance(output, torch.Tensor):
                activation_memory += output.numel() * 4 / (1024**2)
        
        hooks = []
        for layer in model.modules():
            hooks.append(layer.register_forward_hook(hook_fn))
        
        _ = model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    summary_dict = {
        'total_params': param_info['total'],
        'trainable_params': param_info['trainable'],
        'non_trainable_params': param_info['non_trainable'],
        'param_memory_mb': param_memory_mb,
        'activation_memory_mb': activation_memory,
        'total_memory_mb': param_memory_mb + activation_memory,
        'input_size': input_size,
        'device': str(device)
    }
    
    if verbose:
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        print(f"Input size: {input_size}")
        print(f"Device: {device}")
        print(f"\nParameters:")
        print(f"  Total:     {param_info['total']:,}")
        print(f"  Trainable: {param_info['trainable']:,}")
        print(f"  Frozen:    {param_info['non_trainable']:,}")
        print(f"\nMemory Usage:")
        print(f"  Parameters:  {param_memory_mb:.2f} MB")
        print(f"  Activations: {activation_memory:.2f} MB (batch=1)")
        print(f"  Total:       {summary_dict['total_memory_mb']:.2f} MB")
        print(f"\n  Estimated for batch=32: {(param_memory_mb + activation_memory * 32):.2f} MB")
        print("="*60)
        
        # Try to use torchsummary for detailed layer-by-layer info
        if TORCHSUMMARY_AVAILABLE:
            print("\n" + "="*60)
            print("DETAILED LAYER-BY-LAYER SUMMARY")
            print("="*60)
            try:
                summary(model, input_size, device=str(device))
            except Exception as e:
                print(f"⚠️  Could not generate detailed summary: {e}")
        else:
            print("\n💡 Install torchsummary for detailed layer-by-layer breakdown:")
            print("   pip install torchsummary")
    
    return summary_dict


def initialize_model(
    model: nn.Module,
    device: Optional[torch.device] = None,
    print_summary: bool = True
) -> Tuple[nn.Module, torch.device]:
    """
    Initialize model and move to device.
    
    Args:
        model: PyTorch model
        device: Target device (auto-detected if None)
        print_summary: If True, print model summary
        
    Returns:
        Tuple of (model on device, device used)
    """
    if device is None:
        device = get_device()
    
    # Move model to device
    model = model.to(device)
    
    if print_summary:
        # Print trainable parameters
        param_info = count_parameters(model, verbose=True)
        
        print(f"\n✅ Model moved to {device}")
        
        # Calculate memory requirements
        param_memory_mb = param_info['total'] * 4 / (1024**2)
        print(f"\n📊 Model size: {param_memory_mb:.2f} MB")
        
        if device.type == 'cuda':
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"💾 Available GPU memory: {available_memory_gb:.2f} GB")
    
    return model, device


def create_training_setup(
    model: nn.Module,
    num_classes: int = 22,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    step_size: int = 5,
    gamma: float = 0.5,
    optimizer_type: str = 'adam',
    verbose: bool = True
) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Create complete training setup with loss, optimizer, and scheduler.
    
    Recommended Hyperparameters (Default):
    - Learning Rate: 1e-4 (conservative for fine-tuning)
    - Optimizer: Adam (adaptive learning rate)
    - Weight Decay: 1e-4 (L2 regularization)
    - Scheduler: StepLR (step_size=5, gamma=0.5)
        * Reduces LR by 50% every 5 epochs
        * Helps convergence in later training stages
    
    Args:
        model: PyTorch model
        num_classes: Number of output classes
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        step_size: Number of epochs between LR reductions
        gamma: LR reduction factor
        optimizer_type: 'adam' or 'sgd'
        verbose: If True, print setup details
        
    Returns:
        Tuple of (model, criterion, optimizer, scheduler)
    """
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING SETUP")
        print("="*60)
        print(f"Loss Function: CrossEntropyLoss")
        print(f"Optimizer:     {optimizer_type.upper()}")
        print(f"  Learning Rate:  {learning_rate}")
        print(f"  Weight Decay:   {weight_decay}")
        if optimizer_type.lower() == 'sgd':
            print(f"  Momentum:       0.9")
        print(f"Scheduler:     StepLR")
        print(f"  Step Size:      {step_size} epochs")
        print(f"  Gamma:          {gamma}")
        print(f"  (LR will be multiplied by {gamma} every {step_size} epochs)")
        print("="*60)
    
    return model, criterion, optimizer, scheduler


def setup_model_for_training(
    model: nn.Module,
    device: Optional[torch.device] = None,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    step_size: int = 5,
    gamma: float = 0.5,
    optimizer_type: str = 'adam',
    print_model_summary: bool = True
) -> Dict[str, Any]:
    """
    Complete one-stop setup for model training.
    
    Performs:
    1. Device detection and model placement
    2. Parameter counting
    3. Model summary (optional)
    4. Loss function and optimizer initialization
    5. Learning rate scheduler setup
    
    Args:
        model: PyTorch model (custom CNN or transfer learning)
        device: Target device (auto-detected if None)
        learning_rate: Initial learning rate (default: 1e-4)
        weight_decay: L2 regularization (default: 1e-4)
        step_size: Epochs between LR reduction (default: 5)
        gamma: LR reduction factor (default: 0.5)
        optimizer_type: 'adam' or 'sgd' (default: 'adam')
        print_model_summary: If True, print detailed summary
        
    Returns:
        Dictionary containing:
        - 'model': Model on device
        - 'device': Device used
        - 'criterion': Loss function
        - 'optimizer': Optimizer
        - 'scheduler': LR scheduler
        - 'summary': Model summary dict
    """
    print("\n" + "="*60)
    print("🚀 CROPSHIELD AI - MODEL SETUP")
    print("="*60)
    
    # 1. Initialize model and move to device
    model, device = initialize_model(model, device, print_summary=False)
    
    # 2. Get model summary
    if print_model_summary:
        summary_dict = get_model_summary(model, input_size=(3, 224, 224), device=device)
    else:
        param_info = count_parameters(model, verbose=True)
        summary_dict = {
            'total_params': param_info['total'],
            'trainable_params': param_info['trainable'],
            'non_trainable_params': param_info['non_trainable']
        }
    
    print(f"\n✅ Model moved to {device}")
    
    # 3. Create training setup
    model, criterion, optimizer, scheduler = create_training_setup(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        step_size=step_size,
        gamma=gamma,
        optimizer_type=optimizer_type,
        verbose=True
    )
    
    print("\n✅ Training setup complete!")
    print("="*60)
    
    return {
        'model': model,
        'device': device,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'summary': summary_dict
    }


# ============================================================
# USAGE EXAMPLES
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODEL SETUP - USAGE EXAMPLES")
    print("="*60)
    
    # Example 1: Custom CNN
    print("\n" + "-"*60)
    print("EXAMPLE 1: Custom CNN Setup")
    print("-"*60)
    
    from model_custom_cnn import CropShieldCNN
    
    # Create model
    model = CropShieldCNN(num_classes=22)
    
    # Complete setup (one-stop function)
    setup = setup_model_for_training(
        model=model,
        learning_rate=1e-4,
        weight_decay=1e-4,
        step_size=5,
        gamma=0.5,
        optimizer_type='adam',
        print_model_summary=True
    )
    
    # Extract components
    model = setup['model']
    device = setup['device']
    criterion = setup['criterion']
    optimizer = setup['optimizer']
    scheduler = setup['scheduler']
    
    print("\n✅ Custom CNN ready for training!")
    
    # Test forward pass
    print("\n" + "-"*60)
    print("FORWARD PASS TEST")
    print("-"*60)
    
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Example 2: Transfer Learning (EfficientNet-B0)
    print("\n" + "-"*60)
    print("EXAMPLE 2: Transfer Learning Setup (EfficientNet-B0)")
    print("-"*60)
    
    import torchvision.models as models
    
    # Create pretrained model
    model_transfer = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    # Modify classifier for 22 classes
    num_features = model_transfer.classifier[1].in_features
    model_transfer.classifier[1] = nn.Linear(num_features, 22)
    
    # Setup with different hyperparameters for transfer learning
    setup_transfer = setup_model_for_training(
        model=model_transfer,
        learning_rate=1e-4,  # Lower LR for fine-tuning
        weight_decay=1e-4,
        step_size=5,
        gamma=0.5,
        optimizer_type='adam',
        print_model_summary=False  # Skip detailed summary for brevity
    )
    
    print("\n✅ Transfer learning model ready!")
    
    # Example 3: Manual step-by-step setup
    print("\n" + "-"*60)
    print("EXAMPLE 3: Manual Step-by-Step Setup")
    print("-"*60)
    
    # Create new model
    model_manual = CropShieldCNN(num_classes=22)
    
    # Step 1: Get device
    device = get_device()
    
    # Step 2: Move model to device
    model_manual = model_manual.to(device)
    print(f"✅ Model moved to {device}")
    
    # Step 3: Count parameters
    param_info = count_parameters(model_manual)
    
    # Step 4: Get detailed summary
    print("\n📊 Getting detailed model summary...")
    summary_info = get_model_summary(model_manual, input_size=(3, 224, 224))
    
    # Step 5: Create loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_manual.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    print("\n✅ Manual setup complete!")
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    print("\n📝 Recommended Hyperparameters:")
    print("  • Learning Rate: 1e-4")
    print("  • Optimizer: Adam")
    print("  • Weight Decay: 1e-4")
    print("  • Scheduler: StepLR (step_size=5, gamma=0.5)")
    print("\n💡 Usage:")
    print("  from model_setup import setup_model_for_training")
    print("  setup = setup_model_for_training(model)")
    print("  model = setup['model']")
    print("  device = setup['device']")
    print("  criterion = setup['criterion']")
    print("  optimizer = setup['optimizer']")
    print("  scheduler = setup['scheduler']")
    print("\n🚀 Ready to train!")
    print("="*60)
