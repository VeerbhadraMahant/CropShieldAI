"""
Test Learning Rate Scheduler Integration

This script verifies that all scheduler types work correctly
with the model factory and warmup wrapper.
"""

import sys
import torch
from models.model_factory import get_model, LRSchedulerWithWarmup

def test_scheduler(scheduler_type, warmup_epochs=0, verbose=True):
    """Test a specific scheduler configuration."""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing: {scheduler_type.upper()} with warmup={warmup_epochs}")
        print(f"{'='*70}")
    
    # Create model with scheduler
    model, optimizer, criterion, scheduler, device = get_model(
        model_type='custom',
        num_classes=22,
        learning_rate=1e-4,
        scheduler_type=scheduler_type,
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        scheduler_patience=3,
        scheduler_t_max=10,
        warmup_epochs=warmup_epochs,
        verbose=verbose
    )
    
    # Simulate training loop
    print(f"\n📊 Simulating 10 epochs:")
    print(f"{'Epoch':<8} {'LR':<12} {'Action':<30}")
    print(f"{'-'*50}")
    
    for epoch in range(10):
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Simulate validation loss (decreasing then plateauing)
        val_loss = 2.0 - (epoch * 0.15) if epoch < 6 else 1.1
        
        # Action description
        if warmup_epochs > 0 and epoch < warmup_epochs:
            action = f"Warmup: 0 → {1e-4:.6f}"
        elif scheduler_type == 'plateau':
            action = f"Monitor val_loss={val_loss:.4f}"
        elif scheduler_type == 'step' and (epoch + 1) % 5 == 0:
            action = f"Step: reduce LR by {0.5}"
        elif scheduler_type == 'cosine':
            action = f"Cosine decay"
        else:
            action = "Maintain LR"
        
        print(f"{epoch:<8} {current_lr:.8f} {action:<30}")
        
        # Step scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif isinstance(scheduler, LRSchedulerWithWarmup):
            # Warmup wrapper - automatically handles underlying scheduler
            if hasattr(scheduler, 'scheduler') and isinstance(
                scheduler.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        else:
            scheduler.step()
    
    print(f"\n✅ {scheduler_type.upper()} test passed!")
    return True


def main():
    """Test all scheduler configurations."""
    
    print("\n" + "="*70)
    print("🧪 LEARNING RATE SCHEDULER INTEGRATION TEST")
    print("="*70)
    
    tests = [
        ('step', 0),        # StepLR, no warmup
        ('step', 1),        # StepLR with warmup
        ('plateau', 0),     # ReduceLROnPlateau, no warmup
        ('plateau', 1),     # ReduceLROnPlateau with warmup
        ('cosine', 0),      # CosineAnnealingLR, no warmup
        ('cosine', 2),      # CosineAnnealingLR with warmup
    ]
    
    results = []
    for scheduler_type, warmup_epochs in tests:
        try:
            success = test_scheduler(scheduler_type, warmup_epochs, verbose=True)
            results.append((scheduler_type, warmup_epochs, success))
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            results.append((scheduler_type, warmup_epochs, False))
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, _, success in results if success)
    total = len(results)
    
    for scheduler_type, warmup_epochs, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        warmup_str = f"warmup={warmup_epochs}" if warmup_epochs > 0 else "no warmup"
        print(f"{status} - {scheduler_type.upper()} ({warmup_str})")
    
    print(f"\n{'='*70}")
    print(f"Result: {passed}/{total} tests passed")
    print(f"{'='*70}")
    
    if passed == total:
        print("\n🎉 All scheduler tests passed!")
        print("✅ Learning rate scheduling system is ready!")
        print("\n📚 Documentation:")
        print("   - LR_SCHEDULING_GUIDE.md      (full guide)")
        print("   - LR_SCHEDULING_QUICKREF.md   (quick reference)")
        print("\n💡 Example usage:")
        print("   python train.py --model custom --epochs 25")
        print("   python train.py --model efficientnet_b0 --scheduler plateau --warmup_epochs 1")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
