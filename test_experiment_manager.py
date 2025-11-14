"""
Test experiment management system.

Quick verification that grid search and experiment tracking work correctly.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from experiment_manager import grid_search, ExperimentManager


def test_minimal_sweep():
    """
    Test minimal sweep with very small parameter grid.
    
    This runs 4 experiments (2 LR × 2 WD) for 1 epoch each.
    Should complete in ~2-3 minutes.
    """
    
    print("\n" + "="*80)
    print("🧪 TESTING EXPERIMENT MANAGEMENT SYSTEM")
    print("="*80)
    
    # Minimal parameter grid (4 combinations)
    param_grid = {
        'learning_rate': [1e-4, 3e-4],
        'weight_decay': [0, 1e-5]
    }
    
    print(f"\n📊 Test Configuration:")
    print(f"   Model: custom")
    print(f"   Epochs per run: 1 (quick test)")
    print(f"   Batch size: 32")
    print(f"   Parameter grid: {param_grid}")
    print(f"   Total combinations: 4 (2 LR × 2 WD)")
    print(f"   Expected time: ~2-3 minutes\n")
    
    # Run sweep
    start_time = time.time()
    
    summary = grid_search(
        param_grid=param_grid,
        model_type='custom',
        num_epochs=1,  # Just 1 epoch for testing
        batch_size=32,
        sweep_name='test_sweep',
        save_best_only=True
    )
    
    elapsed = time.time() - start_time
    
    # Verify results
    print(f"\n" + "="*80)
    print("✅ VERIFICATION")
    print("="*80)
    
    success = True
    
    # Check 1: All experiments completed
    completed = [r for r in summary['results'] if r.get('status') == 'completed']
    if len(completed) == 4:
        print(f"✅ All 4 experiments completed")
    else:
        print(f"❌ Only {len(completed)}/4 experiments completed")
        success = False
    
    # Check 2: Best experiment found
    if summary.get('best_exp_id'):
        print(f"✅ Best experiment: {summary['best_exp_id']}")
        print(f"   Val Acc: {summary['best_val_acc']:.2f}%")
    else:
        print(f"❌ No best experiment found")
        success = False
    
    # Check 3: Files saved
    exp_manager = ExperimentManager()
    sweep_id = summary['sweep_id']
    
    # Check config files
    config_count = len(list(exp_manager.runs_dir.glob(f"{sweep_id}*/config.json")))
    if config_count == 4:
        print(f"✅ All 4 config files saved")
    else:
        print(f"❌ Only {config_count}/4 config files found")
        success = False
    
    # Check summary files
    summary_count = len(list(exp_manager.runs_dir.glob(f"{sweep_id}*/summary.json")))
    if summary_count == 4:
        print(f"✅ All 4 summary files saved")
    else:
        print(f"❌ Only {summary_count}/4 summary files found")
        success = False
    
    # Check best model
    best_model = exp_manager.best_dir / f"{sweep_id}_best.pth"
    if best_model.exists():
        print(f"✅ Best model saved")
    else:
        print(f"❌ Best model not found")
        success = False
    
    # Check 4: Time reasonable
    expected_time = 180  # 3 minutes
    if elapsed < expected_time * 1.5:
        print(f"✅ Time: {elapsed:.1f}s (expected ~{expected_time}s)")
    else:
        print(f"⚠️  Time: {elapsed:.1f}s (slower than expected {expected_time}s)")
    
    # Check 5: Results reasonable
    if summary['best_val_acc'] > 10.0:  # Better than random (22 classes → 4.5%)
        print(f"✅ Validation accuracy reasonable (>{10}%)")
    else:
        print(f"⚠️  Validation accuracy low (<{10}%)")
    
    print(f"\n" + "="*80)
    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - Check errors above")
    print("="*80)
    
    # Print summary table
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"{'='*80}")
    print(f"{'Exp ID':<40} {'Val Acc':<10} {'LR':<10} {'WD':<10}")
    print(f"{'-'*80}")
    
    for result in completed:
        exp_id = result['exp_id'].split('_')[-1]  # Just run number
        val_acc = result['val_acc']
        lr = result['config']['learning_rate']
        wd = result['config']['weight_decay']
        
        marker = "🏆" if result['exp_id'] == summary['best_exp_id'] else "  "
        print(f"{marker} {exp_id:<38} {val_acc:>6.2f}%   {lr:<10.2e} {wd:<10.2e}")
    
    print(f"{'='*80}\n")
    
    # Print experiment directory
    print(f"📁 Results saved to:")
    print(f"   {exp_manager.runs_dir / sweep_id}\n")
    
    return success


def test_experiment_manager():
    """Test ExperimentManager utilities."""
    
    print("\n" + "="*80)
    print("🧪 TESTING EXPERIMENT MANAGER UTILITIES")
    print("="*80)
    
    exp_manager = ExperimentManager()
    
    # Test 1: Get best experiment
    try:
        best_exp, best_acc = exp_manager.get_best_experiment(metric='val_acc', mode='max')
        if best_exp:
            print(f"✅ get_best_experiment: {best_exp} ({best_acc:.2f}%)")
        else:
            print(f"⚠️  No experiments found (expected if first run)")
    except Exception as e:
        print(f"❌ get_best_experiment failed: {e}")
    
    # Test 2: Load results
    if best_exp:
        try:
            results = exp_manager.load_results(best_exp)
            if results:
                print(f"✅ load_results: Loaded {len(results)} fields")
            else:
                print(f"❌ load_results: Failed to load")
        except Exception as e:
            print(f"❌ load_results failed: {e}")
    
    # Test 3: Summarize sweep
    try:
        summary = exp_manager.summarize_sweep()
        print(f"✅ summarize_sweep: Found {summary['total_experiments']} experiments")
    except Exception as e:
        print(f"❌ summarize_sweep failed: {e}")
    
    print(f"{"="*80}\n")


if __name__ == "__main__":
    print("\n🚀 Starting experiment management system test...\n")
    
    # Test utilities first
    test_experiment_manager()
    
    # Run minimal sweep
    success = test_minimal_sweep()
    
    # Final message
    if success:
        print("\n✅ Experiment management system working correctly!")
        print("\n📚 Next steps:")
        print("   1. Run full sweep: python experiment_manager.py")
        print("   2. Review guide: EXPERIMENT_MANAGEMENT_GUIDE.md")
        print("   3. Check quick ref: EXPERIMENT_QUICKREF.md\n")
    else:
        print("\n⚠️  Some tests failed - please review errors above\n")
    
    sys.exit(0 if success else 1)
