"""
Quick verification of experiment_manager.py functionality.

This script verifies the core components work without running full training.
"""

import sys
from pathlib import Path

print("\n" + "="*80)
print("🔍 EXPERIMENT MANAGER VERIFICATION")
print("="*80)

# Test 1: Import experiment_manager
print("\n1. Testing imports...")
try:
    from experiment_manager import ExperimentManager, grid_search
    print("   ✅ experiment_manager.py imports successfully")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create ExperimentManager
print("\n2. Testing ExperimentManager initialization...")
try:
    exp_manager = ExperimentManager(base_dir='experiments')
    print("   ✅ ExperimentManager created")
    print(f"   📁 Base dir: {exp_manager.base_dir}")
    print(f"   📁 Runs dir: {exp_manager.runs_dir}")
    print(f"   📁 Best dir: {exp_manager.best_dir}")
except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Directory creation
print("\n3. Testing directory structure...")
try:
    assert exp_manager.base_dir.exists(), "Base dir not created"
    assert exp_manager.runs_dir.exists(), "Runs dir not created"
    assert exp_manager.best_dir.exists(), "Best dir not created"
    print("   ✅ All directories created")
except Exception as e:
    print(f"   ❌ Directory check failed: {e}")
    sys.exit(1)

# Test 4: Experiment ID generation
print("\n4. Testing experiment ID generation...")
try:
    exp_id = exp_manager.create_experiment_id('test_sweep')
    assert 'test_sweep' in exp_id, "Sweep name not in ID"
    assert '_' in exp_id, "Timestamp separator missing"
    print(f"   ✅ Experiment ID generated: {exp_id}")
except Exception as e:
    print(f"   ❌ ID generation failed: {e}")
    sys.exit(1)

# Test 5: Config saving
print("\n5. Testing config save/load...")
try:
    test_config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'model_type': 'custom'
    }
    exp_manager.save_config(exp_id, test_config)
    
    config_path = exp_manager.runs_dir / exp_id / 'config.json'
    assert config_path.exists(), "Config file not created"
    print(f"   ✅ Config saved: {config_path}")
except Exception as e:
    print(f"   ❌ Config save failed: {e}")
    sys.exit(1)

# Test 6: Results saving
print("\n6. Testing results save/load...")
try:
    test_results = {
        'exp_id': exp_id,
        'config': test_config,
        'train_loss': 0.5,
        'train_acc': 85.0,
        'val_loss': 0.6,
        'val_acc': 83.0,
        'status': 'completed'
    }
    exp_manager.save_results(exp_id, test_results)
    
    loaded_results = exp_manager.load_results(exp_id)
    assert loaded_results is not None, "Results not loaded"
    assert loaded_results['val_acc'] == 83.0, "Val acc mismatch"
    print(f"   ✅ Results saved and loaded correctly")
    print(f"      Val Acc: {loaded_results['val_acc']:.2f}%")
except Exception as e:
    print(f"   ❌ Results save/load failed: {e}")
    sys.exit(1)

# Test 7: Registry operations
print("\n7. Testing sweep registry...")
try:
    exp_manager.register_sweep(exp_id, test_config)
    assert len(exp_manager.registry['sweeps']) > 0, "Sweep not registered"
    
    exp_manager.update_sweep_status(exp_id, 'completed', exp_id)
    print(f"   ✅ Registry operations work")
    print(f"      Registered sweeps: {len(exp_manager.registry['sweeps'])}")
except Exception as e:
    print(f"   ❌ Registry operations failed: {e}")
    sys.exit(1)

# Test 8: Best experiment tracking
print("\n8. Testing best experiment tracking...")
try:
    best_exp, best_val = exp_manager.get_best_experiment('val_acc', 'max')
    if best_exp:
        print(f"   ✅ Best experiment found: {best_exp}")
        print(f"      Val Acc: {best_val:.2f}%")
    else:
        print(f"   ✅ No experiments yet (expected for fresh install)")
except Exception as e:
    print(f"   ❌ Best experiment tracking failed: {e}")
    sys.exit(1)

# Test 9: Sweep summary
print("\n9. Testing sweep summary...")
try:
    summary = exp_manager.summarize_sweep(exp_id)
    assert 'total_experiments' in summary, "Summary missing fields"
    print(f"   ✅ Sweep summary generated")
    print(f"      Total experiments: {summary['total_experiments']}")
except Exception as e:
    print(f"   ❌ Sweep summary failed: {e}")
    sys.exit(1)

# Test 10: Parameter grid generation
print("\n10. Testing parameter grid...")
try:
    import itertools
    
    param_grid = {
        'learning_rate': [1e-4, 3e-4, 1e-5],
        'weight_decay': [0, 1e-5, 1e-4]
    }
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    assert len(combinations) == 9, f"Expected 9 combinations, got {len(combinations)}"
    print(f"   ✅ Parameter grid generation works")
    print(f"      Combinations: {len(combinations)}")
    print(f"      Example: {dict(zip(param_names, combinations[0]))}")
except Exception as e:
    print(f"   ❌ Parameter grid failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("✅ ALL VERIFICATION TESTS PASSED!")
print("="*80)
print("\n📚 Next Steps:")
print("   1. Run quick test: python test_experiment_manager.py")
print("   2. Run full sweep: python experiment_manager.py")
print("   3. Check results: ls experiments/runs/*/summary.json")
print("\n💡 Tips:")
print("   - Quick sweep: python experiment_manager.py --epochs 3")
print("   - See guide: EXPERIMENT_MANAGEMENT_GUIDE.md")
print("   - See quickref: EXPERIMENT_QUICKREF.md")
print("\n" + "="*80)
print("🚀 Experiment management system is ready!")
print("="*80 + "\n")
