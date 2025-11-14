# Experiment Management Quick Reference

**Fast hyperparameter sweeps for laptop GPUs.**

---

## ⚡ TL;DR

```bash
# Quick sweep (20 min): Test 18 hyperparameter combinations
python experiment_manager.py

# Check results
ls experiments/runs/*/summary.json

# Full train best config (45 min)
python train.py --config experiments/best/config.json --epochs 25
```

---

## 🎯 Common Commands

### Default Sweep (Custom CNN)

```bash
python experiment_manager.py
```
- **Tests:** 18 combinations (3 LR × 3 WD × 2 schedulers)
- **Time:** ~20 minutes
- **Result:** Best config saved to `experiments/best/`

### EfficientNet Sweep

```bash
python experiment_manager.py --model efficientnet_b0 --batch_size 16
```
- **Tests:** 18 combinations
- **Time:** ~20 minutes
- **Best for:** Transfer learning

### Quick Test (9 combinations)

```bash
python experiment_manager.py --epochs 3
```
- **Tests:** 9 combinations (3 LR × 3 WD)
- **Time:** ~10 minutes
- **Best for:** Fast iteration

---

## 📊 Default Parameter Grid

```python
learning_rate: [1e-4, 3e-4, 1e-5]
weight_decay:  [0, 1e-5, 1e-4]
scheduler:     ['step', 'plateau']
```

**Total combinations:** 3 × 3 × 2 = **18 experiments**

---

## 🔧 Python API

### Basic Sweep

```python
from experiment_manager import grid_search

param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-5],
    'weight_decay': [0, 1e-5, 1e-4]
}

summary = grid_search(
    param_grid=param_grid,
    model_type='custom',
    num_epochs=3,
    batch_size=32
)

print(f"Best Val Acc: {summary['best_val_acc']:.2f}%")
```

### Custom Grid

```python
# Transfer learning
param_grid = {
    'learning_rate': [1e-4, 5e-5, 1e-5],
    'weight_decay': [1e-4, 5e-5],
    'warmup_epochs': [0, 1]
}

# Stability sweep
param_grid = {
    'learning_rate': [1e-4, 3e-4],
    'clip_grad_norm': [None, 0.5, 1.0],
    'accumulation_steps': [1, 2, 4]
}

# Scheduler comparison
param_grid = {
    'learning_rate': [3e-4],
    'scheduler': ['step', 'plateau', 'cosine'],
    'warmup_epochs': [0, 1, 2]
}
```

---

## 📁 Directory Structure

```
experiments/
├── sweep_registry.json              # All sweeps index
├── runs/
│   ├── sweep_20251110_143022_run001/
│   │   ├── config.json              # Hyperparameters
│   │   └── summary.json             # Results
│   ├── sweep_20251110_143022_run002/
│   └── sweep_20251110_143022_summary.json
└── best/
    └── sweep_20251110_143022_best.pth  # Best model
```

---

## 📈 Analyzing Results

### Load Results

```python
from experiment_manager import ExperimentManager

exp_manager = ExperimentManager()

# Get best experiment
best_exp, best_acc = exp_manager.get_best_experiment(
    metric='val_acc',
    mode='max'
)
print(f"Best: {best_exp} ({best_acc:.2f}%)")
```

### Summarize Sweep

```python
# Get summary
summary = exp_manager.summarize_sweep('sweep_20251110_143022')

# Print top 5
for i, exp in enumerate(summary['experiments'][:5], 1):
    result = exp['results']
    config = exp['config']
    print(f"{i}. {result['val_acc']:.2f}% | "
          f"LR={config['learning_rate']}, WD={config['weight_decay']}")
```

### Manual Analysis

```bash
# Count completed experiments
ls experiments/runs/*/summary.json | wc -l

# View best config
cat experiments/best/sweep_*/config.json | jq '.'

# Compare top 3
jq -s 'sort_by(.val_acc) | reverse | .[0:3]' experiments/runs/*/summary.json
```

---

## ⏱️ Time Estimates (Laptop)

| Epochs | Combinations | Total Time |
|--------|--------------|------------|
| 3 | 9 | 10 min |
| 3 | 18 | 20 min |
| 3 | 36 | 40 min |
| 5 | 18 | 30 min |
| 10 | 18 | 2 hours |

**Formula:** `time = epochs × combinations × 60-90 seconds`

---

## 🎯 Metric Selection

### Primary: Validation Accuracy

```python
best_exp, best_acc = exp_manager.get_best_experiment(
    metric='val_acc',
    mode='max'
)
```

**Why?**
- Direct measure of generalization
- Easy to interpret
- Aligns with goal (correct classification)

### Alternative: Validation Loss

```python
best_exp, best_loss = exp_manager.get_best_experiment(
    metric='val_loss',
    mode='min'
)
```

**When?**
- Calibration important
- Imbalanced datasets
- Multi-label problems

---

## 🚀 Scaling Strategies

### Option 1: Optuna (Sequential)

```bash
pip install optuna
```

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    wd = trial.suggest_loguniform('wd', 1e-6, 1e-3)
    
    # Train model with these params
    val_acc = train_model(lr, wd)
    return val_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best: {study.best_params}")
```

**When to use:**
- Budget > 2 hours
- Single GPU
- Want smart search (Bayesian optimization)

### Option 2: Ray Tune (Parallel)

```bash
pip install "ray[tune]"
```

```python
from ray import tune

search_space = {
    'lr': tune.loguniform(1e-5, 1e-3),
    'wd': tune.loguniform(1e-6, 1e-3)
}

analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=50,
    resources_per_trial={'gpu': 1}
)
```

**When to use:**
- Budget > 4 hours
- Multiple GPUs
- Want parallel experiments

---

## 💡 Best Practices

### 1. Start with 3 Epochs

```bash
python experiment_manager.py --epochs 3
```

**Why?**
- 3 epochs enough to see trends
- 10x faster than full training
- Can test more combinations

### 2. Tune Learning Rate First

```python
# Step 1: Find best LR
param_grid = {'learning_rate': [1e-4, 3e-4, 1e-5, 1e-6]}

# Step 2: Fine-tune weight decay
param_grid = {
    'learning_rate': [3e-4],  # Best from step 1
    'weight_decay': [0, 1e-5, 1e-4, 1e-3]
}
```

### 3. Full Train Top 3

```bash
# After sweep, pick top 3 configs
python train.py --config experiments/runs/sweep_run001/config.json --epochs 25
python train.py --config experiments/runs/sweep_run005/config.json --epochs 25
python train.py --config experiments/runs/sweep_run012/config.json --epochs 25
```

### 4. Monitor GPU Usage

```bash
# Terminal 1: Run sweep
python experiment_manager.py

# Terminal 2: Watch GPU
watch -n 1 nvidia-smi

# Terminal 3: Count completed
watch -n 5 'ls experiments/runs/*/summary.json | wc -l'
```

---

## 🔍 Troubleshooting

### Issue: Sweep Too Slow

**Problem:** 18 combinations × 3 epochs taking > 30 minutes

**Solutions:**
```bash
# Reduce combinations
python experiment_manager.py  # Use default 9 combinations

# Reduce batch size (faster but less accurate)
python experiment_manager.py --batch_size 64

# Use lighter augmentation
# Edit experiment_manager.py: augmentation_mode='conservative'
```

### Issue: Out of Memory

**Problem:** CUDA OOM during sweep

**Solutions:**
```bash
# Reduce batch size
python experiment_manager.py --batch_size 16

# Reduce model size
python experiment_manager.py --model custom  # Smaller than EfficientNet

# Enable gradient accumulation
# Edit param_grid to include 'accumulation_steps': [2, 4]
```

### Issue: All Experiments Fail

**Problem:** All runs show status='failed'

**Solutions:**
```python
# Check individual error
exp_manager = ExperimentManager()
results = exp_manager.load_results('sweep_20251110_143022_run001')
print(results['error'])

# Common fixes:
# 1. Check dataset path
# 2. Verify GPU available
# 3. Check disk space for saving
```

---

## 📋 Quick Checklist

Before running sweep:

- [ ] Dataset ready (`Database/` exists)
- [ ] GPU available (`nvidia-smi` shows GPU)
- [ ] Disk space sufficient (>2GB for experiments)
- [ ] Dependencies installed (`torch`, `torchvision`)

After running sweep:

- [ ] Check completion: `ls experiments/runs/*/summary.json | wc -l`
- [ ] View best config: `cat experiments/best/*/config.json`
- [ ] Full train top 3 configs (25 epochs each)
- [ ] Evaluate all 3 models
- [ ] Select final model for deployment

---

## 🎯 Decision Tree

```
Need to tune hyperparameters?
│
├─ Budget < 30 min?
│  └─ Use: experiment_manager.py (3 epochs, 9-18 configs)
│     Command: python experiment_manager.py
│
├─ Budget 30 min - 2 hours?
│  └─ Use: experiment_manager.py (5 epochs, 18-36 configs)
│     Command: python experiment_manager.py --epochs 5
│
├─ Budget > 2 hours + single GPU?
│  └─ Use: Optuna (smart search, 50+ trials)
│     See: EXPERIMENT_MANAGEMENT_GUIDE.md
│
└─ Budget > 4 hours + multiple GPUs?
   └─ Use: Ray Tune (parallel search, 100+ trials)
      See: EXPERIMENT_MANAGEMENT_GUIDE.md
```

---

## 📚 Resources

- **Full Guide:** `EXPERIMENT_MANAGEMENT_GUIDE.md`
- **Code:** `experiment_manager.py`
- **Results:** `experiments/`

---

**Happy experimenting! 🧪**
