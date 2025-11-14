# Experiment Management Guide

**Complete system for hyperparameter sweeps and experiment tracking.**

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Local Grid Search](#local-grid-search)
3. [Experiment Tracking](#experiment-tracking)
4. [Quick Sweep Strategy](#quick-sweep-strategy)
5. [Metric Selection](#metric-selection)
6. [Scaling with Ray Tune](#scaling-with-ray-tune)
7. [Scaling with Optuna](#scaling-with-optuna)
8. [Best Practices](#best-practices)

---

## 🚀 Quick Start

### Run Your First Sweep (3 minutes)

```bash
# Default: LR + Weight Decay sweep, 3 epochs each
python experiment_manager.py

# Custom model sweep
python experiment_manager.py --model custom --epochs 3

# EfficientNet sweep
python experiment_manager.py --model efficientnet_b0 --epochs 3 --batch_size 16
```

**What happens:**
- Tests 18 hyperparameter combinations (3 LR × 3 WD × 2 schedulers)
- Each run: 3 epochs (~30-60 seconds)
- Total time: ~15-20 minutes on laptop
- Auto-saves all configs and results
- Picks best model by validation accuracy

---

## 📊 Local Grid Search

### Basic Usage

```python
from experiment_manager import grid_search

# Define parameter grid
param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-5],
    'weight_decay': [0, 1e-5, 1e-4],
    'scheduler': ['step', 'plateau']
}

# Run sweep
summary = grid_search(
    param_grid=param_grid,
    model_type='custom',
    num_epochs=3,  # Quick: 3 epochs only
    batch_size=32,
    sweep_name='my_first_sweep'
)

# Results
print(f"Best Val Acc: {summary['best_val_acc']:.2f}%")
print(f"Best Config: {summary['best_exp_id']}")
```

### Parameter Grid Options

```python
# Minimal sweep (9 combinations, ~10 min)
param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-5],
    'weight_decay': [0, 1e-5, 1e-4]
}

# Scheduler sweep (18 combinations, ~20 min)
param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-5],
    'weight_decay': [0, 1e-5, 1e-4],
    'scheduler': ['step', 'plateau']
}

# Full sweep (36 combinations, ~40 min)
param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-5],
    'weight_decay': [0, 1e-5, 1e-4],
    'scheduler': ['step', 'plateau'],
    'clip_grad_norm': [None, 1.0]
}

# Transfer learning sweep (12 combinations)
param_grid = {
    'learning_rate': [1e-4, 5e-5, 1e-5],
    'weight_decay': [1e-4, 5e-5],
    'warmup_epochs': [0, 1]
}
```

### Advanced Options

```python
summary = grid_search(
    param_grid=param_grid,
    model_type='efficientnet_b0',      # Model architecture
    num_epochs=3,                      # Epochs per run
    batch_size=16,                     # Batch size
    num_workers=8,                     # DataLoader workers
    augmentation_mode='moderate',      # Augmentation
    sweep_name='efficientnet_sweep',   # Custom name
    device='cuda',                     # Device
    save_best_only=True                # Save only best model
)
```

---

## 🗂️ Experiment Tracking

### Directory Structure

```
experiments/
├── sweep_registry.json           # All sweeps index
├── runs/                         # Individual experiments
│   ├── sweep_20251110_143022_run001/
│   │   ├── config.json           # Hyperparameters
│   │   ├── summary.json          # Results
│   │   └── checkpoints/          # Model checkpoints
│   ├── sweep_20251110_143022_run002/
│   │   └── ...
│   └── sweep_20251110_143022_summary.json  # Sweep summary
└── best/                         # Best models
    └── sweep_20251110_143022_best.pth
```

### Config Format (`config.json`)

```json
{
  "learning_rate": 0.0003,
  "weight_decay": 1e-05,
  "scheduler": "plateau",
  "model_type": "custom",
  "num_epochs": 3,
  "batch_size": 32,
  "augmentation_mode": "moderate"
}
```

### Results Format (`summary.json`)

```json
{
  "exp_id": "sweep_20251110_143022_run001",
  "config": { ... },
  "train_loss": 0.3245,
  "train_acc": 89.42,
  "val_loss": 0.4123,
  "val_acc": 87.65,
  "train_time_seconds": 45.2,
  "history": {
    "train_loss": [1.234, 0.567, 0.345],
    "val_acc": [82.1, 85.3, 87.6]
  },
  "status": "completed"
}
```

### Analyzing Results

```python
from experiment_manager import ExperimentManager

# Load manager
exp_manager = ExperimentManager()

# Get best experiment
best_exp, best_acc = exp_manager.get_best_experiment(
    metric='val_acc',
    mode='max'
)
print(f"Best: {best_exp} with {best_acc:.2f}% val acc")

# Load specific results
results = exp_manager.load_results('sweep_20251110_143022_run001')
print(f"Val Acc: {results['val_acc']:.2f}%")
print(f"Config: {results['config']}")

# Summarize sweep
summary = exp_manager.summarize_sweep('sweep_20251110_143022')
print(f"Total experiments: {summary['total_experiments']}")

# Print top 5
for i, exp in enumerate(summary['experiments'][:5], 1):
    print(f"{i}. {exp['exp_id']}: {exp['results']['val_acc']:.2f}%")
```

---

## ⚡ Quick Sweep Strategy

### Why 3 Epochs?

**Goal:** Find promising hyperparameters quickly, then do full training.

**Strategy:**
1. **Quick sweep (3 epochs):** Test many combinations fast (~20 min)
2. **Pick top 3:** Select best performing configs
3. **Full training (25 epochs):** Train top 3 for full epochs (~1 hour each)
4. **Final selection:** Pick absolute best for deployment

**Benefits:**
- 3 epochs enough to see trends (good vs bad hyperparameters)
- 3 epochs × 18 configs = 54 epochs total ≈ 2 full runs
- Saves 90% time vs full sweep (18 × 25 epochs = 450 epochs!)

### Stopping Criterion

```python
# Option 1: Fixed epochs (recommended for quick sweeps)
num_epochs = 3  # Fast: see trends in 3 epochs

# Option 2: Early stopping (for longer sweeps)
num_epochs = 10
early_stopping_patience = 3  # Stop if no improvement for 3 epochs

# Option 3: Time-based (for overnight sweeps)
max_time_per_experiment = 300  # 5 minutes max
```

### Quick Sweep Recommendations

| Sweep Type | Epochs | Combinations | Total Time (Laptop) |
|------------|--------|--------------|---------------------|
| **Minimal** | 3 | 9 (3 LR × 3 WD) | ~10 min |
| **Standard** | 3 | 18 (+ scheduler) | ~20 min |
| **Full** | 3 | 36 (+ clip_grad) | ~40 min |
| **Deep** | 5 | 18 | ~30 min |
| **Overnight** | 10 | 18 | ~2 hours |

**Recommendation:** Start with **Standard (3 epochs, 18 combinations)**.

---

## 📈 Metric Selection

### Primary Metric: **Validation Accuracy**

**Why `val_acc`?**
- Direct measure of generalization
- Easy to interpret (percentage correct)
- Aligns with final goal (classify correctly)

**When to use `val_loss` instead:**
- Calibration important (probabilities matter)
- Multi-label problems
- Very imbalanced datasets

### Ranking Runs

```python
# Rank by validation accuracy (default)
best_exp, best_acc = exp_manager.get_best_experiment(
    metric='val_acc',
    mode='max'  # Higher is better
)

# Rank by validation loss (alternative)
best_exp, best_loss = exp_manager.get_best_experiment(
    metric='val_loss',
    mode='min'  # Lower is better
)

# Custom: Rank by train-val gap (generalization)
for exp in experiments:
    gap = exp['results']['train_acc'] - exp['results']['val_acc']
    # Prefer smaller gap (less overfitting)
```

### Secondary Metrics

Consider these for tie-breaking:

1. **Generalization Gap:** `train_acc - val_acc` (lower is better)
   - Large gap = overfitting
   - Prefer configs with small gap

2. **Training Speed:** `train_time_seconds` (lower is better)
   - Important for production retraining

3. **Stability:** Variance of `val_acc` across epochs
   - Smooth curves = stable training

4. **Convergence Speed:** Epochs to reach 90% of best accuracy
   - Faster learners may generalize better

### Multi-Metric Selection

```python
# Score function combining multiple metrics
def score_experiment(results):
    val_acc = results['val_acc']
    train_acc = results['train_acc']
    train_time = results['train_time_seconds']
    
    # Penalize overfitting
    overfit_penalty = max(0, train_acc - val_acc - 3.0)
    
    # Reward speed (normalized)
    speed_bonus = 1.0 / (train_time / 60.0)  # Points per minute
    
    score = val_acc - overfit_penalty + 0.1 * speed_bonus
    return score

# Rank by composite score
experiments.sort(key=lambda x: score_experiment(x['results']), reverse=True)
```

---

## 🚀 Scaling with Ray Tune

### Why Ray Tune?

**Benefits:**
- Parallel experiments (use all GPU/CPU cores)
- Advanced search: Bayesian, HyperBand, ASHA
- Automatic checkpointing and resuming
- Trial scheduling (early stopping bad runs)

**When to use:**
- Many hyperparameters (>50 combinations)
- Long training times (>10 epochs)
- Multiple GPUs available
- Budget for cloud compute

### Installation

```bash
pip install "ray[tune]" optuna
```

### Ray Tune Example

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import torch

def train_model(config):
    """Training function for Ray Tune."""
    # Load data
    train_loader, val_loader, _, class_names, _ = make_loaders(
        batch_size=config['batch_size']
    )
    
    # Create model
    model, optimizer, criterion, scheduler, device = get_model(
        model_type=config['model_type'],
        num_classes=len(class_names),
        learning_rate=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(class_names)
    )
    
    # Train and report to Ray
    for epoch in range(config['num_epochs']):
        history = trainer.train(max_epochs=epoch+1)
        
        # Report metrics to Ray Tune
        tune.report(
            val_acc=history['val_acc'][-1],
            val_loss=history['val_loss'][-1]
        )

# Define search space
search_space = {
    'lr': tune.loguniform(1e-5, 1e-3),
    'weight_decay': tune.loguniform(1e-6, 1e-3),
    'batch_size': tune.choice([16, 32, 64]),
    'model_type': tune.choice(['custom', 'efficientnet_b0']),
    'num_epochs': 10
}

# ASHA scheduler: early stopping for bad trials
scheduler = ASHAScheduler(
    metric='val_acc',
    mode='max',
    max_t=10,  # Max epochs
    grace_period=3,  # Min epochs before stopping
    reduction_factor=2
)

# Run tuning
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=20,  # Number of trials
    scheduler=scheduler,
    resources_per_trial={'gpu': 1}
)

# Get best trial
best_trial = analysis.get_best_trial('val_acc', 'max', 'last')
print(f"Best config: {best_trial.config}")
print(f"Best val_acc: {best_trial.last_result['val_acc']:.2f}%")
```

### Ray Tune Search Algorithms

```python
# 1. Grid Search (exhaustive)
from ray.tune.search import grid_search
search_space = {
    'lr': grid_search([1e-4, 3e-4, 1e-5]),
    'weight_decay': grid_search([0, 1e-5, 1e-4])
}

# 2. Random Search (sample randomly)
search_space = {
    'lr': tune.loguniform(1e-5, 1e-3),
    'weight_decay': tune.loguniform(1e-6, 1e-3)
}

# 3. Bayesian Optimization (smart search)
from ray.tune.search.optuna import OptunaSearch
search_alg = OptunaSearch(
    metric='val_acc',
    mode='max'
)

# 4. HyperBand (adaptive budget)
from ray.tune.schedulers import HyperBandScheduler
scheduler = HyperBandScheduler(
    metric='val_acc',
    mode='max',
    max_t=25  # Max epochs
)
```

---

## 🎯 Scaling with Optuna

### Why Optuna?

**Benefits:**
- Smart search (Tree-structured Parzen Estimator)
- Pruning (stop bad trials early)
- Simple API (easier than Ray Tune)
- Good for sequential experiments

**When to use:**
- Single GPU/machine
- Sequential trials (not parallel)
- Want smart search without Ray complexity

### Optuna Example

```python
import optuna

def objective(trial):
    """Objective function for Optuna."""
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    scheduler = trial.suggest_categorical('scheduler', ['step', 'plateau'])
    
    # Load data
    train_loader, val_loader, _, class_names, _ = make_loaders(
        batch_size=batch_size
    )
    
    # Create model
    model, optimizer, criterion, sched, device = get_model(
        model_type='custom',
        num_classes=len(class_names),
        learning_rate=lr,
        weight_decay=weight_decay,
        scheduler_type=scheduler
    )
    
    # Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=sched,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(class_names)
    )
    
    history = trainer.train(max_epochs=5)
    
    # Return metric to optimize
    return history['val_acc'][-1]

# Create study
study = optuna.create_study(
    direction='maximize',  # Maximize val_acc
    pruner=optuna.pruners.MedianPruner()  # Prune bad trials
)

# Run optimization
study.optimize(objective, n_trials=50)

# Results
print(f"Best value: {study.best_value:.2f}%")
print(f"Best params: {study.best_params}")

# Visualize
import optuna.visualization as vis
fig = vis.plot_optimization_history(study)
fig.show()
```

### Optuna Advanced Features

```python
# 1. Multi-objective optimization
study = optuna.create_study(
    directions=['maximize', 'minimize'],  # [val_acc, train_time]
    sampler=optuna.samplers.NSGAIISampler()
)

# 2. Conditional parameters
def objective(trial):
    model_type = trial.suggest_categorical('model', ['custom', 'efficientnet'])
    
    if model_type == 'efficientnet':
        # EfficientNet-specific params
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-4)
    else:
        # Custom CNN params
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-3)

# 3. Pruning (early stopping)
for epoch in range(10):
    val_acc = train_one_epoch()
    
    # Report intermediate value
    trial.report(val_acc, epoch)
    
    # Prune if not promising
    if trial.should_prune():
        raise optuna.TrialPruned()
```

---

## 💡 Best Practices

### 1. Start Small, Scale Up

```python
# Step 1: Quick local sweep (3 epochs, 9-18 combinations)
python experiment_manager.py --epochs 3

# Step 2: Full train top 3 configs (25 epochs each)
python train.py --config experiments/runs/best_run/config.json --epochs 25

# Step 3: Scale with Ray Tune (if needed)
# Only if >50 combinations or multiple GPUs available
```

### 2. Sweep Strategy by Budget

| Time Budget | Strategy | Tool |
|-------------|----------|------|
| **30 min** | Grid: 3 epochs, 9 configs | `experiment_manager.py` |
| **1 hour** | Grid: 3 epochs, 18 configs | `experiment_manager.py` |
| **4 hours** | Grid: 5 epochs, 36 configs | `experiment_manager.py` |
| **Overnight** | Random: 10 epochs, 50 trials | Optuna |
| **Weekend** | Bayesian: 25 epochs, 100 trials | Ray Tune + Optuna |

### 3. Parameter Priority

**Tune in this order:**

1. **Learning Rate** (highest impact)
   - Try: `[1e-4, 3e-4, 1e-5]`
   - Transfer learning: `[1e-4, 5e-5, 1e-5]`

2. **Weight Decay** (regularization)
   - Try: `[0, 1e-5, 1e-4]`

3. **Scheduler** (convergence)
   - Try: `['step', 'plateau']`

4. **Batch Size** (if VRAM allows)
   - Try: `[16, 32, 64]`

5. **Other** (fine-tuning)
   - `clip_grad_norm`, `warmup_epochs`, `accumulation_steps`

### 4. Laptop-Friendly Configurations

```python
# ✅ Good: Fast sweep (20 min)
param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-5],
    'weight_decay': [0, 1e-5, 1e-4]
}
num_epochs = 3

# ❌ Bad: Too slow (4+ hours)
param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-5, 1e-6],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3],
    'batch_size': [16, 32, 64],
    'scheduler': ['step', 'plateau', 'cosine']
}
num_epochs = 10  # 4 × 4 × 3 × 3 = 144 configs × 10 epochs = WAY TOO MUCH
```

### 5. Monitoring Progress

```bash
# Terminal 1: Run sweep
python experiment_manager.py

# Terminal 2: Watch results (live)
watch -n 5 'ls -lh experiments/runs/*/summary.json | wc -l'

# Terminal 3: Check GPU usage
watch -n 1 nvidia-smi
```

### 6. Resuming Interrupted Sweeps

```python
# Load experiment manager
exp_manager = ExperimentManager()

# Check which experiments completed
summary = exp_manager.summarize_sweep('sweep_20251110_143022')
completed = [e for e in summary['experiments'] if e['results'].get('status') == 'completed']

print(f"Completed: {len(completed)} / {summary['total_experiments']}")

# Re-run failed experiments manually
failed_configs = [e['config'] for e in summary['experiments'] 
                  if e['results'].get('status') == 'failed']
```

---

## 📚 Summary

### Quick Reference

| Task | Command | Time |
|------|---------|------|
| **Quick sweep** | `python experiment_manager.py` | 20 min |
| **Custom model** | `python experiment_manager.py --model custom --epochs 3` | 20 min |
| **Transfer learning** | `python experiment_manager.py --model efficientnet_b0 --epochs 3` | 20 min |
| **Analyze results** | See `experiments/runs/` | - |
| **Scale with Optuna** | Use code above | 2+ hours |
| **Scale with Ray Tune** | Use code above | 4+ hours |

### Decision Tree

```
Need hyperparameter tuning?
│
├─ Budget < 1 hour?
│  └─ Use: experiment_manager.py (3 epochs, 9-18 configs)
│
├─ Budget 1-4 hours?
│  └─ Use: experiment_manager.py (5 epochs, 18-36 configs)
│
├─ Budget > 4 hours + single GPU?
│  └─ Use: Optuna (10+ epochs, 50-100 trials)
│
└─ Budget > 4 hours + multiple GPUs?
   └─ Use: Ray Tune (25 epochs, 100+ trials, parallel)
```

### Key Metrics

- **Primary:** `val_acc` (validation accuracy) - maximize
- **Secondary:** `train_acc - val_acc` (generalization gap) - minimize
- **Tertiary:** `train_time_seconds` (speed) - minimize

### Next Steps

1. **Run first sweep:** `python experiment_manager.py`
2. **Review results:** Check `experiments/runs/`
3. **Full train top 3:** Use best configs for 25 epochs
4. **Scale if needed:** Optuna (sequential) or Ray Tune (parallel)

---

**Ready to optimize! 🚀**
