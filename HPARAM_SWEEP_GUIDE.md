# 🔬 Automated Hyperparameter Optimization Guide

## **Created: `scripts/hparam_sweep.py`**

**Fully autonomous hyperparameter tuning - no user input required!**

---

## 🚀 Quick Start

### One Command - Complete Optimization

```bash
python scripts/hparam_sweep.py
```

**That's it! The script will:**
1. ✅ Run 5 experiments with different hyperparameters (5 epochs each)
2. ✅ Select the best configuration automatically
3. ✅ Retrain for 25 epochs with the best config
4. ✅ Save optimized model to `models/cropshield_cnn_best.pth`

**No arguments. No prompts. Fully automatic.**

---

## 📋 What It Does

### Phase 1: Quick Hyperparameter Sweep (~40 minutes)

The script tests 5 carefully chosen configurations:

| Config | Learning Rate | Weight Decay | Dropout | Strategy |
|--------|--------------|--------------|---------|----------|
| **1** | 0.001 | 0.0001 | 0.3 | Baseline (higher LR) |
| **2** | 0.0005 | 0.0001 | 0.3 | Moderate LR |
| **3** | 0.0001 | 0.0005 | 0.3 | Conservative LR, higher WD |
| **4** | 0.0005 | 0.0005 | 0.5 | Medium LR, high dropout |
| **5** | 0.0001 | 0.001 | 0.5 | Max regularization |

Each configuration is trained for **5 epochs** to quickly evaluate performance.

### Phase 2: Final Retrain (~50 minutes)

After identifying the best configuration:
- Trains full model for **25 epochs**
- Uses **early stopping** (patience: 10 epochs)
- Saves best model automatically
- Includes **mixed precision training** for speed
- Uses **learning rate scheduling** for optimal convergence

---

## 📊 Sample Output

```
================================================================================
🔬 CROPSHIELD AI - AUTOMATED HYPERPARAMETER OPTIMIZATION
================================================================================

================================================================================
🔬 HYPERPARAMETER SWEEP
================================================================================
📅 Started: 2025-11-14 16:00:00
🖥️  Device: cuda (NVIDIA GeForce RTX 4060 Laptop GPU)
📊 Quick Epochs: 5 | Final Epochs: 25
🔍 Testing 5 configurations
================================================================================

📋 Configurations to test:
   1. LR=0.00100, WD=0.00010, Dropout=0.3
   2. LR=0.00050, WD=0.00010, Dropout=0.3
   3. LR=0.00010, WD=0.00050, Dropout=0.3
   4. LR=0.00050, WD=0.00050, Dropout=0.5
   5. LR=0.00010, WD=0.00100, Dropout=0.5

================================================================================
🧪 EXPERIMENT exp_001
================================================================================
📊 Config: LR=0.00100, WD=0.00010, Dropout=0.3

Epoch 1/5: Train Loss=2.123, Train Acc=35.2% | Val Loss=1.877, Val Acc=42.1%
Epoch 2/5: Train Loss=1.654, Train Acc=48.7% | Val Loss=1.543, Val Acc=54.3%
Epoch 3/5: Train Loss=1.321, Train Acc=58.9% | Val Loss=1.299, Val Acc=62.5%
Epoch 4/5: Train Loss=1.099, Train Acc=66.4% | Val Loss=1.123, Val Acc=68.2%
Epoch 5/5: Train Loss=0.932, Train Acc=72.1% | Val Loss=1.012, Val Acc=71.5%

✅ Experiment Complete! Best Val Acc: 71.5% | Time: 8.2 min
💾 Saved: experiments\experiment_exp_001.json

[... experiments 2-5 ...]

================================================================================
📊 SWEEP SUMMARY
================================================================================
✅ Completed 5 experiments

🏆 Best Configuration:
   Experiment: exp_001
   Learning Rate: 0.00100
   Weight Decay: 0.00010
   Dropout: 0.3
   Best Val Acc: 71.5%

📋 Rankings:
   1. exp_001: 71.5% (LR=0.00100, WD=0.00010, Dropout=0.3)
   2. exp_002: 69.7% (LR=0.00050, WD=0.00010, Dropout=0.3)
   3. exp_004: 68.9% (LR=0.00050, WD=0.00050, Dropout=0.5)
   4. exp_003: 67.2% (LR=0.00010, WD=0.00050, Dropout=0.3)
   5. exp_005: 65.8% (LR=0.00010, WD=0.00100, Dropout=0.5)

💾 Summary saved: experiments\sweep_summary.json

================================================================================
🎯 FINAL RETRAIN WITH BEST CONFIG
================================================================================
📊 Best Config: LR=0.00100, WD=0.00010, Dropout=0.3
⏱️  Training: 25 epochs with early stopping
💾 Output: models/cropshield_cnn_best.pth

Epoch 1/25:  Train Loss=2.099, Acc=36.5% | Val Loss=1.843, Acc=43.2% | 🎉 New best!
Epoch 2/25:  Train Loss=1.623, Acc=49.8% | Val Loss=1.512, Acc=55.6% | 🎉 New best!
Epoch 3/25:  Train Loss=1.345, Acc=57.2% | Val Loss=1.287, Acc=63.4% | 🎉 New best!
...
Epoch 18/25: Train Loss=0.213, Acc=94.2% | Val Loss=0.457, Acc=89.5% | 🎉 New best!
Epoch 19/25: Train Loss=0.199, Acc=94.8% | Val Loss=0.462, Acc=89.3% | No improvement (1/10)
...
Epoch 25/25: Train Loss=0.156, Acc=96.1% | Val Loss=0.489, Acc=88.7% | No improvement (7/10)

⏹️  Training complete (25 epochs)

================================================================================
✅ FINAL RETRAIN COMPLETE!
================================================================================
⏱️  Training Time: 45.3 minutes
🎯 Best Val Accuracy: 89.5% (Epoch 18)
💾 Model Saved: models\cropshield_cnn_best.pth
💾 Results Saved: experiments\final_retrain_results.json

================================================================================
🎉 HYPERPARAMETER OPTIMIZATION COMPLETE!
================================================================================

📊 Final Results:
   Experiments: 5 completed
   Best Config: LR=0.00100, WD=0.00010, Dropout=0.3
   Best Val Acc: 89.5%
   Total Time: 1 hour 23 minutes

📁 Output Files:
   ✅ models\cropshield_cnn_best.pth
   ✅ experiments\sweep_summary.json
   ✅ experiments\final_retrain_results.json
   ✅ experiments\experiment_exp_001.json (× 5)

================================================================================
```

---

## 📁 Output Files

### Directory Structure

```
CropShieldAI/
├── models/
│   └── cropshield_cnn_best.pth       # ⭐ Optimized model weights
│
└── experiments/
    ├── experiment_exp_001.json       # Config 1 results
    ├── experiment_exp_002.json       # Config 2 results
    ├── experiment_exp_003.json       # Config 3 results
    ├── experiment_exp_004.json       # Config 4 results
    ├── experiment_exp_005.json       # Config 5 results
    ├── sweep_summary.json            # Overall sweep summary
    └── final_retrain_results.json    # Final model training results
```

---

## 📊 Output File Formats

### 1. Experiment Results: `experiment_exp_001.json`

```json
{
  "exp_id": "exp_001",
  "config": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "dropout": 0.3
  },
  "history": {
    "train_loss": [2.1234, 1.6543, 1.3210, 1.0987, 0.9321],
    "train_acc": [35.2, 48.7, 58.9, 66.4, 72.1],
    "val_loss": [1.8765, 1.5432, 1.2987, 1.1234, 1.0123],
    "val_acc": [42.1, 54.3, 62.5, 68.2, 71.5]
  },
  "best_val_acc": 71.5,
  "final_val_acc": 71.5,
  "elapsed_time": 492.3,
  "timestamp": "2025-11-14 16:08:15"
}
```

**Fields:**
- `exp_id`: Unique experiment identifier
- `config`: Hyperparameters used
- `history`: Training/validation metrics per epoch
- `best_val_acc`: Highest validation accuracy achieved
- `final_val_acc`: Accuracy at final epoch
- `elapsed_time`: Total training time (seconds)
- `timestamp`: When experiment completed

### 2. Sweep Summary: `sweep_summary.json`

```json
{
  "total_experiments": 5,
  "quick_epochs": 5,
  "search_space": {
    "learning_rate": [0.001, 0.0005, 0.0001],
    "weight_decay": [0.0001, 0.0005, 0.001],
    "dropout": [0.3, 0.5]
  },
  "best_config": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "dropout": 0.3
  },
  "best_val_acc": 71.5,
  "best_exp_id": "exp_001",
  "all_experiments": [
    {
      "exp_id": "exp_001",
      "config": {"learning_rate": 0.001, "weight_decay": 0.0001, "dropout": 0.3},
      "best_val_acc": 71.5,
      "final_val_acc": 71.5
    },
    ...
  ],
  "timestamp": "2025-11-14 16:45:30"
}
```

**Use this file to:**
- See which configuration won
- Compare all 5 configurations
- Understand hyperparameter sensitivity
- Plan future optimization strategies

### 3. Final Retrain Results: `final_retrain_results.json`

```json
{
  "best_config": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "dropout": 0.3
  },
  "best_val_acc": 89.5,
  "best_epoch": 18,
  "total_epochs": 25,
  "training_time": 2718.4,
  "history": {
    "train_loss": [2.099, 1.623, ..., 0.156],
    "train_acc": [36.5, 49.8, ..., 96.1],
    "val_loss": [1.843, 1.512, ..., 0.489],
    "val_acc": [43.2, 55.6, ..., 88.7]
  },
  "timestamp": "2025-11-14 17:30:45"
}
```

**Fields:**
- `best_config`: Winning hyperparameters
- `best_val_acc`: Peak validation accuracy
- `best_epoch`: When best model was found
- `total_epochs`: How many epochs trained
- `training_time`: Total seconds
- `history`: Complete training curves

### 4. Model Checkpoint: `cropshield_cnn_best.pth`

PyTorch checkpoint containing:
- **Model weights** (optimized parameters)
- **Optimizer state** (for resuming training)
- **Training history** (loss/accuracy curves)
- **Hyperparameter configuration** (metadata)
- **Class names** (22 plant disease classes)

**Load the model:**
```python
checkpoint = torch.load('models/cropshield_cnn_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## ⏱️ Time Estimates

### Total Pipeline Duration

| Phase | Duration | Details |
|-------|----------|---------|
| **Quick Sweep** | ~40 min | 5 configs × 5 epochs × ~8 min |
| **Final Retrain** | ~50 min | 25 epochs × ~2 min (with early stopping) |
| **Total** | **~90 min** | **~1.5 hours** |

### Per-Component Timing (RTX 4060)

- **Per Epoch:** ~1.5-2 minutes
- **Quick Experiment:** ~8-10 minutes (5 epochs)
- **Full Sweep:** ~40-50 minutes (all 5 experiments)
- **Final Retrain:** ~45-60 minutes (25 epochs with early stopping)

**Note:** Times may vary based on:
- GPU speed (RTX 4060 Laptop used for estimates)
- Dataset size (22,387 images)
- Batch size (default: 32)
- System load

---

## 🎯 Use Cases

### 1. First-Time Model Optimization

```bash
# After training baseline model
python train_auto.py --epochs 25

# Optimize hyperparameters
python scripts/hparam_sweep.py

# Result: models/cropshield_cnn_best.pth
```

### 2. Improve Suboptimal Performance

```bash
# If validation accuracy is low (<80%)
python scripts/hparam_sweep.py

# Script will find better hyperparameters automatically
```

### 3. Experiment Tracking & Reproducibility

```bash
# Run optimization
python scripts/hparam_sweep.py

# Review all experiments
cat experiments/sweep_summary.json

# Check best configuration
cat experiments/final_retrain_results.json
```

### 4. Before Deployment

```bash
# Ensure optimal performance before production
python scripts/hparam_sweep.py
python quick_evaluate.py  # Test optimized model
python test_model_inference.py  # Verify predictions
```

---

## 📈 Interpreting Results

### 1. Sweep Summary Analysis

**Look at:** `experiments/sweep_summary.json`

**Questions to answer:**
- **Which config won?** → Check `best_exp_id` and `best_config`
- **How close were other configs?** → Compare `best_val_acc` across experiments
- **Is there a pattern?** → Does higher/lower LR consistently perform better?

**Example interpretations:**
```
Scenario 1: Config 1 wins by large margin (71.5% vs 65-68%)
→ Higher learning rate works well for this dataset

Scenario 2: All configs within 1-2% (69-71%)
→ Model is robust; hyperparameters less critical

Scenario 3: High dropout configs fail (60% vs 70%)
→ Model needs capacity; reduce regularization
```

### 2. Final Retrain Analysis

**Look at:** `experiments/final_retrain_results.json`

**Key metrics:**
- **best_val_acc:** Final optimized model accuracy
- **best_epoch:** When peak performance occurred
- **total_epochs:** How many epochs needed

**What to check:**
```python
import json

with open('experiments/final_retrain_results.json') as f:
    results = json.load(f)

# Did training converge?
print(f"Best: {results['best_val_acc']:.2f}% at epoch {results['best_epoch']}")

# Was early stopping effective?
if results['best_epoch'] < results['total_epochs'] - 10:
    print("✅ Early stopping saved time!")
else:
    print("❌ Consider training longer")

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(results['history']['val_acc'])
plt.title('Validation Accuracy Over Time')
plt.show()
```

### 3. Hyperparameter Sensitivity

**Compare all 5 experiments:**

```python
import json
import glob

# Load all experiments
experiments = []
for file in sorted(glob.glob('experiments/experiment_*.json')):
    with open(file) as f:
        experiments.append(json.load(f))

# Analyze learning rate effect
for exp in sorted(experiments, key=lambda x: x['config']['learning_rate'], reverse=True):
    lr = exp['config']['learning_rate']
    acc = exp['best_val_acc']
    print(f"LR={lr:.4f} → {acc:.1f}%")

# Output:
# LR=0.0010 → 71.5%  ← Higher LR wins
# LR=0.0005 → 69.7%
# LR=0.0001 → 67.2%
```

**Insights:**
- **Learning rate matters most** if large accuracy differences
- **Weight decay matters** if configs with same LR differ significantly
- **Dropout matters** if high-dropout configs underperform

---

## 🔧 Customization

### Modify Search Space

Edit `scripts/hparam_sweep.py` (around line 40):

```python
# Current search space
self.search_space = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'weight_decay': [0.0001, 0.0005, 0.001],
    'dropout': [0.3, 0.5]
}

# Example: Test more learning rates
self.search_space = {
    'learning_rate': [0.01, 0.005, 0.001, 0.0005],  # Added higher LRs
    'weight_decay': [0.0001, 0.001],                 # Simplified
    'dropout': [0.3]                                  # Fixed
}
```

### Adjust Training Duration

```python
# At bottom of script in main()
sweep = HyperparameterSweep(
    quick_epochs=3,    # Faster sweep (default: 5)
    final_epochs=50,   # Longer final train (default: 25)
    batch_size=64      # Larger batches (default: 32)
)
```

### Add More Configurations

Modify `generate_configs()` method:

```python
def generate_configs(self) -> List[Dict]:
    configs = [
        # Original 5 configs...
        
        # Add new configs
        {'learning_rate': 0.002, 'weight_decay': 0.0001, 'dropout': 0.4},
        {'learning_rate': 0.0003, 'weight_decay': 0.0002, 'dropout': 0.35},
    ]
    return configs
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```python
sweep = HyperparameterSweep(batch_size=16)  # Default: 32
```

2. **Reduce model size in `model_setup.py`:**
```python
# Reduce filters/layers if using custom CNN
```

3. **Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

### Issue: Training Takes Too Long

**Symptoms:** Sweep taking >3 hours

**Solutions:**

1. **Reduce epochs:**
```python
sweep = HyperparameterSweep(quick_epochs=3, final_epochs=15)
```

2. **Test fewer configs:**
```python
# In generate_configs(), comment out some configs
configs = [
    self._make_config(0.001, 0.0001, 0.3),  # Keep this
    # self._make_config(0.0005, 0.0001, 0.3),  # Skip this
    ...
]
```

3. **Increase batch size (if GPU has room):**
```python
sweep = HyperparameterSweep(batch_size=64)
```

### Issue: All Configs Perform Poorly

**Symptoms:** All experiments <60% validation accuracy

**Solutions:**

1. **Check data loading:**
```bash
python quick_verify.py  # Verify dataset integrity
```

2. **Verify model architecture:**
```python
from model_setup import get_model
model = get_model('custom', num_classes=22)
print(model)  # Check architecture
```

3. **Try different search space:**
```python
# Higher learning rates
self.search_space = {
    'learning_rate': [0.01, 0.005, 0.001],  # Higher range
    ...
}
```

### Issue: No Improvement During Final Retrain

**Symptoms:** Best epoch = 1-5, no improvement after

**Solutions:**

1. **Check if model converged too fast:**
   - View `experiments/final_retrain_results.json`
   - Look at training curves

2. **Reduce learning rate for final train:**
```python
# In retrain_with_best_config(), modify optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=self.best_config['learning_rate'] * 0.5,  # Half the sweep LR
    ...
)
```

3. **Increase patience:**
```python
# In retrain_with_best_config()
patience = 20  # Default: 10
```

---

## ✅ Features Summary

### What the Script Does

| Feature | Description | Benefit |
|---------|-------------|---------|
| **5 Configurations** | Tests diverse hyperparameter combinations | Finds optimal settings |
| **Quick Experiments** | 5 epochs per config | Fast evaluation (~8 min each) |
| **Auto Selection** | Chooses best by validation accuracy | No manual analysis needed |
| **Final Retrain** | 25 epochs with best config | Production-ready model |
| **Early Stopping** | Stops if no improvement (10 epochs) | Saves time |
| **Mixed Precision** | AMP for faster training | ~1.5× speedup |
| **LR Scheduling** | StepLR (step=5, gamma=0.5) | Better convergence |
| **JSON Logging** | All results saved automatically | Full reproducibility |
| **Progress Display** | Real-time training updates | Monitor progress |
| **No User Input** | Fully autonomous execution | Set it and forget it |

---

## 📚 Next Steps

### After Optimization Completes

1. **Evaluate Optimized Model:**
```bash
python quick_evaluate.py
# Uses models/cropshield_cnn_best.pth automatically
```

2. **Test Inference:**
```bash
python test_model_inference.py
# Tests optimized model on random images
```

3. **Compare to Baseline:**
```python
# Evaluate both models
python quick_evaluate.py  # Tests cropshield_cnn_best.pth

# Compare results
cat results/test_metrics.json
```

4. **Deploy Optimized Model:**
```bash
# Export to ONNX
python export_onnx.py --model models/cropshield_cnn_best.pth

# Run Streamlit app
streamlit run app.py
```

---

## 🎉 Summary

**Script:** `scripts/hparam_sweep.py`

**Purpose:** Automated hyperparameter optimization

**What it optimizes:**
- ⚡ Learning Rate (3 values)
- 🎯 Weight Decay (3 values)
- 🔒 Dropout (2 values)

**How it works:**
1. Tests 5 carefully chosen configurations (5 epochs each)
2. Selects best configuration automatically
3. Retrains for 25 epochs with best config
4. Saves optimized model

**Usage:**
```bash
python scripts/hparam_sweep.py
```

**Time:** ~90 minutes

**Output:**
- ⭐ `models/cropshield_cnn_best.pth` - Optimized model
- 📊 `experiments/sweep_summary.json` - All results
- 📈 `experiments/final_retrain_results.json` - Training curves
- 📝 `experiments/experiment_*.json` - Individual experiments

**Fully automated. No user input. Production-ready results.**

---

*CropShield AI - Automated Hyperparameter Optimization*  
*Created: 2025-11-14*
