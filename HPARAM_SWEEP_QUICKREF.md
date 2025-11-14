# 🚀 Hyperparameter Optimization - Quick Reference

## One-Line Command

```bash
python scripts/hparam_sweep.py
```

**No arguments. No prompts. Fully automatic!**

---

## ⏱️ Time Required

- **Quick Sweep:** 40 minutes (5 experiments × 5 epochs)
- **Final Retrain:** 50 minutes (25 epochs)
- **Total:** ~90 minutes (~1.5 hours)

---

## 📊 What Gets Optimized

| Parameter | Tested Values | Purpose |
|-----------|---------------|---------|
| **Learning Rate** | 0.001, 0.0005, 0.0001 | Training speed |
| **Weight Decay** | 0.0001, 0.0005, 0.001 | Regularization |
| **Dropout** | 0.3, 0.5 | Overfitting prevention |

**Total:** 5 configurations tested

---

## 📁 Output Files

```
models/
└── cropshield_cnn_best.pth          ⭐ Optimized model

experiments/
├── experiment_exp_001.json          Config 1 results
├── experiment_exp_002.json          Config 2 results
├── experiment_exp_003.json          Config 3 results
├── experiment_exp_004.json          Config 4 results
├── experiment_exp_005.json          Config 5 results
├── sweep_summary.json               📊 Overall summary
└── final_retrain_results.json       Final training results
```

---

## 🔍 Check Results

### View Best Configuration
```bash
cat experiments/sweep_summary.json
```

### View Final Model Performance
```bash
cat experiments/final_retrain_results.json
```

### Evaluate Optimized Model
```bash
python quick_evaluate.py
```

### Test Inference
```bash
python test_model_inference.py
```

---

## 📈 Sample Output

```
🔬 HYPERPARAMETER SWEEP
Testing 5 configurations...

🧪 EXPERIMENT exp_001
Config: LR=0.00100, WD=0.00010, Dropout=0.3
Epoch 5/5: Val Acc=71.5% ✅

[... 4 more experiments ...]

🏆 Best Configuration:
   LR=0.00100, WD=0.00010, Dropout=0.3
   Val Acc: 71.5%

🎯 FINAL RETRAIN (25 epochs)
Best Val Acc: 89.5% (Epoch 18)
Model saved: models/cropshield_cnn_best.pth

✅ OPTIMIZATION COMPLETE!
Total time: 1 hour 23 minutes
```

---

## 🔧 Quick Customization

### Faster Sweep
```python
# Edit hparam_sweep.py at bottom
sweep = HyperparameterSweep(
    quick_epochs=3,    # Default: 5
    final_epochs=15    # Default: 25
)
```

### Smaller Batches (if GPU memory issues)
```python
sweep = HyperparameterSweep(batch_size=16)  # Default: 32
```

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce batch size: `batch_size=16` |
| **Takes too long** | Reduce epochs: `quick_epochs=3` |
| **Poor results** | Check data: `python quick_verify.py` |

---

## ✅ Complete Workflow

```bash
# 1. Train baseline model (if not done)
python train_auto.py --epochs 25

# 2. Optimize hyperparameters
python scripts/hparam_sweep.py      # ~90 minutes

# 3. Evaluate optimized model
python quick_evaluate.py

# 4. Test inference
python test_model_inference.py

# 5. Deploy (optional)
python export_onnx.py --model models/cropshield_cnn_best.pth
streamlit run app.py
```

---

## 📚 Full Documentation

See: `HPARAM_SWEEP_GUIDE.md`

---

*CropShield AI - Hyperparameter Optimization*
