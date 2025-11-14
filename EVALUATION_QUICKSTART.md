# 🎯 Quick Evaluation Guide - CropShield AI

## ✅ NEW: Fully Automatic Evaluation Script

**File Created:** `quick_evaluate.py`

### 🚀 One Command - Complete Evaluation

```bash
python quick_evaluate.py
```

**That's it! No arguments, no configuration needed.**

---

## 📊 What It Does

1. ✅ **Auto-loads** trained model from `models/cropshield_cnn.pth`
2. ✅ **Auto-detects** GPU/CPU
3. ✅ **Loads** test dataset (2,240 images)
4. ✅ **Computes** all metrics (accuracy, precision, recall, F1)
5. ✅ **Generates** confusion matrix PNG
6. ✅ **Saves** detailed metrics JSON
7. ✅ **Prints** summary to console

---

## 📈 Sample Output

```
================================================================================
📊 CROPSHIELD AI - MODEL EVALUATION
================================================================================
🖥️  Device: cuda
   GPU: NVIDIA GeForce RTX 4060 Laptop GPU

📥 Loading Model...
✅ Model Loaded:
   Type: custom
   Classes: 22
   Training Epoch: 25
   Best Val Acc: 89.5%

📂 Loading Test Dataset...
✅ Test Dataset: 2240 images

🧪 Evaluating on Test Set...
Testing: 100%|████████████████████| 70/70 [00:20<00:00, 3.5it/s]

📈 Computing Metrics...
📊 Generating Confusion Matrix...
✅ Saved: results\confusion_matrix.png

💾 Saving Metrics...
✅ Saved: results\test_metrics.json

================================================================================
✅ EVALUATION COMPLETE!
================================================================================

📊 RESULTS:
   ✅ Test Accuracy:  91.3%
   Precision (Macro): 89.7%
   Recall (Macro):    88.9%
   F1-Score (Macro):  89.2%

🏆 TOP 5 CLASSES (by F1-Score):
   1. Tomato__healthy: 98.5% F1
   2. Potato__healthy: 97.2% F1
   3. Sugarcane__healthy: 96.8% F1
   4. Tomato__bacterial_spot: 95.1% F1
   5. Potato__early_blight: 94.3% F1

📁 OUTPUT FILES:
   results\confusion_matrix.png
   results\test_metrics.json

================================================================================
✅ Test Accuracy: 91.3%
================================================================================
```

---

## 📁 Output Files

### `results/confusion_matrix.png`
- High-resolution confusion matrix (16x14 inches, 150 DPI)
- Shows all 22 disease classes
- Color-coded heatmap (blue = more predictions)
- Includes test accuracy in title

### `results/test_metrics.json`
Complete metrics including:
```json
{
  "test_accuracy": 91.3,
  "precision_macro": 89.7,
  "recall_macro": 88.9,
  "f1_macro": 89.2,
  "per_class_metrics": {
    "Potato__early_blight": {
      "precision": 93.2,
      "recall": 95.5,
      "f1_score": 94.3,
      "support": 110
    },
    ...
  },
  "confusion_matrix": [[...], ...],
  "total_samples": 2240,
  "num_classes": 22,
  "class_names": [...]
}
```

---

## 🔧 Alternative: Advanced Evaluation

If you need custom options, use `evaluate.py`:

```bash
# With defaults (now works automatically too!)
python evaluate.py

# Custom options
python evaluate.py --checkpoint models/cropshield_cnn.pth --batch_size 64
```

---

## ⏱️ Time Required

- **RTX 4060 GPU:** ~20-30 seconds
- **CPU:** ~5 minutes

---

## 🎯 Use Cases

### After Training
```bash
python train_auto.py --epochs 25
python quick_evaluate.py  # Verify model quality
```

### Before Deployment
```bash
python quick_evaluate.py  # Check if accuracy meets requirements
```

### For Reports
```bash
python quick_evaluate.py  # Generate confusion matrix and metrics
# Share results/confusion_matrix.png and results/test_metrics.json
```

---

## 📊 Metrics Explained

- **Test Accuracy:** Overall percentage of correct predictions
- **Precision:** Of predicted positives, how many were correct
- **Recall:** Of actual positives, how many were found
- **F1-Score:** Harmonic mean of precision and recall

---

## ✅ Quick Commands

```bash
# Run evaluation
python quick_evaluate.py

# View confusion matrix
start results\confusion_matrix.png

# View metrics
cat results\test_metrics.json | python -m json.tool
```

---

**That's it! Fully automatic model evaluation in one command.** 🚀

*Last Updated: 2025-11-11*
