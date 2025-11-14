# Evaluation Quick Reference

## 🚀 Quick Start

```bash
# Evaluate best model
python evaluate.py --model custom --checkpoint checkpoints/best.pth

# Evaluate EfficientNet
python evaluate.py --model efficientnet_b0 --checkpoint checkpoints/best.pth
```

## 📊 Output Files

All files saved to `results/` directory:

| File | Type | Content |
|------|------|---------|
| `confusion_matrix.png` | PNG | Normalized confusion matrix heatmap |
| `class_report.csv` | CSV | Per-class precision, recall, F1, support |
| `test_metrics.json` | JSON | Complete metrics dictionary |

## 🎯 Key Metrics

| Metric | Description | Range | Good Value |
|--------|-------------|-------|------------|
| **Accuracy** | Overall correct predictions | 0-1 | >0.85 |
| **Macro F1** | Avg F1 across classes (unweighted) | 0-1 | >0.80 |
| **Weighted F1** | Avg F1 weighted by support | 0-1 | >0.82 |
| **Precision** | Correct positive predictions | 0-1 | >0.80 |
| **Recall** | Detected actual positives | 0-1 | >0.80 |

## ⚡ Common Commands

### Standard Evaluation
```bash
python evaluate.py --model custom --checkpoint checkpoints/best.pth
```

### Fast Evaluation (Larger Batch)
```bash
python evaluate.py --model custom --checkpoint checkpoints/best.pth --batch_size 128
```

### Custom Results Directory
```bash
python evaluate.py --model custom --checkpoint checkpoints/best.pth --results_dir results/eval_20250109
```

### Evaluate Specific Checkpoint
```bash
python evaluate.py --model custom --checkpoint checkpoints/custom_epoch20_20250109_144530.pth
```

## 🔧 Command-Line Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | **required** | Model type (`custom` or `efficientnet_b0`) |
| `--checkpoint` | **required** | Path to checkpoint file |
| `--batch_size` | `64` | Batch size for evaluation |
| `--num_workers` | `8` | Data loading workers |
| `--amp` | `True` | Mixed precision inference |
| `--results_dir` | `results` | Output directory |

## 📈 Expected Performance

### Custom CNN (4.7M params)
- Accuracy: 75-85%
- Macro F1: 0.72-0.82
- Inference: ~660 img/s (AMP)
- Evaluation time: ~5-10s (2,239 samples)

### EfficientNet-B0 (4.0M params)
- Accuracy: 92-96%
- Macro F1: 0.90-0.94
- Inference: ~520 img/s (AMP)
- Evaluation time: ~8-15s (2,239 samples)

## 🎨 Confusion Matrix Interpretation

**Diagonal (bright blue)**: Correct predictions  
**Off-diagonal (lighter)**: Misclassifications  

Example:
```
            Predicted
          │ A │ B │ C │
      ────┼───┼───┼───┤
    A │ 90│ 5 │ 5 │  ← 90% correct for class A
Actual B │ 3 │ 85│ 12│  ← 85% correct for class B
    C │ 2 │ 8 │ 90│  ← 90% correct for class C
```

## 📊 CSV Report Format

```csv
class_name,precision,recall,f1_score,support
Tomato__healthy,0.9234,0.9156,0.9195,179
Potato__healthy,0.9123,0.9042,0.9082,156
...
```

Sorted by F1 score (best to worst)

## 💾 JSON Report Structure

```json
{
  "overall": {
    "accuracy": 0.8542,
    "macro_f1": 0.8321,
    "weighted_f1": 0.8498,
    "num_samples": 2239,
    "num_classes": 22
  },
  "per_class": {
    "Tomato__healthy": {
      "precision": 0.9234,
      "recall": 0.9156,
      "f1_score": 0.9195,
      "support": 179
    }
  }
}
```

## 🔍 Metric Definitions

**Precision** = True Positives / (True Positives + False Positives)  
*"When model predicts this class, how often is it right?"*

**Recall** = True Positives / (True Positives + False Negatives)  
*"When this class appears, how often does model detect it?"*

**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)  
*"Balanced measure of precision and recall"*

**Support** = Number of actual samples in test set  
*"How many test samples for this class?"*

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `--batch_size 32` |
| Checkpoint not found | Check path with `ls checkpoints/` |
| Model mismatch | Ensure `--model` matches checkpoint architecture |
| Slow evaluation | Increase `--batch_size 128` or `--num_workers 16` |

## 🧪 Test Before Full Evaluation

```bash
# Quick check with smaller batch
python evaluate.py --model custom --checkpoint checkpoints/best.pth --batch_size 32
```

## 📚 Full Documentation

- **Comprehensive Guide**: `EVALUATION_GUIDE.md`
- **Training Guide**: `TRAINING_GUIDE.md`
- **Resume Guide**: `RESUME_TRAINING_EXAMPLE.md`

## 🎯 Workflow

1. **Train model**
   ```bash
   python train.py --model custom --epochs 25 --early_stopping 5
   ```

2. **Evaluate best checkpoint**
   ```bash
   python evaluate.py --model custom --checkpoint checkpoints/best.pth
   ```

3. **Review results**
   - Open `results/confusion_matrix.png`
   - Check `results/class_report.csv` for weak classes
   - Read `results/test_metrics.json` for detailed metrics

4. **Analyze and improve** (if needed)
   - Identify confused classes from confusion matrix
   - Target weak classes (low F1) with more augmentation
   - Fine-tune or retrain with adjusted hyperparameters

5. **Deploy** (if satisfactory)
   - Use `checkpoints/best.pth` for inference
   - Implement GradCAM for explainability
   - Create Streamlit web app

---

**For detailed examples and best practices, see `EVALUATION_GUIDE.md`**
