# Model Evaluation Guide

## Overview

The `evaluate.py` script provides comprehensive evaluation of trained CropShield AI models with detailed metrics, visualizations, and reports.

## Features

✅ **Overall Metrics**
- Accuracy (overall test set performance)
- Macro F1 Score (unweighted average across classes)
- Weighted F1 Score (weighted by class support)

✅ **Per-Class Metrics**
- Precision (true positives / predicted positives)
- Recall (true positives / actual positives)
- F1 Score (harmonic mean of precision and recall)
- Support (number of samples per class)

✅ **Visualizations**
- Confusion matrix (normalized, saved as high-res PNG)
- Top 5 and bottom 5 performing classes

✅ **Export Formats**
- JSON: Complete metrics dictionary
- CSV: Per-class metrics table
- PNG: Confusion matrix heatmap

✅ **Optimizations**
- Mixed-precision inference (AMP) for faster evaluation
- Efficient batch processing
- CPU-based numpy operations to avoid GPU memory accumulation
- Progress bar with tqdm

## Quick Start

### Evaluate Best Model

```bash
# Custom CNN
python evaluate.py --model custom --checkpoint checkpoints/best.pth

# EfficientNet-B0
python evaluate.py --model efficientnet_b0 --checkpoint checkpoints/best.pth
```

### Evaluate Specific Checkpoint

```bash
# Evaluate a specific epoch checkpoint
python evaluate.py --model custom --checkpoint checkpoints/custom_epoch20_20250109_144530.pth
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | **required** | Model architecture (`custom` or `efficientnet_b0`) |
| `--checkpoint` | str | **required** | Path to trained model checkpoint |
| `--batch_size` | int | `64` | Batch size for evaluation |
| `--num_workers` | int | `8` | Number of data loading workers |
| `--amp` | bool | `True` | Use mixed precision inference |
| `--results_dir` | str | `results` | Directory to save results |

## Output Files

After evaluation completes, the following files are created in the `results/` directory:

### 1. `confusion_matrix.png`
- **Type**: PNG image (300 DPI, high resolution)
- **Content**: Normalized confusion matrix heatmap
- **Format**: Percentage values showing prediction distribution
- **Visualization**: Blue heatmap with class labels on both axes

### 2. `class_report.csv`
- **Type**: CSV file
- **Columns**:
  - `class_name`: Name of the disease class
  - `precision`: Precision score (0-1)
  - `recall`: Recall score (0-1)
  - `f1_score`: F1 score (0-1)
  - `support`: Number of test samples
- **Sorted**: By F1 score (descending)

### 3. `test_metrics.json`
- **Type**: JSON file
- **Structure**:
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
      },
      ...
    }
  }
  ```

## Usage Examples

### Example 1: Evaluate After Training

```bash
# Train model
python train.py --model custom --epochs 25 --early_stopping 5 --batch_size 32

# Evaluate best model
python evaluate.py --model custom --checkpoint checkpoints/best.pth
```

### Example 2: Compare Multiple Checkpoints

```bash
# Evaluate epoch 10 checkpoint
python evaluate.py --model custom --checkpoint checkpoints/custom_epoch10_20250109_143022.pth --results_dir results/epoch10

# Evaluate epoch 20 checkpoint
python evaluate.py --model custom --checkpoint checkpoints/custom_epoch20_20250109_144530.pth --results_dir results/epoch20

# Evaluate best checkpoint
python evaluate.py --model custom --checkpoint checkpoints/best.pth --results_dir results/best
```

### Example 3: Larger Batch Size for Speed

```bash
# Use larger batch size for faster evaluation (if memory allows)
python evaluate.py --model custom --checkpoint checkpoints/best.pth --batch_size 128
```

### Example 4: Custom Results Directory

```bash
# Save results to specific directory
python evaluate.py --model custom --checkpoint checkpoints/best.pth --results_dir evaluation_2025_01_09
```

## Interpreting Results

### Overall Metrics

**Accuracy**
- Percentage of correctly classified samples
- Example: `0.8542` = 85.42% accuracy
- Use for overall performance assessment

**Macro F1 Score**
- Unweighted average of F1 scores across all classes
- Treats all classes equally (good for imbalanced datasets)
- Example: `0.8321` = good performance across diverse classes

**Weighted F1 Score**
- Weighted average by class support
- Better reflects performance on common classes
- Example: `0.8498` = overall balanced performance

### Per-Class Metrics

**Precision**
- "When model predicts this class, how often is it correct?"
- High precision = few false positives
- Important when false alarms are costly

**Recall**
- "When this class appears, how often does model detect it?"
- High recall = few false negatives
- Important when missing cases is costly

**F1 Score**
- Harmonic mean of precision and recall
- Balances both metrics
- Best for overall per-class performance

**Support**
- Number of actual samples in test set
- Use to assess metric reliability
- Low support = less reliable metrics

### Confusion Matrix

**Diagonal Values (top-left to bottom-right)**
- Correct predictions for each class
- Higher values = better performance

**Off-Diagonal Values**
- Misclassifications
- Shows which classes are confused with each other
- Example: If `Tomato__early_blight` row has high value in `Tomato__late_blight` column, model confuses these two

## Console Output Example

```
============================================================
🔍 MODEL EVALUATOR INITIALIZED
============================================================
Device:              cuda
Mixed precision:     True
Test samples:        2239
Batch size:          64
Number of classes:   22
Results directory:   results
============================================================

📊 Collecting predictions on test set...
Evaluating: 100%|████████████████| 35/35 [00:03<00:00, 10.23batch/s]
✅ Predictions collected: 2239 samples in 3.4s (658.5 img/s)

📈 Computing metrics...

============================================================
📊 EVALUATION RESULTS
============================================================
Overall Accuracy:    0.8542 (85.42%)
Macro F1:            0.8321
Weighted F1:         0.8498
Test samples:        2239
============================================================

📊 Creating confusion matrix visualization...
✅ Confusion matrix saved: results/confusion_matrix.png

📄 Creating per-class metrics CSV...
✅ Class report saved: results/class_report.csv

============================================================
🏆 TOP 5 CLASSES (by F1 Score)
============================================================
Tomato__healthy                          | F1: 0.9195 | Precision: 0.9234 | Recall: 0.9156
Potato__healthy                          | F1: 0.9082 | Precision: 0.9123 | Recall: 0.9042
Wheat__healthy                           | F1: 0.8967 | Precision: 0.8876 | Recall: 0.9060
Sugarcane__healthy                       | F1: 0.8854 | Precision: 0.8934 | Recall: 0.8775
Tomato__late_blight                      | F1: 0.8723 | Precision: 0.8654 | Recall: 0.8793

============================================================
⚠️  BOTTOM 5 CLASSES (by F1 Score)
============================================================
Tomato__spider_mites_(two_spotted_spider_mite) | F1: 0.7234 | Precision: 0.7012 | Recall: 0.7467
Sugarcane__red_stripe                    | F1: 0.7345 | Precision: 0.7156 | Recall: 0.7545
Wheat__septoria                          | F1: 0.7456 | Precision: 0.7389 | Recall: 0.7524
Tomato__target_spot                      | F1: 0.7567 | Precision: 0.7423 | Recall: 0.7715
Tomato__septoria_leaf_spot              | F1: 0.7678 | Precision: 0.7534 | Recall: 0.7826
============================================================

💾 Saving metrics to JSON...
✅ Metrics saved: results/test_metrics.json

============================================================
✅ EVALUATION COMPLETE
============================================================
Total time:          8.2s
Results saved to:    results
============================================================

============================================================
📊 FINAL SUMMARY
============================================================
Overall Accuracy:    0.8542 (85.42%)
Macro F1 Score:      0.8321
Weighted F1 Score:   0.8498

Results saved to:
  📊 results/confusion_matrix.png
  📄 results/class_report.csv
  💾 results/test_metrics.json
============================================================
```

## Programmatic Usage

You can also use the evaluation components in your own scripts:

```python
from evaluate import ModelEvaluator, load_checkpoint
from models.model_factory import get_model
from fast_dataset import make_loaders
import torch

# Load data
_, _, test_loader, class_names, _ = make_loaders(batch_size=64, augmentation_mode='none')

# Create model
model, _, _, _, device = get_model('custom', num_classes=len(class_names))

# Load checkpoint
model = load_checkpoint('checkpoints/best.pth', model, device)

# Create evaluator
evaluator = ModelEvaluator(
    model=model,
    test_loader=test_loader,
    class_names=class_names,
    device=device,
    use_amp=True,
    results_dir='custom_results'
)

# Run evaluation
metrics = evaluator.evaluate()

# Access metrics
print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
print(f"Macro F1: {metrics['overall']['macro_f1']:.4f}")

# Access per-class metrics
for class_name, class_metrics in metrics['per_class'].items():
    print(f"{class_name}: F1={class_metrics['f1_score']:.4f}")
```

## Performance Tips

### 1. Batch Size Optimization

```bash
# Small batch (memory constrained)
python evaluate.py --model custom --checkpoint checkpoints/best.pth --batch_size 32

# Medium batch (balanced)
python evaluate.py --model custom --checkpoint checkpoints/best.pth --batch_size 64

# Large batch (fast evaluation)
python evaluate.py --model custom --checkpoint checkpoints/best.pth --batch_size 128
```

### 2. Mixed Precision Inference

Mixed precision (AMP) is enabled by default for 2x speedup on GPU:
- Enabled: ~660 img/s
- Disabled: ~330 img/s

To disable (for debugging):
```bash
python evaluate.py --model custom --checkpoint checkpoints/best.pth --amp False
```

### 3. Data Loading Workers

Adjust based on your CPU:
```bash
# More workers (faster CPU)
python evaluate.py --model custom --checkpoint checkpoints/best.pth --num_workers 16

# Fewer workers (slower CPU or debugging)
python evaluate.py --model custom --checkpoint checkpoints/best.pth --num_workers 4
```

## Common Issues & Solutions

### Issue 1: Out of Memory

**Symptom**: `CUDA out of memory` error

**Solution**: Reduce batch size
```bash
python evaluate.py --model custom --checkpoint checkpoints/best.pth --batch_size 32
```

### Issue 2: Checkpoint Not Found

**Symptom**: `FileNotFoundError: Checkpoint not found`

**Solution**: Verify checkpoint path
```bash
# List available checkpoints
ls checkpoints/

# Use correct path
python evaluate.py --model custom --checkpoint checkpoints/best.pth
```

### Issue 3: Model Architecture Mismatch

**Symptom**: `RuntimeError: Error loading state_dict`

**Solution**: Ensure `--model` matches the checkpoint's architecture
```bash
# If checkpoint is from custom CNN
python evaluate.py --model custom --checkpoint checkpoints/custom_best.pth

# If checkpoint is from EfficientNet
python evaluate.py --model efficientnet_b0 --checkpoint checkpoints/efficientnet_best.pth
```

## Best Practices

1. **Always evaluate on best checkpoint** for final results
   ```bash
   python evaluate.py --model custom --checkpoint checkpoints/best.pth
   ```

2. **Use descriptive results directories** for multiple evaluations
   ```bash
   python evaluate.py --model custom --checkpoint checkpoints/best.pth --results_dir results/final_eval
   ```

3. **Check per-class metrics** to identify weak classes
   ```bash
   # After evaluation, review CSV for low F1 scores
   cat results/class_report.csv | sort -t',' -k4 -n | head -5
   ```

4. **Compare confusion matrix** to understand misclassifications
   - Open `results/confusion_matrix.png`
   - Look for bright off-diagonal cells (common confusions)

5. **Document evaluation results** in your project
   ```bash
   # Save terminal output to file
   python evaluate.py --model custom --checkpoint checkpoints/best.pth | tee evaluation_log.txt
   ```

## Next Steps After Evaluation

1. **Analyze Results**
   - Review confusion matrix for common misclassifications
   - Identify weak classes (low F1 scores)
   - Compare to validation performance (check for overfitting)

2. **Improve Model** (if needed)
   - Augment data for weak classes
   - Adjust class weights for imbalanced classes
   - Fine-tune with different hyperparameters
   - Try ensemble methods

3. **Deploy Model**
   - If metrics are satisfactory, proceed to deployment
   - Use `checkpoints/best.pth` for inference
   - Implement GradCAM for explainability
   - Create Streamlit web app

4. **Generate Report**
   - Combine confusion matrix, metrics, and visualizations
   - Document model performance
   - Share with stakeholders

## Summary

The evaluation script provides:
- ✅ **Comprehensive metrics** (accuracy, F1, precision, recall, support)
- ✅ **Visual confusion matrix** (normalized heatmap)
- ✅ **CSV export** (per-class metrics)
- ✅ **JSON export** (complete metrics)
- ✅ **Efficient inference** (AMP, batch processing)
- ✅ **Memory-safe** (CPU-based aggregation)
- ✅ **Easy to use** (simple CLI interface)

Run evaluation after training to assess model performance and identify areas for improvement!
