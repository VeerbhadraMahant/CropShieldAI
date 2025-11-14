# CropShield AI - QA Inference Test Guide

Complete guide for end-to-end inference pipeline validation.

---

## 📋 Overview

The QA inference test script (`scripts/test_inference.py`) provides comprehensive validation of your trained model's inference pipeline, including:

✅ Model loading and caching validation  
✅ Diverse image sampling (one per class)  
✅ Prediction accuracy measurement  
✅ Inference speed benchmarking  
✅ GradCAM visualization generation  
✅ Detailed test report generation  

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required
pip install torch torchvision pillow numpy

# Optional (for progress bars)
pip install tqdm

# Required files
models/cropshield_cnn.pth      # Trained model
class_to_idx.json              # Class mapping
Database_resized/              # Test images
```

### Basic Usage

```bash
# Test with 5 random images (one per class ideally)
python scripts/test_inference.py

# Test with 10 images
python scripts/test_inference.py --num_samples 10

# Test specific classes
python scripts/test_inference.py --classes Potato__healthy Tomato__healthy Wheat__healthy

# Skip GradCAM generation (faster)
python scripts/test_inference.py --skip_gradcam
```

---

## 📊 Sample Output

### Console Output

```
======================================================================
🧪 CropShield AI - QA Inference Test
======================================================================
   Model: models\cropshield_cnn.pth
   Dataset: Database_resized
   Output: results\test_gradcams
   Device: cuda
======================================================================

📋 Validating setup...
   ✓ Model found: models\cropshield_cnn.pth
   ✓ Class mapping found: class_to_idx.json
   ✓ Dataset found: 22 classes
   ✅ Validation passed!

📦 Loading model...
✅ GPU Inference: NVIDIA GeForce RTX 4060
   ✓ Model loaded successfully
   ✓ Classes: 22
   ✓ Load time: 0.234s

🎲 Sampling 5 test images...
   ✓ Available classes: 22
   ✓ Sampled 5 images

🔮 Running inference on 5 images...

Inference: 100%|████████████████████████████████| 5/5 [00:00<00:00, 12.34 images/s]

🔥 Generating GradCAM visualizations...

   ✓ Target layer: Sequential

GradCAM: 100%|██████████████████████████████████| 5/5 [00:02<00:00,  2.15 images/s]

   ✅ Generated 5/5 GradCAM overlays
   ✓ Saved to: results\test_gradcams

======================================================================
📊 INFERENCE TEST SUMMARY
======================================================================

🎯 Test Results:
   Total images tested: 5
   Successful predictions: 5
   Top-1 accuracy: 80.0%
   Top-3 accuracy: 100.0%

⚡ Inference Performance:
   Average time per image: 0.081s
   Std deviation: 0.012s
   Min time: 0.067s
   Max time: 0.098s
   Total inference time: 0.405s

📋 Detailed Predictions:
======================================================================

✅ Image 1: Potato__early_blight_001.jpg
   True: Potato__early_blight
   Predicted: Potato__early_blight (98.7%)
   Top-3:
      → 1. Potato__early_blight                      98.67%
        2. Potato__late_blight                        1.12%
        3. Tomato__early_blight                       0.15%

⚠️  Image 2: Tomato__healthy_042.jpg
   True: Tomato__healthy
   Predicted: Tomato__leaf_mold (65.3%)
   Top-3:
        1. Tomato__leaf_mold                          65.32%
      → 2. Tomato__healthy                            23.45%
        3. Tomato__septoria_leaf_spot                  8.91%

✅ Image 3: Wheat__brown_rust_015.jpg
   True: Wheat__brown_rust
   Predicted: Wheat__brown_rust (99.2%)
   Top-3:
      → 1. Wheat__brown_rust                          99.23%
        2. Wheat__yellow_rust                          0.56%
        3. Wheat__septoria                             0.18%

✅ Image 4: Sugarcane__red_stripe_008.jpg
   True: Sugarcane__red_stripe
   Predicted: Sugarcane__red_stripe (96.5%)
   Top-3:
      → 1. Sugarcane__red_stripe                      96.54%
        2. Sugarcane__rust                             2.34%
        3. Sugarcane__red_rot                          0.89%

✅ Image 5: Potato__healthy_023.jpg
   True: Potato__healthy
   Predicted: Potato__healthy (99.8%)
   Top-3:
      → 1. Potato__healthy                            99.81%
        2. Tomato__healthy                             0.12%
        3. Wheat__healthy                              0.04%

======================================================================

💾 Report saved to: results\test_inference_report.txt

✅ QA test complete!
   Top-1 accuracy: 80.0%
   Average inference time: 0.081s
   GradCAM overlays: results\test_gradcams
   Report: results\test_inference_report.txt
```

---

## 🎯 Command Line Options

### Model Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | models/cropshield_cnn.pth | Path to trained model |
| `--class_mapping` | class_to_idx.json | Path to class mapping JSON |
| `--dataset` | Database_resized | Dataset directory with test images |

### Test Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--num_samples` | 5 | Number of test images |
| `--classes` | All | Specific classes to test |
| `--top_k` | 3 | Number of top predictions |
| `--seed` | 42 | Random seed for reproducibility |

### Output Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | results/test_gradcams | GradCAM output directory |
| `--skip_gradcam` | False | Skip GradCAM generation |

### Device Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--device` | auto | Device: cpu, cuda, or auto |

---

## 📁 Output Files

### 1. GradCAM Visualizations

**Location:** `results/test_gradcams/`

**Files:**
```
results/test_gradcams/
├── Potato__early_blight_001_gradcam.jpg
├── Tomato__healthy_042_gradcam.jpg
├── Wheat__brown_rust_015_gradcam.jpg
├── Sugarcane__red_stripe_008_gradcam.jpg
└── Potato__healthy_023_gradcam.jpg
```

Each file shows the original image with a heatmap overlay highlighting regions that influenced the prediction.

---

### 2. Test Report

**Location:** `results/test_inference_report.txt`

**Content:**
```
======================================================================
CropShield AI - Inference QA Test Report
======================================================================

Test Date: 2025-11-10 14:30:45
Model: models\cropshield_cnn.pth
Dataset: Database_resized
Device: cuda
Classes: 22

======================================================================
TEST SUMMARY
======================================================================

Total images: 5
Successful predictions: 5
Top-1 accuracy: 80.00%
Top-3 accuracy: 100.00%

Average inference time: 0.081s
Std deviation: 0.012s
Min time: 0.067s
Max time: 0.098s
Total time: 0.405s

======================================================================
DETAILED RESULTS
======================================================================

Image 1: Potato__early_blight_001.jpg
  True class: Potato__early_blight
  Predictions:
    ✓ 1. Potato__early_blight                      98.67%
      2. Potato__late_blight                        1.12%
      3. Tomato__early_blight                       0.15%
  Inference time: 0.078s
  Correct: Yes
  In top-3: Yes

Image 2: Tomato__healthy_042.jpg
  True class: Tomato__healthy
  Predictions:
      1. Tomato__leaf_mold                          65.32%
    ✓ 2. Tomato__healthy                            23.45%
      3. Tomato__septoria_leaf_spot                  8.91%
  Inference time: 0.067s
  Correct: No
  In top-3: Yes

... (remaining images)
```

---

## 🧪 Testing Scenarios

### Scenario 1: Quick Validation (5 images)

**Use Case:** Fast sanity check after training

```bash
python scripts/test_inference.py --num_samples 5 --skip_gradcam
```

**Duration:** ~5 seconds  
**Output:** Console summary + text report

---

### Scenario 2: Comprehensive Test (20 images)

**Use Case:** Thorough validation before deployment

```bash
python scripts/test_inference.py --num_samples 20
```

**Duration:** ~1-2 minutes  
**Output:** Console summary + GradCAM visualizations + report

---

### Scenario 3: Class-Specific Testing

**Use Case:** Validate specific disease classes

```bash
python scripts/test_inference.py \
    --classes Potato__early_blight Potato__late_blight Potato__healthy \
    --num_samples 10
```

**Duration:** ~30 seconds  
**Output:** Focused testing on potato diseases

---

### Scenario 4: Performance Benchmarking

**Use Case:** Measure inference speed on many images

```bash
python scripts/test_inference.py --num_samples 50 --skip_gradcam
```

**Duration:** ~20 seconds  
**Output:** Detailed timing statistics

---

## 📊 Metrics Explained

### Top-1 Accuracy

**Definition:** Percentage of images where the highest confidence prediction matches the true class.

**Formula:** `(Correct predictions) / (Total predictions)`

**Good Range:** 
- 85-95%: Excellent
- 70-85%: Good
- <70%: Needs improvement

---

### Top-3 Accuracy

**Definition:** Percentage of images where the true class appears in the top 3 predictions.

**Formula:** `(True class in top-3) / (Total predictions)`

**Good Range:**
- 95-100%: Excellent
- 85-95%: Good
- <85%: Needs improvement

**Use Case:** More forgiving metric, useful for similar-looking diseases

---

### Inference Time

**Average Time:** Mean inference time per image

**Components:**
- Image loading: ~10ms
- Preprocessing: ~5ms
- Model inference: ~50-80ms (GPU) or ~200-500ms (CPU)
- Postprocessing: ~2ms

**Good Range (GPU):**
- <100ms: Excellent (real-time)
- 100-200ms: Good (near real-time)
- >200ms: Acceptable (batch processing)

---

## 🔍 Interpreting Results

### ✅ Perfect Prediction

```
✅ Image 1: Potato__early_blight_001.jpg
   True: Potato__early_blight
   Predicted: Potato__early_blight (98.7%)
```

**Interpretation:**
- Model correctly identified the disease
- High confidence (98.7%) indicates strong conviction
- Good GradCAM should show disease lesions

---

### ⚠️ Correct in Top-3

```
⚠️  Image 2: Tomato__healthy_042.jpg
   True: Tomato__healthy
   Predicted: Tomato__leaf_mold (65.3%)
   Top-3:
        1. Tomato__leaf_mold          65.32%
      → 2. Tomato__healthy            23.45%
        3. Tomato__septoria_leaf_spot  8.91%
```

**Interpretation:**
- Model confused healthy tomato with leaf mold
- Lower confidence (65%) suggests uncertainty
- True class is in top-3 (recoverable error)
- May indicate similar visual features or poor image quality

---

### ❌ Incorrect Prediction

```
❌ Image 3: Wheat__healthy_005.jpg
   True: Wheat__healthy
   Predicted: Wheat__septoria (72.1%)
   Top-3:
        1. Wheat__septoria           72.12%
        2. Wheat__yellow_rust        18.34%
        3. Wheat__brown_rust          6.78%
```

**Interpretation:**
- Model completely missed the true class
- True class not in top-3 (serious error)
- Possible causes: poor image, ambiguous symptoms, model bias
- Review GradCAM to understand what model is looking at

---

## 🔥 GradCAM Analysis

### What to Look For

**Good GradCAM (Correct Prediction):**
- Heatmap highlights disease lesions
- Focuses on symptomatic areas
- Ignores background and irrelevant regions

**Bad GradCAM (Incorrect Prediction):**
- Focuses on background or edges
- Highlights wrong regions
- Scattered attention without clear pattern

### Example Interpretations

**Potato Early Blight (Correct):**
- ✅ Heatmap focuses on brown lesions with concentric rings
- ✅ Ignores healthy green leaf tissue
- ✅ Strong activation on disease symptoms

**Tomato Healthy (Misclassified as Leaf Mold):**
- ❌ Heatmap focuses on normal leaf veins
- ❌ Misinterprets texture as disease symptoms
- ⚠️ May indicate model needs more healthy examples

---

## 🛠️ Troubleshooting

### Issue 1: "Model not found"

**Error:**
```
❌ Model not found: models\cropshield_cnn.pth
```

**Solutions:**
```bash
# Check if model exists
ls models/

# Train a model first
python train.py --epochs 10

# Or specify different path
python scripts/test_inference.py --model path/to/your/model.pth
```

---

### Issue 2: "Class mapping not found"

**Error:**
```
❌ Class mapping not found: class_to_idx.json
```

**Solutions:**
```bash
# Generate class mapping
python generate_class_mapping.py

# Or specify different path
python scripts/test_inference.py --class_mapping path/to/class_to_idx.json
```

---

### Issue 3: "No test images found"

**Error:**
```
❌ No test images found. Please check dataset directory.
```

**Solutions:**
```bash
# Check dataset directory
ls Database_resized/

# Specify correct path
python scripts/test_inference.py --dataset Database/

# Check if classes match
python scripts/test_inference.py --classes Potato__healthy Tomato__healthy
```

---

### Issue 4: "CUDA out of memory"

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Use CPU
python scripts/test_inference.py --device cpu

# Reduce batch size (for future batch processing)
# Currently processes images one by one, so shouldn't happen

# Close other GPU applications
```

---

### Issue 5: "tqdm not available"

**Warning:**
```
⚠️  tqdm not available. Install with: pip install tqdm
```

**Solutions:**
```bash
# Install tqdm for progress bars
pip install tqdm

# Script will work without it (just no progress bars)
```

---

### Issue 6: GradCAM generation fails

**Error:**
```
⚠️  GradCAM failed for image.jpg: ...
```

**Solutions:**
```bash
# Skip GradCAM (script continues without it)
python scripts/test_inference.py --skip_gradcam

# Check model architecture compatibility
# GradCAM requires convolutional layers

# Reduce number of samples
python scripts/test_inference.py --num_samples 3
```

---

## 🎓 Best Practices

### 1. Regular Testing

**During Training:**
```bash
# After each epoch or significant training milestone
python scripts/test_inference.py --num_samples 5 --skip_gradcam
```

**Before Deployment:**
```bash
# Comprehensive test with GradCAM
python scripts/test_inference.py --num_samples 20
```

---

### 2. Class Balance Validation

```bash
# Test each class equally
python scripts/test_inference.py \
    --classes Potato__healthy Potato__early_blight Potato__late_blight \
              Tomato__healthy Tomato__bacterial_spot Tomato__early_blight \
              Wheat__healthy Wheat__brown_rust Wheat__yellow_rust
```

---

### 3. Performance Benchmarking

```bash
# CPU benchmark
python scripts/test_inference.py --num_samples 50 --device cpu --skip_gradcam

# GPU benchmark
python scripts/test_inference.py --num_samples 50 --device cuda --skip_gradcam
```

---

### 4. Reproducibility

```bash
# Use same seed for consistent results
python scripts/test_inference.py --seed 42

# Different seed for diverse testing
python scripts/test_inference.py --seed 123
```

---

## 📈 Integration with CI/CD

### Automated Testing

**GitHub Actions Example:**
```yaml
name: QA Inference Test

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run QA test
        run: |
          python scripts/test_inference.py \
            --num_samples 10 \
            --skip_gradcam \
            --device cpu
      
      - name: Check accuracy threshold
        run: |
          python -c "
          import json
          with open('results/test_inference_report.txt', 'r') as f:
              report = f.read()
              # Parse accuracy and fail if < 70%
          "
```

---

## 🎯 Success Criteria

### Minimum Requirements

✅ **Top-1 Accuracy:** ≥ 70%  
✅ **Top-3 Accuracy:** ≥ 85%  
✅ **Average Inference Time (GPU):** < 200ms  
✅ **Average Inference Time (CPU):** < 1000ms  
✅ **GradCAM Generation:** ≥ 90% success rate  

### Excellent Performance

✅ **Top-1 Accuracy:** ≥ 85%  
✅ **Top-3 Accuracy:** ≥ 95%  
✅ **Average Inference Time (GPU):** < 100ms  
✅ **Average Inference Time (CPU):** < 500ms  
✅ **GradCAM Generation:** 100% success rate  

---

## 📚 Related Documentation

- **Training Guide:** `TRAINING_GUIDE.md`
- **Inference Guide:** `INFERENCE_GUIDE.md`
- **GradCAM Guide:** `GRADCAM_GUIDE.md`
- **Deployment Guide:** `DEPLOYMENT_GUIDE.md`

---

## ✅ Checklist

Before deployment, verify:

- [ ] QA test runs successfully
- [ ] Top-1 accuracy ≥ 70%
- [ ] Top-3 accuracy ≥ 85%
- [ ] Inference time acceptable for use case
- [ ] GradCAM overlays look reasonable
- [ ] Test report generated successfully
- [ ] Tested on diverse images (all classes)
- [ ] Tested on both CPU and GPU
- [ ] Edge cases handled gracefully

---

**Ready to test?**

```bash
python scripts/test_inference.py
```

Good luck! 🚀
