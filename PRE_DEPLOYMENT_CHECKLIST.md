# Pre-Deployment Checklist ✅

Use this checklist before running validation to ensure everything is ready.

## Environment Setup

### Python Environment
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated (recommended)
  ```bash
  python -m venv venv
  venv\Scripts\activate  # Windows
  source venv/bin/activate  # Linux/Mac
  ```

### Dependencies
- [ ] All dependencies installed
  ```bash
  pip install -r requirements.txt
  ```
- [ ] PyTorch with CUDA support (for GPU)
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  # Should print: True (for GPU)
  ```
- [ ] Streamlit installed
  ```bash
  python -c "import streamlit; print(streamlit.__version__)"
  ```
- [ ] OpenCV installed (for GradCAM)
  ```bash
  python -c "import cv2; print(cv2.__version__)"
  ```

---

## Model Files

### Model Checkpoint
- [ ] Model trained and saved
  ```bash
  # Check if file exists
  Test-Path models\cropshield_cnn.pth  # Windows
  ls models/cropshield_cnn.pth  # Linux/Mac
  ```
- [ ] Model file size > 0 bytes
- [ ] Model checkpoint not corrupted

### Class Mapping
- [ ] Class mapping generated
  ```bash
  # Generate if missing
  python generate_class_mapping.py
  ```
- [ ] File exists: `models/class_to_idx.json`
- [ ] JSON is valid (can be parsed)
- [ ] Number of classes matches your dataset

---

## File Structure

### Required Files
- [ ] `validate_deployment.py` exists
- [ ] `predict.py` exists (for inference)
- [ ] `utils/gradcam.py` exists (for GradCAM)
- [ ] `app_optimized.py` or `app.py` exists (for Streamlit)
- [ ] `models/` directory exists
- [ ] `models/class_to_idx.json` exists
- [ ] `models/cropshield_cnn.pth` exists (or your model checkpoint)

### Optional Files
- [ ] `requirements.txt` (for CI/CD)
- [ ] `.gitignore` (to exclude large files)
- [ ] Documentation files

---

## Pre-Validation Checks

### Quick Test: Python Imports
```bash
# Test if core modules import without errors
python -c "import torch; import torchvision; import streamlit; import cv2; print('✅ All imports successful')"
```

### Quick Test: Model Load
```bash
# Test if model can be loaded
python -c "from predict import load_model_once; model, classes, device = load_model_once('models/cropshield_cnn.pth'); print(f'✅ Model loaded on {device}')"
```

### Quick Test: Device Detection
```bash
# Test GPU availability
python -c "import torch; print('✅ GPU available' if torch.cuda.is_available() else '⚠️ CPU only')"
```

---

## Validation Execution

### Basic Validation
- [ ] Run basic validation
  ```bash
  python validate_deployment.py
  ```
- [ ] All checks passed? If NO, see troubleshooting below

### Verbose Validation (Recommended)
- [ ] Run verbose validation
  ```bash
  python validate_deployment.py --verbose
  ```
- [ ] Review `validation_results.json`
- [ ] Check detailed metrics

### CI/CD Mode
- [ ] Test CI/CD mode (skip Streamlit)
  ```bash
  python validate_deployment.py --skip-streamlit
  ```

---

## Validation Results Review

### Check 1: File System
- [ ] ✅ Model file exists
- [ ] ✅ Class mapping exists
- [ ] File sizes look correct
- [ ] Number of classes matches dataset

### Check 2: Model Loading
- [ ] ✅ Model loads successfully
- [ ] Device correct (GPU preferred, CPU acceptable)
- [ ] Model in eval mode
- [ ] Parameter count reasonable

### Check 3: Dummy Inference (CRITICAL!)
- [ ] ✅ Output shape correct `[1, num_classes]`
- [ ] ✅ Valid probability distribution
- [ ] Inference time reasonable
- [ ] **NO shape mismatch errors**

### Check 4: GradCAM
- [ ] ✅ GradCAM module imports
- [ ] ✅ Target layer found
- [ ] ✅ Heatmap generated
- [ ] Heatmap values in [0, 1] range

### Check 5: Streamlit
- [ ] ✅ Streamlit installed
- [ ] ✅ App file exists
- [ ] ✅ Syntax valid
- [ ] ✅ Can be imported

### Check 6: Performance
- [ ] ✅ Inference time < 200ms (or your target)
- [ ] Performance consistent (low std dev)
- [ ] No performance warnings

---

## Post-Validation Manual Testing

### Test Streamlit App
- [ ] Start app
  ```bash
  streamlit run app_optimized.py
  ```
- [ ] App loads in browser (http://localhost:8501)
- [ ] Upload test image
- [ ] Prediction displays correctly
- [ ] Confidence scores make sense
- [ ] GradCAM visualization displays
- [ ] Performance metrics shown
- [ ] No errors in terminal

### Test Different Images
- [ ] Test with healthy plant image
- [ ] Test with diseased plant image
- [ ] Test with different image formats (JPG, PNG)
- [ ] Test with different image sizes
- [ ] Test with edge cases (very small/large images)

### Test Performance
- [ ] First prediction (model loads)
- [ ] Second prediction (cached, should be faster)
- [ ] Same image again (GradCAM cached, instant)
- [ ] Performance < 200ms on subsequent runs?

---

## Troubleshooting

### Issue: Model Not Found
```
❌ FAILED | Model file exists
```
**Action:**
- [ ] Train model: `python train.py`
- [ ] Or specify correct path: `--model path/to/model.pth`
- [ ] Check working directory

### Issue: Class Mapping Missing
```
❌ FAILED | Class mapping exists
```
**Action:**
- [ ] Generate mapping: `python generate_class_mapping.py`
- [ ] Verify JSON file created
- [ ] Check JSON is valid

### Issue: Wrong Output Shape
```
❌ FAILED | Output shape correct
Got [1, 10], Expected [1, 22]
```
**Action:**
- [ ] Check model architecture (num_classes parameter)
- [ ] Check class_to_idx.json (should have 22 classes)
- [ ] Retrain model if mismatch confirmed
- [ ] Verify dataset structure

### Issue: GradCAM Import Error
```
❌ FAILED | GradCAM module imports
```
**Action:**
- [ ] Install OpenCV: `pip install opencv-python`
- [ ] Test import: `python -c "import cv2"`
- [ ] Check requirements.txt

### Issue: Performance Too Slow
```
❌ FAILED | Average inference time < 200ms
Avg: 450ms
```
**Action:**
- [ ] Check GPU available: `torch.cuda.is_available()`
- [ ] Use optimized app: `app_optimized.py`
- [ ] Enable model caching (already in app_optimized.py)
- [ ] Or adjust target: `--target-time 500`

### Issue: Streamlit Import Error
```
❌ FAILED | Streamlit installed
```
**Action:**
- [ ] Install Streamlit: `pip install streamlit`
- [ ] Test import: `python -c "import streamlit"`
- [ ] Check Python version (3.8+ required)

---

## Deployment Readiness

### All Checks Passed?
```
✅ ALL CHECKS PASSED!
🚀 System is ready for deployment!
```

If YES:
- [ ] Document validation results
- [ ] Commit changes to version control
- [ ] Tag release version
- [ ] Deploy to production
- [ ] Monitor first few predictions
- [ ] Set up automated health checks

If NO:
- [ ] Review failed checks
- [ ] Fix issues listed
- [ ] Re-run validation
- [ ] Repeat until all pass

---

## Production Deployment

### Pre-Deployment
- [ ] All validation checks passed
- [ ] Manual testing completed
- [ ] Performance acceptable
- [ ] Documentation updated
- [ ] Backup plan ready

### Deployment
- [ ] Deploy to production server
- [ ] Start Streamlit app
- [ ] Verify app accessible
- [ ] Test with real users
- [ ] Monitor performance
- [ ] Monitor errors

### Post-Deployment
- [ ] Test production app
- [ ] Verify predictions correct
- [ ] Check performance metrics
- [ ] Set up monitoring alerts
- [ ] Document deployment
- [ ] Plan maintenance schedule

---

## Continuous Validation

### Daily Checks
- [ ] Run validation on production server
  ```bash
  python validate_deployment.py --skip-streamlit
  ```
- [ ] Monitor performance metrics
- [ ] Check error logs
- [ ] Review user feedback

### Weekly Checks
- [ ] Full validation with Streamlit
- [ ] Test with new images
- [ ] Review performance trends
- [ ] Update documentation

### Monthly Checks
- [ ] Comprehensive validation
- [ ] Performance benchmarking
- [ ] Model evaluation
- [ ] System updates

---

## Validation Command Quick Reference

```bash
# Basic validation
python validate_deployment.py

# Verbose with detailed results
python validate_deployment.py --verbose

# CI/CD mode (skip Streamlit)
python validate_deployment.py --skip-streamlit

# Custom model
python validate_deployment.py --model models/best_model.pth

# Custom performance target
python validate_deployment.py --target-time 100

# All options
python validate_deployment.py \
  --model models/custom.pth \
  --app app_optimized.py \
  --target-time 150 \
  --skip-streamlit \
  --verbose
```

---

## Success Criteria Summary

### Minimum Requirements (Must Pass)
- ✅ Files exist and accessible
- ✅ Model loads without errors
- ✅ **Output shape == [1, num_classes]** ← CRITICAL!
- ✅ Inference completes
- ✅ Streamlit can start

### Recommended (Should Pass)
- ✅ GradCAM works
- ✅ Performance < 500ms
- ✅ No import errors
- ✅ Syntax valid

### Optimal (Target)
- ✅ Performance < 200ms
- ✅ GPU acceleration
- ✅ All checks green
- ✅ Zero warnings

---

## Final Check

Before marking complete:
- [ ] All validation checks passed
- [ ] Manual testing successful
- [ ] Performance acceptable
- [ ] Documentation complete
- [ ] Deployment plan ready
- [ ] Team notified
- [ ] Backup strategy in place

**Status:** 
- [ ] ✅ READY TO DEPLOY
- [ ] ⚠️ NEEDS ATTENTION
- [ ] ❌ NOT READY

---

**Date:** _______________
**Validated By:** _______________
**Next Review:** _______________

---

## Notes

```
(Add any notes, issues, or observations here)








```

---

**Remember:** Run validation before EVERY deployment! 🚀
