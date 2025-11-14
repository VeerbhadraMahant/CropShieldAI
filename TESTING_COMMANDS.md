# 🚀 CropShield AI - Complete Testing & Launch Commands

**Complete command sequence for local testing and app launch**

---

## 📋 Phase 1: System Verification (2 minutes)

```bash
# Navigate to project directory
cd D:\Work\ML\IGC\CropShieldAI\CropShieldAI

# Activate environment
conda activate cropshield

# Quick system check
python quick_verify.py
```

**Expected Output:**
- ✅ All 6 checks passed
- ✅ GPU detected
- ✅ 22,387 images loaded
- ✅ Model factory working

---

## 📋 Phase 2A: Launch App WITHOUT Training (30 seconds)

```bash
# Launch Streamlit (demo mode - no predictions yet)
streamlit run app_optimized.py
```

**Then open:** http://localhost:8501

**Available Features:**
- ✅ Modern UI
- ✅ Image upload
- ⚠️ Predictions (requires training first)

**To stop:** Press `Ctrl+C`

---

## 📋 Phase 2B: Launch App WITH Training (15-20 minutes)

### Step 1: Quick Training (10-15 minutes)

```bash
# Quick 2-epoch training for testing
python train.py --epochs 2 --batch_size 32 --learning_rate 0.001
```

**Expected Output:**
```
Epoch 1/2: 100%|████████| Loss: 2.345 | Acc: 45.2%
Epoch 2/2: 100%|████████| Loss: 1.892 | Acc: 58.7%
✅ Training complete! Best model saved to: best_model.pth
```

### Step 2: Launch App with Trained Model

```bash
# Launch Streamlit with predictions enabled
streamlit run app_optimized.py
```

**Then open:** http://localhost:8501

**Full Features Unlocked:**
- ✅ Disease predictions
- ✅ Confidence scores  
- ✅ GradCAM heatmaps
- ✅ Batch processing

---

## 📋 Phase 3: Command-Line Testing (1 minute)

### Single Image Prediction

```bash
# Test prediction on single image
python predict.py --image Database_resized/Tomato__healthy/sample.jpg
```

**Expected Output:**
```
Predicted: Tomato__healthy
Confidence: 92.45%
```

### Batch Prediction

```bash
# Test batch prediction
python predict.py --batch_dir Database_resized/Potato__early_blight/ --top_k 3
```

### With GradCAM

```bash
# Generate prediction with heatmap
python predict.py --image Database_resized/Sugarcane__rust/sample.jpg --gradcam --save_dir results/
```

---

## 📋 Phase 4: Full Deployment Validation (3 minutes)

```bash
# Run comprehensive validation
python validate_deployment.py --verbose
```

**Expected Output:**
```
Check 1: Dependencies ✅
Check 2: File Structure ✅
Check 3: Model Loading ✅
Check 4: Data Pipeline ✅
Check 5: Inference ✅
Check 6: GradCAM ✅

Result: 6/6 PASSED - System Ready for Production
```

---

## 🎯 Complete Testing Sequence (All-in-One)

Copy and paste this entire block for complete testing:

```bash
# ========== COMPLETE TESTING SEQUENCE ==========

# 1. Verify system
echo "Step 1: System Verification..."
python quick_verify.py

# 2. Quick training (optional - comment out if skipping)
echo "Step 2: Quick Training..."
python train.py --epochs 2 --batch_size 32

# 3. Test command-line inference
echo "Step 3: Testing Inference..."
python predict.py --image Database_resized/Tomato__healthy/0a62dd4e-68e5-4e74-9af4-d8ad92b26c7f___RS_HL 1743.jpg

# 4. Full validation
echo "Step 4: Full Validation..."
python validate_deployment.py --verbose

# 5. Launch Streamlit
echo "Step 5: Launching Streamlit..."
echo "Open http://localhost:8501 in your browser"
streamlit run app_optimized.py
```

---

## 🎯 Quick Launch (No Training - 30 seconds)

```bash
# Fastest way to see the app
cd D:\Work\ML\IGC\CropShieldAI\CropShieldAI
conda activate cropshield
streamlit run app_optimized.py
```

**Open:** http://localhost:8501

---

## 🎯 Production Launch (With Training - 15 minutes)

```bash
# Complete setup with training
cd D:\Work\ML\IGC\CropShieldAI\CropShieldAI
conda activate cropshield

# Train model
python train.py --epochs 2 --batch_size 32

# Launch app
streamlit run app_optimized.py
```

**Open:** http://localhost:8501

---

## 🔧 Monitoring Commands (Use in Separate Terminal)

### Monitor GPU Usage
```bash
# Watch GPU in real-time
nvidia-smi -l 1
```

### Monitor Training Progress
```bash
# View training logs
tensorboard --logdir experiments/
```
**Open:** http://localhost:6006

### Check System Resources
```bash
# View system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%'); print(f'RAM: {psutil.virtual_memory().percent}%')"
```

---

## 🎮 Advanced Testing Commands

### Test Different Augmentation Levels

```bash
# Conservative (minimal augmentation)
python train.py --epochs 2 --augmentation_mode conservative

# Moderate (balanced augmentation)  
python train.py --epochs 2 --augmentation_mode moderate

# Aggressive (heavy augmentation)
python train.py --epochs 2 --augmentation_mode aggressive
```

### Test Different Batch Sizes

```bash
# Small batch (4 GB VRAM)
python train.py --epochs 2 --batch_size 16

# Medium batch (6 GB VRAM)
python train.py --epochs 2 --batch_size 32

# Large batch (8+ GB VRAM)
python train.py --epochs 2 --batch_size 64
```

### Test Model Architectures

```bash
# Custom CNN (default)
python train.py --epochs 2 --model_type custom

# EfficientNet-B0
python train.py --epochs 2 --model_type efficientnet

# ResNet-18
python train.py --epochs 2 --model_type resnet18
```

---

## 📊 Expected Results

### After Quick Training (2 epochs):
- **Training Accuracy:** 45-65%
- **Validation Accuracy:** 40-60%
- **Training Time:** 10-15 minutes (GPU)
- **Model Size:** ~18 MB

### After Full Training (50 epochs):
- **Training Accuracy:** 85-95%
- **Validation Accuracy:** 80-90%
- **Training Time:** 4-6 hours (GPU)
- **Model Size:** ~18 MB

---

## 🐛 Troubleshooting Commands

### Check CUDA
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Check Dependencies
```bash
pip list | findstr "torch streamlit pillow opencv"
```

### Clear GPU Memory
```bash
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

### Reinstall Dependencies
```bash
pip install --force-reinstall -r requirements.txt
```

---

## 🎯 One-Command Launch Options

### Option 1: Instant Launch (No Training)
```bash
conda activate cropshield && streamlit run app_optimized.py
```

### Option 2: Train + Launch
```bash
conda activate cropshield && python train.py --epochs 2 --batch_size 32 && streamlit run app_optimized.py
```

### Option 3: Verify + Launch
```bash
conda activate cropshield && python quick_verify.py && streamlit run app_optimized.py
```

---

## 📞 Useful Keyboard Shortcuts

**While Streamlit is Running:**
- `Ctrl+C` - Stop server
- `R` in browser - Reload app
- `C` in browser - Clear cache
- `Ctrl+Shift+R` - Hard refresh browser

**While Training:**
- `Ctrl+C` - Stop training (saves checkpoint)

---

## ✅ Success Indicators

**System Verification:**
```
✅ All 6 checks passed
✅ GPU detected
✅ 22,387 images loaded
```

**Training:**
```
✅ Epoch progress bars showing
✅ Loss decreasing
✅ Accuracy increasing
✅ Best model saved
```

**Streamlit:**
```
✅ "You can now view your Streamlit app"
✅ Local URL: http://localhost:8501
✅ App opens in browser
```

**Inference:**
```
✅ Predicted class displayed
✅ Confidence score shown
✅ GradCAM heatmap generated (if enabled)
```

---

**🎉 You're all set! Copy any command block above and paste into your terminal!**
