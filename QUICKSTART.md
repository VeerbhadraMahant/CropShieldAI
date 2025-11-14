# 🚀 CropShield AI - Quick Start Guide

**Get up and running in 5 minutes!**

---

## ⚡ Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (optional but recommended)
- 8 GB RAM minimum

---

## 📦 Setup (One-Time)

```bash
# 1. Clone repository
git clone https://github.com/VeerbhadraMahant/CropShieldAI.git
cd CropShieldAI

# 2. Create virtual environment
conda create -n cropshield python=3.10 -y
conda activate cropshield

# 3. Install PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python quick_verify.py
```

---

## 🎯 Usage

### Option 1: Launch Web App (No Training Required)

```bash
streamlit run app_optimized.py
```

**Then open:** http://localhost:8501

**Features:**
- ✅ Upload images via drag & drop
- ✅ Modern UI with confidence charts
- ⚠️ Predictions require trained model (see Option 2)

---

### Option 2: Train Model + Launch App

```bash
# Step 1: Quick training (10-15 minutes)
python train.py --epochs 2 --batch_size 32

# Step 2: Launch app with trained model
streamlit run app_optimized.py
```

**Now open:** http://localhost:8501

**Full features unlocked:**
- ✅ Disease predictions
- ✅ Confidence scores
- ✅ GradCAM heatmaps
- ✅ Batch processing

---

### Option 3: Command-Line Inference

```bash
# Single image prediction
python predict.py --image path/to/image.jpg

# Batch prediction
python predict.py --batch_dir path/to/folder/

# With GradCAM
python predict.py --image path/to/image.jpg --gradcam
```

---

## 📊 Useful Commands

```bash
# Check system status
python quick_verify.py

# Full system validation
python validate_deployment.py --verbose

# View training progress
tensorboard --logdir experiments/

# Stop Streamlit
# Press Ctrl+C in terminal
```

---

## 🎓 Training Options

### Quick Test (5-10 min)
```bash
python train.py --epochs 2 --batch_size 32
```

### Production Training (4-6 hours)
```bash
python train.py --epochs 50 --batch_size 32 --early_stopping_patience 10
```

### Resume Training
```bash
python train.py --resume checkpoints/last_checkpoint.pth
```

---

## 🔧 Common Issues

### Issue: "CUDA out of memory"
```bash
# Solution: Reduce batch size
python train.py --batch_size 16
```

### Issue: "Module not found"
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "No trained model found"
```bash
# Solution: Train model first
python train.py --epochs 2 --batch_size 32
```

---

## 📂 Project Structure

```
CropShieldAI/
├── app_optimized.py          # Streamlit web app (⭐ START HERE)
├── train.py                  # Model training script
├── predict.py                # Command-line inference
├── quick_verify.py           # System verification
├── requirements.txt          # Dependencies
├── Database_resized/         # Dataset (22,387 images)
├── models/                   # Model architectures
└── utils/                    # Helper functions
```

---

## 🎯 Next Steps

1. **Explore the app:** http://localhost:8501
2. **Try example images:** Use samples from `Database_resized/`
3. **Train your model:** `python train.py --epochs 10`
4. **Read full docs:** Check `PROJECT_STRUCTURE.md`

---

## 💡 Tips

- 🔥 Use GPU for 10x faster training
- 📸 Supported formats: JPG, PNG, JPEG
- 🎨 Adjust augmentation: `--augmentation_mode conservative|moderate|aggressive`
- 💾 Models auto-save to `best_model.pth`
- 📊 View metrics in Streamlit sidebar

---

## 📞 Support

- 📖 Full documentation: `ENVIRONMENT_SETUP.md`, `PROJECT_STRUCTURE.md`
- 🐛 Issues: Check `BUG_FIX_STATUS.md` (20/20 fixed)
- 🚀 Performance: See `OPTIMIZATION_REPORT.md` (19.8x faster)

---

**✨ You're all set! Happy crop disease detection! 🌾**
