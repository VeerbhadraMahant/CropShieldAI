## 🚀 Quick Start

### Setup (5 minutes)

```bash
# 1. Create environment
conda create -n cropshield python=3.10 -y && conda activate cropshield

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Verify installation
python quick_verify.py
```

### Launch App

```bash
streamlit run app_optimized.py
```

**Open:** http://localhost:8501

### Train Model (Optional)

```bash
# Quick training (10-15 min)
python train.py --epochs 2 --batch_size 32

# Production training (4-6 hours)
python train.py --epochs 50 --batch_size 32
```

### Command-Line Inference

```bash
python predict.py --image path/to/image.jpg
```

---

## ✨ Features

- 🎨 **Modern Web UI** - Drag & drop image upload
- 🔥 **GradCAM Visualization** - See what the model sees
- 📈 **Confidence Charts** - Interactive Plotly visualizations
- ⚡ **Optimized Performance** - 19.8x faster than baseline
- 🐛 **Production Ready** - All 20 bugs fixed
- 🌾 **22 Crop Diseases** - Potato, Tomato, Wheat, Sugarcane

---

## 🎯 System Requirements

- Python 3.10+
- 8 GB RAM (16 GB recommended)
- NVIDIA GPU with CUDA 11.8+ (optional but 10x faster)
- 25 GB storage (dataset + models)

---

## 📚 Documentation

- [Quick Start](QUICKSTART.md) - Get running in 5 minutes
- [Environment Setup](ENVIRONMENT_SETUP.md) - Detailed installation
- [Project Structure](PROJECT_STRUCTURE.md) - Directory layout & file connections
- [Training Guide](TRAINING_GUIDE.md) - Model training details
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment

---

## 🔧 Common Commands

```bash
# Verify system
python quick_verify.py

# Full validation
python validate_deployment.py --verbose

# Train with custom settings
python train.py --epochs 10 --batch_size 32 --learning_rate 0.001

# Inference with GradCAM
python predict.py --image sample.jpg --gradcam

# Stop Streamlit
# Press Ctrl+C in terminal
```

---

## 💡 Tips

- 🔥 **GPU:** Training is 10x faster with CUDA-enabled GPU
- 📸 **Formats:** Supports JPG, PNG, JPEG
- 🎨 **Augmentation:** Use `--augmentation_mode conservative|moderate|aggressive`
- 💾 **Auto-save:** Best model saves to `best_model.pth`
- 📊 **Monitoring:** Check Streamlit sidebar for real-time metrics

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size: `--batch_size 16` |
| Module not found | Reinstall: `pip install -r requirements.txt` |
| No model found | Train first: `python train.py --epochs 2` |
| Slow training | Enable GPU or reduce `--num_workers` |

---

**✨ Ready to detect crop diseases! 🌾**

For detailed guides, see [QUICKSTART.md](QUICKSTART.md) or [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
