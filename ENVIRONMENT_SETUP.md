# 🔧 CropShield AI - Environment Setup Guide
**Complete Installation Instructions for Local Development & Testing**

---

## 📋 Prerequisites

### System Requirements
- **OS:** Windows 10/11, Ubuntu 20.04+, or macOS 10.15+
- **Python:** 3.10 or 3.11 (recommended: 3.10)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Storage:** 25 GB free space (5 GB for dependencies + 20 GB for dataset)
- **GPU (Optional):** NVIDIA GPU with CUDA 11.8+ for faster training

### Software Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git (for cloning repository)
- CUDA Toolkit 11.8 or 12.1 (if using GPU)

---

## 🚀 Quick Setup (Recommended)

### Option 1: Conda Environment (Recommended)

```bash
# 1. Create new conda environment with Python 3.10
conda create -n cropshield python=3.10 -y

# 2. Activate environment
conda activate cropshield

# 3. Install PyTorch with CUDA support (GPU)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (no GPU):
pip install torch torchvision torchaudio

# 4. Install all other dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Option 2: Python Virtual Environment (venv)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install PyTorch (GPU or CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install all other dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 📦 Step-by-Step Installation Guide

### Step 1: Clone Repository
```bash
# Clone the repository
git clone https://github.com/VeerbhadraMahant/CropShieldAI.git
cd CropShieldAI

# Or if you have the project locally
cd /path/to/CropShieldAI
```

### Step 2: Create Environment
```bash
# Using Conda (Recommended)
conda create -n cropshield python=3.10 -y
conda activate cropshield

# OR using venv
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### Step 3: Install PyTorch

**Check Your CUDA Version First:**
```bash
# Windows
nvidia-smi

# Linux
nvidia-smi

# Output will show CUDA Version (e.g., CUDA Version: 11.8)
```

**Install PyTorch Based on CUDA Version:**

```bash
# CUDA 11.8 (Most Common)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (Newer GPUs)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# CPU Only (No GPU)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

### Step 4: Install Core Dependencies
```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt

# If you encounter conflicts, install minimal requirements:
pip install streamlit pillow numpy pandas scikit-learn matplotlib seaborn opencv-python tqdm
```

### Step 5: Install Optional Dependencies

**For Better Performance (Pillow-SIMD):**
```bash
# Uninstall regular Pillow first
pip uninstall pillow -y

# Install Pillow-SIMD (SIMD optimized, 4-6x faster)
pip install pillow-simd
```

**For ONNX Export:**
```bash
pip install onnx onnxruntime
# OR for GPU inference:
pip install onnx onnxruntime-gpu
```

**For Advanced Augmentation:**
```bash
pip install albumentations
```

**For Development/Testing:**
```bash
pip install pytest pytest-cov black flake8 mypy jupyter
```

### Step 6: Verify Installation

**Create Verification Script:**
```bash
python -c "
import sys
print('=== CropShield AI Environment Verification ===\n')

# Check Python version
print(f'Python Version: {sys.version}')

# Check core packages
try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'✓ CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  CUDA Version: {torch.version.cuda}')
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('✗ PyTorch not installed')

try:
    import torchvision
    print(f'✓ torchvision: {torchvision.__version__}')
except ImportError:
    print('✗ torchvision not installed')

try:
    import streamlit
    print(f'✓ Streamlit: {streamlit.__version__}')
except ImportError:
    print('✗ Streamlit not installed')

try:
    import cv2
    print(f'✓ OpenCV: {cv2.__version__}')
except ImportError:
    print('✗ OpenCV not installed')

try:
    import numpy
    print(f'✓ NumPy: {numpy.__version__}')
except ImportError:
    print('✗ NumPy not installed')

try:
    import pandas
    print(f'✓ Pandas: {pandas.__version__}')
except ImportError:
    print('✗ Pandas not installed')

try:
    import sklearn
    print(f'✓ scikit-learn: {sklearn.__version__}')
except ImportError:
    print('✗ scikit-learn not installed')

try:
    import matplotlib
    print(f'✓ Matplotlib: {matplotlib.__version__}')
except ImportError:
    print('✗ Matplotlib not installed')

print('\n=== Verification Complete ===')
"
```

---

## 🎯 Platform-Specific Instructions

### Windows 10/11

```powershell
# 1. Open Anaconda Prompt or PowerShell as Administrator

# 2. Create environment
conda create -n cropshield python=3.10 -y
conda activate cropshield

# 3. Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Test
python validate_deployment.py --skip-streamlit
```

**Common Windows Issues:**

**Issue 1: Long Path Names**
```powershell
# Enable long paths in Windows
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Issue 2: Visual C++ Build Tools**
```powershell
# Download and install Visual C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Ubuntu/Linux

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install system dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
sudo apt install -y libjpeg-dev zlib1g-dev

# 3. Create environment
python3.10 -m venv venv
source venv/bin/activate

# 4. Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install dependencies
pip install -r requirements.txt

# 6. Test
python validate_deployment.py --skip-streamlit
```

### macOS (Apple Silicon M1/M2)

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python 3.10
brew install python@3.10

# 3. Create environment
python3.10 -m venv venv
source venv/bin/activate

# 4. Install PyTorch (MPS acceleration for Apple Silicon)
pip install torch torchvision torchaudio

# 5. Install dependencies
pip install -r requirements.txt

# 6. Test
python validate_deployment.py --skip-streamlit
```

---

## 🔍 Dependency Breakdown

### Essential Dependencies (Required)
```txt
torch>=2.0.0                # Deep learning framework
torchvision>=0.15.0         # Image models and transforms
Pillow>=10.0.0              # Image processing
opencv-python>=4.8.0        # GradCAM visualization
numpy>=1.24.0               # Numerical computing
pandas>=2.0.0               # Data analysis
scikit-learn>=1.3.0         # ML utilities
matplotlib>=3.7.0           # Plotting
seaborn>=0.12.0            # Statistical plots
streamlit>=1.28.0          # Web app
tqdm>=4.66.0               # Progress bars
```

### Optional Dependencies (Performance)
```txt
pillow-simd>=9.0.0         # Faster image processing
albumentations>=1.3.0      # Advanced augmentation
```

### Optional Dependencies (Export)
```txt
onnx>=1.14.0               # Model export
onnxruntime>=1.15.0        # ONNX inference
```

### Development Dependencies (Testing)
```txt
pytest>=7.4.0              # Testing
black>=23.7.0              # Formatting
flake8>=6.1.0              # Linting
jupyter>=1.0.0             # Notebooks
```

---

## ⚡ Performance Optimization

### 1. Install Pillow-SIMD (4-6x Faster Image Processing)
```bash
pip uninstall pillow -y
pip install pillow-simd
```

### 2. Enable CUDA (GPU Acceleration)
```bash
# Verify CUDA is working
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Use Mixed Precision Training (Automatic in train.py)
```python
# Already implemented in train.py:
from torch.cuda.amp import autocast, GradScaler
```

### 4. Optimize DataLoader Workers
```bash
# Edit train.py or fast_dataset.py:
# num_workers = os.cpu_count() - 1  # Use all CPUs except 1
```

---

## 🧪 Testing Your Setup

### Test 1: Environment Verification
```bash
python -c "
import torch
import torchvision
import streamlit
import cv2
import numpy as np
import pandas as pd
print('✅ All core packages installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Test 2: Model Loading Test
```bash
python -c "
from models.model_factory import create_model
model = create_model('custom', num_classes=38)
print(f'✅ Model created: {model.__class__.__name__}')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

### Test 3: Data Loading Test
```bash
python -c "
from fast_dataset import create_fast_dataloaders
loaders = create_fast_dataloaders(
    data_dir='Database_resized',
    batch_size=8,
    num_workers=0
)
print('✅ DataLoader created successfully!')
print(f'Train batches: {len(loaders[\"train\"])}')
"
```

### Test 4: Transform Pipeline Test
```bash
python demo_transforms.py
```

### Test 5: Full Deployment Validation
```bash
# This runs all 6 validation checks
python validate_deployment.py --verbose
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
python train.py --batch_size 16  # Instead of 32

# Solution 2: Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Solution 3: Use gradient accumulation
python train.py --batch_size 16 --gradient_accumulation_steps 2
```

### Issue 2: Import Errors
```bash
# Solution: Ensure you're in the project directory
cd CropShieldAI
python -c "import sys; print(sys.path)"

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
$env:PYTHONPATH += ";$(Get-Location)"     # Windows PowerShell
```

### Issue 3: Pillow-SIMD Installation Fails
```bash
# Fallback to regular Pillow
pip uninstall pillow-simd -y
pip install Pillow
```

### Issue 4: OpenCV Import Error
```bash
# Uninstall all OpenCV packages
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y

# Reinstall clean version
pip install opencv-python==4.8.1.78
```

### Issue 5: Streamlit Not Found
```bash
# Reinstall Streamlit
pip uninstall streamlit -y
pip install streamlit==1.28.1
```

---

## 📊 Environment Size Estimates

| Configuration | Size | Components |
|---------------|------|------------|
| **Minimal** | ~2.5 GB | PyTorch CPU + Streamlit + essentials |
| **Standard** | ~4 GB | PyTorch GPU + all required deps |
| **Full Development** | ~6 GB | All deps + dev tools + Jupyter |
| **With Dataset** | ~24 GB | Full + 20 GB dataset |

---

## 🔄 Updating Dependencies

### Update All Packages
```bash
# Update pip first
python -m pip install --upgrade pip

# Update all packages to latest compatible versions
pip install --upgrade -r requirements.txt

# Or update specific packages
pip install --upgrade torch torchvision streamlit
```

### Check for Outdated Packages
```bash
pip list --outdated
```

### Create New Requirements File
```bash
# Export current environment
pip freeze > requirements_frozen.txt

# Or export only what's needed
pipreqs . --force
```

---

## 🎓 Environment Management Tips

### Save Environment State
```bash
# Conda
conda env export > environment.yml

# Pip
pip freeze > requirements_frozen.txt
```

### Recreate Environment
```bash
# From conda YAML
conda env create -f environment.yml

# From pip requirements
pip install -r requirements_frozen.txt
```

### Delete Environment
```bash
# Conda
conda deactivate
conda env remove -n cropshield

# Venv
deactivate  # First deactivate
rm -rf venv  # Then delete folder
```

### List All Environments
```bash
# Conda
conda env list

# Venv (no built-in command, check directories manually)
```

---

## ✅ Post-Installation Checklist

After setup, verify everything works:

- [ ] Python 3.10 or 3.11 installed
- [ ] Virtual environment created and activated
- [ ] PyTorch installed (check with `torch.__version__`)
- [ ] CUDA available if using GPU (`torch.cuda.is_available()`)
- [ ] All core packages imported without errors
- [ ] `validate_deployment.py --skip-streamlit` passes
- [ ] Can create model: `from models.model_factory import create_model`
- [ ] Can load data: `from fast_dataset import create_fast_dataloaders`
- [ ] Streamlit runs: `streamlit run app_optimized.py`

---

## 🚀 Next Steps After Setup

1. **Prepare Data:**
   ```bash
   python scripts/resize_images.py
   python generate_class_mapping.py
   ```

2. **Test Training (Quick):**
   ```bash
   python train.py --epochs 2 --batch_size 8
   ```

3. **Run Validation:**
   ```bash
   python validate_deployment.py --verbose
   ```

4. **Launch App:**
   ```bash
   streamlit run app_optimized.py
   ```

---

## 📞 Support Resources

### Documentation
- `PROJECT_STRUCTURE.md` - Complete project overview
- `TRAINING_GUIDE.md` - Training instructions
- `DEPLOYMENT_GUIDE.md` - Deployment guide
- `requirements.txt` - All dependencies

### Common Commands
```bash
# Check environment
conda info --envs

# Check installed packages
pip list

# Check package details
pip show torch

# Search for package
pip search streamlit
```

---

**✅ Your environment is now ready for CropShield AI development and testing!**

**Happy Coding! 🌾✨**
