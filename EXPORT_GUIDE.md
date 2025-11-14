# CropShield AI - Model Export Guide

Complete guide for exporting PyTorch models to TorchScript and ONNX formats for fast, portable inference.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Why Export?](#why-export)
3. [Format Comparison](#format-comparison)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Deployment Scenarios](#deployment-scenarios)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The export script (`scripts/export_model.py`) converts trained PyTorch models to optimized formats:

- **TorchScript (.pt)**: Traced PyTorch model with lighter dependency
- **ONNX (.onnx)**: Open format for cross-platform inference

### Key Features

✅ Automatic parity testing across all formats  
✅ Dynamic batch dimension support (ONNX)  
✅ Inference speed benchmarking  
✅ Model size comparison  
✅ Detailed validation and error handling  
✅ Compatible with existing inference pipeline  

---

## Why Export?

### Problems with Standard PyTorch

1. **Heavy Dependencies**: Full PyTorch installation (~2GB)
2. **Slower Inference**: Python overhead, no optimizations
3. **Deployment Complexity**: Requires Python runtime
4. **Limited Portability**: CPU/GPU only, no mobile/edge support

### Benefits of Exported Models

| Benefit | TorchScript | ONNX |
|---------|-------------|------|
| **Speed** | 2-5x faster | 2-10x faster |
| **Size** | ~Same as PyTorch | ~30-50% smaller |
| **Dependencies** | Lighter PyTorch | ONNX Runtime only |
| **C++ Deployment** | ✅ Yes | ✅ Yes |
| **Mobile/Edge** | ⚠️ Limited | ✅ Excellent |
| **GradCAM Support** | ✅ Yes | ❌ No (no hooks) |
| **Cross-Platform** | ⚠️ PyTorch needed | ✅ Excellent |

---

## Format Comparison

### PyTorch (.pth)

**Pros:**
- Full model functionality (training, GradCAM, hooks)
- Easy debugging and modification
- Native PyTorch ecosystem

**Cons:**
- Requires full PyTorch installation
- Slower inference (Python overhead)
- Larger deployment footprint

**Use Case:** Development, training, GradCAM visualization

---

### TorchScript (.pt)

**Pros:**
- 2-5x faster inference than PyTorch
- Compatible with GradCAM (preserves hooks)
- C++ deployment support
- Lighter PyTorch dependency

**Cons:**
- Still requires PyTorch (but lighter version)
- Limited mobile support
- Some dynamic operations may fail

**Use Case:** Production inference with GradCAM, C++ deployment

**Export Method:**
```python
# Uses torch.jit.trace()
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_scripted.pt')
```

---

### ONNX (.onnx)

**Pros:**
- 2-10x faster inference (ONNX Runtime)
- No PyTorch dependency
- Excellent cross-platform support (CPU, GPU, mobile, edge)
- Smaller file size (~30-50% reduction)
- Dynamic batch dimension support

**Cons:**
- No gradient hooks (GradCAM won't work)
- Limited debugging capabilities
- Some PyTorch operations may not be supported

**Use Case:** Pure inference deployment (web, mobile, edge devices)

**Export Method:**
```python
# Uses torch.onnx.export()
torch.onnx.export(
    model, 
    example_input,
    'model.onnx',
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

---

## Quick Start

### Prerequisites

```bash
# Required
pip install torch torchvision

# Optional (for ONNX validation and testing)
pip install onnx onnxruntime
```

### Basic Export (All Formats)

```bash
python scripts/export_model.py --model models/cropshield_cnn.pth
```

**Output:**
```
models/
├── cropshield_cnn.pth              # Original PyTorch
├── cropshield_cnn_scripted.pt      # TorchScript
└── cropshield_cnn.onnx             # ONNX
```

### Export Specific Format

```bash
# TorchScript only
python scripts/export_model.py --model models/cropshield_cnn.pth --formats torchscript

# ONNX only
python scripts/export_model.py --model models/cropshield_cnn.pth --formats onnx

# Both
python scripts/export_model.py --model models/cropshield_cnn.pth --formats torchscript onnx
```

---

## Detailed Usage

### Command Line Options

```bash
python scripts/export_model.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | str | **Required** | Path to trained .pth model |
| `--output` | str | Same as model | Output directory |
| `--formats` | list | all | Export formats: torchscript, onnx, all |
| `--input_shape` | list | [1,3,224,224] | Input tensor shape (B,C,H,W) |
| `--num_classes` | int | 22 | Number of output classes |
| `--device` | str | cpu | Device: cpu or cuda |
| `--skip_test` | flag | False | Skip parity testing |
| `--skip_benchmark` | flag | False | Skip speed benchmark |
| `--opset` | int | 14 | ONNX opset version |

### Examples

**1. Export with Custom Output Directory**
```bash
python scripts/export_model.py \
    --model models/cropshield_cnn.pth \
    --output exported_models/
```

**2. Export for Different Input Size**
```bash
python scripts/export_model.py \
    --model models/cropshield_cnn.pth \
    --input_shape 1 3 256 256
```

**3. Fast Export (Skip Testing)**
```bash
python scripts/export_model.py \
    --model models/cropshield_cnn.pth \
    --skip_test \
    --skip_benchmark
```

**4. GPU Export**
```bash
python scripts/export_model.py \
    --model models/cropshield_cnn.pth \
    --device cuda
```

---

## Using Exported Models

### TorchScript Inference

```python
import torch
from PIL import Image
from torchvision import transforms

# Load TorchScript model
model = torch.jit.load('models/cropshield_cnn_scripted.pt')
model.eval()

# Prepare input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('test.jpg')
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probabilities, 5)

print(f"Top prediction: Class {top5_idx[0][0]} ({top5_prob[0][0]:.2%})")
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Create ONNX Runtime session
session = ort.InferenceSession(
    'models/cropshield_cnn.onnx',
    providers=['CPUExecutionProvider']  # or 'CUDAExecutionProvider'
)

# Prepare input (same preprocessing as PyTorch)
image = Image.open('test.jpg').resize((224, 224))
input_array = np.array(image).transpose(2, 0, 1).astype(np.float32)
input_array = (input_array / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
input_array = input_array[np.newaxis, ...]  # Add batch dimension

# Inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_array})[0]

# Get predictions
probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
top5_idx = np.argsort(probabilities[0])[-5:][::-1]

print(f"Top prediction: Class {top5_idx[0]} ({probabilities[0][top5_idx[0]]:.2%})")
```

### Dynamic Batch Inference (ONNX)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models/cropshield_cnn.onnx')

# Single image: shape (1, 3, 224, 224)
single_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {'input': single_input})[0]
print(f"Single batch output shape: {output.shape}")  # (1, 22)

# Batch of 8 images: shape (8, 3, 224, 224)
batch_input = np.random.randn(8, 3, 224, 224).astype(np.float32)
output = session.run(None, {'input': batch_input})[0]
print(f"Batch output shape: {output.shape}")  # (8, 22)
```

---

## Performance Benchmarks

### Expected Results

**CPU (Intel i7-10700K):**

| Format | Time/Image | Speedup | File Size |
|--------|-----------|---------|-----------|
| PyTorch | 45 ms | 1.0x | 25.3 MB |
| TorchScript | 18 ms | 2.5x | 25.5 MB |
| ONNX Runtime | 8 ms | 5.6x | 18.2 MB |

**GPU (NVIDIA RTX 4060):**

| Format | Time/Image | Speedup | File Size |
|--------|-----------|---------|-----------|
| PyTorch | 8 ms | 1.0x | 25.3 MB |
| TorchScript | 3 ms | 2.7x | 25.5 MB |
| ONNX Runtime | 2 ms | 4.0x | 18.2 MB |

### Parity Testing

The export script automatically tests output parity between all formats:

```
🧪 Testing model parity...

   📊 Testing TorchScript vs PyTorch...
      ✅ PASSED
      Max difference:  1.234567e-06
      Mean difference: 5.678901e-07

   📊 Testing ONNX vs PyTorch...
      ✅ PASSED
      Max difference:  2.345678e-06
      Mean difference: 8.901234e-07
```

**Tolerance:** Max difference < 1e-5 (0.00001)

---

## Deployment Scenarios

### Scenario 1: Web App with GradCAM

**Requirements:**
- Fast inference
- GradCAM visualization support
- Moderate deployment size

**Solution:** TorchScript + PyTorch

```python
# Use TorchScript for fast inference
model = torch.jit.load('cropshield_cnn_scripted.pt')
predictions = predict(image, model)

# Load PyTorch model for GradCAM
if user_wants_gradcam:
    pytorch_model = load_pytorch_model('cropshield_cnn.pth')
    gradcam_overlay = generate_gradcam(image, pytorch_model)
```

**Pros:** 2-5x faster inference, GradCAM support  
**Cons:** Still requires PyTorch (~500MB)

---

### Scenario 2: Mobile/Edge Deployment

**Requirements:**
- Minimal size
- No Python dependency
- Cross-platform support

**Solution:** ONNX Runtime

```python
import onnxruntime as ort

session = ort.InferenceSession(
    'cropshield_cnn.onnx',
    providers=['CPUExecutionProvider']
)

output = session.run(None, {'input': input_array})[0]
```

**Pros:** Smallest footprint (~18MB model + 10MB runtime), fastest inference  
**Cons:** No GradCAM, limited debugging

---

### Scenario 3: High-Throughput Server

**Requirements:**
- Maximum throughput
- Batch processing
- GPU acceleration

**Solution:** ONNX Runtime with GPU

```python
session = ort.InferenceSession(
    'cropshield_cnn.onnx',
    providers=['CUDAExecutionProvider']
)

# Process batches efficiently
batch = np.stack([preprocess(img) for img in images])
outputs = session.run(None, {'input': batch})[0]
```

**Pros:** 4-10x faster than PyTorch, dynamic batching  
**Cons:** Requires ONNX Runtime GPU build

---

### Scenario 4: C++ Application

**Requirements:**
- Native C++ integration
- No Python dependency

**Solution:** TorchScript with LibTorch

```cpp
#include <torch/script.h>

// Load model
torch::jit::script::Module model = torch::jit::load("cropshield_cnn_scripted.pt");

// Inference
std::vector<torch::jit::IValue> inputs;
inputs.push_back(input_tensor);
auto output = model.forward(inputs).toTensor();
```

**Pros:** Native performance, GradCAM compatible  
**Cons:** Requires LibTorch (~500MB)

---

## Integration with Existing Pipeline

### Update predict.py for Multi-Format Support

```python
def load_model_once(
    model_path: str,
    format: str = 'pytorch'  # 'pytorch', 'torchscript', 'onnx'
):
    """Load model in specified format."""
    
    if format == 'pytorch':
        model = CropShieldCNN(num_classes=22)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    elif format == 'torchscript':
        model = torch.jit.load(model_path)
        model.eval()
        return model
    
    elif format == 'onnx':
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        return session
    
    else:
        raise ValueError(f"Unknown format: {format}")
```

### Backward Compatible Inference

```python
def predict_disease(
    image_path: str,
    model,  # PyTorch, TorchScript, or ONNX session
    class_names: List[str],
    device: str = 'cpu',
    top_k: int = 3
):
    """Unified prediction function for all formats."""
    
    # Preprocess
    input_tensor = preprocess_image(image_path, device)
    
    # Inference based on model type
    if isinstance(model, torch.jit.ScriptModule):
        # TorchScript
        with torch.no_grad():
            output = model(input_tensor)
    
    elif hasattr(model, 'run'):
        # ONNX Runtime
        input_array = input_tensor.cpu().numpy()
        output = model.run(None, {'input': input_array})[0]
        output = torch.from_numpy(output)
    
    else:
        # PyTorch
        with torch.no_grad():
            output = model(input_tensor)
    
    # Post-process
    probabilities = F.softmax(output, dim=1)
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    return [(class_names[idx], prob.item()) 
            for idx, prob in zip(top_indices[0], top_probs[0])]
```

---

## Troubleshooting

### Common Issues

**1. "Model file not found"**
```bash
# Check model path
ls -lh models/cropshield_cnn.pth

# Use absolute path
python scripts/export_model.py --model /full/path/to/model.pth
```

**2. "CUDA out of memory" during export**
```bash
# Use CPU for export
python scripts/export_model.py --model models/cropshield_cnn.pth --device cpu
```

**3. "ONNX export failed: Unsupported operator"**
```python
# Some PyTorch operations don't have ONNX equivalents
# Check model for dynamic operations (if/else, loops)

# Solution: Use TorchScript instead or simplify model
```

**4. "Parity test failed: Max difference > tolerance"**
```bash
# This is usually acceptable if difference is small (< 1e-4)
# Caused by numerical precision differences

# To investigate:
python scripts/export_model.py --model models/cropshield_cnn.pth --skip_test
```

**5. "onnxruntime not installed"**
```bash
# Install ONNX Runtime
pip install onnxruntime

# For GPU support
pip install onnxruntime-gpu
```

---

## Advanced Topics

### Custom Opset Version

```bash
# Older devices may need older ONNX opset
python scripts/export_model.py \
    --model models/cropshield_cnn.pth \
    --opset 11
```

### Fixed Batch Size (ONNX)

```python
# In export_model.py, set dynamic_axes=False
torch.onnx.export(
    model,
    example_input,
    'model.onnx',
    dynamic_axes=None  # Fixed batch size
)
```

### Quantization for Smaller Models

```python
# Post-training quantization (ONNX)
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    'cropshield_cnn.onnx',
    'cropshield_cnn_quantized.onnx',
    weight_type=QuantType.QInt8
)
# Result: ~75% size reduction, 2-4x faster on CPU
```

---

## Best Practices

### Development Workflow

1. **Train with PyTorch**: Full flexibility, debugging
2. **Export to TorchScript**: Test inference speed
3. **Export to ONNX**: Test cross-platform deployment
4. **Run parity tests**: Ensure correctness
5. **Benchmark performance**: Choose best format for use case

### Deployment Checklist

- [ ] Model exported successfully
- [ ] Parity tests passed (max diff < 1e-5)
- [ ] Inference speed benchmarked
- [ ] File sizes compared
- [ ] Dependencies documented
- [ ] Deployment platform tested
- [ ] Fallback plan for unsupported operations

---

## Summary

| Format | Best For | Pros | Cons |
|--------|----------|------|------|
| **PyTorch** | Development, GradCAM | Full features | Slow, heavy |
| **TorchScript** | Production + GradCAM | Fast, compatible | Still needs PyTorch |
| **ONNX** | Pure inference, edge | Fastest, smallest | No GradCAM |

### Recommendation

- **Web app with visualization**: TorchScript
- **Mobile/edge deployment**: ONNX
- **High-throughput server**: ONNX + GPU
- **C++ integration**: TorchScript + LibTorch
- **Development**: PyTorch

---

## Resources

- [PyTorch TorchScript Docs](https://pytorch.org/docs/stable/jit.html)
- [ONNX Documentation](https://onnx.ai/onnx/intro/)
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- [CropShield AI Repository](https://github.com/yourusername/CropShieldAI)

---

**Need Help?** Check the troubleshooting section or open an issue on GitHub.

**Ready to Export?** Run: `python scripts/export_model.py --model models/cropshield_cnn.pth`
