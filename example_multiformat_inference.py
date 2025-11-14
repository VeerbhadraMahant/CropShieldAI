"""
CropShield AI - Multi-Format Inference Example
==============================================

Demonstrates how to use PyTorch, TorchScript, and ONNX models
for inference with the same unified API.

Features:
- Automatic format detection
- Unified prediction API
- GradCAM compatibility check
- Performance comparison
- Fallback mechanisms

Usage:
    # TorchScript (fast, GradCAM compatible)
    python example_multiformat_inference.py --model models/cropshield_cnn_scripted.pt --image test.jpg
    
    # ONNX (fastest, no GradCAM)
    python example_multiformat_inference.py --model models/cropshield_cnn.onnx --image test.jpg
    
    # PyTorch (full features)
    python example_multiformat_inference.py --model models/cropshield_cnn.pth --image test.jpg
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union

# Import transforms
from transforms import get_validation_transforms

# Import model architecture
from model_custom_cnn import CropShieldCNN


class MultiFormatInference:
    """
    Unified inference class supporting PyTorch, TorchScript, and ONNX models.
    
    Automatically detects model format and provides consistent API across all formats.
    """
    
    def __init__(
        self,
        model_path: str,
        class_names_path: str = 'class_to_idx.json',
        device: str = 'cpu'
    ):
        """
        Initialize multi-format inference.
        
        Args:
            model_path: Path to model file (.pth, .pt, or .onnx)
            class_names_path: Path to class mapping JSON
            device: Device for inference ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = device
        
        # Detect format
        self.format = self._detect_format()
        print(f"📦 Detected model format: {self.format}")
        
        # Load model
        self.model = self._load_model()
        
        # Load class names
        self.class_names = self._load_class_names(class_names_path)
        print(f"✓ Loaded {len(self.class_names)} classes")
        
        # Get transforms
        self.transform = get_validation_transforms()
        
        # Check GradCAM compatibility
        self.gradcam_compatible = self.format in ['pytorch', 'torchscript']
        if not self.gradcam_compatible:
            print(f"⚠️  GradCAM not supported with {self.format} format")
    
    def _detect_format(self) -> str:
        """Detect model format from file extension."""
        suffix = self.model_path.suffix.lower()
        
        if suffix == '.pth':
            return 'pytorch'
        elif suffix == '.pt':
            return 'torchscript'
        elif suffix == '.onnx':
            return 'onnx'
        else:
            raise ValueError(f"Unknown model format: {suffix}")
    
    def _load_model(self):
        """Load model based on detected format."""
        if self.format == 'pytorch':
            return self._load_pytorch_model()
        elif self.format == 'torchscript':
            return self._load_torchscript_model()
        elif self.format == 'onnx':
            return self._load_onnx_model()
    
    def _load_pytorch_model(self) -> torch.nn.Module:
        """Load PyTorch model from .pth file."""
        print(f"📂 Loading PyTorch model...")
        
        model = CropShieldCNN(num_classes=len(self.class_names) if hasattr(self, 'class_names') else 22)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(self.device)
        
        return model
    
    def _load_torchscript_model(self):
        """Load TorchScript model from .pt file."""
        print(f"📂 Loading TorchScript model...")
        
        model = torch.jit.load(str(self.model_path), map_location=self.device)
        model.eval()
        
        return model
    
    def _load_onnx_model(self):
        """Load ONNX model from .onnx file."""
        print(f"📂 Loading ONNX model...")
        
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX Runtime not installed. Install with: pip install onnxruntime"
            )
        
        # Determine provider
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        session = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        
        print(f"   ✓ Provider: {session.get_providers()[0]}")
        
        return session
    
    def _load_class_names(self, path: str) -> List[str]:
        """Load class names from JSON file."""
        with open(path, 'r') as f:
            class_to_idx = json.load(f)
        
        # Convert to list sorted by index
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        
        return class_names
    
    def preprocess_image(self, image_path: str) -> Union[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Preprocessed tensor (PyTorch/TorchScript) or array (ONNX)
        """
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        if self.format == 'onnx':
            # ONNX needs numpy array
            return input_tensor.numpy()
        else:
            # PyTorch and TorchScript
            return input_tensor.to(self.device)
    
    def predict(
        self,
        image_path: str,
        top_k: int = 5
    ) -> Tuple[List[Tuple[str, float]], float]:
        """
        Predict disease class for input image.
        
        Args:
            image_path: Path to input image
            top_k: Number of top predictions to return
        
        Returns:
            Tuple of (predictions, inference_time):
                - predictions: List of (class_name, confidence) tuples
                - inference_time: Inference time in milliseconds
        """
        # Preprocess
        input_data = self.preprocess_image(image_path)
        
        # Inference
        start_time = time.time()
        
        if self.format == 'pytorch' or self.format == 'torchscript':
            with torch.no_grad():
                output = self.model(input_data)
            
            probabilities = F.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Convert to list
            predictions = [
                (self.class_names[idx.item()], prob.item())
                for idx, prob in zip(top_indices[0], top_probs[0])
            ]
        
        elif self.format == 'onnx':
            # ONNX Runtime inference
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: input_data})[0]
            
            # Softmax
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            probabilities = exp_output / np.sum(exp_output, axis=1, keepdims=True)
            
            # Top-k
            top_indices = np.argsort(probabilities[0])[-top_k:][::-1]
            
            predictions = [
                (self.class_names[idx], float(probabilities[0][idx]))
                for idx in top_indices
            ]
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return predictions, inference_time
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 8
    ) -> Tuple[List[List[Tuple[str, float]]], float]:
        """
        Batch prediction for multiple images.
        
        Note: Only ONNX format supports dynamic batching efficiently.
        PyTorch/TorchScript will process images one by one.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size (only used for ONNX)
        
        Returns:
            Tuple of (all_predictions, total_time):
                - all_predictions: List of prediction lists for each image
                - total_time: Total processing time in milliseconds
        """
        if self.format == 'onnx':
            # True batch processing for ONNX
            return self._predict_batch_onnx(image_paths, batch_size)
        else:
            # Sequential processing for PyTorch/TorchScript
            return self._predict_batch_sequential(image_paths)
    
    def _predict_batch_sequential(
        self,
        image_paths: List[str]
    ) -> Tuple[List[List[Tuple[str, float]]], float]:
        """Sequential batch processing."""
        all_predictions = []
        start_time = time.time()
        
        for image_path in image_paths:
            predictions, _ = self.predict(image_path, top_k=5)
            all_predictions.append(predictions)
        
        total_time = (time.time() - start_time) * 1000
        
        return all_predictions, total_time
    
    def _predict_batch_onnx(
        self,
        image_paths: List[str],
        batch_size: int
    ) -> Tuple[List[List[Tuple[str, float]]], float]:
        """Batch processing for ONNX (supports dynamic batching)."""
        all_predictions = []
        start_time = time.time()
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Preprocess batch
            batch_inputs = np.stack([
                self.preprocess_image(path) for path in batch_paths
            ])
            
            # Inference
            input_name = self.model.get_inputs()[0].name
            batch_output = self.model.run(None, {input_name: batch_inputs})[0]
            
            # Softmax
            exp_output = np.exp(batch_output - np.max(batch_output, axis=1, keepdims=True))
            probabilities = exp_output / np.sum(exp_output, axis=1, keepdims=True)
            
            # Top-k for each image in batch
            for probs in probabilities:
                top_indices = np.argsort(probs)[-5:][::-1]
                predictions = [
                    (self.class_names[idx], float(probs[idx]))
                    for idx in top_indices
                ]
                all_predictions.append(predictions)
        
        total_time = (time.time() - start_time) * 1000
        
        return all_predictions, total_time


def compare_formats(
    image_path: str,
    pytorch_model: Optional[str] = None,
    torchscript_model: Optional[str] = None,
    onnx_model: Optional[str] = None,
    class_names_path: str = 'class_to_idx.json',
    num_runs: int = 10
):
    """
    Compare inference speed across different model formats.
    
    Args:
        image_path: Path to test image
        pytorch_model: Path to .pth model (optional)
        torchscript_model: Path to .pt model (optional)
        onnx_model: Path to .onnx model (optional)
        class_names_path: Path to class names JSON
        num_runs: Number of inference runs for averaging
    """
    print("\n" + "="*70)
    print("⚡ Model Format Comparison")
    print("="*70 + "\n")
    
    results = {}
    
    # Test each format
    for name, model_path in [
        ('PyTorch', pytorch_model),
        ('TorchScript', torchscript_model),
        ('ONNX', onnx_model)
    ]:
        if model_path and Path(model_path).exists():
            print(f"Testing {name}...")
            
            try:
                inference = MultiFormatInference(model_path, class_names_path)
                
                # Warmup
                for _ in range(3):
                    _, _ = inference.predict(image_path)
                
                # Benchmark
                times = []
                for _ in range(num_runs):
                    predictions, time_ms = inference.predict(image_path)
                    times.append(time_ms)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                results[name] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'predictions': predictions
                }
                
                print(f"   ✓ Average time: {avg_time:.2f} ± {std_time:.2f} ms\n")
            
            except Exception as e:
                print(f"   ❌ Error: {e}\n")
    
    # Print comparison
    if len(results) > 1:
        print("\n" + "="*70)
        print("📊 Performance Comparison")
        print("="*70 + "\n")
        
        baseline = results.get('PyTorch', list(results.values())[0])
        baseline_time = baseline['avg_time']
        
        for name, result in results.items():
            speedup = baseline_time / result['avg_time']
            print(f"{name:15} {result['avg_time']:6.2f} ms  ({speedup:.2f}x)")
        
        print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-format inference for CropShield AI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model (.pth, .pt, or .onnx)'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--class_names',
        type=str,
        default='class_to_idx.json',
        help='Path to class names JSON (default: class_to_idx.json)'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of top predictions (default: 5)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all available model formats'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare mode: Test all formats
        model_dir = Path(args.model).parent
        model_base = Path(args.model).stem.replace('_scripted', '').replace('.onnx', '')
        
        compare_formats(
            image_path=args.image,
            pytorch_model=str(model_dir / f"{model_base}.pth"),
            torchscript_model=str(model_dir / f"{model_base}_scripted.pt"),
            onnx_model=str(model_dir / f"{model_base}.onnx"),
            class_names_path=args.class_names
        )
    
    else:
        # Single inference
        print("\n" + "="*70)
        print("🔮 CropShield AI - Multi-Format Inference")
        print("="*70 + "\n")
        
        # Initialize inference
        inference = MultiFormatInference(
            model_path=args.model,
            class_names_path=args.class_names,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Predict
        print(f"\n📸 Analyzing image: {args.image}")
        predictions, inference_time = inference.predict(args.image, top_k=args.top_k)
        
        # Display results
        print(f"\n⚡ Inference time: {inference_time:.2f} ms")
        print(f"\n🎯 Top-{args.top_k} Predictions:\n")
        
        for i, (class_name, confidence) in enumerate(predictions, 1):
            bar_length = int(confidence * 50)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"   {i}. {class_name:30s} {confidence:6.2%} {bar}")
        
        # GradCAM notice
        if not inference.gradcam_compatible:
            print(f"\n⚠️  Note: GradCAM visualization not supported with {inference.format} format")
            print(f"    Use .pth or .pt model for GradCAM")
        
        print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
