"""
CropShield AI - QA Inference Test Script
=========================================

End-to-end validation of inference pipeline with performance metrics.

Features:
- Loads model once (tests caching)
- Samples images from Database_resized/ (one per class ideally)
- Runs prediction on each sample
- Measures inference time statistics
- Generates GradCAM visualizations
- Validates predictions and outputs
- Progress tracking with tqdm

Usage:
    # Basic test (5 random images)
    python scripts/test_inference.py
    
    # Test with specific model
    python scripts/test_inference.py --model models/cropshield_cnn.pth
    
    # Test more images
    python scripts/test_inference.py --num_samples 10
    
    # Skip GradCAM generation (faster)
    python scripts/test_inference.py --skip_gradcam
    
    # Test specific classes
    python scripts/test_inference.py --classes Potato__healthy Tomato__healthy

Output:
    - Console: Test results with predictions and timings
    - results/test_gradcams/: GradCAM overlay images (optional)
    - results/test_inference_report.txt: Detailed test report
"""

import torch
import numpy as np
import json
import random
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys
from datetime import datetime

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available. Install with: pip install tqdm")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import inference modules
from predict import load_model_once, predict_disease
from utils.gradcam import generate_gradcam_visualization, get_target_layer


class InferenceQA:
    """
    QA test suite for CropShield AI inference pipeline.
    
    Tests model loading, prediction accuracy, inference speed,
    and GradCAM visualization generation.
    """
    
    def __init__(
        self,
        model_path: str = 'models/cropshield_cnn.pth',
        class_mapping_path: str = 'class_to_idx.json',
        dataset_dir: str = 'Database_resized',
        output_dir: str = 'results/test_gradcams',
        device: Optional[str] = None
    ):
        """
        Initialize QA test suite.
        
        Args:
            model_path: Path to trained model
            class_mapping_path: Path to class mapping JSON
            dataset_dir: Directory containing test images
            output_dir: Directory for GradCAM outputs
            device: Device for inference ('cpu', 'cuda', or None for auto)
        """
        self.model_path = Path(model_path)
        self.class_mapping_path = Path(class_mapping_path)
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize placeholders
        self.model = None
        self.class_names = None
        self.test_images = []
        self.results = []
        
        print(f"\n{'='*70}")
        print(f"🧪 CropShield AI - QA Inference Test")
        print(f"{'='*70}")
        print(f"   Model: {self.model_path}")
        print(f"   Dataset: {self.dataset_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Device: {self.device}")
        print(f"{'='*70}\n")
    
    def validate_setup(self) -> bool:
        """
        Validate that all required files and directories exist.
        
        Returns:
            True if validation passes, False otherwise
        """
        print("📋 Validating setup...")
        
        errors = []
        
        # Check model file
        if not self.model_path.exists():
            errors.append(f"❌ Model not found: {self.model_path}")
        else:
            print(f"   ✓ Model found: {self.model_path}")
        
        # Check class mapping
        if not self.class_mapping_path.exists():
            errors.append(f"❌ Class mapping not found: {self.class_mapping_path}")
        else:
            print(f"   ✓ Class mapping found: {self.class_mapping_path}")
        
        # Check dataset directory
        if not self.dataset_dir.exists():
            errors.append(f"❌ Dataset directory not found: {self.dataset_dir}")
        else:
            class_dirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]
            print(f"   ✓ Dataset found: {len(class_dirs)} classes")
        
        if errors:
            print("\n❌ Validation failed:")
            for error in errors:
                print(f"   {error}")
            return False
        
        print("   ✅ Validation passed!\n")
        return True
    
    def load_model(self) -> bool:
        """
        Load model using the inference pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        print("📦 Loading model...")
        
        try:
            start_time = time.time()
            
            self.model, self.class_names, _ = load_model_once(
                str(self.model_path),
                device=self.device
            )
            
            load_time = time.time() - start_time
            
            print(f"   ✓ Model loaded successfully")
            print(f"   ✓ Classes: {len(self.class_names)}")
            print(f"   ✓ Load time: {load_time:.3f}s\n")
            
            return True
        
        except Exception as e:
            print(f"   ❌ Model loading failed: {e}\n")
            import traceback
            traceback.print_exc()
            return False
    
    def sample_test_images(
        self,
        num_samples: int = 5,
        target_classes: Optional[List[str]] = None,
        seed: int = 42
    ) -> List[Tuple[Path, str]]:
        """
        Sample test images from dataset.
        
        Strategy:
        1. If target_classes specified, sample from those classes
        2. Otherwise, try to get one image per class (diverse sampling)
        3. If fewer classes than samples, randomly sample remaining
        
        Args:
            num_samples: Number of images to sample
            target_classes: Specific classes to sample from (optional)
            seed: Random seed for reproducibility
        
        Returns:
            List of (image_path, true_class) tuples
        """
        print(f"🎲 Sampling {num_samples} test images...")
        
        random.seed(seed)
        
        # Get all class directories
        class_dirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]
        class_dirs = sorted(class_dirs, key=lambda x: x.name)
        
        # Filter by target classes if specified
        if target_classes:
            class_dirs = [d for d in class_dirs if d.name in target_classes]
            if not class_dirs:
                print(f"   ⚠️  No matching classes found: {target_classes}")
                return []
        
        print(f"   ✓ Available classes: {len(class_dirs)}")
        
        sampled_images = []
        
        # Strategy: One image per class (diverse sampling)
        random.shuffle(class_dirs)
        
        for class_dir in class_dirs[:num_samples]:
            # Get all images in this class
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(class_dir.glob(ext)))
            
            if image_files:
                # Randomly pick one image
                image_path = random.choice(image_files)
                true_class = class_dir.name
                sampled_images.append((image_path, true_class))
        
        # If we need more samples, randomly sample from all classes
        if len(sampled_images) < num_samples:
            remaining = num_samples - len(sampled_images)
            print(f"   ⚠️  Only {len(sampled_images)} classes available, sampling {remaining} more randomly")
            
            all_images = []
            for class_dir in class_dirs:
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    images = list(class_dir.glob(ext))
                    all_images.extend([(img, class_dir.name) for img in images])
            
            # Remove already sampled images
            sampled_paths = {path for path, _ in sampled_images}
            all_images = [(path, cls) for path, cls in all_images if path not in sampled_paths]
            
            # Sample remaining
            if all_images:
                additional = random.sample(all_images, min(remaining, len(all_images)))
                sampled_images.extend(additional)
        
        print(f"   ✓ Sampled {len(sampled_images)} images\n")
        
        self.test_images = sampled_images
        return sampled_images
    
    def run_inference(
        self,
        top_k: int = 3,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Run inference on all test images.
        
        Args:
            top_k: Number of top predictions to return
            verbose: Print detailed results
        
        Returns:
            Dictionary with inference results and statistics
        """
        print(f"🔮 Running inference on {len(self.test_images)} images...\n")
        
        inference_times = []
        correct_predictions = 0
        top_k_correct = 0
        
        # Use tqdm if available
        if TQDM_AVAILABLE:
            iterator = tqdm(self.test_images, desc="Inference", unit="image")
        else:
            iterator = self.test_images
            print("   Running inference (no progress bar)...")
        
        for idx, (image_path, true_class) in enumerate(iterator):
            try:
                # Run prediction
                predictions, inference_time = predict_disease(
                    str(image_path),
                    self.model,
                    self.class_names,
                    self.device,
                    top_k=top_k
                )
                
                inference_times.append(inference_time)
                
                # Check if prediction is correct
                predicted_class = predictions[0][0]
                is_correct = (predicted_class == true_class)
                
                if is_correct:
                    correct_predictions += 1
                
                # Check if true class is in top-k
                predicted_classes = [pred[0] for pred in predictions]
                if true_class in predicted_classes:
                    top_k_correct += 1
                
                # Store result
                result = {
                    'index': idx + 1,
                    'image_path': str(image_path),
                    'true_class': true_class,
                    'predictions': predictions,
                    'inference_time': inference_time,
                    'is_correct': is_correct,
                    'in_top_k': true_class in predicted_classes
                }
                self.results.append(result)
                
                # Print detailed result if verbose
                if verbose and not TQDM_AVAILABLE:
                    self._print_result(result, top_k)
            
            except Exception as e:
                print(f"\n   ❌ Error processing {image_path}: {e}")
                continue
        
        # Calculate statistics
        stats = {
            'total_images': len(self.test_images),
            'successful_predictions': len(self.results),
            'top_1_accuracy': correct_predictions / len(self.results) if self.results else 0,
            'top_k_accuracy': top_k_correct / len(self.results) if self.results else 0,
            'avg_inference_time': np.mean(inference_times) if inference_times else 0,
            'std_inference_time': np.std(inference_times) if inference_times else 0,
            'min_inference_time': np.min(inference_times) if inference_times else 0,
            'max_inference_time': np.max(inference_times) if inference_times else 0,
            'total_inference_time': np.sum(inference_times) if inference_times else 0
        }
        
        return stats
    
    def _print_result(self, result: Dict, top_k: int):
        """Print a single inference result."""
        print(f"\n   Image {result['index']}: {Path(result['image_path']).name}")
        print(f"   True class: {result['true_class']}")
        print(f"   Predictions:")
        
        for i, (class_name, confidence) in enumerate(result['predictions'][:top_k], 1):
            marker = "✓" if class_name == result['true_class'] else " "
            print(f"      {marker} {i}. {class_name:40s} {confidence:6.2%}")
        
        print(f"   Inference time: {result['inference_time']:.3f}s")
    
    def generate_gradcam_overlays(
        self,
        verbose: bool = True
    ) -> int:
        """
        Generate GradCAM visualizations for all test images.
        
        Args:
            verbose: Print progress messages
        
        Returns:
            Number of GradCAM overlays generated successfully
        """
        print(f"\n🔥 Generating GradCAM visualizations...\n")
        
        if not self.results:
            print("   ⚠️  No results available. Run inference first.")
            return 0
        
        # Get target layer
        try:
            target_layer = get_target_layer(self.model)
            if verbose:
                print(f"   ✓ Target layer: {target_layer.__class__.__name__}")
        except Exception as e:
            print(f"   ❌ Could not determine target layer: {e}")
            return 0
        
        success_count = 0
        
        # Use tqdm if available
        if TQDM_AVAILABLE:
            iterator = tqdm(self.results, desc="GradCAM", unit="image")
        else:
            iterator = self.results
            if verbose:
                print("   Generating overlays (no progress bar)...")
        
        for result in iterator:
            try:
                image_path = result['image_path']
                predicted_class = result['predictions'][0][0]
                true_class = result['true_class']
                
                # Generate GradCAM
                overlay = generate_gradcam_visualization(
                    model=self.model,
                    image_path=image_path,
                    target_layer=target_layer,
                    device=self.device,
                    colormap='jet'
                )
                
                # Save overlay
                image_name = Path(image_path).stem
                output_name = f"{image_name}_gradcam.jpg"
                output_path = self.output_dir / output_name
                
                from PIL import Image
                Image.fromarray(overlay).save(output_path)
                
                success_count += 1
            
            except Exception as e:
                if verbose and not TQDM_AVAILABLE:
                    print(f"   ⚠️  GradCAM failed for {Path(image_path).name}: {e}")
                continue
        
        if verbose:
            print(f"\n   ✅ Generated {success_count}/{len(self.results)} GradCAM overlays")
            print(f"   ✓ Saved to: {self.output_dir}\n")
        
        return success_count
    
    def print_summary(self, stats: Dict):
        """
        Print comprehensive test summary.
        
        Args:
            stats: Statistics dictionary from run_inference()
        """
        print(f"\n{'='*70}")
        print(f"📊 INFERENCE TEST SUMMARY")
        print(f"{'='*70}\n")
        
        # Basic stats
        print(f"🎯 Test Results:")
        print(f"   Total images tested: {stats['total_images']}")
        print(f"   Successful predictions: {stats['successful_predictions']}")
        print(f"   Top-1 accuracy: {stats['top_1_accuracy']:.1%}")
        print(f"   Top-3 accuracy: {stats['top_k_accuracy']:.1%}")
        
        # Timing stats
        print(f"\n⚡ Inference Performance:")
        print(f"   Average time per image: {stats['avg_inference_time']:.3f}s")
        print(f"   Std deviation: {stats['std_inference_time']:.3f}s")
        print(f"   Min time: {stats['min_inference_time']:.3f}s")
        print(f"   Max time: {stats['max_inference_time']:.3f}s")
        print(f"   Total inference time: {stats['total_inference_time']:.3f}s")
        
        # Detailed results
        print(f"\n📋 Detailed Predictions:")
        print(f"{'='*70}")
        
        for result in self.results:
            image_name = Path(result['image_path']).name
            true_class = result['true_class']
            predicted_class = result['predictions'][0][0]
            confidence = result['predictions'][0][1]
            
            # Status indicator
            if result['is_correct']:
                status = "✅"
            elif result['in_top_k']:
                status = "⚠️ "
            else:
                status = "❌"
            
            print(f"\n{status} Image {result['index']}: {image_name}")
            print(f"   True: {true_class}")
            print(f"   Predicted: {predicted_class} ({confidence:.1%})")
            
            if len(result['predictions']) > 1:
                print(f"   Top-3:")
                for i, (cls, conf) in enumerate(result['predictions'][:3], 1):
                    marker = "→" if cls == true_class else " "
                    print(f"      {marker} {i}. {cls:40s} {conf:6.2%}")
        
        print(f"\n{'='*70}")
    
    def save_report(self, stats: Dict, report_path: str = 'results/test_inference_report.txt'):
        """
        Save detailed test report to file.
        
        Args:
            stats: Statistics dictionary from run_inference()
            report_path: Path to save report
        """
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("CropShield AI - Inference QA Test Report\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Dataset: {self.dataset_dir}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Classes: {len(self.class_names)}\n\n")
            
            f.write("="*70 + "\n")
            f.write("TEST SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total images: {stats['total_images']}\n")
            f.write(f"Successful predictions: {stats['successful_predictions']}\n")
            f.write(f"Top-1 accuracy: {stats['top_1_accuracy']:.2%}\n")
            f.write(f"Top-3 accuracy: {stats['top_k_accuracy']:.2%}\n\n")
            
            f.write(f"Average inference time: {stats['avg_inference_time']:.3f}s\n")
            f.write(f"Std deviation: {stats['std_inference_time']:.3f}s\n")
            f.write(f"Min time: {stats['min_inference_time']:.3f}s\n")
            f.write(f"Max time: {stats['max_inference_time']:.3f}s\n")
            f.write(f"Total time: {stats['total_inference_time']:.3f}s\n\n")
            
            f.write("="*70 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for result in self.results:
                f.write(f"Image {result['index']}: {Path(result['image_path']).name}\n")
                f.write(f"  True class: {result['true_class']}\n")
                f.write(f"  Predictions:\n")
                
                for i, (cls, conf) in enumerate(result['predictions'][:3], 1):
                    marker = "✓" if cls == result['true_class'] else " "
                    f.write(f"    {marker} {i}. {cls:40s} {conf:6.2%}\n")
                
                f.write(f"  Inference time: {result['inference_time']:.3f}s\n")
                f.write(f"  Correct: {'Yes' if result['is_correct'] else 'No'}\n")
                f.write(f"  In top-3: {'Yes' if result['in_top_k'] else 'No'}\n\n")
        
        print(f"💾 Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='QA test script for CropShield AI inference pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with 5 random images
  python scripts/test_inference.py
  
  # Test with 10 images
  python scripts/test_inference.py --num_samples 10
  
  # Test specific classes
  python scripts/test_inference.py --classes Potato__healthy Tomato__healthy
  
  # Skip GradCAM generation (faster)
  python scripts/test_inference.py --skip_gradcam
  
  # Custom model and dataset
  python scripts/test_inference.py --model models/best_model.pth --dataset Database/
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/cropshield_cnn.pth',
        help='Path to trained model (default: models/cropshield_cnn.pth)'
    )
    
    parser.add_argument(
        '--class_mapping',
        type=str,
        default='class_to_idx.json',
        help='Path to class mapping JSON (default: class_to_idx.json)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='Database_resized',
        help='Dataset directory (default: Database_resized)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/test_gradcams',
        help='Output directory for GradCAM images (default: results/test_gradcams)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of test images (default: 5)'
    )
    
    parser.add_argument(
        '--classes',
        nargs='+',
        default=None,
        help='Specific classes to test (default: all classes)'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Number of top predictions (default: 3)'
    )
    
    parser.add_argument(
        '--skip_gradcam',
        action='store_true',
        help='Skip GradCAM generation (faster)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device for inference (default: auto)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Determine device
    device = None if args.device == 'auto' else args.device
    
    # Initialize QA suite
    qa = InferenceQA(
        model_path=args.model,
        class_mapping_path=args.class_mapping,
        dataset_dir=args.dataset,
        output_dir=args.output,
        device=device
    )
    
    # Run validation
    if not qa.validate_setup():
        print("\n❌ Setup validation failed. Please fix errors and try again.\n")
        sys.exit(1)
    
    # Load model
    if not qa.load_model():
        print("\n❌ Model loading failed. Please check model file and try again.\n")
        sys.exit(1)
    
    # Sample test images
    test_images = qa.sample_test_images(
        num_samples=args.num_samples,
        target_classes=args.classes,
        seed=args.seed
    )
    
    if not test_images:
        print("\n❌ No test images found. Please check dataset directory.\n")
        sys.exit(1)
    
    # Run inference
    stats = qa.run_inference(top_k=args.top_k, verbose=False)
    
    # Generate GradCAM overlays
    if not args.skip_gradcam:
        qa.generate_gradcam_overlays(verbose=True)
    else:
        print("\n⏭️  Skipping GradCAM generation\n")
    
    # Print summary
    qa.print_summary(stats)
    
    # Save report
    qa.save_report(stats)
    
    # Final message
    print(f"\n✅ QA test complete!")
    print(f"   Top-1 accuracy: {stats['top_1_accuracy']:.1%}")
    print(f"   Average inference time: {stats['avg_inference_time']:.3f}s")
    
    if not args.skip_gradcam:
        print(f"   GradCAM overlays: {args.output}")
    
    print(f"   Report: results/test_inference_report.txt\n")


if __name__ == '__main__':
    main()
