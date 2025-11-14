"""
CropShield AI - Quick Inference Test
=====================================

Minimal automated test that:
- Loads trained model (models/cropshield_cnn.pth)
- Tests on 3-5 random images from dataset
- Shows top-3 predictions with confidence scores
- Generates GradCAM overlay for visual verification

Usage:
    python test_model_inference.py
    
Fully automated - no user input required!
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Import project modules
from models.model_factory import get_model
from app_utils import create_gradcam_overlay, get_preprocessing_transform


def load_model_and_classes(model_path='models/cropshield_cnn.pth'):
    """Load trained model and class mappings"""
    print("\n" + "="*80)
    print("🧠 LOADING MODEL")
    print("="*80)
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Train first: python train_auto.py")
        return None, None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    model_type = checkpoint.get('model_type', 'custom')
    
    print(f"✅ Model: {model_type}")
    print(f"   Classes: {num_classes}")
    print(f"   Best Val Acc: {checkpoint.get('best_val_acc', 0):.1f}%")
    
    model, _, _, _, _ = get_model(model_type=model_type, num_classes=num_classes, learning_rate=0.001)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, device


def get_random_images(data_dir='Database_resized', num=5):
    """Get random test images"""
    print("\n" + "="*80)
    print("📸 SELECTING IMAGES")
    print("="*80)
    
    data_path = Path(data_dir)
    all_images = []
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            all_images.extend(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')))
    
    random.seed(42)
    selected = random.sample(all_images, min(num, len(all_images)))
    
    print(f"✅ Selected {len(selected)} images:")
    for i, img in enumerate(selected, 1):
        print(f"   {i}. {img.name} (True: {img.parent.name})")
    
    return selected


def predict_top3(model, image_path, device, class_names):
    """Run inference and get top-3 predictions"""
    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    transform = get_preprocessing_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_indices = probs.topk(3, dim=1)
    
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append({
            'class': class_names[idx.item()],
            'confidence': prob.item() * 100
        })
    
    return image, image_tensor, results


def test_inference(model, class_names, device, test_images):
    """Test inference on all images"""
    print("\n" + "="*80)
    print("🔮 RUNNING INFERENCE")
    print("="*80)
    
    results = []
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n📸 Image {i}: {img_path.name}")
        print(f"   True: {img_path.parent.name}")
        
        image, image_tensor, predictions = predict_top3(model, img_path, device, class_names)
        
        print(f"\n   🎯 Top-3 Predictions:")
        for j, pred in enumerate(predictions, 1):
            bar = "█" * int(pred['confidence'] / 5)
            print(f"      {j}. {pred['class']:<35} {pred['confidence']:>5.1f}% {bar}")
        
        is_correct = (predictions[0]['class'] == img_path.parent.name)
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        print(f"   {status}")
        
        results.append({
            'path': img_path,
            'image': image,
            'image_tensor': image_tensor,
            'true': img_path.parent.name,
            'predictions': predictions,
            'correct': is_correct
        })
    
    return results


def generate_gradcam(model, device, result, class_names):
    """Generate GradCAM visualization"""
    print("\n" + "="*80)
    print("🔥 GENERATING GRADCAM")
    print("="*80)
    
    img_path = result['path']
    image = result['image']
    image_tensor = result['image_tensor']
    pred = result['predictions'][0]
    
    print(f"📸 Image: {img_path.name}")
    print(f"🎯 Prediction: {pred['class']} ({pred['confidence']:.1f}%)")
    
    try:
        # Find target layer
        target_layer = None
        if hasattr(model, 'features'):
            target_layer = model.features[-2]
        else:
            for module in reversed(list(model.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break
        
        if target_layer is None:
            print("⚠️  No conv layer found for GradCAM")
            return None
        
        # Get predicted class index
        with torch.no_grad():
            outputs = model(image_tensor)
            pred_idx = outputs.argmax(dim=1).item()
        
        # Generate GradCAM
        gradcam_overlay = create_gradcam_overlay(
            model=model,
            image_tensor=image_tensor,
            target_layer=target_layer,
            class_idx=pred_idx,
            device=device
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        
        axes[1].imshow(gradcam_overlay)
        axes[1].axis('off')
        axes[1].set_title(
            f'GradCAM\n{pred["class"]}\n{pred["confidence"]:.1f}%',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.suptitle(f'CropShield AI - GradCAM Test', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f'gradcam_test_{timestamp}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ GradCAM saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"⚠️  GradCAM error: {e}")
        return None


def print_summary(results):
    """Print test summary"""
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    print(f"\n✅ Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    
    print(f"\n📋 Results:")
    for i, r in enumerate(results, 1):
        status = "✅" if r['correct'] else "❌"
        pred = r['predictions'][0]
        print(f"   {i}. {status} {r['path'].name}")
        print(f"      Predicted: {pred['class']} ({pred['confidence']:.1f}%)")
        print(f"      True: {r['true']}")
    
    print("\n" + "="*80)


def main():
    """Main test function"""
    print("\n" + "="*80)
    print("🌾 CROPSHIELD AI - INFERENCE TEST")
    print("="*80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        # Load model
        model, class_names, device = load_model_and_classes()
        if model is None:
            return False
        
        # Get test images
        test_images = get_random_images(num=5)
        
        # Run inference
        results = test_inference(model, class_names, device, test_images)
        
        # Generate GradCAM for first image
        gradcam_path = generate_gradcam(model, device, results[0], class_names)
        
        # Print summary
        print_summary(results)
        
        print("\n🎉 TEST COMPLETE!")
        if gradcam_path:
            print(f"📁 GradCAM: {gradcam_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
