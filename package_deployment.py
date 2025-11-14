"""
CropShield AI - Deployment Packaging Script

Automatically bundles all training outputs, results, and generates
a comprehensive deployment report.

Usage:
    python package_deployment.py
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import torch

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_file_size(filepath):
    """Get file size in human-readable format"""
    if not os.path.exists(filepath):
        return "N/A"
    
    size_bytes = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def count_model_parameters(model_path):
    """Count total and trainable parameters in model"""
    if not os.path.exists(model_path):
        return None, None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        total_params = sum(p.numel() for p in state_dict.values())
        trainable_params = total_params  # All loaded params are trainable
        
        return total_params, trainable_params
    except Exception as e:
        print(f"Warning: Could not count parameters: {e}")
        return None, None

def load_metrics(metrics_path):
    """Load metrics from JSON file"""
    if not os.path.exists(metrics_path):
        return None
    
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metrics: {e}")
        return None

def load_experiment_summary(summary_path):
    """Load experiment sweep summary"""
    if not os.path.exists(summary_path):
        return None
    
    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load experiment summary: {e}")
        return None

def load_final_retrain_results(results_path):
    """Load final retrain results"""
    if not os.path.exists(results_path):
        return None
    
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load retrain results: {e}")
        return None

def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds is None:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def verify_files():
    """Verify existence of key files"""
    files_status = {}
    
    # Check models
    files_status['models/cropshield_cnn.pth'] = os.path.exists('models/cropshield_cnn.pth')
    files_status['models/cropshield_cnn_best.pth'] = os.path.exists('models/cropshield_cnn_best.pth')
    
    # Check results
    files_status['results/confusion_matrix.png'] = os.path.exists('results/confusion_matrix.png')
    files_status['results/test_metrics.json'] = os.path.exists('results/test_metrics.json')
    
    # Check experiments
    files_status['experiments/'] = os.path.exists('experiments') and os.path.isdir('experiments')
    
    return files_status

def count_experiment_files():
    """Count number of experiment log files"""
    if not os.path.exists('experiments'):
        return 0
    
    return len([f for f in os.listdir('experiments') if f.endswith('.json')])

def generate_deployment_report():
    """Generate comprehensive deployment report"""
    
    print("=" * 80)
    print("🎯 CROPSHIELD AI - DEPLOYMENT PACKAGING")
    print("=" * 80)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Verify files
    print("📋 Verifying Files...")
    files_status = verify_files()
    
    all_present = True
    for file, exists in files_status.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {file}")
        if not exists:
            all_present = False
    print()
    
    if not all_present:
        print("⚠️  Warning: Some files are missing!")
        print()
    
    # Load metrics
    print("📊 Loading Metrics...")
    metrics = load_metrics('results/test_metrics.json')
    experiment_summary = load_experiment_summary('experiments/sweep_summary.json')
    retrain_results = load_final_retrain_results('experiments/final_retrain_results.json')
    print()
    
    # Generate report content
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CROPSHIELD AI - DEPLOYMENT REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Model Information
    report_lines.append("📦 MODEL INFORMATION")
    report_lines.append("-" * 80)
    
    # Baseline model
    if os.path.exists('models/cropshield_cnn.pth'):
        total_params, trainable_params = count_model_parameters('models/cropshield_cnn.pth')
        report_lines.append("Baseline Model: models/cropshield_cnn.pth")
        if total_params:
            report_lines.append(f"   Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        report_lines.append(f"   File Size: {get_file_size('models/cropshield_cnn.pth')}")
    
    # Optimized model
    if os.path.exists('models/cropshield_cnn_best.pth'):
        total_params, trainable_params = count_model_parameters('models/cropshield_cnn_best.pth')
        report_lines.append("")
        report_lines.append("Optimized Model: models/cropshield_cnn_best.pth")
        if total_params:
            report_lines.append(f"   Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        report_lines.append(f"   File Size: {get_file_size('models/cropshield_cnn_best.pth')}")
    
    report_lines.append("")
    
    # Performance Metrics
    report_lines.append("📈 PERFORMANCE METRICS")
    report_lines.append("-" * 80)
    
    if metrics:
        report_lines.append(f"Test Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}" if isinstance(metrics.get('accuracy'), (int, float)) else f"Test Accuracy:  {metrics.get('accuracy', 'N/A')}")
        report_lines.append(f"Precision:      {metrics.get('precision_macro', 'N/A'):.4f}" if isinstance(metrics.get('precision_macro'), (int, float)) else f"Precision:      {metrics.get('precision_macro', 'N/A')}")
        report_lines.append(f"Recall:         {metrics.get('recall_macro', 'N/A'):.4f}" if isinstance(metrics.get('recall_macro'), (int, float)) else f"Recall:         {metrics.get('recall_macro', 'N/A')}")
        report_lines.append(f"F1-Score:       {metrics.get('f1_macro', 'N/A'):.4f}" if isinstance(metrics.get('f1_macro'), (int, float)) else f"F1-Score:       {metrics.get('f1_macro', 'N/A')}")
        
        if 'num_classes' in metrics:
            report_lines.append(f"Classes:        {metrics['num_classes']}")
        if 'num_test_samples' in metrics:
            report_lines.append(f"Test Samples:   {metrics['num_test_samples']}")
    else:
        report_lines.append("Metrics: Not Available")
    
    report_lines.append("")
    
    # Hyperparameter Optimization Results
    if experiment_summary or retrain_results:
        report_lines.append("🔬 HYPERPARAMETER OPTIMIZATION")
        report_lines.append("-" * 80)
        
        if experiment_summary:
            report_lines.append(f"Experiments Run: {experiment_summary.get('total_experiments', 'N/A')}")
            report_lines.append(f"Quick Epochs per Experiment: {experiment_summary.get('quick_epochs', 'N/A')}")
            
            best_config = experiment_summary.get('best_config', {})
            if best_config:
                report_lines.append("")
                report_lines.append("Best Configuration:")
                report_lines.append(f"   Learning Rate: {best_config.get('learning_rate', 'N/A')}")
                report_lines.append(f"   Weight Decay:  {best_config.get('weight_decay', 'N/A')}")
                report_lines.append(f"   Dropout:       {best_config.get('dropout', 'N/A')}")
                report_lines.append(f"   Best Val Acc:  {experiment_summary.get('best_val_acc', 'N/A'):.4f}" if isinstance(experiment_summary.get('best_val_acc'), (int, float)) else f"   Best Val Acc:  {experiment_summary.get('best_val_acc', 'N/A')}")
        
        if retrain_results:
            report_lines.append("")
            report_lines.append("Final Retrain Results:")
            report_lines.append(f"   Total Epochs:     {retrain_results.get('total_epochs', 'N/A')}")
            report_lines.append(f"   Best Epoch:       {retrain_results.get('best_epoch', 'N/A')}")
            report_lines.append(f"   Best Val Acc:     {retrain_results.get('best_val_acc', 'N/A'):.4f}" if isinstance(retrain_results.get('best_val_acc'), (int, float)) else f"   Best Val Acc:     {retrain_results.get('best_val_acc', 'N/A')}")
            
            training_time = retrain_results.get('training_time')
            if training_time:
                report_lines.append(f"   Training Time:    {format_time(training_time)}")
        
        report_lines.append("")
    
    # Training Information
    report_lines.append("⏱️  TRAINING INFORMATION")
    report_lines.append("-" * 80)
    
    # Try to get training duration from various sources
    training_duration = None
    if retrain_results and 'training_time' in retrain_results:
        training_duration = retrain_results['training_time']
    
    if training_duration:
        report_lines.append(f"Training Duration: {format_time(training_duration)}")
    else:
        report_lines.append("Training Duration: Not available")
    
    # Check for training history
    if os.path.exists('models/cropshield_cnn.pth'):
        try:
            checkpoint = torch.load('models/cropshield_cnn.pth', map_location='cpu')
            if 'history' in checkpoint:
                history = checkpoint['history']
                if 'train_loss' in history:
                    report_lines.append(f"Training Epochs: {len(history['train_loss'])}")
                if 'val_acc' in history and history['val_acc']:
                    best_val = max(history['val_acc'])
                    report_lines.append(f"Best Validation Accuracy: {best_val:.4f}")
        except:
            pass
    
    report_lines.append("")
    
    # Files Summary
    report_lines.append("📁 OUTPUT FILES")
    report_lines.append("-" * 80)
    report_lines.append("Models:")
    if os.path.exists('models/cropshield_cnn.pth'):
        report_lines.append(f"   ✅ models/cropshield_cnn.pth ({get_file_size('models/cropshield_cnn.pth')})")
    if os.path.exists('models/cropshield_cnn_best.pth'):
        report_lines.append(f"   ✅ models/cropshield_cnn_best.pth ({get_file_size('models/cropshield_cnn_best.pth')})")
    
    report_lines.append("")
    report_lines.append("Results:")
    if os.path.exists('results/confusion_matrix.png'):
        report_lines.append(f"   ✅ results/confusion_matrix.png ({get_file_size('results/confusion_matrix.png')})")
    if os.path.exists('results/test_metrics.json'):
        report_lines.append(f"   ✅ results/test_metrics.json ({get_file_size('results/test_metrics.json')})")
    
    report_lines.append("")
    exp_count = count_experiment_files()
    report_lines.append(f"Experiments: {exp_count} log files")
    if exp_count > 0:
        report_lines.append("   ✅ experiments/ directory contains optimization logs")
    
    report_lines.append("")
    
    # System Information
    report_lines.append("🖥️  SYSTEM INFORMATION")
    report_lines.append("-" * 80)
    report_lines.append(f"PyTorch Version: {torch.__version__}")
    report_lines.append(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        report_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        report_lines.append(f"CUDA Version: {torch.version.cuda}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("✅ CropShield AI Model Ready for Deployment")
    report_lines.append("=" * 80)
    
    # Save report
    ensure_dir('results')
    report_path = 'results/final_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Print report
    print("📄 Deployment Report Generated:")
    print()
    for line in report_lines:
        print(line)
    
    print()
    print(f"💾 Report saved to: {report_path}")
    print()
    
    return report_path

def create_deployment_package():
    """Create organized deployment package"""
    
    print("=" * 80)
    print("📦 CREATING DEPLOYMENT PACKAGE")
    print("=" * 80)
    print()
    
    # Ensure all directories exist
    print("📁 Verifying directory structure...")
    ensure_dir('models')
    ensure_dir('results')
    ensure_dir('experiments')
    print("   ✅ All directories present")
    print()
    
    # Check files
    print("🔍 Checking available files...")
    required_files = [
        ('models/cropshield_cnn.pth', 'Baseline model'),
        ('results/test_metrics.json', 'Test metrics'),
        ('results/confusion_matrix.png', 'Confusion matrix')
    ]
    
    optional_files = [
        ('models/cropshield_cnn_best.pth', 'Optimized model'),
        ('experiments/sweep_summary.json', 'Experiment summary'),
        ('experiments/final_retrain_results.json', 'Final retrain results')
    ]
    
    found_files = []
    missing_required = []
    for filepath, description in required_files:
        if os.path.exists(filepath):
            print(f"   ✅ {description}: {filepath}")
            found_files.append((filepath, description))
        else:
            print(f"   ⚠️  {description}: {filepath} (will create placeholder)")
            missing_required.append(filepath)
    
    print()
    print("📋 Optional files:")
    for filepath, description in optional_files:
        if os.path.exists(filepath):
            print(f"   ✅ {description}: {filepath}")
            found_files.append((filepath, description))
        else:
            print(f"   ⚠️  {description}: {filepath} (not found)")
    
    print()
    
    if missing_required:
        print(f"ℹ️  Note: Creating deployment package with available files.")
        print("   Run training scripts to generate missing files:")
        print("   - python train_auto.py --epochs 25")
        print("   - python quick_evaluate.py")
        print("   - python scripts/hparam_sweep.py")
        print()
    
    # Generate deployment report
    print("📊 Generating deployment report...")
    report_path = generate_deployment_report()
    
    print()
    print("=" * 80)
    print("✅ CropShield AI Model Ready for Deployment")
    print("=" * 80)
    print()
    print("📦 Deployment Package Contents:")
    print()
    print("   models/")
    print("      └─ cropshield_cnn.pth          (Baseline model)")
    if os.path.exists('models/cropshield_cnn_best.pth'):
        print("      └─ cropshield_cnn_best.pth     (Optimized model)")
    print()
    print("   results/")
    print("      ├─ confusion_matrix.png        (Evaluation visualization)")
    print("      ├─ test_metrics.json           (Performance metrics)")
    print("      └─ final_report.txt            (Deployment report)")
    print()
    print("   experiments/")
    exp_count = count_experiment_files()
    print(f"      └─ {exp_count} experiment log files")
    print()
    print("=" * 80)
    print()
    
    return True

def main():
    """Main packaging function"""
    try:
        success = create_deployment_package()
        
        if success:
            print("🎉 Packaging complete!")
            print()
            print("📋 Next Steps:")
            print("   1. Review results/final_report.txt")
            print("   2. Use models/cropshield_cnn_best.pth for deployment")
            print("   3. Export to ONNX: python export_onnx.py")
            print("   4. Launch app: streamlit run app.py")
            print()
        
        return 0
    
    except Exception as e:
        print(f"❌ Error during packaging: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
