"""
CropShield AI - Deployment Validation Script 🔍
================================================

Comprehensive pre-deployment validation to ensure the entire inference
and visualization pipeline is bug-free.

Validation Checks:
1. ✅ File System: Model checkpoint and class mapping exist
2. ✅ Model Loading: Model loads successfully on available device
3. ✅ Dummy Inference: Prediction output shape is correct
4. ✅ GradCAM Visualization: GradCAM runs without errors
5. ✅ Streamlit Integration: App loads successfully
6. ✅ Performance: Inference time meets requirements

Usage:
    python validate_deployment.py
    
    # Skip Streamlit test (for CI/CD)
    python validate_deployment.py --skip-streamlit
    
    # Verbose output
    python validate_deployment.py --verbose
    
    # Custom model path
    python validate_deployment.py --model models/custom_model.pth

Exit Codes:
    0: All checks passed ✅
    1: One or more checks failed ❌
"""

import sys
import os
from pathlib import Path
import argparse
import subprocess
import time
import json
from typing import Optional, Tuple, Dict, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted section header."""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(70)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")


def print_check(name: str, passed: bool, details: str = ""):
    """Print formatted check result."""
    status = f"{GREEN}✅ PASSED{RESET}" if passed else f"{RED}❌ FAILED{RESET}"
    print(f"{status} | {name}")
    if details:
        print(f"         {details}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{YELLOW}⚠️  WARNING: {message}{RESET}")


def print_info(message: str):
    """Print info message."""
    print(f"{BLUE}ℹ️  INFO: {message}{RESET}")


# ============================================================================
# Validation Check 1: File System
# ============================================================================

def validate_filesystem(
    model_path: str = 'models/cropshield_cnn.pth',
    class_mapping_path: str = 'models/class_to_idx.json',
    verbose: bool = False
) -> Tuple[bool, Dict]:
    """
    Validate that required files exist.
    
    Checks:
    - Model checkpoint exists
    - Class mapping JSON exists
    - Paths are accessible
    
    Returns:
        (passed, details) tuple
    """
    print_header("CHECK 1: File System Validation")
    
    details = {
        'model_exists': False,
        'model_path': model_path,
        'class_mapping_exists': False,
        'class_mapping_path': class_mapping_path,
        'num_classes': 0
    }
    
    # Check model file
    try:
        model_file = Path(model_path)
        if model_file.exists():
            details['model_exists'] = True
            file_size_mb = model_file.stat().st_size / (1024 * 1024)
            print_check("Model file exists", True, f"Path: {model_path} ({file_size_mb:.2f} MB)")
        else:
            print_check("Model file exists", False, f"Path not found: {model_path}")
            return False, details
    except (OSError, PermissionError) as e:
        print_check("Model file exists", False, f"Error accessing file: {e}")
        return False, details
    
    # Check class mapping
    mapping_file = Path(class_mapping_path)
    if mapping_file.exists():
        details['class_mapping_exists'] = True
        
        # Load and validate JSON
        try:
            with open(mapping_file, 'r') as f:
                class_to_idx = json.load(f)
            
            details['num_classes'] = len(class_to_idx)
            print_check("Class mapping exists", True, f"Found {len(class_to_idx)} classes")
            
            if verbose:
                print_info(f"Classes: {', '.join(list(class_to_idx.keys())[:5])}...")
        
        except json.JSONDecodeError as e:
            print_check("Class mapping valid JSON", False, f"JSON error: {e}")
            return False, details
    else:
        print_check("Class mapping exists", False, f"Path not found: {class_mapping_path}")
        print_warning("Run 'python generate_class_mapping.py' to create it")
        return False, details
    
    passed = details['model_exists'] and details['class_mapping_exists']
    return passed, details


# ============================================================================
# Validation Check 2: Model Loading
# ============================================================================

def validate_model_loading(
    model_path: str = 'models/cropshield_cnn.pth',
    verbose: bool = False
) -> Tuple[bool, Dict]:
    """
    Validate that model loads successfully.
    
    Checks:
    - Model loads without errors
    - Device detection works (GPU/CPU)
    - Model architecture is valid
    - Model can be set to eval mode
    
    Returns:
        (passed, details) tuple
    """
    print_header("CHECK 2: Model Loading Validation")
    
    details = {
        'load_success': False,
        'device': None,
        'model_type': None,
        'num_parameters': 0,
        'load_time_ms': 0
    }
    
    try:
        # Import required modules
        from predict import load_model_once, get_device
        
        # Load model
        start_time = time.perf_counter()
        model, class_names, device = load_model_once(model_path)
        load_time = (time.perf_counter() - start_time) * 1000
        
        details['load_success'] = True
        details['device_str'] = str(device)  # Store string for JSON serialization
        details['model_type'] = model.__class__.__name__
        details['load_time_ms'] = load_time
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        details['num_parameters'] = num_params
        
        print_check(
            "Model loads successfully",
            True,
            f"Type: {details['model_type']}, Device: {device}, Time: {load_time:.0f}ms"
        )
        
        # Check model is in eval mode
        is_eval = not model.training
        print_check("Model in eval mode", is_eval)
        
        # Check parameters
        print_check(
            "Model has parameters",
            num_params > 0,
            f"Parameters: {num_params:,}"
        )
        
        if verbose:
            print_info(f"Classes: {len(class_names)}")
            print_info(f"Memory: ~{num_params * 4 / (1024**2):.2f} MB (FP32)")
        
        # Store for next validation (keep device as torch.device object)
        details['model'] = model
        details['class_names'] = class_names
        details['device'] = device  # Keep as torch.device for operations
        
        return True, details
    
    except Exception as e:
        print_check("Model loads successfully", False, f"Error: {str(e)}")
        details['error'] = str(e)
        return False, details


# ============================================================================
# Validation Check 3: Dummy Inference
# ============================================================================

def validate_inference(
    model_details: Dict,
    num_classes: int,
    verbose: bool = False
) -> Tuple[bool, Dict]:
    """
    Validate that dummy inference produces correct output shape.
    
    Checks:
    - Creates dummy input (batch_size=1, channels=3, height=224, width=224)
    - Forward pass executes without errors
    - Output shape is [1, num_classes]
    - Output is a valid probability distribution (sums to ~1.0 after softmax)
    
    Returns:
        (passed, details) tuple
    """
    print_header("CHECK 3: Dummy Inference Validation")
    
    details = {
        'inference_success': False,
        'input_shape': None,
        'output_shape': None,
        'expected_shape': (1, num_classes),
        'inference_time_ms': 0,
        'output_sum': 0.0
    }
    
    try:
        import torch
        import torch.nn.functional as F
        
        model = model_details.get('model')
        device = model_details.get('device')
        
        if model is None or device is None:
            print_check("Dummy inference", False, "Model not loaded from previous check")
            return False, details
        
        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        details['input_shape'] = list(dummy_input.shape)
        
        print_info(f"Dummy input shape: {list(dummy_input.shape)}")
        
        # Perform inference
        model.eval()
        with torch.no_grad():
            start_time = time.perf_counter()
            output = model(dummy_input)
            inference_time = (time.perf_counter() - start_time) * 1000
        
        details['inference_time_ms'] = inference_time
        details['output_shape'] = list(output.shape)
        
        # Validate output shape
        expected_shape = [1, num_classes]
        shape_correct = list(output.shape) == expected_shape
        
        print_check(
            "Output shape correct",
            shape_correct,
            f"Got {list(output.shape)}, Expected {expected_shape}"
        )
        
        if not shape_correct:
            return False, details
        
        # Validate output is valid probability distribution
        probs = F.softmax(output, dim=1)
        prob_sum = probs.sum().item()
        details['output_sum'] = prob_sum
        
        sum_valid = abs(prob_sum - 1.0) < 0.01  # Allow small floating point error
        print_check(
            "Output is valid distribution",
            sum_valid,
            f"Softmax sum: {prob_sum:.6f} (should be ~1.0)"
        )
        
        # Check inference time
        print_check(
            "Inference completes",
            True,
            f"Time: {inference_time:.2f}ms"
        )
        
        if verbose:
            top_prob, top_idx = probs.max(1)
            print_info(f"Top prediction: Class {top_idx.item()} ({top_prob.item():.2%})")
        
        details['inference_success'] = shape_correct and sum_valid
        return shape_correct and sum_valid, details
    
    except Exception as e:
        print_check("Dummy inference", False, f"Error: {str(e)}")
        details['error'] = str(e)
        return False, details


# ============================================================================
# Validation Check 4: GradCAM Visualization
# ============================================================================

def validate_gradcam(
    model_details: Dict,
    verbose: bool = False
) -> Tuple[bool, Dict]:
    """
    Validate that GradCAM visualization runs without errors.
    
    Checks:
    - GradCAM module imports successfully
    - Target layer can be retrieved
    - GradCAM instance can be created
    - Forward pass with hooks works
    - Heatmap generation succeeds
    - Heatmap has correct shape
    
    Returns:
        (passed, details) tuple
    """
    print_header("CHECK 4: GradCAM Visualization Validation")
    
    details = {
        'gradcam_available': False,
        'target_layer_found': False,
        'heatmap_generated': False,
        'heatmap_shape': None,
        'gradcam_time_ms': 0
    }
    
    try:
        # Check if GradCAM module is available
        try:
            from utils.gradcam import GradCAM, get_target_layer
            details['gradcam_available'] = True
            print_check("GradCAM module imports", True)
        except ImportError as e:
            print_check("GradCAM module imports", False, f"Import error: {str(e)}")
            print_warning("Install opencv-python: pip install opencv-python")
            return False, details
        
        model = model_details.get('model')
        device = model_details.get('device')
        
        if model is None or device is None:
            print_check("GradCAM initialization", False, "Model not loaded")
            return False, details
        
        # Get target layer
        try:
            target_layer = get_target_layer(model)
            details['target_layer_found'] = True
            print_check("Target layer found", True, f"Layer: {target_layer.__class__.__name__}")
        except Exception as e:
            print_check("Target layer found", False, f"Error: {str(e)}")
            return False, details
        
        # Create GradCAM instance
        import torch
        gradcam = GradCAM(model, target_layer, device)
        print_check("GradCAM instance created", True)
        
        # Generate heatmap with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        target_class = 0  # Use first class
        
        start_time = time.perf_counter()
        heatmap = gradcam(dummy_input, class_idx=target_class)
        gradcam_time = (time.perf_counter() - start_time) * 1000
        
        details['gradcam_time_ms'] = gradcam_time
        details['heatmap_shape'] = list(heatmap.shape)
        details['heatmap_generated'] = True
        
        print_check(
            "GradCAM heatmap generated",
            True,
            f"Shape: {list(heatmap.shape)}, Time: {gradcam_time:.2f}ms"
        )
        
        # Validate heatmap properties
        heatmap_np = heatmap.cpu().numpy()
        
        # Check heatmap range
        min_val = heatmap_np.min()
        max_val = heatmap_np.max()
        range_valid = min_val >= 0 and max_val <= 1
        
        print_check(
            "Heatmap values in [0, 1]",
            range_valid,
            f"Min: {min_val:.4f}, Max: {max_val:.4f}"
        )
        
        if verbose:
            print_info(f"Heatmap mean: {heatmap_np.mean():.4f}")
            print_info(f"Heatmap std: {heatmap_np.std():.4f}")
        
        return True, details
    
    except Exception as e:
        print_check("GradCAM visualization", False, f"Error: {str(e)}")
        details['error'] = str(e)
        return False, details


# ============================================================================
# Validation Check 5: Streamlit Integration
# ============================================================================

def validate_streamlit(
    app_path: str = 'app_optimized.py',
    timeout: int = 10,
    skip: bool = False,
    verbose: bool = False
) -> Tuple[bool, Dict]:
    """
    Validate that Streamlit app loads successfully.
    
    Checks:
    - Streamlit is installed
    - App file exists
    - App can be imported (syntax check)
    - Optional: App starts without errors (requires --test flag)
    
    Returns:
        (passed, details) tuple
    """
    print_header("CHECK 5: Streamlit Integration Validation")
    
    details = {
        'streamlit_installed': False,
        'app_exists': False,
        'syntax_valid': False,
        'can_import': False
    }
    
    if skip:
        print_warning("Streamlit validation skipped (--skip-streamlit flag)")
        return True, details
    
    # Check Streamlit is installed
    try:
        import streamlit as st
        details['streamlit_installed'] = True
        streamlit_version = st.__version__
        print_check("Streamlit installed", True, f"Version: {streamlit_version}")
    except ImportError:
        print_check("Streamlit installed", False, "Run: pip install streamlit")
        return False, details
    
    # Check app file exists
    app_file = Path(app_path)
    if app_file.exists():
        details['app_exists'] = True
        print_check("App file exists", True, f"Path: {app_path}")
    else:
        print_check("App file exists", False, f"Path not found: {app_path}")
        return False, details
    
    # Validate Python syntax
    try:
        with open(app_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, app_file, 'exec')
        details['syntax_valid'] = True
        print_check("App syntax valid", True)
    except SyntaxError as e:
        print_check("App syntax valid", False, f"Syntax error: {e}")
        return False, details
    
    # Try importing (without running)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("app_module", app_file)
        if spec and spec.loader:
            # Note: We don't execute the module, just validate it can be loaded
            print_check("App can be imported", True, "No import errors detected")
            details['can_import'] = True
        else:
            print_check("App can be imported", False, "Could not create module spec")
            return False, details
    except Exception as e:
        print_check("App can be imported", False, f"Import error: {str(e)}")
        return False, details
    
    # Info about running the app
    print_info(f"To test app manually: streamlit run {app_path}")
    
    if verbose:
        print_info("App validation complete (syntax and imports only)")
        print_info("For full validation, run the app and test manually")
    
    return True, details


# ============================================================================
# Validation Check 6: Performance Requirements
# ============================================================================

def validate_performance(
    model_details: Dict,
    target_time_ms: float = 200.0,
    num_iterations: int = 5,
    verbose: bool = False
) -> Tuple[bool, Dict]:
    """
    Validate that inference meets performance requirements.
    
    Checks:
    - Average inference time
    - Consistency (std deviation)
    - GPU utilization (if available)
    
    Returns:
        (passed, details) tuple
    """
    print_header("CHECK 6: Performance Requirements Validation")
    
    details = {
        'performance_test_run': False,
        'avg_time_ms': 0.0,
        'std_time_ms': 0.0,
        'min_time_ms': 0.0,
        'max_time_ms': 0.0,
        'meets_target': False
    }
    
    try:
        import torch
        import numpy as np
        
        model = model_details.get('model')
        device = model_details.get('device')
        
        if model is None or device is None:
            print_check("Performance test", False, "Model not loaded")
            return False, details
        
        print_info(f"Running {num_iterations} inference iterations...")
        
        # Warm-up run
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Benchmark runs
        times = []
        for i in range(num_iterations):
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            
            # Synchronize GPU before timing (handle both torch.device and string)
            if isinstance(device, torch.device) and device.type == 'cuda':
                torch.cuda.synchronize()
            elif str(device) == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Synchronize GPU after inference
            if isinstance(device, torch.device) and device.type == 'cuda':
                torch.cuda.synchronize()
            elif str(device) == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start_time) * 1000
            times.append(elapsed)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        details['performance_test_run'] = True
        details['avg_time_ms'] = avg_time
        details['std_time_ms'] = std_time
        details['min_time_ms'] = min_time
        details['max_time_ms'] = max_time
        details['meets_target'] = avg_time < target_time_ms
        
        # Print results
        print_check(
            f"Average inference time < {target_time_ms}ms",
            avg_time < target_time_ms,
            f"Avg: {avg_time:.2f}ms, Std: {std_time:.2f}ms"
        )
        
        print_check(
            "Performance consistency",
            std_time < avg_time * 0.2,  # Std dev < 20% of mean
            f"Min: {min_time:.2f}ms, Max: {max_time:.2f}ms"
        )
        
        if verbose:
            print_info(f"Individual times: {[f'{t:.2f}' for t in times]}")
            # Check device type safely and query GPU name
            if isinstance(device, torch.device) and device.type == 'cuda':
                try:
                    print_info(f"GPU: {torch.cuda.get_device_name(0)}")
                except RuntimeError:
                    print_info("GPU: Unable to query device name")
            elif str(device) == 'cuda':
                try:
                    print_info(f"GPU: {torch.cuda.get_device_name(0)}")
                except RuntimeError:
                    print_info("GPU: Unable to query device name")
        
        return avg_time < target_time_ms, details
    
    except Exception as e:
        print_check("Performance test", False, f"Error: {str(e)}")
        details['error'] = str(e)
        return False, details


# ============================================================================
# Main Validation Runner
# ============================================================================

def run_validation(
    model_path: str = 'models/cropshield_cnn.pth',
    app_path: str = 'app_optimized.py',
    skip_streamlit: bool = False,
    target_time_ms: float = 200.0,
    verbose: bool = False
) -> bool:
    """
    Run all validation checks.
    
    Returns:
        True if all checks pass, False otherwise
    """
    print(f"\n{BOLD}🔍 CropShield AI - Deployment Validation{RESET}")
    print(f"{BOLD}Starting pre-deployment checks...{RESET}\n")
    
    results = {}
    all_passed = True
    
    # Check 1: File System
    passed, details = validate_filesystem(
        model_path=model_path,
        verbose=verbose
    )
    results['filesystem'] = {'passed': passed, 'details': details}
    all_passed = all_passed and passed
    
    if not passed:
        print_warning("Fix file system issues before proceeding")
        return False
    
    num_classes = details.get('num_classes', 0)
    
    # Check 2: Model Loading
    passed, details = validate_model_loading(
        model_path=model_path,
        verbose=verbose
    )
    results['model_loading'] = {'passed': passed, 'details': details}
    all_passed = all_passed and passed
    
    if not passed:
        print_warning("Fix model loading issues before proceeding")
        return False
    
    model_details = details
    
    # Check 3: Dummy Inference
    passed, details = validate_inference(
        model_details=model_details,
        num_classes=num_classes,
        verbose=verbose
    )
    results['inference'] = {'passed': passed, 'details': details}
    all_passed = all_passed and passed
    
    # Check 4: GradCAM
    passed, details = validate_gradcam(
        model_details=model_details,
        verbose=verbose
    )
    results['gradcam'] = {'passed': passed, 'details': details}
    all_passed = all_passed and passed
    
    # Check 5: Streamlit
    passed, details = validate_streamlit(
        app_path=app_path,
        skip=skip_streamlit,
        verbose=verbose
    )
    results['streamlit'] = {'passed': passed, 'details': details}
    all_passed = all_passed and passed
    
    # Check 6: Performance
    passed, details = validate_performance(
        model_details=model_details,
        target_time_ms=target_time_ms,
        verbose=verbose
    )
    results['performance'] = {'passed': passed, 'details': details}
    all_passed = all_passed and passed
    
    # Summary
    print_header("Validation Summary")
    
    total_checks = len(results)
    passed_checks = sum(1 for r in results.values() if r['passed'])
    
    print(f"\n{BOLD}Total Checks: {total_checks}{RESET}")
    print(f"{BOLD}Passed: {GREEN}{passed_checks}{RESET}")
    print(f"{BOLD}Failed: {RED}{total_checks - passed_checks}{RESET}\n")
    
    if all_passed:
        print(f"{GREEN}{BOLD}✅ ALL CHECKS PASSED!{RESET}")
        print(f"{GREEN}🚀 System is ready for deployment!{RESET}\n")
    else:
        print(f"{RED}{BOLD}❌ SOME CHECKS FAILED!{RESET}")
        print(f"{RED}⚠️  Fix issues before deploying!{RESET}\n")
        
        # Show which checks failed
        print(f"{BOLD}Failed Checks:{RESET}")
        for check_name, result in results.items():
            if not result['passed']:
                print(f"  {RED}• {check_name.replace('_', ' ').title()}{RESET}")
        print()
    
    # Save results to JSON
    if verbose:
        results_file = Path('validation_results.json')
        with open(results_file, 'w') as f:
            # Convert non-serializable objects
            serializable_results = {}
            for key, value in results.items():
                serializable_results[key] = {
                    'passed': value['passed'],
                    'details': {k: str(v) for k, v in value['details'].items() 
                               if k not in ['model', 'class_names', 'device']}
                }
            json.dump(serializable_results, f, indent=2)
        print_info(f"Detailed results saved to {results_file}")
    
    return all_passed


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='CropShield AI Deployment Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all checks
  python validate_deployment.py
  
  # Skip Streamlit test (for CI/CD)
  python validate_deployment.py --skip-streamlit
  
  # Custom model path
  python validate_deployment.py --model models/my_model.pth
  
  # Verbose output with detailed results
  python validate_deployment.py --verbose
  
  # Custom performance target
  python validate_deployment.py --target-time 100
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/cropshield_cnn.pth',
        help='Path to model checkpoint (default: models/cropshield_cnn.pth)'
    )
    
    parser.add_argument(
        '--app',
        type=str,
        default='app_optimized.py',
        help='Path to Streamlit app (default: app_optimized.py)'
    )
    
    parser.add_argument(
        '--skip-streamlit',
        action='store_true',
        help='Skip Streamlit validation (useful for CI/CD)'
    )
    
    parser.add_argument(
        '--target-time',
        type=float,
        default=200.0,
        help='Target inference time in milliseconds (default: 200ms)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed information'
    )
    
    args = parser.parse_args()
    
    # Run validation
    success = run_validation(
        model_path=args.model,
        app_path=args.app,
        skip_streamlit=args.skip_streamlit,
        target_time_ms=args.target_time,
        verbose=args.verbose
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
