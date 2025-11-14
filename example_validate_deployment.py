"""
Example: Using Deployment Validation Script
============================================

This example shows how to use the deployment validation system
after training your model.
"""

# ============================================================================
# Step 1: Train Your Model (if not already done)
# ============================================================================

# Option A: Train from scratch
# python train.py

# Option B: Use existing model checkpoint
# Make sure models/cropshield_cnn.pth exists


# ============================================================================
# Step 2: Generate Class Mapping (if not already done)
# ============================================================================

# python generate_class_mapping.py


# ============================================================================
# Step 3: Run Basic Validation
# ============================================================================

# Run all checks with default settings
# python validate_deployment.py

# Expected output if all checks pass:
"""
🔍 CropShield AI - Deployment Validation
Starting pre-deployment checks...

✅ PASSED | Model file exists
✅ PASSED | Class mapping exists
✅ PASSED | Model loads successfully
✅ PASSED | Output shape correct
✅ PASSED | GradCAM heatmap generated
✅ PASSED | Streamlit installed
✅ PASSED | Average inference time < 200ms

✅ ALL CHECKS PASSED!
🚀 System is ready for deployment!
"""


# ============================================================================
# Step 4: Run Verbose Validation (Recommended)
# ============================================================================

# Get detailed information about each check
# python validate_deployment.py --verbose

# This creates validation_results.json with detailed metrics


# ============================================================================
# Step 5: Test with Different Model Checkpoints
# ============================================================================

# Validate a specific model checkpoint
# python validate_deployment.py --model experiments/best_model.pth


# ============================================================================
# Step 6: CI/CD Integration
# ============================================================================

# Skip Streamlit test for automated pipelines
# python validate_deployment.py --skip-streamlit


# ============================================================================
# Step 7: Custom Performance Targets
# ============================================================================

# For high-end GPU (target 100ms)
# python validate_deployment.py --target-time 100

# For CPU deployment (target 1000ms)
# python validate_deployment.py --target-time 1000


# ============================================================================
# Step 8: Programmatic Usage
# ============================================================================

if __name__ == "__main__":
    # Import the validation function
    from validate_deployment import run_validation
    
    # Run validation programmatically
    success = run_validation(
        model_path='models/cropshield_cnn.pth',
        app_path='app_optimized.py',
        skip_streamlit=False,
        target_time_ms=200.0,
        verbose=True
    )
    
    if success:
        print("\n✅ All checks passed! Ready to deploy.")
        
        # Your deployment code here
        # deploy_to_production()
        
    else:
        print("\n❌ Validation failed! Fix issues before deploying.")
        
        # Your error handling here
        # send_alert_to_team()


# ============================================================================
# Step 9: Continuous Monitoring
# ============================================================================

def setup_continuous_validation():
    """
    Example: Run validation periodically in production.
    """
    import schedule
    import time
    from validate_deployment import run_validation
    
    def health_check():
        print("Running periodic health check...")
        result = run_validation(skip_streamlit=True, verbose=False)
        
        if not result:
            print("⚠️ Health check failed! Sending alert...")
            # send_alert("Deployment validation failed!")
        else:
            print("✅ Health check passed!")
    
    # Run every hour
    schedule.every(1).hour.do(health_check)
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)


# ============================================================================
# Step 10: Integration with Deployment Script
# ============================================================================

def deploy_with_validation():
    """
    Example: Validate before deploying.
    """
    from validate_deployment import run_validation
    import subprocess
    
    print("🔍 Running pre-deployment validation...")
    
    # Validate system
    success = run_validation(
        model_path='models/cropshield_cnn.pth',
        app_path='app_optimized.py',
        skip_streamlit=False,
        target_time_ms=200.0,
        verbose=True
    )
    
    if not success:
        print("❌ Validation failed! Deployment aborted.")
        return False
    
    print("\n✅ Validation passed! Starting deployment...")
    
    # Deploy to production
    try:
        # Example: Start Streamlit app
        subprocess.Popen([
            'streamlit', 'run', 'app_optimized.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0'
        ])
        
        print("🚀 Deployment successful!")
        return True
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return False


# ============================================================================
# Common Usage Patterns
# ============================================================================

"""
# Pattern 1: Quick Check
python validate_deployment.py

# Pattern 2: Detailed Check with Results Export
python validate_deployment.py --verbose

# Pattern 3: CI/CD Pipeline
python validate_deployment.py --skip-streamlit || exit 1

# Pattern 4: Custom Model
python validate_deployment.py --model experiments/best_model.pth

# Pattern 5: Custom Target
python validate_deployment.py --target-time 150

# Pattern 6: Combined Options
python validate_deployment.py \
  --model models/production_model.pth \
  --app app_optimized.py \
  --target-time 100 \
  --verbose
"""


# ============================================================================
# Troubleshooting
# ============================================================================

"""
Issue 1: Model Not Found
-------------------------
Error: ❌ FAILED | Model file exists
       Path not found: models/cropshield_cnn.pth

Solution:
1. Train model: python train.py
2. Or specify correct path: python validate_deployment.py --model path/to/model.pth


Issue 2: Class Mapping Missing
-------------------------------
Error: ❌ FAILED | Class mapping exists
       Path not found: models/class_to_idx.json

Solution:
python generate_class_mapping.py


Issue 3: Wrong Output Shape
----------------------------
Error: ❌ FAILED | Output shape correct
       Got [1, 10], Expected [1, 22]

Solution:
- Model trained on different dataset
- Retrain model with correct dataset
- Check model architecture matches num_classes


Issue 4: GradCAM Import Error
------------------------------
Error: ❌ FAILED | GradCAM module imports
       Import error: No module named 'cv2'

Solution:
pip install opencv-python


Issue 5: Performance Too Slow
------------------------------
Error: ❌ FAILED | Average inference time < 200ms
       Avg: 450ms

Solutions:
- Check GPU: torch.cuda.is_available()
- Use optimized app: app_optimized.py
- Adjust target: python validate_deployment.py --target-time 500
"""
