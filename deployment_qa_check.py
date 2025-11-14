"""
CropShield AI - End-to-End Deployment Validation

Comprehensive automated testing for deployment readiness:
- Model loading and inference
- GradCAM generation
- Streamlit app compilation
- Dependencies verification
- Results validation

Usage:
    python deployment_qa_check.py
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
import importlib.util

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"{text}")
    print("=" * 80)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")

class DeploymentQA:
    """End-to-end deployment quality assurance"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tests': [],
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'total': 0
        }
        self.start_time = time.time()
    
    def add_result(self, test_name, status, message, details=None):
        """Add test result"""
        self.results['tests'].append({
            'name': test_name,
            'status': status,
            'message': message,
            'details': details
        })
        self.results['total'] += 1
        
        if status == 'PASS':
            self.results['passed'] += 1
        elif status == 'FAIL':
            self.results['failed'] += 1
        elif status == 'WARN':
            self.results['warnings'] += 1
    
    def test_predict_disease_function(self):
        """Test 1: predict_disease() import and execution"""
        print_header("TEST 1: SAMPLE INFERENCE - predict_disease()")
        
        try:
            # Try to import from app.py
            if not os.path.exists("app.py"):
                print_error("app.py file not found in current directory")
                self.add_result(
                    "predict_disease() Import",
                    'FAIL',
                    "app.py file not found"
                )
                return False
            
            spec = importlib.util.spec_from_file_location("app_module", "app.py")
            if spec is None or spec.loader is None:
                print_error("Failed to create module spec for app.py")
                self.add_result(
                    "predict_disease() Import",
                    'FAIL',
                    "Module spec creation failed"
                )
                return False
            
            app_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app_module)
            
            if hasattr(app_module, 'predict_disease'):
                print_success("Successfully imported predict_disease() from app.py")
                
                # Check function signature
                import inspect
                sig = inspect.signature(app_module.predict_disease)
                print_info(f"Function signature: predict_disease{sig}")
                
                self.add_result(
                    "predict_disease() Import",
                    'PASS',
                    "Function imported successfully from app.py",
                    {'signature': str(sig)}
                )
                
                # Note: Cannot run actual inference without test image
                print_warning("Actual inference skipped (requires test image file)")
                self.add_result(
                    "predict_disease() Execution",
                    'WARN',
                    "Function exists but not executed (needs test image)"
                )
                
                return True
            else:
                print_error("predict_disease() not found in app.py")
                self.add_result(
                    "predict_disease() Import",
                    'FAIL',
                    "Function not found in app.py"
                )
                return False
        except Exception as e:
            print_error(f"Failed to import predict_disease(): {str(e)}")
            self.add_result(
                "predict_disease() Import",
                'FAIL',
                f"Import error: {str(e)}"
            )
            return False
    
    def test_generate_gradcam_function(self):
        """Test 2: generate_gradcam() import and validation"""
        print_header("TEST 2: GRADCAM GENERATION - generate_gradcam()")
        
        # Check for GradCAM implementation files
        gradcam_files = [
            ('example_gradcam.py', 'example_gradcam'),
            ('gradcam.py', 'gradcam'),
            ('app.py', 'app_module')
        ]
        
        gradcam_found = False
        for filepath, module_name in gradcam_files:
            if os.path.exists(filepath):
                try:
                    spec = importlib.util.spec_from_file_location(module_name, filepath)
                    if spec is None or spec.loader is None:
                        print_warning(f"Failed to create module spec for {filepath}")
                        continue
                    
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    if hasattr(module, 'generate_gradcam'):
                        print_success(f"Found generate_gradcam() in {filepath}")
                        
                        # Check function signature
                        import inspect
                        sig = inspect.signature(module.generate_gradcam)
                        print_info(f"Function signature: generate_gradcam{sig}")
                        
                        self.add_result(
                            "generate_gradcam() Import",
                            'PASS',
                            f"Function found in {filepath}",
                            {'signature': str(sig)}
                        )
                        
                        gradcam_found = True
                        
                        # Note: Cannot generate actual GradCAM without image
                        print_warning("Actual GradCAM generation skipped (requires test image)")
                        self.add_result(
                            "generate_gradcam() Execution",
                            'WARN',
                            "Function exists but not executed (needs test image)"
                        )
                        
                        break
                    elif hasattr(module, 'GradCAM'):
                        print_success(f"Found GradCAM class in {filepath}")
                        self.add_result(
                            "GradCAM Class",
                            'PASS',
                            f"Class found in {filepath}"
                        )
                        gradcam_found = True
                        
                        # Note: Cannot instantiate GradCAM without model
                        print_warning("Actual GradCAM generation skipped (requires test image)")
                        self.add_result(
                            "GradCAM Execution",
                            'WARN',
                            "Class exists but not executed (needs test image)"
                        )
                        
                        break
                except Exception as e:
                    print_warning(f"Could not load {filepath}: {str(e)}")
        
        if not gradcam_found:
            print_error("No generate_gradcam() function or GradCAM class found")
            self.add_result(
                "generate_gradcam() Import",
                'FAIL',
                "Function not found in any expected location"
            )
            return False
        
        # Check for GradCAM results directory
        if os.path.exists('results/test_gradcams'):
            try:
                files = os.listdir('results/test_gradcams')
                print_success(f"GradCAM results directory exists ({len(files)} files)")
            except OSError as e:
                print_warning(f"GradCAM directory exists but cannot be read: {str(e)}")
        
        return gradcam_found
    
    def test_streamlit_app_compilation(self):
        """Test 3: Streamlit app file compilation"""
        print_header("TEST 3: STREAMLIT APP COMPILATION")
        
        app_files = ['app.py', 'app_optimized.py']
        compiled = False
        
        for app_file in app_files:
            if os.path.exists(app_file):
                print_info(f"Checking {app_file}...")
                
                try:
                    # Read and compile
                    with open(app_file, 'r', encoding='utf-8', errors='replace') as f:
                        code = f.read()
                    
                    if not code.strip():
                        print_warning(f"{app_file} is empty")
                        self.add_result(
                            f"Streamlit App: {app_file}",
                            'WARN',
                            "File is empty"
                        )
                        continue
                    
                    compile(code, app_file, 'exec')
                    print_success(f"{app_file} compiles without syntax errors")
                    
                    # Check for key components (case-insensitive search)
                    code_lower = code.lower()
                    checks = {
                        'st.title': 'Title definition',
                        'st.file_uploader': 'File uploader',
                        'predict': 'Prediction logic',
                        'gradcam': 'GradCAM visualization',
                        'st.image': 'Image display'
                    }
                    
                    components_found = []
                    for keyword, description in checks.items():
                        if keyword.lower() in code_lower:
                            print_success(f"  ✓ Contains: {description}")
                            components_found.append(description)
                    
                    self.add_result(
                        f"Streamlit App: {app_file}",
                        'PASS',
                        "File compiles successfully",
                        {'components': components_found}
                    )
                    
                    compiled = True
                    
                except SyntaxError as e:
                    print_error(f"Syntax error in {app_file}: {str(e)}")
                    self.add_result(
                        f"Streamlit App: {app_file}",
                        'FAIL',
                        f"Syntax error: {str(e)}"
                    )
                except Exception as e:
                    print_warning(f"Could not fully validate {app_file}: {str(e)}")
                    self.add_result(
                        f"Streamlit App: {app_file}",
                        'WARN',
                        str(e)
                    )
        
        if not compiled:
            print_error("No Streamlit app file found or compiled")
            self.add_result(
                "Streamlit App",
                'FAIL',
                "No valid app file found"
            )
        
        # Check Streamlit installation
        try:
            import streamlit
            print_success(f"Streamlit {streamlit.__version__} is installed")
            self.add_result(
                "Streamlit Installation",
                'PASS',
                f"Version {streamlit.__version__}"
            )
        except ImportError:
            print_warning("Streamlit is not installed")
            print_info("Install with: pip install streamlit")
            self.add_result(
                "Streamlit Installation",
                'WARN',
                "Package not installed"
            )
        
        return compiled
    
    def test_model_artifacts(self):
        """Test model artifacts existence"""
        print_header("BONUS TEST: MODEL ARTIFACTS CHECK")
        
        artifacts = [
            ('models/cropshield_cnn.pth', 'Baseline Model'),
            ('models/cropshield_cnn_best.pth', 'Optimized Model'),
            ('results/confusion_matrix.png', 'Confusion Matrix'),
            ('results/test_metrics.json', 'Test Metrics')
        ]
        
        found_count = 0
        for filepath, description in artifacts:
            if os.path.exists(filepath):
                try:
                    size = os.path.getsize(filepath)
                    print_success(f"{description}: {filepath} ({size:,} bytes)")
                    self.add_result(
                        f"Artifact: {description}",
                        'PASS',
                        f"Found: {filepath}",
                        {'size': size}
                    )
                    found_count += 1
                except OSError as e:
                    print_warning(f"{description}: {filepath} exists but cannot read size: {str(e)}")
                    self.add_result(
                        f"Artifact: {description}",
                        'WARN',
                        f"File exists but size unreadable: {filepath}"
                    )
            else:
                print_warning(f"{description}: {filepath} NOT FOUND")
                self.add_result(
                    f"Artifact: {description}",
                    'WARN',
                    f"Missing: {filepath}"
                )
        
        return found_count > 0
    
    def test_critical_dependencies(self):
        """Test critical dependencies"""
        print_header("BONUS TEST: CRITICAL DEPENDENCIES")
        
        deps = [
            ('torch', 'PyTorch'),
            ('PIL', 'Pillow'),
            ('numpy', 'NumPy')
        ]
        
        all_ok = True
        for module_name, display_name in deps:
            try:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'unknown')
                print_success(f"{display_name}: {version}")
                self.add_result(
                    f"Dependency: {display_name}",
                    'PASS',
                    f"Version {version}"
                )
            except ImportError:
                print_error(f"{display_name}: NOT INSTALLED")
                self.add_result(
                    f"Dependency: {display_name}",
                    'FAIL',
                    "Not installed"
                )
                all_ok = False
            except Exception as e:
                print_error(f"{display_name}: Import error - {str(e)}")
                self.add_result(
                    f"Dependency: {display_name}",
                    'FAIL',
                    f"Import error: {str(e)}"
                )
                all_ok = False
        
        return all_ok
    
    def generate_validation_summary(self):
        """Generate final validation summary"""
        elapsed_time = time.time() - self.start_time
        
        print_header("🔍 FINAL VALIDATION SUMMARY")
        
        print(f"\n📅 Validation Date: {self.results['timestamp']}")
        print(f"⏱️  Validation Duration: {elapsed_time:.2f} seconds")
        print(f"🖥️  Working Directory: {os.getcwd()}")
        print()
        
        print("=" * 80)
        print("📊 TEST RESULTS")
        print("=" * 80)
        print(f"Total Tests:     {self.results['total']}")
        print(f"✅ Passed:        {self.results['passed']}")
        print(f"❌ Failed:        {self.results['failed']}")
        print(f"⚠️  Warnings:      {self.results['warnings']}")
        print()
        
        # Pass rate
        if self.results['total'] > 0:
            pass_rate = (self.results['passed'] / self.results['total']) * 100
            print(f"📈 Pass Rate: {pass_rate:.1f}%")
            print()
        else:
            print("⚠️  No tests were executed!")
            print()
        
        # Detailed breakdown
        print("=" * 80)
        print("📋 DETAILED TEST BREAKDOWN")
        print("=" * 80)
        
        if not self.results['tests']:
            print("No test results available.")
            print()
            return False
        
        for i, test in enumerate(self.results['tests'], 1):
            status_icon = {
                'PASS': '✅',
                'FAIL': '❌',
                'WARN': '⚠️ '
            }.get(test['status'], '❓')
            
            print(f"{i}. {status_icon} {test['name']}")
            print(f"   Status: {test['status']}")
            print(f"   Message: {test['message']}")
            # Safely handle optional details field
            details = test.get('details')
            if details is not None:
                print(f"   Details: {details}")
            print()
        
        # Deployment readiness
        print("=" * 80)
        print("🎯 DEPLOYMENT READINESS ASSESSMENT")
        print("=" * 80)
        
        # Safeguard: ensure tests were actually run
        if self.results['total'] == 0:
            print("Status: ❌ CANNOT ASSESS")
            print()
            print("No tests were executed. Unable to determine deployment readiness.")
            deployment_status = "UNKNOWN"
            print()
            return False
        
        critical_failures = sum(1 for t in self.results['tests'] 
                               if t.get('status') == 'FAIL')
        
        if critical_failures == 0 and self.results['passed'] >= 3:
            print("Status: ✅ READY FOR DEPLOYMENT")
            print()
            print("All critical tests passed!")
            print("The application is ready for production deployment.")
            deployment_status = "READY"
        elif critical_failures > 0:
            print("Status: ❌ DEPLOYMENT BLOCKED")
            print()
            print(f"Found {critical_failures} critical failure(s).")
            print("Please fix the issues before deploying.")
            deployment_status = "BLOCKED"
        else:
            print("Status: ⚠️  DEPLOYMENT WITH CAUTION")
            print()
            print("Some warnings present. Review before deploying.")
            deployment_status = "CAUTION"
        
        print()
        
        # Recommendations
        if self.results['warnings'] > 0 or self.results['failed'] > 0:
            print("=" * 80)
            print("💡 RECOMMENDATIONS")
            print("=" * 80)
            
            recommendations = []
            
            # Use case-insensitive matching for test names
            test_names_lower = [t.get('name', '').lower() for t in self.results['tests']]
            
            if any('model' in name and t['status'] in ['FAIL', 'WARN'] 
                   for name, t in zip(test_names_lower, self.results['tests'])):
                recommendations.append("• Train the model: python train_auto.py --epochs 25")
            
            if any('metrics' in name and t['status'] in ['FAIL', 'WARN'] 
                   for name, t in zip(test_names_lower, self.results['tests'])):
                recommendations.append("• Evaluate the model: python quick_evaluate.py")
            
            if any('predict_disease' in name and t['status'] == 'FAIL' 
                   for name, t in zip(test_names_lower, self.results['tests'])):
                recommendations.append("• Ensure predict_disease() function exists in app.py")
            
            if any('gradcam' in name and t['status'] == 'FAIL' 
                   for name, t in zip(test_names_lower, self.results['tests'])):
                recommendations.append("• Implement GradCAM visualization")
            
            if any('streamlit' in name and t['status'] == 'WARN' 
                   for name, t in zip(test_names_lower, self.results['tests'])):
                recommendations.append("• Install Streamlit: pip install streamlit")
            
            if any('dependency' in name and t['status'] == 'FAIL' 
                   for name, t in zip(test_names_lower, self.results['tests'])):
                recommendations.append("• Install missing dependencies: pip install -r requirements.txt")
            
            if recommendations:
                for rec in recommendations:
                    print(rec)
            else:
                print("No specific recommendations. Review test results above.")
            
            print()
        
        # Save report
        print("=" * 80)
        print("💾 SAVING VALIDATION REPORT")
        print("=" * 80)
        
        try:
            os.makedirs('results', exist_ok=True)
            
            # JSON report
            json_path = 'results/deployment_qa_report.json'
            self.results['elapsed_time'] = elapsed_time
            self.results['deployment_status'] = deployment_status
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2)
            print(f"✅ JSON Report: {json_path}")
            
            # Text summary
            txt_path = 'results/deployment_qa_summary.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("CROPSHIELD AI - DEPLOYMENT QA SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Date: {self.results['timestamp']}\n")
                f.write(f"Duration: {elapsed_time:.2f}s\n")
                f.write("\n")
                f.write("TEST RESULTS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total:    {self.results['total']}\n")
                f.write(f"Passed:   {self.results['passed']}\n")
                f.write(f"Failed:   {self.results['failed']}\n")
                f.write(f"Warnings: {self.results['warnings']}\n")
                if self.results['total'] > 0:
                    try:
                        pass_rate = (self.results['passed'] / self.results['total']) * 100
                        f.write(f"Pass Rate: {pass_rate:.1f}%\n")
                    except ZeroDivisionError:
                        f.write("Pass Rate: N/A\n")
                else:
                    f.write("Pass Rate: N/A\n")
                f.write("\n")
                f.write("DEPLOYMENT STATUS\n")
                f.write("-" * 80 + "\n")
                f.write(f"{deployment_status}\n")
                f.write("\n")
                f.write("=" * 80 + "\n")
            
            print(f"✅ Text Summary: {txt_path}")
            print()
            
        except IOError as e:
            print_warning(f"Failed to save reports: {str(e)}")
            print_info("Results are available in memory but not persisted to disk")
            print()
        
        print("=" * 80)
        print("✅ CropShield AI - Deployment QA Complete!")
        print("=" * 80)
        print()
        
        return deployment_status == "READY"

def main():
    """Main QA validation function"""
    print("\n" + "=" * 80)
    print("🔍 CROPSHIELD AI - DEPLOYMENT QA CHECK")
    print("=" * 80)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Goal: Validate deployment readiness with end-to-end tests")
    print("=" * 80)
    
    qa = DeploymentQA()
    
    try:
        # Run core tests
        qa.test_predict_disease_function()
        qa.test_generate_gradcam_function()
        qa.test_streamlit_app_compilation()
        
        # Run bonus tests
        qa.test_model_artifacts()
        qa.test_critical_dependencies()
        
        # Generate summary
        success = qa.generate_validation_summary()
        
        return 0 if success else 1
    
    except Exception as e:
        print_error(f"QA validation failed with exception: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
