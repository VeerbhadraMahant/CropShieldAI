"""
Example Streamlit App Using app_utils
======================================

Demonstrates how to use utils/app_utils.py functions
for a clean, reusable Streamlit application.

Run with: streamlit run example_streamlit_with_utils.py
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utilities
from utils.app_utils import (
    load_class_names,
    uploaded_file_to_pil,
    display_predictions,
    show_gradcam_overlay,
    create_confidence_indicator,
    get_crop_emoji,
    resize_image,
    save_prediction_history
)

# Import inference (if available)
try:
    from predict import load_model_once, predict_disease
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    st.warning("⚠️ predict.py not found - using mock predictions")

# Import GradCAM (if available)
try:
    from utils.gradcam import generate_gradcam_visualization
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="CropShield AI - Clean Example",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Session State Initialization
# ============================================================================

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'model' not in st.session_state:
    st.session_state['model'] = None

if 'class_names' not in st.session_state:
    st.session_state['class_names'] = None


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_resource
def load_model_cached():
    """Load model once and cache."""
    if INFERENCE_AVAILABLE:
        try:
            model, class_names, device = load_model_once('models/cropshield_cnn.pth')
            return model, class_names, device
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None, None, None
    return None, None, None


def mock_predictions():
    """Generate mock predictions for demo."""
    import random
    
    diseases = [
        'Potato__late_blight',
        'Potato__early_blight',
        'Potato__healthy',
        'Tomato__bacterial_spot',
        'Wheat__leaf_rust'
    ]
    
    # Random confidences that sum to ~1.0
    confs = sorted([random.random() for _ in range(3)], reverse=True)
    total = sum(confs)
    confs = [c / total for c in confs]
    
    return [(diseases[i], confs[i]) for i in range(3)]


# ============================================================================
# Main App
# ============================================================================

def main():
    # Header
    st.title("🌱 CropShield AI - Disease Detection")
    st.markdown("**Example app using clean, reusable utilities**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Configuration")
        
        # Crop selection
        crop_type = st.selectbox(
            "Select Crop Type",
            ["Wheat", "Potato", "Tomato", "Sugarcane"],
            help="Select the type of crop you're analyzing"
        )
        
        # Get crop emoji
        crop_emoji = get_crop_emoji(crop_type)
        st.markdown(f"### {crop_emoji} {crop_type}")
        
        st.markdown("---")
        
        # Visualization options
        st.markdown("### 🎨 Visualization")
        show_gradcam = st.checkbox("Show GradCAM Explainability", value=True)
        use_plotly = st.checkbox("Use Plotly Charts", value=True)
        
        st.markdown("---")
        
        # Model info
        st.markdown("### ℹ️ Model Info")
        if INFERENCE_AVAILABLE:
            st.success("✅ Inference available")
        else:
            st.info("ℹ️ Using mock predictions")
        
        if GRADCAM_AVAILABLE:
            st.success("✅ GradCAM available")
        else:
            st.info("ℹ️ GradCAM not available")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image of your plant",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image showing the affected leaf or plant part"
        )
        
        if uploaded_file:
            # Convert to PIL using utility
            image = uploaded_file_to_pil(uploaded_file)
            
            if image:
                # Display original image
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Show image info
                st.caption(f"📏 Size: {image.size[0]} x {image.size[1]} pixels")
                st.caption(f"🎨 Mode: {image.mode}")
    
    with col2:
        st.markdown("### 🔬 Analysis Results")
        
        if uploaded_file and image:
            # Analyze button
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing image..."):
                    # Get predictions
                    if INFERENCE_AVAILABLE and st.session_state['model'] is None:
                        # Load model
                        model, class_names, device = load_model_cached()
                        st.session_state['model'] = model
                        st.session_state['class_names'] = class_names
                    
                    if INFERENCE_AVAILABLE and st.session_state['model']:
                        # Real inference
                        try:
                            import tempfile
                            
                            # Save to temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                                image.save(tmp.name)
                                tmp_path = tmp.name
                            
                            # Predict
                            predictions = predict_disease(
                                tmp_path,
                                st.session_state['model'],
                                st.session_state['class_names'],
                                device=device,
                                top_k=3
                            )
                            
                            # Clean up
                            os.unlink(tmp_path)
                        
                        except Exception as e:
                            st.error(f"Inference failed: {e}")
                            predictions = mock_predictions()
                    else:
                        # Mock predictions
                        predictions = mock_predictions()
                    
                    # Store in session state
                    st.session_state['last_predictions'] = predictions
                    st.session_state['last_image'] = image
                    
                    # Save to history
                    entry = save_prediction_history(
                        image, 
                        predictions,
                        metadata={'crop': crop_type}
                    )
                    st.session_state['history'].append(entry)
        
        # Display results if available
        if 'last_predictions' in st.session_state:
            predictions = st.session_state['last_predictions']
            
            # Top prediction with confidence indicator
            top_disease, top_conf = predictions[0]
            
            st.markdown("#### 🎯 Top Prediction")
            st.markdown(f"### {top_disease.replace('_', ' ').title()}")
            
            # Confidence indicator using utility
            conf_html = create_confidence_indicator(top_conf)
            st.markdown(conf_html, unsafe_allow_html=True)
    
    # Full width results section
    if 'last_predictions' in st.session_state and 'last_image' in st.session_state:
        st.markdown("---")
        
        # Predictions chart
        st.markdown("### 📊 Detailed Predictions")
        
        # Use utility to display predictions
        display_predictions(
            st.session_state['last_predictions'],
            title="Top 3 Disease Predictions",
            use_plotly=use_plotly,
            height=400,
            color_scale="viridis"
        )
        
        # GradCAM visualization
        if show_gradcam and GRADCAM_AVAILABLE:
            st.markdown("---")
            st.markdown("### 🔥 Model Explainability (GradCAM)")
            st.caption("See which parts of the image influenced the model's decision")
            
            try:
                with st.spinner("Generating GradCAM visualization..."):
                    # Generate GradCAM
                    import tempfile
                    
                    # Save image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        st.session_state['last_image'].save(tmp.name)
                        tmp_path = tmp.name
                    
                    # Generate overlay
                    if st.session_state['model']:
                        overlay = generate_gradcam_visualization(
                            st.session_state['model'],
                            tmp_path,
                            st.session_state['model'].block4,  # Assuming Custom CNN
                            device=device
                        )
                        
                        # Display using utility
                        show_gradcam_overlay(
                            st.session_state['last_image'],
                            overlay,
                            titles=("📷 Original Image", "🔥 Model Focus Areas"),
                            captions=(
                                "Input image for analysis",
                                "Red = High attention, Blue = Low attention"
                            )
                        )
                        
                        # Clean up
                        os.unlink(tmp_path)
                    else:
                        st.info("ℹ️ Load trained model to see GradCAM visualization")
            
            except Exception as e:
                st.error(f"GradCAM generation failed: {e}")
        
        # Interpretation guide
        with st.expander("ℹ️ How to interpret results"):
            st.markdown("""
            **Confidence Levels:**
            - ✅ **High (>70%)**: Very confident prediction - likely correct
            - ⚠️ **Medium (40-70%)**: Moderate confidence - consider additional factors
            - ❌ **Low (<40%)**: Low confidence - may need expert verification
            
            **GradCAM Heatmap:**
            - 🔴 **Red areas**: Model focused strongly here (disease symptoms)
            - 🟡 **Yellow/Orange**: Moderate attention
            - 🔵 **Blue areas**: Model ignored these regions
            
            **Recommendations:**
            - Always verify with agricultural expert for critical decisions
            - Consider multiple images from different angles
            - Check environmental conditions (weather, soil, etc.)
            """)
    
    # History section
    if st.session_state['history']:
        st.markdown("---")
        st.markdown("### 📜 Analysis History")
        
        st.info(f"📊 Total analyses: {len(st.session_state['history'])}")
        
        if st.button("🗑️ Clear History"):
            st.session_state['history'] = []
            st.rerun()
        
        # Show recent history
        with st.expander("View recent analyses"):
            for i, entry in enumerate(reversed(st.session_state['history'][-5:]), 1):
                st.markdown(f"**Analysis {len(st.session_state['history']) - i + 1}**")
                st.caption(f"⏰ {entry['timestamp']}")
                
                predictions = entry['predictions']
                top_pred = predictions[0]
                st.markdown(f"🎯 **{top_pred[0]}** ({top_pred[1]*100:.1f}%)")
                
                if 'metadata' in entry and 'crop' in entry['metadata']:
                    crop = entry['metadata']['crop']
                    st.caption(f"{get_crop_emoji(crop)} Crop: {crop}")
                
                st.markdown("---")


# ============================================================================
# Run App
# ============================================================================

if __name__ == "__main__":
    main()
