# 🌿 Diagnosis Page
# Image upload, crop selection, and disease diagnosis

import streamlit as st
import sys
import os
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui_components import load_custom_css, header, upload_card, weather_expander, result_card
from utils.mock_api import predict

st.set_page_config(
    page_title="Diagnosis - CropShield AI",
    page_icon="🌿",
    layout="wide"
)

load_custom_css()

# Initialize session state
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'weather_data' not in st.session_state:
    st.session_state['weather_data'] = None

# Page header
header("🌿 Plant Disease Diagnosis", "Upload an image to detect diseases using AI")

# Instructions banner
st.info("📋 **Instructions:** Upload a clear image of your plant showing the affected areas. Select the crop type and optionally add weather data for better recommendations.")

st.markdown("<br>", unsafe_allow_html=True)

# Upload card section
uploaded_file, selected_crop, analyze_clicked = upload_card()

st.markdown("<br>", unsafe_allow_html=True)

# Weather data expander
st.markdown("### 🌦️ Environmental Conditions (Optional)")
weather_data = weather_expander()

# Store weather data in session state
st.session_state['weather_data'] = weather_data

st.markdown("<br>", unsafe_allow_html=True)

# Analysis section
if uploaded_file is not None and analyze_clicked:
    st.markdown("---")
    st.markdown("### 🔬 AI Analysis in Progress...")
    
    # Show spinner and progress bar
    with st.spinner('🤖 AI is analyzing your plant image...'):
        # Progress bar simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate analysis stages
        stages = [
            (0, "Loading image..."),
            (20, "Preprocessing image data..."),
            (40, "Running AI model inference..."),
            (60, "Analyzing disease patterns..."),
            (80, "Calculating confidence scores..."),
            (90, "Generating recommendations..."),
            (100, "Analysis complete!")
        ]
        
        for progress, stage_text in stages:
            status_text.text(stage_text)
            progress_bar.progress(progress)
            time.sleep(0.3)  # Simulate processing time
        
        # Read image bytes
        image_bytes = uploaded_file.getvalue()
        
        # Call predict API
        result = predict(
            image_bytes=image_bytes,
            crop=selected_crop,
            weather=weather_data
        )
        
        # Store result in session state
        st.session_state['last_result'] = result
        
        # Create history entry
        diseases = result.get('diseases', [])
        primary_disease = diseases[0] if diseases else {"name": "Unknown", "confidence": 0}
        
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "crop": selected_crop,
            "disease": primary_disease.get("name", "Unknown"),
            "confidence": primary_disease.get("confidence", 0),
            "pesticide_saved_l": round(10.0 + (primary_disease.get("confidence", 0) * 0.1), 2),
            "water_saved_l": round(400.0 + (primary_disease.get("confidence", 0) * 5.0), 2)
        }
        
        # Append to history
        st.session_state['history'].append(history_entry)
        
        status_text.empty()
        progress_bar.empty()
    
    st.success("✅ Analysis completed successfully!")
    
    st.markdown("<br>", unsafe_allow_html=True)

# Display results if available
if st.session_state['last_result'] is not None:
    st.markdown("---")
    
    # Display result card
    result_card(st.session_state['last_result'])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display GradCAM if available
    gradcam = st.session_state['last_result'].get('gradcam')
    if gradcam:
        st.markdown("### 🎯 AI Attention Heatmap (GradCAM)")
        st.markdown(
            """
            <div class="metric-card" style="padding: 1rem;">
                <p style="color: #666; margin-bottom: 1rem;">
                    This heatmap shows which parts of the image the AI focused on to make its prediction.
                    Warmer colors (red/yellow) indicate areas of high attention.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(gradcam, caption="AI Attention Regions", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Navigation buttons
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            if st.button("📋 View Detailed Recommendations", type="primary", use_container_width=True, key="view_rec_button"):
                st.session_state['page'] = 'recommendations'
                st.switch_page("pages/3_Recommendations.py")
        
        with button_col2:
            if st.button("📊 View Impact Metrics", use_container_width=True, key="view_impact_button"):
                st.session_state['page'] = 'impact'
                st.switch_page("pages/4_Impact_Metrics.py")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Additional information section
    with st.expander("ℹ️ Understanding Your Results"):
        st.markdown("""
        **Confidence Score:** 
        - **90-100%**: Very high confidence - immediate action recommended
        - **75-89%**: High confidence - monitor closely and prepare treatment
        - **60-74%**: Moderate confidence - continue monitoring
        - **Below 60%**: Low confidence - consider re-scanning with better image
        
        **Health Status:**
        - **✅ Healthy**: No diseases detected - continue preventive care
        - **⚠️ Diseased**: Disease detected - follow recommended treatments
        
        **Next Steps:**
        1. Review the recommended treatments in the Recommendations page
        2. Choose between chemical or organic treatment options
        3. Follow the dosage and timing instructions carefully
        4. Monitor your plant's progress over the next few days
        5. Track your environmental impact in the Impact Metrics page
        """)
    
    # Quick tips
    st.markdown("### 💡 Quick Tips")
    
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    
    with tip_col1:
        st.markdown(
            """
            <div class="metric-card" style="padding: 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📸</div>
                <h4 style="color: #2d5016; margin-bottom: 0.5rem;">Better Photos</h4>
                <p style="color: #666; font-size: 0.9rem;">
                    Take photos in natural daylight, focus on affected areas, avoid blurry images
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with tip_col2:
        st.markdown(
            """
            <div class="metric-card" style="padding: 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⏰</div>
                <h4 style="color: #2d5016; margin-bottom: 0.5rem;">Early Detection</h4>
                <p style="color: #666; font-size: 0.9rem;">
                    Scan your crops regularly to catch diseases early when treatment is most effective
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with tip_col3:
        st.markdown(
            """
            <div class="metric-card" style="padding: 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">�</div>
                <h4 style="color: #2d5016; margin-bottom: 0.5rem;">Preventive Care</h4>
                <p style="color: #666; font-size: 0.9rem;">
                    Follow prevention tips even for healthy plants to avoid future infections
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    # Show placeholder when no analysis has been done
    if uploaded_file is None:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem; background: #f5f5f5; border-radius: 15px; 
                 border: 2px dashed #7cb342;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📷</div>
                <h3 style="color: #2d5016; margin-bottom: 1rem;">No Image Uploaded</h3>
                <p style="color: #666; font-size: 1.1rem;">
                    Upload a plant image above to get started with AI diagnosis
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Sidebar information
with st.sidebar:
    st.markdown("### 📊 Session Statistics")
    
    total_scans = len(st.session_state['history'])
    
    st.metric("Total Scans", total_scans)
    
    if total_scans > 0:
        recent_entry = st.session_state['history'][-1]
        st.metric("Last Scan", recent_entry['crop'])
        st.metric("Confidence", f"{recent_entry['confidence']:.1f}%")
    
    st.markdown("---")
    
    st.markdown("### 🌾 Supported Crops")
    crops = ['Wheat', 'Rice', 'Potato', 'Tomato', 'Maize']
    for crop in crops:
        st.markdown(f"✅ {crop}")
    
    st.markdown("---")
    
    st.markdown("### 🔗 Quick Links")
    if st.button("🏠 Home", use_container_width=True, key="nav_home_diag"):
        st.switch_page("pages/1_Home.py")
    
    if st.button("💧 Recommendations", use_container_width=True, key="nav_rec_diag"):
        st.switch_page("pages/3_Recommendations.py")
    
    if st.button("📊 Impact Metrics", use_container_width=True, key="nav_impact_diag"):
        st.switch_page("pages/4_Impact_Metrics.py")
    
    if st.button("❓ Help", use_container_width=True, key="nav_help_diag"):
        st.switch_page("pages/5_Help.py")

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 1rem; color: #999; font-size: 0.9rem;">
        💡 <strong>Pro Tip:</strong> For best results, take multiple photos from different angles 
        and in good lighting conditions
    </div>
    """,
    unsafe_allow_html=True
)
