# 🏠 Home Page
# Landing page with introduction and navigation to diagnosis

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui_components import load_custom_css, header

st.set_page_config(
    page_title="Home - CropShield AI",
    page_icon="🏠",
    layout="wide"
)

load_custom_css()

# Main header
header("🌾 CropShield AI", "Smart Plant Health & Disease Detection")

# Hero section with description
st.markdown(
    """
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f1f8e9 0%, #ffffff 100%); 
         border-radius: 15px; margin: 2rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h3 style="color: #2d5016; margin-bottom: 1rem;">
            AI-Powered Plant Disease Detection & Smart Recommendations
        </h3>
        <p style="color: #558b2f; font-size: 1.1rem; line-height: 1.6;">
            Protect your crops with cutting-edge AI technology. Get instant diagnosis, 
            treatment recommendations, and track your environmental impact.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Three-step instructions
st.markdown("### 📋 How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card" style="text-align: center; padding: 2rem; min-height: 280px;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📸</div>
<h3 style="color: #2d5016; margin-bottom: 1rem;">1. Upload Image</h3>
            <p style="color: #666; line-height: 1.6;">
                Take a clear photo of the affected plant parts and upload it. 
                Our AI supports Wheat, Potato, Tomato, and Sugarcane.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="metric-card" style="text-align: center; padding: 2rem; min-height: 280px;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🌦️</div>
            <h3 style="color: #2d5016; margin-bottom: 1rem;">2. Add Weather Data</h3>
            <p style="color: #666; line-height: 1.6;">
                Optionally provide local weather conditions (temperature, humidity, rainfall) 
                for climate-aware recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-card" style="text-align: center; padding: 2rem; min-height: 280px;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🔬</div>
            <h3 style="color: #2d5016; margin-bottom: 1rem;">3. Get Diagnosis</h3>
            <p style="color: #666; line-height: 1.6;">
                Receive instant AI-powered diagnosis with confidence scores, 
                treatment options, and actionable solutions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# Call-to-action button
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #2d5016; margin-bottom: 1.5rem;">Ready to diagnose your crop?</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Large diagnosis button
    if st.button("🌿 Diagnose My Plant", type="primary", use_container_width=True, key="diagnose_button"):
        st.session_state['page'] = 'diagnosis'
        st.switch_page("pages/2_Diagnosis.py")

st.markdown("<br><br>", unsafe_allow_html=True)

# Features section
st.markdown("### ✨ Why Choose CropShield AI?")

feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)

with feature_col1:
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚡</div>
            <h4 style="color: #2d5016; font-size: 1rem; margin-bottom: 0.5rem;">Instant Results</h4>
            <p style="color: #666; font-size: 0.9rem;">Get diagnosis in seconds</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with feature_col2:
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎯</div>
            <h4 style="color: #2d5016; font-size: 1rem; margin-bottom: 0.5rem;">High Accuracy</h4>
            <p style="color: #666; font-size: 0.9rem;">85-95% detection rate</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with feature_col3:
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌱</div>
            <h4 style="color: #2d5016; font-size: 1rem; margin-bottom: 0.5rem;">Organic Options</h4>
            <p style="color: #666; font-size: 0.9rem;">Eco-friendly treatments</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with feature_col4:
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
            <h4 style="color: #2d5016; font-size: 1rem; margin-bottom: 0.5rem;">Impact Tracking</h4>
            <p style="color: #666; font-size: 0.9rem;">Monitor your savings</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# Statistics section
st.markdown("### 📈 Our Impact")

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.metric(label="🌾 Crops Analyzed", value="10,000+", delta="Growing daily")

with stat_col2:
    st.metric(label="👨‍🌾 Farmers Helped", value="5,000+", delta="Across India")

with stat_col3:
    st.metric(label="💧 Water Saved", value="2M Liters", delta="Environmental impact")

with stat_col4:
    st.metric(label="🧪 Pesticide Reduced", value="45%", delta="Healthier farms")

st.markdown("<br><br>", unsafe_allow_html=True)

# Testimonials section
st.markdown("### 💬 What Farmers Say")

test_col1, test_col2, test_col3 = st.columns(3)

with test_col1:
    st.markdown(
        """
        <div class="metric-card" style="padding: 1.5rem; min-height: 200px;">
            <div style="color: #ffa500; font-size: 1.5rem; margin-bottom: 0.5rem;">⭐⭐⭐⭐⭐</div>
            <p style="color: #666; font-style: italic; margin-bottom: 1rem;">
                "CropShield AI helped me detect wheat rust early. Saved my entire harvest!"
            </p>
            <p style="color: #2d5016; font-weight: bold; margin: 0;">- Rajesh Kumar</p>
            <p style="color: #999; font-size: 0.9rem;">Punjab</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with test_col2:
    st.markdown(
        """
        <div class="metric-card" style="padding: 1.5rem; min-height: 200px;">
            <div style="color: #ffa500; font-size: 1.5rem; margin-bottom: 0.5rem;">⭐⭐⭐⭐⭐</div>
            <p style="color: #666; font-style: italic; margin-bottom: 1rem;">
                "The organic treatment recommendations are excellent. No more harsh chemicals!"
            </p>
            <p style="color: #2d5016; font-weight: bold; margin: 0;">- Lakshmi Devi</p>
            <p style="color: #999; font-size: 0.9rem;">Tamil Nadu</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with test_col3:
    st.markdown(
        """
        <div class="metric-card" style="padding: 1.5rem; min-height: 200px;">
            <div style="color: #ffa500; font-size: 1.5rem; margin-bottom: 0.5rem;">⭐⭐⭐⭐⭐</div>
            <p style="color: #666; font-style: italic; margin-bottom: 1rem;">
                "Easy to use, accurate results. A must-have tool for modern farmers!"
            </p>
            <p style="color: #2d5016; font-weight: bold; margin: 0;">- Suresh Patil</p>
            <p style="color: #999; font-size: 0.9rem;">Maharashtra</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br><br><br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0; color: #666;">
        <p style="margin-bottom: 0.5rem;">
            <strong>CropShield AI</strong> - Empowering Farmers with AI Technology
        </p>
        <p style="font-size: 0.9rem; margin-bottom: 1rem;">
            🌾 Protect Your Crops | 🌍 Save the Environment | 📈 Maximize Yields
        </p>
        <p style="font-size: 0.85rem; color: #999;">
            © 2025 CropShield AI. All rights reserved. | Made with ❤️ for Indian Farmers
        </p>
        <p style="font-size: 0.8rem; color: #bbb; margin-top: 0.5rem;">
            Version 1.0.0 | Powered by Advanced Machine Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
