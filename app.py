# CropShield AI - Main Application Entry Point
# This is the main landing page for the Streamlit app

import streamlit as st
import os
import sys

# Page configuration
st.set_page_config(
    page_title="CropShield AI - AI-Powered Plant Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/VeerbhadraMahant/CropShieldAI',
        'Report a bug': 'https://github.com/VeerbhadraMahant/CropShieldAI/issues',
        'About': '# CropShield AI\n## AI-Powered Plant Disease Detection\n\nEmpowering farmers with cutting-edge AI technology for early disease detection and sustainable agriculture.'
    }
)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load custom CSS
def load_css():
    """Load custom CSS styles"""
    css_path = os.path.join(os.path.dirname(__file__), "utils", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None

if 'weather_data' not in st.session_state:
    st.session_state['weather_data'] = None

if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# Sidebar branding
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #2d5016; font-size: 2rem; margin: 0;">🌾 CropShield AI</h1>
            <p style="color: #7cb342; font-size: 0.9rem; margin-top: 0.5rem;">
                AI-Powered Crop Protection
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    
    page_selection = st.selectbox(
        "Navigate",
        options=["Home", "Diagnosis", "Recommendations", "Impact Metrics", "Help"],
        index=["Home", "Diagnosis", "Recommendations", "Impact Metrics", "Help"].index(st.session_state.get('page', 'Home')) if st.session_state.get('page', 'Home') in ["Home", "Diagnosis", "Recommendations", "Impact Metrics", "Help"] else 0,
        key="navigation_selector"
    )
    
    # Update session state
    st.session_state['page'] = page_selection
    
    st.markdown("---")
    
    # Quick stats in sidebar
    st.markdown("### 📊 Session Stats")
    total_scans = len(st.session_state.get('history', []))
    st.metric("Total Scans", total_scans)
    
    if st.session_state.get('last_result'):
        st.success("✅ Last scan available")
    else:
        st.info("ℹ️ No scans yet")
    
    st.markdown("---")
    
    # App info
    st.markdown("### ℹ️ About")
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #666; line-height: 1.6;">
            <p><strong>Version:</strong> 1.0.0 (Prototype)</p>
            <p><strong>Status:</strong> Demo</p>
            <p><strong>Crops:</strong> 5 types</p>
            <p><strong>Diseases:</strong> 9+ common diseases</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Support section
    st.markdown("### 🤝 Support")
    if st.button("📖 User Guide", use_container_width=True):
        st.session_state['page'] = 'Help'
        st.rerun()
    
    if st.button("🐛 Report Issue", use_container_width=True):
        st.info("GitHub Issues: Coming soon!")
    
    st.markdown("---")
    
    # Social links
    st.markdown(
        """
        <div style="text-align: center; font-size: 0.8rem; color: #999;">
            <p>Follow Us:</p>
            <p>
                <a href="#" style="color: #7cb342; text-decoration: none;">🌐 Website</a> | 
                <a href="#" style="color: #7cb342; text-decoration: none;">💼 LinkedIn</a><br>
                <a href="#" style="color: #7cb342; text-decoration: none;">🐦 Twitter</a> | 
                <a href="#" style="color: #7cb342; text-decoration: none;">📘 Facebook</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main content area - Route to appropriate page
if page_selection == "Home":
    # Home page content
    from utils.ui_components import header
    
    header("🌾 CropShield AI", "Smart Plant Health & Disease Detection")
    
    # Hero section
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
                    Our AI supports Wheat, Rice, Potato, Tomato, and Maize.
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
    
    # Call-to-action
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
        
        if st.button("🌿 Start Diagnosis", type="primary", use_container_width=True):
            st.session_state['page'] = 'Diagnosis'
            st.rerun()
    
    # Features showcase
    st.markdown("<br><br>", unsafe_allow_html=True)
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

elif page_selection == "Diagnosis":
    # Redirect to diagnosis page
    st.switch_page("pages/2_Diagnosis.py")

elif page_selection == "Recommendations":
    # Redirect to recommendations page
    st.switch_page("pages/3_Recommendations.py")

elif page_selection == "Impact Metrics":
    # Redirect to impact metrics page
    st.switch_page("pages/4_Impact_Metrics.py")

elif page_selection == "Help":
    # Redirect to help page
    st.switch_page("pages/5_Help.py")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
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
            Version 1.0.0 (Prototype) | Powered by Advanced Machine Learning
        </p>
        <p style="font-size: 0.75rem; color: #ccc; margin-top: 1rem;">
            ⚠️ This is a prototype with mock data for demonstration purposes.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
