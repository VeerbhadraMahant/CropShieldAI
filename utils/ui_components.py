# Reusable UI components for the Streamlit app
# Contains custom styled components and helper functions

import streamlit as st
import os
from PIL import Image


def load_custom_css():
    """Load custom CSS styles"""
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def header(title, subtitle=None):
    """
    Render a centered header with title and optional subtitle
    
    Args:
        title: Main title text
        subtitle: Optional subtitle text
    """
    st.markdown(
        f"""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="main-title" style="color: #2d5016; font-size: 2.5rem; margin-bottom: 0.5rem;">
                {title}
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if subtitle:
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: -1rem; margin-bottom: 2rem;">
                <p class="subtitle" style="color: #558b2f; font-size: 1.2rem; font-weight: 400;">
                    {subtitle}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


def upload_card():
    """
    Render image upload card with crop selection
    
    Returns:
        tuple: (uploaded_file, selected_crop, analyze_button_clicked)
    """
    st.markdown(
        """
        <div class="metric-card" style="padding: 2rem; margin: 1rem 0;">
            <h3 style="color: #2d5016; margin-bottom: 1rem;">📸 Upload Plant Image</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a plant image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Upload a clear image of the affected plant parts"
        )
    
    with col2:
        selected_crop = st.selectbox(
            "🌾 Select Crop Type",
            options=['Wheat', 'Rice', 'Potato', 'Tomato', 'Maize'],
            index=0
        )
    
    # Show image preview if uploaded
    if uploaded_file is not None:
        st.markdown("---")
        col_preview, col_button = st.columns([2, 1])
        
        with col_preview:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col_button:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_clicked = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True,
                help="Click to start AI analysis"
            )
    else:
        analyze_clicked = False
    
    return uploaded_file, selected_crop, analyze_clicked


def weather_expander():
    """
    Render weather data input expander
    
    Returns:
        dict: Weather data {temperature, humidity, rainfall, soil_ph}
    """
    with st.expander("🌦️ Add Local Weather Data (Optional)", expanded=False):
        st.markdown("*Provide environmental conditions for more accurate recommendations*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.number_input(
                "🌡️ Temperature (°C)",
                min_value=-10.0,
                max_value=50.0,
                value=25.0,
                step=0.5,
                help="Current ambient temperature"
            )
            
            humidity = st.number_input(
                "💧 Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0,
                help="Relative humidity percentage"
            )
        
        with col2:
            rainfall = st.number_input(
                "🌧️ Rainfall (mm)",
                min_value=0.0,
                max_value=500.0,
                value=10.0,
                step=1.0,
                help="Recent rainfall in millimeters"
            )
            
            soil_ph = st.number_input(
                "🧪 Soil pH",
                min_value=3.0,
                max_value=10.0,
                value=6.5,
                step=0.1,
                help="Soil pH level (optional)"
            )
        
        use_mock = st.button("📍 Use My Current Weather", help="Simulate with mock data")
        
        if use_mock:
            import random
            temperature = round(random.uniform(20, 35), 1)
            humidity = round(random.uniform(40, 85), 1)
            rainfall = round(random.uniform(0, 50), 1)
            st.success(f"✅ Mock weather loaded: {temperature}°C, {humidity}% humidity, {rainfall}mm rainfall")
    
    return {
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "soil_ph": soil_ph
    }


def result_card(result):
    """
    Display diagnosis results in a styled card
    
    Args:
        result: dict with keys: crop, diseases (list), gradcam
    """
    crop = result.get("crop", "Unknown")
    diseases = result.get("diseases", [])
    
    # Determine overall health status
    if not diseases or (len(diseases) == 1 and diseases[0].get("name") == "Healthy"):
        status = "Healthy"
        status_color = "#4caf50"
        status_icon = "✅"
    else:
        status = "Diseased"
        status_color = "#f44336"
        status_icon = "⚠️"
    
    st.markdown(
        f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f1f8e9 0%, #ffffff 100%); 
             padding: 2rem; margin: 2rem 0; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <h2 style="color: #2d5016; margin-bottom: 1rem;">🔬 Diagnosis Results</h2>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <h3 style="color: #558b2f; margin: 0;">Crop: {crop}</h3>
                </div>
                <div style="background: {status_color}; color: white; padding: 0.5rem 1.5rem; 
                     border-radius: 20px; font-weight: bold;">
                    {status_icon} {status}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display each disease
    for idx, disease in enumerate(diseases):
        disease_name = disease.get("name", "Unknown")
        confidence = disease.get("confidence", 0)
        symptoms = disease.get("symptoms", "No symptoms described")
        
        # Confidence badge color
        if confidence >= 90:
            badge_color = "#f44336"  # High confidence - red
        elif confidence >= 75:
            badge_color = "#ff9800"  # Medium confidence - orange
        else:
            badge_color = "#ffc107"  # Lower confidence - yellow
        
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 🦠 {disease_name}")
            st.markdown(f"**Symptoms:** {symptoms}")
        
        with col2:
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 1rem;">
                    <div style="background: {badge_color}; color: white; padding: 1rem; 
                         border-radius: 10px; font-size: 1.5rem; font-weight: bold;">
                        {confidence:.1f}%
                    </div>
                    <p style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">Confidence</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Show GradCAM placeholder if available
    gradcam = result.get("gradcam")
    if gradcam:
        st.markdown("---")
        st.markdown("### 🎯 Attention Heatmap (GradCAM)")
        st.image(gradcam, caption="AI attention regions", use_container_width=True)


def recommendation_card(recommendation, key_suffix=""):
    """
    Display treatment recommendations in a styled card
    
    Args:
        recommendation: dict with treatment details
        key_suffix: unique suffix for button keys (to avoid duplicate IDs)
    """
    disease = recommendation.get("disease", "Unknown")
    chemical = recommendation.get("chemical_treatment", "Not specified")
    organic = recommendation.get("organic_treatment", "Not specified")
    dosage = recommendation.get("dosage", "Not specified")
    prevention = recommendation.get("prevention", "Not specified")
    yield_est = recommendation.get("yield_estimate", "N/A")
    climate_advice = recommendation.get("climate_advice", [])
    
    st.markdown(
        f"""
        <div class="metric-card" style="background: #ffffff; padding: 2rem; margin: 2rem 0; 
             border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <h2 style="color: #2d5016; margin-bottom: 1rem;">💊 Treatment Recommendations</h2>
            <h3 style="color: #558b2f;">Disease: {disease}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Climate advice alerts
    if climate_advice:
        st.info("**🌦️ Climate-Adjusted Advice:**\n" + "\n".join([f"- {advice}" for advice in climate_advice]))
    
    # Treatment options in tabs
    tab1, tab2, tab3 = st.tabs(["🧪 Chemical Treatment", "🌱 Organic Treatment", "🛡️ Prevention"])
    
    with tab1:
        st.markdown(f"**Recommended Treatment:**")
        st.success(chemical)
        st.markdown(f"**📋 Dosage & Timing:**")
        st.info(dosage)
    
    with tab2:
        st.markdown(f"**Organic Alternatives:**")
        st.success(organic)
        st.markdown("*Organic treatments are safer for the environment and beneficial insects*")
    
    with tab3:
        st.markdown(f"**Preventive Measures:**")
        st.warning(prevention)
    
    # Yield estimate and download button
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.metric(
            label="📊 Expected Yield Preservation",
            value=yield_est,
            help="Estimated yield with proper treatment"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📥 Download Report", type="primary", use_container_width=True, key=f"download_report_{key_suffix}"):
            st.success("✅ Report generation coming soon!")
            st.info("Report will include: Diagnosis, Recommendations, and Treatment Schedule")


def impact_cards(impact):
    """
    Display environmental impact metrics in three cards
    
    Args:
        impact: dict with pesticide_reduction_pct, water_saved_liters, yield_preservation_pct
    """
    pesticide_reduction = impact.get("pesticide_reduction_pct", 0)
    water_saved = impact.get("water_saved_liters", 0)
    yield_preservation = impact.get("yield_preservation_pct", 0)
    
    st.markdown("### 🌍 Environmental Impact Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                 padding: 1.5rem; text-align: center; border-radius: 12px; 
                 box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🧪</div>
                <div style="font-size: 2rem; font-weight: bold; color: #2d5016; margin-bottom: 0.5rem;">
                    {pesticide_reduction:.1f}%
                </div>
                <div style="color: #558b2f; font-weight: 600;">
                    Pesticide Reduction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%); 
                 padding: 1.5rem; text-align: center; border-radius: 12px; 
                 box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">💧</div>
                <div style="font-size: 2rem; font-weight: bold; color: #01579b; margin-bottom: 0.5rem;">
                    {water_saved:.0f}L
                </div>
                <div style="color: #0277bd; font-weight: 600;">
                    Water Saved
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                 padding: 1.5rem; text-align: center; border-radius: 12px; 
                 box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌾</div>
                <div style="font-size: 2rem; font-weight: bold; color: #e65100; margin-bottom: 0.5rem;">
                    {yield_preservation:.1f}%
                </div>
                <div style="color: #f57c00; font-weight: 600;">
                    Yield Preservation
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)


def render_faq_section():
    """Render FAQ section with common questions"""
    st.markdown("### ❓ Frequently Asked Questions")
    
    with st.expander("🌱 Why is my plant getting sick?"):
        st.write("""
        Plants can get sick due to various factors:
        - **Pathogens**: Fungi, bacteria, viruses, and nematodes
        - **Environmental stress**: Extreme temperatures, drought, or excess water
        - **Nutrient deficiency**: Lack of essential nutrients in soil
        - **Poor agricultural practices**: Improper spacing, contaminated tools
        - **Weather conditions**: High humidity, excessive rainfall
        
        Early detection using AI can help prevent disease spread!
        """)
    
    with st.expander("💊 How to apply treatments safely?"):
        st.write("""
        **Safety Guidelines:**
        1. Always wear protective equipment (gloves, mask, goggles)
        2. Read product labels carefully before application
        3. Apply during cool hours (early morning/late evening)
        4. Avoid spraying on windy days
        5. Keep chemicals away from children and pets
        6. Follow recommended dosage - more is not better
        7. Maintain proper record of treatments
        8. Dispose of empty containers responsibly
        """)
    
    with st.expander("🌾 When should I consult an expert?"):
        st.write("""
        Consult agricultural experts when:
        - Disease spreads rapidly despite treatment
        - Multiple crops are affected simultaneously
        - You're unsure about treatment selection
        - Soil quality has significantly degraded
        - Pest infestation is severe
        - Yield has dropped more than 30%
        - You need guidance on organic farming transition
        """)
    
    with st.expander("📱 How accurate is AI diagnosis?"):
        st.write("""
        Our AI model provides:
        - **High accuracy** (85-95%) for common crop diseases
        - **Confidence scores** to indicate reliability
        - **Early detection** before visible symptoms worsen
        - **Climate-aware recommendations** for your region
        
        Note: AI is a tool to assist farmers, not replace expert consultation for complex cases.
        """)


def render_contact_section():
    """Render contact and support section"""
    st.markdown("### 📞 Get Expert Help")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="metric-card" style="padding: 1.5rem; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">👨‍🌾</div>
                <h4 style="color: #2d5016;">Contact Agricultural Expert</h4>
                <p style="color: #666;">Get personalized advice from certified agronomists</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("📱 WhatsApp Expert", type="primary", use_container_width=True):
            st.success("Opening WhatsApp... (Feature coming soon)")
    
    with col2:
        st.markdown(
            """
            <div class="metric-card" style="padding: 1.5rem; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🏥</div>
                <h4 style="color: #2d5016;">Emergency Helpline</h4>
                <p style="color: #666;">24/7 support for critical crop issues</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("📞 Call Helpline", type="primary", use_container_width=True):
            st.info("Helpline: 1800-XXX-XXXX (Toll Free)")


def language_selector():
    """Render language selection dropdown"""
    languages = {
        "English": "🇬🇧",
        "हिंदी (Hindi)": "🇮🇳",
        "తెలుగు (Telugu)": "🇮🇳",
        "தமிழ் (Tamil)": "🇮🇳",
        "मराठी (Marathi)": "🇮🇳"
    }
    
    selected = st.selectbox(
        "🌐 Language / भाषा",
        options=list(languages.keys()),
        index=0
    )
    
    if selected != "English":
        st.info(f"Language switching to {selected} - Feature coming soon!")
    
    return selected
