# 💧 Recommendations Page
# Treatment recommendations and prevention steps

import streamlit as st
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui_components import load_custom_css, header, recommendation_card
from utils.mock_api import recommend

st.set_page_config(
    page_title="Recommendations - CropShield AI",
    page_icon="💧",
    layout="wide"
)

load_custom_css()

# Page header
header("💧 Treatment Recommendations", "Personalized treatment plans for your crops")

# Prototype disclaimer
st.warning("⚠️ **Prototype Notice:** Recommendations shown are mock data for demonstration purposes. Always consult agricultural experts for production use.")

st.markdown("<br>", unsafe_allow_html=True)

# Check if diagnosis results exist
last_result = st.session_state.get('last_result')
weather_data = st.session_state.get('weather_data')

if last_result is None:
    # No diagnosis available
    st.markdown(
        """
        <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
             border-radius: 15px; margin: 2rem 0;">
            <div style="font-size: 5rem; margin-bottom: 1rem;">🔬</div>
            <h2 style="color: #e65100; margin-bottom: 1rem;">No Diagnosis Available</h2>
            <p style="color: #666; font-size: 1.2rem; margin-bottom: 2rem;">
                Please run a diagnosis first to get personalized treatment recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🌿 Go to Diagnosis", type="primary", use_container_width=True, key="go_to_diag_rec"):
            st.switch_page("pages/2_Diagnosis.py")
    
    st.stop()

# Extract diagnosis information
crop = last_result.get('crop', 'Unknown')
diseases = last_result.get('diseases', [])

# Summary section
st.markdown(
    f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
         padding: 2rem; margin-bottom: 2rem; border-radius: 12px;">
        <h3 style="color: #2d5016; margin-bottom: 1rem;">📋 Diagnosis Summary</h3>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p style="color: #558b2f; font-size: 1.2rem; margin: 0;">
                    <strong>Crop:</strong> {crop} | <strong>Diseases Detected:</strong> {len(diseases)}
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Check if healthy
if len(diseases) == 1 and diseases[0].get('name', '').lower() == 'healthy':
    st.success("🎉 Great news! Your plant appears to be healthy!")
    
    st.markdown(
        """
        <div class="metric-card" style="padding: 2rem; margin: 2rem 0;">
            <h3 style="color: #2d5016; margin-bottom: 1rem;">🌱 Preventive Care Recommendations</h3>
            <p style="color: #666; line-height: 1.8;">
                Even though your plant is healthy, continue following these preventive measures:
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Get recommendations for healthy plants
    recommendation = recommend("Healthy", crop, weather_data)
    recommendation_card(recommendation)
else:
    # Display recommendations for each disease
    st.markdown("### 💊 Treatment Plans")
    
    for idx, disease in enumerate(diseases):
        disease_name = disease.get('name', 'Unknown')
        confidence = disease.get('confidence', 0)
        
        # Disease header with confidence
        st.markdown(
            f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #fff3e0 0%, #ffffff 100%); 
                 padding: 1.5rem; margin: 1.5rem 0; border-left: 5px solid #ff9800;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="color: #e65100; margin: 0;">
                        🦠 Disease {idx + 1}: {disease_name}
                    </h3>
                    <div style="background: #ff9800; color: white; padding: 0.5rem 1rem; 
                         border-radius: 20px; font-weight: bold;">
                        {confidence:.1f}% Confidence
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Get recommendations from API
        recommendation = recommend(disease_name, crop, weather_data)
        
        # Display recommendation card with unique key
        recommendation_card(recommendation, key_suffix=f"disease_{idx}")
        
        st.markdown("<br>", unsafe_allow_html=True)

# Generate downloadable report
st.markdown("---")
st.markdown("### 📥 Download Treatment Report")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        <div class="metric-card" style="padding: 1.5rem;">
            <p style="color: #666; margin: 0;">
                Download a comprehensive text report including diagnosis results, 
                treatment recommendations, and prevention guidelines for your records.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    # Generate text report
    def generate_text_report():
        """Generate a text-based treatment report"""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("CROPSHIELD AI - TREATMENT REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Crop Type: {crop}")
        report_lines.append(f"Total Diseases Detected: {len(diseases)}")
        report_lines.append("\n" + "=" * 70)
        
        # Add each disease and recommendation
        for idx, disease in enumerate(diseases):
            disease_name = disease.get('name', 'Unknown')
            confidence = disease.get('confidence', 0)
            symptoms = disease.get('symptoms', 'Not specified')
            
            report_lines.append(f"\n\nDISEASE {idx + 1}: {disease_name.upper()}")
            report_lines.append("-" * 70)
            report_lines.append(f"Confidence Score: {confidence:.1f}%")
            report_lines.append(f"Symptoms: {symptoms}")
            
            # Get recommendations
            rec = recommend(disease_name, crop, weather_data)
            
            report_lines.append(f"\nCHEMICAL TREATMENT:")
            report_lines.append(f"  {rec.get('chemical_treatment', 'Not specified')}")
            
            report_lines.append(f"\nORGANIC TREATMENT:")
            report_lines.append(f"  {rec.get('organic_treatment', 'Not specified')}")
            
            report_lines.append(f"\nDOSAGE & TIMING:")
            report_lines.append(f"  {rec.get('dosage', 'Not specified')}")
            
            report_lines.append(f"\nPREVENTION:")
            report_lines.append(f"  {rec.get('prevention', 'Not specified')}")
            
            report_lines.append(f"\nEXPECTED YIELD PRESERVATION: {rec.get('yield_estimate', 'N/A')}")
            
            # Climate advice
            climate_advice = rec.get('climate_advice', [])
            if climate_advice:
                report_lines.append(f"\nCLIMATE-ADJUSTED ADVICE:")
                for advice in climate_advice:
                    report_lines.append(f"  - {advice}")
        
        # Weather conditions
        if weather_data:
            report_lines.append("\n\n" + "=" * 70)
            report_lines.append("WEATHER CONDITIONS")
            report_lines.append("-" * 70)
            report_lines.append(f"Temperature: {weather_data.get('temperature', 'N/A')}°C")
            report_lines.append(f"Humidity: {weather_data.get('humidity', 'N/A')}%")
            report_lines.append(f"Rainfall: {weather_data.get('rainfall', 'N/A')}mm")
            report_lines.append(f"Soil pH: {weather_data.get('soil_ph', 'N/A')}")
        
        # Footer
        report_lines.append("\n\n" + "=" * 70)
        report_lines.append("DISCLAIMER")
        report_lines.append("-" * 70)
        report_lines.append("This is a prototype report with mock data for demonstration purposes.")
        report_lines.append("Always consult qualified agricultural experts before applying treatments.")
        report_lines.append("Follow all safety guidelines when handling pesticides and chemicals.")
        report_lines.append("\n" + "=" * 70)
        report_lines.append("© 2025 CropShield AI - Empowering Farmers with AI Technology")
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    # Generate report content
    report_text = generate_text_report()
    
    # Download button
    st.download_button(
        label="� Download as Text",
        data=report_text,
        file_name=f"cropshield_report_{crop}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        type="primary",
        use_container_width=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Treatment schedule planner
with st.expander("📅 Treatment Schedule Planner"):
    st.markdown("### Create Your Treatment Timeline")
    
    st.info("💡 **Tip:** Plan your treatment schedule based on weather forecasts and disease severity.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", datetime.now())
        treatment_frequency = st.selectbox(
            "Treatment Frequency",
            ["Every 7 days", "Every 10 days", "Every 14 days", "As needed"]
        )
    
    with col2:
        num_applications = st.number_input("Number of Applications", min_value=1, max_value=10, value=3)
        reminder_enabled = st.checkbox("Enable Reminders", value=False)
    
    if st.button("📋 Generate Schedule", use_container_width=True, key="generate_schedule_rec"):
        st.success("✅ Treatment schedule feature coming soon!")
        st.info(f"Planned: {num_applications} treatments starting from {start_date}")

st.markdown("<br>", unsafe_allow_html=True)

# Best practices section
st.markdown("### 🌟 Treatment Best Practices")

practice_col1, practice_col2, practice_col3 = st.columns(3)

with practice_col1:
    st.markdown(
        """
        <div class="metric-card" style="padding: 1.5rem; min-height: 200px;">
            <div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">⏰</div>
            <h4 style="color: #2d5016; text-align: center;">Timing Matters</h4>
            <p style="color: #666; font-size: 0.9rem; text-align: center;">
                Apply treatments early morning (6-9 AM) or late evening (4-6 PM) 
                when temperatures are cooler
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with practice_col2:
    st.markdown(
        """
        <div class="metric-card" style="padding: 1.5rem; min-height: 200px;">
            <div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">🧪</div>
            <h4 style="color: #2d5016; text-align: center;">Correct Dosage</h4>
            <p style="color: #666; font-size: 0.9rem; text-align: center;">
                Always follow recommended dosages. More is not better and can harm crops 
                and environment
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with practice_col3:
    st.markdown(
        """
        <div class="metric-card" style="padding: 1.5rem; min-height: 200px;">
            <div style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">🛡️</div>
            <h4 style="color: #2d5016; text-align: center;">Safety First</h4>
            <p style="color: #666; font-size: 0.9rem; text-align: center;">
                Wear protective equipment: gloves, mask, and goggles. 
                Keep chemicals away from children
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Navigation section
st.markdown("---")
st.markdown("### 🔗 What's Next?")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("🌿 New Diagnosis", use_container_width=True, key="nav_new_diag_rec"):
        st.switch_page("pages/2_Diagnosis.py")

with nav_col2:
    if st.button("📊 View Impact Metrics", use_container_width=True, key="nav_impact_rec"):
        st.switch_page("pages/4_Impact_Metrics.py")

with nav_col3:
    if st.button("❓ Get Help", use_container_width=True, key="nav_help_rec"):
        st.switch_page("pages/5_Help.py")

# Sidebar with quick info
with st.sidebar:
    st.markdown("### 💊 Treatment Summary")
    
    if diseases:
        st.metric("Diseases Detected", len(diseases))
        st.metric("Crop Type", crop)
        
        if len(diseases) > 0 and diseases[0].get('name', '').lower() != 'healthy':
            avg_confidence = sum(d.get('confidence', 0) for d in diseases) / len(diseases)
            st.metric("Avg. Confidence", f"{avg_confidence:.1f}%")
    
    st.markdown("---")
    
    st.markdown("### 📞 Need Expert Help?")
    st.markdown(
        """
        <div style="padding: 1rem; background: #f1f8e9; border-radius: 8px; text-align: center;">
            <p style="color: #2d5016; margin: 0; font-size: 0.9rem;">
                Have questions about treatments? Contact our agricultural experts.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("📱 Contact Expert", use_container_width=True, key="contact_expert_rec"):
        st.info("Expert consultation coming soon!")

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 1rem; color: #999; font-size: 0.9rem;">
        ⚠️ <strong>Important:</strong> This is prototype data for demonstration. 
        Always consult certified agronomists for actual treatment decisions.
    </div>
    """,
    unsafe_allow_html=True
)
