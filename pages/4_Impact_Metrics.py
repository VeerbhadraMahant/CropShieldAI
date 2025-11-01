# 📊 Impact Metrics Page
# Environmental impact visualization and metrics

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui_components import load_custom_css, header, impact_cards
from utils.mock_api import MOCK_HISTORY, impact

st.set_page_config(
    page_title="Impact Metrics - CropShield AI",
    page_icon="📊",
    layout="wide"
)

load_custom_css()

# Page header
header("📊 Environmental Impact Metrics", "Track your contribution to sustainable agriculture")

# Get history data
user_history = st.session_state.get('history', [])

# Use user history if available, otherwise use mock data
if len(user_history) > 0:
    history_data = user_history
    data_source = "Your Session Data"
    is_user_data = True
else:
    history_data = MOCK_HISTORY
    data_source = "Demo Data (Sample)"
    is_user_data = False

# Data source indicator
if is_user_data:
    st.success(f"📈 Showing {data_source} - {len(history_data)} scan(s) recorded")
else:
    st.info(f"📊 Showing {data_source} - Run diagnosis to see your personalized metrics")

st.markdown("<br>", unsafe_allow_html=True)

# Calculate impact metrics
impact_data = impact(history_data)

# Display impact cards
impact_cards(impact_data)

st.markdown("<br><br>", unsafe_allow_html=True)

# Detailed statistics section
st.markdown("### 📈 Detailed Impact Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_scans = len(history_data)
    st.metric(
        label="Total Scans",
        value=total_scans,
        help="Number of disease detection scans performed"
    )

with col2:
    if history_data:
        avg_confidence = sum(entry.get('confidence', 0) for entry in history_data) / len(history_data)
        st.metric(
            label="Avg. Confidence",
            value=f"{avg_confidence:.1f}%",
            help="Average confidence score of disease detections"
        )
    else:
        st.metric(label="Avg. Confidence", value="N/A")

with col3:
    unique_crops = len(set(entry.get('crop', 'Unknown') for entry in history_data))
    st.metric(
        label="Crops Monitored",
        value=unique_crops,
        help="Number of different crop types analyzed"
    )

with col4:
    unique_diseases = len(set(entry.get('disease', 'Unknown') for entry in history_data))
    st.metric(
        label="Diseases Detected",
        value=unique_diseases,
        help="Number of unique diseases identified"
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# Time series visualization
st.markdown("### 📉 Impact Trends Over Time")

timeseries = impact_data.get('timeseries', [])

if len(timeseries) > 0:
    # Prepare data for plotting
    dates = [entry['date'] for entry in timeseries]
    pesticide_reduction = [entry['pesticide_reduction'] for entry in timeseries]
    water_saved = [entry['water_saved'] for entry in timeseries]
    
    # Create subplot with two y-axes
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # Add pesticide reduction trace
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pesticide_reduction,
            name="Pesticide Reduction (L)",
            mode='lines+markers',
            line=dict(color='#7cb342', width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Pesticide:</b> %{y:.2f}L<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add water saved trace
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=water_saved,
            name="Water Saved (L)",
            mode='lines+markers',
            line=dict(color='#2196f3', width=3),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Water:</b> %{y:.2f}L<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title="Cumulative Environmental Savings",
        hovermode='x unified',
        plot_bgcolor='white',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axis
    fig.update_xaxes(
        title_text="Date",
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0'
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="<b>Pesticide Reduction (L)</b>",
        secondary_y=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0'
    )
    
    fig.update_yaxes(
        title_text="<b>Water Saved (L)</b>",
        secondary_y=True,
        showgrid=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ No time series data available. Run more diagnoses to see trends.")

st.markdown("<br>", unsafe_allow_html=True)

# Crop-wise breakdown
if history_data:
    st.markdown("### 🌾 Crop-wise Impact Breakdown")
    
    # Aggregate by crop
    crop_stats = {}
    for entry in history_data:
        crop = entry.get('crop', 'Unknown')
        if crop not in crop_stats:
            crop_stats[crop] = {
                'count': 0,
                'pesticide': 0,
                'water': 0,
                'diseases': []
            }
        crop_stats[crop]['count'] += 1
        crop_stats[crop]['pesticide'] += entry.get('pesticide_saved_l', 0)
        crop_stats[crop]['water'] += entry.get('water_saved_l', 0)
        crop_stats[crop]['diseases'].append(entry.get('disease', 'Unknown'))
    
    # Display crop statistics
    for crop, stats in crop_stats.items():
        with st.expander(f"🌱 {crop} - {stats['count']} scan(s)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Pesticide Saved", f"{stats['pesticide']:.1f}L")
            
            with col2:
                st.metric("Water Saved", f"{stats['water']:.1f}L")
            
            with col3:
                unique_diseases = len(set(stats['diseases']))
                st.metric("Diseases Found", unique_diseases)
            
            st.markdown("**Detected Diseases:**")
            disease_list = ', '.join(set(stats['diseases']))
            st.write(disease_list)

st.markdown("<br><br>", unsafe_allow_html=True)

# Disease frequency chart
if history_data:
    st.markdown("### 🦠 Most Common Diseases")
    
    # Count disease occurrences
    disease_counts = {}
    for entry in history_data:
        disease = entry.get('disease', 'Unknown')
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    # Sort by frequency
    sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_diseases:
        diseases = [d[0] for d in sorted_diseases[:5]]  # Top 5
        counts = [d[1] for d in sorted_diseases[:5]]
        
        # Create bar chart
        fig_diseases = go.Figure(data=[
            go.Bar(
                x=counts,
                y=diseases,
                orientation='h',
                marker=dict(
                    color=['#f44336', '#ff9800', '#ffc107', '#7cb342', '#4caf50'][:len(diseases)],
                    line=dict(color='white', width=2)
                ),
                text=counts,
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Occurrences: %{x}<extra></extra>'
            )
        ])
        
        fig_diseases.update_layout(
            title="Top 5 Most Detected Diseases",
            xaxis_title="Number of Occurrences",
            yaxis_title="Disease",
            plot_bgcolor='white',
            height=400,
            showlegend=False
        )
        
        fig_diseases.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        fig_diseases.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig_diseases, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Environmental impact comparison
st.markdown("### 🌍 Your Environmental Contribution")

comparison_col1, comparison_col2 = st.columns(2)

with comparison_col1:
    st.markdown(
        f"""
        <div class="metric-card" style="padding: 2rem; background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);">
            <h4 style="color: #2d5016; margin-bottom: 1rem;">💧 Water Conservation</h4>
            <p style="color: #666; line-height: 1.8; margin-bottom: 1rem;">
                You've saved <strong>{impact_data.get('water_saved_liters', 0):.1f} liters</strong> of water through 
                smart disease detection and targeted treatments.
            </p>
            <p style="color: #558b2f; font-size: 0.9rem;">
                ≈ {impact_data.get('water_saved_liters', 0) / 150:.1f} days of drinking water for a family
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with comparison_col2:
    st.markdown(
        f"""
        <div class="metric-card" style="padding: 2rem; background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%);">
            <h4 style="color: #01579b; margin-bottom: 1rem;">🧪 Chemical Reduction</h4>
            <p style="color: #666; line-height: 1.8; margin-bottom: 1rem;">
                You've reduced pesticide use by <strong>{impact_data.get('pesticide_reduction_pct', 0):.1f}%</strong> 
                compared to traditional methods.
            </p>
            <p style="color: #0277bd; font-size: 0.9rem;">
                Healthier soil, safer crops, better yields
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# Action buttons section
st.markdown("### ⚙️ Data Management")

action_col1, action_col2, action_col3 = st.columns([1, 1, 1])

with action_col1:
    if is_user_data:
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True, help="Delete all session data", key="clear_history_impact"):
            st.session_state['history'] = []
            st.session_state['last_result'] = None
            st.rerun()
    else:
        st.button("🗑️ Clear History", disabled=True, use_container_width=True, help="No user data to clear", key="clear_history_disabled_impact")

with action_col2:
    if st.button("🔄 Refresh Metrics", use_container_width=True, key="refresh_metrics_impact"):
        st.rerun()

with action_col3:
    if st.button("📥 Export Data", use_container_width=True, key="export_data_impact"):
        # Generate CSV export
        if history_data:
            import io
            
            # Create CSV content
            csv_lines = ["Timestamp,Crop,Disease,Confidence,Pesticide Saved (L),Water Saved (L)"]
            for entry in history_data:
                csv_lines.append(
                    f"{entry.get('timestamp', '')},{entry.get('crop', '')},{entry.get('disease', '')},"
                    f"{entry.get('confidence', 0):.1f},{entry.get('pesticide_saved_l', 0):.2f},"
                    f"{entry.get('water_saved_l', 0):.2f}"
                )
            csv_content = "\n".join(csv_lines)
            
            st.download_button(
                label="� Download CSV",
                data=csv_content,
                file_name="cropshield_impact_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No data to export")

st.markdown("<br><br>", unsafe_allow_html=True)

# Goals and achievements
st.markdown("### 🎯 Sustainability Goals")

goals_col1, goals_col2, goals_col3 = st.columns(3)

with goals_col1:
    water_goal = 5000  # liters
    water_progress = min(100, (impact_data.get('water_saved_liters', 0) / water_goal) * 100)
    st.markdown(
        f"""
        <div class="metric-card" style="padding: 1.5rem; text-align: center;">
            <h4 style="color: #2d5016; margin-bottom: 1rem;">💧 Water Goal</h4>
            <div style="font-size: 2rem; color: #2196f3; font-weight: bold; margin-bottom: 0.5rem;">
                {water_progress:.0f}%
            </div>
            <p style="color: #666; font-size: 0.9rem; margin: 0;">
                {impact_data.get('water_saved_liters', 0):.0f}L / {water_goal}L
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(water_progress / 100)

with goals_col2:
    scan_goal = 50
    scan_progress = min(100, (len(history_data) / scan_goal) * 100)
    st.markdown(
        f"""
        <div class="metric-card" style="padding: 1.5rem; text-align: center;">
            <h4 style="color: #2d5016; margin-bottom: 1rem;">🔬 Scan Goal</h4>
            <div style="font-size: 2rem; color: #7cb342; font-weight: bold; margin-bottom: 0.5rem;">
                {scan_progress:.0f}%
            </div>
            <p style="color: #666; font-size: 0.9rem; margin: 0;">
                {len(history_data)} / {scan_goal} scans
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(scan_progress / 100)

with goals_col3:
    yield_goal = 95  # percentage
    yield_current = impact_data.get('yield_preservation_pct', 0)
    yield_progress = min(100, (yield_current / yield_goal) * 100)
    st.markdown(
        f"""
        <div class="metric-card" style="padding: 1.5rem; text-align: center;">
            <h4 style="color: #2d5016; margin-bottom: 1rem;">🌾 Yield Goal</h4>
            <div style="font-size: 2rem; color: #ff9800; font-weight: bold; margin-bottom: 0.5rem;">
                {yield_progress:.0f}%
            </div>
            <p style="color: #666; font-size: 0.9rem; margin: 0;">
                {yield_current:.1f}% / {yield_goal}% preserved
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(yield_progress / 100)

st.markdown("<br><br>", unsafe_allow_html=True)

# Navigation
st.markdown("---")
st.markdown("### 🔗 Continue Your Journey")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("🌿 New Diagnosis", use_container_width=True, key="nav_diag_impact"):
        st.switch_page("pages/2_Diagnosis.py")

with nav_col2:
    if st.button("💧 View Recommendations", use_container_width=True, key="nav_rec_impact"):
        st.switch_page("pages/3_Recommendations.py")

with nav_col3:
    if st.button("❓ Get Help", use_container_width=True, key="nav_help_impact"):
        st.switch_page("pages/5_Help.py")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Quick Stats")
    
    st.metric("Total Impact Score", f"{min(100, len(history_data) * 5)}")
    
    st.markdown("---")
    
    st.markdown("### 🏆 Achievements")
    
    # Award badges based on history
    if len(history_data) >= 1:
        st.success("✅ First Scan Complete")
    if len(history_data) >= 5:
        st.success("✅ 5 Scans Milestone")
    if len(history_data) >= 10:
        st.success("✅ 10 Scans Milestone")
    if impact_data.get('water_saved_liters', 0) >= 1000:
        st.success("✅ Water Hero (1000L+)")
    
    if len(history_data) == 0:
        st.info("Complete scans to unlock achievements!")

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 1rem; color: #999; font-size: 0.9rem;">
        🌍 <strong>Every scan contributes to a more sustainable future for agriculture.</strong>
        Keep monitoring your crops to maximize impact!
    </div>
    """,
    unsafe_allow_html=True
)
