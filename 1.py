import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Tracker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .day-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    
    .energy-tip {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'daily_data' not in st.session_state:
    st.session_state.daily_data = {}
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}

# Main header
st.markdown('<h1 class="main-header">⚡ Energy Consumption Tracker</h1>', unsafe_allow_html=True)

# Sidebar for user information
with st.sidebar:
    st.markdown("### 👤 Personal Information")
    
    name = st.text_input("📝 Enter your name:", placeholder="Your name here...")
    age = st.number_input("🎂 Enter your age:", min_value=1, max_value=120, value=25)
    city = st.text_input("🏙️ Enter your city:", placeholder="City name...")
    area = st.text_input("📍 Enter your area name:", placeholder="Area/Locality...")
    flat_tenament = st.selectbox("🏠 Living in:", ["Flat", "Tenament"])
    facility = st.selectbox("🏡 BHK Type:", ["1BHK", "2BHK", "3BHK"])
    
    # Store user info
    st.session_state.user_info = {
        'name': name,
        'age': age,
        'city': city,
        'area': area,
        'flat_tenament': flat_tenament,
        'facility': facility
    }

# Main content area
if name:  # Only show if name is entered
    st.markdown(f"### Welcome, {name}! 👋")
    
    # Display user info cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏠 Home Type</h3>
            <p>{facility} {flat_tenament}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📍 Location</h3>
            <p>{area}, {city}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👤 Age</h3>
            <p>{age} years</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Calculate base energy based on BHK type
    def calculate_base_energy(bhk_type):
        if bhk_type == "1BHK":
            return 2 * 0.4 + 2 * 0.8  # 2.4 kWh
        elif bhk_type == "2BHK":
            return 3 * 0.4 + 3 * 0.8  # 3.6 kWh
        elif bhk_type == "3BHK":
            return 4 * 0.4 + 4 * 0.8  # 4.8 kWh
        return 0
    
    base_energy = calculate_base_energy(facility)
    
    # Daily energy tracking
    st.markdown("### 📅 Daily Energy Consumption")
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Create tabs for each day
    tabs = st.tabs(days)
    
    for i, day in enumerate(days):
        with tabs[i]:
            st.markdown(f'<div class="day-section">', unsafe_allow_html=True)
            st.markdown(f"#### {day}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Base Consumption:**")
                st.info(f"Lights & Fans: {base_energy:.1f} kWh")
                
                # Appliance usage
                st.markdown("**Additional Appliances:**")
                ac_usage = st.checkbox(f"🌡️ Air Conditioner", key=f"ac_{day}")
                fridge_usage = st.checkbox(f"❄️ Refrigerator", key=f"fridge_{day}")
                wm_usage = st.checkbox(f"🧺 Washing Machine", key=f"wm_{day}")
            
            with col2:
                # Calculate daily energy
                daily_energy = base_energy
                additional_energy = 0
                
                if ac_usage:
                    additional_energy += 2
                    daily_energy += 2
                if fridge_usage:
                    additional_energy += 2
                    daily_energy += 2
                if wm_usage:
                    additional_energy += 2
                    daily_energy += 2
                
                # Display energy breakdown
                st.markdown("**Energy Breakdown:**")
                
                # Create a mini pie chart
                if additional_energy > 0:
                    fig = go.Figure(data=[go.Pie(
                        labels=['Base (Lights & Fans)', 'Additional Appliances'],
                        values=[base_energy, additional_energy],
                        hole=0.3,
                        marker_colors=['#ff7f0e', '#1f77b4']
                    )])
                    fig.update_layout(
                        height=200,
                        margin=dict(t=20, b=20, l=20, r=20),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display total
                st.metric(
                    label="Total Daily Energy",
                    value=f"{daily_energy:.1f} kWh",
                    delta=f"+{additional_energy:.1f} kWh from appliances" if additional_energy > 0 else "Base consumption only"
                )
            
            # Store daily data
            st.session_state.daily_data[day] = {
                'base_energy': base_energy,
                'ac': ac_usage,
                'fridge': fridge_usage,
                'washing_machine': wm_usage,
                'total_energy': daily_energy
            }
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Weekly summary
    st.markdown("---")
    st.markdown("### 📊 Weekly Energy Summary")
    
    # Calculate weekly total
    weekly_total = sum([data['total_energy'] for data in st.session_state.daily_data.values()])
    
    # Create summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔋 Total Weekly Energy", f"{weekly_total:.1f} kWh")
    
    with col2:
        avg_daily = weekly_total / 7 if weekly_total > 0 else 0
        st.metric("📈 Average Daily", f"{avg_daily:.1f} kWh")
    
    with col3:
        estimated_cost = weekly_total * 5  # Assuming ₹5 per kWh
        st.metric("💰 Estimated Weekly Cost", f"₹{estimated_cost:.0f}")
    
    with col4:
        monthly_projection = weekly_total * 4.33
        st.metric("🗓️ Monthly Projection", f"{monthly_projection:.1f} kWh")
    
    # Weekly chart
    if st.session_state.daily_data:
        chart_data = []
        for day, data in st.session_state.daily_data.items():
            chart_data.append({
                'Day': day,
                'Base Energy': data['base_energy'],
                'Additional Energy': data['total_energy'] - data['base_energy'],
                'Total Energy': data['total_energy']
            })
        
        df = pd.DataFrame(chart_data)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Base Energy (Lights & Fans)',
            x=df['Day'],
            y=df['Base Energy'],
            marker_color='#ff7f0e'
        ))
        
        fig.add_trace(go.Bar(
            name='Additional Appliances',
            x=df['Day'],
            y=df['Additional Energy'],
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title='Daily Energy Consumption Breakdown',
            xaxis_title='Day of Week',
            yaxis_title='Energy (kWh)',
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Energy saving tips
        st.markdown("### 💡 Energy Saving Tips")
        
        tips = [
            "🌡️ Set AC temperature to 24°C or higher to save energy",
            "💡 Use LED bulbs - they consume 80% less energy than incandescent bulbs",
            "❄️ Keep refrigerator at optimal temperature (37-40°F)",
            "🧺 Use washing machine with full loads to maximize efficiency",
            "🌙 Unplug electronics when not in use to avoid phantom loads",
            "🏠 Improve insulation to reduce heating/cooling costs"
        ]
        
        for tip in tips:
            st.markdown(f'<div class="energy-tip">{tip}</div>', unsafe_allow_html=True)

else:
    st.info("👈 Please enter your name in the sidebar to get started!")
    
    # Show some preview content
    st.markdown("### ⚡ About This App")
    st.write("""
    This Energy Consumption Tracker helps you:
    - 📊 Monitor daily energy usage
    - 💰 Calculate estimated electricity costs
    - 📈 Visualize consumption patterns
    - 💡 Get energy-saving tips
    - 🎯 Track weekly and monthly projections
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Track your energy, save the planet! 🌍")