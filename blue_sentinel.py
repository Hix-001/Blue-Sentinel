"""
BLUE SENTINEL - UNIFIED REAL-TIME AI DEMONSTRATION (ENHANCED VERSION)
Government Presentation Version with Live Vessel Tracking & AI Predictions

This unified application combines:
- AI Model Training & Inference (LSTM)
- Real-time Vessel Tracking & Prediction
- Government-Ready Professional Presentation
- Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import tensorflow as tf
from tensorflow import keras
import random
from datetime import datetime, timedelta
import time

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Blue Sentinel - Real-Time AI Demo",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== GEOGRAPHIC REFERENCE DATA ====================
MAJOR_CITIES = [
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "type": "Major Port", "country": "India"},
    {"name": "Kochi", "lat": 9.9312, "lon": 76.2673, "type": "Naval Base", "country": "India"},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "type": "Major Port", "country": "India"},
    {"name": "Colombo", "lat": 6.9271, "lon": 79.8612, "type": "Capital/Port", "country": "Sri Lanka"},
    {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "type": "Naval HQ", "country": "India"},
    {"name": "Tuticorin", "lat": 8.8000, "lon": 78.1500, "type": "Port", "country": "India"},
    {"name": "Mangalore", "lat": 12.9141, "lon": 74.8560, "type": "Port", "country": "India"},
    {"name": "Goa", "lat": 15.2993, "lon": 74.1240, "type": "Naval Base", "country": "India"},
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011, "type": "Major Port", "country": "Pakistan"},
    {"name": "Malé", "lat": 4.1755, "lon": 73.5093, "type": "Capital", "country": "Maldives"},
]

ISLANDS_AND_REGIONS = [
    {"name": "Lakshadweep", "lat": 10.5667, "lon": 72.6417, "type": "Island Territory"},
    {"name": "Andaman & Nicobar", "lat": 11.7401, "lon": 92.6586, "type": "Island Territory"},
    {"name": "Minicoy Island", "lat": 8.2833, "lon": 73.0500, "type": "Island"},
]

WATER_BODIES = [
    {"name": "Arabian Sea", "lat": 16.0, "lon": 66.0},
    {"name": "Bay of Bengal", "lat": 15.0, "lon": 88.0},
    {"name": "Indian Ocean", "lat": 5.0, "lon": 75.0},
    {"name": "Laccadive Sea", "lat": 10.0, "lon": 74.0},
]

# ==================== STYLING ====================
st.markdown("""
    <style>
    /* Global Theme */
    .stApp {
        background-color: #0a0e17;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Header Styling */
    .demo-header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #1a1a2e 0%, #0a0e17 100%);
        border-radius: 15px;
        border: 2px solid rgba(0, 229, 255, 0.3);
        margin-bottom: 30px;
    }
    
    .demo-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 10px;
    }
    
    .demo-subtitle {
        font-size: 1.2rem;
        color: #00E5FF;
        letter-spacing: 2px;
    }
    
    /* Metrics */
    .metric-label {
        font-size: 0.8rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Risk Colors */
    .risk-high { color: #FF4B4B; }
    .risk-med { color: #FFA500; }
    .risk-low { color: #00E5FF; }
    
    /* Live Feed */
    .live-feed {
        background: rgba(0, 0, 0, 0.5);
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        color: #00ff00;
        height: 300px;
        overflow-y: auto;
        border: 1px solid rgba(0, 255, 0, 0.3);
    }
    
    .feed-item {
        padding: 5px 0;
        border-bottom: 1px solid rgba(0, 255, 0, 0.1);
    }
    
    /* Status Indicators */
    .status-active {
        color: #00ff00;
        font-weight: bold;
    }
    
    .status-warning {
        color: #FFA500;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #FF4B4B 0%, #ff1744 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #05080f;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Data Source Status */
    .data-source {
        padding: 10px;
        margin: 5px 0;
        background: rgba(0, 255, 0, 0.1);
        border-left: 3px solid #00ff00;
        border-radius: 5px;
    }
    
    /* Map Legend */
    .map-legend {
        background: rgba(0, 0, 0, 0.7);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 10px;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 10px;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== AI MODEL SETUP ====================
@st.cache_resource
def load_ai_model():
    """Load the trained LSTM vessel prediction model"""
    try:
        model = keras.models.load_model("vessel_predictor.keras")
        return model, True
    except Exception as e:
        st.warning(f"Model file not found. Running in simulation mode. ({e})")
        return None, False

# ==================== DATA SIMULATION ====================
class VesselTracker:
    """Real-time vessel tracking and prediction system"""
    
    def __init__(self, mmsi, name, lat, lon, vessel_type, flag):
        self.mmsi = mmsi
        self.name = name
        self.lat = lat
        self.lon = lon
        self.vessel_type = vessel_type
        self.flag = flag
        self.sog = random.uniform(2.0, 15.0)
        self.cog = random.uniform(0, 360)
        self.nav_status = random.choice(["Under way using engine", "Fishing", "At anchor"])
        
        # Risk indicators
        self.dark_periods = random.choices([0, 1, 2, 3, 5], weights=[0.6, 0.2, 0.1, 0.05, 0.05])[0]
        self.lstm_anomaly = random.random()
        self.mpa_proximity = random.uniform(0, 50)
        
        # Historical track (last 10 positions)
        self.history = self._generate_history()
        
        # Prediction results
        self.predicted_lat = None
        self.predicted_lon = None
        self.risk_score = 0.0
        self.risk_level = "LOW"
        self.risk_reasons = []
        self.predicted_activity = "Normal Transit"
        
        # Geographic context
        self.region = self._determine_region()
    
    def _determine_region(self):
        """Determine which region/water body the vessel is in"""
        if self.lon < 74:
            return "Arabian Sea (Western Coast)"
        elif self.lon > 85:
            return "Bay of Bengal (Eastern Coast)"
        elif self.lat < 8:
            return "Indian Ocean (Southern Waters)"
        else:
            return "Laccadive Sea (Southwest India)"
    
    def _generate_history(self):
        """Generate historical track data"""
        history = []
        curr_lat, curr_lon = self.lat, self.lon
        for _ in range(10):
            # Add some random walk
            curr_lat += np.random.uniform(-0.02, 0.02)
            curr_lon += np.random.uniform(-0.02, 0.02)
            history.append([curr_lat, curr_lon, self.sog, self.cog])
        return history
    
    def update_position(self):
        """Simulate real-time position update"""
        # Move vessel based on speed and course
        self.lat += (self.sog * 0.005) * np.cos(np.radians(self.cog))
        self.lon += (self.sog * 0.005) * np.sin(np.radians(self.cog))
        
        # Add some randomness
        self.sog += np.random.uniform(-0.3, 0.3)
        self.sog = max(0.5, min(20, self.sog))
        self.cog += np.random.uniform(-5, 5)
        self.cog = self.cog % 360
        
        # Update history
        self.history.append([self.lat, self.lon, self.sog, self.cog])
        if len(self.history) > 10:
            self.history.pop(0)
        
        # Update region
        self.region = self._determine_region()
    
    def predict_trajectory(self, model=None):
        """Run AI prediction on vessel trajectory"""
        if model is not None:
            # Use actual LSTM model
            track_array = np.array([self.history])
            prediction = model.predict(track_array, verbose=0)
            self.predicted_lat, self.predicted_lon = prediction[0]
        else:
            # Simulation mode
            self.predicted_lat = self.lat + (self.sog * 0.02) * np.cos(np.radians(self.cog))
            self.predicted_lon = self.lon + (self.sog * 0.02) * np.sin(np.radians(self.cog))
        
        # Calculate risk score
        self._calculate_risk()
    
    def _calculate_risk(self):
        """Calculate IUU risk score based on multiple factors"""
        score = 0.0
        reasons = []
        
        # LSTM Anomaly Score (0-35 points)
        lstm_contrib = self.lstm_anomaly * 35
        score += lstm_contrib
        if self.lstm_anomaly > 0.7:
            reasons.append(f"Abnormal Maneuvering Detected ({self.lstm_anomaly:.2f})")
        
        # Dark Ship Events (0-20 points)
        if self.dark_periods > 2:
            score += 20
            reasons.append(f"Dark Ship Pattern ({self.dark_periods} events in 24h)")
        
        # Loitering Behavior (0-15 points)
        if self.vessel_type in ["Fishing", "Trawler"] and self.sog < 2.0:
            score += 15
            reasons.append("Loitering Behavior (Fishing Probability: 92%)")
        
        # MPA Proximity (0-25 points)
        if self.mpa_proximity < 10:
            score += 25
            reasons.append(f"Approaching Protected Zone ({self.mpa_proximity:.1f} km)")
        
        # Unknown Flag (0-10 points)
        if self.flag == "Unknown":
            score += 10
            reasons.append("Unknown Flag State")
        
        # Trajectory Deviation (0-15 points)
        if self.predicted_lat and self.predicted_lon:
            deviation = np.sqrt((self.predicted_lat - self.lat)**2 + (self.predicted_lon - self.lon)**2)
            if deviation > 0.1:
                score += 15
                reasons.append(f"Unusual Trajectory Deviation ({deviation*111:.1f} km)")
        
        # Normalize to 0-100
        self.risk_score = min(100.0, score)
        self.risk_reasons = reasons
        
        # Determine risk level and activity
        if self.risk_score > 75:
            self.risk_level = "HIGH"
            self.predicted_activity = "Illegal Fishing (High Probability)"
        elif self.risk_score > 40:
            self.risk_level = "MEDIUM"
            self.predicted_activity = "Suspicious Transit"
        else:
            self.risk_level = "LOW"
            self.predicted_activity = "Normal Operations"

def generate_demo_vessels(n=30):
    """Generate a fleet of vessels for demonstration"""
    vessels = []
    
    # Define some preset high-risk vessels for demo
    high_risk_vessels = [
        {"mmsi": 419852630, "name": "EASTERN STAR", "lat": 8.52, "lon": 76.91, "type": "Fishing Trawler", "flag": "Unknown"},
        {"mmsi": 412589630, "name": "OCEAN HUNTER", "lat": 7.85, "lon": 75.50, "type": "Fishing", "flag": "Unknown"},
        {"mmsi": 423156789, "name": "DARK PEARL", "lat": 9.10, "lon": 78.20, "type": "Trawler", "flag": "Unknown"},
    ]
    
    for vessel_data in high_risk_vessels:
        vessel = VesselTracker(
            mmsi=vessel_data["mmsi"],
            name=vessel_data["name"],
            lat=vessel_data["lat"],
            lon=vessel_data["lon"],
            vessel_type=vessel_data["type"],
            flag=vessel_data["flag"]
        )
        # Make these high risk
        vessel.dark_periods = random.randint(3, 5)
        vessel.lstm_anomaly = random.uniform(0.7, 0.95)
        vessels.append(vessel)
    
    # Generate remaining vessels
    for i in range(n - len(high_risk_vessels)):
        vessel = VesselTracker(
            mmsi=random.randint(200000000, 700000000),
            name=f"VESSEL_{random.randint(1000, 9999)}",
            lat=random.uniform(5.0, 12.0),
            lon=random.uniform(72.0, 80.0),
            vessel_type=random.choice(["Cargo", "Tanker", "Fishing", "Passenger"]),
            flag=random.choice(["India", "Sri Lanka", "China", "Panama", "Liberia", "Unknown"])
        )
        vessels.append(vessel)
    
    return vessels

# ==================== UI COMPONENTS ====================

def render_header():
    """Render professional government demo header"""
    st.markdown("""
        <div class="demo-header">
            <div class="demo-title">🛡️ BLUE SENTINEL AI SYSTEM</div>
            <div class="demo-subtitle">Maritime Surveillance & Predictive Intelligence</div>
            <p style="color: #888; margin-top: 10px; font-size: 0.9rem;">
                Government of India - Ministry of Defence | Indian Navy & Coast Guard
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_data_note():
    """Render note about data source (SIMULATION vs REAL)"""
    st.markdown("""
        <div class="glass-card" style="background: rgba(255, 165, 0, 0.1); border-color: rgba(255, 165, 0, 0.3);">
            <h4 style="color: #FFA500; margin-top: 0;">📊 DATA SOURCE INFORMATION</h4>
            <p><strong>Current Mode:</strong> SIMULATION / DEMONSTRATION</p>
            <p>The 30 vessels displayed are <strong>SIMULATED</strong> for demonstration purposes. They are NOT real live vessel positions.</p>
            <p><strong>How it works:</strong></p>
            <ul style="margin-left: 20px;">
                <li>Mock data engine generates realistic vessel behavior patterns</li>
                <li>AI risk scoring uses actual algorithms that would work with real AIS data</li>
                <li>Vessel movements are simulated based on maritime navigation patterns</li>
            </ul>
            <p><strong>For REAL LIVE tracking, the system would integrate with:</strong></p>
            <ul style="margin-left: 20px;">
                <li>🛰️ AIS (Automatic Identification System) networks - MarineTraffic API, VesselFinder, AISHub</li>
                <li>🛰️ GSAT-7 Indian Naval Satellite for regional coverage</li>
                <li>🛰️ Indian Coast Guard AIS shore stations</li>
                <li>🛰️ ISRO Earth Observation satellites</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def render_data_sources():
    """Render data source connection status"""
    st.markdown("### 📡 Data Sources")
    st.caption("(Demo Mode - Simulated)")
    
    sources = [
        {"name": "GSAT-7 Satellite", "status": "SIMULATED", "coverage": "Indian Ocean"},
        {"name": "Spire Maritime AIS", "status": "SIMULATED", "coverage": "Global"},
        {"name": "Coast Guard AIS", "status": "SIMULATED", "coverage": "India"},
        {"name": "ISRO EO Satellites", "status": "SIMULATED", "coverage": "Regional"},
    ]
    
    for source in sources:
        st.markdown(f"""
            <div class="data-source">
                <strong>{source['name']}</strong><br>
                <span style="color: #FFA500;">[{source['status']}]</span> | {source['coverage']}
            </div>
        """, unsafe_allow_html=True)

def render_system_metrics(vessels, model_loaded):
    """Render system performance metrics"""
    high_risk = sum(1 for v in vessels if v.risk_level == "HIGH")
    medium_risk = sum(1 for v in vessels if v.risk_level == "MEDIUM")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">High Risk Targets</div>
                <div class="metric-value risk-high">{high_risk}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Medium Risk Targets</div>
                <div class="metric-value risk-med">{medium_risk}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Vessels Tracked</div>
                <div class="metric-value risk-low">{len(vessels)}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        status_text = "AI ONLINE" if model_loaded else "SIMULATION"
        status_color = "risk-low" if model_loaded else "risk-med"
        st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">AI Status</div>
                <div class="metric-value {status_color}">{status_text}</div>
            </div>
        """, unsafe_allow_html=True)

def render_live_analysis_feed(vessels):
    """Render live AI analysis feed"""
    st.markdown("### 🧠 Live AI Analysis Feed")
    
    # Get recent high-risk vessels
    high_risk_vessels = [v for v in vessels if v.risk_level in ["HIGH", "MEDIUM"]]
    high_risk_vessels = sorted(high_risk_vessels, key=lambda x: x.risk_score, reverse=True)[:10]
    
    feed_html = '<div class="live-feed">'
    current_time = datetime.now()
    
    for i, vessel in enumerate(high_risk_vessels):
        timestamp = (current_time - timedelta(seconds=i*3)).strftime('%H:%M:%S')
        icon = "🚨" if vessel.risk_level == "HIGH" else "⚠️"
        feed_html += f"""
            <div class="feed-item">
                {icon} [{timestamp}] ANALYZING MMSI {vessel.mmsi} ({vessel.name})<br>
                &nbsp;&nbsp;&nbsp;&nbsp;→ Risk Level: {vessel.risk_level} ({vessel.risk_score:.1f}%)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;→ Activity: {vessel.predicted_activity}<br>
                &nbsp;&nbsp;&nbsp;&nbsp;→ Location: {vessel.region}
            </div>
        """
    
    feed_html += '</div>'
    st.markdown(feed_html, unsafe_allow_html=True)

def render_map(vessels):
    """Render interactive map with vessel positions and geographic context"""
    st.markdown("### 🗺️ Real-Time Vessel Tracking Map")
    
    # Prepare vessel data for map
    map_data = []
    
    for v in vessels:
        # Color coding by risk level
        if v.risk_level == "HIGH":
            color = [255, 75, 75, 200]
        elif v.risk_level == "MEDIUM":
            color = [255, 165, 0, 200]
        else:
            color = [0, 229, 255, 150]
        
        map_data.append({
            "name": str(v.name),
            "mmsi": int(v.mmsi),
            "lat": float(v.lat),
            "lon": float(v.lon),
            "color": color,
            "risk": float(v.risk_score),
            "activity": str(v.predicted_activity),
            "region": str(v.region)
        })
    
    df_map = pd.DataFrame(map_data)
    
    # Simplified geographic labels - only major ports
    city_labels = []
    major_ports = ["Mumbai", "Kochi", "Chennai", "Colombo"]
    for city in MAJOR_CITIES:
        if city['name'] in major_ports:
            city_labels.append({
                "name": city['name'],
                "coordinates": [float(city['lon']), float(city['lat'])],
                "color": [255, 255, 100, 200],
                "size": 10
            })
    
    # Indian EEZ boundary (simplified)
    eez_boundary = [{
        "path": [
            [68.0, 8.0], [72.0, 10.0], [74.0, 12.0], [76.0, 12.5], 
            [78.0, 12.0], [80.0, 10.0], [82.0, 8.0], [82.0, 6.0],
            [80.0, 4.0], [76.0, 3.0], [72.0, 4.0], [68.0, 6.0], [68.0, 8.0]
        ],
        "color": [0, 255, 100, 80]
    }]
    
    # Coastline reference (simplified)
    coastline_india = [{
        "path": [
            [72.8, 19.0], [73.0, 17.0], [74.0, 15.0], [74.5, 13.0],
            [75.0, 12.0], [76.0, 10.0], [76.5, 9.0], [77.0, 8.5],
            [78.0, 8.2], [79.0, 9.0], [80.0, 10.0], [80.5, 13.0],
            [82.0, 16.0], [83.0, 17.5]
        ],
        "color": [255, 255, 255, 100]
    }]
    
    coastline_srilanka = [{
        "path": [
            [79.5, 9.5], [80.0, 8.5], [81.0, 7.5], [81.5, 6.5],
            [80.5, 6.0], [79.8, 6.5], [79.5, 7.5], [79.5, 9.5]
        ],
        "color": [255, 255, 255, 100]
    }]
    
    # Create layers (simplified for performance)
    layers = []
    
    # 1. EEZ Boundary
    eez_layer = pdk.Layer(
        "PathLayer",
        data=eez_boundary,
        get_path="path",
        get_color="color",
        width_scale=20,
        width_min_pixels=2,
        get_width=2,
    )
    layers.append(eez_layer)
    
    # 2. Coastlines
    coastline_layer_india = pdk.Layer(
        "PathLayer",
        data=coastline_india,
        get_path="path",
        get_color="color",
        width_scale=30,
        width_min_pixels=3,
        get_width=3,
    )
    layers.append(coastline_layer_india)
    
    coastline_layer_sl = pdk.Layer(
        "PathLayer",
        data=coastline_srilanka,
        get_path="path",
        get_color="color",
        width_scale=30,
        width_min_pixels=3,
        get_width=3,
    )
    layers.append(coastline_layer_sl)
    
    # 3. Vessel positions
    vessel_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius=15000,
        pickable=True,
    )
    layers.append(vessel_layer)
    
    # 4. Major port labels (simplified)
    if city_labels:
        city_layer = pdk.Layer(
            "TextLayer",
            data=city_labels,
            get_position="coordinates",
            get_text="name",
            get_color="color",
            get_size="size",
            get_alignment_baseline="'bottom'",
            get_pixel_offset=[0, -10],
        )
        layers.append(city_layer)
    
    # Set view state with better initial positioning
    view_state = pdk.ViewState(
        latitude=10.0,
        longitude=77.0,
        zoom=5.5,
        pitch=0,
        bearing=0
    )
    
    # Create deck
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={
            "html": "<b>{name}</b><br/>MMSI: {mmsi}<br/>Risk: {risk}%<br/>Activity: {activity}<br/>Region: {region}",
            "style": {
                "backgroundColor": "rgba(0, 0, 0, 0.9)",
                "color": "white",
                "fontSize": "12px",
                "padding": "10px",
                "borderRadius": "5px"
            }
        }
    )
    
    st.pydeck_chart(deck)
    
    # Map legend
    st.markdown("""
        <div class="map-legend">
            <h4 style="margin-top: 0; color: #00E5FF;">Map Legend</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                <div class="legend-item">
                    <div class="legend-color" style="background: rgb(255, 75, 75);"></div>
                    <span>High Risk Vessel</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgb(255, 165, 0);"></div>
                    <span>Medium Risk Vessel</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgb(0, 229, 255);"></div>
                    <span>Low Risk Vessel</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def main():
    render_header()
    render_data_note()
    
    model, model_loaded = load_ai_model()
    
    # Initialize refresh rate in session state
    if 'refresh_rate' not in st.session_state:
        st.session_state.refresh_rate = 3
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/radar.png", width=80)
        st.title("Control Center")
        render_data_sources()
        st.markdown("---")
        st.markdown("### ⚙️ Demo Controls")
        
        # Refresh rate slider
        refresh_rate = st.slider(
            "Auto-Refresh Rate (seconds)",
            min_value=1,
            max_value=10,
            value=st.session_state.refresh_rate,
            help="How often the simulation updates"
        )
        st.session_state.refresh_rate = refresh_rate
        
        if st.button("🔄 Regenerate Simulation"):
            if 'vessels' in st.session_state:
                del st.session_state.vessels
            st.rerun()
        st.markdown("---")
        st.caption("Blue Sentinel Demo v2.0")

    # Logic
    if 'vessels' not in st.session_state:
        st.session_state.vessels = generate_demo_vessels()
    
    # Update positions
    for v in st.session_state.vessels:
        v.update_position()
        v.predict_trajectory(model)
        
    vessels = st.session_state.vessels
    
    render_system_metrics(vessels, model_loaded)
    
    col_map, col_feed = st.columns([2, 1])
    with col_map:
        render_map(vessels)
    with col_feed:
        render_live_analysis_feed(vessels)
    
    # Vessel Data Table
    st.markdown("### 📊 Vessel Data Table")
    vessel_data = []
    for v in vessels:
        vessel_data.append({
            "MMSI": v.mmsi,
            "Name": v.name,
            "Type": v.vessel_type,
            "Flag": v.flag,
            "Lat": f"{v.lat:.4f}",
            "Lon": f"{v.lon:.4f}",
            "Speed (kn)": f"{v.sog:.1f}",
            "Course": f"{v.cog:.0f}°",
            "Risk": f"{v.risk_score:.1f}%",
            "Risk Level": v.risk_level,
            "Region": v.region
        })
    
    df_vessels = pd.DataFrame(vessel_data)
    st.dataframe(df_vessels, use_container_width=True, height=400)
    
    # Auto-refresh with configurable rate
    time.sleep(st.session_state.refresh_rate)
    st.rerun()

if __name__ == "__main__":
    main()
