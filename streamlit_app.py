import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🌱",
    layout="wide"
)

MODEL_PATH = "models/"

required_files = [
    "autoencoder_soil.keras",
    "encoder_soil.keras",
    "isolation_forest_soil.pkl",
    "scaler_soil.pkl",
    "insight_soil.pkl"
]

missing = [f for f in required_files if not os.path.exists(MODEL_PATH + f)]
if missing:
    st.error(f"Missing files: {missing}")
    st.stop()

# =========================
# LOAD MODELS
# =========================
autoencoder = tf.keras.models.load_model(MODEL_PATH + "autoencoder_soil.keras", compile=False)
encoder = tf.keras.models.load_model(MODEL_PATH + "encoder_soil.keras", compile=False)
iso = joblib.load(MODEL_PATH + "isolation_forest_soil.pkl")
scaler = joblib.load(MODEL_PATH + "scaler_soil.pkl")
insight_data = joblib.load(MODEL_PATH + "insight_soil.pkl")

feature_cols = insight_data["feature_cols"]
mean_normal = insight_data["mean_normal"]
std_normal = insight_data["std_normal"]

# =========================
# STYLE (UNCHANGED)
# =========================
st.markdown("""
<style>
.block-container {padding: 1.5rem 2rem;}
.title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg,#16a34a,#0284c7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub {color:#64748b;}
.card {
    padding: 16px;
    border-radius: 16px;
    background: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

button[aria-label="Increment value"],
button[aria-label="Decrement value"] {
    background-color: #d1fae5 !important;
    color: #065f46 !important;
    border: none !important;
    border-radius: 6px !important;
}

button[aria-label="Increment value"]:hover,
button[aria-label="Decrement value"]:hover {
    background-color: #a7f3d0 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #16a34a, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 800 !important;
}

.metric-green div[data-testid="stMetricValue"] {
    color: #16a34a !important;
}

.metric-red div[data-testid="stMetricValue"] {
    color: #ef4444 !important;
}

div[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("</br><br><div class='title'>🌱 IoT Sensor Anomaly Detection Monitoring Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Hybrid Anomaly Detection using Autoencoder and Isolation Forest</div>", unsafe_allow_html=True)

# =========================
# INSIGHT FUNCTION (IMPROVED TEXT ONLY)
# =========================
def generate_status(row):
    values = row[feature_cols].values.astype(float)

    if np.sum(values == 0) == len(feature_cols):
        return (
            "Critical system alert: all sensors report zero values. "
            "This strongly indicates a power supply failure or the device is offline."
        )

    if np.any(values == 0):
        zero_params = [col for col in feature_cols if row[col] == 0]
        return (
            "Sensor malfunction detected. "
            f"No readings from: {', '.join(zero_params)}. "
            "Possible causes include sensor damage, loose wiring, or calibration issues."
        )

    if row["model_flag"] == 0:
        return "System status normal: all sensor readings are within expected range."

    z = {
        col: (row[col] - mean_normal[col]) / (std_normal[col] + 1e-9)
        for col in feature_cols
    }

    main = max(z, key=lambda x: abs(z[x]))
    high = z[main] >= 0

    mapping = {
        "hu": ["Soil moisture is below normal range.", "Soil moisture is above normal range."],
        "ta": ["Temperature is lower than expected.", "Temperature is higher than expected."],
        "ec": ["Electrical conductivity is lower than normal.", "Electrical conductivity is higher than normal."],
        "ph": ["Soil condition is more acidic than usual.", "Soil condition is more alkaline than usual."],
        "n": ["Nitrogen level is lower than expected.", "Nitrogen level is higher than expected."],
        "p": ["Phosphorus level is lower than expected.", "Phosphorus level is higher than expected."],
        "k": ["Potassium level is lower than expected.", "Potassium level is higher than expected."]
    }

    return mapping[main][1 if high else 0]

# =========================
# LAYOUT (UNCHANGED)
# =========================
left, right = st.columns([1.1, 1])

with left:
    st.markdown("<h3 style='color:#16a34a;'>Sensor Input</h3>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    hu = c1.number_input("Humidity", value=33.5)
    ta = c2.number_input("Temperature", value=25.6)
    ec = c3.number_input("Conductivity", value=650.0)

    ph = c1.number_input("pH Level", value=5.0)
    n = c2.number_input("Nitrogen", value=108.0)
    p = c3.number_input("Phosphorus", value=295.0)

    k = st.number_input("Potassium", value=288.0)

    run = st.button("Analyze Now", use_container_width=True)

with right:
    st.markdown("<h3 style='color:#16a34a;'>Insight Panel</h3>", unsafe_allow_html=True)
    placeholder = st.empty()

    if not run:
        with placeholder.container():
            st.info("Waiting for input...")
            c1, c2 = st.columns(2)
            c1.metric("Status", "-")
            c2.metric("Risk Score", "-")

# =========================
# ENGINE (UNCHANGED LOGIC)
# =========================
if run:

    input_dict = {
        "hu": hu,
        "ta": ta,
        "ec": ec,
        "ph": ph,
        "n": n,
        "p": p,
        "k": k
    }

    values = np.array([input_dict[col] for col in feature_cols])
    zero_params = [col for col in feature_cols if input_dict[col] == 0]

    sensor_issues = []

    if len(zero_params) == len(feature_cols):
        sensor_issues.append("Critical system failure: all sensors are inactive (zero readings).")
    elif len(zero_params) > 0:
        sensor_issues.append("Sensor malfunction detected on: " + ", ".join(zero_params))

    X = values.reshape(1, -1)
    X_scaled = scaler.transform(X)

    latent = encoder.predict(X_scaled, verbose=0)
    pred = int(iso.predict(latent)[0])
    score = float(-iso.decision_function(latent)[0])

    model_anomaly = (pred == -1)

    z_scores = {
        col: (val - mean_normal[col]) / (std_normal[col] + 1e-9)
        for col, val in zip(feature_cols, values)
    }

    abnormal_features = [col for col, z in z_scores.items() if abs(z) > 2.5]

    anomalies = []

    anomalies.extend(sensor_issues)

    if model_anomaly:
        anomalies.append("Anomaly detected: data pattern significantly deviates from normal behavior.")

    if abnormal_features:
        anomalies.append("Extreme variation detected in: " + ", ".join(abnormal_features))

    flag = len(anomalies) > 0

    row = pd.Series(values, index=feature_cols)
    row["model_flag"] = model_anomaly

    insight = " | ".join(anomalies) if anomalies else generate_status(row)

    # =========================
    # UI OUTPUT (UNCHANGED)
    # =========================
    with right:
        placeholder.empty()

        if sensor_issues:
            st.error("⚠️ SENSOR ISSUE DETECTED")
        elif model_anomaly:
            st.warning("⚠️ MODEL ANOMALY DETECTED")
        elif abnormal_features:
            st.warning("⚠️ FEATURE DEVIATION DETECTED")
        else:
            st.success("✅ SOIL CONDITION NORMAL")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"""
                <div style="color:#6b7280; font-size:13px;">Status</div>
                <div style="
                    font-size:26px;
                    font-weight:800;
                    color:{'#ef4444' if flag else '#16a34a'};
                ">
                    {'Anomaly' if flag else 'Normal'}
                </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div style="color:#6b7280; font-size:13px;">Risk Score</div>
            <div style="
                font-size:26px;
                font-weight:800;
                color:#000000;
            ">
                {score:.3f}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class='card'>
            <b>Detected Anomalies ({len(anomalies)})</b><br>
            {'<br>'.join(anomalies) if anomalies else 'No issues detected'}
            </div>
            """,
            unsafe_allow_html=True
        )