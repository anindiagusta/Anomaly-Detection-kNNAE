import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Anomaly Detection IoT Sensor Data",
    page_icon="🌱",
    layout="wide"
)

# =====================================================
# STYLE
# =====================================================
st.markdown("""
<style>
.block-container{
    padding:1rem 2rem;
}

/* input kecil */
input[type=number]{
    height:35px;
    font-size:13px;
}

/* title */
.main-title{
    font-size:34px;
    font-weight:800;
    background:linear-gradient(90deg,#16a34a,#0284c7);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.sub-title{
    color:#64748b;
    margin-bottom:10px;
}

/* metric */
.metric-box{
    background:#f8fafc;
    border-radius:14px;
    padding:12px;
    text-align:center;
}

.small{
    font-size:12px;
    color:#64748b;
}

/* tombol full lebar */
.stButton > button {
    width: 100% !important;
    display: block;
    background: linear-gradient(90deg,#16a34a,#0284c7);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.6rem;
}

/* paksa container tombol ikut full */
div.stButton {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.write("")
st.write("")
st.markdown("<div class='main-title'>🌱 Anomaly Detection IoT Sensor Data</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Hybrid Anomaly Detection using Autoencoder and Isolation Forest</div>", unsafe_allow_html=True)

# =====================================================
# DEFAULT DATA
# =====================================================
manual_tests = {
    "S1":  [33.5, 25.6, 650, 5.0, 108, 295, 288],
    "S2":  [41.2, 24.9, 410, 5.4, 72, 210, 202],
    "S3":  [0, 0, 0, 0, 0, 0, 0],
    "S4":  [33.5, 25.6, 0, 5.0, 108, 295, 288],
    "S5":  [80, 38, 1200, 8.5, 300, 500, 450],
    "S6":  [12, 18, 90, 3.5, 5, 20, 15],
    "S7":  [12, 18, 90, 3.5, 5, 20, 15]
}

features = ["hu", "ta", "ec", "ph", "n", "p", "k"]
sensor_ids = list(manual_tests.keys())

# =====================================================
# LAYOUT
# =====================================================
left, right = st.columns([1.4, 1])

# =====================================================
# INPUT TABLE
# =====================================================
with left:

    st.markdown("### Sensor Input")

    # Header
    header = st.columns(8)
    header[0].markdown("**Sensor**")
    for i, f in enumerate(features):
        header[i+1].markdown(f"**{f.upper()}**")

    sensor_data = []

    for s in sensor_ids:

        cols = st.columns(8)
        cols[0].markdown(f"**{s}**")

        row = []
        for i, f in enumerate(features):
            val = cols[i+1].number_input(
                "",
                value=float(manual_tests[s][i]),
                key=f"{s}_{f}",
                label_visibility="collapsed"
            )
            row.append(val)

        sensor_data.append(row)

    run = st.button("Analyze Now")

# =====================================================
# DEFAULT RIGHT
# =====================================================
with right:
    if not run:
        st.markdown("### Analysis Result")
        st.info("Waiting for input...")

# =====================================================
# ENGINE
# =====================================================
if run:

    X = np.array(sensor_data)

    # -----------------------------
    # RULE BASE (ZERO = HARD ANOMALY)
    # -----------------------------
    rule_alerts = []
    forced_anomaly_idx = set()

    for i, row in enumerate(X):

        if np.all(row == 0):
            rule_alerts.append(f"{sensor_ids[i]} mati (semua parameter 0)")
            forced_anomaly_idx.add(i)

        elif np.any(row == 0):
            rule_alerts.append(f"{sensor_ids[i]} ada parameter bernilai 0")
            forced_anomaly_idx.add(i)

    # -----------------------------
    # SCALING
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # AUTOENCODER
    # -----------------------------
    inp = Input(shape=(7,))
    x = Dense(5, activation="relu")(inp)
    x = Dense(3, activation="relu")(x)
    x = Dense(5, activation="relu")(x)
    out = Dense(7)(x)

    ae = Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(X_scaled, X_scaled, epochs=50, verbose=0)

    recon = ae.predict(X_scaled, verbose=0)
    errors = np.mean(np.square(X_scaled - recon), axis=1)

    # -----------------------------
    # ISOLATION FOREST
    # -----------------------------
    iso = IsolationForest(contamination=0.2, random_state=42)
    preds = iso.fit_predict(X_scaled)
    scores = -iso.decision_function(X_scaled)

    # -----------------------------
    # DETECTION
    # -----------------------------
    anomaly_idx = set()
    messages = []

    for i in range(len(sensor_ids)):

        if preds[i] == -1 or errors[i] > np.mean(errors) + 2*np.std(errors):
            anomaly_idx.add(i)
            messages.append(f"{sensor_ids[i]} berbeda dari pola sensor lain")

    # 🔥 OVERRIDE RULE
    anomaly_idx = anomaly_idx.union(forced_anomaly_idx)

    # -----------------------------
    # FINAL
    # -----------------------------
    final_alerts = rule_alerts + messages
    flag = len(anomaly_idx) > 0
    worst_sensor = int(np.argmax(scores))
    total_score = float(np.sum(scores))

    # =====================================================
    # OUTPUT
    # =====================================================
    with right:

        st.markdown("### Analysis Result")

        if flag:
            st.error("⚠️ Anomaly Detected")
        else:
            st.success("✅ All Sensors Normal")

        c1, c2 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class='metric-box'>
            <div class='small'>Risk Score</div>
            <h3>{total_score:.3f}</h3>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='metric-box'>
            <div class='small'>Highest Risk</div>
            <h3>{sensor_ids[worst_sensor]}</h3>
            </div>
            """, unsafe_allow_html=True)

        # SENSOR STATUS
        st.markdown("### Sensor Status")
        cols = st.columns(2)

        for i in range(7):
            if i in anomaly_idx:
                cols[i % 2].warning(f"{sensor_ids[i]} anomaly")
            else:
                cols[i % 2].success(f"{sensor_ids[i]} normal")

        # DETAIL
        st.markdown("### Detailed Insight")

        if final_alerts:
            for msg in final_alerts:
                st.write("•", msg)
        else:
            st.write("All sensors consistent.")