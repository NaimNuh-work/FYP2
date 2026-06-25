import streamlit as st
import pandas as pd
import numpy as np
import cv2
import time

# --- Page Configuration ---
# Setting layout to "wide" is crucial to fit everything like the image
st.set_page_config(page_title="RODEND Edge AI Node", page_icon="🧪", layout="wide")

# --- Custom CSS for Styling (Optional) ---
# Streamlit has a default theme, but you can force dark sidebars and specific colors here
st.markdown("""
    <style>
    .metric-container { background-color: #f8f9fa; border-radius: 10px; padding: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🧪 RODEND")
st.sidebar.caption("EDGE AI NODE")
st.sidebar.radio("Navigation", ["Control Dashboard", "Camera Manager", "Analytics"])
st.sidebar.markdown("---")
st.sidebar.success("Node Status: Active\n\nHost IP: 10.19.27.76")

# --- TOP HEADER ---
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.write("**Station Code:** FAC-NORTH-1A")
with header_col2:
    st.write("🟢 AI Engine Running")

st.markdown("---")

# --- METRICS ROW ---
# Creates 4 equal columns for the top stats
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(label="LIVE SAFETY STATE", value="Danger", delta="Critical: High activity triggers detected.",
              delta_color="inverse")
with m2:
    st.metric(label="TOTAL DETECTIONS (24H)", value="59", delta="Active surveillance ports", delta_color="off")
with m3:
    st.metric(label="CONFIDENCE CUTOFF", value="50%", delta="Adjust to suppress false triggers", delta_color="off")
with m4:
    st.metric(label="CONNECTED SENSORS", value="1 / 1", delta="CAM 1 ONLINE (RTSP/USB)", delta_color="normal")

st.markdown("---")

# --- MAIN CONTENT: VIDEO & SIDE PANEL ---
# Splits the middle section: 70% width for video, 30% for controls
col_video, col_panel = st.columns([2.5, 1])

with col_video:
    st.write("🔴 **LIVE OPERATIONS VIDEO FEED**")

    # The placeholder where video frames will be continuously injected
    video_placeholder = st.empty()

    # --- LIVE VIDEO LOGIC ---
    # NOTE: Replace "video.mp4" with an IP camera RTSP link like "rtsp://192.168.1.100:554/stream"
    # Or use 0 for local testing only.
    start_feed = st.button("Start Live Feed")

    if start_feed:
        # Using a dummy video or RTSP link.
        cap = cv2.VideoCapture("video.mp4")  # CHANGE THIS to your RTSP link or video file

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Video stream ended or connection lost.")
                break

            # Convert BGR (OpenCV) to RGB (Streamlit)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Insert your ResNet/U-Net prediction logic here ---
            # result = model.predict(frame)
            # frame = cv2.rectangle(frame, ...)

            # Display frame
            video_placeholder.image(frame, channels="RGB", use_column_width=True)
            time.sleep(0.03)  # Simulates roughly 30 FPS

with col_panel:
    st.write("**AI SENSITIVITY CONFIGURATIONS**")
    confidence = st.slider("Min. Confidence Cutoff", min_value=0, max_value=100, value=50, format="%d%%")
    st.caption("Detections registering below this cutoff will be ignored by database pipelines.")

    st.write("---")

    st.write("**HISTORIC LOG QUEUE**")
    # Mock data for the logs table
    log_data = {
        "TIME": ["02:56:28", "02:55:59", "02:51:56", "02:51:46"],
        "CAM": ["CAM 1", "CAM 1", "CAM 1", "CAM 1"],
        "CONFIDENCE": ["53%", "58%", "51%", "65%"]
    }
    df_logs = pd.DataFrame(log_data)
    st.dataframe(df_logs, use_container_width=True, hide_index=True)

st.markdown("---")

# --- BOTTOM CHART ---
st.write("**24-HOUR DETECTION FREQUENCY GRAPH**")
# Generate dummy data for the smooth area chart
chart_data = pd.DataFrame(
    np.random.randn(24, 1).cumsum() + 20,
    columns=['Detections']
)
st.area_chart(chart_data, color="#81c784")