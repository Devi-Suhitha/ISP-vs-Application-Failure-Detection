import streamlit as st
import numpy as np
import pandas as pd
from src.utils import load_object

# Page config
st.set_page_config(page_title="OTT Root Cause Classification", layout="centered")

st.title("📡 OTT Performance Root Cause Classification")
st.write("Predict whether issue is ISP Side or Application Side")

# Load artifacts
model = load_object("artifacts/model.pkl")
preprocessor = load_object("artifacts/preprocessor.pkl")

st.subheader("🌐 Network Metrics")

ping = st.number_input("Ping Latency (ms)", value=50.0)
packet_loss = st.number_input("Packet Loss (%)", value=1.0)
jitter = st.number_input("Jitter (ms)", value=5.0)
download_speed = st.number_input("Download Speed (Mbps)", value=40.0)
dns_time = st.number_input("DNS Time (ms)", value=30.0)

st.subheader("🎥 Streaming Metrics")

buffer_time = st.number_input("Initial Buffer Time (sec)", value=1.0)
buffer_events = st.number_input("Buffering Events", value=1)
bitrate = st.number_input("Average Bitrate (kbps)", value=5000)
resolution = st.number_input("Resolution (720/1080/2160)", value=1080)
segment_time = st.number_input("Segment Download Time (ms)", value=150.0)

st.subheader("🖥 Server Metrics")

server_cpu = st.number_input("Server CPU (%)", value=60.0)
server_memory = st.number_input("Server Memory (%)", value=60.0)
error_rate = st.number_input("5xx Error Rate (%)", value=1.0)
db_time = st.number_input("DB Response Time (ms)", value=100.0)

st.subheader("📊 Context")

peak_hour = st.selectbox("Peak Hour?", [0, 1])
activity_type = st.selectbox("Activity Type", ["streaming", "browsing", "searching"])
isp_name = st.selectbox("ISP Name", ["Comcast", "AT&T", "Verizon", "Vodafone", "Airtel", "T-Mobile"])
region = st.selectbox("Region", ["North America", "Europe", "Asia-Pacific", "South America"])
device_type = st.selectbox("Device Type", ["Android", "iOS", "SmartTV", "Web"])


if st.button("Predict Root Cause"):

    input_data = pd.DataFrame([{
        "ping_latency_ms": ping,
        "packet_loss_percent": packet_loss,
        "jitter_ms": jitter,
        "download_speed_mbps": download_speed,
        "dns_time_ms": dns_time,
        "initial_buffer_time_sec": buffer_time,
        "buffering_events": buffer_events,
        "avg_bitrate_kbps": bitrate,
        "resolution": resolution,
        "segment_download_time_ms": segment_time,
        "server_cpu_percent": server_cpu,
        "server_memory_percent": server_memory,
        "error_rate_5xx_percent": error_rate,
        "db_response_time_ms": db_time,
        "peak_hour_flag": peak_hour,
        "activity_type": activity_type,
        "isp_name": isp_name,
        "region": region,
        "device_type": device_type
    }])

    transformed_input = preprocessor.transform(input_data)

    prediction = model.predict(transformed_input)

    if prediction[0] == 0:
        st.error("ISP Side Issue Detected")
    else:
        st.success("Application Side Issue Detected")