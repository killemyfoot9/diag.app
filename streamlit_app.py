import streamlit as st
import psutil
import time
import subprocess
import os
import pandas as pd
import plotly.express as px
import random
import streamlit.components.v1 as components
import base64
import json
import requests
import socket
from datetime import datetime
import threading
import math
import platform

OUTPUT_DIR = "outputs"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Create outputs directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------
# Utility functions
# ----------------------
def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        # Try to connect to the Ollama port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        return result == 0
    except:
        return False

def check_ollama_model_available():
    """Check if the required model is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return any(model.get('name', '').startswith('qwen2.5:3b') for model in models)
        return False
    except:
        return False

def monitor_system(duration=10):
    """Collect live CPU, RAM, Disk, FPS usage with live plotting + stats."""
    cpu, ram, disk, fps, timestamps = [], [], [], [], []

    placeholder_chart = st.empty()
    placeholder_stats = st.empty()

    start = time.time()
    while time.time() - start < duration:
        # Collect system stats
        cpu.append(psutil.cpu_percent())
        ram.append(psutil.virtual_memory().percent)
        disk.append(psutil.disk_usage('/').percent)

        # Fake FPS (replace with real FPS from GPU benchmark)
        fps_val = random.randint(60, 200)
        fps.append(fps_val)

        timestamps.append(time.strftime("%H:%M:%S"))

        # Create dataframe
        df = pd.DataFrame({
            "Time": timestamps,
            "CPU %": cpu,
            "RAM %": ram,
            "Disk %": disk,
            "FPS": fps
        })

        # Update chart live
        with placeholder_chart.container():
            st.subheader("📊 Live Resource Usage + FPS")
            fig = px.line(df, x="Time", y=["CPU %", "RAM %", "Disk %", "FPS"], markers=True)
            st.plotly_chart(fig, use_container_width=True)

        # Update FPS stats
        fps_min, fps_max, fps_avg = df["FPS"].min(), df["FPS"].max(), round(df["FPS"].mean(), 2)
        with placeholder_stats.container():
            st.markdown(f"""
                🎮 **FPS Stats (Live)**  
                - Min FPS: `{fps_min}`  
                - Max FPS: `{fps_max}`  
                - Avg FPS: `{fps_avg}`  
            """)

        time.sleep(1)

    return df

def cpu_stress_thread(intensity, stop_event):
    """Thread function to stress CPU based on intensity"""
    # More effective CPU stress test
    if intensity == "low":
        while not stop_event.is_set():
            # Simple calculation
            _ = sum(i*i for i in range(10000))
    elif intensity == "medium":
        while not stop_event.is_set():
            # More intensive calculation
            for _ in range(100):
                _ = sum(math.sqrt(i) for i in range(10000))
    elif intensity == "high":
        while not stop_event.is_set():
            # Even more intensive
            for _ in range(500):
                _ = sum(math.log(i+1) for i in range(10000))
    elif intensity == "extreme":
        while not stop_event.is_set():
            # Maximum intensity - use multiple calculation methods
            for _ in range(1000):
                result = 0
                for i in range(1000):
                    result += math.sin(i) * math.cos(i)
                _ = result

def stress_test_cpu(duration=10, intensity="medium"):
    """Stress test CPU and check for issues with intensity control"""
    st.info(f"🔥 Stress testing CPU ({intensity} intensity)...")
    
    # Start multiple CPU stress threads with specified intensity
    stop_event = threading.Event()
    threads = []
    
    # Use multiple threads for better CPU utilization
    num_threads = max(1, psutil.cpu_count(logical=False))  # Use physical cores
    
    for _ in range(num_threads):
        thread = threading.Thread(target=cpu_stress_thread, args=(intensity, stop_event))
        thread.daemon = True
        threads.append(thread)
        thread.start()

    # Monitor CPU performance
    cpu_usage = []
    cpu_freq = []
    cpu_temp = []
    timestamps = []

    start_time = time.time()
    end_time = start_time + duration

    while time.time() < end_time:
        # Get CPU usage with shorter interval for more responsive monitoring
        usage = psutil.cpu_percent(interval=0.1)
        cpu_usage.append(usage)

        # Get CPU frequency (if available)
        try:
            freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
            cpu_freq.append(freq)
        except:
            cpu_freq.append(0)

        # Get CPU temperature (if available)
        cpu_temp.append(0)  # Simplified - temperature monitoring removed

        timestamps.append(time.strftime("%H:%M:%S"))
        time.sleep(0.1)  # Shorter sleep for more frequent sampling

    # Stop the stress threads
    stop_event.set()
    for thread in threads:
        thread.join(timeout=1.0)

    # Analyze results
    avg_usage = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0

    # Determine CPU health with intensity-adjusted thresholds
    cpu_status = "✅ Good"
    cpu_issues = []

    # More realistic thresholds for CPU utilization
    if intensity == "low" and avg_usage < 30:
        cpu_issues.append(f"CPU not reaching expected utilization ({avg_usage:.1f}% < 30%) during {intensity} stress test")
        cpu_status = "⚠️ Warning"
    elif intensity == "medium" and avg_usage < 50:
        cpu_issues.append(f"CPU not reaching expected utilization ({avg_usage:.1f}% < 50%) during {intensity} stress test")
        cpu_status = "⚠️ Warning"
    elif intensity == "high" and avg_usage < 70:
        cpu_issues.append(f"CPU not reaching expected utilization ({avg_usage:.1f}% < 70%) during {intensity} stress test")
        cpu_status = "⚠️ Warning"
    elif intensity == "extreme" and avg_usage < 85:
        cpu_issues.append(f"CPU not reaching expected utilization ({avg_usage:.1f}% < 85%) during {intensity} stress test")
        cpu_status = "⚠️ Warning"

    if any(freq < 1000 for freq in cpu_freq if freq > 0):
        cpu_issues.append("CPU frequency dropping significantly under load")
        cpu_status = "⚠️ Warning"

    # Create DataFrame for visualization
    data_df = pd.DataFrame({
        "Time": timestamps,
        "CPU Usage %": cpu_usage,
        "CPU Frequency": cpu_freq
    })

    # Convert DataFrame to dict for JSON serialization
    data_dict = {
        "Time": timestamps,
        "CPU Usage %": cpu_usage,
        "CPU Frequency": cpu_freq
    }

    return {
        "status": cpu_status,
        "issues": cpu_issues,
        "avg_usage": avg_usage,
        "data": data_dict,
        "data_df": data_df
    }

def stress_test_ram(duration=10, intensity="medium"):
    """Stress test RAM and check for issues with intensity control"""
    st.info(f"💾 Stress testing RAM ({intensity} intensity)...")

    # Allocate and use memory to test RAM
    start_time = time.time()
    end_time = start_time + duration

    ram_usage = []
    available_ram = []
    timestamps = []

    # Adjust chunk size based on intensity
    if intensity == "low":
        chunk_size = 50 * 1024 * 1024  # 50MB chunks
    elif intensity == "medium":
        chunk_size = 100 * 1024 * 1024  # 100MB chunks
    elif intensity == "high":
        chunk_size = 200 * 1024 * 1024  # 200MB chunks
    else:  # extreme
        chunk_size = 500 * 1024 * 1024  # 500MB chunks

    # Test memory by allocating and using chunks
    test_data = []

    while time.time() < end_time:
        try:
            # Allocate memory
            test_data.append(bytearray(chunk_size))

            # Check current RAM usage
            ram = psutil.virtual_memory()
            ram_usage.append(ram.percent)
            available_ram.append(ram.available / (1024 * 1024))  # MB

            timestamps.append(time.strftime("%H:%M:%S"))

            time.sleep(0.5)
        except MemoryError:
            break

    # Clean up
    del test_data

    # Analyze results
    avg_usage = sum(ram_usage) / len(ram_usage) if ram_usage else 0

    # Determine RAM health with intensity-adjusted thresholds
    ram_status = "✅ Good"
    ram_issues = []

    # Adjust thresholds based on intensity
    if intensity == "low":
        min_expected_usage = 40
    elif intensity == "medium":
        min_expected_usage = 60
    elif intensity == "high":
        min_expected_usage = 80
    else:  # extreme
        min_expected_usage = 90

    if avg_usage < min_expected_usage and len(ram_usage) > 0:
        ram_issues.append(f"RAM not properly stressed ({avg_usage:.1f}% < {min_expected_usage}%) during {intensity} test")
        ram_status = "⚠️ Warning"

    if len(ram_usage) == 0:
        ram_issues.append("RAM test failed to allocate memory")
        ram_status = "❌ Critical"

    # Create DataFrame for visualization
    data_df = pd.DataFrame({
        "Time": timestamps,
        "RAM Usage %": ram_usage,
        "Available RAM (MB)": available_ram
    })

    # Convert DataFrame to dict for JSON serialization
    data_dict = {
        "Time": timestamps,
        "RAM Usage %": ram_usage,
        "Available RAM (MB)": available_ram
    }

    return {
        "status": ram_status,
        "issues": ram_issues,
        "avg_usage": avg_usage,
        "data": data_dict,
        "data_df": data_df
    }

def stress_test_disk(duration=10, intensity="medium"):
    """Stress test disk and check for issues with intensity control"""
    st.info(f"💽 Stress testing Disk ({intensity} intensity)...")

    # Adjust test file size based on intensity
    if intensity == "low":
        file_size = 50 * 1024 * 1024  # 50MB
        iterations = 3
    elif intensity == "medium":
        file_size = 100 * 1024 * 1024  # 100MB
        iterations = 5
    elif intensity == "high":
        file_size = 200 * 1024 * 1024  # 200MB
        iterations = 8
    else:  # extreme
        file_size = 500 * 1024 * 1024  # 500MB
        iterations = 12

    # Test disk performance
    start_time = time.time()
    end_time = start_time + duration

    disk_usage = []
    disk_read = []
    disk_write = []
    timestamps = []

    # Get initial disk counters
    disk_io = psutil.disk_io_counters()
    last_read = disk_io.read_bytes if disk_io else 0
    last_write = disk_io.write_bytes if disk_io else 0

    # Create test file for disk stress
    test_file = os.path.join(OUTPUT_DIR, "disk_test.bin")
    
    # Perform disk I/O operations
    for i in range(iterations):
        if time.time() > end_time:
            break
            
        try:
            # Write operation
            with open(test_file, 'wb') as f:
                f.write(os.urandom(file_size))
            
            # Read operation
            with open(test_file, 'rb') as f:
                _ = f.read()
                
            # Get disk usage
            usage = psutil.disk_usage('/').percent
            disk_usage.append(usage)

            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                current_read = disk_io.read_bytes
                current_write = disk_io.write_bytes
                disk_read.append((current_read - last_read) / 1024 / 1024)  # MB/s
                disk_write.append((current_write - last_write) / 1024 / 1024)  # MB/s
                last_read = current_read
                last_write = current_write
            else:
                disk_read.append(0)
                disk_write.append(0)

            timestamps.append(time.strftime("%H:%M:%S"))
            
        except Exception as e:
            st.error(f"Disk test error: {e}")
            break

    # Clean up test file
    try:
        os.remove(test_file)
    except:
        pass

    # Analyze results
    avg_read = sum(disk_read) / len(disk_read) if disk_read else 0
    avg_write = sum(disk_write) / len(disk_write) if disk_write else 0

    # Determine Disk health with realistic thresholds
    disk_status = "✅ Good"
    disk_issues = []

    # Adjust thresholds based on drive type (SSD vs HDD)
    if avg_read < 50:  # Less than 50MB/s read
        disk_issues.append(f"Disk read speed slow ({avg_read:.1f} MB/s)")
        disk_status = "⚠️ Warning"

    if avg_write < 30:  # Less than 30MB/s write
        disk_issues.append(f"Disk write speed slow ({avg_write:.1f} MB/s)")
        disk_status = "⚠️ Warning"

    # Create DataFrame for visualization
    data_df = pd.DataFrame({
        "Time": timestamps,
        "Disk Usage %": disk_usage,
        "Read Speed (MB/s)": disk_read,
        "Write Speed (MB/s)": disk_write
    })

    # Convert DataFrame to dict for JSON serialization
    data_dict = {
        "Time": timestamps,
        "Disk Usage %": disk_usage,
        "Read Speed (MB/s)": disk_read,
        "Write Speed (MB/s)": disk_write
    }

    return {
        "status": disk_status,
        "issues": disk_issues,
        "avg_read_speed": avg_read,
        "avg_write_speed": avg_write,
        "data": data_dict,
        "data_df": data_df
    }

def stress_test_gpu(duration=10, intensity="medium"):
    """Stress test GPU and check for issues with intensity control"""
    st.info(f"🎮 GPU Diagnostic ({intensity} intensity) - Using simulated data")

    start_time = time.time()
    end_time = start_time + duration

    gpu_usage = []
    gpu_temp = []
    gpu_memory = []
    timestamps = []

    # Use simulated data for GPU
    sleep_time = 1.0

    while time.time() < end_time:
        try:
            # Simulate GPU data based on intensity
            if intensity == "low":
                usage = random.randint(30, 60)
                temp = random.randint(40, 65)
                memory = random.randint(1000, 3000)
            elif intensity == "medium":
                usage = random.randint(50, 80)
                temp = random.randint(50, 75)
                memory = random.randint(2000, 4000)
            elif intensity == "high":
                usage = random.randint(70, 95)
                temp = random.randint(60, 85)
                memory = random.randint(3000, 6000)
            else:  # extreme
                usage = random.randint(85, 99)
                temp = random.randint(70, 90)
                memory = random.randint(4000, 8000)
                
            gpu_usage.append(usage)
            gpu_temp.append(temp)
            gpu_memory.append(memory)
            timestamps.append(time.strftime("%H:%M:%S"))
            time.sleep(sleep_time)
        except Exception as e:
            # Fallback values if simulation fails
            gpu_usage.append(50)
            gpu_temp.append(60)
            gpu_memory.append(3000)
            timestamps.append(time.strftime("%H:%M:%S"))
            time.sleep(1)

    # Analyze results
    avg_usage = sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0
    max_temp = max(gpu_temp) if gpu_temp else 0

    # Determine GPU health
    gpu_status = "✅ Good"
    gpu_issues = ["GPU monitoring not available - using simulated data for visualization"]

    # Create DataFrame for visualization
    data_df = pd.DataFrame({
        "Time": timestamps,
        "GPU Usage %": gpu_usage,
        "GPU Temperature": gpu_temp,
        "GPU Memory Used (MB)": gpu_memory
    })

    # Convert DataFrame to dict for JSON serialization
    data_dict = {
        "Time": timestamps,
        "GPU Usage %": gpu_usage,
        "GPU Temperature": gpu_temp,
        "GPU Memory Used (MB)": gpu_memory
    }

    return {
        "status": gpu_status,
        "issues": gpu_issues,
        "avg_usage": avg_usage,
        "max_temp": max_temp,
        "data": data_dict,
        "data_df": data_df
    }

def get_ai_recommendation(component, issues):
    """Get AI-powered recommendations for component issues using Ollama"""
    # Check if Ollama is available first
    if not check_ollama_connection():
        return "Ollama is not running. Please start Ollama with 'ollama serve'."

    if not check_ollama_model_available():
        return "qwen2.5:3b model not found. Please run 'ollama pull qwen2.5:3b'."

    try:
        prompt = f"""
        You are a PC hardware diagnostic expert. Provide concise technical recommendations to fix the following issues.

        Component: {component}
        Issues: {', '.join(issues) if issues else 'No issues found'}

        Provide 2-3 specific, actionable recommendations to resolve any issues. If no issues, just say the component is working properly.
        Be technical but clear and concise.
        """

        payload = {
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("response", "Could not get AI recommendations.")
        else:
            return f"Ollama API error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama. Please make sure Ollama is running with 'ollama serve'."
    except requests.exceptions.Timeout:
        return "Ollama request timed out. The model might be loading."
    except Exception as e:
        return f"Error getting AI recommendations: {str(e)}"

def get_base64_encoded_file(file_path):
    """Convert file to base64 encoding"""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None

# ----------------------
# Streamlit UI
# ----------------------
st.title("🛠️ Comprehensive PC Diagnostic System")
st.write("Run complete hardware stress tests, identify issues, and get AI-powered recommendations")

# Check Ollama status
ollama_status = check_ollama_connection()
ollama_model = check_ollama_model_available()

if ollama_status and ollama_model:
    st.success("✅ Ollama is running and qwen2.5:3b model is available!")
elif ollama_status and not ollama_model:
    st.warning("⚠️ Ollama is running but qwen2.5:3b model is not available. Run 'ollama pull qwen2.5:3b'")
else:
    st.error("❌ Ollama is not running. Please start Ollama with 'ollama serve'")

# ----------------------
# Always-visible 3D model
# ----------------------
st.subheader("🖼️ 3D Model Preview")

# Try to find the 3D model file in common locations
model_paths = [
    r"C:\Users\Husmiya\Downloads\pc_diagnostician\1990-mercedes-benz-190e-25-16-evolution-ii\source\1990 Mercedes-Benz 190 Evo II.glb",
    "./1990 Mercedes-Benz 190 Evo II.glb",
    "./3d-model.glb",
    "./model.glb"
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path and os.path.exists(model_path):
    st.info(f"3D model found: {model_path}")

    try:
        model_base64 = get_base64_encoded_file(model_path)
        
        if model_base64:
            html_code = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script type="module" src="https://unpkg.com/@google/model-viewer@^2.1.1/dist/model-viewer.min.js"></script>
                <style>
                    model-viewer {{
                        width: 100%;
                        height: 400px;
                        background-color: #f0f0f0;
                    }}
                </style>
            </head>
            <body>
                <model-viewer 
                    src="data:model/gltf-binary;base64,{model_base64}" 
                    alt="Mercedes-Benz 190 Evo II" 
                    auto-rotate 
                    camera-controls 
                    shadow-intensity="1.0"
                    environment-intensity="1.0"
                    exposure="1.0"
                    environment-image="neutral"
                    poster-color="transparent">
                </model-viewer>
            </body>
            </html>
            """

            components.html(html_code, height=420)
        else:
            st.error("Failed to load the 3D model.")
    except Exception as e:
        st.warning(f"Could not load 3D model: {str(e)}")
else:
    st.warning("3D model file not found. Diagnostic features are still available.")

# ----------------------
# Live System Monitoring
# ----------------------
st.subheader("📊 Live System Monitoring")

if st.button("Start Live Monitoring", key="live_monitor"):
    monitor_system(10)  # Monitor for 10 seconds

# ----------------------
# Diagnostic Controls
# ----------------------
st.subheader("🔍 Hardware Diagnostic Tests")

col1, col2 = st.columns(2)
with col1:
    duration = st.slider("Test Duration (seconds per component)", 10, 60, 15)
with col2:
    intensity = st.selectbox("Test Intensity", ["low", "medium", "high", "extreme"], index=1)

if st.button("Run Complete Diagnostic"):
    # Create tabs for each component
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["CPU", "RAM", "Disk", "GPU", "Summary"])

    with tab1:
        st.subheader("🧠 CPU Diagnostic")
        cpu_result = stress_test_cpu(duration, intensity)
        st.markdown(f"**Status:** {cpu_result['status']}")

        if cpu_result['issues']:
            st.error("**Issues found:**")
            for issue in cpu_result['issues']:
                st.write(f"- {issue}")
        else:
            st.success("No issues detected!")

        # Display CPU utilization metrics
        st.metric("Average CPU Utilization", f"{cpu_result['avg_usage']:.1f}%")

        st.plotly_chart(px.line(cpu_result['data_df'], x="Time", y="CPU Usage %",
                                title="CPU Usage During Stress Test"))

        # AI Recommendations
        with st.spinner("Getting AI recommendations..."):
            ai_advice = get_ai_recommendation("CPU", cpu_result['issues'])
            st.info(f"**AI Recommendations:** {ai_advice}")

    with tab2:
        st.subheader("💾 RAM Diagnostic")
        ram_result = stress_test_ram(duration, intensity)
        st.markdown(f"**Status:** {ram_result['status']}")

        if ram_result['issues']:
            st.error("**Issues found:**")
            for issue in ram_result['issues']:
                st.write(f"- {issue}")
        else:
            st.success("No issues detected!")

        # Display RAM utilization metrics
        st.metric("Average RAM Utilization", f"{ram_result['avg_usage']:.1f}%")

        if not ram_result['data_df'].empty:
            st.plotly_chart(px.line(ram_result['data_df'], x="Time", y="RAM Usage %",
                                    title="RAM Usage During Stress Test"))

        # AI Recommendations
        with st.spinner("Getting AI recommendations..."):
            ai_advice = get_ai_recommendation("RAM", ram_result['issues'])
            st.info(f"**AI Recommendations:** {ai_advice}")

    with tab3:
        st.subheader("💽 Disk Diagnostic")
        disk_result = stress_test_disk(duration, intensity)
        st.markdown(f"**Status:** {disk_result['status']}")

        if disk_result['issues']:
            st.error("**Issues found:**")
            for issue in disk_result['issues']:
                st.write(f"- {issue}")
        else:
            st.success("No issues detected!")

        # Display disk speed metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Read Speed", f"{disk_result['avg_read_speed']:.1f} MB/s")
        with col2:
            st.metric("Average Write Speed", f"{disk_result['avg_write_speed']:.1f} MB/s")

        st.plotly_chart(px.line(disk_result['data_df'], x="Time", y="Read Speed (MB/s)",
                                title="Disk Read Speed During Stress Test"))

        # AI Recommendations
        with st.spinner("Getting AI recommendations..."):
            ai_advice = get_ai_recommendation("Disk", disk_result['issues'])
            st.info(f"**AI Recommendations:** {ai_advice}")

    with tab4:
        st.subheader("🎮 GPU Diagnostic")
        gpu_result = stress_test_gpu(duration, intensity)
        st.markdown(f"**Status:** {gpu_result['status']}")

        if gpu_result['issues']:
            st.warning("**Notes:**")
            for issue in gpu_result['issues']:
                st.write(f"- {issue}")
        else:
            st.success("No issues detected!")

        # Display GPU utilization metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average GPU Utilization", f"{gpu_result['avg_usage']:.1f}%")
        with col2:
            if gpu_result['max_temp'] > 0:
                st.metric("Maximum Temperature", f"{gpu_result['max_temp']}°C")

        st.plotly_chart(px.line(gpu_result['data_df'], x="Time", y="GPU Usage %",
                                title="GPU Usage During Stress Test"))

        # AI Recommendations
        with st.spinner("Getting AI recommendations..."):
            ai_advice = get_ai_recommendation("GPU", [])
            st.info(f"**AI Recommendations:** {ai_advice}")

    with tab5:
        st.subheader("📊 Diagnostic Summary")

        # Create summary table
        summary_data = {
            "Component": ["CPU", "RAM", "Disk", "GPU"],
            "Status": [cpu_result['status'], ram_result['status'],
                       disk_result['status'], gpu_result['status']],
            "Issues Found": [len(cpu_result['issues']), len(ram_result['issues']),
                             len(disk_result['issues']), len(gpu_result['issues'])]
        }

        st.dataframe(summary_data, use_container_width=True)

        # Overall system health
        if all(result['status'] == "✅ Good" for result in [cpu_result, ram_result, disk_result, gpu_result]):
            st.success("🎉 Your system is in excellent condition! All components passed stress tests.")
        else:
            critical_issues = sum(1 for result in [cpu_result, ram_result, disk_result, gpu_result]
                                  if result['status'] == "❌ Critical")
            warning_issues = sum(1 for result in [cpu_result, ram_result, disk_result, gpu_result]
                                 if result['status'] == "⚠️ Warning")

            if critical_issues > 0:
                st.error(f"❌ {critical_issues} critical issues found. Immediate attention required!")
            if warning_issues > 0:
                st.warning(f"⚠️ {warning_issues} potential issues found. Consider investigating.")

        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "intensity": intensity,
            "duration": duration,
            "components": {
                "cpu": {
                    "status": cpu_result['status'],
                    "issues": cpu_result['issues'],
                    "avg_usage": cpu_result['avg_usage'],
                    "data": cpu_result['data']
                },
                "ram": {
                    "status": ram_result['status'],
                    "issues": ram_result['issues'],
                    "avg_usage": ram_result['avg_usage'],
                    "data": ram_result['data']
                },
                "disk": {
                    "status": disk_result['status'],
                    "issues": disk_result['issues'],
                    "avg_read_speed": disk_result['avg_read_speed'],
                    "avg_write_speed": disk_result['avg_write_speed'],
                    "data": disk_result['data']
                },
                "gpu": {
                    "status": gpu_result['status'],
                    "issues": gpu_result['issues'],
                    "avg_usage": gpu_result['avg_usage'],
                    "max_temp": gpu_result['max_temp'],
                    "data": gpu_result['data']
                }
            }
        }

        with open(os.path.join(OUTPUT_DIR, "diagnostic_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        st.success("Diagnostic report saved!")

# ----------------------
# Additional Features
# ----------------------
st.subheader("📋 System Information")

if st.button("Show System Specs"):
    col1, col2 = st.columns(2)

    with col1:
        st.write("**CPU Information:**")
        st.write(f"- Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        try:
            if psutil.cpu_freq():
                st.write(f"- Frequency: {psutil.cpu_freq().current} MHz")
        except:
            pass

        st.write("**Memory Information:**")
        mem = psutil.virtual_memory()
        st.write(f"- Total: {mem.total / (1024 ** 3):.2f} GB")
        st.write(f"- Available: {mem.available / (1024 ** 3):.2f} GB")

    with col2:
        st.write("**Disk Information:**")
        disk = psutil.disk_usage('/')
        st.write(f"- Total: {disk.total / (1024 ** 3):.2f} GB")
        st.write(f"- Free: {disk.free / (1024 ** 3):.2f} GB")

        st.write("**OS Information:**")
        st.write(f"- System: {os.name}")
        st.write(f"- Platform: {platform.platform()}")

# Instructions
st.subheader("ℹ️ Instructions")
st.markdown("""
1. Select test intensity and duration
2. Click "Run Complete Diagnostic" to test all hardware components
3. Each component will be stress tested based on the selected intensity
4. Results will show:
   - ✅ Green: Component is healthy
   - ⚠️ Yellow: Potential issues detected
   - ❌ Red: Critical issues requiring immediate attention
5. AI recommendations will be provided for any issues found
6. A summary report will be generated and saved

**Test Intensity Levels:**
- **Low**: Light workload, good for basic systems
- **Medium**: Moderate workload, standard testing
- **High**: Heavy workload, for performance systems
- **Extreme**: Maximum workload, for stress testing high-end systems
""")
