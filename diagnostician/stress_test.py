# system_stress.py
import multiprocessing
import math
import time
import os
import sys
import threading
import subprocess
import psutil

# ========================
# System Monitor
# ========================
def monitor_system(duration, log_path="outputs/stress_monitor.csv", interval=1):
    """
    Logs CPU, RAM, GPU, and Disk usage while stress is running.
    """
    import GPUtil

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write("time,cpu_percent,ram_percent,gpu_load,gpu_temp,disk_read,disk_write\n")

        start = time.time()
        prev_read, prev_write = 0, 0

        while time.time() - start < duration:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_load = gpu.load * 100
                gpu_temp = gpu.temperature
            else:
                gpu_load, gpu_temp = 0, 0

            disk = psutil.disk_io_counters()
            disk_read = disk.read_bytes - prev_read
            disk_write = disk.write_bytes - prev_write
            prev_read, prev_write = disk.read_bytes, disk.write_bytes

            f.write(f"{time.time()-start:.1f},{cpu},{ram},{gpu_load},{gpu_temp},{disk_read},{disk_write}\n")
            f.flush()
            time.sleep(interval)

# ========================
# CPU Stress Test
# ========================
def cpu_worker(duration):
    end = time.time() + duration
    while time.time() < end:
        math.sqrt(1234567.89) ** 5

def stress_cpu(duration=30):
    print("[*] Starting CPU stress test...")
    procs = []
    for _ in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=cpu_worker, args=(duration,))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    print("[✓] CPU stress test completed.")

# ========================
# RAM Stress Test
# ========================
def stress_ram(duration=30, size_mb=200):
    print("[*] Starting RAM stress test...")
    data = []
    end = time.time() + duration
    try:
        while time.time() < end:
            data.append(bytearray(size_mb * 1024 * 1024))  # allocate memory
    except MemoryError:
        print("[!] MemoryError: Not enough RAM! Stopping test.")
    finally:
        print("[✓] RAM stress test completed.")
        del data

# ========================
# Disk Stress Test
# ========================
def stress_disk(duration=30, file_size_mb=100):
    print("[*] Starting Disk stress test...")
    filename = "stress_test.tmp"
    block = os.urandom(1024 * 1024)  # 1MB block
    end = time.time() + duration
    try:
        with open(filename, "wb") as f:
            while time.time() < end:
                f.write(block * file_size_mb)  # write file_size_mb MB
                f.flush()
        os.remove(filename)
    except Exception as e:
        print(f"[!] Disk stress test error: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)
        print("[✓] Disk stress test completed.")

# ========================
# GPU Stress Test (PyTorch)
# ========================
def stress_gpu(duration=30):
    print("[*] Starting GPU stress test...")
    try:
        import torch
        if torch.cuda.is_available():
            end = time.time() + duration
            while time.time() < end:
                a = torch.rand((5000, 5000), device="cuda")
                b = torch.mm(a, a)
                del b
            print("[✓] GPU stress test completed.")
        else:
            print("[x] No CUDA GPU available.")
    except Exception as e:
        print(f"[!] GPU stress skipped: {e}")

# ========================
# Full Stress Test
# ========================
def run_full_stress(duration=30):
    print(f"\n🔥 Running FULL SYSTEM STRESS TEST ({duration}s each) 🔥")

    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_system, args=(duration * 4,))
    monitor_thread.start()

    # Stress components one by one
    stress_cpu(duration)
    stress_ram(duration)
    stress_disk(duration)
    stress_gpu(duration)

    monitor_thread.join()
    print("\n✅ Full system stress test completed. Logs saved to outputs/stress_monitor.csv\n")


if __name__ == "__main__":
    run_full_stress(duration=20)

