import platform, subprocess, json, datetime, shutil, time
import psutil
from cpuinfo import get_cpu_info

def _run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return ""

def _to_float(x):
    try: return float(x)
    except: return None

def gpu_info():
    # Prefer NVIDIA via nvidia-smi
    gpus = []
    if shutil.which("nvidia-smi"):
        out = _run('nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu --format=csv,noheader,nounits')
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                name, drv, vram_mb, temp = parts[0], parts[1], parts[2], parts[3]
                gpus.append({
                    "name": name,
                    "driver": drv,
                    "vram_gb": _to_float(vram_mb)/1024 if vram_mb else None,
                    "temp_c": _to_float(temp)
                })
    else:
        # Best-effort fallback
        if platform.system() == "Windows":
            name = _run("wmic path win32_VideoController get name /value").replace("Name=","").splitlines()[0] if "Name=" in _run("wmic path win32_VideoController get name /value") else ""
            gpus.append({"name": name, "driver": "", "vram_gb": None, "temp_c": None})
        else:
            gpus.append({"name": "Unknown", "driver": "", "vram_gb": None, "temp_c": None})
    return gpus

def temps_psutil():
    out = {"cpu_c": None, "gpu_c": None}
    if hasattr(psutil, "sensors_temperatures"):
        s = psutil.sensors_temperatures(fahrenheit=False) or {}
        # CPU: get highest core temp
        for name, entries in s.items():
            for e in entries:
                label = (e.label or name or "").lower()
                t = e.current
                if t is None: 
                    continue
                if "cpu" in label or "core" in label:
                    out["cpu_c"] = max(out["cpu_c"] or t, t)
                if "gpu" in label or "nv" in label:
                    out["gpu_c"] = max(out["gpu_c"] or t, t)
    return out

def disks():
    arr = []
    for p in psutil.disk_partitions(all=False):
        if p.fstype and p.mountpoint:
            try:
                usage = psutil.disk_usage(p.mountpoint)
                arr.append({
                    "device": p.device,
                    "mount": p.mountpoint,
                    "fs": p.fstype,
                    "size_gb": round(usage.total/1e9,2),
                    "free_gb": round(usage.free/1e9,2),
                })
            except Exception:
                pass
    return arr

def processes_top(n=10, sample_s=1.5):
    psutil.cpu_percent(interval=None)
    time.sleep(sample_s)
    procs = []
    for proc in psutil.process_iter(["name","pid","cpu_percent","memory_info"]):
        try:
            procs.append({
                "name": proc.info["name"] or "proc",
                "pid": proc.info["pid"],
                "cpu_pct": proc.info["cpu_percent"] or 0.0,
                "mem_mb": round((proc.info["memory_info"].rss or 0)/1e6,1)
            })
        except Exception:
            pass
    procs.sort(key=lambda x:(x["cpu_pct"], x["mem_mb"]), reverse=True)
    return procs[:n]

def os_info():
    uname = platform.uname()
    boot = datetime.datetime.fromtimestamp(psutil.boot_time())
    uptime_h = (datetime.datetime.now() - boot).total_seconds()/3600
    return {
        "name": uname.system,
        "version": uname.version,
        "build": uname.release,
        "uptime_hours": round(uptime_h,1),
        "last_boot": boot.isoformat()
    }

def perf_snapshot(duration_s=3):
    psutil.cpu_percent(interval=None)
    time.sleep(duration_s)
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "cpu_load_pct": psutil.cpu_percent(interval=None),
        "mem_used_gb": round(vm.used/1e9,2),
        "ram_gb": round(vm.total/1e9,2),
        "swap_used_gb": round(swap.used/1e9,2),
    }

def drivers_windows():
    if platform.system() != "Windows":
        return []
    out = _run("wmic path Win32_PnPSignedDriver get DeviceName,DriverVersion /format:csv")
    drivers = []
    for line in out.splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts)>=3 and parts[1] and parts[2]:
            drivers.append({"device": parts[1], "version": parts[2]})
    return drivers

def collect_snapshot(duration_s=3):
    cpu = get_cpu_info()
    temps = temps_psutil()
    gpus = gpu_info()
    snapshot = {
        "hw": {
            "cpu": {
                "name": cpu.get("brand_raw",""),
                "cores_physical": psutil.cpu_count(logical=False) or psutil.cpu_count(),
                "cores_logical": psutil.cpu_count(logical=True),
                "max_freq_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else None
            },
            "gpu": gpus,
            "ram_gb": round(psutil.virtual_memory().total/1e9,2),
            "disks": disks(),
            "temps": temps
        },
        "os": os_info(),
        "perf": perf_snapshot(duration_s=duration_s),
        "drivers": drivers_windows(),
        "processes_top": processes_top()
    }
    return snapshot
