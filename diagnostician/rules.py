def apply_rules(snapshot: dict):
    hits = []

    # Temps
    cpu_t = (snapshot.get("hw",{}).get("temps",{}) or {}).get("cpu_c")
    if cpu_t is not None and cpu_t > 90:
        hits.append({"component":"CPU","issue":"Overheating","evidence":f"CPU temp {cpu_t}°C > 90°C","severity":"high","fix":"Clean fans/heatsink, reapply thermal paste, improve airflow, reduce background load."})
    gpu_list = snapshot.get("hw",{}).get("gpu",[]) or []
    for g in gpu_list:
        if g.get("temp_c") and g["temp_c"] > 85:
            hits.append({"component":"GPU","issue":"Overheating","evidence":f"GPU temp {g['temp_c']}°C > 85°C","severity":"high","fix":"Increase fan curve, dust cleaning, check case airflow, update drivers."})

    # RAM pressure
    ram_gb = snapshot.get("perf",{}).get("ram_gb") or snapshot.get("hw",{}).get("ram_gb")
    mem_used = snapshot.get("perf",{}).get("mem_used_gb")
    swap_used = snapshot.get("perf",{}).get("swap_used_gb")
    if ram_gb and mem_used and (mem_used/ram_gb) > 0.9:
        hits.append({"component":"RAM","issue":"High memory pressure","evidence":f"mem_used {mem_used}GB / {ram_gb}GB > 90%","severity":"medium","fix":"Close heavy apps, add RAM, reduce browser tabs, optimize startup."})
    if swap_used and swap_used > 1:
        hits.append({"component":"RAM","issue":"Swap in use","evidence":f"swap_used {swap_used}GB","severity":"low","fix":"Add RAM or reduce memory usage; check background processes."})

    # Disk space
    for d in snapshot.get("hw",{}).get("disks",[]):
        size = d.get("size_gb"); free = d.get("free_gb")
        if size and free is not None and size>0 and (free/size) < 0.1:
            hits.append({"component":"Disk","issue":"Low free space","evidence":f"{d.get('device')} free {free}GB / {size}GB < 10%","severity":"medium","fix":"Clean temporary files, remove unused apps, move large files to another drive."})

    # Uptime
    up = snapshot.get("os",{}).get("uptime_hours")
    if up and up > 240:
        hits.append({"component":"OS","issue":"Long uptime","evidence":f"uptime {up}h > 240h","severity":"low","fix":"Reboot to clear leaks and apply updates."})

    # GPU driver "unknown" flag
    if not gpu_list or (len(gpu_list)==1 and not gpu_list[0].get("name")):
        hits.append({"component":"GPU","issue":"GPU not detected via tools","evidence":"nvidia-smi/WMI not reporting properly","severity":"low","fix":"Reinstall GPU drivers, verify hardware seating."})

    return hits
