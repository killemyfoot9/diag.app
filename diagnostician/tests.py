import time, psutil

def cpu_burst(seconds=3):
    """Short CPU load probe (safe)."""
    end = time.time() + seconds
    x = 0
    while time.time() < end:
        x += sum(i*i for i in range(50000))
    return {"ok": True, "note":"CPU burst completed", "dummy": x}

def mem_probe(megabytes=256):
    """Try to allocate & free a memory block to detect immediate RAM issues (very light)."""
    try:
        block = bytearray(megabytes*1024*1024)
        for i in range(0, len(block), 4096):
            block[i] = 1
        del block
        return {"ok": True, "note": f"Allocated {megabytes}MB OK"}
    except MemoryError:
        return {"ok": False, "note": f"Failed to allocate {megabytes}MB"}

def disk_probe():
    parts = psutil.disk_partitions(all=False)
    issues = []
    for p in parts:
        try:
            usage = psutil.disk_usage(p.mountpoint)
            if usage.percent > 90:
                issues.append({"mount": p.mountpoint, "issue":"low_free_space", "used_pct": usage.percent})
        except Exception:
            pass
    return {"ok": len(issues)==0, "issues": issues}
