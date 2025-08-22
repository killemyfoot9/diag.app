# PC Diagnostician (Python + Ollama)

This project inspects your **hardware + OS**, flags likely **faulty components** or **software issues**, and asks a local **Ollama** model to explain root cause and fixes.

## What it does
- Collects telemetry: CPU, GPU, RAM, disks (SMART-lite), drivers (Windows), OS, processes, temps
- Runs **quick tests**: CPU burst, memory allocation probe, disk free-space, GPU presence
- Applies **rules** to pinpoint likely trouble (thermal, storage, driver age, RAM pressure, etc.)
- Sends snapshot + rule hits to **Ollama** for a plain-English diagnosis + fixes
- Outputs **pandas DataFrames** and a `report.md`

## Requirements
- Python 3.9+
- (Recommended) Windows with admin rights for some WMI queries; Linux/macOS also supported
- Ollama installed and running: https://ollama.com
- A pulled model (choose one):
  - `ollama pull qwen2.5:14b-instruct` (recommended), or
  - `ollama pull llama3:8b`, `ollama pull deepseek-r1:7b`

### Python packages
```
pip install -r requirements.txt
```

## Run
```
python -m diagnostician.main --model qwen2.5:14b-instruct --duration 3 --no-stress
# or a fuller pass with a short stress probe:
python -m diagnostician.main --model qwen2.5:14b-instruct --duration 8
```

Outputs:
- `outputs/snapshot.json` — machine state
- `outputs/rules.json` — rule hits
- `outputs/ollama_analysis.json` — the model’s diagnosis
- `outputs/report.md` — human-readable summary
- DataFrames printed to console (and optionally saved as CSVs)

> **Note:** Quick tests are conservative. For serious stress testing (e.g., memtest86, Prime95, FurMark), run dedicated tools **at your own risk**.
