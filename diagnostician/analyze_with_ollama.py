import requests, json

def ask_ollama(snapshot: dict, rules: list, model: str, url="http://localhost:11434/api/generate"):
    prompt = f"""
You are a senior PC diagnostics engineer. Using the snapshot and rule hits:
1) Identify the SINGLE most likely faulty component (CPU, GPU, RAM, Disk, Motherboard, OS, Drivers, Network, or Other). If multiple, rank top 3.
2) Explain evidence with concrete numbers from the snapshot.
3) Provide step-by-step fixes (permanent + temporary).
4) Add any preventive maintenance tips.

Return strict JSON with fields: component_ranked, rationale, fixes, prevention.
Snapshot:
{json.dumps(snapshot, indent=2)}

Rule_Hits:
{json.dumps(rules, indent=2)}
"""
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    text = r.json().get("response","").strip()
    # Try to parse JSON from model; if not JSON, wrap it.
    try:
        out = json.loads(text)
    except Exception:
        out = {"component_ranked": [], "rationale": text, "fixes": [], "prevention": []}
    return out
