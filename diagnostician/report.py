from datetime import datetime
import json, pandas as pd, pathlib

def df_top_processes(snapshot):
    rows = snapshot.get("processes_top",[])
    return pd.DataFrame(rows)

def df_disks(snapshot):
    rows = snapshot.get("hw",{}).get("disks",[])
    return pd.DataFrame(rows)

def df_gpus(snapshot):
    rows = snapshot.get("hw",{}).get("gpu",[])
    return pd.DataFrame(rows)

def save_all(outputs_dir, snapshot, rules, ollama_out):
    p = pathlib.Path(outputs_dir); p.mkdir(parents=True, exist_ok=True)
    (p/"snapshot.json").write_text(json.dumps(snapshot, indent=2))
    (p/"rules.json").write_text(json.dumps(rules, indent=2))
    (p/"ollama_analysis.json").write_text(json.dumps(ollama_out, indent=2))

    md = []
    md.append(f"# PC Diagnostician Report\nGenerated: {datetime.now().isoformat()}\n")
    md.append("## Summary\n")
    if ollama_out.get("component_ranked"):
        md.append(f"- **Likely components**: {ollama_out['component_ranked']}\n")
    if ollama_out.get("rationale"):
        md.append("### Rationale\n")
        md.append(f"{ollama_out['rationale']}\n")
    if ollama_out.get("fixes"):
        md.append("### Fixes\n")
        for fx in ollama_out["fixes"]:
            md.append(f"- {fx}")
        md.append("")
    if ollama_out.get("prevention"):
        md.append("### Prevention\n")
        for pr in ollama_out["prevention"]:
            md.append(f"- {pr}")
        md.append("")

    (p/"report.md").write_text("\n".join(md))

    # Also CSVs
    df1 = df_top_processes(snapshot); df1.to_csv(p/"processes_top.csv", index=False)
    df2 = df_disks(snapshot); df2.to_csv(p/"disks.csv", index=False)
    df3 = df_gpus(snapshot); df3.to_csv(p/"gpus.csv", index=False)

    return {
        "processes_df": df1,
        "disks_df": df2,
        "gpus_df": df3
    }
