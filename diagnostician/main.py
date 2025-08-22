import argparse, json, pandas as pd, os
from .collect import collect_snapshot
from .rules import apply_rules
from . import tests as quicktests
from .analyze_with_ollama import ask_ollama
from .report import save_all
from diagnostician.stress_test import run_full_stress

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", action="store_true", help="Run full system stress test")
    ap.add_argument("--model", default="qwen2.5:14b-instruct", help="Ollama model name")
    ap.add_argument("--duration", type=int, default=3, help="Perf sampling seconds")
    ap.add_argument("--no-stress", action="store_true", help="Disable short stress probes")
    ap.add_argument("--outputs", default="outputs", help="Output directory")
    args = ap.parse_args()

    if args.stress:
        run_full_stress(duration=args.duration)
        return

    print("Collecting snapshot...")
    snap = collect_snapshot(duration_s=args.duration)

    print("Running quick tests...")
    tests = {}
    if not args.no_stress:
        tests["cpu_burst"] = quicktests.cpu_burst(seconds=min(8, max(2, args.duration)))
        tests["mem_probe"] = quicktests.mem_probe(megabytes=256)
    tests["disk_probe"] = quicktests.disk_probe()
    snap["quick_tests"] = tests

    print("Applying rules...")
    hits = apply_rules(snap)

    print("Skipping Ollama analysis (Ollama server not running).")
    analysis = {"status": "skipped"}

    print("Saving report & DataFrames...")
    saved = save_all(args.outputs, snap, hits, analysis)

    # Print DataFrames for the user (preference)
    print("\n=== Top Processes ===")
    print(saved["processes_df"].head(20).to_string(index=False))
    print("\n=== Disks ===")
    print(saved["disks_df"].to_string(index=False))
    print("\n=== GPUs ===")
    print(saved["gpus_df"].to_string(index=False))

    print(f"\nDone. See '{args.outputs}' for JSON and report.md")

if __name__ == "__main__":
    main()

