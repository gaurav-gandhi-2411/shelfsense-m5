"""
Consolidate results from run_day3.py and run_sarima_sarimax_reduced.py.
Reads per-method JSON files from data/processed/day3_results/ and
the main run's JSON if available, and writes the final report files.
"""
from __future__ import annotations

import sys
import os
import json

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_REPO = PROJ_ROOT
REPORTS_DIR = os.path.join(PROJ_ROOT, "reports")
RESULTS_DIR = os.path.join(MAIN_REPO, "data", "processed", "day3_results")

# Try loading main JSON first
main_json = os.path.join(REPORTS_DIR, "day3_classical_scores.json")
if os.path.exists(main_json):
    with open(main_json) as f:
        main_data = json.load(f)
    sn28_ref = main_data["seasonal_naive_28_reference"]
    methods_data = main_data["methods"]
    print(f"Loaded main JSON with methods: {list(methods_data.keys())}")
else:
    print("Main JSON not found, building from per-method files")
    methods_data = {}
    sn28_ref = None

# Load per-method result files (may override or supplement)
for method in ["ets", "arima", "sarima", "sarimax"]:
    meta_path = os.path.join(RESULTS_DIR, f"{method}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            methods_data[method] = json.load(f)
        print(f"  Loaded {method} from {meta_path}")

if sn28_ref is None:
    print("WARNING: No SN28 reference found, using placeholder")
    sn28_ref = {"wrmsse_sample": 0.6965, "wrmsse_by_category": {"FOODS": 0.6512, "HOUSEHOLD": 1.2335, "HOBBIES": 1.7794}}

sn28_wrmsse = sn28_ref["wrmsse_sample"]
sn28_cat = sn28_ref.get("wrmsse_by_category", {})

print(f"\nSN28 reference: {sn28_wrmsse:.4f}")
print(f"Methods available: {list(methods_data.keys())}")

for m, r in methods_data.items():
    s = r.get("wrmsse_sample", "N/A")
    fb = r.get("n_fallbacks", "N/A")
    t = r.get("wall_time_seconds", "N/A")
    note = r.get("note", "")
    if isinstance(s, float):
        print(f"  {m.upper()}: WRMSSE={s:.4f}, fallbacks={fb}, time={t:.0f}s {note}")
    else:
        print(f"  {m.upper()}: {s}")

# Save consolidated JSON
consolidated = {
    "seasonal_naive_28_reference": sn28_ref,
    "methods": methods_data,
}
out_path = os.path.join(REPORTS_DIR, "day3_classical_scores.json")
with open(out_path, "w") as f:
    json.dump(consolidated, f, indent=2)
print(f"\nSaved consolidated results to {out_path}")

# Run update_reports.py
import subprocess
result = subprocess.run(
    [sys.executable, os.path.join(PROJ_ROOT, "scripts", "update_reports.py")],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
