"""
Precompute feature discovery results for the first 4 targets.
Uses MI + Permutation only (Correlation crashes on categorical targets).
"""
import requests, json, os

BASE = "http://localhost:8000"

# Get columns list  
overview = requests.get(f"{BASE}/api/dataset/overview").json()
all_cols = [c for c in overview['columns_list'] if c not in ['pid', 'Household_Id', 'agedays']]
targets = all_cols[:4]

print(f"Precomputing for targets: {targets}")

cache = {}
for t in targets:
    print(f"\n--- Running discovery for: {t} ---")
    try:
        res = requests.post(f"{BASE}/api/models/feature_discovery", json={
            "target": t,
            "methods": ["Mutual Information", "SHAP", "Permutation"],
            "sample_size": 15000
        }, timeout=180)
        if res.status_code != 200:
            print(f"  HTTP {res.status_code}: {res.text[:200]}")
            continue
        data = res.json()
        top10 = data['results'][:10]
        cache[t] = {
            "target": t,
            "task_type": data.get('task_type', 'regression'),
            "results": top10
        }
        print(f"  OK - {len(top10)} features. Top 3: {[r['feature'] for r in top10[:3]]}")
    except Exception as e:
        print(f"  ERROR: {e}")

out_path = os.path.join("backend", "discovery_cache.json")
with open(out_path, "w") as f:
    json.dump(cache, f, indent=2)

print(f"\nSaved cache to {out_path} ({len(cache)} targets)")
