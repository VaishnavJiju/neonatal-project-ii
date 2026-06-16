"""
Temporal Intelligence Lab — Phase 4: Build Hybrid Datasets (V4)
===============================================================
Joins V3 embeddings with V1 clinical features on (pid, agedays).
Creates a combined feature set: temporal embeddings + tabular clinical features.

Output:
  mal_ed_data/multi_targets/v4/
    {target}_hybrid_{model_type}.parquet
"""

import pandas as pd
import numpy as np
import os, json

# ============================================================
# CONFIG
# ============================================================
V1_DIR = 'mal_ed_data/multi_targets/v1'
V3_DIR = 'mal_ed_data/multi_targets/v3'
OUTPUT_DIR = 'mal_ed_data/multi_targets/v4'
TARGETS = ['delta_baz', 'illness', 'recovery']
MODEL_TYPES = ['lstm', 'tcn']

# Mapping from our target names to V1 dataset files
V1_DATASET_MAP = {
    'delta_baz': 'dataset_baz_delta.parquet',
    'illness':   'dataset_illness_burden.parquet',
    'recovery':  'dataset_illness_burden.parquet',  # Recovery uses illness dataset as base
}

# V1 feature columns to use (exclude target/pid/agedays)
V1_EXCLUDE = ['pid', 'agedays', 'target_delta', 'target_velocity', 'burden_target']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# BUILD HYBRID
# ============================================================
def build_hybrid(target, model_type):
    print(f"  Building hybrid for '{target}' + {model_type.upper()}...")
    
    # Load V3 embeddings
    emb_path = os.path.join(V3_DIR, f'{target}_{model_type}_embeddings.parquet')
    emb_df = pd.read_parquet(emb_path)
    emb_cols = [c for c in emb_df.columns if c.startswith('embedding_')]
    
    # Load V1 tabular features
    v1_file = V1_DATASET_MAP[target]
    v1_df = pd.read_parquet(os.path.join(V1_DIR, v1_file))
    
    # Identify V1 feature columns (exclude identifiers and targets)
    v1_feature_cols = [c for c in v1_df.columns if c not in V1_EXCLUDE]
    
    # Encode categoricals in V1 if needed
    for col in v1_feature_cols:
        if v1_df[col].dtype.name == 'category' or v1_df[col].dtype == object:
            v1_df[col] = (v1_df[col] == 'Yes').astype(float)
    
    # Join on pid + agedays (nearest match since V1 is daily, V3 is at BAZ visits)
    # Strategy: for each embedding row, find the closest V1 row by agedays within the same pid
    hybrid_rows = []
    v1_grouped = v1_df.groupby('pid')
    
    matched = 0
    unmatched = 0
    
    for _, emb_row in emb_df.iterrows():
        pid = emb_row['pid']
        age = emb_row['agedays']
        
        if pid not in v1_grouped.groups:
            unmatched += 1
            continue
        
        child_v1 = v1_grouped.get_group(pid)
        # Find closest agedays
        diffs = (child_v1['agedays'] - age).abs()
        closest_idx = diffs.idxmin()
        closest_row = child_v1.loc[closest_idx]
        
        row_data = {
            'pid': pid,
            'agedays': age,
            'target': emb_row['target'],
        }
        # Add embeddings
        for ec in emb_cols:
            row_data[ec] = emb_row[ec]
        # Add V1 features
        for vc in v1_feature_cols:
            row_data[f'v1_{vc}'] = closest_row[vc]
        
        hybrid_rows.append(row_data)
        matched += 1
    
    hybrid_df = pd.DataFrame(hybrid_rows)
    
    # Drop any rows with NaN
    before = len(hybrid_df)
    hybrid_df = hybrid_df.dropna()
    after = len(hybrid_df)
    
    out_path = os.path.join(OUTPUT_DIR, f'{target}_hybrid_{model_type}.parquet')
    hybrid_df.to_parquet(out_path, index=False)
    
    n_emb = len(emb_cols)
    n_v1 = len(v1_feature_cols)
    print(f"    Matched: {matched:,} | Unmatched: {unmatched} | Dropped NaN: {before - after}")
    print(f"    Features: {n_emb} embeddings + {n_v1} V1 clinical = {n_emb + n_v1} total")
    print(f"    Saved: {out_path} ({len(hybrid_df):,} rows)")
    return len(hybrid_df)

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("TEMPORAL INTELLIGENCE LAB - Phase 4: Hybrid Dataset Builder")
    print("=" * 70)
    
    stats = {}
    for target in TARGETS:
        print(f"\nTarget: {target}")
        for mt in MODEL_TYPES:
            n = build_hybrid(target, mt)
            stats[f'{target}_hybrid_{mt}'] = f'{n} rows'
    
    meta = {
        'targets': TARGETS,
        'model_types': MODEL_TYPES,
        'datasets': stats,
        'join_strategy': 'pid + nearest agedays',
    }
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Phase 4 COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
