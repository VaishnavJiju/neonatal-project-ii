"""
Temporal Intelligence Lab — Phase 2.5: Enhanced Sequence Dataset Builder (V2)
==============================================================================
Converts raw longitudinal MAL-ED data into leakage-safe, windowed
sequence datasets suitable for LSTM/TCN modeling.

DESIGN DECISIONS:
-----------------
1. Sequences are built from ANTHROPOMETRIC VISIT rows only (rows where
   BMI-for-age z-score is measured), not daily surveillance rows.
   Rationale: BAZ is measured ~monthly. Using daily rows would produce
   sequences of nearly identical values with no temporal learning signal.

2. Between each pair of anthropometric visits, we aggregate the DAILY
   surveillance data (illness, diarrhea, fever, antibiotics) to produce
   summary features: count of illness days, proportion of days sick, etc.

3. ENHANCED FEATURES (Phase 2.5):
   - Short-term memory: illness/diarrhea/antibiotics in last 3 visits
   - Trend signals: weight_delta, height_delta over last 2 visits
   - State flags: is_currently_ill, is_recovering
   - Age phase: infant(0) / toddler(1) / child(2)

4. Targets (4 total):
   - delta_baz: BAZ(t) - BAZ(t-1)  [growth velocity]
   - illness_burden: sum of illness days in the NEXT inter-visit window
   - diarrhea_burden: sum of diarrhea days in the NEXT inter-visit window
   - time_to_recovery: days until illness==0 AND diarrhea==0 (capped at 60)

5. Anti-leakage: NO engineered features (target_prev, burden_*, recovery_*)
   are included. Only raw clinical observations and safe backward-looking
   derived features.

OUTPUT:
-------
mal_ed_data/multi_targets/v2/
  delta_baz_sequences.parquet
  illness_sequences.parquet
  diarrhea_sequences.parquet
  recovery_sequences.parquet
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
WINDOW_SIZE = 5  # Number of past visits in each sequence
DATA_PATH = 'mal_ed_data/mal_ed_final.parquet'
OUTPUT_DIR = 'mal_ed_data/multi_targets/v2'
BAZ_COL = 'BMI-for-age z-score'
RECOVERY_CAP = 60  # Max days for time-to-recovery target

# Raw features to extract per timestep
CATEGORICAL_COLS = [
    'Any illness, caregiver report',
    'Diarrhea, caregiver report',
    'Fever, caregiver report',
    'Use of antibiotics, caregiver report',
]

NUMERIC_COLS = [
    'Weight (kg)',
    'Height (cm)',
    'WAMI index',
    'Maternal education (years)',
]

def load_and_prepare():
    """Load raw data, sort, and encode categoricals."""
    print("[1/5] Loading raw dataset...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(['pid', 'agedays']).reset_index(drop=True)
    print(f"       Loaded {len(df):,} rows, {df['pid'].nunique():,} children")
    
    # Binary-encode categorical columns: Yes=1, else=0
    for col in CATEGORICAL_COLS:
        binary_name = col.replace(', caregiver report', '').replace(' ', '_').lower()
        df[f'bin_{binary_name}'] = (df[col] == 'Yes').astype(float)
    
    return df

def compute_age_group(agedays):
    """Classify age phase: infant(0) / toddler(1) / child(2)."""
    if agedays < 365:
        return 0.0  # infant (< 1 year)
    elif agedays < 730:
        return 1.0  # toddler (1-2 years)
    else:
        return 2.0  # child (2+ years)

def build_anthropometric_visits(df):
    """
    Extract rows where BAZ is measured and aggregate inter-visit
    daily surveillance data into summary features.
    Includes enhanced temporal features for Phase 2.5.
    """
    print("[2/5] Building enhanced anthropometric visit table...")
    
    baz_mask = df[BAZ_COL].notna()
    baz_df = df[baz_mask].copy()
    print(f"       {len(baz_df):,} anthropometric visits across {baz_df['pid'].nunique():,} children")
    
    results = []
    
    for pid, child_df in df.groupby('pid'):
        child_df = child_df.sort_values('agedays')
        baz_rows = child_df[child_df[BAZ_COL].notna()]
        
        if len(baz_rows) < 2:
            continue
        
        baz_indices = baz_rows.index.tolist()
        baz_ages = baz_rows['agedays'].values
        baz_values = baz_rows[BAZ_COL].values
        weight_values = baz_rows['Weight (kg)'].values
        height_values = baz_rows['Height (cm)'].values
        
        for i in range(len(baz_indices)):
            row = baz_rows.loc[baz_indices[i]]
            age_now = baz_ages[i]
            
            # --- Backward-looking aggregation (since last BAZ visit) ---
            if i == 0:
                days_since_last = 0.0
                illness_days = 0.0
                diarrhea_days = 0.0
                fever_days = 0.0
                antibiotic_days = 0.0
                total_days_in_window = 1.0
            else:
                prev_age = baz_ages[i - 1]
                window = child_df[(child_df['agedays'] > prev_age) & (child_df['agedays'] <= age_now)]
                days_since_last = age_now - prev_age
                total_days_in_window = max(len(window), 1)
                illness_days = window['bin_any_illness'].sum()
                diarrhea_days = window['bin_diarrhea'].sum()
                fever_days = window['bin_fever'].sum()
                antibiotic_days = window['bin_use_of_antibiotics'].sum()
            
            # --- ENHANCED: Short-term memory (last 3 visits) ---
            illness_last_3 = 0.0
            diarrhea_last_3 = 0.0
            antibiotics_last_3 = 0.0
            lookback_start = max(0, i - 3)
            for j in range(lookback_start, i):
                prev_a = baz_ages[j]
                cur_a = baz_ages[j + 1] if j + 1 <= i else age_now
                mem_window = child_df[(child_df['agedays'] > prev_a) & (child_df['agedays'] <= cur_a)]
                illness_last_3 += mem_window['bin_any_illness'].sum()
                diarrhea_last_3 += mem_window['bin_diarrhea'].sum()
                antibiotics_last_3 += mem_window['bin_use_of_antibiotics'].sum()
            
            # --- ENHANCED: Trend signals (last 2 visits) ---
            if i >= 2:
                weight_delta_last_2 = float(weight_values[i] - weight_values[i - 2])
                height_delta_last_2 = float(height_values[i] - height_values[i - 2])
            elif i == 1:
                weight_delta_last_2 = float(weight_values[i] - weight_values[i - 1])
                height_delta_last_2 = float(height_values[i] - height_values[i - 1])
            else:
                weight_delta_last_2 = 0.0
                height_delta_last_2 = 0.0
            
            # --- ENHANCED: State flags ---
            is_currently_ill = 1.0 if illness_days > 0 else 0.0
            # is_recovering: was ill in the window before last, but not in this window
            if i >= 2:
                prev_prev_age = baz_ages[i - 2]
                prev_age_check = baz_ages[i - 1]
                prev_window = child_df[(child_df['agedays'] > prev_prev_age) & (child_df['agedays'] <= prev_age_check)]
                was_ill_before = prev_window['bin_any_illness'].sum() > 0
                is_recovering = 1.0 if (was_ill_before and illness_days == 0) else 0.0
            else:
                is_recovering = 0.0
            
            # --- ENHANCED: Age phase ---
            age_group = compute_age_group(age_now)
            
            # --- FORWARD TARGETS ---
            if i < len(baz_indices) - 1:
                next_age = baz_ages[i + 1]
                forward_window = child_df[(child_df['agedays'] > age_now) & (child_df['agedays'] <= next_age)]
                forward_illness_days = forward_window['bin_any_illness'].sum()
                forward_diarrhea_days = forward_window['bin_diarrhea'].sum()
                
                # Time-to-recovery: days until first day with no illness AND no diarrhea
                recovery_rows = forward_window[
                    (forward_window['bin_any_illness'] == 0) & (forward_window['bin_diarrhea'] == 0)
                ]
                if len(recovery_rows) > 0:
                    first_clear_age = recovery_rows['agedays'].iloc[0]
                    time_to_recovery = float(first_clear_age - age_now)
                else:
                    time_to_recovery = min(float(next_age - age_now), RECOVERY_CAP)
            else:
                forward_illness_days = np.nan
                forward_diarrhea_days = np.nan
                time_to_recovery = np.nan
            
            # Delta BAZ target
            if i > 0:
                delta_baz = baz_values[i] - baz_values[i - 1]
            else:
                delta_baz = np.nan
            
            results.append({
                'pid': pid,
                'agedays': age_now,
                'visit_idx': i,
                # --- Per-timestep features (ORIGINAL) ---
                'baz': baz_values[i],
                'weight_kg': row['Weight (kg)'],
                'height_cm': row['Height (cm)'],
                'wami_index': row['WAMI index'],
                'maternal_education': row['Maternal education (years)'],
                'time_delta': days_since_last,
                # Aggregated surveillance (backward-looking, safe)
                'illness_days': illness_days,
                'diarrhea_days': diarrhea_days,
                'fever_days': fever_days,
                'antibiotic_days': antibiotic_days,
                'illness_rate': illness_days / total_days_in_window,
                'diarrhea_rate': diarrhea_days / total_days_in_window,
                'fever_rate': fever_days / total_days_in_window,
                'antibiotic_rate': antibiotic_days / total_days_in_window,
                # --- ENHANCED FEATURES (Phase 2.5) ---
                'illness_last_3': illness_last_3,
                'diarrhea_last_3': diarrhea_last_3,
                'antibiotics_last_3': antibiotics_last_3,
                'weight_delta_last_2': weight_delta_last_2,
                'height_delta_last_2': height_delta_last_2,
                'is_currently_ill': is_currently_ill,
                'is_recovering': is_recovering,
                'age_group': age_group,
                # --- Targets ---
                'target_delta_baz': delta_baz,
                'target_illness_burden': forward_illness_days,
                'target_time_to_recovery': time_to_recovery,
            })
    
    visit_df = pd.DataFrame(results)
    print(f"       Built {len(visit_df):,} visit records for {visit_df['pid'].nunique():,} children")
    
    # Report enhanced feature stats
    print(f"       Enhanced features added: 8 new temporal signals")
    print(f"       New target: time_to_recovery")
    return visit_df

# Feature list for sequence construction (23 features per timestep)
FEATURE_COLS = [
    'agedays', 'time_delta',
    'baz', 'weight_kg', 'height_cm',
    'wami_index', 'maternal_education',
    'illness_days', 'diarrhea_days', 'fever_days', 'antibiotic_days',
    'illness_rate', 'diarrhea_rate', 'fever_rate', 'antibiotic_rate',
    # Enhanced (Phase 2.5)
    'illness_last_3', 'diarrhea_last_3', 'antibiotics_last_3',
    'weight_delta_last_2', 'height_delta_last_2',
    'is_currently_ill', 'is_recovering',
    'age_group',
]

def build_sequences(visit_df, target_col, dataset_name):
    """
    Slide a window of WINDOW_SIZE visits to create (X_seq, y) pairs.
    Each sequence consists of WINDOW_SIZE past visits; the target is from
    the visit immediately AFTER the window.
    """
    print(f"  Building '{dataset_name}' sequences (window={WINDOW_SIZE})...")
    
    sequences = []
    
    for pid, group in visit_df.groupby('pid'):
        group = group.sort_values('agedays')
        n = len(group)
        
        for t in range(WINDOW_SIZE, n):
            target_val = group.iloc[t][target_col]
            
            # Skip if target is NaN
            if pd.isna(target_val):
                continue
            
            # Extract the sequence window [t-WINDOW_SIZE ... t-1]
            window = group.iloc[t - WINDOW_SIZE : t]
            
            seq_data = window[FEATURE_COLS].values.flatten().tolist()
            
            sequences.append({
                'pid': pid,
                'agedays_target': group.iloc[t]['agedays'],
                'sequence_id': f"{pid}_{t}",
                'y': float(target_val),
                **{f'f_{i}': v for i, v in enumerate(seq_data)}
            })
    
    seq_df = pd.DataFrame(sequences)
    n_features = len(FEATURE_COLS)
    
    print(f"    Generated {len(seq_df):,} sequences")
    print(f"    Shape: ({len(seq_df)}, {WINDOW_SIZE} timesteps x {n_features} features = {WINDOW_SIZE * n_features} flat cols)")
    print(f"    Target '{target_col}' stats:")
    print(f"      Mean: {seq_df['y'].mean():.4f}")
    print(f"      Std:  {seq_df['y'].std():.4f}")
    print(f"      Min:  {seq_df['y'].min():.4f}")
    print(f"      Max:  {seq_df['y'].max():.4f}")
    
    # NaN check
    nan_count = seq_df.isna().sum().sum()
    if nan_count > 0:
        print(f"    WARNING: {nan_count} NaN values detected!")
    else:
        print(f"    PASSED: Zero NaN values in dataset")
    
    return seq_df, n_features

def save_dataset(seq_df, n_features, dataset_name):
    """Save dataset."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{dataset_name}_sequences.parquet')
    seq_df.to_parquet(out_path, index=False)
    print(f"    Saved: {out_path} ({len(seq_df):,} rows)")
    return out_path

def validate_no_leakage(seq_df, n_features_per_step, dataset_name):
    """Verify that no future information leaked into sequences."""
    print(f"  Validating anti-leakage for '{dataset_name}'...")
    
    # Check 1: Target agedays must be AFTER the last feature agedays
    issues = 0
    for _, row in seq_df.head(500).iterrows():
        target_age = row['agedays_target']
        last_timestep_start = (WINDOW_SIZE - 1) * n_features_per_step
        last_window_age = row[f'f_{last_timestep_start}']
        if last_window_age >= target_age:
            issues += 1
    
    if issues == 0:
        print(f"    PASSED: No temporal leakage detected (checked 500 samples)")
    else:
        print(f"    FAILED: {issues} samples have future data in sequence!")
    
    # Check 2: Verify no forbidden columns
    forbidden = ['target', 'target_prev', 'target_velocity', 'burden_', 'recovery_']
    for fc in FEATURE_COLS:
        for fb in forbidden:
            if fb in fc:
                print(f"    FAILED: Forbidden feature '{fc}' detected!")
                return
    print(f"    PASSED: No forbidden features in feature set")

def main():
    print("=" * 70)
    print("TEMPORAL INTELLIGENCE LAB - Phase 2.5: Enhanced Sequence Builder")
    print("=" * 70)
    
    df = load_and_prepare()
    visit_df = build_anthropometric_visits(df)
    
    # Define all targets
    TARGET_MAP = {
        'delta_baz':  ('target_delta_baz',       'Delta BAZ (Growth Velocity)'),
        'illness':    ('target_illness_burden',   'Illness Burden (Forward-Looking)'),
        'recovery':   ('target_time_to_recovery', 'Time-to-Recovery'),
    }
    
    dataset_stats = {}
    n_feat = len(FEATURE_COLS)
    
    for ds_name, (target_col, label) in TARGET_MAP.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {label}")
        print(f"{'='*70}")
        
        seq_df, n_f = build_sequences(visit_df, target_col, ds_name)
        save_dataset(seq_df, n_f, ds_name)
        validate_no_leakage(seq_df, n_f, ds_name)
        dataset_stats[ds_name] = f'{len(seq_df)} sequences'
    
    # Save metadata
    metadata = {
        'version': '2.5',
        'window_size': WINDOW_SIZE,
        'features_per_timestep': n_feat,
        'feature_names': FEATURE_COLS,
        'enhanced_features': [
            'illness_last_3', 'diarrhea_last_3', 'antibiotics_last_3',
            'weight_delta_last_2', 'height_delta_last_2',
            'is_currently_ill', 'is_recovering', 'age_group',
        ],
        'targets': {
            'delta_baz': 'BAZ(t) - BAZ(t-1)',
            'illness_burden': 'Sum of illness days in next inter-visit window',
            'time_to_recovery': f'Days until illness==0 AND diarrhea==0 (capped at {RECOVERY_CAP})',
        },
        'datasets': dataset_stats,
    }
    meta_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")
    
    print("\n" + "=" * 70)
    print("Phase 2.5 Dataset Build COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
