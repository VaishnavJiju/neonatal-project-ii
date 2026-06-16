import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

print("Loading monthly dataset...")
df = pd.read_parquet('mal_ed_data/mal_ed_monthly_trajectories.parquet')

# Identify targets and features
exclude_ids = ['pid', 'agedays', 'Household_Id', 'age_month']
zscore_cols = [c for c in df.columns if 'z-score' in c.lower()]
candidate_features = [c for c in df.columns if c not in exclude_ids and c not in zscore_cols]

print(f"Dataset: {df.shape}")
print(f"Targets: {len(zscore_cols)} z-score columns")
print(f"Candidate Features: {len(candidate_features)}")

# Subsample for speed
df_sample = df.sample(n=min(30000, len(df)), random_state=42)

# Encode categoricals
for col in df_sample.select_dtypes(include=['object', 'category']).columns:
    if col in candidate_features:
        df_sample[col] = LabelEncoder().fit_transform(df_sample[col].astype(str))

print("\n" + "="*80)
print("MULTI-TARGET FEATURE DISCOVERY (Mutual Information)")
print("="*80)

all_results = {}

for target in zscore_cols:
    work = df_sample[candidate_features + [target]].dropna(subset=[target])
    X = work[candidate_features].astype(float).fillna(0)
    y = work[target].astype(float)
    
    mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    ranked = pd.Series(mi_scores, index=candidate_features).sort_values(ascending=False)
    
    # Store top 10
    all_results[target] = ranked.head(10)
    
    print(f"\n--- TARGET: {target} ---")
    for i, (feat, score) in enumerate(ranked.head(10).items(), 1):
        print(f"  {i:2d}. {feat:<55s} MI={score:.4f}")

# Cross-target consensus: which features appear in top 10 across ALL targets?
print("\n" + "="*80)
print("CROSS-TARGET CONSENSUS (Features appearing in top 10 for multiple targets)")
print("="*80)

feature_counts = {}
for target, scores in all_results.items():
    for feat in scores.index:
        feature_counts[feat] = feature_counts.get(feat, 0) + 1

consensus = sorted(feature_counts.items(), key=lambda x: -x[1])
print(f"\n{'Feature':<55s} {'Targets':>10s}")
print("-" * 67)
for feat, count in consensus[:20]:
    print(f"{feat:<55s} {count:>5d}/{len(zscore_cols)}")
