import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from backend.main import apply_clinical_preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

df = pd.read_parquet('mal_ed_data/mal_ed_final.parquet')
df = apply_clinical_preprocessing(df)
target = 'BMI-for-age z-score'
horizon = 30
df_clean = df.copy()
df_clean['forecast_age'] = df_clean['agedays'] + horizon
df_future = df_clean[['pid', 'agedays', target]].copy().rename(columns={'agedays': 'forecast_age', target: 'target_future'})
df_clean = df_clean.sort_values('forecast_age')
df_future = df_future.sort_values('forecast_age')
df_clean = pd.merge_asof(df_clean, df_future, on='forecast_age', by='pid', direction='nearest', tolerance=7).sort_values(['pid', 'agedays'])
df_clean[target] = df_clean['target_future'] - df_clean[target]

core_features = [f'{target}_prev', f'{target}_velocity_30d', 'illness_days_30d', 'diarrhea_days_30d', 'days_since_illness', 'days_since_diarrhea', 'agedays']
df_clean = df_clean[core_features + [target, 'pid']].dropna()

# Downsample for speed
df_clean = df_clean.sample(100000, random_state=42)

X = df_clean[core_features]
y = df_clean[target]
groups = df_clean['pid']

gkf = GroupKFold(n_splits=3)
base_r2 = []
for tr, te in gkf.split(X, y, groups):
    m = RandomForestRegressor(n_estimators=30, max_depth=6, n_jobs=-1, random_state=42)
    m.fit(X.iloc[tr], y.iloc[tr])
    base_r2.append(r2_score(y.iloc[te], m.predict(X.iloc[te])))
base = np.mean(base_r2)
print(f'Baseline R2: {base:.4f}')

results = []
for f in core_features:
    r2_list = []
    X_abl = X.drop(columns=[f])
    for tr, te in gkf.split(X_abl, y, groups):
        m = RandomForestRegressor(n_estimators=30, max_depth=6, n_jobs=-1, random_state=42)
        m.fit(X_abl.iloc[tr], y.iloc[tr])
        r2_list.append(r2_score(y.iloc[te], m.predict(X_abl.iloc[te])))
    abl = np.mean(r2_list)
    impact = base - abl
    results.append({'Feature Dropped': f, 'Ablated R2': abl, 'Impact': impact})

res_df = pd.DataFrame(results).sort_values('Impact', ascending=False)
print('\n--- ABLATION TEST RESULTS ---')
print(res_df.to_string(index=False))
