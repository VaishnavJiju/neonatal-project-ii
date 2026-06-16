import pandas as pd, numpy as np, os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression

# Load daily dataset (BMI-for-age Z)
DATA_PATH = os.path.join('mal_ed_data', 'mal_ed_final.parquet')
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError('Dataset not found')

df = pd.read_parquet(DATA_PATH)
# Identify target column
TARGET = [c for c in df.columns if 'bmi' in c.lower() and 'z-score' in c.lower()][0]
# Exclude IDs and other z-scores
exclude_ids = ['pid', 'agedays', 'Household_Id', 'age_month']
z_score_cols = [c for c in df.columns if 'z-score' in c.lower()]
candidate_features = [c for c in df.columns if c not in exclude_ids + z_score_cols]
# Sample for speed
sample_n = min(200000, len(df))
df_sample = df.sample(n=sample_n, random_state=42)
# Encode categoricals
for col in df_sample.select_dtypes(include=['object', 'category']).columns:
    if col in candidate_features:
        df_sample[col] = LabelEncoder().fit_transform(df_sample[col].astype(str))
# MI ranking
mi_scores = mutual_info_regression(df_sample[candidate_features].fillna(0), df_sample[TARGET], random_state=42)
mi_rank = pd.Series(mi_scores, index=candidate_features).sort_values(ascending=False)
# Correlation ranking
corrs = df_sample[candidate_features + [TARGET]].corr()[TARGET].abs().sort_values(ascending=False)
# Union top 20
selected_features = list(set(mi_rank.head(20).index).union(set(corrs.head(20).index)))
print('Selected features count:', len(selected_features))
# Prepare data (group-aware split)
work_df = df[[*selected_features, TARGET, 'pid']].dropna(subset=[TARGET])
for col in work_df.select_dtypes(include=['object', 'category']).columns:
    work_df[col] = LabelEncoder().fit_transform(work_df[col].astype(str))
X = work_df[selected_features].astype(float)
y = work_df[TARGET].astype(float)
# Group split by pid
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=work_df['pid']))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
# Models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42, verbosity=0),
    'CatBoost': CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05, random_seed=42, verbose=0),
    'SVM': SVR(kernel='rbf', C=1.0),
    'Linear Regression': LinearRegression()
}
print('\nModel performance:')
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name:15s} | RMSE: {rmse:.4f} | R2: {r2:.4f} | MAE: {mae:.4f}")
