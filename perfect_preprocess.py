import pandas as pd
import numpy as np
import os
import re
import gc

def clean_name(name):
    return re.sub(r'\s*\[.*?\]', '', str(name)).strip()

def perfect_preprocess():
    print("Starting Perfect Preprocessing (Phase 1.6)...")
    
    input_path = "mal_ed_data/mal_ed_master.parquet"
    output_path = "mal_ed_data/mal_ed_final.parquet"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print("1. Loading Master Dataset...")
    df = pd.read_parquet(input_path)
    print(f"   -> Raw Shape: {df.shape}")

    # Stage 1: Namespace Cleaning (Strip Brackets)
    print("2. Standardizing column names...")
    df.columns = [clean_name(c) for c in df.columns]

    # Stage 2: Column Selection & Smart Pruning
    print("3. Categorizing features...")
    id_cols = ['pid', 'agedays', 'Household_Id']
    keep_keywords = [
        'z-score', 'episode', 'pathogen', 'pos', 'neg', 'test', 'vaccine', 
        'breastfeed', 'intake', 'sex', 'weight', 'height', 'gestational', 
        'country', 'urban', 'rural', 'income', 'water', 'sanitation', 'education',
        'diarrhea', 'fever', 'cough'
    ]
    
    missing_pct = df.isnull().mean()
    cols_to_keep = []
    for col in df.columns:
        if col in id_cols:
            cols_to_keep.append(col)
            continue
        if any(key in col.lower() for key in keep_keywords) or missing_pct[col] < 0.90:
            # DROP Data Leakage variables: Variables that summarize the "future" 5 years of the child's life
            if 'total' in col.lower() or 'cumulative' in col.lower():
                continue
                
            if df[col].nunique() > 1:
                cols_to_keep.append(col)

    df = df[cols_to_keep]

    # Drop useless unique-row IDs
    if 'Participant_repeated_measure_Id' in df.columns:
        df = df.drop(columns=['Participant_repeated_measure_Id'])
    if 'Household_repeated_measure_Id' in df.columns:
        df = df.drop(columns=['Household_repeated_measure_Id'])
    if 'Household data collection date' in df.columns:
        df = df.drop(columns=['Household data collection date'])
    if 'Observation date' in df.columns:
        df = df.drop(columns=['Observation date'])

    # Drop date leakage columns (act as proxies for country/age, not clinical features)
    leakage_drops = [
        'Initial illness surveillance date',
        'Final illness surveillance date',
        'Birth date',
        'Currency exchange rate to USD',
    ]
    for col in leakage_drops:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"   -> Dropped leakage column: {col}")

    print(f"   -> Pruned Shape: {df.shape}")

    # Stage 3: Targeted Imputation
    print("4. Executing Clinical Imputation...")
    
    # Identify dtypes correctly
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # A. Symptoms & Pathogens (Smart Fill)
    symptom_keywords = ['episode', 'pos', 'neg', 'test', 'vaccine', 'diarrhea', 'fever', 'cough', 'blood', 'vomit']
    
    # Separate numeric vs categorical symptoms
    numeric_symptoms = [c for c in numeric_cols if any(k in c.lower() for k in symptom_keywords)]
    categorical_symptoms = [c for c in categorical_cols if any(k in c.lower() for k in symptom_keywords)]
    
    print(f"   -> Zero-filling {len(numeric_symptoms)} numeric clinical columns...")
    df[numeric_symptoms] = df[numeric_symptoms].fillna(0)
    
    print(f"   -> Filling {len(categorical_symptoms)} categorical status columns with 'Unknown'...")
    for col in categorical_symptoms:
        df[col] = df[col].astype(str).replace(['nan', 'None'], 'Unknown')

    # B. Growth Markers
    growth_cols = [c for c in numeric_cols if 'z-score' in c.lower()]
    # B1. Clamp z-scores to biologically valid range [-6, +6] to fix obvious measurement errors
    print(f"   -> Clamping z-scores to [-6, +6] (leaving unmeasured days as NaN)...")
    for col in growth_cols:
        df[col] = df[col].clip(lower=-6.0, upper=6.0)

    # C. Baseline & Remaining (Mode/Median Fill)
    print("   -> Final baseline imputation...")
    for col in categorical_cols:
        if col not in id_cols and col not in categorical_symptoms:
            df[col] = df[col].astype(str).replace(['nan', 'None'], 'Unknown')
            
    print("   -> Forward/Backward filling static baseline features per patient...")
    df = df.sort_values(['pid', 'agedays'])
    static_cols = [c for c in numeric_cols if c not in id_cols and c not in growth_cols and c not in numeric_symptoms]
    if static_cols:
        df[static_cols] = df.groupby("pid")[static_cols].ffill().bfill()

    # Stage 4: Optimization
    print("5. Optimization...")
    for col in df.columns:
        if col not in id_cols and df[col].dtype == object:
            df[col] = df[col].astype('category')
        
    df = df.dropna(subset=['pid', 'agedays'])

    print(f"6. Saving final dataset...")
    df.to_parquet(output_path, engine='pyarrow')
    print(f"DONE! Final Shape: {df.shape}")

if __name__ == "__main__":
    perfect_preprocess()
