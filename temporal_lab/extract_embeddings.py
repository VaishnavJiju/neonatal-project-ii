"""
Temporal Intelligence Lab — Phase 3: Embedding Extraction (V3)
==============================================================
Passes sequences through trained LSTM/TCN models and extracts the
final hidden state as a fixed-length embedding vector.

Output:
  mal_ed_data/multi_targets/v3/
    {target}_{model_type}_embeddings.parquet
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os, json, pickle
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# MODEL DEFINITIONS (must match training)
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc2(self.relu(self.fc1(out[:, -1, :])))
    def get_embedding(self, x):
        """Extract the last hidden state as embedding (pre-FC)."""
        out, _ = self.lstm(x)
        return out[:, -1, :]  # (batch, hidden_size=64)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels=[64, 64], kernel_size=3):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            d = 2**i
            ic = input_size if i == 0 else num_channels[i-1]
            p = (kernel_size - 1) * d
            layers.append(TCNBlock(ic, num_channels[i], kernel_size, 1, d, p))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.fc(self.tcn(x)[:, :, -1])
    def get_embedding(self, x):
        """Extract the last TCN output as embedding (pre-FC)."""
        x = x.transpose(1, 2)
        out = self.tcn(x)
        return out[:, :, -1]  # (batch, num_channels[-1]=64)

# ============================================================
# CONFIG
# ============================================================
SEQ_DATA_DIR = 'mal_ed_data/multi_targets/v2'
MODEL_DIR = 'models/v2'
OUTPUT_DIR = 'mal_ed_data/multi_targets/v3'
TARGETS = ['delta_baz', 'illness', 'recovery']
MODEL_TYPES = ['lstm', 'tcn']

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(SEQ_DATA_DIR, 'metadata.json'), 'r') as f:
    META = json.load(f)
SEQ_LEN = META['window_size']
N_FEATURES = META['features_per_timestep']

# ============================================================
# EXTRACTION
# ============================================================
def extract_embeddings(target, model_type):
    print(f"  Extracting {model_type.upper()} embeddings for '{target}'...")
    
    # Load sequence data
    df = pd.read_parquet(os.path.join(SEQ_DATA_DIR, f'{target}_sequences.parquet'))
    feat_cols = [f'f_{i}' for i in range(SEQ_LEN * N_FEATURES)]
    X = df[feat_cols].values
    pids = df['pid'].values
    agedays = df['agedays_target'].values
    y = df['y'].values
    
    # Load scaler and scale
    with open(os.path.join(MODEL_DIR, f'scaler_{target}.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    X_3d = X.reshape(-1, SEQ_LEN, N_FEATURES)
    X_scaled = scaler.transform(X_3d.reshape(-1, N_FEATURES)).reshape(-1, SEQ_LEN, N_FEATURES)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Load model
    if model_type == 'lstm':
        model = LSTMModel(N_FEATURES)
    else:
        model = TCNModel(N_FEATURES)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{model_type}_{target}.pt'), map_location='cpu'))
    model.eval()
    
    # Extract embeddings in batches
    embeddings = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i:i+batch_size]
            emb = model.get_embedding(batch).numpy()
            embeddings.append(emb)
    
    embeddings = np.vstack(embeddings)
    emb_dim = embeddings.shape[1]
    print(f"    Embedding shape: ({len(embeddings)}, {emb_dim})")
    
    # Build output dataframe
    emb_cols = {f'embedding_{j}': embeddings[:, j] for j in range(emb_dim)}
    out_df = pd.DataFrame({
        'pid': pids,
        'agedays': agedays,
        **emb_cols,
        'target': y,
    })
    
    out_path = os.path.join(OUTPUT_DIR, f'{target}_{model_type}_embeddings.parquet')
    out_df.to_parquet(out_path, index=False)
    print(f"    Saved: {out_path} ({len(out_df):,} rows)")
    return len(out_df), emb_dim

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("TEMPORAL INTELLIGENCE LAB - Phase 3: Embedding Extraction")
    print("=" * 70)
    
    stats = {}
    for target in TARGETS:
        print(f"\nTarget: {target}")
        for mt in MODEL_TYPES:
            n_rows, emb_dim = extract_embeddings(target, mt)
            stats[f'{target}_{mt}'] = {'rows': n_rows, 'embedding_dim': emb_dim}
    
    # Save metadata
    meta = {
        'embedding_dim': 64,
        'source_models': MODEL_TYPES,
        'targets': TARGETS,
        'datasets': stats,
    }
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Phase 3 COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
