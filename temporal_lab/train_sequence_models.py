"""
Temporal Intelligence Lab — Phase 2.5: Train Sequence Models (V2)
=================================================================
Trains LSTM and TCN models on all 4 enhanced sequence datasets.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = 'mal_ed_data/multi_targets/v2'
OUTPUT_DIR = 'models/v2'
BATCH_SIZE = 256
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, 'metadata.json'), 'r') as f:
    META = json.load(f)

SEQ_LEN = META['window_size']
N_FEATURES = META['features_per_timestep']

ALL_TARGETS = ['delta_baz', 'illness', 'recovery']

# ============================================================
# DATA
# ============================================================

def load_and_split_data(dataset_name):
    print(f"  Loading '{dataset_name}' dataset...")
    df = pd.read_parquet(os.path.join(DATA_DIR, f'{dataset_name}_sequences.parquet'))
    
    feat_cols = [f'f_{i}' for i in range(SEQ_LEN * N_FEATURES)]
    X = df[feat_cols].values
    y = df['y'].values
    groups = df['pid'].values
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    X_train_3d = X_train.reshape(-1, SEQ_LEN, N_FEATURES)
    X_val_3d = X_val.reshape(-1, SEQ_LEN, N_FEATURES)
    
    scaler = StandardScaler()
    X_train_flat = X_train_3d.reshape(-1, N_FEATURES)
    scaler.fit(X_train_flat)
    
    X_train_scaled = scaler.transform(X_train_flat).reshape(-1, SEQ_LEN, N_FEATURES)
    X_val_scaled = scaler.transform(X_val_3d.reshape(-1, N_FEATURES)).reshape(-1, SEQ_LEN, N_FEATURES)
    
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"    Train: {len(X_train)} | Val: {len(X_val)} | Features/step: {N_FEATURES}")
    return train_loader, val_loader, scaler

# ============================================================
# ARCHITECTURES
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
        last_out = out[:, -1, :]
        out = self.relu(self.fc1(last_out))
        return self.fc2(out)

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
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        last_out = out[:, :, -1]
        return self.fc(last_out)

# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_loader, val_loader, model_name, target_name):
    print(f"\n  --- Training {model_name.upper()} for {target_name} ---")
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                val_loss += criterion(model(X_b), y_b).item() * X_b.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{model_name}_{target_name}.pt'))
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            
        if epochs_no_improve >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}. Best Val: {best_val_loss:.4f}")
            break
    
    print(f"    Done. Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("TEMPORAL INTELLIGENCE LAB - Phase 2.5: Train All Sequence Models")
    print(f"Device: {DEVICE} | Targets: {ALL_TARGETS}")
    print("=" * 70)
    
    results = {}
    
    for target in ALL_TARGETS:
        print(f"\n{'='*70}")
        print(f"TARGET: {target}")
        print(f"{'='*70}")
        
        train_loader, val_loader, scaler = load_and_split_data(target)
        
        # Save scaler
        scaler_path = os.path.join(OUTPUT_DIR, f'scaler_{target}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # LSTM
        lstm = LSTMModel(input_size=N_FEATURES)
        lstm_loss = train_model(lstm, train_loader, val_loader, 'lstm', target)
        
        # TCN
        tcn = TCNModel(input_size=N_FEATURES)
        tcn_loss = train_model(tcn, train_loader, val_loader, 'tcn', target)
        
        results[target] = {'lstm_val_loss': lstm_loss, 'tcn_val_loss': tcn_loss}
    
    # Save training summary
    summary_path = os.path.join(OUTPUT_DIR, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Phase 2.5 Training COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    for t, r in results.items():
        print(f"  {t:15s} | LSTM Val: {r['lstm_val_loss']:.4f} | TCN Val: {r['tcn_val_loss']:.4f}")

if __name__ == '__main__':
    main()
