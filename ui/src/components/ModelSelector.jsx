// UI component for selecting a model
import React from 'react';

export default function ModelSelector({ model, setModel }) {
  const MODELS = ['Random Forest', 'XGBoost', 'CatBoost'];
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Algorithm</label>
      <select className="glass-input" value={model} onChange={(e) => setModel(e.target.value)} style={{ width: '100%' }}>
        {MODELS.map((m) => (
          <option key={m} value={m}>{m}</option>
        ))}
      </select>
    </div>
  );
}
