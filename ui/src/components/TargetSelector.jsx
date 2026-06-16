// UI component for selecting a target
import React from 'react';

const TARGETS = [
  { value: 'classification_target', label: 'Diarrheal Incidence (Risk)' },
  { value: 'target', label: 'BAZ Trajectory (AR)' },
  { value: 'target_delta', label: 'BAZ Velocity (ΔBAZ)' },
  { value: 'burden_target', label: 'Illness Burden (Longitudinal)' }
];

export default function TargetSelector({ target, setTarget }) {
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Target Paradigm</label>
      <select className="glass-input" value={target} onChange={(e) => setTarget(e.target.value)} style={{ width: '100%' }}>
        {TARGETS.map((t) => (
          <option key={t.value} value={t.value}>{t.label}</option>
        ))}
      </select>
    </div>
  );
}
