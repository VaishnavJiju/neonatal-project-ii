// UI component for selecting model version (v0 or v1)
import React from 'react';

export default function VersionSelector({ version, setVersion }) {
  const VERSIONS = ['v0', 'v1'];
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Model Version</label>
      <select className="glass-input" value={version} onChange={(e) => setVersion(e.target.value)} style={{ width: '100%' }}>
        {VERSIONS.map((v) => (
          <option key={v} value={v}>{v}</option>
        ))}
      </select>
    </div>
  );
}
