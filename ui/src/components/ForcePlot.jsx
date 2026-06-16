import React from 'react';

const ForcePlot = ({ detail, sampleIndex = 0 }) => {
    if (!detail || !detail.shap_values || !detail.feature_names) return null;

    const { shap_values, feature_names, feature_values, expected_value } = detail;
    const row = shap_values[sampleIndex];
    const vals = feature_values[sampleIndex];
    if (!row) return null;

    // Combine features with their SHAP values
    const contributions = feature_names.map((name, i) => ({
        name,
        shap: row[i],
        val: vals[i]
    })).filter(c => Math.abs(c.shap) > 0.001)
       .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

    const positive = contributions.filter(c => c.shap > 0);
    const negative = contributions.filter(c => c.shap < 0);
    
    const sumPos = positive.reduce((acc, c) => acc + c.shap, 0);
    const sumNeg = Math.abs(negative.reduce((acc, c) => acc + c.shap, 0));
    
    const totalRange = Math.max(0.1, sumPos + sumNeg + Math.abs(expected_value) * 0.2); // Base width
    const prediction = expected_value + sumPos - sumNeg;

    return (
        <div style={{ width: '100%', padding: '2rem' }}>
            <h4 style={{ color: '#fff', fontSize: '1rem', marginBottom: '2rem', fontWeight: 400 }}>
                Local Explanation (Sample #{sampleIndex})
            </h4>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', fontSize: '0.9rem' }}>
                <div style={{ color: 'var(--text-muted)' }}>Base Value: <span style={{ color: '#fff' }}>{expected_value.toFixed(3)}</span></div>
                <div style={{ color: 'var(--primary)', fontWeight: 600 }}>Prediction: {prediction.toFixed(3)}</div>
            </div>

            <div className="force-plot-container">
                <div className="force-plot-bar">
                    {/* Negative contributors (pushing prediction down) */}
                    {negative.map((c, i) => (
                        <div 
                            key={i} 
                            className="force-segment negative" 
                            style={{ width: `${(Math.abs(c.shap) / (sumPos + sumNeg + Math.abs(expected_value))) * 100}%` }}
                            title={`${c.name}: ${c.shap.toFixed(3)} (Value: ${c.val})`}
                        >
                            {Math.abs(c.shap) > 0.05 * (sumPos + sumNeg) && c.name}
                        </div>
                    ))}
                    
                    {/* Positive contributors (pushing prediction up) */}
                    {positive.map((c, i) => (
                        <div 
                            key={i} 
                            className="force-segment positive" 
                            style={{ width: `${(c.shap / (sumPos + sumNeg + Math.abs(expected_value))) * 100}%` }}
                            title={`${c.name}: +${c.shap.toFixed(3)} (Value: ${c.val})`}
                        >
                            {c.shap > 0.05 * (sumPos + sumNeg) && c.name}
                        </div>
                    ))}
                </div>
                
                <div className="force-labels">
                    <span>lower impact</span>
                    <div style={{ width: '2px', height: '10px', background: 'var(--primary)' }}></div>
                    <span>higher impact</span>
                </div>
            </div>

            <div style={{ marginTop: '2rem' }}>
                <h5 style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '1rem', textTransform: 'uppercase' }}>Top Contributors</h5>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div>
                        <p style={{ color: '#ff4444', fontSize: '0.75rem', marginBottom: '8px' }}>Pushed Higher ↑</p>
                        {positive.slice(0, 5).map((c, i) => (
                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px' }}>
                                <span>{c.name}</span>
                                <span style={{ color: '#ff4444' }}>+{c.shap.toFixed(3)}</span>
                            </div>
                        ))}
                    </div>
                    <div>
                        <p style={{ color: '#4444ff', fontSize: '0.75rem', marginBottom: '8px' }}>Pushed Lower ↓</p>
                        {negative.slice(0, 5).map((c, i) => (
                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px' }}>
                                <span>{c.name}</span>
                                <span style={{ color: '#4444ff' }}>{c.shap.toFixed(3)}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ForcePlot;
