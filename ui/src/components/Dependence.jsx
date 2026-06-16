import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Dependence = ({ detail, feature }) => {
    if (!detail || !feature) return null;

    const { shap_values, feature_names, feature_values } = detail;
    const fIdx = feature_names.indexOf(feature);
    if (fIdx === -1) return null;

    const data = shap_values.map((row, i) => ({
        val: feature_values[i][fIdx],
        shap: row[fIdx]
    }));

    return (
        <div style={{ width: '100%', height: '100%', minHeight: '500px', display: 'flex', flexDirection: 'column' }}>
            <h4 style={{ color: '#fff', fontSize: '1rem', marginBottom: '1.5rem', fontWeight: 400 }}>SHAP Dependence: {feature}</h4>
            <div style={{ flex: 1 }}>
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis 
                            type="number" 
                            dataKey="val" 
                            name={feature} 
                            stroke="var(--text-muted)" 
                            label={{ value: 'Feature Value', position: 'bottom', fill: 'var(--text-muted)', fontSize: 12 }} 
                        />
                        <YAxis 
                            type="number" 
                            dataKey="shap" 
                            name="SHAP" 
                            stroke="var(--text-muted)" 
                            label={{ value: 'SHAP Value', angle: -90, position: 'left', fill: 'var(--text-muted)', fontSize: 12 }} 
                        />
                        <Tooltip 
                            cursor={{ strokeDasharray: '3 3' }} 
                            contentStyle={{ background: 'var(--nav-bg)', border: '1px solid var(--glass-border)', borderRadius: '8px' }}
                        />
                        <Scatter name={feature} data={data} fill="var(--primary)" opacity={0.6} />
                    </ScatterChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default Dependence;
