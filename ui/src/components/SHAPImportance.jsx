import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';

const SHAPImportance = ({ data, topN = 15 }) => {
    if (!data || Object.keys(data).length === 0) return null;

    // Convert object to sorted array
    const chartData = Object.entries(data)
        .map(([feature, score]) => ({ feature, score }))
        .sort((a, b) => b.score - a.score)
        .slice(0, topN);

    return (
        <div style={{ width: '100%', height: '100%', minHeight: '500px', display: 'flex', flexDirection: 'column' }}>
            <h4 style={{ color: '#fff', fontSize: '1rem', marginBottom: '1.5rem', fontWeight: 400 }}>Global Feature Impact (Mean |SHAP|)</h4>
            <div style={{ flex: 1 }}>
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                        <XAxis type="number" stroke="var(--text-muted)" fontSize={12} tick={{ fill: 'var(--text-muted)' }} />
                        <YAxis 
                            dataKey="feature" 
                            type="category" 
                            width={150} 
                            stroke="transparent" 
                            tick={{ fill: '#fff', fontSize: 11, fontWeight: 500 }}
                        />
                        <Tooltip 
                            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                            contentStyle={{ background: 'var(--nav-bg)', border: '1px solid var(--glass-border)', borderRadius: '8px' }}
                        />
                        <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                            {chartData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill="var(--primary)" opacity={Math.max(0.3, 1 - (index * 0.05))} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default SHAPImportance;
