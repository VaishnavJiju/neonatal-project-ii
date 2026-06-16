import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, ComposedChart, Bar, Area, AreaChart } from 'recharts';

const RegressionGraphs = ({ predictions, graphType }) => {
    if (!predictions || predictions.length === 0) return null;

    // 1. Prediction vs Actual
    // We need a reference line y=x. Recharts doesn't have a simple way to draw a diagonal line easily, 
    // we'll calculate points for it.
    const allVals = predictions.flatMap(p => [p.y_true, p.y_pred]);
    const minVal = Math.min(...allVals);
    const maxVal = Math.max(...allVals);
    const refLineData = [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }];

    // 2. Residuals
    const residualData = predictions.map(p => ({
        actual: p.y_true,
        residual: p.y_pred - p.y_true
    }));

    // 3. Error Distribution (Histogram)
    // Simple binning
    const residuals = residualData.map(d => d.residual);
    const nBins = 20;
    const resMin = Math.min(...residuals);
    const resMax = Math.max(...residuals);
    const binSize = (resMax - resMin) / nBins;
    const bins = Array.from({ length: nBins }, (_, i) => ({
        bin: (resMin + i * binSize).toFixed(2),
        count: 0
    }));
    residuals.forEach(r => {
        const bIdx = Math.min(nBins - 1, Math.floor((r - resMin) / binSize));
        bins[bIdx].count++;
    });

    const renderGraph = () => {
        switch (graphType) {
            case 'Prediction vs Actual':
                return (
                    <div className="glass-card" style={{ padding: '1.5rem', height: '500px', display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-muted)' }}>Prediction vs Actual</h4>
                        <div style={{ flex: 1 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis type="number" dataKey="y_true" name="Actual" stroke="var(--text-muted)" fontSize={11} domain={['auto', 'auto']} label={{ value: 'Actual Values', position: 'bottom', fill: 'var(--text-muted)', fontSize: 11 }} />
                                    <YAxis type="number" dataKey="y_pred" name="Predicted" stroke="var(--text-muted)" fontSize={11} domain={['auto', 'auto']} label={{ value: 'Predicted Values', angle: -90, position: 'left', fill: 'var(--text-muted)', fontSize: 11 }} />
                                    <Tooltip contentStyle={{ background: 'var(--nav-bg)', border: '1px solid var(--glass-border)' }} />
                                    <Scatter name="Predictions" data={predictions} fill="var(--primary)" opacity={0.5} />
                                    <Line data={refLineData} dataKey="y" xDataKey="x" stroke="#ff4444" strokeDasharray="5 5" dot={false} activeDot={false} isAnimationActive={false} />
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                );
            case 'Residual Plot':
                return (
                    <div className="glass-card" style={{ padding: '1.5rem', height: '500px', display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-muted)' }}>Residual Plot</h4>
                        <div style={{ flex: 1 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis type="number" dataKey="actual" name="Actual" stroke="var(--text-muted)" fontSize={11} label={{ value: 'Actual Values', position: 'bottom', fill: 'var(--text-muted)', fontSize: 11 }} />
                                    <YAxis type="number" dataKey="residual" name="Residual" stroke="var(--text-muted)" fontSize={11} label={{ value: 'Residuals (Predicted - Actual)', angle: -90, position: 'left', fill: 'var(--text-muted)', fontSize: 11 }} />
                                    <Tooltip contentStyle={{ background: 'var(--nav-bg)', border: '1px solid var(--glass-border)' }} />
                                    <Scatter name="Residuals" data={residualData} fill="#ffaa00" opacity={0.5} />
                                    <Line data={[{actual: minVal, residual: 0}, {actual: maxVal, residual: 0}]} dataKey="residual" xDataKey="actual" stroke="rgba(255,255,255,0.3)" strokeDasharray="5 5" dot={false} />
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                );
            case 'Error Distribution':
                return (
                    <div className="glass-card" style={{ padding: '1.5rem', height: '500px', display: 'flex', flexDirection: 'column' }}>
                        <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-muted)' }}>Error Distribution</h4>
                        <div style={{ flex: 1 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={bins} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="bin" stroke="var(--text-muted)" fontSize={10} label={{ value: 'Residual Bin', position: 'bottom', fill: 'var(--text-muted)', fontSize: 11 }} />
                                    <YAxis stroke="var(--text-muted)" fontSize={10} label={{ value: 'Count', angle: -90, position: 'left', fill: 'var(--text-muted)', fontSize: 11 }} />
                                    <Tooltip contentStyle={{ background: 'var(--nav-bg)', border: '1px solid var(--glass-border)' }} />
                                    <Area type="monotone" dataKey="count" stroke="var(--primary)" fill="var(--primary)" fillOpacity={0.2} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                );
            case 'Time-Series Trajectory':
                return (
                    <div className="glass-card" style={{ padding: '1.5rem', height: '500px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                        <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-muted)', position: 'absolute', top: '1.5rem', left: '1.5rem' }}>Time-Series Trajectory</h4>
                        <div style={{ textAlign: 'center', opacity: 0.5 }}>
                            <p style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>Coming Soon</p>
                            <p style={{ fontSize: '0.8rem', maxWidth: '200px' }}>Requires registry update with temporal metadata (agedays, pid).</p>
                        </div>
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <div style={{ width: '100%', height: '100%' }}>
            {renderGraph()}
        </div>
    );
};

export default RegressionGraphs;
