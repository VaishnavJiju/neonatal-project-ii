import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, ScatterChart, Scatter } from 'recharts';

const ClassificationGraphs = ({ curves, target, model, version, predictions, graphType }) => {
    const [threshold, setThreshold] = useState(0.5);
    const [thresholdMetrics, setThresholdMetrics] = useState(null);
    const [loadingMetrics, setLoadingMetrics] = useState(false);

    useEffect(() => {
        const fetchMetrics = async () => {
            setLoadingMetrics(true);
            try {
                const res = await axios.get(`/api/models/registry/threshold_metrics`, {
                    params: { target, model, version, threshold }
                });
                setThresholdMetrics(res.data.data);
            } catch (err) {
                console.error("Error fetching threshold metrics:", err);
            } finally {
                setLoadingMetrics(false);
            }
        };
        const timer = setTimeout(fetchMetrics, 150); // Debounce
        return () => clearTimeout(timer);
    }, [threshold, target, model, version]);

    if (!curves) return null;

    // Distribution Data
    const probDist = [];
    const nBins = 20;
    for (let i = 0; i < nBins; i++) {
        probDist.push({ bin: (i / nBins).toFixed(2), class0: 0, class1: 0 });
    }
    predictions.forEach(p => {
        const bIdx = Math.min(nBins - 1, Math.floor(p.y_prob * nBins));
        if (p.y_true === 1) probDist[bIdx].class1++;
        else probDist[bIdx].class0++;
    });

    const renderGraph = () => {
        switch (graphType) {
            case 'Threshold Analysis':
                return (
                    <div className="glass-card" style={{ padding: '2rem' }}>
                        <h3 style={{ fontSize: '1.1rem', marginBottom: '1.5rem', fontWeight: 500 }}>Interactive Threshold Analyzer</h3>
                        
                        <div className="threshold-slider-container">
                            <div className="threshold-slider-label">
                                <span style={{ color: 'var(--text-muted)' }}>Decision Threshold</span>
                                <span className="threshold-value">{threshold.toFixed(2)}</span>
                            </div>
                            <input 
                                type="range" 
                                min="0" 
                                max="1" 
                                step="0.01" 
                                value={threshold} 
                                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                                style={{ width: '100%' }}
                            />
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '3rem' }}>
                            {/* Confusion Matrix */}
                            <div>
                                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem', textTransform: 'uppercase' }}>Confusion Matrix</p>
                                {thresholdMetrics ? (
                                    <div className="confusion-matrix">
                                        <div className="cm-cell correct">
                                            <span className="cm-label">True Negative</span>
                                            <span className="cm-value">{thresholdMetrics.confusion_matrix[0][0]}</span>
                                        </div>
                                        <div className="cm-cell">
                                            <span className="cm-label">False Positive</span>
                                            <span className="cm-value">{thresholdMetrics.confusion_matrix[0][1]}</span>
                                        </div>
                                        <div className="cm-cell">
                                            <span className="cm-label">False Negative</span>
                                            <span className="cm-value">{thresholdMetrics.confusion_matrix[1][0]}</span>
                                        </div>
                                        <div className="cm-cell correct">
                                            <span className="cm-label">True Positive</span>
                                            <span className="cm-value">{thresholdMetrics.confusion_matrix[1][1]}</span>
                                        </div>
                                    </div>
                                ) : <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Loading...</div>}
                            </div>

                            {/* Derived Metrics */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', justifyContent: 'center' }}>
                                 <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>Derived Performance</p>
                                 {thresholdMetrics && (
                                     <>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--glass-border)' }}>
                                            <span style={{ color: 'var(--text-muted)' }}>Precision</span>
                                            <span style={{ color: '#fff', fontWeight: 600 }}>{(thresholdMetrics.precision * 100).toFixed(1)}%</span>
                                        </div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--glass-border)' }}>
                                            <span style={{ color: 'var(--text-muted)' }}>Recall (Sensitivity)</span>
                                            <span style={{ color: '#fff', fontWeight: 600 }}>{(thresholdMetrics.recall * 100).toFixed(1)}%</span>
                                        </div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--glass-border)' }}>
                                            <span style={{ color: 'var(--text-muted)' }}>F1-Score</span>
                                            <span style={{ color: 'var(--primary)', fontWeight: 600 }}>{thresholdMetrics.f1.toFixed(3)}</span>
                                        </div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--glass-border)' }}>
                                            <span style={{ color: 'var(--text-muted)' }}>F2-Score (Clinical)</span>
                                            <span style={{ color: '#ff4444', fontWeight: 600 }}>{thresholdMetrics.f2.toFixed(3)}</span>
                                        </div>
                                     </>
                                 )}
                            </div>
                        </div>
                    </div>
                );
            case 'ROC Curve':
                return (
                    <div className="glass-card" style={{ padding: '1.5rem', height: '500px' }}>
                        <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-muted)' }}>ROC Curve (AUC: {curves.roc.auc.toFixed(3)})</h4>
                        <ResponsiveContainer width="100%" height="90%">
                            <LineChart data={curves.roc.data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="var(--text-muted)" fontSize={10} label={{ value: 'False Positive Rate', position: 'bottom', fill: 'var(--text-muted)', fontSize: 10 }} />
                                <YAxis dataKey="tpr" type="number" domain={[0, 1]} stroke="var(--text-muted)" fontSize={10} label={{ value: 'True Positive Rate', angle: -90, position: 'left', fill: 'var(--text-muted)', fontSize: 10 }} />
                                <Line type="monotone" dataKey="tpr" stroke="var(--primary)" dot={false} strokeWidth={2} />
                                <Line data={[{fpr:0, tpr:0}, {fpr:1, tpr:1}]} dataKey="tpr" xDataKey="fpr" stroke="rgba(255,255,255,0.2)" strokeDasharray="5 5" dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                );
            case 'Precision-Recall Curve':
                return (
                    <div className="glass-card" style={{ padding: '1.5rem', height: '500px' }}>
                        <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-muted)' }}>Precision-Recall Curve (AP: {curves.pr.avg_precision.toFixed(3)})</h4>
                        <ResponsiveContainer width="100%" height="90%">
                            <LineChart data={curves.pr.data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="recall" type="number" domain={[0, 1]} stroke="var(--text-muted)" fontSize={10} label={{ value: 'Recall', position: 'bottom', fill: 'var(--text-muted)', fontSize: 10 }} />
                                <YAxis dataKey="precision" type="number" domain={[0, 1]} stroke="var(--text-muted)" fontSize={10} label={{ value: 'Precision', angle: -90, position: 'left', fill: 'var(--text-muted)', fontSize: 10 }} />
                                <Line type="monotone" dataKey="precision" stroke="#ffaa00" dot={false} strokeWidth={2} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                );
            case 'Probability Distribution':
                return (
                    <div className="glass-card" style={{ padding: '1.5rem', height: '500px' }}>
                        <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-muted)' }}>Probability Distribution</h4>
                        <ResponsiveContainer width="100%" height="90%">
                            <AreaChart data={probDist} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="bin" stroke="var(--text-muted)" fontSize={10} label={{ value: 'Predicted Probability Bin', position: 'bottom', fill: 'var(--text-muted)', fontSize: 10 }} />
                                <YAxis stroke="var(--text-muted)" fontSize={10} label={{ value: 'Count', angle: -90, position: 'left', fill: 'var(--text-muted)', fontSize: 10 }} />
                                <Tooltip contentStyle={{ background: 'var(--nav-bg)', border: '1px solid var(--glass-border)' }} />
                                <Area type="monotone" dataKey="class0" stackId="1" stroke="#4444ff" fill="#4444ff" fillOpacity={0.4} name="Negative Cases" />
                                <Area type="monotone" dataKey="class1" stackId="1" stroke="#ff4444" fill="#ff4444" fillOpacity={0.4} name="Positive Cases" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                );
            case 'Calibration Curve':
                return (
                    <div className="glass-card" style={{ padding: '1.5rem', height: '500px' }}>
                        <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-muted)' }}>Calibration Curve</h4>
                        <ResponsiveContainer width="100%" height="90%">
                            <LineChart data={curves.calibration.data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                <XAxis dataKey="mean_predicted" type="number" domain={[0, 1]} stroke="var(--text-muted)" fontSize={10} label={{ value: 'Mean Predicted Probability', position: 'bottom', fill: 'var(--text-muted)', fontSize: 10 }} />
                                <YAxis dataKey="fraction_positive" type="number" domain={[0, 1]} stroke="var(--text-muted)" fontSize={10} label={{ value: 'Fraction of Positives', angle: -90, position: 'left', fill: 'var(--text-muted)', fontSize: 10 }} />
                                <Line type="monotone" dataKey="fraction_positive" stroke="#00ffaa" dot={{ r: 4 }} strokeWidth={2} />
                                <Line data={[{mean_predicted:0, fraction_positive:0}, {mean_predicted:1, fraction_positive:1}]} dataKey="fraction_positive" xDataKey="mean_predicted" stroke="rgba(255,255,255,0.2)" strokeDasharray="5 5" dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
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

export default ClassificationGraphs;
