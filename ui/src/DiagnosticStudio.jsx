import React, { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import { Activity, AlertCircle, Layers, Zap, BarChart2 } from 'lucide-react';

// Components
import TargetSelector from './components/TargetSelector';
import ModelSelector from './components/ModelSelector';
import SHAPImportance from './components/SHAPImportance';
import Beeswarm from './components/Beeswarm';
import Dependence from './components/Dependence';
import ForcePlot from './components/ForcePlot';
import RegressionGraphs from './components/RegressionGraphs';
import ClassificationGraphs from './components/ClassificationGraphs';
import EmptyState from './components/EmptyState';
import { useCopilot } from './context/CopilotContext';

// --- Stat helpers for graph data summaries ---
const std = (arr) => {
    const m = arr.reduce((a, b) => a + b, 0) / arr.length;
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
};
const median = (arr) => {
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
};
const skewness = (arr) => {
    const m = arr.reduce((a, b) => a + b, 0) / arr.length;
    const s = std(arr);
    if (s === 0) return 0;
    return arr.reduce((acc, v) => acc + ((v - m) / s) ** 3, 0) / arr.length;
};
const computeCorrelation = (x, y) => {
    const n = x.length;
    const mx = x.reduce((a, b) => a + b, 0) / n;
    const my = y.reduce((a, b) => a + b, 0) / n;
    let num = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) {
        num += (x[i] - mx) * (y[i] - my);
        dx += (x[i] - mx) ** 2;
        dy += (y[i] - my) ** 2;
    }
    return dx && dy ? num / Math.sqrt(dx * dy) : 0;
};

export default function DiagnosticStudio() {
    // --- State ---
    const [target, setTarget] = useState('classification_target');
    const [model, setModel] = useState('XGBoost');
    const [version, setVersion] = useState('v1');
    const [activeTab, setActiveTab] = useState('shap'); // 'shap' | 'explorer'
    const [shapSubTab, setShapSubTab] = useState('importance');
    const [graphType, setGraphType] = useState('Prediction vs Actual'); // Default regression graph
    const [selectedFeature, setSelectedFeature] = useState('');
    const [forceSampleIdx, setForceSampleIdx] = useState(0);

    const { updateCopilotContext } = useCopilot();

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [data, setData] = useState({
        preds: [],
        shap: null,
        curves: null,
        meta: null
    });

    // --- Cache ---
    const cacheRef = useRef({}); // { [key]: { preds, shap, curves, meta } }

    const isClassification = target === 'classification_target';
    const taskType = isClassification ? 'classification' : 'regression';
    const currentKey = `${target}|${model}|${version}`;

    // --- Fetching Logic ---
    useEffect(() => {
        const loadBaseData = async () => {
            setLoading(true);
            setError(null);
            
            try {
                if (cacheRef.current[currentKey]) {
                    setData(cacheRef.current[currentKey]);
                    // If we need SHAP but don't have it yet (lazy load)
                    if (activeTab === 'shap' && !cacheRef.current[currentKey].shap) {
                         await fetchShapOnly();
                    }
                } else {
                    // Initial load for this config
                    const [predsRes, metaRes, curvesRes] = await Promise.all([
                        axios.get(`/api/models/registry/predictions`, { params: { target, model, version } }),
                        axios.get(`/api/models/registry/metadata`, { params: { target, model, version } }),
                        isClassification ? axios.get(`/api/models/registry/classification_curves`, { params: { target, model, version } }) : Promise.resolve({ data: { data: null } })
                    ]);

                    const newData = {
                        preds: predsRes.data.data,
                        meta: metaRes.data.data,
                        curves: curvesRes.data.data,
                        shap: null
                    };

                    cacheRef.current[currentKey] = newData;
                    setData(newData);

                    // Lazy load SHAP if active
                    if (activeTab === 'shap') {
                        await fetchShapOnly();
                    }
                }
            } catch (err) {
                console.error("DiagnosticStudio Error:", err);
                setError("Failed to load clinical insights for this configuration.");
            } finally {
                setLoading(false);
            }
        };

        const fetchShapOnly = async () => {
            try {
                const res = await axios.get(`/api/models/registry/shap_detail`, { params: { target, model, version } });
                const shap = res.data.data;
                cacheRef.current[currentKey].shap = shap;
                setData(prev => ({ ...prev, shap }));
            } catch (err) {
                console.warn("SHAP failed to load", err);
            }
        };

        loadBaseData();
    }, [target, model, version, activeTab]);

    // Update default graphType when taskType changes
    useEffect(() => {
        if (isClassification) {
            setGraphType('Threshold Analysis');
        } else {
            setGraphType('Prediction vs Actual');
        }
    }, [isClassification]);

    // Update selected feature when SHAP data arrives
    useEffect(() => {
        if (data.shap?.feature_names?.length > 0 && !selectedFeature) {
            setSelectedFeature(data.shap.feature_names[0]);
        }
    }, [data.shap, selectedFeature]);

    // Push context to Copilot
    useEffect(() => {
        const activeGraphType = activeTab === 'shap' ? shapSubTab : graphType;

        // Build rich graph data based on what's currently visible
        let graphData = {};

        // --- SHAP DATA ---
        if (activeTab === 'shap' && data.shap) {
            const s = data.shap;
            if (shapSubTab === 'importance' && s.feature_names && s.shap_values) {
                // Compute mean |SHAP| per feature
                const meanAbsShap = s.feature_names.map((f, i) => {
                    const vals = s.shap_values.map(row => Math.abs(row[i]));
                    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                    return { feature: f, mean_abs_shap: parseFloat(mean.toFixed(4)) };
                }).sort((a, b) => b.mean_abs_shap - a.mean_abs_shap).slice(0, 10);
                graphData.shap_importance = meanAbsShap;
                graphData.description = `Top 10 features by mean absolute SHAP value. Higher = more influential on predictions.`;
            } else if (shapSubTab === 'beeswarm' && s.feature_names && s.shap_values) {
                const topFeats = s.feature_names.slice(0, 8);
                graphData.beeswarm_features = topFeats;
                graphData.description = `Beeswarm plot showing distribution of SHAP values for top features. Red dots = high feature value, Blue = low. Points right of center push prediction UP.`;
            } else if (shapSubTab === 'dependence' && selectedFeature && s.feature_names) {
                const fi = s.feature_names.indexOf(selectedFeature);
                if (fi >= 0 && s.shap_values && s.feature_values) {
                    const fVals = s.feature_values.map(row => row[fi]);
                    const sVals = s.shap_values.map(row => row[fi]);
                    const corr = computeCorrelation(fVals, sVals);
                    graphData.dependence = {
                        feature: selectedFeature,
                        correlation: parseFloat(corr.toFixed(3)),
                        feature_range: [parseFloat(Math.min(...fVals).toFixed(3)), parseFloat(Math.max(...fVals).toFixed(3))],
                        shap_range: [parseFloat(Math.min(...sVals).toFixed(4)), parseFloat(Math.max(...sVals).toFixed(4))],
                        n_points: fVals.length
                    };
                    graphData.description = `Dependence plot for '${selectedFeature}'. Correlation between feature value and SHAP impact: ${corr.toFixed(3)}. ${Math.abs(corr) > 0.3 ? 'Strong relationship detected.' : 'Weak/non-linear relationship.'}`;
                }
            } else if (shapSubTab === 'force' && s.shap_values && s.feature_names) {
                const idx = Math.min(forceSampleIdx, s.shap_values.length - 1);
                const sampleShap = s.shap_values[idx];
                const sampleFeat = s.feature_values ? s.feature_values[idx] : null;
                const pairs = s.feature_names.map((f, i) => ({
                    feature: f, shap: parseFloat(sampleShap[i].toFixed(4)),
                    value: sampleFeat ? parseFloat(sampleFeat[i].toFixed(3)) : null
                })).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap)).slice(0, 8);
                const pushing_up = pairs.filter(p => p.shap > 0);
                const pushing_down = pairs.filter(p => p.shap < 0);
                graphData.force_plot = {
                    sample_index: idx,
                    base_value: s.base_value ? parseFloat(s.base_value.toFixed(4)) : null,
                    pushing_risk_up: pushing_up,
                    pushing_risk_down: pushing_down
                };
                graphData.description = `Force plot for sample #${idx}. Shows which features push prediction UP (risk) vs DOWN (protective).`;
            }
        }

        // --- REGRESSION GRAPHS ---
        if (activeTab === 'explorer' && data.preds?.length > 0 && !isClassification) {
            const preds = data.preds;
            const actuals = preds.map(p => p.y_true);
            const predicted = preds.map(p => p.y_pred);
            const residuals = preds.map(p => p.y_pred - p.y_true);
            const meanActual = actuals.reduce((a, b) => a + b, 0) / actuals.length;
            const meanPred = predicted.reduce((a, b) => a + b, 0) / predicted.length;
            const meanResidual = residuals.reduce((a, b) => a + b, 0) / residuals.length;
            const absResiduals = residuals.map(Math.abs);
            const mae = absResiduals.reduce((a, b) => a + b, 0) / absResiduals.length;

            if (graphType === 'Prediction vs Actual') {
                graphData.prediction_scatter = {
                    n_points: preds.length,
                    actual_range: [parseFloat(Math.min(...actuals).toFixed(3)), parseFloat(Math.max(...actuals).toFixed(3))],
                    predicted_range: [parseFloat(Math.min(...predicted).toFixed(3)), parseFloat(Math.max(...predicted).toFixed(3))],
                    mean_actual: parseFloat(meanActual.toFixed(3)),
                    mean_predicted: parseFloat(meanPred.toFixed(3)),
                    mae: parseFloat(mae.toFixed(4))
                };
                graphData.description = `Prediction vs Actual scatter. ${preds.length} test samples. MAE: ${mae.toFixed(4)}. Points near the diagonal = good predictions.`;
            } else if (graphType === 'Residual Plot') {
                const posResiduals = residuals.filter(r => r > 0).length;
                const negResiduals = residuals.filter(r => r < 0).length;
                graphData.residuals = {
                    mean_residual: parseFloat(meanResidual.toFixed(4)),
                    std_residual: parseFloat(std(residuals).toFixed(4)),
                    max_overpredict: parseFloat(Math.max(...residuals).toFixed(3)),
                    max_underpredict: parseFloat(Math.min(...residuals).toFixed(3)),
                    pct_overpredicted: parseFloat((posResiduals / preds.length * 100).toFixed(1)),
                    pct_underpredicted: parseFloat((negResiduals / preds.length * 100).toFixed(1))
                };
                graphData.description = `Residual plot. Mean residual: ${meanResidual.toFixed(4)} (ideal=0). ${posResiduals > negResiduals ? 'Model tends to overpredict.' : 'Model tends to underpredict.'}`;
            } else if (graphType === 'Error Distribution') {
                graphData.error_distribution = {
                    mean_error: parseFloat(meanResidual.toFixed(4)),
                    median_error: parseFloat(median(residuals).toFixed(4)),
                    std_error: parseFloat(std(residuals).toFixed(4)),
                    skewness: residuals.length > 0 ? parseFloat(skewness(residuals).toFixed(3)) : 0
                };
                graphData.description = `Error distribution histogram. ${Math.abs(graphData.error_distribution.skewness) < 0.5 ? 'Errors are roughly symmetric (good).' : 'Errors are skewed — model may be biased.'}`;
            }
        }

        // --- CLASSIFICATION GRAPHS ---
        if (activeTab === 'explorer' && data.preds?.length > 0 && isClassification) {
            const preds = data.preds;
            if (graphType === 'Threshold Analysis' && data.curves) {
                const tp = preds.filter(p => p.y_true === 1 && p.y_prob >= 0.5).length;
                const fp = preds.filter(p => p.y_true === 0 && p.y_prob >= 0.5).length;
                const tn = preds.filter(p => p.y_true === 0 && p.y_prob < 0.5).length;
                const fn = preds.filter(p => p.y_true === 1 && p.y_prob < 0.5).length;
                graphData.confusion_matrix = { tp, fp, tn, fn, total: preds.length };
                graphData.description = `Confusion matrix at threshold=0.5. TP:${tp}, FP:${fp}, TN:${tn}, FN:${fn}. Precision: ${(tp/(tp+fp||1)*100).toFixed(1)}%, Recall: ${(tp/(tp+fn||1)*100).toFixed(1)}%`;
            } else if (graphType === 'ROC Curve' && data.curves?.roc) {
                graphData.roc = { auc: data.curves.roc.auc, n_points: data.curves.roc.data?.length || 0 };
                graphData.description = `ROC Curve. AUC: ${data.curves.roc.auc.toFixed(3)}. ${data.curves.roc.auc > 0.9 ? 'Excellent discriminative ability.' : data.curves.roc.auc > 0.8 ? 'Good discriminative ability.' : 'Moderate discriminative ability.'}`;
            } else if (graphType === 'Precision-Recall Curve' && data.curves?.pr) {
                graphData.pr = { avg_precision: data.curves.pr.avg_precision };
                graphData.description = `Precision-Recall curve. Average Precision: ${data.curves.pr.avg_precision.toFixed(3)}. ${data.curves.pr.avg_precision > 0.8 ? 'Strong performance even with class imbalance.' : 'May struggle with minority class.'}`;
            } else if (graphType === 'Calibration Curve' && data.curves?.calibration) {
                graphData.calibration = { n_bins: data.curves.calibration.data?.length || 0 };
                graphData.description = `Calibration curve. Shows if predicted probabilities match actual outcomes. Points near the diagonal = well-calibrated model.`;
            } else if (graphType === 'Probability Distribution') {
                const pos = preds.filter(p => p.y_true === 1);
                const neg = preds.filter(p => p.y_true === 0);
                graphData.prob_distribution = {
                    positive_mean_prob: parseFloat((pos.reduce((a,p) => a+p.y_prob,0)/pos.length).toFixed(3)),
                    negative_mean_prob: parseFloat((neg.reduce((a,p) => a+p.y_prob,0)/neg.length).toFixed(3)),
                    separation: parseFloat(Math.abs((pos.reduce((a,p) => a+p.y_prob,0)/pos.length) - (neg.reduce((a,p) => a+p.y_prob,0)/neg.length)).toFixed(3))
                };
                graphData.description = `Probability distribution. Class separation: ${graphData.prob_distribution.separation.toFixed(3)}. ${graphData.prob_distribution.separation > 0.4 ? 'Good separation between classes.' : 'Classes overlap significantly.'}`;
            }
        }

        updateCopilotContext({
            tab: 'diagnostic',
            target,
            model,
            graphType: activeGraphType,
            metrics: data.meta?.metrics || null,
            graphData,
            graphSummary: graphData.description || `User is viewing ${activeGraphType} for model ${model} on target ${target}.`
        });
    }, [target, model, activeTab, shapSubTab, graphType, data, selectedFeature, forceSampleIdx, isClassification, updateCopilotContext]);


    // --- Render Helpers ---
    const renderContent = () => {
        if (activeTab === 'shap') {
            if (!data.shap) return <EmptyState message="Loading SHAP explanations..." icon="search" />;
            
            switch (shapSubTab) {
                case 'importance': return <SHAPImportance data={data.shap.global_importance} />;
                case 'beeswarm': return <Beeswarm detail={data.shap} />;
                case 'dependence': return <Dependence detail={data.shap} feature={selectedFeature} />;
                case 'force': return <ForcePlot detail={data.shap} sampleIndex={forceSampleIdx} />;
                default: return null;
            }
        } else {
            if (!data.preds || data.preds.length === 0) return <EmptyState />;
            
            if (isClassification) {
                return <ClassificationGraphs 
                    curves={data.curves} 
                    predictions={data.preds} 
                    target={target} 
                    model={model} 
                    version={version} 
                    graphType={graphType}
                />;
            } else {
                return <RegressionGraphs predictions={data.preds} graphType={graphType} />;
            }
        }
    };

    return (
        <div style={{ animation: 'fadeIn 0.5s ease' }}>
            <header style={{ marginBottom: '2.5rem' }}>
                <h1 style={{ fontSize: '2.8rem', fontWeight: 600, letterSpacing: '-0.02em', marginBottom: '0.5rem' }} className="gradient-text">
                    Diagnostic Studio
                </h1>
                <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem' }}>
                    Clinical Decision Support & Explainable AI Pipeline
                </p>
            </header>

            <div className="sub-tabs" style={{ marginBottom: '2.5rem' }}>
                <button className={`sub-tab ${activeTab === 'shap' ? 'active' : ''}`} onClick={() => setActiveTab('shap')}>Explainability (SHAP)</button>
                <button className={`sub-tab ${activeTab === 'explorer' ? 'active' : ''}`} onClick={() => setActiveTab('explorer')}>Performance Explorer</button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '2.5rem' }}>
                {/* Sidebar Controls */}
                <div className="glass-card" style={{ padding: '1.8rem', alignSelf: 'start', position: 'sticky', top: '100px' }}>
                    <div style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <Layers size={20} color="var(--primary)" />
                        <h3 style={{ fontSize: '1.1rem', fontWeight: 500 }}>Configurator</h3>
                    </div>

                    <TargetSelector target={target} setTarget={setTarget} />
                    <ModelSelector model={model} setModel={setModel} />

                    {/* Removed Active Lens from Sidebar */}

                    <div style={{ marginTop: '2.5rem', paddingTop: '2rem', borderTop: '1px solid var(--glass-border)' }}>
                    {activeTab === 'explorer' && (
                        <div style={{ marginTop: '2rem', animation: 'fadeIn 0.3s ease' }}>
                            <label className="sidebar-label">Graph Visualization</label>
                            <select className="glass-input" value={graphType} onChange={(e) => setGraphType(e.target.value)}>
                                {isClassification ? (
                                    <>
                                        <option value="Threshold Analysis">Threshold Analysis</option>
                                        <option value="ROC Curve">ROC Curve</option>
                                        <option value="Precision-Recall Curve">Precision-Recall Curve</option>
                                        <option value="Calibration Curve">Calibration Curve</option>
                                        <option value="Probability Distribution">Probability Distribution</option>
                                    </>
                                ) : (
                                    <>
                                        <option value="Prediction vs Actual">Prediction vs Actual</option>
                                        <option value="Residual Plot">Residual Plot</option>
                                        <option value="Error Distribution">Error Distribution</option>
                                        <option value="Time-Series Trajectory">Time-Series Trajectory</option>
                                    </>
                                )}
                            </select>
                        </div>
                    )}

                    {activeTab === 'shap' && (
                        <div style={{ marginTop: '2rem', animation: 'fadeIn 0.3s ease' }}>
                            <label className="sidebar-label">XAI Projection</label>
                            <select className="glass-input" value={shapSubTab} onChange={(e) => setShapSubTab(e.target.value)}>
                                <option value="importance">Global Importance</option>
                                <option value="beeswarm">Beeswarm (Impact)</option>
                                <option value="dependence">Dependence (Trend)</option>
                                <option value="force">Local Force Plot</option>
                            </select>

                            {shapSubTab === 'dependence' && (
                                <div style={{ marginTop: '1rem' }}>
                                    <label className="sidebar-label">Feature Focus</label>
                                    <select className="glass-input" value={selectedFeature} onChange={(e) => setSelectedFeature(e.target.value)}>
                                        {data.shap?.feature_names?.map(f => <option key={f} value={f}>{f}</option>)}
                                    </select>
                                </div>
                            )}

                            {shapSubTab === 'force' && (
                                <div style={{ marginTop: '1rem' }}>
                                    <label className="sidebar-label">Sample Index</label>
                                    <input 
                                        type="number" 
                                        className="glass-input" 
                                        value={forceSampleIdx} 
                                        onChange={(e) => setForceSampleIdx(parseInt(e.target.value))}
                                        min={0}
                                        max={data.shap?.shap_values?.length - 1}
                                    />
                                </div>
                            )}
                        </div>
                    )}
                    </div>
                </div>

                {/* Main Viewport */}
                <div className="glass-card" style={{ padding: '1.5rem', minHeight: '550px', display: 'flex', flexDirection: 'column' }}>
                    {loading && !data.preds.length ? (
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                            <Activity className="spin" size={48} color="var(--primary)" />
                            <p style={{ marginTop: '1.5rem', color: 'var(--text-muted)' }}>Synthesizing clinical insights...</p>
                        </div>
                    ) : error ? (
                        <EmptyState message={error} icon="info" />
                    ) : (
                        <div style={{ animation: 'fadeIn 0.4s ease', flex: 1 }}>
                             {renderContent()}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
