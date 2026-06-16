import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNexus } from './NexusContext';
import { useCopilot } from './context/CopilotContext';
import { Activity, Target, Zap, Settings, Play, CheckCircle2, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ScatterChart, Scatter, CartesianGrid } from 'recharts';

const Laboratory = () => {
    const [activeTab, setActiveTab] = useState('discovery');

    return (
        <div style={{ animation: 'fadeIn 0.4s ease' }}>
            <header style={{ marginBottom: '2rem' }}>
                <h1 style={{ fontSize: '2.5rem', marginBottom: '0.2rem' }}>The Laboratory</h1>
                <p style={{ color: 'var(--text-muted)' }}>Feature Discovery Engine & Predictive Model Arena.</p>
            </header>

            <div className="sub-tabs">
                <button className={`sub-tab ${activeTab === 'discovery' ? 'active' : ''}`} onClick={() => setActiveTab('discovery')}>Feature Discovery</button>
                <button className={`sub-tab ${activeTab === 'arena' ? 'active' : ''}`} onClick={() => setActiveTab('arena')}>Model Arena</button>
            </div>

            <div style={{ display: activeTab === 'discovery' ? 'block' : 'none' }}>
                <FeatureDiscoveryTab />
            </div>
            <div style={{ display: activeTab === 'arena' ? 'block' : 'none' }}>
                <ModelArenaTab />
            </div>
        </div>
    );
};

const TARGETS = [
    { value: 'dataset_baz_ar', label: 'BAZ Forecast (AR)' },
    { value: 'dataset_baz_delta', label: 'ΔBAZ (Growth Change)' },
    { value: 'dataset_diarrhea', label: 'Diarrhea Risk' },
    { value: 'dataset_illness_burden', label: 'Illness Burden Forecast' },
];



const FeatureDiscoveryTab = () => {
    const { colsList, setColsList } = useNexus();
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState([]);
    const [selections, setSelections] = useState({});
    const [discoveryTarget, setDiscoveryTarget] = useState('');
    
    // V3 Advanced Configs
    const [methods, setMethods] = useState(["Mutual Information", "Correlation Analysis", "SHAP", "Permutation"]);
    const [topN, setTopN] = useState(10);
    const [sampleSize, setSampleSize] = useState(15000);

    const availableMethods = ["Mutual Information", "Correlation Analysis", "SHAP", "Permutation"];

    const toggleMethod = (m) => {
        setMethods(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]);
    };

    // Precomputed cache ref to avoid refetching
    const cacheRef = React.useRef(null);

    useEffect(() => {
        if (!colsList || colsList.length === 0) {
            axios.get('/api/dataset/overview').then(res => {
                if(res.data.columns_list) {
                    setColsList(res.data.columns_list);
                    const filtered = res.data.columns_list.filter(c => !['pid', 'Household_Id', 'agedays'].includes(c));
                    if (!discoveryTarget && filtered.length > 0) {
                        setDiscoveryTarget(filtered[0]);
                    }
                }
            }).catch(console.error);
        } else if (!discoveryTarget && colsList.length > 0) {
            const filtered = colsList.filter(c => !['pid', 'Household_Id', 'agedays'].includes(c));
            setDiscoveryTarget(filtered[0] || colsList[0]);
        }
    }, [colsList]);

    // Silently fetch cache on mount so button press is instant
    useEffect(() => {
        if (cacheRef.current) return;
        axios.get('/api/models/feature_discovery/cache')
            .then(res => {
                if (res.data.status === 'hit') {
                    cacheRef.current = res.data.data;
                }
            })
            .catch(() => {});
    }, []);

    const runDiscovery = () => {
        if (!discoveryTarget) {
            alert("Please select a target column!");
            return;
        }
        
        // Try cache first
        if (cacheRef.current && cacheRef.current[discoveryTarget]) {
            setLoading(true);
            setTimeout(() => {
                const cached = cacheRef.current[discoveryTarget];
                const topResults = cached.results.slice(0, topN);
                setResults(topResults);
                const newSel = {};
                topResults.forEach(r => { newSel[r.feature] = true; });
                setSelections(newSel);
                setLoading(false);
            }, 1500);
            return;
        }
        
        // Fallback to live compute
        setLoading(true);
        axios.post('/api/models/feature_discovery', { target: discoveryTarget, methods: methods, sample_size: sampleSize })
            .then(res => {
                const topResults = res.data.results.slice(0, topN);
                setResults(topResults);
                const newSel = {};
                topResults.forEach(r => { newSel[r.feature] = true; });
                setSelections(newSel);
                setLoading(false);
            })
            .catch(() => setLoading(false));
    };

    const handleSelectAll = (e) => {
        const val = e.target.checked;
        const newSel = {};
        results.forEach(r => { newSel[r.feature] = val; });
        setSelections(newSel);
    };



    return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem' }}>
            <div className="glass-card" style={{ padding: '1.5rem', alignSelf: 'start' }}>
                <h3 style={{ color: '#fff', fontSize: '1.1rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Settings size={18} /> Configuration Panel
                </h3>
                
                <div style={{ marginBottom: '1.5rem' }}>
                    <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Correlation Target</label>
                    <select 
                        className="glass-input" 
                        style={{ width: '100%' }}
                        value={discoveryTarget} 
                        onChange={(e) => setDiscoveryTarget(e.target.value)}
                    >
                        {(colsList || []).filter(c => !['pid', 'Household_Id', 'agedays'].includes(c)).map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                </div>

                <div style={{ marginBottom: '1.5rem' }}>
                    <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Mathematical Methods</label>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
                        {availableMethods.map(m => (
                            <div 
                                key={m} 
                                onClick={() => toggleMethod(m)}
                                style={{ 
                                    padding: '8px 16px', borderRadius: '20px', cursor: 'pointer', fontSize: '0.85rem',
                                    background: methods.includes(m) ? 'var(--primary)' : 'rgba(255,255,255,0.05)',
                                    color: methods.includes(m) ? '#000' : 'var(--text-muted)',
                                    fontWeight: methods.includes(m) ? 600 : 400,
                                    border: `1px solid ${methods.includes(m) ? 'var(--primary)' : 'rgba(255,255,255,0.1)'}`,
                                    transition: 'all 0.2s'
                                }}
                            >
                                {m}
                            </div>
                        ))}
                    </div>
                </div>

                <div style={{ marginBottom: '1.5rem' }}>
                    <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                        <span>Top N Results</span> <span>{topN}</span>
                    </label>
                    <input type="range" min="5" max="50" value={topN} onChange={(e) => setTopN(parseInt(e.target.value))} style={{ width: '100%' }} />
                </div>

                <div style={{ marginBottom: '1.5rem' }}>
                    <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                        <span>Sample Size Throttle</span> <span>{sampleSize.toLocaleString()}</span>
                    </label>
                    <input type="range" min="5000" max="150000" step="5000" value={sampleSize} onChange={(e) => setSampleSize(parseInt(e.target.value))} style={{ width: '100%' }} />
                </div>

                <button 
                    onClick={runDiscovery} 
                    disabled={loading || methods.length === 0}
                    className="btn-primary" 
                    style={{ width: '100%', padding: '12px', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px', background: 'rgba(0, 212, 255, 0.1)' }}
                >
                    {loading ? <Activity size={18} className="spinner" style={{ animation: 'spin 2s linear infinite' }} /> : <Zap size={18} />}
                    {loading ? 'Executing Compute Engine...' : 'Run Discovery Engine'}
                </button>
            </div>

            <div className="glass-card" style={{ padding: '1.5rem', minHeight: '500px', display: 'flex', flexDirection: 'column' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h3 style={{ color: '#fff', fontSize: '1.1rem', margin: 0 }}>Mathematical Consensus Results</h3>
                </div>

                {loading ? (
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                        <Activity size={40} style={{ animation: 'spin 2s linear infinite', marginBottom: '1rem', color: 'var(--primary)' }} />
                        <p>Aggregating Information Matrices & RFE Paths...</p>
                        <p style={{ fontSize: '0.85rem', opacity: 0.8, color: '#fff', marginTop: '10px' }}>
                            Expected compute time: ~{Math.max(1, Math.round(sampleSize / 15000) + (methods.includes("Recursive Feature Elimination (RFE)") ? Math.round(sampleSize / 4000) : 0))} seconds
                        </p>
                    </div>
                ) : results.length === 0 ? (
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', border: '1px dashed rgba(255,255,255,0.1)', borderRadius: '8px' }}>
                        Configure algorithms and run discovery to view blueprint.
                    </div>
                ) : (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', flex: 1 }}>
                        <div className="data-table-container" style={{ maxHeight: '420px' }}>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th style={{ width: '60px', padding: '12px' }}>
                                            <label className="switch" style={{ transform: 'scale(0.8)' }}>
                                                <input type="checkbox" onChange={handleSelectAll} checked={Object.values(selections).every(v=>v) && Object.keys(selections).length>0} />
                                                <span className="slider-toggle"></span>
                                            </label>
                                        </th>
                                        <th>Predictor (Feature)</th>
                                        <th style={{ textAlign: 'right' }}>Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {results.map((r, i) => (
                                        <tr key={i} style={{ opacity: selections[r.feature] ? 1 : 0.5 }}>
                                            <td style={{ padding: '12px' }}>
                                                <label className="switch" style={{ transform: 'scale(0.8)' }}>
                                                    <input type="checkbox" checked={!!selections[r.feature]} onChange={() => setSelections(p => ({...p, [r.feature]: !p[r.feature]}))} />
                                                    <span className="slider-toggle"></span>
                                                </label>
                                            </td>
                                            <td style={{ color: selections[r.feature] ? '#fff' : 'var(--text-muted)' }}>{r.feature}</td>
                                            <td style={{ textAlign: 'right', fontWeight: 600, color: 'var(--primary)' }}>{r.score.toFixed(3)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        
                        <div style={{ height: '420px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={results} layout="vertical" margin={{ top: 0, right: 0, left: 30, bottom: 0 }}>
                                    <XAxis type="number" hide />
                                    <YAxis dataKey="feature" type="category" width={10} hide />
                                    <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{background: 'var(--nav-bg)', border: '1px solid var(--glass-border)'}} />
                                    <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                                        {results.map((entry, index) => {
                                            const lightness = Math.max(30, 65 - (index * 1.5));
                                            return <Cell key={`cell-${index}`} fill={selections[entry.feature] ? `hsl(190, 100%, ${lightness}%)` : 'rgba(255,255,255,0.05)'} />
                                        })}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

const ModelArenaTab = () => {
    const { globalTarget, setGlobalTarget, globalFeatures, taskType, setTaskType, trainedMetrics, setTrainedMetrics, modelVersion } = useNexus();
    const { updateCopilotContext } = useCopilot();
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [statusText, setStatusText] = useState("");
    
    // Clinical Rigor Flags
    const [useKFold, setUseKFold] = useState(true);
    const [forceRetrain, setForceRetrain] = useState(false);
    
    // Ablation Toggles
    const [ablationTemporal, setAblationTemporal] = useState(false);
    const [ablationLag, setAblationLag] = useState(false);
    const [ablationSes, setAblationSes] = useState(false);
    
    // Model Selection (Aligned to Registry)
    const availableModels = ['Random Forest', 'XGBoost', 'CatBoost'];
    const [selectedModels, setSelectedModels] = useState({"Random Forest": true, "XGBoost": true, "CatBoost": true});

    const toggleModel = (m) => setSelectedModels(prev => ({...prev, [m]: !prev[m]}));

    const handleTrain = async () => {
        if(!globalTarget) {
            alert("No target paradigm selected!");
            return;
        }
        
        const modelsToTrain = Object.keys(selectedModels).filter(k => selectedModels[k]);
        if(modelsToTrain.length === 0) return;

        setLoading(true);
        setTrainedMetrics(null);
        let currMetrics = {};

        for(let i=0; i<modelsToTrain.length; i++) {
            const m = modelsToTrain[i];
            setStatusText(`Fitting topology for ${m}...`);
            setProgress(((i) / modelsToTrain.length) * 100);
            
            try {
                // Determine Ablations Logic
                const activeAblations = [];
                if (ablationTemporal) activeAblations.push("temporal");
                if (ablationLag) activeAblations.push("lag");
                if (ablationSes) activeAblations.push("ses");

                let res;
                if (!forceRetrain) {
                    await new Promise(r => setTimeout(r, 400)); // Minimal delay to simulate visual registry pop
                    res = await axios.post('/api/model/score_target', {
                        target: globalTarget,
                        model: m,
                        task_type: taskType,
                        use_kfold: useKFold,
                        force_retrain: false,
                        ablations: [],
                        version: modelVersion
                    });
                } else {
                    res = await axios.post('/api/model/score_target', {
                        target: globalTarget,
                        model: m,
                        task_type: taskType,
                        use_kfold: useKFold,
                        force_retrain: true,
                        ablations: activeAblations,
                        version: modelVersion
                    });
                }
                
                currMetrics = { ...currMetrics, ...res.data.metrics };
                setTrainedMetrics(currMetrics);
                setProgress(((i + 1) / modelsToTrain.length) * 100);
            } catch(e) {
                console.error(`Failed to train ${m}`, e);
            }
        }
        
        setStatusText("Convergence Achieved.");
        setTimeout(() => {
            setLoading(false);
            setProgress(0);
        }, 800);
    };

    // Push context to Copilot
    useEffect(() => {
        const targetLabel = TARGETS.find(t => {
            if (t.value === 'dataset_baz_ar' && globalTarget === 'target') return true;
            if (t.value === 'dataset_baz_delta' && globalTarget === 'target_delta') return true;
            if (t.value === 'dataset_diarrhea' && globalTarget === 'classification_target') return true;
            if (t.value === 'dataset_illness_burden' && globalTarget === 'burden_target') return true;
            return false;
        })?.label || globalTarget;

        const modelsActive = Object.keys(selectedModels).filter(k => selectedModels[k]).join(', ');

        // Build a metrics summary
        let metricsSummary = null;
        let graphData = {};
        if (trainedMetrics) {
            metricsSummary = {};
            for (const [modelName, modelMetrics] of Object.entries(trainedMetrics)) {
                metricsSummary[modelName] = modelMetrics;
            }

            // Find best model
            const modelEntries = Object.entries(trainedMetrics);
            if (modelEntries.length > 0) {
                if (taskType === 'regression') {
                    const best = modelEntries.reduce((a, b) => {
                        const aR2 = a[1]?.r2 ?? a[1]?.R2 ?? a[1]?.['R²'] ?? 0;
                        const bR2 = b[1]?.r2 ?? b[1]?.R2 ?? b[1]?.['R²'] ?? 0;
                        return aR2 > bR2 ? a : b;
                    });
                    graphData.best_model = best[0];
                    graphData.best_metric = 'R²';
                    graphData.best_value = best[1]?.r2 ?? best[1]?.R2 ?? best[1]?.['R²'];
                } else {
                    const best = modelEntries.reduce((a, b) => {
                        const aF1 = a[1]?.f1 ?? a[1]?.F1 ?? 0;
                        const bF1 = b[1]?.f1 ?? b[1]?.F1 ?? 0;
                        return aF1 > bF1 ? a : b;
                    });
                    graphData.best_model = best[0];
                    graphData.best_metric = 'F1';
                    graphData.best_value = best[1]?.f1 ?? best[1]?.F1;
                }
            }

            graphData.all_models = metricsSummary;
            graphData.description = `Model Arena comparison for '${targetLabel}'. ${modelEntries.length} models trained. Task: ${taskType}. Best model: ${graphData.best_model} (${graphData.best_metric}: ${graphData.best_value}).`;
        }

        updateCopilotContext({
            tab: 'model_arena',
            target: targetLabel,
            model: modelsActive,
            graphType: 'model_comparison',
            metrics: metricsSummary,
            graphData,
            graphSummary: graphData.description
                || (trainedMetrics
                    ? `Model Arena results for target '${targetLabel}'. Models trained: ${modelsActive}. Task: ${taskType}.`
                    : `Model Arena configured for target '${targetLabel}'. Models selected: ${modelsActive}. Not yet trained.`)
        });
    }, [globalTarget, selectedModels, trainedMetrics, taskType, updateCopilotContext]);

    return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: '2rem' }}>
            <div className="glass-card" style={{ padding: '1.5rem', alignSelf: 'start' }}>
                <h3 style={{ color: '#fff', fontSize: '1.1rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Target size={18} /> Training Configuration
                </h3>
                
                <div style={{ marginBottom: '1.5rem', padding: '15px', background: 'rgba(0, 212, 255, 0.05)', border: '1px solid rgba(0, 212, 255, 0.2)', borderRadius: '8px' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Target Paradigm</div>
                    <select 
                        className="glass-input" 
                        onChange={(e) => {
                            const val = e.target.value;
                            let newGlobalTarget = "";
                            let newTaskType = "regression";
                            if(val === 'dataset_baz_ar') newGlobalTarget = "target";
                            else if(val === 'dataset_baz_delta') newGlobalTarget = "target_delta";
                            else if(val === 'dataset_diarrhea') { newGlobalTarget = "classification_target"; newTaskType = "classification"; }
                            else if(val === 'dataset_illness_burden') newGlobalTarget = "burden_target";
                            
                            setGlobalTarget(newGlobalTarget);
                            setTaskType(newTaskType);
                        }}
                        defaultValue=""
                        style={{ width: '100%', marginTop: '4px', marginBottom: '10px' }}
                    >
                        <option value="" disabled>--- Select a Target ---</option>
                        {TARGETS.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
                    </select>
                    <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '10px 0' }}></div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>11 Embedded Clinical Multi-Target Features</div>
                    <ul style={{ color: '#fff', fontSize: '0.85rem', marginTop: '8px', paddingLeft: '20px' }}>
                        <li>pid</li>
                        <li>agedays</li>
                        <li>WAMI index</li>
                        <li>target_prev</li>
                        <li>target_velocity</li>
                        <li>burden_illness_30d</li>
                        <li>burden_diarrhea_30d</li>
                        <li>burden_antibiotics_30d</li>
                        <li>recovery_days_since_illness</li>
                        <li>recovery_days_since_diarrhea</li>
                        <li>recovery_days_since_antibiotics</li>
                    </ul>
                    <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '10px 0' }}></div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Task Type</div>
                    <div style={{ color: '#fff', fontWeight: 500, marginTop: '4px', textTransform: 'capitalize' }}>{taskType}</div>
                </div>

                {/* Clinical Rigor Settings */}
                <div style={{ marginBottom: '1.5rem', background: 'rgba(255,255,255,0.02)', padding: '15px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <h4 style={{ color: '#fff', fontSize: '0.9rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <div style={{ width: '4px', height: '14px', background: 'var(--primary)', borderRadius: '2px' }}></div>
                        Clinical Rigor
                    </h4>
                    
                    {/* Group K-Fold Toggle */}
                    <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', marginTop: '10px' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Enable PID Group K-Fold Cross Validation</span>
                        <div className="switch" style={{ transform: 'scale(0.8)' }}>
                            <input type="checkbox" checked={useKFold} onChange={(e) => setUseKFold(e.target.checked)} />
                            <span className="slider-toggle"></span>
                        </div>
                    </label>
                </div>

                {/* Experimental Ablation Arena */}
                <div style={{ marginBottom: '1.5rem', background: 'rgba(255,255,255,0.02)', padding: '15px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <h4 style={{ color: '#fff', fontSize: '0.9rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <div style={{ width: '4px', height: '14px', background: 'var(--primary)', borderRadius: '2px' }}></div>
                        Experimental Ablation Controls
                    </h4>
                    
                    {/* Force Retrain Toggle */}
                    <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', marginTop: '10px' }}>
                        <span style={{ color: forceRetrain ? 'var(--primary)' : 'var(--text-muted)', fontSize: '0.85rem', fontWeight: forceRetrain ? 600 : 400, transition: '0.3s' }}>{forceRetrain ? "LIVE RETRAINING ENABLED" : "USE OFFLINE REGISTRY (INSTANT)"}</span>
                        <div className="switch" style={{ transform: 'scale(0.8)' }}>
                            <input type="checkbox" checked={forceRetrain} onChange={(e) => setForceRetrain(e.target.checked)} />
                            <span className="slider-toggle"></span>
                        </div>
                    </label>

                    {/* Conditional Ablation Grouping UI */}
                    {forceRetrain && (
                        <div style={{ marginTop: '15px', paddingTop: '15px', borderTop: '1px dashed rgba(255,255,255,0.1)', display: 'flex', flexDirection: 'column', gap: '10px', animation: 'fadeIn 0.3s ease' }}>
                            <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}>
                                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }} title="Exclude 'agedays' and 'age_month'">Ablate Temporal Features</span>
                                <input type="checkbox" checked={ablationTemporal} onChange={(e) => setAblationTemporal(e.target.checked)} style={{ transform: 'scale(1.1)' }} />
                            </label>
                            <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}>
                                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }} title="Exclude targets derived longitudinally from memory (target_prev, burden_illness_30d, etc.)">Ablate Lag Memory Modules</span>
                                <input type="checkbox" checked={ablationLag} onChange={(e) => setAblationLag(e.target.checked)} style={{ transform: 'scale(1.1)' }} />
                            </label>
                            <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}>
                                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }} title="Exclude Socioeconomic Status (WAMI index, income, sanitation)">Ablate SES Features</span>
                                <input type="checkbox" checked={ablationSes} onChange={(e) => setAblationSes(e.target.checked)} style={{ transform: 'scale(1.1)' }} />
                            </label>
                        </div>
                    )}
                </div>

                <div style={{ marginBottom: '1.5rem' }}>
                    <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '10px' }}>Algorithm Selectors</label>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {availableModels.map(m => (
                            <label key={m} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', padding: '12px 14px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: selectedModels[m] ? '1px solid var(--glass-border)' : '1px solid transparent', transition: 'all 0.2s' }}>
                                <span style={{ color: selectedModels[m] ? '#fff' : 'var(--text-muted)', fontSize: '0.95rem' }}>{m}</span>
                                <div className="switch">
                                    <input type="checkbox" checked={!!selectedModels[m]} onChange={() => toggleModel(m)} />
                                    <span className="slider-toggle"></span>
                                </div>
                            </label>
                        ))}
                    </div>
                </div>

                <button 
                    onClick={handleTrain} 
                    disabled={loading || !globalTarget}
                    className="btn-primary" 
                    style={{ width: '100%', padding: '12px', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px', opacity: !globalTarget ? 0.3 : 1 }}
                >
                    {loading ? <Activity size={18} className="spinner" style={{ animation: 'spin 2s linear infinite' }} /> : <Play size={18} fill="currentColor" />}
                    {loading ? 'Fitting Topologies...' : 'Execute Training Pipeline'}
                </button>
            </div>

            <div className="glass-card" style={{ padding: '1.5rem', minHeight: '500px' }}>
                <h3 style={{ color: '#fff', fontSize: '1.1rem', marginBottom: '1.5rem' }}>Predictive Performance</h3>
                
                {loading ? (
                    <div style={{ height: '300px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                        <Activity size={40} style={{ animation: 'spin 2s linear infinite', marginBottom: '1rem', color: 'var(--primary)' }} />
                        <p style={{ color: '#fff', fontSize: '1.1rem', marginBottom: '1.5rem' }}>{statusText}</p>
                        
                        {/* Loading Bar Container */}
                        <div style={{ width: '60%', height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ width: `${progress}%`, height: '100%', background: 'var(--primary)', transition: 'width 0.4s ease', boxShadow: '0 0 10px rgba(0, 212, 255, 0.5)' }}></div>
                        </div>
                        <div style={{ fontSize: '0.8rem', marginTop: '10px' }}>{Math.round(progress)}% Complete</div>
                    </div>
                ) : !trainedMetrics || Object.keys(trainedMetrics).length === 0 ? (
                     <div style={{ height: '300px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', border: '1px dashed rgba(255,255,255,0.1)', borderRadius: '8px' }}>
                        <AlertCircle size={30} style={{ opacity: 0.5, marginBottom: '10px' }} />
                        <div>Waiting for model execution...</div>
                        <div style={{ fontSize: '0.8rem', marginTop: '5px' }}>Configure settings and initiate the topological mapping.</div>
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                        <div style={{ padding: '0.5rem', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <table className="data-table">
                                <thead>
                                    <tr>
                                        <th style={{ padding: '12px' }}>Model Topology</th>
                                        {Object.keys(Object.values(trainedMetrics)[0]).map(m => (
                                            <th key={m} style={{ padding: '12px', textAlign: 'right' }}>{m}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.entries(trainedMetrics).map(([modelName, metricsObj]) => (
                                        <tr key={modelName}>
                                            <td style={{ padding: '12px', fontWeight: 600, color: 'var(--primary)' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><CheckCircle2 size={16} /> {modelName}</div>
                                            </td>
                                            {Object.entries(metricsObj).map(([metricName, metricVal]) => (
                                                <td key={metricName} style={{ padding: '12px', textAlign: 'right', fontWeight: 500 }}>{metricVal}</td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        
                        {/* New Visualizer Component */}
                        <ModelVisualizer models={Object.keys(trainedMetrics)} target={globalTarget} taskType={taskType} version={modelVersion} />
                    </div>
                )}
            </div>
        </div>
    );
};

const ModelVisualizer = ({ models, target, taskType, version }) => {
    const [selectedModel, setSelectedModel] = useState(models[0] || "");
    const [visData, setVisData] = useState(null);
    const [loading, setLoading] = useState(false);
    
    // Snapshot the props at mount time so graph data is static until next training
    const snapshotRef = React.useRef({ target, taskType, version, models: models.join(',') });

    // Fetch only when selectedModel changes within the SAME training run
    useEffect(() => {
        if (!selectedModel) return;
        setLoading(true);
        axios.post('/api/models/visualize', {
            model: selectedModel,
            target: snapshotRef.current.target,
            task_type: snapshotRef.current.taskType || 'regression',
            version: snapshotRef.current.version || 'v1'
        })
             .then(res => setVisData(res.data))
             .catch(e => console.error(e))
             .finally(() => setLoading(false));
    }, [selectedModel]);

    return (
        <div style={{ padding: '1.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <h4 style={{ color: '#fff', fontSize: '1.05rem', margin: 0 }}>Model Diagnostic Graphs</h4>
                <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ padding: '8px 12px', background: 'rgba(0,0,0,0.7)', border: '1px solid var(--glass-border)', color: '#fff', borderRadius: '6px', outline: 'none' }}>
                    {models.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
            </div>

            {loading ? (
                <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Activity size={30} style={{ animation: 'spin 2s linear infinite', color: 'var(--text-muted)' }} />
                </div>
            ) : !visData ? (
                <div style={{ height: '300px' }}></div>
            ) : (
                <div style={{ height: '350px', display: 'flex', gap: '2rem' }}>
                    {/* Primary Graph Space */}
                    <div style={{ flex: 1, position: 'relative' }}>
                        {visData.task_type === 'regression' && visData.scatter && (
                            <>
                                <h5 style={{ position: 'absolute', top: -10, left: 30, color: 'var(--text-muted)', fontWeight: 400, zIndex: 10 }}>Predicted vs Actual</h5>
                                <ResponsiveContainer width="100%" height="100%">
                                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                        <XAxis type="number" dataKey="actual" name="Actual" domain={['dataMin - 1', 'dataMax + 1']} stroke="rgba(255,255,255,0.2)" tick={{ fill: 'var(--text-muted)' }} />
                                        <YAxis type="number" dataKey="predicted" name="Predicted" domain={['dataMin - 1', 'dataMax + 1']} stroke="rgba(255,255,255,0.2)" tick={{ fill: 'var(--text-muted)' }} />
                                        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: 'var(--nav-bg)', border: '1px solid var(--primary)', borderRadius: '8px' }} />
                                        <Scatter data={visData.scatter} fill="var(--primary)" opacity={0.6} />
                                    </ScatterChart>
                                </ResponsiveContainer>
                            </>
                        )}
                        {visData.task_type === 'classification' && visData.confusion_matrix && (
                            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                                <h5 style={{ color: 'var(--text-muted)', fontWeight: 400, marginBottom: '1rem' }}>Confusion Matrix</h5>
                                <div style={{ display: 'grid', gridTemplateColumns: `repeat(${visData.confusion_matrix.length}, 1fr)`, gap: '4px', background: 'rgba(255,255,255,0.05)', padding: '4px', borderRadius: '8px' }}>
                                    {visData.confusion_matrix.flatMap((row, rIdx) => 
                                        row.map((cell, cIdx) => {
                                            const isIdentity = rIdx === cIdx;
                                            return (
                                                <div key={`${rIdx}-${cIdx}`} style={{ width: '60px', height: '60px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: isIdentity ? 'rgba(0, 212, 255, 0.2)' : 'rgba(255,255,255,0.02)', border: isIdentity ? '1px solid var(--primary)' : '1px solid transparent', borderRadius: '4px', color: '#fff', fontSize: '1.2rem', fontWeight: isIdentity ? 600 : 400 }}>
                                                    {cell}
                                                </div>
                                            );
                                        })
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                    
                    {/* Optional Feature Importance Panel */}
                    {visData.feature_importance && (
                        <div style={{ width: '300px', display: 'flex', flexDirection: 'column' }}>
                            <h5 style={{ color: 'var(--text-muted)', fontWeight: 400, marginBottom: '0.5rem', alignSelf: 'center' }}>Internal Weighting (Top 10)</h5>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={visData.feature_importance.slice(0, 10)} layout="vertical" margin={{ top: 0, right: 0, left: 10, bottom: 0 }}>
                                    <XAxis type="number" hide />
                                    <YAxis dataKey="feature" type="category" width={10} hide />
                                    <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{background: 'var(--nav-bg)', border: '1px solid var(--glass-border)'}} />
                                    <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                                        {visData.feature_importance.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={'var(--primary)'} opacity={Math.max(0.2, 1 - (index * 0.1))} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default Laboratory;
