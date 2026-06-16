import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Brain, Compass, Cpu, Info, Zap, AlertTriangle, Play } from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, ReferenceLine, Legend, BarChart, Bar } from 'recharts';
import { useCopilot } from './context/CopilotContext';

export default function TemporalLab() {
  const [activeTab, setActiveTab] = useState('sequence');
  
  return (
    <div style={{ animation: 'fadeIn 0.4s ease', padding: '1rem' }}>
      <header style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.2rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <Activity color="var(--primary)" size={32} /> Temporal Intelligence Lab
        </h1>
        <p style={{ color: 'var(--text-muted)' }}>Advanced sequence forecasting, temporal embeddings, and hybrid what-if simulations.</p>
      </header>

      <div className="sub-tabs" style={{ marginBottom: '2rem' }}>
        <button 
          className={`sub-tab ${activeTab === 'sequence' ? 'active' : ''}`} 
          onClick={() => setActiveTab('sequence')}
          style={{ fontSize: '1rem', fontWeight: 500 }}
        >
          Sequence Modeling
        </button>
        <button 
          className={`sub-tab ${activeTab === 'simulation' ? 'active' : ''}`} 
          onClick={() => setActiveTab('simulation')}
          style={{ fontSize: '1rem', fontWeight: 500 }}
        >
          Simulation Engine
        </button>
      </div>

      {activeTab === 'sequence' && <SequenceModelingTab />}
      {activeTab === 'simulation' && <SimulationEngineTab />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// TAB 1: SEQUENCE MODELING
// ---------------------------------------------------------------------------
function SequenceModelingTab() {
  const { updateCopilotContext } = useCopilot();
  const [target, setTarget] = useState('recovery');
  const [model, setModel] = useState('lstm');
  
  // Stages: idle -> training -> trained -> extracting -> embedded -> merging -> hybrid
  const [stage, setStage] = useState('idle');
  const [metrics, setMetrics] = useState(null);
  const [embeddings, setEmbeddings] = useState(null);
  const [predictionSamples, setPredictionSamples] = useState(null);
  const [activeGraph, setActiveGraph] = useState('performance');

  const handleTrain = () => {
    setStage('training');
    setTimeout(() => {
      axios.get('/api/temporal/metrics').then(res => {
        setMetrics(res.data.data);
        setPredictionSamples(res.data.samples);
        setStage('trained');
      }).catch(err => {
        console.error(err);
        setStage('idle');
      });
    }, 1500);
  };

  const handleExtract = () => {
    setStage('extracting');
    setTimeout(() => {
      axios.get(`/api/temporal/embeddings?target=${target}&model_type=${model}`).then(res => {
        setEmbeddings(res.data.data);
        setStage('embedded');
      }).catch(err => {
        console.error(err);
        setStage('trained');
      });
    }, 2000);
  };

  const handleMerge = () => {
    setStage('merging');
    setTimeout(() => {
      setStage('hybrid');
    }, 2000);
  };

  // Helper to filter metrics based on stage
  const getDisplayMetrics = () => {
    if (!metrics || !metrics[target]) return [];
    const allMetrics = metrics[target];
    if (stage === 'trained' || stage === 'extracting' || stage === 'embedded' || stage === 'merging') {
      // Show only Baseline, V1, and the selected sequence model (LSTM or TCN)
      return allMetrics.filter(m => m.Model.includes('Baseline') || m.Model.includes('V1') || m.Model.toLowerCase() === model);
    }
    if (stage === 'hybrid') {
      // Show Baseline, V1, selected model, and the matching hybrid model
      return allMetrics.filter(m => 
        m.Model.includes('Baseline') || 
        m.Model.includes('V1') || 
        m.Model.toLowerCase() === model || 
        m.Model.toLowerCase() === `hybrid_${model}`
      );
    }
    return [];
  };

  const displayMetrics = getDisplayMetrics();
  const baseR2 = displayMetrics.length > 0 ? displayMetrics.find(m => m.Model.includes('Baseline'))?.R2 ?? 0 : 0;

  // Mock bar chart data for R2 comparison
  const chartData = displayMetrics.filter(m => !m.Model.includes('Baseline')).map(m => ({
    name: m.Model,
    R2: m.R2,
    MAE: m.MAE
  }));

  const getScatterData = (modelKey) => {
    if (!predictionSamples || !predictionSamples[target] || !predictionSamples[target][modelKey]) return [];
    const data = predictionSamples[target][modelKey];
    return data.actual.map((act, i) => ({
      actual: act,
      predicted: data.predicted[i],
      error: data.predicted[i] - act
    }));
  };

  const getErrorHistogram = (modelKey) => {
    if (!predictionSamples || !predictionSamples[target] || !predictionSamples[target][modelKey]) return [];
    const data = predictionSamples[target][modelKey];
    const errors = data.predicted.map((p, i) => p - data.actual[i]);
    const buckets = { '<-20':0, '-20 to -10':0, '-10 to 0':0, '0 to 10':0, '10 to 20':0, '>20':0 };
    errors.forEach(e => {
      if(e < -20) buckets['<-20']++;
      else if(e < -10) buckets['-20 to -10']++;
      else if(e < 0) buckets['-10 to 0']++;
      else if(e < 10) buckets['0 to 10']++;
      else if(e < 20) buckets['10 to 20']++;
      else buckets['>20']++;
    });
    return Object.keys(buckets).map(k => ({ range: k, count: buckets[k] }));
  };

  useEffect(() => {
    let graphData = {};

    // Build scoped metrics that match what the user actually sees (matching getDisplayMetrics logic)
    let scopedMetrics = null;
    if (metrics && metrics[target]) {
      const allMetrics = metrics[target];
      let visibleModels = [];
      if (stage === 'trained' || stage === 'extracting' || stage === 'embedded' || stage === 'merging') {
        visibleModels = allMetrics.filter(m => m.Model.includes('Baseline') || m.Model.includes('V1') || m.Model.toLowerCase() === model);
      } else if (stage === 'hybrid') {
        visibleModels = allMetrics.filter(m => 
          m.Model.includes('Baseline') || 
          m.Model.includes('V1') || 
          m.Model.toLowerCase() === model || 
          m.Model.toLowerCase() === `hybrid_${model}`
        );
      }
      if (visibleModels.length > 0) {
        scopedMetrics = visibleModels;
      }
    }

    // Get the currently active model key for context
    const activeModelKey = stage === 'hybrid' ? `Hybrid_${model.toUpperCase()}` : model.toUpperCase();

    // Get prediction data for the currently selected model
    const currentPreds = predictionSamples?.[target]?.[activeModelKey];

    if (currentPreds && activeGraph === 'scatter') {
        const actuals = currentPreds.actual;
        const preds = currentPreds.predicted;
        const errors = actuals.map((a, i) => Math.abs(a - preds[i]));
        graphData.scatter_stats = {
            n_samples: actuals.length,
            mean_actual: parseFloat((actuals.reduce((a,b) => a+b,0)/actuals.length).toFixed(3)),
            mean_predicted: parseFloat((preds.reduce((a,b) => a+b,0)/preds.length).toFixed(3)),
            mean_abs_error: parseFloat((errors.reduce((a,b) => a+b,0)/errors.length).toFixed(4)),
        };
        graphData.description = `Prediction scatter for ${activeModelKey} on target '${target}'. ${actuals.length} samples. Mean error: ${graphData.scatter_stats.mean_abs_error}`;
    }

    if (currentPreds && activeGraph === 'error') {
        const errors = currentPreds.actual.map((a, i) => Math.abs(a - currentPreds.predicted[i]));
        const buckets = { '<5':0, '5-10':0, '10-20':0, '20-30':0, '>30':0 };
        errors.forEach(e => {
            if(e < 5) buckets['<5']++;
            else if(e < 10) buckets['5-10']++;
            else if(e < 20) buckets['10-20']++;
            else if(e < 30) buckets['20-30']++;
            else buckets['>30']++;
        });
        graphData.error_histogram = Object.keys(buckets).map(k => ({ range: k, count: buckets[k] }));
        graphData.description = `Error distribution for ${activeModelKey} on target '${target}'. Shows how prediction errors are spread across ranges.`;
    }

    if (activeGraph === 'performance' && metrics) {
        graphData.description = `Performance overview for target '${target}'. Comparing models on MAE, RMSE, and R².`;
    }

    updateCopilotContext({
        tab: 'sequence_modeling',
        target,
        model: activeModelKey,
        graphType: activeGraph,
        stage: stage,
        metrics: scopedMetrics,
        graphData,
        graphSummary: graphData.description || `User is viewing Sequence Modeling (${target}/${activeModelKey}), stage: ${stage}`
    });
  }, [target, model, activeGraph, metrics, predictionSamples, stage, updateCopilotContext]);

  try {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '350px 1fr', gap: '2rem' }}>
        
        {/* LEFT PANEL: Controls & Pipeline Actions */}
        <div className="glass-card" style={{ padding: '1.5rem', height: 'fit-content' }}>
          <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Cpu size={18} /> Pipeline Simulator
          </h3>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Target Outcome</label>
            <select value={target} onChange={(e) => { setTarget(e.target.value); setStage('idle'); }} disabled={stage !== 'idle'} style={{ width: '100%', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}>
              <option value="illness">Illness Burden</option>
              <option value="recovery">Time-to-Recovery</option>
              <option value="delta_baz">ΔBAZ</option>
            </select>
          </div>

          <div style={{ marginBottom: '2rem' }}>
            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Sequence Backbone</label>
            <select value={model} onChange={(e) => { setModel(e.target.value); setStage('idle'); }} disabled={stage !== 'idle'} style={{ width: '100%', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}>
              <option value="lstm">LSTM (Long Short-Term Memory)</option>
              <option value="tcn">TCN (Temporal Convolutional Network)</option>
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', borderTop: '1px dashed rgba(255,255,255,0.1)', paddingTop: '1.5rem' }}>
            
            {/* STEP 1: Train Sequence Model */}
            <div style={{ opacity: 1, transition: '0.3s' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Step 1: Learn Temporal Patterns</div>
              <button 
                onClick={handleTrain}
                disabled={stage !== 'idle'}
                style={{ width: '100%', padding: '0.8rem', background: stage === 'idle' ? 'var(--primary)' : 'rgba(255,255,255,0.05)', color: stage === 'idle' ? '#000' : 'var(--text-muted)', border: 'none', borderRadius: '4px', fontWeight: 'bold', cursor: stage === 'idle' ? 'pointer' : 'not-allowed', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}>
                {stage === 'training' ? <><Activity size={16} className="spinner" style={{ animation: 'spin 2s linear infinite' }} /> Training...</> : (stage === 'idle' ? 'Train Sequence Model' : '✓ Model Trained')}
              </button>
            </div>

            {/* STEP 2: Extract Embeddings */}
            <div style={{ opacity: (stage === 'idle' || stage === 'training') ? 0.3 : 1, transition: '0.3s' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Step 2: Compress into Embeddings</div>
              <button 
                onClick={handleExtract}
                disabled={stage !== 'trained'}
                style={{ width: '100%', padding: '0.8rem', background: stage === 'trained' ? 'var(--primary)' : 'rgba(255,255,255,0.05)', color: stage === 'trained' ? '#000' : 'var(--text-muted)', border: 'none', borderRadius: '4px', fontWeight: 'bold', cursor: stage === 'trained' ? 'pointer' : 'not-allowed', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}>
                {stage === 'extracting' ? <><Zap size={16} style={{ animation: 'pulse 1s infinite' }} /> Extracting Latent Space...</> : (['embedded', 'merging', 'hybrid'].includes(stage) ? '✓ Embeddings Extracted' : 'Extract Embeddings')}
              </button>
            </div>

            {/* STEP 3: Build Hybrid */}
            <div style={{ opacity: ['embedded', 'merging', 'hybrid'].includes(stage) ? 1 : 0.3, transition: '0.3s' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Step 3: Combine with Clinical Context</div>
              <button 
                onClick={handleMerge}
                disabled={stage !== 'embedded'}
                style={{ width: '100%', padding: '0.8rem', background: stage === 'embedded' ? 'var(--primary)' : 'rgba(255,255,255,0.05)', color: stage === 'embedded' ? '#000' : 'var(--text-muted)', border: 'none', borderRadius: '4px', fontWeight: 'bold', cursor: stage === 'embedded' ? 'pointer' : 'not-allowed', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}>
                {stage === 'merging' ? <><Brain size={16} style={{ animation: 'pulse 1s infinite' }} /> Building Hybrid...</> : (stage === 'hybrid' ? '✓ Hybrid Compiled' : 'Merge with XGBoost')}
              </button>
            </div>

            {stage === 'hybrid' && (
              <button onClick={() => setStage('idle')} style={{ marginTop: '1rem', background: 'transparent', border: '1px solid var(--primary)', color: 'var(--primary)', padding: '0.5rem', borderRadius: '4px', cursor: 'pointer' }}>
                Reset Pipeline
              </button>
            )}

          </div>
        </div>

        {/* RIGHT PANEL: Results (Tables + Graphs) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          
          {stage === 'idle' || stage === 'training' ? (
            <div className="glass-card" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', borderStyle: 'dashed' }}>
               {stage === 'training' ? (
                 <>
                   <Activity size={48} style={{ animation: 'spin 2s linear infinite', marginBottom: '1rem', color: 'var(--primary)' }} />
                   <p>Training {model.toUpperCase()} on temporal sequences...</p>
                 </>
               ) : (
                 <>
                   <Cpu size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                   <p>Configure target and backbone, then initiate sequence training.</p>
                 </>
               )}
            </div>
          ) : (
            <div style={{ animation: 'fadeIn 0.4s ease' }}>
              
              {/* Embedded State Visualization */}
              {(stage === 'embedded' || stage === 'extracting') && (
                 <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
                   <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem' }}><Zap size={16} /> Latent Space Projection</h3>
                   {stage === 'extracting' ? (
                      <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--primary)' }}>
                        <svg width="200" height="40" viewBox="0 0 200 40">
                          <rect x="10" y="5" width="50" height="30" rx="4" fill="var(--glass-border)" stroke="#fff" />
                          <text x="35" y="25" fill="#fff" fontSize="10" textAnchor="middle">{model.toUpperCase()}</text>
                          <line x1="60" y1="20" x2="140" y2="20" stroke="var(--text-muted)" strokeDasharray="4 4" />
                          <rect x="140" y="5" width="50" height="30" rx="4" fill="var(--glass-border)" stroke="#fff" />
                          <text x="165" y="25" fill="#fff" fontSize="10" textAnchor="middle">Latent</text>
                          <circle r="4" fill="#00d4ff" filter="drop-shadow(0 0 4px #00d4ff)">
                            <animate attributeName="cx" values="60;140;140" keyTimes="0;0.5;1" dur="2s" repeatCount="indefinite" />
                            <animate attributeName="cy" values="20;20;20" dur="2s" repeatCount="indefinite" />
                            <animate attributeName="opacity" values="1;1;0" keyTimes="0;0.5;1" dur="2s" repeatCount="indefinite" />
                          </circle>
                        </svg>
                      </div>
                   ) : (
                     <div style={{ height: '300px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis type="number" dataKey="x" hide />
                          <YAxis type="number" dataKey="y" hide />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#111', border: '1px solid var(--glass-border)' }} />
                          <Scatter name="Embeddings" data={embeddings} fill="var(--primary)" opacity={0.6} />
                        </ScatterChart>
                      </ResponsiveContainer>
                     </div>
                   )}
                 </div>
              )}
              
              {/* Merging Animation */}
              {stage === 'merging' && (
                <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                   <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem', alignSelf: 'flex-start' }}><Brain size={16} /> Building Hybrid</h3>
                   <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                     <svg width="200" height="60" viewBox="0 0 200 60">
                       <rect x="10" y="5" width="50" height="20" rx="4" fill="var(--glass-border)" stroke="#fff" />
                       <text x="35" y="18" fill="#fff" fontSize="10" textAnchor="middle">Latent</text>
                       <rect x="10" y="35" width="50" height="20" rx="4" fill="var(--glass-border)" stroke="#fff" />
                       <text x="35" y="48" fill="#fff" fontSize="10" textAnchor="middle">Clinical</text>
                       <line x1="60" y1="15" x2="140" y2="30" stroke="var(--text-muted)" strokeDasharray="4 4" />
                       <line x1="60" y1="45" x2="140" y2="30" stroke="var(--text-muted)" strokeDasharray="4 4" />
                       <rect x="140" y="15" width="50" height="30" rx="4" fill="var(--primary)" fillOpacity="0.2" stroke="var(--primary)" strokeWidth="2">
                         <animate attributeName="fill-opacity" values="0.2;0.8;0.2" dur="2s" repeatCount="indefinite" />
                       </rect>
                       <text x="165" y="35" fill="#fff" fontSize="10" textAnchor="middle" fontWeight="bold">XGBOOST</text>
                       <circle r="4" fill="#00d4ff" filter="drop-shadow(0 0 4px #00d4ff)">
                         <animate attributeName="cx" values="60;140;140" keyTimes="0;0.5;1" dur="2s" repeatCount="indefinite" />
                         <animate attributeName="cy" values="15;30;30" keyTimes="0;0.5;1" dur="2s" repeatCount="indefinite" />
                         <animate attributeName="opacity" values="1;1;0" keyTimes="0;0.5;1" dur="2s" repeatCount="indefinite" />
                       </circle>
                       <circle r="4" fill="#4ade80" filter="drop-shadow(0 0 4px #4ade80)">
                         <animate attributeName="cx" values="60;140;140" keyTimes="0;0.5;1" dur="2s" repeatCount="indefinite" />
                         <animate attributeName="cy" values="45;30;30" keyTimes="0;0.5;1" dur="2s" repeatCount="indefinite" />
                         <animate attributeName="opacity" values="1;1;0" keyTimes="0;0.5;1" dur="2s" repeatCount="indefinite" />
                       </circle>
                     </svg>
                   </div>
                </div>
              )}

              {/* Metrics Table */}
              {stage !== 'extracting' && stage !== 'embedded' && stage !== 'merging' && (
                <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
                  <h3 style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem', marginBottom: '1rem', textTransform: 'uppercase', letterSpacing: '1px' }}>
                    Target: {target.replace('_', ' ')}
                  </h3>
                  <table style={{ width: '100%', fontSize: '0.85rem' }}>
                    <thead>
                      <tr style={{ color: 'var(--primary)', textAlign: 'left' }}>
                        <th style={{ padding: '0.5rem' }}>Model</th>
                        <th style={{ padding: '0.5rem' }}>R²</th>
                        <th style={{ padding: '0.5rem' }}>MAE</th>
                        <th style={{ padding: '0.5rem' }}>Lift</th>
                      </tr>
                    </thead>
                    <tbody>
                      {displayMetrics.map((r, i) => {
                        const lift = (r.R2 - baseR2).toFixed(4);
                        return (
                        <tr key={i} style={{ background: i % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'transparent' }}>
                          <td style={{ padding: '0.5rem', fontWeight: r.Model.includes('Hybrid') ? 'bold' : 'normal', color: r.Model.includes('Hybrid') ? 'var(--primary)' : 'inherit' }}>
                            {r.Model}
                          </td>
                          <td style={{ padding: '0.5rem' }}>{r.R2}</td>
                          <td style={{ padding: '0.5rem' }}>{r.MAE}</td>
                          <td style={{ padding: '0.5rem' }}>
                            <span style={{ color: lift > 0 ? '#4ade80' : '#f87171' }}>
                              {lift > 0 ? '+' : ''}{lift}
                            </span>
                          </td>
                        </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Performance Graphs Selector */}
              {(stage === 'trained' || stage === 'hybrid') && (
                <div className="glass-card" style={{ padding: '1.5rem' }}>
                  <div style={{ display: 'flex', gap: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem', marginBottom: '1rem' }}>
                    <button 
                      onClick={() => setActiveGraph('performance')}
                      style={{ background: activeGraph === 'performance' ? 'var(--primary)' : 'transparent', color: activeGraph === 'performance' ? '#000' : 'var(--text-muted)', border: '1px solid var(--primary)', padding: '0.5rem 1rem', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>
                      R² Validation
                    </button>
                    <button 
                      onClick={() => setActiveGraph('residuals')}
                      disabled={!predictionSamples}
                      style={{ background: activeGraph === 'residuals' ? 'var(--primary)' : 'transparent', color: activeGraph === 'residuals' ? '#000' : 'var(--text-muted)', border: '1px solid var(--primary)', padding: '0.5rem 1rem', borderRadius: '4px', cursor: predictionSamples ? 'pointer' : 'not-allowed', fontWeight: 'bold', opacity: predictionSamples ? 1 : 0.5 }}>
                      Residuals
                    </button>
                    <button 
                      onClick={() => setActiveGraph('error')}
                      disabled={!predictionSamples}
                      style={{ background: activeGraph === 'error' ? 'var(--primary)' : 'transparent', color: activeGraph === 'error' ? '#000' : 'var(--text-muted)', border: '1px solid var(--primary)', padding: '0.5rem 1rem', borderRadius: '4px', cursor: predictionSamples ? 'pointer' : 'not-allowed', fontWeight: 'bold', opacity: predictionSamples ? 1 : 0.5 }}>
                      Error Distribution
                    </button>
                  </div>

                  <div style={{ height: '300px' }}>
                    {activeGraph === 'performance' && (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis dataKey="name" stroke="var(--text-muted)" />
                          <YAxis stroke="var(--text-muted)" domain={['auto', 'auto']} />
                          <Tooltip contentStyle={{ backgroundColor: '#111', border: '1px solid var(--glass-border)' }} />
                          <Bar dataKey="R2" fill="var(--primary)" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    )}
                    
                    {activeGraph === 'residuals' && predictionSamples && (
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis type="number" dataKey="actual" name="Actual" stroke="var(--text-muted)" />
                          <YAxis type="number" dataKey="predicted" name="Predicted" stroke="var(--text-muted)" />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#111', border: '1px solid var(--glass-border)' }} />
                          <Scatter name={stage === 'hybrid' ? `Hybrid_${model.toUpperCase()}` : model.toUpperCase()} data={getScatterData(stage === 'hybrid' ? `Hybrid_${model.toUpperCase()}` : model.toUpperCase())} fill="#4ade80" opacity={0.6} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    )}

                    {activeGraph === 'error' && predictionSamples && (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={getErrorHistogram(stage === 'hybrid' ? `Hybrid_${model.toUpperCase()}` : model.toUpperCase())} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis dataKey="range" stroke="var(--text-muted)" />
                          <YAxis stroke="var(--text-muted)" />
                          <Tooltip contentStyle={{ backgroundColor: '#111', border: '1px solid var(--glass-border)' }} />
                          <Bar dataKey="count" fill="var(--primary)" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>
              )}

            </div>
          )}
        </div>

      </div>
    );
  } catch (error) {
    return (
      <div style={{ color: 'red', padding: '2rem', background: '#220000', border: '1px solid red', borderRadius: '8px' }}>
        <h2>React Render Error in SequenceModelingTab</h2>
        <pre>{error.toString()}</pre>
        <pre>{error.stack}</pre>
      </div>
    );
  }
}

// ---------------------------------------------------------------------------
// TAB 3: SIMULATION ENGINE
// ---------------------------------------------------------------------------
function SimulationEngineTab() {
  const { updateCopilotContext } = useCopilot();
  const [target, setTarget] = useState('recovery');
  const [modelType, setModelType] = useState('lstm');
  const [features, setFeatures] = useState({});
  const [featureMeta, setFeatureMeta] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Child selector state
  const [pids, setPids] = useState([]);
  const [selectedPid, setSelectedPid] = useState('');
  const [childEmbeddings, setChildEmbeddings] = useState(null);
  const [childActual, setChildActual] = useState(null);
  const [childInfo, setChildInfo] = useState(null);
  const [pidSearch, setPidSearch] = useState('');

  // Load feature metadata + PID list when target/model changes
  useEffect(() => {
    // Load feature metadata
    axios.get(`/api/temporal/simulate/feature_meta?target=${target}&model_type=${modelType}`)
      .then(res => {
        setFeatureMeta(res.data.features);
        // Initialize features with median values
        const init = {};
        res.data.features.forEach(f => { init[f.name] = f.median; });
        setFeatures(init);
      }).catch(console.error);

    // Load PID list
    axios.get(`/api/temporal/simulate/children?target=${target}&model_type=${modelType}`)
      .then(res => setPids(res.data.pids))
      .catch(console.error);

    // Reset child selection
    setSelectedPid('');
    setChildEmbeddings(null);
    setChildActual(null);
    setChildInfo(null);
    setResult(null);
  }, [target, modelType]);

  // When a child is selected, load their features
  const handleSelectChild = (pid) => {
    if (!pid) {
      setSelectedPid('');
      setChildEmbeddings(null);
      setChildActual(null);
      setChildInfo(null);
      // Reset to median values
      const init = {};
      featureMeta.forEach(f => { init[f.name] = f.median; });
      setFeatures(init);
      return;
    }
    setSelectedPid(pid);
    axios.get(`/api/temporal/simulate/child/${pid}?target=${target}&model_type=${modelType}`)
      .then(res => {
        const data = res.data;
        setFeatures(data.features);
        setChildEmbeddings(data.embeddings);
        setChildActual(data.target_actual);
        setChildInfo({ agedays: data.agedays, numObs: data.num_observations });
      }).catch(console.error);
  };

  const handleSimulate = () => {
    setLoading(true);
    const payload = { 
      target, 
      model_type: modelType, 
      features
    };
    if (selectedPid) {
      payload.pid = selectedPid;
      payload.embeddings = childEmbeddings;
    }
    axios.post('/api/temporal/simulate', payload)
      .then(res => {
        setResult(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  };

  const handleFeatureChange = (key, value) => {
    setFeatures(prev => ({ ...prev, [key]: Number(value) }));
  };

  const trajectoryData = result && result.sensitivity ? result.sensitivity : [];
  const filteredPids = pidSearch ? pids.filter(p => p.includes(pidSearch)).slice(0, 50) : pids.slice(0, 50);

  useEffect(() => {
    let graphData = {};

    if (result) {
        graphData.simulation_result = {
            prediction: result.prediction,
            actual: childActual,
            error: childActual ? parseFloat(Math.abs(result.prediction - childActual).toFixed(4)) : null,
            selected_pid: selectedPid || null,
        };

        // Include the feature slider values the user has set
        graphData.feature_inputs = features;

        // Include sensitivity data if available
        if (result.sensitivity && result.sensitivity.length > 0) {
            graphData.sensitivity_curve = {
                n_points: result.sensitivity.length,
                min_prediction: parseFloat(Math.min(...result.sensitivity.map(s => s.prediction)).toFixed(3)),
                max_prediction: parseFloat(Math.max(...result.sensitivity.map(s => s.prediction)).toFixed(3)),
                prediction_range: parseFloat((Math.max(...result.sensitivity.map(s => s.prediction)) - Math.min(...result.sensitivity.map(s => s.prediction))).toFixed(3)),
            };
        }

        graphData.description = `Simulation for PID ${selectedPid || 'N/A'}. Model: ${modelType}. Predicted: ${result.prediction?.toFixed(3)}. Actual: ${childActual?.toFixed(3) || 'N/A'}. Features: ${Object.entries(features).map(([k,v]) => `${k}=${v}`).join(', ')}`;
    }

    updateCopilotContext({
        tab: 'simulation_engine',
        target,
        model: modelType,
        graphType: 'sensitivity_trajectory',
        metrics: result ? { prediction: result.prediction, actual: childActual } : null,
        graphData,
        graphSummary: graphData.description || 'Simulation idle.'
    });
  }, [target, modelType, result, childActual, selectedPid, features, updateCopilotContext]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '2rem' }}>
      
      {/* Sidebar: Controls */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

        {/* Target & Model Selectors */}
        <div className="glass-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}><Compass size={18} /> Simulation Engine</h3>
          
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Target Outcome</label>
            <select value={target} onChange={(e) => setTarget(e.target.value)} style={{ width: '100%', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}>
              <option value="recovery">Time to Recovery</option>
              <option value="illness">Illness Burden (Next Window)</option>
              <option value="delta_baz">Growth Velocity (ΔBAZ)</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Hybrid Backbone</label>
            <select value={modelType} onChange={(e) => setModelType(e.target.value)} style={{ width: '100%', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}>
              <option value="lstm">Hybrid LSTM</option>
              <option value="tcn">Hybrid TCN</option>
            </select>
          </div>
        </div>

        {/* Child Selector */}
        <div className="glass-card" style={{ padding: '1.5rem' }}>
          <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Activity size={14} /> Patient Selection
          </h4>
          <input 
            type="text" 
            placeholder="Search by PID..." 
            value={pidSearch}
            onChange={(e) => setPidSearch(e.target.value)}
            style={{ width: '100%', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px', marginBottom: '0.5rem', boxSizing: 'border-box' }}
          />
          <select 
            value={selectedPid} 
            onChange={(e) => handleSelectChild(e.target.value)}
            style={{ width: '100%', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', color: '#fff', border: '1px solid var(--glass-border)', borderRadius: '4px' }}
            size={5}
          >
            <option value="">Population Median (Default)</option>
            {filteredPids.map(pid => (
              <option key={pid} value={pid}>{pid}</option>
            ))}
          </select>
          {selectedPid && childInfo && (
            <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: 'rgba(0, 212, 255, 0.05)', borderRadius: '4px', border: '1px solid rgba(0, 212, 255, 0.2)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Selected: <span style={{ color: 'var(--primary)' }}>{selectedPid.substring(0, 16)}...</span></div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Age: {childInfo.agedays ? `${Math.round(childInfo.agedays / 30.44)}mo` : 'N/A'} · 
                Observations: {childInfo.numObs} · 
                Actual: <span style={{ color: '#4ade80', fontWeight: 'bold' }}>{childActual?.toFixed(2)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Dynamic Feature Sliders */}
        <div className="glass-card" style={{ padding: '1.5rem', maxHeight: '350px', overflowY: 'auto' }}>
          <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem' }}>Clinical Signals ({featureMeta.length})</h4>
          
          {featureMeta.map(f => (
            <div key={f.name} style={{ marginBottom: '0.75rem' }}>
              <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.3rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>{f.label}</span>
                <span style={{ fontWeight: 'bold' }}>{f.is_binary ? (features[f.name] >= 0.5 ? 'Yes' : 'No') : (features[f.name] ?? f.median)}</span>
              </label>
              {f.is_binary ? (
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button onClick={() => handleFeatureChange(f.name, 0)} style={{ flex: 1, padding: '0.3rem', background: features[f.name] < 0.5 ? 'var(--primary)' : 'rgba(255,255,255,0.05)', color: features[f.name] < 0.5 ? '#000' : 'var(--text-muted)', border: '1px solid var(--glass-border)', borderRadius: '4px', cursor: 'pointer', fontSize: '0.75rem' }}>No</button>
                  <button onClick={() => handleFeatureChange(f.name, 1)} style={{ flex: 1, padding: '0.3rem', background: features[f.name] >= 0.5 ? 'var(--primary)' : 'rgba(255,255,255,0.05)', color: features[f.name] >= 0.5 ? '#000' : 'var(--text-muted)', border: '1px solid var(--glass-border)', borderRadius: '4px', cursor: 'pointer', fontSize: '0.75rem' }}>Yes</button>
                </div>
              ) : (
                <input type="range" min={f.min} max={f.max} step={f.step} value={features[f.name] ?? f.median} onChange={(e) => handleFeatureChange(f.name, e.target.value)} style={{ width: '100%' }} />
              )}
            </div>
          ))}
        </div>

        {/* Run Button */}
        <button 
          onClick={handleSimulate}
          disabled={loading}
          style={{ width: '100%', padding: '0.8rem', background: 'var(--primary)', color: '#000', border: 'none', borderRadius: '4px', fontWeight: 'bold', cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}>
          {loading ? <><Activity size={16} style={{ animation: 'spin 1s linear infinite' }} /> Simulating...</> : <><Play size={16} /> Run Simulation</>}
        </button>
      </div>

      {/* Main Area: Results */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        
        {result ? (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: selectedPid ? 'repeat(4, 1fr)' : 'repeat(3, 1fr)', gap: '1rem' }}>
              <div className="glass-card" style={{ padding: '1.5rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Predicted {target.replace('_', ' ')}</div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--primary)' }}>{result.prediction.toFixed(2)}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{target === 'recovery' ? 'days' : 'units'}</div>
              </div>
              {selectedPid && childActual !== null && (
                <div className="glass-card" style={{ padding: '1.5rem', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Actual (Ground Truth)</div>
                  <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#4ade80' }}>{childActual.toFixed(2)}</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{target === 'recovery' ? 'days' : 'units'}</div>
                </div>
              )}
              <div className="glass-card" style={{ padding: '1.5rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Population Median</div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#fff' }}>{result.pop_median.toFixed(2)}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{target === 'recovery' ? 'days' : 'units'}</div>
              </div>
              <div className="glass-card" style={{ padding: '1.5rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '4px' }}>
                  <AlertTriangle size={14} /> {selectedPid ? 'Prediction Error' : 'Deviation'}
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: (selectedPid ? Math.abs(result.prediction - childActual) : Math.abs(result.prediction - result.pop_median)) < 5 ? '#4ade80' : '#fbbf24' }}>
                  {selectedPid && childActual !== null 
                    ? Math.abs(result.prediction - childActual).toFixed(2) 
                    : Math.abs(result.prediction - result.pop_median).toFixed(2)}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{selectedPid ? 'abs error' : (result.prediction > result.pop_median ? 'above' : 'below')}</div>
              </div>
            </div>

            <div className="glass-card" style={{ padding: '2rem', flex: 1 }}>
              <h3 style={{ marginBottom: '1.5rem', textTransform: 'capitalize' }}>Sensitivity: {result.sensitivity_feature.replace('v1_', '').replace(/_/g, ' ')}</h3>
              <div style={{ height: '300px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trajectoryData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="feature_value" type="number" domain={['auto', 'auto']} label={{ value: result.sensitivity_feature.replace('v1_', '').replace(/_/g, ' '), position: 'insideBottom', offset: -10, fill: 'var(--text-muted)' }} stroke="var(--text-muted)" />
                    <YAxis label={{ value: 'Prediction', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)' }} stroke="var(--text-muted)" domain={['auto', 'auto']} />
                    <Tooltip contentStyle={{ backgroundColor: '#111', border: '1px solid var(--glass-border)' }} />
                    <ReferenceLine x={features[result.sensitivity_feature]} stroke="#4ade80" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: selectedPid ? 'This Child' : 'Current', fill: '#4ade80' }} />
                    {selectedPid && childActual !== null && (
                      <ReferenceLine y={childActual} stroke="#fbbf24" strokeDasharray="3 3" label={{ position: 'insideTopRight', value: 'Actual', fill: '#fbbf24' }} />
                    )}
                    <Line type="monotone" name="Predicted Outcome" dataKey="prediction" stroke="var(--primary)" strokeWidth={3} dot={{ r: 5, fill: 'var(--primary)' }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        ) : (
          <div className="glass-card" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', borderStyle: 'dashed' }}>
            <Compass size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
            <p>{selectedPid ? `Patient ${selectedPid.substring(0, 12)}... loaded. Run simulation to predict.` : 'Select a patient or adjust parameters, then run simulation.'}</p>
          </div>
        )}

      </div>
    </div>
  );
}
