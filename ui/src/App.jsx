import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import axios from 'axios';
import { Database, FlaskConical, Stethoscope, Scale, BookOpen, Download, Search, Activity, BarChart2, Layers } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { NexusProvider, useNexus } from './NexusContext';
import { CopilotProvider, useCopilot } from './context/CopilotContext';
import Laboratory from './Laboratory';
import DiagnosticStudio from './DiagnosticStudio';
import TemporalLab from './TemporalLab';
import CopilotWidget from './components/CopilotWidget';


// --- LAYOUT (Elegant Top Nav + Main Area) ---
const Layout = ({ children }) => {
  const location = useLocation();
  const { modelVersion, setModelVersion } = useNexus();
  
  const navItems = [
    { path: '/', label: 'Explorer', icon: <Database size={16} /> },
    { path: '/laboratory', label: 'Laboratory', icon: <FlaskConical size={16} /> },
    { path: '/diagnostic', label: 'Diagnostic', icon: <Stethoscope size={16} /> },
    { path: '/temporal', label: 'Temporal Intelligence', icon: <Activity size={16} /> }
  ];

  return (
    <div className="app-container" style={{ gridTemplateColumns: '1fr' }}>
      <div className="main-area">
        <nav className="top-nav" style={{ position: 'relative', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          
        {/* Left Anchor: Brand */}
        <div style={{ position: 'absolute', left: '2rem', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ background: 'var(--primary)', padding: '8px', borderRadius: '8px', display: 'flex' }}>
                <Layers size={22} color="#000" strokeWidth={2.5} />
            </div>
            <div>
                <h1 style={{ margin: 0, fontSize: '1.2rem', letterSpacing: '2px', fontWeight: '700', color: '#fff' }}>MAL-ED</h1>
                <div style={{ fontSize: '0.7rem', color: 'var(--primary)', letterSpacing: '4px' }}>NEXUS AI</div>
            </div>
        </div>
          
        {/* Center: Nav Tabs */}
        <div className="nav-links" style={{ display: 'flex', gap: '2rem' }}>
          {navItems.map((item) => (
            <Link 
              key={item.path} 
              to={item.path} 
              className={`nav-item ${location.pathname === item.path ? 'active' : ''}`}
            >
              {item.icon}
              <span>{item.label}</span>
            </Link>
          ))}
        </div>

        {/* Right Anchor: User Profile */}
        <div style={{ position: 'absolute', right: '2rem', display: 'flex', alignItems: 'center', gap: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ textAlign: 'right' }}>
                 <p style={{fontSize: '0.8rem', color: '#fff', margin: 0}}>Dr. Supervisor</p>
                 <p style={{fontSize: '0.65rem', color: 'var(--text-muted)', margin: 0}}>Admin Mode</p>
              </div>
              <div style={{width: '32px', height: '32px', borderRadius: '50%', border: '1px solid var(--glass-border)', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
                  <span style={{fontSize: '0.8rem', color: 'var(--primary)'}}>DS</span>
              </div>
            </div>
        </div>
        
      </nav>

      <main className="main-content">
        {children}
      </main>
      </div>
    </div>
  );
};

// --- MODULE: INTAKE & EXPLORER ---
const IntakeExplorer = () => {
  const [activeTab, setActiveTab] = useState('explorer');
  const { updateCopilotContext } = useCopilot();

  useEffect(() => {
    updateCopilotContext({
      tab: 'explorer',
      subTab: activeTab === 'explorer' ? 'data_preview' : 'codebook',
      graphType: activeTab === 'explorer' ? 'dataset_overview' : 'clinical_codebook',
      graphSummary: activeTab === 'explorer' 
        ? 'User is viewing the Data Preview & Metrics tab with dataset statistics, sample rows, and univariate distribution charts.'
        : 'User is viewing the Clinical Codebook — a searchable dictionary of all features, their categories, and definitions.'
    });
  }, [activeTab]);

  return (
    <div style={{ animation: 'fadeIn 0.4s ease' }}>
      <header style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.2rem' }}>Intake & Explorer</h1>
        <p style={{ color: 'var(--text-muted)' }}>Global overview of patient records and visual telemetry.</p>
      </header>

      <div className="sub-tabs">
        <button className={`sub-tab ${activeTab === 'explorer' ? 'active' : ''}`} onClick={() => setActiveTab('explorer')}>Data Preview & Metrics</button>
        <button className={`sub-tab ${activeTab === 'dictionary' ? 'active' : ''}`} onClick={() => setActiveTab('dictionary')}>Clinical Codebook</button>
      </div>

      {activeTab === 'explorer' ? <DataExplorerTab /> : <DataDictionaryTab />}
    </div>
  );
};

const DataExplorerTab = () => {
  const [stats, setStats] = useState({ rows: 0, columns: 0, unique: 0, comp: 0, colsList: [], loading: true });
  const [sample, setSample] = useState({ columns: [], data: [], loading: true });
  
  // Univariate State
  const [selectedCol, setSelectedCol] = useState("");
  const [chartData, setChartData] = useState([]);
  const [chartLoading, setChartLoading] = useState(false);
  const { updateCopilotContext } = useCopilot();

  // Push dataset stats to copilot when data loads
  useEffect(() => {
    if (!stats.loading) {
      updateCopilotContext({
        tab: 'explorer',
        subTab: 'data_preview',
        graphType: 'univariate_distribution',
        target: selectedCol || 'none',
        graphData: {
          dataset_stats: {
            total_rows: stats.rows,
            total_columns: stats.columns,
            unique_children: stats.unique,
            completeness: stats.comp
          },
          selected_column: selectedCol,
          description: selectedCol 
            ? `Viewing univariate distribution for '${selectedCol}'. Dataset has ${stats.rows.toLocaleString()} rows, ${stats.columns} features, ${stats.unique.toLocaleString()} unique children, ${stats.comp}% completeness.`
            : `Dataset overview: ${stats.rows.toLocaleString()} rows, ${stats.columns} features, ${stats.unique.toLocaleString()} unique children, ${stats.comp}% completeness.`
        },
        graphSummary: selectedCol 
          ? `User is viewing the distribution of '${selectedCol}' in the Data Explorer.`
          : `User is viewing dataset overview metrics.`
      });
    }
  }, [stats.loading, selectedCol]);

  useEffect(() => {
    // Stats Fetch
    axios.get('/api/dataset/overview')
      .then(res => {
        setStats({ 
            rows: res.data.rows, 
            columns: res.data.columns, 
            unique: res.data.unique_children,
            comp: res.data.completeness,
            colsList: res.data.columns_list,
            loading: false 
        });
        if(res.data.columns_list.length > 0) {
            setSelectedCol("BMI-for-age z-score"); // Default start
        }
      })
      .catch(err => {
        setStats(prev => ({ ...prev, loading: false }));
      });

    // Sample Fetch
    axios.get('/api/dataset/sample')
      .then(res => setSample({ columns: res.data.columns, data: res.data.data, loading: false }))
      .catch(err => setSample({ columns: [], data: [], loading: false }));
  }, []);

  useEffect(() => {
      if(selectedCol) {
          setChartLoading(true);
          axios.get(`/api/dataset/univariate?column=${encodeURIComponent(selectedCol)}`)
              .then(res => {
                  setChartData(res.data.data);
                  setChartLoading(false);
              })
              .catch(err => {
                  setChartLoading(false);
              });
      }
  }, [selectedCol]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* 4 Elegant Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
        <div className="glass-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Total Observations</h3>
          <div style={{ fontSize: '2rem', fontWeight: 600, color: '#fff', marginTop: '10px' }}>
            {stats.loading ? '...' : stats.rows.toLocaleString()}
          </div>
        </div>
        <div className="glass-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Clinical Features</h3>
          <div style={{ fontSize: '2rem', fontWeight: 600, color: '#fff', marginTop: '10px' }}>
            {stats.loading ? '...' : stats.columns}
          </div>
        </div>
        <div className="glass-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Unique Children</h3>
          <div style={{ fontSize: '2rem', fontWeight: 600, color: 'var(--primary)', marginTop: '10px' }}>
            {stats.loading ? '...' : stats.unique.toLocaleString()}
          </div>
        </div>
        <div className="glass-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Data Completeness</h3>
          <div style={{ fontSize: '2rem', fontWeight: 600, color: '#fff', marginTop: '10px' }}>
            {stats.loading ? '...' : `${stats.comp}%`}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2rem' }}>
          {/* Data Table */}
          <div className="glass-card">
            <div style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--glass-border)' }}>
              <h3 style={{ color: '#fff', fontSize: '1.1rem' }}>Live Tensor Feed (Sample)</h3>
            </div>
            {sample.loading ? (
                <div style={{ padding: '4rem', textAlign: 'center' }}>
                    <Activity size={30} className="spinner" style={{ color: 'var(--text-muted)', animation: 'spin 2s linear infinite' }} />
                </div>
            ) : (
                <div className="data-table-container" style={{ maxHeight: '300px' }}>
                  <table className="data-table">
                    <thead>
                      <tr>{sample.columns.map(col => <th key={col}>{col}</th>)}</tr>
                    </thead>
                    <tbody>
                      {sample.data.map((row, idx) => (
                        <tr key={idx}>
                          {sample.columns.map(col => {
                             const val = row[col];
                             const displayVal = typeof val === 'number' && !Number.isInteger(val) ? val.toFixed(3) : val;
                             return <td key={col}>{displayVal}</td>;
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
            )}
          </div>

          {/* Univariate Chart */}
          <div className="glass-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ color: '#fff', fontSize: '1.1rem', marginBottom: '1rem' }}>Distribution Analysis</h3>
            <select 
                value={selectedCol} 
                onChange={(e) => setSelectedCol(e.target.value)}
                style={{ width: '100%', padding: '8px', borderRadius: '6px', background: 'var(--nav-bg)', color: '#fff', border: '1px solid var(--glass-border)', outline: 'none', marginBottom: '1rem' }}
            >
                {stats.colsList.map(c => <option key={c} value={c}>{c}</option>)}
            </select>

            <div style={{ flex: 1, minHeight: '200px' }}>
                {chartLoading ? (
                    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <BarChart2 size={24} style={{ color: 'var(--primary)', opacity: 0.5, animation: 'pulse 1.5s infinite' }} />
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData}>
                        <XAxis dataKey={chartData[0]?.bin ? "bin" : "name"} stroke="var(--text-muted)" fontSize={10} angle={-45} textAnchor="end" height={60} />
                        <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{background: 'var(--nav-bg)', border: '1px solid var(--glass-border)', borderRadius: '8px'}} />
                        <Bar dataKey="count" fill="var(--primary)" opacity={0.8} radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                )}
            </div>
          </div>
      </div>
    </div>
  );
};

const DataDictionaryTab = () => {
    const [dict, setDict] = useState([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState("");
    const { updateCopilotContext } = useCopilot();

    useEffect(() => {
        axios.get('/api/dataset/dictionary')
            .then(res => {
                setDict(res.data.dictionary);
                setLoading(false);
            })
            .catch(err => setLoading(false));
    }, []);

    // Push codebook context to copilot
    useEffect(() => {
        if (!loading && dict.length > 0) {
            const categories = [...new Set(dict.map(d => d.Category).filter(Boolean))];
            updateCopilotContext({
                tab: 'explorer',
                subTab: 'codebook',
                graphType: 'clinical_codebook',
                graphData: {
                    total_features: dict.length,
                    categories: categories.slice(0, 15),
                    search_term: search || null,
                    filtered_count: search ? dict.filter(item => 
                        (item.Feature?.toLowerCase() || "").includes(search.toLowerCase()) ||
                        (item.Definition?.toLowerCase() || "").includes(search.toLowerCase()) ||
                        (item.Category?.toLowerCase() || "").includes(search.toLowerCase())
                    ).length : dict.length,
                    description: search 
                        ? `User is searching the codebook for '${search}'. ${dict.length} total features across ${categories.length} categories.`
                        : `Clinical codebook with ${dict.length} features across ${categories.length} categories.`
                },
                graphSummary: search 
                    ? `User is searching the Clinical Codebook for '${search}'.`
                    : `User is browsing the Clinical Codebook (${dict.length} features).`
            });
        }
    }, [loading, search]);

    const filtered = dict.filter(item => 
        (item.Feature?.toLowerCase() || "").includes(search.toLowerCase()) ||
        (item.Definition?.toLowerCase() || "").includes(search.toLowerCase()) ||
        (item.Category?.toLowerCase() || "").includes(search.toLowerCase())
    );

    return (
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', height: '600px' }}>
            <div style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--glass-border)', background: 'rgba(0,0,0,0.2)' }}>
                <h3 style={{ color: '#fff', fontSize: '1.2rem', margin: 0 }}>Clinical Codebook</h3>
                <div style={{ position: 'relative', width: '300px' }}>
                    <Search size={16} style={{ position: 'absolute', left: '10px', top: '10px', color: 'var(--text-muted)' }} />
                    <input 
                        type="text" 
                        placeholder="Search features, domains, definitions..." 
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        style={{ width: '100%', padding: '8px 10px 8px 34px', borderRadius: '20px', background: 'rgba(255,255,255,0.05)', color: '#fff', border: '1px solid var(--glass-border)', outline: 'none' }}
                    />
                </div>
            </div>

            {loading ? (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Activity className="spinner" size={30} style={{ color: 'var(--text-muted)' }} />
                </div>
            ) : (
                <div className="data-table-container" style={{ flex: 1, maxHeight: 'none' }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                          <th style={{ width: '25%' }}>Research Variable</th>
                          <th style={{ width: '20%' }}>Domain</th>
                          <th style={{ width: '55%' }}>Clinical Definition</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.map((item, idx) => (
                        <tr key={idx}>
                          <td style={{ color: 'var(--primary)', fontWeight: 500 }}>{item.Feature}</td>
                          <td><span style={{ padding: '2px 8px', background: 'rgba(255,255,255,0.1)', borderRadius: '12px', fontSize: '0.75rem' }}>{item.Category}</span></td>
                          <td style={{ whiteSpace: 'normal', lineHeight: 1.5, paddingRight: '2rem' }}>{item.Definition}</td>
                        </tr>
                      ))}
                      {filtered.length === 0 && (
                          <tr><td colSpan="3" style={{ textAlign: 'center', padding: '2rem' }}>No variables matched your search.</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
            )}
        </div>
    );
};

// --- PLACEHOLDERS ---
const PlaceholderView = ({ title }) => (
  <div style={{ animation: 'fadeIn 0.4s ease' }}>
    <header style={{ marginBottom: '2rem' }}>
      <h1 style={{ fontSize: '2.5rem', marginBottom: '0.2rem' }}>{title}</h1>
      <p style={{ color: 'var(--text-muted)' }}>This module is scheduled for implementation.</p>
    </header>
    <div className="glass-card" style={{ padding: '2rem', textAlign: 'center', minHeight: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', borderStyle: 'dashed' }}>
       <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', letterSpacing: '2px', textTransform: 'uppercase' }}>Under Construction</p>
    </div>
  </div>
);

function App() {
  return (
    <NexusProvider>
      <CopilotProvider>
        <Router>
          <Layout>
            <Routes>
              <Route path="/" element={<IntakeExplorer />} />
              <Route path="/laboratory" element={<Laboratory />} />
              <Route path="/diagnostic" element={<DiagnosticStudio />} />
              <Route path="/temporal" element={<TemporalLab />} />
              <Route path="/copilot" element={<PlaceholderView title="Intelligence Center" />} />
            </Routes>
          </Layout>
        </Router>
        <CopilotWidget />
      </CopilotProvider>
    </NexusProvider>
  );
}

export default App;
