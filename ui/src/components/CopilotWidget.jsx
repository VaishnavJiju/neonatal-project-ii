import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { MessageSquare, X, Settings, Send, Bot, User, Key, ChevronDown, Activity, Lightbulb, GripVertical, ToggleLeft, ToggleRight } from 'lucide-react';
import { useCopilot } from '../context/CopilotContext';

const CopilotWidget = () => {
    const { 
        context, chatHistory, addMessage, clearHistory, 
        suggestions, settings, setSettings, 
        isWidgetOpen, setIsWidgetOpen,
        isTyping, setIsTyping
    } = useCopilot();

    const [input, setInput] = useState('');
    const [showSettings, setShowSettings] = useState(false);
    const [showDebug, setShowDebug] = useState(false);
    const [streamingText, setStreamingText] = useState('');
    const [conciseMode, setConciseMode] = useState(false);
    const [widgetSize, setWidgetSize] = useState({ width: 400, height: 600 });
    const messagesEndRef = useRef(null);
    const isResizing = useRef(false);
    const startPos = useRef({ x: 0, y: 0 });
    const startSize = useRef({ width: 400, height: 600 });

    // Resize handlers
    const handleResizeStart = useCallback((e) => {
        e.preventDefault();
        isResizing.current = true;
        startPos.current = { x: e.clientX, y: e.clientY };
        startSize.current = { ...widgetSize };
        document.body.style.cursor = 'nwse-resize';
        document.body.style.userSelect = 'none';
    }, [widgetSize]);

    useEffect(() => {
        const handleMouseMove = (e) => {
            if (!isResizing.current) return;
            // Since anchored bottom-right, dragging left increases width, dragging up increases height
            const dw = startPos.current.x - e.clientX;
            const dh = startPos.current.y - e.clientY;
            setWidgetSize({
                width: Math.max(320, Math.min(800, startSize.current.width + dw)),
                height: Math.max(400, Math.min(900, startSize.current.height + dh))
            });
        };
        const handleMouseUp = () => {
            if (isResizing.current) {
                isResizing.current = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }
        };
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, []);

    // Auto-scroll to bottom of chat
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [chatHistory, isTyping, suggestions, streamingText]);

    const handleSend = async (queryOverride = null) => {
        const query = queryOverride || input;
        if (!query.trim() || isTyping) return;
        
        if (!settings.apiKey) {
            setShowSettings(true);
            return;
        }

        setInput('');
        addMessage('user', query);
        setIsTyping(true);
        setStreamingText('');

        try {
            const res = await axios.post('/api/copilot/query', {
                query: query,
                context: context,
                api_key: settings.apiKey,
                model_name: settings.model,
                temperature: settings.temperature,
                history: chatHistory,
                concise_mode: conciseMode
            });

            const fullText = res.data.response;
            // Animate the response word-by-word
            const words = fullText.split(' ');
            let currentIndex = 0;
            
            const typeInterval = setInterval(() => {
                currentIndex++;
                const partial = words.slice(0, currentIndex).join(' ');
                setStreamingText(partial);
                
                if (currentIndex >= words.length) {
                    clearInterval(typeInterval);
                    setStreamingText('');
                    addMessage('assistant', fullText);
                    setIsTyping(false);
                }
            }, 30);
        } catch (err) {
            console.error("Copilot Error:", err);
            setStreamingText('');
            addMessage('assistant', `Error: ${err.response?.data?.detail || err.message}`);
            setIsTyping(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    if (!isWidgetOpen) {
        return (
            <button 
                onClick={() => setIsWidgetOpen(true)}
                style={{
                    position: 'fixed', bottom: '2rem', right: '2rem', zIndex: 9999,
                    width: '60px', height: '60px', borderRadius: '50%',
                    background: 'var(--primary)', color: '#000', border: 'none',
                    boxShadow: '0 4px 20px rgba(0, 212, 255, 0.4)',
                    display: 'flex', justifyContent: 'center', alignItems: 'center',
                    cursor: 'pointer', transition: 'transform 0.2s',
                    animation: 'bounceIn 0.5s ease'
                }}
                onMouseOver={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
                onMouseOut={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
                <MessageSquare size={28} />
            </button>
        );
    }

    return (
        <div className="glass-card" style={{
            position: 'fixed', bottom: '2rem', right: '2rem', zIndex: 9999,
            width: `${widgetSize.width}px`, height: `${widgetSize.height}px`, 
            display: 'flex', flexDirection: 'column',
            boxShadow: '0 10px 40px rgba(0,0,0,0.5)',
            animation: 'slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
            overflow: 'hidden'
        }}>
            {/* Resize Handle (top-left corner) */}
            <div 
                onMouseDown={handleResizeStart}
                style={{
                    position: 'absolute', top: 0, left: 0, width: '20px', height: '20px',
                    cursor: 'nwse-resize', zIndex: 10,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    opacity: 0.4, transition: 'opacity 0.2s'
                }}
                onMouseOver={(e) => e.currentTarget.style.opacity = '1'}
                onMouseOut={(e) => e.currentTarget.style.opacity = '0.4'}
            >
                <svg width="10" height="10" viewBox="0 0 10 10">
                    <line x1="0" y1="10" x2="10" y2="0" stroke="var(--primary)" strokeWidth="1.5" />
                    <line x1="0" y1="6" x2="6" y2="0" stroke="var(--primary)" strokeWidth="1.5" />
                    <line x1="0" y1="2" x2="2" y2="0" stroke="var(--primary)" strokeWidth="1.5" />
                </svg>
            </div>
            {/* Header */}
            <div style={{ 
                padding: '1rem', borderBottom: '1px solid var(--glass-border)', 
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                background: 'rgba(0,0,0,0.3)'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Bot size={20} color="var(--primary)" />
                    <h3 style={{ margin: 0, fontSize: '1.1rem' }}>Nexus Copilot</h3>
                    <button 
                        onClick={() => setConciseMode(!conciseMode)} 
                        title={conciseMode ? 'Concise Mode ON' : 'Concise Mode OFF'}
                        style={{ 
                            display: 'flex', alignItems: 'center', gap: '4px',
                            background: conciseMode ? 'rgba(0, 212, 255, 0.15)' : 'rgba(255,255,255,0.05)', 
                            border: `1px solid ${conciseMode ? 'var(--primary)' : 'var(--glass-border)'}`, 
                            borderRadius: '12px', padding: '2px 8px', cursor: 'pointer', transition: 'all 0.2s',
                            fontSize: '0.65rem', color: conciseMode ? 'var(--primary)' : 'var(--text-muted)', fontWeight: 600
                        }}
                    >
                        {conciseMode ? <ToggleRight size={12} /> : <ToggleLeft size={12} />}
                        Concise
                    </button>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button onClick={() => setShowDebug(!showDebug)} style={{ background: 'transparent', border: 'none', color: showDebug ? 'var(--primary)' : 'var(--text-muted)', cursor: 'pointer' }} title="Toggle Debug Context"><Activity size={16} /></button>
                    <button onClick={() => setShowSettings(!showSettings)} style={{ background: 'transparent', border: 'none', color: showSettings ? '#fff' : 'var(--text-muted)', cursor: 'pointer' }}><Settings size={16} /></button>
                    <button onClick={() => setIsWidgetOpen(false)} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}><X size={16} /></button>
                </div>
            </div>

            {/* Debug Context Banner */}
            {showDebug && (
                <div style={{ padding: '0.5rem', background: 'rgba(0,0,0,0.5)', borderBottom: '1px solid var(--glass-border)', fontSize: '0.7rem', color: 'var(--primary)', fontFamily: 'monospace', maxHeight: '100px', overflowY: 'auto' }}>
                    <pre style={{ margin: 0 }}>{JSON.stringify(context, null, 2)}</pre>
                </div>
            )}

            {/* Settings Panel */}
            {showSettings ? (
                <div style={{ padding: '1.5rem', flex: 1, overflowY: 'auto', background: 'rgba(0,0,0,0.2)' }}>
                    <h4 style={{ marginBottom: '1.5rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem' }}>Copilot Settings</h4>
                    
                    <div style={{ marginBottom: '1.5rem' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}><Key size={14} /> Groq API Key</label>
                        <input 
                            type="password" 
                            value={settings.apiKey} 
                            onChange={(e) => setSettings({...settings, apiKey: e.target.value})}
                            placeholder="gsk_..."
                            style={{ width: '100%', padding: '0.8rem', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--glass-border)', borderRadius: '4px', color: '#fff', boxSizing: 'border-box' }}
                        />
                        {!settings.apiKey && <span style={{ fontSize: '0.7rem', color: '#ef4444' }}>Required for Copilot to function.</span>}
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                            <span>Temperature</span>
                            <span>{settings.temperature}</span>
                        </label>
                        <input 
                            type="range" min="0" max="1" step="0.1" 
                            value={settings.temperature} 
                            onChange={(e) => setSettings({...settings, temperature: parseFloat(e.target.value)})}
                            style={{ width: '100%' }}
                        />
                    </div>

                    <button 
                        onClick={() => clearHistory()}
                        style={{ width: '100%', padding: '0.8rem', background: 'transparent', border: '1px solid rgba(239, 68, 68, 0.5)', color: '#ef4444', borderRadius: '4px', cursor: 'pointer', marginBottom: '1rem' }}>
                        Clear Conversation History
                    </button>

                    <button 
                        onClick={() => setShowSettings(false)}
                        style={{ width: '100%', padding: '0.8rem', background: 'var(--primary)', color: '#000', border: 'none', borderRadius: '4px', fontWeight: 'bold', cursor: 'pointer' }}>
                        Done
                    </button>
                </div>
            ) : (
                <>
                    {/* Chat Area */}
                    <div style={{ flex: 1, padding: '1rem', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {chatHistory.map((msg, i) => (
                            <div key={i} style={{ 
                                display: 'flex', gap: '8px', 
                                alignItems: 'flex-start',
                                flexDirection: msg.role === 'user' ? 'row-reverse' : 'row'
                            }}>
                                <div style={{ 
                                    width: '28px', height: '28px', borderRadius: '50%', 
                                    background: msg.role === 'user' ? 'rgba(255,255,255,0.1)' : 'rgba(0, 212, 255, 0.1)',
                                    display: 'flex', justifyContent: 'center', alignItems: 'center',
                                    flexShrink: 0, marginTop: '10px'
                                }}>
                                    {msg.role === 'user' ? <User size={14} color="#fff" /> : <Bot size={14} color="var(--primary)" />}
                                </div>
                                <div style={{ 
                                    background: msg.role === 'user' ? 'rgba(255,255,255,0.05)' : 'transparent',
                                    padding: '0.75rem', borderRadius: '8px',
                                    border: msg.role === 'user' ? '1px solid var(--glass-border)' : 'none',
                                    fontSize: '0.85rem', lineHeight: '1.5',
                                    maxWidth: '85%',
                                    color: msg.role === 'user' ? '#fff' : 'var(--text-main)',
                                }}
                                    className={msg.role === 'assistant' ? 'copilot-markdown' : ''}
                                >
                                    {msg.role === 'assistant' ? (
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                                    ) : msg.content}
                                </div>
                            </div>
                        ))}
                        {isTyping && (
                            <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
                                <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'rgba(0, 212, 255, 0.1)', display: 'flex', justifyContent: 'center', alignItems: 'center', flexShrink: 0, marginTop: '10px' }}>
                                    <Bot size={14} color="var(--primary)" />
                                </div>
                                {streamingText ? (
                                    <div style={{ 
                                        fontSize: '0.85rem', lineHeight: '1.5', 
                                        color: 'var(--text-main)', maxWidth: '85%' 
                                    }}
                                        className="copilot-markdown"
                                    >
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamingText}</ReactMarkdown>
                                        <span style={{ 
                                            display: 'inline-block', width: '2px', height: '1em', 
                                            background: 'var(--primary)', marginLeft: '2px', 
                                            animation: 'blink 0.8s step-end infinite', verticalAlign: 'text-bottom' 
                                        }} />
                                    </div>
                                ) : (
                                    <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '6px', paddingTop: '6px' }}>
                                        <span style={{ animation: 'pulse 1.5s infinite' }}>Thinking</span>
                                        <span style={{ display: 'inline-flex', gap: '3px' }}>
                                            <span style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--primary)', animation: 'bounce 1.4s infinite ease-in-out', animationDelay: '0s' }} />
                                            <span style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--primary)', animation: 'bounce 1.4s infinite ease-in-out', animationDelay: '0.2s' }} />
                                            <span style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--primary)', animation: 'bounce 1.4s infinite ease-in-out', animationDelay: '0.4s' }} />
                                        </span>
                                    </div>
                                )}
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Suggestions Area */}
                    {suggestions.length > 0 && !isTyping && (
                        <div style={{ padding: '0.5rem 1rem', display: 'flex', gap: '8px', overflowX: 'auto', borderTop: '1px solid var(--glass-border)' }} className="hide-scrollbar">
                            <Lightbulb size={14} color="var(--primary)" style={{ flexShrink: 0, marginTop: '6px' }} />
                            {suggestions.map((s, i) => (
                                <button key={i} onClick={() => handleSend(s)} style={{ 
                                    whiteSpace: 'nowrap', padding: '0.4rem 0.8rem', background: 'rgba(0, 212, 255, 0.05)', 
                                    border: '1px solid rgba(0, 212, 255, 0.2)', borderRadius: '16px', color: 'var(--primary)', 
                                    fontSize: '0.75rem', cursor: 'pointer', transition: 'all 0.2s'
                                }} onMouseOver={(e) => e.target.style.background = 'rgba(0, 212, 255, 0.15)'} onMouseOut={(e) => e.target.style.background = 'rgba(0, 212, 255, 0.05)'}>
                                    {s}
                                </button>
                            ))}
                        </div>
                    )}

                    {/* Input Area */}
                    <div style={{ padding: '1rem', borderTop: '1px solid var(--glass-border)', background: 'rgba(0,0,0,0.3)' }}>
                        {!settings.apiKey ? (
                             <button onClick={() => setShowSettings(true)} style={{ width: '100%', padding: '0.8rem', background: 'transparent', border: '1px dashed var(--glass-border)', color: 'var(--text-muted)', borderRadius: '4px', cursor: 'pointer' }}>
                                Please configure API Key first
                            </button>
                        ) : (
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <textarea 
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    placeholder="Ask about this graph..."
                                    style={{ 
                                        flex: 1, padding: '0.8rem', background: 'rgba(255,255,255,0.02)', 
                                        border: '1px solid var(--glass-border)', borderRadius: '8px', 
                                        color: '#fff', resize: 'none', height: '40px', minHeight: '40px', 
                                        maxHeight: '120px', overflow: 'auto',
                                        fontFamily: 'inherit', fontSize: '0.9rem' 
                                    }}
                                />
                                <button 
                                    onClick={() => handleSend()}
                                    disabled={!input.trim() || isTyping}
                                    style={{ 
                                        width: '45px', height: '45px', borderRadius: '8px', 
                                        background: input.trim() && !isTyping ? 'var(--primary)' : 'rgba(255,255,255,0.05)', 
                                        color: input.trim() && !isTyping ? '#000' : 'var(--text-muted)', 
                                        border: 'none', cursor: input.trim() && !isTyping ? 'pointer' : 'not-allowed',
                                        display: 'flex', justifyContent: 'center', alignItems: 'center'
                                    }}
                                >
                                    <Send size={18} />
                                </button>
                            </div>
                        )}
                    </div>
                </>
            )}
            
            <style jsx>{`
                @keyframes slideUp {
                    from { opacity: 0; transform: translateY(20px) scale(0.95); }
                    to { opacity: 1; transform: translateY(0) scale(1); }
                }
                @keyframes bounceIn {
                    0% { opacity: 0; transform: scale(0.5); }
                    50% { opacity: 1; transform: scale(1.2); }
                    100% { opacity: 1; transform: scale(1); }
                }
                .hide-scrollbar::-webkit-scrollbar {
                    display: none;
                }
                .hide-scrollbar {
                    -ms-overflow-style: none;
                    scrollbar-width: none;
                }
                @keyframes blink {
                    50% { opacity: 0; }
                }
                @keyframes bounce {
                    0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
                    40% { transform: scale(1); opacity: 1; }
                }
            `}</style>
        </div>
    );
};

export default CopilotWidget;
