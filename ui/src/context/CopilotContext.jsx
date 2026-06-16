import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';

const CopilotContext = createContext();

export const CopilotProvider = ({ children }) => {
    // Current UI context being watched
    const [context, setContext] = useState({
        tab: '',
        target: '',
        model: '',
        graphType: '',
        metrics: null,
        graphData: null,
        graphSummary: ''
    });

    // Chat history
    const [chatHistory, setChatHistory] = useState([
        { role: 'assistant', content: "Hi! I'm your Copilot. I'm watching your dashboard context. How can I help?" }
    ]);

    // Dynamic suggestions based on context
    const [suggestions, setSuggestions] = useState([]);

    // Copilot Settings
    const [settings, setSettings] = useState(() => {
        const saved = localStorage.getItem('copilot_settings');
        let parsed = saved ? JSON.parse(saved) : null;
        if (parsed && (parsed.model === 'llama3-8b-8192' || parsed.model === 'gpt0ss 120b' || parsed.model === 'llama-3.1-8b-instant')) {
            parsed.model = 'openai/gpt-oss-120b';
        }
        return parsed || { apiKey: '', model: 'openai/gpt-oss-120b', temperature: 0.5 };
    });

    const [isWidgetOpen, setIsWidgetOpen] = useState(false);
    const [isTyping, setIsTyping] = useState(false);

    // Save settings to local storage when they change
    useEffect(() => {
        localStorage.setItem('copilot_settings', JSON.stringify(settings));
    }, [settings]);

    // Generator for suggestions based on current context
    const generateSuggestions = useCallback((ctx) => {
        let newSuggestions = [];

        // Base suggestion — always available
        newSuggestions.push("Explain this view");

        // Explorer tab
        if (ctx.tab === 'explorer') {
            if (ctx.subTab === 'data_preview') {
                newSuggestions.push("Summarize this dataset");
                if (ctx.target && ctx.target !== 'none') {
                    newSuggestions.push(`What does '${ctx.target}' tell us clinically?`);
                }
            } else if (ctx.subTab === 'codebook') {
                newSuggestions.push("What categories of features are available?");
                newSuggestions.push("Which features are most clinically relevant?");
            }
        }

        // Laboratory tab (Feature Discovery + Model Arena)
        if (ctx.tab === 'feature_discovery') {
            newSuggestions.push("Which features matter most for this target?");
            newSuggestions.push("Are there any surprising features here?");
        }
        if (ctx.tab === 'model_arena') {
            newSuggestions.push("Which model performs best and why?");
            newSuggestions.push("Should I trust these metrics?");
        }

        // Diagnostic Studio tab
        if (ctx.tab === 'diagnostic_studio' || ctx.tab === 'diagnostic') {
            if (ctx.graphType?.toLowerCase().includes('prediction') || ctx.graphType?.toLowerCase().includes('scatter')) {
                newSuggestions.push("Is this model overfitting?");
                newSuggestions.push("Why are predictions deviating at higher values?");
            } else if (ctx.graphType?.toLowerCase().includes('residual')) {
                newSuggestions.push("What do these residuals mean?");
                newSuggestions.push("Is the model biased?");
            } else if (ctx.graphType?.toLowerCase().includes('force')) {
                newSuggestions.push("Explain this SHAP force plot");
            } else if (ctx.graphType?.toLowerCase().includes('importance') || ctx.graphType?.toLowerCase().includes('beeswarm')) {
                newSuggestions.push("What are the most critical features?");
                newSuggestions.push("Why does this feature have such high importance?");
            } else if (ctx.graphType?.toLowerCase().includes('dependence')) {
                newSuggestions.push("What does this dependence pattern mean?");
            } else if (ctx.graphType?.toLowerCase().includes('threshold') || ctx.graphType?.toLowerCase().includes('roc') || ctx.graphType?.toLowerCase().includes('calibration')) {
                newSuggestions.push("How well is this classifier calibrated?");
                newSuggestions.push("What threshold should I use clinically?");
            } else {
                newSuggestions.push("How does this model perform overall?");
                newSuggestions.push("What do these metrics mean clinically?");
            }
        }

        // Sequence Modeling / Temporal Lab tab
        if (ctx.tab === 'sequence_modeling') {
            newSuggestions.push("How does the hybrid model compare to V1?");
            if (ctx.graphType === 'scatter') {
                newSuggestions.push("Are predictions accurate for extreme values?");
            } else if (ctx.graphType === 'error') {
                newSuggestions.push("Where do the largest errors occur?");
            } else if (ctx.graphType === 'performance') {
                newSuggestions.push("Which model tier performs best?");
            } else if (ctx.graphType === 'embedding') {
                newSuggestions.push("What do these embedding clusters represent?");
            }
        }

        // Target specific
        if (ctx.target?.toLowerCase().includes('recovery')) {
            newSuggestions.push("What affects recovery time the most?");
        } else if (ctx.target?.toLowerCase().includes('baz') || ctx.target?.toLowerCase().includes('delta')) {
            newSuggestions.push("What drives changes in growth velocity?");
        } else if (ctx.target?.toLowerCase().includes('illness') || ctx.target?.toLowerCase().includes('burden')) {
            newSuggestions.push("Why are most illness predictions zero?");
        }

        // Model specific
        if (ctx.model?.toLowerCase().includes('hybrid') || ctx.model?.toLowerCase().includes('lstm') || ctx.model?.toLowerCase().includes('tcn')) {
            newSuggestions.push("How do temporal embeddings improve predictions?");
        }

        // Ensure uniqueness and limit to 4
        newSuggestions = [...new Set(newSuggestions)].slice(0, 4);
        setSuggestions(newSuggestions);
    }, []);

    // Update Context and trigger suggestions
    const updateCopilotContext = useCallback((newPartialContext) => {
        setContext(prev => {
            const next = { ...prev, ...newPartialContext };
            generateSuggestions(next);
            return next;
        });
    }, [generateSuggestions]);

    const addMessage = (role, content) => {
        setChatHistory(prev => [...prev, { role, content }]);
    };

    const clearHistory = () => {
        setChatHistory([{ role: 'assistant', content: "Conversation cleared. I am still watching your context!" }]);
    };

    return (
        <CopilotContext.Provider value={{
            context,
            updateCopilotContext,
            chatHistory,
            setChatHistory,
            addMessage,
            clearHistory,
            suggestions,
            settings,
            setSettings,
            isWidgetOpen,
            setIsWidgetOpen,
            isTyping,
            setIsTyping
        }}>
            {children}
        </CopilotContext.Provider>
    );
};

export const useCopilot = () => useContext(CopilotContext);
