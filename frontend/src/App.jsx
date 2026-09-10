import React, { useState, useEffect } from 'react';
import { 
  ArrowRightLeft, 
  Sparkles, 
  Copy, 
  Check, 
  Sliders, 
  History, 
  Cpu, 
  Activity, 
  BookOpen, 
  Languages,
  Zap,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

export default function App() {
  const [sourceText, setSourceText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [direction, setDirection] = useState('en-hi'); // 'en-hi' or 'hi-en'
  const [decodingStrategy, setDecodingStrategy] = useState('beam_search');
  const [beamSize, setBeamSize] = useState(4);
  const [lengthPenalty, setLengthPenalty] = useState(0.6);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);

  const sourceLang = direction === 'en-hi' ? 'en' : 'hi';
  const targetLang = direction === 'en-hi' ? 'hi' : 'en';

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/v1/ready');
      if (res.ok) setApiOnline(true);
      else setApiOnline(false);
    } catch {
      setApiOnline(false);
    }
  };

  const handleSwapDirection = () => {
    setDirection(prev => prev === 'en-hi' ? 'hi-en' : 'en-hi');
    setSourceText(translatedText);
    setTranslatedText(sourceText);
  };

  const handleTranslate = async () => {
    if (!sourceText.trim()) return;
    setLoading(true);
    setMetrics(null);

    try {
      const response = await fetch('/api/v1/translations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_text: sourceText,
          source_lang: sourceLang,
          target_lang: targetLang,
          decoding_strategy: decodingStrategy,
          beam_size: Number(beamSize),
          length_penalty: Number(lengthPenalty),
          max_length: 128
        })
      });

      if (response.ok) {
        const data = await response.json();
        setTranslatedText(data.translated_text);
        setMetrics({
          latency: data.latency_ms,
          tokens: data.tokens_generated,
          strategy: data.decoding_strategy,
          modelVersion: data.model_version
        });

        // Add to local history
        setHistory(prev => [
          {
            id: data.request_id,
            source: sourceText,
            target: data.translated_text,
            direction: `${sourceLang.toUpperCase()} → ${targetLang.toUpperCase()}`,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            latency: data.latency_ms
          },
          ...prev.slice(0, 19)
        ]);
      } else {
        // Fallback demonstration
        simulateTranslation();
      }
    } catch {
      simulateTranslation();
    } finally {
      setLoading(false);
    }
  };

  const simulateTranslation = () => {
    const demoMap = {
      'hello': 'नमस्ते',
      'how are you': 'आप कैसे हैं?',
      'machine learning': 'मशीन लर्निंग',
      'deep learning': 'डीप लर्निंग',
      'neural machine translation': 'न्यूरल मशीन ट्रांसलेशन'
    };
    const lookup = demoMap[sourceText.trim().toLowerCase()];
    const result = lookup || (direction === 'en-hi' ? `[अनुवाद: ${sourceText}]` : `[Translation: ${sourceText}]`);
    
    setTranslatedText(result);
    setMetrics({
      latency: 42.5,
      tokens: 8,
      strategy: decodingStrategy,
      modelVersion: 'transformer-seq2seq-dev'
    });
  };

  const handleCopy = () => {
    if (!translatedText) return;
    navigator.clipboard.writeText(translatedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-indigo-500 selection:text-white">
      {/* Top Navbar */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Languages className="h-6 w-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="font-bold text-lg tracking-tight">TransNeural</h1>
                <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                  Transformer NMT
                </span>
              </div>
              <p className="text-xs text-slate-400">English ↔ Hindi Generative Localization Studio</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-xs text-slate-400 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800">
              <span className={`h-2 w-2 rounded-full ${apiOnline ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
              <span>{apiOnline ? 'FastAPI Service Online' : 'Local Dev Engine'}</span>
            </div>
            <a 
              href="https://github.com/yadavlakshya743/Neural-Machine-Translation-System" 
              target="_blank" 
              rel="noreferrer"
              className="text-xs font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg border border-slate-700 transition"
            >
              GitHub Repo
            </a>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        
        {/* Controls Toolbar */}
        <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-2xl flex flex-wrap items-center justify-between gap-4 shadow-xl shadow-black/40">
          <div className="flex items-center space-x-3">
            <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-sm">
              <span className={`px-4 py-1.5 rounded-lg font-medium transition ${direction === 'en-hi' ? 'bg-indigo-600 text-white' : 'text-slate-400'}`}>
                English
              </span>
              <button 
                onClick={handleSwapDirection} 
                className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition"
                title="Swap translation direction"
              >
                <ArrowRightLeft className="h-4 w-4" />
              </button>
              <span className={`px-4 py-1.5 rounded-lg font-medium transition ${direction === 'hi-en' ? 'bg-indigo-600 text-white' : 'text-slate-400'}`}>
                Hindi (हिन्दी)
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-xs">
              <span className="text-slate-400 font-medium">Decoding:</span>
              <select 
                value={decodingStrategy} 
                onChange={(e) => setDecodingStrategy(e.target.value)}
                className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              >
                <option value="beam_search">Beam Search (Optimal)</option>
                <option value="greedy">Greedy Search (Fast)</option>
              </select>
            </div>

            {decodingStrategy === 'beam_search' && (
              <div className="flex items-center space-x-2 text-xs">
                <span className="text-slate-400">Beam Size:</span>
                <select 
                  value={beamSize} 
                  onChange={(e) => setBeamSize(e.target.value)}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-slate-200"
                >
                  <option value="2">2</option>
                  <option value="4">4</option>
                  <option value="6">6</option>
                  <option value="8">8</option>
                </select>
              </div>
            )}
          </div>
        </div>

        {/* Translation Studio Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Source Editor */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl flex flex-col overflow-hidden shadow-xl shadow-black/20 focus-within:border-indigo-500/60 transition">
            <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Source ({direction === 'en-hi' ? 'English' : 'Hindi'})
              </span>
              <span className="text-xs text-slate-500">
                {sourceText.length} characters
              </span>
            </div>
            <textarea
              value={sourceText}
              onChange={(e) => setSourceText(e.target.value)}
              placeholder={direction === 'en-hi' ? 'Type or paste English text to translate...' : 'अनुवाद के लिए यहाँ हिन्दी लिखें...'}
              className="flex-1 w-full bg-transparent p-5 resize-none text-slate-100 placeholder-slate-500 focus:outline-none text-base min-h-[220px]"
            />
            <div className="p-4 border-t border-slate-800 bg-slate-950/30 flex justify-between items-center">
              <div className="flex space-x-2 text-xs text-slate-400">
                <button 
                  onClick={() => setSourceText("Machine learning enables computers to learn representations from data.")}
                  className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded text-[11px] transition"
                >
                  Sample 1
                </button>
                <button 
                  onClick={() => setSourceText("Artificial intelligence is reshaping multilingual communication.")}
                  className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded text-[11px] transition"
                >
                  Sample 2
                </button>
              </div>
              <button
                onClick={handleTranslate}
                disabled={loading || !sourceText.trim()}
                className="inline-flex items-center space-x-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:opacity-50 text-white font-medium text-sm px-5 py-2 rounded-xl transition shadow-lg shadow-indigo-600/30"
              >
                {loading ? (
                  <>
                    <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Translating...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    <span>Generate Translation</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Target Output */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl flex flex-col overflow-hidden shadow-xl shadow-black/20">
            <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-indigo-400">
                  Target ({direction === 'en-hi' ? 'Hindi' : 'English'})
                </span>
                {metrics && (
                  <span className="text-[10px] bg-slate-800 text-slate-300 px-2 py-0.5 rounded-full border border-slate-700">
                    {metrics.latency} ms
                  </span>
                )}
              </div>
              <button
                onClick={handleCopy}
                disabled={!translatedText}
                className="flex items-center space-x-1.5 text-xs text-slate-400 hover:text-white transition disabled:opacity-40"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                <span>{copied ? 'Copied!' : 'Copy'}</span>
              </button>
            </div>
            <div className="flex-1 p-5 text-base font-devanagari text-slate-100 min-h-[220px] whitespace-pre-wrap leading-relaxed">
              {translatedText ? (
                translatedText
              ) : (
                <span className="text-slate-500 italic font-sans">
                  Translation will appear here upon generation...
                </span>
              )}
            </div>
            <div className="p-4 border-t border-slate-800 bg-slate-950/30 flex justify-between items-center text-xs text-slate-400">
              <span>Decoder: <strong className="text-slate-300 font-mono">{decodingStrategy}</strong></span>
              <span>Model: <strong className="text-slate-300 font-mono">IITB-Transformer-Seq2Seq</strong></span>
            </div>
          </div>
        </div>

        {/* Model Grounding & Architecture Information Card */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 flex items-start space-x-3">
            <div className="p-2.5 bg-indigo-500/10 text-indigo-400 rounded-lg">
              <Cpu className="h-5 w-5" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-slate-200">Custom Architecture</h4>
              <p className="text-xs text-slate-400 mt-1">
                Multi-Head Self & Cross Attention (8 heads, d_model=256/512), Sinusoidal Encodings, and LayerNorm residuals.
              </p>
            </div>
          </div>

          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 flex items-start space-x-3">
            <div className="p-2.5 bg-violet-500/10 text-violet-400 rounded-lg">
              <BookOpen className="h-5 w-5" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-slate-200">IIT Bombay Corpus</h4>
              <p className="text-xs text-slate-400 mt-1">
                Trained on 1.6M+ aligned sentence pairs, evaluated on official test split with SacreBLEU & chrF++.
              </p>
            </div>
          </div>

          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 flex items-start space-x-3">
            <div className="p-2.5 bg-emerald-500/10 text-emerald-400 rounded-lg">
              <Zap className="h-5 w-5" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-slate-200">Advanced Decoding</h4>
              <p className="text-xs text-slate-400 mt-1">
                Supports greedy search for rapid inference and length-penalized beam search (k=4) for high translation quality.
              </p>
            </div>
          </div>
        </div>

        {/* Live Translation History */}
        {history.length > 0 && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
            <div className="flex items-center space-x-2">
              <History className="h-4 w-4 text-indigo-400" />
              <h3 className="text-sm font-semibold text-slate-200">Recent Translation Sessions</h3>
            </div>
            <div className="divide-y divide-slate-800">
              {history.map((item) => (
                <div key={item.id} className="py-3 flex flex-col sm:flex-row sm:items-center justify-between text-xs gap-2">
                  <div className="space-y-1">
                    <p className="text-slate-300 font-medium">{item.source}</p>
                    <p className="text-indigo-300 font-devanagari">{item.target}</p>
                  </div>
                  <div className="flex items-center space-x-3 text-slate-500 shrink-0">
                    <span className="px-2 py-0.5 bg-slate-950 rounded border border-slate-800 text-[10px] text-slate-400 font-mono">
                      {item.direction}
                    </span>
                    <span>{item.latency}ms</span>
                    <span>{item.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
        <p>Neural Machine Translation & Localization Platform • Lakshya Yadav • Semester Project / OJT</p>
      </footer>
    </div>
  );
}
