import React, { useState } from 'react';
import { ArrowRightLeft, Sparkles, Copy, Check, AlertCircle } from 'lucide-react';
import { api, errorMessage } from '../api';

const SAMPLES = {
  'en-hi': ['I am going to college.', 'The government has announced a new policy for farmers.', 'India is a beautiful country.'],
  'hi-en': ['मैं कॉलेज जा रहा हूँ।', 'सरकार ने किसानों के लिए नई नीति की घोषणा की है।', 'भारत एक सुंदर देश है।'],
};

export default function Translate({ ready, onTranslated }) {
  const [sourceText, setSourceText] = useState('');
  const [result, setResult] = useState(null);
  const [direction, setDirection] = useState('en-hi');
  const [strategy, setStrategy] = useState('beam_search');
  const [beamSize, setBeamSize] = useState(4);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const [src, tgt] = direction.split('-');
  const available = ready?.loaded_directions?.includes(direction);

  const swap = () => {
    setDirection((d) => (d === 'en-hi' ? 'hi-en' : 'en-hi'));
    if (result) setSourceText(result.translated_text);
    setResult(null);
    setError('');
  };

  const translate = async () => {
    if (!sourceText.trim()) return;
    setLoading(true);
    setError('');
    try {
      const { data } = await api.post('/translations', {
        source_text: sourceText, source_lang: src, target_lang: tgt,
        decoding_strategy: strategy, beam_size: Number(beamSize), length_penalty: 0.6, max_length: 128,
      });
      setResult(data);
      onTranslated?.();
    } catch (err) {
      setResult(null);
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const copy = () => {
    if (!result) return;
    navigator.clipboard?.writeText(result.translated_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="space-y-6">
      <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-2xl flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-sm">
          <span className="px-4 py-1.5 rounded-lg font-medium bg-indigo-600 text-white">{src === 'en' ? 'English' : 'Hindi (हिन्दी)'}</span>
          <button onClick={swap} className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition" title="Swap direction">
            <ArrowRightLeft className="h-4 w-4" />
          </button>
          <span className="px-4 py-1.5 rounded-lg font-medium text-slate-300">{tgt === 'en' ? 'English' : 'Hindi (हिन्दी)'}</span>
        </div>
        <div className="flex flex-wrap items-center gap-4 text-xs">
          <label className="flex items-center space-x-2">
            <span className="text-slate-400 font-medium">Decoding</span>
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200">
              <option value="beam_search">Beam search (better quality)</option>
              <option value="greedy">Greedy (fastest)</option>
            </select>
          </label>
          {strategy === 'beam_search' && (
            <label className="flex items-center space-x-2">
              <span className="text-slate-400">Beam size</span>
              <select value={beamSize} onChange={(e) => setBeamSize(e.target.value)}
                className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-slate-200">
                {[2, 4, 6, 8].map((k) => <option key={k} value={k}>{k}</option>)}
              </select>
            </label>
          )}
        </div>
      </div>

      {ready && !available && (
        <div className="flex items-start space-x-2 text-xs text-amber-200 bg-amber-500/10 border border-amber-500/30 rounded-xl p-3">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>No trained {src.toUpperCase()}→{tgt.toUpperCase()} model is loaded yet. Loaded: {ready.loaded_directions.join(', ') || 'none'}.</span>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl flex flex-col overflow-hidden focus-within:border-indigo-500/60 transition">
          <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-950/40 text-xs">
            <span className="font-semibold uppercase tracking-wider text-slate-400">Source ({src === 'en' ? 'English' : 'Hindi'})</span>
            <span className="text-slate-500">{sourceText.length}/1000</span>
          </div>
          <textarea value={sourceText} maxLength={1000} onChange={(e) => setSourceText(e.target.value)}
            placeholder={src === 'en' ? 'Type English text to translate…' : 'अनुवाद के लिए हिन्दी लिखें…'}
            className="flex-1 w-full bg-transparent p-5 resize-none text-slate-100 placeholder-slate-500 focus:outline-none text-base min-h-[200px] font-devanagari" />
          <div className="p-4 border-t border-slate-800 bg-slate-950/30 flex flex-wrap gap-3 justify-between items-center">
            <div className="flex flex-wrap gap-2">
              {SAMPLES[direction].map((s, i) => (
                <button key={s} onClick={() => setSourceText(s)} className="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded text-[11px] text-slate-300 transition">
                  Sample {i + 1}
                </button>
              ))}
            </div>
            <button onClick={translate} disabled={loading || !sourceText.trim()}
              className="inline-flex items-center space-x-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:opacity-50 text-white font-medium text-sm px-5 py-2 rounded-xl transition">
              {loading ? <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> : <Sparkles className="h-4 w-4" />}
              <span>{loading ? 'Translating…' : 'Translate'}</span>
            </button>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-950/40 text-xs">
            <span className="font-semibold uppercase tracking-wider text-indigo-400">Model output ({tgt === 'en' ? 'English' : 'Hindi'})</span>
            <button onClick={copy} disabled={!result} className="flex items-center space-x-1.5 text-slate-400 hover:text-white disabled:opacity-40">
              {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
          </div>
          <div className="flex-1 p-5 text-base font-devanagari text-slate-100 min-h-[200px] whitespace-pre-wrap leading-relaxed">
            {error ? (
              <div className="flex items-start space-x-2 text-sm text-rose-300 font-sans"><AlertCircle className="h-4 w-4 mt-0.5 shrink-0" /><span>{error}</span></div>
            ) : result ? result.translated_text : (
              <span className="text-slate-500 italic font-sans">The model-generated translation appears here.</span>
            )}
          </div>
          <div className="p-4 border-t border-slate-800 bg-slate-950/30 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-400">
            <span>Model <strong className="block text-slate-200 font-mono truncate">{result?.model_version ?? '—'}</strong></span>
            <span>Decoding <strong className="block text-slate-200 font-mono">{result ? `${result.decoding_strategy}${result.beam_size ? ` k=${result.beam_size}` : ''}` : '—'}</strong></span>
            <span>Latency <strong className="block text-slate-200 font-mono">{result ? `${result.latency_ms} ms` : '—'}</strong></span>
            <span>Tokens <strong className="block text-slate-200 font-mono">{result?.tokens_generated ?? '—'}</strong></span>
          </div>
        </div>
      </div>
      <p className="text-[11px] text-slate-500">
        Output is generated by the project's own Transformer model and may contain errors. Automatic metrics (BLEU/chrF++) measure overlap with references, not guaranteed meaning.
      </p>
    </div>
  );
}
