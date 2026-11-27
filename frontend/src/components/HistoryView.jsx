import React, { useEffect, useState, useCallback } from 'react';
import { History, RefreshCw, AlertCircle } from 'lucide-react';
import { api, errorMessage } from '../api';

const PAGE = 20;

export default function HistoryView({ refreshKey }) {
  const [page, setPage] = useState({ total: 0, items: [] });
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const { data } = await api.get('/translations', { params: { limit: PAGE, offset } });
      setPage(data);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [offset]);

  useEffect(() => { load(); }, [load, refreshKey]);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <History className="h-4 w-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-200">Your translation history</h3>
          <span className="text-xs text-slate-500">({page.total} saved, visible only to you)</span>
        </div>
        <button onClick={load} className="text-xs text-slate-400 hover:text-white flex items-center space-x-1">
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} /><span>Refresh</span>
        </button>
      </div>

      {error && <div className="flex items-center space-x-2 text-xs text-rose-300"><AlertCircle className="h-4 w-4" /><span>{error}</span></div>}
      {!error && page.items.length === 0 && !loading && (
        <p className="text-sm text-slate-500 py-6 text-center">No translations yet. Translate something and it will be saved here.</p>
      )}

      <div className="divide-y divide-slate-800">
        {page.items.map((item) => (
          <div key={item.request_id} className="py-3 grid grid-cols-1 md:grid-cols-[1fr_auto] gap-2 text-sm">
            <div className="space-y-1 min-w-0">
              <p className="text-slate-300 font-devanagari break-words">{item.source_text}</p>
              <p className="text-indigo-300 font-devanagari break-words">{item.translated_text}</p>
            </div>
            <div className="flex flex-wrap md:flex-col md:items-end gap-x-3 gap-y-1 text-[11px] text-slate-500">
              <span className="px-2 py-0.5 bg-slate-950 rounded border border-slate-800 font-mono text-slate-400">
                {item.source_lang.toUpperCase()} → {item.target_lang.toUpperCase()}
              </span>
              <span className="font-mono">{item.decoding_strategy}{item.beam_size ? ` k=${item.beam_size}` : ''} · {item.latency_ms} ms</span>
              <span className="font-mono truncate max-w-[220px]">{item.model_version}</span>
              <span>{item.created_at ? new Date(item.created_at).toLocaleString() : ''}</span>
            </div>
          </div>
        ))}
      </div>

      {page.total > PAGE && (
        <div className="flex justify-between text-xs text-slate-400">
          <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))} className="disabled:opacity-40 hover:text-white">← Newer</button>
          <span>{offset + 1}–{Math.min(offset + PAGE, page.total)} of {page.total}</span>
          <button disabled={offset + PAGE >= page.total} onClick={() => setOffset(offset + PAGE)} className="disabled:opacity-40 hover:text-white">Older →</button>
        </div>
      )}
    </div>
  );
}
