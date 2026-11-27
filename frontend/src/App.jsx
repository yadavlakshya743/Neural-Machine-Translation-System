import React, { useCallback, useEffect, useState } from 'react';
import { Languages, LogOut, Sparkles, History, BarChart3 } from 'lucide-react';
import { api, getSession, clearSession, setUnauthorizedHandler } from './api';
import Login from './components/Login';
import Translate from './components/Translate';
import HistoryView from './components/HistoryView';
import Dashboard from './components/Dashboard';

const TABS = [
  { id: 'translate', label: 'Translate', icon: Sparkles },
  { id: 'history', label: 'History', icon: History },
  { id: 'evaluation', label: 'Evaluation', icon: BarChart3 },
];

export default function App() {
  const [session, setSession] = useState(getSession());
  const [tab, setTab] = useState('translate');
  const [ready, setReady] = useState(null);
  const [apiState, setApiState] = useState('checking');
  const [historyKey, setHistoryKey] = useState(0);

  const logout = useCallback(() => {
    api.post('/auth/logout').catch(() => {});
    clearSession();
    setSession(null);
  }, []);

  useEffect(() => { setUnauthorizedHandler(() => { clearSession(); setSession(null); }); }, []);

  useEffect(() => {
    // /ready returns 503 (with a body) when no model is loaded: still "online".
    api.get('/ready', { validateStatus: () => true })
      .then((r) => { setReady(r.data); setApiState(r.status === 200 ? 'ready' : 'no-model'); })
      .catch(() => setApiState('offline'));
  }, [session]);

  if (!session) return <Login onLogin={setSession} />;

  const status = {
    ready: ['bg-emerald-400', `Model ready (${ready?.loaded_directions?.join(', ')})`],
    'no-model': ['bg-amber-400', 'API online, no model loaded'],
    offline: ['bg-rose-400', 'API offline'],
    checking: ['bg-slate-500', 'Connecting…'],
  }[apiState];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center space-x-3 min-w-0">
            <div className="h-10 w-10 shrink-0 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center">
              <Languages className="h-6 w-6 text-white" />
            </div>
            <div className="min-w-0">
              <h1 className="font-bold text-lg tracking-tight">TransNeural</h1>
              <p className="text-xs text-slate-400 truncate">English ↔ Hindi · custom Transformer · IIT Bombay corpus</p>
            </div>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="hidden sm:flex items-center space-x-2 text-slate-400 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800">
              <span className={`h-2 w-2 rounded-full ${status[0]}`} /><span>{status[1]}</span>
            </span>
            <span className="text-slate-300">{session.username}</span>
            <button onClick={logout} className="flex items-center space-x-1 text-slate-400 hover:text-white bg-slate-800 px-2.5 py-1.5 rounded-lg border border-slate-700">
              <LogOut className="h-3.5 w-3.5" /><span>Sign out</span>
            </button>
          </div>
        </div>
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 flex gap-1">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setTab(id)}
              className={`flex items-center gap-1.5 px-3 py-2 text-sm border-b-2 transition ${tab === id ? 'border-indigo-500 text-white' : 'border-transparent text-slate-400 hover:text-slate-200'}`}>
              <Icon className="h-4 w-4" />{label}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        {tab === 'translate' && <Translate ready={ready} onTranslated={() => setHistoryKey((k) => k + 1)} />}
        {tab === 'history' && <HistoryView refreshKey={historyKey} />}
        {tab === 'evaluation' && <Dashboard />}
      </main>

      <footer className="border-t border-slate-800 py-5 text-center text-xs text-slate-500">
        Neural Machine Translation & Localization Platform · Grounded in “Attention Is All You Need” (Vaswani et al., 2017)
      </footer>
    </div>
  );
}
