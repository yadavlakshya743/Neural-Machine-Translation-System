import React, { useState } from 'react';
import { Languages, LogIn, UserPlus, AlertCircle } from 'lucide-react';
import { api, saveSession, errorMessage } from '../api';

export default function Login({ onLogin }) {
  const [mode, setMode] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    setBusy(true);
    try {
      if (mode === 'register') {
        await api.post('/auth/register', { username, password });
      }
      const { data } = await api.post('/auth/login', { username, password });
      saveSession(data.access_token, data.username);
      onLogin({ token: data.access_token, username: data.username });
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-4">
      <form onSubmit={submit} className="w-full max-w-sm bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-5 shadow-xl shadow-black/40">
        <div className="flex items-center space-x-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center">
            <Languages className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg">TransNeural</h1>
            <p className="text-xs text-slate-400">English ↔ Hindi Transformer NMT</p>
          </div>
        </div>

        <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-800 text-sm">
          {['login', 'register'].map((m) => (
            <button key={m} type="button" onClick={() => { setMode(m); setError(''); }}
              className={`flex-1 py-1.5 rounded-lg font-medium transition ${mode === m ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}>
              {m === 'login' ? 'Sign in' : 'Create account'}
            </button>
          ))}
        </div>

        <label className="block space-y-1.5">
          <span className="text-xs text-slate-400">Username</span>
          <input value={username} onChange={(e) => setUsername(e.target.value)} required minLength={3} maxLength={40}
            autoComplete="username"
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500" />
        </label>
        <label className="block space-y-1.5">
          <span className="text-xs text-slate-400">Password {mode === 'register' && '(min. 8 characters)'}</span>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500" />
        </label>

        {error && (
          <div className="flex items-start space-x-2 text-xs text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg p-2.5">
            <AlertCircle className="h-4 w-4 shrink-0" /><span>{error}</span>
          </div>
        )}

        <button type="submit" disabled={busy}
          className="w-full inline-flex items-center justify-center space-x-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:opacity-50 text-white font-medium text-sm py-2.5 rounded-xl transition">
          {mode === 'login' ? <LogIn className="h-4 w-4" /> : <UserPlus className="h-4 w-4" />}
          <span>{busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account & sign in'}</span>
        </button>
      </form>
    </div>
  );
}
