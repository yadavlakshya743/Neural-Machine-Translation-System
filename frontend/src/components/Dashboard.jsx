import React, { useEffect, useState } from 'react';
import { BarChart3, Cpu, Database, RefreshCw, AlertCircle } from 'lucide-react';
import { api, errorMessage } from '../api';
import LineChart, { SERIES_COLORS } from './LineChart';

const fmt = (v, d = 2) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [models, setModels] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const [ev, m] = await Promise.all([api.get('/evaluations'), api.get('/models/active')]);
      setData(ev.data);
      setModels(m.data);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => { load(); }, []);

  // Hide throwaway pipeline-test runs; colour follows the run (stable order by name).
  const runs = (data?.experiments || []).filter((r) => !r.run_name.startsWith('smoke'));
  const colorOf = (name) => SERIES_COLORS[runs.findIndex((r) => r.run_name === name) % SERIES_COLORS.length];

  const lossSeries = runs.filter((r) => r.train_curve.length).map((r) => ({
    name: r.run_name, color: colorOf(r.run_name), points: r.train_curve.map((p) => ({ x: p.step, y: p.loss })),
  }));
  const bleuSeries = runs.filter((r) => r.epochs.length).map((r) => ({
    name: r.run_name, color: colorOf(r.run_name),
    points: r.epochs.filter((e) => e.dev_bleu !== undefined).map((e) => ({ x: e.epoch, y: e.dev_bleu })),
  }));
  const report = data?.data_report;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <BarChart3 className="h-4 w-4 text-indigo-400" />
          <h2 className="text-sm font-semibold text-slate-200">Experiments & evaluation</h2>
        </div>
        <button onClick={load} className="text-xs text-slate-400 hover:text-white flex items-center space-x-1">
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} /><span>Refresh</span>
        </button>
      </div>
      {error && <div className="flex items-center space-x-2 text-xs text-rose-300"><AlertCircle className="h-4 w-4" /><span>{error}</span></div>}

      {/* Loaded models */}
      <section className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
        <div className="flex items-center space-x-2"><Cpu className="h-4 w-4 text-indigo-400" /><h3 className="text-sm font-semibold text-slate-200">Models serving translations</h3></div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {(models?.loaded || []).map((m) => (
            <div key={m.direction} className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 text-xs space-y-1.5">
              <div className="flex justify-between"><span className="font-mono text-slate-200">{m.run_name}</span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300">{m.direction.replace('-', ' → ').toUpperCase()}</span></div>
              <div className="text-slate-400">
                {m.parameters?.toLocaleString()} parameters · d_model {m.d_model} · {m.n_heads} heads · {m.n_encoder_layers}+{m.n_decoder_layers} layers · d_ff {m.d_ff}
              </div>
              <div className="text-slate-400">Vocab {m.src_vocab_size?.toLocaleString()} / {m.tgt_vocab_size?.toLocaleString()} · trained on {m.train_pairs?.toLocaleString()} pairs · {m.trained_epochs ?? '—'} epochs · best dev BLEU <strong className="text-slate-200">{fmt(m.best_dev_bleu)}</strong></div>
              <div className="text-slate-500 font-mono truncate" title={m.tokenizer_sha256?.src}>tokenizer sha256 {m.tokenizer_sha256?.src?.slice(0, 12)}… · {m.device}</div>
            </div>
          ))}
          {models?.missing_directions?.map((d) => (
            <div key={d} className="border border-dashed border-slate-700 rounded-xl p-4 text-xs text-slate-500">
              {d.replace('-', ' → ').toUpperCase()}: no trained model loaded yet.
            </div>
          ))}
        </div>
      </section>

      {/* Curves */}
      <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <LineChart title="Training loss (label-smoothed cross-entropy)" series={lossSeries}
            xLabel="optimizer step" yLabel="loss" yFormat={(v) => fmt(v, 2)} />
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <LineChart title="Dev-set BLEU after each epoch (official dev split)" series={bleuSeries} showMarkers
            xLabel="epoch" yLabel="BLEU" yFormat={(v) => fmt(v, 1)} />
        </div>
      </section>

      {/* Per-run tables (also the accessible table view of the charts) */}
      {runs.map((r) => (
        <section key={r.run_name} className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: colorOf(r.run_name) }} />
              <h3 className="text-sm font-semibold text-slate-200 font-mono">{r.run_name}</h3>
              <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">{r.status}</span>
            </div>
            <span className="text-[11px] text-slate-400">
              {r.hours_trained} h · step {r.global_step?.toLocaleString()} · lr {r.training.learning_rate} · warmup {r.training.warmup_steps} · label smoothing {r.training.label_smoothing}
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="text-slate-400 border-b border-slate-800">
                <tr><th className="py-1.5 pr-4">Epoch</th><th className="pr-4">Step</th><th className="pr-4">Dev loss</th><th className="pr-4">Dev BLEU</th><th className="pr-4">Dev chrF++</th><th>Best</th></tr>
              </thead>
              <tbody className="text-slate-300 font-mono">
                {r.epochs.map((e) => (
                  <tr key={e.epoch} className="border-b border-slate-800/60">
                    <td className="py-1.5 pr-4">{e.epoch}</td><td className="pr-4">{e.step?.toLocaleString()}</td>
                    <td className="pr-4">{fmt(e.val_loss, 3)}</td><td className="pr-4">{fmt(e.dev_bleu)}</td>
                    <td className="pr-4">{fmt(e.dev_chrf2)}</td><td>{e.best ? '✓' : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {r.evaluations.length > 0 && (
            <div className="space-y-3">
              <h4 className="text-xs font-semibold text-slate-300">Official test split (never used for training or model selection)</h4>
              <table className="w-full text-xs text-left">
                <thead className="text-slate-400 border-b border-slate-800">
                  <tr><th className="py-1.5 pr-4">Decoding</th><th className="pr-4">BLEU</th><th className="pr-4">chrF++</th><th className="pr-4">Sentences</th><th>ms / sentence</th></tr>
                </thead>
                <tbody className="text-slate-200 font-mono">
                  {r.evaluations.filter((e) => e.split === 'test').map((e) => (
                    <tr key={`${e.decoding.strategy}${e.decoding.beam_size}`} className="border-b border-slate-800/60">
                      <td className="py-1.5 pr-4">{e.decoding.strategy}{e.decoding.strategy === 'beam_search' ? ` (k=${e.decoding.beam_size})` : ''}</td>
                      <td className="pr-4">{fmt(e.bleu)}</td><td className="pr-4">{fmt(e.chrf2)}</td>
                      <td className="pr-4">{e.samples_evaluated?.toLocaleString()}</td><td>{fmt(e.ms_per_sentence, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <details className="text-xs">
                <summary className="cursor-pointer text-slate-400 hover:text-white">Example outputs (source / reference / model)</summary>
                <div className="mt-2 space-y-3">
                  {(r.evaluations.find((e) => e.split === 'test')?.examples || []).map((x, i) => (
                    <div key={i} className="font-devanagari space-y-0.5 border-l-2 border-slate-700 pl-3">
                      <p className="text-slate-300"><span className="text-slate-500 font-sans">SRC </span>{x.source}</p>
                      <p className="text-slate-400"><span className="text-slate-500 font-sans">REF </span>{x.reference}</p>
                      <p className="text-indigo-300"><span className="text-slate-500 font-sans">HYP </span>{x.hypothesis}</p>
                    </div>
                  ))}
                </div>
              </details>
              <p className="text-[10px] text-slate-500 font-mono break-all">{r.evaluations[0]?.bleu_signature}</p>
            </div>
          )}
        </section>
      ))}

      {/* Data pipeline */}
      {report && (
        <section className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center space-x-2"><Database className="h-4 w-4 text-indigo-400" /><h3 className="text-sm font-semibold text-slate-200">Dataset: IIT Bombay English–Hindi (official split)</h3></div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
            {[
              ['Raw train pairs', report.train_cleaning.raw],
              ['Filtered (length/ratio/empty)', report.train_cleaning.rejected_by_filters],
              ['Duplicates removed', report.train_cleaning.duplicates],
              ['Dev/test overlap removed', report.train_cleaning.leakage_overlap],
              ['Training pairs kept', report.train_cleaning.kept],
            ].map(([label, v]) => (
              <div key={label} className="bg-slate-950/60 border border-slate-800 rounded-xl p-3">
                <div className="text-slate-400">{label}</div>
                <div className="text-lg font-semibold text-slate-100 font-mono">{v?.toLocaleString()}</div>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-slate-500">
            Dev {report.lengths?.dev_en?.sentences} sentences · Test {report.lengths?.test_en?.sentences} sentences · Vocab {report.vocab_size?.en?.toLocaleString()} (EN) / {report.vocab_size?.hi?.toLocaleString()} (HI) SentencePiece BPE
          </p>
        </section>
      )}
    </div>
  );
}
