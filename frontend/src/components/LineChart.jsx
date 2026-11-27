import React, { useMemo, useRef, useState } from 'react';

// Categorical series colors (dark mode), validated against the slate-900 surface:
// all checks pass, worst adjacent CVD dE 8.4. Assigned by series index in a fixed order.
export const SERIES_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500'];

const W = 640, H = 260, PAD = { top: 16, right: 16, bottom: 36, left: 48 };

function niceTicks(min, max, count = 5) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || step0;
  const start = Math.floor(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + step * 0.5; v += step) ticks.push(+v.toFixed(6));
  return ticks;
}

/**
 * series: [{ name, color, points: [{x, y}] }]  -- one y-axis only.
 */
export default function LineChart({ series, xLabel, yLabel, yFormat = (v) => v, showMarkers = false, title }) {
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);

  const { xs, ys, xTicks, yTicks, sx, sy } = useMemo(() => {
    const all = series.flatMap((s) => s.points);
    const xsV = all.map((p) => p.x), ysV = all.map((p) => p.y);
    const xMin = Math.min(...xsV), xMax = Math.max(...xsV);
    const yT = niceTicks(Math.min(0, ...ysV), Math.max(...ysV));
    const xT = niceTicks(xMin, xMax, 6).filter((t) => t >= xMin && t <= xMax);
    const yMin = yT[0], yMax = yT[yT.length - 1];
    const sxF = (x) => PAD.left + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * (W - PAD.left - PAD.right);
    const syF = (y) => H - PAD.bottom - ((y - yMin) / Math.max(1e-9, yMax - yMin)) * (H - PAD.top - PAD.bottom);
    return { xs: xsV, ys: ysV, xTicks: xT, yTicks: yT, sx: sxF, sy: syF };
  }, [series]);

  if (!series.length || !xs.length) {
    return <p className="text-sm text-slate-500 py-8 text-center">No data yet.</p>;
  }

  const onMove = (e) => {
    const rect = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = null;
    series.forEach((s) => s.points.forEach((p) => {
      const d = Math.abs(sx(p.x) - px);
      if (!best || d < best.d) best = { d, x: p.x };
    }));
    if (!best) return;
    const rows = series.map((s) => {
      const p = s.points.reduce((a, b) => (Math.abs(b.x - best.x) < Math.abs(a.x - best.x) ? b : a));
      return { name: s.name, color: s.color, x: p.x, y: p.y };
    });
    setHover({ x: best.x, rows, left: (sx(best.x) / W) * 100 });
  };

  return (
    <figure className="space-y-2">
      {title && <figcaption className="text-xs font-semibold text-slate-300">{title}</figcaption>}
      {series.length >= 2 && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-300" aria-label="Legend">
          {series.map((s) => (
            <span key={s.name} className="inline-flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 rounded" style={{ background: s.color }} />{s.name}
            </span>
          ))}
        </div>
      )}
      <div className="relative">
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img"
          aria-label={`${title || ''}: ${yLabel} by ${xLabel}`}
          onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line x1={PAD.left} x2={W - PAD.right} y1={sy(t)} y2={sy(t)} stroke="#1e293b" strokeWidth="1" />
              <text x={PAD.left - 6} y={sy(t)} textAnchor="end" dominantBaseline="middle" fontSize="10" fill="#94a3b8">{yFormat(t)}</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={`x${t}`} x={sx(t)} y={H - PAD.bottom + 14} textAnchor="middle" fontSize="10" fill="#94a3b8">
              {t >= 1000 ? `${t / 1000}k` : t}
            </text>
          ))}
          <text x={(W + PAD.left) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="#64748b">{xLabel}</text>
          <text x={12} y={(H - PAD.bottom) / 2} textAnchor="middle" fontSize="10" fill="#64748b"
            transform={`rotate(-90 12 ${(H - PAD.bottom) / 2})`}>{yLabel}</text>

          {series.map((s) => (
            <g key={s.name}>
              <polyline fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')} />
              {showMarkers && s.points.map((p) => (
                <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r="4" fill={s.color} stroke="#0f172a" strokeWidth="2" />
              ))}
            </g>
          ))}

          {hover && (
            <g pointerEvents="none">
              <line x1={sx(hover.x)} x2={sx(hover.x)} y1={PAD.top} y2={H - PAD.bottom} stroke="#475569" strokeDasharray="3 3" />
              {hover.rows.map((r) => (
                <circle key={r.name} cx={sx(r.x)} cy={sy(r.y)} r="5" fill={r.color} stroke="#0f172a" strokeWidth="2" />
              ))}
            </g>
          )}
        </svg>
        {hover && (
          <div className="absolute top-2 pointer-events-none bg-slate-950/95 border border-slate-700 rounded-lg px-2.5 py-1.5 text-[11px] shadow-lg"
            style={{ left: `${hover.left}%`, transform: hover.left > 60 ? 'translateX(-105%)' : 'translateX(8px)' }}>
            <div className="text-slate-400 mb-0.5">{xLabel} {hover.x}</div>
            {hover.rows.map((r) => (
              <div key={r.name} className="flex items-center gap-1.5 text-slate-200 whitespace-nowrap">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: r.color }} />
                <span className="text-slate-400">{r.name}</span><strong className="font-mono">{yFormat(r.y)}</strong>
              </div>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}
