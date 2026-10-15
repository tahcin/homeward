import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

// One hand-built horizontal chart that covers every figure on the Evidence page:
// bars from a zero baseline, or dots with 95% interval whiskers, in groups of one or more series.

export interface Item {
  series: string;
  value: number | null;
  lo?: number | null;
  hi?: number | null;
  color: string;
  hollow?: boolean;
  /** A reference value (e.g. the published figure): drawn as a hollow ring joined to the dot. */
  ref?: number | null;
  /** Text at the mark's end; defaults to the formatted value. */
  display?: string;
  /** Extra lines for the tooltip. */
  detail?: string[];
}
export interface Group {
  label: string;
  sub?: string;
  items: Item[];
  strong?: boolean;
}

export const INK = "var(--ink)";
export const COLORS = {
  gemma: "#a64b25", // the app accent: Gemma 4, our systems
  compare: "#2f6fb0", // the comparison system (Whisper, other model makers)
  context: "#b9aa9c", // de-emphasised context rows
  muted: "#a99a8d", // comparators next to our own system
};

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(el);
    setW(Math.floor(el.getBoundingClientRect().width));
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

function niceTicks(min: number, max: number, target: number) {
  const span = max - min;
  const raw = span / Math.max(1, target);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(Math.round(v * 1000) / 1000);
  return out;
}

interface Tip {
  x: number;
  y: number;
  value: string;
  lines: string[];
}

export function HChart({
  groups,
  domain,
  mark = "bar",
  format = (v) => `${v}`,
  axisLabel,
  labelWidth = 190,
  ariaLabel,
  ticks: tickCount,
}: {
  groups: Group[];
  domain: [number, number];
  mark?: "bar" | "dot";
  format?: (v: number) => string;
  axisLabel: string;
  labelWidth?: number;
  ariaLabel: string;
  ticks?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<Tip | null>(null);
  const [hover, setHover] = useState<string | null>(null);

  const narrow = width > 0 && width < 560;
  const barH = mark === "bar" ? 16 : 18;
  const gap = 2; // surface gap between touching bars
  const groupGap = narrow ? 14 : 12;
  const labelW = narrow ? 0 : labelWidth;
  const valueRoom = 64; // room past the mark for its value label
  const left = labelW + (narrow ? 0 : 12);
  const right = 12;
  const plotW = Math.max(60, width - left - right - (mark === "bar" ? valueRoom : 0));
  const [d0, d1] = domain;
  const x = (v: number) => left + ((Math.min(Math.max(v, d0), d1) - d0) / (d1 - d0)) * plotW;
  const ticks = niceTicks(d0, d1, tickCount ?? (narrow ? 4 : 5));

  let y = 8;
  const layout = groups.map((g) => {
    const groupLabelH = narrow ? (g.sub ? 38 : 22) : 0;
    const top = y + groupLabelH;
    const itemsH = g.items.length * (barH + gap) - gap;
    // A two-line label (name + sub) needs more room than a single mark; centre the marks in it.
    const contentH = Math.max(itemsH, !narrow && g.sub ? 34 : 0);
    const off = (contentH - itemsH) / 2;
    const rows = g.items.map((it, i) => ({ it, y: top + off + i * (barH + gap) }));
    const h = groupLabelH + contentH;
    const out = { g, y, top, rows, h, groupLabelH };
    y += h + groupGap;
    return out;
  });
  const plotBottom = y - groupGap + 8;
  const H = plotBottom + 46;

  const show = (e: React.PointerEvent | React.FocusEvent, key: string, g: Group, it: Item, cy: number) => {
    setHover(key);
    const box = ref.current?.getBoundingClientRect();
    let px: number;
    if ("clientX" in e && box) px = e.clientX - box.left;
    else px = it.value === null ? left : x(it.value);
    const lines = [g.label + (g.items.length > 1 || it.series !== g.label ? ` · ${it.series}` : ""), ...(it.detail ?? [])];
    setTip({ x: px, y: cy, value: it.value === null ? "Not measured" : it.display ?? format(it.value), lines });
  };
  const hide = () => {
    setHover(null);
    setTip(null);
  };

  return (
    <div ref={ref} className="ev-chart" onPointerLeave={hide}>
      {width > 0 && (
        <svg width={width} height={H} role="img" aria-label={ariaLabel}>
          {/* grid + axis */}
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={4} y2={plotBottom} stroke="var(--line)" strokeWidth={1} shapeRendering="crispEdges" />
              <text x={x(t)} y={plotBottom + 16} textAnchor="middle" className="ev-tick">
                {format(t)}
              </text>
            </g>
          ))}
          {mark === "bar" && <line x1={x(d0)} x2={x(d0)} y1={4} y2={plotBottom} stroke="var(--line-strong)" strokeWidth={1} shapeRendering="crispEdges" />}
          <text x={left + plotW / 2} y={plotBottom + 38} textAnchor="middle" className="ev-axis">
            {axisLabel}
          </text>

          {layout.map(({ g, y: gy, top, rows, h, groupLabelH }) => (
            <g key={g.label}>
              {narrow ? (
                <>
                  <text x={0} y={gy + 14} className={g.strong ? "ev-glabel strong" : "ev-glabel"}>
                    {g.label}
                  </text>
                  {g.sub && (
                    <text x={0} y={gy + 30} className="ev-gsub">
                      {g.sub}
                    </text>
                  )}
                </>
              ) : (
                <>
                  <text x={labelW} y={top + (h - groupLabelH) / 2 + (g.sub ? -2 : 4)} textAnchor="end" className={g.strong ? "ev-glabel strong" : "ev-glabel"}>
                    {g.label}
                  </text>
                  {g.sub && (
                    <text x={labelW} y={top + (h - groupLabelH) / 2 + 13} textAnchor="end" className="ev-gsub">
                      {g.sub}
                    </text>
                  )}
                </>
              )}
              {rows.map(({ it, y: ry }) => {
                const key = `${g.label}|${it.series}`;
                const cy = ry + barH / 2;
                const dim = hover !== null && hover !== key;
                const v = it.value;
                const label = v === null ? "Not measured" : it.display ?? format(v);
                return (
                  <g
                    key={key}
                    tabIndex={0}
                    role="img"
                    aria-label={`${g.label}, ${it.series}: ${label}`}
                    onPointerMove={(e) => show(e, key, g, it, cy)}
                    onFocus={(e) => show(e, key, g, it, cy)}
                    onBlur={hide}
                    style={{ opacity: dim ? 0.55 : 1, transition: "opacity 0.12s ease", outline: "none" }}
                  >
                    {/* hit target: the whole row, wider than the mark */}
                    <rect x={left - 4} y={ry - gap / 2 - 1} width={plotW + valueRoom + 8} height={barH + gap + 2} fill="transparent" />
                    {v !== null && mark === "bar" && <BarShape x0={x(d0)} x1={x(v)} y={ry} h={barH} color={it.color} />}
                    {v !== null && mark === "dot" && (
                      <>
                        {it.lo != null && it.hi != null && (
                          <>
                            <line x1={x(it.lo)} x2={x(it.hi)} y1={cy} y2={cy} stroke={it.color} strokeWidth={2} strokeLinecap="round" />
                            <line x1={x(it.lo)} x2={x(it.lo)} y1={cy - 5} y2={cy + 5} stroke={it.color} strokeWidth={2} strokeLinecap="round" />
                            <line x1={x(it.hi)} x2={x(it.hi)} y1={cy - 5} y2={cy + 5} stroke={it.color} strokeWidth={2} strokeLinecap="round" />
                          </>
                        )}
                        {it.ref != null && (
                          <>
                            <line x1={x(it.ref)} x2={x(v)} y1={cy} y2={cy} stroke={it.color} strokeOpacity={0.45} strokeWidth={2} />
                            <circle cx={x(it.ref)} cy={cy} r={5} fill="var(--surface)" stroke={it.color} strokeWidth={2} />
                          </>
                        )}
                        <circle cx={x(v)} cy={cy} r={it.hollow ? 5 : 6} fill={it.hollow ? "var(--surface)" : it.color} stroke={it.hollow ? it.color : "var(--surface)"} strokeWidth={2} />
                      </>
                    )}
                    {mark === "bar" && (
                      <text x={(v === null ? x(d0) : x(v)) + 6} y={cy + 4} className="ev-val">
                        {label}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      )}
      {tip && (
        <div className="ev-tip" style={{ left: Math.min(Math.max(tip.x, 90), width - 90), top: tip.y }}>
          <strong>{tip.value}</strong>
          {tip.lines.map((l, i) => (
            <span key={i}>{l}</span>
          ))}
        </div>
      )}
    </div>
  );
}

/** A bar with a 4px rounded data end and a square baseline end. */
function BarShape({ x0, x1, y, h, color }: { x0: number; x1: number; y: number; h: number; color: string }) {
  const w = Math.max(0, x1 - x0);
  if (w < 1) return <rect x={x0} y={y} width={1} height={h} fill={color} />;
  const r = Math.min(4, w, h / 2);
  const d = `M${x0} ${y} H${x0 + w - r} Q${x0 + w} ${y} ${x0 + w} ${y + r} V${y + h - r} Q${x0 + w} ${y + h} ${x0 + w - r} ${y + h} H${x0} Z`;
  return <path d={d} fill={color} />;
}

export function Legend({ items }: { items: { label: string; color: string; shape?: "bar" | "dot" | "ring" }[] }) {
  return (
    <div className="ev-legend" aria-hidden="true">
      {items.map((i) => (
        <span key={i.label}>
          <svg width="14" height="12" viewBox="0 0 14 12">
            {i.shape === "dot" ? (
              <circle cx="7" cy="6" r="5" fill={i.color} />
            ) : i.shape === "ring" ? (
              <circle cx="7" cy="6" r="4" fill="var(--surface)" stroke={i.color} strokeWidth="2" />
            ) : (
              <rect x="0" y="2" width="14" height="8" rx="2" fill={i.color} />
            )}
          </svg>
          {i.label}
        </span>
      ))}
    </div>
  );
}

export function ChartCard({
  id,
  title,
  sub,
  children,
  caveats,
  source,
  table,
}: {
  id: string;
  title: string;
  sub?: ReactNode;
  children: ReactNode;
  caveats?: ReactNode[];
  source: ReactNode;
  table?: { head: string[]; rows: (string | number)[][]; numeric?: boolean[] };
}) {
  return (
    <figure className="card ev-card" aria-labelledby={`${id}-t`}>
      <h3 id={`${id}-t`}>{title}</h3>
      {sub && <p className="ev-sub">{sub}</p>}
      {children}
      {caveats && caveats.length > 0 && (
        <ul className="ev-caveats">
          {caveats.map((c, i) => (
            <li key={i}>{c}</li>
          ))}
        </ul>
      )}
      <figcaption className="ev-source">{source}</figcaption>
      {table && (
        <details className="ev-table">
          <summary>Show the numbers</summary>
          <div className="table-scroll">
            <table className="audit">
              <thead>
                <tr>
                  {table.head.map((h, i) => (
                    <th key={i} style={table.numeric?.[i] ? { textAlign: "right" } : undefined}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((r, i) => (
                  <tr key={i}>
                    {r.map((c, j) => (
                      <td key={j} className={table.numeric?.[j] ? "num" : undefined}>
                        {c}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </figure>
  );
}
