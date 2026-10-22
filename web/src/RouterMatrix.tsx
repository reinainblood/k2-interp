import { useEffect, useRef, useState } from "react";
import type { Frame } from "./types";
import { number } from "./types";

type Props = {
  frame: Frame | null;
  layer: number;
  onLayer: (layer: number) => void;
  theme: string;
};
export function RouterMatrix({ frame, layer, onLayer, theme }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<"weight" | "score">("weight");
  const [hover, setHover] = useState<{
    layer: number;
    expert: number;
    value?: number;
  } | null>(null);
  const layers = frame?.layers || {};
  const values = Object.values(layers).flatMap((v) =>
    mode === "score" ? v.router_scores || [] : v.expert_weights || [],
  );
  const max = Math.max(0.0001, ...values);
  const observed = Object.values(layers).filter((l) => l.expert_ids).length;
  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext("2d");
    if (!el || !ctx) return;
    const ratio = window.devicePixelRatio || 1;
    const w = 1008,
      h = 394;
    el.width = w * ratio;
    el.height = h * ratio;
    ctx.scale(ratio, ratio);
    const dark = theme === "dark";
    ctx.fillStyle = dark ? "#101616" : "#f1f5f3";
    ctx.fillRect(0, 0, w, h);
    ctx.font = '10px "IBM Plex Mono", monospace';
    for (let l = 0; l < 61; l++) {
      const row = layers[String(l)];
      const known = !!row?.expert_ids;
      for (let e = 0; e < 192; e++) {
        const slot = row?.expert_ids?.indexOf(e) ?? -1;
        const value =
          mode === "score"
            ? row?.router_scores?.[e]
            : slot < 0
              ? 0
              : row?.expert_weights?.[slot];
        if (!known)
          ctx.fillStyle = dark
            ? l % 2
              ? "#1a2222"
              : "#161d1d"
            : l % 2
              ? "#e4ebe7"
              : "#e9efec";
        else if (!value) ctx.fillStyle = dark ? "#1c2d29" : "#d8e8df";
        else {
          const t = Math.max(0, Math.min(1, value / max));
          ctx.fillStyle = dark
            ? `rgb(${Math.round(40 + 122 * t)},${Math.round(84 + 153 * t)},${Math.round(66 + 109 * t)})`
            : `rgb(${Math.round(180 - 157 * t)},${Math.round(217 - 103 * t)},${Math.round(197 - 112 * t)})`;
        }
        ctx.fillRect(39 + e * 5, 22 + l * 6, 4, 5);
      }
      if (l % 5 === 0 || l === 60) {
        ctx.fillStyle = dark ? "#a8bcb4" : "#40584c";
        ctx.fillText(String(l).padStart(2, "0"), 12, 27 + l * 6);
      }
    }
    ctx.strokeStyle = dark ? "#e2eae5" : "#1a3325";
    ctx.lineWidth = 1;
    ctx.strokeRect(37.5, 20.5 + layer * 6, 963, 8);
    ctx.fillStyle = dark ? "#a8bcb4" : "#40584c";
    for (let e = 0; e < 192; e += 16) ctx.fillText(String(e), 39 + e * 5, 12);
    ctx.fillText("191", 978, 12);
  }, [layers, layer, max, mode, theme]);
  function pointer(event: React.MouseEvent<HTMLCanvasElement>, select = false) {
    const r = event.currentTarget.getBoundingClientRect();
    const e = Math.floor(
      (((event.clientX - r.left) / r.width) * 1008 - 39) / 5,
    );
    const l = Math.floor((((event.clientY - r.top) / r.height) * 394 - 22) / 6);
    if (e < 0 || e >= 192 || l < 0 || l > 60) return;
    const row = layers[String(l)];
    const slot = row?.expert_ids?.indexOf(e) ?? -1;
    const value =
      mode === "score"
        ? row?.router_scores?.[e]
        : row?.expert_ids
          ? slot < 0
            ? 0
            : row.expert_weights?.[slot]
          : undefined;
    setHover({ layer: l, expert: e, value });
    if (select) onLayer(l);
  }
  return (
    <section className="instrument matrix-panel" aria-labelledby="router-title">
      <div className="panel-heading">
        <h2 id="router-title">
          Router {mode === "score" ? "scores" : "weights"}
        </h2>
        <div className="segmented" aria-label="Router metric">
          <button
            aria-pressed={mode === "weight"}
            onClick={() => setMode("weight")}
          >
            Top-k
          </button>
          <button
            aria-pressed={mode === "score"}
            onClick={() => setMode("score")}
          >
            Sigmoid
          </button>
        </div>
        <span className="readout">{observed}/58 MoE</span>
      </div>
      <div className="matrix-axis">
        <span>Layer</span>
        <span>Logical expert ID</span>
        <span>{frame?.phase || "No capture"}</span>
      </div>
      <canvas
        ref={canvas}
        className="router-matrix"
        role="img"
        aria-label={`Router ${mode} matrix, 61 layers by 192 logical experts; ${observed} captured sparse layers. Select layer with the Layer control.`}
        onPointerMove={pointer}
        onPointerLeave={() => setHover(null)}
        onClick={(event) => pointer(event, true)}
      />
      <div className="graph-foot">
        <span>
          {hover
            ? `L${hover.layer} / E${hover.expert} / ${number(hover.value, 6)}`
            : `p${frame?.position ?? "·"} / ${frame?.projection || "No capture"}`}
        </span>
        <span className="legend">
          <i />0 <b /> {values.length ? number(max, 4) : "·"}
        </span>
      </div>
    </section>
  );
}
