import { useEffect, useRef, useState } from "react";
import type { Frame } from "./types";
import { tokenLabel } from "./types";

export function TokenStrip({
  frame,
  selected,
  onSelect,
  follow,
  onFollow,
}: {
  frame: Frame | null;
  selected: number;
  onSelect: (p: number) => void;
  follow: boolean;
  onFollow: () => void;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const [windowStart, setWindowStart] = useState(0);
  const tokens = frame?.token_text || [];
  const ids = [...(frame?.input_ids || []), ...(frame?.output_ids || [])];
  useEffect(() => {
    if (follow) setWindowStart(Math.max(0, selected - 90));
  }, [selected, follow]);
  useEffect(() => {
    ref.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [selected]);
  const start = Math.min(windowStart, Math.max(0, tokens.length - 1));
  return (
    <section className="instrument token-panel">
      <div className="panel-heading">
        <h2>Token stream</h2>
        <span className="readout">
          {frame?.input_tokens ?? 0} in / {frame?.output_ids?.length ?? 0} out
        </span>
        <button aria-pressed={follow} onClick={onFollow}>
          Follow
        </button>
      </div>
      <div className="token-controls">
        <label>
          Position{" "}
          <input
            type="number"
            min="0"
            max={Math.max(0, tokens.length - 1)}
            value={selected}
            onChange={(e) => {
              const p = Number(e.target.value);
              setWindowStart(Math.max(0, p - 40));
              onSelect(p);
            }}
          />
        </label>
        <input
          aria-label="Token position"
          type="range"
          min="0"
          max={Math.max(0, tokens.length - 1)}
          value={selected}
          onChange={(e) => {
            const p = Number(e.target.value);
            setWindowStart(Math.max(0, p - 40));
            onSelect(p);
          }}
        />
        <span className="readout">
          ID{" "}
          {ids[selected] ??
            (frame?.position === selected
              ? frame?.current_token_ids?.[0]
              : undefined) ??
            "·"}
        </span>
      </div>
      <div className="token-list" aria-label="Captured input token positions">
        {!tokens.length && <span className="empty-inline">No capture</span>}
        {start > 0 && (
          <button
            className="token-page"
            onClick={() => setWindowStart(Math.max(0, start - 120))}
          >
            ← {start}
          </button>
        )}
        {tokens.slice(start, start + 180).map((token, i) => {
          const p = i + start;
          return (
            <button
              ref={p === selected ? ref : undefined}
              className={`token ${p >= (frame?.input_tokens ?? Infinity) ? "output-token" : ""}`}
              aria-pressed={p === selected}
              title={`p${p} / ID ${ids[p] ?? "?"} / ${token}`}
              key={p}
              onClick={() => onSelect(p)}
            >
              <small>{p}</small>
              {tokenLabel(token) || "∅"}
            </button>
          );
        })}
        {start + 180 < tokens.length && (
          <button
            className="token-page"
            onClick={() => setWindowStart(start + 120)}
          >
            {tokens.length - start - 180} →
          </button>
        )}
      </div>
      <div className="graph-foot">
        <span>Input position i → logits i+1</span>
        <span>
          {frame?.phase === "prefill"
            ? "Parallel prefill"
            : frame?.phase || "·"}
        </span>
      </div>
    </section>
  );
}
