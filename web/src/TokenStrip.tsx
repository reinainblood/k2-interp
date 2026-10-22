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
  const listRef = useRef<HTMLDivElement>(null);
  const [windowStart, setWindowStart] = useState(0);
  const [query, setQuery] = useState("");
  const tokens = frame?.token_text || [];
  const ids = [...(frame?.input_ids || []), ...(frame?.output_ids || [])];
  useEffect(() => {
    if (follow) setWindowStart(Math.max(0, selected - 90));
    else
      setWindowStart((start) =>
        selected < start || selected >= start + 180
          ? Math.max(0, selected - 40)
          : start,
      );
  }, [selected, follow]);
  useEffect(() => {
    const list = listRef.current,
      token = ref.current;
    if (list && token) {
      const delta =
        token.getBoundingClientRect().top - list.getBoundingClientRect().top;
      if (delta < 0 || delta + token.offsetHeight > list.clientHeight)
        list.scrollTop += delta - 24;
    }
  }, [selected]);
  const start = Math.min(windowStart, Math.max(0, tokens.length - 1));
  const matches = query
    ? tokens.flatMap((text, p) =>
        tokenLabel(text).toLowerCase().includes(query.toLowerCase()) ? [p] : [],
      )
    : [];
  function findNext() {
    const next = matches.find((p) => p > selected) ?? matches[0];
    if (next != null) {
      setWindowStart(Math.max(0, next - 40));
      onSelect(next);
    }
  }
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
      <div className="token-search">
        <label>
          Find token
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") findNext();
            }}
          />
        </label>
        <span className="readout">
          {query ? `${matches.length} matches` : "·"}
        </span>
        <button disabled={!matches.length} onClick={findNext}>
          Next
        </button>
      </div>
      <div
        className="token-list"
        ref={listRef}
        aria-label="Captured input token positions"
      >
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
