import type { Frame, Layer } from "./types";
import { number } from "./types";

export function ExpertGraph({
  value,
  layer,
}: {
  value?: Layer;
  layer: number;
}) {
  const ids = value?.expert_ids || [];
  const weights = value?.expert_weights || [];
  const max = Math.max(...weights, 0.0001);
  return (
    <section className="instrument expert-panel" aria-labelledby="expert-title">
      <div className="panel-heading">
        <h2 id="expert-title">Selected experts</h2>
        <span className="readout">
          L{layer} / k={ids.length || "·"}
        </span>
      </div>
      {!ids.length ? (
        <div className="empty-plot">
          {layer < 3 ? "Dense MLP" : "No capture"}
        </div>
      ) : (
        <>
          <div className="expert-columns">
            <span>Expert</span>
            <span>Routing weight</span>
            <span>Weighted output L2</span>
          </div>
          <div className="expert-rows">
            {ids.map((id, i) => (
              <div className="expert-row" key={id}>
                <span className="expert-id">
                  E{String(id).padStart(3, "0")}
                </span>
                <div className="weight-measure">
                  <div
                    className="weight-bar"
                    style={{ width: `${(weights[i] / max) * 100}%` }}
                  />
                  <span>{number(weights[i], 5)}</span>
                </div>
                <span className="expert-norm">
                  {number(value?.expert_norms?.[String(id)], 3)}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
      <div className="graph-foot">
        <span>
          Σw{" "}
          {number(
            weights.length ? weights.reduce((a, b) => a + b, 0) : undefined,
            4,
          )}
        </span>
        <span>Δk {number(value?.routing_margin, 6)}</span>
        <span>Hσ {number(value?.router_entropy, 4)}</span>
      </div>
    </section>
  );
}

export function ResidualGraph({
  frame,
  selectedLayer,
  onLayer,
}: {
  frame: Frame | null;
  selectedLayer: number;
  onLayer: (l: number) => void;
}) {
  const points = Object.entries(frame?.layers || {})
    .filter(([, v]) => v.norms?.resid_post != null)
    .map(([l, v]) => [Number(l), v.norms!.resid_post])
    .sort((a, b) => a[0] - b[0]);
  const max = Math.max(...points.map((p) => p[1]), 1);
  return (
    <section className="instrument residual-panel">
      <div className="panel-heading">
        <h2>Residual L2</h2>
        <span className="readout">resid_post</span>
      </div>
      {!points.length ? (
        <div className="empty-plot">No capture</div>
      ) : (
        <svg
          viewBox="0 0 460 180"
          role="img"
          aria-label={`Residual norms at ${points.length} captured layers`}
        >
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <line
                className="grid-line"
                x1="48"
                x2="442"
                y1={140 - t * 110}
                y2={140 - t * 110}
              />
              <text className="axis" x="4" y={144 - t * 110}>
                {number(t * max, 0)}
              </text>
            </g>
          ))}
          {[0, 10, 20, 30, 40, 50, 60].map((l) => (
            <text
              className="axis"
              key={l}
              x={48 + (l / 60) * 390}
              y="165"
              textAnchor="middle"
            >
              {l}
            </text>
          ))}
          {points.map(([l, v]) => (
            <g key={l}>
              <line
                className="stem"
                x1={48 + (l / 60) * 390}
                x2={48 + (l / 60) * 390}
                y1="140"
                y2={140 - (v / max) * 110}
              />
              <circle
                className={
                  l === selectedLayer ? "point selected-point" : "point"
                }
                cx={48 + (l / 60) * 390}
                cy={140 - (v / max) * 110}
                r="4"
              />
              <title>{`L${l}: ${v}`}</title>
            </g>
          ))}
        </svg>
      )}
      <div className="graph-foot">
        <span>Layer</span>
        <div className="layer-pills">
          {points.map(([l, v]) => (
            <button
              key={l}
              aria-pressed={l === selectedLayer}
              onClick={() => onLayer(l)}
            >
              L{l} {number(v, 0)}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

export function LoadGraph({ value, phase }: { value?: Layer; phase?: string }) {
  const counts = value?.forward_expert_counts || [];
  const max = Math.max(...counts, 1);
  return (
    <section className="instrument load-panel">
      <div className="panel-heading">
        <h2>Forward expert load</h2>
        <span className="readout">{phase || "·"}</span>
      </div>
      {!counts.length ? (
        <div className="empty-plot">No aggregate capture</div>
      ) : (
        <svg
          viewBox="0 0 460 180"
          role="img"
          aria-label="Exact expert selection counts across this forward's input positions"
        >
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <line
                className="grid-line"
                x1="34"
                x2="448"
                y1={140 - t * 110}
                y2={140 - t * 110}
              />
              <text className="axis" x="3" y={144 - t * 110}>
                {number(t * max, 0)}
              </text>
            </g>
          ))}
          {counts.map((v, e) => (
            <rect
              className="data-bar"
              key={e}
              x={34 + (e / counts.length) * 414}
              y={140 - (v / max) * 110}
              width={(414 / counts.length) * 0.8}
              height={(v / max) * 110}
            >
              <title>{`E${e}: ${v}`}</title>
            </rect>
          ))}
          {[0, 48, 96, 144, 191].map((e) => (
            <text
              key={e}
              className="axis"
              x={34 + (e / 192) * 414}
              y="165"
              textAnchor="middle"
            >
              {e}
            </text>
          ))}
        </svg>
      )}
      <div className="graph-foot">
        <span>Logical expert ID</span>
        <span>
          {counts.length
            ? `${counts.reduce((a, b) => a + b, 0)} selections`
            : "·"}
        </span>
      </div>
    </section>
  );
}

export function LogprobGraph({ frame }: { frame: Frame | null }) {
  const values = frame?.output_logprobs || [];
  const floor = Math.min(-1, ...values);
  return (
    <section className="instrument logprob-panel">
      <div className="panel-heading">
        <h2>Output log probability</h2>
        <span className="readout">nats / {values.length} tokens</span>
      </div>
      {!values.length ? (
        <div className="empty-plot compact">No output</div>
      ) : (
        <svg
          viewBox="0 0 1000 90"
          role="img"
          aria-label="Sampled output token log probabilities in nats"
        >
          <line className="grid-line" x1="40" x2="990" y1="12" y2="12" />
          <text className="axis" x="6" y="16">
            0
          </text>
          <text className="axis" x="6" y="74">
            {number(floor, 1)}
          </text>
          {values.map((v, i) => (
            <line
              className="prob-line"
              key={i}
              x1={40 + (i / Math.max(values.length - 1, 1)) * 950}
              x2={40 + (i / Math.max(values.length - 1, 1)) * 950}
              y1="12"
              y2={12 + (v / floor) * 60}
            >
              <title>{`Output ${i}: ${v}`}</title>
            </line>
          ))}
        </svg>
      )}
    </section>
  );
}
