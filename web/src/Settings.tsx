import { useEffect, useState } from "react";
export type SettingsValue = {
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  seed: number;
  reasoning_effort: string;
  max_context: number;
  layers: string;
  expert_norms: boolean;
};
export const defaults: SettingsValue = {
  max_new_tokens: 512,
  temperature: 1.0,
  top_p: 0.95,
  seed: 0,
  reasoning_effort: "high",
  max_context: 16384,
  layers: "10,30,50",
  expert_norms: true,
};
export function Settings({
  value,
  onChange,
  system,
  onSystem,
  disabled,
}: {
  value: SettingsValue;
  onChange: (v: SettingsValue) => void;
  system: string;
  onSystem: (s: string) => Promise<void>;
  disabled: boolean;
}) {
  const [draft, setDraft] = useState(system);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => setDraft(system), [system]);
  const field = (key: keyof SettingsValue, n: number | string | boolean) =>
    onChange({ ...value, [key]: n });
  return (
    <details className="settings">
      <summary>System / Reasoning / Capture</summary>
      <div className="settings-body">
        <label>
          System prompt
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={6}
            disabled={disabled}
          />
        </label>
        <div className="system-actions">
          <span>{draft === system ? "Saved" : "Unsaved"}</span>
          <button
            disabled={disabled || saving || draft === system}
            onClick={async () => {
              setSaving(true);
              try {
                await onSystem(draft);
                setError("");
              } catch (e) {
                setError(String(e));
              } finally {
                setSaving(false);
              }
            }}
          >
            {saving ? "Saving" : "Apply system"}
          </button>
        </div>
        {error && (
          <output className="error" role="alert">
            {error}
          </output>
        )}
        <div className="settings-grid">
          <label>
            Reasoning effort
            <select
              value={value.reasoning_effort}
              onChange={(e) => field("reasoning_effort", e.target.value)}
              disabled={disabled}
            >
              <option>high</option>
              <option>medium</option>
              <option>low</option>
            </select>
          </label>
          <label>
            Output tokens
            <input
              type="number"
              min="1"
              max="8192"
              value={value.max_new_tokens}
              onChange={(e) => field("max_new_tokens", Number(e.target.value))}
              disabled={disabled}
            />
          </label>
          <label>
            Temperature
            <input
              type="number"
              min="0"
              max="5"
              step="0.05"
              value={value.temperature}
              onChange={(e) => field("temperature", Number(e.target.value))}
              disabled={disabled}
            />
          </label>
          <label>
            Top-p
            <input
              type="number"
              min="0.01"
              max="1"
              step="0.05"
              value={value.top_p}
              onChange={(e) => field("top_p", Number(e.target.value))}
              disabled={disabled}
            />
          </label>
          <label>
            Seed
            <input
              type="number"
              min="0"
              value={value.seed}
              onChange={(e) => field("seed", Number(e.target.value))}
              disabled={disabled}
            />
          </label>
          <label>
            Context tokens
            <input
              type="number"
              min="512"
              max="65536"
              step="512"
              value={value.max_context}
              onChange={(e) => field("max_context", Number(e.target.value))}
              disabled={disabled}
            />
          </label>
          <label>
            Residual layers
            <input
              value={value.layers}
              onChange={(e) => field("layers", e.target.value)}
              disabled={disabled}
            />
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={value.expert_norms}
              onChange={(e) => field("expert_norms", e.target.checked)}
              disabled={disabled}
            />
            Expert output L2
          </label>
        </div>
        <div className="settings-meta">
          BF16 / 2 GB capture bound / router L3:60
        </div>
      </div>
    </details>
  );
}
