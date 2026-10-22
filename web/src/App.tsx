import { useEffect, useRef, useState } from "react";
import { api, downloadCapture, turnPath } from "./api";
import { Chat } from "./Chat";
import {
  ExpertGraph,
  LoadGraph,
  LogprobGraph,
  ResidualGraph,
} from "./DetailGraphs";
import { RouterMatrix } from "./RouterMatrix";
import { TokenStrip } from "./TokenStrip";
import { useWorkbench } from "./useWorkbench";
import type { Frame, Session } from "./types";
import { number } from "./types";

export default function App() {
  const w = useWorkbench();
  const [theme, setTheme] = useState(
    localStorage.getItem("k2-theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"),
  );
  const [layer, setLayer] = useState(30);
  const [position, setPosition] = useState(0);
  const [follow, setFollow] = useState(true);
  const [inspection, setInspection] = useState<Frame | null>(null);
  const [inspectBusy, setInspectBusy] = useState(false);
  const [tag, setTag] = useState("");
  const [downloading, setDownloading] = useState(false);
  const [checkpoint, setCheckpoint] = useState("sft_3_1500");
  const [system, setSystem] = useState("");
  const [ticket, setTicket] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const pendingSelection = useRef<{
    runId: string;
    position: number;
    layer: number;
  } | null>(null);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("k2-theme", theme);
  }, [theme]);
  useEffect(() => {
    const saved = pendingSelection.current;
    if (saved?.runId === w.runId) {
      setFollow(false);
      setPosition(saved.position);
      setLayer(saved.layer);
      pendingSelection.current = null;
    } else {
      setFollow(true);
      setPosition(0);
    }
    setInspection(null);
  }, [w.runId]);
  useEffect(() => {
    if (follow && w.frame?.position != null) setPosition(w.frame.position);
  }, [w.frame?.position, follow]);
  useEffect(() => {
    if (follow || !w.session || !w.runId) return;
    const controller = new AbortController();
    setInspectBusy(true);
    setInspection(null);
    const timer = setTimeout(() => {
      void api<Frame>(
        turnPath(w.session!.id, w.runId) + `/tokens/${position}`,
        undefined,
        controller.signal,
      )
        .then((value) => {
          setInspection(value);
          w.setError("");
        })
        .catch((e) => {
          if (!controller.signal.aborted) w.setError(String(e));
        })
        .finally(() => {
          if (!controller.signal.aborted) setInspectBusy(false);
        });
    }, 100);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [w.session?.id, w.runId, position, follow]);
  const view = follow ? w.frame : inspection;
  async function capture() {
    if (!w.session || !w.runId || !view) return;
    setDownloading(true);
    w.setError("");
    const captureSession = w.session.id;
    try {
      await downloadCapture(turnPath(w.session.id, w.runId), {
        position,
        layer,
        tag,
      });
      const updated = await api<Session>(`/sessions/${captureSession}`);
      w.setSession((current) =>
        current?.id === captureSession ? updated : current,
      );
      setTag("");
    } catch (e) {
      w.setError(String(e));
    } finally {
      setDownloading(false);
    }
  }
  const choose = (p: number) => {
    setPosition(p);
    setFollow(false);
  };
  const shownLayers = Object.keys(view?.layers || {}).length;
  return (
    <div className="shell">
      <header className="app-header">
        <a className="brand" href="/">
          K2<span>/</span>INTERP
        </a>
        <span className="environment">FDR</span>
        <span
          className={`connection ${w.connection === "Connected" ? "connected" : ""}`}
          role="status"
        >
          {w.connection}
        </span>
        <div className="header-actions">
          <button
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? "Light" : "Dark"}
          </button>
          {w.authorized && (
            <button
              onClick={() =>
                void api("/logout", {}).then(() => location.reload())
              }
            >
              Lock
            </button>
          )}
        </div>
      </header>
      {!w.authorized ? (
        <main className="login">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void w.login(ticket);
            }}
          >
            <h1>Access</h1>
            <label>
              Access ticket
              <input
                type="password"
                autoComplete="off"
                value={ticket}
                onChange={(e) => setTicket(e.target.value)}
              />
            </label>
            <button className="primary" type="submit">
              Connect
            </button>
            <code>uv run python scripts/open_workbench.py</code>
            {w.error && (
              <output className="error" role="alert">
                {w.error}
              </output>
            )}
          </form>
        </main>
      ) : (
        <>
          <div className="session-bar">
            <label>
              Session
              <select
                aria-label="Session"
                value={w.session?.id || ""}
                onChange={(e) =>
                  void w
                    .selectSession(e.target.value)
                    .catch((e) => w.setError(String(e)))
                }
              >
                <option value="" disabled>
                  ∅
                </option>
                {w.sessions.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.title || "Empty session"} / {s.turns}
                  </option>
                ))}
              </select>
            </label>
            <button onClick={() => dialog.current?.showModal()}>New</button>
            <span className="checkpoint-label">
              {w.session?.checkpoint || "·"}
            </span>
            <span className="session-id">{w.session?.id.slice(-12) || ""}</span>
            {w.session && (
              <a
                className="download-chat"
                href={`/api/sessions/${w.session.id}/export`}
                download
              >
                Chat ↓
              </a>
            )}
          </div>
          {w.error && (
            <div className="global-error" role="alert">
              <span>{w.error}</span>
              <button aria-label="Dismiss error" onClick={() => w.setError("")}>
                ×
              </button>
            </div>
          )}
          <main className="workspace">
            <Chat
              session={w.session}
              frame={w.frame}
              runId={w.runId}
              onSelect={w.setRunId}
              onSend={w.send}
              onCancel={w.cancel}
              onSystem={w.updateSystem}
              sending={w.sending}
            />
            <div className="observatory">
              <div className="capture-toolbar">
                <div className="phase">
                  <span
                    className={
                      w.frame &&
                      !["complete", "failed", "cancelled"].includes(
                        w.frame.status,
                      )
                        ? "live-mark"
                        : ""
                    }
                  />
                  {w.frame?.status || "Idle"}
                </div>
                <span className="readout">
                  {inspectBusy ? "Reading" : `step ${view?.step ?? "·"}`}
                </span>
                <span className="readout">
                  {number(
                    w.frame?.tensor_bytes ? w.frame.tensor_bytes / 1e6 : 0,
                    1,
                  )}{" "}
                  MB
                </span>
                <label>
                  Layer
                  <input
                    type="number"
                    value={layer}
                    min="0"
                    max="60"
                    onChange={(e) =>
                      setLayer(
                        Math.min(60, Math.max(0, Number(e.target.value))),
                      )
                    }
                  />
                </label>
                <input
                  className="tag-input"
                  aria-label="Capture tag"
                  placeholder="tag"
                  value={tag}
                  maxLength={120}
                  onChange={(e) => setTag(e.target.value)}
                />
                <button
                  className="capture-button"
                  disabled={
                    view?.position == null || downloading || inspectBusy
                  }
                  onClick={() => void capture()}
                >
                  {downloading ? "Packaging" : "Capture ↓"}
                </button>
              </div>
              <TokenStrip
                frame={w.frame}
                selected={position}
                onSelect={choose}
                follow={follow}
                onFollow={() => setFollow((v) => !v)}
              />
              <RouterMatrix
                frame={view}
                layer={layer}
                onLayer={setLayer}
                theme={theme}
              />
              <div className="plot-grid">
                <ExpertGraph
                  value={view?.layers?.[String(layer)]}
                  layer={layer}
                />
                <ResidualGraph
                  frame={view}
                  selectedLayer={layer}
                  onLayer={setLayer}
                />
                <LoadGraph
                  value={view?.layers?.[String(layer)]}
                  phase={view?.phase}
                />
              </div>
              <LogprobGraph frame={w.frame} />
              <div className="metadata-grid">
                <details className="instrument provenance">
                  <summary>
                    Run provenance <span>{shownLayers} layers</span>
                  </summary>
                  <dl>
                    <dt>Run</dt>
                    <dd>{w.runId || "·"}</dd>
                    <dt>Checkpoint</dt>
                    <dd>{w.session?.checkpoint || "·"}</dd>
                    <dt>Updated UTC</dt>
                    <dd>{w.frame?.updated_at || "·"}</dd>
                    <dt>Committed UTC</dt>
                    <dd>{w.frame?.committed_at || "·"}</dd>
                    <dt>Transport errors</dt>
                    <dd>{w.frame?.transport_errors ?? 0}</dd>
                  </dl>
                  <pre>{JSON.stringify(w.frame?.identity || {}, null, 2)}</pre>
                </details>
                <details className="instrument captures">
                  <summary>
                    Captures <span>{w.session?.captures.length || 0}</span>
                  </summary>
                  {w.session?.captures.map((c) => (
                    <div className="capture-entry" key={c.capture_id}>
                      <button
                        onClick={() => {
                          if (w.runId === c.run_id) {
                            choose(c.position);
                            setLayer(c.layer);
                          } else {
                            pendingSelection.current = {
                              runId: c.run_id,
                              position: c.position,
                              layer: c.layer,
                            };
                            w.setRunId(c.run_id);
                          }
                        }}
                      >
                        {c.tag || `p${c.position} / L${c.layer}`}
                      </button>
                      <time>{c.created_at}</time>
                    </div>
                  ))}
                </details>
              </div>
            </div>
          </main>
          <dialog ref={dialog} className="session-dialog">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void w
                  .create(checkpoint, system)
                  .then(() => dialog.current?.close())
                  .catch((e) => w.setError(String(e)));
              }}
            >
              <div className="dialog-heading">
                <h2>Session</h2>
                <button
                  type="button"
                  aria-label="Close session dialog"
                  onClick={() => dialog.current?.close()}
                >
                  ×
                </button>
              </div>
              <label>
                Checkpoint
                <select
                  value={checkpoint}
                  onChange={(e) => setCheckpoint(e.target.value)}
                >
                  {w.checkpoints.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </label>
              <label>
                System prompt
                <textarea
                  rows={6}
                  value={system}
                  onChange={(e) => setSystem(e.target.value)}
                />
              </label>
              <button className="primary" type="submit">
                Create
              </button>
            </form>
          </dialog>
        </>
      )}
    </div>
  );
}
