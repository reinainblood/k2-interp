import { useEffect, useRef, useState } from "react";
import type { Frame, Session } from "./types";
import { terminal } from "./types";
import { Settings, defaults } from "./Settings";

export function Chat({
  session,
  frame,
  runId,
  onSelect,
  onSend,
  onCancel,
  onSystem,
  sending,
}: {
  session: Session | null;
  frame: Frame | null;
  runId: string;
  onSelect: (id: string) => void;
  onSend: (body: unknown) => Promise<boolean | undefined>;
  onCancel: () => Promise<void>;
  onSystem: (s: string) => Promise<void>;
  sending: boolean;
}) {
  const [text, setText] = useState("");
  const [settings, setSettings] = useState(defaults);
  useEffect(() => {
    const stored = session?.turns.find(
      (turn) => turn.run_id === runId,
    )?.settings;
    setSettings(
      stored
        ? {
            ...defaults,
            ...stored.generation,
            reasoning_effort: stored.reasoning_effort,
            max_context: stored.max_context,
            layers: stored.layers.join(","),
            expert_norms: stored.expert_norms,
          }
        : defaults,
    );
  }, [session?.id, runId]);
  const [error, setError] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const bottom = useRef<HTMLDivElement>(null);
  const scrollPane = useRef<HTMLDivElement>(null);
  const active =
    session?.turns.some((t) => !terminal(t.status)) ||
    (frame && !terminal(frame.status)) ||
    sending;
  useEffect(() => {
    if (autoScroll && scrollPane.current)
      scrollPane.current.scrollTop = scrollPane.current.scrollHeight;
  }, [frame?.output_ids?.length, session?.turns.length, autoScroll]);
  async function submit() {
    if (!text.trim() || active || !session) return;
    const layers = settings.layers
      .split(",")
      .filter((x) => x.trim())
      .map(Number);
    if (
      layers.some((l) => !Number.isInteger(l) || l < 0 || l > 60) ||
      new Set(layers).size !== layers.length
    ) {
      setError("Invalid layer indices");
      return;
    }
    const {
      max_new_tokens,
      temperature,
      top_p,
      seed,
      reasoning_effort,
      max_context,
      expert_norms,
    } = settings;
    if (
      await onSend({
        text,
        request_id: crypto.randomUUID(),
        generation: { max_new_tokens, temperature, top_p, seed },
        layers,
        reasoning_effort,
        max_context,
        expert_norms,
      })
    ) {
      setText("");
      setError("");
      setAutoScroll(true);
    }
  }
  return (
    <aside className="chat-column" aria-label="Conversation">
      <div className="chat-heading">
        <h2>Conversation</h2>
        <span>{session?.turns.length || 0} turns</span>
        <button
          aria-pressed={autoScroll}
          onClick={() => setAutoScroll((v) => !v)}
        >
          Scroll
        </button>
      </div>
      <div className="chat-scroll" ref={scrollPane}>
        {session?.turns.map((turn, i) => {
          const live = turn.run_id === frame?.run_id ? frame : null;
          const message =
            turn.assistant ||
            live?.result?.assistant_message ||
            live?.assistant_message;
          const status = live?.status || turn.status;
          return (
            <article
              key={turn.run_id}
              className={`chat-turn ${runId === turn.run_id ? "selected-turn" : ""}`}
            >
              <button
                className="turn-link"
                onClick={() => onSelect(turn.run_id)}
                aria-pressed={runId === turn.run_id}
              >
                <span>Turn {String(i + 1).padStart(2, "0")}</span>
                <span>
                  {new Date(turn.created_at).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </span>
                <span>{status}</span>
              </button>
              <div className="message user-message">
                <span className="role">user</span>
                <p>{turn.user}</p>
              </div>
              <div className="message assistant-message">
                <span className="role">K2</span>
                {message?.reasoning_content && (
                  <details className="reasoning">
                    <summary>Reasoning</summary>
                    <pre>{message.reasoning_content}</pre>
                  </details>
                )}
                <p>{message?.content || (!terminal(status) ? "···" : "")}</p>
                {turn.finish_reason === "length" && (
                  <span className="truncation">length</span>
                )}
                {(live?.error || turn.error) && (
                  <output className="error">
                    {(live?.error || turn.error)?.message}
                  </output>
                )}
              </div>
            </article>
          );
        })}
        {!session?.turns.length && (
          <div className="empty-conversation">
            <span>∅</span>
            <span>No turns</span>
          </div>
        )}
        <div ref={bottom} />
      </div>
      {session && (
        <Settings
          key={session.id}
          value={settings}
          onChange={setSettings}
          system={session.system}
          onSystem={onSystem}
          disabled={!!active}
        />
      )}
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <label htmlFor="message">Message</label>
        <textarea
          id="message"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={!session || !!active}
          rows={3}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              void submit();
            }
          }}
        />
        {error && (
          <output role="alert" className="error">
            {error}
          </output>
        )}
        <div className="composer-actions">
          <span>⌘ ↵</span>
          {active ? (
            <button
              type="button"
              onClick={() => void onCancel().catch((e) => setError(String(e)))}
            >
              Stop
            </button>
          ) : (
            <button
              className="primary"
              disabled={!session || !text.trim() || sending}
              type="submit"
            >
              Send
            </button>
          )}
        </div>
      </form>
    </aside>
  );
}
