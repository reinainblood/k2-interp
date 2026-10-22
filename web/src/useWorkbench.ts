import { useCallback, useEffect, useRef, useState } from "react";
import { api, turnPath } from "./api";
import type { Frame, Session, SessionItem, Turn } from "./types";
import { terminal } from "./types";

export function useWorkbench() {
  const [authorized, setAuthorized] = useState(false);
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [runId, setRunId] = useState("");
  const [frame, setFrame] = useState<Frame | null>(null);
  const [error, setError] = useState("");
  const [connection, setConnection] = useState("Connecting");
  const [sending, setSending] = useState(false);
  const source = useRef<EventSource | null>(null);
  const refreshSessions = useCallback(
    async () => setSessions(await api<SessionItem[]>("/sessions")),
    [],
  );
  const selectSession = useCallback(async (id: string) => {
    source.current?.close();
    setFrame(null);
    setError("");
    const s = await api<Session>(`/sessions/${id}`);
    setSession(s);
    setRunId(s.turns.at(-1)?.run_id || "");
    localStorage.setItem("k2-session", id);
  }, []);
  const boot = useCallback(async () => {
    const config = await api<{ checkpoints: string[] }>("/config");
    setCheckpoints(config.checkpoints);
    setAuthorized(true);
    setConnection("Connected");
    const list = await api<SessionItem[]>("/sessions");
    setSessions(list);
    const remembered = localStorage.getItem("k2-session");
    const id = list.find((s) => s.id === remembered)?.id || list[0]?.id;
    if (id) await selectSession(id);
  }, [selectSession]);
  const login = useCallback(
    async (ticket: string) => {
      try {
        await api("/login", { ticket });
        await boot();
        setError("");
      } catch (e) {
        setError(String(e));
        setConnection("Unauthenticated");
      }
    },
    [boot],
  );
  useEffect(() => {
    const ticket = new URLSearchParams(location.hash.slice(1)).get("ticket");
    if (ticket) {
      history.replaceState(null, "", location.pathname);
      void login(ticket);
    } else void boot().catch(() => setConnection("Unauthenticated"));
    return () => source.current?.close();
  }, [boot, login]);
  useEffect(() => {
    source.current?.close();
    setFrame(null);
    if (!session || !runId) return;
    let cancelled = false;
    const url = turnPath(session.id, runId);
    const accept = (value: Frame) => {
      if (cancelled) return;
      setFrame(value);
      setConnection("Connected");
      if (terminal(value.status)) {
        source.current?.close();
        void api<Session>(`/sessions/${session.id}`).then((s) => {
          if (!cancelled) setSession(s);
        });
        void refreshSessions();
      }
    };
    void api<Frame>(url + "/state")
      .then((value) => {
        accept(value);
        if (cancelled || terminal(value.status)) return;
        const events = new EventSource("/api" + url + "/events");
        source.current = events;
        events.onmessage = (e) => accept(JSON.parse(e.data));
        events.onerror = () => {
          if (!cancelled) setConnection("Reconnecting");
        };
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
      source.current?.close();
    };
  }, [session?.id, runId, refreshSessions]);
  const create = async (checkpoint: string, system: string) => {
    const s = await api<Session>("/sessions", { checkpoint, system });
    setSession(s);
    setRunId("");
    setFrame(null);
    localStorage.setItem("k2-session", s.id);
    await refreshSessions();
    return s;
  };
  const updateSystem = async (system: string) => {
    if (!session) return;
    const s = await api<Session>(`/sessions/${session.id}/settings`, {
      checkpoint: session.checkpoint,
      system,
    });
    setSession(s);
  };
  const send = async (body: unknown) => {
    if (!session) return;
    setSending(true);
    setError("");
    try {
      const turn = await api<Turn>(`/sessions/${session.id}/turns`, body);
      setSession((s) =>
        s
          ? {
              ...s,
              turns: [...s.turns.filter((t) => t.run_id !== turn.run_id), turn],
            }
          : s,
      );
      setRunId(turn.run_id);
      await refreshSessions();
      return true;
    } catch (e) {
      setError(String(e));
      return false;
    } finally {
      setSending(false);
    }
  };
  const cancel = async () => {
    const active = session?.turns.find((t) => !terminal(t.status));
    if (session && active)
      await api(turnPath(session.id, active.run_id) + "/cancel", {});
  };
  return {
    authorized,
    checkpoints,
    sessions,
    session,
    runId,
    frame,
    error,
    connection,
    sending,
    setError,
    setRunId,
    setSession,
    refreshSessions,
    selectSession,
    create,
    send,
    cancel,
    login,
    updateSystem,
  };
}
