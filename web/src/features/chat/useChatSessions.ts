import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { saveSessions } from "../../services/sessionStorage";
import type { Message, Session } from "../../types";
import { archiveSessionState, createSessionTitle, ensureVisibleSession, resolveVisibleSession } from "../../utils/session";

const welcome = "你好，我是智行助手。你可以让我查询天气、搜索资料、计算表达式，或在需要时明确告诉我“记住这件事”。";

function newSession(title = `会话-${new Date().toISOString().slice(0, 10)}`): Session {
  return {
    id: crypto.randomUUID(),
    title,
    messages: [{ id: crypto.randomUUID(), role: "assistant", content: welcome }],
    updatedAt: Date.now(),
  };
}

export type UpdateMessages = (sessionId: string, update: (messages: Message[]) => Message[]) => void;

export function useChatSessions(storageKey: string, savedSessions: Session[] | null) {
  const [sessions, setSessions] = useState(() => ensureVisibleSession(savedSessions ?? [], newSession));
  const savedSnapshot = useRef(sessions);
  const saveQueue = useRef(Promise.resolve());
  const [selectedId, selectSession] = useState<string>();
  const [drafts, setDrafts] = useState<ReadonlyMap<string, string>>(() => new Map());
  const [saveFailed, setSaveFailed] = useState(false);
  // 初始化与归档操作都保证至少保留一个可用会话。
  const activeSession = useMemo(() => resolveVisibleSession(sessions, selectedId)!, [sessions, selectedId]);
  const visibleSessions = useMemo(
    () => sessions.filter((session) => !session.archived).sort((a, b) => b.updatedAt - a.updatedAt),
    [sessions],
  );

  useEffect(() => {
    saveQueue.current = saveQueue.current.then(async () => {
      if (savedSnapshot.current === sessions) return;
      try {
        await saveSessions(storageKey, sessions, savedSnapshot.current);
        savedSnapshot.current = sessions;
        setSaveFailed(false);
      } catch {
        setSaveFailed(true);
      }
    });
  }, [storageKey, sessions]);

  const updateMessages = useCallback<UpdateMessages>((sessionId, update) => {
    setSessions((items) => items.map((item) => item.id === sessionId
      ? { ...item, messages: update(item.messages), updatedAt: Date.now() }
      : item));
  }, []);

  const updateDraft = (sessionId: string, update: (current: string) => string) => {
    setDrafts((current) => {
      const value = update(current.get(sessionId) ?? "");
      const next = new Map(current);
      if (value) next.set(sessionId, value);
      else next.delete(sessionId);
      return next;
    });
  };

  const createSession = () => {
    const session = newSession(createSessionTitle());
    setSessions((items) => [session, ...items]);
    selectSession(session.id);
  };

  const renameSession = (title: string) => setSessions((items) => items.map((item) => item.id === activeSession.id
    ? { ...item, title, updatedAt: Date.now() }
    : item));

  const archiveSession = (id: string) => {
    const next = archiveSessionState(sessions, activeSession.id, id, newSession(createSessionTitle()));
    setSessions(next.sessions);
    selectSession(next.activeId);
    updateDraft(id, () => "");
  };

  return {
    activeSession,
    visibleSessions,
    draft: drafts.get(activeSession.id) ?? "",
    saveFailed,
    updateMessages,
    selectSession,
    createSession,
    renameSession,
    archiveSession,
    setDraft: (value: string) => updateDraft(activeSession.id, () => value),
    appendPrompt: (template: string) => updateDraft(activeSession.id, (current) => current.trim() ? `${current}\n${template}` : template),
  };
}
