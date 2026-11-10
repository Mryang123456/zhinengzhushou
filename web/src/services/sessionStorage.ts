import type { Message, Session } from "../types";
import { accessSessionSnapshot } from "./sessionDatabase";

const LIMIT = 40;
const INTERRUPTED_NOTICE = "回答已中断，可以重新发送。";

function restoreInterruptedReply(message: Message): Message {
  if (message.pending) {
    return { ...message, pending: false, notice: INTERRUPTED_NOTICE, retryable: !message.approval };
  }
  if (message.notice === INTERRUPTED_NOTICE && message.retryable === undefined && !message.approval) {
    return { ...message, retryable: true };
  }
  return message;
}

export function sessionStorageKey(apiUrl: string, userId: string | null): string {
  const scope = userId === null ? "guest" : `account:${encodeURIComponent(userId)}`;
  return `zhixing-agent:sessions:v2:${encodeURIComponent(apiUrl)}:${scope}`;
}

function readLegacySessions(key: string): Session[] {
  try {
    const value = JSON.parse(localStorage.getItem(key) ?? "[]");
    return Array.isArray(value) ? value as Session[] : [];
  } catch {
    return [];
  }
}

export async function loadSessions(key: string): Promise<Session[] | null> {
  let snapshot: Session[] | undefined;
  try {
    snapshot = await accessSessionSnapshot(key);
  } catch {
    // 存储不可用时，仍可读取此前按账号保存的记录。
  }
  try {
    const value = snapshot ?? readLegacySessions(key);
    if (!value.length) return null;
    const seen = new Map<string, number>();
    return value.map((session) => {
      const count = (seen.get(session.title) ?? 0) + 1;
      seen.set(session.title, count);
      return {
        ...session,
        title: count === 1 ? session.title : `${session.title}（${count}）`,
        messages: session.messages.map(restoreInterruptedReply),
      };
    });
  } catch {
    return null;
  }
}

function mergeSession(stored: Session, previous: Session | undefined, current: Session): Session {
  const priorMessages = new Map(previous?.messages.map((message) => [message.id, message]));
  const messages = new Map(stored.messages.map((message) => [message.id, message]));
  for (const message of current.messages) {
    if (priorMessages.get(message.id) !== message) messages.set(message.id, message);
  }
  return {
    ...stored,
    title: previous?.title !== current.title ? current.title : stored.title,
    archived: previous?.archived !== current.archived ? current.archived : stored.archived,
    updatedAt: Math.max(stored.updatedAt, current.updatedAt),
    messages: [...messages.values()],
  };
}

function prepareForStorage(session: Session): Session {
  return {
    ...session,
    messages: session.messages.slice(-100).map((entry) => {
      const message = restoreInterruptedReply(entry);
      if (!message.approval) return message;
      const { approval } = message;
      return {
        ...message,
        approval: {
          ...approval,
          confirmationToken: "",
          status: approval.status === "pending" || approval.status === "executing" ? "failed" : approval.status,
        },
      };
    }),
  };
}

export async function saveSessions(key: string, sessions: Session[], previousSessions: Session[] = []) {
  const previousById = new Map(previousSessions.map((session) => [session.id, session]));
  const changed = sessions.filter((session) => previousById.get(session.id) !== session);
  if (!changed.length) return;

  await accessSessionSnapshot(key, (snapshot) => {
    const storedSessions = snapshot ?? readLegacySessions(key);
    const byId = new Map(storedSessions.map((session) => [session.id, session]));
    for (const session of changed) {
      const stored = byId.get(session.id);
      byId.set(session.id, stored ? mergeSession(stored, previousById.get(session.id), session) : session);
    }
    return [...byId.values()]
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, LIMIT)
      .map(prepareForStorage);
  });
}
