import type { Session } from "../types";

const KEY = "zhixing-agent:sessions:v1";
const LIMIT = 40;

export function loadSessions(): Session[] | null {
  try { const value = JSON.parse(localStorage.getItem(KEY) ?? "null") as Session[] | null; if (!Array.isArray(value)) return null; const seen = new Map<string, number>(); return value.map((session) => { const count = (seen.get(session.title) ?? 0) + 1; seen.set(session.title, count); return count === 1 ? session : { ...session, title: `${session.title}（${count}）` }; }); } catch { return null; }
}

export function saveSessions(sessions: Session[]) {
  const safe = sessions.slice(0, LIMIT).map((session) => ({ ...session, messages: session.messages.slice(-100).map((message) => message.approval ? { ...message, approval: { ...message.approval, confirmationToken: "", status: message.approval.status === "pending" ? "failed" : message.approval.status } } : message) }));
  localStorage.setItem(KEY, JSON.stringify(safe));
}
