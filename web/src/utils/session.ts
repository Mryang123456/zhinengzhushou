import type { Session } from "../types";

export function createSessionTitle(now = new Date(), suffix = crypto.randomUUID().slice(0, 6)) {
  return `会话-${now.toISOString().replace(/[:.]/g, "-")}-${suffix}`;
}

export function archiveSessionState(sessions: Session[], activeId: string, archivedId: string, replacement: Session) {
  const archived = sessions.map((session) => session.id === archivedId ? { ...session, archived: true, updatedAt: Date.now() } : session);
  const available = archived.filter((session) => !session.archived);
  if (!available.length) return { sessions: [replacement, ...archived], activeId: replacement.id };
  return { sessions: archived, activeId: activeId === archivedId ? available[0].id : activeId };
}
