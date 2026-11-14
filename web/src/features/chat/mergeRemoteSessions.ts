import type { Message, Session } from "../../types";

export function mergeRemoteSessions(local: Session[], remote: Session[]): Session[] {
  const merged = new Map(local.map((session) => [session.id, session]));
  for (const server of remote) {
    const current = merged.get(server.id);
    if (!current) {
      merged.set(server.id, server);
      continue;
    }
    const localIds = new Set(current.messages.map((message) => message.id));
    const missing: Message[] = [];
    for (let index = 0; index < server.messages.length; index += 2) {
      const question = server.messages[index];
      const answer = server.messages[index + 1];
      if (question?.role === "user" && answer?.role === "assistant" && !localIds.has(question.id)) {
        missing.push(question, answer);
      }
    }
    if (missing.length) {
      const messages = [...current.messages];
      // 保持本地待恢复的最后一轮仍在末尾，供现有断流恢复逻辑接续。
      const pending = messages.at(-1);
      const insertion = pending?.pending || pending?.resume || pending?.submission || pending?.retryable
        ? Math.max(0, messages.length - 2) : messages.length;
      messages.splice(insertion, 0, ...missing);
      merged.set(server.id, { ...current, messages, updatedAt: Math.max(current.updatedAt, server.updatedAt) });
    }
  }
  return [...merged.values()];
}
