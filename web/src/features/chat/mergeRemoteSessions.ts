import type { Message, Session } from "../../types";
import { mergeMessageOrder } from "../../utils/messageOrder";
import { applyStreamEvent } from "./messageEvents";

function restoreReply(reply: Message): Message {
  const { activity, ...message } = reply;
  return (activity ?? []).reduce((current, event) => applyStreamEvent(current, event), message);
}

export function mergeRemoteSessions(local: Session[], remote: Session[], isRunning: (id: string) => boolean): Session[] {
  const merged = new Map(local.map((session) => [session.id, session]));
  for (const server of remote) {
    const current = merged.get(server.id);
    if (!current) {
      merged.set(server.id, { ...server, messages: server.messages.map(restoreReply) });
      continue;
    }
    const messages = [...current.messages];
    const localIndexes = new Map(messages.map((message, index) => [message.id, index]));
    const ordered: Message[] = [];
    let replaced = false;
    for (let remoteIndex = 0; remoteIndex < server.messages.length; remoteIndex += 2) {
      const question = server.messages[remoteIndex];
      const answer = server.messages[remoteIndex + 1];
      if (question?.role !== "user" || answer?.role !== "assistant") continue;
      const index = localIndexes.get(question.id);
      if (index === undefined) {
        ordered.push(question, restoreReply(answer));
        continue;
      }
      const localAnswer = messages[index + 1];
      const pendingLast = index + 1 === messages.length - 1
        && (localAnswer?.pending || localAnswer?.resume || localAnswer?.submission || localAnswer?.retryable || isRunning(server.id));
      if (!pendingLast) {
        ordered.push(messages[index]);
        if (localAnswer?.role === "assistant") ordered.push(localAnswer);
      }
      if (localAnswer?.role !== "assistant" || localAnswer.approval) continue;
      const incoming = answer.completedAt ?? 0;
      const isLast = index + 1 === messages.length - 1;
      if (isLast && isRunning(server.id)) continue;
      const unfinished = localAnswer.pending || localAnswer.retryable || localAnswer.resume || localAnswer.submission;
      if (localAnswer.submission?.retry && !localAnswer.resume) continue;
      if (localAnswer.resume && localAnswer.resume.runId !== answer.completedRunId) continue;
      const sameRun = localAnswer.completedRunId && localAnswer.completedRunId === answer.completedRunId;
      if (localAnswer.completedRunId && !sameRun && localAnswer.completedAt === undefined) continue;
      if (incoming > (localAnswer.completedAt ?? 0) && (unfinished || sameRun || localAnswer.content === answer.content || localAnswer.completedAt)) {
        const restored = restoreReply(answer);
        messages[index + 1] = { ...restored, id: localAnswer.id, toolEvents: restored.toolEvents ?? localAnswer.toolEvents };
        if (!pendingLast) ordered[ordered.length - 1] = messages[index + 1];
        replaced = true;
      }
    }
    const last = messages.at(-1);
    const holdLast = last?.role === "assistant" && messages.at(-2)?.role === "user"
      && (last.pending || last.resume || last.submission || last.retryable || isRunning(server.id));
    const combined = mergeMessageOrder(messages, ordered, holdLast ? messages.length - 2 : messages.length);
    if (replaced || combined.length !== messages.length || combined.some((message, index) => message !== messages[index])) {
      merged.set(server.id, { ...current, messages: combined, updatedAt: Math.max(current.updatedAt, server.updatedAt) });
    }
  }
  return [...merged.values()];
}
