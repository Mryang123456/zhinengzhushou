import { useEffect, useRef } from "react";
import { useAgentStream } from "../../hooks/useAgentStream";
import { confirmMcpApproval } from "../../services/agentApi";
import type { Message, Session } from "../../types";
import { applyStreamEvent } from "./messageEvents";
import type { UpdateMessages } from "./useChatSessions";

type Options = {
  apiUrl: string;
  apiKey: string;
  accessToken: string;
  activeSession: Session;
  updateMessages: UpdateMessages;
  refreshConnection: () => void;
};

type ReplyOptions = {
  retry?: boolean;
  resume?: Message["resume"];
  recover?: Message["submission"];
};

export function useChatMessages({ apiUrl, apiKey, accessToken, activeSession, updateMessages, refreshConnection }: Options) {
  const { runningIds, isRunning, send, stop } = useAgentStream(apiUrl);
  const sessionId = activeSession.id;
  const isSending = runningIds.has(sessionId);
  const lastReply = activeSession.messages.at(-1);
  const lastQuestion = activeSession.messages.at(-2);
  const resumedRuns = useRef(new Set<string>());
  const canRetry = !isSending && lastReply?.role === "assistant" && !lastReply.pending
    && lastReply.retryable && (!lastReply.approval || lastReply.resume) && lastQuestion?.role === "user";

  const updateMessage = (threadId: string, messageId: string, update: (message: Message) => Message) => {
    updateMessages(threadId, (items) => items.map((item) => item.id === messageId ? update(item) : item));
  };

  const runReply = async (threadId: string, replyId: string, question: Message, { retry = false, resume, recover }: ReplyOptions = {}) => {
    const submission = recover ?? { retry, attemptId: retry ? crypto.randomUUID() : undefined };
    if (!resume) updateMessage(threadId, replyId, (item) => ({ ...item, submission }));
    let replaceOnFirstToken = !resume && (retry || Boolean(recover));
    try {
      await send({
        question: question.content, messageId: question.id, threadId, apiKey, accessToken, resume,
        retry: submission.retry, attemptId: submission.attemptId, recoverSubmission: recover ? true : undefined,
      }, (event) => {
        const replaceContent = replaceOnFirstToken && event.type === "token" && Boolean(event.content);
        if (replaceContent) replaceOnFirstToken = false;
        updateMessage(threadId, replyId, (item) => {
          if (event.type === "run" && event.run_id) {
            return item.resume?.runId === event.run_id
              ? item : { ...item, resume: { runId: event.run_id, after: "0" }, submission: undefined };
          }
          const updated = applyStreamEvent(item, event, replaceContent);
          if (event.type === "done" || event.type === "error") return { ...updated, resume: undefined, submission: undefined };
          if (updated.resume && event.event_id) return { ...updated, resume: { ...updated.resume, after: event.event_id } };
          return updated;
        });
      });
    } catch (error) {
      const stopped = error instanceof Error && error.name === "AbortError";
      if (!stopped) refreshConnection();
      const message = error instanceof Error ? error.message : "请稍后重试。";
      updateMessage(threadId, replyId, (item) => applyStreamEvent(item, {
        type: "error",
        message: stopped ? "已停止生成。" : message,
        retryable: !stopped,
      }));
    }
  };

  const submit = (value: string) => {
    const text = value.trim();
    if (!text || isRunning(sessionId)) return false;
    const replyId = crypto.randomUUID();
    const question: Message = { id: crypto.randomUUID(), role: "user", content: text };
    updateMessages(sessionId, (items) => [
      ...items,
      question,
      { id: replyId, role: "assistant", content: "", pending: true },
    ]);
    void runReply(sessionId, replyId, question);
    return true;
  };

  const retryLastReply = () => {
    if (!canRetry || !lastReply || !lastQuestion || isRunning(sessionId)) return;
    const resume = lastReply.resume;
    updateMessage(sessionId, lastReply.id, (item) => ({
      ...item, pending: true, notice: undefined, toolEvents: resume ? item.toolEvents : [],
    }));
    void runReply(sessionId, lastReply.id, lastQuestion, { retry: !resume, resume, recover: lastReply.submission });
  };

  useEffect(() => {
    const runId = lastReply?.resume?.runId;
    if (!canRetry || !runId || resumedRuns.current.has(runId)) return;
    // 等当前挂载完成，避免 StrictMode 的首次清理取消刚恢复的请求。
    const timer = window.setTimeout(() => {
      resumedRuns.current.add(runId);
      retryLastReply();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [canRetry, lastReply?.resume?.runId, retryLastReply]);

  const confirmApproval = async (messageId: string) => {
    const approval = activeSession.messages.find((item) => item.id === messageId)?.approval;
    if (!approval || approval.status !== "pending") return;
    updateMessage(sessionId, messageId, (item) => ({ ...item, approval: { ...approval, status: "executing" } }));
    try {
      const response = await confirmMcpApproval(apiUrl, approval.approvalId, approval.confirmationToken, apiKey, accessToken, approval.runId);
      updateMessage(sessionId, messageId, (item) => ({
        ...item,
        approval: { ...approval, status: "completed" },
        content: `${item.content}\n\n已执行已确认操作（${approval.tool}）：\n${response.result}`,
      }));
    } catch (error) {
      updateMessage(sessionId, messageId, (item) => ({
        ...item,
        approval: { ...approval, status: "failed" },
        content: `${item.content}\n\n确认操作失败：${(error as Error).message}`,
      }));
      refreshConnection();
    }
  };

  return {
    runningIds,
    isRunning,
    isSending,
    submit,
    stop: () => {
      void stop(sessionId).catch((error: unknown) => {
        if (!lastReply) return;
        updateMessage(sessionId, lastReply.id, (item) => ({
          ...item, notice: error instanceof Error ? error.message : "暂时无法停止，请稍后重试。",
        }));
      });
    },
    retryMessageId: canRetry ? lastReply.id : undefined,
    retryLastReply,
    confirmApproval,
  };
}
