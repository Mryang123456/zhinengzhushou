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

export function useChatMessages({ apiUrl, apiKey, accessToken, activeSession, updateMessages, refreshConnection }: Options) {
  const { runningIds, isRunning, send, stop } = useAgentStream(apiUrl);
  const sessionId = activeSession.id;
  const isSending = runningIds.has(sessionId);
  const lastReply = activeSession.messages.at(-1);
  const lastQuestion = activeSession.messages.at(-2);
  const canRetry = !isSending && lastReply?.role === "assistant" && !lastReply.pending
    && lastReply.retryable && !lastReply.approval && lastQuestion?.role === "user";

  const updateMessage = (threadId: string, messageId: string, update: (message: Message) => Message) => {
    updateMessages(threadId, (items) => items.map((item) => item.id === messageId ? update(item) : item));
  };

  const runReply = async (threadId: string, replyId: string, question: Message, retry = false) => {
    let replaceOnFirstToken = retry;
    try {
      await send({ question: question.content, messageId: question.id, threadId, apiKey, accessToken, retry }, (event) => {
        const replaceContent = replaceOnFirstToken && event.type === "token" && Boolean(event.content);
        if (replaceContent) replaceOnFirstToken = false;
        updateMessage(threadId, replyId, (item) => applyStreamEvent(item, event, replaceContent));
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
    updateMessage(sessionId, lastReply.id, (item) => ({ ...item, pending: true, notice: undefined, toolEvents: [] }));
    void runReply(sessionId, lastReply.id, lastQuestion, true);
  };

  const confirmApproval = async (messageId: string) => {
    const approval = activeSession.messages.find((item) => item.id === messageId)?.approval;
    if (!approval || approval.status !== "pending") return;
    updateMessage(sessionId, messageId, (item) => ({ ...item, approval: { ...approval, status: "executing" } }));
    try {
      const response = await confirmMcpApproval(apiUrl, approval.approvalId, approval.confirmationToken, apiKey, accessToken);
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
    stop: () => stop(sessionId),
    retryMessageId: canRetry ? lastReply.id : undefined,
    retryLastReply,
    confirmApproval,
  };
}
