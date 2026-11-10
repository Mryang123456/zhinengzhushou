import type { Message, ProgressStage, StreamEvent } from "../../types";

const progressLabels: Record<ProgressStage, string> = {
  preparing: "已收到，正在准备",
  understanding: "正在分析问题",
  context: "正在整理较早的对话",
  drafting: "正在组织回答",
  checking: "正在校验回答",
};

export function applyStreamEvent(message: Message, event: StreamEvent, replaceContent = false): Message {
  if (event.type === "progress" && event.stage && Object.hasOwn(progressLabels, event.stage)) {
    const label = progressLabels[event.stage];
    if (message.toolEvents?.at(-1) === label) return message;
    return { ...message, toolEvents: [...(message.toolEvents ?? []), label] };
  }
  if (event.type === "token") {
    return { ...message, content: (replaceContent ? "" : message.content) + (event.content ?? ""), retryable: false };
  }
  if (event.type === "tool_start" || event.type === "tool_end") {
    const label = event.type === "tool_start" ? "正在调用" : "工具已返回";
    return { ...message, toolEvents: [...(message.toolEvents ?? []), `${label}：${(event.tools ?? ["工具"]).join("、")}`] };
  }
  if (event.type === "approval_required" && event.approval_id && event.confirmation_token && event.tool && event.risk && event.arguments && event.expires_at) {
    return {
      ...message,
      approval: {
        approvalId: event.approval_id,
        confirmationToken: event.confirmation_token,
        tool: event.tool,
        risk: event.risk,
        arguments: event.arguments,
        expiresAt: event.expires_at,
        status: "pending",
      },
      toolEvents: [...(message.toolEvents ?? []), `等待确认：${event.tool}（${event.risk}）`],
    };
  }
  if (event.type === "error") {
    return { ...message, notice: event.message ?? "请求失败，请稍后重试。", pending: false, retryable: event.retryable ?? true };
  }
  if (event.type === "done") {
    return { ...message, content: event.content || message.content, pending: false, retryable: false };
  }
  return message;
}
