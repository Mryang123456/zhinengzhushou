import type { StreamEvent } from "../types";
import { consumeSseBlocks } from "../utils/sse";
type AgentRequest = { question: string; threadId: string; userId: string; apiKey: string };
export async function streamAgentResponse(apiUrl: string, request: AgentRequest, signal: AbortSignal, onEvent: (event: StreamEvent) => void) { const response = await fetch(`${apiUrl}/chat/stream`, { method: "POST", headers: { "Content-Type": "application/json", ...(request.apiKey ? { "X-Agent-Api-Key": request.apiKey } : {}) }, body: JSON.stringify({ question: request.question, thread_id: request.threadId, user_id: request.userId, use_mcp: true, use_memory: true }), signal }); if (!response.ok || !response.body) throw new Error(`服务暂不可用（${response.status}）`); const reader = response.body.getReader(); const decoder = new TextDecoder(); let remainder = ""; while (true) { const { done, value } = await reader.read(); if (done) break; remainder += decoder.decode(value, { stream: true }); const lastBreak = remainder.lastIndexOf("\n\n"); if (lastBreak >= 0) { consumeSseBlocks(remainder.slice(0, lastBreak + 2), onEvent); remainder = remainder.slice(lastBreak + 2); } } if (remainder) consumeSseBlocks(remainder, onEvent); }

export async function confirmMcpApproval(apiUrl: string, approvalId: string, userId: string, confirmationToken: string, apiKey: string): Promise<{ status: string; result: string }> {
  const response = await fetch(`${apiUrl}/mcp/approvals/${encodeURIComponent(approvalId)}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(apiKey ? { "X-Agent-Api-Key": apiKey } : {}) },
    body: JSON.stringify({ user_id: userId, confirmation_token: confirmationToken }),
  });
  const payload = await response.json() as { detail?: string; status?: string; result?: string };
  if (!response.ok || !payload.result) throw new Error(payload.detail ?? "确认操作失败。");
  return { status: payload.status ?? "completed", result: payload.result };
}
