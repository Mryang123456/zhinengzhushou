import type { StreamEvent } from "../types";
import { consumeSseBlocks } from "../utils/sse";
type AgentRequest = { question: string; threadId: string; apiKey: string; accessToken: string };
const authHeaders = (apiKey: string, accessToken: string) => ({ ...(apiKey ? { "X-Agent-Api-Key": apiKey } : {}), ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}) });
export async function streamAgentResponse(apiUrl: string, request: AgentRequest, signal: AbortSignal, onEvent: (event: StreamEvent) => void) { const response = await fetch(`${apiUrl}/chat/stream`, { method: "POST", headers: { "Content-Type": "application/json", ...authHeaders(request.apiKey, request.accessToken) }, body: JSON.stringify({ question: request.question, thread_id: request.threadId, use_mcp: true, use_memory: true }), signal }); if (!response.ok || !response.body) throw new Error(`服务暂不可用（${response.status}）`); const reader = response.body.getReader(); const decoder = new TextDecoder(); let remainder = ""; while (true) { const { done, value } = await reader.read(); if (done) break; remainder += decoder.decode(value, { stream: true }); const lastBreak = remainder.lastIndexOf("\n\n"); if (lastBreak >= 0) { consumeSseBlocks(remainder.slice(0, lastBreak + 2), onEvent); remainder = remainder.slice(lastBreak + 2); } } if (remainder) consumeSseBlocks(remainder, onEvent); }

export async function confirmMcpApproval(apiUrl: string, approvalId: string, confirmationToken: string, apiKey: string, accessToken: string): Promise<{ status: string; result: string }> {
  const response = await fetch(`${apiUrl}/mcp/approvals/${encodeURIComponent(approvalId)}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(apiKey, accessToken) },
    body: JSON.stringify({ confirmation_token: confirmationToken }),
  });
  const payload = await response.json() as { detail?: string; status?: string; result?: string };
  if (!response.ok || !payload.result) throw new Error(payload.detail ?? "确认操作失败。");
  return { status: payload.status ?? "completed", result: payload.result };
}

export async function login(apiUrl: string, username: string, password: string): Promise<{ accessToken: string; userId: string }> { const response = await fetch(`${apiUrl}/auth/token`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password }) }); const payload = await response.json() as { access_token?: string; detail?: string }; if (!response.ok || !payload.access_token) throw new Error(payload.detail ?? "登录失败。"); return { accessToken: payload.access_token, userId: username }; }
