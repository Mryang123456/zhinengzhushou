import type { Message, Session, StreamEvent } from "../types";
import { consumeSseBlocks } from "../utils/sse";
export type AgentRequest = {
  question: string; threadId: string; messageId: string; apiKey: string; accessToken: string;
  retry?: boolean; attemptId?: string; recoverSubmission?: boolean; resume?: Message["resume"];
};
const authHeaders = (apiKey: string, accessToken: string) => ({ ...(apiKey ? { "X-Agent-Api-Key": apiKey } : {}), ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}) });

export async function loadRemoteSessions(apiUrl: string, apiKey: string, accessToken: string, signal: AbortSignal): Promise<Session[]> {
  const response = await fetch(`${apiUrl}/chat/sessions`, {
    headers: authHeaders(apiKey, accessToken), cache: "no-store", signal,
  });
  if (!response.ok) throw serviceRequestError(response.status);
  return (await response.json()).sessions;
}

export type ConnectionStatus = "checking" | "connected" | "login-required" | "session-expired"
  | "api-key-required" | "invalid-api-key" | "unreachable";

export const connectionLabels: Record<ConnectionStatus, string> = {
  checking: "检测中",
  connected: "服务已连接",
  "login-required": "需要登录",
  "session-expired": "登录已失效",
  "api-key-required": "需要访问密钥",
  "invalid-api-key": "访问密钥无效",
  unreachable: "服务不可达",
};

function requestIdentity(apiUrl: string, accessToken: string) {
  return fetch(`${apiUrl}/auth/me`, {
    headers: authHeaders("", accessToken),
    cache: "no-store",
  });
}

export async function checkConnection(apiUrl: string, apiKey: string, accessToken: string, signal: AbortSignal): Promise<ConnectionStatus> {
  const response = await fetch(`${apiUrl}/connection/status`, {
    headers: authHeaders(apiKey, accessToken),
    cache: "no-store",
    signal,
  });
  if (!response.ok) return "unreachable";
  const payload = await response.json();
  const status = payload?.status;
  return typeof status === "string" && status !== "checking" && Object.hasOwn(connectionLabels, status)
    ? status as ConnectionStatus
    : "unreachable";
}

function serviceRequestError(status: number): Error {
  if (status === 401) return new Error("访问凭据无效或已失效，请检查登录状态与访问密钥。");
  return new Error(`服务暂不可用（${status}）`);
}

export async function streamAgentResponse(apiUrl: string, request: AgentRequest, signal: AbortSignal, onEvent: (event: StreamEvent) => void) {
  const headers = authHeaders(request.apiKey, request.accessToken);
  let runId = request.resume?.runId;
  let cursor = request.resume?.after ?? "0";
  let terminal = false;
  const receive = (event: StreamEvent) => {
    if (terminal) return;
    if (event.type === "run" && event.run_id) runId = event.run_id;
    if (event.event_id && /^\d+$/.test(event.event_id)) {
      if (BigInt(event.event_id) <= BigInt(cursor)) return;
      cursor = event.event_id;
    }
    terminal = event.type === "done" || event.type === "error";
    onEvent(event);
  };

  for (let attempt = 0; attempt <= 3; attempt++) {
    signal.throwIfAborted();
    const response = await (runId
      ? fetch(`${apiUrl}/chat/runs/${encodeURIComponent(runId)}/events?after=${encodeURIComponent(cursor)}`, { headers, signal })
      : fetch(`${apiUrl}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...headers },
        body: JSON.stringify({
          question: request.question, thread_id: request.threadId, message_id: request.messageId,
          retry: request.retry ?? false, attempt_id: request.attemptId,
          recover_submission: request.recoverSubmission,
        }),
        signal,
      })).catch((error: unknown) => {
        if (runId && !signal.aborted) return null;
        throw error;
      });
    if (response?.ok) {
      try {
        await consumeAgentStream(response, receive, () => terminal);
      } catch (error) {
        if (!runId || signal.aborted) throw error;
      }
      if (terminal) return;
    } else if (response && (!runId || (response.status < 500 && response.status !== 429))) {
      throw serviceRequestError(response.status);
    }
    signal.throwIfAborted();
    if (!runId || attempt === 3) break;
    await new Promise<void>((resolve) => setTimeout(resolve, 300 * 2 ** attempt));
  }
  throw new Error("回答传输中断，可以重新发送。");
}

async function consumeAgentStream(response: Response, receive: (event: StreamEvent) => void, isTerminal: () => boolean) {
  if (!response.body) throw new Error("服务未返回回答数据。");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let remainder = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      remainder += decoder.decode(value, { stream: true });
      const lastBreak = remainder.lastIndexOf("\n\n");
      if (lastBreak >= 0) {
        consumeSseBlocks(remainder.slice(0, lastBreak + 2), receive);
        remainder = remainder.slice(lastBreak + 2);
      }
      if (isTerminal()) {
        await reader.cancel().catch(() => undefined);
        return;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function cancelAgentRun(apiUrl: string, runId: string, request: AgentRequest) {
  const response = await fetch(`${apiUrl}/chat/runs/${encodeURIComponent(runId)}/cancel`, {
    method: "POST", headers: authHeaders(request.apiKey, request.accessToken),
  });
  if (!response.ok) throw serviceRequestError(response.status);
}

export async function confirmMcpApproval(apiUrl: string, approvalId: string, confirmationToken: string, apiKey: string, accessToken: string, runId?: string): Promise<{ status: string; result: string }> {
  const response = await fetch(`${apiUrl}/mcp/approvals/${encodeURIComponent(approvalId)}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(apiKey, accessToken) },
    body: JSON.stringify(confirmationToken ? { confirmation_token: confirmationToken } : { run_id: runId }),
  });
  const payload = await response.json() as { detail?: string; status?: string; result?: string };
  if (!response.ok || !payload.result) throw new Error(payload.detail ?? "确认操作失败。");
  return { status: payload.status ?? "completed", result: payload.result };
}

export async function login(apiUrl: string, username: string, password: string): Promise<{ accessToken: string; userId: string }> {
  const response = await fetch(`${apiUrl}/auth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const payload = await response.json() as { access_token?: string; detail?: string };
  if (!response.ok || !payload.access_token) throw new Error(payload.detail ?? "登录失败。");

  const identity = await requestIdentity(apiUrl, payload.access_token);
  if (!identity.ok) throw new Error("无法确认登录身份，请重新登录。");
  const { user_id, authentication_required } = await identity.json();
  if (authentication_required !== true || typeof user_id !== "string" || !user_id.trim()) {
    throw new Error("服务未返回有效的登录身份。");
  }
  return { accessToken: payload.access_token, userId: user_id };
}
