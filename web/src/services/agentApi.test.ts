import { afterEach, expect, it, vi } from "vitest";
import { checkConnection, streamAgentResponse } from "./agentApi";

afterEach(() => vi.unstubAllGlobals());

it.each([
  [200, "connected", "connected"],
  [200, "login-required", "login-required"],
  [200, "session-expired", "session-expired"],
  [200, "api-key-required", "api-key-required"],
  [200, "invalid-api-key", "invalid-api-key"],
  [200, "unexpected", "unreachable"],
  [503, "connected", "unreachable"],
] as const)("将 HTTP %s 的 %s 映射为 %s", async (httpStatus, status, expected) => {
  const fetchStub = vi.fn().mockResolvedValue(Response.json({ status }, { status: httpStatus }));
  vi.stubGlobal("fetch", fetchStub);
  const signal = new AbortController().signal;
  await expect(checkConnection("http://localhost", "service-key", "token", signal)).resolves.toBe(expected);
  expect(fetchStub).toHaveBeenCalledWith("http://localhost/connection/status", {
    headers: { "X-Agent-Api-Key": "service-key", Authorization: "Bearer token" },
    cache: "no-store",
    signal,
  });
});

it("重试沿用同一个问题 ID，并关闭可能重复执行的工具", async () => {
  const fetchStub = vi.fn().mockImplementation(() =>
    Promise.resolve(new Response('data: {"type":"done","content":"完成"}\n\n'))
  );
  vi.stubGlobal("fetch", fetchStub);
  const question = { question: "原问题", threadId: "thread", messageId: "question-uuid", apiKey: "", accessToken: "" };
  const signal = new AbortController().signal;
  await streamAgentResponse("http://localhost", question, signal, vi.fn());
  await streamAgentResponse("http://localhost", { ...question, retry: true }, signal, vi.fn());
  expect(fetchStub.mock.calls.map((call) => JSON.parse(call[1].body))).toEqual([
    { question: "原问题", thread_id: "thread", message_id: "question-uuid", retry: false },
    { question: "原问题", thread_id: "thread", message_id: "question-uuid", retry: true },
  ]);
});
