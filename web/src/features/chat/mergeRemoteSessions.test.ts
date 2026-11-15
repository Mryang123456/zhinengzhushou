import { expect, test } from "vitest";
import type { Message, Session } from "../../types";
import { mergeRemoteSessions } from "./mergeRemoteSessions";

const question: Message = { id: "q1", role: "user", content: "查资料" };
const remote: Session = {
  id: "session", title: "远程会话", updatedAt: 40, messages: [
    question,
    { id: "q1:answer", role: "assistant", content: "完整回答", completedAt: 40, completedRunId: "run",
      elapsedMs: 1520,
      activity: [{ type: "tool_end", tools: ["web_search"], web_result_count: 3 },
        { type: "evidence_review", action: "search" },
        { type: "web_pages_read", pages: [{ url: "https://example.org", domain: "example.org", title: "正文资料" }] }],
      sources: [{ url: "https://example.org", domain: "example.org", title: "来源" }] },
  ],
};

function fromLocal(reply: Partial<Message>): Session {
  return { ...remote, title: "自定义标题", messages: [
    question, { id: "local-reply", role: "assistant", content: "部分回答", ...reply },
  ] };
}

test("restores public activity on another device", () => {
  const [session] = mergeRemoteSessions([], [remote], () => false);
  expect(session.messages[1].toolEvents).toEqual(["本批找到 3 条网页结果", "现有资料还缺少关键证据",
    { label: "已读取 1 篇网页正文", pages: [{ url: "https://example.org", domain: "example.org", title: "正文资料" }] }]);
  expect(session.messages[1].elapsedMs).toBe(1520);
  expect(session.messages[1].activity).toBeUndefined();
});

test("reconciles a completed remote run with the same interrupted local reply", () => {
  const [session] = mergeRemoteSessions(
    [fromLocal({ resume: { runId: "run", after: "11" }, retryable: true })], [remote], () => false,
  );
  expect(session.title).toBe("自定义标题");
  expect(session.messages).toHaveLength(2);
  expect(session.messages[1]).toMatchObject({ id: "local-reply", content: "完整回答", sources: remote.messages[1].sources });
  expect(session.messages[1].resume).toBeUndefined();
});

test("does not replace active replies or another retry with stale history", () => {
  const local = fromLocal({ resume: { runId: "new-run", after: "0" }, retryable: true });
  expect(mergeRemoteSessions([local], [remote], () => false)[0]).toBe(local);
  const active = fromLocal({ resume: { runId: "run", after: "0" }, pending: true });
  expect(mergeRemoteSessions([active], [remote], () => true)[0]).toBe(active);
});

function pair(id: string): Message[] {
  return [
    { id, role: "user", content: id },
    { id: id + ":answer", role: "assistant", content: id },
  ];
}

test("inserts missing turns between shared questions in server order", () => {
  const local = { ...remote, messages: [...pair("a"), ...pair("c")] };
  const server = { ...remote, messages: [...pair("a"), ...pair("b"), ...pair("c")] };
  const [session] = mergeRemoteSessions([local], [server], () => false);
  expect(session.messages.map((message) => message.id)).toEqual(["a", "a:answer", "b", "b:answer", "c", "c:answer"]);
  expect(mergeRemoteSessions([session], [server], () => false)[0]).toBe(session);
});

test("repairs an existing inversion and keeps the running turn last", () => {
  const active = pair("pending");
  const local = { ...remote, messages: [...pair("c"), ...pair("a"), ...active] };
  const server = { ...remote, messages: [...pair("a"), ...pair("b"), ...pair("c"), ...pair("pending")] };
  const [session] = mergeRemoteSessions([local], [server], () => true);
  expect(session.messages.filter((message) => message.role === "user").map((message) => message.id)).toEqual(["a", "b", "c", "pending"]);
  expect(session.messages.at(-1)).toBe(active[1]);
});
