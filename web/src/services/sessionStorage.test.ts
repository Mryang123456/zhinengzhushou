import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { Session } from "../types";
import { loadSessions, saveSessions, sessionStorageKey } from "./sessionStorage";

const snapshots = vi.hoisted(() => new Map<string, Session[]>());
vi.mock("./sessionDatabase", () => ({
  accessSessionSnapshot: async (key: string, update?: (stored: Session[] | undefined) => Session[]) => {
    if (update) snapshots.set(key, update(snapshots.get(key)));
    return snapshots.get(key);
  },
}));
beforeEach(() => snapshots.clear());
afterEach(() => vi.unstubAllGlobals());

it("账号、访客和不同后端分别读取自己的记录，旧共享记录不会被自动归属", async () => {
  const data = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => data.set(key, value),
  });
  const session: Session = { id: "thread-a", title: "A 的会话", updatedAt: 1, messages: [] };
  const api = "http://127.0.0.1:8000";
  data.set("zhixing-agent:sessions:v1", JSON.stringify([session]));

  expect(await loadSessions(sessionStorageKey(api, null))).toBeNull();
  await saveSessions(sessionStorageKey(api, "A"), [session]);
  expect(await loadSessions(sessionStorageKey(api, "B"))).toBeNull();
  expect(await loadSessions(sessionStorageKey("https://example.test", "A"))).toBeNull();
  expect(await loadSessions(sessionStorageKey(api, "A"))).toEqual([session]);
  expect(sessionStorageKey(api, "guest")).not.toBe(sessionStorageKey(api, null));
});

it("保留已收到的片段，恢复时将进行中的回答标记为中断", async () => {
  const data = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => data.set(key, value),
  });
  const key = sessionStorageKey("http://127.0.0.1:8000", "A");
  await saveSessions(key, [{
    id: "thread-a", title: "A 的会话", updatedAt: 1,
    messages: [{ id: "reply", role: "assistant", content: "部分回答", pending: true }],
  }]);
  expect((await loadSessions(key))?.[0].messages[0]).toMatchObject({
    content: "部分回答", pending: false, notice: "回答已中断，可以重新发送。", retryable: true,
  });
});

it("合并不同标签页的会话和消息，并保留另一页已保存的标题", async () => {
  const data = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => data.set(key, value),
  });
  const key = "shared-account";
  const shared: Session = { id: "shared", title: "原标题", updatedAt: 1, messages: [] };
  data.set(key, JSON.stringify([shared]));
  const tabA = (await loadSessions(key))!;
  const tabB = (await loadSessions(key))!;
  const another: Session = { id: "another", title: "新会话", updatedAt: 4, messages: [] };
  await saveSessions(key, [{
    ...tabA[0], title: "A 改了标题", updatedAt: 2,
    messages: [{ id: "a", role: "user", content: "A 的消息" }],
  }], tabA);
  await saveSessions(key, [{
    ...tabB[0], updatedAt: 3,
    messages: [{ id: "b", role: "user", content: "B 的消息" }],
  }, another], tabB);

  const result = (await loadSessions(key))!;
  expect(result.map((session) => session.id)).toEqual(["another", "shared"]);
  expect(result[1].title).toBe("A 改了标题");
  expect(result[1].messages.map((message) => message.id)).toEqual(["a", "b"]);
});

it("读取旧版进行中的消息时结束加载，带审批的回复不可重试", async () => {
  vi.stubGlobal("localStorage", {
    getItem: () => JSON.stringify([{
      id: "thread-a", title: "旧会话", updatedAt: 1,
      messages: [
        { id: "normal", role: "assistant", content: "旧片段", pending: true },
        { id: "approval", role: "assistant", content: "", pending: true, approval: { status: "pending" } },
      ],
    }]),
  });
  const messages = (await loadSessions("old-key"))?.[0].messages;
  expect(messages?.[0]).toMatchObject({ pending: false, retryable: true });
  expect(messages?.[1]).toMatchObject({ pending: false, retryable: false });
});

it("刷新后保留持久化审批的待确认状态，但不保存令牌", async () => {
  await saveSessions("approval-key", [{
    id: "thread", title: "会话", updatedAt: 1,
    messages: [{
      id: "reply", role: "assistant", content: "等待审批",
      approval: {
        approvalId: "approval", runId: "run", confirmationToken: "private-token",
        tool: "calendar", risk: "L3", arguments: {}, expiresAt: 9_999_999_999, status: "pending",
      },
    }],
  }]);
  expect(snapshots.get("approval-key")?.[0].messages[0].approval).toMatchObject({
    runId: "run", confirmationToken: "", status: "pending",
  });
  expect((await loadSessions("approval-key"))?.[0].messages[0].approval?.status).toBe("pending");
});
