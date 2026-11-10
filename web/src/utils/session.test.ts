import { describe, expect, it } from "vitest";
import { archiveSessionState, createSessionTitle, ensureVisibleSession, resolveVisibleSession } from "./session";

describe("createSessionTitle", () => {
  it("在同一毫秒内高频创建时仍生成唯一标题", () => {
    const now = new Date("2026-11-07T11:21:34.000Z");
    const titles = new Set(Array.from({ length: 10_000 }, (_, index) => createSessionTitle(now, index.toString(36).padStart(6, "0"))));
    expect(titles).toHaveLength(10_000);
  });

  it("归档最后一个会话时原子创建并选中新会话", () => {
    const original = { id: "old", title: "旧会话", messages: [], updatedAt: 1 };
    const replacement = { id: "new", title: "新会话", messages: [], updatedAt: 2 };
    const state = archiveSessionState([original], "old", "old", replacement);
    expect(state.activeId).toBe("new");
    expect(state.sessions).toHaveLength(2);
    expect(state.sessions[0]).toEqual(replacement);
    expect(state.sessions[1].archived).toBe(true);
  });
});

describe("恢复会话", () => {
  const archived = { id: "archived", title: "已归档", messages: [], updatedAt: 3, archived: true };
  const visible = { id: "visible", title: "可用会话", messages: [], updatedAt: 2 };
  const replacement = { id: "replacement", title: "新会话", messages: [], updatedAt: 4 };

  it("跳过列表开头或选中项中的归档会话", () => {
    const sessions = ensureVisibleSession([archived, visible], () => replacement);
    expect(sessions).toEqual([archived, visible]);
    expect(resolveVisibleSession(sessions)?.id).toBe(visible.id);
    expect(resolveVisibleSession(sessions, archived.id)?.id).toBe(visible.id);
  });

  it("全部归档时保留旧记录并选中新会话", () => {
    const sessions = ensureVisibleSession([archived], () => replacement);
    expect(sessions).toEqual([replacement, archived]);
    expect(resolveVisibleSession(sessions)?.id).toBe(replacement.id);
  });
});
