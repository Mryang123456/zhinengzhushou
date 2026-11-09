import { describe, expect, it } from "vitest";
import { createSessionTitle } from "./session";

describe("createSessionTitle", () => {
  it("在同一毫秒内高频创建时仍生成唯一标题", () => {
    const now = new Date("2026-11-07T11:21:34.000Z");
    const titles = new Set(Array.from({ length: 10_000 }, (_, index) => createSessionTitle(now, index.toString(36).padStart(6, "0"))));
    expect(titles).toHaveLength(10_000);
  });
});
