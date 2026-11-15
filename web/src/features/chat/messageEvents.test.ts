import { expect, test } from "vitest";
import type { Message } from "../../types";
import { applyStreamEvent, currentActivityStatus, summarizeActivity } from "./messageEvents";

test("plan progress replaces the last snapshot during replay", () => {
  const message: Message = { id: "answer", role: "assistant", content: "", pending: true };
  const first = applyStreamEvent(message, { type: "plan_progress", steps: [
    { id: 0, title: "查询天气", status: "running" },
    { id: 1, title: "条件推荐", status: "pending" },
  ] });
  const second = applyStreamEvent(first, { type: "plan_progress", steps: [
    { id: 0, title: "查询天气", status: "complete" },
    { id: 1, title: "条件推荐", status: "skipped", reason: "预报未显示降雨" },
  ] });
  expect(second.planSteps?.map((step) => step.status)).toEqual(["complete", "skipped"]);
  expect(second.planSteps?.[1].reason).toBe("预报未显示降雨");
});

test("shows why further searches stopped", () => {
  const message: Message = { id: "answer", role: "assistant", content: "" };
  const step = applyStreamEvent(message, {
    type: "evidence_review", action: "partial", stop_reason: "no_new_evidence",
  });
  expect(step.toolEvents).toEqual(["补查未找到新资料，整理已有结果"]);
});

test("summarizes repeated searches and evidence checks without hiding the detailed events", () => {
  const message: Message = {
    id: "answer",
    role: "assistant",
    content: "回答",
    toolEvents: [
      "正在搜索公开资料",
      "本批找到 5 条网页结果",
      "现有资料还缺少关键证据",
      "正在补查资料",
      "正在搜索公开资料",
      "本批找到 5 条网页结果",
      "回答校验通过",
      { label: "已读取 1 篇网页正文", pages: [{ url: "https://example.org", title: "来源", domain: "example.org" }] },
    ],
  };

  expect(summarizeActivity(message)).toBe("已搜索公开资料（2 批，各批合计 10 条），补查 1 次，读取 1 篇网页正文，回答校验通过");
  expect(message.toolEvents).toHaveLength(8);
});

test("summarizes direct answers without implying a web search", () => {
  const message: Message = {
    id: "answer",
    role: "assistant",
    content: "回答",
    toolEvents: ["正在组织回答", "正在校验回答", "回答校验通过"],
  };

  expect(summarizeActivity(message)).toBe("回答校验通过");
});

test("shows final failure even when evidence review found candidate sources", () => {
  const initial: Message = { id: "answer", role: "assistant", content: "" };
  const checked = applyStreamEvent(initial, { type: "evidence_review", action: "answer" });
  expect(summarizeActivity(checked)).toBe("");
  expect(summarizeActivity(applyStreamEvent(checked, { type: "quality", passed: false }))).toBe("回答未通过校验");
  expect(summarizeActivity(applyStreamEvent(checked, { type: "quality", passed: true, partial: true }))).toBe("已核实部分回答");
});

test("keeps the in-progress status concise across repeated search events", () => {
  const message: Message = {
    id: "answer",
    role: "assistant",
    content: "",
    pending: true,
    toolEvents: [
      "正在搜索公开资料",
      "本批找到 5 条网页结果",
      "现有资料还缺少关键证据",
      "正在补查资料",
      "正在搜索公开资料",
    ],
  };

  expect(currentActivityStatus(message)).toBe("正在搜索公开资料 · 已完成 1 批");
  expect(currentActivityStatus({ ...message, toolEvents: [...message.toolEvents!, "正在补查资料"] }))
    .toBe("正在补充缺失的关键证据");
});

test("adds known stage timings to the expanded activity and ignores unknown operations", () => {
  const initial: Message = { id: "answer", role: "assistant", content: "", pending: true };
  const measured = applyStreamEvent(initial, {
    type: "timing",
    operation: "answer_generation",
    elapsed_ms: 2340,
  });
  const unknown = applyStreamEvent(measured, {
    type: "timing",
    operation: "private_operation",
    elapsed_ms: 10,
  });

  expect(measured.toolEvents).toEqual(["阶段耗时：回答生成 2.3 秒"]);
  expect(currentActivityStatus(measured)).toBe("正在准备回答…");
  expect(unknown).toBe(measured);
});

test("shows executed searches and excerpts without counting snippets as read pages", () => {
  const initial: Message = { id: "answer", role: "assistant", content: "", pending: true };
  const searching = applyStreamEvent(initial, { type: "web_queries", queries: ["海岚 A2 充电功率"] });
  const results = applyStreamEvent(searching, { type: "tool_end", tools: ["web_search"], web_result_count: 2 });
  const page = { url: "https://example.org/a2", title: "充电规格", domain: "example.org" };
  const highlighted = applyStreamEvent(results, {
    type: "web_highlights", pages: [{ ...page, excerpt: "有线充电功率为 45 瓦。" }],
  });
  const read = applyStreamEvent(highlighted, { type: "web_pages_read", pages: [page] });

  expect(searching.toolEvents).toEqual([{ label: "正在搜索公开资料", queries: ["海岚 A2 充电功率"], kind: "queries" }]);
  expect(highlighted.toolEvents?.at(-1)).toMatchObject({ kind: "highlights", pages: [{ excerpt: "有线充电功率为 45 瓦。" }] });
  expect(summarizeActivity(highlighted)).toBe("已搜索公开资料（1 批，各批合计 2 条）");
  expect(summarizeActivity(read)).toContain("读取 1 篇网页正文");
});
