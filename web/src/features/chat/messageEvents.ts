import type { ActivityStep, Message, ProgressStage, StreamEvent } from "../../types";

const progressLabels: Record<ProgressStage, string> = {
  preparing: "已收到，正在准备",
  understanding: "正在分析问题",
  context: "正在整理较早的对话",
  drafting: "正在组织回答",
  checking: "正在校验回答",
  searching_web: "正在搜索公开资料",
  verifying_recommendations: "正在核对候选景点是否符合条件",
  revising: "正在根据来源修订回答",
};
const answerOutcomes = ["回答未通过校验", "已核实部分回答", "回答校验通过"];

function addStep(message: Message, step: ActivityStep): Message {
  const steps = message.toolEvents ?? [];
  if (steps.at(-1) === step) return message;
  return { ...message, toolEvents: [...steps, step] };
}

export function summarizeActivity(message: Message): string {
  const steps = message.toolEvents ?? [];
  const labels = steps.map((step) => typeof step === "string" ? step : step.label);
  const searchBatches = labels.filter((label) => /^本批找到 \d+ 条网页结果$/.test(label));
  const resultCount = searchBatches.reduce((total, label) => total + Number(label.match(/\d+/)?.[0] ?? 0), 0);
  const retryCount = labels.filter((label) => label === "正在补查资料").length;
  const readPageCount = steps.reduce((total, step) => {
    if (typeof step !== "string") return total + (step.kind ? 0 : step.pages?.length ?? 0);
    return total + Number(step.match(/^已读取 (\d+) 篇网页正文$/)?.[1] ?? 0);
  }, 0);
  const searchedWeb = labels.some((label) =>
    label.includes("正在搜索公开资料") || label.includes("公开资料搜索") || label.includes("网页结果"),
  );
  const outcome = labels.filter((label) => answerOutcomes.includes(label)).at(-1);
  const summary: string[] = [];

  if (searchedWeb) {
    const results = resultCount ? `，各批合计 ${resultCount} 条` : "";
    summary.push(`已搜索公开资料${searchBatches.length ? `（${searchBatches.length} 批${results}）` : ""}`);
  }
  if (retryCount) summary.push(`补查 ${retryCount} 次`);
  if (readPageCount) summary.push(`读取 ${readPageCount} 篇网页正文`);
  if (outcome) summary.push(outcome);

  return summary.join("，");
}

export function currentActivityStatus(message: Message): string {
  const steps = message.toolEvents ?? [];
  const current = message.pending
    ? [...steps].reverse().find((step) => typeof step !== "string" || !step.startsWith("阶段耗时："))
    : steps.at(-1);
  const label = (typeof current === "string" ? current : current?.label) ?? "正在准备回答…";
  if (!message.pending) return label;

  const completedSearches = steps.filter((step) =>
    typeof step === "string" && /^本批(?:找到 \d+ 条网页结果|未找到网页结果)$/.test(step),
  ).length;
  if (label === "正在搜索公开资料") {
    return `正在搜索公开资料${completedSearches ? ` · 已完成 ${completedSearches} 批` : ""}`;
  }
  if (label === "正在补查资料") return "正在补充缺失的关键证据";
  if (label === "现有资料还缺少关键证据") return "正在核对资料是否足以回答";
  const resultCount = label.match(/^本批找到 (\d+) 条网页结果$/)?.[1];
  if (resultCount) return `搜索已返回 ${resultCount} 条结果，正在继续核对`;
  if (label === "本批未找到网页结果") return "本批搜索未找到结果，正在继续处理";
  return label;
}

function toolLabel(name: string) {
  if (name === "search_knowledge_base") return "知识库检索";
  if (name === "web_search" || name.startsWith("search_")) return "公开资料搜索";
  if (name === "get_weather" || name === "weather_current_weather") return "天气查询";
  if (name === "calculator") return "计算";
  return "其他工具";
}

export function applyStreamEvent(message: Message, event: StreamEvent, replaceContent = false): Message {
  if (event.type === "progress" && event.stage && Object.hasOwn(progressLabels, event.stage)) {
    const label = progressLabels[event.stage];
    return addStep(message, label);
  }
  if (event.type === "token") {
    return { ...message, content: (replaceContent ? "" : message.content) + (event.content ?? ""), retryable: false };
  }
  if (event.type === "tool_start" || event.type === "tool_end") {
    const labels = [...new Set((event.tools ?? []).map(toolLabel))];
    if (event.type === "tool_start" && labels.length === 1 && labels[0] === "公开资料搜索") {
      return addStep(message, progressLabels.searching_web);
    }
    const count = event.web_result_count;
    if (event.type === "tool_end" && typeof count === "number" && Number.isSafeInteger(count) && count >= 0) {
      const otherTools = labels.filter((label) => label !== "公开资料搜索");
      const step = addStep(message, count ? `本批找到 ${count} 条网页结果` : "本批未找到网页结果");
      return otherTools.length ? addStep(step, `${otherTools.join("、")}已返回`) : step;
    }
    const tools = labels.join("、") || "工具";
    return addStep(message, event.type === "tool_start" ? `正在执行${tools}` : `${tools}已返回`);
  }
  if (event.type === "evidence_retry") return addStep(message, "正在补查资料");
  if (event.type === "evidence_review") {
    const labels = {
      answer: "已找到候选证据",
      search: "现有资料还缺少关键证据",
      partial: "现有资料只能核实部分问题",
      clarify: "还需要你补充关键信息",
    };
    return event.action && Object.hasOwn(labels, event.action) ? addStep(message, labels[event.action]) : message;
  }
  if (event.type === "quality" && typeof event.passed === "boolean") {
    return addStep(message, answerOutcomes[event.passed ? (event.partial ? 1 : 2) : 0]);
  }
  if (event.type === "web_queries" && Array.isArray(event.queries)) {
    const queries = event.queries.filter((query) => typeof query === "string" && query.trim());
    return queries.length ? addStep(message, { label: "正在搜索公开资料", queries, kind: "queries" }) : message;
  }
  if ((event.type === "web_pages_read" || event.type === "web_highlights") && Array.isArray(event.pages)) {
    const pages = event.pages.filter((page) => page && /^https?:\/\//i.test(page.url));
    if (!pages.length) return message;
    return addStep(message, event.type === "web_highlights"
      ? { label: "本轮检索到的相关摘录", pages, kind: "highlights" }
      : { label: `已读取 ${pages.length} 篇网页正文`, pages });
  }
  if (event.type === "timing" && event.operation && typeof event.elapsed_ms === "number"
      && Number.isFinite(event.elapsed_ms) && event.elapsed_ms >= 0) {
    const labels: Record<string, string> = {
      answer_generation: "回答生成",
      answer_revision: "回答修订",
      evidence_review: "证据复核",
      evidence_retry_decision: "补查决策",
      intent_planning: "任务拆分",
      intent_resolution: "意图识别",
      tool_batch: "工具执行",
      weather_advice: "天气建议",
    };
    const name = labels[event.operation];
    if (!name) return message;
    return addStep(message, `阶段耗时：${name} ${formatDuration(event.elapsed_ms)}`);
  }
  if (event.type === "approval_required" && event.approval_id && event.confirmation_token && event.tool && event.risk && event.arguments && event.expires_at) {
    return {
      ...message,
      approval: {
        approvalId: event.approval_id,
        confirmationToken: event.confirmation_token,
        runId: message.resume?.runId,
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
    return { ...message, content: event.content || message.content, sources: event.sources, elapsedMs: event.elapsed_ms, pending: false, retryable: false };
  }
  return message;
}

function formatDuration(milliseconds: number): string {
  return milliseconds < 1000
    ? `${Math.round(milliseconds)} 毫秒`
    : `${(milliseconds / 1000).toFixed(1)} 秒`;
}
