import { ChevronDown, ExternalLink, FileText, Search } from "lucide-react";
import type { Message } from "../../types";
import { currentActivityStatus, summarizeActivity } from "../../features/chat/messageEvents";
import "./style.css";

export function ActivityTimeline({ message }: { message: Message }) {
  const steps = message.toolEvents ?? [];
  const timings = steps.filter((step): step is string => typeof step === "string" && step.startsWith("阶段耗时："));
  const hasSearchDetails = steps.some((step) => typeof step === "object" && step.kind);
  const visibleSteps = steps.filter((step, index) => {
    if (typeof step !== "string") return true;
    if (step.startsWith("阶段耗时：")) return false;
    if (hasSearchDetails &&
      ["已收到，正在准备", "正在分析问题", "已找到候选证据", "正在组织回答"].includes(step)) return false;
    const next = steps[index + 1];
    return step !== "正在搜索公开资料" || typeof next !== "object" || next.kind !== "queries";
  });
  const status = currentActivityStatus(message);
  const duration = typeof message.elapsedMs === "number" && Number.isFinite(message.elapsedMs) && message.elapsedMs >= 0
    ? message.elapsedMs < 1000 ? " · 不到 1 秒" : ` · 用时 ${Math.round(message.elapsedMs / 1000)} 秒`
    : "";
  const activitySummary = summarizeActivity(message);
  if (!steps.length && !message.pending) return null;

  const title = message.pending
    ? message.content && !message.retryable ? "正在展示回答…" : status
    : message.approval?.status === "pending" ? "等待确认"
      : message.notice ? "处理已中断"
        : `查询过程${duration}${!duration && activitySummary ? ` · ${activitySummary}` : ""}`;

  if (!steps.length) return <div className="chat-activity-status" role="status">{title}</div>;

  return (
    <details className="chat-activity">
      <summary className="chat-activity-summary">
        <Search size={19} aria-hidden="true" />
        <span role="status">{title}</span>
        <ChevronDown className="chat-activity-chevron" size={15} aria-hidden="true" />
      </summary>
      <ol className="chat-activity-steps" aria-label="查询过程">
        {visibleSteps.map((step, index) => (
          <li key={index} aria-current={message.pending && index === visibleSteps.length - 1 ? "step" : undefined}>
            {typeof step === "string" ? (/^本批找到 \d+ 条网页结果$/.test(step)
              ? <span className="chat-activity-detail-label"><Search size={15} aria-hidden="true" />{step.replace("本批找到", "搜索到")}</span>
              : step) : <>
              <span className="chat-activity-detail-label">
                {step.queries ? <Search size={14} aria-hidden="true" /> : <FileText size={14} aria-hidden="true" />}
                {step.label}
              </span>
              {step.queries && <ul className="chat-activity-queries">
                {step.queries.map((query) => <li key={query}>{query}</li>)}
              </ul>}
              {step.pages && <ul className="chat-activity-pages">
                {step.pages.map((page) => <li key={page.url}>
                  <a href={page.url} target="_blank" rel="noopener noreferrer">
                    {page.title} <ExternalLink size={12} aria-hidden="true"/>
                  </a>
                  {page.excerpt && <p>{page.excerpt}</p>}
                </li>)}
              </ul>}
            </>}
          </li>
        ))}
      </ol>
      {timings.length > 0 && <details className="chat-activity-timings">
        <summary>阶段耗时</summary>
        <ul>{timings.map((step, index) => <li key={index}>{step.slice("阶段耗时：".length)}</li>)}</ul>
      </details>}
    </details>
  );
}
