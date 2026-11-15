import { ArrowDown, Copy, Pencil, RotateCcw } from "lucide-react";
import { Bubble } from "@ant-design/x";
import { Button } from "antd";
import { lazy, Suspense, useLayoutEffect, useRef, useState } from "react";
import type { Message } from "../../types";
import { ActivityTimeline } from "../ActivityTimeline";
import { AnswerSources } from "../AnswerSources";
import "./style.css";

const AnswerMarkdown = lazy(() => import("../AnswerMarkdown"));

type Props = {
  messages: Message[];
  onConfirmApproval: (messageId: string) => void;
  retryMessageId?: string;
  onRetry: () => void;
  regenerateMessageId?: string;
  onRegenerate: () => void;
  onEditQuestion?: (content: string) => void;
};

function isNearBottom(viewport: HTMLDivElement) {
  return viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop <= 40;
}

export function MessageList({ messages, onConfirmApproval, retryMessageId, onRetry, regenerateMessageId, onRegenerate, onEditQuestion }: Props) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const [showJump, setShowJump] = useState(false);

  const handleScroll = () => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const nearBottom = isNearBottom(viewport);
    if (following.current === nearBottom) return;
    following.current = nearBottom;
    setShowJump(!nearBottom);
  };

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    const content = contentRef.current;
    if (!viewport || !content) return;
    const followLatest = () => {
      if (following.current) {
        viewport.scrollTop = viewport.scrollHeight;
      } else if (isNearBottom(viewport)) {
        following.current = true;
        setShowJump(false);
      }
    };
    followLatest();
    const observer = new ResizeObserver(followLatest);
    observer.observe(viewport);
    observer.observe(content);
    return () => observer.disconnect();
  }, []);

  const jumpToLatest = () => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    following.current = true;
    viewport.scrollTop = viewport.scrollHeight;
    viewport.focus({ preventScroll: true });
    setShowJump(false);
  };

  const rows = messages.map((message) => {
    return <article className={`chat-messages-message chat-messages-${message.role}`} key={message.id} aria-label={message.role === "user" ? "你的消息" : "智行助手的回答"}>
      <div className="chat-messages-message-body">
        {message.role === "assistant" ? <ActivityTimeline message={message}/> : null}
        {message.content ? <Bubble
          placement={message.role === "user" ? "end" : "start"}
          variant={message.role === "user" ? "filled" : "borderless"}
          classNames={{
            root: "chat-messages-bubble",
            body: "chat-messages-bubble-body",
            content: "chat-messages-bubble-content",
          }}
          content={message.role === "assistant" && !message.pending
            ? <Suspense fallback={message.content}><AnswerMarkdown content={message.content} sources={message.sources}/></Suspense>
            : <>{message.content}{message.pending && !message.retryable ? <span className="chat-messages-cursor" aria-hidden="true"/> : null}</>}
        /> : null}
        {message.role === "assistant" && !message.pending && message.sources?.length
          ? <AnswerSources sources={message.sources}/>
          : null}
        {message.notice ? <div className="chat-messages-notice" role="status">{message.notice}</div> : null}
        {message.id === retryMessageId
          ? <Button icon={<RotateCcw size={15}/>} size="small" onClick={onRetry}>重试</Button>
          : null}
        {message.approval ? <section className="chat-messages-approval-card">
          <strong>需要确认的 MCP 操作</strong>
          <span>工具：{message.approval.tool} · 风险：{message.approval.risk}</span>
          <code>{JSON.stringify(message.approval.arguments)}</code>
          {message.approval.status === "pending"
            ? <Button type="primary" onClick={() => onConfirmApproval(message.id)}>确认并执行</Button>
            : <span className="chat-messages-approval-status">{message.approval.status === "executing" ? "正在执行…" : message.approval.status === "completed" ? "已完成" : "确认失败或已过期"}</span>}
        </section> : null}
        {message.role === "assistant" && message.content && !message.pending
          ? <div className="chat-messages-actions">
            <button className="chat-messages-action" onClick={() => void navigator.clipboard.writeText(message.content)}><Copy size={14}/>复制</button>
            {message.id === regenerateMessageId ? <button className="chat-messages-action" onClick={onRegenerate}>
              <RotateCcw size={14}/>重新生成
            </button> : null}
          </div>
          : null}
        {message.role === "user" && onEditQuestion ? <div className="chat-messages-actions">
          <button className="chat-messages-action" onClick={() => onEditQuestion(message.content)}>
            <Pencil size={14}/>编辑后重发
          </button>
        </div> : null}
      </div>
    </article>;
  });

  return <div className="chat-messages">
    <div className="chat-messages-conversation-scroll" ref={viewportRef} onScroll={handleScroll} tabIndex={0} role="region" aria-label="对话消息">
      <div className="chat-messages-conversation" ref={contentRef}>{rows}</div>
    </div>
    {showJump ? <Button className="chat-messages-jump-to-latest" shape="round" icon={<ArrowDown size={16}/>} onClick={jumpToLatest}>跳到最新</Button> : null}
  </div>;
}
