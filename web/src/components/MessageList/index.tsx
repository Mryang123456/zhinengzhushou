import { ArrowDown, Bot, Copy, RotateCcw, UserRound } from "lucide-react";
import { Bubble } from "@ant-design/x";
import { Button } from "antd";
import { useLayoutEffect, useRef, useState } from "react";
import type { Message } from "../../types";
import "./style.css";

type Props = {
  messages: Message[];
  onConfirmApproval: (messageId: string) => void;
  retryMessageId?: string;
  onRetry: () => void;
};

function isNearBottom(viewport: HTMLDivElement) {
  return viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop <= 40;
}

export function MessageList({ messages, onConfirmApproval, retryMessageId, onRetry }: Props) {
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
    const steps = message.toolEvents ?? [];
    const status = steps.at(-1) ?? (message.retryable ? "正在重试…" : "正在发送…");
    return <article className={`chat-messages-message chat-messages-${message.role}`} key={message.id}>
      <div className="chat-messages-avatar">{message.role === "user" ? <UserRound size={18}/> : <Bot size={20}/>}</div>
      <div className="chat-messages-message-body">
        <div className="chat-messages-message-name">{message.role === "user" ? "你" : "智行助手"}</div>
        {message.pending ? <div className="chat-messages-progress" role="status">{message.content && !message.retryable ? "正在展示回答…" : status}</div> : null}
        {steps.length > 0 ? <details className="chat-messages-steps">
          <summary>处理过程</summary>
          <ol>{steps.map((step, index) => <li key={index}>{step}</li>)}</ol>
        </details> : null}
        {message.content ? <Bubble
          placement={message.role === "user" ? "end" : "start"}
          variant={message.role === "user" ? "filled" : "borderless"}
          classNames={{ content: "chat-messages-bubble-content" }}
          content={<>{message.content}{message.pending && !message.retryable ? <span className="chat-messages-cursor" aria-hidden="true"/> : null}</>}
        /> : null}
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
          ? <button className="chat-messages-copy" onClick={() => void navigator.clipboard.writeText(message.content)}><Copy size={14}/>复制</button>
          : null}
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
