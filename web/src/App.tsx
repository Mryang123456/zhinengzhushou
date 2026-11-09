import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { ChatHeader } from "./components/ChatHeader";
import { Composer } from "./components/Composer";
import { ConnectionSettings } from "./components/ConnectionSettings";
import { MessageList } from "./components/MessageList";
import { Sidebar } from "./components/Sidebar";
import { useAgentStream } from "./hooks/useAgentStream";
import { confirmMcpApproval, login } from "./services/agentApi";
import { loadSessions, saveSessions } from "./services/sessionStorage";
import { createSessionTitle } from "./utils/session";
import type { Message, Session, StreamEvent } from "./types";

const API_URL = import.meta.env.VITE_AGENT_API_URL ?? "http://127.0.0.1:8000";
const welcome: Message = { id: "welcome", role: "assistant", content: "你好，我是智行助手。你可以让我查询天气、搜索资料、计算表达式，或在需要时明确告诉我“记住这件事”。" };
const createId = () => crypto.randomUUID();
const newSession = (title = `会话-${new Date().toISOString().slice(0, 10)}`): Session => ({ id: createId(), title, messages: [{ ...welcome, id: createId() }], updatedAt: Date.now() });

export default function App() {
  const [sessions, setSessions] = useState<Session[]>(() => loadSessions() ?? [newSession()]);
  const [activeId, setActiveId] = useState(() => sessions[0].id);
  const [question, setQuestion] = useState(""); const [userId, setUserId] = useState("本地开发用户"); const [apiKey, setApiKey] = useState(""); const [accessToken, setAccessToken] = useState("");
  const [showSettings, setShowSettings] = useState(false); const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const [restored, setRestored] = useState(() => Boolean(loadSessions()?.length));
  const { isSending, send, stop } = useAgentStream(API_URL);
  const activeSession = useMemo(() => sessions.find((session) => session.id === activeId) ?? sessions[0], [activeId, sessions]);
  useEffect(() => { saveSessions(sessions); }, [sessions]);
  const updateMessages = useCallback((sessionId: string, updater: (messages: Message[]) => Message[]) => setSessions((items) => items.map((item) => item.id === sessionId ? { ...item, messages: updater(item.messages), updatedAt: Date.now() } : item)), []);
  const updateMessage = useCallback((sessionId: string, messageId: string, event: StreamEvent) => updateMessages(sessionId, (items) => items.map((item) => {
    if (item.id !== messageId) return item;
    if (event.type === "token") return { ...item, content: item.content + (event.content ?? "") };
    if (event.type === "tool_start") return { ...item, toolEvents: [...(item.toolEvents ?? []), `正在调用：${(event.tools ?? ["工具"]).join("、")}`] };
    if (event.type === "tool_end") return { ...item, toolEvents: [...(item.toolEvents ?? []), `调用完成：${(event.tools ?? ["工具"]).join("、")}`] };
    if (event.type === "approval_required" && event.approval_id && event.confirmation_token && event.tool && event.risk && event.arguments && event.expires_at) return { ...item, approval: { approvalId: event.approval_id, confirmationToken: event.confirmation_token, tool: event.tool, risk: event.risk, arguments: event.arguments, expiresAt: event.expires_at, status: "pending" }, toolEvents: [...(item.toolEvents ?? []), `等待确认：${event.tool}（${event.risk}）`] };
    if (event.type === "error") return { ...item, content: event.message ?? "请求失败，请稍后重试。", pending: false };
    if (event.type === "done") return { ...item, content: event.content || item.content, pending: false };
    return item;
  })), [updateMessages]);
  const confirmApproval = async (messageId: string) => {
    const approval = activeSession.messages.find((message) => message.id === messageId)?.approval;
    if (!approval || approval.status !== "pending") return;
    updateMessages(activeSession.id, (items) => items.map((item) => item.id === messageId ? { ...item, approval: { ...approval, status: "executing" } } : item));
    try {
      const response = await confirmMcpApproval(API_URL, approval.approvalId, approval.confirmationToken, apiKey, accessToken);
      updateMessages(activeSession.id, (items) => items.map((item) => item.id === messageId ? { ...item, approval: { ...approval, status: "completed" }, content: `${item.content}\n\n已执行已确认操作（${approval.tool}）：\n${response.result}` } : item));
    } catch (error) {
      updateMessages(activeSession.id, (items) => items.map((item) => item.id === messageId ? { ...item, approval: { ...approval, status: "failed" }, content: `${item.content}\n\n确认操作失败：${(error as Error).message}` } : item));
    }
  };
  const loginUser = async (username: string, password: string) => { const response = await login(API_URL, username, password); setAccessToken(response.accessToken); setUserId(response.userId); };
  const appendPrompt = (template: string) => setQuestion((current) => current.trim() ? `${current}\n${template}` : template);
  const selectSession = (id: string) => { setActiveId(id); if (window.innerWidth <= 760) setSidebarOpen(false); };
  const createSession = () => { if (isSending) return; const session = newSession(createSessionTitle()); setSessions((items) => [session, ...items]); setActiveId(session.id); setQuestion(""); if (window.innerWidth <= 760) setSidebarOpen(false); };
  const renameSession = (title: string) => setSessions((items) => items.map((item) => item.id === activeId ? { ...item, title, updatedAt: Date.now() } : item));
  const archiveSession = (id: string) => { const remaining = sessions.filter((item) => item.id !== id && !item.archived); if (!remaining.length) { createSession(); } setSessions((items) => items.map((item) => item.id === id ? { ...item, archived: true, updatedAt: Date.now() } : item)); if (id === activeId) setActiveId(remaining[0]?.id ?? activeId); };
  const submit = async (event: FormEvent) => { event.preventDefault(); const text = question.trim(); if (!text || isSending) return; const sessionId = activeSession.id; const assistantId = createId(); updateMessages(sessionId, (items) => [...items, { id: createId(), role: "user", content: text }, { id: assistantId, role: "assistant", content: "", toolEvents: [], pending: true }]); setQuestion(""); try { await send({ question: text, threadId: activeSession.title, apiKey, accessToken }, (streamEvent) => updateMessage(sessionId, assistantId, streamEvent)); } catch (error) { const content = (error as Error).name === "AbortError" ? "已停止生成。" : `连接失败：${(error as Error).message}`; updateMessage(sessionId, assistantId, { type: "error", message: content }); } };
  return <main className="app-shell"><Sidebar open={sidebarOpen} sessions={[...sessions.filter((session) => !session.archived)].sort((a, b) => b.updatedAt - a.updatedAt)} activeId={activeId} onNew={createSession} onSelect={selectSession} onArchive={archiveSession} onSettings={() => setShowSettings(true)} onClose={() => setSidebarOpen(false)}/><section className="chat-panel"><ChatHeader sidebarOpen={sidebarOpen} userId={userId} threadTitle={activeSession.title} onThreadTitle={renameSession} onToggleSidebar={() => setSidebarOpen((open) => !open)} onConnection={() => setShowSettings(true)}/>{restored ? <div className="restore-banner">已恢复本地会话。敏感登录令牌不会被保存。<button onClick={() => setRestored(false)}>知道了</button></div> : null}<MessageList messages={activeSession.messages} onConfirmApproval={confirmApproval}/><Composer value={question} isSending={isSending} onChange={setQuestion} onSubmit={submit} onStop={stop} onTool={() => appendPrompt("请帮我查资料：")} onMemory={() => appendPrompt("请记住：")}/></section>{showSettings ? <ConnectionSettings apiUrl={API_URL} apiKey={apiKey} userId={userId} onApiKey={setApiKey} onLogin={loginUser} onClose={() => setShowSettings(false)}/> : null}</main>;
}
