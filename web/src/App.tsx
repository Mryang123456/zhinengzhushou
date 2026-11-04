import { FormEvent, useCallback, useMemo, useState } from "react";
import { ChatHeader } from "./components/ChatHeader";
import { Composer } from "./components/Composer";
import { ConnectionSettings } from "./components/ConnectionSettings";
import { MessageList } from "./components/MessageList";
import { Sidebar } from "./components/Sidebar";
import { useAgentStream } from "./hooks/useAgentStream";
import type { Message, Session, StreamEvent } from "./types";

const API_URL = import.meta.env.VITE_AGENT_API_URL ?? "http://127.0.0.1:8000";
const welcome: Message = { id: "welcome", role: "assistant", content: "你好，我是智行助手。你可以让我查询天气、搜索资料、计算表达式，或在需要时明确告诉我“记住这件事”。" };
const createId = () => crypto.randomUUID();
const newSession = (title = `会话-${new Date().toISOString().slice(0, 10)}`): Session => ({ id: createId(), title, messages: [{ ...welcome, id: createId() }] });

export default function App() {
  const [sessions, setSessions] = useState<Session[]>(() => [newSession(), newSession("产品调研"), newSession("工具调用示例")]);
  const [activeId, setActiveId] = useState(() => sessions[0].id);
  const [question, setQuestion] = useState(""); const [userId, setUserId] = useState("开发者A"); const [apiKey, setApiKey] = useState("");
  const [showSettings, setShowSettings] = useState(false); const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const { isSending, send, stop } = useAgentStream(API_URL);
  const activeSession = useMemo(() => sessions.find((session) => session.id === activeId) ?? sessions[0], [activeId, sessions]);
  const updateMessages = useCallback((sessionId: string, updater: (messages: Message[]) => Message[]) => setSessions((items) => items.map((item) => item.id === sessionId ? { ...item, messages: updater(item.messages) } : item)), []);
  const updateMessage = useCallback((sessionId: string, messageId: string, event: StreamEvent) => updateMessages(sessionId, (items) => items.map((item) => {
    if (item.id !== messageId) return item;
    if (event.type === "token") return { ...item, content: item.content + (event.content ?? "") };
    if (event.type === "tool_start") return { ...item, toolEvents: [...(item.toolEvents ?? []), `正在调用：${(event.tools ?? ["工具"]).join("、")}`] };
    if (event.type === "tool_end") return { ...item, toolEvents: [...(item.toolEvents ?? []), `调用完成：${(event.tools ?? ["工具"]).join("、")}`] };
    if (event.type === "error") return { ...item, content: event.message ?? "请求失败，请稍后重试。", pending: false };
    if (event.type === "done") return { ...item, content: event.content || item.content, pending: false };
    return item;
  })), [updateMessages]);
  const selectSession = (id: string) => { setActiveId(id); if (window.innerWidth <= 760) setSidebarOpen(false); };
  const createSession = () => { if (isSending) return; const session = newSession(`会话-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`); setSessions((items) => [session, ...items]); setActiveId(session.id); setQuestion(""); if (window.innerWidth <= 760) setSidebarOpen(false); };
  const renameSession = (title: string) => setSessions((items) => items.map((item) => item.id === activeId ? { ...item, title } : item));
  const submit = async (event: FormEvent) => { event.preventDefault(); const text = question.trim(); if (!text || isSending) return; const sessionId = activeSession.id; const assistantId = createId(); updateMessages(sessionId, (items) => [...items, { id: createId(), role: "user", content: text }, { id: assistantId, role: "assistant", content: "", toolEvents: [], pending: true }]); setQuestion(""); try { await send({ question: text, threadId: activeSession.title, userId, apiKey }, (streamEvent) => updateMessage(sessionId, assistantId, streamEvent)); } catch (error) { const content = (error as Error).name === "AbortError" ? "已停止生成。" : `连接失败：${(error as Error).message}`; updateMessage(sessionId, assistantId, { type: "error", message: content }); } };
  return <main className="app-shell"><Sidebar open={sidebarOpen} sessions={sessions} activeId={activeId} onNew={createSession} onSelect={selectSession} onSettings={() => setShowSettings(true)} onClose={() => setSidebarOpen(false)}/><section className="chat-panel"><ChatHeader sidebarOpen={sidebarOpen} userId={userId} threadTitle={activeSession.title} onUserId={setUserId} onThreadTitle={renameSession} onToggleSidebar={() => setSidebarOpen((open) => !open)}/><MessageList messages={activeSession.messages}/><Composer value={question} isSending={isSending} onChange={setQuestion} onSubmit={submit} onStop={stop}/></section>{showSettings ? <ConnectionSettings apiUrl={API_URL} apiKey={apiKey} onApiKey={setApiKey} onClose={() => setShowSettings(false)}/> : null}</main>;
}
