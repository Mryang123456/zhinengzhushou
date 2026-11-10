import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChatHeader } from "../components/ChatHeader";
import { Composer } from "../components/Composer";
import { ConnectionSettings } from "../components/ConnectionSettings";
import { MessageList } from "../components/MessageList";
import { Sidebar } from "../components/Sidebar";
import { useAgentStream } from "../hooks/useAgentStream";
import { useConnectionStatus } from "../hooks/useConnectionStatus";
import { confirmMcpApproval, login } from "../services/agentApi";
import { loadSessions, saveSessions, sessionStorageKey } from "../services/sessionStorage";
import { archiveSessionState, createSessionTitle, ensureVisibleSession, resolveVisibleSession } from "../utils/session";
import type { Message, ProgressStage, Session, StreamEvent } from "../types";
import "./style.css";

const API_URL = import.meta.env.VITE_AGENT_API_URL ?? "http://127.0.0.1:8000";
const welcome: Message = { id: "welcome", role: "assistant", content: "你好，我是智行助手。你可以让我查询天气、搜索资料、计算表达式，或在需要时明确告诉我“记住这件事”。" };
const createId = () => crypto.randomUUID();
const progressLabels: Record<ProgressStage, string> = {
  preparing: "已收到，正在准备",
  understanding: "正在分析问题",
  context: "正在整理较早的对话",
  drafting: "正在组织回答",
  checking: "正在校验回答",
};
const newSession = (title = `会话-${new Date().toISOString().slice(0, 10)}`): Session => ({ id: createId(), title, messages: [{ ...welcome, id: createId() }], updatedAt: Date.now() });

export default function App() {
  const [identity, setIdentity] = useState<Awaited<ReturnType<typeof login>> | null>(null);
  const [apiKey, setApiKey] = useState("");
  const storageKey = sessionStorageKey(API_URL, identity?.userId ?? null);
  const loginUser = async (username: string, password: string) => {
    setIdentity(await login(API_URL, username, password));
  };

  return <AccountChatLoader
    key={storageKey}
    storageKey={storageKey}
    userId={identity?.userId ?? "本地开发用户"}
    accessToken={identity?.accessToken ?? ""}
    apiKey={apiKey}
    setApiKey={setApiKey}
    loginUser={loginUser}
  />;
}

type AccountChatProps = {
  storageKey: string;
  userId: string;
  accessToken: string;
  apiKey: string;
  setApiKey: (value: string) => void;
  loginUser: (username: string, password: string) => Promise<void>;
};

function AccountChatLoader(props: AccountChatProps) {
  const { storageKey } = props;
  const [savedSessions, setSavedSessions] = useState<Session[] | null>();
  useEffect(() => {
    let active = true;
    void loadSessions(storageKey).then((sessions) => {
      if (active) setSavedSessions(sessions);
    });
    return () => { active = false; };
  }, [storageKey]);

  if (savedSessions === undefined) return <main className="app-restore-banner" role="status">正在读取本地会话…</main>;
  return <AccountChat {...props} savedSessions={savedSessions}/>;
}

function AccountChat({ storageKey, userId, accessToken, apiKey, setApiKey, loginUser, savedSessions }: AccountChatProps & { savedSessions: Session[] | null }) {
  const [sessions, setSessions] = useState<Session[]>(() => ensureVisibleSession(savedSessions ?? [], newSession));
  const savedSnapshot = useRef(sessions);
  const saveQueue = useRef(Promise.resolve());
  const [selectedId, setActiveId] = useState<string>();
  const [drafts, setDrafts] = useState<ReadonlyMap<string, string>>(() => new Map());
  const [showSettings, setShowSettings] = useState(false); const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const [restored, setRestored] = useState(Boolean(savedSessions));
  const [saveFailed, setSaveFailed] = useState(false);
  const { runningIds, isRunning, send, stop } = useAgentStream(API_URL);
  const connection = useConnectionStatus(API_URL, apiKey, accessToken);
  // 初始化与归档操作都保证至少保留一个可用会话。
  const activeSession = useMemo(() => resolveVisibleSession(sessions, selectedId)!, [selectedId, sessions]);
  const activeId = activeSession.id;
  const isSending = runningIds.has(activeSession.id);
  const question = drafts.get(activeSession.id) ?? "";
  const lastReply = activeSession.messages.at(-1);
  const lastQuestion = activeSession.messages.at(-2);
  const canRetry = !isSending && lastReply?.role === "assistant" && !lastReply.pending
    && lastReply.retryable && !lastReply.approval && lastQuestion?.role === "user";
  useEffect(() => {
    saveQueue.current = saveQueue.current.then(async () => {
      if (savedSnapshot.current === sessions) return;
      try {
        await saveSessions(storageKey, sessions, savedSnapshot.current);
        savedSnapshot.current = sessions;
        setSaveFailed(false);
      } catch {
        setSaveFailed(true);
      }
    });
  }, [storageKey, sessions]);
  const updateMessages = useCallback((sessionId: string, updater: (messages: Message[]) => Message[]) => setSessions((items) => items.map((item) => item.id === sessionId ? { ...item, messages: updater(item.messages), updatedAt: Date.now() } : item)), []);
  const updateMessage = useCallback((sessionId: string, messageId: string, event: StreamEvent, replaceContent = false) => updateMessages(sessionId, (items) => items.map((item) => {
    if (item.id !== messageId) return item;
    if (event.type === "progress" && event.stage && Object.hasOwn(progressLabels, event.stage)) {
      const label = progressLabels[event.stage];
      if (item.toolEvents?.at(-1) === label) return item;
      return { ...item, toolEvents: [...(item.toolEvents ?? []), label] };
    }
    if (event.type === "token") return { ...item, content: (replaceContent ? "" : item.content) + (event.content ?? ""), retryable: false };
    if (event.type === "tool_start") return { ...item, toolEvents: [...(item.toolEvents ?? []), `正在调用：${(event.tools ?? ["工具"]).join("、")}`] };
    if (event.type === "tool_end") return { ...item, toolEvents: [...(item.toolEvents ?? []), `工具已返回：${(event.tools ?? ["工具"]).join("、")}`] };
    if (event.type === "approval_required" && event.approval_id && event.confirmation_token && event.tool && event.risk && event.arguments && event.expires_at) return { ...item, approval: { approvalId: event.approval_id, confirmationToken: event.confirmation_token, tool: event.tool, risk: event.risk, arguments: event.arguments, expiresAt: event.expires_at, status: "pending" }, toolEvents: [...(item.toolEvents ?? []), `等待确认：${event.tool}（${event.risk}）`] };
    if (event.type === "error") return { ...item, notice: event.message ?? "请求失败，请稍后重试。", pending: false, retryable: event.retryable ?? true };
    if (event.type === "done") return { ...item, content: event.content || item.content, pending: false, retryable: false };
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
      void connection.refresh();
    }
  };
  const updateDraft = (sessionId: string, update: (current: string) => string) => {
    setDrafts((current) => {
      const value = update(current.get(sessionId) ?? "");
      const next = new Map(current);
      if (value) next.set(sessionId, value);
      else next.delete(sessionId);
      return next;
    });
  };
  const appendPrompt = (template: string) => updateDraft(activeSession.id, (current) => current.trim() ? `${current}\n${template}` : template);
  const selectSession = (id: string) => { setActiveId(id); if (window.innerWidth <= 760) setSidebarOpen(false); };
  const createSession = () => { const session = newSession(createSessionTitle()); setSessions((items) => [session, ...items]); setActiveId(session.id); if (window.innerWidth <= 760) setSidebarOpen(false); };
  const renameSession = (title: string) => setSessions((items) => items.map((item) => item.id === activeId ? { ...item, title, updatedAt: Date.now() } : item));
  const archiveSession = (id: string) => { if (isRunning(id)) return; const replacement = newSession(createSessionTitle()); const next = archiveSessionState(sessions, activeId, id, replacement); setSessions(next.sessions); setActiveId(next.activeId); updateDraft(id, () => ""); };
  const runReply = async (sessionId: string, replyId: string, question: Message, retry = false) => {
    let replaceOnFirstToken = retry;
    try {
      await send({ question: question.content, messageId: question.id, threadId: sessionId, apiKey, accessToken, retry }, (event) => {
        const replaceContent = replaceOnFirstToken && event.type === "token" && Boolean(event.content);
        if (replaceContent) replaceOnFirstToken = false;
        updateMessage(sessionId, replyId, event, replaceContent);
      });
    } catch (error) {
      const stopped = error instanceof Error && error.name === "AbortError";
      if (!stopped) void connection.refresh();
      const message = error instanceof Error ? error.message : "请稍后重试。";
      updateMessage(sessionId, replyId, {
        type: "error",
        message: stopped ? "已停止生成。" : message,
        retryable: !stopped,
      });
    }
  };

  const submit = (value: string) => {
    const text = value.trim();
    const sessionId = activeSession.id;
    if (!text || isRunning(sessionId)) return;
    const replyId = createId();
    const question: Message = { id: createId(), role: "user", content: text };
    updateMessages(sessionId, (items) => [
      ...items,
      question,
      { id: replyId, role: "assistant", content: "", pending: true },
    ]);
    updateDraft(sessionId, () => "");
    void runReply(sessionId, replyId, question);
  };

  const retryLastReply = () => {
    const sessionId = activeSession.id;
    if (!canRetry || !lastReply || !lastQuestion || isRunning(sessionId)) return;
    updateMessages(sessionId, (items) => items.map((item) => item.id === lastReply.id
      ? { ...item, pending: true, notice: undefined, toolEvents: [] }
      : item));
    void runReply(sessionId, lastReply.id, lastQuestion, true);
  };

  return <main className="app-shell">
    <Sidebar open={sidebarOpen} sessions={sessions.filter((session) => !session.archived).sort((a, b) => b.updatedAt - a.updatedAt)} activeId={activeId} runningIds={runningIds} onNew={createSession} onSelect={selectSession} onArchive={archiveSession} onSettings={() => setShowSettings(true)} onClose={() => setSidebarOpen(false)}/>
    <section className="app-chat-panel">
      <ChatHeader sidebarOpen={sidebarOpen} userId={userId} threadTitle={activeSession.title} connectionStatus={connection.status} onThreadTitle={renameSession} onToggleSidebar={() => setSidebarOpen((open) => !open)} onConnection={() => setShowSettings(true)}/>
      {restored ? <div className="app-restore-banner">已恢复本地会话。敏感登录令牌不会被保存。<button onClick={() => setRestored(false)}>知道了</button></div> : null}
      {saveFailed ? <div className="app-restore-banner" role="alert">浏览器无法保存聊天记录，刷新前请复制需要保留的内容。</div> : null}
      <MessageList key={`messages:${activeSession.id}`} messages={activeSession.messages} onConfirmApproval={confirmApproval} retryMessageId={canRetry ? lastReply.id : undefined} onRetry={retryLastReply}/>
      <Composer key={activeSession.id} value={question} isSending={isSending} onChange={(value) => updateDraft(activeSession.id, () => value)} onSubmit={submit} onStop={() => stop(activeSession.id)} onTool={() => appendPrompt("请帮我查资料：")} onMemory={() => appendPrompt("请记住：")}/>
    </section>
    {showSettings ? <ConnectionSettings apiUrl={API_URL} apiKey={apiKey} userId={userId} connectionStatus={connection.status} checking={connection.checking} onCheckConnection={connection.refresh} onApiKey={setApiKey} onLogin={loginUser} onClose={() => setShowSettings(false)}/> : null}
  </main>;
}
