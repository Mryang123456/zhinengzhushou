import { useEffect, useState } from "react";
import { ChatHeader } from "../../components/ChatHeader";
import { Composer } from "../../components/Composer";
import { ConnectionSettings } from "../../components/ConnectionSettings";
import { MessageList } from "../../components/MessageList";
import { Sidebar } from "../../components/Sidebar";
import { useConnectionStatus } from "../../hooks/useConnectionStatus";
import { loadRemoteSessions, login } from "../../services/agentApi";
import { loadSessions, sessionStorageKey } from "../../services/sessionStorage";
import type { Session } from "../../types";
import { useChatMessages } from "./useChatMessages";
import { useChatSessions } from "./useChatSessions";
import "./style.css";

const API_URL = import.meta.env.VITE_AGENT_API_URL ?? "http://127.0.0.1:8000";

type AccountChatProps = {
  storageKey: string;
  userId: string;
  accessToken: string;
  apiKey: string;
  setApiKey: (value: string) => void;
  loginUser: (username: string, password: string) => Promise<void>;
};

export function ChatPage() {
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
  return <AccountChat {...props} savedSessions={savedSessions} />;
}

function AccountChat({ storageKey, userId, accessToken, apiKey, setApiKey, loginUser, savedSessions }: AccountChatProps & { savedSessions: Session[] | null }) {
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const [restored, setRestored] = useState(Boolean(savedSessions));
  const connection = useConnectionStatus(API_URL, apiKey, accessToken);
  const chat = useChatSessions(storageKey, savedSessions);
  const { syncSessions } = chat;
  useEffect(() => {
    if (!accessToken) return;
    const controller = new AbortController();
    const sync = () => {
      void loadRemoteSessions(API_URL, apiKey, accessToken, controller.signal)
        .then((sessions) => { if (!controller.signal.aborted) syncSessions(sessions); })
        .catch(() => { /* 本地记录仍可用，下一次窗口聚焦重试。 */ });
    };
    sync();
    window.addEventListener("focus", sync);
    return () => {
      controller.abort();
      window.removeEventListener("focus", sync);
    };
  }, [accessToken, apiKey, syncSessions]);
  const { activeSession } = chat;
  const messages = useChatMessages({
    apiUrl: API_URL, apiKey, accessToken, activeSession,
    updateMessages: chat.updateMessages,
    refreshConnection: connection.refresh,
  });

  const closeMobileSidebar = () => {
    if (window.innerWidth <= 760) setSidebarOpen(false);
  };

  return <main className="app-shell">
    <Sidebar
      open={sidebarOpen}
      sessions={chat.visibleSessions}
      activeId={activeSession.id}
      runningIds={messages.runningIds}
      onNew={() => { chat.createSession(); closeMobileSidebar(); }}
      onSelect={(id) => { chat.selectSession(id); closeMobileSidebar(); }}
      onArchive={(id) => { if (!messages.isRunning(id)) chat.archiveSession(id); }}
      onSettings={() => setShowSettings(true)}
      onClose={() => setSidebarOpen(false)}
    />
    <section className="app-chat-panel">
      <ChatHeader
        sidebarOpen={sidebarOpen}
        userId={userId}
        threadTitle={activeSession.title}
        connectionStatus={connection.status}
        onThreadTitle={chat.renameSession}
        onToggleSidebar={() => setSidebarOpen((open) => !open)}
        onConnection={() => setShowSettings(true)}
      />
      {restored ? <div className="app-restore-banner">已恢复本地会话。敏感登录令牌不会被保存。<button onClick={() => setRestored(false)}>知道了</button></div> : null}
      {chat.saveFailed ? <div className="app-restore-banner" role="alert">浏览器无法保存聊天记录，刷新前请复制需要保留的内容。</div> : null}
      <MessageList
        key={`messages:${activeSession.id}`}
        messages={activeSession.messages}
        onConfirmApproval={messages.confirmApproval}
        retryMessageId={messages.retryMessageId}
        onRetry={messages.retryLastReply}
      />
      <Composer
        key={activeSession.id}
        value={chat.draft}
        isSending={messages.isSending}
        onChange={chat.setDraft}
        onSubmit={(value) => { if (messages.submit(value)) chat.setDraft(""); }}
        onStop={messages.stop}
        onTool={() => chat.appendPrompt("请帮我查资料：")}
        onMemory={() => chat.appendPrompt("请记住：")}
      />
    </section>
    {showSettings ? <ConnectionSettings
      apiUrl={API_URL}
      apiKey={apiKey}
      userId={userId}
      connectionStatus={connection.status}
      checking={connection.checking}
      onCheckConnection={connection.refresh}
      onApiKey={setApiKey}
      onLogin={loginUser}
      onClose={() => setShowSettings(false)}
    /> : null}
  </main>;
}
