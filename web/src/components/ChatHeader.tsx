import { ChevronDown, Database, Menu, PanelLeftClose, UserRound } from "lucide-react";
import { connectionLabels, type ConnectionStatus } from "../services/agentApi";

type Props = {
  sidebarOpen: boolean;
  userId: string;
  threadTitle: string;
  connectionStatus: ConnectionStatus;
  onThreadTitle: (value: string) => void;
  onToggleSidebar: () => void;
  onConnection: () => void;
};

export function ChatHeader({ sidebarOpen, userId, threadTitle, connectionStatus, onThreadTitle, onToggleSidebar, onConnection }: Props) {
  const label = connectionLabels[connectionStatus];
  return <header className="topbar">
    <button className="icon-button" onClick={onToggleSidebar} aria-label="切换导航">{sidebarOpen ? <PanelLeftClose size={20}/> : <Menu size={20}/>}</button>
    <div className="context-select"><UserRound size={17}/><input aria-label="当前登录用户" value={userId} readOnly/><ChevronDown size={16}/></div>
    <span className="divider"/>
    <div className="context-select thread"><Database size={17}/><input aria-label="会话线程" value={threadTitle} onChange={(e) => onThreadTitle(e.target.value)}/><ChevronDown size={16}/></div>
    <button className="connection connection-state" data-status={connectionStatus} onClick={onConnection} aria-label="打开连接与登录设置" aria-description={label} title={label}>
      <i aria-hidden="true"/><span aria-live="polite">{label}</span>
    </button>
  </header>;
}
