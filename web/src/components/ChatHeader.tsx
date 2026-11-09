import { ChevronDown, Database, Menu, PanelLeftClose, UserRound } from "lucide-react";

type Props = {
  sidebarOpen: boolean;
  userId: string;
  threadTitle: string;
  onThreadTitle: (value: string) => void;
  onToggleSidebar: () => void;
  onConnection: () => void;
};

export function ChatHeader({ sidebarOpen, userId, threadTitle, onThreadTitle, onToggleSidebar, onConnection }: Props) {
  return <header className="topbar"><button className="icon-button" onClick={onToggleSidebar} aria-label="切换导航">{sidebarOpen ? <PanelLeftClose size={20}/> : <Menu size={20}/>}</button><div className="context-select"><UserRound size={17}/><input aria-label="当前登录用户" value={userId} readOnly /><ChevronDown size={16}/></div><span className="divider"/><div className="context-select thread"><Database size={17}/><input aria-label="会话线程" value={threadTitle} onChange={(e) => onThreadTitle(e.target.value)} /><ChevronDown size={16}/></div><button className="connection" onClick={onConnection} aria-label="打开连接与登录设置"><i/>服务已连接</button></header>;
}
