import { ChevronDown, Database, Menu, PanelLeftClose, UserRound } from "lucide-react";

type Props = {
  sidebarOpen: boolean;
  userId: string;
  threadTitle: string;
  onUserId: (value: string) => void;
  onThreadTitle: (value: string) => void;
  onToggleSidebar: () => void;
};

export function ChatHeader({ sidebarOpen, userId, threadTitle, onUserId, onThreadTitle, onToggleSidebar }: Props) {
  return <header className="topbar"><button className="icon-button" onClick={onToggleSidebar} aria-label="切换导航">{sidebarOpen ? <PanelLeftClose size={20}/> : <Menu size={20}/>}</button><div className="context-select"><UserRound size={17}/><input aria-label="用户标识" value={userId} onChange={(e) => onUserId(e.target.value)} /><ChevronDown size={16}/></div><span className="divider"/><div className="context-select thread"><Database size={17}/><input aria-label="会话线程" value={threadTitle} onChange={(e) => onThreadTitle(e.target.value)} /><ChevronDown size={16}/></div><span className="connection"><i/>服务已连接</span></header>;
}
