import { Archive, Bot, CircleHelp, CirclePlus, Search, Settings2, Sparkles, X } from "lucide-react";
import { useMemo, useState } from "react";
import type { Session } from "../types";
type Props = {
  open: boolean;
  sessions: Session[];
  activeId: string;
  runningIds: ReadonlySet<string>;
  onNew: () => void;
  onSelect: (id: string) => void;
  onArchive: (id: string) => void;
  onSettings: () => void;
  onClose: () => void;
};

export function Sidebar({ open, sessions, activeId, runningIds, onNew, onSelect, onArchive, onSettings, onClose }: Props) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => sessions.filter((session) => session.title.toLowerCase().includes(query.toLowerCase())), [sessions, query]);
  if (!open) return null;
  return <>
    <button className="sidebar-backdrop" aria-label="关闭导航" onClick={onClose}/>
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-icon"><Bot size={22}/></span><strong>智行助手</strong>
        <button className="icon-button mobile-only" onClick={onClose} aria-label="关闭导航"><X size={18}/></button>
      </div>
      <button className="new-chat" onClick={onNew}><CirclePlus size={19}/>新建对话</button>
      <label className="session-search">
        <Search size={16}/>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索历史会话" aria-label="搜索历史会话"/>
      </label>
      <section className="session-nav">
        <p>对话会话</p>
        {filtered.map((session) => {
          const running = runningIds.has(session.id);
          return <div className={`session-row ${session.id === activeId ? "selected" : ""}`} key={session.id}>
            <button className="session" onClick={() => onSelect(session.id)}>
              <Sparkles size={16}/><span>{session.title}</span>
              {running ? <span className="session-running">生成中</span> : null}
            </button>
            <button className="archive" aria-label="归档会话" title={running ? "请先停止生成" : "归档会话"} disabled={running} onClick={() => onArchive(session.id)}>
              <Archive size={14}/>
            </button>
          </div>;
        })}
      </section>
      <div className="sidebar-bottom">
        <button onClick={onSettings}><Settings2 size={18}/>连接设置</button>
        <button><CircleHelp size={18}/>使用说明</button>
      </div>
    </aside>
  </>;
}
