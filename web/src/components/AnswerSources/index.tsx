import { ExternalLink, Globe } from "lucide-react";
import type { AnswerSource } from "../../types";
import "./style.css";

export function AnswerSources({ sources }: { sources: AnswerSource[] }) {
  const publicLinks = sources.filter(({ url }) => /^https?:\/\//i.test(url));
  if (!publicLinks.length) return null;
  const domains = [...new Set(publicLinks.map(({ domain }) => domain))].filter(Boolean).slice(0, 3);
  return <details className="answer-sources">
    <summary>
      <span className="answer-sources-icons" aria-hidden="true">
        {domains.length ? domains.map((domain) => <span key={domain} title={domain}>{domain.replace(/^www\./, "").charAt(0).toUpperCase()}</span>)
          : <Globe size={15}/>}
      </span>
      <span>{publicLinks.length} 个引用网页</span>
    </summary>
    <ul>
      {publicLinks.map(({ url, title, domain }) => <li key={url}>
        <a href={url} target="_blank" rel="noopener noreferrer">{title}<ExternalLink size={13} aria-hidden="true"/></a>
        <span>{domain}</span>
      </li>)}
    </ul>
  </details>;
}
