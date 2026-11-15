import { memo, useMemo } from "react";
import Markdown, { defaultUrlTransform, type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import "./style.css";

import type { AnswerSource } from "../../types";

const table: Components["table"] = ({ children }) => <div className="answer-markdown-table" tabIndex={0} role="region" aria-label="回答中的表格">
    <table>{children}</table>
  </div>;
const plugins = [remarkGfm];
const blockedElements = ["img", "input"];

function linkUrl(url: string) {
  return /^https?:\/\//i.test(url) ? defaultUrlTransform(url) : "";
}

export default memo(function AnswerMarkdown({ content, sources }: { content: string; sources?: AnswerSource[] }) {
  const components = useMemo<Components>(() => {
    const titles = new Map(sources?.map(({ url, title }) => [url, title]));
    return {
      table,
      a: ({ href, children }) => {
        if (!href) return <>{children}</>;
        const label = typeof children === "string" ? children : "";
        const citation = /^(?:来源\s*)?(\d+)$/.exec(label);
        const title = titles.get(href);
        if (citation && (title || label.startsWith("来源"))) {
          const description = title ? `来源 ${citation[1]} · ${title}` : `来源 ${citation[1]}`;
          return <a className="answer-markdown-citation" href={href} target="_blank" rel="noopener noreferrer"
            title={description} aria-label={`打开${description}`}>{citation[1]}</a>;
        }
        return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
      },
    };
  }, [sources]);
  return <div className="answer-markdown">
    <Markdown skipHtml urlTransform={linkUrl} disallowedElements={blockedElements} components={components} remarkPlugins={plugins}>
      {content}
    </Markdown>
  </div>;
});
