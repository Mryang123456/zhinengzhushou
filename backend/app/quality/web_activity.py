"""构建用户可见的搜索活动，内容来自真实查询与已选网页摘录。"""

from urllib.parse import urlparse

from backend.app.content_safety import sanitize_untrusted_text
from backend.app.quality.validator import web_answer_sources
from backend.app.quality.web_fetch import public_result_url
from backend.app.quality.web_passages import extract_web_passages


def search_queries(calls: list[dict], allowed: set[str]) -> list[str]:
    queries = {}
    for call in calls:
        query = call["args"].get("query")
        if call["name"] in allowed and isinstance(query, str):
            text = " ".join(sanitize_untrusted_text(query.strip(), 500).split())
            if text:
                queries[text[:160]] = None
    return list(queries)[:6]


def search_highlights(question: str, messages: list, new_urls: set[str]) -> list[dict[str, str]]:
    highlights = []
    domains = set()
    for source in web_answer_sources(question, messages):
        url = source["url"]
        if url not in new_urls or source["kind"] == "仅标题" or not public_result_url(url):
            continue
        domain = urlparse(url).hostname or ""
        if domain in domains:
            continue
        domains.add(domain)
        excerpt = extract_web_passages(sanitize_untrusted_text(source["excerpt"], 600), question, limit=180)
        highlights.append({
            "url": url, "domain": domain, "title": sanitize_untrusted_text(source["title"], 160) or domain,
            "excerpt": " ".join(excerpt.split()),
        })
        if len(highlights) == 3:
            break
    return highlights
