"""通过配置的正文获取服务补全搜索结果。"""

import asyncio
import json
import os
import re
import time
from ipaddress import ip_address
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from backend.app.content_safety import contains_secret, sanitize_untrusted_text
from backend.app.observability import observe_duration
from backend.app.quality.web_passages import extract_web_passages


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_page_text(url: str, key: str) -> str:
    if not public_result_url(url):
        return ""
    request = Request(
        "https://api.deepseek.com/v1/tools/fetch",
        data=json.dumps({"url": url}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with build_opener(_NoRedirect()).open(request, timeout=4) as response:
            body = json.loads(response.read(256_000)).get("content", "")
    except (OSError, ValueError, AttributeError):
        return ""
    finally:
        observe_duration("web_page_fetch", time.monotonic() - started)
    return sanitize_untrusted_text(body, 60_000) if isinstance(body, str) else ""


async def enrich_search_results(content: str, query: str, *, read_pages: list[dict[str, str]] | None = None) -> str:
    """为本轮搜索结果补正文；失败时保留原搜索结果。"""
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key or os.getenv("AGENT_DEEPSEEK_FETCH_ENABLED", "").lower() not in {"1", "true"}:
        return content
    try:
        records = json.loads(content)
    except ValueError:
        return content
    if not isinstance(records, list):
        return content
    candidates = {}
    for item in records:
        url = item.get("href") if isinstance(item, dict) else None
        if isinstance(url, str) and public_result_url(url) and not contains_secret(url):
            candidates.setdefault(url, item)
    semaphore = asyncio.Semaphore(8)

    async def fetch(url: str) -> str:
        async with semaphore:
            return await asyncio.to_thread(fetch_page_text, url, key)

    pages = await asyncio.gather(*(fetch(url) for url in candidates))
    for (url, item), page in zip(candidates.items(), pages):
        if page.strip() and read_pages is not None:
            domain = urlparse(url).hostname or ""
            title = " ".join(sanitize_untrusted_text(item.get("title") or "", 180).split())
            read_pages.append({"url": url, "title": title or domain, "domain": domain})
        excerpt = extract_web_passages(page, query, limit=1200, require_match=True)
        if excerpt:
            original = str(item.get("body") or "").strip()
            item["body"] = "\n".join(part for part in (excerpt, original) if part)
    return json.dumps(records, ensure_ascii=False)


def public_result_url(url: str) -> bool:
    if len(url) > 8192 or not re.fullmatch(r"https?://[^\s<>\\\"']+", url):
        return False
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").rstrip(".").lower()
        if parsed.username is not None or parsed.password is not None or parsed.port not in (None, 80, 443):
            return False
    except ValueError:
        return False
    try:
        return ip_address(host).is_global
    except ValueError:
        return "." in host and not host.endswith((".localhost", ".local", ".internal", ".test")) and bool(re.search(r"[a-z]$", host))
