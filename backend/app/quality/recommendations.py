"""验证每个景点的必需条件，并仅输出有对应证据的候选。"""

import asyncio
import json
import os
import re
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field

from backend.app.content_safety import contains_secret, is_direct_prompt_override, sanitize_untrusted_text


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=2, max_length=60)
    source: int = Field(ge=1, le=8)
    quote: str = Field(min_length=4, max_length=220)


class Recommendations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[Candidate] = Field(max_length=8)


INDOOR = re.compile(r"室内|indoor", re.I)
FAMILY = re.compile(r"亲子|适合(?:带)?(?:孩子|儿童|小朋友)|儿童(?:游乐|馆|区|体验)|family.friendly|for (?:kids|children)", re.I)
NEGATED = re.compile(r"不适合|不推荐|不宜|不对儿童|室外为主|非室内|not suitable|not indoor", re.I)


def recommendation_constraints(request: str) -> tuple[bool, bool]:
    return bool(INDOOR.search(request)), bool(re.search(r"孩子|小孩|儿童|小朋友|亲子", request))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch_excerpt(url: str, key: str, location: str) -> str:
    request = Request(
        "https://api.deepseek.com/v1/tools/fetch",
        data=json.dumps({"url": url}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with build_opener(_NoRedirect()).open(request, timeout=4) as response:
            body = json.loads(response.read(256_000)).get("content", "")
    except (OSError, ValueError, AttributeError):
        return ""
    if not isinstance(body, str):
        return ""
    chunks = re.split(r"[\n。！？]+", sanitize_untrusted_text(body, 60_000))
    relevant = [chunk for chunk in chunks if INDOOR.search(chunk) or FAMILY.search(chunk) or (location and location in chunk)]
    return "。".join(" ".join(chunk.split()) for chunk in relevant[:16])[:1800]


async def enrich_recommendation_sources(sources: list[dict], location: str, *, indoor: bool, family: bool) -> list[dict]:
    if os.getenv("AGENT_DEEPSEEK_FETCH_ENABLED", "").lower() not in {"1", "true"}:
        return sources
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return sources
    candidates = [
        (index, source) for index, source in enumerate(sources)
        if (indoor and not INDOOR.search(source["excerpt"])) or (family and not FAMILY.search(source["excerpt"]))
    ][:3]
    excerpts = await asyncio.gather(*(
        asyncio.to_thread(_fetch_excerpt, source["url"], key, location) for _, source in candidates
    ))
    enriched = list(sources)
    for (index, source), excerpt in zip(candidates, excerpts):
        if excerpt:
            enriched[index] = {**source, "excerpt": source["excerpt"].rstrip("。") + "。" + excerpt, "kind": "正文或摘要"}
    return enriched


def verified_recommendations(plan: Recommendations, sources: list[dict], location: str, indoor: bool, family: bool, count: int) -> list[str]:
    lines = []
    seen = set()
    for item in plan.items:
        name, quote = item.name.strip(), " ".join(item.quote.split())
        if item.source > len(sources) or name in seen or not re.fullmatch(r"[\w ·（）()\-]{2,60}", name):
            continue
        source = sources[item.source - 1]
        if source["kind"] == "仅标题" or not quote or contains_secret(quote) or is_direct_prompt_override(quote):
            continue
        if re.search(r"[^\w\s，。；、（）()·,.：:\-—]", quote) or re.search(r"[。；;]|\.\s", quote.rstrip("。.")):
            continue
        sentences = re.split(r"(?<=[。！？；])|(?<=\.)\s+", source["excerpt"])
        if not any(quote.rstrip("。.") == sentence.strip().rstrip("。.") for sentence in sentences):
            continue
        if name not in quote or NEGATED.search(quote):
            continue
        if location and location not in quote:
            continue
        if (indoor and not INDOOR.search(quote)) or (family and not FAMILY.search(quote)):
            continue
        seen.add(name)
        lines.append(f"- **{name}**：{quote} [来源]({source['url']})")
        if len(lines) >= count:
            break
    return lines
