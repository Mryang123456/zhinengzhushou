"""从网页正文和搜索摘要中提取与本轮问题相关的原文。"""

import re
from unicodedata import normalize


_TERMS = re.compile(r"[\u3400-\u9fff]+|[a-z0-9]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|(?<=\.)(?=\s)|\r?\n")
_TABLE_BLOCK = re.compile(r"(?m)(?:^[^\r\n]*\|[^\r\n]*(?:\r?\n|$)){2,}")


def _terms(text: str) -> set[str]:
    terms = set()
    for token in _TERMS.findall(normalize("NFKC", text).casefold()):
        if re.match(r"[\u3400-\u9fff]", token) and len(token) > 1:
            terms.update(token[index:index + 2] for index in range(len(token) - 1))
        else:
            terms.add(token)
    return terms


def extract_web_passages(
    text: str, question: str, *, query: str = "", limit: int = 600, require_match: bool = False,
) -> str:
    """按问题词及补查词挑选片段，按原文顺序返回。"""
    question_terms = _terms(question)
    query_terms = _terms(query) - question_terms
    windows = []
    for sentence in _SENTENCE_BOUNDARY.split(text):
        sentence = " ".join(sentence.split())
        if len(sentence) <= limit:
            if sentence:
                windows.append(sentence)
        else:
            # 搜索摘要和网页表格可能没有句号。较短的重叠窗口可保留多处命中。
            width = max(1, limit // 2)
            stride = max(1, width * 3 // 4)
            windows.extend(sentence[start:start + width] for start in range(0, len(sentence) - width + stride, stride))

    windows = list(dict.fromkeys(windows))
    candidate_terms = [_terms(window) for window in windows]
    ranked = sorted(
        range(len(windows)),
        key=lambda index: (
            2 * len(question_terms & candidate_terms[index]) + len(query_terms & candidate_terms[index]),
            -index,
        ),
        reverse=True,
    )
    if not ranked:
        return ""
    searched_terms = question_terms | query_terms
    matches = [index for index in ranked if searched_terms & candidate_terms[index]]
    # 场馆、赛程和规格通常逐行列出，表头、单位和对应关系需要一起保留。
    tables = [
        (block, text[max(0, match.start() - 140):match.start()].splitlines())
        for match in _TABLE_BLOCK.finditer(text)
        if len(block := match[0].strip()) <= limit
    ]
    if tables:
        table, table_score = max(
            ((block, len(searched_terms & _terms(" ".join([*context[-2:], block])))) for block, context in tables),
            key=lambda item: item[1],
        )
        if (table_score and table_score >= len(searched_terms & candidate_terms[ranked[0]])) or (
            not matches and not require_match and len(tables) == 1
        ):
            remaining = limit - len(table)
            table_lines = {" ".join(line.split()) for line in table.splitlines()}
            nearby = dict.fromkeys(
                index for start in matches for index in (start, start + 1, start - 1)
                if 0 <= index < len(windows) and windows[index] not in table_lines
            )
            extras = []
            for index in nearby:
                size = len(windows[index]) + 1
                if size <= remaining:
                    extras.append(windows[index])
                    remaining -= size
                if len(extras) == 3:
                    break
            return "\n".join([table, *extras])
    if not matches:
        if require_match:
            return ""
        # 搜索服务也可能用同义词找到匹配。保留少量连续原文，让模型判断。
        matches = list(range(min(3, len(windows))))

    remaining = limit
    selected = []
    # 先保留命中片段，再用紧邻的上下句补足主体、时间或单位。
    neighbors = dict.fromkeys(
        adjacent for index in matches for adjacent in (index - 1, index + 1)
        if 0 <= adjacent < len(windows) and adjacent not in matches
    )
    for index in [*matches, *neighbors]:
        if len(selected) >= 5:
            break
        size = len(windows[index]) + bool(selected)
        if size <= remaining:
            selected.append(index)
            remaining -= size
    return "\n".join(windows[index] for index in sorted(selected))
