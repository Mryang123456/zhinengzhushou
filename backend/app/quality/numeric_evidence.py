"""用正文中的同一陈述核对数值及其限定条件。"""

import re
import unicodedata
from decimal import Decimal

DATE = re.compile(r"(?<!\d)(\d{4})[-/年]\s*(\d{1,2})[-/月]\s*(\d{1,2})(?:日|号)?(?!\d)")
PERIOD = re.compile(r"(?:(\d{4})(?:财年|年)?\s*)?(?:第?([一二三四1-4])季度|Q([1-4]))", re.I)
NUMBER = re.compile(
    r"(?<![\d.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*"
    r"(?P<unit>个百分点|bp\b|%|(?:万亿|亿|万|千)(?:美元|人民币|欧元|港元|元|人|吨)?|美元|人民币|欧元|港元|元|吨|公里|千克|毫克|克|米|人|倍)?",
    re.I,
)
METRIC = re.compile(r"营业收入|营业额|营收|净利润|净利|毛利润|毛利率|毛利|净利率|营业利润|利润|成本|销量|销售额|出货量|收入|亏损")
ALIASES = {"营业收入": "营收", "营业额": "营收", "净利": "净利润", "毛利": "毛利润"}
DIRECTION = re.compile(r"同比|环比|增长|上升|增加|上涨|下降|减少|下滑|下跌")
DIRECTION_ALIASES = {"上升": "增长", "增加": "增长", "上涨": "增长", "减少": "下降", "下滑": "下降", "下跌": "下降"}
ENTITY_PREFIX = re.compile(r"^(?:据(?:该|此)?(?:报道|公告|财报)[，,:：]?|根据(?:该|此)?(?:报道|公告|财报)[，,:：]?|其中[，,]?|数据显示[，,:：]?)+")
CLAUSE_BREAK = re.compile(r"[;；。！？!\n]|[，,](?!\d{3}(?!\d))")
CHANGE_RANGE = re.compile(
    r"(?:从|由)\s*([+-]?\d[\d,.]*\s*[^\d,;；。]*?)"
    r"(?:降至|降到|下降到|增至|增到|增长到|升至|升到|涨至|涨到|变为|调整为|调整至|到|至)(?=\s*[+-]?\d)"
)
COUNT_STATEMENT = re.compile(
    r"^(?P<subject>[\u4e00-\u9fff]{1,18}?)(?P<action>新增|新建|开设|增设|建成|减少|关闭)"
    r"\s*\d+\s*(?P<object>[\u4e00-\u9fff]+)"
)


def _normal(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().casefold()


def _periods(text: str) -> set[tuple[str, int]]:
    return {(year or "", "一二三四".index(q) + 1 if q in "一二三四" else int(q))
            for year, chinese, english in PERIOD.findall(text) if (q := chinese or english)}


def _dates(text: str) -> set[tuple[int, ...]]:
    return {tuple(map(int, match)) for match in DATE.findall(text)}


def _numbers(text: str) -> set[tuple[Decimal, str]]:
    text = PERIOD.sub(" ", DATE.sub(" ", text))
    values = set()
    for match in NUMBER.finditer(text):
        unit = match["unit"] or ""
        number = match.group()[:-len(unit)] if unit else match.group()
        values.add((Decimal(number.strip().replace(",", "")), unit))
    return values


def _metrics(text: str) -> set[str]:
    return {ALIASES.get(metric, metric) for metric in METRIC.findall(text)}


def _financial_subject(text: str) -> str:
    metric = METRIC.search(text)
    if not metric:
        return ""
    prefix = PERIOD.sub("", DATE.sub("", text[:metric.start()]))
    prefix = ENTITY_PREFIX.sub("", prefix)
    return re.sub(r"(?:的|披露|公布|报告|录得|实现|财报|显示|[\s,:：()])+?$", "", prefix).strip()


def _directions(text: str) -> tuple[str, ...]:
    return tuple(DIRECTION_ALIASES.get(item, item) for item in DIRECTION.findall(text))


def _numeric_roles_match(claim: str, source: str) -> bool:
    prior = CHANGE_RANGE.search(source)
    claimed_range = CHANGE_RANGE.search(claim)
    if claimed_range:
        return bool(prior and _numbers(claimed_range[1]) == _numbers(prior[1])
                    and _numbers(claim[claimed_range.end():]) == _numbers(source[prior.end():]))
    if prior:
        return _numbers(claim) <= _numbers(source[prior.end():])
    return True


def supports_numeric_claim(claim: str, statements: set[str], publication_dates: set[str]) -> bool:
    claim = _normal(claim)
    if not re.search(r"\d", claim):
        return True
    # 保留已支持的来源日期标注。附在句尾括号里的日期可来自发布时间。
    publication_dates = {value for item in publication_dates for value in _dates(_normal(item))}
    trailing_date = re.search(r"\(([^()]*)\)$", claim)
    if trailing_date and DATE.fullmatch(trailing_date[1]) and _dates(trailing_date[1]) <= publication_dates:
        claim = claim[:trailing_date.start()]
    return any(_supports_clause(claim, _normal(candidate)) for candidate in statements)


def _supports_clause(claim: str, source: str) -> bool:
    dates, periods, numbers = _dates(claim), _periods(claim), _numbers(claim)
    if not (dates or periods or numbers):
        return True
    metrics, subject, directions = _metrics(claim), _financial_subject(claim), _directions(claim)
    # 若同一句包含多项财务数字，先按分句定位指标与金额，避免跨分句串用。
    if metrics:
        return any(
            metrics <= _metrics(part) and (not subject or subject == _financial_subject(part))
            and dates <= _dates(part) and periods <= _periods(part) and numbers <= _numbers(part)
            and _numeric_roles_match(claim, part)
            and (not directions or directions == _directions(part))
            and (len(METRIC.findall(part)) == 1 or claim.strip() in part)
            for part in CLAUSE_BREAK.split(source)
        )
    count_claim = COUNT_STATEMENT.match(ENTITY_PREFIX.sub("", claim))
    if count_claim:
        return any(
            candidate.groupdict() == count_claim.groupdict()
            and dates <= _dates(part) and periods <= _periods(part) and numbers <= _numbers(part)
            for part in CLAUSE_BREAK.split(source) if (candidate := COUNT_STATEMENT.match(ENTITY_PREFIX.sub("", part.strip())))
        )
    return dates <= _dates(source) and periods <= _periods(source) and numbers <= _numbers(source) and _numeric_roles_match(claim, source)
