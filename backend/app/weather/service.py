"""基于 Open-Meteo 的通用地点天气实况与多日预报。"""

from __future__ import annotations

import re
from datetime import date as Date
from urllib.parse import urlencode


class WeatherQueryError(ValueError):
    """地点、日期或天气响应无法满足查询。"""


CURRENT_ALIASES = {"", "current", "now", "当前", "现在", "实时"}
RELATIVE_DAY_INDEX = {
    "today": 0, "今天": 0, "今日": 0,
    "tomorrow": 1, "明天": 1, "明日": 1,
    "day after tomorrow": 2, "后天": 2,
}
WEEKDAYS = {
    "一": 0, "1": 0, "二": 1, "2": 1, "三": 2, "3": 2,
    "四": 3, "4": 3, "五": 4, "5": 4, "六": 5, "6": 5,
    "日": 6, "天": 6, "末": 6, "7": 6,
}
DAILY_FIELDS = (
    "weather_code,temperature_2m_max,temperature_2m_min,"
    "apparent_temperature_max,apparent_temperature_min,"
    "precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
)
CURRENT_FIELDS = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,"
    "precipitation,weather_code,wind_speed_10m"
)


def requested_relative_day(question: str) -> str | None:
    """仅在原文给出唯一明确的整日相对日期时覆盖模型参数。"""
    days = {RELATIVE_DAY_INDEX[term] for term in re.findall(r"大后天|今天|今日|明天|明日|后天", question)
            if term in RELATIVE_DAY_INDEX}
    if len(days) != 1 or re.search(r"大后天|昨天|前天|今早|今晚|明早|明晚|(?:星期|周)[一二三四五六日天末\d]|\d{4}-\d{2}-\d{2}|\d+\s*月\s*\d+", question):
        return None
    return ("今天", "明天", "后天")[next(iter(days))]


def clothing_advice(report: str) -> str:
    """按已核验预报的最低体感温度给出分层穿衣建议。"""
    temperatures = re.findall(r"(?:最低体感|体感)\s*(-?\d+(?:\.\d+)?)\s*°C", report)
    if not temperatures:
        temperatures = re.findall(r"(?:最低气温|气温)\s*(-?\d+(?:\.\d+)?)\s*°C", report)
    if not temperatures:
        return "预报缺少可用温度，暂时无法具体建议穿什么。"
    low = min(map(float, temperatures))
    if low < 8:
        layers = "建议穿保暖内层和厚外套。"
    elif low < 17:
        layers = "建议穿长袖和长裤，带一件可穿脱的薄外套。"
    elif low < 24:
        layers = "建议穿轻薄长袖或短袖，怕凉可以带一件薄外套。"
    else:
        layers = "建议穿轻薄、透气的衣服。"
    if re.search(r"雨|雷暴", report) or any(int(value) >= 60 for value in re.findall(r"最大降水概率\s*(\d+)%", report)):
        layers += "带上雨具，鞋子可选防滑、耐湿的款式。"
    return layers


def weather_code_description(code: int) -> str:
    descriptions = {
        0: "晴朗", 1: "大致晴朗", 2: "局部多云", 3: "阴天", 45: "有雾", 48: "雾凇",
        51: "毛毛雨", 53: "毛毛雨", 55: "强毛毛雨", 61: "小雨", 63: "中雨",
        65: "大雨", 71: "小雪", 73: "中雪", 75: "大雪", 80: "阵雨",
        81: "中等阵雨", 82: "强阵雨", 95: "雷暴", 96: "伴冰雹的雷暴", 99: "强冰雹雷暴",
    }
    return descriptions.get(code, f"天气代码 {code}")


def _target_index(requested: str, available: list[str]) -> tuple[str, int]:
    normalized = " ".join(requested.strip().lower().split())
    if normalized in CURRENT_ALIASES:
        return "current", 0
    if normalized in RELATIVE_DAY_INDEX:
        return "forecast", RELATIVE_DAY_INDEX[normalized]

    exact = normalized
    chinese_full = re.fullmatch(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", normalized)
    if chinese_full:
        exact = f"{int(chinese_full[1]):04d}-{int(chinese_full[2]):02d}-{int(chinese_full[3]):02d}"
    chinese_short = re.fullmatch(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", normalized)
    if chinese_short:
        month_day = (int(chinese_short[1]), int(chinese_short[2]))
        for index, item in enumerate(available):
            parsed = Date.fromisoformat(item)
            if (parsed.month, parsed.day) == month_day:
                return "forecast", index
        raise WeatherQueryError("指定日期不在当前可用预报范围内。")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", exact):
        try:
            return "forecast", available.index(exact)
        except ValueError as error:
            raise WeatherQueryError("指定日期不在当前可用预报范围内。") from error

    weekday = re.fullmatch(r"(?:本周|下周|这周|星期|周|礼拜)([一二三四五六日天末1-7])", normalized)
    if weekday:
        target = WEEKDAYS[weekday[1]]
        start = 7 if normalized.startswith("下周") else 0
        for index, item in enumerate(available[start:], start=start):
            if Date.fromisoformat(item).weekday() == target:
                return "forecast", index
    raise WeatherQueryError("日期支持 current、今天、明天、后天、星期表达或 YYYY-MM-DD。")


def _daily_value(daily: dict, field: str, index: int):
    values = daily.get(field)
    if not isinstance(values, list) or index >= len(values):
        raise WeatherQueryError("天气服务返回的预报字段不完整。")
    return values[index]


def query_weather(location: str, requested_date: str, days: int, get_json) -> str:
    """查询任意地点的当前天气或未来1～7天预报。"""
    location = location.strip()
    if not location or len(location) > 80 or not 1 <= days <= 7:
        raise WeatherQueryError("地点不能为空且不超过80字符，预报天数必须在1到7之间。")
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
        {"name": location, "count": 1, "language": "zh", "format": "json"}
    )
    matches = get_json(geocode_url).get("results", [])
    if not matches:
        raise WeatherQueryError("未找到指定地点，请检查地点名称。")
    place = matches[0]
    forecast_url = "https://api.open-meteo.com/v1/forecast?" + urlencode({
        "latitude": place["latitude"], "longitude": place["longitude"],
        "current": CURRENT_FIELDS, "daily": DAILY_FIELDS,
        "forecast_days": 16, "timezone": "auto",
    })
    payload = get_json(forecast_url)
    daily = payload["daily"]
    available = daily["time"]
    mode, start = _target_index(requested_date, available)
    place_name = place["name"]

    if mode == "current" and days == 1:
        current = payload["current"]
        return (
            f"{place_name}当前天气：{weather_code_description(current['weather_code'])}；"
            f"气温 {current['temperature_2m']}°C，体感 {current['apparent_temperature']}°C，"
            f"湿度 {current['relative_humidity_2m']}%，风速 {current['wind_speed_10m']} km/h，"
            f"降水 {current['precipitation']} mm。"
        )
    if mode == "current":
        start = 0
    if start + days > len(available):
        raise WeatherQueryError("请求的日期范围超出当前可用预报范围。")

    reports = []
    for index in range(start, start + days):
        reports.append(
            f"{place_name}天气预报（{available[index]}）："
            f"{weather_code_description(_daily_value(daily, 'weather_code', index))}；"
            f"最高气温 {_daily_value(daily, 'temperature_2m_max', index)}°C，"
            f"最低气温 {_daily_value(daily, 'temperature_2m_min', index)}°C，"
            f"最高体感 {_daily_value(daily, 'apparent_temperature_max', index)}°C，"
            f"最低体感 {_daily_value(daily, 'apparent_temperature_min', index)}°C，"
            f"最大降水概率 {_daily_value(daily, 'precipitation_probability_max', index)}%，"
            f"预计降水 {_daily_value(daily, 'precipitation_sum', index)} mm，"
            f"最大风速 {_daily_value(daily, 'wind_speed_10m_max', index)} km/h。"
        )
    return "\n".join(reports)
