"""校验模型生成的天气建议，预报事实由后端单独展示。"""

import re

from pydantic import BaseModel, ConfigDict, ValidationError

from backend.app.content_safety import contains_secret, is_direct_prompt_override
from backend.app.models.tool_protocol import contains_textual_tool_protocol


class WeatherAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activity: str
    clothing: str


def validated_advice(raw: str, *, activity: bool, clothing: bool) -> WeatherAdvice | None:
    try:
        advice = WeatherAdvice.model_validate_json(raw)
    except ValidationError:
        return None
    for text, requested in ((advice.activity, activity), (advice.clothing, clothing)):
        if bool(text.strip()) != requested:
            return None
        if (
            len(text) > 320
            or re.search(r"\d|https?://|knowledge://|[<>]", text, re.I)
            or contains_secret(text)
            or is_direct_prompt_override(text)
            or contains_textual_tool_protocol(text)
        ):
            return None
    return advice
