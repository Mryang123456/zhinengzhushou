"""严格的确定性意图门控：不确定时优先澄清，不执行工具。"""

from backend.app.skills.registry import Skill


def classify(question: str, skills: list[Skill]) -> dict:
    normalized = question.lower().strip()
    scored = [(skill, sum(trigger in normalized for trigger in skill.triggers)) for skill in skills]
    matched = [(skill, score) for skill, score in scored if score]
    if not matched:
        return {"status": "clarify", "reason": "未识别到受支持意图", "skills": [], "confidence": 0.0,
                "question": "为了避免使用错误工具，请说明你的目标：是联网调研、出行规划，还是其他任务？"}
    matched.sort(key=lambda item: item[1], reverse=True)
    if len(matched) > 1 and matched[0][1] == matched[1][1]:
        return {"status": "clarify", "reason": "多个意图置信度相同", "skills": [], "confidence": 0.5,
                "question": f"你的问题同时可能属于“{matched[0][0].name}”和“{matched[1][0].name}”。请说明优先目标。"}
    skill, score = matched[0]
    confidence = min(0.95, 0.7 + 0.15 * score)
    has_required_context = not skill.required_context or any(item in normalized for item in skill.required_context)
    if not has_required_context:
        return {"status": "clarify", "reason": "缺少关键上下文", "skills": [], "confidence": confidence,
                "question": f"为准确执行“{skill.name}”，请补充目的地或地点信息。"}
    return {"status": "allow", "reason": "规则匹配通过", "skills": [skill.name], "confidence": confidence, "question": ""}
