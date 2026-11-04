"""轻量、可审计的本地 Skill 注册表。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    triggers: tuple[str, ...]
    tool_patterns: tuple[str, ...]
    required_context: tuple[str, ...]
    instructions: str


def load_skills(skills_dir: Path) -> list[Skill]:
    """从每个子目录的 SKILL.md 加载白名单内的本地技能。"""
    skills: list[Skill] = []
    for path in sorted(skills_dir.glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text.startswith("---"):
            continue
        _, frontmatter, instructions = text.split("---", 2)
        fields = dict(
            line.split(":", 1) for line in frontmatter.splitlines() if ":" in line
        )
        name = fields.get("name", "").strip()
        description = fields.get("description", "").strip()
        triggers = tuple(item.strip().lower() for item in fields.get("triggers", "").split(",") if item.strip())
        tool_patterns = tuple(item.strip() for item in fields.get("tools", "").split(",") if item.strip())
        required_context = tuple(item.strip().lower() for item in fields.get("required_context", "").split(",") if item.strip())
        if name and description and triggers and tool_patterns and instructions.strip():
            skills.append(Skill(name, description, triggers, tool_patterns, required_context, instructions.strip()))
    return skills


def match_skills(question: str, skills: list[Skill], limit: int = 2) -> list[Skill]:
    """按显式触发词进行确定性路由，避免模型任意加载本地指令。"""
    normalized = question.lower()
    return [skill for skill in skills if any(word in normalized for word in skill.triggers)][:limit]


def format_skill_context(skills: list[Skill]) -> str:
    if not skills:
        return "（本回合未匹配到专用 Skill，遵循通用助手规则。）"
    return "\n\n".join(
        f"已启用 Skill：{skill.name}\n{skill.instructions}" for skill in skills
    )


def select_tool_names(skills: list[Skill], available_names: list[str]) -> list[str]:
    """将 Skill 工具模式展开为实际白名单，支持 search_* 这类前缀模式。"""
    patterns = {pattern for skill in skills for pattern in skill.tool_patterns}
    selected = []
    for name in available_names:
        if name in {"calculator", "list_skills", "remember_memory", "recall_memory", "forget_memory"}:
            selected.append(name)
        elif any(name == pattern or (pattern.endswith("*") and name.startswith(pattern[:-1])) for pattern in patterns):
            selected.append(name)
    return selected
