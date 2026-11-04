"""项目路径与环境配置的单一入口。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
ENV_FILE = PROJECT_ROOT / ".env.local"
MCP_CONFIG_FILE = BACKEND_ROOT / "config" / "mcp.json"
MEMORY_DB_FILE = PROJECT_ROOT / "data" / "memory.db"
MEMORY_SCHEMA_FILE = BACKEND_ROOT / "db" / "init.sql"
SKILLS_DIR = BACKEND_ROOT / "skills"
