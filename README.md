# 基础 LangChain + LangGraph 智能体

这是一个可运行的命令行智能体：LangChain 提供 GPT 模型和工具封装，LangGraph 将“模型判断 → 工具执行 → 模型总结”编排成有状态图。

## 能力边界

- 使用 GPT 回答问题；
- 可在任意回合按需连续调用：受限的 `calculator`、`get_weather`（城市当前天气）和 `web_search`（联网检索）；
- 默认还会作为 MCP 客户端连接 `mcp.json` 中配置的服务：本地 `weather_mcp_server.py` 与 DDGS 搜索服务；其工具会以 `weather_`、`search_` 前缀加入同一个 LangGraph `ToolNode`；
- 联网工具只读取公开信息，不具备文件、Shell 或网络写入权限；
- 使用同一个 `--thread-id` 时保留对话检查点；配置 PostgreSQL 后可跨进程、跨重启恢复；
- 提供按 `user_id` 隔离的 SQLite 长期记忆；只有用户明确要求“记住”时才保存，支持检索与删除；
- 不执行文件、Shell、网络写入或其他高风险工具。

## 运行

在本目录执行：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python agent.py --once "计算 (17 + 3) * 2"
```

连续对话：

```powershell
python agent.py --thread-id demo-01
```

输入 `/quit` 退出。

默认开启命令行流式输出：模型文本会逐段显示，工具调用会显示开始与完成状态。若要恢复一次性输出，追加 `--no-stream`。

## 记忆模块

短期记忆（任务执行状态）与长期记忆（用户明确保存的事实）彼此独立。未配置数据库时，任务状态由 `MemorySaver` 在当前进程内保存，长期记忆保存在 `data/memory.db`；配置 PostgreSQL 后，两者都可在重启后恢复。模型可调用以下工具：

- `remember_memory`：仅在你明确要求“记住”时保存非敏感信息；
- `recall_memory`：检索当前用户的长期记忆；
- `forget_memory`：仅在你明确要求删除时删除指定编号的记忆。

不同用户使用不同的 `--user-id`，例如：

```powershell
python agent.py --user-id 张三
```

长期记忆默认不会保存 API Key、密码等敏感值；可用 `--no-memory` 临时关闭。

### PostgreSQL + pgvector

生产环境推荐 PostgreSQL + pgvector。`docker-compose.yml` 和 `db/init.sql` 已准备好数据库、用户隔离索引与 `embedding vector(1536)` 列。

在 `.env.local` 配置非示例密码和数据库地址后启动：

```text
POSTGRES_PASSWORD=请设置强密码
MEMORY_DATABASE_URL=postgresql://agent:与上方相同的强密码@localhost:5432/agent_memory
CHECKPOINT_DATABASE_URL=postgresql://agent:与上方相同的强密码@localhost:5432/agent_memory
```

```powershell
docker compose up -d
python agent.py --user-id 张三
```

当 `MEMORY_DATABASE_URL` 存在时，Agent 自动优先使用 PostgreSQL；当 `CHECKPOINT_DATABASE_URL` 存在时，LangGraph 自动创建自己的 checkpoint 表并持久化每一步状态。未配置时分别回退为 SQLite 长期记忆与进程内检查点，便于本地开发。检查点键由 `user_id:thread_id` 构成，避免不同用户复用同名线程时串话。数据库容器尚未在当前电脑启动，因为未检测到 Docker。

## SSE 流式接口

启动本地服务：

```powershell
uvicorn server:app --port 8000
```

随后向 `POST http://127.0.0.1:8000/chat/stream` 发送 JSON：

```json
{"question":"查询新加坡天气", "thread_id":"用户-001", "user_id":"张三", "use_mcp":true, "use_memory":true}
```

响应为 `text/event-stream`，事件类型包括 `token`、`tool_start`、`tool_end`、`done` 和 `error`。前端可用原生 `fetch` 读取流，或使用 SSE 兼容客户端逐条渲染。

## MCP 配置

`mcp.json` 是唯一允许启动 MCP 服务的显式配置。默认提供两个 stdio 服务：天气和 DDGS 搜索。智能体启动时发现 MCP 工具，之后模型可以和本地工具一样按需多次调用它们。

增加一个 MCP 服务时，手动编辑该文件，例如：

```json
{
  "mcpServers": {
    "my-service": {
      "command": "your-mcp-command",
      "args": ["serve"],
      "transport": "stdio"
    }
  }
}
```

只添加你信任的服务：stdio MCP 服务以本机进程运行。临时禁用所有 MCP 服务可使用 `python agent.py --no-mcp`。

## 服务端基础防护

启动服务：

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

健康检查地址为 `GET /healthz`，不会调用模型或工具。服务对每个客户端默认限制为每分钟 20 次请求；可通过 `AGENT_RATE_LIMIT` 和 `AGENT_RATE_WINDOW_SECONDS` 调整。

生产环境请在 `.env.local` 设置 `AGENT_API_KEY` 为随机长密钥。设置后，请求 `/chat/stream` 必须携带 `X-Agent-Api-Key` 请求头；未设置时仅适合本地开发。服务端日志保留具体异常，但 SSE 客户端不会收到可能含敏感细节的原始错误信息。

## 结构

```text
START → agent → tools? → agent → END
```

未配置 `CHECKPOINT_DATABASE_URL` 时由 `MemorySaver` 负责进程内 checkpoint；配置 PostgreSQL 地址后，自动使用数据库型 checkpointer 并跨进程恢复。请使用稳定的 `user_id` 与 `thread_id`。

默认模型由 `.env.local` 的 `MODEL_PROVIDER=openai` 与 `MODEL_NAME=gpt-4.1-mini` 决定。

## 模型工厂与切换

[models.py](models.py) 是唯一的大模型初始化入口。切换模型只需修改 `.env.local`：

```text
# OpenAI
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4.1-mini

# Ollama（本地模型）
# MODEL_PROVIDER=ollama
# MODEL_NAME=qwen3:8b
# MODEL_BASE_URL=http://localhost:11434

# Anthropic
# MODEL_PROVIDER=anthropic
# MODEL_NAME=claude-sonnet-4-5

# Google
# MODEL_PROVIDER=google
# MODEL_NAME=gemini-2.5-flash
```

OpenAI 的基础适配器已安装。若要改用 Ollama、Anthropic 或 Google，只需安装对应可选依赖：

```powershell
python -m pip install -r requirements-model-providers.txt
```

模型调用接口保持一致，因此工具、MCP、记忆、规划和 SSE 均不需要修改。

天气使用 [Open-Meteo](https://open-meteo.com/en/docs)，网页搜索使用 [DDGS](https://pypi.org/project/ddgs/)。二者均需要本机能访问互联网。

## Web 聊天前端

中文 React + TypeScript 聊天前端位于 `web/`，支持 SSE 逐字输出、工具调用状态、用户/会话标识、停止生成、访问密钥临时输入与移动端布局。

```powershell
cd web
npm install
npm run dev
```

默认连接 `http://127.0.0.1:8000`。需要先在另一个终端运行后端：

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

如果后端设置了 `AGENT_API_KEY`，在前端“连接设置”中填写；该值只保留在当前页面内存中。开发环境的跨域白名单默认为 Vite 的两个本机地址，生产部署请用 `AGENT_CORS_ORIGINS` 配置明确的前端域名。

