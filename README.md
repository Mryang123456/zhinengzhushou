# 基础 LangChain + LangGraph 智能体

这是一个可运行的命令行智能体：LangChain 提供 GPT 模型和工具封装，LangGraph 将“模型判断 → 工具执行 → 模型总结”编排成有状态图。

## 能力边界

- 使用 GPT 回答问题；
- 可在任意回合按需连续调用：受限的 `calculator`、`get_weather`（城市当前天气）和 `web_search`（联网检索）；
- 默认还会作为 MCP 客户端连接 `backend/config/mcp.json` 中配置的服务：本地天气 MCP 与 DDGS 搜索服务；其工具会以 `weather_`、`search_` 前缀加入同一个 LangGraph `ToolNode`；
- 联网工具只读取公开信息，不具备文件、Shell 或网络写入权限；
- 使用同一个 `--thread-id` 时保留对话检查点；配置 PostgreSQL 后可跨进程、跨重启恢复；
- 提供按 `user_id` 隔离的 SQLite 长期记忆；只有用户明确要求“记住”时才保存，支持检索与删除；
- 不执行文件、Shell、网络写入或其他高风险工具。

## 运行

在本目录执行：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
python -m backend.main --once "计算 (17 + 3) * 2"
```

连续对话：

```powershell
python -m backend.main --thread-id demo-01
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
python -m backend.main --user-id 张三
```

长期记忆默认不会保存 API Key、密码等敏感值；可用 `--no-memory` 临时关闭。

记忆会拦截常见密钥、密码、令牌、银行卡号和手机号模式；默认保留 365 天，可用 `MEMORY_TTL_DAYS` 调整。SQLite 与 PostgreSQL 都会写入并过滤 `expires_at`，到期记忆不会再进入 Agent 上下文或检索结果。

### 长对话上下文治理

Checkpoint 会保留完整消息用于断点恢复与审计，但不会再把全文无限传给模型。服务端按上下文 token 预算保留最近的完整对话回合；超过预算的较早历史会合并为结构化摘要（偏好、已确认结论、待办和约束），再与最近回合、相关长期记忆一起发送。摘要调用超时或失败时会使用确定性压缩记录，仍保持上下文上限。

```text
# 默认模型上下文窗口为 128K token，预留 8K token 给回答与工具调用。
# 仅在当前模型至少支持 128K 上下文时使用该配置。
CONTEXT_WINDOW_TOKENS=131072
CONTEXT_RESPONSE_RESERVE_TOKENS=8192
CONTEXT_SUMMARY_TIMEOUT_SECONDS=15
```

不同模型应按实际窗口调整 `CONTEXT_WINDOW_TOKENS`；摘要仅是背景，当前用户问题和本回合工具证据始终优先。

### PostgreSQL + pgvector

生产环境推荐 PostgreSQL + pgvector。`docker-compose.yml` 和 `backend/db/init.sql` 已准备好数据库、用户隔离索引与 `embedding vector(1536)` 列。

在 `.env.local` 配置非示例密码和数据库地址后启动：

```text
POSTGRES_PASSWORD=请设置强密码
MEMORY_DATABASE_URL=postgresql://agent:与上方相同的强密码@localhost:5432/agent_memory
CHECKPOINT_DATABASE_URL=postgresql://agent:与上方相同的强密码@localhost:5432/agent_memory
```

```powershell
docker compose up -d
python -m backend.main --user-id 张三
```

当 `MEMORY_DATABASE_URL` 存在时，Agent 自动优先使用 PostgreSQL；当 `CHECKPOINT_DATABASE_URL` 存在时，LangGraph 自动创建自己的 checkpoint 表并持久化每一步状态。未配置时分别回退为 SQLite 长期记忆与进程内检查点，便于本地开发。检查点键由 `user_id:thread_id` 构成，避免不同用户复用同名线程时串话。数据库容器尚未在当前电脑启动，因为未检测到 Docker。

## SSE 流式接口

启动本地服务：

```powershell
python -m uvicorn backend.app.api.server:app --port 8000
```

随后向 `POST http://127.0.0.1:8000/chat/stream` 发送 JSON：

```json
{"question":"查询新加坡天气", "thread_id":"会话-001", "use_mcp":true, "use_memory":true}
```

响应为 `text/event-stream`，事件类型包括 `token`、`tool_start`、`tool_end`、`done` 和 `error`。前端可用原生 `fetch` 读取流，或使用 SSE 兼容客户端逐条渲染。

## MCP 配置

`backend/config/mcp.json` 是唯一允许启动 MCP 服务的显式配置。默认提供两个 stdio 服务：天气和 DDGS 搜索。智能体启动时发现 MCP 工具，之后模型可以和本地工具一样按需多次调用它们。

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

只添加你信任的服务：stdio MCP 服务以本机进程运行。临时禁用所有 MCP 服务可使用 `python -m backend.main --no-mcp`。

### MCP 治理

`backend/config/mcp.json` 除服务配置外还包含治理信息：允许服务、工具风险级别、工具参数策略、超时和重试。L0/L1 工具可以自动执行；未登记、私有读取或写入类工具默认 L2。服务端会校验 MCP 服务是否出现在 `allowedServers` 中，并只向模型暴露 `toolPolicies` 已登记的工具。

每个已登记工具必须明确允许参数名、类型和边界；例如天气地点最多 80 个字符，搜索关键词最多 500 个字符、`max_results` 只能在 1 到 8。策略外工具、额外参数、错误类型及超出边界的参数都会在执行前审计并拒绝，审批确认接口也会重新校验同一策略。

每个 MCP 服务还必须在 `serverPolicies` 中登记其命令、参数和传输方式，实际配置不一致即拒绝加载。工具层默认单批最多 3 次、单回合最多 6 次调用，并分别有单工具 20 秒、单请求 90 秒超时；客户端断开连接时服务会停止流式任务。可通过环境变量调整这些预算。

L2-L4 工具会创建一个 5 分钟有效、一次性的审批记录，并通过 SSE 发送工具名、风险等级和脱敏参数预览。前端点击“确认并执行”后，会携带一次性令牌请求 `POST /mcp/approvals/{approval_id}/confirm`；令牌在服务端以哈希保存，并由 SQLite 原子消费，避免重复执行。审批及执行结果会写入 `data/mcp_audit.jsonl`。当前示例的天气与搜索均为 L1，因此只有你新增 L2 及以上 MCP 工具时才会出现审批卡片。

审批确认后，原始工具输出不会直接返回界面，而是会作为不可信证据重新进入受控续答与质量校验。生产多实例部署应启用 Redis 作为审批与限流的共享状态；stdio MCP 服务也应运行在容器或沙箱中。

## Skill 技能系统

项目内 Skill 位于 `backend/skills/<技能名称>/SKILL.md`。每个文件都包含名称、用途、触发词、允许工具和工作准则。启动时后端只加载这个受控目录中的 Skill；每个用户问题会按触发词匹配最多两个技能，并将其准则注入当前 Agent 回合。

内置示例包括：

- `联网调研`：要求先检索、交叉核对并附上来源链接；
- `出行规划`：要求补足地点、时间与偏好，必要时先查询天气，并说明地图数据限制。

模型也可调用 `list_skills` 查看可用技能。新增 Skill 时请复制现有 `SKILL.md` 格式；不要允许用户输入路径、URL 或聊天内容直接作为 Skill 加载来源。

Skill 匹配后，模型只会看到该 Skill 声明的工具及基础低风险工具；工具节点会再次验证调用名称。比如命中 `联网调研` 后允许 `web_search` 与 `search_*`，但不允许天气工具；命中 `出行规划` 后才允许天气与搜索工具。

## 严格意图门控

每个请求先经过 `backend/app/intent/policy.py`：它会结合显式触发词、同义表达、轻微拼写容错和基础任务规则进行评分；未识别、近似竞争意图或出行缺少地点时，工作流直接返回澄清问题，不调用模型和工具。明确连接词下最多合并两个高置信技能。仅当规则判定通过，才进入 Agent 并生成本回合工具白名单。意图决策会作为 SSE `intent` 事件输出，便于前端和日志系统审计。

回答质量门会校验时效性工具证据、网页链接/知识库引用的可追溯性，以及关键事实是否紧邻其对应证据；无法逐条溯源时会安全降级而不是输出确定性结论。

## 可观测性与 CI

每个 HTTP 请求会生成或继承 `X-Request-Id`，并写入结构化事件日志；`GET /metrics` 提供 Prometheus 文本格式的基础事件计数。仓库的 `.github/workflows/ci.yml` 会在推送和 PR 时运行后端单元测试、Python 编译检查与前端生产构建。

## 知识库与 pgvector

知识库使用 PostgreSQL + pgvector，表结构位于 `backend/db/knowledge.sql`，Docker 初始化时也会创建对应表和向量索引。配置 `.env.local`：

```text
KNOWLEDGE_DATABASE_URL=postgresql://agent:你的密码@localhost:5432/agent_memory
# 可选：只有设置后才调用付费 Embedding；未设置时使用关键词检索回退。
KNOWLEDGE_EMBEDDING_PROVIDER=openai
KNOWLEDGE_EMBEDDING_MODEL=text-embedding-3-small
# 仅用于向量候选的原始余弦相似度门槛，范围 [-1, 1]。
KNOWLEDGE_MIN_SCORE=0.22
```

基础版支持导入 UTF-8 的 `.txt`、`.md`、`.json` 文件：

```powershell
python -m backend.app.knowledge.cli .\资料\产品说明.md --owner-id 开发者A --scope private
```

导入时必须指定所属用户或租户；`private` 仅所属用户可检索，`shared` 才允许其他用户检索。导入会切分文本、记录来源并按配置生成向量。Agent 仅在配置知识库地址后暴露 `search_knowledge_base` 工具，且该工具归属于“联网调研”Skill 的白名单。检索将向量和全文候选融合、重排，保留达到余弦门槛或全文命中的候选，再截取 Top-K。结果携带 `knowledge://文档ID#片段` 引用，文档内容按不可信资料处理。

### RAG 检索评测

CI 使用合成候选回归集，覆盖语义改写、阈值过滤、全文召回与无证据场景。报告包含候选池召回率、过滤前重排 Recall@K、最终 Recall@K、Precision@K、MRR@K、NDCG@K 和无证据样本正确返回空结果的比例。Precision@K 固定以 K 为分母，检索指标只对有标注证据的样本取平均。

```powershell
python -m backend.evals.rag backend/evals/rag_regression.jsonl --mode fixture -k 1 --min-recall 1 --min-no-evidence-accuracy 1
```

真实评测使用已导入测试语料的知识库和标注 JSONL。每行包括 `id`、`question`、`user_id` 和 `relevant_chunks`。引用 ID 应与该测试库快照一致，无证据样本的 `relevant_chunks` 为空数组。连接可使用只读账号，评测过程不初始化表。启用向量检索时，评测会按现有配置请求 Embedding。

```powershell
python -m backend.evals.rag .\rag_golden.jsonl --mode live -k 5 --require-hybrid
```

按业务基线设置 `--min-recall` 与 `--min-no-evidence-accuracy`，低于门槛时命令退出码为 1。合成回归结果用于验证排序与过滤逻辑，真实检索质量以业务评测集的报告为准。

## 服务端基础防护

启动服务：

```powershell
python -m uvicorn backend.app.api.server:app --host 127.0.0.1 --port 8000
```

健康检查地址为 `GET /healthz`，不会调用模型或工具。服务对每个客户端默认限制为每分钟 20 次请求；可通过 `AGENT_RATE_LIMIT` 和 `AGENT_RATE_WINDOW_SECONDS` 调整。登录接口另按“来源 IP + 登录名哈希”限制为每 15 分钟 5 次，可通过 `LOGIN_RATE_LIMIT` 与 `LOGIN_RATE_WINDOW_SECONDS` 调整。SSE 每 15 秒会发送注释心跳，减少反向代理因空闲而切断长连接的情况。

生产环境请在 `.env.local` 设置 `AGENT_API_KEY` 为随机长密钥。设置后，请求 `/chat/stream` 必须携带 `X-Agent-Api-Key` 请求头；未设置时仅适合本地开发。服务端日志保留具体异常，但 SSE 客户端不会收到可能含敏感细节的原始错误信息。

### JWT 身份认证

本地开发默认 `AUTH_REQUIRED=false`，服务端会使用固定的 `local-development` 身份，且不再接受客户端传递的 `user_id`。生产环境必须在 `.env.local` 配置：

```text
AUTH_REQUIRED=true
JWT_SECRET=至少32字符的随机密钥
AUTH_BOOTSTRAP_USER=admin
AUTH_BOOTSTRAP_PASSWORD_HASH=pbkdf2_sha256$...
```

可在本机生成密码哈希（请自行输入密码，不要将明文提交到仓库）：

```powershell
python -c "from backend.app.security import password_hash; print(password_hash(input('密码: ')))"
```

前端“连接与登录”中登录后会将短时 JWT 仅存于页面内存，并在聊天与 MCP 审批请求的 `Authorization: Bearer` 请求头中发送。服务端从 JWT 的 `sub` 取得用户身份，因此记忆、知识库 ACL、会话检查点和审批不再信任前端提供的用户标识。生产建议将初始账号登录替换为企业 OIDC/SSO。

设置 `ENVIRONMENT=production` 后，服务会在启动时 fail-closed：强制要求 JWT、初始认证账号、明确的 HTTPS 跨域白名单、Redis、PostgreSQL 记忆与 Checkpoint，以及 `MULTI_INSTANCE_MODE=true`。任何缺失都会拒绝启动，避免生产环境静默退回到本地文件或免认证模式。

### 多实例安全状态

多实例部署时必须启用以下配置，服务启动会校验并拒绝降级到进程内或本地文件状态：

```text
MULTI_INSTANCE_MODE=true
REDIS_PASSWORD=强密码
REDIS_URL=redis://:强密码@localhost:6379/0
MEMORY_DATABASE_URL=postgresql://agent:密码@localhost:5432/agent_memory
CHECKPOINT_DATABASE_URL=postgresql://agent:密码@localhost:5432/agent_memory
```

Redis 负责跨实例的请求限流和 MCP 一次性审批令牌；审批令牌通过 Lua 脚本原子消费，避免请求被负载均衡到不同实例后重复执行。PostgreSQL 负责长期记忆、知识库和 LangGraph checkpoint。`docker compose up -d` 会同时启动 PostgreSQL 与 Redis；数据库和 Redis 端口只绑定本机、Redis 强制密码并带健康检查。公网部署仍应通过私有网络、TLS 终止层、备份和受管密钥服务访问，不应直接暴露这两个端口。

### MCP 运行隔离

当前示例 MCP 使用受注册表约束的本地 stdio 服务，适用于开发。生产环境应把每个 MCP 服务放入最小权限容器，设置只读文件系统、非 root 用户、CPU/内存/PID 限制和网络出口白名单；服务配置中的 `isolation` 必须改为 `container`，再设置 `MCP_SANDBOX_REQUIRED=true`。该开关会拒绝启动任何仍声明为 `local` 的 MCP 服务，防止生产环境误用未隔离进程。

## 结构

```text
START → agent → tools? → agent → END
```

未配置 `CHECKPOINT_DATABASE_URL` 时由 `MemorySaver` 负责进程内 checkpoint；配置 PostgreSQL 地址后，自动使用数据库型 checkpointer 并跨进程恢复。请使用稳定的 `user_id` 与 `thread_id`。

Web 请求按用户和会话串行写入 Checkpoint。等待上一轮完成的默认上限为 120 秒，可用 `AGENT_THREAD_QUEUE_TIMEOUT_SECONDS` 调整。使用 PostgreSQL 时，服务会在同一连接上持有会话级咨询锁并写入 Checkpoint。数据库地址应使用直连或会话池模式。

默认模型由 `.env.local` 的 `MODEL_PROVIDER=openai` 与 `MODEL_NAME=gpt-4.1-mini` 决定。

## 模型工厂与切换

[backend/app/models/factory.py](backend/app/models/factory.py) 是唯一的大模型初始化入口。切换模型只需修改 `.env.local`：

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
python -m pip install -r backend/requirements-model-providers.txt
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
python -m uvicorn backend.app.api.server:app --host 127.0.0.1 --port 8000
```

如果后端设置了 `AGENT_API_KEY`，在前端“连接设置”中填写；该值只保留在当前页面内存中。开发环境的跨域白名单默认为 Vite 的两个本机地址，生产部署请用 `AGENT_CORS_ORIGINS` 配置明确的前端域名。

