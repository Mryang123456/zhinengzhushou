# 智能 Agent 整体架构与运行流程

![智能 Agent 整体架构与运行流程](agent-architecture.png)

## 五层职责

| 层级 | 核心职责 | 主要实现 |
|---|---|---|
| UI 层 | 对话交互、会话管理、流式展示、账号隔离与断流恢复 | React、TypeScript、Ant Design X、IndexedDB、SSE |
| Agent 编排层 | 身份与幂等、意图识别、条件路由、ReAct 循环、上下文治理、质量门与断点恢复 | FastAPI、LangGraph、Checkpoint、Chat Run Worker |
| 工具层 | 候选工具召回、动态白名单、参数校验、预算、超时、审批和隔离执行 | 内置工具、Skill、MCP、语义工具检索 |
| 知识层 | 文档切片、混合检索、邻居扩展、证据组重排、长期记忆和 Embedding 治理 | PostgreSQL、pgvector、Cross-Encoder、Embedding Gateway |
| 集成层 | 统一连接模型、数据库、缓存、MCP 和外部公开服务 | OpenAI、Anthropic、Google、Ollama、PostgreSQL、Redis、MCP |

## 可维护架构源图

```mermaid
flowchart TB
    subgraph UI["1. UI 层"]
        React["React / TypeScript"]
        Ant["Ant Design X"]
        IDB["IndexedDB"]
        SSE["SSE 流式交互与断流恢复"]
    end

    subgraph Agent["2. Agent 编排层"]
        API["FastAPI 接入、JWT、请求幂等"]
        Intent["意图识别与 Conditional Edge"]
        Graph["LangGraph 状态机与 ReAct 循环"]
        Context["上下文预算与滚动摘要"]
        Quality["回答质量门"]
        Checkpoint["Chat Run + Checkpoint 断点恢复"]
    end

    subgraph Tools["3. 工具层"]
        Retrieval["语义工具检索 Top-K"]
        Allowlist["动态工具白名单"]
        Governance["参数校验、预算、超时"]
        MCP["MCP 审批与隔离"]
        Builtins["天气 / 搜索 / 计算 / 记忆"]
    end

    subgraph Knowledge["4. 知识层"]
        Chunk["结构化切片"]
        Hybrid["向量 + 全文混合召回"]
        Rerank["邻居扩展 + Cross-Encoder 两阶段重排"]
        Memory["长期记忆"]
        Embedding["Embedding Gateway"]
    end

    subgraph Integration["5. 集成层"]
        Models["OpenAI / Anthropic / Google / Ollama"]
        PG["PostgreSQL + pgvector"]
        Redis["Redis"]
        MCPServices["MCP 服务"]
        External["联网搜索 / 天气服务"]
    end

    User(["用户提问"]) --> React
    React -->|"JWT + HTTP/SSE"| API
    API --> Intent
    Intent -->|"普通问答"| Context
    Intent -->|"知识查询"| Hybrid
    Intent -->|"工具任务"| Retrieval
    Retrieval --> Allowlist --> Governance --> Graph
    Hybrid --> Rerank --> Graph
    Memory --> Context --> Graph
    Graph -->|"需要工具"| Governance
    Graph -->|"形成草稿"| Quality
    Quality -->|"通过"| Checkpoint
    Quality -->|"修正或安全降级"| Graph
    Checkpoint --> SSE --> React

    Embedding --> Models
    Hybrid --> PG
    Memory --> PG
    Checkpoint --> PG
    Governance --> MCP
    MCP --> MCPServices
    Builtins --> External
    Allowlist --> Redis
    Embedding --> Redis
```

## 端到端运行流程

```text
用户提问
→ JWT 身份认证与 API Key 校验
→ message_id 请求幂等与会话级锁
→ LangGraph 意图识别
→ RAG、长期记忆或工具候选路由
→ 主模型执行 ReAct 推理循环
→ 引用、事实和工具状态质量校验
→ Chat Run、事件和 Checkpoint 原子持久化
→ SSE 按事件游标返回前端
→ 断流或 Worker 故障后从持久状态恢复
```

## 横向治理

- 安全隔离：JWT 身份、知识 ACL、工具白名单、MCP 审批、Prompt Injection 防护。
- 可靠性：请求幂等、会话锁、断点恢复、超时、重试、限流、熔断和降级。
- 可观测性：Request ID、结构化事件、Prometheus 指标和 RAG 评估报告。
- 多实例一致性：PostgreSQL 持久状态、Redis 分布式限流与审批、Embedding SingleFlight。
- 工程质量：后端单元测试、RAG 回归门禁、前端测试和 GitHub Actions CI。
