# 智行助手 Java Service

该目录是智行助手的独立 Spring Boot 业务服务基础框架。第一阶段只验证 Java 服务自身和 Java 到 Python Agent 的连通性，不修改现有 React → FastAPI 调用链。

## 技术栈

- Java 17
- Spring Boot 3.5.16
- Spring WebFlux
- Spring Boot Actuator
- Maven

## 当前能力

- `GET /api/health`：Java 服务健康检查，返回统一响应和 Trace ID。
- `GET /actuator/health`：Spring Boot Actuator 健康检查。
- `GET /api/internal/agent/status`：通过 WebClient 探测现有 Python `/readyz`。
- 全局统一异常响应。
- `X-Trace-Id` 透传；未提供时自动生成。
- Python Agent 地址、访问密钥和超时均可通过环境变量配置。

## 模块结构

项目采用“按业务模块组织、模块内部分层”的结构：

```text
com.zhixing.service
├── agent/
│   ├── api/                         # HTTP 接口
│   ├── application/                 # 用例编排
│   ├── domain/                      # 领域模型
│   ├── exception/                   # Agent 业务异常
│   └── infrastructure/client/       # Python Agent 适配器
├── health/api/                      # 健康检查接口
├── config/                          # Spring 与外部服务配置
└── shared/
    ├── api/                         # 通用响应协议
    ├── exception/                   # 全局异常处理
    └── observability/               # Trace 与可观测性
```

依赖方向保持为 `api → application → domain`，外部 Python 调用由 `infrastructure` 实现。Controller 不直接拼装 WebClient 请求，便于后续替换协议、增加鉴权或编写隔离测试。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `JAVA_SERVICE_PORT` | `8080` | Java 服务端口 |
| `AGENT_BASE_URL` | `http://127.0.0.1:8000` | Python Agent 地址 |
| `AGENT_API_KEY` | 空 | Python Agent 可选访问密钥 |
| `AGENT_CONNECT_TIMEOUT` | `2s` | 连接超时 |
| `AGENT_READ_TIMEOUT` | `5s` | 读取超时 |

## 启动

确保使用 Java 17 或更高版本和 Maven 3.6.3 或更高版本：

```powershell
java -version
mvn -version
cd E:\mutilagent\multi-agent-claude-code\langgraph-agent\java-service
mvn -s maven-settings.xml spring-boot:run
```

验证 Java 服务：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/health
```

Python 后端已在 8000 端口运行时，验证跨服务连接：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/internal/agent/status
```

运行测试：

```powershell
mvn -s maven-settings.xml test
```

当前电脑还可以直接使用已安装的 JDK 21 与 Maven：

```powershell
$env:JAVA_HOME = "F:\java\jdk21"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
& "F:\java\apache-maven-3.8.6-bin\apache-maven-3.8.6\bin\mvn.cmd" -s maven-settings.xml test
```

## 边界

当前 Java 服务不接管 JWT、用户、租户、会话、消息、数据库或 SSE。后续有明确业务需求后，再按模块逐步从 Python Service 迁移，Python 的 LangGraph、RAG、MCP 和模型能力保持不变。

