package com.zhixing.service.agent.infrastructure.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.NullNode;
import com.zhixing.service.agent.domain.AgentStatus;
import com.zhixing.service.agent.exception.AgentUnavailableException;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Component
public class PythonAgentClient {

    private final WebClient agentWebClient;

    public PythonAgentClient(WebClient agentWebClient) {
        this.agentWebClient = agentWebClient;
    }

    public Mono<AgentStatus> readiness(String traceId) {
        return agentWebClient.get()
                .uri("/readyz")
                .header("X-Trace-Id", traceId)
                .exchangeToMono(response -> {
                    int status = response.statusCode().value();
                    return response.bodyToMono(JsonNode.class)
                            .defaultIfEmpty(NullNode.getInstance())
                            .map(body -> new AgentStatus(response.statusCode().is2xxSuccessful(), status, body));
                })
                .onErrorMap(error -> new AgentUnavailableException(
                        "无法连接 Python Agent 服务，请确认服务已在配置地址启动。",
                        error
                ));
    }
}

