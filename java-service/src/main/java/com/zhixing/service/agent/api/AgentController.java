package com.zhixing.service.agent.api;

import com.zhixing.service.agent.application.AgentStatusService;
import com.zhixing.service.agent.domain.AgentStatus;
import com.zhixing.service.shared.api.ApiResponse;
import com.zhixing.service.shared.observability.TraceIdFilter;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/api/internal/agent")
public class AgentController {

    private final AgentStatusService agentStatusService;

    public AgentController(AgentStatusService agentStatusService) {
        this.agentStatusService = agentStatusService;
    }

    @GetMapping("/status")
    Mono<ApiResponse<AgentStatus>> status(ServerWebExchange exchange) {
        String traceId = TraceIdFilter.from(exchange);
        return agentStatusService.readiness(traceId)
                .map(status -> ApiResponse.success(status, traceId));
    }
}

