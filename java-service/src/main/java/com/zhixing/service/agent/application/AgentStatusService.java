package com.zhixing.service.agent.application;

import com.zhixing.service.agent.domain.AgentStatus;
import com.zhixing.service.agent.infrastructure.client.PythonAgentClient;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

@Service
public class AgentStatusService {

    private final PythonAgentClient pythonAgentClient;

    public AgentStatusService(PythonAgentClient pythonAgentClient) {
        this.pythonAgentClient = pythonAgentClient;
    }

    public Mono<AgentStatus> readiness(String traceId) {
        return pythonAgentClient.readiness(traceId);
    }
}

