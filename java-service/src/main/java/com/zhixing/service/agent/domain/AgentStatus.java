package com.zhixing.service.agent.domain;

import com.fasterxml.jackson.databind.JsonNode;

public record AgentStatus(
        boolean available,
        int httpStatus,
        JsonNode response
) {
}

