package com.zhixing.service.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;

@ConfigurationProperties(prefix = "agent")
public record AgentProperties(
        String baseUrl,
        String apiKey,
        Duration connectTimeout,
        Duration readTimeout
) {
    public AgentProperties {
        baseUrl = baseUrl == null || baseUrl.isBlank() ? "http://127.0.0.1:8000" : baseUrl;
        apiKey = apiKey == null ? "" : apiKey;
        connectTimeout = connectTimeout == null ? Duration.ofSeconds(2) : connectTimeout;
        readTimeout = readTimeout == null ? Duration.ofSeconds(5) : readTimeout;
    }
}

