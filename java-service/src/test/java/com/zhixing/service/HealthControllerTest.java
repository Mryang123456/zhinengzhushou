package com.zhixing.service;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.test.web.reactive.server.WebTestClient;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class HealthControllerTest {

    @LocalServerPort
    private int port;

    @Test
    void healthEndpointReturnsTraceableResponse() {
        WebTestClient.bindToServer()
                .baseUrl("http://127.0.0.1:" + port)
                .build()
                .get()
                .uri("/api/health")
                .header("X-Trace-Id", "test-trace-id")
                .exchange()
                .expectStatus().isOk()
                .expectHeader().valueEquals("X-Trace-Id", "test-trace-id")
                .expectBody()
                .jsonPath("$.success").isEqualTo(true)
                .jsonPath("$.data.service").isEqualTo("zhixing-service")
                .jsonPath("$.data.status").isEqualTo("UP")
                .jsonPath("$.traceId").isEqualTo("test-trace-id");
    }
}

