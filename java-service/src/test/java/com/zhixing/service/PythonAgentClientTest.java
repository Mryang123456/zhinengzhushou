package com.zhixing.service;

import com.zhixing.service.agent.infrastructure.client.PythonAgentClient;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.ClientRequest;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;

class PythonAgentClientTest {

    @Test
    void readinessMapsPythonResponseAndForwardsTraceId() {
        AtomicReference<ClientRequest> captured = new AtomicReference<>();
        WebClient client = WebClient.builder()
                .baseUrl("http://agent.test")
                .exchangeFunction(request -> {
                    captured.set(request);
                    return Mono.just(ClientResponse.create(HttpStatus.OK)
                            .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                            .body("{\"status\":\"ready\",\"checks\":{}}")
                            .build());
                })
                .build();

        StepVerifier.create(new PythonAgentClient(client).readiness("trace-001"))
                .assertNext(status -> {
                    assertThat(status.available()).isTrue();
                    assertThat(status.httpStatus()).isEqualTo(200);
                    assertThat(status.response().path("status").asText()).isEqualTo("ready");
                })
                .verifyComplete();

        assertThat(captured.get().url().getPath()).isEqualTo("/readyz");
        assertThat(captured.get().headers().getFirst("X-Trace-Id")).isEqualTo("trace-001");
    }
}

