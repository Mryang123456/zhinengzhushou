package com.zhixing.service.config;

import io.netty.channel.ChannelOption;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

@Configuration
public class WebClientConfig {

    @Bean
    WebClient agentWebClient(AgentProperties properties, WebClient.Builder builder) {
        HttpClient httpClient = HttpClient.create()
                .option(
                        ChannelOption.CONNECT_TIMEOUT_MILLIS,
                        Math.toIntExact(properties.connectTimeout().toMillis())
                )
                .responseTimeout(properties.readTimeout());

        WebClient.Builder configured = builder
                .baseUrl(properties.baseUrl())
                .clientConnector(new ReactorClientHttpConnector(httpClient));

        if (!properties.apiKey().isBlank()) {
            configured.defaultHeader("X-API-Key", properties.apiKey());
        }
        return configured.defaultHeader(HttpHeaders.ACCEPT, "application/json").build();
    }
}

