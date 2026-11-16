package com.zhixing.service.health.api;

import com.zhixing.service.shared.api.ApiResponse;
import com.zhixing.service.shared.observability.TraceIdFilter;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ServerWebExchange;

import java.util.Map;

@RestController
@RequestMapping("/api/health")
public class HealthController {

    @GetMapping
    ApiResponse<Map<String, String>> health(ServerWebExchange exchange) {
        return ApiResponse.success(
                Map.of("service", "zhixing-service", "status", "UP"),
                TraceIdFilter.from(exchange)
        );
    }
}

