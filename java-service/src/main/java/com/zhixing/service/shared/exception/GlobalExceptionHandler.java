package com.zhixing.service.shared.exception;

import com.zhixing.service.agent.exception.AgentUnavailableException;
import com.zhixing.service.shared.api.ApiResponse;
import com.zhixing.service.shared.observability.TraceIdFilter;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.server.ServerWebExchange;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(AgentUnavailableException.class)
    ResponseEntity<ApiResponse<Void>> handleAgentUnavailable(
            AgentUnavailableException exception,
            ServerWebExchange exchange
    ) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(ApiResponse.failure(
                        "AGENT_UNAVAILABLE",
                        exception.getMessage(),
                        TraceIdFilter.from(exchange)
                ));
    }

    @ExceptionHandler(Exception.class)
    ResponseEntity<ApiResponse<Void>> handleUnexpected(
            Exception exception,
            ServerWebExchange exchange
    ) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(ApiResponse.failure(
                        "INTERNAL_ERROR",
                        "服务处理请求时发生异常。",
                        TraceIdFilter.from(exchange)
                ));
    }
}

