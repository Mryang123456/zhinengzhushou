package com.zhixing.service.shared.observability;

import org.slf4j.MDC;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.web.server.WebFilter;
import org.springframework.web.server.WebFilterChain;
import reactor.core.publisher.Mono;

import java.util.UUID;

@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class TraceIdFilter implements WebFilter {

    public static final String TRACE_ID_HEADER = "X-Trace-Id";
    public static final String TRACE_ID_ATTRIBUTE = "traceId";

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {
        String incoming = exchange.getRequest().getHeaders().getFirst(TRACE_ID_HEADER);
        String traceId = StringUtils.hasText(incoming) ? incoming : UUID.randomUUID().toString();
        exchange.getAttributes().put(TRACE_ID_ATTRIBUTE, traceId);
        exchange.getResponse().getHeaders().set(TRACE_ID_HEADER, traceId);

        return chain.filter(exchange)
                .contextWrite(context -> context.put(TRACE_ID_ATTRIBUTE, traceId))
                .doOnEach(signal -> {
                    if (signal.getContextView().hasKey(TRACE_ID_ATTRIBUTE)) {
                        MDC.put(TRACE_ID_ATTRIBUTE, signal.getContextView().get(TRACE_ID_ATTRIBUTE));
                    }
                })
                .doFinally(ignored -> MDC.remove(TRACE_ID_ATTRIBUTE));
    }

    public static String from(ServerWebExchange exchange) {
        return exchange.getAttributeOrDefault(TRACE_ID_ATTRIBUTE, "unknown");
    }
}

