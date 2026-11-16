package com.zhixing.service;

import com.zhixing.service.config.AgentProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties(AgentProperties.class)
public class ZhixingServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(ZhixingServiceApplication.class, args);
    }
}

