import { useCallback, useEffect, useRef, useState } from "react";
import { cancelAgentRun, streamAgentResponse, type AgentRequest } from "../services/agentApi";
import type { StreamEvent } from "../types";
export function useAgentStream(apiUrl: string) {
  const [runningIds, setRunningIds] = useState<ReadonlySet<string>>(() => new Set());
  const controllersRef = useRef(new Map<string, {
    controller: AbortController; request: AgentRequest; runId?: string;
  }>());
  const isRunning = useCallback((threadId: string) => controllersRef.current.has(threadId), []);
  const stop = useCallback(async (threadId: string) => {
    const active = controllersRef.current.get(threadId);
    if (!active) return;
    if (active.runId) await cancelAgentRun(apiUrl, active.runId, active.request);
    else active.controller.abort();
  }, [apiUrl]);
  useEffect(() => {
    const controllers = controllersRef.current;
    return () => {
      for (const { controller } of controllers.values()) controller.abort();
      controllers.clear();
    };
  }, []);

  const send = useCallback(async (request: AgentRequest, onEvent: (event: StreamEvent) => void) => {
    const { threadId } = request;
    const controllers = controllersRef.current;
    if (controllers.has(threadId)) return;
    const controller = new AbortController();
    const active = { controller, request, runId: request.resume?.runId };
    controllers.set(threadId, active);
    setRunningIds((ids) => new Set(ids).add(threadId));
    try {
      await streamAgentResponse(apiUrl, request, controller.signal, (event) => {
        if (event.type === "run" && event.run_id) active.runId = event.run_id;
        if (!controller.signal.aborted) onEvent(event);
      });
    } finally {
      if (controllers.get(threadId) === active) {
        controllers.delete(threadId);
        setRunningIds((ids) => {
          const next = new Set(ids);
          next.delete(threadId);
          return next;
        });
      }
    }
  }, [apiUrl]);
  return { runningIds, isRunning, send, stop };
}
