import { useCallback, useEffect, useRef, useState } from "react";
import { streamAgentResponse, type AgentRequest } from "../services/agentApi";
import type { StreamEvent } from "../types";
export function useAgentStream(apiUrl: string) {
  const [runningIds, setRunningIds] = useState<ReadonlySet<string>>(() => new Set());
  const controllersRef = useRef(new Map<string, AbortController>());
  const isRunning = useCallback((threadId: string) => controllersRef.current.has(threadId), []);
  const stop = useCallback((threadId: string) => controllersRef.current.get(threadId)?.abort(), []);
  useEffect(() => {
    const controllers = controllersRef.current;
    return () => {
      for (const controller of controllers.values()) controller.abort();
      controllers.clear();
    };
  }, []);

  const send = useCallback(async (request: AgentRequest, onEvent: (event: StreamEvent) => void) => {
    const { threadId } = request;
    const controllers = controllersRef.current;
    if (controllers.has(threadId)) return;
    const controller = new AbortController();
    controllers.set(threadId, controller);
    setRunningIds((ids) => new Set(ids).add(threadId));
    try {
      await streamAgentResponse(apiUrl, request, controller.signal, (event) => {
        if (!controller.signal.aborted) onEvent(event);
      });
    } finally {
      if (controllers.get(threadId) === controller) {
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
