import { useCallback, useEffect, useRef, useState } from "react";
import { checkConnection, type ConnectionStatus } from "../services/agentApi";

const POLL_INTERVAL_MS = 30_000;
const TIMEOUT_MS = 5_000;
const CREDENTIAL_DEBOUNCE_MS = 350;

export function useConnectionStatus(apiUrl: string, apiKey: string, accessToken: string) {
  const [result, setResult] = useState<{
    apiUrl: string;
    apiKey: string;
    accessToken: string;
    status: ConnectionStatus;
    checking: boolean;
  }>();
  const refreshRef = useRef<(() => void) | null>(null);
  const refresh = useCallback(() => refreshRef.current?.(), []);

  useEffect(() => {
    let active: AbortController | null = null;
    const probe = async () => {
      if (active) return;
      const controller = new AbortController();
      active = controller;
      setResult((previous) => ({
        apiUrl, apiKey, accessToken, checking: true,
        status: previous?.apiUrl === apiUrl && previous.apiKey === apiKey && previous.accessToken === accessToken
          ? previous.status
          : "checking",
      }));
      let status: ConnectionStatus = "unreachable";
      try {
        status = await checkConnection(apiUrl, apiKey, accessToken, AbortSignal.any([
          controller.signal,
          AbortSignal.timeout(TIMEOUT_MS),
        ]));
      } catch {
        // 网络失败、响应无法识别或请求超时。
      } finally {
        if (!controller.signal.aborted) {
          active = null;
          setResult({ apiUrl, apiKey, accessToken, status, checking: false });
        }
      }
    };

    refreshRef.current = () => { void probe(); };
    const initialTimer = window.setTimeout(probe, CREDENTIAL_DEBOUNCE_MS);
    const checkWhenVisible = () => {
      if (!document.hidden) void probe();
    };
    const timer = window.setInterval(checkWhenVisible, POLL_INTERVAL_MS);
    window.addEventListener("focus", checkWhenVisible);
    document.addEventListener("visibilitychange", checkWhenVisible);
    return () => {
      refreshRef.current = null;
      active?.abort();
      window.clearTimeout(initialTimer);
      window.clearInterval(timer);
      window.removeEventListener("focus", checkWhenVisible);
      document.removeEventListener("visibilitychange", checkWhenVisible);
    };
  }, [apiUrl, apiKey, accessToken]);

  if (result?.apiUrl !== apiUrl || result.apiKey !== apiKey || result.accessToken !== accessToken) {
    return { status: "checking" as const, checking: true, refresh };
  }
  return { status: result.status, checking: result.checking, refresh };
}
