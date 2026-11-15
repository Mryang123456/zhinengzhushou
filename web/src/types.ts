export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolEvents?: ActivityStep[];
  planSteps?: PlanStep[];
  elapsedMs?: number;
  activity?: StreamEvent[];
  completedAt?: number;
  completedRunId?: string;
  sources?: AnswerSource[];
  pending?: boolean;
  retryable?: boolean;
  notice?: string;
  approval?: PendingApproval;
  resume?: { runId: string; after: string };
  submission?: { retry: boolean; attemptId?: string };
};

export type AnswerSource = { url: string; title: string; domain: string };
export type ActivityPage = AnswerSource & { excerpt?: string };
export type PlanStep = {
  id: number;
  title: string;
  status: "pending" | "running" | "reviewing" | "supplementing" | "complete" | "partial" | "skipped" | "blocked";
  reason?: string;
};
export type ActivityStep = string | {
  label: string;
  pages?: ActivityPage[];
  queries?: string[];
  kind?: "highlights" | "queries";
};

export type PendingApproval = {
  approvalId: string;
  confirmationToken: string;
  runId?: string;
  tool: string;
  risk: string;
  arguments: Record<string, unknown>;
  expiresAt: number;
  status: "pending" | "executing" | "completed" | "failed";
};

export type ProgressStage = "preparing" | "understanding" | "context" | "drafting" | "checking" | "searching_web" | "verifying_recommendations" | "revising";

export type StreamEvent = {
  type: "run" | "progress" | "plan_progress" | "token" | "tool_start" | "tool_end" | "evidence_retry" | "evidence_review" | "quality" | "web_pages_read" | "web_queries" | "web_highlights" | "timing" | "approval_required" | "done" | "error";
  run_id?: string;
  event_id?: string;
  stage?: ProgressStage;
  steps?: PlanStep[];
  content?: string;
  tools?: string[];
  web_result_count?: number;
  elapsed_ms?: number;
  operation?: string;
  action?: "answer" | "search" | "clarify" | "partial";
  stop_reason?: "no_new_evidence" | "budget_exhausted";
  passed?: boolean;
  partial?: boolean;
  pages?: ActivityPage[];
  queries?: string[];
  sources?: AnswerSource[];
  message?: string;
  retryable?: boolean;
  approval_id?: string;
  confirmation_token?: string;
  tool?: string;
  risk?: string;
  arguments?: Record<string, unknown>;
  expires_at?: number;
};

export type Session = { id: string; title: string; messages: Message[]; updatedAt: number; archived?: boolean; };
