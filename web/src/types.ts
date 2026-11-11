export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolEvents?: string[];
  pending?: boolean;
  retryable?: boolean;
  notice?: string;
  approval?: PendingApproval;
  resume?: { runId: string; after: string };
  submission?: { retry: boolean; attemptId?: string };
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

export type ProgressStage = "preparing" | "understanding" | "context" | "drafting" | "checking";

export type StreamEvent = {
  type: "run" | "progress" | "token" | "tool_start" | "tool_end" | "approval_required" | "done" | "error";
  run_id?: string;
  event_id?: string;
  stage?: ProgressStage;
  content?: string;
  tools?: string[];
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
