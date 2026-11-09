export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolEvents?: string[];
  pending?: boolean;
  approval?: PendingApproval;
};

export type PendingApproval = {
  approvalId: string;
  confirmationToken: string;
  tool: string;
  risk: string;
  arguments: Record<string, unknown>;
  expiresAt: number;
  status: "pending" | "executing" | "completed" | "failed";
};

export type StreamEvent = {
  type: "token" | "tool_start" | "tool_end" | "approval_required" | "done" | "error";
  content?: string;
  tools?: string[];
  message?: string;
  approval_id?: string;
  confirmation_token?: string;
  tool?: string;
  risk?: string;
  arguments?: Record<string, unknown>;
  expires_at?: number;
};

export type Session = { id: string; title: string; messages: Message[]; updatedAt: number; archived?: boolean; };
