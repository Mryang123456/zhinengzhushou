export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolEvents?: string[];
  pending?: boolean;
};

export type StreamEvent = {
  type: "token" | "tool_start" | "tool_end" | "done" | "error";
  content?: string;
  tools?: string[];
  message?: string;
};

export type Session = { id: string; title: string; messages: Message[]; };
