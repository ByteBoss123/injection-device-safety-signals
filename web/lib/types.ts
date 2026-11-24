export interface Claim {
  text: string;
  citations: string[];
}

export interface Answer {
  answer: string;
  insufficient_evidence: boolean;
  claims: Claim[];
  dropped_claims: (Claim & { invalid_citations: string[] })[];
  uncited_inline_ids: string[];
}

export interface TraceEvent {
  t: number;
  kind: "model" | "tool" | "guard_redaction" | "validation" | "approval" | "error";
  [key: string]: unknown;
}

export interface PendingAction {
  tool: string;
  args: { title: string; summary: string; citations: string[] };
  invalid_citations: string[];
  approved: boolean | null;
  reviewer?: string;
}

export interface Run {
  id: string;
  question: string;
  status: "running" | "answered" | "awaiting_approval" | "error";
  answer: Answer | null;
  pending_action: PendingAction | null;
  retrieved: string[];
  trace: TraceEvent[];
  usage: { prompt: number; completion: number };
}
