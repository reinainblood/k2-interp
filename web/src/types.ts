export type Message = {
  role: string;
  content: string;
  reasoning_content?: string;
  think_fast?: string;
  think_faster?: string;
};
export type Layer = {
  expert_ids?: number[];
  expert_weights?: number[];
  router_scores?: number[];
  router_logits?: number[];
  selection_scores?: number[];
  router_entropy?: number;
  routing_margin?: number;
  norms?: Record<string, number>;
  rms?: Record<string, number>;
  expert_norms?: Record<string, number>;
  forward_expert_counts?: number[];
  forward_expert_mass?: number[];
};
export type Frame = {
  run_id: string;
  status: string;
  sequence: number;
  position?: number;
  step?: number;
  phase?: string;
  layers?: Record<string, Layer>;
  token_text?: string[];
  input_ids?: number[];
  input_tokens?: number;
  output_ids?: number[];
  output_logprobs?: number[];
  output_text?: string;
  tensor_bytes?: number;
  current_token_ids?: number[];
  assistant_message?: Message;
  forward_complete?: boolean;
  positions?: number[];
  updated_at?: string;
  committed_at?: string;
  last_layer?: number;
  projection?: string;
  identity?: Record<string, unknown>;
  transport_errors?: number;
  error?: { type: string; message: string };
  recovered_from_artifacts?: boolean;
  result?: {
    assistant_message: Message;
    result: { finish_reason: string; output_ids: number[] };
  };
};
export type Turn = {
  run_id: string;
  user: string;
  assistant?: Message;
  status: string;
  created_at: string;
  finish_reason?: string;
  error?: { type: string; message: string };
};
export type Session = {
  id: string;
  checkpoint: string;
  system: string;
  turns: Turn[];
  captures: {
    capture_id: string;
    tag: string;
    position: number;
    layer: number;
    created_at: string;
    run_id: string;
  }[];
  created_at: string;
};
export type SessionItem = {
  id: string;
  checkpoint: string;
  title: string;
  turns: number;
  created_at: string;
};
export const terminal = (status: string) =>
  ["complete", "failed", "cancelled"].includes(status);
export const number = (n: number | undefined, digits = 3) =>
  n == null
    ? "\u00b7"
    : n.toLocaleString("en-US", { maximumFractionDigits: digits });
export const tokenLabel = (text: string) =>
  text.replaceAll("Ġ", " ").replaceAll("Ċ", "\n").replaceAll("▁", " ");
