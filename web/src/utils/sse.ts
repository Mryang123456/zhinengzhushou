import type { StreamEvent } from "../types";
export function consumeSseBlocks(text: string, onEvent: (event: StreamEvent) => void) { for (const block of text.split("\n\n")) { const data = block.split("\n").find((line) => line.startsWith("data: ")); if (!data) continue; try { onEvent(JSON.parse(data.slice(6)) as StreamEvent); } catch { /* 丢弃不完整事件 */ } } }
