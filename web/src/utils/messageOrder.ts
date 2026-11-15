import type { Message } from "../types";

/** 用共同消息作锚点，合并一个有序快照，同时保留另一份记录独有的消息。 */
export function mergeMessageOrder(previous: Message[], ordered: Message[], tailStart = previous.length): Message[] {
  const orderedIds = new Set(ordered.map((message) => message.id));
  const beforeAnchor = new Map<string, Message[]>();
  let unmatched: Message[] = [];
  let shared = false;
  for (const message of previous.slice(0, tailStart)) {
    if (orderedIds.has(message.id)) {
      shared = true;
      beforeAnchor.set(message.id, unmatched);
      unmatched = [];
    } else {
      unmatched.push(message);
    }
  }
  const tail = previous.slice(tailStart);
  if (!shared) return [...unmatched, ...ordered, ...tail];
  return [...ordered.flatMap((message) => [...(beforeAnchor.get(message.id) ?? []), message]), ...unmatched, ...tail];
}
