export function createSessionTitle(now = new Date(), suffix = crypto.randomUUID().slice(0, 6)) {
  return `会话-${now.toISOString().replace(/[:.]/g, "-")}-${suffix}`;
}
