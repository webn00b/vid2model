export function shouldAllowSourceRootYawFlipCandidates(sourceClipYawSummary = null) {
  return !Boolean(sourceClipYawSummary?.looksCentered);
}

export function buildViewerRootYawCandidates(
  rawFacingYaw,
  quantizeFacingYaw,
  { sourceClipYawSummary = null } = {}
) {
  const set = new Set();
  const list = [];
  const push = (value) => {
    if (!Number.isFinite(value)) return;
    let angle = Math.atan2(Math.sin(value), Math.cos(value));
    if (Math.abs(angle) < 1e-6) angle = 0;
    const key = Number(angle.toFixed(6));
    if (set.has(key)) return;
    set.add(key);
    list.push(angle);
  };

  const allowSourceFlipCandidates = shouldAllowSourceRootYawFlipCandidates(sourceClipYawSummary);
  push(0);
  if (allowSourceFlipCandidates) {
    push(Math.PI);
    push(-Math.PI);
  }
  push(quantizeFacingYaw(-rawFacingYaw));
  if (allowSourceFlipCandidates) {
    push(quantizeFacingYaw(Math.PI - rawFacingYaw));
  }
  push(-rawFacingYaw);
  return list;
}
