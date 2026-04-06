import { RETARGET_BODY_CORE_CANONICAL, RETARGET_BODY_CANONICAL } from "./retarget-constants.js";

function normalizeCanonicalKeys(keys, allowedCanonicals) {
  if (!Array.isArray(keys)) return [];
  return keys
    .map((key) => String(key || "").trim())
    .filter((key) => key && allowedCanonicals.has(key));
}

export function resolveRetargetStageCanonicalFilter(stage, { bodyCanonicalKeys = null, bodyCanonicalMode = "" } = {}) {
  const normalizedStage = String(stage || "").trim().toLowerCase();
  if (normalizedStage !== "body") return null;

  const profileKeys = normalizeCanonicalKeys(bodyCanonicalKeys, RETARGET_BODY_CANONICAL);
  if (profileKeys.length) {
    return new Set(profileKeys);
  }
  if (String(bodyCanonicalMode || "").trim().toLowerCase() === "core") {
    return RETARGET_BODY_CORE_CANONICAL;
  }
  return RETARGET_BODY_CANONICAL;
}

export function resolveBodyMetricCanonicalFilter(stage, profile = {}) {
  const normalizedStage = String(stage || "").trim().toLowerCase();
  if (normalizedStage !== "body") {
    return RETARGET_BODY_CANONICAL;
  }

  const stageFilter = resolveRetargetStageCanonicalFilter(normalizedStage, profile);
  const narrowedCore = Array.from(stageFilter || []).filter((canonical) =>
    RETARGET_BODY_CORE_CANONICAL.has(canonical)
  );
  return narrowedCore.length ? new Set(narrowedCore) : RETARGET_BODY_CORE_CANONICAL;
}
