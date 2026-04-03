import { canonicalBoneKey, normalizeBoneName } from "./bone-utils.js";

export function canonicalBonePreferenceScore(bone) {
  const rawName = String(bone?.name || "");
  const rawLower = rawName.toLowerCase();
  const normalized = normalizeBoneName(rawName);
  const canonical = canonicalBoneKey(rawName);
  if (!canonical) return Number.NEGATIVE_INFINITY;

  let score = 0;
  const normalizedCanonical = normalizeBoneName(canonical);
  if (normalized === normalizedCanonical) score += 120;
  if (normalized.endsWith(normalizedCanonical)) score += 35;
  if (normalized.startsWith(normalizedCanonical)) score += 20;

  if (bone?.children?.length) {
    score += Math.min(bone.children.length, 4) * 4;
  }

  if (/(^|[^a-z0-9])(ik|fk)([^a-z0-9]|$)/.test(rawLower)) score -= 80;
  if (/(twist|roll|helper|socket|offset|corrective|driver|pole|target|dummy|ctrl|control)/.test(rawLower)) score -= 60;
  if (/(end|nub|tip)([^a-z0-9]|$)/.test(rawLower)) score -= 45;

  return score;
}

export function buildCanonicalBoneMap(bones) {
  const map = new Map();
  for (const bone of bones || []) {
    const key = canonicalBoneKey(bone?.name);
    if (!key) continue;
    const existing = map.get(key);
    if (!existing || canonicalBonePreferenceScore(bone) > canonicalBonePreferenceScore(existing)) {
      map.set(key, bone);
    }
  }
  return map;
}
