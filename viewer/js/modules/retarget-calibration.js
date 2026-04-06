import * as THREE from "three";
import { canonicalBoneKey, parseTrackName } from "./bone-utils.js";
import { BODY_PARENT, BODY_SCALE_REFERENCE, FINGER_PARENT } from "./retarget-constants.js";

function getBoneBindPosition(bone) {
  const bind = bone?.userData?.__bindPosition;
  if (bind?.isVector3) return bind;
  return bone?.position?.isVector3 ? bone.position : null;
}

function getBindSegmentLength(bone) {
  const bindPos = getBoneBindPosition(bone);
  if (!bindPos) return null;
  const len = bindPos.length();
  return Number.isFinite(len) && len > 1e-6 ? len : null;
}

function median(values) {
  if (!values?.length) return null;
  const sorted = values
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function fingerScaleLimits(canonical) {
  if (canonical.includes("Metacarpal")) return { min: 0.65, max: 2.1 };
  if (canonical.includes("Proximal")) return { min: 0.65, max: 2.3 };
  if (canonical.includes("Intermediate")) return { min: 0.65, max: 2.5 };
  if (canonical.includes("Distal")) return { min: 0.65, max: 2.7 };
  return { min: 0.65, max: 2.4 };
}

function bodyScaleLimits(canonical) {
  if (canonical === "spine" || canonical === "chest") {
    return { min: 0.82, max: 1.22 };
  }
  if (canonical === "upperChest") {
    return { min: 0.35, max: 1.45 };
  }
  if (canonical === "neck" || canonical === "head") {
    return { min: 0.2, max: 1.8 };
  }
  if (canonical === "leftShoulder" || canonical === "rightShoulder") {
    return { min: 0.55, max: 1.35 };
  }
  if (canonical.includes("UpperArm") || canonical.includes("LowerArm")) {
    return { min: 0.55, max: 1.35 };
  }
  if (canonical.includes("UpperLeg") || canonical.includes("LowerLeg")) {
    return { min: 0.8, max: 1.25 };
  }
  if (canonical.includes("Hand")) {
    return { min: 0.55, max: 1.35 };
  }
  if (canonical.includes("Foot") || canonical.includes("Toes")) {
    return { min: 0.78, max: 1.3 };
  }
  return { min: 0.8, max: 1.25 };
}

function estimateGlobalRigScale(sourceMap, targetMap) {
  const ratios = [];
  for (const [key, parentKey] of BODY_SCALE_REFERENCE.entries()) {
    const sourceBone = sourceMap.get(key) || null;
    const targetBone = targetMap.get(key) || null;
    const sourceParent = sourceMap.get(parentKey) || null;
    const targetParent = targetMap.get(parentKey) || null;
    if (!sourceBone || !targetBone || !sourceParent || !targetParent) continue;
    const sourceLen = getBindSegmentLength(sourceBone);
    const targetLen = getBindSegmentLength(targetBone);
    if (!(sourceLen > 1e-6 && targetLen > 1e-6)) continue;
    const ratio = targetLen / sourceLen;
    if (Number.isFinite(ratio) && ratio > 1e-6) {
      ratios.push(ratio);
    }
  }
  const med = median(ratios);
  return Number.isFinite(med) ? THREE.MathUtils.clamp(med, 1e-4, 100) : 1;
}

function collectTrackPresenceByBone(clip) {
  const trackByBone = new Map();
  for (const t of clip?.tracks || []) {
    const parsed = parseTrackName(t.name);
    if (!parsed) continue;
    const row = trackByBone.get(parsed.bone) || { hasQ: false, hasP: false };
    if (parsed.property === "quaternion") row.hasQ = true;
    if (parsed.property === "position") row.hasP = true;
    trackByBone.set(parsed.bone, row);
  }
  return trackByBone;
}

export function buildBodyLengthCalibration(sourceBones, targetBones, clip, buildCanonicalBoneMap) {
  const sourceMap = buildCanonicalBoneMap(sourceBones);
  const targetMap = buildCanonicalBoneMap(targetBones);
  const trackByBone = collectTrackPresenceByBone(clip);
  const globalScale = estimateGlobalRigScale(sourceMap, targetMap);
  const entries = [];
  let clampedCount = 0;
  for (const [key, parentKey] of BODY_PARENT.entries()) {
    const targetBone = targetMap.get(key) || null;
    const sourceBone = sourceMap.get(key) || null;
    const targetParent = targetMap.get(parentKey) || null;
    const sourceParent = sourceMap.get(parentKey) || null;
    if (!targetBone || !sourceBone || !targetParent || !sourceParent) continue;
    const track = trackByBone.get(targetBone.name);
    if (track?.hasP && key !== "hips") continue;
    const sourceLen = getBindSegmentLength(sourceBone);
    const targetLen = getBindSegmentLength(targetBone);
    if (!(sourceLen > 1e-6 && targetLen > 1e-6)) continue;
    const expectedTargetLen = sourceLen * globalScale;
    const rawScale = expectedTargetLen / targetLen;
    if (!Number.isFinite(rawScale)) continue;
    const limits = bodyScaleLimits(key);
    const scale = THREE.MathUtils.clamp(rawScale, limits.min, limits.max);
    if (Math.abs(scale - rawScale) > 1e-6) clampedCount += 1;
    if (Math.abs(scale - 1) < 0.03) continue;
    const bindPos = getBoneBindPosition(targetBone)?.clone?.() || targetBone.position.clone();
    if (bindPos.lengthSq() < 1e-10) continue;
    entries.push({
      bone: targetBone,
      canonical: key,
      scale: Number(scale.toFixed(4)),
      rawScale: Number(rawScale.toFixed(4)),
      sourceLen: Number(sourceLen.toFixed(5)),
      targetLen: Number(targetLen.toFixed(5)),
      expectedTargetLen: Number(expectedTargetLen.toFixed(5)),
      scaledPos: bindPos.multiplyScalar(scale),
    });
  }
  return entries.length
    ? {
        entries,
        globalScale: Number(globalScale.toFixed(5)),
        clampedCount,
      }
    : null;
}

export function buildFingerLengthCalibration(sourceBones, targetBones, clip, buildCanonicalBoneMap) {
  const sourceMap = buildCanonicalBoneMap(sourceBones);
  const targetMap = buildCanonicalBoneMap(targetBones);
  const trackByBone = collectTrackPresenceByBone(clip);
  const globalScale = estimateGlobalRigScale(sourceMap, targetMap);
  const entries = [];
  let clampedCount = 0;
  let minRawScale = Number.POSITIVE_INFINITY;
  let maxRawScale = Number.NEGATIVE_INFINITY;
  for (const [key, parentKey] of FINGER_PARENT.entries()) {
    const targetBone = targetMap.get(key) || null;
    const sourceBone = sourceMap.get(key) || null;
    const targetParent = targetMap.get(parentKey) || null;
    const sourceParent = sourceMap.get(parentKey) || null;
    if (!targetBone || !sourceBone || !targetParent || !sourceParent) continue;
    const track = trackByBone.get(targetBone.name);
    if (track?.hasP) continue;
    const sourceLen = getBindSegmentLength(sourceBone);
    const targetLen = getBindSegmentLength(targetBone);
    if (!(sourceLen > 1e-6 && targetLen > 1e-6)) continue;
    const expectedTargetLen = sourceLen * globalScale;
    const rawScale = expectedTargetLen / targetLen;
    if (!Number.isFinite(rawScale)) continue;
    minRawScale = Math.min(minRawScale, rawScale);
    maxRawScale = Math.max(maxRawScale, rawScale);
    const limits = fingerScaleLimits(key);
    const scale = THREE.MathUtils.clamp(rawScale, limits.min, limits.max);
    if (Math.abs(scale - rawScale) > 1e-6) clampedCount += 1;
    if (!Number.isFinite(scale) || Math.abs(scale - 1) < 0.06) continue;
    const bindPos = getBoneBindPosition(targetBone)?.clone?.() || targetBone.position.clone();
    if (bindPos.lengthSq() < 1e-10) continue;
    entries.push({
      bone: targetBone,
      canonical: key,
      scale: Number(scale.toFixed(4)),
      rawScale: Number(rawScale.toFixed(4)),
      sourceLen: Number(sourceLen.toFixed(5)),
      targetLen: Number(targetLen.toFixed(5)),
      expectedTargetLen: Number(expectedTargetLen.toFixed(5)),
      scaledPos: bindPos.multiplyScalar(scale),
    });
  }
  return entries.length
    ? {
        entries,
        globalScale: Number(globalScale.toFixed(5)),
        clampedCount,
        minRawScale: Number.isFinite(minRawScale) ? Number(minRawScale.toFixed(4)) : null,
        maxRawScale: Number.isFinite(maxRawScale) ? Number(maxRawScale.toFixed(4)) : null,
      }
    : null;
}

export function applyBoneLengthCalibration(plan, modelRoot) {
  if (!plan?.entries?.length) return;
  for (const e of plan.entries) {
    e.bone.position.copy(e.scaledPos);
  }
  modelRoot?.updateMatrixWorld(true);
}

export function resetBoneLengthCalibration(plan, modelRoot) {
  if (!plan?.entries?.length) return;
  for (const e of plan.entries) {
    const bindPos = getBoneBindPosition(e.bone);
    if (bindPos?.isVector3) {
      e.bone.position.copy(bindPos);
    }
  }
  modelRoot?.updateMatrixWorld(true);
}

export function snapshotCanonicalBonePositions(bones, canonicalKeys) {
  const keySet = new Set(canonicalKeys || []);
  const out = [];
  for (const bone of bones || []) {
    const canonical = canonicalBoneKey(bone.name) || "";
    if (!keySet.has(canonical)) continue;
    out.push({
      bone,
      canonical,
      position: bone.position.clone(),
    });
  }
  return out;
}

export function restoreBonePositionSnapshot(snapshot, modelRoot) {
  if (!snapshot?.length) return;
  for (const row of snapshot) {
    row.bone.position.copy(row.position);
  }
  modelRoot?.updateMatrixWorld(true);
}

export function applyFingerLengthCalibration(plan, modelRoot) {
  applyBoneLengthCalibration(plan, modelRoot);
}

export function buildArmRefinementCalibration({
  sourceBones,
  targetBones,
  namesTargetToSource,
  sourceClip,
  buildCanonicalBoneMap,
  collectAlignmentDiagnostics,
  updateWorld,
}) {
  const targetByCanonical = buildCanonicalBoneMap(targetBones);
  const multipliers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3];
  const sideDefs = [
    {
      side: "left",
      keys: ["leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand"],
    },
    {
      side: "right",
      keys: ["rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand"],
    },
  ];
  const planEntries = [];
  const sideReports = [];

  for (const def of sideDefs) {
    const chain = def.keys
      .map((k) => ({ canonical: k, bone: targetByCanonical.get(k) || null }))
      .filter((x) => !!x.bone);
    if (!chain.length) continue;
    const baseline = chain.map((x) => x.bone.position.clone());
    const measureErr = () => {
      const report = collectAlignmentDiagnostics({
        targetBones: chain.map((x) => x.bone),
        sourceBones,
        namesTargetToSource,
        sourceClip,
        maxRows: 8,
        overlayYawOverride: 0,
      });
      return Number.isFinite(report?.avgPosErrNorm) ? report.avgPosErrNorm : report?.avgPosErr;
    };
    const baselineErr = measureErr();
    let bestErr = Number.isFinite(baselineErr) ? baselineErr : Number.POSITIVE_INFINITY;
    let bestMul = 1;

    for (const m of multipliers) {
      for (let i = 0; i < chain.length; i += 1) {
        chain[i].bone.position.copy(baseline[i]).multiplyScalar(m);
      }
      updateWorld?.();
      const err = measureErr();
      if (Number.isFinite(err) && err < bestErr) {
        bestErr = err;
        bestMul = m;
      }
    }

    const applied =
      Number.isFinite(bestErr) &&
      Number.isFinite(baselineErr) &&
      bestErr <= baselineErr - 0.005 &&
      Math.abs(bestMul - 1) > 1e-3;

    if (applied) {
      for (let i = 0; i < chain.length; i += 1) {
        const bone = chain[i].bone;
        const scaled = baseline[i].clone().multiplyScalar(bestMul);
        bone.position.copy(scaled);
        planEntries.push({
          bone,
          canonical: chain[i].canonical,
          scaledPos: scaled.clone(),
        });
      }
    } else {
      for (let i = 0; i < chain.length; i += 1) {
        chain[i].bone.position.copy(baseline[i]);
      }
    }
    sideReports.push({
      side: def.side,
      baselineErr: Number.isFinite(baselineErr) ? Number(baselineErr.toFixed(5)) : null,
      bestErr: Number.isFinite(bestErr) ? Number(bestErr.toFixed(5)) : null,
      multiplier: Number(bestMul.toFixed(2)),
      applied,
    });
  }
  updateWorld?.();
  return {
    entries: planEntries,
    sides: sideReports,
  };
}
