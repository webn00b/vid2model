import * as THREE from "three";
import { BODY_PARENT, RETARGET_BODY_CANONICAL } from "./retarget-constants.js";

export const CANONICAL_MOTION_BODY_KEYS = Object.freeze([
  "hips",
  "spine",
  "chest",
  "upperChest",
  "neck",
  "head",
  "leftShoulder",
  "rightShoulder",
  "leftUpperArm",
  "rightUpperArm",
  "leftLowerArm",
  "rightLowerArm",
  "leftHand",
  "rightHand",
  "leftUpperLeg",
  "rightUpperLeg",
  "leftLowerLeg",
  "rightLowerLeg",
  "leftFoot",
  "rightFoot",
  "leftToes",
  "rightToes",
]);

export const CANONICAL_SOLVER_ORDER = Object.freeze([
  "hips",
  "spine",
  "chest",
  "upperChest",
  "neck",
  "head",
  "leftShoulder",
  "rightShoulder",
  "leftUpperArm",
  "rightUpperArm",
  "leftLowerArm",
  "rightLowerArm",
  "leftHand",
  "rightHand",
  "leftUpperLeg",
  "rightUpperLeg",
  "leftLowerLeg",
  "rightLowerLeg",
  "leftFoot",
  "rightFoot",
  "leftToes",
  "rightToes",
]);

const _segmentChildByCanonical = new Map([
  ["hips", "spine"],
  ["spine", "chest"],
  ["chest", "upperChest"],
  ["upperChest", "neck"],
  ["neck", "head"],
  ["leftShoulder", "leftUpperArm"],
  ["rightShoulder", "rightUpperArm"],
  ["leftUpperArm", "leftLowerArm"],
  ["rightUpperArm", "rightLowerArm"],
  ["leftLowerArm", "leftHand"],
  ["rightLowerArm", "rightHand"],
  ["leftUpperLeg", "leftLowerLeg"],
  ["rightUpperLeg", "rightLowerLeg"],
  ["leftLowerLeg", "leftFoot"],
  ["rightLowerLeg", "rightFoot"],
  ["leftFoot", "leftToes"],
  ["rightFoot", "rightToes"],
]);

const _tmpV1 = new THREE.Vector3();
const _tmpV2 = new THREE.Vector3();

export function toRoundedVec3Array(vec) {
  if (!vec || !Number.isFinite(vec.x) || !Number.isFinite(vec.y) || !Number.isFinite(vec.z)) return null;
  return [
    Number(vec.x.toFixed(6)),
    Number(vec.y.toFixed(6)),
    Number(vec.z.toFixed(6)),
  ];
}

export function toRoundedQuatArray(quat) {
  if (!quat || !Number.isFinite(quat.x) || !Number.isFinite(quat.y) || !Number.isFinite(quat.z) || !Number.isFinite(quat.w)) {
    return null;
  }
  return [
    Number(quat.x.toFixed(6)),
    Number(quat.y.toFixed(6)),
    Number(quat.z.toFixed(6)),
    Number(quat.w.toFixed(6)),
  ];
}

export function arrayToVector3(values, out = new THREE.Vector3()) {
  if (!Array.isArray(values) || values.length !== 3 || !values.every(Number.isFinite)) return null;
  return out.set(values[0], values[1], values[2]);
}

export function arrayToQuaternion(values, out = new THREE.Quaternion()) {
  if (!Array.isArray(values) || values.length !== 4 || !values.every(Number.isFinite)) return null;
  return out.set(values[0], values[1], values[2], values[3]).normalize();
}

export function canonicalMotionKeysForStage(stage, canonicalFilter = null) {
  const normalizedStage = String(stage || "body").trim().toLowerCase();
  const keys = normalizedStage === "body" ? CANONICAL_MOTION_BODY_KEYS : CANONICAL_MOTION_BODY_KEYS;
  if (!canonicalFilter) return [...keys];
  return keys.filter((key) => canonicalFilter.has(key));
}

export function getCanonicalSegmentChild(canonical) {
  return _segmentChildByCanonical.get(canonical) || null;
}

export function collectClipSampleTimes(clip) {
  const firstTrack = clip?.tracks?.find((track) => Array.isArray(track?.times) || track?.times?.length > 0) || null;
  const times = firstTrack?.times ? Array.from(firstTrack.times) : [];
  if (!times.length) {
    const duration = Number(clip?.duration || 0);
    if (!(duration > 0)) return [0];
    const fallback = [];
    const dt = 1 / 30;
    for (let t = 0; t < duration; t += dt) {
      fallback.push(Number(t.toFixed(6)));
    }
    fallback.push(Number(duration.toFixed(6)));
    return fallback;
  }
  if (times[0] !== 0) {
    times.unshift(0);
  }
  return times.map((time) => Number(Number(time || 0).toFixed(6)));
}

export function selectEvaluationTimes(times, maxSamples = 12) {
  if (!Array.isArray(times) || !times.length) return [0];
  if (times.length <= maxSamples) return [...times];
  const out = [];
  const lastIndex = times.length - 1;
  for (let i = 0; i < maxSamples; i += 1) {
    const index = Math.round((i / Math.max(1, maxSamples - 1)) * lastIndex);
    const value = times[index];
    if (out[out.length - 1] !== value) out.push(value);
  }
  if (out[out.length - 1] !== times[lastIndex]) out.push(times[lastIndex]);
  return out;
}

export function buildNamesByCanonical(canonicalMap) {
  const out = {};
  for (const key of CANONICAL_MOTION_BODY_KEYS) {
    const bone = canonicalMap.get(key);
    if (bone?.name) out[key] = bone.name;
  }
  return out;
}

export function buildNamesTargetToSourceByCanonical(targetBones, sourceCanonicalNames, canonicalBoneKey) {
  const out = {};
  for (const bone of targetBones || []) {
    const canonical = canonicalBoneKey(bone?.name) || "";
    const sourceName = sourceCanonicalNames?.[canonical] || "";
    if (canonical && sourceName) {
      out[bone.name] = sourceName;
    }
  }
  return out;
}

export function computeSegmentLength(canonicalMap, canonical) {
  const bone = canonicalMap.get(canonical) || null;
  if (!bone?.isBone) return 0;
  const childCanonical = getCanonicalSegmentChild(canonical);
  const child = childCanonical ? canonicalMap.get(childCanonical) || null : null;
  if (!child?.isBone) return 0;
  bone.getWorldPosition(_tmpV1);
  child.getWorldPosition(_tmpV2);
  return Number(_tmpV1.distanceTo(_tmpV2).toFixed(6));
}

export function buildSegmentLengths(canonicalMap, keys = CANONICAL_MOTION_BODY_KEYS) {
  const out = {};
  for (const key of keys) {
    if (!RETARGET_BODY_CANONICAL.has(key)) continue;
    out[key] = computeSegmentLength(canonicalMap, key);
  }
  return out;
}

export function buildRestPoseChains(keys = CANONICAL_MOTION_BODY_KEYS) {
  const chains = {
    torso: ["hips", "spine", "chest", "upperChest", "neck", "head"],
    leftArm: ["leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand"],
    rightArm: ["rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand"],
    leftLeg: ["hips", "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes"],
    rightLeg: ["hips", "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes"],
  };
  const allowed = new Set(keys);
  return Object.fromEntries(
    Object.entries(chains).map(([name, chain]) => [name, chain.filter((key) => allowed.has(key))])
  );
}

export function getCanonicalParent(canonical) {
  return BODY_PARENT.get(canonical) || null;
}
