import * as THREE from "three";
import { canonicalBoneKey } from "./bone-utils.js";
import {
  applyPairRestOrientationCorrection,
  applyPairInvertRotationOverride,
  buildProfiledChains,
  computeRetargetPosScale,
  createRetargetPair,
  initializeRetargetPairsRestState,
  initializePairRestWorldDelta,
  restoreCachedPairCalibration,
} from "./retarget-plan-utils.js";

const PARENT_RELATIVE_REST_DELTA_CANONICAL = new Set([
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
  "leftLowerLeg",
  "rightLowerLeg",
]);

const WORLD_REST_TRANSFER_CANONICAL = new Set([
  "leftUpperLeg",
  "rightUpperLeg",
  "leftFoot",
  "rightFoot",
  "leftToes",
  "rightToes",
]);
const NO_CACHE_REST_CALIBRATION_CANONICAL = new Set([
  "leftUpperLeg",
  "rightUpperLeg",
  "leftLowerLeg",
  "rightLowerLeg",
  "leftFoot",
  "rightFoot",
  "leftToes",
  "rightToes",
]);
const _worldTransferQ1 = new THREE.Quaternion();
const _worldTransferQ2 = new THREE.Quaternion();
const _worldTransferQ3 = new THREE.Quaternion();
const _legPlaneV1 = new THREE.Vector3();
const _legPlaneV2 = new THREE.Vector3();
const _legPlaneV3 = new THREE.Vector3();
const _legPlaneV4 = new THREE.Vector3();
const _legPlaneV5 = new THREE.Vector3();
const _legPlaneV6 = new THREE.Vector3();
const _legPlaneV7 = new THREE.Vector3();
const _legPlaneV8 = new THREE.Vector3();
const _legPlaneV9 = new THREE.Vector3();
const _legPlaneQ1 = new THREE.Quaternion();
const _legPlaneQ2 = new THREE.Quaternion();
const _legPlaneQ3 = new THREE.Quaternion();
function pushFootDebug(entry) {
  if (typeof window === "undefined") return;
  const rows = Array.isArray(window.__vid2modelFootCorrectionDebug)
    ? window.__vid2modelFootCorrectionDebug
    : [];
  rows.push(entry);
  window.__vid2modelFootCorrectionDebug = rows;
}

function pairProfileKey(targetName, sourceName) {
  return `${targetName || ""}=>${sourceName || ""}`;
}

function serializeQuaternion(q) {
  if (!q?.isQuaternion) return null;
  return [
    Number(q.x.toFixed(8)),
    Number(q.y.toFixed(8)),
    Number(q.z.toFixed(8)),
    Number(q.w.toFixed(8)),
  ];
}

function deserializeQuaternion(data, outQ) {
  if (!Array.isArray(data) || data.length !== 4 || !outQ) return false;
  if (!data.every((v) => Number.isFinite(v))) return false;
  outQ.set(data[0], data[1], data[2], data[3]).normalize();
  return true;
}

function getBendNormal(startPos, midPos, endPos, outNormal) {
  outNormal.copy(midPos).sub(startPos);
  _legPlaneV8.copy(endPos).sub(midPos);
  if (outNormal.lengthSq() < 1e-10 || _legPlaneV8.lengthSq() < 1e-10) return false;
  outNormal.normalize();
  _legPlaneV8.normalize();
  outNormal.cross(_legPlaneV8);
  if (outNormal.lengthSq() < 1e-10) return false;
  outNormal.normalize();
  return true;
}

function setBoneWorldQuaternion(bone, worldQ, parentWorldQ, localQ) {
  if (!bone || !worldQ) return false;
  if (bone.parent?.isBone) {
    bone.parent.getWorldQuaternion(parentWorldQ);
    bone.quaternion.copy(parentWorldQ.invert().multiply(worldQ)).normalize();
  } else {
    bone.quaternion.copy(worldQ).normalize();
  }
  return true;
}

function getPrimaryChildWorldDirection(bone, outDir) {
  if (!bone || !outDir) return false;
  bone.getWorldPosition(_legPlaneV1);
  let bestChild = null;
  let bestLenSq = 0;
  for (const child of bone.children || []) {
    if (!child?.isBone) continue;
    child.getWorldPosition(_legPlaneV2);
    const lenSq = _legPlaneV2.distanceToSquared(_legPlaneV1);
    if (lenSq > bestLenSq) {
      bestLenSq = lenSq;
      bestChild = child;
    }
  }
  if (!bestChild || bestLenSq < 1e-10) return false;
  bestChild.getWorldPosition(_legPlaneV2);
  outDir.copy(_legPlaneV2).sub(_legPlaneV1);
  if (outDir.lengthSq() < 1e-10) return false;
  outDir.normalize();
  return true;
}

function getWorldDirectionFromLocal(bone, localDir, outDir) {
  if (!bone || !localDir || !outDir) return false;
  if (localDir.lengthSq() < 1e-10) return false;
  bone.getWorldQuaternion(_legPlaneQ1);
  outDir.copy(localDir).applyQuaternion(_legPlaneQ1);
  if (outDir.lengthSq() < 1e-10) return false;
  outDir.normalize();
  return true;
}

function getFootForwardWorldDirection(footBone, toesBone, outDir, localDirHint = null) {
  if (!footBone || !outDir) return false;
  if (toesBone?.isBone) {
    footBone.getWorldPosition(_legPlaneV1);
    toesBone.getWorldPosition(_legPlaneV2);
    outDir.copy(_legPlaneV2).sub(_legPlaneV1);
    if (outDir.lengthSq() >= 1e-10) {
      outDir.normalize();
      return true;
    }
  }
  if (getWorldDirectionFromLocal(footBone, localDirHint, outDir)) {
    return true;
  }
  return getPrimaryChildWorldDirection(footBone, outDir);
}

function computeFootDot(chain) {
  const footTarget = chain?.foot?.target || null;
  const footSource = chain?.foot?.source || null;
  const toesTarget = chain?.toes?.target || null;
  const toesSource = chain?.toes?.source || null;
  const targetLocalDir = chain?.foot?.targetPrimaryChildDirLocal || null;
  const sourceLocalDir = chain?.foot?.sourcePrimaryChildDirLocal || null;
  if (!footTarget || !footSource) return null;
  const targetDir = _legPlaneV7;
  const sourceDir = _legPlaneV8;
  if (!getFootForwardWorldDirection(footTarget, toesTarget, targetDir, targetLocalDir)) return null;
  if (!getFootForwardWorldDirection(footSource, toesSource, sourceDir, sourceLocalDir)) return null;
  return Number(targetDir.dot(sourceDir).toFixed(4));
}

function applyUpperLegDirectionCorrection(chain) {
  const upperTarget = chain?.upper?.target || null;
  const lowerTarget = chain?.lower?.target || null;
  const upperSource = chain?.upper?.source || null;
  const lowerSource = chain?.lower?.source || null;
  if (!upperTarget || !lowerTarget || !upperSource || !lowerSource) return false;

  upperTarget.getWorldPosition(_legPlaneV1);
  lowerTarget.getWorldPosition(_legPlaneV2);
  upperSource.getWorldPosition(_legPlaneV3);
  lowerSource.getWorldPosition(_legPlaneV4);

  const targetDir = _legPlaneV5.subVectors(_legPlaneV2, _legPlaneV1);
  const sourceDir = _legPlaneV6.subVectors(_legPlaneV4, _legPlaneV3);
  if (targetDir.lengthSq() < 1e-10 || sourceDir.lengthSq() < 1e-10) return false;
  targetDir.normalize();
  sourceDir.normalize();

  const dot = Math.max(-1, Math.min(1, targetDir.dot(sourceDir)));
  if (dot > 0.9999) return false;
  _legPlaneQ1.setFromUnitVectors(targetDir, sourceDir).normalize();
  upperTarget.getWorldQuaternion(_legPlaneQ2);
  _legPlaneQ2.premultiply(_legPlaneQ1).normalize();
  setBoneWorldQuaternion(upperTarget, _legPlaneQ2, _legPlaneQ3, _legPlaneQ1);
  upperTarget.updateMatrixWorld(true);
  return true;
}

function applyKneePlaneCorrection(chain) {
  const upperTarget = chain?.upper?.target || null;
  const lowerTarget = chain?.lower?.target || null;
  const footTarget = chain?.foot?.target || null;
  const upperSource = chain?.upper?.source || null;
  const lowerSource = chain?.lower?.source || null;
  const footSource = chain?.foot?.source || null;
  if (!upperTarget || !lowerTarget || !footTarget || !upperSource || !lowerSource || !footSource) return false;

  upperTarget.getWorldPosition(_legPlaneV1);
  lowerTarget.getWorldPosition(_legPlaneV2);
  footTarget.getWorldPosition(_legPlaneV3);
  _legPlaneV9.copy(_legPlaneV3);
  upperSource.getWorldPosition(_legPlaneV4);
  lowerSource.getWorldPosition(_legPlaneV5);
  footSource.getWorldPosition(_legPlaneV6);

  const upperLen = _legPlaneV1.distanceTo(_legPlaneV2);
  const lowerLen = _legPlaneV2.distanceTo(_legPlaneV3);
  if (!(upperLen > 1e-5) || !(lowerLen > 1e-5)) return false;
  if (!getBendNormal(_legPlaneV4, _legPlaneV5, _legPlaneV6, _legPlaneV7)) return false;

  const rootToEnd = _legPlaneV6.copy(_legPlaneV3).sub(_legPlaneV1);
  const dist = rootToEnd.length();
  if (!(dist > 1e-5)) return false;
  rootToEnd.multiplyScalar(1 / dist);

  const planeNormal = _legPlaneV7.addScaledVector(rootToEnd, -_legPlaneV7.dot(rootToEnd));
  if (planeNormal.lengthSq() < 1e-10) return false;
  planeNormal.normalize();
  const bendDir = _legPlaneV8.crossVectors(planeNormal, rootToEnd);
  if (bendDir.lengthSq() < 1e-10) return false;
  bendDir.normalize();

  const clampedDist = Math.min(dist, upperLen + lowerLen - 1e-5);
  const along = (clampedDist * clampedDist + upperLen * upperLen - lowerLen * lowerLen) / (2 * clampedDist);
  const heightSq = Math.max(0, upperLen * upperLen - along * along);
  const height = Math.sqrt(heightSq);

  const kneeCandidateA = _legPlaneV4.copy(_legPlaneV1).addScaledVector(rootToEnd, along).addScaledVector(bendDir, height);
  const kneeCandidateB = _legPlaneV5.copy(_legPlaneV1).addScaledVector(rootToEnd, along).addScaledVector(bendDir, -height);
  const normalAOk = getBendNormal(_legPlaneV1, kneeCandidateA, _legPlaneV3, _legPlaneV6);
  const dotA = normalAOk ? _legPlaneV6.dot(planeNormal) : -Infinity;
  const normalBOk = getBendNormal(_legPlaneV1, kneeCandidateB, _legPlaneV3, _legPlaneV6);
  const dotB = normalBOk ? _legPlaneV6.dot(planeNormal) : -Infinity;
  const desiredKnee = dotA >= dotB ? kneeCandidateA.clone() : kneeCandidateB.clone();

  const currentUpperDir = _legPlaneV2.subVectors(_legPlaneV2, _legPlaneV1);
  const desiredUpperDir = _legPlaneV4.subVectors(desiredKnee, _legPlaneV1);
  if (currentUpperDir.lengthSq() < 1e-10 || desiredUpperDir.lengthSq() < 1e-10) return false;
  currentUpperDir.normalize();
  desiredUpperDir.normalize();
  _legPlaneQ1.setFromUnitVectors(currentUpperDir, desiredUpperDir).normalize();
  upperTarget.getWorldQuaternion(_legPlaneQ2);
  _legPlaneQ2.premultiply(_legPlaneQ1).normalize();
  setBoneWorldQuaternion(upperTarget, _legPlaneQ2, _legPlaneQ3, _legPlaneQ1);
  upperTarget.updateMatrixWorld(true);

  lowerTarget.getWorldPosition(_legPlaneV2);
  footTarget.getWorldPosition(_legPlaneV3);
  const currentLowerDir = _legPlaneV5.subVectors(_legPlaneV3, _legPlaneV2);
  const desiredLowerDir = _legPlaneV6.subVectors(_legPlaneV9, _legPlaneV2);
  if (currentLowerDir.lengthSq() < 1e-10 || desiredLowerDir.lengthSq() < 1e-10) return false;
  currentLowerDir.normalize();
  desiredLowerDir.normalize();
  _legPlaneQ1.setFromUnitVectors(currentLowerDir, desiredLowerDir).normalize();
  lowerTarget.getWorldQuaternion(_legPlaneQ2);
  _legPlaneQ2.premultiply(_legPlaneQ1).normalize();
  setBoneWorldQuaternion(lowerTarget, _legPlaneQ2, _legPlaneQ3, _legPlaneQ1);
  lowerTarget.updateMatrixWorld(true);
  return true;
}

function applyElbowPlaneCorrection(chain) {
  const upperTarget = chain?.upper?.target || null;
  const lowerTarget = chain?.lower?.target || null;
  const handTarget = chain?.hand?.target || null;
  const upperSource = chain?.upper?.source || null;
  const lowerSource = chain?.lower?.source || null;
  const handSource = chain?.hand?.source || null;
  if (!upperTarget || !lowerTarget || !handTarget || !upperSource || !lowerSource || !handSource) return false;

  upperTarget.getWorldPosition(_legPlaneV1);
  lowerTarget.getWorldPosition(_legPlaneV2);
  handTarget.getWorldPosition(_legPlaneV3);
  upperSource.getWorldPosition(_legPlaneV4);
  lowerSource.getWorldPosition(_legPlaneV5);
  handSource.getWorldPosition(_legPlaneV6);

  if (!getBendNormal(_legPlaneV1, _legPlaneV2, _legPlaneV3, _legPlaneV7)) return false;
  if (!getBendNormal(_legPlaneV4, _legPlaneV5, _legPlaneV6, _legPlaneV8)) return false;

  const lowerToHand = _legPlaneV9.subVectors(_legPlaneV3, _legPlaneV2);
  if (lowerToHand.lengthSq() < 1e-10) return false;
  lowerToHand.normalize();

  const projectedTarget = _legPlaneV1.copy(_legPlaneV7).sub(lowerToHand.clone().multiplyScalar(_legPlaneV7.dot(lowerToHand)));
  const projectedSource = _legPlaneV4.copy(_legPlaneV8).sub(lowerToHand.clone().multiplyScalar(_legPlaneV8.dot(lowerToHand)));
  if (projectedTarget.lengthSq() < 1e-10 || projectedSource.lengthSq() < 1e-10) return false;
  projectedTarget.normalize();
  projectedSource.normalize();
  const dot = Math.max(-1, Math.min(1, projectedTarget.dot(projectedSource)));
  if (dot > 0.9999) return false;

  _legPlaneQ1.setFromUnitVectors(projectedTarget, projectedSource).normalize();
  lowerTarget.getWorldQuaternion(_legPlaneQ2);
  _legPlaneQ2.premultiply(_legPlaneQ1).normalize();
  setBoneWorldQuaternion(lowerTarget, _legPlaneQ2, _legPlaneQ3, _legPlaneQ1);
  lowerTarget.updateMatrixWorld(true);
  return true;
}

function applyShinDirectionCorrection(chain) {
  const lowerTarget = chain?.lower?.target || null;
  const footTarget = chain?.foot?.target || null;
  const lowerSource = chain?.lower?.source || null;
  const footSource = chain?.foot?.source || null;
  if (!lowerTarget || !footTarget || !lowerSource || !footSource) return false;

  lowerTarget.getWorldPosition(_legPlaneV1);
  footTarget.getWorldPosition(_legPlaneV2);
  lowerSource.getWorldPosition(_legPlaneV3);
  footSource.getWorldPosition(_legPlaneV4);

  const targetDir = _legPlaneV5.subVectors(_legPlaneV2, _legPlaneV1);
  const sourceDir = _legPlaneV6.subVectors(_legPlaneV4, _legPlaneV3);
  if (targetDir.lengthSq() < 1e-10 || sourceDir.lengthSq() < 1e-10) return false;
  targetDir.normalize();
  sourceDir.normalize();

  const dot = Math.max(-1, Math.min(1, targetDir.dot(sourceDir)));
  if (dot > 0.9999) return false;
  _legPlaneQ1.setFromUnitVectors(targetDir, sourceDir).normalize();
  lowerTarget.getWorldQuaternion(_legPlaneQ2);
  _legPlaneQ2.premultiply(_legPlaneQ1).normalize();
  setBoneWorldQuaternion(lowerTarget, _legPlaneQ2, _legPlaneQ3, _legPlaneQ1);
  lowerTarget.updateMatrixWorld(true);
  return true;
}

function applyFootDirectionCorrection(chain) {
  const footTarget = chain?.foot?.target || null;
  const footSource = chain?.foot?.source || null;
  const toesTarget = chain?.toes?.target || null;
  const toesSource = chain?.toes?.source || null;
  const targetLocalDir = chain?.foot?.targetPrimaryChildDirLocal || null;
  const sourceLocalDir = chain?.foot?.sourcePrimaryChildDirLocal || null;
  if (!footTarget || !footSource) return false;
  const targetDir = _legPlaneV1;
  const sourceDir = _legPlaneV2;
  if (!getFootForwardWorldDirection(footTarget, toesTarget, targetDir, targetLocalDir)) return false;
  if (!getFootForwardWorldDirection(footSource, toesSource, sourceDir, sourceLocalDir)) return false;
  if (targetDir.lengthSq() < 1e-10 || sourceDir.lengthSq() < 1e-10) return false;
  _legPlaneQ1.setFromUnitVectors(targetDir, sourceDir).normalize();
  footTarget.getWorldQuaternion(_legPlaneQ2);
  _legPlaneQ2.premultiply(_legPlaneQ1).normalize();
  setBoneWorldQuaternion(footTarget, _legPlaneQ2, _legPlaneQ3, _legPlaneQ1);
  footTarget.updateMatrixWorld(true);
  return true;
}

function applyFootPlaneCorrection(chain) {
  const lowerTarget = chain?.lower?.target || null;
  const footTarget = chain?.foot?.target || null;
  const lowerSource = chain?.lower?.source || null;
  const footSource = chain?.foot?.source || null;
  const toesTarget = chain?.toes?.target || null;
  const toesSource = chain?.toes?.source || null;
  const targetLocalDir = chain?.foot?.targetPrimaryChildDirLocal || null;
  const sourceLocalDir = chain?.foot?.sourcePrimaryChildDirLocal || null;
  if (!lowerTarget || !footTarget || !lowerSource || !footSource) return false;

  const targetFootDir = _legPlaneV1;
  const sourceFootDir = _legPlaneV2;
  if (!getFootForwardWorldDirection(footTarget, toesTarget, targetFootDir, targetLocalDir)) return false;
  if (!getFootForwardWorldDirection(footSource, toesSource, sourceFootDir, sourceLocalDir)) return false;

  lowerTarget.getWorldPosition(_legPlaneV3);
  footTarget.getWorldPosition(_legPlaneV4);
  lowerSource.getWorldPosition(_legPlaneV5);
  footSource.getWorldPosition(_legPlaneV6);

  const targetShinDir = _legPlaneV7.subVectors(_legPlaneV4, _legPlaneV3);
  const sourceShinDir = _legPlaneV8.subVectors(_legPlaneV6, _legPlaneV5);
  if (targetShinDir.lengthSq() < 1e-10 || sourceShinDir.lengthSq() < 1e-10) return false;
  targetShinDir.normalize();
  sourceShinDir.normalize();

  const targetNormal = _legPlaneV9.crossVectors(targetShinDir, targetFootDir);
  const sourceNormal = _legPlaneV3.crossVectors(sourceShinDir, sourceFootDir);
  targetNormal.addScaledVector(targetFootDir, -targetNormal.dot(targetFootDir));
  sourceNormal.addScaledVector(sourceFootDir, -sourceNormal.dot(sourceFootDir));
  if (targetNormal.lengthSq() < 1e-10 || sourceNormal.lengthSq() < 1e-10) return false;
  targetNormal.normalize();
  sourceNormal.normalize();

  const dot = Math.max(-1, Math.min(1, targetNormal.dot(sourceNormal)));
  const angle = Math.acos(dot);
  if (!(angle > 1e-5)) return false;
  const sign = Math.sign(targetFootDir.dot(_legPlaneV4.crossVectors(targetNormal, sourceNormal))) || 1;
  _legPlaneQ1.setFromAxisAngle(targetFootDir, angle * sign).normalize();
  footTarget.getWorldQuaternion(_legPlaneQ2);
  _legPlaneQ2.premultiply(_legPlaneQ1).normalize();
  setBoneWorldQuaternion(footTarget, _legPlaneQ2, _legPlaneQ3, _legPlaneQ1);
  footTarget.updateMatrixWorld(true);
  return true;
}

function applyFootMirrorCorrection(chain) {
  const lowerTarget = chain?.lower?.target || null;
  const footTarget = chain?.foot?.target || null;
  const lowerSource = chain?.lower?.source || null;
  const footSource = chain?.foot?.source || null;
  const toesTarget = chain?.toes?.target || null;
  const toesSource = chain?.toes?.source || null;
  const targetLocalDir = chain?.foot?.targetPrimaryChildDirLocal || null;
  const sourceLocalDir = chain?.foot?.sourcePrimaryChildDirLocal || null;
  if (!lowerTarget || !footTarget || !lowerSource || !footSource) return false;

  const targetFootDir = _legPlaneV1;
  const sourceFootDir = _legPlaneV2;
  if (!getFootForwardWorldDirection(footTarget, toesTarget, targetFootDir, targetLocalDir)) return false;
  if (!getFootForwardWorldDirection(footSource, toesSource, sourceFootDir, sourceLocalDir)) return false;
  const footDot = targetFootDir.dot(sourceFootDir);
  if (!(footDot < 0)) return false;

  lowerTarget.getWorldPosition(_legPlaneV3);
  footTarget.getWorldPosition(_legPlaneV4);
  const shinAxis = _legPlaneV5.subVectors(_legPlaneV4, _legPlaneV3);
  if (shinAxis.lengthSq() < 1e-10) return false;
  shinAxis.normalize();

  _legPlaneQ1.setFromAxisAngle(shinAxis, Math.PI).normalize();
  footTarget.getWorldQuaternion(_legPlaneQ2);
  _legPlaneQ2.premultiply(_legPlaneQ1).normalize();
  setBoneWorldQuaternion(footTarget, _legPlaneQ2, _legPlaneQ3, _legPlaneQ1);
  footTarget.updateMatrixWorld(true);
  return true;
}

export function resetModelRootOrientation({ modelRoot, getModelSkeletonRootBone }) {
  if (!modelRoot) return;
  const baseQ = modelRoot.userData?.__baseQuaternion;
  if (baseQ?.isQuaternion) {
    modelRoot.quaternion.copy(baseQ);
  }
  const baseP = modelRoot.userData?.__basePosition;
  if (baseP?.isVector3) {
    modelRoot.position.copy(baseP);
  }
  const rootBone = getModelSkeletonRootBone?.();
  if (rootBone && rootBone !== modelRoot) {
    const rootBaseQ = rootBone.userData?.__retargetBaseQuaternion;
    const rootBaseP = rootBone.userData?.__retargetBasePosition;
    if (rootBaseQ?.isQuaternion) {
      rootBone.quaternion.copy(rootBaseQ);
    }
    if (rootBaseP?.isVector3) {
      rootBone.position.copy(rootBaseP);
    }
  }
  modelRoot.updateMatrixWorld(true);
}

export function applyModelRootYaw({ modelRoot, yawRad, axisY, rootYawQ = null }) {
  if (!modelRoot || !Number.isFinite(yawRad) || Math.abs(yawRad) < 1e-5) return 0;
  const baseQ = modelRoot.userData?.__baseQuaternion;
  if (baseQ?.isQuaternion) {
    modelRoot.quaternion.copy(baseQ);
  }
  const yawQ = rootYawQ || new THREE.Quaternion();
  yawQ.setFromAxisAngle(axisY || new THREE.Vector3(0, 1, 0), yawRad);
  modelRoot.quaternion.premultiply(yawQ);
  modelRoot.updateMatrixWorld(true);
  return yawRad;
}

export function clearSourceOverlay({ sourceOverlay, scene }) {
  if (!sourceOverlay) return null;
  scene.remove(sourceOverlay.lines);
  scene.remove(sourceOverlay.points);
  sourceOverlay.lines.geometry.dispose();
  sourceOverlay.lines.material.dispose();
  sourceOverlay.points.geometry.dispose();
  sourceOverlay.points.material.dispose();
  return null;
}

function findBoneByCanonical(bones, keys) {
  if (!bones?.length) return null;
  const keySet = new Set(keys);
  for (const bone of bones) {
    const key = canonicalBoneKey(bone.name);
    if (keySet.has(key)) return bone;
  }
  return null;
}

function estimateFacingVector(bones) {
  const left = findBoneByCanonical(bones, ["leftShoulder", "leftUpperArm"]);
  const right = findBoneByCanonical(bones, ["rightShoulder", "rightUpperArm"]);
  const hips = findBoneByCanonical(bones, ["hips"]);
  const head = findBoneByCanonical(bones, ["head", "neck", "upperChest", "chest"]);
  if (!left || !right || !hips || !head) return null;

  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();
  const v3 = new THREE.Vector3();
  const v4 = new THREE.Vector3();
  const v5 = new THREE.Vector3();
  left.getWorldPosition(v1);
  right.getWorldPosition(v2);
  hips.getWorldPosition(v3);
  head.getWorldPosition(v4);
  const across = v2.sub(v1);
  const up = v4.sub(v3);
  if (across.lengthSq() < 1e-9 || up.lengthSq() < 1e-9) return null;

  const forward = v5.crossVectors(across.normalize(), up.normalize());
  if (forward.lengthSq() < 1e-9) return null;
  return forward.normalize().clone();
}

export function estimateFacingYawOffset(sourceBones, targetBones) {
  const s = estimateFacingVector(sourceBones);
  const t = estimateFacingVector(targetBones);
  if (!s || !t) return 0;
  s.y = 0;
  t.y = 0;
  if (s.lengthSq() < 1e-9 || t.lengthSq() < 1e-9) return 0;
  s.normalize();
  t.normalize();
  const crossY = s.clone().cross(t).y;
  const dot = Math.max(-1, Math.min(1, s.dot(t)));
  const angle = Math.atan2(crossY, dot);
  return Number.isFinite(angle) ? angle : 0;
}

export function updateSourceOverlay({ sourceOverlay, overlayUpAxis = null, overlayPivot = null }) {
  if (!sourceOverlay) return;
  const { bones, edges, pointAttr, lineAttr, overlayYaw, pivotBone } = sourceOverlay;
  const applyYaw = Number.isFinite(overlayYaw) && Math.abs(overlayYaw) > 1e-5;
  const pivot = overlayPivot || new THREE.Vector3();
  const upAxis = overlayUpAxis || new THREE.Vector3(0, 1, 0);
  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();
  if (applyYaw && pivotBone) {
    pivotBone.getWorldPosition(pivot);
  }

  for (let i = 0; i < bones.length; i += 1) {
    bones[i].getWorldPosition(v1);
    if (applyYaw) {
      v1.sub(pivot).applyAxisAngle(upAxis, overlayYaw).add(pivot);
    }
    pointAttr.array[i * 3 + 0] = v1.x;
    pointAttr.array[i * 3 + 1] = v1.y;
    pointAttr.array[i * 3 + 2] = v1.z;
  }
  pointAttr.needsUpdate = true;

  for (let i = 0; i < edges.length; i += 1) {
    const edge = edges[i];
    edge[0].getWorldPosition(v1);
    edge[1].getWorldPosition(v2);
    if (applyYaw) {
      v1.sub(pivot).applyAxisAngle(upAxis, overlayYaw).add(pivot);
      v2.sub(pivot).applyAxisAngle(upAxis, overlayYaw).add(pivot);
    }
    const base = i * 6;
    lineAttr.array[base + 0] = v1.x;
    lineAttr.array[base + 1] = v1.y;
    lineAttr.array[base + 2] = v1.z;
    lineAttr.array[base + 3] = v2.x;
    lineAttr.array[base + 4] = v2.y;
    lineAttr.array[base + 5] = v2.z;
  }
  lineAttr.needsUpdate = true;
}

export function createSourceOverlay({
  skeleton,
  scene,
  sourceOverlay,
  skeletonColor,
  sourcePointColor,
  clearSourceOverlay,
  updateSourceOverlay,
}) {
  clearSourceOverlay?.();
  const bones = skeleton?.bones || [];
  if (!bones.length) return null;

  const boneSet = new Set(bones);
  const edges = [];
  for (const bone of bones) {
    if (bone.parent && boneSet.has(bone.parent)) {
      edges.push([bone.parent, bone]);
    }
  }

  const pointGeometry = new THREE.BufferGeometry();
  const pointAttr = new THREE.BufferAttribute(new Float32Array(bones.length * 3), 3);
  pointGeometry.setAttribute("position", pointAttr);

  const lineGeometry = new THREE.BufferGeometry();
  const lineAttr = new THREE.BufferAttribute(new Float32Array(Math.max(1, edges.length * 2) * 3), 3);
  lineGeometry.setAttribute("position", lineAttr);

  const lineMaterial = new THREE.LineBasicMaterial({
    color: skeletonColor,
    transparent: true,
    opacity: 0.95,
    depthTest: false,
    depthWrite: false,
  });
  const pointsMaterial = new THREE.PointsMaterial({
    color: sourcePointColor,
    size: 8,
    sizeAttenuation: false,
    transparent: true,
    opacity: 1,
    depthTest: false,
    depthWrite: false,
  });

  const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
  lines.renderOrder = 998;
  lines.frustumCulled = false;
  const points = new THREE.Points(pointGeometry, pointsMaterial);
  points.renderOrder = 999;
  points.frustumCulled = false;

  scene.add(lines);
  scene.add(points);
  const nextOverlay = {
    bones,
    edges,
    lines,
    points,
    pointAttr,
    lineAttr,
    overlayYaw: 0,
    pivotBone: findBoneByCanonical(bones, ["hips"]) || bones[0] || null,
  };
  updateSourceOverlay?.(nextOverlay);
  return nextOverlay;
}

export function buildLiveRetargetPlan({
  skinnedMeshes,
  targetBones = null,
  sourceBones,
  namesTargetToSource,
  mixer,
  modelRoot,
  buildRestOrientationCorrection,
  cachedProfile = null,
  profile = null,
}) {
  const sourceByName = new Map(sourceBones.map((b) => [b.name, b]));
  const cachedPairMap = new Map(
    ((cachedProfile?.pairs || []).map((pair) => [pairProfileKey(pair.target, pair.source), pair]))
  );
  const uniqueSkeletons = [];
  const seenSkeletonIds = new Set();
  for (const mesh of skinnedMeshes || []) {
    const skeleton = mesh.skeleton;
    if (!skeleton?.bones?.length) continue;
    const skeletonId = skeleton.uuid || mesh.uuid;
    if (seenSkeletonIds.has(skeletonId)) continue;
    seenSkeletonIds.add(skeletonId);
    uniqueSkeletons.push(skeleton);
  }

  const pairs = [];
  const explicitTargetBones = Array.isArray(targetBones) ? targetBones.filter((bone) => bone?.isBone) : [];
  if (explicitTargetBones.length) {
    for (const targetBone of explicitTargetBones) {
      const sourceName = namesTargetToSource[targetBone.name];
      if (!sourceName) continue;
      const sourceBone = sourceByName.get(sourceName);
      if (!sourceBone) continue;
      pairs.push(createRetargetPair(targetBone, sourceBone));
    }
  }
  for (const skeleton of uniqueSkeletons) {
    for (const targetBone of skeleton.bones) {
      if (explicitTargetBones.length && explicitTargetBones.includes(targetBone)) continue;
      const sourceName = namesTargetToSource[targetBone.name];
      if (!sourceName) continue;
      const sourceBone = sourceByName.get(sourceName);
      if (!sourceBone) continue;
      pairs.push(createRetargetPair(targetBone, sourceBone));
    }
  }

  if (!pairs.length) return null;
  pairs.sort((a, b) => a.depth - b.depth);

  const sourceTime = mixer ? mixer.time : 0;
  if (mixer) {
    mixer.setTime(0);
  }
  for (const skeleton of uniqueSkeletons) {
    skeleton.pose();
  }
  modelRoot?.updateMatrixWorld(true);

  initializeRetargetPairsRestState(pairs);

  let calibratedPairs = 0;
  for (const pair of pairs) {
    if (pair.isHips) continue;
    applyPairInvertRotationOverride(pair, profile);
    if (WORLD_REST_TRANSFER_CANONICAL.has(pair.canonical) && pair.target.parent) {
      pair.useWorldRestTransfer = true;
      calibratedPairs += 1;
      continue;
    }
    const shouldUseCachedPair = !NO_CACHE_REST_CALIBRATION_CANONICAL.has(pair.canonical);
    const cachedPair = shouldUseCachedPair
      ? (cachedPairMap.get(pairProfileKey(pair.target.name, pair.source.name)) || null)
      : null;
    if (
      restoreCachedPairCalibration(pair, cachedPair, {
        parentRelativeRestDeltaCanonicals: PARENT_RELATIVE_REST_DELTA_CANONICAL,
        deserializeQuaternion,
      })
    ) {
      calibratedPairs += 1;
      continue;
    }
    initializePairRestWorldDelta(pair);
    if (
      PARENT_RELATIVE_REST_DELTA_CANONICAL.has(pair.canonical) &&
      pair.target.parent
    ) {
      pair.useParentRelativeRestDelta = true;
      calibratedPairs += 1;
      continue;
    }
    if (!applyPairRestOrientationCorrection(pair, buildRestOrientationCorrection)) {
      continue;
    }
    calibratedPairs += 1;
  }

  let posScale = 1;
  const targetEvalBones = [];
  for (const bone of explicitTargetBones) targetEvalBones.push(bone);
  for (const skeleton of uniqueSkeletons) {
    for (const bone of skeleton.bones) targetEvalBones.push(bone);
  }
  posScale = computeRetargetPosScale(sourceBones, targetEvalBones);

  if (mixer) {
    mixer.setTime(sourceTime);
  }
  modelRoot?.updateMatrixWorld(true);

  const targetRefBones = explicitTargetBones.length ? explicitTargetBones : (uniqueSkeletons[0]?.bones || []);
  const rawYawOffset = estimateFacingYawOffset(sourceBones, targetRefBones);
  const absRawYaw = Math.abs(rawYawOffset);
  let yawOffset = -rawYawOffset;
  if (absRawYaw < THREE.MathUtils.degToRad(45)) {
    yawOffset = 0;
  } else if (absRawYaw > THREE.MathUtils.degToRad(120)) {
    yawOffset = Math.sign(yawOffset || 1) * Math.PI;
  }
  if (Number.isFinite(cachedProfile?.posScale) && cachedProfile.posScale > 1e-6) {
    posScale = cachedProfile.posScale;
  }
  if (Number.isFinite(cachedProfile?.yawOffset)) {
    yawOffset = cachedProfile.yawOffset;
  }

  const { legChains, armChains } = buildProfiledChains(pairs, profile);

  return {
    pairs,
    uniqueSkeletons,
    legChains,
    armChains,
    posScale,
    yawOffset,
    calibratedPairs,
  };
}

export function exportLiveRetargetProfile(plan) {
  if (!plan?.pairs?.length) return null;
  return {
    posScale: Number.isFinite(plan.posScale) ? Number(plan.posScale.toFixed(8)) : null,
    yawOffset: Number.isFinite(plan.yawOffset) ? Number(plan.yawOffset.toFixed(8)) : null,
    calibratedPairs: plan.calibratedPairs || 0,
    pairs: plan.pairs.map((pair) => ({
      target: pair.target?.name || "",
      source: pair.source?.name || "",
      canonical: pair.canonical || "",
      hasRestCorrection: !!pair.hasRestCorrection,
      useParentRelativeRestDelta: !!pair.useParentRelativeRestDelta,
      useWorldRestTransfer: !!pair.useWorldRestTransfer,
      restCorrectionQ: pair.hasRestCorrection ? serializeQuaternion(pair.restCorrectionQ) : null,
      restWorldDeltaQ: pair.useParentRelativeRestDelta ? serializeQuaternion(pair.restWorldDeltaQ) : null,
    })),
  };
}

export function applyLiveRetargetPose({
  plan,
  modelRoot,
  liveAxisY = null,
  liveYawQ = null,
  liveQ = null,
  liveQ2 = null,
  liveQ3 = null,
  liveV = null,
}) {
  if (!plan?.pairs?.length) return;
  const yaw = Number.isFinite(plan.yawOffset) ? plan.yawOffset : 0;
  const invertRotationDelta = !!plan.invertRotationDelta;
  const yawQ = liveYawQ || new THREE.Quaternion();
  const deltaQ = liveQ || new THREE.Quaternion();
  const parentWorldQ = liveQ2 || new THREE.Quaternion();
  const targetWorldQ = liveQ3 || new THREE.Quaternion();
  const deltaV = liveV || new THREE.Vector3();
  if (Math.abs(yaw) > 1e-5) {
    yawQ.setFromAxisAngle(liveAxisY || new THREE.Vector3(0, 1, 0), yaw);
  }
  if (typeof window !== "undefined") {
    window.__vid2modelFootCorrectionDebug = [];
  }
  for (const pair of plan.pairs) {
    const pairInvertRotationDelta =
      typeof pair.invertRotationDeltaOverride === "boolean"
        ? pair.invertRotationDeltaOverride
        : invertRotationDelta;
    if (!pair.isHips && pair.useParentRelativeRestDelta) {
      pair.source.getWorldQuaternion(targetWorldQ);
      targetWorldQ.multiply(pair.restWorldDeltaQ).normalize();
      if (pair.target.parent) {
        pair.target.parent.getWorldQuaternion(parentWorldQ);
        pair.target.quaternion.copy(parentWorldQ.invert().multiply(targetWorldQ)).normalize();
      } else {
        pair.target.quaternion.copy(targetWorldQ).normalize();
      }
      pair.target.position.copy(pair.targetRestPos);
      pair.target.updateMatrixWorld(true);
      continue;
    }
    if (!pair.isHips && pair.useWorldRestTransfer && pair.target.parent) {
      pair.source.getWorldQuaternion(deltaQ);
      deltaQ.multiply(pair.sourceRestWorldQInv).normalize();
      targetWorldQ.copy(deltaQ).multiply(pair.targetRestWorldQ).normalize();
      pair.target.parent.getWorldQuaternion(parentWorldQ);
      pair.target.quaternion.copy(parentWorldQ.invert().multiply(targetWorldQ)).normalize();
      pair.target.position.copy(pair.targetRestPos);
      pair.target.updateMatrixWorld(true);
      continue;
    }
    if (pairInvertRotationDelta) {
      deltaQ.copy(pair.source.quaternion).invert().multiply(pair.sourceRestQ);
    } else {
      deltaQ.copy(pair.sourceRestQ).invert().multiply(pair.source.quaternion);
    }
    if (pair.hasRestCorrection) {
      deltaQ.premultiply(pair.restCorrectionQ).multiply(pair.restCorrectionQInv);
    }
    if (pair.isHips && Math.abs(yaw) > 1e-5) {
      pair.target.quaternion.copy(pair.targetRestQ).multiply(yawQ).multiply(deltaQ).normalize();
    } else {
      pair.target.quaternion.copy(pair.targetRestQ).multiply(deltaQ).normalize();
    }
    if (pair.isHips) {
      deltaV.copy(pair.source.position).sub(pair.sourceRestPos).multiplyScalar(plan.posScale);
      if (Math.abs(yaw) > 1e-5) {
        deltaV.applyQuaternion(yawQ);
      }
      pair.target.position.copy(pair.targetRestPos).add(deltaV);
    } else {
      pair.target.position.copy(pair.targetRestPos);
    }
    pair.target.updateMatrixWorld(true);
  }
  if (Array.isArray(plan.legChains) && plan.legChains.length) {
    for (const chain of plan.legChains) {
      if (chain.enableUpperLegDirectionCorrection) {
        applyUpperLegDirectionCorrection(chain);
      }
      const footDotStart = computeFootDot(chain);
      applyKneePlaneCorrection(chain);
      pushFootDebug({ side: chain.side, step: "kneePlane", footDotBefore: footDotStart, footDotAfter: computeFootDot(chain) });
      if (chain.enableShinDirectionCorrection) {
        const before = computeFootDot(chain);
        applyShinDirectionCorrection(chain);
        pushFootDebug({ side: chain.side, step: "shinDirection", footDotBefore: before, footDotAfter: computeFootDot(chain) });
      }
      if (chain.enableFootPlaneCorrection) {
        const before = computeFootDot(chain);
        applyFootPlaneCorrection(chain);
        pushFootDebug({ side: chain.side, step: "footPlane", footDotBefore: before, footDotAfter: computeFootDot(chain) });
      }
      if (chain.enableFootMirrorCorrection) {
        const before = computeFootDot(chain);
        applyFootMirrorCorrection(chain);
        pushFootDebug({ side: chain.side, step: "footMirror", footDotBefore: before, footDotAfter: computeFootDot(chain) });
      }
      if (chain.enableFootDirectionCorrection) {
        const before = computeFootDot(chain);
        applyFootDirectionCorrection(chain);
        pushFootDebug({ side: chain.side, step: "footDirection.post", footDotBefore: before, footDotAfter: computeFootDot(chain) });
      }
    }
  }
  if (Array.isArray(plan.armChains) && plan.armChains.length) {
    for (const chain of plan.armChains) {
      if (chain.enableElbowPlaneCorrection) {
        applyElbowPlaneCorrection(chain);
      }
    }
  }
  modelRoot?.updateMatrixWorld(true);
}
