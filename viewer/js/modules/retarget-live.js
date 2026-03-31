import * as THREE from "three";
import { canonicalBoneKey } from "./bone-utils.js";

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

function getBoneDepth(bone) {
  let depth = 0;
  let node = bone?.parent || null;
  while (node) {
    depth += 1;
    node = node.parent || null;
  }
  return depth;
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
      pairs.push({
        target: targetBone,
        source: sourceBone,
        canonical:
          canonicalBoneKey(targetBone.name) ||
          canonicalBoneKey(sourceBone.name) ||
          "",
        depth: getBoneDepth(targetBone),
        targetRestQ: new THREE.Quaternion(),
        sourceRestQ: new THREE.Quaternion(),
        targetRestWorldQ: new THREE.Quaternion(),
        sourceRestWorldQ: new THREE.Quaternion(),
        sourceRestWorldQInv: new THREE.Quaternion(),
        targetRestPos: new THREE.Vector3(),
        sourceRestPos: new THREE.Vector3(),
        restWorldDeltaQ: new THREE.Quaternion(),
        restCorrectionQ: new THREE.Quaternion(),
        restCorrectionQInv: new THREE.Quaternion(),
        hasRestCorrection: false,
        useParentRelativeRestDelta: false,
        useWorldRestTransfer: false,
        isHips:
          canonicalBoneKey(targetBone.name) === "hips" ||
          canonicalBoneKey(sourceBone.name) === "hips",
      });
    }
  }
  for (const skeleton of uniqueSkeletons) {
    for (const targetBone of skeleton.bones) {
      if (explicitTargetBones.length && explicitTargetBones.includes(targetBone)) continue;
      const sourceName = namesTargetToSource[targetBone.name];
      if (!sourceName) continue;
      const sourceBone = sourceByName.get(sourceName);
      if (!sourceBone) continue;
      pairs.push({
        target: targetBone,
        source: sourceBone,
        canonical:
          canonicalBoneKey(targetBone.name) ||
          canonicalBoneKey(sourceBone.name) ||
          "",
        depth: getBoneDepth(targetBone),
        targetRestQ: new THREE.Quaternion(),
        sourceRestQ: new THREE.Quaternion(),
        targetRestWorldQ: new THREE.Quaternion(),
        sourceRestWorldQ: new THREE.Quaternion(),
        sourceRestWorldQInv: new THREE.Quaternion(),
        targetRestPos: new THREE.Vector3(),
        sourceRestPos: new THREE.Vector3(),
        restWorldDeltaQ: new THREE.Quaternion(),
        restCorrectionQ: new THREE.Quaternion(),
        restCorrectionQInv: new THREE.Quaternion(),
        hasRestCorrection: false,
        useParentRelativeRestDelta: false,
        useWorldRestTransfer: false,
        isHips:
          canonicalBoneKey(targetBone.name) === "hips" ||
          canonicalBoneKey(sourceBone.name) === "hips",
      });
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

  for (const pair of pairs) {
    pair.targetRestQ.copy(pair.target.quaternion);
    pair.sourceRestQ.copy(pair.source.quaternion);
    pair.target.getWorldQuaternion(pair.targetRestWorldQ);
    pair.source.getWorldQuaternion(pair.sourceRestWorldQ);
    pair.sourceRestWorldQInv.copy(pair.sourceRestWorldQ).invert();
    pair.targetRestPos.copy(pair.target.position);
    pair.sourceRestPos.copy(pair.source.position);
  }

  let calibratedPairs = 0;
  for (const pair of pairs) {
    if (pair.isHips) continue;
    if (WORLD_REST_TRANSFER_CANONICAL.has(pair.canonical) && pair.target.parent) {
      pair.useWorldRestTransfer = true;
      calibratedPairs += 1;
      continue;
    }
    const shouldUseCachedPair = !NO_CACHE_REST_CALIBRATION_CANONICAL.has(pair.canonical);
    const cachedPair = shouldUseCachedPair
      ? (cachedPairMap.get(pairProfileKey(pair.target.name, pair.source.name)) || null)
      : null;
    if (cachedPair) {
      if (
        cachedPair.useParentRelativeRestDelta &&
        PARENT_RELATIVE_REST_DELTA_CANONICAL.has(pair.canonical) &&
        deserializeQuaternion(cachedPair.restWorldDeltaQ, pair.restWorldDeltaQ) &&
        pair.target.parent
      ) {
        pair.useParentRelativeRestDelta = true;
        calibratedPairs += 1;
        continue;
      }
      if (
        cachedPair.hasRestCorrection &&
        deserializeQuaternion(cachedPair.restCorrectionQ, pair.restCorrectionQ)
      ) {
        pair.restCorrectionQInv.copy(pair.restCorrectionQ).invert();
        pair.hasRestCorrection = true;
        calibratedPairs += 1;
        continue;
      }
    }
    pair.source.getWorldQuaternion(pair.restCorrectionQInv);
    pair.target.getWorldQuaternion(pair.restWorldDeltaQ);
    pair.restWorldDeltaQ.premultiply(pair.restCorrectionQInv.invert()).normalize();
    if (
      PARENT_RELATIVE_REST_DELTA_CANONICAL.has(pair.canonical) &&
      pair.target.parent
    ) {
      pair.useParentRelativeRestDelta = true;
      calibratedPairs += 1;
      continue;
    }
    const corr = buildRestOrientationCorrection?.(pair.source, pair.target) || null;
    if (!corr) continue;
    pair.restCorrectionQ.copy(corr);
    pair.restCorrectionQInv.copy(corr).invert();
    pair.hasRestCorrection = true;
    calibratedPairs += 1;
  }

  let posScale = 1;
  try {
    const sourceBox = new THREE.Box3().setFromPoints(sourceBones.map((b) => b.getWorldPosition(new THREE.Vector3())));
    const targetEvalBones = [];
    for (const bone of explicitTargetBones) targetEvalBones.push(bone);
    for (const skeleton of uniqueSkeletons) {
      for (const b of skeleton.bones) targetEvalBones.push(b);
    }
    const targetBox = new THREE.Box3().setFromPoints(targetEvalBones.map((b) => b.getWorldPosition(new THREE.Vector3())));
    const sourceHWorld = Math.max(1e-6, sourceBox.max.y - sourceBox.min.y);
    const targetH = Math.max(1e-6, targetBox.max.y - targetBox.min.y);
    const sourceRootScale = new THREE.Vector3(1, 1, 1);
    if (sourceBones[0]) {
      sourceBones[0].getWorldScale(sourceRootScale);
    }
    const sourceScaleY = Math.max(1e-6, Math.abs(sourceRootScale.y));
    const sourceHUnscaled = sourceHWorld / sourceScaleY;
    posScale = targetH / sourceHUnscaled;
  } catch (err) {
    posScale = 1;
  }

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

  return {
    pairs,
    uniqueSkeletons,
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
  for (const pair of plan.pairs) {
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
    if (invertRotationDelta) {
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
  modelRoot?.updateMatrixWorld(true);
}
