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
const LEG_LOCAL_CORRECTION_CANONICAL = new Set([
  "leftUpperLeg",
  "rightUpperLeg",
  "leftLowerLeg",
  "rightLowerLeg",
  "leftFoot",
  "rightFoot",
  "leftToes",
  "rightToes",
]);

function getBoneDepth(bone) {
  let depth = 0;
  let node = bone?.parent || null;
  while (node) {
    depth += 1;
    node = node.parent || null;
  }
  return depth;
}

export function buildVrmDirectBodyPlan({
  targetBones,
  sourceBones,
  namesTargetToSource,
  mixer,
  modelRoot,
  buildRestOrientationCorrection,
}) {
  const sourceByName = new Map((sourceBones || []).map((b) => [b.name, b]));
  const pairs = [];
  for (const targetBone of targetBones || []) {
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
      isHips:
        canonicalBoneKey(targetBone.name) === "hips" ||
        canonicalBoneKey(sourceBone.name) === "hips",
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
    });
  }

  if (!pairs.length) return null;
  pairs.sort((a, b) => a.depth - b.depth);

  const sourceTime = mixer ? mixer.time : 0;
  if (mixer) mixer.setTime(0);
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

  for (const pair of pairs) {
    if (pair.isHips) continue;
    pair.source.getWorldQuaternion(pair.restCorrectionQInv);
    pair.target.getWorldQuaternion(pair.restWorldDeltaQ);
    pair.restWorldDeltaQ.premultiply(pair.restCorrectionQInv.invert()).normalize();
    if (LEG_LOCAL_CORRECTION_CANONICAL.has(pair.canonical)) {
      const corr = buildRestOrientationCorrection?.(pair.source, pair.target) || null;
      if (!corr) continue;
      pair.restCorrectionQ.copy(corr);
      pair.restCorrectionQInv.copy(corr).invert();
      pair.hasRestCorrection = true;
      continue;
    }
    if (WORLD_REST_TRANSFER_CANONICAL.has(pair.canonical) && pair.target.parent) {
      pair.useWorldRestTransfer = true;
      continue;
    }
    if (PARENT_RELATIVE_REST_DELTA_CANONICAL.has(pair.canonical) && pair.target.parent) {
      pair.useParentRelativeRestDelta = true;
      continue;
    }
    const corr = buildRestOrientationCorrection?.(pair.source, pair.target) || null;
    if (!corr) continue;
    pair.restCorrectionQ.copy(corr);
    pair.restCorrectionQInv.copy(corr).invert();
    pair.hasRestCorrection = true;
  }

  let posScale = 1;
  try {
    const sourceBox = new THREE.Box3().setFromPoints(sourceBones.map((b) => b.getWorldPosition(new THREE.Vector3())));
    const targetBox = new THREE.Box3().setFromPoints((targetBones || []).map((b) => b.getWorldPosition(new THREE.Vector3())));
    const sourceHWorld = Math.max(1e-6, sourceBox.max.y - sourceBox.min.y);
    const targetH = Math.max(1e-6, targetBox.max.y - targetBox.min.y);
    const sourceRootScale = new THREE.Vector3(1, 1, 1);
    if (sourceBones[0]) sourceBones[0].getWorldScale(sourceRootScale);
    const sourceScaleY = Math.max(1e-6, Math.abs(sourceRootScale.y));
    const sourceHUnscaled = sourceHWorld / sourceScaleY;
    posScale = targetH / sourceHUnscaled;
  } catch (err) {
    posScale = 1;
  }

  if (mixer) mixer.setTime(sourceTime);
  modelRoot?.updateMatrixWorld(true);

  return {
    pairs,
    uniqueSkeletons: [{ bones: targetBones || [] }],
    posScale,
    yawOffset: 0,
    invertRotationDelta: true,
    calibratedPairs: pairs.filter((pair) =>
      pair.useWorldRestTransfer || pair.useParentRelativeRestDelta || pair.hasRestCorrection
    ).length,
    directVrm: true,
  };
}
