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
const _childDirV1 = new THREE.Vector3();
const _childDirV2 = new THREE.Vector3();
const _childDirQ1 = new THREE.Quaternion();

function getBoneDepth(bone) {
  let depth = 0;
  let node = bone?.parent || null;
  while (node) {
    depth += 1;
    node = node.parent || null;
  }
  return depth;
}

function getPrimaryChildDirectionLocal(bone) {
  if (!bone?.isBone) return null;
  bone.getWorldPosition(_childDirV1);
  let bestChild = null;
  let bestLenSq = 0;
  for (const child of bone.children || []) {
    if (!child?.isBone) continue;
    child.getWorldPosition(_childDirV2);
    const lenSq = _childDirV2.distanceToSquared(_childDirV1);
    if (lenSq > bestLenSq) {
      bestLenSq = lenSq;
      bestChild = child;
    }
  }
  if (!bestChild || bestLenSq < 1e-10) return null;
  bestChild.getWorldPosition(_childDirV2);
  const dir = new THREE.Vector3().copy(_childDirV2).sub(_childDirV1);
  bone.getWorldQuaternion(_childDirQ1);
  dir.applyQuaternion(_childDirQ1.invert());
  if (dir.lengthSq() < 1e-10) return null;
  return dir.normalize();
}

function buildLegChainPairs(pairs, side) {
  if (!Array.isArray(pairs) || !side) return null;
  const byCanonical = new Map(pairs.map((pair) => [pair.canonical, pair]));
  const upper = byCanonical.get(`${side}UpperLeg`) || null;
  const lower = byCanonical.get(`${side}LowerLeg`) || null;
  const foot = byCanonical.get(`${side}Foot`) || null;
  const toes = byCanonical.get(`${side}Toes`) || null;
  if (!upper || !lower || !foot) return null;
  return {
    side,
    upper,
    lower,
    foot,
    toes,
    enableUpperLegDirectionCorrection: false,
    enableShinDirectionCorrection: false,
    enableFootDirectionCorrection: false,
    enableFootPlaneCorrection: false,
    enableFootMirrorCorrection: false,
  };
}

function buildArmChainPairs(pairs, side) {
  if (!Array.isArray(pairs) || !side) return null;
  const byCanonical = new Map(pairs.map((pair) => [pair.canonical, pair]));
  const upper = byCanonical.get(`${side}UpperArm`) || null;
  const lower = byCanonical.get(`${side}LowerArm`) || null;
  const hand = byCanonical.get(`${side}Hand`) || null;
  if (!upper || !lower || !hand) return null;
  return {
    side,
    upper,
    lower,
    hand,
    enableElbowPlaneCorrection: false,
  };
}

export function buildVrmDirectBodyPlan({
  targetBones,
  sourceBones,
  namesTargetToSource,
  mixer,
  modelRoot,
  buildRestOrientationCorrection,
  profile = null,
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
      invertRotationDeltaOverride: null,
      targetPrimaryChildDirLocal: null,
      sourcePrimaryChildDirLocal: null,
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
    pair.targetPrimaryChildDirLocal = getPrimaryChildDirectionLocal(pair.target);
    pair.sourcePrimaryChildDirLocal = getPrimaryChildDirectionLocal(pair.source);
  }

  for (const pair of pairs) {
    if (pair.isHips) continue;
    const disableWorldRestTransfer = !!(
      pair.canonical &&
      profile?.disableWorldRestTransferByCanonical?.[pair.canonical]
    );
    const disableParentRelativeRestDelta = !!(
      pair.canonical &&
      profile?.disableParentRelativeRestDeltaByCanonical?.[pair.canonical]
    );
    if (
      pair.canonical &&
      typeof profile?.invertRotationDeltaByCanonical?.[pair.canonical] === "boolean"
    ) {
      pair.invertRotationDeltaOverride = profile.invertRotationDeltaByCanonical[pair.canonical];
    }
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
    if (!disableWorldRestTransfer && WORLD_REST_TRANSFER_CANONICAL.has(pair.canonical) && pair.target.parent) {
      pair.useWorldRestTransfer = true;
      continue;
    }
    if (
      !disableParentRelativeRestDelta &&
      PARENT_RELATIVE_REST_DELTA_CANONICAL.has(pair.canonical) &&
      pair.target.parent
    ) {
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

  const legChains = [];
  const armChains = [];
  if (profile?.enableKneePlaneCorrectionBySide?.left) {
    const leftChain = buildLegChainPairs(pairs, "left");
    if (leftChain) {
      leftChain.enableUpperLegDirectionCorrection = !!profile?.enableUpperLegDirectionCorrectionBySide?.left;
      leftChain.enableShinDirectionCorrection = !!profile?.enableShinDirectionCorrectionBySide?.left;
      leftChain.enableFootDirectionCorrection = !!profile?.enableFootDirectionCorrectionBySide?.left;
      leftChain.enableFootPlaneCorrection = !!profile?.enableFootPlaneCorrectionBySide?.left;
      leftChain.enableFootMirrorCorrection = !!profile?.enableFootMirrorCorrectionBySide?.left;
      legChains.push(leftChain);
    }
  }
  if (profile?.enableKneePlaneCorrectionBySide?.right) {
    const rightChain = buildLegChainPairs(pairs, "right");
    if (rightChain) {
      rightChain.enableUpperLegDirectionCorrection = !!profile?.enableUpperLegDirectionCorrectionBySide?.right;
      rightChain.enableShinDirectionCorrection = !!profile?.enableShinDirectionCorrectionBySide?.right;
      rightChain.enableFootDirectionCorrection = !!profile?.enableFootDirectionCorrectionBySide?.right;
      rightChain.enableFootPlaneCorrection = !!profile?.enableFootPlaneCorrectionBySide?.right;
      rightChain.enableFootMirrorCorrection = !!profile?.enableFootMirrorCorrectionBySide?.right;
      legChains.push(rightChain);
    }
  }
  if (profile?.enableElbowPlaneCorrectionBySide?.left) {
    const leftArmChain = buildArmChainPairs(pairs, "left");
    if (leftArmChain) {
      leftArmChain.enableElbowPlaneCorrection = !!profile?.enableElbowPlaneCorrectionBySide?.left;
      armChains.push(leftArmChain);
    }
  }
  if (profile?.enableElbowPlaneCorrectionBySide?.right) {
    const rightArmChain = buildArmChainPairs(pairs, "right");
    if (rightArmChain) {
      rightArmChain.enableElbowPlaneCorrection = !!profile?.enableElbowPlaneCorrectionBySide?.right;
      armChains.push(rightArmChain);
    }
  }

  return {
    pairs,
    legChains,
    armChains,
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
