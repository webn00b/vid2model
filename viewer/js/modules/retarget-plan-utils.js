import * as THREE from "three";
import { canonicalBoneKey } from "./bone-utils.js";
import {
  buildArmChainPairs,
  buildLegChainPairs,
  getBoneDepth,
  getPrimaryChildDirectionLocal,
} from "./retarget-chain-utils.js";

export function createRetargetPair(targetBone, sourceBone) {
  return {
    target: targetBone,
    source: sourceBone,
    canonical:
      canonicalBoneKey(targetBone?.name) ||
      canonicalBoneKey(sourceBone?.name) ||
      "",
    depth: getBoneDepth(targetBone),
    isHips:
      canonicalBoneKey(targetBone?.name) === "hips" ||
      canonicalBoneKey(sourceBone?.name) === "hips",
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
  };
}

export function initializeRetargetPairsRestState(pairs) {
  for (const pair of pairs || []) {
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
}

export function applyPairInvertRotationOverride(pair, profile = null) {
  if (
    pair?.canonical &&
    typeof profile?.invertRotationDeltaByCanonical?.[pair.canonical] === "boolean"
  ) {
    pair.invertRotationDeltaOverride =
      profile.invertRotationDeltaByCanonical[pair.canonical];
  }
}

export function initializePairRestWorldDelta(pair) {
  if (!pair?.source || !pair?.target) return false;
  pair.source.getWorldQuaternion(pair.restCorrectionQInv);
  pair.target.getWorldQuaternion(pair.restWorldDeltaQ);
  pair.restWorldDeltaQ
    .premultiply(pair.restCorrectionQInv.invert())
    .normalize();
  return true;
}

export function applyPairRestOrientationCorrection(
  pair,
  buildRestOrientationCorrection
) {
  const corr =
    buildRestOrientationCorrection?.(pair?.source, pair?.target) || null;
  if (!corr) return false;
  pair.restCorrectionQ.copy(corr);
  pair.restCorrectionQInv.copy(corr).invert();
  pair.hasRestCorrection = true;
  return true;
}

export function restoreCachedPairCalibration(
  pair,
  cachedPair,
  {
    parentRelativeRestDeltaCanonicals = null,
    deserializeQuaternion = null,
  } = {}
) {
  if (!pair || !cachedPair || typeof deserializeQuaternion !== "function") {
    return false;
  }
  if (
    cachedPair.useParentRelativeRestDelta &&
    parentRelativeRestDeltaCanonicals?.has(pair.canonical) &&
    deserializeQuaternion(cachedPair.restWorldDeltaQ, pair.restWorldDeltaQ) &&
    pair.target.parent
  ) {
    pair.useParentRelativeRestDelta = true;
    return true;
  }
  if (
    cachedPair.hasRestCorrection &&
    deserializeQuaternion(cachedPair.restCorrectionQ, pair.restCorrectionQ)
  ) {
    pair.restCorrectionQInv.copy(pair.restCorrectionQ).invert();
    pair.hasRestCorrection = true;
    return true;
  }
  return false;
}

export function buildProfiledChains(pairs, profile = null) {
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

  return { legChains, armChains };
}

export function computeRetargetPosScale(sourceBones, targetBones) {
  try {
    const sourcePoints = (sourceBones || []).map((bone) =>
      bone.getWorldPosition(new THREE.Vector3())
    );
    const targetPoints = (targetBones || []).map((bone) =>
      bone.getWorldPosition(new THREE.Vector3())
    );
    if (!sourcePoints.length || !targetPoints.length) return 1;
    const sourceBox = new THREE.Box3().setFromPoints(sourcePoints);
    const targetBox = new THREE.Box3().setFromPoints(targetPoints);
    const sourceHWorld = Math.max(1e-6, sourceBox.max.y - sourceBox.min.y);
    const targetH = Math.max(1e-6, targetBox.max.y - targetBox.min.y);
    const sourceRootScale = new THREE.Vector3(1, 1, 1);
    if (sourceBones?.[0]) {
      sourceBones[0].getWorldScale(sourceRootScale);
    }
    const sourceScaleY = Math.max(1e-6, Math.abs(sourceRootScale.y));
    const sourceHUnscaled = sourceHWorld / sourceScaleY;
    return targetH / sourceHUnscaled;
  } catch (err) {
    return 1;
  }
}
