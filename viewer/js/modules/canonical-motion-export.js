import * as THREE from "three";
import {
  buildNamesByCanonical,
  buildRestPoseChains,
  buildSegmentLengths,
  canonicalMotionKeysForStage,
  collectClipSampleTimes,
  getCanonicalSegmentChild,
  toRoundedQuatArray,
  toRoundedVec3Array,
} from "./canonical-motion-utils.js";

const _posA = new THREE.Vector3();
const _posB = new THREE.Vector3();
const _posC = new THREE.Vector3();
const _dirA = new THREE.Vector3();
const _dirB = new THREE.Vector3();
const _dirC = new THREE.Vector3();
const _quatA = new THREE.Quaternion();
const _quatB = new THREE.Quaternion();
const _eulerA = new THREE.Euler(0, 0, 0, "YXZ");

function getBoneWorldPos(bone, out) {
  if (!bone?.isBone || !out) return null;
  bone.getWorldPosition(out);
  return out;
}

function getWorldDirection(fromBone, toBone, out) {
  if (!getBoneWorldPos(fromBone, _posA) || !getBoneWorldPos(toBone, _posB) || !out) return null;
  out.copy(_posB).sub(_posA);
  if (out.lengthSq() < 1e-10) return null;
  return out.normalize();
}

function getBoneDirection(canonicalMap, canonical, out) {
  const bone = canonicalMap.get(canonical) || null;
  const childCanonical = getCanonicalSegmentChild(canonical);
  const child = childCanonical ? canonicalMap.get(childCanonical) || null : null;
  if (bone?.isBone && child?.isBone) {
    return getWorldDirection(bone, child, out);
  }
  if (bone?.isBone && bone.parent?.isBone) {
    return getWorldDirection(bone.parent, bone, out);
  }
  return null;
}

function computeChainNormal(aBone, bBone, cBone, out) {
  if (!getBoneWorldPos(aBone, _posA) || !getBoneWorldPos(bBone, _posB) || !getBoneWorldPos(cBone, _posC) || !out) {
    return null;
  }
  _dirA.copy(_posB).sub(_posA);
  _dirB.copy(_posC).sub(_posB);
  if (_dirA.lengthSq() < 1e-10 || _dirB.lengthSq() < 1e-10) return null;
  _dirA.normalize();
  _dirB.normalize();
  out.copy(_dirA).cross(_dirB);
  if (out.lengthSq() < 1e-10) return null;
  return out.normalize();
}

function computeShoulderSpan(canonicalMap, out) {
  const left = canonicalMap.get("leftShoulder") || canonicalMap.get("leftUpperArm") || null;
  const right = canonicalMap.get("rightShoulder") || canonicalMap.get("rightUpperArm") || null;
  if (!left?.isBone || !right?.isBone) return null;
  return getWorldDirection(left, right, out);
}

function computeHipSpan(canonicalMap, out) {
  const left = canonicalMap.get("leftUpperLeg") || null;
  const right = canonicalMap.get("rightUpperLeg") || null;
  if (!left?.isBone || !right?.isBone) return null;
  return getWorldDirection(left, right, out);
}

function computeSegmentNormal(canonicalMap, canonical, out) {
  switch (canonical) {
    case "hips":
      return computeHipSpan(canonicalMap, out) || computeShoulderSpan(canonicalMap, out);
    case "spine":
    case "chest":
    case "upperChest":
    case "neck":
    case "head":
      return computeShoulderSpan(canonicalMap, out);
    case "leftUpperArm":
      return computeChainNormal(
        canonicalMap.get("leftUpperArm"),
        canonicalMap.get("leftLowerArm"),
        canonicalMap.get("leftHand"),
        out
      );
    case "rightUpperArm":
      return computeChainNormal(
        canonicalMap.get("rightUpperArm"),
        canonicalMap.get("rightLowerArm"),
        canonicalMap.get("rightHand"),
        out
      );
    case "leftLowerArm":
    case "leftHand":
      return computeChainNormal(
        canonicalMap.get("leftUpperArm"),
        canonicalMap.get("leftLowerArm"),
        canonicalMap.get("leftHand"),
        out
      );
    case "rightLowerArm":
    case "rightHand":
      return computeChainNormal(
        canonicalMap.get("rightUpperArm"),
        canonicalMap.get("rightLowerArm"),
        canonicalMap.get("rightHand"),
        out
      );
    case "leftUpperLeg":
      return computeChainNormal(
        canonicalMap.get("leftUpperLeg"),
        canonicalMap.get("leftLowerLeg"),
        canonicalMap.get("leftFoot"),
        out
      );
    case "rightUpperLeg":
      return computeChainNormal(
        canonicalMap.get("rightUpperLeg"),
        canonicalMap.get("rightLowerLeg"),
        canonicalMap.get("rightFoot"),
        out
      );
    case "leftLowerLeg":
    case "leftFoot":
      return computeChainNormal(
        canonicalMap.get("leftUpperLeg"),
        canonicalMap.get("leftLowerLeg"),
        canonicalMap.get("leftFoot"),
        out
      );
    case "rightLowerLeg":
    case "rightFoot":
      return computeChainNormal(
        canonicalMap.get("rightUpperLeg"),
        canonicalMap.get("rightLowerLeg"),
        canonicalMap.get("rightFoot"),
        out
      );
    default:
      return null;
  }
}

function buildFacingQuaternionFromHips(hipsBone) {
  if (!hipsBone?.isBone) return [0, 0, 0, 1];
  hipsBone.getWorldQuaternion(_quatA);
  _eulerA.setFromQuaternion(_quatA, "YXZ");
  _quatB.setFromEuler(new THREE.Euler(0, _eulerA.y, 0, "YXZ"));
  return toRoundedQuatArray(_quatB) || [0, 0, 0, 1];
}

function buildFrame(canonicalMap, time, keys) {
  const hipsBone = canonicalMap.get("hips") || null;
  const frame = {
    time: Number(Number(time || 0).toFixed(6)),
    root: {
      position: hipsBone?.isBone ? toRoundedVec3Array(getBoneWorldPos(hipsBone, _posA)) : [0, 0, 0],
      facingQuat: buildFacingQuaternionFromHips(hipsBone),
    },
    segments: {},
    contacts: {
      leftFootPlant: false,
      rightFootPlant: false,
    },
  };

  const leftFoot = canonicalMap.get("leftFoot") || null;
  const rightFoot = canonicalMap.get("rightFoot") || null;
  const leftToes = canonicalMap.get("leftToes") || null;
  const rightToes = canonicalMap.get("rightToes") || null;
  const leftNormal = leftFoot && leftToes ? computeChainNormal(canonicalMap.get("leftLowerLeg"), leftFoot, leftToes, _dirA) : null;
  const rightNormal = rightFoot && rightToes ? computeChainNormal(canonicalMap.get("rightLowerLeg"), rightFoot, rightToes, _dirB) : null;
  frame.contacts.leftFootPlant = !!(leftNormal && Math.abs(leftNormal.y) > 0.75);
  frame.contacts.rightFootPlant = !!(rightNormal && Math.abs(rightNormal.y) > 0.75);

  for (const canonical of keys) {
    const dir = getBoneDirection(canonicalMap, canonical, _dirA);
    if (!dir) continue;
    const normal = computeSegmentNormal(canonicalMap, canonical, _dirB);
    frame.segments[canonical] = {
      dir: toRoundedVec3Array(dir),
      normal: normal ? toRoundedVec3Array(normal) : null,
      bendHint:
        canonical.includes("UpperArm") ||
        canonical.includes("LowerArm") ||
        canonical.includes("Hand") ||
        canonical.includes("UpperLeg") ||
        canonical.includes("LowerLeg") ||
        canonical.includes("Foot")
          ? (normal ? toRoundedVec3Array(normal) : null)
          : null,
    };
  }

  return frame;
}

export function buildCanonicalMotion({
  sourceResult,
  mixer,
  stage = "body",
  canonicalFilter = null,
  buildCanonicalBoneMap,
}) {
  const clip = sourceResult?.clip || null;
  const bones = sourceResult?.skeleton?.bones || [];
  if (!clip || !bones.length || !mixer || typeof buildCanonicalBoneMap !== "function") {
    return null;
  }

  const keys = canonicalMotionKeysForStage(stage, canonicalFilter);
  const times = collectClipSampleTimes(clip);
  const sourceTime = mixer.time;
  const canonicalMap = buildCanonicalBoneMap(bones);
  const sourceCanonicalNames = buildNamesByCanonical(canonicalMap);
  const hipsBone = canonicalMap.get("hips") || bones[0] || null;
  const rootBone = bones[0] || null;

  try {
    mixer.setTime(0);
    rootBone?.updateMatrixWorld(true);

    const restPose = {
      rootToHips:
        rootBone?.isBone && hipsBone?.isBone
          ? toRoundedVec3Array(
              _posA.copy(getBoneWorldPos(hipsBone, _posA)).sub(getBoneWorldPos(rootBone, _posB))
            )
          : [0, 0, 0],
      segmentLengths: buildSegmentLengths(canonicalMap, keys),
      chains: buildRestPoseChains(keys),
      basis: {
        up: [0, 1, 0],
        forward: [0, 0, 1],
        right: [1, 0, 0],
      },
      sourceCanonicalNames,
    };

    const frames = [];
    for (const time of times) {
      mixer.setTime(time);
      rootBone?.updateMatrixWorld(true);
      frames.push(buildFrame(canonicalMap, time, keys));
    }

    return {
      format: "vid2model.canonical-motion.v1",
      generatedAt: new Date().toISOString(),
      source: {
        clipName: clip.name || "source",
        frameCount: frames.length,
        fps:
          times.length >= 2 && times[1] > times[0]
            ? Number((1 / Math.max(1e-6, times[1] - times[0])).toFixed(6))
            : 30,
        duration: Number(Number(clip.duration || 0).toFixed(6)),
      },
      stage: String(stage || "body").trim().toLowerCase(),
      restPose,
      frames,
    };
  } finally {
    mixer.setTime(sourceTime || 0);
    rootBone?.updateMatrixWorld(true);
  }
}
