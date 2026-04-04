import fs from "node:fs/promises";
import path from "node:path";

import * as THREE from "three";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { BVHLoader } from "three/addons/loaders/BVHLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";

import { canonicalBoneKey, parseTrackName } from "./bone-utils.js";
import {
  buildCanonicalBoneMap,
  buildRenamedClip,
  buildRetargetMap,
  buildStageSourceClip,
  canonicalPoseSignature,
  collectLimbDiagnostics,
  resolvedTrackCountAcrossMeshes,
  resolvedTrackCountForTarget,
  scaleClipRotationsByCanonical,
} from "./retarget-helpers.js";
import {
  buildBindingsForAttempt,
  computePoseMatchError,
  evaluateRootYawCandidates,
  probeMotionForBindings,
} from "./retarget-eval.js";
import {
  applyBoneLengthCalibration,
  applyFingerLengthCalibration,
  buildArmRefinementCalibration,
  buildBodyLengthCalibration,
  buildFingerLengthCalibration,
  resetBoneLengthCalibration,
  restoreBonePositionSnapshot,
  snapshotCanonicalBonePositions,
} from "./retarget-calibration.js";
import {
  buildRootYawCandidates,
  collectAlignmentDiagnostics,
  computeHipsYawError,
  dumpRetargetAlignmentDiagnostics,
  summarizeSourceRootYawClip,
} from "./retarget-analysis.js";
import {
  applyLiveRetargetPose,
  applyModelRootYaw,
  buildLiveRetargetPlan,
  estimateFacingYawOffset,
  resetModelRootOrientation,
} from "./retarget-live.js";
import { buildVrmDirectBodyPlan } from "./retarget-vrm.js";
import { createViewerAlignmentTools } from "./viewer-alignment.js";
import { createViewerChainDiagnostics } from "./viewer-chain-diagnostics.js";
import { applyVrmHumanoidBoneNames, findSkinnedMeshes } from "./viewer-model-loader.js";
import { createViewerModelAnalysisTools } from "./viewer-model-analysis.js";
import { createViewerRigProfileService } from "./viewer-rig-profile-service.js";
import { createViewerRuntimeDiagnostics } from "./viewer-runtime-diagnostics.js";
import { collectRetargetAttempts, selectRetargetAttempt } from "./viewer-retarget-attempts.js";
import { autoNameTargetBones, maybeApplyTopologyFallback } from "./viewer-topology-fallback.js";
import {
  ARM_REFINEMENT_CANONICAL,
  RETARGET_BODY_CANONICAL,
  RETARGET_STAGES,
} from "./retarget-constants.js";
import {
  resolveBodyMetricCanonicalFilter,
  resolveRetargetStageCanonicalFilter,
} from "./retarget-stage-contract.js";
import { getBuiltinRigProfile } from "./rig-profiles.js";
import { installHeadlessNodeRuntime } from "./headless-node-runtime.js";

const VRM_HUMANOID_BONE_NAMES = [
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
  "leftThumbMetacarpal",
  "leftThumbProximal",
  "leftThumbDistal",
  "leftIndexProximal",
  "leftIndexIntermediate",
  "leftIndexDistal",
  "leftMiddleProximal",
  "leftMiddleIntermediate",
  "leftMiddleDistal",
  "leftRingProximal",
  "leftRingIntermediate",
  "leftRingDistal",
  "leftLittleProximal",
  "leftLittleIntermediate",
  "leftLittleDistal",
  "rightThumbMetacarpal",
  "rightThumbProximal",
  "rightThumbDistal",
  "rightIndexProximal",
  "rightIndexIntermediate",
  "rightIndexDistal",
  "rightMiddleProximal",
  "rightMiddleIntermediate",
  "rightMiddleDistal",
  "rightRingProximal",
  "rightRingIntermediate",
  "rightRingDistal",
  "rightLittleProximal",
  "rightLittleIntermediate",
  "rightLittleDistal",
];

function hashString(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function collectModelBoneRows(skinnedMeshes) {
  const rows = [];
  const seenBoneIds = new Set();
  for (const mesh of skinnedMeshes || []) {
    for (const bone of mesh.skeleton?.bones || []) {
      const boneId = bone.uuid || `${mesh.uuid}:${bone.name}`;
      if (seenBoneIds.has(boneId)) continue;
      seenBoneIds.add(boneId);
      rows.push({
        bone: bone.name || "(unnamed)",
        canonical: canonicalBoneKey(bone.name) || "",
        parent: bone.parent?.isBone ? bone.parent.name : "",
        mesh: mesh.name || "(unnamed-skinned-mesh)",
      });
    }
  }
  return rows;
}

function buildModelRigFingerprint(skinnedMeshes, label = "") {
  const rows = collectModelBoneRows(skinnedMeshes);
  const raw = [
    String(label || ""),
    ...rows.map((row) => `${row.mesh}|${row.parent}|${row.bone}|${row.canonical}`),
  ].join("\n");
  return `rig:${hashString(raw)}`;
}

function augmentRetargetTargetBonesWithFallback(baseBones, fallbackBones, stage) {
  const result = [];
  const seen = new Set();
  for (const bone of baseBones || []) {
    if (!bone?.isBone) continue;
    const id = bone.uuid || bone.name;
    if (seen.has(id)) continue;
    seen.add(id);
    result.push(bone);
  }
  if (stage !== "body") return result;

  const requiredCanonicals = ["leftToes", "rightToes"];
  const presentCanonicals = new Set(result.map((bone) => canonicalBoneKey(bone.name)).filter(Boolean));
  for (const canonical of requiredCanonicals) {
    if (presentCanonicals.has(canonical)) continue;
    const fallbackBone =
      (fallbackBones || []).find((bone) => (canonicalBoneKey(bone.name) || "") === canonical) || null;
    if (!fallbackBone?.isBone) continue;
    const id = fallbackBone.uuid || fallbackBone.name;
    if (seen.has(id)) continue;
    seen.add(id);
    result.push(fallbackBone);
  }
  return result;
}

function resolveRetargetTargetBones({
  stage,
  modelSkinnedMesh,
  modelVrmHumanoidBones,
  modelVrmNormalizedHumanoidBones,
  preferNormalized = false,
}) {
  const fallback = modelSkinnedMesh?.skeleton?.bones || [];
  const vrmBones =
    preferNormalized && modelVrmNormalizedHumanoidBones.length
      ? modelVrmNormalizedHumanoidBones
      : modelVrmHumanoidBones;
  if (!vrmBones.length) return fallback;
  const base =
    stage === "body"
      ? vrmBones.filter((bone) => RETARGET_BODY_CANONICAL.has(canonicalBoneKey(bone.name) || ""))
      : vrmBones.slice();
  return augmentRetargetTargetBonesWithFallback(base, fallback, stage);
}

function clipUsesBonesSyntax(clip) {
  return (clip?.tracks || []).some((track) => track.name.startsWith(".bones["));
}

function shortErr(err, limit = 120) {
  const text = String(err?.message || err || "");
  return text.length > limit ? `${text.slice(0, limit)}...` : text;
}

function createMemoryStorage(initialEntries = []) {
  const store = new Map();
  if (initialEntries.length) {
    store.set("vid2model.rigProfiles.headless", JSON.stringify(initialEntries));
  }
  return {
    getItem(key) {
      return store.has(key) ? store.get(key) : null;
    },
    setItem(key, value) {
      store.set(key, String(value));
    },
  };
}

function sanitizeForJson(value) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch (err) {
    return {
      __serializeError: String(err?.message || err || "serialize_error"),
      text: String(value),
    };
  }
}

function buildEventMap(records) {
  const out = {};
  for (const record of records || []) {
    if (!record?.event) continue;
    if (!Object.hasOwn(out, record.event)) {
      out[record.event] = record.payload;
      continue;
    }
    if (Array.isArray(out[record.event])) {
      out[record.event].push(record.payload);
      continue;
    }
    out[record.event] = [out[record.event], record.payload];
  }
  return out;
}

function applyRigProfileNames(baseResult, profile, targetBones, sourceBones, canonicalFilter) {
  if (!profile?.namesTargetToSource) return baseResult;
  const targetByName = new Map((targetBones || []).map((bone) => [bone.name, bone]));
  const sourceNameSet = new Set((sourceBones || []).map((bone) => bone.name));
  const names = { ...(baseResult?.names || {}) };
  for (const [targetName, sourceName] of Object.entries(profile.namesTargetToSource || {})) {
    const targetBone = targetByName.get(targetName);
    if (!targetBone || !sourceNameSet.has(sourceName)) continue;
    const canonical = canonicalBoneKey(targetBone.name);
    if (canonicalFilter && (!canonical || !canonicalFilter.has(canonical))) continue;
    names[targetName] = sourceName;
  }

  let matched = 0;
  let canonicalCandidates = 0;
  const unmatchedSample = [];
  const unmatchedHumanoid = [];
  for (const bone of targetBones || []) {
    const canonical = canonicalBoneKey(bone.name);
    if (canonicalFilter && (!canonical || !canonicalFilter.has(canonical))) continue;
    if (canonical) canonicalCandidates += 1;
    if (names[bone.name] && sourceNameSet.has(names[bone.name])) {
      matched += 1;
    } else if (unmatchedSample.length < 30) {
      unmatchedSample.push({ target: bone.name, canonical: canonical || "n/a" });
      if (canonical && unmatchedHumanoid.length < 30) {
        unmatchedHumanoid.push({ target: bone.name, canonical });
      }
    }
  }

  return {
    names,
    matched,
    canonicalCandidates,
    unmatchedSample,
    unmatchedHumanoid,
    sourceMatched: new Set(Object.values(names).filter((name) => sourceNameSet.has(name))).size,
  };
}

function maybeSwapMirroredHumanoidSides(baseResult, targetBones, sourceBones, canonicalFilter) {
  if (!baseResult?.names) return { ...baseResult, mirroredSidesApplied: false };
  const targetMap = buildCanonicalBoneMap(targetBones || []);
  const sourceMap = buildCanonicalBoneMap(sourceBones || []);
  const probes = [
    ["leftUpperLeg", "rightUpperLeg"],
    ["leftLowerLeg", "rightLowerLeg"],
    ["leftFoot", "rightFoot"],
    ["leftUpperArm", "rightUpperArm"],
    ["leftLowerArm", "rightLowerArm"],
    ["leftHand", "rightHand"],
  ];
  const v1 = new THREE.Vector3();
  const v2 = new THREE.Vector3();
  let mirroredVotes = 0;
  let totalVotes = 0;
  for (const [leftKey, rightKey] of probes) {
    const sourceLeft = sourceMap.get(leftKey);
    const sourceRight = sourceMap.get(rightKey);
    const targetLeft = targetMap.get(leftKey);
    const targetRight = targetMap.get(rightKey);
    if (!sourceLeft || !sourceRight || !targetLeft || !targetRight) continue;
    sourceLeft.getWorldPosition(v1);
    sourceRight.getWorldPosition(v2);
    const sourceSign = Math.sign(v1.x - v2.x);
    targetLeft.getWorldPosition(v1);
    targetRight.getWorldPosition(v2);
    const targetSign = Math.sign(v1.x - v2.x);
    if (!sourceSign || !targetSign) continue;
    totalVotes += 1;
    if (sourceSign !== targetSign) mirroredVotes += 1;
  }
  if (!totalVotes || mirroredVotes < Math.ceil(totalVotes / 2)) {
    return { ...baseResult, mirroredSidesApplied: false };
  }

  const sourceByCanonical = new Map();
  for (const bone of sourceBones || []) {
    const canonical = canonicalBoneKey(bone.name) || "";
    if (canonical) sourceByCanonical.set(canonical, bone.name);
  }
  const swapCanonical = (canonical) => {
    if (canonical.startsWith("left")) return `right${canonical.slice(4)}`;
    if (canonical.startsWith("right")) return `left${canonical.slice(5)}`;
    return canonical;
  };

  const names = { ...(baseResult.names || {}) };
  for (const bone of targetBones || []) {
    const canonical = canonicalBoneKey(bone.name) || "";
    if (!canonical) continue;
    if (canonicalFilter && !canonicalFilter.has(canonical)) continue;
    const swapped = swapCanonical(canonical);
    if (swapped === canonical) continue;
    const sourceName = sourceByCanonical.get(swapped);
    if (sourceName) names[bone.name] = sourceName;
  }

  let matched = 0;
  let canonicalCandidates = 0;
  const unmatchedSample = [];
  const unmatchedHumanoid = [];
  const sourceNameSet = new Set((sourceBones || []).map((bone) => bone.name));
  for (const bone of targetBones || []) {
    const canonical = canonicalBoneKey(bone.name);
    if (canonicalFilter && (!canonical || !canonicalFilter.has(canonical))) continue;
    if (canonical) canonicalCandidates += 1;
    if (names[bone.name] && sourceNameSet.has(names[bone.name])) {
      matched += 1;
    } else if (unmatchedSample.length < 30) {
      unmatchedSample.push({ target: bone.name, canonical: canonical || "n/a" });
      if (canonical && unmatchedHumanoid.length < 30) {
        unmatchedHumanoid.push({ target: bone.name, canonical });
      }
    }
  }

  return {
    names,
    matched,
    canonicalCandidates,
    unmatchedSample,
    unmatchedHumanoid,
    sourceMatched: new Set(Object.values(names).filter((name) => sourceNameSet.has(name))).size,
    mirroredSidesApplied: true,
  };
}

function maybeSwapArmSidesByChain(baseResult, targetBones, sourceBones, canonicalFilter) {
  if (!baseResult?.names) return { ...baseResult, mirroredArmSidesApplied: false };
  const targetMap = buildCanonicalBoneMap(targetBones || []);
  const sourceMap = buildCanonicalBoneMap(sourceBones || []);
  const armCanonicals = [
    "leftShoulder",
    "leftUpperArm",
    "leftLowerArm",
    "leftHand",
    "rightShoulder",
    "rightUpperArm",
    "rightLowerArm",
    "rightHand",
  ];
  const fingerPrefixes = ["Thumb", "Index", "Middle", "Ring", "Little"];
  for (const side of ["left", "right"]) {
    for (const finger of fingerPrefixes) {
      armCanonicals.push(`${side}${finger}Metacarpal`);
      armCanonicals.push(`${side}${finger}Proximal`);
      armCanonicals.push(`${side}${finger}Intermediate`);
      armCanonicals.push(`${side}${finger}Distal`);
    }
  }

  let assignmentSameVotes = 0;
  let assignmentSwappedVotes = 0;
  let majorChainSwappedVotes = 0;
  let majorChainTotal = 0;
  for (const bone of targetBones || []) {
    const canonical = canonicalBoneKey(bone.name) || "";
    if (!canonical || !armCanonicals.includes(canonical)) continue;
    if (canonicalFilter && !canonicalFilter.has(canonical)) continue;
    const mappedSourceName = baseResult.names?.[bone.name] || "";
    const mappedCanonical = canonicalBoneKey(mappedSourceName) || "";
    if (!mappedCanonical) continue;
    if (
      (canonical.startsWith("left") && mappedCanonical.startsWith("left")) ||
      (canonical.startsWith("right") && mappedCanonical.startsWith("right"))
    ) {
      assignmentSameVotes += 1;
    } else if (
      (canonical.startsWith("left") && mappedCanonical.startsWith("right")) ||
      (canonical.startsWith("right") && mappedCanonical.startsWith("left"))
    ) {
      assignmentSwappedVotes += 1;
      if (
        canonical.endsWith("UpperArm") ||
        canonical.endsWith("LowerArm") ||
        canonical.endsWith("Hand")
      ) {
        majorChainSwappedVotes += 1;
      }
    }
    if (
      canonical.endsWith("UpperArm") ||
      canonical.endsWith("LowerArm") ||
      canonical.endsWith("Hand")
    ) {
      majorChainTotal += 1;
    }
  }

  const segmentAngle = (a0, a1, b0, b1) => {
    if (!a0 || !a1 || !b0 || !b1) return null;
    const va = new THREE.Vector3().subVectors(a1, a0);
    const vb = new THREE.Vector3().subVectors(b1, b0);
    if (va.lengthSq() < 1e-10 || vb.lengthSq() < 1e-10) return null;
    return THREE.MathUtils.radToDeg(va.normalize().angleTo(vb.normalize()));
  };
  const scoreArmPair = (targetSide, sourceSide) => {
    const targetUpper = targetMap.get(`${targetSide}UpperArm`) || null;
    const targetLower = targetMap.get(`${targetSide}LowerArm`) || null;
    const targetHand = targetMap.get(`${targetSide}Hand`) || null;
    const sourceUpper = sourceMap.get(`${sourceSide}UpperArm`) || null;
    const sourceLower = sourceMap.get(`${sourceSide}LowerArm`) || null;
    const sourceHand = sourceMap.get(`${sourceSide}Hand`) || null;
    if (!targetUpper || !targetLower || !targetHand || !sourceUpper || !sourceLower || !sourceHand) return null;

    const t0 = new THREE.Vector3();
    const t1 = new THREE.Vector3();
    const t2 = new THREE.Vector3();
    const s0 = new THREE.Vector3();
    const s1 = new THREE.Vector3();
    const s2 = new THREE.Vector3();
    targetUpper.getWorldPosition(t0);
    targetLower.getWorldPosition(t1);
    targetHand.getWorldPosition(t2);
    sourceUpper.getWorldPosition(s0);
    sourceLower.getWorldPosition(s1);
    sourceHand.getWorldPosition(s2);

    const angles = [
      segmentAngle(t0, t1, s0, s1),
      segmentAngle(t1, t2, s1, s2),
    ].filter((value) => Number.isFinite(value));
    if (!angles.length) return null;
    return angles.reduce((sum, value) => sum + value, 0) / angles.length;
  };

  const sameScores = [
    scoreArmPair("left", "left"),
    scoreArmPair("right", "right"),
  ].filter((value) => Number.isFinite(value));
  const swappedScores = [
    scoreArmPair("left", "right"),
    scoreArmPair("right", "left"),
  ].filter((value) => Number.isFinite(value));
  const sameAvg = sameScores.length
    ? sameScores.reduce((sum, value) => sum + value, 0) / sameScores.length
    : null;
  const swappedAvg = swappedScores.length
    ? swappedScores.reduce((sum, value) => sum + value, 0) / swappedScores.length
    : null;
  const shouldSwapByAssignments =
    assignmentSwappedVotes >= 4 && assignmentSwappedVotes > assignmentSameVotes;
  const shouldSwapByMajorChain = majorChainTotal >= 6 && majorChainSwappedVotes >= 4;
  const shouldSwapByGeometry =
    Number.isFinite(sameAvg) &&
    Number.isFinite(swappedAvg) &&
    swappedAvg + 15 < sameAvg;
  if (!shouldSwapByAssignments && !shouldSwapByMajorChain && !shouldSwapByGeometry) {
    return {
      ...baseResult,
      mirroredArmSidesApplied: false,
      armSideSwapScore: {
        assignmentSameVotes,
        assignmentSwappedVotes,
        majorChainSwappedVotes,
        majorChainTotal,
        sameAvg: Number.isFinite(sameAvg) ? Number(sameAvg.toFixed(3)) : null,
        swappedAvg: Number.isFinite(swappedAvg) ? Number(swappedAvg.toFixed(3)) : null,
      },
    };
  }

  const sourceByCanonical = new Map();
  for (const bone of sourceBones || []) {
    const canonical = canonicalBoneKey(bone.name) || "";
    if (canonical) sourceByCanonical.set(canonical, bone.name);
  }
  const swapCanonical = (canonical) => {
    if (canonical.startsWith("left")) return `right${canonical.slice(4)}`;
    if (canonical.startsWith("right")) return `left${canonical.slice(5)}`;
    return canonical;
  };

  const names = { ...(baseResult.names || {}) };
  for (const bone of targetBones || []) {
    const canonical = canonicalBoneKey(bone.name) || "";
    if (!canonical || !armCanonicals.includes(canonical)) continue;
    if (canonicalFilter && !canonicalFilter.has(canonical)) continue;
    const swapped = swapCanonical(canonical);
    const sourceName = sourceByCanonical.get(swapped);
    if (sourceName) names[bone.name] = sourceName;
  }

  return {
    ...baseResult,
    names,
    mirroredArmSidesApplied: true,
    armSideSwapScore: {
      assignmentSameVotes,
      assignmentSwappedVotes,
      majorChainSwappedVotes,
      majorChainTotal,
      sameAvg: Number.isFinite(sameAvg) ? Number(sameAvg.toFixed(3)) : null,
      swappedAvg: Number.isFinite(swappedAvg) ? Number(swappedAvg.toFixed(3)) : null,
    },
  };
}

function quantizeFacingYaw(rad) {
  if (!Number.isFinite(rad)) return 0;
  const abs = Math.abs(rad);
  if (abs < THREE.MathUtils.degToRad(45)) return 0;
  if (abs > THREE.MathUtils.degToRad(120)) return Math.sign(rad || 1) * Math.PI;
  return rad;
}

function createRestOrientationCorrectionFactory(profile, recordRestCorrectionLog) {
  const calibV1 = new THREE.Vector3();
  const calibV2 = new THREE.Vector3();
  const calibV3 = new THREE.Vector3();
  const calibV4 = new THREE.Vector3();
  const calibQ1 = new THREE.Quaternion();
  const calibQ2 = new THREE.Quaternion();
  const calibM1 = new THREE.Matrix4();

  function getPrimaryChildDirectionLocal(bone, outDir) {
    if (!bone || !outDir) return false;
    bone.getWorldPosition(calibV1);
    let bestChild = null;
    let bestLenSq = 0;
    for (const child of bone.children || []) {
      if (!child?.isBone) continue;
      child.getWorldPosition(calibV2);
      const lenSq = calibV2.distanceToSquared(calibV1);
      if (lenSq > bestLenSq) {
        bestLenSq = lenSq;
        bestChild = child;
      }
    }
    if (!bestChild || bestLenSq < 1e-8) return false;
    bestChild.getWorldPosition(calibV2);
    outDir.copy(calibV2).sub(calibV1);
    bone.getWorldQuaternion(calibQ1);
    outDir.applyQuaternion(calibQ1.invert());
    if (outDir.lengthSq() < 1e-10) return false;
    outDir.normalize();
    return true;
  }

  function getReferenceDirectionLocal(bone, primaryDir, outDir) {
    if (!bone || !primaryDir || !outDir) return false;
    bone.getWorldPosition(calibV1);
    bone.getWorldQuaternion(calibQ1);
    const canonical = canonicalBoneKey(bone.name) || "";
    const invBoneQ = calibQ1.invert();
    const tryWorldVector = (worldVector) => {
      outDir.copy(worldVector).applyQuaternion(invBoneQ);
      outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
      if (outDir.lengthSq() < 1e-8) return false;
      outDir.normalize();
      return true;
    };

    if (canonical === "leftUpperLeg" || canonical === "rightUpperLeg") {
      const siblingCanonical = canonical === "leftUpperLeg" ? "rightUpperLeg" : "leftUpperLeg";
      for (const sibling of bone.parent?.children || []) {
        if (!sibling?.isBone || sibling === bone) continue;
        if ((canonicalBoneKey(sibling.name) || "") !== siblingCanonical) continue;
        sibling.getWorldPosition(calibV2);
        outDir.copy(calibV2).sub(calibV1).applyQuaternion(invBoneQ);
        outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
        if (outDir.lengthSq() >= 1e-8) {
          outDir.normalize();
          return true;
        }
      }
    }

    if (bone.parent?.isBone) {
      bone.parent.getWorldPosition(calibV2);
      outDir.copy(calibV2).sub(calibV1).applyQuaternion(invBoneQ);
      outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
      if (outDir.lengthSq() >= 1e-8) {
        outDir.normalize();
        return true;
      }
    }

    let bestChild = null;
    let bestLenSq = 0;
    for (const child of bone.children || []) {
      if (!child?.isBone) continue;
      child.getWorldPosition(calibV2);
      outDir.copy(calibV2).sub(calibV1);
      const lenSq = outDir.lengthSq();
      if (lenSq > bestLenSq) {
        bestLenSq = lenSq;
        bestChild = child;
      }
    }
    if (bestChild) {
      for (const child of bone.children || []) {
        if (!child?.isBone || child === bestChild) continue;
        child.getWorldPosition(calibV2);
        outDir.copy(calibV2).sub(calibV1).applyQuaternion(invBoneQ);
        outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
        if (outDir.lengthSq() >= 1e-8) {
          outDir.normalize();
          return true;
        }
      }
    }

    return (
      tryWorldVector(calibV2.set(1, 0, 0)) ||
      tryWorldVector(calibV2.set(0, 1, 0)) ||
      tryWorldVector(calibV2.set(0, 0, 1))
    );
  }

  function buildBoneLocalFrame(bone, outQ, options = null) {
    if (!bone || !outQ) return false;
    if (!getPrimaryChildDirectionLocal(bone, calibV1)) return false;
    if (!getReferenceDirectionLocal(bone, calibV1, calibV2)) return false;
    if (options?.flipReferenceAxis) {
      calibV2.multiplyScalar(-1);
    }
    calibV3.crossVectors(calibV2, calibV1);
    if (calibV3.lengthSq() < 1e-8) return false;
    calibV3.normalize();
    calibV4.crossVectors(calibV1, calibV3);
    if (calibV4.lengthSq() < 1e-8) return false;
    calibV4.normalize();
    calibM1.makeBasis(calibV4, calibV1, calibV3);
    outQ.setFromRotationMatrix(calibM1).normalize();
    return true;
  }

  return function buildRestOrientationCorrection(sourceBone, targetBone) {
    const canonical =
      canonicalBoneKey(targetBone?.name) ||
      canonicalBoneKey(sourceBone?.name) ||
      "";
    const flipReferenceAxis = !!(
      canonical &&
      profile?.flipReferenceAxisByCanonical?.[canonical]
    );
    const twistDeg = canonical
      ? Number(profile?.twistDegByCanonical?.[canonical] || 0)
      : 0;
    const offsetEntry = canonical
      ? profile?.restCorrectionEulerDegByCanonical?.[canonical] || null
      : null;
    const profileOffset =
      offsetEntry && (offsetEntry.x || offsetEntry.y || offsetEntry.z)
        ? new THREE.Quaternion().setFromEuler(
            new THREE.Euler(
              THREE.MathUtils.degToRad(Number(offsetEntry.x || 0)),
              THREE.MathUtils.degToRad(Number(offsetEntry.y || 0)),
              THREE.MathUtils.degToRad(Number(offsetEntry.z || 0)),
              "XYZ"
            )
          ).normalize()
        : null;
    const profileOffsetDeg = profileOffset
      ? {
          x: Number(Number(offsetEntry.x || 0).toFixed(2)),
          y: Number(Number(offsetEntry.y || 0).toFixed(2)),
          z: Number(Number(offsetEntry.z || 0).toFixed(2)),
        }
      : null;
    let twistQ = null;
    if (Math.abs(twistDeg) > 1e-4) {
      const twistAxis = new THREE.Vector3();
      if (getPrimaryChildDirectionLocal(targetBone, twistAxis)) {
        twistQ = new THREE.Quaternion()
          .setFromAxisAngle(twistAxis, THREE.MathUtils.degToRad(twistDeg))
          .normalize();
      }
    }
    const sourceFrame = calibQ1;
    const targetFrame = calibQ2;
    if (
      buildBoneLocalFrame(sourceBone, sourceFrame) &&
      buildBoneLocalFrame(targetBone, targetFrame, { flipReferenceAxis })
    ) {
      const corr = new THREE.Quaternion().copy(targetFrame).multiply(sourceFrame.invert()).normalize();
      const autoAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
      if (profileOffset) {
        corr.premultiply(profileOffset).normalize();
      }
      if (twistQ) {
        corr.premultiply(twistQ).normalize();
      }
      const finalAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
      recordRestCorrectionLog({
        canonical: canonical || "unknown",
        target: targetBone?.name || "",
        source: sourceBone?.name || "",
        method: "local-frame",
        autoAngleDeg,
        profileOffsetDeg,
        flipReferenceAxis,
        twistDeg: Number(twistDeg.toFixed(2)),
        finalAngleDeg,
      });
      if (Math.abs(corr.w) > 0.9995) {
        return twistQ || profileOffset || null;
      }
      return corr;
    }

    const sourceDir = new THREE.Vector3();
    const targetDir = new THREE.Vector3();
    if (!getPrimaryChildDirectionLocal(sourceBone, sourceDir)) return null;
    if (!getPrimaryChildDirectionLocal(targetBone, targetDir)) return null;
    const dot = Math.max(-1, Math.min(1, sourceDir.dot(targetDir)));
    if (dot > 0.9995) {
      recordRestCorrectionLog({
        canonical: canonical || "unknown",
        target: targetBone?.name || "",
        source: sourceBone?.name || "",
        method: "child-direction",
        autoAngleDeg: 0,
        profileOffsetDeg,
        flipReferenceAxis,
        twistDeg: Number(twistDeg.toFixed(2)),
        finalAngleDeg: profileOffset
          ? Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(profileOffset.w)))).toFixed(3))
          : 0,
      });
      if (twistQ && profileOffset) {
        return new THREE.Quaternion().copy(twistQ).premultiply(profileOffset).normalize();
      }
      return twistQ || profileOffset || null;
    }

    const corr = new THREE.Quaternion().setFromUnitVectors(sourceDir, targetDir).normalize();
    const autoAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
    if (profileOffset) {
      corr.premultiply(profileOffset).normalize();
    }
    if (twistQ) {
      corr.premultiply(twistQ).normalize();
    }
    const finalAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
    recordRestCorrectionLog({
      canonical: canonical || "unknown",
      target: targetBone?.name || "",
      source: sourceBone?.name || "",
      method: "child-direction",
      autoAngleDeg,
      profileOffsetDeg,
      flipReferenceAxis,
      twistDeg: Number(twistDeg.toFixed(2)),
      finalAngleDeg,
    });
    return corr;
  };
}

async function loadModel(modelPath) {
  const gltfLoader = new GLTFLoader();
  gltfLoader.register((parser) => new VRMLoaderPlugin(parser));
  const modelBuffer = await fs.readFile(modelPath);
  const arrayBuffer = modelBuffer.buffer.slice(
    modelBuffer.byteOffset,
    modelBuffer.byteOffset + modelBuffer.byteLength
  );
  const gltf = await new Promise((resolve, reject) =>
    gltfLoader.parse(arrayBuffer, "", resolve, reject)
  );
  const vrm = gltf.userData?.vrm || null;
  const modelRoot = vrm?.scene || gltf.scene || gltf.scenes?.[0] || null;
  if (!modelRoot) {
    throw new Error(`Failed to parse model: ${modelPath}`);
  }
  if (vrm) {
    if (typeof VRMUtils.rotateVRM0 === "function") {
      VRMUtils.rotateVRM0(vrm);
    }
    if (typeof VRMUtils.combineSkeletons === "function") {
      VRMUtils.combineSkeletons(modelRoot);
    } else if (typeof VRMUtils.removeUnnecessaryJoints === "function") {
      VRMUtils.removeUnnecessaryJoints(modelRoot);
    }
    modelRoot.rotation.y = Math.PI;
    modelRoot.updateMatrixWorld(true);
  }
  const vrmHumanoidInfo = applyVrmHumanoidBoneNames(vrm, VRM_HUMANOID_BONE_NAMES);
  const modelVrmHumanoidBones = vrmHumanoidInfo.bones || [];
  const modelVrmNormalizedHumanoidBones = vrmHumanoidInfo.normalizedBones || [];
  const modelSkinnedMeshes = findSkinnedMeshes(modelRoot);
  const modelSkinnedMesh = modelSkinnedMeshes[0] || null;
  if (!modelSkinnedMesh) {
    throw new Error(`Model has no skinned mesh: ${modelPath}`);
  }

  const seenBones = new Set();
  for (const mesh of modelSkinnedMeshes) {
    for (const bone of mesh.skeleton?.bones || []) {
      const id = bone.uuid || `${mesh.uuid}:${bone.name}`;
      if (seenBones.has(id)) continue;
      seenBones.add(id);
      bone.userData.__bindPosition = bone.position.clone();
    }
  }

  let totalMissingBefore = 0;
  let totalAutoNamed = 0;
  let totalInferredCanonical = 0;
  for (const mesh of modelSkinnedMeshes) {
    const namingInfo = autoNameTargetBones(mesh);
    totalMissingBefore += namingInfo.missingBefore;
    totalAutoNamed += namingInfo.autoNamed;
    totalInferredCanonical += namingInfo.inferredCanonical;
  }

  modelRoot.userData.__baseQuaternion = modelRoot.quaternion.clone();
  modelRoot.userData.__basePosition = modelRoot.position.clone();

  const rootBone = (() => {
    let node =
      (modelSkinnedMesh.skeleton?.bones || []).find((bone) => canonicalBoneKey(bone.name) === "hips") ||
      modelSkinnedMesh.skeleton?.bones?.[0] ||
      null;
    while (node?.parent?.isBone) {
      node = node.parent;
    }
    return node || null;
  })();
  if (rootBone && rootBone !== modelRoot) {
    rootBone.userData.__retargetBaseQuaternion = rootBone.quaternion.clone();
    rootBone.userData.__retargetBasePosition = rootBone.position.clone();
  }

  return {
    gltf,
    vrm,
    modelRoot,
    modelSkinnedMesh,
    modelSkinnedMeshes,
    modelVrmHumanoidBones,
    modelVrmNormalizedHumanoidBones,
    vrmHumanoidInfo,
    autoNaming: {
      missingBefore: totalMissingBefore,
      autoNamed: totalAutoNamed,
      inferredCanonical: totalInferredCanonical,
    },
  };
}

async function loadSourceBvh(bvhPath) {
  const loader = new BVHLoader();
  const text = await fs.readFile(bvhPath, "utf8");
  const sourceResult = loader.parse(text);
  const sourceRoot = sourceResult?.skeleton?.bones?.[0] || null;
  if (!sourceRoot) {
    throw new Error(`Failed to parse BVH: ${bvhPath}`);
  }
  const mixer = new THREE.AnimationMixer(sourceRoot);
  const action = mixer.clipAction(sourceResult.clip);
  action.reset();
  action.setEffectiveWeight(1);
  action.setEffectiveTimeScale(1);
  action.play();
  mixer.setTime(0);
  sourceRoot.updateMatrixWorld(true);
  return {
    sourceResult,
    sourceRoot,
    mixer,
    action,
  };
}

export async function runHeadlessRetargetValidation({
  modelPath,
  bvhPath,
  stage = "body",
} = {}) {
  installHeadlessNodeRuntime();

  const normalizedStage = String(stage || "body").trim().toLowerCase();
  if (!RETARGET_STAGES.has(normalizedStage)) {
    throw new Error(`Unsupported stage: ${stage}`);
  }

  const resolvedModelPath = path.resolve(String(modelPath || ""));
  const resolvedBvhPath = path.resolve(String(bvhPath || ""));
  const modelLabel = path.basename(resolvedModelPath);
  const bvhLabel = path.basename(resolvedBvhPath);
  const windowRef = {
    __vid2modelDiagMode: "minimal",
    localStorage: createMemoryStorage(),
  };

  const diagRecords = [];
  const diag = (event, payload = {}) => {
    diagRecords.push({
      event,
      payload: sanitizeForJson(payload),
    });
  };

  const modelState = await loadModel(resolvedModelPath);
  const sourceState = await loadSourceBvh(resolvedBvhPath);
  const modelRigFingerprint = buildModelRigFingerprint(
    modelState.modelSkinnedMeshes,
    modelLabel
  );

  diag("model-loaded", {
    file: modelLabel,
    skinnedMeshes: modelState.modelSkinnedMeshes.length,
    vrmHumanoid: modelState.vrmHumanoidInfo.applied
      ? {
          bones: modelState.vrmHumanoidInfo.applied,
          renamed: modelState.vrmHumanoidInfo.renamed,
          normalized: modelState.modelVrmNormalizedHumanoidBones.length,
        }
      : null,
    vrmDirectReady:
      (modelState.modelVrmNormalizedHumanoidBones.length || modelState.modelVrmHumanoidBones.length) > 0,
    topMeshes: modelState.modelSkinnedMeshes.slice(0, 3).map((mesh) => ({
      name: mesh.name || "(unnamed-skinned-mesh)",
      bones: mesh.skeleton.bones.length,
    })),
    autoNaming:
      modelState.autoNaming.missingBefore > 0
        ? modelState.autoNaming
        : null,
  });

  const modelAnalysisTools = createViewerModelAnalysisTools({
    windowRef,
    buildCanonicalBoneMap,
    canonicalBoneKey,
    getModelSkinnedMeshes: () => modelState.modelSkinnedMeshes,
    getModelLabel: () => modelLabel,
    getModelRigFingerprint: () => modelRigFingerprint,
    getModelSkeletonRootBone: () => {
      let node =
        (modelState.modelSkinnedMesh.skeleton?.bones || []).find((bone) => canonicalBoneKey(bone.name) === "hips") ||
        modelState.modelSkinnedMesh.skeleton?.bones?.[0] ||
        null;
      while (node?.parent?.isBone) {
        node = node.parent;
      }
      return node || null;
    },
    getModelVrmHumanoidBones: () => modelState.modelVrmHumanoidBones,
    getModelVrmNormalizedHumanoidBones: () => modelState.modelVrmNormalizedHumanoidBones,
  });

  const rigProfileService = createViewerRigProfileService({
    windowRef,
    storageKey: "vid2model.rigProfiles.headless",
    maxEntries: 12,
    statusValues: new Set(["draft", "validated"]),
    repoManifestUrl: "https://example.invalid/rig-profiles/index.json",
    getBuiltinRigProfile,
    getRetargetStage: () => normalizedStage,
    getCurrentModelRigFingerprint: () => modelRigFingerprint,
    buildRigProfileSeedForCurrentModel: (nextStage = "body") =>
      modelAnalysisTools.buildRigProfileSeedForCurrentModel(nextStage),
    buildSeedCorrectionSummary: (profile) =>
      modelAnalysisTools.buildSeedCorrectionSummary(profile),
  });

  const { buildRigProfileState, publishRigProfileState, loadRigProfile } = rigProfileService;
  const cachedRigProfile = loadRigProfile(modelRigFingerprint, normalizedStage, modelLabel);
  publishRigProfileState(
    buildRigProfileState(cachedRigProfile, {
      modelFingerprint: modelRigFingerprint,
      modelLabel,
      stage: normalizedStage,
      saved: false,
      resolvedFrom: cachedRigProfile?.source || "none",
    })
  );

  const canonicalFilter = resolveRetargetStageCanonicalFilter(normalizedStage, cachedRigProfile);
  const bodyMetricCanonicalFilter = resolveBodyMetricCanonicalFilter(normalizedStage, cachedRigProfile);
  const stageClip = buildStageSourceClip(
    sourceState.sourceResult.clip,
    sourceState.sourceResult.skeleton.bones,
    normalizedStage,
    canonicalFilter
  );
  if (!stageClip) {
    throw new Error(`No source tracks for stage "${normalizedStage}"`);
  }
  const activeStageClip =
    scaleClipRotationsByCanonical(
      stageClip,
      sourceState.sourceResult.skeleton.bones,
      cachedRigProfile?.rotationScaleByCanonical || null
    ) || stageClip;

  const retargetTargetBones = resolveRetargetTargetBones({
    stage: normalizedStage,
    modelSkinnedMesh: modelState.modelSkinnedMesh,
    modelVrmHumanoidBones: modelState.modelVrmHumanoidBones,
    modelVrmNormalizedHumanoidBones: modelState.modelVrmNormalizedHumanoidBones,
  }).filter((bone) => {
    if (!canonicalFilter) return true;
    return canonicalFilter.has(canonicalBoneKey(bone.name) || "");
  });
  const directRetargetTargetBones = resolveRetargetTargetBones({
    stage: normalizedStage,
    modelSkinnedMesh: modelState.modelSkinnedMesh,
    modelVrmHumanoidBones: modelState.modelVrmHumanoidBones,
    modelVrmNormalizedHumanoidBones: modelState.modelVrmNormalizedHumanoidBones,
    preferNormalized: true,
  }).filter((bone) => {
    if (!canonicalFilter) return true;
    return canonicalFilter.has(canonicalBoneKey(bone.name) || "");
  });

  const alignmentTools = createViewerAlignmentTools({
    canonicalBoneKey,
    diag,
    camera: {
      position: new THREE.Vector3(),
    },
    controls: {
      target: new THREE.Vector3(),
      update() {},
    },
    getSkeletonObj: () => sourceState.sourceRoot,
    getSourceResult: () => sourceState.sourceResult,
    getModelRoot: () => modelState.modelRoot,
    getModelSkinnedMesh: () => modelState.modelSkinnedMesh,
    hasSourceOverlay: () => false,
    setSourceOverlayYaw() {},
    updateSourceOverlay() {},
    estimateFacingYawOffset,
  });

  const chainDiagnostics = createViewerChainDiagnostics({
    canonicalBoneKey,
    buildCanonicalBoneMap,
    isVerboseDiagMode: () => false,
  });

  const runtimeDiagnostics = createViewerRuntimeDiagnostics({
    windowRef,
    diag,
    isVerboseDiagMode: () => false,
    collectLimbDiagnostics,
    dumpRetargetAlignmentDiagnostics,
    dumpRestCorrectionLog: chainDiagnostics.dumpRestCorrectionLog,
    buildLegChainDiagnostics: chainDiagnostics.buildLegChainDiagnostics,
    dumpLegChainDiagLog: chainDiagnostics.dumpLegChainDiagLog,
    buildArmChainDiagnostics: chainDiagnostics.buildArmChainDiagnostics,
    dumpArmChainDiagLog: chainDiagnostics.dumpArmChainDiagLog,
    buildTorsoChainDiagnostics: chainDiagnostics.buildTorsoChainDiagnostics,
    dumpTorsoChainDiagLog: chainDiagnostics.dumpTorsoChainDiagLog,
    buildFootChainDiagnostics: chainDiagnostics.buildFootChainDiagnostics,
    dumpFootChainDiagLog: chainDiagnostics.dumpFootChainDiagLog,
    dumpFootCorrectionDebug: chainDiagnostics.dumpFootCorrectionDebug,
    getSkeletonCanonicalRows: () => [],
    resolveSkeletonDumpCanonicals: () => [],
    getRetargetTargetBones: () => retargetTargetBones,
    getRetargetStage: () => normalizedStage,
    getSourceOverlay: () => null,
    overlayUpAxis: new THREE.Vector3(0, 1, 0),
  });

  chainDiagnostics.resetRestCorrectionLog();
  chainDiagnostics.resetLegChainDiagLog();
  chainDiagnostics.resetArmChainDiagLog();
  chainDiagnostics.resetTorsoChainDiagLog();
  chainDiagnostics.resetFootChainDiagLog();

  const buildRestOrientationCorrection = createRestOrientationCorrectionFactory(
    cachedRigProfile,
    chainDiagnostics.recordRestCorrectionLog
  );

  const initialMap = buildRetargetMap(
    retargetTargetBones,
    sourceState.sourceResult.skeleton.bones,
    { canonicalFilter }
  );
  const topologyFallback = maybeApplyTopologyFallback(
    modelState.modelSkinnedMesh,
    sourceState.sourceResult.skeleton.bones,
    canonicalFilter,
    initialMap
  );
  const profiledMapResult = applyRigProfileNames(
    topologyFallback.result,
    cachedRigProfile,
    retargetTargetBones,
    sourceState.sourceResult.skeleton.bones,
    canonicalFilter
  );
  const mirrorSwapMode = String(cachedRigProfile?.mirrorSwap || "").trim().toLowerCase();
  const allowMirrorSwap =
    mirrorSwapMode === "disable"
      ? false
      : mirrorSwapMode === "force" || mirrorSwapMode === "enable"
        ? true
        : true;
  const mirroredMapResult = allowMirrorSwap
    ? maybeSwapMirroredHumanoidSides(
        profiledMapResult,
        retargetTargetBones,
        sourceState.sourceResult.skeleton.bones,
        canonicalFilter
      )
    : { ...profiledMapResult, mirroredSidesApplied: false };
  const armSideSwapMode = String(cachedRigProfile?.armSideSwap || "").trim().toLowerCase();
  const allowArmSideSwap =
    armSideSwapMode === "disable"
      ? false
      : armSideSwapMode === "force" || armSideSwapMode === "enable"
        ? true
        : true;
  const activeMapResult = allowArmSideSwap
    ? maybeSwapArmSidesByChain(
        mirroredMapResult,
        retargetTargetBones,
        sourceState.sourceResult.skeleton.bones,
        canonicalFilter
      )
    : {
        ...mirroredMapResult,
        mirroredArmSidesApplied: false,
        armSideSwapScore: null,
      };

  const {
    names,
    matched,
    unmatchedSample,
    canonicalCandidates,
    unmatchedHumanoid,
    sourceMatched,
  } = activeMapResult;
  const mappedPairs = Object.keys(names).length;

  diag("retarget-input", {
    stage: normalizedStage,
    sourceBones: sourceState.sourceResult.skeleton.bones.length,
    targetBones: retargetTargetBones.length,
    sourceTracks: activeStageClip.tracks.length,
    mappedPairs,
    uniqueSourceMapped: sourceMatched,
    humanoidMatched: canonicalCandidates > 0 ? `${matched}/${canonicalCandidates}` : "n/a",
    mirrorSwap: allowMirrorSwap ? (mirrorSwapMode || "auto") : "disable",
    armSideSwap: allowArmSideSwap ? (armSideSwapMode || "auto") : "disable",
    mirroredSidesApplied: !!activeMapResult.mirroredSidesApplied,
    mirroredArmSidesApplied: !!activeMapResult.mirroredArmSidesApplied,
    armSideSwapScore: activeMapResult.armSideSwapScore || null,
  });
  diag("retarget-topology-fallback", {
    stage: normalizedStage,
    attempted: topologyFallback.attempted,
    applied: topologyFallback.applied,
    reason: topologyFallback.reason,
    inferredRenames: topologyFallback.inferredRenames || 0,
    before: topologyFallback.before || null,
    after: topologyFallback.after || null,
    sample: topologyFallback.sample || [],
  });

  const { retargetAttempts, attemptDebug } = collectRetargetAttempts({
    modelSkinnedMesh: modelState.modelSkinnedMesh,
    modelSkinnedMeshes: modelState.modelSkinnedMeshes,
    modelRoot: modelState.modelRoot,
    sourceResult: sourceState.sourceResult,
    activeStageClip,
    names,
    SkeletonUtils,
    buildRenamedClip,
    resolvedTrackCountAcrossMeshes,
    resolvedTrackCountForTarget,
    shortErr,
  });

  if (!retargetAttempts.length) {
    diag("retarget-fail", {
      reason: "no_tracks",
      stage: normalizedStage,
      unmatched: (unmatchedHumanoid.length ? unmatchedHumanoid : unmatchedSample).slice(0, 8),
    });
    throw new Error("Retarget failed: no tracks produced.");
  }

  let {
    selectedAttempt,
    selectedBindings,
    selectedProbe,
    selectedPoseError,
    selectionDebug,
    preferredMode,
  } = selectRetargetAttempt({
    retargetAttempts,
    cachedRigProfile,
    preferSkeletonOnRenameFallback: !!cachedRigProfile?.preferSkeletonOnRenameFallback,
    modelSkinnedMeshes: modelState.modelSkinnedMeshes,
    modelRoot: modelState.modelRoot,
    modelSkinnedMesh: modelState.modelSkinnedMesh,
    retargetTargetBones,
    sourceResult: sourceState.sourceResult,
    mixer: sourceState.mixer,
    buildBindingsForAttempt,
    clipUsesBonesSyntax,
    resolvedTrackCountForTarget,
    probeMotionForBindings,
    computePoseMatchError,
    buildCanonicalBoneMap,
    canonicalPoseSignature,
    attemptPriority: (label) => {
      if (label === "skeletonutils-skinnedmesh") return 40;
      if (label === "skeletonutils-skinnedmesh-reversed") return 30;
      if (label === "rename-fallback-bones") return 20;
      if (label === "rename-fallback-object") return 10;
      if (label === "skeletonutils-root") return 5;
      if (label === "skeletonutils-root-reversed") return 4;
      return 0;
    },
  });

  if (!selectedBindings || !selectedBindings.mixers.length) {
    diag("retarget-fail", { reason: "no_resolved_tracks", stage: normalizedStage });
    throw new Error("Retarget failed: clip has no resolved tracks on model skeleton.");
  }

  let modelMixers = selectedBindings.mixers;
  let modelActions = selectedBindings.actions;
  let modelMixer = modelMixers[0] || null;
  let modelAction = modelActions[0] || null;
  let liveRetarget = null;
  const clip = selectedAttempt.clip;
  let selectedModeLabel = selectedAttempt.label;

  const isRenameFallback = selectedAttempt.label.startsWith("rename-fallback");
  const rawFacingYaw = estimateFacingYawOffset(
    sourceState.sourceResult.skeleton.bones,
    retargetTargetBones
  );
  const strongFacingMismatch = Math.abs(rawFacingYaw) > THREE.MathUtils.degToRad(100);
  const weakMotion = !!selectedProbe && selectedProbe.score < 0.5;
  const highPoseError = Number.isFinite(selectedPoseError) && selectedPoseError > 0.6;
  const selectedIsSkeletonUtils = selectedAttempt.label.startsWith("skeletonutils");
  const fullHumanoidMatch = canonicalCandidates > 0 && matched === canonicalCandidates;
  const severeFacingPoseMismatch = strongFacingMismatch && highPoseError;
  const autoUseLiveDelta =
    isRenameFallback ||
    (
      selectedIsSkeletonUtils &&
      !!cachedRigProfile?.preferAggressiveLiveDelta &&
      (severeFacingPoseMismatch || (!fullHumanoidMatch && (highPoseError || strongFacingMismatch)))
    ) ||
    (!selectedIsSkeletonUtils && (strongFacingMismatch || weakMotion || highPoseError));
  const profileForceLiveDelta =
    typeof cachedRigProfile?.forceLiveDelta === "boolean" ? cachedRigProfile.forceLiveDelta : null;
  const useLiveDelta =
    profileForceLiveDelta === null ? autoUseLiveDelta : profileForceLiveDelta;

  diag("retarget-live-delta", {
    stage: normalizedStage,
    selectedMode: selectedAttempt.label,
    selectedIsSkeletonUtils,
    profilePolicy: {
      preferSkeletonOnRenameFallback: !!cachedRigProfile?.preferSkeletonOnRenameFallback,
      preferAggressiveLiveDelta: !!cachedRigProfile?.preferAggressiveLiveDelta,
    },
    fullHumanoidMatch,
    preferredMode: preferredMode || null,
    autoUseLiveDelta,
    profileForced: profileForceLiveDelta,
    forced: null,
    useLiveDelta,
    reasons: {
      isRenameFallback,
      strongFacingMismatch,
      weakMotion,
      highPoseError,
      severeFacingPoseMismatch,
    },
  });

  let rootYawCorrection = 0;
  const preferVrmDirectBody =
    normalizedStage === "body" &&
    directRetargetTargetBones.length > 0 &&
    cachedRigProfile?.preferVrmDirectBody === true;

  if (preferVrmDirectBody) {
    const directPlan = buildVrmDirectBodyPlan({
      targetBones: directRetargetTargetBones,
      sourceBones: sourceState.sourceResult.skeleton.bones,
      namesTargetToSource: names,
      mixer: sourceState.mixer,
      modelRoot: modelState.modelRoot,
      buildRestOrientationCorrection,
      profile: cachedRigProfile,
    });
    if (directPlan && directPlan.pairs.length > 0) {
      liveRetarget = directPlan;
      applyLiveRetargetPose({ plan: liveRetarget, modelRoot: modelState.modelRoot });
      modelMixers = [];
      modelActions = [];
      modelMixer = null;
      modelAction = null;
      selectedModeLabel = "vrm-humanoid-direct";
    }
  }

  if (!liveRetarget && selectedIsSkeletonUtils && !useLiveDelta) {
    const profileRootYawDeg =
      Number.isFinite(cachedRigProfile?.rootYawDeg) ? cachedRigProfile.rootYawDeg : null;
    const sourceClipYawSummary = summarizeSourceRootYawClip(clip);
    const sourceClipYawLooksCentered = !!sourceClipYawSummary.looksCentered;
    let zeroRow = null;
    let bestRow = null;
    let shouldUseBest = false;
    let yawEval = { rows: [], bestYaw: 0 };
    const rawYawLooksAligned = Math.abs(rawFacingYaw) < THREE.MathUtils.degToRad(30);
    if (Number.isFinite(profileRootYawDeg)) {
      rootYawCorrection = applyModelRootYaw({
        modelRoot: modelState.modelRoot,
        yawRad: THREE.MathUtils.degToRad(profileRootYawDeg),
        axisY: new THREE.Vector3(0, 1, 0),
      });
    } else {
      const yawCandidates = buildRootYawCandidates(rawFacingYaw, quantizeFacingYaw, {
        sourceClipYawSummary,
      });
      yawEval = evaluateRootYawCandidates({
        candidates: yawCandidates,
        sampleTime: selectedProbe?.sampleTime || 0,
        namesTargetToSource: names,
        sourceClip: clip,
        modelRoot: modelState.modelRoot,
        modelMixers,
        modelSkinnedMesh: modelState.modelSkinnedMesh,
        targetBones: retargetTargetBones,
        sourceResult: sourceState.sourceResult,
        mixer: sourceState.mixer,
        resetModelRootOrientation: () =>
          resetModelRootOrientation({
            modelRoot: modelState.modelRoot,
            getModelSkeletonRootBone: alignmentTools.getModelSkeletonRootBone,
          }),
        applyModelRootYaw: (yawRad) =>
          applyModelRootYaw({
            modelRoot: modelState.modelRoot,
            yawRad,
            axisY: new THREE.Vector3(0, 1, 0),
          }),
        collectAlignmentDiagnostics,
      });
      zeroRow = yawEval.rows.find((row) => Math.abs(row.yawDeg) < 0.01) || null;
      bestRow = yawEval.rows[0] || null;
      const bestIsLargeFlip = !!bestRow && Math.abs(bestRow.yawDeg) > 120;
      const largeFlipLooksRedundant =
        !!bestRow &&
        !!zeroRow &&
        sourceClipYawLooksCentered &&
        bestIsLargeFlip &&
        bestRow.score + 0.08 >= zeroRow.score;
      shouldUseBest =
        !!bestRow &&
        !(rawYawLooksAligned && bestIsLargeFlip) &&
        !largeFlipLooksRedundant &&
        (
          !zeroRow ||
          bestRow.score + 0.03 < zeroRow.score ||
          (Number.isFinite(bestRow.hipsPosErr) && Number.isFinite(zeroRow.hipsPosErr) && bestRow.hipsPosErr + 0.03 < zeroRow.hipsPosErr)
        );
      rootYawCorrection = applyModelRootYaw({
        modelRoot: modelState.modelRoot,
        yawRad: shouldUseBest ? yawEval.bestYaw : 0,
        axisY: new THREE.Vector3(0, 1, 0),
      });
    }

    const hipsYawError = computeHipsYawError(
      retargetTargetBones,
      sourceState.sourceResult.skeleton.bones,
      names
    );
    let hipsYawCorrection = 0;
    let hipsCorrectionApplied = false;
    let hipsCorrectionEval = null;
    const hipsCorrectionWouldLargeFlip = Math.abs(hipsYawError) > THREE.MathUtils.degToRad(120);
    if (
      Math.abs(hipsYawError) > THREE.MathUtils.degToRad(12) &&
      !(Math.abs(rawFacingYaw) < THREE.MathUtils.degToRad(30) && hipsCorrectionWouldLargeFlip)
    ) {
      const correctedYaw = rootYawCorrection - hipsYawError;
      const postEval = evaluateRootYawCandidates({
        candidates: [rootYawCorrection, correctedYaw],
        sampleTime: selectedProbe?.sampleTime || 0,
        namesTargetToSource: names,
        sourceClip: clip,
        modelRoot: modelState.modelRoot,
        modelMixers,
        modelSkinnedMesh: modelState.modelSkinnedMesh,
        targetBones: retargetTargetBones,
        sourceResult: sourceState.sourceResult,
        mixer: sourceState.mixer,
        resetModelRootOrientation: () =>
          resetModelRootOrientation({
            modelRoot: modelState.modelRoot,
            getModelSkeletonRootBone: alignmentTools.getModelSkeletonRootBone,
          }),
        applyModelRootYaw: (yawRad) =>
          applyModelRootYaw({
            modelRoot: modelState.modelRoot,
            yawRad,
            axisY: new THREE.Vector3(0, 1, 0),
          }),
        collectAlignmentDiagnostics,
      });
      const sameYaw = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b))) < 1e-4;
      const currentRow = postEval.rows.find((row) => sameYaw(row.yawRad, rootYawCorrection)) || null;
      const correctedRow = postEval.rows.find((row) => sameYaw(row.yawRad, correctedYaw)) || null;
      hipsCorrectionEval = { current: currentRow, corrected: correctedRow };
      const shouldApplyCorrection =
        !!currentRow &&
        !!correctedRow &&
        correctedRow.score + 0.03 < currentRow.score;
      if (shouldApplyCorrection) {
        hipsYawCorrection = -hipsYawError;
        rootYawCorrection = applyModelRootYaw({
          modelRoot: modelState.modelRoot,
          yawRad: correctedYaw,
          axisY: new THREE.Vector3(0, 1, 0),
        });
        hipsCorrectionApplied = true;
      }
    }
    diag("retarget-root-yaw", {
      stage: normalizedStage,
      rawFacingYawDeg: Number((rawFacingYaw * 180 / Math.PI).toFixed(2)),
      appliedYawDeg: Number((rootYawCorrection * 180 / Math.PI).toFixed(2)),
      hipsYawErrorDeg: Number((hipsYawError * 180 / Math.PI).toFixed(2)),
      hipsYawCorrectionDeg: Number((hipsYawCorrection * 180 / Math.PI).toFixed(2)),
      hipsCorrectionApplied,
      hipsCorrectionEval,
      strongFacingMismatch,
      usedBestCandidate: shouldUseBest,
      usedProfileYaw: Number.isFinite(profileRootYawDeg),
      sourceClipYawSummary,
      sourceYawCandidatePolicy: {
        allowSourceFlipCandidates: !sourceClipYawLooksCentered,
      },
      zeroCandidate: zeroRow,
      bestCandidate: bestRow,
      candidates: yawEval.rows,
    });
  }

  const rebuildLiveRetargetPlan = () =>
    buildLiveRetargetPlan({
      skinnedMeshes: modelState.modelSkinnedMeshes,
      targetBones: retargetTargetBones,
      sourceBones: sourceState.sourceResult.skeleton.bones,
      namesTargetToSource: names,
      mixer: sourceState.mixer,
      modelRoot: modelState.modelRoot,
      buildRestOrientationCorrection,
      cachedProfile: cachedRigProfile?.liveRetarget || null,
      profile: cachedRigProfile,
    });

  if (!liveRetarget && useLiveDelta) {
    const livePlan = rebuildLiveRetargetPlan();
    if (livePlan && livePlan.pairs.length > 0) {
      liveRetarget = livePlan;
      applyLiveRetargetPose({
        plan: liveRetarget,
        modelRoot: modelState.modelRoot,
        liveAxisY: new THREE.Vector3(0, 1, 0),
      });
      modelMixers = [];
      modelActions = [];
      modelMixer = null;
      modelAction = null;
      selectedModeLabel = `${selectedAttempt.label}+live-delta`;
    }
  }

  const sourceTime = sourceState.mixer ? sourceState.mixer.time : 0;
  const syncTime =
    clip.duration > 0
      ? ((sourceTime % clip.duration) + clip.duration) % clip.duration
      : 0;
  if (!liveRetarget) {
    for (const mix of modelMixers) {
      mix.setTime(syncTime);
    }
  } else {
    sourceState.mixer.setTime(syncTime);
    applyLiveRetargetPose({
      plan: liveRetarget,
      modelRoot: modelState.modelRoot,
      liveAxisY: new THREE.Vector3(0, 1, 0),
    });
  }

  let bodyLengthCalibration = null;
  let armLengthCalibration = null;
  let fingerLengthCalibration = null;
  let measureBodyErr = null;
  let bodyErrBaseline = null;

  const updateModelWorld = () => modelState.modelRoot?.updateMatrixWorld(true);

  if (liveRetarget) {
    const bodyEvalCanonical = bodyMetricCanonicalFilter;
    const bodyTargetBones = retargetTargetBones.filter((bone) =>
      bodyEvalCanonical.has(canonicalBoneKey(bone.name) || "")
    );
    measureBodyErr = () => {
      const report = collectAlignmentDiagnostics({
        targetBones: bodyTargetBones.length ? bodyTargetBones : retargetTargetBones,
        sourceBones: sourceState.sourceResult.skeleton.bones,
        namesTargetToSource: names,
        sourceClip: clip,
        maxRows: 5,
      });
      return Number.isFinite(report?.avgPosErrNorm) ? report.avgPosErrNorm : report?.avgPosErr;
    };
    const bodyErrBefore = measureBodyErr();
    if (Number.isFinite(bodyErrBefore)) {
      bodyErrBaseline = bodyErrBefore;
    }
    const attemptedBodyCalibration = buildBodyLengthCalibration(
      sourceState.sourceResult.skeleton.bones,
      modelState.modelSkinnedMesh.skeleton.bones,
      clip,
      buildCanonicalBoneMap
    );
    const filteredBodyCalibration =
      attemptedBodyCalibration && normalizedStage === "body"
        ? {
            ...attemptedBodyCalibration,
            entries: attemptedBodyCalibration.entries.filter((entry) =>
              bodyMetricCanonicalFilter.has(entry.canonical)
            ),
          }
        : attemptedBodyCalibration;
    const suspiciousBodyScale =
      !!filteredBodyCalibration &&
      Number.isFinite(filteredBodyCalibration.globalScale) &&
      (filteredBodyCalibration.globalScale < 0.2 || filteredBodyCalibration.globalScale > 5);
    if (filteredBodyCalibration?.entries?.length && !suspiciousBodyScale) {
      const previousLiveRetarget = liveRetarget;
      applyBoneLengthCalibration(filteredBodyCalibration, modelState.modelRoot);
      const rebuiltLiveRetarget = rebuildLiveRetargetPlan();
      if (rebuiltLiveRetarget?.pairs?.length) {
        liveRetarget = rebuiltLiveRetarget;
        applyLiveRetargetPose({
          plan: liveRetarget,
          modelRoot: modelState.modelRoot,
          liveAxisY: new THREE.Vector3(0, 1, 0),
        });
      }
      const bodyErrAfter = rebuiltLiveRetarget?.pairs?.length ? measureBodyErr() : null;
      const hasBefore = Number.isFinite(bodyErrBefore);
      const hasAfter = Number.isFinite(bodyErrAfter);
      const keepCalibration =
        !!rebuiltLiveRetarget?.pairs?.length &&
        hasAfter &&
        (!hasBefore || bodyErrAfter <= bodyErrBefore - 0.005);
      if (!keepCalibration) {
        resetBoneLengthCalibration(filteredBodyCalibration, modelState.modelRoot);
        liveRetarget = previousLiveRetarget;
        applyLiveRetargetPose({
          plan: liveRetarget,
          modelRoot: modelState.modelRoot,
          liveAxisY: new THREE.Vector3(0, 1, 0),
        });
        bodyLengthCalibration = null;
      } else {
        bodyLengthCalibration = filteredBodyCalibration;
        if (Number.isFinite(bodyErrAfter)) {
          bodyErrBaseline = bodyErrAfter;
        }
        selectedModeLabel = `${selectedModeLabel}+body-calib`;
      }
      diag("retarget-body-calibration", {
        stage: normalizedStage,
        mode: "live-delta",
        applied: keepCalibration,
        bodyErrBefore: Number.isFinite(bodyErrBefore) ? Number(bodyErrBefore.toFixed(5)) : null,
        bodyErrAfter: Number.isFinite(bodyErrAfter) ? Number(bodyErrAfter.toFixed(5)) : null,
        metric: normalizedStage === "body" ? "body-core" : "body-full",
        bones: filteredBodyCalibration.entries.length,
        globalScale: filteredBodyCalibration.globalScale,
        clampedCount: filteredBodyCalibration.clampedCount,
        sample: filteredBodyCalibration.entries.slice(0, 8).map((entry) => ({
          canonical: entry.canonical,
          scale: entry.scale,
          rawScale: entry.rawScale,
          sourceLen: entry.sourceLen,
          targetLen: entry.targetLen,
          expectedTargetLen: entry.expectedTargetLen,
        })),
      });
    }
  } else {
    const bodyEvalCanonical = bodyMetricCanonicalFilter;
    const bodyTargetBones = retargetTargetBones.filter((bone) =>
      bodyEvalCanonical.has(canonicalBoneKey(bone.name) || "")
    );
    measureBodyErr = () => {
      const report = collectAlignmentDiagnostics({
        targetBones: bodyTargetBones,
        sourceBones: sourceState.sourceResult.skeleton.bones,
        namesTargetToSource: names,
        sourceClip: clip,
        maxRows: 5,
      });
      return Number.isFinite(report?.avgPosErrNorm) ? report.avgPosErrNorm : report?.avgPosErr;
    };
    const bodyErrBefore = measureBodyErr();
    if (Number.isFinite(bodyErrBefore)) {
      bodyErrBaseline = bodyErrBefore;
    }
    const attemptedBodyCalibration = buildBodyLengthCalibration(
      sourceState.sourceResult.skeleton.bones,
      modelState.modelSkinnedMesh.skeleton.bones,
      clip,
      buildCanonicalBoneMap
    );
    const filteredBodyCalibration =
      attemptedBodyCalibration && normalizedStage === "body"
        ? {
            ...attemptedBodyCalibration,
            entries: attemptedBodyCalibration.entries.filter((entry) =>
              bodyMetricCanonicalFilter.has(entry.canonical)
            ),
          }
        : attemptedBodyCalibration;
    const suspiciousBodyScale =
      !!filteredBodyCalibration &&
      Number.isFinite(filteredBodyCalibration.globalScale) &&
      (filteredBodyCalibration.globalScale < 0.2 || filteredBodyCalibration.globalScale > 5);
    if (filteredBodyCalibration?.entries?.length && !suspiciousBodyScale) {
      applyBoneLengthCalibration(filteredBodyCalibration, modelState.modelRoot);
      const bodyErrAfter = measureBodyErr();
      const hasBefore = Number.isFinite(bodyErrBefore);
      const hasAfter = Number.isFinite(bodyErrAfter);
      const keepCalibration =
        hasAfter && (!hasBefore || bodyErrAfter <= bodyErrBefore - 0.005);
      if (!keepCalibration) {
        resetBoneLengthCalibration(filteredBodyCalibration, modelState.modelRoot);
        bodyLengthCalibration = null;
      } else {
        bodyLengthCalibration = filteredBodyCalibration;
        if (Number.isFinite(bodyErrAfter)) {
          bodyErrBaseline = bodyErrAfter;
        }
      }
      diag("retarget-body-calibration", {
        stage: normalizedStage,
        applied: keepCalibration,
        bodyErrBefore: Number.isFinite(bodyErrBefore) ? Number(bodyErrBefore.toFixed(5)) : null,
        bodyErrAfter: Number.isFinite(bodyErrAfter) ? Number(bodyErrAfter.toFixed(5)) : null,
        metric: normalizedStage === "body" ? "body-core" : "body-full",
        bones: filteredBodyCalibration.entries.length,
        globalScale: filteredBodyCalibration.globalScale,
        clampedCount: filteredBodyCalibration.clampedCount,
        sample: filteredBodyCalibration.entries.slice(0, 8).map((entry) => ({
          canonical: entry.canonical,
          scale: entry.scale,
          rawScale: entry.rawScale,
          sourceLen: entry.sourceLen,
          targetLen: entry.targetLen,
          expectedTargetLen: entry.expectedTargetLen,
        })),
      });
    }
  }

  if (!liveRetarget && normalizedStage === "body") {
    const armBaselineSnapshot = snapshotCanonicalBonePositions(
      modelState.modelSkinnedMesh.skeleton.bones,
      ARM_REFINEMENT_CANONICAL
    );
    const armRefine = buildArmRefinementCalibration({
      sourceBones: sourceState.sourceResult.skeleton.bones,
      targetBones: modelState.modelSkinnedMesh.skeleton.bones,
      namesTargetToSource: names,
      sourceClip: clip,
      buildCanonicalBoneMap,
      collectAlignmentDiagnostics,
      updateWorld: () => updateModelWorld(),
    });
    const armErrBefore = Number.isFinite(bodyErrBaseline)
      ? bodyErrBaseline
      : (measureBodyErr ? measureBodyErr() : null);
    let armErrAfter = armErrBefore;
    let keepArmRefine = false;
    if (armRefine?.entries?.length) {
      armLengthCalibration = { entries: armRefine.entries };
      applyBoneLengthCalibration(armLengthCalibration, modelState.modelRoot);
      armErrAfter = measureBodyErr ? measureBodyErr() : null;
      const hasBefore = Number.isFinite(armErrBefore);
      const hasAfter = Number.isFinite(armErrAfter);
      keepArmRefine = hasAfter && (!hasBefore || armErrAfter <= armErrBefore - 0.003);
      if (!keepArmRefine) {
        restoreBonePositionSnapshot(armBaselineSnapshot, modelState.modelRoot);
        armLengthCalibration = null;
      } else {
        bodyErrBaseline = armErrAfter;
      }
    }
    diag("retarget-arm-refine", {
      stage: normalizedStage,
      applied: keepArmRefine,
      bodyErrBefore: Number.isFinite(armErrBefore) ? Number(armErrBefore.toFixed(5)) : null,
      bodyErrAfter: Number.isFinite(armErrAfter) ? Number(armErrAfter.toFixed(5)) : null,
      appliedSides: (armRefine?.sides || []).filter((side) => side.applied).length,
      sides: armRefine?.sides || [],
      bones: armRefine?.entries?.length || 0,
    });
  }

  if (!liveRetarget && normalizedStage === "full") {
    fingerLengthCalibration = buildFingerLengthCalibration(
      sourceState.sourceResult.skeleton.bones,
      modelState.modelSkinnedMesh.skeleton.bones,
      clip,
      buildCanonicalBoneMap
    );
    if (fingerLengthCalibration) {
      applyFingerLengthCalibration(fingerLengthCalibration, modelState.modelRoot);
      diag("retarget-finger-calibration", {
        stage: normalizedStage,
        bones: fingerLengthCalibration.entries.length,
        globalScale: fingerLengthCalibration.globalScale,
        clampedCount: fingerLengthCalibration.clampedCount,
        rawScaleRange: {
          min: fingerLengthCalibration.minRawScale,
          max: fingerLengthCalibration.maxRawScale,
        },
        sample: fingerLengthCalibration.entries.slice(0, 8).map((entry) => ({
          canonical: entry.canonical,
          scale: entry.scale,
          rawScale: entry.rawScale,
          sourceLen: entry.sourceLen,
          targetLen: entry.targetLen,
          expectedTargetLen: entry.expectedTargetLen,
        })),
      });
    }
  }

  const hipsAlign = alignmentTools.alignModelHipsToSource(false);
  if (hipsAlign) {
    diag("retarget-hips-align", { stage: normalizedStage, ...hipsAlign });
  }

  const summaryTargetBones = retargetTargetBones.filter((bone) => {
    const canonical = canonicalBoneKey(bone.name) || "";
    if (!canonical) return false;
    return canonicalFilter ? canonicalFilter.has(canonical) : true;
  });
  const postRetargetReport = collectAlignmentDiagnostics({
    targetBones: summaryTargetBones.length ? summaryTargetBones : retargetTargetBones,
    sourceBones: sourceState.sourceResult.skeleton.bones,
    namesTargetToSource: names,
    sourceClip: clip,
    maxRows: 5,
  });
  const postRetargetPoseError =
    Number.isFinite(postRetargetReport?.avgPosErrNorm)
      ? postRetargetReport.avgPosErrNorm
      : postRetargetReport?.avgPosErr;
  const lowerBodyTargetBones = retargetTargetBones.filter((bone) =>
    bodyMetricCanonicalFilter.has(canonicalBoneKey(bone.name) || "")
  );
  const lowerBodyReport = collectAlignmentDiagnostics({
    targetBones: lowerBodyTargetBones.length ? lowerBodyTargetBones : retargetTargetBones,
    sourceBones: sourceState.sourceResult.skeleton.bones,
    namesTargetToSource: names,
    sourceClip: clip,
    maxRows: 5,
  });
  const lowerBodyPostError =
    Number.isFinite(lowerBodyReport?.avgPosErrNorm)
      ? lowerBodyReport.avgPosErrNorm
      : lowerBodyReport?.avgPosErr;
  const lowerBodyRotError =
    Number.isFinite(lowerBodyReport?.avgRotErrDeg) ? lowerBodyReport.avgRotErrDeg : null;

  const activeProfile = loadRigProfile(modelRigFingerprint, normalizedStage, modelLabel) || cachedRigProfile || null;
  publishRigProfileState(
    buildRigProfileState(activeProfile, {
      modelFingerprint: modelRigFingerprint,
      modelLabel,
      stage: normalizedStage,
      saved: false,
      resolvedFrom: activeProfile?.source || "none",
    })
  );

  const originalConsoleLog = console.log;
  const originalConsoleTable = console.table;
  try {
    console.log = (...args) => {
      if (typeof args[0] === "string" && args[0].startsWith("[vid2model/diag]")) {
        return;
      }
      originalConsoleLog(...args);
    };
    console.table = () => {};
    runtimeDiagnostics.publishRetargetDiagnostics({
      retargetStage: normalizedStage,
      names,
      sourceBones: sourceState.sourceResult.skeleton.bones,
      targetBones: retargetTargetBones,
      clip,
      selectedAttempt,
      selectedModeLabel,
      selectedProbe,
      selectedPoseError,
      liveRetarget,
      activeMeshCount: modelState.modelSkinnedMeshes.length,
      attemptDebug,
      selectionDebug,
      cachedRigProfile: activeProfile,
      rigProfileSaved: false,
      mappedPairs,
      sourceMatched,
      matched,
      canonicalCandidates,
      unmatchedHumanoid,
      unmatchedTargetBones: retargetTargetBones.filter((bone) => !names[bone.name]).map((bone) => bone.name),
      postRetargetPoseError,
      lowerBodyPostError,
      lowerBodyRotError,
      rootYawCorrection,
      modelSkinnedMesh: modelState.modelSkinnedMesh,
      sourceResult: sourceState.sourceResult,
    });
  } finally {
    console.log = originalConsoleLog;
    console.table = originalConsoleTable;
  }

  return {
    format: "vid2model.headless-retarget.v1",
    generatedAt: new Date().toISOString(),
    input: {
      stage: normalizedStage,
      modelPath: resolvedModelPath,
      bvhPath: resolvedBvhPath,
    },
    model: {
      label: modelLabel,
      rigFingerprint: modelRigFingerprint,
      isVrm: !!modelState.vrm,
      skinnedMeshes: modelState.modelSkinnedMeshes.length,
      primaryMesh: modelState.modelSkinnedMesh.name || "(unnamed-skinned-mesh)",
      vrmHumanoidApplied: modelState.vrmHumanoidInfo.applied || 0,
      vrmNormalizedBones: modelState.modelVrmNormalizedHumanoidBones.length,
      autoNaming: modelState.autoNaming,
      boneRows: collectModelBoneRows(modelState.modelSkinnedMeshes),
    },
    source: {
      label: bvhLabel,
      bones: sourceState.sourceResult.skeleton.bones.length,
      tracks: sourceState.sourceResult.clip.tracks.length,
      duration: Number(sourceState.sourceResult.clip.duration.toFixed(6)),
    },
    rigProfile: activeProfile
      ? {
          id: activeProfile.id || "",
          source: activeProfile.source || "none",
          validationStatus: activeProfile.validationStatus || "none",
          preferredMode: activeProfile.preferredMode || null,
          basedOnBuiltin: !!activeProfile.basedOnBuiltin,
          lockBuiltin: !!activeProfile.lockBuiltin,
          bodyCanonicalKeys: Array.isArray(activeProfile.bodyCanonicalKeys)
            ? [...activeProfile.bodyCanonicalKeys]
            : null,
        }
      : null,
    mapping: {
      mappedPairs,
      matched,
      sourceMatched,
      canonicalCandidates,
      unmatchedHumanoid,
      unmatchedSample,
      topologyFallback: sanitizeForJson(topologyFallback),
      mirroredSidesApplied: !!activeMapResult.mirroredSidesApplied,
      mirroredArmSidesApplied: !!activeMapResult.mirroredArmSidesApplied,
      armSideSwapScore: activeMapResult.armSideSwapScore || null,
    },
    selection: {
      selectedAttempt: selectedAttempt?.label || null,
      selectedModeLabel,
      preferredMode: preferredMode || null,
      useLiveDelta,
      liveRetarget: !!liveRetarget,
      rootYawDeg: Number((rootYawCorrection * 180 / Math.PI).toFixed(2)),
      probe: selectedProbe
        ? {
            sampleTime: Number(selectedProbe.sampleTime.toFixed(6)),
            maxAngle: Number(selectedProbe.maxAngle.toFixed(6)),
            maxPos: Number(selectedProbe.maxPos.toFixed(6)),
            score: Number(selectedProbe.score.toFixed(6)),
          }
        : null,
      poseError: Number.isFinite(selectedPoseError) ? Number(selectedPoseError.toFixed(6)) : null,
      attemptDebug: sanitizeForJson(attemptDebug),
      selectionDebug: sanitizeForJson(selectionDebug),
    },
    diagnostics: {
      events: buildEventMap(diagRecords),
      records: diagRecords,
      debug: {
        retarget: sanitizeForJson(windowRef.__vid2modelDebug || null),
        rigProfileState: sanitizeForJson(windowRef.__vid2modelRigProfileState || null),
        alignment: sanitizeForJson(windowRef.__vid2modelAlignment || null),
      },
      chainDiagnostics: {
        restCorrections: sanitizeForJson(globalThis.__vid2modelRestCorrections || []),
        leg: sanitizeForJson(globalThis.__vid2modelLegChainDiag || []),
        arm: sanitizeForJson(globalThis.__vid2modelArmChainDiag || []),
        torso: sanitizeForJson(globalThis.__vid2modelTorsoChainDiag || []),
        foot: sanitizeForJson(globalThis.__vid2modelFootChainDiag || []),
        footCorrectionDebug: sanitizeForJson(globalThis.__vid2modelFootCorrectionDebug || []),
      },
      calibrations: {
        body: bodyLengthCalibration
          ? {
              bones: bodyLengthCalibration.entries.length,
              globalScale: bodyLengthCalibration.globalScale,
              clampedCount: bodyLengthCalibration.clampedCount,
            }
          : null,
        arm: armLengthCalibration
          ? {
              bones: armLengthCalibration.entries.length,
            }
          : null,
        finger: fingerLengthCalibration
          ? {
              bones: fingerLengthCalibration.entries.length,
              globalScale: fingerLengthCalibration.globalScale,
              clampedCount: fingerLengthCalibration.clampedCount,
            }
          : null,
      },
    },
  };
}
