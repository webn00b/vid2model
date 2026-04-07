import * as THREE from "three";
import { canonicalBoneKey } from "./bone-utils.js";
import { buildCanonicalBoneMap } from "./retarget-helpers.js";
import { angleBetweenWorldSegments } from "./viewer-chain-diagnostics.js";

export function applyRigProfileNames(baseResult, profile, targetBones, sourceBones, canonicalFilter) {
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
    if (canonicalFilter && (!canonical || !canonicalFilter.has(canonical))) {
      continue;
    }
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

export function maybeSwapMirroredHumanoidSides(baseResult, targetBones, sourceBones, canonicalFilter) {
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

export function maybeSwapArmSidesByChain(baseResult, targetBones, sourceBones, canonicalFilter) {
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
    return angleBetweenWorldSegments(a0, a1, b0, b1);
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
  const shouldSwapByMajorChain =
    majorChainTotal >= 6 && majorChainSwappedVotes >= 4;
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
    if (!canonical) continue;
    if (!armCanonicals.includes(canonical)) continue;
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
