import * as THREE from "three";
import { canonicalBoneKey } from "./bone-utils.js";
import { buildRetargetMap } from "./retarget-helpers.js";

function inferHumanoidCanonicalNamesFromTopology(bones) {
  const boneSet = new Set(bones);
  const childrenMap = new Map();
  for (const bone of bones) {
    childrenMap.set(
      bone,
      bone.children.filter((child) => boneSet.has(child))
    );
  }

  const worldPos = new Map();
  for (const bone of bones) {
    const point = new THREE.Vector3();
    bone.getWorldPosition(point);
    worldPos.set(bone, point);
  }

  const descendantCount = new Map();
  const countDesc = (bone) => {
    let total = 0;
    for (const child of childrenMap.get(bone) || []) {
      total += 1 + countDesc(child);
    }
    descendantCount.set(bone, total);
    return total;
  };
  const roots = bones.filter((bone) => !boneSet.has(bone.parent));
  for (const root of roots) {
    countDesc(root);
  }

  const sortedRoots = (roots.length ? roots : [bones[0]]).slice().sort(
    (a, b) => (descendantCount.get(b) || 0) - (descendantCount.get(a) || 0)
  );
  const hips = sortedRoots[0];
  if (!hips) {
    return new Map();
  }

  const assignment = new Map();
  assignment.set(hips, "hips");

  const getHighestYChild = (bone, exclude = new Set()) => {
    const children = (childrenMap.get(bone) || []).filter(
      (child) => !exclude.has(child)
    );
    if (!children.length) return null;
    return children
      .slice()
      .sort((a, b) => worldPos.get(b).y - worldPos.get(a).y)[0];
  };
  const getLowestYChild = (bone, exclude = new Set()) => {
    const children = (childrenMap.get(bone) || []).filter(
      (child) => !exclude.has(child)
    );
    if (!children.length) return null;
    return children
      .slice()
      .sort((a, b) => worldPos.get(a).y - worldPos.get(b).y)[0];
  };
  const xOf = (bone) => worldPos.get(bone)?.x ?? 0;

  const torsoChain = [];
  let cursor = getHighestYChild(hips);
  while (cursor && torsoChain.length < 5) {
    torsoChain.push(cursor);
    const next = getHighestYChild(cursor);
    if (!next) break;
    if (worldPos.get(next).y < worldPos.get(cursor).y - 1e-4) break;
    cursor = next;
  }

  const spine = torsoChain[0] || null;
  const chest = torsoChain[1] || null;
  let upperChest = null;
  let neck = null;
  let head = null;
  if (torsoChain.length >= 5) {
    upperChest = torsoChain[2];
    neck = torsoChain[3];
    head = torsoChain[4];
  } else if (torsoChain.length === 4) {
    neck = torsoChain[2];
    head = torsoChain[3];
  } else if (torsoChain.length === 3) {
    neck = torsoChain[2];
  }

  if (spine) assignment.set(spine, "spine");
  if (chest) assignment.set(chest, "chest");
  if (upperChest) assignment.set(upperChest, "upperChest");
  if (neck) assignment.set(neck, "neck");
  if (head) assignment.set(head, "head");

  const torsoSet = new Set([hips, ...torsoChain].filter(Boolean));
  const hipsChildren = childrenMap.get(hips) || [];
  const legRoots = hipsChildren
    .filter((child) => !torsoSet.has(child))
    .slice()
    .sort((a, b) => worldPos.get(a).y - worldPos.get(b).y)
    .slice(0, 2);

  const assignLeg = (rootBone, side) => {
    if (!rootBone) return;
    assignment.set(rootBone, `${side}UpperLeg`);
    const lower = getLowestYChild(rootBone);
    if (lower) assignment.set(lower, `${side}LowerLeg`);
    const foot = lower ? getLowestYChild(lower) : null;
    if (foot) assignment.set(foot, `${side}Foot`);
    const toes = foot ? getLowestYChild(foot) : null;
    if (toes) assignment.set(toes, `${side}Toes`);
  };

  if (legRoots.length === 2) {
    const [a, b] = legRoots;
    const leftRoot = xOf(a) <= xOf(b) ? a : b;
    const rightRoot = leftRoot === a ? b : a;
    assignLeg(leftRoot, "left");
    assignLeg(rightRoot, "right");
  } else if (legRoots.length === 1) {
    const side = xOf(legRoots[0]) < xOf(hips) ? "left" : "right";
    assignLeg(legRoots[0], side);
  }

  const shoulderBase = upperChest || chest || spine || hips;
  const shoulderBaseChildren = shoulderBase
    ? childrenMap.get(shoulderBase) || []
    : [];
  const upFromShoulderBase = shoulderBase
    ? getHighestYChild(shoulderBase)
    : null;
  let armRoots = shoulderBaseChildren.filter((child) => child !== upFromShoulderBase);

  if (armRoots.length < 2 && chest && shoulderBase !== chest) {
    const chestChildren = childrenMap.get(chest) || [];
    const chestUp = getHighestYChild(chest);
    const extra = chestChildren.filter((child) => child !== chestUp);
    armRoots = [...new Set([...armRoots, ...extra])];
  }

  armRoots = armRoots
    .filter((child) => !assignment.has(child))
    .slice()
    .sort((a, b) => Math.abs(xOf(b) - xOf(hips)) - Math.abs(xOf(a) - xOf(hips)))
    .slice(0, 2);

  const assignArm = (rootBone, side) => {
    if (!rootBone) return;
    const c1 =
      (childrenMap.get(rootBone) || [])
        .slice()
        .sort(
          (a, b) =>
            Math.abs(xOf(b) - xOf(hips)) - Math.abs(xOf(a) - xOf(hips))
        )[0] || null;
    const c2 = c1
      ? (childrenMap.get(c1) || [])
          .slice()
          .sort(
            (a, b) =>
              Math.abs(xOf(b) - xOf(hips)) - Math.abs(xOf(a) - xOf(hips))
          )[0] || null
      : null;
    const c3 = c2
      ? (childrenMap.get(c2) || [])
          .slice()
          .sort(
            (a, b) =>
              Math.abs(xOf(b) - xOf(hips)) - Math.abs(xOf(a) - xOf(hips))
          )[0] || null
      : null;

    if (c1 && c2 && c3) {
      assignment.set(rootBone, `${side}Shoulder`);
      assignment.set(c1, `${side}UpperArm`);
      assignment.set(c2, `${side}LowerArm`);
      assignment.set(c3, `${side}Hand`);
    } else if (c1 && c2) {
      assignment.set(rootBone, `${side}UpperArm`);
      assignment.set(c1, `${side}LowerArm`);
      assignment.set(c2, `${side}Hand`);
    } else if (c1) {
      assignment.set(rootBone, `${side}UpperArm`);
      assignment.set(c1, `${side}LowerArm`);
    }
  };

  if (armRoots.length === 2) {
    const [a, b] = armRoots;
    const leftRoot = xOf(a) <= xOf(b) ? a : b;
    const rightRoot = leftRoot === a ? b : a;
    assignArm(leftRoot, "left");
    assignArm(rightRoot, "right");
  } else if (armRoots.length === 1) {
    const side = xOf(armRoots[0]) < xOf(hips) ? "left" : "right";
    assignArm(armRoots[0], side);
  }

  return assignment;
}

export function autoNameTargetBones(skinnedMesh) {
  const bones = skinnedMesh?.skeleton?.bones || [];
  if (!bones.length) {
    return { missingBefore: 0, autoNamed: 0, inferredCanonical: 0 };
  }

  let missingBefore = 0;
  for (const bone of bones) {
    if (!bone.name || !bone.name.trim()) {
      missingBefore += 1;
    }
  }

  let autoNamed = 0;
  const usedNames = new Set();
  for (const bone of bones) {
    const name = (bone.name || "").trim();
    if (name) usedNames.add(name);
  }

  for (let i = 0; i < bones.length; i += 1) {
    const bone = bones[i];
    if (!bone.name || !bone.name.trim()) {
      let generated = `autoBone_${String(i).padStart(3, "0")}`;
      while (usedNames.has(generated)) {
        generated = `${generated}_x`;
      }
      bone.name = generated;
      usedNames.add(generated);
      autoNamed += 1;
    }
  }

  let inferredCanonical = 0;
  if (missingBefore > 0) {
    skinnedMesh.updateMatrixWorld(true);
    const inferred = inferHumanoidCanonicalNamesFromTopology(bones);
    for (const [bone, canonical] of inferred.entries()) {
      if (!canonical) continue;
      if (usedNames.has(canonical) && bone.name !== canonical) continue;
      if (bone.name !== canonical) {
        usedNames.delete(bone.name);
        bone.name = canonical;
        usedNames.add(canonical);
        inferredCanonical += 1;
      }
    }
  }

  return { missingBefore, autoNamed, inferredCanonical };
}

function isLowMatchRetargetMap(result, sourceTotal) {
  const targetCoverage =
    result.canonicalCandidates > 0
      ? result.matched / result.canonicalCandidates
      : 0;
  const sourceCoverage = sourceTotal > 0 ? result.sourceMatched / sourceTotal : 0;
  return (
    result.canonicalCandidates === 0 ||
    targetCoverage < 0.75 ||
    (targetCoverage < 0.9 && sourceCoverage < 0.35)
  );
}

function isRetargetMapBetter(candidate, baseline) {
  if (!candidate) return false;
  if (!baseline) return true;
  if (candidate.matched !== baseline.matched) {
    return candidate.matched > baseline.matched;
  }
  if (candidate.sourceMatched !== baseline.sourceMatched) {
    return candidate.sourceMatched > baseline.sourceMatched;
  }
  return Object.keys(candidate.names || {}).length >
    Object.keys(baseline.names || {}).length;
}

function buildTopologyFallbackRenamePlan(skinnedMesh, baseResult, canonicalFilter) {
  const bones = skinnedMesh?.skeleton?.bones || [];
  if (!bones.length) return null;

  skinnedMesh.updateMatrixWorld(true);
  const inferred = inferHumanoidCanonicalNamesFromTopology(bones);
  if (!inferred.size) return null;

  const reservedCanonical = new Set();
  for (const targetName of Object.keys(baseResult?.names || {})) {
    const canonical = canonicalBoneKey(targetName);
    if (canonical) reservedCanonical.add(canonical);
  }

  const reservedNames = new Set(
    bones
      .map((bone) => String(bone?.name || "").trim())
      .filter((name) => !!name)
  );

  const plan = [];
  for (const [bone, canonical] of inferred.entries()) {
    if (!canonical) continue;
    if (canonicalFilter && !canonicalFilter.has(canonical)) continue;
    if (baseResult?.names?.[bone.name]) continue;

    const currentCanonical = canonicalBoneKey(bone.name);
    if (currentCanonical === canonical) continue;
    if (reservedCanonical.has(canonical)) continue;
    if (reservedNames.has(canonical) && bone.name !== canonical) continue;

    plan.push({
      bone,
      from: bone.name,
      to: canonical,
      previousCanonical: currentCanonical || null,
    });
    reservedCanonical.add(canonical);
    reservedNames.delete(bone.name);
    reservedNames.add(canonical);
  }

  if (!plan.length) return null;
  return {
    plan,
    sample: plan.slice(0, 12).map((row) => ({
      from: row.from,
      to: row.to,
      previousCanonical: row.previousCanonical,
    })),
  };
}

function applyBoneRenamePlan(plan) {
  for (const row of plan || []) {
    row.bone.name = row.to;
  }
}

function revertBoneRenamePlan(plan) {
  for (const row of plan || []) {
    row.bone.name = row.from;
  }
}

export function maybeApplyTopologyFallback(
  skinnedMesh,
  sourceBones,
  canonicalFilter,
  baseResult
) {
  const sourceTotal = sourceBones?.length || 0;
  const shouldTry = isLowMatchRetargetMap(baseResult, sourceTotal);
  const baseMappedPairs = Object.keys(baseResult?.names || {}).length;
  const baseTargetCoverage =
    baseResult?.canonicalCandidates > 0
      ? baseResult.matched / baseResult.canonicalCandidates
      : 0;
  const baseSourceCoverage = sourceTotal > 0 ? baseResult.sourceMatched / sourceTotal : 0;

  if (!shouldTry) {
    return {
      result: baseResult,
      attempted: false,
      applied: false,
      reason: "coverage-ok",
    };
  }

  const renameInfo = buildTopologyFallbackRenamePlan(
    skinnedMesh,
    baseResult,
    canonicalFilter
  );
  if (!renameInfo?.plan?.length) {
    return {
      result: baseResult,
      attempted: true,
      applied: false,
      reason: "no-inferred-renames",
      before: {
        mappedPairs: baseMappedPairs,
        targetCoverage: Number(baseTargetCoverage.toFixed(4)),
        sourceCoverage: Number(baseSourceCoverage.toFixed(4)),
      },
    };
  }

  applyBoneRenamePlan(renameInfo.plan);
  const topologyResult = buildRetargetMap(skinnedMesh.skeleton.bones, sourceBones, {
    canonicalFilter,
  });
  const topologyMappedPairs = Object.keys(topologyResult.names || {}).length;
  const topologyTargetCoverage =
    topologyResult.canonicalCandidates > 0
      ? topologyResult.matched / topologyResult.canonicalCandidates
      : 0;
  const topologySourceCoverage =
    sourceTotal > 0 ? topologyResult.sourceMatched / sourceTotal : 0;
  const better = isRetargetMapBetter(topologyResult, baseResult);

  if (!better) {
    revertBoneRenamePlan(renameInfo.plan);
  }

  return {
    result: better ? topologyResult : baseResult,
    attempted: true,
    applied: better,
    reason: better ? "improved-coverage" : "not-better",
    inferredRenames: renameInfo.plan.length,
    sample: renameInfo.sample,
    before: {
      mappedPairs: baseMappedPairs,
      targetCoverage: Number(baseTargetCoverage.toFixed(4)),
      sourceCoverage: Number(baseSourceCoverage.toFixed(4)),
    },
    after: {
      mappedPairs: topologyMappedPairs,
      targetCoverage: Number(topologyTargetCoverage.toFixed(4)),
      sourceCoverage: Number(topologySourceCoverage.toFixed(4)),
    },
  };
}
