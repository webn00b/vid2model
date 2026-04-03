import * as THREE from "three";
import { getPrimaryChildDirectionLocal } from "./retarget-chain-utils.js";

const _worldPosA = new THREE.Vector3();
const _worldPosB = new THREE.Vector3();
const BODY_STAGE_CANONICAL = [
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
];
const FULL_STAGE_CANONICAL = [...BODY_STAGE_CANONICAL];

function toFixedArray(vec, size = 3) {
  if (!vec) return null;
  if (size === 4) {
    return [
      Number(vec.x.toFixed(6)),
      Number(vec.y.toFixed(6)),
      Number(vec.z.toFixed(6)),
      Number(vec.w.toFixed(6)),
    ];
  }
  return [
    Number(vec.x.toFixed(6)),
    Number(vec.y.toFixed(6)),
    Number(vec.z.toFixed(6)),
  ];
}

function buildUniqueBoneList(skinnedMeshes) {
  const out = [];
  const seen = new Set();
  for (const mesh of skinnedMeshes || []) {
    for (const bone of mesh.skeleton?.bones || []) {
      const id = bone.uuid || `${mesh.uuid}:${bone.name}`;
      if (seen.has(id)) continue;
      seen.add(id);
      out.push(bone);
    }
  }
  return out;
}

function buildCanonicalToTargetMap(rows) {
  const out = {};
  for (const row of rows || []) {
    if (!row?.canonical) continue;
    out[row.canonical] = row.name;
  }
  return out;
}

function segmentLength(canonicalMap, a, b) {
  const first = canonicalMap.get(a);
  const second = canonicalMap.get(b);
  if (!first?.isBone || !second?.isBone) return 0;
  first.getWorldPosition(_worldPosA);
  second.getWorldPosition(_worldPosB);
  return Number(_worldPosA.distanceTo(_worldPosB).toFixed(6));
}

function buildSegmentLengths(canonicalMap) {
  return {
    spine: segmentLength(canonicalMap, "hips", "spine"),
    chest: segmentLength(canonicalMap, "spine", "chest"),
    upperChest: segmentLength(canonicalMap, "chest", "upperChest"),
    neck: segmentLength(canonicalMap, "upperChest", "neck"),
    head: segmentLength(canonicalMap, "neck", "head"),
    leftUpperArm: segmentLength(canonicalMap, "leftShoulder", "leftUpperArm"),
    leftLowerArm: segmentLength(canonicalMap, "leftUpperArm", "leftLowerArm"),
    leftHand: segmentLength(canonicalMap, "leftLowerArm", "leftHand"),
    rightUpperArm: segmentLength(canonicalMap, "rightShoulder", "rightUpperArm"),
    rightLowerArm: segmentLength(canonicalMap, "rightUpperArm", "rightLowerArm"),
    rightHand: segmentLength(canonicalMap, "rightLowerArm", "rightHand"),
    leftUpperLeg: segmentLength(canonicalMap, "hips", "leftUpperLeg"),
    leftLowerLeg: segmentLength(canonicalMap, "leftUpperLeg", "leftLowerLeg"),
    leftFoot: segmentLength(canonicalMap, "leftLowerLeg", "leftFoot"),
    leftToes: segmentLength(canonicalMap, "leftFoot", "leftToes"),
    rightUpperLeg: segmentLength(canonicalMap, "hips", "rightUpperLeg"),
    rightLowerLeg: segmentLength(canonicalMap, "rightUpperLeg", "rightLowerLeg"),
    rightFoot: segmentLength(canonicalMap, "rightLowerLeg", "rightFoot"),
    rightToes: segmentLength(canonicalMap, "rightFoot", "rightToes"),
  };
}

function sumPositive(values) {
  return Number(
    values
      .filter((value) => Number.isFinite(value) && value > 0)
      .reduce((acc, value) => acc + value, 0)
      .toFixed(6)
  );
}

function ratio(numerator, denominator) {
  if (!(denominator > 1e-6)) return 0;
  return Number((numerator / denominator).toFixed(6));
}

function buildProportions(segmentLengths) {
  const torso = sumPositive([
    segmentLengths.spine,
    segmentLengths.chest,
    segmentLengths.upperChest,
    segmentLengths.neck,
    segmentLengths.head,
  ]);
  const leftArm = sumPositive([
    segmentLengths.leftUpperArm,
    segmentLengths.leftLowerArm,
    segmentLengths.leftHand,
  ]);
  const rightArm = sumPositive([
    segmentLengths.rightUpperArm,
    segmentLengths.rightLowerArm,
    segmentLengths.rightHand,
  ]);
  const leftLeg = sumPositive([
    segmentLengths.leftUpperLeg,
    segmentLengths.leftLowerLeg,
    segmentLengths.leftFoot,
    segmentLengths.leftToes,
  ]);
  const rightLeg = sumPositive([
    segmentLengths.rightUpperLeg,
    segmentLengths.rightLowerLeg,
    segmentLengths.rightFoot,
    segmentLengths.rightToes,
  ]);
  const avgArm = Number((((leftArm + rightArm) * 0.5)).toFixed(6));
  const avgLeg = Number((((leftLeg + rightLeg) * 0.5)).toFixed(6));
  return {
    torso,
    leftArm,
    rightArm,
    leftLeg,
    rightLeg,
    armToTorso: ratio(avgArm, torso),
    legToTorso: ratio(avgLeg, torso),
    leftRightArmRatio: ratio(leftArm, rightArm || leftArm),
    leftRightLegRatio: ratio(leftLeg, rightLeg || leftLeg),
  };
}

function buildFootHints(canonicalMap, side) {
  const foot = canonicalMap.get(`${side}Foot`) || null;
  const toes = canonicalMap.get(`${side}Toes`) || null;
  if (!foot?.isBone) return null;
  const primaryChildDirLocal = getPrimaryChildDirectionLocal(foot);
  let forward = primaryChildDirLocal;
  if ((!forward || forward.lengthSq() < 1e-10) && toes?.isBone) {
    foot.getWorldPosition(_worldPosA);
    toes.getWorldPosition(_worldPosB);
    forward = new THREE.Vector3().copy(_worldPosB).sub(_worldPosA);
    foot.getWorldQuaternion(new THREE.Quaternion()).invert();
  }
  return {
    forwardLocal: forward ? toFixedArray(forward) : null,
    hasToesBone: !!toes?.isBone,
  };
}

function buildSeedCorrectionSummary(profile) {
  const summary = [];
  const armEnabled = !!(
    profile?.enableShoulderDirectionCorrectionBySide?.left ||
    profile?.enableShoulderDirectionCorrectionBySide?.right ||
    profile?.enableUpperArmDirectionCorrectionBySide?.left ||
    profile?.enableUpperArmDirectionCorrectionBySide?.right ||
    profile?.enableElbowPlaneCorrectionBySide?.left ||
    profile?.enableElbowPlaneCorrectionBySide?.right ||
    profile?.enableForearmDirectionCorrectionBySide?.left ||
    profile?.enableForearmDirectionCorrectionBySide?.right
  );
  const legEnabled = !!(
    profile?.enableKneePlaneCorrectionBySide?.left ||
    profile?.enableKneePlaneCorrectionBySide?.right ||
    profile?.enableUpperLegDirectionCorrectionBySide?.left ||
    profile?.enableUpperLegDirectionCorrectionBySide?.right ||
    profile?.enableShinDirectionCorrectionBySide?.left ||
    profile?.enableShinDirectionCorrectionBySide?.right
  );
  const footEnabled = !!(
    profile?.enableFootDirectionCorrectionBySide?.left ||
    profile?.enableFootDirectionCorrectionBySide?.right ||
    profile?.enableFootPlaneCorrectionBySide?.left ||
    profile?.enableFootPlaneCorrectionBySide?.right ||
    profile?.enableFootMirrorCorrectionBySide?.left ||
    profile?.enableFootMirrorCorrectionBySide?.right
  );
  if (armEnabled) summary.push("arms");
  if (legEnabled) summary.push("legs");
  if (footEnabled) summary.push("feet");
  return summary;
}

export function createViewerModelAnalysisTools({
  windowRef,
  buildCanonicalBoneMap,
  canonicalBoneKey,
  getModelSkinnedMeshes,
  getModelLabel,
  getModelRigFingerprint,
  getModelSkeletonRootBone,
  getModelVrmHumanoidBones,
  getModelVrmNormalizedHumanoidBones,
}) {
  function buildCurrentModelAnalysis() {
    const skinnedMeshes = getModelSkinnedMeshes() || [];
    if (!skinnedMeshes.length) return null;
    const bones = buildUniqueBoneList(skinnedMeshes);
    if (!bones.length) return null;
    bones[0].updateMatrixWorld(true);
    const canonicalMap = buildCanonicalBoneMap(bones);
    const rows = bones.map((bone) => {
      const bindPos = bone.userData.__bindPosition?.isVector3 ? bone.userData.__bindPosition : bone.position;
      const bindQuat = bone.quaternion;
      const primaryDir = getPrimaryChildDirectionLocal(bone);
      return {
        name: bone.name || "",
        canonical: canonicalBoneKey(bone.name) || "",
        parent: bone.parent?.isBone ? bone.parent.name || "" : "",
        depth: (() => {
          let depth = 0;
          let node = bone.parent || null;
          while (node) {
            depth += 1;
            node = node.parent || null;
          }
          return depth;
        })(),
        childCount: (bone.children || []).filter((child) => child?.isBone).length,
        bindPositionLocal: bindPos?.isVector3 ? toFixedArray(bindPos) : null,
        bindQuaternionLocal: bindQuat?.isQuaternion ? toFixedArray(bindQuat, 4) : null,
        primaryChildDirLocal: primaryDir ? toFixedArray(primaryDir) : null,
      };
    });
    const segmentLengths = buildSegmentLengths(canonicalMap);
    const rootBone = getModelSkeletonRootBone?.() || bones[0] || null;
    const hipsBone = canonicalMap.get("hips") || rootBone;
    return {
      format: "vid2model.model-analysis.v1",
      generatedAt: new Date().toISOString(),
      modelLabel: getModelLabel(),
      modelFingerprint: getModelRigFingerprint(),
      skinnedMeshCount: skinnedMeshes.length,
      boneCount: rows.length,
      humanoid: {
        rawBones: (getModelVrmHumanoidBones?.() || []).map((bone) => bone.name || ""),
        normalizedBones: (getModelVrmNormalizedHumanoidBones?.() || []).map((bone) => bone.name || ""),
        canonicalToTarget: buildCanonicalToTargetMap(rows),
      },
      root: {
        name: rootBone?.name || "",
        bindQuaternionLocal: rootBone?.quaternion?.isQuaternion ? toFixedArray(rootBone.quaternion, 4) : null,
      },
      hips: {
        name: hipsBone?.name || "",
        bindPositionLocal: hipsBone?.position?.isVector3 ? toFixedArray(hipsBone.position) : null,
        bindQuaternionLocal: hipsBone?.quaternion?.isQuaternion ? toFixedArray(hipsBone.quaternion, 4) : null,
        primaryChildDirLocal: (() => {
          const dir = hipsBone ? getPrimaryChildDirectionLocal(hipsBone) : null;
          return dir ? toFixedArray(dir) : null;
        })(),
      },
      footHints: {
        left: buildFootHints(canonicalMap, "left"),
        right: buildFootHints(canonicalMap, "right"),
      },
      segmentLengths,
      proportions: buildProportions(segmentLengths),
      bones: rows,
    };
  }

  function exportCurrentModelAnalysis(download = false, filename = "") {
    const payload = buildCurrentModelAnalysis();
    if (!payload) {
      console.warn("[vid2model/diag] model-analysis: no model loaded");
      return null;
    }
    windowRef.__vid2modelModelAnalysis = payload;
    if (download) {
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      });
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download =
        String(filename || "").trim() ||
        `${(payload.modelLabel || "model").replace(/\.[^.]+$/, "") || "model"}.model-analysis.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(href), 0);
    }
    console.log("[vid2model/diag] model-analysis", {
      modelLabel: payload.modelLabel,
      modelFingerprint: payload.modelFingerprint,
      bones: payload.boneCount,
      downloaded: !!download,
    });
    return payload;
  }

  function buildRigProfileSeedFromAnalysis(analysis, stage = "body") {
    if (!analysis?.modelFingerprint) return null;
    const normalizedStage = String(stage || "body").trim().toLowerCase();
    const canonicalToTarget = analysis?.humanoid?.canonicalToTarget || {};
    const namesTargetToSource = {};
    for (const [canonical, targetName] of Object.entries(canonicalToTarget)) {
      if (!canonical || !targetName) continue;
      namesTargetToSource[String(targetName)] = String(canonical);
    }
    const segmentLengths = analysis?.segmentLengths || {};
    const hasLeftArm =
      Number(segmentLengths.leftUpperArm || 0) > 0 && Number(segmentLengths.leftLowerArm || 0) > 0;
    const hasRightArm =
      Number(segmentLengths.rightUpperArm || 0) > 0 && Number(segmentLengths.rightLowerArm || 0) > 0;
    const hasLeftLeg =
      Number(segmentLengths.leftUpperLeg || 0) > 0 && Number(segmentLengths.leftLowerLeg || 0) > 0;
    const hasRightLeg =
      Number(segmentLengths.rightUpperLeg || 0) > 0 && Number(segmentLengths.rightLowerLeg || 0) > 0;
    return {
      id: `${analysis.modelFingerprint}:${normalizedStage}:seed`,
      modelLabel: String(analysis.modelLabel || ""),
      modelFingerprint: String(analysis.modelFingerprint || ""),
      stage: normalizedStage,
      source: "model-analysis-seed",
      validationStatus: "draft",
      autoSaved: false,
      autoGenerated: true,
      basedOnModelAnalysis: true,
      generatedAt: analysis.generatedAt || new Date().toISOString(),
      preferredMode: "skeletonutils-skinnedmesh",
      preferSkeletonOnRenameFallback: true,
      mirrorSwap: "disable",
      namesTargetToSource,
      bodyCanonicalKeys:
        normalizedStage === "full" ? [...FULL_STAGE_CANONICAL] : [...BODY_STAGE_CANONICAL],
      rotationScaleByCanonical: {
        spine: 0.5,
        chest: 0.3,
        upperChest: 0.24,
        neck: 0.45,
        head: 0.65,
        leftShoulder: 0.18,
        rightShoulder: 0.18,
        leftUpperArm: 0.55,
        rightUpperArm: 0.55,
        leftLowerArm: 0.78,
        rightLowerArm: 0.78,
        leftHand: 0.1,
        rightHand: 0.1,
      },
      enableShoulderDirectionCorrectionBySide: { left: hasLeftArm, right: hasRightArm },
      enableUpperArmDirectionCorrectionBySide: { left: hasLeftArm, right: hasRightArm },
      enableElbowPlaneCorrectionBySide: { left: hasLeftArm, right: hasRightArm },
      enableForearmDirectionCorrectionBySide: { left: hasLeftArm, right: hasRightArm },
      enableKneePlaneCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
      enableUpperLegDirectionCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
      enableShinDirectionCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
      enableFootDirectionCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
      enableFootPlaneCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
      enableFootMirrorCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
      modelAnalysisSummary: {
        boneCount: Number(analysis.boneCount || 0),
        skinnedMeshCount: Number(analysis.skinnedMeshCount || 0),
        proportions: analysis.proportions || {},
      },
      inferredCorrections: buildSeedCorrectionSummary({
        enableShoulderDirectionCorrectionBySide: { left: hasLeftArm, right: hasRightArm },
        enableUpperArmDirectionCorrectionBySide: { left: hasLeftArm, right: hasRightArm },
        enableElbowPlaneCorrectionBySide: { left: hasLeftArm, right: hasRightArm },
        enableForearmDirectionCorrectionBySide: { left: hasLeftArm, right: hasRightArm },
        enableKneePlaneCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
        enableUpperLegDirectionCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
        enableShinDirectionCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
        enableFootDirectionCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
        enableFootPlaneCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
        enableFootMirrorCorrectionBySide: { left: hasLeftLeg, right: hasRightLeg },
      }),
    };
  }

  function buildRigProfileSeedForCurrentModel(stage = "body") {
    return buildRigProfileSeedFromAnalysis(buildCurrentModelAnalysis(), stage);
  }

  return {
    buildCurrentModelAnalysis,
    exportCurrentModelAnalysis,
    buildRigProfileSeedFromAnalysis,
    buildRigProfileSeedForCurrentModel,
    buildSeedCorrectionSummary,
  };
}
