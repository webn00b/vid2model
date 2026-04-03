import * as THREE from "three";

function formatVec3(vec) {
  if (!vec || !Number.isFinite(vec.x) || !Number.isFinite(vec.y) || !Number.isFinite(vec.z)) return "";
  return `${vec.x.toFixed(2)}, ${vec.y.toFixed(2)}, ${vec.z.toFixed(2)}`;
}

export function createViewerSkeletonDebugTools({
  canonicalBoneKey,
  buildCanonicalBoneMap,
  getPrimaryChildBone,
}) {
  const tmpWorldPosA = new THREE.Vector3();
  const tmpWorldPosB = new THREE.Vector3();

  function getBoneWorldPosString(bone) {
    if (!bone?.isBone) return "";
    bone.getWorldPosition(tmpWorldPosA);
    return formatVec3(tmpWorldPosA);
  }

  function getBonePrimaryChildWorldDirString(bone) {
    if (!bone?.isBone) return "";
    const child = getPrimaryChildBone(bone);
    if (!child) return "";
    bone.getWorldPosition(tmpWorldPosA);
    child.getWorldPosition(tmpWorldPosB);
    tmpWorldPosB.sub(tmpWorldPosA);
    if (tmpWorldPosB.lengthSq() < 1e-10) return "";
    tmpWorldPosB.normalize();
    return formatVec3(tmpWorldPosB);
  }

  function getSkeletonCanonicalRows({ targetBones = [], sourceBones = [], names = {}, canonicals = [] } = {}) {
    const targetMap = buildCanonicalBoneMap(targetBones || []);
    const sourceMap = buildCanonicalBoneMap(sourceBones || []);
    return (canonicals || []).map((canonical, index) => {
      const targetBone = targetMap.get(canonical) || null;
      const sourceCanonical = targetBone
        ? (canonicalBoneKey(names[targetBone.name] || canonical) || canonical)
        : canonical;
      const sourceBone = sourceMap.get(sourceCanonical) || null;
      return {
        index,
        canonical,
        targetBone: targetBone?.name || "",
        targetParent: targetBone?.parent?.isBone ? targetBone.parent.name : "",
        targetChild: getPrimaryChildBone(targetBone)?.name || "",
        targetWorldPos: getBoneWorldPosString(targetBone),
        targetDir: getBonePrimaryChildWorldDirString(targetBone),
        mappedSourceCanonical: sourceCanonical,
        sourceBone: sourceBone?.name || "",
        sourceParent: sourceBone?.parent?.isBone ? sourceBone.parent.name : "",
        sourceChild: getPrimaryChildBone(sourceBone)?.name || "",
        sourceWorldPos: getBoneWorldPosString(sourceBone),
        sourceDir: getBonePrimaryChildWorldDirString(sourceBone),
      };
    });
  }

  function resolveSkeletonDumpCanonicals(scope = "legs") {
    const normalized = String(scope || "legs").trim().toLowerCase();
    if (normalized === "left-arm" || normalized === "arm-left") {
      return ["leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand"];
    }
    if (normalized === "right-arm" || normalized === "arm-right") {
      return ["rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand"];
    }
    if (normalized === "arms") {
      return [
        "leftShoulder",
        "leftUpperArm",
        "leftLowerArm",
        "leftHand",
        "rightShoulder",
        "rightUpperArm",
        "rightLowerArm",
        "rightHand",
      ];
    }
    if (normalized === "torso") {
      return ["hips", "spine", "chest", "upperChest", "neck", "head"];
    }
    if (normalized === "head") {
      return ["upperChest", "neck", "head"];
    }
    if (normalized === "left" || normalized === "left-leg") {
      return ["leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes"];
    }
    if (normalized === "right" || normalized === "right-leg") {
      return ["rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes"];
    }
    if (normalized === "body") {
      return [
        "hips",
        "spine",
        "chest",
        "upperChest",
        "leftShoulder",
        "leftUpperArm",
        "leftLowerArm",
        "leftHand",
        "rightShoulder",
        "rightUpperArm",
        "rightLowerArm",
        "rightHand",
        "leftUpperLeg",
        "leftLowerLeg",
        "leftFoot",
        "leftToes",
        "rightUpperLeg",
        "rightLowerLeg",
        "rightFoot",
        "rightToes",
      ];
    }
    return [
      "leftUpperLeg",
      "leftLowerLeg",
      "leftFoot",
      "leftToes",
      "rightUpperLeg",
      "rightLowerLeg",
      "rightFoot",
      "rightToes",
    ];
  }

  return {
    getSkeletonCanonicalRows,
    resolveSkeletonDumpCanonicals,
  };
}
