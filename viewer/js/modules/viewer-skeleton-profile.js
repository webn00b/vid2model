export function createViewerSkeletonProfileTools({
  windowRef,
  buildCanonicalBoneMap,
  getModelSkinnedMesh,
  getModelLabel,
  getModelRigFingerprint,
  vrmHumanoidBoneNames,
}) {
  function buildDefaultSkeletonProfileBlend() {
    const out = {};
    for (const joint of vrmHumanoidBoneNames) {
      let blend = 1.0;
      if (joint === "hips") blend = 0.85;
      else if (joint === "spine") blend = 0.55;
      else if (joint === "chest") blend = 0.45;
      else if (joint === "upperChest") blend = 0.4;
      else if (joint === "neck" || joint === "head") blend = 0.5;
      else if (joint.includes("Shoulder")) blend = 0.35;
      else if (joint.includes("UpperArm")) blend = 0.3;
      else if (joint.includes("LowerArm")) blend = 0.25;
      else if (joint.includes("Hand")) blend = 0.2;
      else if (
        joint.includes("Thumb") ||
        joint.includes("Index") ||
        joint.includes("Middle") ||
        joint.includes("Ring") ||
        joint.includes("Little")
      ) {
        blend = 0.12;
      } else if (
        joint.includes("UpperLeg") ||
        joint.includes("LowerLeg") ||
        joint.includes("Foot") ||
        joint.includes("Toes")
      ) {
        blend = 1.0;
      }
      out[joint] = blend;
    }
    return out;
  }

  function buildCurrentModelSkeletonProfile() {
    const modelSkinnedMesh = getModelSkinnedMesh();
    if (!modelSkinnedMesh) return null;
    const canonicalMap = buildCanonicalBoneMap(modelSkinnedMesh.skeleton?.bones || []);
    const jointOffsets = {};
    for (const joint of vrmHumanoidBoneNames) {
      const bone = canonicalMap.get(joint) || null;
      if (!bone?.isBone) continue;
      const bindPos = bone.userData.__bindPosition?.isVector3
        ? bone.userData.__bindPosition
        : bone.position;
      if (!bindPos?.isVector3) continue;
      jointOffsets[joint] = [
        Number(bindPos.x.toFixed(6)),
        Number(bindPos.y.toFixed(6)),
        Number(bindPos.z.toFixed(6)),
      ];
    }
    return {
      format: "vid2model.skeleton-profile.v1",
      generatedAt: new Date().toISOString(),
      modelLabel: getModelLabel(),
      modelFingerprint: getModelRigFingerprint(),
      joint_offsets: jointOffsets,
      joint_blend: buildDefaultSkeletonProfileBlend(),
    };
  }

  function exportCurrentModelSkeletonProfile(download = false, filename = "") {
    const payload = buildCurrentModelSkeletonProfile();
    if (!payload) {
      console.warn("[vid2model/diag] skeleton-profile: no model loaded");
      return null;
    }
    windowRef.__vid2modelSkeletonProfile = payload;
    if (download) {
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      });
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download =
        String(filename || "").trim() ||
        `${(payload.modelLabel || "model").replace(/\.[^.]+$/, "") || "model"}.skeleton-profile.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(href), 0);
    }
    console.log("[vid2model/diag] skeleton-profile", {
      modelLabel: payload.modelLabel,
      modelFingerprint: payload.modelFingerprint,
      joints: Object.keys(payload.joint_offsets || {}).length,
      downloaded: !!download,
    });
    return payload;
  }

  return {
    buildDefaultSkeletonProfileBlend,
    buildCurrentModelSkeletonProfile,
    exportCurrentModelSkeletonProfile,
  };
}
