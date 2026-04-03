import * as THREE from "three";

export function createViewerParsedModelApplier({
  state,
  clearModel,
  scene,
  setStatus,
  diag,
  VRMUtils,
  applyVrmHumanoidBoneNames,
  vrmHumanoidBoneNames,
  findSkinnedMeshes,
  getModelSkeletonRootBone,
  logModelBones,
  collectModelBoneRows,
  logRuntimeModelBones,
  autoNameTargetBones,
  buildModelRigFingerprint,
  syncSourceDisplayToModel,
  fitToSkeleton,
  refreshBoneLabels,
  loadDefaultVrmAnimation,
  defaultVrmAnimationUrl,
  updateTimelineUi,
  applyBvhToModel,
  ensureRepoRigProfilesForModel,
}) {
  return function applyParsedModel(gltf, label) {
    clearModel();
    const vrm = gltf.userData?.vrm || null;
    state.modelRoot = vrm?.scene || gltf.scene || gltf.scenes?.[0] || null;
    state.modelLabel = label || "";
    if (!state.modelRoot) {
      setStatus(`Failed to parse model: ${label}`);
      return;
    }
    if (vrm) {
      if (typeof VRMUtils.rotateVRM0 === "function") {
        VRMUtils.rotateVRM0(vrm);
      }
      if (typeof VRMUtils.combineSkeletons === "function") {
        VRMUtils.combineSkeletons(state.modelRoot);
      } else if (typeof VRMUtils.removeUnnecessaryJoints === "function") {
        VRMUtils.removeUnnecessaryJoints(state.modelRoot);
      }
      state.modelRoot.rotation.y = Math.PI;
      state.modelRoot.updateMatrixWorld(true);
    }
    const vrmHumanoidInfo = applyVrmHumanoidBoneNames(vrm, vrmHumanoidBoneNames);
    state.modelVrmHumanoidBones = vrmHumanoidInfo.bones || [];
    state.modelVrmNormalizedHumanoidBones = vrmHumanoidInfo.normalizedBones || [];
    const vrmDirectBones =
      state.modelVrmNormalizedHumanoidBones.length > 0
        ? state.modelVrmNormalizedHumanoidBones
        : state.modelVrmHumanoidBones;

    scene.add(state.modelRoot);
    state.modelRoot.userData.__baseQuaternion = state.modelRoot.quaternion.clone();
    state.modelRoot.userData.__basePosition = state.modelRoot.position.clone();
    state.modelSkinnedMeshes = findSkinnedMeshes(state.modelRoot);
    state.modelSkinnedMesh = state.modelSkinnedMeshes[0] || null;
    if (!state.modelSkinnedMesh) {
      setStatus(`Model loaded, but no skinned mesh found: ${label}`);
      return;
    }

    const rootBone = getModelSkeletonRootBone();
    if (rootBone && rootBone !== state.modelRoot) {
      rootBone.userData.__retargetBaseQuaternion = rootBone.quaternion.clone();
      rootBone.userData.__retargetBasePosition = rootBone.position.clone();
    }

    const seenBones = new Set();
    for (const mesh of state.modelSkinnedMeshes) {
      for (const bone of mesh.skeleton?.bones || []) {
        const id = bone.uuid || `${mesh.uuid}:${bone.name}`;
        if (seenBones.has(id)) continue;
        seenBones.add(id);
        bone.userData.__bindPosition = bone.position.clone();
      }
    }

    logModelBones(
      state.modelSkinnedMeshes,
      collectModelBoneRows,
      logRuntimeModelBones
    );

    let totalMissingBefore = 0;
    let totalAutoNamed = 0;
    let totalInferredCanonical = 0;
    for (const mesh of state.modelSkinnedMeshes) {
      const namingInfo = autoNameTargetBones(mesh);
      totalMissingBefore += namingInfo.missingBefore;
      totalAutoNamed += namingInfo.autoNamed;
      totalInferredCanonical += namingInfo.inferredCanonical;
    }

    state.modelRigFingerprint = buildModelRigFingerprint(
      state.modelSkinnedMeshes,
      state.modelLabel
    );
    diag("model-loaded", {
      file: label,
      skinnedMeshes: state.modelSkinnedMeshes.length,
      vrmHumanoid: vrmHumanoidInfo.applied
        ? {
            bones: vrmHumanoidInfo.applied,
            renamed: vrmHumanoidInfo.renamed,
            normalized: state.modelVrmNormalizedHumanoidBones.length,
          }
        : null,
      vrmDirectReady: vrmDirectBones.length > 0,
      topMeshes: state.modelSkinnedMeshes.slice(0, 3).map((mesh) => ({
        name: mesh.name || "(unnamed-skinned-mesh)",
        bones: mesh.skeleton.bones.length,
      })),
      autoNaming:
        totalMissingBefore > 0
          ? {
              missingBefore: totalMissingBefore,
              autoNamed: totalAutoNamed,
              inferredCanonical: totalInferredCanonical,
            }
          : null,
    });

    if (state.sourceResult) {
      syncSourceDisplayToModel();
    }
    fitToSkeleton(state.modelRoot);
    refreshBoneLabels();
    setStatus(
      `Model loaded: ${label} (skinned meshes: ${state.modelSkinnedMeshes.length})`
    );

    if (vrm && state.modelRoot) {
      const token = state.modelDefaultAnimationToken;
      loadDefaultVrmAnimation(defaultVrmAnimationUrl, vrm)
        .then((clip) => {
          if (!clip || token !== state.modelDefaultAnimationToken || !state.modelRoot) {
            return;
          }
          if (state.modelMixers.length || state.modelActions.length) return;
          const mix = new THREE.AnimationMixer(state.modelRoot);
          const action = mix.clipAction(clip);
          action.reset();
          action.setEffectiveWeight(1);
          action.setEffectiveTimeScale(1);
          action.play();
          state.modelMixer = mix;
          state.modelAction = action;
          state.modelMixers = [mix];
          state.modelActions = [action];
          state.isPlaying = true;
          updateTimelineUi(0);
          diag("model-default-animation", {
            file: label,
            clip: clip.name,
            duration: Number(clip.duration.toFixed(3)),
            url: defaultVrmAnimationUrl,
          });
        })
        .catch((err) => {
          console.warn("Default VRM animation failed:", err);
        });
    }

    const repoProfilesPromise = Promise.resolve(
      ensureRepoRigProfilesForModel
        ? ensureRepoRigProfilesForModel({
            modelFingerprint: state.modelRigFingerprint,
            modelLabel: state.modelLabel,
          })
        : null
    );
    if (state.sourceResult) {
      repoProfilesPromise.finally(() => {
        applyBvhToModel();
      });
    }
  };
}
