export function collectRetargetAttempts({
  modelSkinnedMesh,
  modelSkinnedMeshes,
  modelRoot,
  sourceResult,
  activeStageClip,
  names,
  SkeletonUtils,
  buildRenamedClip,
  resolvedTrackCountAcrossMeshes,
  resolvedTrackCountForTarget,
  shortErr,
}) {
  const retargetAttempts = [];
  const attemptDebug = [];

  const pushAttempt = (label, bindingRoot, fn) => {
    try {
      const candidate = fn();
      if (candidate && candidate.tracks && candidate.tracks.length > 0) {
        const resolvedTracks =
          bindingRoot === "skinned"
            ? resolvedTrackCountAcrossMeshes(candidate, modelSkinnedMeshes)
            : resolvedTrackCountForTarget(candidate, modelSkinnedMesh.skeleton.bones);
        retargetAttempts.push({ label, clip: candidate, bindingRoot, resolvedTracks });
        attemptDebug.push({
          label,
          bindingRoot,
          tracks: candidate.tracks.length,
          resolvedTracks,
          ok: true,
          error: "",
        });
      } else {
        attemptDebug.push({
          label,
          bindingRoot,
          tracks: 0,
          resolvedTracks: 0,
          ok: false,
          error: "empty clip",
        });
      }
    } catch (err) {
      attemptDebug.push({
        label,
        bindingRoot,
        tracks: 0,
        resolvedTracks: 0,
        ok: false,
        error: shortErr(err),
      });
    }
  };

  pushAttempt("skeletonutils-skinnedmesh", "skinned", () =>
    SkeletonUtils.retargetClip(modelSkinnedMesh, sourceResult.skeleton, activeStageClip, {
      names,
      hip: "hips",
      useFirstFramePosition: true,
      preserveBonePositions: false,
    })
  );

  const namesSourceToTarget = Object.fromEntries(
    Object.entries(names).map(([target, source]) => [source, target])
  );

  pushAttempt("skeletonutils-skinnedmesh-reversed", "skinned", () =>
    SkeletonUtils.retargetClip(modelSkinnedMesh, sourceResult.skeleton, activeStageClip, {
      names: namesSourceToTarget,
      hip: "hips",
      useFirstFramePosition: true,
      preserveBonePositions: false,
    })
  );

  if (modelRoot) {
    pushAttempt("skeletonutils-root", "root", () =>
      SkeletonUtils.retargetClip(modelRoot, sourceResult.skeleton, activeStageClip, {
        names,
        hip: "hips",
        useFirstFramePosition: true,
        preserveBonePositions: false,
      })
    );
    pushAttempt("skeletonutils-root-reversed", "root", () =>
      SkeletonUtils.retargetClip(modelRoot, sourceResult.skeleton, activeStageClip, {
        names: namesSourceToTarget,
        hip: "hips",
        useFirstFramePosition: true,
        preserveBonePositions: false,
      })
    );
  }

  const sourceRootBoneName = sourceResult.skeleton.bones?.[0]?.name || "hips";
  pushAttempt("rename-fallback-bones", "skinned", () =>
    buildRenamedClip(activeStageClip, names, sourceRootBoneName, "bones")
  );
  pushAttempt("rename-fallback-object", "root", () =>
    buildRenamedClip(activeStageClip, names, sourceRootBoneName, "object")
  );

  return { retargetAttempts, attemptDebug };
}

export function selectRetargetAttempt({
  retargetAttempts,
  cachedRigProfile,
  preferSkeletonOnRenameFallback,
  modelSkinnedMeshes,
  modelRoot,
  modelSkinnedMesh,
  retargetTargetBones,
  sourceResult,
  mixer,
  buildBindingsForAttempt,
  clipUsesBonesSyntax,
  resolvedTrackCountForTarget,
  probeMotionForBindings,
  computePoseMatchError,
  buildCanonicalBoneMap,
  canonicalPoseSignature,
  attemptPriority,
}) {
  const rankedAttempts = [...retargetAttempts].sort((a, b) => {
    if (b.resolvedTracks !== a.resolvedTracks) return b.resolvedTracks - a.resolvedTracks;
    return b.clip.tracks.length - a.clip.tracks.length;
  });
  const primaryAttempt = rankedAttempts[0];
  const skeletonSkinned =
    rankedAttempts.find((attempt) => attempt.label === "skeletonutils-skinnedmesh") || null;
  const skeletonSkinnedReversed =
    rankedAttempts.find((attempt) => attempt.label === "skeletonutils-skinnedmesh-reversed") || null;
  const fallbackObject =
    rankedAttempts.find((attempt) => attempt.label === "rename-fallback-object") || null;
  const fallbackBones =
    rankedAttempts.find((attempt) => attempt.label === "rename-fallback-bones") || null;
  const candidateAttempts = [
    primaryAttempt,
    skeletonSkinned,
    skeletonSkinnedReversed,
    fallbackObject,
    fallbackBones,
  ].filter(
    (attempt, index, arr) =>
      attempt &&
      arr.findIndex((value) => value && value.label === attempt.label) === index
  );

  const selectionDebug = [];
  let selectedAttempt = null;
  let selectedBindings = null;
  let selectedProbe = null;
  let selectedPoseError = Number.POSITIVE_INFINITY;

  for (const attempt of candidateAttempts) {
    const bindings = buildBindingsForAttempt({
      attempt,
      clip: attempt.clip,
      modelSkinnedMeshes,
      modelRoot,
      modelSkinnedMesh,
      clipUsesBonesSyntax,
      resolvedTrackCountForTarget,
    });
    if (!bindings.mixers.length) {
      selectionDebug.push({
        label: attempt.label,
        bindingRoot: attempt.bindingRoot,
        tracks: attempt.clip.tracks.length,
        resolvedTracks: attempt.resolvedTracks,
        mixers: 0,
        probeAngle: 0,
        probePos: 0,
        probeScore: 0,
        sampleTime: 0,
        poseError: Number.POSITIVE_INFINITY,
        ok: false,
      });
      continue;
    }

    const probe = probeMotionForBindings({
      bindings,
      clip: attempt.clip,
      modelSkinnedMesh,
      modelRoot,
      targetBones: retargetTargetBones,
    });
    const poseError = computePoseMatchError({
      bindings,
      sampleTime: probe.sampleTime,
      modelSkinnedMesh,
      targetBones: retargetTargetBones,
      sourceResult,
      mixer,
      modelRoot,
      buildCanonicalBoneMap,
      canonicalPoseSignature,
    });
    selectionDebug.push({
      label: attempt.label,
      bindingRoot: attempt.bindingRoot,
      tracks: attempt.clip.tracks.length,
      resolvedTracks: attempt.resolvedTracks,
      mixers: bindings.mixers.length,
      probeAngle: Number(probe.maxAngle.toFixed(6)),
      probePos: Number(probe.maxPos.toFixed(6)),
      probeScore: Number(probe.score.toFixed(6)),
      sampleTime: Number(probe.sampleTime.toFixed(3)),
      poseError: Number.isFinite(poseError)
        ? Number(poseError.toFixed(6))
        : Number.POSITIVE_INFINITY,
      ok: true,
    });

    const better = (() => {
      if (!selectedAttempt) return true;
      const resolvedDiff = attempt.resolvedTracks - selectedAttempt.resolvedTracks;
      if (resolvedDiff > 2) return true;
      if (resolvedDiff < -2) return false;
      const poseDiff = selectedPoseError - poseError;
      if (Number.isFinite(poseDiff) && Math.abs(poseDiff) > 1e-4) {
        return poseDiff > 0;
      }
      const priorityDiff =
        attemptPriority(attempt.label) - attemptPriority(selectedAttempt.label);
      if (priorityDiff > 0) return true;
      if (priorityDiff < 0) return false;
      return probe.score > selectedProbe.score + 1e-7;
    })();

    if (better) {
      selectedAttempt = attempt;
      selectedBindings = bindings;
      selectedProbe = probe;
      selectedPoseError = poseError;
    }
  }

  const preferredMode = String(cachedRigProfile?.preferredMode || "").trim();
  if (preferredMode) {
    const preferredAttempt =
      candidateAttempts.find((attempt) => attempt?.label === preferredMode) || null;
    if (preferredAttempt && preferredAttempt.label !== selectedAttempt?.label) {
      const preferredBindings = buildBindingsForAttempt({
        attempt: preferredAttempt,
        clip: preferredAttempt.clip,
        modelSkinnedMeshes,
        modelRoot,
        modelSkinnedMesh,
        clipUsesBonesSyntax,
        resolvedTrackCountForTarget,
      });
      if (preferredBindings.mixers.length) {
        const preferredProbe = probeMotionForBindings({
          bindings: preferredBindings,
          clip: preferredAttempt.clip,
          modelSkinnedMesh,
          modelRoot,
          targetBones: retargetTargetBones,
        });
        const preferredPoseError = computePoseMatchError({
          bindings: preferredBindings,
          sampleTime: preferredProbe.sampleTime,
          modelSkinnedMesh,
          targetBones: retargetTargetBones,
          sourceResult,
          mixer,
          modelRoot,
          buildCanonicalBoneMap,
          canonicalPoseSignature,
        });
        selectionDebug.push({
          label: preferredAttempt.label,
          bindingRoot: preferredAttempt.bindingRoot,
          tracks: preferredAttempt.clip.tracks.length,
          resolvedTracks: preferredAttempt.resolvedTracks,
          mixers: preferredBindings.mixers.length,
          probeAngle: Number(preferredProbe.maxAngle.toFixed(6)),
          probePos: Number(preferredProbe.maxPos.toFixed(6)),
          probeScore: Number(preferredProbe.score.toFixed(6)),
          sampleTime: Number(preferredProbe.sampleTime.toFixed(3)),
          poseError: Number.isFinite(preferredPoseError)
            ? Number(preferredPoseError.toFixed(6))
            : Number.POSITIVE_INFINITY,
          ok: true,
          forcedByProfile: true,
        });
        selectedAttempt = preferredAttempt;
        selectedBindings = preferredBindings;
        selectedProbe = preferredProbe;
        selectedPoseError = preferredPoseError;
      }
    }
  }

  if (
    preferSkeletonOnRenameFallback &&
    selectedAttempt?.label?.startsWith("rename-fallback") &&
    skeletonSkinned
  ) {
    const skeletonBindings = buildBindingsForAttempt({
      attempt: skeletonSkinned,
      clip: skeletonSkinned.clip,
      modelSkinnedMeshes,
      modelRoot,
      modelSkinnedMesh,
      clipUsesBonesSyntax,
      resolvedTrackCountForTarget,
    });
    if (skeletonBindings.mixers.length) {
      const skeletonProbe = probeMotionForBindings({
        bindings: skeletonBindings,
        clip: skeletonSkinned.clip,
        modelSkinnedMesh,
        modelRoot,
        targetBones: retargetTargetBones,
      });
      const skeletonPoseError = computePoseMatchError({
        bindings: skeletonBindings,
        sampleTime: skeletonProbe.sampleTime,
        modelSkinnedMesh,
        targetBones: retargetTargetBones,
        sourceResult,
        mixer,
        modelRoot,
        buildCanonicalBoneMap,
        canonicalPoseSignature,
      });
      const resolvedGap = selectedAttempt.resolvedTracks - skeletonSkinned.resolvedTracks;
      const poseGap = skeletonPoseError - selectedPoseError;
      const shouldPreferSkeleton =
        resolvedGap <= 2 &&
        (!Number.isFinite(skeletonPoseError) ||
          !Number.isFinite(selectedPoseError) ||
          poseGap <= 0.35);
      if (shouldPreferSkeleton) {
        selectedAttempt = skeletonSkinned;
        selectedBindings = skeletonBindings;
        selectedProbe = skeletonProbe;
        selectedPoseError = skeletonPoseError;
      }
    }
  }

  return {
    selectedAttempt,
    selectedBindings,
    selectedProbe,
    selectedPoseError,
    selectionDebug,
    preferredMode,
  };
}
