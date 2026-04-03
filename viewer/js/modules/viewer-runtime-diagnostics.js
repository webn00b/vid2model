export function createViewerRuntimeDiagnostics({
  windowRef,
  diag,
  isVerboseDiagMode,
  collectLimbDiagnostics,
  dumpRetargetAlignmentDiagnostics,
  dumpRestCorrectionLog,
  buildLegChainDiagnostics,
  dumpLegChainDiagLog,
  buildArmChainDiagnostics,
  dumpArmChainDiagLog,
  buildTorsoChainDiagnostics,
  dumpTorsoChainDiagLog,
  buildFootChainDiagnostics,
  dumpFootChainDiagLog,
  dumpFootCorrectionDebug,
  getSkeletonCanonicalRows,
  resolveSkeletonDumpCanonicals,
  getRetargetTargetBones,
  getRetargetStage,
  getSourceOverlay,
  overlayUpAxis,
}) {
  function logModelBones(rows = []) {
    windowRef.__vid2modelModelBones = rows;
    if (isVerboseDiagMode()) {
      console.log("[vid2model/model-bones] total:", rows.length);
      console.table(rows);
    }
  }

  function publishRetargetDiagnostics({
    retargetStage,
    names,
    sourceBones = [],
    targetBones = [],
    clip,
    selectedAttempt,
    selectedModeLabel,
    selectedProbe,
    selectedPoseError,
    liveRetarget,
    activeMeshCount,
    attemptDebug,
    selectionDebug,
    cachedRigProfile,
    rigProfileSaved,
    mappedPairs,
    sourceMatched,
    matched,
    canonicalCandidates,
    unmatchedHumanoid = [],
    unmatchedTargetBones = [],
    postRetargetPoseError,
    lowerBodyPostError,
    lowerBodyRotError,
    rootYawCorrection,
    modelSkinnedMesh,
    sourceResult,
  }) {
    const sourceTotal = sourceBones.length;
    const limbDiag = collectLimbDiagnostics(targetBones, sourceBones, names, clip);
    const unmatched = unmatchedHumanoid.slice(0, 6);

    windowRef.__vid2modelDebug = {
      stage: retargetStage,
      names,
      sourceBones: sourceBones.map((b) => b.name),
      targetBones: targetBones.map((b) => b.name),
      clipTracks: clip.tracks.map((t) => t.name),
      attemptDebug,
      bestMode: selectedModeLabel,
      selectionDebug,
      motionProbe: selectedProbe,
      poseError: selectedPoseError,
      liveRetarget: liveRetarget
        ? {
            pairs: liveRetarget.pairs.length,
            skeletons: liveRetarget.uniqueSkeletons.length,
            posScale: liveRetarget.posScale,
            calibratedPairs: liveRetarget.calibratedPairs || 0,
            yawOffsetDeg: Number((liveRetarget.yawOffset * 180 / Math.PI).toFixed(2)),
          }
        : null,
      activeMeshCount,
    };

    windowRef.__vid2modelUnmatchedTargetBones = unmatchedTargetBones;

    diag("retarget-map-details", {
      stage: retargetStage,
      totalTargetBones: targetBones.length,
      mappedTargetBones: targetBones.length - unmatchedTargetBones.length,
      unmappedTargetBones: unmatchedTargetBones.length,
      sample: unmatchedTargetBones.slice(0, 20),
    });

    diag("retarget-result", {
      stage: retargetStage,
      mode: selectedModeLabel,
      tracks: clip.tracks.length,
      resolvedTracks: selectedAttempt.resolvedTracks,
      activeMeshes: activeMeshCount,
      motionProbe: selectedProbe
        ? {
            sampleTime: Number(selectedProbe.sampleTime.toFixed(3)),
            maxAngle: Number(selectedProbe.maxAngle.toFixed(6)),
            maxPos: Number(selectedProbe.maxPos.toFixed(6)),
            score: Number(selectedProbe.score.toFixed(6)),
          }
        : null,
      poseError: Number.isFinite(selectedPoseError) ? Number(selectedPoseError.toFixed(6)) : null,
      liveRetarget: liveRetarget
        ? {
            pairs: liveRetarget.pairs.length,
            skeletons: liveRetarget.uniqueSkeletons.length,
            posScale: Number(liveRetarget.posScale.toFixed(6)),
            calibratedPairs: liveRetarget.calibratedPairs || 0,
            yawOffsetDeg: Number((liveRetarget.yawOffset * 180 / Math.PI).toFixed(2)),
          }
        : null,
      unmatched,
    });

    diag("retarget-summary", {
      stage: retargetStage,
      mode: selectedModeLabel,
      mappedPairs,
      sourceMatched,
      sourceTotal,
      humanoidMatched: canonicalCandidates > 0 ? `${matched}/${canonicalCandidates}` : "n/a",
      tracks: clip.tracks.length,
      resolvedTracks: selectedAttempt.resolvedTracks,
      poseError: Number.isFinite(selectedPoseError) ? Number(selectedPoseError.toFixed(4)) : null,
      postPoseError: Number.isFinite(postRetargetPoseError) ? Number(postRetargetPoseError.toFixed(4)) : null,
      lowerBodyPostError: Number.isFinite(lowerBodyPostError) ? Number(lowerBodyPostError.toFixed(4)) : null,
      lowerBodyRotError: Number.isFinite(lowerBodyRotError) ? Number(lowerBodyRotError.toFixed(2)) : null,
      rootYawDeg: Number((rootYawCorrection * 180 / Math.PI).toFixed(2)),
      yawOffsetDeg: liveRetarget ? Number((liveRetarget.yawOffset * 180 / Math.PI).toFixed(2)) : 0,
      calibratedPairs: liveRetarget ? liveRetarget.calibratedPairs || 0 : 0,
      rigProfile: cachedRigProfile ? (cachedRigProfile.source || "hit") : "miss",
      rigProfileSaved,
      liveDelta: !!liveRetarget,
    });

    dumpRestCorrectionLog(`stage=${retargetStage} mode=${selectedModeLabel}`);
    buildLegChainDiagnostics({
      reason: `stage=${retargetStage} mode=${selectedModeLabel}`,
      targetBones,
      sourceBones,
      names: windowRef.__vid2modelDebug?.names || {},
    });
    dumpLegChainDiagLog(`stage=${retargetStage} mode=${selectedModeLabel}`);
    buildArmChainDiagnostics({
      reason: `stage=${retargetStage} mode=${selectedModeLabel}`,
      targetBones,
      sourceBones,
      names: windowRef.__vid2modelDebug?.names || {},
    });
    dumpArmChainDiagLog(`stage=${retargetStage} mode=${selectedModeLabel}`);
    buildTorsoChainDiagnostics({
      reason: `stage=${retargetStage} mode=${selectedModeLabel}`,
      targetBones,
      sourceBones,
      names: windowRef.__vid2modelDebug?.names || {},
    });
    dumpTorsoChainDiagLog(`stage=${retargetStage} mode=${selectedModeLabel}`);
    buildFootChainDiagnostics({
      reason: `stage=${retargetStage} mode=${selectedModeLabel}`,
      targetBones,
      sourceBones,
      names: windowRef.__vid2modelDebug?.names || {},
    });
    dumpFootChainDiagLog(`stage=${retargetStage} mode=${selectedModeLabel}`);
    dumpFootCorrectionDebug(`stage=${retargetStage} mode=${selectedModeLabel}`);

    diag("retarget-limbs", {
      stage: retargetStage,
      mode: selectedModeLabel,
      issues: limbDiag.issuesCount,
      total: limbDiag.total,
      details: limbDiag.issues.slice(0, 12),
    });

    const alignment = dumpRetargetAlignmentDiagnostics({
      reason: "auto-retarget",
      modelSkinnedMesh,
      sourceResult,
      names: windowRef.__vid2modelDebug?.names || {},
      sourceOverlay: getSourceOverlay(),
      overlayUpAxis,
      windowRef,
    });
    if (alignment) {
      diag("retarget-alignment", {
        stage: retargetStage,
        compared: alignment.totalCompared,
        avgPosErr: alignment.avgPosErr,
        avgRotErrDeg: alignment.avgRotErrDeg,
        hipsPosErr: alignment.hipsPosErr,
        overlayYawDeg: alignment.overlayYawDeg,
        worstPosition: alignment.worstPosition.slice(0, 5),
      });
    }
  }

  function dumpAlignment(reason, { modelSkinnedMesh, sourceResult, names }) {
    return dumpRetargetAlignmentDiagnostics({
      reason,
      modelSkinnedMesh,
      sourceResult,
      names,
      sourceOverlay: getSourceOverlay(),
      overlayUpAxis,
      windowRef,
    });
  }

  function dumpSkeleton(scope = "legs", { sourceBones = [], targetBones = [], names = {} } = {}) {
    const canonicals = resolveSkeletonDumpCanonicals(scope);
    const rows = getSkeletonCanonicalRows({
      targetBones,
      sourceBones,
      names,
      canonicals,
    });
    windowRef.__vid2modelSkeletonDump = rows;
    console.log("[vid2model/diag] skeleton-dump", {
      scope,
      total: rows.length,
      stage: getRetargetStage(),
    });
    console.table(rows);
    return rows;
  }

  return {
    logModelBones,
    publishRetargetDiagnostics,
    dumpAlignment,
    dumpSkeleton,
  };
}
