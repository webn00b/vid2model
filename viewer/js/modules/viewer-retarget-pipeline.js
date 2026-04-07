import * as THREE from "three";
import { RETARGET_BODY_CORE_CANONICAL, ARM_REFINEMENT_CANONICAL } from "./retarget-constants.js";
import { canonicalBoneKey } from "./bone-utils.js";
import {
  buildStageSourceClip,
  scaleClipRotationsByCanonical,
  buildRetargetMap,
  buildCanonicalBoneMap,
} from "./retarget-helpers.js";
import { collectRetargetAttempts, selectRetargetAttempt } from "./viewer-retarget-attempts.js";
import { buildVrmDirectBodyPlan } from "./retarget-vrm.js";
import {
  buildRootYawCandidates,
  collectAlignmentDiagnostics,
  computeHipsYawError,
  getLiveDeltaOverride,
  summarizeSourceRootYawClip,
} from "./retarget-analysis.js";
import { evaluateRootYawCandidates, probeMotionForBindings, computePoseMatchError, buildBindingsForAttempt } from "./retarget-eval.js";
import {
  buildBodyLengthCalibration,
  buildArmRefinementCalibration,
  buildFingerLengthCalibration,
  applyBoneLengthCalibration,
  resetBoneLengthCalibration,
  restoreBonePositionSnapshot,
  snapshotCanonicalBonePositions,
} from "./retarget-calibration.js";

export function runRetargetPipeline(ctx) {
  const phase1 = phase1_prepareMapping(ctx);
  if (!phase1) return null;

  const phase2 = phase2_selectAttempt(ctx, phase1);
  if (!phase2) return null;

  const phase3 = phase3_rootYawCorrection(ctx, phase1, phase2);
  if (!phase3) return null;

  const phase4 = phase4_calibration(ctx, phase1, phase2, phase3);
  if (!phase4) return null;

  phase5_finalize(ctx, phase1, phase2, phase3, phase4);

  return { phase1, phase2, phase3, phase4 };
}

function phase1_prepareMapping(ctx) {
  const {
    sourceResult, getRetargetStage, loadRigProfile, modelRigFingerprint, modelLabel,
    publishRigProfileState, buildRigProfileState, getCanonicalFilterForStage,
    getBodyMetricCanonicalFilter, scaleClipRotationsByCanonicalFn, getRetargetTargetBones,
    resetModelRootOrientation, shouldUseVrmDirectBody, maybeApplyTopologyFallback,
    applyRigProfileNames, maybeSwapMirroredHumanoidSides, maybeSwapArmSidesByChain, diag,
    resetRestCorrectionLog, resetLegChainDiagLog, resetArmChainDiagLog,
    resetTorsoChainDiagLog, resetFootChainDiagLog,
  } = ctx;

  resetRestCorrectionLog();
  resetLegChainDiagLog();
  resetArmChainDiagLog();
  resetTorsoChainDiagLog();
  resetFootChainDiagLog();
  resetModelRootOrientation();

  const retargetStage = getRetargetStage();
  const cachedRigProfile = loadRigProfile(modelRigFingerprint, retargetStage, modelLabel);
  publishRigProfileState(buildRigProfileState(cachedRigProfile, {
    modelFingerprint: modelRigFingerprint,
    modelLabel,
    stage: retargetStage,
    saved: false,
  }));

  const canonicalFilter = getCanonicalFilterForStage(retargetStage, cachedRigProfile);
  const bodyMetricCanonicalFilter = getBodyMetricCanonicalFilter(retargetStage, cachedRigProfile);

  const stageClip = buildStageSourceClip(
    sourceResult.clip,
    sourceResult.skeleton.bones,
    retargetStage,
    canonicalFilter
  );
  if (!stageClip) {
    ctx.setStatus(`Retarget failed: no source tracks for stage "${retargetStage}".`);
    diag("retarget-fail", { reason: "empty_stage_clip", stage: retargetStage });
    return null;
  }

  const activeStageClip = scaleClipRotationsByCanonicalFn(
    stageClip,
    sourceResult.skeleton.bones,
    cachedRigProfile?.rotationScaleByCanonical || null
  ) || stageClip;

  const retargetTargetBones = getRetargetTargetBones(retargetStage).filter((bone) => {
    if (!canonicalFilter) return true;
    return canonicalFilter.has(canonicalBoneKey(bone.name) || "");
  });

  const normalizedDirectBones = getRetargetTargetBones(retargetStage, { preferNormalized: true }).filter((bone) => {
    if (!canonicalFilter) return true;
    return canonicalFilter.has(canonicalBoneKey(bone.name) || "");
  });
  const directRetargetTargetBones = normalizedDirectBones.length ? normalizedDirectBones : retargetTargetBones;

  const preferVrmDirectBody =
    retargetStage === "body" &&
    directRetargetTargetBones.length > 0 &&
    (shouldUseVrmDirectBody() || cachedRigProfile?.preferVrmDirectBody === true);

  const preferSkeletonOnRenameFallback = !!cachedRigProfile?.preferSkeletonOnRenameFallback;
  const preferAggressiveLiveDelta = !!cachedRigProfile?.preferAggressiveLiveDelta;

  const initialMap = buildRetargetMap(
    retargetTargetBones,
    sourceResult.skeleton.bones,
    { canonicalFilter }
  );

  const topologyFallback = maybeApplyTopologyFallback(
    ctx.modelSkinnedMesh,
    sourceResult.skeleton.bones,
    canonicalFilter,
    initialMap
  );

  const profiledMapResult = applyRigProfileNames(
    topologyFallback.result,
    cachedRigProfile,
    retargetTargetBones,
    sourceResult.skeleton.bones,
    canonicalFilter
  );

  const mirrorSwapMode = String(cachedRigProfile?.mirrorSwap || "").trim().toLowerCase();
  const allowMirrorSwap = mirrorSwapMode !== "disable";
  const mirroredMapResult = allowMirrorSwap
    ? maybeSwapMirroredHumanoidSides(profiledMapResult, retargetTargetBones, sourceResult.skeleton.bones, canonicalFilter)
    : { ...profiledMapResult, mirroredSidesApplied: false };

  const armSideSwapMode = String(cachedRigProfile?.armSideSwap || "").trim().toLowerCase();
  const allowArmSideSwap = armSideSwapMode !== "disable";
  const activeMapResult = allowArmSideSwap
    ? maybeSwapArmSidesByChain(mirroredMapResult, retargetTargetBones, sourceResult.skeleton.bones, canonicalFilter)
    : { ...mirroredMapResult, mirroredArmSidesApplied: false, armSideSwapScore: null };

  const { names, matched, unmatchedSample, canonicalCandidates, unmatchedHumanoid, sourceMatched } = activeMapResult;
  const mappedPairs = Object.keys(names).length;

  diag("retarget-input", {
    stage: retargetStage,
    sourceBones: sourceResult.skeleton.bones.length,
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
    stage: retargetStage,
    attempted: topologyFallback.attempted,
    applied: topologyFallback.applied,
    reason: topologyFallback.reason,
    inferredRenames: topologyFallback.inferredRenames || 0,
    before: topologyFallback.before || null,
    after: topologyFallback.after || null,
    sample: topologyFallback.sample || [],
  });

  return {
    retargetStage, cachedRigProfile, canonicalFilter, bodyMetricCanonicalFilter,
    activeStageClip, retargetTargetBones, directRetargetTargetBones,
    preferVrmDirectBody, preferSkeletonOnRenameFallback, preferAggressiveLiveDelta,
    names, matched, unmatchedSample, canonicalCandidates, unmatchedHumanoid,
    sourceMatched, mappedPairs, allowMirrorSwap, mirrorSwapMode,
    allowArmSideSwap, armSideSwapMode, activeMapResult,
  };
}

function phase2_selectAttempt(ctx, phase1) {
  const {
    retargetStage, cachedRigProfile, activeStageClip, names,
    retargetTargetBones, preferSkeletonOnRenameFallback, canonicalCandidates,
    unmatchedHumanoid, unmatchedSample,
  } = phase1;

  const {
    modelSkinnedMesh, modelSkinnedMeshes, modelRoot, sourceResult,
    SkeletonUtils, buildRenamedClip, resolvedTrackCountAcrossMeshes,
    resolvedTrackCountForTarget, shortErr, selectRetargetAttemptFn,
    buildBindingsForAttempt, clipUsesBonesSyntax, probeMotionForBindingsFn,
    computePoseMatchErrorFn, buildCanonicalBoneMapFn, canonicalPoseSignatureFn,
    attemptPriorityFn, mixer, diag, setStatus,
  } = ctx;

  const { retargetAttempts, attemptDebug } = collectRetargetAttempts({
    modelSkinnedMesh, modelSkinnedMeshes, modelRoot, sourceResult,
    activeStageClip, names, SkeletonUtils, buildRenamedClip,
    resolvedTrackCountAcrossMeshes, resolvedTrackCountForTarget, shortErr,
  });

  ctx.setModelMixers([]);
  ctx.setModelActions([]);
  ctx.setModelMixer(null);
  ctx.setModelAction(null);
  ctx.setLiveRetarget(null);

  if (!retargetAttempts.length) {
    setStatus("Retarget failed: 0 tracks produced. Bone names do not match.");
    diag("retarget-fail", { reason: "no_tracks", stage: retargetStage, unmatched: (unmatchedHumanoid.length ? unmatchedHumanoid : unmatchedSample).slice(0, 8) });
    return null;
  }

  const selection = selectRetargetAttemptFn({
    retargetAttempts, cachedRigProfile, preferSkeletonOnRenameFallback,
    modelSkinnedMeshes, modelRoot, modelSkinnedMesh, retargetTargetBones,
    sourceResult, mixer, buildBindingsForAttempt, clipUsesBonesSyntax,
    resolvedTrackCountForTarget, probeMotionForBindings: probeMotionForBindingsFn,
    computePoseMatchError: computePoseMatchErrorFn,
    buildCanonicalBoneMap: buildCanonicalBoneMapFn,
    canonicalPoseSignature: canonicalPoseSignatureFn,
    attemptPriority: attemptPriorityFn,
  });

  if (!selection.selectedBindings || !selection.selectedBindings.mixers.length) {
    setStatus("Retarget failed: clip has no resolved tracks on model skeleton.");
    diag("retarget-fail", { reason: "no_resolved_tracks", stage: retargetStage });
    return null;
  }

  ctx.setModelMixers(selection.selectedBindings.mixers);
  ctx.setModelActions(selection.selectedBindings.actions);
  ctx.setModelMixer(selection.selectedBindings.mixers[0]);
  ctx.setModelAction(selection.selectedBindings.actions[0]);

  const clip = selection.selectedAttempt.clip;
  let selectedModeLabel = selection.selectedAttempt.label;
  const isRenameFallback = selectedModeLabel.startsWith("rename-fallback");

  const rawFacingYaw = ctx.estimateFacingYawOffset(sourceResult.skeleton.bones, retargetTargetBones);
  const strongFacingMismatch = Math.abs(rawFacingYaw) > THREE.MathUtils.degToRad(100);
  const weakMotion = !!selection.selectedProbe && selection.selectedProbe.score < 0.5;
  const highPoseError = Number.isFinite(selection.selectedPoseError) && selection.selectedPoseError > 0.6;
  const selectedIsSkeletonUtils = selectedModeLabel.startsWith("skeletonutils");
  const fullHumanoidMatch = canonicalCandidates > 0 && phase1.matched === canonicalCandidates;
  const severeFacingPoseMismatch = strongFacingMismatch && highPoseError;

  const autoUseLiveDelta =
    isRenameFallback ||
    (selectedIsSkeletonUtils && phase1.preferAggressiveLiveDelta && (severeFacingPoseMismatch || (!fullHumanoidMatch && (highPoseError || strongFacingMismatch)))) ||
    (!selectedIsSkeletonUtils && (strongFacingMismatch || weakMotion || highPoseError));

  const profileForceLiveDelta = typeof cachedRigProfile?.forceLiveDelta === "boolean" ? cachedRigProfile.forceLiveDelta : null;
  const forcedLiveDelta = getLiveDeltaOverride(window);
  const useLiveDelta = forcedLiveDelta === null ? (profileForceLiveDelta === null ? autoUseLiveDelta : profileForceLiveDelta) : forcedLiveDelta;

  diag("retarget-live-delta", {
    stage: retargetStage, selectedMode: selectedModeLabel, selectedIsSkeletonUtils,
    profilePolicy: { preferSkeletonOnRenameFallback, preferAggressiveLiveDelta: phase1.preferAggressiveLiveDelta },
    fullHumanoidMatch, preferredMode: selection.preferredMode || null, autoUseLiveDelta,
    profileForced: profileForceLiveDelta, forced: forcedLiveDelta, useLiveDelta,
    reasons: { isRenameFallback, strongFacingMismatch, weakMotion, highPoseError, severeFacingPoseMismatch },
  });

  return {
    selectedAttempt: selection.selectedAttempt, selectedBindings: selection.selectedBindings,
    selectedProbe: selection.selectedProbe, selectedPoseError: selection.selectedPoseError,
    selectionDebug: selection.selectionDebug, preferredMode: selection.preferredMode,
    clip, selectedModeLabel, isRenameFallback, rawFacingYaw, strongFacingMismatch,
    weakMotion, highPoseError, selectedIsSkeletonUtils, fullHumanoidMatch,
    severeFacingPoseMismatch, autoUseLiveDelta, profileForceLiveDelta,
    forcedLiveDelta, useLiveDelta, attemptDebug,
  };
}

function phase3_rootYawCorrection(ctx, phase1, phase2) {
  const {
    retargetStage, cachedRigProfile, names, retargetTargetBones,
    preferVrmDirectBody, directRetargetTargetBones,
  } = phase1;
  const {
    selectedAttempt, selectedProbe, clip, selectedModeLabel: initialModeLabel,
    isRenameFallback, rawFacingYaw, strongFacingMismatch, selectedIsSkeletonUtils,
    useLiveDelta,
  } = phase2;

  let selectedModeLabel = initialModeLabel;
  let rootYawCorrection = 0;

  if (preferVrmDirectBody) {
    const directPlan = buildVrmDirectBodyPlan({
      targetBones: directRetargetTargetBones,
      sourceBones: ctx.sourceResult.skeleton.bones,
      namesTargetToSource: names,
      mixer: ctx.mixer,
      modelRoot: ctx.modelRoot,
      buildRestOrientationCorrection: ctx.buildRestOrientationCorrection,
      profile: cachedRigProfile,
    });
    if (directPlan && directPlan.pairs.length > 0) {
      ctx.setLiveRetarget(directPlan);
      ctx.applyLiveRetargetPose(directPlan);
      ctx.setModelMixers([]);
      ctx.setModelActions([]);
      ctx.setModelMixer(null);
      ctx.setModelAction(null);
      selectedModeLabel = "vrm-humanoid-direct";
    }
  }

  if (!ctx.liveRetarget && selectedIsSkeletonUtils && !useLiveDelta) {
    const profileRootYawDeg = Number.isFinite(cachedRigProfile?.rootYawDeg) ? cachedRigProfile.rootYawDeg : null;
    const sourceClipYawSummary = summarizeSourceRootYawClip(clip);
    const sourceClipYawLooksCentered = !!sourceClipYawSummary.looksCentered;
    const rawYawLooksAligned = Math.abs(rawFacingYaw) < THREE.MathUtils.degToRad(30);

    if (Number.isFinite(profileRootYawDeg)) {
      rootYawCorrection = ctx.applyModelRootYaw(THREE.MathUtils.degToRad(profileRootYawDeg));
    } else {
      const yawCandidates = buildRootYawCandidates(rawFacingYaw, ctx.quantizeFacingYaw, { sourceClipYawSummary });
      const yawEval = ctx.evaluateRootYawCandidates({
        candidates: yawCandidates,
        sampleTime: selectedProbe?.sampleTime || 0,
        namesTargetToSource: names,
        sourceClip: clip,
        modelRoot: ctx.modelRoot,
        modelMixers: ctx.modelMixers,
        modelSkinnedMesh: ctx.modelSkinnedMesh,
        targetBones: retargetTargetBones,
        sourceResult: ctx.sourceResult,
        mixer: ctx.mixer,
        resetModelRootOrientation: ctx.resetModelRootOrientation,
        applyModelRootYaw: ctx.applyModelRootYaw,
        collectAlignmentDiagnostics: (args) => collectAlignmentDiagnostics({
          ...args,
          sourceOverlay: ctx.getSourceOverlay(),
          overlayUpAxis: ctx.overlayUpAxis,
        }),
      });

      const zeroRow = yawEval.rows.find((r) => Math.abs(r.yawDeg) < 0.01) || null;
      const bestRow = yawEval.rows[0] || null;
      const bestIsLargeFlip = !!bestRow && Math.abs(bestRow.yawDeg) > 120;
      const largeFlipLooksRedundant = !!bestRow && !!zeroRow && sourceClipYawLooksCentered && bestIsLargeFlip && bestRow.score + 0.08 >= zeroRow.score;
      const shouldUseBest = !!bestRow && !(rawYawLooksAligned && bestIsLargeFlip) && !largeFlipLooksRedundant && (!zeroRow || bestRow.score + 0.03 < zeroRow.score || (Number.isFinite(bestRow.hipsPosErr) && Number.isFinite(zeroRow.hipsPosErr) && bestRow.hipsPosErr + 0.03 < zeroRow.hipsPosErr));

      rootYawCorrection = ctx.applyModelRootYaw(shouldUseBest ? yawEval.bestYaw : 0);

      if (ctx.hasSourceOverlay()) {
        ctx.setSourceOverlayYaw(0);
        ctx.updateSourceOverlay();
      }

      const hipsYawError = computeHipsYawError(retargetTargetBones, ctx.sourceResult.skeleton.bones, names);
      let hipsYawCorrection = 0;
      let hipsCorrectionApplied = false;
      let hipsCorrectionEval = null;

      if (Math.abs(hipsYawError) > THREE.MathUtils.degToRad(12) && !(rawYawLooksAligned && Math.abs(hipsYawError) > THREE.MathUtils.degToRad(120))) {
        const correctedYaw = rootYawCorrection - hipsYawError;
        const postEval = ctx.evaluateRootYawCandidates({
          candidates: [rootYawCorrection, correctedYaw],
          sampleTime: selectedProbe?.sampleTime || 0,
          namesTargetToSource: names,
          sourceClip: clip,
          modelRoot: ctx.modelRoot,
          modelMixers: ctx.modelMixers,
          modelSkinnedMesh: ctx.modelSkinnedMesh,
          targetBones: retargetTargetBones,
          sourceResult: ctx.sourceResult,
          mixer: ctx.mixer,
          resetModelRootOrientation: ctx.resetModelRootOrientation,
          applyModelRootYaw: ctx.applyModelRootYaw,
          collectAlignmentDiagnostics: (args) => collectAlignmentDiagnostics({
            ...args,
            sourceOverlay: ctx.getSourceOverlay(),
            overlayUpAxis: ctx.overlayUpAxis,
          }),
        });

        const sameYaw = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b))) < 1e-4;
        const currentRow = postEval.rows.find((r) => sameYaw(r.yawRad, rootYawCorrection)) || null;
        const correctedRow = postEval.rows.find((r) => sameYaw(r.yawRad, correctedYaw)) || null;
        hipsCorrectionEval = { current: currentRow, corrected: correctedRow };

        if (!!currentRow && !!correctedRow && correctedRow.score + 0.03 < currentRow.score) {
          hipsYawCorrection = -hipsYawError;
          rootYawCorrection = ctx.applyModelRootYaw(correctedYaw);
          hipsCorrectionApplied = true;
          if (ctx.hasSourceOverlay()) {
            ctx.setSourceOverlayYaw(0);
            ctx.updateSourceOverlay();
          }
        }
      }

      ctx.diag("retarget-root-yaw", {
        stage: retargetStage,
        rawFacingYawDeg: Number((rawFacingYaw * 180 / Math.PI).toFixed(2)),
        appliedYawDeg: Number((rootYawCorrection * 180 / Math.PI).toFixed(2)),
        hipsYawErrorDeg: Number((hipsYawError * 180 / Math.PI).toFixed(2)),
        hipsYawCorrectionDeg: Number((hipsYawCorrection * 180 / Math.PI).toFixed(2)),
        hipsCorrectionApplied, hipsCorrectionEval, strongFacingMismatch,
        usedProfileYaw: Number.isFinite(profileRootYawDeg),
        sourceClipYawSummary,
        sourceYawCandidatePolicy: { allowSourceFlipCandidates: !sourceClipYawLooksCentered },
      });
    }
  }

  const rebuildLiveRetargetPlan = () => {
    if (preferVrmDirectBody) {
      return buildVrmDirectBodyPlan({
        targetBones: directRetargetTargetBones,
        sourceBones: ctx.sourceResult.skeleton.bones,
        namesTargetToSource: names,
        mixer: ctx.mixer,
        modelRoot: ctx.modelRoot,
        buildRestOrientationCorrection: ctx.buildRestOrientationCorrection,
        profile: cachedRigProfile,
      });
    }
    return ctx.buildLiveRetargetPlan(ctx.modelSkinnedMeshes, ctx.sourceResult.skeleton.bones, names, retargetTargetBones);
  };

  if (!ctx.liveRetarget && useLiveDelta) {
    const livePlan = rebuildLiveRetargetPlan();
    if (livePlan && livePlan.pairs.length > 0) {
      ctx.setLiveRetarget(livePlan);
      if (ctx.hasSourceOverlay()) {
        ctx.setSourceOverlayYaw(rawFacingYaw + livePlan.yawOffset);
        ctx.updateSourceOverlay();
      }
      ctx.applyLiveRetargetPose(livePlan);
      ctx.setModelMixers([]);
      ctx.setModelActions([]);
      ctx.setModelMixer(null);
      ctx.setModelAction(null);
      selectedModeLabel = `${selectedAttempt.label}+live-delta`;
    }
  }

  const sourceTime = ctx.mixer ? ctx.mixer.time : 0;
  const syncTime = clip.duration > 0 ? ((sourceTime % clip.duration) + clip.duration) % clip.duration : 0;
  if (!ctx.liveRetarget) {
    for (const mix of ctx.modelMixers) mix.setTime(syncTime);
  }

  return { rootYawCorrection, selectedModeLabel, rebuildLiveRetargetPlan, syncTime, rawFacingYaw };
}

function phase4_calibration(ctx, phase1, phase2, phase3) {
  const { retargetStage, bodyMetricCanonicalFilter, names, retargetTargetBones } = phase1;
  const { clip } = phase2;
  const { rebuildLiveRetargetPlan, rawFacingYaw } = phase3;

  ctx.setBodyLengthCalibration(null);
  ctx.setArmLengthCalibration(null);
  ctx.setFingerLengthCalibration(null);

  let measureBodyErr = null;
  let bodyErrBaseline = null;

  const buildMeasureBodyErr = (bodyTargetBones) => () => {
    const report = collectAlignmentDiagnostics({
      targetBones: bodyTargetBones.length ? bodyTargetBones : retargetTargetBones,
      sourceBones: ctx.sourceResult.skeleton.bones,
      namesTargetToSource: names,
      sourceClip: clip,
      maxRows: 5,
      overlayYawOverride: 0,
      sourceOverlay: ctx.getSourceOverlay(),
      overlayUpAxis: ctx.overlayUpAxis,
    });
    return Number.isFinite(report?.avgPosErrNorm) ? report.avgPosErrNorm : report?.avgPosErr;
  };

  const runBodyCalibration = (isLiveDelta) => {
    const bodyEvalCanonical = bodyMetricCanonicalFilter;
    const bodyTargetBones = retargetTargetBones.filter((b) => bodyEvalCanonical.has(canonicalBoneKey(b.name) || ""));
    measureBodyErr = buildMeasureBodyErr(bodyTargetBones);

    const bodyErrBefore = measureBodyErr();
    if (Number.isFinite(bodyErrBefore)) bodyErrBaseline = bodyErrBefore;

    const attemptedBodyCalibration = buildBodyLengthCalibration(
      ctx.sourceResult.skeleton.bones,
      ctx.modelSkinnedMesh.skeleton.bones,
      clip,
      buildCanonicalBoneMap
    );

    const filteredBodyCalibration = attemptedBodyCalibration && retargetStage === "body"
      ? { ...attemptedBodyCalibration, entries: attemptedBodyCalibration.entries.filter((e) => bodyMetricCanonicalFilter.has(e.canonical)) }
      : attemptedBodyCalibration;

    const suspiciousBodyScale = !!filteredBodyCalibration && Number.isFinite(filteredBodyCalibration.globalScale) && (filteredBodyCalibration.globalScale < 0.2 || filteredBodyCalibration.globalScale > 5);

    if (!filteredBodyCalibration?.entries?.length || suspiciousBodyScale) {
      if (filteredBodyCalibration?.entries?.length) {
        ctx.diag("retarget-body-calibration", {
          stage: retargetStage, mode: isLiveDelta ? "live-delta" : undefined,
          applied: false, skippedReason: "suspicious-global-scale",
          bodyErrBefore: Number.isFinite(bodyErrBefore) ? Number(bodyErrBefore.toFixed(5)) : null,
          bodyErrAfter: Number.isFinite(bodyErrBefore) ? Number(bodyErrBefore.toFixed(5)) : null,
          metric: retargetStage === "body" ? "body-core" : "body-full",
          bones: filteredBodyCalibration.entries.length,
          globalScale: filteredBodyCalibration.globalScale,
          clampedCount: filteredBodyCalibration.clampedCount,
          sample: filteredBodyCalibration.entries.slice(0, 8).map((e) => ({ canonical: e.canonical, scale: e.scale, rawScale: e.rawScale, sourceLen: e.sourceLen, targetLen: e.targetLen, expectedTargetLen: e.expectedTargetLen })),
        });
      }
      return { kept: false, phase2 };
    }

    const previousLiveRetarget = ctx.liveRetarget;
    applyBoneLengthCalibration(filteredBodyCalibration, ctx.modelRoot);

    if (isLiveDelta) {
      const rebuiltLiveRetarget = rebuildLiveRetargetPlan();
      if (rebuiltLiveRetarget?.pairs?.length) {
        ctx.setLiveRetarget(rebuiltLiveRetarget);
        if (ctx.hasSourceOverlay()) {
          ctx.setSourceOverlayYaw(rawFacingYaw + rebuiltLiveRetarget.yawOffset);
          ctx.updateSourceOverlay();
        }
        ctx.applyLiveRetargetPose(rebuiltLiveRetarget);
      }
      const bodyErrAfter = rebuiltLiveRetarget?.pairs?.length ? measureBodyErr() : null;
      const hasBefore = Number.isFinite(bodyErrBefore);
      const hasAfter = Number.isFinite(bodyErrAfter);
      const keepCalibration = !!rebuiltLiveRetarget?.pairs?.length && hasAfter && (!hasBefore || bodyErrAfter <= bodyErrBefore - 0.005);

      if (!keepCalibration) {
        resetBoneLengthCalibration(filteredBodyCalibration, ctx.modelRoot);
        ctx.setLiveRetarget(previousLiveRetarget);
        if (ctx.hasSourceOverlay()) {
          ctx.setSourceOverlayYaw(rawFacingYaw + previousLiveRetarget.yawOffset);
          ctx.updateSourceOverlay();
        }
        ctx.applyLiveRetargetPose(previousLiveRetarget);
        ctx.setBodyLengthCalibration(null);
      } else {
        ctx.setBodyLengthCalibration(filteredBodyCalibration);
        if (Number.isFinite(bodyErrAfter)) bodyErrBaseline = bodyErrAfter;
        phase2.selectedModeLabel = `${phase2.selectedModeLabel}+body-calib`;
      }

      ctx.diag("retarget-body-calibration", {
        stage: retargetStage, mode: "live-delta", applied: keepCalibration,
        bodyErrBefore: Number.isFinite(bodyErrBefore) ? Number(bodyErrBefore.toFixed(5)) : null,
        bodyErrAfter: Number.isFinite(bodyErrAfter) ? Number(bodyErrAfter.toFixed(5)) : null,
        metric: retargetStage === "body" ? "body-core" : "body-full",
        bones: filteredBodyCalibration.entries.length,
        globalScale: filteredBodyCalibration.globalScale,
        clampedCount: filteredBodyCalibration.clampedCount,
        sample: filteredBodyCalibration.entries.slice(0, 8).map((e) => ({ canonical: e.canonical, scale: e.scale, rawScale: e.rawScale, sourceLen: e.sourceLen, targetLen: e.targetLen, expectedTargetLen: e.expectedTargetLen })),
      });

      return { kept: keepCalibration, phase2 };
    } else {
      const bodyErrAfter = measureBodyErr();
      const hasBefore = Number.isFinite(bodyErrBefore);
      const hasAfter = Number.isFinite(bodyErrAfter);
      const keepCalibration = hasAfter && (!hasBefore || bodyErrAfter <= bodyErrBefore - 0.005);

      if (!keepCalibration) {
        resetBoneLengthCalibration(filteredBodyCalibration, ctx.modelRoot);
        ctx.setBodyLengthCalibration(null);
      } else {
        ctx.setBodyLengthCalibration(filteredBodyCalibration);
        if (Number.isFinite(bodyErrAfter)) bodyErrBaseline = bodyErrAfter;
      }

      ctx.diag("retarget-body-calibration", {
        stage: retargetStage, applied: keepCalibration,
        bodyErrBefore: Number.isFinite(bodyErrBefore) ? Number(bodyErrBefore.toFixed(5)) : null,
        bodyErrAfter: Number.isFinite(bodyErrAfter) ? Number(bodyErrAfter.toFixed(5)) : null,
        metric: retargetStage === "body" ? "body-core" : "body-full",
        bones: filteredBodyCalibration.entries.length,
        globalScale: filteredBodyCalibration.globalScale,
        clampedCount: filteredBodyCalibration.clampedCount,
        sample: filteredBodyCalibration.entries.slice(0, 8).map((e) => ({ canonical: e.canonical, scale: e.scale, rawScale: e.rawScale, sourceLen: e.sourceLen, targetLen: e.targetLen, expectedTargetLen: e.expectedTargetLen })),
      });

      return { kept: keepCalibration, phase2 };
    }
  };

  if (ctx.liveRetarget) {
    runBodyCalibration(true);
  } else {
    runBodyCalibration(false);
  }

  if (!ctx.liveRetarget && retargetStage === "body") {
    const armBaselineSnapshot = snapshotCanonicalBonePositions(ctx.modelSkinnedMesh.skeleton.bones, ARM_REFINEMENT_CANONICAL);
    const armRefine = buildArmRefinementCalibration({
      sourceBones: ctx.sourceResult.skeleton.bones,
      targetBones: ctx.modelSkinnedMesh.skeleton.bones,
      namesTargetToSource: names,
      sourceClip: clip,
      buildCanonicalBoneMap,
      collectAlignmentDiagnostics,
      updateWorld: () => ctx.modelRoot?.updateMatrixWorld(true),
    });

    const armErrBefore = Number.isFinite(bodyErrBaseline) ? bodyErrBaseline : (measureBodyErr ? measureBodyErr() : null);
    let armErrAfter = armErrBefore;
    let keepArmRefine = false;

    if (armRefine?.entries?.length) {
      ctx.setArmLengthCalibration({ entries: armRefine.entries });
      applyBoneLengthCalibration(ctx.armLengthCalibration, ctx.modelRoot);
      armErrAfter = measureBodyErr ? measureBodyErr() : null;
      const hasBefore = Number.isFinite(armErrBefore);
      const hasAfter = Number.isFinite(armErrAfter);
      keepArmRefine = hasAfter && (!hasBefore || armErrAfter <= armErrBefore - 0.003);

      if (!keepArmRefine) {
        restoreBonePositionSnapshot(armBaselineSnapshot, ctx.modelRoot);
        ctx.setArmLengthCalibration(null);
      } else {
        bodyErrBaseline = armErrAfter;
      }
    }

    ctx.diag("retarget-arm-refine", {
      stage: retargetStage, applied: keepArmRefine,
      bodyErrBefore: Number.isFinite(armErrBefore) ? Number(armErrBefore.toFixed(5)) : null,
      bodyErrAfter: Number.isFinite(armErrAfter) ? Number(armErrAfter.toFixed(5)) : null,
      appliedSides: (armRefine?.sides || []).filter((s) => s.applied).length,
      sides: armRefine?.sides || [],
      bones: armRefine?.entries?.length || 0,
    });
  }

  if (!ctx.liveRetarget && retargetStage === "full") {
    ctx.setFingerLengthCalibration(buildFingerLengthCalibration(
      ctx.sourceResult.skeleton.bones,
      ctx.modelSkinnedMesh.skeleton.bones,
      clip,
      buildCanonicalBoneMap
    ));

    if (ctx.fingerLengthCalibration) {
      applyBoneLengthCalibration(ctx.fingerLengthCalibration, ctx.modelRoot);
      ctx.diag("retarget-finger-calibration", {
        stage: retargetStage,
        bones: ctx.fingerLengthCalibration.entries.length,
        globalScale: ctx.fingerLengthCalibration.globalScale,
        clampedCount: ctx.fingerLengthCalibration.clampedCount,
        rawScaleRange: { min: ctx.fingerLengthCalibration.minRawScale, max: ctx.fingerLengthCalibration.maxRawScale },
        sample: ctx.fingerLengthCalibration.entries.slice(0, 8).map((e) => ({ canonical: e.canonical, scale: e.scale, rawScale: e.rawScale, sourceLen: e.sourceLen, targetLen: e.targetLen, expectedTargetLen: e.expectedTargetLen })),
      });
    }
  }

  return { measureBodyErr, bodyErrBaseline };
}

function phase5_finalize(ctx, phase1, phase2, phase3, phase4) {
  const { retargetStage, names, retargetTargetBones, canonicalCandidates, matched, sourceMatched, cachedRigProfile } = phase1;
  const { selectedAttempt, selectedProbe, selectedPoseError, clip, selectedModeLabel, attemptDebug, selectionDebug } = phase2;
  const { rootYawCorrection, syncTime } = phase3;

  const hipsAlign = ctx.alignModelHipsToSource(false);
  if (hipsAlign) ctx.diag("retarget-hips-align", { stage: retargetStage, ...hipsAlign });

  const summaryTargetBones = retargetTargetBones.filter((bone) => {
    const canonical = canonicalBoneKey(bone.name) || "";
    if (!canonical) return false;
    return phase1.canonicalFilter ? phase1.canonicalFilter.has(canonical) : true;
  });

  const postRetargetReport = collectAlignmentDiagnostics({
    targetBones: summaryTargetBones.length ? summaryTargetBones : retargetTargetBones,
    sourceBones: ctx.sourceResult.skeleton.bones,
    namesTargetToSource: names,
    sourceClip: clip,
    maxRows: 5,
    overlayYawOverride: 0,
    sourceOverlay: ctx.getSourceOverlay(),
    overlayUpAxis: ctx.overlayUpAxis,
  });

  const postRetargetPoseError = Number.isFinite(postRetargetReport?.avgPosErrNorm) ? postRetargetReport.avgPosErrNorm : postRetargetReport?.avgPosErr;

  const lowerBodyTargetBones = retargetTargetBones.filter((bone) => RETARGET_BODY_CORE_CANONICAL.has(canonicalBoneKey(bone.name) || ""));
  const lowerBodyReport = collectAlignmentDiagnostics({
    targetBones: lowerBodyTargetBones.length ? lowerBodyTargetBones : retargetTargetBones,
    sourceBones: ctx.sourceResult.skeleton.bones,
    namesTargetToSource: names,
    sourceClip: clip,
    maxRows: 5,
    overlayYawOverride: 0,
    sourceOverlay: ctx.getSourceOverlay(),
    overlayUpAxis: ctx.overlayUpAxis,
  });

  const lowerBodyPostError = Number.isFinite(lowerBodyReport?.avgPosErrNorm) ? lowerBodyReport.avgPosErrNorm : lowerBodyReport?.avgPosErr;
  const lowerBodyRotError = Number.isFinite(lowerBodyReport?.avgRotErrDeg) ? lowerBodyReport.avgRotErrDeg : null;

  ctx.setIsPlaying(true);
  ctx.updateTimelineUi(syncTime);

  const sourceTotal = ctx.sourceResult.skeleton.bones.length;
  const total = retargetTargetBones.length;
  const targetCoverage = canonicalCandidates > 0 ? matched / canonicalCandidates : 0;
  const sourceCoverage = sourceTotal > 0 ? sourceMatched / sourceTotal : 0;
  let lowMatch = "";
  if (targetCoverage < 0.75 || (targetCoverage < 0.9 && sourceCoverage < 0.35)) {
    lowMatch = " low humanoid match, try another model/rig.";
  }
  const candidateInfo = canonicalCandidates > 0 ? `, humanoid targets ${matched}/${canonicalCandidates}` : "";
  const activeMeshCount = ctx.liveRetarget ? ctx.liveRetarget.uniqueSkeletons.length : ctx.modelMixers.length;

  let rigProfileSaved = false;
  if (Number.isFinite(postRetargetPoseError) && postRetargetPoseError <= 0.75) {
    const nextRigProfileCandidate = {
      modelLabel: ctx.modelLabel,
      modelFingerprint: ctx.modelRigFingerprint,
      stage: retargetStage,
      namesTargetToSource: names,
      mode: selectedModeLabel,
      rootYawDeg: Number((rootYawCorrection * 180 / Math.PI).toFixed(2)),
      postPoseError: Number(postRetargetPoseError.toFixed(6)),
      liveRetarget: ctx.liveRetarget ? ctx.exportLiveRetargetProfile(ctx.liveRetarget) : null,
      basedOnBuiltin: cachedRigProfile?.id || cachedRigProfile?.basedOnBuiltin || null,
    };
    ctx.setLatestRigProfileCandidate(nextRigProfileCandidate);
    rigProfileSaved = ctx.saveRigProfile({ ...nextRigProfileCandidate, source: "localStorage" });
  }

  const activeProfile = ctx.loadRigProfile(ctx.modelRigFingerprint, retargetStage, ctx.modelLabel) || cachedRigProfile || null;
  const profileStateLabel = activeProfile?.validationStatus || (activeProfile ? "cached" : "none");
  const profileSourceLabel = activeProfile?.source === "model-analysis-seed" ? "seed" : profileStateLabel;
  const inferredCorrections = Array.isArray(activeProfile?.inferredCorrections) ? activeProfile.inferredCorrections : ctx.buildSeedCorrectionSummary(activeProfile);
  const correctionInfo = inferredCorrections.length ? `, inferred ${inferredCorrections.join("+")}` : "";

  ctx.publishRigProfileState(ctx.buildRigProfileState(activeProfile, {
    modelFingerprint: ctx.modelRigFingerprint,
    modelLabel: ctx.modelLabel,
    stage: retargetStage,
    saved: rigProfileSaved,
  }));

  ctx.setStatus(`Model retargeted [${retargetStage}] (source ${sourceMatched}/${sourceTotal}${candidateInfo}, all ${matched}/${total}, tracks ${clip.tracks.length}, mode ${selectedModeLabel}, active meshes ${activeMeshCount}, profile ${profileSourceLabel}${correctionInfo}).${lowMatch}`);

  const unmatchedTargetBones = retargetTargetBones.map((b) => b.name).filter((name) => !names[name]);
  ctx.publishRetargetDiagnostics({
    retargetStage, names, sourceBones: ctx.sourceResult.skeleton.bones,
    targetBones: retargetTargetBones, clip, selectedAttempt, selectedModeLabel,
    selectedProbe, selectedPoseError, liveRetarget: ctx.liveRetarget,
    activeMeshCount, attemptDebug, selectionDebug, cachedRigProfile, rigProfileSaved,
    mappedPairs: phase1.mappedPairs, sourceMatched, matched, canonicalCandidates,
    unmatchedHumanoid: phase1.unmatchedHumanoid, unmatchedTargetBones,
    postRetargetPoseError, lowerBodyPostError, lowerBodyRotError, rootYawCorrection,
    modelSkinnedMesh: ctx.modelSkinnedMesh, sourceResult: ctx.sourceResult,
  });

  ctx.resetModelRootOrientation();
  if (ctx.liveRetarget) ctx.applyLiveRetargetPose(ctx.liveRetarget);
  if (ctx.sourceResult && ctx.modelSkinnedMesh) ctx.syncSourceDisplayToModel();
}
