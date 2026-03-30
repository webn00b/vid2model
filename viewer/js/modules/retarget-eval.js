import * as THREE from "three";
import { parseTrackName } from "./bone-utils.js";

export function evaluateRootYawCandidates({
  candidates,
  sampleTime,
  namesTargetToSource,
  sourceClip,
  modelRoot,
  modelMixers,
  modelSkinnedMesh,
  sourceResult,
  mixer,
  resetModelRootOrientation,
  applyModelRootYaw,
  collectAlignmentDiagnostics,
}) {
  if (!modelRoot || !modelMixers.length || !modelSkinnedMesh?.skeleton?.bones?.length || !sourceResult?.skeleton?.bones?.length) {
    return { bestYaw: 0, rows: [] };
  }
  const sourceTime = mixer ? mixer.time : 0;
  const modelTimes = modelMixers.map((mix) => mix.time);
  const t = sampleTime > 0 ? sampleTime : 1 / 30;
  const rows = [];
  try {
    for (const yaw of candidates) {
      resetModelRootOrientation();
      const appliedYaw = applyModelRootYaw(yaw);
      if (mixer) mixer.setTime(t);
      for (const mix of modelMixers) {
        mix.setTime(t);
      }
      modelRoot.updateMatrixWorld(true);
      const alignment = collectAlignmentDiagnostics({
        targetBones: modelSkinnedMesh.skeleton.bones,
        sourceBones: sourceResult.skeleton.bones,
        namesTargetToSource,
        sourceClip,
        maxRows: 5,
        overlayYawOverride: 0,
      });
      const avgPosErr = Number.isFinite(alignment.avgPosErrNorm) ? alignment.avgPosErrNorm : alignment.avgPosErr;
      const hips = Number.isFinite(alignment.hipsPosErrNorm)
        ? alignment.hipsPosErrNorm
        : (Number.isFinite(alignment.hipsPosErr) ? alignment.hipsPosErr : avgPosErr);
      const score = avgPosErr + alignment.avgRotErrDeg * 0.0012 + hips * 1.25;
      rows.push({
        yawRad: appliedYaw,
        yawDeg: Number((appliedYaw * 180 / Math.PI).toFixed(2)),
        score: Number(score.toFixed(6)),
        avgPosErr: alignment.avgPosErr,
        avgPosErrNorm: alignment.avgPosErrNorm,
        avgRotErrDeg: alignment.avgRotErrDeg,
        hipsPosErr: alignment.hipsPosErr,
        hipsPosErrNorm: alignment.hipsPosErrNorm,
      });
    }
  } finally {
    if (mixer) mixer.setTime(sourceTime);
    for (let i = 0; i < modelMixers.length; i += 1) {
      modelMixers[i].setTime(modelTimes[i] || 0);
    }
    resetModelRootOrientation();
  }
  rows.sort((a, b) => a.score - b.score);
  const bestYaw = rows.length ? rows[0].yawRad : 0;
  return { bestYaw, rows };
}

export function buildBindingsForAttempt({
  attempt,
  clip,
  modelSkinnedMeshes,
  modelRoot,
  modelSkinnedMesh,
  clipUsesBonesSyntax,
  resolvedTrackCountForTarget,
}) {
  const mixers = [];
  const actions = [];
  const activeBindings = [];
  const useSkinnedBinding =
    attempt.bindingRoot === "skinned" ||
    (attempt.bindingRoot === "auto" && clipUsesBonesSyntax(clip));

  if (useSkinnedBinding) {
    const seenSkeletonIds = new Set();
    for (const mesh of modelSkinnedMeshes) {
      const skeletonId = mesh.skeleton?.uuid || mesh.uuid;
      if (seenSkeletonIds.has(skeletonId)) continue;
      seenSkeletonIds.add(skeletonId);
      const resolvedTracks = resolvedTrackCountForTarget(clip, mesh.skeleton.bones);
      if (resolvedTracks <= 0) continue;
      const mix = new THREE.AnimationMixer(mesh);
      const action = mix.clipAction(clip);
      action.reset();
      action.setEffectiveWeight(1);
      action.setEffectiveTimeScale(1);
      action.play();
      mixers.push(mix);
      actions.push(action);
      activeBindings.push({
        mesh: mesh.name || "(unnamed-skinned-mesh)",
        bones: mesh.skeleton.bones.length,
        resolvedTracks,
      });
    }
  } else {
    const bindingRoot = modelRoot || modelSkinnedMesh;
    if (bindingRoot) {
      const mix = new THREE.AnimationMixer(bindingRoot);
      const action = mix.clipAction(clip);
      action.reset();
      action.setEffectiveWeight(1);
      action.setEffectiveTimeScale(1);
      action.play();
      mixers.push(mix);
      actions.push(action);
      activeBindings.push({
        mesh: bindingRoot.name || "(model-root)",
        bones: modelSkinnedMesh?.skeleton?.bones?.length || 0,
        resolvedTracks: modelSkinnedMesh?.skeleton?.bones
          ? resolvedTrackCountForTarget(clip, modelSkinnedMesh.skeleton.bones)
          : clip.tracks.length,
      });
    }
  }

  return { mixers, actions, activeBindings, useSkinnedBinding };
}

export function probeMotionForBindings({ bindings, clip, modelSkinnedMesh, modelRoot }) {
  if (!bindings.mixers.length || !modelSkinnedMesh?.skeleton?.bones?.length) {
    return { sampleTime: 0, maxAngle: 0, maxPos: 0, score: 0 };
  }

  const trackedBoneNames = new Set();
  for (const track of clip.tracks) {
    const parsed = parseTrackName(track.name);
    if (parsed?.bone) trackedBoneNames.add(parsed.bone);
  }

  let probeBones = modelSkinnedMesh.skeleton.bones.filter((b) => trackedBoneNames.has(b.name));
  if (!probeBones.length) {
    probeBones = modelSkinnedMesh.skeleton.bones;
  }
  probeBones = probeBones.slice(0, 64);

  const sampleTime =
    clip.duration > 0
      ? Math.max(1 / 30, Math.min(clip.duration * 0.35, Math.max(clip.duration - 1e-3, 1 / 30)))
      : 0;
  if (sampleTime <= 0) {
    return { sampleTime: 0, maxAngle: 0, maxPos: 0, score: 0 };
  }

  for (const mix of bindings.mixers) {
    mix.setTime(0);
  }
  modelRoot?.updateMatrixWorld(true);

  const before = probeBones.map((b) => ({
    bone: b,
    q: b.quaternion.clone(),
    p: b.position.clone(),
  }));

  for (const mix of bindings.mixers) {
    mix.setTime(sampleTime);
  }
  modelRoot?.updateMatrixWorld(true);

  let maxAngle = 0;
  let maxPos = 0;
  for (const s of before) {
    maxAngle = Math.max(maxAngle, s.q.angleTo(s.bone.quaternion));
    maxPos = Math.max(maxPos, s.p.distanceTo(s.bone.position));
  }

  for (const mix of bindings.mixers) {
    mix.setTime(0);
  }
  modelRoot?.updateMatrixWorld(true);

  const score = maxAngle * 1000 + maxPos;
  return { sampleTime, maxAngle, maxPos, score };
}

export function computePoseMatchError({
  bindings,
  sampleTime,
  modelSkinnedMesh,
  sourceResult,
  mixer,
  modelRoot,
  buildCanonicalBoneMap,
  canonicalPoseSignature,
}) {
  if (!bindings?.mixers?.length || !modelSkinnedMesh?.skeleton?.bones?.length || !sourceResult?.skeleton?.bones?.length) {
    return Number.POSITIVE_INFINITY;
  }
  if (!(sampleTime > 0)) {
    return Number.POSITIVE_INFINITY;
  }

  const keys = [
    "head",
    "neck",
    "upperChest",
    "chest",
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
  ];
  const sourceTime = mixer ? mixer.time : 0;
  const modelTimes = bindings.mixers.map((mix) => mix.time);
  try {
    if (mixer) mixer.setTime(sampleTime);
    for (const mix of bindings.mixers) {
      mix.setTime(sampleTime);
    }
    sourceResult.skeleton.bones?.[0]?.updateMatrixWorld(true);
    modelRoot?.updateMatrixWorld(true);

    const sourceMap = buildCanonicalBoneMap(sourceResult.skeleton.bones);
    const targetMap = buildCanonicalBoneMap(modelSkinnedMesh.skeleton.bones);
    const sourceSig = canonicalPoseSignature(sourceMap, keys);
    const targetSig = canonicalPoseSignature(targetMap, keys);
    if (!sourceSig || !targetSig) return Number.POSITIVE_INFINITY;

    let angleSum = 0;
    let lenSum = 0;
    let count = 0;
    for (const key of keys) {
      const sv = sourceSig.vectors.get(key);
      const tv = targetSig.vectors.get(key);
      if (!sv || !tv) continue;
      const sn = sv.clone().normalize();
      const tn = tv.clone().normalize();
      angleSum += sn.angleTo(tn);
      lenSum += Math.abs(tv.length() / targetSig.scale - sv.length() / sourceSig.scale);
      count += 1;
    }
    if (count < 4) return Number.POSITIVE_INFINITY;
    const avgAngle = angleSum / count;
    const avgLen = lenSum / count;
    return avgAngle + avgLen * 0.35;
  } finally {
    if (mixer) mixer.setTime(sourceTime);
    for (let i = 0; i < bindings.mixers.length; i += 1) {
      bindings.mixers[i].setTime(modelTimes[i] || 0);
    }
    sourceResult.skeleton.bones?.[0]?.updateMatrixWorld(true);
    modelRoot?.updateMatrixWorld(true);
  }
}
