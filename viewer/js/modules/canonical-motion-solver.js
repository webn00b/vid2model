import * as THREE from "three";
import {
  arrayToVector3,
  buildNamesTargetToSourceByCanonical,
  CANONICAL_SOLVER_ORDER,
  getCanonicalParent,
  getCanonicalSegmentChild,
  selectEvaluationTimes,
} from "./canonical-motion-utils.js";
import { getPrimaryChildDirectionLocal } from "./retarget-chain-utils.js";

const _vec1 = new THREE.Vector3();
const _vec2 = new THREE.Vector3();
const _vec3 = new THREE.Vector3();
const _vec4 = new THREE.Vector3();
const _vec5 = new THREE.Vector3();
const _quat1 = new THREE.Quaternion();
const _quat2 = new THREE.Quaternion();
const _quat3 = new THREE.Quaternion();
const _mat1 = new THREE.Matrix4();

function buildSnapshot(modelRoot, bones) {
  return {
    modelRoot: modelRoot
      ? {
          target: modelRoot,
          position: modelRoot.position.clone(),
          quaternion: modelRoot.quaternion.clone(),
          scale: modelRoot.scale.clone(),
        }
      : null,
    bones: (bones || []).map((bone) => ({
      bone,
      position: bone.position.clone(),
      quaternion: bone.quaternion.clone(),
      scale: bone.scale.clone(),
    })),
  };
}

function restoreSnapshot(snapshot) {
  if (!snapshot) return;
  if (snapshot.modelRoot?.position) {
    const root = snapshot.modelRoot.target || null;
    if (root) {
      root.position.copy(snapshot.modelRoot.position);
      root.quaternion.copy(snapshot.modelRoot.quaternion);
      root.scale.copy(snapshot.modelRoot.scale);
    }
  }
  for (const entry of snapshot.bones || []) {
    entry.bone.position.copy(entry.position);
    entry.bone.quaternion.copy(entry.quaternion);
    entry.bone.scale.copy(entry.scale);
  }
}

function createSnapshot(modelRoot, bones) {
  return buildSnapshot(modelRoot, bones);
}

function getReferenceDirectionLocal(bone, primaryDir, outDir) {
  if (!bone?.isBone || !primaryDir || !outDir) return false;
  bone.getWorldPosition(_vec1);
  bone.getWorldQuaternion(_quat1);
  const invBoneQ = _quat1.invert();
  const tryWorldVector = (worldVector) => {
    outDir.copy(worldVector).applyQuaternion(invBoneQ);
    outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
    if (outDir.lengthSq() < 1e-8) return false;
    outDir.normalize();
    return true;
  };

  if (bone.parent?.isBone) {
    bone.parent.getWorldPosition(_vec2);
    outDir.copy(_vec2).sub(_vec1).applyQuaternion(invBoneQ);
    outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
    if (outDir.lengthSq() >= 1e-8) {
      outDir.normalize();
      return true;
    }
  }

  let bestChild = null;
  let bestLenSq = 0;
  for (const child of bone.children || []) {
    if (!child?.isBone) continue;
    child.getWorldPosition(_vec2);
    const lenSq = _vec2.distanceToSquared(_vec1);
    if (lenSq > bestLenSq) {
      bestLenSq = lenSq;
      bestChild = child;
    }
  }
  if (bestChild) {
    for (const child of bone.children || []) {
      if (!child?.isBone || child === bestChild) continue;
      child.getWorldPosition(_vec2);
      outDir.copy(_vec2).sub(_vec1).applyQuaternion(invBoneQ);
      outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
      if (outDir.lengthSq() >= 1e-8) {
        outDir.normalize();
        return true;
      }
    }
  }

  return (
    tryWorldVector(_vec2.set(1, 0, 0)) ||
    tryWorldVector(_vec2.set(0, 1, 0)) ||
    tryWorldVector(_vec2.set(0, 0, 1))
  );
}

function buildLocalFrame(primaryDirLocal, referenceDirLocal, outQ) {
  _vec1.copy(primaryDirLocal).normalize();
  _vec2.copy(referenceDirLocal);
  _vec2.addScaledVector(_vec1, -_vec2.dot(_vec1));
  if (_vec1.lengthSq() < 1e-8 || _vec2.lengthSq() < 1e-8) return false;
  _vec2.normalize();
  _vec3.crossVectors(_vec2, _vec1);
  if (_vec3.lengthSq() < 1e-8) return false;
  _vec3.normalize();
  _vec4.crossVectors(_vec1, _vec3).normalize();
  _mat1.makeBasis(_vec4, _vec1, _vec3);
  outQ.setFromRotationMatrix(_mat1).normalize();
  return true;
}

function buildWorldFrame(dirWorld, referenceWorld, outQ) {
  _vec1.copy(dirWorld).normalize();
  _vec2.copy(referenceWorld);
  _vec2.addScaledVector(_vec1, -_vec2.dot(_vec1));
  if (_vec1.lengthSq() < 1e-8 || _vec2.lengthSq() < 1e-8) return false;
  _vec2.normalize();
  _vec3.crossVectors(_vec2, _vec1);
  if (_vec3.lengthSq() < 1e-8) return false;
  _vec3.normalize();
  _vec4.crossVectors(_vec1, _vec3).normalize();
  _mat1.makeBasis(_vec4, _vec1, _vec3);
  outQ.setFromRotationMatrix(_mat1).normalize();
  return true;
}

function buildBoneSolveMeta(canonical, bone, canonicalMap) {
  if (!bone?.isBone) return null;
  const primaryChildDirLocal = getPrimaryChildDirectionLocal(bone);
  if (!primaryChildDirLocal) return null;
  const referenceDirLocal = new THREE.Vector3();
  if (!getReferenceDirectionLocal(bone, primaryChildDirLocal, referenceDirLocal)) return null;
  const restLocalFrame = new THREE.Quaternion();
  if (!buildLocalFrame(primaryChildDirLocal, referenceDirLocal, restLocalFrame)) return null;
  const childCanonical = getCanonicalSegmentChild(canonical);
  return {
    canonical,
    bone,
    childCanonical,
    restLocalFrame,
    restLocalFrameInv: restLocalFrame.clone().invert(),
    referenceDirLocal: referenceDirLocal.clone(),
  };
}

function buildSolverState(targetBones, canonicalBoneKey, buildCanonicalBoneMap) {
  const canonicalMap = buildCanonicalBoneMap(targetBones || []);
  const metaByCanonical = new Map();
  for (const canonical of CANONICAL_SOLVER_ORDER) {
    const bone = canonicalMap.get(canonical) || null;
    const meta = buildBoneSolveMeta(canonical, bone, canonicalMap);
    if (meta) metaByCanonical.set(canonical, meta);
  }
  return { canonicalMap, metaByCanonical };
}

function solveBoneFrame(meta, frameSegment, solverState) {
  if (!meta?.bone || !frameSegment?.dir) return false;
  const desiredDir = arrayToVector3(frameSegment.dir, _vec1);
  if (!desiredDir || desiredDir.lengthSq() < 1e-8) return false;
  let desiredReference =
    arrayToVector3(frameSegment.normal, _vec2) ||
    arrayToVector3(frameSegment.bendHint, _vec2);
  if (!desiredReference || desiredReference.lengthSq() < 1e-8) {
    const parentCanonical = getCanonicalParent(meta.canonical);
    const parentMeta = parentCanonical ? solverState.metaByCanonical.get(parentCanonical) || null : null;
    if (parentMeta?.bone) {
      parentMeta.bone.getWorldQuaternion(_quat1);
      desiredReference = _vec2.copy(meta.referenceDirLocal).applyQuaternion(_quat1);
    } else {
      desiredReference = _vec2.set(1, 0, 0);
    }
  }
  if (!buildWorldFrame(desiredDir, desiredReference, _quat1)) return false;
  _quat2.copy(_quat1).multiply(meta.restLocalFrameInv).normalize();
  if (meta.bone.parent?.isBone) {
    meta.bone.parent.getWorldQuaternion(_quat3);
    meta.bone.quaternion.copy(_quat3.invert().multiply(_quat2)).normalize();
  } else {
    meta.bone.quaternion.copy(_quat2);
  }
  meta.bone.updateMatrixWorld(true);
  return true;
}

function aggregateAverage(values) {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return null;
  return Number((finite.reduce((sum, value) => sum + value, 0) / finite.length).toFixed(6));
}

export function evaluateCanonicalMotion({
  canonicalMotion,
  modelRoot,
  targetBones,
  sourceResult,
  sourceMixer,
  canonicalBoneKey,
  buildCanonicalBoneMap,
  collectAlignmentDiagnostics,
  buildLegChainDiagnostics,
  buildArmChainDiagnostics,
  buildTorsoChainDiagnostics,
  alignModelHipsToSource,
  canonicalFilter = null,
  bodyMetricCanonicalFilter = null,
  maxSamples = 12,
}) {
  if (!canonicalMotion?.frames?.length || !targetBones?.length || !sourceResult?.skeleton?.bones?.length) {
    return {
      ran: false,
      reason: "missing-input",
    };
  }

  const filteredTargetBones = canonicalFilter
    ? targetBones.filter((bone) => canonicalFilter.has(canonicalBoneKey(bone.name) || ""))
    : [...targetBones];
  const solverState = buildSolverState(filteredTargetBones, canonicalBoneKey, buildCanonicalBoneMap);
  const sourceCanonicalNames = canonicalMotion.restPose?.sourceCanonicalNames || {};
  const namesTargetToSource = buildNamesTargetToSourceByCanonical(
    filteredTargetBones,
    sourceCanonicalNames,
    canonicalBoneKey
  );
  const sampleTimes = selectEvaluationTimes(
    canonicalMotion.frames.map((frame) => Number(frame.time || 0)),
    maxSamples
  );
  const frameByTime = new Map(canonicalMotion.frames.map((frame) => [Number(frame.time || 0), frame]));
  const sourceTime = sourceMixer?.time || 0;
  const snapshot = createSnapshot(modelRoot, targetBones);
  const poseErrors = [];
  const rotErrors = [];
  const lowerBodyRotErrors = [];
  const sampleRows = [];
  let legsMirrored = false;

  try {
    for (const time of sampleTimes) {
      const frame = frameByTime.get(time) || null;
      if (!frame) continue;
      restoreSnapshot(snapshot);
      sourceMixer?.setTime(time);
      sourceResult.skeleton.bones?.[0]?.updateMatrixWorld(true);
      modelRoot?.updateMatrixWorld(true);

      for (const canonical of CANONICAL_SOLVER_ORDER) {
        solveBoneFrame(solverState.metaByCanonical.get(canonical) || null, frame.segments?.[canonical] || null, solverState);
      }
      modelRoot?.updateMatrixWorld(true);
      alignModelHipsToSource?.(false);
      modelRoot?.updateMatrixWorld(true);

      const report = collectAlignmentDiagnostics({
        targetBones: filteredTargetBones,
        sourceBones: sourceResult.skeleton.bones,
        namesTargetToSource,
        sourceClip: sourceResult.clip,
        maxRows: 5,
      });
      const lowerBodyBones = filteredTargetBones.filter((bone) =>
        bodyMetricCanonicalFilter?.has(canonicalBoneKey(bone.name) || "")
      );
      const lowerBodyReport = collectAlignmentDiagnostics({
        targetBones: lowerBodyBones.length ? lowerBodyBones : filteredTargetBones,
        sourceBones: sourceResult.skeleton.bones,
        namesTargetToSource,
        sourceClip: sourceResult.clip,
        maxRows: 5,
      });
      const legRows = buildLegChainDiagnostics({
        targetBones: filteredTargetBones,
        sourceBones: sourceResult.skeleton.bones,
        names: namesTargetToSource,
        reason: `canonical-solver t=${time.toFixed(3)}`,
      });
      buildArmChainDiagnostics({
        targetBones: filteredTargetBones,
        sourceBones: sourceResult.skeleton.bones,
        names: namesTargetToSource,
        reason: `canonical-solver t=${time.toFixed(3)}`,
      });
      buildTorsoChainDiagnostics({
        targetBones: filteredTargetBones,
        sourceBones: sourceResult.skeleton.bones,
        names: namesTargetToSource,
        reason: `canonical-solver t=${time.toFixed(3)}`,
      });

      const poseError = Number.isFinite(report?.avgPosErrNorm) ? report.avgPosErrNorm : report?.avgPosErr;
      const rotError = Number.isFinite(report?.avgRotErrDeg) ? report.avgRotErrDeg : null;
      const lowerBodyRotError = Number.isFinite(lowerBodyReport?.avgRotErrDeg) ? lowerBodyReport.avgRotErrDeg : null;
      poseErrors.push(poseError);
      rotErrors.push(rotError);
      lowerBodyRotErrors.push(lowerBodyRotError);
      if ((legRows || []).some((row) => row?.bendMirrored === true)) {
        legsMirrored = true;
      }
      sampleRows.push({
        time: Number(time.toFixed(6)),
        poseError: Number.isFinite(poseError) ? Number(poseError.toFixed(6)) : null,
        avgRotErrDeg: Number.isFinite(rotError) ? Number(rotError.toFixed(6)) : null,
        lowerBodyRotError: Number.isFinite(lowerBodyRotError) ? Number(lowerBodyRotError.toFixed(6)) : null,
        legsMirrored: (legRows || []).some((row) => row?.bendMirrored === true),
      });
    }
  } finally {
    restoreSnapshot(snapshot);
    sourceMixer?.setTime(sourceTime);
    sourceResult.skeleton.bones?.[0]?.updateMatrixWorld(true);
    modelRoot?.updateMatrixWorld(true);
  }

  return {
    ran: true,
    stage: canonicalMotion.stage,
    exportFormat: canonicalMotion.format,
    solverFormat: "vid2model.canonical-solve.v1",
    sampleCount: sampleRows.length,
    summary: {
      poseError: aggregateAverage(poseErrors),
      avgRotErrDeg: aggregateAverage(rotErrors),
      lowerBodyRotError: aggregateAverage(lowerBodyRotErrors),
      legsMirrored,
    },
    samples: sampleRows,
  };
}
