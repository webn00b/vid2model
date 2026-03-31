import * as THREE from "three";
import { canonicalBoneKey, parseTrackName } from "./bone-utils.js";

function estimateBoneSetHeight(bones) {
  if (!bones?.length) return null;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  const pos = new THREE.Vector3();
  let count = 0;
  for (const bone of bones) {
    bone.getWorldPosition(pos);
    if (!Number.isFinite(pos.y)) continue;
    minY = Math.min(minY, pos.y);
    maxY = Math.max(maxY, pos.y);
    count += 1;
  }
  if (!count) return null;
  const h = maxY - minY;
  return Number.isFinite(h) && h > 1e-6 ? h : null;
}

export function collectTrackPresenceByBone(clip) {
  const trackByBone = new Map();
  for (const t of clip?.tracks || []) {
    const parsed = parseTrackName(t.name);
    if (!parsed) continue;
    const row = trackByBone.get(parsed.bone) || { hasQ: false, hasP: false };
    if (parsed.property === "quaternion") row.hasQ = true;
    if (parsed.property === "position") row.hasP = true;
    trackByBone.set(parsed.bone, row);
  }
  return trackByBone;
}

export function collectAlignmentDiagnostics({
  targetBones,
  sourceBones,
  namesTargetToSource,
  sourceClip,
  maxRows = 24,
  overlayYawOverride = null,
  sourceOverlay = null,
  overlayUpAxis = null,
}) {
  const sourceByName = new Map((sourceBones || []).map((b) => [b.name, b]));
  const sourceTrackByBone = collectTrackPresenceByBone(sourceClip);
  const rows = [];
  const unmappedCanonical = [];
  const sourceMissing = [];
  const sourceHeight = estimateBoneSetHeight(sourceBones);
  const targetHeight = estimateBoneSetHeight(targetBones);
  const scaleRefRaw =
    Number.isFinite(sourceHeight) && Number.isFinite(targetHeight)
      ? (sourceHeight + targetHeight) / 2
      : (targetHeight || sourceHeight || 1);
  const scaleRef = Math.max(1e-6, scaleRefRaw || 1);

  const overlayYaw =
    Number.isFinite(overlayYawOverride)
      ? overlayYawOverride
      : (sourceOverlay?.overlayYaw || 0);
  const applyOverlayYaw = Number.isFinite(overlayYaw) && Math.abs(overlayYaw) > 1e-5;
  const pivot = new THREE.Vector3();
  if (applyOverlayYaw && sourceOverlay?.pivotBone) {
    sourceOverlay.pivotBone.getWorldPosition(pivot);
  }
  const overlayYawQ = new THREE.Quaternion();
  if (applyOverlayYaw) {
    overlayYawQ.setFromAxisAngle(overlayUpAxis || new THREE.Vector3(0, 1, 0), overlayYaw);
  }

  const srcPos = new THREE.Vector3();
  const dstPos = new THREE.Vector3();
  const srcQ = new THREE.Quaternion();
  const dstQ = new THREE.Quaternion();

  for (const targetBone of targetBones || []) {
    const canonical = canonicalBoneKey(targetBone.name) || "";
    const sourceName = namesTargetToSource[targetBone.name] || "";
    if (!sourceName) {
      if (canonical) {
        unmappedCanonical.push({
          target: targetBone.name,
          canonical,
        });
      }
      continue;
    }
    const sourceBone = sourceByName.get(sourceName);
    if (!sourceBone) {
      sourceMissing.push({
        target: targetBone.name,
        source: sourceName,
        canonical,
      });
      continue;
    }

    sourceBone.getWorldPosition(srcPos);
    if (applyOverlayYaw) {
      srcPos.sub(pivot).applyQuaternion(overlayYawQ).add(pivot);
    }
    targetBone.getWorldPosition(dstPos);
    const posErr = dstPos.distanceTo(srcPos);

    sourceBone.getWorldQuaternion(srcQ);
    if (applyOverlayYaw) {
      srcQ.premultiply(overlayYawQ);
    }
    targetBone.getWorldQuaternion(dstQ);
    const rotErrDeg = THREE.MathUtils.radToDeg(srcQ.angleTo(dstQ));
    const trackState = sourceTrackByBone.get(sourceName) || { hasQ: false, hasP: false };

    rows.push({
      target: targetBone.name,
      source: sourceName,
      canonical,
      posErr: Number(posErr.toFixed(5)),
      posErrNorm: Number((posErr / scaleRef).toFixed(5)),
      rotErrDeg: Number(rotErrDeg.toFixed(3)),
      sourceHasQ: trackState.hasQ,
      sourceHasP: trackState.hasP,
    });
  }

  const sortedByPos = rows.slice().sort((a, b) => b.posErr - a.posErr);
  const sortedByRot = rows.slice().sort((a, b) => b.rotErrDeg - a.rotErrDeg);
  const avgPosErr = rows.length ? rows.reduce((sum, r) => sum + r.posErr, 0) / rows.length : 0;
  const avgPosErrNorm = rows.length ? rows.reduce((sum, r) => sum + r.posErrNorm, 0) / rows.length : 0;
  const avgRotErrDeg = rows.length ? rows.reduce((sum, r) => sum + r.rotErrDeg, 0) / rows.length : 0;

  const hipsTarget = (targetBones || []).find((b) => canonicalBoneKey(b.name) === "hips") || null;
  const hipsSourceName = hipsTarget ? namesTargetToSource[hipsTarget.name] || "" : "";
  const hipsSource = hipsSourceName ? sourceByName.get(hipsSourceName) || null : null;
  let hipsPosErr = null;
  let hipsPosErrNorm = null;
  if (hipsTarget && hipsSource) {
    hipsTarget.getWorldPosition(dstPos);
    hipsSource.getWorldPosition(srcPos);
    if (applyOverlayYaw) {
      srcPos.sub(pivot).applyQuaternion(overlayYawQ).add(pivot);
    }
    const hipsDist = dstPos.distanceTo(srcPos);
    hipsPosErr = Number(hipsDist.toFixed(5));
    hipsPosErrNorm = Number((hipsDist / scaleRef).toFixed(5));
  }

  return {
    totalCompared: rows.length,
    avgPosErr: Number(avgPosErr.toFixed(5)),
    avgPosErrNorm: Number(avgPosErrNorm.toFixed(5)),
    avgRotErrDeg: Number(avgRotErrDeg.toFixed(3)),
    overlayYawDeg: Number((overlayYaw * 180 / Math.PI).toFixed(2)),
    sourceHeight: Number.isFinite(sourceHeight) ? Number(sourceHeight.toFixed(5)) : null,
    targetHeight: Number.isFinite(targetHeight) ? Number(targetHeight.toFixed(5)) : null,
    scaleRef: Number(scaleRef.toFixed(5)),
    hipsPosErr,
    hipsPosErrNorm,
    worstPosition: sortedByPos.slice(0, maxRows),
    worstRotation: sortedByRot.slice(0, maxRows),
    unmappedCanonical: unmappedCanonical.slice(0, maxRows),
    sourceMissing: sourceMissing.slice(0, maxRows),
  };
}

export function dumpRetargetAlignmentDiagnostics({
  reason = "manual",
  modelSkinnedMesh,
  sourceResult,
  names,
  sourceOverlay = null,
  overlayUpAxis = null,
  windowRef = null,
}) {
  if (!modelSkinnedMesh?.skeleton?.bones?.length || !sourceResult?.skeleton?.bones?.length) {
    console.warn("[vid2model/diag] retarget alignment dump skipped: source/model is not ready");
    return null;
  }
  const report = collectAlignmentDiagnostics({
    targetBones: modelSkinnedMesh.skeleton.bones,
    sourceBones: sourceResult.skeleton.bones,
    namesTargetToSource: names,
    sourceClip: sourceResult.clip,
    maxRows: 20,
    sourceOverlay,
    overlayUpAxis,
  });
  if (windowRef) {
    windowRef.__vid2modelAlignment = report;
  }

  const diagMode = String(windowRef?.__vid2modelDiagMode || "minimal").trim().toLowerCase();
  const shouldPrintVerbose = reason === "manual" || diagMode === "verbose";
  if (shouldPrintVerbose) {
    console.log("[vid2model/diag] alignment-dump", { reason, ...report });
    if (report.worstPosition.length) console.table(report.worstPosition);
    if (report.worstRotation.length) console.table(report.worstRotation);
    if (report.unmappedCanonical.length) {
      console.log("[vid2model/diag] unmapped-canonical");
      console.table(report.unmappedCanonical);
    }
    if (report.sourceMissing.length) {
      console.log("[vid2model/diag] source-missing");
      console.table(report.sourceMissing);
    }
  }
  return report;
}

export function getLiveDeltaOverride(windowRef) {
  const v = windowRef?.__vid2modelForceLiveDelta;
  if (v === true || v === false) return v;
  return null;
}

export function computeHipsYawError(targetBones, sourceBones, namesTargetToSource) {
  const targetHips = (targetBones || []).find((b) => canonicalBoneKey(b.name) === "hips") || null;
  if (!targetHips) return 0;
  const sourceByName = new Map((sourceBones || []).map((b) => [b.name, b]));
  const sourceHipsName =
    namesTargetToSource[targetHips.name] ||
    (sourceBones || []).find((b) => canonicalBoneKey(b.name) === "hips")?.name ||
    "";
  if (!sourceHipsName) return 0;
  const sourceHips = sourceByName.get(sourceHipsName);
  if (!sourceHips) return 0;

  const srcQ = new THREE.Quaternion();
  const dstQ = new THREE.Quaternion();
  sourceHips.getWorldQuaternion(srcQ);
  targetHips.getWorldQuaternion(dstQ);

  const srcF = new THREE.Vector3(0, 0, 1).applyQuaternion(srcQ);
  const dstF = new THREE.Vector3(0, 0, 1).applyQuaternion(dstQ);
  srcF.y = 0;
  dstF.y = 0;
  if (srcF.lengthSq() < 1e-9 || dstF.lengthSq() < 1e-9) return 0;
  srcF.normalize();
  dstF.normalize();
  const crossY = srcF.clone().cross(dstF).y;
  const dot = THREE.MathUtils.clamp(srcF.dot(dstF), -1, 1);
  const angle = Math.atan2(crossY, dot);
  return Number.isFinite(angle) ? angle : 0;
}

export function buildRootYawCandidates(rawFacingYaw, quantizeFacingYaw) {
  const set = new Set();
  const list = [];
  const push = (v) => {
    if (!Number.isFinite(v)) return;
    let a = Math.atan2(Math.sin(v), Math.cos(v));
    if (Math.abs(a) < 1e-6) a = 0;
    const key = Number(a.toFixed(6));
    if (set.has(key)) return;
    set.add(key);
    list.push(a);
  };
  push(0);
  push(Math.PI);
  push(-Math.PI);
  push(quantizeFacingYaw(-rawFacingYaw));
  push(quantizeFacingYaw(Math.PI - rawFacingYaw));
  push(-rawFacingYaw);
  return list;
}
