import * as THREE from "three";

export function toRoundedVec3(vec) {
  if (!vec || !Number.isFinite(vec.x) || !Number.isFinite(vec.y) || !Number.isFinite(vec.z)) return null;
  return {
    x: Number(vec.x.toFixed(4)),
    y: Number(vec.y.toFixed(4)),
    z: Number(vec.z.toFixed(4)),
  };
}

export function angleBetweenWorldSegments(aStart, aEnd, bStart, bEnd) {
  if (!aStart || !aEnd || !bStart || !bEnd) return null;
  const a = new THREE.Vector3().subVectors(aEnd, aStart);
  const b = new THREE.Vector3().subVectors(bEnd, bStart);
  const aLen = a.length();
  const bLen = b.length();
  if (!(aLen > 1e-6) || !(bLen > 1e-6)) return null;
  a.multiplyScalar(1 / aLen);
  b.multiplyScalar(1 / bLen);
  return Number(THREE.MathUtils.radToDeg(a.angleTo(b)).toFixed(3));
}

export function computeBendNormal(startPos, midPos, endPos) {
  if (!startPos || !midPos || !endPos) return null;
  const a = new THREE.Vector3().subVectors(midPos, startPos);
  const b = new THREE.Vector3().subVectors(endPos, midPos);
  const aLen = a.length();
  const bLen = b.length();
  if (!(aLen > 1e-6) || !(bLen > 1e-6)) return null;
  a.multiplyScalar(1 / aLen);
  b.multiplyScalar(1 / bLen);
  const normal = new THREE.Vector3().crossVectors(a, b);
  if (normal.lengthSq() < 1e-10) return null;
  return normal.normalize();
}

export function getPreferredChildBone(bone, canonicalBoneKey) {
  if (!bone?.isBone) return null;
  const canonical = canonicalBoneKey(bone.name) || "";
  const children = (bone.children || []).filter((child) => child?.isBone);
  if (!children.length) return null;

  if (canonical === "leftHand" || canonical === "rightHand") {
    const preferredFingerRoots = [
      canonical.replace("Hand", "MiddleProximal"),
      canonical.replace("Hand", "IndexProximal"),
      canonical.replace("Hand", "RingProximal"),
      canonical.replace("Hand", "LittleProximal"),
      canonical.replace("Hand", "ThumbMetacarpal"),
      canonical.replace("Hand", "ThumbProximal"),
    ];
    for (const preferred of preferredFingerRoots) {
      const match = children.find((child) => (canonicalBoneKey(child.name) || "") === preferred);
      if (match) return match;
    }
  }

  if (canonical === "leftFoot" || canonical === "rightFoot") {
    const preferredToes = canonical.replace("Foot", "Toes");
    const match = children.find((child) => (canonicalBoneKey(child.name) || "") === preferredToes);
    if (match) return match;
  }

  return null;
}

export function getPrimaryChildBone(bone, canonicalBoneKey, tmpWorldPosA, tmpWorldPosB) {
  if (!bone?.isBone) return null;
  const preferred = getPreferredChildBone(bone, canonicalBoneKey);
  if (preferred) return preferred;
  bone.getWorldPosition(tmpWorldPosA);
  let bestChild = null;
  let bestLenSq = 0;
  for (const child of bone.children || []) {
    if (!child?.isBone) continue;
    child.getWorldPosition(tmpWorldPosB);
    const lenSq = tmpWorldPosB.distanceToSquared(tmpWorldPosA);
    if (lenSq > bestLenSq) {
      bestLenSq = lenSq;
      bestChild = child;
    }
  }
  return bestLenSq > 1e-10 ? bestChild : null;
}

export function createViewerChainDiagnostics({
  canonicalBoneKey,
  buildCanonicalBoneMap,
  isVerboseDiagMode,
}) {
  let restCorrectionLog = [];
  let legChainDiagLog = [];
  let armChainDiagLog = [];
  let torsoChainDiagLog = [];
  let footChainDiagLog = [];
  const tmpWorldPosA = new THREE.Vector3();
  const tmpWorldPosB = new THREE.Vector3();

  function resetRestCorrectionLog() {
    restCorrectionLog = [];
    window.__vid2modelRestCorrections = restCorrectionLog;
  }

  function resetLegChainDiagLog() {
    legChainDiagLog = [];
    window.__vid2modelLegChainDiag = legChainDiagLog;
  }

  function resetArmChainDiagLog() {
    armChainDiagLog = [];
    window.__vid2modelArmChainDiag = armChainDiagLog;
  }

  function resetTorsoChainDiagLog() {
    torsoChainDiagLog = [];
    window.__vid2modelTorsoChainDiag = torsoChainDiagLog;
  }

  function resetFootChainDiagLog() {
    footChainDiagLog = [];
    window.__vid2modelFootChainDiag = footChainDiagLog;
  }

  function recordRestCorrectionLog(entry) {
    if (!entry?.canonical) return;
    const key = `${entry.canonical}|${entry.target || ""}|${entry.source || ""}`;
    if (restCorrectionLog.some((row) => `${row.canonical}|${row.target || ""}|${row.source || ""}` === key)) {
      return;
    }
    restCorrectionLog.push(entry);
    window.__vid2modelRestCorrections = restCorrectionLog;
  }

  function dumpRestCorrectionLog(reason = "retarget") {
    const rows = Array.isArray(restCorrectionLog) ? restCorrectionLog.slice() : [];
    window.__vid2modelRestCorrections = rows;
    if (!rows.length) return;
    const suspicious = rows.filter((row) => (row.finalAngleDeg || 0) >= 90 || (row.autoAngleDeg || 0) >= 90);
    if (!isVerboseDiagMode() && !suspicious.length) return;
    console.log("[vid2model/diag] rest-corrections", {
      reason,
      total: rows.length,
      suspicious: suspicious.length,
    });
    if (isVerboseDiagMode()) {
      console.table(rows);
    } else if (suspicious.length >= 6) {
      console.table(suspicious);
    }
  }

  function buildLegChainDiagnostics({ targetBones = [], sourceBones = [], names = {}, reason = "retarget" } = {}) {
    const targetMap = buildCanonicalBoneMap(targetBones || []);
    const sourceMap = buildCanonicalBoneMap(sourceBones || []);
    const rows = [];
    const legSides = ["left", "right"];
    for (const side of legSides) {
      const upperKey = `${side}UpperLeg`;
      const lowerKey = `${side}LowerLeg`;
      const footKey = `${side}Foot`;
      const toesKey = `${side}Toes`;
      const targetUpper = targetMap.get(upperKey) || null;
      const targetLower = targetMap.get(lowerKey) || null;
      const targetFoot = targetMap.get(footKey) || null;
      const targetToes = targetMap.get(toesKey) || null;
      if (!targetUpper || !targetLower || !targetFoot) continue;

      const mappedUpperCanonical = canonicalBoneKey(names[targetUpper.name] || upperKey) || upperKey;
      const mappedLowerCanonical = canonicalBoneKey(names[targetLower.name] || lowerKey) || lowerKey;
      const mappedFootCanonical = canonicalBoneKey(names[targetFoot.name] || footKey) || footKey;
      const mappedToesCanonical = canonicalBoneKey(targetToes ? (names[targetToes.name] || toesKey) : toesKey) || toesKey;

      const sourceUpper = sourceMap.get(mappedUpperCanonical) || null;
      const sourceLower = sourceMap.get(mappedLowerCanonical) || null;
      const sourceFoot = sourceMap.get(mappedFootCanonical) || null;
      const sourceToes = sourceMap.get(mappedToesCanonical) || null;
      if (!sourceUpper || !sourceLower || !sourceFoot) continue;

      const targetUpperPos = new THREE.Vector3();
      const targetLowerPos = new THREE.Vector3();
      const targetFootPos = new THREE.Vector3();
      const targetToesPos = targetToes ? new THREE.Vector3() : null;
      const sourceUpperPos = new THREE.Vector3();
      const sourceLowerPos = new THREE.Vector3();
      const sourceFootPos = new THREE.Vector3();
      const sourceToesPos = sourceToes ? new THREE.Vector3() : null;

      targetUpper.getWorldPosition(targetUpperPos);
      targetLower.getWorldPosition(targetLowerPos);
      targetFoot.getWorldPosition(targetFootPos);
      if (targetToes && targetToesPos) targetToes.getWorldPosition(targetToesPos);
      sourceUpper.getWorldPosition(sourceUpperPos);
      sourceLower.getWorldPosition(sourceLowerPos);
      sourceFoot.getWorldPosition(sourceFootPos);
      if (sourceToes && sourceToesPos) sourceToes.getWorldPosition(sourceToesPos);

      const targetBendNormal = computeBendNormal(targetUpperPos, targetLowerPos, targetFootPos);
      const sourceBendNormal = computeBendNormal(sourceUpperPos, sourceLowerPos, sourceFootPos);
      const bendDot =
        targetBendNormal && sourceBendNormal
          ? Number(targetBendNormal.dot(sourceBendNormal).toFixed(4))
          : null;
      const bendAngleDeg =
        targetBendNormal && sourceBendNormal
          ? Number(THREE.MathUtils.radToDeg(targetBendNormal.angleTo(sourceBendNormal)).toFixed(3))
          : null;

      rows.push({
        side,
        reason,
        target: {
          upper: targetUpper.name,
          lower: targetLower.name,
          foot: targetFoot.name,
          toes: targetToes?.name || null,
        },
        source: {
          upper: sourceUpper.name,
          lower: sourceLower.name,
          foot: sourceFoot.name,
          toes: sourceToes?.name || null,
        },
        sourceCanonical: {
          upper: mappedUpperCanonical,
          lower: mappedLowerCanonical,
          foot: mappedFootCanonical,
          toes: sourceToes ? mappedToesCanonical : null,
        },
        thighAngleDeg: angleBetweenWorldSegments(targetUpperPos, targetLowerPos, sourceUpperPos, sourceLowerPos),
        shinAngleDeg: angleBetweenWorldSegments(targetLowerPos, targetFootPos, sourceLowerPos, sourceFootPos),
        footAngleDeg:
          targetToesPos && sourceToesPos
            ? angleBetweenWorldSegments(targetFootPos, targetToesPos, sourceFootPos, sourceToesPos)
            : null,
        bendAngleDeg,
        bendDot,
        bendMirrored: bendDot != null ? bendDot < 0 : null,
        targetBendNormal: toRoundedVec3(targetBendNormal),
        sourceBendNormal: toRoundedVec3(sourceBendNormal),
      });
    }
    legChainDiagLog = rows;
    window.__vid2modelLegChainDiag = rows;
    return rows;
  }

  function dumpLegChainDiagLog(reason = "retarget") {
    const rows = Array.isArray(legChainDiagLog) ? legChainDiagLog.slice() : [];
    window.__vid2modelLegChainDiag = rows;
    if (!rows.length) return;
    const suspicious = rows.filter((row) => {
      const segmentWorst = Math.max(row.thighAngleDeg || 0, row.shinAngleDeg || 0, row.footAngleDeg || 0);
      return row.bendMirrored === true || segmentWorst >= 90 || (row.bendAngleDeg || 0) >= 90;
    });
    if (!isVerboseDiagMode() && !suspicious.length) return;
    console.log("[vid2model/diag] leg-chain", {
      reason,
      total: rows.length,
      suspicious: suspicious.length,
    });
    if (isVerboseDiagMode()) {
      console.table(rows);
    } else if (suspicious.length) {
      console.table(suspicious);
    }
  }

  function buildArmChainDiagnostics({ targetBones = [], sourceBones = [], names = {}, reason = "retarget" } = {}) {
    const targetMap = buildCanonicalBoneMap(targetBones || []);
    const sourceMap = buildCanonicalBoneMap(sourceBones || []);
    const rows = [];
    const armSides = ["left", "right"];
    for (const side of armSides) {
      const upperKey = `${side}UpperArm`;
      const lowerKey = `${side}LowerArm`;
      const handKey = `${side}Hand`;
      const targetUpper = targetMap.get(upperKey) || null;
      const targetLower = targetMap.get(lowerKey) || null;
      const targetHand = targetMap.get(handKey) || null;
      if (!targetUpper || !targetLower || !targetHand) continue;

      const mappedUpperCanonical = canonicalBoneKey(names[targetUpper.name] || upperKey) || upperKey;
      const mappedLowerCanonical = canonicalBoneKey(names[targetLower.name] || lowerKey) || lowerKey;
      const mappedHandCanonical = canonicalBoneKey(names[targetHand.name] || handKey) || handKey;

      const sourceUpper = sourceMap.get(mappedUpperCanonical) || null;
      const sourceLower = sourceMap.get(mappedLowerCanonical) || null;
      const sourceHand = sourceMap.get(mappedHandCanonical) || null;
      if (!sourceUpper || !sourceLower || !sourceHand) continue;

      const targetUpperPos = new THREE.Vector3();
      const targetLowerPos = new THREE.Vector3();
      const targetHandPos = new THREE.Vector3();
      const sourceUpperPos = new THREE.Vector3();
      const sourceLowerPos = new THREE.Vector3();
      const sourceHandPos = new THREE.Vector3();
      const targetHandChild = getPrimaryChildBone(targetHand, canonicalBoneKey, tmpWorldPosA, tmpWorldPosB);
      const sourceHandChild = getPrimaryChildBone(sourceHand, canonicalBoneKey, tmpWorldPosA, tmpWorldPosB);
      const targetHandChildPos = targetHandChild ? new THREE.Vector3() : null;
      const sourceHandChildPos = sourceHandChild ? new THREE.Vector3() : null;

      targetUpper.getWorldPosition(targetUpperPos);
      targetLower.getWorldPosition(targetLowerPos);
      targetHand.getWorldPosition(targetHandPos);
      if (targetHandChild && targetHandChildPos) targetHandChild.getWorldPosition(targetHandChildPos);
      sourceUpper.getWorldPosition(sourceUpperPos);
      sourceLower.getWorldPosition(sourceLowerPos);
      sourceHand.getWorldPosition(sourceHandPos);
      if (sourceHandChild && sourceHandChildPos) sourceHandChild.getWorldPosition(sourceHandChildPos);

      const targetBendNormal = computeBendNormal(targetUpperPos, targetLowerPos, targetHandPos);
      const sourceBendNormal = computeBendNormal(sourceUpperPos, sourceLowerPos, sourceHandPos);
      const bendDot =
        targetBendNormal && sourceBendNormal
          ? Number(targetBendNormal.dot(sourceBendNormal).toFixed(4))
          : null;
      const bendAngleDeg =
        targetBendNormal && sourceBendNormal
          ? Number(THREE.MathUtils.radToDeg(targetBendNormal.angleTo(sourceBendNormal)).toFixed(3))
          : null;

      rows.push({
        side,
        reason,
        target: {
          upper: targetUpper.name,
          lower: targetLower.name,
          hand: targetHand.name,
          handChild: targetHandChild?.name || null,
        },
        source: {
          upper: sourceUpper.name,
          lower: sourceLower.name,
          hand: sourceHand.name,
          handChild: sourceHandChild?.name || null,
        },
        sourceCanonical: {
          upper: mappedUpperCanonical,
          lower: mappedLowerCanonical,
          hand: mappedHandCanonical,
        },
        upperAngleDeg: angleBetweenWorldSegments(targetUpperPos, targetLowerPos, sourceUpperPos, sourceLowerPos),
        lowerAngleDeg: angleBetweenWorldSegments(targetLowerPos, targetHandPos, sourceLowerPos, sourceHandPos),
        handAngleDeg:
          targetHandChildPos && sourceHandChildPos
            ? angleBetweenWorldSegments(targetHandPos, targetHandChildPos, sourceHandPos, sourceHandChildPos)
            : null,
        bendAngleDeg,
        bendDot,
        bendMirrored: bendDot != null ? bendDot < 0 : null,
        targetBendNormal: toRoundedVec3(targetBendNormal),
        sourceBendNormal: toRoundedVec3(sourceBendNormal),
      });
    }
    armChainDiagLog = rows;
    window.__vid2modelArmChainDiag = rows;
    return rows;
  }

  function dumpArmChainDiagLog(reason = "retarget") {
    const rows = Array.isArray(armChainDiagLog) ? armChainDiagLog.slice() : [];
    window.__vid2modelArmChainDiag = rows;
    if (!rows.length) return;
    const suspicious = rows.filter((row) => {
      const segmentWorst = Math.max(row.upperAngleDeg || 0, row.lowerAngleDeg || 0, row.handAngleDeg || 0);
      return row.bendMirrored === true || segmentWorst >= 90 || (row.bendAngleDeg || 0) >= 90;
    });
    if (!isVerboseDiagMode() && !suspicious.length) return;
    console.log("[vid2model/diag] arm-chain", {
      reason,
      total: rows.length,
      suspicious: suspicious.length,
    });
    if (isVerboseDiagMode()) {
      console.table(rows);
    } else if (suspicious.length) {
      console.table(suspicious);
    }
  }

  function buildTorsoChainDiagnostics({ targetBones = [], sourceBones = [], names = {}, reason = "retarget" } = {}) {
    const targetMap = buildCanonicalBoneMap(targetBones || []);
    const sourceMap = buildCanonicalBoneMap(sourceBones || []);
    const rows = [];
    const chain = [
      ["hips", "spine"],
      ["spine", "chest"],
      ["chest", "upperChest"],
      ["upperChest", "neck"],
      ["neck", "head"],
    ];
    for (const [startKey, endKey] of chain) {
      const targetStart = targetMap.get(startKey) || null;
      const targetEnd = targetMap.get(endKey) || null;
      if (!targetStart || !targetEnd) continue;
      const mappedStartCanonical = canonicalBoneKey(names[targetStart.name] || startKey) || startKey;
      const mappedEndCanonical = canonicalBoneKey(names[targetEnd.name] || endKey) || endKey;
      const sourceStart = sourceMap.get(mappedStartCanonical) || null;
      const sourceEnd = sourceMap.get(mappedEndCanonical) || null;
      if (!sourceStart || !sourceEnd) continue;

      const targetStartPos = new THREE.Vector3();
      const targetEndPos = new THREE.Vector3();
      const sourceStartPos = new THREE.Vector3();
      const sourceEndPos = new THREE.Vector3();
      targetStart.getWorldPosition(targetStartPos);
      targetEnd.getWorldPosition(targetEndPos);
      sourceStart.getWorldPosition(sourceStartPos);
      sourceEnd.getWorldPosition(sourceEndPos);

      const targetDir = new THREE.Vector3().subVectors(targetEndPos, targetStartPos).normalize();
      const sourceDir = new THREE.Vector3().subVectors(sourceEndPos, sourceStartPos).normalize();
      const segmentAngleDeg = angleBetweenWorldSegments(targetStartPos, targetEndPos, sourceStartPos, sourceEndPos);
      const segmentDot =
        targetDir.lengthSq() > 1e-10 && sourceDir.lengthSq() > 1e-10
          ? Number(targetDir.dot(sourceDir).toFixed(4))
          : null;
      const targetLen = targetEndPos.distanceTo(targetStartPos);
      const sourceLen = sourceEndPos.distanceTo(sourceStartPos);
      const lengthRatio = targetLen > 1e-6 && sourceLen > 1e-6 ? Number((targetLen / sourceLen).toFixed(4)) : null;

      rows.push({
        reason,
        segment: `${startKey}->${endKey}`,
        target: {
          start: targetStart.name,
          end: targetEnd.name,
        },
        source: {
          start: sourceStart.name,
          end: sourceEnd.name,
        },
        sourceCanonical: {
          start: mappedStartCanonical,
          end: mappedEndCanonical,
        },
        segmentAngleDeg,
        segmentDot,
        targetLen: Number(targetLen.toFixed(5)),
        sourceLen: Number(sourceLen.toFixed(5)),
        lengthRatio,
        targetDir: toRoundedVec3(targetDir),
        sourceDir: toRoundedVec3(sourceDir),
      });
    }
    const targetLeftShoulder = targetMap.get("leftShoulder") || null;
    const targetRightShoulder = targetMap.get("rightShoulder") || null;
    const sourceLeftShoulder = sourceMap.get("leftShoulder") || null;
    const sourceRightShoulder = sourceMap.get("rightShoulder") || null;
    if (targetLeftShoulder && targetRightShoulder && sourceLeftShoulder && sourceRightShoulder) {
      const tls = new THREE.Vector3();
      const trs = new THREE.Vector3();
      const sls = new THREE.Vector3();
      const srs = new THREE.Vector3();
      targetLeftShoulder.getWorldPosition(tls);
      targetRightShoulder.getWorldPosition(trs);
      sourceLeftShoulder.getWorldPosition(sls);
      sourceRightShoulder.getWorldPosition(srs);
      rows.push({
        reason,
        segment: "shoulder-span",
        target: { start: targetLeftShoulder.name, end: targetRightShoulder.name },
        source: { start: sourceLeftShoulder.name, end: sourceRightShoulder.name },
        sourceCanonical: { start: "leftShoulder", end: "rightShoulder" },
        segmentAngleDeg: angleBetweenWorldSegments(tls, trs, sls, srs),
        segmentDot: null,
        targetLen: Number(tls.distanceTo(trs).toFixed(5)),
        sourceLen: Number(sls.distanceTo(srs).toFixed(5)),
        lengthRatio:
          tls.distanceTo(trs) > 1e-6 && sls.distanceTo(srs) > 1e-6
            ? Number((tls.distanceTo(trs) / sls.distanceTo(srs)).toFixed(4))
            : null,
        targetDir: toRoundedVec3(new THREE.Vector3().subVectors(trs, tls).normalize()),
        sourceDir: toRoundedVec3(new THREE.Vector3().subVectors(srs, sls).normalize()),
      });
    }
    torsoChainDiagLog = rows;
    window.__vid2modelTorsoChainDiag = rows;
    return rows;
  }

  function dumpTorsoChainDiagLog(reason = "retarget") {
    const rows = Array.isArray(torsoChainDiagLog) ? torsoChainDiagLog.slice() : [];
    window.__vid2modelTorsoChainDiag = rows;
    if (!rows.length) return;
    const suspicious = rows.filter((row) => {
      return (row.segmentAngleDeg || 0) >= 45 || (row.lengthRatio != null && (row.lengthRatio > 1.35 || row.lengthRatio < 0.75));
    });
    if (!isVerboseDiagMode() && !suspicious.length) return;
    console.log("[vid2model/diag] torso-chain", {
      reason,
      total: rows.length,
      suspicious: suspicious.length,
    });
    if (isVerboseDiagMode()) {
      console.table(rows);
    } else if (suspicious.length) {
      console.table(suspicious);
    }
  }

  function buildFootChainDiagnostics({ targetBones = [], sourceBones = [], names = {}, reason = "retarget" } = {}) {
    const targetMap = buildCanonicalBoneMap(targetBones || []);
    const sourceMap = buildCanonicalBoneMap(sourceBones || []);
    const rows = [];
    const legSides = ["left", "right"];
    for (const side of legSides) {
      const footKey = `${side}Foot`;
      const toesKey = `${side}Toes`;
      const targetFoot = targetMap.get(footKey) || null;
      const targetToes = targetMap.get(toesKey) || getPrimaryChildBone(targetFoot, canonicalBoneKey, tmpWorldPosA, tmpWorldPosB) || null;
      if (!targetFoot || !targetToes) continue;
      const mappedFootCanonical = canonicalBoneKey(names[targetFoot.name] || footKey) || footKey;
      const mappedToesCanonical = canonicalBoneKey(names[targetToes.name] || toesKey) || toesKey;
      const sourceFoot = sourceMap.get(mappedFootCanonical) || null;
      const sourceToes = sourceMap.get(mappedToesCanonical) || getPrimaryChildBone(sourceFoot, canonicalBoneKey, tmpWorldPosA, tmpWorldPosB) || null;
      if (!sourceFoot || !sourceToes) continue;

      const targetFootPos = new THREE.Vector3();
      const targetToesPos = new THREE.Vector3();
      const sourceFootPos = new THREE.Vector3();
      const sourceToesPos = new THREE.Vector3();
      targetFoot.getWorldPosition(targetFootPos);
      targetToes.getWorldPosition(targetToesPos);
      sourceFoot.getWorldPosition(sourceFootPos);
      sourceToes.getWorldPosition(sourceToesPos);

      const footAngleDeg = angleBetweenWorldSegments(targetFootPos, targetToesPos, sourceFootPos, sourceToesPos);
      const targetDir = targetToesPos.sub(targetFootPos).normalize();
      const sourceDir = sourceToesPos.sub(sourceFootPos).normalize();
      const footDot =
        targetDir.lengthSq() > 1e-10 && sourceDir.lengthSq() > 1e-10
          ? Number(targetDir.dot(sourceDir).toFixed(4))
          : null;

      rows.push({
        side,
        reason,
        target: {
          foot: targetFoot.name,
          toes: targetToes.name,
        },
        source: {
          foot: sourceFoot.name,
          toes: sourceToes.name,
        },
        sourceCanonical: {
          foot: mappedFootCanonical,
          toes: mappedToesCanonical,
        },
        footAngleDeg,
        footDot,
        footMirrored: footDot != null ? footDot < 0 : null,
        targetFootDir: toRoundedVec3(targetDir),
        sourceFootDir: toRoundedVec3(sourceDir),
      });
    }
    footChainDiagLog = rows;
    window.__vid2modelFootChainDiag = rows;
    return rows;
  }

  function dumpFootChainDiagLog(reason = "retarget") {
    const rows = Array.isArray(footChainDiagLog) ? footChainDiagLog.slice() : [];
    window.__vid2modelFootChainDiag = rows;
    if (!rows.length) return;
    const suspicious = rows.filter((row) => row.footMirrored === true || (row.footAngleDeg || 0) >= 45);
    if (!isVerboseDiagMode() && !suspicious.length) return;
    console.log("[vid2model/diag] foot-chain", {
      reason,
      total: rows.length,
      suspicious: suspicious.length,
    });
    if (isVerboseDiagMode() || !suspicious.length) {
      console.table(rows);
    } else {
      console.table(suspicious);
    }
  }

  function dumpFootCorrectionDebug(reason = "retarget") {
    const rows = Array.isArray(window.__vid2modelFootCorrectionDebug)
      ? window.__vid2modelFootCorrectionDebug.slice()
      : [];
    window.__vid2modelFootCorrectionDebug = rows;
    if (!rows.length) return;
    const suspicious = rows.filter((row) => {
      const before = Number.isFinite(row?.footDotBefore) ? row.footDotBefore : null;
      const after = Number.isFinite(row?.footDotAfter) ? row.footDotAfter : null;
      if (before == null || after == null) return false;
      return after <= before + 1e-4;
    });
    if (!isVerboseDiagMode() && !suspicious.length) return;
    console.log("[vid2model/diag] foot-correction-debug", {
      reason,
      total: rows.length,
      suspicious: suspicious.length,
    });
    console.table(suspicious.length ? suspicious : rows);
  }

  return {
    resetRestCorrectionLog,
    resetLegChainDiagLog,
    resetArmChainDiagLog,
    resetTorsoChainDiagLog,
    resetFootChainDiagLog,
    recordRestCorrectionLog,
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
  };
}
