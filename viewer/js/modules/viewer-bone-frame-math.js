import * as THREE from "three";
import { canonicalBoneKey } from "./bone-utils.js";

export function createViewerBoneFrameMath({
  getPreferredChildBone,
  loadRigProfile,
  getRetargetStage,
  getModelRigFingerprint,
  getModelLabel,
  recordRestCorrectionLog,
}) {
  const _calibV1 = new THREE.Vector3();
  const _calibV2 = new THREE.Vector3();
  const _calibV3 = new THREE.Vector3();
  const _calibV4 = new THREE.Vector3();
  const _calibQ1 = new THREE.Quaternion();
  const _calibQ2 = new THREE.Quaternion();
  const _calibM1 = new THREE.Matrix4();

  function getPrimaryChildDirectionLocal(bone, outDir) {
    if (!bone || !outDir) return false;
    bone.getWorldPosition(_calibV1);
    const preferredChild = getPreferredChildBone(bone);
    if (preferredChild) {
      preferredChild.getWorldPosition(_calibV2);
      outDir.copy(_calibV2).sub(_calibV1);
      bone.getWorldQuaternion(_calibQ1);
      outDir.applyQuaternion(_calibQ1.invert());
      if (outDir.lengthSq() >= 1e-10) {
        outDir.normalize();
        return true;
      }
    }
    let bestChild = null;
    let bestLenSq = 0;
    for (const child of bone.children || []) {
      if (!child?.isBone) continue;
      child.getWorldPosition(_calibV2);
      const lenSq = _calibV2.distanceToSquared(_calibV1);
      if (lenSq > bestLenSq) {
        bestLenSq = lenSq;
        bestChild = child;
      }
    }
    if (!bestChild || bestLenSq < 1e-8) return false;
    bestChild.getWorldPosition(_calibV2);
    outDir.copy(_calibV2).sub(_calibV1);
    bone.getWorldQuaternion(_calibQ1);
    outDir.applyQuaternion(_calibQ1.invert());
    if (outDir.lengthSq() < 1e-10) return false;
    outDir.normalize();
    return true;
  }

  function getReferenceDirectionLocal(bone, primaryDir, outDir) {
    if (!bone || !primaryDir || !outDir) return false;
    bone.getWorldPosition(_calibV1);
    bone.getWorldQuaternion(_calibQ1);
    const canonical = canonicalBoneKey(bone.name) || "";
    const invBoneQ = _calibQ1.invert();
    const tryWorldVector = (worldVector) => {
      outDir.copy(worldVector).applyQuaternion(invBoneQ);
      outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
      if (outDir.lengthSq() < 1e-8) return false;
      outDir.normalize();
      return true;
    };

    if (canonical === "leftUpperLeg" || canonical === "rightUpperLeg") {
      const siblingCanonical = canonical === "leftUpperLeg" ? "rightUpperLeg" : "leftUpperLeg";
      for (const sibling of bone.parent?.children || []) {
        if (!sibling?.isBone || sibling === bone) continue;
        if ((canonicalBoneKey(sibling.name) || "") !== siblingCanonical) continue;
        sibling.getWorldPosition(_calibV2);
        outDir.copy(_calibV2).sub(_calibV1).applyQuaternion(invBoneQ);
        outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
        if (outDir.lengthSq() >= 1e-8) {
          outDir.normalize();
          return true;
        }
      }
    }

    if (bone.parent?.isBone) {
      bone.parent.getWorldPosition(_calibV2);
      outDir.copy(_calibV2).sub(_calibV1).applyQuaternion(invBoneQ);
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
      child.getWorldPosition(_calibV2);
      outDir.copy(_calibV2).sub(_calibV1);
      const lenSq = outDir.lengthSq();
      if (lenSq > bestLenSq) {
        bestLenSq = lenSq;
        bestChild = child;
      }
    }
    if (bestChild) {
      for (const child of bone.children || []) {
        if (!child?.isBone || child === bestChild) continue;
        child.getWorldPosition(_calibV2);
        outDir.copy(_calibV2).sub(_calibV1).applyQuaternion(invBoneQ);
        outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
        if (outDir.lengthSq() >= 1e-8) {
          outDir.normalize();
          return true;
        }
      }
    }

    return (
      tryWorldVector(_calibV2.set(1, 0, 0)) ||
      tryWorldVector(_calibV2.set(0, 1, 0)) ||
      tryWorldVector(_calibV2.set(0, 0, 1))
    );
  }

  function buildBoneLocalFrame(bone, outQ, options = null) {
    if (!bone || !outQ) return false;
    if (!getPrimaryChildDirectionLocal(bone, _calibV1)) return false;
    if (!getReferenceDirectionLocal(bone, _calibV1, _calibV2)) return false;
    if (options?.flipReferenceAxis) {
      _calibV2.multiplyScalar(-1);
    }
    _calibV3.crossVectors(_calibV2, _calibV1);
    if (_calibV3.lengthSq() < 1e-8) return false;
    _calibV3.normalize();
    _calibV4.crossVectors(_calibV1, _calibV3);
    if (_calibV4.lengthSq() < 1e-8) return false;
    _calibV4.normalize();
    _calibM1.makeBasis(_calibV4, _calibV1, _calibV3);
    outQ.setFromRotationMatrix(_calibM1).normalize();
    return true;
  }

  function buildRestOrientationCorrection(sourceBone, targetBone) {
    const canonical =
      canonicalBoneKey(targetBone?.name) ||
      canonicalBoneKey(sourceBone?.name) ||
      "";
    const restCorrectionProfile = loadRigProfile(getModelRigFingerprint(), getRetargetStage(), getModelLabel());
    const flipReferenceAxis = !!(
      canonical &&
      restCorrectionProfile?.flipReferenceAxisByCanonical?.[canonical]
    );
    const twistDeg = canonical
      ? Number(restCorrectionProfile?.twistDegByCanonical?.[canonical] || 0)
      : 0;
    const offsetEntry = canonical
      ? restCorrectionProfile?.restCorrectionEulerDegByCanonical?.[canonical] || null
      : null;
    const profileOffset =
      offsetEntry && (offsetEntry.x || offsetEntry.y || offsetEntry.z)
        ? new THREE.Quaternion().setFromEuler(
            new THREE.Euler(
              THREE.MathUtils.degToRad(Number(offsetEntry.x || 0)),
              THREE.MathUtils.degToRad(Number(offsetEntry.y || 0)),
              THREE.MathUtils.degToRad(Number(offsetEntry.z || 0)),
              "XYZ"
            )
          ).normalize()
        : null;
    const profileOffsetDeg = profileOffset
      ? {
          x: Number(Number(offsetEntry.x || 0).toFixed(2)),
          y: Number(Number(offsetEntry.y || 0).toFixed(2)),
          z: Number(Number(offsetEntry.z || 0).toFixed(2)),
        }
      : null;
    let twistQ = null;
    if (Math.abs(twistDeg) > 1e-4) {
      const twistAxis = new THREE.Vector3();
      if (getPrimaryChildDirectionLocal(targetBone, twistAxis)) {
        twistQ = new THREE.Quaternion()
          .setFromAxisAngle(twistAxis, THREE.MathUtils.degToRad(twistDeg))
          .normalize();
      }
    }
    const sourceFrame = _calibQ1;
    const targetFrame = _calibQ2;
    if (
      buildBoneLocalFrame(sourceBone, sourceFrame) &&
      buildBoneLocalFrame(targetBone, targetFrame, { flipReferenceAxis })
    ) {
      const corr = new THREE.Quaternion().copy(targetFrame).multiply(sourceFrame.invert()).normalize();
      const autoAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
      if (profileOffset) {
        corr.premultiply(profileOffset).normalize();
      }
      if (twistQ) {
        corr.premultiply(twistQ).normalize();
      }
      const finalAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
      recordRestCorrectionLog({
        canonical: canonical || "unknown",
        target: targetBone?.name || "",
        source: sourceBone?.name || "",
        method: "local-frame",
        autoAngleDeg,
        profileOffsetDeg,
        flipReferenceAxis,
        twistDeg: Number(twistDeg.toFixed(2)),
        finalAngleDeg,
      });
      if (Math.abs(corr.w) > 0.9995) {
        return twistQ || profileOffset || null;
      }
      return corr;
    }
    const sourceDir = new THREE.Vector3();
    const targetDir = new THREE.Vector3();
    if (!getPrimaryChildDirectionLocal(sourceBone, sourceDir)) return null;
    if (!getPrimaryChildDirectionLocal(targetBone, targetDir)) return null;
    const dot = Math.max(-1, Math.min(1, sourceDir.dot(targetDir)));
    if (dot > 0.9995) {
      recordRestCorrectionLog({
        canonical: canonical || "unknown",
        target: targetBone?.name || "",
        source: sourceBone?.name || "",
        method: "child-direction",
        autoAngleDeg: 0,
        profileOffsetDeg,
        flipReferenceAxis,
        twistDeg: Number(twistDeg.toFixed(2)),
        finalAngleDeg: profileOffset
          ? Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(profileOffset.w)))).toFixed(3))
          : 0,
      });
      if (twistQ && profileOffset) {
        return new THREE.Quaternion().copy(twistQ).premultiply(profileOffset).normalize();
      }
      return twistQ || profileOffset || null;
    }
    const corr = new THREE.Quaternion().setFromUnitVectors(sourceDir, targetDir).normalize();
    const autoAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
    if (profileOffset) {
      corr.premultiply(profileOffset).normalize();
    }
    if (twistQ) {
      corr.premultiply(twistQ).normalize();
    }
    const finalAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
    recordRestCorrectionLog({
      canonical: canonical || "unknown",
      target: targetBone?.name || "",
      source: sourceBone?.name || "",
      method: "child-direction",
      autoAngleDeg,
      profileOffsetDeg,
      flipReferenceAxis,
      twistDeg: Number(twistDeg.toFixed(2)),
      finalAngleDeg,
    });
    return corr;
  }

  return {
    getPrimaryChildDirectionLocal,
    getReferenceDirectionLocal,
    buildBoneLocalFrame,
    buildRestOrientationCorrection,
  };
}
