import * as THREE from "three";

export function createViewerAlignmentTools({
  canonicalBoneKey,
  diag,
  camera,
  controls,
  getSkeletonObj,
  getSourceResult,
  getModelRoot,
  getModelSkinnedMesh,
  hasSourceOverlay,
  setSourceOverlayYaw,
  updateSourceOverlay,
  estimateFacingYawOffset,
}) {
  const tmpWorldPosA = new THREE.Vector3();
  const tmpWorldPosB = new THREE.Vector3();
  const tmpWorldDelta = new THREE.Vector3();

  function objectHeight(root) {
    if (!root) return 0;
    const box = new THREE.Box3().setFromObject(root);
    const height = box.max.y - box.min.y;
    return Number.isFinite(height) && height > 0 ? height : 0;
  }

  function fitToSkeleton(root) {
    const box = new THREE.Box3().setFromObject(root);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const dist = maxDim * 1.8;
    camera.position.set(center.x + dist, center.y + dist * 0.8, center.z + dist);
    controls.target.copy(center);
    controls.update();
  }

  function findHipsBone(bones) {
    if (!bones?.length) return null;
    return bones.find((bone) => canonicalBoneKey(bone.name) === "hips") || bones[0] || null;
  }

  function findFootLevelY(bones) {
    if (!bones?.length) return null;
    const footKeys = new Set(["leftFoot", "rightFoot", "leftToes", "rightToes"]);
    const ys = [];
    const point = new THREE.Vector3();
    for (const bone of bones) {
      const key = canonicalBoneKey(bone.name);
      if (!footKeys.has(key)) continue;
      bone.getWorldPosition(point);
      ys.push(point.y);
    }
    if (!ys.length) return null;
    return ys.reduce((sum, value) => sum + value, 0) / ys.length;
  }

  function computeHipsWorldError(sourceBones, targetBones) {
    const sourceHips = findHipsBone(sourceBones);
    const targetHips = findHipsBone(targetBones);
    if (!sourceHips || !targetHips) return null;
    sourceHips.getWorldPosition(tmpWorldPosA);
    targetHips.getWorldPosition(tmpWorldPosB);
    return tmpWorldPosA.distanceTo(tmpWorldPosB);
  }

  function getModelSkeletonRootBone() {
    const modelSkinnedMesh = getModelSkinnedMesh();
    if (!modelSkinnedMesh?.skeleton?.bones?.length) return null;
    let node =
      findHipsBone(modelSkinnedMesh.skeleton.bones) ||
      modelSkinnedMesh.skeleton.bones[0];
    while (node?.parent?.isBone) {
      node = node.parent;
    }
    return node || null;
  }

  function applyWorldDeltaToObject(obj, deltaWorld) {
    if (!obj || !deltaWorld || deltaWorld.lengthSq() < 1e-12) return;
    if (!obj.parent) {
      obj.position.add(deltaWorld);
      return;
    }
    obj.getWorldPosition(tmpWorldPosA);
    tmpWorldPosA.add(deltaWorld);
    obj.parent.worldToLocal(tmpWorldPosA);
    obj.position.copy(tmpWorldPosA);
  }

  function syncSourceDisplayToModel() {
    const skeletonObj = getSkeletonObj();
    const sourceResult = getSourceResult();
    const modelRoot = getModelRoot();
    const modelSkinnedMesh = getModelSkinnedMesh();
    if (
      !skeletonObj ||
      !sourceResult?.skeleton?.bones?.length ||
      !modelRoot ||
      !modelSkinnedMesh?.skeleton?.bones?.length
    ) {
      return;
    }
    skeletonObj.position.set(0, 0, 0);
    skeletonObj.scale.setScalar(1);
    skeletonObj.updateMatrixWorld(true);
    modelRoot.updateMatrixWorld(true);

    const sourceHeight = objectHeight(skeletonObj);
    const targetHeight = objectHeight(modelRoot);
    if (sourceHeight > 1e-6 && targetHeight > 1e-6) {
      const displayScale = Math.max(
        1e-6,
        Math.min(1e6, targetHeight / sourceHeight)
      );
      skeletonObj.scale.setScalar(displayScale);
    }
    skeletonObj.updateMatrixWorld(true);

    const sourceHips = findHipsBone(sourceResult.skeleton.bones);
    const targetHips = findHipsBone(modelSkinnedMesh.skeleton.bones);
    if (sourceHips && targetHips) {
      const src = new THREE.Vector3();
      const dst = new THREE.Vector3();
      sourceHips.getWorldPosition(src);
      targetHips.getWorldPosition(dst);
      skeletonObj.position.add(dst.sub(src));
    }
    skeletonObj.updateMatrixWorld(true);

    const sourceFeetY = findFootLevelY(sourceResult.skeleton.bones);
    const targetFeetY = findFootLevelY(modelSkinnedMesh.skeleton.bones);
    if (Number.isFinite(sourceFeetY) && Number.isFinite(targetFeetY)) {
      skeletonObj.position.y += targetFeetY - sourceFeetY;
      skeletonObj.updateMatrixWorld(true);
    }
    let overlayYaw = 0;
    if (hasSourceOverlay()) {
      overlayYaw = estimateFacingYawOffset(
        sourceResult.skeleton.bones,
        modelSkinnedMesh.skeleton.bones
      );
      setSourceOverlayYaw(0);
    }
    diag("display-sync", {
      sourceHeight: Number(sourceHeight.toFixed(4)),
      targetHeight: Number(targetHeight.toFixed(4)),
      displayScale: Number(skeletonObj.scale.y.toFixed(6)),
      overlayYawDeg: Number(((overlayYaw * 180) / Math.PI).toFixed(2)),
    });
    updateSourceOverlay();
  }

  function alignSourceHipsToModel(lockFeet = false) {
    const skeletonObj = getSkeletonObj();
    const sourceResult = getSourceResult();
    const modelRoot = getModelRoot();
    const modelSkinnedMesh = getModelSkinnedMesh();
    if (
      !skeletonObj ||
      !sourceResult?.skeleton?.bones?.length ||
      !modelRoot ||
      !modelSkinnedMesh?.skeleton?.bones?.length
    ) {
      return;
    }
    skeletonObj.updateMatrixWorld(true);
    modelRoot.updateMatrixWorld(true);
    const sourceHips = findHipsBone(sourceResult.skeleton.bones);
    const targetHips = findHipsBone(modelSkinnedMesh.skeleton.bones);
    if (sourceHips && targetHips) {
      const src = new THREE.Vector3();
      const dst = new THREE.Vector3();
      sourceHips.getWorldPosition(src);
      targetHips.getWorldPosition(dst);
      skeletonObj.position.add(dst.sub(src));
    }
    skeletonObj.updateMatrixWorld(true);
    if (lockFeet) {
      const sourceFeetY = findFootLevelY(sourceResult.skeleton.bones);
      const targetFeetY = findFootLevelY(modelSkinnedMesh.skeleton.bones);
      if (Number.isFinite(sourceFeetY) && Number.isFinite(targetFeetY)) {
        skeletonObj.position.y += targetFeetY - sourceFeetY;
        skeletonObj.updateMatrixWorld(true);
      }
    }
  }

  function alignModelHipsToSource(lockFeet = false) {
    const skeletonObj = getSkeletonObj();
    const sourceResult = getSourceResult();
    const modelRoot = getModelRoot();
    const modelSkinnedMesh = getModelSkinnedMesh();
    if (
      !skeletonObj ||
      !sourceResult?.skeleton?.bones?.length ||
      !modelRoot ||
      !modelSkinnedMesh?.skeleton?.bones?.length
    ) {
      return null;
    }
    skeletonObj.updateMatrixWorld(true);
    modelRoot.updateMatrixWorld(true);
    const sourceHips = findHipsBone(sourceResult.skeleton.bones);
    const targetHips = findHipsBone(modelSkinnedMesh.skeleton.bones);
    if (!sourceHips || !targetHips) {
      return {
        applied: false,
        target: "none",
        beforeErr: null,
        deltaX: 0,
        deltaY: 0,
        deltaZ: 0,
        lockFeet,
        hipsPosErr: null,
      };
    }
    sourceHips.getWorldPosition(tmpWorldPosA);
    targetHips.getWorldPosition(tmpWorldPosB);
    tmpWorldDelta.copy(tmpWorldPosA).sub(tmpWorldPosB);
    const delta = tmpWorldDelta.clone();
    const beforeErrRaw = computeHipsWorldError(
      sourceResult.skeleton.bones,
      modelSkinnedMesh.skeleton.bones
    );
    const beforeErr = Number.isFinite(beforeErrRaw)
      ? Number(beforeErrRaw.toFixed(5))
      : null;
    const candidates = [modelRoot, getModelSkeletonRootBone()].filter(
      (node, index, arr) =>
        node && arr.findIndex((value) => value && value.uuid === node.uuid) === index
    );
    let bestTarget = "none";
    let bestErr = Number.POSITIVE_INFINITY;
    for (const node of candidates) {
      const prevPos = node.position.clone();
      applyWorldDeltaToObject(node, delta);
      modelRoot.updateMatrixWorld(true);
      const err = computeHipsWorldError(
        sourceResult.skeleton.bones,
        modelSkinnedMesh.skeleton.bones
      );
      const errNum = Number.isFinite(err) ? err : Number.POSITIVE_INFINITY;
      if (errNum < bestErr) {
        bestErr = errNum;
        bestTarget = node === modelRoot ? "modelRoot" : "skeletonRootBone";
      }
      node.position.copy(prevPos);
    }
    if (bestTarget === "skeletonRootBone") {
      const rootBone = getModelSkeletonRootBone();
      if (rootBone) {
        applyWorldDeltaToObject(rootBone, delta);
      }
    } else {
      applyWorldDeltaToObject(modelRoot, delta);
    }
    modelRoot.updateMatrixWorld(true);
    if (lockFeet) {
      const sourceFeetY = findFootLevelY(sourceResult.skeleton.bones);
      const targetFeetY = findFootLevelY(modelSkinnedMesh.skeleton.bones);
      if (Number.isFinite(sourceFeetY) && Number.isFinite(targetFeetY)) {
        modelRoot.position.y += sourceFeetY - targetFeetY;
        modelRoot.updateMatrixWorld(true);
      }
    }
    const hipsPosErrRaw = computeHipsWorldError(
      sourceResult.skeleton.bones,
      modelSkinnedMesh.skeleton.bones
    );
    const hipsPosErr = Number.isFinite(hipsPosErrRaw)
      ? Number(hipsPosErrRaw.toFixed(5))
      : null;
    return {
      applied: true,
      target: bestTarget,
      beforeErr,
      deltaX: Number(delta.x.toFixed(5)),
      deltaY: Number(delta.y.toFixed(5)),
      deltaZ: Number(delta.z.toFixed(5)),
      lockFeet,
      hipsPosErr,
    };
  }

  return {
    objectHeight,
    fitToSkeleton,
    getModelSkeletonRootBone,
    syncSourceDisplayToModel,
    alignSourceHipsToModel,
    alignModelHipsToSource,
  };
}
