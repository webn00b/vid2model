import * as THREE from "three";

export function createViewerSourceAxesDebug({
  scene,
  buildCanonicalBoneMap,
}) {
  let sourceAxesDebug = null;
  const sourceAxisOrigin = new THREE.Vector3();
  const sourceAxisHead = new THREE.Vector3();
  const sourceAxisLeft = new THREE.Vector3();
  const sourceAxisRight = new THREE.Vector3();
  const sourceAxisForward = new THREE.Vector3();
  const sourceAxisSide = new THREE.Vector3();
  const sourceAxisUp = new THREE.Vector3();

  function clearSourceAxesDebug() {
    if (!sourceAxesDebug) return;
    scene.remove(sourceAxesDebug.group);
    sourceAxesDebug = null;
  }

  function createSourceAxesDebug(skeleton) {
    clearSourceAxesDebug();
    const group = new THREE.Group();
    const axisLength = 55;
    const headLength = 12;
    const headWidth = 6;
    const upArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), axisLength, 0x22c55e, headLength, headWidth);
    const rightArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), axisLength, 0xef4444, headLength, headWidth);
    const forwardArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), axisLength, 0x3b82f6, headLength, headWidth);
    group.add(upArrow, rightArrow, forwardArrow);
    scene.add(group);
    sourceAxesDebug = {
      skeleton,
      group,
      upArrow,
      rightArrow,
      forwardArrow,
    };
    updateSourceAxesDebug();
  }

  function updateSourceAxesDebug() {
    if (!sourceAxesDebug?.skeleton?.bones?.length) return;
    const canonicalMap = buildCanonicalBoneMap(sourceAxesDebug.skeleton.bones);
    const hips = canonicalMap.get("hips") || sourceAxesDebug.skeleton.bones[0];
    if (!hips) return;
    hips.updateMatrixWorld(true);
    hips.getWorldPosition(sourceAxisOrigin);

    const head = canonicalMap.get("head") || canonicalMap.get("neck") || canonicalMap.get("upperChest") || canonicalMap.get("chest");
    const leftHand = canonicalMap.get("leftHand");
    const rightHand = canonicalMap.get("rightHand");

    if (head) {
      head.getWorldPosition(sourceAxisHead);
      sourceAxisUp.copy(sourceAxisHead).sub(sourceAxisOrigin).normalize();
    } else {
      sourceAxisUp.set(0, 1, 0);
    }
    if (sourceAxisUp.lengthSq() < 1e-8) sourceAxisUp.set(0, 1, 0);

    if (leftHand && rightHand) {
      leftHand.getWorldPosition(sourceAxisLeft);
      rightHand.getWorldPosition(sourceAxisRight);
      sourceAxisSide.copy(sourceAxisRight).sub(sourceAxisLeft).normalize();
    } else {
      sourceAxisSide.set(1, 0, 0);
    }
    if (sourceAxisSide.lengthSq() < 1e-8) sourceAxisSide.set(1, 0, 0);

    sourceAxisForward.crossVectors(sourceAxisSide, sourceAxisUp).normalize();
    if (sourceAxisForward.lengthSq() < 1e-8) sourceAxisForward.set(0, 0, 1);

    const axisLength = 55;
    sourceAxesDebug.upArrow.position.copy(sourceAxisOrigin);
    sourceAxesDebug.upArrow.setDirection(sourceAxisUp);
    sourceAxesDebug.upArrow.setLength(axisLength, 12, 6);

    sourceAxesDebug.rightArrow.position.copy(sourceAxisOrigin);
    sourceAxesDebug.rightArrow.setDirection(sourceAxisSide);
    sourceAxesDebug.rightArrow.setLength(axisLength, 12, 6);

    sourceAxesDebug.forwardArrow.position.copy(sourceAxisOrigin);
    sourceAxesDebug.forwardArrow.setDirection(sourceAxisForward);
    sourceAxesDebug.forwardArrow.setLength(axisLength, 12, 6);
  }

  return {
    clearSourceAxesDebug,
    createSourceAxesDebug,
    updateSourceAxesDebug,
  };
}
