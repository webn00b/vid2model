import * as THREE from "three";

const _childDirV1 = new THREE.Vector3();
const _childDirV2 = new THREE.Vector3();
const _childDirQ1 = new THREE.Quaternion();

export function getBoneDepth(bone) {
  let depth = 0;
  let node = bone?.parent || null;
  while (node) {
    depth += 1;
    node = node.parent || null;
  }
  return depth;
}

export function getPrimaryChildDirectionLocal(bone) {
  if (!bone?.isBone) return null;
  bone.getWorldPosition(_childDirV1);
  let bestChild = null;
  let bestLenSq = 0;
  for (const child of bone.children || []) {
    if (!child?.isBone) continue;
    child.getWorldPosition(_childDirV2);
    const lenSq = _childDirV2.distanceToSquared(_childDirV1);
    if (lenSq > bestLenSq) {
      bestLenSq = lenSq;
      bestChild = child;
    }
  }
  if (!bestChild || bestLenSq < 1e-10) return null;
  bestChild.getWorldPosition(_childDirV2);
  const dir = new THREE.Vector3().copy(_childDirV2).sub(_childDirV1);
  bone.getWorldQuaternion(_childDirQ1);
  dir.applyQuaternion(_childDirQ1.invert());
  if (dir.lengthSq() < 1e-10) return null;
  return dir.normalize();
}

export function buildLegChainPairs(pairs, side) {
  if (!Array.isArray(pairs) || !side) return null;
  const byCanonical = new Map(pairs.map((pair) => [pair.canonical, pair]));
  const upper = byCanonical.get(`${side}UpperLeg`) || null;
  const lower = byCanonical.get(`${side}LowerLeg`) || null;
  const foot = byCanonical.get(`${side}Foot`) || null;
  const toes = byCanonical.get(`${side}Toes`) || null;
  if (!upper || !lower || !foot) return null;
  return {
    side,
    upper,
    lower,
    foot,
    toes,
    enableKneePlaneCorrection: false,
    enableUpperLegDirectionCorrection: false,
    enableShinDirectionCorrection: false,
    enableFootDirectionCorrection: false,
    enableFootPlaneCorrection: false,
    enableFootMirrorCorrection: false,
  };
}

export function buildArmChainPairs(pairs, side) {
  if (!Array.isArray(pairs) || !side) return null;
  const byCanonical = new Map(pairs.map((pair) => [pair.canonical, pair]));
  const shoulder = byCanonical.get(`${side}Shoulder`) || null;
  const upper = byCanonical.get(`${side}UpperArm`) || null;
  const lower = byCanonical.get(`${side}LowerArm`) || null;
  const hand = byCanonical.get(`${side}Hand`) || null;
  if (!upper || !lower || !hand) return null;
  return {
    side,
    shoulder,
    upper,
    lower,
    hand,
    enableShoulderDirectionCorrection: false,
    enableUpperArmDirectionCorrection: false,
    enableElbowPlaneCorrection: false,
    enableForearmDirectionCorrection: false,
  };
}
