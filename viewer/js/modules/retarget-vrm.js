import * as THREE from "three";
import {
  applyPairInvertRotationOverride,
  applyPairRestOrientationCorrection,
  buildProfiledChains,
  computeRetargetPosScale,
  createRetargetPair,
  initializeRetargetPairsRestState,
  initializePairRestWorldDelta,
} from "./retarget-plan-utils.js";

const PARENT_RELATIVE_REST_DELTA_CANONICAL = new Set([
  "spine",
  "chest",
  "upperChest",
  "neck",
  "head",
  "leftShoulder",
  "rightShoulder",
  "leftUpperArm",
  "rightUpperArm",
  "leftLowerArm",
  "rightLowerArm",
  "leftHand",
  "rightHand",
  "leftLowerLeg",
  "rightLowerLeg",
]);

const WORLD_REST_TRANSFER_CANONICAL = new Set([
  "leftUpperLeg",
  "rightUpperLeg",
  "leftFoot",
  "rightFoot",
  "leftToes",
  "rightToes",
]);
const LEG_LOCAL_CORRECTION_CANONICAL = new Set([
  "leftUpperLeg",
  "rightUpperLeg",
  "leftLowerLeg",
  "rightLowerLeg",
  "leftFoot",
  "rightFoot",
  "leftToes",
  "rightToes",
]);
export function buildVrmDirectBodyPlan({
  targetBones,
  sourceBones,
  namesTargetToSource,
  mixer,
  modelRoot,
  buildRestOrientationCorrection,
  profile = null,
}) {
  const sourceByName = new Map((sourceBones || []).map((b) => [b.name, b]));
  const pairs = [];
  for (const targetBone of targetBones || []) {
    const sourceName = namesTargetToSource[targetBone.name];
    if (!sourceName) continue;
    const sourceBone = sourceByName.get(sourceName);
    if (!sourceBone) continue;
    pairs.push(createRetargetPair(targetBone, sourceBone));
  }

  if (!pairs.length) return null;
  pairs.sort((a, b) => a.depth - b.depth);

  const sourceTime = mixer ? mixer.time : 0;
  if (mixer) mixer.setTime(0);
  modelRoot?.updateMatrixWorld(true);

  initializeRetargetPairsRestState(pairs);

  for (const pair of pairs) {
    if (pair.isHips) continue;
    const disableWorldRestTransfer = !!(
      pair.canonical &&
      profile?.disableWorldRestTransferByCanonical?.[pair.canonical]
    );
    const disableParentRelativeRestDelta = !!(
      pair.canonical &&
      profile?.disableParentRelativeRestDeltaByCanonical?.[pair.canonical]
    );
    applyPairInvertRotationOverride(pair, profile);
    initializePairRestWorldDelta(pair);
    if (LEG_LOCAL_CORRECTION_CANONICAL.has(pair.canonical)) {
      if (!applyPairRestOrientationCorrection(pair, buildRestOrientationCorrection)) {
        continue;
      }
      continue;
    }
    if (!disableWorldRestTransfer && WORLD_REST_TRANSFER_CANONICAL.has(pair.canonical) && pair.target.parent) {
      pair.useWorldRestTransfer = true;
      continue;
    }
    if (
      !disableParentRelativeRestDelta &&
      PARENT_RELATIVE_REST_DELTA_CANONICAL.has(pair.canonical) &&
      pair.target.parent
    ) {
      pair.useParentRelativeRestDelta = true;
      continue;
    }
    applyPairRestOrientationCorrection(pair, buildRestOrientationCorrection);
  }

  const posScale = computeRetargetPosScale(sourceBones, targetBones || []);

  if (mixer) mixer.setTime(sourceTime);
  modelRoot?.updateMatrixWorld(true);

  const { legChains, armChains } = buildProfiledChains(pairs, profile);

  return {
    pairs,
    legChains,
    armChains,
    uniqueSkeletons: [{ bones: targetBones || [] }],
    posScale,
    yawOffset: 0,
    invertRotationDelta: true,
    calibratedPairs: pairs.filter((pair) =>
      pair.useWorldRestTransfer || pair.useParentRelativeRestDelta || pair.hasRestCorrection
    ).length,
    directVrm: true,
  };
}
