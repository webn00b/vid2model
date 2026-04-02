const VRM_HUMANOID_IDENTITY_NAMES = [
  "hips",
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
  "leftUpperLeg",
  "rightUpperLeg",
  "leftLowerLeg",
  "rightLowerLeg",
  "leftFoot",
  "rightFoot",
  "leftToes",
  "rightToes",
  "leftThumbMetacarpal",
  "leftThumbProximal",
  "leftThumbDistal",
  "leftIndexProximal",
  "leftIndexIntermediate",
  "leftIndexDistal",
  "leftMiddleProximal",
  "leftMiddleIntermediate",
  "leftMiddleDistal",
  "leftRingProximal",
  "leftRingIntermediate",
  "leftRingDistal",
  "leftLittleProximal",
  "leftLittleIntermediate",
  "leftLittleDistal",
  "rightThumbMetacarpal",
  "rightThumbProximal",
  "rightThumbDistal",
  "rightIndexProximal",
  "rightIndexIntermediate",
  "rightIndexDistal",
  "rightMiddleProximal",
  "rightMiddleIntermediate",
  "rightMiddleDistal",
  "rightRingProximal",
  "rightRingIntermediate",
  "rightRingDistal",
  "rightLittleProximal",
  "rightLittleIntermediate",
  "rightLittleDistal",
];

function buildIdentityMapping(names) {
  return Object.fromEntries((names || []).map((name) => [name, name]));
}

function buildMoonGirlMapping() {
  return buildIdentityMapping(VRM_HUMANOID_IDENTITY_NAMES);
}

function buildMoonGirlUpperBodyRotationScale() {
  return {
    spine: 0.55,
    chest: 0.28,
    upperChest: 0.22,
    neck: 0.5,
    head: 0.75,
    leftShoulder: 0.06,
    rightShoulder: 0.06,
    leftUpperArm: 0.08,
    rightUpperArm: 0.08,
    leftLowerArm: 0.1,
    rightLowerArm: 0.1,
    leftHand: 0.12,
    rightHand: 0.12,
  };
}

function buildMoonGirlBodyCanonicalKeys() {
  return [
    "hips",
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
    "leftUpperLeg",
    "rightUpperLeg",
    "leftLowerLeg",
    "rightLowerLeg",
    "leftFoot",
    "rightFoot",
    "leftToes",
    "rightToes",
  ];
}

const BUILTIN_RIG_PROFILES = [
  {
    id: "en-0-v1",
    modelLabelPattern: /(^|[\\/])en_0\.vrm$/i,
    stage: "body",
    preferredMode: "skeletonutils-skinnedmesh",
    mirrorSwap: "disable",
    preferSkeletonOnRenameFallback: true,
    preferAggressiveLiveDelta: true,
    source: "repo",
  },
  {
    id: "en-0-v1",
    modelLabelPattern: /(^|[\\/])en_0\.vrm$/i,
    stage: "full",
    preferredMode: "skeletonutils-skinnedmesh",
    mirrorSwap: "disable",
    preferSkeletonOnRenameFallback: true,
    preferAggressiveLiveDelta: true,
    source: "repo",
  },
  {
    id: "moon-girl-v1",
    modelLabelPattern: /(^|[\\/])MoonGirl\.vrm$/i,
    stage: "body",
    namesTargetToSource: buildMoonGirlMapping(),
    mode: "builtin-manual-map",
    lockBuiltin: true,
    bodyCanonicalKeys: buildMoonGirlBodyCanonicalKeys(),
    preferredMode: "skeletonutils-skinnedmesh",
    forceLiveDelta: false,
    rotationScaleByCanonical: buildMoonGirlUpperBodyRotationScale(),
    mirrorSwap: "disable",
    preferSkeletonOnRenameFallback: true,
    source: "repo",
  },
  {
    id: "moon-girl-v1",
    modelLabelPattern: /(^|[\\/])MoonGirl\.vrm$/i,
    stage: "full",
    namesTargetToSource: buildMoonGirlMapping(),
    mode: "builtin-manual-map",
    lockBuiltin: true,
    preferredMode: "skeletonutils-skinnedmesh",
    forceLiveDelta: false,
    rotationScaleByCanonical: buildMoonGirlUpperBodyRotationScale(),
    mirrorSwap: "disable",
    preferSkeletonOnRenameFallback: true,
    source: "repo",
  },
];

export function getBuiltinRigProfile({ modelFingerprint = "", modelLabel = "", stage = "" } = {}) {
  const normalizedStage = String(stage || "").trim().toLowerCase();
  if (!normalizedStage) return null;
  const normalizedLabel = String(modelLabel || "");
  const normalizedFingerprint = String(modelFingerprint || "");
  return (
    BUILTIN_RIG_PROFILES.find((entry) => {
      if (entry.stage !== normalizedStage) return false;
      if (entry.modelFingerprint && entry.modelFingerprint === normalizedFingerprint) return true;
      if (entry.modelLabelPattern && entry.modelLabelPattern.test(normalizedLabel)) return true;
      return false;
    }) || null
  );
}
