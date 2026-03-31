import * as THREE from "three";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";

const MIXAMO_VRM_RIG_MAP = {
  mixamorigHips: "hips",
  mixamorigSpine: "spine",
  mixamorigSpine1: "chest",
  mixamorigSpine2: "upperChest",
  mixamorigNeck: "neck",
  mixamorigHead: "head",
  mixamorigLeftShoulder: "leftShoulder",
  mixamorigLeftArm: "leftUpperArm",
  mixamorigLeftForeArm: "leftLowerArm",
  mixamorigLeftHand: "leftHand",
  mixamorigLeftHandThumb1: "leftThumbMetacarpal",
  mixamorigLeftHandThumb2: "leftThumbProximal",
  mixamorigLeftHandThumb3: "leftThumbDistal",
  mixamorigLeftHandIndex1: "leftIndexProximal",
  mixamorigLeftHandIndex2: "leftIndexIntermediate",
  mixamorigLeftHandIndex3: "leftIndexDistal",
  mixamorigLeftHandMiddle1: "leftMiddleProximal",
  mixamorigLeftHandMiddle2: "leftMiddleIntermediate",
  mixamorigLeftHandMiddle3: "leftMiddleDistal",
  mixamorigLeftHandRing1: "leftRingProximal",
  mixamorigLeftHandRing2: "leftRingIntermediate",
  mixamorigLeftHandRing3: "leftRingDistal",
  mixamorigLeftHandPinky1: "leftLittleProximal",
  mixamorigLeftHandPinky2: "leftLittleIntermediate",
  mixamorigLeftHandPinky3: "leftLittleDistal",
  mixamorigRightShoulder: "rightShoulder",
  mixamorigRightArm: "rightUpperArm",
  mixamorigRightForeArm: "rightLowerArm",
  mixamorigRightHand: "rightHand",
  mixamorigRightHandPinky1: "rightLittleProximal",
  mixamorigRightHandPinky2: "rightLittleIntermediate",
  mixamorigRightHandPinky3: "rightLittleDistal",
  mixamorigRightHandRing1: "rightRingProximal",
  mixamorigRightHandRing2: "rightRingIntermediate",
  mixamorigRightHandRing3: "rightRingDistal",
  mixamorigRightHandMiddle1: "rightMiddleProximal",
  mixamorigRightHandMiddle2: "rightMiddleIntermediate",
  mixamorigRightHandMiddle3: "rightMiddleDistal",
  mixamorigRightHandIndex1: "rightIndexProximal",
  mixamorigRightHandIndex2: "rightIndexIntermediate",
  mixamorigRightHandIndex3: "rightIndexDistal",
  mixamorigRightHandThumb1: "rightThumbMetacarpal",
  mixamorigRightHandThumb2: "rightThumbProximal",
  mixamorigRightHandThumb3: "rightThumbDistal",
  mixamorigLeftUpLeg: "leftUpperLeg",
  mixamorigLeftLeg: "leftLowerLeg",
  mixamorigLeftFoot: "leftFoot",
  mixamorigLeftToeBase: "leftToes",
  mixamorigRightUpLeg: "rightUpperLeg",
  mixamorigRightLeg: "rightLowerLeg",
  mixamorigRightFoot: "rightFoot",
  mixamorigRightToeBase: "rightToes",
};

async function fetchAnimationBlob(url) {
  const response = await fetch(url, {
    mode: "cors",
    credentials: "omit",
    headers: {
      Accept: "application/octet-stream,*/*",
    },
  });
  if (!response.ok) {
    throw new Error(`Failed to load animation: ${response.status} ${response.statusText}`);
  }
  return response.blob();
}

export async function loadDefaultVrmAnimation(url, vrm) {
  if (!url || !vrm?.humanoid) return null;

  const animBlob = await fetchAnimationBlob(url);
  const animUrl = URL.createObjectURL(animBlob);
  const loader = new FBXLoader();

  try {
    const asset = await new Promise((resolve, reject) => {
      loader.load(animUrl, resolve, undefined, reject);
    });

    const clip = THREE.AnimationClip.findByName(asset.animations, "mixamo.com");
    if (!clip) {
      throw new Error("No Mixamo animation found in FBX file");
    }

    const tracks = [];
    const restRotationInverse = new THREE.Quaternion();
    const parentRestWorldRotation = new THREE.Quaternion();
    const quat = new THREE.Quaternion();
    const vec3 = new THREE.Vector3();

    const hipsNode = asset.getObjectByName("mixamorigHips");
    if (!hipsNode) {
      throw new Error("No hips bone found in animation");
    }

    const motionHipsHeight = Math.max(1e-6, hipsNode.position.y || 1);
    const vrmBoneNode = (boneName) =>
      vrm.humanoid.getRawBoneNode?.(boneName) ||
      vrm.humanoid.getNormalizedBoneNode?.(boneName) ||
      null;
    const vrmHipsNode = vrmBoneNode("hips");
    const vrmHipsY = vrmHipsNode ? vrmHipsNode.getWorldPosition(vec3).y : null;
    const vrmRootY = vrm.scene.getWorldPosition(vec3).y;
    if (!Number.isFinite(vrmHipsY) || !Number.isFinite(vrmRootY)) {
      throw new Error("Could not determine VRM hips position");
    }
    const hipsPositionScale = Math.abs(vrmHipsY - vrmRootY) / motionHipsHeight;

    clip.tracks.forEach((track) => {
      const [mixamoRigName, propertyName] = track.name.split(".");
      const vrmBoneName = MIXAMO_VRM_RIG_MAP[mixamoRigName];
      const vrmNode = vrmBoneNode(vrmBoneName);
      const vrmNodeName = vrmNode?.name;
      const mixamoRigNode = asset.getObjectByName(mixamoRigName);

      if (vrmNodeName == null || mixamoRigNode == null) return;

      mixamoRigNode.getWorldQuaternion(restRotationInverse).invert();
      mixamoRigNode.parent?.getWorldQuaternion(parentRestWorldRotation);

      if (track instanceof THREE.QuaternionKeyframeTrack) {
        const values = track.values.slice();
        for (let i = 0; i < values.length; i += 4) {
          quat.fromArray(values.slice(i, i + 4));
          quat.premultiply(parentRestWorldRotation).multiply(restRotationInverse);
          if (vrm.meta?.metaVersion === "0") {
            quat.x *= -1;
            quat.z *= -1;
          }
          quat.toArray(values, i);
        }
        tracks.push(new THREE.QuaternionKeyframeTrack(`${vrmNodeName}.${propertyName}`, track.times, values));
      } else if (track instanceof THREE.VectorKeyframeTrack) {
        const values = track.values.map((v, i) =>
          (vrm.meta?.metaVersion === "0" && i % 3 !== 1 ? -v : v) * hipsPositionScale
        );
        tracks.push(new THREE.VectorKeyframeTrack(`${vrmNodeName}.${propertyName}`, track.times, values));
      }
    });

    if (!tracks.length) return null;
    return new THREE.AnimationClip("vrmDefaultIdle", clip.duration, tracks);
  } finally {
    URL.revokeObjectURL(animUrl);
  }
}

export const DEFAULT_VRM_ANIMATION_URL =
  new URL("../../assets/animations/FightIdle.fbx", import.meta.url).href;
