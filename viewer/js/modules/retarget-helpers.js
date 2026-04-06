import * as THREE from "three";
import { canonicalBoneKey, normalizeBoneName, parseTrackName } from "./bone-utils.js";
import { buildCanonicalBoneMap, canonicalBonePreferenceScore } from "./canonical-bone-map.js";

export { buildCanonicalBoneMap } from "./canonical-bone-map.js";

const RETARGET_ALIAS = new Map([
  ["pelvis", "hips"],
  ["hip", "hips"],
  ["spine1", "chest"],
  ["spine2", "upperChest"],
  ["neck1", "neck"],
  ["leftarm", "leftUpperArm"],
  ["leftforearm", "leftLowerArm"],
  ["rightarm", "rightUpperArm"],
  ["rightforearm", "rightLowerArm"],
  ["leftupleg", "leftUpperLeg"],
  ["leftleg", "leftLowerLeg"],
  ["lefttoebase", "leftToes"],
  ["rightupleg", "rightUpperLeg"],
  ["rightleg", "rightLowerLeg"],
  ["righttoebase", "rightToes"],
  ["jbiplthumb1", "leftThumbMetacarpal"],
  ["jbiplthumb2", "leftThumbProximal"],
  ["jbiplthumb3", "leftThumbDistal"],
  ["jbiprthumb1", "rightThumbMetacarpal"],
  ["jbiprthumb2", "rightThumbProximal"],
  ["jbiprthumb3", "rightThumbDistal"],
  ["lefthandthumb1", "leftThumbMetacarpal"],
  ["lefthandthumb2", "leftThumbProximal"],
  ["lefthandthumb3", "leftThumbDistal"],
  ["lefthandindex1", "leftIndexProximal"],
  ["lefthandindex2", "leftIndexIntermediate"],
  ["lefthandindex3", "leftIndexDistal"],
  ["lefthandmiddle1", "leftMiddleProximal"],
  ["lefthandmiddle2", "leftMiddleIntermediate"],
  ["lefthandmiddle3", "leftMiddleDistal"],
  ["lefthandring1", "leftRingProximal"],
  ["lefthandring2", "leftRingIntermediate"],
  ["lefthandring3", "leftRingDistal"],
  ["lefthandpinky1", "leftLittleProximal"],
  ["lefthandpinky2", "leftLittleIntermediate"],
  ["lefthandpinky3", "leftLittleDistal"],
  ["righthandthumb1", "rightThumbMetacarpal"],
  ["righthandthumb2", "rightThumbProximal"],
  ["righthandthumb3", "rightThumbDistal"],
  ["righthandindex1", "rightIndexProximal"],
  ["righthandindex2", "rightIndexIntermediate"],
  ["righthandindex3", "rightIndexDistal"],
  ["righthandmiddle1", "rightMiddleProximal"],
  ["righthandmiddle2", "rightMiddleIntermediate"],
  ["righthandmiddle3", "rightMiddleDistal"],
  ["righthandring1", "rightRingProximal"],
  ["righthandring2", "rightRingIntermediate"],
  ["righthandring3", "rightRingDistal"],
  ["righthandpinky1", "rightLittleProximal"],
  ["righthandpinky2", "rightLittleIntermediate"],
  ["righthandpinky3", "rightLittleDistal"],
]);

const RETARGET_CANONICAL_ALIAS = new Map([
  ["leftThumbIntermediate", ["leftThumbProximal", "leftThumbDistal"]],
  ["rightThumbIntermediate", ["rightThumbProximal", "rightThumbDistal"]],
  ["leftThumbProximal", ["leftThumbIntermediate", "leftThumbMetacarpal"]],
  ["rightThumbProximal", ["rightThumbIntermediate", "rightThumbMetacarpal"]],
  ["leftThumbMetacarpal", ["leftThumbProximal"]],
  ["rightThumbMetacarpal", ["rightThumbProximal"]],
  ["leftRingProximal", ["leftMiddleProximal", "leftIndexProximal", "leftLittleProximal"]],
  ["leftRingIntermediate", ["leftMiddleIntermediate", "leftIndexIntermediate", "leftLittleIntermediate"]],
  ["leftRingDistal", ["leftMiddleDistal", "leftIndexDistal", "leftLittleDistal"]],
  ["rightRingProximal", ["rightMiddleProximal", "rightIndexProximal", "rightLittleProximal"]],
  ["rightRingIntermediate", ["rightMiddleIntermediate", "rightIndexIntermediate", "rightLittleIntermediate"]],
  ["rightRingDistal", ["rightMiddleDistal", "rightIndexDistal", "rightLittleDistal"]],
  ["leftLittleProximal", ["leftRingProximal", "leftMiddleProximal"]],
  ["leftLittleIntermediate", ["leftRingIntermediate", "leftMiddleIntermediate"]],
  ["leftLittleDistal", ["leftRingDistal", "leftMiddleDistal"]],
  ["rightLittleProximal", ["rightRingProximal", "rightMiddleProximal"]],
  ["rightLittleIntermediate", ["rightRingIntermediate", "rightMiddleIntermediate"]],
  ["rightLittleDistal", ["rightRingDistal", "rightMiddleDistal"]],
  ["leftMiddleProximal", ["leftRingProximal", "leftIndexProximal"]],
  ["leftMiddleIntermediate", ["leftRingIntermediate", "leftIndexIntermediate"]],
  ["leftMiddleDistal", ["leftRingDistal", "leftIndexDistal"]],
  ["rightMiddleProximal", ["rightRingProximal", "rightIndexProximal"]],
  ["rightMiddleIntermediate", ["rightRingIntermediate", "rightIndexIntermediate"]],
  ["rightMiddleDistal", ["rightRingDistal", "rightIndexDistal"]],
]);

export function buildRetargetMap(targetBones, sourceBones, options = {}) {
  const canonicalFilter = options.canonicalFilter || null;
  const sourceByNorm = new Map();
  const sourceByCanonical = new Map();
  const sourceNameSet = new Set();
  for (const bone of sourceBones) {
    sourceByNorm.set(normalizeBoneName(bone.name), bone.name);
    const key = canonicalBoneKey(bone.name);
    if (key) {
      const existing = sourceByCanonical.get(key);
      if (!existing || canonicalBonePreferenceScore(bone) > canonicalBonePreferenceScore({ name: existing })) {
        sourceByCanonical.set(key, bone.name);
      }
    }
    sourceNameSet.add(bone.name);
  }

  const names = {};
  let matched = 0;
  const unmatchedSample = [];
  let canonicalCandidates = 0;
  const unmatchedHumanoid = [];
  for (const bone of targetBones) {
    const norm = normalizeBoneName(bone.name);
    const canonical = canonicalBoneKey(bone.name);
    if (canonicalFilter && (!canonical || !canonicalFilter.has(canonical))) {
      continue;
    }
    if (canonical) {
      canonicalCandidates += 1;
    }
    let sourceName = canonical ? sourceByCanonical.get(canonical) : undefined;
    if (!sourceName) {
      sourceName = sourceByNorm.get(norm);
    }
    if (!sourceName && canonical && RETARGET_CANONICAL_ALIAS.has(canonical)) {
      for (const altCanonical of RETARGET_CANONICAL_ALIAS.get(canonical)) {
        const alt = sourceByCanonical.get(altCanonical);
        if (alt) {
          sourceName = alt;
          break;
        }
      }
    }
    if (!sourceName && RETARGET_ALIAS.has(norm)) {
      sourceName = RETARGET_ALIAS.get(norm);
    }
    if (sourceName && sourceNameSet.has(sourceName)) {
      names[bone.name] = sourceName;
      matched += 1;
    } else if (unmatchedSample.length < 30) {
      unmatchedSample.push({ target: bone.name, canonical: canonical || "n/a" });
      if (canonical && unmatchedHumanoid.length < 30) {
        unmatchedHumanoid.push({ target: bone.name, canonical });
      }
    }
  }
  const sourceMatched = new Set(Object.values(names)).size;
  return { names, matched, unmatchedSample, canonicalCandidates, unmatchedHumanoid, sourceMatched };
}

export function buildStageSourceClip(sourceClip, sourceBones, stage, canonicalFilter) {
  if (!sourceClip) return null;
  if (!canonicalFilter) return sourceClip;
  const sourceCanonicalByName = new Map();
  for (const bone of sourceBones || []) {
    sourceCanonicalByName.set(bone.name, canonicalBoneKey(bone.name) || "");
  }
  const tracks = [];
  for (const track of sourceClip.tracks || []) {
    const parsed = parseTrackName(track.name);
    if (!parsed) continue;
    const canonical = sourceCanonicalByName.get(parsed.bone) || "";
    if (!canonicalFilter.has(canonical)) continue;
    if (parsed.property === "position" && canonical !== "hips") continue;
    tracks.push(track.clone());
  }
  if (!tracks.length) return null;
  return new THREE.AnimationClip(`${sourceClip.name || "retarget"}_${stage}`, sourceClip.duration, tracks);
}

export function scaleClipRotationsByCanonical(sourceClip, sourceBones, rotationScaleByCanonical) {
  if (!sourceClip) return null;
  if (!rotationScaleByCanonical || typeof rotationScaleByCanonical !== "object") return sourceClip;
  const sourceCanonicalByName = new Map();
  for (const bone of sourceBones || []) {
    sourceCanonicalByName.set(bone.name, canonicalBoneKey(bone.name) || "");
  }
  const identityQ = new THREE.Quaternion();
  const rawScale = Object.fromEntries(
    Object.entries(rotationScaleByCanonical).filter(([, value]) => Number.isFinite(value))
  );
  if (!Object.keys(rawScale).length) return sourceClip;

  const tracks = [];
  for (const track of sourceClip.tracks || []) {
    const parsed = parseTrackName(track.name);
    if (!parsed || parsed.property !== "quaternion") {
      tracks.push(track.clone());
      continue;
    }
    const canonical = sourceCanonicalByName.get(parsed.bone) || "";
    const scale = Number(rawScale[canonical]);
    if (!Number.isFinite(scale) || Math.abs(scale - 1) < 1e-6) {
      tracks.push(track.clone());
      continue;
    }
    const clampedScale = Math.max(0, scale);
    const cloned = track.clone();
    const values = cloned.values.slice();
    for (let i = 0; i + 3 < values.length; i += 4) {
      const q = new THREE.Quaternion(values[i], values[i + 1], values[i + 2], values[i + 3]).normalize();
      q.slerp(identityQ, 1 - clampedScale).normalize();
      values[i] = q.x;
      values[i + 1] = q.y;
      values[i + 2] = q.z;
      values[i + 3] = q.w;
    }
    cloned.values = values;
    tracks.push(cloned);
  }
  return new THREE.AnimationClip(`${sourceClip.name || "retarget"}_scaled`, sourceClip.duration, tracks);
}

export function attemptPriority(label) {
  if (label === "skeletonutils-skinnedmesh") return 40;
  if (label === "skeletonutils-skinnedmesh-reversed") return 30;
  if (label === "rename-fallback-bones") return 20;
  if (label === "rename-fallback-object") return 10;
  if (label === "skeletonutils-root") return 5;
  if (label === "skeletonutils-root-reversed") return 4;
  return 0;
}

export function buildRenamedClip(sourceClip, namesTargetToSource, sourceRootBoneName, outputBinding = "auto") {
  const sourceToTarget = new Map();
  for (const [targetName, sourceName] of Object.entries(namesTargetToSource)) {
    if (!sourceToTarget.has(sourceName)) {
      sourceToTarget.set(sourceName, targetName);
    }
  }

  const autoBonesSyntax = sourceClip.tracks.some((track) => track.name.startsWith(".bones["));
  const preferBonesSyntax =
    outputBinding === "bones" ? true : outputBinding === "object" ? false : autoBonesSyntax;
  const tracks = [];
  for (const track of sourceClip.tracks) {
    const parsed = parseTrackName(track.name);
    if (!parsed) continue;
    const targetBoneName = sourceToTarget.get(parsed.bone);
    if (!targetBoneName) continue;
    if (parsed.property === "position" && parsed.bone !== sourceRootBoneName) continue;

    const cloned = track.clone();
    cloned.name = preferBonesSyntax
      ? `.bones[${targetBoneName}].${parsed.property}`
      : `${targetBoneName}.${parsed.property}`;
    tracks.push(cloned);
  }

  if (!tracks.length) return null;
  return new THREE.AnimationClip(`${sourceClip.name || "retarget"}_renamed`, sourceClip.duration, tracks);
}

export function resolvedTrackCountForTarget(clip, targetBones) {
  const targetNames = new Set(targetBones.map((b) => b.name));
  let resolved = 0;
  for (const track of clip.tracks) {
    const parsed = parseTrackName(track.name);
    if (!parsed) continue;
    if (targetNames.has(parsed.bone)) {
      resolved += 1;
    }
  }
  return resolved;
}

export function resolvedTrackCountAcrossMeshes(clip, skinnedMeshes) {
  let best = 0;
  for (const mesh of skinnedMeshes || []) {
    const resolved = resolvedTrackCountForTarget(clip, mesh.skeleton.bones);
    if (resolved > best) {
      best = resolved;
    }
  }
  return best;
}

export function collectLimbDiagnostics(targetBones, sourceBones, namesTargetToSource, clip) {
  const sourceByName = new Map((sourceBones || []).map((b) => [b.name, b]));
  const sourceByCanonical = new Map();
  const targetByCanonical = new Map();

  for (const b of sourceBones || []) {
    const key = canonicalBoneKey(b.name);
    if (key && !sourceByCanonical.has(key)) sourceByCanonical.set(key, b);
  }
  for (const b of targetBones || []) {
    const key = canonicalBoneKey(b.name);
    if (key && !targetByCanonical.has(key)) targetByCanonical.set(key, b);
  }

  const trackByBone = new Map();
  for (const t of clip?.tracks || []) {
    const parsed = parseTrackName(t.name);
    if (!parsed) continue;
    const row = trackByBone.get(parsed.bone) || { hasQ: false, hasP: false };
    if (parsed.property === "quaternion") row.hasQ = true;
    if (parsed.property === "position") row.hasP = true;
    trackByBone.set(parsed.bone, row);
  }

  const expectedParent = new Map([
    ["leftLowerArm", "leftUpperArm"],
    ["leftHand", "leftLowerArm"],
    ["rightLowerArm", "rightUpperArm"],
    ["rightHand", "rightLowerArm"],
    ["leftLowerLeg", "leftUpperLeg"],
    ["leftFoot", "leftLowerLeg"],
    ["rightLowerLeg", "rightUpperLeg"],
    ["rightFoot", "rightLowerLeg"],
  ]);

  const keys = [
    "leftUpperArm",
    "leftLowerArm",
    "leftHand",
    "rightUpperArm",
    "rightLowerArm",
    "rightHand",
    "leftUpperLeg",
    "leftLowerLeg",
    "leftFoot",
    "rightUpperLeg",
    "rightLowerLeg",
    "rightFoot",
  ];

  const entries = [];
  for (const key of keys) {
    const targetBone = targetByCanonical.get(key) || null;
    const targetName = targetBone?.name || null;
    const sourceName = targetName ? namesTargetToSource[targetName] || null : null;
    const sourceBone = sourceName ? sourceByName.get(sourceName) || null : sourceByCanonical.get(key) || null;
    const tracks = targetName ? trackByBone.get(targetName) : null;
    const parentCanonical = targetBone?.parent ? canonicalBoneKey(targetBone.parent.name) : null;
    const expected = expectedParent.get(key) || null;
    const parentOk = expected ? parentCanonical === expected : true;

    entries.push({
      canonical: key,
      target: targetName || "missing",
      source: sourceBone?.name || sourceName || "missing",
      hasQ: !!tracks?.hasQ,
      hasP: !!tracks?.hasP,
      parent: parentCanonical || "n/a",
      expectedParent: expected || "n/a",
      parentOk,
    });
  }

  const issues = entries.filter((e) => e.target === "missing" || e.source === "missing" || !e.hasQ || !e.parentOk);
  return { total: entries.length, issuesCount: issues.length, issues, entries };
}

export function canonicalPoseSignature(boneMap, keys) {
  const hips = boneMap.get("hips");
  if (!hips) return null;
  const h = new THREE.Vector3();
  hips.getWorldPosition(h);
  const vectors = new Map();
  for (const key of keys) {
    const bone = boneMap.get(key);
    if (!bone) continue;
    const p = new THREE.Vector3();
    bone.getWorldPosition(p);
    const v = p.sub(h);
    if (v.lengthSq() < 1e-10) continue;
    vectors.set(key, v);
  }
  if (!vectors.size) return null;
  const headVec = vectors.get("head") || vectors.get("neck") || vectors.get("upperChest") || vectors.get("chest");
  const scale =
    headVec && headVec.length() > 1e-6
      ? headVec.length()
      : [...vectors.values()].reduce((s, v) => s + v.length(), 0) / vectors.size;
  return { vectors, scale: Math.max(scale, 1e-6) };
}
