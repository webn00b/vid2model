import { XR2ANIM_LEGACY_ALIAS } from "./retarget-constants.js";

const SIDE_PREFIX_RE = /^(left|right|l|r)(?=(shoulder|clavicle|collar|upperarm|forearm|lowerarm|arm|hand|wrist|upperleg|upleg|thigh|lowerleg|calf|leg|knee|foot|ankle|toe|thumb|index|middle|mid|ring|pinky|little))/;
const SIDE_SUFFIX_RE = /(shoulder|clavicle|collar|upperarm|forearm|lowerarm|arm|hand|wrist|upperleg|upleg|thigh|lowerleg|calf|leg|knee|foot|ankle|toe|thumb|index|middle|mid|ring|pinky|little)(left|right|l|r)$/;
const FINGER_SEGMENT_1_RE = /(thumb|index|middle|mid|ring|pinky|little|finger|親指|人指|中指|薬指|小指)1/;
const FINGER_SEGMENT_2_RE = /(thumb|index|middle|mid|ring|pinky|little|finger|親指|人指|中指|薬指|小指)2/;
const FINGER_SEGMENT_3_RE = /(thumb|index|middle|mid|ring|pinky|little|finger|親指|人指|中指|薬指|小指)3/;

const PREFIX_STRIP_RULES = [/^mixamorig/, /^armature/, /^valvebiped/, /^bip0*1?/, /^jbip[clr]?/];
const FIXED_CANONICAL_RULES = [
  ["hips", ["hips", "pelvis", "腰", "センター"]],
  ["upperChest", ["upperchest", "spine2", "spine3", "上半身2", "上半身3"]],
  ["chest", ["chest", "spine1", "上半身"]],
  ["spine", ["spine", "下半身"]],
  ["neck", ["neck", "首"]],
  ["head", ["head", "頭"]],
];
const EXCLUDED_CANONICAL_RULES = ["hair", "skirt", "hood", "string", "bust", "breast", "physics", "spring", "tail", "facial", "faceeye"];
const LIMB_CANONICAL_RULES = [
  ["Shoulder", ["shoulder", "clavicle", "collar", "肩"], ["upperarm", "arm", "腕"], ["forearm", "lowerarm", "elbow", "肘", "ひじ"], ["hand", "wrist", "手首", "手"], ["twist", "捩"]],
  ["UpperArm", ["upperarm", "arm", "腕"], ["forearm", "lowerarm", "elbow", "肘", "ひじ"], ["hand", "wrist", "手首", "手"], ["twist", "捩"]],
  ["LowerArm", ["forearm", "lowerarm", "elbow", "肘", "ひじ"]],
  ["Hand", ["hand", "wrist", "手首", "手"], ["thumb", "index", "middle", "mid", "ring", "pinky", "little", "親指", "人指", "中指", "薬指", "小指"], []],
  ["UpperLeg", ["upleg", "upperleg", "thigh", "太腿"]],
  ["LowerLeg", ["lowerleg", "calf", "knee", "膝", "ひざ"], ["leg", "足"], ["upleg", "upper", "foot", "ankle", "toe", "足首", "つま先"]],
  ["Foot", ["foot", "ankle", "足首"]],
  ["Toes", ["toe", "toebase", "つま先", "爪先"]],
];
export function normalizeBoneName(name) {
  return String(name || "")
    .normalize("NFKC")
    .replace(/^[^:]*:/, "")
    .replace(/[^a-zA-Z0-9\u3040-\u30ff\u3400-\u9fff]+/g, "")
    .toLowerCase();
}

function stripCanonicalPrefixes(norm) {
  let result = norm;
  for (const pattern of PREFIX_STRIP_RULES) {
    result = result.replace(pattern, "");
  }
  return result;
}

function makeTokenMatcher(norm, raw) {
  return (...tokens) => tokens.some((token) => norm.includes(token) || raw.includes(token));
}

function detectSide(norm, raw, has) {
  const sidePrefix = norm.match(SIDE_PREFIX_RE);
  const sideSuffix = norm.match(SIDE_SUFFIX_RE);
  if (
    has("left", "jbipl", "左", "hidari", "_l_", ".l", "-l") ||
    sidePrefix?.[1] === "left" ||
    sidePrefix?.[1] === "l" ||
    sideSuffix?.[2] === "left" ||
    sideSuffix?.[2] === "l" ||
    /^l[^a-z0-9]/.test(raw) ||
    /[^a-z0-9]l$/.test(raw)
  ) {
    return "left";
  }
  if (
    has("right", "jbipr", "右", "migi", "_r_", ".r", "-r") ||
    sidePrefix?.[1] === "right" ||
    sidePrefix?.[1] === "r" ||
    sideSuffix?.[2] === "right" ||
    sideSuffix?.[2] === "r" ||
    /^r[^a-z0-9]/.test(raw) ||
    /[^a-z0-9]r$/.test(raw)
  ) {
    return "right";
  }
  return "";
}

function detectFixedCanonical(has) {
  for (const [canonical, tokens] of FIXED_CANONICAL_RULES) {
    if (canonical !== "head" && has(...tokens)) return canonical;
  }
  if (has("head", "頭") && !has("headtop", "頭先")) return "head";
  return null;
}

function isExcludedBone(has) {
  return has(...EXCLUDED_CANONICAL_RULES);
}

function detectLimbCanonical(side, has) {
  if (!side) return null;
  for (const [suffix, includeTokens, excludeA = [], excludeB = [], excludeC = []] of LIMB_CANONICAL_RULES) {
    if (!has(...includeTokens)) continue;
    if (excludeA.length && has(...excludeA)) continue;
    if (excludeB.length && has(...excludeB)) continue;
    if (excludeC.length && has(...excludeC)) continue;
    return `${side}${suffix}`;
  }
  return null;
}

function detectFingerSegment(norm, has, fingerType) {
  if (has("metacarpal", "基節")) return "Metacarpal";
  if (has("proximal", "近位", "第一")) return "Proximal";
  if (has("intermediate", "中位", "第二")) return fingerType === "Thumb" ? "Proximal" : "Intermediate";
  if (has("distal", "tip", "遠位", "第三")) return "Distal";
  if (FINGER_SEGMENT_1_RE.test(norm)) {
    return fingerType === "Thumb" ? "Metacarpal" : "Proximal";
  }
  if (FINGER_SEGMENT_2_RE.test(norm)) {
    return fingerType === "Thumb" ? "Proximal" : "Intermediate";
  }
  if (FINGER_SEGMENT_3_RE.test(norm)) {
    return "Distal";
  }
  return null;
}

export function canonicalBoneKey(name) {
  const rawName = String(name || "");
  const raw = rawName.toLowerCase();
  const norm = stripCanonicalPrefixes(normalizeBoneName(rawName));
  if (XR2ANIM_LEGACY_ALIAS.has(norm)) {
    return XR2ANIM_LEGACY_ALIAS.get(norm);
  }
  const has = makeTokenMatcher(norm, raw);
  const side = detectSide(norm, raw, has);

  const fixedCanonical = detectFixedCanonical(has);
  if (fixedCanonical) return fixedCanonical;

  if (isExcludedBone(has)) {
    return null;
  }

  const limbCanonical = detectLimbCanonical(side, has);
  if (limbCanonical) return limbCanonical;

  const segmentForFinger = (fingerType) => {
    return detectFingerSegment(norm, has, fingerType);
  };

  const hasRingToken =
    raw.includes("薬指") ||
    /(?:^|[^a-z0-9])ring(?:[0-9]|[^a-z0-9]|$)/.test(raw) ||
    /(?:hand|finger)ring[0-9]?/.test(norm) ||
    /(?:^|[^a-z])(?:left|right|l|r)?ring(?:metacarpal|proximal|intermediate|distal|tip|[0-9]|$)/.test(norm);
  const isThumb = has("thumb", "親指");
  const isIndex = has("index", "人指");
  const isMiddle = has("middle", "mid", "中指");
  const isRing = hasRingToken && !has("string", "spring");
  const isLittle = has("pinky", "little", "小指");

  if (side && isThumb) {
    return `${side}Thumb${segmentForFinger("Thumb") || "Distal"}`;
  }
  if (side && isIndex) {
    return `${side}Index${segmentForFinger("Index") || "Distal"}`;
  }
  if (side && isMiddle) {
    return `${side}Middle${segmentForFinger("Middle") || "Distal"}`;
  }
  if (side && isRing) {
    return `${side}Ring${segmentForFinger("Ring") || "Distal"}`;
  }
  if (side && isLittle) {
    return `${side}Little${segmentForFinger("Little") || "Distal"}`;
  }

  return null;
}

export function parseTrackName(trackName) {
  let m = trackName.match(/^\.bones\[(.+)\]\.(position|quaternion)$/);
  if (m) {
    return { bone: m[1], property: m[2], bonesSyntax: true };
  }
  m = trackName.match(/^([^.[\]]+)\.(position|quaternion)$/);
  if (m) {
    return { bone: m[1], property: m[2], bonesSyntax: false };
  }
  return null;
}
