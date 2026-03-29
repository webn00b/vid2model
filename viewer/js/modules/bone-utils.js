import { XR2ANIM_LEGACY_ALIAS } from "./retarget-constants.js";

export function normalizeBoneName(name) {
  return String(name || "")
    .normalize("NFKC")
    .replace(/^[^:]*:/, "")
    .replace(/[^a-zA-Z0-9\u3040-\u30ff\u3400-\u9fff]+/g, "")
    .toLowerCase();
}

export function canonicalBoneKey(name) {
  const rawName = String(name || "");
  const raw = rawName.toLowerCase();
  const norm = normalizeBoneName(rawName)
    .replace(/^mixamorig/, "")
    .replace(/^armature/, "")
    .replace(/^valvebiped/, "")
    .replace(/^bip0*1?/, "")
    .replace(/^jbip[clr]?/, "");
  if (XR2ANIM_LEGACY_ALIAS.has(norm)) {
    return XR2ANIM_LEGACY_ALIAS.get(norm);
  }
  const has = (...tokens) => tokens.some((t) => norm.includes(t) || raw.includes(t));
  const hasRingToken =
    raw.includes("薬指") ||
    /(?:^|[^a-z0-9])ring(?:[0-9]|[^a-z0-9]|$)/.test(raw) ||
    /(?:hand|finger)ring[0-9]?/.test(norm) ||
    /(?:^|[^a-z])(?:left|right|l|r)?ring(?:metacarpal|proximal|intermediate|distal|tip|[0-9]|$)/.test(norm);
  const sidePrefix = norm.match(
    /^(left|right|l|r)(?=(shoulder|clavicle|collar|upperarm|forearm|lowerarm|arm|hand|wrist|upperleg|upleg|thigh|lowerleg|calf|leg|knee|foot|ankle|toe|thumb|index|middle|mid|ring|pinky|little))/
  );
  const sideSuffix = norm.match(
    /(shoulder|clavicle|collar|upperarm|forearm|lowerarm|arm|hand|wrist|upperleg|upleg|thigh|lowerleg|calf|leg|knee|foot|ankle|toe|thumb|index|middle|mid|ring|pinky|little)(left|right|l|r)$/
  );

  let side = "";
  if (
    has("left", "jbipl", "左", "hidari", "_l_", ".l", "-l") ||
    sidePrefix?.[1] === "left" ||
    sidePrefix?.[1] === "l" ||
    sideSuffix?.[2] === "left" ||
    sideSuffix?.[2] === "l" ||
    /^l[^a-z0-9]/.test(raw) ||
    /[^a-z0-9]l$/.test(raw)
  ) {
    side = "left";
  } else if (
    has("right", "jbipr", "右", "migi", "_r_", ".r", "-r") ||
    sidePrefix?.[1] === "right" ||
    sidePrefix?.[1] === "r" ||
    sideSuffix?.[2] === "right" ||
    sideSuffix?.[2] === "r" ||
    /^r[^a-z0-9]/.test(raw) ||
    /[^a-z0-9]r$/.test(raw)
  ) {
    side = "right";
  }

  if (has("hips", "pelvis", "腰", "センター")) return "hips";
  if (has("upperchest", "spine2", "spine3", "上半身2", "上半身3")) return "upperChest";
  if (has("chest", "spine1", "上半身")) return "chest";
  if (has("spine", "下半身")) return "spine";
  if (has("neck", "首")) return "neck";
  if (has("head", "頭") && !has("headtop", "頭先")) return "head";

  if (
    has("hair", "skirt", "hood", "string", "bust", "breast", "physics", "spring", "tail", "facial", "faceeye")
  ) {
    return null;
  }

  if (side && has("shoulder", "clavicle", "collar", "肩")) {
    return `${side}Shoulder`;
  }
  if (
    side &&
    (
      has("upperarm", "arm", "腕") &&
      !has("forearm", "lowerarm", "elbow", "肘", "ひじ") &&
      !has("hand", "wrist", "手首", "手") &&
      !has("twist", "捩")
    )
  ) {
    return `${side}UpperArm`;
  }
  if (side && has("forearm", "lowerarm", "elbow", "肘", "ひじ")) {
    return `${side}LowerArm`;
  }
  if (
    side &&
    has("hand", "wrist", "手首", "手") &&
    !has("thumb", "index", "middle", "mid", "ring", "pinky", "little", "親指", "人指", "中指", "薬指", "小指")
  ) {
    return `${side}Hand`;
  }
  if (side && has("upleg", "upperleg", "thigh", "太腿")) {
    return `${side}UpperLeg`;
  }
  if (
    side &&
    (
      has("lowerleg", "calf", "knee", "膝", "ひざ") ||
      (has("leg", "足") && !has("upleg", "upper", "foot", "ankle", "toe", "足首", "つま先"))
    )
  ) {
    return `${side}LowerLeg`;
  }
  if (side && has("foot", "ankle", "足首")) {
    return `${side}Foot`;
  }
  if (side && has("toe", "toebase", "つま先", "爪先")) {
    return `${side}Toes`;
  }

  const segmentForFinger = (fingerType) => {
    if (has("metacarpal", "基節")) return "Metacarpal";
    if (has("proximal", "近位", "第一")) return "Proximal";
    if (has("intermediate", "中位", "第二")) return fingerType === "Thumb" ? "Proximal" : "Intermediate";
    if (has("distal", "tip", "遠位", "第三")) return "Distal";

    if (/(thumb|index|middle|mid|ring|pinky|little|finger|親指|人指|中指|薬指|小指)1/.test(norm)) {
      return fingerType === "Thumb" ? "Metacarpal" : "Proximal";
    }
    if (/(thumb|index|middle|mid|ring|pinky|little|finger|親指|人指|中指|薬指|小指)2/.test(norm)) {
      return fingerType === "Thumb" ? "Proximal" : "Intermediate";
    }
    if (/(thumb|index|middle|mid|ring|pinky|little|finger|親指|人指|中指|薬指|小指)3/.test(norm)) {
      return "Distal";
    }
    return null;
  };

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
