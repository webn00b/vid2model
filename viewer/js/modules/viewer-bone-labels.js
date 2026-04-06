import * as THREE from "three";

export function createViewerBoneLabels({
  scene,
  canonicalBoneKey,
  buildCanonicalBoneMap,
  resolveSkeletonDumpCanonicals,
  getRetargetStage,
  getRetargetTargetBones,
  getSourceBones,
  getDebugNames,
  getModelRoot,
  objectHeight,
}) {
  const tmpWorldPosA = new THREE.Vector3();
  const tmpWorldPosB = new THREE.Vector3();
  let boneLabelDebug = {
    source: null,
    target: null,
  };

  function disposeBoneLabelSprite(sprite) {
    if (!sprite) return;
    if (sprite.material?.map) sprite.material.map.dispose();
    if (sprite.material) sprite.material.dispose();
  }

  function clearBoneLabels(kind = "both") {
    const keys = kind === "both" ? ["source", "target"] : [kind];
    for (const key of keys) {
      const entry = boneLabelDebug[key];
      if (!entry) continue;
      scene.remove(entry.group);
      for (const sprite of entry.sprites || []) {
        disposeBoneLabelSprite(sprite);
      }
      boneLabelDebug[key] = null;
    }
  }

  function createBoneLabelSprite(text, palette = {}) {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 196;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const bg = palette.bg || "rgba(15, 23, 42, 0.82)";
    const fg = palette.fg || "#f8fafc";
    const stroke = palette.stroke || "rgba(148, 163, 184, 0.9)";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = bg;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 4;
    const radius = 18;
    ctx.beginPath();
    ctx.moveTo(radius, 10);
    ctx.lineTo(canvas.width - radius, 10);
    ctx.quadraticCurveTo(canvas.width - 10, 10, canvas.width - 10, radius);
    ctx.lineTo(canvas.width - 10, canvas.height - radius - 10);
    ctx.quadraticCurveTo(canvas.width - 10, canvas.height - 10, canvas.width - radius, canvas.height - 10);
    ctx.lineTo(radius, canvas.height - 10);
    ctx.quadraticCurveTo(10, canvas.height - 10, 10, canvas.height - radius - 10);
    ctx.lineTo(10, radius);
    ctx.quadraticCurveTo(10, 10, radius, 10);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    const lines = String(text || "").split("\n").filter(Boolean).slice(0, 3);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = fg;
    if (lines.length >= 3) {
      ctx.font = "bold 34px sans-serif";
      ctx.fillText(lines[0], canvas.width / 2, 42);
      ctx.font = "24px monospace";
      ctx.fillText(lines[1], canvas.width / 2, 96);
      ctx.fillText(lines[2], canvas.width / 2, 146);
    } else if (lines.length === 2) {
      ctx.font = "bold 38px sans-serif";
      ctx.fillText(lines[0], canvas.width / 2, 58);
      ctx.font = "26px monospace";
      ctx.fillText(lines[1], canvas.width / 2, 118);
    } else {
      ctx.font = "bold 42px sans-serif";
      ctx.fillText(lines[0] || "", canvas.width / 2, canvas.height / 2);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      sizeAttenuation: true,
    });
    const sprite = new THREE.Sprite(material);
    sprite.renderOrder = 1000;
    return sprite;
  }

  function getNearestSourceCanonicalForBone(targetBone, scope = "legs") {
    if (!targetBone?.isBone) return "";
    const sourceBones = getSourceBones();
    if (!sourceBones.length) return "";
    const canonicals = resolveSkeletonDumpCanonicals(scope);
    if (!canonicals?.length) return "";
    const sourceMap = buildCanonicalBoneMap(sourceBones);
    targetBone.getWorldPosition(tmpWorldPosA);
    let bestCanonical = "";
    let bestDistanceSq = Infinity;
    for (const canonical of canonicals) {
      const sourceBone = sourceMap.get(canonical) || null;
      if (!sourceBone?.isBone) continue;
      sourceBone.getWorldPosition(tmpWorldPosB);
      const distanceSq = tmpWorldPosA.distanceToSquared(tmpWorldPosB);
      if (distanceSq < bestDistanceSq) {
        bestDistanceSq = distanceSq;
        bestCanonical = canonical;
      }
    }
    return bestCanonical;
  }

  function getBoneLabelText(kind, canonical, bone, scope = "legs") {
    const prefix = kind === "target" ? "MODEL" : "SRC";
    const label = String(canonical || bone?.name || "").trim();
    const rawName = String(bone?.name || "").trim();
    if (kind === "target") {
      const names = getDebugNames();
      const mappedSourceCanonical =
        bone?.name ? (canonicalBoneKey(names[bone.name] || "") || "") : "";
      const nearestSourceCanonical = getNearestSourceCanonicalForBone(bone, scope);
      if (!label) return rawName ? `${prefix}\n${rawName}` : prefix;
      const lines = [`${prefix} ${label}`];
      if (mappedSourceCanonical) {
        lines.push(`MAP SRC ${mappedSourceCanonical}`);
      }
      if (nearestSourceCanonical && nearestSourceCanonical !== mappedSourceCanonical) {
        lines.push(`POS SRC ${nearestSourceCanonical}`);
      }
      if (lines.length > 1) return lines.join("\n");
    }
    if (!label) return rawName ? `${prefix}\n${rawName}` : prefix;
    if (!rawName || rawName === label) return `${prefix}\n${label}`;
    return `${prefix} ${label}\n${rawName}`;
  }

  function getBoneLabelScale(kind) {
    const root = kind === "target" ? getModelRoot() : (getSourceBones()?.[0] || null);
    const height = objectHeight(root);
    const scale = Number.isFinite(height) && height > 0 ? height * 0.14 : 0.18;
    return Math.max(0.08, Math.min(0.38, scale));
  }

  function buildBoneLabelDebug(kind = "source", scope = "legs") {
    const canonicals = resolveSkeletonDumpCanonicals(scope);
    const bones =
      kind === "target"
        ? getRetargetTargetBones(getRetargetStage(), { preferNormalized: false })
        : getSourceBones();
    if (!bones.length || !canonicals.length) return null;
    const boneMap = buildCanonicalBoneMap(bones);
    const group = new THREE.Group();
    const sprites = [];
    const palette =
      kind === "target"
        ? { bg: "rgba(120, 53, 15, 0.84)", fg: "#fff7ed", stroke: "rgba(251, 146, 60, 0.95)" }
        : { bg: "rgba(30, 41, 59, 0.84)", fg: "#eff6ff", stroke: "rgba(96, 165, 250, 0.95)" };
    const scale = getBoneLabelScale(kind);
    for (const canonical of canonicals) {
      const bone = boneMap.get(canonical) || null;
      if (!bone?.isBone) continue;
      const sprite = createBoneLabelSprite(getBoneLabelText(kind, canonical, bone, scope), palette);
      if (!sprite) continue;
      sprite.scale.set(scale, scale * 0.34, 1);
      group.add(sprite);
      sprites.push(sprite);
    }
    if (!sprites.length) return null;
    scene.add(group);
    return { group, sprites, bones: canonicals.map((canonical) => boneMap.get(canonical) || null), scale };
  }

  function updateBoneLabels() {
    const yOffset = 0.06;
    const xOffset = 0.04;
    for (const [kind, entry] of Object.entries(boneLabelDebug)) {
      if (!entry?.sprites?.length) continue;
      for (let i = 0; i < entry.sprites.length; i += 1) {
        const sprite = entry.sprites[i];
        const bone = entry.bones[i];
        if (!sprite || !bone?.isBone) continue;
        bone.getWorldPosition(tmpWorldPosA);
        const canonical = canonicalBoneKey(bone.name) || "";
        const sideX = canonical.startsWith("left") ? -xOffset : canonical.startsWith("right") ? xOffset : 0;
        sprite.position.set(tmpWorldPosA.x + sideX, tmpWorldPosA.y + yOffset, tmpWorldPosA.z);
      }
    }
  }

  function refreshBoneLabels() {
    const cfg = window.__vid2modelBoneLabels || { enabled: false, which: "both", scope: "legs" };
    clearBoneLabels("both");
    if (!cfg.enabled) return;
    if (cfg.which === "source" || cfg.which === "both") {
      boneLabelDebug.source = buildBoneLabelDebug("source", cfg.scope);
    }
    if (cfg.which === "target" || cfg.which === "both") {
      boneLabelDebug.target = buildBoneLabelDebug("target", cfg.scope);
    }
    updateBoneLabels();
  }

  return {
    clearBoneLabels,
    refreshBoneLabels,
    updateBoneLabels,
  };
}
