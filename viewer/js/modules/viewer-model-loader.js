export function applyVrmHumanoidBoneNames(vrm, vrmHumanoidBoneNames) {
  const humanoid = vrm?.humanoid || null;
  if (!humanoid) return { applied: 0, renamed: 0, bones: [], normalizedBones: [] };
  let applied = 0;
  let renamed = 0;
  const bones = [];
  const normalizedBones = [];
  for (const boneName of vrmHumanoidBoneNames || []) {
    const bone = humanoid.getRawBoneNode?.(boneName) || null;
    const normalizedBone = humanoid.getNormalizedBoneNode?.(boneName) || null;
    if (!bone?.isBone) continue;
    applied += 1;
    bones.push(bone);
    if (normalizedBone?.isBone) {
      normalizedBone.userData.__vrmHumanoidName = boneName;
      normalizedBones.push(normalizedBone);
    }
    bone.userData.__vrmHumanoidName = boneName;
    bone.userData.__originalBoneName = bone.userData.__originalBoneName || bone.name;
    if (bone.name !== boneName) {
      bone.name = boneName;
      renamed += 1;
    }
  }
  return { applied, renamed, bones, normalizedBones };
}

export function scoreSkinnedMesh(obj) {
  const bonesCount = obj.skeleton?.bones?.length || 0;
  const verts =
    obj.geometry && obj.geometry.attributes && obj.geometry.attributes.position
      ? obj.geometry.attributes.position.count
      : 0;
  const name = String(obj.name || "").toLowerCase();
  let nameBias = 0;
  if (/(body|torso|main|character)/.test(name)) {
    nameBias += 300000000;
  }
  if (/(hair|bang|fringe|ponytail|skirt|cloth|cape|acc|accessory|weapon)/.test(name)) {
    nameBias -= 200000000;
  }
  return bonesCount * 1000000 + verts + nameBias;
}

export function findSkinnedMeshes(root) {
  const found = [];
  root.traverse((obj) => {
    if (!(obj.isSkinnedMesh && obj.skeleton && obj.skeleton.bones?.length)) {
      return;
    }
    found.push(obj);
  });
  found.sort((a, b) => scoreSkinnedMesh(b) - scoreSkinnedMesh(a));
  return found;
}

export function logModelBones(skinnedMeshes, collectModelBoneRows, logRuntimeModelBones) {
  const rows = collectModelBoneRows(skinnedMeshes).map((row, index) => ({
    index,
    ...row,
  }));
  logRuntimeModelBones(rows);
}

export function createViewerModelLoader({
  gltfLoader,
  setStatus,
  onParsedModel,
  setModelFileNameText,
  defaultModelName = "6493143135142452442.glb",
  importMetaUrl,
}) {
  function loadModelBuffer(buffer, label) {
    setStatus(`Loading model: ${label} ...`);
    gltfLoader.parse(
      buffer,
      "",
      (gltf) => {
        onParsedModel(gltf, label);
      },
      (err) => {
        console.error(err);
        setStatus(`Failed to load model: ${label}`);
      }
    );
  }

  function loadModelFile(file) {
    file.arrayBuffer().then((buffer) => {
      loadModelBuffer(buffer, file.name);
    });
  }

  async function loadDefaultModel() {
    const url = new URL(`../models/${defaultModelName}`, importMetaUrl).href;
    setModelFileNameText?.(`${defaultModelName} (default)`);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buffer = await res.arrayBuffer();
      loadModelBuffer(buffer, defaultModelName);
    } catch (err) {
      console.error(err);
      setStatus(`Default model not found: viewer/models/${defaultModelName}`);
    }
  }

  return {
    loadModelBuffer,
    loadModelFile,
    loadDefaultModel,
  };
}
