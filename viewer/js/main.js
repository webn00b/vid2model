    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";
    import { BVHLoader } from "three/addons/loaders/BVHLoader.js";
    import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
    import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";

    const wrap = document.getElementById("canvas-wrap");
    const statusEl = document.getElementById("status");
    const fileInput = document.getElementById("file-input");
    const modelInput = document.getElementById("model-input");
    const bvhFileNameEl = document.getElementById("bvh-file-name");
    const modelFileNameEl = document.getElementById("model-file-name");
    const btnRetarget = document.getElementById("retarget");
    const btnLoadDefault = document.getElementById("load-default");
    const btnZoomIn = document.getElementById("zoom-in");
    const btnZoomOut = document.getElementById("zoom-out");
    const btnPlay = document.getElementById("play");
    const btnPause = document.getElementById("pause");
    const btnStop = document.getElementById("stop");
    const timeline = document.getElementById("timeline");
    const timeEl = document.getElementById("time");
    const btnResetCamera = document.getElementById("reset-camera");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fbff);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    wrap.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 100, 0);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x7f8ea3, 1.0);
    scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(200, 300, 180);
    scene.add(dir);

    const grid = new THREE.GridHelper(1200, 30, 0x9aa9be, 0xc7d2e2);
    scene.add(grid);
    const axes = new THREE.AxesHelper(120);
    scene.add(axes);

    camera.position.set(260, 200, 260);
    controls.update();

    let mixer = null;
    let skeletonObj = null;
    let sourceResult = null;
    let currentClip = null;
    let currentAction = null;
    let modelRoot = null;
    let modelSkinnedMesh = null;
    let modelSkinnedMeshes = [];
    let modelMixer = null;
    let modelAction = null;
    let modelMixers = [];
    let modelActions = [];
    let liveRetarget = null;
    let sourceOverlay = null;
    let isPlaying = false;
    let isScrubbing = false;
    const loader = new BVHLoader();
    const gltfLoader = new GLTFLoader();
    const clock = new THREE.Clock();
    const DIAG_PREFIX = "[vid2model/diag]";
    const SKELETON_COLOR = 0xff2d55;
    const SOURCE_POINT_COLOR = 0xfff400;
    const _overlayV1 = new THREE.Vector3();
    const _overlayV2 = new THREE.Vector3();
    const _overlayV3 = new THREE.Vector3();
    const _overlayV4 = new THREE.Vector3();
    const _overlayV5 = new THREE.Vector3();
    const _liveYawQ = new THREE.Quaternion();
    const _liveAxisY = new THREE.Vector3(0, 1, 0);
    const _overlayPivot = new THREE.Vector3();
    const _overlayUpAxis = new THREE.Vector3(0, 1, 0);

    function diag(event, payload = {}) {
      console.log(DIAG_PREFIX, event, payload);
    }

    function shortErr(err, limit = 120) {
      const text = String(err?.message || err || "");
      return text.length > limit ? `${text.slice(0, limit)}...` : text;
    }

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function clearSourceOverlay() {
      if (!sourceOverlay) return;
      scene.remove(sourceOverlay.lines);
      scene.remove(sourceOverlay.points);
      sourceOverlay.lines.geometry.dispose();
      sourceOverlay.lines.material.dispose();
      sourceOverlay.points.geometry.dispose();
      sourceOverlay.points.material.dispose();
      sourceOverlay = null;
    }

    function findBoneByCanonical(bones, keys) {
      if (!bones?.length) return null;
      const keySet = new Set(keys);
      for (const bone of bones) {
        const key = canonicalBoneKey(bone.name);
        if (keySet.has(key)) return bone;
      }
      return null;
    }

    function estimateFacingVector(bones) {
      const left = findBoneByCanonical(bones, ["leftShoulder", "leftUpperArm"]);
      const right = findBoneByCanonical(bones, ["rightShoulder", "rightUpperArm"]);
      const hips = findBoneByCanonical(bones, ["hips"]);
      const head = findBoneByCanonical(bones, ["head", "neck", "upperChest", "chest"]);
      if (!left || !right || !hips || !head) return null;

      left.getWorldPosition(_overlayV1);
      right.getWorldPosition(_overlayV2);
      hips.getWorldPosition(_overlayV3);
      head.getWorldPosition(_overlayV4);
      const across = _overlayV2.sub(_overlayV1);
      const up = _overlayV4.sub(_overlayV3);
      if (across.lengthSq() < 1e-9 || up.lengthSq() < 1e-9) return null;

      const forward = _overlayV5.crossVectors(across.normalize(), up.normalize());
      if (forward.lengthSq() < 1e-9) return null;
      return forward.normalize().clone();
    }

    function estimateFacingYawOffset(sourceBones, targetBones) {
      const s = estimateFacingVector(sourceBones);
      const t = estimateFacingVector(targetBones);
      if (!s || !t) return 0;
      s.y = 0;
      t.y = 0;
      if (s.lengthSq() < 1e-9 || t.lengthSq() < 1e-9) return 0;
      s.normalize();
      t.normalize();
      const crossY = s.clone().cross(t).y;
      const dot = Math.max(-1, Math.min(1, s.dot(t)));
      const angle = Math.atan2(crossY, dot);
      return Number.isFinite(angle) ? angle : 0;
    }

    function updateSourceOverlay() {
      if (!sourceOverlay) return;
      const { bones, edges, pointAttr, lineAttr, overlayYaw, pivotBone } = sourceOverlay;
      const applyYaw = Number.isFinite(overlayYaw) && Math.abs(overlayYaw) > 1e-5;
      if (applyYaw && pivotBone) {
        pivotBone.getWorldPosition(_overlayPivot);
      }

      for (let i = 0; i < bones.length; i += 1) {
        bones[i].getWorldPosition(_overlayV1);
        if (applyYaw) {
          _overlayV1.sub(_overlayPivot).applyAxisAngle(_overlayUpAxis, overlayYaw).add(_overlayPivot);
        }
        pointAttr.array[i * 3 + 0] = _overlayV1.x;
        pointAttr.array[i * 3 + 1] = _overlayV1.y;
        pointAttr.array[i * 3 + 2] = _overlayV1.z;
      }
      pointAttr.needsUpdate = true;

      for (let i = 0; i < edges.length; i += 1) {
        const edge = edges[i];
        edge[0].getWorldPosition(_overlayV1);
        edge[1].getWorldPosition(_overlayV2);
        if (applyYaw) {
          _overlayV1.sub(_overlayPivot).applyAxisAngle(_overlayUpAxis, overlayYaw).add(_overlayPivot);
          _overlayV2.sub(_overlayPivot).applyAxisAngle(_overlayUpAxis, overlayYaw).add(_overlayPivot);
        }
        const base = i * 6;
        lineAttr.array[base + 0] = _overlayV1.x;
        lineAttr.array[base + 1] = _overlayV1.y;
        lineAttr.array[base + 2] = _overlayV1.z;
        lineAttr.array[base + 3] = _overlayV2.x;
        lineAttr.array[base + 4] = _overlayV2.y;
        lineAttr.array[base + 5] = _overlayV2.z;
      }
      lineAttr.needsUpdate = true;
    }

    function createSourceOverlay(skeleton) {
      clearSourceOverlay();
      const bones = skeleton?.bones || [];
      if (!bones.length) return;

      const boneSet = new Set(bones);
      const edges = [];
      for (const bone of bones) {
        if (bone.parent && boneSet.has(bone.parent)) {
          edges.push([bone.parent, bone]);
        }
      }

      const pointGeometry = new THREE.BufferGeometry();
      const pointAttr = new THREE.BufferAttribute(new Float32Array(bones.length * 3), 3);
      pointGeometry.setAttribute("position", pointAttr);

      const lineGeometry = new THREE.BufferGeometry();
      const lineAttr = new THREE.BufferAttribute(new Float32Array(Math.max(1, edges.length * 2) * 3), 3);
      lineGeometry.setAttribute("position", lineAttr);

      const lineMaterial = new THREE.LineBasicMaterial({
        color: SKELETON_COLOR,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
        depthWrite: false,
      });
      const pointsMaterial = new THREE.PointsMaterial({
        color: SOURCE_POINT_COLOR,
        size: 8,
        sizeAttenuation: false,
        transparent: true,
        opacity: 1,
        depthTest: false,
        depthWrite: false,
      });

      const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
      lines.renderOrder = 998;
      lines.frustumCulled = false;
      const points = new THREE.Points(pointGeometry, pointsMaterial);
      points.renderOrder = 999;
      points.frustumCulled = false;

      scene.add(lines);
      scene.add(points);
      sourceOverlay = {
        bones,
        edges,
        lines,
        points,
        pointAttr,
        lineAttr,
        overlayYaw: 0,
        pivotBone: findBoneByCanonical(bones, ["hips"]) || bones[0] || null,
      };
      updateSourceOverlay();
    }

    function fitToSkeleton(root) {
      const box = new THREE.Box3().setFromObject(root);
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);
      const maxDim = Math.max(size.x, size.y, size.z, 1);
      const dist = maxDim * 1.8;
      camera.position.set(center.x + dist, center.y + dist * 0.8, center.z + dist);
      controls.target.copy(center);
      controls.update();
    }

    function objectHeight(root) {
      if (!root) return 0;
      const box = new THREE.Box3().setFromObject(root);
      const h = box.max.y - box.min.y;
      return Number.isFinite(h) && h > 0 ? h : 0;
    }

    function findHipsBone(bones) {
      if (!bones?.length) return null;
      return bones.find((b) => canonicalBoneKey(b.name) === "hips") || bones[0] || null;
    }

    function findFootLevelY(bones) {
      if (!bones?.length) return null;
      const footKeys = new Set(["leftFoot", "rightFoot", "leftToes", "rightToes"]);
      const ys = [];
      const p = new THREE.Vector3();
      for (const bone of bones) {
        const key = canonicalBoneKey(bone.name);
        if (!footKeys.has(key)) continue;
        bone.getWorldPosition(p);
        ys.push(p.y);
      }
      if (!ys.length) return null;
      return ys.reduce((a, b) => a + b, 0) / ys.length;
    }

    function syncSourceDisplayToModel() {
      if (!skeletonObj || !sourceResult?.skeleton?.bones?.length || !modelRoot || !modelSkinnedMesh?.skeleton?.bones?.length) {
        return;
      }
      skeletonObj.position.set(0, 0, 0);
      skeletonObj.scale.setScalar(1);
      skeletonObj.updateMatrixWorld(true);
      modelRoot.updateMatrixWorld(true);

      const sourceH = objectHeight(skeletonObj);
      const targetH = objectHeight(modelRoot);
      if (sourceH > 1e-6 && targetH > 1e-6) {
        const displayScale = Math.max(1e-6, Math.min(1e6, targetH / sourceH));
        skeletonObj.scale.setScalar(displayScale);
      }
      skeletonObj.updateMatrixWorld(true);

      const sourceHips = findHipsBone(sourceResult.skeleton.bones);
      const targetHips = findHipsBone(modelSkinnedMesh.skeleton.bones);
      if (sourceHips && targetHips) {
        const src = new THREE.Vector3();
        const dst = new THREE.Vector3();
        sourceHips.getWorldPosition(src);
        targetHips.getWorldPosition(dst);
        skeletonObj.position.add(dst.sub(src));
      }
      skeletonObj.updateMatrixWorld(true);

      // Second pass: align by feet level to compensate different hip pivot conventions.
      const sourceFeetY = findFootLevelY(sourceResult.skeleton.bones);
      const targetFeetY = findFootLevelY(modelSkinnedMesh.skeleton.bones);
      if (Number.isFinite(sourceFeetY) && Number.isFinite(targetFeetY)) {
        skeletonObj.position.y += targetFeetY - sourceFeetY;
        skeletonObj.updateMatrixWorld(true);
      }
      let overlayYaw = 0;
      if (sourceOverlay) {
        overlayYaw = estimateFacingYawOffset(sourceResult.skeleton.bones, modelSkinnedMesh.skeleton.bones);
        sourceOverlay.overlayYaw = Number.isFinite(overlayYaw) ? overlayYaw : 0;
      }
      diag("display-sync", {
        sourceHeight: Number(sourceH.toFixed(4)),
        targetHeight: Number(targetH.toFixed(4)),
        displayScale: Number(skeletonObj.scale.y.toFixed(6)),
        overlayYawDeg: Number((overlayYaw * 180 / Math.PI).toFixed(2)),
      });
      updateSourceOverlay();
    }

    function getActiveDuration() {
      const sourceDuration = currentClip ? currentClip.duration : 0;
      const modelDuration = modelAction ? modelAction.getClip().duration : 0;
      return Math.max(sourceDuration, modelDuration, 0);
    }

    function updateTimelineUi(time = 0) {
      const duration = getActiveDuration();
      timeline.min = "0";
      timeline.max = String(duration > 0 ? duration : 1);
      const clampedTime = Math.max(0, Math.min(duration > 0 ? duration : 0, time));
      timeline.value = String(clampedTime);
      timeEl.textContent = `${clampedTime.toFixed(2)} / ${(duration > 0 ? duration : 0).toFixed(2)}`;
    }

    function clearSource() {
      if (skeletonObj) {
        scene.remove(skeletonObj);
      }
      skeletonObj = null;
      clearSourceOverlay();
      sourceResult = null;
      mixer = null;
      currentClip = null;
      currentAction = null;
      liveRetarget = null;
      updateTimelineUi(0);
    }

    function clearModel() {
      if (modelRoot) {
        scene.remove(modelRoot);
      }
      modelRoot = null;
      modelSkinnedMesh = null;
      modelSkinnedMeshes = [];
      modelMixer = null;
      modelAction = null;
      modelMixers = [];
      modelActions = [];
      liveRetarget = null;
      updateTimelineUi(0);
    }

    function zoomBy(factor) {
      const toCam = camera.position.clone().sub(controls.target);
      const dist = toCam.length();
      const next = Math.max(20, Math.min(4000, dist * factor));
      if (dist > 1e-6) {
        toCam.setLength(next);
        camera.position.copy(controls.target.clone().add(toCam));
        controls.update();
        setStatus(`Zoom: ${Math.round(next)}`);
      }
    }

    function normalizeBoneName(name) {
      return String(name || "")
        .normalize("NFKC")
        .replace(/^[^:]*:/, "")
        .replace(/[^a-zA-Z0-9\u3040-\u30ff\u3400-\u9fff]+/g, "")
        .toLowerCase();
    }

    function canonicalBoneKey(name) {
      const rawName = String(name || "");
      const raw = rawName.toLowerCase();
      const norm = normalizeBoneName(rawName)
        .replace(/^mixamorig/, "")
        .replace(/^armature/, "")
        .replace(/^valvebiped/, "")
        .replace(/^bip0*1?/, "")
        .replace(/^jbip[clr]?/, "");
      const has = (...tokens) => tokens.some((t) => norm.includes(t) || raw.includes(t));
      const hasRingToken =
        raw.includes("薬指") ||
        /(?:^|[^a-z0-9])ring(?:[0-9]|[^a-z0-9]|$)/.test(raw) ||
        /(?:hand|finger)ring[0-9]?/.test(norm);

      let side = "";
      if (
        has("left", "jbipl", "左", "hidari", "_l_", ".l", "-l") ||
        /^l[^a-z0-9]/.test(raw) ||
        /[^a-z0-9]l$/.test(raw)
      ) {
        side = "left";
      } else if (
        has("right", "jbipr", "右", "migi", "_r_", ".r", "-r") ||
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

      // Skip obvious accessory/secondary chains that often produce false positives.
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

    function buildRetargetMap(targetBones, sourceBones) {
      const sourceByNorm = new Map();
      const sourceByCanonical = new Map();
      const sourceNameSet = new Set();
      for (const bone of sourceBones) {
        sourceByNorm.set(normalizeBoneName(bone.name), bone.name);
        const key = canonicalBoneKey(bone.name);
        if (key) {
          sourceByCanonical.set(key, bone.name);
        }
        sourceNameSet.add(bone.name);
      }

      const alias = new Map([
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
      const canonicalAlias = new Map([
        ["leftThumbIntermediate", ["leftThumbProximal", "leftThumbDistal"]],
        ["rightThumbIntermediate", ["rightThumbProximal", "rightThumbDistal"]],
        ["leftThumbProximal", ["leftThumbIntermediate", "leftThumbMetacarpal"]],
        ["rightThumbProximal", ["rightThumbIntermediate", "rightThumbMetacarpal"]],
        ["leftThumbMetacarpal", ["leftThumbProximal"]],
        ["rightThumbMetacarpal", ["rightThumbProximal"]],
      ]);

      const names = {};
      let matched = 0;
      const unmatchedSample = [];
      let canonicalCandidates = 0;
      const unmatchedHumanoid = [];
      for (const bone of targetBones) {
        const norm = normalizeBoneName(bone.name);
        const canonical = canonicalBoneKey(bone.name);
        if (canonical) {
          canonicalCandidates += 1;
        }
        let sourceName = canonical ? sourceByCanonical.get(canonical) : undefined;
        if (!sourceName) {
          sourceName = sourceByNorm.get(norm);
        }
        if (!sourceName && canonical && canonicalAlias.has(canonical)) {
          for (const altCanonical of canonicalAlias.get(canonical)) {
            const alt = sourceByCanonical.get(altCanonical);
            if (alt) {
              sourceName = alt;
              break;
            }
          }
        }
        if (!sourceName && alias.has(norm)) {
          sourceName = alias.get(norm);
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

    function attemptPriority(label) {
      if (label === "skeletonutils-skinnedmesh") return 40;
      if (label === "skeletonutils-skinnedmesh-reversed") return 30;
      if (label === "rename-fallback-bones") return 20;
      if (label === "rename-fallback-object") return 10;
      if (label === "skeletonutils-root") return 5;
      if (label === "skeletonutils-root-reversed") return 4;
      return 0;
    }

    function parseTrackName(trackName) {
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

    function buildRenamedClip(sourceClip, namesTargetToSource, sourceRootBoneName, outputBinding = "auto") {
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
        if (!parsed) {
          continue;
        }
        const targetBoneName = sourceToTarget.get(parsed.bone);
        if (!targetBoneName) {
          continue;
        }
        if (parsed.property === "position" && parsed.bone !== sourceRootBoneName) {
          continue;
        }

        const cloned = track.clone();
        cloned.name = preferBonesSyntax
          ? `.bones[${targetBoneName}].${parsed.property}`
          : `${targetBoneName}.${parsed.property}`;
        tracks.push(cloned);
      }

      if (!tracks.length) {
        return null;
      }
      return new THREE.AnimationClip(`${sourceClip.name || "retarget"}_renamed`, sourceClip.duration, tracks);
    }

    function clipUsesBonesSyntax(clip) {
      return clip.tracks.some((t) => t.name.startsWith(".bones["));
    }

    function resolvedTrackCountForTarget(clip, targetBones) {
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

    function scoreSkinnedMesh(obj) {
      const bonesCount = obj.skeleton?.bones?.length || 0;
      const verts =
        obj.geometry && obj.geometry.attributes && obj.geometry.attributes.position
          ? obj.geometry.attributes.position.count
          : 0;
      const n = String(obj.name || "").toLowerCase();
      let nameBias = 0;
      if (/(body|torso|main|character)/.test(n)) {
        nameBias += 300000000;
      }
      if (/(hair|bang|fringe|ponytail|skirt|cloth|cape|acc|accessory|weapon)/.test(n)) {
        nameBias -= 200000000;
      }
      return bonesCount * 1000000 + verts + nameBias;
    }

    function findSkinnedMeshes(root) {
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

    function resolvedTrackCountAcrossMeshes(clip, skinnedMeshes) {
      let best = 0;
      for (const mesh of skinnedMeshes || []) {
        const resolved = resolvedTrackCountForTarget(clip, mesh.skeleton.bones);
        if (resolved > best) {
          best = resolved;
        }
      }
      return best;
    }

    function buildBindingsForAttempt(attempt, clip) {
      const mixers = [];
      const actions = [];
      const activeBindings = [];
      const useSkinnedBinding =
        attempt.bindingRoot === "skinned" ||
        (attempt.bindingRoot === "auto" && clipUsesBonesSyntax(clip));

      if (useSkinnedBinding) {
        const seenSkeletonIds = new Set();
        for (const mesh of modelSkinnedMeshes) {
          const skeletonId = mesh.skeleton?.uuid || mesh.uuid;
          if (seenSkeletonIds.has(skeletonId)) {
            continue;
          }
          seenSkeletonIds.add(skeletonId);
          const resolvedTracks = resolvedTrackCountForTarget(clip, mesh.skeleton.bones);
          if (resolvedTracks <= 0) {
            continue;
          }
          const mix = new THREE.AnimationMixer(mesh);
          const action = mix.clipAction(clip);
          action.reset();
          action.setEffectiveWeight(1);
          action.setEffectiveTimeScale(1);
          action.play();
          mixers.push(mix);
          actions.push(action);
          activeBindings.push({
            mesh: mesh.name || "(unnamed-skinned-mesh)",
            bones: mesh.skeleton.bones.length,
            resolvedTracks,
          });
        }
      } else {
        const bindingRoot = modelRoot || modelSkinnedMesh;
        if (bindingRoot) {
          const mix = new THREE.AnimationMixer(bindingRoot);
          const action = mix.clipAction(clip);
          action.reset();
          action.setEffectiveWeight(1);
          action.setEffectiveTimeScale(1);
          action.play();
          mixers.push(mix);
          actions.push(action);
          activeBindings.push({
            mesh: bindingRoot.name || "(model-root)",
            bones: modelSkinnedMesh?.skeleton?.bones?.length || 0,
            resolvedTracks: modelSkinnedMesh?.skeleton?.bones
              ? resolvedTrackCountForTarget(clip, modelSkinnedMesh.skeleton.bones)
              : clip.tracks.length,
          });
        }
      }

      return { mixers, actions, activeBindings, useSkinnedBinding };
    }

    function probeMotionForBindings(bindings, clip) {
      if (!bindings.mixers.length || !modelSkinnedMesh?.skeleton?.bones?.length) {
        return { sampleTime: 0, maxAngle: 0, maxPos: 0, score: 0 };
      }

      const trackedBoneNames = new Set();
      for (const track of clip.tracks) {
        const parsed = parseTrackName(track.name);
        if (parsed?.bone) {
          trackedBoneNames.add(parsed.bone);
        }
      }

      let probeBones = modelSkinnedMesh.skeleton.bones.filter((b) => trackedBoneNames.has(b.name));
      if (!probeBones.length) {
        probeBones = modelSkinnedMesh.skeleton.bones;
      }
      probeBones = probeBones.slice(0, 64);

      const sampleTime =
        clip.duration > 0
          ? Math.max(1 / 30, Math.min(clip.duration * 0.35, Math.max(clip.duration - 1e-3, 1 / 30)))
          : 0;
      if (sampleTime <= 0) {
        return { sampleTime: 0, maxAngle: 0, maxPos: 0, score: 0 };
      }

      for (const mix of bindings.mixers) {
        mix.setTime(0);
      }
      if (modelRoot) {
        modelRoot.updateMatrixWorld(true);
      }

      const before = probeBones.map((b) => ({
        bone: b,
        q: b.quaternion.clone(),
        p: b.position.clone(),
      }));

      for (const mix of bindings.mixers) {
        mix.setTime(sampleTime);
      }
      if (modelRoot) {
        modelRoot.updateMatrixWorld(true);
      }

      let maxAngle = 0;
      let maxPos = 0;
      for (const s of before) {
        maxAngle = Math.max(maxAngle, s.q.angleTo(s.bone.quaternion));
        maxPos = Math.max(maxPos, s.p.distanceTo(s.bone.position));
      }

      for (const mix of bindings.mixers) {
        mix.setTime(0);
      }
      if (modelRoot) {
        modelRoot.updateMatrixWorld(true);
      }

      const score = maxAngle * 1000 + maxPos;
      return { sampleTime, maxAngle, maxPos, score };
    }

    function buildCanonicalBoneMap(bones) {
      const map = new Map();
      for (const bone of bones || []) {
        const key = canonicalBoneKey(bone.name);
        if (!key || map.has(key)) continue;
        map.set(key, bone);
      }
      return map;
    }

    function canonicalPoseSignature(boneMap, keys) {
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

    function computePoseMatchError(bindings, sampleTime) {
      if (!bindings?.mixers?.length || !modelSkinnedMesh?.skeleton?.bones?.length || !sourceResult?.skeleton?.bones?.length) {
        return Number.POSITIVE_INFINITY;
      }
      if (!(sampleTime > 0)) {
        return Number.POSITIVE_INFINITY;
      }

      const keys = [
        "head",
        "neck",
        "upperChest",
        "chest",
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
      ];
      const sourceTime = mixer ? mixer.time : 0;
      const modelTimes = bindings.mixers.map((mix) => mix.time);
      try {
        if (mixer) mixer.setTime(sampleTime);
        for (const mix of bindings.mixers) {
          mix.setTime(sampleTime);
        }
        sourceResult.skeleton.bones?.[0]?.updateMatrixWorld(true);
        modelRoot?.updateMatrixWorld(true);

        const sourceMap = buildCanonicalBoneMap(sourceResult.skeleton.bones);
        const targetMap = buildCanonicalBoneMap(modelSkinnedMesh.skeleton.bones);
        const sourceSig = canonicalPoseSignature(sourceMap, keys);
        const targetSig = canonicalPoseSignature(targetMap, keys);
        if (!sourceSig || !targetSig) return Number.POSITIVE_INFINITY;

        let angleSum = 0;
        let lenSum = 0;
        let count = 0;
        for (const key of keys) {
          const sv = sourceSig.vectors.get(key);
          const tv = targetSig.vectors.get(key);
          if (!sv || !tv) continue;
          const sn = sv.clone().normalize();
          const tn = tv.clone().normalize();
          angleSum += sn.angleTo(tn);
          lenSum += Math.abs(tv.length() / targetSig.scale - sv.length() / sourceSig.scale);
          count += 1;
        }
        if (count < 4) return Number.POSITIVE_INFINITY;
        const avgAngle = angleSum / count;
        const avgLen = lenSum / count;
        return avgAngle + avgLen * 0.35;
      } finally {
        if (mixer) mixer.setTime(sourceTime);
        for (let i = 0; i < bindings.mixers.length; i += 1) {
          bindings.mixers[i].setTime(modelTimes[i] || 0);
        }
        sourceResult.skeleton.bones?.[0]?.updateMatrixWorld(true);
        modelRoot?.updateMatrixWorld(true);
      }
    }

    function buildLiveRetargetPlan(skinnedMeshes, sourceBones, namesTargetToSource) {
      const sourceByName = new Map(sourceBones.map((b) => [b.name, b]));
      const uniqueSkeletons = [];
      const seenSkeletonIds = new Set();
      for (const mesh of skinnedMeshes || []) {
        const skeleton = mesh.skeleton;
        if (!skeleton?.bones?.length) continue;
        const skeletonId = skeleton.uuid || mesh.uuid;
        if (seenSkeletonIds.has(skeletonId)) continue;
        seenSkeletonIds.add(skeletonId);
        uniqueSkeletons.push(skeleton);
      }

      const pairs = [];
      for (const skeleton of uniqueSkeletons) {
        for (const targetBone of skeleton.bones) {
          const sourceName = namesTargetToSource[targetBone.name];
          if (!sourceName) continue;
          const sourceBone = sourceByName.get(sourceName);
          if (!sourceBone) continue;
          pairs.push({
            target: targetBone,
            source: sourceBone,
            targetRestQ: new THREE.Quaternion(),
            sourceRestQ: new THREE.Quaternion(),
            targetRestPos: new THREE.Vector3(),
            sourceRestPos: new THREE.Vector3(),
            isHips:
              canonicalBoneKey(targetBone.name) === "hips" ||
              canonicalBoneKey(sourceBone.name) === "hips",
          });
        }
      }

      if (!pairs.length) {
        return null;
      }

      const sourceTime = mixer ? mixer.time : 0;
      if (mixer) {
        mixer.setTime(0);
      }
      for (const skeleton of uniqueSkeletons) {
        skeleton.pose();
      }
      if (modelRoot) {
        modelRoot.updateMatrixWorld(true);
      }

      for (const pair of pairs) {
        pair.targetRestQ.copy(pair.target.quaternion);
        pair.sourceRestQ.copy(pair.source.quaternion);
        pair.targetRestPos.copy(pair.target.position);
        pair.sourceRestPos.copy(pair.source.position);
      }

      let posScale = 1;
      try {
        const sourceBox = new THREE.Box3().setFromPoints(sourceBones.map((b) => b.getWorldPosition(new THREE.Vector3())));
        const targetBones = [];
        for (const skeleton of uniqueSkeletons) {
          for (const b of skeleton.bones) targetBones.push(b);
        }
        const targetBox = new THREE.Box3().setFromPoints(targetBones.map((b) => b.getWorldPosition(new THREE.Vector3())));
        const sourceHWorld = Math.max(1e-6, sourceBox.max.y - sourceBox.min.y);
        const targetH = Math.max(1e-6, targetBox.max.y - targetBox.min.y);
        const sourceRootScale = new THREE.Vector3(1, 1, 1);
        if (sourceBones[0]) {
          sourceBones[0].getWorldScale(sourceRootScale);
        }
        const sourceScaleY = Math.max(1e-6, Math.abs(sourceRootScale.y));
        const sourceHUnscaled = sourceHWorld / sourceScaleY;
        posScale = targetH / sourceHUnscaled;
      } catch (err) {
        posScale = 1;
      }

      if (mixer) {
        mixer.setTime(sourceTime);
      }
      if (modelRoot) {
        modelRoot.updateMatrixWorld(true);
      }

      const targetRefBones = uniqueSkeletons[0]?.bones || [];
      const rawYawOffset = estimateFacingYawOffset(sourceBones, targetRefBones);
      const absRawYaw = Math.abs(rawYawOffset);
      let yawOffset = rawYawOffset;
      // Keep yaw correction conservative: either no correction or a clean 180deg flip.
      if (absRawYaw < THREE.MathUtils.degToRad(45)) {
        yawOffset = 0;
      } else if (absRawYaw > THREE.MathUtils.degToRad(120)) {
        yawOffset = Math.sign(rawYawOffset || 1) * Math.PI;
      }

      return {
        pairs,
        uniqueSkeletons,
        posScale,
        yawOffset,
      };
    }

    const _liveQ = new THREE.Quaternion();
    const _liveV = new THREE.Vector3();
    function applyLiveRetargetPose(plan) {
      if (!plan?.pairs?.length) {
        return;
      }
      const yaw = Number.isFinite(plan.yawOffset) ? plan.yawOffset : 0;
      if (Math.abs(yaw) > 1e-5) {
        _liveYawQ.setFromAxisAngle(_liveAxisY, yaw);
      }
      for (const pair of plan.pairs) {
        _liveQ.copy(pair.sourceRestQ).invert().multiply(pair.source.quaternion);
        if (pair.isHips && Math.abs(yaw) > 1e-5) {
          // Apply a constant facing offset at hips so the whole body aligns in world space.
          pair.target.quaternion.copy(pair.targetRestQ).multiply(_liveYawQ).multiply(_liveQ).normalize();
        } else {
          pair.target.quaternion.copy(pair.targetRestQ).multiply(_liveQ).normalize();
        }
        if (pair.isHips) {
          _liveV.copy(pair.source.position).sub(pair.sourceRestPos).multiplyScalar(plan.posScale);
          if (Math.abs(yaw) > 1e-5) {
            _liveV.applyQuaternion(_liveYawQ);
          }
          pair.target.position.copy(pair.targetRestPos).add(_liveV);
        } else {
          pair.target.position.copy(pair.targetRestPos);
        }
      }
      if (modelRoot) {
        modelRoot.updateMatrixWorld(true);
      }
    }

    function inferHumanoidCanonicalNamesFromTopology(bones) {
      const boneSet = new Set(bones);
      const childrenMap = new Map();
      for (const bone of bones) {
        childrenMap.set(
          bone,
          bone.children.filter((c) => boneSet.has(c))
        );
      }

      const worldPos = new Map();
      for (const bone of bones) {
        const p = new THREE.Vector3();
        bone.getWorldPosition(p);
        worldPos.set(bone, p);
      }

      const descendantCount = new Map();
      const countDesc = (bone) => {
        let total = 0;
        for (const c of childrenMap.get(bone) || []) {
          total += 1 + countDesc(c);
        }
        descendantCount.set(bone, total);
        return total;
      };
      const roots = bones.filter((b) => !boneSet.has(b.parent));
      for (const root of roots) {
        countDesc(root);
      }

      const sortedRoots = (roots.length ? roots : [bones[0]]).slice().sort(
        (a, b) => (descendantCount.get(b) || 0) - (descendantCount.get(a) || 0)
      );
      const hips = sortedRoots[0];
      if (!hips) {
        return new Map();
      }

      const assignment = new Map();
      assignment.set(hips, "hips");

      const getHighestYChild = (bone, exclude = new Set()) => {
        const children = (childrenMap.get(bone) || []).filter((c) => !exclude.has(c));
        if (!children.length) return null;
        return children.slice().sort((a, b) => worldPos.get(b).y - worldPos.get(a).y)[0];
      };
      const getLowestYChild = (bone, exclude = new Set()) => {
        const children = (childrenMap.get(bone) || []).filter((c) => !exclude.has(c));
        if (!children.length) return null;
        return children.slice().sort((a, b) => worldPos.get(a).y - worldPos.get(b).y)[0];
      };
      const xOf = (bone) => worldPos.get(bone)?.x ?? 0;

      // Build vertical torso chain from hips upwards.
      const torsoChain = [];
      let cursor = getHighestYChild(hips);
      while (cursor && torsoChain.length < 5) {
        torsoChain.push(cursor);
        const next = getHighestYChild(cursor);
        if (!next) break;
        if (worldPos.get(next).y < worldPos.get(cursor).y - 1e-4) break;
        cursor = next;
      }

      const spine = torsoChain[0] || null;
      const chest = torsoChain[1] || null;
      let upperChest = null;
      let neck = null;
      let head = null;
      if (torsoChain.length >= 5) {
        upperChest = torsoChain[2];
        neck = torsoChain[3];
        head = torsoChain[4];
      } else if (torsoChain.length === 4) {
        neck = torsoChain[2];
        head = torsoChain[3];
      } else if (torsoChain.length === 3) {
        neck = torsoChain[2];
      }

      if (spine) assignment.set(spine, "spine");
      if (chest) assignment.set(chest, "chest");
      if (upperChest) assignment.set(upperChest, "upperChest");
      if (neck) assignment.set(neck, "neck");
      if (head) assignment.set(head, "head");

      const torsoSet = new Set([hips, ...torsoChain].filter(Boolean));
      const hipsChildren = childrenMap.get(hips) || [];
      const legRoots = hipsChildren
        .filter((c) => !torsoSet.has(c))
        .slice()
        .sort((a, b) => worldPos.get(a).y - worldPos.get(b).y)
        .slice(0, 2);

      const assignLeg = (rootBone, side) => {
        if (!rootBone) return;
        assignment.set(rootBone, `${side}UpperLeg`);
        const lower = getLowestYChild(rootBone);
        if (lower) assignment.set(lower, `${side}LowerLeg`);
        const foot = lower ? getLowestYChild(lower) : null;
        if (foot) assignment.set(foot, `${side}Foot`);
        const toes = foot ? getLowestYChild(foot) : null;
        if (toes) assignment.set(toes, `${side}Toes`);
      };

      if (legRoots.length === 2) {
        const [a, b] = legRoots;
        const leftRoot = xOf(a) <= xOf(b) ? a : b;
        const rightRoot = leftRoot === a ? b : a;
        assignLeg(leftRoot, "left");
        assignLeg(rightRoot, "right");
      } else if (legRoots.length === 1) {
        const side = xOf(legRoots[0]) < xOf(hips) ? "left" : "right";
        assignLeg(legRoots[0], side);
      }

      const shoulderBase = upperChest || chest || spine || hips;
      const shoulderBaseChildren = shoulderBase ? (childrenMap.get(shoulderBase) || []) : [];
      const upFromShoulderBase = shoulderBase ? getHighestYChild(shoulderBase) : null;
      let armRoots = shoulderBaseChildren.filter((c) => c !== upFromShoulderBase);

      if (armRoots.length < 2 && chest && shoulderBase !== chest) {
        const chestChildren = childrenMap.get(chest) || [];
        const chestUp = getHighestYChild(chest);
        const extra = chestChildren.filter((c) => c !== chestUp);
        armRoots = [...new Set([...armRoots, ...extra])];
      }

      armRoots = armRoots
        .filter((c) => !assignment.has(c))
        .slice()
        .sort((a, b) => Math.abs(xOf(b) - xOf(hips)) - Math.abs(xOf(a) - xOf(hips)))
        .slice(0, 2);

      const assignArm = (rootBone, side) => {
        if (!rootBone) return;
        const c1 = (childrenMap.get(rootBone) || []).slice().sort(
          (a, b) => Math.abs(xOf(b) - xOf(hips)) - Math.abs(xOf(a) - xOf(hips))
        )[0] || null;
        const c2 = c1
          ? (childrenMap.get(c1) || []).slice().sort(
              (a, b) => Math.abs(xOf(b) - xOf(hips)) - Math.abs(xOf(a) - xOf(hips))
            )[0] || null
          : null;
        const c3 = c2
          ? (childrenMap.get(c2) || []).slice().sort(
              (a, b) => Math.abs(xOf(b) - xOf(hips)) - Math.abs(xOf(a) - xOf(hips))
            )[0] || null
          : null;

        if (c1 && c2 && c3) {
          assignment.set(rootBone, `${side}Shoulder`);
          assignment.set(c1, `${side}UpperArm`);
          assignment.set(c2, `${side}LowerArm`);
          assignment.set(c3, `${side}Hand`);
        } else if (c1 && c2) {
          assignment.set(rootBone, `${side}UpperArm`);
          assignment.set(c1, `${side}LowerArm`);
          assignment.set(c2, `${side}Hand`);
        } else if (c1) {
          assignment.set(rootBone, `${side}UpperArm`);
          assignment.set(c1, `${side}LowerArm`);
        }
      };

      if (armRoots.length === 2) {
        const [a, b] = armRoots;
        const leftRoot = xOf(a) <= xOf(b) ? a : b;
        const rightRoot = leftRoot === a ? b : a;
        assignArm(leftRoot, "left");
        assignArm(rightRoot, "right");
      } else if (armRoots.length === 1) {
        const side = xOf(armRoots[0]) < xOf(hips) ? "left" : "right";
        assignArm(armRoots[0], side);
      }

      return assignment;
    }

    function autoNameTargetBones(skinnedMesh) {
      const bones = skinnedMesh?.skeleton?.bones || [];
      if (!bones.length) {
        return { missingBefore: 0, autoNamed: 0, inferredCanonical: 0 };
      }

      let missingBefore = 0;
      for (const bone of bones) {
        if (!bone.name || !bone.name.trim()) {
          missingBefore += 1;
        }
      }

      let autoNamed = 0;
      const usedNames = new Set();
      for (const bone of bones) {
        const n = (bone.name || "").trim();
        if (n) usedNames.add(n);
      }

      for (let i = 0; i < bones.length; i += 1) {
        const bone = bones[i];
        if (!bone.name || !bone.name.trim()) {
          let generated = `autoBone_${String(i).padStart(3, "0")}`;
          while (usedNames.has(generated)) {
            generated = `${generated}_x`;
          }
          bone.name = generated;
          usedNames.add(generated);
          autoNamed += 1;
        }
      }

      let inferredCanonical = 0;
      if (missingBefore > 0) {
        skinnedMesh.updateMatrixWorld(true);
        const inferred = inferHumanoidCanonicalNamesFromTopology(bones);
        for (const [bone, canonical] of inferred.entries()) {
          if (!canonical) continue;
          if (usedNames.has(canonical) && bone.name !== canonical) continue;
          if (bone.name !== canonical) {
            usedNames.delete(bone.name);
            bone.name = canonical;
            usedNames.add(canonical);
            inferredCanonical += 1;
          }
        }
      }

      return { missingBefore, autoNamed, inferredCanonical };
    }

    function applyBvhToModel() {
      if (!sourceResult) {
        setStatus("Load BVH first.");
        return;
      }
      if (!modelSkinnedMesh) {
        setStatus("Load model (.glb/.vrm) first.");
        return;
      }

      try {
        const { names, matched, unmatchedSample, canonicalCandidates, unmatchedHumanoid, sourceMatched } = buildRetargetMap(
          modelSkinnedMesh.skeleton.bones,
          sourceResult.skeleton.bones
        );
        const mappedPairs = Object.keys(names).length;
        diag("retarget-input", {
          sourceBones: sourceResult.skeleton.bones.length,
          targetBones: modelSkinnedMesh.skeleton.bones.length,
          sourceTracks: sourceResult.clip.tracks.length,
          mappedPairs,
          uniqueSourceMapped: sourceMatched,
          humanoidMatched: canonicalCandidates > 0 ? `${matched}/${canonicalCandidates}` : "n/a",
        });

        const retargetAttempts = [];
        const attemptDebug = [];
        const pushAttempt = (label, bindingRoot, fn) => {
          try {
            const candidate = fn();
            if (candidate && candidate.tracks && candidate.tracks.length > 0) {
              const resolvedTracks =
                bindingRoot === "skinned"
                  ? resolvedTrackCountAcrossMeshes(candidate, modelSkinnedMeshes)
                  : resolvedTrackCountForTarget(candidate, modelSkinnedMesh.skeleton.bones);
              retargetAttempts.push({ label, clip: candidate, bindingRoot, resolvedTracks });
              attemptDebug.push({
                label,
                bindingRoot,
                tracks: candidate.tracks.length,
                resolvedTracks,
                ok: true,
                error: "",
              });
            } else {
              attemptDebug.push({ label, bindingRoot, tracks: 0, resolvedTracks: 0, ok: false, error: "empty clip" });
            }
          } catch (err) {
            attemptDebug.push({
              label,
              bindingRoot,
              tracks: 0,
              resolvedTracks: 0,
              ok: false,
              error: shortErr(err),
            });
          }
        };

        pushAttempt("skeletonutils-skinnedmesh", "skinned", () =>
          SkeletonUtils.retargetClip(modelSkinnedMesh, sourceResult.skeleton, sourceResult.clip, {
            names,
            hip: "hips",
            useFirstFramePosition: true,
            preserveBonePositions: false,
          })
        );
        const namesSourceToTarget = Object.fromEntries(Object.entries(names).map(([target, source]) => [source, target]));
        pushAttempt("skeletonutils-skinnedmesh-reversed", "skinned", () =>
          SkeletonUtils.retargetClip(modelSkinnedMesh, sourceResult.skeleton, sourceResult.clip, {
            names: namesSourceToTarget,
            hip: "hips",
            useFirstFramePosition: true,
            preserveBonePositions: false,
          })
        );
        if (modelRoot) {
          pushAttempt("skeletonutils-root", "root", () =>
            SkeletonUtils.retargetClip(modelRoot, sourceResult.skeleton, sourceResult.clip, {
              names,
              hip: "hips",
              useFirstFramePosition: true,
              preserveBonePositions: false,
            })
          );
          pushAttempt("skeletonutils-root-reversed", "root", () =>
            SkeletonUtils.retargetClip(modelRoot, sourceResult.skeleton, sourceResult.clip, {
              names: namesSourceToTarget,
              hip: "hips",
              useFirstFramePosition: true,
              preserveBonePositions: false,
            })
          );
        }

        const sourceRootBoneName = sourceResult.skeleton.bones?.[0]?.name || "hips";
        pushAttempt("rename-fallback-bones", "skinned", () =>
          buildRenamedClip(sourceResult.clip, names, sourceRootBoneName, "bones")
        );
        pushAttempt("rename-fallback-object", "root", () =>
          buildRenamedClip(sourceResult.clip, names, sourceRootBoneName, "object")
        );

        modelMixers = [];
        modelActions = [];
        modelMixer = null;
        modelAction = null;
        liveRetarget = null;

        if (!retargetAttempts.length) {
          setStatus("Retarget failed: 0 tracks produced. Bone names do not match.");
          const unmatched = (unmatchedHumanoid.length ? unmatchedHumanoid : unmatchedSample).slice(0, 8);
          diag("retarget-fail", { reason: "no_tracks", unmatched });
          return;
        }

        const rankedAttempts = [...retargetAttempts].sort((a, b) => {
          if (b.resolvedTracks !== a.resolvedTracks) return b.resolvedTracks - a.resolvedTracks;
          return b.clip.tracks.length - a.clip.tracks.length;
        });
        const primaryAttempt = rankedAttempts[0];
        const skeletonSkinned = rankedAttempts.find((a) => a.label === "skeletonutils-skinnedmesh");
        const skeletonSkinnedReversed = rankedAttempts.find((a) => a.label === "skeletonutils-skinnedmesh-reversed");
        const fallbackObject = rankedAttempts.find((a) => a.label === "rename-fallback-object");
        const fallbackBones = rankedAttempts.find((a) => a.label === "rename-fallback-bones");
        const candidateAttempts = [
          primaryAttempt,
          skeletonSkinned,
          skeletonSkinnedReversed,
          fallbackObject,
          fallbackBones,
        ].filter(
          (a, i, arr) => a && arr.findIndex((b) => b && b.label === a.label) === i
        );

        const selectionDebug = [];
        let selectedAttempt = null;
        let selectedBindings = null;
        let selectedProbe = null;
        let selectedPoseError = Number.POSITIVE_INFINITY;
        for (const attempt of candidateAttempts) {
          const bindings = buildBindingsForAttempt(attempt, attempt.clip);
          if (!bindings.mixers.length) {
            selectionDebug.push({
              label: attempt.label,
              bindingRoot: attempt.bindingRoot,
              tracks: attempt.clip.tracks.length,
              resolvedTracks: attempt.resolvedTracks,
              mixers: 0,
              probeAngle: 0,
              probePos: 0,
              probeScore: 0,
              sampleTime: 0,
              poseError: Number.POSITIVE_INFINITY,
              ok: false,
            });
            continue;
          }
          const probe = probeMotionForBindings(bindings, attempt.clip);
          const poseError = computePoseMatchError(bindings, probe.sampleTime);
          selectionDebug.push({
            label: attempt.label,
            bindingRoot: attempt.bindingRoot,
            tracks: attempt.clip.tracks.length,
            resolvedTracks: attempt.resolvedTracks,
            mixers: bindings.mixers.length,
            probeAngle: Number(probe.maxAngle.toFixed(6)),
            probePos: Number(probe.maxPos.toFixed(6)),
            probeScore: Number(probe.score.toFixed(6)),
            sampleTime: Number(probe.sampleTime.toFixed(3)),
            poseError: Number.isFinite(poseError) ? Number(poseError.toFixed(6)) : Number.POSITIVE_INFINITY,
            ok: true,
          });

          const better = (() => {
            if (!selectedAttempt) return true;
            const resolvedDiff = attempt.resolvedTracks - selectedAttempt.resolvedTracks;
            if (resolvedDiff > 2) return true;
            if (resolvedDiff < -2) return false;
            const poseDiff = selectedPoseError - poseError; // lower poseError is better
            if (Number.isFinite(poseDiff) && Math.abs(poseDiff) > 1e-4) {
              return poseDiff > 0;
            }
            const priorityDiff = attemptPriority(attempt.label) - attemptPriority(selectedAttempt.label);
            if (priorityDiff > 0) return true;
            if (priorityDiff < 0) return false;
            return probe.score > selectedProbe.score + 1e-7;
          })();
          if (better) {
            selectedAttempt = attempt;
            selectedBindings = bindings;
            selectedProbe = probe;
            selectedPoseError = poseError;
          }
        }

        if (!selectedBindings || !selectedBindings.mixers.length) {
          setStatus("Retarget failed: clip has no resolved tracks on model skeleton.");
          diag("retarget-fail", { reason: "no_resolved_tracks" });
          return;
        }

        modelMixers = selectedBindings.mixers;
        modelActions = selectedBindings.actions;
        modelMixer = modelMixers[0];
        modelAction = modelActions[0];
        const clip = selectedAttempt.clip;
        let selectedModeLabel = selectedAttempt.label;

        const isRenameFallback = selectedAttempt.label.startsWith("rename-fallback");
        const rawFacingYaw = estimateFacingYawOffset(
          sourceResult.skeleton.bones,
          modelSkinnedMesh.skeleton.bones
        );
        const strongFacingMismatch = Math.abs(rawFacingYaw) > THREE.MathUtils.degToRad(100);
        const weakMotion = !!selectedProbe && selectedProbe.score < 0.5;
        const highPoseError = Number.isFinite(selectedPoseError) && selectedPoseError > 0.6;
        const useLiveDelta = isRenameFallback || strongFacingMismatch || weakMotion || highPoseError;
        if (useLiveDelta) {
          const livePlan = buildLiveRetargetPlan(
            modelSkinnedMeshes,
            sourceResult.skeleton.bones,
            names
          );
          if (livePlan && livePlan.pairs.length > 0) {
            liveRetarget = livePlan;
            applyLiveRetargetPose(liveRetarget);
            modelMixers = [];
            modelActions = [];
            modelMixer = null;
            modelAction = null;
            selectedModeLabel = `${selectedAttempt.label}+live-delta`;
          }
        }

        isPlaying = true;
        updateTimelineUi(0);
        const sourceTotal = sourceResult.skeleton.bones.length;
        const total = modelSkinnedMesh.skeleton.bones.length;
        const targetCoverage = canonicalCandidates > 0 ? matched / canonicalCandidates : 0;
        const sourceCoverage = sourceTotal > 0 ? sourceMatched / sourceTotal : 0;
        let lowMatch = "";
        if (targetCoverage < 0.75 || (targetCoverage < 0.9 && sourceCoverage < 0.35)) {
          lowMatch = " low humanoid match, try another model/rig.";
        }
        const candidateInfo = canonicalCandidates > 0 ? `, humanoid targets ${matched}/${canonicalCandidates}` : "";
        const activeMeshCount = liveRetarget ? liveRetarget.uniqueSkeletons.length : modelMixers.length;
        setStatus(
          `Model retargeted (source ${sourceMatched}/${sourceTotal}${candidateInfo}, all ${matched}/${total}, tracks ${clip.tracks.length}, mode ${selectedModeLabel}, active meshes ${activeMeshCount}).${lowMatch}`
        );
        const unmatched = unmatchedHumanoid.slice(0, 6);
        window.__vid2modelDebug = {
          names,
          sourceBones: sourceResult.skeleton.bones.map((b) => b.name),
          targetBones: modelSkinnedMesh.skeleton.bones.map((b) => b.name),
          clipTracks: clip.tracks.map((t) => t.name),
          attemptDebug,
          bestMode: selectedModeLabel,
          selectionDebug,
          motionProbe: selectedProbe,
          poseError: selectedPoseError,
          liveRetarget: liveRetarget
            ? {
                pairs: liveRetarget.pairs.length,
                skeletons: liveRetarget.uniqueSkeletons.length,
                posScale: liveRetarget.posScale,
                yawOffsetDeg: Number((liveRetarget.yawOffset * 180 / Math.PI).toFixed(2)),
              }
            : null,
          activeMeshCount,
        };
        diag("retarget-result", {
          mode: selectedModeLabel,
          tracks: clip.tracks.length,
          resolvedTracks: selectedAttempt.resolvedTracks,
          activeMeshes: activeMeshCount,
          motionProbe: selectedProbe
            ? {
                sampleTime: Number(selectedProbe.sampleTime.toFixed(3)),
                maxAngle: Number(selectedProbe.maxAngle.toFixed(6)),
                maxPos: Number(selectedProbe.maxPos.toFixed(6)),
                score: Number(selectedProbe.score.toFixed(6)),
              }
            : null,
          poseError: Number.isFinite(selectedPoseError) ? Number(selectedPoseError.toFixed(6)) : null,
          liveRetarget: liveRetarget
            ? {
                pairs: liveRetarget.pairs.length,
                skeletons: liveRetarget.uniqueSkeletons.length,
                posScale: Number(liveRetarget.posScale.toFixed(6)),
                yawOffsetDeg: Number((liveRetarget.yawOffset * 180 / Math.PI).toFixed(2)),
              }
            : null,
          unmatched,
        });
        diag("retarget-summary", {
          mode: selectedModeLabel,
          mappedPairs,
          sourceMatched,
          sourceTotal,
          humanoidMatched: canonicalCandidates > 0 ? `${matched}/${canonicalCandidates}` : "n/a",
          tracks: clip.tracks.length,
          resolvedTracks: selectedAttempt.resolvedTracks,
          poseError: Number.isFinite(selectedPoseError) ? Number(selectedPoseError.toFixed(4)) : null,
          yawOffsetDeg: liveRetarget ? Number((liveRetarget.yawOffset * 180 / Math.PI).toFixed(2)) : 0,
          liveDelta: !!liveRetarget,
        });
      } catch (err) {
        console.error(err);
        setStatus("Retarget failed. Check model rig names.");
      }
    }

    function loadBvhText(text, label) {
      try {
        const result = loader.parse(text);
        clearSource();

        skeletonObj = new THREE.Group();
        const helper = new THREE.SkeletonHelper(result.skeleton.bones[0]);
        helper.skeleton = result.skeleton;
        helper.material.linewidth = 2;
        helper.material.color.set(SKELETON_COLOR);
        helper.material.depthTest = false;
        helper.material.depthWrite = false;
        helper.material.transparent = true;
        helper.material.opacity = 0.95;
        helper.renderOrder = 999;
        helper.frustumCulled = false;
        skeletonObj.add(result.skeleton.bones[0]);
        skeletonObj.add(helper);
        scene.add(skeletonObj);
        createSourceOverlay(result.skeleton);

        mixer = new THREE.AnimationMixer(result.skeleton.bones[0]);
        sourceResult = result;
        currentClip = result.clip;
        currentAction = mixer.clipAction(currentClip);
        currentAction.play();
        isPlaying = true;
        updateTimelineUi(0);

        if (modelRoot && modelSkinnedMesh) {
          syncSourceDisplayToModel();
          fitToSkeleton(modelRoot);
        } else {
          fitToSkeleton(skeletonObj);
        }
        setStatus(`Loaded: ${label} (${Math.round(currentClip.duration * 100) / 100}s)`);
        if (modelSkinnedMesh) {
          applyBvhToModel();
        }
      } catch (err) {
        console.error(err);
        setStatus(`Failed to load: ${label}`);
      }
    }

    function applyParsedModel(gltf, label) {
      clearModel();
      modelRoot = gltf.scene || gltf.scenes?.[0] || null;
      if (!modelRoot) {
        setStatus(`Failed to parse model: ${label}`);
        return;
      }
      scene.add(modelRoot);
      modelSkinnedMeshes = findSkinnedMeshes(modelRoot);
      modelSkinnedMesh = modelSkinnedMeshes[0] || null;
      if (!modelSkinnedMesh) {
        setStatus(`Model loaded, but no skinned mesh found: ${label}`);
        return;
      }
      let totalMissingBefore = 0;
      let totalAutoNamed = 0;
      let totalInferredCanonical = 0;
      for (const mesh of modelSkinnedMeshes) {
        const namingInfo = autoNameTargetBones(mesh);
        totalMissingBefore += namingInfo.missingBefore;
        totalAutoNamed += namingInfo.autoNamed;
        totalInferredCanonical += namingInfo.inferredCanonical;
      }
      diag("model-loaded", {
        file: label,
        skinnedMeshes: modelSkinnedMeshes.length,
        topMeshes: modelSkinnedMeshes.slice(0, 3).map((m) => ({
          name: m.name || "(unnamed-skinned-mesh)",
          bones: m.skeleton.bones.length,
        })),
        autoNaming:
          totalMissingBefore > 0
            ? {
                missingBefore: totalMissingBefore,
                autoNamed: totalAutoNamed,
                inferredCanonical: totalInferredCanonical,
              }
            : null,
      });
      if (sourceResult) {
        syncSourceDisplayToModel();
      }
      fitToSkeleton(modelRoot);
      setStatus(`Model loaded: ${label} (skinned meshes: ${modelSkinnedMeshes.length})`);
      if (sourceResult) {
        applyBvhToModel();
      }
    }

    function loadModelBuffer(buffer, label) {
      setStatus(`Loading model: ${label} ...`);
      gltfLoader.parse(
        buffer,
        "",
        (gltf) => {
          applyParsedModel(gltf, label);
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
      const defaultModelName = "en_0.vrm";
      const url = new URL("../models/en_0.vrm", import.meta.url).href;
      modelFileNameEl.textContent = `${defaultModelName} (default)`;
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

    async function loadDefault() {
      setStatus("Loading output/think.bvh ...");
      try {
        const res = await fetch("../output/think.bvh");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        loadBvhText(text, "output/think.bvh");
      } catch (err) {
        console.error(err);
        setStatus("Cannot load output/think.bvh. Start local server from vid2model.");
      }
    }

    fileInput.addEventListener("change", async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      bvhFileNameEl.textContent = file.name;
      setStatus(`Loading ${file.name} ...`);
      const text = await file.text();
      loadBvhText(text, file.name);
    });

    modelInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      modelFileNameEl.textContent = file.name;
      loadModelFile(file);
    });

    btnLoadDefault.addEventListener("click", loadDefault);
    btnRetarget.addEventListener("click", applyBvhToModel);
    btnZoomIn.addEventListener("click", () => zoomBy(0.85));
    btnZoomOut.addEventListener("click", () => zoomBy(1.2));
    btnPlay.addEventListener("click", () => {
      if (!mixer && !modelMixers.length) return;
      isPlaying = true;
      if (currentAction) currentAction.paused = false;
      for (const action of modelActions) {
        action.paused = false;
      }
      setStatus("Playback: play");
    });
    btnPause.addEventListener("click", () => {
      if (!mixer && !modelMixers.length) return;
      isPlaying = false;
      if (currentAction) currentAction.paused = true;
      for (const action of modelActions) {
        action.paused = true;
      }
      setStatus("Playback: pause");
    });
    btnStop.addEventListener("click", () => {
      if (!mixer && !modelMixers.length) return;
      isPlaying = false;
      if (currentAction) currentAction.paused = true;
      for (const action of modelActions) {
        action.paused = true;
      }
      if (mixer) mixer.setTime(0);
      for (const mix of modelMixers) {
        mix.setTime(0);
      }
      if (liveRetarget) {
        applyLiveRetargetPose(liveRetarget);
      }
      updateTimelineUi(0);
      setStatus("Playback: stop");
    });
    timeline.addEventListener("input", () => {
      const duration = getActiveDuration();
      if (!duration || (!mixer && !modelMixers.length)) return;
      isScrubbing = true;
      const t = Math.max(0, Math.min(duration, Number(timeline.value)));
      if (mixer) mixer.setTime(t);
      for (const mix of modelMixers) {
        mix.setTime(t);
      }
      if (liveRetarget) {
        applyLiveRetargetPose(liveRetarget);
      }
      timeEl.textContent = `${t.toFixed(2)} / ${duration.toFixed(2)}`;
      setStatus(`Scrub: ${t.toFixed(2)}s`);
    });
    timeline.addEventListener("change", () => {
      isScrubbing = false;
    });
    btnResetCamera.addEventListener("click", () => {
      camera.position.set(260, 200, 260);
      controls.target.set(0, 100, 0);
      controls.update();
      setStatus("Camera reset");
    });

    function onResize() {
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener("resize", onResize);
    onResize();
    updateTimelineUi(0);
    loadDefaultModel();

    function animate() {
      requestAnimationFrame(animate);
      const dt = clock.getDelta();
        if (isPlaying && !isScrubbing) {
          if (mixer) mixer.update(dt);
          for (const mix of modelMixers) {
            mix.update(dt);
          }
        if (liveRetarget) {
          applyLiveRetargetPose(liveRetarget);
        }
        }

        if (sourceOverlay) {
          updateSourceOverlay();
        }

        const duration = Math.max(getActiveDuration(), 1e-6);
      const refMixer = mixer || modelMixers[0] || modelMixer;
      if (refMixer) {
        const t = refMixer.time % duration;
        if (!isScrubbing) {
          timeline.value = String(t);
        }
        timeEl.textContent = `${t.toFixed(2)} / ${duration.toFixed(2)}`;
      } else if (!isScrubbing) {
        updateTimelineUi(0);
      }

      controls.update();
      renderer.render(scene, camera);
    }
    animate();
