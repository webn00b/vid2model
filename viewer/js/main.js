    import * as THREE from "three";
    import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";
    import { BVHLoader } from "three/addons/loaders/BVHLoader.js";
    import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
    import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";
    import {
      ARM_REFINEMENT_CANONICAL,
      RETARGET_BODY_CORE_CANONICAL,
      RETARGET_BODY_CANONICAL,
      RETARGET_STAGES,
    } from "./modules/retarget-constants.js";
    import { createDiag, DIAG_FILE_LOG_DEFAULT_URL } from "./modules/diag.js";
    import { canonicalBoneKey, normalizeBoneName, parseTrackName } from "./modules/bone-utils.js";
    import {
      applyBoneLengthCalibration,
      applyFingerLengthCalibration,
      buildArmRefinementCalibration,
      buildBodyLengthCalibration,
      buildFingerLengthCalibration,
      resetBoneLengthCalibration,
      restoreBonePositionSnapshot,
      snapshotCanonicalBonePositions,
    } from "./modules/retarget-calibration.js";
    import {
      buildRootYawCandidates,
      collectAlignmentDiagnostics,
      computeHipsYawError,
      dumpRetargetAlignmentDiagnostics,
      getLiveDeltaOverride,
    } from "./modules/retarget-analysis.js";
    import {
      buildBindingsForAttempt,
      computePoseMatchError,
      evaluateRootYawCandidates,
      probeMotionForBindings,
    } from "./modules/retarget-eval.js";
    import {
      applyLiveRetargetPose as applyLiveRetargetPoseModule,
      applyModelRootYaw as applyModelRootYawModule,
      buildLiveRetargetPlan as buildLiveRetargetPlanModule,
      clearSourceOverlay as clearSourceOverlayModule,
      createSourceOverlay as createSourceOverlayModule,
      estimateFacingYawOffset as estimateFacingYawOffsetModule,
      exportLiveRetargetProfile as exportLiveRetargetProfileModule,
      resetModelRootOrientation as resetModelRootOrientationModule,
      updateSourceOverlay as updateSourceOverlayModule,
    } from "./modules/retarget-live.js";
    import { DEFAULT_VRM_ANIMATION_URL, loadDefaultVrmAnimation } from "./modules/default-animation.js";
    import { buildVrmDirectBodyPlan } from "./modules/retarget-vrm.js";
    import {
      attemptPriority,
      buildCanonicalBoneMap,
      buildRenamedClip,
      buildRetargetMap,
      buildStageSourceClip,
      canonicalPoseSignature,
      collectLimbDiagnostics,
      resolvedTrackCountAcrossMeshes,
      resolvedTrackCountForTarget,
    } from "./modules/retarget-helpers.js";
    import { setupViewerUi } from "./modules/ui-controls.js";

    const wrap = document.getElementById("canvas-wrap");
    const statusEl = document.getElementById("status");
    const fileInput = document.getElementById("file-input");
    const modelInput = document.getElementById("model-input");
    const bvhFileNameEl = document.getElementById("bvh-file-name");
    const modelFileNameEl = document.getElementById("model-file-name");
    const btnRetarget = document.getElementById("retarget");
    const btnRetargetFab = document.getElementById("retarget-fab");
    const btnLoadDefault = document.getElementById("load-default");
    const btnZoomIn = document.getElementById("zoom-in");
    const btnZoomOut = document.getElementById("zoom-out");
    const btnPlay = document.getElementById("play");
    const btnPause = document.getElementById("pause");
    const btnStop = document.getElementById("stop");
    const timeline = document.getElementById("timeline");
    const timeEl = document.getElementById("time");
    const btnResetCamera = document.getElementById("reset-camera");
    const RIG_PROFILE_STORAGE_KEY = "vid2model.rigProfiles.v16";
    const MAX_RIG_PROFILE_ENTRIES = 12;

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
    let modelDefaultAnimationToken = 0;
    let modelLabel = "";
    let modelRigFingerprint = "";
    let modelVrmHumanoidBones = [];
    let modelVrmNormalizedHumanoidBones = [];
    let liveRetarget = null;
    let bodyLengthCalibration = null;
    let armLengthCalibration = null;
    let fingerLengthCalibration = null;
    let sourceOverlay = null;
    let sourceAxesDebug = null;
    let isPlaying = false;
    let isScrubbing = false;
    const loader = new BVHLoader();
    const gltfLoader = new GLTFLoader();
    const clock = new THREE.Clock();
    gltfLoader.register((parser) => new VRMLoaderPlugin(parser));
    const { diag } = createDiag(window);
    const SKELETON_COLOR = 0xff2d55;
    const SOURCE_POINT_COLOR = 0xfff400;
    const _overlayV1 = new THREE.Vector3();
    const _overlayV2 = new THREE.Vector3();
    const _overlayV3 = new THREE.Vector3();
    const _overlayV4 = new THREE.Vector3();
    const _overlayV5 = new THREE.Vector3();
    const _tmpWorldPosA = new THREE.Vector3();
    const _tmpWorldPosB = new THREE.Vector3();
    const _tmpWorldDelta = new THREE.Vector3();
    const _liveYawQ = new THREE.Quaternion();
    const _liveQ2 = new THREE.Quaternion();
    const _liveQ3 = new THREE.Quaternion();
    const _liveAxisY = new THREE.Vector3(0, 1, 0);
    const _rootYawQ = new THREE.Quaternion();
    const _calibV1 = new THREE.Vector3();
    const _calibV2 = new THREE.Vector3();
    const _calibV3 = new THREE.Vector3();
    const _calibV4 = new THREE.Vector3();
    const _calibQ1 = new THREE.Quaternion();
    const _calibQ2 = new THREE.Quaternion();
    const _calibM1 = new THREE.Matrix4();
    const _overlayPivot = new THREE.Vector3();
    const _overlayUpAxis = new THREE.Vector3(0, 1, 0);
    const _sourceAxisOrigin = new THREE.Vector3();
    const _sourceAxisHead = new THREE.Vector3();
    const _sourceAxisLeft = new THREE.Vector3();
    const _sourceAxisRight = new THREE.Vector3();
    const _sourceAxisForward = new THREE.Vector3();
    const _sourceAxisSide = new THREE.Vector3();
    const _sourceAxisUp = new THREE.Vector3();

    function shortErr(err, limit = 120) {
      const text = String(err?.message || err || "");
      return text.length > limit ? `${text.slice(0, limit)}...` : text;
    }

    const VRM_HUMANOID_BONE_NAMES = [
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

    function applyVrmHumanoidBoneNames(vrm) {
      const humanoid = vrm?.humanoid || null;
      if (!humanoid) return { applied: 0, renamed: 0, bones: [], normalizedBones: [] };
      let applied = 0;
      let renamed = 0;
      const bones = [];
      const normalizedBones = [];
      for (const boneName of VRM_HUMANOID_BONE_NAMES) {
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

    function getRetargetTargetBones(stage = getRetargetStage(), options = {}) {
      const fallback = modelSkinnedMesh?.skeleton?.bones || [];
      const preferNormalized = !!options.preferNormalized;
      const vrmBones =
        preferNormalized && modelVrmNormalizedHumanoidBones.length
          ? modelVrmNormalizedHumanoidBones
          : modelVrmHumanoidBones;
      if (!vrmBones.length) return fallback;
      if (stage === "body") {
        return vrmBones.filter((bone) =>
          RETARGET_BODY_CANONICAL.has(canonicalBoneKey(bone.name) || "")
        );
      }
      return vrmBones.slice();
    }

    function hashString(text) {
      let hash = 2166136261;
      for (let i = 0; i < text.length; i += 1) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
      }
      return (hash >>> 0).toString(16).padStart(8, "0");
    }

    function collectModelBoneRows(skinnedMeshes) {
      const rows = [];
      const seenBoneIds = new Set();
      for (const mesh of skinnedMeshes || []) {
        for (const bone of mesh.skeleton?.bones || []) {
          const boneId = bone.uuid || `${mesh.uuid}:${bone.name}`;
          if (seenBoneIds.has(boneId)) continue;
          seenBoneIds.add(boneId);
          rows.push({
            bone: bone.name || "(unnamed)",
            canonical: canonicalBoneKey(bone.name) || "",
            parent: bone.parent?.isBone ? bone.parent.name : "",
            mesh: mesh.name || "(unnamed-skinned-mesh)",
          });
        }
      }
      return rows;
    }

    function buildModelRigFingerprint(skinnedMeshes, label = "") {
      const rows = collectModelBoneRows(skinnedMeshes);
      const raw = [
        String(label || ""),
        ...rows.map((row) => `${row.mesh}|${row.parent}|${row.bone}|${row.canonical}`),
      ].join("\n");
      return `rig:${hashString(raw)}`;
    }

    function readStoredRigProfiles() {
      try {
        const raw = window.localStorage?.getItem(RIG_PROFILE_STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        return Array.isArray(parsed) ? parsed : [];
      } catch (err) {
        return [];
      }
    }

    function writeStoredRigProfiles(entries) {
      try {
        window.localStorage?.setItem(RIG_PROFILE_STORAGE_KEY, JSON.stringify(entries));
        return true;
      } catch (err) {
        return false;
      }
    }

    function loadRigProfile(modelFingerprint, stage) {
      if (!modelFingerprint || !stage) return null;
      return (
        readStoredRigProfiles().find(
          (entry) => entry?.modelFingerprint === modelFingerprint && entry?.stage === stage
        ) || null
      );
    }

    function saveRigProfile(entry) {
      if (!entry?.modelFingerprint || !entry?.stage) return false;
      const existing = loadRigProfile(entry.modelFingerprint, entry.stage);
      const existingError = existing?.postPoseError;
      const nextError = entry?.postPoseError;
      if (
        Number.isFinite(existingError) &&
        Number.isFinite(nextError) &&
        existingError <= nextError + 1e-6
      ) {
        return false;
      }
      const rows = readStoredRigProfiles().filter(
        (item) => !(item?.modelFingerprint === entry.modelFingerprint && item?.stage === entry.stage)
      );
      rows.unshift({
        ...entry,
        updatedAt: new Date().toISOString(),
      });
      return writeStoredRigProfiles(rows.slice(0, MAX_RIG_PROFILE_ENTRIES));
    }

    function applyRigProfileNames(baseResult, profile, targetBones, sourceBones, canonicalFilter) {
      if (!profile?.namesTargetToSource) return baseResult;
      const targetByName = new Map((targetBones || []).map((bone) => [bone.name, bone]));
      const sourceNameSet = new Set((sourceBones || []).map((bone) => bone.name));
      const names = { ...(baseResult?.names || {}) };
      for (const [targetName, sourceName] of Object.entries(profile.namesTargetToSource || {})) {
        const targetBone = targetByName.get(targetName);
        if (!targetBone || !sourceNameSet.has(sourceName)) continue;
        const canonical = canonicalBoneKey(targetBone.name);
        if (canonicalFilter && (!canonical || !canonicalFilter.has(canonical))) continue;
        names[targetName] = sourceName;
      }

      let matched = 0;
      let canonicalCandidates = 0;
      const unmatchedSample = [];
      const unmatchedHumanoid = [];
      for (const bone of targetBones || []) {
        const canonical = canonicalBoneKey(bone.name);
        if (canonicalFilter && (!canonical || !canonicalFilter.has(canonical))) {
          continue;
        }
        if (canonical) canonicalCandidates += 1;
        if (names[bone.name] && sourceNameSet.has(names[bone.name])) {
          matched += 1;
        } else if (unmatchedSample.length < 30) {
          unmatchedSample.push({ target: bone.name, canonical: canonical || "n/a" });
          if (canonical && unmatchedHumanoid.length < 30) {
            unmatchedHumanoid.push({ target: bone.name, canonical });
          }
        }
      }

      return {
        names,
        matched,
        canonicalCandidates,
        unmatchedSample,
        unmatchedHumanoid,
        sourceMatched: new Set(Object.values(names).filter((name) => sourceNameSet.has(name))).size,
      };
    }

    function maybeSwapMirroredHumanoidSides(baseResult, targetBones, sourceBones, canonicalFilter) {
      if (!baseResult?.names) return { ...baseResult, mirroredSidesApplied: false };
      const targetMap = buildCanonicalBoneMap(targetBones || []);
      const sourceMap = buildCanonicalBoneMap(sourceBones || []);
      const probes = [
        ["leftUpperLeg", "rightUpperLeg"],
        ["leftLowerLeg", "rightLowerLeg"],
        ["leftFoot", "rightFoot"],
        ["leftUpperArm", "rightUpperArm"],
        ["leftLowerArm", "rightLowerArm"],
        ["leftHand", "rightHand"],
      ];
      const v1 = new THREE.Vector3();
      const v2 = new THREE.Vector3();
      let mirroredVotes = 0;
      let totalVotes = 0;
      for (const [leftKey, rightKey] of probes) {
        const sourceLeft = sourceMap.get(leftKey);
        const sourceRight = sourceMap.get(rightKey);
        const targetLeft = targetMap.get(leftKey);
        const targetRight = targetMap.get(rightKey);
        if (!sourceLeft || !sourceRight || !targetLeft || !targetRight) continue;
        sourceLeft.getWorldPosition(v1);
        sourceRight.getWorldPosition(v2);
        const sourceSign = Math.sign(v1.x - v2.x);
        targetLeft.getWorldPosition(v1);
        targetRight.getWorldPosition(v2);
        const targetSign = Math.sign(v1.x - v2.x);
        if (!sourceSign || !targetSign) continue;
        totalVotes += 1;
        if (sourceSign !== targetSign) mirroredVotes += 1;
      }
      if (!totalVotes || mirroredVotes < Math.ceil(totalVotes / 2)) {
        return { ...baseResult, mirroredSidesApplied: false };
      }

      const sourceByCanonical = new Map();
      for (const bone of sourceBones || []) {
        const canonical = canonicalBoneKey(bone.name) || "";
        if (canonical) sourceByCanonical.set(canonical, bone.name);
      }
      const swapCanonical = (canonical) => {
        if (canonical.startsWith("left")) return `right${canonical.slice(4)}`;
        if (canonical.startsWith("right")) return `left${canonical.slice(5)}`;
        return canonical;
      };

      const names = { ...(baseResult.names || {}) };
      for (const bone of targetBones || []) {
        const canonical = canonicalBoneKey(bone.name) || "";
        if (!canonical) continue;
        if (canonicalFilter && !canonicalFilter.has(canonical)) continue;
        const swapped = swapCanonical(canonical);
        if (swapped === canonical) continue;
        const sourceName = sourceByCanonical.get(swapped);
        if (sourceName) names[bone.name] = sourceName;
      }

      let matched = 0;
      let canonicalCandidates = 0;
      const unmatchedSample = [];
      const unmatchedHumanoid = [];
      const sourceNameSet = new Set((sourceBones || []).map((bone) => bone.name));
      for (const bone of targetBones || []) {
        const canonical = canonicalBoneKey(bone.name);
        if (canonicalFilter && (!canonical || !canonicalFilter.has(canonical))) continue;
        if (canonical) canonicalCandidates += 1;
        if (names[bone.name] && sourceNameSet.has(names[bone.name])) {
          matched += 1;
        } else if (unmatchedSample.length < 30) {
          unmatchedSample.push({ target: bone.name, canonical: canonical || "n/a" });
          if (canonical && unmatchedHumanoid.length < 30) {
            unmatchedHumanoid.push({ target: bone.name, canonical });
          }
        }
      }

      return {
        names,
        matched,
        canonicalCandidates,
        unmatchedSample,
        unmatchedHumanoid,
        sourceMatched: new Set(Object.values(names).filter((name) => sourceNameSet.has(name))).size,
        mirroredSidesApplied: true,
      };
    }

    function quantizeFacingYaw(rad) {
      if (!Number.isFinite(rad)) return 0;
      const abs = Math.abs(rad);
      if (abs < THREE.MathUtils.degToRad(45)) return 0;
      if (abs > THREE.MathUtils.degToRad(120)) return Math.sign(rad || 1) * Math.PI;
      return rad;
    }

    function getRetargetStage() {
      const raw = String(window.__vid2modelRetargetStage || "body").trim().toLowerCase();
      return RETARGET_STAGES.has(raw) ? raw : "body";
    }

    function shouldUseVrmDirectBody() {
      return window.__vid2modelUseVrmDirect === true;
    }

    function getCanonicalFilterForStage(stage) {
      if (stage === "body") return RETARGET_BODY_CANONICAL;
      return null;
    }

    function resetModelRootOrientation() {
      resetModelRootOrientationModule({ modelRoot, getModelSkeletonRootBone });
    }

    function applyModelRootYaw(yawRad) {
      return applyModelRootYawModule({
        modelRoot,
        yawRad,
        axisY: _liveAxisY,
        rootYawQ: _rootYawQ,
      });
    }

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function createTextSprite(text, color = "#ffffff") {
      const canvas = document.createElement("canvas");
      canvas.width = 256;
      canvas.height = 96;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = "bold 36px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.lineWidth = 8;
      ctx.strokeStyle = "rgba(0,0,0,0.75)";
      ctx.fillStyle = color;
      ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
      ctx.fillText(text, canvas.width / 2, canvas.height / 2);
      const texture = new THREE.CanvasTexture(canvas);
      texture.needsUpdate = true;
      const material = new THREE.SpriteMaterial({
        map: texture,
        depthTest: false,
        depthWrite: false,
        transparent: true,
      });
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(26, 10, 1);
      sprite.renderOrder = 1000;
      return sprite;
    }

    function clearSourceAxesDebug() {
      if (!sourceAxesDebug) return;
      scene.remove(sourceAxesDebug.group);
      for (const sprite of Object.values(sourceAxesDebug.labels)) {
        sprite.material.map?.dispose?.();
        sprite.material.dispose?.();
      }
      for (const sprite of sourceAxesDebug.boneLabels || []) {
        sprite.material.map?.dispose?.();
        sprite.material.dispose?.();
      }
      sourceAxesDebug = null;
    }

    function createSourceAxesDebug(skeleton) {
      clearSourceAxesDebug();
      const group = new THREE.Group();
      const axisLength = 55;
      const headLength = 12;
      const headWidth = 6;
      const upArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), axisLength, 0x22c55e, headLength, headWidth);
      const rightArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), axisLength, 0xef4444, headLength, headWidth);
      const forwardArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), axisLength, 0x3b82f6, headLength, headWidth);
      const labels = {
        up: createTextSprite("UP", "#22c55e"),
        right: createTextSprite("RIGHT", "#ef4444"),
        forward: createTextSprite("FWD", "#3b82f6"),
        head: createTextSprite("HEAD", "#ffffff"),
        leftHand: createTextSprite("L", "#f59e0b"),
        rightHand: createTextSprite("R", "#f59e0b"),
        leftFoot: createTextSprite("LF", "#f59e0b"),
        rightFoot: createTextSprite("RF", "#f59e0b"),
      };
      const boneLabels = (skeleton?.bones || []).map((bone) => {
        const canonical = canonicalBoneKey(bone.name);
        const labelText = canonical || bone.name || "bone";
        const sprite = createTextSprite(labelText, canonical ? "#f8fafc" : "#cbd5e1");
        sprite.scale.set(22, 8, 1);
        return sprite;
      });
      group.add(upArrow, rightArrow, forwardArrow, ...Object.values(labels), ...boneLabels);
      scene.add(group);
      sourceAxesDebug = {
        skeleton,
        group,
        upArrow,
        rightArrow,
        forwardArrow,
        labels,
        boneLabels,
      };
      updateSourceAxesDebug();
    }

    function updateSourceAxesDebug() {
      if (!sourceAxesDebug?.skeleton?.bones?.length) return;
      const canonicalMap = buildCanonicalBoneMap(sourceAxesDebug.skeleton.bones);
      const hips = canonicalMap.get("hips") || sourceAxesDebug.skeleton.bones[0];
      if (!hips) return;
      hips.updateMatrixWorld(true);
      hips.getWorldPosition(_sourceAxisOrigin);

      const head = canonicalMap.get("head") || canonicalMap.get("neck") || canonicalMap.get("upperChest") || canonicalMap.get("chest");
      const leftHand = canonicalMap.get("leftHand");
      const rightHand = canonicalMap.get("rightHand");
      const leftFoot = canonicalMap.get("leftFoot") || canonicalMap.get("leftToes");
      const rightFoot = canonicalMap.get("rightFoot") || canonicalMap.get("rightToes");

      if (head) {
        head.getWorldPosition(_sourceAxisHead);
        _sourceAxisUp.copy(_sourceAxisHead).sub(_sourceAxisOrigin).normalize();
      } else {
        _sourceAxisUp.set(0, 1, 0);
      }
      if (_sourceAxisUp.lengthSq() < 1e-8) _sourceAxisUp.set(0, 1, 0);

      if (leftHand && rightHand) {
        leftHand.getWorldPosition(_sourceAxisLeft);
        rightHand.getWorldPosition(_sourceAxisRight);
        _sourceAxisSide.copy(_sourceAxisRight).sub(_sourceAxisLeft).normalize();
      } else {
        _sourceAxisSide.set(1, 0, 0);
      }
      if (_sourceAxisSide.lengthSq() < 1e-8) _sourceAxisSide.set(1, 0, 0);

      _sourceAxisForward.crossVectors(_sourceAxisSide, _sourceAxisUp).normalize();
      if (_sourceAxisForward.lengthSq() < 1e-8) _sourceAxisForward.set(0, 0, 1);

      const axisLength = 55;
      sourceAxesDebug.upArrow.position.copy(_sourceAxisOrigin);
      sourceAxesDebug.upArrow.setDirection(_sourceAxisUp);
      sourceAxesDebug.upArrow.setLength(axisLength, 12, 6);

      sourceAxesDebug.rightArrow.position.copy(_sourceAxisOrigin);
      sourceAxesDebug.rightArrow.setDirection(_sourceAxisSide);
      sourceAxesDebug.rightArrow.setLength(axisLength, 12, 6);

      sourceAxesDebug.forwardArrow.position.copy(_sourceAxisOrigin);
      sourceAxesDebug.forwardArrow.setDirection(_sourceAxisForward);
      sourceAxesDebug.forwardArrow.setLength(axisLength, 12, 6);

      sourceAxesDebug.labels.up.position.copy(_sourceAxisOrigin).addScaledVector(_sourceAxisUp, axisLength + 14);
      sourceAxesDebug.labels.right.position.copy(_sourceAxisOrigin).addScaledVector(_sourceAxisSide, axisLength + 14);
      sourceAxesDebug.labels.forward.position.copy(_sourceAxisOrigin).addScaledVector(_sourceAxisForward, axisLength + 14);

      if (head) {
        sourceAxesDebug.labels.head.visible = true;
        sourceAxesDebug.labels.head.position.copy(_sourceAxisHead).add(new THREE.Vector3(0, 10, 0));
      } else {
        sourceAxesDebug.labels.head.visible = false;
      }
      for (const [key, bone] of [["leftHand", leftHand], ["rightHand", rightHand], ["leftFoot", leftFoot], ["rightFoot", rightFoot]]) {
        const sprite = sourceAxesDebug.labels[key];
        if (bone) {
          bone.getWorldPosition(_tmpWorldPosA);
          sprite.visible = true;
          sprite.position.copy(_tmpWorldPosA).add(new THREE.Vector3(0, 6, 0));
        } else {
          sprite.visible = false;
        }
      }

      const showBoneLabels = window.__vid2modelShowBoneLabels !== false;
      const bones = sourceAxesDebug.skeleton.bones || [];
      for (let i = 0; i < bones.length; i += 1) {
        const sprite = sourceAxesDebug.boneLabels?.[i];
        const bone = bones[i];
        if (!sprite || !bone) continue;
        sprite.visible = showBoneLabels;
        if (!showBoneLabels) continue;
        bone.getWorldPosition(_tmpWorldPosA);
        sprite.position.copy(_tmpWorldPosA).add(new THREE.Vector3(0, 3.5, 0));
      }
    }

    function clearSourceOverlay() {
      sourceOverlay = clearSourceOverlayModule({ sourceOverlay, scene });
    }

    function estimateFacingYawOffset(sourceBones, targetBones) {
      return estimateFacingYawOffsetModule(sourceBones, targetBones);
    }

    function updateSourceOverlay() {
      updateSourceOverlayModule({
        sourceOverlay,
        overlayUpAxis: _overlayUpAxis,
        overlayPivot: _overlayPivot,
      });
      updateSourceAxesDebug();
    }

    function createSourceOverlay(skeleton) {
      sourceOverlay = createSourceOverlayModule({
        skeleton,
        scene,
        sourceOverlay,
        skeletonColor: SKELETON_COLOR,
        sourcePointColor: SOURCE_POINT_COLOR,
        clearSourceOverlay: () => {
          sourceOverlay = clearSourceOverlayModule({ sourceOverlay, scene });
        },
        updateSourceOverlay: (overlay) =>
          updateSourceOverlayModule({
            sourceOverlay: overlay,
            overlayUpAxis: _overlayUpAxis,
            overlayPivot: _overlayPivot,
          }),
      });
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

    function computeHipsWorldError(sourceBones, targetBones) {
      const sourceHips = findHipsBone(sourceBones);
      const targetHips = findHipsBone(targetBones);
      if (!sourceHips || !targetHips) return null;
      sourceHips.getWorldPosition(_tmpWorldPosA);
      targetHips.getWorldPosition(_tmpWorldPosB);
      return _tmpWorldPosA.distanceTo(_tmpWorldPosB);
    }

    function getModelSkeletonRootBone() {
      if (!modelSkinnedMesh?.skeleton?.bones?.length) return null;
      let node = findHipsBone(modelSkinnedMesh.skeleton.bones) || modelSkinnedMesh.skeleton.bones[0];
      while (node?.parent?.isBone) {
        node = node.parent;
      }
      return node || null;
    }

    function applyWorldDeltaToObject(obj, deltaWorld) {
      if (!obj || !deltaWorld || deltaWorld.lengthSq() < 1e-12) return;
      if (!obj.parent) {
        obj.position.add(deltaWorld);
        return;
      }
      obj.getWorldPosition(_tmpWorldPosA);
      _tmpWorldPosA.add(deltaWorld);
      obj.parent.worldToLocal(_tmpWorldPosA);
      obj.position.copy(_tmpWorldPosA);
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
        // Keep source overlay in source coordinates by default; extra yaw is applied
        // only in live-delta mode where explicit runtime correction is used.
        overlayYaw = estimateFacingYawOffset(sourceResult.skeleton.bones, modelSkinnedMesh.skeleton.bones);
        sourceOverlay.overlayYaw = 0;
      }
      diag("display-sync", {
        sourceHeight: Number(sourceH.toFixed(4)),
        targetHeight: Number(targetH.toFixed(4)),
        displayScale: Number(skeletonObj.scale.y.toFixed(6)),
        overlayYawDeg: Number((overlayYaw * 180 / Math.PI).toFixed(2)),
      });
      updateSourceOverlay();
    }

    function alignSourceHipsToModel(lockFeet = false) {
      if (!skeletonObj || !sourceResult?.skeleton?.bones?.length || !modelRoot || !modelSkinnedMesh?.skeleton?.bones?.length) {
        return;
      }
      skeletonObj.updateMatrixWorld(true);
      modelRoot.updateMatrixWorld(true);
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
      if (lockFeet) {
        const sourceFeetY = findFootLevelY(sourceResult.skeleton.bones);
        const targetFeetY = findFootLevelY(modelSkinnedMesh.skeleton.bones);
        if (Number.isFinite(sourceFeetY) && Number.isFinite(targetFeetY)) {
          skeletonObj.position.y += targetFeetY - sourceFeetY;
          skeletonObj.updateMatrixWorld(true);
        }
      }
    }

    function alignModelHipsToSource(lockFeet = false) {
      if (!skeletonObj || !sourceResult?.skeleton?.bones?.length || !modelRoot || !modelSkinnedMesh?.skeleton?.bones?.length) {
        return null;
      }
      skeletonObj.updateMatrixWorld(true);
      modelRoot.updateMatrixWorld(true);
      const sourceHips = findHipsBone(sourceResult.skeleton.bones);
      const targetHips = findHipsBone(modelSkinnedMesh.skeleton.bones);
      if (!sourceHips || !targetHips) {
        return { applied: false, target: "none", beforeErr: null, deltaX: 0, deltaY: 0, deltaZ: 0, lockFeet, hipsPosErr: null };
      }
      sourceHips.getWorldPosition(_tmpWorldPosA);
      targetHips.getWorldPosition(_tmpWorldPosB);
      _tmpWorldDelta.copy(_tmpWorldPosA).sub(_tmpWorldPosB);
      const delta = _tmpWorldDelta.clone();
      const beforeErrRaw = computeHipsWorldError(sourceResult.skeleton.bones, modelSkinnedMesh.skeleton.bones);
      const beforeErr = Number.isFinite(beforeErrRaw) ? Number(beforeErrRaw.toFixed(5)) : null;
      const candidates = [modelRoot, getModelSkeletonRootBone()].filter(
        (node, idx, arr) => node && arr.findIndex((v) => v && v.uuid === node.uuid) === idx
      );
      let bestTarget = "none";
      let bestErr = Number.POSITIVE_INFINITY;
      for (const node of candidates) {
        const prevPos = node.position.clone();
        applyWorldDeltaToObject(node, delta);
        modelRoot.updateMatrixWorld(true);
        const err = computeHipsWorldError(sourceResult.skeleton.bones, modelSkinnedMesh.skeleton.bones);
        const errNum = Number.isFinite(err) ? err : Number.POSITIVE_INFINITY;
        if (errNum < bestErr) {
          bestErr = errNum;
          bestTarget = node === modelRoot ? "modelRoot" : "skeletonRootBone";
        }
        node.position.copy(prevPos);
      }
      if (bestTarget === "skeletonRootBone") {
        const rootBone = getModelSkeletonRootBone();
        if (rootBone) {
          applyWorldDeltaToObject(rootBone, delta);
        }
      } else {
        applyWorldDeltaToObject(modelRoot, delta);
      }
      modelRoot.updateMatrixWorld(true);
      if (lockFeet) {
        const sourceFeetY = findFootLevelY(sourceResult.skeleton.bones);
        const targetFeetY = findFootLevelY(modelSkinnedMesh.skeleton.bones);
        if (Number.isFinite(sourceFeetY) && Number.isFinite(targetFeetY)) {
          modelRoot.position.y += sourceFeetY - targetFeetY;
          modelRoot.updateMatrixWorld(true);
        }
      }
      const hipsPosErrRaw = computeHipsWorldError(sourceResult.skeleton.bones, modelSkinnedMesh.skeleton.bones);
      const hipsPosErr = Number.isFinite(hipsPosErrRaw) ? Number(hipsPosErrRaw.toFixed(5)) : null;
      return {
        applied: true,
        target: bestTarget,
        beforeErr,
        deltaX: Number(delta.x.toFixed(5)),
        deltaY: Number(delta.y.toFixed(5)),
        deltaZ: Number(delta.z.toFixed(5)),
        lockFeet,
        hipsPosErr,
      };
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
      clearSourceAxesDebug();
      sourceResult = null;
      mixer = null;
      currentClip = null;
      currentAction = null;
      liveRetarget = null;
      bodyLengthCalibration = null;
      armLengthCalibration = null;
      fingerLengthCalibration = null;
      updateTimelineUi(0);
    }

    function clearModel() {
      if (modelRoot) {
        scene.remove(modelRoot);
      }
      modelDefaultAnimationToken += 1;
      modelRoot = null;
      modelLabel = "";
      modelRigFingerprint = "";
      modelVrmHumanoidBones = [];
      modelVrmNormalizedHumanoidBones = [];
      modelSkinnedMesh = null;
      modelSkinnedMeshes = [];
      modelMixer = null;
      modelAction = null;
      modelMixers = [];
      modelActions = [];
      liveRetarget = null;
      bodyLengthCalibration = null;
      armLengthCalibration = null;
      fingerLengthCalibration = null;
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

    function clipUsesBonesSyntax(clip) {
      return clip.tracks.some((t) => t.name.startsWith(".bones["));
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

    function logModelBones(skinnedMeshes) {
      const rows = collectModelBoneRows(skinnedMeshes).map((row, index) => ({ index, ...row }));
      window.__vid2modelModelBones = rows;
      if (String(window.__vid2modelDiagMode || "minimal").trim().toLowerCase() === "verbose") {
        console.log("[vid2model/model-bones] total:", rows.length);
        console.table(rows);
      }
    }

    function getPrimaryChildDirectionLocal(bone, outDir) {
      if (!bone || !outDir) return false;
      bone.getWorldPosition(_calibV1);
      let bestChild = null;
      let bestLenSq = 0;
      for (const child of bone.children || []) {
        if (!child?.isBone) continue;
        child.getWorldPosition(_calibV2);
        const lenSq = _calibV2.distanceToSquared(_calibV1);
        if (lenSq > bestLenSq) {
          bestLenSq = lenSq;
          bestChild = child;
        }
      }
      if (!bestChild || bestLenSq < 1e-8) return false;
      bestChild.getWorldPosition(_calibV2);
      outDir.copy(_calibV2).sub(_calibV1);
      bone.getWorldQuaternion(_calibQ1);
      outDir.applyQuaternion(_calibQ1.invert());
      if (outDir.lengthSq() < 1e-10) return false;
      outDir.normalize();
      return true;
    }

    function getReferenceDirectionLocal(bone, primaryDir, outDir) {
      if (!bone || !primaryDir || !outDir) return false;
      bone.getWorldPosition(_calibV1);
      bone.getWorldQuaternion(_calibQ1);
      const canonical = canonicalBoneKey(bone.name) || "";
      const invBoneQ = _calibQ1.invert();
      const tryWorldVector = (worldVector) => {
        outDir.copy(worldVector).applyQuaternion(invBoneQ);
        outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
        if (outDir.lengthSq() < 1e-8) return false;
        outDir.normalize();
        return true;
      };

      if (canonical === "leftUpperLeg" || canonical === "rightUpperLeg") {
        const siblingCanonical = canonical === "leftUpperLeg" ? "rightUpperLeg" : "leftUpperLeg";
        for (const sibling of bone.parent?.children || []) {
          if (!sibling?.isBone || sibling === bone) continue;
          if ((canonicalBoneKey(sibling.name) || "") !== siblingCanonical) continue;
          sibling.getWorldPosition(_calibV2);
          outDir.copy(_calibV2).sub(_calibV1).applyQuaternion(invBoneQ);
          outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
          if (outDir.lengthSq() >= 1e-8) {
            outDir.normalize();
            return true;
          }
        }
      }

      if (bone.parent?.isBone) {
        bone.parent.getWorldPosition(_calibV2);
        outDir.copy(_calibV2).sub(_calibV1).applyQuaternion(invBoneQ);
        outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
        if (outDir.lengthSq() >= 1e-8) {
          outDir.normalize();
          return true;
        }
      }

      let bestChild = null;
      let bestLenSq = 0;
      for (const child of bone.children || []) {
        if (!child?.isBone) continue;
        child.getWorldPosition(_calibV2);
        outDir.copy(_calibV2).sub(_calibV1);
        const lenSq = outDir.lengthSq();
        if (lenSq > bestLenSq) {
          bestLenSq = lenSq;
          bestChild = child;
        }
      }
      if (bestChild) {
        for (const child of bone.children || []) {
          if (!child?.isBone || child === bestChild) continue;
          child.getWorldPosition(_calibV2);
          outDir.copy(_calibV2).sub(_calibV1).applyQuaternion(invBoneQ);
          outDir.addScaledVector(primaryDir, -outDir.dot(primaryDir));
          if (outDir.lengthSq() >= 1e-8) {
            outDir.normalize();
            return true;
          }
        }
      }

      return (
        tryWorldVector(_calibV2.set(1, 0, 0)) ||
        tryWorldVector(_calibV2.set(0, 1, 0)) ||
        tryWorldVector(_calibV2.set(0, 0, 1))
      );
    }

    function buildBoneLocalFrame(bone, outQ) {
      if (!bone || !outQ) return false;
      if (!getPrimaryChildDirectionLocal(bone, _calibV1)) return false;
      if (!getReferenceDirectionLocal(bone, _calibV1, _calibV2)) return false;
      _calibV3.crossVectors(_calibV2, _calibV1);
      if (_calibV3.lengthSq() < 1e-8) return false;
      _calibV3.normalize();
      _calibV4.crossVectors(_calibV1, _calibV3);
      if (_calibV4.lengthSq() < 1e-8) return false;
      _calibV4.normalize();
      _calibM1.makeBasis(_calibV4, _calibV1, _calibV3);
      outQ.setFromRotationMatrix(_calibM1).normalize();
      return true;
    }

    function buildRestOrientationCorrection(sourceBone, targetBone) {
      const sourceFrame = _calibQ1;
      const targetFrame = _calibQ2;
      if (buildBoneLocalFrame(sourceBone, sourceFrame) && buildBoneLocalFrame(targetBone, targetFrame)) {
        const corr = new THREE.Quaternion().copy(targetFrame).multiply(sourceFrame.invert()).normalize();
        if (Math.abs(corr.w) > 0.9995) {
          return null;
        }
        return corr;
      }
      const sourceDir = new THREE.Vector3();
      const targetDir = new THREE.Vector3();
      if (!getPrimaryChildDirectionLocal(sourceBone, sourceDir)) return null;
      if (!getPrimaryChildDirectionLocal(targetBone, targetDir)) return null;
      const dot = Math.max(-1, Math.min(1, sourceDir.dot(targetDir)));
      if (dot > 0.9995) return null;
      return new THREE.Quaternion().setFromUnitVectors(sourceDir, targetDir).normalize();
    }

    function buildLiveRetargetPlan(skinnedMeshes, sourceBones, namesTargetToSource, targetBones = null) {
      return buildLiveRetargetPlanModule({
        skinnedMeshes,
        targetBones,
        sourceBones,
        namesTargetToSource,
        mixer,
        modelRoot,
        buildRestOrientationCorrection,
        cachedProfile: loadRigProfile(modelRigFingerprint, getRetargetStage())?.liveRetarget || null,
      });
    }

    function exportLiveRetargetProfile(plan) {
      return exportLiveRetargetProfileModule(plan);
    }

    const _liveQ = new THREE.Quaternion();
    const _liveV = new THREE.Vector3();
    function applyLiveRetargetPose(plan) {
      applyLiveRetargetPoseModule({
        plan,
        modelRoot,
        liveAxisY: _liveAxisY,
        liveYawQ: _liveYawQ,
        liveQ: _liveQ,
        liveQ2: _liveQ2,
        liveQ3: _liveQ3,
        liveV: _liveV,
      });
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

    function isLowMatchRetargetMap(result, sourceTotal) {
      const targetCoverage = result.canonicalCandidates > 0 ? result.matched / result.canonicalCandidates : 0;
      const sourceCoverage = sourceTotal > 0 ? result.sourceMatched / sourceTotal : 0;
      return (
        result.canonicalCandidates === 0 ||
        targetCoverage < 0.75 ||
        (targetCoverage < 0.9 && sourceCoverage < 0.35)
      );
    }

    function isRetargetMapBetter(candidate, baseline) {
      if (!candidate) return false;
      if (!baseline) return true;
      if (candidate.matched !== baseline.matched) return candidate.matched > baseline.matched;
      if (candidate.sourceMatched !== baseline.sourceMatched) return candidate.sourceMatched > baseline.sourceMatched;
      return Object.keys(candidate.names || {}).length > Object.keys(baseline.names || {}).length;
    }

    function buildTopologyFallbackRenamePlan(skinnedMesh, baseResult, canonicalFilter) {
      const bones = skinnedMesh?.skeleton?.bones || [];
      if (!bones.length) return null;

      skinnedMesh.updateMatrixWorld(true);
      const inferred = inferHumanoidCanonicalNamesFromTopology(bones);
      if (!inferred.size) return null;

      const reservedCanonical = new Set();
      for (const targetName of Object.keys(baseResult?.names || {})) {
        const canonical = canonicalBoneKey(targetName);
        if (canonical) reservedCanonical.add(canonical);
      }

      const reservedNames = new Set(
        bones
          .map((bone) => String(bone?.name || "").trim())
          .filter((name) => !!name)
      );

      const plan = [];
      for (const [bone, canonical] of inferred.entries()) {
        if (!canonical) continue;
        if (canonicalFilter && !canonicalFilter.has(canonical)) continue;
        if (baseResult?.names?.[bone.name]) continue;

        const currentCanonical = canonicalBoneKey(bone.name);
        if (currentCanonical === canonical) continue;
        if (reservedCanonical.has(canonical)) continue;
        if (reservedNames.has(canonical) && bone.name !== canonical) continue;

        plan.push({
          bone,
          from: bone.name,
          to: canonical,
          previousCanonical: currentCanonical || null,
        });
        reservedCanonical.add(canonical);
        reservedNames.delete(bone.name);
        reservedNames.add(canonical);
      }

      if (!plan.length) return null;
      return {
        plan,
        sample: plan.slice(0, 12).map((row) => ({
          from: row.from,
          to: row.to,
          previousCanonical: row.previousCanonical,
        })),
      };
    }

    function applyBoneRenamePlan(plan) {
      for (const row of plan || []) {
        row.bone.name = row.to;
      }
    }

    function revertBoneRenamePlan(plan) {
      for (const row of plan || []) {
        row.bone.name = row.from;
      }
    }

    function maybeApplyTopologyFallback(skinnedMesh, sourceBones, canonicalFilter, baseResult) {
      const sourceTotal = sourceBones?.length || 0;
      const shouldTry = isLowMatchRetargetMap(baseResult, sourceTotal);
      const baseMappedPairs = Object.keys(baseResult?.names || {}).length;
      const baseTargetCoverage =
        baseResult?.canonicalCandidates > 0 ? baseResult.matched / baseResult.canonicalCandidates : 0;
      const baseSourceCoverage = sourceTotal > 0 ? baseResult.sourceMatched / sourceTotal : 0;

      if (!shouldTry) {
        return {
          result: baseResult,
          attempted: false,
          applied: false,
          reason: "coverage-ok",
        };
      }

      const renameInfo = buildTopologyFallbackRenamePlan(skinnedMesh, baseResult, canonicalFilter);
      if (!renameInfo?.plan?.length) {
        return {
          result: baseResult,
          attempted: true,
          applied: false,
          reason: "no-inferred-renames",
          before: {
            mappedPairs: baseMappedPairs,
            targetCoverage: Number(baseTargetCoverage.toFixed(4)),
            sourceCoverage: Number(baseSourceCoverage.toFixed(4)),
          },
        };
      }

      applyBoneRenamePlan(renameInfo.plan);
      const topologyResult = buildRetargetMap(skinnedMesh.skeleton.bones, sourceBones, { canonicalFilter });
      const topologyMappedPairs = Object.keys(topologyResult.names || {}).length;
      const topologyTargetCoverage =
        topologyResult.canonicalCandidates > 0 ? topologyResult.matched / topologyResult.canonicalCandidates : 0;
      const topologySourceCoverage =
        sourceTotal > 0 ? topologyResult.sourceMatched / sourceTotal : 0;
      const better = isRetargetMapBetter(topologyResult, baseResult);

      if (!better) {
        revertBoneRenamePlan(renameInfo.plan);
      }

      return {
        result: better ? topologyResult : baseResult,
        attempted: true,
        applied: better,
        reason: better ? "improved-coverage" : "not-better",
        inferredRenames: renameInfo.plan.length,
        sample: renameInfo.sample,
        before: {
          mappedPairs: baseMappedPairs,
          targetCoverage: Number(baseTargetCoverage.toFixed(4)),
          sourceCoverage: Number(baseSourceCoverage.toFixed(4)),
        },
        after: {
          mappedPairs: topologyMappedPairs,
          targetCoverage: Number(topologyTargetCoverage.toFixed(4)),
          sourceCoverage: Number(topologySourceCoverage.toFixed(4)),
        },
      };
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
        const retargetStage = getRetargetStage();
        const stageClip = buildStageSourceClip(
          sourceResult.clip,
          sourceResult.skeleton.bones,
          retargetStage,
          getCanonicalFilterForStage(retargetStage)
        );
        if (!stageClip) {
          setStatus(`Retarget failed: no source tracks for stage "${retargetStage}".`);
          diag("retarget-fail", { reason: "empty_stage_clip", stage: retargetStage });
          return;
        }
        const canonicalFilter = getCanonicalFilterForStage(retargetStage);
        const retargetTargetBones = getRetargetTargetBones(retargetStage);
        const normalizedDirectBones = getRetargetTargetBones(retargetStage, { preferNormalized: true });
        const directRetargetTargetBones = normalizedDirectBones.length ? normalizedDirectBones : retargetTargetBones;
        resetModelRootOrientation();
        const modelUsesPreferredRetargetHeuristics = /(^|[\\/])(en_0|MoonGirl)\.vrm$/i.test(modelLabel);
        const preferVrmDirectBody =
          retargetStage === "body" &&
          directRetargetTargetBones.length > 0 &&
          shouldUseVrmDirectBody();
        const cachedRigProfile = loadRigProfile(modelRigFingerprint, retargetStage);
        let rigProfileSaved = false;
        const initialMap = buildRetargetMap(
          retargetTargetBones,
          sourceResult.skeleton.bones,
          { canonicalFilter }
        );
        const topologyFallback = maybeApplyTopologyFallback(
          modelSkinnedMesh,
          sourceResult.skeleton.bones,
          canonicalFilter,
          initialMap
        );
        const profiledMapResult = applyRigProfileNames(
          topologyFallback.result,
          cachedRigProfile,
          retargetTargetBones,
          sourceResult.skeleton.bones,
          canonicalFilter
        );
        const activeMapResult =
          modelVrmHumanoidBones.length > 0 && modelUsesPreferredRetargetHeuristics
            ? { ...profiledMapResult, mirroredSidesApplied: false }
            : maybeSwapMirroredHumanoidSides(
                profiledMapResult,
                retargetTargetBones,
                sourceResult.skeleton.bones,
                canonicalFilter
              );
        const {
          names,
          matched,
          unmatchedSample,
          canonicalCandidates,
          unmatchedHumanoid,
          sourceMatched,
        } = activeMapResult;
        const mappedPairs = Object.keys(names).length;
        diag("retarget-input", {
          stage: retargetStage,
          sourceBones: sourceResult.skeleton.bones.length,
          targetBones: retargetTargetBones.length,
          sourceTracks: stageClip.tracks.length,
          mappedPairs,
          uniqueSourceMapped: sourceMatched,
          humanoidMatched: canonicalCandidates > 0 ? `${matched}/${canonicalCandidates}` : "n/a",
        });
        diag("retarget-topology-fallback", {
          stage: retargetStage,
          attempted: topologyFallback.attempted,
          applied: topologyFallback.applied,
          reason: topologyFallback.reason,
          inferredRenames: topologyFallback.inferredRenames || 0,
          before: topologyFallback.before || null,
          after: topologyFallback.after || null,
          sample: topologyFallback.sample || [],
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
          SkeletonUtils.retargetClip(modelSkinnedMesh, sourceResult.skeleton, stageClip, {
            names,
            hip: "hips",
            useFirstFramePosition: true,
            preserveBonePositions: false,
          })
        );
        const namesSourceToTarget = Object.fromEntries(Object.entries(names).map(([target, source]) => [source, target]));
        pushAttempt("skeletonutils-skinnedmesh-reversed", "skinned", () =>
          SkeletonUtils.retargetClip(modelSkinnedMesh, sourceResult.skeleton, stageClip, {
            names: namesSourceToTarget,
            hip: "hips",
            useFirstFramePosition: true,
            preserveBonePositions: false,
          })
        );
        if (modelRoot) {
          pushAttempt("skeletonutils-root", "root", () =>
            SkeletonUtils.retargetClip(modelRoot, sourceResult.skeleton, stageClip, {
              names,
              hip: "hips",
              useFirstFramePosition: true,
              preserveBonePositions: false,
            })
          );
          pushAttempt("skeletonutils-root-reversed", "root", () =>
            SkeletonUtils.retargetClip(modelRoot, sourceResult.skeleton, stageClip, {
              names: namesSourceToTarget,
              hip: "hips",
              useFirstFramePosition: true,
              preserveBonePositions: false,
            })
          );
        }

        const sourceRootBoneName = sourceResult.skeleton.bones?.[0]?.name || "hips";
        pushAttempt("rename-fallback-bones", "skinned", () =>
          buildRenamedClip(stageClip, names, sourceRootBoneName, "bones")
        );
        pushAttempt("rename-fallback-object", "root", () =>
          buildRenamedClip(stageClip, names, sourceRootBoneName, "object")
        );

        modelMixers = [];
        modelActions = [];
        modelMixer = null;
        modelAction = null;
        liveRetarget = null;

        if (!retargetAttempts.length) {
          setStatus("Retarget failed: 0 tracks produced. Bone names do not match.");
          const unmatched = (unmatchedHumanoid.length ? unmatchedHumanoid : unmatchedSample).slice(0, 8);
          diag("retarget-fail", { reason: "no_tracks", stage: retargetStage, unmatched });
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
          const bindings = buildBindingsForAttempt({
            attempt,
            clip: attempt.clip,
            modelSkinnedMeshes,
            modelRoot,
            modelSkinnedMesh,
            clipUsesBonesSyntax,
            resolvedTrackCountForTarget,
          });
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
          const probe = probeMotionForBindings({
            bindings,
            clip: attempt.clip,
            modelSkinnedMesh,
            modelRoot,
            targetBones: retargetTargetBones,
          });
          const poseError = computePoseMatchError({
            bindings,
            sampleTime: probe.sampleTime,
            modelSkinnedMesh,
            targetBones: retargetTargetBones,
            sourceResult,
            mixer,
            modelRoot,
            buildCanonicalBoneMap,
            canonicalPoseSignature,
          });
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
          diag("retarget-fail", { reason: "no_resolved_tracks", stage: retargetStage });
          return;
        }

        if (
          modelUsesPreferredRetargetHeuristics &&
          selectedAttempt?.label?.startsWith("rename-fallback") &&
          skeletonSkinned
        ) {
          const skeletonBindings = buildBindingsForAttempt({
            attempt: skeletonSkinned,
            clip: skeletonSkinned.clip,
            modelSkinnedMeshes,
            modelRoot,
            modelSkinnedMesh,
            clipUsesBonesSyntax,
            resolvedTrackCountForTarget,
          });
          if (skeletonBindings.mixers.length) {
            const skeletonProbe = probeMotionForBindings({
              bindings: skeletonBindings,
              clip: skeletonSkinned.clip,
              modelSkinnedMesh,
              modelRoot,
              targetBones: retargetTargetBones,
            });
            const skeletonPoseError = computePoseMatchError({
              bindings: skeletonBindings,
              sampleTime: skeletonProbe.sampleTime,
              modelSkinnedMesh,
              targetBones: retargetTargetBones,
              sourceResult,
              mixer,
              modelRoot,
              buildCanonicalBoneMap,
              canonicalPoseSignature,
            });
            const resolvedGap = selectedAttempt.resolvedTracks - skeletonSkinned.resolvedTracks;
            const poseGap = skeletonPoseError - selectedPoseError;
            const shouldPreferSkeleton =
              resolvedGap <= 2 &&
              (!Number.isFinite(skeletonPoseError) || !Number.isFinite(selectedPoseError) || poseGap <= 0.35);
            if (shouldPreferSkeleton) {
              selectedAttempt = skeletonSkinned;
              selectedBindings = skeletonBindings;
              selectedProbe = skeletonProbe;
              selectedPoseError = skeletonPoseError;
            }
          }
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
          retargetTargetBones
        );
        const strongFacingMismatch = Math.abs(rawFacingYaw) > THREE.MathUtils.degToRad(100);
        const weakMotion = !!selectedProbe && selectedProbe.score < 0.5;
        const highPoseError = Number.isFinite(selectedPoseError) && selectedPoseError > 0.6;
        const selectedIsSkeletonUtils = selectedAttempt.label.startsWith("skeletonutils");
        const fullHumanoidMatch = canonicalCandidates > 0 && matched === canonicalCandidates;
        const severeFacingPoseMismatch = strongFacingMismatch && highPoseError;
        const autoUseLiveDelta =
          isRenameFallback ||
          (
            selectedIsSkeletonUtils &&
            modelUsesPreferredRetargetHeuristics &&
            (severeFacingPoseMismatch || (!fullHumanoidMatch && (highPoseError || strongFacingMismatch)))
          ) ||
          (!selectedIsSkeletonUtils && (strongFacingMismatch || weakMotion || highPoseError));
        const forcedLiveDelta = getLiveDeltaOverride(window);
        const useLiveDelta = forcedLiveDelta === null ? autoUseLiveDelta : forcedLiveDelta;
        diag("retarget-live-delta", {
          stage: retargetStage,
          selectedMode: selectedAttempt.label,
          selectedIsSkeletonUtils,
          modelUsesPreferredRetargetHeuristics,
          fullHumanoidMatch,
          autoUseLiveDelta,
          forced: forcedLiveDelta,
          useLiveDelta,
          reasons: {
            isRenameFallback,
            strongFacingMismatch,
            weakMotion,
            highPoseError,
            severeFacingPoseMismatch,
          },
        });
        let rootYawCorrection = 0;
        if (preferVrmDirectBody) {
          const directPlan = buildVrmDirectBodyPlan({
            targetBones: directRetargetTargetBones,
            sourceBones: sourceResult.skeleton.bones,
            namesTargetToSource: names,
            mixer,
            modelRoot,
            buildRestOrientationCorrection,
          });
          if (directPlan && directPlan.pairs.length > 0) {
            liveRetarget = directPlan;
            applyLiveRetargetPose(liveRetarget);
            modelMixers = [];
            modelActions = [];
            modelMixer = null;
            modelAction = null;
            selectedModeLabel = "vrm-humanoid-direct";
          }
        }
        if (!liveRetarget && selectedIsSkeletonUtils && !useLiveDelta) {
          const yawCandidates = buildRootYawCandidates(rawFacingYaw, quantizeFacingYaw);
          const yawEval = evaluateRootYawCandidates({
            candidates: yawCandidates,
            sampleTime: selectedProbe?.sampleTime || 0,
            namesTargetToSource: names,
            sourceClip: clip,
            modelRoot,
            modelMixers,
            modelSkinnedMesh,
            targetBones: retargetTargetBones,
            sourceResult,
            mixer,
            resetModelRootOrientation,
            applyModelRootYaw,
            collectAlignmentDiagnostics: (args) =>
              collectAlignmentDiagnostics({
                ...args,
                sourceOverlay,
                overlayUpAxis: _overlayUpAxis,
              }),
          });
          const zeroRow = yawEval.rows.find((r) => Math.abs(r.yawDeg) < 0.01) || null;
          const bestRow = yawEval.rows[0] || null;
          const bestIsLargeFlip = !!bestRow && Math.abs(bestRow.yawDeg) > 120;
          const rawYawLooksAligned = Math.abs(rawFacingYaw) < THREE.MathUtils.degToRad(30);
          const shouldUseBest =
            !!bestRow &&
            !(rawYawLooksAligned && bestIsLargeFlip) &&
            (
              !zeroRow ||
              bestRow.score + 0.03 < zeroRow.score ||
              (Number.isFinite(bestRow.hipsPosErr) && Number.isFinite(zeroRow.hipsPosErr) && bestRow.hipsPosErr + 0.03 < zeroRow.hipsPosErr)
            );
          rootYawCorrection = applyModelRootYaw(shouldUseBest ? yawEval.bestYaw : 0);
          if (sourceOverlay) {
            sourceOverlay.overlayYaw = 0;
            updateSourceOverlay();
          }
          const hipsYawError = computeHipsYawError(
            retargetTargetBones,
            sourceResult.skeleton.bones,
            names
          );
          let hipsYawCorrection = 0;
          let hipsCorrectionApplied = false;
          let hipsCorrectionEval = null;
          const hipsCorrectionWouldLargeFlip = Math.abs(hipsYawError) > THREE.MathUtils.degToRad(120);
          if (
            Math.abs(hipsYawError) > THREE.MathUtils.degToRad(12) &&
            !(rawYawLooksAligned && hipsCorrectionWouldLargeFlip)
          ) {
            const correctedYaw = rootYawCorrection - hipsYawError;
            const postEval = evaluateRootYawCandidates({
              candidates: [rootYawCorrection, correctedYaw],
              sampleTime: selectedProbe?.sampleTime || 0,
              namesTargetToSource: names,
              sourceClip: clip,
              modelRoot,
              modelMixers,
              modelSkinnedMesh,
              targetBones: retargetTargetBones,
              sourceResult,
              mixer,
              resetModelRootOrientation,
              applyModelRootYaw,
              collectAlignmentDiagnostics: (args) =>
                collectAlignmentDiagnostics({
                  ...args,
                  sourceOverlay,
                  overlayUpAxis: _overlayUpAxis,
                }),
            });
            const sameYaw = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b))) < 1e-4;
            const currentRow = postEval.rows.find((r) => sameYaw(r.yawRad, rootYawCorrection)) || null;
            const correctedRow = postEval.rows.find((r) => sameYaw(r.yawRad, correctedYaw)) || null;
            hipsCorrectionEval = { current: currentRow, corrected: correctedRow };
            const shouldApplyCorrection =
              !!currentRow &&
              !!correctedRow &&
              correctedRow.score + 0.03 < currentRow.score;
            if (shouldApplyCorrection) {
              hipsYawCorrection = -hipsYawError;
              rootYawCorrection = applyModelRootYaw(correctedYaw);
              hipsCorrectionApplied = true;
              if (sourceOverlay) {
                sourceOverlay.overlayYaw = 0;
                updateSourceOverlay();
              }
            }
          }
          diag("retarget-root-yaw", {
            stage: retargetStage,
            rawFacingYawDeg: Number((rawFacingYaw * 180 / Math.PI).toFixed(2)),
            appliedYawDeg: Number((rootYawCorrection * 180 / Math.PI).toFixed(2)),
            hipsYawErrorDeg: Number((hipsYawError * 180 / Math.PI).toFixed(2)),
            hipsYawCorrectionDeg: Number((hipsYawCorrection * 180 / Math.PI).toFixed(2)),
            hipsCorrectionApplied,
            hipsCorrectionEval,
            strongFacingMismatch,
            usedBestCandidate: shouldUseBest,
            zeroCandidate: zeroRow,
            bestCandidate: bestRow,
            candidates: yawEval.rows,
          });
        }
        if (!liveRetarget && useLiveDelta) {
          const livePlan = buildLiveRetargetPlan(
            modelSkinnedMeshes,
            sourceResult.skeleton.bones,
            names,
            retargetTargetBones
          );
          if (livePlan && livePlan.pairs.length > 0) {
            liveRetarget = livePlan;
            if (sourceOverlay) {
              const effectiveOverlayYaw = rawFacingYaw + liveRetarget.yawOffset;
              sourceOverlay.overlayYaw = Number.isFinite(effectiveOverlayYaw) ? effectiveOverlayYaw : 0;
              updateSourceOverlay();
            }
            applyLiveRetargetPose(liveRetarget);
            modelMixers = [];
            modelActions = [];
            modelMixer = null;
            modelAction = null;
            selectedModeLabel = `${selectedAttempt.label}+live-delta`;
          }
        }

        const sourceTime = mixer ? mixer.time : 0;
        const syncTime =
          clip.duration > 0
            ? ((sourceTime % clip.duration) + clip.duration) % clip.duration
            : 0;
        if (!liveRetarget) {
          for (const mix of modelMixers) {
            mix.setTime(syncTime);
          }
        }
        bodyLengthCalibration = null;
        armLengthCalibration = null;
        fingerLengthCalibration = null;
        let measureBodyErr = null;
        let bodyErrBaseline = null;
        if (!liveRetarget) {
          const bodyEvalCanonical =
            retargetStage === "body" ? RETARGET_BODY_CORE_CANONICAL : RETARGET_BODY_CANONICAL;
          const bodyTargetBones = retargetTargetBones.filter((b) =>
            bodyEvalCanonical.has(canonicalBoneKey(b.name) || "")
          );
          measureBodyErr = () => {
            const report = collectAlignmentDiagnostics({
              targetBones: bodyTargetBones,
              sourceBones: sourceResult.skeleton.bones,
              namesTargetToSource: names,
              sourceClip: clip,
              maxRows: 5,
              overlayYawOverride: 0,
              sourceOverlay,
              overlayUpAxis: _overlayUpAxis,
            });
            return Number.isFinite(report?.avgPosErrNorm) ? report.avgPosErrNorm : report?.avgPosErr;
          };
          const bodyErrBefore = measureBodyErr();
          if (Number.isFinite(bodyErrBefore)) {
            bodyErrBaseline = bodyErrBefore;
          }
          const attemptedBodyCalibration = buildBodyLengthCalibration(
            sourceResult.skeleton.bones,
            modelSkinnedMesh.skeleton.bones,
            clip,
            buildCanonicalBoneMap
          );
          const filteredBodyCalibration =
            attemptedBodyCalibration && retargetStage === "body"
              ? {
                  ...attemptedBodyCalibration,
                  entries: attemptedBodyCalibration.entries.filter((e) =>
                    RETARGET_BODY_CORE_CANONICAL.has(e.canonical)
                  ),
                }
              : attemptedBodyCalibration;
          const suspiciousBodyScale =
            !!filteredBodyCalibration &&
            Number.isFinite(filteredBodyCalibration.globalScale) &&
            (filteredBodyCalibration.globalScale < 0.2 || filteredBodyCalibration.globalScale > 5);
          if (filteredBodyCalibration?.entries?.length && !suspiciousBodyScale) {
            applyBoneLengthCalibration(filteredBodyCalibration, modelRoot);
            const bodyErrAfter = measureBodyErr();
            const hasBefore = Number.isFinite(bodyErrBefore);
            const hasAfter = Number.isFinite(bodyErrAfter);
            const keepCalibration =
              hasAfter && (!hasBefore || bodyErrAfter <= bodyErrBefore - 0.005);
            if (!keepCalibration) {
              resetBoneLengthCalibration(filteredBodyCalibration, modelRoot);
              bodyLengthCalibration = null;
            } else {
              bodyLengthCalibration = filteredBodyCalibration;
              if (Number.isFinite(bodyErrAfter)) {
                bodyErrBaseline = bodyErrAfter;
              }
            }
            diag("retarget-body-calibration", {
              stage: retargetStage,
              applied: keepCalibration,
              bodyErrBefore: Number.isFinite(bodyErrBefore) ? Number(bodyErrBefore.toFixed(5)) : null,
              bodyErrAfter: Number.isFinite(bodyErrAfter) ? Number(bodyErrAfter.toFixed(5)) : null,
              metric: retargetStage === "body" ? "body-core" : "body-full",
              bones: filteredBodyCalibration.entries.length,
              globalScale: filteredBodyCalibration.globalScale,
              clampedCount: filteredBodyCalibration.clampedCount,
              sample: filteredBodyCalibration.entries.slice(0, 8).map((e) => ({
                    canonical: e.canonical,
                    scale: e.scale,
                    rawScale: e.rawScale,
                    sourceLen: e.sourceLen,
                    targetLen: e.targetLen,
                    expectedTargetLen: e.expectedTargetLen,
                  })),
            });
          } else if (filteredBodyCalibration?.entries?.length) {
            diag("retarget-body-calibration", {
              stage: retargetStage,
              applied: false,
              skippedReason: "suspicious-global-scale",
              bodyErrBefore: Number.isFinite(bodyErrBefore) ? Number(bodyErrBefore.toFixed(5)) : null,
              bodyErrAfter: Number.isFinite(bodyErrBefore) ? Number(bodyErrBefore.toFixed(5)) : null,
              metric: retargetStage === "body" ? "body-core" : "body-full",
              bones: filteredBodyCalibration.entries.length,
              globalScale: filteredBodyCalibration.globalScale,
              clampedCount: filteredBodyCalibration.clampedCount,
              sample: filteredBodyCalibration.entries.slice(0, 8).map((e) => ({
                    canonical: e.canonical,
                    scale: e.scale,
                    rawScale: e.rawScale,
                    sourceLen: e.sourceLen,
                    targetLen: e.targetLen,
                    expectedTargetLen: e.expectedTargetLen,
                  })),
            });
          }
        }
        if (!liveRetarget && retargetStage === "body") {
          const armBaselineSnapshot = snapshotCanonicalBonePositions(
            modelSkinnedMesh.skeleton.bones,
            ARM_REFINEMENT_CANONICAL
          );
          const armRefine = buildArmRefinementCalibration({
            sourceBones: sourceResult.skeleton.bones,
            targetBones: modelSkinnedMesh.skeleton.bones,
            namesTargetToSource: names,
            sourceClip: clip,
            buildCanonicalBoneMap,
            collectAlignmentDiagnostics,
            updateWorld: () => modelRoot?.updateMatrixWorld(true),
          });
          const armErrBefore = Number.isFinite(bodyErrBaseline)
            ? bodyErrBaseline
            : (measureBodyErr ? measureBodyErr() : null);
          let armErrAfter = armErrBefore;
          let keepArmRefine = false;
          if (armRefine?.entries?.length) {
            armLengthCalibration = { entries: armRefine.entries };
            applyBoneLengthCalibration(armLengthCalibration, modelRoot);
            armErrAfter = measureBodyErr ? measureBodyErr() : null;
            const hasBefore = Number.isFinite(armErrBefore);
            const hasAfter = Number.isFinite(armErrAfter);
            keepArmRefine =
              hasAfter && (!hasBefore || armErrAfter <= armErrBefore - 0.003);
            if (!keepArmRefine) {
              restoreBonePositionSnapshot(armBaselineSnapshot, modelRoot);
              armLengthCalibration = null;
            } else {
              bodyErrBaseline = armErrAfter;
            }
          }
          diag("retarget-arm-refine", {
            stage: retargetStage,
            applied: keepArmRefine,
            bodyErrBefore: Number.isFinite(armErrBefore) ? Number(armErrBefore.toFixed(5)) : null,
            bodyErrAfter: Number.isFinite(armErrAfter) ? Number(armErrAfter.toFixed(5)) : null,
            appliedSides: (armRefine?.sides || []).filter((s) => s.applied).length,
            sides: armRefine?.sides || [],
            bones: armRefine?.entries?.length || 0,
          });
        }
        if (!liveRetarget && retargetStage === "full") {
          fingerLengthCalibration = buildFingerLengthCalibration(
            sourceResult.skeleton.bones,
            modelSkinnedMesh.skeleton.bones,
            clip,
            buildCanonicalBoneMap
          );
          if (fingerLengthCalibration) {
            applyBoneLengthCalibration(fingerLengthCalibration, modelRoot);
            diag("retarget-finger-calibration", {
              stage: retargetStage,
              bones: fingerLengthCalibration.entries.length,
              globalScale: fingerLengthCalibration.globalScale,
              clampedCount: fingerLengthCalibration.clampedCount,
              rawScaleRange: {
                min: fingerLengthCalibration.minRawScale,
                max: fingerLengthCalibration.maxRawScale,
              },
              sample: fingerLengthCalibration.entries.slice(0, 8).map((e) => ({
                canonical: e.canonical,
                scale: e.scale,
                rawScale: e.rawScale,
                sourceLen: e.sourceLen,
                targetLen: e.targetLen,
                expectedTargetLen: e.expectedTargetLen,
              })),
            });
          }
        }
        const hipsAlign = alignModelHipsToSource(false);
        if (hipsAlign) {
          diag("retarget-hips-align", { stage: retargetStage, ...hipsAlign });
        }
        const summaryCanonicalFilter = getCanonicalFilterForStage(retargetStage);
        const summaryTargetBones = retargetTargetBones.filter((bone) => {
          const canonical = canonicalBoneKey(bone.name) || "";
          if (!canonical) return false;
          return summaryCanonicalFilter ? summaryCanonicalFilter.has(canonical) : true;
        });
        const postRetargetReport = collectAlignmentDiagnostics({
          targetBones: summaryTargetBones.length ? summaryTargetBones : retargetTargetBones,
          sourceBones: sourceResult.skeleton.bones,
          namesTargetToSource: names,
          sourceClip: clip,
          maxRows: 5,
          overlayYawOverride: 0,
          sourceOverlay,
          overlayUpAxis: _overlayUpAxis,
        });
        const postRetargetPoseError =
          Number.isFinite(postRetargetReport?.avgPosErrNorm)
            ? postRetargetReport.avgPosErrNorm
            : postRetargetReport?.avgPosErr;
        const lowerBodyTargetBones = retargetTargetBones.filter((bone) =>
          RETARGET_BODY_CORE_CANONICAL.has(canonicalBoneKey(bone.name) || "")
        );
        const lowerBodyReport = collectAlignmentDiagnostics({
          targetBones: lowerBodyTargetBones.length ? lowerBodyTargetBones : retargetTargetBones,
          sourceBones: sourceResult.skeleton.bones,
          namesTargetToSource: names,
          sourceClip: clip,
          maxRows: 5,
          overlayYawOverride: 0,
          sourceOverlay,
          overlayUpAxis: _overlayUpAxis,
        });
        const lowerBodyPostError =
          Number.isFinite(lowerBodyReport?.avgPosErrNorm)
            ? lowerBodyReport.avgPosErrNorm
            : lowerBodyReport?.avgPosErr;
        const lowerBodyRotError =
          Number.isFinite(lowerBodyReport?.avgRotErrDeg)
            ? lowerBodyReport.avgRotErrDeg
            : null;

        isPlaying = true;
        updateTimelineUi(syncTime);
        const sourceTotal = sourceResult.skeleton.bones.length;
        const total = retargetTargetBones.length;
        const targetCoverage = canonicalCandidates > 0 ? matched / canonicalCandidates : 0;
        const sourceCoverage = sourceTotal > 0 ? sourceMatched / sourceTotal : 0;
        let lowMatch = "";
        if (targetCoverage < 0.75 || (targetCoverage < 0.9 && sourceCoverage < 0.35)) {
          lowMatch = " low humanoid match, try another model/rig.";
        }
        const candidateInfo = canonicalCandidates > 0 ? `, humanoid targets ${matched}/${canonicalCandidates}` : "";
        const activeMeshCount = liveRetarget ? liveRetarget.uniqueSkeletons.length : modelMixers.length;
        const limbDiag = collectLimbDiagnostics(
          retargetTargetBones,
          sourceResult.skeleton.bones,
          names,
          clip
        );
        setStatus(
          `Model retargeted [${retargetStage}] (source ${sourceMatched}/${sourceTotal}${candidateInfo}, all ${matched}/${total}, tracks ${clip.tracks.length}, mode ${selectedModeLabel}, active meshes ${activeMeshCount}).${lowMatch}`
        );
        const unmatched = unmatchedHumanoid.slice(0, 6);
        window.__vid2modelDebug = {
          stage: retargetStage,
          names,
          sourceBones: sourceResult.skeleton.bones.map((b) => b.name),
          targetBones: retargetTargetBones.map((b) => b.name),
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
                calibratedPairs: liveRetarget.calibratedPairs || 0,
                yawOffsetDeg: Number((liveRetarget.yawOffset * 180 / Math.PI).toFixed(2)),
              }
            : null,
          activeMeshCount,
        };
        if (Number.isFinite(postRetargetPoseError) && postRetargetPoseError <= 0.75) {
          rigProfileSaved = saveRigProfile({
            modelLabel,
            modelFingerprint: modelRigFingerprint,
            stage: retargetStage,
            namesTargetToSource: names,
            mode: selectedModeLabel,
            rootYawDeg: Number((rootYawCorrection * 180 / Math.PI).toFixed(2)),
            postPoseError: Number(postRetargetPoseError.toFixed(6)),
            liveRetarget: liveRetarget ? exportLiveRetargetProfile(liveRetarget) : null,
          });
        }
        const unmatchedTargetBones = retargetTargetBones
          .map((b) => b.name)
          .filter((name) => !names[name]);
        window.__vid2modelUnmatchedTargetBones = unmatchedTargetBones;
        diag("retarget-map-details", {
          stage: retargetStage,
          totalTargetBones: retargetTargetBones.length,
          mappedTargetBones: retargetTargetBones.length - unmatchedTargetBones.length,
          unmappedTargetBones: unmatchedTargetBones.length,
          sample: unmatchedTargetBones.slice(0, 20),
        });
        diag("retarget-result", {
          stage: retargetStage,
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
                calibratedPairs: liveRetarget.calibratedPairs || 0,
                yawOffsetDeg: Number((liveRetarget.yawOffset * 180 / Math.PI).toFixed(2)),
              }
            : null,
          unmatched,
        });
        diag("retarget-summary", {
          stage: retargetStage,
          mode: selectedModeLabel,
          mappedPairs,
          sourceMatched,
          sourceTotal,
          humanoidMatched: canonicalCandidates > 0 ? `${matched}/${canonicalCandidates}` : "n/a",
          tracks: clip.tracks.length,
          resolvedTracks: selectedAttempt.resolvedTracks,
          poseError: Number.isFinite(selectedPoseError) ? Number(selectedPoseError.toFixed(4)) : null,
          postPoseError: Number.isFinite(postRetargetPoseError) ? Number(postRetargetPoseError.toFixed(4)) : null,
          lowerBodyPostError: Number.isFinite(lowerBodyPostError) ? Number(lowerBodyPostError.toFixed(4)) : null,
          lowerBodyRotError: Number.isFinite(lowerBodyRotError) ? Number(lowerBodyRotError.toFixed(2)) : null,
          rootYawDeg: Number((rootYawCorrection * 180 / Math.PI).toFixed(2)),
          yawOffsetDeg: liveRetarget ? Number((liveRetarget.yawOffset * 180 / Math.PI).toFixed(2)) : 0,
          calibratedPairs: liveRetarget ? liveRetarget.calibratedPairs || 0 : 0,
          rigProfile: cachedRigProfile ? "hit" : "miss",
          rigProfileSaved,
          liveDelta: !!liveRetarget,
        });
        diag("retarget-limbs", {
          stage: retargetStage,
          mode: selectedModeLabel,
          issues: limbDiag.issuesCount,
          total: limbDiag.total,
          details: limbDiag.issues.slice(0, 12),
        });
        const alignment = dumpRetargetAlignmentDiagnostics({
          reason: "auto-retarget",
          modelSkinnedMesh,
          sourceResult,
          names: window.__vid2modelDebug?.names || {},
          sourceOverlay,
          overlayUpAxis: _overlayUpAxis,
          windowRef: window,
        });
        if (alignment) {
          diag("retarget-alignment", {
            stage: retargetStage,
            compared: alignment.totalCompared,
            avgPosErr: alignment.avgPosErr,
            avgRotErrDeg: alignment.avgRotErrDeg,
            hipsPosErr: alignment.hipsPosErr,
            overlayYawDeg: alignment.overlayYawDeg,
            worstPosition: alignment.worstPosition.slice(0, 5),
          });
        }
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
        createSourceAxesDebug(result.skeleton);
        for (const bone of result.skeleton.bones || []) {
          bone.userData.__bindPosition = bone.position.clone();
        }

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

    window.__vid2modelDumpAlignment = (reason = "manual") =>
      dumpRetargetAlignmentDiagnostics({
        reason,
        modelSkinnedMesh,
        sourceResult,
        names: window.__vid2modelDebug?.names || {},
        sourceOverlay,
        overlayUpAxis: _overlayUpAxis,
        windowRef: window,
      });
    // Retarget stages:
    //   "body" (torso+arms+legs, no fingers) for base alignment debug
    //   "full" (includes fingers)
    window.__vid2modelRetargetStage = "body";
    window.__vid2modelSetRetargetStage = (stage = "body", autoRetarget = true) => {
      const normalized = String(stage || "").trim().toLowerCase();
      const next = RETARGET_STAGES.has(normalized) ? normalized : "body";
      window.__vid2modelRetargetStage = next;
      console.log("[vid2model/diag] retarget-stage:", next);
      if (autoRetarget && sourceResult && modelSkinnedMesh) {
        applyBvhToModel();
      }
      return next;
    };
    // Browser console diagnostics:
    //   "minimal" prints only the main retarget decisions
    //   "verbose" restores the full diagnostic stream
    window.__vid2modelDiagMode = "minimal";
    window.__vid2modelSetDiagMode = (mode = "minimal") => {
      const next = String(mode || "minimal").trim().toLowerCase() === "verbose" ? "verbose" : "minimal";
      window.__vid2modelDiagMode = next;
      console.log("[vid2model/diag] console mode:", next);
      return next;
    };
    // Source skeleton labels:
    //   window.__vid2modelShowBoneLabels = true | false
    window.__vid2modelShowBoneLabels = true;
    // Optional manual override for troubleshooting:
    //   window.__vid2modelForceLiveDelta = true | false | null
    window.__vid2modelForceLiveDelta = null;
    // Experimental VRM direct retarget:
    //   window.__vid2modelUseVrmDirect = true | false
    window.__vid2modelUseVrmDirect = false;
    // Optional file logger for diagnostics:
    //   window.__vid2modelStartFileDiag() / window.__vid2modelStopFileDiag()
    // Logs are sent to a local endpoint and can be tailed from terminal.
    window.__vid2modelDiagFileLogger = { enabled: false, url: DIAG_FILE_LOG_DEFAULT_URL };
    window.__vid2modelStartFileDiag = (url = DIAG_FILE_LOG_DEFAULT_URL) => {
      window.__vid2modelDiagFileLogger = { enabled: true, url };
      console.log("[vid2model/diag] file logger enabled:", url);
    };
    window.__vid2modelStopFileDiag = () => {
      window.__vid2modelDiagFileLogger = { ...window.__vid2modelDiagFileLogger, enabled: false };
      console.log("[vid2model/diag] file logger disabled");
    };

    function applyParsedModel(gltf, label) {
      clearModel();
      const vrm = gltf.userData?.vrm || null;
      modelRoot = vrm?.scene || gltf.scene || gltf.scenes?.[0] || null;
      modelLabel = label || "";
      if (!modelRoot) {
        setStatus(`Failed to parse model: ${label}`);
        return;
      }
      if (vrm) {
        if (typeof VRMUtils.rotateVRM0 === "function") {
          VRMUtils.rotateVRM0(vrm);
        }
        if (typeof VRMUtils.combineSkeletons === "function") {
          VRMUtils.combineSkeletons(modelRoot);
        } else if (typeof VRMUtils.removeUnnecessaryJoints === "function") {
          VRMUtils.removeUnnecessaryJoints(modelRoot);
        }
        // Match the OSA viewer convention: VRM faces the camera-facing direction by default.
        modelRoot.rotation.y = Math.PI;
        modelRoot.updateMatrixWorld(true);
      }
      const vrmHumanoidInfo = applyVrmHumanoidBoneNames(vrm);
      modelVrmHumanoidBones = vrmHumanoidInfo.bones || [];
      modelVrmNormalizedHumanoidBones = vrmHumanoidInfo.normalizedBones || [];
      const vrmDirectBones =
        modelVrmNormalizedHumanoidBones.length > 0
          ? modelVrmNormalizedHumanoidBones
          : modelVrmHumanoidBones;
      scene.add(modelRoot);
      modelRoot.userData.__baseQuaternion = modelRoot.quaternion.clone();
      modelRoot.userData.__basePosition = modelRoot.position.clone();
      modelSkinnedMeshes = findSkinnedMeshes(modelRoot);
      modelSkinnedMesh = modelSkinnedMeshes[0] || null;
      if (!modelSkinnedMesh) {
        setStatus(`Model loaded, but no skinned mesh found: ${label}`);
        return;
      }
      const rootBone = getModelSkeletonRootBone();
      if (rootBone && rootBone !== modelRoot) {
        rootBone.userData.__retargetBaseQuaternion = rootBone.quaternion.clone();
        rootBone.userData.__retargetBasePosition = rootBone.position.clone();
      }
      const seenBones = new Set();
      for (const mesh of modelSkinnedMeshes) {
        for (const bone of mesh.skeleton?.bones || []) {
          const id = bone.uuid || `${mesh.uuid}:${bone.name}`;
          if (seenBones.has(id)) continue;
          seenBones.add(id);
          bone.userData.__bindPosition = bone.position.clone();
        }
      }
      logModelBones(modelSkinnedMeshes);
      let totalMissingBefore = 0;
      let totalAutoNamed = 0;
      let totalInferredCanonical = 0;
      for (const mesh of modelSkinnedMeshes) {
        const namingInfo = autoNameTargetBones(mesh);
        totalMissingBefore += namingInfo.missingBefore;
        totalAutoNamed += namingInfo.autoNamed;
        totalInferredCanonical += namingInfo.inferredCanonical;
      }
      modelRigFingerprint = buildModelRigFingerprint(modelSkinnedMeshes, modelLabel);
      diag("model-loaded", {
        file: label,
        skinnedMeshes: modelSkinnedMeshes.length,
        vrmHumanoid: vrmHumanoidInfo.applied
          ? {
              bones: vrmHumanoidInfo.applied,
              renamed: vrmHumanoidInfo.renamed,
              normalized: modelVrmNormalizedHumanoidBones.length,
            }
          : null,
        vrmDirectReady: vrmDirectBones.length > 0,
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
      if (vrm && modelRoot) {
        const token = modelDefaultAnimationToken;
        loadDefaultVrmAnimation(DEFAULT_VRM_ANIMATION_URL, vrm)
          .then((clip) => {
            if (!clip || token !== modelDefaultAnimationToken || !modelRoot) return;
            if (modelMixers.length || modelActions.length) return;
            const mix = new THREE.AnimationMixer(modelRoot);
            const action = mix.clipAction(clip);
            action.reset();
            action.setEffectiveWeight(1);
            action.setEffectiveTimeScale(1);
            action.play();
            modelMixer = mix;
            modelAction = action;
            modelMixers = [mix];
            modelActions = [action];
            isPlaying = true;
            updateTimelineUi(0);
            diag("model-default-animation", {
              file: label,
              clip: clip.name,
              duration: Number(clip.duration.toFixed(3)),
              url: DEFAULT_VRM_ANIMATION_URL,
            });
          })
          .catch((err) => {
            console.warn("Default VRM animation failed:", err);
          });
      }
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
      const defaultModelName = "MoonGirl.vrm";
      const url = new URL("../models/MoonGirl.vrm", import.meta.url).href;
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

    setupViewerUi({
      elements: {
        fileInput,
        modelInput,
        bvhFileNameEl,
        modelFileNameEl,
        btnLoadDefault,
        btnRetarget,
        btnRetargetFab,
        btnZoomIn,
        btnZoomOut,
        btnPlay,
        btnPause,
        btnStop,
        timeline,
        timeEl,
        btnResetCamera,
      },
      ops: {
        loadDefault,
        applyBvhToModel,
        zoomBy,
        loadBvhText,
        loadModelFile,
        setStatus,
        getActiveDuration,
        updateTimelineUi,
        setIsScrubbing: (next) => {
          isScrubbing = !!next;
        },
        setIsPlaying: (next) => {
          isPlaying = !!next;
        },
        getPlaybackRefs: () => ({
          mixer,
          modelMixers,
          currentAction,
          modelActions,
          liveRetarget,
          bodyLengthCalibration,
          armLengthCalibration,
          fingerLengthCalibration,
        }),
        applyLiveRetargetPose,
        applyBoneLengthCalibration: (plan) => applyBoneLengthCalibration(plan, modelRoot),
        applyFingerLengthCalibration: (plan) => applyFingerLengthCalibration(plan, modelRoot),
        alignModelHipsToSource,
        resetCamera: () => {
          camera.position.set(260, 200, 260);
          controls.target.set(0, 100, 0);
          controls.update();
          setStatus("Camera reset");
        },
      },
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
        if (bodyLengthCalibration && !liveRetarget) {
          applyBoneLengthCalibration(bodyLengthCalibration, modelRoot);
        }
        if (armLengthCalibration && !liveRetarget) {
          applyBoneLengthCalibration(armLengthCalibration, modelRoot);
        }
        if (fingerLengthCalibration && !liveRetarget) {
          applyFingerLengthCalibration(fingerLengthCalibration, modelRoot);
        }
        alignModelHipsToSource(false);
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
