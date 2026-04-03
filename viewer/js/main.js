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
    import { getBuiltinRigProfile } from "./modules/rig-profiles.js";
    import {
      attemptPriority,
      buildCanonicalBoneMap,
      buildRenamedClip,
      buildRetargetMap,
      scaleClipRotationsByCanonical,
      buildStageSourceClip,
      canonicalPoseSignature,
      collectLimbDiagnostics,
      resolvedTrackCountAcrossMeshes,
      resolvedTrackCountForTarget,
    } from "./modules/retarget-helpers.js";
    import {
      createViewerChainDiagnostics,
      getPreferredChildBone as getPreferredChildBoneModule,
      getPrimaryChildBone as getPrimaryChildBoneModule,
    } from "./modules/viewer-chain-diagnostics.js";
    import { createViewerBoneLabels } from "./modules/viewer-bone-labels.js";
    import { createViewerAlignmentTools } from "./modules/viewer-alignment.js";
    import { createViewerRuntimeDiagnostics } from "./modules/viewer-runtime-diagnostics.js";
    import { createViewerSourceAxesDebug } from "./modules/viewer-source-axes-debug.js";
    import {
      applyVrmHumanoidBoneNames,
      createViewerModelLoader,
      findSkinnedMeshes,
      logModelBones,
    } from "./modules/viewer-model-loader.js";
    import { createViewerParsedModelApplier } from "./modules/viewer-parsed-model.js";
    import {
      collectRetargetAttempts,
      selectRetargetAttempt,
    } from "./modules/viewer-retarget-attempts.js";
    import { createViewerSourceOverlay } from "./modules/viewer-source-overlay.js";
    import { createViewerSkeletonProfileTools } from "./modules/viewer-skeleton-profile.js";
    import { createViewerSkeletonDebugTools } from "./modules/viewer-skeleton-debug.js";
    import {
      autoNameTargetBones,
      maybeApplyTopologyFallback,
    } from "./modules/viewer-topology-fallback.js";
    import { setupViewerUi } from "./modules/ui-controls.js";

    const wrap = document.getElementById("canvas-wrap");
    const statusEl = document.getElementById("status");
    const fileInput = document.getElementById("file-input");
    const modelInput = document.getElementById("model-input");
    const bvhFileNameEl = document.getElementById("bvh-file-name");
    const modelFileNameEl = document.getElementById("model-file-name");
    const btnRetarget = document.getElementById("retarget");
    const btnValidateProfile = document.getElementById("validate-profile");
    const btnExportProfile = document.getElementById("export-profile");
    const btnImportProfile = document.getElementById("import-profile");
    const profileInput = document.getElementById("profile-input");
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
    const RIG_PROFILE_STATUS_VALUES = new Set(["draft", "validated"]);
    const REPO_RIG_PROFILE_MANIFEST_URL = new URL("../rig-profiles/index.json", import.meta.url).href;

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
    let latestRigProfileCandidate = null;
    let latestRigProfileState = {
      modelLabel: "",
      modelFingerprint: "",
      stage: "",
      source: "none",
      validationStatus: "none",
      hasProfile: false,
      saved: false,
      basedOnBuiltin: false,
      profileId: "",
    };
    let repoRigProfileManifest = null;
    let repoRigProfileManifestPromise = null;
    let repoRigProfiles = [];
    let isPlaying = false;
    let isScrubbing = false;
    const loader = new BVHLoader();
    const gltfLoader = new GLTFLoader();
    const clock = new THREE.Clock();
    gltfLoader.register((parser) => new VRMLoaderPlugin(parser));
    const { diag } = createDiag(window);
    const SKELETON_COLOR = 0xff2d55;
    const SOURCE_POINT_COLOR = 0xfff400;
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

    function augmentRetargetTargetBonesWithFallback(baseBones, fallbackBones, stage) {
      const result = [];
      const seen = new Set();
      for (const bone of baseBones || []) {
        if (!bone?.isBone) continue;
        const id = bone.uuid || bone.name;
        if (seen.has(id)) continue;
        seen.add(id);
        result.push(bone);
      }
      if (stage !== "body") return result;

      const requiredCanonicals = ["leftToes", "rightToes"];
      const presentCanonicals = new Set(result.map((bone) => canonicalBoneKey(bone.name)).filter(Boolean));
      for (const canonical of requiredCanonicals) {
        if (presentCanonicals.has(canonical)) continue;
        const fallbackBone =
          (fallbackBones || []).find((bone) => (canonicalBoneKey(bone.name) || "") === canonical) || null;
        if (!fallbackBone?.isBone) continue;
        const id = fallbackBone.uuid || fallbackBone.name;
        if (seen.has(id)) continue;
        seen.add(id);
        result.push(fallbackBone);
      }
      return result;
    }

    function getRetargetTargetBones(stage = getRetargetStage(), options = {}) {
      const fallback = modelSkinnedMesh?.skeleton?.bones || [];
      const preferNormalized = !!options.preferNormalized;
      const vrmBones =
        preferNormalized && modelVrmNormalizedHumanoidBones.length
          ? modelVrmNormalizedHumanoidBones
          : modelVrmHumanoidBones;
      if (!vrmBones.length) return fallback;
      const base = stage === "body"
        ? vrmBones.filter((bone) =>
            RETARGET_BODY_CANONICAL.has(canonicalBoneKey(bone.name) || "")
          )
        : vrmBones.slice();
      return augmentRetargetTargetBonesWithFallback(base, fallback, stage);
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
        return Array.isArray(parsed)
          ? parsed
              .map((entry) => normalizeRigProfileEntry(entry))
              .filter(Boolean)
          : [];
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

    function normalizeRigProfileEntry(entry) {
      if (!entry || typeof entry !== "object") return null;
      const modelFingerprint = String(entry.modelFingerprint || "").trim();
      const stage = String(entry.stage || "").trim().toLowerCase();
      if (!modelFingerprint || !stage) return null;
      const validationStatus = String(entry.validationStatus || "draft").trim().toLowerCase();
      return {
        ...entry,
        modelLabel: String(entry.modelLabel || ""),
        modelFingerprint,
        stage,
        source: String(entry.source || "localStorage"),
        validationStatus: RIG_PROFILE_STATUS_VALUES.has(validationStatus) ? validationStatus : "draft",
        autoSaved: entry.autoSaved !== false,
        updatedAt: entry.updatedAt || null,
        validatedAt: entry.validatedAt || null,
      };
    }

    function getRigProfilePriority(entry) {
      if (!entry) return -1;
      if (entry.validationStatus === "validated") return 2;
      if (entry.validationStatus === "draft") return 1;
      return 0;
    }

    function getRigProfileSourcePriority(entry) {
      const source = String(entry?.source || "").trim().toLowerCase();
      const isRepoSource = source === "repo-manifest" || source === "repo";
      if (entry?.validationStatus === "validated") {
        return isRepoSource ? 1 : 2;
      }
      return isRepoSource ? -1 : 0;
    }

    function compareRigProfiles(a, b) {
      const sourcePriorityDiff = getRigProfileSourcePriority(b) - getRigProfileSourcePriority(a);
      if (sourcePriorityDiff !== 0) return sourcePriorityDiff;
      const priorityDiff = getRigProfilePriority(b) - getRigProfilePriority(a);
      if (priorityDiff !== 0) return priorityDiff;
      const aErr = Number.isFinite(a?.postPoseError) ? a.postPoseError : Number.POSITIVE_INFINITY;
      const bErr = Number.isFinite(b?.postPoseError) ? b.postPoseError : Number.POSITIVE_INFINITY;
      if (aErr !== bErr) return aErr - bErr;
      const aTime = Date.parse(a?.validatedAt || a?.updatedAt || "") || 0;
      const bTime = Date.parse(b?.validatedAt || b?.updatedAt || "") || 0;
      return bTime - aTime;
    }

    function findStoredRigProfiles(modelFingerprint, stage) {
      const normalizedFingerprint = String(modelFingerprint || "").trim();
      const normalizedStage = String(stage || "").trim().toLowerCase();
      if (!normalizedFingerprint || !normalizedStage) return [];
      return readStoredRigProfiles()
        .filter((entry) => entry.modelFingerprint === normalizedFingerprint && entry.stage === normalizedStage)
        .sort(compareRigProfiles);
    }

    function findRepoRigProfiles(modelFingerprint, stage) {
      const normalizedFingerprint = String(modelFingerprint || "").trim();
      const normalizedStage = String(stage || "").trim().toLowerCase();
      if (!normalizedFingerprint || !normalizedStage) return [];
      return repoRigProfiles
        .filter((entry) => entry.modelFingerprint === normalizedFingerprint && entry.stage === normalizedStage)
        .sort(compareRigProfiles);
    }

    function normalizeRepoRigProfileManifest(payload) {
      if (!payload || typeof payload !== "object") return [];
      if (payload.format && payload.format !== "vid2model.rig-profile-manifest.v1") return [];
      const rows = Array.isArray(payload.profiles) ? payload.profiles : [];
      return rows
        .map((entry) => {
          const modelFingerprint = String(entry?.modelFingerprint || "").trim();
          const stage = String(entry?.stage || "").trim().toLowerCase();
          const path = String(entry?.path || "").trim();
          if (!modelFingerprint || !stage || !path) return null;
          return {
            modelFingerprint,
            modelLabel: String(entry?.modelLabel || ""),
            stage,
            path,
          };
        })
        .filter(Boolean);
    }

    async function ensureRepoRigProfileManifest() {
      if (repoRigProfileManifest) return repoRigProfileManifest;
      if (repoRigProfileManifestPromise) return repoRigProfileManifestPromise;
      repoRigProfileManifestPromise = fetch(REPO_RIG_PROFILE_MANIFEST_URL)
        .then(async (res) => {
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
          }
          const payload = await res.json();
          repoRigProfileManifest = normalizeRepoRigProfileManifest(payload);
          return repoRigProfileManifest;
        })
        .catch((err) => {
          console.warn("[vid2model/diag] rig-profile-manifest unavailable:", err?.message || err);
          repoRigProfileManifest = [];
          return repoRigProfileManifest;
        })
        .finally(() => {
          repoRigProfileManifestPromise = null;
        });
      return repoRigProfileManifestPromise;
    }

    async function loadRepoRigProfileEntry(entry) {
      const url = new URL(entry.path, REPO_RIG_PROFILE_MANIFEST_URL).href;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const payload = await res.json();
      const profile = normalizeRigProfileEntry(payload?.profile || payload);
      if (!profile) {
        throw new Error("Invalid rig profile payload");
      }
      const loaded = {
        ...profile,
        modelFingerprint: entry.modelFingerprint || profile.modelFingerprint,
        modelLabel: entry.modelLabel || payload?.modelLabel || profile.modelLabel || "",
        stage: entry.stage || profile.stage,
        source: "repo-manifest",
        validationStatus: "validated",
      };
      repoRigProfiles = [
        loaded,
        ...repoRigProfiles.filter(
          (item) => !(item.modelFingerprint === loaded.modelFingerprint && item.stage === loaded.stage)
        ),
      ];
      return loaded;
    }

    async function ensureRepoRigProfilesForModel({
      modelFingerprint = modelRigFingerprint,
      modelLabel: nextModelLabel = "",
    } = {}) {
      const manifest = await ensureRepoRigProfileManifest();
      const entries = manifest.filter((entry) => entry.modelFingerprint === modelFingerprint);
      if (!entries.length) return [];
      const localValidatedStages = new Set(
        readStoredRigProfiles()
          .filter((entry) => entry.modelFingerprint === modelFingerprint && entry.validationStatus === "validated")
          .map((entry) => entry.stage)
      );
      const loaded = [];
      for (const entry of entries) {
        if (localValidatedStages.has(entry.stage)) continue;
        try {
          const profile = await loadRepoRigProfileEntry(entry);
          loaded.push(profile);
        } catch (err) {
          console.warn("[vid2model/diag] rig-profile-manifest entry failed:", {
            modelFingerprint,
            stage: entry.stage,
            path: entry.path,
            error: String(err?.message || err),
          });
        }
      }
      if (loaded.length) {
        const activeProfile = loadRigProfile(modelFingerprint, getRetargetStage(), nextModelLabel);
        publishRigProfileState(buildRigProfileState(activeProfile, {
          modelFingerprint,
          modelLabel: nextModelLabel,
          stage: getRetargetStage(),
          saved: false,
        }));
      }
      return loaded;
    }

    function buildRigProfileState(profile, {
      modelFingerprint = "",
      modelLabel = "",
      stage = "",
      saved = false,
    } = {}) {
      return {
        modelLabel: String(modelLabel || profile?.modelLabel || ""),
        modelFingerprint: String(modelFingerprint || profile?.modelFingerprint || ""),
        stage: String(stage || profile?.stage || ""),
        source: String(profile?.source || "none"),
        validationStatus: String(profile?.validationStatus || "none"),
        hasProfile: !!profile,
        saved: !!saved,
        basedOnBuiltin: !!profile?.basedOnBuiltin,
        profileId: String(profile?.id || ""),
        updatedAt: profile?.updatedAt || null,
        validatedAt: profile?.validatedAt || null,
      };
    }

    function publishRigProfileState(state) {
      latestRigProfileState = {
        ...latestRigProfileState,
        ...(state || {}),
      };
      window.__vid2modelRigProfileState = { ...latestRigProfileState };
      return window.__vid2modelRigProfileState;
    }

    function loadRigProfile(modelFingerprint, stage, modelLabel = "") {
      if (!stage) return null;
      const builtin = getBuiltinRigProfile({ modelFingerprint, modelLabel, stage });
      if (builtin?.lockBuiltin) {
        return {
          ...builtin,
          validationStatus: builtin.validationStatus || "validated",
          source: builtin.source || "repo",
        };
      }
      const storedCandidates = modelFingerprint
        ? [
            ...findStoredRigProfiles(modelFingerprint, stage),
            ...findRepoRigProfiles(modelFingerprint, stage),
          ].sort(compareRigProfiles)
        : [];
      const stored = storedCandidates[0] || null;
      if (stored && builtin) {
        return {
          ...builtin,
          ...stored,
          namesTargetToSource: {
            ...(builtin.namesTargetToSource || {}),
            ...(stored.namesTargetToSource || {}),
          },
          liveRetarget: stored.liveRetarget ?? builtin.liveRetarget ?? null,
          source: stored.source || "localStorage",
          basedOnBuiltin: builtin.id || true,
        };
      }
      if (stored) {
        return { ...stored, source: stored.source || "localStorage" };
      }
      return builtin;
    }

    function saveRigProfile(entry, options = {}) {
      if (!entry?.modelFingerprint || !entry?.stage) return false;
      const validationStatus = String(options.validationStatus || entry.validationStatus || "draft").trim().toLowerCase();
      const normalizedStatus = RIG_PROFILE_STATUS_VALUES.has(validationStatus) ? validationStatus : "draft";
      const overwriteValidated = options.overwriteValidated === true;
      const storedProfiles = findStoredRigProfiles(entry.modelFingerprint, entry.stage);
      const existingValidated = storedProfiles.find((item) => item.validationStatus === "validated") || null;
      if (existingValidated && normalizedStatus !== "validated" && !overwriteValidated) {
        return false;
      }
      const existingSameStatus =
        storedProfiles.find((item) => item.validationStatus === normalizedStatus) || null;
      const existingError = existingSameStatus?.postPoseError;
      const nextError = entry?.postPoseError;
      if (
        Number.isFinite(existingError) &&
        Number.isFinite(nextError) &&
        existingError <= nextError + 1e-6
      ) {
        return false;
      }
      const rows = readStoredRigProfiles().filter(
        (item) => !(
          item?.modelFingerprint === entry.modelFingerprint &&
          item?.stage === entry.stage &&
          (
            item?.validationStatus === normalizedStatus ||
            (normalizedStatus === "validated" && item?.validationStatus === "draft")
          )
        )
      );
      const timestamp = new Date().toISOString();
      rows.unshift({
        ...entry,
        validationStatus: normalizedStatus,
        autoSaved: normalizedStatus !== "validated",
        updatedAt: timestamp,
        validatedAt: normalizedStatus === "validated" ? timestamp : entry?.validatedAt || null,
      });
      return writeStoredRigProfiles(rows.slice(0, MAX_RIG_PROFILE_ENTRIES));
    }

    function validateCurrentRigProfile() {
      const candidate = latestRigProfileCandidate;
      if (!candidate?.modelFingerprint || !candidate?.stage) {
        setStatus("No rig profile candidate yet. Retarget the model first.");
        return false;
      }
      const saved = saveRigProfile(
        {
          ...candidate,
          source: "localStorage",
        },
        {
          validationStatus: "validated",
          overwriteValidated: true,
        }
      );
      const validated = loadRigProfile(candidate.modelFingerprint, candidate.stage, candidate.modelLabel);
      publishRigProfileState(buildRigProfileState(validated, {
        modelFingerprint: candidate.modelFingerprint,
        modelLabel: candidate.modelLabel,
        stage: candidate.stage,
        saved,
      }));
      if (!saved) {
        setStatus(`Rig profile already validated for ${candidate.stage}.`);
        return false;
      }
      setStatus(`Rig profile validated for ${candidate.modelLabel || "model"} [${candidate.stage}].`);
      console.log("[vid2model/diag] rig-profile-validated", validated);
      return true;
    }

    function getBestPortableRigProfile({
      modelFingerprint = modelRigFingerprint,
      stage = getRetargetStage(),
      allowDraft = false,
    } = {}) {
      const profiles = findStoredRigProfiles(modelFingerprint, stage);
      const validated = profiles.find((entry) => entry.validationStatus === "validated") || null;
      if (validated) return validated;
      return allowDraft ? (profiles[0] || null) : null;
    }

    function buildPortableRigProfilePayload(profile) {
      if (!profile?.modelFingerprint || !profile?.stage) return null;
      return {
        format: "vid2model.rig-profile.v1",
        exportedAt: new Date().toISOString(),
        modelLabel: String(profile.modelLabel || ""),
        modelFingerprint: String(profile.modelFingerprint || ""),
        stage: String(profile.stage || ""),
        profile: {
          ...profile,
          validationStatus: "validated",
          source: "imported-json",
        },
      };
    }

    function buildSuggestedRigProfileDownloadName(payload, override = "") {
      if (override && String(override).trim()) {
        return String(override).trim();
      }
      return `${(payload.modelLabel || "model").replace(/\.[^.]+$/, "") || "model"}.${payload.stage}.rig-profile.json`;
    }

    function buildRegisterRigProfileCommand({
      inputPath = "",
      filename = "",
      allowDraft = false,
      payload = null,
    } = {}) {
      const activePayload = payload || window.__vid2modelRigProfileExport || buildPortableRigProfilePayload(
        getBestPortableRigProfile({ allowDraft })
      );
      if (!activePayload) return "";
      const suggestedFilename = buildSuggestedRigProfileDownloadName(activePayload, filename);
      const resolvedInput = String(inputPath || "").trim() || `/path/to/${suggestedFilename}`;
      return `python3 tools/register_rig_profile.py --input ${resolvedInput}`;
    }

    function exportCurrentRigProfile(download = false, filename = "", allowDraft = false) {
      const profile = getBestPortableRigProfile({ allowDraft });
      if (!profile) {
        setStatus(allowDraft ? "No rig profile available for export." : "No validated rig profile yet. Validate a profile first.");
        return null;
      }
      const payload = buildPortableRigProfilePayload(profile);
      if (!payload) {
        setStatus("Failed to build rig profile export.");
        return null;
      }
      const suggestedFilename = buildSuggestedRigProfileDownloadName(payload, filename);
      const registerCommand = buildRegisterRigProfileCommand({
        filename: suggestedFilename,
        allowDraft,
        payload,
      });
      payload.suggestedFilename = suggestedFilename;
      payload.registerCommand = registerCommand;
      window.__vid2modelRigProfileExport = payload;
      window.__vid2modelRigProfileExportCommand = registerCommand;
      if (download) {
        const blob = new Blob([JSON.stringify(payload, null, 2)], {
          type: "application/json",
        });
        const href = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = href;
        anchor.download = suggestedFilename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        setTimeout(() => URL.revokeObjectURL(href), 0);
      }
      setStatus(`Rig profile exported [${payload.stage}] (${payload.modelLabel || payload.modelFingerprint}).`);
      console.log("[vid2model/diag] rig-profile-export", payload);
      console.log("[vid2model/diag] rig-profile-register-command", registerCommand);
      return payload;
    }

    function importRigProfilePayload(payload, autoRetarget = true) {
      const format = String(payload?.format || "").trim();
      const profile = normalizeRigProfileEntry(payload?.profile || payload);
      if (!profile) {
        throw new Error("Invalid rig profile payload");
      }
      if (format && format !== "vid2model.rig-profile.v1") {
        throw new Error(`Unsupported rig profile format: ${format}`);
      }
      const saved = saveRigProfile(
        {
          ...profile,
          modelLabel: String(payload?.modelLabel || profile.modelLabel || ""),
          modelFingerprint: String(payload?.modelFingerprint || profile.modelFingerprint || ""),
          stage: String(payload?.stage || profile.stage || ""),
          source: "imported-json",
        },
        {
          validationStatus: "validated",
          overwriteValidated: true,
        }
      );
      const imported = loadRigProfile(
        String(payload?.modelFingerprint || profile.modelFingerprint || ""),
        String(payload?.stage || profile.stage || ""),
        String(payload?.modelLabel || profile.modelLabel || "")
      );
      publishRigProfileState(buildRigProfileState(imported, {
        modelFingerprint: imported?.modelFingerprint || profile.modelFingerprint,
        modelLabel: imported?.modelLabel || profile.modelLabel,
        stage: imported?.stage || profile.stage,
        saved,
      }));
      const matchesCurrentModel =
        imported &&
        imported.modelFingerprint === modelRigFingerprint &&
        imported.stage === getRetargetStage();
      setStatus(
        matchesCurrentModel
          ? `Rig profile imported and ready [${imported.stage}].`
          : `Rig profile imported for ${imported?.modelLabel || imported?.modelFingerprint || "model"}.`
      );
      console.log("[vid2model/diag] rig-profile-import", imported);
      if (matchesCurrentModel && autoRetarget && sourceResult && modelSkinnedMesh) {
        applyBvhToModel();
      }
      return imported;
    }

    async function importRigProfileFile() {
      if (!profileInput) {
        setStatus("Rig profile import is not available in this browser.");
        return null;
      }
      return new Promise((resolve) => {
        profileInput.value = "";
        profileInput.onchange = async (event) => {
          const file = event.target?.files?.[0] || null;
          if (!file) {
            resolve(null);
            return;
          }
          try {
            const text = await file.text();
            const payload = JSON.parse(text);
            const imported = importRigProfilePayload(payload, true);
            resolve(imported);
          } catch (err) {
            console.error(err);
            setStatus(`Rig profile import failed: ${String(err?.message || err)}`);
            resolve(null);
          } finally {
            profileInput.value = "";
            profileInput.onchange = null;
          }
        };
        profileInput.click();
      });
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

    function maybeSwapArmSidesByChain(baseResult, targetBones, sourceBones, canonicalFilter) {
      if (!baseResult?.names) return { ...baseResult, mirroredArmSidesApplied: false };
      const targetMap = buildCanonicalBoneMap(targetBones || []);
      const sourceMap = buildCanonicalBoneMap(sourceBones || []);
      const armCanonicals = [
        "leftShoulder",
        "leftUpperArm",
        "leftLowerArm",
        "leftHand",
        "rightShoulder",
        "rightUpperArm",
        "rightLowerArm",
        "rightHand",
      ];
      const fingerPrefixes = ["Thumb", "Index", "Middle", "Ring", "Little"];
      for (const side of ["left", "right"]) {
        for (const finger of fingerPrefixes) {
          armCanonicals.push(`${side}${finger}Metacarpal`);
          armCanonicals.push(`${side}${finger}Proximal`);
          armCanonicals.push(`${side}${finger}Intermediate`);
          armCanonicals.push(`${side}${finger}Distal`);
        }
      }

      let assignmentSameVotes = 0;
      let assignmentSwappedVotes = 0;
      let majorChainSwappedVotes = 0;
      let majorChainTotal = 0;
      for (const bone of targetBones || []) {
        const canonical = canonicalBoneKey(bone.name) || "";
        if (!canonical || !armCanonicals.includes(canonical)) continue;
        if (canonicalFilter && !canonicalFilter.has(canonical)) continue;
        const mappedSourceName = baseResult.names?.[bone.name] || "";
        const mappedCanonical = canonicalBoneKey(mappedSourceName) || "";
        if (!mappedCanonical) continue;
        if (
          (canonical.startsWith("left") && mappedCanonical.startsWith("left")) ||
          (canonical.startsWith("right") && mappedCanonical.startsWith("right"))
        ) {
          assignmentSameVotes += 1;
        } else if (
          (canonical.startsWith("left") && mappedCanonical.startsWith("right")) ||
          (canonical.startsWith("right") && mappedCanonical.startsWith("left"))
        ) {
          assignmentSwappedVotes += 1;
          if (
            canonical.endsWith("UpperArm") ||
            canonical.endsWith("LowerArm") ||
            canonical.endsWith("Hand")
          ) {
            majorChainSwappedVotes += 1;
          }
        }
        if (
          canonical.endsWith("UpperArm") ||
          canonical.endsWith("LowerArm") ||
          canonical.endsWith("Hand")
        ) {
          majorChainTotal += 1;
        }
      }

      const segmentAngle = (a0, a1, b0, b1) => {
        if (!a0 || !a1 || !b0 || !b1) return null;
        return angleBetweenWorldSegments(a0, a1, b0, b1);
      };
      const scoreArmPair = (targetSide, sourceSide) => {
        const targetUpper = targetMap.get(`${targetSide}UpperArm`) || null;
        const targetLower = targetMap.get(`${targetSide}LowerArm`) || null;
        const targetHand = targetMap.get(`${targetSide}Hand`) || null;
        const sourceUpper = sourceMap.get(`${sourceSide}UpperArm`) || null;
        const sourceLower = sourceMap.get(`${sourceSide}LowerArm`) || null;
        const sourceHand = sourceMap.get(`${sourceSide}Hand`) || null;
        if (!targetUpper || !targetLower || !targetHand || !sourceUpper || !sourceLower || !sourceHand) return null;

        const t0 = new THREE.Vector3();
        const t1 = new THREE.Vector3();
        const t2 = new THREE.Vector3();
        const s0 = new THREE.Vector3();
        const s1 = new THREE.Vector3();
        const s2 = new THREE.Vector3();
        targetUpper.getWorldPosition(t0);
        targetLower.getWorldPosition(t1);
        targetHand.getWorldPosition(t2);
        sourceUpper.getWorldPosition(s0);
        sourceLower.getWorldPosition(s1);
        sourceHand.getWorldPosition(s2);

        const angles = [
          segmentAngle(t0, t1, s0, s1),
          segmentAngle(t1, t2, s1, s2),
        ].filter((value) => Number.isFinite(value));
        if (!angles.length) return null;
        return angles.reduce((sum, value) => sum + value, 0) / angles.length;
      };

      const sameScores = [
        scoreArmPair("left", "left"),
        scoreArmPair("right", "right"),
      ].filter((value) => Number.isFinite(value));
      const swappedScores = [
        scoreArmPair("left", "right"),
        scoreArmPair("right", "left"),
      ].filter((value) => Number.isFinite(value));
      const sameAvg = sameScores.length
        ? sameScores.reduce((sum, value) => sum + value, 0) / sameScores.length
        : null;
      const swappedAvg = swappedScores.length
        ? swappedScores.reduce((sum, value) => sum + value, 0) / swappedScores.length
        : null;
      const shouldSwapByAssignments =
        assignmentSwappedVotes >= 4 && assignmentSwappedVotes > assignmentSameVotes;
      const shouldSwapByMajorChain =
        majorChainTotal >= 6 && majorChainSwappedVotes >= 4;
      const shouldSwapByGeometry =
        Number.isFinite(sameAvg) &&
        Number.isFinite(swappedAvg) &&
        swappedAvg + 15 < sameAvg;
      if (!shouldSwapByAssignments && !shouldSwapByMajorChain && !shouldSwapByGeometry) {
        return {
          ...baseResult,
          mirroredArmSidesApplied: false,
          armSideSwapScore: {
            assignmentSameVotes,
            assignmentSwappedVotes,
            majorChainSwappedVotes,
            majorChainTotal,
            sameAvg: Number.isFinite(sameAvg) ? Number(sameAvg.toFixed(3)) : null,
            swappedAvg: Number.isFinite(swappedAvg) ? Number(swappedAvg.toFixed(3)) : null,
          },
        };
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
        if (!armCanonicals.includes(canonical)) continue;
        if (canonicalFilter && !canonicalFilter.has(canonical)) continue;
        const swapped = swapCanonical(canonical);
        const sourceName = sourceByCanonical.get(swapped);
        if (sourceName) names[bone.name] = sourceName;
      }

      return {
        ...baseResult,
        names,
        mirroredArmSidesApplied: true,
        armSideSwapScore: {
          assignmentSameVotes,
          assignmentSwappedVotes,
          majorChainSwappedVotes,
          majorChainTotal,
          sameAvg: Number.isFinite(sameAvg) ? Number(sameAvg.toFixed(3)) : null,
          swappedAvg: Number.isFinite(swappedAvg) ? Number(swappedAvg.toFixed(3)) : null,
        },
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

    function isVerboseDiagMode() {
      return String(window.__vid2modelDiagMode || "minimal").trim().toLowerCase() === "verbose";
    }

    const chainDiagnostics = createViewerChainDiagnostics({
      canonicalBoneKey,
      buildCanonicalBoneMap,
      isVerboseDiagMode,
    });

    const {
      resetRestCorrectionLog,
      resetLegChainDiagLog,
      resetArmChainDiagLog,
      resetTorsoChainDiagLog,
      resetFootChainDiagLog,
      recordRestCorrectionLog,
      dumpRestCorrectionLog,
      buildLegChainDiagnostics,
      dumpLegChainDiagLog,
      buildArmChainDiagnostics,
      dumpArmChainDiagLog,
      buildTorsoChainDiagnostics,
      dumpTorsoChainDiagLog,
      buildFootChainDiagnostics,
      dumpFootChainDiagLog,
      dumpFootCorrectionDebug,
    } = chainDiagnostics;

    function getPreferredChildBone(bone) {
      return getPreferredChildBoneModule(bone, canonicalBoneKey);
    }

    function getPrimaryChildBone(bone) {
      return getPrimaryChildBoneModule(bone, canonicalBoneKey, _tmpWorldPosA, _tmpWorldPosB);
    }

    const alignmentTools = createViewerAlignmentTools({
      canonicalBoneKey,
      diag,
      camera,
      controls,
      getSkeletonObj: () => skeletonObj,
      getSourceResult: () => sourceResult,
      getModelRoot: () => modelRoot,
      getModelSkinnedMesh: () => modelSkinnedMesh,
      hasSourceOverlay: () => hasSourceOverlay(),
      setSourceOverlayYaw: (yaw) => setSourceOverlayYaw(yaw),
      updateSourceOverlay: () => updateSourceOverlay(),
      estimateFacingYawOffset,
    });

    const {
      objectHeight,
      fitToSkeleton,
      getModelSkeletonRootBone,
      syncSourceDisplayToModel,
      alignSourceHipsToModel,
      alignModelHipsToSource,
    } = alignmentTools;

    const skeletonProfileTools = createViewerSkeletonProfileTools({
      windowRef: window,
      buildCanonicalBoneMap,
      getModelSkinnedMesh: () => modelSkinnedMesh,
      getModelLabel: () => modelLabel,
      getModelRigFingerprint: () => modelRigFingerprint,
      vrmHumanoidBoneNames: VRM_HUMANOID_BONE_NAMES,
    });

    const {
      exportCurrentModelSkeletonProfile,
    } = skeletonProfileTools;

    const skeletonDebugTools = createViewerSkeletonDebugTools({
      canonicalBoneKey,
      buildCanonicalBoneMap,
      getPrimaryChildBone,
    });

    const {
      getSkeletonCanonicalRows,
      resolveSkeletonDumpCanonicals,
    } = skeletonDebugTools;

    const boneLabels = createViewerBoneLabels({
      scene,
      canonicalBoneKey,
      buildCanonicalBoneMap,
      resolveSkeletonDumpCanonicals,
      getRetargetStage,
      getRetargetTargetBones,
      getSourceBones: () => sourceResult?.skeleton?.bones || [],
      getDebugNames: () => window.__vid2modelDebug?.names || {},
      getModelRoot: () => modelRoot,
      objectHeight,
    });

    const {
      clearBoneLabels,
      refreshBoneLabels,
      updateBoneLabels,
    } = boneLabels;

    const sourceAxesDebugTools = createViewerSourceAxesDebug({
      scene,
      buildCanonicalBoneMap,
    });

    const {
      clearSourceAxesDebug,
      createSourceAxesDebug,
      updateSourceAxesDebug,
    } = sourceAxesDebugTools;

    const sourceOverlayTools = createViewerSourceOverlay({
      scene,
      clearSourceOverlayModule,
      createSourceOverlayModule,
      updateSourceOverlayModule,
      skeletonColor: SKELETON_COLOR,
      sourcePointColor: SOURCE_POINT_COLOR,
      overlayUpAxis: _overlayUpAxis,
      overlayPivot: _overlayPivot,
      refreshBoneLabels,
      updateSourceAxesDebug,
    });

    const {
      getSourceOverlay,
      hasSourceOverlay,
      clearSourceOverlay,
      updateSourceOverlay,
      createSourceOverlay,
      setSourceOverlayYaw,
    } = sourceOverlayTools;

    const runtimeDiagnostics = createViewerRuntimeDiagnostics({
      windowRef: window,
      diag,
      isVerboseDiagMode,
      collectLimbDiagnostics,
      dumpRetargetAlignmentDiagnostics,
      dumpRestCorrectionLog,
      buildLegChainDiagnostics,
      dumpLegChainDiagLog,
      buildArmChainDiagnostics,
      dumpArmChainDiagLog,
      buildTorsoChainDiagnostics,
      dumpTorsoChainDiagLog,
      buildFootChainDiagnostics,
      dumpFootChainDiagLog,
      dumpFootCorrectionDebug,
      getSkeletonCanonicalRows,
      resolveSkeletonDumpCanonicals,
      getRetargetTargetBones,
      getRetargetStage,
      getSourceOverlay,
      overlayUpAxis: _overlayUpAxis,
    });

    const {
      logModelBones: logRuntimeModelBones,
      publishRetargetDiagnostics,
      dumpAlignment: dumpRuntimeAlignment,
      dumpSkeleton: dumpRuntimeSkeleton,
    } = runtimeDiagnostics;

    function estimateFacingYawOffset(sourceBones, targetBones) {
      return estimateFacingYawOffsetModule(sourceBones, targetBones);
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
      clearBoneLabels("source");
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
      clearBoneLabels("target");
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

    function getPrimaryChildDirectionLocal(bone, outDir) {
      if (!bone || !outDir) return false;
      bone.getWorldPosition(_calibV1);
      const preferredChild = getPreferredChildBone(bone);
      if (preferredChild) {
        preferredChild.getWorldPosition(_calibV2);
        outDir.copy(_calibV2).sub(_calibV1);
        bone.getWorldQuaternion(_calibQ1);
        outDir.applyQuaternion(_calibQ1.invert());
        if (outDir.lengthSq() >= 1e-10) {
          outDir.normalize();
          return true;
        }
      }
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

    function buildBoneLocalFrame(bone, outQ, options = null) {
      if (!bone || !outQ) return false;
      if (!getPrimaryChildDirectionLocal(bone, _calibV1)) return false;
      if (!getReferenceDirectionLocal(bone, _calibV1, _calibV2)) return false;
      if (options?.flipReferenceAxis) {
        _calibV2.multiplyScalar(-1);
      }
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
      const canonical =
        canonicalBoneKey(targetBone?.name) ||
        canonicalBoneKey(sourceBone?.name) ||
        "";
      const restCorrectionProfile = loadRigProfile(modelRigFingerprint, getRetargetStage(), modelLabel);
      const flipReferenceAxis = !!(
        canonical &&
        restCorrectionProfile?.flipReferenceAxisByCanonical?.[canonical]
      );
      const twistDeg = canonical
        ? Number(restCorrectionProfile?.twistDegByCanonical?.[canonical] || 0)
        : 0;
      const offsetEntry = canonical
        ? restCorrectionProfile?.restCorrectionEulerDegByCanonical?.[canonical] || null
        : null;
      const profileOffset =
        offsetEntry && (offsetEntry.x || offsetEntry.y || offsetEntry.z)
          ? new THREE.Quaternion().setFromEuler(
              new THREE.Euler(
                THREE.MathUtils.degToRad(Number(offsetEntry.x || 0)),
                THREE.MathUtils.degToRad(Number(offsetEntry.y || 0)),
                THREE.MathUtils.degToRad(Number(offsetEntry.z || 0)),
                "XYZ"
              )
            ).normalize()
          : null;
      const profileOffsetDeg = profileOffset
        ? {
            x: Number(Number(offsetEntry.x || 0).toFixed(2)),
            y: Number(Number(offsetEntry.y || 0).toFixed(2)),
            z: Number(Number(offsetEntry.z || 0).toFixed(2)),
          }
        : null;
      let twistQ = null;
      if (Math.abs(twistDeg) > 1e-4) {
        const twistAxis = new THREE.Vector3();
        if (getPrimaryChildDirectionLocal(targetBone, twistAxis)) {
          twistQ = new THREE.Quaternion()
            .setFromAxisAngle(twistAxis, THREE.MathUtils.degToRad(twistDeg))
            .normalize();
        }
      }
      const sourceFrame = _calibQ1;
      const targetFrame = _calibQ2;
      if (
        buildBoneLocalFrame(sourceBone, sourceFrame) &&
        buildBoneLocalFrame(targetBone, targetFrame, { flipReferenceAxis })
      ) {
        const corr = new THREE.Quaternion().copy(targetFrame).multiply(sourceFrame.invert()).normalize();
        const autoAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
        if (profileOffset) {
          corr.premultiply(profileOffset).normalize();
        }
        if (twistQ) {
          corr.premultiply(twistQ).normalize();
        }
        const finalAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
        recordRestCorrectionLog({
          canonical: canonical || "unknown",
          target: targetBone?.name || "",
          source: sourceBone?.name || "",
          method: "local-frame",
          autoAngleDeg,
          profileOffsetDeg,
          flipReferenceAxis,
          twistDeg: Number(twistDeg.toFixed(2)),
          finalAngleDeg,
        });
        if (Math.abs(corr.w) > 0.9995) {
          return twistQ || profileOffset || null;
        }
        return corr;
      }
      const sourceDir = new THREE.Vector3();
      const targetDir = new THREE.Vector3();
      if (!getPrimaryChildDirectionLocal(sourceBone, sourceDir)) return null;
      if (!getPrimaryChildDirectionLocal(targetBone, targetDir)) return null;
      const dot = Math.max(-1, Math.min(1, sourceDir.dot(targetDir)));
      if (dot > 0.9995) {
        recordRestCorrectionLog({
          canonical: canonical || "unknown",
          target: targetBone?.name || "",
          source: sourceBone?.name || "",
          method: "child-direction",
          autoAngleDeg: 0,
          profileOffsetDeg,
          flipReferenceAxis,
          twistDeg: Number(twistDeg.toFixed(2)),
          finalAngleDeg: profileOffset
            ? Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(profileOffset.w)))).toFixed(3))
            : 0,
        });
        if (twistQ && profileOffset) {
          return new THREE.Quaternion().copy(twistQ).premultiply(profileOffset).normalize();
        }
        return twistQ || profileOffset || null;
      }
      const corr = new THREE.Quaternion().setFromUnitVectors(sourceDir, targetDir).normalize();
      const autoAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
      if (profileOffset) {
        corr.premultiply(profileOffset).normalize();
      }
      if (twistQ) {
        corr.premultiply(twistQ).normalize();
      }
      const finalAngleDeg = Number(THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(corr.w)))).toFixed(3));
      recordRestCorrectionLog({
        canonical: canonical || "unknown",
        target: targetBone?.name || "",
        source: sourceBone?.name || "",
        method: "child-direction",
        autoAngleDeg,
        profileOffsetDeg,
        flipReferenceAxis,
        twistDeg: Number(twistDeg.toFixed(2)),
        finalAngleDeg,
      });
      return corr;
    }

    function buildLiveRetargetPlan(skinnedMeshes, sourceBones, namesTargetToSource, targetBones = null) {
      const rigProfile = loadRigProfile(modelRigFingerprint, getRetargetStage(), modelLabel);
      return buildLiveRetargetPlanModule({
        skinnedMeshes,
        targetBones,
        sourceBones,
        namesTargetToSource,
        mixer,
        modelRoot,
        buildRestOrientationCorrection,
        cachedProfile: rigProfile?.liveRetarget || null,
        profile: rigProfile,
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
        resetRestCorrectionLog();
        resetLegChainDiagLog();
        resetArmChainDiagLog();
        resetTorsoChainDiagLog();
        resetFootChainDiagLog();
        const retargetStage = getRetargetStage();
        const cachedRigProfile = loadRigProfile(modelRigFingerprint, retargetStage, modelLabel);
        publishRigProfileState(buildRigProfileState(cachedRigProfile, {
          modelFingerprint: modelRigFingerprint,
          modelLabel,
          stage: retargetStage,
          saved: false,
        }));
        const profileBodyCanonicalKeys =
          retargetStage === "body" && Array.isArray(cachedRigProfile?.bodyCanonicalKeys)
            ? cachedRigProfile.bodyCanonicalKeys
            : null;
        const useBodyCoreRetarget =
          !profileBodyCanonicalKeys &&
          retargetStage === "body" &&
          String(cachedRigProfile?.bodyCanonicalMode || "").trim().toLowerCase() === "core";
        const canonicalFilter = profileBodyCanonicalKeys
          ? new Set(profileBodyCanonicalKeys.map((name) => String(name || "").trim()).filter(Boolean))
          : useBodyCoreRetarget
            ? RETARGET_BODY_CORE_CANONICAL
            : getCanonicalFilterForStage(retargetStage);
        const stageClip = buildStageSourceClip(
          sourceResult.clip,
          sourceResult.skeleton.bones,
          retargetStage,
          canonicalFilter
        );
        if (!stageClip) {
          setStatus(`Retarget failed: no source tracks for stage "${retargetStage}".`);
          diag("retarget-fail", { reason: "empty_stage_clip", stage: retargetStage });
          return;
        }
        const activeStageClip = scaleClipRotationsByCanonical(
          stageClip,
          sourceResult.skeleton.bones,
          cachedRigProfile?.rotationScaleByCanonical || null
        ) || stageClip;
        const retargetTargetBones = getRetargetTargetBones(retargetStage).filter((bone) =>
          canonicalFilter.has(canonicalBoneKey(bone.name) || "")
        );
        const normalizedDirectBones = getRetargetTargetBones(retargetStage, { preferNormalized: true }).filter((bone) =>
          canonicalFilter.has(canonicalBoneKey(bone.name) || "")
        );
        const directRetargetTargetBones = normalizedDirectBones.length ? normalizedDirectBones : retargetTargetBones;
        resetModelRootOrientation();
        const preferVrmDirectBody =
          retargetStage === "body" &&
          directRetargetTargetBones.length > 0 &&
          (shouldUseVrmDirectBody() || cachedRigProfile?.preferVrmDirectBody === true);
        const preferSkeletonOnRenameFallback = !!cachedRigProfile?.preferSkeletonOnRenameFallback;
        const preferAggressiveLiveDelta = !!cachedRigProfile?.preferAggressiveLiveDelta;
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
        const mirrorSwapMode = String(cachedRigProfile?.mirrorSwap || "").trim().toLowerCase();
        const allowMirrorSwap =
          mirrorSwapMode === "disable"
            ? false
            : mirrorSwapMode === "force" || mirrorSwapMode === "enable"
              ? true
              : true;
        const mirroredMapResult = allowMirrorSwap
          ? maybeSwapMirroredHumanoidSides(
              profiledMapResult,
              retargetTargetBones,
              sourceResult.skeleton.bones,
              canonicalFilter
            )
          : { ...profiledMapResult, mirroredSidesApplied: false };
        const armSideSwapMode = String(cachedRigProfile?.armSideSwap || "").trim().toLowerCase();
        const allowArmSideSwap =
          armSideSwapMode === "disable"
            ? false
            : armSideSwapMode === "force" || armSideSwapMode === "enable"
              ? true
              : true;
        const activeMapResult = allowArmSideSwap
          ? maybeSwapArmSidesByChain(
              mirroredMapResult,
              retargetTargetBones,
              sourceResult.skeleton.bones,
              canonicalFilter
            )
          : {
              ...mirroredMapResult,
              mirroredArmSidesApplied: false,
              armSideSwapScore: null,
            };
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
          sourceTracks: activeStageClip.tracks.length,
          mappedPairs,
          uniqueSourceMapped: sourceMatched,
          humanoidMatched: canonicalCandidates > 0 ? `${matched}/${canonicalCandidates}` : "n/a",
          mirrorSwap: allowMirrorSwap ? (mirrorSwapMode || "auto") : "disable",
          armSideSwap: allowArmSideSwap ? (armSideSwapMode || "auto") : "disable",
          mirroredSidesApplied: !!activeMapResult.mirroredSidesApplied,
          mirroredArmSidesApplied: !!activeMapResult.mirroredArmSidesApplied,
          armSideSwapScore: activeMapResult.armSideSwapScore || null,
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

        const {
          retargetAttempts,
          attemptDebug,
        } = collectRetargetAttempts({
          modelSkinnedMesh,
          modelSkinnedMeshes,
          modelRoot,
          sourceResult,
          activeStageClip,
          names,
          SkeletonUtils,
          buildRenamedClip,
          resolvedTrackCountAcrossMeshes,
          resolvedTrackCountForTarget,
          shortErr,
        });

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

        let {
          selectedAttempt,
          selectedBindings,
          selectedProbe,
          selectedPoseError,
          selectionDebug,
          preferredMode,
        } = selectRetargetAttempt({
          retargetAttempts,
          cachedRigProfile,
          preferSkeletonOnRenameFallback,
          modelSkinnedMeshes,
          modelRoot,
          modelSkinnedMesh,
          retargetTargetBones,
          sourceResult,
          mixer,
          buildBindingsForAttempt,
          clipUsesBonesSyntax,
          resolvedTrackCountForTarget,
          probeMotionForBindings,
          computePoseMatchError,
          buildCanonicalBoneMap,
          canonicalPoseSignature,
          attemptPriority,
        });

        if (!selectedBindings || !selectedBindings.mixers.length) {
          setStatus("Retarget failed: clip has no resolved tracks on model skeleton.");
          diag("retarget-fail", { reason: "no_resolved_tracks", stage: retargetStage });
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
            preferAggressiveLiveDelta &&
            (severeFacingPoseMismatch || (!fullHumanoidMatch && (highPoseError || strongFacingMismatch)))
          ) ||
          (!selectedIsSkeletonUtils && (strongFacingMismatch || weakMotion || highPoseError));
        const profileForceLiveDelta =
          typeof cachedRigProfile?.forceLiveDelta === "boolean" ? cachedRigProfile.forceLiveDelta : null;
        const forcedLiveDelta = getLiveDeltaOverride(window);
        const useLiveDelta =
          forcedLiveDelta === null
            ? (profileForceLiveDelta === null ? autoUseLiveDelta : profileForceLiveDelta)
            : forcedLiveDelta;
        diag("retarget-live-delta", {
          stage: retargetStage,
          selectedMode: selectedAttempt.label,
          selectedIsSkeletonUtils,
          profilePolicy: {
            preferSkeletonOnRenameFallback,
            preferAggressiveLiveDelta,
          },
          fullHumanoidMatch,
          preferredMode: preferredMode || null,
          autoUseLiveDelta,
          profileForced: profileForceLiveDelta,
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
            profile: cachedRigProfile,
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
          const profileRootYawDeg =
            Number.isFinite(cachedRigProfile?.rootYawDeg) ? cachedRigProfile.rootYawDeg : null;
          let zeroRow = null;
          let bestRow = null;
          let shouldUseBest = false;
          let yawEval = { rows: [], bestYaw: 0 };
          const rawYawLooksAligned = Math.abs(rawFacingYaw) < THREE.MathUtils.degToRad(30);
          if (Number.isFinite(profileRootYawDeg)) {
            rootYawCorrection = applyModelRootYaw(THREE.MathUtils.degToRad(profileRootYawDeg));
          } else {
            const yawCandidates = buildRootYawCandidates(rawFacingYaw, quantizeFacingYaw);
            yawEval = evaluateRootYawCandidates({
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
                  sourceOverlay: getSourceOverlay(),
                  overlayUpAxis: _overlayUpAxis,
                }),
            });
            zeroRow = yawEval.rows.find((r) => Math.abs(r.yawDeg) < 0.01) || null;
            bestRow = yawEval.rows[0] || null;
            const bestIsLargeFlip = !!bestRow && Math.abs(bestRow.yawDeg) > 120;
            shouldUseBest =
              !!bestRow &&
              !(rawYawLooksAligned && bestIsLargeFlip) &&
              (
                !zeroRow ||
                bestRow.score + 0.03 < zeroRow.score ||
                (Number.isFinite(bestRow.hipsPosErr) && Number.isFinite(zeroRow.hipsPosErr) && bestRow.hipsPosErr + 0.03 < zeroRow.hipsPosErr)
              );
            rootYawCorrection = applyModelRootYaw(shouldUseBest ? yawEval.bestYaw : 0);
          }
          if (hasSourceOverlay()) {
            setSourceOverlayYaw(0);
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
                  sourceOverlay: getSourceOverlay(),
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
              if (hasSourceOverlay()) {
                setSourceOverlayYaw(0);
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
            usedProfileYaw: Number.isFinite(profileRootYawDeg),
            zeroCandidate: zeroRow,
            bestCandidate: bestRow,
            candidates: yawEval.rows,
          });
        }
        const rebuildLiveRetargetPlan = () => {
          if (preferVrmDirectBody) {
            return buildVrmDirectBodyPlan({
              targetBones: directRetargetTargetBones,
              sourceBones: sourceResult.skeleton.bones,
              namesTargetToSource: names,
              mixer,
              modelRoot,
              buildRestOrientationCorrection,
              profile: cachedRigProfile,
            });
          }
          return buildLiveRetargetPlan(
            modelSkinnedMeshes,
            sourceResult.skeleton.bones,
            names,
            retargetTargetBones
          );
        };
        if (!liveRetarget && useLiveDelta) {
          const livePlan = rebuildLiveRetargetPlan();
          if (livePlan && livePlan.pairs.length > 0) {
            liveRetarget = livePlan;
            if (hasSourceOverlay()) {
              const effectiveOverlayYaw = rawFacingYaw + liveRetarget.yawOffset;
              setSourceOverlayYaw(effectiveOverlayYaw);
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
        if (liveRetarget) {
          const bodyEvalCanonical =
            retargetStage === "body" ? RETARGET_BODY_CORE_CANONICAL : RETARGET_BODY_CANONICAL;
          const bodyTargetBones = retargetTargetBones.filter((b) =>
            bodyEvalCanonical.has(canonicalBoneKey(b.name) || "")
          );
          measureBodyErr = () => {
            const report = collectAlignmentDiagnostics({
              targetBones: bodyTargetBones.length ? bodyTargetBones : retargetTargetBones,
              sourceBones: sourceResult.skeleton.bones,
              namesTargetToSource: names,
              sourceClip: clip,
              maxRows: 5,
              overlayYawOverride: 0,
              sourceOverlay: getSourceOverlay(),
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
            const previousLiveRetarget = liveRetarget;
            applyBoneLengthCalibration(filteredBodyCalibration, modelRoot);
            const rebuiltLiveRetarget = rebuildLiveRetargetPlan();
            if (rebuiltLiveRetarget?.pairs?.length) {
              liveRetarget = rebuiltLiveRetarget;
              if (hasSourceOverlay()) {
                const effectiveOverlayYaw = rawFacingYaw + liveRetarget.yawOffset;
                setSourceOverlayYaw(effectiveOverlayYaw);
                updateSourceOverlay();
              }
              applyLiveRetargetPose(liveRetarget);
            }
            const bodyErrAfter = rebuiltLiveRetarget?.pairs?.length ? measureBodyErr() : null;
            const hasBefore = Number.isFinite(bodyErrBefore);
            const hasAfter = Number.isFinite(bodyErrAfter);
            const keepCalibration =
              !!rebuiltLiveRetarget?.pairs?.length &&
              hasAfter &&
              (!hasBefore || bodyErrAfter <= bodyErrBefore - 0.005);
            if (!keepCalibration) {
              resetBoneLengthCalibration(filteredBodyCalibration, modelRoot);
              liveRetarget = previousLiveRetarget;
              if (hasSourceOverlay()) {
                const effectiveOverlayYaw = rawFacingYaw + liveRetarget.yawOffset;
                setSourceOverlayYaw(effectiveOverlayYaw);
                updateSourceOverlay();
              }
              applyLiveRetargetPose(liveRetarget);
              bodyLengthCalibration = null;
            } else {
              bodyLengthCalibration = filteredBodyCalibration;
              if (Number.isFinite(bodyErrAfter)) {
                bodyErrBaseline = bodyErrAfter;
              }
              selectedModeLabel = `${selectedModeLabel}+body-calib`;
            }
            diag("retarget-body-calibration", {
              stage: retargetStage,
              mode: "live-delta",
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
              mode: "live-delta",
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
              sourceOverlay: getSourceOverlay(),
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
          sourceOverlay: getSourceOverlay(),
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
          sourceOverlay: getSourceOverlay(),
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
        if (Number.isFinite(postRetargetPoseError) && postRetargetPoseError <= 0.75) {
          latestRigProfileCandidate = {
            modelLabel,
            modelFingerprint: modelRigFingerprint,
            stage: retargetStage,
            namesTargetToSource: names,
            mode: selectedModeLabel,
            rootYawDeg: Number((rootYawCorrection * 180 / Math.PI).toFixed(2)),
            postPoseError: Number(postRetargetPoseError.toFixed(6)),
            liveRetarget: liveRetarget ? exportLiveRetargetProfile(liveRetarget) : null,
            basedOnBuiltin: cachedRigProfile?.id || cachedRigProfile?.basedOnBuiltin || null,
          };
          rigProfileSaved = saveRigProfile({
            ...latestRigProfileCandidate,
            source: "localStorage",
          });
        }
        const activeProfile =
          loadRigProfile(modelRigFingerprint, retargetStage, modelLabel) || cachedRigProfile || null;
        const profileStateLabel = activeProfile?.validationStatus || (activeProfile ? "cached" : "none");
        publishRigProfileState(buildRigProfileState(activeProfile, {
          modelFingerprint: modelRigFingerprint,
          modelLabel,
          stage: retargetStage,
          saved: rigProfileSaved,
        }));
        setStatus(
          `Model retargeted [${retargetStage}] (source ${sourceMatched}/${sourceTotal}${candidateInfo}, all ${matched}/${total}, tracks ${clip.tracks.length}, mode ${selectedModeLabel}, active meshes ${activeMeshCount}, profile ${profileStateLabel}).${lowMatch}`
        );
        const unmatchedTargetBones = retargetTargetBones
          .map((b) => b.name)
          .filter((name) => !names[name]);
        publishRetargetDiagnostics({
          retargetStage,
          names,
          sourceBones: sourceResult.skeleton.bones,
          targetBones: retargetTargetBones,
          clip,
          selectedAttempt,
          selectedModeLabel,
          selectedProbe,
          selectedPoseError,
          liveRetarget,
          activeMeshCount,
          attemptDebug,
          selectionDebug,
          cachedRigProfile,
          rigProfileSaved,
          mappedPairs,
          sourceMatched,
          matched,
          canonicalCandidates,
          unmatchedHumanoid,
          unmatchedTargetBones,
          postRetargetPoseError,
          lowerBodyPostError,
          lowerBodyRotError,
          rootYawCorrection,
          modelSkinnedMesh,
          sourceResult,
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
      dumpRuntimeAlignment(reason, {
        modelSkinnedMesh,
        sourceResult,
        names: window.__vid2modelDebug?.names || {},
      });
    window.__vid2modelDumpSkeleton = (scope = "legs") =>
      dumpRuntimeSkeleton(scope, {
        sourceBones: sourceResult?.skeleton?.bones || [],
        targetBones: getRetargetTargetBones(getRetargetStage(), { preferNormalized: false }),
        names: window.__vid2modelDebug?.names || {},
      });
    window.__vid2modelBoneLabels = { enabled: false, which: "both", scope: "legs" };
    window.__vid2modelShowBoneLabels = (which = "both", scope = "legs") => {
      const normalizedWhich = ["source", "target", "both"].includes(String(which || "").trim().toLowerCase())
        ? String(which || "").trim().toLowerCase()
        : "both";
      const normalizedScope = String(scope || "legs").trim().toLowerCase();
      window.__vid2modelBoneLabels = {
        enabled: true,
        which: normalizedWhich,
        scope: normalizedScope,
      };
      refreshBoneLabels();
      console.log("[vid2model/diag] bone-labels", {
        enabled: true,
        which: normalizedWhich,
        scope: normalizedScope,
      });
      return window.__vid2modelBoneLabels;
    };
    window.__vid2modelHideBoneLabels = () => {
      window.__vid2modelBoneLabels = { ...window.__vid2modelBoneLabels, enabled: false };
      clearBoneLabels("both");
      console.log("[vid2model/diag] bone-labels", { enabled: false });
      return window.__vid2modelBoneLabels;
    };
    window.__vid2modelRefreshBoneLabels = () => {
      refreshBoneLabels();
      return window.__vid2modelBoneLabels;
    };
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

    const modelStateBridge = {
      get sourceResult() { return sourceResult; },
      get isPlaying() { return isPlaying; },
      set isPlaying(next) { isPlaying = !!next; },
      get modelRoot() { return modelRoot; },
      set modelRoot(next) { modelRoot = next; },
      get modelSkinnedMesh() { return modelSkinnedMesh; },
      set modelSkinnedMesh(next) { modelSkinnedMesh = next; },
      get modelSkinnedMeshes() { return modelSkinnedMeshes; },
      set modelSkinnedMeshes(next) { modelSkinnedMeshes = next; },
      get modelMixer() { return modelMixer; },
      set modelMixer(next) { modelMixer = next; },
      get modelAction() { return modelAction; },
      set modelAction(next) { modelAction = next; },
      get modelMixers() { return modelMixers; },
      set modelMixers(next) { modelMixers = next; },
      get modelActions() { return modelActions; },
      set modelActions(next) { modelActions = next; },
      get modelDefaultAnimationToken() { return modelDefaultAnimationToken; },
      get modelLabel() { return modelLabel; },
      set modelLabel(next) { modelLabel = next; },
      get modelRigFingerprint() { return modelRigFingerprint; },
      set modelRigFingerprint(next) { modelRigFingerprint = next; },
      get modelVrmHumanoidBones() { return modelVrmHumanoidBones; },
      set modelVrmHumanoidBones(next) { modelVrmHumanoidBones = next; },
      get modelVrmNormalizedHumanoidBones() { return modelVrmNormalizedHumanoidBones; },
      set modelVrmNormalizedHumanoidBones(next) { modelVrmNormalizedHumanoidBones = next; },
    };

    const applyParsedModel = createViewerParsedModelApplier({
      state: modelStateBridge,
      clearModel,
      scene,
      setStatus,
      diag,
      VRMUtils,
      applyVrmHumanoidBoneNames,
      vrmHumanoidBoneNames: VRM_HUMANOID_BONE_NAMES,
      findSkinnedMeshes,
      getModelSkeletonRootBone,
      logModelBones,
      collectModelBoneRows,
      logRuntimeModelBones,
      autoNameTargetBones,
      buildModelRigFingerprint,
      syncSourceDisplayToModel,
      fitToSkeleton,
      refreshBoneLabels,
      loadDefaultVrmAnimation,
      defaultVrmAnimationUrl: DEFAULT_VRM_ANIMATION_URL,
      updateTimelineUi,
      applyBvhToModel,
      ensureRepoRigProfilesForModel,
    });

    window.__vid2modelExportSkeletonProfile = (download = false, filename = "") =>
      exportCurrentModelSkeletonProfile(download, filename);
    window.__vid2modelExportRigProfile = (download = false, filename = "", allowDraft = false) =>
      exportCurrentRigProfile(download, filename, allowDraft);
    window.__vid2modelImportRigProfile = (payload, autoRetarget = true) =>
      importRigProfilePayload(payload, autoRetarget);
    window.__vid2modelBuildRegisterRigProfileCommand = (inputPath = "", allowDraft = false) =>
      buildRegisterRigProfileCommand({ inputPath, allowDraft });
    window.__vid2modelListRepoRigProfiles = (fingerprint = modelRigFingerprint, stage = "") =>
      (stage
        ? findRepoRigProfiles(fingerprint, stage)
        : repoRigProfiles.filter((entry) => !fingerprint || entry.modelFingerprint === fingerprint));
    window.__vid2modelGetRigProfileState = () => ({ ...latestRigProfileState });
    window.__vid2modelValidateRigProfile = () => validateCurrentRigProfile();
    window.__vid2modelListRigProfiles = (fingerprint = modelRigFingerprint, stage = "") =>
      (stage
        ? findStoredRigProfiles(fingerprint, stage)
        : readStoredRigProfiles().filter((entry) => !fingerprint || entry.modelFingerprint === fingerprint));

    const modelLoader = createViewerModelLoader({
      gltfLoader,
      setStatus,
      onParsedModel: applyParsedModel,
      setModelFileNameText: (text) => {
        modelFileNameEl.textContent = text;
      },
      defaultModelName: "6493143135142452442.glb",
      importMetaUrl: import.meta.url,
    });

    const {
      loadModelBuffer,
      loadModelFile,
      loadDefaultModel,
    } = modelLoader;

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
        btnValidateProfile,
        btnExportProfile,
        btnImportProfile,
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
        validateCurrentRigProfile,
        exportCurrentRigProfile,
        importRigProfileFile,
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

        if (hasSourceOverlay()) {
          updateSourceOverlay();
        }
        updateBoneLabels();

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
