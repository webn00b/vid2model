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
      summarizeSourceRootYawClip,
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
      resolveBodyMetricCanonicalFilter,
      resolveRetargetStageCanonicalFilter,
    } from "./modules/retarget-stage-contract.js";
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
    import { computeBvhGroundY } from "./modules/viewer-alignment.js";
    import { createViewerRuntimeDiagnostics } from "./modules/viewer-runtime-diagnostics.js";
    import { createViewerSourceAxesDebug } from "./modules/viewer-source-axes-debug.js";
    import {
      applyVrmHumanoidBoneNames,
      createViewerModelLoader,
      findSkinnedMeshes,
      logModelBones,
    } from "./modules/viewer-model-loader.js";
    import { createViewerParsedModelApplier } from "./modules/viewer-parsed-model.js";
    import { createViewerController } from "./modules/viewer-controller.js";
    import { createViewerRigProfileService } from "./modules/viewer-rig-profile-service.js";
    import {
      collectRetargetAttempts,
      selectRetargetAttempt,
    } from "./modules/viewer-retarget-attempts.js";
    import { createViewerSourceOverlay } from "./modules/viewer-source-overlay.js";
    import { createViewerSkeletonProfileTools } from "./modules/viewer-skeleton-profile.js";
    import { createViewerModelAnalysisTools } from "./modules/viewer-model-analysis.js";
    import { createViewerSkeletonDebugTools } from "./modules/viewer-skeleton-debug.js";
    import {
      autoNameTargetBones,
      maybeApplyTopologyFallback,
    } from "./modules/viewer-topology-fallback.js";
    import { setupViewerUi } from "./modules/ui-controls.js";
    import { runRetargetPipeline } from "./modules/viewer-retarget-pipeline.js";
    import { createViewerBoneFrameMath } from "./modules/viewer-bone-frame-math.js";
    import { createViewerFileIo } from "./modules/viewer-file-io.js";
    import {
      applyRigProfileNames,
      maybeSwapMirroredHumanoidSides,
      maybeSwapArmSidesByChain,
    } from "./modules/viewer-rig-profile-application.js";

    const wrap = document.getElementById("canvas-wrap");
    const statusEl = document.getElementById("status");
    const fileInput = document.getElementById("file-input");
    const modelInput = document.getElementById("model-input");
    const bvhFileNameEl = document.getElementById("bvh-file-name");
    const modelFileNameEl = document.getElementById("model-file-name");
    const btnAutoSetup = document.getElementById("auto-setup");
    const btnSaveModelSetup = document.getElementById("save-model-setup");
    const btnRetarget = document.getElementById("retarget");
    const btnValidateProfile = document.getElementById("validate-profile");
    const btnExportProfile = document.getElementById("export-profile");
    const btnExportModelAnalysis = document.getElementById("export-model-analysis");
    const btnImportProfile = document.getElementById("import-profile");
    const profileInput = document.getElementById("profile-input");
    const btnRetargetFab = document.getElementById("retarget-fab");
    const btnLoadDefault = document.getElementById("load-default");
const btnPlayToggle = document.getElementById("play-toggle");
    const btnStop = document.getElementById("stop");
    const btnToggleSkeleton = document.getElementById("toggle-skeleton");
    const btnToggleModel = document.getElementById("toggle-model");
    const btnDarkToggle = document.getElementById("dark-toggle");
    const btnToolsToggle = document.getElementById("tools-toggle");
    const toolsGroup = document.getElementById("tools-group");
    const timeline = document.getElementById("timeline");
    const timeEl = document.getElementById("time");
    const btnResetCamera = document.getElementById("reset-camera");
    const animationList = document.getElementById("animation-list");
    const RIG_PROFILE_STORAGE_KEY = "vid2model.rigProfiles.v16";
    const MAX_RIG_PROFILE_ENTRIES = 12;
    const RIG_PROFILE_STATUS_VALUES = new Set(["draft", "validated"]);
    const REPO_RIG_PROFILE_MANIFEST_URL = new URL("../rig-profiles/index.json", import.meta.url).href;

    const scene = new THREE.Scene();
    const _isDarkOnLoad = localStorage.getItem("vid2model.darkMode") === "1";
    scene.background = new THREE.Color(_isDarkOnLoad ? 0x0d1117 : 0xf8fbff);

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
    let isPlaying = false;
    let isScrubbing = false;
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

    // Unified state container for retarget pipeline
    const viewerState = {
      get sourceResult() { return sourceResult; },
      set sourceResult(v) { sourceResult = v; },
      get modelSkinnedMesh() { return modelSkinnedMesh; },
      set modelSkinnedMesh(v) { modelSkinnedMesh = v; },
      get modelSkinnedMeshes() { return modelSkinnedMeshes; },
      set modelSkinnedMeshes(v) { modelSkinnedMeshes = v; },
      get modelRoot() { return modelRoot; },
      set modelRoot(v) { modelRoot = v; },
      get modelMixer() { return modelMixer; },
      set modelMixer(v) { modelMixer = v; },
      get modelAction() { return modelAction; },
      set modelAction(v) { modelAction = v; },
      get modelMixers() { return modelMixers; },
      set modelMixers(v) { modelMixers = v; },
      get modelActions() { return modelActions; },
      set modelActions(v) { modelActions = v; },
      get modelLabel() { return modelLabel; },
      set modelLabel(v) { modelLabel = v; },
      get modelRigFingerprint() { return modelRigFingerprint; },
      set modelRigFingerprint(v) { modelRigFingerprint = v; },
      get modelVrmHumanoidBones() { return modelVrmHumanoidBones; },
      set modelVrmHumanoidBones(v) { modelVrmHumanoidBones = v; },
      get modelVrmNormalizedHumanoidBones() { return modelVrmNormalizedHumanoidBones; },
      set modelVrmNormalizedHumanoidBones(v) { modelVrmNormalizedHumanoidBones = v; },
      get liveRetarget() { return liveRetarget; },
      set liveRetarget(v) { liveRetarget = v; },
      get bodyLengthCalibration() { return bodyLengthCalibration; },
      set bodyLengthCalibration(v) { bodyLengthCalibration = v; },
      get armLengthCalibration() { return armLengthCalibration; },
      set armLengthCalibration(v) { armLengthCalibration = v; },
      get fingerLengthCalibration() { return fingerLengthCalibration; },
      set fingerLengthCalibration(v) { fingerLengthCalibration = v; },
      get isPlaying() { return isPlaying; },
      set isPlaying(v) { isPlaying = v; },
      get mixer() { return mixer; },
      set mixer(v) { mixer = v; },
      _liveAxisY,
      _liveYawQ,
      _liveQ2,
      _liveQ3,
      _rootYawQ,
      _overlayUpAxis,
    };
    const loader = new BVHLoader();
    const gltfLoader = new GLTFLoader();
    const clock = new THREE.Clock();
    gltfLoader.register((parser) => new VRMLoaderPlugin(parser));
    const { diag } = createDiag(window);
    const SKELETON_COLOR = 0xff2d55;
    const SOURCE_POINT_COLOR = 0xfff400;

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

    let modelAnalysisTools = null;

    function buildRigProfileSeedForCurrentModel(stage = "body") {
      return modelAnalysisTools?.buildRigProfileSeedForCurrentModel(stage) || null;
    }

    function buildSeedCorrectionSummary(profile) {
      return modelAnalysisTools?.buildSeedCorrectionSummary(profile) || [];
    }

    const rigProfileService = createViewerRigProfileService({
      windowRef: window,
      storageKey: RIG_PROFILE_STORAGE_KEY,
      maxEntries: MAX_RIG_PROFILE_ENTRIES,
      statusValues: RIG_PROFILE_STATUS_VALUES,
      repoManifestUrl: REPO_RIG_PROFILE_MANIFEST_URL,
      getBuiltinRigProfile,
      getRetargetStage,
      getCurrentModelRigFingerprint: () => modelRigFingerprint,
      buildRigProfileSeedForCurrentModel,
      buildSeedCorrectionSummary,
    });

    const {
      readStoredRigProfiles,
      findStoredRigProfiles,
      findRepoRigProfiles,
      ensureRepoRigProfilesForModel,
      buildRigProfileState,
      publishRigProfileState,
      getLatestRigProfileState,
      setLatestRigProfileCandidate,
      loadRigProfile,
      saveRigProfile,
      validateCurrentRigProfile: validateRigProfileWithService,
      buildRegisterRigProfileCommand,
      exportCurrentRigProfile: exportRigProfileWithService,
      importRigProfilePayload: importRigProfilePayloadWithService,
      listRepoRigProfiles,
      listRigProfiles,
    } = rigProfileService;

    function validateCurrentRigProfile() {
      return validateRigProfileWithService({ setStatus });
    }

    function autoSetupModel() {
      if (!sourceResult) {
        setStatus("Load BVH first, then click Auto Setup.");
        return false;
      }
      if (!modelSkinnedMesh) {
        setStatus("Load model (.glb/.vrm) first, then click Auto Setup.");
        return false;
      }
      applyBvhToModel();
      return true;
    }

    function saveModelSetup() {
      const activeStage = getRetargetStage();
      const activeProfile = modelRigFingerprint
        ? loadRigProfile(modelRigFingerprint, activeStage, modelLabel)
        : null;
      if (activeProfile?.validationStatus === "validated") {
        const sourceLabel =
          activeProfile.source === "repo"
            ? "shared profile"
            : activeProfile.source === "localStorage"
              ? "saved local profile"
              : "validated profile";
        publishRigProfileState(buildRigProfileState(activeProfile, {
          modelFingerprint: modelRigFingerprint,
          modelLabel,
          stage: activeStage,
          saved: true,
        }));
        setStatus(`Model setup is already ready (${sourceLabel}) [${activeStage}].`);
        return true;
      }
      if (validateCurrentRigProfile()) {
        setStatus(`Model setup saved for ${modelLabel || "model"} [${activeStage}].`);
        return true;
      }
      return false;
    }

    function exportCurrentRigProfile(download = false, filename = "", allowDraft = false) {
      return exportRigProfileWithService(download, filename, allowDraft, { setStatus });
    }

    function importRigProfilePayload(payload, autoRetarget = true) {
      return importRigProfilePayloadWithService(payload, {
        modelRigFingerprint,
        autoRetarget,
        sourceResult,
        modelSkinnedMesh,
        setStatus,
        onAutoRetarget: () => applyBvhToModel(),
      });
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

    function getCanonicalFilterForStage(stage, profile = null) {
      return resolveRetargetStageCanonicalFilter(stage, profile);
    }

    function getBodyMetricCanonicalFilter(stage, profile = null) {
      return resolveBodyMetricCanonicalFilter(stage, profile);
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

    const statusTextEl = document.getElementById("status-text");
    function setStatus(text) {
      const busy = text.startsWith("Loading");
      statusEl.classList.toggle("busy", busy);
      if (statusTextEl) statusTextEl.textContent = text;
      else statusEl.textContent = text;
    }

    function setSceneBg(theme) {
      scene.background = new THREE.Color(theme === "dark" ? 0x0d1117 : 0xf8fbff);
    }

    const VIS_SKELETON_KEY = "vid2model.showSkeleton";
    const VIS_MODEL_KEY = "vid2model.showModel";

    function _saveVisibility(key, visible) {
      localStorage.setItem(key, visible ? "1" : "0");
      const params = new URLSearchParams(location.search);
      params.set(key.replace("vid2model.", ""), visible ? "1" : "0");
      history.replaceState(null, "", "?" + params.toString());
    }

    function _loadVisibility(key) {
      const param = new URLSearchParams(location.search).get(key.replace("vid2model.", ""));
      if (param !== null) return param !== "0";
      const stored = localStorage.getItem(key);
      return stored !== "0"; // default visible
    }

    function toggleSkeleton() {
      if (!skeletonObj) return _loadVisibility(VIS_SKELETON_KEY);
      skeletonObj.visible = !skeletonObj.visible;
      _saveVisibility(VIS_SKELETON_KEY, skeletonObj.visible);
      return skeletonObj.visible;
    }

    function toggleModel() {
      if (!modelRoot) return _loadVisibility(VIS_MODEL_KEY);
      modelRoot.visible = !modelRoot.visible;
      _saveVisibility(VIS_MODEL_KEY, modelRoot.visible);
      return modelRoot.visible;
    }

    function applySkeletonVisibility() {
      if (skeletonObj) skeletonObj.visible = _loadVisibility(VIS_SKELETON_KEY);
    }

    function applyModelVisibility() {
      if (modelRoot) modelRoot.visible = _loadVisibility(VIS_MODEL_KEY);
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

    modelAnalysisTools = createViewerModelAnalysisTools({
      windowRef: window,
      buildCanonicalBoneMap,
      canonicalBoneKey,
      getModelSkinnedMeshes: () => modelSkinnedMeshes,
      getModelLabel: () => modelLabel,
      getModelRigFingerprint: () => modelRigFingerprint,
      getModelSkeletonRootBone,
      getModelVrmHumanoidBones: () => modelVrmHumanoidBones,
      getModelVrmNormalizedHumanoidBones: () => modelVrmNormalizedHumanoidBones,
    });

    const {
      exportCurrentModelAnalysis,
    } = modelAnalysisTools;

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

    const boneFrameMath = createViewerBoneFrameMath({
      getPreferredChildBone,
      loadRigProfile,
      getRetargetStage,
      getModelRigFingerprint: () => modelRigFingerprint,
      getModelLabel: () => modelLabel,
      recordRestCorrectionLog,
    });

    const { buildRestOrientationCorrection } = boneFrameMath;

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
        const ctx = {
          sourceResult,
          modelSkinnedMesh,
          modelSkinnedMeshes,
          modelRoot,
          modelLabel,
          modelRigFingerprint,
          mixer,
          liveRetarget,
          modelMixers,
          modelActions,
          modelMixer,
          modelAction,
          bodyLengthCalibration,
          armLengthCalibration,
          fingerLengthCalibration,
          overlayUpAxis: _overlayUpAxis,
          SkeletonUtils,
          diag,
          setStatus,
          getRetargetStage,
          loadRigProfile,
          publishRigProfileState,
          buildRigProfileState,
          getCanonicalFilterForStage,
          getBodyMetricCanonicalFilter,
          scaleClipRotationsByCanonicalFn: scaleClipRotationsByCanonical,
          getRetargetTargetBones,
          resetModelRootOrientation,
          applyModelRootYaw,
          shouldUseVrmDirectBody,
          maybeApplyTopologyFallback,
          applyRigProfileNames,
          maybeSwapMirroredHumanoidSides,
          maybeSwapArmSidesByChain,
          buildRenamedClip,
          resolvedTrackCountAcrossMeshes,
          resolvedTrackCountForTarget,
          shortErr,
          selectRetargetAttemptFn: selectRetargetAttempt,
          buildBindingsForAttempt,
          clipUsesBonesSyntax,
          probeMotionForBindingsFn: probeMotionForBindings,
          computePoseMatchErrorFn: computePoseMatchError,
          buildCanonicalBoneMapFn: buildCanonicalBoneMap,
          canonicalPoseSignatureFn: canonicalPoseSignature,
          attemptPriorityFn: attemptPriority,
          estimateFacingYawOffset,
          quantizeFacingYaw,
          evaluateRootYawCandidates,
          buildLiveRetargetPlan,
          applyLiveRetargetPose,
          buildRestOrientationCorrection,
          hasSourceOverlay,
          setSourceOverlayYaw,
          updateSourceOverlay,
          getSourceOverlay,
          alignModelHipsToSource,
          updateTimelineUi,
          setLatestRigProfileCandidate,
          saveRigProfile,
          publishRetargetDiagnostics,
          syncSourceDisplayToModel,
          buildSeedCorrectionSummary,
          exportLiveRetargetProfile,
          resetRestCorrectionLog,
          resetLegChainDiagLog,
          resetArmChainDiagLog,
          resetTorsoChainDiagLog,
          resetFootChainDiagLog,
          setModelMixers(v) { modelMixers = v; ctx.modelMixers = v; },
          setModelActions(v) { modelActions = v; ctx.modelActions = v; },
          setModelMixer(v) { modelMixer = v; ctx.modelMixer = v; },
          setModelAction(v) { modelAction = v; ctx.modelAction = v; },
          setLiveRetarget(v) { liveRetarget = v; ctx.liveRetarget = v; },
          setBodyLengthCalibration(v) { bodyLengthCalibration = v; ctx.bodyLengthCalibration = v; },
          setArmLengthCalibration(v) { armLengthCalibration = v; ctx.armLengthCalibration = v; },
          setFingerLengthCalibration(v) { fingerLengthCalibration = v; ctx.fingerLengthCalibration = v; },
          setIsPlaying(v) { isPlaying = v; },
        };

        runRetargetPipeline(ctx);
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
        applySkeletonVisibility();
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
          // Apply first frame so bones have correct world positions before measuring
          mixer.update(0);
          skeletonObj.updateMatrixWorld(true);

          // Shift skeletonObj so the clip's lowest support point sits on the grid.
          const groundY = computeBvhGroundY({
            bones: result.skeleton.bones,
            mixer,
            clip: currentClip,
            canonicalBoneKey,
          });
          if (Number.isFinite(groundY)) {
            skeletonObj.position.y -= groundY;
            skeletonObj.updateMatrixWorld(true);
          }

          fitToSkeleton(skeletonObj);
        }
        setStatus(`Loaded: ${label} (${Math.round(currentClip.duration * 100) / 100}s)`);
        if (modelSkinnedMesh) {
          window.__dbgHipsOnce = true;
          applyBvhToModel();
          // Log AFTER retargeting to see where bones ended up
          {
            const _wp2 = new THREE.Vector3();
            let _minY2 = Infinity, _minName2 = "";
            for (const bone of modelSkinnedMesh.skeleton.bones) {
              bone.getWorldPosition(_wp2);
              if (_wp2.y < _minY2) { _minY2 = _wp2.y; _minName2 = bone.name; }
            }
            console.log("[ground-snap/after-retarget] modelRoot.position.y=", modelRoot.position.y,
              "minBoneY=", _minY2, "(bone:", _minName2, ")");
          }
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
      set modelRoot(next) { modelRoot = next; applyModelVisibility(); },
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
    window.__vid2modelExportModelAnalysis = (download = false, filename = "") =>
      exportCurrentModelAnalysis(download, filename);
    window.__vid2modelBuildRigProfileSeed = (stage = getRetargetStage()) =>
      buildRigProfileSeedForCurrentModel(stage);
    window.__vid2modelAutoSetupModel = () => autoSetupModel();
    window.__vid2modelSaveModelSetup = () => saveModelSetup();
    window.__vid2modelExportRigProfile = (download = false, filename = "", allowDraft = false) =>
      exportCurrentRigProfile(download, filename, allowDraft);
    window.__vid2modelImportRigProfile = (payload, autoRetarget = true) =>
      importRigProfilePayload(payload, autoRetarget);
    window.__vid2modelBuildRegisterRigProfileCommand = (inputPath = "", allowDraft = false) =>
      buildRegisterRigProfileCommand({ inputPath, allowDraft });
    window.__vid2modelListRepoRigProfiles = (fingerprint = modelRigFingerprint, stage = "") =>
      listRepoRigProfiles(fingerprint, stage);
    window.__vid2modelGetRigProfileState = () => getLatestRigProfileState();
    window.__vid2modelValidateRigProfile = () => validateCurrentRigProfile();
    window.__vid2modelListRigProfiles = (fingerprint = modelRigFingerprint, stage = "") =>
      listRigProfiles(fingerprint, stage);

    const modelLoader = createViewerModelLoader({
      gltfLoader,
      setStatus,
      onParsedModel: applyParsedModel,
      setModelFileNameText: (text) => {
        modelFileNameEl.textContent = text;
      },
      defaultModelName: "MoonGirl.vrm",
      importMetaUrl: import.meta.url,
    });

    const {
      loadModelBuffer,
      loadModelFile,
      loadDefaultModel,
    } = modelLoader;

    const fileIo = createViewerFileIo({
      setStatus,
      loadBvhText,
      animationListEl: animationList,
    });

    const { loadBvhFileByName, loadDefault, loadAnimationsList } = fileIo;

    const viewerController = createViewerController({
      setStatus,
      loadBvhText,
      loadModelFile,
      loadDefault,
      loadBvhFileByName,
      autoSetupModel,
      saveModelSetup,
      retarget: applyBvhToModel,
      validateCurrentRigProfile,
      exportCurrentRigProfile,
      exportCurrentModelAnalysis,
      importRigProfileFile,
      zoomBy,
      resetCamera: () => {
        camera.position.set(260, 200, 260);
        controls.target.set(0, 100, 0);
        controls.update();
        setStatus("Camera reset");
      },
      getActiveDuration,
      updateTimelineUi,
      getPlaybackState: () => ({
        mixer,
        modelMixers,
        currentAction,
        modelActions,
        liveRetarget,
        bodyLengthCalibration,
        armLengthCalibration,
        fingerLengthCalibration,
      }),
      setIsPlaying: (next) => {
        isPlaying = !!next;
      },
      setIsScrubbing: (next) => {
        isScrubbing = !!next;
      },
      applyLiveRetargetPose,
      applyBoneLengthCalibration: (plan) => applyBoneLengthCalibration(plan, modelRoot),
      applyFingerLengthCalibration: (plan) => applyFingerLengthCalibration(plan, modelRoot),
      alignModelHipsToSource,
    });

    setupViewerUi({
      elements: {
        fileInput,
        modelInput,
        bvhFileNameEl,
        modelFileNameEl,
        animationList,
        btnLoadDefault,
        btnAutoSetup,
        btnSaveModelSetup,
        btnRetarget,
        btnValidateProfile,
        btnExportProfile,
        btnExportModelAnalysis,
        btnImportProfile,
        btnRetargetFab,
        btnPlayToggle,
        btnStop,
        btnToggleSkeleton,
        btnToggleModel,
        btnDarkToggle,
        btnToolsToggle,
        toolsGroup,
        timeline,
        timeEl,
        btnResetCamera,
        statusEl,
      },
      ops: { ...viewerController, toggleSkeleton, toggleModel },
      getIsPlaying: () => isPlaying,
      setSceneBg,
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
    loadAnimationsList();

    function animate() {
      requestAnimationFrame(animate);
      const dt = clock.getDelta();
        if (isPlaying && !isScrubbing) {
          if (mixer) mixer.update(dt);
          // In live-retarget mode, applyLiveRetargetPose drives all target bones directly.
          // Running modelMixers would displace modelRoot via root-motion tracks each frame.
          if (!liveRetarget) {
            for (const mix of modelMixers) {
              mix.update(dt);
            }
          }
        if (liveRetarget) {
          resetModelRootOrientation();
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
        // In live retarget mode, applyLiveRetargetPose already positions hips correctly
        // via posScale. alignModelHipsToSource would override it with raw world delta,
        // lifting the model off the ground.
        if (!liveRetarget) {
          alignModelHipsToSource(false);
        }
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
