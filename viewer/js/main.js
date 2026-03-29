    import * as THREE from "three";
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
      resetModelRootOrientation as resetModelRootOrientationModule,
      updateSourceOverlay as updateSourceOverlayModule,
    } from "./modules/retarget-live.js";
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
    let modelLabel = "";
    let liveRetarget = null;
    let bodyLengthCalibration = null;
    let armLengthCalibration = null;
    let fingerLengthCalibration = null;
    let sourceOverlay = null;
    let isPlaying = false;
    let isScrubbing = false;
    const loader = new BVHLoader();
    const gltfLoader = new GLTFLoader();
    const clock = new THREE.Clock();
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
    const _liveAxisY = new THREE.Vector3(0, 1, 0);
    const _rootYawQ = new THREE.Quaternion();
    const _calibV1 = new THREE.Vector3();
    const _calibV2 = new THREE.Vector3();
    const _calibQ1 = new THREE.Quaternion();
    const _overlayPivot = new THREE.Vector3();
    const _overlayUpAxis = new THREE.Vector3(0, 1, 0);

    function shortErr(err, limit = 120) {
      const text = String(err?.message || err || "");
      return text.length > limit ? `${text.slice(0, limit)}...` : text;
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
      modelRoot = null;
      modelLabel = "";
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
      const rows = [];
      const seenBoneIds = new Set();
      for (const mesh of skinnedMeshes || []) {
        for (const bone of mesh.skeleton?.bones || []) {
          const boneId = bone.uuid || `${mesh.uuid}:${bone.name}`;
          if (seenBoneIds.has(boneId)) continue;
          seenBoneIds.add(boneId);
          rows.push({
            index: rows.length,
            bone: bone.name || "(unnamed)",
            canonical: canonicalBoneKey(bone.name) || "",
            parent: bone.parent?.isBone ? bone.parent.name : "",
            mesh: mesh.name || "(unnamed-skinned-mesh)",
          });
        }
      }
      window.__vid2modelModelBones = rows;
      console.log("[vid2model/model-bones] total:", rows.length);
      console.table(rows);
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

    function buildLocalAxisCorrection(sourceBone, targetBone) {
      const sourceDir = new THREE.Vector3();
      const targetDir = new THREE.Vector3();
      if (!getPrimaryChildDirectionLocal(sourceBone, sourceDir)) return null;
      if (!getPrimaryChildDirectionLocal(targetBone, targetDir)) return null;
      const dot = Math.max(-1, Math.min(1, sourceDir.dot(targetDir)));
      if (dot > 0.9995) return null;
      return new THREE.Quaternion().setFromUnitVectors(sourceDir, targetDir).normalize();
    }

    function buildLiveRetargetPlan(skinnedMeshes, sourceBones, namesTargetToSource) {
      return buildLiveRetargetPlanModule({
        skinnedMeshes,
        sourceBones,
        namesTargetToSource,
        mixer,
        modelRoot,
        buildLocalAxisCorrection,
      });
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
        resetModelRootOrientation();
        const modelIsEn0 = /(^|[\\/])en_0\.vrm$/i.test(modelLabel);
        const { names, matched, unmatchedSample, canonicalCandidates, unmatchedHumanoid, sourceMatched } = buildRetargetMap(
          modelSkinnedMesh.skeleton.bones,
          sourceResult.skeleton.bones,
          { canonicalFilter }
        );
        const mappedPairs = Object.keys(names).length;
        diag("retarget-input", {
          stage: retargetStage,
          sourceBones: sourceResult.skeleton.bones.length,
          targetBones: modelSkinnedMesh.skeleton.bones.length,
          sourceTracks: stageClip.tracks.length,
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
          });
          const poseError = computePoseMatchError({
            bindings,
            sampleTime: probe.sampleTime,
            modelSkinnedMesh,
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
          modelIsEn0 &&
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
            });
            const skeletonPoseError = computePoseMatchError({
              bindings: skeletonBindings,
              sampleTime: skeletonProbe.sampleTime,
              modelSkinnedMesh,
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
          modelSkinnedMesh.skeleton.bones
        );
        const strongFacingMismatch = Math.abs(rawFacingYaw) > THREE.MathUtils.degToRad(100);
        const weakMotion = !!selectedProbe && selectedProbe.score < 0.5;
        const highPoseError = Number.isFinite(selectedPoseError) && selectedPoseError > 0.6;
        const selectedIsSkeletonUtils = selectedAttempt.label.startsWith("skeletonutils");
        const fullHumanoidMatch = canonicalCandidates > 0 && matched === canonicalCandidates;
        const autoUseLiveDelta =
          isRenameFallback ||
          (selectedIsSkeletonUtils && modelIsEn0 && !fullHumanoidMatch && (highPoseError || strongFacingMismatch)) ||
          (!selectedIsSkeletonUtils && (strongFacingMismatch || weakMotion || highPoseError));
        const forcedLiveDelta = getLiveDeltaOverride(window);
        const useLiveDelta = forcedLiveDelta === null ? autoUseLiveDelta : forcedLiveDelta;
        diag("retarget-live-delta", {
          stage: retargetStage,
          selectedMode: selectedAttempt.label,
          selectedIsSkeletonUtils,
          modelIsEn0,
          fullHumanoidMatch,
          autoUseLiveDelta,
          forced: forcedLiveDelta,
          useLiveDelta,
          reasons: {
            isRenameFallback,
            strongFacingMismatch,
            weakMotion,
            highPoseError,
          },
        });
        let rootYawCorrection = 0;
        if (selectedIsSkeletonUtils && !useLiveDelta) {
          const yawCandidates = buildRootYawCandidates(rawFacingYaw, quantizeFacingYaw);
          const yawEval = evaluateRootYawCandidates({
            candidates: yawCandidates,
            sampleTime: selectedProbe?.sampleTime || 0,
            namesTargetToSource: names,
            sourceClip: clip,
            modelRoot,
            modelMixers,
            modelSkinnedMesh,
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
          const shouldUseBest =
            !!bestRow &&
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
            modelSkinnedMesh.skeleton.bones,
            sourceResult.skeleton.bones,
            names
          );
          let hipsYawCorrection = 0;
          let hipsCorrectionApplied = false;
          let hipsCorrectionEval = null;
          if (Math.abs(hipsYawError) > THREE.MathUtils.degToRad(12)) {
            const correctedYaw = rootYawCorrection - hipsYawError;
            const postEval = evaluateRootYawCandidates({
              candidates: [rootYawCorrection, correctedYaw],
              sampleTime: selectedProbe?.sampleTime || 0,
              namesTargetToSource: names,
              sourceClip: clip,
              modelRoot,
              modelMixers,
              modelSkinnedMesh,
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
        if (useLiveDelta) {
          const livePlan = buildLiveRetargetPlan(
            modelSkinnedMeshes,
            sourceResult.skeleton.bones,
            names
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
          const bodyTargetBones = modelSkinnedMesh.skeleton.bones.filter((b) =>
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
          if (filteredBodyCalibration?.entries?.length) {
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

        isPlaying = true;
        updateTimelineUi(syncTime);
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
        const limbDiag = collectLimbDiagnostics(
          modelSkinnedMesh.skeleton.bones,
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
                calibratedPairs: liveRetarget.calibratedPairs || 0,
                yawOffsetDeg: Number((liveRetarget.yawOffset * 180 / Math.PI).toFixed(2)),
              }
            : null,
          activeMeshCount,
        };
        const unmatchedTargetBones = modelSkinnedMesh.skeleton.bones
          .map((b) => b.name)
          .filter((name) => !names[name]);
        window.__vid2modelUnmatchedTargetBones = unmatchedTargetBones;
        diag("retarget-map-details", {
          stage: retargetStage,
          totalTargetBones: modelSkinnedMesh.skeleton.bones.length,
          mappedTargetBones: modelSkinnedMesh.skeleton.bones.length - unmatchedTargetBones.length,
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
          rootYawDeg: Number((rootYawCorrection * 180 / Math.PI).toFixed(2)),
          yawOffsetDeg: liveRetarget ? Number((liveRetarget.yawOffset * 180 / Math.PI).toFixed(2)) : 0,
          calibratedPairs: liveRetarget ? liveRetarget.calibratedPairs || 0 : 0,
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
    // Optional manual override for troubleshooting:
    //   window.__vid2modelForceLiveDelta = true | false | null
    window.__vid2modelForceLiveDelta = null;
    // Optional file logger for diagnostics:
    //   window.__vid2modelStartFileDiag() / window.__vid2modelStopFileDiag()
    // Logs are sent to a local endpoint and can be tailed from terminal.
    window.__vid2modelDiagFileLogger = { enabled: true, url: DIAG_FILE_LOG_DEFAULT_URL };
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
      modelRoot = gltf.scene || gltf.scenes?.[0] || null;
      modelLabel = label || "";
      if (!modelRoot) {
        setStatus(`Failed to parse model: ${label}`);
        return;
      }
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
