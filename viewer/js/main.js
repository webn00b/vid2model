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
    let fingerLengthCalibration = null;
    let sourceOverlay = null;
    let isPlaying = false;
    let isScrubbing = false;
    const loader = new BVHLoader();
    const gltfLoader = new GLTFLoader();
    const clock = new THREE.Clock();
    const DIAG_PREFIX = "[vid2model/diag]";
    const DIAG_FILE_LOG_DEFAULT_URL = "http://127.0.0.1:8765/diag";
    const DIAG_EVENTS = new Set([
      "retarget-summary",
      "retarget-limbs",
      "retarget-fail",
      "retarget-alignment",
      "retarget-map-details",
      "retarget-live-delta",
      "retarget-root-yaw",
      "retarget-hips-align",
      "retarget-finger-calibration",
    ]);
    let diagSeq = 0;
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
    const XR2ANIM_LEGACY_ALIAS = new Map([
      ["center", "hips"],
      ["upperbody", "chest"],
      ["upperbody2", "upperChest"],
      ["lowerbody", "spine"],
      ["lcollar", "leftShoulder"],
      ["rcollar", "rightShoulder"],
      ["lshldr", "leftUpperArm"],
      ["rshl", "rightUpperArm"],
      ["lforearm", "leftLowerArm"],
      ["rforearm", "rightLowerArm"],
      ["lhand", "leftHand"],
      ["rhand", "rightHand"],
      ["lthigh", "leftUpperLeg"],
      ["rthigh", "rightUpperLeg"],
      ["lshin", "leftLowerLeg"],
      ["rshin", "rightLowerLeg"],
      ["ltoe", "leftToes"],
      ["rtoe", "rightToes"],
      ["lthumb1", "leftThumbMetacarpal"],
      ["lthumb2", "leftThumbProximal"],
      ["lthumb3", "leftThumbDistal"],
      ["rthumb1", "rightThumbMetacarpal"],
      ["rthumb2", "rightThumbProximal"],
      ["rthumb3", "rightThumbDistal"],
      ["lindex1", "leftIndexProximal"],
      ["lindex2", "leftIndexIntermediate"],
      ["lindex3", "leftIndexDistal"],
      ["rindex1", "rightIndexProximal"],
      ["rindex2", "rightIndexIntermediate"],
      ["rindex3", "rightIndexDistal"],
      ["lmid1", "leftMiddleProximal"],
      ["lmid2", "leftMiddleIntermediate"],
      ["lmid3", "leftMiddleDistal"],
      ["rmid1", "rightMiddleProximal"],
      ["rmid2", "rightMiddleIntermediate"],
      ["rmid3", "rightMiddleDistal"],
      ["lring1", "leftRingProximal"],
      ["lring2", "leftRingIntermediate"],
      ["lring3", "leftRingDistal"],
      ["rring1", "rightRingProximal"],
      ["rring2", "rightRingIntermediate"],
      ["rring3", "rightRingDistal"],
      ["lpinky1", "leftLittleProximal"],
      ["lpinky2", "leftLittleIntermediate"],
      ["lpinky3", "leftLittleDistal"],
      ["rpinky1", "rightLittleProximal"],
      ["rpinky2", "rightLittleIntermediate"],
      ["rpinky3", "rightLittleDistal"],
      ["thumb01l", "leftThumbMetacarpal"],
      ["thumb02l", "leftThumbProximal"],
      ["thumb03l", "leftThumbDistal"],
      ["thumb01r", "rightThumbMetacarpal"],
      ["thumb02r", "rightThumbProximal"],
      ["thumb03r", "rightThumbDistal"],
      ["findex01l", "leftIndexProximal"],
      ["findex02l", "leftIndexIntermediate"],
      ["findex03l", "leftIndexDistal"],
      ["findex01r", "rightIndexProximal"],
      ["findex02r", "rightIndexIntermediate"],
      ["findex03r", "rightIndexDistal"],
      ["fmiddle01l", "leftMiddleProximal"],
      ["fmiddle02l", "leftMiddleIntermediate"],
      ["fmiddle03l", "leftMiddleDistal"],
      ["fmiddle01r", "rightMiddleProximal"],
      ["fmiddle02r", "rightMiddleIntermediate"],
      ["fmiddle03r", "rightMiddleDistal"],
      ["fring01l", "leftRingProximal"],
      ["fring02l", "leftRingIntermediate"],
      ["fring03l", "leftRingDistal"],
      ["fring01r", "rightRingProximal"],
      ["fring02r", "rightRingIntermediate"],
      ["fring03r", "rightRingDistal"],
      ["fpinky01l", "leftLittleProximal"],
      ["fpinky02l", "leftLittleIntermediate"],
      ["fpinky03l", "leftLittleDistal"],
      ["fpinky01r", "rightLittleProximal"],
      ["fpinky02r", "rightLittleIntermediate"],
      ["fpinky03r", "rightLittleDistal"],
    ]);
    const RETARGET_STAGES = new Set(["body", "full"]);
    const RETARGET_BODY_CANONICAL = new Set([
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
    ]);

    function sanitizeDiagPayload(payload) {
      try {
        return JSON.parse(JSON.stringify(payload));
      } catch (err) {
        return { __serializeError: shortErr(err), text: String(payload) };
      }
    }

    function getDiagFileLoggerConfig() {
      const cfg = window.__vid2modelDiagFileLogger || {};
      return {
        enabled: !!cfg.enabled,
        url: String(cfg.url || DIAG_FILE_LOG_DEFAULT_URL),
      };
    }

    function sendDiagToFile(event, payload) {
      const cfg = getDiagFileLoggerConfig();
      if (!cfg.enabled || !cfg.url) return;
      const record = {
        ts: new Date().toISOString(),
        seq: diagSeq++,
        event,
        payload: sanitizeDiagPayload(payload),
      };
      const body = JSON.stringify(record);
      fetch(cfg.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        keepalive: true,
        mode: "cors",
        credentials: "omit",
      }).catch(() => {});
    }

    function diag(event, payload = {}) {
      if (!DIAG_EVENTS.has(event)) return;
      console.log(DIAG_PREFIX, event, payload);
      sendDiagToFile(event, payload);
    }

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
      if (!modelRoot) return;
      const baseQ = modelRoot.userData?.__baseQuaternion;
      if (baseQ?.isQuaternion) {
        modelRoot.quaternion.copy(baseQ);
      }
      const baseP = modelRoot.userData?.__basePosition;
      if (baseP?.isVector3) {
        modelRoot.position.copy(baseP);
      }
      const rootBone = getModelSkeletonRootBone();
      if (rootBone && rootBone !== modelRoot) {
        const rootBaseQ = rootBone.userData?.__retargetBaseQuaternion;
        const rootBaseP = rootBone.userData?.__retargetBasePosition;
        if (rootBaseQ?.isQuaternion) {
          rootBone.quaternion.copy(rootBaseQ);
        }
        if (rootBaseP?.isVector3) {
          rootBone.position.copy(rootBaseP);
        }
      }
      modelRoot.updateMatrixWorld(true);
    }

    function applyModelRootYaw(yawRad) {
      if (!modelRoot || !Number.isFinite(yawRad) || Math.abs(yawRad) < 1e-5) return 0;
      const baseQ = modelRoot.userData?.__baseQuaternion;
      if (baseQ?.isQuaternion) {
        modelRoot.quaternion.copy(baseQ);
      }
      _rootYawQ.setFromAxisAngle(_liveAxisY, yawRad);
      modelRoot.quaternion.premultiply(_rootYawQ);
      modelRoot.updateMatrixWorld(true);
      return yawRad;
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

    function buildRetargetMap(targetBones, sourceBones, options = {}) {
      const canonicalFilter = options.canonicalFilter || null;
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
        // Finger fallback: if one chain is missing in source BVH, borrow from nearest chain
        // so target fingers still follow hand motion instead of freezing.
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

    function buildStageSourceClip(sourceClip, sourceBones, stage) {
      if (!sourceClip) return null;
      const canonicalFilter = getCanonicalFilterForStage(stage);
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
      return new THREE.AnimationClip(
        `${sourceClip.name || "retarget"}_${stage}`,
        sourceClip.duration,
        tracks
      );
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

    function collectLimbDiagnostics(targetBones, sourceBones, namesTargetToSource, clip) {
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

    const FINGER_PARENT = new Map([
      ["leftThumbMetacarpal", "leftHand"],
      ["leftThumbProximal", "leftThumbMetacarpal"],
      ["leftThumbDistal", "leftThumbProximal"],
      ["leftIndexProximal", "leftHand"],
      ["leftIndexIntermediate", "leftIndexProximal"],
      ["leftIndexDistal", "leftIndexIntermediate"],
      ["leftMiddleProximal", "leftHand"],
      ["leftMiddleIntermediate", "leftMiddleProximal"],
      ["leftMiddleDistal", "leftMiddleIntermediate"],
      ["leftRingProximal", "leftHand"],
      ["leftRingIntermediate", "leftRingProximal"],
      ["leftRingDistal", "leftRingIntermediate"],
      ["leftLittleProximal", "leftHand"],
      ["leftLittleIntermediate", "leftLittleProximal"],
      ["leftLittleDistal", "leftLittleIntermediate"],
      ["rightThumbMetacarpal", "rightHand"],
      ["rightThumbProximal", "rightThumbMetacarpal"],
      ["rightThumbDistal", "rightThumbProximal"],
      ["rightIndexProximal", "rightHand"],
      ["rightIndexIntermediate", "rightIndexProximal"],
      ["rightIndexDistal", "rightIndexIntermediate"],
      ["rightMiddleProximal", "rightHand"],
      ["rightMiddleIntermediate", "rightMiddleProximal"],
      ["rightMiddleDistal", "rightMiddleIntermediate"],
      ["rightRingProximal", "rightHand"],
      ["rightRingIntermediate", "rightRingProximal"],
      ["rightRingDistal", "rightRingIntermediate"],
      ["rightLittleProximal", "rightHand"],
      ["rightLittleIntermediate", "rightLittleProximal"],
      ["rightLittleDistal", "rightLittleIntermediate"],
    ]);

    const BODY_SCALE_REFERENCE = new Map([
      ["spine", "hips"],
      ["chest", "spine"],
      ["upperChest", "chest"],
      ["leftUpperArm", "leftShoulder"],
      ["leftLowerArm", "leftUpperArm"],
      ["rightUpperArm", "rightShoulder"],
      ["rightLowerArm", "rightUpperArm"],
      ["leftUpperLeg", "hips"],
      ["leftLowerLeg", "leftUpperLeg"],
      ["rightUpperLeg", "hips"],
      ["rightLowerLeg", "rightUpperLeg"],
    ]);

    function getBoneBindPosition(bone) {
      const bind = bone?.userData?.__bindPosition;
      if (bind?.isVector3) return bind;
      return bone?.position?.isVector3 ? bone.position : null;
    }

    function getBindSegmentLength(bone) {
      const bindPos = getBoneBindPosition(bone);
      if (!bindPos) return null;
      const len = bindPos.length();
      return Number.isFinite(len) && len > 1e-6 ? len : null;
    }

    function median(values) {
      if (!values?.length) return null;
      const sorted = values
        .filter((v) => Number.isFinite(v))
        .sort((a, b) => a - b);
      if (!sorted.length) return null;
      const mid = Math.floor(sorted.length / 2);
      if (sorted.length % 2 === 1) return sorted[mid];
      return (sorted[mid - 1] + sorted[mid]) / 2;
    }

    function fingerScaleLimits(canonical) {
      if (canonical.includes("Metacarpal")) return { min: 0.65, max: 2.1 };
      if (canonical.includes("Proximal")) return { min: 0.65, max: 2.3 };
      if (canonical.includes("Intermediate")) return { min: 0.65, max: 2.5 };
      if (canonical.includes("Distal")) return { min: 0.65, max: 2.7 };
      return { min: 0.65, max: 2.4 };
    }

    function estimateGlobalRigScale(sourceMap, targetMap) {
      const ratios = [];
      for (const [key, parentKey] of BODY_SCALE_REFERENCE.entries()) {
        const sourceBone = sourceMap.get(key) || null;
        const targetBone = targetMap.get(key) || null;
        const sourceParent = sourceMap.get(parentKey) || null;
        const targetParent = targetMap.get(parentKey) || null;
        if (!sourceBone || !targetBone || !sourceParent || !targetParent) continue;
        const sourceLen = getBindSegmentLength(sourceBone);
        const targetLen = getBindSegmentLength(targetBone);
        if (!(sourceLen > 1e-6 && targetLen > 1e-6)) continue;
        const ratio = targetLen / sourceLen;
        if (Number.isFinite(ratio) && ratio > 1e-6) {
          ratios.push(ratio);
        }
      }
      const med = median(ratios);
      return Number.isFinite(med) ? THREE.MathUtils.clamp(med, 0.15, 5) : 1;
    }

    function buildFingerLengthCalibration(sourceBones, targetBones, clip) {
      const sourceMap = buildCanonicalBoneMap(sourceBones);
      const targetMap = buildCanonicalBoneMap(targetBones);
      const trackByBone = collectTrackPresenceByBone(clip);
      const globalScale = estimateGlobalRigScale(sourceMap, targetMap);
      const entries = [];
      let clampedCount = 0;
      let minRawScale = Number.POSITIVE_INFINITY;
      let maxRawScale = Number.NEGATIVE_INFINITY;
      for (const [key, parentKey] of FINGER_PARENT.entries()) {
        const targetBone = targetMap.get(key) || null;
        const sourceBone = sourceMap.get(key) || null;
        const targetParent = targetMap.get(parentKey) || null;
        const sourceParent = sourceMap.get(parentKey) || null;
        if (!targetBone || !sourceBone || !targetParent || !sourceParent) continue;
        const track = trackByBone.get(targetBone.name);
        if (track?.hasP) continue;
        const sourceLen = getBindSegmentLength(sourceBone);
        const targetLen = getBindSegmentLength(targetBone);
        if (!(sourceLen > 1e-6 && targetLen > 1e-6)) continue;
        const expectedTargetLen = sourceLen * globalScale;
        const rawScale = expectedTargetLen / targetLen;
        if (!Number.isFinite(rawScale)) continue;
        minRawScale = Math.min(minRawScale, rawScale);
        maxRawScale = Math.max(maxRawScale, rawScale);
        const limits = fingerScaleLimits(key);
        const scale = THREE.MathUtils.clamp(rawScale, limits.min, limits.max);
        if (Math.abs(scale - rawScale) > 1e-6) clampedCount += 1;
        if (!Number.isFinite(scale) || Math.abs(scale - 1) < 0.06) continue;
        const bindPos = getBoneBindPosition(targetBone)?.clone?.() || targetBone.position.clone();
        if (bindPos.lengthSq() < 1e-10) continue;
        entries.push({
          bone: targetBone,
          canonical: key,
          scale: Number(scale.toFixed(4)),
          rawScale: Number(rawScale.toFixed(4)),
          sourceLen: Number(sourceLen.toFixed(5)),
          targetLen: Number(targetLen.toFixed(5)),
          expectedTargetLen: Number(expectedTargetLen.toFixed(5)),
          scaledPos: bindPos.multiplyScalar(scale),
        });
      }
      return entries.length
        ? {
            entries,
            globalScale: Number(globalScale.toFixed(5)),
            clampedCount,
            minRawScale: Number.isFinite(minRawScale) ? Number(minRawScale.toFixed(4)) : null,
            maxRawScale: Number.isFinite(maxRawScale) ? Number(maxRawScale.toFixed(4)) : null,
          }
        : null;
    }

    function applyFingerLengthCalibration(plan) {
      if (!plan?.entries?.length) return;
      for (const e of plan.entries) {
        e.bone.position.copy(e.scaledPos);
      }
      modelRoot?.updateMatrixWorld(true);
    }

    function collectTrackPresenceByBone(clip) {
      const trackByBone = new Map();
      for (const t of clip?.tracks || []) {
        const parsed = parseTrackName(t.name);
        if (!parsed) continue;
        const row = trackByBone.get(parsed.bone) || { hasQ: false, hasP: false };
        if (parsed.property === "quaternion") row.hasQ = true;
        if (parsed.property === "position") row.hasP = true;
        trackByBone.set(parsed.bone, row);
      }
      return trackByBone;
    }

    function collectAlignmentDiagnostics(
      targetBones,
      sourceBones,
      namesTargetToSource,
      sourceClip,
      maxRows = 24,
      overlayYawOverride = null
    ) {
      const sourceByName = new Map((sourceBones || []).map((b) => [b.name, b]));
      const sourceTrackByBone = collectTrackPresenceByBone(sourceClip);
      const rows = [];
      const unmappedCanonical = [];
      const sourceMissing = [];

      const overlayYaw =
        Number.isFinite(overlayYawOverride)
          ? overlayYawOverride
          : (sourceOverlay?.overlayYaw || 0);
      const applyOverlayYaw = Number.isFinite(overlayYaw) && Math.abs(overlayYaw) > 1e-5;
      const pivot = new THREE.Vector3();
      if (applyOverlayYaw && sourceOverlay?.pivotBone) {
        sourceOverlay.pivotBone.getWorldPosition(pivot);
      }
      const overlayYawQ = new THREE.Quaternion();
      if (applyOverlayYaw) {
        overlayYawQ.setFromAxisAngle(_overlayUpAxis, overlayYaw);
      }

      const srcPos = new THREE.Vector3();
      const dstPos = new THREE.Vector3();
      const srcQ = new THREE.Quaternion();
      const dstQ = new THREE.Quaternion();

      for (const targetBone of targetBones || []) {
        const canonical = canonicalBoneKey(targetBone.name) || "";
        const sourceName = namesTargetToSource[targetBone.name] || "";
        if (!sourceName) {
          if (canonical) {
            unmappedCanonical.push({
              target: targetBone.name,
              canonical,
            });
          }
          continue;
        }
        const sourceBone = sourceByName.get(sourceName);
        if (!sourceBone) {
          sourceMissing.push({
            target: targetBone.name,
            source: sourceName,
            canonical,
          });
          continue;
        }

        sourceBone.getWorldPosition(srcPos);
        if (applyOverlayYaw) {
          srcPos.sub(pivot).applyQuaternion(overlayYawQ).add(pivot);
        }
        targetBone.getWorldPosition(dstPos);
        const posErr = dstPos.distanceTo(srcPos);

        sourceBone.getWorldQuaternion(srcQ);
        if (applyOverlayYaw) {
          srcQ.premultiply(overlayYawQ);
        }
        targetBone.getWorldQuaternion(dstQ);
        const rotErrDeg = THREE.MathUtils.radToDeg(srcQ.angleTo(dstQ));
        const trackState = sourceTrackByBone.get(sourceName) || { hasQ: false, hasP: false };

        rows.push({
          target: targetBone.name,
          source: sourceName,
          canonical,
          posErr: Number(posErr.toFixed(5)),
          rotErrDeg: Number(rotErrDeg.toFixed(3)),
          sourceHasQ: trackState.hasQ,
          sourceHasP: trackState.hasP,
        });
      }

      const sortedByPos = rows.slice().sort((a, b) => b.posErr - a.posErr);
      const sortedByRot = rows.slice().sort((a, b) => b.rotErrDeg - a.rotErrDeg);
      const avgPosErr = rows.length ? rows.reduce((sum, r) => sum + r.posErr, 0) / rows.length : 0;
      const avgRotErrDeg = rows.length ? rows.reduce((sum, r) => sum + r.rotErrDeg, 0) / rows.length : 0;

      const hipsTarget = targetBones.find((b) => canonicalBoneKey(b.name) === "hips") || null;
      const hipsSourceName = hipsTarget ? namesTargetToSource[hipsTarget.name] || "" : "";
      const hipsSource = hipsSourceName ? sourceByName.get(hipsSourceName) || null : null;
      let hipsPosErr = null;
      if (hipsTarget && hipsSource) {
        hipsTarget.getWorldPosition(dstPos);
        hipsSource.getWorldPosition(srcPos);
        if (applyOverlayYaw) {
          srcPos.sub(pivot).applyQuaternion(overlayYawQ).add(pivot);
        }
        hipsPosErr = Number(dstPos.distanceTo(srcPos).toFixed(5));
      }

      return {
        totalCompared: rows.length,
        avgPosErr: Number(avgPosErr.toFixed(5)),
        avgRotErrDeg: Number(avgRotErrDeg.toFixed(3)),
        overlayYawDeg: Number((overlayYaw * 180 / Math.PI).toFixed(2)),
        hipsPosErr,
        worstPosition: sortedByPos.slice(0, maxRows),
        worstRotation: sortedByRot.slice(0, maxRows),
        unmappedCanonical: unmappedCanonical.slice(0, maxRows),
        sourceMissing: sourceMissing.slice(0, maxRows),
      };
    }

    function dumpRetargetAlignmentDiagnostics(reason = "manual") {
      if (!modelSkinnedMesh?.skeleton?.bones?.length || !sourceResult?.skeleton?.bones?.length) {
        console.warn("[vid2model/diag] retarget alignment dump skipped: source/model is not ready");
        return null;
      }
      const names = window.__vid2modelDebug?.names || {};
      const report = collectAlignmentDiagnostics(
        modelSkinnedMesh.skeleton.bones,
        sourceResult.skeleton.bones,
        names,
        sourceResult.clip,
        20
      );
      window.__vid2modelAlignment = report;
      console.log("[vid2model/diag] alignment-dump", { reason, ...report });
      if (report.worstPosition.length) console.table(report.worstPosition);
      if (report.worstRotation.length) console.table(report.worstRotation);
      if (report.unmappedCanonical.length) {
        console.log("[vid2model/diag] unmapped-canonical");
        console.table(report.unmappedCanonical);
      }
      if (report.sourceMissing.length) {
        console.log("[vid2model/diag] source-missing");
        console.table(report.sourceMissing);
      }
      return report;
    }

    function getLiveDeltaOverride() {
      const v = window.__vid2modelForceLiveDelta;
      if (v === true || v === false) return v;
      return null;
    }

    function computeHipsYawError(targetBones, sourceBones, namesTargetToSource) {
      const targetHips = (targetBones || []).find((b) => canonicalBoneKey(b.name) === "hips") || null;
      if (!targetHips) return 0;
      const sourceByName = new Map((sourceBones || []).map((b) => [b.name, b]));
      const sourceHipsName =
        namesTargetToSource[targetHips.name] ||
        (sourceBones || []).find((b) => canonicalBoneKey(b.name) === "hips")?.name ||
        "";
      if (!sourceHipsName) return 0;
      const sourceHips = sourceByName.get(sourceHipsName);
      if (!sourceHips) return 0;

      const srcQ = new THREE.Quaternion();
      const dstQ = new THREE.Quaternion();
      sourceHips.getWorldQuaternion(srcQ);
      targetHips.getWorldQuaternion(dstQ);

      const srcF = new THREE.Vector3(0, 0, 1).applyQuaternion(srcQ);
      const dstF = new THREE.Vector3(0, 0, 1).applyQuaternion(dstQ);
      srcF.y = 0;
      dstF.y = 0;
      if (srcF.lengthSq() < 1e-9 || dstF.lengthSq() < 1e-9) return 0;
      srcF.normalize();
      dstF.normalize();
      const crossY = srcF.clone().cross(dstF).y;
      const dot = THREE.MathUtils.clamp(srcF.dot(dstF), -1, 1);
      const angle = Math.atan2(crossY, dot); // source -> target
      return Number.isFinite(angle) ? angle : 0;
    }

    function buildRootYawCandidates(rawFacingYaw) {
      const set = new Set();
      const list = [];
      const push = (v) => {
        if (!Number.isFinite(v)) return;
        let a = Math.atan2(Math.sin(v), Math.cos(v)); // normalize to [-PI, PI]
        if (Math.abs(a) < 1e-6) a = 0;
        const key = Number(a.toFixed(6));
        if (set.has(key)) return;
        set.add(key);
        list.push(a);
      };
      push(0);
      push(Math.PI);
      push(-Math.PI);
      push(quantizeFacingYaw(-rawFacingYaw));
      push(quantizeFacingYaw(Math.PI - rawFacingYaw));
      push(-rawFacingYaw);
      return list;
    }

    function evaluateRootYawCandidates(candidates, sampleTime, namesTargetToSource, sourceClip) {
      if (!modelRoot || !modelMixers.length || !modelSkinnedMesh?.skeleton?.bones?.length || !sourceResult?.skeleton?.bones?.length) {
        return { bestYaw: 0, rows: [] };
      }
      const sourceTime = mixer ? mixer.time : 0;
      const modelTimes = modelMixers.map((mix) => mix.time);
      const t = sampleTime > 0 ? sampleTime : 1 / 30;
      const rows = [];
      try {
        for (const yaw of candidates) {
          resetModelRootOrientation();
          const appliedYaw = applyModelRootYaw(yaw);
          if (mixer) mixer.setTime(t);
          for (const mix of modelMixers) {
            mix.setTime(t);
          }
          modelRoot.updateMatrixWorld(true);
          const alignment = collectAlignmentDiagnostics(
            modelSkinnedMesh.skeleton.bones,
            sourceResult.skeleton.bones,
            namesTargetToSource,
            sourceClip,
            5,
            0
          );
          const hips = Number.isFinite(alignment.hipsPosErr) ? alignment.hipsPosErr : alignment.avgPosErr;
          const score = alignment.avgPosErr + alignment.avgRotErrDeg * 0.003 + hips * 1.25;
          rows.push({
            yawRad: appliedYaw,
            yawDeg: Number((appliedYaw * 180 / Math.PI).toFixed(2)),
            score: Number(score.toFixed(6)),
            avgPosErr: alignment.avgPosErr,
            avgRotErrDeg: alignment.avgRotErrDeg,
            hipsPosErr: alignment.hipsPosErr,
          });
        }
      } finally {
        if (mixer) mixer.setTime(sourceTime);
        for (let i = 0; i < modelMixers.length; i += 1) {
          modelMixers[i].setTime(modelTimes[i] || 0);
        }
        resetModelRootOrientation();
      }
      rows.sort((a, b) => a.score - b.score);
      const bestYaw = rows.length ? rows[0].yawRad : 0;
      return { bestYaw, rows };
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
            axisCorrectionQ: new THREE.Quaternion(),
            axisCorrectionQInv: new THREE.Quaternion(),
            hasAxisCorrection: false,
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

      let calibratedPairs = 0;
      for (const pair of pairs) {
        if (pair.isHips) continue;
        const corr = buildLocalAxisCorrection(pair.source, pair.target);
        if (!corr) continue;
        pair.axisCorrectionQ.copy(corr);
        pair.axisCorrectionQInv.copy(corr).invert();
        pair.hasAxisCorrection = true;
        calibratedPairs += 1;
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
      // retarget yaw should rotate target-facing direction toward source-facing direction.
      let yawOffset = -rawYawOffset;
      // Keep yaw correction conservative: either no correction or a clean 180deg flip.
      if (absRawYaw < THREE.MathUtils.degToRad(45)) {
        yawOffset = 0;
      } else if (absRawYaw > THREE.MathUtils.degToRad(120)) {
        yawOffset = Math.sign(yawOffset || 1) * Math.PI;
      }

      return {
        pairs,
        uniqueSkeletons,
        posScale,
        yawOffset,
        calibratedPairs,
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
        if (pair.hasAxisCorrection) {
          _liveQ.premultiply(pair.axisCorrectionQ).multiply(pair.axisCorrectionQInv);
        }
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
        const retargetStage = getRetargetStage();
        const stageClip = buildStageSourceClip(sourceResult.clip, sourceResult.skeleton.bones, retargetStage);
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
          diag("retarget-fail", { reason: "no_resolved_tracks", stage: retargetStage });
          return;
        }

        if (
          modelIsEn0 &&
          selectedAttempt?.label?.startsWith("rename-fallback") &&
          skeletonSkinned
        ) {
          const skeletonBindings = buildBindingsForAttempt(skeletonSkinned, skeletonSkinned.clip);
          if (skeletonBindings.mixers.length) {
            const skeletonProbe = probeMotionForBindings(skeletonBindings, skeletonSkinned.clip);
            const skeletonPoseError = computePoseMatchError(skeletonBindings, skeletonProbe.sampleTime);
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
        const forcedLiveDelta = getLiveDeltaOverride();
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
          const yawCandidates = buildRootYawCandidates(rawFacingYaw);
          const yawEval = evaluateRootYawCandidates(
            yawCandidates,
            selectedProbe?.sampleTime || 0,
            names,
            clip
          );
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
            const postEval = evaluateRootYawCandidates(
              [rootYawCorrection, correctedYaw],
              selectedProbe?.sampleTime || 0,
              names,
              clip
            );
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
        fingerLengthCalibration = null;
        if (!liveRetarget && retargetStage === "full") {
          fingerLengthCalibration = buildFingerLengthCalibration(
            sourceResult.skeleton.bones,
            modelSkinnedMesh.skeleton.bones,
            clip
          );
          if (fingerLengthCalibration) {
            applyFingerLengthCalibration(fingerLengthCalibration);
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
        const alignment = dumpRetargetAlignmentDiagnostics("auto-retarget");
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

    window.__vid2modelDumpAlignment = (reason = "manual") => dumpRetargetAlignmentDiagnostics(reason);
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
    if (btnRetargetFab) {
      btnRetargetFab.addEventListener("click", applyBvhToModel);
    }
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
      if (fingerLengthCalibration && !liveRetarget) {
        applyFingerLengthCalibration(fingerLengthCalibration);
      }
      alignModelHipsToSource(false);
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
      if (fingerLengthCalibration && !liveRetarget) {
        applyFingerLengthCalibration(fingerLengthCalibration);
      }
      alignModelHipsToSource(false);
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
        if (fingerLengthCalibration && !liveRetarget) {
          applyFingerLengthCalibration(fingerLengthCalibration);
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
