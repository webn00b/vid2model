export const DIAG_PREFIX = "[vid2model/diag]";
export const DIAG_FILE_LOG_DEFAULT_URL = "http://127.0.0.1:8765/diag";

const DIAG_EVENTS = new Set([
  "model-loaded",
  "model-default-animation",
  "retarget-input",
  "retarget-summary",
  "retarget-limbs",
  "retarget-fail",
  "retarget-alignment",
  "retarget-map-details",
  "retarget-topology-fallback",
  "retarget-live-delta",
  "retarget-root-yaw",
  "retarget-hips-align",
  "retarget-body-calibration",
  "retarget-arm-refine",
  "retarget-finger-calibration",
]);

const DIAG_CONSOLE_EVENTS_MINIMAL = new Set([
  "model-loaded",
  "retarget-fail",
  "retarget-topology-fallback",
  "retarget-live-delta",
  "retarget-root-yaw",
  "retarget-summary",
]);

function sanitizeDiagPayload(payload) {
  try {
    return JSON.parse(JSON.stringify(payload));
  } catch (err) {
    return {
      __serializeError: String(err?.message || err || "serialize_error"),
      text: String(payload),
    };
  }
}

function compactArray(value, limit = 4) {
  if (!Array.isArray(value)) return value;
  if (value.length <= limit) return value;
  return {
    count: value.length,
    preview: value.slice(0, limit),
  };
}

function compactConsolePayload(event, payload) {
  if (!payload || typeof payload !== "object") return payload;
  switch (event) {
    case "model-loaded":
      return {
        file: payload.file,
        skinnedMeshes: payload.skinnedMeshes,
        vrmHumanoid: payload.vrmHumanoid,
        vrmDirectReady: payload.vrmDirectReady,
        topMeshes: compactArray(payload.topMeshes, 2),
      };
    case "model-default-animation":
      return {
        file: payload.file,
        clip: payload.clip,
        duration: payload.duration,
        url: payload.url,
      };
    case "retarget-fail":
      return {
        stage: payload.stage,
        reason: payload.reason,
        unmatched: compactArray(payload.unmatched, 3),
      };
    case "retarget-topology-fallback":
      return {
        stage: payload.stage,
        attempted: payload.attempted,
        applied: payload.applied,
        reason: payload.reason,
        inferredRenames: payload.inferredRenames,
      };
    case "retarget-live-delta":
      return {
        stage: payload.stage,
        selectedMode: payload.selectedMode,
        useLiveDelta: payload.useLiveDelta,
        autoUseLiveDelta: payload.autoUseLiveDelta,
        reasons: payload.reasons,
      };
    case "retarget-root-yaw":
      return {
        stage: payload.stage,
        rawFacingYawDeg: payload.rawFacingYawDeg,
        appliedYawDeg: payload.appliedYawDeg,
        hipsYawErrorDeg: payload.hipsYawErrorDeg,
        strongFacingMismatch: payload.strongFacingMismatch,
        usedBestCandidate: payload.usedBestCandidate,
      };
    case "retarget-summary":
      return {
        stage: payload.stage,
        mode: payload.mode,
        mappedPairs: payload.mappedPairs,
        humanoidMatched: payload.humanoidMatched,
        poseError: payload.poseError,
        postPoseError: payload.postPoseError,
        lowerBodyPostError: payload.lowerBodyPostError,
        lowerBodyRotError: payload.lowerBodyRotError,
        rootYawDeg: payload.rootYawDeg,
        yawOffsetDeg: payload.yawOffsetDeg,
        rigProfile: payload.rigProfile,
        rigProfileSaved: payload.rigProfileSaved,
        liveDelta: payload.liveDelta,
      };
    default:
      return payload;
  }
}

function formatConsoleLine(event, payload) {
  if (!payload || typeof payload !== "object") return "";
  switch (event) {
    case "model-loaded":
      return `file=${payload.file || "n/a"} skinnedMeshes=${payload.skinnedMeshes ?? 0} vrmRaw=${payload.vrmHumanoid?.bones ?? 0} vrmNorm=${payload.vrmHumanoid?.normalized ?? 0} vrmDirect=${payload.vrmDirectReady ?? false}`;
    case "model-default-animation":
      return `file=${payload.file || "n/a"} clip=${payload.clip || "n/a"} duration=${payload.duration ?? "n/a"} url=${payload.url || "n/a"}`;
    case "retarget-fail":
      return `stage=${payload.stage || "n/a"} reason=${payload.reason || "unknown"}`;
    case "retarget-topology-fallback":
      return `stage=${payload.stage || "n/a"} attempted=${!!payload.attempted} applied=${!!payload.applied} reason=${payload.reason || "n/a"} renames=${payload.inferredRenames ?? 0}`;
    case "retarget-live-delta":
      return `stage=${payload.stage || "n/a"} mode=${payload.selectedMode || "n/a"} liveDelta=${!!payload.useLiveDelta} auto=${!!payload.autoUseLiveDelta}`;
    case "retarget-root-yaw":
      return `stage=${payload.stage || "n/a"} rawYaw=${payload.rawFacingYawDeg ?? "n/a"} appliedYaw=${payload.appliedYawDeg ?? "n/a"} hipsErr=${payload.hipsYawErrorDeg ?? "n/a"}`;
    case "retarget-summary":
      return `stage=${payload.stage || "n/a"} mode=${payload.mode || "n/a"} mapped=${payload.mappedPairs ?? "n/a"} humanoid=${payload.humanoidMatched || "n/a"} poseError=${payload.poseError ?? "n/a"} postPoseError=${payload.postPoseError ?? "n/a"} legs=${payload.lowerBodyPostError ?? "n/a"} legsRot=${payload.lowerBodyRotError ?? "n/a"} rootYaw=${payload.rootYawDeg ?? "n/a"} yawOffset=${payload.yawOffsetDeg ?? "n/a"} profile=${payload.rigProfile || "n/a"} saved=${!!payload.rigProfileSaved} liveDelta=${!!payload.liveDelta}`;
    default:
      return "";
  }
}

export function createDiag(windowObject = window) {
  let seq = 0;

  function getDiagConsoleMode() {
    const raw = String(windowObject.__vid2modelDiagMode || "minimal").trim().toLowerCase();
    return raw === "verbose" ? "verbose" : "minimal";
  }

  function shouldLogToConsole(event) {
    if (getDiagConsoleMode() === "verbose") return true;
    if (event === "retarget-topology-fallback") {
      return true;
    }
    return DIAG_CONSOLE_EVENTS_MINIMAL.has(event);
  }

  function getDiagFileLoggerConfig() {
    const cfg = windowObject.__vid2modelDiagFileLogger || {};
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
      seq: seq++,
      event,
      payload: sanitizeDiagPayload(payload),
    };
    fetch(cfg.url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(record),
      keepalive: true,
      mode: "cors",
      credentials: "omit",
    }).catch(() => {});
  }

  function diag(event, payload = {}) {
    if (!DIAG_EVENTS.has(event)) return;
    const safePayload = sanitizeDiagPayload(payload);
    if (
      event === "retarget-topology-fallback" &&
      getDiagConsoleMode() !== "verbose" &&
      !safePayload.applied &&
      !safePayload.attempted
    ) {
      sendDiagToFile(event, payload);
      return;
    }
    if (shouldLogToConsole(event)) {
      const line = formatConsoleLine(event, safePayload);
      if (line) {
        console.log(`${DIAG_PREFIX} ${event}: ${line}`);
      } else {
        console.log(DIAG_PREFIX, event, compactConsolePayload(event, safePayload));
      }
    }
    sendDiagToFile(event, payload);
  }

  return {
    diag,
    getDiagFileLoggerConfig,
    getDiagConsoleMode,
  };
}
