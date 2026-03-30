export const DIAG_PREFIX = "[vid2model/diag]";
export const DIAG_FILE_LOG_DEFAULT_URL = "http://127.0.0.1:8765/diag";

const DIAG_EVENTS = new Set([
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

export function createDiag(windowObject = window) {
  let seq = 0;

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
    console.log(DIAG_PREFIX, event, payload);
    sendDiagToFile(event, payload);
  }

  return {
    diag,
    getDiagFileLoggerConfig,
  };
}
