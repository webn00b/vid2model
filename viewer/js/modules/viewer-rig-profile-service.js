export function createViewerRigProfileService({
  windowRef,
  storageKey,
  maxEntries,
  statusValues,
  repoManifestUrl,
  getBuiltinRigProfile,
  getRetargetStage,
  getCurrentModelRigFingerprint,
  buildRigProfileSeedForCurrentModel,
  buildSeedCorrectionSummary,
}) {
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
      validationStatus: statusValues.has(validationStatus) ? validationStatus : "draft",
      autoSaved: entry.autoSaved !== false,
      updatedAt: entry.updatedAt || null,
      validatedAt: entry.validatedAt || null,
    };
  }

  function readStoredRigProfiles() {
    try {
      const raw = windowRef.localStorage?.getItem(storageKey);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed.map((entry) => normalizeRigProfileEntry(entry)).filter(Boolean) : [];
    } catch {
      return [];
    }
  }

  function writeStoredRigProfiles(entries) {
    try {
      windowRef.localStorage?.setItem(storageKey, JSON.stringify(entries));
      return true;
    } catch {
      return false;
    }
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
    repoRigProfileManifestPromise = fetch(repoManifestUrl)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
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
    const url = new URL(entry.path, repoManifestUrl).href;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    const profile = normalizeRigProfileEntry(payload?.profile || payload);
    if (!profile) throw new Error("Invalid rig profile payload");
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

  function buildRigProfileState(profile, {
    modelFingerprint = "",
    modelLabel = "",
    stage = "",
    saved = false,
    resolvedFrom = "",
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
      autoGenerated: !!profile?.autoGenerated,
      basedOnModelAnalysis: !!profile?.basedOnModelAnalysis,
      inferredCorrections: Array.isArray(profile?.inferredCorrections)
        ? [...profile.inferredCorrections]
        : buildSeedCorrectionSummary(profile),
      profileId: String(profile?.id || ""),
      updatedAt: profile?.updatedAt || null,
      validatedAt: profile?.validatedAt || null,
      resolvedFrom: String(resolvedFrom || "none"),
    };
  }

  function publishRigProfileState(state) {
    latestRigProfileState = {
      ...latestRigProfileState,
      ...(state || {}),
    };
    windowRef.__vid2modelRigProfileState = { ...latestRigProfileState };
    return windowRef.__vid2modelRigProfileState;
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
    if (modelFingerprint && modelFingerprint === getCurrentModelRigFingerprint()) {
      const seed = buildRigProfileSeedForCurrentModel(stage);
      if (seed) return seed;
    }
    return builtin;
  }

  function saveRigProfile(entry, options = {}) {
    if (!entry?.modelFingerprint || !entry?.stage) return false;
    const validationStatus = String(options.validationStatus || entry.validationStatus || "draft").trim().toLowerCase();
    const normalizedStatus = statusValues.has(validationStatus) ? validationStatus : "draft";
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
    if (Number.isFinite(existingError) && Number.isFinite(nextError) && existingError <= nextError + 1e-6) {
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
    return writeStoredRigProfiles(rows.slice(0, maxEntries));
  }

  async function ensureRepoRigProfilesForModel({
    modelFingerprint = getCurrentModelRigFingerprint(),
    modelLabel = "",
    onActiveProfileLoaded = null,
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
    if (loaded.length && typeof onActiveProfileLoaded === "function") {
      const activeProfile = loadRigProfile(modelFingerprint, getRetargetStage(), modelLabel);
      onActiveProfileLoaded(activeProfile);
    }
    return loaded;
  }

  function setLatestRigProfileCandidate(candidate) {
    latestRigProfileCandidate = candidate || null;
    return latestRigProfileCandidate;
  }

  function getLatestRigProfileCandidate() {
    return latestRigProfileCandidate;
  }

  function validateCurrentRigProfile({ setStatus } = {}) {
    const candidate = latestRigProfileCandidate;
    if (!candidate?.modelFingerprint || !candidate?.stage) {
      setStatus?.("No rig profile candidate yet. Retarget the model first.");
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
      resolvedFrom: validated?.source || "none",
    }));
    if (!saved) {
      setStatus?.(`Rig profile already validated for ${candidate.stage}.`);
      return false;
    }
    setStatus?.(`Rig profile validated for ${candidate.modelLabel || "model"} [${candidate.stage}].`);
    console.log("[vid2model/diag] rig-profile-validated", validated);
    return true;
  }

  function getBestPortableRigProfile({
    modelFingerprint = getCurrentModelRigFingerprint(),
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
    if (override && String(override).trim()) return String(override).trim();
    return `${(payload.modelLabel || "model").replace(/\.[^.]+$/, "") || "model"}.${payload.stage}.rig-profile.json`;
  }

  function buildRegisterRigProfileCommand({ inputPath = "", filename = "", allowDraft = false, payload = null } = {}) {
    const activePayload = payload || windowRef.__vid2modelRigProfileExport || buildPortableRigProfilePayload(
      getBestPortableRigProfile({ allowDraft })
    );
    if (!activePayload) return "";
    const suggestedFilename = buildSuggestedRigProfileDownloadName(activePayload, filename);
    const resolvedInput = String(inputPath || "").trim() || `/path/to/${suggestedFilename}`;
    return `python3 tools/register_rig_profile.py --input ${resolvedInput}`;
  }

  function exportCurrentRigProfile(download = false, filename = "", allowDraft = false, { setStatus } = {}) {
    const profile = getBestPortableRigProfile({ allowDraft });
    if (!profile) {
      setStatus?.(allowDraft ? "No rig profile available for export." : "No validated rig profile yet. Validate a profile first.");
      return null;
    }
    const payload = buildPortableRigProfilePayload(profile);
    if (!payload) {
      setStatus?.("Failed to build rig profile export.");
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
    windowRef.__vid2modelRigProfileExport = payload;
    windowRef.__vid2modelRigProfileExportCommand = registerCommand;
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
    setStatus?.(`Rig profile exported [${payload.stage}] (${payload.modelLabel || payload.modelFingerprint}).`);
    console.log("[vid2model/diag] rig-profile-export", payload);
    console.log("[vid2model/diag] rig-profile-register-command", registerCommand);
    return payload;
  }

  function importRigProfilePayload(payload, {
    modelRigFingerprint = getCurrentModelRigFingerprint(),
    autoRetarget = true,
    sourceResult = null,
    modelSkinnedMesh = null,
    setStatus,
    onAutoRetarget = null,
  } = {}) {
    const format = String(payload?.format || "").trim();
    const profile = normalizeRigProfileEntry(payload?.profile || payload);
    if (!profile) throw new Error("Invalid rig profile payload");
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
      resolvedFrom: imported?.source || "none",
    }));
    const matchesCurrentModel =
      imported &&
      imported.modelFingerprint === modelRigFingerprint &&
      imported.stage === getRetargetStage();
    setStatus?.(
      matchesCurrentModel
        ? `Rig profile imported and ready [${imported.stage}].`
        : `Rig profile imported for ${imported?.modelLabel || imported?.modelFingerprint || "model"}.`
    );
    console.log("[vid2model/diag] rig-profile-import", imported);
    if (matchesCurrentModel && autoRetarget && sourceResult && modelSkinnedMesh) {
      onAutoRetarget?.();
    }
    return imported;
  }

  return {
    readStoredRigProfiles,
    normalizeRigProfileEntry,
    findStoredRigProfiles,
    findRepoRigProfiles,
    ensureRepoRigProfilesForModel,
    buildRigProfileState,
    publishRigProfileState,
    getLatestRigProfileState: () => ({ ...latestRigProfileState }),
    setLatestRigProfileCandidate,
    getLatestRigProfileCandidate,
    loadRigProfile,
    saveRigProfile,
    validateCurrentRigProfile,
    getBestPortableRigProfile,
    buildPortableRigProfilePayload,
    buildSuggestedRigProfileDownloadName,
    buildRegisterRigProfileCommand,
    exportCurrentRigProfile,
    importRigProfilePayload,
    listRepoRigProfiles: (fingerprint = getCurrentModelRigFingerprint(), stage = "") =>
      (stage
        ? findRepoRigProfiles(fingerprint, stage)
        : repoRigProfiles.filter((entry) => !fingerprint || entry.modelFingerprint === fingerprint)),
    listRigProfiles: (fingerprint = getCurrentModelRigFingerprint(), stage = "") =>
      (stage
        ? findStoredRigProfiles(fingerprint, stage)
        : readStoredRigProfiles().filter((entry) => !fingerprint || entry.modelFingerprint === fingerprint)),
  };
}
