export function setupViewerUi({ elements, ops, getIsPlaying, getSceneBg, setSceneBg }) {
  const {
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
  } = elements;

  const {
    loadDefault,
    autoSetupModel,
    saveModelSetup,
    retarget,
    validateCurrentRigProfile,
    exportCurrentRigProfile,
    exportCurrentModelAnalysis,
    importRigProfileFile,
    loadBvhFile,
    loadBvhFileByName,
    loadModelFile,
    play,
    pause,
    stop,
    scrubTo,
    finishScrub,
    resetCamera,
    stepFrames,
    toggleSkeleton,
    toggleModel,
  } = ops;

  // --- File inputs ---
  fileInput?.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (bvhFileNameEl) bvhFileNameEl.textContent = file.name;
    await loadBvhFile(file);
  });

  animationList?.addEventListener("change", async (e) => {
    const filename = e.target.value;
    if (!filename) return;
    if (bvhFileNameEl) bvhFileNameEl.textContent = filename;
    await loadBvhFileByName(filename);
  });

  modelInput?.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (modelFileNameEl) modelFileNameEl.textContent = file.name;
    loadModelFile(file);
  });

  // --- Main buttons ---
  btnLoadDefault?.addEventListener("click", loadDefault);
  btnAutoSetup?.addEventListener("click", autoSetupModel);
  btnSaveModelSetup?.addEventListener("click", saveModelSetup);
  btnRetarget?.addEventListener("click", retarget);
  btnValidateProfile?.addEventListener("click", validateCurrentRigProfile);
  btnExportProfile?.addEventListener("click", () => exportCurrentRigProfile(true));
  btnExportModelAnalysis?.addEventListener("click", () => exportCurrentModelAnalysis(true));
  btnImportProfile?.addEventListener("click", () => importRigProfileFile());
  btnRetargetFab?.addEventListener("click", retarget);

  // --- Play toggle ---
  function updatePlayButton(playing) {
    if (!btnPlayToggle) return;
    btnPlayToggle.innerHTML = playing ? "&#9646;&#9646; Pause" : "&#9654; Play";
  }

  btnPlayToggle?.addEventListener("click", () => {
    const playing = getIsPlaying?.();
    if (playing) {
      pause();
      updatePlayButton(false);
    } else {
      play();
      updatePlayButton(true);
    }
  });

  btnStop?.addEventListener("click", () => {
    stop();
    updatePlayButton(false);
  });

  // --- Visibility toggles ---
  function _readVisParam(key) {
    const param = new URLSearchParams(location.search).get(key);
    if (param !== null) return param !== "0";
    return localStorage.getItem("vid2model." + key) !== "0";
  }

  if (btnToggleSkeleton) {
    const vis = _readVisParam("showSkeleton");
    btnToggleSkeleton.textContent = vis ? "Hide Skeleton" : "Show Skeleton";
    btnToggleSkeleton.addEventListener("click", () => {
      const visible = toggleSkeleton?.();
      btnToggleSkeleton.textContent = visible ? "Hide Skeleton" : "Show Skeleton";
    });
  }

  if (btnToggleModel) {
    const vis = _readVisParam("showModel");
    btnToggleModel.textContent = vis ? "Hide Model" : "Show Model";
    btnToggleModel.addEventListener("click", () => {
      const visible = toggleModel?.();
      btnToggleModel.textContent = visible ? "Hide Model" : "Show Model";
    });
  }

  // --- Timeline ---
  timeline?.addEventListener("input", () => {
    const result = scrubTo(Number(timeline.value));
    if (timeEl && result.ok) {
      timeEl.textContent = `${result.time.toFixed(2)} / ${result.duration.toFixed(2)}`;
    }
  });

  timeline?.addEventListener("change", () => {
    finishScrub();
  });

  // --- Camera ---
  btnResetCamera?.addEventListener("click", resetCamera);

  // --- Dark mode ---
  const root = document.documentElement;
  const DARK_KEY = "vid2model.darkMode";
  if (localStorage.getItem(DARK_KEY) === "1") {
    root.setAttribute("data-theme", "dark");
    setSceneBg?.("dark");
  }
  btnDarkToggle?.addEventListener("click", () => {
    const isDark = root.getAttribute("data-theme") === "dark";
    root.setAttribute("data-theme", isDark ? "light" : "dark");
    localStorage.setItem(DARK_KEY, isDark ? "0" : "1");
    setSceneBg?.(isDark ? "light" : "dark");
  });

  // --- Tools collapsible ---
  btnToolsToggle?.addEventListener("click", () => {
    const open = toolsGroup?.classList.toggle("open");
    if (btnToolsToggle) {
      btnToolsToggle.innerHTML = open ? "Tools &#9650;" : "Tools &#9660;";
    }
  });

  // --- Keyboard shortcuts ---
  document.addEventListener("keydown", (e) => {
    // Ignore when typing in inputs
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;

    switch (e.code) {
      case "Space": {
        e.preventDefault();
        const playing = getIsPlaying?.();
        if (playing) { pause(); updatePlayButton(false); }
        else { play(); updatePlayButton(true); }
        break;
      }
      case "ArrowLeft": {
        e.preventDefault();
        const r = stepFrames?.(-1);
        if (r?.ok && timeEl) timeEl.textContent = `${r.time.toFixed(2)} / ${r.duration.toFixed(2)}`;
        break;
      }
      case "ArrowRight": {
        e.preventDefault();
        const r = stepFrames?.(1);
        if (r?.ok && timeEl) timeEl.textContent = `${r.time.toFixed(2)} / ${r.duration.toFixed(2)}`;
        break;
      }
      case "KeyR":
        if (!e.metaKey && !e.ctrlKey) { e.preventDefault(); resetCamera?.(); }
        break;
    }
  });

  return { updatePlayButton };
}
