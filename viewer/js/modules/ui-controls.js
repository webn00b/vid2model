export function setupViewerUi({ elements, ops }) {
  const {
    fileInput,
    modelInput,
    bvhFileNameEl,
    modelFileNameEl,
    btnLoadDefault,
    btnAutoSetup,
    btnSaveModelSetup,
    btnRetarget,
    btnValidateProfile,
    btnExportProfile,
    btnExportModelAnalysis,
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
    zoomIn,
    zoomOut,
    loadBvhFile,
    loadModelFile,
    play,
    pause,
    stop,
    scrubTo,
    finishScrub,
    resetCamera,
  } = ops;

  fileInput?.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (bvhFileNameEl) bvhFileNameEl.textContent = file.name;
    await loadBvhFile(file);
  });

  modelInput?.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (modelFileNameEl) modelFileNameEl.textContent = file.name;
    loadModelFile(file);
  });

  btnLoadDefault?.addEventListener("click", loadDefault);
  btnAutoSetup?.addEventListener("click", autoSetupModel);
  btnSaveModelSetup?.addEventListener("click", saveModelSetup);
  btnRetarget?.addEventListener("click", retarget);
  btnValidateProfile?.addEventListener("click", validateCurrentRigProfile);
  btnExportProfile?.addEventListener("click", () => exportCurrentRigProfile(true));
  btnExportModelAnalysis?.addEventListener("click", () => exportCurrentModelAnalysis(true));
  btnImportProfile?.addEventListener("click", () => importRigProfileFile());
  if (btnRetargetFab) {
    btnRetargetFab.addEventListener("click", retarget);
  }
  btnZoomIn?.addEventListener("click", zoomIn);
  btnZoomOut?.addEventListener("click", zoomOut);
  btnPlay?.addEventListener("click", play);
  btnPause?.addEventListener("click", pause);
  btnStop?.addEventListener("click", stop);

  timeline?.addEventListener("input", () => {
    const result = scrubTo(Number(timeline.value));
    if (timeEl && result.ok) {
      timeEl.textContent = `${result.time.toFixed(2)} / ${result.duration.toFixed(2)}`;
    }
  });

  timeline?.addEventListener("change", () => {
    finishScrub();
  });

  btnResetCamera?.addEventListener("click", () => {
    resetCamera();
  });
}
