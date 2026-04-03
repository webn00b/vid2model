export function setupViewerUi({ elements, ops }) {
  const {
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
  } = elements;

  const {
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
    setIsScrubbing,
    setIsPlaying,
    getPlaybackRefs,
    applyLiveRetargetPose,
    applyBoneLengthCalibration,
    applyFingerLengthCalibration,
    alignModelHipsToSource,
    resetCamera,
  } = ops;

  fileInput?.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (bvhFileNameEl) bvhFileNameEl.textContent = file.name;
    setStatus(`Loading ${file.name} ...`);
    const text = await file.text();
    loadBvhText(text, file.name);
  });

  modelInput?.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (modelFileNameEl) modelFileNameEl.textContent = file.name;
    loadModelFile(file);
  });

  btnLoadDefault?.addEventListener("click", loadDefault);
  btnRetarget?.addEventListener("click", applyBvhToModel);
  btnValidateProfile?.addEventListener("click", validateCurrentRigProfile);
  btnExportProfile?.addEventListener("click", () => exportCurrentRigProfile(true));
  btnImportProfile?.addEventListener("click", () => importRigProfileFile());
  if (btnRetargetFab) {
    btnRetargetFab.addEventListener("click", applyBvhToModel);
  }
  btnZoomIn?.addEventListener("click", () => zoomBy(0.85));
  btnZoomOut?.addEventListener("click", () => zoomBy(1.2));

  btnPlay?.addEventListener("click", () => {
    const playback = getPlaybackRefs();
    if (!playback.mixer && !playback.modelMixers.length) return;
    setIsPlaying(true);
    if (playback.currentAction) playback.currentAction.paused = false;
    for (const action of playback.modelActions) {
      action.paused = false;
    }
    setStatus("Playback: play");
  });

  btnPause?.addEventListener("click", () => {
    const playback = getPlaybackRefs();
    if (!playback.mixer && !playback.modelMixers.length) return;
    setIsPlaying(false);
    if (playback.currentAction) playback.currentAction.paused = true;
    for (const action of playback.modelActions) {
      action.paused = true;
    }
    setStatus("Playback: pause");
  });

  btnStop?.addEventListener("click", () => {
    const playback = getPlaybackRefs();
    if (!playback.mixer && !playback.modelMixers.length) return;
    setIsPlaying(false);
    if (playback.currentAction) playback.currentAction.paused = true;
    for (const action of playback.modelActions) {
      action.paused = true;
    }
    if (playback.mixer) playback.mixer.setTime(0);
    for (const mix of playback.modelMixers) {
      mix.setTime(0);
    }
    if (playback.liveRetarget) {
      applyLiveRetargetPose(playback.liveRetarget);
    }
    if (playback.bodyLengthCalibration && !playback.liveRetarget) {
      applyBoneLengthCalibration(playback.bodyLengthCalibration);
    }
    if (playback.armLengthCalibration && !playback.liveRetarget) {
      applyBoneLengthCalibration(playback.armLengthCalibration);
    }
    if (playback.fingerLengthCalibration && !playback.liveRetarget) {
      applyFingerLengthCalibration(playback.fingerLengthCalibration);
    }
    alignModelHipsToSource(false);
    updateTimelineUi(0);
    setStatus("Playback: stop");
  });

  timeline?.addEventListener("input", () => {
    const duration = getActiveDuration();
    const playback = getPlaybackRefs();
    if (!duration || (!playback.mixer && !playback.modelMixers.length)) return;

    setIsScrubbing(true);
    const t = Math.max(0, Math.min(duration, Number(timeline.value)));
    if (playback.mixer) playback.mixer.setTime(t);
    for (const mix of playback.modelMixers) {
      mix.setTime(t);
    }
    if (playback.liveRetarget) {
      applyLiveRetargetPose(playback.liveRetarget);
    }
    if (playback.bodyLengthCalibration && !playback.liveRetarget) {
      applyBoneLengthCalibration(playback.bodyLengthCalibration);
    }
    if (playback.armLengthCalibration && !playback.liveRetarget) {
      applyBoneLengthCalibration(playback.armLengthCalibration);
    }
    if (playback.fingerLengthCalibration && !playback.liveRetarget) {
      applyFingerLengthCalibration(playback.fingerLengthCalibration);
    }
    alignModelHipsToSource(false);
    if (timeEl) {
      timeEl.textContent = `${t.toFixed(2)} / ${duration.toFixed(2)}`;
    }
    setStatus(`Scrub: ${t.toFixed(2)}s`);
  });

  timeline?.addEventListener("change", () => {
    setIsScrubbing(false);
  });

  btnResetCamera?.addEventListener("click", () => {
    resetCamera();
  });
}
