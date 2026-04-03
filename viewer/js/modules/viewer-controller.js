export function createViewerController({
  setStatus,
  loadBvhText,
  loadModelFile,
  loadDefault,
  autoSetupModel,
  saveModelSetup,
  retarget,
  validateCurrentRigProfile,
  exportCurrentRigProfile,
  exportCurrentModelAnalysis,
  importRigProfileFile,
  zoomBy,
  resetCamera,
  getActiveDuration,
  updateTimelineUi,
  getPlaybackState,
  setIsPlaying,
  setIsScrubbing,
  applyLiveRetargetPose,
  applyBoneLengthCalibration,
  applyFingerLengthCalibration,
  alignModelHipsToSource,
}) {
  async function loadBvhFile(file) {
    if (!file) return;
    setStatus(`Loading ${file.name} ...`);
    const text = await file.text();
    loadBvhText(text, file.name);
  }

  function play() {
    const playback = getPlaybackState();
    if (!playback.mixer && !playback.modelMixers.length) return false;
    setIsPlaying(true);
    if (playback.currentAction) playback.currentAction.paused = false;
    for (const action of playback.modelActions) {
      action.paused = false;
    }
    setStatus("Playback: play");
    return true;
  }

  function pause() {
    const playback = getPlaybackState();
    if (!playback.mixer && !playback.modelMixers.length) return false;
    setIsPlaying(false);
    if (playback.currentAction) playback.currentAction.paused = true;
    for (const action of playback.modelActions) {
      action.paused = true;
    }
    setStatus("Playback: pause");
    return true;
  }

  function reapplyPlaybackSideEffects(playback) {
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
  }

  function stop() {
    const playback = getPlaybackState();
    if (!playback.mixer && !playback.modelMixers.length) return false;
    setIsPlaying(false);
    if (playback.currentAction) playback.currentAction.paused = true;
    for (const action of playback.modelActions) {
      action.paused = true;
    }
    if (playback.mixer) playback.mixer.setTime(0);
    for (const mix of playback.modelMixers) {
      mix.setTime(0);
    }
    reapplyPlaybackSideEffects(playback);
    updateTimelineUi(0);
    setStatus("Playback: stop");
    return true;
  }

  function scrubTo(time) {
    const duration = getActiveDuration();
    const playback = getPlaybackState();
    if (!duration || (!playback.mixer && !playback.modelMixers.length)) {
      return { ok: false, time: 0, duration };
    }
    setIsScrubbing(true);
    const nextTime = Math.max(0, Math.min(duration, Number(time) || 0));
    if (playback.mixer) playback.mixer.setTime(nextTime);
    for (const mix of playback.modelMixers) {
      mix.setTime(nextTime);
    }
    reapplyPlaybackSideEffects(playback);
    updateTimelineUi(nextTime);
    setStatus(`Scrub: ${nextTime.toFixed(2)}s`);
    return { ok: true, time: nextTime, duration };
  }

  function finishScrub() {
    setIsScrubbing(false);
  }

  return {
    loadBvhFile,
    loadModelFile,
    loadDefault,
    autoSetupModel,
    saveModelSetup,
    retarget,
    validateCurrentRigProfile,
    exportCurrentRigProfile,
    exportCurrentModelAnalysis,
    importRigProfileFile,
    zoomIn: () => zoomBy(0.85),
    zoomOut: () => zoomBy(1.2),
    resetCamera,
    play,
    pause,
    stop,
    scrubTo,
    finishScrub,
  };
}
