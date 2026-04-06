export function createViewerSourceOverlay({
  scene,
  clearSourceOverlayModule,
  createSourceOverlayModule,
  updateSourceOverlayModule,
  skeletonColor,
  sourcePointColor,
  overlayUpAxis,
  overlayPivot,
  refreshBoneLabels,
  updateSourceAxesDebug,
}) {
  let sourceOverlay = null;

  function getSourceOverlay() {
    return sourceOverlay;
  }

  function hasSourceOverlay() {
    return !!sourceOverlay;
  }

  function clearSourceOverlay() {
    sourceOverlay = clearSourceOverlayModule({ sourceOverlay, scene });
  }

  function updateSourceOverlay() {
    updateSourceOverlayModule({
      sourceOverlay,
      overlayUpAxis,
      overlayPivot,
    });
    updateSourceAxesDebug();
  }

  function createSourceOverlay(skeleton) {
    sourceOverlay = createSourceOverlayModule({
      skeleton,
      scene,
      sourceOverlay,
      skeletonColor,
      sourcePointColor,
      clearSourceOverlay: () => {
        sourceOverlay = clearSourceOverlayModule({ sourceOverlay, scene });
      },
      updateSourceOverlay: (overlay) =>
        updateSourceOverlayModule({
          sourceOverlay: overlay,
          overlayUpAxis,
          overlayPivot,
        }),
    });
    refreshBoneLabels();
  }

  function setSourceOverlayYaw(yaw = 0) {
    if (!sourceOverlay) return false;
    sourceOverlay.overlayYaw = Number.isFinite(yaw) ? yaw : 0;
    return true;
  }

  return {
    getSourceOverlay,
    hasSourceOverlay,
    clearSourceOverlay,
    updateSourceOverlay,
    createSourceOverlay,
    setSourceOverlayYaw,
  };
}
