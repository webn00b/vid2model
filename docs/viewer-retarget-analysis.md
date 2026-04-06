# Viewer Retarget, Rig-Profile, and Model-Fit Analysis

This note maps the viewer-side path that turns a processed source clip into model motion and highlights where skeleton-to-model mismatch can still appear after the Python export is already "clean".

## Entry Points

- `viewer/js/main.js`
  - `autoSetupModel()` seeds a setup from live model analysis, repo profiles, local profiles, and built-in fallbacks.
  - `saveModelSetup()` persists the current setup as a local draft.
  - `applyBvhToModel()` is the main retarget entry point. It builds the stage clip, resolves mapping, chooses a retarget mode, applies alignment and calibration, then publishes diagnostics.
- `viewer/js/modules/viewer-model-analysis.js`
  - Produces analysis rows and seed hints used when there is no validated profile yet.
- `viewer/js/modules/viewer-runtime-diagnostics.js`
  - Publishes the post-selection and post-calibration diagnostics needed to explain mismatch cases.

## Rig-Profile Priority

The viewer-side ownership of mapping and mode selection is intentionally layered:

1. Validated local profile.
2. Validated repo profile.
3. Local draft profile from a previous successful setup.
4. In-memory auto-analysis seed from `viewer-model-analysis`.
5. Built-in fallback heuristics.

This priority matters because some mismatch cases are not caused by the source clip at all; they come from stale or low-confidence target-side assumptions winning before a stronger mapping is available.

## Retarget Path in `applyBvhToModel()`

The main viewer flow is:

1. Load source result, active stage, and cached rig-profile context.
2. Choose the retarget stage clip and optional canonical filtering.
3. Build the initial target-to-source name map.
4. Expand the map with topology fallback and rig-profile overrides.
5. Collect multiple retarget attempts:
   - `skeletonutils-skinnedmesh`
   - `skeletonutils-skinnedmesh-reversed`
   - `skeletonutils-root`
   - `skeletonutils-root-reversed`
   - rename-based fallbacks for `bones` and `object` syntax
6. Score and select the best attempt using resolved track count, motion probe, pose-match error, and preferred-mode hints.
7. Build the final execution mode:
   - `SkeletonUtils` clip retarget
   - live delta retarget
   - VRM-direct flow when applicable
8. Evaluate root-yaw ownership and apply the selected correction.
9. Apply target-side calibration passes:
   - body length calibration
   - finger length calibration
   - arm refinement calibration
   - hips alignment / overlay alignment
10. Publish diagnostics, persist draft profile data, and expose debug state.

## Model-Fit-Sensitive Modules

- `viewer/js/main.js`
  - Orchestrates stage choice, profile priority, mode selection, and post-retarget calibration order.
- `viewer/js/modules/viewer-retarget-attempts.js`
  - Defines which candidate retarget modes are even considered and how the final attempt is chosen.
- `viewer/js/modules/retarget-eval.js`
  - Filters tracks, builds animation bindings, probes visible motion, and computes canonical pose-match error.
- `viewer/js/modules/retarget-analysis.js`
  - Builds root-yaw candidates and summarizes source/target orientation evidence.
- `viewer/js/modules/root-yaw-contract.js`
  - Prevents the viewer from reintroducing large source flips when the clip already looks centered after Python export.
- `viewer/js/modules/viewer-topology-fallback.js`
  - Supplies mapping fallback when names are incomplete or non-standard.
- `viewer/js/modules/canonical-bone-map.js`
  - Prefers primary deform bones over helper/socket/end/control bones for canonical role ownership.
- `viewer/js/modules/retarget-calibration.js`
  - Applies target-side segment-length and arm-refinement corrections when clip data alone cannot fit the model well.
- `viewer/js/modules/viewer-alignment.js`
  - Aligns hips, overlay placement, display scale, and world-space offsets between source and model.
- `viewer/js/modules/viewer-runtime-diagnostics.js`
  - Emits `retarget-summary`, `retarget-alignment`, `retarget-limbs`, and other logs used to confirm where mismatch persists.

## Calibration Hooks

The main calibration hooks are target-side and happen after a candidate retarget path already exists:

- `buildBodyLengthCalibration()`
  - Estimates a global rig scale from canonical segments, then clamps per-bone body scaling into conservative ranges.
- `buildFingerLengthCalibration()`
  - Repeats the same idea for fingers with wider limits for distal chains.
- `buildArmRefinementCalibration()`
  - Searches per-side arm multipliers and keeps them only if alignment error improves by a real margin.
- `alignModelHipsToSource()` / source overlay helpers
  - Fix final world-space offset, hips placement, and display synchronization after the retarget mode has already been chosen.

These hooks improve fit, but they can also mask upstream mapping problems if diagnostics are not read together.

## Likely Retarget-Side Mismatch Causes

- Wrong profile wins early.
  - A stale validated or draft profile can override better live analysis and lock the viewer into the wrong names or preferred mode.
- Candidate scoring prefers the wrong retarget mode.
  - High resolved-track count does not guarantee perceptual fit; two attempts can bind many tracks but differ materially in pose-match or lower-body stability.
- Non-standard rigs expose ambiguous canonical ownership.
  - Helper, socket, end, and control bones can still distort mapping quality when a model uses unusual naming or duplicate chains.
- Root-yaw correction fights the imported clip.
  - Even after the recent contract fix, borderline clips can still be sensitive when facing evidence from source and model is weak or asymmetric.
- Target proportions differ more than the calibration windows expect.
  - Segment clamps are intentionally conservative, so stylized or compact rigs may remain visibly off even after calibration.
- Alignment is correct in local pose space but wrong in world space.
  - Hips/root placement and overlay sync can make a retarget look broken even when the underlying chain rotations are mostly correct.
- Cleanup happens in the wrong order for a specific model.
  - A usable result can degrade if yaw, hips alignment, and length calibration are applied in a sequence that amplifies an earlier approximation.

## Practical Diagnostic Read Order

When inspecting a mismatch case, the fastest path is:

1. Check `retarget-summary` for selected mode, mapped pairs, pose error, and whether a cached profile won.
2. Check `retarget-root-yaw` and `retarget-alignment` to separate orientation problems from mapping problems.
3. Check `selectionDebug` / `attemptDebug` to see whether a different candidate was nearly as good.
4. Check `retarget-limbs` plus body/finger/arm calibration reports for proportion-driven failures.

This division is useful for the final synthesis task because it separates source-side cleanup issues from target-side mapping, profile, and calibration issues.
