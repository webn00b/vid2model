# Viewer Modules Refactor Plan

This note scopes `viewer/js/modules` cleanup for `vid2model-1g6` without changing runtime behavior.

## Current Inventory

`viewer/js/modules` currently contains 29 flat modules.

### 1. Foundation / Pure Helpers

These are mostly data or pure helper modules and are the safest refactor surface:

- `bone-utils.js`
- `canonical-bone-map.js`
- `retarget-chain-utils.js`
- `retarget-constants.js`
- `retarget-plan-utils.js`
- `root-yaw-contract.js`

### 2. Retarget Core

These contain clip mapping, candidate evaluation, calibration, and retarget execution logic:

- `retarget-helpers.js`
- `retarget-analysis.js`
- `retarget-eval.js`
- `retarget-calibration.js`
- `retarget-live.js`
- `retarget-vrm.js`
- `viewer-retarget-attempts.js`

### 3. Viewer Runtime Services

These are viewer-specific services that operate on loaded models, source overlays, diagnostics, or debug state:

- `viewer-model-loader.js`
- `viewer-parsed-model.js`
- `viewer-model-analysis.js`
- `viewer-skeleton-profile.js`
- `viewer-topology-fallback.js`
- `viewer-alignment.js`
- `viewer-runtime-diagnostics.js`
- `viewer-chain-diagnostics.js`
- `viewer-source-overlay.js`
- `viewer-source-axes-debug.js`
- `viewer-skeleton-debug.js`
- `viewer-bone-labels.js`
- `diag.js`
- `default-animation.js`

### 4. Profile / Config Data

- `rig-profiles.js`

This one behaves more like repository-backed configuration than like a runtime service.

### 5. UI Layer

- `ui-controls.js`

This is the only clearly UI-facing module, but it currently knows too much about playback, calibration re-application, and retarget state shape.

## Current Structural Smells

### `main.js` is the real service container

[`viewer/js/main.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/main.js) imports almost the entire module directory directly and still owns:

- DOM wiring
- runtime state
- playback state
- profile priority
- retarget orchestration
- diagnostics publication
- calibration application order

That means many modules are "helpers under a god-orchestrator" rather than independent subsystems.

### Export style is inconsistent

The folder mixes:

- pure function modules (`retarget-eval.js`, `retarget-helpers.js`)
- factory-style toolkits (`createViewerAlignmentTools`, `createViewerModelAnalysisTools`)
- data registries (`rig-profiles.js`, `retarget-constants.js`)
- direct UI installers (`setupViewerUi`)

The API style is not wrong, but the inconsistency makes it harder to predict where a new concern belongs.

### Viewer-specific and retarget-core concerns are mixed together

Examples:

- `viewer-topology-fallback.js` contains target-bone naming heuristics, but also leans on retarget mapping logic.
- `viewer-retarget-attempts.js` is conceptually retarget-core selection, yet it is named as a viewer module.
- `viewer-runtime-diagnostics.js` publishes retarget diagnostics even though much of the payload is retarget-core state.

### UI controls know runtime repair steps

[`viewer/js/modules/ui-controls.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/ui-controls.js) is not just binding buttons. It also knows that stop/scrub must:

- reapply live retarget pose
- reapply body/arm/finger calibration
- re-align hips

That is runtime orchestration knowledge and should not live in the UI boundary.

## Target Responsibility Boundaries

### `core/retarget` boundary

Goal: pure or mostly pure motion/rig logic with minimal DOM/viewer assumptions.

Should own:

- canonical bone naming
- track parsing and clip filtering
- mapping construction
- attempt scoring and selection
- root-yaw candidate policy
- calibration plan construction
- live retarget pair and pose math

Candidate modules:

- `bone-utils.js`
- `canonical-bone-map.js`
- `retarget-chain-utils.js`
- `retarget-constants.js`
- `retarget-plan-utils.js`
- `root-yaw-contract.js`
- `retarget-helpers.js`
- `retarget-analysis.js`
- `retarget-eval.js`
- `retarget-calibration.js`
- `retarget-live.js`
- `retarget-vrm.js`
- `viewer-retarget-attempts.js` -> should be renamed/moved toward retarget-core naming later

### `viewer/runtime` boundary

Goal: manage scene objects, loaded model state, overlays, diagnostics, and viewer-specific adaptation.

Should own:

- model loading and parsed-model application
- model analysis export
- skeleton profile export
- topology fallback that depends on real target skeletons
- alignment and overlay synchronization
- runtime diagnostics sinks
- debug visuals

Candidate modules:

- `viewer-model-loader.js`
- `viewer-parsed-model.js`
- `viewer-model-analysis.js`
- `viewer-skeleton-profile.js`
- `viewer-topology-fallback.js`
- `viewer-alignment.js`
- `viewer-runtime-diagnostics.js`
- `viewer-chain-diagnostics.js`
- `viewer-source-overlay.js`
- `viewer-source-axes-debug.js`
- `viewer-skeleton-debug.js`
- `viewer-bone-labels.js`
- `diag.js`
- `default-animation.js`

### `viewer/profile` boundary

Goal: treat rig profile data and profile selection as a dedicated concern rather than scattered conditionals in `main.js`.

Should own:

- built-in profile registry
- profile storage schema and versioning
- profile priority rules
- repo/local/draft/validated lookup policy
- profile freshness or drift heuristics later

Candidate modules:

- `rig-profiles.js`
- profile selection logic currently embedded in [`viewer/js/main.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/main.js)

### `viewer/ui` boundary

Goal: UI modules only translate user gestures into operations exposed by runtime/controller APIs.

Should own:

- DOM event binding
- element state updates
- invoking high-level commands like `play()`, `pause()`, `scrubTo(t)`, `retarget()`

Should not own:

- recalibration logic
- direct access to mixer internals
- direct knowledge of live-retarget repair sequence

## Recommended Phased Plan

### Phase 1. Documented boundary cleanup with zero moves

Low risk, immediate value.

- Keep file layout unchanged.
- Introduce naming conventions in docs and comments:
  - `core` helpers
  - `runtime` services
  - `ui` adapter
  - `profile` registry/service
- Normalize mental model before any code motion.

### Phase 2. Extract one viewer controller facade from `main.js`

Highest leverage refactor.

Create a single runtime/controller object in `main.js` or a new module that owns:

- playback commands
- retarget command
- calibration re-application
- alignment refresh
- profile save/validate/export actions

Then make `ui-controls.js` call only that facade.

Expected benefit:
- cuts the biggest UI/runtime coupling first
- preserves behavior because underlying functions can stay where they are

### Phase 3. Split profile policy from main orchestration

Move profile lookup and selection policy out of `main.js` into a dedicated profile service module.

This service should answer questions like:

- which profile wins
- why it won
- whether it is built-in, repo, local draft, or validated
- which metadata should be published in diagnostics

Expected benefit:
- reduces retarget-path branching inside `applyBvhToModel()`
- makes future freshness/drift rules testable in isolation

### Phase 4. Rename or regroup retarget-core modules

After facade extraction, regroup names without behavior change.

Best candidates:

- move `viewer-retarget-attempts.js` toward a retarget-core name
- separate pure diagnostics math from viewer diagnostic publishing
- keep `retarget-live.js` math distinct from scene/overlay side effects

Expected benefit:
- better discoverability for future retarget work
- less confusion about which modules are safe to unit test without viewer scene setup

### Phase 5. Optional physical directory split

Only after the API boundaries are already stable.

Possible future layout:

- `viewer/js/modules/core/`
- `viewer/js/modules/runtime/`
- `viewer/js/modules/profile/`
- `viewer/js/modules/ui/`

This should be last, not first, to avoid large noisy moves before boundaries are proven.

## Safe First Work Items

If we start implementation after approval, the safest first slice is:

1. introduce a viewer controller facade for playback/retarget/profile actions,
2. slim down `ui-controls.js` to only call facade methods,
3. extract profile priority logic out of `main.js`,
4. only then consider file moves or renames.

## Non-Goals For The First Refactor Pass

- no behavior changes in retarget output
- no renaming of diagnostics fields
- no large folder move in the first implementation slice
- no VRM-specific logic rewrite

That keeps the refactor reviewable and makes regression testing much simpler.
