# Skeleton-to-Model Mismatch Hypotheses and Next Steps

This note synthesizes the Python source-pipeline analysis and the viewer retarget analysis into a short diagnosis for `vid2model-e5c.3`.

## Prioritized Hypotheses

### 1. Split ownership of orientation and stabilization still creates borderline double-correction cases

Most likely impact: high.

Why:
- Python already canonicalizes pose, applies pose corrections, cleanup, and root-yaw normalization before export.
- Viewer still evaluates source-facing evidence, chooses root-yaw candidates, aligns hips, and may apply target-side corrections that visually resemble a second stabilization pass.
- The recent root-yaw contract reduced the worst case, but borderline clips can still degrade when source-side normalization and viewer-side alignment are both "reasonable" in isolation.

Primary files/functions:
- [`vid2model_lib/pipeline_motion_transforms.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_motion_transforms.py)
  - `normalize_motion_root_yaw(...)`
  - `apply_manual_root_yaw_offset(...)`
- [`vid2model_lib/pipeline_cleanup.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_cleanup.py)
  - `cleanup_pose_frames(...)`
- [`viewer/js/modules/retarget-analysis.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/retarget-analysis.js)
- [`viewer/js/modules/root-yaw-contract.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/root-yaw-contract.js)
- [`viewer/js/modules/viewer-alignment.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/viewer-alignment.js)
  - `alignModelHipsToSource(...)`

Diagnostics to compare:
- `diagnostics.root_yaw`
- `diagnostics.source_stages.motion`
- viewer `retarget-root-yaw`
- viewer `retarget-alignment`

### 2. Source cleanup and pose correction can still over-regularize motion before retarget even starts

Most likely impact: high.

Why:
- We already fixed one aggressive limb-floor issue, which strongly suggests this class of failure is real, not theoretical.
- Cleanup and correction stages still contain segment constraints, pelvis stabilization, smoothing, and IK logic that can erase model-specific cues needed later by the viewer.
- Once exported motion is already "generic humanoid enough", viewer calibration can only compensate, not reconstruct the lost structure.

Primary files/functions:
- [`vid2model_lib/pipeline_retarget.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_retarget.py)
  - `apply_pose_corrections(...)`
- [`vid2model_lib/pipeline_auto_pose.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_auto_pose.py)
  - `resolve_auto_pose_corrections(...)`
- [`vid2model_lib/pipeline_cleanup.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_cleanup.py)
  - `cleanup_pose_frames(...)`

Diagnostics to compare:
- `diagnostics.source_stages.pose.comparisons.canonical_to_corrected_pre_cleanup`
- `diagnostics.source_stages.pose.comparisons.corrected_pre_cleanup_to_post_cleanup_pre_loop`
- `diagnostics.cleanup`
- `diagnostics.quality.retarget_risk`

### 3. Cached rig-profile priority can lock the viewer into stale mapping or preferred-mode choices

Most likely impact: medium-high.

Why:
- Viewer intentionally prefers validated and draft profiles before live model-analysis seeds.
- That is good for stable models, but harmful when a profile was saved against an earlier clip, a slightly different rig revision, or before recent retarget fixes.
- This can make mismatch appear model-specific even when the real problem is profile drift.

Primary files/functions:
- [`viewer/js/main.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/main.js)
  - `loadRigProfile(...)`
  - `autoSetupModel(...)`
  - `saveModelSetup(...)`
  - `validateCurrentRigProfile(...)`
- [`viewer/js/modules/viewer-model-analysis.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/viewer-model-analysis.js)
- [`viewer/RETARGET_NOTES.md`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/RETARGET_NOTES.md)

Diagnostics to compare:
- viewer `retarget-summary.rigProfile`
- viewer `selectionDebug`
- exported/imported rig-profile contents

### 4. Attempt selection and canonical mapping still favor structurally plausible but visually wrong retargets on atypical rigs

Most likely impact: medium.

Why:
- Recent fixes improved canonical ownership for helper/socket/end/control bones, but atypical rigs can still produce multiple plausible candidates.
- The current attempt scoring balances resolved tracks, motion probe, pose error, and preferred mode. That is strong, but it can still choose the wrong candidate when the metrics disagree.
- This is especially likely on rigs with duplicate chains, partial finger data, or unusual root hierarchies.

Primary files/functions:
- [`viewer/js/modules/viewer-retarget-attempts.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/viewer-retarget-attempts.js)
  - `collectRetargetAttempts(...)`
  - `selectRetargetAttempt(...)`
- [`viewer/js/modules/retarget-eval.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/retarget-eval.js)
  - `buildBindingsForAttempt(...)`
  - `probeMotionForBindings(...)`
  - `computePoseMatchError(...)`
- [`viewer/js/modules/viewer-topology-fallback.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/viewer-topology-fallback.js)
- [`viewer/js/modules/canonical-bone-map.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/canonical-bone-map.js)

Diagnostics to compare:
- viewer `attemptDebug`
- viewer `selectionDebug`
- viewer `retarget-map-details`
- viewer `retarget-limbs`

### 5. Target-side calibration is helpful but too conservative for some stylized or compact models

Most likely impact: medium.

Why:
- Body, finger, and arm calibration all clamp changes into safe windows.
- That reduces catastrophic distortions, but it also means compact, stylized, or disproportionate models can remain visibly off even when source motion is otherwise acceptable.
- In those cases the mismatch is not "bad retarget" so much as "insufficient allowed adaptation".

Primary files/functions:
- [`viewer/js/modules/retarget-calibration.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/retarget-calibration.js)
  - `buildBodyLengthCalibration(...)`
  - `buildFingerLengthCalibration(...)`
  - `buildArmRefinementCalibration(...)`
- [`viewer/js/modules/viewer-runtime-diagnostics.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/viewer-runtime-diagnostics.js)

Diagnostics to compare:
- viewer calibration reports in debug state
- viewer `retarget-summary`
- viewer `retarget-alignment`
- viewer `retarget-limbs`

## Recommended Next Implementation Sequence

### Coder

1. Add a cross-stage "ownership summary" diagnostic that records which side last changed yaw, hips alignment, and length calibration assumptions.
2. Add profile freshness guards so viewer can down-rank stale `draft` or `validated` profiles when model analysis or candidate metrics strongly disagree.
3. Extend attempt-selection diagnostics with a clearer "why winner beat runner-up" report.

Best first files:
- [`vid2model_lib/pipeline_motion_transforms.py`](/Users/fedor/projects/personal/videoToModel/vid2model/vid2model_lib/pipeline_motion_transforms.py)
- [`viewer/js/main.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/main.js)
- [`viewer/js/modules/viewer-retarget-attempts.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/viewer-retarget-attempts.js)
- [`viewer/js/modules/viewer-runtime-diagnostics.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/viewer-runtime-diagnostics.js)

### Tester

1. Expand the regression runner with scenario expectations that distinguish source-risk from viewer-risk, not just pass/fail execution.
2. Add at least one fixture that uses a compact or stylized rig plus an old cached profile to reproduce profile-priority drift.
3. Capture runner output snapshots for one "good generic rig" and one "atypical compact rig".

Best first files:
- [`tools/run_regression_checks.py`](/Users/fedor/projects/personal/videoToModel/vid2model/tools/run_regression_checks.py)
- [`tests/test_regression_runner.py`](/Users/fedor/projects/personal/videoToModel/vid2model/tests/test_regression_runner.py)
- [`tests/atypical-rig-mapping.test.mjs`](/Users/fedor/projects/personal/videoToModel/vid2model/tests/atypical-rig-mapping.test.mjs)
- [`tests/root-yaw-contract.test.mjs`](/Users/fedor/projects/personal/videoToModel/vid2model/tests/root-yaw-contract.test.mjs)

### Reviewer

1. Review whether new diagnostics really separate source ownership from viewer ownership instead of duplicating current logs.
2. Review whether any profile freshness rule can silently demote a user-approved validated profile in surprising ways.
3. Review whether new calibration freedom improves stylized rigs without regressing generic VRM/GLB cases.

## Practical Decision

If only one implementation slice is approved next, the highest-signal slice is:

1. profile freshness and candidate-selection transparency in the viewer,
2. then ownership diagnostics across Python root-yaw and viewer alignment,
3. then wider calibration experiments for stylized rigs.

This order should reduce ambiguity fastest, because it tells us whether remaining failures are primarily stale target assumptions or genuinely damaged source motion.
