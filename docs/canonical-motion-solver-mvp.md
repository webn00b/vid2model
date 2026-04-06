# Canonical Motion And Target Solver MVP

This note defines the first `canonical-motion` contract for `vid2model-s0b`.

The goal is to stop treating exported `BVH` rotation channels as the final motion representation for every target rig. Instead, the viewer/headless path should be able to:

1. read source motion,
2. normalize it into a target-agnostic body representation,
3. solve that representation onto a concrete model skeleton,
4. compare the solved result against the current retarget path before any wider rollout.

## Why This Exists

The current viewer path already does a lot of useful work:

- canonical name mapping,
- mode selection,
- root-yaw ownership,
- body and finger calibration,
- live-delta repair for hard rigs.

That is good enough for many rigs, but it still starts from a foreign skeleton clip. When a source `BVH` and a target model disagree on local axes, bend planes, proportions, or rest-pose semantics, the current path has to recover using heuristics after the fact.

The canonical-motion path moves the abstraction up one level:

- source motion becomes "human body intent",
- target solve becomes a separate concern,
- model-specific fixes become solver or profile policy instead of clip surgery.

## MVP Scope

The MVP is intentionally body-only.

Included:

- root / hips trajectory
- spine through head
- shoulders, arms, and hands
- upper legs, lower legs, feet, and toes
- stage-aware `body` vs `full` canonical filtering
- headless-only comparison path first

Explicitly out of scope for MVP:

- finger articulation solving
- physics, contacts, or runtime IK beyond simple body chain hints
- viewer UI switching as the default path
- replacing the existing `SkeletonUtils` path immediately

## Canonical Motion Contract

The first contract should be stored as plain JSON-serializable data and avoid direct `THREE` instances.

Top-level shape:

```json
{
  "format": "vid2model.canonical-motion.v1",
  "generatedAt": "2026-04-04T00:00:00.000Z",
  "source": {
    "clipName": "think.bvh",
    "frameCount": 120,
    "fps": 30,
    "duration": 4
  },
  "stage": "full",
  "restPose": {},
  "frames": []
}
```

### `restPose`

`restPose` captures the source-side canonical rig in a solver-friendly way. It is not the target model analysis and not the raw `BVH` hierarchy dump.

Required fields:

- `rootToHips`: local root-to-hips offset in canonical space
- `segmentLengths`: canonical segment lengths by semantic key
- `chains`: rest-chain summaries for torso, left/right arm, left/right leg
- `basis`: canonical frame basis used during export

Example:

```json
{
  "rootToHips": [0, 0.92, 0],
  "segmentLengths": {
    "spine": 0.12,
    "chest": 0.1,
    "upperChest": 0.08,
    "neck": 0.05,
    "head": 0.09,
    "leftUpperArm": 0.14,
    "leftLowerArm": 0.13,
    "leftHand": 0.05,
    "leftUpperLeg": 0.22,
    "leftLowerLeg": 0.23,
    "leftFoot": 0.11
  },
  "chains": {
    "torso": ["hips", "spine", "chest", "upperChest", "neck", "head"],
    "leftArm": ["leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand"],
    "rightArm": ["rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand"],
    "leftLeg": ["hips", "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes"],
    "rightLeg": ["hips", "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes"]
  },
  "basis": {
    "up": [0, 1, 0],
    "forward": [0, 0, 1],
    "right": [1, 0, 0]
  }
}
```

### `frames[]`

Each frame should describe body intent in canonical space rather than raw foreign-bone local rotations.

Required fields per frame:

- `time`
- `root.position`
- `root.facingQuat`
- `segments`
- `contacts`

`segments` should use semantic keys and expose world- or root-relative direction targets plus bend hints:

```json
{
  "time": 0.033333,
  "root": {
    "position": [0.01, 0.93, 0.02],
    "facingQuat": [0, 0.12, 0, 0.99]
  },
  "segments": {
    "spine": { "dir": [0.02, 0.99, 0.01] },
    "chest": { "dir": [0.08, 0.97, 0.2] },
    "neck": { "dir": [0.01, 0.99, 0.04] },
    "head": { "dir": [0.03, 0.98, 0.15] },
    "leftUpperArm": {
      "dir": [-0.82, 0.28, 0.49],
      "bendHint": [-0.06, 0.73, 0.67]
    },
    "leftLowerArm": {
      "dir": [-0.69, -0.14, 0.71],
      "bendHint": [-0.04, 0.72, 0.69]
    },
    "leftHand": {
      "dir": [-0.51, -0.2, 0.84],
      "normal": [-0.04, 0.97, 0.22]
    },
    "leftUpperLeg": {
      "dir": [-0.06, -0.98, 0.16],
      "bendHint": [-0.92, 0, -0.39]
    },
    "leftLowerLeg": {
      "dir": [-0.02, -0.97, 0.24],
      "bendHint": [-0.9, 0.02, -0.44]
    },
    "leftFoot": {
      "dir": [0.14, -0.08, 0.98],
      "normal": [0, 0.99, 0.02]
    }
  },
  "contacts": {
    "leftFootPlant": true,
    "rightFootPlant": false
  }
}
```

## Canonical Semantic Keys

The MVP should reuse the viewer canonical naming that already exists in:

- [`viewer/js/modules/bone-utils.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/bone-utils.js)
- [`viewer/js/modules/retarget-stage-contract.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/retarget-stage-contract.js)

Body MVP semantic keys:

- `hips`
- `spine`
- `chest`
- `upperChest`
- `neck`
- `head`
- `leftShoulder`
- `rightShoulder`
- `leftUpperArm`
- `rightUpperArm`
- `leftLowerArm`
- `rightLowerArm`
- `leftHand`
- `rightHand`
- `leftUpperLeg`
- `rightUpperLeg`
- `leftLowerLeg`
- `rightLowerLeg`
- `leftFoot`
- `rightFoot`
- `leftToes`
- `rightToes`

`full` stage keeps the same body semantic set in MVP. Finger semantics can be added later in `v2` rather than inflating the first solver.

## Target Model Contract

The target solver should not inspect arbitrary viewer state. It should consume a stable, model-analysis-oriented payload derived from:

- [`viewer/js/modules/viewer-model-analysis.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/viewer-model-analysis.js)

Required target inputs:

- `modelFingerprint`
- `humanoid.canonicalToTarget`
- `segmentLengths`
- `hips.bindPositionLocal`
- `footHints.left/right`
- `bones[]` rows with `bindQuaternionLocal` and `primaryChildDirLocal`

The solver can start with the existing `vid2model.model-analysis.v1` payload and add a thin adapter rather than inventing a second model-analysis format immediately.

## Solver MVP Responsibilities

The first solver should be deterministic and layered.

### 1. Root solve

- place root / hips from `frame.root.position`
- apply source-facing yaw from `frame.root.facingQuat`
- preserve target-model root ownership rules already enforced by viewer root alignment

### 2. Torso solve

- solve `hips -> spine -> chest -> upperChest -> neck -> head`
- match segment directions while preserving target segment lengths
- avoid per-bone twist optimization in MVP; only direction matching is required

### 3. Arm solve

- solve shoulder, upper arm, lower arm, hand directions
- use `bendHint` as elbow plane preference
- preserve side ownership and avoid mirrored swap heuristics inside the solver

### 4. Leg solve

- solve upper leg, lower leg, foot, and toes
- use `bendHint` as knee plane preference
- use foot `dir` and `normal` plus target `footHints`
- allow simple foot-plant stabilization when `contacts.*FootPlant` is true

### 5. Export solved pose

- return a frame-wise target pose summary and, for headless MVP, a `THREE.AnimationClip` or equivalent pose application result that existing diagnostics can already inspect

## Module Boundaries

New modules should live under `viewer/js/modules` first so the MVP can reuse existing math and diagnostics.

Suggested MVP files:

- `canonical-motion-export.js`
  - build `vid2model.canonical-motion.v1` from source result / stage clip
- `canonical-motion-utils.js`
  - shared semantic helpers, vector serialization, chain iteration
- `canonical-motion-solver.js`
  - solve canonical frames onto target model analysis / skeleton
- `canonical-motion-headless.js`
  - bridge canonical exporter + solver into the existing headless runner

Existing modules to reuse rather than duplicate:

- [`viewer/js/modules/retarget-chain-utils.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/retarget-chain-utils.js)
- [`viewer/js/modules/retarget-plan-utils.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/retarget-plan-utils.js)
- [`viewer/js/modules/retarget-analysis.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/retarget-analysis.js)
- [`viewer/js/modules/viewer-chain-diagnostics.js`](/Users/fedor/projects/personal/videoToModel/vid2model/viewer/js/modules/viewer-chain-diagnostics.js)

## Headless-First Rollout

The MVP should not replace the browser path immediately.

Rollout order:

1. Add exporter that builds `canonical-motion.v1` from the already loaded source result.
2. Add a body-only solver that can pose a target skeleton in memory.
3. Extend headless validation to run both:
   - existing retarget path
   - canonical solver path
4. Emit comparison diagnostics:
   - selected canonical solver mode
   - canonical post-solve pose error
   - canonical lower-body rotation error
   - torso / arm / leg chain diagnostics
5. Keep existing retarget as the default winner until canonical solver proves better on regression fixtures.

This keeps the first release measurable and reversible.

## Comparison Contract For Headless

Headless output should grow by composition, not by replacing the existing format.

Add a `canonicalComparison` block:

```json
{
  "canonicalComparison": {
    "ran": true,
    "stage": "full",
    "exportFormat": "vid2model.canonical-motion.v1",
    "solverFormat": "vid2model.canonical-solve.v1",
    "summary": {
      "poseError": 0.19,
      "lowerBodyRotError": 88.4,
      "legsMirrored": false
    }
  }
}
```

This lets regression checks compare old and new paths side by side without changing current consumers.

## Acceptance Criteria For MVP

`vid2model-s0b` should be considered successful when all of the following are true:

1. A source clip can be exported into `vid2model.canonical-motion.v1` for `body` and `full`.
2. A target model with `model-analysis.v1` can be posed by the canonical solver without browser-only dependencies.
3. Headless validation can emit a canonical comparison summary next to the existing retarget summary.
4. At least one regression fixture demonstrates that canonical solver output is stable and machine-readable.
5. Existing `SkeletonUtils` retarget remains available as fallback during the rollout.

## Non-Goals

The MVP is not trying to solve every remaining mismatch immediately.

Non-goals:

- perfect fingers
- final production-quality IK
- automatic replacement of repo rig profiles
- deleting current retarget calibration code
- changing the Python exporter contract before the headless comparison proves value

## Follow-Up After MVP

If the MVP is promising, the next slices should be:

1. add finger semantics as `canonical-motion.v2`,
2. add solver scoring into retarget-attempt selection,
3. expose canonical solver as an opt-in browser mode,
4. gradually promote model-specific solver policy into validated rig profiles.
