# Project Guide for AI Agents (Claude/Cursor)

This document provides comprehensive context for AI agents working on `vid2model` project.

## Project Overview

**vid2model** converts human motion from video into skeletal animation and retargets it onto VRM models.

### What It Does
- Extracts pose sequences from video using MediaPipe or neural pose estimators (4D-Humans/HMR2.0)
- Cleans up motion (fixes gaps, side-swaps, noise, stabilizes foot contact and pelvis motion)
- Retargets motion to VRM/.glb models with intelligent bone mapping
- Validates retargeting quality with diagnostic metrics
- Generates rigging profiles for different character models

### Two Main Components

1. **Python Pipeline** (`vid2model_lib/`) - Converts video → BVH/motion data
2. **Browser Viewer** (`viewer/`) - Loads BVH, displays VRM models, performs retargeting in Three.js

---

## Architecture

### Python Pipeline (`vid2model_lib/`)

Entry point: `cli.py` and `pipeline.py`

**Pipeline stages** (in order):
1. **Extract** - Pose estimation (MediaPipe or SMPL/HMR2.0)
2. **Normalize** - Convert to consistent bone structure
3. **Channel processing** - Validate channels, fill gaps
4. **Mirror detection** - Detect left/right side swaps
5. **Cleanup** - Smooth motion, fix frame gaps, noise reduction
6. **Motion transforms** - Apply rest offsets, stability corrections
7. **Loop analysis** - Check if motion can loop
8. **Retarget** (optional) - Transform to target VRM skeleton
9. **Export** - BVH, JSON, CSV, NPZ, TRC formats

Key files:
- `pipeline.py` - Main pipeline orchestration
- `pipeline_cleanup.py` - Smoothing, gap filling, noise reduction
- `pipeline_retarget.py` - Motion retargeting logic
- `pipeline_loop.py` - Loop detection and extraction

### JavaScript Viewer (`viewer/`)

Entry point: `viewer/index.html` and `viewer/js/main.js`

**Key modules** (`viewer/js/modules/`):
- `viewer-model-loader.js` - Loads BVH, VRM, GLB models
- `retarget-live.js` - Real-time retargeting engine (the main logic)
- `retarget-plan-utils.js` - Builds retargeting plans
- `retarget-chain-utils.js` - Leg/arm chain analysis
- `canonical-motion-solver.js` - Quality validation
- `rig-profiles.js` - Manages bone mapping profiles

---

## Critical Knowledge: VRM Rotation Compensation

**IMPORTANT**: VRM models are loaded with `modelRoot.rotation.y = Math.PI` to face the viewer. This causes a 180° rotation that must be compensated in retargeting code.

### The Problem
- VRM's modelRoot has Y-axis rotation of π radians (180°)
- This rotates all child bones' world positions/orientations
- Source skeleton (MediaPipe/SMPL) has no such rotation
- When applying motion from source to target, the rotation frame mismatch causes bones to be oriented incorrectly

### The Solution (in `retarget-live.js`)

**For hips** (line ~982):
- Removed special yaw rotation compensation (was causing 180° flip)
- Hips now use same formula as other bones: `targetRestQ * deltaQ`
- Position still uses yaw compensation for facing direction

**For world-transfer bones** (feet, upper legs - line ~971):
- Extract baseYaw from modelRoot's `__baseQuaternion`
- When baseYaw > 0.1 radians (detects VRM):
  - Invert the delta quaternion: `deltaQ.invert()`
  - Apply correction: `deltaQ^-1 * Ry(-baseYaw) * targetRestWorldQ`
  - This removes the VRM rotation from target rest pose before applying source delta

Example fix (commit 7b81bf6 and follow-ups):
```javascript
// Pre-compute VRM base yaw
const baseQ = modelRoot?.userData?.__baseQuaternion;
let baseYaw = 0;
if (baseQ?.isQuaternion) {
  baseYaw = 2 * Math.atan2(baseQ.y, baseQ.w);
}

// For world-transfer bones (feet, etc)
if (Math.abs(baseYaw) > 0.1) {
  deltaQ.invert().normalize();
  parentWorldQ.setFromAxisAngle(axisY, -baseYaw);
  targetWorldQ.copy(deltaQ).multiply(parentWorldQ).multiply(pair.targetRestWorldQ).normalize();
}
```

**When modifying retargeting logic:**
- Always test with VRM models (they have baseYaw ≈ π)
- Test with non-VRM GLB models (they have baseYaw ≈ 0)
- Use `headless-retarget-validation.test.mjs` to validate
- Check both rotation AND position of all body parts

---

## Key Concepts

### Bone Mapping / Canonical Names
All bones are mapped to canonical names (HumanIK standard):
- Hip bones: `hips`, `leftUpperLeg`, `rightUpperLeg`, `leftLowerLeg`, `rightLowerLeg`, `leftFoot`, `rightFoot`
- Spine: `spine`, `chest`, `upperChest`, `neck`, `head`
- Arms: `leftShoulder`, `rightShoulder`, `leftUpperArm`, `rightUpperArm`, `leftLowerArm`, `rightLowerArm`, `leftHand`, `rightHand`
- Digits: `leftThumbProximal`, `rightIndex...` etc

See `viewer/js/modules/bone-utils.js` for mapping logic.

### Retargeting Modes

Different bone types use different retargeting strategies:

1. **Parent-relative rest delta** (spine, chest, upper body)
   - Used by: `PARENT_RELATIVE_REST_DELTA_CANONICAL` set
   - Formula: `targetRestQ * sourceCurrentDelta`
   - Works in parent-relative frame

2. **World rest transfer** (legs, feet)
   - Used by: `WORLD_REST_TRANSFER_CANONICAL` set
   - Formula: `deltaQ * targetRestWorldQ` (with VRM compensation)
   - Works in world frame, handles long chains

3. **Special modes** (feet direction/plane corrections)
   - After initial retarget, applies post-processing
   - Corrects foot orientation, plane alignment, mirror issues
   - Uses functions like `applyFootDirectionCorrection()`, `applyFootPlaneCorrection()`

### Rest Quaternions
- `targetRestQ` - Bone's local quaternion at rest (in rest pose)
- `targetRestWorldQ` - Bone's world quaternion at rest
- `sourceRestQ` - Source skeleton bone at rest
- These are computed in `initializeRetargetPairsRestState()` and cached in retarget plan

---

## Common Tasks & How to Handle Them

### Fix VRM Model Orientation Issues
1. Check if issue is VRM-specific (test with non-VRM models)
2. Check baseYaw extraction: `const baseYaw = 2 * Math.atan2(baseQ.y, baseQ.w)`
3. For rotation problems: verify compensation in `applyLiveRetargetPose()`
4. For position problems: check yawQ application in position delta code
5. Test with `node tests/headless-retarget-validation.test.mjs`

### Add New Bone Type Support
1. Define canonical name in `bone-utils.js`
2. Add to appropriate canonical set (`PARENT_RELATIVE_REST_DELTA_CANONICAL` etc)
3. Check if needs special correction in `buildProfiledChains()`
4. Update tests if introducing new behavior

### Improve Retargeting Quality
1. Check `viewer/js/modules/retarget-eval.js` for quality metrics
2. Modify correction logic in `retarget-plan-utils.js` (rest orientation correction)
3. Adjust chain corrections in `applyLiveRetargetPose()` (foot/arm corrections)
4. Run `canonical-motion-solver.js` for validation
5. Validate with test: check `lowerBodyRotError` and other metrics

### Profile Validation
- Profiles stored in `viewer/rig-profiles/` as JSON
- Format: `{ humanoidBoneMap: {...}, posScale: number, yawOffset: number, ... }`
- Validation in `headless-retarget-validation.test.mjs`
- Test threshold for `lowerBodyRotError` is around 135° (see test file)

---

## Testing

### Run All Tests
```bash
node tests/headless-retarget-validation.test.mjs
node tests/root-yaw-contract.test.mjs
node tests/sample-a-rig-profile.test.mjs
node tests/atypical-rig-mapping.test.mjs
```

### Test VRM Validation Specifically
```bash
node tests/headless-retarget-validation.test.mjs 2>&1 | grep -A5 "VRM humanoid"
```

Key test file: `tests/headless-retarget-validation.test.mjs`
- Tests MoonGirl VRM model
- Checks `lowerBodyRotError < 135` (canonical comparison)
- Validates lower-body doesn't mirror
- Ensures humanoid context preserved in output

---

## Build & Test Commands

```bash
# Run all validation tests
node tests/headless-retarget-validation.test.mjs

# Start local viewer server
python3 -m http.server 8080

# Convert video with MediaPipe (fast)
./convert.sh input.mp4 output/output.bvh

# Convert with SMPL neural model (accurate)
./convert_video_smpl.sh input.mp4 output/output.bvh

# Setup SMPL backend (one time)
./setup_smpl_backend.sh
```

---

## Code Conventions & Patterns

### Quaternion Math (Three.js)
- Always normalize after operations: `.normalize()`
- Multiplication order matters: `q1.multiply(q2)` = q1 * q2 in world space
- World → Local: `parent.invert().multiply(world)`
- Local → World: `parent.multiply(local)`
- Prefer reusing temp quaternions instead of creating new ones

### Temporary Variables in Loops
Don't create new objects in loops. Reuse temporaries:
```javascript
const deltaQ = liveQ || new THREE.Quaternion();  // Reuse
deltaQ.copy(sourceQ).multiply(targetQ);           // Compute in-place
```

### Bone Traversal
```javascript
function traverse(bone, callback) {
  callback(bone);
  for (const child of bone.children || []) {
    traverse(child, callback);
  }
}
```

### Canonical Names
```javascript
import { canonicalBoneKey } from "./bone-utils.js";

const key = canonicalBoneKey(bone.name);  // Returns canonical name or null
if (key === "hips") { /* special handling */ }
```

---

## Using Beads (bd) for Tasks

This project uses **beads** for issue tracking:

```bash
bd ready                    # Find available work
bd show <id>               # View issue details
bd create --title="..."    # Create new issue
bd update <id> --claim     # Claim work
bd close <id>              # Mark complete
bd remember "insight"      # Save persistent knowledge
```

**Important rules:**
- ALL task tracking goes through `bd`, NOT TodoWrite
- Session completion requires: `git pull --rebase && bd dolt push && git push`
- Use `bd prime` for full command reference

---

## Recent Fixes & Known Issues

### Fixed Issues (April 2026)

1. **VRM Hip Rotation (Commit 7b81bf6)**
   - Problem: Hips were rotating 180° in opposite direction
   - Fix: Removed yaw rotation compensation from hips formula
   - Code: Line ~982 in `retarget-live.js`

2. **VRM Feet/Legs Rotation (Commit ~April 5)**
   - Problem: Feet and legs were rotated 180° and rotating opposite direction
   - Fix: Added baseYaw compensation and delta inversion for world-transfer bones
   - Code: Line ~971 in `retarget-live.js`, checks `Math.abs(baseYaw) > 0.1`

3. **Spine Y must NOT be scaled by `upper_body_rotation_scale` (Commit 59fad8e)**
   - Problem: `apply_upper_body_rotation_scale` was scaling spine Y (≈±178°) along with Z/X. This dropped spine Y to ~41°, making `estimateFacingYawOffset` return ≈π → `strongFacingMismatch=true` → viewer forced live delta → model rotated 180°
   - Root cause: Spine local Y ≈±178° is a **system-coords artifact** from `normalize_motion_root_yaw` flipping hips by -180°. It is NOT real motion — it's the baseline rest orientation. Real spine motion lives in Z (forward lean) and X (side tilt) only.
   - Fix: Skip `base+2` for `joint_name == "spine"` in `apply_upper_body_rotation_scale()`
   - Code: `vid2model_lib/pipeline_motion_transforms.py`

### Known Patterns to Watch
- VRM models ALWAYS have `modelRoot.rotation.y = Math.PI`
- This baseYaw must be extracted and compensated in retargeting
- Non-VRM models have `baseYaw ≈ 0`, so compensation doesn't affect them
- Test both VRM and non-VRM models when modifying retargeting logic

---

## Resources

- **Beads Docs**: Run `bd prime` in terminal
- **Three.js Docs**: https://threejs.org/docs/
- **VRM Spec**: https://vrm.dev/
- **Canonical Bone Names**: `bone-utils.js` in viewer code

---

## Quick Reference: File Locations

| What | Where |
|------|-------|
| Python pipeline entry | `cli.py`, `pipeline.py` |
| Retargeting core logic | `viewer/js/modules/retarget-live.js` |
| Rest quaternion setup | `viewer/js/modules/retarget-plan-utils.js` |
| Bone chain analysis | `viewer/js/modules/retarget-chain-utils.js` |
| Quality validation | `viewer/js/modules/canonical-motion-solver.js` |
| Bone mapping | `viewer/js/modules/bone-utils.js` |
| Tests | `tests/` directory |
| VRM profiles | `viewer/rig-profiles/` |
| Cleanup algorithms | `vid2model_lib/pipeline_cleanup.py` |
| Motion retarget (Python) | `vid2model_lib/pipeline_retarget.py` |

