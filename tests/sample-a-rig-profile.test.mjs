import test from "node:test";
import assert from "node:assert/strict";

import { getBuiltinRigProfile } from "../viewer/js/modules/rig-profiles.js";

test("sample-a body builtin profile simplifies upper torso anchors and keeps softened upper-body scaling", () => {
  const profile = getBuiltinRigProfile({
    modelFingerprint: "rig:729cb56b",
    modelLabel: "6493143135142452442.glb",
    stage: "body",
  });

  assert.ok(profile);
  assert.equal(profile.id, "sample-a-glb-v1");
  assert.equal(profile.lockBuiltin, true);

  assert.ok(!profile.bodyCanonicalKeys.includes("upperChest"));
  assert.ok(!profile.bodyCanonicalKeys.includes("neck"));
  assert.ok(!profile.bodyCanonicalKeys.includes("head"));
  assert.ok(profile.bodyCanonicalKeys.includes("leftShoulder"));
  assert.ok(profile.bodyCanonicalKeys.includes("rightShoulder"));
  assert.ok(profile.bodyCanonicalKeys.includes("leftHand"));
  assert.ok(profile.bodyCanonicalKeys.includes("rightHand"));

  assert.equal(profile.rotationScaleByCanonical.spine, 0.5);
  assert.equal(profile.rotationScaleByCanonical.upperChest, 0.24);
  assert.equal(profile.rotationScaleByCanonical.leftShoulder, 0.18);
  assert.equal(profile.rotationScaleByCanonical.leftUpperArm, 0.55);
  assert.equal(profile.rotationScaleByCanonical.leftLowerArm, 0.78);
  assert.equal(profile.rotationScaleByCanonical.leftHand, 0.1);
});
