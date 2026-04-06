import test from "node:test";
import assert from "node:assert/strict";

import { getBuiltinRigProfile } from "../viewer/js/modules/rig-profiles.js";
import { createViewerRigProfileService } from "../viewer/js/modules/viewer-rig-profile-service.js";
import {
  resolveBodyMetricCanonicalFilter,
  resolveRetargetStageCanonicalFilter,
} from "../viewer/js/modules/retarget-stage-contract.js";

function createMemoryStorage(initialEntries = []) {
  const store = new Map();
  if (initialEntries.length) {
    store.set("vid2model.rigProfiles.test", JSON.stringify(initialEntries));
  }
  return {
    getItem(key) {
      return store.has(key) ? store.get(key) : null;
    },
    setItem(key, value) {
      store.set(key, String(value));
    },
  };
}

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

test("sample-a body metric filter keeps body-core checks but respects the simplified torso contract", () => {
  const profile = getBuiltinRigProfile({
    modelFingerprint: "rig:729cb56b",
    modelLabel: "6493143135142452442.glb",
    stage: "body",
  });

  const stageFilter = resolveRetargetStageCanonicalFilter("body", profile);
  const metricFilter = resolveBodyMetricCanonicalFilter("body", profile);

  assert.ok(stageFilter);
  assert.ok(metricFilter);
  assert.ok(stageFilter.has("leftHand"));
  assert.ok(stageFilter.has("rightHand"));
  assert.ok(!stageFilter.has("upperChest"));
  assert.ok(!stageFilter.has("neck"));
  assert.ok(!stageFilter.has("head"));

  assert.ok(metricFilter.has("hips"));
  assert.ok(metricFilter.has("spine"));
  assert.ok(metricFilter.has("chest"));
  assert.ok(metricFilter.has("leftUpperLeg"));
  assert.ok(metricFilter.has("rightFoot"));
  assert.ok(!metricFilter.has("upperChest"));
  assert.ok(!metricFilter.has("neck"));
  assert.ok(!metricFilter.has("head"));
  assert.ok(!metricFilter.has("leftHand"));
});

test("full-stage metric filter keeps the broader body contract", () => {
  const metricFilter = resolveBodyMetricCanonicalFilter("full");

  assert.ok(metricFilter.has("upperChest"));
  assert.ok(metricFilter.has("neck"));
  assert.ok(metricFilter.has("head"));
  assert.ok(metricFilter.has("leftHand"));
  assert.ok(metricFilter.has("rightHand"));
});

test("sample-a locked builtin profile wins over stored torso-heavy profiles", () => {
  const windowRef = {
    localStorage: createMemoryStorage([
      {
        id: "stale-sample-a",
        modelLabel: "6493143135142452442.glb",
        modelFingerprint: "rig:729cb56b",
        stage: "body",
        source: "localStorage",
        validationStatus: "validated",
        bodyCanonicalKeys: [
          "hips",
          "spine",
          "chest",
          "upperChest",
          "neck",
          "head",
          "leftUpperLeg",
          "rightUpperLeg",
        ],
        namesTargetToSource: {
          StaleUpperChest: "upperChest",
        },
      },
    ]),
  };
  const service = createViewerRigProfileService({
    windowRef,
    storageKey: "vid2model.rigProfiles.test",
    maxEntries: 12,
    statusValues: new Set(["draft", "validated"]),
    repoManifestUrl: "https://example.invalid/rig-profiles/index.json",
    getBuiltinRigProfile,
    getRetargetStage: () => "body",
    getCurrentModelRigFingerprint: () => "rig:729cb56b",
    buildRigProfileSeedForCurrentModel: () => null,
    buildSeedCorrectionSummary: () => [],
  });

  const profile = service.loadRigProfile("rig:729cb56b", "body", "6493143135142452442.glb");

  assert.ok(profile);
  assert.equal(profile.id, "sample-a-glb-v1");
  assert.equal(profile.lockBuiltin, true);
  assert.equal(profile.source, "repo");
  assert.equal(profile.validationStatus, "validated");
  assert.ok(!profile.bodyCanonicalKeys.includes("upperChest"));
  assert.ok(!profile.bodyCanonicalKeys.includes("neck"));
  assert.ok(!profile.bodyCanonicalKeys.includes("head"));
  assert.ok(!Object.hasOwn(profile.namesTargetToSource, "StaleUpperChest"));
});
