import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { runHeadlessRetargetValidation } from "../viewer/js/modules/headless-retarget-validation.js";

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const modelGlbPath = path.join(projectRoot, "viewer/models/low_poly_humanoid_robot.glb");
const modelVrmPath = path.join(projectRoot, "viewer/models/MoonGirl.vrm");
const bvhPath = path.join(projectRoot, "output/think.bvh");
const cliPath = path.join(projectRoot, "tools/headless_retarget_validation.mjs");

function getSingleEvent(result, eventName) {
  const payload = result?.diagnostics?.events?.[eventName];
  if (Array.isArray(payload)) return payload[payload.length - 1] || null;
  return payload || null;
}

function assertStableHeadlessShape(result, { stage, isVrm }) {
  assert.equal(result.format, "vid2model.headless-retarget.v1");
  assert.equal(result.input.stage, stage);
  assert.equal(result.model.isVrm, isVrm);
  assert.ok(result.model.rigFingerprint.startsWith("rig:"));
  assert.ok(result.model.skinnedMeshes >= 1);
  assert.ok(result.source.bones > 0);
  assert.ok(result.source.tracks > 0);
  assert.ok(result.mapping.mappedPairs > 0);
  assert.ok(result.mapping.matched > 0);
  assert.ok(result.selection.selectedAttempt);
  assert.ok(result.selection.selectedModeLabel);
  assert.equal(typeof result.selection.liveRetarget, "boolean");
  assert.ok(Array.isArray(result.diagnostics.records));
  assert.ok(result.diagnostics.records.length > 0);

  const summary = getSingleEvent(result, "retarget-summary");
  assert.ok(summary, "expected retarget-summary diagnostic");
  assert.equal(summary.stage, stage);
  assert.equal(summary.mode, result.selection.selectedModeLabel);
  assert.equal(summary.liveDelta, result.selection.liveRetarget);
  assert.ok(summary.mappedPairs > 0);

  assert.equal(result.canonicalComparison?.ran, true);
  assert.equal(result.canonicalComparison?.stage, stage);
  assert.equal(result.canonicalComparison?.exportFormat, "vid2model.canonical-motion.v1");
  assert.equal(result.canonicalComparison?.solverFormat, "vid2model.canonical-solve.v1");
  assert.ok((result.canonicalComparison?.sampleCount || 0) > 0);
  assert.equal(typeof result.canonicalComparison?.summary?.legsMirrored, "boolean");
  assert.ok(Number.isFinite(result.canonicalComparison?.summary?.poseError));
}

function assertNoCrossSideHandMapping(result) {
  const worstPosition = result?.diagnostics?.events?.["retarget-alignment"]?.worstPosition || [];
  const handAndFingerRows = worstPosition.filter((row) =>
    /Hand|Thumb|Index|Middle|Ring|Little/.test(String(row?.target || ""))
  );
  for (const row of handAndFingerRows) {
    const target = String(row?.target || "");
    const source = String(row?.source || "");
    if (target.startsWith("left")) {
      assert.ok(!source.startsWith("right"), `unexpected cross-side mapping: ${target} -> ${source}`);
    }
    if (target.startsWith("right")) {
      assert.ok(!source.startsWith("left"), `unexpected cross-side mapping: ${target} -> ${source}`);
    }
  }
}

test("headless module emits machine-readable diagnostics for GLB body validation", async () => {
  const result = await runHeadlessRetargetValidation({
    modelPath: modelGlbPath,
    bvhPath,
    stage: "body",
  });

  assertStableHeadlessShape(result, {
    stage: "body",
    isVrm: false,
  });
  assert.match(result.rigProfile?.source || "", /^model-analysis/);
  assert.equal(result.diagnostics.calibrations.body?.bones > 0, true);
});

test("headless module keeps VRM humanoid context in machine-readable output", async () => {
  const result = await runHeadlessRetargetValidation({
    modelPath: modelVrmPath,
    bvhPath,
    stage: "full",
  });

  assertStableHeadlessShape(result, {
    stage: "full",
    isVrm: true,
  });
  assert.ok(result.model.vrmHumanoidApplied > 0);
  assert.equal(result.rigProfile?.id, "moon-girl-v1");
  assert.equal(result.rigProfile?.source, "repo");
  assert.equal(result.rigProfile?.lockBuiltin, true);
  assert.equal(result.selection.liveRetarget, true);
  assert.match(result.selection.selectedModeLabel || "", /\+live-delta$/);
  assert.equal(result.mapping.mirroredArmSidesApplied, false);
  assert.equal(result.mapping.armSideSwapScore, null);
  assertNoCrossSideHandMapping(result);
  assert.ok(
    (result.diagnostics.chainDiagnostics.leg || []).every((row) => row?.bendMirrored === false),
    "expected MoonGirl lower-body bend diagnostics to stay non-mirrored"
  );
  assert.ok(
    (result.canonicalComparison?.summary?.lowerBodyRotError || Number.POSITIVE_INFINITY) < 125,
    "expected lower-body rotation mismatch to improve materially (canonical comparison)"
  );
});

test("headless CLI writes the same JSON contract for full-stage validation", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "vid2model-headless-"));
  const outPath = path.join(tempDir, "result.json");

  try {
    const { stdout } = await execFileAsync(process.execPath, [
      cliPath,
      "--model",
      modelGlbPath,
      "--bvh",
      bvhPath,
      "--stage",
      "full",
      "--out",
      outPath,
    ], {
      cwd: projectRoot,
      maxBuffer: 20 * 1024 * 1024,
    });

    const stdoutJson = JSON.parse(stdout);
    const fileJson = JSON.parse(await fs.readFile(outPath, "utf8"));

    assert.deepEqual(fileJson, stdoutJson);
    assertStableHeadlessShape(stdoutJson, {
      stage: "full",
      isVrm: false,
    });
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
});
