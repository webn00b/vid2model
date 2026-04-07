/**
 * Regression tests for SLOVO sign-language BVH files retargeted onto MoonGirl VRM.
 *
 * These files have a distinctive pipeline artifact: normalize_motion_root_yaw flips hips
 * by −180°, so spine local Y ends up ≈ ±178°. This swaps left/right shoulder world
 * positions and can confuse estimateFacingVector. The tests lock down the correct
 * behaviour so any future change to estimateFacingVector or initializeRetargetPlan
 * immediately fails if it introduces a regression.
 *
 * Run:
 *   node tests/slovo-retarget.test.mjs
 */

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { runHeadlessRetargetValidation } from "../viewer/js/modules/headless-retarget-validation.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const modelVrmPath = path.join(projectRoot, "viewer/models/MoonGirl.vrm");
const slovoDir = path.join(projectRoot, "output/slovo_привет/привет");

// ─── helpers ────────────────────────────────────────────────────────────────

function getLiveDeltaEvent(result) {
  const payload = result?.diagnostics?.events?.["retarget-live-delta"];
  return Array.isArray(payload) ? payload[payload.length - 1] : payload ?? null;
}

function getRetargetSummary(result) {
  const payload = result?.diagnostics?.events?.["retarget-summary"];
  return Array.isArray(payload) ? payload[payload.length - 1] : payload ?? null;
}

async function runSlovo(bvhFile) {
  const bvhPath = path.join(slovoDir, bvhFile);
  return runHeadlessRetargetValidation({ modelPath: modelVrmPath, bvhPath, stage: "full" });
}

async function bvhFilesExist() {
  try {
    const entries = await fs.readdir(slovoDir);
    return entries.filter((f) => f.endsWith(".bvh"));
  } catch {
    return [];
  }
}

// ─── tests ──────────────────────────────────────────────────────────────────

const BVH_FILES = [
  "02105473.bvh",
  "055a4aa1.bvh",
  "8b68e645.bvh",
  "a69e018c.bvh",
  "f17a6060.bvh",
];

// Verify test data is present before running anything.
const presentFiles = await bvhFilesExist();
const missingFiles = BVH_FILES.filter((f) => !presentFiles.includes(f));

if (missingFiles.length > 0) {
  console.warn(
    `[slovo-retarget] Skipping: missing BVH files in ${slovoDir}:\n  ${missingFiles.join("\n  ")}\n` +
    `  Run: tools/slovo_ingest.py --sign "Привет!" --split all --count 0 --output-dir output/slovo_привет`
  );
  process.exit(0);
}

// ── per-file contracts ───────────────────────────────────────────────────────

for (const bvhFile of BVH_FILES) {
  test(`${bvhFile}: no strongFacingMismatch (spine Y artifact handled)`, async () => {
    const result = await runSlovo(bvhFile);
    const lde = getLiveDeltaEvent(result);
    assert.ok(lde, "retarget-live-delta event must be present");
    assert.equal(
      lde.reasons?.strongFacingMismatch,
      false,
      `strongFacingMismatch should be false — spine Y ≈ ±180° artifact must not invert estimateFacingVector`
    );
  });

  test(`${bvhFile}: yawOffsetDeg in expected range for root_yaw_offset=90 + VRM baseYaw`, async () => {
    const result = await runSlovo(bvhFile);
    const summary = getRetargetSummary(result);
    assert.ok(summary, "retarget-summary event must be present");
    const yaw = summary.yawOffsetDeg;
    assert.ok(
      Number.isFinite(yaw) && yaw >= 45 && yaw <= 135,
      `yawOffsetDeg=${yaw} should be 45–135° for SLOVO (hips Y≈90°) retargeted onto VRM (baseYaw=π)`
    );
  });

  test(`${bvhFile}: lowerBodyRotError < 135° (lower body not mirrored)`, async () => {
    const result = await runSlovo(bvhFile);
    const summary = getRetargetSummary(result);
    assert.ok(summary, "retarget-summary event must be present");
    const err = summary.lowerBodyRotError;
    assert.ok(
      Number.isFinite(err) && err < 135,
      `lowerBodyRotError=${err?.toFixed(1)}° should be < 135° (lower-body must not be mirrored)`
    );
  });
}

// ── cross-file contract ──────────────────────────────────────────────────────

test("all SLOVO greeting files: live-delta forced by MoonGirl profile", async () => {
  const results = await Promise.all(BVH_FILES.map(runSlovo));
  for (let i = 0; i < BVH_FILES.length; i++) {
    assert.equal(
      results[i].selection.liveRetarget,
      true,
      `${BVH_FILES[i]}: liveRetarget should be true (forced by MoonGirl forceLiveDelta profile)`
    );
  }
});

test("all SLOVO greeting files: no cross-side hand mapping", async () => {
  const results = await Promise.all(BVH_FILES.map(runSlovo));
  for (let i = 0; i < BVH_FILES.length; i++) {
    const worstPos = results[i].diagnostics?.events?.["retarget-alignment"]?.worstPosition ?? [];
    for (const row of worstPos) {
      const target = String(row?.target ?? "");
      const source = String(row?.source ?? "");
      if (/Hand|Thumb|Index|Middle|Ring|Little/.test(target)) {
        if (target.startsWith("left"))
          assert.ok(!source.startsWith("right"), `${BVH_FILES[i]}: cross-side hand mapping: ${target} ← ${source}`);
        if (target.startsWith("right"))
          assert.ok(!source.startsWith("left"), `${BVH_FILES[i]}: cross-side hand mapping: ${target} ← ${source}`);
      }
    }
  }
});
