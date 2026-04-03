import test from "node:test";
import assert from "node:assert/strict";

import {
  buildViewerRootYawCandidates,
  shouldAllowSourceRootYawFlipCandidates,
} from "../viewer/js/modules/root-yaw-contract.js";

function quantizeFacingYaw(rad) {
  const quarterTurn = Math.PI / 2;
  return Math.round(rad / quarterTurn) * quarterTurn;
}

function assertAnglesAlmostEqual(actual, expected) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i += 1) {
    assert.ok(Math.abs(actual[i] - expected[i]) < 1e-9, `angle[${i}] mismatch: ${actual[i]} vs ${expected[i]}`);
  }
}

test("centered source clips suppress unconditional 180-degree flip candidates", () => {
  const centeredSummary = {
    looksCentered: true,
    nearZeroRatio: 0.9,
    nearPiRatio: 0.0,
    absMedianYawDeg: 8,
  };

  const candidates = buildViewerRootYawCandidates(Math.PI * 0.6, quantizeFacingYaw, {
    sourceClipYawSummary: centeredSummary,
  });

  assert.equal(shouldAllowSourceRootYawFlipCandidates(centeredSummary), false);
  assertAnglesAlmostEqual(candidates, [0, -Math.PI / 2, -Math.PI * 0.6]);
});

test("uncentered source clips still keep legacy flip candidates", () => {
  const uncenteredSummary = {
    looksCentered: false,
    nearZeroRatio: 0.1,
    nearPiRatio: 0.7,
    absMedianYawDeg: 145,
  };

  const candidates = buildViewerRootYawCandidates(Math.PI * 0.6, quantizeFacingYaw, {
    sourceClipYawSummary: uncenteredSummary,
  });

  assert.equal(shouldAllowSourceRootYawFlipCandidates(uncenteredSummary), true);
  assertAnglesAlmostEqual(candidates, [0, Math.PI, -Math.PI, -Math.PI / 2, Math.PI / 2, -Math.PI * 0.6]);
});
