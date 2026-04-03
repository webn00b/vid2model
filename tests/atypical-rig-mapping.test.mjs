import test from "node:test";
import assert from "node:assert/strict";

import { buildCanonicalBoneMap } from "../viewer/js/modules/canonical-bone-map.js";
import { canonicalBoneKey } from "../viewer/js/modules/bone-utils.js";

function bone(name, children = []) {
  return { name, children };
}

test("buildCanonicalBoneMap prefers primary bones over helper-like duplicates", () => {
  const leftFootHelper = bone("LeftFootHelper");
  const leftFoot = bone("LeftFoot", [bone("LeftToeBase")]);
  const map = buildCanonicalBoneMap([leftFootHelper, leftFoot]);

  assert.equal(map.get("leftFoot")?.name, "LeftFoot");
});

test("buildCanonicalBoneMap prefers primary source bones over helper-like canonical duplicates on both sides", () => {
  const map = buildCanonicalBoneMap([
    bone("LeftFootHelper"),
    bone("LeftFoot"),
    bone("RightFootSocket"),
    bone("RightFoot", [bone("RightToeBase")]),
  ]);

  assert.equal(canonicalBoneKey("LeftFootHelper"), "leftFoot");
  assert.equal(canonicalBoneKey("RightFootSocket"), "rightFoot");
  assert.equal(map.get("leftFoot")?.name, "LeftFoot");
  assert.equal(map.get("rightFoot")?.name, "RightFoot");
});
