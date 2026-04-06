import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import * as THREE from "../viewer/vendor/three/three.module.js";
import { BVHLoader } from "../viewer/vendor/three/addons/loaders/BVHLoader.js";
import { canonicalBoneKey } from "../viewer/js/modules/bone-utils.js";
import { computeBvhGroundY } from "../viewer/js/modules/viewer-alignment.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const bvhPath = path.join(projectRoot, "output/blackman.bvh");

test("BVH ground snap samples the clip instead of trusting the first frame", async () => {
  const text = await fs.readFile(bvhPath, "utf8");
  const loader = new BVHLoader();
  const result = loader.parse(text);
  const bones = result?.skeleton?.bones || [];
  assert.ok(bones.length > 0, "expected BVH bones");

  const mixer = new THREE.AnimationMixer(bones[0]);
  mixer.clipAction(result.clip).play();
  mixer.setTime(0);
  bones[0].updateMatrixWorld(true);

  const firstFrameGroundY = computeBvhGroundY({
    bones,
    canonicalBoneKey,
  });
  const sampledGroundY = computeBvhGroundY({
    bones,
    mixer,
    clip: result.clip,
    canonicalBoneKey,
  });

  assert.equal(Number.isFinite(firstFrameGroundY), true);
  assert.equal(Number.isFinite(sampledGroundY), true);
  assert.ok(
    firstFrameGroundY - sampledGroundY > 5,
    `expected sampled ground to be lower than first frame, got first=${firstFrameGroundY}, sampled=${sampledGroundY}`
  );
});
