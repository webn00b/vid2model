#!/usr/bin/env node

import fs from "node:fs/promises";
import process from "node:process";

import { runHeadlessRetargetValidation } from "../viewer/js/modules/headless-retarget-validation.js";

function parseArgs(argv) {
  const args = {
    stage: "body",
    pretty: false,
    out: "",
    model: "",
    bvh: "",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const value = argv[i];
    if (value === "--model") {
      args.model = argv[++i] || "";
      continue;
    }
    if (value === "--bvh") {
      args.bvh = argv[++i] || "";
      continue;
    }
    if (value === "--stage") {
      args.stage = argv[++i] || "";
      continue;
    }
    if (value === "--out") {
      args.out = argv[++i] || "";
      continue;
    }
    if (value === "--pretty") {
      args.pretty = true;
      continue;
    }
    if (value === "--help" || value === "-h") {
      return { ...args, help: true };
    }
    throw new Error(`Unknown argument: ${value}`);
  }

  return args;
}

function usage() {
  return [
    "Headless retarget validation for BVH + GLB/VRM.",
    "",
    "Usage:",
    "  node tools/headless_retarget_validation.mjs --model <path> --bvh <path> [--stage body|full] [--out <json>] [--pretty]",
  ].join("\n");
}

async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.help) {
    console.log(usage());
    return 0;
  }
  if (!args.model || !args.bvh) {
    throw new Error("Both --model and --bvh are required.");
  }

  const result = await runHeadlessRetargetValidation({
    modelPath: args.model,
    bvhPath: args.bvh,
    stage: args.stage || "body",
  });
  const text = JSON.stringify(result, null, args.pretty ? 2 : 0);
  if (args.out) {
    await fs.writeFile(args.out, `${text}\n`, "utf8");
  }
  process.stdout.write(`${text}\n`);
  return 0;
}

main().catch((err) => {
  const message = String(err?.message || err || "Unknown error");
  console.error(message);
  process.exitCode = 1;
});
