// ============================================================================
// Image provider tests — OpenRouter only (one gateway, no per-provider keys)
// ----------------------------------------------------------------------------
// The assertions are about the REQUEST WE BUILD: each model family accepts
// different parameters, and the generic body 400s on Recraft.
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";

import {
  isTypeSpecialist, snapAspect, imageRequestBody, classConsistentChain,
} from "../src/creative/imageProviders.js";

test("isTypeSpecialist: only Recraft sets the type itself (real fonts)", () => {
  assert.equal(isTypeSpecialist("recraft/recraft-v4"), true);
  assert.equal(isTypeSpecialist("recraft/recraft-v3"), true);
  assert.equal(isTypeSpecialist("recraft/recraft-v4-vector"), true);
  assert.equal(isTypeSpecialist("bytedance-seed/seedream-5-0-flash"), false);
  assert.equal(isTypeSpecialist("openai/gpt-image-2"), false);
  assert.equal(isTypeSpecialist(""), false);
});

test("isTypeSpecialist: IDEOGRAM is not on OpenRouter, so it is never a specialist", () => {
  // A configured `ideogram/...` slug would 404 on /images. It must not be
  // treated as a type specialist (that would skip our type layer for a model
  // that never answers).
  assert.equal(isTypeSpecialist("ideogram/ideogram-v3"), false);
});

test("snapAspect: Recraft's supported ratios only, snapped by shape", () => {
  assert.equal(snapAspect("recraft/recraft-v4", "1:1"), "1:1");
  assert.equal(snapAspect("recraft/recraft-v4", "16:9"), "16:9");
  assert.equal(snapAspect("recraft/recraft-v4", "9:16"), "9:16");
  // 4:5 is not supported -> the closest supported portrait ratio
  assert.equal(snapAspect("recraft/recraft-v4", "4:5"), "3:4");
  assert.equal(snapAspect("recraft/recraft-v4", "2:3"), "3:4");
  assert.equal(snapAspect("recraft/recraft-v4", "3:2"), "4:3");
  // A model with no restrictions keeps the ratio it was given
  assert.equal(snapAspect("bytedance-seed/seedream-5-0-flash", "4:5"), "4:5");
});

test("imageRequestBody: the generic body for a normal model", () => {
  const body = imageRequestBody("bytedance-seed/seedream-5-0-flash", "a poster", {
    aspect_ratio: "4:5", output_format: "png", n: 1,
  });
  assert.equal(body.model, "bytedance-seed/seedream-5-0-flash");
  assert.equal(body.prompt, "a poster");
  assert.equal(body.aspect_ratio, "4:5");
  assert.equal(body.n, 1);
  assert.equal(body.output_format, "png");
});

test("imageRequestBody: Recraft rejects n and output_format — they must not be sent", () => {
  const body = imageRequestBody("recraft/recraft-v4", "a poster", {
    aspect_ratio: "4:5", output_format: "png", n: 1,
  });
  assert.equal(body.n, undefined, "Recraft 400s on n");
  assert.equal(body.output_format, undefined, "Recraft 400s on output_format");
  assert.equal(body.aspect_ratio, "3:4", "the ratio is snapped to one Recraft accepts");
});

test("imageRequestBody: an explicit drop wins over the model defaults", () => {
  const body = imageRequestBody("bytedance-seed/seedream-5-0-flash", "x", {
    aspect_ratio: "4:5", n: 1, drop: ["aspect_ratio", "n"],
  });
  assert.equal(body.aspect_ratio, undefined);
  assert.equal(body.n, undefined);
});

test("imageRequestBody: input references are passed through", () => {
  const refs = [{ type: "image_url", image_url: { url: "https://x/y.png" } }];
  const body = imageRequestBody("recraft/recraft-v4", "x", { input_references: refs });
  assert.deepEqual(body.input_references, refs);
});

test("classConsistentChain: the chain never mixes a type specialist with a general model", () => {
  const chain = classConsistentChain(
    ["bytedance-seed/seedream-5-0-flash", "recraft/recraft-v4", "openai/gpt-image-2"],
    { specialist: false },
  );
  assert.deepEqual(chain, ["bytedance-seed/seedream-5-0-flash", "openai/gpt-image-2"]);
  assert.ok(!chain.includes("recraft/recraft-v4"));

  const specialistChain = classConsistentChain(
    ["recraft/recraft-v4", "bytedance-seed/seedream-5-0-flash"],
    { specialist: true },
  );
  assert.deepEqual(specialistChain, ["recraft/recraft-v4"]);
});

test("classConsistentChain: an empty chain stays empty", () => {
  assert.deepEqual(classConsistentChain([], { specialist: false }), []);
  assert.deepEqual(classConsistentChain([null, undefined], { specialist: false }), []);
});
