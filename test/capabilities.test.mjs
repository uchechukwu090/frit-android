// ============================================================================
// FRIT CREATIVE CAPABILITIES — the UI's option chips come from here
// ----------------------------------------------------------------------------
// The regression this protects: the app used to hardcode 4:5 for every image
// model (Recraft rejects it) and 4s/720p for every video (a model that only
// does 6-8s refused it). The capability object is now the single source the
// studios render from, so these tests pin what it reports for live provider
// metadata, for a known family, and for a completely unknown model.
// ============================================================================

import test from "node:test";
import assert from "node:assert";
import {
  capValues, capMax, findModelRecord, imageCapabilities, videoCapabilities,
  creativeCapabilities, IMAGE_CAP_FALLBACK, VIDEO_CAP_FALLBACK,
} from "../src/creative/capabilities.js";

test("capValues reads arrays and enum-style parameter descriptors", () => {
  assert.deepEqual(capValues(["1:1", "16:9"]), ["1:1", "16:9"]);
  assert.deepEqual(capValues({ type: "enum", enum: ["720p", "1080p"] }), ["720p", "1080p"]);
  assert.deepEqual(capValues({ values: [1, 2, 3] }), ["1", "2", "3"]);
  assert.equal(capValues(undefined), null);
  assert.equal(capValues([]), null);
  assert.equal(capValues({}), null);
});

test("capMax reads range descriptors, plain numbers, and rejects junk", () => {
  assert.equal(capMax({ type: "range", min: 1, max: 4 }), 4);
  assert.equal(capMax(3), 3);
  assert.equal(capMax(0), null);
  assert.equal(capMax("nope"), null);
  assert.equal(capMax(null), null);
});

test("findModelRecord prefers exact, then canonical, then family", () => {
  const list = [
    { id: "recraft/recraft-v4" },
    { id: "bytedance-seed/seedream-5-0-flash", canonical_slug: "seedream" },
  ];
  assert.equal(findModelRecord(list, "recraft/recraft-v4").id, "recraft/recraft-v4");
  assert.equal(findModelRecord(list, "SEEDREAM").id, "bytedance-seed/seedream-5-0-flash");
  // Unknown family never borrows another model's capabilities.
  assert.equal(findModelRecord(list, "openai/gpt-image-1"), null);
});

test("image caps: live metadata wins and the type specialist is flagged", () => {
  const list = [{
    id: "recraft/recraft-v4", name: "Recraft V4",
    supported_parameters: {
      aspect_ratio: { type: "enum", enum: ["1:1", "4:3", "3:4", "16:9", "9:16", "auto"] },
      n: { type: "range", min: 1, max: 1 },
    },
  }];
  const caps = imageCapabilities("recraft/recraft-v4", list);
  assert.equal(caps.source, "live");
  assert.deepEqual(caps.aspects, ["1:1", "4:3", "3:4", "16:9", "9:16", "auto"]);
  assert.equal(caps.maxN, 1);
  // 4:5 is exactly what the old hardcoded UI offered and Recraft refuses.
  assert.ok(!caps.aspects.includes("4:5"));
  assert.equal(caps.typeSpecialist, true);
});

test("image caps: unknown family falls back to the family table, not to another model", () => {
  const recraft = imageCapabilities("recraft/recraft-v4", []);
  assert.equal(recraft.source, "known");
  assert.ok(recraft.aspects.includes("auto"), "recraft family table is used when the list is empty");

  const unknown = imageCapabilities("nobody/unknown-model", []);
  assert.equal(unknown.source, "known");
  assert.deepEqual(unknown.aspects, IMAGE_CAP_FALLBACK.aspects);
  assert.equal(unknown.reference, IMAGE_CAP_FALLBACK.reference);
});

test("video caps: durations/resolutions come from the model, not the default", () => {
  const list = [{
    id: "bytedance/seedance-2.5", name: "Seedance 2.5",
    supported_durations: [6, 8, 10],
    supported_resolutions: ["720p", "1080p"],
    supported_aspect_ratios: ["16:9", "9:16"],
    generate_audio: false,
    allowed_passthrough_parameters: ["input_references"],
  }];
  const caps = videoCapabilities("bytedance/seedance-2.5", list, "genjutsu");
  assert.equal(caps.role, "genjutsu");
  assert.equal(caps.source, "live");
  assert.deepEqual(caps.durations, [6, 8, 10]);
  assert.equal(caps.durations.includes(4), false, "4s is NOT offered by this model");
  assert.deepEqual(caps.resolutions, ["720p", "1080p"]);
  assert.equal(caps.audio, false);
  assert.equal(caps.reference, true);
});

test("video caps: a model with no metadata gets the conservative default", () => {
  const caps = videoCapabilities("heygen/heygen-video-1", []);
  assert.equal(caps.source, "known");
  assert.deepEqual(caps.durations, VIDEO_CAP_FALLBACK.durations);
  assert.deepEqual(caps.resolutions, VIDEO_CAP_FALLBACK.resolutions);
  assert.equal(caps.audio, null);
});

test("payload: every role is present and reports where its values came from", () => {
  const payload = creativeCapabilities(
    {
      imageModel: "bytedance-seed/seedream-5-0-flash",
      designModel: "",
      videoModel: "heygen/heygen-video-1",
      genjutsuModel: "bytedance/seedance-2.5",
      avatarModel: "heygen/avatar-iv",
    },
    {
      images: [{ id: "bytedance-seed/seedream-5-0-flash", supported_parameters: { aspect_ratio: ["1:1", "16:9"] } }],
      videos: [],
    },
  );
  assert.equal(payload.ok, true);
  assert.deepEqual(payload.source, { images: "live", videos: "fallback" });
  assert.equal(payload.roles.image.source, "live");
  assert.deepEqual(payload.roles.image.aspects, ["1:1", "16:9"]);
  // Blank design model inherits the image model.
  assert.equal(payload.roles.design.model, "bytedance-seed/seedream-5-0-flash");
  assert.equal(payload.roles.design.role, "design");
  for (const role of ["image", "design", "video", "genjutsu", "avatar"]) {
    assert.ok(payload.roles[role], `${role} must be present`);
    assert.ok(Array.isArray(payload.roles[role].aspects));
    assert.ok(payload.roles[role].aspects.length > 0);
  }
});
