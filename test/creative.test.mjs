// Unit tests for the pure Creative Suite functions (no network, no DB).
// Run: npm test  (node --test)
import { test } from "node:test";
import assert from "node:assert/strict";

import { pickCraft, buildDesignPrompt, variationCount } from "../src/creative/design.js";
import { CINEMA_OPTIONS, SHOT_RECIPES, buildDirectorPrompt } from "../src/creative/cinema.js";
import {
  TEMPLATES,
  GRAPHICS_TEMPLATES,
  EXPORT_PRESETS,
  getTemplates,
  getTemplate,
  getGraphicsTemplate,
  applyTemplate,
} from "../src/creative/templateLibrary.js";
import { STYLE_PRESETS } from "../src/creative/promptEngine.js";

// ---------------- design.js: craft routing ----------------

test("pickCraft routes copy-heavy briefs to the typography model", () => {
  assert.equal(pickCraft({ headline: "Mega Clearance Sale Today" }), "type");
});

test("pickCraft routes short wordmarks and empty briefs to vector", () => {
  assert.equal(pickCraft({}), "vector");
  assert.equal(pickCraft({ headline: "FRIT" }), "vector");
  assert.equal(pickCraft({ headline: "Mega Clearance Sale" }), "vector"); // 3 words is still a wordmark
});

test("pickCraft routes any body copy to typography regardless of headline length", () => {
  assert.equal(pickCraft({ headline: "FRIT", body: "Since 2019" }), "type");
});

// ---------------- design.js: prompt construction ----------------

test("buildDesignPrompt quotes headline exactly (typography models render quoted copy)", () => {
  const p = buildDesignPrompt({ headline: "Owambe Saturday" });
  assert.ok(p.includes('"Owambe Saturday"'), "headline must appear as an exact quoted string");
  assert.ok(p.includes("render EXACTLY as quoted"));
});

test("buildDesignPrompt applies the 60-30-10 color rule from brand + palette", () => {
  const p = buildDesignPrompt({
    headline: "Sale",
    palette: "gold",
    brand: { colors: ["#111111", "#222222"] },
  });
  assert.ok(p.includes("dominant #111111 60%"));
  assert.ok(p.includes("secondary #222222 30%"));
  assert.ok(p.includes("accent gold 10%"));
});

test("buildDesignPrompt variations are distinct art directions that cycle", () => {
  const base = { headline: "Hello" };
  const v0 = buildDesignPrompt({ ...base, variation: 0 });
  const v1 = buildDesignPrompt({ ...base, variation: 1 });
  const v3 = buildDesignPrompt({ ...base, variation: 3 });
  assert.notEqual(v0, v1, "variation 0 and 1 must differ");
  assert.equal(v0, v3, "variation 3 wraps back to art direction 0");
});

test("buildDesignPrompt always forbids gibberish text and watermarks", () => {
  const p = buildDesignPrompt({ headline: "Any" });
  assert.ok(p.includes("no gibberish text"));
  assert.ok(p.includes("no watermark"));
});

test("variationCount clamps to 1..3 and coerces strings", () => {
  assert.equal(variationCount(0), 1);
  assert.equal(variationCount(2), 2);
  assert.equal(variationCount(5), 3);
  assert.equal(variationCount("4"), 3);
  assert.equal(variationCount("nonsense"), 1);
});

// ---------------- cinema.js: director prompt ----------------

test("buildDirectorPrompt describes shallow vs deep focus from aperture", () => {
  assert.ok(buildDirectorPrompt({ aperture: "f/1.4" }).includes("shallow depth of field"));
  assert.ok(buildDirectorPrompt({ aperture: "f/8" }).includes("deep focus"));
});

test("buildDirectorPrompt always includes a negative prompt", () => {
  assert.ok(buildDirectorPrompt({}).includes("Avoid: blurry"));
  const custom = buildDirectorPrompt({ negative: "no text please" });
  assert.ok(custom.includes("Avoid: no text please"));
  assert.ok(!custom.includes("watermark, text overlay"), "custom negative must replace the default");
});

test("buildDirectorPrompt omits the era line for Timeless", () => {
  assert.ok(!buildDirectorPrompt({ era: "Timeless" }).includes("Era aesthetic"));
  assert.ok(buildDirectorPrompt({ era: "1980s" }).includes("Era aesthetic: 1980s"));
});

test("buildDirectorPrompt names the camera and color grade", () => {
  const p = buildDirectorPrompt({ camera: "Alexa 65", palette: "Nollywood Glow" });
  assert.ok(p.includes("Shot on Alexa 65"));
  assert.ok(p.includes("Color grade: Nollywood Glow"));
});

// ---------------- cinema.js: shot recipe integrity ----------------

test("SHOT_RECIPES have unique ids", () => {
  const ids = SHOT_RECIPES.map((r) => r.id);
  assert.equal(new Set(ids).size, ids.length, "duplicate recipe ids");
});

test("every SHOT_RECIPE param uses a value that exists in CINEMA_OPTIONS", () => {
  const listChecks = {
    genre: CINEMA_OPTIONS.genres,
    era: CINEMA_OPTIONS.eras,
    tempo: CINEMA_OPTIONS.tempos,
    camera: CINEMA_OPTIONS.cameras,
    lens: CINEMA_OPTIONS.lenses,
    aperture: CINEMA_OPTIONS.apertures,
    move: CINEMA_OPTIONS.moves,
    angle: CINEMA_OPTIONS.angles,
    lighting: CINEMA_OPTIONS.lighting,
    palette: CINEMA_OPTIONS.palettes,
    emotion: CINEMA_OPTIONS.emotions,
  };
  for (const recipe of SHOT_RECIPES) {
    for (const [key, allowed] of Object.entries(listChecks)) {
      const v = recipe.params[key];
      if (v == null) continue;
      assert.ok(
        allowed.includes(v),
        `recipe "${recipe.id}" has ${key}="${v}" not present in CINEMA_OPTIONS.${key}`,
      );
    }
  }
});

test("CINEMA_OPTIONS exposes every surface the Director's Panel needs", () => {
  for (const key of ["genres", "eras", "tempos", "cameras", "lenses", "apertures", "moves", "angles", "lighting", "palettes", "emotions", "aspect_ratios"]) {
    assert.ok(Array.isArray(CINEMA_OPTIONS[key]) && CINEMA_OPTIONS[key].length > 0, `missing options: ${key}`);
  }
});

// ---------------- templateLibrary.js ----------------

test("getTemplates filters by category", () => {
  const posts = getTemplates("social_post", null);
  assert.ok(posts.length > 0);
  for (const t of posts) assert.equal(t.category, "social_post");
});

test("getTemplates platform filter includes 'multi' platform templates", () => {
  const tiktok = getTemplates(null, "tiktok");
  assert.ok(tiktok.length > 0);
  for (const t of tiktok) assert.ok(t.platform === "tiktok" || t.platform === "multi");
  assert.ok(tiktok.some((t) => t.platform === "multi"), "multi-platform templates must surface for any platform filter");
});

test("getTemplate / getGraphicsTemplate return null for unknown ids", () => {
  assert.equal(getTemplate("no_such_template"), null);
  assert.equal(getGraphicsTemplate("no_such_template"), null);
});

test("applyTemplate maps assets onto zones in order", () => {
  const out = applyTemplate("tpl_yt_thumb", ["asset-url-1"]);
  assert.equal(out.aspect_ratio, "16:9");
  assert.equal(out.zones[0].asset, "asset-url-1");
  assert.equal(out.zones[1].asset, null, "zones beyond the asset list must be null, not crash");
});

test("applyTemplate errors cleanly on unknown ids", () => {
  assert.deepEqual(applyTemplate("missing", []), { error: "Template not found" });
});

test("template ids are unique across both libraries", () => {
  const ids = [...TEMPLATES, ...GRAPHICS_TEMPLATES].map((t) => t.id);
  assert.equal(new Set(ids).size, ids.length, "duplicate template ids");
});

test("every export preset has a parseable resolution and mp4 format", () => {
  for (const [platform, preset] of Object.entries(EXPORT_PRESETS)) {
    const [w, h] = String(preset.resolution).split("x").map(Number);
    assert.ok(w > 0 && h > 0, `${platform}: unparseable resolution ${preset.resolution}`);
    assert.equal(preset.format, "mp4", `${platform}: unexpected format`);
    assert.ok(preset.aspect_ratio && preset.fps > 0, `${platform}: incomplete preset`);
  }
});

// ---------------- promptEngine.js ----------------

test("STYLE_PRESETS have unique ids and cover the core styles", () => {
  const ids = STYLE_PRESETS.map((p) => p.id);
  assert.equal(new Set(ids).size, ids.length, "duplicate style preset ids");
  for (const id of ["realistic", "cartoon", "anime", "cyberpunk"]) {
    assert.ok(ids.includes(id), `missing core style preset: ${id}`);
  }
});
