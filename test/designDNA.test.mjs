// ============================================================================
// Design DNA tests — the anti-copy guarantees
// ----------------------------------------------------------------------------
// Hermetic: no network, no model, no image. These assert the RULES:
//   * a reference's CONTENT (subject/motifs) can never reach the brief,
//   * selected references are structurally far apart, not top-of-search,
//   * a brief that mirrors a reference is pushed away from it,
//   * the same request produces different prompts on different runs.
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";

import {
  hashSeed, seededPick, seededRotate, styleTokenFor, artDirectionFor,
  splitDNA, structuralOnly, selectDiverse, structureDistance,
  enforceDivergence, briefAxes, sharedAxes,
  paletteTemp, hueBand, gridFamily, typePairingOf,
  GRID_FAMILIES, STYLE_TOKENS,
} from "../src/creative/designDNA.js";

// ------------------------------------------------------------------ seed ---

test("hashSeed: deterministic for the same parts, different for different ones", () => {
  assert.equal(hashSeed("a", "b"), hashSeed("a", "b"));
  assert.notEqual(hashSeed("a", "b"), hashSeed("a", "c"));
  assert.equal(typeof hashSeed("x"), "number");
});

test("seededPick / seededRotate: same seed -> same pick, and rotation is a rotation", () => {
  const arr = ["a", "b", "c", "d"];
  assert.equal(seededPick(arr, 7), seededPick(arr, 7));
  assert.ok(arr.includes(seededPick(arr, 99)));
  const rot = seededRotate(arr, 2);
  assert.equal(rot.length, arr.length);
  assert.deepEqual([...rot].sort(), [...arr].sort());
});

test("styleTokenFor / artDirectionFor: seed-derived and always from the roster", () => {
  assert.ok(STYLE_TOKENS.includes(styleTokenFor(11)));
  assert.ok(STYLE_TOKENS.includes(styleTokenFor(9999)));
  assert.notEqual(styleTokenFor(1), styleTokenFor(2) + "___never");
  assert.ok(typeof artDirectionFor(5) === "string" && artDirectionFor(5).length > 10);
});

test("artDirectionFor: two seeds rarely land on the same visual route", () => {
  const dirs = new Set([1, 2, 3, 4, 5, 6, 7, 8].map((s) => artDirectionFor(s)));
  assert.ok(dirs.size >= 6, `expected variety, got ${dirs.size} distinct directions`);
  assert.ok(artDirectionFor(7).includes("Finish:"), "the direction carries a finish too");
});

// ------------------------------------------------------------------- dna ---

const V2_DNA = {
  structure: {
    grid: "centered", alignment: "center", focal: "middle",
    whitespace_pct: 30, palette_temp: "warm", type_contrast: "high",
    type_pairing: "serif-sans", motif_density: "sparse", texture: "paper",
  },
  palette: [{ hex: "#ff0000", role: "dominant", share_pct: 60 }],
  typography: { display: "Didot", body: "Inter", contrast: "high" },
  content: { subject: "a birthday cake with candles", motifs: ["balloons", "confetti"] },
  mood: "celebratory",
};

test("splitDNA: v2 schema keeps structure and content apart", () => {
  const d = splitDNA(V2_DNA);
  assert.equal(d.structure.grid, "centered");
  assert.equal(d.structure.palette_temp, "warm");
  assert.equal(d.content.subject, "a birthday cake with candles");
  assert.deepEqual(d.content.motifs, ["balloons", "confetti"]);
  assert.equal(d.palette[0].hex, "#FF0000");
});

test("splitDNA: the legacy flat schema still parses", () => {
  const d = splitDNA({
    layout: { grid: "rule of thirds", alignment: "left" },
    typography: { display: "Grotesk" },
    palette: [{ hex: "#0e3b2e", role: "dominant" }],
    mood: "warm",
  });
  assert.equal(d.structure.grid, "rule-of-thirds");
  assert.equal(d.structure.alignment, "left");
  assert.equal(d.typography.display, "Grotesk");
  assert.equal(d.palette[0].hex, "#0E3B2E");
});

test("splitDNA: out-of-range enum values fall back instead of corrupting the axes", () => {
  const d = splitDNA({ structure: { grid: "spiral", alignment: "sideways", palette_temp: "lava" } });
  assert.ok(GRID_FAMILIES.includes(d.structure.grid));
  assert.ok(["left", "center", "right"].includes(d.structure.alignment));
  assert.ok(d.structure.palette_temp.length > 0);
});

test("structuralOnly: the CONTENT bucket never survives the crossing", () => {
  const d = splitDNA(V2_DNA);
  const so = structuralOnly(d);
  const json = JSON.stringify(so);
  assert.ok(!json.includes("birthday cake"));
  assert.ok(!json.includes("balloons"));
  assert.ok(json.includes("centered"));
});

test("structuralOnly: a raw (un-split) DNA blob is split first", () => {
  const so = structuralOnly(V2_DNA);
  assert.ok(!JSON.stringify(so).includes("birthday cake"));
  assert.equal(so.structure.grid, "centered");
});

// ------------------------------------------------------------- diversity ---

const refA = { ref: "a", structure: { grid: "centered", alignment: "center", palette_temp: "warm", type_pairing: "serif-sans", motif_density: "sparse", texture: "paper" } };
const refB = { ref: "b", structure: { grid: "centered", alignment: "center", palette_temp: "warm", type_pairing: "serif-sans", motif_density: "sparse", texture: "paper" } };
const refC = { ref: "c", structure: { grid: "diagonal", alignment: "right", palette_temp: "cool", type_pairing: "mono-sans", motif_density: "dense", texture: "grain" } };

test("structureDistance: identical structures score 0, different ones score higher", () => {
  assert.equal(structureDistance(refA.structure, refB.structure), 0);
  assert.ok(structureDistance(refA.structure, refC.structure) >= 5);
});

test("selectDiverse: prefers the far reference over a duplicate of the first", () => {
  const chosen = selectDiverse([refA, refB, refC], { max: 2 });
  assert.equal(chosen.length, 2);
  assert.equal(chosen[0].ref, "a");
  assert.equal(chosen[1].ref, "c", "the near-duplicate must lose to the structurally distant one");
});

test("selectDiverse: never returns more than max, and tolerates a short list", () => {
  assert.equal(selectDiverse([refA], { max: 3 }).length, 1);
  assert.equal(selectDiverse([], { max: 3 }).length, 0);
  assert.equal(selectDiverse([refA, refB, refC], { max: 0 }).length, 0);
});

// ------------------------------------------------------------ divergence ---

function briefLike(structure, extra = {}) {
  return {
    layout: { grid: structure.grid, alignment: structure.alignment, margins_pct: 8 },
    palette: [{ hex: "#112233", role: "dominant", share_pct: 60 }, { hex: "#FFEEDD", role: "accent", share_pct: 40 }],
    typography: [{ role: "headline", font: "serif display", size_px: 90, weight: 800, tracking: -1, case: "UPPER" }],
    elements: [{ type: "icon", description: "x", placement: "top", size_pct: 10 }],
    background: { description: "flat", color_hex: "#FFEEDD" },
    copy: { headline: "HI" },
    ...extra,
  };
}

test("briefAxes: reads the brief's own structural fingerprint", () => {
  const axes = briefAxes(briefLike(refA.structure));
  assert.equal(axes.grid, "centered");
  assert.equal(axes.alignment, "center");
  assert.equal(axes.pairing, "serif-sans");
  assert.equal(axes.density, "sparse");
});

test("sharedAxes: counts how many axes a brief mirrors in a reference", () => {
  const axes = briefAxes(briefLike(refA.structure));
  assert.ok(sharedAxes(axes, refA.structure) >= 4);
  assert.ok(sharedAxes(axes, refC.structure) <= 2);
});

test("enforceDivergence: a brief that mirrors a reference is pushed off it", () => {
  const brief = briefLike(refA.structure);
  const { brief: out, changed } = enforceDivergence(brief, [refA.structure], { seed: 3, allowPaletteShift: true });
  assert.ok(changed.length > 0, "something must change");
  assert.notEqual(out.layout.grid, refA.structure.grid, "the grid must move");
  assert.notEqual(out.layout.alignment, refA.structure.alignment, "the alignment must move");
});

test("enforceDivergence: a pinned palette is never hue-shifted", () => {
  const brief = briefLike(refA.structure);
  const before = JSON.stringify(brief.palette);
  const { brief: out } = enforceDivergence(brief, [refA.structure], { seed: 3, allowPaletteShift: false });
  assert.equal(JSON.stringify(out.palette), before, "brand/user colours are untouchable");
});

test("enforceDivergence: a brief that is already different is left alone", () => {
  const brief = briefLike(refC.structure, {
    palette: [{ hex: "#0044AA", role: "dominant", share_pct: 60 }, { hex: "#111111", role: "accent", share_pct: 40 }],
    typography: [{ role: "headline", font: "Roboto Mono", size_px: 90, weight: 800, tracking: -1, case: "UPPER" }],
    elements: [1, 2, 3, 4, 5].map(() => ({ type: "icon", description: "x", placement: "top", size_pct: 10 })),
    background: { description: "coarse grain texture", color_hex: "#0044AA" },
  });
  const before = JSON.stringify(brief.layout);
  const { changed } = enforceDivergence(brief, [refA.structure], { seed: 3 });
  assert.equal(changed.length, 0, "a brief that already differs must not be touched");
  assert.equal(JSON.stringify(brief.layout), before);
});

// ---------------------------------------------------------------- colour ---

test("paletteTemp: warm, cool and neutral hexes classify correctly", () => {
  assert.equal(paletteTemp(["#FF5500", "#FFCC00"]), "warm");
  assert.equal(paletteTemp(["#0044AA", "#00AACC"]), "cool");
  assert.equal(paletteTemp(["#888888", "#777777"]), "neutral");
});

test("hueBand: hexes map onto 30-degree buckets", () => {
  assert.equal(hueBand("#FF0000"), 0);
  assert.equal(hueBand("#00FF00"), 4);
  assert.equal(hueBand("#0000FF"), 8);
  assert.equal(hueBand("nonsense"), -1);
});

test("gridFamily: free-text grid descriptions map onto the grid families", () => {
  assert.equal(gridFamily("rule of thirds"), "rule-of-thirds");
  assert.equal(gridFamily("Centre aligned"), "centered");
  assert.equal(gridFamily("full bleed image"), "full-bleed");
  assert.equal(gridFamily("something else"), "rule-of-thirds");
});

test("typePairingOf: font names map onto the pairing axis", () => {
  assert.equal(typePairingOf(["Playfair Display", "Inter"]), "serif-sans");
  assert.equal(typePairingOf(["Space Grotesk"]), "sans-sans");
  assert.equal(typePairingOf(["Roboto Mono"]), "mono-sans");
});
