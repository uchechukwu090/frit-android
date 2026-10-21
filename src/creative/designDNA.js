// ============================================================================
// DESIGN DNA — the transferable half of a visual reference
// ----------------------------------------------------------------------------
// Every reference image has two halves:
//
//   STRUCTURE — HOW it is built: grid, alignment, whitespace, palette
//               temperature, type contrast, motif density, texture.
//   CONTENT  — WHAT it is about: the subject, the motifs, the people.
//
// Only STRUCTURE may cross into our brief. Content always comes from the
// user. That split is the hard guarantee against "the agent just re-skinned
// the reference": the reference can tell us *how* a birthday poster is
// built, never *what* it should be about.
//
// Everything here is pure and deterministic so it is unit-testable without
// a network, a model or an image.
// ============================================================================

export const GRID_FAMILIES = [
  "centered", "split", "rule-of-thirds", "asymmetric",
  "modular", "diagonal", "full-bleed", "editorial-column",
];
export const ALIGNMENTS = ["left", "center", "right"];
export const PALETTE_TEMPS = ["warm", "cool", "neutral", "earthy"];
export const TYPE_CONTRASTS = ["high", "medium", "low"];
export const TYPE_PAIRINGS = ["serif-sans", "sans-sans", "display-sans", "mono-sans", "script-sans"];
export const MOTIF_DENSITIES = ["sparse", "balanced", "dense"];
export const TEXTURES = ["flat", "grain", "gradient", "paper", "photo", "collage"];

// Style vocabulary used to diversify the inspiration queries. Two jobs:
//   1. change WHICH results come back (the same goal must not browse the same
//      three pins forever),
//   2. bias the result set toward a family of styles so the references
//      selected afterwards are far apart from each other.
export const STYLE_TOKENS = [
  "swiss editorial", "bauhaus geometric", "art deco luxe", "risograph print",
  "brutalist raw", "minimal japanese", "y2k collage", "retro seventies",
  "midcentury modern", "luxury serif", "tech gradient", "handmade folk",
];

export const ART_DIRECTIONS = [
  "Bold minimalism: one strong focal element, huge headline, generous whitespace, premium print finish",
  "Rich maximalism: layered textures, vibrant colour-blocking, energetic composition, street-poster energy",
  "Clean editorial: grid-aligned layout, refined serif-sans pairing, muted tones, magazine polish",
  "Tactile analogue: paper grain, risograph texture, imperfect edges, hand-set type",
  "High-contrast modern: near-black field, one electric accent, oversized type, tight crop",
  "Soft celebratory: warm gradient field, bokeh depth, airy type, gentle shadows",
];

const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, Number(n) || 0));
const str = (v, max = 120) => String(v ?? "").replace(/\s+/g, " ").trim().slice(0, max);
const num = (v, fallback = 0) => (Number.isFinite(Number(v)) ? Number(v) : fallback);

// ---------------------------------------------------------------------------
// SEEDING — the run must never depend on wall-clock luck
// ---------------------------------------------------------------------------

/** FNV-1a over the joined parts. Same parts -> same seed, always. */
export function hashSeed(...parts) {
  const s = parts.filter((p) => p !== undefined && p !== null && p !== "").join("|");
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

export function seededIndex(len, seed) {
  return len > 0 ? Math.abs(Number(seed) || 0) % len : 0;
}

export function seededPick(arr, seed) {
  return Array.isArray(arr) && arr.length ? arr[seededIndex(arr.length, seed)] : undefined;
}

/** Rotate an array by the seed — the cheap way to make "top 3 of the same
 *  search" different on every run without changing the ranking itself. */
export function seededRotate(arr, seed) {
  if (!Array.isArray(arr) || !arr.length) return [];
  const k = seededIndex(arr.length, seed);
  return [...arr.slice(k), ...arr.slice(0, k)];
}

/** A style token for a run. Derived from the seed so the same user request
 *  browses a DIFFERENT style family next time. */
export function styleTokenFor(seed, tokens = STYLE_TOKENS) {
  return seededPick(tokens, seed) || tokens[0];
}

// The FINISH is drawn from a different pool than the composition, with a
// different hash, so two runs rarely land on the same visual route.
export const FINISH_TREATMENTS = [
  "flat vector finish, no gradients, no drop shadows",
  "risograph two-colour overprint with visible grain",
  "soft studio light, gentle vignette, generous negative space",
  "hard daylight, crisp shadows, saturated colour blocking",
  "paper collage, torn deckled edges, printed texture",
  "neon night light, wet reflective surfaces",
  "clean print-ready poster, flat CMYK-safe colour",
  "cinematic grade, cool shadows, warm highlights",
];

/** The art direction line appended to the render prompt — seed-derived,
 *  because the image model is near-deterministic for an identical prompt. */
export function artDirectionFor(seed, directions = ART_DIRECTIONS, finishes = FINISH_TREATMENTS) {
  const a = seededPick(directions, seed) || directions[0];
  const f = seededPick(finishes, hashSeed(seed, "finish")) || finishes[0];
  return `${a}. Finish: ${f}`;
}

// ---------------------------------------------------------------------------
// COLOUR MATH
// ---------------------------------------------------------------------------

export function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || "").trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

export function rgbToHsl({ r, g, b }) {
  const rr = r / 255, gg = g / 255, bb = b / 255;
  const max = Math.max(rr, gg, bb), min = Math.min(rr, gg, bb);
  const l = (max + min) / 2;
  const d = max - min;
  let h = 0, s = 0;
  if (d !== 0) {
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === rr) h = ((gg - bb) / d + (gg < bb ? 6 : 0)) * 60;
    else if (max === gg) h = ((bb - rr) / d + 2) * 60;
    else h = ((rr - gg) / d + 4) * 60;
  }
  return { h: Math.round(h), s: Math.round(s * 100) / 100, l: Math.round(l * 100) / 100 };
}

/** Hue bucket 0..11 (30 degrees each). Used to force a palette off the
 *  reference's hue band. */
export function hueBand(hex) {
  const rgb = hexToRgb(hex);
  if (!rgb) return -1;
  return Math.floor(rgbToHsl(rgb).h / 30) % 12;
}

/** Warm / cool / neutral / earthy for a set of hexes. Saturation is NOT a
 *  temperature — that lives on its own axis (`palette_mood`), so a vivid
 *  orange is still WARM. */
export function paletteTemp(hexes) {
  const list = (Array.isArray(hexes) ? hexes : [hexes]).filter(Boolean).map(hexToRgb).filter(Boolean);
  if (!list.length) return "neutral";
  const hsl = list.map(rgbToHsl);
  const sat = hsl.reduce((n, c) => n + c.s, 0) / hsl.length;
  const light = hsl.reduce((n, c) => n + c.l, 0) / hsl.length;
  if (sat < 0.14 || (sat < 0.3 && light > 0.82)) return "neutral";
  // circular mean of hue, weighted by saturation
  let x = 0, y = 0;
  for (const c of hsl) {
    const rad = (c.h * Math.PI) / 180;
    x += Math.cos(rad) * c.s;
    y += Math.sin(rad) * c.s;
  }
  let h = (Math.atan2(y, x) * 180) / Math.PI;
  if (h < 0) h += 360;
  if (h >= 340 || h <= 75) return "warm";
  if (h >= 160 && h <= 300) return "cool";
  return "earthy";
}

// ---------------------------------------------------------------------------
// ENUM NORMALISATION — the model speaks in prose, the code speaks in enums
// ---------------------------------------------------------------------------

function normEnum(value, allowed, fallback) {
  const v = String(value ?? "").trim().toLowerCase().replace(/[\s_]+/g, "-");
  if (!v) return fallback;
  if (allowed.includes(v)) return v;
  for (const a of allowed) {
    if (v.includes(a)) return a;
  }
  return fallback;
}

/** Map any free-text grid description onto a grid family. */
export function gridFamily(value) {
  const v = String(value ?? "").toLowerCase();
  if (!v) return "rule-of-thirds";
  if (/rule|third|grid|classic/.test(v)) return "rule-of-thirds";
  if (/center|centre|middle|axis|radial|mandala/.test(v)) return "centered";
  if (/split|half|two.column|duo|bipart/.test(v)) return "split";
  if (/asym|offset|off.balance|broken/.test(v)) return "asymmetric";
  if (/modular|block|tile|grid.system|swiss.module/.test(v)) return "modular";
  if (/diagonal|slash|angle|tilt/.test(v)) return "diagonal";
  if (/full.bleed|edge.to.edge|bleed|immersive/.test(v)) return "full-bleed";
  if (/column|editorial|magazine|article/.test(v)) return "editorial-column";
  return "rule-of-thirds";
}

/** Derive the type pairing from the brief's font names. A single serif
 *  display face over a neutral body still reads as a serif/sans pairing. */
export function typePairingOf(fonts) {
  const list = (Array.isArray(fonts) ? fonts : [fonts]).map((f) => String(f || "").toLowerCase());
  const has = (re) => list.some((f) => re.test(f));
  if (has(/mono|code|terminal/)) return "mono-sans";
  if (has(/script|hand|callig|brush|marker/)) return "script-sans";
  if (has(/serif|roman|garamond|didot|bodoni|playfair|lora|times|georgia|trajan|baskerville/)) return "serif-sans";
  if (has(/display|black|heavy|fat|impact/)) return "display-sans";
  return "sans-sans";
}

export function normalizePalette(list) {
  const out = [];
  for (const p of Array.isArray(list) ? list.slice(0, 5) : []) {
    const raw = typeof p === "string" ? p : p?.hex;
    const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(String(raw || "").trim());
    if (!m) continue;
    let hex = `#${m[1].toUpperCase()}`;
    if (hex.length === 4) hex = `#${hex[1]}${hex[1]}${hex[2]}${hex[2]}${hex[3]}${hex[3]}`;
    if (out.some((c) => c.hex === hex)) continue;
    out.push({
      hex,
      role: str(typeof p === "object" ? p?.role : "", 24) || (out.length ? "secondary" : "dominant"),
      share_pct: clamp(num(typeof p === "object" ? p?.share_pct : 0, 0), 0, 100),
    });
  }
  return out;
}

export function normalizeTypography(t) {
  if (!t || typeof t !== "object") return {};
  const pick = (v, max = 60) => str(v, max);
  const out = {};
  if (t.display) out.display = pick(t.display);
  if (t.body) out.body = pick(t.body);
  if (t.sizes) out.sizes = pick(t.sizes);
  if (t.case) out.case = pick(t.case, 16);
  if (t.contrast) out.contrast = normEnum(t.contrast, TYPE_CONTRASTS, "medium");
  return out;
}

// ---------------------------------------------------------------------------
// THE SPLIT
// ---------------------------------------------------------------------------

/**
 * Split a raw vision-model DNA blob into structure + content.
 *
 * Accepts the v2 schema ({structure, content}) and the older flat schema
 * ({layout, typography, palette, ...}) so a model that ignores the new
 * instruction still produces something usable.
 */
export function splitDNA(raw) {
  const r = raw && typeof raw === "object" ? raw : {};
  const s = r.structure && typeof r.structure === "object" ? r.structure : {};
  const c = r.content && typeof r.content === "object" ? r.content : {};
  const legacyLayout = r.layout && typeof r.layout === "object" ? r.layout : {};
  const legacyTypo = r.typography && typeof r.typography === "object" ? r.typography : {};

  const structure = {
    grid: normEnum(s.grid ?? legacyLayout.grid, GRID_FAMILIES, "rule-of-thirds"),
    alignment: normEnum(s.alignment ?? legacyLayout.alignment, ALIGNMENTS, "left"),
    focal: str(s.focal ?? legacyLayout.focal, 60),
    whitespace_pct: Math.round(clamp(num(s.whitespace_pct, 22), 0, 80)),
    palette_temp: normEnum(s.palette_temp, PALETTE_TEMPS, "neutral"),
    palette_mood: str(s.palette_mood, 40),
    type_contrast: normEnum(s.type_contrast, TYPE_CONTRASTS, "medium"),
    type_pairing: normEnum(s.type_pairing ?? legacyTypo.contrast, TYPE_PAIRINGS, "sans-sans"),
    motif_density: normEnum(s.motif_density, MOTIF_DENSITIES, "balanced"),
    texture: normEnum(s.texture, TEXTURES, "flat"),
    hierarchy_ratio: str(s.hierarchy_ratio, 16),
    imagery_treatment: str(s.imagery_treatment, 80),
  };

  const content = {
    subject: str(c.subject, 140),
    motifs: (Array.isArray(c.motifs) ? c.motifs : []).map((m) => str(m, 60)).filter(Boolean).slice(0, 8),
    notes: str(c.notes, 160),
  };

  const palette = normalizePalette(r.palette);
  const typography = normalizeTypography(r.typography);
  const mood = str(r.mood, 80);

  return { structure, content, palette, typography, mood };
}

/** The ONLY part of a reference that may be shown to the brief writer.
 *  Content is dropped on purpose — this is the anti-copy guarantee. */
export function structuralOnly(dna) {
  const split = dna && dna.structure ? dna : splitDNA(dna);
  return {
    structure: split.structure,
    palette: split.palette,
    typography: split.typography,
    mood: split.mood,
  };
}

// ---------------------------------------------------------------------------
// DIVERSITY — pick references that are FAR apart, not the top 3 of one search
// ---------------------------------------------------------------------------

export function structureSignature(st) {
  const s = st || {};
  return [
    s.grid, s.alignment, s.palette_temp, s.type_pairing, s.motif_density, s.texture,
  ];
}

/** Number of structural axes on which two references disagree (0..6). */
export function structureDistance(a, b) {
  const A = structureSignature(a), B = structureSignature(b);
  let d = 0;
  for (let i = 0; i < A.length; i++) if (A[i] !== B[i]) d++;
  return d;
}

/**
 * Farthest-point selection: take the first candidate, then repeatedly take
 * the candidate that is structurally FURTHEST from everything already chosen.
 * Two pins from the same Pinterest board score 0-1 apart and lose to a pin
 * from a different grid + palette family.
 *
 * items: [{ structure, ... }]
 */
export function selectDiverse(items, { max = 3 } = {}) {
  if (max <= 0) return [];
  const list = (Array.isArray(items) ? items : []).filter((x) => x && x.structure);
  if (list.length <= max) return [...list];
  const chosen = [list[0]];
  const rest = list.slice(1);
  while (chosen.length < max && rest.length) {
    let bestIdx = 0;
    let bestScore = -1;
    for (let i = 0; i < rest.length; i++) {
      const cand = rest[i];
      const nearest = Math.min(...chosen.map((ch) => structureDistance(ch.structure, cand.structure)));
      if (nearest > bestScore) { bestScore = nearest; bestIdx = i; }
    }
    chosen.push(rest.splice(bestIdx, 1)[0]);
  }
  return chosen;
}

// ---------------------------------------------------------------------------
// ANTI-ANCHOR — the brief must not be a re-skin of the reference
// ---------------------------------------------------------------------------

/** The structural axes a brief occupies. */
export function briefAxes(brief) {
  const b = brief || {};
  const fonts = (b.typography || []).map((t) => t.font);
  return {
    grid: gridFamily(b.layout?.grid),
    alignment: String(b.layout?.alignment || "left").toLowerCase(),
    temp: paletteTemp((b.palette || []).map((p) => p.hex)),
    pairing: typePairingOf(fonts),
    density: (b.elements || []).length >= 5 ? "dense" : (b.elements || []).length >= 2 ? "balanced" : "sparse",
    texture: b.background?.description?.match(/grain|texture|paper/i) ? "paper" : "flat",
  };
}

/** How many structural axes the brief shares with a reference. */
export function sharedAxes(briefAxes, refStructure) {
  const ref = {
    grid: refStructure.grid,
    alignment: refStructure.alignment,
    temp: refStructure.palette_temp,
    pairing: refStructure.type_pairing,
    density: refStructure.motif_density,
    texture: refStructure.texture,
  };
  let n = 0;
  for (const k of Object.keys(ref)) if (briefAxes[k] === ref[k]) n++;
  return n;
}

function nextDifferent(list, current) {
  const i = list.indexOf(current);
  return list[(i + 1 + list.length) % list.length];
}

/**
 * If the brief mirrors a reference on too many axes, push it away: a
 * different grid family, a different alignment, a different type pairing and
 * (only when the user did not pin a palette) a different hue band.
 *
 * Mutates and returns the brief; also returns what changed so the UI can
 * show the reasoning.
 */
export function enforceDivergence(brief, structures, { seed = 1, maxShared = 3, allowPaletteShift = true } = {}) {
  const refs = (Array.isArray(structures) ? structures : []).filter(Boolean);
  if (!refs.length || !brief) return { brief, changed: [] };
  const changed = [];
  const axes = briefAxes(brief);

  for (const ref of refs) {
    if (sharedAxes(axes, ref) < maxShared) continue;

    if (axes.grid === ref.grid) {
      const next = nextDifferent(GRID_FAMILIES, ref.grid);
      brief.layout = { ...(brief.layout || {}), grid: next };
      axes.grid = next;
      changed.push(`grid → ${next}`);
    }
    if (axes.alignment === ref.alignment) {
      const next = nextDifferent(ALIGNMENTS, ref.alignment);
      brief.layout = { ...(brief.layout || {}), alignment: next };
      axes.alignment = next;
      changed.push(`alignment → ${next}`);
    }
    if (axes.pairing === ref.type_pairing) {
      const next = nextDifferent(TYPE_PAIRINGS, ref.type_pairing);
      for (const t of brief.typography || []) {
        t.font = t.role === "headline" && ref.type_pairing === "serif-sans" ? "grotesk sans" : "humanist serif";
      }
      axes.pairing = next;
      changed.push(`type pairing → ${next}`);
    }
    if (allowPaletteShift && axes.temp === ref.palette_temp) {
      const shifted = shiftPaletteHue(brief.palette || [], seed);
      if (shifted.length) {
        brief.palette = shifted;
        axes.temp = paletteTemp(shifted.map((p) => p.hex));
        changed.push("palette hue band shifted");
      }
    }
    break; // one reference is enough to trigger the push
  }
  return { brief, changed };
}

/** Rotate the palette's hue bands by one bucket so the colours land in a
 *  different family from the reference (brand-pinned palettes are never
 *  passed here — the caller gates that with allowPaletteShift). */
export function shiftPaletteHue(palette, seed = 1) {
  const list = Array.isArray(palette) ? palette : [];
  if (!list.length) return [];
  const step = (Math.abs(Number(seed) || 1) % 3) + 1; // 1..3 buckets = 30..90 degrees
  const out = [];
  for (const c of list) {
    const rgb = hexToRgb(c.hex);
    if (!rgb) { out.push(c); continue; }
    const hsl = rgbToHsl(rgb);
    const nh = (hsl.h + step * 30) % 360;
    out.push({ ...c, hex: hslToHex(nh, hsl.s, hsl.l) });
  }
  return out;
}

export function hslToHex(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = (((h % 360) + 360) % 360) / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0, g = 0, b = 0;
  if (hp < 1) { r = c; g = x; } else if (hp < 2) { r = x; g = c; }
  else if (hp < 3) { g = c; b = x; } else if (hp < 4) { g = x; b = c; }
  else if (hp < 5) { r = x; b = c; } else { r = c; b = x; }
  const m = l - c / 2;
  const to = (v) => Math.round((v + m) * 255).toString(16).padStart(2, "0").toUpperCase();
  return `#${to(r)}${to(g)}${to(b)}`;
}
