// ============================================================================
// DESIGN AGENT — Week 6
// ----------------------------------------------------------------------------
// The Design Studio used to be one shot: template + headline + palette -> image.
// This module turns it into a three-stage agent:
//
//   1. RESEARCH   — search the web for visual references (Pinterest / Behance /
//                   Dribbble are what the user names; keyless sources still work:
//                   Bing image search, then Openverse as a fallback).
//   2. BRIEF      — read each reference with a vision model, then synthesize ONE
//                   detailed brief: canvas, layout grid + alignment, palette
//                   with 60-30-10 shares, typography (font / size / weight /
//                   tracking / case), elements, imagery, background, exact copy.
//   3. RENDER     — assemble a deterministic prompt from the brief, generate,
//                   then VERIFY the render against the brief and retry once with
//                   the verifier's own complaints as corrections.
//
// The anti-copy rules live in ./designDNA.js:
//   * a reference is split into STRUCTURE (grid, whitespace, palette
//     temperature, type contrast — borrowable) and CONTENT (subject, motifs —
//     never borrowable). Only structure crosses into the brief.
//   * references are picked for MAXIMUM STRUCTURAL SPREAD, not top-of-search,
//     so three pins from one board cannot masquerade as three references.
//   * every run carries a SEED: the query style token, the reference order and
//     the art direction all derive from it, so the same request never browses
//     the same three images twice.
//   * after the brief is written it is PUSHED AWAY from the references
//     (enforceDivergence) so the result is a new design, not a re-skin.
//
// Everything is dependency-injected (`chat`, `generate`, `fetchImpl`) so the
// whole pipeline is unit-testable with no network: see test/designAgent.test.mjs.
// Pure helpers (queries, URL ranking, brief normalisation, prompt assembly,
// verification judgement) are exported individually.
// ============================================================================

import {
  hashSeed, styleTokenFor, artDirectionFor,
  splitDNA, structuralOnly, selectDiverse, enforceDivergence,
  briefAxes, sharedAxes,
} from "./designDNA.js";

const DEFAULT_ASPECT = "1:1";
const DEFAULT_PALETTE = [
  { hex: "#0E3B2E", role: "dominant", share_pct: 60 },
  { hex: "#F4EFE4", role: "secondary", share_pct: 30 },
  { hex: "#D4A574", role: "accent", share_pct: 10 },
];
const TYPE_ROLES = ["headline", "subhead", "body", "cta"];
// Poster sizes, not document sizes. The old body floor (12px) produced copy
// that was literally unreadable on a 1080 canvas — the reviewer called the
// body "missing" because 16px type is invisible on a poster.
const TYPE_SIZE = {
  headline: [48, 160],
  subhead: [24, 64],
  body: [20, 40],
  cta: [16, 48],
};

const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, Number(n) || 0));

// ---------------------------------------------------------------------------
// 1. RESEARCH
// ---------------------------------------------------------------------------

/** Search queries for visual references. Pinterest/Behance/Dribbble get site
 *  filters because that is where design inspiration actually lives.
 *
 *  A STYLE TOKEN derived from the run seed is mixed into every query: it
 *  changes WHICH results come back (the same request must not browse the
 *  same three pins forever) and it biases the pool toward a family of styles
 *  so the selected references end up far apart from each other. */
export function planInspirationQueries(goal, {
  sources = ["pinterest", "behance", "dribbble"],
  maxQueries = 3,
  seed = 0,
  styleToken = null,
} = {}) {
  const g = String(goal || "").replace(/\s+/g, " ").trim().slice(0, 120);
  if (!g) return [];
  const token = String(styleToken || styleTokenFor(seed) || "").trim();
  const tail = token ? ` ${token}` : "";
  const out = [];
  for (const s of sources) {
    const site = String(s || "").toLowerCase().replace(/[^a-z.]/g, "");
    if (site) out.push(`site:${site}.com ${g}${tail} design`);
    if (out.length >= maxQueries) break;
  }
  if (out.length === 0) out.push(`${g}${tail} graphic design inspiration`);
  out.push(`${g}${tail} poster layout color typography inspiration`);
  return [...new Set(out)].slice(0, Math.max(1, maxQueries));
}

const BAD_IMAGE_HINTS = /(sprite|avatar|favicon|logo[-_/]?icon|placeholder|blank\.|pixel\.|spacer|1x1|loading)/i;
const IMAGE_EXT = /\.(jpe?g|png|webp)(?:[?#]|$)/i;

/** True when a URL is a plausible full-size design reference (not a sprite,
 *  avatar, SVG icon or data URI). Deliberately conservative: a wrong reference
 *  wastes a vision call, a missing one costs nothing. */
export function looksLikeUsableImage(url) {
  const u = String(url || "").trim();
  if (!/^https?:\/\//i.test(u)) return false;
  if (BAD_IMAGE_HINTS.test(u)) return false;
  if (/\.(svg|gif|ico)(?:[?#]|$)/i.test(u)) return false;
  // Pinterest/Behance CDNs often omit the extension; allow known image CDNs.
  if (!IMAGE_EXT.test(u) && !/(pinimg|behance|bbci|dribbble|unsplash|openverse|staticflickr)/i.test(u)) return false;
  return true;
}

/** Pull image URLs out of search-result HTML. Bing embeds them as
 *  `murl&quot;:&quot;…&quot;`; pages/searches also expose og:image. */
export function extractImageUrls(html, baseUrl = "") {
  const src = String(html || "");
  const found = [];
  const push = (u) => {
    let v = String(u || "").trim().replace(/^["']|["']$/g, "");
    v = v.replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/\\\//g, "/");
    if (!v) return;
    if (v.startsWith("//")) v = "https:" + v;
    else if (v.startsWith("/") && baseUrl) {
      try { v = new URL(v, baseUrl).href; } catch { return; }
    }
    found.push(v);
  };
  // Bing: murl in the m="" JSON blob (entity-encoded or raw).
  for (const m of src.matchAll(/murl(?:&quot;|")\s*:\s*(?:&quot;|")([^"&]+)/g)) push(m[1]);
  for (const m of src.matchAll(/(?:&quot;|")image_url(?:&quot;|")\s*:\s*(?:&quot;|")([^"&]+)/g)) push(m[1]);
  // og:image / twitter:image (page-scrape fallback).
  for (const m of src.matchAll(/<meta[^>]+(?:property|name)=["'](?:og:image|twitter:image)["'][^>]+content=["']([^"']+)["']/gi)) push(m[1]);
  // Last resort: absolute image URLs in the markup.
  for (const m of src.matchAll(/https?:\/\/[^\s"'<>\\]+\.(?:jpe?g|png|webp)[^\s"'<>\\]*/gi)) push(m[0]);
  return [...new Set(found)];
}

const HOST_BONUS = [
  [/pinterest|pinimg/i, 3],
  [/behance/i, 3],
  [/dribbble/i, 3],
  [/unsplash|pexels|pixabay/i, 1],
];

/** Dedupe, drop junk and prefer the inspiration hosts the user named.
 *
 *  The seed adds a sub-1.0 jitter so two runs over the SAME result set do
 *  not pick the same three images — the host bonus still dominates, so a
 *  Pinterest/Behance pin always outranks a random blog image. */
export function rankInspirationUrls(urls, { max = 3, seed = 0 } = {}) {
  const seen = new Set();
  const scored = [];
  for (const raw of urls || []) {
    const url = String(raw || "");
    if (!looksLikeUsableImage(url)) continue;
    const key = url.replace(/[?#].*$/, "");
    if (seen.has(key)) continue;
    seen.add(key);
    let score = 0;
    for (const [re, bonus] of HOST_BONUS) if (re.test(url)) score += bonus;
    if (IMAGE_EXT.test(url)) score += 1;
    // Seed jitter. It must be able to REORDER results inside a host tier
    // (otherwise the same request browses the same three images forever)
    // but never let a random blog image outrank an inspiration host:
    // max random = 1 + 2.5 = 3.5 < min inspiration host = 3 + 1 = 4.
    score += (hashSeed(url, seed) % 250) / 100;
    scored.push({ url, score });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(0, max)).map((s) => s.url);
}

/** Bing image search (keyless HTML) — the primary reference source. */
export async function bingImageSearch(query, { fetchImpl, limit = 10 } = {}) {
  const q = String(query || "").trim();
  if (!q || !fetchImpl) return [];
  const url = `https://www.bing.com/images/search?q=${encodeURIComponent(q)}&form=HDRSC2&first=1`;
  const res = await fetchImpl(url, {
    headers: { "User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/126 Safari/537.36" },
    signal: AbortSignal.timeout(12_000),
  });
  if (!res?.ok) return [];
  return extractImageUrls(await res.text(), url).slice(0, limit);
}

/** Openverse (keyless JSON, CC-licensed) — reliable fallback when Bing is
 *  blocked or returns nothing usable. */
export async function openverseSearch(query, { fetchImpl, limit = 10 } = {}) {
  const q = String(query || "").trim();
  if (!q || !fetchImpl) return [];
  const url = `https://api.openverse.org/v1/images/?q=${encodeURIComponent(q)}&page_size=${Math.min(20, limit)}`;
  const res = await fetchImpl(url, { signal: AbortSignal.timeout(12_000) });
  if (!res?.ok) return [];
  const data = await res.json().catch(() => ({}));
  return (data?.results || [])
    .map((r) => r.url || r.thumbnail)
    .filter(Boolean)
    .slice(0, limit);
}

/** Search every planned query, merge, rank, cap. Never throws: inspiration is
 *  an enhancement, not a prerequisite. */
export async function searchInspiration(goal, {
  sources, maxRefs = 3, maxQueries = 3, fetchImpl, seed = 0, fetchLimit = 12,
} = {}) {
  const queries = planInspirationQueries(goal, { sources, maxQueries, seed });
  const gathered = [];
  const errors = [];
  for (const q of queries) {
    try {
      gathered.push(...(await bingImageSearch(q, { fetchImpl, limit: fetchLimit })));
    } catch (e) { errors.push(`bing: ${e.message}`); }
    if (rankInspirationUrls(gathered, { max: maxRefs, seed }).length >= maxRefs) break;
  }
  if (rankInspirationUrls(gathered, { max: maxRefs, seed }).length < maxRefs) {
    for (const q of queries.slice(0, 1)) {
      try {
        gathered.push(...(await openverseSearch(q, { fetchImpl, limit: fetchLimit })));
      } catch (e) { errors.push(`openverse: ${e.message}`); }
    }
  }
  return { refs: rankInspirationUrls(gathered, { max: maxRefs, seed }), queries, errors };
}

/** Merge two research passes (the widened search) without losing refs. */
export function mergeInspiration(a, b, { maxRefs = 3, seed = 0 } = {}) {
  const refs = rankInspirationUrls([...(a?.refs || []), ...(b?.refs || [])], { max: maxRefs, seed });
  const queries = [...new Set([...(a?.queries || []), ...(b?.queries || [])])];
  const errors = [...(a?.errors || []), ...(b?.errors || [])];
  return { refs, queries, errors };
}

// ---------------------------------------------------------------------------
// 2. BRIEF
// ---------------------------------------------------------------------------

export function normalizeHex(value, fallback = "") {
  let v = String(value || "").trim();
  if (!v) return fallback;
  if (!v.startsWith("#")) v = `#${v}`;
  if (/^#[0-9a-f]{3}$/i.test(v)) v = `#${v[1]}${v[1]}${v[2]}${v[2]}${v[3]}${v[3]}`;
  return /^#[0-9a-f]{6}$/i.test(v) ? v.toUpperCase() : fallback;
}

function num(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function s(value, max = 200) {
  return String(value ?? "").replace(/\s+/g, " ").trim().slice(0, max);
}

/** A brief that is ALWAYS complete, whatever the model returned. */
export function defaultBrief(input = {}) {
  return {
    canvas: { aspect_ratio: s(input.aspect_ratio, 12) || DEFAULT_ASPECT, usage: s(input.usage, 40) || "social graphic" },
    layout: { grid: "rule-of-thirds", margins_pct: 8, alignment: "left", focal: "upper-left third", reading_order: "headline → subhead → body → cta" },
    palette: DEFAULT_PALETTE.map((c) => ({ ...c })),
    typography: [
      { role: "headline", font: s(input.font, 40) || "geometric sans", size_px: 96, weight: 800, tracking: -1, case: "UPPER" },
      { role: "subhead", font: s(input.font, 40) || "humanist sans", size_px: 30, weight: 500, tracking: 0, case: "SENTENCE" },
      { role: "cta", font: s(input.font, 40) || "semi-bold sans", size_px: 22, weight: 700, tracking: 1, case: "UPPER" },
    ],
    elements: [
      { type: "shape", description: "bold color block anchoring the headline", placement: "left third, full bleed", size_pct: 40 },
      { type: "icon", description: "simple line icon matching the subject", placement: "top-right corner", size_pct: 10 },
    ],
    imagery: { description: "", treatment: "clean cut-out, subtle shadow", placement: "right half" },
    background: { description: "flat brand color with a soft paper texture", color_hex: "#F4EFE4" },
    copy: { headline: s(input.headline, 120), subhead: "", body: s(input.body, 300), cta: s(input.cta, 80) },
    mood: s(input.mood, 80) || "confident, premium, uncluttered",
    checks: ["headline legible and spelled exactly", "palette matches the brief", "no gibberish text"],
  };
}

/** Merge a model-produced brief into the canonical shape. User copy always
 *  wins over the model's guess; numeric fields are clamped to sane ranges. */
export function normalizeBrief(raw, input = {}) {
  const base = defaultBrief(input);
  const r = raw && typeof raw === "object" ? raw : {};
  const out = { ...base };

  if (r.canvas) out.canvas = {
    aspect_ratio: s(r.canvas.aspect_ratio, 12) || base.canvas.aspect_ratio,
    usage: s(r.canvas.usage, 40) || base.canvas.usage,
  };
  if (r.layout) out.layout = {
    grid: s(r.layout.grid, 40) || base.layout.grid,
    margins_pct: clamp(num(r.layout.margins_pct, base.layout.margins_pct), 2, 20),
    alignment: /^(left|center|right)$/i.test(s(r.layout.alignment)) ? s(r.layout.alignment).toLowerCase() : base.layout.alignment,
    focal: s(r.layout.focal, 60) || base.layout.focal,
    reading_order: s(r.layout.reading_order, 80) || base.layout.reading_order,
  };

  // Palette: 3–5 entries, valid hexes, shares normalised toward 60-30-10 only
  // when the model gave nothing usable. `colors` is accepted as an alias.
  const palette = [];
  const rawPalette = Array.isArray(r.palette) ? r.palette
    : (Array.isArray(r.colors) ? r.colors
      : (Array.isArray(r.palette?.colors) ? r.palette.colors : null));
  for (const p of (rawPalette || []).slice(0, 5)) {
    const hex = normalizeHex(p?.hex || p, "");
    if (!hex || palette.some((c) => c.hex === hex)) continue;
    palette.push({ hex, role: s(p?.role, 24) || (palette.length ? "secondary" : "dominant"), share_pct: clamp(num(p?.share_pct, 0), 0, 100) });
  }
  if (palette.length >= 2) {
    const total = palette.reduce((n, c) => n + c.share_pct, 0);
    out.palette = total > 0
      ? palette.map((c) => ({ ...c, share_pct: Math.round((c.share_pct / total) * 100) }))
      : palette.map((c, i) => ({ ...c, share_pct: [60, 30, 10, 5, 5][i] ?? 5 }));
  } else if (palette.length === 1) {
    out.palette = [palette[0], ...DEFAULT_PALETTE.slice(1).map((c) => ({ ...c }))];
  }

  // Typography: keep known roles, force a size range, always cover headline.
  const typo = [];
  for (const t of Array.isArray(r.typography) ? r.typography.slice(0, 5) : []) {
    const role = s(t?.role, 12).toLowerCase();
    if (!TYPE_ROLES.includes(role) || typo.some((x) => x.role === role)) continue;
    const [lo, hi] = TYPE_SIZE[role];
    typo.push({
      role,
      font: s(t?.font, 40) || base.typography[0].font,
      size_px: Math.round(clamp(num(t.size_px, base.typography.find((b) => b.role === role)?.size_px), lo, hi)),
      weight: Math.round(clamp(num(t.weight, 600), 100, 900) / 100) * 100,
      tracking: clamp(num(t.tracking, 0), -4, 8),
      case: /^(upper|lower|sentence|title)$/i.test(s(t?.case)) ? s(t?.case).toUpperCase() : "SENTENCE",
    });
  }
  if (!typo.some((t) => t.role === "headline")) typo.unshift(base.typography[0]);
  out.typography = typo.sort((a, b) => TYPE_ROLES.indexOf(a.role) - TYPE_ROLES.indexOf(b.role));

  // Keep the brief COMPACT. A reasoning model spends its budget on thinking
  // first, and a long brief JSON gets truncated before the answer lands.
  const elements = [];
  for (const e of Array.isArray(r.elements) ? r.elements.slice(0, 6) : []) {
    const description = s(e?.description ?? e, 120);
    if (!description) continue;
    elements.push({
      type: s(e?.type, 16) || "element",
      description,
      placement: s(e?.placement, 60) || "balanced in the grid",
      size_pct: Math.round(clamp(num(e?.size_pct, 15), 1, 100)),
    });
  }
  out.elements = elements.length ? elements : base.elements.map((e) => ({ ...e }));

  if (r.imagery) out.imagery = {
    description: s(r.imagery.description, 200) || base.imagery.description,
    treatment: s(r.imagery.treatment, 80) || base.imagery.treatment,
    placement: s(r.imagery.placement, 60) || base.imagery.placement,
  };
  if (r.background) out.background = {
    description: s(r.background.description, 160) || base.background.description,
    color_hex: normalizeHex(r.background.color_hex, out.palette[0]?.hex || base.background.color_hex),
  };

  // The copy is typeset VERBATIM, so it is cleaned first: a placeholder
  // ("Let's celebrate [Name] in style!") or a quoted string would otherwise
  // be set on the artwork exactly as the model wrote it.
  const clean = (v) => {
    const t = s(v, 200);
    if (!t) return "";
    if (/\[[^\]]{1,60}\]|\{[^\}]{1,60}\}|<[^>]{1,60}>|\b(?:lorem ipsum|xxx+)\b/i.test(t)) return "";
    return t.replace(/^["'“”‘’`\s]+/, "").replace(/["'“”‘’`\s]+$/, "").trim();
  };
  out.copy = {
    headline: clean(input.headline) || clean(r.copy?.headline),
    subhead: clean(r.copy?.subhead),
    body: clean(input.body) || clean(r.copy?.body),
    cta: clean(input.cta) || clean(r.copy?.cta),
  };
  out.mood = s(r.mood, 80) || base.mood;
  out.checks = (Array.isArray(r.checks) ? r.checks.map((c) => s(c, 90)).filter(Boolean) : base.checks).slice(0, 5);
  out.generation_prompt = s(r.generation_prompt, 900);
  out.negative_prompt = s(r.negative_prompt, 300) || "gibberish text, watermark, lorem ipsum, cluttered collage, blurry type, extra fingers, low contrast";
  return out;
}

/**
 * A brief built WITHOUT the model, straight from the reference DNA.
 *
 * This is the fallback when the brief writer times out. Falling back to the
 * DEFAULT brief would throw away everything the agent learned from the
 * references; this keeps the references' grammar (grid, alignment, palette,
 * whitespace) and only loses the model's judgement.
 */
export function briefFromReferences(goal, input = {}, references = []) {
  const base = defaultBrief(input);
  const out = JSON.parse(JSON.stringify(base));
  const ref = references[0] || {};
  const st = ref.structure || {};
  if (st.grid) out.layout.grid = st.grid;
  if (st.alignment) out.layout.alignment = st.alignment;
  if (st.focal) out.layout.focal = st.focal;
  if (Number.isFinite(Number(st.whitespace_pct))) {
    out.layout.margins_pct = clamp(Math.round(Number(st.whitespace_pct) / 2.5), 4, 18);
  }
  const pal = (ref.palette || []).filter((p) => p && p.hex);
  if (pal.length >= 2) {
    const total = pal.reduce((n, c) => n + (Number(c.share_pct) || 0), 0);
    out.palette = total > 0
      ? pal.map((c, i) => ({ hex: normalizeHex(c.hex), role: c.role || (i ? "secondary" : "dominant"), share_pct: Math.round(((Number(c.share_pct) || 0) / total) * 100) }))
      : pal.map((c, i) => ({ hex: normalizeHex(c.hex), role: c.role || (i ? "secondary" : "dominant"), share_pct: [60, 30, 10, 5, 5][i] ?? 5 }));
  }
  if (ref.typography?.display) {
    const font = String(ref.typography.display).trim();
    // Guard against the vision model answering "N/A" / "unknown".
    if (font && !/^(n\/?a|unknown|none|null)$/i.test(font)) {
      out.typography = out.typography.map((t) => (t.role === "headline" ? { ...t, font } : t));
    }
  }
  if (ref.mood) out.mood = ref.mood;
  out.generation_prompt = "";
  out.negative_prompt = "gibberish text, watermark, lorem ipsum, cluttered collage, blurry type, extra fingers, low contrast";
  return out;
}

/** The deterministic prompt sent to the image model. This is the artifact the
 *  user asked for in step 2: arrangement, alignment, colours, fonts, sizes,
 *  icons/images, background and the exact copy — every field spelled out.
 *
 *  `artDirection` is the run's seed-derived visual route. The image models
 *  are near-deterministic for an identical prompt, so the SAME request must
 *  never produce the SAME prompt — that is what makes repeat runs feel
 *  creative instead of canned. */
export function briefToPrompt(brief, { artDirection = null, typesetByUs = false } = {}) {
  const b = brief || defaultBrief();
  const c = b.copy || {};
  const L = [];
  const pal = (b.palette || []).map((p) => `${p.hex} (${p.role || "color"}, ${p.share_pct}%)`).join(", ");
  const typo = (b.typography || []).map((t) => `${t.role}: ${t.font}, ${t.size_px}px, weight ${t.weight}, tracking ${t.tracking}, ${t.case}`).join("; ");
  const els = (b.elements || []).map((e) => `${e.type} — ${e.description} (${e.placement}, ~${e.size_pct}% of the canvas)`).join("; ");

  if (typesetByUs) {
    // THE TYPE IS SET BY US, AFTER the render. The model is asked for a
    // BACKGROUND PLATE, not a poster: a poster request makes it write its
    // own (mangled) lettering, and then the copy appears twice.
    L.push(`ARTWORK ONLY — the background plate for a poster. The typography is added afterwards, so this image must contain NO text at all.`);
    L.push(`Graphic design — ${b.canvas?.usage || "social graphic"}, ${b.canvas?.aspect_ratio || DEFAULT_ASPECT} aspect ratio.`);
    L.push(`COMPOSITION: ${b.layout?.grid} grid, ${b.layout?.margins_pct}% safe margins, ${b.layout?.alignment} alignment, focal point ${b.layout?.focal}. Reserve a calm, EMPTY area for the headline — ${b.layout?.alignment}-aligned, across the upper third. One focal idea only, generous breathing room.`);
    L.push(`COLORS (60-30-10 balance, exact hexes, no drifting hues): ${pal}. Background: ${b.background?.description || "flat brand color"} ${b.background?.color_hex || ""}.`);
    if (els) L.push(`ARTWORK: ${els}.`);
    if (b.imagery?.description) L.push(`IMAGERY: ${b.imagery.description} — ${b.imagery.treatment}, ${b.imagery.placement}.`);
    L.push(`TYPE PLAN (set later, do NOT draw it): ${typo}.`);
    L.push(`MOOD: ${b.mood}.`);
    if (artDirection) L.push(`ART DIRECTION: ${artDirection}.`);
    L.push("DO NOT RENDER: any letters, words, numbers, logos, slogans, captions, signage, watermarks, badges with wording or gibberish text. The only typography in this poster is added afterwards.");
    if (b.negative_prompt) L.push(`AVOID: ${b.negative_prompt}, any text, any letters, any numbers, any typography, any caption, any watermark.`);
    return L.join(" ");
  }

  L.push(`Graphic design — ${b.canvas?.usage || "social graphic"}, ${b.canvas?.aspect_ratio || DEFAULT_ASPECT} aspect ratio.`);
  L.push(`LAYOUT: ${b.layout?.grid} grid, ${b.layout?.margins_pct}% safe margins, ${b.layout?.alignment} alignment, focal point ${b.layout?.focal}; reading order ${b.layout?.reading_order}. Keep generous breathing room, one focal idea only.`);
  L.push(`COLORS (60-30-10 balance, exact hexes, no drifting hues): ${pal}. Background: ${b.background?.description || "flat brand color"} ${b.background?.color_hex || ""}.`);
  L.push(`TYPOGRAPHY: ${typo}. Hierarchy must be obvious at thumbnail size; text left-aligned to the grid unless stated.`);
  if (c.headline) L.push(`Headline (render EXACTLY as quoted, dominant type): "${c.headline}".`);
  if (c.subhead) L.push(`Subhead (render exactly): "${c.subhead}".`);
  if (c.body) L.push(`Body copy (render exactly, smaller than the headline): "${c.body}".`);
  if (c.cta) L.push(`CTA as a distinct button or badge (render exactly): "${c.cta}".`);
  if (els) L.push(`ELEMENTS: ${els}.`);
  if (b.imagery?.description) L.push(`IMAGERY: ${b.imagery.description} — ${b.imagery.treatment}, ${b.imagery.placement}.`);
  L.push(`MOOD: ${b.mood}.`);
  if (artDirection) L.push(`ART DIRECTION: ${artDirection}.`);
  L.push("Crisp legible type, correct spelling, print-ready finish, no watermark, no lorem ipsum, no unreadable gibberish.");
  if (b.negative_prompt) L.push(`AVOID: ${b.negative_prompt}.`);
  return L.join(" ");
}

// ---------------------------------------------------------------------------
// 3. RENDER + VERIFY
// ---------------------------------------------------------------------------

/** Turn the verifier's JSON into a decision. A missing/failed check is NOT a
 *  pass: the render ships, but the caller is told it is unverified. */
export function judgeVerification(raw, { threshold = 0.72 } = {}) {
  const r = raw && typeof raw === "object" ? raw : {};
  const score = clamp(num(r.match_score ?? r.score, 0), 0, 1);
  const issues = (Array.isArray(r.issues) ? r.issues : []).map((i) => s(i, 140)).filter(Boolean).slice(0, 6);
  const pass = typeof r.pass === "boolean" ? r.pass : score >= threshold;
  return { score: Math.round(score * 100) / 100, pass, issues, threshold, checked: true };
}

/** Best-effort image bytes for the verifier: public URLs are fetched (so the
 *  provider never depends on hotlink rules), everything else is passed through. */
/** Vision providers need a real image URL or a data: URI. A bare base64 blob
 *  (what the image API returns as image_b64) is useless to them — and asking a
 *  model to review one produced "no render was provided", i.e. a silent
 *  verification failure. This normalises all three shapes. */
export function asImageUrl(src) {
  const v = String(src || "").trim();
  if (!v) return null;
  if (v.startsWith("data:")) return v;
  if (/^https?:\/\//i.test(v)) return v;
  if (v.length > 200 && /^[A-Za-z0-9+/=\s]+$/.test(v)) return `data:image/png;base64,${v.replace(/\s+/g, "")}`;
  return null;
}

async function shrinkForVision(buf) {
  // A full-size render is a poor API payload (600KB+ base64): slow, sometimes
  // rejected, and the reviewer only needs layout/palette/text legibility.
  // Best-effort — without sharp, or on any error, the original bytes are used.
  try {
    const sharp = (await import("sharp")).default;
    const out = await sharp(buf).resize({ width: 1024, height: 1024, fit: "inside", withoutEnlargement: true }).jpeg({ quality: 82 }).toBuffer();
    if (out?.length && out.length < buf.length) return { buf: out, mime: "image/jpeg" };
  } catch { /* keep original */ }
  return { buf, mime: null };
}

/** Fetch remote image bytes once, with the hotlink-friendly UA the CDNs want.
 *  Returns null when the host refuses us — which is exactly the case the
 *  reference transport (creative/refs.js) exists to solve. */
async function fetchImageBytes(src, fetchImpl, maxBytes = 8 * 1024 * 1024) {
  if (!fetchImpl || !/^https?:\/\//i.test(String(src || ""))) return null;
  try {
    const res = await fetchImpl(src, {
      headers: { "User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/126 Safari/537.36" },
      signal: AbortSignal.timeout(12_000),
    });
    if (!res?.ok) return null;
    const buf = Buffer.from(await res.arrayBuffer());
    if (!buf.length || buf.length > maxBytes) return null;
    const mime = /\.png(?:[?#]|$)/i.test(src) ? "image/png" : /\.webp(?:[?#]|$)/i.test(src) ? "image/webp" : "image/jpeg";
    return { buf, mime };
  } catch {
    return null;
  }
}

async function imageForVision(src, fetchImpl, maxBytes = 6 * 1024 * 1024) {
  const v = asImageUrl(src);
  if (!v) return null;
  let buf = null;
  let mime = null;
  if (v.startsWith("data:")) {
    const comma = v.indexOf(",");
    if (comma < 0) return null;
    mime = v.slice(5, comma).split(";")[0] || "image/png";
    try { buf = Buffer.from(v.slice(comma + 1), "base64"); } catch { return v; }
    if (buf.length <= 300 * 1024) return v;
  } else if (fetchImpl && /^https?:\/\//i.test(v)) {
    try {
      const res = await fetchImpl(v, { signal: AbortSignal.timeout(10_000) });
      if (!res?.ok) return v;
      buf = Buffer.from(await res.arrayBuffer());
      if (!buf.length || buf.length > maxBytes) return v;
      mime = /\.png(?:[?#]|$)/i.test(v) ? "image/png" : /\.webp(?:[?#]|$)/i.test(v) ? "image/webp" : "image/jpeg";
    } catch {
      return v;
    }
  } else {
    return v;
  }
  const small = await shrinkForVision(buf);
  return `data:${small.mime || mime || "image/png"};base64,${small.buf.toString("base64")}`;
}

function parseJsonLoose(text) {
  const t = String(text || "");
  const fenced = t.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1] : t;
  let parsed = null;
  try { parsed = JSON.parse(candidate); } catch { /* fall through */ }
  if (parsed == null) {
    try { parsed = JSON.parse(candidate.match(/\{[\s\S]*\}/)?.[0] || "{}"); } catch { parsed = null; }
  }
  if (!parsed || typeof parsed !== "object") return parsed || {};
  // Some models wrap the answer: {"brief": {...}} or {"output": {...}}.
  // Unwrap a single wrapper key so the brief is still found.
  const keys = Object.keys(parsed);
  if (keys.length === 1 && parsed[keys[0]] && typeof parsed[keys[0]] === "object" && !Array.isArray(parsed[keys[0]])) {
    const inner = parsed[keys[0]];
    if (inner.palette || inner.layout || inner.copy || inner.typography) return inner;
  }
  return parsed;
}

/**
 * Read one reference with the vision model.
 *
 * The reference is asked for its STRUCTURE (how it is built) and, in a
 * separate bucket, its CONTENT (what it is about). Only the structure is
 * allowed downstream — that is the mechanical guarantee that the agent
 * studies the reference instead of copying it.
 */
export async function describeReference(ref, goal, { chat, fetchImpl, loaded = null } = {}) {
  if (!chat) return null;
  // `loaded` comes from creative/refs.js: browser-like fetch, public-proxy
  // fallback, <=1024px downscale and a content-addressed cache id. The
  // inline data URI is preferred so the provider never has to fetch anything.
  const image = loaded?.dataUri || loaded?.url || (await imageForVision(ref, fetchImpl));
  if (!image) return null;
  const prompt = [
    "You are a senior graphic designer reading a visual reference for a new piece.",
    `The brief is: ${s(goal, 300)}`,
    "Extract the transferable DESIGN DNA — describe HOW the piece is BUILT, never WHAT it depicts.",
    "Return ONLY JSON with this exact shape:",
    '{"structure":{"grid":"centered|split|rule-of-thirds|asymmetric|modular|diagonal|full-bleed|editorial-column",',
    '"alignment":"left|center|right","focal":"where the eye lands first, in plain words",',
    '"whitespace_pct":20,"palette_temp":"warm|cool|neutral|earthy|bold","palette_mood":"muted|vibrant|pastel|monochrome|high-contrast",',
    '"type_contrast":"high|medium|low","type_pairing":"serif-sans|sans-sans|display-sans|mono-sans|script-sans",',
    '"motif_density":"sparse|balanced|dense","texture":"flat|grain|gradient|paper|photo|collage",',
    '"hierarchy_ratio":"3:1","imagery_treatment":"how the imagery is treated, e.g. cut-out, duotone, masked"},',
    '"palette":[{"hex":"#RRGGBB","role":"dominant|secondary|accent","share_pct":60}],',
    '"typography":{"display":"...","body":"...","sizes":"...","case":"...","contrast":"high|medium|low"},',
    '"content":{"subject":"what the reference literally shows","motifs":["..."]},',
    '"mood":"..."}',
    "The CONTENT bucket is recorded for auditing only. It must never be reused.",
  ].join("\n");
  const out = await chat([
    { role: "user", content: [
      { type: "text", text: prompt },
      { type: "image_url", image_url: { url: image } },
    ] },
  ], { max_tokens: 3000, temperature: 0.2 });
  const raw = parseJsonLoose(out);
  const split = splitDNA(raw);
  return { ref, dna: raw, structure: split.structure, content: split.content, palette: split.palette, typography: split.typography, mood: split.mood };
}

/** The decisive structural axes of a reference, as one compact line. The
 *  brief writer needs the GRAMMAR of the reference, not its whole DNA — a
 *  compact block keeps the synthesis call inside its time budget. */
export function referenceDNAline(ref, index) {
  // Accepts either a reference object ({ref, dna}) or a bare DNA blob.
  const dna = ref?.structure
    ? { structure: ref.structure, palette: ref.palette, typography: ref.typography, mood: ref.mood }
    : (ref?.dna || ref);
  const so = structuralOnly(dna);
  const s = so.structure || {};
  const p = (so.palette || []).slice(0, 3).map((c) => c.hex).join("/");
  const t = so.typography || {};
  return `REF${index + 1} grid=${s.grid} align=${s.alignment} temp=${s.palette_temp} mood=${s.palette_mood || "?"} contrast=${s.type_contrast} pairing=${s.type_pairing} density=${s.motif_density} texture=${s.texture} whitespace=${s.whitespace_pct}% palette=${p || "?"} type=${t.display || "?"}/${t.body || "?"}`;
}

/** Ask the fast model for a structured brief, briefed by the reference DNA. */
export async function synthesizeBrief(goal, input = {}, references = [], { chat } = {}) {
  if (!chat) return {};
  // STRUCTURE ONLY. The reference's subject/motifs never reach the brief
  // writer, so the result cannot be a re-skin of the reference.
  const refBlock = references
    .filter((r) => r?.dna || r?.structure)
    .slice(0, 3)
    .map((r, i) => referenceDNAline(r, i))
    .join("\n");
  const system = [
    "You are a graphic-design art director writing ONE production brief. Be concrete and measurable.",
    "You may borrow the references' STRUCTURE (grid, whitespace, type contrast, palette temperature).",
    "You must NOT reproduce a reference's subject, motifs, colours or imagery — the content comes from the user only.",
    "Never invent brand assets. Keep the user's exact copy.",
    "Return ONLY JSON:",
    '{"canvas":{"aspect_ratio":"1:1","usage":"instagram post"},',
    '"layout":{"grid":"rule-of-thirds","margins_pct":8,"alignment":"left","focal":"...","reading_order":"..."},',
    '"palette":[{"hex":"#RRGGBB","role":"dominant","share_pct":60}],',
    '"typography":[{"role":"headline","font":"...","size_px":96,"weight":800,"tracking":-1,"case":"UPPER"}],',
    '"elements":[{"type":"icon|shape|badge|photo","description":"...","placement":"...","size_pct":10}],',
    '"imagery":{"description":"...","treatment":"...","placement":"..."},',
    '"background":{"description":"...","color_hex":"#RRGGBB"},',
    '"copy":{"headline":"...","subhead":"...","body":"...","cta":"..."},',
    '"mood":"...","checks":["..."],"negative_prompt":"..."}',
  ].join("\n");
  const user = [
    `GOAL: ${s(goal, 400)}`,
    input.headline ? `MUST-USE HEADLINE: "${s(input.headline, 120)}"` : "",
    input.body ? `MUST-USE BODY: "${s(input.body, 300)}"` : "",
    input.cta ? `MUST-USE CTA: "${s(input.cta, 80)}"` : "",
    input.palette ? `USER COLORS (honour these): ${s(input.palette, 120)}` : "",
    input.font ? `USER FONT VOICE: ${s(input.font, 60)}` : "",
    input.brand?.colors?.length ? `BRAND COLORS: ${input.brand.colors.join(", ")}` : "",
    input.brand?.font ? `BRAND FONT: ${s(input.brand.font, 60)}` : "",
    input.style ? `STYLE DIRECTION: ${s(input.style, 160)}` : "",
    refBlock ? `\n${refBlock}\n` : "\n(no references available — design from the goal alone)\n",
  ].filter(Boolean).join("\n");
  const messages = [
    { role: "system", content: system },
    { role: "user", content: user },
  ];
  const first = parseJsonLoose(await chat(messages, { max_tokens: 3000, temperature: 0.4 }));
  if (first && (first.palette || first.layout || first.copy || first.typography)) return first;
  // Some models answer with prose instead of JSON. One corrective retry is much
  // cheaper than silently shipping the default brief (Week 6: that happened).
  let retryText = "";
  const retry = parseJsonLoose(retryText = await chat([
    ...messages,
    { role: "assistant", content: "(reply was not JSON)" },
    { role: "user", content: "Return ONLY the JSON object described above — no prose, no markdown, no explanation." },
  ], { max_tokens: 3000, temperature: 0 }));
  if (retry && (retry.palette || retry.layout || retry.copy || retry.typography)) return retry;
  // Neither reply was usable. THROW, so the caller falls back to the reference
  // DNA and says so. Returning {} here silently shipped a default brief that
  // ignored everything the agent learned from the references.
  throw new Error(`the brief writer did not return usable JSON (reply started: ${String(retryText).replace(/\s+/g, " ").slice(0, 160) || "empty"})`);
}

/** The copy string for a role AS TYPESET (the brief's case rule applied). */
export function typesetString(brief, role) {
  const text = brief?.copy?.[role];
  if (!text) return "";
  const spec = (brief?.typography || []).find((t) => t.role === role);
  const mode = String(spec?.case || "SENTENCE").toUpperCase();
  if (mode === "UPPER") return String(text).toUpperCase();
  if (mode === "LOWER") return String(text).toLowerCase();
  if (mode === "TITLE") return String(text).replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase());
  return String(text);
}

/** Does the render match the brief? Vision model, structured answer. */
export async function verifyRender(render, brief, { chat, fetchImpl, threshold = 0.72 } = {}) {
  if (!chat) return { score: 0, pass: false, issues: ["verification unavailable"], threshold, checked: false };
  const image = await imageForVision(render, fetchImpl);
  if (!image) return { score: 0, pass: false, issues: ["no render to verify"], threshold, checked: false };
  const facts = {
    palette: (brief?.palette || []).map((p) => p.hex),
    // The strings AS TYPESET (the brief's case rule applied). The reviewer
    // must judge what was actually set, not the raw copy, or it flags an
    // intentional all-caps CTA as a spelling error.
    copy_rule: "the copy below is the brief's exact copy with the case rule applied",
    headline: typesetString(brief, "headline"),
    subhead: typesetString(brief, "subhead"),
    body: typesetString(brief, "body"),
    cta: typesetString(brief, "cta"),
    alignment: brief?.layout?.alignment,
    // Judge the TYPE BLOCK, not the artwork's focal point. The renderer was
    // asked for a background plate, so the artwork's own composition is not
    // the brief's alignment — the type we set on top is.
    alignment_note: "the copy block is set flush " + (brief?.layout?.alignment || "left") + " within the safe margins; judge the type, not the artwork's focal point",
    // The brief's px are relative to a 1080-wide canvas; the render may be a
    // different size. Without this the reviewer calls a correctly scaled
    // headline "too small".
    type_base_canvas_px: 1080,
    type_size_note: "the sizes below are px on a 1080-wide canvas; the render canvas may differ, so judge them PROPORTIONALLY (96px headline = ~8.9% of the canvas width)",
    // The type is set with the fonts the render host actually has
    // (DejaVu Sans / DejaVu Serif / DejaVu Sans Mono). Judge legibility,
    // hierarchy and weight — not whether it is literally the named family.
    type_font_note: "the copy is set in the host's available families (DejaVu Sans / DejaVu Serif / DejaVu Sans Mono); judge legibility, weight and hierarchy, not the exact typeface named in the brief",
    typography: (brief?.typography || []).map((t) => `${t.role}:${t.font} ${t.size_px}px`),
    checks: brief?.checks || [],
  };
  const ask = (nudge = "") => chat([
    { role: "user", content: [
      { type: "text", text: [
        "You are a strict design reviewer. Compare the RENDER to the BRIEF and score the match.",
        `BRIEF: ${JSON.stringify(facts)}`,
        "Check: exact copy spelling, palette adherence, hierarchy/alignment, required elements present, no gibberish text.",
        'Return ONLY JSON: {"match_score":0.0,"pass":true|false,"issues":["specific, actionable problem"],"matches":["what is right"]}',
        nudge,
      ].filter(Boolean).join("\n") },
      { type: "image_url", image_url: { url: image } },
    ] },
  ], { max_tokens: 4000, temperature: 0 });

  let text = await ask();
  let parsed = parseJsonLoose(text);
  const usable = (p) => p && (p.match_score != null || p.score != null || typeof p.pass === "boolean");
  if (!usable(parsed)) {
    // Prose instead of JSON is common enough to warrant one corrective retry
    // (Week 6: the first live verification came back unparsable).
    text = await ask("Your previous reply could not be parsed. Answer with the JSON object ONLY.");
    parsed = parseJsonLoose(text);
  }
  const judged = judgeVerification(parsed, { threshold });
  if (!usable(parsed)) {
    judged.issues = [`reviewer reply was not usable JSON (${s(text, 120) || "empty"})`];
  }
  return judged;
}

// ---------------------------------------------------------------------------
// ORCHESTRATION
// ---------------------------------------------------------------------------

async function withTimeout(promise, ms, label) {
  if (!(ms > 0)) throw new Error(`${label}: no budget left`);
  const p = Promise.resolve(promise);
  p.catch(() => {});
  let timer;
  try {
    return await Promise.race([
      p,
      new Promise((_, reject) => {
        timer = setTimeout(() => {
          const e = new Error(`${label} timed out`);
          e.timedOut = true;
          reject(e);
        }, ms);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * Run the three stages. Deps (all injected, all optional except generate):
 *   chat(messages, {max_tokens,temperature}) -> string content
 *   generate(prompt, {aspect_ratio, output_format, brief, model}) -> {image_url|image_b64, model}
 *   searchImages(goal, opts) -> {refs, queries, errors}   (or searchInspiration)
 *   fetchImpl(url, init) -> fetch-like
 *   proxyReference(url) -> public URL for the reference bytes (optional)
 *   applyType(buffer, brief) -> {buffer, applied} (optional type layer)
 *   now() -> ms
 */
export async function runDesignAgent(input = {}, deps = {}, opts = {}) {
  const now = deps.now || (() => Date.now());
  const t0 = now();
  const totalMs = opts.totalMs ?? 210_000;
  const maxAttempts = input.verify === false ? 1 : Math.min(3, Math.max(1, Number(input.maxAttempts) || 2));
  const maxRefs = Math.min(4, Math.max(0, Number(input.max_refs ?? 3)));
  const minRefs = Math.min(maxRefs, Math.max(0, Number(input.min_refs ?? 3)));
  const threshold = Number(input.threshold) || 0.72;
  const stages = [];
  const warnings = [];
  const left = () => Math.max(0, totalMs - (now() - t0));
  const mark = (stage) => stages.push({ stage, ms: now() - t0 });

  const goal = s(input.goal || input.prompt || input.headline, 400);
  if (!goal) throw new Error("goal required");

  // The SEED. Everything that could repeat — the style token in the query,
  // the reference order, the art direction — derives from it, so the same
  // request never browses and draws the same thing twice.
  const seed = Number.isFinite(Number(input.seed))
    ? Math.abs(Number(input.seed)) >>> 0
    : hashSeed(input.seed ?? "", input.uid ?? "", goal, now());
  const styleToken = styleTokenFor(seed);
  const artDirection = artDirectionFor(seed);

  // ---- stage 1: research ------------------------------------------------
  let inspiration = { refs: [], queries: [], errors: [] };
  const candidateCount = Math.min(6, Math.max(maxRefs, minRefs) + 2);
  if (input.inspiration !== false && maxRefs > 0) {
    try {
      const searchFn = deps.searchImages || ((g, o) => searchInspiration(g, { ...o, fetchImpl: deps.fetchImpl }));
      const first = await withTimeout(
        searchFn(goal, { maxRefs: candidateCount, sources: input.sources, seed, fetchImpl: deps.fetchImpl }),
        Math.min(20_000, left() * 0.2), "inspiration search"
      );
      inspiration = first || inspiration;
      // The user asked for AT LEAST three references studied. If the named
      // hosts did not deliver, widen the query (no site: filter) once.
      if ((inspiration.refs || []).length < minRefs && left() > 12_000) {
        const wide = await withTimeout(
          searchFn(goal, { maxRefs: candidateCount, sources: [], seed: seed + 7919, fetchImpl: deps.fetchImpl }),
          Math.min(20_000, left() * 0.2), "widened inspiration search"
        );
        inspiration = mergeInspiration(inspiration, wide, { maxRefs: candidateCount, seed });
      }
    } catch (e) {
      console.warn("[design-agent] inspiration search failed:", e.message);
      warnings.push(`inspiration skipped: ${e.message}`);
    }
  }
  mark("research");

  // ---- stage 2: read references, then write the brief -------------------
  // Read in small batches, not all at once: five concurrent vision calls
  // congest the provider (they time out) and every timeout is an EXTRA paid
  // call when the chain falls through. Stop as soon as we have enough.
  let references = [];
  if (inspiration.refs?.length && (deps.chat || deps.describeReference)) {
    const describe = deps.describeReference || ((ref, g, extra) => describeReference(ref, g, { chat: deps.chat, fetchImpl: deps.fetchImpl, loaded: extra?.loaded || null }));
    const candidates = inspiration.refs.slice(0, candidateCount);
    const BATCH = 2;
    // Read one more than we need: the diversity pick needs a spare to choose
    // from, and a spare is also the fallback when one read fails.
    const want = Math.min(candidates.length, maxRefs + 1);
    for (let i = 0; i < candidates.length && references.length < want; i += BATCH) {
      const slice = candidates.slice(i, i + BATCH);
      const batch = await Promise.all(slice.map(async (ref) => {
        try {
          // The reference transport fetches the bytes ONCE, browser-like, and
          // falls back to a public origin-side proxy when the CDN refuses us.
          const loaded = deps.loadReference ? await deps.loadReference(ref) : null;
          // The reads are concurrent, so a generous cap costs no wall-clock.
          const out = await withTimeout(describe(ref, goal, { loaded }), Math.min(40_000, Math.max(15_000, left() * 0.35)), "reference read");
          return out ? { ...out, ref_image: loaded?.url || null, ref_id: loaded?.id || null } : null;
        } catch (e) {
          console.warn("[design-agent] reference read failed:", e.message);
          warnings.push(`reference unreadable (${s(ref, 80)}): ${e.message}`);
          return null;
        }
      }));
      references.push(...batch.filter(Boolean));
    }
  }
  references = references.map((r) => ({
      ...r,
      structure: r.structure || splitDNA(r.dna).structure,
      palette: r.palette || splitDNA(r.dna).palette,
      typography: r.typography || splitDNA(r.dna).typography,
      mood: r.mood || splitDNA(r.dna).mood,
    }));
  mark("references");

  // Pick the references that are STRUCTURALLY FAR APART, not the top three of
  // one search. Three pins from the same board would otherwise masquerade as
  // three references.
  const allCandidates = references;
  const diverse = selectDiverse(allCandidates, { max: maxRefs });
  const dropped = allCandidates.filter((r) => !diverse.includes(r));

  let brief;
  let raw = null;
  let briefFromFallback = false;
  try {
    // The brief writer only needs TEXT (the reference DNA is already
    // extracted), so it runs on the fast model. The vision model is reserved
    // for reading references and reviewing the render.
    const synth = deps.synthesizeBrief || ((g, i, r) => synthesizeBrief(g, i, r, { chat: deps.chatText || deps.chat }));
    raw = await withTimeout(synth(goal, input, diverse), Math.min(75_000, Math.max(30_000, left() * 0.6)), "brief synthesis");
    brief = normalizeBrief(raw, input);
  } catch (e) {
    console.warn(`[design-agent] brief synthesis failed:`, e.message);
    // The model timed out or failed. Do NOT throw the references away: build
    // the brief straight from their DNA instead of falling back to a default
    // that ignores everything the agent learned.
    warnings.push(`brief synthesis fell back to the reference DNA: ${e.message}`);
    briefFromFallback = true;
    try {
      brief = normalizeBrief(briefFromReferences(goal, input, diverse), input);
    } catch {
      brief = normalizeBrief({}, input);
    }
  }
  // A palette the model never produced is a silent downgrade — say so.
  if (!Array.isArray(raw?.palette) && !Array.isArray(raw?.colors) && raw) {
    warnings.push("the brief writer returned no palette — the default colour system was used");
  }

  // Push the brief AWAY from the references: if it mirrors one too closely it
  // is a re-skin, not a design. Brand/user-pinned colours are never touched.
  // (A brief that already came FROM the reference DNA is already a synthesis —
  // pushing it away again would leave a generic default, so it is exempt.)
  const refStructures = diverse.map((r) => r.structure).filter(Boolean);
  const userPinned = !!(input.palette || input.brand?.colors?.length);
  let divergence = { changed: [] };
  if (briefFromFallback) {
    divergence = { changed: ["brief built from the reference DNA (divergence guard skipped)"] };
  } else {
    try {
      divergence = enforceDivergence(brief, refStructures, {
        seed,
        maxShared: 3,
        allowPaletteShift: !userPinned,
      });
      brief = divergence.brief;
    } catch (e) {
      console.warn("[design-agent] divergence check failed:", e.message);
      warnings.push(`divergence check failed: ${e.message}`);
    }
  }
  mark("brief");

  // ---- stage 3: render + verify (one corrective retry) ------------------
  if (!deps.generate) throw new Error("generate dependency missing");
  let verification = null;
  let render = null;
  let attempt = 0;
  let correction = "";
  // Who sets the type?
  //   typesetByUs  — a general image model draws the ARTWORK only and we
  //                  typeset the real copy on top. This is the path that
  //                  kills the "AI poster" look: real glyphs, real
  //                  hierarchy, no mangled lettering, no duplicated copy.
  //   otherwise    — a typography specialist (Ideogram/Recraft) sets the
  //                  type itself with real fonts, so we must not draw it
  //                  again on top.
  const typesetByUs = deps.typesetByUs ? !!deps.typesetByUs(input.model) : true;
  const promptBase = briefToPrompt(brief, { artDirection, typesetByUs });
  while (attempt < maxAttempts && left() > 5_000) {
    attempt++;
    const prompt = promptBase + (correction ? ` CORRECTIONS FROM REVIEW: ${correction}.` : "");
    try {
      render = await withTimeout(
        deps.generate(prompt, {
          aspect_ratio: brief.canvas.aspect_ratio,
          output_format: input.output_format || "png",
          model: input.model,
          brief,
          seed: seed + attempt,
          negative_prompt: brief.negative_prompt,
        }),
        // The render gets whatever is LEFT: it is the one stage that must not
        // be cut short, and a cut render means no design at all.
        Math.min(90_000, Math.max(20_000, left() - 2_000)),
        "render"
      );
    } catch (e) {
      console.warn(`[design-agent] render attempt ${attempt} failed:`, e.message);
      warnings.push(`render attempt ${attempt} failed: ${e.message}`);
      render = null;
      break;
    }
    // The TYPE LAYER: real glyphs set from the brief on top of the artwork.
    // Only when we are the ones setting the type — a typography specialist
    // has already rendered it and drawing it again would double the copy.
    if (deps.applyType && render && input.type_layer !== false && typesetByUs) {
      try {
        const source = render.image_url
          ? await fetchToBuffer(render.image_url, deps.fetchImpl)
          : (render.image_b64 ? Buffer.from(String(render.image_b64), "base64") : null);
        if (source) {
          const layered = await deps.applyType(source, brief, { model: render.model });
          if (layered?.applied && layered.buffer) {
            render = { ...render, image_b64: layered.buffer.toString("base64"), image_url: null, media_type: "image/png", type_layer: true };
          } else if (layered?.reason) {
            warnings.push(`type layer skipped: ${layered.reason}`);
          }
        } else {
          console.warn("[design-agent] type layer skipped: the render bytes could not be fetched");
          warnings.push("type layer skipped: the render bytes could not be fetched");
        }
      } catch (e) {
        console.warn("[design-agent] type layer failed:", e.message);
        warnings.push(`type layer failed: ${e.message}`);
      }
    }
    // Review needs a reviewer: either the injected one or a chat model for the
    // built-in verifier. Without either we still ship the render, but we say so.
    if (input.verify === false || (!deps.verifyRender && !deps.chat)) {
      verification = { score: 0, pass: false, issues: [], threshold, checked: false };
      break;
    }
    try {
      const verify = deps.verifyRender || ((r, b) => verifyRender(r, b, { chat: deps.chat, fetchImpl: deps.fetchImpl, threshold }));
      verification = await withTimeout(verify(render.image_url || render.image_b64, brief), Math.min(45_000, Math.max(12_000, left() - 1_000)), "verification");
    } catch (e) {
      console.warn("[design-agent] verification failed:", e.message);
      warnings.push(`verification failed: ${e.message}`);
      verification = { score: 0, pass: false, issues: [], threshold, checked: false };
      break;
    }
    if (verification.pass) break;
    correction = verification.issues.slice(0, 3).join("; ");
    if (!correction) break;
  }
  mark("render");

  return {
    ok: !!render,
    goal,
    seed,
    style_token: styleToken,
    art_direction: artDirection,
    brief,
    prompt: promptBase,
    inspiration: {
      queries: inspiration.queries || [],
      refs: inspiration.refs || [],
      read: diverse.length,
      studied: diverse.map((r) => ({ ref: r.ref, image_url: r.ref_image || null, structure: r.structure })),
      dropped: dropped.map((r) => ({ ref: r.ref, image_url: r.ref_image || null, structure: r.structure })),
      errors: inspiration.errors || [],
    },
    divergence: { changed: divergence.changed || [], shared_axes: refStructures.map((s) => sharedAxes(briefAxes(brief), s)) },
    render: render ? {
      image_url: render.image_url || null,
      image_b64: render.image_b64 || null,
      model_used: render.model || null,
      // OpenRouter returns whatever the model produced (Recraft hands back
      // webp, Seedream png). The client MUST use this or it will wrap a
      // webp payload in a data:image/png URI and show nothing.
      media_type: render.media_type || (render.type_layer ? "image/png" : "image/png"),
      type_layer: !!render.type_layer,
    } : null,
    verification: verification || null,
    attempts: attempt,
    stages,
    warnings,
  };
}

async function fetchToBuffer(url, fetchImpl) {
  if (!fetchImpl || !/^https?:\/\//i.test(String(url || ""))) return null;
  try {
    const res = await fetchImpl(url, { signal: AbortSignal.timeout(20_000) });
    if (!res?.ok) return null;
    return Buffer.from(await res.arrayBuffer());
  } catch {
    return null;
  }
}

export { hashSeed, styleTokenFor, artDirectionFor, splitDNA, structuralOnly, selectDiverse, enforceDivergence, briefAxes, sharedAxes };
