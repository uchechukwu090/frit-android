// ============================================================================
// IMAGE PROVIDER KNOWLEDGE — OpenRouter only
// ----------------------------------------------------------------------------
// ONE gateway: `POST https://openrouter.ai/api/v1/images`. There are no
// per-provider API keys: no IDEOGRAM_API_KEY, no RECRAFT_API_KEY.
//
// What the catalogue actually serves (checked live, 61 image models):
//   * RECRAFT is there — 16 slugs: recraft/recraft-v3, recraft/recraft-v4,
//     recraft/recraft-v4-pro, recraft/recraft-v4-vector, recraft/recraft-v4-styles,
//     recraft/recraft-v4-utility, recraft/recraft-v4.1 and their variants.
//   * IDEOGRAM IS NOT on OpenRouter. Do not configure it as a model slug —
//     every call to it 404s and burns a slot in the fallback chain.
//
// Each model has its OWN accepted parameters, and the generic body does not
// work for all of them. Recraft, for example, REJECTS `n` and
// `output_format` and only accepts the aspect ratios 1:1, 4:3, 3:4, 16:9,
// 9:16 and auto. Sending the generic body to Recraft is a 400, so the request
// is built per model family here.
//
// On typography: OpenRouter does NOT forward Recraft's native `texts[]`
// payload (verified live — the text never appears in the render). The real,
// exact typography comes from our own type layer (creative/typeLayer.js),
// which sets the copy from the brief with real glyphs, real tracking and
// real hierarchy. Recraft is therefore treated as a model that sets the type
// ITSELF (it renders real fonts when the copy is quoted in the prompt), and
// the type layer is skipped for it — never both, or the copy appears twice.
// ============================================================================

/** Aspect ratios each model family accepts. */
const ASPECTS = {
  recraft: ["1:1", "4:3", "3:4", "16:9", "9:16", "auto"],
};

/** Parameters each model family rejects outright. */
const REJECTS = {
  recraft: ["n", "output_format"],
};

const familyOf = (model) => String(model || "").toLowerCase().split("/")[0];

/**
 * True when the model renders REAL type itself (Recraft's font library).
 * Those models must not be given a second typeset pass on top — the copy
 * would appear twice. Every other model gets our type layer.
 */
export function isTypeSpecialist(model) {
  return familyOf(model) === "recraft";
}

/** Snap an aspect ratio to one the model family actually accepts. */
export function snapAspect(model, aspect, fallback = "1:1") {
  const allowed = ASPECTS[familyOf(model)];
  const v = String(aspect || "").trim();
  if (!allowed) return v || fallback;
  if (!v) return allowed.includes("auto") ? "auto" : fallback;
  if (allowed.includes(v)) return v;
  const m = /^(\d+)\s*:\s*(\d+)$/.exec(v);
  if (!m) return allowed.includes("auto") ? "auto" : fallback;
  const r = Number(m[2]) / Number(m[1]);
  let best = allowed.includes("auto") ? "auto" : fallback;
  let bestErr = Infinity;
  for (const a of allowed) {
    if (a === "auto") continue;
    const [w, h] = a.split(":").map(Number);
    const err = Math.abs((h / w) - r);
    if (err < bestErr) { bestErr = err; best = a; }
  }
  return best;
}

/**
 * The OpenRouter /images body for a model, with only the parameters that
 * model accepts.
 */
export function imageRequestBody(model, prompt, opts = {}) {
  const family = familyOf(model);
  const rejects = new Set(REJECTS[family] || []);
  for (const extra of opts.drop || []) rejects.add(extra);
  const body = { model, prompt: String(prompt || "") };
  if (opts.aspect_ratio && !rejects.has("aspect_ratio")) {
    body.aspect_ratio = snapAspect(model, opts.aspect_ratio);
  }
  if (!rejects.has("n")) body.n = opts.n || 1;
  if (!rejects.has("output_format") && opts.output_format) body.output_format = opts.output_format;
  if (opts.input_references?.length) body.input_references = opts.input_references;
  return body;
}

/**
 * A model chain for the design agent. The chain must stay in ONE class:
 * the prompt tells the model whether WE set the type, so a chain that mixed
 * a type specialist with a general model would hand half the chain the wrong
 * prompt (and double the copy).
 */
export function classConsistentChain(models, { specialist }) {
  const list = (Array.isArray(models) ? models : []).filter(Boolean);
  if (!list.length) return [];
  return list.filter((m) => isTypeSpecialist(m) === specialist);
}
