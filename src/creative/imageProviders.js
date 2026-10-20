// ============================================================================
// IMAGE PROVIDERS — OpenRouter (default), Ideogram, Recraft
// ----------------------------------------------------------------------------
// One normalised return shape for every provider so nothing downstream has
// to care who drew the pixels:
//
//   { ok: true, model, image_url, image_b64, media_type }
//
// Provider selection is by MODEL SLUG PREFIX, which means the existing craft
// routing lights up with zero extra plumbing:
//
//   OR_TYPE_MODEL=ideogram/ideogram-v3    -> typography-led posters
//   OR_VECTOR_MODEL=recraft/recraft-v3    -> vector/logo work
//
// Neither Ideogram nor Recraft accepts reference images the way OpenRouter
// does — and that is a FEATURE here: the inspiration references must never
// reach the renderer.
// ============================================================================

import { briefToTextSpecs } from "./typeLayer.js";

export const PROVIDER_IDEOGRAM = "ideogram";
export const PROVIDER_RECRAFT = "recraft";
export const PROVIDER_OPENROUTER = "openrouter";

/** Which provider handles this model? Explicit `opts.provider` wins. */
export function providerFor(model, opts = {}) {
  if (opts.provider) return String(opts.provider).toLowerCase();
  const m = String(model || "").toLowerCase();
  if (m.startsWith("ideogram/") || m.includes(":ideogram")) return PROVIDER_IDEOGRAM;
  if (m.startsWith("recraft/") || m.includes(":recraft")) return PROVIDER_RECRAFT;
  return PROVIDER_OPENROUTER;
}

function keyFor(provider) {
  if (provider === PROVIDER_IDEOGRAM) return process.env.IDEOGRAM_API_KEY || "";
  if (provider === PROVIDER_RECRAFT) return process.env.RECRAFT_API_KEY || "";
  return "";
}

export function providerAvailable(provider) {
  if (provider === PROVIDER_OPENROUTER) return true;
  return !!keyFor(provider);
}

// ---------------------------------------------------------------------------
// ASPECT / SIZE
// ---------------------------------------------------------------------------

const IDEOGRAM_ASPECTS = ["1:1", "2:3", "3:2", "4:5", "5:4", "16:9", "21:9", "9:16", "3:4", "4:3"];

/** Ideogram only accepts a fixed aspect list — snap to the closest one by
 *  area so a "4:5" poster never silently becomes a square. */
export function nearestIdeogramAspect(aspect, fallback = "1:1") {
  const m = /^(\d+)\s*:\s*(\d+)$/.exec(String(aspect || "").trim());
  if (!m) return fallback;
  const r = Number(m[2]) / Number(m[1]);
  let best = fallback;
  let bestErr = Infinity;
  for (const a of IDEOGRAM_ASPECTS) {
    const [w, h] = a.split(":").map(Number);
    const err = Math.abs((h / w) - r);
    if (err < bestErr) { bestErr = err; best = a; }
  }
  return best;
}

export function sizeForAspect(aspect, base = 1024) {
  const m = /^(\d+)\s*:\s*(\d+)$/.exec(String(aspect || "1:1").trim());
  if (!m) return { width: base, height: base };
  const r = Number(m[2]) / Number(m[1]);
  const width = base;
  const height = Math.max(64, Math.round(base * r));
  return { width, height };
}

// ---------------------------------------------------------------------------
// IDEOGRAM
// ---------------------------------------------------------------------------

/**
 * Ideogram v3 — the typography specialist. Quoted copy in the prompt is
 * rendered far more faithfully than by general image models, which is why
 * the brief quotes every string.
 */
export async function ideogramGenerate(prompt, opts = {}) {
  const apiKey = keyFor(PROVIDER_IDEOGRAM);
  if (!apiKey) throw new Error("IDEOGRAM_API_KEY missing — add it to server/.env to enable Ideogram");
  const model = opts.model || "ideogram/ideogram-v3";
  const body = {
    prompt: String(prompt || "").slice(0, 6000),
    rendering_speed: opts.rendering_speed || "TURBO",
    // Our prompt is already a full art direction — magic_prompt would rewrite
    // it and undo the brief.
    magic_prompt: false,
    style_type: opts.style_type || "DESIGN",
    resolution: opts.resolution || "2K",
    num_images: 1,
    aspect_ratio: nearestIdeogramAspect(opts.aspect_ratio),
  };
  if (opts.negative_prompt) body.negative_prompt = String(opts.negative_prompt).slice(0, 500);
  if (opts.seed != null && opts.seed !== "") {
    body.seed = Math.abs(Number(opts.seed) || 0) % 2147483647;
  }
  const res = await fetch("https://api.ideogram.ai/v3/generate", {
    method: "POST",
    headers: { "Api-Key": apiKey, "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(180_000),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(`Ideogram HTTP ${res.status}: ${data?.error?.message || data?.detail || JSON.stringify(data).slice(0, 240)}`);
  }
  const item = data?.data?.[0] || {};
  const image_url = item.url || null;
  const image_b64 = item.b64_json || item.base64 || null;
  if (!image_url && !image_b64) {
    throw new Error(`Ideogram returned no image: ${JSON.stringify(data).slice(0, 240)}`);
  }
  return { ok: true, model, image_url, image_b64, media_type: "image/png", provider: PROVIDER_IDEOGRAM };
}

// ---------------------------------------------------------------------------
// RECRAFT
// ---------------------------------------------------------------------------

// Recraft's own font roster. The brief's font names are mapped onto these so
// the rendered typography uses REAL letterforms instead of a hallucinated
// approximation of the requested typeface.
const RECRAFT_FONTS = [
  "noto_sans", "noto_serif", "inter", "montserrat", "poppins", "roboto_mono",
  "lora", "playfair_display", "bricolage_grotesque", "titan_one", "shrikhand",
  "oswald", "anton", "acme", "righteous", "space_grotesk",
];

const RECRAFT_STYLES = [
  "realistic_image", "digital_illustration", "vector_illustration", "icon",
  "design", "logo", "stylized_illustration",
];

export function mapFontToRecraft(fontName) {
  const f = String(fontName || "").toLowerCase();
  if (/mono|code/.test(f)) return "roboto_mono";
  if (/playfair|garamond|didot|bodoni|serif display/.test(f)) return "playfair_display";
  if (/serif|lora|times|georgia|roman/.test(f)) return "noto_serif";
  if (/bricolage|grotesk/.test(f)) return "bricolage_grotesque";
  if (/titan|impact|black|heavy|display/.test(f)) return "titan_one";
  if (/shrikhand|script|hand/.test(f)) return "shrikhand";
  if (/oswald|condensed/.test(f)) return "oswald";
  if (/anton/.test(f)) return "anton";
  if (/poppins|geometric/.test(f)) return "poppins";
  if (/montserrat/.test(f)) return "montserrat";
  if (/righteous/.test(f)) return "righteous";
  return "noto_sans";
}

export function mapStyleToRecraft(style) {
  const s = String(style || "").toLowerCase();
  if (/vector|flat/.test(s)) return "vector_illustration";
  if (/logo|mark|icon/.test(s)) return "logo";
  if (/icon/.test(s)) return "icon";
  if (/photo|real|photograph/.test(s)) return "realistic_image";
  if (/digital/.test(s)) return "digital_illustration";
  return "digital_illustration";
}

/**
 * Recraft v3 — the typography specialist with REAL fonts. Its `texts` field
 * takes structured type (family, size, weight, colour, position), which is
 * the closest thing to "a designer set this type" available from an API.
 */
export async function recraftGenerate(prompt, opts = {}) {
  const apiKey = keyFor(PROVIDER_RECRAFT);
  if (!apiKey) throw new Error("RECRAFT_API_KEY missing — add it to server/.env to enable Recraft");
  const model = opts.model || "recraft/recraft-v3";
  const size = sizeForAspect(opts.aspect_ratio, 1024);
  const brief = opts.brief || null;
  const colors = (brief?.palette || []).map((p) => p.hex).slice(0, 5);
  const texts = brief ? briefToTextSpecs(brief, { size }) : [];

  const body = {
    prompt: String(prompt || "").slice(0, 6000),
    style: RECRAFT_STYLES.includes(opts.style) ? opts.style : mapStyleToRecraft(opts.style || brief?.texture),
    image_size: { width: size.width, height: size.height },
    num_images: 1,
    format: "png",
  };
  if (colors.length) body.colors = colors;
  if (texts.length) body.texts = texts;
  if (opts.negative_prompt) body.negative_prompt = String(opts.negative_prompt).slice(0, 500);

  const res = await fetch("https://api.recraft.ai/v3/images/generations", {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(180_000),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(`Recraft HTTP ${res.status}: ${data?.error?.message || data?.detail || JSON.stringify(data).slice(0, 240)}`);
  }
  const image_url = data?.image?.url || data?.images?.[0]?.url || null;
  const image_b64 = data?.image?.b64_json || data?.images?.[0]?.b64_json || null;
  if (!image_url && !image_b64) {
    throw new Error(`Recraft returned no image: ${JSON.stringify(data).slice(0, 240)}`);
  }
  return { ok: true, model, image_url, image_b64, media_type: "image/png", provider: PROVIDER_RECRAFT };
}

// ---------------------------------------------------------------------------
// DISPATCH
// ---------------------------------------------------------------------------

export async function generateWithProvider(provider, prompt, opts = {}) {
  if (provider === PROVIDER_IDEOGRAM) return ideogramGenerate(prompt, opts);
  if (provider === PROVIDER_RECRAFT) return recraftGenerate(prompt, opts);
  throw new Error(`unknown image provider '${provider}'`);
}

export { RECRAFT_FONTS, RECRAFT_STYLES };
