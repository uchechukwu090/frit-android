// ============================================================================
// CREATIVE CAPABILITIES — what the ACTIVE model can actually render
// ----------------------------------------------------------------------------
// The studios used to hardcode their own option lists, so the UI offered
// choices the model rejects (4:5 on Recraft, a 6s clip on a 4s model) and the
// server silently snapped them away. This module turns OpenRouter's own model
// metadata into a small, stable capability object the app renders from:
//
//   { role, model, label, kinds, aspects, resolutions, durations, maxN,
//     reference, audio, source }
//
// `source` is "live" when the values came from the provider metadata and
// "known"/"fallback" when we are using the family table or the documented
// defaults. Never invent a capability: an unknown field stays null/empty and
// the caller falls back to a conservative default.
// ============================================================================

import { aspectsFor, isTypeSpecialist } from "./imageProviders.js";

export const IMAGE_CAP_FALLBACK = {
  aspects: ["1:1", "4:3", "3:4", "16:9", "9:16"],
  resolutions: null,
  maxN: 4,
  reference: true,
  outputFormats: null,
};

export const VIDEO_CAP_FALLBACK = {
  aspects: ["16:9", "9:16", "1:1"],
  resolutions: ["720p"],
  durations: [4],
  audio: null,
  frameImages: [],
  reference: false,
};

/** Read the accepted values out of an OpenRouter `supported_parameters` entry. */
export function capValues(p) {
  if (p == null) return null;
  if (Array.isArray(p)) {
    const out = p.filter((v) => typeof v === "string" || typeof v === "number").map(String);
    return out.length ? out : null;
  }
  if (typeof p === "object") {
    const vals = p.enum || p.values || p.options || p.allowed_values;
    if (Array.isArray(vals) && vals.length) return vals.map(String);
  }
  return null;
}

/** Max of a range-style capability (`{type:"range", max}` or a plain number). */
export function capMax(p) {
  if (p == null) return null;
  const n = typeof p === "object" ? Number(p.max ?? p.maximum) : Number(p);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** Find the active slug in a provider list: exact, canonical, then family. */
export function findModelRecord(list, slug) {
  const want = String(slug || "").toLowerCase();
  if (!want || !Array.isArray(list)) return null;
  const exact = list.find((m) => String(m?.id || "").toLowerCase() === want);
  if (exact) return exact;
  const canon = list.find((m) => String(m?.canonical_slug || "").toLowerCase() === want);
  if (canon) return canon;
  const family = want.split("/")[0];
  return list.find((m) => String(m?.id || "").toLowerCase().startsWith(`${family}/`)) || null;
}

/** What an image model accepts — live metadata first, family table second. */
export function imageCapabilities(model, list, role = "image") {
  const rec = findModelRecord(list, model);
  const sp = rec?.supported_parameters || null;
  return {
    role,
    model,
    label: rec?.name || model,
    // Request kinds this role can produce. Images only.
    kinds: ["image"],
    aspects: capValues(sp?.aspect_ratio) || aspectsFor(model) || IMAGE_CAP_FALLBACK.aspects,
    resolutions: capValues(sp?.resolution) || IMAGE_CAP_FALLBACK.resolutions,
    maxN: capMax(sp?.n) || IMAGE_CAP_FALLBACK.maxN,
    reference: sp ? sp.input_references != null : IMAGE_CAP_FALLBACK.reference,
    outputFormats: capValues(sp?.output_format) || IMAGE_CAP_FALLBACK.outputFormats,
    // A type specialist renders real fonts itself; the type layer must not run.
    typeSpecialist: isTypeSpecialist(model),
    source: sp ? "live" : "known",
  };
}

/** What a video model accepts — only the fields the UI actually exposes. */
export function videoCapabilities(model, list, role = "video") {
  const rec = findModelRecord(list, model);
  const live = !!rec && !!(rec.supported_durations || rec.supported_resolutions);
  const frameImages = Array.isArray(rec?.supported_frame_images) ? rec.supported_frame_images : [];
  const passthrough = Array.isArray(rec?.allowed_passthrough_parameters) ? rec.allowed_passthrough_parameters : [];
  const durations = (Array.isArray(rec?.supported_durations) ? rec.supported_durations : [])
    .map(Number).filter((n) => Number.isFinite(n) && n > 0);
  return {
    role,
    model,
    label: rec?.name || model,
    kinds: ["video"],
    aspects: (Array.isArray(rec?.supported_aspect_ratios) && rec.supported_aspect_ratios.length)
      ? rec.supported_aspect_ratios.map(String)
      : VIDEO_CAP_FALLBACK.aspects,
    resolutions: (Array.isArray(rec?.supported_resolutions) && rec.supported_resolutions.length)
      ? rec.supported_resolutions.map(String)
      : VIDEO_CAP_FALLBACK.resolutions,
    durations: durations.length ? durations : VIDEO_CAP_FALLBACK.durations,
    audio: typeof rec?.generate_audio === "boolean" ? rec.generate_audio : VIDEO_CAP_FALLBACK.audio,
    frameImages,
    // Reference images are a request capability, so look at the passthrough list.
    reference: passthrough.includes("input_references") || VIDEO_CAP_FALLBACK.reference,
    source: live ? "live" : "known",
  };
}

/** The whole payload the app consumes (GET /creative-suite/capabilities). */
export function creativeCapabilities({ imageModel, designModel, videoModel, genjutsuModel, avatarModel }, { images = [], videos = [] } = {}) {
  const image = imageCapabilities(imageModel, images, "image");
  const design = imageCapabilities(designModel || imageModel, images, "design");
  return {
    ok: true,
    generated_at: Date.now(),
    source: { images: images.length ? "live" : "fallback", videos: videos.length ? "live" : "fallback" },
    roles: {
      image,
      design,
      video: videoCapabilities(videoModel, videos, "video"),
      genjutsu: videoCapabilities(genjutsuModel, videos, "genjutsu"),
      avatar: videoCapabilities(avatarModel, videos, "avatar"),
    },
  };
}
