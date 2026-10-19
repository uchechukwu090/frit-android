import fetch from "node-fetch";
import { M } from "../model_config.js";
import { buildCues } from "./subtitles.js";

function orKey() {
  return process.env.OPENROUTER_API_KEY || "";
}

function orHeaders() {
  return {
    Authorization: `Bearer ${orKey()}`,
    "Content-Type": "application/json",
    "HTTP-Referer": "https://frit.local",
    "X-Title": "FRIT",
  };
}

async function callOpenRouterImage(prompt, opts = {}) {
  if (!orKey()) throw new Error("OPENROUTER_API_KEY missing");
  const body = { model: opts.model || M("OR_IMAGE_MODEL"), prompt: String(prompt || "") };
  if (opts.output_format) body.output_format = opts.output_format;
  if (opts.input_references?.length) body.input_references = opts.input_references;
  if (opts.aspect_ratio) body.aspect_ratio = opts.aspect_ratio;
  const res = await fetch("https://openrouter.ai/api/v1/images", {
    method: "POST",
    headers: orHeaders(),
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(180_000),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error?.message || `Image API HTTP ${res.status}`);
  const item = data?.data?.[0] || data?.data || data?.images?.[0] || {};
  const out = {
    ok: true,
    model: body.model,
    image_url: item.url || null,
    image_b64: item.b64_json || item.b64Json || item.base64 || null,
  };
  if (!out.image_url && !out.image_b64) throw new Error("Image API returned no image");
  return out;
}

async function callOpenRouterChat(messages, maxTokens = 800) {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: orHeaders(),
    body: JSON.stringify({
      model: M("OR_FAST_MODEL"),
      messages,
      temperature: 0.3,
      max_tokens: maxTokens,
    }),
    signal: AbortSignal.timeout(90_000),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error?.message || `Chat API HTTP ${res.status}`);
  return data;
}

function safeJsonParse(text) {
  try { return JSON.parse(text.match(/\{[\s\S]*\}/)?.[0] || "{}"); } catch { return {}; }
}

export async function removeBackground(imageBuffer) {
  const base64 = imageBuffer.toString("base64");
  return callOpenRouterImage("Remove the background, make it transparent", {
    output_format: "png",
    input_references: [{ type: "image_url", image_url: { url: `data:image/png;base64,${base64}` } }],
  });
}

export async function retouch(imageBuffer, options = {}) {
  const base64 = imageBuffer.toString("base64");
  const parts = [];
  if (options.skin_smooth) parts.push("smooth skin texture");
  if (options.object_removal) parts.push("remove unwanted objects");
  if (options.lighting) parts.push("correct lighting and exposure");
  const instruction = parts.length ? `Edit this image: ${parts.join(", ")}` : "Enhance and retouch this image";
  return callOpenRouterImage(instruction, {
    input_references: [{ type: "image_url", image_url: { url: `data:image/png;base64,${base64}` } }],
  });
}

export async function styleTransfer(imageBuffer, style) {
  const base64 = imageBuffer.toString("base64");
  const stylePrompts = {
    vangogh: "in the style of Van Gogh, oil painting, swirling brushstrokes, post-impressionist",
    cyberpunk: "cyberpunk style, neon lights, futuristic, high contrast, blade runner aesthetic",
    watercolor: "watercolor painting, soft washes, artistic, delicate colors",
    cartoon: "cartoon style, bold outlines, vibrant colors, animated look",
    abstract: "abstract art, geometric shapes, bold colors, non-representational",
    realistic: "photorealistic, high detail, professional photography, 8k",
    oilpainting: "classical oil painting, rich textures, renaissance style",
    sketch: "pencil sketch, monochrome, detailed linework, hand-drawn",
    pixelart: "pixel art style, 16-bit, retro game aesthetic",
    anime: "anime style, manga illustration, vibrant, detailed",
  };
  const stylePrompt = stylePrompts[style] || style;
  return callOpenRouterImage(`Transform this image ${stylePrompt}`, {
    input_references: [{ type: "image_url", image_url: { url: `data:image/png;base64,${base64}` } }],
  });
}

export async function generativeFill(imageBuffer, maskBuffer, prompt) {
  const base64 = imageBuffer.toString("base64");
  const mask64 = maskBuffer.toString("base64");
  return callOpenRouterImage(prompt || "Fill the masked area seamlessly, matching the surrounding context", {
    output_format: "png",
    input_references: [
      { type: "image_url", image_url: { url: `data:image/png;base64,${base64}` } },
      { type: "image_url", image_url: { url: `data:image/png;base64,${mask64}` } },
    ],
  });
}

export async function upscaleImage(imageBuffer, scale = 2) {
  const base64 = imageBuffer.toString("base64");
  return callOpenRouterImage(`Upscale this image ${scale}x, enhance details, maintain quality`, {
    input_references: [{ type: "image_url", image_url: { url: `data:image/png;base64,${base64}` } }],
  });
}

// F-13: request REAL timestamps. The STT endpoint supports
// response_format=verbose_json + timestamp_granularities=["word","segment"],
// which returns the provider's own word/segment boundaries and the true audio
// duration. Not every provider supports it, so the request degrades: a 400
// falls back to the plain transcript and we fall back to a duration-weighted
// estimate instead of the old fixed "0.4s per word" guess.
export async function generateSubtitles(audioBuffer, language = "auto", opts = {}) {
  const b64 = audioBuffer.toString("base64");
  const model = M("OR_STT_MODEL");
  const baseBody = {
    model,
    input_audio: { data: b64, format: opts.format || "wav" },
    ...(language && language !== "auto" ? { language } : {}),
  };
  const body = { ...baseBody, response_format: "verbose_json", timestamp_granularities: ["word", "segment"] };

  let res = await fetch("https://openrouter.ai/api/v1/audio/transcriptions", {
    method: "POST",
    headers: orHeaders(),
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(120_000),
  });

  // Provider rejected verbose_json (or the model can't do timestamps) — retry
  // with the plain transcript so the user still gets captions.
  let verbose = true;
  if (!res.ok && (res.status === 400 || res.status === 422)) {
    res = await fetch("https://openrouter.ai/api/v1/audio/transcriptions", {
      method: "POST",
      headers: orHeaders(),
      body: JSON.stringify(baseBody),
      signal: AbortSignal.timeout(120_000),
    });
    verbose = false;
  }
  if (!res.ok) return { subtitles: [], text: "", timing: "none" };

  const data = await res.json().catch(() => ({}));
  const text = String(data.text || "").trim();
  if (!text) return { subtitles: [], text: "", timing: "none" };

  const duration = Number(data.duration) || Number(data?.usage?.seconds) || 0;
  const { cues, timing } = buildCues({
    words: verbose ? data.words : null,
    segments: verbose ? data.segments : null,
    text,
    duration,
  });
  return {
    subtitles: cues,
    text,
    timing,
    duration: duration || (cues.length ? cues[cues.length - 1].end : 0),
    language: data.language || (language === "auto" ? null : language),
  };
}

// F-13: translate each cue individually so the ORIGINAL timings stay valid.
// The old implementation translated the whole transcript as one blob and then
// re-chunked the translated words by count, which silently shifted every cue
// after the first and produced mismatched subtitles.
export async function translateSubtitles(subtitles, targetLanguage) {
  const list = Array.isArray(subtitles) ? subtitles : [];
  if (!list.length) return [];

  const numbered = list.map((s, i) => `${i + 1}. ${String(s.text || "").trim()}`).join("\n");
  const out = await callOpenRouterChat([
    {
      role: "system",
      content:
        `Translate each numbered line into ${targetLanguage}. Keep every line on its own line, keep the "N. " prefix, and keep the same number of lines. ` +
        `Return ONLY the translated lines, no commentary.`,
    },
    { role: "user", content: numbered },
  ], 2000);
  const raw = String(out.choices?.[0]?.message?.content || "");
  const parsed = parseNumberedLines(raw, list.length);

  return list.map((s, i) => {
    const t = parsed[i];
    return { start: s.start, end: s.end, text: t || String(s.text || "") };
  });
}

// Pull "N. text" lines out of a model reply, tolerating markdown fences,
// blank lines and a missing/renumbered prefix.
function parseNumberedLines(raw, expected) {
  const lines = String(raw || "")
    .split(/\r?\n/)
    .map((l) => l.replace(/^\s*```[a-z]*\s*/i, "").trim())
    .filter(Boolean);
  const picked = [];
  for (const line of lines) {
    const m = line.match(/^(\d+)\s*[.)]\s*(.*)$/);
    if (m) picked.push({ n: Number(m[1]), text: m[2].trim() });
    else if (picked.length) picked[picked.length - 1].text += " " + line;
    else picked.push({ n: picked.length + 1, text: line });
  }
  if (!picked.length) return [];
  const out = new Array(expected).fill("");
  for (const p of picked) {
    const idx = p.n - 1;
    if (idx >= 0 && idx < expected && !out[idx]) out[idx] = p.text;
  }
  // If numbering was lost, keep the reply order.
  if (out.every((t) => !t)) {
    picked.forEach((p, i) => { if (i < expected) out[i] = p.text; });
  }
  return out;
}

export async function storyboard(script, style = "", numFrames = 6) {
  const out = await callOpenRouterChat([
    { role: "system", content: `Convert this script into a storyboard with ${numFrames} frames. For each frame, provide a detailed image generation prompt. ${style ? `Style: ${style}.` : ""} Return JSON: {frames: [{index, description, prompt}]}` },
    { role: "user", content: script.slice(0, 4000) },
  ], 2000);
  const text = out.choices?.[0]?.message?.content || "";
  return safeJsonParse(text).frames || [];
}

export async function analyzeImage(imageBuffer) {
  const base64 = imageBuffer.toString("base64");
  const out = await callOpenRouterChat([
    { role: "user", content: [
      { type: "text", text: "Analyze this image and suggest creative improvements. Return JSON: {suggestions: [{type, description, params}]}" },
      { type: "image_url", image_url: { url: `data:image/png;base64,${base64}` } },
    ]},
  ], 800);
  const text = out.choices?.[0]?.message?.content || "";
  return safeJsonParse(text).suggestions || [];
}
