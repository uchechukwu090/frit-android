import fetch from "node-fetch";
import { M } from "../model_config.js";

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || "";

function orHeaders() {
  return {
    Authorization: `Bearer ${OPENROUTER_API_KEY}`,
    "Content-Type": "application/json",
    "HTTP-Referer": "https://frit.local",
    "X-Title": "FRIT",
  };
}

async function callOpenRouterImage(prompt, opts = {}) {
  if (!OPENROUTER_API_KEY) throw new Error("OPENROUTER_API_KEY missing");
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

export async function generateSubtitles(audioBuffer, language = "auto") {
  const b64 = audioBuffer.toString("base64");
  const res = await fetch("https://openrouter.ai/api/v1/audio/transcriptions", {
    method: "POST",
    headers: orHeaders(),
    body: JSON.stringify({ model: M("OR_STT_MODEL"), input_audio: { data: b64, format: "wav" }, language: language === "auto" ? undefined : language }),
    signal: AbortSignal.timeout(120_000),
  });
  if (!res.ok) return { subtitles: [], text: "" };
  const data = await res.json().catch(() => ({}));
  const text = data.text || "";
  if (!text) return { subtitles: [], text: "" };
  const words = text.split(" ");
  const subtitles = [];
  const wordsPerChunk = 5;
  const secondsPerWord = 0.4;
  for (let i = 0; i < words.length; i += wordsPerChunk) {
    const chunk = words.slice(i, i + wordsPerChunk).join(" ");
    subtitles.push({ start: i * secondsPerWord, end: (i + wordsPerChunk) * secondsPerWord, text: chunk });
  }
  return { subtitles, text };
}

export async function translateSubtitles(subtitles, targetLanguage) {
  const text = subtitles.map(s => s.text).join(" ");
  const out = await callOpenRouterChat([
    { role: "system", content: `Translate the following text to ${targetLanguage}. Return ONLY the translated text, no explanations.` },
    { role: "user", content: text },
  ], 500);
  const translated = out.choices?.[0]?.message?.content || text;
  const translatedWords = translated.split(" ");
  const wordsPerChunk = Math.ceil(translatedWords.length / subtitles.length);
  return subtitles.map((s, i) => ({
    start: s.start,
    end: s.end,
    text: translatedWords.slice(i * wordsPerChunk, (i + 1) * wordsPerChunk).join(" "),
  }));
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
