import fetch from "node-fetch";

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || "";
const OR_FAST_MODEL = process.env.OR_FAST_MODEL || "openai/gpt-oss-20b";

function orHeaders() {
  return {
    Authorization: `Bearer ${OPENROUTER_API_KEY}`,
    "Content-Type": "application/json",
    "HTTP-Referer": "https://frit.local",
    "X-Title": "FRIT",
  };
}

async function callOpenRouterChat(messages, maxTokens = 800) {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: orHeaders(),
    body: JSON.stringify({ model: OR_FAST_MODEL, messages, temperature: 0.3, max_tokens: maxTokens }),
    signal: AbortSignal.timeout(90_000),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error?.message || `Chat API HTTP ${res.status}`);
  return data;
}

function safeJsonParse(text) {
  try { return JSON.parse(text.match(/\{[\s\S]*\}/)?.[0] || "{}"); } catch { return {}; }
}

export async function suggestPrompts(idea, context = "") {
  const out = await callOpenRouterChat([
    { role: "system", content: "You are a creative prompt engineer. Given a vague idea, generate 3 detailed, specific creative prompts suitable for AI image/video generation. Each prompt should include subject, style, lighting, composition, and mood. Return JSON: {suggestions: [{prompt, description, style_preset}]}" },
    { role: "user", content: `Idea: ${idea}${context ? `\nContext: ${context}` : ""}` },
  ], 800);
  const text = out.choices?.[0]?.message?.content || "";
  return safeJsonParse(text).suggestions || [];
}

export async function enhancePrompt(prompt, context = "") {
  const out = await callOpenRouterChat([
    { role: "system", content: "You are a prompt enhancer. Rewrite the user's prompt with professional photography/film terminology: camera angle, lighting, depth of field, color grading, composition. Also generate a negative prompt. Return JSON: {enhanced_prompt, negative_prompt, technical_details}" },
    { role: "user", content: `Prompt: ${prompt}${context ? `\nContext: ${context}` : ""}` },
  ], 600);
  const text = out.choices?.[0]?.message?.content || "";
  return safeJsonParse(text);
}

export async function recommendStyles(imageAnalysis) {
  const out = await callOpenRouterChat([
    { role: "system", content: "Based on the image analysis, recommend 3-5 creative filters, transitions, or styles that would enhance it. Return JSON: {recommendations: [{name, description, params, preview_hint}]}" },
    { role: "user", content: `Image analysis: ${JSON.stringify(imageAnalysis).slice(0, 1000)}` },
  ], 500);
  const text = out.choices?.[0]?.message?.content || "";
  return safeJsonParse(text).recommendations || [];
}

export async function generateNegativePrompt(subject) {
  const out = await callOpenRouterChat([
    { role: "system", content: "Generate a negative prompt for AI image generation. List things to AVOID in the image. Return ONLY the negative prompt text, comma-separated, no explanations." },
    { role: "user", content: `Subject: ${subject}` },
  ], 200);
  return out.choices?.[0]?.message?.content || "blurry, low quality, distorted, deformed, ugly, bad anatomy";
}

export const STYLE_PRESETS = [
  { id: "realistic", name: "Realistic", category: "photography", description: "Photorealistic, high detail, professional" },
  { id: "cartoon", name: "Cartoon", category: "illustration", description: "Bold outlines, vibrant colors, animated" },
  { id: "abstract", name: "Abstract", category: "art", description: "Geometric shapes, bold colors, non-representational" },
  { id: "vangogh", name: "Van Gogh", category: "art", description: "Oil painting, swirling brushstrokes, post-impressionist" },
  { id: "cyberpunk", name: "Cyberpunk", category: "art", description: "Neon lights, futuristic, high contrast" },
  { id: "watercolor", name: "Watercolor", category: "art", description: "Soft washes, delicate colors, artistic" },
  { id: "oilpainting", name: "Oil Painting", category: "art", description: "Classical oil painting, rich textures" },
  { id: "sketch", name: "Sketch", category: "illustration", description: "Pencil sketch, monochrome, detailed linework" },
  { id: "pixelart", name: "Pixel Art", category: "retro", description: "16-bit, retro game aesthetic" },
  { id: "anime", name: "Anime", category: "illustration", description: "Anime style, manga illustration, vibrant" },
  { id: "noir", name: "Film Noir", category: "photography", description: "Black and white, high contrast, dramatic shadows" },
  { id: "vintage", name: "Vintage", category: "photography", description: "Retro film look, faded colors, grain" },
];
