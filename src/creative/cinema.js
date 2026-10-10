import fetch from "node-fetch";
import { M } from "../model_config.js";

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || "";
const directorModel = () => M("OR_DIRECTOR_MODEL") || M("OR_FAST_MODEL");

function orHeaders() {
  return {
    Authorization: `Bearer ${OPENROUTER_API_KEY}`,
    "Content-Type": "application/json",
    "HTTP-Referer": "https://frit.local",
    "X-Title": "FRIT",
  };
}

async function orChat(messages, maxTokens = 1200) {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: orHeaders(),
    body: JSON.stringify({ model: directorModel(), messages, temperature: 0.4, max_tokens: maxTokens }),
    signal: AbortSignal.timeout(120_000),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error?.message || `Chat API HTTP ${res.status}`);
  return data;
}

function safeJsonParse(text) {
  try { return JSON.parse(text.match(/\{[\s\S]*\}/)?.[0] || "{}"); } catch { return {}; }
}

export const CINEMA_OPTIONS = {
  genres: ["General", "Action", "Epic", "Drama", "Comedy", "Horror", "Noir", "Documentary", "Sci-Fi", "Romance"],
  eras: ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s", "Timeless"],
  tempos: ["Single Shot", "Slow Burn", "Linear", "Ramp Up", "Flash In", "Flash Out", "Slow-mo", "Bullet Time", "Impact", "Chaotic"],
  cameras: ["35mm Film", "65mm Film", "Alexa 65", "RED Komodo", "8mm Film", "16mm Film", "Super 8", "DV Camcorder", "VHS Handycam", "Broadcast TV", "Smartphone", "Drone", "IMAX"],
  lenses: ["Clean Sharp", "Anamorphic", "Vintage Soft", "Fisheye", "Macro", "Tilt-Shift", "Telephoto 200mm", "Wide 18mm", "50mm Prime", "35mm Prime", "85mm Portrait"],
  apertures: ["f/1.4", "f/1.8", "f/2.8", "f/4", "f/5.6", "f/8", "f/11"],
  moves: ["Static", "Pan Left", "Pan Right", "Tilt Up", "Tilt Down", "Dolly In", "Dolly Out", "Tracking", "Crane Up", "Crane Down", "Orbit", "POV", "Robot Arm", "Helicopter Shot", "FPV Drone", "Handheld", "Gimbal Walk", "Vertigo Dolly-Zoom", "Whip Pan", "Slow Push-In", "360 Spin"],
  angles: ["Eye Level", "Low Angle", "High Angle", "Bird's Eye", "Worm's Eye", "Dutch Tilt", "Over Shoulder", "Close-Up", "Extreme Close-Up", "Wide Establishing", "Aerial"],
  lighting: ["Natural Daylight", "Golden Hour", "Blue Hour", "Neon Night", "Moody Low-Key", "Bright High-Key", "Candlelight", "Studio Softbox", "Backlit Silhouette", "Custom"],
  palettes: ["Teal & Orange", "Noir B&W", "Vintage Fade", "Cyberpunk Neon", "Desert Warm", "Forest Green", "Pastel Dream", "Crimson Drama", "Arctic Blue", "Golden Epic", "Horror Desat", "Romance Blush", "Documentary Neutral", "Music Video Pop", "Film Stock 35mm", "Bleach Bypass", "Lagos Neon", "Sahel Dust", "Nollywood Glow", "Afrobeats Pop", "K-Drama Soft", "Anime Cel", "Giallo Red", "Wes Anderson Pastel", "Euphoria Glitter", "Jungle Lush", "Monsoon Grey", "Sahara Gold"],
  emotions: ["Neutral", "Joy", "Anger", "Sorrow", "Fear", "Surprise", "Determination", "Serene"],
  aspect_ratios: ["16:9", "9:16", "1:1", "4:5", "2.39:1"],
};

// Shot Recipes: one-tap curated looks (camera+lens+light+grade in one).
// Several are FRIT-originals tuned for Nollywood/Afrobeats/K-drama creators.
export const SHOT_RECIPES = [
  { id: "nollywood_epic", name: "Nollywood Epic", description: "Big drama, golden warmth, sweeping moves", params: { genre: "Drama", era: "2020s", tempo: "Ramp Up", camera: "Alexa 65", lens: "Anamorphic", aperture: "f/2.8", move: "Crane Up", angle: "Wide Establishing", lighting: "Golden Hour", palette: "Nollywood Glow", emotion: "Determination" } },
  { id: "afrobeats_video", name: "Afrobeats Video", description: "High-energy party visuals, neon pop", params: { genre: "General", era: "2020s", tempo: "Chaotic", camera: "RED Komodo", lens: "Wide 18mm", aperture: "f/1.8", move: "360 Spin", angle: "Low Angle", lighting: "Neon Night", palette: "Afrobeats Pop", emotion: "Joy" } },
  { id: "kdrama_romance", name: "K-Drama Romance", description: "Soft glow, slow push-ins, tender grade", params: { genre: "Romance", era: "2020s", tempo: "Slow Burn", camera: "35mm Film", lens: "85mm Portrait", aperture: "f/1.4", move: "Slow Push-In", angle: "Close-Up", lighting: "Golden Hour", palette: "K-Drama Soft", emotion: "Serene" } },
  { id: "wes_symmetry", name: "Wes Symmetry", description: "Dead-center pastel storybook frames", params: { genre: "Comedy", era: "Timeless", tempo: "Linear", camera: "35mm Film", lens: "50mm Prime", aperture: "f/8", move: "Static", angle: "Eye Level", lighting: "Bright High-Key", palette: "Wes Anderson Pastel", emotion: "Neutral" } },
  { id: "bbc_doc", name: "True Documentary", description: "Neutral, honest, handheld realism", params: { genre: "Documentary", era: "2020s", tempo: "Linear", camera: "Broadcast TV", lens: "35mm Prime", aperture: "f/4", move: "Handheld", angle: "Eye Level", lighting: "Natural Daylight", palette: "Documentary Neutral", emotion: "Neutral" } },
  { id: "nike_energy", name: "Sports Ad Energy", description: "Gritty slow-mo impact moments", params: { genre: "Action", era: "2020s", tempo: "Impact", camera: "RED Komodo", lens: "Telephoto 200mm", aperture: "f/2.8", move: "Tracking", angle: "Low Angle", lighting: "Backlit Silhouette", palette: "Teal & Orange", emotion: "Determination" } },
  { id: "horror_night", name: "Horror Night", description: "Dread-filled desaturated darkness", params: { genre: "Horror", era: "Timeless", tempo: "Slow Burn", camera: "16mm Film", lens: "Wide 18mm", aperture: "f/1.8", move: "Slow Push-In", angle: "Dutch Tilt", lighting: "Moody Low-Key", palette: "Horror Desat", emotion: "Fear" } },
  { id: "anime_opening", name: "Anime Opening", description: "Cel-shaded high-speed title energy", params: { genre: "General", era: "2020s", tempo: "Flash In", camera: "Drone", lens: "Wide 18mm", aperture: "f/4", move: "FPV Drone", angle: "Bird's Eye", lighting: "Natural Daylight", palette: "Anime Cel", emotion: "Determination" } },
  { id: "news_desk", name: "News Report", description: "Clean broadcast credibility", params: { genre: "Documentary", era: "2020s", tempo: "Single Shot", camera: "Broadcast TV", lens: "50mm Prime", aperture: "f/5.6", move: "Static", angle: "Eye Level", lighting: "Studio Softbox", palette: "Documentary Neutral", emotion: "Neutral" } },
  { id: "noir_detective", name: "Neo-Noir", description: "Rain, neon, and moral ambiguity", params: { genre: "Noir", era: "1980s", tempo: "Slow Burn", camera: "35mm Film", lens: "Anamorphic", aperture: "f/1.8", move: "Dolly In", angle: "Dutch Tilt", lighting: "Neon Night", palette: "Noir B&W", emotion: "Determination" } },
];

export function buildDirectorPrompt(p = {}) {
  const bits = [];
  if (p.prompt) bits.push(`Scene: ${p.prompt}`);
  if (p.genre) bits.push(`Genre: ${p.genre}`);
  if (p.era && p.era !== "Timeless") bits.push(`Era aesthetic: ${p.era}`);
  if (p.tempo) bits.push(`Cutting tempo: ${p.tempo}`);
  if (p.camera) bits.push(`Shot on ${p.camera}`);
  if (p.lens) bits.push(`${p.lens} lens`);
  if (p.aperture) bits.push(`aperture ${p.aperture}, ${["f/1.4", "f/1.8", "f/2.8"].includes(p.aperture) ? "shallow depth of field, creamy bokeh" : "deep focus"}`);
  if (p.move) bits.push(`Camera move: ${p.move}`);
  if (p.angle) bits.push(`Camera angle: ${p.angle}`);
  if (p.lighting) bits.push(`Lighting: ${p.lighting}`);
  if (p.palette) bits.push(`Color grade: ${p.palette}`);
  if (p.emotion && p.emotion !== "Neutral") bits.push(`Performance: ${p.emotion.toLowerCase()} expression, expressive acting`);
  if (p.negative) bits.push(`Avoid: ${p.negative}`);
  else bits.push("Avoid: blurry, low quality, distorted faces, deformed hands, watermark, text overlay");
  return bits.join(". ") + ".";
}

// AI Director: brief -> shot list with pre-filled cinematography params.
export async function aiDirect(brief, numShots = 4) {
  const out = await orChat([
    { role: "system", content: `You are a film director. Break the brief into ${numShots} shots. Return ONLY JSON: {shots:[{index, description, prompt, genre, camera, lens, aperture, move, angle, lighting, palette, emotion, duration_s}]}. Valid genres: ${CINEMA_OPTIONS.genres.join(", ")}. Valid moves: ${CINEMA_OPTIONS.moves.join(", ")}. Valid angles: ${CINEMA_OPTIONS.angles.join(", ")}. Valid lighting: ${CINEMA_OPTIONS.lighting.join(", ")}. Valid palettes: ${CINEMA_OPTIONS.palettes.join(", ")}. Durations 3-8s each.` },
    { role: "user", content: String(brief || "").slice(0, 3000) },
  ], 2000);
  const text = out.choices?.[0]?.message?.content || "";
  const parsed = safeJsonParse(text);
  const shots = Array.isArray(parsed.shots) ? parsed.shots : [];
  return shots.map((s, i) => ({
    index: s.index ?? i + 1,
    description: s.description || "",
    prompt: s.prompt || s.description || "",
    genre: s.genre || "General",
    camera: s.camera || "35mm Film",
    lens: s.lens || "50mm Prime",
    aperture: s.aperture || "f/2.8",
    move: s.move || "Static",
    angle: s.angle || "Eye Level",
    lighting: s.lighting || "Natural Daylight",
    palette: s.palette || "Film Stock 35mm",
    emotion: s.emotion || "Neutral",
    duration_s: Math.min(8, Math.max(3, Number(s.duration_s) || 4)),
  }));
}

// Katana: one natural-language command -> executable edit ops.
export async function katanaParse(command) {
  const out = await orChat([
    { role: "system", content: `You are a video edit planner. Convert the user's one-line edit command into JSON ops. Return ONLY JSON: {ops:[{type, params}], reply}. Valid types: trim{start_s,end_s}, captions{language,style}, translate_captions{target_language}, speed{factor}, upscale{resolution}, replace_audio{note}, concat{note}, extract_frame{timestamp_s}, export{platform}. Keep ops minimal — only what the command asks.` },
    { role: "user", content: String(command || "").slice(0, 1000) },
  ], 600);
  const text = out.choices?.[0]?.message?.content || "";
  const parsed = safeJsonParse(text);
  return { ops: Array.isArray(parsed.ops) ? parsed.ops : [], reply: parsed.reply || "" };
}

// Influencer persona: attributes -> persona + reusable character prompts.
export async function influencerPersona(attrs = {}) {
  const desc = Object.entries(attrs).map(([k, v]) => `${k}: ${v}`).join(", ");
  const out = await orChat([
    { role: "system", content: "You are a virtual-influencer designer. Given attributes, return ONLY JSON: {name, handle, bio, niche, voice_tone, character_prompt, content_pillars:[...], post_ideas:[{caption, visual_prompt}]}. character_prompt must be a reusable face/look description (age, ethnicity, hair, style, defining features) that keeps the SAME person across generations. 5 post ideas." },
    { role: "user", content: desc.slice(0, 1500) || "lifestyle influencer" },
  ], 1500);
  return safeJsonParse(out.choices?.[0]?.message?.content || "");
}

// Ads: brief (+ product context) -> script + shot prompts + static ad prompt.
export async function adPlan(brief, product = "") {
  const out = await orChat([
    { role: "system", content: "You are an ad creative director. Return ONLY JSON: {hook, script_30s, shots:[{kind:'image'|'video', prompt, duration_s}], static_ad_prompt, caption, cta}. Shots max 4. static_ad_prompt is a detailed prompt for a 1:1 product ad poster with space for headline text." },
    { role: "user", content: `Product: ${String(product).slice(0, 800)}\nBrief: ${String(brief).slice(0, 2000)}` },
  ], 1500);
  return safeJsonParse(out.choices?.[0]?.message?.content || "");
}
