// model_config.js — single source of truth for every AI model slug.
// Resolution order: Creator-console override (model_overrides table) > env var > baked default.
// Server submodules (creative/*) import M() from here and resolve slugs at
// REQUEST time, so console changes take effect instantly with no restart.
// index.js additionally rebuilds its chat fallback chains on change.
let _db = null;
export function setModelDb(db) {
  _db = db;
  try {
    _db.exec(`CREATE TABLE IF NOT EXISTS model_overrides (
      key TEXT PRIMARY KEY, value TEXT, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )`);
  } catch {}
}
export const MODEL_DEFAULTS = {
  OR_PRIMARY_MODEL: "z-ai/glm-5.3-flash",
  OR_FAST_MODEL: "openai/gpt-oss-20b",
  OR_DRAFT_MODEL: "deepseek/deepseek-v4-flash",
  OR_IMAGE_MODEL: "bytedance-seed/seedream-5-0-flash",
  OR_STT_MODEL: "openai/whisper-large-v3",
  OR_DESIGN_MODEL: "inclusionai/ming-image-0.1-design",
  OR_LAYER_MODEL: "inclusionai/ming-image-0.1-design-layer",
  OR_VIDEO_MODEL: "heygen/heygen-video-1", // HeyGen Video: promo $0.01/s WITH audio
  OR_GENJUTSU_MODEL: "bytedance/seedance-2.5",
  OR_GRAPHICS_MODEL: "",
  OR_TYPE_MODEL: "",
  OR_VECTOR_MODEL: "",
  OR_AVATAR_MODEL: "heygen/avatar-iv", // talking-head avatar primary ($0.05/s)
  OR_DIRECTOR_MODEL: "", // blank = follows the fast model
  OR_FREE_FAST: "qwen/qwen3.8-27b:free",
  OR_FREE_DRAFT: "nvidia/nemotron-3-super-120b-a12b:free",
};
export const MODEL_KEYS = Object.keys(MODEL_DEFAULTS);
export const MODEL_META = {
  OR_PRIMARY_MODEL: { label: "Primary brain", hint: "chat, tools, agent, vision" },
  OR_FAST_MODEL: { label: "Fast model", hint: "greetings, verifier, grounding" },
  OR_DRAFT_MODEL: { label: "Draft model", hint: "cheap bulk reasoning" },
  OR_IMAGE_MODEL: { label: "Image model", hint: "$0.018/img Seedream Flash class" },
  OR_STT_MODEL: { label: "Speech-to-text", hint: "transcribe + subtitles" },
  OR_DESIGN_MODEL: { label: "Design model", hint: "legible-text graphics" },
  OR_LAYER_MODEL: { label: "Layer model", hint: "flat-design decomposition" },
  OR_VIDEO_MODEL: { label: "Video model (primary)", hint: "text-to-video clips" },
  OR_GENJUTSU_MODEL: { label: "Genjutsu model", hint: "reference-heavy cinema shots" },
  OR_GRAPHICS_MODEL: { label: "Graphics model", hint: "blank = design model" },
  OR_TYPE_MODEL: { label: "Type model", hint: "typography posters (blank = off)" },
  OR_VECTOR_MODEL: { label: "Vector model", hint: "logos/brand (blank = off)" },
  OR_AVATAR_MODEL: { label: "Avatar model (primary)", hint: "photo+voice talking head" },
  OR_DIRECTOR_MODEL: { label: "Cinema director", hint: "blank = fast model" },
  OR_FREE_FAST: { label: "Free fast", hint: ":free fallback quota" },
  OR_FREE_DRAFT: { label: "Free draft", hint: ":free fallback quota" },
};
export function overrideOf(key) {
  try {
    if (!_db) return null;
    const r = _db.prepare("SELECT value FROM model_overrides WHERE key = ?").get(key);
    const v = String(r?.value || "").trim();
    return v || null;
  } catch { return null; }
}
export const M = (key) => overrideOf(key) || process.env[key] || MODEL_DEFAULTS[key] || "";
