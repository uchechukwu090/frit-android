// ============================================================================
// DIRECTOR AGENT — the design agent's three stages, for motion
// ----------------------------------------------------------------------------
// The Design Agent researches -> briefs -> renders -> verifies a still. The
// AI Director needs the same skeleton, with two differences:
//
//   * the reference medium is a LOOK, not a layout — grade, lighting, lens
//     character, depth, framing. Same structure/content split: the look's
//     subject is never borrowed, only its visual grammar.
//   * the unit of work is N SHOTS, and verification is per shot: the rendered
//     clip is sampled into frames and each frame is checked against the shot
//     it was supposed to be. THAT is what "observe and correct" means — the
//     old route only retried on an exception, it never looked at the result.
//
// Deps are injected so the whole thing is testable with no network:
//   planFilm(goal, numShots, looks)   -> shots
//   renderShot(shot, {attempt, correction}) -> {ok, kind, url, base64}
//   framesOf(shot)                    -> [dataUri]  (frames sampled from the clip)
//   verifyFrame(frameUri, shot)       -> {score, pass, issues}
// ============================================================================

import { hashSeed, styleTokenFor, splitDNA, structuralOnly } from "./designDNA.js";

const LOOK_TAXONOMY = {
  grade_temp: ["warm", "cool", "neutral"],
  grade_contrast: ["low", "medium", "high"],
  lighting: ["soft", "hard", "neon", "golden", "overcast", "mixed"],
  palette_temp: ["warm", "cool", "neutral", "earthy", "bold"],
  framing: ["wide", "medium", "close", "extreme-close"],
  lens_character: ["wide", "normal", "tele", "macro", "anamorphic"],
  depth: ["shallow", "medium", "deep"],
  texture: ["clean", "filmic", "grainy", "vibrant", "desaturated"],
  composition: ["centered", "rule-of-thirds", "leading-lines", "frame-within-frame"],
};

const MOVES = ["Static", "Pan Left", "Pan Right", "Tilt Up", "Tilt Down", "Dolly In", "Dolly Out", "Tracking", "Crane Up", "Crane Down", "Orbit", "Handheld", "Gimbal Walk", "Slow Push-In"];
const ANGLES = ["Eye Level", "Low Angle", "High Angle", "Bird's Eye", "Worm's Eye", "Dutch Tilt", "Over Shoulder", "Close-Up", "Extreme Close-Up", "Wide Establishing", "Aerial"];
const LENSES = ["Clean Sharp", "Anamorphic", "Vintage Soft", "Fisheye", "Macro", "Tilt-Shift", "Telephoto 200mm", "Wide 18mm", "50mm Prime", "35mm Prime", "85mm Portrait"];
const APERTURES = ["f/1.4", "f/1.8", "f/2.8", "f/4", "f/5.6", "f/8", "f/11"];

const str = (v, max = 200) => String(v ?? "").replace(/\s+/g, " ").trim().slice(0, max);
const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, Number(n) || 0));

function parseJsonLoose(text) {
  const t = String(text || "");
  const fenced = t.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1] : t;
  try { return JSON.parse(candidate); } catch { /* fall through */ }
  try { return JSON.parse(candidate.match(/\{[\s\S]*\}/)?.[0] || "{}"); } catch { return {}; }
}

// ---------------------------------------------------------------------------
// 1. RESEARCH — look references
// ---------------------------------------------------------------------------

/** Look research uses cinematography vocabulary, not layout vocabulary: the
 *  Director borrows a GRADE, not a poster grid. */
export function planLookQueries(goal, { seed = 0, maxQueries = 3, sources = ["pinterest", "behance"] } = {}) {
  const g = String(goal || "").replace(/\s+/g, " ").trim().slice(0, 120);
  if (!g) return [];
  const token = styleTokenFor(seed, ["cinematic still", "film still", "colour grade", "moody portrait", "documentary frame"]);
  const out = [];
  for (const s of sources) {
    const site = String(s || "").toLowerCase().replace(/[^a-z.]/g, "");
    if (site) out.push(`site:${site}.com ${g} ${token} cinematography`);
    if (out.length >= maxQueries) break;
  }
  out.push(`${g} ${token} cinematography lighting colour grade`);
  return [...new Set(out)].slice(0, Math.max(1, maxQueries));
}

/** Normalise a look reference into a borrowable LOOK (no subject). */
export function normalizeLook(raw) {
  const r = raw && typeof raw === "object" ? raw : {};
  const l = r.look && typeof r.look === "object" ? r.look : r;
  const out = {};
  for (const [key, allowed] of Object.entries(LOOK_TAXONOMY)) {
    const v = String(l?.[key] ?? "").trim().toLowerCase();
    out[key] = allowed.includes(v) ? v : allowed[0];
  }
  out.framing = String(l?.framing || "").toLowerCase().includes("close") ? "close" : out.framing;
  const split = splitDNA(r);
  return { look: out, palette: split.palette, mood: split.mood };
}

/** Read one look reference with the vision model. */
export async function readLookReference(ref, goal, { chat, fetchImpl, proxyUrl = null } = {}) {
  if (!chat) return null;
  let image = null;
  const v = String(ref || "");
  if (v.startsWith("data:")) {
    image = v;
  } else {
    try {
      const res = await fetchImpl(v, { signal: AbortSignal.timeout(12_000) });
      if (res?.ok) {
        const buf = Buffer.from(await res.arrayBuffer());
        if (buf.length && buf.length < 6 * 1024 * 1024) image = `data:image/jpeg;base64,${buf.toString("base64")}`;
      }
    } catch { /* fall through to the proxy / raw url */ }
  }
  if (!image) image = proxyUrl || v;
  if (!image) return null;

  const prompt = [
    "You are a cinematographer studying a look reference for a new film.",
    `The film is about: ${str(goal, 300)}`,
    "Describe the LOOK — how it was shot and graded — never the story it tells.",
    "Return ONLY JSON:",
    '{"look":{"grade_temp":"warm|cool|neutral","grade_contrast":"low|medium|high",',
    '"lighting":"soft|hard|neon|golden|overcast|mixed","palette_temp":"warm|cool|neutral|earthy|bold",',
    '"framing":"wide|medium|close|extreme-close","lens_character":"wide|normal|tele|macro|anamorphic",',
    '"depth":"shallow|medium|deep","texture":"clean|filmic|grainy|vibrant|desaturated",',
    '"composition":"centered|rule-of-thirds|leading-lines|frame-within-frame"},',
    '"palette":[{"hex":"#RRGGBB","role":"dominant|secondary|accent"}],"mood":"..."}',
  ].join("\n");
  const out = await chat([
    { role: "user", content: [
      { type: "text", text: prompt },
      { type: "image_url", image_url: { url: image } },
    ] },
  ], { max_tokens: 700, temperature: 0.2 });
  const raw = parseJsonLoose(out);
  const norm = normalizeLook(raw);
  return { ref, dna: raw, ...norm, structure: norm.look };
}

// ---------------------------------------------------------------------------
// 2. BRIEF — the shot list
// ---------------------------------------------------------------------------

/** Ask the director model for the shot list, briefed by the look DNA.
 *  Only the LOOK may be borrowed — the subject is the user's. */
export async function synthesizeShotPlan(goal, numShots = 4, looks = [], { chat } = {}) {
  if (!chat) return [];
  const lookBlock = looks
    .filter((l) => l?.look)
    .slice(0, 3)
    .map((l, i) => `LOOK ${i + 1} (visual grammar only, never its subject): ${JSON.stringify({ look: l.look, palette: l.palette }).slice(0, 600)}`)
    .join("\n");
  const system = [
    "You are a film director breaking a brief into a shot list.",
    "Each shot must be distinct: a different camera move, a different angle, a different framing.",
    "You may borrow the LOOK's grade, lighting and lens character. You must NOT reuse its subject or story.",
    "Return ONLY JSON:",
    '{"shots":[{"index":1,"description":"...","prompt":"what happens in the frame, in one sentence",',
    '"genre":"...","camera":"...","lens":"...","aperture":"f/2.8","move":"...","angle":"...",',
    '"lighting":"...","palette":"...","emotion":"...","duration_s":4}]}',
    `Valid moves: ${MOVES.join(", ")}.`,
    `Valid angles: ${ANGLES.join(", ")}.`,
    `Valid lenses: ${LENSES.join(", ")}.`,
    `Valid apertures: ${APERTURES.join(", ")}.`,
    "Durations 3-8s each.",
  ].join("\n");
  const user = [`FILM BRIEF: ${str(goal, 2000)}`, `SHOTS: ${numShots}`, lookBlock ? `\n${lookBlock}\n` : ""].filter(Boolean).join("\n");
  const messages = [{ role: "system", content: system }, { role: "user", content: user }];
  const first = parseJsonLoose(await chat(messages, { max_tokens: 2000, temperature: 0.5 }));
  let shots = Array.isArray(first.shots) ? first.shots : [];
  if (!shots.length) {
    const retry = parseJsonLoose(await chat([
      ...messages,
      { role: "assistant", content: "(reply was not JSON)" },
      { role: "user", content: "Return ONLY the JSON object described above — no prose." },
    ], { max_tokens: 2000, temperature: 0 }));
    shots = Array.isArray(retry.shots) ? retry.shots : [];
  }
  return shots.map((s, i) => normalizeShot(s, i));
}

export function normalizeShot(s = {}, i = 0) {
  return {
    index: s.index ?? i + 1,
    description: str(s.description, 300),
    prompt: str(s.prompt || s.description, 600),
    genre: str(s.genre, 40) || "General",
    camera: str(s.camera, 40) || "35mm Film",
    lens: str(s.lens, 40) || "50mm Prime",
    aperture: str(s.aperture, 12) || "f/2.8",
    move: str(s.move, 40) || "Static",
    angle: str(s.angle, 40) || "Eye Level",
    lighting: str(s.lighting, 40) || "Natural Daylight",
    palette: str(s.palette, 60) || "Film Stock 35mm",
    emotion: str(s.emotion, 40) || "Neutral",
    duration_s: clamp(Number(s.duration_s) || 4, 3, 8),
  };
}

/**
 * Shot variety. A director's shot list with the same move three times reads
 * as a slideshow, not a film. When two shots collide on (move, angle), the
 * later one is re-pointed at an unused option — deterministic, testable.
 */
export function enforceShotVariety(shots) {
  const list = Array.isArray(shots) ? shots.map((s) => ({ ...s })) : [];
  const seen = new Set();
  let changed = 0;
  for (const shot of list) {
    const key = `${shot.move}|${shot.angle}`;
    if (!seen.has(key)) { seen.add(key); continue; }
    const moveIdx = MOVES.indexOf(shot.move);
    const angleIdx = ANGLES.indexOf(shot.angle);
    const nextMove = MOVES[(moveIdx + 1 + changed) % MOVES.length];
    const nextAngle = ANGLES[(angleIdx + 1 + changed) % ANGLES.length];
    shot.move = nextMove;
    shot.angle = nextAngle;
    seen.add(`${nextMove}|${nextAngle}`);
    changed++;
  }
  return { shots: list, changed };
}

// ---------------------------------------------------------------------------
// 3. RENDER + VERIFY
// ---------------------------------------------------------------------------

/** Does this frame match the shot it was supposed to be? */
export async function verifyShotFrame(frame, shot, { chat, threshold = 0.7 } = {}) {
  if (!chat) return { score: 0, pass: false, issues: ["verification unavailable"], threshold, checked: false };
  if (!frame) return { score: 0, pass: false, issues: ["no frame to verify"], threshold, checked: false };
  const facts = {
    subject: shot?.prompt || shot?.description || "",
    framing: shot?.angle, move: shot?.move, lens: shot?.lens,
    lighting: shot?.lighting, grade: shot?.palette, emotion: shot?.emotion,
  };
  const ask = (nudge = "") => chat([
    { role: "user", content: [
      { type: "text", text: [
        "You are a strict director reviewing a rendered frame against its shot list.",
        `SHOT: ${JSON.stringify(facts)}`,
        "Check: is the subject present, does the framing/angle match, does the lighting and colour grade match, is the image clean (no artifacts, no extra limbs, no text).",
        'Return ONLY JSON: {"match_score":0.0,"pass":true|false,"issues":["specific, actionable problem"]}',
        nudge,
      ].filter(Boolean).join("\n") },
      { type: "image_url", image_url: { url: frame } },
    ] },
  ], { max_tokens: 500, temperature: 0 });

  let text = await ask();
  let parsed = parseJsonLoose(text);
  const usable = (p) => p && (p.match_score != null || p.score != null || typeof p.pass === "boolean");
  if (!usable(parsed)) {
    text = await ask("Your previous reply could not be parsed. Answer with the JSON object ONLY.");
    parsed = parseJsonLoose(text);
  }
  const score = clamp(Number(parsed?.match_score ?? parsed?.score) || 0, 0, 1);
  const issues = (Array.isArray(parsed?.issues) ? parsed.issues : []).map((i) => str(i, 140)).filter(Boolean).slice(0, 5);
  const pass = typeof parsed?.pass === "boolean" ? parsed.pass : score >= threshold;
  return { score: Math.round(score * 100) / 100, pass, issues, threshold, checked: true };
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
 * Run the director pipeline.
 *
 * Deps:
 *   research(goal, {seed})      -> {refs: [...]}          (optional)
 *   readLook(ref, goal)         -> {ref, look, palette}   (optional)
 *   planFilm(goal, n, looks)    -> shots
 *   renderShot(shot, {attempt, correction}) -> {ok, kind, url, base64}
 *   framesOf(shot)              -> [dataUri]
 *   verifyFrame(frameUri, shot) -> {score, pass, issues}
 */
export async function runDirectorAgent(input = {}, deps = {}, opts = {}) {
  const now = deps.now || (() => Date.now());
  const t0 = now();
  const totalMs = opts.totalMs ?? 480_000;
  const stages = [];
  const warnings = [];
  const left = () => Math.max(0, totalMs - (now() - t0));
  const mark = (stage) => stages.push({ stage, ms: now() - t0 });

  const goal = str(input.brief || input.goal || input.prompt, 2000);
  if (!goal) throw new Error("brief required");
  const numShots = clamp(Number(input.num_shots) || 4, 2, 8);
  const maxAttempts = input.auto_correct === false ? 1 : clamp(Number(input.max_attempts) || 2, 1, 3);
  const threshold = Number(input.threshold) || 0.7;
  const seed = Number.isFinite(Number(input.seed))
    ? Math.abs(Number(input.seed)) >>> 0
    : hashSeed(input.seed ?? "", input.uid ?? "", goal, now());

  // ---- research ----------------------------------------------------------
  let refs = [];
  if (input.research !== false && deps.research) {
    try {
      const r = await withTimeout(deps.research(goal, { seed, maxRefs: 3 }), Math.min(25_000, left() * 0.1), "look research");
      refs = r?.refs || [];
    } catch (e) {
      warnings.push(`look research skipped: ${e.message}`);
    }
  }
  let looks = [];
  if (refs.length && deps.readLook) {
    looks = (await Promise.all(refs.slice(0, 3).map(async (ref) => {
      try {
        return await withTimeout(deps.readLook(ref, goal), Math.min(25_000, left() * 0.1), "look read");
      } catch (e) {
        warnings.push(`look unreadable (${str(ref, 60)}): ${e.message}`);
        return null;
      }
    }))).filter(Boolean);
  }
  mark("research");

  // ---- brief -------------------------------------------------------------
  let shots = [];
  try {
    const plan = deps.planFilm || ((g, n, l) => synthesizeShotPlan(g, n, l, { chat: deps.chat }));
    const raw = await withTimeout(plan(goal, numShots, looks), Math.min(45_000, left() * 0.15), "shot plan");
    shots = (raw || []).map((s, i) => normalizeShot(s, i));
  } catch (e) {
    warnings.push(`shot plan failed: ${e.message}`);
    shots = [];
  }
  const variety = enforceShotVariety(shots);
  shots = variety.shots;
  mark("brief");

  // ---- render + verify per shot -----------------------------------------
  // Round-based: every shot renders in PARALLEL (video generation is slow),
  // then each rendered frame is checked against its shot. Failed shots go
  // into the next round with the reviewer's own complaints as corrections.
  const rendered = [];
  const byIndex = new Map();
  let pending = shots.map((shot) => ({ shot, attempt: 0, correction: "", last: null }));

  for (let round = 0; round < maxAttempts && pending.length; round++) {
    const batch = await Promise.all(pending.map(async (item) => {
      const { shot } = item;
      const attempt = item.attempt + 1;
      let result = null;
      try {
        const r = await withTimeout(
          deps.renderShot(shot, { attempt, correction: item.correction }),
          Math.min(150_000, Math.max(10_000, left() - 5_000)),
          `render shot ${shot.index}`
        );
        if (!r?.ok) throw new Error(r?.error || "render returned nothing");
        result = { ...r, attempt };
        // Verify the FRAME, not the request. Video shots are sampled into
        // frames first; still shots are their own frame.
        if (input.verify === false || !deps.verifyFrame) {
          return { item: { ...item, attempt, last: result }, final: { ...shot, ...result, attempts: attempt }, retry: false };
        }
        const frames = deps.framesOf ? await deps.framesOf(result, shot) : [];
        const frame = frames?.[0] || (result.kind === "image" ? (result.url || null) : null);
        if (!frame) {
          return { item: { ...item, attempt, last: result }, final: { ...shot, ...result, attempts: attempt }, retry: false };
        }
        const verdict = await withTimeout(
          deps.verifyFrame(frame, shot),
          Math.min(30_000, Math.max(5_000, left() - 2_000)),
          `verify shot ${shot.index}`
        );
        result.verification = verdict;
        const retry = !verdict.pass && attempt < maxAttempts && (verdict.issues || []).length > 0;
        const correction = (verdict.issues || []).slice(0, 3).join("; ");
        return {
          item: { shot, attempt, correction, last: result },
          final: { ...shot, ...result, attempts: attempt },
          retry,
        };
      } catch (e) {
        warnings.push(`shot ${shot.index} attempt ${attempt} failed: ${e.message}`);
        return {
          item: { ...item, attempt, last: { ok: false, error: e.message } },
          final: { ...shot, ok: false, error: e.message, attempts: attempt },
          retry: false,
        };
      }
    }));

    for (const b of batch) byIndex.set(b.final.index, b.final);
    pending = batch.filter((b) => b.retry).map((b) => b.item);
  }
  for (const shot of shots) {
    if (byIndex.has(shot.index)) rendered.push(byIndex.get(shot.index));
  }
  mark("render");

  return {
    ok: rendered.some((r) => r.ok),
    goal,
    seed,
    shots: rendered,
    looks: looks.map((l) => ({ ref: l.ref, look: l.look, palette: l.palette })),
    shot_variety_changes: variety.changed,
    stages,
    warnings,
  };
}

export { LOOK_TAXONOMY, MOVES, ANGLES, LENSES, APERTURES, structuralOnly };
