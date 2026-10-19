// ============================================================================
// FRIT SUBTITLES — real timing helpers (audit F-13)
// ----------------------------------------------------------------------------
// Before this module the server fabricated every caption timestamp with a
// fixed "0.4 seconds per word" guess, so burned-in subtitles drifted out of
// sync with the audio and SRT files were useless outside the app.
//
// These helpers are PURE (no network, no ffmpeg, no db) so they are unit
// tested in test/subtitles.test.mjs. Three tiers of accuracy, best first:
//   1. word timestamps   -> cues break on real word boundaries
//   2. segment timestamps-> cues follow the provider's own segmentation
//   3. duration estimate -> cues are distributed over the REAL audio duration,
//                           weighted by character length (not a fixed guess)
// ============================================================================

export const MAX_CUE_WORDS = 14;
export const MAX_CUE_CHARS = 84;
export const MAX_CUE_SECONDS = 6;
// A cue must not run into the next one — captions reading simultaneously
// look broken. This is the enforced gap between consecutive cues.
export const CUE_GAP_SECONDS = 0.04;

const SENTENCE_END = /[.!?…]["')\]]?$/;

/**
 * Group word-level timestamps into readable cues.
 * @param {Array<{word?:string,text?:string,start?:number,end?:number}>} words
 * @param {{maxWords?:number,maxChars?:number,maxSeconds?:number}} [opts]
 * @returns {Array<{start:number,end:number,text:string}>}
 */
export function buildCuesFromWords(words, opts = {}) {
  const maxWords = opts.maxWords ?? MAX_CUE_WORDS;
  const maxChars = opts.maxChars ?? MAX_CUE_CHARS;
  const maxSeconds = opts.maxSeconds ?? MAX_CUE_SECONDS;
  if (!Array.isArray(words) || !words.length) return [];

  const cues = [];
  let cur = null;

  const flush = () => {
    if (!cur) return;
    const text = cur.words.join(" ").trim();
    if (text) cues.push({ start: cur.start, end: cur.end, text });
    cur = null;
  };

  for (const w of words) {
    if (!w || typeof w !== "object") continue;
    const text = String(w.word ?? w.text ?? "").trim();
    if (!text) continue;
    const start = Number(w.start);
    const end = Number(w.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) continue;

    // A word that would push the current cue past the limits starts a NEW cue
    // instead of extending it — otherwise a long word could push the cue over
    // maxSeconds (and past the next word's start).
    if (cur) {
      const joined = cur.words.join(" ");
      const wouldOverflow =
        cur.words.length + 1 > maxWords ||
        (joined + " " + text).length > maxChars ||
        Math.max(cur.end, end) - cur.start > maxSeconds;
      if (wouldOverflow) flush();
    }

    if (!cur) {
      cur = { start, end, words: [] };
    } else {
      cur.start = Math.min(cur.start, start);
      cur.end = Math.max(cur.end, end);
    }
    cur.words.push(text);

    // A sentence ending always closes the cue.
    if (SENTENCE_END.test(text)) flush();
  }
  flush();
  return normalizeCues(cues);
}

/**
 * Fallback tier: provider gave us segment timestamps but no word timings.
 * @param {Array<{start?:number,end?:number,text?:string}>} segments
 */
export function buildCuesFromSegments(segments, opts = {}) {
  if (!Array.isArray(segments) || !segments.length) return [];
  const maxChars = opts.maxChars ?? MAX_CUE_CHARS;
  const cues = [];
  for (const s of segments) {
    if (!s || typeof s !== "object") continue;
    const text = String(s.text ?? "").trim();
    const start = Number(s.start);
    const end = Number(s.end);
    if (!text || !Number.isFinite(start) || !Number.isFinite(end)) continue;
    // Long provider segments get split on word boundaries so a single 40-word
    // blob doesn't become one unreadable caption.
    const parts = splitLongText(text, maxChars);
    const span = Math.max(0, end - start);
    const total = parts.reduce((n, p) => n + p.length, 0) || 1;
    let cursor = start;
    for (const p of parts) {
      const dur = (p.length / total) * span;
      cues.push({ start: cursor, end: cursor + dur, text: p });
      cursor += dur;
    }
  }
  return normalizeCues(cues);
}

/**
 * Last-resort tier: we only know the text and how long the audio is.
 * Cue lengths are weighted by character count, so a long sentence occupies
 * proportionally more screen time than a short one. This is still an estimate,
 * but it lands on the real audio length instead of drifting forever.
 * @param {string} text
 * @param {number} durationSec real audio duration in seconds
 */
export function buildCuesFromText(text, durationSec, opts = {}) {
  const maxWords = opts.maxWords ?? MAX_CUE_WORDS;
  const maxChars = opts.maxChars ?? MAX_CUE_CHARS;
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return [];
  const duration = Number(durationSec);
  if (!Number.isFinite(duration) || duration <= 0) return [];

  const words = clean.split(" ");
  const groups = [];
  let cur = [];
  for (const w of words) {
    cur.push(w);
    const joined = cur.join(" ");
    if (cur.length >= maxWords || joined.length > maxChars || SENTENCE_END.test(w)) {
      groups.push(cur);
      cur = [];
    }
  }
  if (cur.length) groups.push(cur);

  const weights = groups.map((g) => g.join(" ").length);
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  let cursor = 0;
  const cues = [];
  for (let i = 0; i < groups.length; i++) {
    const dur = (weights[i] / total) * duration;
    const isLast = i === groups.length - 1;
    // The last cue ends exactly at the audio end, so captions cover the whole
    // clip instead of stopping short of the real duration.
    const end = isLast ? duration : cursor + dur;
    cues.push({ start: cursor, end, text: groups[i].join(" ") });
    cursor = end;
  }
  return normalizeCues(cues);
}

/**
 * Enforce the invariants every consumer (ffmpeg burn-in, SRT export, the
 * Android client) relies on: finite numbers, start < end, monotonic and
 * non-overlapping, and a visible gap between consecutive cues.
 */
export function normalizeCues(cues) {
  if (!Array.isArray(cues) || !cues.length) return [];
  const out = [];
  for (const c of cues) {
    if (!c) continue;
    const text = String(c.text ?? "").replace(/\s+/g, " ").trim();
    let start = Number(c.start);
    let end = Number(c.end);
    if (!text) continue;
    if (!Number.isFinite(start)) start = 0;
    if (!Number.isFinite(end) || end <= start) end = start + 0.2;
    out.push({ start, end, text });
  }
  out.sort((a, b) => a.start - b.start);

  for (let i = 1; i < out.length; i++) {
    const prev = out[i - 1];
    const cur = out[i];
    if (cur.start < prev.end + CUE_GAP_SECONDS) {
      // Overlap: shorten the previous cue so it ends CUE_GAP_SECONDS before
      // the current one starts. If that would collapse the previous cue
      // entirely, push the current cue forward instead.
      const trimmedEnd = cur.start - CUE_GAP_SECONDS;
      if (trimmedEnd > prev.start) {
        prev.end = trimmedEnd;
      } else {
        cur.start = prev.end + CUE_GAP_SECONDS;
        if (cur.end <= cur.start) cur.end = cur.start + 0.2;
      }
    }
  }
  return out;
}

/** Split a blob of text into caption-sized chunks on word boundaries. */
export function splitLongText(text, maxChars = MAX_CUE_CHARS) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return [];
  if (clean.length <= maxChars) return [clean];
  const words = clean.split(" ");
  const out = [];
  let cur = [];
  for (const w of words) {
    cur.push(w);
    if (cur.join(" ").length > maxChars) {
      cur.pop();
      if (cur.length) out.push(cur.join(" "));
      cur = [w];
    }
  }
  if (cur.length) out.push(cur.join(" "));
  return out.length ? out : [clean];
}

/** SRT timestamp: HH:MM:SS,mmm */
export function srtTimestamp(seconds) {
  const t = Math.max(0, Number(seconds) || 0);
  const ms = Math.round(t * 1000);
  const h = String(Math.floor(ms / 3600000)).padStart(2, "0");
  const m = String(Math.floor((ms % 3600000) / 60000)).padStart(2, "0");
  const s = String(Math.floor((ms % 60000) / 1000)).padStart(2, "0");
  const milli = String(ms % 1000).padStart(3, "0");
  return `${h}:${m}:${s},${milli}`;
}

/** Full SRT document for a cue list. */
export function toSrt(cues) {
  const list = normalizeCues(cues);
  return list
    .map((c, i) => `${i + 1}\n${srtTimestamp(c.start)} --> ${srtTimestamp(c.end)}\n${c.text}`)
    .join("\n\n") + (list.length ? "\n" : "");
}

/**
 * Pick the best available timing source and build cues from it.
 * @param {{words?:Array,segments?:Array,text?:string,duration?:number}} payload
 * @returns {{cues:Array,timing:"word"|"segment"|"estimated"}}
 */
export function buildCues(payload = {}) {
  const words = Array.isArray(payload.words) ? payload.words : [];
  if (words.length) {
    const cues = buildCuesFromWords(words);
    if (cues.length) return { cues, timing: "word" };
  }
  const segments = Array.isArray(payload.segments) ? payload.segments : [];
  if (segments.length) {
    const cues = buildCuesFromSegments(segments);
    if (cues.length) return { cues, timing: "segment" };
  }
  const cues = buildCuesFromText(payload.text, payload.duration);
  return { cues, timing: cues.length ? "estimated" : "none" };
}
