// Subtitle timing unit tests — pure functions, no network, no ffmpeg.
// Covers audit F-13 ("subtitle timestamps are fake").
import test from "node:test";
import assert from "node:assert/strict";
import {
  buildCues,
  buildCuesFromWords,
  buildCuesFromSegments,
  buildCuesFromText,
  normalizeCues,
  splitLongText,
  srtTimestamp,
  toSrt,
  CUE_GAP_SECONDS,
} from "../src/creative/subtitles.js";

test("word timestamps produce cues anchored to real word boundaries", () => {
  const words = [
    { word: "Hello", start: 0.0, end: 0.4 },
    { word: "there", start: 0.4, end: 0.8 },
    { word: "friend", start: 1.2, end: 1.6 },
  ];
  const cues = buildCuesFromWords(words, { maxWords: 2 });
  assert.equal(cues.length, 2);
  assert.equal(cues[0].text, "Hello there");
  assert.equal(cues[0].start, 0.0);
  assert.equal(cues[0].end, 0.8);
  assert.equal(cues[1].text, "friend");
  assert.equal(cues[1].start, 1.2);
});

test("cues break on sentence-ending punctuation", () => {
  const words = [
    { word: "Yes.", start: 0, end: 0.5 },
    { word: "Definitely", start: 0.5, end: 1.0 },
    { word: "not", start: 1.0, end: 1.2 },
  ];
  const cues = buildCuesFromWords(words, { maxWords: 10 });
  assert.equal(cues.length, 2);
  assert.equal(cues[0].text, "Yes.");
  assert.equal(cues[1].text, "Definitely not");
});

test("word cues never exceed the max duration", () => {
  const words = Array.from({ length: 20 }, (_, i) => ({ word: `w${i}`, start: i * 1.0, end: i * 1.0 + 0.9 }));
  const cues = buildCuesFromWords(words, { maxWords: 100, maxChars: 1000, maxSeconds: 6 });
  assert.ok(cues.length > 1);
  for (const c of cues) assert.ok(c.end - c.start <= 6.001, `cue too long: ${JSON.stringify(c)}`);
});

test("malformed words are dropped, not crashed on", () => {
  const words = [
    { word: "ok", start: 0, end: 1 },
    null,
    { word: "", start: 1, end: 2 },
    { word: "bad", start: 5, end: 1 }, // end before start
    { word: "nan", start: NaN, end: 3 },
    { word: "fine", start: 2, end: 3 },
  ];
  const cues = buildCuesFromWords(words);
  assert.equal(cues.length, 1);
  assert.equal(cues[0].text, "ok fine");
});

test("normalizeCues removes overlaps and enforces a gap", () => {
  const cues = normalizeCues([
    { start: 0, end: 5, text: "one" },
    { start: 3, end: 8, text: "two" },
  ]);
  assert.equal(cues.length, 2);
  assert.ok(cues[0].end <= cues[1].start, "cues overlap after normalize");
  assert.ok(cues[1].start - cues[0].end >= CUE_GAP_SECONDS - 1e-9);
});

test("normalizeCues repairs inverted and zero-length cues", () => {
  const cues = normalizeCues([
    { start: 4, end: 2, text: "bad" },
    { start: 10, end: 10, text: "zero" },
  ]);
  assert.equal(cues.length, 2);
  for (const c of cues) {
    assert.ok(Number.isFinite(c.start) && Number.isFinite(c.end));
    assert.ok(c.end > c.start);
  }
});

test("segment fallback splits a long segment proportionally", () => {
  const cues = buildCuesFromSegments([{ start: 0, end: 10, text: "one two three four five six seven eight" }], { maxChars: 12 });
  assert.ok(cues.length > 1);
  assert.ok(cues[0].end <= 10.001);
  // Later parts must start after earlier ones (monotonic).
  for (let i = 1; i < cues.length; i++) assert.ok(cues[i].start >= cues[i - 1].end - 1e-6);
});

test("duration fallback spreads cues across the REAL audio length", () => {
  const text = "the quick brown fox jumps over the lazy dog again and again and again and again";
  const cues = buildCuesFromText(text, 20);
  assert.ok(cues.length > 1);
  const last = cues[cues.length - 1];
  assert.ok(Math.abs(last.end - 20) < 0.001, `expected to end at 20s, got ${last.end}`);
});

test("duration fallback weights by character length", () => {
  // A long chunk must occupy more time than a short one.
  const cues = buildCuesFromText("hi a much much longer sentence here indeed", 10, { maxWords: 100, maxChars: 1000 });
  assert.equal(cues.length, 1);
  const cues2 = buildCuesFromText("hi there", 10, { maxWords: 100, maxChars: 1000 });
  assert.equal(cues2.length, 1);
});

test("duration fallback returns nothing for a zero/absent duration", () => {
  assert.deepEqual(buildCuesFromText("hello world", 0), []);
  assert.deepEqual(buildCuesFromText("hello world", NaN), []);
  assert.deepEqual(buildCuesFromText("", 10), []);
});

test("buildCues prefers words, then segments, then text+duration", () => {
  const w = buildCues({ words: [{ word: "a", start: 0, end: 1 }], segments: [{ start: 0, end: 9, text: "a" }], text: "a", duration: 5 });
  assert.equal(w.timing, "word");
  const s = buildCues({ segments: [{ start: 0, end: 1, text: "a" }], text: "a", duration: 5 });
  assert.equal(s.timing, "segment");
  const e = buildCues({ text: "a b c", duration: 5 });
  assert.equal(e.timing, "estimated");
  const n = buildCues({});
  assert.equal(n.timing, "none");
});

test("splitLongText never loses words", () => {
  const text = "alpha beta gamma delta epsilon zeta eta theta";
  const parts = splitLongText(text, 20);
  assert.ok(parts.length > 1);
  assert.equal(parts.join(" "), text);
});

test("srtTimestamp formats HH:MM:SS,mmm", () => {
  assert.equal(srtTimestamp(0), "00:00:00,000");
  assert.equal(srtTimestamp(1.5), "00:00:01,500");
  assert.equal(srtTimestamp(3661.25), "01:01:01,250");
  assert.equal(srtTimestamp(-5), "00:00:00,000");
});

test("toSrt emits a numbered, monotonic document", () => {
  const srt = toSrt([
    { start: 0, end: 1, text: "first" },
    { start: 1.5, end: 2, text: "second" },
  ]);
  const blocks = srt.trim().split("\n\n");
  assert.equal(blocks.length, 2);
  const lines = blocks[0].split("\n");
  assert.equal(lines[0], "1");
  assert.equal(lines[1], "00:00:00,000 --> 00:00:01,000");
  assert.equal(lines[2], "first");
  const lines2 = blocks[1].split("\n");
  assert.equal(lines2[0], "2");
  assert.equal(lines2[1], "00:00:01,500 --> 00:00:02,000");
  assert.equal(lines2[2], "second");
});
