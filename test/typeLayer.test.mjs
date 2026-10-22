// ============================================================================
// Type layer tests — real typography set from the brief
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";

import {
  planTypeLayout, briefToTextSpecs, applyTypeLayer, pickTypeColor,
  contrastRatio, luminance, applyCase, wrapText, fontStack, escapeXml,
  estimateWidth, fitTypeSize, onAccentColor,
} from "../src/creative/typeLayer.js";

const brief = {
  canvas: { aspect_ratio: "4:5" },
  layout: { grid: "rule-of-thirds", margins_pct: 10, alignment: "left", focal: "left third" },
  palette: [
    { hex: "#0E3B2E", role: "dominant", share_pct: 60 },
    { hex: "#F4EFE4", role: "secondary", share_pct: 30 },
    { hex: "#D4A574", role: "accent", share_pct: 10 },
  ],
  typography: [
    { role: "headline", font: "grotesk sans", size_px: 96, weight: 800, tracking: -1, case: "UPPER" },
    { role: "subhead", font: "humanist sans", size_px: 30, weight: 500, tracking: 0, case: "SENTENCE" },
    { role: "body", font: "humanist sans", size_px: 16, weight: 400, tracking: 0, case: "SENTENCE" },
    { role: "cta", font: "semibold sans", size_px: 22, weight: 700, tracking: 1, case: "UPPER" },
  ],
  copy: { headline: "COLD BREW", subhead: "now pouring", body: "small batch, brewed daily", cta: "Order now" },
  background: { description: "cream paper", color_hex: "#F4EFE4" },
};

test("luminance / contrastRatio: readable maths", () => {
  assert.ok(luminance("#FFFFFF") > 0.9);
  assert.ok(luminance("#000000") < 0.05);
  assert.ok(contrastRatio("#000000", "#FFFFFF") > 15);
});

test("pickTypeColor: picks the palette colour with the best contrast on the background", () => {
  assert.equal(pickTypeColor([{ hex: "#000000" }, { hex: "#FFFFFF" }], "#FFFFFF"), "#000000");
  assert.equal(pickTypeColor([{ hex: "#FFFFFF" }, { hex: "#000000" }], "#000000"), "#FFFFFF");
});

test("pickTypeColor: falls back to black/white when nothing in the palette is readable", () => {
  assert.equal(pickTypeColor([{ hex: "#EEEEEE" }], "#F0F0F0"), "#111111");
  assert.equal(pickTypeColor([{ hex: "#222222" }], "#111111"), "#FFFFFF");
});

test("onAccentColor: the type on the badge is judged against the BADGE, not against white", () => {
  // A dark badge must carry light type, and a bright badge dark type.
  assert.equal(onAccentColor("#223E1E"), "#FFFFFF");
  assert.equal(onAccentColor("#73FF00"), "#111111");
});

test("applyCase: UPPER / LOWER / TITLE / SENTENCE", () => {
  assert.equal(applyCase("cold brew", "UPPER"), "COLD BREW");
  assert.equal(applyCase("COLD BREW", "LOWER"), "cold brew");
  assert.equal(applyCase("cold brew", "TITLE"), "Cold Brew");
  assert.equal(applyCase("cold brew", "SENTENCE"), "cold brew");
});

test("wrapText: wraps on width and caps the line count", () => {
  const lines = wrapText("one two three four five six seven eight nine ten", 40, 200, 700, 3);
  assert.ok(lines.length <= 3);
  assert.ok(lines[0].includes("one"));
});

test("fontStack: maps a font name onto a real stack", () => {
  assert.ok(/DejaVu Serif/.test(fontStack("Playfair Display")));
  assert.ok(/DejaVu Sans Mono/.test(fontStack("Roboto Mono")));
  assert.ok(/DejaVu Sans/.test(fontStack("Inter")));
});

test("escapeXml: neutralises the characters that break an SVG", () => {
  assert.equal(escapeXml(`a&b<c>"d"`), "a&amp;b&lt;c&gt;&quot;d&quot;");
});

test("planTypeLayout: orders the type, honours the margins and the alignment", () => {
  const plan = planTypeLayout(brief, { width: 1000, height: 1250 });
  assert.equal(plan.margin, 100);
  assert.equal(plan.anchor, "start");
  const roles = plan.items.map((i) => i.role);
  assert.equal(roles[0], "headline");
  assert.ok(roles.includes("cta"));
  const headline = plan.items[0];
  assert.ok(headline.size > plan.items[1].size, "the headline must be the largest");
  assert.equal(headline.x, 100);
  const cta = plan.cta;
  assert.ok(cta.box, "the CTA must get a badge box");
  assert.ok(cta.box.w > 0 && cta.box.h > 0);
});

test("planTypeLayout: centre alignment moves the anchor and the x", () => {
  const plan = planTypeLayout({ ...brief, layout: { ...brief.layout, alignment: "center" } }, { width: 1000, height: 1250 });
  assert.equal(plan.anchor, "middle");
  assert.equal(plan.items[0].x, 500);
});

test("planTypeLayout: right alignment anchors to the right edge", () => {
  const plan = planTypeLayout({ ...brief, layout: { ...brief.layout, alignment: "right" } }, { width: 1000, height: 1250 });
  assert.equal(plan.anchor, "end");
  assert.equal(plan.items[0].x, 900);
});

test("briefToTextSpecs: produces Recraft-shaped type payloads", () => {
  const specs = briefToTextSpecs(brief, { size: { width: 1024, height: 1280 } });
  assert.ok(specs.length >= 3);
  const headline = specs.find((s) => s.text.includes("COLD BREW"));
  assert.ok(headline, "the headline must be typeset");
  // The brief's px are relative to a 1080 canvas; the Recraft canvas is 1024.
  assert.ok(headline.font_size > 85 && headline.font_size < 100, `scaled headline size: ${headline.font_size}`);
  assert.equal(headline.font_weight, "800");
  assert.ok(headline.x >= 0 && headline.x <= 100);
  assert.ok(headline.y >= 0 && headline.y <= 100);
  assert.ok(specs.every((s) => typeof s.text_color === "string" && s.text_color.startsWith("#")));
});

test("briefToTextSpecs: no copy means no specs", () => {
  const specs = briefToTextSpecs({ ...brief, copy: {} }, { size: { width: 1024, height: 1024 } });
  assert.equal(specs.length, 0);
});

test("planTypeLayout: a headline that would overflow is shrunk to FIT the column", () => {
  const long = { ...brief, copy: { ...brief.copy, headline: "AN EXTREMELY LONG HEADLINE THAT CANNOT POSSIBLY FIT" } };
  const plan = planTypeLayout(long, { width: 1000, height: 1250 });
  const h = plan.items[0];
  const widest = h.lines.reduce((m, l) => Math.max(m, estimateWidth(l, h.size, h.weight)), 0);
  assert.ok(widest <= plan.maxWidth, `the headline must fit: ${widest} > ${plan.maxWidth}`);
  assert.ok(h.size < 96, "the size must shrink rather than overflow");
});

test("planTypeLayout: the CTA badge is wide enough for its own text", () => {
  const plan = planTypeLayout(brief, { width: 1000, height: 1250 });
  const cta = plan.cta;
  const textW = estimateWidth(cta.text, cta.size, cta.weight);
  assert.ok(cta.box.w >= textW, `the badge must contain the text: ${cta.box.w} < ${textW}`);
  assert.ok(cta.box.x + cta.box.w <= 1000, "the badge must stay inside the canvas");
});

test("planTypeLayout: right-aligned copy is FLUSH right, badge and label alike", () => {
  const plan = planTypeLayout({ ...brief, layout: { ...brief.layout, alignment: "right" } }, { width: 1000, height: 1250 });
  const cta = plan.cta;
  assert.equal(cta.anchor, "end", "the label must be right-anchored inside the badge");
  assert.equal(cta.box.x + cta.box.w, 1000 - 100, "the badge must be flush with the right margin");
  assert.ok(cta.x <= cta.box.x + cta.box.w, "the label must end inside the badge");
  assert.equal(plan.items[0].anchor, "end");
  assert.equal(plan.items[0].x, 1000 - 100);
});

test("planTypeLayout: left-aligned copy is flush left", () => {
  const plan = planTypeLayout(brief, { width: 1000, height: 1250 });
  assert.equal(plan.items[0].x, 100);
  assert.equal(plan.cta.anchor, "start");
  assert.equal(plan.cta.box.x, 100);
});

test("applyTypeLayer: composites real glyphs over the artwork (or says why not)", async () => {
  const sharp = (await import("sharp")).default;
  const artwork = await sharp({
    create: { width: 1000, height: 1250, channels: 3, background: { r: 244, g: 239, b: 228 } },
  }).png().toBuffer();

  const out = await applyTypeLayer(artwork, brief);
  assert.ok(typeof out.applied === "boolean");
  if (out.applied) {
    assert.ok(out.buffer.length > 0);
    const { data, info } = await sharp(out.buffer).raw().toBuffer({ resolveWithObject: true });
    let dark = 0;
    for (let i = 0; i < data.length; i += info.channels) {
      if (data[i] < 120 && data[i + 1] < 120 && data[i + 2] < 120) dark++;
    }
    assert.ok(dark > 200, `type must actually be drawn (dark pixels: ${dark})`);
  } else {
    assert.ok(out.reason && out.reason.length > 5, "a skipped type layer must explain itself");
  }
});

test("applyTypeLayer: a brief with no copy leaves the artwork untouched", async () => {
  const sharp = (await import("sharp")).default;
  const artwork = await sharp({ create: { width: 400, height: 400, channels: 3, background: { r: 10, g: 20, b: 30 } } }).png().toBuffer();
  const out = await applyTypeLayer(artwork, { ...brief, copy: {} });
  assert.equal(out.applied, false);
  assert.equal(out.buffer, artwork);
});

test("applyTypeLayer: a scrim band is laid over a BUSY backdrop (a photo or pattern)", async () => {
  const sharp = (await import("sharp")).default;
  // High-contrast stripes behind the copy: the type cannot be read reliably
  // on it, so the layer must lay a scrim band.
  const stripe = Buffer.from(
    `<svg width="1000" height="1250" xmlns="http://www.w3.org/2000/svg">` +
    Array.from({ length: 40 }, (_, i) => `<rect y="${i * 32}" width="1000" height="32" fill="${i % 2 ? "#ffffff" : "#111111"}"/>`).join("") +
    `</svg>`,
  );
  const artwork = await sharp(stripe).png().toBuffer();
  const out = await applyTypeLayer(artwork, brief);
  assert.equal(out.applied, true);
  assert.equal(out.scrim, true, "a busy backdrop must get a scrim");
});

test("applyTypeLayer: a flat, readable backdrop gets NO scrim", async () => {
  const sharp = (await import("sharp")).default;
  const artwork = await sharp({
    create: { width: 1000, height: 1250, channels: 3, background: { r: 14, g: 59, b: 46 } },
  }).png().toBuffer();
  const out = await applyTypeLayer(artwork, {
    ...brief,
    palette: [{ hex: "#F4EFE4", role: "secondary", share_pct: 100 }],
    background: { description: "deep green", color_hex: "#0E3B2E" },
  });
  assert.equal(out.applied, true);
  assert.equal(out.scrim, false, "cream on deep green is already readable");
});
