// ============================================================================
// TYPE LAYER — the brief rendered as REAL type, on top of the artwork
// ----------------------------------------------------------------------------
// Why: the #1 reason a piece "looks AI generated" is not the composition, it
// is the TYPOGRAPHY. Image models still mangle letterforms, weights and
// spacing. So the renderer paints the ARTWORK and this module sets the
// TYPE — real glyphs, real sizes, real tracking, from the brief.
//
// The brief is the single source of truth for both layers:
//   brief.typography -> font / size / weight / tracking / case
//   brief.copy       -> the exact strings
//   brief.layout     -> alignment, margins, reading order
//   brief.palette    -> the colours
//
// Two consumers:
//   1. renderTypeLayer()  — sharp/SVG composite (works anywhere sharp runs)
//   2. briefToTextSpecs() — Recraft's native `texts` field (real fonts too)
// ============================================================================

import { hexToRgb } from "./designDNA.js";

// ---------------------------------------------------------------------------
// COLOUR
// ---------------------------------------------------------------------------

export function luminance(hex) {
  const rgb = hexToRgb(hex);
  if (!rgb) return 0.5;
  const f = (c) => {
    const v = c / 255;
    return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * f(rgb.r) + 0.7152 * f(rgb.g) + 0.0722 * f(rgb.b);
}

export function contrastRatio(a, b) {
  const l1 = luminance(a), l2 = luminance(b);
  const hi = Math.max(l1, l2), lo = Math.min(l1, l2);
  return (hi + 0.05) / (lo + 0.05);
}

/** Pick the most legible palette colour against the background. */
export function pickTypeColor(palette, backgroundHex) {
  const bg = backgroundHex || "#FFFFFF";
  const options = (palette || []).map((p) => p.hex).filter(Boolean);
  if (!options.length) return luminance(bg) > 0.5 ? "#111111" : "#FFFFFF";
  let best = options[0];
  let bestRatio = contrastRatio(best, bg);
  for (const hex of options.slice(1)) {
    const ratio = contrastRatio(hex, bg);
    if (ratio > bestRatio) { bestRatio = ratio; best = hex; }
  }
  // Nothing in the palette is readable — fall back to the extreme that works.
  if (bestRatio < 3) return luminance(bg) > 0.45 ? "#111111" : "#FFFFFF";
  return best;
}

// ---------------------------------------------------------------------------
// FONTS
// ---------------------------------------------------------------------------

/**
 * Map a brief font name onto a CSS stack whose FIRST family is likely to be
 * installed. DejaVu is present on the sandbox image; Liberation is the
 * common Linux fallback; the rest are graceful web fallbacks.
 */
export function fontStack(fontName) {
  const f = String(fontName || "").toLowerCase();
  if (/mono|code|terminal/.test(f)) return "'DejaVu Sans Mono','Liberation Mono',Consolas,monospace";
  if (/script|hand|callig|brush|marker/.test(f)) return "'DejaVu Serif','Liberation Serif',cursive";
  if (/serif|roman|garamond|didot|bodoni|playfair|lora|georgia|times|trajan|baskerville/.test(f)) {
    return "'DejaVu Serif','Liberation Serif',Georgia,'Times New Roman',serif";
  }
  if (/grotesk|sans|helvetica|arial|inter|neue|futura|poppins|montserrat|verdana|roboto|open sans/.test(f)) {
    return "'DejaVu Sans','Liberation Sans',Helvetica,Arial,sans-serif";
  }
  if (/display|black|heavy|fat|impact|condensed/.test(f)) {
    return "'DejaVu Sans','Liberation Sans',Impact,'Arial Narrow',sans-serif";
  }
  return "'DejaVu Sans','Liberation Sans',Helvetica,Arial,sans-serif";
}

/** True when the host can actually rasterise SVG text (fontconfig present). */
let _textProbe = null;
export async function probeTextRendering() {
  if (_textProbe !== null) return _textProbe;
  try {
    const sharp = (await import("sharp")).default;
    const svg = `<svg width="220" height="90" xmlns="http://www.w3.org/2000/svg"><rect width="220" height="90" fill="#000"/><text x="10" y="62" font-family="DejaVu Sans, sans-serif" font-size="52" font-weight="700" fill="#fff">MMMM</text></svg>`;
    const { data, info } = await sharp(Buffer.from(svg)).raw().toBuffer({ resolveWithObject: true });
    let bright = 0;
    for (let i = 0; i < data.length; i += info.channels) {
      if (data[i] > 180 && data[i + 1] > 180 && data[i + 2] > 180) bright++;
    }
    _textProbe = bright > 30;
  } catch {
    _textProbe = false;
  }
  return _textProbe;
}

// ---------------------------------------------------------------------------
// TEXT HELPERS
// ---------------------------------------------------------------------------

export function escapeXml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export function applyCase(text, mode) {
  const t = String(text ?? "");
  const m = String(mode || "").toUpperCase();
  if (m === "UPPER") return t.toUpperCase();
  if (m === "LOWER") return t.toLowerCase();
  if (m === "TITLE") return t.replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase());
  return t;
}

/**
 * Clean a copy string before it is typeset.
 *
 * Two contaminations come straight from the model and are very visible on
 * the finished poster:
 *   * PLACEHOLDERS — "Let's celebrate [Name] in style!" The type layer sets
 *     the copy VERBATIM, so the bracket went straight onto the artwork.
 *   * QUOTES — '"HAPPY BIRTHDAY"' — the model quotes the copy in its JSON.
 *
 * A string carrying a placeholder is DROPPED (it is worse than absent: the
 * user did not ask for it and it cannot be filled in).
 */
export function cleanCopy(text) {
  const raw = String(text ?? "").trim();
  if (!raw) return "";
  if (/\[[^\]]{1,60}\]|\{[^\}]{1,60}\}|<[^>]{1,60}>|\b(?:lorem ipsum|xxx+)\b/i.test(raw)) return "";
  // Strip one layer of matching surrounding quotes.
  const unquoted = raw.replace(/^["'“”‘’`\s]+/, "").replace(/["'“”‘’`\s]+$/, "");
  return unquoted.trim();
}

/** Rough advance width per character, in em.
 *
 *  The estimate must err on the SAFE side: a headline that overflows the
 *  margin is the single most common "AI poster" tell, and a clipped
 *  headline is worse than a slightly small one. Uppercase is wider than
 *  mixed case, and a bold display face is wider again. */
export function estimateWidth(text, fontSizePx, weight = 400) {
  const s = String(text || "");
  const bold = Number(weight) >= 600;
  let factor = bold ? 0.66 : 0.55;
  const letters = s.replace(/[^A-Za-z]/g, "");
  if (letters.length > 1) {
    const upper = (letters.match(/[A-Z]/g) || []).length / letters.length;
    if (upper > 0.7) factor *= 1.1;      // all-caps runs wide
    else if (upper > 0.2) factor *= 1.04;
  }
  return s.length * fontSizePx * factor;
}

/** Greedy wrap to a pixel width. */
export function wrapText(text, fontSizePx, maxWidthPx, weight = 400, maxLines = 4) {
  const words = String(text || "").split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const lines = [];
  let line = "";
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (estimateWidth(candidate, fontSizePx, weight) <= maxWidthPx || !line) {
      line = candidate;
    } else {
      lines.push(line);
      line = word;
      if (lines.length >= maxLines) break;
    }
  }
  if (lines.length < maxLines && line) lines.push(line);
  return lines;
}

/** Shrink a type block until its widest line fits. A designer sets the type
 *  to FIT the space; a naive renderer clips it. Returns the fitted size. */
export function fitTypeSize(text, sizePx, maxWidthPx, weight = 400, tracking = 0, floor = 0.6) {
  let s = sizePx;
  const trackingPx = Math.abs(Number(tracking) || 0) * String(text || "").length;
  for (let i = 0; i < 8; i++) {
    const w = estimateWidth(text, s, weight) + trackingPx;
    if (w <= maxWidthPx || s <= sizePx * floor) break;
    s = Math.floor(s * 0.93);
  }
  return s;
}

// ---------------------------------------------------------------------------
// LAYOUT PLAN — one plan, two renderers (SVG + Recraft)
// ---------------------------------------------------------------------------

/**
 * Turn a brief into a positioned type plan. Coordinates are in pixels for a
 * canvas of `width` x `height`.
 */
export function planTypeLayout(brief, { width, height, baseWidth = 1080 } = {}) {
  const b = brief || {};
  const scale = width / baseWidth;
  const marginPct = Number(b.layout?.margins_pct ?? 8);
  const margin = Math.round((marginPct / 100) * width);
  const alignment = String(b.layout?.alignment || "left").toLowerCase();
  const anchor = alignment === "center" ? "middle" : alignment === "right" ? "end" : "start";
  const x = alignment === "center" ? Math.round(width / 2) : alignment === "right" ? width - margin : margin;
  // 3% safety: the width estimate is an estimate.
  const maxWidth = Math.round((width - margin * 2) * 0.97);

  const typo = b.typography || [];
  const roleFor = (role) => typo.find((t) => t.role === role) || {};
  const copy = b.copy || {};

  const headline = roleFor("headline");
  const subhead = roleFor("subhead");
  const body = roleFor("body");
  const cta = roleFor("cta");

  const headlineSize = Math.max(24, Math.round((headline.size_px || 96) * scale));
  const subheadSize = Math.max(12, Math.round((subhead.size_px || 30) * scale));
  const bodySize = Math.max(10, Math.round((body.size_px || 16) * scale));
  const ctaSize = Math.max(11, Math.round((cta.size_px || 22) * scale));

  const items = [];
  let y = margin + headlineSize * 0.82;

  const push = (text, spec, size, { gap = 0, lines = 3, weight, tracking, cased } = {}) => {
    if (!text) return;
    const t = applyCase(cleanCopy(text), cased ?? spec.case);
    if (!t) return;
    const w = weight ?? spec.weight ?? 400;
    const tr = tracking ?? spec.tracking ?? 0;
    // Fit the block to the column before it is set: a headline that overflows
    // the margin is the single most common "AI poster" tell.
    let fitted = size;
    let wrapped = wrapText(t, fitted, maxWidth, w, lines);
    for (let i = 0; i < 6; i++) {
      const widest = wrapped.reduce((m, l) => Math.max(m, estimateWidth(l, fitted, w) + Math.abs(tr) * l.length), 0);
      if (widest <= maxWidth || fitted <= size * 0.6) break;
      fitted = Math.floor(fitted * 0.93);
      wrapped = wrapText(t, fitted, maxWidth, w, lines);
    }
    if (!wrapped.length) return;
    // Leading: display type is set solid, running copy needs air. 106% of the
    // font size made the body lines overlap — the reviewer called it
    // "garbled and illegible".
    const leading = spec.role === "headline" ? 1.04 : 1.28;
    items.push({
      role: spec.role,
      text: t,
      lines: wrapped,
      size: fitted,
      weight: w,
      tracking: tr,
      font: spec.font || "sans-serif",
      x,
      y,
      anchor,
      align: alignment,
      width: maxWidth,
      leading,
    });
    y += wrapped.length * fitted * leading + gap;
  };

  push(copy.headline, { ...headline, role: "headline" }, headlineSize, { gap: Math.round(headlineSize * 0.18), lines: 3 });
  push(copy.subhead, { ...subhead, role: "subhead" }, subheadSize, { gap: Math.round(subheadSize * 0.5), lines: 2 });
  push(copy.body, { ...body, role: "body" }, bodySize, { gap: 0, lines: 4 });

  // CTA sits at the bottom, as a badge.
  let ctaItem = null;
  if (copy.cta) {
    const ctaText = applyCase(cleanCopy(copy.cta), cta.case);
    if (ctaText) {
      const ctaFitted = fitTypeSize(ctaText, ctaSize, maxWidth, cta.weight || 700, cta.tracking || 0);
      const textW = estimateWidth(ctaText, ctaFitted, cta.weight || 700) + Math.abs(cta.tracking || 0) * ctaText.length;
      const pad = Math.round(ctaFitted * 0.8);
      const w = Math.round(textW + pad * 2);
      const h = Math.round(ctaFitted * 2);
      const cx = alignment === "center" ? Math.round(width / 2 - w / 2) : alignment === "right" ? width - margin - w : margin;
      const cy = Math.max(0, height - margin - h);
      // The badge is flush with the margin AND the type inside it is flush
      // with the badge's own edge — a centred label in a right-hand badge
      // reads as "not aligned right".
      const ctaAnchor = alignment === "center" ? "middle" : alignment === "right" ? "end" : "start";
      const ctaX = alignment === "center" ? cx + w / 2 : alignment === "right" ? cx + w - pad : cx + pad;
      ctaItem = {
        role: "cta",
        text: ctaText,
        lines: [ctaText],
        size: ctaFitted,
        weight: cta.weight || 700,
        tracking: cta.tracking || 0,
        font: cta.font || "sans-serif",
        x: ctaX,
        y: cy + h * 0.5 + ctaFitted * 0.34,
        anchor: ctaAnchor,
        box: { x: cx, y: cy, w, h, r: Math.round(h / 2) },
      };
      items.push(ctaItem);
    }
  }

  return {
    width, height, margin, alignment, anchor, maxWidth,
    items,
    cta: ctaItem,
    colors: {
      background: b.background?.color_hex || "#FFFFFF",
      type: pickTypeColor(b.palette, b.background?.color_hex),
      accent: (b.palette || []).find((p) => p.role === "accent")?.hex
        || (b.palette || [])[0]?.hex || "#111111",
    },
  };
}

/** The colour of the type ON the accent badge. It must be judged against the
 *  BADGE, not against white — a dark accent with dark text is invisible. */
export function onAccentColor(accentHex) {
  return pickTypeColor([{ hex: accentHex }], accentHex);
}

/** Recraft's native `texts` payload — real fonts, real positions (percent). */
export function briefToTextSpecs(brief, { size } = {}) {
  const width = size?.width || 1024;
  const height = size?.height || 1024;
  const plan = planTypeLayout(brief, { width, height });
  const specs = [];
  for (const item of plan.items) {
    if (!item.lines || !item.lines.length) continue;
    const text = item.lines.join(" ");
    specs.push({
      text: text.slice(0, 180),
      font_family: mapFontFamily(item.font),
      font_size: Math.max(8, Math.round(item.size)),
      font_weight: String(item.weight || 400),
      text_color: item.role === "cta" ? onAccentColor(plan.colors.accent) : plan.colors.type,
      text_align: item.anchor === "middle" ? "center" : item.align === "right" ? "right" : "left",
      x: Math.round((item.x / width) * 100),
      y: Math.round((item.y / height) * 100),
      letter_spacing: Math.round((item.tracking || 0) * 10) / 10,
      ...(item.role === "cta" && item.box ? {
        text_background: { color: plan.colors.accent },
      } : {}),
    });
  }
  return specs;
}

function mapFontFamily(fontName) {
  const f = String(fontName || "").toLowerCase();
  if (/mono|code/.test(f)) return "roboto_mono";
  if (/playfair|garamond|didot|bodoni/.test(f)) return "playfair_display";
  if (/serif|lora|times|georgia/.test(f)) return "noto_serif";
  if (/titan|impact|black|heavy|display/.test(f)) return "titan_one";
  if (/shrikhand|script|hand/.test(f)) return "shrikhand";
  if (/oswald|condensed/.test(f)) return "oswald";
  if (/bricolage|grotesk/.test(f)) return "bricolage_grotesque";
  if (/poppins|geometric/.test(f)) return "poppins";
  if (/montserrat/.test(f)) return "montserrat";
  if (/anton/.test(f)) return "anton";
  return "noto_sans";
}

// ---------------------------------------------------------------------------
// SVG COMPOSITE
// ---------------------------------------------------------------------------

function svgForPlan(plan) {
  const parts = [];
  parts.push(`<svg width="${plan.width}" height="${plan.height}" xmlns="http://www.w3.org/2000/svg">`);
  // The scrim goes UNDER the type: a soft colour band that guarantees the
  // copy is readable and covers the stray lettering an image model leaves
  // behind when it ignores "no text in the image".
  if (plan.scrim) {
    const { y0, y1, color, alpha } = plan.scrim;
    const h = Math.max(1, y1 - y0);
    parts.push(
      `<defs><linearGradient id="scrim" x1="0" y1="0" x2="0" y2="1">` +
      `<stop offset="0%" stop-color="${color}" stop-opacity="${alpha}"/>` +
      `<stop offset="12%" stop-color="${color}" stop-opacity="${alpha}"/>` +
      `<stop offset="88%" stop-color="${color}" stop-opacity="${alpha}"/>` +
      `<stop offset="100%" stop-color="${color}" stop-opacity="0"/>` +
      `</linearGradient></defs>` +
      `<rect x="0" y="${y0}" width="${plan.width}" height="${h + Math.round(h * 0.12)}" fill="url(#scrim)"/>`
    );
  }
  for (const item of plan.items) {
    const family = escapeXml(fontStack(item.font));
    // Each line carries the colour that was picked against the artwork
    // actually under IT.
    const fill = item.color || (item.role === "cta" ? onAccentColor(plan.colors.accent) : plan.colors.type);
    if (item.box) {
      parts.push(`<rect x="${item.box.x}" y="${item.box.y}" width="${item.box.w}" height="${item.box.h}" rx="${item.box.r}" fill="${plan.colors.accent}"/>`);
    }
    const lines = item.lines && item.lines.length ? item.lines : [item.text];
    const step = item.size * (item.leading || 1.1);
    lines.forEach((line, i) => {
      // Line N sits N steps below line 0. (It used to be `step` for every
      // line after the first, so lines 2+ were drawn on top of each other.)
      const dy = i * step;
      parts.push(
        `<text x="${item.x}" y="${item.y + dy}" font-family="${family}" font-size="${item.size}" ` +
        `font-weight="${item.weight}" letter-spacing="${item.tracking || 0}" ` +
        `text-anchor="${item.anchor}" fill="${fill}" xml:space="preserve">${escapeXml(line)}</text>`
      );
    });
  }
  parts.push("</svg>");
  return parts.join("");
}

/**
 * The mean luminance of the artwork behind ONE line of type.
 *
 * The brief's background colour is the DESIGN's background, not what the
 * renderer actually painted (it may be a photo, a gradient, a band). Judging
 * the type against the brief alone let "white on a bright photo" through,
 * and the reviewer rightly called it illegible. This measures the real
 * pixels, per line — a block that spans a dark area and a bright band needs
 * a different decision for each line.
 */
export async function measureLineBackdrop(buffer, item, plan, { sharp }) {
  try {
    const top = Math.max(0, Math.min(plan.height - 1, Math.round(item.y - item.size * 1.05)));
    const height = Math.max(2, Math.min(plan.height - top, Math.round(item.size * 1.5)));
    const left = Math.max(0, Math.min(plan.width - 1, Math.round(plan.margin)));
    const width = Math.max(2, Math.min(plan.width - left, plan.width - plan.margin * 2));
    const { channels } = await sharp(buffer)
      .extract({ left, top, width, height })
      .stats();
    const ch = channels[0] || {};
    const means = Array.isArray(ch.mean) ? ch.mean : [ch.mean, ch.mean, ch.mean];
    const stdevs = Array.isArray(ch.stdev) ? ch.stdev : [ch.stdev, ch.stdev, ch.stdev];
    const luma = (0.299 * (means[0] || 0) + 0.587 * (means[1] || 0) + 0.114 * (means[2] || 0)) / 255;
    const stdev = ((stdevs[0] || 0) + (stdevs[1] || 0) + (stdevs[2] || 0)) / 3 / 255;
    return { luma, busy: stdev > 0.16 };
  } catch {
    return null;
  }
}

/** A grey with the given relative luminance. */
export function greyAt(luma) {
  const v = Math.round(Math.min(1, Math.max(0, luma)) * 255);
  const h = v.toString(16).padStart(2, "0");
  return `#${h}${h}${h}`;
}

/**
 * Composite the brief's type onto the rendered artwork.
 * Returns the original buffer untouched when the host cannot rasterise text.
 */
export async function applyTypeLayer(imageBuffer, brief, { width, height } = {}) {
  const copy = brief?.copy || {};
  if (!copy.headline && !copy.subhead && !copy.body && !copy.cta) {
    return { buffer: imageBuffer, applied: false, reason: "no copy to typeset" };
  }
  if (!(await probeTextRendering())) {
    return { buffer: imageBuffer, applied: false, reason: "host cannot rasterise text (no fontconfig)" };
  }
  try {
    const sharp = (await import("sharp")).default;
    const meta = await sharp(imageBuffer).metadata();
    const w = width || meta.width || 1080;
    const h = height || meta.height || Math.round(w * 1.25);
    const plan = planTypeLayout(brief, { width: w, height: h });

    // Measure the artwork under EACH line and pick the type colour per line.
    // A scrim band goes down only when a line is unreadable even after the
    // colour is re-picked, or when the backdrop is busy enough that any
    // single colour will fight it.
    let scrim = null;
    for (const item of plan.items) {
      const back = await measureLineBackdrop(imageBuffer, item, plan, { sharp });
      if (!back) continue;
      const bgHex = greyAt(back.luma);
      const best = pickTypeColor([{ hex: plan.colors.type }, { hex: "#111111" }, { hex: "#FFFFFF" }], bgHex);
      item.color = best;
      item.backdrop = bgHex;
      if (contrastRatio(best, bgHex) < 4.5 || back.busy) {
        item.needsScrim = true;
      }
    }
    const needy = plan.items.filter((i) => i.needsScrim);
    if (needy.length) {
      // One soft band behind the whole copy block, in the brief's own type
      // colour: it guarantees legibility AND covers the stray lettering an
      // image model leaves when it ignores "no text in the image".
      const band = plan.colors.type;
      const scrimItems = needy.map((i) => i);
      // (`plan.size` does not exist — it made y0/y1 NaN, which broke the SVG
      // and silently dropped the whole type layer. Use the items' own sizes.)
      const y0 = Math.max(0, Math.round(Math.min(...scrimItems.map((i) => i.y - i.size * 1.2))));
      const y1 = Math.min(h, Math.round(Math.max(...scrimItems.map((i) => i.y + Math.max(0, (i.lines?.length || 1) - 1) * i.size * (i.leading || 1.1) + i.size * 0.9))));
      plan.scrim = { y0, y1, color: band, alpha: 0.82 };
      scrim = plan.scrim;
      // Every line now sits on the band, so re-pick against the band colour.
      for (const item of plan.items) {
        item.color = pickTypeColor([{ hex: band }], band);
      }
    }

    const svg = svgForPlan(plan);
    const overlay = await sharp(Buffer.from(svg)).png().toBuffer();
    const out = await sharp(imageBuffer)
      .resize(w, h, { fit: "cover" })
      .composite([{ input: overlay, top: 0, left: 0 }])
      .png()
      .toBuffer();
    return {
      buffer: out,
      applied: true,
      scrim: !!scrim,
      items: plan.items.map((i) => ({ role: i.role, lines: i.lines.length, size: i.size })),
      colors: plan.colors,
    };
  } catch (e) {
    // The artwork is still a valid deliverable without the type layer, so this
    // degrades instead of failing — but it must be visible in the log.
    console.warn("[type-layer] failed:", e.message);
    return { buffer: imageBuffer, applied: false, reason: `type layer failed: ${e.message}` };
  }
}
