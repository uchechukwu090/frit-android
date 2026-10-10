// Design-rule prompt engine: what keeps our graphics from looking like
// generic AI output. Encodes working-designer rules (Pinterest/Canva-grade
// briefs) into every generation:
//
// 1. Quoted exact strings — typography models render quoted copy far more
//    accurately than described copy (the Ideogram pattern).
// 2. 60-30-10 color rule — dominant / secondary / accent from brand+palette.
// 3. Type hierarchy — headline > sub > body > CTA, each with a size role.
// 4. Layout grid — rule-of-thirds placement + safe margins + breathing room.
// 5. One idea per piece — a single focal point, never a collage of concepts.
// 6. Variations as art directions — same brief, distinct visual routes.

export const CRAFTS = ["auto", "type", "vector", "photo"];

export function pickCraft({ template, headline, body }) {
  const words = String(headline || "").trim().split(/\s+/).filter(Boolean).length;
  if (!headline && !body) return "vector"; // pure mark / icon work
  if (body || words > 3) return "type"; // copy-heavy -> typography specialist
  if (words > 0) return "vector"; // short wordmark/headline -> vector-clean
  return "type";
}

const ART_DIRECTIONS = [
  "Bold minimalism: one strong focal image, huge headline, generous whitespace, premium print finish",
  "Rich maximalism: layered textures, vibrant color-blocking, energetic composition, street-poster feel",
  "Clean editorial: grid-aligned Swiss layout, refined serif-sans pairing, muted tones, magazine polish",
];

export function buildDesignPrompt({ template, headline, body, cta, palette, brand, styleMood, variation = 0 }) {
  const L = [];
  L.push(tplLine(template));
  if (headline) L.push(`Headline (render EXACTLY as quoted): "${String(headline).slice(0, 120)}". Headline dominates — largest type on the page.`);
  if (body) L.push(`Supporting copy (render exactly as quoted, smaller than headline): "${String(body).slice(0, 300)}".`);
  if (cta) L.push(`Call-to-action as a distinct button or badge, short and punchy (render exactly): "${String(cta).slice(0, 80)}".`);
  if (palette || brand?.colors?.length) {
    const cols = [...(brand?.colors || []), palette].filter(Boolean).slice(0, 4);
    L.push(`Color system 60-30-10: dominant ${cols[0] || "warm cream"} 60%, secondary ${cols[1] || "deep espresso"} 30%, accent ${cols[2] || palette || "gold"} 10%. Exact brand hexes, no drifting hues.`);
  }
  if (brand?.font) L.push(`Typography voice: ${brand.font}-style letterforms${brand?.tone ? `, ${brand.tone} mood` : ""}.`);
  if (styleMood) L.push(`Mood-board direction: ${styleMood}.`);
  L.push("Layout: rule-of-thirds focal placement, clear margins, breathing room around type, one focal idea only — never a collage.");
  L.push(ART_DIRECTIONS[variation % ART_DIRECTIONS.length] + ".");
  L.push("Crisp legible typography throughout, correct spelling, print-ready finish, no gibberish text, no watermark, no lorem ipsum.");
  return L.join(" ");
}

function tplLine(t) {
  if (!t) return "Professional graphic design, intentional composition.";
  return `Graphic design (${t.name}): ${t.hint}. Zones in play: ${(t.zones || []).join(", ")}.`;
}

export function variationCount(n) {
  return Math.min(3, Math.max(1, Number(n) || 1));
}
