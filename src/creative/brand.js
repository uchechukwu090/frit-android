// ============================================================================
// FRIT BRAND KIT — server-side brand lock (audit F-13)
// ----------------------------------------------------------------------------
// Before this module the brand lock was built CLIENT-SIDE: the Android app
// formatted a "Brand lock [Name]: colors #A, #B; font ..." string and pasted
// it into the prompt. That meant:
//   - only the Design Studio applied it (other studios never saw it),
//   - the server could not verify or re-apply the lock (a request from any
//     other client, or a retry, silently lost the brand),
//   - the client's text-splicing fought the server's own brand handling in
//     /creative-suite/design, which already looked the kit up by id.
//
// The lock is now resolved SERVER-SIDE from a brandkit_id and applied to the
// prompt by the server, so every studio gets it and the client only has to
// send the id.
// ============================================================================

/**
 * Normalize a brandkit DB row into the shape the prompt builders expect.
 * @param {object|null} row
 */
export function normalizeBrand(row) {
  if (!row) return null;
  const colors = safeColors(row.colors);
  if (!colors.length && !row.font && !row.palette && !row.tone && !row.logo_url) return null;
  return {
    name: String(row.name || "").slice(0, 80),
    colors,
    font: String(row.font || "").slice(0, 60),
    palette: String(row.palette || "").slice(0, 60),
    tone: String(row.tone || "").slice(0, 120),
    logo_url: String(row.logo_url || "").slice(0, 500),
  };
}

function safeColors(raw) {
  try {
    const arr = typeof raw === "string" ? JSON.parse(raw || "[]") : raw;
    if (!Array.isArray(arr)) return [];
    return arr
      .map((c) => String(c || "").trim())
      .filter((c) => /^#?[0-9a-fA-F]{6}$/.test(c))
      .slice(0, 6);
  } catch {
    return [];
  }
}

/**
 * Build the brand-lock fragment that gets appended to a generation prompt.
 * This is the ONE place the lock text is written, so every studio (images,
 * video, cinema, design) locks identically.
 */
export function brandPromptFragment(brand) {
  if (!brand) return "";
  const parts = [];
  if (brand.name) parts.push(`brand "${brand.name}"`);
  if (brand.colors.length) parts.push(`exact palette ${brand.colors.join(", ")}`);
  if (brand.font) parts.push(`${brand.font}-style typography`);
  if (brand.palette) parts.push(`grade toward ${brand.palette}`);
  if (brand.tone) parts.push(`${brand.tone} mood`);
  if (!parts.length) return "";
  return (
    `Brand lock (${parts.join("; ")}). Stay on-brand: use the exact brand hexes without ` +
    `drifting hues, keep the stated type voice, and match the stated mood.`
  );
}

/**
 * Append the brand lock to a prompt. Returns the prompt unchanged when there
 * is no brand, so callers can always use the result.
 */
export function applyBrandToPrompt(prompt, brand) {
  const base = String(prompt || "");
  const frag = brandPromptFragment(brand);
  if (!frag) return base;
  return base ? `${base}. ${frag}` : frag;
}

/**
 * Look a brand kit up for a user. Ownership is enforced in SQL, so a user
 * can never apply someone else's kit.
 * @returns {object|null} normalized brand
 */
export function resolveBrandKit(db, user_id, brandkit_id) {
  const id = String(brandkit_id || "").trim();
  if (!id) return null;
  try {
    const row = db
      .prepare("SELECT name, colors, font, logo_url, palette, tone FROM creative_brandkits WHERE id = ? AND user_id = ?")
      .get(id, user_id);
    return normalizeBrand(row);
  } catch {
    return null;
  }
}
