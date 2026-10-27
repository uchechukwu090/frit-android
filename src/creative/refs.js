// ============================================================================
// REFS — inspiration reference transport
// ----------------------------------------------------------------------------
// The Design Agent needs the vision model to actually LOOK at Pinterest /
// Behance artwork. Two practical problems stop a plain `fetch` from working:
//
//   1. HOTLINK PROTECTION — i.pinimg.com and the Behance CDN refuse requests
//      that do not look like a browser, and refuse requests from a different
//      origin. A vision provider fetching the URL directly gets a 403.
//   2. PAYLOAD SIZE — a full Pinterest pin is 1-4MB; sending that to a
//      multimodal model is slow, expensive and sometimes rejected.
//
// This module solves both with one path:
//   fetch (browser-like headers) -> content-type check -> size cap ->
//   sharp downscale to <=1024px JPEG -> cache on disk under a
//   content-addressed id -> serve that id from our own endpoint.
//
// The fallback for a host that blocks even a browser-like fetch is
// images.weserv.nl, a keyless image proxy that fetches origin-side and
// re-serves the bytes. That URL is also what we hand to a vision provider
// when we cannot inline the bytes ourselves.
//
// Everything is dependency-injected (fetchImpl) so it is unit-testable.
// ============================================================================

import { createHash } from "node:crypto";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";

const UA = "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";
const MAX_BYTES = 8 * 1024 * 1024;
const VISION_MAX_EDGE = 1024;
const CACHE_DIR = process.env.FRIT_REF_CACHE || path.join(os.tmpdir(), "frit-refs");

const MIME_BY_EXT = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
  ".gif": "image/gif",
};

let sharp = null;
try {
  sharp = (await import("sharp")).default;
} catch {
  sharp = null;
}

/** Content-addressed id for a URL — unguessable, stable, cache-friendly. */
export function refId(url) {
  return createHash("sha256").update(String(url || "")).digest("hex").slice(0, 32);
}

/**
 * A keyless public image proxy. The vision provider fetches THIS url, and
 * the proxy fetches the origin server-side — which is exactly what a
 * hotlink-protected CDN allows.
 */
export function proxyImageUrl(url, { output = "jpg", q = 85 } = {}) {
  const raw = String(url || "").trim();
  if (!/^https?:\/\//i.test(raw)) return null;
  const host = raw.replace(/^https?:\/\//i, "");
  return `https://images.weserv.nl/?url=${encodeURIComponent(host)}&output=${output}&q=${q}`;
}

/** For documentation and UI explanations: which hosts this transport tries,
 *  and why a reference can still come back empty.
 */
export const REFERENCE_SOURCES = {
  search: ["bing image search (keyless HTML)", "openverse (keyless JSON)"],
  hostsThatUsuallyWork: ["unsplash", "pexels", "pixabay"],
  hostsThatOftenHotlinkProtect: ["pinterest", "behance", "dribbble"],
  fallbackProxy: "images.weserv.nl (origin-side fetch, no hotlink ban)",
  note: "Pinterest/Behance/Dribbble results frequently refuse direct server fetch; the proxy is tried next, and if both fail the reference is skipped rather than blocking the whole run.",
};

function mimeFromContentType(ct, url = "") {
  const v = String(ct || "").split(";")[0].trim().toLowerCase();
  if (v === "image/jpeg" || v === "image/jpg") return "image/jpeg";
  if (v === "image/png") return "image/png";
  if (v === "image/webp") return "image/webp";
  if (v === "image/gif") return "image/gif";
  if (v === "image/avif") return "image/avif";
  if (v.startsWith("image/")) return v;
  const ext = path.extname(String(url).split("?")[0]).toLowerCase();
  return MIME_BY_EXT[ext] || null;
}

/**
 * Fetch an image, browser-like. Returns { buf, mime, url } or null — a
 * reference is an enhancement, never a prerequisite, so every failure path
 * is a soft null rather than a throw.
 */
export async function fetchImageBuffer(url, { fetchImpl = fetch, timeoutMs = 15_000, maxBytes = MAX_BYTES } = {}) {
  const u = String(url || "").trim();
  if (!/^https?:\/\//i.test(u) || !fetchImpl) return null;
  try {
    const res = await fetchImpl(u, {
      headers: {
        "User-Agent": UA,
        Accept: "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
      },
      redirect: "follow",
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res?.ok) return null;
    const mime = mimeFromContentType(res.headers?.get?.("content-type"), u);
    if (!mime) return null;
    const buf = Buffer.from(await res.arrayBuffer());
    if (!buf.length || buf.length > maxBytes) return null;
    return { buf, mime, url: u };
  } catch {
    return null;
  }
}

/** Downscale to a JPEG small enough for a multimodal model. */
export async function shrinkForVision(buf, { maxEdge = VISION_MAX_EDGE, quality = 82 } = {}) {
  if (!sharp) return { buf, mime: null };
  try {
    const out = await sharp(buf)
      .resize({ width: maxEdge, height: maxEdge, fit: "inside", withoutEnlargement: true })
      .jpeg({ quality })
      .toBuffer();
    if (out?.length) return { buf: out, mime: "image/jpeg" };
  } catch { /* keep original */ }
  return { buf, mime: null };
}

/** Persist bytes under a content-addressed id and return the local path. */
export async function storeRef(url, buf, mime = "image/jpeg") {
  const id = refId(url);
  try {
    if (!existsSync(CACHE_DIR)) await mkdir(CACHE_DIR, { recursive: true });
    const file = path.join(CACHE_DIR, `${id}.${mime === "image/png" ? "png" : "jpg"}`);
    await writeFile(file, buf);
    return { id, file, mime };
  } catch {
    return { id, file: null, mime };
  }
}

/** Read a cached ref back. */
export async function readRef(id) {
  const safe = String(id || "").replace(/[^a-f0-9]/gi, "");
  if (!safe) return null;
  for (const ext of ["jpg", "png"]) {
    const file = path.join(CACHE_DIR, `${safe}.${ext}`);
    try {
      const buf = await readFile(file);
      return { buf, mime: ext === "png" ? "image/png" : "image/jpeg" };
    } catch { /* try next */ }
  }
  return null;
}

/**
 * THE reference path used by the Design Agent.
 *
 *  1. try to fetch the origin directly (browser-like headers)
 *  2. if that fails, try the public proxy (origin-side fetch, no hotlink ban)
 *  3. shrink to <=1024px JPEG
 *  4. cache under a content-addressed id
 *
 * Returns { id, dataUri, mime, bytes } or null. `dataUri` is what a vision
 * model can read directly; `id` is what our own endpoint can serve so the
 * app can show the reference without a third-party hotlink.
 *
 * NOTE: Pinterest/Behance/Dribbble results frequently refuse direct server
 * fetch (hotlink protection). The proxy (images.weserv.nl) is tried next;
 * if both fail, the reference is skipped rather than blocking the whole run.
 * That is expected and is NOT a bug — the agent still designs from the goal.
 */
export async function loadReference(url, { fetchImpl = fetch, cache = true } = {}) {
  // 1. origin fetch (browser-like).
  let got = await fetchImageBuffer(url, { fetchImpl });
  // 2. if the host hotlink-protects us, try the origin-side proxy; prefer the
  //    original format so SVG/PNG references are not destroyed as JPEG.
  if (!got) {
    // Try the origin-side proxy; keep SVG/PNG refs as their original format
    // instead of destroying them as JPEG.
    const format = (got?.mime || mimeFromContentType(null, url) || "").startsWith("image/svg") ? "svg" : "jpg";
    const proxied = proxyImageUrl(url, { output: format, q: format === "svg" ? 100 : 85 });
    if (proxied) got = await fetchImageBuffer(proxied, { fetchImpl });
  }
  if (!got) return null;
  const small = await shrinkForVision(got.buf);
  let smallBuf = small.buf;
  let mime = small.mime || got.mime || "image/jpeg";
  if (!smallBuf && got.buf.length <= 150 * 1024) {
    smallBuf = got.buf;
    mime = got.mime || mimeFromContentType(null, url) || "image/jpeg";
  }
  if (!smallBuf) {
    smallBuf = got.buf.slice(0, 600 * 1024);
    mime = got.mime || "image/jpeg";
  }
  const id = cache ? (await storeRef(url, smallBuf, mime)).id : null;
  return {
    id,
    mime,
    bytes: smallBuf.length,
    dataUri: `data:${mime};base64,${smallBuf.toString("base64")}`,
  };
}
