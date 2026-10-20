// ============================================================================
// REFERENCE CACHE — server-side bytes for the images the agent studied
// ----------------------------------------------------------------------------
// Why this exists:
//   1. Pinterest / Behance CDNs hotlink-protect a lot of their images. When
//      the vision provider (not our server) fetches the URL it gets a 403 and
//      the agent silently loses a reference.
//   2. The Android client wants to SHOW which references were studied, so
//      the brief can be audited ("where did this come from?").
//   3. The same reference is often read by more than one model; caching the
//      bytes once keeps that to a single upstream fetch.
//
// The proxy id is a 192-bit random hex — unguessable, so the route can be
// public (vision providers cannot send our bearer token).
// ============================================================================

const DEFAULT_TTL_MS = 30 * 60 * 1000;
const DEFAULT_MAX_ENTRIES = 64;
const DEFAULT_MAX_BYTES = 8 * 1024 * 1024;

export const REFERENCE_PATH_PREFIX = "/creative-suite/refs/";

export function createRefStore({ ttlMs = DEFAULT_TTL_MS, maxEntries = DEFAULT_MAX_ENTRIES, maxBytes = DEFAULT_MAX_BYTES } = {}) {
  const store = new Map(); // id -> { buf, mime, url, createdAt }

  function sweep(now = Date.now()) {
    for (const [id, entry] of store) {
      if (now - entry.createdAt > ttlMs) store.delete(id);
    }
    while (store.size > maxEntries) {
      const oldest = [...store.entries()].sort((a, b) => a[1].createdAt - b[1].createdAt)[0];
      if (!oldest) break;
      store.delete(oldest[0]);
    }
  }

  function put(id, buf, mime, url) {
    store.set(id, { buf, mime, url, createdAt: Date.now() });
    sweep();
    return id;
  }

  function get(id) {
    const entry = store.get(String(id || ""));
    if (!entry) return null;
    if (Date.now() - entry.createdAt > ttlMs) { store.delete(entry.id || id); return null; }
    return entry;
  }

  /** Fetch a remote reference once and keep the bytes. Returns the entry. */
  async function fetchAndStore(url, { fetchImpl = fetch, id = randomId() } = {}) {
    const res = await fetchImpl(url, {
      headers: { "User-Agent": "Mozilla/5.0 (compatible; FRIT/1.0)" },
      signal: AbortSignal.timeout(15_000),
    });
    if (!res?.ok) throw new Error(`reference fetch HTTP ${res?.status || 0}`);
    const buf = Buffer.from(await res.arrayBuffer());
    if (!buf.length) throw new Error("reference fetch returned no bytes");
    if (buf.length > maxBytes) throw new Error("reference too large");
    const mime = contentTypeFor(url, res.headers?.get?.("content-type"));
    put(id, buf, mime, url);
    return { id, buf, mime, url };
  }

  return { store, put, get, fetchAndStore, sweep, size: () => store.size };
}

export function randomId() {
  const bytes = new Uint8Array(24);
  try {
    // Node's crypto is available in every supported runtime; fall back to a
    // Math.random id only if it is not (never happens on the server).
    // eslint-disable-next-line no-undef
    const { webcrypto } = globalThis;
    webcrypto.getRandomValues(bytes);
  } catch {
    for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  }
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function contentTypeFor(url, header) {
  const h = String(header || "").toLowerCase();
  if (h.includes("png")) return "image/png";
  if (h.includes("webp")) return "image/webp";
  if (h.includes("gif")) return "image/gif";
  if (h.includes("jpeg") || h.includes("jpg")) return "image/jpeg";
  const u = String(url || "").toLowerCase().split("?")[0];
  if (u.endsWith(".png")) return "image/png";
  if (u.endsWith(".webp")) return "image/webp";
  if (u.endsWith(".gif")) return "image/gif";
  if (u.endsWith(".jpg") || u.endsWith(".jpeg")) return "image/jpeg";
  return "image/jpeg";
}

/** A single process-wide store — the server is the only consumer. */
let _store = null;
export function refStore() {
  if (!_store) _store = createRefStore();
  return _store;
}
