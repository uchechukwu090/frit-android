// ============================================================================
// FRIT DEEP SEARCH — multi-engine web research pipeline
// ----------------------------------------------------------------------------
// Replaces the old single-call DuckDuckGo instant-answer with a real research
// engine:
//   1. DuckDuckGo HTML  (https://html.duckduckgo.com/html/?q=)  — ~30 results
//   2. Bing             (https://www.bing.com/search?q=)        — ~20 results
//   3. Brave            (https://search.brave.com/search?q=)    — ~15 results
//
// Each engine is scraped keyless (plain fetch + HTML parsing), results are
// deduped by normalized URL, ranked with engine weight, and returned as a real
// result list (title / URL / snippet) — NOT a single abstract paragraph.
//
// Dorking operators (Google/Bing syntax): intitle:, inurl:, site:, filetype:,
// -site:, "exact phrase", AROUND(n), etc. are passed straight through to the
// engines, which honor them natively. DuckDuckGo honors a subset.
// ============================================================================

const UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";

const ENGINE_WEIGHT = { duckduckgo: 1.0, bing: 0.9, brave: 0.8, braveapi: 1.1 };

// 1h digest cache — search results don't change minute to minute; this also
// shields us when an engine bot-walls us mid-day.
const _searchCache = new Map();
const SEARCH_TTL_MS = 60 * 60 * 1000;
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of _searchCache.entries()) if (now > v.exp) _searchCache.delete(k);
}, 10 * 60 * 1000).unref?.();

// Brave official API (free tier key, no funding needed). Skipped when unset.
const BRAVE_API_KEY = process.env.BRAVE_API_KEY || "";
async function searchBraveApi(query) {
  if (!BRAVE_API_KEY) throw new Error("BRAVE_API_KEY unset");
  const res = await fetch(`https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=10&freshness=py`, {
    headers: { "X-Subscription-Token": BRAVE_API_KEY, Accept: "application/json" },
    signal: AbortSignal.timeout(12_000),
  });
  if (!res.ok) throw new Error(`Brave API HTTP ${res.status}`);
  const data = await res.json();
  return (data.web?.results || []).slice(0, 10).map(r => ({
    title: r.title || "",
    url: cleanUrl(r.url || ""),
    snippet: (r.description || "").slice(0, 300),
    engine: "braveapi",
    published: r.page_age || r.age || "",
  }));
}

// ---------------------------------------------------------------------------
// HTML helpers
// ---------------------------------------------------------------------------
function decodeEntities(s) {
  if (!s) return "";
  return String(s)
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#0?39;/g, "'")
    .replace(/&#x27;/g, "'")
    .replace(/&#x22;/g, '"')
    .replace(/&nbsp;/g, " ")
    .replace(/&#(\d+);/g, (_m, d) => {
      try { return String.fromCodePoint(parseInt(d, 10)); } catch { return ""; }
    });
}

function stripTags(s) {
  if (!s) return "";
  return decodeEntities(String(s).replace(/<[^>]*>/g, " ").replace(/\s+/g, " ")).trim();
}

function cleanUrl(u) {
  if (!u) return "";
  let out = u.trim();
  try {
    const url = new URL(out.startsWith("http") ? out : `https://${out}`);
    // Strip common tracking params so the same page from different engines dedupes.
    for (const p of ["utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term", "fbclid", "gclid", "ref", "trk"]) {
      url.searchParams.delete(p);
    }
    url.hash = "";
    out = url.toString();
  } catch { /* keep raw */ }
  return out.replace(/\/$/, "");
}

async function fetchHtml(url, timeoutMs = 12000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      headers: {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
      },
      redirect: "follow",
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    return text.length > 8 * 1024 * 1024 ? text.slice(0, 8 * 1024 * 1024) : text;
  } catch (err) {
    throw new Error(`fetch ${url} failed: ${err.message}`);
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Engine scrapers
// ---------------------------------------------------------------------------
async function searchDuckDuckGo(query) {
  const html = await fetchHtml(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}&kl=us-en`);
  const results = [];
  // <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=...">
  const linkRe = /<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
  const snipRe = /<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>/gi;
  const links = [...html.matchAll(linkRe)];
  const snippets = [...html.matchAll(snipRe)];
  for (let i = 0; i < links.length; i++) {
    const [, rawUrl, rawTitle] = links[i];
    let target = resolveDdgUrl(rawUrl);
    if (!target || !target.startsWith("http") || /duckduckgo\.com|google\.com|bing\.com|bing\.net/i.test(target)) continue;
    results.push({
      title: stripTags(rawTitle),
      url: cleanUrl(target),
      snippet: snippets[i] ? stripTags(snippets[i][1]) : "",
      engine: "duckduckgo",
    });
    if (results.length >= 25) break;
  }
  return results;
}

function resolveDdgUrl(rawUrl) {
  try {
    const decoded = decodeEntities(rawUrl);
    const withProto = decoded.startsWith("//") ? "https:" + decoded : decoded;
    const urlObj = new URL(withProto);
    const uddg = urlObj.searchParams.get("uddg");
    if (uddg) {
      try { return decodeURIComponent(uddg); } catch { return uddg; }
    }
    return withProto;
  } catch {
    return null;
  }
}

async function searchBing(query) {
  const html = await fetchHtml(`https://www.bing.com/search?q=${encodeURIComponent(query)}&setlang=en&cc=us`);
  const results = [];
  // <li class="b_algo"> ... <h2><a href="...">TITLE</a></h2> ... <p>SNIPPET</p>
  const re = /<li class="b_algo"[\s\S]*?<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>\s*<\/h2>([\s\S]*?)<\/li>/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    const [, rawUrl, rawTitle, rest] = m;
    const url = cleanUrl(resolveBingUrl(rawUrl));
    if (!url.startsWith("http") || /bing\.com|go\.microsoft\.com|microsoft\.com\/search/i.test(url)) continue;
    const p = rest.match(/<p[^>]*>([\s\S]*?)<\/p>/i);
    results.push({
      title: stripTags(rawTitle),
      url,
      snippet: p ? stripTags(p[1]) : "",
      engine: "bing",
    });
    if (results.length >= 25) break;
  }
  return results;
}

function resolveBingUrl(rawUrl) {
  const decoded = decodeEntities(rawUrl);
  // Bing organic results are /ck/a?!...&u=a1<base64-url> redirect links.
  const m = decoded.match(/[?&]u=a1([A-Za-z0-9+/=\-_]+)/);
  if (m) {
    try {
      let s = m[1].replace(/-/g, "+").replace(/_/g, "/");
      while (s.length % 4) s += "=";
      const out = Buffer.from(s, "base64").toString("utf8");
      if (out.startsWith("http")) return out;
    } catch { /* fall through */ }
  }
  return decoded;
}

async function searchBrave(query) {
  const html = await fetchHtml(`https://search.brave.com/search?q=${encodeURIComponent(query)}&source=web`);
  const results = [];
  // Brave markup: <div class="snippet fdb" data-pos="N"> with
  //   <a class="snippet-title" href="URL">TITLE</a> and <div class="snippet-description">...</div>
  const re = /<div[^>]*class="[^"]*snippet[^"]*"[^>]*data-pos="\d+"[\s\S]*?<a[^>]*class="[^"]*snippet-title[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>([\s\S]*?)<\/div>/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    const [, rawUrl, rawTitle, rest] = m;
    const url = cleanUrl(decodeEntities(rawUrl));
    if (!url.startsWith("http") || /brave\.com|search\.brave/i.test(url)) continue;
    const desc = rest.match(/<div[^>]*class="[^"]*snippet-description[^"]*"[^>]*>([\s\S]*?)<\/div>/i);
    results.push({
      title: stripTags(rawTitle),
      url,
      snippet: desc ? stripTags(desc[1]) : "",
      engine: "brave",
    });
    if (results.length >= 25) break;
  }
  return results;
}

// ---------------------------------------------------------------------------
// Page fetch + date extraction
// ---------------------------------------------------------------------------
// Snippets lie by omission — the agent was citing pages it never opened.
// Fetch the top-3 result pages and extract readable text (fail-soft: snippet
// survives when fetch dies). Also pull publish dates so 2024 news stops
// leaking into "current" answers.
function extractDate(html) {
  const m = html.match(/<meta[^>]+(?:article:published_time|publish_date|datePublished|og:updated_time|article:modified_time)[^>]+content="([^"]+)"/i)
    || html.match(/<time[^>]+datetime="([^"]+)"/i)
    || html.match(/"datePublished"\s*:\s*"([^"]+)"/i);
  if (!m) return "";
  const d = new Date(m[1]);
  return isNaN(d.getTime()) ? "" : d.toISOString().slice(0, 10);
}

async function fetchPageText(url) {
  const html = await fetchHtml(url, 10_000);
  const date = extractDate(html);
  const text = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<nav[\s\S]*?<\/nav>/gi, " ")
    .replace(/<header[\s\S]*?<\/header>/gi, " ")
    .replace(/<footer[\s\S]*?<\/footer>/gi, " ");
  const paras = [...text.matchAll(/<p[^>]*>([\s\S]*?)<\/p>/gi)]
    .map(p => stripTags(p[1]))
    .filter(p => p.length > 40);
  const body = (paras.slice(0, 12).join("\n") || stripTags(text)).replace(/\s+/g, " ").trim();
  return { text: body.slice(0, 1500), date };
}
function normalizeHost(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
}

function dedupeAndRank(named) {
  const seen = new Set();
  const ranked = [];
  for (const { name, rows } of named) {
    const weight = ENGINE_WEIGHT[name] || 0.7;
    for (const r of rows || []) {
      if (!r.url || !r.title) continue;
      const key = `${normalizeHost(r.url)}|${r.title.toLowerCase().replace(/\W+/g, " ").trim().slice(0, 60)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      ranked.push({ ...r, _score: weight });
    }
  }
  ranked.sort((a, b) => b._score - a._score);
  return ranked.slice(0, 12).map(({ _score, ...r }) => r);
}

/**
 * Deep multi-engine web search.
 * @param {string} query  raw query, dorking operators supported
 * @param {string} [mode] "structured" returns the full result list; anything
 *                        else returns a compact text digest for LLM consumption
 */
export async function deepSearch(query, mode = "text") {
  const q = String(query || "").trim();
  if (!q) return { ok: false, error: "query required" };
  const cacheKey = `${mode}:${q.toLowerCase()}`;
  const hit = _searchCache.get(cacheKey);
  if (hit && Date.now() < hit.exp) return { ...hit.val, cached: true };

  const attempts = [
    { name: "duckduckgo", fn: () => searchDuckDuckGo(q) },
    { name: "bing", fn: () => searchBing(q) },
    { name: "brave", fn: () => searchBrave(q) },
    { name: "braveapi", fn: () => searchBraveApi(q) },
  ];

  const named = [];
  const failures = [];
  // Fire all engines in parallel but isolate failures so one blocked/bot-walled
  // engine never kills the whole search.
  await Promise.all(
    attempts.map(async ({ name, fn }) => {
      try {
        named.push({ name, rows: await fn() });
      } catch (err) {
        failures.push(`${name} (${err.message})`);
      }
    })
  );

  const results = dedupeAndRank(named);

  if (!results.length && failures.length) {
    return { ok: false, query: q, error: `All search engines failed: ${failures.join(" | ")}` };
  }

  // Open the top-3 pages and attach real content + publish dates (fail-soft).
  await Promise.all(results.slice(0, 3).map(async (r) => {
    try {
      const page = await fetchPageText(r.url);
      if (page.text) r.page_text = page.text;
      if (page.date) r.published = page.date;
    } catch { /* snippet survives */ }
  }));

  if (mode === "structured") {
    const out = {
      ok: true,
      query: q,
      total: results.length,
      engines: { queried: attempts.map(a => a.name), failed: failures },
      results,
    };
    _searchCache.set(cacheKey, { exp: Date.now() + SEARCH_TTL_MS, val: out });
    return out;
  }

  // Compact text digest — readable by the LLM brain in one shot. Opened pages
  // carry real quotes + dates; warn the brain off undated items for news queries.
  const digest = results.map((r, i) => {
    const date = r.published ? ` [${r.published}]` : "";
    const extra = r.page_text ? `\n   > ${r.page_text.slice(0, 400)}` : "";
    return `${i + 1}. ${r.title}${date} — ${r.url}\n   ${(r.snippet || "(no snippet)").slice(0, 220)}${extra}`;
  }).join("\n");
  const intro = results.length
    ? `Top ${results.length} results for "${q}":`
    : `No results found for "${q}".`;
  const failNote = failures.length ? `\n[engines unavailable: ${failures.join(", ")}]` : "";
  const out = { ok: true, query: q, total: results.length, data: `${intro}\n${digest}${failNote}` };
  _searchCache.set(cacheKey, { exp: Date.now() + SEARCH_TTL_MS, val: out });
  return out;
}