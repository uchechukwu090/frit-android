// ============================================================================
// FRIT OPTIONS CHAIN ADAPTER — one interface, swappable $0 → paid sources
// ----------------------------------------------------------------------------
// Every provider returns the SAME normalized shape, so the scanner, backtest
// and journal never know where quotes came from:
//
//   {
//     symbol, underlying, quoteTime, expirations: ["YYYY-MM-DD"...],
//     contracts: [{ type: "call"|"put", strike, expiration, dte,
//                   bid, ask, mid, last, volume, oi, iv }],
//     source: "tradier" | "yahoo",
//     delayed: true,            // every $0 source is delayed or unofficial
//     provisional: boolean,    // true = unofficial/scrapable, may break
//     disclaimer: string        // shown next to every signal built from this
//   }
//
// Rules (from the project brief):
//   1. Fills are NEVER modeled at mid — the scanner uses bid/ask touch.
//   2. CBOE delayed-quote pages are OFF-LIMITS (they ban scraper IPs).
//   3. Tradier activates only when TRADIER_TOKEN is set (sandbox paper token);
//      otherwise Yahoo serves as the provisional $0 source.
//
// Yahoo Finance endpoints are UNOFFICIAL: no SLA, rate-limited, ToS-grey.
// Cache aggressively (15 min) and never hammer — one symbol/expiry at a time.
// ============================================================================

const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36";

const _cache = new Map(); // key -> { at, data }
const CACHE_MS = 15 * 60 * 1000;

function cacheGet(key) {
  const hit = _cache.get(key);
  if (hit && Date.now() - hit.at < CACHE_MS) return hit.data;
  _cache.delete(key);
  return null;
}
function cachePut(key, data) {
  _cache.set(key, { at: Date.now(), data });
  if (_cache.size > 60) _cache.delete(_cache.keys().next().value);
}

async function getJson(url, { headers = {}, timeoutMs = 20000 } = {}) {
  const res = await fetch(url, {
    headers: { "User-Agent": UA, Accept: "application/json", ...headers },
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(`chain HTTP ${res.status} for ${url.split("?")[0]}`);
  return res.json();
}

function ymd(epochSec) {
  return new Date(epochSec * 1000).toISOString().slice(0, 10);
}
function dteOf(expirationYmd, nowMs = Date.now()) {
  return Math.max(0, Math.round((new Date(expirationYmd + "T16:00:00Z").getTime() - nowMs) / 86400000));
}

// ---------------- Tradier (sandbox paper token; delayed chains, ORATS greeks) --
function tradierToken() {
  return process.env.TRADIER_TOKEN || "";
}
export function tradierConfigured() {
  return tradierToken().length > 10;
}

async function tradierChain(symbol) {
  const base = (process.env.TRADIER_BASE || "https://sandbox.tradier.com").replace(/\/$/, "");
  const H = { Authorization: `Bearer ${tradierToken()}`, Accept: "application/json" };
  const expJ = await getJson(
    `${base}/v1/markets/options/expirations?symbol=${encodeURIComponent(symbol)}&includeAllRoots=false`, { headers: H }
  );
  const dates = expJ?.expirations?.date;
  const expList = (Array.isArray(dates) ? dates : [dates]).filter(Boolean);
  const wanted = expList.filter((d) => dteOf(d) >= 14).slice(0, 6);
  if (!wanted.length) throw new Error("tradier: no expirations >= 14 DTE");
  const contracts = [];
  for (const exp of wanted) {
    const cj = await getJson(
      `${base}/v1/markets/options/chains?symbol=${encodeURIComponent(symbol)}&expiration=${exp}&greeks=true`, { headers: H }
    );
    const opts = cj?.options?.option;
    const list = Array.isArray(opts) ? opts : (opts ? [opts] : []);
    for (const o of list) {
      const bid = Number(o.bid), ask = Number(o.ask);
      if (!(o.strike > 0) || !(bid >= 0) || !(ask >= 0)) continue;
      contracts.push({
        type: String(o.option_type || "").toLowerCase() === "put" ? "put" : "call",
        strike: Number(o.strike), expiration: exp, dte: dteOf(exp),
        bid, ask, mid: bid > 0 && ask > 0 ? (bid + ask) / 2 : Number(o.last) || 0,
        last: Number(o.last) || 0,
        volume: Number(o.volume) || 0, oi: Number(o.open_interest) || 0,
        iv: Number(o.greeks?.mid_iv ?? o.greeks?.smv_vol ?? NaN) || null,
      });
    }
  }
  // Underlying from a stock quote (indices unavailable in sandbox — caller
  // passes S for index underlyings; see getChain fallback).
  let S = NaN;
  try {
    const qj = await getJson(`${base}/v1/markets/quotes?symbols=${encodeURIComponent(symbol)}`, { headers: H });
    const q = qj?.quotes?.quote;
    const first = Array.isArray(q) ? q[0] : q;
    S = Number(first?.last ?? first?.close ?? NaN);
  } catch { /* underlying optional */ }
  return {
    symbol, underlying: Number.isFinite(S) ? S : null, quoteTime: Date.now(),
    expirations: wanted, contracts,
    source: "tradier", delayed: true, provisional: false,
    disclaimer: "Tradier sandbox — delayed ~15 min, paper only",
  };
}

// ---------------- Yahoo (provisional $0; unofficial, delayed, fragile) ---------
async function yahooChain(symbol) {
  const root = `https://query1.finance.yahoo.com/v7/finance/options/${encodeURIComponent(symbol)}`;
  const first = await getJson(root);
  const res = first?.optionChain?.result?.[0];
  if (!res) throw new Error("yahoo: no optionChain result (rate-limited or bad symbol?)");
  const S = Number(res?.quote?.regularMarketPrice);
  if (!(S > 0)) throw new Error("yahoo: no underlying price");
  const expEpochs = (res.expirationDates || []).filter((e) => dteOf(ymd(e)) >= 14).slice(0, 6);
  const contracts = [];
  const pushSide = (arr, type, exp) => {
    for (const c of arr || []) {
      const strike = Number(c.strike), bid = Number(c.bid ?? NaN), ask = Number(c.ask ?? NaN);
      if (!(strike > 0)) continue;
      contracts.push({
        type, strike, expiration: exp, dte: dteOf(exp),
        bid: Number.isFinite(bid) ? bid : 0, ask: Number.isFinite(ask) ? ask : 0,
        mid: Number(c.lastPrice) || 0, last: Number(c.lastPrice) || 0,
        volume: Number(c.volume) || 0, oi: Number(c.openInterest) || 0,
        iv: Number.isFinite(Number(c.impliedVolatility)) ? Number(c.impliedVolatility) : null,
      });
    }
  };
  // First expiry block ships inside the root response.
  const firstExp = ymd(res.options?.[0]?.expirationDate ?? res.expirationDates?.[0] ?? 0);
  for (const blk of res.options || []) {
    const exp = ymd(blk.expirationDate);
    pushSide(blk.calls, "call", exp);
    pushSide(blk.puts, "put", exp);
  }
  // Remaining expirations need one request each — polite, sequential.
  for (const e of expEpochs.slice(1)) {
    const exp = ymd(e);
    if (contracts.some((c) => c.expiration === exp)) continue;
    await new Promise((r) => setTimeout(r, 800)); // do not hammer
    try {
      const j = await getJson(`${root}?date=${e}`);
      for (const blk of j?.optionChain?.result?.[0]?.options || []) {
        pushSide(blk.calls, "call", exp);
        pushSide(blk.puts, "put", exp);
      }
    } catch (err) {
      console.warn(`[chains] yahoo expiry ${exp} failed:`, err.message);
    }
  }
  void firstExp;
  return {
    symbol, underlying: S, quoteTime: Date.now(),
    expirations: [...new Set(contracts.map((c) => c.expiration))].sort(),
    contracts,
    source: "yahoo", delayed: true, provisional: true,
    disclaimer: "Unofficial delayed quotes for research only — verify before risking capital",
  };
}

// ---------------- Public entry -------------------------------------------------
/**
 * @param {string} symbol  "SPY" | "QQQ" (equity/ETF underlyings)
 * @param {{underlying?: number}} opts  S override (index underlyings in sandbox)
 */
export async function getChain(symbol, opts = {}) {
  const sym = String(symbol || "").toUpperCase();
  const key = `chain:${sym}`;
  const hit = cacheGet(key);
  if (hit) return hit;
  let chain;
  if (tradierConfigured()) {
    chain = await tradierChain(sym);
    if (opts.underlying > 0 && !Number.isFinite(chain.underlying)) chain.underlying = opts.underlying;
  } else {
    chain = await yahooChain(sym);
  }
  if (!chain.contracts.length) throw new Error(`${chain.source}: empty chain for ${sym}`);
  cachePut(key, chain);
  return chain;
}

/** Daily closes for RV/backtest. Stooq free CSV first (no key), Finnhub optional. */
export async function fetchDailyCloses(symbol, { years = 3 } = {}) {
  const stooq = { SPY: "spy.us", QQQ: "qqq.us", IWM: "iwm.us", XSP: "xsp.us", EURUSD: "eurusd", GBPUSD: "gbpusd", USDJPY: "usdjpy", XAUUSD: "xauusd" }[String(symbol).toUpperCase()];
  if (stooq) {
    try {
      const res = await fetch(`https://stooq.com/q/d/l/?s=${stooq}&i=d`, { signal: AbortSignal.timeout(20000) });
      if (res.ok) {
        const rows = (await res.text()).trim().split("\n").slice(1);
        const closes = rows.map((r) => {
          const [, , , , c] = r.split(",");
          return Number(c);
        }).filter((c) => c > 0);
        const cutoff = Date.now() - years * 365 * 86400000;
        void cutoff;
        return closes.slice(-Math.round(years * 252) - 70);
      }
    } catch (e) {
      console.warn("[chains] stooq failed, trying finnhub:", e.message);
    }
  }
  const key = process.env.FINNHUB_KEY;
  if (!key) throw new Error("no daily closes: stooq failed and FINNHUB_KEY unset");
  const map = { SPY: "SPY", QQQ: "QQQ", IWM: "IWM", EURUSD: "OANDA:EUR_USD", GBPUSD: "OANDA:GBP_USD", USDJPY: "OANDA:USD_JPY", XAUUSD: "OANDA:XAU_USD" };
  const fsym = map[String(symbol).toUpperCase()] || symbol;
  const to = Math.floor(Date.now() / 1000), from = to - Math.round(years * 365.25 * 86400) - 70 * 86400;
  const j = await getJson(`https://finnhub.io/api/v1/stock/candle?symbol=${encodeURIComponent(fsym)}&resolution=D&from=${from}&to=${to}&token=${key}`);
  if (j.s !== "ok" || !Array.isArray(j.c)) throw new Error("finnhub candles failed");
  return j.c.filter((c) => c > 0);
}
