// ============================================================================
// FRIT MA-RSI COMBO ENGINE (30M Primary + 4H Confirmation)
// ----------------------------------------------------------------------------
// - Primary TF       : 30-minute (30M)
// - Confirmation TF  : 4-hour (4H)
// - Rule             : 4H EMA 9 > 21 = Bullish; 4H EMA 9 < 21 = Bearish.
//                      Trades are taken strictly in alignment with the 4H trend.
// - Persistent State : Does NOT require a fresh crossover this exact minute;
//                      existing EMA alignment defines the active trend regime.
// - Dual Scenarios   :
//     1. Immediate Entry  : Triggered when 30M aligns with 4H.
//     2. Pullback Re-entry: STRUCTURE-ANCHORED Fib zone (50-61.8% retracement
//        of the last 30M impulse leg, 30M EMA 21 only as a confluence filter).
//        Spot-EMA levels and one-step crossover math are NEVER published as
//        entries — the former migrates before price arrives, the latter solves
//        a one-bar crash print, not a pullback level (see fibPullbackZone).
// - Pullback Health  : Detects exhaustion vs real reversal using 30M volume,
//                      RSI, and swing structure without any extra API calls.
// - Cost gating      : Every published setup is repriced through the retail
//                      cost model (spread + commission + slippage, costs.js).
//                      netRR < 1.0 suppresses the setup — a trade that loses
//                      money even when directionally right is not a setup.
// - Range module     : When 4H is NEUTRAL the trend engine stands down and a
//                      mean-reversion range engine takes the floor: proven edges
//                      (2+ touches each), EMA compression, RSI-confirmed fades,
//                      breakout invalidation. Full rules in detectRange().
// - Hybrid TP/SL     :
//     - TP1: Nearest 30M swing liquidity pool capped at 1.5x ATR (high win-rate).
//     - TP2: 2.5x ATR runner.
//     - SL : Capped by ATR and anchored to recent 30M swing structure.
// - Zero Extra API Calls: Pullback health, structure breaks, and levels are
//   computed directly on the 30M array to preserve Twelve Data free tier limits.
// ============================================================================

import { applyCosts, NET_RR_BLOCK, NET_RR_WEAK } from "./costs.js";

const CACHE_TTL_MS = { "4h": 40 * 60 * 1000, "30m": 10 * 60 * 1000 };
const MAX_REQUESTS_PER_MINUTE = 6;
const MIN_TRADE_CONFIDENCE = 55;

const FAST_PERIOD = 9;
const SLOW_PERIOD = 21;
const RSI_PERIOD = 14;
const ATR_PERIOD = 14;
const ADX_PERIOD = 14;
const VOL_SMA_PERIOD = 20;
const ATR_SL_MULT = 1.5;
const MAX_SL_ATR_MULT = 2.5;
const MIN_RR = 1.5;

// 4H trend is only trusted when ADX is strong RELATIVE TO THAT SYMBOL'S OWN
// history (percentile rank), not a fixed absolute number — a fixed cutoff
// (e.g. ADX >= 25) means very different things on XAUUSD vs a quiet FX pair.
const ADX_STRONG_PERCENTILE = 60;
const PULLBACK_PROB_THRESHOLD = 55; // >= this -> treat pullback as healthy, wait for zone
const REVERSAL_PROB_THRESHOLD = 55; // >= this -> treat as real reversal, trade at current price

const SESSION_START_HOUR = 7;  // GMT, London/NY window
const SESSION_END_HOUR = 17;
const WEEKEND_DAYS = new Set([0, 6]);
const CRYPTO_SET = new Set(["BTCUSD", "ETHUSD", "BTCUSDT", "ETHUSDT"]);

// ---------------------------------------------------------------------------
// Pure Math & Indicator Helpers
// ---------------------------------------------------------------------------
function emaSeries(values, period) {
  if (!values.length) return [];
  const k = 2 / (period + 1);
  const out = [values[0]];
  for (let i = 1; i < values.length; i++) out.push(values[i] * k + out[i - 1] * (1 - k));
  return out;
}

function rsiSeries(closes, period = 14) {
  if (closes.length < period + 1) return [];
  const gains = [], losses = [];
  for (let i = 1; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    gains.push(Math.max(d, 0));
    losses.push(Math.max(-d, 0));
  }
  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
  const out = new Array(period).fill(null);
  out.push(avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss));
  for (let i = period; i < gains.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
    out.push(avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss));
  }
  return out;
}

function trueRange(c) {
  return Math.max(c.high - c.low, Math.abs(c.high - c.close), Math.abs(c.low - c.close));
}

function atr(candles, period = 14) {
  if (candles.length < period + 1) return null;
  const trs = [];
  for (let i = 1; i < candles.length; i++) trs.push(trueRange(candles[i]));
  const seed = trs.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let a = seed;
  for (let i = period; i < trs.length; i++) a = (a * (period - 1) + trs[i]) / period;
  return a;
}

function smaLast(values, period) {
  if (values.length < period) return null;
  const w = values.slice(-period);
  return w.reduce((a, b) => a + b, 0) / period;
}

// Wilder's DI+/DI-/ADX, returned as full series (aligned to each other, not
// to the candle index) so we can percentile-rank ADX against its own history.
function adxSeries(candles, period = 14) {
  const n = candles.length;
  if (n < period * 2) return { adx: [], diPlus: [], diMinus: [] };
  const tr = [], plusDM = [], minusDM = [];
  for (let i = 1; i < n; i++) {
    const c = candles[i], p = candles[i - 1];
    tr.push(trueRange(c));
    const upMove = c.high - p.high;
    const downMove = p.low - c.low;
    plusDM.push(upMove > downMove && upMove > 0 ? upMove : 0);
    minusDM.push(downMove > upMove && downMove > 0 ? downMove : 0);
  }
  const wilderSmooth = (arr) => {
    const out = [];
    let sum = arr.slice(0, period).reduce((a, b) => a + b, 0);
    out.push(sum);
    for (let i = period; i < arr.length; i++) {
      sum = sum - sum / period + arr[i];
      out.push(sum);
    }
    return out;
  };
  const trS = wilderSmooth(tr), plusS = wilderSmooth(plusDM), minusS = wilderSmooth(minusDM);
  const diPlus = [], diMinus = [], dx = [];
  for (let i = 0; i < trS.length; i++) {
    const dp = trS[i] === 0 ? 0 : (plusS[i] / trS[i]) * 100;
    const dm = trS[i] === 0 ? 0 : (minusS[i] / trS[i]) * 100;
    diPlus.push(dp);
    diMinus.push(dm);
    const sum = dp + dm;
    dx.push(sum === 0 ? 0 : (Math.abs(dp - dm) / sum) * 100);
  }
  const adx = [];
  if (dx.length >= period) {
    let avg = dx.slice(0, period).reduce((a, b) => a + b, 0) / period;
    adx.push(avg);
    for (let i = period; i < dx.length; i++) {
      avg = (avg * (period - 1) + dx[i]) / period;
      adx.push(avg);
    }
  }
  return { adx, diPlus, diMinus };
}

// Percentile rank of the latest value within its own trailing window (0-100).
// This is the cross-pair fix: "is ADX strong" becomes "strong FOR THIS SYMBOL
// relative to its own recent range" instead of a fixed number shared across
// wildly different instruments (XAUUSD vs a quiet FX pair vs BTCUSD).
function percentileRank(series, lookback = 100) {
  if (!series.length) return 50;
  const window = series.slice(-lookback);
  const last = window[window.length - 1];
  const below = window.filter(v => v < last).length;
  return (below / window.length) * 100;
}

// How deep the current pullback has retraced the prior impulse leg, as a
// Fibonacci-style ratio. Already symbol-agnostic since it's a ratio, not a
// raw price distance.
function computeRetracePct(swings, price, macroSide) {
  if (macroSide === "BULLISH" && swings.prevLow != null && swings.lastHigh != null && swings.lastHigh !== swings.prevLow) {
    return ((swings.lastHigh - price) / (swings.lastHigh - swings.prevLow)) * 100;
  }
  if (macroSide === "BEARISH" && swings.prevHigh != null && swings.lastLow != null && swings.prevHigh !== swings.lastLow) {
    return ((price - swings.lastLow) / (swings.prevHigh - swings.lastLow)) * 100;
  }
  return null;
}

// Pullback-vs-reversal probability. Every input here is either a percentile
// (self-relative to the symbol's own history) or a ratio (already symbol-
// agnostic), so this one function is meant to be trusted across ALL pairs
// without per-instrument retuning.
function scorePullback({ adxPercentile, diSpreadPercentile, retracePct, volRatio }) {
  let persist = 0, decay = 0;

  if (adxPercentile >= 60) persist += 35;
  else if (adxPercentile <= 30) decay += 35;

  if (diSpreadPercentile >= 60) persist += 25;
  else if (diSpreadPercentile <= 30) decay += 25;

  if (retracePct != null) {
    if (retracePct <= 50) persist += 25;
    else if (retracePct >= 61.8) decay += 25;
  }

  if (volRatio <= 1.1) persist += 15;
  else if (volRatio >= 1.3) decay += 15;

  const total = persist + decay || 1;
  return {
    pullbackProb: +(persist / total * 100).toFixed(1),
    reversalProb: +(decay / total * 100).toFixed(1),
  };
}

// Swings detector on 30M array (finds both swing highs and swing lows)
function detectSwings(candles, lookback = 3) {
  const highs = [];
  const lows = [];
  for (let i = candles.length - lookback - 1; i >= lookback; i--) {
    const c = candles[i];
    let isLow = true, isHigh = true;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (candles[j].low <= c.low) isLow = false;
      if (candles[j].high >= c.high) isHigh = false;
    }
    if (isHigh) highs.push({ index: i, price: c.high });
    if (isLow) lows.push({ index: i, price: c.low });
    if (highs.length >= 4 && lows.length >= 4) break;
  }
  return {
    lastHigh: highs[0]?.price ?? null,
    lastLow: lows[0]?.price ?? null,
    prevHigh: highs[1]?.price ?? null,
    prevLow: lows[1]?.price ?? null,
    highs,
    lows,
  };
}

// Calculate the anticipated price at which EMA 9 will cross EMA 21
// DIAGNOSTIC ONLY — do NOT use as an entry (see fibPullbackZone below).
function calculateAnticipatedCrossoverPrice(ema9, ema21) {
  const alpha9 = 2 / (FAST_PERIOD + 1);   // 0.20
  const alpha21 = 2 / (SLOW_PERIOD + 1); // 0.090909
  const numerator = ema21 * (1 - alpha21) - ema9 * (1 - alpha9);
  const denominator = alpha9 - alpha21;
  if (denominator === 0) return null;
  return numerator / denominator;
}

// ---------------------------------------------------------------------------
// Structure-anchored pullback zone (the fix for EMA-spot entries).
//
// Why the old "EMA21 spot + one-step crossover price" fails in live markets:
//  1. EMA21 is a MOVING target. A resting limit at "where EMA21 is now"
//     fills late or on the wrong side, because EMA21 itself migrates toward
//     price during the pullback. Spot prints of dynamic averages are stale
//     the moment after they are read.
//  2. The one-step crossover price answers "what single close would force
//     EMA9 == EMA21 on the NEXT bar?" In a healthy trend that print sits
//     absurdly far from market (e.g. EMA9=1.1000, EMA21=1.0980 → P*≈1.0833,
//     ~8x ATR away). That is a crash print, not a pullback level — price only
//     ever gets there on a full reversal, at which point a long limit is the
//     worst possible fill.
// What the market DOES respect: prior impulse structure. Pullbacks terminate
// at Fib retracements of the leg that created the trend (50-61.8% is where
// profit-taking exhausts and trend followers re-enter), confirmed by the
// slower average passing through the same band. So:
//  - ANCHOR = 50-61.8% Fib band of the last 30M impulse leg (prevLow→lastHigh
//    for longs, prevHigh→lastLow for shorts). Entry at the deeper (61.8%) edge.
//  - FILTER = 30M EMA21 must lie inside/near the band (±0.15 ATR) for full
//    confidence; a Fib zone without EMA confluence still trades, flagged.
//  - INVALIDATION = 78.6% retracement (close beyond = leg failed, stand down).
//    SL sits beyond invalidation + buffer, capped at MAX_SL_ATR_MULT.
//  - TP1 = impulse extreme (pays only if RR >= MIN_RR, else 1.5R projection),
//    TP2 = 2.5R runner (unchanged runner logic).
// Returns null when nothing tradeable exists (leg too small = Fib ratios are
// noise; zone >2 ATR away = chasing; RR < MIN_RR = bad business). Null means
// "publish NO pullback level this run" — strictly better than a fake level.
// All distances are ATR-relative, so this holds across XAUUSD, FX and crypto.
// ---------------------------------------------------------------------------
function fibPullbackZone({ swings, macroSide, emaSlow, atr, price }) {
  const isLong = macroSide === "BUY";
  const legStart = isLong ? swings.prevLow : swings.prevHigh;
  const legEnd = isLong ? swings.lastHigh : swings.lastLow;
  const legRange = (legStart != null && legEnd != null) ? Math.abs(legEnd - legStart) : 0;

  // --- Path 1 (preferred): real impulse structure ---------------------------
  // Leg must span >= 1 ATR or the Fib ratios slice noise, not structure.
  const legOk = legStart != null && legEnd != null &&
    (isLong ? legEnd > legStart : legStart > legEnd) && legRange >= atr;
  if (legOk) {
    const lvl = (pct) => isLong ? legEnd - pct * legRange : legEnd + pct * legRange;
    const fib50 = lvl(0.50);
    const fib618 = lvl(0.618);
    const fib786 = lvl(0.786);
    const entry = fib618; // deeper edge = better R:R; confirmation trigger forbids front-running it
    const invalidation = fib786;

    const lo = Math.min(fib50, fib618);
    const hi = Math.max(fib50, fib618);
    const tol = 0.15 * atr;
    const confluence = emaSlow != null && emaSlow >= lo - tol && emaSlow <= hi + tol;

    let slDist = Math.abs(entry - invalidation) + 0.15 * atr;
    slDist = Math.min(slDist, MAX_SL_ATR_MULT * atr); // cap (existing regime)
    slDist = Math.max(slDist, 0.75 * atr);             // floor (survive noise)
    const sl = isLong ? entry - slDist : entry + slDist;

    // TP1 pays at the impulse extreme; if the extreme is too close to pay
    // MIN_RR, there is no business case — fall back to a 1.5R projection so
    // reported RR never advertises a sub-standard trade.
    const extremePays = Math.abs(legEnd - entry) >= MIN_RR * slDist;
    const tp1 = extremePays
      ? legEnd
      : (isLong ? entry + 1.5 * slDist : entry - 1.5 * slDist);
    const tp2 = isLong ? entry + 2.5 * slDist : entry - 2.5 * slDist;
    const rr = slDist > 0 ? Math.abs(tp1 - entry) / slDist : 0;

    if (Math.abs(entry - price) > 2 * atr) return null; // too far — chasing, not a setup
    // Epsilon: a computed rr of exactly MIN_RR (e.g. the 1.5R projection)
    // can land at 1.4999999999999998 in floating point — that is NOT a fail.
    if (rr < MIN_RR - 1e-9) return null;                 // bad business — no level

    return {
      entry, zoneLow: lo, zoneHigh: hi, fib50, fib618, fib786, invalidation,
      sl, tp1, tp2, rr, confluence, weak: false, legStart, legEnd,
    };
  }

  // --- Path 2 (weak fallback): trend too young for a swing leg ---------------
  // EMA21 band, explicitly flagged weak so the brain treats it as a WATCH
  // level, never a resting-limit grade zone.
  if (emaSlow == null) return null;
  const slDist = ATR_SL_MULT * atr;
  const entry = emaSlow;
  if (Math.abs(entry - price) > 2 * atr) return null;
  const sl = isLong ? entry - slDist : entry + slDist;
  return {
    entry,
    zoneLow: entry - 0.25 * atr, zoneHigh: entry + 0.25 * atr,
    fib50: null, fib618: null, fib786: null, invalidation: sl,
    sl,
    tp1: isLong ? entry + 1.5 * slDist : entry - 1.5 * slDist,
    tp2: isLong ? entry + 2.5 * slDist : entry - 2.5 * slDist,
    rr: 1.5, confluence: false, weak: true, legStart: null, legEnd: null,
  };
}

// ---------------------------------------------------------------------------
// RANGE ENGINE — for choppy markets where the trend engine must stand down.
//
// When 4H is NEUTRAL the worst thing a trend system can do is keep trading.
// This module fades PROVEN range edges instead. Rules, in order (all must
// pass — a range fade with one missing ingredient is just a guess):
//  1. PROVEN EDGES: >=2 swing highs within 0.25 ATR of the ceiling AND >=2
//     swing lows within 0.25 ATR of the floor. One touch is noise.
//  2. TRADEABLE WIDTH: 1.5–6 ATR. Tighter and costs eat the move (a 1R gross
//     win is ~0.5R net on XAUUSD); wider and it is a trend leg, not a range.
//  3. EMA COMPRESSION (relative): |EMA9 − EMA21| under 25% of range width and
//     under 0.5 ATR. Judged against the range because the drift to the edge
//     always opens the averages somewhat; a real trend leg blows past both.
//  4. PRICE AT AN EDGE: within 0.3 ATR of floor (fade up) or ceiling (fade
//     down). Mid-range is a no-trade — never chase the middle.
//  5. RSI CONFIRMATION: fade up needs RSI <= 55 (no buying into a rocket),
//     fade down needs RSI >= 45. Mild by design — extremes rarely print, and
//     waiting for RSI<30 at range lows misses most valid fades.
// Usage: limit order AT the edge (not market mid-range), SL beyond the edge +
// 0.35 ATR buffer, TP1 = midline, TP2 = opposite edge. INVALIDATION = 30M
// close beyond the edge + buffer: the range is broken, stand down — never
// flip to breakout-chasing inside this module.
// Returns { range:true, tradable, ... } — tradable:false with a `why` string
// still documents the range so the brain can watch it without trading it.
// ---------------------------------------------------------------------------
function detectRange({ swings, atr, emaFast, emaSlow, rsiNow, price }) {
  const highs = ((swings.highs || []).map(h => h.price)).slice(0, 4);
  const lows = ((swings.lows || []).map(l => l.price)).slice(0, 4);
  if (highs.length < 2 || lows.length < 2) return null; // not enough structure to judge
  const upper = Math.max(...highs);
  const lower = Math.min(...lows);
  if (!(upper > lower)) return null;
  const width = upper - lower;
  const base = { range: true, upper, lower, width, widthAtr: width / atr };
  if (width < 1.5 * atr) return { ...base, tradable: false, why: `range too tight (${(width / atr).toFixed(1)} ATR) — costs would eat the move` };
  if (width > 6 * atr) return { ...base, tradable: false, why: `range too wide (${(width / atr).toFixed(1)} ATR) — this is a trend leg, not a range` };
  const tol = 0.25 * atr;
  const touchesUp = highs.filter(h => Math.abs(h - upper) <= tol).length;
  const touchesLo = lows.filter(l => Math.abs(l - lower) <= tol).length;
  if (touchesUp < 2 || touchesLo < 2) {
    return { ...base, tradable: false, touchesUp, touchesLo, why: `edges unproven (${touchesUp} ceiling / ${touchesLo} floor touches — need 2+ each)` };
  }
  // EMA compression (relative, not absolute): the drift to the edge always
  // separates the averages somewhat, so the knot is judged against the range
  // itself — gap must be under 25% of width AND under 0.5 ATR. A genuine trend
  // leg opens the gap far beyond both within a few bars.
  const maxGap = Math.min(0.5 * atr, 0.25 * width);
  if (Math.abs(emaFast - emaSlow) > maxGap) {
    return { ...base, tradable: false, touchesUp, touchesLo, why: "EMAs apart — trending, not ranging" };
  }
  const edgeTol = 0.3 * atr;
  const atLower = Math.abs(price - lower) <= edgeTol;
  const atUpper = Math.abs(price - upper) <= edgeTol;
  if (!atLower && !atUpper) {
    return { ...base, tradable: false, touchesUp, touchesLo, why: "price mid-range — never chase the middle, wait for an edge" };
  }
  const side = atLower ? "BUY" : "SELL";
  if (side === "BUY" && rsiNow > 55) {
    return { ...base, tradable: false, touchesUp, touchesLo, why: `RSI ${rsiNow.toFixed(0)} too hot to fade up from the floor` };
  }
  if (side === "SELL" && rsiNow < 45) {
    return { ...base, tradable: false, touchesUp, touchesLo, why: `RSI ${rsiNow.toFixed(0)} too cold to fade down from the ceiling` };
  }
  const edge = side === "BUY" ? lower : upper;
  const sl = side === "BUY" ? edge - 0.35 * atr : edge + 0.35 * atr;
  const slDist = Math.abs(edge - sl); // = 0.35 ATR by construction
  const mid = (upper + lower) / 2;
  const opposite = side === "BUY" ? upper : lower;
  const tp1 = mid;
  const tp2 = opposite;
  const rr = slDist > 0 ? Math.abs(tp1 - edge) / slDist : 0;
  if (rr < MIN_RR - 1e-9) return { ...base, tradable: false, touchesUp, touchesLo, why: "geometry pays below minimum RR" };
  return {
    ...base, tradable: true, side, touchesUp, touchesLo,
    edge, entry: edge, sl, tp1, tp2, rr, mid,
    invalidation: sl, // 30M close beyond edge+buffer = range broken
  };
}

function sessionStatus(symbol) {
  if (CRYPTO_SET.has(symbol)) return { ok: true, reason: "Crypto — 24/7 market" };
  const d = new Date();
  const day = d.getUTCDay(), hour = d.getUTCHours();
  if (WEEKEND_DAYS.has(day)) return { ok: false, reason: "Weekend (GMT)" };
  if (hour < SESSION_START_HOUR || hour >= SESSION_END_HOUR) {
    return { ok: false, reason: `Outside London/NY session (${SESSION_START_HOUR}:00-${SESSION_END_HOUR}:00 GMT)` };
  }
  return { ok: true, reason: "London/NY session active" };
}

// ---------------------------------------------------------------------------
// Engine Implementation
// ---------------------------------------------------------------------------
export class MTFStrategyEngine {
  constructor(deps = {}) {
    this.fetchCandles = deps.fetchCandles;
    this.checkNewsFilter = deps.checkNewsFilter;
    this.calculateLotSize = deps.calculateLotSize;
    this.addTradeMemory = deps.addTradeMemory;
    // Optional: pullback journal recorder (journal.js via index.js). Every
    // published actionable setup is recorded for accuracy scoring.
    this.recordSetup = deps.recordSetup || null;
    this._cache = new Map();
    this._requestTimes = [];
    this.stats = { api_calls: 0, cache_hits: 0, rate_limited: 0, last_run: null };
  }

  _budgetAvailable() {
    const now = Date.now();
    this._requestTimes = this._requestTimes.filter(t => now - t < 60_000);
    return this._requestTimes.length < MAX_REQUESTS_PER_MINUTE;
  }

  _registerRequest() {
    this._requestTimes.push(Date.now());
    this.stats.api_calls++;
  }

  async _candles(symbol, interval, size) {
    const key = `${symbol}:${interval}:${size}`;
    const hit = this._cache.get(key);
    const ttl = CACHE_TTL_MS[interval] || 15 * 60 * 1000;
    if (hit && Date.now() - hit.ts < ttl) {
      this.stats.cache_hits++;
      return hit.data;
    }
    if (!this._budgetAvailable()) {
      this.stats.rate_limited++;
      return hit ? hit.data : null;
    }
    this._registerRequest();
    const data = await this.fetchCandles(symbol, interval, size);
    if (data && data.length >= 20) this._cache.set(key, { ts: Date.now(), data });
    return data;
  }

  async analyze(symbol, options = {}) {
    const sym = String(symbol || "").toUpperCase();
    const t0 = Date.now();

    // Primary: 30m (120 candles = 60 hours) | Confirmation: 4h (80 candles = ~13 days)
    const [candles30M, candles4H] = await Promise.all([
      this._candles(sym, "30m", 120),
      this._candles(sym, "4h", 80),
    ]);

    const need30M = SLOW_PERIOD + RSI_PERIOD + VOL_SMA_PERIOD + 10;
    const need4H = Math.max(SLOW_PERIOD + 5, ADX_PERIOD * 2 + 5); // ADX needs 2x period to warm up
    if (!candles30M || candles30M.length < need30M || !candles4H || candles4H.length < need4H) {
      return {
        symbol: sym, decision: "DATA_UNAVAILABLE",
        reason: "Insufficient candle data (need 30M + 4H history)",
        strategy: "ma_rsi_30m_v2", timestamp: Date.now(), elapsed_ms: Date.now() - t0,
      };
    }

    const price = candles30M.at(-1).close;
    const dp = sym === "XAUUSD" || sym === "XAGUSD" ? 2 : CRYPTO_SET.has(sym) ? 2 : 5;
    const fmt = (x, d = dp) => (x == null ? null : Number(x.toFixed(d)));

    // =========================================================================
    // 1. 4H Confirmation Timeframe (Macro Direction)
    // =========================================================================
    const closes4H = candles4H.map(c => c.close);
    const ema4hFast = emaSeries(closes4H, FAST_PERIOD).at(-1);
    const ema4hSlow = emaSeries(closes4H, SLOW_PERIOD).at(-1);
    const rsi4h = rsiSeries(closes4H, RSI_PERIOD).at(-1) ?? 50;
    const { adx: adx4hSeries, diPlus: diPlus4hSeries, diMinus: diMinus4hSeries } = adxSeries(candles4H, ADX_PERIOD);
    const adx4h = adx4hSeries.at(-1) ?? 0;
    const diPlus4h = diPlus4hSeries.at(-1) ?? 0;
    const diMinus4h = diMinus4hSeries.at(-1) ?? 0;
    // ADX judged against THIS symbol's own recent range, not a fixed number —
    // see ADX_STRONG_PERCENTILE comment above.
    const adxPercentile4h = percentileRank(adx4hSeries, 100);
    const diSpread4hSeries = diPlus4hSeries.map((v, i) => Math.abs(v - diMinus4hSeries[i]));

    const ema4hBullish = ema4hFast > ema4hSlow;
    const ema4hBearish = ema4hFast < ema4hSlow;
    const rsi4hBullish = rsi4h > 50;
    const diBullish4h = diPlus4h > diMinus4h;
    const trendStrong4h = adxPercentile4h >= ADX_STRONG_PERCENTILE;

    // 4H trend requires EMA + RSI + DI to all agree, AND ADX to confirm the
    // trend has real strength relative to this symbol's own history. Any
    // disagreement, or a weak/choppy ADX regime, collapses to NEUTRAL —
    // stricter than EMA alone, which is the point: fewer, more trustworthy
    // 4H trend calls feeding the 30M execution layer.
    const macroBullish = trendStrong4h && ema4hBullish && rsi4hBullish && diBullish4h;
    const macroBearish = trendStrong4h && ema4hBearish && !rsi4hBullish && !diBullish4h;
    const macroTrend = macroBullish ? "BULLISH" : macroBearish ? "BEARISH" : "NEUTRAL";

    // =========================================================================
    // 2. 30M Primary Timeframe (Persistent Alignment + Crossover State)
    // =========================================================================
    const closes30M = candles30M.map(c => c.close);
    const emaFast30MSeries = emaSeries(closes30M, FAST_PERIOD);
    const emaSlow30MSeries = emaSeries(closes30M, SLOW_PERIOD);
    const n = emaFast30MSeries.length;

    const ema30mFast = emaFast30MSeries[n - 1];
    const ema30mSlow = emaSlow30MSeries[n - 1];

    // Current state (persistent - does not need to cross right now)
    const primaryBullish = ema30mFast > ema30mSlow;
    const primaryBearish = ema30mFast < ema30mSlow;

    // Fresh crossover on the most recent 1-2 candles
    const freshCrossUp = emaFast30MSeries[n - 2] <= emaSlow30MSeries[n - 2] && emaFast30MSeries[n - 1] > emaSlow30MSeries[n - 1];
    const freshCrossDown = emaFast30MSeries[n - 2] >= emaSlow30MSeries[n - 2] && emaFast30MSeries[n - 1] < emaSlow30MSeries[n - 1];

    // Anticipated Crossover Price Level
    const anticipatedCrossoverPrice = calculateAnticipatedCrossoverPrice(ema30mFast, ema30mSlow);

    // =========================================================================
    // 3. Volatility, Swings & Pullback Health (Zero Extra API Calls)
    // =========================================================================
    const atr30M = atr(candles30M, ATR_PERIOD) || 1.0;
    const swings = detectSwings(candles30M, 3);
    const rsiNow = rsiSeries(closes30M, RSI_PERIOD).at(-1) || 50;

    // Volume Health on 30M
    const volumes30M = candles30M.map(c => c.volume || 0);
    const volSma = smaLast(volumes30M, VOL_SMA_PERIOD) || 1;
    const lastVol = volumes30M.at(-1);
    const isCounterTrendVolLow = lastVol <= volSma * 1.1; // low volume counter-move = healthy pullback

    // Pullback Exhaustion & Integrity Assessment
    let pullbackStatus = "IN_TREND";
    let isPullbackHealthy = true;
    let isRealReversal = false;
    let reversalSide = null;

    // Pullback vs. real-reversal probability, using only self-relative
    // (percentile / ratio) features so this scoring holds across every pair
    // FRIT trades — no per-symbol threshold retuning needed.
    let pullbackProb = null, reversalProb = null;

    if (macroBearish && primaryBullish) {
      pullbackStatus = "BULLISH_PULLBACK_IN_BEAR_TREND";
      const brokeSwingHigh = swings.lastHigh != null && price > swings.lastHigh;
      const retracePct = computeRetracePct(swings, price, "BEARISH");
      const scored = scorePullback({
        adxPercentile: adxPercentile4h,
        diSpreadPercentile: percentileRank(diSpread4hSeries, 100),
        retracePct,
        volRatio: lastVol / volSma,
      });
      pullbackProb = scored.pullbackProb;
      reversalProb = scored.reversalProb;

      if (brokeSwingHigh || reversalProb >= REVERSAL_PROB_THRESHOLD) {
        isPullbackHealthy = false;
        isRealReversal = true;
        reversalSide = "BUY";
        pullbackStatus = "INVALID_PULLBACK_REAL_BULLISH_REVERSAL";
      } else {
        isPullbackHealthy = pullbackProb >= PULLBACK_PROB_THRESHOLD;
      }
    } else if (macroBullish && primaryBearish) {
      pullbackStatus = "BEARISH_PULLBACK_IN_BULL_TREND";
      const brokeSwingLow = swings.lastLow != null && price < swings.lastLow;
      const retracePct = computeRetracePct(swings, price, "BULLISH");
      const scored = scorePullback({
        adxPercentile: adxPercentile4h,
        diSpreadPercentile: percentileRank(diSpread4hSeries, 100),
        retracePct,
        volRatio: lastVol / volSma,
      });
      pullbackProb = scored.pullbackProb;
      reversalProb = scored.reversalProb;

      if (brokeSwingLow || reversalProb >= REVERSAL_PROB_THRESHOLD) {
        isPullbackHealthy = false;
        isRealReversal = true;
        reversalSide = "SELL";
        pullbackStatus = "INVALID_PULLBACK_REAL_BEARISH_REVERSAL";
      } else {
        isPullbackHealthy = pullbackProb >= PULLBACK_PROB_THRESHOLD;
      }
    }

    // =========================================================================
    // 4. Session & News Advisory
    // =========================================================================
    const session = sessionStatus(sym);
    let news = { blocked: false, has_news: false };
    if (this.checkNewsFilter) {
      try { news = await this.checkNewsFilter(sym); } catch { news = { blocked: false, has_news: false }; }
    }

    // =========================================================================
    // 5. Scenarios: Immediate Entry vs Pullback/Anticipated Re-entry
    // =========================================================================
    const isAligned = (macroBullish && primaryBullish) || (macroBearish && primaryBearish);
    const activeSide = macroBullish ? "BUY" : macroBearish ? "SELL" : "NEUTRAL";

    let scenario1 = null;
    let scenario2 = null;
    let scenarioRange = null;
    let rangeInfo = null;

    if (isRealReversal && reversalSide) {
      // RULE: When the pullback checker observes that a real reversal move is playing out,
      // no new trend-continuation entry will occur. Dual scenarios are canceled and
      // replaced with EXACTLY 1 IMMEDIATE ENTRY in the reversal direction.
      const isLong = reversalSide === "BUY";
      const slDist = Math.min(MAX_SL_ATR_MULT * atr30M, Math.max(ATR_SL_MULT * atr30M, isLong ? (swings.lastLow ? price - swings.lastLow + 0.1 * atr30M : ATR_SL_MULT * atr30M) : (swings.lastHigh ? swings.lastHigh - price + 0.1 * atr30M : ATR_SL_MULT * atr30M)));
      const sl = isLong ? price - slDist : price + slDist;
      const tp1 = isLong ? price + 1.8 * slDist : price - 1.8 * slDist;
      const tp2 = isLong ? price + 2.5 * slDist : price - 2.5 * slDist;
      const rr = slDist > 0 ? Math.abs(tp1 - price) / slDist : 1.8;

      scenario1 = {
        name: `Immediate ${reversalSide} Real Reversal Entry (Single Scenario)`,
        action: reversalSide,
        entry: fmt(price),
        sl: fmt(sl),
        tp1: fmt(tp1),
        tp2: fmt(tp2),
        rr: Number(rr.toFixed(2)),
        condition: "Pullback invalidated: aggressive volume & structure break confirm real reversal. Dual scenarios canceled.",
      };
      scenario2 = null; // No second scenario: old trend is broken!
    } else {
      // Normal flow:
      // Scenario 1: Immediate Entry (when 30M is aligned with 4H)
      if (isAligned && activeSide !== "NEUTRAL") {
        const isLong = activeSide === "BUY";
        const slDist = Math.min(MAX_SL_ATR_MULT * atr30M, Math.max(ATR_SL_MULT * atr30M, isLong ? (swings.lastLow ? price - swings.lastLow + 0.1 * atr30M : ATR_SL_MULT * atr30M) : (swings.lastHigh ? swings.lastHigh - price + 0.1 * atr30M : ATR_SL_MULT * atr30M)));
        const sl = isLong ? price - slDist : price + slDist;

        const tp1Structural = isLong ? (swings.lastHigh && swings.lastHigh > price ? swings.lastHigh : price + 1.5 * slDist) : (swings.lastLow && swings.lastLow < price ? swings.lastLow : price - 1.5 * slDist);
        const tp1 = isLong ? Math.min(tp1Structural, price + 1.8 * slDist) : Math.max(tp1Structural, price - 1.8 * slDist);
        const tp2 = isLong ? price + 2.5 * slDist : price - 2.5 * slDist;
        const rr = slDist > 0 ? Math.abs(tp1 - price) / slDist : 1.5;

        scenario1 = {
          name: "Immediate Momentum Entry",
          action: activeSide,
          entry: fmt(price),
          sl: fmt(sl),
          tp1: fmt(tp1),
          tp2: fmt(tp2),
          rr: Number(rr.toFixed(2)),
          condition: freshCrossUp || freshCrossDown ? "Fresh 30M EMA cross confirmed" : "30M EMA persistent alignment with 4H trend",
        };
      }

      // Scenario 2: structure-anchored Fib pullback zone (only if healthy).
      // fibPullbackZone returns null when no tradeable structure exists — in
      // that case NO level is published (a missing level beats a fake one).
      if (activeSide !== "NEUTRAL" && isPullbackHealthy) {
        const isLong = activeSide === "BUY";
        const zone = fibPullbackZone({
          swings, macroSide: activeSide, emaSlow: ema30mSlow, atr: atr30M, price,
        });
        // One-step crossover math kept as a diagnostic only — never an entry.
        const estCross = anticipatedCrossoverPrice ? fmt(anticipatedCrossoverPrice) : null;

        if (zone != null) {
          const band = `${fmt(Math.min(zone.zoneLow, zone.zoneHigh))} – ${fmt(Math.max(zone.zoneLow, zone.zoneHigh))}`;
          scenario2 = {
            name: zone.weak ? "Pullback Watch (EMA21, no structure yet)" : "Pullback / Fib Re-entry",
            action: activeSide,
            // Headline entry is the ZONE edge — NOT current price. result.entry
            // picks this up via primaryScenario?.entry, so WAIT_PULLBACK
            // reports "wait at the zone" instead of market.
            entry: fmt(zone.entry),
            entry_band: band,
            watch_zone: zone.weak
              ? `${fmt(ema30mSlow)} (30M EMA 21 — no swing leg yet, WATCH ONLY, not a limit grade level)`
              : `${band} (50-61.8% Fib of ${fmt(zone.legStart)}→${fmt(zone.legEnd)}${zone.confluence ? `, EMA21 confluence @${fmt(ema30mSlow)}` : ", no EMA confluence — Fib structure only"})`,
            anticipated_crossover_level: estCross,
            confirmation_trigger: isLong
              ? `Wait for 30M rejection INSIDE ${band} and EMA 9 curving back above EMA 21 — do NOT front-run the band edge, do NOT market-buy into the fall`
              : `Wait for 30M rejection INSIDE ${band} and EMA 9 curving back below EMA 21 — do NOT front-run the band edge, do NOT market-sell into the rally`,
            invalidation: `Zone dead on 30M CLOSE beyond ${fmt(zone.invalidation)} — stand down and wait for new structure, never average down`,
            estimated_sl: fmt(zone.sl),
            estimated_tp1: fmt(zone.tp1),
            estimated_tp2: fmt(zone.tp2),
            rr: Number(zone.rr.toFixed(2)),
            fib_confluence: zone.confluence,
            zone_weak: zone.weak,
            pullback_health: {
              status: pullbackStatus,
              is_healthy: isPullbackHealthy,
              volume_ok: isCounterTrendVolLow,
              pullback_probability: pullbackProb,
              reversal_probability: reversalProb,
              adx_percentile_4h: Number(adxPercentile4h.toFixed(1)),
              note: pullbackProb != null
                ? `Pullback probability ${pullbackProb}% vs reversal ${reversalProb}%, based on 4H ADX/DI strength (percentile-ranked per symbol) and retracement depth.` +
                  (zone.weak ? " Zone is EMA-watch only (no swing leg to anchor Fib yet)." : ` Fib zone ${band}${zone.confluence ? " with EMA21 confluence." : " (Fib structure, no EMA confluence)."}`)
                : "Not enough swing structure to score retracement depth yet.",
            },
          };
        } else {
          scenario2 = null; // untradeable zone (too far / no structure / RR < 1.5) — never chase
        }
      }
    }

    // ---- Range module: only when the trend engine has nothing (macro NEUTRAL,
    // no reversal in progress). Costs gate applies identically: a fade whose
    // net RR is negative after spread/commission/slippage is not published.
    if (macroTrend === "NEUTRAL" && !isRealReversal) {
      rangeInfo = detectRange({
        swings, atr: atr30M, emaFast: ema30mFast, emaSlow: ema30mSlow,
        rsiNow, price,
      });
      if (rangeInfo && rangeInfo.tradable) {
        const rc = applyCosts({ symbol: sym, entry: rangeInfo.entry, sl: rangeInfo.sl, tp: rangeInfo.tp1 });
        if (!rc.blocked) {
          const rBand = `${fmt(rangeInfo.edge)} edge (±${fmt(0.3 * atr30M)})`;
          scenarioRange = {
            name: "Range Fade (limit at proven edge)",
            action: rangeInfo.side,
            entry: fmt(rangeInfo.entry),
            entry_band: rBand,
            watch_zone: `${rBand} of ${fmt(rangeInfo.lower)} – ${fmt(rangeInfo.upper)} (${rangeInfo.touchesLo} floor / ${rangeInfo.touchesUp} ceiling touches, width ${(rangeInfo.width / atr30M).toFixed(1)} ATR)`,
            confirmation_trigger: rangeInfo.side === "BUY"
              ? `Limit buy AT the floor ${fmt(rangeInfo.edge)} on 30M rejection (wick + RSI holding) — never market-buy mid-range`
              : `Limit sell AT the ceiling ${fmt(rangeInfo.edge)} on 30M rejection (wick + RSI holding) — never market-sell mid-range`,
            invalidation: `Range dead on 30M CLOSE beyond ${fmt(rangeInfo.invalidation)} — stand down, do NOT flip to breakout-chasing`,
            estimated_sl: fmt(rangeInfo.sl),
            estimated_tp1: fmt(rangeInfo.tp1),
            estimated_tp2: fmt(rangeInfo.tp2),
            rr: Number(rangeInfo.rr.toFixed(2)),
            costs: rc,
            range_touches: `${rangeInfo.touchesLo}L/${rangeInfo.touchesUp}U`,
            range_width_atr: Number(rangeInfo.widthAtr.toFixed(1)),
          };
          if (this.recordSetup) {
            try {
              this.recordSetup({
                symbol: sym, strategy: "range_fade", direction: rangeInfo.side,
                entry: rangeInfo.entry, zoneLow: Math.min(rangeInfo.edge - 0.3 * atr30M, rangeInfo.edge + 0.3 * atr30M),
                zoneHigh: Math.max(rangeInfo.edge - 0.3 * atr30M, rangeInfo.edge + 0.3 * atr30M),
                sl: rangeInfo.sl, tp1: rangeInfo.tp1, plannedRR: rangeInfo.rr,
                confluence: true, pullbackProb: null,
                note: `range ${fmt(rangeInfo.lower)}-${fmt(rangeInfo.upper)} ${rangeInfo.touchesLo}L/${rangeInfo.touchesUp}U touches`,
              });
            } catch { /* journal best-effort */ }
          }
        } else {
          rangeInfo = { ...rangeInfo, tradable: false, why: `costs eat the fade (net RR ${rc.netRR} — spread/slip wider than the edge geometry pays)` };
        }
      }
    }

    // ---- Cost gate on the trend pullback zone too (same honesty rule). ----
    if (scenario2 && !scenario2.zone_weak) {
      const tc = applyCosts({ symbol: sym, entry: Number(scenario2.entry), sl: Number(scenario2.estimated_sl), tp: Number(scenario2.estimated_tp1) });
      scenario2.costs = tc;
      if (tc.blocked) {
        scenario2 = null; // negative after costs even when right — not a setup
      } else if (this.recordSetup) {
        try {
          this.recordSetup({
            symbol: sym, strategy: "trend_pullback",
            direction: scenario2.action, entry: Number(scenario2.entry),
            zoneLow: scenario2.entry_band ? Number(String(scenario2.entry_band).split("–")[0]) : null,
            zoneHigh: scenario2.entry_band ? Number(String(scenario2.entry_band).split("–")[1]) : null,
            sl: Number(scenario2.estimated_sl), tp1: Number(scenario2.estimated_tp1),
            plannedRR: scenario2.rr, confluence: !!scenario2.fib_confluence,
            pullbackProb, note: `4H ${macroTrend}, pullback ${pullbackProb ?? "n/a"}%`,
          });
        } catch { /* journal best-effort */ }
      }
    } else if (scenario2 && this.recordSetup) {
      scenario2.costs = applyCosts({ symbol: sym, entry: Number(scenario2.entry), sl: Number(scenario2.estimated_sl), tp: Number(scenario2.estimated_tp1) });
    }

    // ---- Cost gate on immediate (market) entries: BUY/SELL execute at
    // market, so a blocked net RR downgrades the whole decision to WAIT. ----
    if (scenario1) {
      scenario1.costs = applyCosts({ symbol: sym, entry: Number(scenario1.entry), sl: Number(scenario1.sl), tp: Number(scenario1.tp1) });
    }

    // Determine primary decision output
    let decision = "WAIT";
    let conf = 50;
    const reasons = [`4H Macro Trend: ${macroTrend} (EMA9/21 + RSI50 + DI+/DI-, gated by ADX percentile ${adxPercentile4h.toFixed(0)})`];

    if (isRealReversal && reversalSide) {
      decision = reversalSide;
      conf = 68;
      reasons.push(`Pullback Invalidated: Real ${reversalSide} Reversal confirmed on 30M structure/volume.`);
      reasons.push("Old trend broken — no pullback re-entry will occur. 1 single immediate reversal entry provided.");
      if (session.ok) conf += 5;
    } else if (isAligned && activeSide !== "NEUTRAL") {
      decision = activeSide;
      conf = 65;
      reasons.push(`30M Primary Trend: Aligned with 4H (${activeSide})`);
      if (freshCrossUp || freshCrossDown) { conf += 15; reasons.push("Fresh 30M EMA crossover just occurred"); }
      else { conf += 10; reasons.push("Persistent 30M EMA directional alignment"); }
      if (adxPercentile4h >= ADX_STRONG_PERCENTILE) { conf += 8; reasons.push(`4H ADX percentile ${adxPercentile4h.toFixed(0)} — strong trend for this pair`); }
      if (session.ok) { conf += 5; reasons.push(session.reason); }
    } else if (activeSide !== "NEUTRAL") {
      decision = "WAIT_PULLBACK";
      conf = 45 + Math.round(((pullbackProb ?? 50) - 50) / 5); // scale slightly with pullback confidence
      // Zone quality moves the needle: Fib+EMA confluence earns confidence,
      // a weak EMA-watch zone (or no zone at all) surrenders it.
      if (scenario2) {
        if (!scenario2.zone_weak && scenario2.fib_confluence) { conf += 8; reasons.push("Fib 50-61.8% zone backed by EMA21 confluence — high-quality wait"); }
        else if (scenario2.zone_weak) { conf -= 8; reasons.push("No swing leg yet — EMA-watch only, NOT a limit grade level"); }
      } else {
        conf -= 5;
        reasons.push("No tradeable pullback zone right now (too far / sub-standard RR) — patience, not a market entry");
      }
      reasons.push(`4H is ${macroTrend}, but 30M is undergoing a healthy pullback (pullback probability ${pullbackProb ?? "n/a"}%). Refer to Scenario 2 for re-entry.`);
    } else if (scenarioRange) {
      // Macro NEUTRAL + proven range edges + price at an edge = fade the edge.
      decision = "WAIT_RANGE";
      conf = 55 + Math.min((rangeInfo.touchesUp + rangeInfo.touchesLo), 6);
      if ((rangeInfo.side === "BUY" && rsiNow < 35) || (rangeInfo.side === "SELL" && rsiNow > 65)) {
        conf += 5;
        reasons.push("RSI deeply stretched at the edge — exhaustion favors the fade");
      }
      if (scenarioRange.costs && scenarioRange.costs.weak) {
        conf -= 8;
        reasons.push(`Costs consume ${scenarioRange.costs.costPctOfRisk}% of risk (net RR ${scenarioRange.costs.netRR}) — fade only with full confluence`);
      }
      if (!session.ok) { conf -= 5; reasons.push("Outside liquid session — range edges are less trustworthy now"); }
      conf = Math.min(conf, 80);
      reasons.push(`4H NEUTRAL — trend engine stands down. Range ${fmt(rangeInfo.lower)} – ${fmt(rangeInfo.upper)} (${rangeInfo.touchesLo} floor / ${rangeInfo.touchesUp} ceiling touches). Price at the ${rangeInfo.side === "BUY" ? "floor" : "ceiling"}: limit-fade toward midline, invalidation on close beyond the edge.`);
    } else {
      if (rangeInfo && rangeInfo.range && rangeInfo.why) {
        reasons.push(`Range watched but NOT traded: ${rangeInfo.why}.`);
      } else {
        reasons.push("4H trend is neutral / transitioning.");
      }
    }

    // Costs kill market entries: an aligned/reversal BUY/SELL with netRR < 1
    // after spread/commission/slippage becomes WAIT — executing it would lock
    // in negative expectancy. Weak (not blocked) costs just cost confidence.
    if ((decision === "BUY" || decision === "SELL") && scenario1 && scenario1.costs) {
      if (scenario1.costs.blocked) {
        reasons.push(`Market entry blocked: costs consume ${scenario1.costs.costPctOfRisk ?? "?"}% of risk (net RR ${scenario1.costs.netRR}) — a win would still lose money. Standing down.`);
        decision = "WAIT";
        conf = 40;
      } else if (scenario1.costs.weak) {
        conf -= 8;
        reasons.push(`High friction: costs consume ${scenario1.costs.costPctOfRisk}% of risk (net RR ${scenario1.costs.netRR}) — size down or skip`);
      }
    }

    // News advisory adjustments
    if (news.has_news) {
      reasons.push(news.reason);
      if (news.recommendation) reasons.push(`Advisory: ${news.recommendation}`);
    }

    const primaryScenario = scenario1 || scenario2 || scenarioRange;
    const result = {
      symbol: sym,
      price: fmt(price),
      strategy: "ma_rsi_30m_v2",
      decision,
      direction: macroTrend,
      confidence: Math.max(20, Math.min(95, conf)),
      reasons,
      entry: primaryScenario?.entry ?? (scenario2 ? fmt(price) : null),
      sl: primaryScenario?.sl ?? scenario2?.estimated_sl ?? scenarioRange?.estimated_sl ?? null,
      tp: primaryScenario?.tp1 ?? scenario2?.estimated_tp1 ?? scenarioRange?.estimated_tp1 ?? null,
      tp2: primaryScenario?.tp2 ?? scenarioRange?.estimated_tp2 ?? null,
      rr: primaryScenario?.rr ?? 1.8,
      costs: primaryScenario?.costs ?? null, // net-RR after spread/commission/slippage (costs.js)
      // Explicit entry context for downstream consumers (/trade reason line,
      // position monitor). WAIT_* -> limit-style zone; BUY/SELL -> market.
      entry_ctx: {
        zone: decision === "WAIT_PULLBACK"
          ? (scenario2?.entry_band || scenario2?.watch_zone || null)
          : decision === "WAIT_RANGE"
            ? (scenarioRange?.entry_band || scenarioRange?.watch_zone || null)
            : ((decision === "BUY" || decision === "SELL") ? fmt(price) : null),
        type: decision === "WAIT_PULLBACK" ? "pullback_limit" : decision === "WAIT_RANGE" ? "range_limit" : "market",
      },
      scenarios: {
        scenario_1_immediate: scenario1,
        scenario_2_pullback_crossover: scenario2,
        scenario_range_fade: scenarioRange,
      },
      regime: {
        timeframe_primary: "30m",
        timeframe_confirm: "4h",
        macro_trend: macroTrend,
        macro_ema_fast: fmt(ema4hFast),
        macro_ema_slow: fmt(ema4hSlow),
        macro_rsi_4h: Number(rsi4h.toFixed(1)),
        macro_adx_4h: Number(adx4h.toFixed(1)),
        macro_adx_percentile_4h: Number(adxPercentile4h.toFixed(1)),
        macro_di_plus_4h: Number(diPlus4h.toFixed(1)),
        macro_di_minus_4h: Number(diMinus4h.toFixed(1)),
        primary_ema_fast: fmt(ema30mFast),
        primary_ema_slow: fmt(ema30mSlow),
        anticipated_crossover_price: fmt(anticipatedCrossoverPrice),
      },
      structure_30m: {
        last_swing_high: fmt(swings.lastHigh),
        last_swing_low: fmt(swings.lastLow),
        atr_30m: fmt(atr30M),
        rsi_30m: Number(rsiNow.toFixed(1)), // informational only — 30M decisions use EMA + pullback probability, not RSI
        pullback_status: pullbackStatus,
        pullback_healthy: isPullbackHealthy,
        pullback_probability: pullbackProb,
        reversal_probability: reversalProb,
        range_30m: rangeInfo && rangeInfo.range ? {
          upper: fmt(rangeInfo.upper),
          lower: fmt(rangeInfo.lower),
          width_atr: Number(rangeInfo.widthAtr.toFixed(1)),
          touches_ceiling: rangeInfo.touchesUp ?? null,
          touches_floor: rangeInfo.touchesLo ?? null,
          tradable: !!scenarioRange,
          why: rangeInfo.why || (scenarioRange ? "edges proven, price at edge, fading toward midline" : null),
        } : null,
      },
      guards: {
        session,
        news_advisory: news.has_news ? news.reason : "No high-impact news active",
        news_recommendation: news.recommendation ?? null,
      },
      timestamp: Date.now(),
      elapsed_ms: Date.now() - t0,
    };

    if (decision === "BUY" || decision === "SELL") {
      result.reason = `Aligned 4H + 30M ${macroTrend} setup. ${primaryScenario?.condition || ""}`;
    } else if (decision === "WAIT_PULLBACK") {
      result.reason = scenario2
        ? `4H is ${macroTrend}; 30M pullback in progress. Wait for the Fib zone ${scenario2.entry_band || scenario2.watch_zone} — ${scenario2.zone_weak ? "EMA-watch only, no limit order" : "limit at the zone edge on rejection, never chase"}. Dead on close beyond ${scenario2.invalidation ? scenario2.invalidation.replace(/^Zone dead on 30M CLOSE beyond /, "") : "?"}`
        : `4H is ${macroTrend}; 30M pullback in progress but no tradeable zone right now (too far to chase or RR < ${MIN_RR}) — stand by for new structure, do NOT market-enter.`;
    } else if (decision === "WAIT_RANGE") {
      result.reason = scenarioRange
        ? `4H NEUTRAL — range ${scenarioRange.watch_zone}. ${scenarioRange.confirmation_trigger}. ${scenarioRange.invalidation}.`
        : "4H NEUTRAL with range structure nearby — waiting for price to reach a proven edge.";
    } else {
      result.reason = reasons[reasons.length - 1] || "Waiting for clear alignment";
    }

    this.stats.last_run = result;
    return result;
  }

  async run(symbol, options = {}) {
    const sym = String(symbol || "").toUpperCase();
    const t0 = Date.now();
    const analysis = await this.analyze(sym, options);
    if (analysis.decision === "DATA_UNAVAILABLE") return analysis;

    const out = {
      decision: analysis.decision,
      symbol: sym,
      strategy: "ma_rsi_30m_v2",
      confidence: analysis.confidence,
      reason: analysis.reason ?? null,
      reasons: analysis.reasons ?? [],
      entry: analysis.entry,
      sl: analysis.sl,
      tp: analysis.tp,
      tp2: analysis.tp2,
      rr: analysis.rr,
      costs: analysis.costs ?? null,
      entry_ctx: analysis.entry_ctx ?? null,
      scenarios: analysis.scenarios,
      regime: analysis.regime,
      structure_30m: analysis.structure_30m,
      guards: analysis.guards,
      analysis,
      timestamp: Date.now(),
      elapsed_ms: Date.now() - t0,
    };

    if (analysis.decision === "BUY" || analysis.decision === "SELL") {
      out.lot_size = this.calculateLotSize
        ? this.calculateLotSize({
            symbol: sym,
            balance: options.balance || 1000,
            riskPercent: options.riskPercent || 1,
            entry: analysis.entry,
            stopLoss: analysis.sl,
          })
        : 0.01;

      if (this.addTradeMemory) {
        try {
          this.addTradeMemory(sym, {
            direction: analysis.decision,
            pattern: `ma_rsi_30m:${analysis.regime?.macro_trend ?? "?"}`,
            outcome: "pending",
            note: `30M/4H conf=${analysis.confidence}% rr=${analysis.rr} rsi=${analysis.structure_30m?.rsi_30m}`,
          });
        } catch { /* journal best-effort */ }
      }
    }
    return out;
  }

  cacheStatus() {
    return {
      strategy: "ma_rsi_30m_v2",
      api_calls: this.stats.api_calls,
      cache_hits: this.stats.cache_hits,
      rate_limited_calls: this.stats.rate_limited,
      cache_entries: this._cache.size,
      rate_budget: `${this._requestTimes.length}/${MAX_REQUESTS_PER_MINUTE} in last 60s`,
      last_run: this.stats.last_run
        ? {
            symbol: this.stats.last_run.symbol,
            decision: this.stats.last_run.decision,
            confidence: this.stats.last_run.confidence,
            macro_trend: this.stats.last_run.regime?.macro_trend,
          }
        : null,
    };
  }
}