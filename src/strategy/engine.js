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
//     2. Pullback Re-entry: Calculated target zone (50-61.8% Fib / EMA 21)
//        and anticipated crossover level before it happens.
// - Pullback Health  : Detects exhaustion vs real reversal using 30M volume,
//                      RSI, and swing structure without any extra API calls.
// - Hybrid TP/SL     :
//     - TP1: Nearest 30M swing liquidity pool capped at 1.5x ATR (high win-rate).
//     - TP2: 2.5x ATR runner.
//     - SL : Capped by ATR and anchored to recent 30M swing structure.
// - Zero Extra API Calls: Pullback health, structure breaks, and levels are
//   computed directly on the 30M array to preserve Twelve Data free tier limits.
// ============================================================================

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
function calculateAnticipatedCrossoverPrice(ema9, ema21) {
  const alpha9 = 2 / (FAST_PERIOD + 1);   // 0.20
  const alpha21 = 2 / (SLOW_PERIOD + 1); // 0.090909
  const numerator = ema21 * (1 - alpha21) - ema9 * (1 - alpha9);
  const denominator = alpha9 - alpha21;
  if (denominator === 0) return null;
  return numerator / denominator;
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

      // Scenario 2: Pullback & Anticipated Crossover Entry (only if pullback is healthy)
      if (activeSide !== "NEUTRAL" && isPullbackHealthy) {
        const isLong = activeSide === "BUY";
        const pullTargetPrice = fmt(ema30mSlow);
        const estCross = anticipatedCrossoverPrice ? fmt(anticipatedCrossoverPrice) : null;
        const slDist = ATR_SL_MULT * atr30M;
        const estSl = isLong ? (pullTargetPrice ? pullTargetPrice - slDist : price - slDist) : (pullTargetPrice ? pullTargetPrice + slDist : price + slDist);
        const estTp1 = isLong ? (estSl ? pullTargetPrice + 2 * slDist : price + 2 * slDist) : (estSl ? pullTargetPrice - 2 * slDist : price - 2 * slDist);

        scenario2 = {
          name: "Pullback / Anticipated Re-entry",
          action: activeSide,
          // Headline entry for a pullback setup is the ZONE itself (anticipated
          // crossover level, else the 30M EMA21 watch price) — NOT current
          // price. result.entry picks this up via primaryScenario?.entry, so
          // WAIT_PULLBACK now reports "wait at the zone" instead of market.
          entry: estCross ?? pullTargetPrice,
          watch_zone: `${fmt(ema30mSlow)} (30M EMA 21)`,
          anticipated_crossover_level: estCross,
          confirmation_trigger: isLong
            ? `Wait for 30M candle rejection at ~${pullTargetPrice} and EMA 9 curving back above EMA 21`
            : `Wait for 30M candle rejection at ~${pullTargetPrice} and EMA 9 curving back below EMA 21`,
          estimated_sl: fmt(estSl),
          estimated_tp1: fmt(estTp1),
          pullback_health: {
            status: pullbackStatus,
            is_healthy: isPullbackHealthy,
            volume_ok: isCounterTrendVolLow,
            pullback_probability: pullbackProb,
            reversal_probability: reversalProb,
            adx_percentile_4h: Number(adxPercentile4h.toFixed(1)),
            note: pullbackProb != null
              ? `Pullback probability ${pullbackProb}% vs reversal ${reversalProb}%, based on 4H ADX/DI strength (percentile-ranked per symbol) and retracement depth.`
              : "Not enough swing structure to score retracement depth yet.",
          },
        };
      }
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
      reasons.push(`4H is ${macroTrend}, but 30M is undergoing a healthy pullback (pullback probability ${pullbackProb ?? "n/a"}%). Refer to Scenario 2 for re-entry.`);
    } else {
      reasons.push("4H trend is neutral / transitioning.");
    }

    // News advisory adjustments
    if (news.has_news) {
      reasons.push(news.reason);
      if (news.recommendation) reasons.push(`Advisory: ${news.recommendation}`);
    }

    const primaryScenario = scenario1 || scenario2;
    const result = {
      symbol: sym,
      price: fmt(price),
      strategy: "ma_rsi_30m_v2",
      decision,
      direction: macroTrend,
      confidence: Math.max(20, Math.min(95, conf)),
      reasons,
      entry: primaryScenario?.entry ?? (scenario2 ? fmt(price) : null),
      sl: primaryScenario?.sl ?? scenario2?.estimated_sl ?? null,
      tp: primaryScenario?.tp1 ?? scenario2?.estimated_tp1 ?? null,
      tp2: primaryScenario?.tp2 ?? null,
      rr: primaryScenario?.rr ?? 1.8,
      // Explicit entry context for downstream consumers (/trade reason line,
      // position monitor). WAIT_PULLBACK -> the pullback zone as a limit-style
      // entry; BUY/SELL -> market at current price; else null.
      entry_ctx: {
        zone: decision === "WAIT_PULLBACK"
          ? (scenario2?.anticipated_crossover_level || scenario2?.watch_zone || fmt(price))
          : ((decision === "BUY" || decision === "SELL") ? fmt(price) : null),
        type: decision === "WAIT_PULLBACK" ? "pullback_limit" : "market",
      },
      scenarios: {
        scenario_1_immediate: scenario1,
        scenario_2_pullback_crossover: scenario2,
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
      result.reason = `4H is ${macroTrend}; 30M pullback in progress. Watch anticipated level ~${scenario2?.anticipated_crossover_level || scenario2?.watch_zone}`;
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