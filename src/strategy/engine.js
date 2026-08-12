// ============================================================================
// FRIT MTF STRATEGY ENGINE — "Build with accuracy, intelligence, directions"
// ----------------------------------------------------------------------------
// Multi-timeframe directional engine designed for Twelve Data FREE tier
// (8 credits/min, 800/day) — every request is cached with interval-appropriate
// TTLs and a sliding-window rate budget guards the 8/min limit.
//
// Design (low-lag by construction — see docs/STRATEGY.md):
//   Regime  (1D) : EMA50/200 + EMA200 slope  -> deliberately SLOW (regime gate)
//   Structure(4H): swing pivots -> HH/HL/LH/LL -> last swing stop
//   Entry   (1H) : Volume-Profile POC/VAH/VAL + rolling VWAP + ATR
//                  -> pullback into POC-VAL (long) / POC-VAH (short) zone
//                  -> rejection candle trigger (price action, not an indicator)
//   Guards       : news (live Forex Factory calendar), London/NY session,
//                  recent spike, extension > 2*ATR, RR >= 1.8, SL <= 2.5*ATR
//
// No TA library. Everything is plain math so the build stays dependency-free.
// ============================================================================

const CACHE_TTL_MS = {
  "1day": 6 * 60 * 60 * 1000,
  "4h":   40 * 60 * 1000,
  "1h":   15 * 60 * 1000,
};

// Twelve Data free tier = 8 credits/min. Stay well under: 6/min, then fall back
// to stale cache or a clean RATE_LIMITED response instead of hammering the API.
const MAX_REQUESTS_PER_MINUTE = 6;

// London open ~07:00 GMT, NY afternoon ~17:00 GMT. Gold is a London/NY asset.
const SESSION_START_HOUR = 7;   // GMT
const SESSION_END_HOUR = 17;    // GMT (exclusive)
const WEEKEND_DAYS = new Set([0, 6]); // Sun, Sat

// Confidence floor before the engine will emit a signal.
const MIN_TRADE_CONFIDENCE = 62;

// ---------------------------------------------------------------------------
// Pure math helpers
// ---------------------------------------------------------------------------

function ema(values, period) {
  const k = 2 / (period + 1);
  let result = values[0];
  for (let i = 1; i < values.length; i++) {
    result = values[i] * k + result * (1 - k);
  }
  return result;
}

function emaSeries(values, period) {
  if (!values.length) return [];
  const k = 2 / (period + 1);
  const out = [values[0]];
  for (let i = 1; i < values.length; i++) {
    out.push(values[i] * k + out[i - 1] * (1 - k));
  }
  return out;
}

function trueRange(c) {
  return Math.max(c.high - c.low, Math.abs(c.high - c.close), Math.abs(c.low - c.close));
}

// Wilder-smoothed ATR — adapts to volatility instead of lagging behind it.
function atr(candles, period = 14) {
  if (candles.length < period + 1) return null;
  const trs = [];
  for (let i = 1; i < candles.length; i++) trs.push(trueRange(candles[i]));
  const seed = trs.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let a = seed;
  for (let i = period; i < trs.length; i++) a = (a * (period - 1) + trs[i]) / period;
  return a;
}

// Swing pivot detection — price structure, NOT an indicator. No lag: a pivot
// is only confirmed by the market, never by a smoothed line.
function findSwings(candles, lookback = 3) {
  const highs = [], lows = [];
  for (let i = lookback; i < candles.length - lookback; i++) {
    const c = candles[i];
    let isHigh = true, isLow = true;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (candles[j].high >= c.high) isHigh = false;
      if (candles[j].low <= c.low) isLow = false;
    }
    if (isHigh) highs.push({ index: i, price: c.high, time: c.time });
    if (isLow) lows.push({ index: i, price: c.low, time: c.time });
  }
  return { highs, lows };
}

// Classify the last two pivots into market-structure trend.
function classifyStructure(swings) {
  const hs = swings.highs.slice(-2);
  const ls = swings.lows.slice(-2);
  const hh = hs.length === 2 && hs[1].price > hs[0].price;
  const hl = ls.length === 2 && ls[1].price > ls[0].price;
  const lh = hs.length === 2 && hs[1].price < hs[0].price;
  const ll = ls.length === 2 && ls[1].price < ls[0].price;
  let trend = "neutral";
  if (hh && hl) trend = "up";
  else if (lh && ll) trend = "down";
  return {
    trend,
    higher_highs: hh, higher_lows: hl, lower_highs: lh, lower_lows: ll,
    last_swing_low: ls.at(-1)?.price ?? null,
    last_swing_high: hs.at(-1)?.price ?? null,
  };
}

// Volume profile (POC / VAH / VAL) over a window. Forex volume is tick-derived
// and unreliable, so when raw volume is absent we proxy it with candle range.
function volumeProfile(candles, buckets = 40) {
  if (!candles || candles.length < 20) return null;
  const window = candles.slice(-100);
  const hi = Math.max(...window.map(c => c.high));
  const lo = Math.min(...window.map(c => c.low));
  if (hi === lo) return null;
  const rawVol = window.reduce((s, c) => s + (c.volume || 0), 0);
  const hasRealVol = rawVol > window.length * 2;
  const bucketSize = (hi - lo) / buckets;
  const vap = new Array(buckets).fill(0);
  for (const c of window) {
    const range = c.high - c.low || bucketSize;
    const body = Math.abs(c.close - c.open) || range * 0.3;
    const vol = hasRealVol ? (c.volume > 0 ? c.volume : range) : range * (1 + body / range);
    for (let b = 0; b < buckets; b++) {
      const bLo = lo + b * bucketSize, bHi = bLo + bucketSize;
      const overlap = Math.max(0, Math.min(c.high, bHi) - Math.max(c.low, bLo));
      vap[b] += vol * (overlap / range);
    }
  }
  const pocIdx = vap.indexOf(Math.max(...vap));
  const poc = lo + (pocIdx + 0.5) * bucketSize;
  const total = vap.reduce((a, b) => a + b, 0);
  let loIdx = pocIdx, hiIdx = pocIdx, acc = vap[pocIdx];
  while (acc < total * 0.7) {
    const addLo = loIdx > 0 ? vap[loIdx - 1] : 0;
    const addHi = hiIdx < buckets - 1 ? vap[hiIdx + 1] : 0;
    if (!addLo && !addHi) break;
    if (addHi >= addLo) { hiIdx++; acc += addHi; } else { loIdx--; acc += addLo; }
  }
  return {
    poc, vah: lo + (hiIdx + 1) * bucketSize, val: lo + loIdx * bucketSize,
    range_hi: hi, range_lo: lo, volume_mode: hasRealVol ? "real" : "proxy",
  };
}

// Rolling anchored VWAP (last N bars). Cumulative by definition — zero lag.
function vwap(candles, n = 48) {
  const window = candles.slice(-n);
  let pv = 0, v = 0;
  for (const c of window) {
    const vol = (c.volume || 0) > 0 ? c.volume : (c.high - c.low || 1);
    pv += ((c.high + c.low + c.close) / 3) * vol;
    v += vol;
  }
  return v > 0 ? pv / v : null;
}

// Rejection / trigger candles — pure price action.
function rejectionCandle(candles) {
  const b = candles.at(-1);
  if (!b) return null;
  const body = Math.abs(b.close - b.open);
  const upper = b.high - Math.max(b.open, b.close);
  const lower = Math.min(b.open, b.close) - b.low;
  if (body <= 0) return null;
  if (lower > body * 2 && upper < body) return { kind: "pinbar", side: "bullish" };
  if (upper > body * 2 && lower < body) return { kind: "pinbar", side: "bearish" };
  const a = candles.at(-2);
  if (a) {
    const aBear = a.close < a.open, aBull = a.close > a.open;
    const bBull = b.close > b.open, bBear = b.close < b.open;
    if (aBear && bBull && b.open <= a.close && b.close >= a.open) return { kind: "engulfing", side: "bullish" };
    if (aBull && bBear && b.open >= a.close && b.close <= a.open) return { kind: "engulfing", side: "bearish" };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Session + event guards
// ---------------------------------------------------------------------------

function sessionStatus() {
  const d = new Date();
  const day = d.getUTCDay();
  const hour = d.getUTCHours();
  if (WEEKEND_DAYS.has(day)) return { ok: false, reason: "Weekend (GMT)", hour, day };
  if (hour < SESSION_START_HOUR || hour >= SESSION_END_HOUR) {
    return { ok: false, reason: `Outside London/NY session (${SESSION_START_HOUR}:00-${SESSION_END_HOUR}:00 GMT)`, hour, day };
  }
  return { ok: true, reason: "London/NY session active", hour, day };
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

export class MTFStrategyEngine {
  constructor(deps = {}) {
    this.fetchCandles = deps.fetchCandles;
    this.checkNewsFilter = deps.checkNewsFilter;
    this.calculateLotSize = deps.calculateLotSize;
    this.addTradeMemory = deps.addTradeMemory;
    this._cache = new Map();       // candles with interval-appropriate TTL
    this._requestTimes = [];       // sliding window for the 8/min budget
    this.stats = { api_calls: 0, cache_hits: 0, rate_limited: 0, last_run: null };
  }

  // ---- rate budget ------------------------------------------------------
  _budgetAvailable() {
    const now = Date.now();
    this._requestTimes = this._requestTimes.filter(t => now - t < 60_000);
    return this._requestTimes.length < MAX_REQUESTS_PER_MINUTE;
  }

  _registerRequest() {
    this._requestTimes.push(Date.now());
    this.stats.api_calls++;
  }

  // ---- cached candle fetch ----------------------------------------------
  async _candles(symbol, interval, size) {
    const key = `${symbol}:${interval}:${size}`;
    const hit = this._cache.get(key);
    if (hit && Date.now() - hit.ts < CACHE_TTL_MS[interval]) {
      this.stats.cache_hits++;
      return hit.data;
    }
    if (!this._budgetAvailable()) {
      this.stats.rate_limited++;
      if (hit) return hit.data; // stale-but-better-than-nothing
      return null;
    }
    this._registerRequest();
    const data = await this.fetchCandles(symbol, interval, size);
    if (data && data.length >= 20) {
      this._cache.set(key, { ts: Date.now(), data });
    }
    return data;
  }

  // -----------------------------------------------------------------------
  // FULL multi-timeframe directional report — the "directions" view
  // -----------------------------------------------------------------------
  async analyze(symbol, options = {}) {
    const sym = String(symbol || "").toUpperCase();
    const t0 = Date.now();
    const candles1D = await this._candles(sym, "1day", 240);
    const candles4H = await this._candles(sym, "4h", 220);
    const candles1H = await this._candles(sym, "1h", 170);

    const missing = [];
    if (!candles1D || candles1D.length < 50) missing.push("1D");
    if (!candles4H || candles4H.length < 60) missing.push("4H");
    if (!candles1H || candles1H.length < 60) missing.push("1H");
    if (missing.length) {
      return {
        symbol: sym, decision: "DATA_UNAVAILABLE",
        reason: `Insufficient candle data (missing: ${missing.join(", ")})`,
        strategy: "mtf_v1", timestamp: Date.now(), elapsed_ms: Date.now() - t0,
      };
    }

    const price = candles1H.at(-1).close;
    const dp = sym === "XAUUSD" || sym === "XAGUSD" ? 2 : 5;

    // --- Regime (1D) ---
    const closesD = candles1D.map(c => c.close);
    const ema50 = ema(closesD, 50);
    const ema200 = ema(closesD, 200);
    const slope200 = closesD.length >= 202
      ? ema(closesD.slice(0, -2), 200) - ema(closesD.slice(0, -3), 200)
      : 0;
    let regime = "neutral";
    let regimeScore = 0;
    const priceAboveEma50 = price > ema50;
    if (ema50 > ema200 && slope200 > 0) { regime = "bull"; regimeScore = priceAboveEma50 ? 2 : 1; }
    else if (ema50 < ema200 && slope200 < 0) { regime = "bear"; regimeScore = priceAboveEma50 ? -1 : -2; }
    else if (ema50 > ema200) { regime = "partial_bull"; regimeScore = 1; }
    else if (ema50 < ema200) { regime = "partial_bear"; regimeScore = -1; }

    // --- Structure (4H) ---
    const struct4H = classifyStructure(findSwings(candles4H, 3));

    // --- Entry context (1H) ---
    const profile = volumeProfile(candles1H);
    const vwap1H = vwap(candles1H, 48);
    const atr1H = atr(candles1H, 14);
    const atr4H = atr(candles4H, 14);
    const trigger = rejectionCandle(candles1H);
    const lastRange = candles1H.at(-1).high - candles1H.at(-1).low;

    // Pullback zone: for longs, price within [VAL, POC]; shorts [POC, VAH].
    let zone = null;
    if (profile) {
      if (price >= profile.val && price <= profile.poc) zone = "long_pullback";
      else if (price >= profile.poc && price <= profile.vah) zone = "short_pullback";
      else if (price < profile.val) zone = "below_value";
      else if (price > profile.vah) zone = "above_value";
    }
    const extended = atr1H && profile ? Math.abs(price - profile.poc) > 2.0 * atr1H : false;
    const spiked = atr1H ? lastRange > 4 * atr1H : false;

    // --- Guards ---
    const session = sessionStatus();
    let news = { blocked: false };
    if (this.checkNewsFilter) {
      try { news = await this.checkNewsFilter(sym); } catch { news = { blocked: false }; }
    }

    // --- Direction evaluation (long side shown; short is mirrored) ---
    const regimeOKLong = regimeScore >= 1;
    const regimeOKShort = regimeScore <= -1;
    const structOKLong = struct4H.trend === "up";
    const structOKShort = struct4H.trend === "down";

    const longCandidates = [];
    const shortCandidates = [];

    if (regimeOKLong && structOKLong && zone === "long_pullback" && !extended && !spiked) {
      longCandidates.push({ entry: "long_pullback", trigger });
    }
    if (regimeOKShort && structOKShort && zone === "short_pullback" && !extended && !spiked) {
      shortCandidates.push({ entry: "short_pullback", trigger });
    }

    // If a clean aligned setup exists, evaluate levels + confidence.
    // Evaluate one side: honest stacked score + truthful reasons.
    const regimeLabel = regimeScore === 2 ? "bullish (EMA50>200, slope up)"
      : regimeScore === 1 ? "partial bullish"
      : regimeScore === 0 ? "neutral"
      : regimeScore === -1 ? "partial bearish" : "bearish (EMA50<200, slope down)";
    const structLabel = struct4H.trend === "up" ? "up (HH/HL)"
      : struct4H.trend === "down" ? "down (LH/LL)" : "neutral";
    const evaluate = (side, aligned) => {
      let conf = 50;
      const reasons = [];
      if (side === "long") {
        conf += regimeScore * 6;                                   // +12 full, +6 partial
        conf += structOKLong ? 10 : 0;
        reasons.push(`1D regime ${regimeLabel}`);
        reasons.push(`4H structure ${structLabel}`);
      } else {
        conf += -regimeScore * 6;
        conf += structOKShort ? 10 : 0;
        reasons.push(`1D regime ${regimeLabel}`);
        reasons.push(`4H structure ${structLabel}`);
      }
      if (aligned) {
        conf += 10; reasons.push("Price in pullback zone (POC band)");
      }
      if (trigger) {
        const rightSide = (side === "long" && trigger.side === "bullish") || (side === "short" && trigger.side === "bearish");
        if (rightSide) { conf += 8; reasons.push(`Trigger candle: ${trigger.kind} ${trigger.side}`); }
        else conf -= 4;
      }
      if (profile) {
        const abovePoc = price > profile.poc;
        const biasOk = (side === "long" && !abovePoc) || (side === "short" && abovePoc);
        if (biasOk) { conf += 5; reasons.push("Auction bias aligns (within value area)"); }
      }
      if (session.ok) { conf += 5; reasons.push(session.reason); }
      else reasons.push(session.reason);
      if (!news.blocked) { conf += 5; reasons.push("No high-impact news within window"); }
      if (spiked) conf -= 8;
      if (extended) conf -= 15;
      conf = Math.max(0, Math.min(95, Math.round(conf)));
      return { conf, reasons };
    };

    // Compute levels for each side regardless, so the report always shows them.
    const slLong = struct4H.last_swing_low
      ? struct4H.last_swing_low - (atr4H ? 0.25 * atr4H : 0)
      : null;
    const slShort = struct4H.last_swing_high
      ? struct4H.last_swing_high + (atr4H ? 0.25 * atr4H : 0)
      : null;

    const rr = (sl, tp) => (sl != null && tp != null && price !== sl) ? Math.abs(tp - price) / Math.abs(price - sl) : 0;

    const longLvl = { entry: price, sl: slLong, tp: slLong != null ? price + 2 * (price - slLong) : null, tp2: slLong != null ? price + (price - slLong) : null };
    const shortLvl = { entry: price, sl: slShort, tp: slShort != null ? price - 2 * (slShort - price) : null, tp2: slShort != null ? price - (slShort - price) : null };
    longLvl.rr = rr(longLvl.sl, longLvl.tp);       // full 2R target
    shortLvl.rr = rr(shortLvl.sl, shortLvl.tp);

    // SL sanity: wider than 2.5 ATR(1H) is a no-go (stop too far from trigger).
    const longOk = longLvl.sl != null && atr1H && (price - longLvl.sl) <= 2.5 * atr1H && longLvl.rr >= 1.8;
    const shortOk = shortLvl.sl != null && atr1H && (shortLvl.sl - price) <= 2.5 * atr1H && shortLvl.rr >= 1.8;

    // Pick the aligned side that passed level sanity.
    let signal = "WAIT";
    let side = null, levels = null, conf = 0, reasons = [];
    const longSetup = longCandidates.length > 0 && longOk;
    const shortSetup = shortCandidates.length > 0 && shortOk;
    if (longSetup || shortSetup) {
      if (longSetup && shortSetup) {
        const l = evaluate("long", true), s = evaluate("short", true);
        if (l.conf >= s.conf) { side = "long"; conf = l.conf; reasons = l.reasons; }
        else { side = "short"; conf = s.conf; reasons = s.reasons; }
      } else if (longSetup) { side = "long"; const e = evaluate("long", true); conf = e.conf; reasons = e.reasons; }
      else { side = "short"; const e = evaluate("short", true); conf = e.conf; reasons = e.reasons; }
    }

    if (side) {
      levels = side === "long" ? longLvl : shortLvl;
      if (conf >= MIN_TRADE_CONFIDENCE) signal = side === "long" ? "BUY" : "SELL";
      else signal = "WAIT";
    } else {
      const e = evaluate("long", false);
      const e2 = evaluate("short", false);
      conf = Math.max(e.conf, e2.conf);
      reasons = e.conf >= e2.conf ? e.reasons : e2.reasons;
      if (!longSetup && longOk === false && longCandidates.length) reasons.push("Long candidate filtered (SL too wide or RR < 1.8)");
      if (!shortSetup && shortOk === false && shortCandidates.length) reasons.push("Short candidate filtered (SL too wide or RR < 1.8)");
    }

    const fmt = (x, d = dp) => (x == null ? null : Number(x.toFixed(d)));

    const result = {
      symbol: sym,
      price: fmt(price),
      strategy: "mtf_v1",
      decision: signal,
      direction: signal === "BUY" ? "BULLISH" : signal === "SELL" ? "BEARISH" : "NEUTRAL",
      confidence: conf,
      reasons,
      entry: levels ? fmt(levels.entry) : null,
      sl: levels ? fmt(levels.sl) : null,
      tp: levels ? fmt(levels.tp) : null,
      tp2: levels ? fmt(levels.tp2) : null,
      rr: levels ? Number(levels.rr.toFixed(2)) : null,
      regime: {
        trend: regime, score: regimeScore, ema50: fmt(ema50), ema200: fmt(ema200),
        ema200_slope: Number(slope200.toFixed(5)), price_above_ema50: priceAboveEma50,
      },
      structure: {
        trend: struct4H.trend, higher_highs: struct4H.higher_highs, higher_lows: struct4H.higher_lows,
        lower_highs: struct4H.lower_highs, lower_lows: struct4H.lower_lows,
        last_swing_low: fmt(struct4H.last_swing_low), last_swing_high: fmt(struct4H.last_swing_high),
      },
      entry_ctx: {
        zone, extended, spiked, atr_1h: fmt(atr1H), atr_4h: fmt(atr4H),
        poc: profile ? fmt(profile.poc) : null,
        vah: profile ? fmt(profile.vah) : null,
        val: profile ? fmt(profile.val) : null,
        vwap_1h: vwap1H != null ? fmt(vwap1H) : null,
        volume_mode: profile?.volume_mode ?? "n/a",
        trigger_candle: trigger,
        long_levels: { sl: fmt(longLvl.sl), tp: fmt(longLvl.tp), rr: Number(longLvl.rr.toFixed(2)) },
        short_levels: { sl: fmt(shortLvl.sl), tp: fmt(shortLvl.tp), rr: Number(shortLvl.rr.toFixed(2)) },
      },
      guards: {
        session, news_blocked: news.blocked,
        news_reason: news.reason ?? null,
        news_events: news.events ?? [],
      },
      timestamp: Date.now(),
      elapsed_ms: Date.now() - t0,
    };

    if (news.blocked) {
      result.decision = "NO_TRADE";
      result.reason = `News blackout: ${news.reason}`;
    } else if (!session.ok) {
      result.decision = "WAIT";
      result.reason = session.reason;
    } else if (signal === "BUY" || signal === "SELL") {
      result.reason = "Setup aligned across all timeframes";
    } else {
      result.reason = "No aligned setup right now";
    }

    this.stats.last_run = result;
    return result;
  }

  // -----------------------------------------------------------------------
  // pipeline.run()-compatible decision for /trade — paper or live
  // -----------------------------------------------------------------------
  async run(symbol, options = {}) {
    const sym = String(symbol || "").toUpperCase();
    const t0 = Date.now();
    const analysis = await this.analyze(sym, options);
    if (analysis.decision === "DATA_UNAVAILABLE") return analysis;

    const out = {
      decision: analysis.decision,
      symbol: sym,
      strategy: "mtf_v1",
      confidence: analysis.confidence,
      reason: analysis.reason ?? null,
      reasons: analysis.reasons ?? [],
      entry: analysis.entry,
      sl: analysis.sl,
      tp: analysis.tp,
      tp2: analysis.tp2,
      rr: analysis.rr,
      regime: analysis.regime,
      structure: analysis.structure,
      entry_ctx: analysis.entry_ctx,
      guards: analysis.guards,
      analysis,
      timestamp: Date.now(),
      elapsed_ms: Date.now() - t0,
    };

    if (analysis.decision === "BUY" || analysis.decision === "SELL") {
      out.lot_size = this.calculateLotSize
        ? this.calculateLotSize({ symbol: sym, balance: options.balance || 1000, riskPercent: options.riskPercent || 1, entry: analysis.entry, stopLoss: analysis.sl })
        : 0.01;
      if (this.addTradeMemory) {
        try {
          this.addTradeMemory(sym, {
            direction: analysis.decision,
            pattern: `mtf_v1:${analysis.structure?.trend ?? "?"}:${analysis.entry_ctx?.zone ?? "?"}`,
            outcome: "pending",
            note: `MTF v1 conf=${analysis.confidence}% rr=${analysis.rr} trigger=${analysis.entry_ctx?.trigger_candle?.kind ?? "none"}`,
          });
        } catch { /* journal best-effort */ }
      }
    }
    return out;
  }

  cacheStatus() {
    return {
      strategy: "mtf_v1",
      api_calls: this.stats.api_calls,
      cache_hits: this.stats.cache_hits,
      rate_limited_calls: this.stats.rate_limited,
      cache_entries: this._cache.size,
      rate_budget: `${this._requestTimes.length}/${MAX_REQUESTS_PER_MINUTE} in last 60s`,
      last_run: this.stats.last_run ? { symbol: this.stats.last_run.symbol, decision: this.stats.last_run.decision, confidence: this.stats.last_run.confidence } : null,
    };
  }
}
