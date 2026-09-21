// ============================================================================
// FRIT MA-RSI COMBO ENGINE — EMA(9/21) crossover + RSI(14) band, 4H-confirmed
// ----------------------------------------------------------------------------
// Drop-in replacement for MTFStrategyEngine (strategy/engine.js). Same
// constructor deps and same run()/analyze()/cacheStatus() contract, so
// scheduler.js and the /trade "enhanced" pipeline in index.js work with only
// an import swap — no other wiring changes needed.
//
// Primary TF : 1H  (kept — matches your existing data pipeline + scheduler
//              cadence; the source doc's M15/M30 for crypto/FX was never
//              validated against your setup, so not worth chasing)
// Confirm TF : 4H  (preserves the doc's ~1:4 primary:confirmation ratio)
//
// NOTE ON THE SOURCE DOC: it defines the 4H trend filter as "slow EMA(21)
// above fast EMA(9) => uptrend" — the reverse of the usual convention
// (normally fast > slow = uptrend, since the fast EMA reacts quicker to
// rising price). Implemented LITERALLY as the doc states — see
// CONFIRM_TREND_INVERTED below. Flip it to false if this was a doc typo and
// not deliberate; you should paper-trade both readings before trusting either.
// ============================================================================

const CONFIRM_TREND_INVERTED = true;

const CACHE_TTL_MS = { "4h": 40 * 60 * 1000, "1h": 15 * 60 * 1000 };
const MAX_REQUESTS_PER_MINUTE = 6;
const MIN_TRADE_CONFIDENCE = 60;

const FAST_PERIOD = 9;
const SLOW_PERIOD = 21;
const RSI_PERIOD = 14;
const ATR_PERIOD = 14;
const VOL_SMA_PERIOD = 20;
const ATR_SL_MULT = 1.5;
const MAX_SL_ATR_MULT = 2.5; // sanity cap, same philosophy as mtf_v1
const MIN_RR = 1.8;

const SESSION_START_HOUR = 7;  // GMT, London/NY window — skipped for crypto
const SESSION_END_HOUR = 17;
const WEEKEND_DAYS = new Set([0, 6]);
const CRYPTO_SET = new Set(["BTCUSD", "ETHUSD", "BTCUSDT", "ETHUSDT"]); // extend as needed

// ---------------------------------------------------------------------------
// Pure math helpers
// ---------------------------------------------------------------------------
function emaSeries(values, period) {
  if (!values.length) return [];
  const k = 2 / (period + 1);
  const out = [values[0]];
  for (let i = 1; i < values.length; i++) out.push(values[i] * k + out[i - 1] * (1 - k));
  return out;
}

// Wilder RSI — standard smoothing, matches ta-lib/pandas-ta default.
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

// Last confirmed swing low/high, used only as an SL sanity anchor.
function lastSwing(candles, lookback = 3) {
  let low = null, high = null;
  for (let i = candles.length - lookback - 1; i >= lookback; i--) {
    const c = candles[i];
    let isLow = true, isHigh = true;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (candles[j].low <= c.low) isLow = false;
      if (candles[j].high >= c.high) isHigh = false;
    }
    if (isLow && low == null) low = c.low;
    if (isHigh && high == null) high = c.high;
    if (low != null && high != null) break;
  }
  return { low, high };
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
// Engine
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
    if (hit && Date.now() - hit.ts < CACHE_TTL_MS[interval]) {
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
    const [candles1H, candles4H] = await Promise.all([
      this._candles(sym, "1h", 150),
      this._candles(sym, "4h", 80),
    ]);

    const need1H = SLOW_PERIOD + RSI_PERIOD + VOL_SMA_PERIOD + 10;
    if (!candles1H || candles1H.length < need1H || !candles4H || candles4H.length < SLOW_PERIOD + 5) {
      return {
        symbol: sym, decision: "DATA_UNAVAILABLE",
        reason: "Insufficient candle data (need 1H + 4H history)",
        strategy: "ma_rsi_v1", timestamp: Date.now(), elapsed_ms: Date.now() - t0,
      };
    }

    const price = candles1H.at(-1).close;
    const dp = sym === "XAUUSD" || sym === "XAGUSD" ? 2 : CRYPTO_SET.has(sym) ? 2 : 5;
    const fmt = (x, d = dp) => (x == null ? null : Number(x.toFixed(d)));

    // --- 4H trend/confirmation filter ---
    const closes4H = candles4H.map(c => c.close);
    const ema4hFast = emaSeries(closes4H, FAST_PERIOD).at(-1);
    const ema4hSlow = emaSeries(closes4H, SLOW_PERIOD).at(-1);
    const trendUp = CONFIRM_TREND_INVERTED ? ema4hSlow > ema4hFast : ema4hFast > ema4hSlow;
    const trendDown = CONFIRM_TREND_INVERTED ? ema4hSlow < ema4hFast : ema4hFast < ema4hSlow;
    // Doc's "wait for next candle to confirm" adapted to a stateless poll:
    // require the most recently CLOSED 4H candle already closed on the
    // correct side of the 4H slow EMA, rather than waiting on an unformed one.
    const confirmLong = candles4H.at(-1).close > ema4hSlow;
    const confirmShort = candles4H.at(-1).close < ema4hSlow;

    // --- 1H EMA crossover ---
    const closes1H = candles1H.map(c => c.close);
    const emaFast1H = emaSeries(closes1H, FAST_PERIOD);
    const emaSlow1H = emaSeries(closes1H, SLOW_PERIOD);
    const n = emaFast1H.length;
    const crossUp = emaFast1H[n - 2] <= emaSlow1H[n - 2] && emaFast1H[n - 1] > emaSlow1H[n - 1];
    const crossDown = emaFast1H[n - 2] >= emaSlow1H[n - 2] && emaFast1H[n - 1] < emaSlow1H[n - 1];

    // --- RSI(14) band ---
    const rsiNow = rsiSeries(closes1H, RSI_PERIOD).at(-1);
    const rsiInBand = rsiNow > 30 && rsiNow < 70;
    const rsiLongSweet = rsiNow > 30 && rsiNow < 55;
    const rsiShortSweet = rsiNow > 45 && rsiNow < 70;

    // --- Volume filter (best-effort; forex volume is often tick-derived/unreliable) ---
    const volumes1H = candles1H.map(c => c.volume || 0);
    const rawVol = volumes1H.slice(-VOL_SMA_PERIOD).reduce((a, b) => a + b, 0);
    const hasRealVol = rawVol > VOL_SMA_PERIOD * 2;
    const volSma = smaLast(volumes1H, VOL_SMA_PERIOD);
    const volOk = !hasRealVol ? true : (volumes1H.at(-1) > (volSma ?? 0)); // don't gate on an unreliable proxy

    const atr1H = atr(candles1H, ATR_PERIOD);
    const swing = lastSwing(candles1H, 3);
    const session = sessionStatus(sym);
    let news = { blocked: false };
    if (this.checkNewsFilter) {
      try { news = await this.checkNewsFilter(sym); } catch { news = { blocked: false }; }
    }

    const longSetup = crossUp && trendUp && confirmLong && rsiInBand && volOk;
    const shortSetup = crossDown && trendDown && confirmShort && rsiInBand && volOk;

    // --- Levels: fixed ATR multiple, sanity-widened to clear the last swing,
    //     capped so a stop never drifts past MAX_SL_ATR_MULT * ATR ---
    const buildLevels = (side) => {
      const atrDist = ATR_SL_MULT * atr1H;
      let slDist = atrDist;
      if (side === "long" && swing.low != null && price - swing.low > atrDist) {
        slDist = Math.min(price - swing.low + 0.1 * atr1H, MAX_SL_ATR_MULT * atr1H);
      } else if (side === "short" && swing.high != null && swing.high - price > atrDist) {
        slDist = Math.min(swing.high - price + 0.1 * atr1H, MAX_SL_ATR_MULT * atr1H);
      }
      const sl = side === "long" ? price - slDist : price + slDist;
      const tp = side === "long" ? price + 2 * slDist : price - 2 * slDist;   // 1:2 RR
      const tp2 = side === "long" ? price + 1 * slDist : price - 1 * slDist;  // 1:1 partial
      const rr = slDist > 0 ? Math.abs(tp - price) / slDist : 0;
      return { entry: price, sl, tp, tp2, rr };
    };

    const scoreSide = (side) => {
      let conf = 50;
      const reasons = [`4H trend ${trendUp && side === "long" ? "up" : trendDown && side === "short" ? "down" : "misaligned"}`];
      conf += 15; // crossover fired
      conf += 10; // trend aligned (only reached if setup true)
      conf += 8;  // confirmation TF aligned
      reasons.push(`EMA(${FAST_PERIOD}/${SLOW_PERIOD}) ${side === "long" ? "bullish" : "bearish"} crossover on 1H`);
      reasons.push(`4H close confirms (vs slow EMA)`);
      if (side === "long" ? rsiLongSweet : rsiShortSweet) { conf += 7; reasons.push(`RSI ${rsiNow.toFixed(1)} in sweet spot`); }
      else { conf += 3; reasons.push(`RSI ${rsiNow.toFixed(1)} in band but outside sweet spot`); }
      if (hasRealVol && volOk) { conf += 5; reasons.push("Volume above 20-SMA"); }
      if (session.ok) { conf += 5; reasons.push(session.reason); } else reasons.push(session.reason);
      if (!news.blocked) { conf += 5; reasons.push("No high-impact news within window"); }
      return { conf: Math.max(0, Math.min(95, Math.round(conf))), reasons };
    };

    let signal = "WAIT", side = null, levels = null, conf = 0, reasons = [];
    if (longSetup || shortSetup) {
      side = longSetup ? "long" : "short";
      levels = buildLevels(side);
      const scored = scoreSide(side);
      conf = scored.conf; reasons = scored.reasons;
      if (levels.rr < MIN_RR) reasons.push(`Filtered: RR ${levels.rr.toFixed(2)} below minimum ${MIN_RR}`);
      else if (news.blocked) { signal = "NO_TRADE"; }
      else if (!session.ok) { signal = "WAIT"; }
      else if (conf >= MIN_TRADE_CONFIDENCE) { signal = side === "long" ? "BUY" : "SELL"; }
    } else {
      reasons = [
        crossUp || crossDown ? "Crossover fired but trend/confirmation/RSI/volume filter blocked it" : "No EMA crossover on 1H",
      ];
    }

    const result = {
      symbol: sym,
      price: fmt(price),
      strategy: "ma_rsi_v1",
      decision: signal,
      direction: signal === "BUY" ? "BULLISH" : signal === "SELL" ? "BEARISH" : "NEUTRAL",
      confidence: conf,
      reasons,
      entry: levels ? fmt(levels.entry) : null,
      sl: levels ? fmt(levels.sl) : null,
      tp: levels ? fmt(levels.tp) : null,
      tp2: levels ? fmt(levels.tp2) : null,
      rr: levels ? Number(levels.rr.toFixed(2)) : null,
      regime: { trend: trendUp ? "bull" : trendDown ? "bear" : "neutral", ema4h_fast: fmt(ema4hFast), ema4h_slow: fmt(ema4hSlow) },
      structure: { trend: trendUp ? "up" : trendDown ? "down" : "neutral", last_swing_low: fmt(swing.low), last_swing_high: fmt(swing.high) },
      entry_ctx: {
        rsi: Number(rsiNow.toFixed(1)), rsi_band_ok: rsiInBand, atr_1h: fmt(atr1H),
        ema1h_fast: fmt(emaFast1H.at(-1)), ema1h_slow: fmt(emaSlow1H.at(-1)),
        cross_up: crossUp, cross_down: crossDown, confirm_long: confirmLong, confirm_short: confirmShort,
        volume_mode: hasRealVol ? "real" : "unreliable/skipped",
      },
      guards: { session, news_blocked: news.blocked, news_reason: news.reason ?? null },
      timestamp: Date.now(),
      elapsed_ms: Date.now() - t0,
    };

    if (news.blocked) result.reason = `News blackout: ${news.reason}`;
    else if (!session.ok) result.reason = session.reason;
    else if (signal === "BUY" || signal === "SELL") result.reason = "MA-RSI setup aligned across 1H/4H";
    else result.reason = reasons[reasons.length - 1] || "No aligned setup right now";

    this.stats.last_run = result;
    return result;
  }

  // pipeline.run()-compatible decision for /trade — same shape as mtf_v1
  async run(symbol, options = {}) {
    const sym = String(symbol || "").toUpperCase();
    const t0 = Date.now();
    const analysis = await this.analyze(sym, options);
    if (analysis.decision === "DATA_UNAVAILABLE") return analysis;

    const out = {
      decision: analysis.decision, symbol: sym, strategy: "ma_rsi_v1",
      confidence: analysis.confidence, reason: analysis.reason ?? null, reasons: analysis.reasons ?? [],
      entry: analysis.entry, sl: analysis.sl, tp: analysis.tp, tp2: analysis.tp2, rr: analysis.rr,
      regime: analysis.regime, structure: analysis.structure, entry_ctx: analysis.entry_ctx, guards: analysis.guards,
      analysis, timestamp: Date.now(), elapsed_ms: Date.now() - t0,
    };

    if (analysis.decision === "BUY" || analysis.decision === "SELL") {
      out.lot_size = this.calculateLotSize
        ? this.calculateLotSize({ symbol: sym, balance: options.balance || 1000, riskPercent: options.riskPercent || 1, entry: analysis.entry, stopLoss: analysis.sl })
        : 0.01;
      if (this.addTradeMemory) {
        try {
          this.addTradeMemory(sym, {
            direction: analysis.decision,
            pattern: `ma_rsi_v1:${analysis.structure?.trend ?? "?"}`,
            outcome: "pending",
            note: `MA-RSI conf=${analysis.confidence}% rr=${analysis.rr} rsi=${analysis.entry_ctx?.rsi}`,
          });
        } catch { /* journal best-effort */ }
      }
    }
    return out;
  }

  cacheStatus() {
    return {
      strategy: "ma_rsi_v1",
      api_calls: this.stats.api_calls,
      cache_hits: this.stats.cache_hits,
      rate_limited_calls: this.stats.rate_limited,
      cache_entries: this._cache.size,
      rate_budget: `${this._requestTimes.length}/${MAX_REQUESTS_PER_MINUTE} in last 60s`,
      last_run: this.stats.last_run ? { symbol: this.stats.last_run.symbol, decision: this.stats.last_run.decision, confidence: this.stats.last_run.confidence } : null,
    };
  }
}