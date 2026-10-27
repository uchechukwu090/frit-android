// ============================================================================
// FRIT PREDICTIVE VOLUME ENGINE — "is activity about to expand with the trend?"
// ----------------------------------------------------------------------------
// The existing volume matrix is a FILTER (where/which/when). This is the
// FORECASTER the trending bots consult once a signal is already live: it reads
// the 5-minute bars (primary) and the 1-minute bars (pace inside the current
// 5m window) and scores the odds that activity EXPANDS over the next 1-3 five-
// minute bars in the direction of the live trend.
//
// It is deliberately a transparent, additive scoring model, not a black box:
//   +  relative volume of the last closed bars vs a robust (median) baseline
//   +  slope of log-volume over the last 6 bars (volume is BUILDING)
//   +  1m pace: last 3 one-minute bars vs their own baseline, and accelerating
//   +  compression -> release (range coiled, then volume turns up)
//   +  alignment: price displacement and close-location flow agree with trend
//   +  breakout of the prior 20-bar extreme on >= 1.3x volume
//   -  opposing flow is penalised; a 5x+ spike or a rejected/overextended bar is
//      EXHAUSTION and BLOCKS a continuation trigger instead of feeding it.
//
// HONEST DATA LIMITS (labelled in every result, never faked):
//   - Spot FX and gold have NO open interest and NO block/bulk-deal prints;
//     those exist for exchange-traded futures/stocks only. "Volume" on FX/XAU
//     is the broker's TICK volume (number of price updates). It is a good
//     activity proxy and what MT5 itself shows, but it is not traded size.
//   - When a feed reports no volume at all, the engine falls back to bar RANGE
//     (true range) as the activity proxy, caps its score and says so.
//   - The score is a ranking, not a calibrated probability. backtestForecast()
//     measures the real hit-rate and lift on whatever history you feed it.
// ============================================================================

const FIVE_MIN = 5 * 60_000;
const MIN_5M_BARS = 60;

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);
function median(a) {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function toMs(t) {
  if (typeof t === "number") return t < 1e12 ? t * 1000 : t;
  const p = Date.parse(t);
  return Number.isFinite(p) ? p : NaN;
}

/** Clean, ascending-by-time copy with a numeric `t` (ms). */
export function prepCandles(candles) {
  if (!Array.isArray(candles)) return [];
  return candles
    .map((c) => ({
      t: toMs(c.time ?? c.t ?? c.datetime),
      open: Number(c.open), high: Number(c.high), low: Number(c.low), close: Number(c.close),
      volume: Number(c.volume ?? c.tickVolume ?? 0) || 0,
    }))
    .filter((c) => Number.isFinite(c.t) && c.close > 0 && c.high >= c.low)
    .sort((a, b) => a.t - b.t);
}

function trueRange(c, prevClose) {
  return Math.max(c.high - c.low, Math.abs(c.high - prevClose), Math.abs(c.low - prevClose));
}

export function atr(bars, n = 14) {
  if (bars.length < 2) return 0;
  const trs = [];
  for (let i = 1; i < bars.length; i++) trs.push(trueRange(bars[i], bars[i - 1].close));
  return mean(trs.slice(-n));
}

function ema(values, n) {
  if (!values.length) return null;
  const k = 2 / (n + 1);
  let e = values[0];
  for (let i = 1; i < values.length; i++) e = values[i] * k + e * (1 - k);
  return e;
}

/**
 * Activity series: real/tick volume when the feed has it (>= 80% of bars),
 * otherwise true range as an explicitly-labelled proxy.
 */
export function activitySeries(bars) {
  const withVol = bars.filter((b) => b.volume > 0).length;
  const coverage = bars.length ? withVol / bars.length : 0;
  if (coverage >= 0.8) return { values: bars.map((b) => b.volume), source: "volume", coverage };
  return {
    values: bars.map((b, i) => trueRange(b, i ? bars[i - 1].close : b.open)),
    source: "range_proxy",
    coverage,
  };
}

function olsSlope(ys) {
  const n = ys.length;
  if (n < 3) return 0;
  const xm = (n - 1) / 2;
  const ym = mean(ys);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (i - xm) * (ys[i] - ym); den += (i - xm) ** 2; }
  return den ? num / den : 0;
}

/** Trend from the bars themselves: EMA20 vs EMA50, ignoring a flat tangle. */
export function trendOf(bars) {
  if (bars.length < 55) return null;
  const closes = bars.map((b) => b.close);
  const e20 = ema(closes.slice(-120), 20);
  const e50 = ema(closes.slice(-200), 50);
  const a = atr(bars, 14);
  if (!(a > 0) || Math.abs(e20 - e50) < 0.15 * a) return null;
  return e20 > e50 ? "BUY" : "SELL";
}

function decimalsFor(price) {
  return price >= 1000 ? 2 : price >= 100 ? 3 : price >= 10 ? 4 : 5;
}

/** Fresh SL/TP for a continuation entry at `price`, sized off 5m ATR. */
export function freshLevels(price, direction, atr5, { rr = 1.5, slAtr = 1.5 } = {}) {
  const sign = direction === "SELL" ? -1 : 1;
  const d = decimalsFor(price);
  const risk = slAtr * atr5;
  return {
    entry: Number(price.toFixed(d)),
    sl: Number((price - sign * risk).toFixed(d)),
    tp: Number((price + sign * risk * rr).toFixed(d)),
    rr,
    slAtr,
  };
}

const unavailable = (why, extra = {}) => ({
  available: false, expansion: false, score: 0, reasons: [why], ...extra,
});

/**
 * @param c5     5-minute candles (primary timeframe of the forecast)
 * @param c1     1-minute candles (pace inside the current 5m window); optional
 * @param trend  "BUY" | "SELL" — the live signal's direction (derived if null)
 * @param now    clock override for tests/backtests
 */
export function forecastVolume({ c5, c1 = [], trend = null, now = Date.now(), rr = 1.5, opts = {} } = {}) {
  const o = {
    baselineBars: 48, slopeBars: 6, expandScore: 60, exhaustRvol: 5, slAtr: 1.5,
    proxyPenalty: 5, ...opts,
  };
  const all = prepCandles(c5);
  if (all.length < MIN_5M_BARS) return unavailable(`need ${MIN_5M_BARS}+ five-minute candles (have ${all.length})`);

  const last = all[all.length - 1];
  const forming = now - last.t >= 0 && now - last.t < FIVE_MIN;
  const bars = forming ? all.slice(0, -1) : all;
  if (bars.length < MIN_5M_BARS - 1) return unavailable("not enough closed 5m candles");

  const dir = trend === "BUY" || trend === "SELL" ? trend : trendOf(bars);
  if (!dir) {
    return unavailable("no clear trend to forecast volume against", { available: true, direction: null });
  }
  const sign = dir === "BUY" ? 1 : -1;

  const act = activitySeries(bars);
  const A = act.values;
  const baseWindow = A.slice(-(o.baselineBars + 3), -3);
  const baseline = median(baseWindow);
  if (!(baseline > 0) || baseWindow.length < 30) return unavailable("no usable activity baseline");

  const cur = bars[bars.length - 1];
  const atrNow = atr(bars, 14);
  const atrLong = atr(bars, 48);
  if (!(atrNow > 0)) return unavailable("zero volatility");

  const rvol5 = A[A.length - 1] / baseline;
  const rvol2 = mean(A.slice(-2)) / baseline;
  const logs = A.slice(-o.slopeBars).map((v) => Math.log(Math.max(v, baseline * 0.05)));
  const slope = olsSlope(logs);

  const shortAtr = atr(bars.slice(-8), 5);
  const compression = atrLong > 0 && shortAtr / atrLong < 0.85;

  const dispAligned = ((cur.close - bars[bars.length - 4].close) / atrNow) * sign;
  let cvdNum = 0, cvdDen = 0;
  for (let i = bars.length - 6; i < bars.length; i++) {
    const b = bars[i];
    const range = b.high - b.low;
    const loc = range > 0 ? (b.close - b.open) / range : 0;
    cvdNum += loc * A[i];
    cvdDen += A[i];
  }
  const cvdAligned = (cvdDen > 0 ? cvdNum / cvdDen : 0) * sign;

  const prior = bars.slice(-21, -1);
  const breakout = sign > 0
    ? cur.close > Math.max(...prior.map((b) => b.high))
    : cur.close < Math.min(...prior.map((b) => b.low));

  // 1m pace
  let rvol1 = null, accel = false, paceQuality = "none";
  const m1all = prepCandles(c1);
  if (m1all.length >= 25) {
    const lastM = m1all[m1all.length - 1];
    const formingM = now - lastM.t >= 0 && now - lastM.t < 60_000;
    const m1 = formingM ? m1all.slice(0, -1) : m1all;
    const a1 = activitySeries(m1).values;
    const base1 = median(a1.slice(-33, -3));
    if (base1 > 0 && a1.length >= 25) {
      rvol1 = mean(a1.slice(-3)) / base1;
      accel = mean(a1.slice(-3)) > 1.15 * mean(a1.slice(-6, -3));
      paceQuality = activitySeries(m1).source;
    }
  }

  // ---- exhaustion guards (a spike is not a runway) ----
  const ema20 = ema(bars.map((b) => b.close).slice(-80), 20);
  const extended = Math.abs(cur.close - ema20) / atrNow > 3.5;
  const range = cur.high - cur.low;
  const closeLoc = range > 0 ? ((cur.close - cur.low) / range) : 0.5; // 0 = at low, 1 = at high
  const rejected = rvol5 >= 3 && (sign > 0 ? closeLoc < 0.3 : closeLoc > 0.7);
  const spike = rvol5 >= o.exhaustRvol || (rvol1 != null && rvol1 >= o.exhaustRvol * 1.2);
  const exhaustion = spike || rejected || extended;

  // ---- score ----
  const sR = clamp((Math.min(rvol2, o.exhaustRvol) - 1) / 1.5, 0, 1) * 20;
  const sSlope = clamp(slope / 0.10, 0, 1) * 20;
  const sPace = rvol1 != null ? clamp((rvol1 - 1) / 1.5, 0, 1) * 15 + (accel ? 5 : 0) : null;
  const sCoil = compression && slope > 0.02 ? 10 : 0;
  const sAlign = (dispAligned > 0.3 ? 10 : dispAligned < -0.3 ? -10 : 0)
    + (cvdAligned > 0.1 ? 10 : cvdAligned < -0.1 ? -10 : 0);
  const sBreak = breakout && rvol2 >= 1.3 ? 10 : 0;

  let raw = sR + sSlope + (sPace ?? 0) + sCoil + sAlign + sBreak;
  if (sPace == null) raw = (raw * 100) / 80; // no 1m leg: rescale the 80 points that exist
  let score = Math.round(clamp(raw, 0, 100));
  const proxy = act.source !== "volume";
  if (proxy) score = Math.min(score, 80);
  const needScore = o.expandScore + (proxy ? o.proxyPenalty : 0);

  const aligned = sAlign >= 10;
  const expansion = score >= needScore && aligned && !exhaustion;

  const reasons = [];
  reasons.push(`${proxy ? "range-proxy" : "tick-volume"} 5m rvol ${rvol2.toFixed(2)}x, slope ${(slope * 100).toFixed(1)}%/bar`);
  if (rvol1 != null) reasons.push(`1m pace ${rvol1.toFixed(2)}x${accel ? " and accelerating" : ""}`);
  if (compression) reasons.push("range was compressed before the volume turn");
  if (breakout) reasons.push(`broke the prior 20-bar ${sign > 0 ? "high" : "low"}${rvol2 >= 1.3 ? " on strong volume" : " (volume weak)"}`);
  reasons.push(aligned ? "price/flow agree with the trend" : "price/flow do not clearly confirm the trend");
  if (sAlign < 0) reasons.push("flow is AGAINST the trend — expansion would be a reversal risk, not continuation");
  if (spike) reasons.push(`volume spike >= ${o.exhaustRvol}x — exhaustion risk, no continuation trigger`);
  if (rejected) reasons.push("high-volume bar rejected at its extreme — exhaustion risk");
  if (extended) reasons.push("price stretched > 3.5 ATR from its EMA20 — too late to chase");

  const price = (m1all.length ? m1all[m1all.length - 1].close : last.close);
  return {
    available: true,
    expansion,
    direction: dir,
    score,
    needScore,
    exhaustion,
    source: act.source,
    coverage: Number(act.coverage.toFixed(2)),
    features: {
      rvol5: Number(rvol5.toFixed(2)), rvol2: Number(rvol2.toFixed(2)), slope: Number(slope.toFixed(4)),
      rvol1: rvol1 == null ? null : Number(rvol1.toFixed(2)), accel, paceQuality,
      compression, dispAligned: Number(dispAligned.toFixed(2)), cvdAligned: Number(cvdAligned.toFixed(2)),
      breakout, extended, rejected,
    },
    price,
    atr5: atrNow,
    levels: freshLevels(price, dir, atrNow, { rr, slAtr: o.slAtr }),
    reasons,
    dataNote: proxy
      ? "feed has no volume: bar range used as the activity proxy (score capped)"
      : "broker TICK volume (price-update count), not traded size; spot FX/gold has no open interest or block prints",
  };
}

/**
 * Honest evaluation: on history, how often did a trigger precede a REAL
 * expansion (mean activity of the next `horizon` bars >= expandRatio x its
 * baseline) compared with how often expansion happens anyway? lift > 1 means
 * the trigger adds information; lift ~ 1 means it does not.
 */
export function backtestForecast(c5, { horizon = 3, expandRatio = 1.4, start = 70, opts = {} } = {}) {
  const bars = prepCandles(c5);
  if (bars.length < start + horizon + 5) return { ok: false, why: `need ${start + horizon + 5}+ candles` };
  const act = activitySeries(bars).values;
  let samples = 0, baseHits = 0, triggers = 0, hits = 0, followThrough = 0;
  for (let i = start; i <= bars.length - horizon - 1; i++) {
    const slice = bars.slice(0, i + 1);
    const trend = trendOf(slice);
    if (!trend) continue;
    const baseline = median(act.slice(Math.max(0, i - 50), i - 2));
    if (!(baseline > 0)) continue;
    const future = mean(act.slice(i + 1, i + 1 + horizon)) / baseline;
    const expanded = future >= expandRatio;
    samples++;
    if (expanded) baseHits++;
    const f = forecastVolume({ c5: slice, trend, now: slice[i].t + FIVE_MIN, opts });
    if (f.available && f.expansion) {
      triggers++;
      if (expanded) hits++;
      const sgn = trend === "BUY" ? 1 : -1;
      const a = atr(slice, 14);
      if (a > 0 && ((bars[i + horizon].close - bars[i].close) / a) * sgn >= 0.5) followThrough++;
    }
  }
  const baseRate = samples ? baseHits / samples : 0;
  const hitRate = triggers ? hits / triggers : 0;
  return {
    ok: true, samples, triggers,
    baseRate: Number(baseRate.toFixed(3)),
    hitRate: Number(hitRate.toFixed(3)),
    lift: baseRate > 0 && triggers ? Number((hitRate / baseRate).toFixed(2)) : null,
    priceFollowThroughRate: triggers ? Number((followThrough / triggers).toFixed(3)) : null,
    note: triggers < 30 ? "fewer than 30 triggers — treat as anecdotal, not evidence" : "ok",
  };
}
