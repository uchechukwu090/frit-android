// ============================================================================
// FRIT VOLUME MATRIX — 5 timeframes, one bias read
// ----------------------------------------------------------------------------
// The matrix is a FILTER and BIAS layer, not a predictor. It answers three
// questions with data we already fetch per candle (no tick feed required):
//   WHERE   — anchor value areas (4H VP) and session balance (30M VP/VWAP)
//   WHICH   — directional bias from price-vs-VWAP + cumulative volume delta
//   WHEN    — order-flow spikes (RVOL) that make an entry safer or dangerous
//
// HONEST DATA LIMITS (labeled, never faked):
//   - Volume Profile is built from CANDLES, not tick prints: each candle's
//     volume is distributed evenly across [low, high]. This is the standard
//     candle-based approximation. It is close to true VP but can blur zones on
//     wide-range bars — so zones are reported as bands, not precise levels.
//   - CVD here is a CLOSE-LOCATION PROXY, not true order-flow delta (true
//     delta needs a tick feed we do not have). Per candle: sign(close-open) /
//     (high-low) x volume, accumulated. Directionally useful; do NOT treat
//     its magnitude as institutional size.
//   - 1M RVOL is real where the feed has 1m bars (Binance, Dukascopy carve,
//     Alpaca crypto). For Finnhub-sourced FX the 1m leg falls back to the
//     freshest available bar and says so.
// Net output: bias BULL/BEAR/NEUTRAL + confidence + per-leg reasons. Strong
// disagreement with a trend entry is meant to SUPPRESS it, not to replace it.
// ============================================================================

const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);

/** Value-area volume profile from candles (candle-based approximation). */
export function volumeProfile(candles, { levels = 40, bucketSize = null } = {}) {
  if (!Array.isArray(candles) || !candles.length) return null;
  const lo = Math.min(...candles.map((c) => c.low));
  const hi = Math.max(...candles.map((c) => c.high));
  if (!(hi > lo)) return null;
  const step = bucketSize || (hi - lo) / levels;
  const buckets = new Map();
  let totalVol = 0;
  for (const c of candles) {
    if (!(c.volume > 0)) continue;
    const n = Math.max(1, Math.ceil((c.high - c.low) / step));
    const per = c.volume / n;
    for (let p = c.low; p < c.high; p += step) {
      const key = Math.floor(p / step) * step;
      const prev = buckets.get(key) || 0;
      buckets.set(key, +(prev + per).toFixed(4));
      totalVol += per;
    }
  }
  if (!buckets.size || totalVol <= 0) return null;
  // POC = highest-volume bucket; value area = 70% of volume around POC.
  let poc = null, pocVol = -1;
  for (const [k, v] of buckets) if (v > pocVol) { pocVol = v; poc = k; }
  const sorted = [...buckets.entries()].sort((a, b) => a[0] - b[0]);
  const pocIdx = sorted.findIndex(([k]) => k === poc);
  let accum = pocVol; // the POC bucket itself is always inside the value area
  let vaLo = poc, vaHi = poc;
  // Walk outward from the POC, always taking the HEAVIER neighbour, until 70%
  // of the volume is included. F-44 (Week 5): the old walk only added a
  // bucket when BOTH sides existed, so a POC sitting on the lowest (or
  // highest) price bucket never accumulated any volume and the value area
  // collapsed to a single level — the advertised "70% value area" was
  // silently empty for exactly those profiles (heavy volume at the lows).
  let d = 0, u = 0;
  while (accum < totalVol * 0.7) {
    const down = sorted[pocIdx - d - 1];
    const up = sorted[pocIdx + u + 1];
    if (!down && !up) break;
    if (down && (!up || down[1] >= up[1])) {
      accum += down[1]; vaLo = down[0]; d++;
    } else {
      accum += up[1]; vaHi = up[0]; u++;
    }
    if (d + u > sorted.length) break;
  }
  return {
    poc: +poc.toFixed(5),
    valueAreaLow: +vaLo.toFixed(5),
    valueAreaHigh: +vaHi.toFixed(5),
    totalVolume: +totalVol.toFixed(1),
    approximation: "candle-distributed (not tick VP)",
  };
}

/** Cumulative typical-price VWAP over the given (already session-scoped) bars. */
export function vwap(candles) {
  if (!Array.isArray(candles) || !candles.length) return null;
  let pv = 0, v = 0;
  for (const c of candles) {
    const tp = (c.high + c.low + c.close) / 3;
    pv += tp * (c.volume || 0);
    v += c.volume || 0;
  }
  return v > 0 ? pv / v : null;
}

/** Close-location volume-delta PROXY, accumulated over the last n bars. */
export function cvdProxy(candles, n = 20) {
  if (!Array.isArray(candles) || candles.length < 2) return null;
  let out = 0;
  for (const c of candles.slice(-n)) {
    const range = c.high - c.low;
    const loc = range > 0 ? (c.close - c.open) / range : 0; // -1..1
    out += loc * (c.volume || 0);
  }
  return out; // sign = net aggression; magnitude = proxy, not size
}

/** Relative volume: last bar vs mean of prior n. */
export function rvol(candles, n = 20) {
  if (!Array.isArray(candles) || candles.length < n + 2) return null;
  const avg = mean(candles.slice(-n - 1, -1).map((c) => c.volume));
  const cur = candles[candles.length - 1].volume;
  return avg > 0 ? cur / avg : null;
}

/**
 * Full matrix across the five timeframes.
 * candlesN: { "4h": [], "30m": [], "15m": [], "5m": [], "1m": [] }
 * @returns { bias, confidence, legs, zones, why[] }
 */
export function analyzeVolumeMatrix(symbol, candlesN = {}) {
  const sym = String(symbol || "").toUpperCase();
  const base = {
    symbol: sym, strategy: "volume_matrix", family: "VOLUME",
    bias: "NEUTRAL", confidence: 0, legs: {}, zones: {}, why: [],
  };
  const need = { "4h": 40, "30m": 40, "15m": 30, "5m": 30, "1m": 20 };
  const avail = Object.keys(candlesN).filter((k) => Array.isArray(candlesN[k]) && candlesN[k].length > 0);
  if (!avail.length) return { ...base, why: ["no candle data for any matrix timeframe"] };

  // 4H anchor VP -> macro value area.
  const avp = volumeProfile(candlesN["4h"] || []);
  if (avp) base.zones.anchor = { poc: avp.poc, vaLow: avp.valueAreaLow, vaHigh: avp.valueAreaHigh, approx: avp.approximation };
  base.legs.avp = avp ? "ok" : "insufficient";

  // 30M session VP + VWAP -> daily bias + fair value.
  const v30 = candlesN["30m"] || [];
  const svp = volumeProfile(v30, { levels: 24 });
  const vw = vwap(v30);
  const c30 = v30[v30.length - 1];
  const priceVsVwap = vw != null && c30 ? (c30.close - vw) / Math.abs(vw || 1) : null;
  if (svp) base.zones.session = { poc: svp.poc, vaLow: svp.valueAreaLow, vaHigh: svp.valueAreaHigh };
  base.legs.svp = svp ? "ok" : "insufficient";
  base.legs.vwap = vw != null ? +vw.toFixed(5) : null;

  // 15M VWAP trend of fair value.
  const vw15 = vwap(candlesN["15m"] || []);
  base.legs.vwap15m = vw15 != null ? +vw15.toFixed(5) : null;

  // 5M CVD proxy slope.
  const cvd = cvdProxy(candlesN["5m"] || [], 20);
  base.legs.cvd5m = cvd != null ? +cvd.toFixed(1) : null;
  base.legs.cvdIsProxy = true;

  // 1M RVOL spike.
  const rv = rvol(candlesN["1m"] || [], 20);
  base.legs.rvol1m = rv != null ? +rv.toFixed(2) : null;
  base.legs.rvol1mQuality = candlesN["1m"]?.length >= 22 ? "1m" : (candlesN["1m"]?.length ? "fallback" : "none");

  // ---- net bias: price-vs-VWAP (weight 3), CVD proxy (2), 15M VWAP slope (1)
  let score = 0;
  const why = [];
  if (priceVsVwap != null) {
    if (priceVsVwap > 0.001) { score += 3; why.push(`30M price above session VWAP (+${(priceVsVwap * 100).toFixed(2)}%)`); }
    else if (priceVsVwap < -0.001) { score -= 3; why.push(`30M price below session VWAP (${(priceVsVwap * 100).toFixed(2)}%)`); }
    else why.push("30M price at session VWAP — no bias");
  }
  if (vw15 != null && vw != null) {
    const slope = (vw15 - vw) / Math.abs(vw || 1);
    if (slope > 0.001) { score += 1; why.push("15M fair value rising above session VWAP"); }
    else if (slope < -0.001) { score -= 1; why.push("15M fair value falling below session VWAP"); }
  }
  if (cvd != null) {
    // Normalize CVD by session volume so magnitude is comparable.
    const svol = v30.reduce((a, c) => a + (c.volume || 0), 0) || 1;
    const cd = cvd / svol;
    if (cd > 0.15) { score += 2; why.push(`5M buy aggression (cvd proxy +${cd.toFixed(2)})`); }
    else if (cd < -0.15) { score -= 2; why.push(`5M sell aggression (cvd proxy ${cd.toFixed(2)})`); }
    else why.push("5M flow balanced (cvd proxy flat)");
  }
  if (rv != null && rv >= 1.5) {
    why.push(`1M volume spike ${rv.toFixed(1)}x — order-flow event, favor spike-direction entries`);
    // A spike amplifies the existing bias rather than creating one.
    score += score > 0 ? 1 : score < 0 ? -1 : 0;
  }
  if (rv != null && rv >= 3) why.push("CAUTION: 3x+ volume spike often marks exhaustion — scale in, don't chase");

  base.bias = score >= 2 ? "BULL" : score <= -2 ? "BEAR" : "NEUTRAL";
  base.confidence = Math.min(85, 30 + Math.abs(score) * 10 + (rv >= 1.5 ? 5 : 0));
  base.score = score;
  base.why = why.length ? why : ["matrix computed but no consensus"];
  base.zones.approx = "candle-distributed VP (close to true VP; wide bars blur zones)";
  return base;
}
