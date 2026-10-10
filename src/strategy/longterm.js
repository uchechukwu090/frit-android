// ============================================================================
// FRIT LONG-TERM ENGINE — daily trend system for investors (FX, crypto, ETFs)
// ----------------------------------------------------------------------------
// Weekly-style position trading on daily closes: 200DMA regime, 50/200 golden
// cross, MACD momentum, RSI guard, Donchian-55 breakout entries, 2xATR stops,
// 4xATR targets (2R). No leverage assumptions, no intraday noise — a setup a
// week is a busy month. Output decisions are BUY/SELL/WAIT with entry/SL/TP
// so the existing risk gate, bridge, journal and phone UI consume them
// unchanged. Needs 210+ daily closes; fewer = honest WAIT, never a signal.
// ============================================================================

const SMA = (arr, n) => {
  if (arr.length < n) return null;
  const w = arr.slice(-n);
  return w.reduce((a, b) => a + b, 0) / n;
};

function rsi(closes, n = 14) {
  if (closes.length < n + 1) return null;
  let g = 0, l = 0;
  for (let i = closes.length - n; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    if (d > 0) g += d; else l -= d;
  }
  if (l === 0) return 100;
  const rs = (g / n) / (l / n);
  return 100 - 100 / (1 + rs);
}

function atr(closes, highs, lows, n = 14) {
  // Closes-only fallback: mean absolute daily change (no H/L available).
  if (!highs || !lows || highs.length < n || lows.length < n) {
    if (closes.length < n + 1) return null;
    let s = 0;
    for (let i = closes.length - n; i < closes.length; i++) s += Math.abs(closes[i] - closes[i - 1]);
    return s / n;
  }
  let s = 0;
  for (let i = closes.length - n; i < closes.length; i++) {
    const pc = closes[i - 1];
    s += Math.max(highs[i] - lows[i], Math.abs(highs[i] - pc), Math.abs(lows[i] - pc));
  }
  return s / n;
}

function emaSeries(arr, n) {
  if (arr.length < n) return null;
  const k = 2 / (n + 1);
  let e = arr.slice(0, n).reduce((a, b) => a + b, 0) / n;
  const out = [e];
  for (let i = n; i < arr.length; i++) { e = arr[i] * k + e * (1 - k); out.push(e); }
  return out;
}

function macdHist(closes) {
  const f = emaSeries(closes, 12), s = emaSeries(closes, 26);
  if (!f || !s) return null;
  const off = f.length - s.length;
  const line = s.map((v, i) => f[off + i] - v);
  const sig = emaSeries(line, 9);
  if (!sig) return null;
  return line[line.length - 1] - sig[sig.length - 1];
}

function donchian(closes, n = 55) {
  if (closes.length < n) return null;
  const w = closes.slice(-n);
  return { high: Math.max(...w), low: Math.min(...w) };
}

function smaAt(closes, n, back = 0) {
  const arr = back > 0 ? closes.slice(0, -back) : closes;
  return SMA(arr, n);
}

/**
 * @param closes ascending daily closes (210+)
 * @param opts {highs, lows} optional daily H/L for true ATR
 */
export function analyzeLongterm(symbol, closes, opts = {}) {
  const sym = String(symbol || "").toUpperCase();
  const wait = (why) => ({
    symbol: sym, strategy: "donchian_trend_lt", decision: "WAIT",
    confidence: 0, entry: null, sl: null, tp: null, rr: 2.0,
    reasons: Array.isArray(why) ? why : [why], regime: "UNKNOWN", details: null,
  });
  if (!Array.isArray(closes) || closes.length < 210 || !closes.every((c) => c > 0)) {
    return wait("need 210+ daily closes — no history, no signal");
  }
  const S = closes[closes.length - 1];
  const ma50 = SMA(closes, 50), ma200 = SMA(closes, 200);
  const rsi14 = rsi(closes, 14);
  const atr14 = atr(closes, opts.highs, opts.lows, 14);
  const hist = macdHist(closes);
  const dc = donchian(closes, 55);
  if ([ma50, ma200, rsi14, atr14, hist].some((v) => v == null) || !dc) {
    return wait("indicator math failed on this history — no signal");
  }
  const regime = S >= ma200 ? "BULL" : "BEAR";
  // Freshness: cross happened within the last 10 sessions.
  let crossUp = false, crossDn = false;
  for (let b = 0; b < 10; b++) {
    const f = smaAt(closes, 50, b), s = smaAt(closes, 200, b);
    const fPrev = smaAt(closes, 50, b + 1), sPrev = smaAt(closes, 200, b + 1);
    if (f == null || s == null || fPrev == null || sPrev == null) break;
    if (fPrev <= sPrev && f > s) crossUp = true;
    if (fPrev >= sPrev && f < s) crossDn = true;
  }
  const alignedUp = ma50 > ma200, alignedDn = ma50 < ma200;
  const brokeHigh = S >= dc.high * 0.999;
  const brokeLow = S <= dc.low * 1.001;
  const reasons = [`200DMA regime: ${regime} (price ${S} vs ${ma200.toFixed(5)})`];
  let decision = "WAIT", conf = 0;

  // RSI is one-sided on purpose: breakouts are overbought by construction, so
  // there is no ceiling on longs (and no floor on shorts). The guard only
  // vetoes buying into weakness (RSI<35) or shorting into strength (RSI>65).
  // Tops and bottoms are handled by the 2xATR stop, not the oscillator.
  const longOk = regime === "BULL" && (crossUp || alignedUp) && rsi14 > 35 && brokeHigh;
  const shortOk = regime === "BEAR" && (crossDn || alignedDn) && rsi14 < 65 && brokeLow;

  if (longOk || shortOk) {
    const isLong = longOk;
    decision = isLong ? "BUY" : "SELL";
    // MACD is confidence-only, never a gate: on smooth grinds the histogram
    // hovers near zero and flips sign randomly — gating on it is a coin flip.
    const macdWith = isLong ? hist > 0 : hist < 0;
    conf = 55
      + ((isLong ? crossUp : crossDn) ? 10 : 0)
      + (macdWith ? 10 : 0)
      + 5  // RSI guard passed
      + 10 // Donchian breakout
      + ((isLong ? regime === "BULL" : regime === "BEAR") ? 5 : 0);
    conf = Math.max(20, Math.min(90, conf));
    const slDist = 2 * atr14;
    const entry = S;
    const sl = isLong ? entry - slDist : entry + slDist;
    const tp = isLong ? entry + 4 * atr14 : entry - 4 * atr14;
    reasons.push(
      `${isLong ? "Golden" : "Death"} alignment 50/200${(isLong ? crossUp : crossDn) ? " (fresh cross ≤10d)" : " (persistent)"}`,
      `MACD histogram ${hist >= 0 ? "+" : ""}${hist.toFixed(5)} ${isLong ? "expanding up" : "expanding down"}`,
      `RSI14 ${rsi14.toFixed(1)} — momentum, not exhaustion`,
      `Donchian-55 ${isLong ? "breakout" : "breakdown"} (55d ${isLong ? "high" : "low"} ${isLong ? dc.high.toFixed(5) : dc.low.toFixed(5)})`,
      `Stop 2xATR, target 4xATR (2R). Investors: trail below fast DMA once +1R; exit on weekly close back under 50DMA.`
    );
    return {
      symbol: sym, strategy: "donchian_trend_lt", decision, confidence: conf,
      entry, sl, tp, rr: 2.0, reasons, regime,
      details: {
        ma50, ma200, rsi14, atr14, macdHist: hist,
        donchianHigh: dc.high, donchianLow: dc.low,
        crossFresh: isLong ? crossUp : crossDn,
      },
    };
  }
  if (regime === "BEAR") reasons.push("below 200DMA — investors stand aside, no longs");
  else reasons.push("above 200DMA but no breakout + momentum alignment — wait for the setup");
  if (!brokeHigh && !brokeLow) reasons.push("inside the 55-day range — nothing to break out of yet");
  return {
    symbol: sym, strategy: "donchian_trend_lt", decision, confidence: 0,
    entry: null, sl: null, tp: null, rr: 2.0, reasons, regime,
    details: { ma50, ma200, rsi14, atr14, macdHist: hist, donchianHigh: dc.high, donchianLow: dc.low },
  };
}
