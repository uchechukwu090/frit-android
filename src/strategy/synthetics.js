// ============================================================================
// FRIT SYNTHETICS ENGINE — generator-aware analysis for Deriv synthetics
// ----------------------------------------------------------------------------
// Why the FX engines do NOT transfer: RSI/MAs/ATR assume order flow with
// persistent regimes (trends continue, pullbacks retrace structure). Deriv
// synthetics are engineered RNG: constant-volatility random walks (R_*/1HZ*),
// scheduled jump processes (BOOM/CRASH), or fixed drift (RDBULL/RDBEAR).
// There is no order flow to read, no sessions, no gaps, no news. A
// moving-average crossover on a martingale is a coin flip minus costs.
//
// So this engine does not look for momentum. It looks for what each
// generator actually does:
//   VOL   (R_10..R_100, 1HZ*, STPRNG): constant-vol random walk. Only known
//          edge shape is mean-reversion to the rolling mean — fade |z|>2
//          extremes, and ONLY when volatility is normal-or-high (a volatility
//          squeeze precedes expansion: never fade a coiling market). Time stop
//          on every trade: reversion either happens fast or the regime moved.
//   SPIKE (BOOM*N, CRASH*N, JD*, GDA*): jump processes. Spikes are
//          unpredictable — never positioned FOR. Post-spike reversion is the
//          only trade: a tick-velocity spike (>8x median tick size) fades back
//          toward the pre-spike level with a tight invalidation beyond the
//          spike extreme. Short hold, small size.
//   TREND (RDBULL*, RDBEAR*, DEX*UP/DN): engineered drift/bias. Direction-
//          locked Donchian breakout (UP/BULL buys only, DN/BEAR sells only).
//          This is the one family where trend logic belongs.
//   REGIME (DSI10/20/30): drift-switching bull/bear/sideways regimes lasting
//          ~N minutes on average. Follow significant drift with a time stop
//          at half the average regime life; stand aside when driftless.
// Unknown codes -> WAIT. Not knowing the generator means not trading it.
// MULT contracts have no spread (payout is formulaic), so there is no cost
// gate here — stake sizing (executor-level) is the risk control.
// ============================================================================

export function classifyFamily(symbol) {
  const s = String(symbol || "").toUpperCase();
  if (/^(R_\d+|1HZ\d+V?|STPRNG\d*)$/.test(s)) return "VOL";
  if (/^(BOOM\d+N|CRASH\d+N|JD\d+|GDA\d+)$/.test(s)) return "SPIKE";
  if (/^RDBULL/.test(s)) return { family: "TREND", lock: "BUY" };
  if (/^RDBEAR/.test(s)) return { family: "TREND", lock: "SELL" };
  // DEX directional-bias indices trade like locked trends (UP buys only).
  const dex = s.match(/^DEX\d+(UP|DN)$/);
  if (dex) return { family: "TREND", lock: dex[1] === "UP" ? "BUY" : "SELL" };
  // Drift-switching indices (DSI10/20/30): bull/bear/sideways regimes that
  // last ~N minutes on average. Number captured for the time stop.
  const dsi = s.match(/^DSI(\d+)$/);
  if (dsi) return { family: "REGIME", regimeMin: Math.max(5, Math.min(60, Number(dsi[1]))) };
  return "UNKNOWN";
}

const sma = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const stdev = (a) => {
  if (a.length < 2) return 0;
  const m = sma(a);
  return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1));
};
const median = (a) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

function atrOf(closes, n = 14) {
  if (closes.length < n + 1) return null;
  let s = 0;
  for (let i = closes.length - n; i < closes.length; i++) s += Math.abs(closes[i] - closes[i - 1]);
  return s / n;
}

// ---- VOL family: z-score fade ----------------------------------------------
function analyzeVol(symbol, closes, cfg = {}) {
  const N = cfg.lookback || 20;
  const zMin = cfg.zMin || 2.0;
  if (!Array.isArray(closes) || closes.length < N + 60) {
    return { decision: "WAIT", why: `need ${N + 60}+ closes for vol bands` };
  }
  const S = closes[closes.length - 1];
  const win = closes.slice(-N);
  const mean = sma(win);
  const sd = stdev(win);
  if (!(sd > 0)) return { decision: "WAIT", why: "zero variance window" };
  const z = (S - mean) / sd;
  // Squeeze guard: current 20-bar vol vs the full trailing history's vol.
  // Coiling markets expand — fading them is catching a falling knife
  // mid-compression. (Median-of-windows fails here: when the flat regime
  // dominates history the median itself is ~zero.)
  let volRatio = 1;
  if (closes.length >= 120) {
    const histVol = stdev(closes.slice(-120));
    volRatio = histVol > 0 ? sd / histVol : 1;
  }
  if (volRatio < 0.7) {
    return { decision: "WAIT", why: `volatility squeeze (current ${(volRatio * 100).toFixed(0)}% of trailing) — expansion coming, no fades` };
  }
  if (Math.abs(z) < zMin) return { decision: "WAIT", why: `|z|=${Math.abs(z).toFixed(2)} inside ${zMin} — no extreme` };
  const isShort = z > 0; // price stretched above mean -> fade down
  const tp = mean;
  const risk = Math.abs(S - mean) + 0.5 * sd;
  const sl = isShort ? S + risk : S - risk;
  const rr = Math.abs(tp - S) / (risk || 1);
  return {
    decision: isShort ? "SELL" : "BUY",
    entry: S, sl, tp, rr: +rr.toFixed(2),
    confidence: Math.max(40, Math.min(80, Math.round(45 + (Math.abs(z) - zMin) * 20))),
    maxHoldMin: cfg.maxHoldMin || 30,
    reasons: [
      `VOL fade: price ${Math.abs(z).toFixed(2)}σ from ${N}-bar mean (${S} vs ${mean.toFixed(5)})`,
      `volatility normal-or-high (${(volRatio * 100).toFixed(0)}% of trailing) — no squeeze`,
      `target the mean, stop beyond entry + 0.5σ. Reversion or regime — time stop ${cfg.maxHoldMin || 30}min decides.`,
    ],
    details: { z: +z.toFixed(3), mean, sd, volRatio: +volRatio.toFixed(2) },
  };
}

// ---- SPIKE family: post-spike reversion ------------------------------------
function analyzeSpike(symbol, ticks, cfg = {}) {
  // ticks: ascending recent tick prices (need 60+). Spike = last move > k x
  // median tick size, where k defaults 8.
  const k = cfg.spikeK || 8;
  if (!Array.isArray(ticks) || ticks.length < 60) {
    return { decision: "WAIT", why: "need 60+ ticks for spike detection" };
  }
  const moves = [];
  for (let i = 1; i < ticks.length; i++) moves.push(Math.abs(ticks[i] - ticks[i - 1]));
  const med = median(moves);
  const last = Math.abs(ticks[ticks.length - 1] - ticks[ticks.length - 2]);
  if (!(med > 0)) return { decision: "WAIT", why: "dead tape" };
  if (last < k * med) {
    return { decision: "WAIT", why: `no spike (last move ${(last / med).toFixed(1)}x median, need ${k}x) — spikes can't be predicted, only faded` };
  }
  const dirUp = ticks[ticks.length - 1] > ticks[ticks.length - 2];
  const preSpike = ticks[ticks.length - 2];
  const S = ticks[ticks.length - 1];
  // Fade back toward pre-spike; invalidation just beyond the spike extreme.
  const sl = dirUp ? S + 1.5 * med : S - 1.5 * med;
  return {
    decision: dirUp ? "SELL" : "BUY",
    entry: S, sl, tp: preSpike,
    rr: +((Math.abs(preSpike - S) / Math.abs(sl - S)) || 0).toFixed(2),
    confidence: 55,
    maxHoldMin: cfg.maxHoldMin || 15,
    reasons: [
      `SPIKE detected: last tick ${(last / med).toFixed(0)}x median size — fading back to pre-spike ${preSpike}`,
      `stop beyond the spike extreme. Spikes cluster: one reversion attempt only, then stand down ${cfg.maxHoldMin || 15}min.`,
    ],
    details: { spikeMult: +(last / med).toFixed(1), medianTick: med },
  };
}

// ---- REGIME family: drift-switching indices (DSI10/20/30) --------------------
// Documented property: bull/bear/sideways regimes lasting ~N minutes on
// average. Trade: measure drift over the last N/2 minutes (1m closes); a drift
// beyond 1.5σ is a regime in progress — follow it, with a time stop at N/2
// (never ride a regime past its average life) and a 1.5xATR stop. Insignificant
// drift means sideways regime: WAIT, never fade-chop (that's the VOL family's
// job on a different instrument).
function analyzeRegime(symbol, closes, regimeMin, cfg = {}) {
  const window = Math.max(5, Math.round(regimeMin / 2));
  if (!Array.isArray(closes) || closes.length < window + 15) {
    return { decision: "WAIT", why: `need ${window + 15}+ 1m closes for a ${regimeMin}min regime read` };
  }
  const w = closes.slice(-window);
  const diffs = [];
  for (let i = 1; i < w.length; i++) diffs.push(w[i] - w[i - 1]);
  const sd = stdev(diffs);
  if (!(sd > 0)) return { decision: "WAIT", why: "dead tape" };
  const drift = w[w.length - 1] - w[0];
  const z = drift / (sd * Math.sqrt(w.length));
  if (Math.abs(z) < 1.5) {
    return { decision: "WAIT", why: `drift z=${z.toFixed(2)} — sideways regime, stand aside` };
  }
  const isLong = z > 0;
  const atr = atrOf(closes, 14) || sd * 2;
  const S = closes[closes.length - 1];
  const slDist = 1.5 * atr;
  return {
    decision: isLong ? "BUY" : "SELL",
    entry: S, sl: isLong ? S - slDist : S + slDist,
    tp: isLong ? S + 2 * slDist : S - 2 * slDist, rr: 2.0,
    confidence: Math.max(45, Math.min(70, Math.round(50 + (Math.abs(z) - 1.5) * 15))),
    maxHoldMin: Math.round(regimeMin / 2),
    reasons: [
      `DSI regime read: ${window}min drift z=${z.toFixed(2)} ${isLong ? "up" : "down"} — following the active regime`,
      `time stop ${Math.round(regimeMin / 2)}min (half the average regime life); stop 1.5xATR, target 2R`,
    ],
    details: { driftZ: +z.toFixed(3), regimeMin, windowMin: window },
  };
}

// ---- TREND family: direction-locked Donchian ---------------------------------
function analyzeTrend(symbol, closes, lock, cfg = {}) {
  const N = cfg.donchian || 20;
  const atrN = cfg.atr || 14;
  if (!Array.isArray(closes) || closes.length < N + atrN + 5) {
    return { decision: "WAIT", why: `need ${N + atrN + 5}+ closes` };
  }
  const S = closes[closes.length - 1];
  const w = closes.slice(-N);
  const hi = Math.max(...w), lo = Math.min(...w);
  const atr = atrOf(closes, atrN);
  if (!(atr > 0)) return { decision: "WAIT", why: "no ATR" };
  const brokeHigh = S >= hi * 0.9995;
  const brokeLow = S <= lo * 1.0005;
  if (lock === "BUY" && brokeHigh) {
    return {
      decision: "BUY", entry: S, sl: S - 2 * atr, tp: S + 4 * atr, rr: 2.0, confidence: 65,
      maxHoldMin: cfg.maxHoldMin || 240,
      reasons: [`RDBULL breakout: ${N}-bar high ${hi.toFixed(5)}, direction locked long by generator`, `stop 2xATR, target 4xATR`],
      details: { donchianHigh: hi, donchianLow: lo, atr },
    };
  }
  if (lock === "SELL" && brokeLow) {
    return {
      decision: "SELL", entry: S, sl: S + 2 * atr, tp: S - 4 * atr, rr: 2.0, confidence: 65,
      maxHoldMin: cfg.maxHoldMin || 240,
      reasons: [`RDBEAR breakdown: ${N}-bar low ${lo.toFixed(5)}, direction locked short by generator`, `stop 2xATR, target 4xATR`],
    };
  }
  // Wrong-way break on a locked family is not a counter-trend signal — WAIT.
  return { decision: "WAIT", why: lock === "BUY" ? "RDBULL: no upside breakout — longs only, never fade" : "RDBEAR: no downside breakdown — shorts only, never fade" };
}

/**
 * @param closesOrTicks ascending closes (VOL/TREND) or raw ticks (SPIKE)
 */
export function analyzeSynth(symbol, closesOrTicks, opts = {}) {
  const sym = String(symbol || "").toUpperCase();
  const fam = classifyFamily(sym);
  const family = typeof fam === "string" ? fam : fam.family;
  const base = { symbol: sym, strategy: `synth_${family.toLowerCase()}`, family, confidence: 0, entry: null, sl: null, tp: null, rr: null, reasons: [], regime: family };
  if (family === "UNKNOWN") {
    return { ...base, decision: "WAIT", reasons: [`unknown generator for ${sym} — not knowing the process means not trading it`] };
  }
  const arr = Array.isArray(closesOrTicks) ? closesOrTicks.filter((c) => c > 0) : [];
  let out;
  if (family === "VOL") out = analyzeVol(sym, arr, opts);
  else if (family === "SPIKE") out = analyzeSpike(sym, arr, opts);
  else if (family === "REGIME") out = analyzeRegime(sym, arr, fam.regimeMin, opts);
  else out = analyzeTrend(sym, arr, fam.lock, opts);
  if (out.decision === "WAIT") return { ...base, decision: "WAIT", reasons: [out.why] };
  return { ...base, ...out };
}
