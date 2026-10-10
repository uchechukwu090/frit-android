// ============================================================================
// FRIT FX OPTION-STYLE SIGNALS — model-based, explicitly synthetic
// ----------------------------------------------------------------------------
// There is NO free listed FX options chain, so these signals are COMPUTED,
// not quoted: realized-vol → IV proxy → expected-move bands → strangle
// premium estimate via Black-Scholes. Every output carries synthetic: true
// and a disclaimer saying so. They answer the retail-usable question:
// "how far can price plausibly run, and is premium expanding or compressing?"
//
// Spot tactics attached to each signal:
//   - COMPRESSION (rv20/rv60 < 0.8): fade the band edges, tight invalidation.
//   - EXPANSION   (rv20/rv60 > 1.3): DO NOT fade — breakout regime, stand down
//     or trade the break with the band as the trigger line.
//   - NORMAL: fade edges only with the MTF engine's agreement (caller joins).
// Output is a SIGNAL. Nothing here places orders.
// ============================================================================

import { bsPrice, expectedMove } from "./pricing.js";

export const FX_DEFAULTS = {
  ivMarkup: 0.20,        // IV proxy = RV20 x 1.20 (sellers' markup guess)
  compression: 0.8, expansion: 1.3,
  bandDays: 1,           // 1-day expected-move bands
  strangleDte: 7,
  periodsPerYear: 252,   // daily closes
};

function rvOf(closes, len) {
  if (closes.length < len + 1) return null;
  const w = closes.slice(-len - 1);
  const rets = [];
  for (let i = 1; i < w.length; i++) {
    if (!(w[i] > 0) || !(w[i - 1] > 0)) return null;
    rets.push(Math.log(w[i] / w[i - 1]));
  }
  const m = rets.reduce((a, b) => a + b, 0) / rets.length;
  const v = rets.reduce((a, b) => a + (b - m) ** 2, 0) / (rets.length - 1 || 1);
  return Math.sqrt(v * 252);
}

/**
 * @param closes ascending daily closes (65+ for RV60)
 */
export function scanFx({ symbol, closes, cfg = {} }) {
  const C = { ...FX_DEFAULTS, ...cfg };
  const S = closes[closes.length - 1];
  if (!(S > 0)) return { skip: true, why: "no spot price" };
  const rv20 = rvOf(closes, 20), rv60 = rvOf(closes, 60);
  if (rv20 == null) return { skip: true, why: "need 21+ closes" };
  const ratio = rv60 != null ? rv20 / rv60 : 1;
  const ivProxy = rv20 * (1 + C.ivMarkup);
  const em1 = expectedMove(S, ivProxy, C.bandDays);
  const em7 = expectedMove(S, ivProxy, 7);
  if (em1 == null) return { skip: true, why: "expected-move math failed" };
  const regime = ratio < C.compression ? "COMPRESSION" : ratio > C.expansion ? "EXPANSION" : "NORMAL";
  // ±1σ 7-day strangle premium estimate (OTM strikes, modeled IV).
  const T = C.strangleDte / 365;
  const callK = S + em7, putK = S - em7;
  const callPx = bsPrice({ S, K: callK, T, r: 0.02, q: 0.01, sigma: ivProxy, type: "call" });
  const putPx = bsPrice({ S, K: putK, T, r: 0.02, q: 0.01, sigma: ivProxy, type: "put" });
  const tactic = regime === "EXPANSION"
    ? `Do NOT fade — breakout regime. Use ${(S + em1).toFixed(5)} / ${(S - em1).toFixed(5)} as trigger lines for momentum, not targets.`
    : regime === "COMPRESSION"
      ? `Fade ${(S + em1).toFixed(5)} / ${(S - em1).toFixed(5)} toward spot with tight invalidation beyond the band.`
      : `Fade edges only with trend-engine agreement; otherwise stand by.`;
  return {
    signal: {
      id: `fxopt_${symbol}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      symbol, kind: "FX_EXPECTED_MOVE", decision: "FX_LEVELS",
      spot: S,
      band1d: [+(S - em1).toFixed(5), +(S + em1).toFixed(5)],
      band7d: [+(S - em7).toFixed(5), +(S + em7).toFixed(5)],
      rv20: +(rv20 * 100).toFixed(2), rv60: rv60 != null ? +(rv60 * 100).toFixed(2) : null,
      ivProxy: +(ivProxy * 100).toFixed(2), regime,
      strangle7d: {
        callStrike: +callK.toFixed(5), putStrike: +putK.toFixed(5),
        estPremium: callPx != null && putPx != null ? +((callPx + putPx)).toFixed(5) : null,
      },
      tactic,
      reasoning: [
        `1-day expected move ±${em1.toFixed(5)} from ${S} (modeled ${(ivProxy * 100).toFixed(1)}% IV)`,
        `Regime ${regime}: RV20 ${(rv20 * 100).toFixed(1)}%${rv60 != null ? ` vs RV60 ${(rv60 * 100).toFixed(1)}%` : ""}`,
        tactic,
      ],
      synthetic: true,
      disclaimer: "Computed from realized vol — not a market quote. No listed FX chain exists at $0.",
      timestamp: Date.now(),
    },
  };
}
