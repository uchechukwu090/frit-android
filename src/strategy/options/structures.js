// ============================================================================
// FRIT OPTIONS STRUCTURES — defined-risk spread builder + cost-aware gating
// ----------------------------------------------------------------------------
// Strategy A only: vertical credit spreads and iron condors. Naked legs are
// rejected at construction (maxLoss must be finite), not at execution.
// Money math is per-share; multiply by 100 x contracts for dollars.
// Costs follow the repo's costs.js philosophy: a spread whose NET credit
// (after per-contract commission+fees and the bid/ask cross) is <= 0 is
// negative-expectancy by construction and blocked here.
// ============================================================================

import { greeks, normCdf } from "./pricing.js";

const MULT = 100; // shares per equity-option contract

function legValue(leg) {
  return (leg.side === "short" ? 1 : -1) * leg.mid;
}

/**
 * Vertical credit spread. legs: two entries {type, strike, side, mid, bid, ask}.
 * Short leg must be OTM vs S and nearer S than the long leg.
 */
export function creditSpread({ S, legs, contracts = 1, dte = null }) {
  if (!Array.isArray(legs) || legs.length !== 2) return null;
  const [a, b] = legs;
  const short = legs.find((l) => l.side === "short");
  const long = legs.find((l) => l.side === "long");
  if (!short || !long || short.type !== long.type) return null;
  if (!(S > 0) || !(short.mid >= 0) || !(long.mid >= 0)) return null;
  const isPut = short.type === "put";
  const width = isPut ? short.strike - long.strike : long.strike - short.strike;
  if (!(width > 0)) return null; // long leg must cap the short leg — else naked
  // Short leg must be OTM (defined-risk premium selling, never ITM shorts).
  if (isPut && short.strike >= S) return null;
  if (!isPut && short.strike <= S) return null;
  const credit = short.mid - long.mid;
  if (!(credit > 0)) return null; // debit, not a credit spread
  const maxLoss = width - credit;
  const breakeven = isPut ? short.strike - credit : short.strike + credit;
  return {
    kind: isPut ? "bull_put_spread" : "bear_call_spread",
    legs, contracts, dte,
    credit, width, maxLoss,
    maxProfit: credit,
    rr: credit / maxLoss,
    breakevens: [breakeven],
    maxProfitDollars: credit * MULT * contracts,
    maxLossDollars: maxLoss * MULT * contracts,
  };
}

/** Iron condor from two wing credits. All four strikes must straddle S OTM. */
export function ironCondor({ S, putShort, putLong, callShort, callLong, putCredit, callCredit, contracts = 1, dte = null }) {
  for (const v of [S, putShort, putLong, callShort, callLong, putCredit, callCredit]) {
    if (!(v > 0) && v !== 0) return null;
  }
  if (!(putLong < putShort && putShort < S && S < callShort && callShort < callLong)) return null;
  if (!(putCredit > 0) || !(callCredit > 0)) return null;
  const credit = putCredit + callCredit;
  const maxLoss = Math.min(putShort - putLong, callLong - callShort) - credit;
  if (!(maxLoss > 0)) return null;
  return {
    kind: "iron_condor",
    legs: [
      { type: "put", strike: putShort, side: "short", mid: putCredit + 0 }, // per-wing detail lives in scan output
      { type: "put", strike: putLong, side: "long", mid: 0 },
      { type: "call", strike: callShort, side: "short", mid: 0 },
      { type: "call", strike: callLong, side: "long", mid: 0 },
    ],
    contracts, dte,
    credit, width: Math.min(putShort - putLong, callLong - callShort),
    maxLoss, maxProfit: credit,
    rr: credit / maxLoss,
    breakevens: [putShort - credit, callShort + credit],
    maxProfitDollars: credit * MULT * contracts,
    maxLossDollars: maxLoss * MULT * contracts,
  };
}

/**
 * Net credit in dollars after costs. costSpec: {perContract} — commission +
 * exchange + regulatory fees per contract per round trip. The bid/ask cross is
 * already in `mid`-vs-fill conservatism at scan time (fills modeled at the
 * touch side, never mid — see Phase 1 scanner).
 */
export function netCredit(structure, costSpec = { perContract: 0.65 }) {
  if (!structure) return null;
  const perContract = Number(costSpec.perContract) || 0;
  const legs = structure.legs.length;
  const fees = perContract * legs * (structure.contracts || 1);
  return structure.maxProfitDollars - fees;
}

/** Block/weak gate mirroring costs.js: net credit must clear fees with margin. */
export function creditGate(structure, costSpec) {
  const net = netCredit(structure, costSpec);
  if (net == null) return { ok: false, reason: "no structure" };
  if (net <= 0) return { ok: false, reason: `fees consume the credit (net $${net.toFixed(2)})` };
  const ratio = net / structure.maxProfitDollars;
  if (ratio < 0.7) return { ok: true, weak: true, reason: `fees eat ${(100 - ratio * 100).toFixed(0)}% of credit` };
  return { ok: true, weak: false, reason: "" };
}

/** Net signed greeks of a spread's legs (long legs negate). Null leg greeks are skipped. */
export function spreadGreeks(legGreeks) {
  const out = { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
  for (const lg of legGreeks || []) {
    const sign = lg.side === "short" ? -1 : 1;
    for (const k of Object.keys(out)) out[k] += sign * (Number(lg[k]) || 0);
  }
  return out;
}

/** Greeks of a priced structure from per-leg quotes. Needs sigma/T/r/q per leg. */
export function priceStructureGreeks(legs, { T, r = 0, q = 0 }) {
  const per = [];
  for (const leg of legs) {
    if (!(leg.sigma > 0)) continue;
    const g = greeks({ S: leg.S, K: leg.strike, T, r, q, sigma: leg.sigma, type: leg.type });
    if (g) per.push({ ...g, side: leg.side });
  }
  return spreadGreeks(per);
}

/**
 * Approximate probability the short strikes survive to expiry under a normal
 * move assumption: P(S_T above put-short) for bull-put, etc. Uses the
 * structure's own IV (sigma) — a ranking heuristic, not a forecast.
 */
export function probProfit({ S, breakeven, sigma, T, direction }) {
  if (!(S > 0) || !(sigma > 0) || !(T > 0)) return null;
  const z = Math.log(breakeven / S) / (sigma * Math.sqrt(T));
  return direction === "above" ? 1 - normCdf(z) : normCdf(z);
}
