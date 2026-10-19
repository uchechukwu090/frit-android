// ============================================================================
// FRIT NEWS ENGINE — trade the event, don't just dodge it
// ----------------------------------------------------------------------------
// What already existed: a macro calendar FILTER (FF calendar, 30-min advisory)
// that says "careful around news". That is avoidance, not trading. This module
// is the trading engine around events, and it only needs data we already pay
// $0 for: Finnhub free earnings calendar + our own daily closes + the chain's
// own IV quotes.
//
// The edge, stated plainly: options around scheduled events are priced by
// FEAR, not by the event's actual size. So the tradeable question is always
// "is the market's implied move bigger or smaller than what usually happens?"
//
//   1. EARNINGS (stocks/ETFs). Historical realized 1-day moves around the last
//      N earnings give the empirical distribution. The ATM straddle premium
//      gives the market's implied move. implied > historical x 1.2 = premium
//      rich -> the event is OVERPRICED -> sell premium AFTER the event (IV
//      crush pays even when direction is right). implied < historical x 0.9 =
//      premium cheap -> buy the straddle BEFORE. Anything between = no edge.
//   2. MACRO (FX/metals). High-impact events don't have a listed chain, but
//      they do have a tell: pre-event ATR contraction. When realized vol
//      collapses into a scheduled event, expansion follows — so we place
//      breakout triggers at the compression range edges instead of guessing
//      direction. No chain needed, just daily closes + the FF calendar.
//
// Everything returns REGIME + why. "NONE" is a valid, common answer — an
// engine that always finds a trade is lying.
// ============================================================================

import { expectedMove } from "./options/pricing.js";

const DAY = 86400000;

// ---- shared helpers ----
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const stdev = (a) => {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1));
};

/**
 * Historical absolute 1-day returns around past earnings, from DATED closes.
 * @param datedCloses [{t: ms, c: price}] daily closes with real timestamps
 * @param earningsTs  ascending ms timestamps of past earnings dates
 * @param maxSamples  how many past events to measure (default 8)
 * @returns { n, avg, p75, worst } in RETURN terms (0.031 = 3.1%); nulls when
 *          there is not enough data to say anything honest.
 */
export function earningsMovesFromDated(datedCloses, earningsTs, maxSamples = 8) {
  if (!Array.isArray(datedCloses) || datedCloses.length < 10 || !Array.isArray(earningsTs)) {
    return { n: 0, avg: null, p75: null, worst: null };
  }
  const rows = [...datedCloses].filter((r) => r && r.c > 0 && Number.isFinite(r.t)).sort((a, b) => a.t - b.t);
  const moves = [];
  for (const ts of earningsTs) {
    if (moves.length >= maxSamples) break;
    const before = rows.filter((r) => r.t <= ts + 6 * 3600 * 1000).pop();
    const after = rows.find((r) => r.t > ts + 6 * 3600 * 1000);
    if (!before || !after || !(before.c > 0) || !(after.c > 0)) continue;
    moves.push(Math.abs(after.c / before.c - 1));
  }
  if (!moves.length) return { n: 0, avg: null, p75: null, worst: null };
  const sorted = [...moves].sort((a, b) => a - b);
  return {
    n: moves.length,
    avg: +mean(moves).toFixed(4),
    p75: +sorted[Math.floor(sorted.length * 0.75)].toFixed(4),
    worst: +sorted[sorted.length - 1].toFixed(4),
  };
}

/**
 * Earnings/news engine for US stocks + ETFs.
 * @param datedCloses [{t, c}] daily closes with real timestamps
 * @param atmStraddle ATM straddle premium per share (bid+ask touch, from chain)
 * @param earnings { nextTs, pastTs[] }
 * @param spot current underlying price
 */
export function analyzeEarningsEvent({ symbol, datedCloses, atmStraddle, spot, earnings }) {
  const sym = String(symbol || "").toUpperCase();
  const base = { symbol: sym, strategy: "news_earnings", family: "EVENT", regime: "NONE", decision: "WAIT", confidence: 0, reasons: [] };
  if (!(spot > 0) || !earnings?.nextTs) {
    return { ...base, regime: "NO_SCHEDULE", reasons: ["no upcoming earnings date — nothing to price"] };
  }
  const daysTo = Math.round((earnings.nextTs - Date.now()) / DAY);
  const hist = earningsMovesFromDated(datedCloses, earnings.pastTs || []);
  const impliedPct = atmStraddle > 0 ? atmStraddle / spot : null;

  const reasons = [
    `earnings in ${daysTo}d`,
    hist.avg != null ? `historical move avg ${(hist.avg * 100).toFixed(1)}% (worst ${(hist.worst * 100).toFixed(1)}%, n=${hist.n})` : "no historical earnings moves available",
    impliedPct != null ? `market-implied move ${(impliedPct * 100).toFixed(1)}%` : "no chain straddle for implied move",
  ];

  // No implied or no history -> we cannot compare, so no edge. Say so.
  if (impliedPct == null || hist.avg == null) {
    return { ...base, regime: "UNPRICEABLE", reasons: [...reasons, "need both a chain straddle and history — no comparison, no trade"] };
  }
  const ratio = impliedPct / hist.avg;
  reasons.push(`implied/historical = ${ratio.toFixed(2)}x`);

  if (daysTo < 0 || daysTo > 21) {
    return { ...base, regime: daysTo < 0 ? "POST_EVENT_OLD" : "TOO_FAR", reasons: [...reasons, "outside the tradeable window (-2d..21d)"] };
  }
  // RICH premium: event overpriced. The paid event is the IV crush AFTER.
  if (ratio >= 1.2 && daysTo <= 2) {
    return {
      ...base, regime: "PRE_EVENT_RICH", decision: "ARM_SELL_AFTER_EVENT", confidence: 60,
      reasons: [...reasons, "premium overprices the event — SELL premium after the release (crush pays even on a correct directional guess)"],
    };
  }
  // CHEAP premium: event underpriced -> own convexity before it happens.
  if (ratio <= 0.9) {
    return {
      ...base, regime: "PRE_EVENT_CHEAP", decision: "BUY_STRADDLE", confidence: 60,
      reasons: [...reasons, "premium underprices the event — buy the straddle before the release"],
    };
  }
  // Post-event within 2 days with rich IV: classic IV-crush sale.
  if (daysTo < 0 && ratio >= 1.0) {
    return {
      ...base, regime: "POST_EVENT_CRUSH", decision: "SELL_PREMIUM", confidence: 55,
      reasons: [...reasons, "event just passed and IV stays rich — sell the crush"],
    };
  }
  return { ...base, regime: "FAIR", reasons: [...reasons, "premium fairly prices the event — no edge, stand aside"] };
}

/**
 * Macro compression engine (FX/metals/crypto around high-impact news).
 * Pre-event ATR contraction is the tell: realized vol collapses into a
 * scheduled event, then expands. We place breakout triggers at the contraction
 * range edges — direction comes from the break, not from a guess.
 * No chain needed; works on daily closes + any high-impact calendar.
 */
export function analyzeMacroEvent({ symbol, closes, atr, eventWithinHours }) {
  const sym = String(symbol || "").toUpperCase();
  const base = { symbol: sym, strategy: "news_macro", family: "MACRO", regime: "NONE", decision: "WAIT", confidence: 0, reasons: [] };
  if (!(eventWithinHours > 0)) return { ...base, regime: "NO_EVENT", reasons: ["no high-impact event in the window"] };
  if (!Array.isArray(closes) || closes.length < 30 || !(atr > 0)) {
    return { ...base, regime: "UNPRICEABLE", reasons: ["need 30+ closes and ATR"] };
  }
  const w = closes.slice(-20);
  const sd = stdev(w) / mean(w); // relative realized vol over the last 20 sessions
  const range = Math.max(...w) - Math.min(...w);
  const compression = range / atr; // how many ATRs the range spans
  const reasons = [
    `high-impact event in ${eventWithinHours.toFixed(1)}h`,
    `20d relative vol ${(sd * 100).toFixed(2)}%`,
    `range spans ${compression.toFixed(1)} ATR`,
  ];
  // Contraction: the 20d range is tight relative to current ATR, and realized
  // vol is low. Expansion into the event tends to follow.
  if (compression <= 3 || sd <= 0.012) {
    const upper = Math.max(...w), lower = Math.min(...w);
    return {
      ...base, regime: "PRE_EVENT_COMPRESSION", decision: "BREAKOUT_ARMED", confidence: 55,
      triggers: { long: +upper.toFixed(5), short: +lower.toFixed(5), invalidBelow: +lower.toFixed(5), invalidAbove: +upper.toFixed(5) },
      reasons: [...reasons, "pre-event compression — trade the BREAK of the range edges, not a direction guess; no break, no trade"],
    };
  }
  return { ...base, regime: "PRE_EVENT_EXPANSION", reasons: [...reasons, "vol already expanding — breakout entries would be chased, stand aside"] };
}
