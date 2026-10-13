// ============================================================================
// FRIT VOL ENGINE — volatility analysis for options (NOT direction)
// ----------------------------------------------------------------------------
// RSI/MAs/ATR answer "where is price going". Options are priced on a different
// question: "is the market's implied distribution mispriced vs reality".
// This engine answers that question from three measurements:
//   1. IV LEVEL vs history (rank/percentile): sell premium when high, buy
//      when low — the variance-risk-premium harvest, both directions.
//   2. TERM STRUCTURE slope: contango (far > near) favors calendars and
//      carries no event fear; backwardation (near > far) means stress or an
//      event — demand wider wings or stand down.
//   3. SKEW (25Δ put IV minus 25Δ call IV): steep put skew = crash pricing;
//      verticals tilt toward the cheap side.
// Plus event proximity (earnings/macro within the expiry): long premium into
// known events is lottery pricing — the selector says so explicitly instead
// of selling into a scheduled explosion.
//
// Output is a STRATEGY SELECTION, not a direction:
//   SELL_VERTICAL (high IV, no event) | IRON_CONDOR (high IV, range, no event)
//   LONG_STRADDLE (low IV or pre-event) | NONE (backwardation, squeeze, thin)
// The premium scanner consumes this: NONE vetoes, the rest pick the structure.
// ============================================================================

/**
 * @param expiries [{ expiration, dte, atmIv, put25dIv?, call25dIv? }]
 * @param ivHistory trailing ATM-IV history for rank (same expiry tenor ideally)
 * @param rv20 current 20d realized vol (decimal) for IV-vs-realized check
 * @param eventDays { within7d: bool, within21d: bool, name? }
 */
export function analyzeVol({ expiries = [], ivHistory = [], rv20 = null, eventDays = null } = {}) {
  const base = {
    strategy: "NONE", ivRank: null, ivHistoryN: ivHistory.length,
    termSlope: null, skew25d: null, regime: "UNKNOWN", reasons: [],
  };
  const tenors = expiries
    .filter((e) => e && e.atmIv > 0 && e.dte >= 7)
    .sort((a, b) => a.dte - b.dte);
  if (tenors.length < 1) {
    return { ...base, reasons: ["no clean expiry tenor for IV anchor"] };
  }
  const front = tenors[0];
  // IV rank against history (falls back to IV-vs-realized when history thin).
  let rank = null;
  const hist = ivHistory.filter((v) => v > 0);
  if (hist.length >= 20) {
    const lo = Math.min(...hist), hi = Math.max(...hist);
    rank = hi > lo ? Math.max(0, Math.min(100, ((front.atmIv - lo) / (hi - lo)) * 100)) : 50;
  }
  // Term slope: (far ATM IV - near ATM IV) / far, per 30d of DTE gap.
  let slope = null;
  if (tenors.length >= 2) {
    const back = tenors[tenors.length - 1];
    const gap = back.dte - front.dte;
    if (gap >= 14 && front.atmIv > 0) {
      slope = ((back.atmIv - front.atmIv) / front.atmIv) / (gap / 30);
    }
  }
  // 25Δ skew from the front tenor when quoted.
  let skew = null;
  if (front.put25dIv > 0 && front.call25dIv > 0) {
    skew = front.put25dIv - front.call25dIv;
  }
  const eventNear = !!(eventDays && (eventDays.within7d || eventDays.within21d));
  const eventName = eventDays?.name || "scheduled event";

  const reasons = [];
  if (rank != null) reasons.push(`IV rank ${rank.toFixed(0)} (n=${hist.length}) on the ${front.dte}DTE tenor`);
  else if (rv20 != null && rv20 > 0) {
    const premium = ((front.atmIv - rv20) / rv20) * 100;
    reasons.push(`no IV history — IV ${(front.atmIv * 100).toFixed(1)}% vs realized ${(rv20 * 100).toFixed(1)}% (${premium >= 0 ? "+" : ""}${premium.toFixed(0)}% premium)`);
  }
  if (slope != null) {
    reasons.push(slope >= 0
      ? `term structure in contango (+${(slope * 100).toFixed(1)}%/30d) — no near-term fear`
      : `term structure INVERTED (${(slope * 100).toFixed(1)}%/30d) — stress or event bid in front`);
  }
  if (skew != null) reasons.push(`25Δ skew ${skew >= 0 ? "+" : ""}${(skew * 100).toFixed(1)}pts (puts ${skew >= 0 ? "rich" : "cheap"} vs calls)`);
  if (eventNear) reasons.push(`${eventName} inside the holding window — premium is lottery-priced`);

  // ---- selection ----
  const highIv = rank != null ? rank >= 40 : (rv20 != null && front.atmIv > rv20 * 1.1);
  const lowIv = rank != null ? rank <= 25 : false;
  const stressed = slope != null && slope < -0.02;
  let strategy = "NONE", regime = "NORMAL";
  if (stressed) {
    regime = "STRESS";
    strategy = "NONE";
    reasons.push("backwardation veto: sell nothing into stress — wings reprice mid-trade");
  } else if (eventNear) {
    regime = "EVENT";
    strategy = "LONG_STRADDLE";
    reasons.push("event play: BUY the move (long straddle/strangle), never sell into a scheduled explosion");
  } else if (highIv) {
    regime = "HIGH_IV";
    strategy = "SELL_VERTICAL";
    reasons.push("rich premium, no event, no stress — sell verticals (condor when range-bound, single wing with trend)");
  } else if (lowIv) {
    regime = "LOW_IV";
    strategy = "LONG_STRADDLE";
    reasons.push("cheap premium — buy convexity (long straddle), defined risk, no naked short vega");
  } else {
    regime = "NORMAL";
    strategy = "NONE";
    reasons.push("fair premium, no edge shape — no structure pays for its risk here");
  }
  return {
    ...base, strategy, regime,
    ivRank: rank != null ? +rank.toFixed(1) : null,
    termSlope: slope != null ? +slope.toFixed(4) : null,
    skew25d: skew != null ? +skew.toFixed(4) : null,
    frontDte: front.dte, frontAtmIv: front.atmIv,
    reasons,
  };
}
