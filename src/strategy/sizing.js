// ============================================================================
// FRIT POSITION SIZING — Kelly criterion with a risk-profile fraction
// ----------------------------------------------------------------------------
// Why Kelly: a fixed 0.01 lot is either reckless on a $200 account or timid
// on a $50,000 one, and it ignores the edge you actually have. Kelly sizes
// the bet to the measured edge, so the same signal risks a different amount
// on a different account or a different win-rate.
//
// The maths (classic Kelly for a binary payoff):
//   f* = (p*b - q) / b      p = win rate, b = win/loss odds (RR), q = 1 - p
// Full Kelly is the growth-optimal bet — it is also the bet that gets you
// ruined after a losing streak, because it assumes the edge is permanent and
// the sample is large. So we never bet full Kelly:
//   AGGRESSIVE   -> 1/2 of Kelly
//   MODERATE     -> 1/4 of Kelly
//   CONSERVATIVE -> 1/8 of Kelly
// The fraction is the user's declared risk appetite (persisted per account),
// which is the only place that judgement belongs.
//
// SAFETY RAILS (applied AFTER Kelly, in this order):
//   1. Negative Kelly -> size 0. A negative edge is not a small bet, it is
//      NO bet. Sizing into a losing edge is how accounts die.
//   2. Hard per-trade risk cap (RISK_MAX_TRADE_PCT, default 2%) — Kelly can
//      ask for 30% of the account on a huge edge; the cap says no.
//   3. Min/max lot clamp so a rounding artifact can never place a 0.00 lot.
//   4. Confidence haircut — a 60%-confidence signal is scaled by 0.60, so a
//      weak signal can never be sized as if it were a strong one.
//
// Everything is computed in PRICE units first (pip math) and converted to
// lots last, so the same code is correct for FX, metals, crypto and indices
// regardless of quote conventions.
// ============================================================================

export const RISK_PROFILES = {
  AGGRESSIVE: { label: "Aggressive", kellyFraction: 0.5 },
  MODERATE: { label: "Moderate", kellyFraction: 0.25 },
  CONSERVATIVE: { label: "Conservative", kellyFraction: 0.125 },
};

export const DEFAULT_RISK_PROFILE = "MODERATE";

// Normalizes the many spellings that reach us (personal data is free text).
export function normalizeRiskProfile(v) {
  const s = String(v || "").trim().toUpperCase();
  if (!s) return DEFAULT_RISK_PROFILE;
  if (s.startsWith("AGGR") || s.startsWith("RISK") || s === "HIGH") return "AGGRESSIVE";
  if (s.startsWith("CONSERV") || s === "LOW" || s === "SAFE") return "CONSERVATIVE";
  if (s.startsWith("MODER") || s === "MED" || s === "BALANCED") return "MODERATE";
  return DEFAULT_RISK_PROFILE;
}

export function riskProfileInfo(profile) {
  const p = normalizeRiskProfile(profile);
  return { profile: p, ...RISK_PROFILES[p] };
}

/**
 * Kelly fraction of the account to risk on one trade.
 * @param winRate 0..1 measured win rate of the setup (0.45 default = "no edge")
 * @param rr      win/loss odds of the setup (the signal's R:R)
 * @param profile AGGRESSIVE | MODERATE | CONSERVATIVE
 * @returns { fraction, kelly, capped, reason }
 */
export function kellyFraction({ winRate = 0.45, rr = 1.5, profile = DEFAULT_RISK_PROFILE } = {}) {
  const p = Math.min(0.95, Math.max(0.05, Number(winRate) || 0.45));
  const b = Math.max(0.1, Number(rr) || 1.5);
  const q = 1 - p;
  const kelly = (p * b - q) / b; // classic Kelly
  if (!(kelly > 0)) {
    return {
      fraction: 0,
      kelly: Number(kelly.toFixed(4)),
      capped: false,
      reason: `no edge (win ${(p * 100).toFixed(0)}% at ${b.toFixed(2)}R is negative expectancy) — size 0`,
    };
  }
  const info = riskProfileInfo(profile);
  let fraction = kelly * info.kellyFraction;
  let reason = `Kelly ${(kelly * 100).toFixed(1)}% x ${info.label} ${(info.kellyFraction * 100).toFixed(1)}%`;
  // Hard per-trade backstop. This is a CEILING on an already-fractional
  // Kelly bet, not the sizing rule itself: 1/2 Kelly on a strong edge is a
  // large bet by design, and a 2% cap would collapse AGGRESSIVE and MODERATE
  // onto the same number — the profile would stop meaning anything. The
  // backstop only bites on absurd edges.
  const maxPct = Number(process.env.KELLY_MAX_TRADE_PCT) || 10;
  if (fraction * 100 > maxPct) {
    fraction = maxPct / 100;
    reason += ` — backstopped at ${maxPct}% per trade`;
  }
  return {
    fraction: Number(fraction.toFixed(6)),
    kelly: Number(kelly.toFixed(4)),
    capped: fraction * 100 >= maxPct - 1e-9,
    maxFraction: maxPct / 100,
    reason,
  };
}

/**
 * Dollar risk of `lot` lots over the entry→stop distance, priced from the
 * server's pip table. 0 when the symbol has no pip row (unknown risk).
 */
export function riskUsdForLot({ symbol, entry, sl, lot, pipConfig = {} } = {}) {
  const cfg = pipConfig[String(symbol || "").toUpperCase()];
  const e = Number(entry), s = Number(sl), l = Number(lot);
  if (!cfg || !(e > 0) || !(s > 0) || !(l > 0) || e === s) return 0;
  const dist = Math.abs(e - s);
  const risk = cfg.pipSize && cfg.pipValue
    ? (dist / cfg.pipSize) * cfg.pipValue * l
    : dist * 10 * l;
  return Number(risk.toFixed(2));
}

/**
 * FIXED-lot sizing: the user pinned the lot, so nobody re-derives it.
 * This is the "auto lot size" checkbox being OFF in the app — the whole point
 * is that the 0.01 hard cap disappears and the number the user typed is the
 * number that is sent. The per-trade risk gate still runs afterwards.
 */
export function fixedLotSizing({
  symbol, entry, sl, lot, pipConfig = {}, minLot = 0.01, maxLot = 100,
} = {}) {
  const n = Number(lot);
  const raw = Number.isFinite(n) && n > 0 ? n : minLot;
  const fixed = Number(Math.min(maxLot, Math.max(minLot, Math.round(raw * 100) / 100)).toFixed(2));
  const riskUsd = riskUsdForLot({ symbol, entry, sl, lot: fixed, pipConfig });
  return {
    lotSize: fixed,
    riskUsd,
    fraction: null,
    mode: "fixed",
    reason: `fixed ${fixed} lot(s) from your account settings (auto lot engine off)`
      + (riskUsd > 0 ? ` — risking about $${riskUsd.toFixed(2)} at this stop` : ""),
  };
}

/**
 * Lot size for a setup.
 * @param symbol    pair
 * @param balance   account balance the user entered (the sizing base)
 * @param entry     entry price
 * @param sl        stop-loss price (the risk distance)
 * @param tp        take-profit price (optional — only used for the RR input)
 * @param rr        planned R:R when tp is absent
 * @param winRate   measured win rate for this setup
 * @param profile   risk profile
 * @param pipConfig the server's PIP_CONFIG (pipSize / pipValue per symbol)
 * @param confidence 0..100 — applied as a haircut on the fraction
 */
export function recommendLotSize({
  symbol, balance = 1000, entry, sl, tp = null, rr = null,
  winRate = 0.45, profile = DEFAULT_RISK_PROFILE, pipConfig = {},
  confidence = null, minLot = 0.01, maxLot = 5,
} = {}) {
  const sym = String(symbol || "").toUpperCase();
  const bal = Number(balance) > 0 ? Number(balance) : 0;
  const e = Number(entry);
  const s = Number(sl);
  if (!(bal > 0)) {
    return { lotSize: 0, reason: "no account balance set — enter it in Watchlist to size trades", riskUsd: 0, fraction: 0 };
  }
  if (!(e > 0) || !(s > 0) || e === s) {
    return { lotSize: 0, reason: "no entry/SL — cannot size a trade without a stop distance", riskUsd: 0, fraction: 0 };
  }
  // Planned R:R from the levels when not supplied.
  const t = Number(tp);
  const plannedRr = rr != null && Number(rr) > 0
    ? Number(rr)
    : (t > 0 ? Math.abs(t - e) / Math.abs(e - s) : 1.5);

  const kelly = kellyFraction({ winRate, rr: plannedRr, profile });
  if (!(kelly.fraction > 0)) {
    return {
      lotSize: 0, riskUsd: 0, fraction: 0, kelly: kelly.kelly,
      reason: kelly.reason, plannedRr: Number(plannedRr.toFixed(2)),
    };
  }
  // Confidence haircut: a weak signal can never be sized like a strong one.
  let confFactor = 1;
  if (confidence != null && Number.isFinite(Number(confidence))) {
    confFactor = Math.min(1, Math.max(0.25, Number(confidence) / 100));
  }
  const fraction = kelly.fraction * confFactor;
  const riskUsd = bal * fraction;

  const cfg = pipConfig[sym] || {};
  const dist = Math.abs(e - s);
  let lotSize = 0;
  let basis = "";
  if (cfg.pipSize && cfg.pipValue) {
    // P&L per lot per unit price move = (move / pipSize) * pipValue.
    const perLot = (dist / cfg.pipSize) * cfg.pipValue;
    lotSize = perLot > 0 ? riskUsd / perLot : 0;
    basis = `${(dist / cfg.pipSize).toFixed(1)} pips at $${cfg.pipValue}/pip`;
  } else if (cfg.isCrypto) {
    const perLot = dist * (cfg.pipValue || 1);
    lotSize = perLot > 0 ? riskUsd / perLot : 0;
    basis = `${dist.toFixed(6)} price units per lot`;
  } else {
    // Unknown symbol: fall back to a conservative dollar approximation so a
    // missing PIP row can never silently produce a huge lot.
    const perLot = dist * 10;
    lotSize = perLot > 0 ? riskUsd / perLot : 0;
    basis = "approximate (no pip row for this symbol)";
  }
  const clamped = Math.max(minLot, Math.min(maxLot, Math.round(lotSize * 100) / 100));
  const finalLot = lotSize < minLot ? 0 : clamped;
  const finalRisk = finalLot > 0
    ? (cfg.pipSize && cfg.pipValue)
      ? ((dist / cfg.pipSize) * cfg.pipValue * finalLot)
      : (dist * 10 * finalLot)
    : 0;

  return {
    lotSize: finalLot,
    rawLotSize: Number(lotSize.toFixed(4)),
    riskUsd: Number(finalRisk.toFixed(2)),
    fraction: Number(fraction.toFixed(6)),
    kelly: kelly.kelly,
    riskProfile: normalizeRiskProfile(profile),
    plannedRr: Number(plannedRr.toFixed(2)),
    balance: bal,
    basis,
    reason: kelly.reason
      + (confFactor < 1 ? ` x confidence ${(confFactor * 100).toFixed(0)}%` : "")
      + ` -> risk $${finalRisk.toFixed(2)} of $${bal.toFixed(2)} (${((finalRisk / bal) * 100).toFixed(2)}%) at ${finalLot} lot(s) [${basis}]`,
  };
}
