// ============================================================================
// FRIT RETAIL TRANSACTION COST MODEL
// ----------------------------------------------------------------------------
// A signal's paper RR is a lie until costs are subtracted. Retail pays three
// times on every round trip:
//   1. SPREAD  — crossed in full on market entry (limit entries pay ~half on
//      the fill plus the exit cross; we model the conservative full cross).
//   2. COMMISSION — $/lot round-turn. Zero on most retail spread-only MT5
//      accounts, ~$7 on ECN. Per-symbol overrideable.
//   3. SLIPPAGE — modelled on both entry and exit (news/fast markets slip).
//
// All math here is LOT-INVARIANT and done in PRICE terms, so the engine can
// gate setups without knowing account size:
//   netRisk = grossRisk + costPrice        (costs widen what you can lose)
//   netWin  = grossWin  - costPrice        (costs shrink what you can win)
//   netRR   = netWin / netRisk
// Gates: netRR < 1.0  -> BLOCK (the trade loses money even when right).
//        netRR < 1.3  -> WEAK (confidence penalty, trade only with confluence).
//
// Defaults below are typical retail MT5 standard-account numbers, NOT tight
// ECN quotes. Override per deployment with COST_OVERRIDES_JSON, e.g.
//   {"XAUUSD":{"spread":0.28},"EURUSD":{"spread":0.00012,"commission":7}}
// Unknown symbols fall back to ~2bps of price (documented guess, flagged).
// USD conversion (for display / journal $ PnL) needs pip size+value, supplied
// by the caller (index.js PIP_CONFIG) — see costToUsd().
// ============================================================================

import { isSynthetic } from "./synthetics.js";

const DEFAULT_SYMBOL_COSTS = {
  // spread/slippage in PRICE units, commission in USD per 1.0 lot round-turn
  EURUSD: { spread: 0.00016, commission: 0, slippage: 0.00005 },
  GBPUSD: { spread: 0.00022, commission: 0, slippage: 0.00005 },
  AUDUSD: { spread: 0.00018, commission: 0, slippage: 0.00005 },
  NZDUSD: { spread: 0.00025, commission: 0, slippage: 0.00006 },
  USDCHF: { spread: 0.00018, commission: 0, slippage: 0.00005 },
  USDCAD: { spread: 0.00022, commission: 0, slippage: 0.00006 },
  EURGBP: { spread: 0.00018, commission: 0, slippage: 0.00005 },
  USDJPY: { spread: 0.018, commission: 0, slippage: 0.005 },
  GBPJPY: { spread: 0.028, commission: 0, slippage: 0.008 },
  EURJPY: { spread: 0.022, commission: 0, slippage: 0.006 },
  XAUUSD: { spread: 0.35, commission: 0, slippage: 0.10 },
  XAGUSD: { spread: 0.035, commission: 0, slippage: 0.008 },
  BTCUSD: { spread: 35, commission: 0, slippage: 10 },
  ETHUSD: { spread: 2.5, commission: 0, slippage: 0.8 },
  SOLUSD: { spread: 0.08, commission: 0, slippage: 0.03 },
  BNBUSD: { spread: 0.4, commission: 0, slippage: 0.15 },
  XRPUSD: { spread: 0.0008, commission: 0, slippage: 0.0003 },
  DOGEUSD: { spread: 0.0004, commission: 0, slippage: 0.0002 },
  ADAUSD: { spread: 0.001, commission: 0, slippage: 0.0004 },
  // Stock indices (MT5 CFDs). Typical retail numbers — index spreads widen
  // fast in news/overnight; verify against YOUR broker and override via
  // COST_OVERRIDES_JSON. Non-USD-denominated indices (GER40/EUR, UK100X/GBP,
  // JPN225/JPY) carry FX-conversion slop the price model does not capture.
  US30: { spread: 2.0, commission: 0, slippage: 0.5 },
  NAS100: { spread: 1.8, commission: 0, slippage: 0.5 },
  SPX500: { spread: 0.6, commission: 0, slippage: 0.15 },
  GER40: { spread: 1.5, commission: 0, slippage: 0.4 },
  UK100X: { spread: 1.8, commission: 0, slippage: 0.5 },
  JPN225: { spread: 15, commission: 0, slippage: 4 },
  AUS200: { spread: 2.0, commission: 0, slippage: 0.5 },
  FRA40: { spread: 1.5, commission: 0, slippage: 0.4 },
  EU50: { spread: 2.0, commission: 0, slippage: 0.5 },
  US2000: { spread: 0.5, commission: 0, slippage: 0.15 },
  USDX: { spread: 0.02, commission: 0, slippage: 0.005 },
  // Energy CFDs. USOIL/UKOIL trade nearly 24h weekdays with a daily break;
  // NATGAS spreads can triple in inventory news — size down Wednesdays.
  USOIL: { spread: 0.04, commission: 0, slippage: 0.01 },
  UKOIL: { spread: 0.05, commission: 0, slippage: 0.012 },
  NATGAS: { spread: 0.004, commission: 0, slippage: 0.001 },
};

export const NET_RR_BLOCK = 1.0; // below this the setup is negative-expectancy by construction
export const NET_RR_WEAK = 1.3;   // below this trade only with confluence + confidence penalty

let _overrides = null;
function overrides() {
  if (_overrides !== null) return _overrides;
  _overrides = {};
  try {
    const raw = process.env.COST_OVERRIDES_JSON;
    if (raw) {
      const parsed = JSON.parse(raw);
      for (const [k, v] of Object.entries(parsed)) {
        if (v && typeof v === "object") _overrides[String(k).toUpperCase()] = v;
      }
    }
  } catch (e) {
    console.warn("[costs] COST_OVERRIDES_JSON parse failed:", e.message);
  }
  return _overrides;
}

/**
 * Re-read COST_OVERRIDES_JSON (and any per-symbol markup) on the next lookup.
 * The overrides are cached because this module is called per signal; this is
 * the supported way to apply a config change (or a test's own cost table)
 * without restarting the process.
 */
export function resetCostCache() {
  _overrides = null;
}

/** { spread, commission, slippage, estimated } — estimated=true for unknown symbols. */
export function getSymbolCost(symbol) {
  const sym = String(symbol || "").toUpperCase();
  const ov = overrides()[sym] || {};
  const base = DEFAULT_SYMBOL_COSTS[sym];
  if (base) return { ...base, ...ov, estimated: false };
  return {
    spread: ov.spread ?? null, // resolved against price by costInPrice()
    commission: ov.commission ?? 0,
    slippage: ov.slippage ?? null,
    estimated: true,
  };
}

/** Total friction in PRICE units (spread + 2x slippage). Unknown symbols: ~2bps of price. */
export function costInPrice(symbol, refPrice) {
  const c = getSymbolCost(symbol);
  const px = Number(refPrice) || 0;
  const spread = c.spread ?? (px > 0 ? px * 0.0002 : 0);
  const slip = c.slippage ?? (px > 0 ? px * 0.0001 : 0);
  return Math.max(0, spread + 2 * slip);
}

/**
 * Net-RR for a setup. Lot-invariant.
 * @returns {{ costPrice, grossRisk, grossWin, netRisk, netWin, netRR, costPctOfRisk, blocked, weak }}
 */
export function applyCosts({ symbol, entry, sl, tp }) {
  const e = Number(entry), s = Number(sl), t = Number(tp);
  const costPrice = costInPrice(symbol, e);
  const grossRisk = Math.abs(e - s);
  const grossWin = Math.abs(t - e);
  if (!(grossRisk > 0) || !(grossWin > 0)) {
    return { costPrice, grossRisk, grossWin, netRisk: grossRisk, netWin: grossWin, netRR: 0, costPctOfRisk: null, blocked: true, weak: true };
  }
  const netRisk = grossRisk + costPrice;
  const netWin = grossWin - costPrice;
  const netRR = netWin / netRisk;
  // F-45: the published R-R was ROUNDED to 2dp while the blocked/weak gates
  // used the UNROUNDED value, so a raw 0.999 could be displayed as "1.0" and
  // simultaneously flagged `blocked: true`. Truncate (never round up) so the
  // number the user sees and the gate always agree, and the gate keeps its
  // conservative edge at the boundary.
  const netRRShown = Math.floor(netRR * 100) / 100;
  return {
    costPrice,
    grossRisk,
    grossWin,
    netRisk,
    netWin,
    netRR: netRRShown,
    costPctOfRisk: Number(((costPrice / grossRisk) * 100).toFixed(1)),
    blocked: netRRShown < NET_RR_BLOCK,
    weak: netRRShown < NET_RR_WEAK,
  };
}

// ============================================================================
// SYNTHETIC INDICES — a DIFFERENT cost model, because the cost is in a
// different place
// ----------------------------------------------------------------------------
// Deriv synthetics are not FX, and the contract type decides where the cost
// lives (verified against Deriv's own documentation):
//
//   * MULTIPLIER (what the scalping bot places): no bid/ask spread is crossed.
//     The cost is a MARKUP on the payout — Deriv's API markup documentation
//     states an app may add up to 3% of the potential payout
//     (developers.deriv.com/docs/intro/markup). A payout markup reduces the
//     WIN, not the risk, so it is modelled as a percentage of the TARGET
//     distance. It never widens the stop.
//
//   * RISE/FALL (what the pocket bot places): there is no separately charged
//     spread either. The cost sits inside the quoted payout, and the pocket
//     engine already gates on its own break-even — 1/(1+payout) — before any
//     contract is placed. See strategy/pocket/engine.js.
//
//   * CFD: Deriv publishes a MINIMUM spread per instrument on its synthetic-
//     indices specification table. Those are minimums from a live table, in
//     that instrument's own index points, and they are NOT the cost of a
//     multiplier or an options contract. A handful of verified published
//     values are recorded below as documentation only; nothing in this file
//     silently substitutes a CFD spread for a multiplier markup.
//
// Anything we cannot source stays flagged `estimated: true`, and every figure
// is overridable per deployment with COST_OVERRIDES_JSON (see the header).
// ============================================================================

/**
 * Payout markup for a multiplier contract on a synthetic. Deriv documents a
 * ceiling of 3%; the default models the common 2% case. A deployment that
 * knows its own markup overrides it: {"R_100":{"markup":0.01}}.
 */
export const SYNTHETIC_MARKUP_PCT = 0.02;
export const SYNTHETIC_MARKUP_CAP = 0.03;

/**
 * Published MINIMUM spreads (index points) from Deriv's synthetic-indices
 * specification table, recorded as documentation. These describe CFD trading
 * only — they are not applied by applyCostsFor().
 * Source: https://deriv.com/markets/derived-indices/synthetic-indices
 */
export const SYNTHETIC_PUBLISHED_MIN_SPREAD = {
  BOOM50: 1.2719,
  BOOM100: 7.375,
  BOOM150: 0.1705,
  BOOM200: 7.615,
  BOOM300: 0.5,
  BOOM500: 0.535,
  BOOM600: 0.774,
  BOOM900: 0.968,
  BOOM1000: 1.7245,
};

/** The payout markup (as a fraction) for a symbol; overridable per symbol. */
export function markupFor(symbol) {
  const sym = String(symbol || "").toUpperCase();
  const ov = overrides()[sym] || {};
  const raw = Number(ov.markup);
  if (Number.isFinite(raw) && raw >= 0) return Math.min(raw, SYNTHETIC_MARKUP_CAP);
  return SYNTHETIC_MARKUP_PCT;
}

/**
 * Net-RR for a MULTIPLIER setup on a synthetic.
 *
 * No spread is crossed, so the risk is unchanged; the markup comes off the
 * win. Shape matches applyCosts() so callers can treat the two models alike.
 */
export function applyPayoutMarkup({ symbol, entry, sl, tp }) {
  const e = Number(entry), s = Number(sl), t = Number(tp);
  const markupPct = markupFor(symbol);
  const grossRisk = Math.abs(e - s);
  const grossWin = Math.abs(t - e);
  if (!(grossRisk > 0) || !(grossWin > 0)) {
    return {
      model: "payout_markup", markupPct, costPrice: 0, grossRisk, grossWin,
      netRisk: grossRisk, netWin: grossWin, netRR: 0, costPctOfRisk: null,
      blocked: true, weak: true, estimated: false,
    };
  }
  const costPrice = grossWin * markupPct;
  const netWin = grossWin - costPrice;
  const netRisk = grossRisk;
  const netRR = netWin / netRisk;
  const netRRShown = Math.floor(netRR * 100) / 100;
  return {
    model: "payout_markup",
    markupPct,
    costPrice,
    grossRisk,
    grossWin,
    netRisk,
    netWin,
    netRR: netRRShown,
    costPctOfRisk: Number(((costPrice / grossRisk) * 100).toFixed(1)),
    blocked: netRRShown < NET_RR_BLOCK,
    weak: netRRShown < NET_RR_WEAK,
    // The markup is a deployment assumption (0-3%), not a published per-symbol
    // number — say so, so the UI and the journal never present it as exact.
    estimated: true,
  };
}

/**
 * The cost gate for a setup, whichever market it is on.
 *
 * Chooses the model from the SYMBOL, not the caller: a synthetic is priced by
 * payout markup, everything else by spread + commission + slippage. Callers
 * that already know which model applies (or that want to force one) pass it.
 */
export function applyCostsFor({ symbol, entry, sl, tp, model = null }) {
  const chosen = model || (isSynthetic(symbol) ? "payout_markup" : "spread");
  if (chosen === "payout_markup") return applyPayoutMarkup({ symbol, entry, sl, tp });
  return { model: "spread", ...applyCosts({ symbol, entry, sl, tp }) };
}

/**
 * USD friction for display/journaling. Needs the caller's pip table
 * (index.js PIP_CONFIG): { pipSize, pipValue } per 1.0 lot.
 */
export function costToUsd({ symbol, lotSize = 0.01, pipSize, pipValue, refPrice = 0 }) {
  const c = getSymbolCost(symbol);
  const commission = (c.commission || 0) * lotSize;
  if (!pipSize || !pipValue) return { spreadSlipUsd: null, commissionUsd: commission, totalUsd: null, estimated: true };
  const priceCost = costInPrice(symbol, refPrice);
  const spreadSlipUsd = (priceCost / pipSize) * pipValue * lotSize;
  return {
    spreadSlipUsd: Number(spreadSlipUsd.toFixed(2)),
    commissionUsd: Number(commission.toFixed(2)),
    totalUsd: Number((spreadSlipUsd + commission).toFixed(2)),
    estimated: c.estimated,
  };
}
