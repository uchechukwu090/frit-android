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
  return {
    costPrice,
    grossRisk,
    grossWin,
    netRisk,
    netWin,
    netRR: Number(netRR.toFixed(2)),
    costPctOfRisk: Number(((costPrice / grossRisk) * 100).toFixed(1)),
    blocked: netRR < NET_RR_BLOCK,
    weak: netRR < NET_RR_WEAK,
  };
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
