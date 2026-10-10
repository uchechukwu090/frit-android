// ============================================================================
// FRIT OPTIONS BACKTEST — event-driven, bid/ask fills, full costs
// ----------------------------------------------------------------------------
// Why a modeled chain instead of historical chains: no $0 source sells
// historical intraday/EOD chains (ORATS history is paid; CBOE bans scraping).
// So each day's chain is GENERATED from that day's realized vol:
//   ATM IV = RV20 x (1 + vrpMarkup), smile = atm x (1 + skew*m + curve*m^2),
//   bid/ask = mid +/- half-spread that widens OTM. Fills: sell @ bid,
//   buy @ ask — never mid. This tests STRATEGY LOGIC (entries, exits, costs),
// not data quality; it cannot validate the VRP edge itself. Labeled as such.
//
// One position at a time (Strategy A iron condor, 5-wide, ~13 delta wings,
// ~35 DTE modeled expiry). Exits: 50% credit, 2x-credit stop, 21 DTE.
// Walk-forward calibrates vrpMarkup on the train half, tests on the rest.
// ============================================================================

import { bsPrice } from "./pricing.js";

export const BT_DEFAULTS = {
  width: 5, targetDelta: 0.10, entryDte: 35, exitDte: 21,
  minIvRank: 40, maxRvRatio: 1.5, minCreditFrac: 0.05,
  vrpMarkup: 0.25, skew: -0.12, curve: 0.15,
  baseSpreadPct: 0.02, otmSpreadK: 0.15, minHalfSpread: 0.02,
  perContractFee: 0.50, contracts: 1,
  tpFrac: 0.5, stopMult: 2.0,
  trials: 25, // strategy-space trials assumed for the deflated Sharpe
};

function rv(closes, end, len) {
  if (end - len + 1 < 0) return null;
  const w = closes.slice(end - len + 1, end + 1);
  const rets = [];
  for (let i = 1; i < w.length; i++) {
    if (!(w[i] > 0) || !(w[i - 1] > 0)) return null;
    rets.push(Math.log(w[i] / w[i - 1]));
  }
  const m = rets.reduce((a, b) => a + b, 0) / rets.length;
  const v = rets.reduce((a, b) => a + (b - m) ** 2, 0) / (rets.length - 1 || 1);
  return Math.sqrt(v * 252);
}

function smileIv(atm, S, K) {
  const m = Math.log(K / S);
  return atm * (1 + BT_DEFAULTS.skew * m + BT_DEFAULTS.curve * m * m);
}

// Modeled chain for one day: 1-point strike steps over ±8% (a 1%-step grid
// can never host an exact 5-wide wing — the long leg would match nothing).
// Exported for tests/inspection.
export function modelDay(S, atmIv, cfg) {
  const T = cfg.entryDte / 365;
  const out = [];
  const half = Math.round(S * 0.12); // 13Δ wings at 30%+ IV sit ~10% OTM
  for (let d = -half; d <= half; d++) {
    const K = S + d;
    for (const type of ["put", "call"]) {
      const iv = smileIv(atmIv, S, K);
      const mid = bsPrice({ S, K, T, r: 0.04, q: 0.013, sigma: iv, type });
      if (!(mid > 0.05)) continue;
      const m = Math.abs(Math.log(K / S));
      const hs = Math.max(cfg.minHalfSpread, mid * (cfg.baseSpreadPct + cfg.otmSpreadK * m));
      out.push({ type, strike: K, iv, mid, bid: +(mid - hs).toFixed(2), ask: +(mid + hs).toFixed(2) });
    }
  }
  return out;
}

export function pickWing(dayChain, S, type, cfg) {
  const T = cfg.entryDte / 365;
  const isPut = type === "put";
  const cands = [];
  for (const c of dayChain) {
    if (c.type !== type) continue;
    if (isPut ? c.strike >= S : c.strike <= S) continue;
    // Delta from the contract's own modeled IV (same math as the scanner).
    const vsqrt = c.iv * Math.sqrt(T);
    const d1 = (Math.log(S / c.strike) + (0.04 - 0.013 + 0.5 * c.iv * c.iv) * T) / vsqrt;
    const delta = isPut ? -(1 - ncdf(d1)) : ncdf(d1);
    const ad = Math.abs(delta);
    if (ad >= 0.07 && ad <= 0.15) cands.push({ c, ad });
  }
  cands.sort((a, b) => Math.abs(a.ad - cfg.targetDelta) - Math.abs(b.ad - cfg.targetDelta));
  for (const { c: short } of cands) {
    // Strike match tolerance: modeled grids inherit the spot's float dust
    // (S=596.0299 → K=565.0299 vs a toFixed(2) lookup of 565.03), so exact
    // equality would silently match nothing. A cent is far below any real
    // strike grid step ($1+ on SPY/QQQ).
    const longStrike = isPut ? short.strike - cfg.width : short.strike + cfg.width;
    const long = dayChain.find((x) => x.type === type && Math.abs(x.strike - longStrike) < 0.011);
    if (!long) continue;
    const credit = short.bid - long.ask; // touch fills
    if (!(credit >= cfg.minCreditFrac * cfg.width)) continue;
    return { short, long, credit };
  }
  return null;
}

function ncdf(x) {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1, ax = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return 0.5 * (1 + sign * y);
}
/** Mark-to-model value of open wings: shorts marked at ASK (cost to close), longs at BID. */
export function markWings(wings, S, atmIv, cfg, dteLeft) {
  // Remaining life, NOT entry DTE: without this the mark never decays and the
  // strategy that sells time decay can never book a winner.
  const T = Math.max(dteLeft, 1) / 365;
  let val = 0;
  for (const w of wings) {
    for (const [leg, side] of [[w.short, "short"], [w.long, "long"]]) {
      const iv = smileIv(atmIv, S, leg.strike);
      const mid = bsPrice({ S, K: leg.strike, T, r: 0.04, q: 0.013, sigma: iv, type: leg.type });
      if (!(mid > 0)) return null;
      const m = Math.abs(Math.log(leg.strike / S));
      const hs = Math.max(cfg.minHalfSpread, mid * (cfg.baseSpreadPct + cfg.otmSpreadK * m));
      val += side === "short" ? -(mid + hs) : (mid - hs);
    }
  }
  return val; // net value of the position (negative = we owe to close)
}

/** Round-trip fees for one condor (4 legs): open + close. */
function fees(cfg) {
  return cfg.perContractFee * 4 * cfg.contracts;
}

/**
 * @param closes ascending daily closes (need 90+ for RV60 + IV history)
 * @returns { trades, stats }
 */
export function runBacktest(closes, cfg = {}) {
  const C = { ...BT_DEFAULTS, ...cfg };
  const trades = [];
  const atmHist = [];
  let pos = null;
  const start = 70;
  for (let i = start; i < closes.length; i++) {
    const S = closes[i];
    if (!(S > 0)) { pos = null; continue; }
    const r20 = rv(closes, i, 20), r60 = rv(closes, i, 60);
    if (r20 == null) continue;
    const atm = r20 * (1 + C.vrpMarkup);
    atmHist.push(atm);
    // IV rank vs trailing 1y of modeled ATM IV
    const hist = atmHist.slice(-252);
    const lo = Math.min(...hist), hi = Math.max(...hist);
    const rank = hi > lo ? ((atm - lo) / (hi - lo)) * 100 : 50;

    if (!pos) {
      if (r60 != null && r20 / r60 > C.maxRvRatio) continue;
      if (rank < C.minIvRank) continue;
      const day = modelDay(S, atm, C);
      const pw = pickWing(day, S, "put", C);
      const cw = pickWing(day, S, "call", C);
      if (!pw || !cw) continue;
      const credit = pw.credit + cw.credit;
      pos = {
        entryIdx: i, entryS: S, dte: C.entryDte, credit,
        wings: [
          { short: pw.short, long: pw.long, type: "put" },
          { short: cw.short, long: cw.long, type: "call" },
        ],
        maxLoss: C.width - credit,
        // Sticky marking IV: listed IV does not reprice 1:1 with every
        // 20-day realized wiggle (that is the VRP term structure). A slow EWMA
        // keeps transient RV-measurement noise from whipsawing the mark, while
        // a sustained expansion still drags it up and stops the trade out.
        markIv: atm,
      };
      continue;
    }
    // Manage open position
    pos.dte -= 1;
    pos.markIv = 0.9 * pos.markIv + 0.1 * atm;
    const mv = markWings(pos.wings, S, pos.markIv, C, pos.dte);
    if (mv == null) { pos = null; continue; } // model broke — flat, uncounted
    const pnl = pos.credit + mv; // credit received minus cost to close
    const close = (why) => {
      const net = pnl * 100 * C.contracts - fees(C);
      trades.push({
        entryIdx: pos.entryIdx, exitIdx: i, why,
        entryS: +pos.entryS.toFixed(2), exitS: +S.toFixed(2),
        credit: +pos.credit.toFixed(2), pnl: +net.toFixed(2),
        dteLeft: pos.dte,
      });
      pos = null;
    };
    if (pnl >= pos.credit * C.tpFrac) close("take_profit");
    else if (pnl <= -pos.credit * C.stopMult) close("stop");
    else if (pos.dte <= C.exitDte) close("dte_exit");
    else if (S <= pos.wings[0].short.strike || S >= pos.wings[1].short.strike) {
      // Short strike touched: stop only if the loss already exceeds 1x credit
      if (pnl <= -pos.credit) close("strike_touch_stop");
    }
  }
  return { trades, stats: summarize(trades) };
}

export function summarize(trades) {
  const n = trades.length;
  const pnls = trades.map((t) => t.pnl);
  const total = pnls.reduce((a, b) => a + b, 0);
  const wins = pnls.filter((p) => p > 0).length;
  const mean = n ? total / n : 0;
  const sd = n > 1 ? Math.sqrt(pnls.reduce((a, p) => a + (p - mean) ** 2, 0) / (n - 1)) : 0;
  const sharpe = sd > 0 ? (mean / sd) * Math.sqrt(252 / 30) : 0; // per-trade (~30d) -> annual
  let peak = 0, maxDd = 0, cum = 0;
  for (const p of pnls) { cum += p; peak = Math.max(peak, cum); maxDd = Math.max(maxDd, peak - cum); }
  // Simplified deflated Sharpe (Bailey–Lopez de Prado, first-order trials
  // correction): subtract the expected best Sharpe under the null for T
  // independent strategy trials, E[max SR0] ≈ sqrt(2 ln T / n).
  const T = BT_DEFAULTS.trials;
  const eNull = Math.sqrt(2 * Math.log(T) / Math.max(n, 1));
  const deflated = sd > 0 ? Math.max(0, (mean / sd) * Math.sqrt(252 / 30) - eNull) : 0;
  return {
    n, wins, winRate: n ? wins / n : 0,
    total: +total.toFixed(2), avg: n ? +mean.toFixed(2) : 0,
    sharpe: +sharpe.toFixed(2), deflatedSharpe: +deflated.toFixed(2),
    maxDd: +maxDd.toFixed(2),
    modeledChain: true,
  };
}

/** Walk-forward: grid-search vrpMarkup on train, evaluate on test. */
export function walkForward(closes, { trainFrac = 0.5, grid = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4] } = {}) {
  const cut = Math.floor(closes.length * trainFrac);
  let best = { markup: grid[0], total: -Infinity };
  for (const m of grid) {
    const r = runBacktest(closes.slice(0, cut), { vrpMarkup: m });
    if (r.stats.total > best.total) best = { markup: m, total: r.stats.total };
  }
  const test = runBacktest(closes.slice(cut - 70), { vrpMarkup: best.markup });
  return { bestMarkup: best.markup, trainTotal: +best.total.toFixed(2), test };
}
