// ============================================================================
// FRIT OPTIONS SCANNER — Strategy A: defined-risk premium selling (paper only)
// ----------------------------------------------------------------------------
// Sell 21–60 DTE vertical credit spreads (or iron condors) on SPY/QQQ when:
//   - IV rank is elevated (sellers are paid for the variance risk premium),
//   - realized vol is NOT exploding (no catching a vol spike),
//   - the short strike sits at ~7–12 delta with a capped long wing,
//   - fills are modeled at the TOUCH (short @ bid, long @ ask), never mid,
//   - net credit clears fees with margin (structures.creditGate).
//
// Exits (signal metadata, enforced by backtest + future monitor):
//   take-profit at 50% of credit, hard close at 21 DTE, stop at 2x credit.
// No naked legs, no 0DTE, no earnings-week entries (caller passes event flag).
// Output is a SIGNAL. Nothing here places orders.
// ============================================================================

import { greeks } from "./pricing.js";
import { creditSpread, ironCondor, creditGate } from "./structures.js";

export const SCAN_DEFAULTS = {
  minDte: 21, maxDte: 60, targetDte: 40,
  minIvRank: 40,
  maxRvRatio: 1.5,          // rv20/rv60 above this = vol exploding, stand down
  maxDriftZ: 1.0,           // |20d drift| above this blocks the threatened wing:
                            // grind-up kills call wings, grind-down kills put wings
  shortDeltaLo: 0.07, shortDeltaHi: 0.12,
  widths: [5, 10],          // point widths to try (SPY/QQQ scale)
  minCreditFrac: 0.05,      // credit must be >= 5% of width (13Δ-wing condor
                            // economics: ~70%+ win rate at low RoC, not a bug)
  maxShortSpreadPct: 0.20,  // short-leg bid/ask width vs mid
  maxLongSpreadPct: 0.40,
  minOi: 50,                // skipped when the feed omits OI
  perContractFee: 0.50,
  contracts: 1,
};

export function ensureOptionsTables(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS options_iv (
      symbol TEXT NOT NULL, day TEXT NOT NULL, atm_iv REAL, rv20 REAL,
      PRIMARY KEY (symbol, day)
    );
    CREATE TABLE IF NOT EXISTS options_signals (
      id TEXT PRIMARY KEY, user_id TEXT NOT NULL, symbol TEXT NOT NULL,
      kind TEXT, decision TEXT, legs_json TEXT, credit REAL, max_loss REAL,
      rr REAL, breakevens_json TEXT, iv_rank REAL, atm_iv REAL, rv20 REAL,
      dte INTEGER, expiration TEXT, exits_json TEXT, reasoning TEXT,
      source TEXT, provisional INTEGER, synthetic INTEGER,
      timestamp INTEGER, is_read INTEGER DEFAULT 0
    );
  `);
}

export function recordIvSnapshot(db, symbol, atmIv, rv20) {
  try {
    const day = new Date().toISOString().slice(0, 10);
    db.prepare(
      "INSERT OR REPLACE INTO options_iv (symbol, day, atm_iv, rv20) VALUES (?,?,?,?)"
    ).run(symbol, day, atmIv, rv20);
  } catch { /* history is best-effort */ }
}

export function ivRankFromDb(db, symbol, current) {
  try {
    const rows = db.prepare(
      "SELECT atm_iv FROM options_iv WHERE symbol = ? ORDER BY day DESC LIMIT 252"
    ).all(symbol).map((r) => r.atm_iv).filter((v) => v > 0);
    if (rows.length < 20 || !(current > 0)) return { rank: null, n: rows.length };
    const lo = Math.min(...rows), hi = Math.max(...rows);
    if (hi <= lo) return { rank: 50, n: rows.length };
    return { rank: Math.max(0, Math.min(100, ((current - lo) / (hi - lo)) * 100)), n: rows.length };
  } catch { return { rank: null, n: 0 }; }
}

function realizedVol20(closes) {
  if (closes.length < 22) return null;
  const rets = [];
  const w = closes.slice(-21);
  for (let i = 1; i < w.length; i++) rets.push(Math.log(w[i] / w[i - 1]));
  const m = rets.reduce((a, b) => a + b, 0) / rets.length;
  const v = rets.reduce((a, b) => a + (b - m) ** 2, 0) / (rets.length - 1);
  return Math.sqrt(v * 252);
}
function realizedVol60(closes) {
  if (closes.length < 61) return null;
  const rets = [];
  const w = closes.slice(-61);
  for (let i = 1; i < w.length; i++) rets.push(Math.log(w[i] / w[i - 1]));
  const m = rets.reduce((a, b) => a + b, 0) / rets.length;
  const v = rets.reduce((a, b) => a + (b - m) ** 2, 0) / (rets.length - 1);
  return Math.sqrt(v * 252);
}

/** Median IV of near-ATM contracts with real two-sided quotes. */
export function atmIvOf(contracts, S, expiration) {
  const near = contracts.filter((c) =>
    c.expiration === expiration &&
    Math.abs(c.strike - S) / S <= 0.02 &&
    c.bid > 0 && c.ask > c.bid && c.iv > 0 && c.iv < 5
  );
  if (!near.length) return null;
  const ivs = near.map((c) => c.iv).sort((a, b) => a - b);
  return ivs[Math.floor(ivs.length / 2)];
}

function pickExpiry(chain, cfg) {
  const cands = chain.expirations.filter((e) => {
    const d = Math.round((new Date(e + "T16:00:00Z").getTime() - Date.now()) / 86400000);
    return d >= cfg.minDte && d <= cfg.maxDte;
  });
  if (!cands.length) return null;
  cands.sort((a, b) => Math.abs(daysTo(b) - cfg.targetDte) - Math.abs(daysTo(a) - cfg.targetDte));
  return cands[0];
  function daysTo(e) {
    return Math.round((new Date(e + "T16:00:00Z").getTime() - Date.now()) / 86400000);
  }
}

function deltaOf(type, S, K, T, sigma, r = 0.04, q = 0.015) {
  const g = greeks({ S, K, T, r, q, sigma, type });
  return g ? g.delta : null;
}

function liquidEnough(c, maxSpreadPct, minOi) {
  if (!(c.ask > c.bid) || !(c.bid > 0)) return false;
  const mid = (c.bid + c.ask) / 2;
  if (mid <= 0.05) return false; // penny options: unfillable edge
  if ((c.ask - c.bid) / mid > maxSpreadPct) return false;
  if (c.oi != null && c.volume != null && c.oi < minOi && c.volume < minOi) return false;
  return true;
}

function buildWing({ S, contracts, expiration, dte, type, cfg, r, q, surfaceIv }) {
  // Candidate short strikes: |delta| inside the target band, nearest first.
  const T = dte / 365;
  const side = contracts.filter((c) => c.type === type && c.expiration === expiration);
  const isPut = type === "put";
  const ranked = [];
  for (const c of side) {
    if (isPut ? c.strike >= S : c.strike <= S) continue; // OTM only
    const sigma = c.iv > 0 && c.iv < 5 ? c.iv : surfaceIv;
    if (!(sigma > 0)) continue;
    const d = deltaOf(type, S, c.strike, T, sigma, r, q);
    if (d == null) continue;
    const ad = Math.abs(d);
    if (ad >= cfg.shortDeltaLo && ad <= cfg.shortDeltaHi + 0.06) ranked.push({ c, ad, sigma });
  }
  ranked.sort((a, b) => Math.abs(a.ad - 0.10) - Math.abs(b.ad - 0.10));
  for (const { c: short } of ranked) {
    if (!liquidEnough(short, cfg.maxShortSpreadPct, cfg.minOi)) continue;
    for (const width of cfg.widths) {
      const longStrike = isPut ? short.strike - width : short.strike + width;
      const long = side.find((c) => Math.abs(c.strike - longStrike) < 0.011);
      if (!long || !liquidEnough(long, cfg.maxLongSpreadPct, cfg.minOi)) continue;
      // Touch fills: we receive the short bid, we pay the long ask.
      const credit = short.bid - long.ask;
      if (!(credit >= cfg.minCreditFrac * width)) continue;
      const s = creditSpread({
        S,
        legs: [
          { type, strike: short.strike, side: "short", mid: credit + long.ask, bid: short.bid, ask: short.ask },
          { type, strike: long.strike, side: "long", mid: long.ask, bid: long.bid, ask: long.ask },
        ],
        contracts: cfg.contracts, dte,
      });
      if (!s) continue;
      s.shortDelta = ranked.find((x) => x.c === short)?.ad ?? null;
      return s;
    }
  }
  return null;
}

/**
 * Scan one symbol. Returns { signal } or { skip, why } — a missing signal
 * beats a fake one. `db` optional (IV-rank history); without it rank is null
 * and the signal is marked weak.
 */
export function scanSymbol({ symbol, chain, closes, db = null, cfg = {} }) {
  const C = { ...SCAN_DEFAULTS, ...cfg };
  const S = chain.underlying;
  if (!(S > 0)) return { skip: true, why: "no underlying price" };
  const rv20 = realizedVol20(closes);
  const rv60 = realizedVol60(closes);
  if (rv20 == null) return { skip: true, why: "need 22+ daily closes for RV" };
  if (rv60 != null && rv20 / rv60 > C.maxRvRatio) {
    return { skip: true, why: `realized vol exploding (20d ${(rv20 * 100).toFixed(1)}% vs 60d ${(rv60 * 100).toFixed(1)}%) — stand down` };
  }
  const expiration = pickExpiry(chain, C);
  if (!expiration) return { skip: true, why: `no expiration ${C.minDte}-${C.maxDte} DTE` };
  const dte = Math.round((new Date(expiration + "T16:00:00Z").getTime() - Date.now()) / 86400000);
  const atmIv = atmIvOf(chain.contracts, S, expiration);
  if (!(atmIv > 0)) return { skip: true, why: "no clean ATM quotes for IV anchor" };
  if (db) recordIvSnapshot(db, symbol, atmIv, rv20);
  const { rank: ivRank, n: ivN } = db
    ? ivRankFromDb(db, symbol, atmIv)
    : { rank: null, n: 0 };
  if (ivRank != null && ivRank < C.minIvRank) {
    return { skip: true, why: `IV rank ${ivRank.toFixed(0)} below ${C.minIvRank} — premium too cheap to sell` };
  }
  const r = 0.04, q = symbol === "QQQ" ? 0.006 : 0.013;
  // Trend gate: a 20-day drift beyond ±1σ runs over the wing it moves toward.
  // Side-aware — an uptrend kills only the call wing, so a bull-put spread on
  // the safe side can still fire. Flat tape blocks nothing.
  let blockCall = false, blockPut = false, driftZ = null;
  if (closes.length >= 22 && rv20 > 0) {
    // Measured on the tape's own prints (chain S can legitimately differ from
    // the last close across sources/sessions). Floored denominator: a steady
    // grind has ~zero variance, which would otherwise make z undefined — and
    // a grind is exactly what runs over wings.
    const last = closes[closes.length - 1];
    const drift = last > 0 && closes[closes.length - 1 - 20] > 0
      ? Math.log(last / closes[closes.length - 1 - 20]) : 0;
    const sig20 = rv20 * Math.sqrt(20 / 252);
    driftZ = drift / Math.max(sig20, 0.005);
    if (driftZ > C.maxDriftZ) blockCall = true;
    if (driftZ < -C.maxDriftZ) blockPut = true;
  }
  const putWing = blockPut ? null : buildWing({ S, contracts: chain.contracts, expiration, dte, type: "put", cfg: C, r, q, surfaceIv: atmIv });
  const callWing = blockCall ? null : buildWing({ S, contracts: chain.contracts, expiration, dte, type: "call", cfg: C, r, q, surfaceIv: atmIv });
  if (!putWing && !callWing) {
    const why = blockPut || blockCall
      ? `trend gate: 20d drift z=${driftZ != null ? driftZ.toFixed(2) : "?"} blocks the threatened wing and no safe wing cleared the gates`
      : "no wing cleared delta/liquidity/credit gates";
    return { skip: true, why };
  }

  const legs = [];
  let credit = 0, width = 0;
  for (const w of [putWing, callWing]) {
    if (!w) continue;
    legs.push(...w.legs);
    credit += w.credit; width = Math.max(width, w.width);
  }
  const structure = legs.length === 4
    ? ironCondor({
      S,
      putShort: putWing.legs[0].strike, putLong: putWing.legs[1].strike,
      callShort: callWing.legs[0].strike, callLong: callWing.legs[1].strike,
      putCredit: putWing.credit, callCredit: callWing.credit,
      contracts: C.contracts, dte,
    })
    : { ...(putWing || callWing), contracts: C.contracts, dte };
  if (!structure) return { skip: true, why: "structure math failed" };
  const gate = creditGate(structure, { perContract: C.perContractFee });
  if (!gate.ok) return { skip: true, why: gate.reason };
  const weakIv = ivRank == null;
  const weakCredit = gate.weak === true;

  const confidence = Math.max(20, Math.min(85,
    45 + (ivRank != null ? ivRank * 0.25 : 0) + (legs.length === 4 ? 8 : 0) - (weakCredit ? 10 : 0)
  ));
  const feeDollars = C.perContractFee * legs.length * C.contracts;
  const reasoning = [
    `Sell ${(credit).toFixed(2)} credit, risk ${(structure.maxLoss).toFixed(2)} (${(credit / structure.maxLoss * 100).toFixed(0)}% RoC)`,
    ivRank != null ? `IV rank ${ivRank.toFixed(0)} (n=${ivN}) — premium worth selling` : "IV rank unknown (no history yet) — size down",
    `RV20 ${(rv20 * 100).toFixed(1)}%${rv60 != null ? ` vs RV60 ${(rv60 * 100).toFixed(1)}% — no vol explosion` : ""}`,
    driftZ != null ? `20d drift z=${driftZ.toFixed(2)}${blockCall ? " — call wing blocked (uptrend)" : blockPut ? " — put wing blocked (downtrend)" : " — no trend block"}` : "drift unknown (short history)",
    `Fills at touch (short @ bid, long @ ask); fees $${feeDollars.toFixed(2)} modeled`,
    `Exit at 50% credit, hard close by 21 DTE, stop at 2x credit`,
  ];
  return {
    signal: {
      id: `opt_${symbol}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      symbol, kind: legs.length === 4 ? "IRON_CONDOR" : (putWing ? "BULL_PUT_SPREAD" : "BEAR_CALL_SPREAD"),
      decision: "SELL_PREMIUM",
      legs: legs.map((l) => ({ type: l.type, strike: l.strike, side: l.side })),
      contracts: C.contracts, feeDollars: Number(feeDollars.toFixed(2)),
      driftZ: driftZ != null ? Number(driftZ.toFixed(2)) : null,
      credit: Number(credit.toFixed(2)),
      maxLoss: Number(structure.maxLoss.toFixed(2)),
      maxLossDollars: Number((structure.maxLoss * 100 * C.contracts).toFixed(2)),
      rr: Number((credit / structure.maxLoss).toFixed(2)),
      breakevens: structure.breakevens.map((b) => Number(b.toFixed(2))),
      shortDeltas: [putWing?.shortDelta ?? null, callWing?.shortDelta ?? null],
      ivRank: ivRank != null ? Number(ivRank.toFixed(1)) : null,
      ivHistoryN: ivN,
      atmIv: Number((atmIv * 100).toFixed(1)),
      rv20: Number((rv20 * 100).toFixed(1)),
      dte, expiration,
      exits: {
        takeProfitCredit: Number((credit * 0.5).toFixed(2)),
        closeByDte: 21,
        stopLossDollars: Number((credit * 2 * 100 * C.contracts).toFixed(2)),
      },
      weak: weakIv || weakCredit,
      reasoning: reasoning,
      dataSource: { source: chain.source, delayed: chain.delayed, provisional: chain.provisional },
      disclaimer: chain.disclaimer,
      synthetic: false,
      timestamp: Date.now(),
    },
  };
}
