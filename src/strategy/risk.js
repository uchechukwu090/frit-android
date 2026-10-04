// ============================================================================
// FRIT PORTFOLIO RISK LAYER — account-level guards every order passes through
// ----------------------------------------------------------------------------
// Signals are per-symbol; blowups are portfolio-wide. This layer sits between
// the engine and the MT5 bridge (both /trade and the scheduler executor) and
// enforces four rules:
//
//  1. MAX OPEN POSITIONS — no more than N live MONITORING positions.
//  2. PER-TRADE RISK CAP — a single trade may not risk more than X% of
//     balance (second opinion on top of the lot-size engine).
//  3. CORRELATION GUARD — correlated symbols must not stack same-way risk:
//     per-group position caps + a NET USD-exposure cap. Buying EURUSD and
//     GBPUSD together is ONE dollar-short bet wearing two coats.
//  4. DAILY LOSS KILL-SWITCH — if today's realized losses breach Y% of
//     balance, ALL new trades are blocked until tomorrow. No revenge trading.
//
// All limits are env-overridable: RISK_MAX_OPEN (4), RISK_MAX_TRADE_PCT (2),
// RISK_MAX_DAILY_LOSS_PCT (4), RISK_MAX_USD_NET_PCT (3), RISK_MAX_PER_GROUP (2).
// P&L math uses the caller's pip table (index.js PIP_CONFIG); USD-quoted
// assumption is documented approximation — this is a guardrail, not accounting.
// ============================================================================

const num = (v, dflt) => {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : dflt;
};

export function riskLimits() {
  return {
    maxOpen: Math.floor(num(process.env.RISK_MAX_OPEN, 4)),
    maxTradePct: num(process.env.RISK_MAX_TRADE_PCT, 2),
    maxDailyLossPct: num(process.env.RISK_MAX_DAILY_LOSS_PCT, 4),
    maxUsdNetPct: num(process.env.RISK_MAX_USD_NET_PCT, 3),
    maxPerGroup: Math.floor(num(process.env.RISK_MAX_PER_GROUP, 2)),
  };
}

// Correlation groups: members routinely move together.
const CORR_GROUPS = {
  USD_MAJORS: ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],
  USD_QUOTE: ["USDCHF", "USDCAD", "USDJPY"],
  JPY_CROSS: ["USDJPY", "GBPJPY", "EURJPY"],
  METALS: ["XAUUSD", "XAGUSD"],
  CRYPTO: ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD", "DOGEUSD", "ADAUSD"],
};

/**
 * Signed USD exposure of a position: +1 = long USD, -1 = short USD, 0 = none.
 * XXXUSD buys are short-USD; USDXXX buys are long-USD; sells flip the sign.
 */
export function usdExposureSign(symbol, action) {
  const sym = String(symbol || "").toUpperCase();
  const dir = String(action || "").toLowerCase() === "sell" ? -1 : 1;
  if (/^[A-Z]{3}USD$/.test(sym) || sym === "XAUUSD" || sym === "XAGUSD") return -dir;
  if (/^USD[A-Z]{3}$/.test(sym)) return dir;
  return 0;
}

export function groupOf(symbol) {
  const sym = String(symbol || "").toUpperCase();
  for (const [g, members] of Object.entries(CORR_GROUPS)) {
    if (members.includes(sym)) return g;
  }
  return "OTHER";
}

export class PortfolioRisk {
  constructor({ db, pipConfig = {} } = {}) {
    if (!db) throw new Error("PortfolioRisk requires db");
    this.db = db;
    this.pipConfig = pipConfig;
  }

  _openPositions() {
    try {
      return this.db.prepare("SELECT * FROM positions WHERE status='MONITORING'").all();
    } catch {
      return [];
    }
  }

  _realizedToday() {
    try {
      return this.db.prepare(
        "SELECT * FROM positions WHERE status='RESOLVED' AND date(resolved_at) = date('now')"
      ).all();
    } catch {
      return [];
    }
  }

  /** Signed USD P&L of a closed position (USD-quoted approximation). */
  pnlUsd(pos) {
    try {
      const cfg = this.pipConfig[String(pos.symbol || "").toUpperCase()];
      if (!cfg || pos.entry == null || pos.close_price == null) return null;
      const dir = String(pos.action).toLowerCase() === "sell" ? -1 : 1;
      const dist = (Number(pos.close_price) - Number(pos.entry)) * dir;
      const lot = Number(pos.lot_size) || 0.01;
      if (cfg.isCrypto) return dist * 1 * lot; // crypto: $1 per price-unit per lot
      return (dist / cfg.pipSize) * cfg.pipValue * lot;
    } catch {
      return null;
    }
  }

  /** Risk USD of a prospective trade from its stop distance. */
  riskUsd({ symbol, lotSize, entry, sl }) {
    const cfg = this.pipConfig[String(symbol || "").toUpperCase()];
    const dist = Math.abs(Number(entry) - Number(sl));
    const lot = Number(lotSize) || 0.01;
    if (!cfg || !(dist > 0)) return null;
    if (cfg.isCrypto) return dist * lot;
    return (dist / cfg.pipSize) * cfg.pipValue * lot;
  }

  /**
   * Pre-trade check. Returns { allowed, reasons[], exposure }.
   * Never throws — a risk layer that crashes open is worse than none.
   */
  check({ symbol, action, lotSize, entry, sl, balance = 1000 } = {}) {
    const limits = riskLimits();
    const reasons = [];
    const sym = String(symbol || "").toUpperCase();
    const bal = Number(balance) > 0 ? Number(balance) : 1000;
    const exposure = { openPositions: 0, newRiskUsd: null, newRiskPct: null, usdNetPct: null, groupCounts: {} };

    try {
      const open = this._openPositions();
      exposure.openPositions = open.length;

      // 1. Max open positions
      if (open.length >= limits.maxOpen) {
        reasons.push(`Too many live positions (${open.length}/${limits.maxOpen}) — close or wait before adding ${sym}`);
      }

      // 2. Per-trade risk cap
      const rUsd = this.riskUsd({ symbol: sym, lotSize, entry, sl });
      if (rUsd != null) {
        exposure.newRiskUsd = Number(rUsd.toFixed(2));
        exposure.newRiskPct = Number(((rUsd / bal) * 100).toFixed(2));
        if (exposure.newRiskPct > limits.maxTradePct) {
          reasons.push(`Single-trade risk ${exposure.newRiskPct}% exceeds cap ${limits.maxTradePct}% — reduce size`);
        }
      }

      // 3a. Per-group cap (correlation guard, part 1)
      const groups = {};
      for (const p of open) {
        const g = groupOf(p.symbol);
        groups[g] = (groups[g] || 0) + 1;
      }
      const myGroup = groupOf(sym);
      groups[myGroup] = (groups[myGroup] || 0) + 1;
      exposure.groupCounts = groups;
      if ((groups[myGroup] || 0) > limits.maxPerGroup && myGroup !== "OTHER") {
        reasons.push(`Correlation guard: ${myGroup} would hold ${groups[myGroup]} positions (cap ${limits.maxPerGroup}) — these move together`);
      }

      // 3b. Net USD exposure cap (correlation guard, part 2)
      let usdNet = 0;
      for (const p of open) {
        const pr = this.riskUsd({ symbol: p.symbol, lotSize: p.lot_size, entry: p.entry, sl: p.sl });
        if (pr != null) usdNet += usdExposureSign(p.symbol, p.action) * pr;
      }
      if (rUsd != null) usdNet += usdExposureSign(sym, action) * rUsd;
      exposure.usdNetPct = Number(((Math.abs(usdNet) / bal) * 100).toFixed(2));
      if (exposure.usdNetPct > limits.maxUsdNetPct) {
        reasons.push(`Net USD exposure ${exposure.usdNetPct}% exceeds cap ${limits.maxUsdNetPct}% — this book is one dollar bet`);
      }

      // 4. Daily loss kill-switch
      const today = this._realizedToday();
      let realized = 0;
      let counted = 0;
      for (const p of today) {
        if (p.outcome !== "win" && p.outcome !== "loss") continue;
        const v = this.pnlUsd(p);
        if (v != null) { realized += v; counted++; }
      }
      exposure.realizedTodayUsd = Number(realized.toFixed(2));
      exposure.realizedTodayPct = Number(((realized / bal) * 100).toFixed(2));
      if (realized < 0 && Math.abs(realized) / bal * 100 >= limits.maxDailyLossPct) {
        reasons.push(`KILL-SWITCH: down ${exposure.realizedTodayPct}% today (limit ${limits.maxDailyLossPct}%) — no new trades until tomorrow. No revenge trading.`);
      }
    } catch (e) {
      console.warn("[PortfolioRisk] check failed open:", e.message);
    }

    return { allowed: reasons.length === 0, reasons, exposure, limits };
  }

  /** Read-only status for GET /risk/status. */
  status(balance = 1000) {
    const open = this._openPositions();
    const today = this._realizedToday();
    let realized = 0;
    for (const p of today) {
      if (p.outcome !== "win" && p.outcome !== "loss") continue;
      const v = this.pnlUsd(p);
      if (v != null) realized += v;
    }
    return {
      limits: riskLimits(),
      open: open.map(p => ({ id: p.id, symbol: p.symbol, action: p.action, lot_size: p.lot_size, entry: p.entry, sl: p.sl, tp: p.tp, source: p.source })),
      openCount: open.length,
      realizedTodayUsd: Number(realized.toFixed(2)),
      killSwitchTripped: realized < 0 && Math.abs(realized) / Number(balance || 1000) * 100 >= riskLimits().maxDailyLossPct,
    };
  }
}
