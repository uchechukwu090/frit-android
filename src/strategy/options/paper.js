// ============================================================================
// FRIT OPTIONS PAPER LEDGER — the verifiable track record behind the signals
// ----------------------------------------------------------------------------
// Every equity signal that passes the risk gate opens a PAPER position here
// (no broker, no money). Open positions are marked with REAL chain quotes —
// shorts at the ASK (cost to close), longs at the BID — never mids, never
// modeled vols. The same exit discipline as the strategy (50% TP, 2x stop,
// 21-DTE hard close) resolves them. What cannot be marked (missing leg,
// unreachable chain) is flagged stale, never faked.
//
// This ledger is the subscription story: published paper P&L with defined
// exits beats claimed backtests. Live trading unlocks only after this ledger
// earns it on real chains over time.
// ============================================================================

export function ensurePaperTables(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS options_paper (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      signal_id TEXT,
      symbol TEXT NOT NULL,
      kind TEXT,
      legs_json TEXT NOT NULL,
      contracts INTEGER DEFAULT 1,
      credit REAL NOT NULL,
      max_loss REAL,
      fee_dollars REAL DEFAULT 0,
      entry_spot REAL,
      expiration TEXT,
      tp_credit REAL,
      stop_dollars REAL,
      status TEXT DEFAULT 'OPEN',
      exit_why TEXT,
      exit_spot REAL,
      exit_time INTEGER,
      pnl REAL,
      unrealized REAL,
      stale INTEGER DEFAULT 0,
      opened_at INTEGER,
      marked_at INTEGER
    );
  `);
}

export function maxOpenPositions() {
  return Math.max(1, Math.floor(Number(process.env.OPT_MAX_OPEN) || 3));
}

/** Open a paper position for a gated signal. Correlated-book guard: cap OPEN. */
export function openPaperPosition(db, user_id, signal) {
  ensurePaperTables(db);
  try {
    const open = db.prepare(
      "SELECT COUNT(*) AS c FROM options_paper WHERE user_id = ? AND status = 'OPEN'"
    ).get(user_id)?.c || 0;
    if (open >= maxOpenPositions()) {
      return { skipped: true, why: `paper book full (${open} open, cap ${maxOpenPositions()}) — correlated premium bets capped` };
    }
    const id = `px_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    db.prepare(`
      INSERT INTO options_paper
      (id, user_id, signal_id, symbol, kind, legs_json, contracts, credit,
       max_loss, fee_dollars, entry_spot, expiration, tp_credit, stop_dollars,
       status, opened_at, marked_at)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'OPEN', ?, ?)
    `).run(
      id, user_id, signal.id || null, signal.symbol, signal.kind || null,
      JSON.stringify(signal.legs || []), signal.contracts || 1, signal.credit,
      signal.maxLoss ?? null, signal.feeDollars ?? 0,
      signal.entrySpot ?? null, signal.expiration || null,
      (signal.exits?.takeProfitCredit ?? signal.credit * 0.5),
      (signal.exits?.stopLossDollars ?? signal.credit * 2 * 100 * (signal.contracts || 1)),
      Date.now(), Date.now()
    );
    return { id };
  } catch (e) {
    return { skipped: true, why: `paper open failed: ${e.message}` };
  }
}

function dteOf(expiration, nowMs = Date.now()) {
  if (!expiration) return null;
  return Math.round((new Date(expiration + "T16:00:00Z").getTime() - nowMs) / 86400000);
}

function intrinsic(type, S, K) {
  if (!(S > 0) || !(K > 0)) return 0;
  return type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
}

/**
 * Mark every OPEN position with live chain quotes and resolve exits.
 * @param getChainFn async (symbol) => normalized chain (same shape as chains.js)
 */
export async function resolvePaperBook(db, user_id, getChainFn) {
  ensurePaperTables(db);
  const rows = db.prepare(
    "SELECT * FROM options_paper WHERE user_id = ? AND status = 'OPEN' ORDER BY opened_at ASC"
  ).all(user_id);
  const chainCache = new Map();
  const chainFor = async (symbol) => {
    if (!chainCache.has(symbol)) chainCache.set(symbol, await getChainFn(symbol));
    return chainCache.get(symbol);
  };
  const resolved = [];
  for (const p of rows) {
    try {
      const legs = JSON.parse(p.legs_json || "[]");
      const chain = await chainFor(p.symbol);
      const S = chain.underlying;
      if (!(S > 0)) {
        db.prepare("UPDATE options_paper SET stale = 1, marked_at = ? WHERE id = ?").run(Date.now(), p.id);
        resolved.push({ id: p.id, stale: true, why: "no underlying quote" });
        continue;
      }
      // Match legs to live contracts: shorts marked at ASK, longs at BID.
      let value = 0, missing = false;
      for (const leg of legs) {
        const c = chain.contracts.find((x) =>
          x.type === leg.type && x.strike === leg.strike && x.expiration === p.expiration);
        if (!c || !(c.ask >= 0) || !(c.bid >= 0)) { missing = true; break; }
        value += leg.side === "short" ? -c.ask : c.bid;
      }
      if (missing) {
        db.prepare("UPDATE options_paper SET stale = 1, marked_at = ? WHERE id = ?").run(Date.now(), p.id);
        resolved.push({ id: p.id, stale: true, why: "leg missing from live chain" });
        continue;
      }
      const contracts = p.contracts || 1;
      const gross = (p.credit + value) * 100 * contracts; // unrealized, fees pending
      const dte = dteOf(p.expiration);
      const shortStrikes = legs.filter((l) => l.side === "short").map((l) => l.strike);
      const touched = shortStrikes.some((k) =>
        legs.find((l) => l.side === "short" && l.strike === k)?.type === "put" ? S <= k : S >= k);
      let exitWhy = null;
      if (gross >= (p.tp_credit ?? p.credit * 0.5) * 100 * contracts) exitWhy = "take_profit";
      else if (gross <= -(p.stop_dollars ?? p.credit * 2 * 100 * contracts)) exitWhy = "stop";
      else if (touched && gross <= -(p.credit * 100 * contracts)) exitWhy = "strike_touch_stop";
      else if (dte != null && dte <= 0) exitWhy = "expired_settle";
      else if (dte != null && dte <= 21) exitWhy = "dte_exit";
      if (exitWhy) {
        let pnl;
        if (exitWhy === "expired_settle") {
          // Cash-settled approximation at intrinsic (rare: 21-DTE rule normally
          // closes first). Flagged, not modeled as a fill.
          let settle = 0;
          for (const leg of legs) {
            settle += (leg.side === "short" ? -1 : 1) * intrinsic(leg.type, S, leg.strike);
          }
          pnl = (p.credit + settle) * 100 * contracts - (p.fee_dollars || 0);
        } else {
          pnl = gross - (p.fee_dollars || 0);
        }
        db.prepare(`
          UPDATE options_paper
          SET status = 'CLOSED', exit_why = ?, exit_spot = ?, exit_time = ?,
              pnl = ?, unrealized = NULL, stale = 0, marked_at = ?
          WHERE id = ?
        `).run(exitWhy, S, Date.now(), +pnl.toFixed(2), Date.now(), p.id);
        resolved.push({ id: p.id, closed: true, why: exitWhy, pnl: +pnl.toFixed(2) });
      } else {
        db.prepare("UPDATE options_paper SET unrealized = ?, stale = 0, marked_at = ? WHERE id = ?")
          .run(+gross.toFixed(2), Date.now(), p.id);
        resolved.push({ id: p.id, unrealized: +gross.toFixed(2) });
      }
    } catch (e) {
      db.prepare("UPDATE options_paper SET stale = 1, marked_at = ? WHERE id = ?").run(Date.now(), p.id);
      resolved.push({ id: p.id, stale: true, why: e.message });
    }
  }
  return resolved;
}

/** Public ledger: open book + closed stats. Fees pending on unrealized. */
export function ledgerSummary(db, user_id) {
  ensurePaperTables(db);
  const open = db.prepare(
    "SELECT * FROM options_paper WHERE user_id = ? AND status = 'OPEN' ORDER BY opened_at DESC"
  ).all(user_id);
  const closed = db.prepare(
    "SELECT * FROM options_paper WHERE user_id = ? AND status = 'CLOSED' ORDER BY exit_time ASC"
  ).all(user_id);
  const pnls = closed.map((c) => Number(c.pnl) || 0);
  const total = pnls.reduce((a, b) => a + b, 0);
  const wins = pnls.filter((p) => p > 0).length;
  let peak = 0, maxDd = 0, cum = 0;
  for (const p of pnls) { cum += p; peak = Math.max(peak, cum); maxDd = Math.max(maxDd, peak - cum); }
  const byKind = {};
  for (const c of closed) {
    const k = c.kind || "UNKNOWN";
    byKind[k] = byKind[k] || { n: 0, wins: 0, total: 0 };
    byKind[k].n++;
    if (Number(c.pnl) > 0) byKind[k].wins++;
    byKind[k].total = +(byKind[k].total + Number(c.pnl)).toFixed(2);
  }
  const unrealized = open.reduce((a, p) => a + (Number(p.unrealized) || 0), 0);
  return {
    open: open.map((p) => ({
      id: p.id, symbol: p.symbol, kind: p.kind, credit: p.credit,
      maxLoss: p.max_loss, expiration: p.expiration,
      unrealized: p.unrealized, stale: p.stale === 1, openedAt: p.opened_at,
    })),
    closed: {
      n: closed.length, wins,
      winRate: closed.length ? +(wins / closed.length).toFixed(3) : 0,
      realized: +total.toFixed(2), maxDd: +maxDd.toFixed(2), byKind,
      recent: closed.slice(-20).reverse().map((c) => ({
        id: c.id, symbol: c.symbol, kind: c.kind, why: c.exit_why, pnl: c.pnl,
        exitSpot: c.exit_spot, exitTime: c.exit_time,
      })),
    },
    unrealized: +unrealized.toFixed(2),
    unrealizedNote: "fees pending; marked with live bid/ask touch, never mids",
    updatedAt: Date.now(),
    paperOnly: true,
    disclaimer: "Paper positions on delayed/unofficial quotes. A track record, not a promise — verify before risking capital.",
  };
}
