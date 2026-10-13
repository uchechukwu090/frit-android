// ============================================================================
// FRIT LEDGER MATURITY — is any track record old enough to sell yet?
// ----------------------------------------------------------------------------
// A paper ledger becomes a SELLABLE track record only with both sample size
// and calendar time: resolved >= 30 trades AND >= 60 days tracked. Anything
// less is "growing" — shown honestly with exactly what is missing. This is
// the gate between paper signals and paid subscriptions: no mature book,
// no performance claims, no exceptions.
//
// Sources read (best-effort each; a missing table degrades that row, never
// the endpoint): options_paper, positions (spot/CFD), pocket_paper, deriv
// contracts, alpaca orders, pullback journal.
// ============================================================================

const MIN_TRADES = 30;
const MIN_DAYS = 60;

function daysBetween(a, b) {
  if (!a || !b) return 0;
  return Math.max(0, (b - a) / 86400000);
}

function verdict(n, days) {
  if (n >= MIN_TRADES && days >= MIN_DAYS) {
    return { mature: true, label: "MATURE", missing: "" };
  }
  const parts = [];
  if (n < MIN_TRADES) parts.push(`${MIN_TRADES - n} more resolved trades`);
  if (days < MIN_DAYS) parts.push(`${Math.ceil(MIN_DAYS - days)} more days`);
  return { mature: false, label: "GROWING", missing: parts.join(" + ") };
}

function row(db, sql, params, pick) {
  try {
    const r = db.prepare(sql).get(...params);
    if (!r) return null;
    return pick(r);
  } catch {
    return null;
  }
}

export function maturitySummary(db, user_id) {
  const books = [];
  const push = (name, n, firstTs, lastTs, extra = {}) => {
    const days = daysBetween(firstTs, lastTs || Date.now());
    books.push({ book: name, resolved: n, daysTracked: +days.toFixed(1), ...verdict(n, days), ...extra });
  };

  // Options premium book.
  const opt = row(db,
    `SELECT COUNT(*) AS n, MIN(opened_at) AS firstTs,
            SUM(CASE WHEN status != 'OPEN' THEN 1 ELSE 0 END) AS closed,
            SUM(CASE WHEN status != 'OPEN' AND pnl > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN status != 'OPEN' THEN pnl ELSE 0 END) AS realized
     FROM options_paper WHERE user_id IN (?, 'bot')`,
    [user_id],
    (r) => ({ n: r.closed || 0, first: r.firstTs, wins: r.wins || 0, realized: +(r.realized || 0).toFixed(2) })
  );
  if (opt) push("options", opt.n, opt.first, Date.now(), { winRate: opt.n ? +(opt.wins / opt.n).toFixed(3) : null, realized: opt.realized });

  // Spot/CFD positions (intraday, longterm, scheduler, manual, Deriv/Alpaca paper).
  const spot = row(db,
    `SELECT COUNT(*) AS n, MIN(placed_at) AS firstTs,
            SUM(CASE WHEN status = 'RESOLVED' THEN 1 ELSE 0 END) AS closed,
            SUM(CASE WHEN status = 'RESOLVED' AND outcome = 'win' THEN 1 ELSE 0 END) AS wins
     FROM positions`,
    [],
    (r) => ({ n: r.closed || 0, first: r.firstTs ? new Date(String(r.firstTs).replace(" ", "T")).getTime() : null, wins: r.wins || 0 })
  );
  // NOTE: positions has no user_id column (shared monitor) — counts are global.
  if (spot) push("spot", spot.n, spot.first, Date.now(), { winRate: spot.n ? +(spot.wins / spot.n).toFixed(3) : null, scope: "shared" });

  // Pocket fixed-expiry paper.
  const pocket = row(db,
    `SELECT COUNT(*) AS n, MIN(entry_ts) AS firstTs,
            SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS wins
     FROM pocket_paper WHERE user_id = ? AND result != 'OPEN'`,
    [user_id],
    (r) => ({ n: r.n || 0, first: r.firstTs, wins: r.wins || 0 })
  );
  if (pocket) push("pocket", pocket.n, pocket.first, Date.now(), { winRate: pocket.n ? +(pocket.wins / pocket.n).toFixed(3) : null });

  // Deriv contracts (demo/real broker truth).
  const deriv = row(db,
    `SELECT COUNT(*) AS n, MIN(placed_at) AS firstTs,
            SUM(CASE WHEN status = 'SETTLED' AND profit > 0 THEN 1 ELSE 0 END) AS wins
     FROM deriv_contracts WHERE status = 'SETTLED'`,
    [],
    (r) => ({ n: r.n || 0, first: r.firstTs, wins: r.wins || 0 })
  );
  if (deriv) push("deriv", deriv.n, deriv.first, Date.now(), { winRate: deriv.n ? +(deriv.wins / deriv.n).toFixed(3) : null, scope: "shared" });

  // Alpaca orders (fills tracked; P&L lives at the broker).
  const alp = row(db,
    `SELECT COUNT(*) AS n, MIN(placed_at) AS firstTs FROM alpaca_orders WHERE fill_price IS NOT NULL`,
    [],
    (r) => ({ n: r.n || 0, first: r.firstTs })
  );
  if (alp) push("alpaca", alp.n, alp.first, Date.now(), { note: "fills only — P&L at broker" });

  // Journal accuracy (signal-level, all strategies).
  const jou = row(db,
    `SELECT COUNT(*) AS n, MIN(recorded_at) AS firstTs,
            SUM(CASE WHEN status = 'RESOLVED' AND outcome = 'win' THEN 1 ELSE 0 END) AS wins
     FROM pullback_journal WHERE status = 'RESOLVED'`,
    [],
    (r) => ({ n: r.n || 0, first: r.firstTs, wins: r.wins || 0 })
  );
  if (jou) push("journal", jou.n, jou.first, Date.now(), { winRate: jou.n ? +(jou.wins / jou.n).toFixed(3) : null, scope: "shared" });

  const mature = books.filter((b) => b.mature);
  return {
    books,
    matureCount: mature.length,
    sellable: mature.map((b) => b.book),
    verdict: mature.length
      ? `${mature.length} track record(s) mature (${mature.map((b) => b.book).join(", ")}). Performance claims allowed ONLY for these, with dates attached.`
      : "No mature track record yet. No performance claims, no paid signal tiers based on results — growing books only.",
    thresholds: { minTrades: MIN_TRADES, minDays: MIN_DAYS },
    updatedAt: Date.now(),
  };
}
