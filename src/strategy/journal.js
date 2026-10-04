// ============================================================================
// FRIT PULLBACK JOURNAL — auto-recorder for every published setup + accuracy
// ----------------------------------------------------------------------------
// The engine's edge claims (Fib zones, range fades) are hypotheses until this
// table says otherwise. Every published setup is recorded with its planned
// numbers; a resolver pass replays subsequent 30M candles and scores what
// actually happened — TP1 first (win), SL first (loss), or neither within
// the window (expired). Same-bar ambiguity resolves conservatively: SL first.
//
// Wires in two places (both additive, both optional):
//   1. engine dep `recordSetup`  — called on every published setup.
//   2. periodic resolvePending({ fetchCandles }) — replays candles, no trades.
// Endpoints: GET /strategy/journal/stats, GET /strategy/journal/recent,
//            POST /strategy/journal/resolve
// ============================================================================

const DEDUPE_MS = 6 * 60 * 60 * 1000; // same symbol+strategy within 6h = same idea, don't double-count
const DEFAULT_MAX_BARS = 96;          // 48h of 30M candles to prove the setup
const DEFAULT_MAX_AGE_DAYS = 14;

export class PullbackJournal {
  constructor({ db } = {}) {
    if (!db) throw new Error("PullbackJournal requires db");
    this.db = db;
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS pullback_journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        strategy TEXT NOT NULL,
        direction TEXT NOT NULL,
        entry REAL NOT NULL,
        zone_low REAL,
        zone_high REAL,
        sl REAL NOT NULL,
        tp1 REAL NOT NULL,
        planned_rr REAL,
        confluence INTEGER DEFAULT 0,
        pullback_prob REAL,
        recorded_at INTEGER NOT NULL,
        status TEXT DEFAULT 'PENDING',
        resolved_at INTEGER,
        outcome TEXT,
        win_first INTEGER,
        bars_held INTEGER,
        note TEXT
      );
      CREATE INDEX IF NOT EXISTS idx_pbj_symbol_status ON pullback_journal(symbol, status);
    `);
  }

  /** Record a published setup. Returns row id, or null if deduped/invalid. */
  record(setup = {}) {
    try {
      const symbol = String(setup.symbol || "").toUpperCase();
      const strategy = String(setup.strategy || "unknown");
      const entry = Number(setup.entry);
      const sl = Number(setup.sl);
      const tp1 = Number(setup.tp1);
      if (!symbol || !(entry > 0) || !(sl > 0) || !(tp1 > 0)) return null;
      const since = Date.now() - DEDUPE_MS;
      const dupe = this.db.prepare(
        "SELECT id FROM pullback_journal WHERE symbol=? AND strategy=? AND recorded_at>? LIMIT 1"
      ).get(symbol, strategy, since);
      if (dupe) return null;
      const r = this.db.prepare(`
        INSERT INTO pullback_journal
          (symbol, strategy, direction, entry, zone_low, zone_high, sl, tp1,
           planned_rr, confluence, pullback_prob, recorded_at, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        symbol, strategy, String(setup.direction || "").toUpperCase(),
        entry, setup.zoneLow ?? null, setup.zoneHigh ?? null, sl, tp1,
        setup.plannedRR ?? null, setup.confluence ? 1 : 0,
        setup.pullbackProb ?? null, Date.now(),
        String(setup.note || "").slice(0, 300)
      );
      return Number(r.lastInsertRowid);
    } catch (e) {
      console.warn("[PullbackJournal] record failed:", e.message);
      return null;
    }
  }

  /**
   * Resolve PENDING rows by replaying 30M candles after recorded_at.
   * Fill = bar range touches entry. From the fill bar, first touch of TP1
   * vs SL decides; a bar touching both counts SL first (conservative).
   */
  async resolvePending({ fetchCandles, maxBars = DEFAULT_MAX_BARS, maxAgeDays = DEFAULT_MAX_AGE_DAYS } = {}) {
    if (typeof fetchCandles !== "function") return { ok: false, error: "fetchCandles required" };
    const cutoff = Date.now() - maxAgeDays * 24 * 60 * 60 * 1000;
    const rows = this.db.prepare(
      "SELECT * FROM pullback_journal WHERE status='PENDING' AND recorded_at>? ORDER BY recorded_at ASC LIMIT 50"
    ).all(cutoff);
    let resolved = 0, expired = 0, errors = 0;
    for (const row of rows) {
      try {
        const out = await this._resolveOne(row, fetchCandles, maxBars);
        if (out === "resolved") resolved++;
        else if (out === "expired") expired++;
      } catch (e) {
        errors++;
        console.warn(`[PullbackJournal] resolve ${row.id} failed:`, e.message);
      }
    }
    // Anything older than the window that never resolved is history, not signal.
    const stale = this.db.prepare(
      "UPDATE pullback_journal SET status='EXPIRED', outcome='stale', resolved_at=? WHERE status='PENDING' AND recorded_at<=?"
    ).run(Date.now(), cutoff);
    return { ok: true, checked: rows.length, resolved, expired, errors, stale: stale.changes };
  }

  async _resolveOne(row, fetchCandles, maxBars) {
    const candles = await fetchCandles(row.symbol, "30m", 120);
    if (!Array.isArray(candles) || !candles.length) return "pending";
    const isLong = String(row.direction).toUpperCase() === "BUY";
    const bars = candles.filter(c => c.time > row.recorded_at).slice(0, maxBars);
    if (!bars.length) return "pending";
    let filledAt = -1;
    for (let i = 0; i < bars.length; i++) {
      const b = bars[i];
      if (isLong ? b.low <= row.entry : b.high >= row.entry) { filledAt = i; break; }
    }
    if (filledAt < 0) {
      // Never even filled within the window — setup irrelevance, not a loss.
      if (bars.length >= maxBars || Date.now() - row.recorded_at > 3 * 24 * 60 * 60 * 1000) {
        this._mark(row.id, "EXPIRED", "unfilled", 0, bars.length);
        return "expired";
      }
      return "pending";
    }
    for (let i = filledAt; i < bars.length; i++) {
      const b = bars[i];
      const hitTp = isLong ? b.high >= row.tp1 : b.low <= row.tp1;
      const hitSl = isLong ? b.low <= row.sl : b.high >= row.sl;
      if (hitTp || hitSl) {
        const win = hitTp && !hitSl ? true : hitSl && !hitTp ? false : false; // both touched = SL first
        this._mark(row.id, "RESOLVED", win ? "win" : "loss", win ? 1 : 0, i - filledAt + 1);
        return "resolved";
      }
    }
    if (bars.length >= maxBars) {
      this._mark(row.id, "EXPIRED", "timeout", 0, bars.length);
      return "expired";
    }
    return "pending";
  }

  _mark(id, status, outcome, winFirst, barsHeld) {
    this.db.prepare(
      "UPDATE pullback_journal SET status=?, outcome=?, win_first=?, bars_held=?, resolved_at=? WHERE id=?"
    ).run(status, outcome, winFirst, barsHeld, Date.now(), id);
  }

  /** Accuracy + expectancy. Expectancy in R: win pays planned_rr, loss pays -1. */
  stats({ symbol = null, strategy = null } = {}) {
    const where = [];
    const args = [];
    if (symbol) { where.push("symbol=?"); args.push(String(symbol).toUpperCase()); }
    if (strategy) { where.push("strategy=?"); args.push(String(strategy)); }
    const w = where.length ? `WHERE ${where.join(" AND ")}` : "";
    const rows = this.db.prepare(`SELECT * FROM pullback_journal ${w} ORDER BY recorded_at DESC LIMIT 500`).all(...args);
    const resolved = rows.filter(r => r.status === "RESOLVED");
    const wins = resolved.filter(r => r.outcome === "win");
    const losses = resolved.filter(r => r.outcome === "loss");
    const rSum = resolved.reduce((a, r) => a + (r.outcome === "win" ? (Number(r.planned_rr) || 0) : -1), 0);
    const by = {};
    for (const r of rows) {
      const k = `${r.strategy}`;
      by[k] = by[k] || { recorded: 0, resolved: 0, wins: 0, rSum: 0 };
      by[k].recorded++;
      if (r.status === "RESOLVED") {
        by[k].resolved++;
        if (r.outcome === "win") { by[k].wins++; by[k].rSum += Number(r.planned_rr) || 0; }
        else by[k].rSum -= 1;
      }
    }
    for (const k of Object.keys(by)) {
      const b = by[k];
      b.winRate = b.resolved ? Number(((b.wins / b.resolved) * 100).toFixed(1)) : null;
      b.expectancyR = b.resolved ? Number((b.rSum / b.resolved).toFixed(2)) : null;
    }
    return {
      symbol: symbol || "ALL",
      strategy: strategy || "ALL",
      recorded: rows.length,
      pending: rows.filter(r => r.status === "PENDING").length,
      resolved: resolved.length,
      wins: wins.length,
      losses: losses.length,
      expired: rows.filter(r => r.status === "EXPIRED").length,
      winRate: resolved.length ? Number(((wins.length / resolved.length) * 100).toFixed(1)) : null,
      expectancyR: resolved.length ? Number((rSum / resolved.length).toFixed(2)) : null,
      byStrategy: by,
      verdict: resolved.length < 30
        ? `Only ${resolved.length} resolved setups — below the 30-trade minimum. No verdict yet.`
        : (rSum / resolved.length) > 0
          ? `Positive expectancy (+${(rSum / resolved.length).toFixed(2)}R) over ${resolved.length} resolved setups. Edge holds after costs — size with discipline.`
          : `Negative expectancy (${(rSum / resolved.length).toFixed(2)}R) over ${resolved.length} resolved setups. Do NOT size up; re-tune or stand down.`,
    };
  }

  recent(limit = 20) {
    return this.db.prepare("SELECT * FROM pullback_journal ORDER BY recorded_at DESC LIMIT ?").all(Math.min(limit, 100));
  }
}
