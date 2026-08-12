// ============================================================================
// FRIT POSITION MONITOR — additive paper-position state machine
// ----------------------------------------------------------------------------
// Purely additive: this module exists ONLY to watch positions that were already
// placed by the daily trade scheduler or the /trade endpoint. It does not
// place orders, does not touch the /agent loop, and activates nothing on the
// phone. It just polls spot price and closes the bookkeeping when TP or SL is
// reached, so paper trades resolve themselves instead of lingering forever.
//
// State machine:  PLACED -> MONITORING -> RESOLVED (win / loss / cancelled)
//
// Dependencies are injected (zero module deps):
//   new PositionMonitor({
//     getPrice:  async (symbol) => number|null   // spot price fetcher
//     db:        better-sqlite3 instance          // persistence
//     onResolve: async (position) => void          // e.g. trade_memory + paper logger
//     intervalMs: 60_000                           // poll cadence
//   })
// ============================================================================

export class PositionMonitor {
  constructor(deps = {}) {
    this.getPrice = deps.getPrice;
    this.db = deps.db;
    this.onResolve = deps.onResolve || null;
    this._intervalMs = deps.intervalMs || 60_000;
    this._timer = null;
    this._lastTick = null;
    this._ensureTable();
  }

  _ensureTable() {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS positions (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        action TEXT NOT NULL,
        lot_size REAL,
        entry REAL,
        sl REAL,
        tp REAL,
        source TEXT,
        reason TEXT,
        status TEXT DEFAULT 'PLACED',
        paper_trade_id TEXT,
        placed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        monitoring_at DATETIME,
        resolved_at DATETIME,
        outcome TEXT,
        close_price REAL,
        note TEXT
      );
    `);
  }

  /**
   * Register a position and start watching it.
   * @returns {{id, symbol, action, entry, sl, tp, status}}
   */
  register({ symbol, action, lotSize, entry, sl, tp, source = "manual", reason = "", paperTradeId = null }) {
    const sym = String(symbol || "").toUpperCase();
    if (!sym) throw new Error("symbol required");
    if (!["buy", "sell"].includes(String(action).toLowerCase())) throw new Error("action must be buy or sell");
    const id = `${sym}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    this.db.prepare(
      `INSERT INTO positions (id, symbol, action, lot_size, entry, sl, tp, source, reason, status, paper_trade_id)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PLACED', ?)`
    ).run(id, sym, String(action).toLowerCase(), lotSize ?? 0.01, entry ?? null, sl ?? null, tp ?? null, source, reason, paperTradeId ?? null);
    // Placed -> monitoring immediately: polling starts now.
    this.db.prepare("UPDATE positions SET status='MONITORING', monitoring_at=CURRENT_TIMESTAMP WHERE id=?").run(id);
    console.log(`[PositionMonitor] registered ${id} ${String(action).toUpperCase()} ${sym} @ ${entry} | SL:${sl} TP:${tp}`);
    return { id, symbol: sym, action: String(action).toLowerCase(), lotSize, entry, sl, tp, status: "MONITORING" };
  }

  /**
   * Manually resolve a position (also used by auto-resolve).
   * @param {string} id
   * @param {"win"|"loss"|"cancelled"} outcome
   * @param {number|null} closePrice
   * @param {string} note
   */
  resolve(id, outcome, closePrice = null, note = "") {
    const pos = this.db.prepare("SELECT * FROM positions WHERE id=?").get(id);
    if (!pos) return { ok: false, error: "position not found" };
    if (pos.status === "RESOLVED") return { ok: true, already: true, position: pos };
    const outcomeNorm = ["win", "loss", "cancelled"].includes(String(outcome).toLowerCase())
      ? String(outcome).toLowerCase()
      : "cancelled";
    this.db.prepare(
      "UPDATE positions SET status='RESOLVED', outcome=?, close_price=?, resolved_at=CURRENT_TIMESTAMP, note=? WHERE id=?"
    ).run(outcomeNorm, closePrice ?? null, note, id);
    const updated = this.db.prepare("SELECT * FROM positions WHERE id=?").get(id);
    if (this.onResolve) {
      try { this.onResolve(updated); } catch (e) { console.error("[PositionMonitor] onResolve hook error:", e.message); }
    }
    return { ok: true, position: updated };
  }

  async _check(pos) {
    if (!pos || pos.status !== "MONITORING") return;
    const price = await this.getPrice(pos.symbol);
    if (price == null || !isFinite(price) || price <= 0) return; // no data this tick — keep watching
    const action = String(pos.action).toLowerCase();
    let outcome = null;
    if (action === "buy") {
      if (pos.tp != null && price >= pos.tp) outcome = "win";
      else if (pos.sl != null && price <= pos.sl) outcome = "loss";
    } else {
      if (pos.tp != null && price <= pos.tp) outcome = "win";
      else if (pos.sl != null && price >= pos.sl) outcome = "loss";
    }
    if (outcome) {
      const note = outcome === "win" ? `TP hit @ ${price}` : `SL hit @ ${price}`;
      console.log(`[PositionMonitor] ${pos.id} ${pos.symbol} -> ${outcome.toUpperCase()} @ ${price}`);
      this.resolve(pos.id, outcome, price, note);
    }
  }

  async _tick() {
    this._lastTick = new Date().toISOString();
    const rows = this.db.prepare("SELECT * FROM positions WHERE status='MONITORING'").all();
    for (const pos of rows) {
      try { await this._check(pos); } catch (e) { console.error(`[PositionMonitor] check error ${pos.id}:`, e.message); }
    }
  }

  start() {
    if (this._timer) return;
    this._timer = setInterval(() => this._tick(), this._intervalMs);
    this._timer.unref?.();
    console.log(`[PositionMonitor] started (tick ${this._intervalMs}ms)`);
  }

  stop() {
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
  }

  list(limit = 100) {
    return this.db.prepare("SELECT * FROM positions ORDER BY placed_at DESC LIMIT ?").all(limit);
  }

  get(id) {
    return this.db.prepare("SELECT * FROM positions WHERE id=?").get(id) || null;
  }

  status() {
    const rows = this.list(500);
    const count = s => rows.filter(r => r.status === s).length;
    return {
      running: !!this._timer,
      interval_ms: this._intervalMs,
      last_tick: this._lastTick,
      total: rows.length,
      placed: count("PLACED"),
      monitoring: count("MONITORING"),
      resolved: count("RESOLVED"),
      wins: rows.filter(r => r.outcome === "win").length,
      losses: rows.filter(r => r.outcome === "loss").length,
      recent: rows.slice(0, 10).map(p => ({
        id: p.id, symbol: p.symbol, action: p.action, entry: p.entry, sl: p.sl, tp: p.tp,
        status: p.status, outcome: p.outcome, close_price: p.close_price, source: p.source,
        placed_at: p.placed_at,
      })),
    };
  }
}
