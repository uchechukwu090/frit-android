// ============================================================================
// FRIT TRADE TASK SCHEDULER — "everyday analyze XAUUSD and place orders"
// ----------------------------------------------------------------------------
// A zero-dependency (setInterval) daily scheduler so recurring trade tasks run
// without n8n or node-cron. Each task is evaluated once per calendar day at its
// configured GMT time. Execution is PAPER-FIRST: when the engine emits a signal
// the decision is logged and (optionally) sent through the same MT5 bridge path
// as /trade — which is paper mode while MT5_BRIDGE_URL is unset.
//
// The Android "watch the phone, tap the trading app" execution layer plugs in
// here later; this module owns the schedule + the brain, not the hands.
// ============================================================================

export class TradeTaskScheduler {
  constructor(deps = {}) {
    this.engine = deps.engine;            // MTFStrategyEngine
    this.executor = deps.executor;        // async ({symbol, action, lotSize, entry, sl, tp, reason, taskId}) => result
    this.getPrice = deps.getPrice;        // async (symbol) => number | null (optional, for logging)
    this.tasks = new Map();
    this.history = [];
    this.maxHistory = 200;
    this._tickMs = deps.tickMs || 30_000;
    this._timer = null;
    this.lastTickAt = null;
    // FIX (F-05): tasks lived in a memory Map only — every user's scheduled
    // trade task was wiped by any server restart/deploy. When a db is passed
    // (index.js owns the SQLite handle), tasks are persisted and restored.
    this.db = deps.db || null;
    if (this.db) this._loadTasks();
  }

  _ensureTable() {
    try {
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS trade_tasks (
          id TEXT PRIMARY KEY,
          symbol TEXT NOT NULL,
          time TEXT NOT NULL,
          hh INTEGER NOT NULL,
          mm INTEGER NOT NULL,
          risk_percent REAL NOT NULL,
          balance REAL NOT NULL,
          enabled INTEGER NOT NULL DEFAULT 1,
          last_fired_day TEXT,
          created_at INTEGER NOT NULL
        );
      `);
      return true;
    } catch { return false; }
  }

  /** Restore all persisted tasks into the memory Map (memory stays authoritative at runtime). */
  _loadTasks() {
    if (!this.db || !this._ensureTable()) return;
    let rows = [];
    try { rows = this.db.prepare("SELECT * FROM trade_tasks").all(); } catch { return; }
    let restored = 0;
    for (const r of rows) {
      try {
        this.tasks.set(r.id, {
          id: r.id, symbol: r.symbol, time: r.time, hh: r.hh, mm: r.mm,
          riskPercent: r.risk_percent, balance: r.balance,
          enabled: r.enabled === 1, lastFiredDay: r.last_fired_day || null,
          created_at: r.created_at || Date.now(),
        });
        restored++;
      } catch { /* skip corrupt row — never block boot over one task */ }
    }
    if (restored) console.log(`[TradeTaskScheduler] restored ${restored} task(s) from DB`);
  }

  _persist(id) {
    if (!this.db) return;
    const t = this.tasks.get(id);
    if (!t) return;
    try {
      this.db.prepare(`
        INSERT INTO trade_tasks (id, symbol, time, hh, mm, risk_percent, balance, enabled, last_fired_day, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          symbol=excluded.symbol, time=excluded.time, hh=excluded.hh, mm=excluded.mm,
          risk_percent=excluded.risk_percent, balance=excluded.balance, enabled=excluded.enabled,
          last_fired_day=excluded.last_fired_day, created_at=excluded.created_at
      `).run(t.id, t.symbol, t.time, t.hh, t.mm, t.riskPercent, t.balance,
        t.enabled ? 1 : 0, t.lastFiredDay, t.created_at);
    } catch { /* persistence is best-effort; runtime state is authoritative */ }
  }

  _unpersist(id) {
    if (!this.db) return;
    try { this.db.prepare("DELETE FROM trade_tasks WHERE id = ?").run(id); } catch { /* best-effort */ }
  }

  /**
   * @param {object} task
   * @param {string} task.symbol        e.g. "XAUUSD"
   * @param {string} task.time          "HH:MM" in GMT (default "07:00")
   * @param {number} [task.riskPercent] risk % for lot sizing (default 1)
   * @param {number} [task.balance]     account balance for lot sizing (default 1000)
   * @param {string} [task.id]          optional id (default `${symbol}@${time}`)
   */
  add(task = {}) {
    const symbol = String(task.symbol || "").toUpperCase();
    if (!symbol) throw new Error("symbol required");
    const time = String(task.time || "07:00").trim();
    if (!/^\d{1,2}:\d{2}$/.test(time)) throw new Error("time must be HH:MM (GMT)");
    const id = task.id || `${symbol}@${time}`;
    const [hh, mm] = time.split(":").map(Number);
    if (hh > 23 || mm > 59) throw new Error("invalid time");
    const entry = {
      id, symbol, time, hh, mm,
      riskPercent: task.riskPercent ?? 1,
      balance: task.balance ?? 1000,
      enabled: task.enabled !== false,
      lastFiredDay: null,       // YYYY-MM-DD (GMT) of last evaluation — once/day
      created_at: Date.now(),
    };
    this.tasks.set(id, entry);
    this._persist(id);
    return { ok: true, id, ...entry };
  }

  remove(id) {
    const removed = this.tasks.delete(id);
    if (removed) this._unpersist(id);
    return removed;
  }

  start() {
    if (this._timer) return;
    this._timer = setInterval(() => this.tick(), this._tickMs);
    this._timer.unref?.();
    console.log(`[TradeTaskScheduler] started (tick ${this._tickMs}ms, ${this.tasks.size} task(s))`);
  }

  stop() {
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
  }

  async tick() {
    this.lastTickAt = new Date().toISOString();
    const now = new Date();
    const dayKey = now.toISOString().slice(0, 10); // GMT day
    const hh = now.getUTCHours();
    const mm = now.getUTCMinutes();

    for (const task of this.tasks.values()) {
      if (!task.enabled) continue;
      if (task.lastFiredDay === dayKey) continue;   // once per calendar day
      if (task.hh !== hh) continue;
      // Fire when the minute is >= target minute (catches tick drift)
      if (mm < task.mm) continue;

      task.lastFiredDay = dayKey;
      this._persist(task.id); // fired-day guard survives restarts (no double-fires / no refires)
      await this._evaluate(task);
    }
  }

  async _evaluate(task) {
    const start = Date.now();
    const run = {
      task_id: task.id, symbol: task.symbol, time: task.time,
      fired_at: new Date().toISOString(), decision: "ERROR", confidence: 0, note: "",
    };
    try {
      const result = await this.engine.run(task.symbol, {
        interval: "1h",
        riskPercent: task.riskPercent,
        balance: task.balance,
      });

      run.decision = result.decision;
      run.confidence = result.confidence ?? 0;
      run.reason = result.reason ?? null;
      run.entry = result.entry ?? null;
      run.sl = result.sl ?? null;
      run.tp = result.tp ?? null;
      run.rr = result.rr ?? null;

      if (result.decision === "BUY" || result.decision === "SELL") {
        const exec = await this.executor({
          symbol: task.symbol,
          action: result.decision === "BUY" ? "buy" : "sell",
          lotSize: result.lot_size ?? 0.01,
          entry: result.entry,
          sl: result.sl,
          tp: result.tp,
          reason: `MTF v1 scheduled task ${task.id} conf=${result.confidence}%`,
        });
        run.execution = exec;
        run.note = exec?.mode === "paper"
          ? "Signal fired — executed in PAPER mode (MT5_BRIDGE_URL unset)"
          : "Signal fired — sent to MT5 bridge";
        console.log(`[TradeTaskScheduler] ${task.id} -> ${result.decision} conf=${result.confidence}% (${run.note})`);
      } else {
        run.note = result.reason ?? "No signal";
        console.log(`[TradeTaskScheduler] ${task.id} -> ${result.decision}: ${run.note}`);
      }
    } catch (e) {
      run.note = `Error: ${e.message}`;
      console.error(`[TradeTaskScheduler] ${task.id} error:`, e.message);
    } finally {
      run.elapsed_ms = Date.now() - start;
      this.history.push(run);
      if (this.history.length > this.maxHistory) this.history.shift();
    }
  }

  status() {
    return {
      running: !!this._timer,
      tick_ms: this._tickMs,
      last_tick: this.lastTickAt,
      tasks: [...this.tasks.values()].map(t => ({
        id: t.id, symbol: t.symbol, time: t.time, enabled: t.enabled, last_fired_day: t.lastFiredDay,
      })),
      history_size: this.history.length,
      recent: this.history.slice(-10),
    };
  }
}
