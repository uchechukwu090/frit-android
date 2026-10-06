// ============================================================================
// CandleStore — local OHLC rings built from streaming ticks.
//
// The Twelve Data free tier (8 credits/min, 800/day) cannot sustain REST
// polling for 24/7 scanning. Instead, ticks arriving over free WebSocket /
// poll feeds (Binance, Finnhub, Dukascopy) are aggregated here into the exact
// 30M + 4H candle shapes MTFStrategyEngine consumes. Steady-state scanning
// then costs 0 Twelve Data credits; REST is only the backfill/repair path.
//
// Persistence: closed candles live in SQLite (feed_candles) so a restart
// resumes with warm history. The still-forming candle lives in memory and is
// finalized on bucket rollover, firing onClose(symbol, interval, candle).
// ============================================================================

export const INTERVALS = {
  "30m": 30 * 60,
  "4h": 4 * 3600,
};

const KEEP_CLOSED = 300; // per symbol+interval

function bucketStart(tsMs, secs) {
  return Math.floor(tsMs / 1000 / secs) * secs * 1000;
}

export class CandleStore {
  constructor(db, opts = {}) {
    this.db = db;
    this.keep = opts.keep || KEEP_CLOSED;
    this.forming = new Map(); // `${sym}:${iv}` -> candle object
    this.onClose = opts.onClose || null; // (symbol, interval, closedCandle) => void
    this.lastTick = new Map(); // sym -> { price, ts }

    db.exec(`
      CREATE TABLE IF NOT EXISTS feed_candles (
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        open_time INTEGER NOT NULL,
        o REAL NOT NULL, h REAL NOT NULL, l REAL NOT NULL, c REAL NOT NULL,
        v REAL DEFAULT 0,
        source TEXT,
        PRIMARY KEY (symbol, interval, open_time)
      );
      CREATE INDEX IF NOT EXISTS idx_feed_sym_iv ON feed_candles(symbol, interval, open_time DESC);
      CREATE TABLE IF NOT EXISTS feed_meta (
        symbol TEXT PRIMARY KEY,
        last_tick_price REAL,
        last_tick_ts INTEGER
      );
    `);
  }

  _key(sym, iv) { return `${sym}:${iv}`; }

  // ---- tick path (Finnhub/OANDA-style ticks, Dukascopy quotes) ----
  ingestTick(symbol, price, volume = 0, tsMs = Date.now()) {
    const sym = String(symbol || "").toUpperCase();
    const px = Number(price);
    if (!Number.isFinite(px) || px <= 0) return null;
    this.lastTick.set(sym, { price: px, ts: tsMs });
    try {
      this.db.prepare(
        "INSERT OR REPLACE INTO feed_meta (symbol, last_tick_price, last_tick_ts) VALUES (?, ?, ?)"
      ).run(sym, px, tsMs);
    } catch { /* best-effort */ }

    const closed = [];
    for (const iv of Object.keys(INTERVALS)) {
      const c = this._applyTick(sym, iv, px, Number(volume) || 0, tsMs);
      if (c) closed.push({ interval: iv, candle: c });
    }
    return closed;
  }

  _applyTick(sym, iv, px, vol, tsMs) {
    const secs = INTERVALS[iv];
    const bucket = bucketStart(tsMs, secs);
    const key = this._key(sym, iv);
    let f = this.forming.get(key);
    if (!f || f.openTime !== bucket) {
      // Rollover: finalize previous bucket (if it has any ticks).
      let closed = null;
      if (f && f.count > 0) {
        closed = { ...f };
        delete closed.count;
        this._persist(sym, iv, closed, "ticks");
        this._prune(sym, iv);
      }
      f = { openTime: bucket, open: px, high: px, low: px, close: px, volume: 0, count: 0 };
      this.forming.set(key, f);
      if (closed && this.onClose) {
        try { this.onClose(sym, iv, closed); } catch { /* never crash feed on handler */ }
      }
      if (closed) return closed;
      // No previous bucket (first tick ever) — fall through to apply this tick.
    }
    if (px > f.high) f.high = px;
    if (px < f.low) f.low = px;
    f.close = px;
    f.volume += vol;
    f.count += 1;
    return null;
  }

  // ---- pre-built candle path (Binance kline_30m / kline_4h) ----
  // kline: { openTime, open, high, low, close, volume, isClosed }
  ingestKline(symbol, interval, kline) {
    const sym = String(symbol || "").toUpperCase();
    const iv = interval === "4h" ? "4h" : "30m";
    const secs = INTERVALS[iv];
    const bucket = bucketStart(Number(kline.openTime), secs);
    const c = {
      openTime: bucket,
      open: Number(kline.open), high: Number(kline.high),
      low: Number(kline.low), close: Number(kline.close),
      volume: Number(kline.volume) || 0,
    };
    if (![c.open, c.high, c.low, c.close].every(Number.isFinite)) return null;
    this.lastTick.set(sym, { price: c.close, ts: Date.now() });
    const key = this._key(sym, iv);
    if (kline.isClosed) {
      this._persist(sym, iv, c, "kline");
      this._prune(sym, iv);
      this.forming.delete(key); // next tick/kline starts a fresh bucket
      if (this.onClose) {
        try { this.onClose(sym, iv, { ...c }); } catch { /* best-effort */ }
      }
      return { ...c };
    }
    this.forming.set(key, { ...c, count: 1 });
    return null;
  }

  _persist(sym, iv, c, source) {
    try {
      this.db.prepare(`
        INSERT OR REPLACE INTO feed_candles (symbol, interval, open_time, o, h, l, c, v, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(sym, iv, c.openTime, c.open, c.high, c.low, c.close, c.volume || 0, source);
    } catch { /* best-effort */ }
  }

  _prune(sym, iv) {
    try {
      this.db.prepare(`
        DELETE FROM feed_candles WHERE symbol = ? AND interval = ?
        AND open_time NOT IN (
          SELECT open_time FROM feed_candles
          WHERE symbol = ? AND interval = ? ORDER BY open_time DESC LIMIT ?
        )
      `).run(sym, iv, sym, iv, this.keep);
    } catch { /* best-effort */ }
  }

  // Engine-ready array, oldest -> newest: { open, high, low, close, volume, time }
  getCandles(symbol, interval, n = 200) {
    const sym = String(symbol || "").toUpperCase();
    let rows = [];
    try {
      rows = this.db.prepare(`
        SELECT open_time AS t, o, h, l, c, v FROM feed_candles
        WHERE symbol = ? AND interval = ? ORDER BY open_time ASC LIMIT ?
      `).all(sym, interval, n);
    } catch { return []; }
    const out = rows.map(r => ({
      open: r.o, high: r.h, low: r.l, close: r.c, volume: r.v || 0, time: r.t,
    }));
    // Append the still-forming candle so the engine always sees the live edge.
    const f = this.forming.get(this._key(sym, interval));
    if (f && f.count > 0 && (!out.length || f.openTime > out[out.length - 1].time)) {
      out.push({ open: f.open, high: f.high, low: f.low, close: f.close, volume: f.volume, time: f.openTime });
    }
    return out;
  }

  candleCount(symbol, interval) {
    try {
      return this.db.prepare(
        "SELECT COUNT(*) AS n FROM feed_candles WHERE symbol = ? AND interval = ?"
      ).get(String(symbol).toUpperCase(), interval)?.n || 0;
    } catch { return 0; }
  }

  newestTime(symbol, interval) {
    try {
      return this.db.prepare(
        "SELECT MAX(open_time) AS t FROM feed_candles WHERE symbol = ? AND interval = ?"
      ).get(String(symbol).toUpperCase(), interval)?.t || 0;
    } catch { return 0; }
  }

  // Seconds since the newest closed 30M candle — staleness signal for backfill.
  stalenessSec(symbol, interval = "30m") {
    const t = this.newestTime(symbol, interval);
    if (!t) return Infinity;
    return (Date.now() - t) / 1000;
  }

  lastPrice(symbol) {
    const sym = String(symbol || "").toUpperCase();
    const live = this.lastTick.get(sym);
    if (live) return { price: live.price, ts: live.ts, source: "feed" };
    try {
      const row = this.db.prepare(
        "SELECT last_tick_price AS p, last_tick_ts AS t FROM feed_meta WHERE symbol = ?"
      ).get(sym);
      if (row && Number.isFinite(row.p)) return { price: row.p, ts: row.t, source: "feed-cache" };
    } catch { /* ignore */ }
    return null;
  }

  status() {
    let symbols = [];
    try {
      symbols = this.db.prepare(
        "SELECT symbol, interval, COUNT(*) AS n, MAX(open_time) AS t FROM feed_candles GROUP BY symbol, interval"
      ).all();
    } catch { /* ignore */ }
    return {
      intervals: INTERVALS,
      forming: [...this.forming.keys()],
      lastTicks: [...this.lastTick.entries()].map(([s, v]) => ({ symbol: s, ...v })),
      stored: symbols,
    };
  }
}
