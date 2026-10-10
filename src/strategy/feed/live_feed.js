// ============================================================================
// LiveFeed — 24/7 scanning without burning the Twelve Data free tier.
//
// Data flow: provider ticks/klines -> CandleStore (30M/4H rings) ->
// on 30M-candle close -> engine.analyze({candles30M, candles4H}) (0 credits)
// -> actionable signals fanned out to every user watching the symbol.
//
// Provider tiers per symbol (first live one wins, rest stay as fallback):
//   crypto       : Binance WS (keyless) -> Finnhub WS (needs key)
//   forex/metals : Finnhub WS (needs key) -> Dukascopy poll (keyless)
//   stocks       : Finnhub WS (needs key) -> REST-only (budgeted TD refresh)
// Backfill order per symbol: Finnhub REST candles (free) -> Twelve Data REST
// (2 credits, guarded by a daily budget, default 200) -> accumulate live.
// Notification policy (anti-spam): BUY/SELL notifies a user only when the
// decision flipped or the last notify is >6h old; WAIT_PULLBACK/WAIT_RANGE
// are stored (max 1 per 4h per user+symbol) but never push-notified.
// ============================================================================

import { CandleStore } from "./candle_store.js";
import { BinanceFeed, BINANCE_WS_MAP } from "./binance_ws.js";
import { FinnhubFeed, FINNHUB_MAP } from "./finnhub_ws.js";
import { DukascopyFeed } from "./dukascopy_poll.js";

const NEED_30M = 70;
const NEED_4H = 40;
const NOTIFY_COOLDOWN_MS = 6 * 3600 * 1000;
const WAIT_STORE_COOLDOWN_MS = 4 * 3600 * 1000;

export class LiveFeed {
  constructor(deps) {
    this.db = deps.db;
    this.engine = deps.engine;               // MTFStrategyEngine
    this.fetchImpl = deps.fetchImpl;         // node-fetch
    this.fetchCandles = deps.fetchCandles;   // index.js fetchCandles (TD/Binance REST)
    this.formatSignal = deps.formatSignal;   // index.js formatTradingSignal
    this.storeSignal = deps.storeSignal;     // (userId, signal) => void
    this.getWatchlist = deps.getWatchlist;   // () => [{user_id, symbol}]
    this.isCrypto = deps.isCrypto || (() => false);
    this.finnhubKey = deps.finnhubKey || "";
    this.enabled = deps.enabled !== false;
    this.backfillBudget = deps.backfillBudget ?? 200; // TD credits/day max
    this.log = deps.log || console.log;

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS feed_budget (day TEXT PRIMARY KEY, credits INTEGER DEFAULT 0);
      CREATE TABLE IF NOT EXISTS feed_last_signal (
        user_id TEXT NOT NULL, symbol TEXT NOT NULL,
        decision TEXT, entry REAL, ts INTEGER,
        PRIMARY KEY (user_id, symbol)
      );
    `);

    this.store = new CandleStore(this.db, { onClose: (s, iv, c) => this._onCandleClose(s, iv, c) });
    this.providerStatus = {};
    const onStatus = (name, s, d) => { this.providerStatus[name] = { status: s, detail: d, at: Date.now() }; };

    this.binance = new BinanceFeed({ onKline: (s, iv, k) => this.store.ingestKline(s, iv, k), log: this.log, onStatus });
    this.finnhub = this.finnhubKey
      ? new FinnhubFeed({ apiKey: this.finnhubKey, fetchImpl: this.fetchImpl, onTick: (s, p, v, t) => this.store.ingestTick(s, p, v, t), log: this.log, onStatus })
      : null;
    this.dukascopy = new DukascopyFeed({ fetchImpl: this.fetchImpl, onTick: (s, p, v, t) => this.store.ingestTick(s, p, v, t), log: this.log, onStatus });

    this.watched = new Map(); // symbol -> Set(user_id)
    this.scanning = new Set(); // symbols with an in-flight engine run
    this.binanceFallbackTimer = null;
  }

  // ---- lifecycle ----
  async start() {
    if (!this.enabled) { this.log("[livefeed] disabled (FEED_ENABLED=0)"); return; }
    this.log("[livefeed] starting — cache-first pricing active");
    await this.resync();
    this.binance.start();
    if (this.finnhub) this.finnhub.start();
    await this.dukascopy.start(); // no-op if no symbols routed to it
    // If Binance never goes live (region block), route crypto to Finnhub too.
    this.binanceFallbackTimer = setTimeout(() => {
      if (this.binance.status() !== "live" && this.finnhub) {
        const crypto = [...this.watched.keys()].filter(s => BINANCE_WS_MAP[s]);
        if (crypto.length) {
          this.log("[livefeed] binance not live — falling back to finnhub for crypto");
          this.finnhub.setSymbols([...this.finnhub.symbols, ...crypto]);
        }
      }
    }, 45_000);
    if (this.binanceFallbackTimer.unref) this.binanceFallbackTimer.unref();
    // Watchlist resync + REST-only refresh every 5 min.
    this._resyncTimer = setInterval(() => { this.resync().catch(() => {}); }, 5 * 60 * 1000);
    if (this._resyncTimer.unref) this._resyncTimer.unref();
  }

  stop() {
    clearTimeout(this.binanceFallbackTimer);
    clearInterval(this._resyncTimer);
    this.binance.stop(); this.finnhub?.stop(); this.dukascopy.stop();
  }

  // ---- watchlist routing ----
  async resync() {
    let rows = [];
    try { rows = this.getWatchlist() || []; } catch { rows = []; }
    const next = new Map();
    for (const r of rows) {
      const sym = String(r.symbol || "").toUpperCase();
      if (!sym) continue;
      if (!next.has(sym)) next.set(sym, new Set());
      next.get(sym).add(r.user_id);
    }
    const added = [...next.keys()].filter(s => !this.watched.has(s));
    this.watched = next;
    for (const s of added) await this._backfill(s);
    this._routeAll();
    // REST-only safety net: symbols with no live provider (no Finnhub key and
    // Dukascopy bot-blocked from this host) refresh via budgeted REST when
    // their newest 30M candle goes stale — _needsBackfill + budget gate keep
    // this inside the daily credit cap.
    for (const s of [...this.watched.keys()]) {
      if (!this._hasLiveProvider(s)) await this._backfill(s);
    }
  }

  _routeAll() {
    const syms = [...this.watched.keys()];
    const binanceLive = this.binance.status() === "live";
    const finnhubLive = this.finnhub ? this.finnhub.status() === "live" : false;
    // Dynamic failover (re-evaluated every resync, i.e. every 5 min):
    //   crypto       : Binance WS -> Finnhub WS -> REST net
    //   forex/metals : Finnhub WS -> Dukascopy poll -> REST net
    // A provider that dies mid-session loses its symbols to the next tier
    // instead of leaving them dark; recovery routes them back up.
    const crypto = syms.filter(s => BINANCE_WS_MAP[s]);
    // Crypto prefers Binance; Finnhub takes only what Binance can't serve live.
    const fhSyms = this.finnhub
      ? [...new Set([
        ...crypto.filter(() => !binanceLive),
        ...syms.filter(s => FINNHUB_MAP[s] && !BINANCE_WS_MAP[s] && finnhubLive),
      ])]
      : [];
    // Anything without a live WS tier falls to Dukascopy poll (it keeps only
    // symbols it actually lists, via NAME_MAP).
    const dukaSyms = syms.filter(s => !BINANCE_WS_MAP[s] && !(this.finnhub && FINNHUB_MAP[s] && finnhubLive));
    this.binance.setSymbols(crypto);
    if (this.finnhub) this.finnhub.setSymbols(fhSyms);
    const prevDuka = [...(this.dukascopy.symbols || [])].sort().join(",");
    this.dukascopy.setSymbols(dukaSyms);
    const nextDuka = [...(this.dukascopy.symbols || [])].sort().join(",");
    if (nextDuka !== prevDuka && this.dukascopy.running) {
      // Set changed mid-run: restart so new symbols get instrument ids.
      // _resolveIds is one HTTP call; resyncs are 5 min apart.
      this.dukascopy.stop();
      this.dukascopy.start().catch(() => {});
      this.log(`[livefeed] dukascopy rerouted: ${nextDuka || "(none)"}`);
    }
    if (dukaSyms.length && !this.dukascopy.running) this.dukascopy.start().catch(() => {});
    const routing = {};
    for (const s of syms) {
      routing[s] = BINANCE_WS_MAP[s] && binanceLive ? "binance-ws"
        : (this.finnhub && FINNHUB_MAP[s] && finnhubLive) ? "finnhub-ws"
        : (this.dukascopy.symbols || []).includes(s) ? "dukascopy-poll"
        : "rest-only";
    }
    this.routing = routing;
  }

  // ---- backfill (free first, Twelve Data last, budget-guarded) ----
  _budgetUsedToday() {
    const day = new Date().toISOString().slice(0, 10);
    try {
      return this.db.prepare("SELECT credits FROM feed_budget WHERE day = ?").get(day)?.credits || 0;
    } catch { return 0; }
  }

  _budgetSpend(n) {
    const day = new Date().toISOString().slice(0, 10);
    try {
      this.db.prepare(
        "INSERT INTO feed_budget (day, credits) VALUES (?, ?) ON CONFLICT(day) DO UPDATE SET credits = credits + ?"
      ).run(day, n, n);
    } catch { /* ignore */ }
  }

  _hasLiveProvider(sym) {
    if (BINANCE_WS_MAP[sym] && this.binance.status() === "live") return true;
    if (this.finnhub && FINNHUB_MAP[sym] && this.finnhub.status() === "live") return true;
    if (this.dukascopy.status() === "live" && this.dukascopy.symbols.includes(sym)) return true;
    return false;
  }

  _needsBackfill(sym) {
    return this.store.candleCount(sym, "30m") < NEED_30M
      || this.store.candleCount(sym, "4h") < NEED_4H
      || this.store.stalenessSec(sym, "30m") > 2 * 30 * 60;
  }

  async _backfill(sym) {
    if (!this._needsBackfill(sym)) return;
    // 1. Finnhub free REST (forex/stock/crypto) — 0 TD credits.
    if (this.finnhub && FINNHUB_MAP[sym]) {
      try {
        const [m30, h4] = await Promise.all([
          this.finnhub.fetchCandles(sym, "30m", 150),
          this.finnhub.fetchCandles(sym, "4h", 100),
        ]);
        if (m30?.length >= NEED_30M && h4?.length >= NEED_4H) {
          this._seed(sym, m30, h4, "finnhub-rest");
          this.log(`[livefeed] backfilled ${sym} via finnhub (free)`);
          return;
        }
      } catch (e) { this.log(`[livefeed] finnhub backfill ${sym} failed: ${e.message}`); }
    }
    // 2. Twelve Data (/Binance REST for crypto) — budget-guarded.
    //    Crypto via fetchCandles hits Binance REST first (free); only count
    //    budget for non-crypto, which truly spends TD credits.
    const cost = this.isCrypto(sym) ? 0 : 2;
    if (this._budgetUsedToday() + cost > this.backfillBudget) {
      this.log(`[livefeed] backfill budget exhausted — ${sym} accumulates live`);
      return;
    }
    try {
      const [m30, h4] = await Promise.all([
        this.fetchCandles(sym, "30m", 150),
        this.fetchCandles(sym, "4h", 100),
      ]);
      if (m30?.length >= 10 && h4?.length >= 10) {
        this._seed(sym, m30, h4, "rest");
        this._budgetSpend(cost);
        this.log(`[livefeed] backfilled ${sym} via rest (${m30.length}/30m ${h4.length}/4h, cost ${cost})`);
      }
    } catch (e) { this.log(`[livefeed] rest backfill ${sym} failed: ${e.message}`); }
  }

  _seed(sym, m30, h4, source) {
    // Seed oldest->newest as closed candles (skip the live edge — ticks rebuild it).
    for (const c of m30.slice(0, -1)) {
      this.store._persist(sym, "30m", {
        openTime: Math.floor(c.time / 1800000) * 1800000,
        open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume || 0,
      }, source);
    }
    for (const c of h4.slice(0, -1)) {
      this.store._persist(sym, "4h", {
        openTime: Math.floor(c.time / 14400000) * 14400000,
        open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume || 0,
      }, source);
    }
  }

  // ---- scan on candle close ----
  async _onCandleClose(sym, interval, _candle) {
    if (!this.watched.has(sym) || this.scanning.has(sym)) return;
    // Engine scans fire on 30M/4H closes only. Finer intervals (1m/5m/...) just
    // accumulate in the store for the bots UI and scalping freshness — a full
    // 30M/4H analysis on every 1-minute close would burn CPU for zero signal.
    if (interval !== "30m" && interval !== "4h") return;
    // Scan on 30M closes (the engine's primary TF); 4H closes also trigger
    // since regime flips matter. Skip forming-candle noise: only closed events
    // arrive here by construction.
    const c30 = this.store.getCandles(sym, "30m", 150);
    const c4h = this.store.getCandles(sym, "4h", 100);
    if (c30.length < NEED_30M || c4h.length < NEED_4H) {
      // Still warming up — backfill in background (once per symbol per resync).
      return;
    }
    this.scanning.add(sym);
    try {
      const result = await this.engine.analyze(sym, { candles30M: c30, candles4H: c4h });
      const actionable = ["BUY", "SELL", "WAIT_PULLBACK", "WAIT_RANGE"].includes(result.decision);
      if (!actionable) return;
      const signal = this.formatSignal(sym, result);
      const users = [...(this.watched.get(sym) || [])];
      for (const uid of users) this._fanOut(uid, signal);
    } catch (e) {
      this.log(`[livefeed] scan ${sym} failed: ${e.message}`);
    } finally {
      this.scanning.delete(sym);
    }
  }

  _fanOut(userId, signal) {
    const now = Date.now();
    let last = null;
    try {
      last = this.db.prepare(
        "SELECT decision, entry, ts FROM feed_last_signal WHERE user_id = ? AND symbol = ?"
      ).get(userId, signal.symbol);
    } catch { /* ignore */ }
    const isAlert = signal.decision === "BUY" || signal.decision === "SELL";
    const cooldown = isAlert ? NOTIFY_COOLDOWN_MS : WAIT_STORE_COOLDOWN_MS;
    const changed = !last || last.decision !== signal.decision;
    const expired = !last || (now - last.ts) > cooldown;
    if (!changed && !expired) return; // same signal, still fresh — stay quiet
    try {
      this.storeSignal(userId, { ...signal, _livePush: isAlert && (changed || expired) });
      this.db.prepare(
        "INSERT OR REPLACE INTO feed_last_signal (user_id, symbol, decision, entry, ts) VALUES (?, ?, ?, ?, ?)"
      ).run(userId, signal.symbol, signal.decision, signal.entry ?? null, now);
    } catch (e) { this.log(`[livefeed] fanout failed: ${e.message}`); }
  }

  lastPrice(symbol) { return this.store.lastPrice(symbol); }

  status() {
    return {
      enabled: this.enabled,
      watched: [...this.watched.keys()],
      routing: this.routing || {},
      providers: this.providerStatus,
      budgetUsedToday: this._budgetUsedToday(),
      budgetMax: this.backfillBudget,
      store: this.store.status(),
    };
  }
}
