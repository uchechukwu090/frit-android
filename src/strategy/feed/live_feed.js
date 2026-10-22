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
import { FinnhubFeed, FINNHUB_MAP, finnhubSymbol } from "./finnhub_ws.js";
import { DukascopyFeed } from "./dukascopy_poll.js";
import { DerivFeed } from "./deriv_poll.js";
import { AlpacaFeed } from "./alpaca_feed.js";
import { isDerivSymbol, marketOf } from "../rails/router.js";
import { canDerivServe, derivUnderlying } from "../deriv.js";

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
    this.getPocketWatchlist = deps.getPocketWatchlist || null; // separate pocket list
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

    this.binance = new BinanceFeed({
      onKline: (s, iv, k) => this.store.ingestKline(s, iv, k),
      onTrade: (s, p, q, side, ts) => this.store.ingestTrade(s, p, q, ts, side),
      log: this.log, onStatus,
    });
    this.finnhub = this.finnhubKey
      ? new FinnhubFeed({ apiKey: this.finnhubKey, fetchImpl: this.fetchImpl, onTick: (s, p, v, t) => this.store.ingestTick(s, p, v, t), log: this.log, onStatus })
      : null;
    this.dukascopy = new DukascopyFeed({ fetchImpl: this.fetchImpl, onTick: (s, p, v, t) => this.store.ingestTick(s, p, v, t), log: this.log, onStatus });
    // Rail clients are injected (index.js owns the shared Deriv/Alpaca
    // instances) so feeds never construct their own connections.
    this.makeDerivClient = deps.makeDerivClient || (() => { throw new Error("no deriv client"); });
    this.makeAlpacaClient = deps.makeAlpacaClient || (() => { throw new Error("no alpaca client"); });
    this.deriv = new DerivFeed({
      makeClient: this.makeDerivClient,
      onKline: (s, iv, k) => this.store.ingestKline(s, iv, k),
      log: this.log, onStatus,
      // Pocket-watched symbols get 1m/5m/15m polls too (fixed-expiry engine).
      isPocket: (s) => (this.pocketWatched?.has(s) ?? false),
    });
    this.alpaca = new AlpacaFeed({ makeClient: this.makeAlpacaClient, onTick: (s, p, v, t) => this.store.ingestTick(s, p, v, t), log: this.log, onStatus });

    this.watched = new Map(); // symbol -> Set(user_id)
    this.scanning = new Set(); // symbols with an in-flight engine run
    this.binanceFallbackTimer = null;
    this.onCandleClose = deps.onCandleClose || null; // pocket engine hook (every close)
    this.pocketEnabled = deps.pocketEnabled !== false;
  }

  // ---- lifecycle ----
  async start() {    if (!this.enabled) { this.log("[livefeed] disabled (FEED_ENABLED=0)"); return; }
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
    this.deriv.stop(); this.alpaca.stop();
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
    // Pocket-only pairs are fed (candles + backfill) but are NOT engine-scanned:
    // they only run through the fixed-expiry pocket path.
    let pRows = [];
    try { pRows = this.getPocketWatchlist ? this.getPocketWatchlist() || [] : []; } catch { pRows = []; }
    this.pocketWatched = new Map();
    for (const r of pRows) {
      const sym = String(r.symbol || "").toUpperCase();
      if (!sym) continue;
      if (!this.pocketWatched.has(sym)) this.pocketWatched.set(sym, new Set());
      this.pocketWatched.get(sym).add(r.user_id);
      if (!next.has(sym)) next.set(sym, new Set()); // feed it, don't scan it
    }
    const added = [...next.keys()].filter(s => !this.watched.has(s));
    this.watched = next;
    for (const s of added) { await this._backfill(s); await this._backfillPocket(s); }
    await this._routeAll();
    // REST-only safety net: symbols with no live provider (no Finnhub key and
    // Dukascopy bot-blocked from this host) refresh via budgeted REST when
    // their newest 30M candle goes stale — _needsBackfill + budget gate keep
    // this inside the daily credit cap.
    for (const s of [...next.keys()]) {
      if (!this._hasLiveProvider(s)) { await this._backfill(s); await this._backfillPocket(s); }
    }
  }

  /** Drop a pocket-only pair from the feed (user removed it). */
  async unwatchPocket(symbol) {
    const sym = String(symbol || "").toUpperCase();
    this.pocketWatched?.delete(sym);
    const stillMain = (this.watched?.get(sym)?.size || 0) > 0;
    if (!stillMain) this.watched?.delete(sym);
    try { this._routeAll(); } catch { /* ignore */ }
  }

  async _routeAll() {
    const syms = [...this.watched.keys()];
    const binanceLive = this.binance.status() === "live";
    const finnhubLive = this.finnhub ? this.finnhub.status() === "live" : false;
    const alpacaLive = this.alpaca.status() === "live";
    const alpacaKeys = this.alpaca.hasKeys();
    // Dynamic failover (re-evaluated every resync, i.e. every 5 min):
    //   crypto       : Binance WS -> Finnhub WS -> REST net
    //   forex/metals : Finnhub WS -> Dukascopy poll -> REST net
    //   synthetics   : Deriv poll (only source; keyless, free)
    //   US stocks    : Alpaca WS (keys) -> Finnhub WS (raw tickers) ->
    //                  Alpaca REST-delayed poll -> REST net
    // A provider that dies mid-session loses its symbols to the next tier
    // instead of leaving them dark; recovery routes them back up.
    const crypto = syms.filter(s => BINANCE_WS_MAP[s]);
    const synth = syms.filter(s => isDerivSymbol(s));
    const stocks = syms.filter(s => marketOf(s) === "alpaca");
    // Alpaca feed serves stocks when live, or when nothing better exists and
    // it can at least REST-poll (keys or not — start() picks WS vs delayed).
    const toAlpaca = stocks.filter(s => alpacaLive || !finnhubLive);
    // Crypto prefers Binance; Finnhub takes only what Binance can't serve live.
    const fhSyms = this.finnhub
      ? [...new Set([
        ...crypto.filter(() => !binanceLive),
        ...syms.filter(s => FINNHUB_MAP[s] && !BINANCE_WS_MAP[s] && !isDerivSymbol(s) && finnhubLive),
        ...stocks.filter(s => finnhubLive && !alpacaLive && !alpacaKeys),
      ])]
      : [];
    // Anything without a live WS tier falls to Dukascopy poll (it keeps only
    // symbols it actually lists, via NAME_MAP). Synthetics never go here.
    const dukaSyms = syms.filter(s => !BINANCE_WS_MAP[s] && !isDerivSymbol(s)
      && !toAlpaca.includes(s) && !(this.finnhub && FINNHUB_MAP[s] && finnhubLive));
    this.binance.setSymbols(crypto);
    if (this.finnhub) this.finnhub.setSymbols(fhSyms);
    this.alpaca.setSymbols(toAlpaca);
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
    // Free Deriv fallback — the keyless public feed covers ALL 89 Deriv
    // markets, so every deriv-servable symbol without a live claimant above
    // (binance/finnhub/duka/alpaca) is served by Deriv; pure synthetics/OTC
    // are ALWAYS on Deriv (nothing else streams them). Arbitration uses
    // duka's ACCEPTED list, so a bot-blocked duka never starves a pair it
    // nominally owns. Overlap is safe: CandleStore dedupes by bucket and
    // resync re-evaluates every 5 min.
    const dukaServing = this.dukascopy.status() === "live"
      ? new Set(this.dukascopy.symbols || [])
      : new Set();
    const derivExtra = syms.filter(s => !isDerivSymbol(s) && canDerivServe(s)
      && !(BINANCE_WS_MAP[s] && binanceLive)
      && !(this.finnhub && FINNHUB_MAP[s] && finnhubLive)
      && !dukaServing.has(s)
      && !toAlpaca.includes(s));
    const derivSyms = [...synth, ...derivExtra];
    this.deriv.setSymbols(derivSyms);
    const routing = {};
    for (const s of syms) {
      routing[s] = BINANCE_WS_MAP[s] && binanceLive ? "binance-ws"
        : derivSyms.includes(s) ? "deriv-poll"
        : toAlpaca.includes(s) ? (this.alpaca.status() === "live" ? "alpaca-ws" : "alpaca-rest")
        : (this.finnhub && finnhubSymbol(s) && finnhubLive) ? "finnhub-ws"
        : (this.dukascopy.symbols || []).includes(s) ? "dukascopy-poll"
        : "rest-only";
    }
    this.routing = routing;
    // (Re)start late feeds when they gain symbols (early-return when running).
    if (derivSyms.length) await this.deriv.start().catch(() => {});
    if (toAlpaca.length) { try { await this.alpaca.start(); } catch { /* ignore */ } }
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
    // 0. Deriv: free keyless candle history for synthetics, OTC indices and
    // every mapped FX/metals/crypto pair — no budget involved.
    if (canDerivServe(sym)) {
      try {
        const client = this.makeDerivClient();
        const dSym = derivUnderlying(sym) || sym;
        const [m30, h1] = await Promise.all([
          client.candles(dSym, "30m", 200).catch(() => []),
          client.candles(dSym, "1h", 200).catch(() => []),
        ]);
        try { client.close(); } catch { /* ignore */ }
        if (m30?.length >= 10) {
          // 4h history doesn't exist as native Deriv granularity — aggregate
          // exact 4h buckets from 1h history (same OHLC math as live carving).
          const h4 = [];
          const byBucket = new Map();
          for (const c of h1 || []) {
            const b = Math.floor(c.time / 14400000) * 14400000;
            if (!byBucket.has(b)) byBucket.set(b, []);
            byBucket.get(b).push(c);
          }
          for (const [b, cs] of [...byBucket.entries()].sort((x, y) => x[0] - y[0])) {
            cs.sort((x, y) => x.time - y.time);
            h4.push({
              time: b, open: cs[0].open,
              high: Math.max(...cs.map((x) => x.high)),
              low: Math.min(...cs.map((x) => x.low)),
              close: cs[cs.length - 1].close, volume: 0,
            });
          }
          this._seed(sym, m30, h4, "deriv");
          this.log(`[livefeed] backfilled ${sym} via deriv (free, ${m30.length}/30m ${h4.length}/4h)`);
          return;
        }
      } catch (e) { this.log(`[livefeed] deriv backfill ${sym} failed: ${e.message}`); }
    }
    // 0b. US stocks with Alpaca keys: IEX bars (delayed ok for seeding).
    if (marketOf(sym) === "alpaca") {
      try {
        const ac = this.makeAlpacaClient();
        if ((ac.keyId || "").length > 3) {
          const [{ candles: m30 }, { candles: h4 }] = await Promise.all([
            ac.bars(sym, "30m", 150).catch(() => ({ candles: [] })),
            ac.bars(sym, "4h", 100).catch(() => ({ candles: [] })),
          ]);
          if (m30?.length >= 10 && h4?.length >= 10) {
            this._seed(sym, m30, h4, "alpaca-rest-delayed");
            this.log(`[livefeed] backfilled ${sym} via alpaca (delayed)`);
            return;
          }
        }
      } catch (e) { this.log(`[livefeed] alpaca backfill ${sym} failed: ${e.message}`); }
    }
    // 1. Finnhub free REST (forex/stock/crypto) — 0 TD credits.
    if (this.finnhub && FINNHUB_MAP[sym]) {
      try {
        // NOTE: 4h via finnhub is aggregated from 4x1h inside FinnhubFeed.fetchCandles
        // (it requests count*4 hourly candles), so 100 here yields ~100 4h candles.
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

  _seedInterval(sym, interval, bucketMs, candles, source) {
    // Seed oldest->newest as closed candles (skip the live edge — ticks rebuild it).
    for (const c of candles.slice(0, -1)) {
      this.store._persist(sym, interval, {
        openTime: Math.floor(c.time / bucketMs) * bucketMs,
        open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume || 0,
      }, source);
    }
  }

  _seed(sym, m30, h4, source) {
    this._seedInterval(sym, "30m", 1800000, m30, source);
    this._seedInterval(sym, "4h", 14400000, h4, source);
  }

  // Pocket backfill: 1m/5m/15m/1h for the multi-expiry engine v2.
  // Free first (Finnhub REST), then Twelve Data inside the same budget.
  async _backfillPocket(sym) {
    if (!this.pocketEnabled) return;
    const spec = [
      ["1m", 60000, 80, "1min", 3], ["5m", 300000, 130, "5min", 8],
      ["15m", 900000, 130, "15min", 8], ["1h", 3600000, 100, "1h", 40],
    ];
    const need = spec.filter(([iv, , , , min]) =>
      this.store.candleCount(sym, iv) < min || this.store.stalenessSec(sym, iv) > 3600);
    if (!need.length) return;
    // Deriv first for Deriv-served markets: free, keyless, and often the only
    // source (synthetics/OTC indices have no other provider).
    if (canDerivServe(sym)) {
      let client = null;
      try {
        client = this.makeDerivClient();
        const dSym = derivUnderlying(sym) || sym;
        const res = await Promise.all(need.map(([iv]) => client.candles(dSym, iv, 130)));
        let ok = false;
        res.forEach((candles, i) => {
          const [iv, bucket, , , min] = need[i];
          if (candles?.length >= Math.min(min, 30)) {
            this._seedInterval(sym, iv, bucket, candles, "deriv");
            ok = true;
          }
        });
        if (ok) { this.log(`[livefeed] pocket-backfilled ${sym} via deriv (free)`); return; }
      } catch (e) { this.log(`[livefeed] pocket deriv backfill ${sym} failed: ${e.message}`); }
      finally { try { client?.close(); } catch { /* ignore */ } }
    }
    if (this.finnhub && FINNHUB_MAP[sym]) {
      try {
        const res = await Promise.all(need.map(([iv]) => this.finnhub.fetchCandles(sym, iv, 130)));
        let ok = false;
        res.forEach((candles, i) => {
          const [iv, bucket, , , min] = need[i];
          if (candles?.length >= min) {
            this._seedInterval(sym, iv, bucket, candles, "finnhub-rest");
            ok = true;
          }
        });
        if (ok) { this.log(`[livefeed] pocket-backfilled ${sym} via finnhub (free)`); return; }
      } catch (e) { this.log(`[livefeed] pocket finnhub backfill ${sym} failed: ${e.message}`); }
    }
    const cost = this.isCrypto(sym) ? 0 : 1; // crypto REST hits Binance (free)
    if (this._budgetUsedToday() + cost > this.backfillBudget) return;
    try {
      const res = await Promise.all(need.map(([iv, , , tdIv]) => this.fetchCandles(sym, tdIv, 130)));
      res.forEach((candles, i) => {
        const [iv, bucket] = need[i];
        if (candles?.length >= 10) this._seedInterval(sym, iv, bucket, candles, "rest");
      });
      this._budgetSpend(cost);
      this.log(`[livefeed] pocket-backfilled ${sym} via rest (cost ${cost})`);
    } catch (e) { this.log(`[livefeed] pocket rest backfill ${sym} failed: ${e.message}`); }
  }

  // ---- scan on candle close ----
  async _onCandleClose(sym, interval, _candle) {
    // Pocket hook fires for EVERY close (it listens to 5m/1m); MTF below only
    // scans symbols the feed watches.
    if (this.onCandleClose) {
      try { this.onCandleClose(sym, interval); } catch (e) { this.log(`[livefeed] close-hook failed: ${e.message}`); }
    }
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
    const shouldPush = isAlert && (changed || expired);
    // FIXED BUG (signals never appeared in the Signals tab): the push-dedupe
    // used to `return` BEFORE storing, so a signal whose decision had not
    // changed was never written to trading_signals at all. Storage and
    // notification are now decoupled — every actionable signal is PERSISTED
    // (that is the history the user reads), while the PUSH is still deduped
    // so the phone is not buzzed for the same call every candle.
    try {
      this.storeSignal(userId, { ...signal, _livePush: shouldPush });
    } catch (e) { this.log(`[livefeed] fanout store failed: ${e.message}`); }
    if (!shouldPush && !changed) return; // quiet, but the signal is recorded
    try {
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
