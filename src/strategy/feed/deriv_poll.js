// ============================================================================
// DerivFeed — Deriv as a full MARKET-DATA provider via ticks_history polls.
// ----------------------------------------------------------------------------
// (Generalized Oct 2026 for the new "Options" API, public socket
// wss://api.derivws.com/trading/v1/options/ws/public — keyless.)
//
// Covers EVERY market Deriv streams: its native synthetics (R_100, 1HZ100V,
// BOOM/CRASH), the frx* forex/metals, cry* crypto and OTC_* derived indices —
// anything derivUnderlying() maps. So a user can watch plain pairs (XAUUSD,
// EURUSD, BTCUSD) or true synthetics (BOOM1000) and get candles served free,
// even when Binance/Finnhub/Dukascopy have no live rail for them.
//
// No auth is needed for market data; nothing here trades.
//
// Polls style=candles every 20s per watched symbol and ingests every candle
// EXCEPT the still-forming edge (Deriv flags nothing as closed; the last
// candle is by construction incomplete). CandleStore fires close events when
// buckets roll, exactly like the WS feeds — downstream cannot tell the
// difference. Pocket-watched symbols are also polled at 1m/5m/15m so the
// fixed-expiry engine has fresh data on Deriv-only markets. A DerivClient
// factory is injected so tests pass a mock and production passes the real rail.
// ============================================================================

import { isDerivSymbol } from "../rails/router.js";
import { derivUnderlying, canDerivServe } from "../deriv.js";

const POLL_MS = 20000;
const MAIN_INTERVALS = ["30m", "1h"];
const POCKET_INTERVALS = ["1m", "5m", "15m", "1h"];

export class DerivFeed {
  constructor({ makeClient, onKline, log, onStatus, isPocket = null }) {
    this.makeClient = makeClient; // () => DerivClient (real or mock)
    this.onKline = onKline;       // (fritSymbol, interval, kline) => void
    this.log = log || (() => {});
    this.onStatus = onStatus || (() => {});
    this.isPocket = isPocket || (() => false); // (sym) => pocket engine watching
    this.symbols = [];
    this.timer = null;
    this.running = false;
    this._status = "idle";
    this._inflight = false;
  }

  setStatus(s, d = "") {
    this._status = s;
    try { this.onStatus("deriv", s, d); } catch { /* ignore */ }
  }
  status() { return this._status; }

  setSymbols(fritSymbols) {
    this.symbols = [...new Set(fritSymbols.map(s => String(s).toUpperCase()))]
      .filter(s => canDerivServe(s));
  }

  async start() {
    if (this.running) return;
    if (!this.symbols.length) return;
    this.running = true;
    this.setStatus("live", `polling ${this.symbols.join(",")}`);
    this._loop();
  }

  stop() {
    this.running = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    this.setStatus("stopped");
  }

  async _loop() {
    if (!this.running) return;
    if (!this._inflight) {
      this._inflight = true;
      try { await this._poll(); }
      catch (e) {
        this.log(`[deriv] poll failed: ${e.message}`);
        this.setStatus("down", e.message);
      }
      this._inflight = false;
      if (this._status === "down" && this.running) this.setStatus("live", `polling ${this.symbols.join(",")}`);
    }
    if (this.running) {
      this.timer = setTimeout(() => this._loop(), POLL_MS);
      this.timer.unref?.();
    }
  }

  async _poll() {
    const client = this.makeClient();
    try {
      for (const s of this.symbols) {
        const dSym = derivUnderlying(s) || s;
        // 30m + 1h granularity is plenty for MTF scanning on a 20s poll; finer
        // intervals are added only for pocket-watched symbols (1m/5m/15m feed
        // the fixed-expiry engine). Polls re-seed coarse buckets; ticks carve
        // the rest when a live tick provider also covers the symbol.
        const intervals = this.isPocket(s) ? POCKET_INTERVALS : MAIN_INTERVALS;
        for (const iv of intervals) {
          let candles;
          try {
            candles = await client.candles(dSym, iv, 60);
          } catch (e) {
            this.log(`[deriv] ${s} ${iv}: ${e.message}`);
            continue;
          }
          if (!Array.isArray(candles) || candles.length < 2) continue;
          // All but the forming edge are closed by construction.
          for (const c of candles.slice(0, -1)) {
            try {
              this.onKline(s, iv, {
                openTime: c.time,
                open: c.open, high: c.high, low: c.low, close: c.close,
                volume: 0, isClosed: true,
              });
            } catch { /* ignore */ }
          }
          // The edge keeps the forming bucket warm.
          const edge = candles[candles.length - 1];
          try {
            this.onKline(s, iv, {
              openTime: edge.time,
              open: edge.open, high: edge.high, low: edge.low, close: edge.close,
              volume: 0, isClosed: false,
            });
          } catch { /* ignore */ }
        }
      }
    } finally {
      try { client.close(); } catch { /* ignore */ }
    }
  }
}

// Re-exported for callers that only import this module.
export { isDerivSymbol };
