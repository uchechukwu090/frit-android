// ============================================================================
// DerivFeed — synthetic indices via ticks_history candle polls (no auth).
// ----------------------------------------------------------------------------
// Deriv has no keyless tick WebSocket worth holding per symbol for candles,
// so this polls style=candles every 20s per watched synthetic and ingests
// every candle EXCEPT the still-forming edge (Deriv flags nothing as closed;
// the last candle is by construction incomplete). CandleStore fires close
// events when buckets roll, exactly like the WS feeds — downstream cannot
// tell the difference. A DerivClient factory is injected so tests pass a
// mock and production passes the real rail.
// ============================================================================

import { isDerivSymbol } from "../rails/router.js";

const POLL_MS = 20000;

export class DerivFeed {
  constructor({ makeClient, onKline, log, onStatus }) {
    this.makeClient = makeClient; // () => DerivClient (real or mock)
    this.onKline = onKline;       // (fritSymbol, interval, kline) => void
    this.log = log || (() => {});
    this.onStatus = onStatus || (() => {});
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
      .filter(s => isDerivSymbol(s));
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
        // 30m + 1h granularity is plenty for synthetics on a 20s poll; finer
        // intervals carve themselves from Deriv's per-tick flow via the store
        // only when a WS tick feed exists — polls re-seed 30m/1h, ticks carve
        // the rest. (Full 1m history per poll would 10x the payload.)
        for (const iv of ["30m", "1h"]) {
          let candles;
          try {
            candles = await client.candles(s, iv, 60);
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
