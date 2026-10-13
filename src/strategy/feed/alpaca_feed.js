// ============================================================================
// AlpacaFeed — US stocks/ETFs: realtime minute bars over WS, REST fallback.
// ----------------------------------------------------------------------------
// Free plan reality: the WS stream (IEX) is realtime but capped at 30 symbols;
// REST bars are ~15 min delayed. So: WS when keys exist (bars feed the store
// as ticks at each minute close, carving every interval), REST poll every 60s
// otherwise — with routing flagged "rest-delayed" so nobody mistakes delayed
// prints for live ones. Without keys this feed stays down and symbols fall to
// Finnhub WS (raw tickers) or the REST net via LiveFeed failover.
// ============================================================================

import WebSocket from "ws";
import { marketOf } from "../rails/router.js";

const WS_URL = "wss://stream.data.alpaca.markets/v2/iex";
const POLL_MS = 60000;

export class AlpacaFeed {
  constructor({ makeClient, onTick, log, onStatus, WebSocketImpl = WebSocket }) {
    this.makeClient = makeClient; // () => AlpacaClient (real or mock)
    this.onTick = onTick;         // (fritSymbol, price, volume, tsMs) => void
    this.log = log || (() => {});
    this.onStatus = onStatus || (() => {});
    this.WS = WebSocketImpl;
    this.symbols = [];
    this.ws = null;
    this.timer = null;
    this.polling = false;
    this._status = "idle";
  }

  setStatus(s, d = "") {
    this._status = s;
    try { this.onStatus("alpaca", s, d); } catch { /* ignore */ }
  }
  status() { return this._status; }

  setSymbols(fritSymbols) {
    this.symbols = [...new Set(fritSymbols.map(s => String(s).toUpperCase()))]
      .filter(s => marketOf(s) === "alpaca").slice(0, 30); // WS cap
  }

  hasKeys() {
    try {
      const c = this.makeClient();
      const ok = (c.keyId || "").length > 3;
      return ok;
    } catch { return false; }
  }

  async start() {
    if (!this.symbols.length) return;
    if (this.hasKeys()) this._connectWs();
    else {
      // No keys: REST-delayed poll. Better than dark; flagged downstream.
      this.polling = true;
      this.setStatus("rest-delayed", "no keys — 15-min delayed REST poll");
      this._pollLoop();
    }
  }

  stop() {
    try { this.ws?.close(); } catch { /* ignore */ }
    this.ws = null;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    this.polling = false;
    this.setStatus("stopped");
  }

  _connectWs() {
    let keyId = "", secret = "";
    try {
      const c = this.makeClient();
      keyId = c.keyId; secret = c.secret;
    } catch (e) {
      this.setStatus("down", e.message);
      return;
    }
    const ws = new this.WS(WS_URL);
    this.ws = ws;
    ws.on("open", () => {
      ws.send(JSON.stringify({ action: "auth", key: keyId, secret }));
    });
    ws.on("message", (raw) => {
      let arr;
      try { arr = JSON.parse(String(raw)); } catch { return; }
      if (!Array.isArray(arr)) return;
      for (const m of arr) {
        if (m.T === "success") {
          if (m.msg === "connected") continue;
          if (m.msg === "authenticated") {
            this.setStatus("live", `ws ${this.symbols.join(",")}`);
            ws.send(JSON.stringify({ action: "subscribe", bars: this.symbols }));
          }
          continue;
        }
        // Minute bar: {T:"b", S, o,h,l,c, v, t}. Feed the close as a tick so
        // CandleStore carves 1m..4h intervals from one stream.
        if (m.T === "b" && m.S && m.c > 0) {
          try {
            const sym = String(m.S).toUpperCase().replace(".", ".");
            this.onTick(sym, Number(m.c), Number(m.v) || 0, new Date(m.t).getTime() || Date.now());
          } catch { /* ignore */ }
        }
      }
    });
    ws.on("close", () => {
      if (this.ws === ws) this.ws = null;
      this.setStatus("down", "ws closed");
    });
    ws.on("error", () => { /* close handler reports */ });
  }

  async _pollLoop() {
    if (!this.polling) return;
    try {
      const client = this.makeClient();
      for (const s of this.symbols) {
        try {
          const { candles } = await client.bars(s, "15m", 20);
          const last = candles[candles.length - 1];
          if (last) this.onTick(s, last.close, last.volume || 0, last.time);
        } catch (e) { this.log(`[alpaca] poll ${s}: ${e.message}`); }
      }
    } catch (e) { this.log(`[alpaca] poll failed: ${e.message}`); }
    if (this.polling) {
      this.timer = setTimeout(() => this._pollLoop(), POLL_MS);
      this.timer.unref?.();
    }
  }
}
