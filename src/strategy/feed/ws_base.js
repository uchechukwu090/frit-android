// ============================================================================
// WsBase — reconnecting WebSocket client shared by all live-feed providers.
// - exponential backoff reconnect (1s -> 60s max), optional forced recycle
//   (Binance drops connections at the 24h mark — we recycle at 23h)
// - resubscribe hook after every (re)connect
// - status callback so LiveFeed can report provider health
// Uses the `ws` package (zero-dep, works on Node 18+ where global WebSocket
// may not exist).
// ============================================================================

import WebSocket from "ws";

export class WsBase {
  constructor({ name, url, log = null, onStatus = null }) {
    this.name = name;
    this.url = url;
    this.log = log || (() => {});
    this.onStatus = onStatus || (() => {});
    this.ws = null;
    this.wanted = false;
    this.backoffMs = 1000;
    this.recycleTimer = null;
    this.pingTimer = null;
    this._status = "idle";
  }

  setStatus(s, detail = "") {
    this._status = s;
    try { this.onStatus(this.name, s, detail); } catch { /* ignore */ }
  }

  status() { return this._status; }

  start() {
    this.wanted = true;
    this.backoffMs = 1000;
    this._connect();
  }

  stop() {
    this.wanted = false;
    this._clearTimers();
    try { this.ws?.close(); } catch { /* ignore */ }
    this.ws = null;
    this.setStatus("stopped");
  }

  _clearTimers() {
    if (this.recycleTimer) { clearTimeout(this.recycleTimer); this.recycleTimer = null; }
    if (this.pingTimer) { clearInterval(this.pingTimer); this.pingTimer = null; }
  }

  _connect() {
    if (!this.wanted) return;
    this.setStatus("connecting", this.url);
    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch (e) {
      this._scheduleReconnect(`ctor: ${e.message}`);
      return;
    }
    this.ws = ws;

    ws.on("open", () => {
      this.backoffMs = 1000;
      this.setStatus("live");
      try { this.onOpen(); } catch (e) { this.log(`[${this.name}] onOpen failed: ${e.message}`); }
      // Proactive recycle before the server-side 24h drop (Binance).
      this._clearTimers();
      this.recycleTimer = setTimeout(() => {
        this.log(`[${this.name}] proactive 23h recycle`);
        try { ws.close(); } catch { /* ignore */ }
      }, 23 * 3600 * 1000);
      if (this.recycleTimer.unref) this.recycleTimer.unref();
    });

    ws.on("message", (data) => {
      try { this.onMessage(data); } catch (e) { this.log(`[${this.name}] onMessage failed: ${e.message}`); }
    });

    ws.on("ping", () => { try { ws.pong(); } catch { /* ignore */ } });

    const down = (why) => {
      if (this.ws !== ws) return; // stale socket after recycle
      this._clearTimers();
      this.setStatus("down", why);
      this._scheduleReconnect(why);
    };
    ws.on("close", (code, reason) => down(`close ${code} ${String(reason || "").slice(0, 80)}`));
    ws.on("error", (e) => down(`error ${e.message}`));
  }

  _scheduleReconnect(why) {
    if (!this.wanted) return;
    this.log(`[${this.name}] reconnect in ${this.backoffMs}ms (${why})`);
    const wait = this.backoffMs;
    this.backoffMs = Math.min(this.backoffMs * 2, 60_000);
    setTimeout(() => this._connect(), wait);
  }

  send(obj) {
    try {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(obj));
        return true;
      }
    } catch { /* ignore */ }
    return false;
  }

  // Overridden by subclasses.
  onOpen() {}
  onMessage(_data) {}
}
