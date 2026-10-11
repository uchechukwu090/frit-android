// ============================================================================
// BinanceFeed — free keyless crypto klines over public WebSocket.
//
// Endpoint: wss://data-stream.binance.vision (market-data only host).
// Streams: <sym>@kline_1m + <sym>@kline_30m + <sym>@kline_4h — pre-built
// candles pushed every 2s. 1m feeds scalping freshness; 30m/4h map 1:1 onto
// the engine's two timeframes. Up to 1024 streams per connection; we use one
// connection for the whole crypto watchlist.
// NOTE: Nigeria/region reachability varies — LiveFeed treats this as tier-1
// for crypto and falls back to Finnhub crypto (BINANCE:BTCUSDT…) if the
// socket never goes live.
// ============================================================================

import { WsBase } from "./ws_base.js";

// FRIT symbol -> Binance spot symbol (USDT-quoted).
export const BINANCE_WS_MAP = {
  BTCUSD: "BTCUSDT", ETHUSD: "ETHUSDT", SOLUSD: "SOLUSDT", BNBUSD: "BNBUSDT",
  XRPUSD: "XRPUSDT", DOGEUSD: "DOGEUSDT", ADAUSD: "ADAUSDT",
  BTC: "BTCUSDT", ETH: "ETHUSDT", SOL: "SOLUSDT", BNB: "BNBUSDT",
  XRP: "XRPUSDT", DOGE: "DOGEUSDT", ADA: "ADAUSDT",
};

export class BinanceFeed extends WsBase {
  constructor({ onKline, onTrade, log, onStatus }) {
    super({
      name: "binance",
      url: "wss://data-stream.binance.vision/stream",
      log, onStatus,
    });
    this.onKline = onKline; // (fritSymbol, interval, {openTime,open,high,low,close,volume,isClosed})
    this.onTrade = onTrade || null; // (fritSymbol, price, qty, sideSign, tsMs) — order flow
    this.symbols = []; // FRIT symbols
  }

  setSymbols(fritSymbols) {
    this.symbols = [...new Set(fritSymbols.map(s => String(s).toUpperCase()))]
      .filter(s => BINANCE_WS_MAP[s]);
    if (this.ws && this.status() === "live") this._subscribe();
  }

  _streams() {
    const out = [];
    for (const s of this.symbols) {
      const b = BINANCE_WS_MAP[s].toLowerCase();
      // 1m feeds scalping freshness; 30m/4h feed the engine. aggTrade carries
      // buyer-maker flags => real order-flow imbalance. Still far under the
      // 1024-stream cap for any sane watchlist.
      out.push(`${b}@kline_1m`, `${b}@kline_30m`, `${b}@kline_4h`);
      if (this.onTrade) out.push(`${b}@aggTrade`);
    }
    return out.slice(0, 1000);
  }

  onOpen() { this._subscribe(); }

  _subscribe() {
    const streams = this._streams();
    if (!streams.length) return;
    this.send({ method: "SUBSCRIBE", params: streams, id: Date.now() % 100000 });
    if (this.log) this.log(`[binance] subscribed ${this.symbols.join(",")} (${streams.length} streams)`);
  }

  _fritOf(bsym) {
    const up = String(bsym || "").toUpperCase();
    return Object.keys(BINANCE_WS_MAP).find(f => BINANCE_WS_MAP[f] === up) || null;
  }

  onMessage(data) {
    let msg;
    try { msg = JSON.parse(String(data)); } catch { return; }
    const d = msg.data || msg;
    if (!d || !d.e) return; // ignore subscribe-acks etc.
    if (d.e === "aggTrade") {
      if (!this.onTrade) return;
      const frit = this._fritOf(d.s);
      if (!frit) return;
      const px = Number(d.p), qty = Number(d.q) || 0;
      if (!Number.isFinite(px) || px <= 0) return;
      // m=true => buyer IS the maker => seller-initiated (ask hit) => -1.
      try { this.onTrade(frit, px, qty, d.m ? -1 : 1, Number(d.T) || Date.now()); } catch { /* ignore */ }
      return;
    }
    if (d.e !== "kline" || !d.k) return;
    const k = d.k;
    const frit = this._fritOf(d.s);
    if (!frit) return;
    const interval = ["1m", "30m", "4h"].includes(k.i) ? k.i : "30m";
    try {
      this.onKline(frit, interval, {
        openTime: Number(k.t),
        open: Number(k.o), high: Number(k.h),
        low: Number(k.l), close: Number(k.c),
        volume: Number(k.v) || 0,
        isClosed: !!k.x,
      });
    } catch { /* ignore */ }
  }
}
