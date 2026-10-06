// ============================================================================
// BinanceFeed — free keyless crypto klines over public WebSocket.
//
// Endpoint: wss://data-stream.binance.vision (market-data only host).
// Streams: <sym>@kline_30m + <sym>@kline_4h — pre-built candles pushed every
// 2s, mapping 1:1 onto the engine's two timeframes. Up to 1024 streams per
// connection; we use one connection for the whole crypto watchlist.
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
  constructor({ onKline, log, onStatus }) {
    super({
      name: "binance",
      url: "wss://data-stream.binance.vision/stream",
      log, onStatus,
    });
    this.onKline = onKline; // (fritSymbol, interval, {openTime,open,high,low,close,volume,isClosed})
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
      out.push(`${b}@kline_30m`, `${b}@kline_4h`);
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

  onMessage(data) {
    let msg;
    try { msg = JSON.parse(String(data)); } catch { return; }
    const d = msg.data || msg;
    if (!d || d.e !== "kline" || !d.k) return; // ignore subscribe-acks etc.
    const k = d.k;
    const bsym = String(d.s || "").toUpperCase();
    const frit = Object.keys(BINANCE_WS_MAP).find(f => BINANCE_WS_MAP[f] === bsym);
    if (!frit) return;
    const interval = k.i === "4h" ? "4h" : "30m";
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
