// ============================================================================
// FinnhubFeed — primary rail for forex/metals/stocks (and crypto fallback).
//
// Free key, 1 WS connection per key, ~50 symbols: wss://ws.finnhub.io?token=.
// Subscribe format: {"type":"subscribe","symbol":"OANDA:EUR_USD"}.
// Forex symbols are OANDA-sourced (OANDA:XAU_USD works — only FXCM,
// Forex.com and FHFX are excluded from streaming).
// Ticks arrive as {type:"trade",data:[{s,p,t,v}]} and are aggregated into
// 30M/4H candles by CandleStore. Also exposes fetchCandles() over Finnhub's
// free candle REST (forex/candle, stock/candle, crypto/candle) so gap-fill
// doesn't have to spend Twelve Data credits.
// ============================================================================

// FRIT symbol -> Finnhub stream symbol.
export const FINNHUB_MAP = {
  XAUUSD: "OANDA:XAU_USD", XAGUSD: "OANDA:XAG_USD",
  EURUSD: "OANDA:EUR_USD", GBPUSD: "OANDA:GBP_USD", USDJPY: "OANDA:USD_JPY",
  AUDUSD: "OANDA:AUD_USD", USDCHF: "OANDA:USD_CHF", USDCAD: "OANDA:USD_CAD",
  NZDUSD: "OANDA:NZD_USD", GBPJPY: "OANDA:GBP_JPY", EURJPY: "OANDA:EUR_JPY",
  EURGBP: "OANDA:EUR_GBP",
  BTCUSD: "BINANCE:BTCUSDT", ETHUSD: "BINANCE:ETHUSDT", SOLUSD: "BINANCE:SOLUSDT",
  BNBUSD: "BINANCE:BNBUSDT", XRPUSD: "BINANCE:XRPUSDT", DOGEUSD: "BINANCE:DOGEUSDT",
  ADAUSD: "BINANCE:ADAUSDT",
};

import { WsBase } from "./ws_base.js";

export class FinnhubFeed extends WsBase {
  constructor({ apiKey, fetchImpl, onTick, log, onStatus }) {
    super({
      name: "finnhub",
      url: `wss://ws.finnhub.io?token=${apiKey}`,
      log, onStatus,
    });
    this.fetchImpl = fetchImpl; // node-fetch (injected — index.js owns it)
    this.apiKey = apiKey;
    this.onTick = onTick; // (fritSymbol, price, volume, tsMs)
    this.symbols = []; // FRIT symbols
    this.reverse = new Map(); // finnhub symbol -> FRIT
  }

  setSymbols(fritSymbols) {
    this.symbols = [...new Set(fritSymbols.map(s => String(s).toUpperCase()))]
      .filter(s => FINNHUB_MAP[s]).slice(0, 45); // stay under the ~50 cap
    this.reverse = new Map(this.symbols.map(s => [FINNHUB_MAP[s], s]));
    if (this.status() === "live") this._subscribeAll();
  }

  onOpen() { this._subscribeAll(); }

  _subscribeAll() {
    for (const s of this.symbols) {
      this.send({ type: "subscribe", symbol: FINNHUB_MAP[s] });
    }
    if (this.log) this.log(`[finnhub] subscribed ${this.symbols.join(",")}`);
  }

  onMessage(data) {
    let msg;
    try { msg = JSON.parse(String(data)); } catch { return; }
    if (msg.type !== "trade" || !Array.isArray(msg.data)) return;
    for (const t of msg.data) {
      const frit = this.reverse.get(t.s);
      if (!frit) continue;
      const px = Number(t.p);
      if (!Number.isFinite(px) || px <= 0) continue;
      try { this.onTick(frit, px, Number(t.v) || 0, Number(t.t) || Date.now()); } catch { /* ignore */ }
    }
  }

  // Free REST gap-fill: returns engine-shaped candles oldest->newest or null.
  async fetchCandles(fritSymbol, interval /* "30m" | "4h" */, count = 150) {
    const sym = String(fritSymbol || "").toUpperCase();
    const fh = FINNHUB_MAP[sym];
    if (!fh || !this.fetchImpl) return null;
    const secs = interval === "4h" ? 240 * 60 : 30 * 60;
    const resolution = interval === "4h" ? "240" : "30";
    const to = Math.floor(Date.now() / 1000);
    const from = to - secs * (count + 5);
    const [venue, raw] = fh.includes(":") ? fh.split(":") : ["", fh];
    let url;
    if (/^[A-Z]+$/.test(raw) && !raw.includes("_") && !raw.includes("USD")) {
      // US stock, e.g. AAPL
      url = `https://finnhub.io/api/v1/stock/candle?symbol=${encodeURIComponent(raw)}&resolution=${resolution}&from=${from}&to=${to}&token=${this.apiKey}`;
    } else if (venue === "OANDA") {
      url = `https://finnhub.io/api/v1/forex/candle?symbol=${encodeURIComponent(fh)}&resolution=${resolution}&from=${from}&to=${to}&token=${this.apiKey}`;
    } else {
      url = `https://finnhub.io/api/v1/crypto/candle?symbol=${encodeURIComponent(fh)}&resolution=${resolution}&from=${from}&to=${to}&token=${this.apiKey}`;
    }
    try {
      const res = await this.fetchImpl(url, { signal: AbortSignal.timeout(20_000) });
      const d = await res.json();
      if (d.s !== "ok" || !Array.isArray(d.c) || d.c.length < 10) return null;
      return d.c.map((c, i) => ({
        time: d.t[i] * 1000,
        open: d.o[i], high: d.h[i], low: d.l[i], close: c,
        volume: (d.v && d.v[i]) || 0,
      }));
    } catch (e) {
      if (this.log) this.log(`[finnhub] candle fetch failed ${sym}: ${e.message}`);
      return null;
    }
  }
}
