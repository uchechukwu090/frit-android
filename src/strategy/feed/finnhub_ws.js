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

// US stock/ETF tickers stream under their own name (AAPL, SPY). Anything that
// looks like a 1-5 letter ticker (plus BRK.B style) and is NOT a mapped symbol
// resolves raw — this is what lets Alpaca-class symbols ride Finnhub when no
// Alpaca key is set, instead of going dark.
export function finnhubSymbol(fritSymbol) {
  const s = String(fritSymbol || "").toUpperCase();
  if (FINNHUB_MAP[s]) return FINNHUB_MAP[s];
  if (/^[A-Z]{1,5}(\.[A-Z])?$/.test(s)) return s;
  return null;
}

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
      .filter(s => finnhubSymbol(s)).slice(0, 45); // stay under the ~50 cap
    this.reverse = new Map(this.symbols.map(s => [finnhubSymbol(s), s]));
    if (this.status() === "live") this._subscribeAll();
  }

  onOpen() { this._subscribeAll(); }

  _subscribeAll() {
    for (const s of this.symbols) {
      this.send({ type: "subscribe", symbol: finnhubSymbol(s) });
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
  // Supports every bot timeframe (1m..1w); unknown strings fall back to 30m.
  async fetchCandles(fritSymbol, interval = "30m", count = 150) {
    const sym = String(fritSymbol || "").toUpperCase();
    const fh = finnhubSymbol(sym);
    if (!fh || !this.fetchImpl) return null;
    const RES = {
      "1m": [60, "1"], "5m": [300, "5"], "15m": [900, "15"], "30m": [1800, "30"],
      // Finnhub free tier has no 240 resolution — 4h is aggregated from 4x1h candles.
      "1h": [3600, "60"], "4h": [14400, "60"], "1d": [86400, "D"], "1w": [604800, "W"],
    };
    const [secs, resolution] = RES[interval] || RES["30m"];
    // 4h is aggregated from 4x1h: widen the window so `count` 4h candles are covered.
    const need = interval === "4h" ? count * 4 : count;
    const to = Math.floor(Date.now() / 1000);
    const from = to - secs * (need + 5);
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
      const hourly = d.c.map((c, i) => ({
        time: d.t[i] * 1000,
        open: d.o[i], high: d.h[i], low: d.l[i], close: c,
        volume: (d.v && d.v[i]) || 0,
      }));
      if (interval !== "4h") return hourly;
      // FIX (F-21): aggregate by TRUE 4h bucket boundaries — grouping "every
      // 4 consecutive rows" from array start produced misaligned candles
      // whenever `from` isn't 4h-aligned. Bucket keys match CandleStore's live
      // carving (Math.floor(ts / 14400e3)), so backfilled and live candles
      // share the same open times.
      const out = [];
      let bucket = null, buckKey = NaN;
      for (const c of hourly) {
        const key = Math.floor(c.time / 14400000);
        if (key !== buckKey) {
          if (bucket) out.push(bucket);
          buckKey = key;
          bucket = { time: key * 14400000, open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume || 0 };
        } else {
          bucket.high = Math.max(bucket.high, c.high);
          bucket.low = Math.min(bucket.low, c.low);
          bucket.close = c.close;
          bucket.volume += c.volume || 0;
        }
      }
      if (bucket) out.push(bucket);
      return out.length >= 10 ? out : null;
    } catch (e) {
      if (this.log) this.log(`[finnhub] candle fetch failed ${sym}: ${e.message}`);
      return null;
    }
  }
}
