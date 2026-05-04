export class FcsClient {
  constructor(apiKey, config = {}) {
    if (!apiKey) throw new Error("[FcsClient] FCS_API_KEY is required");
    this.apiKey = apiKey;
    this.baseUrl = "https://fcsapi.com/api-v3";
    this.cache = new Map();
    this.cacheTTL = config.cacheTTL || 5 * 60 * 1000;
    this.requestDelay = config.requestDelay || 1200;
    this.lastRequestTime = 0;
    this.maxSymbolsPerRequest = config.maxSymbolsPerRequest || 10;
    this.fcsToSystem = (s) => String(s).replace(/[/\s-]/g, "").toUpperCase();
    this.systemToFcs = (s) => {
      const map = { EURUSD: "EUR/USD", GBPUSD: "GBP/USD", USDJPY: "USD/JPY", AUDUSD: "AUD/USD", USDCHF: "USD/CHF", USDCAD: "USD/CAD", NZDUSD: "NZD/USD", EURGBP: "EUR/GBP", EURJPY: "EUR/JPY", GBPJPY: "GBP/JPY", XAUUSD: "XAU/USD", XAGUSD: "XAG/USD", BTCUSD: "BTC/USD", ETHUSD: "ETH/USD", SOLUSD: "SOL/USD", BNBUSD: "BNB/USD", XRPUSD: "XRP/USD", DOGEUSD: "DOGE/USD", ADAUSD: "ADA/USD", US500: "US500", US100: "US100", DXY: "DXY" };
      return map[String(s).toUpperCase()] || String(s).replace(/([A-Z]{3})([A-Z]{3})/, "$1/$2");
    };
  }
  async _request(endpoint, params = {}) {
    const cacheKey = `${endpoint}:${JSON.stringify(params)}`;
    const cached = this.cache.get(cacheKey);
    if (cached && Date.now() - cached.ts < this.cacheTTL) return cached.data;
    const now = Date.now();
    const wait = Math.max(0, this.requestDelay - (now - this.lastRequestTime));
    if (wait > 0) await new Promise(r => setTimeout(r, wait));
    this.lastRequestTime = Date.now();
    const url = new URL(`${this.baseUrl}${endpoint}`);
    url.searchParams.set("access_key", this.apiKey);
    for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
    const res = await fetch(url.toString());
    if (!res.ok) { const text = await res.text().catch(() => ""); throw new Error(`FCS API ${res.status}: ${text.slice(0, 200)}`); }
    const json = await res.json();
    if (json.status !== "ok" && !json.response) throw new Error(json.msg || "FCS API error");
    this.cache.set(cacheKey, { data: json, ts: Date.now() });
    return json;
  }
  _categorize(symbols) {
    const cats = { forex: [], crypto: [], metals: [], indices: [] };
    for (const s of symbols) {
      const sys = this.fcsToSystem(s);
      if (/^(BTC|ETH|SOL|BNB|XRP|DOGE|ADA)/.test(sys)) cats.crypto.push(s);
      else if (/^(XAU|XAG)/.test(sys)) cats.metals.push(s);
      else if (/^(US500|US100|DXY|SPX|NDX)/.test(sys)) cats.indices.push(s);
      else cats.forex.push(s);
    }
    return cats;
  }
  async getBulkQuotes(symbols) {
    const cats = this._categorize(symbols);
    const results = {};
    const endpoints = { forex: "/forex/latest", crypto: "/crypto/latest", metals: "/commodities/latest", indices: "/indices/latest" };
    for (const [cat, syms] of Object.entries(cats)) {
      if (!syms.length) continue;
      for (let i = 0; i < syms.length; i += this.maxSymbolsPerRequest) {
        const batch = syms.slice(i, i + this.maxSymbolsPerRequest);
        const fcsSyms = batch.map(s => this.systemToFcs(s)).join(",");
        try {
          const data = await this._request(endpoints[cat], { symbols: fcsSyms });
          const items = data.response || [];
          for (const item of items) {
            const sysSym = this.fcsToSystem(item.symbol || item.id || "");
            if (sysSym) results[sysSym] = { symbol: sysSym, price: parseFloat(item.price || item.close || 0), change24h: parseFloat(item.change_percent || item.chg_per || 0), timestamp: Date.now(), source: "fcs" };
          }
        } catch (e) { console.warn(`[FcsClient] Bulk fetch failed for ${cat} batch:`, e.message); }
      }
    }
    return results;
  }
  async getHistoricalCloses(symbols, days = 30) {
    const cats = this._categorize(symbols);
    const endpoints = { forex: "/forex/history", crypto: "/crypto/history", metals: "/commodities/history", indices: "/indices/history" };
    const closesMap = {};
    for (const [cat, syms] of Object.entries(cats)) {
      for (const sym of syms) {
        const fcsSym = this.systemToFcs(sym);
        try {
          const data = await this._request(endpoints[cat], { symbol: fcsSym, period: "1D", limit: days + 5 });
          const candles = data.response || [];
          closesMap[this.fcsToSystem(sym)] = candles.map(c => ({ time: new Date(c.t || c.time || c.date).getTime(), close: parseFloat(c.c || c.close) })).filter(c => !isNaN(c.close) && c.time > 0).sort((a, b) => a.time - b.time);
        } catch (e) { console.warn(`[FcsClient] History fetch failed for ${sym}:`, e.message); }
      }
    }
    return closesMap;
  }
  computeLogReturns(closesMap, minOverlap = 10) {
    const symbols = Object.keys(closesMap).filter(s => closesMap[s].length >= minOverlap);
    if (symbols.length < 3) return null;
    const timeSets = symbols.map(s => new Set(closesMap[s].map(c => c.time)));
    const commonTimes = [...timeSets[0]].filter(t => timeSets.every(set => set.has(t))).sort((a, b) => a - b);
    if (commonTimes.length < minOverlap) return null;
    const alignedCloses = symbols.map(sym => { const map = new Map(closesMap[sym].map(c => [c.time, c.close])); return commonTimes.map(t => map.get(t)); });
    const returnsMatrix = alignedCloses.map(closes => { const returns = []; for (let i = 1; i < closes.length; i++) { if (closes[i] > 0 && closes[i - 1] > 0) returns.push(Math.log(closes[i] / closes[i - 1])); else returns.push(0); } return returns; });
    return { symbols, commonTimes, returnsMatrix };
  }
  async getSystemicRiskData(symbols, days = 20) {
    const closesMap = await this.getHistoricalCloses(symbols, days);
    return this.computeLogReturns(closesMap, days - 5);
  }
  clearCache() { this.cache.clear(); }
  getCacheSize() { return this.cache.size; }
}
