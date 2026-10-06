// ============================================================================
// DukascopyFeed — keyless backup rail for forex/metals (Swiss bank free API).
//
// Official free endpoints (no key, no signup):
//   instrumentList: ?path=api/instrumentList            -> [{id,name,...}]
//   currentPrices:  ?path=api/currentPrices&instruments=  -> quotes for ids
// Role: opportunistic backup only. LiveFeed prefers Finnhub WS for
// forex/metals and only polls here when Finnhub has no key or a symbol isn't
// streaming. WARNING (verified Oct 2026): freeserv.dukascopy.com returns
// HTTP 429 "Bot blocked" to many datacenter/hosting IPs — it often works
// from residential IPs but NOT from Render/VPS hosts. Symbols with no live
// provider at all fall back to budgeted Twelve Data REST refresh instead.
// Poll every 15s; ticks aggregate into 30M/4H candles in CandleStore.
// Parsing is defensive (field names have drifted before) — first successful
// payload shape is logged, unparseable payloads disable the feed gracefully.
// ============================================================================

const BASE = "https://freeserv.dukascopy.com/2.0/?path=api/";

// FRIT symbol -> Dukascopy instrument display name.
const NAME_MAP = {
  EURUSD: "EUR/USD", GBPUSD: "GBP/USD", USDJPY: "USD/JPY", AUDUSD: "AUD/USD",
  USDCHF: "USD/CHF", USDCAD: "USD/CAD", NZDUSD: "NZD/USD", GBPJPY: "GBP/JPY",
  EURJPY: "EUR/JPY", EURGBP: "EUR/GBP", XAUUSD: "XAU/USD", XAGUSD: "XAG/USD",
};

function pickNum(o, keys) {
  for (const k of keys) {
    const v = Number(o[k]);
    if (Number.isFinite(v) && v > 0) return v;
  }
  return NaN;
}

export class DukascopyFeed {
  constructor({ fetchImpl, onTick, log, onStatus, pollMs = 15_000 }) {
    this.fetch = fetchImpl;
    this.onTick = onTick;
    this.log = log || (() => {});
    this.onStatus = onStatus || (() => {});
    this.pollMs = pollMs;
    this.symbols = [];
    this.idBySymbol = new Map(); // FRIT -> dukascopy id
    this.timer = null;
    this.running = false;
    this._status = "idle";
    this.loggedShape = false;
  }

  setStatus(s, d = "") {
    this._status = s;
    try { this.onStatus("dukascopy", s, d); } catch { /* ignore */ }
  }
  status() { return this._status; }

  setSymbols(fritSymbols) {
    this.symbols = [...new Set(fritSymbols.map(s => String(s).toUpperCase()))]
      .filter(s => NAME_MAP[s]);
  }

  async start() {
    if (this.running || !this.symbols.length) return;
    this.running = true;
    this.setStatus("connecting", "resolving instrument ids");
    const ok = await this._resolveIds();
    if (!ok) {
      this.setStatus("down", "instrumentList unusable");
      this.running = false;
      return;
    }
    this.setStatus("live", `polling ${this.symbols.join(",")}`);
    this._loop();
  }

  stop() {
    this.running = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    this.setStatus("stopped");
  }

  async _resolveIds() {
    try {
      const res = await this.fetch(`${BASE}instrumentList`, { signal: AbortSignal.timeout(20_000) });
      const list = await res.json();
      if (!Array.isArray(list) || !list.length) return false;
      const byName = new Map();
      for (const it of list) {
        const nm = String(it.name || it.nameLong || "");
        if (it.id != null && nm) byName.set(nm.toUpperCase().replace(/\s+/g, ""), it.id);
      }
      for (const s of this.symbols) {
        const want = NAME_MAP[s].toUpperCase().replace(/\s+/g, "");
        const id = byName.get(want);
        if (id != null) this.idBySymbol.set(s, id);
      }
      return this.idBySymbol.size > 0;
    } catch (e) {
      this.log(`[dukascopy] instrumentList failed: ${e.message}`);
      return false;
    }
  }

  async _loop() {
    if (!this.running) return;
    try { await this._poll(); } catch (e) { this.log(`[dukascopy] poll failed: ${e.message}`); }
    if (this.running) {
      this.timer = setTimeout(() => this._loop(), this.pollMs);
      if (this.timer.unref) this.timer.unref();
    }
  }

  async _poll() {
    const ids = [...this.idBySymbol.values()].join(",");
    if (!ids) return;
    const res = await this.fetch(`${BASE}currentPrices&instruments=${encodeURIComponent(ids)}`, {
      signal: AbortSignal.timeout(20_000),
    });
    const arr = await res.json();
    if (!Array.isArray(arr)) return;
    if (!this.loggedShape && arr.length) {
      this.log(`[dukascopy] payload shape: ${JSON.stringify(arr[0]).slice(0, 300)}`);
      this.loggedShape = true;
    }
    const idToSym = new Map([...this.idBySymbol.entries()].map(([s, id]) => [String(id), s]));
    const now = Date.now();
    for (const q of arr) {
      const sym = idToSym.get(String(q.instrument ?? q.id ?? q.instrumentId));
      if (!sym) continue;
      const px = pickNum(q, ["bestBid", "bid", "priceBid", "closeBid", "bestAsk", "ask", "priceAsk"]);
      if (!Number.isFinite(px)) continue;
      const ts = Number(q.time || q.timestamp || now);
      try { this.onTick(sym, px, 0, Number.isFinite(ts) ? ts : now); } catch { /* ignore */ }
    }
  }
}
