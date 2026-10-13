// ============================================================================
// ALPACA EXECUTOR — stocks + crypto spot, paper default, live locked.
// ----------------------------------------------------------------------------
// SAFETY DESIGN (same pattern as Deriv):
// - PAPER default: ALPACA_PAPER_KEY/SECRET active. Paper is virtual money,
//   free for anyone (email signup) — safe to run.
// - LIVE needs BOTH ALPACA_LIVE_KEY/SECRET and ALPACA_LIVE=1. Otherwise a live
//   key is never touched.
// - Market orders only, fixed USD notional per trade (default $50 paper).
//   No margin horror: cash account behavior on paper; live users must know
//   their own account type.
// - Fills register into the shared positions monitor with source "alpaca" /
//   "alpaca-live" so the unified paper-portfolio shows them. Alpaca spot has
//   no native SL/TP — engine SL/TP are stored as metadata and resolved via
//   the existing /positions/resolve path (auto-close on SL/TP is future work).
// Endpoints: paper https://paper-api.alpaca.markets, live https://api.alpaca.markets
// Headers: APCA-API-KEY-ID / APCA-API-SECRET-KEY.
// ============================================================================

const PAPER_BASE = "https://paper-api.alpaca.markets";
const LIVE_BASE = "https://api.alpaca.markets";

// Our symbol -> Alpaca symbol (crypto already matches, e.g. BTCUSD).
export function toAlpacaSymbol(sym) {
  const s = String(sym || "").toUpperCase().replace("/", "");
  return s;
}

export function isStockSymbol(sym, { cryptoSet = new Set(), fxHint = null } = {}) {
  const s = String(sym || "").toUpperCase();
  if (!/^[A-Z]{1,5}$/.test(s)) return false;
  if (cryptoSet.has(s)) return false;
  if (/^(XAU|XAG|XPT|XPD)/.test(s)) return false;
  if (fxHint && fxHint.has(s)) return false;
  return true;
}

export class AlpacaClient {
  constructor({ keyId, secret, paper = true, fetchImpl, log = null }) {
    this.keyId = keyId || "";
    this.secret = secret || "";
    this.paper = paper !== false;
    this.fetch = fetchImpl;
    this.log = log || (() => {});
    this._status = "idle";
  }

  get base() { return this.paper ? PAPER_BASE : LIVE_BASE; }
  get configured() { return !!(this.keyId && this.secret); }

  async _req(method, path, body = null, timeoutMs = 20_000) {
    if (!this.configured) return { error: "no keys" };
    try {
      const res = await this.fetch(`${this.base}${path}`, {
        method,
        headers: {
          "APCA-API-KEY-ID": this.keyId,
          "APCA-API-SECRET-KEY": this.secret,
          "Content-Type": "application/json",
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(timeoutMs),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) return { error: data?.message || `HTTP ${res.status}` };
      return { data };
    } catch (e) {
      return { error: e.message };
    }
  }

  async account() {
    const r = await this._req("GET", "/v2/account");
    return r.error ? null : r.data;
  }

  // Market order by USD notional (fractional-capable for stocks).
  async marketOrder({ symbol, side, notional, timeInForce = null }) {
    const sym = toAlpacaSymbol(symbol);
    const isCrypto = /USD$/.test(sym) && sym.length > 5;
    const tif = timeInForce || (isCrypto ? "gtc" : "day");
    return this._req("POST", "/v2/orders", {
      symbol: sym,
      notional: String(notional),
      side: side === "buy" ? "buy" : "sell",
      type: "market",
      time_in_force: tif,
    });
  }

  /**
   * Bracket order: market entry + take-profit limit + stop-loss stop, one
   * call. SL/TP ride natively at the broker — the position is never
   * unprotected between our monitor ticks. qty in shares (fractional ok).
   */
  async bracketOrder({ symbol, side, qty, takeProfit, stopLoss }) {
    const sym = toAlpacaSymbol(symbol);
    const q = Number(qty);
    if (!(q > 0)) return { error: "qty must be positive" };
    if (!(takeProfit > 0) || !(stopLoss > 0)) return { error: "bracket needs takeProfit + stopLoss prices" };
    return this._req("POST", "/v2/orders", {
      symbol: sym,
      qty: String(q),
      side: side === "buy" ? "buy" : "sell",
      type: "market",
      time_in_force: "day",
      order_class: "bracket",
      take_profit: { limit_price: String(takeProfit) },
      stop_loss: { stop_price: String(stopLoss) },
    });
  }

  async getOrder(orderId) {
    return this._req("GET", `/v2/orders/${orderId}`);
  }

  /** Market clock — authority for whether stocks may be submitted. */
  async clock() {
    const r = await this._req("GET", "/v2/clock");
    if (r.error || !r.data) return null;
    return {
      isOpen: r.data.is_open === true,
      nextOpen: r.data.next_open || null,
      nextClose: r.data.next_close || null,
      timestamp: r.data.timestamp || null,
    };
  }

  static barraTimeframe(interval) {
    return { "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min", "1H": "1Hour", "4H": "4Hour", "1D": "1Day", "1W": "1Week" }[interval] || "30Min";
  }

  /** OHLC bars oldest->newest in engine shape. Free plan = IEX, ~15min delayed. */
  async bars(symbol, interval = "30m", limit = 150) {
    // NOTE: market data lives on data.alpaca.markets, not the trade base —
    // _req() would prefix the wrong host, so this issues raw fetch directly.
    if (!this.configured) return { error: "no keys" };
    const sym = toAlpacaSymbol(symbol);
    const tf = AlpacaClient.barraTimeframe(interval);
    try {
      const res = await this.fetch(`https://data.alpaca.markets/v2/stocks/${encodeURIComponent(sym)}/bars?timeframe=${tf}&limit=${Math.min(limit, 500)}&adjustment=all&feed=iex`, {
        headers: { "APCA-API-KEY-ID": this.keyId, "APCA-API-SECRET-KEY": this.secret },
        signal: AbortSignal.timeout(20_000),
      });
      const d = await res.json().catch(() => ({}));
      if (!res.ok || !Array.isArray(d.bars) || d.bars.length < 5) {
        return { error: d?.message || `no bars for ${symbol}` };
      }
      return {
        candles: d.bars.map((b) => ({
          time: new Date(b.t).getTime(),
          open: Number(b.o), high: Number(b.h), low: Number(b.l),
          close: Number(b.c), volume: Number(b.v) || 0,
        })).filter((c) => c.close > 0),
        feed: "iex", delayed: true,
      };
    } catch (e) {
      return { error: e.message };
    }
  }

  /** Latest quote (delayed on free plan). */
  async quote(symbol) {
    if (!this.configured) return { error: "no keys" };
    try {
      const sym = toAlpacaSymbol(symbol);
      const res = await this.fetch(`https://data.alpaca.markets/v2/stocks/${encodeURIComponent(sym)}/quotes/latest?feed=iex`, {
        headers: { "APCA-API-KEY-ID": this.keyId, "APCA-API-SECRET-KEY": this.secret },
        signal: AbortSignal.timeout(20_000),
      });
      const d = await res.json().catch(() => ({}));
      if (!res.ok || !d.quote) return { error: d?.message || `no quote for ${symbol}` };
      return {
        bid: Number(d.quote.bp), ask: Number(d.quote.ap),
        bidSize: Number(d.quote.bs) || 0, askSize: Number(d.quote.as) || 0,
        delayed: true,
      };
    } catch (e) {
      return { error: e.message };
    }
  }

  async positions() {
    const r = await this._req("GET", "/v2/positions");
    return r.error ? [] : (r.data || []);
  }

  async closePosition(symbol, qty = null) {
    const q = qty ? `?qty=${encodeURIComponent(qty)}` : "";
    return this._req("DELETE", `/v2/positions/${toAlpacaSymbol(symbol)}${q}`);
  }
}

export class AlpacaExecutor {
  constructor({ db, paperClient, liveClient, liveAllowed = false, notional = 50, maxPerDay = 20, log = null }) {
    this.db = db;
    this.paper = paperClient;
    this.live = liveClient;
    this.liveAllowed = !!liveAllowed;
    this.notional = Math.min(Math.max(Number(notional) || 50, 1), 5000);
    this.maxPerDay = maxPerDay;
    this.log = log || (() => {});
    db.exec(`
      CREATE TABLE IF NOT EXISTS alpaca_orders (
        order_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        notional REAL NOT NULL,
        fill_price REAL,
        status TEXT DEFAULT 'OPEN',
        mode TEXT DEFAULT 'paper',
        placed_at INTEGER NOT NULL
      );
    `);
  }

  get active() { return this.liveAllowed && this.live.configured ? this.live : this.paper; }
  get mode() { return this.liveAllowed && this.live.configured ? "live" : "paper"; }

  tradesToday() {
    try {
      return this.db.prepare("SELECT COUNT(*) AS c FROM alpaca_orders WHERE placed_at > ?")
        .get(Date.now() - 24 * 3600 * 1000)?.c || 0;
    } catch { return 0; }
  }

  async place({ symbol, action, positionId = null, qty = null, takeProfit = null, stopLoss = null }) {
    const client = this.active;
    if (!client.configured) return { ok: false, why: "no keys" };
    if (this.tradesToday() >= this.maxPerDay) return { ok: false, why: `daily cap (${this.maxPerDay})` };
    const side = String(action).toLowerCase() === "sell" ? "sell" : "buy";
    const sym = toAlpacaSymbol(symbol);
    const isCrypto = /USD$/.test(sym) && sym.length > 5;
    // Market-hours gate (stocks only — crypto trades 24/7). After-hours fills
    // are thin and wide; refuse instead of queuing a day order into the open.
    if (!isCrypto) {
      try {
        const clock = await client.clock();
        if (clock && !clock.isOpen) {
          return { ok: false, why: `market closed (next open ${clock.nextOpen || "unknown"})` };
        }
      } catch { /* clock unreadable — proceed, broker is the final authority */ }
    }
    // With engine levels we place a protected bracket; without them (legacy
    // agent-driven spot buys) the fixed-notional market order stands.
    const useBracket = qty != null && takeProfit != null && stopLoss != null;
    const r = useBracket
      ? await client.bracketOrder({ symbol, side, qty, takeProfit, stopLoss })
      : await client.marketOrder({ symbol, side, notional: this.notional });
    if (r.error) {
      this.log(`[alpaca] order failed ${symbol}: ${r.error}`);
      return { ok: false, why: r.error };
    }
    const o = r.data || {};
    try {
      this.db.prepare(`
        INSERT OR REPLACE INTO alpaca_orders (order_id, symbol, side, notional, fill_price, status, mode, placed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `).run(String(o.id), String(symbol).toUpperCase(), side, useBracket ? Number(qty) || 0 : this.notional,
        o.filled_avg_price != null ? Number(o.filled_avg_price) : null,
        o.status || "new", this.mode, Date.now());
    } catch (e) { this.log(`[alpaca] ledger: ${e.message}`); }
    this.log(`[alpaca] ${this.mode} ${useBracket ? "bracket" : "market"} ${side} ${symbol} id=${o.id} status=${o.status}`);
    return { ok: true, mode: this.mode, order: o, positionId, bracket: useBracket };
  }

  // Refresh order fills; resolve linked monitor positions on full fills? No —
  // spot positions stay OPEN (no expiry). This only syncs fill prices.
  async syncFills() {
    let rows = [];
    try {
      rows = this.db.prepare("SELECT * FROM alpaca_orders WHERE fill_price IS NULL").all();
    } catch { return { synced: 0 }; }
    let synced = 0;
    for (const r of rows) {
      try {
        const client = r.mode === "live" ? this.live : this.paper;
        const o = await client.getOrder(r.order_id);
        if (!o.error && o.data?.filled_avg_price != null) {
          this.db.prepare("UPDATE alpaca_orders SET fill_price = ?, status = ? WHERE order_id = ?")
            .run(Number(o.data.filled_avg_price), o.data.status || r.status, r.order_id);
          synced++;
        }
      } catch { /* per-order best effort */ }
    }
    return { synced };
  }

  async status() {    const client = this.active;
    const acct = client.configured ? await client.account().catch(() => null) : null;
    let openDb = 0;
    try { openDb = this.db.prepare("SELECT COUNT(*) AS c FROM alpaca_orders").get()?.c || 0; } catch { /* ignore */ }
    return {
      configured: client.configured, mode: this.mode,
      liveAllowed: this.liveAllowed, notional: this.notional,
      maxPerDay: this.maxPerDay, tradesToday: this.tradesToday(),
      ordersTracked: openDb,
      buyingPower: acct?.buying_power ?? null, equity: acct?.equity ?? null,
      currency: acct?.currency ?? null,
    };
  }
}

/**
 * Share qty for a dollar risk over a stop distance. Rounds DOWN to 6dp —
 * never up (rounding up would exceed the risk budget). Null when the
 * notional would fall under ~$1 (Alpaca fractional minimum) or on bad input.
 */
export function qtyForRisk({ price, stopDist, riskUsd }) {
  const px = Number(price), dist = Number(stopDist), risk = Number(riskUsd);
  if (!(px > 0) || !(dist > 0) || !(risk > 0)) return null;
  const qty = Math.floor((risk / dist) * 1e6) / 1e6;
  if (!(qty > 0) || qty * px < 1) return null;
  return qty;
}
