// ============================================================================
// DERIV EXECUTOR — free API, Rise/Fall contracts for pocket signals.
// ----------------------------------------------------------------------------
// SAFETY DESIGN (read before touching):
// - DEMO-FIRST: works with a demo token (VRTC* account, virtual $10k, free).
//   A real-money token is REFUSED unless DERIV_LIVE=1 is explicitly set —
//   paper stats must earn live trading first.
// - Every fill registers into the shared positions monitor with source
//   "pocket-deriv" and resolves on BROKER outcome (proposal_open_contract),
//   so the unified paper-portfolio shows it separately from pure paper.
// - Daily trade cap (default 20) + single-stake cap. No martingale, no
//   averaging — one signal, one contract, fixed stake.
// - Deriv symbols: forex/metals are frx-prefixed (frxXAUUSD, frxEURUSD...),
//   crypto spot-like (BTCUSD...). Verified against active_symbols at boot;
//   unmapped/unsupported symbols are skipped with a logged reason.
// Docs: https://developers.deriv.com (authorize/proposal/buy/
// proposal_open_contract). Uses the `ws` package like the feed providers.
// ============================================================================

import WebSocket from "ws";

const WS_URL = "wss://ws.derivws.com/websockets/v3";

// Our symbol -> Deriv symbol candidates (first live one wins).
// Synthetics trade under their own codes (R_100, 1HZ100V, ...) — no mapping
// needed, just verification that MULT contracts exist for them.
export const DERIV_MAP = {
  XAUUSD: ["frxXAUUSD"], XAGUSD: ["frxXAGUSD"],
  EURUSD: ["frxEURUSD"], GBPUSD: ["frxGBPUSD"], USDJPY: ["frxUSDJPY"],
  AUDUSD: ["frxAUDUSD"], USDCHF: ["frxUSDCHF"], USDCAD: ["frxUSDCAD"],
  NZDUSD: ["frxNZDUSD"], GBPJPY: ["frxGBPJPY"], EURJPY: ["frxEURJPY"],
  EURGBP: ["frxEURGBP"],
  BTCUSD: ["BTCUSD", "cryBTCUSD"], ETHUSD: ["ETHUSD", "cryETHUSD"],
  SOLUSD: ["SOLUSD"], BNBUSD: ["BNBUSD"], XRPUSD: ["XRPUSD"],
  DOGEUSD: ["DOGEUSD"], ADAUSD: ["ADAUSD"],
};

export function buildProposal(symbol, direction, stake, durationMin, currency = "USD") {
  return {
    proposal: 1,
    amount: stake,
    basis: "stake",
    contract_type: direction === "CALL" ? "CALL" : "PUT",
    currency,
    duration: durationMin,
    duration_unit: "m",
    symbol,
  };
}

export class DerivClient {
  constructor({ token, appId = "1089", log = null, onStatus = null, wsUrl = null, WebSocketImpl = null } = {}) {
    this.token = token;
    this.appId = String(appId || "1089");
    this.wsBase = wsUrl || WS_URL; // injectable for tests; default is production
    this.WSImpl = WebSocketImpl || WebSocket;
    this.log = log || (() => {});
    this.onStatus = onStatus || (() => {});
    this.ws = null;
    this.reqId = 0;
    this.pending = new Map();
    this.ready = false;
    this.account = null; // { loginid, isDemo }
    this._status = "idle";
    this.stopping = false;
  }

  setStatus(s, d = "") {
    this._status = s;
    try { this.onStatus(s, d); } catch { /* ignore */ }
  }
  status() { return this._status; }

  async connect() {
    if (this.ready) return true;
    // No token = anonymous data-only mode (ticks_history/candles need no
    // auth). Trading methods still refuse without a token further down.
    const anonymous = !this.token;
    return new Promise((resolve) => {
      this.setStatus("connecting");
      let ws;
      try {
        ws = new this.WSImpl(`${this.wsBase}?app_id=${encodeURIComponent(this.appId)}`);
      } catch (e) {
        this.setStatus("down", e.message);
        return resolve(false);
      }
      this.ws = ws;
      const fail = (why) => {
        this.setStatus("down", why);
        try { ws.close(); } catch { /* ignore */ }
        this.ws = null; this.ready = false;
        resolve(false);
      };
      const timer = setTimeout(() => fail("connect timeout"), 20_000);
      if (timer.unref) timer.unref();
      ws.on("open", async () => {
        try {
          if (anonymous) {
            this.account = { loginid: "anonymous", isDemo: true };
            this.ready = true;
            clearTimeout(timer);
            this.setStatus("live", "anonymous data-only");
            resolve(true);
            return;
          }
          const r = await this._call({ authorize: this.token }, 15_000);
          if (r.error) return (clearTimeout(timer), fail(`authorize: ${r.error.message || r.error.code}`));
          const loginid = r.authorize?.loginid || "";
          // is_virtual is authoritative; VRTC-loginid regex kept as fallback
          // (some token types omit the flag).
          const flag = r.authorize?.is_virtual;
          this.account = { loginid, isDemo: flag === 1 || flag === true || /VRTC/i.test(loginid) };
          this.ready = true;
          clearTimeout(timer);
          this.setStatus("live", `${loginid} (${this.account.isDemo ? "DEMO" : "REAL"})`);
          resolve(true);
        } catch (e) { clearTimeout(timer); fail(e.message); }
      });
      ws.on("message", (data) => {
        let msg;
        try { msg = JSON.parse(String(data)); } catch { return; }
        const id = msg.req_id;
        if (id != null && this.pending.has(id)) {
          const { resolve } = this.pending.get(id);
          this.pending.delete(id);
          resolve(msg);
        }
      });
      ws.on("close", () => {
        this.ready = false;
        if (!this.stopping) this.setStatus("down", "socket closed");
        for (const [, p] of this.pending) { try { p.resolve({ error: { message: "socket closed" } }); } catch { /* ignore */ } }
        this.pending.clear();
      });
      ws.on("error", () => { /* close follows */ });
    });
  }

  _call(obj, timeoutMs = 20_000) {
    return new Promise((resolve) => {
      if (!this.ws || this.ws.readyState !== this.WSImpl.OPEN) {
        return resolve({ error: { message: "not connected" } });
      }
      const id = ++this.reqId;
      this.pending.set(id, { resolve });
      try { this.ws.send(JSON.stringify({ ...obj, req_id: id })); } catch {
        this.pending.delete(id);
        return resolve({ error: { message: "send failed" } });
      }
      const t = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          resolve({ error: { message: "timeout" } });
        }
      }, timeoutMs);
      if (t.unref) t.unref();
    });
  }

  async call(obj, timeoutMs) {
    if (!this.ready) {
      const ok = await this.connect();
      if (!ok) return { error: { message: "deriv unavailable" } };
    }
    const r = await this._call(obj, timeoutMs);
    if (r?.error?.message === "not connected" || r?.error?.message === "socket closed") {
      this.ready = false;
      const ok = await this.connect();
      if (!ok) return r;
      return this._call(obj, timeoutMs);
    }
    return r;
  }

  async activeSymbols() {
    const r = await this.call({ active_symbols: "brief", product_type: "basic" });
    if (r.error || !Array.isArray(r.active_symbols)) return [];
    return r.active_symbols.map(s => s.symbol);
  }

  async contractsFor(symbol) {
    const r = await this.call({ contracts_for: symbol });
    if (r.error || !r.contracts_for) return [];
    return r.contracts_for.available || [];
  }

  // ---- market data (works anonymous — no token needed) ----
  static granularity(interval) {
    return { "1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800 }[interval] || 1800;
  }

  /** OHLC candles oldest->newest in engine shape {time,open,high,low,close,volume}. */
  async candles(symbol, interval = "30m", count = 150) {
    const r = await this.call({
      ticks_history: symbol,
      style: "candles",
      granularity: DerivClient.granularity(interval),
      count: Math.min(Math.max(count, 10), 5000),
      end: "latest",
    });
    if (r.error || !Array.isArray(r.candles) || r.candles.length < 5) {
      throw new Error(`deriv candles ${symbol}: ${r.error?.message || "empty"}`);
    }
    return r.candles.map((c) => ({
      time: Number(c.epoch) * 1000,
      open: Number(c.open), high: Number(c.high),
      low: Number(c.low), close: Number(c.close),
      volume: 0,
    })).filter((c) => c.close > 0);
  }

  /** Latest tick price. */
  async tick(symbol) {
    const r = await this.call({ ticks_history: symbol, count: 2, end: "latest", style: "ticks" });
    const prices = r.prices || r.history?.prices;
    if (r.error || !Array.isArray(prices) || !prices.length) {
      throw new Error(`deriv tick ${symbol}: ${r.error?.message || "empty"}`);
    }
    return Number(prices[prices.length - 1]);
  }

  // ---- multipliers (MULTUP/MULTDN): spot-direction contracts with stake =
  // max loss, native take_profit/stop_loss, no expiry. This is the contract
  // family the MTF/longterm engines' BUY/SELL + SL/TP map onto 1:1.
  async proposeMultiplier({ symbol, direction, stake, multiplier = 100, takeProfit = null, stopLoss = null }) {
    const req = {
      proposal: 1,
      contract_type: direction === "sell" ? "MULTDN" : "MULTUP",
      symbol, stake, multiplier,
    };
    if (takeProfit != null) req.take_profit = takeProfit;
    if (stopLoss != null) req.stop_loss = stopLoss;
    const r = await this.call(req);
    const p = r.proposal;
    if (r.error || !p?.id || !(p.ask_price > 0)) {
      throw new Error(`deriv proposal ${symbol}: ${r.error?.message || "no ask"}`);
    }
    return { id: p.id, askPrice: Number(p.ask_price), spot: Number(p.spot) || null };
  }

  /** Buy a multiplier. Callers (DerivExecutor.execute) enforce the demo/live
   *  gate beforehand; the token check here is the last line of defense. */
  async buyMultiplier({ symbol, direction, stake, multiplier, takeProfit = null, stopLoss = null }) {
    if (!this.token) throw new Error("no token — paper mode");
    const prop = await this.proposeMultiplier({ symbol, direction, stake, multiplier, takeProfit, stopLoss });
    if (prop.askPrice > stake * 1.02) {
      throw new Error(`deriv ask ${prop.askPrice} exceeds stake ${stake} — refusing overpriced fill`);
    }
    const r = await this.call({ buy: prop.id, price: stake });
    if (r.error || !r.buy?.contract_id) {
      throw new Error(`deriv buy ${symbol}: ${r.error?.message || "no contract_id"}`);
    }
    return {
      contract_id: String(r.buy.contract_id),
      transaction_id: r.buy.transaction_id != null ? String(r.buy.transaction_id) : null,
      buy_price: Number(r.buy.buy_price) || stake,
      longcode: r.buy.longcode || "",
    };
  }

  /** Early-exit an open contract. */
  async sellContract(contractId) {
    if (!this.token) throw new Error("no token — paper mode");
    const r = await this.call({ sell: String(contractId), price: 0 });
    if (r.error || !r.sell) throw new Error(`deriv sell ${contractId}: ${r.error?.message || "failed"}`);
    return {
      soldFor: Number(r.sell.sold_for) || 0,
      profit: Number(r.sell.profit) || 0,
      transactionId: r.sell.transaction_id != null ? String(r.sell.transaction_id) : null,
    };
  }

  async openContracts() {
    const r = await this.call({ portfolio: 1 });
    if (r.error || !r.portfolio) return [];
    return r.portfolio.contracts || [];
  }

  async balance() {
    const r = await this.call({ balance: 1 });
    if (r.error || !r.balance) return null;
    return { balance: Number(r.balance.balance), currency: r.balance.currency };
  }

  // Full Rise/Fall purchase: proposal -> buy. Returns {contract_id,...} or {error}.
  async buyRiseFall({ symbol, direction, stake, durationMin, currency = "USD" }) {
    const prop = await this.call(buildProposal(symbol, direction, stake, durationMin, currency));
    if (prop.error) return { error: prop.error.message || prop.error.code || "proposal failed" };
    const p = prop.proposal;
    if (!p?.id) return { error: "no proposal id" };
    const buy = await this.call({ buy: p.id, price: p.ask_price });
    if (buy.error) return { error: buy.error.message || buy.error.code || "buy failed" };
    if (!buy.buy?.contract_id) return { error: "no contract id" };
    return {
      contract_id: buy.buy.contract_id,
      transaction_id: buy.buy.transaction_id || null,
      ask_price: p.ask_price, payout: p.payout,
      entry_tick: buy.buy.entry_tick ?? p.spot ?? null,
    };
  }

  async contractStatus(contractId) {
    const r = await this.call({ proposal_open_contract: 1, contract_id: contractId }, 15_000);
    if (r.error || !r.proposal_open_contract) return null;
    const c = r.proposal_open_contract;
    const done = c.is_sold === 1 || (c.status && c.status !== "open");
    return {
      done,
      status: c.status || (c.is_sold ? "sold" : "open"),
      profit: c.profit != null ? Number(c.profit) : null,
      entry: c.entry_tick ?? c.buy_price ?? null,
      exit: c.exit_tick ?? c.sell_price ?? null,
    };
  }

  close() {
    this.stopping = true;
    try { this.ws?.close(); } catch { /* ignore */ }
    this.ws = null; this.ready = false;
    this.setStatus("stopped");
  }
}

// ---- execution policy wrapper (caps + mapping + bookkeeping) ----
export class DerivExecutor {
  constructor({ db, client, liveAllowed = false, stake = 1.0, maxPerDay = 20, currency = "USD", log = null }) {
    this.db = db;
    this.client = client;
    this.liveAllowed = !!liveAllowed;
    this.stake = Math.min(Math.max(Number(stake) || 1, 0.35), 50);
    this.maxPerDay = maxPerDay;
    this.currency = currency;
    this.log = log || (() => {});
    this.symbolCache = new Map(); // ours -> deriv symbol (verified)
    db.exec(`
      CREATE TABLE IF NOT EXISTS deriv_contracts (
        contract_id TEXT PRIMARY KEY,
        position_id TEXT,
        symbol TEXT NOT NULL,
        deriv_symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        stake REAL NOT NULL,
        duration_min INTEGER NOT NULL,
        entry_tick REAL,
        status TEXT DEFAULT 'OPEN',
        profit REAL,
        mode TEXT DEFAULT 'demo',
        placed_at INTEGER NOT NULL
      );
    `);
    // Migration: ledgers created before profit tracking existed.
    try {
      const cols = db.prepare("PRAGMA table_info(deriv_contracts)").all().map(c => c.name);
      if (!cols.includes("profit")) db.exec("ALTER TABLE deriv_contracts ADD COLUMN profit REAL");
    } catch { /* best-effort */ }
  }

  tradesToday() {
    try {
      return this.db.prepare(
        "SELECT COUNT(*) AS c FROM deriv_contracts WHERE placed_at > ?"
      ).get(Date.now() - 24 * 3600 * 1000)?.c || 0;
    } catch { return 0; }
  }

  async resolveSymbol(ours) {    const sym = String(ours).toUpperCase();
    if (this.symbolCache.has(sym)) return this.symbolCache.get(sym);
    // Synthetics are their own Deriv codes — verify MULT availability directly.
    if (/^(R_\d+|1HZ\d+V?|BOOM\d+N|CRASH\d+N|STPRNG\d*|RDBULL\d*|RDBEAR\d*|JD\d+|GDA\d+)$/.test(sym)) {
      try {
        const avail = await this.client.contractsFor(sym);
        const types = new Set((avail || []).flatMap(a => [a.contract_type, ...(a.supported_contract_types || [])].filter(Boolean)));
        if (types.has("MULTUP") && types.has("MULTDN")) {
          this.symbolCache.set(sym, sym);
          return sym;
        }
        return null;
      } catch { return null; }
    }
    const cands = DERIV_MAP[sym] || [];
    // Verify against the broker's own list + Rise/Fall availability.
    let live = [];
    try { live = await this.client.activeSymbols(); } catch { live = []; }
    const liveSet = new Set(live);
    for (const d of cands) {
      if (live.length && !liveSet.has(d)) continue;
      try {
        const avail = await this.client.contractsFor(d);
        const types = new Set((avail || []).flatMap(a => [a.contract_type, ...(a.supported_contract_types || [])].filter(Boolean)));
        if (types.has("CALL") && types.has("PUT")) {
          this.symbolCache.set(sym, d);
          return d;
        }
      } catch { /* try next */ }
    }
    // No verification possible (empty list = API hiccup): trust first candidate.
    if (!live.length && cands.length) {
      this.symbolCache.set(sym, cands[0]);
      return cands[0];
    }
    return null;
  }

  /**
   * MULT-capable Deriv code for any engine symbol: synthetics map to
   * themselves, everything else goes through DERIV_MAP. Availability is
   * VERIFIED via contracts_for (MULTUP+MULTDN) — never assumed, so a market
   * Deriv doesn't list can never be sent there. Returns null when unsupported.
   */
  async resolveMultSymbol(ours) {
    const sym = String(ours).toUpperCase();
    if (this.symbolCache.has(`MULT:${sym}`)) return this.symbolCache.get(`MULT:${sym}`);
    const cands = /^(R_\d+|1HZ\d+V?|BOOM\d+N|CRASH\d+N|STPRNG\d*|RDBULL\d*|RDBEAR\d*|JD\d+|GDA\d+)$/.test(sym)
      ? [sym]
      : (DERIV_MAP[sym] || []);
    if (!cands.length) return null;
    let live = [];
    try { live = await this.client.activeSymbols(); } catch { live = []; }
    const liveSet = new Set(live);
    const check = async (d) => {
      try {
        const avail = await this.client.contractsFor(d);
        const types = new Set((avail || []).flatMap(a => [a.contract_type, ...(a.supported_contract_types || [])].filter(Boolean)));
        return types.has("MULTUP") && types.has("MULTDN");
      } catch { return false; }
    };
    for (const d of cands) {
      if (live.length && !liveSet.has(d)) continue;
      if (await check(d)) {
        this.symbolCache.set(`MULT:${sym}`, d);
        return d;
      }
    }
    // API hiccup (empty list): trust first candidate ONLY for synthetics
    // (their codes are the market); mapped FX/metals stay unverified = null.
    if (!live.length && cands.length && /^(R_|1HZ|BOOM|CRASH)/.test(cands[0])) {
      this.symbolCache.set(`MULT:${sym}`, cands[0]);
      return cands[0];
    }
    return null;
  }

  async execute({ symbol, direction, durationMin, positionId }) {
    if (!this.client.token) return { ok: false, why: "no token" };
    if (direction !== "CALL" && direction !== "PUT") return { ok: false, why: "not directional" };
    if (this.tradesToday() >= this.maxPerDay) {
      return { ok: false, why: `daily cap reached (${this.maxPerDay})` };
    }
    // NOTE: cap is checked before any network so a capped day costs nothing.
    const ok = await this.client.connect();
    if (!ok) return { ok: false, why: "deriv unavailable" };
    if (!this.client.account?.isDemo && !this.liveAllowed) {
      return { ok: false, why: "REAL-money token refused — DERIV_LIVE=1 required" };
    }
    const dSym = await this.resolveSymbol(symbol);
    if (!dSym) return { ok: false, why: `no Rise/Fall market for ${symbol}` };
    const r = await this.client.buyRiseFall({
      symbol: dSym, direction, stake: this.stake, durationMin, currency: this.currency,
    });
    if (r.error) {
      this.log(`[deriv] buy failed ${symbol}: ${r.error}`);
      return { ok: false, why: r.error };
    }
    const mode = this.client.account?.isDemo ? "demo" : "REAL";
    try {
      this.db.prepare(`
        INSERT OR REPLACE INTO deriv_contracts
        (contract_id, position_id, symbol, deriv_symbol, direction, stake, duration_min, entry_tick, status, mode, placed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
      `).run(String(r.contract_id), positionId || null, symbol, dSym, direction,
        this.stake, durationMin, r.entry_tick ?? null, mode, Date.now());
    } catch (e) { this.log(`[deriv] ledger failed: ${e.message}`); }
    this.log(`[deriv] ${mode} ${direction} ${dSym} ${durationMin}m stake=${this.stake} contract=${r.contract_id}`);
    return { ok: true, mode, ...r };
  }

  /**
   * Multiplier purchase for spot-direction engine signals (MTF/longterm on
   * synthetics). direction: "buy"|"sell" (spot). Stake = max loss by
   * construction; native take_profit/stop_loss ride the contract.
   * duration_min is stored as 0 (MULT has no expiry) — NOT NULL safe.
   */
  async executeMult({ symbol, direction, stake, multiplier = 100, takeProfit = null, stopLoss = null }) {
    if (!this.client.token) return { ok: false, why: "no token" };
    const dir = String(direction).toLowerCase() === "sell" ? "sell" : "buy";
    if (this.tradesToday() >= this.maxPerDay) {
      return { ok: false, why: `daily cap reached (${this.maxPerDay})` };
    }
    const ok = await this.client.connect();
    if (!ok) return { ok: false, why: "deriv unavailable" };
    if (!this.client.account?.isDemo && !this.liveAllowed) {
      return { ok: false, why: "REAL-money token refused — DERIV_LIVE=1 required" };
    }
    const dSym = await this.resolveMultSymbol(symbol);
    if (!dSym) return { ok: false, why: `no MULT market for ${symbol} on Deriv` };
    const st = Math.min(Math.max(Number(stake) || this.stake, 0.35), 1000);
    let r;
    try {
      r = await this.client.buyMultiplier({
        symbol: dSym, direction: dir, stake: st, multiplier,
        takeProfit, stopLoss,
      });
    } catch (e) {
      this.log(`[deriv] mult buy failed ${symbol}: ${e.message}`);
      return { ok: false, why: e.message };
    }
    const mode = this.client.account?.isDemo ? "demo" : "REAL";
    try {
      this.db.prepare(`
        INSERT OR REPLACE INTO deriv_contracts
        (contract_id, position_id, symbol, deriv_symbol, direction, stake, duration_min, entry_tick, status, mode, placed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
      `).run(String(r.contract_id), null, symbol, dSym, dir === "buy" ? "MULTUP" : "MULTDN",
        st, 0, null, mode, Date.now());
    } catch (e) { this.log(`[deriv] ledger failed: ${e.message}`); }
    this.log(`[deriv] ${mode} MULT ${dir} ${dSym} stake=${st} contract=${r.contract_id}`);
    return { ok: true, mode, contract_id: r.contract_id, stake: st, derivSymbol: dSym };
  }

  async settleOpen(resolveFn) {
    // resolveFn(positionId, outcome win|loss, closePrice, note) -> void
    let rows = [];
    try {
      rows = this.db.prepare("SELECT * FROM deriv_contracts WHERE status = 'OPEN'").all();
    } catch { return { settled: 0 }; }
    let settled = 0;
    for (const c of rows) {
      try {
        const st = await this.client.contractStatus(c.contract_id);
        if (!st || !st.done || st.profit == null) continue;
        // Rise/Fall without allow-equals: ties lose — zero profit counts as loss.
        const outcome = st.profit > 0 ? "win" : "loss";
        const closePx = st.exit ?? st.entry ?? null;
        this.db.prepare("UPDATE deriv_contracts SET status = 'SETTLED', profit = ? WHERE contract_id = ?")
          .run(st.profit, c.contract_id);
        if (c.position_id) {
          try { await resolveFn(c.position_id, outcome, closePx, `deriv ${c.mode} contract ${c.contract_id} P&L ${st.profit}`); } catch { /* ignore */ }
        }
        settled++;
      } catch { /* per-contract best effort */ }
    }
    return { settled };
  }

  status() {
    let open = 0;
    try { open = this.db.prepare("SELECT COUNT(*) AS c FROM deriv_contracts WHERE status = 'OPEN'").get()?.c || 0; } catch { /* ignore */ }
    return {
      configured: !!this.client.token,
      mode: this.client.account ? (this.client.account.isDemo ? "demo" : "REAL") : "unknown",
      liveAllowed: this.liveAllowed,
      conn: this.client.status(),
      stake: this.stake, maxPerDay: this.maxPerDay, tradesToday: this.tradesToday(),
      openContracts: open,
    };
  }
}
