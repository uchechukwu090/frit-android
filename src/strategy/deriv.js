// ============================================================================
// DERIV EXECUTOR — Deriv "Options" API (api.derivws.com), Rise/Fall + Multiplier
// ----------------------------------------------------------------------------
// MIGRATED (Oct 2026): Deriv retired the legacy websocket API
// (wss://ws.derivws.com/websockets/v3 — now dead) and moved to the new
// "Options" API documented at https://developers.deriv.com. Protocol changes
// honored here, all verified against the live endpoint:
//   - Data socket (public, no auth): wss://api.derivws.com/trading/v1/options/ws/public
//     candles/ticks/active_symbols/contracts_for all work here anonymously.
//   - Trading sockets: there is NO `authorize` handshake on the new API. Instead
//     a REST OTP-URL exchange grants a single-use websocket URL per account:
//       GET  /trading/v1/options/accounts                      (Bearer + App-ID)
//       POST /trading/v1/options/accounts/{accountId}/otp      -> { data.url }
//       connect(url) — no auth message. OTP is single-use/120s, so EVERY
//       connect() re-issues it (only the accounts list itself is cached).
//   - `symbol` -> `underlying_symbol` in proposal/contract requests.
//   - active_symbols rejects `product_type`; items expose `underlying_symbol`.
//   - contract_id is an INTEGER in sell / proposal_open_contract.
//   - Multiplier contracts: `amount`+`basis`+`currency`+`multiplier` (the old
//     `symbol`+`stake` shape is rejected) and NO take_profit/stop_loss (not in
//     the schema — proposals with them are rejected with InputValidationFailed).
//   - Symbol universe has no plain XAUUSD/EURUSD/BTCUSD — only frx*/cry*/
//     OTC_*/synthetics; see DERIV_MAP + canDerivServe below.
// SAFETY DESIGN (unchanged, read before touching):
// - DEMO-FIRST: a demo account is picked by default. REAL-money trading needs
//   DERIV_LIVE=1 *and* a real account — both checked at execution.
// - Daily trade cap + single-stake cap. No martingale, no averaging.
// Every fill registers into the shared positions monitor and resolves on
// BROKER outcome (proposal_open_contract).
// Duration auto-clamping: contracts_for exposes min/max_contract_duration per
// market (e.g. frxEURUSD intraday CALL/PUT = 15m..1d). Requests outside the
// range are clamped with a logged note instead of being rejected blindly.
// ============================================================================

import WebSocket from "ws";

const REST_BASE = "https://api.derivws.com";
export const WS_PUBLIC_URL = "wss://api.derivws.com/trading/v1/options/ws/public";

// Engine symbol -> Deriv underlying (both verified against the live
// active_symbols list, Oct 2026: 89 codes = 46 synthetics + 12 OTC indices +
// 25 forex + 4 commodities + 2 crypto). This is what makes Deriv a full DATA
// PROVIDER: every symbol here (and every pasted frx*/cry*/OTC_*/synthetic code)
// can be watched and fed candles through the keyless public socket, wired in
// live_feed's deriv fallback tier (Dukascopy/Finnhub stay first for FX when
// they are live; Deriv covers what nobody else serves — synthetics, OTC
// indices, and any FX pair whose usual providers are down).
// Deriv's own codes are single-valued now: the legacy un-prefixed quote codes
// (XAUUSD→frxXAUUSD) no longer exist on the API, so frx*/cry* ARE the market.
// Pure synthetics (R_100, 1HZ100V, BOOM/CRASH, stpRNG, JD, RB, WLD*) are
// NEVER routed to execution — no edge can exist on an RNG stream (see
// SYNTHETIC_ONLY); their data still feeds the charts and the MULT rail.
export const DERIV_MAP = {
  // forex (25, all frx*)
  AUDCAD: ["frxAUDCAD"], AUDCHF: ["frxAUDCHF"], AUDJPY: ["frxAUDJPY"],
  AUDNZD: ["frxAUDNZD"], AUDUSD: ["frxAUDUSD"], EURAUD: ["frxEURAUD"],
  EURCAD: ["frxEURCAD"], EURCHF: ["frxEURCHF"], EURGBP: ["frxEURGBP"],
  EURJPY: ["frxEURJPY"], EURNZD: ["frxEURNZD"], EURUSD: ["frxEURUSD"],
  GBPAUD: ["frxGBPAUD"], GBPCAD: ["frxGBPCAD"], GBPCHF: ["frxGBPCHF"],
  GBPJPY: ["frxGBPJPY"], GBPNZD: ["frxGBPNZD"], GBPUSD: ["frxGBPUSD"],
  NZDJPY: ["frxNZDJPY"], NZDUSD: ["frxNZDUSD"], USDCAD: ["frxUSDCAD"],
  USDCHF: ["frxUSDCHF"], USDJPY: ["frxUSDJPY"], USDMXN: ["frxUSDMXN"],
  USDPLN: ["frxUSDPLN"],
  // commodities (4, all frx*)
  XAUUSD: ["frxXAUUSD"], XAGUSD: ["frxXAGUSD"], XPTUSD: ["frxXPTUSD"],
  XPDUSD: ["frxXPDUSD"],
  // crypto (2, all cry*)
  BTCUSD: ["cryBTCUSD"], ETHUSD: ["cryETHUSD"],
  // OTC index aliases: Deriv's derived indices serve the same underlyings
  // MT5 CFDs name differently — free 24/7 data (and legal Rise/Fall demo
  // execution) where an MT5 bridge or paid feed may never exist.
  US30: ["OTC_DJI"], NAS100: ["OTC_NDX"], SPX500: ["OTC_SPC"],
  GER40: ["OTC_GDAXI"], UK100X: ["OTC_FTSE"], JPN225: ["OTC_N225"],
  AUS200: ["OTC_AS51"], FRA40: ["OTC_FCHI"], EU50: ["OTC_SX5E"],
};

// The full live universe (from active_symbols, Oct 2026). Runtime-extendable:
// DerivClient.activeSymbolsDetail() registers whatever the API actually
// offers, so newly listed codes start working without a code change.
export const DERIV_UNIVERSE = new Set([
  ...Object.values(DERIV_MAP).flat(),
  "R_10", "R_25", "R_50", "R_75", "R_100",
  "1HZ10V", "1HZ15V", "1HZ25V", "1HZ30V", "1HZ50V", "1HZ75V", "1HZ90V", "1HZ100V",
  "BOOM50", "BOOM150N", "BOOM300N", "BOOM500", "BOOM600", "BOOM900", "BOOM1000",
  "CRASH50", "CRASH150N", "CRASH300N", "CRASH500", "CRASH600", "CRASH900", "CRASH1000",
  "JD10", "JD25", "JD50", "JD75", "JD100",
  "RB100", "RB200", "RDBEAR", "RDBULL",
  "stpRNG", "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5",
  "WLDUSD", "WLDEUR", "WLDGBP", "WLDAUD", "WLDXAU",
  "OTC_AEX", "OTC_AS51", "OTC_DJI", "OTC_FCHI", "OTC_FTSE", "OTC_GDAXI",
  "OTC_HSI", "OTC_N225", "OTC_NDX", "OTC_SPC", "OTC_SSMI", "OTC_SX5E",
]);

/** Live-list registration: unknown-but-real codes become servable at once. */
export function registerDerivSymbols(list) {
  let added = 0;
  try {
    for (const s of list || []) {
      const code = String(s || "").trim();
      if (code && !DERIV_UNIVERSE.has(code)) { DERIV_UNIVERSE.add(code); added++; }
    }
  } catch { /* best-effort */ }
  return added;
}

/** Case-preserving canonical code for anything the API knows; null otherwise. */
export function canDerivServe(ours) {
  return derivUnderlying(ours) != null;
}


// New-API synthetic families (live active_symbols taxonomy, verified Oct 2026):
//   R_10..R_100, 1HZ10V..1HZ100V (RNG streams), BOOM/CRASH 50|300N|500|600|900|
//   1000 (most newer codes have NO trailing N; only 150N/300N retain it),
//   stpRNG/stpRNG2-5, RDBULL/RDBEAR (no numeric suffix now), JD\d+,
//   RB100/RB200 (range-break), WLD*** (weighted baskets), OTC_* (derived
//   indices: OTC_DJI, OTC_GDAXI, ...).
export const SYNTHETIC_ONLY =
  /^(R_\d+|1HZ\d+V?|BOOM\d+N?|CRASH\d+N?|STPRNG\d*|RDBULL|RDBEAR|JD\d+|RB\d+|WLD[A-Z]{3}|OTC_[A-Z0-9]+)$/;

// RNG (random-number-generator) streams. These are purely synthetic random
// walks — no edge can exist and no market data provider legitimizes trading
// them — so EXECUTION refuses them forever. `SYNTHETIC_ONLY` additionally
// covers OTC_* derived indices, which are NOT random (they mirror real index
// prices on Deriv's own feed) and may be served as data and demo-executed.
export const RNG_SYNTH =
  /^(R_\d+|1HZ\d+V?|BOOM\d+N?|CRASH\d+N?|STPRNG\d*|RDBULL\d*|RDBEAR\d*|JD\d+|RB\d+|WLD[A-Z]{3})$/;

export function buildProposal(symbol, direction, stake, durationMin, currency = "USD") {
  return {
    proposal: 1,
    amount: stake,
    basis: "stake",
    contract_type: direction === "CALL" ? "CALL" : "PUT",
    currency,
    duration: durationMin,
    duration_unit: "m",
    underlying_symbol: symbol,
  };
}

/** Engine symbol -> deriv underlying it can serve (for the DATA rail). */
export function derivUnderlying(ours) {
  const sym = String(ours || "").toUpperCase();
  if (!sym) return null;
  // Exact (case-insensitive) match against the known universe keeps native
  // codes in the API's exact case: "STPRNG" input -> "stpRNG" output.
  for (const c of DERIV_UNIVERSE) {
    if (c.toUpperCase() === sym) return c;
  }
  // User-pasted native prefixes the map doesn't key on: lowercase frx/cry.
  const m = /^(FRX|CRY)([A-Z0-9]+)$/.exec(sym);
  if (m) return m[1].toLowerCase() + m[2];
  if (SYNTHETIC_ONLY.test(sym)) return sym; // unknown-but-shaped synthetic
  const cands = DERIV_MAP[sym];
  return cands ? cands[0] : null;
}

// "15m" -> 15, "1d" -> 1440, "60s" -> 1, "2h" -> 120
export function durationToMinutes(s) {
  const m = /^(\d+)([smhd])$/i.exec(String(s || "").trim());
  if (!m) return null;
  const n = Number(m[1]);
  const f = { s: 1 / 60, m: 1, h: 60, d: 1440 }[m[2].toLowerCase()];
  return Math.round(n * f);
}

export class DerivClient {
  constructor({
    token,
    appId = "",
    log = null,
    onStatus = null,
    wsUrl = null,
    WebSocketImpl = null,
    fetchImpl = null,
    preferReal = false,
  } = {}) {
    this.token = token;
    this.appId = String(appId || "");
    this.wsUrl = wsUrl || WS_PUBLIC_URL; // injectable for tests; default public socket
    this.WSImpl = WebSocketImpl || WebSocket;
    this.fetch = fetchImpl || fetch;
    this.preferReal = !!preferReal;
    this.log = log || (() => {});
    this.onStatus = onStatus || (() => {});
    this.ws = null;
    this.reqId = 0;
    this.pending = new Map();
    this.ready = false;
    // { accountId, isDemo, accountType, currency } — set at connect.
    this.account = null;
    this._status = "idle";
    this.stopping = false;
    this._accountsCache = { at: 0, list: [] };
  }

  setStatus(s, d = "") {
    this._status = s;
    try { this.onStatus(s, d); } catch { /* ignore */ }
  }
  status() { return this._status; }

  // ---- REST (accounts + OTP) ----
  _headers() {
    return {
      Authorization: `Bearer ${this.token}`,
      "Deriv-App-ID": this.appId,
      Origin: "https://home.deriv.com",
      "Content-Type": "application/json",
    };
  }

  async _rest(method, path, body = null, timeoutMs = 15_000) {
    const res = await this.fetch(`${REST_BASE}${path}`, {
      method,
      headers: this._headers(),
      body: body != null ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(timeoutMs),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = data?.errors?.[0] || data?.error || data?.detail || null;
      const msg = err?.message || err?.detail?.message || `HTTP ${res.status}`;
      return { ok: false, status: res.status, message: msg };
    }
    return { ok: true, data };
  }

  /** Trading accounts reachable with this token (60s cache).
   *  Returns { ok, list:[{account_id, account_type, balance, currency}], message}. */
  async accounts() {
    if (Date.now() - this._accountsCache.at < 60_000 && this._accountsCache.list.length) {
      return { ok: true, list: this._accountsCache.list };
    }
    try {
      const r = await this._rest("GET", "/trading/v1/options/accounts");
      if (!r.ok) return { ok: false, list: [], message: r.message };
      const body = r.data?.data;
      const raw = Array.isArray(body) ? body
        : Array.isArray(body?.accounts) ? body.accounts
        : Array.isArray(r.data?.accounts) ? r.data.accounts : [];
      const list = raw.map((a) => ({
        account_id: a.account_id,
        account_type: a.account_type,
        balance: Number(a.balance) || 0,
        currency: a.currency || "USD",
      }));
      if (!list.length) return { ok: false, list: [], message: "token has no trading accounts (missing trade scope?)" };
      this._accountsCache = { at: Date.now(), list };
      return { ok: true, list };
    } catch (e) {
      return { ok: false, list: [], message: e.message };
    }
  }

  /** Pick the account Deriv execution runs on. Demo first unless preferReal. */
  _pickAccount(list) {
    const demo = list.find((a) => a.account_type === "demo");
    const real = list.find((a) => a.account_type === "real");
    return this.preferReal ? (real || demo || list[0]) : (demo || real || list[0]);
  }

  /** Single-use OTP websocket URL; fresh on every call (120s TTL upstream). */
  async _otpUrl() {
    const accs = await this.accounts();
    if (!accs.ok) {
      this.setStatus("down", `accounts: ${accs.message}`);
      return null;
    }
    const pick = this._pickAccount(accs.list);
    if (!pick) {
      this.setStatus("down", "no trading account on token");
      return null;
    }
    this.account = {
      accountId: pick.account_id,
      loginid: pick.account_id,
      isDemo: pick.account_type !== "real",
      accountType: pick.account_type,
      currency: pick.currency,
    };
    try {
      const r = await this._rest("POST", `/trading/v1/options/accounts/${pick.account_id}/otp`);
      if (!r.ok) {
        this.setStatus("down", `otp: ${r.message}`);
        return null;
      }
      const url = r.data?.data?.url || r.data?.url;
      if (!url) {
        this.setStatus("down", "otp: no websocket url in response");
        return null;
      }
      return url;
    } catch (e) {
      this.setStatus("down", `otp: ${e.message}`);
      return null;
    }
  }

  async connect() {
    if (this.ready) return true;
    const anonymous = !this.token;
    if (!anonymous && !this.appId) {
      // PAT auth is header-based; without an app id every REST call 401s.
      this.setStatus("down", "DERIV_APP_ID missing — PAT tokens require a registered app id");
      return false;
    }
    return new Promise((resolve) => {
      this.setStatus("connecting");
      let ws;
      try {
        const url = anonymous ? this.wsUrl : null;
        if (url) {
          ws = new this.WSImpl(url);
        } else {
          this.setStatus("connecting", "requesting OTP websocket");
          // _otpUrl needs no socket; resolve	connect after URL lands.
        }
      } catch (e) {
        this.setStatus("down", e.message);
        return resolve(false);
      }
      if (anonymous) {
        this._finalConnect(ws, resolve, () => {
          this.account = { accountId: "public", loginid: "public", isDemo: true, accountType: "demo" };
        });
        return;
      }
      // Trading path: OTP URL first, THEN open the socket on it.
      this._otpUrl().then((otpUrl) => {
        if (this.stopping) return resolve(false);
        if (!otpUrl) return resolve(false);
        try { ws = new this.WSImpl(otpUrl); } catch (e) {
          this.setStatus("down", e.message);
          return resolve(false);
        }
        this._finalConnect(ws, resolve);
      });
    });
  }

  _finalConnect(ws, resolve, onOpenExtra = null) {
    const timer = setTimeout(() => {
      try { ws.close(); } catch { /* ignore */ }
      this.setStatus("down", "connect timeout");
      resolve(false);
    }, 20_000);
    if (timer.unref) timer.unref();
    this.ws = ws;
    ws.on("open", () => {
      clearTimeout(timer);
      if (onOpenExtra) onOpenExtra();
      this.ready = true;
      this.setStatus("live", this.account
        ? `${this.account.loginid} (${this.account.isDemo ? "DEMO" : "REAL"})`
        : "connected");
      resolve(true);
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
    const r = await this.call({ active_symbols: "brief" });
    if (r.error || !Array.isArray(r.active_symbols)) return [];
    return r.active_symbols.map(s => s.underlying_symbol || s.symbol);
  }

  /** Full active_symbols detail (for status/doctor endpoints); every code the
   *  API actually offers is registered into the data-rail universe so newly
   *  listed symbols start flowing without a code change. */
  async activeSymbolsDetail() {
    const r = await this.call({ active_symbols: "brief" });
    if (r.error || !Array.isArray(r.active_symbols)) return [];
    registerDerivSymbols(r.active_symbols.map((s) => s.underlying_symbol || s.symbol));
    return r.active_symbols.map((s) => ({
      symbol: s.underlying_symbol || s.symbol,
      displayName: s.display_name || s.underlying_symbol || "",
      market: s.market || "",
      submarket: s.submarket || "",
    }));
  }

  async contractsFor(symbol) {
    const r = await this.call({ contracts_for: symbol });
    if (r.error || !r.contracts_for) return [];
    return r.contracts_for.available || [];
  }

  // ---- market data (public socket — no token needed) ----
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
    const prices = r.history?.prices || r.prices;
    if (r.error || !Array.isArray(prices) || !prices.length) {
      throw new Error(`deriv tick ${symbol}: ${r.error?.message || "empty"}`);
    }
    return Number(prices[prices.length - 1]);
  }

  // ---- multipliers (MULTUP/MULTDOWN): stake = max loss, no expiry. NOTE:
  // the new API has NO take_profit/stop_loss on proposals (InputValidationFailed
  // — verified); SL/TP risk shaping must come from the position monitor instead.
  async proposeMultiplier({ symbol, direction, stake, multiplier = 100, takeProfit = null, stopLoss = null, currency = "USD" }) {
    if (takeProfit != null || stopLoss != null) {
      // Once per construction, keep callers honest without breaking signature.
      this.log("[deriv] MULT take_profit/stop_loss not supported by the current Deriv API — ignored");
    }
    const req = {
      proposal: 1,
      contract_type: direction === "sell" ? "MULTDN" : "MULTUP",
      underlying_symbol: symbol,
      amount: stake,
      basis: "stake",
      currency,
      multiplier,
    };
    const r = await this.call(req);
    const p = r.proposal;
    if (r.error || !p?.id || !(Number(p.ask_price) > 0)) {
      throw new Error(`deriv proposal ${symbol}: ${r.error?.message || "no ask"}`);
    }
    return { id: p.id, askPrice: Number(p.ask_price), spot: Number(p.spot) || null };
  }

  /** Buy a multiplier. Callers (DerivExecutor.execute) enforce the demo/live
   *  gate beforehand; the token check here is the last line of defense. */
  async buyMultiplier({ symbol, direction, stake, multiplier, takeProfit = null, stopLoss = null, currency = "USD" }) {
    if (!this.token) throw new Error("no token — paper mode");
    const prop = await this.proposeMultiplier({ symbol, direction, stake, multiplier, takeProfit, stopLoss, currency });
    if (prop.askPrice > stake * 1.02) {
      throw new Error(`deriv ask ${prop.askPrice} exceeds stake ${stake} — refusing overpriced fill`);
    }
    const r = await this.call({ buy: prop.id, price: prop.askPrice });
    if (r.error || !r.buy?.contract_id) {
      throw new Error(`deriv buy ${symbol}: ${r.error?.message || "no contract_id"}`);
    }
    return {
      contract_id: Number(r.buy.contract_id),
      transaction_id: r.buy.transaction_id != null ? String(r.buy.transaction_id) : null,
      buy_price: Number(r.buy.buy_price) || stake,
      longcode: r.buy.longcode || "",
    };
  }

  /** Early-exit an open contract. */
  async sellContract(contractId) {
    if (!this.token) throw new Error("no token — paper mode");
    const r = await this.call({ sell: Number(contractId), price: 0 });
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
    return { balance: Number(r.balance.balance), currency: r.balance.currency || this.account?.currency || "USD" };
  }

  // Full Rise/Fall purchase: proposal -> buy. Returns {contract_id,...} or {error}.
  // Durations outside the market's offered range are CLAMPED (logged) instead
  // of rejection-prone: e.g. a 1m signal on frxEURUSD becomes the market's 15m
  // minimum — the expiry changes but execution survives.
  async buyRiseFall({ symbol, direction, stake, durationMin, currency = "USD" }) {
    let dur = Math.round(Number(durationMin));
    if (!(dur > 0)) return { error: "invalid duration" };
    const range = await this.rfDurationRange(symbol);
    let clamped = false;
    if (range) {
      const c = Math.min(Math.max(dur, range.min), range.max);
      if (c !== dur) { dur = c; clamped = true; }
    }
    const prop = await this.call(buildProposal(symbol, direction, stake, dur, currency));
    if (prop.error) return { error: prop.error.message || prop.error.code || "proposal failed" };
    const p = prop.proposal;
    if (!p?.id) return { error: "no proposal id" };
    const buy = await this.call({ buy: p.id, price: Number(p.ask_price) });
    if (buy.error) return { error: buy.error.message || buy.error.code || "buy failed" };
    if (!buy.buy?.contract_id) return { error: "no contract id" };
    return {
      contract_id: Number(buy.buy.contract_id),
      transaction_id: buy.buy.transaction_id != null ? String(buy.buy.transaction_id) : null,
      ask_price: Number(p.ask_price), payout: Number(p.payout),
      entry_tick: p.spot != null ? Number(p.spot) : null,
      durationMin: dur,
      durationClamped: clamped,
    };
  }

  /** Offered intraday CALL/PUT duration range for a deriv underlying, in
   *  minutes. prefers `intraday` rows; falls back to whatever exists. */
  async rfDurationRange(derivSymbol) {
    const key = `RF:${derivSymbol}`;
    if (this._rfRange) { if (this._rfRange.has(key)) return this._rfRange.get(key); }
    else this._rfRange = new Map();
    try {
      const avail = await this.contractsFor(derivSymbol);
      let rows = (avail || []).filter(a => a.contract_type === "CALL" || a.contract_type === "PUT" || a.contract_type === "CALLOUT" || a.contract_type === "PUTOUT");
      if (!rows.length) return null;
      const intraday = rows.filter(a => a.expiry_type === "intraday");
      if (intraday.length >= 2) rows = intraday;
      const mins = rows.map(a => durationToMinutes(a.min_contract_duration)).filter(v => v > 0);
      const maxs = rows.map(a => durationToMinutes(a.max_contract_duration)).filter(v => v > 0);
      const range = mins.length
        ? { min: Math.min(...mins), max: Math.max(...maxs) || 1440, expiryType: rows[0].expiry_type }
        : null;
      if (range) this._rfRange.set(key, range);
      return range;
    } catch { return null; }
  }

  async contractStatus(contractId) {
    const r = await this.call({ proposal_open_contract: 1, contract_id: Number(contractId) }, 15_000);
    if (r.error || !r.proposal_open_contract) return null;
    const c = r.proposal_open_contract;
    const done = c.is_sold === 1 || (c.status && c.status !== "open");
    return {
      done,
      status: c.status || (c.is_sold ? "sold" : "open"),
      profit: c.profit != null ? Number(c.profit) : null,
      entry: c.entry_spot != null ? Number(c.entry_spot) : (c.buy_price != null ? Number(c.buy_price) : null),
      exit: c.exit_spot != null ? Number(c.exit_spot) : (c.sell_price != null ? Number(c.sell_price) : null),
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
  constructor({ db, client, liveAllowed = false, stake = 1.0, maxPerDay = 20, currency = "USD", log = null, allowRngOnDemo = true }) {
    this.db = db;
    this.client = client;
    this.liveAllowed = !!liveAllowed;
    // RNG synthetics (R_*, 1HZ*, BOOM/CRASH, ...) are tradable on DEMO so every
    // market Deriv lists can be exercised end to end. Real accounts still refuse
    // them. Set DERIV_ALLOW_RNG=0 in the env (read by index.js) to restore the
    // old blanket refusal.
    this.allowRngOnDemo = !!allowRngOnDemo;
    this._matrix = null;
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
      CREATE TABLE IF NOT EXISTS deriv_refusals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        why TEXT NOT NULL,
        at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_deriv_refusals ON deriv_refusals(at DESC);
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

  // Refusal audit trail: every skipped execution is recorded with WHY, so
  // "nothing appeared on Deriv" is always explainable (never silent).
  _refuse(symbol, why) {
    try {
      this.db.prepare("INSERT INTO deriv_refusals (symbol, why, at) VALUES (?, ?, ?)")
        .run(String(symbol).toUpperCase(), String(why).slice(0, 300), Date.now());
    } catch { /* best-effort */ }
    this.log(`[deriv] REFUSED ${symbol}: ${why}`);
    return { ok: false, why };
  }

  refusals(limit = 20) {
    try {
      return this.db.prepare("SELECT symbol, why, at FROM deriv_refusals ORDER BY at DESC LIMIT ?").all(limit);
    } catch { return []; }
  }

  async resolveSymbol(ours) {
    const sym = String(ours).toUpperCase();
    if (this.symbolCache.has(sym)) return this.symbolCache.get(sym);
    // Synthetics are their own Deriv codes — verify Rise/Fall availability
    // directly. OTC_* indices count too: Deriv's derived-index feed mirrors
    // real prices and is not exempt from availability checks.
    const nat = derivUnderlying(sym);
    if (RNG_SYNTH.test(sym) || /^(OTC_|FRX|CRY)/.test(sym) || (nat && DERIV_UNIVERSE.has(nat))) {
      const dSym = nat || sym;
      try {
        const avail = await this.client.contractsFor(dSym);
        const types = new Set((avail || []).map(a => a.contract_type).filter(Boolean));
        if (types.has("CALL") && types.has("PUT")) {
          this.symbolCache.set(sym, dSym);
          return dSym;
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
    // FIX: sym is upper-cased, so the old case-sensitive "stp" never matched
    // STPRNG* and step indices could never be sent a Multiplier.
    // Any code in the live universe (incl. newly listed DEX*/DSI*/GDA* codes that
    // registerDerivSymbols picked up from active_symbols) resolves natively.
    const native = /^(R_|1HZ|BOOM|CRASH|STP|JD|RB|RDB|WLD|OTC_|FRX|CRY)/i.test(sym)
      ? (derivUnderlying(sym) || sym)
      : derivUnderlying(sym);
    const cands = native ? [native] : (DERIV_MAP[sym] || []);
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
    if (!live.length && cands.length && /^(R_|1HZ|BOOM|CRASH|stp|JD|RB)/i.test(cands[0])) {
      this.symbolCache.set(`MULT:${sym}`, cands[0]);
      return cands[0];
    }
    return null;
  }

  async execute({ symbol, direction, durationMin, positionId, stake: stakeIn = null }) {
    const stake = stakeIn != null ? Math.min(Math.max(Number(stakeIn) || this.stake, 0.35), 1000) : this.stake;
    if (!this.client.token) return this._refuse(symbol, "NO_TOKEN — set PAT (or DERIV_DEMO_TOKEN) in server/.env");
    if (direction !== "CALL" && direction !== "PUT") return this._refuse(symbol, "not directional");
    if (this.tradesToday() >= this.maxPerDay) {
      return this._refuse(symbol, `daily cap reached (${this.maxPerDay})`);
    }
    // NOTE: cap is checked before any network so a capped day costs nothing.
    const ok = await this.client.connect();
    if (!ok) return this._refuse(symbol, `deriv unavailable: ${this.client.status()}`);
    if (!this.client.account?.isDemo && !this.liveAllowed) {
      return this._refuse(symbol, "REAL-money token refused — set DERIV_LIVE=1 (demo token expected)");
    }
    if (RNG_SYNTH.test(String(symbol).toUpperCase()) && !(this.allowRngOnDemo && this.client.account?.isDemo)) {
      return this._refuse(symbol, "RNG-stream synthetic refused on a real account (demo only)");
    }
    const dSym = await this.resolveSymbol(symbol);
    if (!dSym) return this._refuse(symbol, `no Rise/Fall market for ${symbol} on this account`);
    const r = await this.client.buyRiseFall({
      symbol: dSym, direction, stake, durationMin, currency: this.currency,
    });
    if (r.error) {
      return this._refuse(symbol, `buy failed: ${r.error}`);
    }
    if (r.durationClamped) {
      this.log(`[deriv] duration ${durationMin}m not offered for ${dSym} — clamped to ${r.durationMin}m`);
    }
    const mode = this.client.account?.isDemo ? "demo" : "REAL";
    try {
      this.db.prepare(`
        INSERT OR REPLACE INTO deriv_contracts
        (contract_id, position_id, symbol, deriv_symbol, direction, stake, duration_min, entry_tick, status, mode, placed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
      `).run(String(r.contract_id), positionId || null, symbol, dSym, direction,
        stake, r.durationMin ?? durationMin, r.entry_tick ?? null, mode, Date.now());
    } catch (e) { this.log(`[deriv] ledger failed: ${e.message}`); }
    this.log(`[deriv] ${mode} ${direction} ${dSym} ${r.durationMin ?? durationMin}m stake=${stake} contract=${r.contract_id}`);
    return { ok: true, mode, ...r };
  }

  /**
   * ONE entry point for ANY Deriv market. Sends the contract family the market
   * actually offers: Multiplier first (price-based engine signals), then
   * Rise/Fall (CALL/PUT) for everything Multiplier doesn't cover. Availability
   * is verified live via contracts_for, so a market is never sent a contract it
   * can't take. Hard failures (daily cap, real-money lock, no token) stop the
   * chain instead of retrying under another contract type.
   * direction: "buy"|"sell"|"CALL"|"PUT".
   */
  async executeBest({ symbol, direction, stake, multiplier = 100, takeProfit = null, stopLoss = null, durationMin = 5, positionId = null, prefer = "auto" }) {
    const d = String(direction).toLowerCase();
    const isSell = d === "sell" || d === "put";
    const tried = [];
    const HARD = /daily cap|REAL-money|no token|deriv unavailable|NO_TOKEN/i;
    if (prefer !== "risefall") {
      const m = await this.executeMult({ symbol, direction: isSell ? "sell" : "buy", stake, multiplier, takeProfit, stopLoss });
      if (m.ok) return { ...m, contractType: isSell ? "MULTDN" : "MULTUP", tried };
      tried.push(`MULT: ${m.why}`);
      if (HARD.test(String(m.why))) return { ok: false, why: tried.join(" | "), tried };
    }
    const r = await this.execute({ symbol, direction: isSell ? "PUT" : "CALL", durationMin, positionId, stake });
    if (r.ok) return { ...r, contractType: isSell ? "PUT" : "CALL", derivSymbol: this.symbolCache.get(String(symbol).toUpperCase()) || null, tried };
    tried.push(`RISEFALL: ${r.why}`);
    return { ok: false, why: tried.join(" | "), tried };
  }

  /**
   * Capability matrix for EVERY market this account can see: which contract
   * families each underlying offers (MULT, Rise/Fall, Digits). 30-min cache.
   * Drives the all-markets test and the /api/trading/deriv/markets endpoint.
   */
  async marketMatrix({ force = false } = {}) {
    if (!force && this._matrix && Date.now() - this._matrix.at < 30 * 60_000) return this._matrix;
    const ok = await this.client.connect();
    if (!ok) return { at: Date.now(), rows: [], error: `deriv unavailable: ${this.client.status()}` };
    const detail = await this.client.activeSymbolsDetail();
    const rows = [];
    for (const dsc of detail) {
      try {
        const avail = await this.client.contractsFor(dsc.symbol);
        const types = new Set((avail || []).flatMap(a => [a.contract_type, ...(a.supported_contract_types || [])].filter(Boolean)));
        rows.push({
          ...dsc,
          mult: types.has("MULTUP") && types.has("MULTDN"),
          riseFall: types.has("CALL") && types.has("PUT"),
          digits: [...types].some(t => /^DIGIT/.test(t)),
          types: [...types].sort(),
        });
      } catch (e) {
        rows.push({ ...dsc, mult: false, riseFall: false, digits: false, types: [], error: e.message });
      }
    }
    this._matrix = { at: Date.now(), rows };
    return this._matrix;
  }

  /**
   * Multiplier purchase for spot-direction engine signals (MTF/longterm on
   * synthetics). direction: "buy"|"sell" (spot). Stake = max loss by
   * construction (MULT links loss to stake; SL/TP not offered by the API).
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
        takeProfit, stopLoss, currency: this.currency,
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
