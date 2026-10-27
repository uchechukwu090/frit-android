// ============================================================================
// FRIT METAAPI RAIL — FBS (MT5) orders over MetaApi.cloud REST
// ----------------------------------------------------------------------------
// MetaApi hosts the MT5 terminal; FRIT places market orders over REST, so no
// VPS / Windows box is needed. This rail owns FX, metals, indices, energy and
// crypto CFDs. Rise/Fall (pocket bot) and every Deriv synthetic stay on Deriv.
//
// Env:
//   METAAPI_TOKEN         API token from the MetaApi dashboard
//   METAAPI_ACCOUNT_ID    the MetaApi account id (a UUID, NOT the MT5 login)
//   METAAPI_REGION        region shown on the account (default new-york)
//   METAAPI_BASE_URL      optional full base-URL override
//   METAAPI_SYMBOL_SUFFIX optional broker suffix, e.g. "m" -> EURUSDm
//
// REST (all under /users/current/accounts/<id>):
//   GET  /symbols/<symbol>/specification   volumeStep / minVolume / maxVolume
//   POST /trade                            ORDER_TYPE_BUY / ORDER_TYPE_SELL
//   GET  /account-information, /positions  health check
// Auth header: auth-token
// ============================================================================

const OK_STRING = new Set([
  "ERR_NO_ERROR", "TRADE_RETCODE_DONE", "TRADE_RETCODE_PLACED", "TRADE_RETCODE_DONE_PARTIAL",
]);
const OK_NUMERIC = new Set([0, 10008, 10009, 10010]);

// Names brokers commonly use for the engine's symbols.
const ALIASES = {
  XAUUSD: ["GOLD"],
  XAGUSD: ["SILVER"],
  US30: ["DJ30", "DJI30", "WS30"],
  NAS100: ["USTEC", "US100", "NDX100"],
  SPX500: ["US500", "SP500"],
  GER40: ["DE40", "DAX40", "DE30"],
  UK100: ["FTSE100"],
  JP225: ["JPN225", "NIKKEI225"],
  USOIL: ["XTIUSD", "WTI", "CRUDEOIL"],
  UKOIL: ["XBRUSD", "BRENT"],
};
const COMMON_SUFFIXES = ["m", ".m", "_m", ".r", "#"];

export function metaApiConfig(env = process.env) {
  const token = String(env.METAAPI_TOKEN || "").trim();
  const accountId = String(env.METAAPI_ACCOUNT_ID || "").trim();
  const region = String(env.METAAPI_REGION || "new-york").trim() || "new-york";
  const override = String(env.METAAPI_BASE_URL || "").trim().replace(/\/+$/, "");
  return {
    ready: !!(token && accountId),
    token,
    accountId,
    region,
    suffix: String(env.METAAPI_SYMBOL_SUFFIX || ""),
    base: override || `https://mt-client-api-v1.${region}.agiliumtrade.ai`,
  };
}

/**
 * Snap a lot to the broker's volume step. MT5 REJECTS a volume that is not an
 * exact multiple of volumeStep, so we floor to the step (never over-risk), then
 * clamp to [minVolume, maxVolume]. Garbage / zero / negative = 0 = "no trade".
 */
export function normalizeVolume(lot, { volumeStep, minVolume, maxVolume } = {}) {
  const v = Number(lot);
  if (!Number.isFinite(v) || v <= 0) return 0;
  const step = Number(volumeStep) > 0 ? Number(volumeStep) : 0.01;
  const min = Number(minVolume) > 0 ? Number(minVolume) : step;
  const max = Number(maxVolume) > 0 ? Number(maxVolume) : Infinity;
  const decimals = Math.min(8, (String(step).split(".")[1] || "").length);
  const snap = (x) => Number((Math.floor(x / step + 1e-9) * step).toFixed(decimals));
  let out = snap(v);
  if (out < min) out = snap(min) < min ? Number((snap(min) + step).toFixed(decimals)) : snap(min);
  if (out > max) out = snap(max);
  return out;
}

/** The ladder of spellings to try against the broker, most likely first. */
export function symbolCandidates(symbol, suffix = "") {
  const base = String(symbol || "").trim().toUpperCase();
  if (!base) return [];
  const out = [base];
  if (suffix) out.push(base + suffix);
  for (const s of COMMON_SUFFIXES) out.push(base + s);
  for (const a of ALIASES[base] || []) {
    out.push(a);
    if (suffix) out.push(a + suffix);
  }
  return [...new Set(out)];
}

export function orderActionType(action) {
  const a = String(action || "").toLowerCase();
  return a === "sell" || a === "put" ? "ORDER_TYPE_SELL" : "ORDER_TYPE_BUY";
}

function clientIdNow() {
  return `F${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`.slice(0, 20);
}

export class MetaApiRail {
  constructor(cfg = {}, { fetchImpl = globalThis.fetch, log = (...a) => console.log(...a), timeoutMs = 30_000 } = {}) {
    this.cfg = cfg;
    this.base = String(cfg.base || "").replace(/\/+$/, "");
    this.token = cfg.token || "";
    this.accountId = cfg.accountId || "";
    this.suffix = cfg.suffix || "";
    this.fetch = fetchImpl;
    this.log = log;
    this.timeoutMs = timeoutMs;
    this._specs = new Map(); // engine symbol -> { at, spec | null }
  }

  get ready() { return !!(this.cfg.ready && this.token && this.accountId); }

  _url(path) {
    return `${this.base}/users/current/accounts/${encodeURIComponent(this.accountId)}${path}`;
  }

  async _call(method, path, body = null) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), this.timeoutMs);
    try {
      const res = await this.fetch(this._url(path), {
        method,
        headers: { "auth-token": this.token, "Content-Type": "application/json", Accept: "application/json" },
        body: body ? JSON.stringify(body) : undefined,
        signal: ctl.signal,
      });
      const text = await res.text().catch(() => "");
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch { data = { message: text.slice(0, 300) }; }
      return { ok: !!res.ok, status: res.status, data };
    } finally {
      clearTimeout(timer);
    }
  }

  /** Find the broker's own spelling for a symbol and its volume rules. */
  async resolveSymbol(symbol) {
    const key = String(symbol || "").toUpperCase();
    const hit = this._specs.get(key);
    if (hit && Date.now() - hit.at < 3_600_000) return hit.spec;
    for (const cand of symbolCandidates(key, this.suffix)) {
      const r = await this._call("GET", `/symbols/${encodeURIComponent(cand)}/specification`);
      if (r.ok && r.data) {
        const spec = {
          symbol: r.data.symbol || cand,
          volumeStep: Number(r.data.volumeStep) || 0.01,
          minVolume: Number(r.data.minVolume) || undefined,
          maxVolume: Number(r.data.maxVolume) || undefined,
        };
        this._specs.set(key, { at: Date.now(), spec });
        return spec;
      }
      if (r.status === 401 || r.status === 403) {
        throw new Error(`MetaApi rejected the token (HTTP ${r.status}) — check METAAPI_TOKEN`);
      }
    }
    this._specs.set(key, { at: Date.now(), spec: null });
    return null;
  }

  async placeMarketOrder({ symbol, action, lotSize, entry = null, sl = null, tp = null, reason = "" }) {
    const side = String(action || "").toLowerCase() === "sell" ? "sell" : "buy";
    if (!this.ready) {
      this.log(`[PAPER TRADE] ${side.toUpperCase()} ${symbol} | Lot:${lotSize} | SL:${sl} | TP:${tp}`);
      return { mode: "paper", status: "simulated", symbol, action: side, lotSize, entry, sl, tp, message: "MetaApi not configured — paper mode" };
    }
    try {
      const spec = await this.resolveSymbol(symbol);
      if (!spec) {
        return {
          mode: "paper", status: "simulated", symbol, action: side, lotSize, entry, sl, tp,
          message: `MetaApi account does not list ${symbol} — not sent`,
        };
      }
      const volume = normalizeVolume(lotSize, spec);
      if (!(volume > 0)) {
        return { mode: "live", status: "refused", symbol, brokerSymbol: spec.symbol, action: side, lotSize, code: "BAD_VOLUME", message: `lot ${lotSize} is not a tradable volume` };
      }
      const body = {
        actionType: orderActionType(side),
        symbol: spec.symbol,
        volume,
        clientId: clientIdNow(),
      };
      if (Number(sl) > 0) body.stopLoss = Number(sl);
      if (Number(tp) > 0) body.takeProfit = Number(tp);
      // A trade is sent ONCE. An ambiguous failure (timeout) is reported, not
      // retried: re-sending could double the position.
      const r = await this._call("POST", "/trade", body);
      const d = r.data || {};
      const ok = r.ok && (OK_STRING.has(String(d.stringCode)) || OK_NUMERIC.has(Number(d.numericCode)));
      if (ok) {
        this.log(`[metaapi] ${side.toUpperCase()} ${spec.symbol} ${volume} lot order=${d.orderId} pos=${d.positionId}`);
        return {
          mode: "live", status: "placed", symbol, brokerSymbol: spec.symbol, action: side,
          lotSize: volume, entry, sl, tp, orderId: d.orderId ?? null, positionId: d.positionId ?? null,
          reason,
        };
      }
      if (r.ok) {
        // Reached the broker and was declined (no money, bad stops, market closed...).
        return {
          mode: "live", status: "refused", symbol, brokerSymbol: spec.symbol, action: side,
          lotSize: volume, code: d.stringCode || d.numericCode || "REFUSED", message: d.message || "order refused by broker",
        };
      }
      return {
        mode: "live", status: "failed", symbol, action: side,
        error: `HTTP ${r.status}: ${d.message || "MetaApi error"}`,
      };
    } catch (e) {
      const msg = e?.name === "AbortError" ? "MetaApi request timed out" : e.message;
      this.log(`[metaapi] ${msg}`);
      return { mode: "live", status: "failed", symbol, action: side, error: msg };
    }
  }

  /** Live round-trip for /api/trading/rails: balance + open positions. */
  async health() {
    if (!this.ready) return { ok: false, configured: false, reason: "METAAPI_TOKEN / METAAPI_ACCOUNT_ID unset" };
    try {
      const info = await this._call("GET", "/account-information");
      if (!info.ok) {
        return { ok: false, configured: true, error: `HTTP ${info.status}: ${info.data?.message || "account-information failed"}` };
      }
      let positions = null;
      try {
        const p = await this._call("GET", "/positions");
        if (p.ok && Array.isArray(p.data)) positions = p.data.length;
      } catch { /* positions are a nice-to-have */ }
      const d = info.data || {};
      return {
        ok: true, configured: true,
        broker: d.broker || null, server: d.server || null, currency: d.currency || null,
        balance: d.balance ?? null, equity: d.equity ?? null, openPositions: positions,
      };
    } catch (e) {
      return { ok: false, configured: true, error: e?.name === "AbortError" ? "MetaApi request timed out" : e.message };
    }
  }
}
