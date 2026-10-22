// ============================================================================
// FRIT MARKET ROUTER — single source of truth: which rail owns a symbol
// ----------------------------------------------------------------------------
//   deriv  : Deriv synthetic indices (R_100, 1HZ100V, BOOM1000N, ...). No MT5
//            broker lists these; they trade 24/7/365 on Deriv's own engine.
//   alpaca : US stocks + ETFs (AAPL, SPY, ...). NYSE hours only.
//   mt5    : everything else (FX, metals, indices, energy, crypto) via the
//            MT5 bridge / phone MT5 app.
//
// Routing is by SYMBOL CLASS, never by which keys happen to be set — so
// behavior never silently flips the day a key is added. Missing keys or
// missing rails degrade to paper mode, never to a different live rail.
// ============================================================================

// Deriv families — refreshed Oct 2026 against the new "Options" API
// (89 live codes). Own synthetics: most BOOM/CRASH codes dropped the trailing
// N (BOOM50, BOOM500, BOOM1000...; only 150N/300N keep it), stpRNG* is
// lowercase-sourced (canonicalized here), RDBULL/RDBEAR lost their numeric
// suffix, RB100/RB200 and WLD*** baskets are new. Deriv's native quote codes
// route here too (frx* forex/metals, cry* crypto, OTC_* derived indices) so a
// user can watch any Deriv market on the free public rail with no bridge.
const DERIV_RE = /^(R_\d+|1HZ\d+V|BOOM\d+N?|CRASH\d+N?|STPRNG\d*|RDBULL\d*|RDBEAR\d*|JD\d+|RB\d+|WLD[A-Z]{3}|OTC_[A-Z0-9]+|(FRX|CRY)[A-Z0-9]{3,20})$/;

// Broker MT5 symbol aliases -> canonical engine symbol. CFD names differ per
// broker (US30 vs DJ30 vs DowJones). Unknown symbols pass through unchanged.
const MT5_ALIASES = {
  DJ30: "US30", DJI: "US30", WS30: "US30",
  US100: "NAS100", USTEC: "NAS100",
  US500: "SPX500", SPX: "SPX500",
  DE40: "GER40", DAX: "GER40", DE30: "GER40",
  UK100: "UK100X", FTSE: "UK100X",
  JP225: "JPN225", NIKKEI: "JPN225",
  WTI: "USOIL", USOILWTI: "USOIL", OILWTI: "USOIL",
  BRENT: "UKOIL", UKOILBRENT: "UKOIL", OILBRENT: "UKOIL",
  XNGUSD: "NATGAS", NGAS: "NATGAS",
  XAUUSDm: "XAUUSD", "XAUUSD.": "XAUUSD",
};

// Engine-canonical index/energy symbols (MT5 rail).
const INDEX_SET = new Set([
  "US30", "NAS100", "SPX500", "GER40", "UK100X", "JPN225",
  "AUS200", "FRA40", "EU50", "US2000", "USDX",
]);
const ENERGY_SET = new Set(["USOIL", "UKOIL", "NATGAS"]);

/** Canonicalize a broker-specific symbol (case-insensitive). */
export function canonicalSymbol(symbol) {
  const s = String(symbol || "").toUpperCase().trim();
  return MT5_ALIASES[s] || s;
}

/**
 * Which execution/data rail owns this symbol: "deriv" | "alpaca" | "mt5".
 * Alpaca = 1-5 letter US tickers (plus BRK.B style) that are NOT known
 * FX/metal/index/energy/crypto symbols.
 */
export function marketOf(symbol, mt5Known = null) {
  const s = canonicalSymbol(symbol);
  if (DERIV_RE.test(s)) return "deriv";
  if (isIndexOrEnergy(s)) return "mt5";
  if (mt5Known instanceof Set && mt5Known.has(s)) return "mt5";
  if (/^[A-Z]{1,5}(\.[A-Z])?$/.test(s) && !/(USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD|XAU|XAG|BTC|ETH|OIL|GAS)$/.test(s)) {
    return "alpaca";
  }
  return "mt5";
}

export function isIndexOrEnergy(symbol) {
  const s = canonicalSymbol(symbol);
  return INDEX_SET.has(s) || ENERGY_SET.has(s);
}

export function isDerivSymbol(symbol) {
  return DERIV_RE.test(canonicalSymbol(symbol));
}

/** Session class for engine gating: 24/7 (crypto+synthetics), stocks (NYSE), else FX window. */
// FIX (F-32): the doc always promised crypto = 24/7, but the code returned
// "fx" for crypto — a latent trap for the next consumer. (The engine itself
// is unaffected: sessionStatus double-covers via CRYPTO_SET.) Now matches the
// documented contract. Pairs mirror live_feed/Binance's crypto universe.
const CRYPTO_RE = /^(BTC|ETH|SOL|BNB|XRP|DOGE|ADA)(USD|USDT)$|^(BTC|ETH|SOL|BNB|XRP|DOGE|ADA)$/;
export function sessionClass(symbol) {
  const s = canonicalSymbol(symbol);
  if (isDerivSymbol(s)) return "always";
  if (CRYPTO_RE.test(s)) return "always";
  if (marketOf(s) === "alpaca") return "nyse";
  return "fx";
}
