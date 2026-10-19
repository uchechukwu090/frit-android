// ============================================================================
// FRIT STRATEGY — unit tests for the pure building blocks (no network)
// ----------------------------------------------------------------------------
// Week-2 (F-07): the strategy layer beyond engine.js had zero coverage. These
// test the pure modules that gate real money:
//   hysteresis.js   — anti-whipsaw reasoning on the published bias
//   rails/router.js — symbol class → execution rail routing
//   options/pricing — Black-Scholes + impliedVol round-trip + put-call parity
// ============================================================================

import test from "node:test";
import assert from "node:assert";
import { structureBroke, applyHysteresis } from "../src/strategy/hysteresis.js";
import { marketOf, canonicalSymbol, sessionClass, isDerivSymbol } from "../src/strategy/rails/router.js";
import { bsPrice, greeks, impliedVol, expectedMove } from "../src/strategy/options/pricing.js";

// ---- hysteresis ------------------------------------------------------------

test("hysteresis: structure break kills a long bias, harmless level ignored", () => {
  assert.equal(structureBroke(100, 99, "BUY"), true);   // traded below the low
  assert.equal(structureBroke(100, 101, "BUY"), false);
  assert.equal(structureBroke(100, 101, "SELL"), true); // short bias dies above
  assert.equal(structureBroke(null, 99, "BUY"), false); // no level = no break
});

test("hysteresis: counter-signal suppressed while bias stands, released on break", () => {
  // Standing long bias whose invalidation (78.6% zone) sits at 100.
  const prev = { side: "BUY", level: 100, decision: "BUY", at: 1 };
  // Engine emits SELL while price is still above the invalidation level.
  const suppressed = applyHysteresis(prev, { decision: "SELL", price: 101 }, 101);
  assert.equal(suppressed.ok, false);
  assert.equal(suppressed.suppressed, true);
  assert.ok(suppressed.reason.includes("suppressed"));
  // Same-direction re-signals always pass through (consistent, not noise).
  const same = applyHysteresis(prev, { decision: "BUY", price: 101 }, 101);
  assert.equal(same.ok, true);
  assert.equal(same.suppressed, false);
  // Once structure truly broke, the opposite signal is released.
  const released = applyHysteresis(prev, { decision: "SELL", price: 99 }, 99);
  assert.equal(released.ok, true);
  assert.equal(released.suppressed, false);
  assert.equal(released.released, true);
});

// ---- rails router ----------------------------------------------------------

test("router: symbol classes route to the right rail", () => {
  assert.equal(marketOf("R_100"), "deriv");
  assert.equal(marketOf("1HZ100V"), "deriv");
  assert.equal(marketOf("BOOM500N"), "deriv"); // Deriv codes carry the trailing N (BOOM500 is not a real code)
  assert.equal(marketOf("AAPL"), "alpaca");
  assert.equal(marketOf("SPY"), "alpaca");
  assert.equal(marketOf("XAUUSD"), "mt5");
  assert.equal(marketOf("EURUSD"), "mt5");
  assert.equal(marketOf("US30"), "mt5");
  assert.equal(marketOf("USOIL"), "mt5");
  // Broker aliases canonicalize before classification.
  assert.equal(canonicalSymbol("dJ30"), "US30");
  assert.equal(canonicalSymbol("ustec"), "NAS100");
  assert.equal(sessionClass("BTCUSD"), "always");
  assert.equal(sessionClass("AAPL"), "nyse");
  assert.equal(sessionClass("XAUUSD"), "fx");
  assert.equal(isDerivSymbol("CRASH1000N"), true);
});

// ---- options pricing -------------------------------------------------------

test("pricing: put-call parity holds for European quotes", () => {
  const S = 100, K = 105, T = 45 / 365, r = 0.045, q = 0.015, sigma = 0.35;
  const call = bsPrice({ S, K, T, r, q, sigma, type: "call" });
  const put = bsPrice({ S, K, T, r, q, sigma, type: "put" });
  const parity = S * Math.exp(-q * T) - K * Math.exp(-r * T);
  assert.ok(Math.abs((call - put) - parity) < 1e-9, `parity diff ${(call - put) - parity}`);
});

test("pricing: impliedVol round-trips a generated mid; garbage returns null", () => {
  const S = 600, K = 595, T = 35 / 365, r = 0.04, q = 0.013, sigma = 0.4;
  const trueCall = bsPrice({ S, K, T, r, q, sigma, type: "call" });
  const solved = impliedVol({ price: trueCall, S, K, T, r, q, type: "call" });
  assert.ok(solved != null && Math.abs(solved - sigma) < 1e-4, `solved ${solved}`);
  // Above the no-arbitrage ceiling → null, never a fake vol.
  assert.equal(impliedVol({ price: S * 2, S, K, T, r, q, type: "call" }), null);
  // Deep below intrinsic → null.
  assert.equal(impliedVol({ price: 0.0001, S, K, T, r, q, type: "call" }), null);
});

test("pricing: greeks sane for an ATM call (delta ~0.5, positive vega/gamma)", () => {
  const S = 100, K = 100, T = 30 / 365, r = 0.04, q = 0.01, sigma = 0.3;
  const g = greeks({ S, K, T, r, q, sigma, type: "call" });
  assert.ok(g != null);
  assert.ok(Math.abs(g.delta - 0.5) < 0.1, `delta ${g.delta}`);
  assert.ok(g.gamma > 0 && g.vega > 0);
  assert.ok(g.theta < 0); // long calls bleed time
  // Invalid inputs return null instead of throwing.
  assert.equal(greeks({ S: -1, K, T, r, q, sigma, type: "call" }), null);
});

test("pricing: expectedMove math", () => {
  assert.ok(Math.abs(expectedMove(100, 0.365, 1) - 100 * 0.365 * Math.sqrt(1 / 365)) < 1e-9);
  assert.equal(expectedMove(100, 0, 1), null);
});
