// ============================================================================
// FRIT SIZING — the two lot modes the app exposes
// ----------------------------------------------------------------------------
// The Watchlist account card offers a checkbox: let the bot size trades (the
// Kelly engine) or pin an exact lot. Both must be honest:
//   * AUTO  -> recommendLotSize() (Kelly x risk-profile fraction, capped)
//   * FIXED -> fixedLotSizing(): the user's number is sent verbatim, with the
//              dollar risk reported from the pip table (the risk gate still
//              runs later, so an oversized lot is refused, not silently shrunk).
// These tests protect the "no more hard-capped 0.01" behaviour.
// ============================================================================

import test from "node:test";
import assert from "node:assert";
import { fixedLotSizing, riskUsdForLot, recommendLotSize } from "../src/strategy/sizing.js";

const PIP_CONFIG = {
  EURUSD: { pipSize: 0.0001, pipValue: 10 },
  XAUUSD: { pipSize: 0.01, pipValue: 1 },
};

test("fixed lot: the user's lot is used exactly, never re-sized", () => {
  const out = fixedLotSizing({
    symbol: "EURUSD", entry: 1.1000, sl: 1.0950, lot: 0.5, pipConfig: PIP_CONFIG,
  });
  assert.equal(out.mode, "fixed");
  assert.equal(out.lotSize, 0.5);
  // 50 pips x $10/pip x 0.5 lot = $250 of risk at this stop.
  assert.equal(out.riskUsd, 250);
  assert.match(out.reason, /fixed 0\.5 lot/);
});

test("fixed lot: 0.01 is still legal (the old hard cap is not the default anymore)", () => {
  const out = fixedLotSizing({ symbol: "EURUSD", entry: 1.1, sl: 1.09, lot: 0.01, pipConfig: PIP_CONFIG });
  assert.equal(out.lotSize, 0.01);
  assert.equal(out.riskUsd, 10);
});

test("fixed lot: clamps to 0.01..100 and rounds to the 0.01 broker step", () => {
  assert.equal(fixedLotSizing({ lot: 250, pipConfig: PIP_CONFIG }).lotSize, 100);
  assert.equal(fixedLotSizing({ lot: 0, pipConfig: PIP_CONFIG }).lotSize, 0.01);
  assert.equal(fixedLotSizing({ lot: -3, pipConfig: PIP_CONFIG }).lotSize, 0.01);
  assert.equal(fixedLotSizing({ lot: 0.123, pipConfig: PIP_CONFIG }).lotSize, 0.12);
  assert.equal(fixedLotSizing({ lot: 0.127, pipConfig: PIP_CONFIG }).lotSize, 0.13);
  assert.equal(fixedLotSizing({ lot: 1.999, pipConfig: PIP_CONFIG }).lotSize, 2);
  assert.equal(fixedLotSizing({ lot: "abc", pipConfig: PIP_CONFIG }).lotSize, 0.01);
});

test("fixed lot: unknown symbol risks 0 rather than inventing a number", () => {
  const out = fixedLotSizing({ symbol: "MYSTERY", entry: 5, sl: 4, lot: 1, pipConfig: PIP_CONFIG });
  assert.equal(out.lotSize, 1);
  assert.equal(out.riskUsd, 0);
  assert.ok(!out.reason.includes("risking"));
});

test("risk per lot: metals use their own pip value", () => {
  // XAUUSD: 0.01 pip size, $1 per pip -> 3.00 move = 300 pips = $300/lot.
  assert.equal(riskUsdForLot({ symbol: "XAUUSD", entry: 2400, sl: 2397, lot: 1, pipConfig: PIP_CONFIG }), 300);
  assert.equal(riskUsdForLot({ symbol: "EURUSD", entry: 1.1, sl: 1.1, lot: 1, pipConfig: PIP_CONFIG }), 0);
  assert.equal(riskUsdForLot({ symbol: "EURUSD", entry: 0, sl: 1.1, lot: 1, pipConfig: PIP_CONFIG }), 0);
});

test("auto lot: a bigger balance is allowed a bigger lot than a small one", () => {
  const args = {
    symbol: "EURUSD", entry: 1.1, sl: 1.095, tp: 1.11, rr: 2,
    winRate: 0.6, profile: "MODERATE", pipConfig: PIP_CONFIG, confidence: 75,
  };
  const small = recommendLotSize({ ...args, balance: 500 });
  const big = recommendLotSize({ ...args, balance: 50_000 });
  assert.ok(small.lotSize > 0, "a 500 account with a real edge still trades");
  assert.ok(big.lotSize > small.lotSize, "and 100x the balance is 100x the lot");
  assert.ok(big.lotSize <= 5, "hard lot ceiling still applies to the engine");
});
