// ============================================================================
// FRIT STRATEGY — deeper unit tests for the pure strategy modules
// ----------------------------------------------------------------------------
// Companion to strategy.test.mjs / strategy_money.test.mjs. Those files cover
// the happy path of the strategy layer; this file goes after the EDGES:
// threshold values that sit exactly ON a gate, empty/garbage input, zero and
// negative numbers, missing fields and the documented guard clauses.
//
// Covered here:
//   costs.js               — net-RR gates, cost model fallbacks, USD conversion
//   pocket/engine.js       — expiry profiles, pullback diagnosis, v2 signals
//   bots.js                — BOT_DEFS registry, scalping analysis, synth adapter
//   synthetics.js          — generator families and the per-family analysis
//   volumematrix.js        — VP/VWAP/CVD/RVOL and the five-timeframe matrix
//   volengine.js           — IV rank / term structure / skew strategy selector
//   longterm.js            — the daily Donchian-55 trend system
//
// Everything is hermetic: no network, no timers, no real clock dependence
// (the pocket session gate is exercised through a crypto symbol, which is
// always in session, and the wall-clock fields are only checked for shape).
// ============================================================================

import test from "node:test";
import assert from "node:assert/strict";

import { NET_RR_BLOCK, NET_RR_WEAK, getSymbolCost, costInPrice, applyCosts, costToUsd } from "../src/strategy/costs.js";
import {
  nextBoundaryMs, profileFor, assessPullback, trigMinutes, signalFromCandles,
  EXPIRY_PROFILES, PB_MAX_WAIT,
} from "../src/strategy/pocket/engine.js";
import { scalpingAnalyze, synthToSignal, BOT_DEFS } from "../src/strategy/bots.js";
import { classifyFamily, analyzeSynth } from "../src/strategy/synthetics.js";
import { volumeProfile, vwap, cvdProxy, rvol, analyzeVolumeMatrix } from "../src/strategy/volumematrix.js";
import { analyzeVol } from "../src/strategy/volengine.js";
import { analyzeLongterm } from "../src/strategy/longterm.js";

// The cost table below is asserted against the DEFAULTS, so make sure no
// deployment override is leaking in from the environment.
delete process.env.COST_OVERRIDES_JSON;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

/** Linear ramp of candles. body/wick are price units, `dir` sets the side. */
function ramp(n, start, step, { body = 0.5, wick = 0.5, volume = 100 } = {}) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const close = start + i * step;
    const open = step >= 0 ? close - body : close + body;
    out.push({ open, high: close + wick, low: open - wick, close, volume });
  }
  return out;
}

/**
 * A 24-bar bull leg with a clean swing structure:
 *   H1 = 110 (i6)  L1 = 104 (i10)  H2 = 116 (i14)  L2 = 109 (i18)  H3 = 115 (i22)
 * swingLast() finds lastHigh 116 / prevLow 104 / lastLow 109 / prevHigh 110.
 */
const SWING_LEG = [
  [100, 100.5, 99.5, 100], [100, 100.5, 99.5, 100], [100, 100.5, 99.5, 100],
  [100, 100.5, 99.5, 100], [100, 100.5, 99.5, 100], [100, 100.5, 99.5, 100],
  [109, 110, 109, 109.5],
  [105, 105.5, 104.5, 105], [105, 105.5, 104.5, 105], [105, 105.5, 104.5, 105],
  [104, 105.5, 104, 105],
  [111, 112, 111, 111.5], [113, 114, 113, 113.5], [114, 115, 109.6, 114.5],
  [115, 116, 115, 115.5],
  [110, 110.5, 109.5, 110], [110, 110.5, 109.5, 110], [110, 110.5, 109.5, 110],
  [109, 110, 109, 109.5],
  [113, 114, 113, 113.5], [114, 114.5, 113.5, 114], [114, 114.5, 113.5, 114],
  [114, 115, 113.5, 114],
  [111, 111.5, 110.5, 111],
].map(([open, high, low, close]) => ({ open, high, low, close }));

// ---------------------------------------------------------------------------
// costs.js
// ---------------------------------------------------------------------------

test("costs: a known symbol returns its table row and is not estimated", () => {
  const c = getSymbolCost("EURUSD");
  assert.equal(c.spread, 0.00016);
  assert.equal(c.slippage, 0.00005);
  assert.equal(c.commission, 0);
  assert.equal(c.estimated, false);
  assert.equal(getSymbolCost("eurusd").spread, 0.00016, "lookup is case-insensitive");
});

test("costs: an unknown symbol returns an estimated profile with no spread", () => {
  const c = getSymbolCost("NOTAREALPAIR");
  assert.equal(c.estimated, true);
  assert.equal(c.spread, null);
  assert.equal(c.slippage, null);
  assert.equal(c.commission, 0);
  assert.equal(getSymbolCost("").estimated, true);
  assert.equal(getSymbolCost(undefined).estimated, true);
});

test("costs: price friction is spread plus twice the slippage", () => {
  assert.equal(costInPrice("EURUSD", 1.1), 0.00016 + 2 * 0.00005);
  assert.equal(costInPrice("BTCUSD", 100000), 35 + 2 * 10);
  // Unknown symbol: ~2bps of price for the spread plus 1bps twice for slippage.
  assert.equal(costInPrice("NOTAREALPAIR", 100), 0.04);
});

test("costs: a non-positive reference price costs nothing on an unknown symbol", () => {
  assert.equal(costInPrice("NOTAREALPAIR", 0), 0);
  assert.equal(costInPrice("NOTAREALPAIR", -5), 0);
});

test("costs: costs widen the risk and shrink the win", () => {
  const r = applyCosts({ symbol: "EURUSD", entry: 1.1, sl: 1.098, tp: 1.104 });
  assert.ok(near(r.grossRisk, 0.002));
  assert.ok(near(r.grossWin, 0.004));
  assert.ok(near(r.netRisk, r.grossRisk + r.costPrice));
  assert.ok(near(r.netWin, r.grossWin - r.costPrice));
  assert.ok(near(r.netRR, 1.65, 1e-2));
  assert.equal(r.blocked, false);
  assert.equal(r.weak, false);
});

test("costs: a zero-risk setup is blocked instead of dividing by zero", () => {
  const r = applyCosts({ symbol: "EURUSD", entry: 1.1, sl: 1.1, tp: 1.2 });
  assert.equal(r.netRR, 0);
  assert.equal(r.blocked, true);
  assert.equal(r.weak, true);
  assert.equal(r.costPctOfRisk, null);
  // A zero target is equally degenerate.
  const noWin = applyCosts({ symbol: "EURUSD", entry: 1.1, sl: 1.09, tp: 1.1 });
  assert.equal(noWin.blocked, true);
  assert.equal(noWin.netRR, 0);
});

test("costs: exactly on the NET_RR_BLOCK threshold the trade survives", () => {
  // BTCUSD friction is exactly 55 price units, so the net R-R is exactly 1.0.
  const at = applyCosts({ symbol: "BTCUSD", entry: 100000, sl: 99000, tp: 101110 });
  assert.equal(at.netRR, NET_RR_BLOCK);
  assert.equal(at.blocked, false, "the block gate is strict: 1.0 is not below 1.0");
  assert.equal(at.weak, true);
  // One price unit less of target pushes the raw R-R under 1.0 -> blocked.
  const under = applyCosts({ symbol: "BTCUSD", entry: 100000, sl: 99000, tp: 101109 });
  assert.equal(under.blocked, true);
});

test("costs: exactly on the NET_RR_WEAK threshold the setup is not weak", () => {
  const at = applyCosts({ symbol: "BTCUSD", entry: 100000, sl: 99000, tp: 100000 + 1.3 * 1055 + 55 });
  assert.equal(at.netRR, NET_RR_WEAK);
  assert.equal(at.weak, false, "the weak gate is strict: 1.3 is not below 1.3");
  assert.equal(at.blocked, false);
  const under = applyCosts({ symbol: "BTCUSD", entry: 100000, sl: 99000, tp: 100000 + 1.29 * 1055 + 55 });
  assert.equal(under.weak, true);
});

test("costs: the reported net R-R and the gate always agree", () => {
  // F-45 regression. Raw R-R is 0.999 -> the report used to show 1.0 while
  // the gate still blocked it, so the UI and the gate disagreed. The
  // published value is now truncated (never rounded up) and the gate uses
  // that same number.
  const r = applyCosts({ symbol: "BTCUSD", entry: 100000, sl: 99000, tp: 100000 + 0.999 * 1055 + 55 });
  assert.equal(r.netRR, 0.99, "a sub-1.0 R-R must not be published as 1.0");
  assert.equal(r.blocked, true, "the gate must agree with the published number");
  assert.equal(r.netRR < 1, r.blocked, "published R-R and blocked flag must never disagree");
});

test("costs: USD friction needs the caller's pip table", () => {
  const u = costToUsd({ symbol: "EURUSD", lotSize: 0.01, pipSize: 0.0001, pipValue: 10, refPrice: 1.1 });
  assert.equal(u.spreadSlipUsd, 0.26);
  assert.equal(u.commissionUsd, 0);
  assert.equal(u.totalUsd, 0.26);
  assert.equal(u.estimated, false);
});

test("costs: without a pip table the USD totals are null, not zero", () => {
  const u = costToUsd({ symbol: "EURUSD" });
  assert.equal(u.spreadSlipUsd, null);
  assert.equal(u.totalUsd, null);
  assert.equal(u.commissionUsd, 0);
  assert.equal(u.estimated, true);
  // A zero pip size is falsy -> same "no pip table" branch.
  assert.equal(costToUsd({ symbol: "EURUSD", pipSize: 0, pipValue: 10, refPrice: 1.1 }).totalUsd, null);
  // So is a missing/zero pip value.
  assert.equal(costToUsd({ symbol: "EURUSD", pipSize: 0.0001, refPrice: 1.1 }).totalUsd, null);
  assert.equal(costToUsd({ symbol: "EURUSD", pipSize: 0.0001, pipValue: 0, refPrice: 1.1 }).totalUsd, null);
});

test("costs: COST_OVERRIDES_JSON overrides the table per symbol", async () => {
  const url = new URL("../src/strategy/costs.js", import.meta.url).href;
  const saved = process.env.COST_OVERRIDES_JSON;
  process.env.COST_OVERRIDES_JSON = JSON.stringify({ EURUSD: { spread: 0.0005, commission: 7 }, xauusd: { spread: 0.9 } });
  try {
    // A fresh module instance so the override cache is not shared.
    const fresh = await import(`${url}?override=1`);
    const eur = fresh.getSymbolCost("EURUSD");
    assert.equal(eur.spread, 0.0005, "the override wins over the table");
    assert.equal(eur.commission, 7);
    assert.equal(eur.slippage, 0.00005, "unspecified fields keep the table value");
    assert.equal(fresh.getSymbolCost("XAUUSD").spread, 0.9, "override keys are upper-cased");
    assert.equal(fresh.costInPrice("EURUSD", 1.1), 0.0005 + 2 * 0.00005);
    const u = fresh.costToUsd({ symbol: "EURUSD", lotSize: 0.02, pipSize: 0.0001, pipValue: 10, refPrice: 1.1 });
    assert.equal(u.commissionUsd, 0.14, "commission books per lot");
    assert.equal(u.totalUsd, 1.34);
  } finally {
    if (saved === undefined) delete process.env.COST_OVERRIDES_JSON;
    else process.env.COST_OVERRIDES_JSON = saved;
  }
});

test("costs: a negative spread override is floored at zero friction", async () => {
  const url = new URL("../src/strategy/costs.js", import.meta.url).href;
  const saved = process.env.COST_OVERRIDES_JSON;
  process.env.COST_OVERRIDES_JSON = JSON.stringify({ "NEGPAIR": { spread: -1 } });
  try {
    const fresh = await import(`${url}?negative=1`);
    assert.equal(fresh.getSymbolCost("NEGPAIR").spread, -1, "the override is taken verbatim");
    assert.equal(fresh.costInPrice("NEGPAIR", 100), 0, "negative friction is clamped to zero");
  } finally {
    if (saved === undefined) delete process.env.COST_OVERRIDES_JSON;
    else process.env.COST_OVERRIDES_JSON = saved;
  }
});

// ---------------------------------------------------------------------------
// pocket/engine.js
// ---------------------------------------------------------------------------

test("pocket: nextBoundaryMs snaps forward to the next interval boundary", () => {
  assert.equal(nextBoundaryMs(60, 0), 0, "t=0 is already on a boundary");
  assert.equal(nextBoundaryMs(60, 1), 60000);
  assert.equal(nextBoundaryMs(60, 59999), 60000);
  assert.equal(nextBoundaryMs(60, 60000), 60000, "a time exactly on the boundary does not roll over");
  assert.equal(nextBoundaryMs(60, 61000), 120000);
  assert.equal(nextBoundaryMs(300, 299999), 300000);
});

test("pocket: profileFor snaps an expiry down to the nearest supported profile", () => {
  assert.equal(profileFor(1).expiryMin, 1);
  assert.equal(profileFor(4).expiryMin, 1);
  assert.equal(profileFor(5).expiryMin, 5);
  assert.equal(profileFor(14).expiryMin, 5);
  assert.equal(profileFor(15).expiryMin, 15);
  assert.equal(profileFor(60).expiryMin, 15, "there is no 60m profile");
  assert.equal(profileFor(0).expiryMin, 1, "an unsupported expiry falls back to the shortest profile");
  assert.equal(profileFor(undefined).expiryMin, 1);
});

test("pocket: the expiry profiles and pullback wait budgets are the documented ones", () => {
  assert.deepEqual(Object.keys(EXPIRY_PROFILES).map(Number).sort((a, b) => a - b), [1, 5, 15]);
  assert.equal(EXPIRY_PROFILES[5].trendIv, "15m");
  assert.equal(EXPIRY_PROFILES[5].trigIv, "5m");
  assert.equal(EXPIRY_PROFILES[1].trendIv, "5m");
  assert.equal(EXPIRY_PROFILES[15].trendIv, "1h");
  assert.deepEqual(PB_MAX_WAIT, { 1: 6, 5: 4, 15: 3 });
});

test("pocket: trigMinutes maps the trigger interval to minutes and defaults to 5", () => {
  assert.equal(trigMinutes("1m"), 1);
  assert.equal(trigMinutes("5m"), 5);
  assert.equal(trigMinutes("15m"), 15);
  assert.equal(trigMinutes("4h"), 5, "an unknown interval falls back to 5 minutes");
  assert.equal(trigMinutes(undefined), 5);
});

test("pocket: a shallow, quiet pullback in a bull leg is HEALTHY", () => {
  const trig = [{ open: 111, high: 111.6, low: 110.4, close: 111 }];
  const pb = assessPullback("PB_HEALTHY", SWING_LEG, trig, 1, 1);
  assert.equal(pb.verdict, "HEALTHY");
  assert.ok(pb.retracePct > 40 && pb.retracePct < 42, `retrace ${pb.retracePct}`);
  assert.equal(pb.trendDir, 1);
  assert.ok(pb.reasons.some((r) => /shallow retrace/.test(r)));
});

test("pocket: a deep retrace with a thrust candle is a REVERSAL", () => {
  const thrust = [{ open: 104, high: 106, low: 103.5, close: 106 }]; // body 2x ATR -> thrust
  const pb = assessPullback("PB_REVERSAL", SWING_LEG, thrust, 1, 1);
  assert.equal(pb.verdict, "REVERSAL");
  assert.ok(pb.retracePct > 61.8, `retrace ${pb.retracePct}`);
  assert.ok(pb.reasons.some((r) => /61\.8/.test(r)));
});

test("pocket: a deep retrace without thrust is UNCERTAIN, not a reversal", () => {
  const quiet = [{ open: 105.9, high: 106.1, low: 105.8, close: 106 }]; // body 0.1x ATR
  const pb = assessPullback("PB_UNCERTAIN", SWING_LEG, quiet, 1, 1);
  assert.equal(pb.verdict, "UNCERTAIN");
  assert.ok(pb.reasons.some((r) => /too deep to trust/.test(r)));
});

test("pocket: a close beyond the leg extreme is a structure break", () => {
  const breakOut = [{ open: 116.5, high: 117, low: 116.4, close: 116.5 }];
  const pb = assessPullback("PB_BREAK", SWING_LEG, breakOut, 1, 1);
  assert.equal(pb.verdict, "REVERSAL");
  assert.ok(pb.reasons.some((r) => /structure break/.test(r)));
  assert.ok(pb.retracePct < 0, "a break leaves the leg, so the retrace goes negative");
});

test("pocket: too little history to contain a swing leg cannot confirm health", () => {
  const pb = assessPullback("PB_NOSWING", SWING_LEG.slice(0, 3), [{ open: 100, high: 100.5, low: 99.5, close: 100 }], 1, 1);
  assert.equal(pb.verdict, "UNCERTAIN");
  assert.equal(pb.retracePct, null);
  assert.ok(pb.reasons.some((r) => /no clean swing leg/.test(r)));
});

test("pocket: a cold ADX ring starts neutral at 50, not weak at 0", () => {
  const leg = [...Array.from({ length: 8 }, () => ({ open: 100, high: 100.5, low: 99.5, close: 100 })), ...SWING_LEG];
  const pb = assessPullback("PB_ADX_COLD", leg, [{ open: 111, high: 111.6, low: 110.4, close: 111 }], 1, 1);
  assert.equal(pb.adxPct, 50, "a cold start must not masquerade as a weak trend");
  assert.ok(!pb.reasons.some((r) => /trend weak/.test(r)));
});

test("pocket: a signal warms up until the profile's candle counts are met", () => {
  const s = signalFromCandles("BTCUSD", ramp(10, 100, 1), ramp(25, 100, 1), { cryptoSet: new Set(["BTCUSD"]), expiryMin: 5 });
  assert.equal(s.decision, "WAIT");
  assert.equal(s.confidence, 0);
  assert.equal(s.entryRef, null);
  assert.match(s.reasons[0], /warming up \(10 15m candles, need 40\)/);
  // A missing trend array is a warm-up, not a crash.
  const none = signalFromCandles("BTCUSD", null, ramp(25, 100, 1), { cryptoSet: new Set(["BTCUSD"]) });
  assert.equal(none.decision, "WAIT");
  assert.match(none.reasons[0], /warming up/);
  // The trigger side has its own minimum.
  const shortTrig = signalFromCandles("BTCUSD", ramp(40, 100, 1), ramp(10, 100, 1), { cryptoSet: new Set(["BTCUSD"]), expiryMin: 5 });
  assert.equal(shortTrig.decision, "WAIT");
  assert.match(shortTrig.reasons[0], /warming up \(10 5m candles, need 25\)/);
});

test("pocket: an aligned crypto uptrend fires a CALL", () => {
  const s = signalFromCandles("BTCUSD", ramp(40, 100, 1), ramp(25, 100, 1), { cryptoSet: new Set(["BTCUSD"]), expiryMin: 5 });
  assert.equal(s.decision, "CALL");
  assert.equal(s.profile, "15m/5m");
  assert.equal(s.expiryMin, 5);
  assert.equal(s.entryRef, 124, "entry references the last closed trigger close");
  assert.equal(s.confidence, 65, "confidence is capped until 30 paper trades exist");
  assert.equal(s.setupKey, "BTCUSD|5m|strong|247");
  assert.ok(s.timing.nextBoundaryMs % 300000 === 0, "the entry is timed on the 5-minute boundary");
  assert.ok(s.timing.msToBoundary >= 0 && s.timing.msToBoundary <= 300000);
});

test("pocket: an aligned crypto downtrend fires a PUT", () => {
  const s = signalFromCandles("BTCUSD", ramp(40, 200, -1), ramp(25, 200, -1), { cryptoSet: new Set(["BTCUSD"]), expiryMin: 5 });
  assert.equal(s.decision, "PUT");
  assert.equal(s.confidence, 65);
  assert.ok(s.reasons.some((r) => /bearish/.test(r)));
});

test("pocket: a counter-trend trigger stands down and reports the pullback read", () => {
  const s = signalFromCandles("ETHUSD", ramp(40, 100, 1), ramp(25, 200, -1), { cryptoSet: new Set(["ETHUSD"]), expiryMin: 5 });
  assert.equal(s.decision, "WAIT");
  assert.equal(s.entryRef, null);
  assert.ok(s.reasons.some((r) => /counter-trend/.test(r)));
  assert.ok(s.pullback, "a counter-trend signal must carry the pullback diagnosis");
  assert.ok(["HEALTHY", "REVERSAL", "UNCERTAIN"].includes(s.pullback.verdict));
});

test("pocket: a flatlined trigger candle is a dead market", () => {
  // Range 0.6 against a 1.5 ATR -> under the 0.5x ATR dead-market rule.
  const trig = ramp(24, 100, 1).concat([{ open: 124, high: 124.6, low: 124, close: 124.1 }]);
  const s = signalFromCandles("BTCUSD", ramp(40, 100, 1), trig, { cryptoSet: new Set(["BTCUSD"]), expiryMin: 5 });
  assert.equal(s.decision, "WAIT");
  assert.match(s.reasons.join(" "), /flatlined/);
});

test("pocket: a trigger body too weak against ATR is refused", () => {
  const trig = ramp(24, 100, 1).concat([{ open: 123.9, high: 124.6, low: 123.4, close: 124.0 }]);
  const s = signalFromCandles("BTCUSD", ramp(40, 100, 1), trig, { cryptoSet: new Set(["BTCUSD"]), expiryMin: 5 });
  assert.equal(s.decision, "WAIT");
  assert.match(s.reasons.join(" "), /body too weak/);
});

test("pocket: opposing order flow pushes the signal under the confidence floor", () => {
  const s = signalFromCandles("BTCUSD", ramp(40, 100, 1), ramp(25, 100, 1), { cryptoSet: new Set(["BTCUSD"]), expiryMin: 5, ofi: -1 });
  assert.equal(s.decision, "WAIT");
  assert.match(s.reasons.join(" "), /order-flow opposes/);
  assert.match(s.reasons.join(" "), /confidence 53 below 60 floor/);
});

test("pocket: agreeing order flow cannot lift confidence past the learning cap", () => {
  const s = signalFromCandles("BTCUSD", ramp(40, 100, 1), ramp(25, 100, 1), { cryptoSet: new Set(["BTCUSD"]), expiryMin: 5, ofi: 1 });
  assert.equal(s.decision, "CALL");
  assert.equal(s.confidence, 65, "confidence is capped at 65 while learning");
  assert.match(s.reasons.join(" "), /learning mode \(0\/30/);
});

test("pocket: a calibrated setup below the payout breakeven is auto-disabled", () => {
  const s = signalFromCandles("BTCUSD", ramp(40, 100, 1), ramp(25, 100, 1), {
    cryptoSet: new Set(["BTCUSD"]), expiryMin: 5, payout: 0.8, calib: { n: 30, wr: 0.5 },
  });
  assert.equal(s.decision, "WAIT");
  assert.match(s.reasons.join(" "), /CALIBRATION KILL/);
});

test("pocket: a calibrated setup exactly on the breakeven survives", () => {
  const payout = 0.8;
  const breakeven = 1 / (1 + payout);
  const s = signalFromCandles("BTCUSD", ramp(40, 100, 1), ramp(25, 100, 1), {
    cryptoSet: new Set(["BTCUSD"]), expiryMin: 5, payout, calib: { n: 30, wr: breakeven + 0.01 },
  });
  assert.equal(s.decision, "CALL");
  assert.equal(s.confidence, 60, "the calibrated confidence lands exactly on the floor");
  assert.match(s.reasons.join(" "), /calibrated/);
});

test("pocket: the calibrated confidence respects the documented 78 cap", () => {
  // F-45 regression. The MAX_CONF=78 clamp was applied BEFORE the
  // calibration blend and never re-applied, so a perfect calibration record
  // published a confidence of 89 while the module header promises a hard
  // cap of 78. The cap is now re-applied after the blend.
  const s = signalFromCandles("BTCUSD", ramp(40, 100, 1), ramp(25, 100, 1), {
    cryptoSet: new Set(["BTCUSD"]), expiryMin: 5, ofi: 1, calib: { n: 30, wr: 1 },
  });
  assert.equal(s.decision, "CALL");
  assert.ok(s.confidence <= 78, `confidence ${s.confidence} must not exceed the documented cap of 78`);
  assert.ok(s.confidence >= 20, "confidence must stay within the floor");
});

test("pocket: a non-crypto symbol is gated by the trading session", () => {  const s = signalFromCandles("EURUSD", ramp(40, 100, 1), ramp(25, 100, 1), { cryptoSet: new Set(["BTCUSD"]), expiryMin: 5 });
  assert.ok(
    s.reasons.some((r) => /weekend|liquidity|London\/NY session/.test(r)),
    `expected a session reason, got ${JSON.stringify(s.reasons)}`,
  );
});

// ---------------------------------------------------------------------------
// bots.js
// ---------------------------------------------------------------------------

test("bots: the registry exposes five bots whose default timeframe is one of theirs", () => {
  assert.deepEqual(Object.keys(BOT_DEFS).sort(), ["intraday", "longterm", "options", "pocket", "scalping"]);
  for (const [name, def] of Object.entries(BOT_DEFS)) {
    assert.ok(def.label, `${name} needs a label`);
    assert.ok(Array.isArray(def.timeframes) && def.timeframes.length, `${name} needs timeframes`);
    assert.ok(def.timeframes.includes(def.defaultTimeframe), `${name} default timeframe must be supported`);
    assert.equal(typeof def.intervalMin, "number");
    assert.equal(typeof def.executes, "boolean");
    assert.ok(def.blurb.length > 20, `${name} needs a blurb`);
  }
});

test("bots: the options bot is signal-only and the long-term bot runs daily", () => {
  assert.equal(BOT_DEFS.options.executes, false, "the options bot never places orders");
  assert.deepEqual(BOT_DEFS.options.timeframes, ["1D"]);
  assert.equal(BOT_DEFS.longterm.intervalMin, 1440);
  assert.deepEqual(BOT_DEFS.longterm.timeframes, ["1D", "1W"]);
  assert.equal(BOT_DEFS.scalping.defaultTimeframe, "5m");
  assert.equal(BOT_DEFS.intraday.defaultTimeframe, "30m");
});

test("bots: scalping analysis warms up below 30 candles", () => {
  const warm = scalpingAnalyze("EURUSD", ramp(29, 100, 1));
  assert.equal(warm.decision, "WAIT");
  assert.equal(warm.confidence, 0);
  assert.equal(warm.entry, null);
  assert.match(warm.reasons.join(" "), /warming up \(29 candles, need 30\+\)/);
});

test("bots: scalping analysis prices a long with the stop below and the targets above", () => {
  const up = scalpingAnalyze("EURUSD", ramp(40, 100, 1), "5m");
  assert.equal(up.decision, "BUY");
  assert.equal(up.symbol, "EURUSD");
  assert.equal(up.entry, 139);
  assert.ok(up.sl < up.entry, "stop below entry for a long");
  assert.ok(up.tp1 > up.entry && up.tp2 > up.tp1, "target 2 runs before target 3");
  assert.ok(near(up.rr, 2 / 1.5, 1e-9), "the published R-R is the 2xATR/1.5xATR ratio");
  assert.ok(up.confidence >= 55 && up.confidence <= 75, `confidence ${up.confidence}`);
  assert.match(up.reasons.join(" "), /trend bullish/);
});

test("bots: scalping analysis prices a short with the stop above and the targets below", () => {
  const down = scalpingAnalyze("EURUSD", ramp(40, 200, -1), "5m");
  assert.equal(down.decision, "SELL");
  assert.ok(down.sl > down.entry, "stop above entry for a short");
  assert.ok(down.tp1 < down.entry && down.tp2 < down.tp1);
  assert.match(down.reasons.join(" "), /trend bearish/);
});

test("bots: scalping analysis refuses a flatlined candle and a body too weak against ATR", () => {
  const flat = ramp(39, 100, 1).concat([{ open: 139, high: 139.6, low: 139, close: 139.1, volume: 100 }]);
  assert.match(scalpingAnalyze("EURUSD", flat).reasons.join(" "), /flatlined candle/);
  const weak = ramp(39, 100, 1).concat([{ open: 139, high: 139.5, low: 138.5, close: 139.05, volume: 100 }]);
  assert.match(scalpingAnalyze("EURUSD", weak).reasons.join(" "), /last body too weak vs ATR/);
});

test("bots: scalping analysis refuses a setup whose confidence is under the floor", () => {
  // A slow ramp inside wide bars: the EMA separation is too small for the
  // bonus (+10) and RSI 100 gives no "room to run" bonus (+5) -> conf 50.
  const candles = [];
  for (let i = 0; i < 40; i++) {
    const close = 100 + i * 0.1;
    const open = close - 5;
    candles.push({ open, high: close + 10, low: open - 10, close, volume: 100 });
  }
  const out = scalpingAnalyze("EURUSD", candles);
  assert.equal(out.decision, "WAIT");
  assert.equal(out.confidence, 0);
  assert.equal(out.entry, null);
  assert.match(out.reasons.join(" "), /confidence 50 below 55 floor/);
});

test("bots: scalping analysis refuses a dead tape whose indicators are not ready", () => {
  const dead = Array.from({ length: 30 }, () => ({ open: 1, high: 1, low: 1, close: 1, volume: 1 }));
  const out = scalpingAnalyze("EURUSD", dead);
  assert.equal(out.decision, "WAIT");
  assert.match(out.reasons.join(" "), /indicators not ready/);
});

test("bots: synthToSignal adapts a synth result into the executable signal shape", () => {
  const s = synthToSignal("R_100", {
    decision: "SELL", confidence: 71, entry: 100, sl: 110, tp: 80, rr: 2, reasons: ["fade the spike", "stop beyond"], maxHoldMin: 30,
  });
  assert.equal(s.symbol, "R_100");
  assert.equal(s.decision, "SELL");
  assert.equal(s.confidence, 71);
  assert.equal(s.entry, 100);
  assert.equal(s.sl, 110);
  assert.equal(s.tp1, 80, "the synth target becomes tp1");
  assert.equal(s.tp2, null);
  assert.equal(s.rr, 2);
  assert.equal(s.scenario, "SYNTH");
  assert.equal(s.reasoning, "fade the spike | stop beyond | time stop 30min");
  assert.ok(s.id.startsWith("sig_R_100_"));
  assert.ok(s.id.endsWith("_sy"));
  assert.equal(typeof s.timestamp, "number");
});

test("bots: synthToSignal downgrades anything that is not BUY or SELL to WAIT", () => {
  const s = synthToSignal("R_100", { decision: "WAIT", confidence: 0, reasons: ["need 80+ closes for vol bands"] });
  assert.equal(s.decision, "WAIT");
  assert.equal(s.confidence, 0);
  assert.equal(s.entry, undefined);
  assert.equal(s.rr, null);
  assert.equal(s.reasoning, "need 80+ closes for vol bands", "a WAIT keeps its reason and no time stop");
});

// ---------------------------------------------------------------------------
// synthetics.js
// ---------------------------------------------------------------------------

test("synthetics: the Deriv generators map onto their families", () => {
  assert.equal(classifyFamily("R_100"), "VOL");
  assert.equal(classifyFamily("1HZ100V"), "VOL");
  assert.equal(classifyFamily("STPRNG"), "VOL");
  assert.equal(classifyFamily("r_100"), "VOL", "classification is case-insensitive");
  assert.equal(classifyFamily("BOOM500N"), "SPIKE");
  assert.equal(classifyFamily("CRASH1000N"), "SPIKE");
  assert.equal(classifyFamily("JD10"), "SPIKE");
  assert.equal(classifyFamily("GDA20"), "SPIKE");
  assert.deepEqual(classifyFamily("RDBULL"), { family: "TREND", lock: "BUY" });
  assert.deepEqual(classifyFamily("RDBEAR"), { family: "TREND", lock: "SELL" });
  assert.deepEqual(classifyFamily("DEX20UP"), { family: "TREND", lock: "BUY" });
  assert.deepEqual(classifyFamily("DEX20DN"), { family: "TREND", lock: "SELL" });
  assert.equal(classifyFamily("EURUSD"), "UNKNOWN");
  assert.equal(classifyFamily(""), "UNKNOWN");
});

test("synthetics: BOOM and CRASH codes need their trailing N", () => {
  assert.equal(classifyFamily("BOOM500"), "UNKNOWN", "BOOM500 is not a real Deriv code");
  assert.equal(classifyFamily("CRASH25"), "UNKNOWN");
  assert.equal(classifyFamily("BOOM500N"), "SPIKE");
});

test("synthetics: the DSI regime length is clamped into 5..60 minutes", () => {
  assert.deepEqual(classifyFamily("DSI10"), { family: "REGIME", regimeMin: 10 });
  assert.equal(classifyFamily("DSI1").regimeMin, 5, "clamped up to the 5-minute floor");
  assert.equal(classifyFamily("DSI100").regimeMin, 60, "clamped down to the 60-minute ceiling");
  assert.equal(classifyFamily("DSI30").regimeMin, 30);
});

test("synthetics: an unknown generator is never traded", () => {
  const out = analyzeSynth("EURUSD", Array.from({ length: 200 }, (_, i) => 100 + i));
  assert.equal(out.decision, "WAIT");
  assert.equal(out.family, "UNKNOWN");
  assert.equal(out.strategy, "synth_unknown");
  assert.equal(out.entry, null);
  assert.match(out.reasons.join(" "), /unknown generator/);
});

test("synthetics: the VOL family fades a stretched price back to the mean", () => {
  const closes = Array.from({ length: 79 }, () => 100).concat([130]);
  const out = analyzeSynth("R_100", closes);
  assert.equal(out.decision, "SELL", "price far above the mean -> fade down");
  assert.equal(out.entry, 130);
  assert.ok(out.sl > out.entry, "the stop sits above entry for a short");
  assert.ok(near(out.tp, 101.5, 1e-9), "the target is the rolling mean");
  assert.equal(out.maxHoldMin, 30, "every VOL trade carries a time stop");
  assert.equal(out.confidence, 80);
  assert.match(out.reasons.join(" "), /VOL fade/);
});

test("synthetics: the VOL family refuses a dead tape and refuses to fade inside the band", () => {
  const dead = analyzeSynth("R_100", Array.from({ length: 100 }, () => 100));
  assert.equal(dead.decision, "WAIT");
  assert.match(dead.reasons.join(" "), /zero variance window/);
  const inside = Array.from({ length: 60 }, () => 90).concat([99, 100, 101, 102, 99, 100, 101, 102, 99, 100, 101, 102, 99, 100, 101, 102, 99, 100, 101, 102]);
  const band = analyzeSynth("R_100", inside);
  assert.equal(band.decision, "WAIT");
  assert.match(band.reasons.join(" "), /inside 2 — no extreme/);
});

test("synthetics: the VOL family needs 80 closes and refuses a volatility squeeze", () => {
  const short = analyzeSynth("R_100", Array.from({ length: 79 }, () => 100));
  assert.equal(short.decision, "WAIT");
  assert.match(short.reasons.join(" "), /need 80\+ closes for vol bands/);
  // Wild history, dead tail -> the squeeze guard vetoes the fade.
  // The current 20-bar volatility is ~3% of the trailing 120-bar volatility.
  const wild = Array.from({ length: 100 }, (_, i) => (i % 2 ? 110 : 90));
  const tail = Array.from({ length: 20 }, (_, i) => (i % 2 ? 100.5 : 100));
  const squeeze = analyzeSynth("R_100", [...wild, ...tail]);
  assert.equal(squeeze.decision, "WAIT");
  assert.match(squeeze.reasons.join(" "), /volatility squeeze/);
});

test("synthetics: the SPIKE family only trades a real spike and fades it back", () => {
  const thin = analyzeSynth("BOOM100N", Array.from({ length: 59 }, (_, i) => 100 + i));
  assert.equal(thin.decision, "WAIT");
  assert.match(thin.reasons.join(" "), /need 60\+ ticks for spike detection/);
  const quiet = Array.from({ length: 80 }, (_, i) => 100 + i);
  const none = analyzeSynth("BOOM100N", quiet);
  assert.equal(none.decision, "WAIT");
  assert.match(none.reasons.join(" "), /no spike \(last move 1\.0x median, need 8x\)/);
  const ticks = Array.from({ length: 60 }, (_, i) => 100 + i).concat([169]);
  const spike = analyzeSynth("BOOM100N", ticks);
  assert.equal(spike.decision, "SELL", "an upward spike is faded down");
  assert.equal(spike.entry, 169);
  assert.equal(spike.tp, 159, "the target is the pre-spike level");
  assert.ok(spike.sl > spike.entry, "the stop sits beyond the spike extreme");
  assert.equal(spike.maxHoldMin, 15);
});

test("synthetics: the REGIME family follows a significant drift with a half-life time stop", () => {
  const closes = Array.from({ length: 20 }, () => 100).concat([100, 100.1, 101, 102.5, 104]);
  const out = analyzeSynth("DSI10", closes);
  assert.equal(out.decision, "BUY");
  assert.equal(out.maxHoldMin, 5, "the time stop is half the average regime life");
  assert.equal(out.rr, 2);
  assert.ok(out.sl < out.entry && out.tp > out.entry);
  assert.match(out.reasons.join(" "), /DSI regime read/);
  const short = analyzeSynth("DSI10", Array.from({ length: 19 }, () => 100));
  assert.equal(short.decision, "WAIT");
  assert.match(short.reasons.join(" "), /need 20\+ 1m closes/);
});

test("synthetics: the TREND family is direction-locked and never fades", () => {
  const rising = Array.from({ length: 40 }, (_, i) => 100 + i);
  const bull = analyzeSynth("RDBULL", rising);
  assert.equal(bull.decision, "BUY");
  assert.equal(bull.confidence, 65);
  assert.equal(bull.maxHoldMin, 240);
  const falling = Array.from({ length: 40 }, (_, i) => 200 - i);
  const bear = analyzeSynth("RDBEAR", falling);
  assert.equal(bear.decision, "SELL");
  // F-45 regression: the SELL branch used to carry no `details` object while
  // the matching BUY branch did, so every short came back `details: undefined`.
  assert.ok(bear.details, "the SELL branch must carry a details object like the BUY branch");
  assert.equal(typeof bear.details.donchianHigh, "number");
  assert.equal(typeof bear.details.donchianLow, "number");
  assert.equal(typeof bear.details.atr, "number");
  const wrongWay = analyzeSynth("RDBULL", falling);
  assert.equal(wrongWay.decision, "WAIT", "a locked-long family never shorts");
  assert.match(wrongWay.reasons.join(" "), /no upside breakout/);
});

// ---------------------------------------------------------------------------
// volumematrix.js
// ---------------------------------------------------------------------------

test("volumematrix: the volume profile returns null when there is nothing to profile", () => {
  assert.equal(volumeProfile([]), null);
  assert.equal(volumeProfile(null), null);
  assert.equal(volumeProfile([{ high: 100, low: 100, volume: 10 }]), null, "a zero-range profile is null");
  assert.equal(volumeProfile([{ high: 101, low: 100, volume: 0 }]), null, "zero volume is null");
});

test("volumematrix: the value area extends below a POC that sits on the top bucket", () => {
  const candles = [
    { high: 2, low: 0, volume: 10 },
    { high: 4, low: 2, volume: 10 },
    { high: 5, low: 4, volume: 20 },
  ];
  const vp = volumeProfile(candles, { bucketSize: 1 });
  assert.equal(vp.poc, 4, "the heaviest bucket is the top one");
  assert.equal(vp.valueAreaHigh, 4);
  assert.equal(vp.valueAreaLow, 2, "the walk must reach the 70% volume target downwards");
  assert.equal(vp.totalVolume, 40);
  assert.equal(vp.approximation, "candle-distributed (not tick VP)");
});

test("volumematrix: total volume is conserved even when float drift would add a bucket", () => {
  // F-45 regression. step = 0.1 made the bucket walk visit 11 points while the
  // divisor said 10, so the profile reported 110 for a candle that traded
  // 100 — and the top bucket got double volume, skewing the 70% value-area
  // walk and the POC anchor. The walk is now index-based.
  const vp = volumeProfile([{ open: 0, high: 1, low: 0, close: 0.5, volume: 100 }], { levels: 10 });
  assert.equal(vp.totalVolume, 100, "the profile must report exactly the volume that traded");
});

test("volumematrix: a profile that divides evenly conserves the traded volume", () => {
  assert.equal(volumeProfile([{ open: 0, high: 0.5, low: 0, close: 0.25, volume: 100 }], { levels: 5 }).totalVolume, 100);
  assert.equal(volumeProfile([{ open: 0, high: 0.9, low: 0, close: 0.45, volume: 100 }], { levels: 7 }).totalVolume, 100);
});

test("volumematrix: vwap is the volume-weighted typical price", () => {
  assert.equal(vwap([{ high: 3, low: 1, close: 2, volume: 10 }]), 2);
  assert.equal(vwap([{ high: 3, low: 1, close: 2, volume: 10 }, { high: 6, low: 4, close: 5, volume: 30 }]), 4.25);
  assert.equal(vwap([]), null);
  assert.equal(vwap(null), null);
  assert.equal(vwap([{ high: 3, low: 1, close: 2, volume: 0 }]), null, "no volume means no VWAP");
});

test("volumematrix: the CVD proxy sums close location times volume over the last bars", () => {
  const up = { open: 10, high: 12, low: 10, close: 12, volume: 100 };
  const down = { open: 12, high: 12, low: 10, close: 10, volume: 50 };
  const flat = { open: 10, high: 10, low: 10, close: 10, volume: 70 };
  assert.equal(cvdProxy([up, down]), 50, "100 up minus 50 down");
  assert.equal(cvdProxy([up, down, flat]), 50, "a zero-range bar contributes nothing");
  assert.equal(cvdProxy([up, down, flat], 1), 0, "the bar limit is honoured");
  assert.equal(cvdProxy([up]), null, "one bar is not enough for a delta read");
  assert.equal(cvdProxy([]), null);
});

test("volumematrix: rvol compares the last bar with the mean of the prior window", () => {
  const bars = Array.from({ length: 22 }, () => ({ volume: 100 }));
  bars[21].volume = 200;
  assert.equal(rvol(bars), 2);
  assert.equal(rvol(bars.slice(0, 21)), null, "21 bars are not enough for a 20-bar window plus the current one");
  assert.equal(rvol([]), null);
  const zero = Array.from({ length: 22 }, () => ({ volume: 0 }));
  assert.equal(rvol(zero), null, "a zero average has no relative volume");
});

test("volumematrix: a matrix with no candle data is neutral and says so", () => {
  const m = analyzeVolumeMatrix("eurusd", {});
  assert.equal(m.symbol, "EURUSD");
  assert.equal(m.bias, "NEUTRAL");
  assert.equal(m.confidence, 0);
  assert.deepEqual(m.why, ["no candle data for any matrix timeframe"]);
});

test("volumematrix: a matrix with only a 4H leg computes nothing but does not lie", () => {
  const m = analyzeVolumeMatrix("EURUSD", { "4h": [{ high: 101, low: 99, close: 100, volume: 10 }] });
  assert.equal(m.bias, "NEUTRAL");
  assert.equal(m.confidence, 30);
  assert.equal(m.score, 0);
  assert.equal(m.legs.avp, "ok");
  assert.equal(m.legs.svp, "insufficient");
  assert.equal(m.legs.vwap, null);
  assert.equal(m.legs.rvol1mQuality, "none");
  assert.deepEqual(m.why, ["matrix computed but no consensus"]);
});

test("volumematrix: price above the session VWAP with buy aggression reads BULL", () => {
  const up = Array.from({ length: 40 }, (_, i) => ({
    open: 100 + i * 0.1, high: 100.5 + i * 0.1, low: 99.5 + i * 0.1, close: 100.4 + i * 0.1, volume: 100,
  }));
  const m = analyzeVolumeMatrix("EURUSD", { "4h": up, "30m": up, "15m": up, "5m": up, "1m": up });
  assert.equal(m.bias, "BULL");
  assert.equal(m.score, 5);
  assert.equal(m.confidence, 80);
  assert.ok(m.why.some((w) => /price above session VWAP/.test(w)));
  assert.ok(m.why.some((w) => /buy aggression/.test(w)));
  assert.equal(m.legs.cvdIsProxy, true, "the CVD leg is always labelled a proxy");
  assert.equal(m.legs.rvol1mQuality, "1m");
});

test("volumematrix: a falling tape reads BEAR", () => {
  const down = Array.from({ length: 40 }, (_, i) => ({
    open: 140 - i * 0.1, high: 140.5 - i * 0.1, low: 139.5 - i * 0.1, close: 139.6 - i * 0.1, volume: 100,
  }));
  const m = analyzeVolumeMatrix("EURUSD", { "30m": down, "15m": down, "5m": down, "1m": down });
  assert.equal(m.bias, "BEAR");
  assert.ok(m.score < 0);
  assert.ok(m.why.some((w) => /price below session VWAP/.test(w)));
  assert.ok(m.why.some((w) => /sell aggression/.test(w)));
});

test("volumematrix: the 1M RVOL leg is flagged as a fallback below 22 bars", () => {
  const bars = (n) => Array.from({ length: n }, () => ({ open: 1, high: 2, low: 0, close: 1, volume: 100 }));
  assert.equal(analyzeVolumeMatrix("X", { "1m": bars(21) }).legs.rvol1mQuality, "fallback");
  assert.equal(analyzeVolumeMatrix("X", { "1m": bars(21) }).legs.rvol1m, null, "no RVOL without a full window");
  assert.equal(analyzeVolumeMatrix("X", { "1m": bars(22) }).legs.rvol1mQuality, "1m");
});

// ---------------------------------------------------------------------------
// volengine.js
// ---------------------------------------------------------------------------

const IV_HIST = Array.from({ length: 40 }, (_, i) => 0.1 + (i % 20) * 0.005); // 0.10 .. 0.195
// A history whose span is exactly 1.0 so the rank boundaries land on exact
// doubles (40 / 39 / 25) instead of a hair above them.
const IV_HIST_EXACT = Array.from({ length: 40 }, (_, i) => 0.25 + (i % 21) * 0.05); // 0.25 .. 1.25
const ivAtRank = (pct) => 0.25 + pct / 100;

test("volengine: without a usable tenor the answer is NONE, never a guess", () => {
  const out = analyzeVol();
  assert.equal(out.strategy, "NONE");
  assert.equal(out.regime, "UNKNOWN");
  assert.equal(out.ivRank, null);
  assert.match(out.reasons.join(" "), /no clean expiry tenor/);
});

test("volengine: tenors under 7 DTE or with a zero IV are filtered out of the anchor", () => {
  const tooShort = analyzeVol({ expiries: [{ expiration: "a", dte: 3, atmIv: 0.3 }], ivHistory: IV_HIST });
  assert.equal(tooShort.strategy, "NONE");
  assert.match(tooShort.reasons.join(" "), /no clean expiry tenor/);
  const noIv = analyzeVol({ expiries: [{ expiration: "a", dte: 30, atmIv: 0 }], ivHistory: IV_HIST });
  assert.equal(noIv.strategy, "NONE");
});

test("volengine: IV rank exactly 40 sells verticals, 39 does nothing", () => {
  const at40 = analyzeVol({ expiries: [{ expiration: "a", dte: 30, atmIv: ivAtRank(40) }], ivHistory: IV_HIST_EXACT });
  assert.equal(at40.ivRank, 40);
  assert.equal(at40.strategy, "SELL_VERTICAL");
  assert.equal(at40.regime, "HIGH_IV");
  const at39 = analyzeVol({ expiries: [{ expiration: "a", dte: 30, atmIv: ivAtRank(39) }], ivHistory: IV_HIST_EXACT });
  assert.equal(at39.ivRank, 39);
  assert.equal(at39.strategy, "NONE");
  assert.equal(at39.regime, "NORMAL");
});

test("volengine: IV rank 25 or below buys a straddle", () => {
  const at25 = analyzeVol({ expiries: [{ expiration: "a", dte: 30, atmIv: ivAtRank(25) }], ivHistory: IV_HIST_EXACT });
  assert.equal(at25.ivRank, 25);
  assert.equal(at25.strategy, "LONG_STRADDLE");
  assert.equal(at25.regime, "LOW_IV");
  const at24 = analyzeVol({ expiries: [{ expiration: "a", dte: 30, atmIv: ivAtRank(24) }], ivHistory: IV_HIST_EXACT });
  assert.equal(at24.strategy, "LONG_STRADDLE");
});

test("volengine: backwardation is a stress veto that beats an event inside the window", () => {
  const stressed = analyzeVol({
    expiries: [{ expiration: "a", dte: 30, atmIv: 0.30 }, { expiration: "b", dte: 60, atmIv: 0.20 }],
    ivHistory: IV_HIST, rv20: 0.18,
  });
  assert.equal(stressed.regime, "STRESS");
  assert.equal(stressed.strategy, "NONE");
  assert.ok(stressed.termSlope < -0.02);
  assert.match(stressed.reasons.join(" "), /backwardation veto/);
  const stressedWithEvent = analyzeVol({
    expiries: [{ expiration: "a", dte: 30, atmIv: 0.30 }, { expiration: "b", dte: 60, atmIv: 0.20 }],
    ivHistory: IV_HIST, rv20: 0.18, eventDays: { within7d: true, name: "FOMC" },
  });
  assert.equal(stressedWithEvent.strategy, "NONE", "stress outranks the event play");
  assert.equal(stressedWithEvent.regime, "STRESS");
  assert.match(stressedWithEvent.reasons.join(" "), /FOMC/);
});

test("volengine: an event inside the holding window turns the plan into a long straddle", () => {
  const out = analyzeVol({
    expiries: [{ expiration: "a", dte: 30, atmIv: 0.30 }],
    ivHistory: IV_HIST, rv20: 0.2, eventDays: { within21d: true, name: "CPI" },
  });
  assert.equal(out.strategy, "LONG_STRADDLE");
  assert.equal(out.regime, "EVENT");
  assert.match(out.reasons.join(" "), /CPI inside the holding window/);
});

test("volengine: without IV history a rich premium is judged against realized vol", () => {
  const rich = analyzeVol({ expiries: [{ expiration: "a", dte: 30, atmIv: 0.30 }], ivHistory: [], rv20: 0.20 });
  assert.equal(rich.ivRank, null);
  assert.equal(rich.ivHistoryN, 0);
  assert.equal(rich.strategy, "SELL_VERTICAL");
  assert.match(rich.reasons.join(" "), /no IV history — IV 30\.0% vs realized 20\.0%/);
});

test("volengine: without IV history a cheap premium is bought, not called fair", () => {
  // F-45 regression. IV half of realized vol: the cheap side had NO
  // fallback branch, so `lowIv` was hard-coded false and a 50% discount was
  // reported as "fair premium, no edge shape" — making LONG_STRADDLE
  // unreachable, even though the module header says "buy when low".
  const cheap = analyzeVol({ expiries: [{ expiration: "a", dte: 30, atmIv: 0.10 }], ivHistory: [], rv20: 0.20 });
  assert.equal(cheap.ivRank, null);
  assert.equal(cheap.strategy, "LONG_STRADDLE", "a 50% discount to realized vol is an edge, not a fair premium");
  assert.equal(cheap.regime, "LOW_IV");
});

test("volengine: 25-delta skew is reported only when both wings are quoted", () => {
  const both = analyzeVol({
    expiries: [{ expiration: "a", dte: 30, atmIv: 0.30, put25dIv: 0.35, call25dIv: 0.30 }],
    ivHistory: IV_HIST,
  });
  assert.equal(both.skew25d, 0.05);
  assert.match(both.reasons.join(" "), /25Δ skew \+5\.0pts/);
  const oneWing = analyzeVol({
    expiries: [{ expiration: "a", dte: 30, atmIv: 0.30, put25dIv: 0.35 }],
    ivHistory: IV_HIST,
  });
  assert.equal(oneWing.skew25d, null);
});

test("volengine: a tenor gap under 14 days produces no term slope", () => {
  const close = analyzeVol({
    expiries: [{ expiration: "a", dte: 30, atmIv: 0.30 }, { expiration: "b", dte: 40, atmIv: 0.25 }],
    ivHistory: IV_HIST,
  });
  assert.equal(close.termSlope, null);
  const wide = analyzeVol({
    expiries: [{ expiration: "a", dte: 30, atmIv: 0.30 }, { expiration: "b", dte: 60, atmIv: 0.25 }],
    ivHistory: IV_HIST,
  });
  assert.ok(wide.termSlope < 0, "a 30-day gap is enough for a slope");
});

// ---------------------------------------------------------------------------
// longterm.js
// ---------------------------------------------------------------------------

test("longterm: fewer than 210 daily closes is an honest WAIT", () => {
  const out = analyzeLongterm("EURUSD", Array.from({ length: 209 }, (_, i) => 100 + i * 0.01));
  assert.equal(out.decision, "WAIT");
  assert.equal(out.confidence, 0);
  assert.equal(out.entry, null);
  assert.equal(out.details, null);
  assert.match(out.reasons.join(" "), /210\+/);
});

test("longterm: a non-positive close invalidates the whole history", () => {
  const closes = Array.from({ length: 210 }, (_, i) => 100 + i * 0.05);
  closes[5] = 0;
  const out = analyzeLongterm("EURUSD", closes);
  assert.equal(out.decision, "WAIT");
  assert.match(out.reasons.join(" "), /210\+/);
  assert.equal(analyzeLongterm("EURUSD", null).decision, "WAIT");
  assert.equal(analyzeLongterm("EURUSD", []).decision, "WAIT");
});

test("longterm: a clean uptrend fires a 2R long with a 2xATR stop", () => {
  const closes = Array.from({ length: 210 }, (_, i) => 100 + i * 0.05);
  const out = analyzeLongterm("eurusd", closes);
  assert.equal(out.symbol, "EURUSD");
  assert.equal(out.decision, "BUY");
  assert.equal(out.regime, "BULL");
  assert.equal(out.entry, 110.45);
  assert.ok(near(out.sl, out.entry - 2 * out.details.atr14, 1e-9), "stop is 2xATR below entry");
  assert.ok(near(out.tp, out.entry + 4 * out.details.atr14, 1e-9), "target is 4xATR above entry");
  assert.equal(out.rr, 2);
  assert.equal(out.details.crossFresh, false, "a persistent alignment is not a fresh cross");
  assert.ok(out.confidence >= 20 && out.confidence <= 90);
});

test("longterm: a clean downtrend fires a 2R short", () => {
  const closes = Array.from({ length: 210 }, (_, i) => 200 - i * 0.05);
  const out = analyzeLongterm("EURUSD", closes);
  assert.equal(out.decision, "SELL");
  assert.equal(out.regime, "BEAR");
  assert.ok(out.sl > out.entry, "stop above entry for a short");
  assert.ok(out.tp < out.entry, "target below entry for a short");
  assert.equal(out.rr, 2);
  assert.equal(out.details.rsi14, 0);
});

test("longterm: a pullback inside the 55-day range waits instead of entering", () => {
  const closes = Array.from({ length: 200 }, (_, i) => 100 + i).concat(Array.from({ length: 10 }, (_, i) => 300 - i));
  const out = analyzeLongterm("EURUSD", closes);
  assert.equal(out.decision, "WAIT");
  assert.equal(out.confidence, 0);
  assert.equal(out.entry, null);
  assert.equal(out.sl, null);
  assert.equal(out.tp, null);
  assert.match(out.reasons.join(" "), /inside the 55-day range/);
});

test("longterm: the RSI guard vetoes a long that broke out anyway", () => {
  // 195 rising bars then 15 bars fading by 0.05% — RSI 0, but the close is
  // still within 0.1% of the 55-day high, so the Donchian breakout is real.
  const closes = Array.from({ length: 195 }, (_, i) => 50 + i * (49 / 195))
    .concat(Array.from({ length: 15 }, (_, i) => 99 - i * 0.0035));
  const out = analyzeLongterm("EURUSD", closes);
  assert.equal(out.decision, "WAIT", "RSI 0 must veto the long");
  assert.equal(out.details.rsi14, 0);
  assert.equal(out.confidence, 0);
  assert.equal(out.entry, null);
});

test("longterm: the RSI guard vetoes a short that broke down anyway", () => {
  // 195 falling bars, then a 0.1%-wide zigzag that lifts RSI over 65 while
  // the close still sits on the 55-day low -> the breakdown is a short, but
  // shorting into strength is vetoed.
  const closes = Array.from({ length: 195 }, (_, i) => 200 - i * 0.5);
  const band = [100];
  for (let k = 1; k <= 14; k++) band.push(band[k - 1] + (k % 2 ? 0.02 : -0.01));
  const out = analyzeLongterm("EURUSD", closes.concat(band));
  assert.equal(out.decision, "WAIT");
  assert.ok(out.details.rsi14 >= 65, `RSI ${out.details.rsi14} must be over the short guard`);
  assert.equal(out.details.donchianLow, 100);
  assert.equal(out.entry, null);
});

test("longterm: a golden cross inside the last 10 sessions is a fresh cross", () => {
  // 202 flat closes, then an 8-bar step up: the 50/200 cross happened a few
  // sessions ago, so it counts as fresh (the +10 confidence bonus).
  const closes = Array.from({ length: 202 }, () => 100).concat(Array.from({ length: 8 }, () => 150));
  const out = analyzeLongterm("EURUSD", closes);
  assert.equal(out.decision, "BUY");
  assert.equal(out.details.crossFresh, true);
  assert.ok(out.reasons.some((r) => /fresh cross ≤10d/.test(r)));
  assert.equal(out.confidence, 90, "confidence is clamped at 90");
});

test("longterm: true highs and lows widen the ATR and therefore the stop", () => {
  const closes = Array.from({ length: 210 }, (_, i) => 100 + i * 0.05);
  const highs = closes.map((c) => c + 1);
  const lows = closes.map((c) => c - 1);
  const withHl = analyzeLongterm("EURUSD", closes, { highs, lows });
  const withoutHl = analyzeLongterm("EURUSD", closes);
  assert.equal(withHl.decision, "BUY");
  assert.ok(withHl.details.atr14 > withoutHl.details.atr14, "true ranges give a bigger ATR");
  assert.ok(withHl.sl < withoutHl.sl, "the wider ATR pushes the stop further away");
  assert.ok(near(withHl.sl, withHl.entry - 2 * withHl.details.atr14, 1e-9));
});
