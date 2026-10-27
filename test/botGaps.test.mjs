// ============================================================================
// THE THREE BOT GAPS — regression tests
// ----------------------------------------------------------------------------
// 1. COST GATE. scalpingAnalyze() published a 1.33 reward:risk that ignored
//    friction entirely. On 1m/5m the round-trip cost is a large share of a
//    2xATR target, so a "2:1 setup" could be negative-expectancy by
//    construction. Every setup is now priced before it is emitted.
// 2. SYNTHETIC ROUTING. Only the intraday bot asked the generator-aware
//    engine. Scalping and pocket ran EMA9/21/RSI/ATR on an engineered RNG
//    stream — a coin flip minus costs. All three route now.
// 3. VOLUME MATRIX. The 5-timeframe bias filter was wired into the
//    mtfAnalyze path only, so a scalp could enter straight against strong
//    opposing flow on the same instrument.
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";
import Database from "better-sqlite3";

import { tickBots, saveBotPair, scalpingAnalyze, ensureBotsTables, ensureBotPairTables } from "../src/strategy/bots.js";
import { applyCostsFor, applyCosts, applyPayoutMarkup, getSymbolCost, resetCostCache } from "../src/strategy/costs.js";
import { classifyFamily, isSynthetic } from "../src/strategy/synthetics.js";

/**
 * A clean uptrend so EMA9 > EMA21 with a controlled true range.
 * ATR ~= step + range, gross RR is the engine's fixed 2.0/1.5 = 1.33.
 */
function trendBars(n = 60, { start = 100, step = 0.1, range = 0.2 } = {}) {
  const out = [];
  let px = start;
  for (let i = 0; i < n; i++) {
    const open = px;
    px += step;
    const close = px;
    out.push({
      open, close, volume: 100,
      high: Math.max(open, close) + range / 2,
      low: Math.min(open, close) - range / 2,
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// 1. The cost gate
// ---------------------------------------------------------------------------

test("costs: a synthetic is priced by payout markup, not by a CFD spread", () => {
  const synth = applyCostsFor({ symbol: "R_100", entry: 1000, sl: 999.5, tp: 1000.5 });
  assert.equal(synth.model, "payout_markup");
  assert.equal(synth.markupPct, 0.02);
  assert.equal(synth.netRisk, synth.grossRisk, "a multiplier crosses no spread — risk is untouched");
  assert.ok(synth.netWin < synth.grossWin, "the markup comes off the win");

  const fx = applyCostsFor({ symbol: "EURUSD", entry: 1.1, sl: 1.0995, tp: 1.1005 });
  assert.equal(fx.model, "spread");
  assert.ok(fx.netRisk > fx.grossRisk, "FX pays the spread on both sides of the risk");
});

test("costs: the payout markup reduces the win, and never widens the stop", () => {
  const r = applyPayoutMarkup({ symbol: "R_75", entry: 100, sl: 98, tp: 104 });
  // 2% of a 4-unit target, off the win.
  assert.equal(Number(r.costPrice.toFixed(4)), 0.08);
  assert.equal(r.netRisk, 2);
  assert.equal(r.netWin, 4 - 0.08);
  assert.equal(Number(r.netRR.toFixed(2)), 1.96);
  assert.equal(r.blocked, false);
});

test("costs: a symbol override sets the markup, and it is clamped to the documented 3% cap", () => {
  const prev = process.env.COST_OVERRIDES_JSON;
  try {
    process.env.COST_OVERRIDES_JSON = JSON.stringify({ R_100: { markup: 0.5 } });
    resetCostCache();
    // 50% is well past Deriv's documented 3% ceiling for a payout markup. A
    // deployment typo must not turn into a silent 50% haircut on every win.
    const r = applyCostsFor({ symbol: "R_100", entry: 100, sl: 99, tp: 102 });
    assert.equal(r.markupPct, 0.03, "the markup is clamped to the 3% cap");
    // A sane override is honoured exactly.
    process.env.COST_OVERRIDES_JSON = JSON.stringify({ R_100: { markup: 0.01 } });
    resetCostCache();
    assert.equal(applyCostsFor({ symbol: "R_100", entry: 100, sl: 99, tp: 102 }).markupPct, 0.01);
  } finally {
    if (prev === undefined) delete process.env.COST_OVERRIDES_JSON;
    else process.env.COST_OVERRIDES_JSON = prev;
    resetCostCache();
  }
});

test("scalping: a setup whose edge dies in costs is NOT emitted as a signal", () => {
  // XAUUSD: 0.35 spread + 0.10 slippage each way = 0.55 of friction. On this
  // fixture the ATR is ~0.3, so a 1.5xATR stop (0.45) is smaller than the
  // round trip itself: the 1.33 gross RR is a lie and the trade is refused.
  const r = scalpingAnalyze("XAUUSD", trendBars(60, { start: 2000, step: 0.1, range: 0.2 }), "5m");
  assert.equal(r.decision, "WAIT");
  assert.ok(r.costs, "the cost read must be attached even when it blocks");
  assert.equal(r.costs.blocked, true);
  assert.ok(r.reasons.some((x) => /COST GATE/.test(x)), `reasons: ${r.reasons.join(" | ")}`);
  assert.equal(r.entry, null, "a blocked setup publishes no levels to trade");
});

test("scalping: the published RR is the NET RR, and the gross geometry stays visible", () => {
  // BTCUSD (spread 35 + 10 slippage each way = 55) on a wide-range fixture, so
  // the round-trip cost is a small share of a 1.5xATR stop. This is the case
  // the gate is supposed to ALLOW — with the honest, net number on it.
  const r = scalpingAnalyze("BTCUSD", trendBars(60, { start: 60000, step: 2000, range: 2000 }), "5m");
  assert.equal(r.decision, "BUY");
  assert.equal(r.rrGross, 1.33);
  assert.ok(r.rr < r.rrGross, `net RR ${r.rr} must be below gross ${r.rrGross}`);
  assert.equal(r.rr, r.costs.netRR, "the advertised RR is the one the trader keeps");
  assert.equal(r.costs.model, "spread");
});

test("scalping: costs are never presented as exact when the symbol is estimated", () => {
  const r = scalpingAnalyze("BTCUSD", trendBars(60, { start: 60000, step: 2000, range: 2000 }), "5m");
  assert.ok(r.reasons.some((x) => /net RR/.test(x)), "the cost line must state the net RR");
  // A symbol with no published table entry is flagged, not silently guessed.
  assert.equal(getSymbolCost("ZZZUNKNOWN").estimated, true);
  const est = applyCosts({ symbol: "ZZZUNKNOWN", entry: 100, sl: 99, tp: 102 });
  assert.ok(est.costPrice > 0, "an unknown symbol still pays something — ~2bps of price");
});

test("scalping: a setup that survives costs but is thin gets a confidence penalty", () => {
  // EURUSD: 2.6 pips of friction on a 30-pip ATR is ~9% of risk — enough to
  // knock the fixed 1.33 gross RR down toward the 1.3 weak floor.
  const r = scalpingAnalyze("EURUSD", trendBars(60, { start: 1.1, step: 0.0030, range: 0.0030 }), "5m");
  assert.ok(r.costs, "costs are always reported, even on a WAIT");
  assert.equal(r.decision, "WAIT", "the cost penalty drops this setup under the confidence floor");
  assert.equal(r.costs.weak, true);
  assert.ok(r.reasons.some((x) => /cost penalty/.test(x)), `reasons: ${r.reasons.join(" | ")}`);
  assert.ok(r.costs.netRR <= r.rrGross);
  // The gate must not present a thin setup as a tradeable one.
  assert.ok(r.reasons.some((x) => /confidence \d+ below 55 floor/.test(x)), `reasons: ${r.reasons.join(" | ")}`);
});

// ---------------------------------------------------------------------------
// 2. The volume-matrix filter
// ---------------------------------------------------------------------------

test("scalping: strong opposing flow vetoes the entry rather than fighting it", () => {
  const bars = trendBars(60, { start: 60000, step: 2000, range: 2000 });
  const clean = scalpingAnalyze("BTCUSD", bars, "5m");
  assert.equal(clean.decision, "BUY", "the fixture is a clean long without the matrix");

  const vetoed = scalpingAnalyze("BTCUSD", bars, "5m", {
    matrixBias: { bias: "BEAR", score: -3, legs: {} },
  });
  assert.equal(vetoed.decision, "WAIT");
  assert.ok(vetoed.reasons.some((x) => /volume matrix disagrees/.test(x)), `reasons: ${vetoed.reasons.join(" | ")}`);
});

test("scalping: agreeing flow confirms and a weak/neutral matrix changes nothing", () => {
  const bars = trendBars(60, { start: 60000, step: 2000, range: 2000 });
  const withFlow = scalpingAnalyze("BTCUSD", bars, "5m", { matrixBias: { bias: "BULL", score: 3 } });
  const without = scalpingAnalyze("BTCUSD", bars, "5m", { matrixBias: null });
  assert.equal(withFlow.decision, "BUY");
  assert.ok(withFlow.confidence >= without.confidence, "flow with the entry must not lower confidence");
  assert.match(withFlow.reasons.join(" | "), /volume matrix agrees/);

  // A weak or neutral read is informational only — never a veto.
  const neutral = scalpingAnalyze("BTCUSD", bars, "5m", { matrixBias: { bias: "NEUTRAL", score: 0 } });
  assert.equal(neutral.decision, "BUY");
  const weakBear = scalpingAnalyze("BTCUSD", bars, "5m", { matrixBias: { bias: "BEAR", score: -1 } });
  assert.equal(weakBear.decision, "BUY", "|score| < 2 is not strong enough to stand down");
});

// ---------------------------------------------------------------------------
// 3. Synthetic awareness
// ---------------------------------------------------------------------------

test("synthetics: BOOM/CRASH codes without the trailing N are still SPIKE family", () => {
  // Only 150N/300N keep the suffix; BOOM500/600/900/1000 do not. Requiring it
  // classified the most-traded Boom indices as UNKNOWN, so they were never
  // analysed at all.
  for (const s of ["BOOM500", "BOOM600", "BOOM900", "BOOM1000", "CRASH500", "CRASH900"]) {
    assert.equal(classifyFamily(s), "SPIKE", `${s} must be SPIKE`);
    assert.equal(isSynthetic(s), true, `${s} must be synthetic`);
  }
  assert.equal(classifyFamily("R_100"), "VOL");
  assert.deepEqual(classifyFamily("RDBULL"), { family: "TREND", lock: "BUY" });
  assert.equal(isSynthetic("EURUSD"), false);
  assert.equal(isSynthetic("XAUUSD"), false);
});

function harness({ bot, pair, minConf = 60, synth, timeframe = "5m" }) {
  const db = new Database(":memory:");
  ensureBotsTables(db);
  ensureBotPairTables(db);
  saveBotPair(db, "u1", bot, pair, {
    timeframe, tradeEnabled: false, signalEnabled: true, minConfidence: minConf, maxTradesDay: 10,
  });
  const calls = { synth: 0, scalping: 0, pocket: 0 };
  const stored = [];
  const deps = {
    db,
    botCooldown: () => true,
    matrixBiasFor: () => null,
    mtfAnalyze: async () => ({ decision: "WAIT", confidence: 0 }),
    formatSignal: (_s, r) => r,
    analyzeSynth: (symbol) => { calls.synth++; return synth(symbol); },
    scalpingAnalyze: () => { calls.scalping++; return { decision: "BUY", confidence: 70, entry: 1, sl: 0.9, tp1: 1.2, reasons: ["EMA cross"] }; },
    pocketScan: () => { calls.pocket++; return { decision: "CALL", confidence: 70, entryRef: 1, reasons: ["EMA cross"] }; },
    longtermAnalyze: () => ({ decision: "WAIT", confidence: 0 }),
    fetchCandles: async () => trendBars(60),
    fetchCloses: async () => Array.from({ length: 80 }, (_, i) => 1000 + i),
    fetchDaily: async () => [],
    getChain: () => null,
    scanSymbol: () => ({}),
    scanFx: () => ({}),
    storeSignal: (userId, signal) => stored.push({ userId, ...signal }),
    storeOptionsSignal: () => {},
    openPaper: () => {},
    notifyOptions: () => {},
    executor: async () => ({ mode: "paper", status: "simulated" }),
    sizeFor: () => ({ lotSize: 0.01, riskUsd: 0, mode: "fixed", reason: "test" }),
    lastPrice: () => 1000,
    storeCandles: () => null,
    log: () => {},
  };
  return { db, deps, calls, stored };
}

const synthBuy = () => ({
  decision: "BUY", confidence: 70, entry: 1000, sl: 995, tp: 1010, rr: 2,
  family: "VOL", reasons: ["VOL fade: price 2.2 sigma from the 20-bar mean"], maxHoldMin: 30,
});

test("bot: scalping on an RNG synthetic asks the generator engine, never the EMA logic", async () => {
  const h = harness({ bot: "scalping", pair: "R_100", synth: synthBuy });
  await tickBots(h.deps);

  assert.equal(h.calls.synth, 1, "the synthetic engine must be consulted");
  assert.equal(h.calls.scalping, 0, "EMA9/21 must never run on an engineered RNG stream");
  assert.equal(h.stored.length, 1);
  assert.equal(h.stored[0].decision, "BUY");
  assert.match(h.stored[0].scenario, /^SYNTH_/, "the trail must say which engine decided");
  assert.match(h.stored[0].reasoning, /generator|VOL fade/);
});

test("bot: pocket on a synthetic maps the generator read onto CALL/PUT", async () => {
  const h = harness({ bot: "pocket", pair: "BOOM500", synth: () => ({ ...synthBuy(), family: "SPIKE" }) });
  await tickBots(h.deps);

  assert.equal(h.calls.synth, 1);
  assert.equal(h.calls.pocket, 0, "the EMA candle-direction engine must not run on a Boom index");
  assert.equal(h.stored.length, 1);
  assert.equal(h.stored[0].decision, "CALL", "BUY from the generator read becomes a Rise contract");
  assert.match(h.stored[0].scenario, /POCKET_SYNTH_SPIKE/);
  assert.equal(h.stored[0].expiryMin, 5);
});

test("bot: a generator WAIT produces no pocket contract on a synthetic", async () => {
  const h = harness({
    bot: "pocket", pair: "CRASH900",
    synth: () => ({ decision: "WAIT", confidence: 0, family: "SPIKE", reasons: ["no spike (last move 1.2x median, need 8x)"] }),
  });
  await tickBots(h.deps);
  assert.equal(h.calls.pocket, 0);
  assert.equal(h.stored.length, 0, "WAIT is WAIT — never a Rise/Fall");
});

test("bot: a real symbol still uses its own engine (the routing is by market, not by bot)", async () => {
  const h = harness({ bot: "scalping", pair: "EURUSD", synth: synthBuy });
  await tickBots(h.deps);
  assert.equal(h.calls.scalping, 1);
  assert.equal(h.calls.synth, 0);
  assert.equal(h.stored.length, 1);
});
