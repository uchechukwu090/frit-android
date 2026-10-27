// ============================================================================
// BOT SIGNAL LIFECYCLE, END TO END — tickBots() with the predictive volume gate
// ----------------------------------------------------------------------------
// The regression this whole feature exists to kill: a trending bot firing a
// brand-new signal (new entry, new TP/SL) at every candle close. These run the
// REAL runner against a stubbed engine and assert:
//   1. one actionable read  -> one signal, one order
//   2. the next candle, same read -> HOLD, no new signal, TP/SL untouched
//   3. a WAIT read while live  -> HOLD, the signal stays
//   4. volume expansion with the trend -> ONE fresh signal with fresh TP/SL,
//      and NO duplicate order for a position that is already open
//   5. no expansion -> still no signal, however long the trend persists
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";
import Database from "better-sqlite3";

import { tickBots, saveBotPair, ensureBotsTables, ensureBotPairTables } from "../src/strategy/bots.js";
import { getActiveState, stateKey } from "../src/strategy/signalstate.js";
import { bars5, bars1 } from "./helpers/volbars.mjs";

const KEY = stateKey("u1", "intraday", "XAUUSD");

function harness(opts = {}) {
  const {
    tradeEnabled = true,
    read = () => "BUY",
    confidence = () => 85,
    price = () => 2005,
    candles = () => ({}),
  } = opts;

  const db = new Database(":memory:");
  ensureBotsTables(db);
  ensureBotPairTables(db);
  saveBotPair(db, "u1", "intraday", "XAUUSD", {
    timeframe: "30m", tradeEnabled, signalEnabled: true,
    minConfidence: 60, maxTradesDay: 10,
  });

  const stored = [];
  const executed = [];
  const deps = {
    db,
    botCooldown: () => true,
    mtfAnalyze: async () => ({ decision: read(), confidence: confidence() }),
    formatSignal: (_sym, r) => ({
      decision: r.decision, confidence: r.confidence,
      entry: 2000, sl: 1990, tp1: 2020, tp2: 2030, rr: 2,
      scenario: "IMMEDIATE", reasoning: "engine read",
    }),
    scalpingAnalyze: () => ({ decision: "WAIT", confidence: 0 }),
    analyzeSynth: () => ({ decision: "WAIT", confidence: 0 }),
    pocketScan: () => ({ decision: "WAIT", confidence: 0 }),
    longtermAnalyze: () => ({ decision: "WAIT", confidence: 0 }),
    fetchCandles: async () => null, // no REST fallback: the feed store decides
    fetchCloses: async () => [],
    fetchDaily: async () => [],
    getChain: () => null,
    scanSymbol: () => ({}),
    scanFx: () => ({}),
    storeSignal: (userId, signal) => stored.push({ userId, ...signal }),
    storeOptionsSignal: () => {},
    openPaper: () => {},
    notifyOptions: () => {},
    executor: async (args) => { executed.push(args); return { mode: "paper", status: "simulated" }; },
    sizeFor: () => ({ lotSize: 0.1, riskUsd: 5, mode: "fixed", reason: "test" }),
    lastPrice: () => price(),
    storeCandles: (_sym, interval) => candles()[interval] || null,
    log: () => {},
  };
  return { db, deps, stored, executed };
}

const state = (db) => getActiveState(db, KEY);
const backdate = (db, ms) => db
  .prepare("UPDATE signal_state SET opened_at = ?, last_trigger_at = ? WHERE key = ?")
  .run(Date.now() - ms, Date.now() - ms, KEY);

test("bot: one actionable read -> one signal and one order", async () => {
  const { db, deps, stored, executed } = harness();
  await tickBots(deps);

  assert.equal(stored.length, 1, "exactly one signal must be stored");
  assert.equal(executed.length, 1, "exactly one order must be sent");
  assert.equal(stored[0].decision, "BUY");
  assert.equal(stored[0].entry, 2000);
  assert.equal(stored[0].sl, 1990);
  assert.equal(stored[0].tp1, 2020);

  const s = state(db);
  assert.ok(s, "the signal must be live");
  assert.equal(s.tp, 2020);
  assert.equal(s.placed, 1, "an order is open for this signal");
  assert.equal(s.retriggers, 0);
});

test("bot: the SAME read on the next candle is a HOLD, not a new signal", async () => {
  const { db, deps, stored, executed } = harness();
  await tickBots(deps);
  await tickBots(deps);

  assert.equal(stored.length, 1, "the old behaviour stored a fresh signal per candle");
  assert.equal(executed.length, 1, "and a fresh order per candle");
  const s = state(db);
  assert.equal(s.entry, 2000, "entry must not be recomputed");
  assert.equal(s.tp, 2020, "TP must not be recomputed");
  assert.equal(s.opened_at, s.last_trigger_at, "nothing retriggered");

  // The hold is visible in the run log but is NOT a trade.
  const holds = db.prepare("SELECT COUNT(*) c FROM bot_runs WHERE decision = 'HOLD'").get().c;
  const placed = db.prepare("SELECT COUNT(*) c FROM bot_runs WHERE placed = 1").get().c;
  assert.equal(holds, 1);
  assert.equal(placed, 1, "only the real fire counts as a placed trade");
});

test("bot: a WAIT read while a signal is live just waits", async () => {
  let decision = "BUY";
  const { db, deps, stored, executed } = harness({ read: () => decision });
  await tickBots(deps);
  assert.equal(stored.length, 1);

  decision = "WAIT";
  const run = await tickBots(deps);

  assert.equal(stored.length, 1, "a WAIT must not open or close anything by itself");
  assert.equal(executed.length, 1);
  assert.ok(state(db), "the signal stays live");
  assert.equal(run[0].decision, "HOLD");
  assert.match(run[0].note, /read is WAIT/);
});

test("bot: volume expansion with the trend re-triggers ONE signal with fresh TP/SL", async () => {
  let mode = "flat";
  const candles = () => (mode === "flat"
    ? { "5m": bars5({ ramp: false }), "1m": bars1({ ramp: false }) }
    : { "5m": bars5(), "1m": bars1() });
  let decision = "BUY";
  const { db, deps, stored, executed } = harness({ candles, read: () => decision });

  await tickBots(deps);
  assert.equal(stored.length, 1);

  // Two candles later, still the same trend, but the engine now says WAIT —
  // and the predictive volume engine sees activity about to expand.
  backdate(db, 20 * 60_000);
  const openedAt = state(db).opened_at;
  decision = "WAIT";
  mode = "expansion";
  const run = await tickBots(deps);

  assert.equal(stored.length, 2, "the expansion is what earns a second signal");
  const fresh = stored[1];
  assert.notEqual(fresh.sl, 1990, "the whole point: a NEW stop");
  assert.notEqual(fresh.tp1, 2020, "and a NEW target");
  assert.ok(fresh.entry > 2000, "re-priced to the market");
  assert.equal(fresh.scenario, "VOLUME");
  assert.match(fresh.reasoning, /VOLUME RETRIGGER #1/);
  assert.ok(fresh.confidence >= 60, `retrigger confidence ${fresh.confidence}`);
  assert.equal(fresh.retriggerOf, KEY);

  const s = state(db);
  assert.equal(s.retriggers, 1);
  assert.equal(s.entry, fresh.entry, "the live signal's levels follow the retrigger");
  assert.equal(s.tp, fresh.tp1);
  assert.equal(s.opened_at, openedAt, "still ONE signal — the open time does not move");

  assert.equal(executed.length, 1, "no duplicate order for a position that is already open");
  assert.equal(run[0].execution, "held");
  assert.match(run[0].note, /no duplicate order/);
  assert.equal(
    db.prepare("SELECT COUNT(*) c FROM bot_runs WHERE placed = 1").get().c, 1,
    "a held retrigger must not consume the daily trade cap",
  );
});

test("bot: no expansion -> no second signal, however long the trend lasts", async () => {
  let decision = "BUY";
  const { db, deps, stored, executed } = harness({
    read: () => decision,
    candles: () => ({ "5m": bars5({ ramp: false }), "1m": bars1({ ramp: false }) }),
  });
  await tickBots(deps);
  backdate(db, 30 * 60_000);
  decision = "WAIT";
  await tickBots(deps);
  backdate(db, 45 * 60_000);
  await tickBots(deps);

  assert.equal(stored.length, 1, "recomputing every candle must not manufacture signals");
  assert.equal(executed.length, 1);
  assert.equal(state(db).retriggers, 0);
});

test("bot: an exhaustion spike must never re-trigger", async () => {
  let decision = "BUY";
  const { deps, stored } = harness({
    read: () => decision,
    candles: () => ({ "5m": bars5({ spike: true }), "1m": bars1() }),
  });
  await tickBots(deps);
  backdate(deps.db, 20 * 60_000);
  decision = "WAIT";
  await tickBots(deps);

  assert.equal(stored.length, 1, "a 7x spike is exhaustion, not fuel for a continuation");
});

test("bot: TP hit with no setup left closes the cycle silently", async () => {
  let price = 2005;
  let decision = "BUY";
  const h = harness({ price: () => price, read: () => decision });
  await tickBots(h.deps);
  assert.equal(h.stored.length, 1);

  price = 2021;      // above TP 2020
  decision = "WAIT"; // the engine has nothing new to say
  const run = await tickBots(h.deps);

  assert.equal(h.stored.length, 1, "a resolved signal must not spawn another");
  assert.equal(state(h.db), null, "the signal is closed");
  const row = h.db.prepare("SELECT status, close_reason FROM signal_state WHERE key = ?").get(KEY);
  assert.equal(row.status, "CLOSED");
  assert.match(row.close_reason, /take-profit reached/);
  assert.equal(run[0].decision, "WAIT");
  assert.match(run[0].note, /take-profit reached/);
});

test("bot: TP hit while the trend is STILL live starts one fresh cycle", async () => {
  let price = 2005;
  const h = harness({ price: () => price }); // read stays BUY
  await tickBots(h.deps);
  assert.equal(h.stored.length, 1);

  price = 2021;
  const run = await tickBots(h.deps);

  assert.equal(h.stored.length, 2, "the same trend earns exactly one new signal");
  assert.equal(h.executed.length, 2, "and one order for it");
  assert.equal(run[0].decision, "BUY");
  const s = state(h.db);
  assert.ok(s, "the new cycle is live");
  assert.equal(s.retriggers, 0);
  assert.equal(s.tp, 2020, "the engine recomputes TP/SL for a NEW signal, never for a live one");
});

test("bot: an opposite actionable read closes and fires the reversal", async () => {
  let decision = "BUY";
  const { db, deps, stored } = harness({ read: () => decision });
  await tickBots(deps);
  assert.equal(stored.length, 1);

  decision = "SELL";
  const run = await tickBots(deps);

  assert.equal(stored.length, 2);
  assert.equal(stored[1].decision, "SELL");
  assert.equal(run[0].decision, "SELL");
  assert.equal(state(db).dir, "SELL", "the live signal is now the reversal");
});

test("bot: a signals-only bot retriggers without ever sending an order", async () => {
  let mode = "flat";
  const { db, deps, stored, executed } = harness({
    tradeEnabled: false,
    read: () => "WAIT",
    candles: () => (mode === "flat"
      ? { "5m": bars5({ ramp: false }), "1m": bars1({ ramp: false }) }
      : { "5m": bars5(), "1m": bars1() }),
  });

  // First fire needs an actionable read, then flip to WAIT for the retrigger.
  let first = true;
  deps.mtfAnalyze = async () => ({ decision: first ? "BUY" : "WAIT", confidence: 85 });
  await tickBots(deps);
  first = false;
  backdate(db, 20 * 60_000);
  mode = "expansion";
  await tickBots(deps);

  assert.equal(executed.length, 0, "signals-only means signals-only, even on a retrigger");
  assert.equal(stored.length, 2);
  assert.equal(stored[1].scenario, "VOLUME");
  assert.equal(state(db).placed, 0);
});
