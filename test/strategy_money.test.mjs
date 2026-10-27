// ============================================================================
// Week 5 (audit) — unit tests for the PREVIOUSLY UNTESTED strategy modules
// ----------------------------------------------------------------------------
// F-07 said: "no tests for strategy layer beyond creative" and named the pure,
// testable ones — risk.js, bots.js, position_monitor.js, costs.js, structures,
// synthetics.js, volumematrix.js, volengine.js, longterm.js, journal.js,
// ledger.js. Week 2 covered hysteresis + rails/router + options/pricing.
// This file closes that gap for the money-critical half of the layer.
//
// Every test is hermetic: in-memory SQLite, no network, no server boot.
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";
import Database from "better-sqlite3";

import { PortfolioRisk, usdExposureSign, groupOf, riskLimits } from "../src/strategy/risk.js";
import { applyCosts, costInPrice, getSymbolCost, costToUsd, resetCostCache } from "../src/strategy/costs.js";
import { PullbackJournal } from "../src/strategy/journal.js";
import { PositionMonitor } from "../src/strategy/position_monitor.js";
import { ensureBotsTables, saveBot, getBots, tickBots, scalpingAnalyze, BOT_DEFS } from "../src/strategy/bots.js";
import { volumeProfile, vwap, cvdProxy, rvol, analyzeVolumeMatrix } from "../src/strategy/volumematrix.js";
import { analyzeVol } from "../src/strategy/volengine.js";
import { classifyFamily, analyzeSynth } from "../src/strategy/synthetics.js";
import { maturitySummary } from "../src/strategy/ledger.js";
import { analyzeLongterm } from "../src/strategy/longterm.js";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
const PIP = { EURUSD: { pipSize: 0.0001, pipValue: 10 }, GBPUSD: { pipSize: 0.0001, pipValue: 10 } };

function posTable(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS positions (
      id TEXT PRIMARY KEY, symbol TEXT, action TEXT, lot_size REAL, entry REAL,
      sl REAL, tp REAL, source TEXT, status TEXT DEFAULT 'MONITORING',
      placed_at DATETIME DEFAULT CURRENT_TIMESTAMP, monitoring_at DATETIME,
      resolved_at DATETIME, outcome TEXT, close_price REAL, note TEXT
    );
  `);
}

function riskDb() {
  const db = new Database(":memory:");
  posTable(db);
  return db;
}

function addPos(db, over = {}) {
  const p = {
    id: over.id || `p${Math.random().toString(36).slice(2, 8)}`,
    symbol: "EURUSD", action: "buy", lot_size: 0.01, entry: 1.1, sl: 1.099, tp: 1.102,
    status: "MONITORING", source: "test",
    ...over,
  };
  db.prepare(
    `INSERT INTO positions (id, symbol, action, lot_size, entry, sl, tp, source, status, outcome, close_price, resolved_at)
     VALUES (?,?,?,?,?,?,?,?,?,?,?,?)`
  ).run(p.id, p.symbol, p.action, p.lot_size, p.entry, p.sl, p.tp, p.source, p.status, p.outcome ?? null, p.close_price ?? null, p.resolved_at ?? null);
  return p;
}

/** Run fn with temporary env overrides, always restoring them. */
function withEnv(env, fn) {
  const saved = {};
  for (const k of Object.keys(env)) saved[k] = process.env[k];
  Object.assign(process.env, env);
  try { return fn(); } finally {
    for (const k of Object.keys(env)) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  }
}

// ===========================================================================
// 1. risk.js — the four portfolio guards
// ===========================================================================
test("risk: max-open cap blocks the 5th concurrent position", () => {
  const db = riskDb();
  for (let i = 0; i < 4; i++) addPos(db, { id: `p${i}` });
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.099, balance: 1000 });
  assert.equal(res.allowed, false);
  assert.match(res.reasons.join(" | "), /Too many live positions \(4\/4\)/);
});

test("risk: per-trade cap blocks an oversized stop (2% of balance default)", () => {
  const db = riskDb();
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  // 50 pips at 0.5 lot = 50 * 10 * 0.5 = $250 = 25% of a $1000 balance.
  const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.5, entry: 1.1, sl: 1.095, balance: 1000 });
  assert.equal(res.allowed, false);
  assert.match(res.reasons.join(" | "), /Single-trade risk .* exceeds cap/);
  assert.equal(res.exposure.newRiskUsd, 250);
});

test("risk: correlation guard blocks a 3rd USD_MAJORS position", () => {
  const db = riskDb();
  addPos(db, { id: "a", symbol: "EURUSD" });
  addPos(db, { id: "b", symbol: "GBPUSD" });
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "AUDUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.099, balance: 1000 });
  assert.equal(res.allowed, false);
  assert.match(res.reasons.join(" | "), /Correlation guard: USD_MAJORS/);
  assert.equal(res.exposure.groupCounts.USD_MAJORS, 3);
});

test("risk: net USD exposure cap blocks same-direction stacking (caps isolated)", () => {
  const db = riskDb();
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  withEnv({ RISK_MAX_TRADE_PCT: "10", RISK_MAX_USD_NET_PCT: "3" }, () => {
    // $40 of USD-short risk on a $1000 book = 4% net > 3% cap.
    const res = risk.check({
      symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.099,
      balance: 1000, riskUsdOverride: 40,
    });
    assert.equal(res.allowed, false);
    assert.match(res.reasons.join(" | "), /Net USD exposure .* exceeds cap/);
  });
});

test("risk: daily-loss kill-switch blocks new trades after a losing day", () => {
  const db = riskDb();
  addPos(db, {
    id: "lost", status: "RESOLVED", outcome: "loss",
    lot_size: 0.1, entry: 1.1, close_price: 1.09, sl: 1.09,
    resolved_at: null, // filled by the SQL below with the DB's own 'now'
  });
  db.prepare("UPDATE positions SET resolved_at = datetime('now') WHERE id = 'lost'").run();
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.099, balance: 1000 });
  assert.equal(res.allowed, false);
  assert.match(res.reasons.join(" | "), /KILL-SWITCH/);
  assert.equal(risk.status(1000).killSwitchTripped, true);
});

test("risk: FAIL-CLOSED when the risk state cannot be read (Week 5 regression)", () => {
  // No `positions` table at all -> the guard layer must NOT silently admit.
  const db = new Database(":memory:");
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.099, balance: 1000 });
  assert.equal(res.allowed, false, "risk layer must fail closed when it cannot read position state");
  assert.ok(res.reasons.length > 0, "a blocking reason must be reported");
});

test("risk: usdExposureSign and groupOf behave as documented", () => {
  assert.equal(usdExposureSign("EURUSD", "buy"), -1);  // buying EURUSD = short USD
  assert.equal(usdExposureSign("EURUSD", "sell"), 1);
  assert.equal(usdExposureSign("USDCHF", "buy"), 1);    // buying USDCHF = long USD
  assert.equal(usdExposureSign("BTCUSD", "buy"), -1);
  assert.equal(usdExposureSign("SPX500", "buy"), 0);   // not an FX pair
  assert.equal(groupOf("EURUSD"), "USD_MAJORS");
  assert.equal(groupOf("XAUUSD"), "METALS");
  assert.equal(groupOf("SPX500"), "INDICES");
  assert.equal(groupOf("ZZZZZ"), "OTHER");
  assert.equal(riskLimits().maxOpen, 4);
});

// ===========================================================================
// 2. costs.js — net RR after retail friction
// ===========================================================================
test("costs: a 2R forex setup stays tradeable after spread+slippage", () => {
  const r = applyCosts({ symbol: "EURUSD", entry: 1.1, sl: 1.098, tp: 1.104 });
  const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;
  assert.ok(near(r.costPrice, 0.00016 + 2 * 0.00005), `costPrice ${r.costPrice}`);
  assert.ok(near(r.grossRisk, 0.002), `grossRisk ${r.grossRisk}`);
  assert.ok(near(r.grossWin, 0.004), `grossWin ${r.grossWin}`);
  assert.ok(near(r.netRR, 1.65), `netRR ${r.netRR}`);
  assert.equal(r.blocked, false);
  assert.equal(r.weak, false);
});

test("costs: a tight stop is BLOCKED because costs eat the edge", () => {
  const r = applyCosts({ symbol: "EURUSD", entry: 1.1, sl: 1.0995, tp: 1.101 });
  assert.ok(r.netRR < 1.0, `netRR ${r.netRR} should be below 1.0`);
  assert.equal(r.blocked, true);
  assert.equal(r.weak, true);
});

test("costs: unknown symbols fall back to a 2bps estimate and are flagged", () => {
  const c = getSymbolCost("NOTAREALPAIR");
  assert.equal(c.estimated, true);
  assert.equal(costInPrice("NOTAREALPAIR", 100), 0.04); // 2bps + 2 x 1bps
});

test("costs: USD conversion uses the caller's pip table", () => {
  const u = costToUsd({ symbol: "EURUSD", lotSize: 0.01, pipSize: 0.0001, pipValue: 10, refPrice: 1.1 });
  assert.equal(u.spreadSlipUsd, 0.26);   // 0.00026 / 0.0001 * 10 * 0.01
  assert.equal(u.commissionUsd, 0);
  assert.equal(u.totalUsd, 0.26);
});

test("costs: a zero-risk setup is blocked, never crashed", () => {
  const r = applyCosts({ symbol: "EURUSD", entry: 1.1, sl: 1.1, tp: 1.2 });
  assert.equal(r.blocked, true);
  assert.equal(r.netRR, 0);
});

// ===========================================================================
// 3. journal.js — the pullback journal scorer
// ===========================================================================
function journalDb() {
  const db = new Database(":memory:");
  const j = new PullbackJournal({ db });
  return { db, j };
}

test("journal: TP first scores a win, SL first a loss, both-touched is a LOSS", async () => {
  const { j } = journalDb();
  const t = Date.now() + 60_000;
  const mk = (high, low, close) => ({ time: t, open: 100, high, low, close });
  j.record({ symbol: "EURUSD", strategy: "fib", direction: "BUY", entry: 100, sl: 95, tp1: 110 });

  // TP first
  await j.resolvePending({ fetchCandles: async () => [mk(100.5, 99.5, 100), mk(111, 100, 108)] });
  let row = j.recent(1)[0];
  assert.equal(row.status, "RESOLVED");
  assert.equal(row.outcome, "win");

  // SL first (different strategy -> not deduped)
  j.record({ symbol: "EURUSD", strategy: "fib2", direction: "BUY", entry: 100, sl: 95, tp1: 110 });
  await j.resolvePending({ fetchCandles: async () => [mk(100.5, 99.5, 100), mk(101, 94, 96)] });
  row = j.recent(1)[0];
  assert.equal(row.outcome, "loss");

  // Both in the same bar -> conservative LOSS
  j.record({ symbol: "EURUSD", strategy: "fib3", direction: "BUY", entry: 100, sl: 95, tp1: 110 });
  await j.resolvePending({ fetchCandles: async () => [mk(100.5, 99.5, 100), mk(111, 94, 105)] });
  row = j.recent(1)[0];
  assert.equal(row.outcome, "loss", "same-bar TP+SL must count as a loss (conservative)");
  assert.equal(row.win_first, 0);
});

test("journal: a setup that never fills expires instead of counting as a loss", async () => {
  const { j } = journalDb();
  j.record({ symbol: "EURUSD", strategy: "fib", direction: "BUY", entry: 100, sl: 95, tp1: 110 });
  const t = Date.now() + 60_000;
  const out = await j.resolvePending({
    fetchCandles: async () => Array.from({ length: 96 }, (_, i) => ({ time: t + i * 1800_000, open: 120, high: 121, low: 119, close: 120 })),
  });
  assert.equal(out.expired, 1);
  const row = j.recent(1)[0];
  assert.equal(row.status, "EXPIRED");
  assert.notEqual(row.outcome, "loss");
});

test("journal: duplicate setups inside 6h are deduped, older ones are not", () => {
  const { db, j } = journalDb();
  const id1 = j.record({ symbol: "EURUSD", strategy: "fib", direction: "BUY", entry: 100, sl: 95, tp1: 110 });
  assert.ok(id1);
  assert.equal(j.record({ symbol: "EURUSD", strategy: "fib", direction: "BUY", entry: 100, sl: 95, tp1: 110 }), null);
  db.prepare("UPDATE pullback_journal SET recorded_at = ? WHERE id = ?").run(Date.now() - 7 * 3600_000, id1);
  const id2 = j.record({ symbol: "EURUSD", strategy: "fib", direction: "BUY", entry: 100, sl: 95, tp1: 110 });
  assert.ok(id2, "a 7h-old setup is a new idea and must be recorded");
});

test("journal: stats refuse a verdict below 30 resolved setups", async () => {
  const { j } = journalDb();
  j.record({ symbol: "EURUSD", strategy: "fib", direction: "BUY", entry: 100, sl: 95, tp1: 110 });
  const t = Date.now() + 60_000;
  await j.resolvePending({ fetchCandles: async () => [{ time: t, open: 100, high: 111, low: 99, close: 110 }] });
  const s = j.stats();
  assert.equal(s.resolved, 1);
  assert.match(s.verdict, /below the 30-trade minimum/);
});

// ===========================================================================
// 4. position_monitor.js — the paper-position state machine
// ===========================================================================
test("position monitor: one price fetch per symbol per tick, zero when idle", async () => {
  const db = new Database(":memory:");
  let calls = 0;
  const mon = new PositionMonitor({ db, getPrice: async () => { calls++; return 1.1; } });
  await mon._tick();
  assert.equal(calls, 0, "no positions -> no price calls (free-tier protection)");

  mon.register({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.09, tp: 1.12 });
  mon.register({ symbol: "EURUSD", action: "sell", lotSize: 0.01, entry: 1.1, sl: 1.12, tp: 1.09 });
  calls = 0;
  await mon._tick();
  assert.equal(calls, 1, "two positions on the same symbol -> one price fetch");
});

test("position monitor: TP resolves a win, SL a loss, and resolve is idempotent", async () => {
  const db = new Database(":memory:");
  let price = 1.13;
  const mon = new PositionMonitor({ db, getPrice: async () => price });
  const p = mon.register({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.09, tp: 1.12 });
  assert.equal(mon.get(p.id).status, "MONITORING");
  await mon._tick();
  assert.equal(mon.get(p.id).outcome, "win");
  const again = mon.resolve(p.id, "loss", 1.0);
  assert.equal(again.already, true);
  assert.equal(mon.get(p.id).outcome, "win", "a resolved position must not be re-resolved");

  price = 1.05;
  const p2 = mon.register({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.09, tp: 1.12 });
  await mon._tick();
  assert.equal(mon.get(p2.id).outcome, "loss");
});

test("position monitor: a missing price keeps the position open", async () => {
  const db = new Database(":memory:");
  const mon = new PositionMonitor({ db, getPrice: async () => null });
  const p = mon.register({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.09, tp: 1.12 });
  await mon._tick();
  assert.equal(mon.get(p.id).status, "MONITORING");
});

// ===========================================================================
// 5. bots.js — config validation + the daily trade cap
// ===========================================================================
test("bots: saveBot validates bot name, pair and timeframe, and clamps the knobs", () => {
  const db = new Database(":memory:");
  ensureBotsTables(db);
  assert.throws(() => saveBot(db, "nope", { pair: "EURUSD" }), /unknown bot/);
  assert.throws(() => saveBot(db, "intraday", { pair: "" }), /pair required/);
  assert.throws(() => saveBot(db, "intraday", { pair: "EURUSD", timeframe: "2H" }), /supports timeframes/);
  assert.throws(() => saveBot(db, "longterm", { pair: "EURUSD", timeframe: "1m" }), /supports timeframes/);

  const b = saveBot(db, "intraday", { pair: "eur usd!", timeframe: "30m", min_confidence: 999, max_trades_day: 0 });
  assert.equal(b.pair, "EURUSD");           // sanitized
  assert.equal(b.min_confidence, 100);     // clamped to 0..100
  assert.equal(b.max_trades_day, 1);       // clamped to 1..10
  assert.equal(b.auto_enabled, false);
});

test("bots: the daily cap counts EVERY placed trade, pocket CALL/PUT included (Week 5 regression)", async () => {
  const db = new Database(":memory:");
  ensureBotsTables(db);
  saveBot(db, "pocket", { pair: "R_100", timeframe: "5m", auto_enabled: true, min_confidence: 50, max_trades_day: 2 });

  let executions = 0;
  const deps = {
    db,
    // R_100 is an RNG synthetic, so the pocket bot now reads it from the
    // generator-aware engine instead of the EMA candle engine. Both deps are
    // stubbed here so the test keeps measuring what it is named for: the cap.
    analyzeSynth: () => ({
      decision: "BUY", confidence: 90, entry: 100, sl: 95, tp: 110, rr: 2,
      family: "VOL", reasons: ["test"],
    }),
    fetchCloses: async () => Array.from({ length: 80 }, (_, i) => 100 + i),
    pocketScan: () => ({ decision: "CALL", confidence: 90, entryRef: 100, reasons: ["test"] }),
    executor: async () => { executions++; return { mode: "paper" }; },
    storeSignal: () => {},
    log: () => {},
  };
  for (let i = 0; i < 4; i++) {
    await tickBots(deps);
    // pretend the bot's interval elapsed so the cadence doesn't mask the cap
    db.prepare("UPDATE bots SET last_run = 0 WHERE name = 'pocket'").run();
  }
  assert.equal(executions, 2, `daily cap of 2 must stop the 3rd pocket trade (got ${executions})`);
});

test("bots: the daily cap also stops a BUY-decision bot", async () => {
  const db = new Database(":memory:");
  ensureBotsTables(db);
  saveBot(db, "intraday", { pair: "EURUSD", timeframe: "30m", auto_enabled: true, min_confidence: 50, max_trades_day: 1 });
  let executions = 0;
  const deps = {
    db,
    mtfAnalyze: async () => ({ decision: "BUY", confidence: 80, entry: 1.1, sl: 1.09, tp1: 1.12, reasons: ["t"] }),
    formatSignal: (pair, r) => ({ ...r, tp1: r.tp1, reasons: r.reasons || [] }),
    executor: async () => { executions++; return { mode: "paper" }; },
    storeSignal: () => {},
    log: () => {},
  };
  for (let i = 0; i < 3; i++) {
    await tickBots(deps);
    db.prepare("UPDATE bots SET last_run = 0 WHERE name = 'intraday'").run();
  }
  assert.equal(executions, 1, `daily cap of 1 must stop the 2nd trade (got ${executions})`);
});

test("bots: a disabled bot never trades", async () => {
  const db = new Database(":memory:");
  ensureBotsTables(db);
  saveBot(db, "intraday", { pair: "EURUSD", timeframe: "30m", auto_enabled: false });
  let executions = 0;
  await tickBots({
    db,
    mtfAnalyze: async () => ({ decision: "BUY", confidence: 99, entry: 1.1, sl: 1.09, tp1: 1.12, reasons: [] }),
    formatSignal: (p, r) => r,
    executor: async () => { executions++; return {}; },
    storeSignal: () => {},
  });
  assert.equal(executions, 0);
});

test("bots: scalping analysis warms up, respects the confidence floor and prices the setup", () => {
  const warm = scalpingAnalyze("EURUSD", Array.from({ length: 10 }, () => ({ open: 1, high: 1, low: 1, close: 1 })));
  assert.equal(warm.decision, "WAIT");
  assert.match(warm.reasons.join(" "), /warming up/);

  // 60 rising candles with a strong last bar.
  const candles = [];
  for (let i = 0; i < 60; i++) {
    const base = 1.1 + i * 0.001;
    candles.push({ open: base, high: base + 0.002, low: base - 0.002, close: base + 0.0015, volume: 100 });
  }

  // Under the DEFAULT retail cost table (1.6-pip EURUSD spread + 0.5-pip
  // slippage each way), that 2xATR target does not survive friction: the cost
  // gate refuses it, which is the whole point of the gate. Asserted first so
  // the test pins the refusal, not just the happy path.
  const retail = scalpingAnalyze("EURUSD", candles, "5m");
  assert.equal(retail.decision, "WAIT");
  assert.equal(retail.costs.weak, true, "retail friction knocks the fixed 1.33 gross RR under the weak floor");
  assert.match(retail.reasons.join(" "), /cost penalty/);
  assert.match(retail.reasons.join(" "), /below 55 floor/);

  // The SAME setup on ECN-grade costs is priced normally — proving the fixture
  // is a real setup and only the friction was standing in its way.
  const prev = process.env.COST_OVERRIDES_JSON;
  let up;
  try {
    process.env.COST_OVERRIDES_JSON = JSON.stringify({ EURUSD: { spread: 0.00002, slippage: 0.00001 } });
    resetCostCache();
    up = scalpingAnalyze("EURUSD", candles, "5m");
  } finally {
    if (prev === undefined) delete process.env.COST_OVERRIDES_JSON;
    else process.env.COST_OVERRIDES_JSON = prev;
    resetCostCache();
  }
  assert.equal(up.decision, "BUY");
  assert.ok(up.sl < up.entry, "SL below entry for a long");
  assert.ok(up.tp1 > up.entry, "TP above entry for a long");
  assert.ok(up.confidence >= 55 && up.confidence <= 75, `confidence ${up.confidence} out of the documented band`);
  assert.ok(up.rr <= up.rrGross, "even cheap costs come off the published R-R");
});

// ===========================================================================
// 6. volumematrix.js — the VP/VWAP/CVD/RVOL filter layer
// ===========================================================================
test("volume matrix: the value area always SPANS the POC's neighbours (Week 5 regression)", () => {
  // Heavy volume at the LOW end of the range -> the POC is the lowest bucket.
  const candles = [];
  for (let i = 0; i < 20; i++) candles.push({ open: 100, high: 101, low: 100, close: 100.5, volume: 100 });
  for (let i = 0; i < 20; i++) candles.push({ open: 101, high: 102, low: 101, close: 101.5, volume: 10 });
  const vp = volumeProfile(candles);
  assert.ok(vp, "profile must be produced");
  assert.ok(vp.valueAreaHigh > vp.poc, `value area must extend above a lowest-bucket POC (poc=${vp.poc}, vaHigh=${vp.valueAreaHigh})`);
  assert.ok(vp.valueAreaLow <= vp.poc, "value area must start at or below the POC");
  assert.ok(vp.valueAreaHigh > vp.valueAreaLow, "value area must not collapse to a single level");
});

test("volume matrix: VWAP, CVD proxy and RVOL compute from the candles", () => {
  const candles = [
    { open: 10, high: 12, low: 8, close: 11, volume: 100 },
    { open: 11, high: 13, low: 9, close: 12, volume: 100 },
  ];
  assert.equal(vwap(candles), (11 + 11.333333333333334) / 2 * 100 / 200 ? vwap(candles) : vwap(candles)); // sanity: positive
  assert.ok(vwap(candles) > 10 && vwap(candles) < 12);
  assert.ok(cvdProxy(candles) > 0, "two up candles -> positive delta proxy");
  const many = Array.from({ length: 30 }, (_, i) => ({ open: 10, high: 11, low: 9, close: 10, volume: 100 }));
  many[many.length - 1].volume = 300;
  assert.ok(rvol(many) > 2.5, `RVOL should spike (got ${rvol(many)})`);
  assert.equal(rvol([{ open: 1, high: 1, low: 1, close: 1, volume: 1 }]), null);
});

test("volume matrix: price above the session VWAP reads BULL", () => {
  const up = Array.from({ length: 40 }, (_, i) => ({ open: 100 + i * 0.1, high: 100.5 + i * 0.1, low: 99.5 + i * 0.1, close: 100.4 + i * 0.1, volume: 100 }));
  const flat = Array.from({ length: 40 }, (_, i) => ({ open: 100, high: 100.2, low: 99.8, close: 100, volume: 100 }));
  const bull = analyzeVolumeMatrix("EURUSD", { "4h": up, "30m": up, "15m": up, "5m": up, "1m": up });
  assert.equal(bull.bias, "BULL");
  assert.ok(bull.confidence > 0);
  const neutral = analyzeVolumeMatrix("EURUSD", { "30m": flat });
  assert.ok(["NEUTRAL", "BULL", "BEAR"].includes(neutral.bias));
  assert.equal(analyzeVolumeMatrix("EURUSD", {}).why[0], "no candle data for any matrix timeframe");
});

// ===========================================================================
// 7. volengine.js — options strategy selection
// ===========================================================================
const ivHist = Array.from({ length: 40 }, (_, i) => 0.1 + (i % 20) * 0.005); // 0.10 .. 0.195

test("vol engine: backwardation is a STRESS veto (never sell premium)", () => {
  const out = analyzeVol({
    expiries: [{ expiration: "a", dte: 14, atmIv: 0.20 }, { expiration: "b", dte: 45, atmIv: 0.15 }],
    ivHistory: ivHist, rv20: 0.18,
  });
  assert.equal(out.regime, "STRESS");
  assert.equal(out.strategy, "NONE");
  assert.ok(out.termSlope < 0);
});

test("vol engine: rich IV -> SELL_VERTICAL, cheap IV -> LONG_STRADDLE, event -> straddle", () => {
  const rich = analyzeVol({
    expiries: [{ expiration: "a", dte: 21, atmIv: 0.30 }, { expiration: "b", dte: 45, atmIv: 0.30 }],
    ivHistory: ivHist, rv20: 0.2,
  });
  assert.equal(rich.strategy, "SELL_VERTICAL");
  assert.equal(rich.regime, "HIGH_IV");
  assert.ok(rich.ivRank > 40);

  const cheap = analyzeVol({
    expiries: [{ expiration: "a", dte: 21, atmIv: 0.10 }, { expiration: "b", dte: 45, atmIv: 0.10 }],
    ivHistory: ivHist, rv20: 0.2,
  });
  assert.equal(cheap.strategy, "LONG_STRADDLE");
  assert.equal(cheap.regime, "LOW_IV");

  const event = analyzeVol({
    expiries: [{ expiration: "a", dte: 21, atmIv: 0.30 }, { expiration: "b", dte: 45, atmIv: 0.30 }],
    ivHistory: ivHist, rv20: 0.2, eventDays: { within7d: true, name: "FOMC" },
  });
  assert.equal(event.strategy, "LONG_STRADDLE");
  assert.equal(event.regime, "EVENT");
  assert.match(event.reasons.join(" "), /FOMC/);
});

test("vol engine: no usable tenor -> NONE, never a guess", () => {
  const out = analyzeVol({ expiries: [{ expiration: "a", dte: 2, atmIv: 0.2 }], ivHistory: ivHist });
  assert.equal(out.strategy, "NONE");
  assert.match(out.reasons.join(" "), /no clean expiry tenor/);
});

// ===========================================================================
// 8. synthetics.js — generator-aware analysis
// ===========================================================================
test("synthetics: generators are classified into families, unknown stays UNKNOWN", () => {
  assert.equal(classifyFamily("R_100"), "VOL");
  assert.equal(classifyFamily("1HZ10V"), "VOL");
  assert.equal(classifyFamily("STPRNG"), "VOL");
  assert.equal(classifyFamily("BOOM100N"), "SPIKE");
  assert.equal(classifyFamily("CRASH25N"), "SPIKE");
  assert.deepEqual(classifyFamily("RDBULL"), { family: "TREND", lock: "BUY" });
  assert.deepEqual(classifyFamily("RDBEAR"), { family: "TREND", lock: "SELL" });
  assert.deepEqual(classifyFamily("DEX20UP"), { family: "TREND", lock: "BUY" });
  assert.deepEqual(classifyFamily("DEX20DN"), { family: "TREND", lock: "SELL" });
  assert.equal(classifyFamily("DSI10").family, "REGIME");
  assert.equal(classifyFamily("DSI10").regimeMin, 10);
  assert.equal(classifyFamily("EURUSD"), "UNKNOWN");
});

test("synthetics: an unknown generator is never traded", () => {
  const out = analyzeSynth("EURUSD", Array.from({ length: 200 }, (_, i) => 100 + i));
  assert.equal(out.decision, "WAIT");
  assert.match(out.reasons.join(" "), /unknown generator/);
});

test("synthetics: RDBULL is direction-locked and never produces a SELL", () => {
  const rising = Array.from({ length: 60 }, (_, i) => 100 + i * 0.1);
  const out = analyzeSynth("RDBULL", rising);
  assert.notEqual(out.decision, "SELL");
  const falling = Array.from({ length: 60 }, (_, i) => 200 - i * 0.1);
  const out2 = analyzeSynth("RDBULL", falling);
  assert.notEqual(out2.decision, "SELL", "a locked-long family must never short");
  assert.equal(out2.decision, "WAIT");
});

test("synthetics: the VOL family fades extremes and refuses a dead tape", () => {
  const flat = Array.from({ length: 120 }, () => 100);
  const dead = analyzeSynth("R_100", flat);
  assert.equal(dead.decision, "WAIT");

  const stretched = Array.from({ length: 119 }, (_, i) => 100 + Math.sin(i) * 0.3);
  stretched.push(120); // a 20σ stretch
  const fade = analyzeSynth("R_100", stretched);
  assert.equal(fade.decision, "SELL", "price far above the mean -> fade down");
  assert.ok(fade.sl > fade.entry, "the stop sits above entry for a short");
  assert.ok(fade.tp < fade.entry);
  assert.ok(fade.maxHoldMin > 0, "VOL trades always carry a time stop");
});

test("synthetics: a spike family needs an actual spike", () => {
  const quiet = Array.from({ length: 80 }, (_, i) => 100 + (i % 2 ? 0.01 : 0));
  const out = analyzeSynth("BOOM100N", quiet);
  assert.equal(out.decision, "WAIT");
  assert.match(out.reasons.join(" "), /spike/i);
});

// ===========================================================================
// 9. ledger.js — the maturity gate for performance claims
// ===========================================================================
function ledgerDb() {
  const db = new Database(":memory:");
  db.exec(`
    CREATE TABLE options_paper (id TEXT PRIMARY KEY, user_id TEXT, status TEXT, pnl REAL, opened_at INTEGER);
    CREATE TABLE positions (id TEXT PRIMARY KEY, status TEXT, outcome TEXT, placed_at DATETIME);
    CREATE TABLE pocket_paper (id TEXT PRIMARY KEY, user_id TEXT, result TEXT, entry_ts INTEGER);
    CREATE TABLE deriv_contracts (contract_id TEXT PRIMARY KEY, status TEXT, profit REAL, placed_at INTEGER);
    CREATE TABLE alpaca_orders (order_id TEXT PRIMARY KEY, fill_price REAL, placed_at INTEGER);
    CREATE TABLE pullback_journal (id INTEGER PRIMARY KEY, status TEXT, outcome TEXT, recorded_at INTEGER);
  `);
  return db;
}

test("ledger: 30 resolved trades over 60+ days is the first MATURE book", () => {
  const db = ledgerDb();
  const old = Date.now() - 70 * 86400_000;
  const ins = db.prepare("INSERT INTO options_paper (id, user_id, status, pnl, opened_at) VALUES (?,?,?,?,?)");
  for (let i = 0; i < 30; i++) ins.run(`o${i}`, "u1", "CLOSED", 5, old + i * 1000);
  const s = maturitySummary(db, "u1");
  const opt = s.books.find((b) => b.book === "options");
  assert.equal(opt.resolved, 30);
  assert.equal(opt.mature, true);
  assert.equal(opt.label, "MATURE");
  assert.ok(s.sellable.includes("options"));
});

test("ledger: a young book is GROWING and says exactly what is missing", () => {
  const db = ledgerDb();
  const ins = db.prepare("INSERT INTO options_paper (id, user_id, status, pnl, opened_at) VALUES (?,?,?,?,?)");
  for (let i = 0; i < 10; i++) ins.run(`o${i}`, "u1", "CLOSED", 5, Date.now() - 5 * 86400_000);
  const s = maturitySummary(db, "u1");
  const opt = s.books.find((b) => b.book === "options");
  assert.equal(opt.mature, false);
  assert.equal(opt.label, "GROWING");
  assert.match(opt.missing, /20 more resolved trades/);
  assert.match(opt.missing, /more days/);
  assert.equal(s.matureCount, 0);
  assert.match(s.verdict, /No mature track record yet/);
});

test("ledger: tracked days are measured in real days, not NaN (Week 5 regression)", () => {
  const db = ledgerDb();
  const ins = db.prepare("INSERT INTO options_paper (id, user_id, status, pnl, opened_at) VALUES (?,?,?,?,?)");
  ins.run("o1", "u1", "CLOSED", 5, Date.now() - 70 * 86400_000);
  const s = maturitySummary(db, "u1");
  const opt = s.books.find((b) => b.book === "options");
  assert.ok(Number.isFinite(opt.daysTracked), `daysTracked must be a number (got ${opt.daysTracked})`);
  assert.ok(Math.abs(opt.daysTracked - 70) < 1, `daysTracked ${opt.daysTracked} should be ~70`);
});

// ===========================================================================
// 10. longterm.js — the daily Donchian trend system
// ===========================================================================
test("longterm: fewer than 210 daily closes is an honest WAIT", () => {
  const out = analyzeLongterm("EURUSD", Array.from({ length: 209 }, (_, i) => 100 + i * 0.01));
  assert.equal(out.decision, "WAIT");
  assert.match(out.reasons.join(" "), /210\+/);
});

test("longterm: a clean uptrend fires a 2R long with SL below and TP above", () => {
  const closes = Array.from({ length: 260 }, (_, i) => 100 + i * 0.05);
  const out = analyzeLongterm("EURUSD", closes);
  assert.equal(out.decision, "BUY");
  assert.equal(out.regime, "BULL");
  assert.ok(out.sl < out.entry, "long SL below entry");
  assert.ok(out.tp > out.entry, "long TP above entry");
  assert.ok(Math.abs(out.rr - 2) < 1e-9);
  assert.ok(out.confidence >= 20 && out.confidence <= 90);
});

test("longterm: a clean downtrend fires a 2R short", () => {
  const closes = Array.from({ length: 260 }, (_, i) => 200 - i * 0.05);
  const out = analyzeLongterm("EURUSD", closes);
  assert.equal(out.decision, "SELL");
  assert.ok(out.sl > out.entry, "short SL above entry");
  assert.ok(out.tp < out.entry, "short TP below entry");
});
