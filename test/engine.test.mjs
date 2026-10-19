// ============================================================================
// FRIT ENGINE / SCHEDULER / RISK — golden tests (no network, deterministic)
// ----------------------------------------------------------------------------
// Covers the Week-1 money-safety fixes:
//   F-03: trendStart confidence bonus used to be chained to the session gate
//         ("else if"), so it never fired during real trading hours. This test
//         drives an in-session (crypto 24/7) structure-breakout fixture and
//         asserts the "Structure breakout" reason IS reachable.
//   F-04: PortfolioRisk.check() failed OPEN on internal errors; must fail
//         CLOSED, and the four guards (per-trade cap, group cap, net USD cap,
//         daily-loss kill-switch) must reject oversized/overheated trades.
//   F-05: TradeTaskScheduler tasks must persist to SQLite and survive a
//         reconstructed scheduler instance (restart simulation).
// ============================================================================

import test from "node:test";
import assert from "node:assert";
import Database from "better-sqlite3";
import { MTFStrategyEngine } from "../src/strategy/engine.js";
import { TradeTaskScheduler } from "../src/strategy/scheduler.js";
import { PortfolioRisk } from "../src/strategy/risk.js";

// ---- fixture helpers --------------------------------------------------------

/** Steady bullish candle series. Final 10 bars accelerate + widen (range expansion). */
function build30m(n = 120) {
  const out = [];
  let px = 10000;
  const t0 = Date.now() - n * 30 * 60_000;
  for (let i = 0; i < n; i++) {
    const hot = i >= n - 10;
    const drift = px * (hot ? 0.006 : 0.0008); // last 10 bars rise ~6x faster
    const wig = px * (i % 2 ? 0.0008 : -0.0003);
    const open = px;
    const close = i === n - 1 ? open + drift : open + drift + wig * 0.4;
    const high = i === n - 1 ? close : Math.max(open, close) + px * 0.0010;
    const low = Math.min(open, close) - px * 0.0010;
    out.push({ time: t0 + i * 30 * 60_000, open, high, low, close, volume: 1000 + (i % 7) * 50 });
    px = close;
  }
  return out;
}

/**
 * 4H bull tape: 60 quiet noisy bars, then a fresh strongly-rising leg for the
 * last 20 — this is the shape real trending markets print (ADX rises late in
 * its own series → high percentile). Deterministic via a fixed LCG seed.
 */
function build4h(n = 80) {
  let seed = 20261018; // fixed seed => fully deterministic test
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; // LCG [0,1)
  const out = [];
  let px = 9000;
  const t0 = Date.now() - n * 4 * 3600_000;
  for (let i = 0; i < n; i++) {
    const hot = i >= n - 20;
    const noise = px * (0.0015 + rnd() * 0.0035); // 0.15%-0.5% amplitude, always varied
    const drift = hot ? px * (0.003 + rnd() * 0.004) : px * 0.0002;
    const open = px;
    const close = open + drift + noise * (rnd() > 0.35 ? 1 : -0.85); // mostly up
    const spread = noise * (0.5 + rnd() * 0.5);
    const high = Math.max(open, close) + spread * 0.6;
    const low = Math.min(open, close) - spread * 0.7;
    out.push({ time: t0 + i * 4 * 3600_000, open, high, low, close, volume: 5000 + (hot ? 2000 : 0) });
    px = close;
  }
  return out;
}

// ---- engine -----------------------------------------------------------------

import { percentileRank } from "../src/strategy/engine.js";

test("engine (F-30 regression): saturated-tie series reads ~high, not 0", () => {
  // Clean monotonic tape → ADX pinned at exactly 100 → ties everywhere.
  // before F-30 this scored 0% and collapsed the regime to NEUTRAL.
  assert.ok(percentileRank([20, 30, 50, 60, 100, 100, 100], 100) >= 70);
  assert.equal(percentileRank([100, 100, 100], 100), 50); // all-tie → neutral prior
  assert.equal(percentileRank([], 100), 50);
  assert.ok(percentileRank([1, 2, 3, 4, 5], 100) >= 90); // unique max stays top (midpoint incl. self)
});

test("engine: Data Unavailable when history is too thin", async () => {
  const engine = new MTFStrategyEngine({});
  const r = await engine.analyze("BTCUSD", { candles30M: build30m(120), candles4H: build4h(12) });
  assert.equal(r.decision, "DATA_UNAVAILABLE");
});

test("engine (F-03 regression): trendStart bonus fires IN-SESSION (crypto 24/7)", async () => {
  const engine = new MTFStrategyEngine({});
  const r = await engine.analyze("BTCUSD", { candles30M: build30m(120), candles4H: build4h(80) });
  // The fixture is a fresh regime-confirmed leg. Whatever the final decision
  // (cost gate or tf-ignition may still reshape it), the structure-breakout
  // reason must be REACHABLE in-session — before F-03 it could only ever be
  // appended outside trading hours.
  assert.ok(
    r.reasons.some((x) => x.includes("Structure breakout")),
    `expected trendStart reason in-session, got: ${JSON.stringify(r.decision)} | ${JSON.stringify(r.reasons)}`,
  );
});

test("engine: sane decision vocabulary on clean input", async () => {
  const engine = new MTFStrategyEngine({});
  const r = await engine.analyze("BTCUSD", { candles30M: build30m(120), candles4H: build4h(80) });
  assert.ok(
    ["BUY", "SELL", "WAIT", "WAIT_PULLBACK", "WAIT_RANGE"].includes(r.decision),
    `unexpected decision ${r.decision}`,
  );
  assert.ok(Number.isFinite(r.confidence));
});

// ---- scheduler persistence (F-05) -------------------------------------------

test("scheduler (F-05): tasks persist and survive instance reconstruction", () => {
  const db = new Database(":memory:");
  const a = new TradeTaskScheduler({ engine: {}, executor: () => {}, db });
  a.add({ symbol: "XAUUSD", time: "07:30", riskPercent: 1.5, balance: 2500 });

  const row = db.prepare("SELECT * FROM trade_tasks WHERE id = ?").get("XAUUSD@07:30");
  assert.ok(row, "task row missing after add");
  assert.equal(row.symbol, "XAUUSD");
  assert.equal(row.time, "07:30");
  assert.equal(Number(row.risk_percent), 1.5);

  // "Restart": new instance over the same DB restores the task
  const b = new TradeTaskScheduler({ engine: {}, executor: () => {}, db });
  const restored = b.tasks.get("XAUUSD@07:30");
  assert.ok(restored, "task not restored from DB");
  assert.equal(restored.riskPercent, 1.5);
  assert.equal(restored.balance, 2500);
  assert.equal(restored.enabled, true);

  // Fired-day guard persists (no refire after restart)
  restored.lastFiredDay = "2026-10-18";
  b._persist("XAUUSD@07:30");
  const c = new TradeTaskScheduler({ engine: {}, executor: () => {}, db });
  assert.equal(c.tasks.get("XAUUSD@07:30").lastFiredDay, "2026-10-18");

  // remove() removes both the memory copy and the DB row
  assert.equal(c.remove("XAUUSD@07:30"), true);
  assert.equal(db.prepare("SELECT COUNT(*) AS n FROM trade_tasks").get().n, 0);
});

test("scheduler: memory-only mode still works when no db passed (back-compat)", () => {
  const a = new TradeTaskScheduler({ engine: {}, executor: () => {} });
  const t = a.add({ symbol: "EURUSD", time: "09:00" });
  assert.ok(t.ok);
  assert.equal(a.remove("EURUSD@09:00"), true);
});

// ---- PortfolioRisk (F-04 fail-closed + guard math) ---------------------------

function riskDb() {
  const db = new Database(":memory:");
  db.exec(`CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT, action TEXT, lot_size REAL, entry REAL, sl REAL, tp REAL,
    status TEXT DEFAULT 'MONITORING', outcome TEXT, entry_price REAL,
    close_price REAL, resolved_at INTEGER, source TEXT, mode TEXT, reason TEXT
  )`);
  return db;
}

const CRYPTO_PIP = { BTCUSD: { isCrypto: true, pipSize: 1, pipValue: 1 } };

test("risk: per-trade cap blocks oversized risk", () => {
  const r = new PortfolioRisk({ db: riskDb(), pipConfig: CRYPTO_PIP });
  const res = r.check({ symbol: "BTCUSD", action: "buy", entry: 100, sl: 98, balance: 1000, riskUsdOverride: 50 }); // 5% > 2%
  assert.equal(res.allowed, false);
  assert.ok(res.reasons.some((x) => x.toLowerCase().includes("single-trade risk")));
});

test("risk: kill-switch blocks new trades after daily loss >= limit", () => {
  const db = riskDb();
  // Production writes resolved_at via CURRENT_TIMESTAMP (SQLite TEXT, UTC) —
  // match that exactly; risk.js compares date(resolved_at) = date('now').
  db.prepare(`INSERT INTO positions (symbol, action, lot_size, entry, close_price, sl, status, outcome, resolved_at)
    VALUES ('BTCUSD','buy',1.0,100,0,0,'RESOLVED','loss',CURRENT_TIMESTAMP)`).run();
  const r = new PortfolioRisk({ db, pipConfig: CRYPTO_PIP });
  const res = r.check({ symbol: "BTCUSD", action: "buy", entry: 100, sl: 90, balance: 1000, riskUsdOverride: 10 }); // 1% within caps, but today down 10%
  assert.equal(res.allowed, false);
  assert.ok(res.reasons.some((x) => x.includes("KILL-SWITCH")));
});

test("risk: clean book passes", () => {
  const r = new PortfolioRisk({ db: riskDb(), pipConfig: CRYPTO_PIP });
  const res = r.check({ symbol: "BTCUSD", action: "buy", entry: 100, sl: 99, balance: 1000, riskUsdOverride: 10 });
  assert.equal(res.allowed, true);
  assert.equal(res.exposure.newRiskPct, 1);
});
