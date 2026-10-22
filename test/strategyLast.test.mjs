// ============================================================================
// FRIT STRATEGY — the LAST uncovered modules
// ----------------------------------------------------------------------------
// Companion to strategy.test.mjs / strategy_money.test.mjs / strategyMore.test.mjs.
// Those files covered the pure strategy layer; this file closes the remaining
// three holes:
//
//   options/structures.js — the defined-risk spread builder + cost gate.
//     It had NO coverage at all, so every exported function is exercised here
//     including its guard clauses (wrong leg count, ITM short, debit spread,
//     naked wing, zero/NaN premiums, missing fields).
//
//   pocket/engine.js (DB-backed exports) — the parts strategyMore.test.mjs
//     deliberately skipped as "not pure": the pullback-arm table, the
//     calibration bucket, the paper ledger and its settlement/stats.
//     Driven against an in-memory better-sqlite3 (`:memory:`), the same way
//     billing.test.mjs does it. The module's own ensurePocketTables() builds
//     the schema, so the tests use the production DDL verbatim.
//
//   strategy/risk.js — only the parts strategy_money.test.mjs did NOT cover:
//     the constructor, the env-driven limits, pnlUsd()/riskUsd(), the
//     dollar-override path, the exact threshold boundaries, the daily-loss
//     bookkeeping rules and status(). The four headline guards (max-open,
//     per-trade cap, group cap, kill-switch) are already covered there and
//     are deliberately NOT duplicated — only their boundaries are.
//
// Everything is hermetic: in-memory SQLite, no network, no server boot.
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";
import Database from "better-sqlite3";

import {
  creditSpread, ironCondor, netCredit, creditGate,
  spreadGreeks, priceStructureGreeks, probProfit,
} from "../src/strategy/options/structures.js";
import { greeks } from "../src/strategy/options/pricing.js";
import {
  upsertArm, getArm, clearArm, calibFor, ensurePocketTables,
  recordPaper, settleOpen, pocketStats,
} from "../src/strategy/pocket/engine.js";
import { PortfolioRisk, usdExposureSign, groupOf, riskLimits } from "../src/strategy/risk.js";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

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
// 1. options/structures.js — defined-risk spreads + the cost gate
// ===========================================================================
/** A bear-call vertical: short 110, long 120, spot 100. */
const bearCall = (over = {}) => ({
  S: 100,
  legs: [
    { type: "call", strike: 110, side: "short", mid: 2.0, bid: 1.9, ask: 2.1 },
    { type: "call", strike: 120, side: "long", mid: 0.5, bid: 0.4, ask: 0.6 },
  ],
  ...over,
});

/** A bull-put vertical: short 90, long 80, spot 100. */
const bullPut = (over = {}) => ({
  S: 100,
  legs: [
    { type: "put", strike: 90, side: "short", mid: 2.0, bid: 1.9, ask: 2.1 },
    { type: "put", strike: 80, side: "long", mid: 0.5, bid: 0.4, ask: 0.6 },
  ],
  ...over,
});

test("structures: a bear call credit spread prices its credit, width and breakeven", () => {
  const s = creditSpread(bearCall());
  assert.equal(s.kind, "bear_call_spread");
  assert.equal(s.credit, 1.5);
  assert.equal(s.width, 10);
  assert.equal(s.maxLoss, 8.5);
  assert.equal(s.maxProfit, 1.5);
  assert.deepEqual(s.breakevens, [111.5]);
  assert.ok(near(s.rr, 1.5 / 8.5));
  assert.equal(s.maxProfitDollars, 150);  // 1.5 x 100 shares
  assert.equal(s.maxLossDollars, 850);
});

test("structures: a bull put credit spread is labelled and breaks even below the short strike", () => {
  const s = creditSpread(bullPut());
  assert.equal(s.kind, "bull_put_spread");
  assert.equal(s.width, 10);
  assert.equal(s.maxLoss, 8.5);
  assert.deepEqual(s.breakevens, [88.5]);
});

test("structures: a spread needs exactly two legs of the same type", () => {
  assert.equal(creditSpread({ S: 100, legs: [] }), null);
  assert.equal(creditSpread({ S: 100, legs: [bearCall().legs[0]] }), null);
  assert.equal(creditSpread({ S: 100, legs: [...bearCall().legs, { type: "call", strike: 130, side: "short", mid: 0.1 }] }), null);
  assert.equal(creditSpread({ S: 100, legs: "not-an-array" }), null);
  // a call short leg paired with a put long leg is not a spread
  assert.equal(creditSpread({ S: 100, legs: [
    { type: "call", strike: 110, side: "short", mid: 2 },
    { type: "put", strike: 120, side: "long", mid: 0.5 },
  ] }), null);
});

test("structures: a short strike at or inside the money is rejected", () => {
  assert.equal(creditSpread(bearCall({ S: 110 })), null, "call short at the money is ITM");
  assert.equal(creditSpread(bearCall({ S: 111 })), null, "call short above the money is ITM");
  assert.equal(creditSpread(bullPut({ S: 90 })), null, "put short at the money is ITM");
  assert.equal(creditSpread(bullPut({ S: 89 })), null, "put short above the money is ITM");
});

test("structures: a long leg that does not cap the short leg is rejected as naked", () => {
  const nakedCall = bearCall();
  nakedCall.legs[1].strike = 100; // long strike BELOW the short strike
  assert.equal(creditSpread(nakedCall), null);
  const nakedPut = bullPut();
  nakedPut.legs[1].strike = 95;   // long strike ABOVE the short strike
  assert.equal(creditSpread(nakedPut), null);
});

test("structures: a debit spread (no credit) is rejected", () => {
  const debit = bearCall();
  debit.legs[0].mid = 0.5;
  debit.legs[1].mid = 1.0;
  assert.equal(creditSpread(debit), null);
  const flat = bearCall();
  flat.legs[0].mid = 1.0;
  flat.legs[1].mid = 1.0;
  assert.equal(creditSpread(flat), null, "a zero credit is not a credit spread");
});

test("structures: a negative or NaN premium is rejected but a zero premium is legal", () => {
  const neg = bearCall();
  neg.legs[0].mid = -1;
  assert.equal(creditSpread(neg), null);
  const negLong = bearCall();
  negLong.legs[1].mid = -0.5;
  assert.equal(creditSpread(negLong), null);
  const nan = bearCall();
  nan.legs[0].mid = NaN;
  assert.equal(creditSpread(nan), null);
  const zero = bearCall();
  zero.legs[1].mid = 0;
  const s = creditSpread(zero);
  assert.ok(s, "a zero long premium is still a valid credit spread");
  assert.equal(s.credit, 2, "the whole short premium is the credit");
});

test("structures: a non-positive spot price is rejected", () => {
  assert.equal(creditSpread(bearCall({ S: 0 })), null);
  assert.equal(creditSpread(bearCall({ S: -5 })), null);
  assert.equal(creditSpread(bearCall({ S: undefined })), null);
});

test("structures: the contract count scales the dollar figures and dte is passed through", () => {
  const s = creditSpread(bearCall({ contracts: 3, dte: 40 }));
  assert.equal(s.contracts, 3);
  assert.equal(s.dte, 40);
  assert.equal(s.maxProfitDollars, 450);
  assert.equal(s.maxLossDollars, 2550);
});

test("structures: an iron condor prices both wings and both breakevens", () => {
  const ic = ironCondor({
    S: 100, putShort: 90, putLong: 80, callShort: 110, callLong: 120,
    putCredit: 1, callCredit: 1.2, contracts: 1, dte: 35,
  });
  assert.equal(ic.kind, "iron_condor");
  assert.equal(ic.credit, 2.2);
  assert.equal(ic.width, 10);
  assert.equal(ic.maxLoss, 7.8);
  assert.ok(near(ic.rr, 2.2 / 7.8));
  assert.deepEqual(ic.breakevens, [87.8, 112.2]);
  assert.ok(near(ic.maxProfitDollars, 220));
  assert.ok(near(ic.maxLossDollars, 780));
});

test("structures: an iron condor needs its four strikes strictly straddling the spot", () => {
  const base = { S: 100, putShort: 90, putLong: 80, callShort: 110, callLong: 120, putCredit: 1, callCredit: 1 };
  assert.ok(ironCondor(base));
  assert.equal(ironCondor({ ...base, putLong: 95 }), null, "put long must sit below put short");
  assert.equal(ironCondor({ ...base, putShort: 105 }), null, "put short must sit below spot");
  assert.equal(ironCondor({ ...base, callShort: 100 }), null, "call short must sit above spot");
  assert.equal(ironCondor({ ...base, callLong: 105 }), null, "call long must sit above call short");
});

test("structures: an iron condor with a missing or non-positive credit is rejected", () => {
  const base = { S: 100, putShort: 90, putLong: 80, callShort: 110, callLong: 120, putCredit: 1, callCredit: 1 };
  assert.equal(ironCondor({ ...base, putCredit: 0 }), null);
  assert.equal(ironCondor({ ...base, callCredit: -1 }), null);
  assert.equal(ironCondor({ ...base, putCredit: undefined }), null, "a missing credit is not a credit");
});

test("structures: an iron condor whose credits exceed the narrowest wing is rejected", () => {
  const ic = ironCondor({
    S: 100, putShort: 90, putLong: 89, callShort: 110, callLong: 120,
    putCredit: 0.6, callCredit: 0.6,
  });
  assert.equal(ic, null, "credits wider than the wing leave no defined risk");
});

test("structures: an iron condor reports four legs and puts the premium on the put wing only", () => {
  const ic = ironCondor({
    S: 100, putShort: 90, putLong: 80, callShort: 110, callLong: 120,
    putCredit: 1, callCredit: 1.2,
  });
  assert.equal(ic.legs.length, 4);
  assert.equal(ic.legs[0].side, "short");
  assert.equal(ic.legs[0].mid, 1);
  assert.equal(ic.legs[1].mid, 0);
  assert.equal(ic.legs[2].mid, 0, "the call wing carries no per-leg premium");
  assert.equal(ic.legs[3].mid, 0);
});

test("structures: net credit is the gross credit minus a per-contract fee for every leg", () => {
  const s = creditSpread(bearCall());
  const net = netCredit(s, { perContract: 0.65 });
  assert.ok(near(net, 150 - 0.65 * 2));
});

test("structures: net credit is null when there is no structure", () => {
  assert.equal(netCredit(null), null);
  assert.equal(netCredit(undefined), null);
});

test("structures: a missing or zero fee spec means no fees at all", () => {
  const s = creditSpread(bearCall());
  assert.equal(netCredit(s, {}), 150, "an undefined perContract is treated as zero cost");
  assert.equal(netCredit(s, { perContract: 0 }), 150);
});

test("structures: the fee scales with the number of contracts", () => {
  const s = creditSpread(bearCall({ contracts: 3 }));
  assert.ok(near(netCredit(s, { perContract: 0.65 }), 450 - 0.65 * 2 * 3));
});

test("structures: the credit gate blocks a structure whose fees consume the whole credit", () => {
  const tiny = bearCall();
  tiny.legs[0].mid = 0.51;
  tiny.legs[1].mid = 0.5;
  const s = creditSpread(tiny);
  const gate = creditGate(s, { perContract: 0.65 });
  assert.equal(gate.ok, false);
  assert.match(gate.reason, /fees consume the credit/);
  assert.notEqual(gate.weak, true, "a blocked structure is not merely weak");
});

test("structures: the credit gate flags a weak structure when fees eat more than 30% of the credit", () => {
  const thin = bearCall();
  thin.legs[0].mid = 0.5199;
  thin.legs[1].mid = 0.5;
  // $1.99 of credit, $0.61 of fees -> 69.3% of the credit survives
  const gate = creditGate(creditSpread(thin), { perContract: 0.305 });
  assert.equal(gate.ok, true);
  assert.equal(gate.weak, true);
  assert.match(gate.reason, /fees eat \d+% of credit/);
});

test("structures: the credit gate is not weak when exactly 70% of the credit survives", () => {
  const edge = bearCall();
  edge.legs[0].mid = 0.52;
  edge.legs[1].mid = 0.5;
  // $2.00 of credit, $0.60 of fees -> exactly 70% survives
  const gate = creditGate(creditSpread(edge), { perContract: 0.3 });
  assert.equal(gate.ok, true);
  assert.equal(gate.weak, false, "the weak threshold is exclusive: exactly 70% is not weak");
  assert.equal(gate.reason, "");
});

test("structures: a clean structure passes the gate with no fee warning", () => {
  const gate = creditGate(creditSpread(bearCall()), { perContract: 0.65 });
  assert.equal(gate.ok, true);
  assert.equal(gate.weak, false);
  assert.equal(gate.reason, "");
});

test("structures: the credit gate refuses a missing structure", () => {
  const gate = creditGate(null);
  assert.equal(gate.ok, false);
  assert.equal(gate.reason, "no structure");
});

test("structures: spread greeks negate the short leg and add the long leg", () => {
  const out = spreadGreeks([
    { side: "short", delta: -0.45, gamma: 0.05, theta: -0.10, vega: 0.20, rho: 0.30 },
    { side: "long", delta: 0.30, gamma: 0.04, theta: -0.08, vega: 0.18, rho: 0.25 },
  ]);
  assert.ok(near(out.delta, 0.75), `delta ${out.delta}`);
  assert.ok(near(out.gamma, -0.01), `gamma ${out.gamma}`);
  assert.ok(near(out.theta, 0.02), `theta ${out.theta}`);
  assert.ok(near(out.vega, -0.02), `vega ${out.vega}`);
  assert.ok(near(out.rho, -0.05), `rho ${out.rho}`);
});

test("structures: a missing or non-numeric greek counts as zero and a leg without a side is a long", () => {
  const out = spreadGreeks([
    { delta: "abc", gamma: undefined },
    { delta: 0.5 },
  ]);
  assert.equal(out.delta, 0.5);
  assert.equal(out.gamma, 0);
  assert.equal(out.theta, 0);
});

test("structures: spread greeks on an empty list are all zero", () => {
  const out = spreadGreeks([]);
  assert.deepEqual(out, { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 });
  assert.deepEqual(spreadGreeks(null), { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 });
});

test("structures: priced structure greeks skip a leg with no volatility, no spot or no type", () => {
  const T = 30 / 365;
  const out = priceStructureGreeks([
    { S: 100, strike: 100, type: "call", side: "short", sigma: 0.2 },
    { S: 100, strike: 100, type: "call", side: "long", sigma: 0 },   // no vol
    { strike: 100, type: "call", side: "long", sigma: 0.2 },         // no spot
    { S: 100, strike: 100, type: "banana", side: "long", sigma: 0.2 }, // no type
  ], { T });
  const solo = greeks({ S: 100, K: 100, T, r: 0, q: 0, sigma: 0.2, type: "call" });
  assert.ok(near(out.delta, -solo.delta), "only the priced short leg contributes");
  assert.ok(near(out.vega, -solo.vega));
});

test("structures: priced structure greeks net two priced legs", () => {
  const T = 30 / 365;
  const out = priceStructureGreeks([
    { S: 100, strike: 100, type: "call", side: "short", sigma: 0.2 },
    { S: 100, strike: 100, type: "call", side: "long", sigma: 0.3 },
  ], { T });
  const short = greeks({ S: 100, K: 100, T, r: 0, q: 0, sigma: 0.2, type: "call" });
  const long = greeks({ S: 100, K: 100, T, r: 0, q: 0, sigma: 0.3, type: "call" });
  assert.ok(near(out.delta, -short.delta + long.delta));
});

test("structures: prob profit is null when the inputs cannot support a distribution", () => {
  assert.equal(probProfit({ S: 0, breakeven: 95, sigma: 0.2, T: 0.08 }), null);
  assert.equal(probProfit({ S: 100, breakeven: 95, sigma: 0, T: 0.08 }), null);
  assert.equal(probProfit({ S: 100, breakeven: 95, sigma: -0.2, T: 0.08 }), null);
  assert.equal(probProfit({ S: 100, breakeven: 95, sigma: 0.2, T: 0 }), null);
});

test("structures: prob profit reads high above a nearby breakeven and mirrors below it", () => {
  const args = { S: 100, breakeven: 95, sigma: 0.2, T: 30 / 365 };
  const above = probProfit({ ...args, direction: "above" });
  const below = probProfit({ ...args, direction: "below" });
  assert.ok(above > 0.8 && above < 0.83, `above ${above} out of range`);
  assert.ok(below > 0.17 && below < 0.2, `below ${below} out of range`);
  assert.ok(near(above + below, 1), "the two directions must be complements");
});

test("structures: prob profit with a missing breakeven is null, not NaN", () => {
  // F-50: breakeven was unvalidated, so a missing one produced NaN. NaN
  // compares false against everything, so a caller testing `p > 0` read it
  // as "no edge" instead of "no data". It is now null like every other
  // unusable input.
  const p = probProfit({ S: 100, sigma: 0.2, T: 0.08 });
  assert.equal(p, null, `expected null for a missing breakeven, got ${p}`);
});

// ===========================================================================
// 2. pocket/engine.js — the DB-backed exports (arms, calibration, paper ledger)
// ===========================================================================
function pocketDb() {
  const db = new Database(":memory:");
  ensurePocketTables(db);
  return db;
}

/** Insert a paper row directly so expiry/result can be forced. */
function insertPaper(db, over = {}) {
  const r = {
    id: over.id || `t_${Math.random().toString(36).slice(2, 9)}`,
    user_id: over.user_id ?? "__pocket__",
    symbol: over.symbol ?? "EURUSD",
    direction: over.direction ?? "CALL",
    entry_price: over.entry_price ?? 100,
    entry_ts: over.entry_ts ?? Date.now() - 10 * 60_000,
    expiry_min: over.expiry_min ?? 5,
    expiry_ts: over.expiry_ts ?? Date.now() - 1000, // expired unless overridden
    payout: over.payout ?? 0.8,
    setup_key: over.setup_key ?? null,
    result: over.result ?? "OPEN",
  };
  db.prepare(
    `INSERT INTO pocket_paper (id,user_id,symbol,direction,entry_price,entry_ts,expiry_min,expiry_ts,payout,setup_key,result)
     VALUES (?,?,?,?,?,?,?,?,?,?,?)`
  ).run(r.id, r.user_id, r.symbol, r.direction, r.entry_price, r.entry_ts,
    r.expiry_min, r.expiry_ts, r.payout, r.setup_key, r.result);
  return r;
}

test("pocket: ensurePocketTables creates the paper, calibration and arm tables", () => {
  const db = new Database(":memory:");
  ensurePocketTables(db);
  const names = db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all().map((r) => r.name);
  assert.ok(names.includes("pocket_paper"));
  assert.ok(names.includes("pocket_calib"));
  assert.ok(names.includes("pocket_arms"));
  const idx = db.prepare("SELECT name FROM sqlite_master WHERE type='index'").all().map((r) => r.name);
  assert.ok(idx.includes("idx_pocket_user_sym"));
  ensurePocketTables(db); // idempotent: a second call must not throw
  assert.ok(db.prepare("SELECT COUNT(*) c FROM pocket_paper").get().c === 0);
});

test("pocket: ensurePocketTables migrates an old paper table that has no setup_key column", () => {
  const db = new Database(":memory:");
  db.exec(`CREATE TABLE pocket_paper (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, symbol TEXT NOT NULL, direction TEXT NOT NULL,
    entry_price REAL NOT NULL, entry_ts INTEGER NOT NULL, expiry_min INTEGER NOT NULL,
    expiry_ts INTEGER NOT NULL, settle_price REAL, result TEXT DEFAULT 'OPEN',
    payout REAL DEFAULT 0.8, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)`);
  ensurePocketTables(db);
  const cols = db.prepare("PRAGMA table_info(pocket_paper)").all().map((c) => c.name);
  assert.ok(cols.includes("setup_key"), "the setup_key migration must run");
});

test("pocket: an arm is stored with its direction, bucket and expiry", () => {
  const db = pocketDb();
  const expires = Date.now() + 60_000;
  upsertArm(db, { symbol: "EURUSD", expiryMin: 5, direction: "CALL", setupKey: "EURUSD|5m|strong|ny", expiresTs: expires, note: "pullback" });
  const arm = getArm(db, "EURUSD", 5);
  assert.equal(arm.direction, "CALL");
  assert.equal(arm.setup_key, "EURUSD|5m|strong|ny");
  assert.equal(arm.note, "pullback");
  assert.equal(arm.expires_ts, expires);
});

test("pocket: arming the same symbol and expiry replaces the previous arm", () => {
  const db = pocketDb();
  upsertArm(db, { symbol: "EURUSD", expiryMin: 5, direction: "CALL", expiresTs: Date.now() + 60_000 });
  upsertArm(db, { symbol: "EURUSD", expiryMin: 5, direction: "PUT", expiresTs: Date.now() + 60_000 });
  assert.equal(db.prepare("SELECT COUNT(*) c FROM pocket_arms").get().c, 1);
  assert.equal(getArm(db, "EURUSD", 5).direction, "PUT");
});

test("pocket: an expired arm is not returned and is cleared from the table", () => {
  const db = pocketDb();
  upsertArm(db, { symbol: "EURUSD", expiryMin: 5, direction: "CALL", expiresTs: Date.now() - 1 });
  assert.equal(getArm(db, "EURUSD", 5), null, "an expired arm must never be handed back");
  assert.equal(db.prepare("SELECT COUNT(*) c FROM pocket_arms").get().c, 0, "the expired arm is deleted");
});

test("pocket: clearArm removes the arm and is a no-op when there is none", () => {
  const db = pocketDb();
  upsertArm(db, { symbol: "EURUSD", expiryMin: 5, direction: "CALL", expiresTs: Date.now() + 60_000 });
  clearArm(db, "EURUSD", 5);
  assert.equal(getArm(db, "EURUSD", 5), null);
  clearArm(db, "EURUSD", 5); // no throw on a missing arm
  clearArm(db, "NOSUCH", 1);
});

test("pocket: calibration is read back as a win rate over the sample", () => {
  const db = pocketDb();
  db.prepare("INSERT INTO pocket_calib (bucket, n, wins, updated_at) VALUES (?,?,?,?)").run("EURUSD|5m|strong|ny", 10, 6, Date.now());
  const c = calibFor(db, "EURUSD|5m|strong|ny");
  assert.equal(c.n, 10);
  assert.equal(c.wr, 0.6);
});

test("pocket: an empty or unknown calibration bucket reads as no calibration", () => {
  const db = pocketDb();
  db.prepare("INSERT INTO pocket_calib (bucket, n, wins, updated_at) VALUES (?,?,?,?)").run("empty", 0, 0, Date.now());
  assert.equal(calibFor(db, "empty"), null);
  assert.equal(calibFor(db, "never-seen"), null);
});

test("pocket: calibration on a database without the tables is null, not a crash", () => {
  const db = new Database(":memory:");
  assert.equal(calibFor(db, "anything"), null);
});

test("pocket: a paper trade is recorded with an expiry derived from the signal", () => {
  const db = pocketDb();
  const signal = { symbol: "EURUSD", decision: "CALL", entryRef: 1.1, expiryMin: 5, setupKey: "EURUSD|5m|strong|ny" };
  const row = recordPaper(db, { userId: "u1", signal, payout: 0.85 });
  assert.match(row.id, /^pk_EURUSD_\d+_[a-z0-9]{4}$/);
  const stored = db.prepare("SELECT * FROM pocket_paper WHERE id = ?").get(row.id);
  assert.equal(stored.user_id, "u1");
  assert.equal(stored.direction, "CALL");
  assert.equal(stored.entry_price, 1.1);
  assert.equal(stored.expiry_min, 5);
  assert.equal(stored.expiry_ts - stored.entry_ts, 5 * 60_000);
  assert.equal(stored.payout, 0.85);
  assert.equal(stored.setup_key, "EURUSD|5m|strong|ny");
  assert.equal(stored.result, "OPEN");
});

test("pocket: a paper trade with no setup bucket is stored with a null bucket", () => {
  const db = pocketDb();
  const signal = { symbol: "EURUSD", decision: "PUT", entryRef: 1.1, expiryMin: 1 };
  const row = recordPaper(db, { userId: "u1", signal, payout: 0.8 });
  const stored = db.prepare("SELECT * FROM pocket_paper WHERE id = ?").get(row.id);
  assert.equal(stored.setup_key, null);
  assert.equal(stored.expiry_ts - stored.entry_ts, 60_000);
});

test("pocket: a trade that has not expired yet is not settled", () => {
  const db = pocketDb();
  insertPaper(db, { expiry_ts: Date.now() + 60_000 });
  const out = settleOpen(db, () => ({ price: 101 }));
  assert.equal(out.settled, 0);
  assert.equal(db.prepare("SELECT result FROM pocket_paper").get().result, "OPEN");
});

test("pocket: a CALL that closed higher is a WIN and a CALL that closed lower is a LOSS", () => {
  const db = pocketDb();
  insertPaper(db, { id: "a", symbol: "EURUSD", direction: "CALL", entry_price: 100 });
  insertPaper(db, { id: "b", symbol: "GBPUSD", direction: "CALL", entry_price: 100 });
  const out = settleOpen(db, (sym) => ({ price: sym === "EURUSD" ? 101 : 99 }));
  assert.equal(out.settled, 2);
  assert.equal(db.prepare("SELECT result FROM pocket_paper WHERE id='a'").get().result, "WIN");
  assert.equal(db.prepare("SELECT settle_price FROM pocket_paper WHERE id='a'").get().settle_price, 101);
  assert.equal(db.prepare("SELECT result FROM pocket_paper WHERE id='b'").get().result, "LOSS");
});

test("pocket: a PUT that closed lower is a WIN", () => {
  const db = pocketDb();
  insertPaper(db, { direction: "PUT", entry_price: 100 });
  settleOpen(db, () => ({ price: 99 }));
  assert.equal(db.prepare("SELECT result FROM pocket_paper").get().result, "WIN");
});

test("pocket: a price exactly at the entry settles as a TIE and feeds no calibration", () => {
  const db = pocketDb();
  insertPaper(db, { setup_key: "bucket" });
  const out = settleOpen(db, () => ({ price: 100 }));
  assert.equal(out.settled, 1, "a tie is still a settled row");
  assert.equal(db.prepare("SELECT result FROM pocket_paper").get().result, "TIE");
  assert.equal(calibFor(db, "bucket"), null, "a tie is not a win and must not move the bucket");
});

test("pocket: a trade with no live price is left open for the next tick", () => {
  const db = pocketDb();
  insertPaper(db, { id: "a", symbol: "EURUSD" });
  insertPaper(db, { id: "b", symbol: "GBPUSD" });
  const out = settleOpen(db, (sym) => (sym === "EURUSD" ? null : { price: NaN }));
  assert.equal(out.settled, 0);
  assert.equal(db.prepare("SELECT COUNT(*) c FROM pocket_paper WHERE result='OPEN'").get().c, 2);
});

test("pocket: settling the system scoreboard row feeds the calibration bucket", () => {
  const db = pocketDb();
  insertPaper(db, { id: "w", user_id: "__pocket__", setup_key: "bucket", direction: "CALL" });
  settleOpen(db, () => ({ price: 101 }));
  let c = calibFor(db, "bucket");
  assert.equal(c.n, 1);
  assert.equal(c.wr, 1);

  insertPaper(db, { id: "l", user_id: "__pocket__", setup_key: "bucket", direction: "CALL" });
  settleOpen(db, () => ({ price: 99 }));
  c = calibFor(db, "bucket");
  assert.equal(c.n, 2);
  assert.equal(c.wr, 0.5);
});

test("pocket: a per-user copy of the same trade does not double-count the calibration", () => {
  const db = pocketDb();
  insertPaper(db, { user_id: "u1", setup_key: "bucket" });
  settleOpen(db, () => ({ price: 101 }));
  assert.equal(calibFor(db, "bucket"), null, "only the system scoreboard row feeds the bucket");
  assert.equal(db.prepare("SELECT result FROM pocket_paper").get().result, "WIN");
});

test("pocket: stats with no settled trades say paper only", () => {
  const db = pocketDb();
  const s = pocketStats(db);
  assert.equal(s.trades, 0);
  assert.equal(s.wins, 0);
  assert.equal(s.winRate, null);
  assert.equal(s.expectancy, null);
  assert.equal(s.breakeven, 1 / 1.8);
  assert.match(s.verdict, /no settled trades yet/);
  assert.deepEqual(s.bySymbol, {});
});

test("pocket: stats report the win rate, expectancy and verdict of the settled book", () => {
  const db = pocketDb();
  for (let i = 0; i < 3; i++) insertPaper(db, { result: "WIN" });
  insertPaper(db, { result: "LOSS" });
  const s = pocketStats(db);
  assert.equal(s.trades, 4);
  assert.equal(s.wins, 3);
  assert.equal(s.losses, 1);
  assert.equal(s.winRate, 0.75);
  assert.ok(near(s.expectancy, 0.75 * 0.8 - 0.25));
  assert.match(s.verdict, /POSITIVE expectancy/);
});

test("pocket: stats count ties separately from wins and losses", () => {
  const db = pocketDb();
  insertPaper(db, { result: "WIN" });
  insertPaper(db, { result: "TIE" });
  const s = pocketStats(db);
  assert.equal(s.trades, 1, "a tie is not a settled win or loss");
  assert.equal(s.ties, 1);
});

test("pocket: open trades are excluded from the stats", () => {
  const db = pocketDb();
  insertPaper(db, { result: "OPEN" });
  insertPaper(db, { result: "WIN" });
  const s = pocketStats(db);
  assert.equal(s.trades, 1);
  assert.equal(s.ties, 0);
});

test("pocket: a negative expectancy is reported against the breakeven it needed", () => {
  const db = pocketDb();
  insertPaper(db, { result: "LOSS" });
  const s = pocketStats(db);
  assert.equal(s.winRate, 0);
  assert.ok(near(s.expectancy, -1));
  assert.match(s.verdict, /NEGATIVE expectancy/);
  assert.match(s.verdict, /needs 55\.6% win rate/);
});

test("pocket: stats filter by symbol case-insensitively and report the per-symbol win rate", () => {
  const db = pocketDb();
  insertPaper(db, { symbol: "EURUSD", result: "WIN" });
  insertPaper(db, { symbol: "EURUSD", result: "LOSS" });
  insertPaper(db, { symbol: "GBPUSD", result: "WIN" });
  const all = pocketStats(db);
  assert.equal(all.trades, 3);
  assert.equal(all.bySymbol.EURUSD.n, 2);
  assert.equal(all.bySymbol.EURUSD.winRate, 0.5);
  assert.equal(all.bySymbol.GBPUSD.winRate, 1);

  const only = pocketStats(db, { symbol: "eurusd" });
  assert.equal(only.trades, 2);
  assert.equal(only.wins, 1);
  assert.deepEqual(Object.keys(only.bySymbol), ["EURUSD"]);
});

test("pocket: stats expose the calibration buckets with their win rates", () => {
  const db = pocketDb();
  db.prepare("INSERT INTO pocket_calib (bucket, n, wins, updated_at) VALUES (?,?,?,?)").run("b1", 4, 3, Date.now());
  const s = pocketStats(db);
  assert.equal(s.calibration.length, 1);
  assert.equal(s.calibration[0].bucket, "b1");
  assert.equal(s.calibration[0].n, 4);
  assert.equal(s.calibration[0].winRate, 0.75);
});

// ===========================================================================
// 3. strategy/risk.js — the parts strategy_money.test.mjs did NOT cover
// ===========================================================================
const PIP = { EURUSD: { pipSize: 0.0001, pipValue: 10 }, GBPUSD: { pipSize: 0.0001, pipValue: 10 } };

function riskDb() {
  const db = new Database(":memory:");
  db.exec(`
    CREATE TABLE positions (
      id TEXT PRIMARY KEY, symbol TEXT, action TEXT, lot_size REAL, entry REAL,
      sl REAL, tp REAL, source TEXT, status TEXT DEFAULT 'MONITORING',
      placed_at DATETIME DEFAULT CURRENT_TIMESTAMP, monitoring_at DATETIME,
      resolved_at DATETIME, outcome TEXT, close_price REAL, note TEXT
    );
  `);
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
  ).run(p.id, p.symbol, p.action, p.lot_size, p.entry, p.sl, p.tp, p.source, p.status,
    p.outcome ?? null, p.close_price ?? null, p.resolved_at ?? null);
  return p;
}

/** Mark a position as closed today (SQLite's own UTC clock, like production). */
function resolveToday(db, id) {
  db.prepare("UPDATE positions SET resolved_at = datetime('now') WHERE id = ?").run(id);
}

test("risk: the constructor refuses to build without a database", () => {
  assert.throws(() => new PortfolioRisk(), /requires db/);
  assert.throws(() => new PortfolioRisk({}), /requires db/);
});

test("risk: a nonsensical env override falls back to the documented default", () => {
  withEnv({ RISK_MAX_OPEN: "0", RISK_MAX_TRADE_PCT: "abc", RISK_MAX_DAILY_LOSS_PCT: "-5" }, () => {
    const l = riskLimits();
    assert.equal(l.maxOpen, 4);
    assert.equal(l.maxTradePct, 2);
    assert.equal(l.maxDailyLossPct, 4);
  });
});

test("risk: the position caps are floored to whole positions", () => {
  withEnv({ RISK_MAX_OPEN: "2.9", RISK_MAX_PER_GROUP: "1.5" }, () => {
    const l = riskLimits();
    assert.equal(l.maxOpen, 2);
    assert.equal(l.maxPerGroup, 1);
  });
});

test("risk: pnlUsd is null for an unknown symbol or an incomplete position", () => {
  const risk = new PortfolioRisk({ db: riskDb(), pipConfig: PIP });
  assert.equal(risk.pnlUsd({ symbol: "ZZZZZ", action: "buy", entry: 1.1, close_price: 1.2, lot_size: 0.01 }), null);
  assert.equal(risk.pnlUsd({ symbol: "EURUSD", action: "buy", entry: 1.1, close_price: null, lot_size: 0.01 }), null);
  assert.equal(risk.pnlUsd({ symbol: "EURUSD", action: "buy", entry: null, close_price: 1.2, lot_size: 0.01 }), null);
});

test("risk: pnlUsd converts a closed forex position into dollars through the pip table", () => {
  const risk = new PortfolioRisk({ db: riskDb(), pipConfig: PIP });
  const win = risk.pnlUsd({ symbol: "EURUSD", action: "buy", entry: 1.1, close_price: 1.11, lot_size: 0.01 });
  assert.ok(near(win, 10), `100 pips * $10/pip * 0.01 lot = $10, got ${win}`);
  const loss = risk.pnlUsd({ symbol: "EURUSD", action: "sell", entry: 1.1, close_price: 1.11, lot_size: 0.01 });
  assert.ok(near(loss, -10), `a sell that closed higher is a loss, got ${loss}`);
});

test("risk: pnlUsd prices crypto as dollars per price unit, not in pips", () => {
  const risk = new PortfolioRisk({ db: riskDb(), pipConfig: { BTCUSD: { isCrypto: true } } });
  const p = risk.pnlUsd({ symbol: "BTCUSD", action: "buy", entry: 100, close_price: 110, lot_size: 2 });
  assert.equal(p, 20);
});

test("risk: a position with no lot size is priced as the minimum 0.01 lot", () => {
  const risk = new PortfolioRisk({ db: riskDb(), pipConfig: PIP });
  const p = risk.pnlUsd({ symbol: "EURUSD", action: "buy", entry: 1.1, close_price: 1.11 });
  assert.ok(near(p, 10));
});

test("risk: riskUsd is null without a pip table and for a stop at the entry price", () => {
  const risk = new PortfolioRisk({ db: riskDb(), pipConfig: PIP });
  assert.equal(risk.riskUsd({ symbol: "ZZZZZ", lotSize: 0.01, entry: 1.1, sl: 1.099 }), null);
  assert.equal(risk.riskUsd({ symbol: "EURUSD", lotSize: 0.01, entry: 1.1, sl: 1.1 }), null);
});

test("risk: riskUsd prices the stop distance through the caller's pip table", () => {
  const risk = new PortfolioRisk({ db: riskDb(), pipConfig: PIP });
  assert.ok(near(risk.riskUsd({ symbol: "EURUSD", lotSize: 0.01, entry: 1.1, sl: 1.099 }), 1),
    "10 pips * $10/pip * 0.01 lot = $1");
});

test("risk: riskUsd prices crypto risk in dollars per price unit", () => {
  const risk = new PortfolioRisk({ db: riskDb(), pipConfig: { BTCUSD: { isCrypto: true } } });
  assert.equal(risk.riskUsd({ symbol: "BTCUSD", lotSize: 1, entry: 100, sl: 98 }), 2);
});

test("risk: a caller-supplied dollar risk is capped even when the symbol has no pip row", () => {
  const db = riskDb();
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "ZZZZZ", action: "buy", lotSize: 1, entry: 10, sl: 9, balance: 1000, riskUsdOverride: 40 });
  assert.equal(res.allowed, false);
  assert.match(res.reasons.join(" | "), /Single-trade risk/);
  assert.equal(res.exposure.newRiskUsd, 40);
});

test("risk: a trade risking exactly the 2% cap is inside the cap, one step over is not", () => {
  const db = riskDb();
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const exact = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.08, balance: 1000 });
  assert.equal(exact.exposure.newRiskPct, 2);
  assert.equal(exact.allowed, true, "the cap is exclusive: exactly 2% is inside it");
  const over = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.079, balance: 1000 });
  assert.equal(over.exposure.newRiskPct, 2.1);
  assert.equal(over.allowed, false);
});

test("risk: the per-trade cap is compared on the RAW percentage, so 2.004% is blocked", () => {
  const db = riskDb();
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  // $20.04 of risk on a $1000 book is 2.004% - over the 2% cap BEFORE
  // rounding. F-50: the gate used the 2-decimal ROUNDED value, so this
  // rounded to 2.00 and sailed through. The gate now uses the exact number
  // (the rounded one is only what we display).
  const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.07996, balance: 1000 });
  assert.equal(res.exposure.newRiskPct, 2, "the DISPLAYED value is still 2dp");
  assert.equal(res.allowed, false, "2.004% is over the 2% cap and must be blocked");
  assert.match(res.reasons.join(" "), /exceeds cap 2%/);
});

test("risk: a zero or missing balance falls back to $1000 instead of dividing by zero", () => {
  const db = riskDb();
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.08, balance: 0 });
  assert.equal(res.exposure.newRiskPct, 2, "the $20 risk is measured against the $1000 fallback");
});

test("risk: a symbol with no pip row and no dollar override BLOCKS, because the size cap cannot be applied", () => {
  const db = riskDb();
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "ZZZZZ", action: "buy", lotSize: 1, entry: 10, sl: 9, balance: 1000 });
  assert.equal(res.exposure.newRiskUsd, null);
  assert.equal(res.exposure.newRiskPct, null);
  // F-50: the risk could not be priced, so the per-trade cap was silently
  // SKIPPED and the trade sailed through with no size limit at all. A risk
  // layer that cannot price the risk must block, not admit.
  assert.equal(res.allowed, false, "an unpriceable risk must be blocked, not allowed");
  assert.match(res.reasons.join(" "), /could not be priced/);
});

test("risk: an explicit dollar override is used even when the symbol has no pip row", () => {
  const db = riskDb();
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  // The Deriv/Alpaca paths know the real dollar risk of the order and pass
  // it in — that is how a symbol without a pip row is still capped.
  const res = risk.check({ symbol: "ZZZZZ", action: "buy", lotSize: 0, entry: 10, sl: 9, balance: 1000, riskUsdOverride: 20 });
  assert.equal(res.exposure.newRiskUsd, 20);
  assert.equal(res.exposure.newRiskPct, 2);
  assert.equal(res.allowed, true, "exactly 2% with a known risk is inside the cap");
});

test("risk: the correlation guard does not apply to symbols outside every group", () => {
  const db = riskDb();
  for (let i = 0; i < 3; i++) addPos(db, { id: `o${i}`, symbol: "ZZZ", lot_size: 0.01, entry: 10, sl: 9.99 });
  const risk = new PortfolioRisk({ db, pipConfig: { ZZZ: { pipSize: 0.01, pipValue: 1 } } });
  const res = risk.check({ symbol: "ZZZ", action: "buy", lotSize: 0.01, entry: 10, sl: 9.99, balance: 1000 });
  assert.equal(res.exposure.groupCounts.OTHER, 4);
  assert.equal(res.allowed, true, "ungrouped symbols are exempt from the per-group cap");
});

test("risk: the group cap counts the prospective trade and stays quiet while the book is AT the cap", () => {
  const db = riskDb();
  addPos(db, { id: "a", symbol: "EURUSD" });
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.099, balance: 1000 });
  assert.equal(res.exposure.groupCounts.USD_MAJORS, 2, "the open book and the prospective trade are counted together");
  assert.equal(res.allowed, true, "two positions in a group are AT the cap of 2, not over it");
});

test("risk: same-direction USD risk stacks into the net exposure cap", () => {
  const db = riskDb();
  addPos(db, { id: "a", symbol: "EURUSD", action: "buy", lot_size: 0.1, entry: 1.1, sl: 1.095 });
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  withEnv({ RISK_MAX_TRADE_PCT: "10" }, () => {
    const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.1, entry: 1.1, sl: 1.095, balance: 1000, riskUsdOverride: 50 });
    // open $50 short-USD + new $50 short-USD = $100 = 10% of a $1000 book
    assert.equal(res.exposure.usdNetPct, 10);
    assert.equal(res.allowed, false);
    assert.match(res.reasons.join(" | "), /Net USD exposure/);
  });
});

test("risk: opposite-direction USD exposure nets out instead of stacking", () => {
  const db = riskDb();
  addPos(db, { id: "a", symbol: "EURUSD", action: "buy", lot_size: 0.1, entry: 1.1, sl: 1.095 });
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  withEnv({ RISK_MAX_TRADE_PCT: "10" }, () => {
    const res = risk.check({ symbol: "EURUSD", action: "sell", lotSize: 0.1, entry: 1.1, sl: 1.105, balance: 1000, riskUsdOverride: 50 });
    assert.equal(res.exposure.usdNetPct, 0, "a long-USD open position cancels a short-USD trade");
    assert.equal(res.allowed, true);
  });
});

test("risk: a loss exactly on the daily limit trips the kill-switch", () => {
  const db = riskDb();
  // Crypto math is exact: -$40 on a $1000 book is exactly 4.00% of the day.
  addPos(db, { id: "l", symbol: "BTCUSD", status: "RESOLVED", outcome: "loss", lot_size: 4, entry: 100, close_price: 90 });
  resolveToday(db, "l");
  const risk = new PortfolioRisk({ db, pipConfig: { BTCUSD: { isCrypto: true } } });
  const res = risk.check({ symbol: "BTCUSD", action: "buy", lotSize: 1, entry: 100, sl: 99, balance: 1000, riskUsdOverride: 1 });
  assert.equal(res.exposure.realizedTodayUsd, -40);
  assert.equal(res.allowed, false);
  assert.match(res.reasons.join(" | "), /KILL-SWITCH/);
});

test("risk: a loss just under the daily limit does not trip the kill-switch", () => {
  const db = riskDb();
  addPos(db, { id: "l", symbol: "BTCUSD", status: "RESOLVED", outcome: "loss", lot_size: 4, entry: 100, close_price: 91 }); // -$36 = 3.6%
  resolveToday(db, "l");
  const risk = new PortfolioRisk({ db, pipConfig: { BTCUSD: { isCrypto: true } } });
  const res = risk.check({ symbol: "BTCUSD", action: "buy", lotSize: 1, entry: 100, sl: 99, balance: 1000, riskUsdOverride: 1 });
  assert.equal(res.exposure.realizedTodayUsd, -36);
  assert.equal(res.allowed, true);
});

test("risk: a forex loss of 400 pips is a hair over the 4% limit and trips the switch", () => {
  const db = riskDb();
  addPos(db, { id: "l", status: "RESOLVED", outcome: "loss", lot_size: 0.01, entry: 1.1, close_price: 1.06 });
  resolveToday(db, "l");
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.099, balance: 1000 });
  assert.equal(res.exposure.realizedTodayUsd, -40, "400 pips * $10/pip * 0.01 lot = $40");
  assert.equal(res.allowed, false);
  assert.match(res.reasons.join(" | "), /KILL-SWITCH/);
});

test("risk: only resolved wins and losses count toward the daily loss total", () => {
  const db = riskDb();
  addPos(db, { id: "br", status: "RESOLVED", outcome: "breakeven", lot_size: 0.01, entry: 1.1, close_price: 1.06 });
  resolveToday(db, "br");
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.099, balance: 1000 });
  assert.equal(res.exposure.realizedTodayUsd, 0);
  assert.equal(res.allowed, true, "a breakeven close is not a loss");
});

test("risk: yesterday's loss does not count toward today's kill-switch", () => {
  const db = riskDb();
  addPos(db, { id: "old", status: "RESOLVED", outcome: "loss", lot_size: 0.01, entry: 1.1, close_price: 1.06 });
  db.prepare("UPDATE positions SET resolved_at = datetime('now','-1 day') WHERE id = 'old'").run();
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.099, balance: 1000 });
  assert.equal(res.exposure.realizedTodayUsd, 0);
  assert.equal(res.allowed, true);
});

test("risk: a winning day reports positive realized P&L and never trips the kill-switch", () => {
  const db = riskDb();
  addPos(db, { id: "w", status: "RESOLVED", outcome: "win", lot_size: 0.01, entry: 1.1, close_price: 1.11 });
  resolveToday(db, "w");
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check({ symbol: "EURUSD", action: "buy", lotSize: 0.01, entry: 1.1, sl: 1.099, balance: 1000 });
  assert.equal(res.exposure.realizedTodayUsd, 10);
  assert.equal(res.allowed, true);
  assert.equal(risk.status(1000).killSwitchTripped, false);
});

test("risk: status reports the open book, the day's P&L and the kill-switch state", () => {
  const db = riskDb();
  addPos(db, { id: "o1", symbol: "EURUSD", action: "buy", lot_size: 0.01, entry: 1.1, sl: 1.099, tp: 1.12, source: "test" });
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const st = risk.status(1000);
  assert.equal(st.openCount, 1);
  assert.equal(st.open[0].id, "o1");
  assert.equal(st.open[0].source, "test");
  assert.equal(st.open[0].tp, 1.12);
  assert.equal(st.realizedTodayUsd, 0);
  assert.equal(st.killSwitchTripped, false);
  assert.equal(st.riskStateError, null);
  assert.equal(st.limits.maxOpen, 4);
});

test("risk: status surfaces a read error instead of pretending the book is empty", () => {
  const db = new Database(":memory:"); // no positions table at all
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const st = risk.status(1000);
  assert.equal(st.openCount, 0);
  assert.ok(st.riskStateError, "the unreadable state must be reported, not swallowed");
});

test("risk: the kill-switch measurement treats a zero balance as $1000, not as infinite risk", () => {
  const db = riskDb();
  addPos(db, { id: "l", status: "RESOLVED", outcome: "loss", lot_size: 0.01, entry: 1.1, close_price: 1.095 });
  resolveToday(db, "l"); // -$5 = 0.5% of a $1000 book
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  assert.equal(risk.status(0).killSwitchTripped, false);
});

test("risk: check never throws on an empty request and BLOCKS it (the risk cannot be priced)", () => {
  const db = riskDb();
  const risk = new PortfolioRisk({ db, pipConfig: PIP });
  const res = risk.check();
  // F-50: an argument-less request used to come back allowed:true with no
  // reasons - no symbol, no priced risk, no guard applied. It still must not
  // THROW, but it must not be ALLOWED either.
  assert.equal(res.allowed, false, "a request with no priced risk must be blocked");
  assert.match(res.reasons.join(" "), /could not be priced/);
  assert.equal(res.exposure.openPositions, 0);
  assert.equal(res.exposure.newRiskUsd, null);
  assert.equal(res.limits.maxOpen, 4);
});

test("risk: the USD exposure sign is case-insensitive and defaults to a long", () => {
  assert.equal(usdExposureSign("eurusd", "buy"), -1);
  assert.equal(usdExposureSign("EURUSD", "SELL"), 1);
  assert.equal(usdExposureSign("usdcad", "buy"), 1);
  assert.equal(usdExposureSign("", "buy"), 0);
  assert.equal(usdExposureSign(null, "sell"), 0);
});

test("risk: group lookup is case-insensitive and unknown symbols land in OTHER", () => {
  assert.equal(groupOf("eurusd"), "USD_MAJORS");
  assert.equal(groupOf("xauusd"), "METALS");
  assert.equal(groupOf(""), "OTHER");
  assert.equal(groupOf(null), "OTHER");
});

test("risk: a symbol listed in two correlation groups is counted in the first one only", () => {
  assert.equal(groupOf("USDJPY"), "USD_QUOTE", "USDJPY sits in USD_QUOTE and JPY_CROSS; the first match wins");
});
