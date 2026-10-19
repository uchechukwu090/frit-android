// Billing unit tests — in-memory SQLite, no network, no server boot.
import test from "node:test";
import assert from "node:assert/strict";
import Database from "better-sqlite3";
import { createBilling } from "../src/billing.js";

const SUB_TIERS = {
  free: { credits_mo: 20, features: ["chat", "image"] },
  pro: { credits_mo: 200, features: ["chat", "image", "video", "creative"] },
};
const CREDIT_COSTS = { turn: 1, image: 4, clip: 65, creative: 2, market_data: 0 };
const FEATURE_CHARGE = { chat: "turn", image: "image", video: "clip", creative: "creative", market_data: "market_data" };
const FEATURE_LABELS = { chat: "Chat", image: "Image generation", video: "Video generation" };

function setup({ tier = "free", month = "2026-10" } = {}) {
  const db = new Database(":memory:");
  db.exec(`
    CREATE TABLE user_credits (user_id TEXT PRIMARY KEY, balance INTEGER DEFAULT 0, total_earned INTEGER DEFAULT 0, total_spent INTEGER DEFAULT 0, plan_grant_month TEXT, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE credit_txns (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, amount INTEGER, kind TEXT, note TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE provider_jobs (provider_id TEXT PRIMARY KEY, user_id TEXT, kind TEXT, provider TEXT DEFAULT 'openrouter', charged INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE api_keys (key_hash TEXT PRIMARY KEY, user_id TEXT, label TEXT);
    CREATE TABLE subscriptions (user_id TEXT PRIMARY KEY, tier TEXT, status TEXT);
  `);
  const state = { tier, month };
  const b = createBilling({
    db, SUB_TIERS, CREDIT_COSTS, FEATURE_CHARGE, FEATURE_LABELS,
    activeTier: () => state.tier,
    monthStr: () => state.month,
  });
  return { db, b, state };
}

test("monthly grant only goes to real installs (anti-farming)", () => {
  const { b } = setup();
  assert.equal(b.grantMonthlyCredits("ghost"), 0);
});

test("monthly grant is idempotent within a month and re-grants next month", () => {
  const { db, b, state } = setup();
  db.prepare("INSERT INTO api_keys (key_hash, user_id) VALUES ('h','u1')").run();
  assert.equal(b.grantMonthlyCredits("u1"), 20);
  assert.equal(b.grantMonthlyCredits("u1"), 20);
  state.month = "2026-11";
  assert.equal(b.grantMonthlyCredits("u1"), 40);
});

test("mid-month upgrade receives the new plan's credits", () => {
  const { db, b, state } = setup();
  db.prepare("INSERT INTO api_keys (key_hash, user_id) VALUES ('h','u1')").run();
  b.grantMonthlyCredits("u1"); // free: 20
  state.tier = "pro";
  assert.equal(b.grantMonthlyCredits("u1"), 220);
});

test("checkEntitlement: locked feature, insufficient credits, ok — and never debits", () => {
  const { db, b } = setup();
  db.prepare("INSERT INTO api_keys (key_hash, user_id) VALUES ('h','u1')").run();
  const locked = b.checkEntitlement("u1", "video");
  assert.equal(locked.ok, false);
  assert.equal(locked.code, "feature_locked");
  const ok = b.checkEntitlement("u1", "image");
  assert.equal(ok.ok, true);
  assert.equal(ok.cost, 4);
  assert.equal(b.creditBalance("u1"), 20); // gate alone must not spend
  b.creditTxn("u2", 2, "test");
  const low = b.checkEntitlement("u2", "image");
  assert.equal(low.ok, false);
  assert.equal(low.code, "insufficient_credits");
  assert.equal(low.needed, 4);
});

test("stt maps to the chat feature", () => {
  const { b } = setup();
  assert.equal(b.planAllows("free", "stt"), true);
  assert.equal(b.planAllows("free", "video"), false);
});

test("debitCredits: debits once, refuses overdraft with no partial debit, zero-cost is free", () => {
  const { db, b } = setup();
  b.creditTxn("u1", 10, "seed");
  assert.deepEqual(b.debitCredits("u1", "image", "x"), { ok: true, balance: 6 });
  assert.equal(b.debitCredits("u1", "clip").ok, false);
  assert.equal(b.creditBalance("u1"), 6);
  const r = b.debitCredits("u1", "market_data");
  assert.equal(r.ok, true);
  assert.equal(r.cost, 0);
  const spends = db.prepare("SELECT COUNT(*) c FROM credit_txns WHERE kind='spend'").get().c;
  assert.equal(spends, 1);
});

test("chargeProviderJob bills exactly once", () => {
  const { db, b } = setup();
  b.creditTxn("u1", 200, "seed");
  b.trackProviderJob("job1", "u1", "clip", "hedra");
  assert.equal(b.chargeProviderJob("job1").ok, true);
  assert.equal(b.creditBalance("u1"), 135);
  const again = b.chargeProviderJob("job1");
  assert.equal(again.already, true);
  assert.equal(b.creditBalance("u1"), 135);
  assert.equal(db.prepare("SELECT charged FROM provider_jobs WHERE provider_id='job1'").get().charged, 1);
});

test("chargeProviderJob: unknown job and short balance leave the job open for retry", () => {
  const { db, b } = setup();
  assert.equal(b.chargeProviderJob("nope").ok, false);
  b.creditTxn("u1", 10, "seed");
  b.trackProviderJob("job2", "u1", "clip");
  assert.equal(b.chargeProviderJob("job2").ok, false);
  assert.equal(db.prepare("SELECT charged FROM provider_jobs WHERE provider_id='job2'").get().charged, 0);
});

test("trackProviderJob is idempotent per provider id", () => {
  const { db, b } = setup();
  b.trackProviderJob("j", "u1", "clip");
  b.trackProviderJob("j", "u2", "avatar");
  const rows = db.prepare("SELECT user_id, kind FROM provider_jobs WHERE provider_id='j'").all();
  assert.equal(rows.length, 1);
  assert.equal(rows[0].user_id, "u1");
});
