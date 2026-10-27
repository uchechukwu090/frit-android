// ============================================================================
// SIGNAL LIFECYCLE — decideLifecycle() + the signal_state store
// ----------------------------------------------------------------------------
// This is the module that stops the bots firing a brand-new signal (and a
// brand-new TP/SL) at every candle close. Its rules, pinned here:
//   no live signal  -> FIRE_NEW only on an actionable read
//   live signal     -> same direction OR WAIT is always a HOLD (levels frozen)
//   TP / SL / age   -> CLOSE, or CLOSE_AND_FIRE when the trend is still live
//   opposite read   -> CLOSE_AND_FIRE (reversal)
//   volume expansion with the live trend -> RETRIGGER (fresh TP/SL), spaced
//   by minRetriggerGapMs, capped by maxRetriggers, and never chased once 70%
//   of the target is already eaten.
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";
import Database from "better-sqlite3";

import {
  decideLifecycle, getActiveState, openState, retriggerState, closeState,
  markPlaced, touchState, stateKey, LIFECYCLE_DEFAULTS,
} from "../src/strategy/signalstate.js";

const KEY = stateKey("u1", "intraday", "xauusd");
const T0 = 1_760_000_000_000;

function liveState(patch = {}) {
  return {
    key: KEY, user_id: "u1", bot: "intraday", pair: "XAUUSD",
    dir: "BUY", entry: 2000, sl: 1990, tp: 2020, rr: 2,
    status: "ACTIVE", opened_at: T0, last_trigger_at: T0, retriggers: 0,
    placed: 0, last_price: 2000,
    ...patch,
  };
}

const read = (decision, confidence = 85) => ({ decision, confidence });
const expanding = (over = {}) => ({
  available: true, expansion: true, direction: "BUY", score: 82, reasons: ["rvol 2.4x"],
  ...over,
});

// ---------------------------------------------------------------- decisions --

test("lifecycle: nothing live -> only an actionable read opens a signal", () => {
  assert.equal(decideLifecycle({ state: null, analysis: read("BUY") }).action, "FIRE_NEW");
  assert.equal(decideLifecycle({ state: null, analysis: read("WAIT") }).action, "NONE");
  assert.equal(decideLifecycle({ state: null, analysis: read("BUY", 40) }).action, "NONE");
  assert.equal(decideLifecycle({ state: null, analysis: read("SELL", 40), minConf: 30 }).action, "FIRE_NEW");
});

test("lifecycle: a live signal HOLDS through the next candle — no new TP/SL", () => {
  // Same-direction actionable read: the old behaviour fired a NEW signal here.
  const a = decideLifecycle({ state: liveState(), analysis: read("BUY"), price: 2005, now: T0 + 60_000 });
  assert.equal(a.action, "HOLD");
  assert.match(a.reason, /TP\/SL unchanged/);

  // A plain WAIT on the next candle also holds — it never closes the signal.
  const b = decideLifecycle({ state: liveState(), analysis: read("WAIT"), price: 2005, now: T0 + 60_000 });
  assert.equal(b.action, "HOLD");
  assert.match(b.reason, /read is WAIT/);

  // Below the confidence floor behaves exactly like WAIT.
  const c = decideLifecycle({ state: liveState(), analysis: read("BUY", 10), price: 2005, now: T0 + 60_000 });
  assert.equal(c.action, "HOLD");
});

test("lifecycle: TP and SL resolve the signal (and re-fire only if still live)", () => {
  const atTp = { now: T0 + 60_000, price: 2021 };
  assert.equal(decideLifecycle({ state: liveState(), analysis: read("WAIT"), ...atTp }).action, "CLOSE");
  assert.match(decideLifecycle({ state: liveState(), analysis: read("WAIT"), ...atTp }).reason, /take-profit reached/);
  assert.equal(
    decideLifecycle({ state: liveState(), analysis: read("BUY"), ...atTp }).action,
    "CLOSE_AND_FIRE",
    "TP hit but the trend is still actionable -> one fresh signal, not zero",
  );

  const atSl = { now: T0 + 60_000, price: 1989 };
  assert.equal(decideLifecycle({ state: liveState(), analysis: read("WAIT"), ...atSl }).action, "CLOSE");
  assert.match(decideLifecycle({ state: liveState(), analysis: read("WAIT"), ...atSl }).reason, /stop-loss reached/);
  assert.equal(
    decideLifecycle({ state: liveState(), analysis: read("BUY"), ...atSl }).action,
    "CLOSE_AND_FIRE",
    "SL hit but the trend is still actionable -> the cycle restarts, it does not silently end",
  );

  // Without a price (feed cold) neither level is assumed to be hit.
  assert.equal(decideLifecycle({ state: liveState(), analysis: read("BUY"), price: null, now: T0 + 60_000 }).action, "HOLD");
});

test("lifecycle: opposite read closes and fires the reversal", () => {
  const a = decideLifecycle({ state: liveState(), analysis: read("SELL"), price: 2005, now: T0 + 60_000 });
  assert.equal(a.action, "CLOSE_AND_FIRE");
  assert.match(a.reason, /reversal/);
  // An opposite read BELOW the floor is not actionable: it just closes.
  assert.equal(
    decideLifecycle({ state: liveState(), analysis: read("SELL", 10), price: 2005, now: T0 + 60_000 }).action,
    "HOLD",
  );
});

test("lifecycle: a signal dies at maxAge instead of living forever", () => {
  const now = T0 + LIFECYCLE_DEFAULTS.maxAgeMs + 1;
  assert.equal(decideLifecycle({ state: liveState(), analysis: read("WAIT"), price: 2005, now }).action, "CLOSE");
  assert.match(decideLifecycle({ state: liveState(), analysis: read("WAIT"), price: 2005, now }).reason, /expired/);
  assert.equal(decideLifecycle({ state: liveState(), analysis: read("BUY"), price: 2005, now }).action, "CLOSE_AND_FIRE");
});

// ------------------------------------------------------------- retriggering --

test("lifecycle: volume expansion with the trend is the ONLY way to fresh levels", () => {
  const a = decideLifecycle({
    state: liveState(), analysis: read("WAIT"), price: 2005,
    now: T0 + 20 * 60_000, vf: expanding(),
  });
  assert.equal(a.action, "RETRIGGER");
  assert.match(a.reason, /volume expansion forecast with the BUY trend \(score 82\)/);

  // No forecast data -> HOLD, never a retrigger on missing data.
  assert.equal(decideLifecycle({
    state: liveState(), analysis: read("WAIT"), price: 2005, now: T0 + 20 * 60_000, vf: null,
  }).action, "HOLD");

  // Expansion for the OTHER direction is irrelevant to this signal.
  assert.equal(decideLifecycle({
    state: liveState(), analysis: read("WAIT"), price: 2005, now: T0 + 20 * 60_000,
    vf: expanding({ direction: "SELL" }),
  }).action, "HOLD");

  // Forecast present but says no expansion -> HOLD.
  assert.equal(decideLifecycle({
    state: liveState(), analysis: read("WAIT"), price: 2005, now: T0 + 20 * 60_000,
    vf: expanding({ expansion: false, score: 30 }),
  }).action, "HOLD");
});

test("lifecycle: retriggers are spaced, capped and never chase the target", () => {
  const now = T0 + 20 * 60_000;

  // 1. spacing: at least minRetriggerGapMs since the last trigger
  assert.equal(decideLifecycle({
    state: liveState({ last_trigger_at: T0 }), analysis: read("WAIT"), price: 2005,
    now: T0 + LIFECYCLE_DEFAULTS.minRetriggerGapMs - 1, vf: expanding(),
  }).action, "HOLD");

  // 2. cap: maxRetriggers per live signal
  assert.equal(decideLifecycle({
    state: liveState({ retriggers: LIFECYCLE_DEFAULTS.maxRetriggers, last_trigger_at: T0 }),
    analysis: read("WAIT"), price: 2005, now, vf: expanding(),
  }).action, "HOLD");

  // 3. chase limit: once 70% of the TP distance is gone, do not add
  const chased = decideLifecycle({
    state: liveState({ tp: 2020, entry: 2000 }), analysis: read("WAIT"),
    price: 2015, now, vf: expanding(), // (2015-2000)/(2020-2000) = 0.75 > 0.7
  });
  assert.equal(chased.action, "HOLD");
  assert.match(chased.reason, /mostly consumed/);

  // ...but a retrigger is allowed while the target is still mostly ahead.
  assert.equal(decideLifecycle({
    state: liveState(), analysis: read("WAIT"), price: 2005, now, vf: expanding(),
  }).action, "RETRIGGER");
});

test("lifecycle: a SELL signal retriggers only on a SELL-side expansion", () => {
  const state = liveState({ dir: "SELL", entry: 2000, sl: 2010, tp: 1980 });
  const a = decideLifecycle({
    state, analysis: read("WAIT"), price: 1995, now: T0 + 20 * 60_000,
    vf: expanding({ direction: "SELL" }),
  });
  assert.equal(a.action, "RETRIGGER");
  assert.match(a.reason, /with the SELL trend/);
  const b = decideLifecycle({
    state, analysis: read("WAIT"), price: 1995, now: T0 + 20 * 60_000,
    vf: expanding({ direction: "BUY" }),
  });
  assert.equal(b.action, "HOLD");
});

// ------------------------------------------------------------------ storage --

function freshDb() {
  return new Database(":memory:");
}

test("signal_state: open -> retrigger -> close round-trips", () => {
  const db = freshDb();
  assert.equal(getActiveState(db, KEY), null);

  openState(db, {
    key: KEY, userId: "u1", bot: "intraday", pair: "XAUUSD",
    dir: "BUY", entry: 2000, sl: 1990, tp: 2020, rr: 2, now: T0, placed: 1,
  });
  let s = getActiveState(db, KEY);
  assert.ok(s, "the signal must be live after the first fire");
  assert.equal(s.dir, "BUY");
  assert.equal(s.entry, 2000);
  assert.equal(s.tp, 2020);
  assert.equal(s.placed, 1);
  assert.equal(s.retriggers, 0);
  assert.equal(s.opened_at, T0);

  // A retrigger REPLACES the levels but keeps the open time and counts up.
  retriggerState(db, KEY, { entry: 2010, sl: 2002, tp: 2034, rr: 1.5, now: T0 + 20 * 60_000 });
  s = getActiveState(db, KEY);
  assert.equal(s.entry, 2010);
  assert.equal(s.sl, 2002);
  assert.equal(s.tp, 2034);
  assert.equal(s.retriggers, 1);
  assert.equal(s.opened_at, T0, "the open time must not move: it is still ONE signal");
  assert.equal(s.last_trigger_at, T0 + 20 * 60_000);
  assert.equal(s.placed, 1, "the retrigger must not forget an order is already open");

  touchState(db, KEY, { price: 2012, now: T0 + 21 * 60_000 });
  assert.equal(getActiveState(db, KEY).last_price, 2012);

  closeState(db, KEY, "take-profit reached", T0 + 22 * 60_000);
  assert.equal(getActiveState(db, KEY), null, "a closed signal is no longer live");
  const closed = db.prepare("SELECT * FROM signal_state WHERE key = ?").get(KEY);
  assert.equal(closed.status, "CLOSED");
  assert.equal(closed.close_reason, "take-profit reached");
  assert.equal(closed.closed_at, T0 + 22 * 60_000);
});

test("signal_state: markPlaced is what stops a retrigger doubling an open order", () => {
  const db = freshDb();
  openState(db, { key: KEY, userId: "u1", bot: "intraday", pair: "XAUUSD", dir: "BUY", entry: 2000, sl: 1990, tp: 2020 });
  assert.equal(getActiveState(db, KEY).placed, 0, "signals-only fire places nothing");
  markPlaced(db, KEY);
  assert.equal(getActiveState(db, KEY).placed, 1);
  markPlaced(db, "nobody:here:XAUUSD"); // unknown key is a no-op, not a crash

  // Re-opening the same key (reversal) resets the flag for the new signal.
  openState(db, { key: KEY, userId: "u1", bot: "intraday", pair: "XAUUSD", dir: "SELL", entry: 2000, sl: 2010, tp: 1980 });
  assert.equal(getActiveState(db, KEY).placed, 0);
  assert.equal(getActiveState(db, KEY).dir, "SELL");
});

test("signal_state: keys are per user, per bot, per pair", () => {
  assert.equal(stateKey("u1", "intraday", "xauusd"), "u1:intraday:XAUUSD");
  assert.notEqual(stateKey("u1", "intraday", "XAUUSD"), stateKey("u2", "intraday", "XAUUSD"));
  assert.notEqual(stateKey("u1", "intraday", "XAUUSD"), stateKey("u1", "scalping", "XAUUSD"));
});
