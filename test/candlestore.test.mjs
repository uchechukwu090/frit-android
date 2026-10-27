// ============================================================================
// CandleStore.getCandles — n means the NEWEST n bars, oldest-first
// ----------------------------------------------------------------------------
// The store used to `ORDER BY open_time ASC LIMIT n`, so with 300 rows kept it
// handed back the OLDEST window and silently dropped the live edge — every
// consumer (pocket closes, the MTF matrix, now the volume forecaster) read
// hours-stale candles. These pin the fix.
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";
import Database from "better-sqlite3";

import { CandleStore } from "../src/strategy/feed/candle_store.js";

const IV = "5m";
const STEP = 5 * 60_000;

function seeded(rows = 300) {
  const db = new Database(":memory:");
  const store = new CandleStore(db);
  const ins = db.prepare(`
    INSERT OR REPLACE INTO feed_candles (symbol, interval, open_time, o, h, l, c, v, source)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'test')
  `);
  for (let i = 0; i < rows; i++) {
    ins.run("XAUUSD", IV, i * STEP, 100 + i, 101 + i, 99 + i, 100.5 + i, 10 + i);
  }
  return { db, store, rows };
}

test("getCandles: returns the NEWEST n bars, oldest-first", () => {
  const { store, rows } = seeded(300);
  const out = store.getCandles("XAUUSD", IV, 50);

  assert.equal(out.length, 50);
  assert.equal(out[0].time, (rows - 50) * STEP, "must start at the 50th-newest bar");
  assert.equal(out.at(-1).time, (rows - 1) * STEP, "and end at the newest bar");
  for (let i = 1; i < out.length; i++) assert.ok(out[i].time > out[i - 1].time, "ascending time");
  assert.equal(out.at(-1).close, 100.5 + rows - 1);
  assert.equal(out.at(-1).volume, 10 + rows - 1);
});

test("getCandles: asking for every kept bar returns all of them in order", () => {
  const { store, rows } = seeded(300);
  const out = store.getCandles("XAUUSD", IV, 300);
  assert.equal(out.length, rows);
  assert.equal(out[0].time, 0);
  assert.equal(out.at(-1).time, (rows - 1) * STEP);
});

test("getCandles: an unknown symbol or interval is empty, not a throw", () => {
  const { store } = seeded(300);
  assert.deepEqual(store.getCandles("EURUSD", IV, 50), []);
  assert.deepEqual(store.getCandles("XAUUSD", "1h", 50), []);
});

test("getCandles: the forming candle is appended after the closed ones", () => {
  const { store } = seeded(10);
  const bucket = 10 * STEP;
  store.forming.set(`XAUUSD:${IV}`, {
    openTime: bucket, open: 999, high: 1001, low: 998, close: 1000, volume: 4, count: 2,
  });
  const out = store.getCandles("XAUUSD", IV, 100);
  assert.equal(out.length, 11);
  assert.equal(out.at(-1).time, bucket);
  assert.equal(out.at(-1).close, 1000);
});
