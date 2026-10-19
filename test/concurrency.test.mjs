// ============================================================================
// Week 6 (audit) — runBounded unit tests
// ----------------------------------------------------------------------------
// The trading scan now has a hard deadline (F-45). These tests pin the
// behaviour the route relies on: order, concurrency cap, per-item timeout, a
// total budget that stops starting new work, and isolation of failures.
// No network, no fake timers — real short delays keep this fast (<1s).
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";

import { runBounded } from "../src/concurrency.js";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

test("runBounded: preserves input order and passes values through", async () => {
  const { results, summary } = await runBounded([3, 1, 2], async (n) => {
    await sleep(n * 5);
    return n * 10;
  }, { concurrency: 3 });

  assert.deepEqual(results.map((r) => r.value), [30, 10, 20]);
  assert.equal(summary.ok, 3);
  assert.equal(summary.failed, 0);
});

test("runBounded: never exceeds the concurrency cap", async () => {
  let live = 0;
  let peak = 0;
  await runBounded([1, 2, 3, 4, 5, 6, 7, 8], async () => {
    live++;
    peak = Math.max(peak, live);
    await sleep(10);
    live--;
  }, { concurrency: 3 });

  assert.ok(peak <= 3, `peak concurrency was ${peak}`);
  assert.ok(peak >= 2, `expected real concurrency, got ${peak}`);
});

test("runBounded: a slow item is marked timedOut and does not block the rest", async () => {
  const started = Date.now();
  const { results, summary } = await runBounded([10, 500, 10], async (ms) => {
    await sleep(ms);
    return ms;
  }, { concurrency: 3, perItemMs: 60 });

  const elapsed = Date.now() - started;
  assert.equal(results[1].ok, false);
  assert.equal(results[1].timedOut, true);
  assert.equal(results[0].ok, true);
  assert.equal(results[2].ok, true);
  assert.equal(summary.timedOut, 1);
  assert.ok(elapsed < 400, `call took ${elapsed}ms`);
});

test("runBounded: total budget marks unstarted items as skipped", async () => {
  const { results, summary } = await runBounded([1, 2, 3, 4, 5, 6], async (n) => {
    await sleep(30);
    return n;
  }, { concurrency: 1, totalMs: 70 });

  assert.equal(results[0].ok, true);
  assert.ok(summary.skipped >= 3, `expected skips, got ${JSON.stringify(summary)}`);
  assert.ok(results.some((r) => r.skipped === true));
});

test("runBounded: a throwing worker is isolated, not fatal", async () => {
  const { results, summary } = await runBounded(["a", "boom", "c"], async (v) => {
    if (v === "boom") throw new Error("provider said no");
    return v;
  }, { concurrency: 2 });

  assert.equal(results[0].value, "a");
  assert.equal(results[1].ok, false);
  assert.equal(results[1].error, "provider said no");
  assert.equal(results[1].timedOut, false);
  assert.equal(results[2].value, "c");
  assert.equal(summary.failed, 1);
});

test("runBounded: empty input resolves with an empty summary", async () => {
  const { results, summary } = await runBounded([], async () => 1);
  assert.deepEqual(results, []);
  assert.deepEqual(summary, { ok: 0, failed: 0, timedOut: 0, skipped: 0 });
});
