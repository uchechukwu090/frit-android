// ============================================================================
// Bounded fan-out — Week 6 (audit)
// ----------------------------------------------------------------------------
// /api/trading/scan used to `await` one symbol at a time with no overall
// deadline. One black-holed market-data call (Binance/TwelveData with no
// AbortSignal) therefore blocked the whole scan past the client's timeout:
// a 2-symbol scan was measured at >30s while a warm 20-symbol scan took 1.4s.
//
// This helper runs a worker over many items with a concurrency cap, a per-item
// deadline and a hard total budget, so a scan ALWAYS answers. It is pure and
// dependency-free so the behaviour is unit-testable (see
// server/test/concurrency.test.mjs).
//
// Result shape is aligned with the input list:
//   { ok: true,  value }                       — worker resolved
//   { ok: false, error, timedOut }             — worker threw or hit perItemMs
//   { ok: false, error, skipped: true }        — total budget ran out first
// ============================================================================

/**
 * @param {Array} items
 * @param {(item:any, index:number)=>Promise<any>} worker
 * @param {{concurrency?:number, perItemMs?:number, totalMs?:number}} [opts]
 * @returns {Promise<{results:Array, summary:{ok:number,failed:number,timedOut:number,skipped:number}>}>}
 */
export async function runBounded(items, worker, opts = {}) {
  const list = Array.from(items || []);
  const { concurrency = 3, perItemMs = 0, totalMs = 0 } = opts;
  const results = new Array(list.length);
  const deadline = totalMs > 0 ? Date.now() + totalMs : Infinity;
  let next = 0;

  async function runOne(item, idx) {
    const budget = deadline - Date.now();
    if (budget <= 0) {
      results[idx] = { ok: false, skipped: true, error: "budget exhausted before this item started" };
      return;
    }
    const cap = perItemMs > 0 ? Math.min(perItemMs, budget) : budget;
    let timer = null;
    try {
      // A late rejection from the losing side of the race must never surface as
      // an unhandledRejection (F-02 exits the process on those).
      const attempt = Promise.resolve().then(() => worker(item, idx));
      attempt.catch(() => {});
      const raced = cap === Infinity ? attempt : Promise.race([
        attempt,
        new Promise((_, reject) => {
          timer = setTimeout(() => {
            const e = new Error(`timed out after ${cap}ms`);
            e.timedOut = true;
            reject(e);
          }, cap);
        }),
      ]);
      results[idx] = { ok: true, value: await raced };
    } catch (e) {
      results[idx] = {
        ok: false,
        error: e?.message || String(e),
        timedOut: e?.timedOut === true,
      };
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  const workers = [];
  const n = Math.max(1, Math.min(concurrency, list.length));
  for (let w = 0; w < n; w++) {
    workers.push((async () => {
      for (;;) {
        const idx = next++;
        if (idx >= list.length) return;
        await runOne(list[idx], idx);
      }
    })());
  }
  await Promise.all(workers);

  const count = (pred) => results.reduce((acc, r) => acc + (pred(r) ? 1 : 0), 0);
  return {
    results,
    summary: {
      ok: count((r) => r.ok),
      failed: count((r) => !r.ok && !r.skipped && !r.timedOut),
      timedOut: count((r) => !r.ok && r.timedOut),
      skipped: count((r) => !r.ok && r.skipped),
    },
  };
}
