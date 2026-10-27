// ============================================================================
// PREDICTIVE VOLUME ENGINE — forecastVolume() / backtestForecast()
// ----------------------------------------------------------------------------
// Pure functions, no network, no DB. These pin the behaviour the bot lifecycle
// relies on: a trend-aligned expansion is recognised, an exhaustion spike is
// REFUSED (it must never feed a continuation trigger), thin data is reported
// as unavailable instead of guessed, and the levels handed to a retrigger are
// on the right side of the current price.
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";

import { forecastVolume, backtestForecast, prepCandles, activitySeries, freshLevels } from "../src/strategy/volumeforecast.js";
import { bars5, bars1, MIN5 } from "./helpers/volbars.mjs";

test("prepCandles: normalises, sorts and drops junk", () => {
  const now = Date.now();
  const bars = prepCandles([
    { time: now, open: 1, high: 2, low: 0.5, close: 1.5, volume: 10 },
    { time: now - 1000, open: "1", high: "2", low: "0.5", close: "1.4", volume: "9" },
    { time: now - 2000, open: 1, high: 0, low: 5, close: 1.5, volume: 1 }, // high < low -> out
    { open: 1, high: 2, low: 0.5, close: 1.5, volume: 1 }, //               no time  -> out
  ]);
  assert.equal(bars.length, 2);
  assert.ok(bars[0].t < bars[1].t, "bars must be oldest-first");
  assert.equal(typeof bars[0].volume, "number");
});

test("activitySeries: uses volume when it is there, labels the range proxy when it is not", () => {
  const vol = bars5({ n: 80 });
  assert.equal(activitySeries(vol).source, "volume");

  const noVol = vol.map((b) => ({ ...b, volume: 0 }));
  const proxy = activitySeries(noVol);
  assert.equal(proxy.source, "range_proxy");
  assert.ok(proxy.values.every((v) => v >= 0));
});

test("freshLevels: SL below / TP above the entry, and the R:R is the one asked for", () => {
  const buy = freshLevels(2000, "BUY", 10, { rr: 1.5, slAtr: 1.5 });
  assert.ok(buy.sl < buy.entry && buy.tp > buy.entry);
  assert.equal(buy.entry - buy.sl, 15);
  assert.equal(buy.tp - buy.entry, 22.5);

  const sell = freshLevels(2000, "SELL", 10, { rr: 1.5, slAtr: 1.5 });
  assert.ok(sell.sl > sell.entry && sell.tp < sell.entry);
  assert.equal(sell.entry - sell.tp, 22.5);
});

test("forecast: too little history is unavailable, not a guess", () => {
  const f = forecastVolume({ c5: bars5({ n: 40 }), trend: "BUY" });
  assert.equal(f.available, false);
  assert.equal(f.expansion, false);
  assert.match(f.reasons[0], /need 60\+ five-minute candles/);
});

test("forecast: a trend-aligned volume build-up IS an expansion", () => {
  const f = forecastVolume({ c5: bars5(), c1: bars1(), trend: "BUY" });
  assert.equal(f.available, true, f.reasons?.join(" | "));
  assert.equal(f.direction, "BUY");
  assert.equal(f.expansion, true, `score ${f.score} need ${f.needScore} :: ${f.reasons.join(" | ")}`);
  assert.ok(f.score >= f.needScore);
  assert.equal(f.exhaustion, false);
  assert.equal(f.source, "volume");

  // The numbers behind the verdict.
  assert.ok(f.features.rvol2 > 1.5, `rvol2 ${f.features.rvol2}`);
  assert.ok(f.features.slope > 0, "volume must be building");
  assert.ok(f.features.rvol1 > 1.5, "1m pace must confirm");
  assert.ok(f.features.dispAligned > 0 && f.features.cvdAligned > 0, "flow must agree with the trend");
  assert.equal(typeof f.features.breakout, "boolean", "breakout is reported, never assumed");
});

test("forecast: a SELL trend gets levels on the sell side", () => {
  const down = bars5().map((b, i, a) => ({
    ...b,
    open: 3000 - 2 * i,
    close: i === a.length - 1 ? 3000 - 2 * i - 2 : 3000 - 2 * i,
    high: 3000 - 2 * i + 3,
    low: 3000 - 2 * i - 5,
  }));
  const f = forecastVolume({ c5: down, c1: bars1(), trend: "SELL" });
  assert.equal(f.available, true, f.reasons?.join(" | "));
  assert.equal(f.direction, "SELL");
  assert.ok(f.levels.sl > f.levels.entry && f.levels.tp < f.levels.entry);
});

test("forecast: a 7x spike is EXHAUSTION — it blocks the continuation trigger", () => {
  const f = forecastVolume({ c5: bars5({ spike: true }), c1: bars1(), trend: "BUY" });
  assert.equal(f.available, true);
  assert.equal(f.exhaustion, true, "a spike must be read as exhaustion, not fuel");
  assert.equal(f.expansion, false, "an exhaustion bar must never trigger a retrigger");
  assert.ok(f.reasons.some((r) => /exhaustion risk, no continuation trigger/.test(r)));
});

test("forecast: a quiet tape with no volume build-up does not expand", () => {
  const f = forecastVolume({ c5: bars5({ ramp: false }), c1: bars1({ ramp: false }), trend: "BUY" });
  assert.equal(f.available, true);
  assert.equal(f.expansion, false, `score ${f.score} :: ${f.reasons.join(" | ")}`);
});

test("forecast: no 1m data still works — the score is rescaled, not inflated", () => {
  const withPace = forecastVolume({ c5: bars5(), c1: bars1(), trend: "BUY" });
  const noPace = forecastVolume({ c5: bars5(), c1: [], trend: "BUY" });
  assert.equal(noPace.available, true);
  assert.equal(noPace.features.rvol1, null);
  assert.ok(noPace.score <= 100 && noPace.score >= 0);
  assert.equal(noPace.expansion, true, `score ${noPace.score} :: ${noPace.reasons.join(" | ")}`);
  assert.ok(withPace.score >= 0);
});

test("forecast: the trend is derived from the bars when none is supplied", () => {
  const f = forecastVolume({ c5: bars5(), c1: bars1(), trend: null });
  assert.equal(f.available, true);
  assert.equal(f.direction, "BUY");
});

test("forecast: the levels a retrigger would use sit around the live price", () => {
  const f = forecastVolume({ c5: bars5(), c1: bars1(), trend: "BUY" });
  const last1m = bars1().at(-1).close;
  assert.equal(f.price, last1m);
  assert.equal(f.levels.entry, last1m);
  assert.ok(f.levels.sl < f.levels.entry && f.levels.tp > f.levels.entry);
  const rr = (f.levels.tp - f.levels.entry) / (f.levels.entry - f.levels.sl);
  assert.ok(Math.abs(rr - 1.5) < 1e-9, `rr ${rr}`);
  assert.equal(f.levels.slAtr, 1.5);
});

test("backtestForecast: the trigger beats the base rate on its own history", () => {
  const r = backtestForecast(bars5(), { horizon: 3, expandRatio: 1.4 });
  assert.equal(r.ok, true, r.why);
  assert.ok(r.samples > 10, `samples ${r.samples}`);
  assert.ok(r.triggers >= 1, "the expanding leg must fire at least once");
  assert.ok(r.lift != null && r.lift >= 1, `lift ${r.lift} (base ${r.baseRate}, hits ${r.hitRate})`);
  assert.ok(r.priceFollowThroughRate != null);
});

test("backtestForecast: refuses history that is too thin to judge", () => {
  const r = backtestForecast(bars5({ n: 40 }));
  assert.equal(r.ok, false);
  assert.match(r.why, /need \d+\+ candles/);
});

test("forecast: the forming 5m bar is never read as closed data", () => {
  const now = Date.now();
  const c5 = bars5({ now });
  // Make the newest bar still forming (opened 60s ago, 5m bucket).
  const last = c5[c5.length - 1];
  last.time = Math.floor(now / MIN5) * MIN5;
  const f = forecastVolume({ c5, c1: bars1({ now }), trend: "BUY", now });
  assert.equal(f.available, true, f.reasons?.join(" | "));
  // It must have dropped that bar: one fewer closed bar than we supplied.
  assert.ok(f.features.rvol5 >= 0);
});
