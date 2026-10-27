import test from "node:test";
import assert from "node:assert/strict";
import { generatePerformanceRecommendations, MARKET_UNIVERSE } from "../src/strategy/recommendations.js";
import { getMarketNewsCalendar, normalizeImpact, generateNewsAnalysis } from "../src/strategy/news_calendar.js";

// ============================================================================
// BACKTEST / TEST SUITE: Performance Recommendations & News Calendar Analysis
// ============================================================================

test("recommendations: market universe covers all required markets", () => {
  const markets = new Set(MARKET_UNIVERSE.map((m) => m.market));
  assert.ok(markets.has("FX"), "Must cover Forex (FX)");
  assert.ok(markets.has("CRYPTO"), "Must cover Crypto");
  assert.ok(markets.has("STOCKS"), "Must cover Stocks");
  assert.ok(markets.has("INDICES"), "Must cover Indices");
  assert.ok(markets.has("SYNTHETICS"), "Must cover Synthetic Rise & Falls");
});

test("recommendations: not random picking — deterministic performance ranking", () => {
  const run1 = generatePerformanceRecommendations();
  const run2 = generatePerformanceRecommendations();

  assert.deepEqual(run1, run2, "Recommendations must be deterministic, never randomly generated");

  // Verify sorted by performance score descending
  for (let i = 1; i < run1.length; i++) {
    assert.ok(
      run1[i - 1].performanceScore >= run1[i].performanceScore,
      `Items must be sorted by performance score descending (${run1[i - 1].performanceScore} >= ${run1[i].performanceScore})`
    );
  }
});

test("recommendations: synthetics generate proper Rise/Fall actions and generator rules", () => {
  const synths = generatePerformanceRecommendations({ market: "SYNTHETICS" });
  assert.ok(synths.length >= 4, "Should have multiple synthetic recommendations");

  for (const s of synths) {
    assert.equal(s.market, "SYNTHETICS");
    assert.ok(
      s.action === "RISE / CALL" || s.action === "FALL / PUT",
      `Synthetic action must be RISE / CALL or FALL / PUT, got ${s.action}`
    );
    assert.ok(s.isSynthetic === true, "Must flag isSynthetic");
    assert.ok(s.performanceScore >= 60 && s.performanceScore <= 100, "Valid score range");
    assert.ok(s.rr >= 1.5, "Risk-reward ratio must be favorable");
  }

  const r75 = synths.find((s) => s.symbol === "R_75");
  assert.ok(r75, "R_75 Volatility 75 index recommendation must be present");
  assert.equal(r75.action, "RISE / CALL", "Oversold random walk generator triggers RISE mean-reversion");

  const crash = synths.find((s) => s.symbol === "CRASH1000");
  assert.ok(crash, "Crash 1000 must be present");
  assert.equal(crash.action, "FALL / PUT", "Overbought drift triggers FALL / PUT contract");
});

test("recommendations: market filtering correctly isolates asset classes", () => {
  const fxOnly = generatePerformanceRecommendations({ market: "FX" });
  assert.ok(fxOnly.length > 0);
  assert.ok(fxOnly.every((item) => item.market === "FX"), "All items must be FX");

  const cryptoOnly = generatePerformanceRecommendations({ market: "CRYPTO" });
  assert.ok(cryptoOnly.length > 0);
  assert.ok(cryptoOnly.every((item) => item.market === "CRYPTO"), "All items must be Crypto");

  const stocksOnly = generatePerformanceRecommendations({ market: "STOCKS" });
  assert.ok(stocksOnly.length > 0);
  assert.ok(stocksOnly.every((item) => item.market === "STOCKS"), "All items must be Stocks");

  const indicesOnly = generatePerformanceRecommendations({ market: "INDICES" });
  assert.ok(indicesOnly.length > 0);
  assert.ok(indicesOnly.every((item) => item.market === "INDICES"), "All items must be Indices");
});

test("recommendations: distinguishes between active trades and long-term investments", () => {
  const investOnly = generatePerformanceRecommendations({ type: "INVEST" });
  assert.ok(investOnly.length > 0);
  assert.ok(investOnly.every((item) => item.type === "INVEST"));

  const tradeOnly = generatePerformanceRecommendations({ type: "TRADE" });
  assert.ok(tradeOnly.length > 0);
  assert.ok(tradeOnly.every((item) => item.type === "TRADE"));
});

test("recommendations: metrics contain required performance indicators", () => {
  const all = generatePerformanceRecommendations();
  for (const item of all) {
    assert.ok(item.symbol, "Symbol must be defined");
    assert.ok(typeof item.performanceScore === "number", "Score must be number");
    assert.ok(typeof item.winRateEst === "number", "Win rate estimate must be number");
    assert.ok(typeof item.rsi === "number", "RSI must be number");
    assert.ok(typeof item.return24h === "number", "24h return must be number");
    assert.ok(item.trend, "Trend description must exist");
    assert.ok(item.reasoning.length > 20, "Detailed reasoning must exist");
    assert.ok(item.grade, "Grade must exist");
    assert.ok(item.entry > 0, "Valid entry");
    assert.ok(item.sl > 0, "Valid SL");
    assert.ok(item.tp1 > 0, "Valid TP1");
  }
});

// --- News Calendar & Analysis Tests ---

test("news: normalizeImpact handles standard labels and color cues", () => {
  assert.equal(normalizeImpact("High"), "HIGH");
  assert.equal(normalizeImpact("high impact"), "HIGH");
  assert.equal(normalizeImpact("red"), "HIGH");
  assert.equal(normalizeImpact("Medium"), "MEDIUM");
  assert.equal(normalizeImpact("orange"), "MEDIUM");
  assert.equal(normalizeImpact("amber"), "MEDIUM");
  assert.equal(normalizeImpact("Low"), "LOW");
  assert.equal(normalizeImpact("yellow"), "LOW");
  assert.equal(normalizeImpact("Holiday"), "HOLIDAY");
});

test("news: getMarketNewsCalendar partitions into upcoming and previous events", async () => {
  const cal = await getMarketNewsCalendar();
  assert.ok(cal.upcoming.length > 0, "Must have upcoming events");
  assert.ok(cal.previous.length > 0, "Must have previous events");

  const now = Date.now();
  // Upcoming events are in the future
  for (const u of cal.upcoming) {
    assert.equal(u.isUpcoming, true);
    assert.ok(u.time > now, "Upcoming event time must be in future");
  }

  // Previous events are in the past
  for (const p of cal.previous) {
    assert.equal(p.isUpcoming, false);
    assert.ok(p.time <= now, "Previous event time must be in past");
  }
});

test("news: previous events contain analytical breakdown towards the news", async () => {
  const cal = await getMarketNewsCalendar();
  const prev = cal.previous;

  for (const p of prev) {
    assert.ok(p.actual != null, "Previous news must have actual value");
    assert.ok(p.outcome != null, "Must have outcome (BEAT/MISSED/IN_LINE)");
    assert.ok(["BEAT", "MISSED", "IN_LINE"].includes(p.outcome), `Outcome ${p.outcome} must be valid`);
    assert.ok(p.variance != null, "Must have variance");
    assert.ok(p.bias != null, "Must have market bias");
    assert.ok(p.marketAnalysis != null && p.marketAnalysis.length > 30, "Must have comprehensive market analysis text");
    assert.ok(Array.isArray(p.affectedPairs) && p.affectedPairs.length > 0, "Must list affected pairs");
    assert.ok(p.volatilityRating != null, "Must have volatility rating");
  }
});

test("news: generateNewsAnalysis accurately calculates beat/miss delta and bias", () => {
  // Test Beat
  const beatEvent = {
    title: "US Non-Farm Payrolls",
    currency: "USD",
    impact: "High",
    actual: "250K",
    forecast: "150K",
    previous: "140K",
  };
  const beatAnalysis = generateNewsAnalysis(beatEvent);
  assert.equal(beatAnalysis.outcome, "BEAT");
  assert.ok(beatAnalysis.bias.includes("HAWKISH / USD BULLISH") || beatAnalysis.bias.includes("BULLISH"));

  // Test Miss
  const missEvent = {
    title: "UK CPI",
    currency: "GBP",
    impact: "High",
    actual: "1.7%",
    forecast: "2.1%",
    previous: "2.2%",
  };
  const missAnalysis = generateNewsAnalysis(missEvent);
  assert.equal(missAnalysis.outcome, "MISSED");
  assert.ok(missAnalysis.bias.includes("DOVISH / GBP BEARISH") || missAnalysis.bias.includes("BEARISH"));
});

test("news: impact filtering properly isolates high impact events", async () => {
  const highOnly = await getMarketNewsCalendar({ impact: "HIGH" });
  assert.ok(highOnly.upcoming.every((e) => e.impact === "HIGH"));
  assert.ok(highOnly.previous.every((e) => e.impact === "HIGH"));
});
