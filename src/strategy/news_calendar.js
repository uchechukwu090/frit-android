// ============================================================================
// FRIT FOREX NEWS CALENDAR & ANALYSIS SYSTEM
// Provides categorized upcoming and previous economic news with detailed
// analytical breakdown towards the market reaction, bias, and affected pairs.
// ============================================================================

export const SEED_ECONOMIC_EVENTS = [
  // --- PREVIOUS / HISTORICAL EVENTS (with actuals & analytical impact breakdown) ---
  {
    id: "hist_usd_nfp_01",
    title: "Non-Farm Employment Change (NFP)",
    country: "USD",
    currency: "USD",
    impact: "High",
    date: new Date(Date.now() - 36 * 3600 * 1000).toISOString(), // 1.5 days ago
    forecast: "147K",
    previous: "159K",
    actual: "254K",
    outcome: "BEAT",
    variance: "+107K above consensus",
    bias: "HAWKISH / USD BULLISH",
    volatilityRating: "Very High (85 pips)",
    marketAnalysis: "Blockbuster jobs surge (+254K vs 147K exp) completely dissolved expectations of another 50 bps jumbo Fed rate cut. Benchmark 10-year US Treasury yields shot up 14 bps to 4.02%, propelling the US Dollar Index (DXY) to 7-week highs and forcing sharp intraday liquidations in EUR/USD and Gold.",
    affectedPairs: ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]
  },
  {
    id: "hist_usd_cpi_02",
    title: "Consumer Price Index (CPI) m/m",
    country: "USD",
    currency: "USD",
    impact: "High",
    date: new Date(Date.now() - 4 * 24 * 3600 * 1000).toISOString(), // 4 days ago
    forecast: "0.1%",
    previous: "0.2%",
    actual: "0.2%",
    outcome: "BEAT",
    variance: "+0.1% higher than expected",
    bias: "HAWKISH / STICKY INFLATION",
    volatilityRating: "High (55 pips)",
    marketAnalysis: "Headline CPI printed sticky at 0.2% MoM against 0.1% consensus, driven by auto insurance and shelter costs. Re-enforced Fed caution regarding the last mile of inflation reduction. EUR/USD dipped 40 pips post-release before stabilizing at support.",
    affectedPairs: ["EUR/USD", "USD/CHF", "USD/CAD", "SPX500"]
  },
  {
    id: "hist_eur_ecb_03",
    title: "ECB Main Refinancing Rate Decision",
    country: "EUR",
    currency: "EUR",
    impact: "High",
    date: new Date(Date.now() - 6 * 24 * 3600 * 1000).toISOString(),
    forecast: "3.40%",
    previous: "3.65%",
    actual: "3.40%",
    outcome: "IN_LINE",
    variance: "Matched 25 bps cut forecast",
    bias: "DOVISH / EUR BEARISH",
    volatilityRating: "Moderate (42 pips)",
    marketAnalysis: "The European Central Bank delivered a widely anticipated 25 bps rate reduction to 3.40%. President Lagarde highlighted sluggish Eurozone PMI composite data and noted downside risks to economic growth, cementing expectations of a back-to-back rate cut cycle.",
    affectedPairs: ["EUR/USD", "EUR/GBP", "EUR/JPY"]
  },
  {
    id: "hist_gbp_cpi_04",
    title: "UK Consumer Price Index (CPI) y/y",
    country: "GBP",
    currency: "GBP",
    impact: "High",
    date: new Date(Date.now() - 8 * 24 * 3600 * 1000).toISOString(),
    forecast: "1.9%",
    previous: "2.2%",
    actual: "1.7%",
    outcome: "MISSED",
    variance: "-0.2% cooler than expected",
    bias: "DOVISH / GBP BEARISH",
    volatilityRating: "High (65 pips)",
    marketAnalysis: "UK inflation tumbled below the BoE's 2.0% target for the first time in over three years (1.7% vs 1.9% expected). Triggered an immediate 60-pip slide in GBP/USD as gilts rallied and markets fully priced an accelerated rate-cutting pace by the Bank of England.",
    affectedPairs: ["GBP/USD", "EUR/GBP", "GBP/JPY"]
  },
  {
    id: "hist_jpy_boj_05",
    title: "BoJ Policy Rate & Statement",
    country: "JPY",
    currency: "JPY",
    impact: "High",
    date: new Date(Date.now() - 10 * 24 * 3600 * 1000).toISOString(),
    forecast: "0.25%",
    previous: "0.25%",
    actual: "0.25%",
    outcome: "IN_LINE",
    variance: "Maintained steady at 0.25%",
    bias: "NEUTRAL / HAWKISH LEAN",
    volatilityRating: "High (70 pips)",
    marketAnalysis: "Bank of Japan kept benchmark rates on hold at 0.25% but Governor Ueda signaled willingness to hike further if underlying inflation and wage renegotiations persist as projected. Yen consolidated with choppy 2-way price action.",
    affectedPairs: ["USD/JPY", "EUR/JPY", "GBPJPY"]
  },
  {
    id: "hist_usd_retail_06",
    title: "Core Retail Sales m/m",
    country: "USD",
    currency: "USD",
    impact: "Medium",
    date: new Date(Date.now() - 5 * 24 * 3600 * 1000).toISOString(),
    forecast: "0.3%",
    previous: "0.2%",
    actual: "0.5%",
    outcome: "BEAT",
    variance: "+0.2% consumer resilience",
    bias: "BULLISH / USD",
    volatilityRating: "Moderate (30 pips)",
    marketAnalysis: "Retail consumer spending continued to defy recession fears with a robust 0.5% MoM print. Confirms GDP growth momentum in the current quarter.",
    affectedPairs: ["EUR/USD", "USD/JPY"]
  },

  // --- UPCOMING EVENTS (Future economic calendar) ---
  {
    id: "up_usd_cpi_next",
    title: "Core CPI m/m",
    country: "USD",
    currency: "USD",
    impact: "High",
    date: new Date(Date.now() + 14 * 3600 * 1000).toISOString(), // In 14 hours
    forecast: "0.3%",
    previous: "0.3%",
    actual: null,
    affectedPairs: ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]
  },
  {
    id: "up_usd_unemployment",
    title: "Unemployment Claims",
    country: "USD",
    currency: "USD",
    impact: "Medium",
    date: new Date(Date.now() + 26 * 3600 * 1000).toISOString(), // In 26 hours
    forecast: "218K",
    previous: "221K",
    actual: null,
    affectedPairs: ["EUR/USD", "USD/JPY"]
  },
  {
    id: "up_eur_flash_pmi",
    title: "Flash Manufacturing PMI",
    country: "EUR",
    currency: "EUR",
    impact: "High",
    date: new Date(Date.now() + 42 * 3600 * 1000).toISOString(),
    forecast: "45.8",
    previous: "45.0",
    actual: null,
    affectedPairs: ["EUR/USD", "EUR/GBP", "GER40"]
  },
  {
    id: "up_gbp_pmi",
    title: "Flash Services PMI",
    country: "GBP",
    currency: "GBP",
    impact: "High",
    date: new Date(Date.now() + 48 * 3600 * 1000).toISOString(),
    forecast: "52.4",
    previous: "52.4",
    actual: null,
    affectedPairs: ["GBP/USD", "EUR/GBP", "UK100"]
  },
  {
    id: "up_usd_gdp_advance",
    title: "Advance GDP q/q",
    country: "USD",
    currency: "USD",
    impact: "High",
    date: new Date(Date.now() + 72 * 3600 * 1000).toISOString(),
    forecast: "3.0%",
    previous: "3.0%",
    actual: null,
    affectedPairs: ["EUR/USD", "SPX500", "USD/JPY"]
  },
  {
    id: "up_usd_core_pce",
    title: "Core PCE Price Index m/m (Fed Preferred Gauge)",
    country: "USD",
    currency: "USD",
    impact: "High",
    date: new Date(Date.now() + 96 * 3600 * 1000).toISOString(),
    forecast: "0.2%",
    previous: "0.1%",
    actual: null,
    affectedPairs: ["EUR/USD", "USD/JPY", "XAU/USD", "BTCUSD"]
  },
  {
    id: "up_aud_cpi_trim",
    title: "Trimmed Mean CPI q/q",
    country: "AUD",
    currency: "AUD",
    impact: "High",
    date: new Date(Date.now() + 110 * 3600 * 1000).toISOString(),
    forecast: "0.8%",
    previous: "1.0%",
    actual: null,
    affectedPairs: ["AUD/USD", "AUD/JPY"]
  },
  {
    id: "up_cad_boc_rate",
    title: "BoC Overnight Rate Decision",
    country: "CAD",
    currency: "CAD",
    impact: "High",
    date: new Date(Date.now() + 120 * 3600 * 1000).toISOString(),
    forecast: "3.75%",
    previous: "4.25%",
    actual: null,
    affectedPairs: ["USD/CAD", "EUR/CAD"]
  },
  {
    id: "up_usd_ism_mfg",
    title: "ISM Manufacturing PMI",
    country: "USD",
    currency: "USD",
    impact: "Medium",
    date: new Date(Date.now() + 144 * 3600 * 1000).toISOString(),
    forecast: "47.6",
    previous: "47.2",
    actual: null,
    affectedPairs: ["EUR/USD", "USD/JPY"]
  },
  {
    id: "up_usd_prelim_uom",
    title: "Prelim UoM Consumer Sentiment",
    country: "USD",
    currency: "USD",
    impact: "Low",
    date: new Date(Date.now() + 160 * 3600 * 1000).toISOString(),
    forecast: "69.0",
    previous: "68.9",
    actual: null,
    affectedPairs: ["USD/JPY"]
  }
];

/**
 * Normalizes impact level into standardized format
 */
export function normalizeImpact(impactStr) {
  const imp = String(impactStr || "").toLowerCase();
  if (imp.includes("high") || imp === "red") return "HIGH";
  if (imp.includes("med") || imp === "orange" || imp === "amber") return "MEDIUM";
  if (imp.includes("low") || imp === "yellow" || imp === "green") return "LOW";
  if (imp.includes("holiday")) return "HOLIDAY";
  return "MEDIUM";
}

/**
 * Generates an analytical breakdown for a historical economic news release
 */
export function generateNewsAnalysis(event) {
  const actual = String(event.actual || "").trim();
  const forecast = String(event.forecast || "").trim();
  const previous = String(event.previous || "").trim();
  const curr = event.currency || "USD";

  if (!actual) {
    return {
      outcome: "PENDING",
      variance: "Pending release",
      bias: "NEUTRAL",
      volatilityRating: "Scheduled",
      marketAnalysis: `Upcoming high-priority release for ${curr}. Expected volatility around release time.`,
      affectedPairs: event.affectedPairs || [`${curr}/USD`, `EUR/${curr}`]
    };
  }

  // Parse numbers if possible
  const actNum = parseFloat(actual.replace(/[^0-9.-]/g, ""));
  const fcNum = parseFloat(forecast.replace(/[^0-9.-]/g, ""));

  let outcome = "IN_LINE";
  let variance = "In line with consensus";
  let bias = "NEUTRAL";

  if (!isNaN(actNum) && !isNaN(fcNum)) {
    const diff = actNum - fcNum;
    if (Math.abs(diff) > 0.0001) {
      if (diff > 0) {
        outcome = "BEAT";
        variance = `+${diff.toFixed(2)} above consensus`;
        bias = `HAWKISH / ${curr} BULLISH`;
      } else {
        outcome = "MISSED";
        variance = `${diff.toFixed(2)} below consensus`;
        bias = `DOVISH / ${curr} BEARISH`;
      }
    }
  }

  const vol = normalizeImpact(event.impact) === "HIGH" ? "High (45-80 pips)" : "Moderate (20-35 pips)";
  const pairs = event.affectedPairs || (curr === "USD" ? ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"] : [`${curr}/USD`, `EUR/${curr}`]);

  return {
    outcome: event.outcome || outcome,
    variance: event.variance || variance,
    bias: event.bias || bias,
    volatilityRating: event.volatilityRating || vol,
    marketAnalysis: event.marketAnalysis || `${event.title} released at ${actual} vs ${forecast} forecast. Created immediate market repricing across ${pairs.join(", ")}.`,
    affectedPairs: pairs
  };
}

/**
 * Fetches and partitions economic events into upcoming and previous with
 * detailed analytical impact breakdowns.
 */
export async function getMarketNewsCalendar(options = {}) {
  const { impact = "ALL", mode = "ALL", currency = "ALL", limit = 40 } = options;
  const now = Date.now();

  let events = [...SEED_ECONOMIC_EVENTS];

  // Map each event to standardized structure
  const processed = events.map((e) => {
    const timeMs = new Date(e.date).getTime();
    const isUpcoming = timeMs > now;
    const impactLevel = normalizeImpact(e.impact);
    const analysis = !isUpcoming ? generateNewsAnalysis(e) : null;

    return {
      id: e.id || `event_${timeMs}_${e.currency}`,
      title: e.title,
      currency: e.currency,
      date: e.date,
      time: timeMs,
      impact: impactLevel,
      forecast: e.forecast || "-",
      previous: e.previous || "-",
      actual: e.actual || null,
      isUpcoming,
      outcome: analysis?.outcome || null,
      variance: analysis?.variance || null,
      bias: analysis?.bias || null,
      volatilityRating: analysis?.volatilityRating || null,
      marketAnalysis: analysis?.marketAnalysis || null,
      affectedPairs: e.affectedPairs || (analysis?.affectedPairs || ["EUR/USD", "USD/JPY"])
    };
  });

  // Filter by impact if requested
  let filtered = processed;
  if (impact && impact.toUpperCase() !== "ALL") {
    filtered = filtered.filter((e) => e.impact === impact.toUpperCase());
  }

  // Filter by currency if requested
  if (currency && currency.toUpperCase() !== "ALL") {
    filtered = filtered.filter((e) => e.currency === currency.toUpperCase());
  }

  // Partition into upcoming and previous
  const upcoming = filtered
    .filter((e) => e.isUpcoming)
    .sort((a, b) => a.time - b.time) // Soonest first
    .slice(0, limit);

  const previous = filtered
    .filter((e) => !e.isUpcoming)
    .sort((a, b) => b.time - a.time) // Most recent first
    .slice(0, limit);

  return {
    upcoming,
    previous,
    total: upcoming.length + previous.length,
    generatedAt: now
  };
}
