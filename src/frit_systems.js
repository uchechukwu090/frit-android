// BEGINNING OF FILE CONTENT OMITTED FOR BREVITY

// FIX 1: Double API Call in EnhancedDecisionPipeline.run
// Reusing analysis data to avoid unnecessary API calls
let analysis = null;
if (crashSignals.actionable && crashMetrics.phase === "trigger") {
  analysis = await this.analyzeSymbol(sym, interval);
  const isCrashCandidate = analysis.direction === "BULLISH" || analysis.direction === "NEUTRAL";
  if (isCrashCandidate) {
    // ...crash trade logic...
  }
}

const newsCheck = await this.checkNewsFilter(sym);
if (!analysis) {
  analysis = await this.analyzeSymbol(sym, interval); // Reuse fetched data
}
if (!analysis || analysis.error || analysis.direction === "NEUTRAL") {
  return { decision: "NO_TRADE", ... };
}

// FIX 2: Symmetric ACP Band Adjustment for SL/TP Matching
if (acpResult.bands && acpResult.acp.c > 0.4) {
  const price = analysis.price;
  const bandWidth = acpResult.bands.moderate;
  bandNote = `ACP bands widened (c=${acpResult.acp.c.toFixed(3)})`;
  
  if (analysis.direction === "BULLISH") {
    adjustedTP = bandWidth[1];
    adjustedSL = bandWidth[0]; // Widen SL symmetrically with TP
  } else {
    adjustedTP = bandWidth[0];
    adjustedSL = bandWidth[1]; // Widen SL symmetrically with TP
  }
}

// FIX 3: Crash Trade Balance and GSRI Scaling
const lotSize = this.calculateLotSize ? this.calculateLotSize({
  symbol: sym,
  balance: options.balance || 1000, // Fix: utilize passed options
  riskPercent: options.riskPercent || 1,
  entry,
  stopLoss: sl,
}) * (gsriScale || 1.0) : 0.01; // Fix: apply GSRI scaling consistently

// FIX 4: Potential String/Math Bug in checkEntryQuality
const relevantLevel = isLong ? parseFloat(support) : parseFloat(resistance);
if (!relevantLevel || relevantLevel <= 0) {
  return { pass: true, reason: "no_level" };
}

// FIX 5: Math Consistency in computeCrashMetrics
const stdReturns = returnsArrays.map(returns => {
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const std = Math.sqrt(returns.reduce((s, r) => s + (r - mean) ** 2, 0) / (returns.length - 1)) || 1e-8; // Sample std for consistency
  return returns.map(r => (r - mean) / std);
});

// FIX 6: Reduce Candle Fetch Or Use All Data in ACPEngine.run
this.appendEvidence(sym, [...candleEvidence, ...engineEvidence]); // Option A: Use all candles
// OR
const fetchedCandles = await fetchCandles(sym, interval, 24); // Option B: Fetch what is needed

// END OF FILE CONTENT OMITTED FOR BREVITY