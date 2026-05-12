// ============================================================================
// FRIT ENHANCED SYSTEMS — ACP + Crash GSRI + EMA/Structure Scanners + Pipeline
// ============================================================================
// ENHANCEMENTS APPLIED:
//   #1 Contradiction-Aware Evidence Weighting (dominance + source discount)
//   #2 Bayesian Confidence Fusion (weighted blend, no multiplicative crush)
//   #3 Structure-Based Scanner (CHoCH/BOS + BB squeeze, parallel to EMA)
//   #4 Instrument-Specific ACP Parameters (lambda/beta per symbol)
//   #5 Execution Quality Filter (directional ATR proximity gate)
//   #6 Adaptive Stability (recency-weighted memory, cold start 2/1)
//   🔧 Critical Lambda Fix (per-hour decay instead of per-minute)
// ============================================================================

class ACPEngine {
  constructor() {
    this.evidenceStreams = new Map();
    this.acpCache = new Map();
    // FIX #4 & LAMBDA: Base lambda changed to per-hour decay (0.08/hr ≈ 8.6h half-life)
    this.defaultParams = {
      beta: 0.01, gamma: 0.01,
      eta1: 0.6, eta2: 0.5, eta3: 0.5,
      lambda0: 0.12, lambda_min: 0.02,
      lambda: 0.08, // per-hour decay
    };
    // FIX #4: Instrument-specific overrides
    this.instrumentParams = {
      USDJPY: { lambda: 0.12, beta: 0.02 },
      GBPJPY: { lambda: 0.10, beta: 0.02 },
      EURJPY: { lambda: 0.10, beta: 0.02 },
      XAUUSD: { lambda: 0.05, beta: 0.005 },
      XAGUSD: { lambda: 0.06, beta: 0.005 },
      BTCUSD: { lambda: 0.09, beta: 0.015 },
      ETHUSD: { lambda: 0.09, beta: 0.015 },
    };
  }

  tagEvidenceFromCandles(candles) {
    const stream = [];
    for (let i = 1; i < candles.length; i++) {
      const prev = candles[i - 1];
      const curr = candles[i];
      const delta = curr.close - prev.close;
      const range = curr.high - curr.low;
      const body = Math.abs(curr.close - curr.open);
      let tag;
      if (body < 0.2 * range || Math.abs(delta) < 0.0001 * curr.close) tag = "ambiguous";
      else if (delta > 0 && curr.close > prev.close) tag = "support";
      else tag = "contradict";
      stream.push({ timestamp: curr.time || Date.now(), type: tag, delta, range, body, source: "candle" });
    }
    return stream;
  }

  tagEvidenceFromEngine(analysis) {
    const stream = [];
    const now = Date.now();
    if (!analysis || analysis.error) return stream;
    const { structure, volatility, patterns, auction, mtf } = analysis;

    if (structure) {
      if (structure.trend === "up" && structure.bos_bullish) stream.push({ timestamp: now, type: "support", source: "structure_bos_bull", weight: 1.5 });
      if (structure.trend === "down" && structure.bos_bearish) stream.push({ timestamp: now, type: "contradict", source: "structure_bos_bear", weight: 1.5 });
      if (structure.trend === "neutral") stream.push({ timestamp: now, type: "ambiguous", source: "structure_neutral", weight: 1.0 });
    }
    if (auction) {
      if (auction.bias === "bullish") stream.push({ timestamp: now, type: "support", source: "auction_bullish", weight: 1.2 });
      else if (auction.bias === "bearish") stream.push({ timestamp: now, type: "contradict", source: "auction_bearish", weight: 1.2 });
      else stream.push({ timestamp: now, type: "ambiguous", source: "auction_mixed", weight: 0.8 });
    }
    if (mtf) {
      if (mtf.trend === "up") stream.push({ timestamp: now, type: "support", source: "mtf_bullish", weight: 1.3 });
      else if (mtf.trend === "down") stream.push({ timestamp: now, type: "contradict", source: "mtf_bearish", weight: 1.3 });
    }
    if (volatility) {
      if (volatility.regime === "compressed") stream.push({ timestamp: now, type: "ambiguous", source: "vol_compressed", weight: 0.7 });
      else if (volatility.regime === "expanding") stream.push({ timestamp: now, type: "support", source: "vol_expanding", weight: 1.0 });
    }
    if (patterns?.length > 0) {
      for (const p of patterns) {
        if (p === "bullish_engulfing" || p === "pinbar_bullish") stream.push({ timestamp: now, type: "support", source: `pattern_${p}`, weight: 1.4 });
        else if (p === "bearish_engulfing" || p === "pinbar_bearish") stream.push({ timestamp: now, type: "contradict", source: `pattern_${p}`, weight: 1.4 });
        else if (p === "indecision") stream.push({ timestamp: now, type: "ambiguous", source: "pattern_indecision", weight: 0.5 });
      }
    }
    return stream;
  }

  appendEvidence(symbol, newEvidence) {
    const sym = String(symbol).toUpperCase();
    if (!this.evidenceStreams.has(sym)) this.evidenceStreams.set(sym, []);
    const stream = this.evidenceStreams.get(sym);
    for (const e of newEvidence) stream.push(e);
    const cutoff = Date.now() - 24 * 60 * 60 * 1000;
    this.evidenceStreams.set(sym, stream.filter(e => e.timestamp > cutoff).slice(-200));
  }

  // FIX #1 & #4 & LAMBDA: Contradiction discount + per-hour decay + instrument params
  computeACP(symbol, currentTime, overrides = {}) {
    const sym = String(symbol).toUpperCase();
    const evidenceStream = this.evidenceStreams.get(sym) || [];
    const instParams = this.instrumentParams[sym] || {};
    const params = { ...this.defaultParams, ...instParams, ...overrides };

    // Pass 1: Raw S/C for dominance ratio
    let rawS = 0, rawC = 0;
    for (const e of evidenceStream) {
      if (e.type === "support") rawS += (e.weight || 1);
      else if (e.type === "contradict") rawC += (e.weight || 1);
    }
    const dominance = rawS / (rawS + rawC + 1e-3);

    // Pass 2: Decay + context-aware contradiction discount
    let S = 0, C = 0, A = 0;
    const lamDecay = params.lambda;
    let discountedC = 0;

    for (const e of evidenceStream) {
      const deltaHours = (currentTime - e.timestamp) / 3600000; // LAMBDA FIX: per-hour
      if (deltaHours < 0) continue;
      const w = Math.exp(-lamDecay * deltaHours);
      let weight = e.weight || 1.0;

      if (e.type === "contradict") {
        const isCandleNoise = e.source === "candle";
        const strongTrend = dominance > 0.65 || dominance < 0.35;
        if (strongTrend && isCandleNoise) { weight *= 0.3; discountedC += weight; }
        else if (strongTrend) { weight *= 0.7; discountedC += weight; }
      }

      const weightedW = w * weight;
      if (e.type === "support") S += weightedW;
      else if (e.type === "contradict") C += weightedW;
      else A += weightedW;
    }

    const p = S / (S + C + params.beta);
    const c = C / (S + C + A + params.gamma);
    const s = 1 / (1 + Math.exp(-(params.eta1 * (overrides.verified_hits || 2) - params.eta2 * (overrides.surprises || 1) - params.eta3 * c)));
    const lamAdapt = params.lambda0 * (1 - s) + params.lambda_min;
    const confidence = p * (1 - c) * s;

    if (overrides.log_acp) {
      console.log(`[ACP ${sym}] dom=${dominance.toFixed(2)} rawC=${rawC.toFixed(1)} discC=${discountedC.toFixed(1)} p=${p.toFixed(3)} c=${c.toFixed(3)} s=${s.toFixed(3)} conf=${confidence.toFixed(3)}`);
    }

    const result = { p, c, s, lambda: lamAdapt, confidence, evidence_count: evidenceStream.length, support_total: S, contradict_total: C, ambiguity_total: A };
    this.acpCache.set(sym, result);
    return result;
  }

  bandWidth(forecastPrice, acpComponents, basisBp = 25) {
    const c = acpComponents.c;
    const s = acpComponents.s;
    const widthBp = basisBp * (1 + c) * (1 - 0.5 * s);
    const widthAbs = forecastPrice * (widthBp / 10000);
    const tight = [forecastPrice - widthAbs, forecastPrice + widthAbs];
    const moderateAbs = forecastPrice * (2 * widthBp / 10000);
    const moderate = [forecastPrice - moderateAbs, forecastPrice + moderateAbs];
    return { tight, moderate, width_bp: widthBp };
  }

  // FIX #6: Recency-weighted stability + cold start 2/1
  async run(symbol, candles, analysis, tradeMemory = []) {
    const sym = String(symbol).toUpperCase();
    const now = Date.now();
    const candleEvidence = this.tagEvidenceFromCandles(candles);
    const engineEvidence = this.tagEvidenceFromEngine(analysis);
    this.appendEvidence(sym, [...candleEvidence.slice(-20), ...engineEvidence]);

    let verifiedHits = 2; // FIX #6: Safer cold start
    let surprises = 1;
    if (tradeMemory?.length > 0) {
      const halfLifeMs = 7 * 24 * 60 * 60 * 1000;
      for (const t of tradeMemory) {
        const ageMs = now - t.timestamp;
        const weight = Math.exp(-Math.log(2) * ageMs / halfLifeMs);
        if (t.outcome === "win") verifiedHits += weight;
        else if (t.outcome === "loss") surprises += weight;
      }
    }

    const acp = this.computeACP(sym, now, { verified_hits: verifiedHits, surprises, log_acp: true });
    let bands = null;
    const price = analysis?.price || candles?.at(-1)?.close;
    if (price && acp) bands = this.bandWidth(price, acp);
    return { acp, bands, verified_hits: verifiedHits, surprises };
  }

  getCached(symbol) { return this.acpCache.get(String(symbol).toUpperCase()) || null; }
  getEvidenceStream(symbol) { return this.evidenceStreams.get(String(symbol).toUpperCase()) || []; }
}

class CrashGSRIEngine {
  constructor() {
    this.basket = { crypto: ["BTCUSD", "ETHUSD", "SOLUSD"], forex: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"], metals: ["XAUUSD"] };
    this.history = []; this.maxHistory = 200; this.lastResult = null; this.lastComputeTime = 0; this.computeIntervalMs = 5 * 60 * 1000;
  }
  computeCovariance(returnsArrays) {
    const N = returnsArrays.length; if (N === 0) return [];
    const T = returnsArrays[0].length;
    const means = returnsArrays.map(r => r.reduce((a, b) => a + b, 0) / T);
    const cov = [];
    for (let i = 0; i < N; i++) { cov[i] = []; for (let j = 0; j < N; j++) { let sum = 0; for (let t = 0; t < T; t++) sum += (returnsArrays[i][t] - means[i]) * (returnsArrays[j][t] - means[j]); cov[i][j] = sum / T; } }
    return cov;
  }
  eigenvaluesSymmetric(matrix) {
    const n = matrix.length; if (n === 0) return []; if (n === 1) return [matrix[0][0]];
    let A = matrix.map(row => [...row]);
    for (let sweep = 0; sweep < 100; sweep++) {
      let maxOff = 0, p = 0, q = 1;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (Math.abs(A[i][j]) > maxOff) { maxOff = Math.abs(A[i][j]); p = i; q = j; }
      if (maxOff < 1e-12) break;
      const app = A[p][p], aqq = A[q][q], apq = A[p][q];
      const theta = (aqq - app) / (2 * apq);
      const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1); const s = t * c;
      const newA = A.map(row => [...row]);
      for (let i = 0; i < n; i++) { if (i !== p && i !== q) { newA[i][p] = c * A[i][p] - s * A[i][q]; newA[i][q] = s * A[i][p] + c * A[i][q]; newA[p][i] = newA[i][p]; newA[q][i] = newA[i][q]; } }
      newA[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq; newA[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq; newA[p][q] = 0; newA[q][p] = 0; A = newA;
    }
    return A.map((row, i) => row[i]).sort((a, b) => b - a);
  }
  computeCrashMetrics(returnsArrays) {
    const N = returnsArrays.length; if (N < 3) return null;
    const stdReturns = returnsArrays.map(returns => { const mean = returns.reduce((a, b) => a + b, 0) / returns.length; const std = Math.sqrt(returns.reduce((s, r) => s + (r - mean) ** 2, 0) / returns.length) || 1e-8; return returns.map(r => (r - mean) / std); });
    const cov = this.computeCovariance(stdReturns);
    const eigenvalues = this.eigenvaluesSymmetric(cov);
    if (eigenvalues.length === 0) return null;
    const eigSum = eigenvalues.reduce((a, b) => a + Math.abs(b), 0) + 1e-12;
    const lambda1 = Math.abs(eigenvalues[0]);
    const concentration = lambda1 / eigSum;
    const p = eigenvalues.map(e => Math.abs(e) / eigSum);
    const clippedP = p.map(pi => Math.max(pi, 1e-12));
    const pSum = clippedP.reduce((a, b) => a + b, 0);
    const normalizedP = clippedP.map(pi => pi / pSum);
    const rawEntropy = -normalizedP.reduce((s, pi) => s + pi * Math.log(pi), 0);
    const normEntropy = N > 1 ? rawEntropy / Math.log(N) : 0;
    const K = lambda1 / N;
    let tau = 0;
    if (this.history.length > 0) { const prevK = this.history[this.history.length - 1].K; tau = (K - prevK) / (Math.abs(prevK) + 1e-8); }
    this.history.push({ K, concentration, norm_entropy: normEntropy, tau, timestamp: Date.now() });
    if (this.history.length > this.maxHistory) this.history.shift();
    let riskScore = 0;
    if (this.history.length > 10) {
      const recent = this.history.slice(-40);
      const tauMean = recent.reduce((s, h) => s + h.tau, 0) / recent.length;
      const tauStd = Math.sqrt(recent.reduce((s, h) => s + (h.tau - tauMean) ** 2, 0) / recent.length) || 1e-8;
      const tauZ = Math.max(0, (tau - tauMean) / tauStd);
      const tauNorm = Math.tanh(tauZ);
      const entNorm = 1 - normEntropy;
      const concNorm = Math.min(Math.max(concentration, 0), 1);
      riskScore = (tauNorm + entNorm + concNorm) / 3;
    }
    const phase = this.determineCrashPhase(concentration, normEntropy, tau, riskScore);
    const magnitude = phase === "trigger" || phase === "crash" ? Math.min(0.15, riskScore * 0.2) : 0;
    return { concentration, norm_entropy: normEntropy, tau, risk_score: riskScore, K, phase, magnitude, eigenvalue_count: eigenvalues.length, assets_analyzed: N, timestamp: Date.now() };
  }
  determineCrashPhase(concentration, normEntropy, tau, riskScore) {
    if (this.history.length < 5) return "insufficient_data";
    const recent = this.history.slice(-5);
    const concTrend = recent[recent.length - 1].concentration - recent[0].concentration;
    const entTrend = recent[recent.length - 1].norm_entropy - recent[0].norm_entropy;
    if (Math.abs(tau) > 0.3 && riskScore > 0.6) return "trigger";
    if (concTrend < -0.05 && entTrend > 0.05 && riskScore > 0.5) return "crash";
    if (concTrend > 0.03 && entTrend < -0.03 && Math.abs(tau) < 0.15) return "setup";
    if (riskScore < 0.35 && entTrend > 0) return "recovery";
    return "normal";
  }
  async compute(fetchCandlesFn) {
    const now = Date.now();
    if (now - this.lastComputeTime < this.computeIntervalMs && this.lastResult) return this.lastResult;
    const allSymbols = [...this.basket.crypto, ...this.basket.forex, ...this.basket.metals];
    const allCandles = {};
    await Promise.all(allSymbols.map(async sym => { try { const candles = await fetchCandlesFn(sym, "1h", 72); if (candles && candles.length >= 30) allCandles[sym] = candles; } catch (e) {} }));
    const availableSymbols = Object.keys(allCandles);
    if (availableSymbols.length < 3) return { phase: "insufficient_data", risk_score: 0.5, concentration: 0, norm_entropy: 1, tau: 0, magnitude: 0, assets_analyzed: availableSymbols.length, timestamp: now };
    const lengths = availableSymbols.map(s => allCandles[s].length);
    const minLen = Math.min(...lengths);
    const returnsArrays = availableSymbols.map(sym => { const candles = allCandles[sym]; const returns = []; for (let i = 1; i < minLen; i++) { const prev = candles[i - 1].close; const curr = candles[i].close; returns.push((curr - prev) / prev); } return returns; });
    const returnLen = Math.min(...returnsArrays.map(r => r.length));
    const alignedReturns = returnsArrays.map(r => r.slice(-returnLen));
    const metrics = this.computeCrashMetrics(alignedReturns);
    if (metrics) { metrics.assets = availableSymbols; this.lastResult = metrics; this.lastComputeTime = now; }
    return metrics || this.lastResult || { phase: "insufficient_data", risk_score: 0.5, timestamp: now };
  }
  getCrashSignals(crashMetrics, analysisResults = {}) {
    if (!crashMetrics || crashMetrics.phase === "insufficient_data") return { actionable: false, signals: [] };
    const signals = [];
    switch (crashMetrics.phase) {
      case "setup": signals.push({ type: "alert", message: "Crash setup detected", confidence: Math.round(crashMetrics.risk_score * 100), action: "PREPARE" }); break;
      case "trigger":
        for (const [sym, analysis] of Object.entries(analysisResults)) {
          if (!analysis || analysis.error) continue;
          if (analysis.direction === "BULLISH" || analysis.direction === "NEUTRAL") {
            signals.push({ type: "crash_short", symbol: sym, direction: "SELL", confidence: Math.round(crashMetrics.risk_score * 80), magnitude_estimate: `${(crashMetrics.magnitude * 100).toFixed(2)}%` });
          }
        }
        if (signals.length === 0) signals.push({ type: "crash_short_basket", direction: "SELL", confidence: Math.round(crashMetrics.risk_score * 70), magnitude_estimate: `${(crashMetrics.magnitude * 100).toFixed(2)}%` });
        break;
      case "crash": signals.push({ type: "crash_active", message: "Crash in progress — hold shorts", action: "HOLD_SHORT", confidence: Math.round(crashMetrics.risk_score * 60) }); break;
      case "recovery": signals.push({ type: "crash_recovery", message: "Crash recovery — exit shorts", action: "CLOSE_SHORTS", confidence: 65 }); break;
    }
    return { actionable: signals.length > 0, phase: crashMetrics.phase, risk_score: crashMetrics.risk_score, signals };
  }
}

class EMAScanner {
  constructor() {
    this.watchlist = { forex: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"], crypto: ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD"], metals: ["XAUUSD"] };
    this.lastCrossover = new Map(); this.lastTrigger = new Map(); this.cooldownMs = 30 * 60 * 1000;
    this.scanResults = new Map(); this.triggerHistory = []; this.running = false; this.scanIntervalId = null;
  }
  async detectCrossover(symbol, fetchCandlesFn, calcEMAFn, interval = "1h") {
    try {
      const candles = await fetchCandlesFn(symbol, interval, 50);
      if (!candles || candles.length < 30) return null;
      const closes = candles.map(c => c.close);
      const ema9 = calcEMAFn(closes, 9); const ema21 = calcEMAFn(closes, 21);
      if (ema9.length < 2 || ema21.length < 2) return null;
      const currentDiff = ema9[ema9.length - 1] - ema21[ema21.length - 1];
      const prevDiff = ema9[ema9.length - 2] - ema21[ema21.length - 2];
      let crossType = null;
      if (prevDiff <= 0 && currentDiff > 0) crossType = "bullish_cross";
      if (prevDiff >= 0 && currentDiff < 0) crossType = "bearish_cross";
      const trend = currentDiff > 0 ? "bullish" : "bearish";
      const separation = Math.abs(currentDiff) / (closes[closes.length - 1] || 1);
      return { symbol, cross: crossType, trend, separation, ema9: ema9[ema9.length - 1], ema21: ema21[ema21.length - 1], price: closes[closes.length - 1], interval };
    } catch (e) { return null; }
  }
  async scanAll(fetchCandlesFn, calcEMAFn, interval = "1h") {
    const results = {};
    const allSymbols = [...this.watchlist.forex, ...this.watchlist.crypto, ...this.watchlist.metals];
    const batchSize = 5;
    for (let i = 0; i < allSymbols.length; i += batchSize) {
      const batch = allSymbols.slice(i, i + batchSize);
      const batchResults = await Promise.all(batch.map(sym => this.detectCrossover(sym, fetchCandlesFn, calcEMAFn, interval)));
      batchResults.forEach((r, idx) => { if (r) results[batch[idx]] = r; });
      if (i + batchSize < allSymbols.length) await new Promise(r => setTimeout(r, 500));
    }
    const crossovers = {};
    for (const [sym, r] of Object.entries(results)) { if (r.cross) { crossovers[sym] = r; this.lastCrossover.set(sym, { type: r.cross, timestamp: Date.now() }); } }
    const forexBullish = Object.entries(results).filter(([s, r]) => this.watchlist.forex.includes(s) && r.trend === "bullish");
    const forexBearish = Object.entries(results).filter(([s, r]) => this.watchlist.forex.includes(s) && r.trend === "bearish");
    const crossBullish = Object.entries(crossovers).filter(([s, r]) => r.cross === "bullish_cross");
    const crossBearish = Object.entries(crossovers).filter(([s, r]) => r.cross === "bearish_cross");
    let trigger = null;
    if (crossBullish.filter(([s]) => this.watchlist.forex.includes(s)).length >= 2) trigger = { direction: "bullish", type: "forex_crossover_confirmation", pairs: crossBullish.map(([s]) => s), confidence: 75 };
    if (crossBearish.filter(([s]) => this.watchlist.forex.includes(s)).length >= 2) trigger = { direction: "bearish", type: "forex_crossover_confirmation", pairs: crossBearish.map(([s]) => s), confidence: 75 };
    const cryptoCrosses = [...crossBullish, ...crossBearish].filter(([s]) => this.watchlist.crypto.includes(s));
    if (cryptoCrosses.length >= 2 && !trigger) trigger = { direction: cryptoCrosses[0][1].cross === "bullish_cross" ? "bullish" : "bearish", type: "crypto_crossover_confirmation", pairs: cryptoCrosses.map(([s]) => s), confidence: 60 };
    let triggerScore = 0;
    if (Object.keys(crossovers).length > 0) triggerScore += 2;
    if (trigger) triggerScore += 2;
    if (forexBullish.length >= 4 || forexBearish.length >= 4) triggerScore += 1;
    const scanResult = { results, crossovers, trigger, trigger_score: triggerScore, forex_bullish_count: forexBullish.length, forex_bearish_count: forexBearish.length, timestamp: Date.now() };
    this.scanResults.set("latest", scanResult);
    if (trigger) { this.triggerHistory.push({ ...trigger, timestamp: Date.now() }); if (this.triggerHistory.length > 100) this.triggerHistory.shift(); }
    return scanResult;
  }
  isInCooldown(symbol) { const lastTime = this.lastTrigger.get(String(symbol).toUpperCase()); return lastTime ? Date.now() - lastTime < this.cooldownMs : false; }
  markTriggered(symbol) { this.lastTrigger.set(String(symbol).toUpperCase(), Date.now()); }
  getLatestScan() { return this.scanResults.get("latest") || null; }
  getTriggerHistory(limit = 20) { return this.triggerHistory.slice(-limit); }
}

// FIX #3: Structure-Based Scanner (Parallel Trigger Source)
class StructureScanner {
  constructor(deps) {
    this.findSwings = deps.findSwings;
    this.calcBB = deps.calcBB;
    this.lastState = new Map();
    this.triggerHistory = [];
  }
  detectShift(symbol, candles, currentBias) {
    if (!candles || candles.length < 30 || !this.findSwings) return null;
    const swings = this.findSwings(candles, 3);
    if (!swings.highs.length || !swings.lows.length) return null;
    const lastHigh = swings.highs[swings.highs.length - 1];
    const prevHigh = swings.highs[swings.highs.length - 2];
    const lastLow = swings.lows[swings.lows.length - 1];
    const prevLow = swings.lows[swings.lows.length - 2];
    const currentClose = candles[candles.length - 1].close;
    let shift = null;
    if (currentBias !== "bearish" && prevHigh && currentClose > lastHigh.price && lastHigh.price > prevHigh.price) shift = { type: "BOS", direction: "bullish", level: lastHigh.price };
    else if (currentBias !== "bullish" && prevLow && currentClose < lastLow.price && lastLow.price < prevLow.price) shift = { type: "BOS", direction: "bearish", level: lastLow.price };
    else if (currentBias === "bearish" && lastHigh && currentClose > lastHigh.price) shift = { type: "CHoCH", direction: "bullish", level: lastHigh.price };
    else if (currentBias === "bullish" && lastLow && currentClose < lastLow.price) shift = { type: "CHoCH", direction: "bearish", level: lastLow.price };
    return shift;
  }
  detectSqueeze(candles) {
    if (!candles || candles.length < 40 || !this.calcBB) return false;
    const closes = candles.map(c => c.close);
    const currentBB = this.calcBB(closes, 20, 2);
    if (!currentBB) return false;
    const currentWidth = (currentBB.upper - currentBB.lower) / currentBB.middle;
    let widthSum = 0, count = 0;
    for (let i = 20; i < closes.length - 20; i += 5) {
      const histBB = this.calcBB(closes.slice(0, i + 20), 20, 2);
      if (histBB) { widthSum += (histBB.upper - histBB.lower) / histBB.middle; count++; }
    }
    const avgWidth = count > 0 ? widthSum / count : currentWidth;
    return currentWidth < avgWidth * 0.6;
  }
  async scanAll(fetchCandlesFn, interval = "1h") {
    const symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "ETHUSD"];
    const triggers = [];
    for (const sym of symbols) {
      try {
        const candles = await fetchCandlesFn(sym, interval, 60);
        if (!candles || candles.length < 40) continue;
        const state = this.lastState.get(sym) || { bias: "neutral" };
        const shift = this.detectShift(sym, candles, state.bias);
        const squeeze = this.detectSqueeze(candles);
        if (shift) {
          state.bias = shift.direction;
          this.lastState.set(sym, state);
          const confidence = squeeze ? 75 : 60;
          triggers.push({ symbol: sym, direction: shift.direction, type: `structure_${shift.type}${squeeze ? "_squeeze" : ""}`, confidence, level: shift.level, timestamp: Date.now() });
          this.triggerHistory.push({ symbol: sym, ...shift, squeeze, timestamp: Date.now() });
          if (this.triggerHistory.length > 100) this.triggerHistory.shift();
        }
      } catch (e) { /* skip */ }
    }
    return triggers;
  }
}

class PaperTradeLogger {
  constructor() { this.trades = new Map(); this.maxPerSymbol = 50; }
  log(trade) {
    const sym = String(trade.symbol || "UNKNOWN").toUpperCase();
    if (!this.trades.has(sym)) this.trades.set(sym, []);
    const entries = this.trades.get(sym);
    const entry = { id: `${sym}_${Date.now()}`, symbol: sym, direction: trade.direction, entry: trade.entry, sl: trade.sl, tp: trade.tp, confidence: trade.confidence, lot_size: trade.lot_size, source: trade.source, outcome: "pending", opened_at: Date.now() };
    entries.push(entry);
    if (entries.length > this.maxPerSymbol) entries.shift();
    return entry;
  }
  resolve(tradeId, outcome, closePrice) {
    for (const [, entries] of this.trades.entries()) {
      const trade = entries.find(e => e.id === tradeId);
      if (trade) {
        trade.outcome = outcome; trade.closed_at = Date.now(); trade.close_price = closePrice;
        if (trade.entry && closePrice) { const dir = trade.direction === "BULLISH" || trade.direction === "BUY" ? 1 : -1; trade.pnl_pips = dir * (closePrice - trade.entry); }
        return trade;
      }
    }
    return null;
  }
  getTrades(symbol, limit = 20) { return (this.trades.get(String(symbol || "").toUpperCase()) || []).slice(-limit); }
  getStats(symbol) {
    const entries = this.trades.get(String(symbol || "").toUpperCase()) || [];
    const resolved = entries.filter(e => e.outcome !== "pending");
    return { total: entries.length, resolved: resolved.length, wins: resolved.filter(e => e.outcome === "win").length, losses: resolved.filter(e => e.outcome === "loss").length, breakeven: resolved.filter(e => e.outcome === "breakeven").length };
  }
  getPendingTrades() { const pending = []; for (const [, entries] of this.trades.entries()) for (const e of entries) if (e.outcome === "pending") pending.push(e); return pending; }
  autoResolve(currentPrices) {
    const resolved = [];
    for (const [, entries] of this.trades.entries()) {
      for (const trade of entries) {
        if (trade.outcome !== "pending") continue;
        const price = currentPrices[trade.symbol]; if (!price) continue;
        const isLong = trade.direction === "BULLISH" || trade.direction === "BUY";
        if (trade.tp && trade.sl) {
          if (isLong) { if (price >= trade.tp) { this.resolve(trade.id, "win", price); resolved.push({ ...trade, outcome: "win" }); } else if (price <= trade.sl) { this.resolve(trade.id, "loss", price); resolved.push({ ...trade, outcome: "loss" }); } }
          else { if (price <= trade.tp) { this.resolve(trade.id, "win", price); resolved.push({ ...trade, outcome: "win" }); } else if (price >= trade.sl) { this.resolve(trade.id, "loss", price); resolved.push({ ...trade, outcome: "loss" }); } }
        }
      }
    }
    return resolved;
  }
}

class EnhancedDecisionPipeline {
  constructor(deps) {
    this.acp = new ACPEngine();
    this.crashGSRI = new CrashGSRIEngine();
    this.scanner = new EMAScanner();
    this.structureScanner = new StructureScanner(deps);
    this.paperLogger = new PaperTradeLogger();
    this.fetchCandles = deps.fetchCandles; this.calcEMA = deps.calcEMA; this.analyzeSymbol = deps.analyzeSymbol;
    this.cacheGet = deps.cacheGet; this.cacheSet = deps.cacheSet; this.checkNewsFilter = deps.checkNewsFilter;
    this.getGsriSnapshot = deps.getGsriSnapshot; this.gsriLotScale = deps.gsriLotScale; this.calculateLotSize = deps.calculateLotSize;
    this.sendToMT5Bridge = deps.sendToMT5Bridge; this.addTradeMemory = deps.addTradeMemory; this.getTradeMemory = deps.getTradeMemory;
    this.pipelineHistory = []; this.maxHistory = 200;
  }

  // FIX #5: Execution Quality Filter
  checkEntryQuality(analysis, direction) {
    const { price, support, resistance, volatility, structure, auction } = analysis;
    const atr = volatility?.atr || price * 0.01;
    const isLong = direction === "BUY";
    const trend = structure?.trend || "neutral";
    const relevantLevel = isLong ? support : resistance;
    if (!relevantLevel || relevantLevel <= 0) return { pass: true, reason: "no_level" };
    const dist = isLong ? (price - relevantLevel) : (relevantLevel - price);
    const distATR = dist / atr;
    if (dist < 0) return { pass: false, reason: "broke_level", alternative: relevantLevel.toFixed(5) };
    const threshold = (trend === (isLong ? "up" : "down")) ? 2.0 : 1.5;
    if (distATR > threshold) {
      const idealEntry = isLong ? support + atr * 0.5 : resistance - atr * 0.5;
      return { pass: false, reason: "floating_in_space", dist_atr: distATR.toFixed(2), alternative: idealEntry.toFixed(5) };
    }
    if (auction?.poc) {
      const pocDist = Math.abs(price - auction.poc) / atr;
      if (pocDist < 0.8) return { pass: true, reason: "near_poc" };
    }
    return { pass: true, reason: "within_atr", dist_atr: distATR.toFixed(2) };
  }

  async run(symbol, interval = "1h", options = {}) {
    const sym = String(symbol).toUpperCase();
    const startTime = Date.now();
    if (this.scanner.isInCooldown(sym)) return { decision: "COOLDOWN", symbol: sym, reason: `Cooldown active`, cooldown_remaining_ms: this.scanner.cooldownMs - (Date.now() - (this.scanner.lastTrigger.get(sym) || Date.now())) };

    const crashMetrics = await this.crashGSRI.compute(this.fetchCandles);
    const crashSignals = this.crashGSRI.getCrashSignals(crashMetrics);
    if (crashSignals.actionable && crashMetrics.phase === "trigger") {
      const analysis = await this.analyzeSymbol(sym, interval);
      const candles = await this.fetchCandles(sym, interval, 120);
      const tradeMem = this.getTradeMemory ? this.getTradeMemory(sym) : [];
      const acpResult = candles ? await this.acp.run(sym, candles, analysis, tradeMem) : { acp: { confidence: 0.5, p: 0.5, c: 0, s: 0.5 }, bands: null };
      const isCrashCandidate = analysis.direction === "BULLISH" || analysis.direction === "NEUTRAL";
      if (isCrashCandidate) {
        const price = analysis.price; const atr = analysis.volatility?.atr || price * 0.01;
        const entry = price; const sl = price + atr * 1.5; const tp = price - atr * 2.5;
        const lotSize = this.calculateLotSize ? this.calculateLotSize({ symbol: sym, balance: 1000, riskPercent: 1, entry, stopLoss: sl }) : 0.01;
        const decision = { decision: "SELL", source: "crash_gsri", symbol: sym, confidence: Math.round(crashMetrics.risk_score * 80), entry, sl: sl.toFixed(5), tp: tp.toFixed(5), lot_size: lotSize, crash_phase: crashMetrics.phase, risk_score: crashMetrics.risk_score };
        this.paperLogger.log({ symbol: sym, direction: "SELL", entry, sl, tp, confidence: decision.confidence, lot_size: lotSize, source: "crash_gsri", acp_confidence: acpResult.acp.confidence, crash_metrics: crashMetrics });
        this.pipelineHistory.push(decision); if (this.pipelineHistory.length > this.maxHistory) this.pipelineHistory.shift();
        this.scanner.markTriggered(sym); return decision;
      }
    }

    const newsCheck = await this.checkNewsFilter(sym);
    if (newsCheck.blocked) return { decision: "NO_TRADE", symbol: sym, reason: `News blackout: ${newsCheck.reason}`, source: "news_filter" };

    const analysis = await this.analyzeSymbol(sym, interval);
    if (!analysis || analysis.error || analysis.direction === "NEUTRAL") return { decision: "NO_TRADE", symbol: sym, reason: analysis?.error || "No clear edge", analysis };
    if ((crashMetrics.phase === "crash" || crashMetrics.phase === "trigger") && analysis.direction === "BULLISH") return { decision: "NO_TRADE", symbol: sym, reason: `Long blocked during crash ${crashMetrics.phase}` };

    // FIX #5: Entry Quality Gate
    const direction = analysis.direction === "BULLISH" ? "BUY" : "SELL";
    const quality = this.checkEntryQuality(analysis, direction);
    if (!quality.pass) return { decision: "NO_TRADE", symbol: sym, reason: `Entry quality: ${quality.reason}`, quality, alternative_entry: quality.alternative, analysis };

    const candles = await this.fetchCandles(sym, interval, 120);
    const tradeMem = this.getTradeMemory ? this.getTradeMemory(sym) : [];
    const acpResult = candles ? await this.acp.run(sym, candles, analysis, tradeMem) : { acp: { confidence: 0.5, p: 0.5, c: 0, s: 0.5, lambda: 0.08 }, bands: null };

    // FIX #2: Bayesian Confidence Fusion (Weighted Blend)
    const engineConfidence = analysis.confidence || 50;
    const acpConfidence = acpResult.acp.confidence;
    const alpha = 0.7;
    const acpScaled = 50 + (acpConfidence - 0.5) * 80;
    let finalConfidence = Math.round(alpha * engineConfidence + (1 - alpha) * acpScaled);
    finalConfidence = Math.max(5, Math.min(95, finalConfidence));

    let adjustedTP = analysis.trade_plan?.tp1;
    let adjustedSL = analysis.trade_plan?.invalidation;
    let bandNote = "";
    if (acpResult.bands && acpResult.acp.c > 0.4) {
      const price = analysis.price;
      const bandWidth = acpResult.bands.moderate;
      bandNote = `ACP bands widened (c=${acpResult.acp.c.toFixed(3)})`;
      adjustedTP = analysis.direction === "BULLISH" ? bandWidth[1] : bandWidth[0];
    }

    let gsriMode = "normal"; let gsriScale = 1.0;
    try {
      const gsriSnap = await this.getGsriSnapshot();
      const gsriScore = parseFloat(gsriSnap?.Risk_Score ?? 0);
      const gsriAlert = parseInt(gsriSnap?.Alert ?? 0);
      if (gsriAlert === 1 || gsriScore > 0.8) gsriMode = "blocked";
      else if (gsriScore > 0.6) { gsriMode = "defend"; finalConfidence = Math.round(finalConfidence * 0.6); gsriScale = this.gsriLotScale(gsriScore); }
      else if (gsriScore < 0.4) { gsriMode = "normal"; finalConfidence = Math.min(95, Math.round(finalConfidence * 1.1)); }
    } catch (e) { gsriMode = "normal"; }

    if (gsriMode === "blocked" && crashMetrics.phase !== "trigger") return { decision: "NO_TRADE", symbol: sym, reason: "GSRI high risk", gsri_mode: gsriMode };
    if (finalConfidence < 30) return { decision: "NO_TRADE", symbol: sym, reason: `Confidence too low (${finalConfidence}%)`, engine_confidence: engineConfidence, acp_confidence: Math.round(acpConfidence * 100) };

    const price = analysis.price;
    const entry = parseFloat(analysis.trade_plan?.entry_zone?.split("-")[0]) || price;
    const sl = parseFloat(adjustedSL) || 0;
    const tp = parseFloat(adjustedTP) || 0;
    const lotSize = this.calculateLotSize ? this.calculateLotSize({ symbol: sym, balance: options.balance || 1000, riskPercent: options.riskPercent || 1, entry, stopLoss: sl }) * gsriScale : 0.01;

    const decision = { decision: direction, source: "enhanced_pipeline", symbol: sym, confidence: finalConfidence, entry, sl: sl.toFixed(5), tp: tp.toFixed(5), lot_size: Math.round(lotSize * 100) / 100, band_note: bandNote, gsri_mode: gsriMode };

    this.paperLogger.log({ symbol: sym, direction, entry, sl, tp, confidence: finalConfidence, lot_size: Math.round(lotSize * 100) / 100, source: "enhanced_pipeline", acp_confidence: acpConfidence, crash_metrics: crashMetrics });
    if (this.addTradeMemory) this.addTradeMemory(sym, { direction, pattern: analysis.patterns?.join(", ") || "none", outcome: "pending", note: `Pipeline: conf=${finalConfidence} acp_p=${acpResult.acp.p.toFixed(3)}` });
    this.scanner.markTriggered(sym);
    this.pipelineHistory.push(decision); if (this.pipelineHistory.length > this.maxHistory) this.pipelineHistory.shift();
    return decision;
  }

  async scan(interval = "1h") { return this.scanner.scanAll(this.fetchCandles, this.calcEMA, interval); }
  getHistory(limit = 20) { return this.pipelineHistory.slice(-limit); }
}

class ScannerLoop {
  constructor(pipeline) {
    this.pipeline = pipeline; this.running = false; this.intervalId = null; this.scanIntervalMs = 2 * 60 * 1000;
    this.lastScanTime = 0; this.scanCount = 0; this.triggersGenerated = 0; this.resolveIntervalId = null; this.resolveIntervalMs = 5 * 60 * 1000;
    this.budget = null; this.fcsClient = null;
  }
  start(intervalMs) {
    if (this.running) return { status: "already_running" };
    this.scanIntervalMs = intervalMs || this.scanIntervalMs; this.running = true;
    this.intervalId = setInterval(async () => { await this.runScan(); }, this.scanIntervalMs);
    this.resolveIntervalId = setInterval(async () => { await this.autoResolveTrades(); }, this.resolveIntervalMs);
    this.runScan();
    return { status: "started", interval_ms: this.scanIntervalMs };
  }
  stop() {
    this.running = false; if (this.intervalId) clearInterval(this.intervalId); if (this.resolveIntervalId) clearInterval(this.resolveIntervalId);
    this.intervalId = null; this.resolveIntervalId = null; return { status: "stopped" };
  }
  async runScan() {
    try {
      this.lastScanTime = Date.now(); this.scanCount++;

      // LAYER 1: GSRI (only every 4 hours to save FCS credits)
      const now = Date.now();
      const gsriDue = !this.pipeline.crashGSRI.lastComputeTime || (now - this.pipeline.crashGSRI.lastComputeTime > 4 * 60 * 60 * 1000);
      let gsriMetrics = this.pipeline.crashGSRI.lastResult || { phase: "normal", risk_score: 0.3 };
      if (gsriDue && this.fcsClient && this.budget?.canCall("fcs")) {
        const gsriBasket = ["EURUSD","GBPUSD","USDJPY","XAUUSD","BTCUSD","ETHUSD","US500","DXY"];
        try {
          const riskData = await this.fcsClient.getSystemicRiskData(gsriBasket, 20);
          if (riskData && riskData.returnsMatrix.length >= 3) {
            gsriMetrics = this.pipeline.crashGSRI.computeCrashMetrics(riskData.returnsMatrix);
            if (gsriMetrics) { gsriMetrics.assets = riskData.symbols; this.pipeline.crashGSRI.lastResult = gsriMetrics; this.pipeline.crashGSRI.lastComputeTime = now; }
          }
          this.budget.recordCall("fcs");
        } catch (e) { console.warn(`[Scanner] FCS GSRI fetch failed, falling back: ${e.message}`); gsriMetrics = await this.pipeline.crashGSRI.compute(this.pipeline.fetchCandles); }
      } else if (gsriDue) { gsriMetrics = await this.pipeline.crashGSRI.compute(this.pipeline.fetchCandles); }

      if (gsriMetrics.phase === "trigger" || gsriMetrics.risk_score > 0.75) console.log(`[Scanner] GSRI ${gsriMetrics.phase} (score: ${gsriMetrics.risk_score.toFixed(2)}) — blocking new longs`);

      // LAYER 2: Rotated Broad Scan (3 pairs per cycle)
      const scanBatch = this.budget?.getRotatedBatch(3) || ["EURUSD","BTCUSD","XAUUSD"];
      const potentialPairs = [];
      for (const sym of scanBatch) {
        if (/^(BTC|ETH|SOL|BNB|XRP|DOGE|ADA)/.test(sym)) {
          try {
            const candles = await this.pipeline.fetchCandles(sym, "1h", 24);
            if (candles && candles.length >= 12) {
              const last = candles[candles.length - 1]; const prev = candles[candles.length - 2];
              const change24h = ((last.close - prev.close) / prev.close) * 100;
              if (Math.abs(change24h) > 0.4) potentialPairs.push(sym);
            }
          } catch (e) { /* skip */ }
        } else {
          const cached = this.pipeline.cacheGet?.(`candles:${sym}:1h:24`);
          if (cached && cached.length >= 12) {
            const last = cached[cached.length - 1]; const prev = cached[cached.length - 2];
            const change24h = ((last.close - prev.close) / prev.close) * 100;
            if (Math.abs(change24h) > 0.3) potentialPairs.push(sym);
          }
        }
      }

      // LAYER 3: Deep Dive (only if budget allows & pre-filter passed)
      for (const sym of potentialPairs) {
        if (this.pipeline.scanner.isInCooldown(sym)) continue;
        if (!this.budget?.canCall("twelve")) { console.warn(`[Scanner] Twelve Data budget exhausted — skipping deep dive for ${sym}`); continue; }
        try {
          const decision = await this.pipeline.run(sym, "1h");
          this.budget?.recordCall("twelve");
          if (decision.decision !== "NO_TRADE" && decision.decision !== "COOLDOWN") {
            this.triggersGenerated++;
            console.log(`[Scanner] SIGNAL: ${decision.decision} ${sym} conf=${decision.confidence}%`);
          }
        } catch (e) { console.error(`[Scanner] Pipeline error ${sym}:`, e.message); }
      }
    } catch (e) { console.error("[Scanner] Loop error:", e.message); }
  }
  async autoResolveTrades() {
    try {
      const pending = this.pipeline.paperLogger.getPendingTrades(); if (pending.length === 0) return;
      const symbols = [...new Set(pending.map(t => t.symbol))]; const prices = {};
      await Promise.all(symbols.map(async sym => { try { const spot = await this.pipeline.fetchCandles(sym, "1min", 1); if (spot && spot.length > 0) prices[sym] = spot[0].close; } catch (e) {} }));
      const resolved = this.pipeline.paperLogger.autoResolve(prices);
      if (resolved.length > 0) console.log(`[ScannerLoop] Auto-resolved ${resolved.length} paper trades`);
    } catch (e) { console.error("[ScannerLoop] Auto-resolve error:", e.message); }
  }
  getStatus() {
    return { running: this.running, scan_interval_ms: this.scanIntervalMs, last_scan_time: this.lastScanTime, scan_count: this.scanCount, triggers_generated: this.triggersGenerated, pending_paper_trades: this.pipeline.paperLogger.getPendingTrades().length };
  }
}

export class FritSystems {
  constructor(deps) {
    this.pipeline = new EnhancedDecisionPipeline(deps);
    this.scannerLoop = new ScannerLoop(this.pipeline);
    this.budget = deps.budgetManager || null;
    this.scannerLoop.budget = this.budget;
    this.scannerLoop.fcsClient = deps.fcsClient || null;
    this.acp = this.pipeline.acp; this.crashGSRI = this.pipeline.crashGSRI;
    this.scanner = this.pipeline.scanner; this.paperLogger = this.pipeline.paperLogger;
  }
  async analyze(symbol, interval = "1h", options = {}) { return this.pipeline.run(symbol, interval, options); }
  async scan(interval = "1h") { return this.pipeline.scan(interval); }
  startScanner(intervalMs) { return this.scannerLoop.start(intervalMs); }
  stopScanner() { return this.scannerLoop.stop(); }
  getStatus() {
    return {
      scanner_loop: this.scannerLoop.getStatus(),
      crash_gsri: { last_phase: this.crashGSRI.lastResult?.phase || "unknown", last_risk_score: this.crashGSRI.lastResult?.risk_score || 0, last_compute: this.crashGSRI.lastComputeTime, history_size: this.crashGSRI.history.length },
      acp: { cached_symbols: this.acp.acpCache.size, evidence_streams: this.acp.evidenceStreams.size },
      paper_trades: { total_symbols: this.paperLogger.trades.size, pending: this.paperLogger.getPendingTrades().length },
      pipeline_history: this.pipeline.pipelineHistory.length,
    };
  }
}
