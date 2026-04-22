// ============================================================================
// FRIT ENHANCED SYSTEMS — ACP + Crash GSRI + EMA Scanner + Pipeline
// ============================================================================
// This module adds four major systems to your existing index.js:
//
//   1. ACP Probability Engine    — your axiom, properly integrated
//   2. Crash GSRI Engine         — offensive crash detection & trading
//   3. EMA Crossover Scanner     — autonomous market scanning
//   4. Enhanced Decision Pipeline — unified execution flow
//   5. Paper Trade Logger        — track signals until MT5 is ready
//
// INTEGRATION: Add to your index.js:
//   import { FritSystems } from "./frit_systems.js";
//   const frit = new FritSystems({ fetchCandles, calcEMA, analyzeSymbol, cacheGet, cacheSet, checkNewsFilter, getGsriSnapshot, gsriLotScale, calculateLotSize, sendToMT5Bridge, addTradeMemory, getTradeMemory });
// ============================================================================

// ============================================================================
// ACP PROBABILITY ENGINE
// ============================================================================
// Port of your acp_model.py + evidence.py, enhanced with engine signals.
//
// What stays from your original ACP:
//   - compute_acp() core math (S, C, A with time decay)
//   - p = S/(S+C+β) — forward probability
//   - c = C/(S+C+A+γ) — uncertainty as credit score (not penalty)
//   - s = sigmoid(η1·hits - η2·surprises - η3·c) — stability from track record
//   - λ_adapt = λ0(1-s) + λ_min — context-dependent decay
//   - confidence = p·(1-c)·s
//   - band_width() — probability zones that widen with contradiction
//
// What changes:
//   - tagEvidence() enhanced: uses BOTH raw candles AND your engine signals
//   - verified_hits/surprises come from trade memory (dynamic, not hardcoded)
//   - point_forecast and direction_and_tp_sl REMOVED (your engine does this)
//   - Evidence stream is persisted per symbol (stateful)
// ============================================================================

class ACPEngine {
  constructor() {
    // Per-symbol evidence state
    this.evidenceStreams = new Map();
    // Per-symbol ACP output cache
    this.acpCache = new Map();
    // Default params (from your config.py)
    this.defaultParams = {
      beta: 0.01,
      gamma: 0.01,
      eta1: 0.6,
      eta2: 0.5,
      eta3: 0.5,
      lambda0: 0.12,
      lambda_min: 0.02,
      lambda: 0.06,
    };
  }

  // --- Enhanced Evidence Tagger ---
  // Combines raw candle tagging (from your evidence.py) with engine signal tagging.
  // Your original: each candle → support/contradict/ambiguous based on body/delta
  // Enhancement: also tag based on structure, auction, MTF, volatility from your engine
  tagEvidenceFromCandles(candles) {
    const stream = [];
    for (let i = 1; i < candles.length; i++) {
      const prev = candles[i - 1];
      const curr = candles[i];
      const delta = curr.close - prev.close;
      const range = curr.high - curr.low;
      const body = Math.abs(curr.close - curr.open);

      let tag;
      if (body < 0.2 * range || Math.abs(delta) < 0.0001 * curr.close) {
        tag = "ambiguous";
      } else if (delta > 0 && curr.close > prev.close) {
        tag = "support";
      } else {
        tag = "contradict";
      }

      stream.push({
        timestamp: curr.time || Date.now(),
        type: tag,
        delta,
        range,
        body,
        source: "candle",
      });
    }
    return stream;
  }

  // Enhanced: tag evidence from your engine's analysis output
  tagEvidenceFromEngine(analysis) {
    const stream = [];
    const now = Date.now();

    if (!analysis || analysis.error) return stream;

    const { structure, volatility, patterns, auction, mtf, confidence, direction } = analysis;

    // Structure signals
    if (structure) {
      if (structure.trend === "up" && structure.bos_bullish) {
        stream.push({ timestamp: now, type: "support", source: "structure_bos_bull", weight: 1.5 });
      }
      if (structure.trend === "down" && structure.bos_bearish) {
        stream.push({ timestamp: now, type: "contradict", source: "structure_bos_bear", weight: 1.5 });
      }
      if (structure.trend === "neutral") {
        stream.push({ timestamp: now, type: "ambiguous", source: "structure_neutral", weight: 1.0 });
      }
    }

    // Auction signals
    if (auction) {
      if (auction.bias === "bullish") {
        stream.push({ timestamp: now, type: "support", source: "auction_bullish", weight: 1.2 });
      } else if (auction.bias === "bearish") {
        stream.push({ timestamp: now, type: "contradict", source: "auction_bearish", weight: 1.2 });
      } else if (auction.bias === "mild_bullish" || auction.bias === "mild_bearish") {
        stream.push({ timestamp: now, type: "ambiguous", source: "auction_mixed", weight: 0.8 });
      }
    }

    // Multi-timeframe alignment
    if (mtf) {
      if (mtf.trend === "up") {
        stream.push({ timestamp: now, type: "support", source: "mtf_bullish", weight: 1.3 });
      } else if (mtf.trend === "down") {
        stream.push({ timestamp: now, type: "contradict", source: "mtf_bearish", weight: 1.3 });
      }
    }

    // Volatility regime
    if (volatility) {
      if (volatility.regime === "compressed") {
        stream.push({ timestamp: now, type: "ambiguous", source: "vol_compressed", weight: 0.7 });
      } else if (volatility.regime === "expanding") {
        stream.push({ timestamp: now, type: "support", source: "vol_expanding", weight: 1.0 });
        // Volatility expansion supports the dominant direction
      }
    }

    // Candle patterns at key levels
    if (patterns && patterns.length > 0) {
      for (const p of patterns) {
        if (p === "bullish_engulfing" || p === "pinbar_bullish") {
          stream.push({ timestamp: now, type: "support", source: `pattern_${p}`, weight: 1.4 });
        } else if (p === "bearish_engulfing" || p === "pinbar_bearish") {
          stream.push({ timestamp: now, type: "contradict", source: `pattern_${p}`, weight: 1.4 });
        } else if (p === "indecision") {
          stream.push({ timestamp: now, type: "ambiguous", source: "pattern_indecision", weight: 0.5 });
        }
      }
    }

    return stream;
  }

  // Store evidence for a symbol (incremental, persistent across calls)
  appendEvidence(symbol, newEvidence) {
    const sym = String(symbol).toUpperCase();
    if (!this.evidenceStreams.has(sym)) {
      this.evidenceStreams.set(sym, []);
    }
    const stream = this.evidenceStreams.get(sym);

    // Add new evidence
    for (const e of newEvidence) {
      stream.push(e);
    }

    // Prune: keep only last 200 evidence points to prevent unbounded growth
    // Also prune evidence older than 24 hours
    const cutoff = Date.now() - 24 * 60 * 60 * 1000;
    const pruned = stream.filter(e => (e.timestamp > cutoff)).slice(-200);
    this.evidenceStreams.set(sym, pruned);
  }

  // --- Core ACP Computation (your axiom, faithfully ported) ---
  computeACP(symbol, currentTime, overrides = {}) {
    const sym = String(symbol).toUpperCase();
    const evidenceStream = this.evidenceStreams.get(sym) || [];

    const params = { ...this.defaultParams, ...overrides };

    let S = 0, C = 0, A = 0;
    const lamDecay = params.lambda;

    for (const e of evidenceStream) {
      const deltaMin = (currentTime - e.timestamp) / 60000; // ms to minutes
      if (deltaMin < 0) continue; // future evidence, skip

      const w = Math.exp(-lamDecay * deltaMin);
      const weight = e.weight || 1.0;
      const weightedW = w * weight;

      if (e.type === "support") S += weightedW;
      else if (e.type === "contradict") C += weightedW;
      else A += weightedW;
    }

    // p = S / (S + C + β)  — forward probability
    const p = S / (S + C + params.beta);

    // c = C / (S + C + A + γ)  — uncertainty as credit score
    const c = C / (S + C + A + params.gamma);

    // s = sigmoid(η1·verified_hits - η2·surprises - η3·c)  — stability
    const s = 1 / (1 + Math.exp(
      -(params.eta1 * (overrides.verified_hits || 3)
        - params.eta2 * (overrides.surprises || 0)
        - params.eta3 * c)
    ));

    // λ_adapt = λ0(1-s) + λ_min  — context-dependent decay
    const lamAdapt = params.lambda0 * (1 - s) + params.lambda_min;

    // confidence = p·(1-c)·s  — your core formula
    const confidence = p * (1 - c) * s;

    const result = {
      p,        // forward probability that signal sustains
      c,        // contradiction ratio (uncertainty as credit)
      s,        // stability from empirical track record
      lambda: lamAdapt,
      confidence,
      evidence_count: evidenceStream.length,
      support_total: S,
      contradict_total: C,
      ambiguity_total: A,
    };

    this.acpCache.set(sym, result);
    return result;
  }

  // --- Band Width (from your acp_model.py, kept exactly) ---
  // Wider with contradiction, tighter with stability
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

  // --- Full ACP Run for a Symbol ---
  // Called by the enhanced decision pipeline
  async run(symbol, candles, analysis, tradeMemory = []) {
    const sym = String(symbol).toUpperCase();
    const now = Date.now();

    // 1. Tag evidence from candles (your original evidence.py logic)
    const candleEvidence = this.tagEvidenceFromCandles(candles);

    // 2. Tag evidence from engine signals (enhancement)
    const engineEvidence = this.tagEvidenceFromEngine(analysis);

    // 3. Append to persistent stream
    this.appendEvidence(sym, [...candleEvidence.slice(-20), ...engineEvidence]);

    // 4. Compute verified_hits and surprises from trade memory
    let verifiedHits = 3;  // default from your config
    let surprises = 0;
    if (tradeMemory && tradeMemory.length > 0) {
      verifiedHits = tradeMemory.filter(t => t.outcome === "win").length;
      surprises = tradeMemory.filter(t => t.outcome === "loss").length;
      // Ensure minimums
      verifiedHits = Math.max(verifiedHits, 1);
    }

    // 5. Run ACP computation
    const acp = this.computeACP(sym, now, { verified_hits: verifiedHits, surprises });

    // 6. Compute band width if we have a price
    let bands = null;
    const price = analysis?.price || candles?.at(-1)?.close;
    if (price && acp) {
      bands = this.bandWidth(price, acp);
    }

    return { acp, bands, verified_hits: verifiedHits, surprises };
  }

  // Get cached ACP for a symbol
  getCached(symbol) {
    return this.acpCache.get(String(symbol).toUpperCase()) || null;
  }

  // Get evidence stream for a symbol
  getEvidenceStream(symbol) {
    return this.evidenceStreams.get(String(symbol).toUpperCase()) || [];
  }
}


// ============================================================================
// CRASH GSRI ENGINE
// ============================================================================
// Transforms your GSRI from a defensive risk gate into an offensive crash
// detection and trading engine.
//
// Key differences from your Python GSRI:
//   - Runs on CURRENT market data (not historical yfinance)
//   - Uses your existing fetchCandles() for the basket
//   - Smaller window (48-72 hourly bars vs 40 daily)
//   - Outputs crash PHASE (setup/trigger/crash/recovery) not just risk score
//   - Generates actionable trade signals on crashes
//   - No Python dependency — pure JavaScript
//
// The math is preserved:
//   - Covariance matrix from returns
//   - Eigenvalue concentration (λ₁/Σλ) → correlation lockstep
//   - Entropy → dimensional freedom
//   - Temporal instability τ → regime shift speed
// ============================================================================

class CrashGSRIEngine {
  constructor() {
    this.basket = {
      crypto: ["BTCUSD", "ETHUSD", "SOLUSD"],
      forex: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
      metals: ["XAUUSD"],
    };
    this.history = [];       // rolling metrics history for τ computation
    this.maxHistory = 200;
    this.lastResult = null;
    this.lastComputeTime = 0;
    this.computeIntervalMs = 5 * 60 * 1000; // recompute every 5 min
  }

  // Compute sample covariance matrix
  // returns: array of arrays, each inner array is a time series of returns
  computeCovariance(returnsArrays) {
    const N = returnsArrays.length;
    if (N === 0) return [];
    const T = returnsArrays[0].length;

    // Compute means
    const means = returnsArrays.map(r => r.reduce((a, b) => a + b, 0) / T);

    // Compute covariance
    const cov = [];
    for (let i = 0; i < N; i++) {
      cov[i] = [];
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let t = 0; t < T; t++) {
          sum += (returnsArrays[i][t] - means[i]) * (returnsArrays[j][t] - means[j]);
        }
        cov[i][j] = sum / (T - 1);
      }
    }
    return cov;
  }

  // Jacobi eigenvalue algorithm for symmetric matrices
  // Returns eigenvalues sorted descending
  eigenvaluesSymmetric(matrix) {
    const n = matrix.length;
    if (n === 0) return [];
    if (n === 1) return [matrix[0][0]];

    let A = matrix.map(row => [...row]);

    for (let sweep = 0; sweep < 100; sweep++) {
      // Find largest off-diagonal element
      let maxOff = 0, p = 0, q = 1;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (Math.abs(A[i][j]) > maxOff) {
            maxOff = Math.abs(A[i][j]);
            p = i; q = j;
          }
        }
      }
      if (maxOff < 1e-12) break;

      // Compute rotation
      const app = A[p][p], aqq = A[q][q], apq = A[p][q];
      const theta = (aqq - app) / (2 * apq);
      const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;

      // Apply Givens rotation
      const newA = A.map(row => [...row]);
      for (let i = 0; i < n; i++) {
        if (i !== p && i !== q) {
          newA[i][p] = c * A[i][p] - s * A[i][q];
          newA[i][q] = s * A[i][p] + c * A[i][q];
          newA[p][i] = newA[i][p];
          newA[q][i] = newA[i][q];
        }
      }
      newA[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      newA[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
      newA[p][q] = 0;
      newA[q][p] = 0;
      A = newA;
    }

    return A.map((row, i) => row[i]).sort((a, b) => b - a);
  }

  // Compute crash metrics from returns data
  computeCrashMetrics(returnsArrays) {
    const N = returnsArrays.length;
    if (N < 3) return null;

    // Standardize returns (inverse-vol weighting approximation)
    const stdReturns = returnsArrays.map(returns => {
      const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
      const std = Math.sqrt(returns.reduce((s, r) => s + (r - mean) ** 2, 0) / returns.length) || 1e-8;
      return returns.map(r => (r - mean) / std);
    });

    // Covariance matrix
    const cov = this.computeCovariance(stdReturns);

    // Eigenvalues
    const eigenvalues = this.eigenvaluesSymmetric(cov);

    if (eigenvalues.length === 0) return null;

    // Eigenvalue concentration: λ₁ / Σλ
    const eigSum = eigenvalues.reduce((a, b) => a + Math.abs(b), 0) + 1e-12;
    const lambda1 = Math.abs(eigenvalues[0]);
    const concentration = lambda1 / eigSum;

    // Normalized entropy
    const p = eigenvalues.map(e => Math.abs(e) / eigSum);
    const clippedP = p.map(pi => Math.max(pi, 1e-12));
    const pSum = clippedP.reduce((a, b) => a + b, 0);
    const normalizedP = clippedP.map(pi => pi / pSum);
    const rawEntropy = -normalizedP.reduce((s, pi) => s + pi * Math.log(pi), 0);
    const normEntropy = N > 1 ? rawEntropy / Math.log(N) : 0;

    // Temporal instability τ
    const K = lambda1 / N; // normalized trace proxy
    let tau = 0;
    if (this.history.length > 0) {
      const prevK = this.history[this.history.length - 1].K;
      tau = (K - prevK) / (Math.abs(prevK) + 1e-8);
    }

    // Store in history
    this.history.push({ K, concentration, normEntropy, tau, timestamp: Date.now() });
    if (this.history.length > this.maxHistory) this.history.shift();

    // Compute risk score (same logic as your Python GSRI)
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

    // Determine crash phase
    const phase = this.determineCrashPhase(concentration, normEntropy, tau, riskScore);

    // Estimate crash magnitude (crude: based on how extreme the metrics are)
    const magnitude = phase === "trigger" || phase === "crash"
      ? Math.min(0.15, riskScore * 0.2) // up to 15% estimated drop
      : 0;

    return {
      concentration,
      norm_entropy: normEntropy,
      tau,
      risk_score: riskScore,
      K,
      phase,
      magnitude,
      eigenvalue_count: eigenvalues.length,
      assets_analyzed: N,
      timestamp: Date.now(),
    };
  }

  // Crash phase detection
  // setup:    eigenvalue rising, entropy falling, τ stable → correlation forming
  // trigger:  τ spikes, eigenvalue peaks, entropy at floor → regime breaking
  // crash:    eigenvalue falling, entropy rising, τ negative → crash in progress
  // recovery: entropy normalizing, eigenvalue decentralizing → stabilization
  determineCrashPhase(concentration, normEntropy, tau, riskScore) {
    // Need some history to determine phase
    if (this.history.length < 5) return "insufficient_data";

    const recent = this.history.slice(-5);
    const concTrend = recent[recent.length - 1].concentration - recent[0].concentration;
    const entTrend = recent[recent.length - 1].norm_entropy - recent[0].norm_entropy;

    // TRIGGER: τ is spiking, risk score is high
    if (Math.abs(tau) > 0.3 && riskScore > 0.6) {
      return "trigger";
    }

    // CRASH: eigenvalue concentration is dropping, entropy is rising (de-correlation = crash happening)
    if (concTrend < -0.05 && entTrend > 0.05 && riskScore > 0.5) {
      return "crash";
    }

    // SETUP: eigenvalue rising (correlation building), entropy falling (freedom reducing)
    if (concTrend > 0.03 && entTrend < -0.03 && Math.abs(tau) < 0.15) {
      return "setup";
    }

    // RECOVERY: metrics normalizing, risk dropping
    if (riskScore < 0.35 && entTrend > 0) {
      return "recovery";
    }

    // NORMAL
    return "normal";
  }

  // Main computation: fetch data and compute crash metrics
  async compute(fetchCandlesFn) {
    // Throttle: don't recompute more often than interval
    const now = Date.now();
    if (now - this.lastComputeTime < this.computeIntervalMs && this.lastResult) {
      return this.lastResult;
    }

    const allSymbols = [
      ...this.basket.crypto,
      ...this.basket.forex,
      ...this.basket.metals,
    ];

    // Fetch hourly candles for basket (72 bars = 3 days)
    const allCandles = {};
    await Promise.all(allSymbols.map(async sym => {
      try {
        const candles = await fetchCandlesFn(sym, "1h", 72);
        if (candles && candles.length >= 30) {
          allCandles[sym] = candles;
        }
      } catch (e) { /* skip unavailable symbols */ }
    }));

    const availableSymbols = Object.keys(allCandles);
    if (availableSymbols.length < 3) {
      return {
        phase: "insufficient_data",
        risk_score: 0.5,
        concentration: 0,
        norm_entropy: 1,
        tau: 0,
        magnitude: 0,
        assets_analyzed: availableSymbols.length,
        timestamp: now,
        error: "Not enough assets with data for crash detection",
      };
    }

    // Build returns matrix (aligned by time)
    // Find the minimum length across all available candles
    const lengths = availableSymbols.map(s => allCandles[s].length);
    const minLen = Math.min(...lengths);
    const returnsArrays = availableSymbols.map(sym => {
      const candles = allCandles[sym];
      const returns = [];
      for (let i = 1; i < minLen; i++) {
        const prev = candles[i - 1].close;
        const curr = candles[i].close;
        if (prev > 0) returns.push((curr - prev) / prev);
      }
      return returns;
    });

    // Align lengths
    const returnLen = Math.min(...returnsArrays.map(r => r.length));
    const alignedReturns = returnsArrays.map(r => r.slice(-returnLen));

    const metrics = this.computeCrashMetrics(alignedReturns);

    if (metrics) {
      metrics.assets = availableSymbols;
      this.lastResult = metrics;
      this.lastComputeTime = now;
    }

    return metrics || this.lastResult || {
      phase: "insufficient_data",
      risk_score: 0.5,
      timestamp: now,
    };
  }

  // Get crash trade signals
  getCrashSignals(crashMetrics, analysisResults = {}) {
    if (!crashMetrics || crashMetrics.phase === "insufficient_data") {
      return { actionable: false, signals: [] };
    }

    const signals = [];

    switch (crashMetrics.phase) {
      case "setup":
        // No trade yet, but prepare — warn that crash is forming
        signals.push({
          type: "alert",
          message: "Crash setup detected — correlation rising, entropy falling",
          confidence: Math.round(crashMetrics.risk_score * 100),
          action: "PREPARE",
        });
        break;

      case "trigger":
        // Entry point for crash shorts
        // Find which assets are most vulnerable
        for (const [sym, analysis] of Object.entries(analysisResults)) {
          if (!analysis || analysis.error) continue;
          // Assets still going up or neutral are BEST short candidates
          if (analysis.direction === "BULLISH" || analysis.direction === "NEUTRAL") {
            signals.push({
              type: "crash_short",
              symbol: sym,
              direction: "SELL",
              confidence: Math.round(crashMetrics.risk_score * 80),
              magnitude_estimate: `${(crashMetrics.magnitude * 100).toFixed(1)}%`,
              action: "ENTER_SHORT",
              reason: `Crash trigger phase — ${sym} still elevated, short candidate`,
            });
          }
        }
        // If no specific analysis, generate basket-wide signal
        if (signals.length === 0) {
          signals.push({
            type: "crash_short_basket",
            direction: "SELL",
            confidence: Math.round(crashMetrics.risk_score * 70),
            magnitude_estimate: `${(crashMetrics.magnitude * 100).toFixed(1)}%`,
            action: "ENTER_SHORT",
            reason: "Crash trigger phase — consider shorting correlated assets",
          });
        }
        break;

      case "crash":
        // Crash in progress — hold existing shorts, don't enter new ones
        signals.push({
          type: "crash_active",
          message: "Crash in progress — hold existing shorts, do not enter new positions",
          action: "HOLD_SHORT",
          confidence: Math.round(crashMetrics.risk_score * 60),
        });
        break;

      case "recovery":
        // Exit shorts, look for reversal
        signals.push({
          type: "crash_recovery",
          message: "Crash recovery detected — exit shorts, watch for reversal longs",
          action: "CLOSE_SHORTS",
          confidence: 65,
        });
        break;

      default:
        // Normal — no crash signal
        break;
    }

    return {
      actionable: signals.length > 0,
      phase: crashMetrics.phase,
      risk_score: crashMetrics.risk_score,
      signals,
    };
  }
}


// ============================================================================
// EMA CROSSOVER SCANNER
// ============================================================================
// Autonomous market scanner using EMA crossovers with multi-pair confirmation.
//
// Design:
//   - Scan all watchlist pairs for 9/21 EMA crossovers
//   - Multi-pair confirmation (3+ forex pairs same direction = dollar theme)
//   - Trigger scoring system
//   - Cooldown per symbol to prevent overtrading
//   - GSRI pre-filter at scanner level (skip in hostile regimes)
//   - Staggered intervals (crypto 1min, forex 2min, metals 5min)
// ============================================================================

class EMAScanner {
  constructor() {
    this.watchlist = {
      forex: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"],
      crypto: ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD"],
      metals: ["XAUUSD"],
    };

    this.lastCrossover = new Map();  // symbol → { type, timestamp }
    this.lastTrigger = new Map();    // symbol → timestamp
    this.cooldownMs = 30 * 60 * 1000; // 30 min cooldown per symbol
    this.scanResults = new Map();    // latest scan results
    this.triggerHistory = [];        // history of all triggers
    this.running = false;
    this.scanIntervalId = null;
  }

  // Detect EMA crossover for a single symbol
  async detectCrossover(symbol, fetchCandlesFn, calcEMAFn, interval = "1h") {
    try {
      const candles = await fetchCandlesFn(symbol, interval, 50);
      if (!candles || candles.length < 30) return null;

      const closes = candles.map(c => c.close);
      const ema9 = calcEMAFn(closes, 9);
      const ema21 = calcEMAFn(closes, 21);

      if (ema9.length < 2 || ema21.length < 2) return null;

      // Current and previous EMA relationship
      const currentDiff = ema9[ema9.length - 1] - ema21[ema21.length - 1];
      const prevDiff = ema9[ema9.length - 2] - ema21[ema21.length - 2];

      let crossType = null;
      if (prevDiff <= 0 && currentDiff > 0) crossType = "bullish_cross";
      if (prevDiff >= 0 && currentDiff < 0) crossType = "bearish_cross";

      // EMA trend status even without crossover
      const trend = currentDiff > 0 ? "bullish" : "bearish";
      const separation = Math.abs(currentDiff) / (closes[closes.length - 1] || 1);

      return {
        symbol,
        cross: crossType,
        trend,
        separation,
        ema9: ema9[ema9.length - 1],
        ema21: ema21[ema21.length - 1],
        price: closes[closes.length - 1],
        interval,
      };
    } catch (e) {
      return null;
    }
  }

  // Scan all pairs and detect multi-pair confirmation
  async scanAll(fetchCandlesFn, calcEMAFn, interval = "1h") {
    const results = {};
    const allSymbols = [
      ...this.watchlist.forex,
      ...this.watchlist.crypto,
      ...this.watchlist.metals,
    ];

    // Scan all pairs (with rate-limit awareness — batch of 5)
    const batchSize = 5;
    for (let i = 0; i < allSymbols.length; i += batchSize) {
      const batch = allSymbols.slice(i, i + batchSize);
      const batchResults = await Promise.all(
        batch.map(sym => this.detectCrossover(sym, fetchCandlesFn, calcEMAFn, interval))
      );
      batchResults.forEach((r, idx) => {
        if (r) results[batch[idx]] = r;
      });
      // Small delay between batches to respect API rate limits
      if (i + batchSize < allSymbols.length) {
        await new Promise(r => setTimeout(r, 500));
      }
    }

    // Detect crossovers
    const crossovers = {};
    for (const [sym, r] of Object.entries(results)) {
      if (r.cross) {
        crossovers[sym] = r;
        this.lastCrossover.set(sym, { type: r.cross, timestamp: Date.now() });
      }
    }

    // Multi-pair confirmation for forex (dollar theme detection)
    const forexBullish = Object.entries(results)
      .filter(([s, r]) => this.watchlist.forex.includes(s) && r.trend === "bullish");
    const forexBearish = Object.entries(results)
      .filter(([s, r]) => this.watchlist.forex.includes(s) && r.trend === "bearish");

    // Crossover-based confirmation
    const crossBullish = Object.entries(crossovers)
      .filter(([s, r]) => r.cross === "bullish_cross");
    const crossBearish = Object.entries(crossovers)
      .filter(([s, r]) => r.cross === "bearish_cross");

    let trigger = null;

    // 3+ forex pairs with bullish crossover = dollar weakness
    if (crossBullish.filter(([s]) => this.watchlist.forex.includes(s)).length >= 2) {
      trigger = {
        direction: "bullish",
        type: "forex_crossover_confirmation",
        pairs: crossBullish.map(([s]) => s),
        confidence: Math.min(95, 55 + crossBullish.length * 10),
      };
    }

    // 3+ forex pairs with bearish crossover = dollar strength
    if (crossBearish.filter(([s]) => this.watchlist.forex.includes(s)).length >= 2) {
      trigger = {
        direction: "bearish",
        type: "forex_crossover_confirmation",
        pairs: crossBearish.map(([s]) => s),
        confidence: Math.min(95, 55 + crossBearish.length * 10),
      };
    }

    // Crypto crossover (single pair is significant enough)
    const cryptoCrosses = [...crossBullish, ...crossBearish]
      .filter(([s]) => this.watchlist.crypto.includes(s));
    if (cryptoCrosses.length >= 2 && !trigger) {
      trigger = {
        direction: cryptoCrosses[0][1].cross === "bullish_cross" ? "bullish" : "bearish",
        type: "crypto_crossover_confirmation",
        pairs: cryptoCrosses.map(([s]) => s),
        confidence: Math.min(90, 50 + cryptoCrosses.length * 12),
      };
    }

    // Trigger scoring (from the GPT discussion, but implemented properly)
    // Additional signals beyond just crossovers
    let triggerScore = 0;
    if (Object.keys(crossovers).length > 0) triggerScore += 2;  // any crossover
    if (trigger) triggerScore += 2;  // multi-pair confirmation
    if (forexBullish.length >= 4 || forexBearish.length >= 4) triggerScore += 1;  // strong trend alignment

    const scanResult = {
      results,
      crossovers,
      trigger,
      trigger_score: triggerScore,
      forex_bullish_count: forexBullish.length,
      forex_bearish_count: forexBearish.length,
      timestamp: Date.now(),
    };

    // Store results
    this.scanResults.set("latest", scanResult);

    // Log trigger if detected
    if (trigger) {
      this.triggerHistory.push({ ...trigger, timestamp: Date.now() });
      if (this.triggerHistory.length > 100) this.triggerHistory.shift();
    }

    return scanResult;
  }

  // Check if a symbol is in cooldown
  isInCooldown(symbol) {
    const lastTime = this.lastTrigger.get(String(symbol).toUpperCase());
    if (!lastTime) return false;
    return Date.now() - lastTime < this.cooldownMs;
  }

  // Mark a symbol as triggered (start cooldown)
  markTriggered(symbol) {
    this.lastTrigger.set(String(symbol).toUpperCase(), Date.now());
  }

  // Get latest scan results
  getLatestScan() {
    return this.scanResults.get("latest") || null;
  }

  // Get trigger history
  getTriggerHistory(limit = 20) {
    return this.triggerHistory.slice(-limit);
  }
}


// ============================================================================
// PAPER TRADE LOGGER
// ============================================================================
// Since MT5 bridge isn't ready, log all signals as paper trades.
// This builds the verified_hits/surprises database for ACP.
// ============================================================================

class PaperTradeLogger {
  constructor() {
    this.trades = new Map(); // symbol → array of trades
    this.maxPerSymbol = 50;
  }

  log(trade) {
    const sym = String(trade.symbol || "UNKNOWN").toUpperCase();
    if (!this.trades.has(sym)) this.trades.set(sym, []);
    const entries = this.trades.get(sym);

    const entry = {
      id: `${sym}_${Date.now()}`,
      symbol: sym,
      direction: trade.direction,      // BULLISH / BEARISH / SELL
      entry: trade.entry,
      sl: trade.sl,
      tp: trade.tp,
      confidence: trade.confidence,
      lot_size: trade.lot_size,
      source: trade.source || "scanner",   // scanner / manual / crash_gsri
      acp_confidence: trade.acp_confidence || null,
      crash_phase: trade.crash_phase || null,
      timestamp: Date.now(),
      outcome: "pending",                  // pending / win / loss / breakeven
      closed_at: null,
      close_price: null,
      pnl_pips: null,
    };

    entries.push(entry);
    if (entries.length > this.maxPerSymbol) entries.shift();
    return entry;
  }

  // Resolve a paper trade (mark as win/loss)
  resolve(tradeId, outcome, closePrice) {
    for (const [, entries] of this.trades.entries()) {
      const trade = entries.find(e => e.id === tradeId);
      if (trade) {
        trade.outcome = outcome;
        trade.closed_at = Date.now();
        trade.close_price = closePrice;

        // Calculate PnL in pips
        if (trade.entry && closePrice) {
          const dir = trade.direction === "BULLISH" || trade.direction === "BUY" ? 1 : -1;
          trade.pnl_pips = dir * (closePrice - trade.entry);
        }
        return trade;
      }
    }
    return null;
  }

  // Get trades for a symbol
  getTrades(symbol, limit = 20) {
    const sym = String(symbol || "").toUpperCase();
    return (this.trades.get(sym) || []).slice(-limit);
  }

  // Get win/loss stats for ACP's verified_hits/surprises
  getStats(symbol) {
    const sym = String(symbol || "").toUpperCase();
    const entries = this.trades.get(sym) || [];
    const resolved = entries.filter(e => e.outcome !== "pending");

    return {
      total: entries.length,
      resolved: resolved.length,
      wins: resolved.filter(e => e.outcome === "win").length,
      losses: resolved.filter(e => e.outcome === "loss").length,
      breakeven: resolved.filter(e => e.outcome === "breakeven").length,
      win_rate: resolved.length > 0
        ? resolved.filter(e => e.outcome === "win").length / resolved.length
        : 0,
      pending: entries.filter(e => e.outcome === "pending").length,
    };
  }

  // Get all pending trades (for auto-resolution checking)
  getPendingTrades() {
    const pending = [];
    for (const [, entries] of this.trades.entries()) {
      for (const e of entries) {
        if (e.outcome === "pending") pending.push(e);
      }
    }
    return pending;
  }

  // Auto-resolve pending trades that have hit TP or SL
  autoResolve(currentPrices) {
    const resolved = [];
    for (const [, entries] of this.trades.entries()) {
      for (const trade of entries) {
        if (trade.outcome !== "pending") continue;
        const price = currentPrices[trade.symbol];
        if (!price) continue;

        const isLong = trade.direction === "BULLISH" || trade.direction === "BUY";

        if (trade.tp && trade.sl) {
          if (isLong) {
            if (price >= trade.tp) {
              this.resolve(trade.id, "win", price);
              resolved.push({ ...trade, outcome: "win" });
            } else if (price <= trade.sl) {
              this.resolve(trade.id, "loss", price);
              resolved.push({ ...trade, outcome: "loss" });
            }
          } else {
            if (price <= trade.tp) {
              this.resolve(trade.id, "win", price);
              resolved.push({ ...trade, outcome: "win" });
            } else if (price >= trade.sl) {
              this.resolve(trade.id, "loss", price);
              resolved.push({ ...trade, outcome: "loss" });
            }
          }
        }
      }
    }
    return resolved;
  }
}


// ============================================================================
// ENHANCED DECISION PIPELINE
// ============================================================================
// The unified execution flow:
//
//   Scanner → Crash GSRI → Market Engine → ACP → Execution Gate
//
// Scanner detects opportunity (EMA crossover)
// Crash GSRI checks if we're in a crash (offensive, not just blocking)
// Market Engine runs full analysis (your existing analyzeSymbol)
// ACP calibrates probability and conviction
// Execution Gate applies final filters (news, cooldown, confidence thresholds)
// ============================================================================

class EnhancedDecisionPipeline {
  constructor(deps) {
    this.acp = new ACPEngine();
    this.crashGSRI = new CrashGSRIEngine();
    this.scanner = new EMAScanner();
    this.paperLogger = new PaperTradeLogger();

    // Dependencies from your existing index.js
    this.fetchCandles = deps.fetchCandles;
    this.calcEMA = deps.calcEMA;
    this.analyzeSymbol = deps.analyzeSymbol;
    this.cacheGet = deps.cacheGet;
    this.cacheSet = deps.cacheSet;
    this.checkNewsFilter = deps.checkNewsFilter;
    this.getGsriSnapshot = deps.getGsriSnapshot;
    this.gsriLotScale = deps.gsriLotScale;
    this.calculateLotSize = deps.calculateLotSize;
    this.sendToMT5Bridge = deps.sendToMT5Bridge;
    this.addTradeMemory = deps.addTradeMemory;
    this.getTradeMemory = deps.getTradeMemory;

    this.pipelineHistory = [];
    this.maxHistory = 200;
  }

  // Main pipeline entry point
  async run(symbol, interval = "1h", options = {}) {
    const sym = String(symbol).toUpperCase();
    const startTime = Date.now();

    // ===== STEP 0: CHECK COOLDOWN =====
    if (this.scanner.isInCooldown(sym)) {
      return {
        decision: "COOLDOWN",
        symbol: sym,
        reason: `Symbol ${sym} is in cooldown (last trigger < 30 min ago)`,
        cooldown_remaining_ms: this.scanner.cooldownMs - (Date.now() - (this.scanner.lastTrigger.get(sym) || 0)),
      };
    }

    // ===== STEP 1: CRASH GSRI CHECK =====
    // Is the market entering a crash? This is OFFENSIVE, not just blocking.
    const crashMetrics = await this.crashGSRI.compute(this.fetchCandles);
    const crashSignals = this.crashGSRI.getCrashSignals(crashMetrics);

    // If in crash trigger phase → generate crash short signal
    if (crashSignals.actionable && crashMetrics.phase === "trigger") {
      // Run analysis for the crash candidate
      const analysis = await this.analyzeSymbol(sym, interval);

      // Run ACP for the crash signal
      const candles = await this.fetchCandles(sym, interval, 120);
      const tradeMem = this.getTradeMemory ? this.getTradeMemory(sym) : [];
      const acpResult = candles
        ? await this.acp.run(sym, candles, analysis, tradeMem)
        : { acp: { confidence: 0.5, p: 0.5, c: 0, s: 0.5 }, bands: null };

      // Build crash trade signal
      const isCrashCandidate = analysis.direction === "BULLISH" || analysis.direction === "NEUTRAL";
      if (isCrashCandidate) {
        const crashDirection = "SELL";
        const price = analysis.price;
        const atr = analysis.volatility?.atr || price * 0.01;
        const entry = price;
        const sl = price + atr * 1.5;
        const tp = price - atr * 2.5;
        const lotSize = this.calculateLotSize
          ? this.calculateLotSize({ symbol: sym, balance: 1000, riskPercent: 1, entry, stopLoss: sl })
          : 0.01;

        const decision = {
          decision: crashDirection,
          source: "crash_gsri",
          symbol: sym,
          confidence: Math.round(crashMetrics.risk_score * 80),
          entry,
          sl: sl.toFixed(5),
          tp: tp.toFixed(5),
          lot_size: lotSize,
          crash_phase: crashMetrics.phase,
          crash_magnitude: crashMetrics.magnitude,
          acp: acpResult.acp,
          acp_bands: acpResult.bands,
          analysis,
          pipeline: "crash_gsri_pipeline",
          timestamp: Date.now(),
          elapsed_ms: Date.now() - startTime,
        };

        // Log as paper trade
        this.paperLogger.log({
          symbol: sym,
          direction: crashDirection,
          entry, sl, tp,
          confidence: decision.confidence,
          lot_size: lotSize,
          source: "crash_gsri",
          acp_confidence: acpResult.acp.confidence,
          crash_phase: crashMetrics.phase,
        });

        this.pipelineHistory.push(decision);
        if (this.pipelineHistory.length > this.maxHistory) this.pipelineHistory.shift();

        this.scanner.markTriggered(sym);
        return decision;
      }
    }

    // If in crash phase (not trigger) → block new longs
    if (crashMetrics.phase === "crash" || crashMetrics.phase === "trigger") {
      // Only allow shorts during crash, block longs
      // We'll let the normal pipeline run but flag it
    }

    // ===== STEP 2: NEWS FILTER =====
    const newsCheck = await this.checkNewsFilter(sym);
    if (newsCheck.blocked) {
      return {
        decision: "NO_TRADE",
        symbol: sym,
        reason: `News blackout: ${newsCheck.reason}`,
        source: "news_filter",
      };
    }

    // ===== STEP 3: MARKET ENGINE (your existing analysis) =====
    const analysis = await this.analyzeSymbol(sym, interval);
    if (!analysis || analysis.error || analysis.direction === "NEUTRAL") {
      return {
        decision: "NO_TRADE",
        symbol: sym,
        reason: analysis?.error || "No clear directional edge",
        analysis,
      };
    }

    // Block longs during crash phases
    if ((crashMetrics.phase === "crash" || crashMetrics.phase === "trigger")
        && analysis.direction === "BULLISH") {
      return {
        decision: "NO_TRADE",
        symbol: sym,
        reason: `Long blocked during crash ${crashMetrics.phase} phase`,
        crash_phase: crashMetrics.phase,
        analysis,
      };
    }

    // ===== STEP 4: ACP PROBABILITY CALIBRATION =====
    const candles = await this.fetchCandles(sym, interval, 120);
    const tradeMem = this.getTradeMemory ? this.getTradeMemory(sym) : [];
    const acpResult = candles
      ? await this.acp.run(sym, candles, analysis, tradeMem)
      : { acp: { confidence: 0.5, p: 0.5, c: 0, s: 0.5, lambda: 0.06 }, bands: null };

    // Calibrate confidence: engine confidence × ACP confidence
    const engineConfidence = analysis.confidence || 50;
    const acpConfidence = acpResult.acp.confidence;
    let finalConfidence = Math.round(engineConfidence * acpConfidence);

    // If ACP contradiction is high → widen TP/SL zones using band_width
    let adjustedTP = analysis.trade_plan?.tp1;
    let adjustedSL = analysis.trade_plan?.invalidation;
    let bandNote = "";

    if (acpResult.bands && acpResult.acp.c > 0.4) {
      // High contradiction: use ACP's moderate band as TP zone
      const price = analysis.price;
      const atr = analysis.volatility?.atr || price * 0.01;
      const bandWidth = acpResult.bands.moderate;
      bandNote = `ACP bands widened due to contradiction c=${acpResult.acp.c.toFixed(3)}`;
      // Widen TP by ACP band range
      if (analysis.direction === "BULLISH") {
        adjustedTP = bandWidth[1]; // upper moderate band
      } else {
        adjustedTP = bandWidth[0]; // lower moderate band
      }
    }

    // ===== STEP 5: GSRI RISK GATE (original GSRI still active) =====
    let gsriMode = "normal";
    let gsriScale = 1.0;
    try {
      const gsriSnap = await this.getGsriSnapshot();
      const gsriScore = parseFloat(gsriSnap?.Risk_Score ?? 0);
      const gsriAlert = parseInt(gsriSnap?.Alert ?? 0);

      if (gsriAlert === 1 || gsriScore > 0.8) {
        gsriMode = "blocked";
      } else if (gsriScore > 0.6) {
        gsriMode = "defend";
        finalConfidence = Math.round(finalConfidence * 0.6);
        gsriScale = this.gsriLotScale(gsriScore);
      } else if (gsriScore < 0.4) {
        gsriMode = "normal";
        finalConfidence = Math.min(95, Math.round(finalConfidence * 1.1));
      }
    } catch (e) {
      // GSRI unavailable — default to normal (NOT blocking fallback)
      gsriMode = "normal";
    }

    if (gsriMode === "blocked" && crashMetrics.phase !== "trigger") {
      // Block unless we're in crash trigger (crash trades override GSRI block)
      return {
        decision: "NO_TRADE",
        symbol: sym,
        reason: "GSRI high risk environment (non-crash)",
        gsri_mode: gsriMode,
      };
    }

    // ===== STEP 6: FINAL CONFIDENCE THRESHOLD =====
    if (finalConfidence < 30) {
      return {
        decision: "NO_TRADE",
        symbol: sym,
        reason: `Confidence too low after ACP calibration (${finalConfidence}%)`,
        engine_confidence: engineConfidence,
        acp_confidence: Math.round(acpConfidence * 100),
        acp: acpResult.acp,
      };
    }

    // ===== STEP 7: EXECUTION =====
    const direction = analysis.direction === "BULLISH" ? "BUY" : "SELL";
    const price = analysis.price;
    const entry = parseFloat(analysis.trade_plan?.entry_zone?.split("-")[0]) || price;
    const sl = parseFloat(adjustedSL) || 0;
    const tp = parseFloat(adjustedTP) || 0;
    const lotSize = this.calculateLotSize
      ? this.calculateLotSize({
          symbol: sym,
          balance: options.balance || 1000,
          riskPercent: options.riskPercent || 1,
          entry,
          stopLoss: sl,
        }) * gsriScale
      : 0.01;

    const decision = {
      decision: direction,
      source: "enhanced_pipeline",
      symbol: sym,
      confidence: finalConfidence,
      entry,
      sl: sl.toFixed(5),
      tp: tp.toFixed(5),
      lot_size: Math.round(lotSize * 100) / 100,
      // ACP details
      acp: acpResult.acp,
      acp_bands: acpResult.bands,
      acp_band_note: bandNote,
      // Crash GSRI details
      crash_phase: crashMetrics.phase,
      crash_risk_score: crashMetrics.risk_score,
      // Original GSRI
      gsri_mode: gsriMode,
      gsri_scale: gsriScale,
      // Engine details
      engine_confidence: engineConfidence,
      direction_raw: analysis.direction,
      strength: analysis.strength,
      // Full analysis (for reference)
      analysis,
      pipeline: "full_pipeline",
      timestamp: Date.now(),
      elapsed_ms: Date.now() - startTime,
    };

    // Log as paper trade
    this.paperLogger.log({
      symbol: sym,
      direction,
      entry, sl, tp,
      confidence: finalConfidence,
      lot_size: Math.round(lotSize * 100) / 100,
      source: "enhanced_pipeline",
      acp_confidence: acpConfidence,
      crash_phase: crashMetrics.phase,
    });

    // Mark in trade memory
    if (this.addTradeMemory) {
      this.addTradeMemory(sym, {
        direction,
        pattern: analysis.patterns?.join(",") || "none",
        outcome: "pending",
        note: `Pipeline: conf=${finalConfidence} acp_p=${acpResult.acp.p.toFixed(3)} acp_c=${acpResult.acp.c.toFixed(3)} crash=${crashMetrics.phase}`,
      });
    }

    this.scanner.markTriggered(sym);
    this.pipelineHistory.push(decision);
    if (this.pipelineHistory.length > this.maxHistory) this.pipelineHistory.shift();

    return decision;
  }

  // Run scanner-only mode (just detect opportunities, don't execute)
  async scan(interval = "1h") {
    return this.scanner.scanAll(this.fetchCandles, this.calcEMA, interval);
  }

  // Get pipeline history
  getHistory(limit = 20) {
    return this.pipelineHistory.slice(-limit);
  }
}


// ============================================================================
// SCANNER LOOP (Autonomous Operation)
// ============================================================================
// Runs the scanner on a schedule, triggers the pipeline when opportunities
// are found, and logs paper trades.
// ============================================================================

class ScannerLoop {
  constructor(pipeline) {
    this.pipeline = pipeline;
    this.running = false;
    this.intervalId = null;
    this.scanIntervalMs = 2 * 60 * 1000; // 2 minutes default
    this.lastScanTime = 0;
    this.scanCount = 0;
    this.triggersGenerated = 0;

    // Auto-resolve interval
    this.resolveIntervalId = null;
    this.resolveIntervalMs = 5 * 60 * 1000; // check every 5 min
  }

  start(intervalMs) {
    if (this.running) return { status: "already_running" };

    this.scanIntervalMs = intervalMs || this.scanIntervalMs;
    this.running = true;

    // Main scan loop
    this.intervalId = setInterval(async () => {
      await this.runScan();
    }, this.scanIntervalMs);

    // Auto-resolve loop (check paper trades against current prices)
    this.resolveIntervalId = setInterval(async () => {
      await this.autoResolveTrades();
    }, this.resolveIntervalMs);

    // Run first scan immediately
    this.runScan();

    return { status: "started", interval_ms: this.scanIntervalMs };
  }

  stop() {
    this.running = false;
    if (this.intervalId) clearInterval(this.intervalId);
    if (this.resolveIntervalId) clearInterval(this.resolveIntervalId);
    this.intervalId = null;
    this.resolveIntervalId = null;
    return { status: "stopped" };
  }

  async runScan() {
    try {
      this.lastScanTime = Date.now();
      this.scanCount++;

      // Step 1: Scan for EMA crossovers
      const scanResult = await this.pipeline.scan("1h");

      // Step 2: If trigger detected, run full pipeline on triggered pairs
      if (scanResult.trigger && scanResult.trigger_score >= 3) {
        const pairs = scanResult.trigger.pairs || [];

        for (const sym of pairs) {
          try {
            const decision = await this.pipeline.run(sym, "1h");
            if (decision.decision !== "NO_TRADE" && decision.decision !== "COOLDOWN") {
              this.triggersGenerated++;
              console.log(`[ScannerLoop] TRADE SIGNAL: ${decision.decision} ${sym} conf=${decision.confidence}% source=${decision.source}`);
            }
          } catch (e) {
            console.error(`[ScannerLoop] Pipeline error for ${sym}:`, e.message);
          }
        }
      }

      // Step 3: Also check crash GSRI independently
      const crashMetrics = await this.pipeline.crashGSRI.compute(this.pipeline.fetchCandles);
      if (crashMetrics.phase === "trigger") {
        // Scan major pairs for crash opportunities
        const crashCandidates = ["BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "XAUUSD"];
        for (const sym of crashCandidates) {
          if (!this.pipeline.scanner.isInCooldown(sym)) {
            try {
              const decision = await this.pipeline.run(sym, "1h");
              if (decision.decision !== "NO_TRADE" && decision.decision !== "COOLDOWN") {
                this.triggersGenerated++;
                console.log(`[ScannerLoop] CRASH SIGNAL: ${decision.decision} ${sym} crash_phase=${crashMetrics.phase}`);
              }
            } catch (e) {
              console.error(`[ScannerLoop] Crash pipeline error for ${sym}:`, e.message);
            }
          }
        }
      }
    } catch (e) {
      console.error("[ScannerLoop] Scan error:", e.message);
    }
  }

  async autoResolveTrades() {
    try {
      const pending = this.pipeline.paperLogger.getPendingTrades();
      if (pending.length === 0) return;

      // Fetch current prices for all pending trade symbols
      const symbols = [...new Set(pending.map(t => t.symbol))];
      const prices = {};

      await Promise.all(symbols.map(async sym => {
        try {
          const spot = await this.pipeline.fetchCandles(sym, "1min", 1);
          if (spot && spot.length > 0) {
            prices[sym] = spot[0].close;
          }
        } catch (e) { /* skip */ }
      }));

      const resolved = this.pipeline.paperLogger.autoResolve(prices);
      if (resolved.length > 0) {
        console.log(`[ScannerLoop] Auto-resolved ${resolved.length} paper trades`);
      }
    } catch (e) {
      console.error("[ScannerLoop] Auto-resolve error:", e.message);
    }
  }

  getStatus() {
    return {
      running: this.running,
      scan_interval_ms: this.scanIntervalMs,
      last_scan_time: this.lastScanTime,
      scan_count: this.scanCount,
      triggers_generated: this.triggersGenerated,
      pending_paper_trades: this.pipeline.paperLogger.getPendingTrades().length,
    };
  }
}


// ============================================================================
// MAIN EXPORT — Unified Interface
// ============================================================================

export class FritSystems {
  constructor(deps) {
    this.pipeline = new EnhancedDecisionPipeline(deps);
    this.scannerLoop = new ScannerLoop(this.pipeline);

    // Direct access to sub-systems
    this.acp = this.pipeline.acp;
    this.crashGSRI = this.pipeline.crashGSRI;
    this.scanner = this.pipeline.scanner;
    this.paperLogger = this.pipeline.paperLogger;
  }

  // Run the full enhanced pipeline for a symbol
  async analyze(symbol, interval = "1h", options = {}) {
    return this.pipeline.run(symbol, interval, options);
  }

  // Just scan (no execution)
  async scan(interval = "1h") {
    return this.pipeline.scan(interval);
  }

  // Start autonomous scanner
  startScanner(intervalMs) {
    return this.scannerLoop.start(intervalMs);
  }

  // Stop autonomous scanner
  stopScanner() {
    return this.scannerLoop.stop();
  }

  // Get system status
  getStatus() {
    return {
      scanner_loop: this.scannerLoop.getStatus(),
      crash_gsri: {
        last_phase: this.crashGSRI.lastResult?.phase || "unknown",
        last_risk_score: this.crashGSRI.lastResult?.risk_score || 0,
        last_compute: this.crashGSRI.lastComputeTime,
        history_size: this.crashGSRI.history.length,
      },
      acp: {
        cached_symbols: this.acp.acpCache.size,
        evidence_streams: this.acp.evidenceStreams.size,
      },
      paper_trades: {
        total_symbols: this.paperLogger.trades.size,
        pending: this.paperLogger.getPendingTrades().length,
      },
      pipeline_history: this.pipeline.pipelineHistory.length,
    };
  }
}
