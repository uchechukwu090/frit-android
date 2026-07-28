// ============================================================================
// FRIT ENHANCED SYSTEMS — SMC+CRT Scanner + Crash GSRI + Paper Logger
// ============================================================================
//
//   1. SMC+CRT Scanner       — scans watchlist for SMC/CRT setups
//   2. Crash GSRI Engine      — offensive crash detection & trading
//   3. Paper Trade Logger     — track signals until MT5 is ready
//   4. Enhanced Decision Pipeline — unified execution flow
//
// INTEGRATION: Add to your index.js:
//   import { FritSystems } from "./frit_systems.js";
//   const frit = new FritSystems({ fetchCandles, analyzeSymbol, cacheGet, cacheSet, checkNewsFilter, getGsriSnapshot, gsriLotScale, calculateLotSize, sendToMT5Bridge, addTradeMemory, getTradeMemory });
// ============================================================================

// ============================================================================
// CRASH GSRI ENGINE
// ============================================================================
// Transforms GSRI from a defensive risk gate into an offensive crash
// detection and trading engine.
//
// Key differences from Python GSRI:
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
    this.history = [];
    this.maxHistory = 200;
    this.lastResult = null;
    this.lastComputeTime = 0;
    this.computeIntervalMs = 5 * 60 * 1000;
  }

  computeCovariance(returnsArrays) {
    const N = returnsArrays.length;
    if (N === 0) return [];
    const T = returnsArrays[0].length;
    const means = returnsArrays.map(r => r.reduce((a, b) => a + b, 0) / T);
    const cov = [];
    for (let i = 0; i < N; i++) {
      cov[i] = [];
      for (let j = 0; j < N; j++) {
        const x = 0.9998; // simplified for JS speed
        cov[i][j] = 0;
        for (let t = 0; t < T; t++) {
          cov[i][j] += (returnsArrays[i][t] - means[i]) * (returnsArrays[j][t] - means[j]);
        }
        cov[i][j] /= (T - 1);
      }
    }
    return cov;
  }

  jacobiEigenvalues(matrix, maxIter = 100) {
    const n = matrix.length;
    if (n === 0) return [];
    let A = matrix.map(row => [...row]);
    let eigenvals = new Array(n).fill(0);
    for (let iter = 0; iter < maxIter; iter++) {
      let p = 0, q = 1;
      let maxOff = Math.abs(A[0][1] || 0);
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const v = Math.abs(A[i][j] || 0);
          if (v > maxOff) { maxOff = v; p = i; q = j; }
        }
      }
      if (maxOff < 1e-10) break;
      const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(1 + t * t);
      const s = t * c;
      const Ap = A.map(row => [...row]);
      for (let i = 0; i < n; i++) {
        A[i][p] = c * Ap[i][p] - s * Ap[i][q];
        A[i][q] = s * Ap[i][p] + c * Ap[i][q];
        A[p][i] = A[i][p];
        A[q][i] = A[i][q];
      }
      A[p][p] = c * c * Ap[p][p] + s * s * Ap[q][q] - 2 * s * c * Ap[p][q];
      A[q][q] = s * s * Ap[p][p] + c * c * Ap[q][q] + 2 * s * c * Ap[p][q];
      A[p][q] = 0; A[q][p] = 0;
    }
    for (let i = 0; i < n; i++) eigenvals[i] = A[i][i];
    eigenvals.sort((a, b) => b - a);
    return eigenvals;
  }

  computeEntropy(eigenvalues) {
    const total = eigenvalues.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;
    return -eigenvalues.reduce((s, v) => {
      const p = v / total;
      return p > 0 ? s + p * Math.log(p) : s;
    }, 0) / Math.log(eigenvalues.length);
  }

  async compute(fetchCandlesFn) {
    const now = Date.now();
    if (this.lastResult && now - this.lastComputeTime < this.computeIntervalMs) {
      return this.lastResult;
    }
    const allSymbols = [...this.basket.crypto, ...this.basket.forex, ...this.basket.metals];
    const bars = await Promise.all(
      allSymbols.map(sym => fetchCandlesFn(sym, "1h", 72).then(d => d || []).catch(() => []))
    );
    const valid = bars.filter(b => b.length >= 48);
    if (valid.length < 3) {
      this.lastResult = { phase: "unknown", risk_score: 0.5, reason: "Insufficient data" };
      this.lastComputeTime = now;
      return this.lastResult;
    }
    const returnsArrays = valid.map(bars => {
      const closes = bars.map(c => c.close);
      const rets = [];
      for (let i = 1; i < closes.length; i++) {
        rets.push((closes[i] - closes[i - 1]) / closes[i - 1]);
      }
      return rets;
    });
    const minLen = Math.min(...returnsArrays.map(r => r.length));
    const aligned = returnsArrays.map(r => r.slice(-minLen));
    const cov = this.computeCovariance(aligned);
    const eigenvals = this.jacobiEigenvalues(cov);
    const totalVar = eigenvals.reduce((a, b) => a + b, 0);
    const concentration = totalVar > 0 ? eigenvals[0] / totalVar : 0;
    const entropy = this.computeEntropy(eigenvals);
    const riskScore = Math.min(1, Math.max(0, concentration * (1 - entropy)));
    this.history.push({ time: now, risk_score: riskScore, concentration, entropy });
    if (this.history.length > this.maxHistory) this.history.shift();
    let phase = "normal";
    let magnitude = 0;
    if (riskScore > 0.75 && entropy < 0.3) {
      phase = "crash";
      magnitude = Math.round(riskScore * 10);
    } else if (riskScore > 0.6) {
      phase = "trigger";
      magnitude = Math.round((riskScore - 0.6) * 10);
    } else if (riskScore > 0.5) {
      phase = "setup";
    }
    if (this.history.length >= 3) {
      const recent = this.history.slice(-3).map(h => h.risk_score);
      if (recent[2] < recent[1] * 0.8 && phase === "crash") phase = "recovery";
    }
    this.lastResult = { phase, risk_score: riskScore, concentration, entropy, magnitude, timestamp: now, basket_size: valid.length };
    this.lastComputeTime = now;
    return this.lastResult;
  }

  getCrashSignals(metrics) {
    if (!metrics || metrics.phase === "unknown") return { actionable: false };
    return { actionable: metrics.phase === "trigger" || metrics.phase === "crash", phase: metrics.phase, confidence: Math.round(metrics.risk_score * 80) };
  }

  getCached() {
    return this.lastResult;
  }
}

// ============================================================================
// SMC+CRT SCANNER
// ============================================================================
// Autonomous market scanner using SMC+CRT concepts — no EMAs/indicators.
//
// Scans watchlist for:
//   - CRT setups (candle range sweep + close-back-in)
//   - SMC setups (CHoCH/MSS + OB/FVG at structure breaks)
//   - Multi-pair confirmation (dollar theme)
//   - Scores by confluence strength
// ============================================================================

const {
  analyzeSMC_CRT,
} = await import("./smc_crt_strategy.js");

let _smcModule = null;
async function getSMCModule() {
  if (!_smcModule) _smcModule = await import("./smc_crt_strategy.js");
  return _smcModule;
}

class SMC_CRT_Scanner {
  constructor() {
    this.watchlist = {
      forex: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"],
      crypto: ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD"],
      metals: ["XAUUSD"],
    };
    this.lastTrigger = new Map();
    this.cooldownMs = 30 * 60 * 1000;
    this.scanResults = new Map();
    this.triggerHistory = [];
  }

  async scanSymbol(symbol, fetchCandlesFn, interval = "1h") {
    try {
      const candles = await fetchCandlesFn(symbol, interval, 100);
      if (!candles || candles.length < 30) return null;
      const price = candles.at(-1).close;
      const atr = calcATRSimple(candles, 14);
      const smc = await (await getSMCModule()).analyzeSMC_CRT(candles, price, atr, null);
      if (!smc || smc.signal === "insufficient_data") return null;

      let signalType = null;
      let priority = 0;
      if (smc.signal === "buy" || smc.signal === "sell") {
        signalType = smc.signal === "buy" ? "bullish_smc" : "bearish_smc";
        priority = smc.crt ? 3 : 2; // CRT setup > SMC-only
      }

      return {
        symbol,
        signal: smc.signal === "buy" ? "bullish" : smc.signal === "sell" ? "bearish" : "neutral",
        signal_type: signalType,
        priority,
        confidence: smc.confidence,
        structure: smc.structure?.trend || "ranging",
        choch: smc.choch,
        crt: smc.crt,
        order_blocks: smc.order_blocks,
        fvgs: smc.fvgs,
        sweeps: smc.sweeps,
        entry: smc.entry,
        reasons: smc.reasons,
        price,
        interval,
      };
    } catch (e) {
      return null;
    }
  }

  async scanAll(fetchCandlesFn, interval = "1h") {
    const results = {};
    const allSymbols = [
      ...this.watchlist.forex,
      ...this.watchlist.crypto,
      ...this.watchlist.metals,
    ];

    const batchSize = 5;
    for (let i = 0; i < allSymbols.length; i += batchSize) {
      const batch = allSymbols.slice(i, i + batchSize);
      const batchResults = await Promise.all(
        batch.map(sym => this.scanSymbol(sym, fetchCandlesFn, interval))
      );
      batchResults.forEach((r, idx) => {
        if (r) results[batch[idx]] = r;
      });
      if (i + batchSize < allSymbols.length) {
        await new Promise(r => setTimeout(r, 500));
      }
    }

    // Categorize signals
    const bullish = Object.entries(results).filter(([, r]) => r.signal === "bullish" && r.confidence >= 40);
    const bearish = Object.entries(results).filter(([, r]) => r.signal === "bearish" && r.confidence >= 40);

    // Multi-pair confirmation for forex (dollar theme)
    const forexBullish = bullish.filter(([s]) => this.watchlist.forex.includes(s));
    const forexBearish = bearish.filter(([s]) => this.watchlist.forex.includes(s));

    // CRT-specific signals (highest priority)
    const crtSetups = Object.entries(results).filter(([, r]) => r.crt && r.confidence >= 50);

    let trigger = null;
    let triggerScore = 0;

    // CRT trigger (2+ pairs with CRT setups in same direction)
    const crtBullish = crtSetups.filter(([, r]) => r.signal === "bullish");
    const crtBearish = crtSetups.filter(([, r]) => r.signal === "bearish");
    if (crtBullish.length >= 1) {
      trigger = {
        direction: "bullish",
        type: "crt_setup",
        pairs: crtBullish.map(([s]) => s),
        confidence: Math.min(95, 55 + crtBullish.length * 10),
      };
      triggerScore += 3;
    } else if (crtBearish.length >= 1) {
      trigger = {
        direction: "bearish",
        type: "crt_setup",
        pairs: crtBearish.map(([s]) => s),
        confidence: Math.min(95, 55 + crtBearish.length * 10),
      };
      triggerScore += 3;
    }

    // Forex multi-pair confirmation
    if (forexBullish.length >= 3 && !trigger) {
      trigger = {
        direction: "bullish",
        type: "forex_smc_confirmation",
        pairs: forexBullish.map(([s]) => s),
        confidence: Math.min(90, 50 + forexBullish.length * 8),
      };
      triggerScore += 2;
    } else if (forexBearish.length >= 3 && !trigger) {
      trigger = {
        direction: "bearish",
        type: "forex_smc_confirmation",
        pairs: forexBearish.map(([s]) => s),
        confidence: Math.min(90, 50 + forexBearish.length * 8),
      };
      triggerScore += 2;
    }

    // CHoCH/MSS signals
    const chochSignals = Object.entries(results).filter(([, r]) => r.choch && r.confidence >= 50);
    const chochBullish = chochSignals.filter(([, r]) => r.choch.direction === "bullish");
    const chochBearish = chochSignals.filter(([, r]) => r.choch.direction === "bearish");
    if (chochBullish.length >= 2) triggerScore += 1;
    if (chochBearish.length >= 2) triggerScore += 1;

    const scanResult = {
      results,
      bullish: bullish.map(([s]) => s),
      bearish: bearish.map(([s]) => s),
      crt_setups: crtSetups.map(([s, r]) => ({ symbol: s, type: r.crt?.type, confidence: r.confidence })),
      trigger,
      trigger_score: triggerScore,
      forex_bullish_count: forexBullish.length,
      forex_bearish_count: forexBearish.length,
      timestamp: Date.now(),
    };

    this.scanResults.set("latest", scanResult);

    if (trigger) {
      this.triggerHistory.push({ ...trigger, timestamp: Date.now() });
      if (this.triggerHistory.length > 100) this.triggerHistory.shift();
    }

    return scanResult;
  }

  isInCooldown(symbol) {
    const lastTime = this.lastTrigger.get(String(symbol).toUpperCase());
    if (!lastTime) return false;
    return Date.now() - lastTime < this.cooldownMs;
  }

  markTriggered(symbol) {
    this.lastTrigger.set(String(symbol).toUpperCase(), Date.now());
  }

  getLatestScan() {
    return this.scanResults.get("latest") || null;
  }

  getTriggerHistory(limit = 20) {
    return this.triggerHistory.slice(-limit);
  }
}

function calcATRSimple(candles, period = 14) {
  if (!candles || candles.length < period + 1) return 0;
  const trs = [];
  for (let i = 1; i < candles.length; i++) {
    const c = candles[i];
    const p = candles[i - 1];
    trs.push(Math.max(c.high - c.low, Math.abs(c.high - p.close), Math.abs(c.low - p.close)));
  }
  const recent = trs.slice(-period);
  return recent.reduce((a, b) => a + b, 0) / recent.length;
}

// ============================================================================
// PAPER TRADE LOGGER
// ============================================================================

class PaperTradeLogger {
  constructor() {
    this.trades = new Map();
    this.maxPerSymbol = 50;
  }

  log(trade) {
    const sym = String(trade.symbol || "UNKNOWN").toUpperCase();
    if (!this.trades.has(sym)) this.trades.set(sym, []);
    const entries = this.trades.get(sym);
    const entry = {
      id: `${sym}_${Date.now()}`,
      symbol: sym,
      direction: trade.direction,
      entry: trade.entry,
      sl: trade.sl,
      tp: trade.tp,
      confidence: trade.confidence,
      lot_size: trade.lot_size,
      source: trade.source || "scanner",
      crash_phase: trade.crash_phase || null,
      timestamp: Date.now(),
      outcome: "pending",
      closed_at: null,
      close_price: null,
      pnl_pips: null,
    };
    entries.push(entry);
    if (entries.length > this.maxPerSymbol) entries.shift();
    return entry;
  }

  resolve(tradeId, outcome, closePrice) {
    for (const [, entries] of this.trades.entries()) {
      const trade = entries.find(e => e.id === tradeId);
      if (trade) {
        trade.outcome = outcome;
        trade.closed_at = Date.now();
        trade.close_price = closePrice;
        if (trade.entry && closePrice) {
          const dir = trade.direction === "BULLISH" || trade.direction === "BUY" ? 1 : -1;
          trade.pnl_pips = dir * (closePrice - trade.entry);
        }
        return trade;
      }
    }
    return null;
  }

  getTrades(symbol, limit = 20) {
    const sym = String(symbol || "").toUpperCase();
    return (this.trades.get(sym) || []).slice(-limit);
  }

  getStats(symbol) {
    const sym = String(symbol || "").toUpperCase();
    const entries = this.trades.get(sym) || [];
    const resolved = entries.filter(e => e.outcome !== "pending");
    const wins = resolved.filter(e => e.outcome === "win").length;
    const losses = resolved.filter(e => e.outcome === "loss").length;
    const total = wins + losses;
    return {
      total_trades: entries.length,
      resolved: resolved.length,
      pending: entries.length - resolved.length,
      wins,
      losses,
      win_rate: total > 0 ? Math.round(wins / total * 100) : 0,
    };
  }

  getPendingTrades() {
    const pending = [];
    for (const [, entries] of this.trades.entries()) {
      pending.push(...entries.filter(e => e.outcome === "pending"));
    }
    return pending;
  }

  autoResolve(currentPrices) {
    const resolved = [];
    for (const [, entries] of this.trades.entries()) {
      for (const trade of entries) {
        if (trade.outcome !== "pending") continue;
        const price = currentPrices[trade.symbol];
        if (!price) continue;
        const dir = trade.direction === "BUY" || trade.direction === "BULLISH" ? 1 : -1;
        if (dir === 1 && price >= trade.tp) {
          trade.outcome = "win";
          trade.closed_at = Date.now();
          trade.close_price = price;
          trade.pnl_pips = price - trade.entry;
          resolved.push(trade);
        } else if (dir === 1 && price <= trade.sl) {
          trade.outcome = "loss";
          trade.closed_at = Date.now();
          trade.close_price = price;
          trade.pnl_pips = price - trade.entry;
          resolved.push(trade);
        } else if (dir === -1 && price <= trade.tp) {
          trade.outcome = "win";
          trade.closed_at = Date.now();
          trade.close_price = price;
          trade.pnl_pips = trade.entry - price;
          resolved.push(trade);
        } else if (dir === -1 && price >= trade.sl) {
          trade.outcome = "loss";
          trade.closed_at = Date.now();
          trade.close_price = price;
          trade.pnl_pips = trade.entry - price;
          resolved.push(trade);
        }
      }
    }
    return resolved;
  }
}

// ============================================================================
// ENHANCED DECISION PIPELINE (SMC+CRT)
// ============================================================================

class EnhancedDecisionPipeline {
  constructor(deps) {
    this.crashGSRI = new CrashGSRIEngine();
    this.scanner = new SMC_CRT_Scanner();
    this.paperLogger = new PaperTradeLogger();

    this.fetchCandles = deps.fetchCandles;
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

  async run(symbol, interval = "1h", options = {}) {
    const sym = String(symbol).toUpperCase();
    const startTime = Date.now();

    // ===== STEP 0: COOLDOWN CHECK =====
    if (this.scanner.isInCooldown(sym)) {
      return {
        decision: "COOLDOWN",
        symbol: sym,
        reason: `Symbol ${sym} is in cooldown (last trigger < 30 min ago)`,
        cooldown_remaining_ms: this.scanner.cooldownMs - (Date.now() - (this.scanner.lastTrigger.get(sym) || 0)),
      };
    }

    // ===== STEP 1: CRASH GSRI =====
    const crashMetrics = await this.crashGSRI.compute(this.fetchCandles);
    const crashSignals = this.crashGSRI.getCrashSignals(crashMetrics);

    if (crashSignals.actionable && crashMetrics.phase === "trigger") {
      const analysis = await this.analyzeSymbol(sym, interval);
      const isCrashCandidate = analysis.direction === "BULLISH" || analysis.direction === "NEUTRAL";
      if (isCrashCandidate) {
        const price = analysis.price;
        const atr = analysis.volatility?.atr || price * 0.01;
        const entry = price;
        const sl = price + atr * 1.5;
        const tp = price - atr * 2.5;
        const lotSize = this.calculateLotSize
          ? this.calculateLotSize({ symbol: sym, balance: options.balance || 1000, riskPercent: 1, entry, stopLoss: sl })
          : 0.01;

        const decision = {
          decision: "SELL",
          source: "crash_gsri",
          symbol: sym,
          confidence: Math.round(crashMetrics.risk_score * 80),
          entry,
          sl: sl.toFixed(5),
          tp: tp.toFixed(5),
          lot_size: lotSize,
          crash_phase: crashMetrics.phase,
          crash_magnitude: crashMetrics.magnitude,
          analysis,
          pipeline: "crash_gsri_pipeline",
          timestamp: Date.now(),
          elapsed_ms: Date.now() - startTime,
        };

        this.paperLogger.log({ symbol: sym, direction: "SELL", entry, sl, tp, confidence: decision.confidence, lot_size: lotSize, source: "crash_gsri", crash_phase: crashMetrics.phase });
        this.pipelineHistory.push(decision);
        if (this.pipelineHistory.length > this.maxHistory) this.pipelineHistory.shift();
        this.scanner.markTriggered(sym);
        return decision;
      }
    }

    if (crashMetrics.phase === "crash" || crashMetrics.phase === "trigger") {
      // Allow shorts but flag longs
    }

    // ===== STEP 2: NEWS FILTER =====
    const newsCheck = await this.checkNewsFilter(sym);
    if (newsCheck.blocked) {
      return { decision: "NO_TRADE", symbol: sym, reason: `News blackout: ${newsCheck.reason}`, source: "news_filter" };
    }

    // ===== STEP 3: MARKET ENGINE (SMC+CRT analysis from index.js) =====
    const analysis = await this.analyzeSymbol(sym, interval);
    if (!analysis || analysis.error || analysis.direction === "NEUTRAL") {
      return { decision: "NO_TRADE", symbol: sym, reason: analysis?.error || "No clear directional edge", analysis };
    }

    if ((crashMetrics.phase === "crash" || crashMetrics.phase === "trigger") && analysis.direction === "BULLISH") {
      return { decision: "NO_TRADE", symbol: sym, reason: `Long blocked during crash ${crashMetrics.phase} phase`, crash_phase: crashMetrics.phase, analysis };
    }

    // ===== STEP 4: GSRI RISK GATE =====
    let gsriMode = "normal";
    let gsriScale = 1.0;
    let finalConfidence = analysis.confidence || 50;
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
      gsriMode = "normal";
    }

    if (gsriMode === "blocked" && crashMetrics.phase !== "trigger") {
      return { decision: "NO_TRADE", symbol: sym, reason: "GSRI high risk environment (non-crash)", gsri_mode: gsriMode };
    }

    // ===== STEP 5: CONFIDENCE THRESHOLD =====
    if (finalConfidence < 30) {
      return { decision: "NO_TRADE", symbol: sym, reason: `Confidence too low (${finalConfidence}%)`, engine_confidence: analysis.confidence };
    }

    // ===== STEP 6: EXECUTION =====
    const direction = analysis.direction === "BULLISH" ? "BUY" : "SELL";
    const price = analysis.price;
    const entry = parseFloat(analysis.trade_plan?.entry_zone?.split("-")[0]) || price;
    const sl = parseFloat(analysis.trade_plan?.invalidation) || 0;
    const tp = parseFloat(analysis.trade_plan?.tp1) || 0;
    const lotSize = this.calculateLotSize
      ? this.calculateLotSize({ symbol: sym, balance: options.balance || 1000, riskPercent: options.riskPercent || 1, entry, stopLoss: sl }) * gsriScale
      : 0.01;

    // Use SMC+CRT entry if available
    const smcEntry = analysis.smc_crt?.entry;
    const finalEntry = smcEntry ? smcEntry.price : entry;
    const finalSL = smcEntry ? smcEntry.invalidation : sl;
    const finalTP1 = smcEntry ? smcEntry.tp1 : tp;

    const decision = {
      decision: direction,
      source: "smc_crt_pipeline",
      symbol: sym,
      confidence: finalConfidence,
      entry: finalEntry,
      sl: finalSL.toFixed(5),
      tp: finalTP1.toFixed(5),
      lot_size: Math.round(lotSize * 100) / 100,
      crash_phase: crashMetrics.phase,
      crash_risk_score: crashMetrics.risk_score,
      gsri_mode: gsriMode,
      gsri_scale: gsriScale,
      engine_confidence: analysis.confidence,
      direction_raw: analysis.direction,
      strength: analysis.strength,
      smc_crt: analysis.smc_crt,
      analysis,
      pipeline: "smc_crt_pipeline",
      timestamp: Date.now(),
      elapsed_ms: Date.now() - startTime,
    };

    this.paperLogger.log({ symbol: sym, direction, entry: finalEntry, sl: finalSL, tp: finalTP1, confidence: finalConfidence, lot_size: Math.round(lotSize * 100) / 100, source: "smc_crt_pipeline", crash_phase: crashMetrics.phase });

    if (this.addTradeMemory) {
      this.addTradeMemory(sym, {
        direction,
        pattern: analysis.patterns?.join(",") || "none",
        outcome: "pending",
        note: `SMC+CRT: conf=${finalConfidence} sig=${analysis.smc_crt?.signal} crash=${crashMetrics.phase}`,
      });
    }

    this.scanner.markTriggered(sym);
    this.pipelineHistory.push(decision);
    if (this.pipelineHistory.length > this.maxHistory) this.pipelineHistory.shift();

    return decision;
  }

  async scan(interval = "1h") {
    return this.scanner.scanAll(this.fetchCandles, interval);
  }

  getHistory(limit = 20) {
    return this.pipelineHistory.slice(-limit);
  }
}

// ============================================================================
// SCANNER LOOP (Autonomous Operation)
// ============================================================================

class ScannerLoop {
  constructor(pipeline) {
    this.pipeline = pipeline;
    this.running = false;
    this.intervalId = null;
    this.scanIntervalMs = 2 * 60 * 1000;
    this.lastScanTime = 0;
    this.scanCount = 0;
    this.triggersGenerated = 0;
    this.resolveIntervalId = null;
    this.resolveIntervalMs = 5 * 60 * 1000;
  }

  start(intervalMs) {
    if (this.running) return { status: "already_running" };
    this.scanIntervalMs = intervalMs || this.scanIntervalMs;
    this.running = true;
    this.intervalId = setInterval(async () => { await this.runScan(); }, this.scanIntervalMs);
    this.resolveIntervalId = setInterval(async () => { await this.autoResolveTrades(); }, this.resolveIntervalMs);
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
      const scanResult = await this.pipeline.scan("1h");

      // Run pipeline on triggered pairs
      if (scanResult.trigger && scanResult.trigger_score >= 2) {
        const pairs = scanResult.trigger.pairs || [];
        for (const sym of pairs) {
          try {
            const decision = await this.pipeline.run(sym, "1h");
            if (decision.decision !== "NO_TRADE" && decision.decision !== "COOLDOWN") {
              this.triggersGenerated++;
              console.log(`[SMC_CRT Scanner] SIGNAL: ${decision.decision} ${sym} conf=${decision.confidence}% source=${decision.source}`);
            }
          } catch (e) {
            console.error(`[SMC_CRT Scanner] Pipeline error for ${sym}:`, e.message);
          }
        }
      }

      // Also check crash GSRI independently
      const crashMetrics = await this.pipeline.crashGSRI.compute(this.pipeline.fetchCandles);
      if (crashMetrics.phase === "trigger") {
        const crashCandidates = ["BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "XAUUSD"];
        for (const sym of crashCandidates) {
          if (!this.pipeline.scanner.isInCooldown(sym)) {
            try {
              const decision = await this.pipeline.run(sym, "1h");
              if (decision.decision !== "NO_TRADE" && decision.decision !== "COOLDOWN") {
                this.triggersGenerated++;
                console.log(`[Crash GSRI] SIGNAL: ${decision.decision} ${sym} crash=${crashMetrics.phase}`);
              }
            } catch (e) {
              console.error(`[Crash GSRI] Pipeline error for ${sym}:`, e.message);
            }
          }
        }
      }
    } catch (e) {
      console.error("[ScannerLoop] Error:", e.message);
    }
  }

  async autoResolveTrades() {
    try {
      const pending = this.pipeline.paperLogger.getPendingTrades();
      if (pending.length === 0) return;
      const symbols = [...new Set(pending.map(t => t.symbol))];
      const prices = {};
      await Promise.all(symbols.map(async sym => {
        try {
          const spot = await this.pipeline.fetchCandles(sym, "1min", 1);
          if (spot && spot.length > 0) prices[sym] = spot[0].close;
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
    this.crashGSRI = this.pipeline.crashGSRI;
    this.scanner = this.pipeline.scanner;
    this.paperLogger = this.pipeline.paperLogger;
  }

  async analyze(symbol, interval = "1h", options = {}) {
    return this.pipeline.run(symbol, interval, options);
  }

  async scan(interval = "1h") {
    return this.pipeline.scan(interval);
  }

  startScanner(intervalMs) {
    return this.scannerLoop.start(intervalMs);
  }

  stopScanner() {
    return this.scannerLoop.stop();
  }

  getStatus() {
    return {
      scanner_loop: this.scannerLoop.getStatus(),
      crash_gsri: {
        last_phase: this.crashGSRI.lastResult?.phase || "unknown",
        last_risk_score: this.crashGSRI.lastResult?.risk_score || 0,
        last_compute: this.crashGSRI.lastComputeTime,
        history_size: this.crashGSRI.history.length,
      },
      paper_trades: {
        total_symbols: this.paperLogger.trades.size,
        pending: this.paperLogger.getPendingTrades().length,
      },
      pipeline_history: this.pipeline.pipelineHistory.length,
    };
  }
}
