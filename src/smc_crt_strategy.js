/**
 * SMC+CRT — Smart Money Concepts + Candle Range Theory (merged)
 *
 * Candle Range Theory (CRT):
 *   - A higher-timeframe candle defines a range (CRT-High / CRT-Low)
 *   - Price sweeps one side of the range (liquidity grab)
 *   - The sweep candle CLOSES BACK INSIDE the range (confirms trap)
 *   - Market Structure Shift (MSS) on lower TF confirms direction
 *   - Entry on retest of FVG or Order Block from the MSS candle
 *   - Take Profit 1 (T1) = opposite side of CRT range
 *   - Take Profit 2 (T2) = next external liquidity level
 *
 * SMC Components:
 *   - Market Structure (HH/HL uptrend, LH/LL downtrend, BOS, CHoCH)
 *   - Order Blocks (OB) — last opposing candle before impulsive move
 *   - Fair Value Gaps (FVG) — 3-candle imbalance
 *   - Liquidity Sweeps — false breakout at swing points
 *   - CHoCH (Change of Character) — trend reversal signal
 *
 * Volume Profile (POC/VAH/VAL) acts as institutional confluence.
 */

// ──────────────────────────────────────────────
// 1. Market Structure — HH/HL (uptrend), LH/LL (downtrend)
// ──────────────────────────────────────────────
export function analyzeMarketStructure(swings) {
  const { highs, lows } = swings;
  if (highs.length < 2 && lows.length < 2) {
    return { trend: "ranging", hh: false, hl: false, lh: false, ll: false };
  }

  const recentHighs = highs.slice(-3);
  const recentLows  = lows.slice(-3);

  const hh = recentHighs.length >= 2 && recentHighs.at(-1).price > recentHighs.at(-2).price;
  const hl = recentLows.length >= 2  && recentLows.at(-1).price > recentLows.at(-2).price;
  const lh = recentHighs.length >= 2 && recentHighs.at(-1).price < recentHighs.at(-2).price;
  const ll = recentLows.length >= 2  && recentLows.at(-1).price < recentLows.at(-2).price;

  let trend = "ranging";
  if (hh && hl) trend = "uptrend";
  else if (lh && ll) trend = "downtrend";
  else if (hh && !hl) trend = "potential_reversal_up";
  else if (ll && !lh) trend = "potential_reversal_down";

  return { trend, hh, hl, lh, ll };
}

// ──────────────────────────────────────────────
// 2. Order Blocks — last opposing candle before a strong move
// ──────────────────────────────────────────────
export function findOrderBlocks(candles) {
  if (!candles || candles.length < 10) return { bullish: [], bearish: [] };

  const bullishOBs = [];
  const bearishOBs = [];
  const lookback = Math.min(candles.length, 60);

  for (let i = 3; i < lookback - 1; i++) {
    const prev = candles[i - 1];
    const cur  = candles[i];
    const next = candles[i + 1];

    const prevBear = prev.close < prev.open;
    const curBull  = cur.close > cur.open;
    const bodyRatio = Math.abs(cur.close - cur.open) / (cur.high - cur.low || 1);

    if (prevBear && curBull && bodyRatio > 0.5) {
      bullishOBs.push({
        type: "bullish",
        price: Math.min(prev.open, prev.close),
        high: prev.high, low: prev.low,
        time: prev.time, strength: bodyRatio,
        mitigated: false,
      });
    }

    const prevBull = prev.close > prev.open;
    const curBear  = cur.close < cur.open;

    if (prevBull && curBear && bodyRatio > 0.5) {
      bearishOBs.push({
        type: "bearish",
        price: Math.max(prev.open, prev.close),
        high: prev.high, low: prev.low,
        time: prev.time, strength: bodyRatio,
        mitigated: false,
      });
    }
  }

  return { bullish: bullishOBs.slice(-5), bearish: bearishOBs.slice(-5) };
}

// ──────────────────────────────────────────────
// 3. Fair Value Gaps (FVG) — 3-candle imbalance
// ──────────────────────────────────────────────
export function findFVGs(candles) {
  if (!candles || candles.length < 5) return { bullish: [], bearish: [] };

  const fvgs = [];
  const lookback = Math.min(candles.length, 80);

  for (let i = 2; i < lookback; i++) {
    const c1 = candles[i - 2];
    const c2 = candles[i - 1];
    const c3 = candles[i];

    if (c1.high < c3.low) {
      fvgs.push({
        type: "bullish",
        gap_high: c1.high, gap_low: c3.low,
        midpoint: (c1.high + c3.low) / 2,
        time: c2.time, filled: false,
      });
    }

    if (c1.low > c3.high) {
      fvgs.push({
        type: "bearish",
        gap_high: c1.low, gap_low: c3.high,
        midpoint: (c1.low + c3.high) / 2,
        time: c2.time, filled: false,
      });
    }
  }

  const bullish = fvgs.filter(f => f.type === "bullish").slice(-3);
  const bearish = fvgs.filter(f => f.type === "bearish").slice(-3);
  return { bullish, bearish };
}

// ──────────────────────────────────────────────
// 4. Market Structure Shift (MSS) + CHoCH
// ──────────────────────────────────────────────
export function detectCHoCH(swings) {
  if (swings.highs.length < 3 || swings.lows.length < 3) {
    return { choch: null, mss: null };
  }

  const recentHighs = swings.highs.slice(-3);
  const recentLows  = swings.lows.slice(-3);

  const pLow = recentLows.at(-2);
  const lLow = recentLows.at(-1);
  const pHigh = recentHighs.at(-2);
  const lHigh = recentHighs.at(-1);

  let choch = null;
  let mss = null;

  if (pLow && lLow && pHigh && lHigh) {
    // Bearish CHoCH: made HH then broke prior swing low
    if (lHigh?.price > pHigh?.price && lLow?.price < pLow?.price) {
      choch = { direction: "bearish", trigger_level: pLow.price, last_swing_high: lHigh.price };
    }
    // Bullish CHoCH: made LL then broke prior swing high
    if (lLow?.price < pLow?.price && lHigh?.price > pHigh?.price) {
      choch = { direction: "bullish", trigger_level: pHigh.price, last_swing_low: lLow.price };
    }
  }

  // MSS = the break that confirms the shift (more precise than CHoCH)
  if (choch) {
    mss = {
      direction: choch.direction,
      level: choch.trigger_level,
      description: `MSS ${choch.direction} @ ${choch.trigger_level}`,
    };
  }

  return { choch, mss };
}

// ──────────────────────────────────────────────
// 5. Liquidity Sweeps — false breakouts at swing points
// ──────────────────────────────────────────────
export function detectLiquiditySweeps(candles, swings) {
  if (!candles || candles.length < 10) return { sweeps: [] };

  const recent = candles.slice(-15);
  const sweeps = [];

  const recentHighs = swings.highs.slice(-3);
  for (const sh of recentHighs) {
    const breakCandle = recent.find(c => c.high > sh.price && c.close < sh.price);
    if (breakCandle) {
      sweeps.push({ type: "bearish_sweep", level: sh.price, time: breakCandle.time });
    }
  }

  const recentLows = swings.lows.slice(-3);
  for (const sl of recentLows) {
    const breakCandle = recent.find(c => c.low < sl.price && c.close > sl.price);
    if (breakCandle) {
      sweeps.push({ type: "bullish_sweep", level: sl.price, time: breakCandle.time });
    }
  }

  return { sweeps };
}

// ──────────────────────────────────────────────
// 6. Candle Range Theory (CRT) Detection
// ──────────────────────────────────────────────
export function detectCRT(candles) {
  if (!candles || candles.length < 5) return null;

  const recent = candles.slice(-10);
  const results = [];

  for (let i = 2; i < recent.length - 1; i++) {
    const anchor = recent[i];           // The range-defining candle
    const sweep  = recent[i + 1];       // The sweep candle
    const crtHigh = Math.max(anchor.high, anchor.open, anchor.close);
    const crtLow  = Math.min(anchor.low, anchor.open, anchor.close);

    // Must close back inside the range
    const sweptHigh = sweep.high > crtHigh;
    const sweptLow  = sweep.low < crtLow;
    const closedInside = sweep.close < crtHigh && sweep.close > crtLow;

    if (!closedInside) continue;

    if (sweptHigh) {
      // Bearish CRT: swept above CRT-High, closed back inside
      results.push({
        type: "bearish_crt",
        anchor_time: anchor.time,
        sweep_time: sweep.time,
        crt_high: crtHigh,
        crt_low: crtLow,
        sweep_high: sweep.high,
        sweep_close: sweep.close,
        range_size: crtHigh - crtLow,
        t1: crtLow,           // Target opposite side
        t2: null,             // External liquidity (set by caller)
        description: `Bearish CRT: swept above ${crtHigh} → closed @ ${sweep.close} inside range`,
      });
    }

    if (sweptLow) {
      // Bullish CRT: swept below CRT-Low, closed back inside
      results.push({
        type: "bullish_crt",
        anchor_time: anchor.time,
        sweep_time: sweep.time,
        crt_high: crtHigh,
        crt_low: crtLow,
        sweep_low: sweep.low,
        sweep_close: sweep.close,
        range_size: crtHigh - crtLow,
        t1: crtHigh,          // Target opposite side
        t2: null,             // External liquidity (set by caller)
        description: `Bullish CRT: swept below ${crtLow} → closed @ ${sweep.close} inside range`,
      });
    }
  }

  // Return the most recent CRT setup
  return results.length > 0 ? results.at(-1) : null;
}

// ──────────────────────────────────────────────
// 7. CRT + SMC Entry Logic
// ──────────────────────────────────────────────
export function findCRTEntry(crtSetup, candles, swings, price, atr) {
  if (!crtSetup) return null;

  const { type, t1, crt_high, crt_low } = crtSetup;
  const isBullish = type === "bullish_crt";
  const isBearish = type === "bearish_crt";

  // Look for an MSS or OB/FVG near the current price after the sweep
  const recentCandles = candles.slice(-8);
  const fvgs = findFVGs(candles);
  const orderBlocks = findOrderBlocks(candles);
  const { choch, mss } = detectCHoCH(swings);

  // Check if a CHoCH/MSS confirms the CRT direction
  let mssAligned = false;
  if (isBullish && choch?.direction === "bullish") mssAligned = true;
  if (isBearish && choch?.direction === "bearish") mssAligned = true;

  // Find nearest FVG or OB that aligns with CRT direction
  let entryPrice = null;
  let invalidation = null;
  let entryReason = "";
  const tolerance = atr * 0.5;

  if (isBullish) {
    // Buy: look for bullish FVG or OB near price
    for (const fvg of fvgs.bullish) {
      if (Math.abs(price - fvg.midpoint) <= tolerance) {
        entryPrice = fvg.midpoint;
        invalidation = fvg.gap_low - atr * 0.3;
        entryReason = `Bullish FVG retest @ ${fvg.midpoint.toFixed(5)}`;
        break;
      }
    }
    if (!entryPrice) {
      for (const ob of orderBlocks.bullish) {
        if (Math.abs(price - ob.price) <= tolerance) {
          entryPrice = ob.price;
          invalidation = ob.low - atr * 0.3;
          entryReason = `Bullish OB retest @ ${ob.price.toFixed(5)}`;
          break;
        }
      }
    }
    // Default entry: retrace to CRT-Low + buffer
    if (!entryPrice) {
      entryPrice = crt_low + atr * 0.15;
      invalidation = crt_low - atr * 0.5;
      entryReason = `CRT buy: retest of CRT-Low @ ${crt_low.toFixed(5)}`;
    }
    // TP targets
    const tp1 = t1; // Opposite side of CRT range
    const tp2 = swings.highs.at(-1)?.price > tp1
      ? swings.highs.at(-1).price
      : t1 + atr * 1.0;

    return {
      type: "crt_buy",
      entry: entryPrice,
      invalidation,
      tp1,
      tp2,
      mss_confirmed: mssAligned,
      reason: entryReason,
      description: `CRT Buy: ${entryReason} | TP1=${tp1.toFixed(5)} TP2=${tp2.toFixed(5)}`,
    };
  }

  if (isBearish) {
    for (const fvg of fvgs.bearish) {
      if (Math.abs(price - fvg.midpoint) <= tolerance) {
        entryPrice = fvg.midpoint;
        invalidation = fvg.gap_high + atr * 0.3;
        entryReason = `Bearish FVG retest @ ${fvg.midpoint.toFixed(5)}`;
        break;
      }
    }
    if (!entryPrice) {
      for (const ob of orderBlocks.bearish) {
        if (Math.abs(price - ob.price) <= tolerance) {
          entryPrice = ob.price;
          invalidation = ob.high + atr * 0.3;
          entryReason = `Bearish OB retest @ ${ob.price.toFixed(5)}`;
          break;
        }
      }
    }
    if (!entryPrice) {
      entryPrice = crt_high - atr * 0.15;
      invalidation = crt_high + atr * 0.5;
      entryReason = `CRT sell: retest of CRT-High @ ${crt_high.toFixed(5)}`;
    }
    const tp1 = t1;
    const tp2 = swings.lows.at(-1)?.price < tp1
      ? swings.lows.at(-1).price
      : t1 - atr * 1.0;

    return {
      type: "crt_sell",
      entry: entryPrice,
      invalidation,
      tp1,
      tp2,
      mss_confirmed: mssAligned,
      reason: entryReason,
      description: `CRT Sell: ${entryReason} | TP1=${tp1.toFixed(5)} TP2=${tp2.toFixed(5)}`,
    };
  }

  return null;
}

// ──────────────────────────────────────────────
// 8. SMC-Only Setup (when no CRT detected)
// ──────────────────────────────────────────────
export function findSMCEntry(candles, swings, price, atr) {
  const orderBlocks = findOrderBlocks(candles);
  const fvgs = findFVGs(candles);
  const structure = analyzeMarketStructure(swings);
  const { choch, mss } = detectCHoCH(swings);
  const { sweeps } = detectLiquiditySweeps(candles, swings);

  const tolerance = atr * 0.4;
  const isUptrend = structure.trend === "uptrend";
  const isDowntrend = structure.trend === "downtrend";
  const hasBullSweep = sweeps.some(s => s.type === "bullish_sweep");
  const hasBearSweep = sweeps.some(s => s.type === "bearish_sweep");

  // === BUY SETUP ===
  // Conditions: uptrend + sweep of sell-side liquidity + CHoCH/MSS bullish + OB/FVG retest
  if (isUptrend || structure.trend === "potential_reversal_up") {
    const bullishConfluences = [];
    if (isUptrend) bullishConfluences.push("uptrend");
    if (hasBullSweep) bullishConfluences.push("liquidity_sweep");
    if (choch?.direction === "bullish") bullishConfluences.push("choch_bullish");

    // Find entry
    if (bullishConfluences.length >= 1) {
      for (const fvg of fvgs.bullish) {
        if (Math.abs(price - fvg.midpoint) <= tolerance) {
          const tp1 = swings.highs.at(-1)?.price || price + atr * 1.2;
          return {
            type: "smc_buy",
            entry: fvg.midpoint,
            invalidation: fvg.gap_low - atr * 0.3,
            tp1, tp2: tp1 + atr * 0.8,
            confluences: bullishConfluences,
            reason: `SMC Buy: FVG @ ${fvg.midpoint.toFixed(5)} + ${bullishConfluences.join(" + ")}`,
          };
        }
      }
      for (const ob of orderBlocks.bullish) {
        if (Math.abs(price - ob.price) <= tolerance) {
          const tp1 = swings.highs.at(-1)?.price || price + atr * 1.2;
          return {
            type: "smc_buy",
            entry: ob.price,
            invalidation: ob.low - atr * 0.3,
            tp1, tp2: tp1 + atr * 0.8,
            confluences: bullishConfluences,
            reason: `SMC Buy: OB @ ${ob.price.toFixed(5)} + ${bullishConfluences.join(" + ")}`,
          };
        }
      }
    }
  }

  // === SELL SETUP ===
  if (isDowntrend || structure.trend === "potential_reversal_down") {
    const bearishConfluences = [];
    if (isDowntrend) bearishConfluences.push("downtrend");
    if (hasBearSweep) bearishConfluences.push("liquidity_sweep");
    if (choch?.direction === "bearish") bearishConfluences.push("choch_bearish");

    if (bearishConfluences.length >= 1) {
      for (const fvg of fvgs.bearish) {
        if (Math.abs(price - fvg.midpoint) <= tolerance) {
          const tp1 = swings.lows.at(-1)?.price || price - atr * 1.2;
          return {
            type: "smc_sell",
            entry: fvg.midpoint,
            invalidation: fvg.gap_high + atr * 0.3,
            tp1, tp2: tp1 - atr * 0.8,
            confluences: bearishConfluences,
            reason: `SMC Sell: FVG @ ${fvg.midpoint.toFixed(5)} + ${bearishConfluences.join(" + ")}`,
          };
        }
      }
      for (const ob of orderBlocks.bearish) {
        if (Math.abs(price - ob.price) <= tolerance) {
          const tp1 = swings.lows.at(-1)?.price || price - atr * 1.2;
          return {
            type: "smc_sell",
            entry: ob.price,
            invalidation: ob.high + atr * 0.3,
            tp1, tp2: tp1 - atr * 0.8,
            confluences: bearishConfluences,
            reason: `SMC Sell: OB @ ${ob.price.toFixed(5)} + ${bearishConfluences.join(" + ")}`,
          };
        }
      }
    }
  }

  return null;
}

// ──────────────────────────────────────────────
// 9. Volume Profile Confluence Scoring
// ──────────────────────────────────────────────
export function scoreVolumeProfileConfluence(price, auction, entryType) {
  if (!auction) return { score: 0, note: "No volume profile data" };

  const { poc, vah, val } = auction;
  let score = 0;
  const notes = [];

  // POC = highest volume node = strongest institutional level
  const distToPOC = Math.abs(price - poc);
  const range = vah - val || 1;
  const pctFromPOC = distToPOC / range;

  if (pctFromPOC < 0.1) {
    score += 2;
    notes.push("Price at POC — institutional equilibrium");
  } else if (pctFromPOC < 0.25) {
    score += 1;
    notes.push("Price near POC");
  }

  // VAH = value area high (resistance)
  if (entryType?.includes("buy")) {
    if (price < val) {
      score += 2;
      notes.push("Price below VAL — discount zone for buys");
    } else if (price < poc) {
      score += 1;
      notes.push("Price below POC — mild discount");
    }
  }

  if (entryType?.includes("sell")) {
    if (price > vah) {
      score += 2;
      notes.push("Price above VAH — premium zone for sells");
    } else if (price > poc) {
      score += 1;
      notes.push("Price above POC — mild premium");
    }
  }

  // HVN/LVN confluences
  const hvnProx = auction.hvn?.some(h => Math.abs(price - h) / range < 0.05);
  if (hvnProx) {
    score += 1;
    notes.push("Near HVN — strong institutional level");
  }

  return { score: Math.min(score, 5), note: notes.join("; ") || "Neutral volume profile" };
}

// ──────────────────────────────────────────────
// 10. MAIN: Merged SMC+CRT Analysis
// ──────────────────────────────────────────────
export function analyzeSMC_CRT(candles, price, atr, auction) {
  if (!candles || candles.length < 20) {
    return { signal: "insufficient_data", confidence: 0 };
  }

  const swings = findSwingsLocal(candles);
  const structure = analyzeMarketStructure(swings);
  const orderBlocks = findOrderBlocks(candles);
  const fvgs = findFVGs(candles);
  const { sweeps } = detectLiquiditySweeps(candles, swings);
  const { choch, mss } = detectCHoCH(swings);
  const crtSetup = detectCRT(candles);
  const crtEntry = crtSetup ? findCRTEntry(crtSetup, candles, swings, price, atr) : null;
  const smcEntry = crtEntry ? null : findSMCEntry(candles, swings, price, atr);
  const vpScore = scoreVolumeProfileConfluence(price, auction, crtEntry?.type || smcEntry?.type);

  const entry = crtEntry || smcEntry;
  const reasons = [];
  let signal = "neutral";
  let confidence = 0;

  if (entry) {
    const isBuy = entry.type?.includes("buy");
    const isSell = entry.type?.includes("sell");
    signal = isBuy ? "buy" : "sell";
    confidence = 50; // base

    // CRT setups are higher confidence
    if (crtEntry) {
      confidence += 15; // CRT premium
      reasons.push(`CRT setup: ${crtEntry.description}`);
      if (crtEntry.mss_confirmed) {
        confidence += 10;
        reasons.push("MSS confirms CRT direction");
      }
    }

    // SMC confluences
    if (smcEntry?.confluences) {
      const condScore = smcEntry.confluences.length * 8;
      confidence += condScore;
      reasons.push(`SMC confluences: ${smcEntry.confluences.join(", ")}`);
    }

    // Structure alignment
    if (structure.trend === "uptrend" && isBuy) { confidence += 10; reasons.push("Structure aligns: uptrend"); }
    else if (structure.trend === "downtrend" && isSell) { confidence += 10; reasons.push("Structure aligns: downtrend"); }
    else if ((structure.trend === "uptrend" && isSell) || (structure.trend === "downtrend" && isBuy)) {
      confidence -= 20;
      reasons.push("WARNING: trading against structure");
    }

    // CHoCH bonus
    if (choch) {
      const chochAligns = (choch.direction === "bullish" && isBuy) || (choch.direction === "bearish" && isSell);
      if (chochAligns) { confidence += 15; reasons.push(`CHoCH confirms: ${choch.direction}`); }
    }

    // Volume Profile confluence
    confidence += vpScore.score * 3;
    if (vpScore.score > 0) reasons.push(vpScore.note);

    // Liquidity sweep bonus
    const sweepAligns = sweeps.some(s =>
      (s.type === "bullish_sweep" && isBuy) || (s.type === "bearish_sweep" && isSell)
    );
    if (sweepAligns) { confidence += 8; reasons.push("Liquidity sweep aligns"); }
  } else {
    // No entry found — report what we see
    if (choch) reasons.push(`CHoCH ${choch.direction} detected but no entry aligned`);
    if (crtSetup) reasons.push("CRT range found but no entry trigger yet");
    if (structure.trend !== "ranging") reasons.push(`Market in ${structure.trend} — waiting for entry`);
    if (sweeps.length > 0) reasons.push("Liquidity sweeps present — watching for reversal");
  }

  confidence = Math.max(0, Math.min(99, Math.round(confidence)));

  return {
    signal,
    confidence,
    structure,
    crt: crtSetup ? {
      type: crtSetup.type,
      crt_high: crtSetup.crt_high,
      crt_low: crtSetup.crt_low,
      t1: crtSetup.t1,
      description: crtSetup.description,
    } : null,
    order_blocks: {
      bullish: orderBlocks.bullish.map(o => ({ price: o.price, strength: o.strength })),
      bearish: orderBlocks.bearish.map(o => ({ price: o.price, strength: o.strength })),
    },
    fvgs: {
      bullish: fvgs.bullish.map(f => ({ gap_low: f.gap_low, gap_high: f.gap_high, midpoint: f.midpoint })),
      bearish: fvgs.bearish.map(f => ({ gap_low: f.gap_low, gap_high: f.gap_high, midpoint: f.midpoint })),
    },
    sweeps: sweeps.map(s => ({ type: s.type, level: s.level })),
    choch: choch ? { direction: choch.direction, trigger_level: choch.trigger_level } : null,
    volume_profile: { score: vpScore.score, note: vpScore.note },
    entry: entry ? {
      type: entry.type,
      price: entry.entry,
      invalidation: entry.invalidation,
      tp1: entry.tp1,
      tp2: entry.tp2,
      reason: entry.reason,
    } : null,
    last_swing_high: swings.highs.at(-1)?.price ?? null,
    last_swing_low: swings.lows.at(-1)?.price ?? null,
    reasons,
  };
}

// ──────────────────────────────────────────────
// Local swing detection
// ──────────────────────────────────────────────
function findSwingsLocal(candles, lookback = 2) {
  const highs = [];
  const lows = [];
  for (let i = lookback; i < candles.length - lookback; i++) {
    const cur = candles[i];
    let isHigh = true, isLow = true;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (candles[j].high >= cur.high) isHigh = false;
      if (candles[j].low  <= cur.low)  isLow  = false;
    }
    if (isHigh) highs.push({ index: i, price: cur.high, time: cur.time });
    if (isLow)  lows.push( { index: i, price: cur.low,  time: cur.time });
  }
  return { highs, lows };
}
