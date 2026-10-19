// ============================================================================
// FRIT SIGNAL HYSTERESIS — stop the engine arguing with itself
// ----------------------------------------------------------------------------
// The observed failure (real one, not hypothetical): once the engine publishes
// a pullback entry in a trend's direction, a lagging EMAs flip mid-pullback
// makes the very next run emit the OPPOSITE trade. The user gets
// "wait to buy the dip" and then "sell now" four candles later — both from
// the same trend, neither confirmed by broken structure.
//
// Fix: signal hysteresis. A published bias (BUY / SELL / WAIT_PULLBACK in a
// direction) STAYS the bias until price actually breaks the structure that
// defined it:
//   - after a pullback entry, an opposite entry needs a close beyond the
//     pullback INVALIDATION level (78.6% retracement) — the leg genuinely died;
//   - after an immediate trend entry, an opposite entry needs the position's
//     own SL level to be broken — i.e. real evidence, not an EMAs twitch.
// Until then the counter signal is suppressed to WAIT with the reason stated.
// Same-direction re-signals pass through untouched (consistent, not noise).
//
// Pure functions on purpose: no state inside, so the caller owns storage and
// tests never need an engine instance.
// ============================================================================

const SIGNS = { BUY: 1, SELL: -1 };

/** Has price broken through the structural level that defined the bias? */
export function structureBroke(level, price, side) {
  if (level == null || !(price > 0)) return false;
  // Long bias dies when price trades below the level; short bias above it.
  return side === "BUY" ? price < level : price > level;
}

/**
 * @param prev  { side, level, decision, at } — last published bias (null = none)
 * @param next  engine result: { decision, price, ... }
 * @param priceNow current price for the break test
 * @returns { ok: boolean, suppressed?: boolean, reason?: string, state }
 */
export function applyHysteresis(prev, next, priceNow) {
  const decision = String(next?.decision || "WAIT");
  const sideOf = (d) => (d === "BUY" ? "BUY" : d === "SELL" ? "SELL" : null);

  // A pullback signal carries the trend's direction as its side: WAIT_PULLBACK
  // after a BULL regime means the bias is LONG (buy the dip). WAIT_RANGE is
  // direction-less fading — never a trend bias to defend.
  const biasSide = (r) => {
    const s = sideOf(r?.decision);
    if (s) return s;
    if (r?.decision === "WAIT_PULLBACK") {
      const dir = r?.scenarios?.scenario_2_pullback_crossover?.action;
      return sideOf(dir);
    }
    return null;
  };

  // No standing bias -> this result becomes the bias, published untouched.
  if (!prev || !prev.side) {
    const side = biasSide(next);
    if (!side) return { ok: true, suppressed: false, state: prev || null };
    return {
      ok: true, suppressed: false,
      state: { side, level: invalidationOf(next, side), decision, at: Date.now() },
    };
  }

  const nextSide = sideOf(decision);
  // Same direction or no direction -> consistent, keep the bias and publish.
  if (!nextSide || nextSide === prev.side) {
    return {
      ok: true, suppressed: false,
      state: { side: prev.side, level: invalidationLevel(prev, next, prev.side), decision, at: Date.now() },
    };
  }

  // Opposite direction: only allowed once structure actually broke.
  if (structureBroke(prev.level, priceNow, prev.side)) {
    return {
      ok: true, suppressed: false,
      state: { side: nextSide, level: invalidationOf(next, nextSide), decision, at: Date.now() },
      released: true,
    };
  }
  return {
    ok: false, suppressed: true,
    reason: `counter-${nextSide} suppressed: standing ${prev.side} bias intact (structure level ${prev.level ?? "n/a"} not broken). The pullback is INSIDE the trend until proven otherwise — no whipsaw.`,
    state: prev,
  };
}

/** The price level whose break invalidates this result's bias. */
function invalidationOf(result, side) {
  const sl = Number(result?.sl);
  const zone = result?.scenarios?.scenario_2_pullback_crossover;
  const zoneInv = zone?.invalidation != null ? Number(zone.invalidation) : null;
  if (zoneInv != null && Number.isFinite(zoneInv)) return zoneInv;
  if (Number.isFinite(sl) && sl > 0) return sl;
  return result?.structure_30m?.[side === "BUY" ? "last_swing_low" : "last_swing_high"] ?? null;
}

/** Keep the tighter (nearer) of the old and new structural levels. */
function invalidationLevel(prev, next, side) {
  const nextLevel = invalidationOf(next, side);
  if (prev.level == null) return nextLevel;
  if (nextLevel == null) return prev.level;
  if (side === "BUY") return Math.max(prev.level, nextLevel); // long: higher low
  return Math.min(prev.level, nextLevel);                    // short: lower high
}
