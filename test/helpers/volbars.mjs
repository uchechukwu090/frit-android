// ============================================================================
// Synthetic bars for the predictive-volume + signal-lifecycle tests.
// ----------------------------------------------------------------------------
// Deterministic (no randomness) so a failure always means the logic changed.
// Shape: a quiet baseline followed by an expanding leg that is ALIGNED with an
// uptrend — price +2 per 5m bar, range 8 (ATR ~8), so the leg is trending but
// not "extended" (>3.5 ATR from EMA20) and never reads as exhaustion.
//   5m baseline volume ~100 (median), ramp = 110..360 over the last 6 bars
//   1m baseline volume ~20,          ramp = 45,55,70 over the last 3 bars
// ============================================================================

export const MIN5 = 5 * 60_000;
export const MIN1 = 60_000;

const baseVol5 = (i) => 100 + ((i % 5) - 2) * 3; // median 100, never zero
const baseVol1 = (i) => 20 + ((i % 5) - 2); //     median 20,  never zero

/**
 * n CLOSED 5m bars ending two bars before `now`.
 * @param ramp    volume expansion leg on the last 6 bars
 * @param spike   one exhaustion-sized spike on the last bar (>= 7x baseline)
 */
export function bars5({ n = 150, now = Date.now(), ramp = true, spike = false } = {}) {
  const start = now - (n + 1) * MIN5;
  const leg = [110, 140, 180, 230, 290, 360];
  const out = [];
  for (let i = 0; i < n; i++) {
    const close = 2000 + 2 * i;
    const open = close - 2;
    let volume = baseVol5(i);
    if (ramp && i >= n - 6) volume = leg[i - (n - 6)];
    if (spike && i === n - 1) volume = 700;
    out.push({ time: start + i * MIN5, open, high: close + 3, low: open - 3, close, volume });
  }
  return out;
}

/** n CLOSED 1m bars: the pace leg inside the current 5m window. */
export function bars1({ n = 60, now = Date.now(), ramp = true } = {}) {
  const start = now - (n + 1) * MIN1;
  const leg = [45, 55, 70];
  const out = [];
  for (let i = 0; i < n; i++) {
    const close = 2280 + i * 0.3; // sits next to the 5m series' last print
    let volume = baseVol1(i);
    if (ramp && i >= n - 3) volume = leg[i - (n - 3)];
    out.push({ time: start + i * MIN1, open: close - 0.5, high: close + 1, low: close - 1.5, close, volume });
  }
  return out;
}
