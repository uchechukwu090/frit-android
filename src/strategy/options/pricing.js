// ============================================================================
// FRIT OPTIONS PRICING — pure-JS Black-Scholes / Greeks / IV solver
// ----------------------------------------------------------------------------
// Zero dependencies. European exercise, continuous dividend yield q, ACT/365
// year fractions supplied by callers. All rates/vols as decimals.
// Conventions:
//   - theta is per-CALENDAR-day decay (divide annual by 365)
//   - vega is per 1 VOL POINT move (e.g. 20% -> 21%), not per 1.0 vol
//   - impliedVol returns null (never throws) when the price violates
//     no-arbitrage bounds or expiry has passed — a missing IV beats a fake one.
// ============================================================================

const SQRT_2PI = Math.sqrt(2 * Math.PI);

export function normPdf(x) {
  return Math.exp(-0.5 * x * x) / SQRT_2PI;
}

// Abramowitz & Stegun 7.1.26 — max error 7.5e-8, plenty for signal work.
export function normCdf(x) {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return 0.5 * (1 + sign * y);
}

function d1d2({ S, K, T, r, q, sigma }) {
  const vsqrt = sigma * Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vsqrt;
  return [d1, d1 - vsqrt];
}

function validInputs({ S, K, T, sigma }) {
  return S > 0 && K > 0 && T > 0 && sigma > 0 &&
    [S, K, T, sigma].every(Number.isFinite);
}

/** European call/put value. Null on invalid inputs. */
export function bsPrice({ S, K, T, r = 0, q = 0, sigma, type }) {
  if (!validInputs({ S, K, T, sigma })) return null;
  if (type !== "call" && type !== "put") return null;
  const [d1, d2] = d1d2({ S, K, T, r, q, sigma });
  const discQ = Math.exp(-q * T), discR = Math.exp(-r * T);
  if (type === "call") return S * discQ * normCdf(d1) - K * discR * normCdf(d2);
  return K * discR * normCdf(-d2) - S * discQ * normCdf(-d1);
}

/** Delta/gamma/theta(per day)/vega(per vol point)/rho(per 1% rates). Null on invalid inputs. */
export function greeks({ S, K, T, r = 0, q = 0, sigma, type }) {
  if (!validInputs({ S, K, T, sigma })) return null;
  if (type !== "call" && type !== "put") return null;
  const [d1, d2] = d1d2({ S, K, T, r, q, sigma });
  const discQ = Math.exp(-q * T), discR = Math.exp(-r * T);
  const pdf = normPdf(d1);
  const delta = type === "call" ? discQ * normCdf(d1) : discQ * (normCdf(d1) - 1);
  const gamma = (discQ * pdf) / (S * sigma * Math.sqrt(T));
  const thetaAnnual = -(S * discQ * pdf * sigma) / (2 * Math.sqrt(T))
    + (type === "call" ? 1 : -1) * (q * S * discQ * normCdf(type === "call" ? d1 : -d1)
      - r * K * discR * normCdf(type === "call" ? d2 : -d2));
  // Note the sign convention: for puts the carry term flips via the (2*flag-1) form above.
  const vega = (S * discQ * pdf * Math.sqrt(T)) / 100; // per vol point
  const rho = (type === "call" ? K * T * discR * normCdf(d2) : -K * T * discR * normCdf(-d2)) / 100;
  return { delta, gamma, theta: thetaAnnual / 365, vega, rho };
}

/**
 * Implied volatility by bisection (60 iters, ~1e-12 price tolerance).
 * Returns null when the quote violates static arbitrage bounds — i.e. the
 * quote is stale/bad — instead of returning a garbage vol.
 */
export function impliedVol({ price, S, K, T, r = 0, q = 0, type }) {
  if (!(price >= 0) || !(S > 0) || !(K > 0) || !(T > 0)) return null;
  const discQ = Math.exp(-q * T), discR = Math.exp(-r * T);
  const fwd = S * discQ;
  const intrinsic = type === "call"
    ? Math.max(fwd - K * discR, 0)
    : Math.max(K * discR - fwd, 0);
  // No-arbitrage ceiling: option can never exceed the forward asset (call) or
  // discounted strike (put).
  const ceiling = type === "call" ? fwd : K * discR;
  if (price < intrinsic - 1e-12 || price > ceiling + 1e-12) return null;
  if (price <= intrinsic + 1e-12) return 0; // pinned at intrinsic — no time value
  let lo = 1e-6, hi = 5.0;
  const px = (s) => bsPrice({ S, K, T, r, q, sigma: s, type });
  if (px(hi) < price) return null; // even 500% vol cannot reach the quote
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    const p = px(mid);
    if (p == null) return null;
    if (Math.abs(p - price) < 1e-12) return mid;
    if (p < price) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/**
 * Annualized realized vol from log returns (close-to-close). Returns null on
 * fewer than 2 prices. `periodsPerYear`: 252 (equity) or 365*24*2 for 30M FX.
 */
export function realizedVol(closes, periodsPerYear = 252) {
  if (!Array.isArray(closes) || closes.length < 2) return null;
  const rets = [];
  for (let i = 1; i < closes.length; i++) {
    if (!(closes[i] > 0) || !(closes[i - 1] > 0)) return null;
    rets.push(Math.log(closes[i] / closes[i - 1]));
  }
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (rets.length - 1 || 1);
  return Math.sqrt(variance * periodsPerYear);
}

/** IV rank 0-100 of `current` against a history array. Null on empty history. */
export function ivRank(current, history) {
  if (!Array.isArray(history) || history.length === 0 || !Number.isFinite(current)) return null;
  const lo = Math.min(...history), hi = Math.max(...history);
  if (hi <= lo) return 50;
  return Math.max(0, Math.min(100, ((current - lo) / (hi - lo)) * 100));
}

/** 1-day expected move from an annualized vol: S * sigma * sqrt(1/365). */
export function expectedMove(S, sigmaAnnual, days = 1) {
  if (!(S > 0) || !(sigmaAnnual > 0) || !(days > 0)) return null;
  return S * sigmaAnnual * Math.sqrt(days / 365);
}
