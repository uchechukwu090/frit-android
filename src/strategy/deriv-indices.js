// ============================================================================
// DERIV INDICES (synthetic) DIGIT BOT — ported strategy families + proof gate
// ----------------------------------------------------------------------------
// Strategy families come from toptraderke/freetradingbots (Even/Odd, Over/Under,
// Differs). The XML/Blockly code itself is NOT copied — only the ideas — and the
// martingale staking in those bots is deliberately NOT ported (see below).
//
// READ THIS FIRST
// - Deriv synthetic indices (R_100, 1HZ100V, ...) are driven by a random number
//   generator. Digit outcomes are independent, so every digit contract is a
//   negative-expectancy bet: the payout is set below fair odds (the house margin).
//   Martingale does not change that; it hides losses until a long streak wipes
//   the account. So this module ships with:
//     1. backtest()/validate()  — tests a strategy against REAL recorded ticks and
//        says whether its win rate beats the payout break-even with statistical
//        significance (z-test). Expect "NO EDGE"; that is the honest answer.
//     2. IndicesBot — DEMO ACCOUNTS ONLY (real-money tokens are refused, there is
//        no override flag), fixed stake, daily-loss / trade / losing-streak caps.
//        Use it to forward-test, not to make money.
// - Nothing here is wired into the runner automatically. deriv.js keeps refusing
//   synthetics in its own executor on purpose.
// ============================================================================

export const INDEX_SYMBOL = /^(R_\d+|1HZ\d+V)$/;

/** Approximate Deriv house margin on digit contracts (real quotes vary). */
export const DEFAULT_MARGIN = 0.03;

// ---------------------------------------------------------------- digits ----

/** Last digit of a quote. pip = number of decimals Deriv quotes for the symbol. */
export function lastDigit(price, pip = 2) {
  const scaled = Math.round(Number(price) * 10 ** pip);
  return ((scaled % 10) + 10) % 10;
}

export function toDigits(prices, pip = 2) {
  return prices.map((p) => lastDigit(p, pip));
}

// ------------------------------------------------------------- contracts ----

export function winProb(c) {
  const b = Number(c.barrier);
  switch (c.type) {
    case "DIGITEVEN": case "DIGITODD": return 0.5;
    case "DIGITOVER": return (9 - b) / 10;
    case "DIGITUNDER": return b / 10;
    case "DIGITDIFF": return 0.9;
    case "DIGITMATCH": return 0.1;
    default: throw new Error(`unknown contract ${c.type}`);
  }
}

export function digitWins(c, d) {
  const b = Number(c.barrier);
  switch (c.type) {
    case "DIGITEVEN": return d % 2 === 0;
    case "DIGITODD": return d % 2 === 1;
    case "DIGITOVER": return d > b;
    case "DIGITUNDER": return d < b;
    case "DIGITDIFF": return d !== b;
    case "DIGITMATCH": return d === b;
    default: return false;
  }
}

/** Total payout multiple (stake included) under the house-margin model. */
export function payoutFor(c, margin = DEFAULT_MARGIN) {
  return (1 - margin) / winProb(c);
}

// ------------------------------------------------------------ strategies ----
// Each takes digits (oldest -> newest) and returns a contract or null.

export const STRATEGIES = {
  /** Even/Odd: after N same-parity digits, bet the parity flips. */
  evenOddReversal(digits, { streak = 4 } = {}) {
    if (digits.length < streak) return null;
    const tail = digits.slice(-streak).map((d) => d % 2);
    if (!tail.every((p) => p === tail[0])) return null;
    return { type: tail[0] === 0 ? "DIGITODD" : "DIGITEVEN" };
  },

  /** Even/Odd: after N same-parity digits, bet the run continues. */
  evenOddContinuation(digits, { streak = 4 } = {}) {
    if (digits.length < streak) return null;
    const tail = digits.slice(-streak).map((d) => d % 2);
    if (!tail.every((p) => p === tail[0])) return null;
    return { type: tail[0] === 0 ? "DIGITEVEN" : "DIGITODD" };
  },

  /** Over/Under: follow a recent bias in the share of high digits (>4). */
  overUnderBias(digits, { window = 100, edge = 0.08 } = {}) {
    if (digits.length < window) return null;
    const share = digits.slice(-window).filter((d) => d > 4).length / window;
    if (share > 0.5 + edge) return { type: "DIGITOVER", barrier: 4 };
    if (share < 0.5 - edge) return { type: "DIGITUNDER", barrier: 5 };
    return null;
  },

  /** Differs: bet the next digit differs from the one just seen. */
  differsLast(digits) {
    if (!digits.length) return null;
    return { type: "DIGITDIFF", barrier: digits[digits.length - 1] };
  },
};

// --------------------------------------------------------------- statistics ----

function erfc(x) {
  // Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7), extended to negative x.
  const z = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * z);
  const y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const r = y * Math.exp(-z * z);
  return x >= 0 ? r : 2 - r;
}

/** One-sided p-value that the observed z is this high by chance. */
export function pOneSided(z) { return 0.5 * erfc(z / Math.SQRT2); }

/** Chi-square of digit frequencies vs uniform (df = 9; > 21.67 is p < 0.01). */
export function digitChi2(digits) {
  const counts = new Array(10).fill(0);
  for (const d of digits) counts[d]++;
  const e = digits.length / 10;
  return counts.reduce((s, c) => s + (c - e) ** 2 / e, 0);
}

/**
 * Walk-forward backtest on a digit series. Each signal is a 1-tick contract
 * settled by the NEXT digit, so there is no look-ahead.
 * Null hypothesis: win rate equals the payout break-even (no edge).
 */
export function backtest(digits, strategy, opts = {}, { margin = DEFAULT_MARGIN, warmup = 120 } = {}) {
  const fn = typeof strategy === "function" ? strategy : STRATEGIES[strategy];
  if (!fn) throw new Error(`unknown strategy ${strategy}`);
  let n = 0, wins = 0, pnl = 0, expWins = 0, variance = 0;
  for (let i = warmup; i < digits.length - 1; i++) {
    const c = fn(digits.slice(0, i + 1), opts);
    if (!c) continue;
    const payout = payoutFor(c, margin);
    const be = 1 / payout; // break-even win probability
    const win = digitWins(c, digits[i + 1]);
    n++; expWins += be; variance += be * (1 - be);
    if (win) { wins++; pnl += payout - 1; } else { pnl -= 1; }
  }
  if (n === 0) return { n: 0, verdict: "NO TRADES" };
  const z = variance > 0 ? (wins - expWins) / Math.sqrt(variance) : 0;
  return {
    n, wins, winRate: wins / n, breakEven: expWins / n,
    evPerTrade: pnl / n, // in stakes; negative = losing money
    z, pValue: pOneSided(z),
  };
}

/**
 * Verdict with a multiple-testing correction: when you try K strategy/symbol
 * combinations, one will look good by luck, so alpha is divided by K.
 */
export function verdict(bt, { alpha = 0.01, tests = 1 } = {}) {
  if (!bt.n) return "NO TRADES";
  if (bt.n < 300) return "TOO FEW TRADES";
  if (bt.evPerTrade > 0 && bt.pValue < alpha / Math.max(1, tests)) return "EDGE? (re-test on fresh ticks)";
  return "NO EDGE";
}

// --------------------------------------------------------------- live bot ----

export class IndicesBot {
  /**
   * DEMO ONLY. Fixed stake, no martingale, hard daily caps.
   * `client` is a connected-capable DerivClient (from deriv.js) holding a DEMO token.
   */
  constructor({
    db, client, symbol = "R_100", strategy = "evenOddReversal", strategyOpts = {},
    stake = 1, maxTradesDay = 30, maxDailyLoss = 5, maxConsecLosses = 5,
    currency = "USD", log = null,
  }) {
    if (!INDEX_SYMBOL.test(symbol)) throw new Error(`not a Volatility index: ${symbol}`);
    if (!STRATEGIES[strategy]) throw new Error(`unknown strategy ${strategy}`);
    this.db = db; this.client = client; this.symbol = symbol;
    this.strategy = strategy; this.strategyOpts = strategyOpts;
    this.stake = Math.min(Math.max(Number(stake) || 1, 0.35), 10);
    this.maxTradesDay = maxTradesDay; this.maxDailyLoss = maxDailyLoss;
    this.maxConsecLosses = maxConsecLosses; this.currency = currency;
    this.log = log || (() => {});
    this.running = false;
    db.exec(`
      CREATE TABLE IF NOT EXISTS deriv_indices_trades (
        contract_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL, strategy TEXT NOT NULL,
        contract_type TEXT NOT NULL, barrier INTEGER,
        stake REAL NOT NULL, profit REAL,
        status TEXT DEFAULT 'OPEN', placed_at INTEGER NOT NULL
      );
    `);
  }

  _dayStart() { const d = new Date(); d.setUTCHours(0, 0, 0, 0); return d.getTime(); }

  stats() {
    const since = this._dayStart();
    const row = this.db.prepare(
      "SELECT COUNT(*) AS n, COALESCE(SUM(profit),0) AS pnl FROM deriv_indices_trades WHERE placed_at >= ?"
    ).get(since);
    const recent = this.db.prepare(
      "SELECT profit FROM deriv_indices_trades WHERE status='SETTLED' ORDER BY placed_at DESC LIMIT ?"
    ).all(this.maxConsecLosses).map((r) => r.profit);
    let consecLosses = 0;
    for (const p of recent) { if (p <= 0) consecLosses++; else break; }
    return { tradesToday: row.n, pnlToday: row.pnl, consecLosses };
  }

  /** One decision cycle: caps -> signal -> buy -> settle. Never throws. */
  async step() {
    try {
      const s = this.stats();
      if (s.tradesToday >= this.maxTradesDay) return { action: "halt", why: `daily trade cap ${this.maxTradesDay}` };
      if (s.pnlToday <= -Math.abs(this.maxDailyLoss)) return { action: "halt", why: `daily loss limit ${this.maxDailyLoss} hit` };
      if (s.consecLosses >= this.maxConsecLosses) return { action: "halt", why: `${this.maxConsecLosses} losses in a row` };

      if (!(await this.client.connect())) return { action: "error", why: "deriv unavailable" };
      if (!this.client.token) return { action: "error", why: "no token — set a DEMO token" };
      if (!this.client.account?.isDemo) return { action: "halt", why: "REAL-money account refused: demo only" };

      const h = await this.client.call({ ticks_history: this.symbol, count: 300, end: "latest", style: "ticks" });
      const prices = h.history?.prices || h.prices;
      if (h.error || !Array.isArray(prices) || prices.length < 120) {
        return { action: "error", why: `ticks: ${h.error?.message || "too few"}` };
      }
      const pip = Number(h.pip_size ?? h.history?.pip_size ?? 2);
      const digits = toDigits(prices, pip);
      const c = STRATEGIES[this.strategy](digits, this.strategyOpts);
      if (!c) return { action: "wait" };

      const req = {
        proposal: 1, amount: this.stake, basis: "stake", contract_type: c.type,
        currency: this.currency, duration: 1, duration_unit: "t",
        underlying_symbol: this.symbol,
      };
      if (c.barrier != null) req.barrier = String(c.barrier);
      const prop = await this.client.call(req);
      const p = prop.proposal;
      if (prop.error || !p?.id) return { action: "error", why: `proposal: ${prop.error?.message || "none"}` };
      const ask = Number(p.ask_price);
      if (!(ask > 0)) return { action: "error", why: "proposal: no ask price" };
      if (ask > this.stake * 1.02) return { action: "error", why: "ask above stake — refused" };

      const buy = await this.client.call({ buy: p.id, price: ask });
      const id = buy.buy?.contract_id;
      if (buy.error || !id) return { action: "error", why: `buy: ${buy.error?.message || "no id"}` };
      this.db.prepare(
        "INSERT OR REPLACE INTO deriv_indices_trades (contract_id,symbol,strategy,contract_type,barrier,stake,status,placed_at) VALUES (?,?,?,?,?,?,'OPEN',?)"
      ).run(String(id), this.symbol, this.strategy, c.type, c.barrier ?? null, this.stake, Date.now());

      for (let i = 0; i < 20; i++) {
        await new Promise((r) => setTimeout(r, 1000));
        const st = await this.client.contractStatus(id);
        if (st?.done && st.profit != null) {
          this.db.prepare("UPDATE deriv_indices_trades SET status='SETTLED', profit=? WHERE contract_id=?")
            .run(st.profit, String(id));
          this.log(`[indices] ${this.symbol} ${c.type}${c.barrier != null ? " " + c.barrier : ""} -> ${st.profit}`);
          return { action: "traded", contract: c, profit: st.profit };
        }
      }
      return { action: "traded", contract: c, profit: null, note: "settlement pending" };
    } catch (e) {
      return { action: "error", why: e.message };
    }
  }

  /** Sequential loop (one contract at a time). stop() or any halt ends it. */
  async run({ intervalMs = 3000 } = {}) {
    this.running = true;
    while (this.running) {
      const r = await this.step();
      if (r.action === "halt") { this.log(`[indices] HALT: ${r.why}`); this.running = false; return r; }
      await new Promise((res) => setTimeout(res, r.action === "traded" ? 500 : intervalMs));
    }
  }
  stop() { this.running = false; }
}
