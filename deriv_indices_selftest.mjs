// Run on YOUR machine (needs internet to Deriv; no token needed):
//   node deriv_indices_selftest.mjs
// Pulls the last 5000 real ticks per index and tests every ported strategy.
import { DerivClient } from "./src/strategy/deriv.js";
import { STRATEGIES, backtest, verdict, toDigits, digitChi2 } from "./src/strategy/deriv-indices.js";

const SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100", "1HZ10V", "1HZ100V"];
const names = Object.keys(STRATEGIES);
const tests = SYMBOLS.length * names.length;

const client = new DerivClient({});
if (!(await client.connect())) { console.error("cannot reach Deriv:", client.status()); process.exit(1); }

console.log(`Testing ${names.length} strategies x ${SYMBOLS.length} indices (alpha 0.01, Bonferroni over ${tests} tests)\n`);
for (const sym of SYMBOLS) {
  const h = await client.call({ ticks_history: sym, count: 5000, end: "latest", style: "ticks" });
  const prices = h.history?.prices || h.prices;
  if (h.error || !Array.isArray(prices)) { console.log(sym, "skipped:", h.error?.message || "no data"); continue; }
  const pip = Number(h.pip_size ?? h.history?.pip_size ?? 2);
  const digits = toDigits(prices, pip);
  console.log(`${sym}  ticks=${digits.length}  digit chi2=${digitChi2(digits).toFixed(1)} (uniform if < 21.7)`);
  for (const name of names) {
    const bt = backtest(digits, name);
    const v = verdict(bt, { tests });
    console.log(
      `  ${name.padEnd(20)} n=${String(bt.n).padStart(5)}  win=${(bt.winRate * 100 || 0).toFixed(1)}%` +
      `  breakeven=${(bt.breakEven * 100 || 0).toFixed(1)}%  EV/trade=${(bt.evPerTrade || 0).toFixed(3)}` +
      `  p=${(bt.pValue ?? 1).toFixed(3)}  -> ${v}`
    );
  }
  console.log();
}
client.close();
