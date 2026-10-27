// ============================================================================
// DERIV ALL-MARKETS TEST — proves every market Deriv lists can take a FRIT signal
// ----------------------------------------------------------------------------
// Run from the server folder (PowerShell):
//   $env:PAT="<your deriv PAT>"; $env:DERIV_APP_ID="<your app id>"
//   node deriv_all_markets_test.mjs --dry            # capability matrix only, no orders
//   node deriv_all_markets_test.mjs                  # one min-stake test trade per market
//   node deriv_all_markets_test.mjs --only=R_100,frxEURUSD,OTC_DJI
//   node deriv_all_markets_test.mjs --stake=1 --limit=10 --keep
//
// What it does:
//   1. Connects with the PAT (OTP websocket flow) and picks the DEMO account.
//      It ABORTS if the account is not demo — this script never touches real money.
//   2. Builds the capability matrix: for every underlying Deriv lists, which
//      contract families are offered (Multiplier / Rise-Fall / Digits).
//   3. Unless --dry, sends one small test signal per tradable market through the
//      exact same DerivExecutor.executeBest() path the FRIT bots use, alternating
//      buy/sell so both directions are exercised (MULTUP/MULTDN, CALL/PUT).
//      Multipliers are closed again right away unless --keep is passed.
//   4. Prints a table and writes deriv_markets_report.json next to this file.
// Orders are recorded in an in-memory ledger, not in FRIT's real database.
// ============================================================================

import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DerivClient, DerivExecutor } from "./src/strategy/deriv.js";

const token = process.env.PAT || process.env.DERIV_PAT || process.env.DERIV_DEMO_TOKEN || process.env.DERIV_TOKEN || "";
const appId = process.env.DERIV_APP_ID || "";

const args = process.argv.slice(2);
const flag = (n) => args.includes(`--${n}`);
const opt = (n, d = null) => {
  const a = args.find((x) => x.startsWith(`--${n}=`));
  return a ? a.slice(n.length + 3) : d;
};
const DRY = flag("dry");
const KEEP = flag("keep");
const STAKE = Math.max(0.35, Number(opt("stake", 1)) || 1);
const LIMIT = Number(opt("limit", 0)) || 0;
const ONLY = (opt("only", "") || "").split(",").map((s) => s.trim()).filter(Boolean);
const DURATION_MIN = Number(opt("duration", 5)) || 5;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pad = (s, n) => String(s ?? "").slice(0, n).padEnd(n);

if (!token) {
  console.error("No token. Set $env:PAT (or DERIV_DEMO_TOKEN) first.");
  process.exit(1);
}
if (!appId) {
  console.error("No DERIV_APP_ID. PAT auth needs the app id header. Set $env:DERIV_APP_ID first.");
  process.exit(1);
}

const db = new Database(":memory:");
const client = new DerivClient({
  token, appId, log: (...a) => console.log(...a),
  onStatus: (s, d) => { if (d) console.log(`[deriv ${s}] ${d}`); },
});
const executor = new DerivExecutor({
  db, client, liveAllowed: false, stake: STAKE, maxPerDay: 100000,
  allowRngOnDemo: true, log: () => {},
});

console.log("Connecting to Deriv...");
const ok = await client.connect();
if (!ok) {
  console.error(`Could not connect: ${client.status()}`);
  console.error("Check: PAT has the Trade scope, DERIV_APP_ID is the registered app id, and the token was not revoked.");
  process.exit(1);
}
const acct = client.account;
console.log(`Account: ${acct?.accountId} (${acct?.accountType}) ${acct?.currency}`);
if (!acct?.isDemo) {
  console.error("This token resolved to a REAL account. Aborting: this test only runs on demo.");
  client.close();
  process.exit(2);
}
const bal = await client.balance().catch(() => null);
if (bal) console.log(`Demo balance: ${bal.balance} ${bal.currency}`);

console.log("\nBuilding capability matrix (every market Deriv lists)...");
const matrix = await executor.marketMatrix({ force: true });
if (matrix.error) {
  console.error(matrix.error);
  client.close();
  process.exit(1);
}
let rows = matrix.rows;
const byMarket = {};
for (const r of rows) {
  const k = r.market || "other";
  (byMarket[k] ||= { total: 0, mult: 0, riseFall: 0, digits: 0 });
  byMarket[k].total++;
  if (r.mult) byMarket[k].mult++;
  if (r.riseFall) byMarket[k].riseFall++;
  if (r.digits) byMarket[k].digits++;
}
console.log(`${rows.length} markets, ${rows.filter((r) => r.mult || r.riseFall).length} tradable with MULT or Rise/Fall\n`);
console.log(pad("market", 18), pad("total", 6), pad("mult", 6), pad("rise/fall", 10), "digits");
for (const [k, v] of Object.entries(byMarket)) {
  console.log(pad(k, 18), pad(v.total, 6), pad(v.mult, 6), pad(v.riseFall, 10), v.digits);
}

const report = { at: new Date().toISOString(), account: acct, stake: STAKE, dry: DRY, matrix: rows, trades: [] };
const outFile = path.join(path.dirname(fileURLToPath(import.meta.url)), "deriv_markets_report.json");

if (DRY) {
  fs.writeFileSync(outFile, JSON.stringify(report, null, 2));
  console.log(`\nDry run done. Report: ${outFile}`);
  client.close();
  process.exit(0);
}

let targets = rows.filter((r) => r.mult || r.riseFall);
if (ONLY.length) targets = targets.filter((r) => ONLY.some((o) => o.toLowerCase() === r.symbol.toLowerCase()));
if (LIMIT) targets = targets.slice(0, LIMIT);

console.log(`\nSending test signals to ${targets.length} market(s), stake ${STAKE}, demo account...\n`);
console.log(pad("symbol", 14), pad("sent as", 9), pad("dir", 5), pad("result", 7), "detail");

let i = 0;
let sent = 0;
for (const r of targets) {
  const direction = i++ % 2 === 0 ? "buy" : "sell";
  const t0 = Date.now();
  let res;
  try {
    res = await executor.executeBest({
      symbol: r.symbol, direction, stake: STAKE, multiplier: 100, durationMin: DURATION_MIN,
    });
  } catch (e) {
    res = { ok: false, why: e.message };
  }
  const row = {
    symbol: r.symbol, market: r.market, direction, ok: !!res.ok,
    contractType: res.contractType || null, contractId: res.contract_id ?? null,
    why: res.ok ? null : res.why, ms: Date.now() - t0,
  };
  if (res.ok) {
    sent++;
    const isMult = /^MULT/.test(res.contractType || "");
    if (isMult && !KEEP && res.contract_id) {
      await sleep(1500);
      try {
        const s = await client.sellContract(res.contract_id);
        row.closed = true;
        row.closePnl = s.profit;
      } catch (e) {
        row.closed = false;
        row.closeError = e.message;
      }
    }
  }
  report.trades.push(row);
  console.log(
    pad(r.symbol, 14), pad(row.contractType || "-", 9), pad(direction, 5),
    pad(row.ok ? "OK" : "FAIL", 7), row.ok ? `id=${row.contractId}${row.closed === false ? " (close failed)" : ""}` : String(row.why).slice(0, 140),
  );
  await sleep(500);
}

const failed = report.trades.filter((t) => !t.ok);
console.log(`\nDone: ${sent}/${targets.length} markets accepted a signal.`);
if (failed.length) {
  console.log("Markets that refused (check the reason, then tell me and I'll fix the mapping):");
  for (const f of failed) console.log(`  ${f.symbol}: ${String(f.why).slice(0, 200)}`);
}
const noContract = rows.filter((r) => !r.mult && !r.riseFall);
if (noContract.length) {
  console.log(`\n${noContract.length} listed market(s) offer neither Multiplier nor Rise/Fall on this account (e.g. digits/accumulator-only):`);
  console.log("  " + noContract.map((r) => r.symbol).join(", "));
}

fs.writeFileSync(outFile, JSON.stringify(report, null, 2));
console.log(`\nReport written: ${outFile}`);
client.close();
process.exit(failed.length ? 3 : 0);
