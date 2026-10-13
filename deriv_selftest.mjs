// Offline self-test: Deriv executor safety (no token, no network to broker).
import Database from "better-sqlite3";
import { DerivClient, DerivExecutor, buildProposal, DERIV_MAP } from "./src/strategy/deriv.js";

const db = new Database(":memory:");
// 1. Proposal payload shape.
const p = buildProposal("frxXAUUSD", "CALL", 1, 5);
console.log(`proposal: ${JSON.stringify(p)}`);
console.log(`map XAUUSD: ${DERIV_MAP.XAUUSD} | BTCUSD: ${DERIV_MAP.BTCUSD}`);
// 2. No token -> safe no-ops, never throws, never trades.
const client = new DerivClient({ token: "" });
const ex = new DerivExecutor({ db, client, liveAllowed: false, stake: 1, maxPerDay: 20 });
console.log(`connect (no token): ${await client.connect()} (expect false)`);
console.log(`status: ${JSON.stringify(ex.status())}`);
const r = await ex.execute({ symbol: "XAUUSD", direction: "CALL", durationMin: 5, positionId: null });
console.log(`execute (no token): ok=${r.ok} why=${r.why} (expect false/no token)`);
// 3. Daily cap enforced without network.
for (let i = 0; i < 20; i++) {
  db.prepare("INSERT INTO deriv_contracts (contract_id, position_id, symbol, deriv_symbol, direction, stake, duration_min, status, mode, placed_at) VALUES (?,?,?,?,?,?,?,?,?,?)")
    .run(`t${i}`, null, "XAUUSD", "frxXAUUSD", "CALL", 1, 5, "OPEN", "demo", Date.now());
}
const ex2 = new DerivExecutor({ db, client: new DerivClient({ token: "fake" }), liveAllowed: false, stake: 1, maxPerDay: 20 });
const r2 = await ex2.execute({ symbol: "XAUUSD", direction: "CALL", durationMin: 5, positionId: null });
console.log(`execute (cap): ok=${r2.ok} why=${r2.why} (expect false/daily cap)`);
console.log(r.ok === false && /daily cap/.test(r2.why || "") ? "SELFTEST PASS" : "SELFTEST FAIL");
