// Offline self-test: pocket v2 (profiles + OFI + calibration) — no network.
import Database from "better-sqlite3";
import { CandleStore } from "./src/strategy/feed/candle_store.js";
import { signalFromCandles, recordPaper, settleOpen, pocketStats, calibFor, profileFor, ensurePocketTables, assessPullback, upsertArm, getArm, clearArm } from "./src/strategy/pocket/engine.js";

const db = new Database(":memory:");
ensurePocketTables(db);
const cryptoSet = new Set(["BTCUSD"]);

// --- 1. Profiles ---
console.log(`profiles: 1m->${JSON.stringify(profileFor(1))} 5m->${JSON.stringify(profileFor(5))} 15m->${JSON.stringify(profileFor(15))}`);

// --- 2. OFI ring via sided trades ---
const store = new CandleStore(db);
let px = 100;
for (let i = 0; i < 120; i++) {
  px += 0.1;
  store.ingestTrade("BTCUSD", px, 2, Date.now() + i * 1000, 1); // buyer-initiated
}
console.log(`ofi (all buys) = ${store.ofi("BTCUSD")}`);
for (let i = 0; i < 120; i++) {
  px -= 0.05;
  store.ingestTrade("BTCUSD", px, 2, Date.now() + 200000 + i * 1000, i % 2 ? 1 : -1);
}
console.log(`ofi (mixed) = ${store.ofi("BTCUSD")?.toFixed(2)} (expect ~0.00)`);
console.log(`ofi (no flow) = ${store.ofi("XAUUSD")} (expect null)`);

// --- 3. Signal on synthetic trend (15m trend + 5m trigger for expiry 5) ---
let p2 = 2650;
const trend = [];
for (let i = 0; i < 60; i++) {
  const o = p2;
  p2 += 0.5 + Math.sin(i / 4) * 0.6;
  trend.push({ open: o, high: Math.max(o, p2) + 0.2, low: Math.min(o, p2) - 0.2, close: p2 });
}
const trig = [];
for (let i = 0; i < 30; i++) {
  const o = p2;
  p2 += 0.9; // same scale as the trend TF bodies (else the ATR gate rightly waits)
  trig.push({ open: o, high: Math.max(o, p2) + 0.15, low: Math.min(o, p2) - 0.15, close: p2 });
}
const base = { cryptoSet, expiryMin: 5, payout: 0.8, ofi: 0.4 };
const probe = signalFromCandles("BTCUSD", trend, trig, base);
console.log(`probe: ${probe.decision} conf=${probe.confidence} setup=${probe.setupKey}`);
console.log(`learning cap applied: ${probe.confidence <= 65 && probe.decision !== "WAIT" ? "yes" : (probe.decision === "WAIT" ? "n/a (WAIT)" : "NO!")}`);

// --- 3b. Counter-trend: trend TF bullish, trigger TF bearish -> must WAIT ---
let p3 = 2700;
const trigDown = [];
for (let i = 0; i < 30; i++) {
  const o = p3;
  p3 -= 0.9;
  // last candle prints a bullish body (the old trap) inside a bearish structure
  const c = i === 29 ? o + 1.2 : p3;
  trigDown.push({ open: o, high: Math.max(o, c) + 0.15, low: Math.min(o, c) - 0.15, close: c });
}
const counter = signalFromCandles("BTCUSD", trend, trigDown, base);
console.log(`counter-trend: ${counter.decision} (expect WAIT) | ${(counter.reasons[0] || "").slice(0, 100)}`);

// --- 4. Calibration: 40 fake paper trades at 70% win on this bucket, then gate ---
if (probe.setupKey) {
  for (let i = 0; i < 40; i++) {
    const s = { ...probe, decision: "CALL", entryRef: 100, expiryMin: 5 };
    const row = recordPaper(db, { userId: "__pocket__", signal: s, payout: 0.8 });
    db.prepare("UPDATE pocket_paper SET expiry_ts = ? WHERE id = ?").run(Date.now() - 1000, row.id);
    settleOpen(db, () => ({ price: i < 28 ? 101 : 99 })); // 28/40 = 70%
  }
  const calib = calibFor(db, probe.setupKey);
  console.log(`calib: n=${calib.n} wr=${calib.wr} (expect 40/0.7)`);
  const gated = signalFromCandles("BTCUSD", trend, trig, { ...base, calib });
  console.log(`gated: ${gated.decision} conf=${gated.confidence} (expect CALL, conf>65 grounded)`);
  // Poison the bucket: 40 more losses -> win rate drops under breakeven -> KILL.
  for (let i = 0; i < 40; i++) {
    const s = { ...probe, decision: "CALL", entryRef: 100, expiryMin: 5 };
    const row = recordPaper(db, { userId: "__pocket__", signal: s, payout: 0.8 });
    db.prepare("UPDATE pocket_paper SET expiry_ts = ? WHERE id = ?").run(Date.now() - 1000, row.id);
    settleOpen(db, () => ({ price: 99 }));
  }
  const calib2 = calibFor(db, probe.setupKey);
  const killed = signalFromCandles("BTCUSD", trend, trig, { ...base, calib: calib2 });
  console.log(`after poisoning: wr=${calib2.wr.toFixed(2)} -> ${killed.decision} (expect WAIT + CALIBRATION KILL)`);
  const st = pocketStats(db, { payout: 0.8 });
  console.log(`stats buckets tracked: ${st.calibration.length}, verdict: ${st.verdict.slice(0, 80)}`);
}
// --- 5. Pullback diagnosis: shallow pullback -> HEALTHY (armable) ---
let p4 = 100;
const tUp = [];
for (let i = 0; i < 60; i++) {
  const o = p4;
  // wavy climb (forms real swing pivots) then a mild pullback
  p4 += i < 45 ? 1.0 + Math.sin(i / 3) * 1.1 : -0.35;
  tUp.push({ open: o, high: Math.max(o, p4) + 0.25, low: Math.min(o, p4) - 0.25, close: p4 });
}
const g1 = [];
for (let i = 0; i < 30; i++) {
  const o = p4 - (29 - i) * 0.35;
  const c = o - 0.25; // gentle drift down, no thrust
  g1.push({ open: o, high: Math.max(o, c) + 0.1, low: Math.min(o, c) - 0.1, close: c });
}
const pb1 = assessPullback("BTCUSD", tUp, g1, 1, 1.5);
console.log(`pullback shallow: ${pb1.verdict} retrace=${pb1.retracePct?.toFixed(0)}% (expect HEALTHY, <=50%)`);

// --- 6. Structure break -> REVERSAL ---
const g2 = g1.map(c => ({ ...c }));
g2.push({ open: p4, high: p4 + 6, low: p4 - 0.1, close: p4 + 5.5 }); // close beyond leg extreme
const pb2 = assessPullback("BTCUSD", tUp, g2, 1, 1.5);
console.log(`structure break: ${pb2.verdict} (expect REVERSAL)`);

// --- 7. Arm lifecycle ---
upsertArm(db, { symbol: "BTCUSD", expiryMin: 5, direction: "CALL", setupKey: "k", expiresTs: Date.now() + 60000, note: "t" });
console.log(`arm stored: ${getArm(db, "BTCUSD", 5)?.direction} (expect CALL)`);
upsertArm(db, { symbol: "BTCUSD", expiryMin: 5, direction: "CALL", setupKey: "k", expiresTs: Date.now() - 1000, note: "t" });
console.log(`arm expired: ${getArm(db, "BTCUSD", 5)} (expect null)`);
upsertArm(db, { symbol: "BTCUSD", expiryMin: 5, direction: "PUT", setupKey: "k", expiresTs: Date.now() + 60000, note: "t" });
clearArm(db, "BTCUSD", 5);
console.log(`arm cleared: ${getArm(db, "BTCUSD", 5)} (expect null)`);
console.log("SELFTEST DONE");
