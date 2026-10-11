// Offline self-test: pocket v2 (profiles + OFI + calibration) — no network.
import Database from "better-sqlite3";
import { CandleStore } from "./src/strategy/feed/candle_store.js";
import { signalFromCandles, recordPaper, settleOpen, pocketStats, calibFor, profileFor, ensurePocketTables } from "./src/strategy/pocket/engine.js";

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
for (let i = 0; i < 20; i++) {
  const o = p2;
  p2 += 0.9; // same scale as the trend TF bodies (else the ATR gate rightly waits)
  trig.push({ open: o, high: Math.max(o, p2) + 0.15, low: Math.min(o, p2) - 0.15, close: p2 });
}
const base = { cryptoSet, expiryMin: 5, payout: 0.8, ofi: 0.4 };
const probe = signalFromCandles("BTCUSD", trend, trig, base);
console.log(`probe: ${probe.decision} conf=${probe.confidence} setup=${probe.setupKey}`);
console.log(`learning cap applied: ${probe.confidence <= 65 && probe.decision !== "WAIT" ? "yes" : (probe.decision === "WAIT" ? "n/a (WAIT)" : "NO!")}`);

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
console.log("SELFTEST DONE");
