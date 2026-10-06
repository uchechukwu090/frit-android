// Offline self-test: CandleStore tick aggregation + engine seam (no network).
import Database from "better-sqlite3";
import { CandleStore } from "./src/strategy/feed/candle_store.js";
import { MTFStrategyEngine } from "./src/strategy/engine.js";

const db = new Database(":memory:");
let closes30 = 0, closes4h = 0;
const store = new CandleStore(db, {
  onClose: (s, iv) => { if (iv === "30m") closes30++; else closes4h++; },
});

// Uptrend with pullbacks: 50 days of 5-min ticks (~14,400 ticks).
let px = 2650;
const start = Date.now() - 50 * 24 * 3600 * 1000;
for (let m = 0; m < 50 * 24 * 12; m++) {
  const t = start + m * 5 * 60 * 1000;
  px += 0.06 + Math.sin(m / 40) * 0.9; // drift + waves (pullbacks included)
  store.ingestTick("XAUUSD", px, 10, t);
}
// Live edge tick at now.
store.ingestTick("XAUUSD", px, 10, Date.now());

const c30 = store.getCandles("XAUUSD", "30m", 150);
const c4h = store.getCandles("XAUUSD", "4h", 100);
console.log(`candles: 30m=${c30.length} 4h=${c4h.length} closed30=${closes30} closed4h=${closes4h}`);
console.log(`lastPrice: ${JSON.stringify(store.lastPrice("XAUUSD"))}`);
console.log(`newest30m age: ${Math.round(store.stalenessSec("XAUUSD", "30m"))}s`);

// Binance kline path.
const k = { openTime: Date.now(), open: px, high: px + 1, low: px - 1, close: px + 0.5, volume: 5, isClosed: true };
const closed = store.ingestKline("BTCUSD", "30m", k);
console.log(`kline ingest closed: ${!!closed} btcCount=${store.candleCount("BTCUSD", "30m")}`);

if (c30.length < 70 || c4h.length < 40) {
  console.log("WARMUP INSUFFICIENT — test inconclusive");
  process.exit(2);
}
const engine = new MTFStrategyEngine({});
const r = await engine.analyze("XAUUSD", { candles30M: c30, candles4H: c4h });
console.log(`engine: decision=${r.decision} conf=${r.confidence} entry=${r.entry} sl=${r.sl} tp=${r.tp} rr=${r.rr}`);
console.log(`regime=${r.regime?.macro_trend} reasons=${r.reasons?.length}`);
console.log("SELFTEST PASS");
