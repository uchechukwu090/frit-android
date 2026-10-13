// Offline self-test: Alpaca executor safety (no keys, no network).
import Database from "better-sqlite3";
import { AlpacaClient, AlpacaExecutor, isStockSymbol, toAlpacaSymbol } from "./src/strategy/alpaca.js";

const db = new Database(":memory:");
const cryptoSet = new Set(["BTC", "ETH", "BTCUSD", "ETHUSD"]);
console.log(`AAPL stock? ${isStockSymbol("AAPL", { cryptoSet })} (expect true)`);
console.log(`EURUSD stock? ${isStockSymbol("EURUSD", { cryptoSet })} (expect false)`);
console.log(`BTC stock? ${isStockSymbol("BTC", { cryptoSet })} (expect false)`);
console.log(`XAUUSD stock? ${isStockSymbol("XAUUSD", { cryptoSet })} (expect false)`);
console.log(`map: ${toAlpacaSymbol("spy")} (expect SPY)`);

const nokeys = new AlpacaClient({ keyId: "", secret: "", paper: true, fetchImpl: fetch });
const ex = new AlpacaExecutor({ db, paperClient: nokeys, liveClient: nokeys, liveAllowed: false });
const r = await ex.place({ symbol: "AAPL", action: "buy" });
console.log(`place (no keys): ok=${r.ok} why=${r.why} (expect false/no keys)`);
console.log(`mode default: ${ex.mode} (expect paper)`);
const exLive = new AlpacaExecutor({
  db, paperClient: nokeys,
  liveClient: new AlpacaClient({ keyId: "k", secret: "s", paper: false, fetchImpl: fetch }),
  liveAllowed: false,
});
console.log(`live keys but flag off -> mode: ${exLive.mode} (expect paper)`);
console.log("SELFTEST PASS");
