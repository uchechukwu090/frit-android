// Offline self-test: watchlist tombstones + pocket watchlist separation.
import Database from "better-sqlite3";

const db = new Database(":memory:");
db.exec(`
  CREATE TABLE IF NOT EXISTS trading_watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, symbol TEXT NOT NULL,
    name TEXT, asset_class TEXT DEFAULT 'FOREX', is_enabled INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(user_id, symbol));
  CREATE TABLE IF NOT EXISTS pocket_watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, symbol TEXT NOT NULL,
    expiry_min INTEGER DEFAULT 5, is_enabled INTEGER DEFAULT 1, sort_order INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(user_id, symbol));
  CREATE TABLE IF NOT EXISTS watchlist_tombstones (
    user_id TEXT NOT NULL, list TEXT NOT NULL, symbol TEXT NOT NULL,
    removed_at DATETIME DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (user_id, list, symbol));
`);

const uid = "u1";
const addMain = (s) => db.prepare("INSERT INTO trading_watchlist (user_id, symbol, name) VALUES (?,?,?)").run(uid, s, s);
addMain("XAUUSD"); addMain("EURUSD"); addMain("BTCUSD");
db.prepare("INSERT INTO pocket_watchlist (user_id, symbol) VALUES (?,?)").run(uid, "XAUUSD");

// Simulate: user deletes XAUUSD from the MAIN list (tombstone written).
db.prepare("DELETE FROM trading_watchlist WHERE user_id = ? AND symbol = ?").run(uid, "XAUUSD");
db.prepare("INSERT OR IGNORE INTO watchlist_tombstones (user_id, list, symbol) VALUES (?, 'main', ?)").run(uid, "XAUUSD");

// Re-add the rows as if a stale cache / bot wrote them back (resurrection attempt).
addMain("XAUUSD");

// The GET filters tombstones (mirrors the route).
const dead = db.prepare("SELECT symbol FROM watchlist_tombstones WHERE user_id = ? AND list = 'main'").all(uid).map(r => r.symbol);
const mainItems = db.prepare("SELECT symbol FROM trading_watchlist WHERE user_id = ? ORDER BY sort_order").all(uid)
  .map(r => r.symbol).filter(s => !dead.includes(s));
console.log(`main after delete+resurrect attempt: [${mainItems.join(",")}] (XAUUSD must be GONE)`);

// Explicit re-add clears the tombstone (mirrors the POST route).
db.prepare("DELETE FROM watchlist_tombstones WHERE user_id = ? AND list = 'main' AND symbol = ?").run(uid, "XAUUSD");
db.prepare("INSERT OR IGNORE INTO trading_watchlist (user_id, symbol, name) VALUES (?,?,?)").run(uid, "XAUUSD", "XAUUSD");
const after2 = db.prepare("SELECT symbol FROM trading_watchlist WHERE user_id = ? ORDER BY sort_order").all(uid).map(r => r.symbol);
console.log(`after explicit re-add: [${after2.join(",")}] (XAUUSD back)`);

// Pocket list is independent of the main list.
const pocket = db.prepare("SELECT symbol FROM pocket_watchlist WHERE user_id = ?").all(uid).map(r => r.symbol);
db.prepare("DELETE FROM pocket_watchlist WHERE user_id = ? AND symbol = ?").run(uid, "XAUUSD");
const pocketAfter = db.prepare("SELECT symbol FROM pocket_watchlist WHERE user_id = ?").all(uid).map(r => r.symbol);
const mainStill = db.prepare("SELECT COUNT(*) AS c FROM trading_watchlist WHERE user_id = ? AND symbol = 'XAUUSD'").get(uid).c;
console.log(`pocket before=[${pocket.join(",")}] after=[${pocketAfter.join(",")}] mainStillHasXAU=${mainStill} (must be 1)`);

console.log(!mainItems.includes("XAUUSD") && after2.includes("XAUUSD") && pocketAfter.length === 0 && mainStill === 1
  ? "SELFTEST PASS" : "SELFTEST FAIL");