// ============================================================================
// POCKET ENGINE — fixed-expiry directional signals (Pocket Option style).
// ----------------------------------------------------------------------------
// Answers ONE question: "CALL or PUT on <symbol> for the next N-minute
// expiry, entered at the next candle open?" Computed ONLY on CLOSED candles
// (a signal that moves mid-candle is repainting — backtests lie, live loses).
//
// v1 rule (deliberately simple, auditable):
//   5m trend   : EMA9 vs EMA21 (persistent alignment, not a fresh cross)
//   5m momentum: last closed 5m body points with the trend, body >= 0.15x ATR
//   1m trigger : last closed 1m body points the same way (falls back to
//                5m-only with lower confidence when 1m history is thin)
//   guards     : candle not flatlined (range >= 0.5x ATR), in-session
//                (FX/metals London/NY 07–17 GMT weekdays; crypto 24/7)
// Confidence capped at 78 and labeled experimental. Paper ledger settles
// every expiry against OUR feed price and reports win-rate vs the breakeven
// the broker's payout demands — the only number that matters.
//
// HONESTY LIMITS (read before trusting a signal):
// - Pocket Option prices OTC pairs on THEIR feed. Our settle price comes from
//   Binance/Finnhub/Dukascopy and can differ by enough to flip a 5m outcome.
//   Paper stats here are INDICATIVE, not broker P&L.
// - No auto-execution: manual signals only. Automating Pocket Option breaches
//   their ToS and gets accounts banned.
// ============================================================================

const EMA_FAST = 9;
const EMA_SLOW = 21;
const ATR_PERIOD = 14;
const MIN_CONF = 60;
const MAX_CONF = 78;

function emaLast(values, period) {
  if (values.length < period) return null;
  const k = 2 / (period + 1);
  let e = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < values.length; i++) e = values[i] * k + e * (1 - k);
  return e;
}

function atrLast(candles, period = ATR_PERIOD) {
  if (candles.length < period + 1) return null;
  const trs = [];
  for (let i = 1; i < candles.length; i++) {
    const c = candles[i], p = candles[i - 1];
    trs.push(Math.max(c.high - c.low, Math.abs(c.high - p.close), Math.abs(c.low - p.close)));
  }
  const seed = trs.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let a = seed;
  for (let i = period; i < trs.length; i++) a = (a * (period - 1) + trs[i]) / period;
  return a;
}

function sessionOk(symbol, cryptoSet) {
  const sym = String(symbol).toUpperCase();
  if (cryptoSet.has(sym)) return { ok: true, reason: "crypto 24/7" };
  const d = new Date();
  const day = d.getUTCDay(), hour = d.getUTCHours() + d.getUTCMinutes() / 60;
  if (day === 0 || day === 6) return { ok: false, reason: "weekend — FX/metals closed" };
  if (hour < 7 || hour >= 17) {
    return { ok: false, reason: `outside London/NY liquidity (07:00–17:00 GMT, now ${hour.toFixed(1)})` };
  }
  return { ok: true, reason: "London/NY session" };
}

export function nextBoundaryMs(intervalSec, nowMs = Date.now()) {
  const s = intervalSec * 1000;
  return Math.ceil(nowMs / s) * s;
}

// Multi-expiry TF profiles. The expiry picks the trend TF above it and the
// trigger TF at/below it; all thresholds are ATR-normalized, which is what
// makes the SAME rule adapt across timeframes instead of needing retuning.
export const EXPIRY_PROFILES = {
  1: { trendIv: "5m", trigIv: "1m", need: { trend: 40, trig: 5 } },
  5: { trendIv: "15m", trigIv: "5m", need: { trend: 40, trig: 8 } },
  15: { trendIv: "1h", trigIv: "15m", need: { trend: 40, trig: 8 } },
};
export function profileFor(expiryMin) {
  const keys = Object.keys(EXPIRY_PROFILES).map(Number).sort((a, b) => a - b);
  let pick = keys[0];
  for (const k of keys) if (expiryMin >= k) pick = k;
  return { expiryMin: pick, ...EXPIRY_PROFILES[pick] };
}

function sessBucket(sym, cryptoSet) {
  if (cryptoSet.has(sym)) return "247";
  const d = new Date();
  if (d.getUTCDay() === 0 || d.getUTCDay() === 6) return "off";
  const h = d.getUTCHours() + d.getUTCMinutes() / 60;
  if (h >= 7 && h < 12) return "london";
  if (h >= 12 && h < 17) return "ny";
  return "off";
}

/**
 * v2 predictive signal.
 * @param trend closed trend-TF candles oldest->newest {open,high,low,close}
 * @param trig closed trigger-TF candles oldest->newest
 * @param opts { cryptoSet, expiryMin, ofi (null|[-1,1]), payout, calib {wr,n}|null }
 */
export function signalFromCandles(symbol, trend, trig, opts = {}) {
  const sym = String(symbol || "").toUpperCase();
  const cryptoSet = opts.cryptoSet || new Set();
  const prof = profileFor(opts.expiryMin || 5);
  const expiryMin = prof.expiryMin;
  const out = {
    symbol: sym, decision: "WAIT", confidence: 0, profile: `${prof.trendIv}/${prof.trigIv}`,
    expiryMin, entryRef: null, reasons: [], timing: null, setupKey: null, timestamp: Date.now(),
  };
  if (!Array.isArray(trend) || trend.length < prof.need.trend) {
    out.reasons.push(`warming up (${trend?.length || 0} ${prof.trendIv} candles, need ${prof.need.trend})`);
    return out;
  }
  if (!Array.isArray(trig) || trig.length < prof.need.trig) {
    out.reasons.push(`warming up (${trig?.length || 0} ${prof.trigIv} candles, need ${prof.need.trig})`);
    return out;
  }
  const closes = trend.map(c => c.close);
  const e9 = emaLast(closes, EMA_FAST);
  const e21 = emaLast(closes, EMA_SLOW);
  const atr = atrLast(trend);
  if (e9 == null || e21 == null || !(atr > 0)) {
    out.reasons.push("indicators not ready");
    return out;
  }
  const bull = e9 > e21, bear = e9 < e21;
  if (!bull && !bear) { out.reasons.push(`${prof.trendIv} EMA9/21 flat — no trend to ride`); return out; }
  const dir = bull ? "CALL" : "PUT";
  const sgn = bull ? 1 : -1;

  const sess = sessionOk(sym, cryptoSet);
  if (!sess.ok) { out.reasons.push(sess.reason); return out; }

  const last = trig[trig.length - 1];
  const body = sgn * (last.close - last.open);
  const range = last.high - last.low;
  if (range < 0.5 * atr) { out.reasons.push(`flatlined ${prof.trigIv} candle (range < 0.5x ATR) — dead market`); return out; }
  if (body < 0.15 * atr) { out.reasons.push(`last ${prof.trigIv} body too weak vs ATR — momentum unconfirmed`); return out; }

  let conf = 55;
  const reasons = [
    `${prof.trendIv} trend ${bull ? "bullish" : "bearish"} (EMA9 ${bull ? ">" : "<"} EMA21)`,
    `last ${prof.trigIv} body ${(body / atr).toFixed(2)}x ATR with trend`,
    sess.reason,
  ];
  const sep = Math.abs(e9 - e21) / atr;
  if (sep > 0.1) { conf += 10; reasons.push(`EMA separation ${sep.toFixed(2)}x ATR — clean trend`); }

  // Order-flow confirm (crypto only — other feeds carry no trade side).
  const ofi = typeof opts.ofi === "number" ? opts.ofi : null;
  if (ofi != null) {
    if (sgn * ofi > 0.05) { conf += 8; reasons.push(`order-flow agrees (OFI ${ofi.toFixed(2)}) — buyers/sellers hitting with us`); }
    else if (sgn * ofi < -0.05) { conf -= 12; reasons.push(`order-flow opposes (OFI ${ofi.toFixed(2)}) — tape fighting the signal`); }
    else reasons.push(`order-flow neutral (OFI ${ofi.toFixed(2)})`);
  } else {
    reasons.push("no order-flow on this feed (crypto/Binance only)");
  }

  conf = Math.max(20, Math.min(MAX_CONF, Math.round(conf)));

  // Setup identity for the calibration layer.
  const sepB = sep >= 0.3 ? "strong" : sep >= 0.1 ? "mid" : "weak";
  out.setupKey = `${sym}|${expiryMin}m|${sepB}|${sessBucket(sym, cryptoSet)}`;

  // Calibration gate: learned win-rate overrides marketing confidence.
  const payout = Number(opts.payout) || 0.8;
  const breakeven = 1 / (1 + payout);
  const calib = opts.calib || null;
  if (calib && calib.n >= 30) {
    if (calib.wr < breakeven + 0.01) {
      out.reasons.push(...reasons, `CALIBRATION KILL: setup won ${(calib.wr * 100).toFixed(1)}% over ${calib.n} (needs >${((breakeven + 0.01) * 100).toFixed(1)}%) — auto-disabled until edge returns`);
      return out;
    }
    conf = Math.round(0.4 * conf + 0.6 * calib.wr * 100);
    reasons.push(`calibrated ${(calib.wr * 100).toFixed(1)}% over ${calib.n} paper trades — confidence grounded in reality`);
  } else {
    conf = Math.min(conf, 65);
    reasons.push(`learning mode (${calib ? calib.n : 0}/30 paper trades) — confidence capped until proven`);
  }

  out.entryRef = last.close;
  out.reasons = reasons;
  out.timing = {
    nextBoundaryMs: nextBoundaryMs(expiryMin * 60),
    msToBoundary: Math.max(0, nextBoundaryMs(expiryMin * 60) - Date.now()),
  };
  if (conf < MIN_CONF) {
    out.reasons.push(`confidence ${conf} below ${MIN_CONF} floor — WAIT`);
    return out;
  }
  out.decision = dir;
  out.confidence = conf;
  return out;
}

/** Read calibration for a setup bucket. */
export function calibFor(db, setupKey) {
  try {
    const r = db.prepare("SELECT n, wins FROM pocket_calib WHERE bucket = ?").get(setupKey);
    if (!r || r.n <= 0) return null;
    return { n: r.n, wr: r.wins / r.n };
  } catch { return null; }
}

// ---- paper ledger (system scoreboard, user_id "__pocket__" + per-user copies) ----
export function ensurePocketTables(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS pocket_paper (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      symbol TEXT NOT NULL,
      direction TEXT NOT NULL,
      entry_price REAL NOT NULL,
      entry_ts INTEGER NOT NULL,
      expiry_min INTEGER NOT NULL,
      expiry_ts INTEGER NOT NULL,
      settle_price REAL,
      result TEXT DEFAULT 'OPEN',
      payout REAL DEFAULT 0.8,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_pocket_user_sym ON pocket_paper(user_id, symbol, entry_ts DESC);
    CREATE TABLE IF NOT EXISTS pocket_calib (
      bucket TEXT PRIMARY KEY,
      n INTEGER DEFAULT 0,
      wins INTEGER DEFAULT 0,
      updated_at INTEGER
    );
  `);
  // Migration for DBs created before setup_key / calib existed.
  try {
    const cols = db.prepare("PRAGMA table_info(pocket_paper)").all().map(c => c.name);
    if (!cols.includes("setup_key")) db.exec("ALTER TABLE pocket_paper ADD COLUMN setup_key TEXT");
  } catch { /* ignore */ }
}

export function recordPaper(db, { userId, signal, payout }) {
  ensurePocketTables(db);
  const entryTs = Date.now();
  const row = {
    id: `pk_${signal.symbol}_${entryTs}_${Math.random().toString(36).slice(2, 6)}`,
    user_id: userId, symbol: signal.symbol, direction: signal.decision,
    entry_price: signal.entryRef, entry_ts: entryTs,
    expiry_min: signal.expiryMin, expiry_ts: entryTs + signal.expiryMin * 60_000,
    payout,
  };
  try {
    db.prepare(`
      INSERT INTO pocket_paper (id, user_id, symbol, direction, entry_price, entry_ts, expiry_min, expiry_ts, payout, setup_key)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(row.id, row.user_id, row.symbol, row.direction, row.entry_price,
      row.entry_ts, row.expiry_min, row.expiry_ts, row.payout, signal.setupKey || null);
  } catch { /* ignore */ }
  return row;
}

/** Settle expired OPEN rows against the live-feed price. Returns settled count. */
export function settleOpen(db, lastPriceFn) {
  ensurePocketTables(db);
  let rows = [];
  try {
    rows = db.prepare("SELECT * FROM pocket_paper WHERE result = 'OPEN' AND expiry_ts <= ?").all(Date.now());
  } catch { return { settled: 0 }; }
  let settled = 0;
  for (const r of rows) {
    try {
      const live = lastPriceFn(r.symbol);
      if (!live || !Number.isFinite(live.price)) continue;
      const px = live.price;
      let result = "TIE";
      if (px > r.entry_price) result = r.direction === "CALL" ? "WIN" : "LOSS";
      else if (px < r.entry_price) result = r.direction === "CALL" ? "LOSS" : "WIN";
      db.prepare("UPDATE pocket_paper SET settle_price = ?, result = ? WHERE id = ?").run(px, result, r.id);
      settled++;
      // Feed the calibration layer (system scoreboard rows only — per-user
      // copies would multi-count the same outcome).
      if (r.user_id === "__pocket__" && r.setup_key && (result === "WIN" || result === "LOSS")) {
        try {
          db.prepare(`
            INSERT INTO pocket_calib (bucket, n, wins, updated_at) VALUES (?, 1, ?, ?)
            ON CONFLICT(bucket) DO UPDATE SET n = n + 1, wins = wins + ?, updated_at = ?
          `).run(r.setup_key, result === "WIN" ? 1 : 0, Date.now(), result === "WIN" ? 1 : 0, Date.now());
        } catch { /* ignore */ }
      }
    } catch { /* per-row best effort */ }
  }
  return { settled };
}

export function pocketStats(db, { symbol = null, payout = 0.8, limit = 500 } = {}) {
  ensurePocketTables(db);
  let rows = [];
  try {
    rows = symbol
      ? db.prepare("SELECT * FROM pocket_paper WHERE result != 'OPEN' AND symbol = ? ORDER BY entry_ts DESC LIMIT ?").all(symbol.toUpperCase(), limit)
      : db.prepare("SELECT * FROM pocket_paper WHERE result != 'OPEN' ORDER BY entry_ts DESC LIMIT ?").all(limit);
  } catch { /* ignore */ }
  const settled = rows.filter(r => r.result === "WIN" || r.result === "LOSS");
  const wins = settled.filter(r => r.result === "WIN").length;
  const wr = settled.length ? wins / settled.length : null;
  const breakeven = 1 / (1 + payout);
  const expectancy = wr == null ? null : wr * payout - (1 - wr) * 1;
  const bySymbol = {};
  for (const r of settled) {
    const b = (bySymbol[r.symbol] = bySymbol[r.symbol] || { n: 0, wins: 0 });
    b.n++; if (r.result === "WIN") b.wins++;
  }
  for (const k of Object.keys(bySymbol)) {
    bySymbol[k].winRate = bySymbol[k].n ? bySymbol[k].wins / bySymbol[k].n : null;
  }
  let calibration = [];
  try {
    calibration = db.prepare("SELECT bucket, n, wins FROM pocket_calib ORDER BY n DESC LIMIT 20").all()
      .map(r => ({ bucket: r.bucket, n: r.n, winRate: r.n ? r.wins / r.n : null }));
  } catch { /* ignore */ }
  return {
    calibration,
    trades: settled.length, wins, losses: settled.length - wins,
    ties: rows.length - settled.length,
    winRate: wr, payout, breakeven,
    expectancy,
    verdict: wr == null ? "no settled trades yet — paper only"
      : expectancy > 0 ? `POSITIVE expectancy (+${expectancy.toFixed(3)}/trade) at ${(payout * 100).toFixed(0)}% payout`
      : `NEGATIVE expectancy (${expectancy.toFixed(3)}/trade) — needs ${(breakeven * 100).toFixed(1)}% win rate, has ${((wr || 0) * 100).toFixed(1)}%`,
    bySymbol,
  };
}
