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
  // trig need >= 23: the alignment gate reads EMA21 on the trigger TF.
  1: { trendIv: "5m", trigIv: "1m", need: { trend: 40, trig: 25 } },
  5: { trendIv: "15m", trigIv: "5m", need: { trend: 40, trig: 25 } },
  15: { trendIv: "1h", trigIv: "15m", need: { trend: 40, trig: 25 } },
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

// ---- pullback-vs-reversal machinery ----
// Wilder ADX(14) + in-memory percentile ring per symbol: "strong trend" means
// strong FOR THIS SYMBOL relative to its own recent range (a fixed ADX cutoff
// means different things on XAUUSD vs a quiet FX pair).
const ADX_PERIOD = 14;
const _adxRings = new Map(); // sym -> [adx...] capped at 120
function adxLast(candles, period = ADX_PERIOD) {
  const n = candles.length;
  if (n < period * 2 + 2) return null;
  const tr = [], pdm = [], mdm = [];
  for (let i = 1; i < n; i++) {
    const c = candles[i], p = candles[i - 1];
    tr.push(Math.max(c.high - c.low, Math.abs(c.high - p.close), Math.abs(c.low - p.close)));
    const up = c.high - p.high, dn = p.low - c.low;
    pdm.push(up > dn && up > 0 ? up : 0);
    mdm.push(dn > up && dn > 0 ? dn : 0);
  }
  const wilder = (arr) => {
    let s = arr.slice(0, period).reduce((a, b) => a + b, 0);
    const out = [s];
    for (let i = period; i < arr.length; i++) { s = s - s / period + arr[i]; out.push(s); }
    return out;
  };
  const trS = wilder(tr), pS = wilder(pdm), mS = wilder(mdm);
  const dx = trS.map((t, i) => {
    if (t === 0) return 0;
    const dp = (pS[i] / t) * 100, dm = (mS[i] / t) * 100;
    return dp + dm === 0 ? 0 : (Math.abs(dp - dm) / (dp + dm)) * 100;
  });
  if (dx.length < period) return null;
  let a = dx.slice(0, period).reduce((x, y) => x + y, 0) / period;
  for (let i = period; i < dx.length; i++) a = (a * (period - 1) + dx[i]) / period;
  return a;
}
function adxPercentile(sym, adx) {
  let ring = _adxRings.get(sym);
  if (!ring) { ring = []; _adxRings.set(sym, ring); }
  ring.push(adx);
  if (ring.length > 120) ring.splice(0, ring.length - 120);
  // No history yet: neutral prior (50), not weak (0) — a cold start must not
  // masquerade as a weak trend.
  if (ring.length < 5) return 50;
  const below = ring.filter(v => v < adx).length;
  return (below / ring.length) * 100;
}

// Swing pivots on the trend TF (price action, no indicator lag).
function swingLast(candles, lookback = 3) {
  let lastHigh = null, lastLow = null, prevLow = null, prevHigh = null;
  let highs = 0, lows = 0;
  for (let i = candles.length - lookback - 1; i >= lookback; i--) {
    const c = candles[i];
    let isH = true, isL = true;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      // Strict inequality: exact ties (common on smooth/low-vol prints) must
      // not erase structure — only a strictly higher high invalidates.
      if (candles[j].high > c.high) isH = false;
      if (candles[j].low < c.low) isL = false;
    }
    if (isH && highs < 2) { if (highs === 0) lastHigh = c.high; else prevHigh = c.high; highs++; }
    if (isL && lows < 2) { if (lows === 0) lastLow = c.low; else prevLow = c.low; lows++; }
    if (highs >= 2 && lows >= 2) break;
  }
  return { lastHigh, lastLow, prevHigh, prevLow };
}

/**
 * Healthy pullback vs real reversal, assessed on CLOSED candles at the moment
 * the alignment gate fires (trend TF vs trigger TF disagree).
 * @returns { verdict: "HEALTHY"|"REVERSAL"|"UNCERTAIN", retracePct, adxPct, reasons[] }
 */
export function assessPullback(symbol, trend, trig, trendDir /* +1 bull, -1 bear */, atr) {
  const out = { verdict: "UNCERTAIN", retracePct: null, adxPct: null, trendDir, reasons: [] };
  const sym = String(symbol).toUpperCase();
  const sw = swingLast(trend);
  const price = trig[trig.length - 1].close;
  // Retrace depth of the last trend leg (ratio — symbol-agnostic).
  let retrace = null;
  if (trendDir > 0 && sw.prevLow != null && sw.lastHigh != null && sw.lastHigh !== sw.prevLow) {
    retrace = ((sw.lastHigh - price) / (sw.lastHigh - sw.prevLow)) * 100;
  } else if (trendDir < 0 && sw.prevHigh != null && sw.lastLow != null && sw.prevHigh !== sw.lastLow) {
    retrace = ((price - sw.lastLow) / (sw.prevHigh - sw.lastLow)) * 100;
  }
  out.retracePct = retrace;
  // Trend strength, self-relative.
  const adx = adxLast(trend);
  const adxPct = adx != null ? adxPercentile(sym, adx) : null;
  out.adxPct = adxPct;
  // Structure break: close beyond the leg extreme = the leg failed.
  const broke = trendDir > 0
    ? (sw.lastHigh != null && price > sw.lastHigh)
    : (sw.lastLow != null && price < sw.lastLow);
  const lastTrig = trig[trig.length - 1];
  const counterBody = Math.abs(lastTrig.close - lastTrig.open) / (atr || 1);
  const thrust = counterBody > 0.8; // violent counter-candle = reversal energy

  if (broke) {
    out.verdict = "REVERSAL";
    out.reasons.push("structure break: price took out the trend-leg extreme — the leg failed, not a pullback");
    return out;
  }
  if (retrace != null && retrace > 61.8) {
    out.verdict = thrust ? "REVERSAL" : "UNCERTAIN";
    out.reasons.push(`deep retrace ${retrace.toFixed(0)}% past 61.8% — ${thrust ? "with thrust: reversal likely" : "too deep to trust, too intact to call dead"}`);
    return out;
  }
  let score = 0;
  const notes = [];
  if (adxPct != null && adxPct >= 55) { score += 2; notes.push(`trend strong for this pair (ADX pct ${adxPct.toFixed(0)})`); }
  else if (adxPct != null && adxPct <= 35) { score -= 2; notes.push(`trend weak for this pair (ADX pct ${adxPct.toFixed(0)})`); }
  if (retrace != null && retrace <= 50) { score += 2; notes.push(`shallow retrace ${retrace.toFixed(0)}% — profit-taking, not a turn`); }
  else if (retrace == null) { notes.push("no clean swing leg to measure — cannot confirm health"); }
  if (!thrust) { score += 1; notes.push("quiet counter-move (no thrust candle)"); }
  else { score -= 2; notes.push("thrust counter-candle — reversal energy"); }
  out.reasons.push(...notes);
  out.verdict = score >= 3 ? "HEALTHY" : score <= -1 ? "REVERSAL" : "UNCERTAIN";
  return out;
}

// Max trig-candles a pullback arm stays valid, per expiry.
export const PB_MAX_WAIT = { 1: 6, 5: 4, 15: 3 };
export function trigMinutes(trigIv) {
  return { "1m": 1, "5m": 5, "15m": 15 }[trigIv] || 5;
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

  // ALIGNMENT GATE (the counter-trend fix): the trigger timeframe's own
  // EMA9/21 must point the same way as the trend timeframe. A with-trend
  // candle body printed against a counter-trend trigger structure is a
  // pullback bounce — fixed-expiry cannot survive it, so we stand down
  // instead of trading it. Never flip, never chase.
  const tCloses = trig.map(c => c.close);
  const te9 = emaLast(tCloses, EMA_FAST);
  const te21 = emaLast(tCloses, EMA_SLOW);
  if (te9 != null && te21 != null && te9 !== te21) {
    const trigBull = te9 > te21;
    if ((bull && !trigBull) || (bear && trigBull)) {
      // Counter-trend: don't just WAIT — diagnose pullback vs reversal so the
      // caller can arm a completion entry or confirm the turn.
      const pb = assessPullback(sym, trend, trig, sgn, atr);
      out.pullback = pb;
      out.reasons.push(
        `counter-trend: ${prof.trendIv} ${bull ? "bullish" : "bearish"} but ${prof.trigIv} ${trigBull ? "bullish" : "bearish"} — pullback in progress, stand down (never trade the bounce)`,
        `pullback read: ${pb.verdict}${pb.retracePct != null ? ` (retrace ${pb.retracePct.toFixed(0)}%)` : ""}${pb.adxPct != null ? `, trend ADX pct ${pb.adxPct.toFixed(0)}` : ""}`,
        ...pb.reasons
      );
      return out;
    }
    out._aligned = true;
  }

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
    `${prof.trendIv}+${prof.trigIv} aligned — no counter-trend pullback overhead`,
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

/** Pullback arms: HEALTHY-pullback completions waiting for re-alignment. */
export function upsertArm(db, { symbol, expiryMin, direction, setupKey, expiresTs, note }) {
  ensurePocketTables(db);
  try {
    db.prepare(`
      INSERT INTO pocket_arms (symbol, expiry_min, direction, setup_key, armed_ts, expires_ts, note)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(symbol, expiry_min) DO UPDATE SET direction = ?, setup_key = ?, armed_ts = ?, expires_ts = ?, note = ?
    `).run(symbol, expiryMin, direction, setupKey || null, Date.now(), expiresTs, note || null,
      direction, setupKey || null, Date.now(), expiresTs, note || null);
  } catch { /* ignore */ }
}
export function getArm(db, symbol, expiryMin) {
  ensurePocketTables(db);
  try {
    const r = db.prepare("SELECT * FROM pocket_arms WHERE symbol = ? AND expiry_min = ?").get(symbol, expiryMin);
    if (!r) return null;
    if (r.expires_ts <= Date.now()) { clearArm(db, symbol, expiryMin); return null; }
    return r;
  } catch { return null; }
}
export function clearArm(db, symbol, expiryMin) {
  try { db.prepare("DELETE FROM pocket_arms WHERE symbol = ? AND expiry_min = ?").run(symbol, expiryMin); } catch { /* ignore */ }
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
    CREATE TABLE IF NOT EXISTS pocket_arms (
      symbol TEXT NOT NULL,
      expiry_min INTEGER NOT NULL,
      direction TEXT NOT NULL,
      setup_key TEXT,
      armed_ts INTEGER NOT NULL,
      expires_ts INTEGER NOT NULL,
      note TEXT,
      PRIMARY KEY (symbol, expiry_min)
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
