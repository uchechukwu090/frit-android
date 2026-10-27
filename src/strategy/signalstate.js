// ============================================================================
// FRIT SIGNAL LIFECYCLE — one live signal per (user, bot, pair), not one per candle
// ----------------------------------------------------------------------------
// Before: every bot tick re-analysed the market and, while the trend persisted,
// issued a brand-new signal (new entry, new TP/SL, new order) at every cadence.
//
// Now a signal is a thing with a life:
//   NONE      -> a fresh actionable read opens a signal           (FIRE_NEW)
//   ACTIVE    -> the bot keeps re-reading the market, but
//                 - same-direction read or WAIT : HOLD. TP/SL are NOT touched.
//                 - price reached TP or SL      : CLOSE (resolved).
//                 - signal older than maxAge    : CLOSE (expired).
//                 - opposite actionable read    : CLOSE_AND_FIRE (reversal).
//                 - predictive volume engine sees activity about to EXPAND
//                   with the live trend         : RETRIGGER, a new signal with
//                   freshly computed TP/SL. Spaced, capped and never chased
//                   once most of the target has already been consumed.
// Pure decision logic (decideLifecycle) is separate from storage so it is
// unit-tested without a database.
// ============================================================================

export function ensureSignalStateTable(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS signal_state (
      key TEXT PRIMARY KEY,
      user_id TEXT, bot TEXT, pair TEXT,
      dir TEXT, entry REAL, sl REAL, tp REAL, rr REAL,
      status TEXT DEFAULT 'ACTIVE',
      opened_at INTEGER, last_trigger_at INTEGER, retriggers INTEGER DEFAULT 0,
      placed INTEGER DEFAULT 0,
      last_seen_at INTEGER, last_price REAL,
      closed_at INTEGER, close_reason TEXT
    );
  `);
  // Older rows pre-date the `placed` flag (was this table ever migrated).
  try {
    const cols = db.prepare("PRAGMA table_info(signal_state)").all().map((c) => c.name);
    if (!cols.includes("placed")) db.exec("ALTER TABLE signal_state ADD COLUMN placed INTEGER DEFAULT 0");
  } catch { /* best-effort */ }
}

export function stateKey(userId, bot, pair) {
  return `${userId}:${bot}:${String(pair).toUpperCase()}`;
}

export function getActiveState(db, key) {
  try {
    ensureSignalStateTable(db);
    return db.prepare("SELECT * FROM signal_state WHERE key = ? AND status = 'ACTIVE'").get(key) || null;
  } catch { return null; }
}

export function openState(db, { key, userId, bot, pair, dir, entry, sl, tp, rr = null, now = Date.now(), retriggers = 0, placed = 0 }) {
  ensureSignalStateTable(db);
  db.prepare(`
    INSERT INTO signal_state (key, user_id, bot, pair, dir, entry, sl, tp, rr, status, opened_at, last_trigger_at, retriggers, placed, last_seen_at, last_price, closed_at, close_reason)
    VALUES (?,?,?,?,?,?,?,?,?, 'ACTIVE', ?,?,?,?,?,?, NULL, NULL)
    ON CONFLICT(key) DO UPDATE SET
      dir=excluded.dir, entry=excluded.entry, sl=excluded.sl, tp=excluded.tp, rr=excluded.rr,
      status='ACTIVE', opened_at=excluded.opened_at, last_trigger_at=excluded.last_trigger_at,
      retriggers=excluded.retriggers, placed=excluded.placed,
      last_seen_at=excluded.last_seen_at, last_price=excluded.last_price,
      closed_at=NULL, close_reason=NULL
  `).run(key, userId, bot, String(pair).toUpperCase(), dir, entry ?? null, sl ?? null, tp ?? null, rr,
    now, now, retriggers, placed, now, entry ?? null);
}

/**
 * An order for this live signal is now open at the broker. A later RETRIGGER
 * refreshes the signal's levels but must NOT send a second order for the same
 * signal — that would double the exposure instead of managing it.
 */
export function markPlaced(db, key) {
  try {
    ensureSignalStateTable(db);
    db.prepare("UPDATE signal_state SET placed = 1 WHERE key = ? AND status = 'ACTIVE'").run(key);
  } catch { /* best-effort */ }
}

/** A retrigger replaces the levels but keeps the original open time. */
export function retriggerState(db, key, { entry, sl, tp, rr = null, now = Date.now() }) {
  ensureSignalStateTable(db);
  db.prepare(`
    UPDATE signal_state SET entry=?, sl=?, tp=?, rr=?, last_trigger_at=?, retriggers=retriggers+1,
      last_seen_at=?, last_price=? WHERE key=? AND status='ACTIVE'
  `).run(entry ?? null, sl ?? null, tp ?? null, rr, now, now, entry ?? null, key);
}

export function touchState(db, key, { price = null, now = Date.now() } = {}) {
  try {
    db.prepare("UPDATE signal_state SET last_seen_at=?, last_price=COALESCE(?, last_price) WHERE key=? AND status='ACTIVE'")
      .run(now, price, key);
  } catch { /* best-effort */ }
}

export function closeState(db, key, reason, now = Date.now()) {
  try {
    db.prepare("UPDATE signal_state SET status='CLOSED', closed_at=?, close_reason=? WHERE key=? AND status='ACTIVE'")
      .run(now, String(reason || "").slice(0, 200), key);
  } catch { /* best-effort */ }
}

export function listActiveStates(db, userId = null) {
  try {
    ensureSignalStateTable(db);
    return userId
      ? db.prepare("SELECT * FROM signal_state WHERE status='ACTIVE' AND user_id = ?").all(userId)
      : db.prepare("SELECT * FROM signal_state WHERE status='ACTIVE'").all();
  } catch { return []; }
}

export const LIFECYCLE_DEFAULTS = {
  maxAgeMs: 12 * 3_600_000,        // intraday signal lives at most 12h
  minRetriggerGapMs: 15 * 60_000,  // >= three 5m bars between triggers
  maxRetriggers: 2,                // per live signal
  chaseLimit: 0.7,                 // no retrigger once 70% of the TP distance is used
};

/**
 * Pure lifecycle decision.
 * @param state     active signal row, or null
 * @param analysis  { decision: "BUY"|"SELL"|"WAIT"|..., confidence }
 * @param minConf   confidence floor for an actionable fresh read
 * @param price     latest price (null if unknown)
 * @param vf        volume-forecast result (or null)
 * @returns { action, reason }
 *   action: NONE | FIRE_NEW | HOLD | RETRIGGER | CLOSE | CLOSE_AND_FIRE
 */
export function decideLifecycle({ state, analysis, minConf = 60, price = null, vf = null, now = Date.now(), cfg = {} }) {
  const c = { ...LIFECYCLE_DEFAULTS, ...cfg };
  const dec = String(analysis?.decision || "WAIT").toUpperCase();
  const dirOf = (d) => (d === "BUY" || d === "CALL" ? "BUY" : d === "SELL" || d === "PUT" ? "SELL" : null);
  const readDir = dirOf(dec);
  const actionable = !!readDir && Number(analysis?.confidence) >= minConf;

  if (!state) {
    return actionable
      ? { action: "FIRE_NEW", reason: "fresh actionable read, no live signal" }
      : { action: "NONE", reason: "no setup" };
  }

  const buy = state.dir === "BUY";
  // Number(null) === 0, so a MISSING price must be filtered before it can look
  // like 0 — otherwise every BUY reads as "SL hit at 0" and every SELL as
  // "TP hit at 0", and a cold feed would close every live signal.
  const p = price == null || price === "" || !(Number(price) > 0) ? null : Number(price);

  if (p != null && state.tp != null && state.sl != null) {
    const hitTp = buy ? p >= state.tp : p <= state.tp;
    const hitSl = buy ? p <= state.sl : p >= state.sl;
    if (hitTp || hitSl) {
      const why = hitTp ? "take-profit reached" : "stop-loss reached";
      return actionable && readDir === state.dir
        ? { action: "CLOSE_AND_FIRE", reason: `${why}; trend still actionable — new signal` }
        : { action: "CLOSE", reason: why };
    }
  }

  if (now - Number(state.opened_at) > c.maxAgeMs) {
    return actionable && readDir === state.dir
      ? { action: "CLOSE_AND_FIRE", reason: "signal expired; trend still actionable — new signal" }
      : { action: "CLOSE", reason: "signal expired" };
  }

  if (actionable && readDir !== state.dir) {
    return { action: "CLOSE_AND_FIRE", reason: `opposite ${readDir} signal — reversal` };
  }

  // ---- same direction (or WAIT): hold unless volume says a fresh leg is coming ----
  const gapOk = now - Number(state.last_trigger_at || state.opened_at) >= c.minRetriggerGapMs;
  const capOk = Number(state.retriggers || 0) < c.maxRetriggers;
  if (vf?.available && vf.expansion && vf.direction === state.dir && gapOk && capOk) {
    if (p != null && state.tp != null && state.entry != null) {
      const total = Math.abs(state.tp - state.entry);
      const used = buy ? p - state.entry : state.entry - p;
      if (total > 0 && used / total > c.chaseLimit) {
        return { action: "HOLD", reason: "volume expansion seen but the target is mostly consumed — not chasing" };
      }
    }
    return {
      action: "RETRIGGER",
      reason: `volume expansion forecast with the ${state.dir} trend (score ${vf.score}) — fresh TP/SL`,
    };
  }

  return {
    action: "HOLD",
    reason: actionable
      ? `holding live ${state.dir} signal — TP/SL unchanged`
      : `holding live ${state.dir} signal — read is ${dec}, waiting`,
  };
}
