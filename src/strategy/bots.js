// ============================================================================
// FRIT BOTS REGISTRY — scalping / pocket / intraday / longterm / options
// ----------------------------------------------------------------------------
// Each bot owns: pair, timeframe, auto toggle, confidence floor, daily cap.
// The runner ticks every 60s but each bot fires on its own cadence.
//
// BOT TYPES:
//   scalping  — TP/SL based, small timeframe (1m/5m/15m). Like the main
//                engine but on a smaller TF. Auto places via Deriv MULT.
//   pocket    — True pocket option: predicts if a candle will end bullish
//                or bearish within a given time. Auto places via Deriv
//                Rise/Fall (CALL/PUT).
//   intraday  — 30M/4H MTF engine (trend + pullback-Fib + range fade).
//   longterm  — Daily Donchian-55 breakout system for investors.
//   options   — Credit spreads on SPY/QQQ + FX expected-move levels.
//                Signal-only: auto = scan + notify + paper ledger.
//
// Timeframes are validated per bot. Note the honest mapping: the intraday
// engine analyzes 30M/4H structure regardless of the selected display TF —
// the TF picks monitoring granularity, not a retuned engine.
//
// SIGNAL LIFECYCLE (directional bots): the engine still re-reads the market
// on every cadence, but a re-read does NOT re-fire. A live signal keeps its
// entry/TP/SL until it resolves (TP/SL/expiry/reversal), and the only thing
// that earns a fresh entry + fresh TP/SL mid-signal is the predictive volume
// engine forecasting an activity expansion WITH the trend. See
// signalstate.js (the decision) and volumeforecast.js (the forecaster).
// ============================================================================

import { getActiveState, openState, retriggerState, closeState, markPlaced, decideLifecycle, stateKey } from "./signalstate.js";
import { forecastVolume } from "./volumeforecast.js";
import { applyCostsFor } from "./costs.js";
import { isSynthetic } from "./synthetics.js";

// Pocket settles by fixed expiry and options never trades — neither has a
// TP/SL signal to hold open, so only these three run the lifecycle.
const LIFECYCLE_BOTS = new Set(["scalping", "intraday", "longterm"]);

/** Latest print from the live feed, or null when it has no data for the pair. */
function livePrice(deps, pair) {
  try {
    const p = Number(deps.lastPrice?.(pair));
    return Number.isFinite(p) && p > 0 ? p : null;
  } catch { return null; }
}

/**
 * Predictive volume forecast for a live signal.
 * 5m is the FORECAST timeframe (does activity expand over the next 1-3 bars?),
 * 1m only measures the pace inside the current 5m window. The local feed
 * store is asked first — free, and it holds the live edge; REST is the
 * fallback for a cold store. Any failure returns null and the lifecycle then
 * treats "no forecast" as "no expansion": a retrigger is never granted on
 * missing data.
 */
async function volumeForecast(deps, pair, trend) {
  try {
    const store = deps.storeCandles;
    let c5 = store ? store(pair, "5m", 300) : null;
    if (!Array.isArray(c5) || c5.length < 60) c5 = await deps.fetchCandles?.(pair, "5m");
    if (!Array.isArray(c5) || c5.length < 60) return null;
    let c1 = store ? store(pair, "1m", 300) : null;
    if (!Array.isArray(c1) || c1.length < 25) c1 = await deps.fetchCandles?.(pair, "1m");
    return forecastVolume({ c5, c1: Array.isArray(c1) ? c1 : [], trend });
  } catch (e) {
    deps.log?.("volume forecast failed:", e.message);
    return null;
  }
}

export const BOT_DEFS = {
  scalping: {
    label: "Scalping bot",
    timeframes: ["1m", "5m", "15m"],
    defaultTimeframe: "5m",
    intervalMin: 5,
    executes: true,
    blurb: "TP/SL scalping on small timeframes (1m/5m/15m). Like the main engine but faster. Auto places via Deriv MULT contracts.",
  },
  pocket: {
    label: "Pocket option bot",
    timeframes: ["1m", "5m", "15m"],
    defaultTimeframe: "5m",
    intervalMin: 5,
    executes: true,
    blurb: "Predicts if a candle will end bullish or bearish within a given time. Auto places via Deriv Rise/Fall (CALL/PUT).",
  },
  intraday: {
    label: "Intraday bot",
    timeframes: ["1m", "5m", "15m", "30m", "1H", "4H"],
    defaultTimeframe: "30m",
    intervalMin: 30,
    executes: true,
    blurb: "30M/4H MTF engine (trend + pullback-Fib + range fade). Auto places via the MT5 bridge when enabled — paper while the bridge is unset.",
  },
  longterm: {
    label: "Long-term bot",
    timeframes: ["1D", "1W"],
    defaultTimeframe: "1D",
    intervalMin: 1440,
    executes: true,
    blurb: "Daily Donchian-55 breakout system (200DMA regime, 2xATR stop, 4xATR target) for investors. Auto places via the MT5 bridge when enabled.",
  },
  options: {
    label: "Options bot",
    timeframes: ["1D"],
    defaultTimeframe: "1D",
    intervalMin: 240,
    executes: false,
    blurb: "Credit spreads on SPY/QQQ + FX expected-move levels. Signal-only: auto = scan + notify + paper ledger. Never places orders.",
  },
};

export function ensureBotsTables(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS bots (
      name TEXT PRIMARY KEY,
      pair TEXT NOT NULL DEFAULT '',
      timeframe TEXT NOT NULL DEFAULT '',
      auto_enabled INTEGER DEFAULT 0,
      min_confidence INTEGER DEFAULT 60,
      max_trades_day INTEGER DEFAULT 2,
      last_run INTEGER,
      updated_at INTEGER
    );
    CREATE TABLE IF NOT EXISTS bot_runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      bot TEXT NOT NULL,
      pair TEXT, timeframe TEXT,
      fired_at INTEGER,
      decision TEXT, confidence INTEGER, note TEXT,
      user_id TEXT DEFAULT 'bot',
      placed INTEGER DEFAULT 0
    );
  `);
  // Per-user attribution: which user's bot produced this run, and whether the
  // run actually PLACED a trade (only placed trades count against the cap).
  try {
    const cols = db.prepare("PRAGMA table_info(bot_runs)").all().map(c => c.name);
    if (!cols.includes("user_id")) db.exec("ALTER TABLE bot_runs ADD COLUMN user_id TEXT DEFAULT 'bot'");
    if (!cols.includes("placed")) db.exec("ALTER TABLE bot_runs ADD COLUMN placed INTEGER DEFAULT 0");
  } catch { /* best-effort */ }
  for (const [name, def] of Object.entries(BOT_DEFS)) {
    try {
      db.prepare(
        "INSERT OR IGNORE INTO bots (name, pair, timeframe, auto_enabled, min_confidence, max_trades_day) VALUES (?,?,?,?,?,?)"
      ).run(name, "", def.defaultTimeframe, 0, 60, 2);
    } catch { /* ignore */ }
  }
}

export function getBots(db) {
  ensureBotsTables(db);
  const rows = db.prepare("SELECT * FROM bots").all();
  const byName = new Map(rows.map((b) => [b.name, b]));
  // Return in BOT_DEFS order (scalping, pocket, intraday, longterm, options)
  // so the UI always shows the same sequence — a plain SELECT has no order.
  return Object.keys(BOT_DEFS).map((name) => {
    const b = byName.get(name);
    if (!b) return null;
    return {
      name: b.name,
      label: BOT_DEFS[b.name]?.label || b.name,
      pair: b.pair || "",
      timeframe: b.timeframe || BOT_DEFS[b.name]?.defaultTimeframe || "",
      auto_enabled: b.auto_enabled === 1,
      min_confidence: b.min_confidence ?? 60,
      max_trades_day: b.max_trades_day ?? 2,
      intervalMin: BOT_DEFS[b.name]?.intervalMin ?? 60,
      executes: BOT_DEFS[b.name]?.executes ?? false,
      blurb: BOT_DEFS[b.name]?.blurb || "",
      last_run: b.last_run || null,
    };
  }).filter(Boolean);
}

export function saveBot(db, name, cfg = {}) {
  ensureBotsTables(db);
  const def = BOT_DEFS[name];
  if (!def) throw new Error(`unknown bot: ${name}`);
  const pair = String(cfg.pair || "").toUpperCase().replace(/[^A-Z0-9_.]/g, "").slice(0, 12);
  if (!pair) throw new Error("pair required (e.g. XAUUSD, EURUSD, BTCUSD, SPY, R_100)");
  const timeframe = String(cfg.timeframe || def.defaultTimeframe);
  if (!def.timeframes.includes(timeframe)) {
    throw new Error(`${name} supports timeframes: ${def.timeframes.join(", ")}`);
  }
  const min_confidence = Math.max(0, Math.min(100, Number(cfg.min_confidence ?? 60)));
  const max_trades_day = Math.max(1, Math.min(10, Number(cfg.max_trades_day ?? 2)));
  const auto_enabled = cfg.auto_enabled ? 1 : 0;
  db.prepare(`
    UPDATE bots SET pair = ?, timeframe = ?, auto_enabled = ?,
      min_confidence = ?, max_trades_day = ?, updated_at = ? WHERE name = ?
  `).run(pair, timeframe, auto_enabled, min_confidence, max_trades_day, Date.now(), name);
  return getBots(db).find((b) => b.name === name);
}

function runsToday(db, name) {
  try {
    // Only PLACED trades count against the daily cap. A signal that fired
    // below the confidence floor (or a signal-only bot) must not consume
    // the user's daily allowance.
    return db.prepare(
      "SELECT COUNT(*) AS c FROM bot_runs WHERE bot = ? AND date(fired_at/1000,'unixepoch') = date('now') AND placed = 1 AND decision IN ('BUY','SELL','CALL','PUT')"
    ).get(name)?.c || 0;
  } catch { return 0; }
}

/** Trades today for ONE user on ONE bot+pair (the per-pair daily cap). */
function runsTodayFor(db, name, pair, userId) {
  try {
    return db.prepare(
      "SELECT COUNT(*) AS c FROM bot_runs WHERE bot = ? AND pair = ? AND user_id = ? AND date(fired_at/1000,'unixepoch') = date('now') AND placed = 1 AND decision IN ('BUY','SELL','CALL','PUT')"
    ).get(name, String(pair).toUpperCase(), userId)?.c || 0;
  } catch { return 0; }
}

/**
 * Per-bot-pair cadence guard. Returns true when the bot may run now, and
 * records the run. Kept in the DB so a server restart cannot make every bot
 * fire at once.
 */
export function botCooldown(db, key, intervalMin) {
  try {
    db.exec("CREATE TABLE IF NOT EXISTS bot_cooldowns (key TEXT PRIMARY KEY, last_run INTEGER)");
    const row = db.prepare("SELECT last_run FROM bot_cooldowns WHERE key = ?").get(key);
    const now = Date.now();
    if (row?.last_run && now - row.last_run < intervalMin * 60_000) return false;
    db.prepare("INSERT INTO bot_cooldowns (key, last_run) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET last_run = ?")
      .run(key, now, now);
    return true;
  } catch { return true; }
}

function logRun(db, bot, pair, timeframe, decision, confidence, note, userId = "bot", placed = 0) {
  try {
    db.prepare(
      "INSERT INTO bot_runs (bot, pair, timeframe, fired_at, decision, confidence, note, user_id, placed) VALUES (?,?,?,?,?,?,?,?,?)"
    ).run(bot, pair, timeframe, Date.now(), decision, confidence ?? 0, String(note || "").slice(0, 500), userId, placed ? 1 : 0);
  } catch { /* best-effort */ }
}

/** Aggregate daily closes -> weekly (5/session groups). FX/crypto trade ~daily. */
function toWeekly(daily) {
  const out = [];
  for (let i = 0; i < daily.length; i += 5) {
    const chunk = daily.slice(i, i + 5);
    out.push(chunk[chunk.length - 1]);
  }
  return out;
}

/** Adapt an analyzeSynth() result to the signal shape the runner executes. */
export function synthToSignal(pair, r) {
  return {
    id: `sig_${pair}_${Date.now()}_sy`,
    symbol: pair,
    decision: ["BUY", "SELL"].includes(r.decision) ? r.decision : "WAIT",
    confidence: r.confidence || 0,
    entry: r.entry, sl: r.sl, tp1: r.tp, tp2: null,
    rr: r.rr ?? null, scenario: "SYNTH",
    reasoning: (r.reasons || []).join(" | ") + (r.maxHoldMin ? ` | time stop ${r.maxHoldMin}min` : ""),
    timestamp: Date.now(),
  };
}

/**
 * One synthetic pair, read by the GENERATOR-AWARE engine.
 *
 * Every directional bot routes a synthetic here — scalping and pocket used to
 * run plain EMA9/21/RSI/ATR on an engineered RNG market, which is a coin flip
 * minus costs no matter how good the signal looks. The engine reads what the
 * generator actually does (z-fade for the constant-vol streams, post-spike
 * reversion for BOOM/CRASH, the locked drift for RDBULL/RDBEAR).
 *
 * Data honesty: the SPIKE family is documented as a TICK detector (a move of
 * >8x the median TICK). It is fed ticks when the caller can supply them; on a
 * 1m-bar fallback it still detects a spike, but of a 1-MINUTE move — so the
 * signal says which series it read instead of pretending they are the same.
 */
async function syntheticRoute(deps, pair, botName) {
  let ticks = null;
  if (typeof deps.recentTicks === "function") {
    try { ticks = await deps.recentTicks(pair, 300); } catch { ticks = null; }
  }
  const hasTicks = Array.isArray(ticks) && ticks.length >= 60;
  const series = hasTicks ? ticks : await deps.fetchCloses(pair, "1m");
  const source = hasTicks ? "ticks" : "1m bars";
  const r = await deps.analyzeSynth(pair, series);
  const sig = synthToSignal(pair, r);
  const family = r.family || null;
  const notes = [];
  if (family === "SPIKE" && !hasTicks) {
    notes.push(`spike read taken from ${source} (tick feed unavailable for this pair) — coarser than the tick-level detector`);
  }
  if (family === "UNKNOWN") {
    notes.push("unknown generator — the engine will not trade a process it cannot identify");
  }
  // Honest label, taken from the executor's own rule (deriv.js RNG_SYNTH): an
  // RNG stream is refused for real-money Rise/Fall and can only ever execute on
  // a demo account. Saying so on the signal is the difference between a demo
  // study and a promise the engine will not keep.
  notes.push("generated series: execution is demo-only — this app refuses RNG streams on a real-money account");

  // A generator read has no calibration history of its own (the pocket
  // calibration buckets are keyed by the EMA setup), so it is held to the same
  // "learning mode" ceiling the calibrated engine applies until a setup has 30
  // measured paper trades behind it. Never publish an unproven read as a
  // high-confidence one.
  const LEARNING_CAP = 65;
  const confidence = Math.min(Number(sig.confidence) || 0, LEARNING_CAP);
  return {
    ...sig,
    confidence,
    scenario: `SYNTH_${family || "?"}`,
    family,
    source,
    uncalibrated: true,
    reasoning: [sig.reasoning, ...notes].filter(Boolean).join(" | "),
  };
}

/**
 * One runner tick. Deps injected from index.js:
 * { db, mtfAnalyze(symbol)->result, analyzeSynth(symbol, closes)->result,
 *   scalpingAnalyze(symbol, closes, tf)->result, pocketScan(symbol, expiry)->result,
 *   longtermAnalyze(symbol, closes)->result, fetchDaily(symbol)->closes[],
 *   fetchCloses(symbol, interval)->closes[], getChain, scanSymbol, scanFx,
 *   storeSignal(userId, signal), formatSignal(symbol, result),
 *   storeOptionsSignal(userId, signal), openPaper(userId, signal),
 *   executor({symbol, action, lotSize, entry, sl, tp, reason, userId})->result,
 *   notifyOptions(userId, signal)->void, log }
 *
 * TWO SOURCES OF WORK:
 *   1. PER-USER BOT-PAIRS (the modern path). Each row is one user's chosen
 *      bot on one of their pairs, with its own timeframe and its own
 *      trade/signal switches. These run FOR THAT USER: the signal is stored
 *      under their user_id and the trade is sized against THEIR balance.
 *   2. LEGACY GLOBAL BOTS (the `bots` table). Still honoured for installs
 *      configured before the per-pair model existed.
 */
export async function tickBots(deps) {
  const { db } = deps;
  ensureBotsTables(db);
  ensureBotPairTables(db);
  const now = Date.now();
  const results = [];

  // ---------- 1. Per-user bot-pairs ----------
  let pairRows = [];
  try {
    pairRows = db.prepare("SELECT * FROM bot_pairs").all();
  } catch { pairRows = []; }

  for (const row of pairRows) {
    const def = BOT_DEFS[row.bot_name];
    if (!def) continue;
    const userId = row.user_id;
    const pair = String(row.pair || "").toUpperCase();
    if (!pair) continue;
    const tradeEnabled = row.trade_enabled === 1;
    const signalEnabled = row.signal_enabled === 1;
    // Both switches off = the user stopped this bot for this pair entirely.
    if (!tradeEnabled && !signalEnabled) continue;
    const minConf = row.min_confidence ?? 60;
    // Cadence guard: one bot-pair fires at most once per its interval.
    const lastKey = `bp:${userId}:${row.bot_name}:${pair}`;
    if (!deps.botCooldown(db, lastKey, def.intervalMin)) continue;
    if (tradeEnabled && runsTodayFor(db, row.bot_name, pair, userId) >= (row.max_trades_day ?? 2)) continue;

    try {
      const out = await runOneBot(deps, {
        botName: row.bot_name, pair, timeframe: row.timeframe || def.defaultTimeframe,
        userId, minConf, tradeEnabled, signalEnabled,
      });
      for (const r of out.results) results.push({ ...r, userId, pair });
      if (out.decision) {
        // placed=1 only when an order was actually sent — that is what the
        // daily cap counts. A signal that fired below the confidence floor
        // must not eat the user's daily allowance.
        logRun(db, row.bot_name, pair, row.timeframe, out.decision, out.confidence || 0,
          `${userId} ${out.note || "fired"}`, userId, out.traded ? 1 : 0);
      }
    } catch (e) {
      logRun(db, row.bot_name, pair, row.timeframe, "ERROR", 0, e.message, userId, 0);
      results.push({ bot: row.bot_name, decision: "ERROR", note: e.message, userId, pair });
    }
  }

  // ---------- 2. Legacy global bots ----------
  for (const bot of getBots(db)) {
    const def = BOT_DEFS[bot.name];
    if (!bot.auto_enabled || !bot.pair) continue;
    if (bot.last_run && now - bot.last_run < def.intervalMin * 60_000) continue;
    if (runsToday(db, bot.name) >= bot.max_trades_day) continue;
    db.prepare("UPDATE bots SET last_run = ? WHERE name = ?").run(now, bot.name);
    try {
      const out = await runOneBot(deps, {
        botName: bot.name, pair: bot.pair, timeframe: bot.timeframe,
        userId: "bot", minConf: bot.min_confidence,
        tradeEnabled: bot.auto_enabled, signalEnabled: true, legacy: true,
      });
      for (const r of out.results) results.push(r);
      // The legacy global bot must log its run too — without this the daily
      // cap never incremented and the bot traded without limit.
      if (out.decision) {
        logRun(db, bot.name, bot.pair, bot.timeframe, out.decision, out.confidence || 0,
          out.note || "fired", "bot", out.traded ? 1 : 0);
      }
    } catch (e) {
      logRun(db, bot.name, bot.pair, bot.timeframe, "ERROR", 0, e.message);
      results.push({ bot: bot.name, decision: "ERROR", note: e.message });
    }
  }
  return results;
}

/**
 * Run a single bot for a single pair. Returns { decision, confidence, results[] }.
 * `tradeEnabled` controls whether an order is placed; `signalEnabled` controls
 * whether the signal is stored/notified. Both can be on at once.
 */
async function runOneBot(deps, { botName, pair, timeframe, userId, minConf, tradeEnabled, signalEnabled, legacy }) {
  const { db } = deps;
  const out = { decision: "WAIT", confidence: 0, note: "", results: [], traded: false };
  const def = BOT_DEFS[botName];

  if (botName === "options") {
    // Scan + notify + paper only. NEVER the executor — no options rail.
    const isFxStyle = /USD$|XAU|XAG|JPY|GBP|EUR|CHF|CAD|AUD|NZD/.test(pair) && !/^(SPY|QQQ|IWM|XSP)$/.test(pair);
    if (isFxStyle) {
      const closes = await deps.fetchDaily(pair);
      const scan = deps.scanFx({ symbol: pair, closes });
      if (scan.signal) {
        deps.storeOptionsSignal(userId, scan.signal);
        deps.notifyOptions?.(userId, scan.signal);
      }
      logRun(db, botName, pair, timeframe, scan.signal ? "FX_LEVELS" : "SKIP", 0, scan.signal ? scan.signal.id : scan.why, userId);
      out.decision = scan.signal ? "FX_LEVELS" : "SKIP";
      out.note = scan.signal ? scan.signal.id : scan.why;
      out.results.push({ bot: botName, decision: out.decision });
      return out;
    }
    const chain = await deps.getChain(pair);
    const closes = await deps.fetchDaily(pair);
    const scan = deps.scanSymbol({ symbol: pair, chain, closes, db });
    if (scan.signal) {
      deps.storeOptionsSignal(userId, scan.signal);
      deps.openPaper?.(userId, { ...scan.signal, entrySpot: chain.underlying });
      deps.notifyOptions?.(userId, scan.signal);
    }
    logRun(db, botName, pair, timeframe, scan.signal ? scan.signal.kind : "SKIP", 0, scan.signal ? scan.signal.id : scan.why, userId);
    out.decision = scan.signal ? scan.signal.kind : "SKIP";
    out.note = scan.signal ? scan.signal.id : scan.why;
    out.results.push({ bot: botName, decision: out.decision });
    return out;
  }

  // ---- analyse ----
  let signal = null;
  if (botName === "scalping") {
    if (isSynthetic(pair)) {
      // An EMA9/21 crossover on an engineered RNG stream is a coin flip minus
      // costs — the generator has no order flow to read. Every directional
      // bot now routes a synthetic to the generator-aware engine, exactly as
      // intraday already did, so the FX logic never runs on an RNG market.
      signal = await syntheticRoute(deps, pair, "scalping");
    } else {
      const candles = await deps.fetchCandles(pair, timeframe);
      if (!candles || candles.length < 30) {
        out.note = "warming up — need 30+ candles";
        out.results.push({ bot: botName, decision: "WAIT", note: out.note });
        return out;
      }
      // The same volume-matrix filter the intraday engine runs: the matrix is
      // computed from the candle store and cached, so passing it here costs
      // nothing and stops a scalp from fighting strong opposing flow.
      const matrixBias = await deps.matrixBiasFor?.(pair);
      const r = deps.scalpingAnalyze(pair, candles, timeframe, { matrixBias });
      signal = {
        id: `sig_${pair}_${Date.now()}_sc`, symbol: pair, decision: r.decision,
        confidence: r.confidence, entry: r.entry, sl: r.sl, tp1: r.tp1, tp2: r.tp2,
        rr: r.rr, rrGross: r.rrGross ?? null, costs: r.costs ?? null,
        scenario: "SCALPING", reasoning: (r.reasons || []).join(" | "),
        timestamp: Date.now(), botSource: "scalping",
      };
    }
  } else if (botName === "pocket") {
    // TRUE pocket option: predicts if the candle ends bullish or bearish
    // within the given time. Expiry follows the chosen timeframe.
    const expiryMin = timeframe === "1m" ? 1 : timeframe === "5m" ? 5 : 15;
    if (isSynthetic(pair)) {
      // Fixed-expiry Rise/Fall on an RNG stream must be read from the
      // GENERATOR, not from an EMA cross. The synthetic engine's direction
      // maps straight onto CALL/PUT (its families carry real structure: a
      // z-fade, a post-spike reversion, or a generator-locked drift).
      const route = await syntheticRoute(deps, pair, "pocket");
      const dir = route.decision === "BUY" ? "CALL" : route.decision === "SELL" ? "PUT" : "WAIT";
      signal = {
        ...route,
        id: `sig_${pair}_${Date.now()}_pksy`, decision: dir,
        sl: null, tp1: null, tp2: null, rr: null,
        scenario: `POCKET_SYNTH_${route.family || "?"}`,
        reasoning: `${route.reasoning} | expiry ${expiryMin}m on the generator read, not an EMA cross`,
        botSource: "pocket", expiryMin,
      };
    } else {
      const r = deps.pocketScan(pair, expiryMin);
      signal = {
        id: `sig_${pair}_${Date.now()}_pk`, symbol: pair, decision: r.decision,
        confidence: r.confidence, entry: r.entryRef, sl: null, tp1: null, tp2: null,
        rr: null, scenario: "POCKET", reasoning: (r.reasons || []).join(" | "),
        timestamp: Date.now(), botSource: "pocket", expiryMin,
      };
    }
  } else if (botName === "longterm") {
    const daily = await deps.fetchDaily(pair);
    const closes = timeframe === "1W" ? toWeekly(daily) : daily;
    const r = deps.longtermAnalyze(pair, closes);
    signal = {
      id: `sig_${pair}_${Date.now()}_lt`, symbol: pair, decision: r.decision,
      confidence: r.confidence, entry: r.entry, sl: r.sl, tp1: r.tp, tp2: null,
      rr: r.rr, scenario: "LONGTERM", reasoning: r.reasons.join(" | "),
      timestamp: Date.now(), botSource: "longterm",
    };
  } else {
    // intraday (and any other directional bot)
    const isSynth = isSynthetic(pair);
    const r = isSynth
      ? await syntheticRoute(deps, pair, "intraday")
      : await deps.mtfAnalyze(pair, {});
    const formatted = isSynth ? r : deps.formatSignal(pair, r);
    signal = { ...formatted, id: `sig_${pair}_${Date.now()}_${isSynth ? "sy" : "in"}`, botSource: "intraday" };
  }

  const actionable = ["BUY", "SELL", "CALL", "PUT"].includes(signal.decision)
    && signal.confidence >= minConf;

  // ---- LIFECYCLE DECISION (directional bots) ------------------------------
  // The read above is recomputed on every cadence — that is the "recomputing
  // entry and the threshold" part. Whether it becomes a SIGNAL is decided
  // here, against whatever is already live for this (user, bot, pair).
  const lifecycle = LIFECYCLE_BOTS.has(botName);
  const key = stateKey(userId, botName, pair);
  const state = lifecycle ? getActiveState(db, key) : null;
  let vf = null;
  let price = null;
  if (state) {
    // Forecast against the LIVE signal's direction: "will activity expand
    // with the trend I am already in?" — not a free-floating volume guess.
    vf = await volumeForecast(deps, pair, state.dir);
    price = livePrice(deps, pair) ?? vf?.price ?? null;
  }
  const life = lifecycle
    ? decideLifecycle({
        state,
        analysis: { decision: signal.decision, confidence: signal.confidence },
        minConf, price, vf,
      })
    : { action: actionable ? "FIRE_NEW" : "NONE", reason: "" };

  const noteOnly = (decision, note) => {
    out.decision = decision;
    out.confidence = signal.confidence;
    out.note = note;
    out.results.push({ bot: botName, decision, confidence: signal.confidence, note });
    return out;
  };

  let mode = life.action === "RETRIGGER" ? "RETRIGGER" : "FIRE_NEW";
  if (lifecycle) {
    if (life.action === "NONE") return noteOnly(signal.decision, life.reason);
    if (life.action === "HOLD") return noteOnly("HOLD", life.reason);
    if (life.action === "CLOSE") {
      closeState(db, key, life.reason);
      // TP/SL hit or expiry: the signal is resolved. Fire only when the read
      // is actionable AND the other direction — a WAIT just ends the cycle.
      if (!actionable) return noteOnly(signal.decision, `closed — ${life.reason}`);
    } else if (life.action === "CLOSE_AND_FIRE") {
      closeState(db, key, life.reason);
    }
    if (mode === "RETRIGGER" && !vf?.levels) mode = "FIRE_NEW"; // defensive
  } else if (!actionable) {
    out.decision = signal.decision;
    out.confidence = signal.confidence;
    out.note = "below confidence floor or no setup";
    // A non-actionable read can still be stored when the user asked for
    // signals — but only the actionable ones are worth the history noise.
    out.results.push({ bot: botName, decision: signal.decision, note: out.note });
    return out;
  }

  out.decision = signal.decision;
  out.confidence = signal.confidence;

  // ---- RETRIGGER: the volume engine earned a fresh entry + fresh TP/SL ----
  if (mode === "RETRIGGER") {
    const lv = vf.levels;
    signal = {
      ...signal,
      id: `sig_${pair}_${Date.now()}_vt`,
      decision: state.dir,
      // The forecast score IS the confidence of a retrigger — the engine's
      // own read may be a plain WAIT right now, and that is fine.
      confidence: Math.max(
        Math.round(vf.score),
        actionable && signal.decision === state.dir ? signal.confidence : 0,
      ),
      entry: lv.entry, sl: lv.sl, tp1: lv.tp, tp2: null, rr: lv.rr,
      scenario: "VOLUME",
      reasoning: `VOLUME RETRIGGER #${Number(state.retriggers || 0) + 1} — ${vf.reasons.join(" | ")}`,
      retriggerOf: key,
    };
    // Levels are replaced, the open time is not: it is still ONE signal.
    retriggerState(db, key, { entry: lv.entry, sl: lv.sl, tp: lv.tp, rr: lv.rr });
    out.decision = signal.decision;
    out.confidence = signal.confidence;
  }

  // ---- lot size: the USER's own setting decides (item 8 + 9) ----
  // sizeFor returns { lotSize, riskUsd, reason, mode } where mode is
  //   "auto"  — the lot size engine (Kelly) sized it from balance + profile
  //   "fixed" — the user pinned an exact lot in Settings/Watchlist
  // Both are passed straight through; the executor must not re-size either.
  // Legacy global bots have no user account, so they use the platform default.
  const sizing = deps.sizeFor
    ? deps.sizeFor(userId, signal)
    : { lotSize: 0.01, reason: "default 0.01 lot (no account configured)", riskUsd: 0 };
  signal.lotSize = sizing.lotSize || null;
  signal.lotReason = sizing.reason;
  signal.riskUsd = sizing.riskUsd || null;
  signal.lotMode = sizing.mode || null;

  // ---- store the signal (for THIS user) ----
  if (signalEnabled) deps.storeSignal(userId, signal);

  // ---- execute ----
  // A RETRIGGER refreshes a signal whose order may ALREADY be at the broker:
  // a second order would double the exposure instead of managing it. The
  // fresh TP/SL still goes out as a signal; a new order is only sent when the
  // original fire never placed one (signal-only, refused, or blocked).
  const duplicateOrder = mode === "RETRIGGER" && Number(state?.placed || 0) === 1;
  if (tradeEnabled && def.executes && duplicateOrder) {
    out.note = "retrigger — fresh TP/SL signalled; open order keeps its own broker TP/SL (no duplicate order)";
    out.results.push({ bot: botName, decision: signal.decision, confidence: signal.confidence, execution: "held", lotSize: sizing.lotSize, retrigger: true, note: out.note });
  } else if (tradeEnabled && def.executes) {
    const exec = await deps.executor({
      symbol: pair,
      action: signal.decision === "SELL" || signal.decision === "PUT" ? "sell" : "buy",
      lotSize: sizing.lotSize || 0.01,
      entry: signal.entry, sl: signal.sl, tp: signal.tp1,
      reason: `${botName} bot ${pair} ${timeframe} conf=${signal.confidence}% lot=${sizing.lotSize}`,
      source: `bot:${botName}`,
      userId,
      botName,
      // "fixed" tells the executor to use the user's exact lot; "auto" means
      // the sizing engine already computed it. Either way, no re-sizing.
      sizingMode: sizing.mode || null,
      // The pocket bot buys a Rise/Fall contract with this expiry; the
      // scalping bot uses TP/SL on a MULT contract instead.
      expiryMin: signal.expiryMin || 0,
    });
    out.note = exec?.mode || "sent";
    // "placed" means an order was actually SENT. A blocked/refused order
    // (risk gate, no Deriv market) is NOT a placed trade and must not
    // consume the daily cap.
    out.traded = !!(exec && exec.mode !== "blocked" && exec.status !== "blocked" && exec.status !== "failed" && exec.status !== "refused");
    out.results.push({ bot: botName, decision: signal.decision, confidence: signal.confidence, execution: exec?.mode || null, lotSize: sizing.lotSize });
  } else {
    out.note = "signal only (trading off)";
    out.results.push({ bot: botName, decision: signal.decision, confidence: signal.confidence, execution: null, lotSize: sizing.lotSize, signalOnly: true, retrigger: mode === "RETRIGGER" });
  }

  // ---- record the signal's life -------------------------------------------
  if (lifecycle) {
    if (mode === "RETRIGGER") {
      // Only when this retrigger actually placed an order does the signal
      // become "placed" — that is what stops the next retrigger doubling it.
      if (out.traded) markPlaced(db, key);
    } else {
      openState(db, {
        key, userId, bot: botName, pair,
        dir: signal.decision, entry: signal.entry, sl: signal.sl, tp: signal.tp1, rr: signal.rr,
        placed: out.traded ? 1 : 0,
      });
    }
  }
  return out;
}

// ============================================================================
// SCALPING ANALYSIS — TP/SL on small timeframes
// ----------------------------------------------------------------------------
// Like the main MTF engine but on 1m/5m/15m. Uses EMA trend + momentum +
// ATR-based TP/SL. No fixed expiry — exits are price-based (TP/SL).
// ============================================================================

const SCALP_EMA_FAST = 9;
const SCALP_EMA_SLOW = 21;
const SCALP_RSI_PERIOD = 14;
const SCALP_ATR_PERIOD = 14;
const SCALP_MIN_CONF = 55;

function emaLast(values, period) {
  if (values.length < period) return null;
  const k = 2 / (period + 1);
  let e = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < values.length; i++) e = values[i] * k + e * (1 - k);
  return e;
}

function atrLast(candles, period = SCALP_ATR_PERIOD) {
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

function rsiLast(closes, period = SCALP_RSI_PERIOD) {
  if (closes.length < period + 1) return 50;
  let avgGain = 0, avgLoss = 0;
  for (let i = closes.length - period; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    if (d > 0) avgGain += d; else avgLoss -= d;
  }
  avgGain /= period; avgLoss /= period;
  if (avgLoss === 0) return 100;
  return 100 - 100 / (1 + avgGain / avgLoss);
}

/**
 * Scalping analysis: TP/SL based on small timeframes.
 * @param pair symbol
 * @param candles array of {open, high, low, close, volume} oldest->newest
 * @param timeframe "1m" | "5m" | "15m"
 * @returns { decision, confidence, entry, sl, tp1, tp2, rr, reasons }
 */
export function scalpingAnalyze(pair, candles, timeframe = "5m", opts = {}) {
  const sym = String(pair || "").toUpperCase();
  const out = {
    symbol: sym, decision: "WAIT", confidence: 0, entry: null, sl: null,
    tp1: null, tp2: null, rr: null, reasons: [],
  };
  if (!Array.isArray(candles) || candles.length < 30) {
    out.reasons.push(`warming up (${candles?.length || 0} candles, need 30+)`);
    return out;
  }
  const closes = candles.map(c => c.close);
  const e9 = emaLast(closes, SCALP_EMA_FAST);
  const e21 = emaLast(closes, SCALP_EMA_SLOW);
  const atr = atrLast(candles);
  const rsi = rsiLast(closes);
  if (e9 == null || e21 == null || !(atr > 0)) {
    out.reasons.push("indicators not ready");
    return out;
  }
  const bull = e9 > e21, bear = e9 < e21;
  if (!bull && !bear) { out.reasons.push("EMA flat — no trend"); return out; }
  const dir = bull ? "BUY" : "SELL";
  const sgn = bull ? 1 : -1;
  const last = candles[candles.length - 1];
  const body = sgn * (last.close - last.open);
  const range = last.high - last.low;
  if (range < 0.5 * atr) { out.reasons.push("flatlined candle — dead market"); return out; }
  if (body < 0.1 * atr) { out.reasons.push("last body too weak vs ATR"); return out; }
  let conf = 50;
  const reasons = [
    `${timeframe} trend ${bull ? "bullish" : "bearish"} (EMA9 ${bull ? ">" : "<"} EMA21)`,
    `RSI ${rsi.toFixed(0)} ${rsi > 60 ? "overbought" : rsi < 40 ? "oversold" : "neutral"}`,
  ];
  const sep = Math.abs(e9 - e21) / atr;
  if (sep > 0.1) { conf += 10; reasons.push(`EMA separation ${sep.toFixed(2)}x ATR`); }
  if (bull && rsi < 60) { conf += 5; reasons.push("not overbought — room to run"); }
  if (bear && rsi > 40) { conf += 5; reasons.push("not oversold — room to fall"); }
  const entry = last.close;
  const slDist = 1.5 * atr;
  const tpDist = 2.0 * atr;
  const sl = bull ? entry - slDist : entry + slDist;
  const tp1 = bull ? entry + tpDist : entry - tpDist;
  const tp2 = bull ? entry + 3.0 * atr : entry - 3.0 * atr;

  // ---- VOLUME-MATRIX VETO ------------------------------------------------
  // The same 5-timeframe bias the intraday engine already filters on. A
  // scalping entry against strong opposing flow is the trade the tape is
  // paying you to take the OTHER side of; stand down rather than fight it.
  const mb = opts.matrixBias || null;
  if (mb && mb.bias && mb.bias !== "NEUTRAL") {
    const flowSide = mb.bias === "BULL" ? "BUY" : "SELL";
    const strength = Math.abs(Number(mb.score) || 0);
    if (flowSide !== dir && strength >= 2) {
      out.reasons.push(
        ...reasons,
        `volume matrix disagrees (${mb.bias}, score ${mb.score}) — flow opposes ${dir}, scalps don't fight flow`,
      );
      return out;
    }
    if (flowSide === dir) { conf += 5; reasons.push(`volume matrix agrees (${mb.bias}, score ${mb.score}) — flow with the entry`); }
  }

  // ---- COST GATE ---------------------------------------------------------
  // The published RR is meaningless until friction is subtracted. Which
  // friction applies depends on the MARKET: a spread+commission+slippage cross
  // on FX/CFDs, a payout markup on a Deriv synthetic. A setup whose NET RR is
  // under 1.0 loses money even when the direction is right, and is not a
  // trade — it is a donation. Never emit it as a signal.
  const costs = applyCostsFor({ symbol: sym, entry, sl, tp: tp1 });
  // Attached BEFORE any early return: a setup blocked by costs (or docked below
  // the confidence floor by them) must still report WHAT priced it out. The
  // caller, the run log and the app all need that number to explain the WAIT.
  out.costs = costs;
  out.rrGross = Number((tpDist / slDist).toFixed(2));
  if (costs.blocked) {
    out.reasons.push(
      ...reasons,
      `COST GATE: net RR ${costs.netRR} after ${costs.model === "payout_markup" ? `payout markup ${(costs.markupPct * 100).toFixed(1)}%` : `friction ${costs.costPrice.toFixed(5)} (${costs.costPctOfRisk}% of risk)`} — the edge does not survive costs on ${timeframe}, no signal`,
    );
    return out;
  }
  if (costs.weak) {
    conf -= 12;
    reasons.push(`cost penalty: net RR ${costs.netRR} (costs ${costs.costPctOfRisk}% of risk) is thin — confidence docked 12`);
  }
  reasons.push(`costs on ${sym}: ${costs.model === "payout_markup" ? `payout markup ${(costs.markupPct * 100).toFixed(1)}%` : `spread+slippage ${costs.costPrice.toFixed(5)}`}${costs.estimated ? " (estimated — override with COST_OVERRIDES_JSON)" : ""} → net RR ${costs.netRR}`);

  conf = Math.max(20, Math.min(75, Math.round(conf)));
  if (conf < SCALP_MIN_CONF) {
    out.reasons.push(...reasons, `confidence ${conf} below ${SCALP_MIN_CONF} floor`);
    return out;
  }
  out.decision = dir;
  out.confidence = conf;
  out.entry = entry;
  out.sl = sl;
  out.tp1 = tp1;
  out.tp2 = tp2;
  // `rr` is the NET reward:risk — the number a trader actually keeps. The
  // gross geometry stays available as rrGross for the UI.
  out.rr = costs.netRR;
  out.reasons = reasons;
  return out;
}

// ============================================================================
// BOT-PAIR ASSOCIATION — assign bots to specific pairs
// ----------------------------------------------------------------------------
// Each pair can have multiple bots assigned. This replaces the old single
// bot-per-config model with a per-pair bot selection system.
// ============================================================================

export function ensureBotPairTables(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS bot_pairs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT NOT NULL,
      bot_name TEXT NOT NULL,
      pair TEXT NOT NULL,
      timeframe TEXT NOT NULL,
      trade_enabled INTEGER DEFAULT 0,
      signal_enabled INTEGER DEFAULT 1,
      min_confidence INTEGER DEFAULT 60,
      max_trades_day INTEGER DEFAULT 2,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(user_id, bot_name, pair)
    );
    CREATE INDEX IF NOT EXISTS idx_bot_pairs_user ON bot_pairs(user_id);
    CREATE INDEX IF NOT EXISTS idx_bot_pairs_pair ON bot_pairs(pair);
  `);
}

export function getBotPairs(db, userId, pair = null) {
  ensureBotPairTables(db);
  try {
    const rows = pair
      ? db.prepare("SELECT * FROM bot_pairs WHERE user_id = ? AND pair = ?").all(userId, pair.toUpperCase())
      : db.prepare("SELECT * FROM bot_pairs WHERE user_id = ?").all(userId);
    return rows.map(r => ({
      id: r.id,
      userId: r.user_id,
      botName: r.bot_name,
      pair: r.pair,
      timeframe: r.timeframe,
      tradeEnabled: r.trade_enabled === 1,
      signalEnabled: r.signal_enabled === 1,
      minConfidence: r.min_confidence ?? 60,
      maxTradesDay: r.max_trades_day ?? 2,
    }));
  } catch { return []; }
}

export function saveBotPair(db, userId, botName, pair, cfg = {}) {
  ensureBotPairTables(db);
  const def = BOT_DEFS[botName];
  if (!def) throw new Error(`unknown bot: ${botName}`);
  const sym = String(pair || "").toUpperCase().replace(/[^A-Z0-9_.]/g, "").slice(0, 12);
  if (!sym) throw new Error("pair required");
  const timeframe = String(cfg.timeframe || def.defaultTimeframe);
  if (!def.timeframes.includes(timeframe)) {
    throw new Error(`${botName} supports timeframes: ${def.timeframes.join(", ")}`);
  }
  const tradeEnabled = cfg.tradeEnabled ? 1 : 0;
  const signalEnabled = cfg.signalEnabled !== false ? 1 : 0;
  const minConfidence = Math.max(0, Math.min(100, Number(cfg.minConfidence ?? 60)));
  const maxTradesDay = Math.max(1, Math.min(10, Number(cfg.maxTradesDay ?? 2)));
  db.prepare(`
    INSERT INTO bot_pairs (user_id, bot_name, pair, timeframe, trade_enabled, signal_enabled, min_confidence, max_trades_day, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(user_id, bot_name, pair) DO UPDATE SET
      timeframe = ?, trade_enabled = ?, signal_enabled = ?, min_confidence = ?, max_trades_day = ?, updated_at = ?
  `).run(userId, botName, sym, timeframe, tradeEnabled, signalEnabled, minConfidence, maxTradesDay, Date.now(),
    timeframe, tradeEnabled, signalEnabled, minConfidence, maxTradesDay, Date.now());
  return getBotPairs(db, userId, sym).find(b => b.botName === botName);
}

export function removeBotPair(db, userId, botName, pair) {
  ensureBotPairTables(db);
  try {
    db.prepare("DELETE FROM bot_pairs WHERE user_id = ? AND bot_name = ? AND pair = ?")
      .run(userId, botName, String(pair).toUpperCase());
    return true;
  } catch { return false; }
}

export function getPairsForBot(db, userId, botName) {
  ensureBotPairTables(db);
  try {
    return db.prepare("SELECT pair FROM bot_pairs WHERE user_id = ? AND bot_name = ?").all(userId, botName).map(r => r.pair);
  } catch { return []; }
}
