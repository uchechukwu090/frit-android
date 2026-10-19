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
// ============================================================================

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
      decision TEXT, confidence INTEGER, note TEXT
    );
  `);
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
  return db.prepare("SELECT * FROM bots").all().map((b) => ({
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
  }));
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
    return db.prepare(
      // F-43 (Week 5): the pocket bot logs CALL/PUT, not BUY/SELL. Counting
      // only BUY/SELL meant the pocket bot's daily cap never incremented —
      // max_trades_day was silently ignored for that bot (unlimited trades
      // per day). Every decision that PLACES a trade now counts.
      "SELECT COUNT(*) AS c FROM bot_runs WHERE bot = ? AND date(fired_at/1000,'unixepoch') = date('now') AND decision IN ('BUY','SELL','CALL','PUT')"
    ).get(name)?.c || 0;
  } catch { return 0; }
}

function logRun(db, bot, pair, timeframe, decision, confidence, note) {
  try {
    db.prepare(
      "INSERT INTO bot_runs (bot, pair, timeframe, fired_at, decision, confidence, note) VALUES (?,?,?,?,?,?,?)"
    ).run(bot, pair, timeframe, Date.now(), decision, confidence ?? 0, String(note || "").slice(0, 500));
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
 * One runner tick. Deps injected from index.js:
 * { db, mtfAnalyze(symbol)->result, analyzeSynth(symbol, closes)->result,
 *   longtermAnalyze(symbol, closes)->result, fetchDaily(symbol)->closes[],
 *   fetchCloses(symbol, interval)->closes[], getChain, scanSymbol, scanFx,
 *   storeSignal(userId, signal), formatSignal(symbol, result),
 *   storeOptionsSignal(userId, signal), openPaper(userId, signal),
 *   executor({symbol, action, lotSize, entry, sl, tp, reason})->result,
 *   notifyOptions(userId, signal)->void, log }
 * userId "bot" owns bot-generated history (per-user fan-out stays in LiveFeed).
 */
export async function tickBots(deps) {
  const { db } = deps;
  ensureBotsTables(db);
  const now = Date.now();
  const results = [];
  for (const bot of getBots(db)) {
    const def = BOT_DEFS[bot.name];
    if (!bot.auto_enabled || !bot.pair) continue;
    if (bot.last_run && now - bot.last_run < def.intervalMin * 60_000) continue;
    if (runsToday(db, bot.name) >= bot.max_trades_day) continue;
    db.prepare("UPDATE bots SET last_run = ? WHERE name = ?").run(now, bot.name);
    try {
      if (bot.name === "intraday") {
        // Deriv synthetics run the generator-aware engine, not the MTF trend
        // engine — MA crossovers on RNG are coin flips minus costs.
        const isSynth = /^R_|^1HZ|^BOOM|^CRASH|^STPRNG|^RDBULL|^RDBEAR|^JD|^GDA/.test(bot.pair);
        const result = isSynth
          ? synthToSignal(bot.pair, await deps.analyzeSynth(bot.pair, await deps.fetchCloses(bot.pair, "1m")))
          : await deps.mtfAnalyze(bot.pair, {});
        const signal = isSynth ? result : deps.formatSignal(bot.pair, result);
        const actionable = ["BUY", "SELL"].includes(signal.decision) && signal.confidence >= bot.min_confidence;
        if (actionable) {
          const exec = await deps.executor({
            symbol: bot.pair,
            action: signal.decision === "BUY" ? "buy" : "sell",
            lotSize: 0.01, entry: signal.entry, sl: signal.sl, tp: signal.tp1,
            reason: `intraday bot ${bot.pair} ${bot.timeframe} conf=${signal.confidence}%`,
            source: "bot:intraday",
          });
          deps.storeSignal("bot", signal);
          logRun(db, bot.name, bot.pair, bot.timeframe, signal.decision, signal.confidence, exec?.mode || "sent");
          results.push({ bot: bot.name, decision: signal.decision, confidence: signal.confidence, execution: exec?.mode || null });
        } else {
          logRun(db, bot.name, bot.pair, bot.timeframe, signal.decision, signal.confidence, "below confidence floor or no setup");
          results.push({ bot: bot.name, decision: signal.decision, note: "no trade" });
        }
      } else if (bot.name === "scalping") {
        // Scalping: TP/SL based on small timeframes. Uses the candle store
        // for fast access to recent candles, computes EMA trend + momentum,
        // and fires BUY/SL with TP/SL levels.
        const candles = await deps.fetchCloses(bot.pair, bot.timeframe);
        if (!candles || candles.length < 30) {
          logRun(db, bot.name, bot.pair, bot.timeframe, "WAIT", 0, "warming up — need 30+ candles");
          results.push({ bot: bot.name, decision: "WAIT", note: "warming up" });
        } else {
          const result = deps.scalpingAnalyze(bot.pair, candles, bot.timeframe);
          const actionable = ["BUY", "SELL"].includes(result.decision) && result.confidence >= bot.min_confidence;
          if (actionable) {
            const exec = await deps.executor({
              symbol: bot.pair,
              action: result.decision === "BUY" ? "buy" : "sell",
              lotSize: 0.01, entry: result.entry, sl: result.sl, tp: result.tp1,
              reason: `scalping bot ${bot.pair} ${bot.timeframe} conf=${result.confidence}%`,
              source: "bot:scalping",
            });
            deps.storeSignal("bot", {
              id: `sig_${bot.pair}_${Date.now()}_sc`, symbol: bot.pair, decision: result.decision,
              confidence: result.confidence, entry: result.entry, sl: result.sl,
              tp1: result.tp1, tp2: result.tp2 || null, rr: result.rr, scenario: "SCALPING",
              reasoning: (result.reasons || []).join(" | "), timestamp: Date.now(),
            });
            logRun(db, bot.name, bot.pair, bot.timeframe, result.decision, result.confidence, exec?.mode || "sent");
            results.push({ bot: bot.name, decision: result.decision, confidence: result.confidence, execution: exec?.mode || null });
          } else {
            logRun(db, bot.name, bot.pair, bot.timeframe, result.decision, result.confidence || 0, "below confidence floor or no setup");
            results.push({ bot: bot.name, decision: result.decision, note: "no trade" });
          }
        }
      } else if (bot.name === "pocket") {
        // Pocket option: predicts if a candle will end bullish or bearish
        // within a given time. Uses the pocket engine's signalFromCandles.
        const expiryMin = bot.timeframe === "1m" ? 1 : bot.timeframe === "5m" ? 5 : 15;
        const signal = deps.pocketScan(bot.pair, expiryMin);
        const actionable = ["CALL", "PUT"].includes(signal.decision) && signal.confidence >= bot.min_confidence;
        if (actionable) {
          const exec = await deps.executor({
            symbol: bot.pair,
            action: signal.decision === "CALL" ? "buy" : "sell",
            lotSize: 0.01, entry: signal.entryRef, sl: null, tp: null,
            reason: `pocket bot ${bot.pair} ${expiryMin}m conf=${signal.confidence}%`,
            source: "bot:pocket",
          });
          deps.storeSignal("bot", {
            id: `sig_${bot.pair}_${Date.now()}_pk`, symbol: bot.pair, decision: signal.decision,
            confidence: signal.confidence, entry: signal.entryRef, sl: null,
            tp1: null, tp2: null, rr: null, scenario: "POCKET",
            reasoning: (signal.reasons || []).join(" | "), timestamp: Date.now(),
          });
          logRun(db, bot.name, bot.pair, bot.timeframe, signal.decision, signal.confidence, exec?.mode || "sent");
          results.push({ bot: bot.name, decision: signal.decision, confidence: signal.confidence, execution: exec?.mode || null });
        } else {
          logRun(db, bot.name, bot.pair, bot.timeframe, signal.decision, signal.confidence || 0, "below confidence floor or no setup");
          results.push({ bot: bot.name, decision: signal.decision, note: "no trade" });
        }
      } else if (bot.name === "longterm") {
        const daily = await deps.fetchDaily(bot.pair);
        const closes = bot.timeframe === "1W" ? toWeekly(daily) : daily;
        const result = deps.longtermAnalyze(bot.pair, closes);
        const actionable = ["BUY", "SELL"].includes(result.decision) && result.confidence >= bot.min_confidence;
        if (actionable) {
          const exec = await deps.executor({
            symbol: bot.pair,
            action: result.decision === "BUY" ? "buy" : "sell",
            lotSize: 0.01, entry: result.entry, sl: result.sl, tp: result.tp,
            reason: `longterm bot ${bot.pair} ${bot.timeframe} conf=${result.confidence}%`,
            source: "bot:longterm",
          });
          deps.storeSignal("bot", {
            id: `sig_${bot.pair}_${Date.now()}_lt`, symbol: bot.pair, decision: result.decision,
            confidence: result.confidence, entry: result.entry, sl: result.sl,
            tp1: result.tp, tp2: null, rr: result.rr, scenario: "LONGTERM",
            reasoning: result.reasons.join(" | "), timestamp: Date.now(),
          });
          logRun(db, bot.name, bot.pair, bot.timeframe, result.decision, result.confidence, exec?.mode || "sent");
          results.push({ bot: bot.name, decision: result.decision, confidence: result.confidence, execution: exec?.mode || null });
        } else {
          logRun(db, bot.name, bot.pair, bot.timeframe, result.decision, result.confidence || 0, "below confidence floor or no setup");
          results.push({ bot: bot.name, decision: result.decision, note: "no trade" });
        }
      } else if (bot.name === "options") {
        // Scan + notify + paper only. NEVER the executor — no options rail.
        const isFxStyle = /USD$|XAU|XAG|JPY|GBP|EUR|CHF|CAD|AUD|NZD/.test(bot.pair) && !/^(SPY|QQQ|IWM|XSP)$/.test(bot.pair);
        if (isFxStyle) {
          const closes = await deps.fetchDaily(bot.pair);
          const out = deps.scanFx({ symbol: bot.pair, closes });
          if (out.signal) {
            deps.storeOptionsSignal("bot", out.signal);
            deps.notifyOptions?.("bot", out.signal);
          }
          logRun(db, bot.name, bot.pair, bot.timeframe, out.signal ? "FX_LEVELS" : "SKIP", 0, out.signal ? out.signal.id : out.why);
          results.push({ bot: bot.name, decision: out.signal ? "FX_LEVELS" : "SKIP" });
        } else {
          const chain = await deps.getChain(bot.pair);
          const closes = await deps.fetchDaily(bot.pair);
          const out = deps.scanSymbol({ symbol: bot.pair, chain, closes, db });
          if (out.signal) {
            deps.storeOptionsSignal("bot", out.signal);
            deps.openPaper?.("bot", { ...out.signal, entrySpot: chain.underlying });
            deps.notifyOptions?.("bot", out.signal);
          }
          logRun(db, bot.name, bot.pair, bot.timeframe, out.signal ? out.signal.kind : "SKIP", 0, out.signal ? out.signal.id : out.why);
          results.push({ bot: bot.name, decision: out.signal ? out.signal.kind : "SKIP" });
        }
      }
    } catch (e) {
      logRun(db, bot.name, bot.pair, bot.timeframe, "ERROR", 0, e.message);
      results.push({ bot: bot.name, decision: "ERROR", note: e.message });
    }
  }
  return results;
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
export function scalpingAnalyze(pair, candles, timeframe = "5m") {
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
  out.rr = tpDist / slDist;
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
