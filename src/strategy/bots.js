// ============================================================================
// FRIT BOTS REGISTRY — options / intraday / longterm, one config surface
// ----------------------------------------------------------------------------
// Each bot owns: pair, timeframe, auto toggle, confidence floor, daily cap.
// The runner ticks every 60s but each bot fires on its own cadence (options
// scans hit Yahoo — at most every 4h or the IP gets banned; longterm is
// daily; intraday every 30m). Intraday/longterm BUY/SELL flow through the
// SAME risk gate + bridge + position monitor as /trade and the scheduler.
// The options bot NEVER executes: auto = scan + notify (+ paper ledger).
// Timeframes are validated per bot. Note the honest mapping: the intraday
// engine analyzes 30M/4H structure regardless of the selected display TF —
// the TF picks monitoring granularity, not a retuned engine.
// ============================================================================

export const BOT_DEFS = {
  options: {
    label: "Options bot",
    timeframes: ["1D"],
    defaultTimeframe: "1D",
    intervalMin: 240,
    executes: false,
    blurb: "Credit spreads on SPY/QQQ + FX expected-move levels. Signal-only: auto = scan + notify + paper ledger. Never places orders.",
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
  const pair = String(cfg.pair || "").toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 12);
  if (!pair) throw new Error("pair required (e.g. XAUUSD, EURUSD, BTCUSD, SPY)");
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
      "SELECT COUNT(*) AS c FROM bot_runs WHERE bot = ? AND date(fired_at/1000,'unixepoch') = date('now') AND decision IN ('BUY','SELL')"
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

/**
 * One runner tick. Deps injected from index.js:
 * { db, mtfAnalyze(symbol)->result, longtermAnalyze(symbol, closes)->result,
 *   fetchDaily(symbol)->closes[], getChain, scanSymbol, scanFx,
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
        const result = await deps.mtfAnalyze(bot.pair, {});
        const signal = deps.formatSignal(bot.pair, result);
        const actionable = ["BUY", "SELL"].includes(signal.decision) && signal.confidence >= bot.min_confidence;
        if (actionable) {
          const exec = await deps.executor({
            symbol: bot.pair,
            action: signal.decision === "BUY" ? "buy" : "sell",
            lotSize: 0.01, entry: signal.entry, sl: signal.sl, tp: signal.tp1,
            reason: `intraday bot ${bot.pair} ${bot.timeframe} conf=${signal.confidence}%`,
          });
          deps.storeSignal("bot", signal);
          logRun(db, bot.name, bot.pair, bot.timeframe, signal.decision, signal.confidence, exec?.mode || "sent");
          results.push({ bot: bot.name, decision: signal.decision, confidence: signal.confidence, execution: exec?.mode || null });
        } else {
          logRun(db, bot.name, bot.pair, bot.timeframe, signal.decision, signal.confidence, "below confidence floor or no setup");
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
