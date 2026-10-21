import "dotenv/config";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import dotenv from "dotenv";
import fetch from "node-fetch";
import FormData from "form-data";
import Database from "better-sqlite3";
import crypto from "crypto";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { MTFStrategyEngine } from "./strategy/engine.js";
import { LiveFeed } from "./strategy/feed/live_feed.js";
import { signalFromCandles, recordPaper, settleOpen, pocketStats, ensurePocketTables, nextBoundaryMs, profileFor, calibFor, upsertArm, getArm, clearArm, trigMinutes, PB_MAX_WAIT } from "./strategy/pocket/engine.js";
import { DerivClient, DerivExecutor, SYNTHETIC_ONLY } from "./strategy/deriv.js";
import { DerivRestClient } from "./strategy/deriv-rest.js";
import { AlpacaClient, AlpacaExecutor, qtyForRisk } from "./strategy/alpaca.js";
import { TradeTaskScheduler } from "./strategy/scheduler.js";
import { PositionMonitor } from "./strategy/position_monitor.js";
import { PullbackJournal } from "./strategy/journal.js";
import { PortfolioRisk } from "./strategy/risk.js";
import { getSymbolCost, costToUsd } from "./strategy/costs.js";
import { getChain, fetchDailyCloses } from "./strategy/options/chains.js";
import { scanSymbol, ensureOptionsTables } from "./strategy/options/scanner.js";
import { analyzeLongterm } from "./strategy/longterm.js";
import { analyzeSynth } from "./strategy/synthetics.js";
import { analyzeVolumeMatrix } from "./strategy/volumematrix.js";
import { createBilling } from "./billing.js";
import { readProviderStream, sseDone, sseError, sseDelta } from "./streamProvider.js";
import { classifyStreamRoute, stripConversationPrefix, ROUTE_TASK, ROUTE_CONVERSATION } from "./streaming.js";
import { getBots, saveBot, tickBots, BOT_DEFS, scalpingAnalyze, ensureBotPairTables, getBotPairs, saveBotPair, removeBotPair, botCooldown } from "./strategy/bots.js";
import { recommendLotSize, kellyFraction, normalizeRiskProfile, riskProfileInfo, RISK_PROFILES, DEFAULT_RISK_PROFILE } from "./strategy/sizing.js";
const RISK_PROFILE_INFO = RISK_PROFILES;
import { marketOf, canonicalSymbol, isDerivSymbol } from "./strategy/rails/router.js";
import { openPaperPosition, resolvePaperBook, ledgerSummary } from "./strategy/options/paper.js";
import { maturitySummary } from "./strategy/ledger.js";
import { runBacktest, walkForward } from "./strategy/options/backtest.js";
import { scanFx } from "./strategy/options/fx.js";
import { deepSearch } from "./deep_search.js";
import * as imageProcessor from "./creative/imageProcessor.js";
import * as aiFeatures from "./creative/aiFeatures.js";
import * as promptEngine from "./creative/promptEngine.js";
import * as templateLibrary from "./creative/templateLibrary.js";
import * as cinema from "./creative/cinema.js";
import * as designStudio from "./creative/design.js";
import * as brandKit from "./creative/brand.js";
import * as designAgent from "./creative/designAgent.js";
import * as imageProviders from "./creative/imageProviders.js";
import * as typeLayer from "./creative/typeLayer.js";
import * as refs from "./creative/refs.js";
import * as directorAgent from "./creative/directorAgent.js";
import { runBounded } from "./concurrency.js";
import { setModelDb, MODEL_DEFAULTS, MODEL_KEYS, MODEL_META, overrideOf, M } from "./model_config.js";

dotenv.config();

process.on("unhandledRejection", (r) => console.error("[unhandledRejection]", r));
// FIX (F-02): uncaughtException was logged and swallowed — the process then
// kept serving in unknown state (a fatal after a half-written SQLite
// transaction leaves corrupt rows nobody sees). Log loudly, then exit(1) after
// a short grace period so pending stdio flushes; the platform restarts clean
// and `gracefulShutdown()` below closes the DB on non-fatal restarts too.
process.on("uncaughtException", (e) => {
  console.error("[uncaughtException] FATAL — shutting down to avoid unknown state:", e);
  setTimeout(() => process.exit(1), 1000).unref?.();
});

const app = express();
const PORT = Number(process.env.PORT || 8787);

// Bounded outbound fetch for MARKET DATA + WEATHER (Week 6 fix, F-45).
// These calls used to have no AbortSignal at all: a stalled (not reset) socket
// hung forever, and /api/trading/scan awaits symbols one at a time, so one
// black-holed upstream call blocked the whole scan past the app's timeout.
// 15s is generous for a single candle/quote page; a normal call is <1s.
const MARKET_FETCH_MS = 15_000;
const boundedFetch = (url, init = {}, ms = MARKET_FETCH_MS) =>
  fetch(url, { ...init, signal: init.signal || AbortSignal.timeout(ms) });
const __dirname = dirname(fileURLToPath(import.meta.url));

// ========================= PERSISTENCE (SQLite) =====================
// Render's local disk is EPHEMERAL (wipes on restart) unless a persistent disk
// is mounted at /data. DB_PATH selects the safe location automatically:
//   DB_PATH env > /data/frit.db (if /data exists) > ./frit.db (dev only).
// Off-site safety: hourly + on-approve backups to Supabase Storage (free 1GB),
// restored automatically on boot when the local file is missing. Worst case:
// <1 hour of data at risk. Boot logs a loud warning when fully ephemeral.
import { existsSync as _existsSync, mkdirSync as _mkdirSync } from "fs";
import { writeFileSync as _writeFileSync, readFileSync as _readFileSync, statSync as _statSync } from "fs";
const _dataDir = "/data";
const DB_PATH = process.env.DB_PATH || (_existsSync(_dataDir) ? join(_dataDir, "frit.db") : join(__dirname, "../frit.db"));
const EPHEMERAL_DB = DB_PATH === join(__dirname, "../frit.db") && process.env.NODE_ENV === "production";
const SUPABASE_URL = (process.env.SUPABASE_URL || "").replace(/\/$/, "");
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_KEY || "";
const SUPABASE_BUCKET = process.env.SUPABASE_BUCKET || "frit-backups";
const HAS_REMOTE_BACKUP = !!(SUPABASE_URL && SUPABASE_KEY);
async function supaBucket() {
  // Create bucket once (idempotent — 409 means it exists).
  await fetch(`${SUPABASE_URL}/storage/v1/bucket`, {
    method: "POST",
    headers: { Authorization: `Bearer ${SUPABASE_KEY}`, apikey: SUPABASE_KEY, "Content-Type": "application/json" },      body: JSON.stringify({ name: SUPABASE_BUCKET, public: false }),
      // Week 6: best-effort bucket creation — it must never delay a cold boot
      // (this runs before server.listen). 5s is plenty; failure is ignored.
      signal: AbortSignal.timeout(5_000),
  }).catch(() => {});
}
async function restoreFromBackup() {
  if (!HAS_REMOTE_BACKUP || _existsSync(DB_PATH)) return false;
  try {
    await supaBucket();
    const r = await fetch(`${SUPABASE_URL}/storage/v1/object/${SUPABASE_BUCKET}/frit-latest.db`, {
      headers: { Authorization: `Bearer ${SUPABASE_KEY}`, apikey: SUPABASE_KEY },
      signal: AbortSignal.timeout(60_000),
    });
    if (!r.ok) { console.warn(`[backup] no remote backup to restore (HTTP ${r.status})`); return false; }
    const buf = Buffer.from(await r.arrayBuffer());
    if (buf.length < 100) return false;
    _writeFileSync(DB_PATH, buf);
    console.log(`[backup] restored frit.db from Supabase (${buf.length} bytes)`);
    return true;
  } catch (e) {
    console.warn("[backup] restore failed:", e.message);
    return false;
  }
}
await restoreFromBackup();
let db;
try {
  db = new Database(DB_PATH);
} catch (e) {
  console.error(`[db] failed to open database at ${DB_PATH}:`, e?.message || e);
  process.exit(1);
}
db.pragma("journal_mode = WAL");
setModelDb(db); // model_overrides table + console-overridable slugs (see model_config.js)

// Initialize Tables
db.exec(`
  CREATE TABLE IF NOT EXISTS agent_sessions (
    id TEXT PRIMARY KEY,
    goal TEXT,
    status TEXT DEFAULT 'pending',
    task_ledger TEXT,
    last_device_state TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS agent_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    step_n INTEGER,
    role TEXT,
    content TEXT,
    tool_calls TEXT,
    tool_results TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES agent_sessions(id)
  );
  CREATE TABLE IF NOT EXISTS trade_memory (
    symbol TEXT,
    direction TEXT,
    pattern TEXT,
    outcome TEXT,
    note TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS wallet_accounts (
    user_id TEXT PRIMARY KEY,
    balance_kobo INTEGER DEFAULT 0,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS wallet_txns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    amount_kobo INTEGER,
    kind TEXT,
    note TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS spent_txids (
    txid TEXT PRIMARY KEY,
    payment_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT,
    payload TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS api_keys (
    key_hash TEXT PRIMARY KEY,
    user_id TEXT,
    label TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS withdrawals (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    amount_kobo INTEGER,
    bank_name TEXT,
    account_number TEXT,
    account_name TEXT,
    status TEXT DEFAULT 'pending',
    admin_note TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS pending_payments (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    tier TEXT,
    amount_kobo INTEGER,
    reference TEXT,
    status TEXT DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS subscriptions (
    user_id TEXT PRIMARY KEY,
    tier TEXT DEFAULT 'free',
    status TEXT DEFAULT 'active',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS activation_codes (
    code TEXT PRIMARY KEY,
    tier TEXT,
    status TEXT DEFAULT 'unused',
    user_id TEXT,
    expires_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS usage_daily (
    user_id TEXT,
    day TEXT,
    turns INTEGER DEFAULT 0,
    clips INTEGER DEFAULT 0,
    images INTEGER DEFAULT 0,
    stt_mins INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, day)
  );
  CREATE TABLE IF NOT EXISTS creative_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    kind TEXT,
    status TEXT DEFAULT 'pending',
    prompt TEXT,
    result_url TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    tier TEXT,
    rating INTEGER,
    message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS user_credits (
    user_id TEXT PRIMARY KEY,
    balance INTEGER DEFAULT 0,
    total_earned INTEGER DEFAULT 0,
    total_spent INTEGER DEFAULT 0,
    plan_grant_month TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS credit_txns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    amount INTEGER,
    kind TEXT,
    note TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS provider_jobs (
    provider_id TEXT PRIMARY KEY,
    user_id TEXT,
    kind TEXT,
    charged INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS creative_projects (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    name TEXT,
    type TEXT DEFAULT 'image',
    status TEXT DEFAULT 'draft',
    thumbnail_url TEXT,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS creative_assets (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    user_id TEXT,
    kind TEXT,
    source TEXT,
    original_url TEXT,
    current_url TEXT,
    thumbnail_url TEXT,
    metadata TEXT,
    version INTEGER DEFAULT 1,
    parent_asset_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES creative_projects(id)
  );
  CREATE TABLE IF NOT EXISTS creative_edits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT,
    user_id TEXT,
    operation TEXT,
    params TEXT,
    result_url TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(asset_id) REFERENCES creative_assets(id)
  );
  CREATE TABLE IF NOT EXISTS creative_styles (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    name TEXT,
    category TEXT,
    parameters TEXT,
    is_default BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS creative_templates (
    id TEXT PRIMARY KEY,
    name TEXT,
    category TEXT,
    platform TEXT,
    aspect_ratio TEXT,
    layout_data TEXT,
    thumbnail_url TEXT,
    is_system BOOLEAN DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS creative_batch_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    operation TEXT,
    asset_ids TEXT,
    params TEXT,
    status TEXT DEFAULT 'pending',
    completed_count INTEGER DEFAULT 0,
    total_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS creative_collaborators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT,
    owner_user_id TEXT,
    collaborator_user_id TEXT,
    role TEXT DEFAULT 'editor',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES creative_projects(id)
  );
  CREATE TABLE IF NOT EXISTS creative_sync_state (
    project_id TEXT PRIMARY KEY,
    user_id TEXT,
    device_id TEXT,
    last_sync_at DATETIME,
    sync_data TEXT,
    FOREIGN KEY(project_id) REFERENCES creative_projects(id)
  );
  CREATE TABLE IF NOT EXISTS creative_elements (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    type TEXT DEFAULT 'character',
    name TEXT,
    reference_urls TEXT,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE TABLE IF NOT EXISTS creative_brandkits (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    name TEXT,
    colors TEXT,
    font TEXT,
    logo_url TEXT,
    palette TEXT,
    tone TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  -- Trading Suite Tables
  CREATE TABLE IF NOT EXISTS trading_watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    asset_class TEXT DEFAULT 'FOREX',
    is_enabled INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, symbol)
  );
  CREATE INDEX IF NOT EXISTS idx_watchlist_user ON trading_watchlist(user_id);

  CREATE TABLE IF NOT EXISTS trading_signals (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    decision TEXT NOT NULL,
    confidence INTEGER,
    entry REAL,
    sl REAL,
    tp1 REAL,
    tp2 REAL,
    rr REAL,
    scenario TEXT,
    reasoning TEXT,
    pullback_health TEXT,
    range_info TEXT,
    costs TEXT,
    timestamp INTEGER NOT NULL,
    is_favorite INTEGER DEFAULT 0,
    is_read INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE INDEX IF NOT EXISTS idx_signals_user_time ON trading_signals(user_id, timestamp DESC);
  CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trading_signals(symbol);

  CREATE TABLE IF NOT EXISTS push_tokens (
    user_id TEXT PRIMARY KEY,
    token TEXT NOT NULL,
    platform TEXT DEFAULT 'android',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS notification_prefs (
    user_id TEXT PRIMARY KEY,
    signal_alerts INTEGER DEFAULT 1,
    tp_sl_alerts INTEGER DEFAULT 1,
    news_alerts INTEGER DEFAULT 1,
    daily_summary INTEGER DEFAULT 1
  );

  CREATE TABLE IF NOT EXISTS sentiment_cache (
    symbol TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    cached_at INTEGER NOT NULL
  );

  -- Pocket watchlist is SEPARATE from the main trading watchlist: pocket pairs
  -- are fixed-expiry only, main pairs are engine-scanned. Never correlated.
  CREATE TABLE IF NOT EXISTS pocket_watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    expiry_min INTEGER DEFAULT 5,
    is_enabled INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, symbol)
  );
  CREATE INDEX IF NOT EXISTS idx_pocket_wl_user ON pocket_watchlist(user_id);

  -- Tombstones: a removed symbol can never reappear, even if an old cached
  -- list or a queued offline op would resurrect it.
  CREATE TABLE IF NOT EXISTS watchlist_tombstones (
    user_id TEXT NOT NULL,
    list TEXT NOT NULL,
    symbol TEXT NOT NULL,
    removed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, list, symbol)
  );

  -- Per-user trading account: the balance the user actually trades on, their
  -- declared risk profile (drives the Kelly fraction), and their own broker
  -- credentials (Deriv token / MT5 account) for users who want the bot to
  -- trade THEIR account instead of the platform's.
  CREATE TABLE IF NOT EXISTS trading_account (
    user_id TEXT PRIMARY KEY,
    balance REAL DEFAULT 1000,
    risk_profile TEXT DEFAULT 'MODERATE',
    deriv_token TEXT,
    deriv_token_mode TEXT DEFAULT 'platform',
    mt5_server TEXT,
    mt5_login TEXT,
    mt5_password TEXT,
    mt5_broker TEXT,
    updated_at INTEGER
  );
`);

// Migrations — older frit.db files predate these columns and CREATE TABLE IF
// NOT EXISTS won't add them to an existing table.
const sessionCols = db.prepare("PRAGMA table_info(agent_sessions)").all().map(c => c.name);
if (!sessionCols.includes("last_device_state")) {
  db.exec("ALTER TABLE agent_sessions ADD COLUMN last_device_state TEXT");
}
if (!sessionCols.includes("memory")) {
  db.exec("ALTER TABLE agent_sessions ADD COLUMN memory TEXT");
}
const stepCols = db.prepare("PRAGMA table_info(agent_steps)").all().map(c => c.name);
if (!stepCols.includes("tool_call_id")) {
  db.exec("ALTER TABLE agent_steps ADD COLUMN tool_call_id TEXT");
}
// Subscriptions need expiry (older frit.db files lack it).
const subCols = db.prepare("PRAGMA table_info(subscriptions)").all().map(c => c.name);
if (!subCols.includes("expires_at")) {
  db.exec("ALTER TABLE subscriptions ADD COLUMN expires_at DATETIME");
}
const sessCols = db.prepare("PRAGMA table_info(agent_sessions)").all().map(c => c.name);
if (!sessCols.includes("user_id")) {
  db.exec("ALTER TABLE agent_sessions ADD COLUMN user_id TEXT DEFAULT 'default'");
}
// Pending payments: purpose distinguishes sub vs wallet-topup orders.
const ppCols = db.prepare("PRAGMA table_info(pending_payments)").all().map(c => c.name);
if (!ppCols.includes("purpose")) {
  db.exec("ALTER TABLE pending_payments ADD COLUMN purpose TEXT DEFAULT 'sub'");
}
// Money trail: hash-chained ledger rows (older DBs lack the columns).
const txnCols = db.prepare("PRAGMA table_info(wallet_txns)").all().map(c => c.name);
if (!txnCols.includes("prev_hash")) {
  db.exec("ALTER TABLE wallet_txns ADD COLUMN prev_hash TEXT DEFAULT 'GENESIS'");
}
if (!txnCols.includes("tx_hash")) {
  db.exec("ALTER TABLE wallet_txns ADD COLUMN tx_hash TEXT DEFAULT ''");
}

// ========================= ENVIRONMENT ==============================
// OpenRouter is now the single LLM/STT/image gateway (Groq + DeepSeek removed).
// Paid lowest-cost: primary=z-ai/glm-5.3-flash, fast=openai/gpt-oss-20b.
// Free tier (:free, 1000/day after $10 credits) sits BEHIND paid in every chain.
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || "";
const HAS_OR = !!OPENROUTER_API_KEY;
// ---- Runtime model config (Creator console changes these LIVE) ----------
// Every model slug below resolves as: admin override (DB) > env var > baked
// default (see model_config.js — submodules resolve via M() at request time).
// The creator swaps models from the app's Creator console
// (GET+POST /admin/models) — new OpenRouter slugs go live with zero code
// change, zero rebuild, zero redeploy. Verify slugs at openrouter.ai/models.
let OR_PRIMARY = M("OR_PRIMARY_MODEL");
let OR_FAST = M("OR_FAST_MODEL");
let OR_DRAFT = M("OR_DRAFT_MODEL");
let OR_IMAGE_MODEL = M("OR_IMAGE_MODEL");
let OR_STT_MODEL = M("OR_STT_MODEL");
// Graphic-design specialists (on demand, both currently FREE via NovitaAI):
// design = text-to-image with legible text (prompt only — NO refs, NO aspect ratio, rejected otherwise);
// layer = decomposes ONE flat design image into RGBA layers (needs exactly 1 input_reference + layer plan).
let OR_DESIGN_MODEL = M("OR_DESIGN_MODEL");
let OR_LAYER_MODEL = M("OR_LAYER_MODEL");
let OR_VIDEO_MODEL = M("OR_VIDEO_MODEL"); // HeyGen Video default (promo $0.01/s with audio, 720p/1080p 4-8s)
// Genjutsu default: Seedance-class reference model — 4-30s, first/last frames,
// up to 50 image/video/audio references (same family Higgsfield Cinema 4.0 uses).
// Override per-call with {model} or globally (console/env OR_GENJUTSU_MODEL).
let OR_GENJUTSU_MODEL = M("OR_GENJUTSU_MODEL");
// Graphics/design default: legible-text design specialist (FREE). Override with
// OR_GRAPHICS_MODEL for Recraft/Ideogram-class models when served on OpenRouter.
let OR_GRAPHICS_MODEL = M("OR_GRAPHICS_MODEL") || OR_DESIGN_MODEL;
// Craft routing (the pro-designer pattern): typography-led work -> type
// specialist (Ideogram-class), logo/vector/brand-system work -> vector
// specialist (Recraft-class), photographic plates -> photo model. Each
// falls back down the chain so an unserved slug never kills the job.
// Verify slugs at openrouter.ai/models before setting these.
let OR_TYPE_MODEL = M("OR_TYPE_MODEL");
let OR_VECTOR_MODEL = M("OR_VECTOR_MODEL");
// Talking-avatar primary (HeyGen Avatar IV via the same OpenRouter /videos
// pipeline — photo + voice, up to 1080p). Hedra stays as fallback engine.
let OR_AVATAR_MODEL = M("OR_AVATAR_MODEL");
// Neural lip-sync provider (optional): Sync Labs sync-2.0. Needs SYNC_LABS_API_KEY
// + PUBLIC_BASE_URL (webhook receiver). Absent either -> transcript-timed dub.
const SYNC_LABS_KEY = process.env.SYNC_LABS_API_KEY || "";
const PUBLIC_BASE_URL = (process.env.PUBLIC_BASE_URL || process.env.RENDER_EXTERNAL_URL || "").replace(/\/$/, "");
// Hedra Character 3: photo + audio -> talking avatar (no source video needed).
// Needs HEDRA_API_KEY + Supabase staging (signed URLs). Absent -> local fallback.
const HEDRA_API_KEY = process.env.HEDRA_API_KEY || "";
async function hedraFetch(path, { method = "GET", body = null } = {}) {
  const attempt = async (auth) => fetch(`https://api.hedra.com${path}`, {
    method,
    headers: { Authorization: `${auth} ${HEDRA_API_KEY}`, "Content-Type": "application/json" },
    ...(body ? { body: JSON.stringify(body) } : {}),
    signal: AbortSignal.timeout(60_000),
  });
  let r = await attempt("Bearer");
  if (r.status === 401) r = await attempt("Key"); // docs vary; retry alt scheme once
  return r;
}
function hedraVideoUrl(job) {
  const outs = job?.outputs || job?.data?.outputs || [];
  for (const o of outs) {
    const u = o?.url || o?.video_url || o?.download_url || o?.src;
    if (typeof u === "string" && /^https?:\/\//.test(u)) return u;
  }
  return null;
}
// Free-tier models (zero token cost, count against 1000/day quota).
// Verified Oct 2026: qwen3.8-27b (best free all-rounder, tools+vision),
// nemotron-3-super (NVIDIA-backed, sticky). gpt-oss-20b:free 429s constantly
// and qwen3-30b-a3b-2507:free isn't a valid slug — both replaced.
let OR_FREE_FAST = process.env.OR_FREE_FAST || "qwen/qwen3.8-27b:free";
let OR_FREE_DRAFT = process.env.OR_FREE_DRAFT || "nvidia/nemotron-3-super-120b-a12b:free";
// The free roster rotates, so the server re-checks OpenRouter's public model
// list at boot + every 6h and picks the top-rated live :free text models.
const PREFERRED_FREE = [
  "qwen/qwen3.8-27b:free",
  "nvidia/nemotron-3-super-120b-a12b:free",
  "google/gemma-4-26b-a4b-it:free",
  "thinkingmachines/inkling-small:free",
  "poolside/laguna-s-2.1:free",
  "cohere/north-mini-code:free",
  "openrouter/free",
];
function pickFreeModels(all) {
  const ids = new Set((Array.isArray(all) ? all : []).map(m => m.id).filter(id =>
    typeof id === "string" && id.endsWith(":free") &&
    !/embed|whisper|tts|moderation|safety|guard/i.test(id)));
  const ranked = PREFERRED_FREE.filter(id => ids.has(id));
  for (const id of ids) if (!ranked.includes(id)) ranked.push(id); // any other live free model
  return ranked;
}
async function refreshFreeModels() {
  try {
    const r = await fetch("https://openrouter.ai/api/v1/models", { signal: AbortSignal.timeout(20_000) });
    const d = await r.json();
    const ranked = pickFreeModels(d?.data);
    if (!ranked.length) return false;
    if (!process.env.OR_FREE_FAST) OR_FREE_FAST = ranked[0];
    if (!process.env.OR_FREE_DRAFT) OR_FREE_DRAFT = ranked[1] || ranked[0];
    for (const role of ["agent", "tools", "conversation", "fast"]) {
      FALLBACK_CHAINS[role] = buildFallbackChain(MODELS[role]);
    }
    console.log(`[free] live free models: ${OR_FREE_FAST} / ${OR_FREE_DRAFT} (${ranked.length} listed)`);
    return true;
  } catch (e) {
    console.warn("[free] refresh failed, keeping defaults:", e.message);
    return false;
  }
}
// Optional provider pinning: OpenRouter prices the SAME model differently per
// provider (GLM Flash $0.15 default route vs $0.075 DeepInfra 50%-off vs
// $0.08385 StreamLake 44%-off). Set OR_PROVIDER_ONLY=deepinfra (or comma list)
// to lock cheap reliable endpoints; empty = auto-route (most reliable).
// OR_PROVIDER_SORT=throughput|latency optionally reorders within the set.
const OR_PROVIDER_ONLY = (process.env.OR_PROVIDER_ONLY || "").split(",").map(s => s.trim()).filter(Boolean);
const OR_PROVIDER_SORT = process.env.OR_PROVIDER_SORT || "";
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;
const ZEN_API_KEY = process.env.OPENCODE_ZEN_API_KEY || "";
const HAS_ZEN = !!ZEN_API_KEY;
const ZEN_MODEL = process.env.OPENCODE_ZEN_MODEL || "glm-5.3-flash";
const MISTRAL_MODEL = process.env.MISTRAL_MODEL || "mistral-large-latest";
const MISTRAL_FAST_MODEL = process.env.MISTRAL_FAST_MODEL || "mistral-small-latest";
const TWELVE_DATA_KEY = process.env.TWELVE_DATA_KEY || "";
// Live-feed (24/7 WS scanning): Finnhub key covers forex/metals/stocks streaming
// (OANDA:XAU_USD etc.). Crypto needs no key (Binance public WS); forex/metals
// fall back to keyless Dukascopy polling when FINNHUB_KEY is unset.
const FINNHUB_KEY = process.env.FINNHUB_KEY || "";
const FEED_ENABLED = process.env.FEED_ENABLED !== "0";
const FEED_BACKFILL_BUDGET = Number(process.env.FEED_BACKFILL_BUDGET || 200);
// Pocket fixed-expiry signals (manual execution only — never auto-trades).
const POCKET_ENABLED = process.env.POCKET_ENABLED !== "0";
const POCKET_EXPIRY_MIN = Number(process.env.POCKET_EXPIRY_MIN || 5);
const POCKET_PAYOUT = Number(process.env.POCKET_PAYOUT || 0.8); // broker payout assumption for stats
// Deriv execution (Rise/Fall). DEMO default: DERIV_DEMO_TOKEN active.
// Real money requires BOTH DERIV_REAL_TOKEN and DERIV_LIVE=1 — otherwise a
// real token is refused at the executor. Auto-execution itself needs
// POCKET_DERIV_AUTO=1 (default 0 = signals + paper only).
const DERIV_APP_ID = process.env.DERIV_APP_ID || "1089";
const DERIV_LIVE = process.env.DERIV_LIVE === "1";
const DERIV_DEMO_TOKEN = process.env.DERIV_DEMO_TOKEN || "";
const DERIV_REAL_TOKEN = process.env.DERIV_REAL_TOKEN || "";
const POCKET_DERIV_AUTO = process.env.POCKET_DERIV_AUTO !== "0"; // demo auto by default (virtual funds only — real needs DERIV_LIVE=1)
const POCKET_DERIV_STAKE = Number(process.env.POCKET_DERIV_STAKE || 1);
const POCKET_DERIV_MAX_DAY = Number(process.env.POCKET_DERIV_MAX_DAY || 20);
const SANDBOX_URL = process.env.SANDBOX_URL || "https://sandbox-rexv.onrender.com";
// The sandbox service requires its own auth token. The AI never sees this —
// the server attaches it when forwarding run_code calls.
const SANDBOX_AUTH = process.env.SANDBOX_AUTH || process.env.SANDBOX_AUTH_TOKEN || "";
const AUTH_TOKEN = process.env.AUTH_TOKEN || process.env.FRIT_AUTH_TOKEN || "";
// FIX (F-11): shared secret for provider completion webhooks (Sync Labs calls
// us back). Embedded in the registered webhookUrl as ?secret=…; requests
// without it are rejected once the secret is set. Unset = legacy behavior.
const WEBHOOK_SECRET = process.env.WEBHOOK_SECRET || "";
// FIX (F-10): browser cross-site allowlist (comma-separated). Native Android
// clients send no Origin header and always pass. Empty = deny all browsers.
const CORS_ORIGINS = String(process.env.CORS_ORIGINS || "").split(",").map((s) => s.trim()).filter(Boolean);
// FIX (F-10b): when set in production, master-token callers may no longer
// self-assert user_id on money/quota paths — identity must come from the
// per-install key (api_keys). Default unset = legacy dev behavior preserved.
const ENFORCE_BOUNDED_MONEY = process.env.ENFORCE_BOUNDED_MONEY === "1";
// Admin actions (approve/mint/refunds/ledger) use a DIFFERENT token so a
// leaked app token can never mint tiers or credit wallets. Falls back to
// AUTH_TOKEN only if unset (dev) — production must set ADMIN_TOKEN.
const ADMIN_TOKEN = process.env.ADMIN_TOKEN || "";
const HAS_ADMIN_SPLIT = !!ADMIN_TOKEN;
const MT5_BRIDGE_URL = process.env.MT5_BRIDGE_URL || "";
// Monetization (manual bank transfer — fill these, never commit real ones):
const PAY_BANK_NAME = process.env.PAY_BANK_NAME || "";
const PAY_ACCOUNT_NUMBER = process.env.PAY_ACCOUNT_NUMBER || "";
const PAY_ACCOUNT_NAME = process.env.PAY_ACCOUNT_NAME || "";

// ========================== VALIDATION =============================
if (!AUTH_TOKEN) {
  console.error("[FATAL] AUTH_TOKEN missing — mandatory for agentic security!");
  process.exit(1);
}
if (!HAS_OR && !MISTRAL_API_KEY) {
  console.error("[FATAL] All provider keys missing — set OPENROUTER_API_KEY in .env");
  process.exit(1);
}
if (!HAS_ADMIN_SPLIT) {
  console.warn("[WARN] ADMIN_TOKEN unset — admin routes fall back to AUTH_TOKEN. Set ADMIN_TOKEN in production!");
}
if (EPHEMERAL_DB) {
  console.warn("[WARN] frit.db is on EPHEMERAL disk — wallets/subs/ledgers WILL be wiped on restart. Mount a persistent disk at /data or set DB_PATH.");
} else {
  console.log(`[DB] using ${DB_PATH}`);
}

// ========================== MIDDLEWARE ==============================
// Identity: per-user API keys (sha256 stored). requireAuth resolves the caller
// to req.authUser. Master AUTH_TOKEN still works (dev/legacy) but then the
// caller may assert any user_id. Admin routes need ADMIN_TOKEN separately.
function hashKey(k) { return crypto.createHash("sha256").update(String(k)).digest("hex"); }
function requireAuth(req, res, next) {
  const header = req.headers["authorization"] || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : "";
  if (AUTH_TOKEN && token === AUTH_TOKEN) {
    req.authUser = null; // master token: legacy path, user_id self-asserted
    return next();
  }
  if (token) {
    const row = db.prepare("SELECT user_id FROM api_keys WHERE key_hash = ?").get(hashKey(token));
    if (row) {
      req.authUser = row.user_id;
      return next();
    }
  }
  return res.status(401).json({ error: "Unauthorized" });
}
// Bound identity: user-key callers CANNOT spoof user_id; master-token callers
// (dev) may. Every money/quota endpoint must use this, never raw body.
// FIX (F-10b): with ENFORCE_BOUNDED_MONEY=1 (production), a master-token caller
// may no longer self-assert a user_id on money paths — the per-install key is
// required. Unset (default) keeps legacy dev behavior.
const _boundWarned = new Set();
function boundUser(req, fallback = "default") {
  if (req.authUser) return req.authUser;
  const asserted = String(req.body?.user_id || req.query?.user_id || "");
  if (ENFORCE_BOUNDED_MONEY && asserted && asserted !== fallback) {
    if (!_boundWarned.has(req.ip)) {
      _boundWarned.add(req.ip);
      console.warn(`[identity] ${req.ip} asserted user_id '${asserted}' with master token while ENFORCE_BOUNDED_MONEY=1 — refused`);
      if (_boundWarned.size > 1000) _boundWarned.clear();
    }
    const err = new Error("Bound identity required: per-install API key needed on this endpoint");
    err.status = 403;
    throw err;
  }
  return asserted || fallback;
}
// Simple per-user rate limits (in-memory; single instance).
const _rl = new Map();
function rateLimit({ windowMs, max, costly = false }) {
  return (req, res, next) => {
    const id = req.authUser || req.ip || "anon";
    const key = `${costly ? "c:" : ""}${id}`;
    const now = Date.now();
    let rec = _rl.get(key);
    if (!rec || now > rec.reset) { rec = { n: 0, reset: now + windowMs }; _rl.set(key, rec); }
    if (++rec.n > max) return res.status(429).json({ error: "Rate limited — slow down" });
    next();
  };
}
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of _rl.entries()) if (now > v.reset) _rl.delete(k);
}, 60_000).unref?.();
const limitNormal = rateLimit({ windowMs: 60_000, max: 90 });
const limitCostly = rateLimit({ windowMs: 60_000, max: 12, costly: true });
function requireAdmin(req, res, next) {
  const header = req.headers["authorization"] || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : "";
  const want = HAS_ADMIN_SPLIT ? ADMIN_TOKEN : AUTH_TOKEN;
  if (!want || token !== want) return res.status(401).json({ error: "Unauthorized" });
  next();
}

app.set("trust proxy", 1);
app.use(express.json({ limit: "40mb" }));
  app.use(express.urlencoded({ limit: "40mb", extended: true }));
app.use(helmet({ contentSecurityPolicy: false }));
// FIX (F-10): was `cors({ origin: "*" })` — every browser on earth could read
// any authenticated response if it ever got hold of a token, and the API intent
// is native-app-first anyway. Now: no-Origin requests (native Android, curl,
// same-origin) always pass; browser Origins must match CORS_ORIGINS exactly.
// An empty allowlist (default) gives NO Access-Control-Allow-Origin header to
// any browser cross-site client — native clients are unaffected.
app.use(cors({
  origin(origin, cb) {
    if (!origin || CORS_ORIGINS.includes(origin)) return cb(null, true);
    return cb(null, false);
  },
}));
app.use(morgan(process.env.NODE_ENV === "production" ? "combined" : "dev"));

// ============ CRASH INGEST (fix F-15 server side) ============
// Android devs' CrashGuard writes crashes to filesDir/crashes and NOTHING ever
// reads them — remote devices were invisible. This endpoint is the ingest.
// It is deliberately unauthenticated (a crash may happen before auth works),
// rate-limited per IP, and the client scrubs PII through PrivacyScrubber
// before sending. Admin reads the feed via /admin/crashes.
db.exec(`
  CREATE TABLE IF NOT EXISTS crash_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    version TEXT,
    model TEXT,
    type TEXT,
    message TEXT,
    stack TEXT,
    screen TEXT,
    created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_crash_created ON crash_reports(created_at DESC);
`);
const limitCrash = rateLimit({ windowMs: 60_000, max: 10 });
app.post("/crash", limitCrash, (req, res) => {
  try {
    const b = req.body || {};
    const type = String(b.type || "crash").slice(0, 60);
    const message = String(b.message || "").slice(0, 500);
    if (!message && !b.stack) return res.status(400).json({ ok: false, error: "message or stack required" });
    db.prepare(
      "INSERT INTO crash_reports (user_id, version, model, type, message, stack, screen, created_at) VALUES (?,?,?,?,?,?,?,?)"
    ).run(
      String(b.user_id || "").slice(0, 64) || null,
      String(b.version || "").slice(0, 40) || null,
      String(b.model || "").slice(0, 80) || null,
      type, message,
      String(b.stack || "").slice(0, 4000) || null,
      String(b.screen || "").slice(0, 300) || null,
      Date.now(),
    );
    res.json({ ok: true });
  } catch (e) {
    res.status(200).json({ ok: false, error: e.message }); // never signal harder than the client can handle
  }
});
app.get("/admin/crashes", requireAdmin, (req, res) => {
  try {
    const limit = Math.min(Number(req.query.limit) || 50, 200);
    const rows = db.prepare("SELECT * FROM crash_reports ORDER BY created_at DESC LIMIT ?").all(limit);
    res.json({ ok: true, crashes: rows, total: db.prepare("SELECT COUNT(*) AS n FROM crash_reports").get().n });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ======================= MODELS =======================
// Cost stack (OpenRouter-only, verified Oct 2026):
// primary brain = z-ai/glm-5.3-flash ($0.15/$0.50, 1M ctx, Terminal-Bench 84.3).
// NOT z-ai/glm-5.3 ($1.40/$4.40, 9x cost). No DeepSeek, no Groq anywhere.
// fast = openai/gpt-oss-20b ($0.02/$0.10) — the gpt-oss-20b-class slot.
// draft = qwen 30B-class cheap. Free :free models sit AFTER paid in chains.
const GO_MODEL = process.env.OPENCODE_GO_MODEL || "glm-5.3-flash";
let AGENT_PRIMARY = process.env.AGENT_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : (HAS_ZEN ? `go:${GO_MODEL}` : MISTRAL_MODEL));
const MODELS = {
  vision: process.env.VISION_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : "mistral-large-latest"),
  agent: AGENT_PRIMARY,
  conversation: process.env.CONVERSATION_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : (HAS_ZEN ? `go:${GO_MODEL}` : MISTRAL_MODEL)),
  tools: process.env.TOOLS_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : HAS_ZEN ? `go:${GO_MODEL}` : MISTRAL_MODEL),
  coding: process.env.CODING_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : "codestral-latest"),
  fast: process.env.FAST_MODEL || (HAS_OR ? `openrouter:${OR_FAST}` : MISTRAL_FAST_MODEL),
  voxtral: process.env.MISTRAL_VOXTRAIL_MODEL || process.env.MISTRAL_VOXTRAL_MODEL || "voxtral-mini-transcribe-realtime",
  local: process.env.LOCAL_MODEL || "gemma-3n-e2b", // on-device offline fallback (Android), not called here
};

function pickModel({ hasImage = false, mode = "auto", taskType = "general" } = {}) {
  if (mode === "vision" || hasImage) return MODELS.vision;
  if (mode === "fast") return MODELS.fast;
  if (mode === "tools") return MODELS.tools;
  if (mode === "agent") return MODELS.agent;
  if (mode === "auto") {
    const complex = ["automation", "planning", "analysis", "trading", "multistep"];
    const simple = ["chat", "greeting", "quick_question"];
    if (complex.includes(taskType)) return MODELS.agent;
    if (simple.includes(taskType)) return MODELS.fast;
  }
  return MODELS.conversation;
}

function buildFallbackChain(primary) {
  // Paid OpenRouter first, then FREE :free quota (1000/day), then legacy keys.
  // Free is preemptible/429-prone — never primary, always fallback.
  const chain = [primary];
  if (HAS_OR) {
    if (primary !== `openrouter:${OR_PRIMARY}`) chain.push(`openrouter:${OR_PRIMARY}`);
    if (primary !== `openrouter:${OR_FAST}`) chain.push(`openrouter:${OR_FAST}`);
    chain.push(`openrouter:${OR_FREE_FAST}`);
    chain.push(`openrouter:${OR_FREE_DRAFT}`);
  }
  if (HAS_ZEN) chain.push(`go:${GO_MODEL}`);
  // Only offer a Mistral fallback when there IS a key for it. The check used to
  // be on MISTRAL_MODEL, which is always truthy (it defaults to
  // "mistral-large-latest") — so with no Mistral key the chain ended on a
  // mistral model that answered "Bearer undefined" and burned a retry.
  if (MISTRAL_API_KEY && MISTRAL_MODEL) chain.push(MISTRAL_MODEL);
  return [...new Set(chain)];
}

function buildVisionFallbackChain(primary) {
  const chain = [primary];
  if (HAS_OR && primary !== `openrouter:${OR_PRIMARY}`) chain.push(`openrouter:${OR_PRIMARY}`); // native multimodal (z-ai/glm-5.3-flash)
  if (MISTRAL_API_KEY) chain.push("mistral-large-latest");
  return [...new Set(chain)];
}

// Bulk-text work (a design brief, a shot list): the DRAFT model first because
// it answers fast and does not spend minutes reasoning. The reasoning model
// (OR_FAST) is the fallback, not the primary — it blew the brief stage's
// time budget on the first live runs.
function buildDraftFallbackChain() {
  const chain = [];
  if (HAS_OR) {
    if (OR_DRAFT) chain.push(`openrouter:${OR_DRAFT}`);
    if (OR_PRIMARY && !chain.includes(`openrouter:${OR_PRIMARY}`)) chain.push(`openrouter:${OR_PRIMARY}`);
    if (OR_FAST && !chain.includes(`openrouter:${OR_FAST}`)) chain.push(`openrouter:${OR_FAST}`);
  }
  if (!chain.length) chain.push(MODELS.conversation);
  return [...new Set(chain)];
}

const FALLBACK_CHAINS = {
  vision: buildVisionFallbackChain(MODELS.vision),
  agent: buildFallbackChain(MODELS.agent),
  tools: buildFallbackChain(MODELS.tools),
  conversation: buildFallbackChain(MODELS.conversation),
  fast: buildFallbackChain(MODELS.fast),
  draft: buildDraftFallbackChain(),
};
// The :free roster rotates — resolve the live list at boot, re-check every 6h.
await refreshFreeModels();
setInterval(() => { void refreshFreeModels(); }, 6 * 3600e3).unref?.();
// Re-resolve every model slug (overrides > env > defaults) and rebuild the
// role chains. Runs at boot and after every Creator-console model change, so
// new OpenRouter slugs take effect instantly with no restart.
function refreshModelConfig() {
  OR_PRIMARY = M("OR_PRIMARY_MODEL");
  OR_FAST = M("OR_FAST_MODEL");
  OR_DRAFT = M("OR_DRAFT_MODEL");
  OR_IMAGE_MODEL = M("OR_IMAGE_MODEL");
  OR_STT_MODEL = M("OR_STT_MODEL");
  OR_DESIGN_MODEL = M("OR_DESIGN_MODEL");
  OR_LAYER_MODEL = M("OR_LAYER_MODEL");
  OR_VIDEO_MODEL = M("OR_VIDEO_MODEL");
  OR_GENJUTSU_MODEL = M("OR_GENJUTSU_MODEL");
  OR_GRAPHICS_MODEL = M("OR_GRAPHICS_MODEL") || OR_DESIGN_MODEL;
  OR_TYPE_MODEL = M("OR_TYPE_MODEL");
  OR_VECTOR_MODEL = M("OR_VECTOR_MODEL");
  OR_AVATAR_MODEL = M("OR_AVATAR_MODEL");
  // :free keys: never clobber the live auto-resolved roster — only pin when
  // the creator (or env) explicitly says so.
  const pinFast = overrideOf("OR_FREE_FAST") || process.env.OR_FREE_FAST;
  const pinDraft = overrideOf("OR_FREE_DRAFT") || process.env.OR_FREE_DRAFT;
  if (pinFast) OR_FREE_FAST = pinFast;
  if (pinDraft) OR_FREE_DRAFT = pinDraft;
  AGENT_PRIMARY = process.env.AGENT_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : (HAS_ZEN ? `go:${GO_MODEL}` : MISTRAL_MODEL));
  MODELS.vision = process.env.VISION_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : "mistral-large-latest");
  MODELS.agent = AGENT_PRIMARY;
  MODELS.conversation = process.env.CONVERSATION_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : (HAS_ZEN ? `go:${GO_MODEL}` : MISTRAL_MODEL));
  MODELS.tools = process.env.TOOLS_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : HAS_ZEN ? `go:${GO_MODEL}` : MISTRAL_MODEL);
  MODELS.coding = process.env.CODING_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : "codestral-latest");
  MODELS.fast = process.env.FAST_MODEL || (HAS_OR ? `openrouter:${OR_FAST}` : MISTRAL_FAST_MODEL);
  FALLBACK_CHAINS.vision = buildVisionFallbackChain(MODELS.vision);
  FALLBACK_CHAINS.draft = buildDraftFallbackChain();
  for (const role of ["agent", "tools", "conversation", "fast"]) {
    FALLBACK_CHAINS[role] = buildFallbackChain(MODELS[role]);
  }
}
refreshModelConfig();

let PHONE_UI_SKILL = "";
try {
  PHONE_UI_SKILL = readFileSync(new URL("../skills/phone-ui.md", import.meta.url), "utf8").trim();
} catch (e) {
  console.warn("[skills] phone-ui.md not loaded:", e.message);
}

const APP_FLOW_KEYWORDS = [
  { keys: ["mt5", "metatrader"], file: "mt5-place-order.md" },
  { keys: ["voice note", "voicenote", "voice message"], file: "whatsapp-voicenote.md" },
  { keys: ["opay"], file: "opay-transfer.md" },
  { keys: ["whatsapp"], file: "whatsapp-send.md" },
];
function loadAppFlows(goal = "") {
  const g = String(goal || "").toLowerCase();
  const out = [];
  for (const { keys, file } of APP_FLOW_KEYWORDS) {
    if (!keys.some(k => g.includes(k))) continue;
    try {
      const text = readFileSync(new URL(`../skills/flows/${file}`, import.meta.url), "utf8").trim();
      if (text) out.push(`APP FLOW (${file}):\n${text}`);
    } catch (_) { /* not written yet — skip */ }
  }
  return out.join("\n\n");
}

// Same signature as mistralChat, but walks a role's fallback chain on failure
// (e.g. Groq rate limit exhausted) instead of bubbling the error straight up.
async function chatWithFallback(role, { messages, tools = null, tool_choice = "auto", temperature, max_tokens, retries, timeoutMs }) {
  const chain = FALLBACK_CHAINS[role] || [MODELS[role] || MODELS.conversation];
  const attempts = [];
  for (let i = 0; i < chain.length; i++) {
    const model = chain[i];
    try {
      const out = await mistralChat({ model, messages, tools, tool_choice, temperature, max_tokens, retries, timeoutMs });
      // An EMPTY completion is not a success: it is a model that answered
      // with nothing (a reasoning model that spent the whole budget on
      // thinking, or a provider quirk). It must move the chain on to the
      // next model, exactly like an error does — otherwise the caller gets
      // "" and silently falls back to a default. A tool call is a real
      // answer even with empty content.
      const msg = out?.choices?.[0]?.message;
      const content = typeof msg?.content === "string"
        ? msg.content
        : (Array.isArray(msg?.content) ? msg.content.map((p) => (typeof p === "string" ? p : (p?.text || ""))).join("") : "");
      if (!content.trim() && !(msg?.tool_calls || []).length) {
        throw new Error("empty completion");
      }
      if (i > 0) console.warn(`[chatWithFallback] role=${role} primary model unavailable - recovered on ${model} (normal fallback)`);
      return out;
    } catch (err) {
      attempts.push(`${model}: ${err.message}`);
      console.warn(`[chatWithFallback] role=${role} model=${model} unavailable: ${err.message}`);
    }
  }

  const allRateLimited = attempts.every(a => /rate limit|429|tpm|too many requests/i.test(a));
  const err = new Error(
    allRateLimited
      ? `All AI providers are currently rate-limited (${chain.join(", ")}). Please wait a bit and try again.`
      : `All AI providers failed for role "${role}": ${attempts.join(" | ")}`
  );
  err.code = "ALL_MODELS_UNAVAILABLE";
  err.attempts = attempts;
  throw err;
}

// ======================= IN-MEMORY CACHE =======================
const _cache = new Map();
setInterval(() => {
  const now = Date.now();
  for (const [key, entry] of _cache.entries()) {
    if (now > entry.exp) _cache.delete(key);
  }
}, 5 * 60 * 1000);

function cacheGet(key) {
  const entry = _cache.get(key);
  if (!entry) return null;
  if (Date.now() > entry.exp) {
    _cache.delete(key);
    return null;
  }
  return entry.val;
}

function cacheSet(key, val, ttlMs = 60000) {
  _cache.set(key, { val, exp: Date.now() + ttlMs });
}

// ========================= HELPERS ==============================
function safeJsonParse(input, fallback = null) {
  try { return JSON.parse(input); } catch { return fallback; }
}

function truncateText(text, maxLen = 1200) {
  const s = String(text || "");
  return s.length > maxLen ? s.slice(0, maxLen) : s;
}

function summarizeMemory(memory = [], maxItems = 6, maxChars = 700) {
  if (!Array.isArray(memory) || !memory.length) return [];
  const picked = [];
  let used = 0;
  for (const item of memory) {
    const s = truncateText(item, 180);
    if (!s || picked.length >= maxItems || used + s.length > maxChars) break;
    picked.push(s);
    used += s.length;
  }
  return picked;
}

function buildMemoryBlock(memory = []) {
  const compact = summarizeMemory(memory, 6, 700);
  if (!compact.length) return "";
  return `User memory:\n${compact.join("\n")}`;
}

// ========================= PROVIDER ADAPTERS ===========================
const MISTRAL_BASE = "https://api.mistral.ai/v1";

// Resolve which provider a model string belongs to.
// "openrouter:..." -> OpenRouter (single gateway for LLM/STT/image/vision)
// "zen:..."/"go:..." -> OpenCode Zen/Go (legacy opt fallback)
// otherwise    -> Mistral (OpenAI-compatible chat completions, supports tool calls)
function resolveProvider(model) {
  if (typeof model === "string" && model.startsWith("openrouter:")) return "openrouter";
  if (typeof model === "string" && model.startsWith("go:")) return "go";
  if (typeof model === "string" && model.startsWith("zen:")) return "zen";
  return "mistral";
}

async function mistralChat({ model, messages, tools = null, temperature = 0.3, max_tokens = 1600, tool_choice = "auto", retries = 3, timeoutMs = 90_000 }) {
  const provider = resolveProvider(model);

  const base = provider === "openrouter" ? "https://openrouter.ai/api/v1/chat/completions"
    : provider === "go" ? "https://opencode.ai/zen/go/v1/chat/completions"
    : provider === "zen" ? "https://opencode.ai/zen/v1/chat/completions"
    : `${MISTRAL_BASE}/chat/completions`;
  const apiKey = provider === "openrouter" ? OPENROUTER_API_KEY : (provider === "zen" || provider === "go") ? ZEN_API_KEY : MISTRAL_API_KEY;
  // F-26: a provider with no key used to be called as "Bearer undefined",
  // which came back as a confusing 401 instead of a clear "this provider is
  // not configured". Fail fast so the fallback chain moves on immediately.
  if (!apiKey) throw new Error(`no API key configured for provider "${provider}" (model "${model}")`);

  const cleanModel = provider === "go" ? model.slice(3)
    : provider === "openrouter" ? model.slice("openrouter:".length)
    : provider === "zen" ? model.slice(model.indexOf(":") + 1)
    : model;

  const body = { model: cleanModel, messages, temperature, max_tokens };
  if (provider === "openrouter" && (OR_PROVIDER_ONLY.length || OR_PROVIDER_SORT)) {
    body.provider = {
      ...(OR_PROVIDER_ONLY.length ? { only: OR_PROVIDER_ONLY } : {}),
      ...(OR_PROVIDER_SORT ? { sort: OR_PROVIDER_SORT } : {}),
      allow_fallbacks: true,
    };
  }
  if (tools?.length) {
    body.tools = tools.map(t => ({
      type: "function",
      function: { name: t.function.name, description: t.function.description, parameters: t.function.parameters },
    }));
    body.tool_choice = tool_choice;
  }

  let lastError;
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const res = await fetch(base, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          ...(provider === "openrouter" ? { "HTTP-Referer": "https://frit.local", "X-Title": "FRIT" } : {}),
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs), // hung providers must not pin workers forever
      });
      const data = await res.json();
      if (!res.ok) {
        lastError = new Error(data?.message || data?.error?.message || JSON.stringify(data));
        const retryable = res.status === 429 || (res.status === 400 && /tool_use_failed|Failed to call a function/i.test(lastError.message));
        if (retryable && attempt < retries) {
          // OpenRouter/Mistral 429 bodies say "Please try again in 11.495s" — honor
          // that instead of a fixed backoff that may be shorter than needed.
          // Free :free models 429 far more often (spare capacity) — falls through
          // to next chain link via chatWithFallback.
          const hinted = lastError.message.match(/try again in ([\d.]+)s/i);
          const waitMs = hinted ? Math.ceil(parseFloat(hinted[1]) * 1000) + 500 : attempt * 2000;
          await new Promise(r => setTimeout(r, waitMs));
          continue;
        }
        throw lastError;
      }
      return data;
    } catch (err) {
      lastError = err;
      // F-08: the final error IS rethrown below, but an intermediate failure
      // was invisible — a provider that is down looked like a clean 503.
      console.warn(`[mistralChat] ${provider} attempt ${attempt}/${retries} failed: ${err.message}`);
      if (attempt < retries) await new Promise(r => setTimeout(r, attempt * 1000));
    }
  }
  throw lastError || new Error(`${provider} API failed after retries`);
}

// ==================== CORE AGENT LOOP ENGINE ====================
// ONE loop. The brain plans against a persisted ledger; server-side tools
// (run_code, search_web, market data, ...) execute HERE and feed back into the
// same brain turn; only Android-UI actions are returned as pending_actions for
// the phone edge. Every resume verifies the last action batch with a cheap
// second model and updates the ledger (done / failed / re-plan).
const MAX_SERVER_TOOL_LOOP = 12;

function getActiveSubtask(ledger) {
  if (!Array.isArray(ledger) || !ledger.length) return null;
  return ledger.find(t => t.status === "active")
      || ledger.find(t => t.status === "pending")
      || ledger.find(t => t.status === "attention");
}

function markSubtask(ledger, status, note) {
  const t = getActiveSubtask(ledger);
  if (t) { t.status = status; if (note) t.note = note; }
  return t;
}

function getLastFailureNote(ledger) {
  if (!Array.isArray(ledger)) return "";
  const failed = ledger.find(t => t.status === "failed");
  return failed?.note ? `"${failed.description}" — ${failed.note}` : "";
}

function formatTradeMemoryForGoal(goal = "") {
  const found = new Set();
  const m = String(goal).match(/\b(EURUSD|GBPUSD|USDJPY|AUDUSD|XAUUSD|USDCHF|USDCAD|NZDUSD|GBPJPY|EURJPY|EURGBP|BTCUSD|ETHUSD|SOLUSD|BNBUSD|XRPUSD|DOGEUSD|ADAUSD|BTC|ETH|SOL|BNB|XRP|DOGE|ADA)\b/gi);
  if (!m) return "";
  for (const sym of m) found.add(sym.toUpperCase());
  return [...found].map(s => formatTradeMemoryForPrompt(s)).join("\n");
}

// Independent cheap-model grader: "did the expected outcome appear on screen?"
async function verifyLastStep(session, ledger, toolResults, deviceState) {
  try {
    const subtask = getActiveSubtask(ledger) || { description: session.goal || "the task" };
    const screenText = String(deviceState?.screen_text || deviceState?.raw || "").slice(0, 1400);
    const actions = (toolResults || []).map(r => {
      const raw = r.content;
      if (raw == null) return "";
      if (typeof raw === "string") {
        try { return JSON.stringify(JSON.parse(raw)).slice(0, 300); } catch { return raw.slice(0, 300); }
      }
      return JSON.stringify(raw).slice(0, 300);
    }).filter(Boolean).join(" | ");
    const prompt = [
      "You are a UI verification model for a phone automation agent.",
      `The current subtask is: ${subtask.description}`,
      `The agent just performed these actions and got these results: ${actions || "(none)"}`,
      `The current screen text is: ${screenText || "(no screen text available)"}`,
      "Did the subtask's expected outcome appear on screen?",
      'Reply with ONLY a JSON object like {"verified": true, "reason": "..."} where verified is true only if the outcome is clearly visible.',
    ].join("\n");
    const out = await chatWithFallback("fast", { messages: [{ role: "user", content: prompt }], temperature: 0, max_tokens: 150 });
    const text = out.choices?.[0]?.message?.content || "";
    const parsed = safeJsonParse(text.match(/\{[\s\S]*\}/)?.[0] || "", null);
    if (parsed && typeof parsed.verified === "boolean") {
      return { verified: parsed.verified, reason: String(parsed.reason || "").slice(0, 200) };
    }
    return { verified: !/(did not|not visible|not found|failed|error|unable)/i.test(text), reason: text.slice(0, 150) };
  } catch (e) {
    console.warn("[verifyLastStep]", e.message);
    return null;
  }
}

const SETTINGS_KEYWORD = /\b(bluetooth|wi-?fi|mobile data|cellular|airplane|flight mode|location|gps|battery|power saver)\b/i;
const SETTINGS_NAV_LABEL = /^(network|network & internet|connections|connected devices|internet|settings)$/i;

function rewriteSettingsNavigation(tc, goal) {
  const m = String(goal || "").match(SETTINGS_KEYWORD);
  if (!m) return tc;
  const n = tc.function.name;
  const a = tc.function.arguments || {};
  const slowOpen = n === "open_app" && /^settings?$/i.test(String(a.app_name || "").trim());
  const slowNav = (n === "tap_button") && SETTINGS_NAV_LABEL.test(String(a.label || "").trim());
  if (!slowOpen && !slowNav) return tc;
  return { ...tc, function: { ...tc.function, name: "execute_local_action", arguments: { action: `open ${m[1].toLowerCase()}` } } };
}

async function runAgentStep(sessionId, toolResults = null, deviceState = null, opts = {}) {
  const session = db.prepare("SELECT * FROM agent_sessions WHERE id = ?").get(sessionId);
  if (!session) throw new Error("Session not found");

  if (deviceState) {
    db.prepare("UPDATE agent_sessions SET last_device_state = ? WHERE id = ?").run(JSON.stringify(deviceState), sessionId);
  }

  const steps = db.prepare("SELECT * FROM agent_steps WHERE session_id = ? ORDER BY step_n ASC").all(sessionId);
  // Sliding window: long sessions would otherwise explode context + cost.
  // Server keeps full history in SQLite; the brain sees the last 40 steps.
  const windowed = steps.length > 40 ? steps.slice(-40) : steps;
  const messages = windowed.map(s => ({
    role: s.role,
    content: s.content,
    ...(s.tool_calls ? { tool_calls: JSON.parse(s.tool_calls) } : {}),
    ...(s.role === "tool" ? { tool_call_id: s.tool_call_id || `legacy_tool_${s.id}` } : {}),
  }));

  const ledger = JSON.parse(session.task_ledger || "[]");
  const memory = session.memory ? safeJsonParse(session.memory, []) : (Array.isArray(opts.memory) ? opts.memory : []);

  // ---- 1. Append Android tool results (this is a resume) ----
  if (toolResults) {
    for (const res of toolResults) {
      const content = JSON.stringify(res.content);
      messages.push({ role: "tool", tool_call_id: res.tool_call_id, name: res.tool, content });
      db.prepare("INSERT INTO agent_steps (session_id, step_n, role, content, tool_call_id) VALUES (?, ?, ?, ?, ?)").run(
        sessionId, messages.length, "tool", content, res.tool_call_id || null
      );
    }
  }

  // Preserve stored device state (like installed_apps) across resumes
  const stored = session.last_device_state ? safeJsonParse(session.last_device_state, {}) : {};
  const mergedDeviceState = { ...stored, ...(deviceState || {}) };
  if (deviceState) {
    db.prepare("UPDATE agent_sessions SET last_device_state = ? WHERE id = ?").run(JSON.stringify(mergedDeviceState), sessionId);
  }

  // ---- 2. Verification of the last action batch (independent second model) ----
  let verification = null;
  if (toolResults && deviceState) {
    verification = await verifyLastStep(session, ledger, toolResults, mergedDeviceState);
    if (verification) {
      const active = getActiveSubtask(ledger);
      if (active && active.status !== "done" && active.status !== "failed") {
        if (verification.verified) {
          active.status = "done";
          active.note = `verified: ${verification.reason}`;
        } else {
          active.fail_count = (active.fail_count || 0) + 1;
          active.note = `verification failed (${active.fail_count}x): ${verification.reason}`;
          if (active.fail_count >= 2) active.status = "failed";
        }
      }
    }
  }

  // ---- 3. Build system prompt (device state + memory + ledger + trade memory) ----
  const activeSubtask = getActiveSubtask(ledger);
  const deviceStateForPrompt = mergedDeviceState;
  const systemPrompt = buildAutomationSystemPrompt({
    deviceState: deviceStateForPrompt,
    memory,
    ledger,
    goal: session.goal,
    tradeMemory: formatTradeMemoryForGoal(session.goal),
    verification,
    lastFailure: getLastFailureNote(ledger),
    userProfile: mergedDeviceState?.user_profile || null,
  }) + `\n\nCURRENT OBJECTIVE: ${activeSubtask?.description || session.goal}`;

  const fullMessages = [{ role: "system", content: systemPrompt }, ...messages];

  // If there's a user-attached image in deviceState, add it to the conversation context
  if (deviceStateForPrompt.user_image && steps.length === 0) {
    fullMessages.push({
      role: "user",
      content: [
        { type: "text", text: "I've attached an image for this task." },
        { type: "image_url", image_url: { url: `data:image/jpeg;base64,${deviceStateForPrompt.user_image}` } }
      ]
    });
  }

  // ---- 4. Brain loop — server-side tools execute here, Android tools return ----
  let lastAssistantText = "";
  let lastToolCalls = [];
  let androidPending = [];
  const turnArtifacts = []; // files produced by run_code this turn, surfaced to the client

  const hasImage = !!(deviceStateForPrompt.user_image && steps.length === 0);
  const modelToUse = pickModel({ hasImage, mode: opts.mode || "auto", taskType: "automation" });
  // Route through the fallback chain for whichever role this resolved to, so an
  // OpenRouter rate limit / free-tier 429 doesn't kill the whole turn.
  const modelRole = modelToUse === MODELS.vision ? "vision"
    : modelToUse === MODELS.agent ? "agent"
    : modelToUse === MODELS.fast ? "fast"
    : modelToUse === MODELS.tools ? "tools"
    : "conversation";

  for (let iter = 0; iter <= MAX_SERVER_TOOL_LOOP; iter++) {
    const out = await chatWithFallback(modelRole, { messages: fullMessages, tools: AGENT_TOOLS, max_tokens: 4000 });
    const msg = out.choices[0].message;
    lastAssistantText = msg.content || "";
    lastToolCalls = extractToolCalls(msg);

    db.prepare("INSERT INTO agent_steps (session_id, step_n, role, content, tool_calls) VALUES (?, ?, ?, ?, ?)").run(
      sessionId, fullMessages.length, "assistant", lastAssistantText, lastToolCalls.length ? JSON.stringify(msg.tool_calls) : null
    );
    fullMessages.push({ role: "assistant", content: lastAssistantText, tool_calls: msg.tool_calls });

    const serverCalls = lastToolCalls.filter(tc => SERVER_SIDE_TOOLS.has(tc.function.name));
    const rawAndroidCalls = lastToolCalls.filter(tc => !SERVER_SIDE_TOOLS.has(tc.function.name));

    // Guard against malformed open_app calls: weaker fallback models sometimes
    // cram the whole task instruction into app_name (e.g. "messenger and reply
    // to the chat that has username fab vicky..."). Catch that here instead of
    // sending garbage to the device — bounce it back to the model to retry.
    const MAX_APP_NAME_LEN = 40;
    const looksLikeSentence = (s) => /\b(and|to|that|reply|check|open|task)\b/i.test(s) && s.split(/\s+/).length > 4;
    const badAppCalls = [];
    const androidCalls = [];
    // Device automation is a paid surface: phone-control tools need the
    // "automation" feature. Blocked calls bounce with an explanatory error
    // (same pattern as malformed calls) so the brain reroutes to chat.
    grantMonthlyCredits(session.user_id);
    const automationTier = activeTier(session.user_id);
    const automationAllowed = planAllows(automationTier, "automation");
    const blockedAutomation = !automationAllowed && rawAndroidCalls.length > 0;
    for (let tc of rawAndroidCalls.map(t => rewriteSettingsNavigation(t, session.goal))) {
      if (!automationAllowed) {
        fullMessages.push({ role: "tool", tool_call_id: tc.id, name: tc.function.name, content: JSON.stringify({
          ok: false,
          error: `Phone automation isn't in your ${automationTier} plan — upgrade to an Automation plan or Pro to control apps on this device. I can still chat, answer questions and analyse markets.`,
        }) });
        continue;
      }
      if (tc.function.name === "open_app") {
        const appName = (tc.function.arguments || {}).app_name || "";
        if (appName.length > MAX_APP_NAME_LEN || looksLikeSentence(appName)) {
          badAppCalls.push(tc);
          continue;
        }

        const resolved = resolveInstalledApp(appName, deviceStateForPrompt);
        if (resolved && resolved !== appName) {
          tc.function.arguments = { ...tc.function.arguments, app_name: resolved };
        }
      }
      androidCalls.push(tc);
    }

    const isLastIter = iter === MAX_SERVER_TOOL_LOOP;
    if (badAppCalls.length) {
      for (const tc of badAppCalls) {
        const content = JSON.stringify({
          ok: false,
          error: "Invalid app_name — it must be ONLY the literal app name (e.g. 'Messenger', 'WhatsApp'), not a sentence or task description. Call open_app again with just the app name, then use separate tool calls (read_screen, tap, type) once it's open to carry out the task.",
        });
        fullMessages.push({ role: "tool", tool_call_id: tc.id, name: tc.function.name, content });
      }
      if (!isLastIter) continue; // let the model self-correct
      // If it's the last iter, we fall through and the serverCalls block (or the final break)
      // will handle the summary turn if needed.
    }

    if (serverCalls.length || (badAppCalls.length && isLastIter)) {
      if (serverCalls.length) {
        for (const tc of serverCalls) {
          const name = tc.function.name;
          const args = tc.function.arguments || {};
          let result;
          const gate = gateLoopTool(session.user_id, name);
          if (!gate.ok) {
            result = { ok: false, error: gate.message };
          } else {
            try {
              result = await runLocalTool(name, args, { deviceState: deviceStateForPrompt });
            } catch (err) {
              result = { ok: false, error: err.message };
            }
            // Bill AFTER success only. Async video: track now, charge on completion.
            if (result?.ok && gate.charge) debitCredits(session.user_id, gate.charge, `agent:${name}`);
            if (result?.ok && name === "generate_video") {
              const pid = result?.data?.id || result?.data?.data?.id;
              trackProviderJob(pid, session.user_id, "clip");
            }
          }
          
          if (name === "run_code" && Array.isArray(result?.data?.artifacts) && result.data.artifacts.length) {
            for (const art of result.data.artifacts) {
              const b64 = art.content_base64 || art.base64 || "";
              if (b64) turnArtifacts.push({ name: art.name, mime: art.mime, size: art.size, base64: b64, content_base64: b64 });
            }
            result = {
              ...result,
              data: {
                ...result.data,
                artifacts: result.data.artifacts.map(a => ({ name: a.name, mime: a.mime, size: a.size, note: "file generated — returned to client separately, not inlined here" })),
              },
            };
          }
          const content = JSON.stringify(result);
          // MEDIA SURFACING (no exceptions): generated images/videos must
          // reach the phone as artifacts — rendered in chat with a Creative
          // Suite link — never only as base64 buried in the tool transcript.
          // (Placed after stringify: the brain keeps the small redacted note;
          // the bytes travel in turnArtifacts instead of the context window.)
          if ((name === "generate_image" || name === "design_image") && result?.ok && result?.data) {
            const d = result.data;
            const rawB64 = d.image_b64 ? String(d.image_b64).replace(/^data:[^,]+,/, "") : "";
            if (rawB64 || d.image_url) {
              turnArtifacts.push({
                name: `generated-${Date.now()}.png`,
                mime: "image/png",
                size: 0,
                base64: rawB64 || null,
                content_base64: rawB64 || null,
                url: d.image_url || null,
              });
              const redacted = {
                ...result,
                data: {
                  ...d,
                  image_b64: rawB64 ? "[image bytes returned to client separately — describe the result to the user, do not paste base64]" : d.image_b64,
                },
              };
              fullMessages.push({ role: "tool", tool_call_id: tc.id, name: name, content: JSON.stringify(redacted) });
              db.prepare("INSERT INTO agent_steps (session_id, step_n, role, content, tool_call_id) VALUES (?, ?, ?, ?, ?)").run(
                sessionId, fullMessages.length, "tool", JSON.stringify(redacted), tc.id
              );
              continue;
            }
          }
          if (name === "generate_video" && result?.ok && result?.data?.job_id) {
            // Async render: no bytes yet — the client shows a tracking card
            // with a Creative Suite link until the video completes.
            turnArtifacts.push({
              name: `video-${result.data.job_id}.mp4`,
              mime: "video/mp4",
              size: 0,
              content: `video_job:${result.data.job_id}`,
            });
          }
          fullMessages.push({ role: "tool", tool_call_id: tc.id, name: name, content });
          db.prepare("INSERT INTO agent_steps (session_id, step_n, role, content, tool_call_id) VALUES (?, ?, ?, ?, ?)").run(
            sessionId, fullMessages.length, "tool", content, tc.id
          );
        }
      }

      if (isLastIter) {
        // execute them anyway, then force a final summary turn with tool_choice:"none"
        const finalOut = await chatWithFallback(modelRole, { messages: fullMessages, tools: AGENT_TOOLS, tool_choice: "none", max_tokens: 4000 });
        lastAssistantText = finalOut.choices[0].message.content || "";
        db.prepare("INSERT INTO agent_steps (session_id, step_n, role, content, tool_calls) VALUES (?, ?, ?, ?, ?)").run(
          sessionId, fullMessages.length, "assistant", lastAssistantText, null
        );
        fullMessages.push({ role: "assistant", content: lastAssistantText });
      } else {
        continue; // feed results back to the brain
      }
    }

    if (blockedAutomation && !serverCalls.length && !isLastIter) continue; // brain reads the block notice and answers in chat
    androidPending = androidCalls;
    break;
  }

  const done = androidPending.length === 0;

  
  const hadActiveTask = ledger.some(
    t => t.status === "active" || t.status === "pending" || t.status === "attention"
  );

  // ---- 5. Ledger + session status on completion ----
  if (done && hadActiveTask) {
    if (verification && !verification.verified) {
      markSubtask(ledger, "attention", verification.reason);
    } else {
      const active = getActiveSubtask(ledger);
      if (active && active.status !== "done" && active.status !== "failed") {
        active.status = "done";
        active.note = "completed";
      }
      for (const t of ledger) {
        if (t.status === "pending" || t.status === "attention") { t.status = "done"; t.note = t.note || "folded into completed run"; }
      }
    }
  }
  const sessionStatus = !hadActiveTask
    ? "chatting"
    : done
      ? (ledger.some(t => t.status === "failed") ? "needs_attention" : "completed")
      : "waiting_for_client";

  db.prepare("UPDATE agent_sessions SET status = ?, task_ledger = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?").run(
    sessionStatus, JSON.stringify(ledger), sessionId
  );

  return {
    sessionId,
    assistant_text: lastAssistantText,
    pending_actions: androidPending.map(tc => ({ id: tc.id, tool: tc.function.name, arguments: tc.function.arguments })),
    done,
    ledger,
    verification,
    artifacts: turnArtifacts, // files (charts, CSVs, reports) from run_code + generated images / video-job cards from the media tools, full bytes included
  };
}

// ==================== AGENT ROUTES ====================
// ==================== STREAMING CHAT SYSTEM PROMPT ====================
// Used by POST /agent/chat/stream (the SSE endpoint). It is deliberately
// simpler than the automation prompt: this path has NO tools, so the model
// must not pretend to have taken actions. It answers in plain prose.
//
// The tone rules mirror CHAT_ROUTER_PROMPT so a streamed reply and a
// non-streamed reply feel like the same assistant.
const CONVERSATION_SYSTEM_PROMPT = `You are FRIT, a helpful AI assistant on the user's phone.

- Reply in plain conversational prose. No markdown headings, no bullet-point essays, no code blocks unless the user explicitly asks for code.
- Mirror the user's live tone and register: casual input gets a short, warm, human answer; formal input (email, job application, bank or complaint wording, "Dear Sir") gets full, structured sentences with no slang.
- Be brief by default. A greeting deserves one or two sentences, not a paragraph. Expand only when the question genuinely needs it.
- You have NO tools in this conversation. If the user asks for something that needs an action (generate an image, look something up live, control the phone), say plainly that they should ask FRIT to do it, and offer the concrete next step. Never claim to have done something you did not do.
- Never invent facts: no fake prices, fake weather, fake news, fake statistics. If you do not know or cannot check, say so.
- Never reveal internal engine diagnostics, tool payloads or raw JSON.
- You were built by the FRIT team, not by another company. If asked who you are, say that.`;

// ==================== CHAT VS TASK PRE-ROUTER ====================
// Pure chat (greetings, small talk, thanks, opinions) must NEVER enter the
// agentic loop — that's what made "hi" produce tool calls. One cheap call to
// the fast model classifies AND answers, so a greeting costs a single turn.
const CHAT_ROUTER_PROMPT = `You are FRIT's message router. Decide whether the user's message is casual conversation or a real task.
CONVERSATION means: greetings, small talk, thanks, "how are you", identity/opinion questions, jokes, or simple questions that need NO tools, NO phone actions, NO data fetching, NO code and NO search. Answer those briefly and warmly.

TASK means: the user wants something DONE — an image, video, logo, poster, or other media generated/edited (always a task, never chit-chat); real information retrieved (weather, market prices, analysis, web research, news), code written/run, a file or webpage created, an app opened or controlled on the phone, a message/call/alarm, a trade, or any multi-step work that needs tools.

STYLE (both kinds): mirror the user's live tone, don't force a fixed personality. A fresh/neutral chat → warm neutral. Official signals (email, job application, bank/complaint, "Dear Sir", formal request) → act official: full sentences, structured, no slang. Casual signals (slang, pidgin, short chatty lines, jokes) → mirror casually and keep it short and human. The stored user profile tone is only the baseline default — the live chat mode always wins.

VAGUE FOLLOW-UPS ("do that for me", "go ahead", "yes", "proceed", "same") ALWAYS reference the history above: if the previous assistant turn proposed, recommended, or described an action, the follow-up is a TASK to perform exactly that — resolve what "that" means from history and do it. Only call it CONVERSATION if there is genuinely nothing actionable anywhere in context. Never answer a follow-up with a generic "let me know what you'd like" when the history already says what they'd like.

Reply with EXACTLY ONE line in this exact format:
CONVERSATION <your brief, friendly reply>
or
TASK`;

async function routeChatOrTask(goal, history = []) {
  const msgs = [
    { role: "system", content: CHAT_ROUTER_PROMPT },
    ...(Array.isArray(history) ? history.slice(-8) : []).map(m => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content: String(m.content || ""),
    })),
    { role: "user", content: String(goal || "") },
  ];
  const out = await chatWithFallback("fast", { messages: msgs, temperature: 0, max_tokens: 220 });
  const text = String(out?.choices?.[0]?.message?.content || "").trim();
  if (/^CONVERSATION\b/i.test(text)) {
    const reply = text.replace(/^CONVERSATION\b/i, "").trim();
    return { isTask: false, reply: reply || "Hey! What can I help you with?" };
  }
  return { isTask: true };
}

function needsEnhancement(goal, history = []) {
  const g = String(goal || "");
  if (!Array.isArray(history) || history.length === 0) return true;
  if (g.length >= 120) return true;
  const t = g.toLowerCase();
  if (/(^|\W)(that|this|it|those|these|yes|yeah|yep|ok(ay)?|sure|go ahead|proceed|do it|continue|next|same)(\W|$)/.test(t)) return true;
  return false;
}

async function enhanceGoalForAgent(rawGoal, history = []) {
  try {
    const histBlock = (Array.isArray(history) ? history.slice(-6) : [])
      .map(m => `${m.role === "assistant" ? "FRIT" : "User"}: ${String(m.content || "").slice(0, 500)}`)
      .join("\n");
    const out = await chatWithFallback("fast", {
      messages: [
        { role: "system", content: "You are a prompt enhancer for FRIT, an Android AI agent with server tools: analyze_market(symbol...), get_market_data(symbol), run_code(code), search_web(query), get_weather, send_whatsapp(contact,message), make_call(number), and phone UI tools (open_app, tap_button, type_text, scroll, read_screen). Rewrite the user's vague command into ONE explicit paragraph: state the intent, name the exact tool/endpoint sequence in order, keep device facts (symbols, lot sizes, names) verbatim. IMPORTANT: the user often says 'that'/'it' meaning something from the recent chat — resolve pronouns using the history below (e.g. 'do that for me' after a research recommendation = perform that research). Output ONLY the rewritten instruction, no preamble." },
        { role: "user", content: `${histBlock ? `Recent chat:\n${histBlock}\n\n` : ""}Latest message: ${String(rawGoal || "")}` }
      ]
    });
    const text = out?.choices?.[0]?.message?.content?.trim();
    if (text && text.length > 10) {
      console.log(`[agent/start] goal enhanced: "${String(rawGoal).slice(0, 120)}" -> "${text.slice(0, 160)}"`);
      return text;
    }
  } catch (e) {
    console.warn("[agent/start] goal enhancement failed, using raw goal:", e.message);
  }
  return rawGoal;
}

// ==================== MARKET DATA ====================
// There is NO keyword shortcut into the trading engine. A chat message that
// merely mentions a symbol, "mt5", "buy" or "trade" must never reach
// mtfStrategy directly — every market answer goes through the agent loop and
// its analyze_market / get_market_data tools, so the brain decides whether a
// tool is actually needed and the user sees a normal AI reply instead of a
// raw engine dump. Trading facts (price/structure/decision) stay available
// inside those tools; the SensitiveGate + confirmation discipline still apply
// to anything that spends money.
const MARKET_SYMBOL_MAP = {
  bitcoin: "BTCUSD", btcusd: "BTCUSD", btc: "BTCUSD",
  ethereum: "ETHUSD", ethusd: "ETHUSD", eth: "ETHUSD",
  solana: "SOLUSD", solusd: "SOLUSD",
  gold: "XAUUSD", xauusd: "XAUUSD", silver: "XAGUSD", xagusd: "XAGUSD",
  eurusd: "EURUSD", gbpusd: "GBPUSD", usdjpy: "USDJPY", usdchf: "USDCHF",
  audusd: "AUDUSD", usdcad: "USDCAD", nzdusd: "NZDUSD",
  nas100: "NAS100", nasdaq: "NAS100", us30: "US30", spx500: "SPX500", us500: "US500",
};
function marketKeyHit(t, key) {
  // Short keys (btc/eth) need word boundaries — "whether" is not Ethereum.
  return key.length <= 3 ? new RegExp(`\\b${key}\\b`).test(t) : t.includes(key);
}
function extractMarketSymbol(text) {
  const t = String(text || "").toLowerCase();
  for (const k of Object.keys(MARKET_SYMBOL_MAP).sort((a, b) => b.length - a.length)) {
    if (marketKeyHit(t, k)) return MARKET_SYMBOL_MAP[k];
  }
  const m = t.match(/\b[a-z]{6,7}\b/);
  if (m) {
    const w = m[0].toUpperCase();
    if (/USD$|JPY$|CHF$|CAD$/.test(w)) return w;
  }
  return null; // no symbol named -> caller decides; never guess BTCUSD
}

app.post("/agent/start", requireAuth, limitNormal, async (req, res) => {
  const { goal, device_state, memory, history } = req.body;
  if (!goal) return res.status(400).json({ error: "Goal required" });
  const uid = boundUser(req);
  const ent = checkEntitlement(uid, "chat");
  if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });

  // Fast pre-route: pure conversation never touches the agentic loop.
  try {
    const routed = await routeChatOrTask(goal, Array.isArray(history) ? history : []);
    if (!routed.isTask) {
      debitCredits(uid, "turn", "chat reply"); // post-success
      return res.json({ done: true, assistant_text: routed.reply });
    }
  } catch (e) {
    // Router failed — fall through to the agent loop rather than erroring out.
    // F-08: log it, because a broken router silently pushes EVERY message
    // down the expensive agent path instead of the cheap chat path.
    console.warn(`[agent/start] chat router failed, falling through to the agent loop: ${e.message}`);
  }

  // Market questions go to the agent loop like everything else. The brain
  // calls analyze_market / get_market_data when it actually needs the data.
  if (!SANDBOX_URL.includes("127.0.0.1")) fetch(`${SANDBOX_URL}/health`).catch(() => {}); // wake sandbox while LLM plans

  const sessionId = `sess_${Date.now()}`;
  const histArr = Array.isArray(history) ? history : [];
  const effectiveGoal = needsEnhancement(goal, histArr)
    ? await enhanceGoalForAgent(goal, histArr)
    : String(goal || "");

  try {
    // Initial Plan / Ledger creation
    const planOut = await chatWithFallback("fast", {
      messages: [
        { role: "system", content: "Break the user's goal into an ordered JSON list of subtasks: [{\"seq\": 1, \"description\": \"...\", \"status\": \"pending\"}]" },
        { role: "user", content: effectiveGoal }
      ]
    });

    let ledger = [];
    try {
      const parsed = JSON.parse(planOut.choices[0].message.content.match(/\[.*\]/s)[0]);
      ledger = (Array.isArray(parsed) ? parsed : []).map((t, i) => ({
        seq: t.seq ?? i + 1,
        description: String(t.description || t.task || effectiveGoal),
        status: t.status || "pending",
      }));
    } catch (e) {
      ledger = [{ seq: 1, description: effectiveGoal, status: "pending" }];
    }

    const safeMemory = Array.isArray(memory) ? memory : [];
    db.prepare("INSERT INTO agent_sessions (id, goal, task_ledger, last_device_state, memory, user_id) VALUES (?, ?, ?, ?, ?, ?)").run(
      sessionId, effectiveGoal, JSON.stringify(ledger), JSON.stringify(device_state || {}), JSON.stringify(safeMemory), uid
    );

    const result = await runAgentStep(sessionId, null, device_state, { memory: safeMemory });
    db.prepare("UPDATE usage_daily SET turns = turns + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    debitCredits(uid, "turn", "agent start"); // post-success: loop failures are not billed
    res.json(result);
  } catch (err) {
    console.error("[agent/start]", err);
    res.status(err.code === "ALL_MODELS_UNAVAILABLE" ? 503 : 500).json({ error: err.message, code: err.code || "UNKNOWN" });
  }
});

app.post("/agent/resume", requireAuth, limitNormal, async (req, res) => {
  const { sessionId, tool_results, device_state } = req.body;
  if (!sessionId || !tool_results) return res.status(400).json({ error: "sessionId and tool_results required" });

  try {
    const sess = db.prepare("SELECT user_id FROM agent_sessions WHERE id = ?").get(sessionId);
    const uid = String(sess?.user_id || "default");
    const ent = checkEntitlement(uid, "chat");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const result = await runAgentStep(sessionId, tool_results, device_state);
    db.prepare("UPDATE usage_daily SET turns = turns + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    debitCredits(uid, "turn", "agent resume"); // post-success: loop failures are not billed
    res.json(result);
  } catch (err) {
    res.status(err.code === "ALL_MODELS_UNAVAILABLE" ? 503 : 500).json({ error: err.message, code: err.code || "UNKNOWN" });
  }
});

app.get("/agent/status", requireAuth, limitNormal, (req, res) => {
  const { sessionId } = req.query;
  const session = db.prepare("SELECT * FROM agent_sessions WHERE id = ?").get(sessionId);
  if (!session) return res.status(404).json({ error: "Not found" });
  if (req.authUser && session.user_id && session.user_id !== req.authUser) {
    return res.status(403).json({ error: "not your session" });
  }

  const steps = db.prepare("SELECT * FROM agent_steps WHERE session_id = ? ORDER BY step_n ASC").all(sessionId);
  res.json({ session, steps });
});

// ==================== STREAMING CHAT (SSE) ====================
// The blocking /agent/start meant the user waited in silence for the whole
// reply to be generated, downloaded and parsed. This endpoint streams the
// assistant's text as the model produces it.
//
// Contract (Server-Sent Events, text/event-stream):
//   event: delta  data: {"text":"partial text"}
//   event: done   data: {"text":"full reply","task":false,"user_id":"..."}
//   event: error  data: {"error":"why it failed"}
//
// HOW IT WORKS: the chat-vs-task router itself is streamed. Its reply starts
// with either "CONVERSATION <text>" or "TASK", so the classification arrives
// in the first few characters and we can decide WHILE the text is still
// arriving:
//   - CONVERSATION -> the rest of the stream is the answer, sent as deltas.
//   - TASK         -> real work needs the agent loop (tool calls, device
//                     actions, verification), so we stop streaming and report
//                     `task: true` in the done frame. The client then runs the
//                     normal /agent/start path. Nothing is lost and no credit
//                     is spent twice.
//
// Billing: one credit per turn, debited AFTER the work succeeds — the same
// rule as /agent/start. A client that disconnects mid-stream is not billed.
app.post("/agent/chat/stream", requireAuth, limitNormal, async (req, res) => {
  const { goal, history, device_state } = req.body || {};
  if (!goal) return res.status(400).json({ error: "Goal required" });
  const uid = boundUser(req);
  const ent = checkEntitlement(uid, "chat");
  if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });

  // SSE headers. `X-Accel-Buffering: no` stops nginx/Render from buffering
  // the whole response — without it the client gets everything at once and
  // the stream is pointless.
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders?.();

  // The client may hang up (user navigated away, network dropped). Abort the
  // upstream fetch so we do not keep paying for tokens nobody will see.
  //
  // NOTE: listen on the RESPONSE, not the request. `req.on("close")` fires as
  // soon as the request body has been consumed (express.json() already read
  // it), which aborted the upstream fetch instantly and silently turned the
  // whole stream into a single buffered chunk. `res.on("close")` fires when
  // the response is finished OR the client disconnected — the semantics we
  // actually want.
  const ac = new AbortController();
  const onClose = () => {
    if (!res.writableEnded) ac.abort();
  };
  res.on("close", onClose);

  let full = "";
  let deltaCount = 0;
  let isTask = false;
  try {
    const chain = FALLBACK_CHAINS.fast || [MODELS.fast].filter(Boolean);
    let lastError = null;
    let decided = false;

    for (const model of chain) {
      const provider = resolveProvider(model);
      const base = provider === "openrouter" ? "https://openrouter.ai/api/v1/chat/completions"
        : provider === "go" ? "https://opencode.ai/zen/go/v1/chat/completions"
        : provider === "zen" ? "https://opencode.ai/zen/v1/chat/completions"
        : `${MISTRAL_BASE}/chat/completions`;
      const apiKey = provider === "openrouter" ? OPENROUTER_API_KEY : (provider === "zen" || provider === "go") ? ZEN_API_KEY : MISTRAL_API_KEY;
      // F-26: skip a provider with no key instead of calling it as
      // "Bearer undefined" and burning a retry on a guaranteed 401.
      if (!apiKey) { lastError = new Error(`no API key configured for provider "${provider}"`); continue; }
      const cleanModel = provider === "go" ? model.slice(3)
        : provider === "openrouter" ? model.slice("openrouter:".length)
        : provider === "zen" ? model.slice(model.indexOf(":") + 1)
        : model;

      const histArr = Array.isArray(history) ? history : [];
      const messages = [
        { role: "system", content: CHAT_ROUTER_PROMPT },
        ...histArr.slice(-8).map((m) => ({
          role: m.role === "assistant" ? "assistant" : "user",
          content: String(m.content || ""),
        })),
        { role: "user", content: String(goal || "") },
      ];

      try {
        const upstream = await fetch(base, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
            ...(provider === "openrouter" ? { "HTTP-Referer": "https://frit.local", "X-Title": "FRIT" } : {}),
          },
          body: JSON.stringify({ model: cleanModel, messages, temperature: 0, max_tokens: 1200, stream: true }),
          signal: ac.signal,
        });
        if (!upstream.ok) {
          const txt = await upstream.text().catch(() => "");
          lastError = new Error(`HTTP ${upstream.status}: ${String(txt).slice(0, 200)}`);
          continue;
        }

        // Buffer the head of the stream until the route is decidable, then
        // release the buffered text (minus the router prefix) as a delta.
        let head = "";
        let headReleased = false;
        const releaseHead = () => {
          if (headReleased) return;
          headReleased = true;
          const body = stripConversationPrefix(head);
          if (body) {
            full += body;
            deltaCount++;
            res.write(sseDelta(body));
          }
          head = "";
        };

        for await (const ev of readProviderStream(upstream, ac.signal)) {
          if (ev.type === "error") {
            lastError = new Error(ev.message);
            break;
          }
          if (ev.type !== "delta") continue;

          if (!decided) {
            head += ev.text;
            const route = classifyStreamRoute(head);
            if (route === ROUTE_TASK) {
              isTask = true;
              decided = true;
              break; // the agent loop handles the rest
            }
            if (route === ROUTE_CONVERSATION) {
              decided = true;
              releaseHead();
            }
            // still unknown -> keep buffering
            continue;
          }
          full += ev.text;
          deltaCount++;
          res.write(sseDelta(ev.text));
        }

        if (isTask) break;
        if (!decided) {
          // The stream ended before we could classify (very short reply).
          // Fall back to the non-streaming router, which is authoritative.
          const routed = await routeChatOrTask(goal, histArr);
          if (routed.isTask) { isTask = true; break; }
          decided = true;
          full = routed.reply;
          deltaCount++;
          res.write(sseDelta(full));
        } else {
          releaseHead();
        }
        break;
      } catch (err) {
        lastError = err;
        if (ac.signal.aborted) break;
        console.warn(`[agent/chat/stream] model=${model} failed: ${err.message}`);
      }
    }

    if (isTask) {
      // Real work: hand it back to the client to run through /agent/start.
      // No credit is spent here — the agent path does its own billing.
      res.write(sseDone("", { task: true, user_id: uid }));
      return;
    }

    if (!full.trim()) {
      // Streaming failed entirely — fall back to the non-streaming path so the
      // user still gets an answer instead of an empty bubble.
      const out = await chatWithFallback("conversation", {
        messages: [
          { role: "system", content: CONVERSATION_SYSTEM_PROMPT },
          ...(Array.isArray(history) ? history.slice(-12) : []).map((m) => ({
            role: m.role === "assistant" ? "assistant" : "user",
            content: String(m.content || ""),
          })),
          { role: "user", content: String(goal || "") },
        ],
        temperature: 0.5,
        max_tokens: 1200,
      });
      full = String(out?.choices?.[0]?.message?.content || "");
      if (!full.trim()) throw lastError || new Error("no response from any model");
      res.write(sseDelta(full));
    }

    db.prepare("UPDATE usage_daily SET turns = turns + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    debitCredits(uid, "turn", "chat stream"); // post-success only
    console.log(`[agent/chat/stream] deltas=${deltaCount} chars=${full.length}`);
    res.write(sseDone(full, { task: false, user_id: uid }));
  } catch (err) {
    console.error("[agent/chat/stream]", err?.message || err);
    try { res.write(sseError(err?.message || "stream failed")); } catch { /* client gone */ }
  } finally {
    res.off("close", onClose);
    try { res.end(); } catch { /* already closed */ }
  }
});


// Transcription via OpenRouter STT (whisper-large-v3 for accuracy).
// POST https://openrouter.ai/api/v1/audio/transcriptions with JSON input_audio.
async function mistralTranscribe(audio_base64, mime_type = "audio/webm") {
  const b64 = String(audio_base64 || "");
  if (!b64 || !OPENROUTER_API_KEY) return "";
  const fmt = String(mime_type || "audio/webm").split(";")[0].trim().split("/")[1] || "wav";
  try {
    const res = await fetch("https://openrouter.ai/api/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://frit.local",
        "X-Title": "FRIT",
      },
      body: JSON.stringify({
        model: OR_STT_MODEL, // openai/whisper-large-v3 (accuracy pick over turbo)
        input_audio: { data: b64, format: fmt },
        language: "en",
      }),
    });
    if (res.ok) {
      const d = await res.json();
      if (d.text) return d.text;
    }
    console.warn(`[Transcribe] OpenRouter STT failed: ${res.status}`);
  } catch (e) { console.warn("[Transcribe] OpenRouter STT error:", e.message); }

  // Fallback: OpenAI Whisper (only if key present)
  if (process.env.OPENAI_API_KEY) {
    try {
      const audioBuffer = Buffer.from(b64, "base64");
      const form = new FormData();
      form.append("file", audioBuffer, { filename: `audio.${fmt}`, contentType: mime_type });
      form.append("model", "whisper-1");
      form.append("response_format", "json");
      const res = await fetch("https://api.openai.com/v1/audio/transcriptions", {
        method: "POST",
        headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}`, ...form.getHeaders() },
        body: form,
      });
      if (res.ok) {
        const d = await res.json();
        if (d.text) return d.text;
      }
      console.warn(`[Transcribe] OpenAI failed: ${res.status}`);
    } catch (e) { console.warn("[Transcribe] OpenAI error:", e.message); }
  }

  return "";
}

// Image generation. ONE gateway: OpenRouter /images. There are no
// per-provider API keys — the catalogue serves 61 image models (including
// the Recraft family), and the request body is built per model family
// because they accept different parameters (Recraft rejects `n` and
// `output_format` and only takes 1:1 / 4:3 / 3:4 / 16:9 / 9:16 / auto).
async function generateImage(prompt, opts = {}) {
  if (!OPENROUTER_API_KEY) throw new Error("OPENROUTER_API_KEY missing");
  const model = opts.model || OR_IMAGE_MODEL;
  const isMing = model.startsWith("inclusionai/ming");
  // Ming models REJECT sizes/aspect ratios instead of reshaping — never send them.
  const body = imageProviders.imageRequestBody(model, prompt, {
    aspect_ratio: opts.aspect_ratio,
    n: opts.n,
    output_format: opts.output_format,
    input_references: opts.input_references,
    drop: isMing ? ["aspect_ratio", "n"] : [],
  });
  let res;
  try {
    res = await fetch("https://openrouter.ai/api/v1/images", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://frit.local",
        "X-Title": "FRIT",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(180_000),
    });
  } catch (e) {
    throw new Error(`Image gen unreachable (${e.message}) — server network or OpenRouter down`);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.error?.message || `Image gen HTTP ${res.status}`;
    const hint = res.status === 402
      ? " — image models need PAID OpenRouter credits (free 1000/day quota never covers /images). Top up at openrouter.ai/credits."
      : res.status === 404
        ? ` — model '${model}' not served on /images right now; check GET /api/v1/images/models.`
        : res.status === 400
          ? ` — '${model}' rejected the request parameters. The served list is GET /api/v1/images/models.`
          : "";
    throw new Error(msg + hint);
  }
  // Normalize shapes: {data:[{b64_json|url}]} | {data:{...}} | {images:[...]}
  const item = data?.data?.[0] || data?.data || data?.images?.[0] || {};
  const out = {
    ok: true,
    model,
    image_url: item.url || null,
    image_b64: item.b64_json || item.b64Json || item.base64 || null,
    media_type: item.media_type || null,
  };
  if (!out.image_url && !out.image_b64) {
    // The API answered but carried no pixels — usually a moderation block or
    // provider quirk. Never return "success with nothing".
    throw new Error(`Image API returned no image (model '${model}'): ${JSON.stringify(data).slice(0, 300)}`);
  }
  return out;
}
// Design-layer decomposition: flat design -> editable RGBA layers (FREE).
async function decomposeDesign(imageRef, prompt, opts = {}) {
  return generateImage(prompt, {
    model: opts.model || OR_LAYER_MODEL,
    output_format: opts.output_format || "png",
    input_references: [{ type: "image_url", image_url: { url: imageRef } }],
  });
}

// BEFORE-MATH OF PRODUCTION: physical grounding pass for image/video prompts.
// Generators don't know a man is ~1.7m and a car ~4.5m — so a cheap fast-model
// pass first computes real sizes, relative scale, camera (focal/angle/height)
// and depth order, then rewrites the prompt with explicit proportions +
// negative prompt. Output feeds generate_image/generate_video, never the raw brief.
async function groundScene(brief) {
  const out = await chatWithFallback("fast", {
    messages: [
      { role: "system", content: "You are a physical grounding engine. Given a scene brief, output ONLY JSON: {subjects:[{name, real_size_m, position, size_in_frame}], camera:{focal_mm, angle, height}, depth_order:[...], lighting, enriched_prompt, negative_prompt}. enriched_prompt MUST state explicit relative proportions, camera and distance. negative_prompt forbids wrong scale (tiny people, oversized objects, warped perspective)." },
      { role: "user", content: String(brief || "").slice(0, 2000) },
    ],
    temperature: 0.2, max_tokens: 700,
  });
  const text = out.choices?.[0]?.message?.content || "";
  const parsed = safeJsonParse(text.match(/\{[\s\S]*\}/)?.[0] || "", null);
  if (parsed?.enriched_prompt) return parsed;
  return { subjects: [], camera: {}, depth_order: [], lighting: "", enriched_prompt: String(brief || ""), negative_prompt: "tiny people, oversized objects, warped proportions, wrong scale" };
}

// Video generation via OpenRouter Videos API (async: submit -> poll -> download).
// Default: heygen/heygen-video-1 (promo $0.01/s WITH audio). Alt: google/veo-3.1-lite,
// minimax/hailuo, alibaba/wan-2.6 — per-call {model} or console OR_VIDEO_MODEL.
function orVideoHeaders() {
  return {
    Authorization: `Bearer ${OPENROUTER_API_KEY}`,
    "Content-Type": "application/json",
    "HTTP-Referer": "https://frit.local",
    "X-Title": "FRIT",
  };
}
async function submitVideo(prompt, opts = {}) {
  if (!OPENROUTER_API_KEY) throw new Error("OPENROUTER_API_KEY missing");
  const res = await fetch("https://openrouter.ai/api/v1/videos", {
    method: "POST",
    headers: orVideoHeaders(),
    body: JSON.stringify({
      model: opts.model || OR_VIDEO_MODEL,
      prompt: String(prompt || ""),
      duration: opts.duration || 4,
      resolution: opts.resolution || "720p",
      aspect_ratio: opts.aspect_ratio || "16:9",
      // Omit when unset: audio-capable models (HeyGen) default to WITH audio,
      // which is what we want for the primary video model.
      ...(opts.generate_audio !== undefined ? { generate_audio: opts.generate_audio } : {}),
      ...(opts.frame_images ? { frame_images: opts.frame_images } : {}),
      ...(opts.input_references ? { input_references: opts.input_references } : {}),
    }),
    signal: AbortSignal.timeout(60_000),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data?.error?.message || `Video submit HTTP ${res.status}`) + (res.status === 402 ? " — video models need PAID OpenRouter credits (free quota never covers /videos). Top up at openrouter.ai/credits." : ""));
  return data; // { id, polling_url, status }
}
async function pollVideo(jobId) {
  const res = await fetch(`https://openrouter.ai/api/v1/videos/${encodeURIComponent(jobId)}`, {
    headers: orVideoHeaders(),
    signal: AbortSignal.timeout(60_000),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error?.message || `Video poll HTTP ${res.status}`);
  return data;
}
async function downloadVideo(jobId, index = 0) {
  const res = await fetch(`https://openrouter.ai/api/v1/videos/${encodeURIComponent(jobId)}/content?index=${index}`, {
    headers: orVideoHeaders(),
    signal: AbortSignal.timeout(300_000),
  });
  if (!res.ok) throw new Error(`Video content HTTP ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

// Wallet helpers (kobo integers, no float drift). Every movement writes a
// hash-chained ledger row: tx_hash = sha256(prev_hash|...). Verify with
// GET /admin/ledger/verify. Balance mutations run inside a transaction so
// concurrent overage debits can't double-spend.
function assertKobo(n, { min = 1, max = 100_000_000_00 } = {}) {
  if (!Number.isInteger(n) || n < min || n > max) throw new Error(`amount must be an integer ${min}..${max} kobo`);
  return n;
}
function appendLedger(user_id, amount_kobo, kind, note) {
  const prev = db.prepare("SELECT tx_hash FROM wallet_txns ORDER BY id DESC LIMIT 1").get();
  const prev_hash = prev?.tx_hash || "GENESIS";
  const ts = new Date().toISOString();
  const tx_hash = crypto.createHash("sha256").update(`${prev_hash}|${user_id}|${amount_kobo}|${kind}|${note}|${ts}`).digest("hex");
  db.prepare("INSERT INTO wallet_txns (user_id, amount_kobo, kind, note, created_at, prev_hash, tx_hash) VALUES (?, ?, ?, ?, ?, ?, ?)").run(user_id, amount_kobo, kind, note, ts, prev_hash, tx_hash);
}
function walletBalance(user_id) {
  const row = db.prepare("SELECT balance_kobo FROM wallet_accounts WHERE user_id = ?").get(user_id);
  return row?.balance_kobo ?? 0;
}
function walletCredit(user_id, amount_kobo, note = "") {
  assertKobo(amount_kobo);
  const run = db.transaction(() => {
    db.prepare("INSERT INTO wallet_accounts (user_id, balance_kobo) VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET balance_kobo = balance_kobo + ?, updated_at = CURRENT_TIMESTAMP").run(user_id, amount_kobo, amount_kobo);
    appendLedger(user_id, amount_kobo, "credit", note);
  });
  run();
  return walletBalance(user_id);
}
function walletSpend(user_id, amount_kobo, note = "") {
  assertKobo(amount_kobo);
  const run = db.transaction(() => {
    const row = db.prepare("SELECT balance_kobo FROM wallet_accounts WHERE user_id = ?").get(user_id);
    const bal = row?.balance_kobo ?? 0;
    if (bal < amount_kobo) throw new Error(`Insufficient wallet balance (${bal} < ${amount_kobo} kobo)`);
    db.prepare("UPDATE wallet_accounts SET balance_kobo = balance_kobo - ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?").run(amount_kobo, user_id);
    appendLedger(user_id, -amount_kobo, "spend", note);
  });
  run();
  return walletBalance(user_id);
}

// ---- Credit ledger (spending currency for all AI features) ----
// Logic lives in ./billing.js (unit-tested in test/billing.test.mjs). These
// wrappers keep every existing call site unchanged. The factory is built lazily
// on first use because SUB_TIERS / CREDIT_COSTS / etc. are declared further down.
let _billing = null;
function billing() {
  return (_billing ??= createBilling({ db, SUB_TIERS, activeTier, monthStr, CREDIT_COSTS, FEATURE_CHARGE, FEATURE_LABELS }));
}
function creditBalance(user_id) { return billing().creditBalance(user_id); }
function creditTxn(user_id, amount, kind, note = "") { return billing().creditTxn(user_id, amount, kind, note); }
function grantMonthlyCredits(user_id) { return billing().grantMonthlyCredits(user_id); }
function planAllows(tier, feature) { return billing().planAllows(tier, feature); }
function checkEntitlement(user_id, feature) { return billing().checkEntitlement(user_id, feature); }
function debitCredits(user_id, charge, note = "") { return billing().debitCredits(user_id, charge, note); }
function trackProviderJob(provider_id, user_id, kind, provider = "openrouter") { return billing().trackProviderJob(provider_id, user_id, kind, provider); }
function chargeProviderJob(provider_id) { return billing().chargeProviderJob(provider_id); }
// Entitlement + credit rules (grant, gate, atomic post-success debit) -> ./billing.js
// Provider-job billing (charge on completion, exactly once) -> ./billing.js
// provider = which API polls this job to completion: "openrouter" (default),
// "hedra", or "synclabs" (webhook-only). The completion sweeper uses it.
// Migration: provider_jobs.provider tells the completion sweeper which API
// polls each job. Pre-existing rows default to openrouter.
try { db.exec("ALTER TABLE provider_jobs ADD COLUMN provider TEXT DEFAULT 'openrouter'"); } catch { /* column already exists */ }

// Completion sweeper: a render that finishes while the app is closed would
// otherwise never be polled — the provider bills us and the user never pays.
// Every 60s, poll a bounded batch of uncharged provider jobs: charge the
// completed ones (idempotent via the charged flag) and close failed ones so
// they are never billed and never re-polled. charged=1 doubles as "closed".
let _sweeperBusy = false;
setInterval(async () => {
  if (_sweeperBusy) return;
  _sweeperBusy = true;
  try {
    const rows = db.prepare("SELECT provider_id, provider FROM provider_jobs WHERE charged = 0 ORDER BY created_at LIMIT 10").all();
    for (const row of rows) {
      try {
        if (row.provider === "hedra") {
          if (!HEDRA_API_KEY) continue;
          const r = await hedraFetch(`/v3/jobs/${encodeURIComponent(row.provider_id)}/status`);
          const j = await r.json().catch(() => ({}));
          const st = String(j.status || j.state || "").toUpperCase();
          if (["COMPLETED", "COMPLETE", "DONE"].includes(st)) chargeProviderJob(row.provider_id);
          else if (["FAILED", "ERROR", "CANCELLED", "EXPIRED"].includes(st)) db.prepare("UPDATE provider_jobs SET charged = 1 WHERE provider_id = ?").run(row.provider_id); // close: failed renders never bill
        } else if (row.provider === "synclabs") {
          continue; // completion arrives via /webhooks/lipsync — nothing to poll
        } else {
          if (!HAS_OR) continue;
          const p = await pollVideo(row.provider_id);
          const st = String(p?.data?.status || p?.status || "").toLowerCase();
          if (st === "completed") chargeProviderJob(row.provider_id);
          else if (["failed", "cancelled", "expired"].includes(st)) db.prepare("UPDATE provider_jobs SET charged = 1 WHERE provider_id = ?").run(row.provider_id); // close: failed renders never bill
        }
      } catch { /* transient provider error — retry next tick */ }
    }
  } catch { /* db hiccup — keep the interval alive */ }
  finally { _sweeperBusy = false; }
}, 60_000).unref?.();
const SUB_TIERS = {
  // FIXED-NAIRA pricing (VAT-inclusive, bank-transfer friendly round numbers).
  // Priced Oct 2026 at ~₦1,350/$ buffer (official ₦1,332, street ₦1,361-85):
  //   Free    ₦0      — acquisition. Worst-case cost ~$1.00/mo (~₦1,350),
  //                      typical ~$0.30 (free-model fallback absorbs chat).
  //   Starter ₦7,500  (~$5.55) — ChatGPT-Go killer (₦7,000) + images/video/trading.
  //                      Worst-case cost ~$3.12 → margin ~44%.
  //   Creator ₦16,500 (~$12.22) — vs Midjourney $10 + Runway $15. Worst ~$9.02
  //                      (all-8s clips) → margin ~26%; typical ~55%.
  //   Pro     ₦35,000 (~$25.93) — price-matched to Plus ₦33,863 w/ VAT but 5x
  //                      features. Worst ~$21.60 → margin ~17%; typical ~50%.
  // Unit economics (OpenRouter list, Oct 2026 — budget at LIST, keep discount
  // as extra margin): chat turn ~$0.0008 (GLM-Flash 6k in/1.2k out, DeepInfra
  // pin halves it); image $0.018 (Seedream Flash); 8s 720p clip w/ audio ~$0.08
  // (HeyGen Video promo $0.01/s); avatar clip ~$0.40/8s (Avatar IV $0.05/s);
  // STT ~$0.006/min. Clips are the margin killer — quotas below are counted
  // in up-to-8s equivalents. Review naira peg quarterly.
  free: { usd: 0, naira_kobo: 0, days: 3650, credits_mo: 200, features: ["chat", "market_data"] },
  starter: { usd: 5.55, naira_kobo: 750000, days: 30, credits_mo: 800, features: ["chat", "automation", "agent", "market_data"] },
  creator: { usd: 12.22, naira_kobo: 1650000, days: 30, credits_mo: 1500, features: ["chat", "image", "video", "voice", "creative", "market_data"] },
  trader: { usd: 8.8, naira_kobo: 1200000, days: 30, credits_mo: 1200, features: ["chat", "trading", "market_data"] },
  pro: { usd: 25.93, naira_kobo: 3500000, days: 30, credits_mo: 3500, features: ["chat", "image", "video", "voice", "creative", "trading", "automation", "agent", "market_data", "pocket"] },
  // Tester: full Pro-level caps for trial runs. NOT purchasable (subscribe
  // blocks it) — only via admin-minted codes (/billing/mint). Granted duration
  // comes from SUB_TIERS days at REDEEM time (see /billing/redeem).
  tester: { usd: 0, naira_kobo: 0, days: 7, credits_mo: 3500, features: ["chat", "image", "video", "voice", "creative", "trading", "automation", "agent", "market_data", "pocket"] },
  owner: { usd: 0, naira_kobo: 0, days: 365000, credits_mo: 999999, features: ["chat", "image", "video", "voice", "creative", "trading", "automation", "agent", "market_data", "pocket"] },
};
export const TIER_ORDER = ["free", "starter", "creator", "trader", "pro", "tester", "owner"];
export const YEARLY_MULT = 10; // annual = 10x monthly (2 months free)
export const tierYearlyKobo = (tier) => (SUB_TIERS[tier]?.naira_kobo || 0) * YEARLY_MULT;

// Credit economics: 1 credit = ₦10 (at ₦1,370/$ peg). Any top-up converts at this rate.
// PRICED WITH HEADROOM (Oct 2026 verified live costs + buffer for hikes):
// - turn: GLM-Flash list $0.15/$0.50 → 6k in/1.2k out ≈ $0.0015 ≈ ₦2. 1cr = 5x now, 2.5x if GLM doubles.
// - image: Seedream Flash $0.018 ≈ ₦25. 4cr (₦40) = 1.6x now; survives a rise to Pro-class $0.045.
// - clip: HeyGen promo $0.01/s → 8s ≈ $0.08 ≈ ₦110; POST-PROMO $0.03/s → $0.24 ≈ ₦329.
//   65cr (₦650) = 6x promo, 2x post-promo, matches the old ₦650 overage. THE buffer that matters.
// - avatar: Avatar IV $0.05/s → 8s $0.40 ≈ ₦548. 65cr = 1.2x; Hedra fallback ~same.
// - stt_min: buffered at OpenAI $0.006/min ≈ ₦8 (OpenRouter route is $0.0005/min — pure margin). 2cr = 2.4x.
// - agent_task: 5 heavy GLM calls ≈ $0.008 ≈ ₦11. 5cr (₦50) = 4.5x.
// - trade_analysis: 1 engine call ≈ $0.0015 ≈ ₦2. 1cr = 5x.
// - run_code: own sandbox ≈ $0 marginal + abuse brake. 2cr.
// - creative: 1 compose call ≈ $0.002 ≈ ₦3. 2cr.
// - market_data: Twelve/Dukascopy free tiers. 0cr (gated by feature only).
export const CREDIT_NAIRA_KOBO = 1000;
export const CREDIT_COSTS = {
  turn: 1,
  image: 4,
  clip: 65,
  avatar: 65,
  stt_min: 2,
  creative: 2,
  agent_task: 5,
  trade_analysis: 1,
  run_code: 2,
  market_data: 0,
};
// Feature (access lock) -> default charge code (spending). Kept as an explicit
// map so a lock and its price can NEVER silently diverge again.
export const FEATURE_CHARGE = {
  chat: "turn", image: "image", video: "clip", voice: "avatar",
  creative: "creative", trading: "trade_analysis", automation: "turn",
  agent: "agent_task", market_data: "market_data", pocket: "market_data",
};
// Feature flags per plan (for UI display and server-side gating).
export const FEATURE_LABELS = {
  chat: "Chat",
  image: "Image generation",
  video: "Video generation",
  voice: "Voice / STT",
  creative: "Creative suite",
  trading: "Trading engine",
  automation: "Phone automation",
  agent: "Advanced agent",
  market_data: "Market data",
  pocket: "Pocket fixed-expiry signals",
};
// Live USD->NGN (keyless er-api, 6h cache, falls back to 1500).
let _fxCache = { rate: 1500, exp: 0 };
async function ngnPerUsd() {
  if (Date.now() < _fxCache.exp) return _fxCache.rate;
  try {
    const r = await fetch("https://open.er-api.com/v6/latest/USD", { signal: AbortSignal.timeout(15_000) });
    const d = await r.json();
    if (d.rates?.NGN) _fxCache = { rate: Number(d.rates.NGN), exp: Date.now() + 6 * 3600e3 };
  } catch (e) { console.warn("[FX]", e.message); }
  return _fxCache.rate;
}
const PEG_FLOOR = Number(process.env.PEG_FLOOR || 1370);
const tierBaseKobo = (tier) => SUB_TIERS[tier]?.naira_kobo ?? 0; // sticker price at the floor
// RATCHET PEG: floor ₦1,370/$ (Oct 2026: official ₦1,332, street ₦1,361-85).
// While the dollar is at/below the floor, users pay the sticker price above.
// When the dollar climbs past the floor, prices scale proportionally so margin
// survives devaluation — rounded to the nearest ₦100 to stay transfer-typed.
// Override the floor with PEG_FLOOR env (e.g. PEG_FLOOR=1500).
const tierNaira = (tier, rate) => {
  const base = tierBaseKobo(tier);
  if (!base) return 0;
  const eff = Math.max(Number(rate) || PEG_FLOOR, PEG_FLOOR);
  const naira = Math.round((base / 100) * eff / PEG_FLOOR / 100) * 100;
  return naira * 100;
};
// Per-unit wallet overage (EVERY line profitable at list cost — overage users
// are your best margin, never throttled into loss):
//   turn ₦30   vs cost ~₦1-2    (15-30x markup)
//   image ₦50  vs cost ~₦24     (~2x)
//   clip ₦650 (up-to-8s equiv) vs cost ₦270 (4s) – ₦540 (8s w/ audio) (~1.2-2.4x)
//   stt ₦15/min vs cost ~₦8     (~2x)
// Old flat ₦400/clip LOST money on 8s clips — never go back to it.
const OVERAGE = { turn_kobo: 3000, clip_kobo: 65000, image_kobo: 5000, stt_min_kobo: 1500 };
const MASTER_CODE = process.env.MASTER_CODE || "";
const BTC_ADDRESS = process.env.BTC_ADDRESS || ""; // native Bitcoin (bc1/1/3…) — empty until you add one
const USDT_TRON_ADDRESS = process.env.USDT_TRON_ADDRESS || "";
// Binance deposits: BOTH assets live on BNB Smart Chain (BEP20) at ONE address
// (your BTC here is BTCB — pegged BTC, not native BTC, so mempool.space can't
// see it; BSC public RPC can, keyless).
const BSC_ADDRESS = (process.env.BSC_ADDRESS || "").toLowerCase();
const BSC_RPC = process.env.BSC_RPC || "https://bsc-dataseed.binance.org/";
const BSC_TOKENS = {
  usdt: { contract: "0x55d398326f99059ff775485246999027b3197955", decimals: 18, usd: 1 },
  btcb: { contract: "0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c", decimals: 18, usd: null }, // priced live via Binance ticker
};
const TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef";
async function bscRpc(method, params) {
  const r = await fetch(BSC_RPC, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
    signal: AbortSignal.timeout(20_000),
  });
  const d = await r.json();
  if (d.error) throw new Error(`BSC RPC: ${d.error.message}`);
  return d.result;
}
async function btcPriceUSD() {
  // Binance public ticker (keyless). Falls back to mempool price feed.
  try {
    const r = await fetch("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", { signal: AbortSignal.timeout(15_000) });
    const d = await r.json();
    if (Number(d.price) > 0) return Number(d.price);
  } catch {}
  const r = await fetch("https://mempool.space/api/v1/prices", { signal: AbortSignal.timeout(15_000) });
  const d = await r.json();
  return Number(d.USD || 0);
}
// Keyless BEP20 verify: receipt logs must show `token` transferring >= expected
// base units to YOUR BSC address, with >= 12 confirmations. Covers USDT + BTCB.
async function verifyBscTx(txhash, asset, expectedUnits) {
  if (!BSC_ADDRESS) throw new Error("BSC receiving address not configured");
  const token = BSC_TOKENS[asset];
  if (!token) throw new Error(`unknown asset ${asset}`);
  const cleanTx = String(txhash).trim();
  const receipt = await bscRpc("eth_getTransactionReceipt", [cleanTx]);
  if (!receipt || !receipt.blockNumber) throw new Error("tx not found (yet) — wait for broadcast and retry");
  const latest = await bscRpc("eth_blockNumber", []);
  const confs = parseInt(latest, 16) - parseInt(receipt.blockNumber, 16);
  let paid = 0n;
  for (const log of receipt.logs || []) {
    if (String(log.address || "").toLowerCase() !== token.contract) continue;
    if (!log.topics || log.topics[0] !== TRANSFER_TOPIC || log.topics.length < 3) continue;
    const to = ("0x" + log.topics[2].slice(-40)).toLowerCase();
    if (to !== BSC_ADDRESS) continue;
    paid += BigInt(log.data || "0x0");
  }
  const need = BigInt(Math.floor(Number(expectedUnits) * 0.97));
  if (paid < need) throw new Error(`tx pays ${paid} base units to us, expected ~${need}`);
  return { paid_units: paid.toString(), confirmed: confs >= 12, confirmations: confs };
}

// Short typeable passcode, e.g. "7KQ2-9MXD" (Crockford base32, no 0/O/1/I).
function makePasscode() {
  const chars = "23456789ABCDEFGHJKMNPQRSTUVWXYZ";
  const bytes = crypto.randomBytes(8);
  let s = "";
  for (let i = 0; i < 8; i++) s += chars[bytes[i] % chars.length];
  return `${s.slice(0, 4)}-${s.slice(4)}`;
}
function makeActivationCode() {
  return `FRIT-${makePasscode()}-${makePasscode().slice(0, 4)}`;
}
// Subscription time: tier is active only while expires_at is in the future.
// Expired -> falls back to free caps (never hard-locks the user out).
function activeTier(user_id) {
  const sub = db.prepare("SELECT tier, expires_at FROM subscriptions WHERE user_id = ? AND status = 'active'").get(user_id);
  if (sub && sub.expires_at && new Date(sub.expires_at).getTime() > Date.now() && SUB_TIERS[sub.tier]) {
    return sub.tier;
  }
  return "free";
}
function todayStr() { return new Date().toISOString().slice(0, 10); }
function monthStr() { return new Date().toISOString().slice(0, 7); }
function getUsage(user_id, day = todayStr()) {
  let row = db.prepare("SELECT * FROM usage_daily WHERE user_id = ? AND day = ?").get(user_id, day);
  if (!row) {
    db.prepare("INSERT INTO usage_daily (user_id, day) VALUES (?, ?)").run(user_id, day);
    row = { user_id, day, turns: 0, clips: 0, images: 0, stt_mins: 0 };
  }
  return row;
}
function monthUsage(user_id) {
  const rows = db.prepare("SELECT * FROM usage_daily WHERE user_id = ? AND day LIKE ?").all(user_id, `${monthStr()}%`);
  return rows.reduce((a, r) => ({ turns: a.turns + r.turns, clips: a.clips + r.clips, images: a.images + r.images, stt_mins: a.stt_mins + r.stt_mins }), { turns: 0, clips: 0, images: 0, stt_mins: 0 });
}
// NOTE: checkCap (old quota + wallet-overage model) is retired — entitlements
// are checkEntitlement (feature lock + credit preview) with debitCredits
// (atomic post-success debit). usage_daily counters remain for analytics.
// Keyless NATIVE-BTC verify via mempool.space (only for bc1/1/3… addresses —
// NOT for BTCB-on-BSC, which verifyBscTx handles). USDT-Tron has no keyless
// API (trongrid needs a key) -> manual approve path in /billing/pending.
async function verifyBtcTx(txid, expectedSats) {
  const r = await fetch(`https://mempool.space/api/tx/${encodeURIComponent(txid)}`, { signal: AbortSignal.timeout(20_000) });
  if (!r.ok) throw new Error("tx not found (yet) — wait for broadcast and retry");
  const tx = await r.json();
  let paid = 0;
  for (const o of tx.vout || []) {
    if (o.scriptpubkey_address === BTC_ADDRESS) paid += Number(o.value || 0);
  }
  if (paid < expectedSats * 0.97) throw new Error(`tx pays ${paid} sats to us, expected ~${expectedSats}`);
  return { paid_sats: paid, confirmed: !!tx.status?.confirmed };
}

function extractToolCalls(msg) {
  return (msg?.tool_calls || []).map(tc => ({
    id: tc.id,
    type: tc.type,
    function: {
      name: tc.function?.name,
      arguments: safeJsonParse(tc.function?.arguments || "{}", {}),
      raw_arguments: tc.function?.arguments || "{}",
    },
  }));
}

// ========================= SANDBOX ==============================
// Attach the sandbox's own auth token (SANDBOX_AUTH from Render env) so the
// AI can run code without the user configuring anything. If unset we still
// try without auth (local/dev sandboxes may not require one).
function sandboxAuthHeaders() {
  return SANDBOX_AUTH ? { "Content-Type": "application/json", Authorization: `Bearer ${SANDBOX_AUTH}` } : { "Content-Type": "application/json" };
}

async function runSandbox(args = {}) {
  const targets = [];
  if (process.env.LOCAL_SANDBOX_URL) targets.push(process.env.LOCAL_SANDBOX_URL);
  targets.push(SANDBOX_URL);
  let lastErr;
  for (const base of targets) {
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        const res = await fetch(`${base}/sandbox/run`, {
          method: "POST",
          headers: sandboxAuthHeaders(),
          body: JSON.stringify(args),
          signal: AbortSignal.timeout(90_000), // covers a cold start
        });
        const data = await res.json().catch(() => ({}));
        if (res.ok) return data;
        const msg = data?.details || data?.error || `Sandbox HTTP ${res.status}`;
        lastErr = new Error(msg);
        if (/busy/i.test(msg) && attempt < 3) {
          await new Promise(r => setTimeout(r, 2000 * attempt));
          continue;
        }
        break; // auth/validation errors won't fix themselves
      } catch (e) {
        lastErr = e;
        break; // network error: try next target
      }
    }
  }
  throw lastErr || new Error("Sandbox unavailable");
}

// ========================= MARKET DATA ============================
function normalizeInterval(v) {
  const iv = String(v || "1h").toLowerCase().trim();
  if (iv === "30m") return "30min"; // engine shorthand -> Twelve Data interval
  const allowed = ["1min", "5min", "15min", "30min", "1h", "4h", "1day", "1week"];
  return allowed.includes(iv) ? iv : "1h";
}

function resolveOutputSize(interval) {
  return ({
    "1min": 500, "5min": 288, "15min": 200, "30min": 150,
    "1h": 168, "4h": 120, "1day": 100, "1week": 52
  })[interval] || 168;
}

function toBinanceInterval(interval) {
  return ({
    "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1day": "1d", "1week": "1w"
  })[interval] || "1h";
}

const TD_SYMBOLS = {
  EURUSD: "EUR/USD", GBPUSD: "GBP/USD", USDJPY: "USD/JPY", AUDUSD: "AUD/USD",
  USDCHF: "USD/CHF", USDCAD: "USD/CAD", NZDUSD: "NZD/USD", XAUUSD: "XAU/USD",
  XAGUSD: "XAG/USD", GBPJPY: "GBP/JPY", EURJPY: "EUR/JPY", EURGBP: "EUR/GBP",
  SPY: "SPY", QQQ: "QQQ", IWM: "IWM", XSP: "XSP",
  BTC: "BTC/USD", ETH: "ETH/USD", SOL: "SOL/USD", BNB: "BNB/USD",
  XRP: "XRP/USD", DOGE: "DOGE/USD", ADA: "ADA/USD",
  BTCUSD: "BTC/USD", ETHUSD: "ETH/USD", SOLUSD: "SOL/USD", BNBUSD: "BNB/USD",
  XRPUSD: "XRP/USD", DOGEUSD: "DOGE/USD", ADAUSD: "ADA/USD",
};

const CRYPTO_SET = new Set([
  "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA",
  "BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD", "DOGEUSD", "ADAUSD"
]);

// Binance spot pairs are USDT-quoted (BTCUSDT), NOT raw fiat pairs like
// "BTCUSD" — api.binance.com has no such symbol, so requests using the old
// mapping below were silently failing on Binance and falling through to
// TwelveData for every crypto candle (defeating the point of using Binance
// as a free fallback).
const BINANCE_SYM = {
  BTC: "BTCUSDT", ETH: "ETHUSDT", SOL: "SOLUSDT", BNB: "BNBUSDT",
  XRP: "XRPUSDT", DOGE: "DOGEUSDT", ADA: "ADAUSDT",
  BTCUSD: "BTCUSDT", ETHUSD: "ETHUSDT", SOLUSD: "SOLUSDT", BNBUSD: "BNBUSDT",
  XRPUSD: "XRPUSDT", DOGEUSD: "DOGEUSDT", ADAUSD: "ADAUSDT",
};

const COINGECKO_IDS = {
  BTC: "bitcoin", ETH: "ethereum", SOL: "solana", BNB: "binancecoin",
  XRP: "ripple", DOGE: "dogecoin", ADA: "cardano",
  BTCUSD: "bitcoin", ETHUSD: "ethereum", SOLUSD: "solana", BNBUSD: "binancecoin",
  XRPUSD: "ripple", DOGEUSD: "dogecoin", ADAUSD: "cardano",
};

const FRANKFURTER_MAP = {
  EURUSD: { base: "EUR", quote: "USD" }, GBPUSD: { base: "GBP", quote: "USD" },
  USDJPY: { base: "USD", quote: "JPY" }, AUDUSD: { base: "AUD", quote: "USD" },
  USDCHF: { base: "USD", quote: "CHF" }, USDCAD: { base: "USD", quote: "CAD" },
  NZDUSD: { base: "NZD", quote: "USD" }, EURGBP: { base: "EUR", quote: "GBP" },
  EURJPY: { base: "EUR", quote: "JPY" }, GBPJPY: { base: "GBP", quote: "JPY" },
};

function normalizeCandles(data, source = "twelve_data") {
  if (!Array.isArray(data)) return [];
  return data.map(c => ({
    time: source === "twelve_data" ? new Date(c.datetime).getTime() : c[0],
    open: parseFloat(source === "twelve_data" ? c.open : c[1]),
    high: parseFloat(source === "twelve_data" ? c.high : c[2]),
    low: parseFloat(source === "twelve_data" ? c.low : c[3]),
    close: parseFloat(source === "twelve_data" ? c.close : c[4]),
    volume: parseFloat(source === "twelve_data" ? c.volume || 0 : c[5] || 0),
  })).filter(c => !isNaN(c.close) && !isNaN(c.open) && c.high >= c.low);
}

async function fetchCandles(symbol, interval = "1h", outputsize = null) {
  const sym = String(symbol || "").toUpperCase();
  const iv = normalizeInterval(interval);
  const size = outputsize || resolveOutputSize(iv);
  const ck = `candles:${sym}:${iv}:${size}`;
  const cached = cacheGet(ck);
  if (cached) return cached;

  // Deriv synthetics: free keyless candle history (no budget, no key).
  // Anonymous client (no token): tick() needs no auth.
  if (isDerivSymbol(sym)) {
    try {
      const dc = new DerivClient({ token: "" });
      // NOTE: iv arrives NORMALIZED (normalizeInterval: "1m"->"1min").
      const tf = ({ "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "1h", "4h": "4h", "1day": "1d", "1week": "1w" })[iv] || "30m";
      const candles = await dc.candles(sym, tf, Math.min(size, 500));
      try { dc.close(); } catch { /* ignore */ }
      if (candles.length >= 10) {
        cacheSet(ck, candles, 60000);
        return candles;
      }
    } catch (e) { console.error("[Deriv candles]", sym, e.message); }
  }

  // US stocks/ETFs: Alpaca bars when keyed, else Finnhub stock/candle.
  if (marketOf(sym) === "alpaca") {
    try {
      const ac = new AlpacaClient({ keyId: ALPACA_PAPER_KEY, secret: ALPACA_PAPER_SECRET, paper: true, fetchImpl: fetch });
      if (ac.configured) {
        const tf = ({ "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "1H", "4h": "4H", "1day": "1D", "1week": "1W" })[iv] || "30m";
        const r = await ac.bars(sym, tf, Math.min(size, 300));
        if (!r.error && r.candles?.length >= 10) {
          cacheSet(ck, r.candles, 60000);
          return r.candles;
        }
      }
    } catch (e) { console.error("[Alpaca candles]", sym, e.message); }
    if (FINNHUB_KEY) {
      try {
        const res = ({ "1min": "1", "5min": "5", "15min": "15", "30min": "30", "1h": "60", "4h": "240", "1day": "D", "1week": "W" })[iv] || "30";
        const to = Math.floor(Date.now() / 1000);
        const from = to - 86400 * 400;
        const url = `https://finnhub.io/api/v1/stock/candle?symbol=${encodeURIComponent(sym)}&resolution=${res}&from=${from}&to=${to}&token=${FINNHUB_KEY}`;
        const r2 = await boundedFetch(url);
        const d = await r2.json();
        if (d.s === "ok" && Array.isArray(d.c) && d.c.length >= 10) {
          const candles = d.c.map((c, i) => ({
            time: d.t[i] * 1000, open: d.o[i], high: d.h[i], low: d.l[i],
            close: c, volume: (d.v && d.v[i]) || 0,
          }));
          cacheSet(ck, candles, 60000);
          return candles;
        }
      } catch (e) { console.error("[Finnhub stock candles]", sym, e.message); }
    }
  }

  // Crypto goes to Binance FIRST (free, no key, ~1200 req/min) so it never
  // touches the TwelveData quota, which forex/XAU/XAG rely on exclusively.
  if (CRYPTO_SET.has(sym) && BINANCE_SYM[sym]) {    try {
      const url = `https://api.binance.com/api/v3/klines?symbol=${BINANCE_SYM[sym]}&interval=${toBinanceInterval(iv)}&limit=${Math.min(size, 1000)}`;
      const res = await boundedFetch(url);
      const arr = await res.json();
      if (Array.isArray(arr) && arr.length >= 10) {
        const candles = normalizeCandles(arr, "binance");
        cacheSet(ck, candles, 60000);
        return candles;
      }
    } catch (e) { console.error("[Binance candles]", sym, e.message); }
  }

  if (TWELVE_DATA_KEY && TD_SYMBOLS[sym]) {
    try {
      const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&interval=${iv}&outputsize=${size}&apikey=${TWELVE_DATA_KEY}`;
      const res = await boundedFetch(url);
      const data = await res.json();
      if (data.status !== "error" && data.values?.length >= 10) {
        const candles = normalizeCandles(data.values.reverse(), "twelve_data");
        cacheSet(ck, candles, 60000);
        return candles;
      }
    } catch (e) { console.error("[TwelveData candles]", sym, e.message); }
  }

  // Fallback for crypto if Binance somehow failed and TwelveData also has no key/entry
  if (BINANCE_SYM[sym]) {
    try {
      const url = `https://api.binance.com/api/v3/klines?symbol=${BINANCE_SYM[sym]}&interval=${toBinanceInterval(iv)}&limit=${Math.min(size, 1000)}`;
      const res = await boundedFetch(url);
      const arr = await res.json();
      if (Array.isArray(arr) && arr.length >= 10) {
        const candles = normalizeCandles(arr, "binance");
        cacheSet(ck, candles, 60000);
        return candles;
      }
    } catch (e) { console.error("[Binance candles]", sym, e.message); }
  }
  return null;
}

async function fetchSpotPrice(symbol) {
  const sym = String(symbol || "").toUpperCase();
  const ck = `spot:${sym}`;
  const cached = cacheGet(ck);
  if (cached) return cached;
  // Live-feed first: streaming ticks are fresher than any REST poll and cost
  // zero credits. (liveFeed is assigned at boot; null until then.)
  try {
    const live = globalThis.__liveFeed?.lastPrice(sym);
    if (live && Number.isFinite(live.price) && Date.now() - live.ts < 5 * 60 * 1000) {
      return { price: live.price, source: live.source || "live-feed" };
    }
  } catch { /* fall through to REST */ }
  let result = null;

  // Deriv synthetics tick (keyless) — also feeds the position monitor.
  // Anonymous client (no token): tick() needs no auth.
  if (!result && isDerivSymbol(sym)) {
    try {
      const dc = new DerivClient({ token: "" });
      const px = await dc.tick(sym);
      try { dc.close(); } catch { /* ignore */ }
      if (px > 0) result = { price: px, source: "deriv" };
    } catch (e) { console.error("[Deriv spot]", sym, e.message); }
  }

  // US stocks: Alpaca quote when keyed, else Finnhub quote.
  if (!result && marketOf(sym) === "alpaca") {
    try {
      const ac = new AlpacaClient({ keyId: ALPACA_PAPER_KEY, secret: ALPACA_PAPER_SECRET, paper: true, fetchImpl: fetch });
      if (ac.configured) {
        const q = await ac.quote(sym);
        if (!q.error) {
          const mid = q.bid > 0 && q.ask > 0 ? (q.bid + q.ask) / 2 : 0;
          if (mid > 0) result = { price: mid, source: "alpaca-delayed" };
        }
      }
    } catch (e) { console.error("[Alpaca spot]", sym, e.message); }
    if (!result && FINNHUB_KEY) {
      try {
        const res = await boundedFetch(`https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(sym)}&token=${FINNHUB_KEY}`);
        const data = await res.json();
        if (data.c > 0) result = { price: data.c, source: "finnhub" };
      } catch (e) { console.error("[Finnhub spot]", sym, e.message); }
    }
  }

  // Crypto: CoinGecko first (free, no key) so crypto spot checks never touch
  // the TwelveData quota either — same reasoning as fetchCandles above.
  if (CRYPTO_SET.has(sym) && COINGECKO_IDS[sym]) {
    try {
      const res = await boundedFetch(`https://api.coingecko.com/api/v3/simple/price?ids=${COINGECKO_IDS[sym]}&vs_currencies=usd&include_24hr_change=true`);
      const data = await res.json();
      const id = COINGECKO_IDS[sym];
      if (data[id]) result = { price: data[id].usd, change24h: data[id].usd_24h_change, source: "coingecko" };
    } catch (e) { console.error("[CoinGecko spot]", sym, e.message); }
  }

  if (!result && TWELVE_DATA_KEY && TD_SYMBOLS[sym]) {
    try {
      const res = await boundedFetch(`https://api.twelvedata.com/price?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&apikey=${TWELVE_DATA_KEY}`);
      const data = await res.json();
      if (data.price) result = { price: parseFloat(data.price), source: "twelvedata" };
    } catch (e) { console.error("[TwelveData spot]", sym, e.message); }
  }

  if (!result && FRANKFURTER_MAP[sym]) {
    try {
      const { base, quote } = FRANKFURTER_MAP[sym];
      const res = await boundedFetch(`https://api.frankfurter.app/latest?from=${base}&to=${quote}`);
      const data = await res.json();
      const rate = data.rates?.[quote];
      if (rate) result = { price: parseFloat(rate), source: "frankfurter" };
    } catch (e) { console.error("[Frankfurter]", sym, e.message); }
  }

  if (!result && sym === "USDNGN") {
    try {
      const res = await boundedFetch("https://open.er-api.com/v6/latest/USD");
      const data = await res.json();
      if (data.rates?.NGN) result = { price: data.rates.NGN, source: "er-api" };
    } catch (e) { console.error("[ER-API USDNGN]", e.message); }
  }

  if (result) cacheSet(ck, result, 30000);
  return result;
}

async function fetch24hDelta(symbol) {
  const sym = String(symbol || "").toUpperCase();
  if (CRYPTO_SET.has(sym)) return null;
  if (!TWELVE_DATA_KEY || !TD_SYMBOLS[sym]) return null;
  const ck = `delta:${sym}`;
  const cached = cacheGet(ck);
  if (cached != null) return cached;
  try {
    const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&interval=1day&outputsize=2&apikey=${TWELVE_DATA_KEY}`;
    const res = await boundedFetch(url);
    const data = await res.json();
    if (data.values?.length >= 2) {
      const today = parseFloat(data.values[0].close);
      const yest = parseFloat(data.values[1].close);
      const delta = ((today - yest) / yest) * 100;
      cacheSet(ck, delta, 300000);
      return delta;
    }
  } catch (e) { console.error("[fetch24hDelta]", sym, e.message); }
  return null;
}

async function fetchMarketPrices(symbols = []) {
  const result = {};
  await Promise.all(symbols.map(async sym => {
    const s = String(sym || "").toUpperCase();
    const spot = await fetchSpotPrice(s);
    if (!spot) {
      result[s] = { symbol: s, price: 0, change24h: 0, currency: "USD", error: "Not found" };
      return;
    }
    const change24h = spot.change24h != null && spot.change24h !== 0
      ? spot.change24h
      : (await fetch24hDelta(s)) ?? 0;
    result[s] = {
      symbol: s,
      price: spot.price,
      change24h,
      currency: s.includes("NGN") ? "NGN" : "USD",
      source: spot.source,
    };
  }));
  return result;
}

// =================== NEWS FILTER ===================
const SYMBOL_CURRENCIES = {
  EURUSD: ["EUR", "USD"], GBPUSD: ["GBP", "USD"], USDJPY: ["USD", "JPY"],
  AUDUSD: ["AUD", "USD"], USDCHF: ["USD", "CHF"], USDCAD: ["USD", "CAD"],
  NZDUSD: ["NZD", "USD"], GBPJPY: ["GBP", "JPY"], EURJPY: ["EUR", "JPY"],
  EURGBP: ["EUR", "GBP"], XAUUSD: ["USD"], XAGUSD: ["USD"],
  BTCUSD: ["USD"], ETHUSD: ["USD"], SOLUSD: ["USD"], BNBUSD: ["USD"],
  XRPUSD: ["USD"], DOGEUSD: ["USD"], ADAUSD: ["USD"],
};

async function fetchHighImpactEvents() {
  const ck = "news_events";
  const cached = cacheGet(ck);
  if (cached) return cached;
  try {
    const res = await boundedFetch("https://nfs.faireconomy.media/ff_calendar_thisweek.json");
    if (!res.ok) return [];
    const data = await res.json();
    const high = data.filter(e => e.impact === "High").map(e => ({
      currency: e.currency,
      title: e.title,
      time: new Date(e.date).getTime(),
    }));
    cacheSet(ck, high, 60 * 60 * 1000);
    return high;
  } catch (err) {
    console.warn("[NewsFilter]", err.message);
    return [];
  }
}

async function checkNewsFilter(symbol) {
  const sym = String(symbol || "").toUpperCase();
  const currencies = SYMBOL_CURRENCIES[sym] || ["USD"];
  const events = await fetchHighImpactEvents();
  const now = Date.now();
  const window = 30 * 60 * 1000;
  const nearby = events.filter(e => currencies.includes(e.currency) && Math.abs(e.time - now) <= window);
  if (nearby.length > 0) {
    const titles = nearby.map(e => `${e.currency} ${e.title}`).join(", ");
    return {
      blocked: false, // Do not hard-kill the analysis; provide strategy advisory
      has_news: true,
      event_active: true,
      reason: `High-impact news within 30 min: ${titles}`,
      recommendation: "High volatility event! Standard trend entries have spread spike risk. Recommendation: (1) Wait 15m post-release for candle stabilization, or (2) Trade the pre-news range breakout with half risk.",
      events: nearby,
    };
  }
  return { blocked: false, has_news: false };
}


// =================== WEB SEARCH + WEATHER ===================
async function webSearch(query) {
  if (!String(query || "").trim()) return "No search terms provided.";
  try {
    const out = await deepSearch(query, "text");
    return out.ok ? out.data : out.error;
  } catch (e) {
    return "Search failed: " + e.message;
  }
}

async function getWeather(city = "Lagos") {
  try {
    const geo = await (await boundedFetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`)).json();
    const loc = geo.results?.[0];
    if (!loc) return "City not found";
    const wx = await (await boundedFetch(`https://api.open-meteo.com/v1/forecast?latitude=${loc.latitude}&longitude=${loc.longitude}&current_weather=true&timezone=auto`)).json();
    const cw = wx.current_weather;
    return `${loc.name}: ${cw.temperature}°C, wind ${cw.windspeed} km/h, ${cw.weathercode <= 1 ? "Clear" : cw.weathercode <= 3 ? "Cloudy" : "Rainy"}`;
  } catch (e) {
    // F-08: the real reason the weather lookup failed (rate limit, DNS, bad
    // city) used to vanish behind a generic string, so a broken upstream was
    // indistinguishable from a missing city.
    console.warn(`[weather] lookup failed for "${city}": ${e.message}`);
    return "Weather unavailable";
  }
}

// =================== LOT SIZE ENGINE ===================
const PIP_CONFIG = {
  EURUSD: { pipSize: 0.0001, pipValue: 10 },
  GBPUSD: { pipSize: 0.0001, pipValue: 10 },
  AUDUSD: { pipSize: 0.0001, pipValue: 10 },
  NZDUSD: { pipSize: 0.0001, pipValue: 10 },
  USDCHF: { pipSize: 0.0001, pipValue: 10 },
  USDCAD: { pipSize: 0.0001, pipValue: 10 },
  EURGBP: { pipSize: 0.0001, pipValue: 10 },
  USDJPY: { pipSize: 0.01, pipValue: 9.3 },
  GBPJPY: { pipSize: 0.01, pipValue: 9.3 },
  EURJPY: { pipSize: 0.01, pipValue: 9.3 },
  XAUUSD: { pipSize: 0.01, pipValue: 100 },
  XAGUSD: { pipSize: 0.001, pipValue: 50 },
  BTCUSD: { pipSize: 1, pipValue: 1, isCrypto: true },
  ETHUSD: { pipSize: 0.1, pipValue: 1, isCrypto: true },
  SOLUSD: { pipSize: 0.01, pipValue: 1, isCrypto: true },
  BNBUSD: { pipSize: 0.01, pipValue: 1, isCrypto: true },
  XRPUSD: { pipSize: 0.0001, pipValue: 1, isCrypto: true },
  DOGEUSD: { pipSize: 0.00001, pipValue: 1, isCrypto: true },
  ADAUSD: { pipSize: 0.00001, pipValue: 1, isCrypto: true },
  // Stock indices: $1 per point per 1.0 lot is the common retail MT5 contract
  // (size 1). Non-USD indices (GER40/EUR, UK100X/GBP, JPN225/JPY) are quoted
  // here in USD terms — a documented approximation; fix YOUR broker's exact
  // values with PIP_OVERRIDES_JSON (same shape, e.g. {"US30":{"pipValue":2}}).
  US30: { pipSize: 1, pipValue: 1 },
  NAS100: { pipSize: 1, pipValue: 1 },
  SPX500: { pipSize: 1, pipValue: 1 },
  GER40: { pipSize: 1, pipValue: 1 },
  UK100X: { pipSize: 1, pipValue: 1 },
  JPN225: { pipSize: 1, pipValue: 1 },
  AUS200: { pipSize: 1, pipValue: 1 },
  FRA40: { pipSize: 1, pipValue: 1 },
  EU50: { pipSize: 1, pipValue: 1 },
  US2000: { pipSize: 1, pipValue: 1 },
  USDX: { pipSize: 0.01, pipValue: 10 },
  // Energy: assumes 1000-bbl USOIL/UKOIL lots ($1000 per $1 move) and 10k-mmBtu
  // NATGAS ($10 per 0.001). Many retail brokers sell 100-bbl minis instead —
  // verify YOUR contract size before live size, override with PIP_OVERRIDES_JSON.
  USOIL: { pipSize: 0.01, pipValue: 10 },
  UKOIL: { pipSize: 0.01, pipValue: 10 },
  NATGAS: { pipSize: 0.001, pipValue: 10 },
};

// Broker-specific corrections without editing code:
//   PIP_OVERRIDES_JSON='{"US30":{"pipSize":1,"pipValue":2},"GER40":{"pipValue":1.1}}'
try {
  const raw = process.env.PIP_OVERRIDES_JSON;
  if (raw) {
    const parsed = JSON.parse(raw);
    for (const [k, v] of Object.entries(parsed)) {
      const sym = String(k).toUpperCase();
      if (v && typeof v === "object" && PIP_CONFIG[sym]) {
        PIP_CONFIG[sym] = { ...PIP_CONFIG[sym], ...v };
        console.log(`[PIP] override ${sym}:`, JSON.stringify(PIP_CONFIG[sym]));
      }
    }
  }
} catch (e) {
  console.warn("[PIP] PIP_OVERRIDES_JSON parse failed:", e.message);
}

function calculateLotSize({ symbol, balance, riskPercent = 1, entry, stopLoss }) {
  const sym = String(symbol || "").toUpperCase();
  const cfg = PIP_CONFIG[sym];
  const MAX_RISK = 2;
  const MAX_LOT = 5;
  const MIN_LOT = 0.01;
  const safeRisk = Math.min(riskPercent, MAX_RISK);
  const riskAmt = (balance || 1000) * (safeRisk / 100);
  if (!entry || !stopLoss || entry === stopLoss) return MIN_LOT;
  if (!cfg) { console.warn(`[LotSize] Unknown symbol ${sym}`); return MIN_LOT; }
  let lotSize;
  if (cfg.isCrypto) {
    const priceDist = Math.abs(entry - stopLoss);
    lotSize = priceDist === 0 ? MIN_LOT : riskAmt / priceDist;
  } else {
    const stopPips = Math.abs(entry - stopLoss) / cfg.pipSize;
    lotSize = stopPips === 0 ? MIN_LOT : riskAmt / (stopPips * cfg.pipValue);
  }
  return Math.max(MIN_LOT, Math.min(MAX_LOT, Math.round(lotSize * 100) / 100));
}

// ==================== MT5 BRIDGE ====================
async function sendToMT5Bridge({ symbol, action, lotSize, entry, sl, tp, reason = "" }) {
  if (!MT5_BRIDGE_URL) {
    console.log(`[PAPER TRADE] ${action.toUpperCase()} ${symbol} | Lot:${lotSize} | Entry:${entry} | SL:${sl} | TP:${tp}`);
    return { mode: "paper", symbol, action, lotSize, entry, sl, tp, status: "simulated", message: "MT5_BRIDGE_URL not set — paper mode" };
  }
  try {
    const res = await fetch(`${MT5_BRIDGE_URL}/place_trade`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${AUTH_TOKEN}` },
      body: JSON.stringify({ symbol, action, lotSize, entry, sl, tp, reason }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || "MT5 bridge error");
    return { mode: "live", ...data };
  } catch (err) {
    console.error("[MT5 Bridge]", err.message);
    return { mode: "live", status: "failed", error: err.message };
  }
}



// ==================== TRADE MEMORY ====================
// SQLite-backed (survives restarts) with an in-memory Map as the read cache.
const tradeMemory = new Map();

// Load existing rows from the trade_memory table into the map at boot.
(function loadTradeMemory() {
  try {
    const rows = db.prepare("SELECT * FROM trade_memory ORDER BY rowid ASC").all();
    for (const r of rows) {
      const sym = String(r.symbol || "").toUpperCase();
      if (!tradeMemory.has(sym)) tradeMemory.set(sym, []);
      const ts = new Date(String(r.timestamp).replace(" ", "T")).getTime() || Date.now();
      tradeMemory.get(sym).push({
        direction: r.direction || null,
        pattern: r.pattern || null,
        outcome: r.outcome || null,
        note: r.note || null,
        timestamp: ts,
      });
    }
    if (rows.length) console.log(`[TradeMemory] loaded ${rows.length} row(s) from SQLite`);
  } catch (e) {
    console.warn("[TradeMemory] load failed:", e.message);
  }
})();

function addTradeMemory(symbol, entry) {
  const sym = String(symbol || "").toUpperCase();
  if (!tradeMemory.has(sym)) tradeMemory.set(sym, []);
  const entries = tradeMemory.get(sym);
  entries.push({ ...entry, timestamp: Date.now() });
  if (entries.length > 20) entries.splice(0, entries.length - 20);
  try {
    db.prepare("INSERT INTO trade_memory (symbol, direction, pattern, outcome, note) VALUES (?, ?, ?, ?, ?)").run(
      sym, entry.direction || null, entry.pattern || null, entry.outcome || null, entry.note || null
    );
  } catch (e) {
    console.warn("[TradeMemory] persist failed:", e.message);
  }
}

function getTradeMemory(symbol, limit = 5) {
  const sym = String(symbol || "").toUpperCase();
  return (tradeMemory.get(sym) || []).slice(-limit);
}

function formatTradeMemoryForPrompt(symbol) {
  const entries = getTradeMemory(symbol, 5);
  if (!entries.length) return "";
  const lines = entries.map(e => {
    const d = new Date(e.timestamp).toISOString().slice(0, 10);
    return `[${d}] ${e.outcome || "?"} | Pattern:${e.pattern || "none"} | Dir:${e.direction || "?"} | Note:${e.note || ""}`;
  });
  return `\nPast trade memory for ${symbol}:\n${lines.join("\n")}`;
}

// ==================== TRADING SUITE HELPERS ====================

function formatTradingSignal(symbol, result) {
  const decision = result.decision || "WAIT";
  const sc = result.scenarios || {};
  const s1 = sc.scenario_1_immediate || null;
  const s2 = sc.scenario_2_pullback_crossover || null;
  const sR = sc.scenario_range_fade || null;
  const isActionable = ["BUY", "SELL"].includes(decision);
  // WAIT_PULLBACK -> PULLBACK zone, WAIT_RANGE -> RANGE fade, else IMMEDIATE/WAIT/REVERSAL
  let scenario = "WAIT";
  if (isActionable) {
    scenario = (result.entry_ctx?.type === "pullback_limit") ? "PULLBACK"
      : (result.entry_ctx?.type === "range_limit") ? "RANGE" : "IMMEDIATE";
    // Reversal single-scenario entries carry "Reversal" in the name
    if (/reversal/i.test(s1?.name || "")) scenario = "REVERSAL";
  } else if (decision === "WAIT_PULLBACK") scenario = "PULLBACK";
  else if (decision === "WAIT_RANGE") scenario = "RANGE";

  const primary = s1 || s2 || sR;
  const entry = Number(result.entry ?? primary?.entry ?? s2?.entry ?? sR?.entry ?? NaN);
  const sl = Number(result.sl ?? primary?.sl ?? s2?.estimated_sl ?? sR?.estimated_sl ?? sR?.sl ?? NaN);
  const tp1 = Number(result.tp ?? primary?.tp1 ?? s2?.estimated_tp1 ?? sR?.estimated_tp1 ?? NaN);
  const tp2 = Number(result.tp2 ?? primary?.tp2 ?? s2?.estimated_tp2 ?? sR?.estimated_tp2 ?? NaN);

  return {
    id: `sig_${symbol}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    symbol,
    decision,
    confidence: result.confidence || 0,
    entry: Number.isFinite(entry) ? entry : null,
    sl: Number.isFinite(sl) ? sl : null,
    tp1: Number.isFinite(tp1) ? tp1 : null,
    tp2: Number.isFinite(tp2) ? tp2 : null,
    rr: result.rr ?? primary?.rr ?? null,
    scenario,
    reasoning: result.reason || result.reasons?.join(" | ") || "Engine analysis",
    pullbackHealth: s2?.pullback_health || result.structure_30m ? {
      status: result.structure_30m?.pullback_status || null,
      is_healthy: result.structure_30m?.pullback_healthy ?? null,
      pullback_probability: result.structure_30m?.pullback_probability ?? null,
      reversal_probability: result.structure_30m?.reversal_probability ?? null,
      ...(s2?.pullback_health || {}),
    } : (s2?.pullback_health || null),
    rangeInfo: result.structure_30m?.range_30m || sR ? {
      upper: result.structure_30m?.range_30m?.upper ?? sR?.watch_zone ?? null,
      lower: result.structure_30m?.range_30m?.lower ?? null,
      tradable: result.structure_30m?.range_30m?.tradable ?? !!sR,
      ...(sR || {}),
    } : null,
    costs: result.costs || primary?.costs || null,
    tpBasis: result.tp_basis || primary?.tp_basis || null,
    regime: result.regime?.macro_trend || result.direction || null,
    entryCtx: result.entry_ctx || null,
    timestamp: Date.now(),
    isFavorite: false,
    isRead: false
  };
}

function ensureSignalColumns() {
  try {
    const cols = db.prepare("PRAGMA table_info(trading_signals)").all().map(c => c.name);
    if (!cols.includes("tp_basis")) db.exec("ALTER TABLE trading_signals ADD COLUMN tp_basis TEXT");
    if (!cols.includes("regime")) db.exec("ALTER TABLE trading_signals ADD COLUMN regime TEXT");
    if (!cols.includes("entry_ctx")) db.exec("ALTER TABLE trading_signals ADD COLUMN entry_ctx TEXT");
    // Which engine/bot produced this signal — the notification bar tags by it.
    if (!cols.includes("bot_source")) db.exec("ALTER TABLE trading_signals ADD COLUMN bot_source TEXT");
    // Lot size recommendation (Kelly-based) attached to every signal.
    if (!cols.includes("lot_size")) db.exec("ALTER TABLE trading_signals ADD COLUMN lot_size REAL");
    if (!cols.includes("lot_reason")) db.exec("ALTER TABLE trading_signals ADD COLUMN lot_reason TEXT");
    if (!cols.includes("risk_usd")) db.exec("ALTER TABLE trading_signals ADD COLUMN risk_usd REAL");
  } catch (e) { console.warn("[signals] migration failed:", e.message); }
}
ensureSignalColumns();

function storeSignalHistory(user_id, signal) {
  try {
    ensureSignalColumns();
    db.prepare(`
      INSERT OR REPLACE INTO trading_signals
      (id, user_id, symbol, decision, confidence, entry, sl, tp1, tp2, rr, scenario, reasoning, pullback_health, range_info, costs, tp_basis, regime, entry_ctx, bot_source, lot_size, lot_reason, risk_usd, timestamp, is_favorite, is_read)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      signal.id, user_id, signal.symbol, signal.decision, signal.confidence,
      signal.entry, signal.sl, signal.tp1, signal.tp2, signal.rr,
      signal.scenario, signal.reasoning,
      signal.pullbackHealth ? JSON.stringify(signal.pullbackHealth) : null,
      signal.rangeInfo ? JSON.stringify(signal.rangeInfo) : null,
      signal.costs ? JSON.stringify(signal.costs) : null,
      signal.tpBasis || null, signal.regime || null,
      signal.entryCtx ? JSON.stringify(signal.entryCtx) : null,
      signal.botSource || signal.bot_source || null,
      signal.lotSize ?? null, signal.lotReason || null, signal.riskUsd ?? null,
      signal.timestamp, 0, 0
    );
  } catch (e) {
    console.warn("[storeSignalHistory] failed:", e.message);
  }
}

function hydrateSignal(row) {
  if (!row) return row;
  const j = (v) => { try { return typeof v === "string" ? JSON.parse(v) : v; } catch { return v; } };
  return {
    ...row,
    pullbackHealth: row.pullback_health ? j(row.pullback_health) : null,
    rangeInfo: row.range_info ? j(row.range_info) : null,
    costs: row.costs ? j(row.costs) : null,
    entryCtx: row.entry_ctx ? j(row.entry_ctx) : null,
    tpBasis: row.tp_basis || null,
    botSource: row.bot_source || "engine",
    lotSize: row.lot_size != null ? Number(row.lot_size) : null,
    lotReason: row.lot_reason || null,
    riskUsd: row.risk_usd != null ? Number(row.risk_usd) : null,
    isFavorite: row.is_favorite === 1,
    isRead: row.is_read === 1,
  };
}

// ==================== OPTIONS SUITE HELPERS (paper signals only) =============
// Nothing here places orders. Options execution has no rail in this build
// (MT5 cannot trade equity options); these endpoints emit signals for the
// phone's floating banner and record them for review.
function storeOptionsSignal(user_id, signal) {
  try {
    ensureOptionsTables(db);
    db.prepare(`
      INSERT OR REPLACE INTO options_signals
      (id, user_id, symbol, kind, decision, legs_json, credit, max_loss, rr,
       breakevens_json, iv_rank, atm_iv, rv20, dte, expiration, exits_json,
       reasoning, source, provisional, synthetic, timestamp, is_read)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
    `).run(
      signal.id, user_id, signal.symbol, signal.kind || null, signal.decision || null,
      signal.legs ? JSON.stringify(signal.legs) : (signal.band1d ? JSON.stringify({ band1d: signal.band1d, band7d: signal.band7d }) : null),
      signal.credit ?? null, signal.maxLossDollars ?? signal.maxLoss ?? null, signal.rr ?? null,
      signal.breakevens ? JSON.stringify(signal.breakevens) : null,
      signal.ivRank ?? null, signal.atmIv ?? null, signal.rv20 ?? null,
      signal.dte ?? null, signal.expiration || null,
      signal.exits ? JSON.stringify(signal.exits) : (signal.strangle7d ? JSON.stringify(signal.strangle7d) : null),
      (signal.reasoning || []).join(" | "),
      signal.dataSource?.source || (signal.synthetic ? "model" : null),
      signal.dataSource?.provisional ? 1 : 0, signal.synthetic ? 1 : 0,
      signal.timestamp
    );
  } catch (e) {
    console.warn("[storeOptionsSignal] failed:", e.message);
  }
}

/** Pre-trade options guard: daily signal cap + modeled max-loss vs balance. */
function optionsRiskCheck(user_id, signal, balance) {
  const maxPerDay = Number(process.env.OPT_MAX_SIGNALS_DAY) || 3;
  const maxTradePct = Number(process.env.RISK_MAX_TRADE_PCT) || 2;
  try {
    ensureOptionsTables(db);
    const today = db.prepare(
      "SELECT COUNT(*) AS c FROM options_signals WHERE user_id = ? AND date(timestamp/1000,'unixepoch') = date('now')"
    ).get(user_id)?.c || 0;
    if (today >= maxPerDay) return { ok: false, reason: `daily options-signal cap reached (${maxPerDay})` };
  } catch { /* cap best-effort */ }
  const riskDollars = Number(signal.maxLossDollars ?? signal.maxLoss ?? 0);
  if (balance > 0 && riskDollars > balance * maxTradePct / 100) {
    return { ok: false, reason: `modeled max loss $${riskDollars.toFixed(2)} exceeds ${maxTradePct}% of $${balance} — size down` };
  }
  return { ok: true };
}

async function fetchSentiment(symbol) {
  const sym = String(symbol).toUpperCase();
  const cacheKey = `sentiment:${sym}`;
  const cached = cacheGet(cacheKey);
  if (cached) return cached;

  try {
    // Use web search for news sentiment
    const queries = [
      `${sym} market news ${new Date().getFullYear()} sentiment`,
      `${sym} technical analysis forecast`,
      `${sym} fundamental outlook`
    ];
    
    const results = await Promise.all(queries.map(q => webSearch(q)));
    
    // Simple sentiment scoring based on keywords
    const text = results.join(" ").toLowerCase();
    const bullishWords = ["bullish", "buy", "long", "support", "breakout", "rally", "surge", "gain", "positive", "optimistic", "upside"];
    const bearishWords = ["bearish", "sell", "short", "resistance", "breakdown", "crash", "drop", "loss", "negative", "pessimistic", "downside"];
    
    let bullScore = 0, bearScore = 0;
    for (const w of bullishWords) bullScore += (text.match(new RegExp(w, "g")) || []).length;
    for (const w of bearishWords) bearScore += (text.match(new RegExp(w, "g")) || []).length;
    
    const total = bullScore + bearScore;
    let sentiment = "NEUTRAL";
    let score = 50;
    if (total > 0) {
      score = Math.round((bullScore / total) * 100);
      if (score > 60) sentiment = "BULLISH";
      else if (score < 40) sentiment = "BEARISH";
    }
    
    const data = {
      symbol: sym,
      sentiment,
      score,
      bullishSignals: bullScore,
      bearishSignals: bearScore,
      summary: results.slice(0, 2).join(" | ").slice(0, 500),
      sources: queries.length,
      cachedAt: Date.now()
    };
    cacheSet(cacheKey, data, 12 * 60 * 60 * 1000);
    return data;
  } catch (e) {
    console.warn("[fetchSentiment] failed:", e.message);
    return { symbol: sym, sentiment: "NEUTRAL", score: 50, summary: "Sentiment unavailable", cachedAt: Date.now() };
  }
}

async function generateSignalExplanation(signal, detailLevel = "full") {
  const sym = signal.symbol;
  const isBuy = signal.decision === "BUY";
  
  // Get fresh analysis for context
  const fresh = await mtfStrategy.analyze(sym, { interval: "1h" });
  
  const factors = [];
  
  // Macro trend factor
  if (fresh.regime?.macro_trend) {
    factors.push({
      name: "4H Macro Trend",
      value: fresh.regime.macro_trend,
      impact: fresh.regime.macro_trend !== "NEUTRAL" ? "HIGH" : "LOW",
      description: `The 4-hour timeframe shows a ${fresh.regime.macro_trend.toLowerCase()} trend (EMA 9/21 alignment + RSI + DI confirmation).`
    });
  }
  
  // 30M alignment
  if (fresh.regime?.primary_trend) {
    factors.push({
      name: "30M Primary Trend",
      value: fresh.regime.primary_trend,
      impact: "HIGH",
      description: `The 30-minute timeframe is ${fresh.regime.primary_trend === "BULLISH" ? "bullish" : "bearish"} (EMA 9 ${fresh.regime.primary_trend === "BULLISH" ? "above" : "below"} EMA 21).`
    });
  }
  
  // Pullback health
  if (signal.pullbackHealth) {
    factors.push({
      name: "Pullback Health",
      value: signal.pullbackHealth.is_healthy ? "HEALTHY" : "WEAK",
      impact: "MEDIUM",
      description: `Pullback probability: ${signal.pullbackHealth.pullback_probability}% vs reversal: ${signal.pullbackHealth.reversal_probability}%. ${signal.pullbackHealth.note || ""}`
    });
  }
  
  // Confluence
  if (signal.rangeInfo?.fib_confluence) {
    factors.push({
      name: "Fib + EMA Confluence",
      value: "YES",
      impact: "HIGH",
      description: "The pullback zone (50-61.8% Fib) aligns with 30M EMA 21, increasing entry quality."
    });
  }
  
  // Costs
  if (signal.costs) {
    factors.push({
      name: "Transaction Costs",
      value: `netRR ${signal.costs.netRR}`,
      impact: signal.costs.blocked ? "CRITICAL" : signal.costs.weak ? "MEDIUM" : "LOW",
      description: `After spread/commission/slippage: net RR = ${signal.costs.netRR} (${signal.costs.costPctOfRisk}% of risk). ${signal.costs.blocked ? "BLOCKED - negative expectancy" : signal.costs.weak ? "WEAK - needs confluence" : "ACCEPTABLE"}`
    });
  }
  
  // News
  if (fresh.news_filter?.has_news) {
    factors.push({
      name: "News Risk",
      value: "ACTIVE",
      impact: "HIGH",
      description: fresh.news_filter.recommendation || fresh.news_filter.reason
    });
  }
  
  const explanation = {
    signalId: signal.id,
    symbol: sym,
    decision: signal.decision,
    confidence: signal.confidence,
    plainEnglish: `${isBuy ? "Buy" : "Sell"} ${sym} at ${signal.entry} with SL ${signal.sl}, TP1 ${signal.tp1}, TP2 ${signal.tp2} (R:R ${signal.rr}). ${fresh.regime?.macro_trend || "Neutral"} macro trend, ${signal.scenario === "PULLBACK" ? "waiting for pullback entry" : "immediate entry"}.`,
    factors,
    riskWarning: signal.costs?.weak ? "Costs consume significant portion of risk - consider smaller size or wait for better confluence." : "Standard risk parameters apply.",
    detailLevel,
    generatedAt: Date.now()
  };
  
  return explanation;
}

function getLearningContent(category = null) {
  const content = {
    basics: [
      { id: "b1", title: "Understanding Forex Pairs", category: "basics", readTime: 5, content: "Forex pairs quote one currency against another..." },
      { id: "b2", title: "What is a Pip?", category: "basics", readTime: 3, content: "A pip is the smallest price move..." },
      { id: "b3", title: "Leverage and Margin", category: "basics", readTime: 7, content: "Leverage allows controlling larger positions..." },
    ],
    strategy: [
      { id: "s1", title: "Trend Following with EMA", category: "strategy", readTime: 10, content: "Our engine uses 30M EMA 9/21 aligned with 4H trend..." },
      { id: "s2", title: "Fibonacci Pullback Entries", category: "strategy", readTime: 8, content: "Structure-anchored entries at 50-61.8% retracement..." },
      { id: "s3", title: "Range Fading Strategy", category: "strategy", readTime: 8, content: "When 4H is neutral, fade proven range edges..." },
    ],
    risk: [
      { id: "r1", title: "Position Sizing Rules", category: "risk", readTime: 6, content: "Never risk more than 1-2% per trade..." },
      { id: "r2", title: "Correlation Management", category: "risk", readTime: 5, content: "EURUSD and GBPUSD move together - don't double risk..." },
      { id: "r3", title: "Daily Loss Limit", category: "risk", readTime: 4, content: "Stop trading if down 4% in a day - no revenge trading..." },
    ],
    psychology: [
      { id: "p1", title: "Discipline Over Prediction", category: "psychology", readTime: 5, content: "The market doesn't care about your opinion..." },
      { id: "p2", title: "Handling Drawdowns", category: "psychology", readTime: 6, content: "Every system has losing streaks - size for survival..." },
    ]
  };
  
  if (category && content[category]) {
    return { category, items: content[category] };
  }
  return { categories: Object.keys(content), all: content };
}

// ==================== POSITION MONITOR ====================
// Additive paper-position state machine. Watches positions registered by the
// daily trade scheduler and /trade; auto-resolves TP/SL via spot polling; logs
// the outcome to trade_memory + the paper logger. Never places orders itself.
const positionMonitor = new PositionMonitor({
  getPrice: async (sym) => { const s = await fetchSpotPrice(sym); return s?.price ?? null; },
  db,
  intervalMs: 60_000,
  onResolve: (pos) => {
    addTradeMemory(pos.symbol, {
      direction: pos.action,
      pattern: "position_monitor",
      outcome: pos.outcome,
      note: `Auto-resolved ${pos.outcome}: ${pos.note || ""} (source=${pos.source})`,
    });
  },
});
positionMonitor.start();

// ==================== MTF STRATEGY ENGINE (v1) ====================
// Multi-timeframe directional engine — default brain for /trade enhanced and
// /enhanced/analyze. Runs on Twelve Data free tier (8 credits/min) with its
// own cache + rate budget. See docs/STRATEGY.md.
// Additive layers: PullbackJournal (auto-records every published setup +
// scores accuracy), PortfolioRisk (pre-trade account guards). Neither changes
// the engine's signal logic — they record it and gate its execution.
const pullbackJournal = new PullbackJournal({ db });
const portfolioRisk = new PortfolioRisk({ db, pipConfig: PIP_CONFIG });
const mtfStrategy = new MTFStrategyEngine({
  fetchCandles,
  checkNewsFilter,
  calculateLotSize,
  addTradeMemory,
  recordSetup: (setup) => pullbackJournal.record(setup),
});

// ==================== LIVE FEED (24/7 WS scanning) ====================
// Free-tier survival design: ticks stream over Binance/Finnhub/Dukascopy,
// CandleStore builds 30M/4H locally, engine scans on candle close at 0
// Twelve Data credits. Backfill uses Finnhub REST first, then budgeted TD.
const liveFeed = new LiveFeed({
  db,
  engine: mtfStrategy,
  fetchImpl: fetch,
  fetchCandles,
  formatSignal: formatTradingSignal,
  storeSignal: (userId, signal) => storeSignalHistory(userId, signal),
  getWatchlist: () => {
    try {
      return db.prepare("SELECT user_id, symbol FROM trading_watchlist WHERE is_enabled = 1").all();
    } catch { return []; }
  },
  getPocketWatchlist: () => {
    try {
      return db.prepare("SELECT user_id, symbol FROM pocket_watchlist WHERE is_enabled = 1").all();
    } catch { return []; }
  },
  isCrypto: (sym) => CRYPTO_SET.has(String(sym).toUpperCase()),
  finnhubKey: FINNHUB_KEY,
  enabled: FEED_ENABLED,
  backfillBudget: FEED_BACKFILL_BUDGET,
  pocketEnabled: POCKET_ENABLED,
  // Shared rail clients (defined below; arrows resolve at call time, post-boot).
  makeDerivClient: () => derivClient,
  makeAlpacaClient: () => alpacaPaper,
  log: (...a) => console.log(...a),
  onCandleClose: (sym, interval) => { try { pocketOnClose(sym, interval); } catch (e) { console.warn("[pocket] close hook:", e.message); } },
});
globalThis.__liveFeed = liveFeed;
if (!FINNHUB_KEY) {
  console.log("[livefeed] FINNHUB_KEY unset — forex/metals run on Dukascopy poll backup; add key for WS streaming");
}
liveFeed.start().catch(e => console.warn("[livefeed] start failed:", e.message));

// ==================== POCKET FIXED-EXPIRY SIGNALS ====================
// Manual-execution signals for Pocket Option style trading (CALL/PUT +
// fixed expiry). Closed-candle only; paper ledger settles against OUR feed
// (broker OTC price may differ — stats are indicative, never P&L).
ensurePocketTables(db);

// Per-user Pocket bot switch. No row = ON (keeps existing behaviour); a row with
// enabled=0 stops that user's paper fan-out and, when every watcher of a symbol
// has it OFF, stops the shared Deriv auto-execution for that symbol too.
db.exec(`CREATE TABLE IF NOT EXISTS pocket_settings (
  user_id TEXT PRIMARY KEY,
  enabled INTEGER NOT NULL DEFAULT 1,
  updated_at INTEGER
)`);
function pocketUserEnabled(userId) {
  try {
    const r = db.prepare("SELECT enabled FROM pocket_settings WHERE user_id = ?").get(userId);
    return r ? r.enabled !== 0 : true;
  } catch { return true; }
}
function pocketSetEnabled(userId, on) {
  db.prepare("INSERT INTO pocket_settings (user_id, enabled, updated_at) VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET enabled = ?, updated_at = ?")
    .run(userId, on ? 1 : 0, Date.now(), on ? 1 : 0, Date.now());
}

function pocketClosed(sym, interval, n) {
  const arr = liveFeed.store.getCandles(sym, interval, n);
  // Drop the still-forming live edge — signals read closed candles only.
  return arr.length > 1 ? arr.slice(0, -1) : arr;
}

function pocketScan(sym, expiryMin = POCKET_EXPIRY_MIN) {
  const prof = profileFor(expiryMin);
  const trend = pocketClosed(sym, prof.trendIv, 160);
  const trig = pocketClosed(sym, prof.trigIv, 120);
  const base = { cryptoSet: CRYPTO_SET, expiryMin: prof.expiryMin, payout: POCKET_PAYOUT, ofi: liveFeed.store.ofi(sym) };
  // Pass 1 (no calibration) reveals the setup bucket; pass 2 applies the
  // learned win-rate gate. Two local runs cost nothing.
  const probe = signalFromCandles(sym, trend, trig, base);
  if (!probe.setupKey) return probe;
  const calib = calibFor(db, probe.setupKey);
  if (!calib) return probe;
  return signalFromCandles(sym, trend, trig, { ...base, calib });
}

function pocketFanOut(sym, signal) {
  // Dedupe: one paper per symbol+expiry per expiry window (system row gates
  // the per-user fan-out).
  try {
    const last = db.prepare(
      "SELECT entry_ts FROM pocket_paper WHERE user_id = '__pocket__' AND symbol = ? AND expiry_min = ? ORDER BY entry_ts DESC LIMIT 1"
    ).get(sym, signal.expiryMin);
    if (last && Date.now() - last.entry_ts < signal.expiryMin * 60_000) return false;
  } catch { /* ignore */ }
  recordPaper(db, { userId: "__pocket__", signal, payout: POCKET_PAYOUT });
  let watchers = [];
  try {
    // Pocket fans out to POCKET watchlist users only (separate list).
    watchers = db.prepare("SELECT user_id FROM pocket_watchlist WHERE symbol = ? AND is_enabled = 1").all(sym);
  } catch { /* ignore */ }
  for (const w of watchers) {
    if (!pocketUserEnabled(w.user_id)) continue; // bot switched OFF for this user
    recordPaper(db, { userId: w.user_id, signal, payout: POCKET_PAYOUT });
  }
  // Live execution (Deriv demo by default; auto flag + caps enforced inside).
  // Skipped when every watcher of this symbol has switched Pocket OFF.
  if (!watchers.length || watchers.some(w => pocketUserEnabled(w.user_id))) {
    pocketDerivExecute(sym, signal).catch(() => {});
  }
  return true;
}

function pocketOnClose(sym, interval) {
  if (!POCKET_ENABLED) return;
  // Scan on trigger-TF closes (1m/5m/15m are the v2 trigger timeframes).
  if (!["1m", "5m", "15m"].includes(interval)) return;
  for (const expiryMin of [1, 5, 15]) {
    const prof = profileFor(expiryMin);
    if (prof.trigIv !== interval) continue;
    const signal = pocketScan(sym, expiryMin);
    const arm = getArm(db, sym, signal.expiryMin);

    // 1. PULLBACK COMPLETION: an armed healthy pullback just re-aligned —
    //    this is the highest-quality entry (trend + completed pullback).
    if (arm && (signal.decision === "CALL" || signal.decision === "PUT") && signal.decision === arm.direction) {
      const pb = { ...signal, setupKey: `${signal.setupKey || `${sym}|${signal.expiryMin}m`}|pb` };
      pb.reasons = [...signal.reasons, `pullback COMPLETED: armed ${new Date(arm.armed_ts).toISOString().slice(11, 16)}GMT, trigger re-aligned with trend`];
      if (pocketFanOut(sym, pb)) {
        clearArm(db, sym, signal.expiryMin);
        console.log(`[pocket] ${pb.decision} ${sym} conf=${pb.confidence} expiry=${pb.expiryMin}m PULLBACK-COMPLETION`);
      }
      continue;
    }
    // Stale/opposite arm dies silently — never traded against.
    if (arm && signal.decision !== "WAIT" && signal.decision !== arm.direction) {
      clearArm(db, sym, signal.expiryMin);
    }

    // 2. IMMEDIATE aligned entries.
    if (signal.decision === "CALL" || signal.decision === "PUT") {
      if (pocketFanOut(sym, signal)) {
        clearArm(db, sym, signal.expiryMin);
        console.log(`[pocket] ${signal.decision} ${sym} conf=${signal.confidence} expiry=${signal.expiryMin}m`);
      }
      continue;
    }

    // 3. ARM on healthy pullbacks (no trade now — wait for re-alignment).
    if (signal.pullback?.verdict === "HEALTHY") {
      const dir = signal.pullback.trendDir > 0 ? "CALL" : "PUT";
      const waitN = PB_MAX_WAIT[signal.expiryMin] || 4;
      upsertArm(db, {
        symbol: sym, expiryMin: signal.expiryMin, direction: dir,
        setupKey: signal.setupKey || null,
        expiresTs: Date.now() + waitN * trigMinutes(prof.trigIv) * 60_000,
        note: `healthy pullback (retrace ${signal.pullback.retracePct?.toFixed(0) ?? "?"}%) — awaiting ${prof.trigIv} re-alignment`,
      });
    }
  }
}

// Settle expired paper trades every 20s against live-feed prices.
setInterval(() => {
  try {
    const r = settleOpen(db, (s) => liveFeed.lastPrice(s));
    if (r.settled) console.log(`[pocket] settled ${r.settled}`);
  } catch (e) { console.warn("[pocket] settle failed:", e.message); }
}, 20_000).unref?.();

// ==================== DERIV EXECUTION (demo default, real locked) ====================
const _derivToken = DERIV_LIVE && DERIV_REAL_TOKEN ? DERIV_REAL_TOKEN : DERIV_DEMO_TOKEN;
const derivClient = new DerivClient({
  token: _derivToken, appId: DERIV_APP_ID,
  log: (...a) => console.log(...a),
});
const derivExecutor = new DerivExecutor({
  db, client: derivClient,
  liveAllowed: DERIV_LIVE, stake: POCKET_DERIV_STAKE, maxPerDay: POCKET_DERIV_MAX_DAY,
  log: (...a) => console.log(...a),
});
// REST surface for account management / payments / app insights (DerivClient
// is WS-only). Same token; Deriv-App-ID header required for PAT auth.
const derivRest = new DerivRestClient({
  token: _derivToken, appId: DERIV_APP_ID, fetchImpl: fetch,
});
if (DERIV_REAL_TOKEN && !DERIV_LIVE) {
  console.log("[deriv] REAL token present but DERIV_LIVE!=1 — demo mode active, real refused");
}
if (!DERIV_DEMO_TOKEN && !DERIV_REAL_TOKEN) {
  console.log("[deriv] no token — signals + paper only. Paste DERIV_DEMO_TOKEN from your Deriv demo account to enable execution.");
}
if (POCKET_DERIV_AUTO) console.log("[deriv] AUTO-EXECUTION ENABLED — pocket signals will buy Rise/Fall contracts");

// ==================== ALPACA (stocks + crypto spot, paper default) ====================
const ALPACA_PAPER_KEY = process.env.ALPACA_PAPER_KEY || "";
const ALPACA_PAPER_SECRET = process.env.ALPACA_PAPER_SECRET || "";
const ALPACA_LIVE_KEY = process.env.ALPACA_LIVE_KEY || "";
const ALPACA_LIVE_SECRET = process.env.ALPACA_LIVE_SECRET || "";
const ALPACA_LIVE = process.env.ALPACA_LIVE === "1";
const ALPACA_AUTO = process.env.ALPACA_AUTO !== "0"; // paper auto by default (virtual); live needs flag
const ALPACA_NOTIONAL = Number(process.env.ALPACA_NOTIONAL || 50);
const ALPACA_MAX_DAY = Number(process.env.ALPACA_MAX_DAY || 20);
const alpacaPaper = new AlpacaClient({ keyId: ALPACA_PAPER_KEY, secret: ALPACA_PAPER_SECRET, paper: true, fetchImpl: fetch });
const alpacaLive = new AlpacaClient({ keyId: ALPACA_LIVE_KEY, secret: ALPACA_LIVE_SECRET, paper: false, fetchImpl: fetch });
const alpacaExecutor = new AlpacaExecutor({
  db, paperClient: alpacaPaper, liveClient: alpacaLive,
  liveAllowed: ALPACA_LIVE, notional: ALPACA_NOTIONAL, maxPerDay: ALPACA_MAX_DAY,
  log: (...a) => console.log(...a),
});
if (ALPACA_LIVE_KEY && !ALPACA_LIVE) console.log("[alpaca] LIVE keys present but ALPACA_LIVE!=1 — paper mode");
if (!ALPACA_PAPER_KEY) console.log("[alpaca] no paper keys — stock signals stay paper-ledged. Add ALPACA_PAPER_KEY/SECRET (free, email signup).");

// Deriv fills + broker-truth settlement into the shared monitor.
async function pocketDerivExecute(sym, signal) {
  if (!POCKET_DERIV_AUTO) return;
  try {
    const openN = derivExecutor.status().openContracts;
    if (openN >= 5) { console.log(`[deriv] ${openN} open — skipping ${sym} (concurrency cap)`); return; }
    const action = signal.decision === "CALL" ? "buy" : "sell";
    const pos = positionMonitor.register({
      symbol: sym, action, lotSize: derivExecutor.stake,
      entry: signal.entryRef ?? null, sl: null, tp: null,
      source: "pocket-deriv", reason: `pocket ${signal.decision} ${sym} ${signal.expiryMin}m conf=${signal.confidence}`,
    });
    const r = await derivExecutor.execute({
      symbol: sym, direction: signal.decision,
      durationMin: signal.expiryMin, positionId: pos?.id || null,
    });
    if (!r.ok && pos?.id) {
      try { positionMonitor.resolve(pos.id, "cancelled", null, `deriv refused: ${r.why}`); } catch (e) { console.warn("[deriv] resolve(cancelled) failed:", pos.id, e.message); }
    }
  } catch (e) { console.warn("[deriv] execute failed:", e.message); }
}

setInterval(async () => {
  try {
    const r = await derivExecutor.settleOpen(async (pid, outcome, px, note) => {
      try { positionMonitor.resolve(pid, outcome, px, note); } catch (e) { console.warn("[deriv] settle resolve failed:", pid, e.message); }
    });
    if (r.settled) console.log(`[deriv] settled ${r.settled} contract(s) on broker outcome`);
  } catch (e) { console.warn("[deriv] settle failed:", e.message); }
  try {
    const s = await alpacaExecutor.syncFills();
    if (s.synced) console.log(`[alpaca] synced ${s.synced} fill(s)`);
  } catch (e) { console.warn("[alpaca] sync failed:", e.message); }
}, 15_000).unref?.();

// Every 30 min, replay 30M candles and resolve journaled setups (win/loss/
// expired). Only symbols with pending rows cost API calls; engine + fetch
// caches absorb the rest. Failures never crash the server.
setInterval(async () => {
  try {
    const r = await pullbackJournal.resolvePending({ fetchCandles });
    if (r.checked) console.log(`[PullbackJournal] auto-resolve: checked=${r.checked} resolved=${r.resolved} expired=${r.expired}`);
  } catch (e) {
    console.warn("[PullbackJournal] auto-resolve failed:", e.message);
  }
}, 30 * 60 * 1000);

// Shared pre-trade gate: portfolio risk layer. Used by /trade AND the
// scheduler executor so no path to the bridge bypasses account guards.
function riskGate({ symbol, action, lotSize, entry, sl, balance }) {
  return portfolioRisk.check({ symbol, action, lotSize, entry, sl, balance: balance || 1000 });
}

// Recurring "everyday analyze XAUUSD" tasks — paper-first by design.
// Shared trade executor: every auto path (/trade, scheduler, bots) funnels
// through rail routing + risk gate + execution + position monitor. No bypasses.
//
// Rails: deriv synthetics -> Deriv MULT (stake = max loss by construction);
// US stocks/ETFs -> Alpaca bracket (native TP/SL, clock-gated, dollar-sized);
// everything else -> MT5 bridge (existing behavior, paper while unset).
// riskUsdOverride is ALWAYS set for deriv/alpaca so the per-trade cap can
// never be silently skipped for lack of a PIP row.
async function executeTradeSignal({ symbol, action, lotSize, entry, sl, tp, reason, source = "manual", balance = 1000, riskPercent = 1, userId = null, botName = null, expiryMin = 0 }) {
  const sym = canonicalSymbol(symbol);
  const rail = marketOf(sym);
  const riskBudget = Math.max(0.5, (Number(balance) || 1000) * (Number(riskPercent) || 1) / 100);
  // The user's OWN balance is the sizing base when we know who is trading.
  if (userId) {
    const acct = getTradingAccount(userId);
    if (acct?.balance > 0) balance = acct.balance;
  }
  if (rail === "deriv") return executeDerivTrade({ symbol: sym, action, entry, sl, tp, reason, source, riskBudget, userId, botName, expiryMin, balance });
  if (rail === "alpaca") return executeAlpacaTrade({ symbol: sym, action, entry, sl, tp, reason, source, riskBudget, balance });
  // F-06: bots pass a placeholder 0.01 lot. Size MT5-bridge trades from the
  // risk budget (balance x riskPercent, capped by calculateLotSize) instead;
  // the risk gate below still has the final say.
  if (String(source).startsWith("bot:") && sl && entry) {
    const sized = calculateLotSize({ symbol: sym, balance, riskPercent, entry, stopLoss: sl });
    if (sized > 0) lotSize = sized;
  }
  const gate = riskGate({ symbol: sym, action, lotSize, entry, sl, balance });
  if (!gate.allowed) {
    console.log(`[executor] RISK-BLOCKED ${sym}: ${gate.reasons.join(" | ")}`);
    return { mode: "blocked", status: "blocked", blocked_by: "risk", reasons: gate.reasons, exposure: gate.exposure };
  }
  // No MT5 bridge (no VPS yet): Deriv MULT covers whatever it lists — the
  // executor verifies MULT availability per symbol, so the only trades that
  // stay paper are ones no connected broker supports. MT5 bridge wins back
  // automatically the moment MT5_BRIDGE_URL is set.
  //
  // This is the path FX and metals take (they are "mt5" rail by symbol class
  // but Deriv lists them as real OTC pairs), so bot orders for EURUSD/XAUUSD
  // reach Deriv through here.
  if (!MT5_BRIDGE_URL) {
    return executeDerivTrade({ symbol: sym, action, entry, sl, tp, reason, source, riskBudget, userId, botName, expiryMin, balance });
  }
  const result = await sendToMT5Bridge({ symbol: sym, action, lotSize, entry, sl, tp, reason });
  if (result.status !== "failed" && sl && tp) {
    try {
      positionMonitor.register({ symbol: sym, action, lotSize, entry, sl, tp, source, reason });
    } catch (e) { console.warn("[PositionMonitor] register failed:", e.message); }
  }
  return result;
}

/** Deriv MULT execution. take_profit/stop_loss are dollar approximations
 *  (moveRatio x multiplier x stake at the default x100); Deriv prices the
 *  proposal exactly and the native contract levels enforce the exit. */
async function executeDerivTrade({ symbol, action, entry, sl, tp, reason, source, riskBudget, userId = null, botName = null, expiryMin = 0, balance = 1000 }) {
  const stake = Math.max(0.5, Math.min(1000, Number(riskBudget) || 5));
  const gateBalance = Number(balance) > 0 ? Number(balance) : 1000;
  const direction = String(action).toLowerCase() === "sell" ? "sell" : "buy";
  const e = Number(entry);
  const mult = Number(process.env.DERIV_MULTIPLIER) || 100;
  // A user who supplied their OWN Deriv token gets their OWN account traded.
  // The platform token (from Render env) is the default for everyone else.
  let executor = derivExecutor;
  let ownExecutor = null;
  if (userId) {
    try {
      const acct = getTradingAccount(userId);
      if (acct?.derivTokenMode === "own" && acct.hasDerivToken) {
        const raw = db.prepare("SELECT deriv_token FROM trading_account WHERE user_id = ?").get(userId)?.deriv_token;
        if (raw && String(raw).length > 4) {
          const client = new DerivClient({ token: String(raw).trim(), appId: DERIV_APP_ID, log: (...a) => console.log(...a) });
          ownExecutor = new DerivExecutor({
            db, client, liveAllowed: DERIV_LIVE, stake, maxPerDay: POCKET_DERIV_MAX_DAY,
            log: (...a) => console.log(...a),
          });
          executor = ownExecutor;
        }
      }
    } catch (e) { console.warn("[deriv] own-token executor failed:", e.message); }
  }
  // TRUE POCKET OPTION: a CALL/PUT (Rise/Fall) contract with a fixed expiry.
  // The pocket bot predicts whether the candle ENDS bullish or bearish within
  // the given time, so it must buy a Rise/Fall contract — not a MULT contract
  // (which has no expiry and rides TP/SL instead). The scalping bot, which
  // DOES carry TP/SL, keeps the MULT path.
  if (source === "bot:pocket" && Number(expiryMin) > 0) {
    const dir = String(action).toLowerCase() === "sell" ? "PUT" : "CALL";
    const gateP = riskGate({ symbol, action: dir === "PUT" ? "sell" : "buy", lotSize: 0, entry, sl: null, balance: gateBalance, riskUsdOverride: stake });
    if (!gateP.allowed) {
      return { mode: "blocked", status: "blocked", blocked_by: "risk", reasons: gateP.reasons, exposure: gateP.exposure };
    }
    try {
      const dSym = await executor.resolveSymbol(symbol);
      if (!dSym) {
        return { mode: "paper", status: "simulated", symbol, action: dir, stake,
                 message: `no Rise/Fall market for ${symbol} on this account — paper mode` };
      }
      const r = await executor.client.buyRiseFall({
        symbol: dSym, direction: dir, stake, durationMin: Number(expiryMin), currency: "USD",
      });
      if (r.error) {
        return { mode: "paper", status: "simulated", symbol, action: dir, stake,
                 message: `Deriv refused (${r.error}) — paper mode` };
      }
      try {
        db.prepare(`
          INSERT OR REPLACE INTO deriv_contracts
          (contract_id, position_id, symbol, deriv_symbol, direction, stake, duration_min, entry_tick, status, mode, placed_at)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
        `).run(String(r.contract_id), null, symbol, dSym, dir, stake, Number(expiryMin),
          r.entry_tick ?? null, executor.client.account?.isDemo ? "demo" : "REAL", Date.now());
      } catch (e) { console.warn("[deriv] pocket ledger failed:", e.message); }
      const mode = executor.client.account?.isDemo ? "demo" : "REAL";
      console.log(`[deriv] ${mode} ${dir} ${dSym} ${expiryMin}m stake=${stake} contract=${r.contract_id}`);
      return { mode: mode === "REAL" ? "deriv-live" : "deriv-demo", status: "sent", symbol, action: dir,
               stake, contract_id: r.contract_id, expiryMin: Number(expiryMin), reason };
    } catch (err) {
      console.log(`[Deriv] pocket paper fallback ${symbol}: ${err.message}`);
      return { mode: "paper", status: "simulated", symbol, action: dir, stake,
               message: `Deriv unavailable (${err.message}) — paper mode` };
    }
  }
  const takeProfit = e > 0 && tp != null ? +((Math.abs(tp - e) / e) * mult * stake).toFixed(2) : null;
  const stopLoss = e > 0 && sl != null ? +Math.min((Math.abs(e - sl) / e) * mult * stake, stake).toFixed(2) : null;
  // FIXED BUG: the risk gate was checked against a HARDCODED $1000 balance
  // while the stake was sized from the user's real balance. A $50k account
  // had its trade judged as if it risked 5% of $1000 — the guard was
  // measuring the wrong account entirely. The user's real balance is used
  // for both sizing and the risk check.
  const gate = riskGate({ symbol, action: direction, lotSize: 0, entry, sl, balance: gateBalance, riskUsdOverride: stake });
  if (!gate.allowed) {
    return { mode: "blocked", status: "blocked", blocked_by: "risk", reasons: gate.reasons, exposure: gate.exposure };
  }
  let result;
  try {
    const r = executor.executeMult
      ? await executor.executeMult({ symbol, direction, stake, multiplier: mult, takeProfit, stopLoss })
      : { ok: false, why: "mult path unavailable" };
    if (!r.ok) throw new Error(r.why || "deriv refused");
    result = { mode: r.mode === "REAL" ? "deriv-live" : "deriv-demo", status: "sent", symbol, action: direction, stake, contract_id: r.contract_id, tp, sl, reason };
    // Tag the source with the bot name so the portfolio attributes it.
    const src = botName ? `${source}` : source;
    try {
      const pos = positionMonitor.register({ symbol, action: direction, lotSize: stake, entry, sl, tp, source: `${src}:${result.mode}`, reason });
      r.positionId = pos?.id;
    } catch (e) { console.warn("[PositionMonitor] deriv register failed:", e.message); }
  } catch (err) {
    console.log(`[Deriv] paper fallback ${symbol}: ${err.message}`);
    result = { mode: "paper", status: "simulated", symbol, action: direction, stake, tp, sl, reason, message: `Deriv unavailable (${err.message}) — paper mode` };
    try {
      positionMonitor.register({ symbol, action: direction, lotSize: stake, entry, sl, tp, source: `${source}:deriv-paper`, reason });
    } catch (e) { console.warn("[PositionMonitor] deriv-paper register failed:", e.message); }
  }
  return result;
}

/** Alpaca bracket execution: dollar-risk share qty, native TP/SL, clock gate. */
async function executeAlpacaTrade({ symbol, action, entry, sl, tp, reason, source, riskBudget, balance = 1000 }) {
  const direction = String(action).toLowerCase() === "sell" ? "sell" : "buy";
  const stopDist = Math.abs(Number(entry) - Number(sl));
  const qty = qtyForRisk({ price: entry, stopDist, riskUsd: riskBudget });
  if (qty == null) {
    return { mode: "blocked", status: "blocked", blocked_by: "sizing", reasons: [`qty math failed (entry=${entry} sl=${sl} risk=$${riskBudget}) — no order`] };
  }
  // Same fix as the Deriv path: judge the risk against the user's REAL
  // balance, not a hardcoded $1000.
  const gate = riskGate({ symbol, action: direction, lotSize: 0, entry, sl, balance: Number(balance) > 0 ? Number(balance) : 1000, riskUsdOverride: stopDist * qty });
  if (!gate.allowed) {
    return { mode: "blocked", status: "blocked", blocked_by: "risk", reasons: gate.reasons, exposure: gate.exposure };
  }
  const r = await alpacaExecutor.place({ symbol, action: direction, qty, takeProfit: tp, stopLoss: sl });
  if (!r.ok) {
    // Refused (no keys / closed market / cap) -> paper-tracked, still watched.
    console.log(`[executor] alpaca refused ${symbol} (${r.why}) — paper-tracked`);
    try {
      positionMonitor.register({ symbol, action: direction, lotSize: qty, entry, sl, tp, source: `${source}:alpaca-paper`, reason: `${reason || ""} [alpaca refused: ${r.why}]` });
    } catch (e) { console.warn("[PositionMonitor] alpaca-paper register failed:", e.message); }
    return { mode: "paper", status: "simulated", symbol, action: direction, qty, entry, sl, tp, reason, message: `Alpaca refused (${r.why}) — paper mode` };
  }
  try {
    positionMonitor.register({
      symbol, action: direction, lotSize: qty, entry, sl, tp,
      source: r.mode === "live" ? "alpaca-live" : "alpaca",
      reason: `${reason || ""} [alpaca ${r.mode} order ${r.order?.id || "?"}]`,
    });
  } catch (e) { console.warn("[PositionMonitor] alpaca register failed:", e.message); }
  return { mode: r.mode === "live" ? "alpaca-live" : "alpaca-paper", status: "sent", order: r.order, qty };
}
const tradeScheduler = new TradeTaskScheduler({
  engine: mtfStrategy,
  executor: (args) => executeTradeSignal({ ...args, source: "scheduler" }),
  db, // FIX (F-05): persist scheduled trade tasks — they previously died with every restart
});
if (process.env.TRADE_TASKS === "true") {
  tradeScheduler.start();
  console.log("[FRIT] Trade task scheduler started (TRADE_TASKS=true)");
}

// ==================== ANDROID AGENT TOOLS ====================
// Contract: tools in SERVER_SIDE_TOOLS execute HERE (inside runAgentStep).
// Everything else in AGENT_TOOLS is returned to the Android app as a
// pending_action and executed by IntegrationCoordinator.executeEdgeTool().
// run_command / write_file / run_sandbox_code are deliberately NOT advertised:
// they stay gated behind DEV_TOOLS_ENABLED=false on the client.
const SERVER_SIDE_TOOLS = new Set(["search_web", "get_weather", "get_market_data", "get_market_news",
  "analyze_market", "run_code", "wait_and_verify", "assert_text_visible", "get_frit_manual",
  "delegate_subtasks", "generate_image", "design_image", "decompose_design", "generate_video", "create_content", "ground_scene"
]);

// ---- In-loop entitlement: WHAT the brain may touch is decided by the PLAN,
// not the route (market questions arrive via /agent/start like everything
// else). Server tools are gated + billed here; device tools need "automation".
// Billed strictly AFTER success — blocked/failed steps cost nothing.
const LOOP_TOOL_FEATURE = {
  analyze_market: "trading",
  get_market_data: "market_data", get_market_news: "market_data",
  run_code: "chat",
  generate_image: "image", design_image: "image", decompose_design: "image",
  generate_video: "video",
  create_content: "creative", ground_scene: "chat",
  delegate_subtasks: "agent",
  search_web: "chat", get_weather: "chat", get_frit_manual: "chat",
  wait_and_verify: "chat", assert_text_visible: "chat",
};
const LOOP_TOOL_CHARGE = {
  analyze_market: "trade_analysis",
  generate_image: "image", design_image: "image", decompose_design: "image",
  run_code: "run_code", delegate_subtasks: "agent_task",
  create_content: "turn",
  // generate_video is async: tracked at submit, charged once on completion.
};
function gateLoopTool(uid, name) {
  const feature = LOOP_TOOL_FEATURE[name] || "chat";
  grantMonthlyCredits(uid);
  const tier = activeTier(uid);
  if (!planAllows(tier, feature)) {
    return { ok: false, message: `${FEATURE_LABELS[feature] || feature} isn't in your ${tier} plan — upgrade to use ${name}. Step blocked, no credits spent.` };
  }
  const charge = LOOP_TOOL_CHARGE[name];
  if (charge && creditBalance(uid) < (CREDIT_COSTS[charge] ?? 1)) {
    return { ok: false, message: `Not enough credits for ${name} (${creditBalance(uid)} left). Top up in Wallet — step blocked, nothing spent.` };
  }
  return { ok: true, charge: charge || null };
}

const AGENT_TOOLS = [
  // ---- Phone/UI control (executed on Android) ----
  { type: "function", function: { name: "open_app", description: "Launch any installed app by name or package (Opay, Facebook, MT5, WhatsApp...). The phone waits ~4s for cold start. ALWAYS follow with read_screen to confirm the right screen loaded before tapping.", parameters: { type: "object", properties: { app_name: { type: "string" }, package: { type: "string" } } } } },
  { type: "function", function: { name: "execute_local_action", description: "Offload app launching, local macros, settings toggles, or device utilities directly to the local on-device engine to save API credits and turns.", parameters: { type: "object", properties: { action: { type: "string", description: "The local action or command to execute (e.g. 'open MT5', 'toggle mobile data', 'set alarm for 7am')" } }, required: ["action"] } } },
  { type: "function", function: { name: "return_to_frit", description: "Bring the FRIT app back to the foreground. Call this when a phone task in another app/settings is finished.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "read_screen", description: "Read visible text from the screen. Use this frequently to observe state and verify the result of every action.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "read_screen_structured", description: "Return exact coordinates of UI elements. Essential for precise clicking.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "tap_button", description: "Tap a button by its visible label/text. If it reports not-found, the phone auto-tries coordinate fallback + scroll — then re-read the screen and retry with tap_coordinates from read_screen_structured.", parameters: { type: "object", properties: { label: { type: "string" } }, required: ["label"] } } },
  { type: "function", function: { name: "tap_coordinates", description: "Tap specific x/y coordinates. Use when text-based tapping fails.", parameters: { type: "object", properties: { x: { type: "number" }, y: { type: "number" } }, required: ["x", "y"] } } },
  { type: "function", function: { name: "type_text", description: "Type text into the focused input field.", parameters: { type: "object", properties: { value: { type: "string" }, field: { type: "string" } }, required: ["value"] } } },
  { type: "function", function: { name: "scroll", description: "Scroll the UI in a direction.", parameters: { type: "object", properties: { direction: { type: "string", enum: ["up", "down", "left", "right"] } }, required: ["direction"] } } },
  { type: "function", function: { name: "go_back", description: "Press the Android back button.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "press_home", description: "Go to the home screen.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "toggle_notifications", description: "Pull down the notification shade. Use to check/read notifications, not for settings toggles (use execute_local_action for wifi/bluetooth/data).", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "toggle_quick_settings", description: "Open Quick Settings panel (the expanded shade with wifi/bluetooth/flashlight tiles).", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "recent_apps", description: "Open the Recent Apps / app switcher screen.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "lock_screen", description: "Lock the device screen immediately. Only use when the user explicitly asks to lock the phone.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "swipe_coordinates", description: "Swipe upward starting from x/y coordinates (e.g. to dismiss a card or scroll a custom-drawn view tap_button/scroll can't reach).", parameters: { type: "object", properties: { x: { type: "number" }, y: { type: "number" } }, required: ["x", "y"] } } },
  { type: "function", function: { name: "long_press_element", description: "Long-press a UI element by its visible text label (opens context menus, drag handles, etc).", parameters: { type: "object", properties: { label: { type: "string" } }, required: ["label"] } } },
  { type: "function", function: { name: "take_native_screenshot", description: "Trigger Android's own screenshot capture (saves to gallery) — different from take_screenshot, which captures for in-agent vision analysis only.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "get_current_app", description: "Return the name of the currently focused app.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "get_current_activity", description: "Return the current Android activity/class.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "take_screenshot", description: "Capture the current screen for later analysis.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "analyze_screenshot", description: "Send the last screenshot to a vision model for UI analysis.", parameters: { type: "object", properties: { prompt: { type: "string" } } } } },
  { type: "function", function: { name: "set_volume", description: "Set media/alarm volume level (0-100).", parameters: { type: "object", properties: { level: { type: "number" } }, required: ["level"] } } },
  { type: "function", function: { name: "get_app_pin", description: "Return the user's stored app-transaction PIN (e.g. OPay payment PIN) from the on-device vault. Call ONLY when an app screen explicitly demands a PIN you were never given. Never speak, print, or echo the PIN into chat — type it straight into the field with type_text and move on.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "listen_ambient", description: "Record N seconds of ambient room audio and transcribe it. Use for contact voice notes: open the chat, tap play on the voice bubble (speaker), then call this with seconds=30. Returns the transcript.", parameters: { type: "object", properties: { seconds: { type: "number", description: "5-60" } } } } },

  // ---- Communication & daily-life tasks (general-purpose) ----
  { type: "function", function: { name: "send_whatsapp", description: "Open a WhatsApp chat with a contact (by name) and draft a message. Then verify and send via UI.", parameters: { type: "object", properties: { contact_name: { type: "string" }, message: { type: "string" } }, required: ["contact_name", "message"] } } },
  { type: "function", function: { name: "send_sms", description: "Open the SMS composer to a contact with a message. Then verify and send via UI.", parameters: { type: "object", properties: { contact_name: { type: "string" }, message: { type: "string" } }, required: ["contact_name", "message"] } } },
  { type: "function", function: { name: "make_call", description: "Place a phone call to a contact. Set converse:true when the user wants FRIT to actually TALK on the call (e.g. 'call John and ask him X') — FRIT waits for the other side to answer, then holds a live half-duplex voice conversation (greet, listen, reply, hangs up on goodbye/silence/5-min cap). With converse:true the task is NOT done when the call is placed — it continues until the conversation ends. With converse false/default the task IS done as soon as the dialer opens.", parameters: { type: "object", properties: { contact_name: { type: "string" }, phone_number: { type: "string" }, converse: { type: "boolean" } }, required: ["contact_name"] } } },
  { type: "function", function: { name: "answer_calls", description: "Enable/disable auto-answer for INCOMING calls. The phone answers and FRIT holds a live half-duplex voice conversation (greet, listen, reply, hang up on goodbye/silence/5-min cap) over the speakerphone bridge. Works in reasonably quiet rooms; do NOT promise studio quality or interruption handling.", parameters: { type: "object", properties: { enabled: { type: "boolean" } } } } },
  { type: "function", function: { name: "set_alarm", description: "Set an alarm at a given time.", parameters: { type: "object", properties: { time: { type: "string" }, label: { type: "string" } }, required: ["time"] } } },
  { type: "function", function: { name: "set_timer", description: "Start a countdown timer.", parameters: { type: "object", properties: { duration: { type: "string" } }, required: ["duration"] } } },
  { type: "function", function: { name: "play_music", description: "Play music or media by query.", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
  { type: "function", function: { name: "navigate_to", description: "Open navigation/GPS to a destination.", parameters: { type: "object", properties: { destination: { type: "string" } }, required: ["destination"] } } },
  { type: "function", function: { name: "take_photo", description: "Open the camera app for a photo.", parameters: { type: "object", properties: { front_camera: { type: "boolean" } } } } },

  // ---- Server-side tools (execute on the server) ----
  { type: "function", function: { name: "run_code", description: "Execute Python/JS code server-side in a real sandbox (NO phone needed). Python has numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn, TA-Lib, requests, openpyxl, python-docx, python-pptx, reportlab, pillow. Limits: 15s runtime, ~256MB memory, nothing persists between calls. To return a visual or file, WRITE it to the working directory: plt.savefig('chart.png') (never plt.show()), df.to_csv('data.csv'), wb.save('report.xlsx'), open('index.html','w'). Files are returned to the user automatically (max 10 files, 3MB each). Use print() for text results. ALWAYS use this instead of the phone for anything computational or for building files/webpages.", parameters: { type: "object", properties: { language: { type: "string", enum: ["python", "javascript"] }, code: { type: "string" }, stdin: { type: "string" }, timeout_ms: { type: "number", description: "max 20000" } }, required: ["language", "code"] } } },
  { type: "function", function: { name: "search_web", description: "Deep web research spanning DuckDuckGo, Bing and Brave. Returns a real ranked result list (titles, URLs, snippets), NOT one abstract. Supports search-dorking operators: site:, intitle:, inurl:, filetype:, -keyword, \"exact phrase\". Call multiple times with refined queries for multi-angle research.", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
  { type: "function", function: { name: "get_weather", description: "Get current weather for a city.", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } },
  { type: "function", function: { name: "get_market_data", description: "Fetch live spot prices for one or more symbols (e.g. XAUUSD, BTCUSD).", parameters: { type: "object", properties: { symbol: { type: "string" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "analyze_market", description: "Full multi-timeframe engine analysis for a symbol: direction, confidence, entry/SL/TP, regime. ALWAYS call this (not memory) before giving any trade opinion.", parameters: { type: "object", properties: { symbol: { type: "string" }, interval: { type: "string" }, balance: { type: "number" }, risk_percent: { type: "number" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "get_market_news", description: "Fetch real-time financial, fundamental, and macroeconomic news for any trading symbol (XAUUSD, BTCUSD, Forex, Stocks) for the current date/year.", parameters: { type: "object", properties: { symbol: { type: "string" }, query: { type: "string" } } } } },
  { type: "function", function: { name: "place_mt5_trade", description: "Place a real market order on MetaTrader 5 via the phone's MT5 agent. Use this after market analysis confirms a high-confidence entry signal.", parameters: { type: "object", properties: { symbol: { type: "string" }, action: { type: "string", enum: ["BUY", "SELL"] }, volume: { type: "number" }, sl: { type: "number" }, tp: { type: "number" } }, required: ["symbol", "action", "volume"] } } },
  { type: "function", function: { name: "modify_mt5_order", description: "Modify SL/TP of an open MT5 position via guided on-device UI steps (Trade tab -> long-press -> Modify). If unsure about MT5 menu layout, call search_web first (e.g. 'MT5 android modify SL TP steps').", parameters: { type: "object", properties: { symbol: { type: "string" }, sl: { type: "number" }, tp: { type: "number" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "close_mt5_order", description: "Close an open MT5 position via guided on-device UI steps (Trade tab -> long-press -> Close).", parameters: { type: "object", properties: { symbol: { type: "string" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "get_mt5_positions", description: "Read currently visible MT5 positions from the phone screen (symbol, side, volume, P/L). Call read_screen first if empty.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "get_systems_status", description: "Get overall server system status.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "wait_and_verify", description: "Wait a moment before verifying state (use after actions that take time).", parameters: { type: "object", properties: { delay_ms: { type: "number" } } } } },
  { type: "function", function: { name: "assert_text_visible", description: "Verify that text is visible on the last screen state.", parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } } },
  { type: "function", function: { name: "get_frit_manual", description: "Get a full reference of every real tool FRIT has, grouped by category, plus how to use the phone's installed-apps list correctly. Call this only if you're unsure what capabilities you have — don't call it for every task.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "delegate_subtasks", description: "Fan OUT independent subtasks to parallel OpenRouter subagents (GLM-5.3-Flash heavy, gpt-oss-20b speed, Qwen draft, zero extra agent turns). Use for: researching several angles at once, drafting + summarizing while you keep driving the phone, comparing options. Args: subtasks = [{task, kind}] where kind is lookup/extract/draft/summarize/research/analyze/code. Max 4 per call. Results come back merged in one response.", parameters: { type: "object", properties: { subtasks: { type: "array", items: { type: "object", properties: { task: { type: "string" }, kind: { type: "string" } }, required: ["task"] } } }, required: ["subtasks"] } } },
  { type: "function", function: { name: "generate_image", description: "Generate an image (default Seedream 5.0 Flash, cheap). ALWAYS ground_scene first for realistic scenes. Needs paid OpenRouter credits.", parameters: { type: "object", properties: { prompt: { type: "string" }, aspect_ratio: { type: "string" }, n: { type: "number" }, model: { type: "string" } }, required: ["prompt"] } } },
  { type: "function", function: { name: "design_image", description: "Graphic-design generation (posters, UI, infographics, legible text) via Ming Design — FREE. Use when the user asks for design work, NOT photos.", parameters: { type: "object", properties: { prompt: { type: "string" }, output_format: { type: "string" } }, required: ["prompt"] } } },
  { type: "function", function: { name: "decompose_design", description: "Decompose a FLAT design image into editable RGBA layers (background + foreground) via Ming Layer — FREE. Needs the image (base64 data URL or http URL) + a layer plan.", parameters: { type: "object", properties: { image: { type: "string" }, prompt: { type: "string" }, layer_plan: { type: "string" } }, required: ["image"] } } },
  { type: "function", function: { name: "generate_video", description: "Generate a short video clip via OpenRouter (veo-3.1-lite 720p 4-8s default). Async: returns job id + polling_url, poll then download. Use for creative content shots.", parameters: { type: "object", properties: { prompt: { type: "string" }, duration: { type: "number" }, aspect_ratio: { type: "string" }, resolution: { type: "string" }, model: { type: "string" } }, required: ["prompt"] } } },
  { type: "function", function: { name: "create_content", description: "Creative mode: turn a video brief/transcript into script + shot list (image/video prompts) + caption. Then call generate_image/generate_video per shot.", parameters: { type: "object", properties: { brief: { type: "string" }, transcript: { type: "string" } } } } },
  { type: "function", function: { name: "ground_scene", description: "Before-math: compute real-world sizes, relative scale, camera and depth order for a scene brief. Returns enriched prompt + negative prompt. ALWAYS call this before generate_image/generate_video for realistic scenes.", parameters: { type: "object", properties: { brief: { type: "string" }, prompt: { type: "string" } } } } },
];

// Auto-generated from AGENT_TOOLS itself, so this can never drift out of sync
// with what's actually callable (unlike a hand-written manual).
function buildFritManual() {
  const lines = ["# FRIT Tool Reference (auto-generated from live tool registry)", ""];
  for (const t of AGENT_TOOLS) {
    if (t.function.name === "get_frit_manual") continue;
    const category = SERVER_SIDE_TOOLS.has(t.function.name) ? "server-side (runs here, no device needed)" : "device-side (executed on the phone)";
    lines.push(`- ${t.function.name} [${category}]: ${t.function.description}`);
  }
  lines.push(
    "",
    "Notes:",
    "- 'open_app' and 'execute_local_action' offload app launching and local device operations directly to the phone's local engine to conserve server tokens.",
    "- 'open_app' works reliably for apps present in the device state's 'Installed apps' list — check that list before assuming an app exists.",
    "- There is no generic message/call tool beyond what's listed above (send_whatsapp, send_sms, make_call). If a task needs an app not covered by a dedicated tool, use open_app + read_screen + tap/type to drive it manually.",
    "- run_code executes real Python/JS server-side via the sandbox. Files it writes (charts, CSVs, reports) are captured as artifacts and returned to the user — use it instead of writing code as plain text, and prefer writing a file when the user wants something visual.",
  );
  return lines.join("\n");
}

// ==================== LOCAL TOOL EXECUTION ====================
async function runLocalTool(name, args = {}, agentState = null) {
  switch (name) {
    case "search_web": return { ok: true, data: await webSearch(args.query || "") };
    case "get_weather": return { ok: true, data: await getWeather(args.city || "Lagos") };
    case "get_market_data": {
      const sym = extractMarketSymbol(args.symbol);
      if (!sym) return { ok: false, error: "No symbol given. Ask which market (e.g. XAUUSD, EURUSD, BTCUSD) instead of guessing one." };
      return { ok: true, data: await fetchMarketPrices([sym]) };
    }
    case "get_market_news": {
      const sym = args.symbol || "market";
      const q = args.query || `${sym} market fundamental news ${new Date().getFullYear()}`;
      return { ok: true, data: await webSearch(q) };
    }
    case "analyze_market": {
      const sym = extractMarketSymbol(args.symbol);
      if (!sym) return { ok: false, error: "No symbol given. Ask which market to analyse instead of guessing one." };
      return { ok: true, data: await mtfStrategy.analyze(sym, { interval: args.interval, balance: args.balance, riskPercent: args.risk_percent }) };
    }
    case "run_code": return { ok: true, data: await runSandbox({ language: args.language, code: args.code, stdin: args.stdin || "", timeout_ms: args.timeout_ms || 15000 }) };
    case "get_frit_manual": return { ok: true, data: buildFritManual() };
    case "generate_image": return { ok: true, data: await generateImage(args.prompt || "", { aspect_ratio: args.aspect_ratio, n: args.n, model: args.model }) };
    case "design_image": return { ok: true, data: await generateImage(args.prompt || "", { model: OR_DESIGN_MODEL, output_format: args.output_format || "png" }) };
    case "decompose_design": return { ok: true, data: await decomposeDesign(args.image || "", args.prompt || args.layer_plan || "decompose into background and foreground layers") };
    case "generate_video": return { ok: true, data: await submitVideo(args.prompt || "", { duration: args.duration, aspect_ratio: args.aspect_ratio, resolution: args.resolution, model: args.model }) };
    case "ground_scene": return { ok: true, data: await groundScene(args.brief || args.prompt || "") };
    case "create_content": {
      // Creative personality: transcript/notes -> GLM script + shot list -> image + video jobs.
      // Client sends text + optional frame descriptions (video bytes stay on-device).
      const brief = String(args.brief || args.transcript || "");
      const out = await chatWithFallback("conversation", {
        messages: [
          { role: "system", content: "You are FRIT Creative. Turn the video brief into: 1) 60-word hook script, 2) 3-5 shot list with per-shot image/video prompts, 3) caption + hashtags. Reply JSON {script, shots:[{kind:'image'|'video', prompt}], caption}." },
          { role: "user", content: brief.slice(0, 4000) },
        ],
        max_tokens: 1500,
      });
      return { ok: true, data: out.choices[0].message.content };
    }
    // Fan-out: independent subtasks run in PARALLEL on OpenRouter (one gateway,
    // different cheap models per kind so rate-limit + cost spread):
    //   - speed lookup/extract -> gpt-oss-20b paid ($0.02/M, fast)
    //   - draft/summarize      -> qwen 30B-class cheap
    //   - heavy research/code  -> GLM-5.3-Flash (beats Qwen 235B on benches)
    // Free :free quota is fallback inside mistralChat chains, not here.
    case "delegate_subtasks": return { ok: true, data: await runSubagents(args.subtasks || []) };
    case "wait_and_verify": {
      const delay = args.delay_ms || 500;
      await new Promise(r => setTimeout(r, delay));
      return { ok: true, data: { status: "ready", observation: "Wait complete. Proceed to read_screen or assert_text_visible to verify state." } };
    }
    case "assert_text_visible": {
      if (!agentState || !agentState.deviceState) return { ok: true, data: { asserted: false, text_searched: args.text, error: "No device state available. Call read_screen first." } };
      const text = String(args.text || "").toLowerCase();
      const screenText = String(agentState.deviceState?.screen_text || "").toLowerCase();
      const found = screenText.includes(text);
      return { ok: true, data: { asserted: found, text_searched: args.text, screen_snippet: screenText.slice(0, 200) } };
    }
    default: return { ok: false, error: "Not a server-side tool" };
  }
}

// ==================== SUBAGENT FAN-OUT ====================
// Primary delegates independent chunks here; they resolve concurrently and
// the merged results come back as ONE tool result (one agent turn, not N).
// GLM-5.3-Flash is heavy (DeepSWE 63.4, Toolathlon 78.4 — beats Qwen 235B).
const SUBAGENT_POOL = [
  { slot: "speed", model: "openrouter:openai/gpt-oss-20b", kinds: ["lookup", "extract", "classify", "quick"] },
  { slot: "draft", model: "openrouter:deepseek/deepseek-v4-flash", kinds: ["draft", "summarize", "rewrite", "plan"] },
  { slot: "heavy", model: "openrouter:z-ai/glm-5.3-flash", kinds: ["research", "compare", "analyze", "code"] },
];
function pickSubagentModel(kind = "") {
  const k = String(kind).toLowerCase();
  const hit = SUBAGENT_POOL.find(p => p.kinds.some(x => k.includes(x)));
  return (hit || SUBAGENT_POOL[0]).model;
}
// Per-kind timeout instead of one fixed 45s for everything: a quick
// lookup/classify that hangs 45s drags down the WHOLE Promise.all merge
// (the batch only returns once every slot has settled), while a genuine
// research/code subtask can legitimately need more than 45s and was getting
// killed as "FAILED: timeout" even when the provider would have come back
// with a real answer a few seconds later.
const SUBAGENT_TIMEOUT_MS = {
  lookup: 15000, extract: 15000, classify: 12000, quick: 12000,
  draft: 30000, summarize: 25000, rewrite: 25000, plan: 30000,
  research: 75000, compare: 60000, analyze: 60000, code: 90000,
};
const SUBAGENT_DEFAULT_TIMEOUT_MS = 30000;
function pickSubagentTimeout(kind = "") {
  const k = String(kind).toLowerCase();
  const hit = Object.keys(SUBAGENT_TIMEOUT_MS).find(x => k.includes(x));
  return hit ? SUBAGENT_TIMEOUT_MS[hit] : SUBAGENT_DEFAULT_TIMEOUT_MS;
}
async function runSubagents(subtasks = []) {
  const jobs = (Array.isArray(subtasks) ? subtasks : []).slice(0, 4).map((t, i) => {
    const task = typeof t === "string" ? t : (t.task || t.goal || "");
    const kind = typeof t === "string" ? "" : (t.kind || "");
    const model = pickSubagentModel(kind);
    const timeoutMs = pickSubagentTimeout(kind);
    const p = mistralChat({
      model,
      messages: [
        { role: "system", content: "You are a FRIT subagent. Do ONLY the subtask below. Reply with the result only — no preamble, no tool calls, under 400 words." },
        { role: "user", content: task },
      ],
      temperature: 0.3, max_tokens: 900,
    }).then(
      out => ({ index: i, model, ok: true, result: String(out?.choices?.[0]?.message?.content || "").slice(0, 2500) }),
      err => ({ index: i, model, ok: false, result: `FAILED: ${err.message}` }),
    );
    // Per-subtask timeout so one slow provider can't stall the merge.
    const timeout = new Promise(res => setTimeout(
      () => res({ index: i, model, ok: false, result: `FAILED: subtask timeout (${Math.round(timeoutMs / 1000)}s)` }), timeoutMs));
    return Promise.race([p, timeout]);
  });
  const settled = await Promise.all(jobs);
  settled.sort((a, b) => a.index - b.index);
  return { subtask_count: settled.length, results: settled };
}

// ==================== AUTOMATION SYSTEM PROMPT ====================
function buildUserProfileBlock(userProfile = null) {
  if (!userProfile || typeof userProfile !== "object") return "";
  const pick = (v) => String(v || "").trim();
  const lines = [];
  const callAs = pick(userProfile.preferred_name) || pick(userProfile.display_name);
  if (callAs) lines.push(`Address the user as: ${callAs.slice(0, 40)}`);
  if (pick(userProfile.display_name) && pick(userProfile.display_name) !== callAs) lines.push(`Full name: ${pick(userProfile.display_name).slice(0, 60)}`);
  if (pick(userProfile.language)) lines.push(`Language: ${pick(userProfile.language).slice(0, 30)}`);
  const loc = [pick(userProfile.city), pick(userProfile.country)].filter(Boolean).join(", ");
  if (loc) lines.push(`Location: ${loc.slice(0, 80)}`);
  if (pick(userProfile.occupation)) lines.push(`Occupation: ${pick(userProfile.occupation).slice(0, 80)}`);
  if (pick(userProfile.work_schedule)) lines.push(`Work schedule: ${pick(userProfile.work_schedule).slice(0, 80)}`);
  if (pick(userProfile.interests)) lines.push(`Interests: ${pick(userProfile.interests).slice(0, 160)}`);
  if (pick(userProfile.trading_risk)) lines.push(`Trading risk: ${pick(userProfile.trading_risk).slice(0, 20)}`);
  // Birthday / emergency contact intentionally NOT forwarded verbatim — the
  // phone resolves them locally; the brain only needs to know they exist.
  if (pick(userProfile.birthday)) lines.push("Birthday: on file (ask the user before using it anywhere)");
  if (pick(userProfile.emergency_contact)) lines.push("Emergency contact: on file on the phone (never ask the brain to dial it blindly — confirm first)");
  if (!lines.length) return "";
  return `User profile (Settings → Personal Details — baseline only, live chat tone always wins):\n${lines.join("\n")}`;
}

function buildAutomationSystemPrompt({ deviceState, memory, ledger = [], goal = "", tradeMemory = "", verification = null, lastFailure = "", userProfile = null }) {
  const deviceStateText = buildDeviceStateBlock(deviceState);
  const memoryText = buildMemoryBlock(summarizeMemory(memory, 6, 700));
  const profileText = buildUserProfileBlock(userProfile || deviceState?.user_profile);
  const ledgerText = Array.isArray(ledger) && ledger.length
    ? `\nTask ledger (tracked server-side — subtasks are marked done/failed automatically):\n${ledger.map(t => `- [${t.status}] ${t.description}${t.note ? ` — ${t.note}` : ""}`).join("\n")}`
    : "";
  const verifyText = verification
    ? `\nVERIFICATION OF LAST ACTION (from an independent verifier model): ${JSON.stringify(verification)}${verification.verified ? "" : " — if not verified, self-correct with a DIFFERENT approach."}`
    : "";
  const failureText = lastFailure
    ? `\nNOTE — a previous attempt failed: ${lastFailure}. If you are retrying, use a DIFFERENT approach.`
    : "";
  return [
    "You are FRIT, an autonomous Android AI agent: observe the screen, plan, act, verify. Be agentic and persistent — finish the task completely, however many steps it takes.",
    "UI rules: screens lag 300-800ms after taps; fuzzy-match labels (whitespace/renames); prefer tap_button over tap_coordinates; on blockage go_back, re-read, or screenshot — never hallucinate success.",
    "Loop: OBSERVE (read_screen) → ANALYZE → ACT (one step) → VERIFY (read_screen again, self-correct).",
    "",
    "HARD RULES — VIOLATING THESE IS A FAILURE:",
    "- Real function calls only — never narrate, describe, or paste fake tool calls/code blocks. If no tool fits, say so in one sentence.",
    "- Greetings/small talk: plain text, ZERO tool calls. But questions answerable by server tools (analysis, search, weather, prices) MUST call the tool immediately — never answer from memory.",
    "- open_app app_name = literal app name only ('WhatsApp'), then separate read/tap/type calls once open.",
    "- Settings toggles: execute_local_action first, then read_screen → tap → verify → return_to_frit. Never hand-navigate Settings menus.",
    "- Division of labor: the phone launches apps; you decide and tap/type inside them.",
    "- open_app targets must come from the Installed apps list — never guess.",
    "- Unfamiliar app UI: spend ONE search_web on the exact flow first (screen text wins on conflict); skip only for flows done this session.",
    "- Parallelize independent research via delegate_subtasks (max 4) instead of sequential turns.",
    "- Past feedback entries are binding: never repeat a recorded bad reply; reuse recorded good approaches.",
    "- Only listed tools exist — use exact tool names, never invented ones.",
    "",
    "TOOL CHOICE (cheapest correct tool wins):",
    "- Code/files/charts/math → run_code (server sandbox — never on the phone, never paste code for the user). Facts/news → search_web. Weather → get_weather. Markets → get_market_data/analyze_market. Phone UI → device tools.",
    "- MEDIA ROUTING (no exceptions): user asks for a picture/photo/logo/poster/flyer/design → generate_image (Seedream default) or design_image for graphic-design/text-heavy work — NEVER answer with a text description of pixels. User asks for video/clip/animation → generate_video. User shows a flat design to split → decompose_design. ALWAYS ground_scene first for realistic scenes.",
    "- Tool failed? Retry differently (tap_coordinates for tap_button, scroll then retry).",
    "- Market news: live get_market_news/search_web only — never memory, never old years.",
    "- Trading analysis happens HERE via a real analyze_market call — read direction/entry/SL/TP from its reply; the phone only EXECUTES orders after analysis.",
    "- NO ENGINE INTERNALS IN USER-FACING TEXT: the tool's `reasons`, `guards`, `antiX`, `structure_30m` and other diagnostic fields are for YOUR reasoning only. Never quote, list, summarise or leak them to the user, and never present a raw JSON/tool dump. Give the user the decision, the levels, the reasoning in plain language, and the risks that matter to them.",
    "- NO RAW ENGINE OUTPUT: even when the user asks about a market, answer as FRIT in normal prose (or a clean chat summary). The user should never see the engine speak for itself, its internal reason strings, indicator dumps, or a pasted tool payload.",
    "- TRADING NUMBERS DISCIPLINE: report the engine's decision/entry_zone/scenario/pullback_health fields VERBATIM. NEVER invent MACD, Bollinger, ADX/DI, or session values the tools did not return — if you want an indicator the engine lacks, compute it with run_code from real candles, never from memory. When decision is WAIT_PULLBACK, present ONLY the pullback-zone entry (limit-style); never substitute a market entry. When a healthy pullback exists, scenario_2 IS the trade — scenario_1 immediates apply only when the engine is aligned. When decision is WAIT_RANGE, present ONLY the range-edge fade (limit at the proven edge toward midline, invalidation on close beyond); never market-enter mid-range. Take-profit levels are structure-anchored and adapt to the current move (targets sit at real swing levels and scale with ATR) — always state the target's basis, and never repeat a stale target the engine has since moved. Every setup carries a costs block (net RR after spread/commission/slippage) — if costs are WEAK, say so and size down.",
    "",
    "PHONE UI SKILL — field-tested patterns for operating any app accurately (loaded from server/skills/phone-ui.md — edit that file to teach new patterns):",
    PHONE_UI_SKILL,
    "",
    loadAppFlows(goal),
    "",
    "Your Goal is to finish the user's task COMPLETELY. If it takes 10 steps, do 10 steps.",
    "",
    goal ? `\nOverall goal: ${truncateText(goal, 500)}` : "",
    ledgerText,
    verifyText,
    failureText,
    "",
    "Device state:",
    deviceStateText,
    tradeMemory ? `\n${tradeMemory}` : "",
    memoryText ? `\n${memoryText}` : "",
    profileText ? `\n${profileText}` : "",
    "",
    "PERSONALITY — DYNAMIC TONE MIRRORING (not a fixed persona):",
    "- The user profile above is only the BASELINE default. The live conversation / thread tone ALWAYS wins.",
    "- Fresh chat with no signal → warm neutral: friendly, plain, no slang, no stiffness.",
    "- Official signals (email, job/school, bank/opay dispute, complaint, MT5/trading instruction, 'Dear Sir/Ma', formal request) → act OFFICIAL: full sentences, polite structure, no slang, no emojis unless the user used one.",
    "- Casual signals (slang, pidgin, short chatty lines, jokes, friends/family chat) → mirror casually: short human replies, light warmth, ever so slight playfulness — never corporate.",
    "- When operating messaging apps on the user's behalf, mirror the THREAD's current tone the same way (neutral default; official only if the thread is formal). Never force the profile tone over what the thread shows.",
    "- Address the user by their profile name only when it sounds natural (greetings, confirmations) — not every turn.",
  ].filter(Boolean).join("\n");
}

function buildDeviceStateBlock(ds = {}) {
  const d = typeof ds === "string" ? { raw: ds } : ds || {};
  const parts = [];
  if (d.raw) parts.push(`Raw: ${truncateText(d.raw, 700)}`);
  if (d.current_app) parts.push(`Current app: ${d.current_app}`);
  if (d.current_activity) parts.push(`Activity: ${d.current_activity}`);
  if (d.screen_text) parts.push(`Screen text: ${truncateText(d.screen_text, 1200)}`);
  if (d.screen_summary) parts.push(`Screen summary: ${truncateText(d.screen_summary, 500)}`);
  if (d.network_status) parts.push(`Network: ${d.network_status}`);
  if (d.battery_level !== undefined && d.battery_level >= 0) parts.push(`Battery: ${d.battery_level}%`);
  if (d.keyboard_open !== undefined) parts.push(`Keyboard open: ${d.keyboard_open}`);
  if (d.screen_available === false || (d.screen_text === "" && d.has_screen_capture)) {
    parts.push(`NOTE: the accessibility layer returned NO readable screen text.${d.has_screen_capture ? " A screen capture IS available — call 'analyze_screenshot' to inspect the UI visually instead of read_screen." : " No screen capture either — open/press_home to an app that exposes text, or tell the user the phone screen has no readable content."}`);
  }
  if (d.screen_note) parts.push(`Screen note: ${truncateText(d.screen_note, 400)}`);
  if (d.user_profile && typeof d.user_profile === "object") {
    const up = d.user_profile;
    const nm = String(up.preferred_name || up.display_name || "").trim();
    const lg = String(up.language || "").trim();
    const lc = [String(up.city || "").trim(), String(up.country || "").trim()].filter(Boolean).join(", ");
    const bits = [];
    if (nm) bits.push(`user: ${nm.slice(0, 40)}`);
    if (lg) bits.push(`lang: ${lg.slice(0, 20)}`);
    if (lc) bits.push(`loc: ${lc.slice(0, 60)}`);
    if (String(up.occupation || "").trim()) bits.push(`job: ${String(up.occupation).trim().slice(0, 60)}`);
    if (bits.length) parts.push(`Profile: ${bits.join(" · ")}`);
  }
  if (Array.isArray(d.installed_apps) && d.installed_apps.length) {
    // Ground open_app in reality: only these names are guaranteed to exist.
    // Capped to keep prompt size sane on large phones (150+ apps is common).
    const names = d.installed_apps.map(a => (typeof a === "string" ? a : (a.label || a.name || a.package))).filter(Boolean);
    parts.push(`Installed apps (${names.length}, use EXACT names with open_app): ${names.slice(0, 200).join(", ")}${names.length > 200 ? ", ..." : ""}`);
  }
  return parts.length ? parts.join("\n") : "No device state provided.";
}

// Fuzzy-match a requested app name against the real installed_apps list from
// device state, so open_app gets corrected before hitting the device instead
// of failing with "not found" (or worse, launching the wrong app).
function resolveInstalledApp(requestedName, deviceState) {
  const list = Array.isArray(deviceState?.installed_apps) ? deviceState.installed_apps : [];
  if (!list.length || !requestedName) return null;
  const target = requestedName.trim().toLowerCase();

  for (const item of list) {
    if (typeof item === "string") {
      if (item.toLowerCase() === target || item.toLowerCase().includes(target)) return item;
    } else if (item && typeof item === "object") {
      const label = (item.label || item.name || "").toLowerCase();
      const pkg = (item.package || "").toLowerCase();
      if (label === target || pkg === target || label.includes(target) || target.includes(label)) {
        return item.label || item.name || item.package;
      }
    }
  }
  return null;
}

// ==================== SCREEN FRAME INGESTION ====================
// Per-user buffers (10 each) — a global buffer leaked User A's screenshots
// into User B's vision calls.
const frameBuffers = new Map();
const FRAME_BUFFER_SIZE = 10;
function userFrames(uid) {
  let b = frameBuffers.get(uid);
  if (!b) { b = []; frameBuffers.set(uid, b); }
  return b;
}

// ==================== ROUTES ====================
app.get("/", (_req, res) => {
  res.json({
    name: "FRIT engine.js Trading Engine & Mistral AI Orchestrator",
    description: "Single trading engine (server/src/strategy/engine.js). Single Mistral API ecosystem.",
    status: "online",
    endpoints: {
      core: ["/health", "/agent/start", "/agent/resume", "/agent/status"],
      market: ["/market/quote", "/market/batch", "/market/analyze", "/trade"],
      positions: ["/positions", "/positions/status", "/positions/resolve"],
      strategy: ["/enhanced/analyze", "/market/strategy", "/strategy/status", "/systems/status", "/strategy/costs", "/strategy/journal/stats", "/strategy/journal/recent", "/strategy/journal/resolve"],
      risk: ["/risk/status", "/risk/check"],
      memory: ["/memory/trade"],
      screen: ["/screen/frame", "/screen/analyze-frame", "/screen/status"],
      transcribe: ["/transcribe"],
      tools: ["/tools/search"],
      images: ["/images/generate", "/images/decompose"],
      video: ["/videos/submit", "/videos/status/:jobId", "/videos/content/:jobId"],
      creative: ["/creative/compose", "/creative/revise", "/creative/assemble", "/creative/ground", "/creative/jobs"],
      agent: ["/agent/start", "/agent/resume", "/agent/status", "/agent/chat/stream"],
      creativeSuite: ["/creative-suite/images/generate", "/creative-suite/images/edit", "/creative-suite/images/remove-background", "/creative-suite/images/retouch", "/creative-suite/images/style-transfer", "/creative-suite/images/upscale", "/creative-suite/images/generative-fill", "/creative-suite/images/batch", "/creative-suite/images/batch/:id/status", "/creative-suite/videos/generate", "/creative-suite/videos/edit", "/creative-suite/videos/subtitles", "/creative-suite/videos/translate-subtitles", "/creative-suite/storyboard", "/creative-suite/templates", "/creative-suite/style-presets", "/creative-suite/prompt-suggest", "/creative-suite/prompt-enhance", "/creative-suite/recommend", "/creative-suite/projects", "/creative-suite/export", "/creative-suite/cinema/options", "/creative-suite/cinema/direct", "/creative-suite/cinema/generate", "/creative-suite/katana", "/creative-suite/genjutsu", "/creative-suite/elements", "/creative-suite/influencer", "/creative-suite/ads", "/creative-suite/lipsync", "/creative-suite/lipsync/:jobId/status", "/creative-suite/videos/models", "/creative-suite/cinema/recipes", "/creative-suite/autopilot", "/creative-suite/brandkit", "/creative-suite/design", "/creative-suite/design/templates", "/creative-suite/design/agent", "/creative-suite/refs/:id", "/creative-suite/director", "/webhooks/lipsync"],
      billing: ["/billing/tiers", "/billing/pay-info", "/billing/subscribe", "/billing/submit-reference", "/billing/pending", "/billing/approve", "/billing/mint", "/billing/redeem", "/billing/status"],
      wallet: ["/wallet/balance", "/wallet/topup", "/wallet/spend", "/wallet/withdraw", "/wallet/withdrawals"],
      admin: ["/admin/inbox", "/admin/keys", "/admin/snapshot", "/admin/ledger/verify", "/admin/backup", "/admin/withdrawals/approve"],
      models: ["/models/free-health"],
    utility: ["/weather"],
    sandbox: ["/sandbox/run"],
  },
});
});

app.get("/health", (_req, res) => {
  res.json({
    status: "active",
    models: MODELS,
    openrouter: { primary: OR_PRIMARY, fast: OR_FAST, draft: OR_DRAFT, image: OR_IMAGE_MODEL, stt: OR_STT_MODEL, video: OR_VIDEO_MODEL, avatar: OR_AVATAR_MODEL, free_fast: OR_FREE_FAST },
    twelve_data: !!TWELVE_DATA_KEY,
    mt5_bridge: !!MT5_BRIDGE_URL,
    sandbox_url: SANDBOX_URL,
    sandbox_auth: !!SANDBOX_AUTH,
    frame_buffers: frameBuffers.size,
    cache_entries: _cache.size,
    uptime: Math.floor(process.uptime()) + "s",
  });
});


app.post("/screen/frame", requireAuth, limitNormal, (req, res) => {
  const { frameData, timestamp, width, height, current_app, screen_text, current_activity } = req.body || {};
  if (!frameData) return res.status(400).json({ error: "frameData required" });
  const buf = userFrames(boundUser(req));
  buf.push({ data: frameData, timestamp: timestamp || Date.now(), width, height });
  if (buf.length > FRAME_BUFFER_SIZE) buf.shift();
  res.json({ status: "received", buffered: buf.length });
});

app.post("/screen/analyze-frame", requireAuth, limitCostly, async (req, res) => {
  const { prompt, frameIndex = -1, frameData } = req.body || {};
  // Inline image support: clients that attach an image (photo picker, camera,
  // arbitrary screenshot) pass frameData directly instead of pre-buffering it.
  if (frameData) {
    try {
      const out = await chatWithFallback("vision", {
        messages: [{
          role: "user",
          content: [
            { type: "text", text: prompt || "Describe the UI elements and any actionable items on this screen." },
            { type: "image_url", image_url: { url: `data:image/jpeg;base64,${frameData}` } },
          ],
        }],
        max_tokens: 800,
      });
      return res.json({ analysis: out.choices[0].message.content, source: "inline" });
    } catch (err) {
      return res.status(500).json({ error: "Inline frame analysis failed", details: err.message });
    }
  }
  const buf = userFrames(boundUser(req));
  if (!buf.length) return res.status(400).json({ error: "No frames in buffer. Android app must send frames first via /screen/frame." });
  const frame = frameIndex >= 0 && frameIndex < buf.length ? buf[frameIndex] : buf.at(-1);
  try {
    const out = await chatWithFallback("vision", {
      messages: [{
        role: "user",
        content: [
          { type: "text", text: prompt || "Describe the UI elements and any actionable items on this screen." },
          { type: "image_url", image_url: { url: `data:image/jpeg;base64,${frame.data}` } },
        ],
      }],
      max_tokens: 500,
    });
    res.json({
      analysis: out.choices[0].message.content,
      frameTimestamp: frame.timestamp,
      frameIndex: frameIndex >= 0 ? frameIndex : buf.length - 1,
    });
  } catch (err) {
    res.status(500).json({ error: "Frame analysis failed", details: err.message });
  }
});

app.get("/screen/status", requireAuth, (req, res) => {
  const buf = userFrames(boundUser(req));
  res.json({
    buffered_frames: buf.length,
    max_buffer: FRAME_BUFFER_SIZE,
    oldest_frame_ts: buf[0]?.timestamp || null,
    newest_frame_ts: buf.at(-1)?.timestamp || null,
  });
});

app.post("/screen/clear", requireAuth, (req, res) => {
  userFrames(boundUser(req)).length = 0;
  res.json({ status: "cleared" });
});

// ==================== MARKET ROUTES ====================
app.get("/market/quote", requireAuth, limitNormal, async (req, res) => {
  try {
    const symbol = String(req.query.symbol || "BTCUSD").toUpperCase();
    const data = await fetchMarketPrices([symbol]);
    res.json(data[symbol] || { error: "Not found" });
  } catch (err) {
    res.status(500).json({ error: "Quote fetch failed", details: err.message });
  }
});

app.post("/market/batch", requireAuth, limitNormal, async (req, res) => {
  try {
    const { symbols = [] } = req.body || {};
    if (!Array.isArray(symbols) || !symbols.length) return res.status(400).json({ error: "symbols array required" });
    if (symbols.length > 20) return res.status(400).json({ error: "max 20 symbols per call" });
    res.json(await fetchMarketPrices(symbols.slice(0, 20).map(s => String(s).toUpperCase().slice(0, 12))));
  } catch (err) {
    res.status(500).json({ error: "Batch fetch failed", details: err.message });
  }
});

// Compatibility shim: /market/analyze legacy contract, backed ONLY by
// MTFStrategyEngine (server/src/strategy/engine.js) — the single trading engine.
async function analyzeSymbol(symbol, interval = "1h", customSize = null) {
  const r = await mtfStrategy.analyze(String(symbol || "XAUUSD").toUpperCase(), { interval });
  const dir = r.decision === "BUY" ? "BULLISH" : r.decision === "SELL" ? "BEARISH" : "NEUTRAL";
  const strength = r.confidence >= 75 ? "STRONG" : r.confidence >= 60 ? "MODERATE" : "WEAK";
  return {
    symbol: String(symbol || "XAUUSD").toUpperCase(),
    price: parseFloat(r.price) || 0,
    direction: dir,
    strength,
    interval,
    confidence: r.confidence,
    trade_plan: {
      entry_zone: r.entry ?? null,
      invalidation: r.sl ?? null,
      tp1: r.tp ?? null,
      tp2: r.tp2 ?? null,
      risk_state: dir === "NEUTRAL" ? "no_trade" : "acceptable",
    },
    concise_signal: {
      direction: dir,
      entry: r.entry || "N/A",
      sl: r.sl || "N/A",
      tp: r.tp || "N/A",
      ai_opinion: r.reason || "",
    },
    news_filter: { blocked: false },
    mtf: { trend: r.regime?.macro_trend || "unknown" },
    engine_decision: r.decision,
    scenarios: r.scenarios ?? null,
    source: "engine-shim",
  };
}

app.all("/market/analyze", requireAuth, async (req, res) => {
  try {
    const body = req.body || {};
    const query = req.query || {};
    const sym = req.method === "GET" ? query.symbol : body.symbol ?? query.symbol;
    if (!sym) return res.status(400).json({ error: "symbol required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "trading");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const symbol = String(sym).toUpperCase();
    const interval = normalizeInterval(req.method === "GET" ? query.interval : body.interval ?? query.interval ?? "1h");
    const outputsize = (req.method === "GET" ? query.outputsize : body.outputsize ?? query.outputsize)
      ? Number(req.method === "GET" ? query.outputsize : body.outputsize ?? query.outputsize)
      : null;
    const out = await analyzeSymbol(symbol, interval, outputsize);
    debitCredits(uid, "trade_analysis", "market analyze"); // post-success
    res.json(out);
  } catch (err) {
    console.error("/market/analyze", err.message);
    res.status(500).json({ error: "Analysis failed", details: err.message });
  }
});

// ==================== TRADE ENDPOINT (engine.js only) ====================
// MTFStrategyEngine (server/src/strategy/engine.js) is the single trading
// engine. /trade always runs through it — no legacy/GSRI/ACP branches.
//
// The engine's internal `reasons` (its own reasoning strings) are NOT part of
// the user payload: they read as the engine talking over the user's head.
// They stay in server logs and are available to the creator console through
// /admin/strategy/diagnostics. Risk-layer refusals use `risk_reasons` — those
// describe the user's own account and ARE shown.
function withoutEngineReasons(result) {
  if (!result || typeof result !== "object") return result;
  const { reasons, ...rest } = result;
  return rest;
}

app.post("/trade", requireAuth, limitNormal, async (req, res) => {
  const { symbol, risk_percent = 1, balance, reason = "", interval = "1h" } = req.body || {};
  if (!symbol) return res.status(400).json({ error: "symbol required" });
  if (balance != null && (!Number.isFinite(Number(balance)) || Number(balance) > 1_000_000)) {
    return res.status(400).json({ error: "balance must be ≤ 1,000,000" });
  }
  const uid = boundUser(req);
  const ent = checkEntitlement(uid, "trading");
  if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });

  try {
    // Deriv synthetics run the generator-aware engine, never the MTF trend
    // engine — its output shape matches what the flow below consumes.
    let result;
    if (isDerivSymbol(symbol)) {
      const candles = await fetchCandles(symbol, "1min", 200);
      const closes = (candles || []).map((c) => c.close).filter((c) => c > 0);
      const s = analyzeSynth(symbol, closes);
      result = {
        decision: s.decision, confidence: s.confidence || 0,
        entry: s.entry, sl: s.sl, tp: s.tp, tp2: null, rr: s.rr,
        reasons: s.reasons, regime: { macro_trend: `SYNTH_${s.family}` },
        structure_30m: null, entry_ctx: { type: "synth", zone: null },
        costs: null, guards: null, lot_size: null,
        synth: { family: s.family, maxHoldMin: s.maxHoldMin ?? null, details: s.details ?? null },
      };
    } else {
      result = await mtfStrategy.run(symbol, { interval, balance: balance || 1000, riskPercent: risk_percent });
    }
    console.log(`[/trade] ${symbol} ${result.decision} — reasons: ${JSON.stringify(result.reasons || [])}`);
    debitCredits(uid, "trade_analysis", "trade engine"); // engine ran — post-success

    if (["NO_TRADE", "WAIT", "WAIT_PULLBACK", "WAIT_RANGE", "COOLDOWN", "DATA_UNAVAILABLE", "DATA_RATE_LIMITED", "ERROR"].includes(result.decision)) {
      return res.status(200).json({ status: "blocked", ...withoutEngineReasons(result) });
    }
    // Non-MT5 rails share the venue executor (own sizing, brackets, gates).
    if (marketOf(symbol) !== "mt5") {
      const routed = await executeTradeSignal({
        symbol, action: result.decision === "BUY" ? "buy" : "sell",
        lotSize: result.lot_size, entry: result.entry, sl: result.sl, tp: result.tp,
        reason: reason || `engine.js: conf=${result.confidence}%`,
        source: "trade", balance: balance || 1000, riskPercent: risk_percent,
      });
      return res.status(200).json({ status: routed.status || "submitted", pipeline: "engine_js", rail: marketOf(symbol), ...withoutEngineReasons(result), execution: routed });
    }
    // Portfolio risk layer: account-level guards BEFORE anything reaches the bridge.
    const gate = riskGate({
      symbol: symbol.toUpperCase(),
      action: result.decision === "BUY" ? "buy" : "sell",
      lotSize: result.lot_size,
      entry: result.entry,
      sl: result.sl,
      balance: balance || 1000,
    });
    if (!gate.allowed) {
      return res.status(200).json({
        status: "blocked", blocked_by: "risk", risk_reasons: gate.reasons,
        exposure: gate.exposure, ...withoutEngineReasons(result),
      });
    }
    const tradeResult = await sendToMT5Bridge({
      symbol: symbol.toUpperCase(),
      action: result.decision === "BUY" ? "buy" : "sell",
      lotSize: result.lot_size,
      entry: result.entry,
      sl: result.sl,
      tp: result.tp,
      reason: reason || `engine.js: conf=${result.confidence}% regime=${result.regime?.macro_trend} zone=${result.entry_ctx?.zone}`,
    });

    // Additive: watch this position until TP/SL is hit (paper-first by design).
    let position = null;
    if (tradeResult.status !== "failed" && result.sl && result.tp) {
      try {
        position = positionMonitor.register({
          symbol: symbol.toUpperCase(),
          action: result.decision === "BUY" ? "buy" : "sell",
          lotSize: result.lot_size,
          entry: result.entry,
          sl: result.sl,
          tp: result.tp,
          source: "engine_js",
          reason: reason || `conf=${result.confidence}%`,
        });
      } catch (e) { console.warn("[PositionMonitor] /trade register failed:", e.message); }
    }

    return res.json({
      status: "submitted",
      pipeline: "engine_js",
      symbol: symbol.toUpperCase(),
      action: result.decision === "BUY" ? "buy" : "sell",
      lotSize: result.lot_size,
      entry: result.entry,
      sl: result.sl,
      tp: result.tp,
      tp2: result.tp2,
      tp_basis: result.tp_basis ?? null,
      rr: result.rr,
      confidence: result.confidence,
      regime: result.regime,
      structure: result.structure_30m,
      entry_ctx: result.entry_ctx,
      guards: result.guards,
      costs: result.costs ?? null,
      costs_usd: (() => {
        try {
          const cfg = PIP_CONFIG[symbol.toUpperCase()];
          return costToUsd({ symbol: symbol.toUpperCase(), lotSize: result.lot_size, pipSize: cfg?.pipSize, pipValue: cfg?.pipValue, refPrice: result.entry });
        } catch { return null; }
      })(),
      risk: { allowed: true, exposure: gate.exposure },
      mt5_result: tradeResult,
      position_id: position?.id ?? null,
    });
  } catch (err) {
    console.error("[/trade]", err.message);
    return res.status(500).json({ error: "Trade failed", details: err.message });
  }
});

// ==================== POSITIONS (PositionMonitor) ====================
// Read-only + manual-resolve views over the additive position state machine.
app.get("/positions", requireAuth, (_req, res) => {
  res.json({ positions: positionMonitor.list(100) });
});

app.get("/positions/status", requireAuth, (_req, res) => {
  res.json(positionMonitor.status());
});

app.post("/positions/resolve", requireAuth, (req, res) => {
  const { id, outcome, close_price, note } = req.body || {};
  if (!id) return res.status(400).json({ error: "id required" });
  const result = positionMonitor.resolve(id, outcome, close_price ?? null, note || "");
  res.json(result.ok ? result : { ok: false, error: result.error });
});

app.post("/memory/trade", requireAuth, (req, res) => {
  const { symbol, outcome, pattern, direction, note } = req.body || {};
  if (!symbol) return res.status(400).json({ error: "symbol required" });
  addTradeMemory(symbol, { outcome, pattern, direction, note });
  res.json({ ok: true, memory_count: tradeMemory.get(symbol.toUpperCase())?.length });
});

app.get("/memory/trade", requireAuth, (req, res) => {
  const symbol = String(req.query.symbol || "").toUpperCase();
  if (!symbol) return res.status(400).json({ error: "symbol required" });
  res.json({ symbol, entries: getTradeMemory(symbol) });
});

// ==================== PULLBACK JOURNAL (setup recorder + accuracy) ======
// Every published setup is auto-recorded by the engine; resolve replays 30M
// candles and scores win/loss/expired. Stats carry the only verdict that
// matters: expectancy in R over resolved setups (30-trade minimum).
app.get("/strategy/journal/stats", requireAuth, (req, res) => {
  try {
    res.json(pullbackJournal.stats({
      symbol: req.query.symbol || null,
      strategy: req.query.strategy || null,
    }));
  } catch (err) {
    res.status(500).json({ error: "Journal stats failed", details: err.message });
  }
});

app.get("/strategy/journal/recent", requireAuth, (req, res) => {
  try {
    res.json({ setups: pullbackJournal.recent(Number(req.query.limit) || 20) });
  } catch (err) {
    res.status(500).json({ error: "Journal recent failed", details: err.message });
  }
});

app.post("/strategy/journal/resolve", requireAuth, async (req, res) => {
  try {
    res.json(await pullbackJournal.resolvePending({ fetchCandles }));
  } catch (err) {
    res.status(500).json({ error: "Journal resolve failed", details: err.message });
  }
});

// ==================== COSTS (retail friction lookup) =====================
app.get("/strategy/costs", requireAuth, (req, res) => {
  const symbol = String(req.query.symbol || "").toUpperCase();
  if (!symbol) return res.status(400).json({ error: "symbol required" });
  const cfg = PIP_CONFIG[symbol];
  res.json({
    symbol,
    ...getSymbolCost(symbol),
    per_lot_usd: costToUsd({ symbol, lotSize: 1, pipSize: cfg?.pipSize, pipValue: cfg?.pipValue }),
  });
});

// ==================== RISK (portfolio guards) ============================
app.get("/risk/status", requireAuth, (req, res) => {
  try {
    res.json(portfolioRisk.status(Number(req.query.balance) || 1000));
  } catch (err) {
    res.status(500).json({ error: "Risk status failed", details: err.message });
  }
});

app.post("/risk/check", requireAuth, (req, res) => {
  const { symbol, action, lotSize, lot_size, entry, sl, balance } = req.body || {};
  if (!symbol || entry == null || sl == null) {
    return res.status(400).json({ error: "symbol, entry and sl required" });
  }
  try {
    res.json(riskGate({
      symbol, action: action || "buy",
      lotSize: lotSize ?? lot_size ?? 0.01, entry, sl, balance,
    }));
  } catch (err) {
    res.status(500).json({ error: "Risk check failed", details: err.message });
  }
});

app.get("/weather", requireAuth, limitNormal, async (req, res) => {
  try {
    const city = String(req.query.city || "Lagos");
    const result = await getWeather(city);
    res.json({ city, result });
  } catch (err) {
    res.status(500).json({ error: "Weather failed", details: err.message });
  }
});

app.post("/transcribe", requireAuth, limitCostly, async (req, res) => {
  try {
    const { audio_base64, mime_type = "audio/webm" } = req.body || {};
    if (!audio_base64) return res.status(400).json({ error: "audio_base64 required" });
    if (typeof audio_base64 !== "string" || audio_base64.length > 20_000_000) return res.status(413).json({ error: "audio too large (max ~15MB)" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "chat");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const text = await mistralTranscribe(audio_base64, mime_type);
    if (!text) return res.status(500).json({ error: "Transcription failed" });
    db.prepare("UPDATE usage_daily SET stt_mins = stt_mins + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    debitCredits(uid, "stt_min", "transcribe"); // post-success: ~1 min voice at 2cr
    res.json({ text, model_used: OR_STT_MODEL });
  } catch (err) {
    console.error("[/transcribe]", err.message);
    res.status(500).json({ error: "Transcription failed", details: err.message });
  }
});

app.post("/images/generate", requireAuth, limitCostly, async (req, res) => {
  try {
    const { prompt, aspect_ratio, n, model } = req.body || {};
    if (!prompt) return res.status(400).json({ error: "prompt required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const data = await generateImage(prompt, { aspect_ratio, n, model });
    db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    debitCredits(uid, "image", "image generate"); // post-success only
    res.json({ ok: true, model: model || OR_IMAGE_MODEL, ...data });
  } catch (err) {
    console.error("[/images/generate]", err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});
app.post("/images/decompose", requireAuth, limitCostly, async (req, res) => {
  try {
    const { image, prompt, layer_plan } = req.body || {};
    if (!image) return res.status(400).json({ error: "image (base64 data URL or http URL) required" });
    const data = await decomposeDesign(image, prompt || layer_plan || "decompose into background and foreground layers");
    res.json({ ok: true, model: OR_LAYER_MODEL, ...data });
  } catch (err) {
    console.error("[/images/decompose]", err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Free-tier health: which :free models answer right now? Free pool rotates and
// 429s often — this tells you in 10s whether free is worth using today.
app.get("/models/free-health", requireAuth, async (_req, res) => {
  const candidates = [OR_FREE_FAST, OR_FREE_DRAFT, "openrouter/free"];
  const results = [];
  for (const m of candidates) {
    const t0 = Date.now();
    try {
      const out = await mistralChat({ model: `openrouter:${m.replace(/^openrouter:/, "")}`, messages: [{ role: "user", content: "ping" }], max_tokens: 5 });
      results.push({ model: m, ok: true, ms: Date.now() - t0, sample: String(out?.choices?.[0]?.message?.content || "").slice(0, 50) });
    } catch (e) {
      results.push({ model: m, ok: false, ms: Date.now() - t0, error: e.message.slice(0, 120) });
    }
  }
  res.json({ quota_note: "free = 50/day default, 1000/day after $10 credits. 20 RPM. Spare capacity: expect 429s.", results });
});

app.post("/sandbox/run", requireAuth, limitCostly, async (req, res) => {
  try {
    const result = await runSandbox(req.body);
    res.json(result);
  } catch (err) {
    console.error("[/sandbox/run]", err.message);
    res.status(500).json({ error: "Sandbox execution failed", details: err.message });
  }
});

// ==================== VIDEO + CREATIVE ====================
// Async: submit returns job id immediately (video takes minutes). Client polls
// /videos/status, then GETs /videos/content for the downloadable MP4.
// Revise = submit again with edited prompt (same model/endpoint continues).
app.post("/videos/submit", requireAuth, limitCostly, async (req, res) => {
  try {
    const { prompt, duration, resolution, aspect_ratio, model } = req.body || {};
    if (!prompt) return res.status(400).json({ error: "prompt required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "video");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const job = await submitVideo(prompt, { duration, resolution, aspect_ratio, model });
    const id = `vid_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt) VALUES (?, ?, 'video', 'submitted', ?)").run(id, uid, String(prompt).slice(0, 1000));
    db.prepare("UPDATE usage_daily SET clips = clips + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    trackProviderJob(job?.id || job?.data?.id, uid, "clip"); // charged once on completion, never on failure
    res.json({ ok: true, job_id: id, provider_job: job });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});
app.get("/videos/status/:jobId", requireAuth, limitNormal, async (req, res) => {
  try {
    const data = await pollVideo(req.params.jobId);
    const st = data?.data?.status || data?.status;
    if (st === "failed" || st === "cancelled" || st === "expired") {
      return res.status(500).json({ ok: false, status: st, error: data?.data?.error || data?.error || `video job ${st} — resubmit (failed generations are never billed)` });
    }
    if (st === "completed") chargeProviderJob(req.params.jobId); // completion = provider bills us = we bill once
    res.json({ ok: true, ...data });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});
app.get("/videos/content/:jobId", requireAuth, limitNormal, async (req, res) => {
  try {
    const jobId = String(req.params.jobId || "").replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 64);
    if (!jobId) return res.status(400).json({ error: "bad job id" });
    // Never hand back an empty/partial file: confirm completion first.
    const poll = await pollVideo(jobId).catch(() => null);
    const st = poll?.data?.status || poll?.status;
    if (poll && st && st !== "completed") {
      return res.status(409).json({ ok: false, status: st, error: `video not ready (status: ${st}) — poll /videos/status until completed` });
    }
    const buf = await downloadVideo(jobId, Number(req.query.index || 0));
    if (!buf?.length) return res.status(500).json({ ok: false, error: "video download came back empty — retry shortly" });
    chargeProviderJob(jobId); // completion billing (no-op if already charged via status)
    res.setHeader("Content-Type", "video/mp4");
    res.setHeader("Content-Disposition", `attachment; filename="${jobId}.mp4"`);
    res.send(buf);
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});
// Creative compose: brief/transcript (+ optional frame notes) -> script + shots.
// Client uploads video bytes; Android extracts audio/frames and sends text here.
// Final stitching happens in /creative/assemble (sandbox ffmpeg).
app.post("/creative/compose", requireAuth, limitCostly, async (req, res) => {
  try {
    const { brief, transcript, frame_notes } = req.body || {};
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "creative");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const input = String(transcript || brief || "");
    if (!input) return res.status(400).json({ error: "transcript or brief required" });
    const ground = await groundScene(input); // before-math: real sizes + camera first
    const out = await chatWithFallback("conversation", {
      messages: [
        { role: "system", content: "You are FRIT Creative. Given a video transcript/brief AND its physical grounding spec, output JSON {script, shots:[{kind:'image'|'video', prompt, duration}], caption, corrections_invite}. Each shot prompt MUST embed the grounding proportions and negative prompt. Shots max 5." },
        { role: "user", content: `Brief: ${input.slice(0, 6000)}${frame_notes ? `\nFrames: ${String(frame_notes).slice(0, 2000)}` : ""}\nGrounding: ${JSON.stringify(ground).slice(0, 2000)}` },
      ],
      max_tokens: 1800,
    });
    const text = out.choices[0].message.content || "";
    const id = `cc_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt, result_url) VALUES (?, ?, 'compose', 'done', ?, ?)").run(id, uid, input.slice(0, 1000), text.slice(0, 4000));
    debitCredits(uid, "creative", "creative compose");
    res.json({ ok: true, job_id: id, draft: text, note: "Reply with corrections to /creative/revise with job_id." });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});
app.post("/creative/revise", requireAuth, limitCostly, async (req, res) => {
  try {
    const { job_id, corrections } = req.body || {};
    if (!job_id || !corrections) return res.status(400).json({ error: "job_id and corrections required" });
    const prev = db.prepare("SELECT * FROM creative_jobs WHERE id = ?").get(job_id);
    if (!prev) return res.status(404).json({ error: "job not found" });
    if (req.authUser && prev.user_id && prev.user_id !== req.authUser) {
      return res.status(403).json({ error: "not your job" });
    }
    const out = await chatWithFallback("conversation", {
      messages: [
        { role: "system", content: "You are FRIT Creative. Revise the prior draft per user corrections. Output updated JSON {script, shots, caption}." },
        { role: "user", content: `Prior: ${(prev.result_url || prev.prompt || "").slice(0, 4000)}\nCorrections: ${String(corrections).slice(0, 2000)}` },
      ],
      max_tokens: 1800,
    });
    const text = out.choices[0].message.content || "";
    db.prepare("UPDATE creative_jobs SET status = 'revised', result_url = ? WHERE id = ?").run(text.slice(0, 4000), job_id);
    res.json({ ok: true, job_id, draft: text });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});
// Creative assemble: clips (OpenRouter jobIds/urls) and/or images -> one MP4.
// Proxies to sandbox /media/assemble (the only place ffmpeg runs). Returns MP4 download.
app.post("/creative/assemble", requireAuth, limitCostly, async (req, res) => {
  try {
    const { clips = [], images = [], per_image_secs = 2 } = req.body || {};
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "creative");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    if ((!Array.isArray(clips) || !clips.length) && (!Array.isArray(images) || !images.length)) {
      return res.status(400).json({ error: "clips[] (jobIds/urls) or images[] (base64) required" });
    }
    const files = [];
    for (const c of (Array.isArray(clips) ? clips : []).slice(0, 4)) {
      const s = String(c);
      let buf;
      if (/^https?:\/\//i.test(s)) {
        const r = await fetch(s, { signal: AbortSignal.timeout(120_000) });
        if (!r.ok) throw new Error(`clip fetch HTTP ${r.status}`);
        buf = Buffer.from(await r.arrayBuffer());
      } else {
        buf = await downloadVideo(s, 0); // OpenRouter video jobId
      }
      if (buf.length > 8 * 1024 * 1024) throw new Error("clip >8MB, keep 720p 4-8s");
      files.push({ base64: buf.toString("base64") });
    }
    for (const b64 of (Array.isArray(images) ? images : []).slice(0, 6 - files.length)) {
      if (String(b64 || "").length > 11 * 1024 * 1024) throw new Error("image >8MB");
      files.push({ base64: String(b64) });
    }
    const op = clips.length ? "concat" : "slideshow";
    const r = await fetch(`${SANDBOX_URL}/media/assemble`, {
      method: "POST",
      headers: sandboxAuthHeaders(),
      body: JSON.stringify({ op, files, per_image_secs }),
      signal: AbortSignal.timeout(120_000),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || !data?.base64) throw new Error(data?.error || `assemble HTTP ${r.status}`);
    const id = `asm_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt) VALUES (?, ?, 'assemble', 'done', ?)").run(id, uid, `op=${op} files=${files.length}`);
    res.json({ ok: true, job_id: id, mime: "video/mp4", size: data.size, base64: data.base64 });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ==================== WALLET + SUBSCRIPTIONS ====================
// Two rails, zero API keys:
// A) Bank transfer: subscribe -> PENDING order + auto passcode -> user pays with
//    passcode as narration -> "I've paid" + passcode -> you approve -> tier+expiry.
// B) Crypto on BNB Smart Chain (your Binance deposits): USDT or BTC to ONE
//    BSC_ADDRESS -> user submits txhash -> auto-verified keyless via BSC public
//    RPC (>=12 confirmations, txhash single-use). User-facing name is always
//    "BTC" (their Binance shows BTC); internally it's the BTCB contract.
//    Native BTC (bc1…) also works via mempool.space IF you later add a native
//    BTC_ADDRESS (network "btc-native").
// Upgrade path later: BTCPay Server (self-hosted, no per-tx key) or
// Paystack/Flutterwave webhooks — same tables.
app.get("/billing/tiers", requireAuth, async (_req, res) => {
  const rate = await ngnPerUsd();
  const tiers = Object.fromEntries(Object.entries(SUB_TIERS).map(([k, t]) => [k, {
    ...t,
    price_kobo: tierNaira(k, rate),
    price_kobo_live: tierNaira(k, rate),
    yearly_kobo: tierNaira(k, rate) * YEARLY_MULT,
    features: t.features || [],
    credits_mo: t.credits_mo || 0,
  }]));
  res.json({
    ok: true, usd_ngn: rate, peg_floor: PEG_FLOOR, tiers, overage: OVERAGE,
    yearly_mult: YEARLY_MULT, order: TIER_ORDER,
    credit_costs: CREDIT_COSTS,
    credits_per_naira: CREDIT_NAIRA_KOBO,
    feature_labels: FEATURE_LABELS,
  });
});
app.get("/billing/pay-info", requireAuth, async (_req, res) => {
  const rate = await ngnPerUsd();
  const n = (kobo) => Math.round(kobo / 100);
  const sK = tierNaira("starter", rate), cK = tierNaira("creator", rate), pK = tierNaira("pro", rate), tK = tierNaira("trader", rate);
  res.json({
    ok: true, usd_ngn: rate, peg_floor: PEG_FLOOR,
    bank: { bank_name: PAY_BANK_NAME || "(set PAY_BANK_NAME)", account_number: PAY_ACCOUNT_NUMBER || "(set PAY_ACCOUNT_NUMBER)", account_name: PAY_ACCOUNT_NAME || "(set PAY_ACCOUNT_NAME)" },
    crypto: {
      bsc_address: BSC_ADDRESS || "(set BSC_ADDRESS)",
      assets: ["USDT (BNB Smart Chain)", "BTC (BNB Smart Chain)"],
      note: "USDT + BTC auto-verify on txhash submit (12+ confirmations). Send ONLY on BNB Smart Chain — other networks will lose funds.",
    },
    tiers: { starter_naira: n(sK), creator_naira: n(cK), pro_naira: n(pK), trader_naira: n(tK), trader_kobo: tK, trader_yearly: n(tK) * YEARLY_MULT,
      starter_kobo: sK, creator_kobo: cK, pro_kobo: pK,
      starter_yearly: n(sK) * YEARLY_MULT, creator_yearly: n(cK) * YEARLY_MULT, pro_yearly: n(pK) * YEARLY_MULT },
    two_pots: {
      subscription: "Your payment for USING the app. Goes to our account as revenue, unlocks your tier for 30 days. Non-refundable, never spendable.",
      wallet: "YOUR money, held for the AI to spend FOR you (flights, orders, services). Fund it via /wallet/topup; the agent debits it via /wallet/spend per booking. Unspent balance stays yours.",
    },
    note: "Prices are pegged at ₦1,370/$ — fixed while the dollar holds, and rise proportionally only if the dollar climbs past the peg. VAT-inclusive; the quoted amount is locked on your order. Yearly = 10x monthly (2 months free). Testers: redeem your trial code under Wallet → Redeem a code.",
  });
});
app.post("/billing/subscribe", requireAuth, limitNormal, async (req, res) => {
  const { tier, method } = req.body || {};
  const user_id = boundUser(req);
  // Only priced tiers are purchasable — free/tester/owner never reach checkout
  // (tester comes from admin-minted trial codes, owner from MASTER_CODE).
  if (!SUB_TIERS[tier] || !tierBaseKobo(tier)) return res.status(400).json({ error: "paid tier required (starter, creator, trader or pro)" });
  const rate = await ngnPerUsd();
  const amount_kobo = tierNaira(tier, rate); // ratchet-pegged, locked at order time
  const id = `pay_${Date.now().toString(36)}${crypto.randomBytes(4).toString("hex")}`;
  const passcode = makePasscode(); // auto-generated, user types it in "I've paid"
  db.prepare("INSERT INTO pending_payments (id, user_id, tier, amount_kobo, reference, purpose) VALUES (?, ?, ?, ?, ?, 'sub')").run(id, user_id, tier, amount_kobo, passcode);
  const m = String(method || "bank").toLowerCase();
  const naira = (amount_kobo / 100).toLocaleString();
  res.json({
    ok: true, payment_id: id, tier, amount_kobo, usd_ngn: rate, passcode,
    ...(m === "crypto"
      ? { pay_to: { bsc_address: BSC_ADDRESS, assets: ["USDT (BNB Smart Chain)", "BTC (BNB Smart Chain)"] }, instruction: `Send USDT or BTC on BNB Smart Chain equal to ~₦${naira} to ${BSC_ADDRESS}, then submit txhash + this passcode: ${passcode}.` }
      : { pay_to: { bank_name: PAY_BANK_NAME, account_number: PAY_ACCOUNT_NUMBER, account_name: PAY_ACCOUNT_NAME }, instruction: `Transfer ₦${naira} with narration ${passcode}, then enter passcode ${passcode} in "I've paid".` }),
  });
});
app.post("/billing/submit-reference", requireAuth, limitNormal, async (req, res) => {
  const { payment_id, passcode, sender_name, txid, network } = req.body || {};
  const pay = db.prepare("SELECT * FROM pending_payments WHERE id = ?").get(payment_id);
  if (!pay || (pay.status !== "pending" && pay.status !== "awaiting_review")) return res.status(404).json({ error: "pending payment not found" });
  // Brute-force guard: 8 passcode tries per hour per order, then re-subscribe.
  const atk = `pc:${payment_id}`;
  const now = Date.now();
  let att = _rl.get(atk);
  if (!att || now > att.reset) { att = { n: 0, reset: now + 3600e3 }; _rl.set(atk, att); }
  if (++att.n > 8) return res.status(429).json({ error: "too many wrong passcodes — create a fresh order" });
  if (String(passcode || "").toUpperCase().replace(/[^A-Z0-9]/gi, "") !== String(pay.reference).replace(/[^A-Z0-9]/gi, "")) {
    return res.status(400).json({ error: "wrong passcode — check the code from subscribe" });
  }
  _rl.delete(atk);
  // Crypto rail (keyless): network = "usdt" | "btc" (both on BSC, 12+ confs)
  // or "btc-native" (native Bitcoin via mempool.space, needs BTC_ADDRESS set).
  // Each txhash works exactly once (spent_txids).
  const net = String(network || (txid ? "usdt" : "")).toLowerCase();
  if (txid && ["usdt", "btc", "btc-native"].includes(net)) {
    try {
      const cleanTx = String(txid).trim();
      if (db.prepare("SELECT txid FROM spent_txids WHERE txid = ?").get(cleanTx)) {
        return res.status(400).json({ error: "txhash already used for another order" });
      }
      const rate = await ngnPerUsd();
      const usd = pay.amount_kobo / 100 / rate; // same live FX as the order quote
      let v;
      if (net === "btc-native") {
        if (!BTC_ADDRESS) throw new Error("native BTC deposits not configured — pay USDT/BTC on BNB Smart Chain instead");
        const price = await btcPriceUSD();
        v = await verifyBtcTx(cleanTx, Math.round((usd / price) * 1e8));
        if (!v.confirmed) {
          db.prepare("UPDATE pending_payments SET status = 'awaiting_review' WHERE id = ?").run(payment_id);
          return res.json({ ok: false, payment_id, status: "awaiting_review", error: "tx seen but unconfirmed — auto-approves after 1 confirmation, or manual review" });
        }
      } else {
        // "btc" = your Binance BTC-on-BSC (BTCB contract under the hood).
        const asset = net === "btc" ? "btcb" : "usdt";
        const token = BSC_TOKENS[asset];
        const unitPrice = asset === "usdt" ? 1 : await btcPriceUSD();
        v = await verifyBscTx(cleanTx, asset, (usd / unitPrice) * (10 ** token.decimals));
        if (!v.confirmed) {
          db.prepare("UPDATE pending_payments SET status = 'awaiting_review' WHERE id = ?").run(payment_id);
          return res.json({ ok: false, payment_id, status: "awaiting_review", error: `tx seen (${v.confirmations}/12 confirmations) — auto-approves at 12, or manual review` });
        }
      }
      db.prepare("INSERT INTO spent_txids (txid, payment_id) VALUES (?, ?)").run(cleanTx, payment_id);
      db.prepare("UPDATE pending_payments SET status = 'approved' WHERE id = ?").run(payment_id);
      if (pay.purpose === "topup") {
        const bal = walletCredit(pay.user_id, pay.amount_kobo, `topup ${payment_id} (${net} ${cleanTx.slice(0, 12)}…)`);
        takeSnapshot("approve-topup-crypto");
        return res.json({ ok: true, payment_id, status: "approved", verified: v, balance_kobo: bal });
      }
      if (pay.purpose === "credit_topup") {
        // Was falling through to the SUBSCRIPTION branch: crypto paid, user got a useless free activation code and 0 credits.
        const credits = Math.floor(pay.amount_kobo / CREDIT_NAIRA_KOBO);
        const bal = creditTxn(pay.user_id, credits, "topup", `credit top-up ${payment_id} (${net} ${cleanTx.slice(0, 12)})`);
        takeSnapshot("approve-credit-topup-crypto");
        return res.json({ ok: true, payment_id, status: "approved", verified: v, credits_added: credits, balance: bal });
      }
      const code = makeActivationCode();
      const exp = new Date(Date.now() + SUB_TIERS[pay.tier].days * 864e5).toISOString();
      db.prepare("INSERT INTO activation_codes (code, tier, status, user_id, expires_at) VALUES (?, ?, 'unused', NULL, ?)").run(code, pay.tier, exp);
      takeSnapshot("approve-sub-crypto");
      return res.json({ ok: true, payment_id, status: "approved", verified: v, activation_code: code, redeem_by: exp, note: `${net.toUpperCase()} confirmed. Enter activation code in app (Wallet → Redeem) before ${exp.slice(0, 10)} — the ${SUB_TIERS[pay.tier].days}-day subscription starts at redeem time.` });
    } catch (e) {
      db.prepare("UPDATE pending_payments SET status = 'awaiting_review' WHERE id = ?").run(payment_id);
      return res.json({ ok: false, payment_id, status: "awaiting_review", error: `auto-verify failed (${e.message}) — queued for manual review` });
    }
  }
  db.prepare("UPDATE pending_payments SET status = 'awaiting_review' WHERE id = ?").run(payment_id);
  res.json({ ok: true, payment_id, status: "awaiting_review", note: `Thanks ${sender_name || "you"} — passcode accepted, confirming shortly.` });
});
app.get("/billing/pending", requireAdmin, (_req, res) => {
  // YOUR admin queue: review sender names / txids vs bank alerts, then approve.
  res.json({ ok: true, pending: db.prepare("SELECT * FROM pending_payments WHERE status IN ('pending','awaiting_review') ORDER BY created_at DESC LIMIT 100").all() });
});
// Credit top-up: any amount → credits at ₦10/credit. Same rails as wallet top-up.
app.post("/billing/topup-credits", requireAuth, limitNormal, async (req, res) => {
  const { amount_kobo, method } = req.body || {};
  const user_id = boundUser(req);
  if (!Number.isInteger(Number(amount_kobo)) || amount_kobo < 10000) return res.status(400).json({ error: "integer amount_kobo (min ₦100) required" });
  const credits = Math.floor(Number(amount_kobo) / CREDIT_NAIRA_KOBO);
  if (credits < 1) return res.status(400).json({ error: `minimum top-up is ₦${CREDIT_NAIRA_KOBO / 100} (1 credit)` });
  const id = `ctop_${Date.now().toString(36)}${crypto.randomBytes(4).toString("hex")}`;
  const passcode = makePasscode();
  db.prepare("INSERT INTO pending_payments (id, user_id, tier, amount_kobo, reference, purpose) VALUES (?, ?, 'free', ?, ?, 'credit_topup')").run(id, user_id, Number(amount_kobo), passcode);
  const m = String(method || "bank").toLowerCase();
  const naira = (Number(amount_kobo) / 100).toLocaleString();
  res.json({
    ok: true, payment_id: id, amount_kobo: Number(amount_kobo), credits, passcode, purpose: "credit top-up",
    ...(m === "crypto"
      ? { pay_to: { bsc_address: BSC_ADDRESS, assets: ["USDT (BNB Smart Chain)", "BTC (BNB Smart Chain)"] }, instruction: `Send USDT or BTC on BNB Smart Chain equal to ~₦${naira} to ${BSC_ADDRESS}, then submit txhash + this passcode: ${passcode}.` }
      : { pay_to: { bank_name: PAY_BANK_NAME, account_number: PAY_ACCOUNT_NUMBER, account_name: PAY_ACCOUNT_NAME }, instruction: `Transfer ₦${naira} with narration ${passcode}, then enter passcode ${passcode} in "I've paid".` }),
  });
});
// Credit balance + ledger for the app.
app.get("/billing/credits", requireAuth, limitNormal, (req, res) => {
  const user_id = boundUser(req);
  grantMonthlyCredits(user_id);
  const bal = creditBalance(user_id);
  const txns = db.prepare("SELECT amount, kind, note, created_at FROM credit_txns WHERE user_id = ? ORDER BY id DESC LIMIT 50").all(user_id);
  const tier = activeTier(user_id);
  res.json({ ok: true, user_id, balance: bal, tier, credits_per_naira: CREDIT_NAIRA_KOBO,
    linked: !!db.prepare("SELECT 1 FROM api_keys WHERE user_id = ?").get(user_id), txns });
});
app.post("/billing/approve", requireAdmin, (req, res) => {
  // YOU call this after confirming the bank alert / txid (same AUTH_TOKEN = admin).
  // Subs return an ACTIVATION CODE (send it to the user); top-ups credit wallet directly.
  const { payment_id } = req.body || {};
  const pay = db.prepare("SELECT * FROM pending_payments WHERE id = ?").get(payment_id);
  if (!pay || (pay.status !== "pending" && pay.status !== "awaiting_review")) return res.status(404).json({ error: "pending payment not found" });
  db.prepare("UPDATE pending_payments SET status = 'approved' WHERE id = ?").run(payment_id);
  if (pay.purpose === "topup") {
    const bal = walletCredit(pay.user_id, pay.amount_kobo, `topup ${payment_id}`);
    takeSnapshot("approve-topup");
    return res.json({ ok: true, purpose: "topup", user_id: pay.user_id, balance_kobo: bal });
  }
  if (pay.purpose === "credit_topup") {
    const credits = Math.floor(pay.amount_kobo / CREDIT_NAIRA_KOBO);
    const bal = creditTxn(pay.user_id, credits, "topup", `credit top-up ${payment_id}`);
    takeSnapshot("approve-credit-topup");
    return res.json({ ok: true, purpose: "credit_topup", user_id: pay.user_id, credits_added: credits, balance: bal });
  }
  const code = makeActivationCode();
  const exp = new Date(Date.now() + SUB_TIERS[pay.tier].days * 864e5).toISOString();
  db.prepare("INSERT INTO activation_codes (code, tier, status, user_id, expires_at) VALUES (?, ?, 'unused', NULL, ?)").run(code, pay.tier, exp);
  takeSnapshot("approve-sub");
  res.json({ ok: true, purpose: "sub", user_id: pay.user_id, tier: pay.tier, activation_code: code, redeem_by: exp,
    note: `Redeem in-app (Wallet → Redeem) before ${exp.slice(0, 10)} — the ${SUB_TIERS[pay.tier].days}-day subscription starts at redeem time.` });
});
app.post("/billing/mint", requireAdmin, (req, res) => {
  // Admin: mint trial codes (default: 7-day tester with full Pro-level caps).
  // `days` = REDEEM-BY window only — the subscription clock (tier days) starts
  // at redeem time, so testers always get their full week. Testers redeem in
  // the app under Wallet → "Redeem a code"; nothing else needed from them.
  // Examples: {tier:"tester", days:14, count:20} for a tester cohort,
  // {tier:"creator", days:90, count:1} for your own long-lived key.
  const { tier = "tester", days, count } = req.body || {};
  if (!SUB_TIERS[tier] || tier === "owner") return res.status(400).json({ error: "valid tier required" });
  const n = Math.min(Math.max(Number(count) || 1, 1), 100);
  const d = Math.min(Math.max(Number(days) || 7, 1), 365);
  const codes = [];
  for (let i = 0; i < n; i++) {
    const code = makeActivationCode();
    const exp = new Date(Date.now() + d * 864e5).toISOString();
    db.prepare("INSERT INTO activation_codes (code, tier, status, user_id, expires_at) VALUES (?, ?, 'unused', NULL, ?)").run(code, `${tier}`, exp);
    codes.push({ code, tier, expires_at: exp });
  }
  res.json({ ok: true, codes, redeem_by: codes[0]?.expires_at || null,
    howto: "Testers open Wallet → 'Redeem a code', paste the code, and get full features until the subscription expires. Re-pasting reports status but never extends it." });
});
app.post("/billing/redeem", requireAuth, limitNormal, (req, res) => {
  // User enters activation code (or your MASTER_CODE) in-app -> tier unlocks.
  const { code } = req.body || {};
  const user_id = boundUser(req);
  if (!code) return res.status(400).json({ error: "code required" });
  const c = String(code).trim();
  if (MASTER_CODE && MASTER_CODE.length >= 8) {    const a = Buffer.from(c), b = Buffer.from(MASTER_CODE);
    if (a.length === b.length && crypto.timingSafeEqual(a, b)) {
      db.prepare("INSERT INTO subscriptions (user_id, tier, status, expires_at) VALUES (?, 'owner', 'active', ?) ON CONFLICT(user_id) DO UPDATE SET tier='owner', status='active', expires_at=?, updated_at=CURRENT_TIMESTAMP").run(user_id, "2999-01-01", "2999-01-01");
      return res.json({ ok: true, tier: "owner", expires_at: "2999-01-01", note: "Owner mode: full potential, never expires." });
    }
  }
  const row = db.prepare("SELECT * FROM activation_codes WHERE code = ?").get(c);
  if (!row) return res.status(400).json({ error: "invalid code — check each character and try again" });
  if (!SUB_TIERS[row.tier]) return res.status(400).json({ error: `this code is for retired tier '${row.tier}' — contact support for a fresh one` });
  // Re-paste NEVER extends: the subscription clock starts at FIRST redeem only.
  // (The old code reset 30 days on every paste — one payment, lifetime sub.)
  // Same-device re-paste just reports the live subscription. Legacy bridge:
  // codes redeemed under the old shared 'default' identity are adoptable ONCE
  // by the first device that re-pastes them.
  const grantFresh = (why) => {
    const exp = new Date(Date.now() + SUB_TIERS[row.tier].days * 864e5).toISOString();
    db.prepare("UPDATE activation_codes SET status = 'used', user_id = ? WHERE code = ?").run(user_id, c);
    db.prepare("INSERT INTO subscriptions (user_id, tier, status, expires_at) VALUES (?, ?, 'active', ?) ON CONFLICT(user_id) DO UPDATE SET tier=?, status='active', expires_at=?, updated_at=CURRENT_TIMESTAMP").run(user_id, row.tier, exp, row.tier, exp);
    return res.json({ ok: true, tier: row.tier, expires_at: exp, refreshed: why });
  };
  if (row.status === "used") {
    if (row.user_id === user_id) {
      const cur = db.prepare("SELECT tier, expires_at FROM subscriptions WHERE user_id = ?").get(user_id);
      const live = cur && cur.expires_at && new Date(cur.expires_at).getTime() > Date.now();
      if (live) return res.json({ ok: true, tier: cur.tier, expires_at: cur.expires_at, refreshed: false, note: `Already active until ${String(cur.expires_at).slice(0, 10)} — re-pasting never extends it.` });
      return res.status(400).json({ error: "code already used and its subscription has expired — subscribe again for a fresh code" });
    }
    if (row.user_id === "default" || !row.user_id) return grantFresh(true); // one-time legacy adoption
    return res.status(400).json({ error: "code already redeemed on another device — ask support for your own" });
  }
  if (row.status !== "unused") return res.status(400).json({ error: "code no longer valid — contact support for a fresh one" });
  if (row.expires_at && new Date(row.expires_at).getTime() <= Date.now()) {
    db.prepare("UPDATE activation_codes SET status = 'expired' WHERE code = ?").run(c);
    return res.status(400).json({ error: "code expired — contact support for a fresh one" });
  }
  return grantFresh(false);
});
app.get("/billing/status", requireAuth, limitNormal, (req, res) => {
  const user_id = boundUser(req);
  grantMonthlyCredits(user_id);
  const sub = db.prepare("SELECT tier, expires_at FROM subscriptions WHERE user_id = ?").get(user_id);
  const tier = activeTier(user_id);
  const t = SUB_TIERS[tier] || {};
  res.json({
    ok: true, user_id, tier, expires_at: sub?.expires_at || null,
    credits: creditBalance(user_id),
    credits_mo: t.credits_mo || 0,
    features: t.features || [],
    feature_labels: FEATURE_LABELS,
    linked: !!db.prepare("SELECT 1 FROM api_keys WHERE user_id = ?").get(user_id),
    usage_today: getUsage(user_id), usage_month: monthUsage(user_id),
    db_persistent: !EPHEMERAL_DB,
  });
});
function takeSnapshot(kind) {
  const payload = JSON.stringify({
    wallet_total_kobo: db.prepare("SELECT COALESCE(SUM(balance_kobo),0) AS s FROM wallet_accounts").get().s,
    ledger_rows: db.prepare("SELECT COUNT(*) AS c FROM wallet_txns").get().c,
    ledger_head: db.prepare("SELECT tx_hash FROM wallet_txns ORDER BY id DESC LIMIT 1").get()?.tx_hash || "GENESIS",
    subs_active: db.prepare("SELECT COUNT(*) AS c FROM subscriptions WHERE status='active'").get().c,
    pending: db.prepare("SELECT COUNT(*) AS c FROM pending_payments WHERE status IN ('pending','awaiting_review')").get().c,
  });
  const r = db.prepare("INSERT INTO snapshots (kind, payload) VALUES (?, ?)").run(kind, payload);
  void backupNow(`snapshot-${kind}`); // off-site copy on every money event, fire-and-forget
  return { id: r.lastInsertRowid, ...JSON.parse(payload) };
}
// Push a consistent DB copy to Supabase Storage (frit-latest.db + timestamped).
// Never throws — backup must not break money flows.
async function backupNow(reason = "manual") {
  if (!HAS_REMOTE_BACKUP) return { ok: false, skipped: true };
  try {
    await supaBucket();
    const tmp = `${DB_PATH}.backup-tmp`;
    await db.backup(tmp);
    const bytes = _readFileSync(tmp);
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    for (const name of [`frit-latest.db`, `frit-${stamp}.db`]) {
      const up = await fetch(`${SUPABASE_URL}/storage/v1/object/${SUPABASE_BUCKET}/${name}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${SUPABASE_KEY}`, apikey: SUPABASE_KEY,
          "Content-Type": "application/octet-stream", "x-upsert": "true",
        },
        body: bytes,
        signal: AbortSignal.timeout(120_000),
      });
      if (!up.ok) throw new Error(`upload ${name} HTTP ${up.status}`);
    }
    // Prune timestamped copies to the newest 7 (frit-latest.db always kept).
    try {
      const list = await (await fetch(`${SUPABASE_URL}/storage/v1/object/list/${SUPABASE_BUCKET}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${SUPABASE_KEY}`, apikey: SUPABASE_KEY, "Content-Type": "application/json" },
        body: JSON.stringify({ prefix: "frit-20", limit: 100 }),
        signal: AbortSignal.timeout(20_000),
      })).json();
      const olds = (list || []).map(f => f.name).filter(n => n !== "frit-latest.db").sort().slice(0, -7);
      for (const n of olds) {
        await fetch(`${SUPABASE_URL}/storage/v1/object/${SUPABASE_BUCKET}/${n}`, {
          method: "DELETE",
          headers: { Authorization: `Bearer ${SUPABASE_KEY}`, apikey: SUPABASE_KEY },
          signal: AbortSignal.timeout(20_000),
        }).catch(() => {});
      }
    } catch (e) { console.warn("[backup] old-backup prune failed:", e.message); }
    try { (await import("fs")).unlinkSync(tmp); } catch { /* tmp already gone */ }
    console.log(`[backup] pushed (${bytes.length} bytes, ${reason})`);
    return { ok: true, bytes: bytes.length };
  } catch (e) {
    console.warn("[backup] push failed:", e.message);
    return { ok: false, error: e.message };
  }
}
setInterval(() => { void backupNow("hourly"); }, 60 * 60 * 1000).unref?.();
app.post("/admin/snapshot", requireAdmin, (_req, res) => res.json({ ok: true, snapshot: takeSnapshot("manual") }));
app.get("/admin/snapshots", requireAdmin, (_req, res) => {
  res.json({ ok: true, snapshots: db.prepare("SELECT * FROM snapshots ORDER BY id DESC LIMIT 100").all() });
});
app.get("/admin/ledger/verify", requireAdmin, (_req, res) => {
  const rows = db.prepare("SELECT id, user_id, amount_kobo, kind, note, created_at, prev_hash, tx_hash FROM wallet_txns ORDER BY id ASC LIMIT 5000").all();
  let prev = "GENESIS";
  for (const r of rows) {
    if (r.prev_hash !== prev) return res.json({ ok: false, break_at: r.id, reason: "prev_hash mismatch — ledger was edited" });
    const recomputed = crypto.createHash("sha256").update(`${r.prev_hash}|${r.user_id}|${r.amount_kobo}|${r.kind}|${r.note}|${r.created_at}`).digest("hex");
    if (recomputed !== r.tx_hash) return res.json({ ok: false, break_at: r.id, reason: "tx_hash mismatch — row was edited" });
    prev = r.tx_hash;
  }
  res.json({ ok: true, rows: rows.length, head: prev });
});
app.post("/admin/backup", requireAdmin, async (_req, res) => {
  try {
    const dir = DB_PATH.startsWith("/data") ? "/data/backups" : join(__dirname, "../backups");
    _mkdirSync(dir, { recursive: true });
    const file = join(dir, `frit-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.db`);
    await db.backup(file);
    const remote = await backupNow("manual");
    res.json({ ok: true, file, remote });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});
app.post("/admin/keys", requireAdmin, (req, res) => {
  // Mint a per-user app key: binds ALL quota/wallet/tier actions to user_id.
  // The app stores this key (not the master token) — spoofing ends here.
  const { user_id, label } = req.body || {};
  if (!user_id) return res.status(400).json({ error: "user_id required" });
  const key = `frit_${crypto.randomBytes(24).toString("hex")}`;
  db.prepare("INSERT INTO api_keys (key_hash, user_id, label) VALUES (?, ?, ?)").run(hashKey(key), String(user_id), String(label || ""));
  res.json({ ok: true, user_id, api_key: key, note: "Shown once — store in the app's TokenStore." });
});
app.get("/admin/keys", requireAdmin, (_req, res) => {
  res.json({ ok: true, keys: db.prepare("SELECT user_id, label, created_at FROM api_keys ORDER BY created_at DESC").all() });
});
// Per-install device keys (anti free-credit farming). The app registers its
// install UUID once, stores the returned key in app-private prefs, and calls
// everything with it — the server then binds quota/credits/tier to a real
// install instead of a self-asserted id. Strictly rate-limited (5/hour/IP);
// one live key per install_id (re-register rotates). Farming now costs a
// reinstall + IP budget per 200 credits instead of one random string.
const limitInstall = rateLimit({ windowMs: 3600e3, max: 5 });
app.post("/auth/install", limitInstall, (req, res) => {
  const install_id = String(req.body?.install_id || "").trim();
  if (!/^app_[0-9a-f]{12}$/.test(install_id)) return res.status(400).json({ error: "install_id required (app_ + 12 hex chars)" });
  const key = `frit_${crypto.randomBytes(24).toString("hex")}`;
  db.transaction(() => {
    db.prepare("DELETE FROM api_keys WHERE user_id = ?").run(install_id);
    db.prepare("INSERT INTO api_keys (key_hash, user_id, label) VALUES (?, ?, 'install')").run(hashKey(key), install_id);
  })();
  res.json({ ok: true, user_id: install_id, api_key: key, note: "Store on device (app-private prefs). Shown once — re-register rotates it." });
});
// ---- Runtime models (Creator console) ----
// The app's Creator console lists these, and the creator swaps any slug live:
// new OpenRouter models go in with zero code change. {key, value} — empty
// value clears the override (falls back to env, then baked default).
const MODEL_SLUG_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?$/;
app.get("/admin/models", requireAdmin, (_req, res) => {
  res.json({
    ok: true, verify_at: "https://openrouter.ai/models",
    models: MODEL_KEYS.map((k) => ({
      key: k, value: M(k),
      override: overrideOf(k), env: process.env[k] || "", default: MODEL_DEFAULTS[k] || "",
      source: overrideOf(k) ? "console" : (process.env[k] ? "env" : "default"),
      ...(MODEL_META[k] || {}),
    })),
  });
});
app.post("/admin/models", requireAdmin, (req, res) => {
  const { key, value } = req.body || {};
  if (!MODEL_KEYS.includes(key)) return res.status(400).json({ error: `unknown model key (pick from ${MODEL_KEYS.join(", ")})` });
  const v = String(value ?? "").trim();
  if (v && !MODEL_SLUG_RE.test(v)) return res.status(400).json({ error: "value must be a model slug like vendor/model-name (optionally :free). Empty clears it." });
  if (!v) db.prepare("DELETE FROM model_overrides WHERE key = ?").run(key);
  else db.prepare("INSERT INTO model_overrides (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP").run(key, v, v);
  refreshModelConfig();
  res.json({ ok: true, key, value: M(key), cleared: !v, source: v ? "console" : (process.env[key] ? "env" : "default") });
});
// ---- Self-describing endpoint catalog (Owner console auto-populates) ----
// Walks the live Express route table, so every endpoint added to this server
// appears in the app's Endpoint console with NO app update. Curated buttons
// in the app stay as shortcuts; this catalog is the source of truth.
const ADMIN_CATALOG_PATHS = new Set([
  "GET /billing/approve", "GET /billing/pending", "POST /billing/approve",
  "POST /billing/mint", "GET /feedback", "GET /admin/keys", "POST /admin/keys",
]);
function endpointCatalog() {
  const out = [];
  for (const layer of (app._router?.stack || [])) {
    if (!layer.route) continue;
    const path = String(layer.route.path || "");
    if (!path.startsWith("/")) continue;
    for (const m of Object.keys(layer.route.methods || {})) {
      if (m === "_all" || m === "head" || m === "options") continue;
      const method = m.toUpperCase();
      const admin = path.startsWith("/admin") || ADMIN_CATALOG_PATHS.has(`${method} ${path}`);
      const params = [...path.matchAll(/:([A-Za-z0-9_]+)/g)].map((x) => x[1]);
      let group = "Other";
      if (/^\/(billing|wallet)/.test(path)) group = admin ? "Money (admin)" : "Money";
      else if (/^\/(creative-suite|videos|images|cinema)/.test(path)) group = "Creative & media";
      else if (/^\/(market|trade|positions|strategy|tasks|agent|screen|tools|transcribe|sandbox|enhanced|memory|weather)/.test(path)) group = "Trading & agent";
      else if (/^\/(models|systems|health|feedback|admin)/.test(path)) group = admin ? "Money (admin)" : "System";
      out.push({
        method, path, group, admin,
        path_params: params,
        needs_body: ["POST", "PUT", "PATCH"].includes(method),
      });
    }
  }
  out.sort((a, b) => a.group.localeCompare(b.group) || a.path.localeCompare(b.path) || a.method.localeCompare(b.method));
  return out;
}
const catalogHandler = (_req, res) => {
  const endpoints = endpointCatalog();
  res.json({ ok: true, count: endpoints.length, endpoints });
};
app.get("/models/endpoints", requireAuth, limitNormal, catalogHandler); // app token: path catalog only
app.get("/admin/endpoints", requireAdmin, catalogHandler); // admin alias
app.get("/wallet/balance", requireAuth, limitNormal, (req, res) => {
  const user_id = boundUser(req);
  res.json({ ok: true, user_id, balance_kobo: walletBalance(user_id) });
});
app.post("/wallet/topup", requireAuth, limitNormal, async (req, res) => {
  // Wallet funding uses the SAME rails as subs (bank passcode or crypto txid):
  // creates a PENDING top-up order — approve credits the wallet, so users can
  // never credit themselves. Same passcode/"I've paid" screen as subscriptions.
  const { amount_kobo, method } = req.body || {};
  const user_id = boundUser(req);
  if (!Number.isInteger(Number(amount_kobo)) || Number(amount_kobo) < 10000) return res.status(400).json({ error: "integer amount_kobo (min ₦100) required" });
  const id = `top_${Date.now().toString(36)}${crypto.randomBytes(4).toString("hex")}`;
  const passcode = makePasscode();
  db.prepare("INSERT INTO pending_payments (id, user_id, tier, amount_kobo, reference, purpose) VALUES (?, ?, 'free', ?, ?, 'topup')").run(id, user_id, Number(amount_kobo), passcode);
  const m = String(method || "bank").toLowerCase();
  const naira = (Number(amount_kobo) / 100).toLocaleString();
  res.json({
    ok: true, payment_id: id, amount_kobo: Number(amount_kobo), passcode, purpose: "wallet top-up (YOUR spending money — not a subscription)",
    ...(m === "crypto"
      ? { pay_to: { btc_address: BTC_ADDRESS, usdt_tron_address: USDT_TRON_ADDRESS }, instruction: `Send crypto equal to ~₦${naira} for YOUR wallet, then submit txid + passcode ${passcode}.` }
      : { pay_to: { bank_name: PAY_BANK_NAME, account_number: PAY_ACCOUNT_NUMBER, account_name: PAY_ACCOUNT_NAME }, instruction: `Transfer ₦${naira} to YOUR wallet with narration ${passcode}, then enter passcode ${passcode} in "I've paid". Subscription and wallet are separate — this does not buy a tier.` }),
  });
});
app.post("/wallet/spend", requireAuth, limitNormal, (req, res) => {
  // Agent calls this when booking/paying for a service on user's behalf.
  try {
    const { amount_kobo, note } = req.body || {};
    const user_id = boundUser(req);
    if (!Number.isInteger(Number(amount_kobo))) return res.status(400).json({ error: "integer amount_kobo required" });
    res.json({ ok: true, balance_kobo: walletSpend(user_id, Number(amount_kobo), String(note || "agent spend").slice(0, 200)) });
  } catch (e) {
    res.status(400).json({ ok: false, error: e.message });
  }
});
// Withdrawal: user sends THEIR bank details, creator pays manually from the
// receiving account and marks paid. No auto-send exists (manual transfer).
app.post("/wallet/withdraw", requireAuth, limitNormal, (req, res) => {
  const { bank_name, account_number, account_name, amount_kobo } = req.body || {};
  const user_id = boundUser(req);
  const amt = Number(amount_kobo);
  if (!bank_name || !account_number || !account_name) return res.status(400).json({ error: "bank_name, account_number and account_name required" });
  if (!Number.isInteger(amt) || amt < 50000) return res.status(400).json({ error: "minimum withdrawal is ₦500" });
  if (walletBalance(user_id) < amt) return res.status(400).json({ error: "insufficient wallet balance" });
  const id = `wd_${Date.now().toString(36)}${crypto.randomBytes(4).toString("hex")}`;
  db.prepare("INSERT INTO withdrawals (id, user_id, amount_kobo, bank_name, account_number, account_name) VALUES (?, ?, ?, ?, ?, ?)").run(
    id, user_id, amt, String(bank_name).slice(0, 60), String(account_number).slice(0, 20), String(account_name).slice(0, 80));
  res.json({ ok: true, withdrawal_id: id, status: "pending", note: "Request sent. The creator pays manually and marks it paid — usually within 24h." });
});
app.get("/wallet/withdrawals", requireAuth, limitNormal, (req, res) => {
  const user_id = boundUser(req);
  res.json({ ok: true, withdrawals: db.prepare("SELECT id, amount_kobo, bank_name, account_number, account_name, status, admin_note, created_at FROM withdrawals WHERE user_id = ? ORDER BY created_at DESC LIMIT 50").all(user_id) });
});
// Creator-only view of the engine's FULL diagnostic output, including the
// internal `reasons` strings that are stripped from every user-facing
// payload. Admin token only — this is how you inspect why the engine decided
// what it decided without the user ever seeing it.
app.get("/admin/strategy/diagnostics", requireAdmin, async (req, res) => {
  const symbol = String(req.query.symbol || "XAUUSD").toUpperCase();
  try {
    const full = await mtfStrategy.analyze(symbol, {
      interval: req.query.interval,
      balance: req.query.balance != null ? Number(req.query.balance) : undefined,
    });
    res.json({
      ok: true,
      symbol,
      note: "Creator diagnostics — includes engine-internal reasons. Never expose to users.",
      reasons: full?.reasons ?? [],
      guards: full?.guards ?? {},
      full,
    });
  } catch (err) {
    res.status(500).json({ error: "Diagnostics failed", details: err.message });
  }
});

// Creator inbox: ONE poll for everything needing your eyes — payments,
// withdrawals, trial feedback. The creator app polls this and notifies.
app.get("/admin/inbox", requireAdmin, (_req, res) => {
  res.json({
    ok: true,
    payments: db.prepare("SELECT id, user_id, tier, amount_kobo, reference, purpose, status, created_at FROM pending_payments WHERE status IN ('pending','awaiting_review') ORDER BY created_at DESC LIMIT 100").all(),
    withdrawals: db.prepare("SELECT * FROM withdrawals WHERE status = 'pending' ORDER BY created_at DESC LIMIT 100").all(),
    feedback_unread: db.prepare("SELECT COUNT(*) AS c FROM feedback").get().c,
    latest_feedback: db.prepare("SELECT * FROM feedback ORDER BY id DESC LIMIT 5").all(),
  });
});
app.post("/admin/withdrawals/approve", requireAdmin, (req, res) => {
  // Call AFTER you send the money from your bank app. Debits the wallet now
  // (not at request time) so failed/duplicate approvals can't double-spend.
  const { id, admin_note } = req.body || {};
  const w = db.prepare("SELECT * FROM withdrawals WHERE id = ?").get(id);
  if (!w || w.status !== "pending") return res.status(404).json({ error: "pending withdrawal not found" });
  try {
    walletSpend(w.user_id, w.amount_kobo, `withdrawal ${id} paid to ${w.bank_name} ${w.account_number}`);
  } catch (e) {
    return res.status(400).json({ ok: false, error: `cannot debit: ${e.message}` });
  }
  db.prepare("UPDATE withdrawals SET status = 'paid', admin_note = ? WHERE id = ?").run(String(admin_note || "").slice(0, 200), id);
  takeSnapshot("withdrawal-paid");
  res.json({ ok: true, id, status: "paid" });
});
app.post("/admin/withdrawals/decline", requireAdmin, (req, res) => {
  const { id, admin_note } = req.body || {};
  const w = db.prepare("SELECT * FROM withdrawals WHERE id = ?").get(id);
  if (!w || w.status !== "pending") return res.status(404).json({ error: "pending withdrawal not found" });
  db.prepare("UPDATE withdrawals SET status = 'declined', admin_note = ? WHERE id = ?").run(String(admin_note || "").slice(0, 200), id);
  res.json({ ok: true, id, status: "declined" });
});
app.get("/creative/jobs", requireAuth, limitNormal, (req, res) => {
  const user_id = boundUser(req);
  res.json({ ok: true, jobs: db.prepare("SELECT id, kind, status, prompt, substr(result_url,1,200) AS draft_head, created_at FROM creative_jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT 50").all(user_id) });
});
app.post("/creative/ground", requireAuth, limitNormal, async (req, res) => {
  try {
    const { brief, prompt } = req.body || {};
    if (!brief && !prompt) return res.status(400).json({ error: "brief required" });
    res.json({ ok: true, grounding: await groundScene(brief || prompt) });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});
// ==================== CREATIVE SUITE ====================
app.post("/creative-suite/images/generate", requireAuth, limitCostly, async (req, res) => {
  try {
    const { prompt, style_preset, aspect_ratio, n, model, reference_images, brandkit_id } = req.body || {};
    if (!prompt) return res.status(400).json({ error: "prompt required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const preset = promptEngine.STYLE_PRESETS.find(p => p.id === style_preset);
    // F-13: brand lock applied server-side so every studio honours it.
    const { brand, prompt: lockedPrompt } = applyBrand(uid, brandkit_id, prompt);
    const enhancedPrompt = preset ? `${lockedPrompt}, ${preset.description}` : lockedPrompt;
    const refs = Array.isArray(reference_images) ? reference_images.filter(u => typeof u === "string" && u.length > 0).slice(0, 3) : [];
    const data = await generateImage(enhancedPrompt, {
      aspect_ratio, n, model,
      input_references: refs.length ? refs.map(u => ({ type: "image_url", image_url: { url: u } })) : undefined,
    });
    db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    const assetId = `img_${Date.now()}`;
    db.prepare("INSERT INTO creative_assets (id, user_id, kind, source, current_url, metadata) VALUES (?, ?, 'image', 'generated', ?, ?)")
      .run(assetId, uid, data.image_url || "", JSON.stringify({ prompt, style_preset, model: data.model, brandkit_id: brandkit_id || null }));
    debitCredits(uid, "image", "image generate"); // post-success only: failed generations are not billed
    res.json({ ok: true, asset_id: assetId, brand_applied: !!brand, ...data });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// F-12: server-side image ingest validation (client checks are advisory).
// Accepts a base64 string or data URL; returns a Buffer, or null when the
// payload is empty, oversized, or not png/jpeg/webp/gif by magic bytes.
const MAX_IMAGE_BYTES = 15 * 1024 * 1024;
function decodeImageB64(input) {
  if (typeof input !== "string" || input.length < 16) return null;
  const raw = input.replace(/^data:image\/[\w.+-]+;base64,/, "");
  if (raw.length > Math.ceil(MAX_IMAGE_BYTES * 4 / 3) + 8) return null;
  const buf = Buffer.from(raw, "base64");
  if (buf.length < 12 || buf.length > MAX_IMAGE_BYTES) return null;
  const isPng = buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47;
  const isJpg = buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff;
  const isGif = buf.toString("ascii", 0, 3) === "GIF";
  const isWebp = buf.toString("ascii", 0, 4) === "RIFF" && buf.toString("ascii", 8, 12) === "WEBP";
  return (isPng || isJpg || isGif || isWebp) ? buf : null;
}
const BAD_IMAGE = { error: "invalid or oversized image (png/jpg/webp/gif, max 15MB)" };

app.post("/creative-suite/images/edit", requireAuth, limitNormal, async (req, res) => {
  try {
    const { image, operations } = req.body || {};
    if (!image || !Array.isArray(operations)) return res.status(400).json({ error: "image and operations required" });
    const imageBuffer = decodeImageB64(image);
    if (!imageBuffer) return res.status(400).json(BAD_IMAGE);
    const resultBuffer = await imageProcessor.applyOperations(imageBuffer, operations);
    const resultBase64 = resultBuffer.toString("base64");
    const meta = await imageProcessor.getImageMetadata(resultBuffer);
    res.json({ ok: true, result: `data:image/png;base64,${resultBase64}`, ...meta });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/images/remove-background", requireAuth, limitCostly, async (req, res) => {
  try {
    const { image } = req.body || {};
    if (!image) return res.status(400).json({ error: "image required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const imageBuffer = decodeImageB64(image);
    if (!imageBuffer) return res.status(400).json(BAD_IMAGE);
    const result = await aiFeatures.removeBackground(imageBuffer);
    debitCredits(uid, "image", "remove background");
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/images/retouch", requireAuth, limitCostly, async (req, res) => {
  try {
    const { image, options } = req.body || {};
    if (!image) return res.status(400).json({ error: "image required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const imageBuffer = decodeImageB64(image);
    if (!imageBuffer) return res.status(400).json(BAD_IMAGE);
    const result = await aiFeatures.retouch(imageBuffer, options || {});
    debitCredits(uid, "image", "retouch");
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/images/style-transfer", requireAuth, limitCostly, async (req, res) => {
  try {
    const { image, style } = req.body || {};
    if (!image || !style) return res.status(400).json({ error: "image and style required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const imageBuffer = decodeImageB64(image);
    if (!imageBuffer) return res.status(400).json(BAD_IMAGE);
    const result = await aiFeatures.styleTransfer(imageBuffer, style);
    debitCredits(uid, "image", "style transfer");
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/images/upscale", requireAuth, limitCostly, async (req, res) => {
  try {
    const { image, scale } = req.body || {};
    if (!image) return res.status(400).json({ error: "image required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const imageBuffer = decodeImageB64(image);
    if (!imageBuffer) return res.status(400).json(BAD_IMAGE);
    const result = await aiFeatures.upscaleImage(imageBuffer, scale || 2);
    debitCredits(uid, "image", "upscale");
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/images/generative-fill", requireAuth, limitCostly, async (req, res) => {
  try {
    const { image, mask, prompt } = req.body || {};
    if (!image || !mask) return res.status(400).json({ error: "image and mask required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const imageBuffer = decodeImageB64(image);
    const maskBuffer = decodeImageB64(mask);
    if (!imageBuffer || !maskBuffer) return res.status(400).json(BAD_IMAGE);
    const result = await aiFeatures.generativeFill(imageBuffer, maskBuffer, prompt);
    debitCredits(uid, "image", "generative fill");
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/images/batch", requireAuth, limitCostly, async (req, res) => {
  try {
    const { images, operation, params } = req.body || {};
    if (!Array.isArray(images) || !images.length) return res.status(400).json({ error: "images array required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const jobId = `batch_${Date.now()}`;
    db.prepare("INSERT INTO creative_batch_jobs (id, user_id, operation, asset_ids, params, status, total_count) VALUES (?, ?, ?, ?, ?, 'processing', ?)")
      .run(jobId, uid, operation, JSON.stringify(images), JSON.stringify(params || {}), images.length);
    // AI ops return {image_url | image_b64} objects; Sharp ops return Buffers.
    // Normalize both to a displayable URL/data-URL (never "[object Object]").
    function normalizeImageResult(r) {
      if (!r) return null;
      if (typeof r === "string") return r;
      if (r.image_url) return r.image_url;
      if (r.image_b64) return `data:image/png;base64,${r.image_b64}`;
      if (Buffer.isBuffer(r)) return `data:image/png;base64,${r.toString("base64")}`;
      return null;
    }
    const results = [];
    const providerOp = ["remove_bg", "upscale", "style_transfer"].includes(operation);
    for (let i = 0; i < images.length; i++) {
      try {
        const imgBuffer = decodeImageB64(String(images[i]));
        if (!imgBuffer) throw new Error("invalid or oversized image");
        let out;
        if (operation === "remove_bg") out = await aiFeatures.removeBackground(imgBuffer);
        else if (operation === "upscale") out = await aiFeatures.upscaleImage(imgBuffer, params?.scale || 2);
        else if (operation === "style_transfer") out = await aiFeatures.styleTransfer(imgBuffer, params?.style || "realistic");
        else out = await imageProcessor.applyOperations(imgBuffer, [{ type: operation, params: params || {} }]);
        const url = normalizeImageResult(out);
        if (!url) throw new Error("empty result from image provider");
        results.push({ index: i, ok: true, result: url });
        if (providerOp) debitCredits(uid, "image", `batch ${operation}`); // post-success per delivered frame
      } catch (e) {
        results.push({ index: i, ok: false, error: e.message });
      }
      db.prepare("UPDATE creative_batch_jobs SET completed_count = ? WHERE id = ?").run(i + 1, jobId);
    }
    db.prepare("UPDATE creative_batch_jobs SET status = 'completed' WHERE id = ?").run(jobId);
    res.json({ ok: true, job_id: jobId, status: "completed", results });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Batch status: the client polls this after POST /images/batch
// (NetworkClient.getBatchStatus). Batches run synchronously today so the row
// is normally already "completed" — the route exists so the poll never 404s.
app.get("/creative-suite/images/batch/:id/status", requireAuth, limitNormal, (req, res) => {
  const row = db.prepare("SELECT * FROM creative_batch_jobs WHERE id = ?").get(req.params.id);
  if (!row) return res.status(404).json({ error: "job not found" });
  if (req.authUser && row.user_id && row.user_id !== req.authUser) return res.status(403).json({ error: "not your job" });
  res.json({ ok: true, job_id: row.id, status: row.status, operation: row.operation, completed_count: row.completed_count, total_count: row.total_count, created_at: row.created_at });
});

app.post("/creative-suite/videos/generate", requireAuth, limitCostly, async (req, res) => {
  try {
    const { prompt, duration, resolution, aspect_ratio, model, brandkit_id } = req.body || {};
    if (!prompt) return res.status(400).json({ error: "prompt required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "video");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    // F-13: brand lock applied server-side so every studio honours it.
    const { brand, prompt: lockedPrompt } = applyBrand(uid, brandkit_id, prompt);
    const job = await submitVideo(lockedPrompt, { duration, resolution, aspect_ratio, model });
    const id = `vid_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt) VALUES (?, ?, 'video', 'submitted', ?)").run(id, uid, String(lockedPrompt).slice(0, 1000));
    db.prepare("UPDATE usage_daily SET clips = clips + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    trackProviderJob(job?.id || job?.data?.id, uid, "clip"); // charged once on completion
    res.json({ ok: true, job_id: id, brand_applied: !!brand, provider_job: job });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Video edit helpers: base64-or-URL -> sandbox-sized base64, and the
// sandbox ffmpeg call itself. Ops chain: each op's MP4 feeds the next.
// F-12b: SSRF guard + media sniffing for the video/audio ingest paths.
// Blocks loopback/private/link-local hosts (cloud metadata 169.254.169.254, LAN,
// localhost). Redirects are refused so a public URL can't bounce to an internal one.
// Known gap: DNS rebinding (hostname resolving to a private IP) is not checked.
function isPrivateHost(host) {
  const h = String(host || "").toLowerCase().replace(/^\[|\]$/g, "");
  if (!h) return true;
  if (h === "localhost" || h.endsWith(".localhost") || h.endsWith(".internal") || h.endsWith(".local")) return true;
  if (h.includes(":")) return h === "::1" || h === "::" || /^(fc|fd|fe80)/.test(h) || h.startsWith("::ffff:");
  if (/^(127\.|10\.|0\.|169\.254\.|192\.168\.)/.test(h)) return true;
  const m = h.match(/^172\.(\d+)\./);
  return !!(m && +m[1] >= 16 && +m[1] <= 31);
}
// mp4/mov/m4a (ftyp), webm/mkv (EBML), avi/wav (RIFF), ogg, mp3 (ID3 / frame sync), flac, aac(ADTS)
function looksLikeMedia(buf) {
  if (!buf || buf.length < 12) return false;
  const a4 = buf.toString("ascii", 4, 8);
  if (a4 === "ftyp") return true;
  if (buf[0] === 0x1a && buf[1] === 0x45 && buf[2] === 0xdf && buf[3] === 0xa3) return true;
  const a0 = buf.toString("ascii", 0, 4);
  if (a0 === "RIFF" || a0 === "OggS" || a0 === "fLaC" || buf.toString("ascii", 0, 3) === "ID3") return true;
  return buf[0] === 0xff && (buf[1] & 0xe0) === 0xe0;
}
async function mediaToB64(s, maxBytes = 8 * 1024 * 1024) {
  const str = String(s || "");
  let buf;
  if (/^https?:\/\//i.test(str)) {
    let host = "";
    try { host = new URL(str).hostname; } catch { throw new Error("invalid media URL"); }
    if (isPrivateHost(host)) throw new Error("media URL host not allowed");
    const r = await fetch(str, { signal: AbortSignal.timeout(120_000), redirect: "error" });
    if (!r.ok) throw new Error(`media fetch HTTP ${r.status}`);
    buf = Buffer.from(await r.arrayBuffer());
  } else {
    buf = Buffer.from(str.replace(/^data:[\w/;-]+;base64,/, ""), "base64");
  }
  if (!buf.length || buf.length > maxBytes) throw new Error(`media empty or >${Math.round(maxBytes / 1024 / 1024)}MB (trim the source first)`);
  if (!looksLikeMedia(buf)) throw new Error("unrecognized media format (expected mp4/webm/mov/mp3/wav/ogg/flac)");
  return buf.toString("base64");
}
async function sandboxMedia(payload) {
  const r = await fetch(`${SANDBOX_URL}/media/assemble`, {
    method: "POST",
    headers: sandboxAuthHeaders(),
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(180_000),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || !data?.base64) throw new Error(data?.error || `media assemble HTTP ${r.status}`);
  return data;
}
// Real timeline ops: trim | replace_audio | subtitles_burn | concat.
// Single call chains ops in order (each MP4 feeds the next).
app.post("/creative-suite/videos/edit", requireAuth, limitCostly, async (req, res) => {
  try {
    const { video, audio, subtitles, clips, operations } = req.body || {};
    if (!video && !(Array.isArray(clips) && clips.length)) return res.status(400).json({ error: "video or clips[] required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "video");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    let current = video ? await mediaToB64(video) : null;
    const applied = [];
    for (const op of (Array.isArray(operations) ? operations : [])) {
      const p = op.params || {};
      if (op.type === "trim") {
        if (!current) throw new Error("trim needs a video first");
        const out = await sandboxMedia({ op: "trim", files: [{ base64: current }], start_s: p.start_s ?? 0, end_s: p.end_s ?? 0 });
        current = out.base64; applied.push({ type: "trim", ok: true, size: out.size });
      } else if (op.type === "replace_audio") {
        if (!current) throw new Error("replace_audio needs a video first");
        const ab = audio || p.audio;
        if (!ab) throw new Error("replace_audio needs audio");
        const out = await sandboxMedia({ op: "replace_audio", files: [{ base64: current }, { base64: await mediaToB64(ab) }] });
        current = out.base64; applied.push({ type: "replace_audio", ok: true, size: out.size });
      } else if (op.type === "subtitles_burn") {
        if (!current) throw new Error("subtitles_burn needs a video first");
        const subs = Array.isArray(subtitles) && subtitles.length ? subtitles : p.subtitles;
        if (!Array.isArray(subs) || !subs.length) throw new Error("subtitles_burn needs subtitles[]");
        const out = await sandboxMedia({ op: "subtitles", files: [{ base64: current }], subtitles: subs });
        current = out.base64; applied.push({ type: "subtitles_burn", ok: true, size: out.size });
      } else if (op.type === "concat") {
        const list = Array.isArray(clips) && clips.length ? clips : (Array.isArray(p.clips) ? p.clips : []);
        if (list.length < 1) throw new Error("concat needs clips[] (2-6 videos)");
        const b64s = [];
        for (const c of list.slice(0, 6)) b64s.push({ base64: await mediaToB64(c) });
        const out = await sandboxMedia({ op: "concat", files: b64s });
        current = out.base64; applied.push({ type: "concat", ok: true, size: out.size, clips: b64s.length });
      } else {
        applied.push({ type: op.type || "unknown", ok: false, error: "unsupported op (use trim|replace_audio|subtitles_burn|concat)" });
      }
    }
    if (!current) return res.status(400).json({ error: "no operations applied — send operations[]" });
    const id = `vedit_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt) VALUES (?, ?, 'video_edit', 'done', ?)").run(id, uid, `ops=${applied.map(a => a.type).join(",")}`);
    // Local ffmpeg work (own sandbox, no provider video model) — compute rate, not clip rate.
    debitCredits(uid, "agent_task", "video edit");
    res.json({ ok: true, job_id: id, applied, mime: "video/mp4", video_base64: current });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Live capability table: which video models support what (durations,
// resolutions, frame control, references) — the app reads this to route
// Genjutsu/Cinema to a capable model instead of guessing.
app.get("/creative-suite/videos/models", requireAuth, limitNormal, async (_req, res) => {
  try {
    const r = await fetch("https://openrouter.ai/api/v1/videos/models", {
      headers: orVideoHeaders(),
      signal: AbortSignal.timeout(30_000),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data?.error?.message || `models HTTP ${r.status}`);
      res.json({ ok: true, defaults: { text_to_video: OR_VIDEO_MODEL, genjutsu: OR_GENJUTSU_MODEL, avatar: OR_AVATAR_MODEL, avatar_fallback: HEDRA_API_KEY ? "hedra-character-3" : "local" }, models: data.data || data.models || [] });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/videos/subtitles", requireAuth, limitCostly, async (req, res) => {
  try {
    const { audio, video, language } = req.body || {};
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "voice");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    // Accept either raw audio or a video file (extract audio track via sandbox ffmpeg)
    let audioBuffer;
    if (audio) {
      audioBuffer = Buffer.from(audio.replace(/^data:[\w/;-]+;base64,/, ""), "base64");
      if (audioBuffer.length > 15 * 1024 * 1024 || !looksLikeMedia(audioBuffer)) return res.status(400).json({ error: "invalid or oversized audio (max 15MB)" });
    } else if (video) {
      const videoBuf = Buffer.from(video.replace(/^data:[\w/;-]+;base64,/, ""), "base64");
      if (videoBuf.length > 30 * 1024 * 1024 || !looksLikeMedia(videoBuf)) return res.status(400).json({ error: "invalid or oversized video (max 30MB)" });
      const r = await fetch(`${SANDBOX_URL}/media/extract-audio`, {
        method: "POST",
        headers: sandboxAuthHeaders(),
        body: JSON.stringify({ video_base64: videoBuf.toString("base64") }),
        signal: AbortSignal.timeout(60_000),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok || !data?.audio_base64) throw new Error(data?.error || `audio extract HTTP ${r.status}`);
      audioBuffer = Buffer.from(data.audio_base64, "base64");
    } else {
      return res.status(400).json({ error: "audio or video required" });
    }
    const result = await aiFeatures.generateSubtitles(audioBuffer, language || "auto");
    if (result.text) debitCredits(uid, "stt_min", "subtitles");
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/videos/translate-subtitles", requireAuth, limitNormal, async (req, res) => {
  try {
    const { subtitles, target_language } = req.body || {};
    if (!subtitles || !target_language) return res.status(400).json({ error: "subtitles and target_language required" });
    if (!Array.isArray(subtitles) || subtitles.length === 0) {
      return res.status(400).json({ error: "subtitles array must not be empty" });
    }
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "creative");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const result = await aiFeatures.translateSubtitles(subtitles, target_language);
    debitCredits(uid, "turn", "translate subtitles");
    res.json({ ok: true, subtitles: result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/storyboard", requireAuth, limitCostly, async (req, res) => {
  try {
    const { script, style, num_frames } = req.body || {};
    if (!script) return res.status(400).json({ error: "script required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const frames = await aiFeatures.storyboard(script, style, num_frames || 6);
    // Parallel fan-out: 8 sequential generations took ~8x single latency.
    const generatedFrames = await Promise.all(frames.slice(0, 8).map(async (frame) => {
      try {
        const img = await generateImage(frame.prompt, { model: OR_IMAGE_MODEL });
        return { ...frame, image_url: img.image_url || (img.image_b64 ? `data:image/png;base64,${img.image_b64}` : null) };
      } catch (e) {
        return { ...frame, image_url: null, error: e.message };
      }
    }));
    // Pay per delivered frame only — failed frames are not billed.
    for (const f of generatedFrames) if (f.image_url) debitCredits(uid, "image", "storyboard frame");
    res.json({ ok: true, frames: generatedFrames });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.get("/creative-suite/templates", requireAuth, limitNormal, (req, res) => {
  const { category, platform } = req.query;
  const templates = templateLibrary.getTemplates(category, platform);
  res.json({ ok: true, templates });
});

app.post("/creative-suite/templates/:id/apply", requireAuth, limitNormal, async (req, res) => {
  try {
    const { assets } = req.body || {};
    const result = templateLibrary.applyTemplate(req.params.id, assets || []);
    if (result.error) return res.status(404).json({ ok: false, error: result.error });
    // Real render: lay the attached images into their zones on a canvas
    // sized to the template's aspect ratio.
    const [w, h] = imageProcessor.dimsForAspect(result.aspect_ratio, 1080);
    const png = await imageProcessor.renderZones(w, h, result.zones || []);
    res.json({
      ok: true, ...result,
      width: w, height: h,
      image_b64: png.toString("base64"),
      image_url: `data:image/png;base64,${png.toString("base64")}`,
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.get("/creative-suite/style-presets", requireAuth, limitNormal, (_req, res) => {
  res.json({ ok: true, presets: promptEngine.STYLE_PRESETS });
});

app.post("/creative-suite/prompt-suggest", requireAuth, limitNormal, async (req, res) => {
  try {
    const { idea, context } = req.body || {};
    if (!idea) return res.status(400).json({ error: "idea required" });
    const suggestions = await promptEngine.suggestPrompts(idea, context);
    res.json({ ok: true, suggestions });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/prompt-enhance", requireAuth, limitNormal, async (req, res) => {
  try {
    const { prompt, context } = req.body || {};
    if (!prompt) return res.status(400).json({ error: "prompt required" });
    const result = await promptEngine.enhancePrompt(prompt, context);
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/recommend", requireAuth, limitCostly, async (req, res) => {
  try {
    const { image, type } = req.body || {};
    if (!image) return res.status(400).json({ error: "image required" });
    const imageBuffer = Buffer.from(image.replace(/^data:image\/\w+;base64,/, ""), "base64");
    const analysis = await aiFeatures.analyzeImage(imageBuffer);
    const recommendations = await promptEngine.recommendStyles(analysis);
    res.json({ ok: true, recommendations });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/projects", requireAuth, limitNormal, (req, res) => {
  try {
    const { name, type, metadata } = req.body || {};
    if (!name) return res.status(400).json({ error: "name required" });
    const uid = boundUser(req);
    const id = `proj_${Date.now()}`;
    db.prepare("INSERT INTO creative_projects (id, user_id, name, type, metadata) VALUES (?, ?, ?, ?, ?)")
      .run(id, uid, name, type || "image", JSON.stringify(metadata || {}));
    res.json({ ok: true, project_id: id });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.get("/creative-suite/projects", requireAuth, limitNormal, (req, res) => {
  const uid = boundUser(req);
  const projects = db.prepare("SELECT p.id, p.name, p.type, p.status, p.thumbnail_url, p.metadata, p.updated_at, (SELECT COUNT(*) FROM creative_assets a WHERE a.project_id = p.id) AS asset_count FROM creative_projects p WHERE p.user_id = ? ORDER BY p.updated_at DESC LIMIT 100").all(uid);
  res.json({ ok: true, projects });
});

app.get("/creative-suite/projects/:id", requireAuth, limitNormal, (req, res) => {
  const uid = boundUser(req);
  const project = db.prepare("SELECT * FROM creative_projects WHERE id = ?").get(req.params.id);
  if (!project) return res.status(404).json({ error: "project not found" });
  if (req.authUser && project.user_id && project.user_id !== req.authUser) return res.status(403).json({ error: "not your project" });
  const assets = db.prepare("SELECT * FROM creative_assets WHERE project_id = ? ORDER BY created_at DESC").all(req.params.id);
  res.json({ ok: true, project: { ...project, assets } });
});

app.put("/creative-suite/projects/:id", requireAuth, limitNormal, (req, res) => {
  try {
    const { name, metadata, status } = req.body || {};
    const existing = db.prepare("SELECT * FROM creative_projects WHERE id = ?").get(req.params.id);
    if (!existing) return res.status(404).json({ error: "project not found" });
    if (req.authUser && existing.user_id && existing.user_id !== req.authUser) return res.status(403).json({ error: "not your project" });
    if (name) db.prepare("UPDATE creative_projects SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?").run(name, req.params.id);
    if (metadata) db.prepare("UPDATE creative_projects SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?").run(JSON.stringify(metadata), req.params.id);
    if (status) db.prepare("UPDATE creative_projects SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?").run(status, req.params.id);
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.delete("/creative-suite/projects/:id", requireAuth, limitNormal, (req, res) => {
  const existing = db.prepare("SELECT * FROM creative_projects WHERE id = ?").get(req.params.id);
  if (!existing) return res.status(404).json({ error: "project not found" });
  if (req.authUser && existing.user_id && existing.user_id !== req.authUser) return res.status(403).json({ error: "not your project" });
  // Week 6 (F-46): creative_collaborators + creative_sync_state carry a
  // FOREIGN KEY to creative_projects(id). Any project that had been synced
  // (the app syncs on open/edits) therefore failed to delete with
  // "FOREIGN KEY constraint failed" -> HTTP 500. Delete children first, in one
  // transaction, so a half-deleted project can never survive.
  try {
    db.transaction(() => {
      db.prepare("DELETE FROM creative_assets WHERE project_id = ?").run(req.params.id);
      db.prepare("DELETE FROM creative_collaborators WHERE project_id = ?").run(req.params.id);
      db.prepare("DELETE FROM creative_sync_state WHERE project_id = ?").run(req.params.id);
      db.prepare("DELETE FROM creative_projects WHERE id = ?").run(req.params.id);
    })();
    res.json({ ok: true });
  } catch (err) {
    console.error("[/creative-suite/projects/:id DELETE]", err.message);
    res.status(500).json({ error: "project delete failed", details: err.message });
  }
});

app.post("/creative-suite/projects/:id/assets", requireAuth, limitNormal, (req, res) => {
  try {
    const { kind, source, url, metadata } = req.body || {};
    if (!kind || !url) return res.status(400).json({ error: "kind and url required" });
    const uid = boundUser(req);
    const proj = db.prepare("SELECT user_id FROM creative_projects WHERE id = ?").get(req.params.id);
    if (!proj) return res.status(404).json({ error: "project not found" });
    if (req.authUser && proj.user_id && proj.user_id !== req.authUser) return res.status(403).json({ error: "not your project" });
    const id = `asset_${Date.now()}`;
    db.prepare("INSERT INTO creative_assets (id, project_id, user_id, kind, source, current_url, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)")
      .run(id, req.params.id, uid, kind, source || "uploaded", url, JSON.stringify(metadata || {}));
    res.json({ ok: true, asset_id: id });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/projects/:id/sync", requireAuth, limitNormal, (req, res) => {
  try {
    const { device_id, sync_data } = req.body || {};
    db.prepare("INSERT INTO creative_sync_state (project_id, user_id, device_id, last_sync_at, sync_data) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?) ON CONFLICT(project_id) DO UPDATE SET device_id = ?, last_sync_at = CURRENT_TIMESTAMP, sync_data = ?")
      .run(req.params.id, boundUser(req), device_id, JSON.stringify(sync_data || {}), device_id, JSON.stringify(sync_data || {}));
    res.json({ ok: true, last_sync_at: new Date().toISOString() });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.get("/creative-suite/projects/:id/sync", requireAuth, limitNormal, (req, res) => {
  const row = db.prepare("SELECT * FROM creative_sync_state WHERE project_id = ?").get(req.params.id);
  res.json({ ok: true, sync_data: row ? JSON.parse(row.sync_data || "{}") : null, last_sync_at: row?.last_sync_at || null });
});

app.post("/creative-suite/projects/:id/collaborators", requireAuth, limitNormal, (req, res) => {
  try {
    const { user_id, role } = req.body || {};
    if (!user_id) return res.status(400).json({ error: "user_id required" });
    db.prepare("INSERT INTO creative_collaborators (project_id, owner_user_id, collaborator_user_id, role) VALUES (?, ?, ?, ?)")
      .run(req.params.id, boundUser(req), user_id, role || "editor");
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.get("/creative-suite/projects/:id/collaborators", requireAuth, limitNormal, (req, res) => {
  const collabs = db.prepare("SELECT * FROM creative_collaborators WHERE project_id = ?").all(req.params.id);
  res.json({ ok: true, collaborators: collabs });
});

app.post("/creative-suite/export", requireAuth, limitCostly, async (req, res) => {
  try {
    const { asset_url, platform, options } = req.body || {};
    if (!asset_url || !platform) return res.status(400).json({ error: "asset_url and platform required" });
    const preset = templateLibrary.EXPORT_PRESETS[platform];
    if (!preset) return res.status(400).json({ error: "unsupported platform" });
    const uid = boundUser(req);
    const isImage = /^data:image\//i.test(asset_url) || /\.(png|jpe?g|webp|gif)(\?|$)/i.test(asset_url);
    const isVideo = /^data:video\//i.test(asset_url) || /\.(mp4|mov|webm|m4v)(\?|$)/i.test(asset_url);
    if (isImage) {
      const ent = checkEntitlement(uid, "image");
      if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
      let buf;
      if (/^https?:\/\//i.test(asset_url)) {
        const r = await fetch(asset_url, { signal: AbortSignal.timeout(60_000) });
        if (!r.ok) throw new Error(`asset fetch HTTP ${r.status}`);
        buf = Buffer.from(await r.arrayBuffer());
      } else {
        buf = Buffer.from(String(asset_url).replace(/^data:image\/\w+;base64,/, ""), "base64");
      }
      const [w, h] = preset.resolution.split("x").map(Number);
      const out = await imageProcessor.resize(buf, w, h, "cover");
      const meta = await imageProcessor.getImageMetadata(out);
      debitCredits(uid, "image", `export ${platform}`);
      return res.json({ ok: true, platform, preset, result: `data:image/png;base64,${out.toString("base64")}`, mime: "image/png", transformed: true, ...meta });
    }
    // Video: re-encode to the platform canvas via the sandbox ffmpeg (F-13).
    // The old behaviour returned the source clip untouched with a note that
    // re-encoding "isn't available here" — so a 4:5 clip shared to TikTok
    // stayed 4:5 and got letterboxed/cropped by the platform instead.
    if (isVideo) {
      const ent = checkEntitlement(uid, "video");
      if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
      const [w, h] = preset.resolution.split("x").map(Number);
      const fit = options?.fit === "contain" ? "contain" : "cover";
      let b64;
      try {
        b64 = await mediaToB64(asset_url);
      } catch (e) {
        return res.status(400).json({ error: `cannot read source video: ${e.message}` });
      }
      const out = await sandboxMedia({
        op: "export",
        files: [{ base64: b64 }],
        width: w,
        height: h,
        fps: preset.fps || 30,
        max_duration: preset.max_duration || 0,
        fit,
      });
      debitCredits(uid, "agent_task", `export ${platform}`);
      return res.json({
        ok: true, platform, preset, transformed: true,
        mime: "video/mp4", size: out.size, video_base64: out.base64,
        width: w, height: h, fps: preset.fps || 30,
        note: `Re-encoded to ${preset.resolution} @ ${preset.fps || 30}fps${preset.max_duration ? ` (max ${preset.max_duration}s)` : ""}.`,
      });
    }
    res.json({ ok: true, platform, preset, result: asset_url, transformed: false, note: "Unrecognized asset type — returning as-is." });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ==================== CINEMA STUDIO / KATANA / GENJUTSU / SOUL ID ====================
// Director's Panel option tables (genre, era, tempo, camera, lens, aperture,
// moves, angles, lighting, palettes, emotions, aspect ratios).
app.get("/creative-suite/cinema/options", requireAuth, limitNormal, (_req, res) => {
  res.json({ ok: true, options: cinema.CINEMA_OPTIONS });
});

// AI Director: brief -> shot list with pre-filled cinematography params.
app.post("/creative-suite/cinema/direct", requireAuth, limitCostly, async (req, res) => {
  try {
    const { brief, num_shots } = req.body || {};
    if (!brief) return res.status(400).json({ error: "brief required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "creative");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const shots = await cinema.aiDirect(brief, Math.min(8, Math.max(1, Number(num_shots) || 4)));
    debitCredits(uid, "turn", "cinema direct");
    res.json({ ok: true, shots });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Cinema generate: prompt + Director's Panel params + optional character/
// location references -> image (sync) or video (async job). References are
// base64 data URLs or http URLs (max 8 per call).
app.post("/creative-suite/cinema/generate", requireAuth, limitCostly, async (req, res) => {
  try {
    const { prompt, kind, director, references, aspect_ratio, duration, resolution, brandkit_id } = req.body || {};
    if (!prompt) return res.status(400).json({ error: "prompt required" });
    const uid = boundUser(req);
    const refs = Array.isArray(references) ? references.slice(0, 8) : [];
    const inputRefs = refs.map(r => ({ type: "image_url", image_url: { url: String(r) } }));
    // F-13: brand lock applied server-side so every studio honours it.
    const { brand, prompt: lockedPrompt } = applyBrand(uid, brandkit_id, prompt);
    const directed = cinema.buildDirectorPrompt({ prompt: lockedPrompt, ...(director || {}) });
    if ((kind || "image") === "video") {
    const ent = checkEntitlement(uid, "video");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
      const job = await submitVideo(directed, {
        duration: duration || 4,
        resolution: resolution || "720p",
        aspect_ratio: aspect_ratio || "16:9",
        ...(inputRefs.length ? { input_references: inputRefs } : {}),
      });
      const id = `vid_${Date.now()}`;
      db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt) VALUES (?, ?, 'video', 'submitted', ?)").run(id, uid, directed.slice(0, 1000));
      db.prepare("UPDATE usage_daily SET clips = clips + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
      trackProviderJob(job?.id || job?.data?.id, uid, "clip"); // charged once on completion (was unbilled before)
      return res.json({ ok: true, kind: "video", job_id: id, brand_applied: !!brand, provider_job: job });
    }
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const data = await generateImage(directed, {
      aspect_ratio: aspect_ratio || undefined,
      ...(inputRefs.length ? { input_references: inputRefs } : {}),
    });
    db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    debitCredits(uid, "image", "cinema image"); // post-success only (was pre-charged before)
    res.json({ ok: true, kind: "image", directed_prompt: directed, brand_applied: !!brand, ...data });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Katana: one natural-language command -> parsed ops -> executed pipeline.
// Body: { command, video? (base64/url), audio? (base64) }. Returns ops +
// per-op results (captions generated via Whisper, trims via sandbox, etc).
app.post("/creative-suite/katana", requireAuth, limitCostly, async (req, res) => {
  try {
    const { command, video, audio, language } = req.body || {};
    if (!command) return res.status(400).json({ error: "command required" });
    const katanaUid = boundUser(req);
    const katanaEnt = checkEntitlement(katanaUid, "chat");
    if (!katanaEnt.ok) return res.status(402).json({ error: katanaEnt.message, code: katanaEnt.code, needed: katanaEnt.needed, balance: katanaEnt.balance });
    const { ops, reply } = await cinema.katanaParse(command);
    const results = [];
    // Chain video ops: each successful op's MP4 feeds the next, so
    // "trim 0-5 then replace audio" works in one command.
    let current = video ? await mediaToB64(video) : null;
    let executed = 0;
    for (const op of ops) {
      try {
        if (op.type === "captions" || op.type === "translate_captions") {
          const srcForCaptions = audio || current || video;
          if (!srcForCaptions) { results.push({ type: op.type, ok: false, error: "audio or video required for captions" }); continue; }
          const buf = Buffer.from(String(srcForCaptions).replace(/^data:[\w/;-]+;base64,/, ""), "base64");
          const subs = await aiFeatures.generateSubtitles(buf, op.params?.language || language || "auto");
          let out = subs.subtitles || [];
          if (op.type === "translate_captions" && op.params?.target_language) {
            out = await aiFeatures.translateSubtitles(out, op.params.target_language);
          }
          results.push({ type: op.type, ok: true, subtitles: out });
          executed++;
        } else if (op.type === "export") {
          results.push({ type: op.type, ok: true, preset: templateLibrary.EXPORT_PRESETS[op.params?.platform] || null });
          executed++;
        } else if (op.type === "trim") {
          if (!current) { results.push({ type: op.type, ok: false, error: "trim needs a video first" }); continue; }
          const out = await sandboxMedia({ op: "trim", files: [{ base64: current }], start_s: op.params?.start_s ?? 0, end_s: op.params?.end_s ?? 0 });
          current = out.base64;
          results.push({ type: op.type, ok: true, size: out.size, video_base64: out.base64 });
          executed++;
        } else if (op.type === "replace_audio") {
          if (!current) { results.push({ type: op.type, ok: false, error: "replace_audio needs a video first" }); continue; }
          const ab = audio || op.params?.audio;
          if (!ab) { results.push({ type: op.type, ok: false, error: "replace_audio needs audio" }); continue; }
          const out = await sandboxMedia({ op: "replace_audio", files: [{ base64: current }, { base64: await mediaToB64(ab) }] });
          current = out.base64;
          results.push({ type: op.type, ok: true, size: out.size, video_base64: out.base64 });
          executed++;
        } else {
          // Honest failure: don't claim a render happened when it didn't.
          results.push({ type: op.type || "unknown", ok: false, error: "not supported in Katana yet — use Video Editor for this op", params: op.params || {} });
        }
      } catch (e) {
        results.push({ type: op.type, ok: false, error: e.message });
      }
    }
    const id = `kat_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt, result_url) VALUES (?, ?, 'katana', 'done', ?, ?)").run(id, boundUser(req), String(command).slice(0, 500), JSON.stringify(results).slice(0, 4000));
    if (executed > 0) debitCredits(katanaUid, "turn", "katana"); // post-success: at least one op ran
    res.json({ ok: true, job_id: id, ops, reply, results, video_base64: current });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Genjutsu: Motion Transfer (keep motion/camera/timing, rebuild scene from
// references) or Object Swap (swap one element, keep the rest). Implemented
// on the video model via frame_images + input_references so motion/timing
// carry over from the source clip.
app.post("/creative-suite/genjutsu", requireAuth, limitCostly, async (req, res) => {
  try {
    const { mode, video, references, description, preset, duration, resolution, model } = req.body || {};
    if (!video) return res.status(400).json({ error: "video (base64 or http URL) required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "video");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const m = (mode || "motion_transfer").toLowerCase();
    const refs = Array.isArray(references) ? references.slice(0, 30) : [];
    const prompt = m === "object_swap"
      ? `Object swap: ${description || preset || "swap the specified element"}. Preserve everything else in the shot exactly — same motion, camera, timing, lighting.`
      : `Motion transfer: keep the source motion, camera work and timing exactly. Rebuild the scene: ${description || preset || "new cast, location and look"}. ${refs.length ? "Match the provided character/location references in every frame." : ""}`;
    const s = String(video);
    // OpenRouter shape: frame_images entries need {frame_type: first_frame|last_frame};
    // source CLIPS + style refs ride as multimodal input_references (Seedance-class).
    const videoRef = { type: "video_url", video_url: { url: s } };
    const refObjs = refs.map(r => ({ type: "image_url", image_url: { url: String(r) } }));
    const job = await submitVideo(prompt, {
      model: model || OR_GENJUTSU_MODEL,
      duration: duration || 4,
      resolution: resolution || "720p",
      input_references: [videoRef, ...refObjs],
    });
    const id = `gen_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt) VALUES (?, ?, 'genjutsu', 'submitted', ?)").run(id, uid, prompt.slice(0, 1000));
    db.prepare("UPDATE usage_daily SET clips = clips + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    trackProviderJob(job?.id || job?.data?.id, uid, "clip"); // charged once on completion
    res.json({ ok: true, job_id: id, mode: m, model: model || OR_GENJUTSU_MODEL, provider_job: job });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Soul ID / Elements: reusable characters, locations, props for consistency.
app.post("/creative-suite/elements", requireAuth, limitNormal, (req, res) => {
  try {
    const { name, type, references, metadata } = req.body || {};
    if (!name) return res.status(400).json({ error: "name required" });
    const id = `el_${Date.now().toString(36)}`;
    db.prepare("INSERT INTO creative_elements (id, user_id, type, name, reference_urls, metadata) VALUES (?, ?, ?, ?, ?, ?)")
      .run(id, boundUser(req), type || "character", name, JSON.stringify(Array.isArray(references) ? references.slice(0, 30) : []), JSON.stringify(metadata || {}));
    res.json({ ok: true, element_id: id });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});
app.get("/creative-suite/elements", requireAuth, limitNormal, (req, res) => {
  const uid = boundUser(req);
  const { type } = req.query || {};
  const rows = type
    ? db.prepare("SELECT * FROM creative_elements WHERE user_id = ? AND type = ? ORDER BY created_at DESC LIMIT 100").all(uid, type)
    : db.prepare("SELECT * FROM creative_elements WHERE user_id = ? ORDER BY created_at DESC LIMIT 100").all(uid);
  res.json({ ok: true, elements: rows.map(r => ({ ...r, reference_urls: JSON.parse(r.reference_urls || "[]"), metadata: JSON.parse(r.metadata || "{}") })) });
});
app.post("/creative-suite/elements/:id/delete", requireAuth, limitNormal, (req, res) => {
  db.prepare("DELETE FROM creative_elements WHERE id = ? AND user_id = ?").run(req.params.id, boundUser(req));
  res.json({ ok: true });
});

// AI Influencer: attributes -> persona + consistent character sheet + post batch.
app.post("/creative-suite/influencer", requireAuth, limitCostly, async (req, res) => {
  try {
    const { attributes, posts } = req.body || {};
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const persona = await cinema.influencerPersona(attributes || {});
    let sheet = null;
    if (persona.character_prompt) {
      try {
        sheet = await generateImage(`Character reference sheet, same person three angles (front, profile, three-quarter), neutral background. ${persona.character_prompt}`, {});
        db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
      } catch (e) { sheet = { error: e.message }; }
    }
    // Post batch: first 3 visuals generated in parallel against the SAME
    // character sheet reference (that is what keeps one consistent face).
    // Remaining ideas ship as ready prompts so the client can render on demand.
    const ideas = Array.isArray(persona.post_ideas) ? persona.post_ideas : [];
    const sheetRef = sheet?.image_url
      ? [{ type: "image_url", image_url: { url: sheet.image_url } }]
      : [];
    const visualCount = Math.min(3, ideas.length);
    const visuals = await Promise.all(ideas.slice(0, visualCount).map(async (idea) => {
      try {
        const img = await generateImage(
          `Same person as reference, photorealistic lifestyle photo. ${idea.visual_prompt || idea.caption || ""} ${persona.character_prompt || ""}`,
          sheetRef.length ? { input_references: sheetRef } : {}
        );
        db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
        return { caption: idea.caption || "", image_url: img.image_url || null, image_b64: img.image_b64 || null };
      } catch (e) {
        return { caption: idea.caption || "", image_url: null, error: e.message };
      }
    }));
    const id = `inf_${Date.now()}`;
    db.prepare("INSERT INTO creative_elements (id, user_id, type, name, reference_urls, metadata) VALUES (?, ?, 'character', ?, ?, ?)")
      .run(id, uid, persona.name || "Influencer", JSON.stringify(sheet?.image_url ? [sheet.image_url] : []), JSON.stringify({ persona, character_prompt: persona.character_prompt || "" }));
    // Pay per delivered visual only (sheet + successful post visuals).
    if (sheet?.image_url) debitCredits(uid, "image", "influencer sheet");
    for (const v of visuals) if (v.image_url) debitCredits(uid, "image", "influencer post");
    res.json({ ok: true, element_id: id, persona, character_sheet: sheet, posts: visuals, queued_prompts: ideas.slice(visualCount).map(i => ({ caption: i.caption || "", visual_prompt: i.visual_prompt || "" })) });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Ads (Marketing Studio): product + brief -> script + shots + static ad poster.
app.post("/creative-suite/ads", requireAuth, limitCostly, async (req, res) => {
  try {
    const { brief, product, product_image } = req.body || {};
    if (!brief) return res.status(400).json({ error: "brief required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const plan = await cinema.adPlan(brief, product || "");
    let poster = null;
    try {
      const refs = product_image ? [{ type: "image_url", image_url: { url: String(product_image) } }] : [];
      poster = await generateImage(`Premium product advertisement poster, studio lighting, 1:1. ${plan.static_ad_prompt || brief}. Product must match reference exactly.`, refs.length ? { input_references: refs } : {});
      db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    } catch (e) { poster = { error: e.message }; }
    // Plan call (~1cr) + delivered poster only — failed posters are not billed.
    debitCredits(uid, "turn", "ads plan");
    if (poster?.image_url || poster?.image_b64) debitCredits(uid, "image", "ads poster");
    const id = `ad_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt, result_url) VALUES (?, ?, 'ad', 'done', ?, ?)").run(id, uid, String(brief).slice(0, 500), JSON.stringify(plan).slice(0, 4000));
    res.json({ ok: true, job_id: id, plan, poster });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Lip-sync, two engines:
// A) Neural (Sync Labs sync-2.0): real mouth re-animation. Needs SYNC_LABS_API_KEY
//    + PUBLIC_BASE_URL (webhook). Media is staged via Supabase signed URLs.
// B) Local (always available): transcript-timed captions + sandbox dub mix.
// The response always says which engine ran — never a silent downgrade.
async function supaStageMedia(buf, filename, mime) {
  if (!HAS_REMOTE_BACKUP) throw new Error("Supabase storage not configured");
  await supaBucket();
  const path = `lipsync/${Date.now()}-${filename}`.replace(/[^a-zA-Z0-9._/-]/g, "_");
  const up = await fetch(`${SUPABASE_URL}/storage/v1/object/${SUPABASE_BUCKET}/${path}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${SUPABASE_KEY}`, apikey: SUPABASE_KEY, "Content-Type": mime, "x-upsert": "true" },
    body: buf,
    signal: AbortSignal.timeout(120_000),
  });
  if (!up.ok) throw new Error(`media stage upload HTTP ${up.status}`);
  const sign = await fetch(`${SUPABASE_URL}/storage/v1/object/sign/${SUPABASE_BUCKET}/${path}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${SUPABASE_KEY}`, apikey: SUPABASE_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({ expiresIn: 3600 }),
    signal: AbortSignal.timeout(30_000),
  });
  const sj = await sign.json().catch(() => ({}));
  if (!sj?.signedURL) throw new Error("media stage sign failed");
  return `${SUPABASE_URL}${sj.signedURL}`;
}
app.post("/webhooks/lipsync", async (req, res) => {
  // Sync Labs completion webhook: ?job=<our creative_jobs id>&secret=<WEBHOOK_SECRET>.
  // This is the only completion signal for the neural engine, so the avatar
  // charge lands here (was charged at queue time — a failed render still
  // billed). The status UPDATE only fires while the job is still in flight,
  // and the charged flag in chargeProviderJob is idempotent, so replayed or
  // forged calls can never double-charge.
  try {
    // FIX (F-11): shared-secret verification. Without it, anyone who can reach
    // PUBLIC_BASE_URL could spoof a "done" webhook and force a premature charge
    // (or tamper with job status). When WEBHOOK_SECRET is set, the secret query
    // param is mandatory and must match. Legacy in-flight jobs registered
    // before the secret go stale — the provider_jobs completion sweeper
    // settles them instead.
    if (WEBHOOK_SECRET && String(req.query.secret || "") !== WEBHOOK_SECRET) {
      console.warn("[webhooks/lipsync] rejected — bad or missing secret");
      return res.status(401).json({ ok: false, error: "bad secret" });
    }
    const ours = String(req.query.job || "");
    const b = req.body || {};
    const url = b.videoUrl || b.video_url || b.output || b.result_url || null;
    const status = /fail|error/i.test(String(b.status || b.state || "")) ? "failed" : (url ? "done" : "processing");
    if (ours) {
      const upd = db.prepare("UPDATE creative_jobs SET status = ?, result_url = ? WHERE id = ? AND status NOT IN ('done', 'failed')").run(status, url ? String(url).slice(0, 1000) : null, ours);
      if (status === "done" && upd.changes) chargeProviderJob(`synclabs:${ours}`); // completion billing, exactly once
    }
    res.json({ ok: true });
  } catch (e) {
    res.status(200).json({ ok: false, error: e.message });
  }
});
app.post("/creative-suite/lipsync", requireAuth, limitCostly, async (req, res) => {
  try {
    const { video, image, audio, language, mode, aspect_ratio, avatar_prompt } = req.body || {};
    if (!audio) return res.status(400).json({ error: "audio required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "voice");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const id = `lip_${Date.now()}`;
    // Engine Y (PRIMARY avatar): HeyGen Avatar IV via OpenRouter /videos —
    // photo + voice -> expressive lip-synced talking head ($0.05/s, to 1080p).
    // Same key as everything else (no extra subscription). Any failure falls
    // through to Hedra, then local — the chain never breaks the request.
    if ((mode === "avatar" || (!video && image)) && image && HAS_OR) {
      try {
        const portrait = String(image).startsWith("data:")
          ? String(image)
          : `data:image/png;base64,${String(image).replace(/^data:[\w/;-]+;base64,/, "")}`;
        const job = await submitVideo(
          String(avatar_prompt || "A person speaking to the camera with natural micro-expressions, head motion and gestures matching the vocal emotion").slice(0, 1000),
          {
            model: OR_AVATAR_MODEL,
            duration: 8,
            resolution: "720p",
            aspect_ratio: aspect_ratio || "9:16",
            input_references: [{ type: "image_url", image_url: { url: portrait } }],
          }
        );
        const pid = job?.id || job?.data?.id;
        if (!pid) throw new Error("heygen submit returned no job id");
        db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt, result_url) VALUES (?, ?, 'lipsync', 'submitted', ?, ?)").run(id, uid, `heygen:${pid} lang=${language || "auto"}`, JSON.stringify(job).slice(0, 1000));
        trackProviderJob(pid, uid, "avatar"); // charged once on completion, never on failure
        return res.json({ ok: true, job_id: id, engine: "heygen", status_url: `/creative-suite/lipsync/${id}/status`, content_url: `/videos/content/${pid}`, note: "HeyGen talking-avatar render queued — poll status until done, then GET content_url." });
      } catch (e) {
        console.warn("[lipsync] heygen engine failed, falling back:", e.message);
      }
    }
    // Engine H (fallback avatar): Hedra Character 3 — one portrait + audio, no video needed.
    if ((mode === "avatar" || (!video && image)) && image && HEDRA_API_KEY && HAS_REMOTE_BACKUP) {
      try {
        const strip = (s) => Buffer.from(String(s).replace(/^data:[\w/;-]+;base64,/, ""), "base64");
        const [portraitUrl, audioUrl] = await Promise.all([
          supaStageMedia(strip(image), "portrait.png", "image/png"),
          supaStageMedia(strip(audio), "voice.mp3", "audio/mpeg"),
        ]);
        const sub = await hedraFetch("/v3/models/hedra-character-3", {
          method: "POST",
          body: {
            input: {
              aspect_ratio: aspect_ratio || "9:16",
              resolution: "720p",
              start_image: { source: "url", url: portraitUrl },
              audio: { source: "url", url: audioUrl },
              prompt: String(avatar_prompt || "A person speaking to the camera, natural micro-expressions").slice(0, 500),
            },
          },
        });
        const sd = await sub.json().catch(() => ({}));
        if (sub.status !== 202 && !sub.ok) throw new Error(sd?.message || sd?.error || `hedra HTTP ${sub.status}`);
        const hid = sd.id || sd.job_id || sd.jobId || null;
        if (!hid) throw new Error("hedra returned no job id");
        db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt, result_url) VALUES (?, ?, 'lipsync', 'submitted', ?, ?)").run(id, uid, `hedra:${hid} lang=${language || "auto"}`, JSON.stringify(sd).slice(0, 1000));
        trackProviderJob(hid, uid, "avatar", "hedra"); // charged once, on completion — never at queue time
        return res.json({ ok: true, job_id: id, engine: "hedra", status_url: `/creative-suite/lipsync/${id}/status`, note: "Talking-avatar render queued — poll status until done." });
      } catch (e) {
        console.warn("[lipsync] hedra engine failed, falling back:", e.message);
      }
    }
    // Engine A: neural lip-sync via Sync Labs (mouth re-animation).
    if (video && SYNC_LABS_KEY && PUBLIC_BASE_URL && HAS_REMOTE_BACKUP) {
      try {
        const strip = (s) => Buffer.from(String(s).replace(/^data:[\w/;-]+;base64,/, ""), "base64");
        const [videoUrl, audioUrl] = await Promise.all([
          supaStageMedia(strip(video), "source.mp4", "video/mp4"),
          supaStageMedia(strip(audio), "voice.mp3", "audio/mpeg"),
        ]);
        const sub = await fetch("https://api.synclabs.so/lipsync", {
          method: "POST",
          headers: { "x-api-key": SYNC_LABS_KEY, "Content-Type": "application/json" },
          body: JSON.stringify({
            videoUrl, audioUrl, model: "sync-2.0",
            // FIX (F-11): secret rides on the registered webhookUrl so the
            // provider echoes it back and the route can verify ownership.
            webhookUrl: `${PUBLIC_BASE_URL}/webhooks/lipsync?job=${id}${WEBHOOK_SECRET ? `&secret=${encodeURIComponent(WEBHOOK_SECRET)}` : ""}`,
          }),
          signal: AbortSignal.timeout(60_000),
        });
        const sd = await sub.json().catch(() => ({}));
        if (!sub.ok) throw new Error(sd?.message || sd?.error || `synclabs HTTP ${sub.status}`);
        db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt, result_url) VALUES (?, ?, 'lipsync', 'submitted', ?, ?)").run(id, uid, `neural lang=${language || "auto"}`, JSON.stringify(sd).slice(0, 1000));
        trackProviderJob(`synclabs:${id}`, uid, "avatar", "synclabs"); // charged once, when the completion webhook lands
        return res.json({ ok: true, job_id: id, engine: "synclabs", status_url: `/creative-suite/lipsync/${id}/status`, provider_job: sd, note: "Neural render queued — poll status until done, then GET the video URL." });
      } catch (e) {
        console.warn("[lipsync] neural engine failed, falling back to local:", e.message);
      }
    }
    // Engine B: local fallback — timed captions + dub mix plan.
    const buf = Buffer.from(String(audio).replace(/^data:[\w/;-]+;base64,/, ""), "base64");
    const { subtitles, text } = await aiFeatures.generateSubtitles(buf, language || "auto");
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt, result_url) VALUES (?, ?, 'lipsync', 'done', ?, ?)").run(id, uid, `local lang=${language || "auto"}`, JSON.stringify(subtitles).slice(0, 4000));
    debitCredits(uid, "stt_min", "local dub"); // local fallback: STT cost only, post-success
    res.json({
      ok: true, job_id: id, engine: "local", transcript: text, subtitles,
      mix: { op: "replace_audio", note: "Render in sandbox: replace video audio track with the voiceover, then burn in subtitles" },
      ...(SYNC_LABS_KEY ? {} : { upgrade: "Set SYNC_LABS_API_KEY + PUBLIC_BASE_URL for neural mouth re-animation." }),
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});
app.get("/creative-suite/lipsync/:jobId/status", requireAuth, limitNormal, async (req, res) => {
  const row = db.prepare("SELECT id, kind, status, prompt, result_url, created_at FROM creative_jobs WHERE id = ?").get(req.params.jobId);
  if (!row) return res.status(404).json({ error: "job not found" });
  // Hedra jobs: poll live (no webhook needed) and flip the row when done.
  const m = String(row.prompt || "").match(/^hedra:(\S+)/);
  // HeyGen jobs (OpenRouter /videos pipeline): poll the provider job; the MP4
  // is served via /videos/content/<providerId> once completed.
  const hy = String(row.prompt || "").match(/^heygen:(\S+)/);
  if (hy && row.status !== "done" && HAS_OR) {
    try {
      const p = await pollVideo(hy[1].split(" ")[0]);
      const st = String(p?.data?.status || p?.status || "").toLowerCase();
      if (st === "completed") {
        const curl = `${PUBLIC_BASE_URL || ""}/videos/content/${hy[1].split(" ")[0]}`;
        db.prepare("UPDATE creative_jobs SET status = 'done', result_url = ? WHERE id = ?").run(curl, row.id);
        chargeProviderJob(hy[1].split(" ")[0]); // completion billing (no-op if charged via /videos/*)
        return res.json({ ok: true, ...row, status: "done", result_url: curl });
      }
      if (st === "failed" || st === "cancelled" || st === "expired") {
        db.prepare("UPDATE creative_jobs SET status = 'failed' WHERE id = ?").run(row.id);
        return res.json({ ok: true, ...row, status: "failed", provider_error: p?.data?.error || p?.error || st });
      }
      return res.json({ ok: true, ...row, provider_status: st || "processing" });
    } catch (e) {
      return res.json({ ok: true, ...row, provider_error: e.message });
    }
  }
  if (m && row.status !== "done" && HEDRA_API_KEY) {
    try {
      const st = await hedraFetch(`/v3/jobs/${encodeURIComponent(m[1])}/status`);
      const sj = await st.json().catch(() => ({}));
      const s = String(sj.status || sj.state || "").toUpperCase();
      if (s === "COMPLETED" || s === "COMPLETE" || s === "DONE") {
        const job = await (await hedraFetch(`/v3/jobs/${encodeURIComponent(m[1])}`)).json().catch(() => ({}));
        const url = hedraVideoUrl(job);
        if (url) {
          db.prepare("UPDATE creative_jobs SET status = 'done', result_url = ? WHERE id = ?").run(url, row.id);
          chargeProviderJob(m[1]); // completion billing, exactly once (was charged at queue time)
          return res.json({ ok: true, ...row, status: "done", result_url: url });
        }
      }
      if (s === "FAILED" || s === "ERROR" || s === "CANCELLED") {
        db.prepare("UPDATE creative_jobs SET status = 'failed' WHERE id = ?").run(row.id);
        return res.json({ ok: true, ...row, status: "failed" });
      }
      return res.json({ ok: true, ...row, provider_status: s || "processing" });
    } catch (e) {
      return res.json({ ok: true, ...row, provider_error: e.message });
    }
  }
  res.json({ ok: true, ...row });
});

// Shot Recipes: curated one-tap looks (several FRIT-originals).
app.get("/creative-suite/cinema/recipes", requireAuth, limitNormal, (_req, res) => {
  res.json({ ok: true, recipes: cinema.SHOT_RECIPES });
});

// Auto-Pilot (FRIT-original): script -> AI-directed shots -> one image per
// shot (parallel) -> assembled MP4 slideshow film via sandbox ffmpeg.
app.post("/creative-suite/autopilot", requireAuth, limitCostly, async (req, res) => {
  try {
    const { script, per_shot_secs, aspect_ratio } = req.body || {};
    if (!script) return res.status(400).json({ error: "script required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "video");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const shots = await cinema.aiDirect(script, 4);
    const perSecs = Math.min(4, Math.max(1, Number(per_shot_secs) || 2));
    const directedShots = await Promise.all(shots.map(async (s) => {
      const directed = cinema.buildDirectorPrompt({
        prompt: s.prompt, genre: s.genre, camera: s.camera, lens: s.lens,
        aperture: s.aperture, move: s.move, angle: s.angle, lighting: s.lighting,
        palette: s.palette, emotion: s.emotion, tempo: "Single Shot",
      });
      try {
        const img = await generateImage(directed, { aspect_ratio: aspect_ratio || "16:9" });
        db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
        let b64 = img.image_b64 || null;
        if (!b64 && img.image_url) {
          const r = await fetch(img.image_url, { signal: AbortSignal.timeout(120_000) });
          if (!r.ok) throw new Error(`frame fetch HTTP ${r.status}`);
          b64 = Buffer.from(await r.arrayBuffer()).toString("base64");
        }
        if (!b64) throw new Error("empty frame");
        if (Buffer.from(b64, "base64").length > 8 * 1024 * 1024) throw new Error("frame >8MB");
        return { ...s, directed_prompt: directed, ok: true, b64 };
      } catch (e) {
        return { ...s, directed_prompt: directed, ok: false, error: e.message };
      }
    }));
    // Sort by shot index before assembly: frames generate in parallel, so
    // completion order can differ from shot order (scrambled films).
    const frames = directedShots
      .filter((g) => g.ok && g.b64)
      .sort((a, b) => (Number(a.index) || 0) - (Number(b.index) || 0));
    if (!frames.length) return res.status(500).json({ ok: false, error: "all shot generations failed", shots: directedShots });
    const files = frames.map((f) => ({ base64: f.b64 }));
    const r = await fetch(`${SANDBOX_URL}/media/assemble`, {
      method: "POST",
      headers: sandboxAuthHeaders(),
      body: JSON.stringify({ op: "slideshow", files, per_image_secs: perSecs }),
      signal: AbortSignal.timeout(180_000),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || !data?.base64) throw new Error(data?.error || `assemble HTTP ${r.status}`);
    const id = `film_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt) VALUES (?, ?, 'autofilm', 'done', ?)").run(id, uid, String(script).slice(0, 500));
    // Slideshow of delivered frames: pay per frame, not clip rate (no provider video model runs here).
    for (const f of files) debitCredits(uid, "image", "autopilot frame");
    res.json({ ok: true, job_id: id, shots: directedShots, film: { mime: "video/mp4", size: data.size, base64: data.base64 } });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Brand Kit (FRIT-original): brand lock (colors/font/logo/palette/tone)
// applied to every generation via prompt injection — no extra model cost.
app.post("/creative-suite/brandkit", requireAuth, limitNormal, (req, res) => {
  try {
    const { name, colors, font, logo_url, palette, tone } = req.body || {};
    if (!name) return res.status(400).json({ error: "name required" });
    const id = `brand_${Date.now().toString(36)}`;
    db.prepare("INSERT INTO creative_brandkits (id, user_id, name, colors, font, logo_url, palette, tone) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
      .run(id, boundUser(req), name, JSON.stringify(Array.isArray(colors) ? colors.slice(0, 6) : []), font || "", logo_url || "", palette || "", tone || "");
    res.json({ ok: true, brandkit_id: id });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});
app.get("/creative-suite/brandkit", requireAuth, limitNormal, (req, res) => {
  const rows = db.prepare("SELECT * FROM creative_brandkits WHERE user_id = ? ORDER BY created_at DESC LIMIT 50").all(boundUser(req));
  res.json({ ok: true, brandkits: rows.map(r => ({ ...r, colors: JSON.parse(r.colors || "[]") })) });
});
app.post("/creative-suite/brandkit/:id/delete", requireAuth, limitNormal, (req, res) => {
  db.prepare("DELETE FROM creative_brandkits WHERE id = ? AND user_id = ?").run(req.params.id, boundUser(req));
  res.json({ ok: true });
});

// F-13 (server-side brand lock): every generation route accepts an optional
// `brandkit_id`. The kit is resolved HERE, from the DB, with the ownership
// check in SQL — the client only sends an id and can never inject brand text
// of its own. Returns { brand, prompt } where prompt already carries the
// lock fragment.
function applyBrand(user_id, brandkit_id, prompt) {
  const brand = brandKit.resolveBrandKit(db, user_id, brandkit_id);
  return { brand, prompt: brandKit.applyBrandToPrompt(prompt, brand) };
}

// Design Studio (graphics niche): flyers, posters, thumbnails, cards with REAL
// legible text. Routed to the design specialist (OR_GRAPHICS_MODEL), never the
// photo model. Template zones + brand kit keep every piece on-brand.
app.get("/creative-suite/design/templates", requireAuth, limitNormal, (_req, res) => {
  res.json({ ok: true, templates: templateLibrary.getGraphicsTemplates() });
});
app.post("/creative-suite/design", requireAuth, limitCostly, async (req, res) => {
  try {
    const { template, headline, body, cta, palette, brandkit_id, logo_url, style_refs, craft, variations, aspect_ratio, font } = req.body || {};
    if (!headline && !template) return res.status(400).json({ error: "headline or template required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const tpl = templateLibrary.getGraphicsTemplate(template || "") || null;
    // F-13: brand lock is resolved server-side (ownership-checked in SQL).
    const { brand } = applyBrand(uid, brandkit_id, "");
    // Craft routing: each deliverable goes to the model built for it.
    const want = (craft || "auto").toLowerCase();
    const resolved = want === "auto" ? designStudio.pickCraft({ template, headline, body }) : want;
    const chain = resolved === "type"
      ? [OR_TYPE_MODEL, OR_GRAPHICS_MODEL, OR_IMAGE_MODEL]
      : resolved === "vector"
        ? [OR_VECTOR_MODEL, OR_GRAPHICS_MODEL, OR_IMAGE_MODEL]
        : resolved === "photo"
          ? [OR_IMAGE_MODEL, OR_GRAPHICS_MODEL]
          : [OR_GRAPHICS_MODEL, OR_IMAGE_MODEL];
    const models = [...new Set(chain.filter(Boolean))];
    if (!models.length) return res.status(500).json({ ok: false, error: "no graphics model configured" });
    const refs = [];
    if (logo_url) refs.push({ type: "image_url", image_url: { url: String(logo_url) } });
    for (const r of (Array.isArray(style_refs) ? style_refs : []).slice(0, 4)) {
      refs.push({ type: "image_url", image_url: { url: String(r) } });
    }
    const n = designStudio.variationCount(variations);
    const base = { template: tpl, headline, body, cta, palette, brand, styleMood: refs.length > 1 ? "match the mood-board references" : "", font };
    // Same brief, distinct art directions, rendered in parallel — pick, don't settle.
    const designs = await Promise.all(Array.from({ length: n }, async (_, v) => {
      const prompt = designStudio.buildDesignPrompt({ ...base, variation: v });
      let lastErr = null;
      for (const model of models) {
        try {
          const out = await generateImage(prompt, {
            model,
            output_format: "png",
            ...(aspect_ratio ? { aspect_ratio } : {}),
            ...(refs.length ? { input_references: refs } : {}),
          });
          db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
          return { ok: true, model_used: model, image_url: out.image_url || null, image_b64: out.image_b64 || null };
        } catch (e) { lastErr = e; }
      }
      return { ok: false, error: lastErr?.message || "all design models failed" };
    }));
    const anyOk = designs.some(d => d.ok);
    const id = `gfx_${Date.now()}`;
    const firstUrl = designs.find(d => d.image_url)?.image_url
      || designs.find(d => d.image_b64)?.image_b64 || "";
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt, result_url) VALUES (?, ?, 'design', ?, ?, ?)").run(id, uid, anyOk ? "done" : "failed", `craft=${resolved} ${String(headline || template || "").slice(0, 300)}`, firstUrl.slice(0, 1000));
    if (!anyOk) {
      return res.status(500).json({ ok: false, error: "all design routes failed", craft: resolved, designs });
    }
    // Charge per delivered design (image rate) — never before the work.
    for (const d of designs) if (d.ok) debitCredits(uid, "image", "design route");
    res.json({ ok: true, job_id: id, template: tpl?.id || null, zones: tpl?.zones || [], craft: resolved, brand_applied: !!brand, designs });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ==================== REFERENCE SERVING ====================
// Pinterest / Behance CDNs hotlink-protect a lot of their images. The
// reference transport (creative/refs.js) fetches the bytes ONCE, browser-like,
// and caches them on disk under a content-addressed id. This route serves
// those bytes back, so the Android client can SHOW which references the
// agent actually studied without hotlinking a third party.
app.get("/creative-suite/refs/:id", limitNormal, async (req, res) => {
  const entry = await refs.readRef(req.params.id);
  if (!entry) return res.status(404).json({ error: "reference expired" });
  res.set("Content-Type", entry.mime || "image/jpeg");
  res.set("Cache-Control", "private, max-age=1800");
  res.set("X-Content-Type-Options", "nosniff");
  res.send(entry.buf);
});

// ==================== DESIGN AGENT (Week 6) ====================
// Three stages the Design Studio was missing: research the web for visual
// references, turn them into ONE detailed brief (layout, alignment, palette,
// fonts, sizes, elements, background, exact copy), then render and VERIFY the
// render against that brief (one corrective retry). The whole trail comes back
// so the app can show the reasoning, not just an image.
// Week 7 hardening:
//   * the reference is split into STRUCTURE / CONTENT and only structure is
//     allowed into the brief (creative/designDNA.js),
//   * references are picked for maximum structural SPREAD, not top-of-search,
//   * every run carries a seed: query style token, reference order and art
//     direction all derive from it, so the same request never produces the
//     same design twice,
//   * the brief is pushed AWAY from the references it studied,
//   * a TYPE LAYER sets the real typography over the rendered artwork.
const designAgentDeps = {
  // retries: 1 + a bounded timeout. The default (3 polite retries that wait
  // out a 429) burned the whole stage budget on one rate-limited provider;
  // with this, a 429 moves to the NEXT MODEL in the chain immediately.
  chat: async (messages, opts = {}) => {
    // One retry: an "empty completion" from a vision model is a provider
    // hiccup, not a verdict. The stage timeout bounds the total.
    let lastErr = null;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const out = await chatWithFallback("vision", {
          messages,
          max_tokens: opts.max_tokens ?? 900,
          temperature: opts.temperature ?? 0.2,
          retries: 1,
          timeoutMs: 45_000,
        });
        const c = String(out?.choices?.[0]?.message?.content || "");
        if (c.trim()) return c;
        lastErr = new Error("the vision model returned an empty completion");
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr || new Error("the vision model returned nothing");
  },
  // The BRIEF writer only needs text (the reference DNA is already text), so
  // it runs on the DRAFT chain. The vision model is reserved for reading the
  // references and for reviewing the render. Using the vision model for the
  // brief blew the stage budget on the first live run.
  chatText: async (messages, opts = {}) => {
    const out = await chatWithFallback("draft", {
      messages,
      max_tokens: opts.max_tokens ?? 1400,
      temperature: opts.temperature ?? 0.4,
      retries: 1,
      timeoutMs: 30_000,
    });
    const c = out?.choices?.[0]?.message?.content;
    return typeof c === "string" ? c : (Array.isArray(c) ? c.map((p) => p?.text || "").join("") : "");
  },
  // Model CHAIN, not a single slug: an unserved/retired graphics model must
  // never kill a finished brief (that is exactly what Week 6 found — the old
  // design default was retired upstream). Explicit caller model wins.
  // The chain stays in ONE class: the prompt tells the model whether WE set
  // the type, so a chain that mixed a type specialist with a general model
  // would hand half the chain the wrong prompt (and double the copy).
  generate: async (prompt, opts = {}) => {
    const wanted = [...new Set([opts.model, OR_GRAPHICS_MODEL, OR_TYPE_MODEL, OR_IMAGE_MODEL].filter(Boolean))];
    const specialist = imageProviders.isTypeSpecialist(wanted[0] || "");
    const chain = imageProviders.classConsistentChain(wanted, { specialist });
    let lastErr = null;
    for (const model of chain) {
      try {
        return await generateImage(prompt, {
          model,
          output_format: opts.output_format || "png",
          ...(opts.aspect_ratio ? { aspect_ratio: opts.aspect_ratio } : {}),
        });
      } catch (e) {
        lastErr = e;
        console.warn(`[design-agent] ${model} failed: ${e.message}`);
      }
    }
    throw lastErr || new Error("no image model configured");
  },
  // Who sets the type? A type specialist (Recraft) renders real fonts itself,
  // so we must NOT draw the copy again on top. Every other model mangles
  // lettering, so WE typeset it over the artwork.
  typesetByUs: (model) => !imageProviders.isTypeSpecialist(model || OR_GRAPHICS_MODEL || OR_IMAGE_MODEL),
  // Reference transport: browser-like fetch, public origin-side proxy
  // fallback, <=1024px downscale, content-addressed disk cache. The inline
  // data URI is what the vision model reads; the proxy URL is what the app
  // shows the user.
  loadReference: async (url) => {
    const got = await refs.loadReference(url, { fetchImpl: fetch });
    if (!got) {
      const fallback = refs.proxyImageUrl(url);
      return { dataUri: null, url: fallback, id: null };
    }
    const base = PUBLIC_BASE_URL || "";
    return {
      dataUri: got.dataUri,
      id: got.id,
      url: base && got.id ? `${base}/creative-suite/refs/${got.id}` : null,
      bytes: got.bytes,
    };
  },
  // The type layer: real glyphs, real sizes, set from the brief.
  applyType: async (buffer, brief) => typeLayer.applyTypeLayer(buffer, brief),
  fetchImpl: fetch,
};

app.post("/creative-suite/design/agent", requireAuth, limitCostly, async (req, res) => {
  try {
    const input = req.body || {};
    const goal = String(input.goal || input.headline || "").trim();
    if (!goal) return res.status(400).json({ error: "goal required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "image");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    // F-13: brand kit resolved server-side (ownership checked in SQL).
    const { brand } = applyBrand(uid, input.brandkit_id, "");
    const out = await designAgent.runDesignAgent({ ...input, goal, brand, uid }, designAgentDeps);
    const id = `dsg_${Date.now()}`;
    const firstUrl = out.render?.image_url || out.render?.image_b64 || "";
    try {
      db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt, result_url) VALUES (?, ?, 'design_agent', ?, ?, ?)")
        .run(id, uid, out.ok ? "done" : "failed", goal.slice(0, 400), String(firstUrl).slice(0, 1000));
    } catch (e) { console.warn("[design-agent] job row failed:", e.message); }
    if (!out.ok) return res.status(502).json({ ok: false, error: "all render attempts failed", job_id: id, ...out });
    // Charge once per DELIVERED design (image rate), never before the work.
    debitCredits(uid, "image", "design agent");
    db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    res.json({ ok: true, job_id: id, brand_applied: !!brand, ...out });
  } catch (err) {
    console.error("[/creative-suite/design/agent]", err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// AI Director: brief (+ optional reference video) -> LOOK research -> shot
// plan -> per-shot generation -> FRAME VERIFICATION (real observe-and-
// correct) -> assembled MP4.
//
// Week 7: "auto_correct" used to mean "retry when the provider threw". It
// now means what it says — the rendered clip is sampled into frames, a
// vision model checks each frame against the shot it was supposed to be, and
// a failed shot is re-rendered with the reviewer's own complaints.
app.post("/creative-suite/director", requireAuth, limitCostly, async (req, res) => {
  try {
    const { brief, reference_video, num_shots, auto_correct, aspect_ratio } = req.body || {};
    if (!brief) return res.status(400).json({ error: "brief required" });
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "video");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });

    // Reference video (user supplied) — passed to the renderer as-is.
    const refUrls = [];
    if (reference_video) {
      const s = String(reference_video);
      if (/^https?:\/\//i.test(s)) refUrls.push(s);
      else {
        const r = await fetch(s, { signal: AbortSignal.timeout(120_000) });
        if (r.ok) {
          const b64 = Buffer.from(await r.arrayBuffer()).toString("base64");
          refUrls.push(`data:video/mp4;base64,${b64}`);
        }
      }
    }

    const deps = {
      chat: async (messages, opts = {}) => {
        const out = await chatWithFallback("vision", {
          messages,
          max_tokens: opts.max_tokens ?? 1200,
          temperature: opts.temperature ?? 0.4,
        });
        return String(out?.choices?.[0]?.message?.content || "");
      },
      // Look research: cinematography vocabulary, not layout vocabulary.
      research: async (goal, { seed }) => {
        const queries = directorAgent.planLookQueries(goal, { seed });
        const refs = [];
        for (const q of queries) {
          try {
            const urls = await designAgent.bingImageSearch(q, { fetchImpl: fetch, limit: 8 });
            refs.push(...urls);
          } catch { /* keep going */ }
          if (refs.length >= 3) break;
        }
        return { refs: designAgent.rankInspirationUrls(refs, { max: 4, seed }) };
      },
      readLook: async (ref, goal) => {
        const loaded = await refs.loadReference(ref, { fetchImpl: fetch });
        return directorAgent.readLookReference(ref, goal, {
          chat: deps.chat,
          loaded: loaded
            ? { dataUri: loaded.dataUri, url: loaded.id ? `${PUBLIC_BASE_URL || ""}/creative-suite/refs/${loaded.id}` : null }
            : { dataUri: null, url: refs.proxyImageUrl(ref) },
        });
      },
      planFilm: (goal, n, looks) => directorAgent.synthesizeShotPlan(goal, n, looks, { chat: deps.chat }),
      renderShot: async (shot, { attempt, correction }) => {
        const directed = cinema.buildDirectorPrompt({
          prompt: shot.prompt, genre: shot.genre, camera: shot.camera, lens: shot.lens,
          aperture: shot.aperture, move: shot.move, angle: shot.angle, lighting: shot.lighting,
          palette: shot.palette, emotion: shot.emotion, tempo: "Single Shot",
        }) + (correction ? ` CORRECTIONS FROM REVIEW: ${correction}.` : "");
        const isVideo = shot.kind !== "image";
        const refs = refUrls.map(u => ({ type: "image_url", image_url: { url: u } }));
        try {
          if (isVideo) {
            const job = await submitVideo(directed, {
              duration: shot.duration_s || 4, resolution: "720p",
              aspect_ratio: aspect_ratio || "16:9",
              ...(refs.length ? { input_references: refs } : {}),
            });
            const vid = await pollVideo(job.id || job.job_id);
            if (vid?.data?.status === "completed" || vid?.status === "completed") {
              const content = await downloadVideo(job.id || job.job_id, 0);
              db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
              return {
                ok: true, kind: "video",
                url: `data:video/mp4;base64,${content.toString("base64")}`,
                base64: content.toString("base64"),
              };
            }
            throw new Error(`clip not ready (${vid?.data?.status || vid?.status || "unknown"})`);
          }
          const img = await generateImage(directed, {
            aspect_ratio: aspect_ratio || "16:9",
            ...(refs.length ? { input_references: refs } : {}),
          });
          const url = img.image_url || (img.image_b64 ? `data:image/png;base64,${img.image_b64}` : null);
          if (!url) throw new Error("no image returned");
          db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
          return { ok: true, kind: "image", url, base64: img.image_b64 || null };
        } catch (e) {
          // F-08: an intermediate attempt's failure was dropped, so a flaky
          // provider looked like a clean success on the retry.
          console.warn(`[director] shot ${shot.index} attempt ${attempt} failed: ${e.message}`);
          return { ok: false, error: e.message };
        }
      },
      // Sample frames out of the rendered clip so it can actually be LOOKED
      // at. Still shots are their own frame.
      framesOf: async (r) => {
        if (r.kind !== "video") return r.url ? [r.url] : [];
        if (!r.base64 || !SANDBOX_URL) return [];
        try {
          const res = await fetch(`${SANDBOX_URL}/media/frames`, {
            method: "POST",
            headers: sandboxAuthHeaders(),
            body: JSON.stringify({ video_base64: r.base64, timestamps: [0.5, 2.0], width: 768 }),
            signal: AbortSignal.timeout(90_000),
          });
          const data = await res.json().catch(() => ({}));
          if (!res.ok || !data?.frames?.length) return [];
          return data.frames.map((f) => `data:${f.mime || "image/jpeg"};base64,${f.base64}`);
        } catch (e) {
          console.warn(`[director] frame extraction failed: ${e.message}`);
          return [];
        }
      },
      verifyFrame: (frame, shot) => directorAgent.verifyShotFrame(frame, shot, { chat: deps.chat }),
    };

    const agentOut = await directorAgent.runDirectorAgent(
      { brief, num_shots: num_shots, auto_correct, uid },
      deps
    );
    const generated = agentOut.shots;
    if (!generated.length) throw new Error("AI Director could not plan the film — try a more specific brief");

    const succeeded = generated.filter(g => g.ok && g.url);
    if (!succeeded.length) {
      return res.status(500).json({ ok: false, error: "all shots failed to generate", shots: generated, warnings: agentOut.warnings });
    }

    // Assemble into a film
    const files = succeeded.map(g => ({ base64: g.url.replace(/^data:[\w/;-]+;base64,/, "") }));
    const isMixed = succeeded.some(g => g.kind === "video");
    const assembleOp = isMixed ? "concat" : "slideshow";
    const r = await fetch(`${SANDBOX_URL}/media/assemble`, {
      method: "POST",
      headers: sandboxAuthHeaders(),
      body: JSON.stringify({
        op: assembleOp, files,
        per_image_secs: 2,
        width: 1280, height: 720,
      }),
      signal: AbortSignal.timeout(180_000),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || !data?.base64) throw new Error(data?.error || `assemble HTTP ${r.status}`);

    const id = `dir_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt) VALUES (?, ?, 'director', 'done', ?)").run(id, uid, String(brief).slice(0, 500));
    // Charge per delivered shot at its own rate (video shots = clip, stills = image).
    for (const g of succeeded) debitCredits(uid, g.kind === "video" ? "clip" : "image", "ai director shot");
    res.json({
      ok: true, job_id: id,
      shots: generated,
      looks: agentOut.looks,
      verification: generated.map((g) => ({ index: g.index, ...(g.verification || {}) })),
      warnings: agentOut.warnings,
      film: { mime: "video/mp4", size: data.size, base64: data.base64 },
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// Trial feedback: testers on trial codes tell us what broke (rating 1-5 + text).
app.post("/feedback", requireAuth, limitNormal, (req, res) => {
  const { message, rating } = req.body || {};
  const user_id = boundUser(req);
  if (!message) return res.status(400).json({ error: "message required" });
  const tier = activeTier(String(user_id));
  const r = db.prepare("INSERT INTO feedback (user_id, tier, rating, message) VALUES (?, ?, ?, ?)").run(String(user_id), tier, Number(rating) || null, String(message).slice(0, 2000));
  res.json({ ok: true, id: r.lastInsertRowid });
});
app.get("/feedback", requireAdmin, (_req, res) => {
  res.json({ ok: true, feedback: db.prepare("SELECT * FROM feedback ORDER BY created_at DESC LIMIT 200").all() });
});
const LEGAL_PAGES = { terms: "TERMS.md", privacy: "PRIVACY.md", refunds: "REFUNDS.md" };
app.get("/legal/:page", (req, res) => {
  const file = LEGAL_PAGES[String(req.params.page || "").toLowerCase()];
  if (!file) return res.status(404).send("unknown legal page");
  try {
    const md = readFileSync(join(__dirname, "../../docs", file), "utf8");
    const html = md
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/^### (.*)$/gm, "<h3>$1</h3>").replace(/^## (.*)$/gm, "<h2>$1</h2>").replace(/^# (.*)$/gm, "<h1>$1</h1>")
      .replace(/^\- (.*)$/gm, "<li>$1</li>").replace(/\n/g, "<br>");
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.send(`<html><body style="font-family:sans-serif;max-width:720px;margin:2em auto;padding:0 1em">${html}</body></html>`);
  } catch (e) {
    res.status(500).send("legal page unavailable");
  }
});

// ==================== WEB SEARCH ====================
// /tools/search — raw web-search endpoint used by the agent brain (search_web tool)
// and by any client that wants real-time info without going through the LLM.
app.post("/tools/search", requireAuth, limitNormal, async (req, res) => {
  const { query, mode = "text" } = req.body || {};
  if (!query) return res.status(400).json({ error: "query required" });
  try {
    const data = await deepSearch(String(query), mode === "structured" ? "structured" : "text");
    res.json({ ok: true, ...data });
  } catch (err) {
    console.error("[/tools/search]", err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ==================== ENHANCED PIPELINE ROUTES ====================
app.post("/enhanced/analyze", requireAuth, async (req, res) => {
  const { symbol, interval = "1h", balance, riskPercent } = req.body || {};
  if (!symbol) return res.status(400).json({ error: "symbol required" });
  const uid = boundUser(req);
  const ent = checkEntitlement(uid, "trading");
  if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
  try {
    const result = await mtfStrategy.analyze(symbol, { interval, balance, riskPercent });
    debitCredits(uid, "trade_analysis", "enhanced analyze"); // post-success
    res.json(result);
  } catch (err) {
    console.error("[/enhanced/analyze]", err.message);
    res.status(500).json({ error: "Enhanced analysis failed", details: err.message });
  }
});

// Full multi-timeframe directional report — "build with directions".
app.get("/market/strategy", requireAuth, async (req, res) => {
  try {
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "trading");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    const sym = String(req.query.symbol || "XAUUSD").toUpperCase();
    const result = await mtfStrategy.analyze(sym, {});
    debitCredits(uid, "trade_analysis", "market strategy"); // post-success
    res.json(result);
  } catch (err) {
    console.error("[/market/strategy]", err.message);
    res.status(500).json({ error: "Strategy analysis failed", details: err.message });
  }
});

// Strategy engine status: cache + Twelve Data free-tier budget + last run.
app.get("/strategy/status", requireAuth, async (req, res) => {
  res.json({ ...mtfStrategy.cacheStatus(), scheduler: tradeScheduler.status() });
});

// Recurring trade tasks ("everyday analyze XAUUSD and place orders").
app.post("/tasks/trade-schedule", requireAuth, async (req, res) => {
  try {
    const { symbol, time, riskPercent, balance, action } = req.body || {};
    const uid = boundUser(req);
    const ent = checkEntitlement(uid, "trading");
    if (!ent.ok) return res.status(402).json({ error: ent.message, code: ent.code, needed: ent.needed, balance: ent.balance });
    if (action === "stop") { tradeScheduler.stop(); return res.json({ ok: true, message: "Scheduler stopped" }); }
    if (action === "start") { tradeScheduler.start(); return res.json({ ok: true, message: "Scheduler started" }); }
    if (action === "remove") { return res.json({ ok: tradeScheduler.remove(symbol) }); }
    const task = tradeScheduler.add({ symbol, time, riskPercent, balance });
    tradeScheduler.start();
    res.json({ ok: true, ...task });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.get("/tasks/trade-schedule", requireAuth, (_req, res) => {
  res.json(tradeScheduler.status());
});

// System status (engine.js + scheduler + positions).
app.get("/systems/status", requireAuth, (_req, res) => {
  res.json({
    engine: "engine_js",
    strategy: mtfStrategy.cacheStatus(),
    scheduler: tradeScheduler.status(),
    positions: positionMonitor.status(),
  });
});

// ==================== TRADING BOTS (scalping / pocket / intraday / longterm / options) ==========
// One config surface for the phone's AI-help Bots UI. Intraday/longterm auto
// flows through executeTradeSignal (risk gate + bridge + monitor); the
// options bot auto-scans + notifies + paper-tracks and NEVER executes.
app.get("/api/trading/bots", requireAuth, async (_req, res) => {
  try {
    res.json({ bots: getBots(db), defs: BOT_DEFS });
  } catch (err) {
    res.status(500).json({ error: "Bots failed", details: err.message });
  }
});

app.post("/api/trading/bots/:name", requireAuth, async (req, res) => {
  try {
    const bot = saveBot(db, req.params.name, req.body || {});
    res.json({ ok: true, bot });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.get("/api/trading/bots/runs", requireAuth, async (req, res) => {
  try {
    const limit = Math.min(Number(req.query.limit) || 30, 100);
    const runs = db.prepare("SELECT * FROM bot_runs ORDER BY fired_at DESC LIMIT ?").all(limit);
    res.json({ runs });
  } catch (err) {
    res.status(500).json({ error: "Bot runs failed", details: err.message });
  }
});

// --- Bot-Pair Association (per-pair bot selection) ---
// Each pair can have multiple bots assigned. This replaces the old single
// bot-per-config model with a per-pair bot selection system.

app.get("/api/trading/bot-pairs", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const pair = req.query.pair ? String(req.query.pair).toUpperCase() : null;
    const items = getBotPairs(db, user_id, pair);
    res.json({ items });
  } catch (err) {
    res.status(500).json({ error: "Bot-pairs fetch failed", details: err.message });
  }
});

app.post("/api/trading/bot-pairs", requireAuth, async (req, res) => {
  try {
    const user_id = requireBotAccess(req, res);
    if (!user_id) return;
    const { botName, pair, timeframe, tradeEnabled, signalEnabled, minConfidence, maxTradesDay } = req.body || {};
    if (!botName || !pair) return res.status(400).json({ error: "botName and pair required" });
    const item = saveBotPair(db, user_id, botName, pair, {
      timeframe, tradeEnabled, signalEnabled, minConfidence, maxTradesDay,
    });
    res.json({ ok: true, item });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.delete("/api/trading/bot-pairs/:botName/:pair", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const ok = removeBotPair(db, user_id, req.params.botName, req.params.pair);
    res.json({ ok });
  } catch (err) {
    res.status(500).json({ error: "Bot-pair remove failed", details: err.message });
  }
});

// --- Notification Aggregation (all bot signals tagged by name) ---
// Returns recent signals from all bots/engines, tagged by source bot name.
app.get("/api/trading/notifications", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const limit = Math.min(Number(req.query.limit) || 50, 200);
    const notifications = [];

    // 1. Bot runs (scalping, pocket, intraday, longterm) — THIS user's runs
    //    first, then the platform's shared book.
    try {
      const runs = db.prepare(`
        SELECT bot, pair, timeframe, decision, confidence, note, fired_at, user_id
        FROM bot_runs WHERE decision IN ('BUY','SELL','CALL','PUT')
        ORDER BY fired_at DESC LIMIT ?
      `).all(limit * 2);
      for (const r of runs) {
        notifications.push({
          id: `bot_${r.user_id || "bot"}_${r.bot}_${r.fired_at}`,
          bot: r.bot,
          type: "bot_signal",
          symbol: r.pair,
          decision: r.decision,
          confidence: r.confidence,
          note: `${r.timeframe || ""} · ${r.note || ""}`.trim(),
          timestamp: r.fired_at,
          read: false,
          mine: r.user_id === user_id,
        });
      }
    } catch { /* best-effort */ }

    // 2. Trading signals (main engine, intraday bot, longterm bot, scalping bot)
    try {
      const sigs = db.prepare(`
        SELECT id, symbol, decision, confidence, reasoning, entry, sl, tp1, tp2, rr,
               scenario, bot_source, lot_size, lot_reason, risk_usd, timestamp
        FROM trading_signals WHERE user_id = ? AND decision IN ('BUY','SELL','CALL','PUT')
        ORDER BY timestamp DESC LIMIT ?
      `).all(user_id, limit);
      for (const s of sigs) {
        // The signal records which engine/bot produced it — that is the tag.
        const botName = s.bot_source || "engine";
        notifications.push({
          id: `sig_${s.id}`,
          bot: botName,
          type: "engine_signal",
          symbol: s.symbol,
          decision: s.decision,
          confidence: s.confidence,
          note: (s.reasoning || "").slice(0, 200),
          timestamp: s.timestamp,
          read: false,
          // Full trade detail, shown when the user taps the bar entry.
          detail: {
            entry: s.entry, sl: s.sl, tp1: s.tp1, tp2: s.tp2, rr: s.rr,
            scenario: s.scenario, reasoning: s.reasoning,
            lotSize: s.lot_size, lotReason: s.lot_reason, riskUsd: s.risk_usd,
          },
        });
      }
    } catch { /* best-effort */ }

    // 3. Deriv contract settlements
    try {
      const contracts = db.prepare(`
        SELECT symbol, direction, stake, profit, mode, placed_at
        FROM deriv_contracts WHERE status = 'SETTLED'
        ORDER BY placed_at DESC LIMIT ?
      `).all(limit);
      for (const c of contracts) {
        notifications.push({
          id: `deriv_${c.placed_at}_${c.symbol}`,
          bot: c.mode === "REAL" ? "deriv-live" : "deriv-demo",
          type: "deriv_settlement",
          symbol: c.symbol,
          decision: c.direction,
          confidence: 0,
          note: `P&L: $${c.profit?.toFixed(2) || "?"} (${c.stake} stake)`,
          timestamp: c.placed_at,
          read: false,
        });
      }
    } catch { /* best-effort */ }

    // Sort by timestamp desc, dedupe by id
    const seen = new Set();
    const sorted = notifications
      .sort((a, b) => b.timestamp - a.timestamp)
      .filter(n => { if (seen.has(n.id)) return false; seen.add(n.id); return true; })
      .slice(0, limit);

    res.json({ notifications, total: sorted.length, serverTime: Date.now() });
  } catch (err) {
    res.status(500).json({ error: "Notifications fetch failed", details: err.message });
  }
});

// --- Volume matrix: 5-timeframe bias filter (VP / VWAP / CVD / RVOL) -------
app.get("/api/trading/volume-matrix", requireAuth, async (req, res) => {
  try {
    const sym = String(req.query.symbol || "EURUSD").toUpperCase();
    const tf = {};
    for (const iv of ["1day", "4h", "30min", "15min", "5min", "1min"]) {
      const candles = await fetchCandles(sym, iv, 120);
      if (candles?.length) {
        const key = { "1day": "1d", "4h": "4h", "30min": "30m", "15min": "15m", "5min": "5m", "1min": "1m" }[iv];
        tf[key] = candles;
      }
    }
    if (!Object.keys(tf).length) return res.status(400).json({ error: `no candle data for ${sym}` });
    res.json(analyzeVolumeMatrix(sym, tf));
  } catch (err) {
    res.status(500).json({ error: "Volume matrix failed", details: err.message });
  }
});

// Bots runner: 60s tick, each bot fires on its own cadence (see bots.js).
setInterval(async () => {
  try {
    await tickBots({
      db,
      mtfAnalyze: async (symbol, opts = {}) => {
        // Matrix bias from the candle store (ticks carve 1m..4h). Cached 5 min
        // so the 60s bot tick never refetches. Null when the store is cold —
        // the engine treats that as "no filter", not as a veto.
        let matrixBias = null;
        try {
          const ck = `mx:${symbol.toUpperCase()}`;
          const cached = cacheGet(ck);
          if (cached) matrixBias = cached;
          else {
            const store = globalThis.__liveFeed?.store;
            const tf = {};
            for (const iv of ["30m", "4h", "15m", "5m", "1m"]) {
              const arr = store?.getCandles(symbol.toUpperCase(), iv, 120) || [];
              if (arr.length) tf[iv] = arr.map((x) => ({ open: x.open, high: x.high, low: x.low, close: x.close, volume: x.volume }));
            }
            if (Object.keys(tf).length >= 2) {
              matrixBias = analyzeVolumeMatrix(symbol, tf);
              cacheSet(ck, matrixBias, 5 * 60 * 1000);
            }
          }
        } catch { /* matrix is a filter, never a blocker */ }
        return mtfStrategy.analyze(symbol, { ...opts, matrixBias });
      },
      analyzeSynth: (symbol, closes) => analyzeSynth(symbol, closes),
      scalpingAnalyze: (symbol, closes, timeframe) => scalpingAnalyze(symbol, closes, timeframe),
      pocketScan: (symbol, expiryMin) => pocketScan(symbol, expiryMin),
      botCooldown: (db, key, intervalMin) => botCooldown(db, key, intervalMin),
      // Kelly lot size against the USER's own balance + risk profile (item 9).
      // The legacy "bot" pseudo-user has no account, so it gets the default.
      sizeFor: (userId, signal) => {
        if (!userId || userId === "bot") {
          return {
            lotSize: 0.01,
            riskUsd: 0,
            reason: "platform default 0.01 lot (no user account attached)",
          };
        }
        const acct = getTradingAccount(userId);
        if (!acct) return { lotSize: 0.01, riskUsd: 0, reason: "no account — 0.01 lot default" };
        const m = measuredWinRate(signal.symbol);
        const sizing = recommendLotSize({
          symbol: signal.symbol,
          balance: acct.balance,
          entry: signal.entry,
          sl: signal.sl,
          tp: signal.tp1,
          rr: signal.rr,
          winRate: m.wr,
          profile: acct.riskProfile,
          pipConfig: PIP_CONFIG,
          confidence: signal.confidence,
        });
        return {
          lotSize: sizing.lotSize || 0.01,
          riskUsd: sizing.riskUsd,
          reason: sizing.reason + (m.reliable
            ? ` (win rate ${(m.wr * 100).toFixed(0)}% over ${m.n} resolved setups)`
            : ` (default 45% win rate — only ${m.n} resolved setups)`),
        };
      },
      fetchCandles: async (symbol, interval) => {
        const iv = interval === "1m" ? "1min" : interval === "5m" ? "5min" : interval === "15m" ? "15min" : interval;
        const candles = await fetchCandles(symbol, iv, 200);
        return candles || null;
      },
      fetchCloses: async (symbol, interval) => {
        const candles = await fetchCandles(symbol, interval === "1m" ? "1min" : interval, 200);
        if (!candles?.length) throw new Error(`no ${interval} candles for ${symbol}`);
        return candles.map((c) => c.close).filter((c) => c > 0);
      },
      longtermAnalyze: (symbol, closes) => analyzeLongterm(symbol, closes),
      fetchDaily: async (symbol) => {
        const candles = await fetchCandles(symbol, "1day", 300);
        if (!candles?.length) throw new Error(`no daily candles for ${symbol}`);
        return candles.map((c) => c.close).filter((c) => c > 0);
      },
      getChain: (symbol) => getChain(symbol),
      scanSymbol: (args) => scanSymbol({ ...args, db }),
      scanFx: (args) => scanFx(args),
      storeSignal: (userId, signal) => storeSignalHistory(userId, signal),
      formatSignal: (symbol, result) => formatTradingSignal(symbol, result),
      storeOptionsSignal: (userId, signal) => storeOptionsSignal(userId, signal),
      openPaper: (userId, signal) => openPaperPosition(db, userId, signal),
      executor: (args) => executeTradeSignal({ ...args, source: args.source || "bots" }),      log: (...a) => console.log("[bots]", ...a),
    });
  } catch (e) {
    console.warn("[bots] tick failed:", e.message);
  }
}, 60_000).unref?.();

// ==================== TRADING ACCOUNT (balance / risk / broker creds) ======
// The balance here is the user's REAL account balance — the base for Kelly
// sizing, the portfolio equity, and every paper + external execution.
// It is per-user and persisted, so it survives restarts and device reboots.
function ensureAccountRow(user_id) {
  try {
    db.prepare("INSERT OR IGNORE INTO trading_account (user_id, balance, risk_profile, updated_at) VALUES (?, 1000, 'MODERATE', ?)")
      .run(user_id, Date.now());
  } catch { /* ignore */ }
}

function getTradingAccount(user_id) {
  ensureAccountRow(user_id);
  try {
    const r = db.prepare("SELECT * FROM trading_account WHERE user_id = ?").get(user_id);
    if (!r) return null;
    return {
      userId: user_id,
      balance: Number(r.balance) > 0 ? Number(r.balance) : 1000,
      riskProfile: r.risk_profile || "MODERATE",
      // Credentials are NEVER echoed back in full — only whether they exist.
      hasDerivToken: !!(r.deriv_token && String(r.deriv_token).length > 4),
      derivTokenMode: r.deriv_token_mode || "platform",
      hasMt5: !!(r.mt5_server && r.mt5_login),
      mt5Server: r.mt5_server || null,
      mt5Login: r.mt5_login ? maskSecret(r.mt5_login) : null,
      mt5Broker: r.mt5_broker || null,
      updatedAt: r.updated_at || null,
    };
  } catch { return null; }
}

function maskSecret(s) {
  const str = String(s || "");
  if (str.length <= 4) return "****";
  return str.slice(0, 2) + "*".repeat(Math.max(3, str.length - 4)) + str.slice(-2);
}

function saveTradingAccount(user_id, patch = {}) {
  ensureAccountRow(user_id);
  const cur = db.prepare("SELECT * FROM trading_account WHERE user_id = ?").get(user_id) || {};
  const balance = patch.balance != null ? Number(patch.balance) : Number(cur.balance);
  if (!Number.isFinite(balance) || balance <= 0) throw new Error("balance must be a positive number");
  if (balance > 100_000_000) throw new Error("balance is unrealistically large");
  const riskProfile = patch.riskProfile != null
    ? normalizeRiskProfile(patch.riskProfile)
    : (cur.risk_profile || "MODERATE");
  const derivToken = patch.derivToken != null
    ? (String(patch.derivToken).trim() || null)
    : (cur.deriv_token || null);
  const derivTokenMode = patch.derivTokenMode != null
    ? (String(patch.derivTokenMode) === "own" ? "own" : "platform")
    : (cur.deriv_token_mode || "platform");
  // Only overwrite MT5 fields when the caller actually supplies them, so a
  // partial update never wipes a stored broker login.
  const mt5Server = patch.mt5Server != null ? String(patch.mt5Server).trim() : (cur.mt5_server || null);
  const mt5Login = patch.mt5Login != null ? String(patch.mt5Login).trim() : (cur.mt5_login || null);
  const mt5Password = patch.mt5Password != null ? String(patch.mt5Password).trim() : (cur.mt5_password || null);
  const mt5Broker = patch.mt5Broker != null ? String(patch.mt5Broker).trim() : (cur.mt5_broker || null);
  db.prepare(`
    INSERT INTO trading_account
      (user_id, balance, risk_profile, deriv_token, deriv_token_mode, mt5_server, mt5_login, mt5_password, mt5_broker, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(user_id) DO UPDATE SET
      balance = ?, risk_profile = ?, deriv_token = ?, deriv_token_mode = ?,
      mt5_server = ?, mt5_login = ?, mt5_password = ?, mt5_broker = ?, updated_at = ?
  `).run(user_id, balance, riskProfile, derivToken, derivTokenMode, mt5Server, mt5Login, mt5Password, mt5Broker, Date.now(),
    balance, riskProfile, derivToken, derivTokenMode, mt5Server, mt5Login, mt5Password, mt5Broker, Date.now());
  return getTradingAccount(user_id);
}

// ==================== TRADING SUITE API ====================
// Comprehensive AI-powered trading endpoints for mobile app

// --- Tier gating for bot operations ---
// Non-paid users cannot: add pairs, toggle 24/7 watching, turn a bot on for
// trading or signals, or choose a timeframe. They receive a 402 with an
// upgrade message AND the wallet subscription link so the app can deep-link
// the user straight to the screen where they pay.
function requireBotAccess(req, res) {
  const user_id = boundUser(req);
  const ent = checkEntitlement(user_id, "trading");
  if (!ent.ok) {
    res.status(402).json({
      error: ent.message,
      code: ent.code || "feature_locked",
      feature: "trading",
      // The app routes this to its Wallet → Subscribe screen.
      upgradeUrl: "wallet://subscribe",
      upgradePath: "/wallet",
      needed: ent.needed,
      balance: ent.balance,
    });
    return null;
  }
  return user_id;
}

// --- Watchlist Management ---
app.get("/api/trading/watchlist", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const watchlist = db.prepare("SELECT * FROM trading_watchlist WHERE user_id = ? ORDER BY sort_order ASC").all(user_id);
    // Tombstoned symbols are filtered out even if a stale row survived, so a
    // deleted pair can never reappear on the next app open.
    let items = watchlist;
    try {
      const dead = db.prepare("SELECT symbol FROM watchlist_tombstones WHERE user_id = ? AND list = 'main'").all(user_id).map(r => r.symbol);
      if (dead.length) items = watchlist.filter(w => !dead.includes(w.symbol));
    } catch { /* table missing (pre-migration) = no filtering */ }
    res.json({ items });
  } catch (err) {
    res.status(500).json({ error: "Watchlist fetch failed", details: err.message });
  }
});

app.post("/api/trading/watchlist", requireAuth, async (req, res) => {
  try {
    const user_id = requireBotAccess(req, res);
    if (!user_id) return;
    const { symbol, name, assetClass } = req.body || {};
    if (!symbol) return res.status(400).json({ error: "symbol required" });
    const sym = String(symbol).toUpperCase();
    // Re-adding something the user deliberately deleted must work, so clear
    // any tombstone on explicit add.
    try { db.prepare("DELETE FROM watchlist_tombstones WHERE user_id = ? AND list = 'main' AND symbol = ?").run(user_id, sym); } catch { /* ignore */ }
    const existing = db.prepare("SELECT * FROM trading_watchlist WHERE user_id = ? AND symbol = ?").get(user_id, sym);
    if (existing) {
      return res.status(409).json({ error: "Already in watchlist" });
    }
    const maxOrder = db.prepare("SELECT COALESCE(MAX(sort_order), 0) as max FROM trading_watchlist WHERE user_id = ?").get(user_id)?.max || 0;
    db.prepare("INSERT INTO trading_watchlist (user_id, symbol, name, asset_class, sort_order) VALUES (?, ?, ?, ?, ?)")
      .run(user_id, sym, name || sym, assetClass || "FOREX", maxOrder + 1);
    const item = db.prepare("SELECT * FROM trading_watchlist WHERE user_id = ? AND symbol = ?").get(user_id, sym);
    res.json({ ok: true, item });
  } catch (err) {
    res.status(500).json({ error: "Watchlist add failed", details: err.message });
  }
});

app.delete("/api/trading/watchlist/:symbol", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const sym = String(req.params.symbol).toUpperCase();
    const result = db.prepare("DELETE FROM trading_watchlist WHERE user_id = ? AND symbol = ?").run(user_id, sym);
    // Tombstone so the pair can never come back (stale cache / bot rows / an
    // offline op must not resurrect it).
    try {
      db.prepare("INSERT OR IGNORE INTO watchlist_tombstones (user_id, list, symbol) VALUES (?, 'main', ?)")
        .run(user_id, sym);
    } catch { /* ignore */ }
    if (result.changes === 0) return res.json({ ok: true, already: false });
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Watchlist remove failed", details: err.message });
  }
});

app.patch("/api/trading/watchlist/:symbol", requireAuth, async (req, res) => {
  try {
    // The 24/7 watch toggle is a bot operation — gated like the rest.
    const user_id = requireBotAccess(req, res);
    if (!user_id) return;
    const sym = String(req.params.symbol).toUpperCase();
    const { isEnabled, sortOrder } = req.body || {};
    const updates = [];
    const values = [];
    if (typeof isEnabled === "boolean") { updates.push("is_enabled = ?"); values.push(isEnabled ? 1 : 0); }
    if (typeof sortOrder === "number") { updates.push("sort_order = ?"); values.push(sortOrder); }
    if (updates.length === 0) return res.status(400).json({ error: "No updates provided" });
    db.prepare(`UPDATE trading_watchlist SET ${updates.join(", ")}, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND symbol = ?`).run(...values, user_id, sym);
    const item = db.prepare("SELECT * FROM trading_watchlist WHERE user_id = ? AND symbol = ?").get(user_id, sym);
    res.json({ ok: true, item });
  } catch (err) {
    res.status(500).json({ error: "Watchlist update failed", details: err.message });
  }
});

// --- Kelly lot-size attachment (item 9) ---
// Every published signal carries a lot-size recommendation computed from the
// user's OWN account balance and declared risk profile. The measured win-rate
// comes from the journal (the only real track record we have); when there is
// not enough history yet, the conservative default is used and the reason
// says so.
function measuredWinRate(symbol = null) {
  try {
    const row = symbol
      ? db.prepare("SELECT COUNT(*) AS n, SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) AS w FROM pullback_journal WHERE status='RESOLVED' AND symbol = ?").get(String(symbol).toUpperCase())
      : db.prepare("SELECT COUNT(*) AS n, SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) AS w FROM pullback_journal WHERE status='RESOLVED'").get();
    const n = row?.n || 0;
    if (n < 20) return { wr: 0.45, n, reliable: false };
    return { wr: (row.w || 0) / n, n, reliable: true };
  } catch { return { wr: 0.45, n: 0, reliable: false }; }
}

function attachLotSize(user_id, signal) {
  try {
    const acct = getTradingAccount(user_id);
    if (!acct) return signal;
    const m = measuredWinRate(signal.symbol);
    const sizing = recommendLotSize({
      symbol: signal.symbol,
      balance: acct.balance,
      entry: signal.entry,
      sl: signal.sl,
      tp: signal.tp1,
      rr: signal.rr,
      winRate: m.wr,
      profile: acct.riskProfile,
      pipConfig: PIP_CONFIG,
      confidence: signal.confidence,
    });
    signal.lotSize = sizing.lotSize || null;
    signal.lotReason = sizing.reason + (m.reliable
      ? ` (win rate ${(m.wr * 100).toFixed(0)}% over ${m.n} resolved setups)`
      : ` (default 45% win rate — only ${m.n} resolved setups so far)`);
    signal.riskUsd = sizing.riskUsd || null;
    signal.balance = acct.balance;
    signal.riskProfile = acct.riskProfile;
  } catch (e) {
    console.warn("[attachLotSize]", e.message);
  }
  return signal;
}

// --- Signal Scanning (batch analyze watchlist) ---
app.post("/api/trading/scan", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { symbols, forceRefresh } = req.body || {};
    let scanSymbols = symbols;
    if (!scanSymbols || !scanSymbols.length) {
      const watchlist = db.prepare("SELECT symbol FROM trading_watchlist WHERE user_id = ? AND is_enabled = 1 ORDER BY sort_order ASC").all(user_id);
      scanSymbols = watchlist.map(w => w.symbol);
    }
    if (!scanSymbols.length) return res.json({ signals: [], scannedAt: Date.now() });

    // Week 6 (F-45): bounded fan-out with a hard deadline. The old loop awaited
    // one symbol at a time with no budget, so a black-holed market-data call
    // blocked the whole scan past the app's timeout (measured: 2 symbols >30s).
    // Concurrency stays low (3) so the TwelveData free tier is not hammered.
    const { results: settled, summary } = await runBounded(
      scanSymbols.slice(0, 20),
      (sym) => mtfStrategy.analyze(sym, {}),
      { concurrency: 3, perItemMs: 12_000, totalMs: 20_000 }
    );

    const results = [];
    const missed = [];
    settled.forEach((r, i) => {
      const sym = scanSymbols[i];
      if (!r.ok) {
        console.warn(`[scan] ${sym} skipped (${r.timedOut ? "timeout" : r.error})`);
        missed.push({ symbol: sym, reason: r.timedOut ? "timeout" : "error" });
        return;
      }
      try {
        const signal = formatTradingSignal(sym, r.value);
        signal.botSource = "engine";
        // Attach the Kelly lot-size recommendation to EVERY signal (item 9):
        // sized against the user's own balance and declared risk profile.
        attachLotSize(user_id, signal);
        // Only persist meaningful signals — plain WAIT/DATA noise would flood
        // history. WAIT_PULLBACK / WAIT_RANGE carry actionable zones.
        const keep = ["BUY", "SELL", "WAIT_PULLBACK", "WAIT_RANGE"].includes(signal.decision);
        if (keep) storeSignalHistory(user_id, signal);
        results.push(signal);
      } catch (e) {
        console.warn(`[scan] ${sym} format failed:`, e.message);
        missed.push({ symbol: sym, reason: "error" });
      }
    });
    const alerts = results.filter(s => ["BUY", "SELL"].includes(s.decision));
    // `partial` tells the client the list is incomplete instead of silently
    // looking like "these 2 pairs are the whole scan".
    res.json({
      signals: results,
      alerts,
      scannedAt: Date.now(),
      partial: summary.timedOut + summary.skipped > 0,
      missed,
    });
  } catch (err) {
    res.status(500).json({ error: "Scan failed", details: err.message });
  }
});

// --- Trading account: balance, risk profile, own broker credentials ---
app.get("/api/trading/account", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    res.json({ account: getTradingAccount(user_id) });
  } catch (err) {
    res.status(500).json({ error: "Account fetch failed", details: err.message });
  }
});

app.post("/api/trading/account", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const account = saveTradingAccount(user_id, req.body || {});
    res.json({ ok: true, account });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// --- Lot size recommendation (item 9) ---
// Kelly-based sizing against the user's own balance + risk profile.
app.post("/api/trading/lot-size", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { symbol, entry, sl, tp, rr, winRate, confidence } = req.body || {};
    const acct = getTradingAccount(user_id);
    const sym = String(symbol || "").toUpperCase();
    if (!sym) return res.status(400).json({ error: "symbol required" });
    const out = recommendLotSize({
      symbol: sym,
      balance: acct?.balance || 1000,
      entry, sl, tp, rr,
      winRate: winRate != null ? Number(winRate) : undefined,
      profile: acct?.riskProfile || DEFAULT_RISK_PROFILE,
      pipConfig: PIP_CONFIG,
      confidence,
    });
    res.json({ ok: true, sizing: out, account: acct });
  } catch (err) {
    res.status(500).json({ error: "Lot size failed", details: err.message });
  }
});

app.get("/api/trading/risk-profiles", requireAuth, async (_req, res) => {
  try {
    res.json({
      profiles: Object.entries(RISK_PROFILE_INFO).map(([k, v]) => ({
        key: k, label: v.label, kellyFraction: v.kellyFraction,
      })),
      default: DEFAULT_RISK_PROFILE,
    });
  } catch (err) {
    res.status(500).json({ error: "Risk profiles failed", details: err.message });
  }
});

// --- Signal History ---
app.get("/api/trading/signals/history", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { symbol, limit = 50, favoritesOnly, unreadOnly } = req.query;
    let query = "SELECT * FROM trading_signals WHERE user_id = ?";
    const params = [user_id];
    if (symbol) { query += " AND symbol = ?"; params.push(String(symbol).toUpperCase()); }
    if (favoritesOnly === "true") { query += " AND is_favorite = 1"; }
    if (unreadOnly === "true") { query += " AND is_read = 0"; }
    query += " ORDER BY timestamp DESC LIMIT ?";
    params.push(Number(limit));
    const signals = db.prepare(query).all(...params).map(hydrateSignal);
    const total = db.prepare(`SELECT COUNT(*) as c FROM trading_signals WHERE user_id = ?${symbol ? " AND symbol = ?" : ""}${favoritesOnly === "true" ? " AND is_favorite = 1" : ""}${unreadOnly === "true" ? " AND is_read = 0" : ""}`).get(...params.slice(0, -1))?.c || 0;
    res.json({ signals, total });
  } catch (err) {
    res.status(500).json({ error: "Signal history failed", details: err.message });
  }
});

app.post("/api/trading/signals/:id/favorite", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const id = req.params.id;
    const { favorite } = req.body;
    db.prepare("UPDATE trading_signals SET is_favorite = ? WHERE id = ? AND user_id = ?").run(favorite ? 1 : 0, id, user_id);
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Favorite toggle failed", details: err.message });
  }
});

app.post("/api/trading/signals/:id/read", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const id = req.params.id;
    db.prepare("UPDATE trading_signals SET is_read = 1 WHERE id = ? AND user_id = ?").run(id, user_id);
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Mark read failed", details: err.message });
  }
});

app.delete("/api/trading/signals/:id", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const id = req.params.id;
    db.prepare("DELETE FROM trading_signals WHERE id = ? AND user_id = ?").run(id, user_id);
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Delete signal failed", details: err.message });
  }
});

// ==================== OPTIONS SUITE (paper signals only) =====================
// No execution rail exists for equity options in this build (MT5 cannot trade
// them; no Tradier/IBKR token configured), so these endpoints SCAN + RECORD.
// The phone's floating banner reads /options/scan and /options/fx-scan.

// --- Equity options scan: Strategy-A credit spreads on SPY/QQQ ---------------
app.post("/options/scan", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { symbols, balance } = req.body || {};
    const scanSymbols = (Array.isArray(symbols) && symbols.length ? symbols : ["SPY", "QQQ"])
      .map((s) => String(s).toUpperCase()).slice(0, 4);
    const bal = Number(balance) || 0;
    const signals = [], skipped = [];
    for (const sym of scanSymbols) {
      try {
        const chain = await getChain(sym);
        const closes = await fetchDailyCloses(sym, { years: 1.3 });
        const out = scanSymbol({ symbol: sym, chain, closes, db });
        if (out.signal) {
          const gate = optionsRiskCheck(user_id, out.signal, bal);
          out.signal.riskGate = gate;
          if (gate.ok) {
            storeOptionsSignal(user_id, out.signal);
            // Paper book: the same signal becomes a tracked paper position so
            // the ledger (not marketing copy) is what sells the subscription.
            const px = openPaperPosition(db, user_id, { ...out.signal, entrySpot: chain.underlying });
            if (px.skipped) out.signal.paperNote = px.why;
            else out.signal.paperId = px.id;
            signals.push(out.signal);
          } else {
            skipped.push({ symbol: sym, why: gate.reason });
          }
        } else {
          skipped.push({ symbol: sym, why: out.why });
        }
      } catch (e) {
        console.warn(`[options/scan] ${sym} failed:`, e.message);
        skipped.push({ symbol: sym, why: e.message });
      }
    }
    res.json({ signals, skipped, scannedAt: Date.now() });
  } catch (err) {
    res.status(500).json({ error: "Options scan failed", details: err.message });
  }
});

// --- FX model signals: expected-move bands + regime (synthetic, no chain) ----
app.post("/options/fx-scan", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { symbols } = req.body || {};
    const scanSymbols = (Array.isArray(symbols) && symbols.length ? symbols : ["EURUSD", "GBPUSD", "XAUUSD"])
      .map((s) => String(s).toUpperCase()).slice(0, 6);
    const signals = [], skipped = [];
    for (const sym of scanSymbols) {
      try {
        const closes = await fetchDailyCloses(sym, { years: 1 });
        const out = scanFx({ symbol: sym, closes });
        if (out.signal) {
          storeOptionsSignal(user_id, out.signal);
          signals.push(out.signal);
        } else skipped.push({ symbol: sym, why: out.why });
      } catch (e) {
        console.warn(`[options/fx-scan] ${sym} failed:`, e.message);
        skipped.push({ symbol: sym, why: e.message });
      }
    }
    res.json({ signals, skipped, scannedAt: Date.now() });
  } catch (err) {
    res.status(500).json({ error: "FX options scan failed", details: err.message });
  }
});

// --- Options signal history ---------------------------------------------------
app.get("/options/signals/history", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { symbol, limit = 50 } = req.query;
    let query = "SELECT * FROM options_signals WHERE user_id = ?";
    const params = [user_id];
    if (symbol) { query += " AND symbol = ?"; params.push(String(symbol).toUpperCase()); }
    query += " ORDER BY timestamp DESC LIMIT ?";
    params.push(Number(limit) || 50);
    ensureOptionsTables(db);
    const rows = db.prepare(query).all(...params).map((r) => ({
      ...r,
      legs: r.legs_json ? JSON.parse(r.legs_json) : null,
      breakevens: r.breakevens_json ? JSON.parse(r.breakevens_json) : null,
      exits: r.exits_json ? JSON.parse(r.exits_json) : null,
      provisional: r.provisional === 1, synthetic: r.synthetic === 1,
    }));
    res.json({ signals: rows, total: rows.length });
  } catch (err) {
    res.status(500).json({ error: "Options history failed", details: err.message });
  }
});

// --- Options paper ledger: the verifiable track record -----------------------
app.get("/options/ledger", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    // Mark before reporting so the book reflects live quotes, not last scan.
    let resolveNote = null;
    try {
      await resolvePaperBook(db, user_id, (sym) => getChain(sym));
    } catch (e) {
      resolveNote = `mark skipped (chain unreachable: ${e.message})`;
    }
    const ledger = ledgerSummary(db, user_id);
    if (resolveNote) ledger.resolveNote = resolveNote;
    res.json(ledger);
  } catch (err) {
    res.status(500).json({ error: "Options ledger failed", details: err.message });
  }
});

app.post("/options/paper/resolve", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const resolved = await resolvePaperBook(db, user_id, (sym) => getChain(sym));
    res.json({ resolved, ledger: ledgerSummary(db, user_id) });
  } catch (err) {
    res.status(500).json({ error: "Paper resolve failed", details: err.message });
  }
});

// --- Ledger maturity: which track records are old enough to sell ---------
app.get("/ledger/maturity", requireAuth, async (req, res) => {
  try {
    res.json(maturitySummary(db, boundUser(req)));
  } catch (err) {
    res.status(500).json({ error: "Maturity check failed", details: err.message });
  }
});

// Hands-free aging: resolve open options paper every 30 min so the books
// mature without anyone opening the app. Failures never crash the server.
setInterval(async () => {
  try {
    ensureOptionsTables(db);
    const owners = db.prepare("SELECT DISTINCT user_id AS u FROM options_paper WHERE status = 'OPEN'").all();
    for (const { u } of owners) {
      try {
        const r = await resolvePaperBook(db, u, (sym) => getChain(sym));
        if (r.length) console.log(`[options-paper] auto-resolve ${u}: ${r.filter((x) => x.closed).length} closed`);
      } catch (e) { console.warn("[options-paper] auto-resolve failed:", e.message); }
    }
  } catch (e) { console.warn("[options-paper] auto-resolve failed:", e.message); }
}, 30 * 60 * 1000).unref?.();

// --- Options backtest: modeled chains, bid/ask fills, full costs -------------
app.post("/options/backtest", requireAuth, limitCostly, async (req, res) => {
  try {
    const { symbol = "SPY", years = 3, walkForward: wf = true, cfg = {} } = req.body || {};
    const sym = String(symbol).toUpperCase();
    const closes = await fetchDailyCloses(sym, { years: Number(years) || 3 });
    if (closes.length < 150) return res.status(400).json({ error: "not enough history for backtest" });
    const out = wf ? walkForward(closes, {}) : { test: runBacktest(closes, cfg) };
    const test = out.test || out;
    res.json({
      symbol: sym, bars: closes.length,
      bestMarkup: out.bestMarkup ?? null, trainTotal: out.trainTotal ?? null,
      stats: test.stats,
      trades: test.trades.slice(0, 20),
      tradeCount: test.trades.length,
      modeledChain: true,
      warning: "Chains are modeled from realized vol (no $0 historical-chain source exists). Validates strategy logic and costs, not the VRP edge itself.",
    });
  } catch (err) {
    res.status(500).json({ error: "Backtest failed", details: err.message });
  }
});

// --- Portfolio ---
// P&L in USD for one position, using the SAME pip math for open and closed
// positions. FIXED BUG: open positions used `(px - entry) * lot * pipVal`,
// which dropped the /pipSize divisor — a 10-pip EURUSD move was reported as
// 0.1 pip of P&L, while closed positions (which used /pipSize) reported the
// truth. Open and closed P&L could therefore never agree, and the equity
// figure drifted. One formula now serves both.
function pnlUsdFor(symbol, action, entry, exit, lotSize) {
  const cfg = PIP_CONFIG[String(symbol || "").toUpperCase()];
  const e = Number(entry), x = Number(exit);
  const lot = Number(lotSize) || 0;
  if (!Number.isFinite(e) || !Number.isFinite(x) || !(lot > 0)) return 0;
  const isLong = String(action || "").toLowerCase() !== "sell";
  const dist = (isLong ? x - e : e - x);
  if (!cfg || !(Number(cfg.pipSize) > 0)) {
    // Unknown symbol: conservative dollar approximation (documented guess).
    return dist * lot * 10;
  }
  if (cfg.isCrypto) return dist * (cfg.pipValue || 1) * lot;
  return (dist / cfg.pipSize) * cfg.pipValue * lot;
}

app.get("/api/trading/portfolio", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    // The user's own stored balance is the base (persisted, survives reboot).
    const acct = getTradingAccount(user_id);
    const balance = Number(req.query.balance) > 0 ? Number(req.query.balance) : (acct?.balance || 1000);
    
    // Get open positions from PositionMonitor
    const openPositions = positionMonitor.list(500).filter(p => p.status === "MONITORING");

    // Get resolved positions for P&L — no slice cap: the portfolio must show
    // the WHOLE history, not the most recent 50.
    const resolvedPositions = positionMonitor.list(2000).filter(p => p.status === "RESOLVED");
    
    // Calculate metrics
    let unrealizedPnl = 0, realizedPnl = 0;
    const positions = [];
    // Garbage in (null entry/lot, dead price feed) used to poison the whole
    // summary with NaN, which the phone rendered literally. Sanitize here.
    const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);
    
    for (const pos of openPositions) {
      const entry = Number(pos.entry), lot = Number(pos.lot_size);
      const spot = await fetchSpotPrice(pos.symbol);
      const px = spot ? Number(spot.price) : NaN;
      if (!spot || !Number.isFinite(entry) || !Number.isFinite(lot) || lot <= 0 || !Number.isFinite(px)) {
        console.warn(`[portfolio] skipping ${pos.symbol} (bad price data)`);
        continue;
      }
      const pnl = pnlUsdFor(pos.symbol, pos.action, entry, px, lot);
      unrealizedPnl += pnl;
      positions.push({
        id: pos.id, symbol: pos.symbol, side: pos.action.toUpperCase(),
        entryPrice: entry, currentPrice: px, lotSize: lot,
        sl: num(pos.sl), tp: num(pos.tp), unrealizedPnl: num(pnl.toFixed(2)), realizedPnl: 0,
        status: "OPEN", openedAt: new Date(pos.placed_at).getTime() || 0, closedAt: null
      });
    }
    
    for (const pos of resolvedPositions) {
      const entry = Number(pos.entry), close = Number(pos.close_price), lot = Number(pos.lot_size);
      const pnl = pnlUsdFor(pos.symbol, pos.action, entry, close, lot);
      realizedPnl += pnl;
      positions.push({
        id: pos.id, symbol: pos.symbol, side: pos.action.toUpperCase(),
        entryPrice: num(entry), currentPrice: num(close), lotSize: lot,
        sl: num(pos.sl), tp: num(pos.tp), unrealizedPnl: 0, realizedPnl: num(pnl.toFixed(2)),
        status: pos.outcome === "win" ? "CLOSED_WIN" : "CLOSED_LOSS",
        openedAt: new Date(pos.placed_at).getTime() || 0, closedAt: new Date(pos.resolved_at).getTime() || 0,
        source: pos.source || "manual", reason: pos.reason || "",
      });
    }
    
    // Risk metrics from PortfolioRisk
    const riskStatus = portfolioRisk.status(balance);
    const uPnl = num(unrealizedPnl.toFixed(2)), rPnl = num(realizedPnl.toFixed(2));
    const tPnl = num((uPnl + rPnl).toFixed(2));
    
    res.json({
      positions,
      summary: {
        totalPositions: positions.filter(p => p.status === "OPEN").length,
        unrealizedPnl: uPnl,
        realizedPnl: rPnl,
        totalPnl: tPnl,
        balance,
        equity: num((balance + uPnl + rPnl).toFixed(2))
      },
      risk: riskStatus,
      account: acct,
    });
  } catch (err) {
    res.status(500).json({ error: "Portfolio fetch failed", details: err.message });
  }
});

// --- Regenerate account balance (item 6) ---
// Recomputes the balance from the ledger: base balance + realized P&L, then
// stores it as the new account balance. This is how a user re-bases their
// account after a manual deposit/withdrawal, or after a sizing bug fix.
app.post("/api/trading/portfolio/regenerate", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const acct = getTradingAccount(user_id);
    const startBalance = Number(req.body?.baseBalance) > 0
      ? Number(req.body.baseBalance)
      : (acct?.balance || 1000);
    // Recompute realized P&L across EVERY resolved position (no slice).
    const rows = positionMonitor.list(5000).filter(p => p.status === "RESOLVED");
    let realized = 0;
    for (const p of rows) realized += pnlUsdFor(p.symbol, p.action, p.entry, p.close_price, p.lot_size);
    // Deriv contracts carry their own broker-truth P&L.
    try {
      const dc = db.prepare("SELECT profit FROM deriv_contracts WHERE status = 'SETTLED' AND profit IS NOT NULL").all();
      for (const c of dc) realized += Number(c.profit) || 0;
    } catch { /* ignore */ }
    const newBalance = Math.max(0, startBalance + realized);
    const saved = saveTradingAccount(user_id, { balance: newBalance });
    res.json({
      ok: true,
      account: saved,
      startBalance,
      realized: Number(realized.toFixed(2)),
      newBalance,
      message: `Balance regenerated: ${startBalance.toFixed(2)} + ${realized.toFixed(2)} realized = ${newBalance.toFixed(2)}`,
    });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// --- Unified paper portfolio: every bot's executions in one ledger ---
// Each trade carries its bot identifier so P&L is attributable:
// intraday / longterm (positions.source bot:*), scheduler, manual,
// options (options_paper), pocket (pocket_paper, P&L in payout units).
// NOTE: this is a PAPER ledger (bridge unset = paper; MT5 live fills are
// tracked by the same monitor when the bridge is set).
function spotPnlUsd(symbol, side, entry, exitPrice, lotSize) {
  const cfg = PIP_CONFIG[String(symbol || "").toUpperCase()];
  const isLong = String(side || "").toUpperCase().startsWith("BUY") || String(side).toLowerCase() === "buy" || String(side).toUpperCase() === "CALL";
  const diff = isLong ? exitPrice - entry : entry - exitPrice;
  if (!Number.isFinite(diff) || !Number.isFinite(entry)) return 0;
  if (!cfg) return diff * (lotSize || 0.01) * 10;
  if (cfg.isCrypto) return diff * (lotSize || 0.01) * cfg.pipValue;
  return (diff / cfg.pipSize) * cfg.pipValue * (lotSize || 0.01);
}

app.get("/api/trading/paper-portfolio", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const trades = [];
    const byBot = {};

    const add = (t) => {
      trades.push(t);
      const b = (byBot[t.bot] = byBot[t.bot] || { bot: t.bot, trades: 0, wins: 0, losses: 0, pnlUsd: 0, pnlUnits: 0 });
      b.trades++;
      if (t.status === "WIN" || t.status === "CLOSED_WIN") b.wins++;
      if (t.status === "LOSS" || t.status === "CLOSED_LOSS") b.losses++;
      if (t.pnlKind === "usd") b.pnlUsd += t.pnl || 0;
      else b.pnlUnits += t.pnl || 0;
    };
    const botOfSource = (s) => {
      const v = String(s || "");
      // Bot sources are tagged "bot:<name>" — cover every bot we run.
      if (v.startsWith("bot:")) {
        const n = v.slice(4).split(":")[0];
        return n || "bot";
      }
      if (v === "scheduler") return "scheduler";
      if (v === "bots") return "intraday";
      if (v.startsWith("pocket-deriv")) return "pocket";
      if (v.startsWith("deriv-")) return "deriv";
      if (v.startsWith("alpaca")) return "alpaca";
      if (v.startsWith("options")) return "options";
      if (v.startsWith("engine")) return "engine";
      if (v.startsWith("manual")) return "manual";
      return "manual";
    };

    // 1. Spot/CFD positions (intraday, longterm, scalping, scheduler, manual).
    try {
      const all = positionMonitor.list(5000);
      for (const p of all) {
        const open = p.status === "MONITORING";
        let exit = p.close_price ?? null;
        if (open) {
          const spot = await fetchSpotPrice(p.symbol);
          if (spot) exit = spot.price;
        }
        const pnl = exit != null && p.entry != null ? spotPnlUsd(p.symbol, p.action, p.entry, exit, p.lot_size) : 0;
        add({
          bot: botOfSource(p.source), source: p.source || "manual",
          symbol: p.symbol, side: String(p.action || "").toUpperCase(),
          entry: p.entry, exit, pnl, pnlKind: "usd",
          status: open ? "OPEN" : (p.outcome === "win" ? "CLOSED_WIN" : "CLOSED_LOSS"),
          openedAt: p.placed_at ? new Date(p.placed_at).getTime() : null,
          closedAt: p.resolved_at ? new Date(p.resolved_at).getTime() : null,
          note: p.note || p.reason || "",
        });
      }
    } catch (e) { console.warn("[paper-portfolio] positions:", e.message); }

    // 2. Options paper book.
    try {
      const rows = db.prepare("SELECT * FROM options_paper WHERE user_id IN (?, 'bot') ORDER BY opened_at DESC LIMIT 200").all(user_id);
      for (const r of rows) {
        const closed = r.status !== "OPEN";
        add({
          bot: "options", source: "options-paper",
          symbol: r.symbol, side: r.kind || "SPREAD",
          entry: r.credit, exit: closed ? (r.pnl != null ? r.credit + r.pnl / 100 / (r.contracts || 1) : null) : null,
          pnl: closed ? (r.pnl || 0) : (r.unrealized || 0), pnlKind: "usd",
          status: closed ? (r.pnl >= 0 ? "CLOSED_WIN" : "CLOSED_LOSS") : "OPEN",
          openedAt: r.opened_at, closedAt: r.exit_time,
          note: closed ? (r.exit_why || "") : `max loss $${r.max_loss ?? "?"}`,
        });
      }
    } catch (e) { console.warn("[paper-portfolio] options:", e.message); }

    // 3. Pocket paper book (units of payout: WIN=+$payout, LOSS=-$1).
    try {
      const rows = db.prepare("SELECT * FROM pocket_paper WHERE user_id = ? AND result != 'OPEN' ORDER BY entry_ts DESC LIMIT 200").all(user_id);
      for (const r of rows) {
        add({
          bot: "pocket", source: "pocket-paper",
          symbol: r.symbol, side: r.direction,
          entry: r.entry_price, exit: r.settle_price,
          pnl: r.result === "WIN" ? r.payout : r.result === "LOSS" ? -1 : 0, pnlKind: "units",
          status: r.result,
          openedAt: r.entry_ts, closedAt: r.expiry_ts,
          note: `${r.expiry_min}m expiry`,
        });
      }
      const openN = db.prepare("SELECT COUNT(*) AS c FROM pocket_paper WHERE user_id = ? AND result = 'OPEN'").get(user_id)?.c || 0;
      if (openN) {
        const b = (byBot.pocket = byBot.pocket || { bot: "pocket", trades: 0, wins: 0, losses: 0, pnlUsd: 0, pnlUnits: 0 });
        b.openTrades = openN;
      }
    } catch (e) { console.warn("[paper-portfolio] pocket:", e.message); }

    // 4. Alpaca fills + live positions (broker truth, both modes).
    try {
      const [pp, lp] = await Promise.all([
        alpacaPaper.positions().catch(() => []),
        (ALPACA_LIVE ? alpacaLive.positions().catch(() => []) : Promise.resolve([])),
      ]);
      for (const [arr, mode] of [[pp, "paper"], [lp, "live"]]) {
        for (const p of arr) {
          add({
            bot: mode === "live" ? "alpaca-live" : "alpaca", source: `alpaca-${mode}`,
            symbol: p.symbol, side: Number(p.qty) >= 0 ? "BUY" : "SELL",
            entry: Number(p.avg_entry_price) || 0, exit: Number(p.current_price) || 0,
            pnl: Number(p.unrealized_pl) || 0, pnlKind: "usd",
            status: "OPEN", openedAt: null, closedAt: null,
            note: `qty ${p.qty} · ${mode}`,
          });
        }
      }
    } catch (e) { console.warn("[paper-portfolio] alpaca:", e.message); }

    // 5. DERIV CONTRACTS — the broker-truth book. The old portfolio only
    //    showed pocket + positions, so Deriv fills (the rail the bots
    //    actually trade on) were invisible. Every contract the executor
    //    bought appears here, open or settled, with its real P&L.
    try {
      const rows = db.prepare("SELECT * FROM deriv_contracts ORDER BY placed_at DESC LIMIT 1000").all();
      for (const c of rows) {
        const settled = c.status === "SETTLED";
        const profit = Number(c.profit) || 0;
        add({
          bot: c.direction === "MULTUP" || c.direction === "MULTDN" ? "deriv" : "pocket",
          source: `deriv-${String(c.mode || "demo")}`,
          symbol: c.symbol,
          side: c.direction,
          entry: c.entry_tick, exit: null,
          pnl: settled ? profit : 0, pnlKind: "usd",
          status: settled ? (profit > 0 ? "CLOSED_WIN" : "CLOSED_LOSS") : "OPEN",
          openedAt: c.placed_at, closedAt: settled ? c.placed_at : null,
          note: `${c.stake} stake · ${c.duration_min ? c.duration_min + "m" : "MULT"} · ${c.mode} · ${c.deriv_symbol || c.symbol}`,
        });
      }
    } catch (e) { console.warn("[paper-portfolio] deriv:", e.message); }

    // 6. JOURNAL SETUPS — every published setup with its outcome. This is
    //    the "did my trades actually work" record across ALL engines.
    try {
      const rows = db.prepare("SELECT * FROM pullback_journal ORDER BY recorded_at DESC LIMIT 1000").all();
      for (const j of rows) {
        const resolved = j.status === "RESOLVED";
        add({
          bot: "engine", source: "journal",
          symbol: j.symbol, side: j.direction,
          entry: j.entry, exit: null,
          pnl: resolved ? (j.outcome === "win" ? (j.planned_rr || 0) : -1) : 0,
          pnlKind: "units", // R-multiples, not dollars
          status: resolved ? (j.outcome === "win" ? "CLOSED_WIN" : "CLOSED_LOSS")
            : (j.status === "EXPIRED" ? "EXPIRED" : "OPEN"),
          openedAt: j.recorded_at, closedAt: j.resolved_at || null,
          note: `${j.strategy} · ${j.outcome || j.status} · R${(j.planned_rr ?? 0).toFixed(2)}`,
        });
      }
    } catch (e) { console.warn("[paper-portfolio] journal:", e.message); }

    trades.sort((a, b) => (b.openedAt || 0) - (a.openedAt || 0));
    // Headline performance per bot, so the portfolio reads like a real book
    // rather than a raw trade dump.
    for (const b of Object.values(byBot)) {
      const closed = b.wins + b.losses;
      b.winRate = closed ? Number(((b.wins / closed) * 100).toFixed(1)) : null;
      b.closed = closed;
      b.open = Math.max(0, b.trades - closed);
      b.pnlUsd = Number((b.pnlUsd || 0).toFixed(2));
      b.pnlUnits = Number((b.pnlUnits || 0).toFixed(2));
    }
    const acct = getTradingAccount(user_id);
    const totalUsd = Object.values(byBot).reduce((a, b) => a + (b.pnlUsd || 0), 0);
    res.json({
      trades: trades.slice(0, 1000),
      byBot,
      total: trades.length,
      totalPnlUsd: Number(totalUsd.toFixed(2)),
      balance: acct?.balance || 1000,
      equity: Number(((acct?.balance || 1000) + totalUsd).toFixed(2)),
      account: acct,
      serverTime: Date.now(),
    });
  } catch (err) {
    res.status(500).json({ error: "Paper portfolio failed", details: err.message });
  }
});

// --- Risk Management ---
app.get("/api/trading/risk", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const balance = Number(req.query.balance) || 1000;
    const riskStatus = portfolioRisk.status(balance);
    res.json(riskStatus);
  } catch (err) {
    res.status(500).json({ error: "Risk status failed", details: err.message });
  }
});

// --- Sentiment Analysis (cached daily) ---
app.get("/api/trading/sentiment", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { symbols } = req.query;
    let symList = symbols ? String(symbols).split(",").map(s => s.trim().toUpperCase()) : [];
    
    if (!symList.length) {
      const watchlist = db.prepare("SELECT symbol FROM trading_watchlist WHERE user_id = ? AND is_enabled = 1").all(user_id);
      symList = watchlist.map(w => w.symbol);
    }
    
    const sentiment = {};
    const today = new Date().toISOString().slice(0, 10);
    for (const sym of symList.slice(0, 10)) {
      const cacheKey = `sentiment:${sym}:${today}`;
      let data = cacheGet(cacheKey);
      if (!data) {
        // SQLite daily cache survives restarts (memory cache does not)
        try {
          const row = db.prepare("SELECT data, cached_at FROM sentiment_cache WHERE symbol = ?").get(`${sym}:${today}`);
          if (row) { data = JSON.parse(row.data); cacheSet(cacheKey, data, 24 * 60 * 60 * 1000); }
        } catch { /* cache miss */ }
      }
      if (!data) {
        data = await fetchSentiment(sym);
        cacheSet(cacheKey, data, 24 * 60 * 60 * 1000);
        try {
          db.prepare("INSERT OR REPLACE INTO sentiment_cache (symbol, data, cached_at) VALUES (?, ?, ?)").run(`${sym}:${today}`, JSON.stringify(data), Date.now());
          // Prune caches older than 7 days
          db.prepare("DELETE FROM sentiment_cache WHERE cached_at < ?").run(Date.now() - 7 * 24 * 3600 * 1000);
        } catch { /* best-effort */ }
      }
      sentiment[sym] = data;
    }
    res.json({ sentiment, cachedAt: Date.now() });
  } catch (err) {
    res.status(500).json({ error: "Sentiment fetch failed", details: err.message });
  }
});

// --- Explainable AI ---
app.post("/api/trading/ai/explain", requireAuth, async (req, res) => {
  try {
    const { signalId, detailLevel = "full" } = req.body || {};
    if (!signalId) return res.status(400).json({ error: "signalId required" });
    
    const signal = db.prepare("SELECT * FROM trading_signals WHERE id = ?").get(signalId);
    if (!signal) return res.status(404).json({ error: "Signal not found" });
    
    const explanation = await generateSignalExplanation(signal, detailLevel);
    res.json(explanation);
  } catch (err) {
    res.status(500).json({ error: "Explanation failed", details: err.message });
  }
});

// --- AI Recommendations ---
app.post("/api/trading/ai/recommendations", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { riskProfile = "MODERATE", capital = 10000 } = req.body || {};
    
    const watchlist = db.prepare("SELECT symbol FROM trading_watchlist WHERE user_id = ? AND is_enabled = 1").all(user_id);
    const symbols = watchlist.map(w => w.symbol);
    
    const recommendations = [];
    for (const sym of symbols.slice(0, 5)) {
      const result = await mtfStrategy.analyze(sym, { balance: capital, riskPercent: riskProfile === "CONSERVATIVE" ? 0.5 : riskProfile === "AGGRESSIVE" ? 2 : 1 });
      if (result.decision === "BUY" || result.decision === "SELL") {
        recommendations.push({
          symbol: sym,
          action: result.decision,
          confidence: result.confidence,
          entry: result.entry,
          sl: result.sl,
          tp1: result.tp,
          tp2: result.tp2,
          rr: result.rr,
          reasoning: result.reason,
          regime: result.regime?.macro_trend,
          positionSize: calculateLotSize({ symbol: sym, balance: capital, riskPercent: 1, entry: result.entry, stopLoss: result.sl })
        });
      }
    }
    
    res.json({ recommendations, generatedAt: Date.now() });
  } catch (err) {
    res.status(500).json({ error: "Recommendations failed", details: err.message });
  }
});

// --- Learning Hub ---
app.get("/api/trading/learning", requireAuth, async (req, res) => {
  try {
    const { category } = req.query;
    const content = getLearningContent(category);
    res.json({ content });
  } catch (err) {
    res.status(500).json({ error: "Learning content failed", details: err.message });
  }
});

// --- Push Notification Registration ---
app.post("/api/trading/notifications/register", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { token, platform } = req.body || {};
    if (!token) return res.status(400).json({ error: "token required" });
    db.prepare("INSERT OR REPLACE INTO push_tokens (user_id, token, platform, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)").run(user_id, token, platform || "android");
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Token registration failed", details: err.message });
  }
});

app.post("/api/trading/notifications/preferences", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { signalAlerts, tpSlAlerts, newsAlerts, dailySummary } = req.body || {};
    db.prepare("INSERT OR REPLACE INTO notification_prefs (user_id, signal_alerts, tp_sl_alerts, news_alerts, daily_summary) VALUES (?, ?, ?, ?, ?)")
      .run(user_id, signalAlerts ? 1 : 0, tpSlAlerts ? 1 : 0, newsAlerts ? 1 : 0, dailySummary ? 1 : 0);
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Preferences update failed", details: err.message });
  }
});

// --- Price-threshold / TP-SL proximity alerts ---
// Client polls this with its open signals; server compares live spot vs
// entry/SL/TP1/TP2 and reports touches so the app can fire notifications.
app.post("/api/trading/alerts/check", requireAuth, async (req, res) => {
  try {
    const { signals } = req.body || {};
    if (!Array.isArray(signals) || !signals.length) return res.json({ alerts: [] });
    const alerts = [];
    for (const s of signals.slice(0, 20)) {
      try {
        const sym = String(s.symbol || "").toUpperCase();
        const spot = await fetchSpotPrice(sym);
        if (!spot) continue;
        const px = spot.price;
        const near = (level) => {
          if (level == null || !Number.isFinite(Number(level))) return false;
          const lv = Number(level);
          const tol = Math.max(Math.abs(px) * 0.0005, 1e-9); // 0.05% touch band
          return Math.abs(px - lv) <= tol;
        };
        if (near(s.tp1)) alerts.push({ symbol: sym, kind: "TP1_HIT", price: px, signalId: s.id || null });
        else if (near(s.tp2)) alerts.push({ symbol: sym, kind: "TP2_HIT", price: px, signalId: s.id || null });
        else if (near(s.sl)) alerts.push({ symbol: sym, kind: "SL_HIT", price: px, signalId: s.id || null });
        else if (near(s.entry)) alerts.push({ symbol: sym, kind: "ENTRY_TOUCH", price: px, signalId: s.id || null });
      } catch { /* per-symbol best effort */ }
    }
    res.json({ alerts, checkedAt: Date.now() });
  } catch (err) {
    res.status(500).json({ error: "Alert check failed", details: err.message });
  }
});

// --- Live feed status + manual resync ---
app.get("/api/trading/feed/status", requireAuth, async (_req, res) => {
  try {
    res.json({ ...liveFeed.status(), serverTime: Date.now() });
  } catch (err) {
    res.status(500).json({ error: "Feed status failed", details: err.message });
  }
});

app.post("/api/trading/feed/resync", requireAuth, async (_req, res) => {
  try {
    await liveFeed.resync();
    res.json({ ok: true, ...liveFeed.status() });
  } catch (err) {
    res.status(500).json({ error: "Feed resync failed", details: err.message });
  }
});

// --- Pocket watchlist (SEPARATE from the main watchlist) ---
app.get("/api/trading/pocket/watchlist", requireAuth, async (req, res) => {
  try {
    if (!requirePocket(req, res)) return;
    const user_id = boundUser(req);
    let items = db.prepare("SELECT * FROM pocket_watchlist WHERE user_id = ? ORDER BY sort_order ASC").all(user_id);
    try {
      const dead = db.prepare("SELECT symbol FROM watchlist_tombstones WHERE user_id = ? AND list = 'pocket'").all(user_id).map(r => r.symbol);
      if (dead.length) items = items.filter(w => !dead.includes(w.symbol));
    } catch { /* no tombstones yet */ }
    res.json({ items });
  } catch (err) {
    res.status(500).json({ error: "Pocket watchlist failed", details: err.message });
  }
});

app.post("/api/trading/pocket/watchlist", requireAuth, async (req, res) => {
  try {
    if (!requirePocket(req, res)) return;
    const user_id = boundUser(req);
    const sym = String(req.body?.symbol || "").toUpperCase().trim();
    if (!sym) return res.status(400).json({ error: "symbol required" });
    const expiryMin = Math.min(60, Math.max(1, Number(req.body?.expiry) || POCKET_EXPIRY_MIN));
    try { db.prepare("DELETE FROM watchlist_tombstones WHERE user_id = ? AND list = 'pocket' AND symbol = ?").run(user_id, sym); } catch { /* ignore */ }
    const dup = db.prepare("SELECT id FROM pocket_watchlist WHERE user_id = ? AND symbol = ?").get(user_id, sym);
    if (dup) {
      db.prepare("UPDATE pocket_watchlist SET expiry_min = ?, is_enabled = 1 WHERE id = ?").run(expiryMin, dup.id);
      return res.json({ ok: true, item: db.prepare("SELECT * FROM pocket_watchlist WHERE id = ?").get(dup.id) });
    }
    const maxOrder = db.prepare("SELECT COALESCE(MAX(sort_order),0) AS m FROM pocket_watchlist WHERE user_id = ?").get(user_id)?.m || 0;
    db.prepare("INSERT INTO pocket_watchlist (user_id, symbol, expiry_min, sort_order) VALUES (?, ?, ?, ?)")
      .run(user_id, sym, expiryMin, maxOrder + 1);
    // Pocket pairs also get engine candles (feed + backfill), but stay out of
    // the MAIN watchlist table — the two lists never correlate.
    try { await liveFeed._backfill(sym); await liveFeed._backfillPocket(sym); } catch { /* best effort */ }
    res.json({ ok: true, item: db.prepare("SELECT * FROM pocket_watchlist WHERE user_id = ? AND symbol = ?").get(user_id, sym) });
  } catch (err) {
    res.status(500).json({ error: "Pocket watchlist add failed", details: err.message });
  }
});

app.delete("/api/trading/pocket/watchlist/:symbol", requireAuth, async (req, res) => {
  try {
    if (!requirePocket(req, res)) return;
    const user_id = boundUser(req);
    const sym = String(req.params.symbol).toUpperCase();
    db.prepare("DELETE FROM pocket_watchlist WHERE user_id = ? AND symbol = ?").run(user_id, sym);
    try {
      db.prepare("INSERT OR IGNORE INTO watchlist_tombstones (user_id, list, symbol) VALUES (?, 'pocket', ?)")
        .run(user_id, sym);
    } catch { /* ignore */ }
    // Stop scanning it immediately (also drops any pending pocket arm).
    try {
      liveFeed.unwatchPocket(sym);
      clearArm(db, sym, 1); clearArm(db, sym, 5); clearArm(db, sym, 15);
    } catch { /* ignore */ }
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Pocket watchlist remove failed", details: err.message });
  }
});

// --- Pocket fixed-expiry signals (manual execution; paper-tracked) ---
// PRO-GATED: pocket is the flagship — trader/pro/tester/owner only.
// Free/starter/creator get 402 + upgrade message (no silent failure).
function requirePocket(req, res) {
  const user_id = boundUser(req);
  const ent = checkEntitlement(user_id, "pocket");
  if (!ent.ok) {
    res.status(402).json({ error: ent.message, code: ent.code || "feature_locked", feature: "pocket" });
    return null;
  }
  return user_id;
}
app.get("/api/trading/pocket/signal", requireAuth, async (req, res) => {
  try {
    const user_id = requirePocket(req, res);
    if (!user_id) return;
    const sym = String(req.query.symbol || "").toUpperCase().replace(/[^A-Z0-9_.]/g, "");
    if (!sym) return res.status(400).json({ error: "symbol required — pick or add a pair first" });
    const expiryMin = Math.min(60, Math.max(1, Number(req.query.expiry) || POCKET_EXPIRY_MIN));
    const signal = pocketScan(sym, expiryMin);
    const arm = getArm(db, sym, signal.expiryMin);
    if (signal.reasons[0]?.startsWith("warming up")) {
      // Kick off a background backfill so the next poll is warm (Finnhub free
      // first, then budgeted REST — same path as the watchlist resync).
      liveFeed._backfillPocket(sym).catch(() => {});
    }
    res.json({ signal, arm, enabled: pocketUserEnabled(user_id), payoutAssumed: POCKET_PAYOUT, serverTime: Date.now() });
  } catch (err) {
    res.status(500).json({ error: "Pocket signal failed", details: err.message });
  }
});

// Pocket bot ON/OFF switch (per user). GET reads, POST {enabled:bool} writes.
app.get("/api/trading/pocket/toggle", requireAuth, (req, res) => {
  const user_id = requirePocket(req, res);
  if (!user_id) return;
  res.json({ ok: true, enabled: pocketUserEnabled(user_id) });
});
app.post("/api/trading/pocket/toggle", requireAuth, (req, res) => {
  try {
    const user_id = requirePocket(req, res);
    if (!user_id) return;
    const on = req.body?.enabled === true || req.body?.enabled === 1 || req.body?.enabled === "true";
    pocketSetEnabled(user_id, on);
    res.json({ ok: true, enabled: on });
  } catch (err) {
    res.status(500).json({ error: "Pocket toggle failed", details: err.message });
  }
});

app.get("/api/trading/pocket/history", requireAuth, async (req, res) => {
  try {
    const user_id = requirePocket(req, res);
    if (!user_id) return;
    const { symbol, limit = 50 } = req.query;
    let q = "SELECT * FROM pocket_paper WHERE (user_id = ? OR user_id = '__pocket__')";
    const params = [user_id];
    if (symbol) { q += " AND symbol = ?"; params.push(String(symbol).toUpperCase()); }
    q += " ORDER BY entry_ts DESC LIMIT ?";
    params.push(Math.min(200, Number(limit) || 50));
    res.json({ trades: db.prepare(q).all(...params) });
  } catch (err) {
    res.status(500).json({ error: "Pocket history failed", details: err.message });
  }
});

app.get("/api/trading/alpaca/status", requireAuth, async (_req, res) => {
  try {
    res.json({ ...(await alpacaExecutor.status()), serverTime: Date.now() });
  } catch (err) {
    res.status(500).json({ error: "Alpaca status failed", details: err.message });
  }
});

app.get("/api/trading/deriv/status", requireAuth, async (req, res) => {
  try {
    if (!requirePocket(req, res)) return;
    const bal = _derivToken ? await derivClient.balance().catch(() => null) : null;
    // Diagnostic detail: WHY did nothing execute? (silent refusals are the
    // #1 support question — answer it here, not in server logs.)
    const diag = {
      tokenPresent: !!_derivToken,
      tokenLooksDemo: /^VRTC/i.test(_derivToken || ""),
      autoEnabled: POCKET_DERIV_AUTO,
      liveFlag: DERIV_LIVE,
      account: derivClient.account,
      lastRefusals: derivExecutor.refusals(20),
    };
    let liveSymbols = [];
    // Only attempt the broker round-trip when a token exists — otherwise the
    // status call hangs on a socket nobody will answer.
    if (_derivToken) {
      try { liveSymbols = (await derivClient.activeSymbols()).filter(s => !SYNTHETIC_ONLY.test(s)).slice(0, 400); } catch { /* ignore */ }
    }
    res.json({
      ...derivExecutor.status(), balance: bal, diag,
      supportedSymbols: liveSymbols,
      serverTime: Date.now(),
    });
  } catch (err) {
    res.status(500).json({ error: "Deriv status failed", details: err.message });
  }
});

// Manual single-contract test (user-initiated, demo only) — proves the whole
// chain works end-to-end: token -> symbol verification -> buy -> ledger.
app.post("/api/trading/deriv/test", requireAuth, async (req, res) => {
  try {
    if (!requirePocket(req, res)) return;
    const symbol = String(req.body?.symbol || "XAUUSD").toUpperCase();
    const direction = req.body?.direction === "PUT" ? "PUT" : "CALL";
    const durationMin = Math.min(60, Math.max(1, Number(req.body?.duration) || 5));
    const stake = Math.min(50, Math.max(0.35, Number(req.body?.stake) || 0.5));
    const live = DERIV_LIVE && DERIV_REAL_TOKEN;
    const ex = new DerivExecutor({
      db, client: derivClient, liveAllowed: live, stake,
      maxPerDay: 100, // a manual test bypasses the daily auto cap on purpose
      log: (...a) => console.log(...a),
    });
    const r = await ex.execute({ symbol, direction, durationMin, positionId: null });
    res.json(r.ok ? { ok: true, ...r } : { ok: false, error: r.why });
  } catch (err) {
    res.status(500).json({ error: "Deriv test failed", details: err.message });
  }
});

// ---- Deriv account management (REST) ----
const DERIV_ERROR = (r) => ({ error: r?.error?.message || "Deriv request failed", code: r?.error?.code || null });

app.get("/api/trading/deriv/account", requireAuth, async (_req, res) => {
  try {
    if (!derivRest.token) return res.status(400).json({ error: "DERIV_TOKEN unset — account management unavailable" });
    const r = await derivRest.getSettings();
    res.json(r.error ? DERIV_ERROR(r) : { settings: r });
  } catch (err) { res.status(500).json({ error: "Deriv account failed", details: err.message }); }
});

app.get("/api/trading/deriv/account/statistics", requireAuth, async (_req, res) => {
  try {
    const r = await derivRest.getAccountStatistics();
    res.json(r.error ? DERIV_ERROR(r) : { statistics: r });
  } catch (err) { res.status(500).json({ error: "Deriv statistics failed", details: err.message }); }
});

app.get("/api/trading/deriv/reality-check", requireAuth, async (_req, res) => {
  try {
    const r = await derivRest.getRealityCheck();
    res.json(r.error ? DERIV_ERROR(r) : { realityCheck: r });
  } catch (err) { res.status(500).json({ error: "Deriv reality check failed", details: err.message }); }
});

app.get("/api/trading/deriv/self-exclusion", requireAuth, async (_req, res) => {
  try {
    const r = await derivRest.getSelfExclusion();
    res.json(r.error ? DERIV_ERROR(r) : { selfExclusion: r });
  } catch (err) { res.status(500).json({ error: "Deriv self-exclusion failed", details: err.message }); }
});

app.post("/api/trading/deriv/self-exclusion", requireAuth, async (req, res) => {
  try {
    const r = await derivRest.setSelfExclusion(req.body || {});
    res.json(r.error ? DERIV_ERROR(r) : { updated: r });
  } catch (err) { res.status(500).json({ error: "Deriv self-exclusion update failed", details: err.message }); }
});

app.get("/api/trading/deriv/statement", requireAuth, async (req, res) => {
  try {
    const limit = Math.min(Number(req.query.limit) || 50, 500);
    const r = await derivRest.statement({ limit });
    res.json(r.error ? DERIV_ERROR(r) : { statement: r });
  } catch (err) { res.status(500).json({ error: "Deriv statement failed", details: err.message }); }
});

// ---- Deriv payments ----
app.get("/api/trading/deriv/payments/agents", requireAuth, async (req, res) => {
  try {
    const country = String(req.query.country || "UG");
    const r = await derivRest.paymentAgents(country);
    res.json(r.error ? DERIV_ERROR(r) : { agents: r });
  } catch (err) { res.status(500).json({ error: "Payment agents failed", details: err.message }); }
});

app.get("/api/trading/deriv/payments/wallets", requireAuth, async (_req, res) => {
  try {
    const r = await derivRest.wallets();
    res.json(r.error ? DERIV_ERROR(r) : { wallets: r });
  } catch (err) { res.status(500).json({ error: "Wallets failed", details: err.message }); }
});

// ---- Deriv application management + markup statistics ----
app.get("/api/trading/deriv/apps", requireAuth, async (_req, res) => {
  try {
    const r = await derivRest.appList();
    res.json(r.error ? DERIV_ERROR(r) : { applications: r });
  } catch (err) { res.status(500).json({ error: "Deriv apps failed", details: err.message }); }
});

app.get("/api/trading/deriv/apps/markup", requireAuth, async (req, res) => {
  try {
    const to = String(req.query.to || new Date().toISOString().slice(0, 10));
    const from = String(req.query.from || new Date(Date.now() - 30 * 86400000).toISOString().slice(0, 10));
    const r = await derivRest.appMarkupStatistics(from, to);
    res.json(r.error ? DERIV_ERROR(r) : { markupStatistics: r });
  } catch (err) { res.status(500).json({ error: "Deriv markup statistics failed", details: err.message }); }
});

app.get("/api/trading/pocket/stats", requireAuth, async (req, res) => {
  try {
    if (!requirePocket(req, res)) return;
    const { symbol } = req.query;
    res.json(pocketStats(db, { symbol: symbol || null, payout: POCKET_PAYOUT }));
  } catch (err) {
    res.status(500).json({ error: "Pocket stats failed", details: err.message });
  }
});

// --- Offline Sync ---
app.get("/api/trading/sync", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { since } = req.query;
    const sinceTs = since ? Number(since) : 0;
    
    const watchlist = db.prepare("SELECT * FROM trading_watchlist WHERE user_id = ? AND updated_at > ?").all(user_id, sinceTs);
    const signals = db.prepare("SELECT * FROM trading_signals WHERE user_id = ? AND timestamp > ? ORDER BY timestamp DESC LIMIT 100").all(user_id, sinceTs);
    const prefs = db.prepare("SELECT * FROM notification_prefs WHERE user_id = ?").get(user_id);
    
    res.json({ watchlist, signals, prefs, serverTime: Date.now() });
  } catch (err) {
    res.status(500).json({ error: "Sync failed", details: err.message });
  }
});

// ====== ERROR HANDLER ======
app.use((err, _req, res, _next) => {
  console.error("Unhandled Error!", err);
  res.status(500).json({ error: "Internal server error", details: err?.message || "Unknown error" });
});

// ====== START ======
const server = app.listen(PORT, () => {
  console.log(`
FRIT - engine.js Trading Engine & OpenRouter AI Orchestrator
Port : ${String(PORT).padEnd(5)}
Single trading engine (server/src/strategy/engine.js):
 - 30M primary + 4H confirmation (EMA 9/21 + RSI + DI, ADX percentile gate)
 - News filter (FF calendar — 30 min advisory)
 - Multi-timeframe confirmation (4H trend regime)
Single OpenRouter gateway:
 - GLM-5.3-Flash — chat + agentic + tools + coding (primary)
 - gpt-oss-20b — fast/verify/router
 - whisper-large-v3 — STT (accuracy pick)
 - bytedance-seed/seedream-5-0-flash — image gen
Infrastructure:
 - Lot size engine (per-pair pip math)
 - MT5 bridge (live) or paper mode
 - Code sandbox (Python/JS)
 - Android automation (State Machine: Verify-Act)
 - Screen frame ingestion -> vision analysis
  `);
});

// ====== GRACEFUL SHUTDOWN (fix F-01) ======
// Every Render/Railway deploy sends SIGTERM. Previously nothing handled it:
// the process was killed mid-flight and better-sqlite3 could truncate an
// in-progress write — exactly where wallet/credit debits live. Now: stop
// accepting requests, drain them, checkpoint WAL, close the DB, exit 0.
async function gracefulShutdown(signal) {
  console.log(`\n[shutdown] ${signal} received — closing server and database…`);
  let forceTimer = null;
  try {
    server.closeAllConnections?.();
    await new Promise((resolve) => {
      forceTimer = setTimeout(() => resolve(), 8000).unref?.();
      server.close(resolve);
    });
    // NOTE (F-01): background timers (trading scheduler, feeds, sweepers) may
    // still hold handles — the process.exit below clears them unconditionally,
    // after the DB close we actually need.
    db?.close?.();
    console.log("[shutdown] database closed cleanly. Bye.");
    process.exit(0);
  } catch (e) {
    console.error("[shutdown] error during graceful shutdown:", e);
    process.exit(1);
  } finally {
    clearTimeout(forceTimer); // no-op after exit; guards the live path only
    forceTimer = null;
  }
}
for (const sig of ["SIGTERM", "SIGINT"]) {
  process.on(sig, () => gracefulShutdown(sig));
}