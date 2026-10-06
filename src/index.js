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
import { TradeTaskScheduler } from "./strategy/scheduler.js";
import { PositionMonitor } from "./strategy/position_monitor.js";
import { PullbackJournal } from "./strategy/journal.js";
import { PortfolioRisk } from "./strategy/risk.js";
import { getSymbolCost, costToUsd } from "./strategy/costs.js";
import { deepSearch } from "./deep_search.js";
import * as imageProcessor from "./creative/imageProcessor.js";
import * as aiFeatures from "./creative/aiFeatures.js";
import * as promptEngine from "./creative/promptEngine.js";
import * as videoProcessor from "./creative/videoProcessor.js";
import * as templateLibrary from "./creative/templateLibrary.js";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT || 8787);
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
    headers: { Authorization: `Bearer ${SUPABASE_KEY}`, apikey: SUPABASE_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({ name: SUPABASE_BUCKET, public: false }),
    signal: AbortSignal.timeout(20_000),
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
const db = new Database(DB_PATH);
db.pragma("journal_mode = WAL");

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
const OR_PRIMARY = process.env.OR_PRIMARY_MODEL || "z-ai/glm-5.3-flash";
const OR_FAST = process.env.OR_FAST_MODEL || "openai/gpt-oss-20b";
const OR_DRAFT = process.env.OR_DRAFT_MODEL || "deepseek/deepseek-v4-flash";
const OR_IMAGE_MODEL = process.env.OR_IMAGE_MODEL || "bytedance-seed/seedream-5-0-flash";
const OR_STT_MODEL = process.env.OR_STT_MODEL || "openai/whisper-large-v3";
// Graphic-design specialists (on demand, both currently FREE via NovitaAI):
// design = text-to-image with legible text (prompt only — NO refs, NO aspect ratio, rejected otherwise);
// layer = decomposes ONE flat design image into RGBA layers (needs exactly 1 input_reference + layer plan).
const OR_DESIGN_MODEL = process.env.OR_DESIGN_MODEL || "inclusionai/ming-image-0.1-design";
const OR_LAYER_MODEL = process.env.OR_LAYER_MODEL || "inclusionai/ming-image-0.1-design-layer";
const OR_VIDEO_MODEL = process.env.OR_VIDEO_MODEL || "google/veo-3.1-lite"; // cheapest OpenRouter video-gen default (720p 4-8s)
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
const SANDBOX_URL = process.env.SANDBOX_URL || "https://sandbox-rexv.onrender.com";
// The sandbox service requires its own auth token. The AI never sees this —
// the server attaches it when forwarding run_code calls.
const SANDBOX_AUTH = process.env.SANDBOX_AUTH || process.env.SANDBOX_AUTH_TOKEN || "";
const AUTH_TOKEN = process.env.AUTH_TOKEN || "";
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
function boundUser(req, fallback = "default") {
  if (req.authUser) return req.authUser;
  return String(req.body?.user_id || req.query?.user_id || fallback);
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
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ limit: "20mb", extended: true }));
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: "*" }));
app.use(morgan(process.env.NODE_ENV === "production" ? "combined" : "dev"));

// ======================= MODELS =======================
// Cost stack (OpenRouter-only, verified Oct 2026):
// primary brain = z-ai/glm-5.3-flash ($0.15/$0.50, 1M ctx, Terminal-Bench 84.3).
// NOT z-ai/glm-5.3 ($1.40/$4.40, 9x cost). No DeepSeek, no Groq anywhere.
// fast = openai/gpt-oss-20b ($0.02/$0.10) — the gpt-oss-20b-class slot.
// draft = qwen 30B-class cheap. Free :free models sit AFTER paid in chains.
const GO_MODEL = process.env.OPENCODE_GO_MODEL || "glm-5.3-flash";
const AGENT_PRIMARY = process.env.AGENT_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : (HAS_ZEN ? `go:${GO_MODEL}` : MISTRAL_MODEL));
const MODELS = {
  vision: process.env.VISION_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : "mistral-large-latest"),
  agent: AGENT_PRIMARY,
  conversation: process.env.CONVERSATION_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : (HAS_ZEN ? `go:${GO_MODEL}` : MISTRAL_MODEL)),
  tools: process.env.TOOLS_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : HAS_ZEN ? `go:${GO_MODEL}` : MISTRAL_MODEL),
  coding: process.env.CODING_MODEL || (HAS_OR ? `openrouter:${OR_PRIMARY}` : "codestral-latest"),
  fast: process.env.FAST_MODEL || (HAS_OR ? `openrouter:${OR_FAST}` : MISTRAL_FAST_MODEL),
  voxtral: process.env.MISTRAL_VOXTRAIL_MODEL || "voxtral-mini-transcribe-realtime",
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
  if (MISTRAL_MODEL) chain.push(MISTRAL_MODEL);
  return [...new Set(chain)];
}

function buildVisionFallbackChain(primary) {
  const chain = [primary];
  if (HAS_OR && primary !== `openrouter:${OR_PRIMARY}`) chain.push(`openrouter:${OR_PRIMARY}`); // native multimodal (z-ai/glm-5.3-flash)
  if (MISTRAL_API_KEY) chain.push("mistral-large-latest");
  return [...new Set(chain)];
}

const FALLBACK_CHAINS = {
  vision: buildVisionFallbackChain(MODELS.vision),
  agent: buildFallbackChain(MODELS.agent),
  tools: buildFallbackChain(MODELS.tools),
  conversation: buildFallbackChain(MODELS.conversation),
  fast: buildFallbackChain(MODELS.fast),
};
// The :free roster rotates — resolve the live list at boot, re-check every 6h.
await refreshFreeModels();
setInterval(() => { void refreshFreeModels(); }, 6 * 3600e3).unref?.();

let PHONE_UI_SKILL = "";
try {
  PHONE_UI_SKILL = readFileSync(new URL("./skills/phone-ui.md", import.meta.url), "utf8").trim();
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
      const text = readFileSync(new URL(`./skills/flows/${file}`, import.meta.url), "utf8").trim();
      if (text) out.push(`APP FLOW (${file}):\n${text}`);
    } catch (_) { /* not written yet — skip */ }
  }
  return out.join("\n\n");
}

// Same signature as mistralChat, but walks a role's fallback chain on failure
// (e.g. Groq rate limit exhausted) instead of bubbling the error straight up.
async function chatWithFallback(role, { messages, tools = null, tool_choice = "auto", temperature, max_tokens }) {
  const chain = FALLBACK_CHAINS[role] || [MODELS[role] || MODELS.conversation];
  const attempts = [];
  for (let i = 0; i < chain.length; i++) {
    const model = chain[i];
    try {
      const out = await mistralChat({ model, messages, tools, tool_choice, temperature, max_tokens });
      if (i > 0) console.warn(`[chatWithFallback] role=${role} primary model unavailable — recovered on ${model} (normal fallback)`);
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

async function mistralChat({ model, messages, tools = null, temperature = 0.3, max_tokens = 1600, tool_choice = "auto", retries = 3 }) {
  const provider = resolveProvider(model);

  const base = provider === "openrouter" ? "https://openrouter.ai/api/v1/chat/completions"
    : provider === "go" ? "https://opencode.ai/zen/go/v1/chat/completions"
    : provider === "zen" ? "https://opencode.ai/zen/v1/chat/completions"
    : `${MISTRAL_BASE}/chat/completions`;
  const apiKey = provider === "openrouter" ? OPENROUTER_API_KEY : (provider === "zen" || provider === "go") ? ZEN_API_KEY : MISTRAL_API_KEY;
  
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
        signal: AbortSignal.timeout(90_000), // hung providers must not pin workers forever
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
    for (let tc of rawAndroidCalls.map(t => rewriteSettingsNavigation(t, session.goal))) {
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
          try {
            result = await runLocalTool(name, args, { deviceState: deviceStateForPrompt });
          } catch (err) {
            result = { ok: false, error: err.message };
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
    artifacts: turnArtifacts, // files (charts, CSVs, reports) generated by run_code this turn, full base64 included
  };
}

// ==================== AGENT ROUTES ====================
// ==================== CHAT VS TASK PRE-ROUTER ====================
// Pure chat (greetings, small talk, thanks, opinions) must NEVER enter the
// agentic loop — that's what made "hi" produce tool calls. One cheap call to
// the fast model classifies AND answers, so a greeting costs a single turn.
const CHAT_ROUTER_PROMPT = `You are FRIT's message router. Decide whether the user's message is casual conversation or a real task.

CONVERSATION means: greetings, small talk, thanks, "how are you", identity/opinion questions, jokes, or simple questions that need NO tools, NO phone actions, NO data fetching, NO code and NO search. Answer those briefly and warmly.

TASK means: the user wants something DONE — real information retrieved (weather, market prices, analysis, web research, news), code written/run, a file or webpage created, an app opened or controlled on the phone, a message/call/alarm, a trade, or any multi-step work that needs tools.

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
  const cap = checkCap(uid, "turn");
  if (!cap.ok) return res.status(402).json({ error: cap.message });

  // Fast pre-route: pure conversation never touches the agentic loop.
  try {
    const routed = await routeChatOrTask(goal, Array.isArray(history) ? history : []);
    if (!routed.isTask) {
      return res.json({ done: true, assistant_text: routed.reply });
    }
  } catch (_) {
    // Router failed — fall through to the agent loop rather than erroring out.
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
    const cap = checkCap(uid, "turn");
    if (!cap.ok) return res.status(402).json({ error: cap.message });
    const result = await runAgentStep(sessionId, tool_results, device_state);
    db.prepare("UPDATE usage_daily SET turns = turns + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
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

// Image generation via OpenRouter Images API (default: Seedream 5.0 Flash).
async function generateImage(prompt, opts = {}) {
  if (!OPENROUTER_API_KEY) throw new Error("OPENROUTER_API_KEY missing");
  const model = opts.model || OR_IMAGE_MODEL;
  const isMing = model.startsWith("inclusionai/ming");
  // Ming models REJECT sizes/aspect ratios instead of reshaping — never send them.
  const body = { model, prompt: String(prompt || "") };
  if (!isMing) {
    if (opts.aspect_ratio) body.aspect_ratio = opts.aspect_ratio;
    body.n = opts.n || 1;
  }
  if (opts.output_format) body.output_format = opts.output_format;
  if (opts.input_references?.length) body.input_references = opts.input_references;
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
// Cheap default: google/veo-3.1-lite 720p 4-8s. Alt: minimax/hailuo, alibaba/wan-2.6.
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
      generate_audio: opts.generate_audio || false,
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
const SUB_TIERS = {
  // USD-pegged: naira price computed LIVE at subscribe time (ngnPerUsd).
  // $10 Creator / $26 Pro. Caps bind worst-case cost ≈ price; overage covers the rest.
  free: { usd: 0, days: 3650, turns_day: 15, clips_mo: 0, images_mo: 10, stt_min_mo: 30 },
  creator: { usd: 10, days: 30, turns_day: 40, clips_mo: 8, images_mo: 100, stt_min_mo: 120 },
  pro: { usd: 26, days: 30, turns_day: 100, clips_mo: 25, images_mo: 400, stt_min_mo: 480 },
  owner: { usd: 0, days: 365000, turns_day: 999999, clips_mo: 999999, images_mo: 999999, stt_min_mo: 999999 },
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
const tierNaira = (tier, rate) => Math.round(SUB_TIERS[tier].usd * rate * 100);
// Per-unit wallet overage (excessive users stay profitable — billed in kobo):
const OVERAGE = { turn_kobo: 2500, clip_kobo: 40000, image_kobo: 4000, stt_min_kobo: 1000 };
const MASTER_CODE = process.env.MASTER_CODE || "";
const BTC_ADDRESS = process.env.BTC_ADDRESS || ""; // native Bitcoin (bc1/1/3…) — empty until you add one
const USDT_TRON_ADDRESS = process.env.USDT_TRON_ADDRESS || "";
// Binance deposits: BOTH assets live on BNB Smart Chain (BEP20) at ONE address
// (your BTC here is BTCB — pegged BTC, not native BTC, so mempool.space can't
// see it; BSC public RPC can, keyless).
const BSC_ADDRESS = (process.env.BSC_ADDRESS || "").toLowerCase();
const BSC_RPC = process.env.BSC_RPC || "https://bsc-dataseed.binance.org/";
const BSC_TOKENS = {
  usdt: { contract: "0x55d398a7e2f2100c2afca4aa5a4be64d0f2452b50", decimals: 18, usd: 1 },
  btcb: { contract: "0x7130d987a9e31d56f7ba9fd9a99d33047c6e216", decimals: 18, usd: null }, // priced live via Binance ticker
};
const TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df288b6e9";
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
// Cap check: returns {ok} or {ok:false, overage_kobo, message}. Overage debits wallet.
function checkCap(user_id, kind) {
  const tier = activeTier(user_id);
  const t = SUB_TIERS[tier];
  const u = getUsage(user_id);
  const m = monthUsage(user_id);
  if (kind === "turn" && u.turns >= t.turns_day) {
    if (walletBalance(user_id) < OVERAGE.turn_kobo) return { ok: false, message: `Daily turn cap reached (${tier}). Top up wallet — overage ₦${OVERAGE.turn_kobo / 100}/turn.` };
    walletSpend(user_id, OVERAGE.turn_kobo, "turn overage");
  }
  if (kind === "clip" && m.clips >= t.clips_mo) {
    if (walletBalance(user_id) < OVERAGE.clip_kobo) return { ok: false, message: `Monthly clip cap reached (${tier}). Top up — overage ₦${OVERAGE.clip_kobo / 100}/clip.` };
    walletSpend(user_id, OVERAGE.clip_kobo, "clip overage");
  }
  if (kind === "image" && m.images >= t.images_mo) {
    if (walletBalance(user_id) < OVERAGE.image_kobo) return { ok: false, message: `Monthly image cap reached (${tier}). Top up — overage ₦${OVERAGE.image_kobo / 100}/image.` };
    walletSpend(user_id, OVERAGE.image_kobo, "image overage");
  }
  if (kind === "stt" && m.stt_mins >= t.stt_min_mo) {
    if (walletBalance(user_id) < OVERAGE.stt_min_kobo) return { ok: false, message: `Monthly voice cap reached (${tier}). Top up — overage ₦${OVERAGE.stt_min_kobo / 100}/min.` };
    walletSpend(user_id, OVERAGE.stt_min_kobo, "stt overage");
  }
  return { ok: true, tier };
}
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

  // Crypto goes to Binance FIRST (free, no key, ~1200 req/min) so it never
  // touches the TwelveData quota, which forex/XAU/XAG rely on exclusively.
  if (CRYPTO_SET.has(sym) && BINANCE_SYM[sym]) {
    try {
      const url = `https://api.binance.com/api/v3/klines?symbol=${BINANCE_SYM[sym]}&interval=${toBinanceInterval(iv)}&limit=${Math.min(size, 1000)}`;
      const res = await fetch(url);
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
      const res = await fetch(url);
      const data = await res.json();
      if (data.status !== "error" && data.values?.length >= 10) {
        const candles = normalizeCandles(data.values.reverse(), "twelve_data");
        cacheSet(ck, candles, 60000);
        return candles;
      }
    } catch (e) { console.error("[TwelveData candles]", sym, e.message); }
  }

  // Fallback for crypto if Binance somehow failed and TwelveData also has no key/entry
  if (!CRYPTO_SET.has(sym) && BINANCE_SYM[sym]) {
    try {
      const url = `https://api.binance.com/api/v3/klines?symbol=${BINANCE_SYM[sym]}&interval=${toBinanceInterval(iv)}&limit=${Math.min(size, 1000)}`;
      const res = await fetch(url);
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

  // Crypto: CoinGecko first (free, no key) so crypto spot checks never touch
  // the TwelveData quota either — same reasoning as fetchCandles above.
  if (CRYPTO_SET.has(sym) && COINGECKO_IDS[sym]) {
    try {
      const res = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${COINGECKO_IDS[sym]}&vs_currencies=usd&include_24hr_change=true`);
      const data = await res.json();
      const id = COINGECKO_IDS[sym];
      if (data[id]) result = { price: data[id].usd, change24h: data[id].usd_24h_change, source: "coingecko" };
    } catch (e) { console.error("[CoinGecko spot]", sym, e.message); }
  }

  if (!result && TWELVE_DATA_KEY && TD_SYMBOLS[sym]) {
    try {
      const res = await fetch(`https://api.twelvedata.com/price?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&apikey=${TWELVE_DATA_KEY}`);
      const data = await res.json();
      if (data.price) result = { price: parseFloat(data.price), source: "twelvedata" };
    } catch (e) { console.error("[TwelveData spot]", sym, e.message); }
  }

  if (!result && FRANKFURTER_MAP[sym]) {
    try {
      const { base, quote } = FRANKFURTER_MAP[sym];
      const res = await fetch(`https://api.frankfurter.app/latest?from=${base}&to=${quote}`);
      const data = await res.json();
      const rate = data.rates?.[quote];
      if (rate) result = { price: parseFloat(rate), source: "frankfurter" };
    } catch (e) { console.error("[Frankfurter]", sym, e.message); }
  }

  if (!result && sym === "USDNGN") {
    try {
      const res = await fetch("https://open.er-api.com/v6/latest/USD");
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
    const res = await fetch(url);
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
    const res = await fetch("https://nfs.faireconomy.media/ff_calendar_thisweek.json");
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
    const geo = await (await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`)).json();
    const loc = geo.results?.[0];
    if (!loc) return "City not found";
    const wx = await (await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${loc.latitude}&longitude=${loc.longitude}&current_weather=true&timezone=auto`)).json();
    const cw = wx.current_weather;
    return `${loc.name}: ${cw.temperature}°C, wind ${cw.windspeed} km/h, ${cw.weathercode <= 1 ? "Clear" : cw.weathercode <= 3 ? "Cloudy" : "Rainy"}`;
  } catch (e) {
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
};

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
  } catch (e) { console.warn("[signals] migration failed:", e.message); }
}
ensureSignalColumns();

function storeSignalHistory(user_id, signal) {
  try {
    ensureSignalColumns();
    db.prepare(`
      INSERT INTO trading_signals
      (id, user_id, symbol, decision, confidence, entry, sl, tp1, tp2, rr, scenario, reasoning, pullback_health, range_info, costs, tp_basis, regime, entry_ctx, timestamp, is_favorite, is_read)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      signal.id, user_id, signal.symbol, signal.decision, signal.confidence,
      signal.entry, signal.sl, signal.tp1, signal.tp2, signal.rr,
      signal.scenario, signal.reasoning,
      signal.pullbackHealth ? JSON.stringify(signal.pullbackHealth) : null,
      signal.rangeInfo ? JSON.stringify(signal.rangeInfo) : null,
      signal.costs ? JSON.stringify(signal.costs) : null,
      signal.tpBasis || null, signal.regime || null,
      signal.entryCtx ? JSON.stringify(signal.entryCtx) : null,
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
    isFavorite: row.is_favorite === 1,
    isRead: row.is_read === 1,
  };
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
  isCrypto: (sym) => CRYPTO_SET.has(String(sym).toUpperCase()),
  finnhubKey: FINNHUB_KEY,
  enabled: FEED_ENABLED,
  backfillBudget: FEED_BACKFILL_BUDGET,
  log: (...a) => console.log(...a),
});
globalThis.__liveFeed = liveFeed;
if (!FINNHUB_KEY) {
  console.log("[livefeed] FINNHUB_KEY unset — forex/metals run on Dukascopy poll backup; add key for WS streaming");
}
liveFeed.start().catch(e => console.warn("[livefeed] start failed:", e.message));

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
const tradeScheduler = new TradeTaskScheduler({
  engine: mtfStrategy,
  executor: async ({ symbol, action, lotSize, entry, sl, tp, reason }) => {
    const gate = riskGate({ symbol, action, lotSize, entry, sl, balance: 1000 });
    if (!gate.allowed) {
      console.log(`[TradeTaskScheduler] RISK-BLOCKED ${symbol}: ${gate.reasons.join(" | ")}`);
      return { mode: "blocked", status: "blocked", blocked_by: "risk", reasons: gate.reasons, exposure: gate.exposure };
    }
    const result = await sendToMT5Bridge({ symbol, action, lotSize, entry, sl, tp, reason });
    if (result.status !== "failed" && sl && tp) {
      try {
        positionMonitor.register({ symbol, action, lotSize, entry, sl, tp, source: "scheduler", reason });
      } catch (e) { console.warn("[PositionMonitor] scheduler register failed:", e.message); }
    }
    return result;
  },
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
      creativeSuite: ["/creative-suite/images/generate", "/creative-suite/images/edit", "/creative-suite/images/remove-background", "/creative-suite/images/retouch", "/creative-suite/images/style-transfer", "/creative-suite/images/upscale", "/creative-suite/images/generative-fill", "/creative-suite/images/batch", "/creative-suite/videos/generate", "/creative-suite/videos/edit", "/creative-suite/videos/subtitles", "/creative-suite/videos/translate-subtitles", "/creative-suite/storyboard", "/creative-suite/templates", "/creative-suite/style-presets", "/creative-suite/prompt-suggest", "/creative-suite/prompt-enhance", "/creative-suite/recommend", "/creative-suite/projects", "/creative-suite/export"],
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
    openrouter: { primary: OR_PRIMARY, fast: OR_FAST, draft: OR_DRAFT, image: OR_IMAGE_MODEL, stt: OR_STT_MODEL, free_fast: OR_FREE_FAST },
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
app.get("/market/quote", async (req, res) => {
  try {
    const symbol = String(req.query.symbol || "BTCUSD").toUpperCase();
    const data = await fetchMarketPrices([symbol]);
    res.json(data[symbol] || { error: "Not found" });
  } catch (err) {
    res.status(500).json({ error: "Quote fetch failed", details: err.message });
  }
});

app.post("/market/batch", async (req, res) => {
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
    const symbol = String(sym).toUpperCase();
    const interval = normalizeInterval(req.method === "GET" ? query.interval : body.interval ?? query.interval ?? "1h");
    const outputsize = (req.method === "GET" ? query.outputsize : body.outputsize ?? query.outputsize)
      ? Number(req.method === "GET" ? query.outputsize : body.outputsize ?? query.outputsize)
      : null;
    res.json(await analyzeSymbol(symbol, interval, outputsize));
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

  try {
    const result = await mtfStrategy.run(symbol, { interval, balance: balance || 1000, riskPercent: risk_percent });
    console.log(`[/trade] ${symbol} ${result.decision} — reasons: ${JSON.stringify(result.reasons || [])}`);

    if (["NO_TRADE", "WAIT", "WAIT_PULLBACK", "WAIT_RANGE", "COOLDOWN", "DATA_UNAVAILABLE", "DATA_RATE_LIMITED", "ERROR"].includes(result.decision)) {
      return res.status(200).json({ status: "blocked", ...withoutEngineReasons(result) });
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

app.get("/weather", async (req, res) => {
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
    const uid = boundUser(req);
    const cap = checkCap(uid, "stt");
    if (!cap.ok) return res.status(402).json({ error: cap.message });
    const text = await mistralTranscribe(audio_base64, mime_type);
    if (!text) return res.status(500).json({ error: "Transcription failed" });
    db.prepare("UPDATE usage_daily SET stt_mins = stt_mins + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
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
    const cap = checkCap(uid, "image");
    if (!cap.ok) return res.status(402).json({ error: cap.message });
    const data = await generateImage(prompt, { aspect_ratio, n, model });
    db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
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
    if (activeTier(uid) === "free") return res.status(402).json({ error: "Video needs Creator or Pro." });
    const cap = checkCap(uid, "clip");
    if (!cap.ok) return res.status(402).json({ error: cap.message });
    const job = await submitVideo(prompt, { duration, resolution, aspect_ratio, model });
    const id = `vid_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt) VALUES (?, ?, 'video', 'submitted', ?)").run(id, uid, String(prompt).slice(0, 1000));
    db.prepare("UPDATE usage_daily SET clips = clips + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
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
      return res.status(500).json({ ok: false, status: st, error: data?.data?.error || data?.error || `video job ${st} — resubmit (failed generations are not billed)` });
    }
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
    if (activeTier(uid) === "free") return res.status(402).json({ error: "Creative mode needs Creator or Pro." });
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
    if (activeTier(uid) === "free") return res.status(402).json({ error: "Creative mode needs Creator or Pro." });
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
  const tiers = Object.fromEntries(Object.entries(SUB_TIERS).map(([k, t]) => [k, { ...t, price_kobo_live: Math.round(t.usd * rate * 100) }]));
  res.json({ ok: true, usd_ngn: rate, tiers, overage: OVERAGE });
});
app.get("/billing/pay-info", requireAuth, async (_req, res) => {
  const rate = await ngnPerUsd();
  res.json({
    ok: true, usd_ngn: rate,
    bank: { bank_name: PAY_BANK_NAME || "(set PAY_BANK_NAME)", account_number: PAY_ACCOUNT_NUMBER || "(set PAY_ACCOUNT_NUMBER)", account_name: PAY_ACCOUNT_NAME || "(set PAY_ACCOUNT_NAME)" },
    crypto: {
      bsc_address: BSC_ADDRESS || "(set BSC_ADDRESS)",
      assets: ["USDT (BNB Smart Chain)", "BTC (BNB Smart Chain)"],
      note: "USDT + BTC auto-verify on txhash submit (12+ confirmations). Send ONLY on BNB Smart Chain — other networks will lose funds.",
    },
    tiers: { creator_naira: Math.round(10 * rate), pro_naira: Math.round(26 * rate) },
    two_pots: {
      subscription: "Your payment for USING the app. Goes to our account as revenue, unlocks your tier for 30 days. Non-refundable, never spendable.",
      wallet: "YOUR money, held for the AI to spend FOR you (flights, orders, services). Fund it via /wallet/topup; the agent debits it via /wallet/spend per booking. Unspent balance stays yours.",
    },
    note: "Prices track the dollar — the app shows the exact naira amount before you pay, and that quoted amount is locked on your order.",
  });
});
app.post("/billing/subscribe", requireAuth, limitNormal, async (req, res) => {
  const { tier, method } = req.body || {};
  const user_id = boundUser(req);
  if (!SUB_TIERS[tier] || tier === "free" || tier === "owner") return res.status(400).json({ error: "paid tier required" });
  const rate = await ngnPerUsd();
  const amount_kobo = tierNaira(tier, rate); // locked at order time — FX moves don't change it after
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
      const code = makeActivationCode();
      const exp = new Date(Date.now() + SUB_TIERS[pay.tier].days * 864e5).toISOString();
      db.prepare("INSERT INTO activation_codes (code, tier, status, user_id, expires_at) VALUES (?, ?, 'unused', NULL, ?)").run(code, pay.tier, exp);
      takeSnapshot("approve-sub-crypto");
      return res.json({ ok: true, payment_id, status: "approved", verified: v, activation_code: code, note: `${net.toUpperCase()} confirmed. Enter activation code in app to unlock ${pay.tier} until ${exp.slice(0, 10)}.` });
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
  const code = makeActivationCode();
  const exp = new Date(Date.now() + SUB_TIERS[pay.tier].days * 864e5).toISOString();
  db.prepare("INSERT INTO activation_codes (code, tier, status, user_id, expires_at) VALUES (?, ?, 'unused', NULL, ?)").run(code, pay.tier, exp);
  takeSnapshot("approve-sub");
  res.json({ ok: true, purpose: "sub", user_id: pay.user_id, tier: pay.tier, activation_code: code, expires_at: exp });
});
app.post("/billing/mint", requireAdmin, (req, res) => {
  // Admin: mint free-trial passcodes (e.g. 3-day Creator) for testers.
  // They redeem in-app via /billing/redeem; each code is single-use with expiry.
  const { tier, days, count } = req.body || {};
  if (!SUB_TIERS[tier] || tier === "owner") return res.status(400).json({ error: "valid tier required" });
  const n = Math.min(Math.max(Number(count) || 1, 1), 100);
  const d = Math.min(Math.max(Number(days) || 3, 1), 30);
  const codes = [];
  for (let i = 0; i < n; i++) {
    const code = makeActivationCode();
    const exp = new Date(Date.now() + d * 864e5).toISOString();
    db.prepare("INSERT INTO activation_codes (code, tier, status, user_id, expires_at) VALUES (?, ?, 'unused', NULL, ?)").run(code, `${tier}`, exp);
    codes.push({ code, tier, expires_at: exp });
  }
  res.json({ ok: true, codes });
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
  if (!row || row.status !== "unused") return res.status(400).json({ error: "invalid or used code" });
  if (row.expires_at && new Date(row.expires_at).getTime() <= Date.now()) {
    db.prepare("UPDATE activation_codes SET status = 'expired' WHERE code = ?").run(c);
    return res.status(400).json({ error: "code expired — contact support for a fresh one" });
  }
  db.prepare("UPDATE activation_codes SET status = 'used', user_id = ? WHERE code = ?").run(user_id, c);
  db.prepare("INSERT INTO subscriptions (user_id, tier, status, expires_at) VALUES (?, ?, 'active', ?) ON CONFLICT(user_id) DO UPDATE SET tier=?, status='active', expires_at=?, updated_at=CURRENT_TIMESTAMP").run(user_id, row.tier, row.expires_at, row.tier, row.expires_at);
  res.json({ ok: true, tier: row.tier, expires_at: row.expires_at });
});
app.get("/billing/status", requireAuth, limitNormal, (req, res) => {
  const user_id = boundUser(req);
  const sub = db.prepare("SELECT tier, expires_at FROM subscriptions WHERE user_id = ?").get(user_id);
  res.json({ ok: true, user_id, tier: activeTier(user_id), expires_at: sub?.expires_at || null, usage_today: getUsage(user_id), usage_month: monthUsage(user_id) });
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
    } catch {}
    try { (await import("fs")).unlinkSync(tmp); } catch {}
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
    const { prompt, style_preset, aspect_ratio, n, model } = req.body || {};
    if (!prompt) return res.status(400).json({ error: "prompt required" });
    const uid = boundUser(req);
    const cap = checkCap(uid, "image");
    if (!cap.ok) return res.status(402).json({ error: cap.message });
    const preset = promptEngine.STYLE_PRESETS.find(p => p.id === style_preset);
    const enhancedPrompt = preset ? `${prompt}, ${preset.description}` : prompt;
    const data = await generateImage(enhancedPrompt, { aspect_ratio, n, model });
    db.prepare("UPDATE usage_daily SET images = images + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    const assetId = `img_${Date.now()}`;
    db.prepare("INSERT INTO creative_assets (id, user_id, kind, source, current_url, metadata) VALUES (?, ?, 'image', 'generated', ?, ?)")
      .run(assetId, uid, data.image_url || "", JSON.stringify({ prompt, style_preset, model: data.model }));
    res.json({ ok: true, asset_id: assetId, ...data });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/images/edit", requireAuth, limitNormal, async (req, res) => {
  try {
    const { image, operations } = req.body || {};
    if (!image || !Array.isArray(operations)) return res.status(400).json({ error: "image and operations required" });
    const imageBuffer = Buffer.from(image.replace(/^data:image\/\w+;base64,/, ""), "base64");
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
    const imageBuffer = Buffer.from(image.replace(/^data:image\/\w+;base64,/, ""), "base64");
    const result = await aiFeatures.removeBackground(imageBuffer);
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/images/retouch", requireAuth, limitCostly, async (req, res) => {
  try {
    const { image, options } = req.body || {};
    if (!image) return res.status(400).json({ error: "image required" });
    const imageBuffer = Buffer.from(image.replace(/^data:image\/\w+;base64,/, ""), "base64");
    const result = await aiFeatures.retouch(imageBuffer, options || {});
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/images/style-transfer", requireAuth, limitCostly, async (req, res) => {
  try {
    const { image, style } = req.body || {};
    if (!image || !style) return res.status(400).json({ error: "image and style required" });
    const imageBuffer = Buffer.from(image.replace(/^data:image\/\w+;base64,/, ""), "base64");
    const result = await aiFeatures.styleTransfer(imageBuffer, style);
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/images/upscale", requireAuth, limitCostly, async (req, res) => {
  try {
    const { image, scale } = req.body || {};
    if (!image) return res.status(400).json({ error: "image required" });
    const imageBuffer = Buffer.from(image.replace(/^data:image\/\w+;base64,/, ""), "base64");
    const result = await aiFeatures.upscaleImage(imageBuffer, scale || 2);
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/images/generative-fill", requireAuth, limitCostly, async (req, res) => {
  try {
    const { image, mask, prompt } = req.body || {};
    if (!image || !mask) return res.status(400).json({ error: "image and mask required" });
    const imageBuffer = Buffer.from(image.replace(/^data:image\/\w+;base64,/, ""), "base64");
    const maskBuffer = Buffer.from(mask.replace(/^data:image\/\w+;base64,/, ""), "base64");
    const result = await aiFeatures.generativeFill(imageBuffer, maskBuffer, prompt);
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
    const jobId = `batch_${Date.now()}`;
    db.prepare("INSERT INTO creative_batch_jobs (id, user_id, operation, asset_ids, params, status, total_count) VALUES (?, ?, ?, ?, ?, 'processing', ?)")
      .run(jobId, uid, operation, JSON.stringify(images), JSON.stringify(params || {}), images.length);
    const results = [];
    for (let i = 0; i < images.length; i++) {
      try {
        const imgBuffer = Buffer.from(images[i].replace(/^data:image\/\w+;base64,/, ""), "base64");
        let resultBuffer;
        if (operation === "remove_bg") resultBuffer = await aiFeatures.removeBackground(imgBuffer);
        else if (operation === "upscale") resultBuffer = await aiFeatures.upscaleImage(imgBuffer, params?.scale || 2);
        else if (operation === "style_transfer") resultBuffer = await aiFeatures.styleTransfer(imgBuffer, params?.style || "realistic");
        else resultBuffer = await imageProcessor.applyOperations(imgBuffer, [{ type: operation, params: params || {} }]);
        results.push({ index: i, ok: true, result: resultBuffer.image_url || resultBuffer.result || `data:image/png;base64,${resultBuffer.toString("base64")}` });
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

app.post("/creative-suite/videos/generate", requireAuth, limitCostly, async (req, res) => {
  try {
    const { prompt, duration, resolution, aspect_ratio, model } = req.body || {};
    if (!prompt) return res.status(400).json({ error: "prompt required" });
    const uid = boundUser(req);
    if (activeTier(uid) === "free") return res.status(402).json({ error: "Video needs Creator or Pro." });
    const cap = checkCap(uid, "clip");
    if (!cap.ok) return res.status(402).json({ error: cap.message });
    const job = await submitVideo(prompt, { duration, resolution, aspect_ratio, model });
    const id = `vid_${Date.now()}`;
    db.prepare("INSERT INTO creative_jobs (id, user_id, kind, status, prompt) VALUES (?, ?, 'video', 'submitted', ?)").run(id, uid, String(prompt).slice(0, 1000));
    db.prepare("UPDATE usage_daily SET clips = clips + 1 WHERE user_id = ? AND day = ?").run(uid, todayStr());
    res.json({ ok: true, job_id: id, provider_job: job });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/videos/edit", requireAuth, limitCostly, async (req, res) => {
  try {
    const { video, operations } = req.body || {};
    if (!video) return res.status(400).json({ error: "video required" });
    const results = await videoProcessor.processVideo(Buffer.from(video, "base64"), operations || []);
    res.json({ ok: true, results });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/videos/subtitles", requireAuth, limitCostly, async (req, res) => {
  try {
    const { audio, language } = req.body || {};
    if (!audio) return res.status(400).json({ error: "audio required" });
    const audioBuffer = Buffer.from(audio.replace(/^data:audio\/\w+;base64,/, ""), "base64");
    const result = await aiFeatures.generateSubtitles(audioBuffer, language || "auto");
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/creative-suite/videos/translate-subtitles", requireAuth, limitNormal, async (req, res) => {
  try {
    const { subtitles, target_language } = req.body || {};
    if (!subtitles || !target_language) return res.status(400).json({ error: "subtitles and target_language required" });
    const result = await aiFeatures.translateSubtitles(subtitles, target_language);
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
    if (activeTier(uid) === "free") return res.status(402).json({ error: "Storyboard needs Creator or Pro." });
    const frames = await aiFeatures.storyboard(script, style, num_frames || 6);
    const generatedFrames = [];
    for (const frame of frames.slice(0, 8)) {
      try {
        const img = await generateImage(frame.prompt, { model: OR_IMAGE_MODEL });
        generatedFrames.push({ ...frame, image_url: img.image_url });
      } catch (e) {
        generatedFrames.push({ ...frame, image_url: null, error: e.message });
      }
    }
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

app.post("/creative-suite/templates/:id/apply", requireAuth, limitCostly, async (req, res) => {
  try {
    const { assets } = req.body || {};
    const result = templateLibrary.applyTemplate(req.params.id, assets || []);
    res.json({ ok: true, ...result });
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
  const projects = db.prepare("SELECT id, name, type, status, thumbnail_url, metadata, updated_at FROM creative_projects WHERE user_id = ? ORDER BY updated_at DESC LIMIT 100").all(uid);
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
  db.prepare("DELETE FROM creative_assets WHERE project_id = ?").run(req.params.id);
  db.prepare("DELETE FROM creative_projects WHERE id = ?").run(req.params.id);
  res.json({ ok: true });
});

app.post("/creative-suite/projects/:id/assets", requireAuth, limitNormal, (req, res) => {
  try {
    const { kind, source, url, metadata } = req.body || {};
    if (!kind || !url) return res.status(400).json({ error: "kind and url required" });
    const uid = boundUser(req);
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
    res.json({ ok: true, platform, preset, asset_url, note: "Export processing delegated to client-side rendering with platform preset" });
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
  try {
    const result = await mtfStrategy.analyze(symbol, { interval, balance, riskPercent });
    res.json(result);
  } catch (err) {
    console.error("[/enhanced/analyze]", err.message);
    res.status(500).json({ error: "Enhanced analysis failed", details: err.message });
  }
});

// Full multi-timeframe directional report — "build with directions".
app.get("/market/strategy", requireAuth, async (req, res) => {
  try {
    const sym = String(req.query.symbol || "XAUUSD").toUpperCase();
    const result = await mtfStrategy.analyze(sym, {});
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

// ==================== TRADING SUITE API ====================
// Comprehensive AI-powered trading endpoints for mobile app

// --- Watchlist Management ---
app.get("/api/trading/watchlist", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const watchlist = db.prepare("SELECT * FROM trading_watchlist WHERE user_id = ? ORDER BY sort_order ASC").all(user_id);
    res.json({ items: watchlist });
  } catch (err) {
    res.status(500).json({ error: "Watchlist fetch failed", details: err.message });
  }
});

app.post("/api/trading/watchlist", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const { symbol, name, assetClass } = req.body || {};
    if (!symbol) return res.status(400).json({ error: "symbol required" });
    const sym = String(symbol).toUpperCase();
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
    if (result.changes === 0) return res.status(404).json({ error: "Not in watchlist" });
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: "Watchlist remove failed", details: err.message });
  }
});

app.patch("/api/trading/watchlist/:symbol", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
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

    const results = [];
    for (const sym of scanSymbols.slice(0, 20)) {
      try {
        const result = await mtfStrategy.analyze(sym, {});
        const signal = formatTradingSignal(sym, result);
        // Only persist meaningful signals — plain WAIT/DATA noise would flood
        // history. WAIT_PULLBACK / WAIT_RANGE carry actionable zones.
        const keep = ["BUY", "SELL", "WAIT_PULLBACK", "WAIT_RANGE"].includes(signal.decision);
        if (keep) storeSignalHistory(user_id, signal);
        results.push(signal);
      } catch (e) {
        console.warn(`[scan] ${sym} failed:`, e.message);
      }
    }
    const alerts = results.filter(s => ["BUY", "SELL"].includes(s.decision));
    res.json({ signals: results, alerts, scannedAt: Date.now() });
  } catch (err) {
    res.status(500).json({ error: "Scan failed", details: err.message });
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

// --- Portfolio ---
app.get("/api/trading/portfolio", requireAuth, async (req, res) => {
  try {
    const user_id = boundUser(req);
    const balance = Number(req.query.balance) || 1000;
    
    // Get open positions from PositionMonitor
    const openPositions = positionMonitor.list(100).filter(p => p.status === "MONITORING");
    
    // Get resolved positions for P&L
    const resolvedPositions = positionMonitor.list(500).filter(p => p.status === "RESOLVED");
    
    // Calculate metrics
    let unrealizedPnl = 0, realizedPnl = 0;
    const positions = [];
    
    for (const pos of openPositions) {
      const spot = await fetchSpotPrice(pos.symbol);
      if (spot) {
        const isLong = pos.action === "buy";
        const pnl = isLong 
          ? (spot.price - pos.entry) * pos.lot_size * (PIP_CONFIG[pos.symbol]?.pipValue || 10)
          : (pos.entry - spot.price) * pos.lot_size * (PIP_CONFIG[pos.symbol]?.pipValue || 10);
        unrealizedPnl += pnl;
        positions.push({
          id: pos.id, symbol: pos.symbol, side: pos.action.toUpperCase(),
          entryPrice: pos.entry, currentPrice: spot.price, lotSize: pos.lot_size,
          sl: pos.sl, tp: pos.tp, unrealizedPnl: pnl, realizedPnl: 0,
          status: "OPEN", openedAt: new Date(pos.placed_at).getTime(), closedAt: null
        });
      }
    }
    
    for (const pos of resolvedPositions.slice(0, 50)) {
      const cfg = PIP_CONFIG[pos.symbol];
      const isLong = pos.action === "buy";
      const pnl = cfg ? (isLong 
        ? ((pos.close_price - pos.entry) / cfg.pipSize) * cfg.pipValue * pos.lot_size
        : ((pos.entry - pos.close_price) / cfg.pipSize) * cfg.pipValue * pos.lot_size
      ) : 0;
      realizedPnl += pnl;
      positions.push({
        id: pos.id, symbol: pos.symbol, side: pos.action.toUpperCase(),
        entryPrice: pos.entry, currentPrice: pos.close_price, lotSize: pos.lot_size,
        sl: pos.sl, tp: pos.tp, unrealizedPnl: 0, realizedPnl: pnl,
        status: pos.outcome === "win" ? "CLOSED_WIN" : "CLOSED_LOSS",
        openedAt: new Date(pos.placed_at).getTime(), closedAt: new Date(pos.resolved_at).getTime()
      });
    }
    
    // Risk metrics from PortfolioRisk
    const riskStatus = portfolioRisk.status(balance);
    
    res.json({
      positions,
      summary: {
        totalPositions: openPositions.length,
        unrealizedPnl: Number(unrealizedPnl.toFixed(2)),
        realizedPnl: Number(realizedPnl.toFixed(2)),
        totalPnl: Number((unrealizedPnl + realizedPnl).toFixed(2)),
        balance,
        equity: balance + unrealizedPnl + realizedPnl
      },
      risk: riskStatus
    });
  } catch (err) {
    res.status(500).json({ error: "Portfolio fetch failed", details: err.message });
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
app.listen(PORT, () => {
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