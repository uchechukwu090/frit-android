import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import dotenv from "dotenv";
import fetch from "node-fetch";
import FormData from "form-data";
import Database from "better-sqlite3";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { MTFStrategyEngine } from "./strategy/engine.js";
import { TradeTaskScheduler } from "./strategy/scheduler.js";
import { PositionMonitor } from "./strategy/position_monitor.js";
import { deepSearch } from "./deep_search.js";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT || 8787);
const __dirname = dirname(fileURLToPath(import.meta.url));

// ========================= PERSISTENCE (SQLite) =====================
const db = new Database(join(__dirname, "../frit.db"));
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

// ========================= ENVIRONMENT ==============================
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const HAS_GROQ = !!GROQ_API_KEY;
const ZEN_API_KEY = process.env.OPENCODE_ZEN_API_KEY || "";
const HAS_ZEN = !!ZEN_API_KEY;

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY || "";
const HAS_DEEPSEEK = !!DEEPSEEK_API_KEY;
const ZEN_MODEL = process.env.OPENCODE_ZEN_MODEL || "deepseek-v4-flash-free";
const MISTRAL_MODEL = process.env.MISTRAL_MODEL || "mistral-large-latest";
const MISTRAL_FAST_MODEL = process.env.MISTRAL_FAST_MODEL || "mistral-small-latest";
const TWELVE_DATA_KEY = process.env.TWELVE_DATA_KEY || "";
const SANDBOX_URL = process.env.SANDBOX_URL || "https://sandbox-rexv.onrender.com";
// The sandbox service requires its own auth token. The AI never sees this —
// the server attaches it when forwarding run_code calls.
const SANDBOX_AUTH = process.env.SANDBOX_AUTH || process.env.SANDBOX_AUTH_TOKEN || "";
const AUTH_TOKEN = process.env.AUTH_TOKEN || "";
const MT5_BRIDGE_URL = process.env.MT5_BRIDGE_URL || "";

// ========================== VALIDATION =============================
if (!AUTH_TOKEN) {
  console.error("[FATAL] AUTH_TOKEN missing — mandatory for agentic security!");
  process.exit(1);
}
if (!HAS_GROQ && !GEMINI_API_KEY && !MISTRAL_API_KEY) {
  console.error("[FATAL] All provider keys missing — set at least one in .env");
  process.exit(1);
}

// ========================== MIDDLEWARE ==============================
function requireAuth(req, res, next) {
  const header = req.headers["authorization"] || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : "";
  if (token !== AUTH_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  next();
}

app.set("trust proxy", 1);
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ limit: "20mb", extended: true }));
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: "*" }));
app.use(morgan(process.env.NODE_ENV === "production" ? "combined" : "dev"));

// ======================= MODELS =======================
const GO_MODEL = process.env.OPENCODE_GO_MODEL || "deepseek-v4-flash"; // -> opencode-go/deepseek-v4-flash
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL || "deepseek-flash"; // DeepSeek-V4.1-Flash, native tool calling (legacy deepseek-chat alias still routes here but is deprecated)
// Cost stack (free-first, verified Sep 30 2026): Groq free tier needs no card
// (~30 RPM / ~1K req/day / 100-500K tok/day — fine for personal use, tight for
// heavy scheduling). Kimi K2 -0905 (versioned ID) leads agent/tools: best free
// agentic tool-caller. Llama 3.3 70B next: most reliable free tools. Funded
// DeepSeek stays in-chain AFTER the free links, so a topped-up key gets used
// via fallthrough without blocking free traffic. Qwen 3.8 (reasoning-burn
// cost) and GPT-OSS 120B (wrong tool calls) are out of every chain.
const AGENT_PRIMARY = process.env.AGENT_MODEL || (HAS_GROQ ? "groq:moonshotai/kimi-k2-instruct-0905" : (HAS_DEEPSEEK ? `deepseek:${DEEPSEEK_MODEL}` : (HAS_ZEN ? `go:${GO_MODEL}` : (GEMINI_API_KEY ? "gemini-3.6-flash" : MISTRAL_MODEL))));
const MODELS = {
  vision: process.env.VISION_MODEL || (GEMINI_API_KEY ? "gemini-3.6-flash" : "mistral-large-latest"),
  agent: AGENT_PRIMARY,
  conversation: process.env.CONVERSATION_MODEL || (HAS_GROQ ? "groq:openai/gpt-oss-20b" : (HAS_ZEN ? `go:${GO_MODEL}` : (GEMINI_API_KEY ? "gemini-3.6-flash" : MISTRAL_MODEL))),
  tools: process.env.TOOLS_MODEL || (HAS_GROQ ? "groq:moonshotai/kimi-k2-instruct-0905" : (HAS_DEEPSEEK ? `deepseek:${DEEPSEEK_MODEL}` : HAS_ZEN ? `go:${GO_MODEL}` : (GEMINI_API_KEY ? "gemini-3.6-flash" : MISTRAL_MODEL))),
  coding: process.env.CODING_MODEL || (HAS_GROQ ? "groq:moonshotai/kimi-k2-instruct-0905" : (HAS_DEEPSEEK ? `deepseek:${DEEPSEEK_MODEL}` : "codestral-latest")),
  fast: process.env.FAST_MODEL || (HAS_GROQ ? "groq:openai/gpt-oss-20b" : (GEMINI_API_KEY ? "gemini-3.6-flash" : MISTRAL_FAST_MODEL)),
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
  const chain = [primary];
  if (HAS_GROQ) {
    // Free links first (unknown IDs 404 and fall through harmlessly).
    chain.push("groq:moonshotai/kimi-k2-instruct-0905");
    chain.push("groq:llama-3.3-70b-versatile");
  }
  if (HAS_DEEPSEEK) chain.push(`deepseek:${DEEPSEEK_MODEL}`); // skipped fast on 402/401 when unfunded
  if (HAS_ZEN) chain.push(`go:${GO_MODEL}`); // harmless to keep listed; simply 401s and falls through if Go is still expired
  if (GEMINI_API_KEY) chain.push("gemini-3.6-flash");
  if (MISTRAL_MODEL) chain.push(MISTRAL_MODEL);
  return [...new Set(chain)];
}

function buildVisionFallbackChain(primary) {
  const chain = [primary];
  if (GEMINI_API_KEY) chain.push("gemini-3.6-flash");
  if (HAS_GROQ) {
    // Llama 4 Scout: multimodal, fast and free on Groq. (Kimi K2 on Groq is
    // text-only, so it can't cover vision.)
    chain.push("groq:meta-llama/llama-4-scout-17b-16e-instruct");
  }
  
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

async function geminiChat({ model, messages, tools = null, temperature = 0.7, max_tokens = 2048 }) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_API_KEY}`;

  const contents = messages.map(m => {
    let parts = [];
    if (typeof m.content === "string") {
      parts = [{ text: m.content }];
    } else if (Array.isArray(m.content)) {
      parts = m.content.map(p => {
        if (p.type === "text") return { text: p.text };
        if (p.type === "image_url") {
          const b64 = p.image_url.url.split(",")[1] || p.image_url.url;
          return { inline_data: { mime_type: "image/jpeg", data: b64 } };
        }
        return null;
      }).filter(Boolean);
    }

    if (m.tool_calls) {
      m.tool_calls.forEach(tc => {
        parts.push({
          functionCall: {
            name: tc.function.name,
            args: typeof tc.function.arguments === "string" ? JSON.parse(tc.function.arguments) : tc.function.arguments
          }
        });
      });
    }

    if (m.role === "tool") {
      return {
        role: "function",
        parts: [{
          functionResponse: {
            name: m.name || m.tool_call_id, // Gemini expects the function name here often
            response: { content: m.content }
          }
        }]
      };
    }

    return { role: m.role === "assistant" ? "model" : "user", parts };
  });

  const body = {
    contents,
    generationConfig: { maxOutputTokens: max_tokens, temperature }
  };

  if (tools?.length) {
    body.tools = [{
      function_declarations: tools.map(t => ({
        name: t.function.name,
        description: t.function.description,
        parameters: t.function.parameters
      }))
    }];
  }

  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error?.message || JSON.stringify(data));

  const candidate = data.candidates?.[0];
  const msgContent = candidate?.content;
  const parts = msgContent?.parts || [];

  let text = "";
  const tool_calls = [];

  parts.forEach(p => {
    if (p.text) text += p.text;
    if (p.functionCall) {
      tool_calls.push({
        id: `call_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
        type: "function",
        function: {
          name: p.functionCall.name,
          arguments: JSON.stringify(p.functionCall.args)
        }
      });
    }
  });

  return { choices: [{ message: { content: text, tool_calls: tool_calls.length ? tool_calls : undefined } }] };
}

// Resolve which provider a model string belongs to.
// "groq:..."  -> Groq (OpenAI-compatible /chat/completions, supports tool calls)
// "zen:..."   -> OpenCode Zen pay-as-you-go (billed against your Zen wallet balance)
// "go:..."    -> OpenCode Go subscription (billed against your $10/mo Go allowance,
//                 a SEPARATE endpoint + model-id format from plain Zen — using the
//                 wrong one silently drains your $0 Zen wallet instead of Go credits)
// "gemini-*"  -> Gemini native generateContent (vision; tools unsupported here)
// otherwise    -> Mistral (OpenAI-compatible chat completions, supports tool calls)
function resolveProvider(model) {
  if (typeof model === "string" && model.startsWith("groq:")) return "groq";
  if (typeof model === "string" && model.startsWith("go:")) return "go";
  if (typeof model === "string" && model.startsWith("zen:")) return "zen";
  if (typeof model === "string" && model.startsWith("deepseek:")) return "deepseek";
  if (typeof model === "string" && model.startsWith("gemini")) return "gemini";
  return "mistral";
}

async function mistralChat({ model, messages, tools = null, temperature = 0.3, max_tokens = 1600, tool_choice = "auto", retries = 3 }) {
  const provider = resolveProvider(model);
  if (provider === "gemini") return geminiChat({ model, messages, tools, temperature, max_tokens });

  const base = provider === "groq" ? "https://api.groq.com/openai/v1/chat/completions"
    : provider === "go" ? "https://opencode.ai/zen/go/v1/chat/completions"
    : provider === "zen" ? "https://opencode.ai/zen/v1/chat/completions"
    : provider === "deepseek" ? "https://api.deepseek.com/chat/completions"
    : `${MISTRAL_BASE}/chat/completions`;
  const apiKey = provider === "groq" ? GROQ_API_KEY : (provider === "zen" || provider === "go") ? ZEN_API_KEY : provider === "deepseek" ? DEEPSEEK_API_KEY : MISTRAL_API_KEY;
  
  const cleanModel = provider === "go" ? model.slice(3)
    : (provider === "groq" || provider === "zen" || provider === "deepseek") ? model.slice(model.indexOf(":") + 1)
    : model;

  const body = { model: cleanModel, messages, temperature, max_tokens };
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
         
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        lastError = new Error(data?.message || data?.error?.message || JSON.stringify(data));
        const retryable = res.status === 429 || (res.status === 400 && /tool_use_failed|Failed to call a function/i.test(lastError.message));
        if (retryable && attempt < retries) {
          // Groq/Mistral 429 bodies say "Please try again in 11.495s" — honor
          // that instead of a fixed backoff that may be shorter than needed.
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
const MAX_SERVER_TOOL_LOOP = 8;

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
  const messages = steps.map(s => ({
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
  // Route through the fallback chain for whichever role this resolved to, so a
  // Groq rate limit (e.g. openai/gpt-oss-120b) doesn't kill the whole turn.
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

// ==================== DETERMINISTIC MARKET PRE-ROUTE ====================
// Pure quote/analysis goals are answered straight from the trading engine
// (Twelve Data spot + MTF strategy) instead of entering the LLM loop, where
// the brain sometimes narrates analysis from memory instead of calling
// analyze_market/get_market_data. Trade EXECUTION ("buy X 0.01") is NOT
// short-circuited — orders stay inside the agent loop with its verification
// and confirmation discipline.
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
  return "BTCUSD";
}
function hasMarketContext(t) {
  if (/\bmt5\b|forex|crypto|\bmarket\b|\blot\b|\btrade\b|\btrading\b/.test(t)) return true;
  return Object.keys(MARKET_SYMBOL_MAP).some(k => marketKeyHit(t, k));
}
// Returns { kind: "quote"|"analysis", symbol } or null. Trade orders return
// null on purpose so they stay in the agent loop (confirmation discipline).
function detectMarketGoal(rawGoal) {
  const t = String(rawGoal || "").toLowerCase();
  if (!hasMarketContext(t)) return null;
  const side = /\bbuy\b|\bsell\b|\blong\b|\bshort\b/.test(t);
  const num = /\d+(\.\d+)?/.test(t);
  if (side && num) return null; // real order -> agent loop
  if (/price|quote|\brate\b|how much|cost of/.test(t)) return { kind: "quote", symbol: extractMarketSymbol(t) };
  if (/analy|signal|forecast|predict|chart|outlook|\btrade\b|\btrading\b|\bmt5\b|\bbuy\b|\bsell\b|\blong\b|\bshort\b/.test(t)) {
    return { kind: "analysis", symbol: extractMarketSymbol(t) };
  }
  return null;
}
function formatQuote(symbol, prices) {
  const q = prices && prices[symbol];
  if (!q || q.error || !q.price) return `${symbol}: live price unavailable right now.`;
  const ch = Number(q.change24h || 0);
  return `${symbol}: ${q.price} ${q.currency || "USD"} (24h ${ch >= 0 ? "+" : ""}${ch.toFixed(2)}%) — live engine quote.`;
}
function formatEngineAnalysis(symbol, a) {
  if (!a || typeof a !== "object") return `${symbol}: engine returned no data.`;
  if (a.decision === "DATA_UNAVAILABLE") return `${symbol}: live data unavailable (${a.reason || "no candles"}).`;
  const L = [`${a.symbol || symbol} live engine: ${a.decision}${a.direction ? ` (${a.direction})` : ""}${a.confidence != null ? ` — confidence ${a.confidence}%` : ""}`];
  if (a.price != null) L.push(`Price: ${a.price}`);
  if (a.entry != null) L.push(`Entry: ${a.entry}`);
  if (a.sl != null) L.push(`SL: ${a.sl}`);
  if (a.tp != null) L.push(`TP: ${a.tp}`);
  if (Array.isArray(a.reasons) && a.reasons.length) L.push(`Reasons: ${a.reasons.slice(0, 4).join("; ")}`);
  if (a.entry_ctx && a.entry_ctx.zone != null) L.push(`Entry zone (${a.entry_ctx.type || "limit"}): ${a.entry_ctx.zone}`);
  return L.join("\n");
}

app.post("/agent/start", requireAuth, async (req, res) => {
  const { goal, device_state, memory, history } = req.body;
  if (!goal) return res.status(400).json({ error: "Goal required" });

  // Fast pre-route: pure conversation never touches the agentic loop.
  try {
    const routed = await routeChatOrTask(goal, Array.isArray(history) ? history : []);
    if (!routed.isTask) {
      return res.json({ done: true, assistant_text: routed.reply });
    }
  } catch (_) {
    // Router failed — fall through to the agent loop rather than erroring out.
  }

  // Deterministic market pre-route: engine data first, LLM never invents prices.
  try {
    const mg = detectMarketGoal(goal);
    if (mg) {
      if (mg.kind === "quote") {
        const prices = await fetchMarketPrices([mg.symbol]);
        return res.json({ done: true, assistant_text: formatQuote(mg.symbol, prices) });
      }
      const analysis = await mtfStrategy.analyze(mg.symbol, {});
      return res.json({ done: true, assistant_text: formatEngineAnalysis(mg.symbol, analysis) });
    }
  } catch (e) {
    console.warn("[agent/start] market pre-route failed, using agent loop:", e.message);
  }

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
    db.prepare("INSERT INTO agent_sessions (id, goal, task_ledger, last_device_state, memory) VALUES (?, ?, ?, ?, ?)").run(
      sessionId, effectiveGoal, JSON.stringify(ledger), JSON.stringify(device_state || {}), JSON.stringify(safeMemory)
    );

    const result = await runAgentStep(sessionId, null, device_state, { memory: safeMemory });
    res.json(result);
  } catch (err) {
    console.error("[agent/start]", err);
    res.status(err.code === "ALL_MODELS_UNAVAILABLE" ? 503 : 500).json({ error: err.message, code: err.code || "UNKNOWN" });
  }
});

app.post("/agent/resume", requireAuth, async (req, res) => {
  const { sessionId, tool_results, device_state } = req.body;
  if (!sessionId || !tool_results) return res.status(400).json({ error: "sessionId and tool_results required" });

  try {
    const result = await runAgentStep(sessionId, tool_results, device_state);
    res.json(result);
  } catch (err) {
    res.status(err.code === "ALL_MODELS_UNAVAILABLE" ? 503 : 500).json({ error: err.message, code: err.code || "UNKNOWN" });
  }
});

app.get("/agent/status", requireAuth, (req, res) => {
  const { sessionId } = req.query;
  const session = db.prepare("SELECT * FROM agent_sessions WHERE id = ?").get(sessionId);
  if (!session) return res.status(404).json({ error: "Not found" });

  const steps = db.prepare("SELECT * FROM agent_steps WHERE session_id = ? ORDER BY step_n ASC").all(sessionId);
  res.json({ session, steps });
});


// Transcription helper (Mistral/Voxtral primary block removed — Groq Whisper
// is the live path). Decodes the base64 body once for the fallbacks below.
async function mistralTranscribe(audio_base64, mime_type = "audio/webm") {
  const audioBuffer = Buffer.from(String(audio_base64 || ""), "base64");
  const cleanMime = String(mime_type || "audio/webm").split(";")[0].trim() || "audio/webm";

  // Fallback: Groq Whisper
  if (process.env.GROQ_API_KEY) {
    try {
      const form = new FormData();
      const ext = cleanMime.split("/")[1] || "wav";
      form.append("file", audioBuffer, { filename: `audio.${ext}`, contentType: cleanMime });
      form.append("model", "whisper-large-v3"); // flagship accuracy — best for accents/noise (Groq's own guidance for error-sensitive use)
      form.append("language", "en"); // user's English incl. Nigerian accent: improves accuracy + latency
      form.append("prompt", "Hey Frit, FRIT, OPay, MT5, MetaTrader, XAUUSD, mummy, Airtel."); // guides proper-noun spelling ("frit", not "fritz")
      form.append("temperature", "0.0");
      form.append("response_format", "json");
      const res = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
        method: "POST",
        headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}`, ...form.getHeaders() },
        body: form,
      });
      if (res.ok) {
        const d = await res.json();
        if (d.text) return d.text;
      }
      console.warn(`[MistralTranscribe] Groq failed: ${res.status}`);
    } catch (e) { console.warn("[MistralTranscribe] Groq error:", e.message); }
  }

  // Fallback: OpenAI Whisper
  if (process.env.OPENAI_API_KEY) {
    try {
      const form = new FormData();
      const ext = cleanMime.split("/")[1] || "wav";
      form.append("file", audioBuffer, { filename: `audio.${ext}`, contentType: cleanMime });
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
      console.warn(`[MistralTranscribe] OpenAI failed: ${res.status}`);
    } catch (e) { console.warn("[MistralTranscribe] OpenAI error:", e.message); }
  }

  return "";
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
    "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
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
const mtfStrategy = new MTFStrategyEngine({
  fetchCandles,
  checkNewsFilter,
  calculateLotSize,
  addTradeMemory,
});

// Recurring "everyday analyze XAUUSD" tasks — paper-first by design.
const tradeScheduler = new TradeTaskScheduler({
  engine: mtfStrategy,
  executor: async ({ symbol, action, lotSize, entry, sl, tp, reason }) => {
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
  "analyze_market", "run_code", "wait_and_verify", "assert_text_visible", "get_frit_manual"
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
  { type: "function", function: { name: "get_market_news", description: "Fetch real-time financial, fundamental, and macroeconomic news for any trading symbol (XAUUSD, BTCUSD, Forex, Stocks) for the current date/year.", parameters: { type: "object", properties: { symbol: { type: "string" }, query: { type: "string" } } } } },
  { type: "function", function: { name: "place_mt5_trade", description: "Place a real market order on MetaTrader 5 via the phone's MT5 agent. Use this after market analysis confirms a high-confidence entry signal.", parameters: { type: "object", properties: { symbol: { type: "string" }, action: { type: "string", enum: ["BUY", "SELL"] }, volume: { type: "number" }, sl: { type: "number" }, tp: { type: "number" } }, required: ["symbol", "action", "volume"] } } },
  { type: "function", function: { name: "modify_mt5_order", description: "Modify SL/TP of an open MT5 position via guided on-device UI steps (Trade tab -> long-press -> Modify). If unsure about MT5 menu layout, call search_web first (e.g. 'MT5 android modify SL TP steps').", parameters: { type: "object", properties: { symbol: { type: "string" }, sl: { type: "number" }, tp: { type: "number" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "close_mt5_order", description: "Close an open MT5 position via guided on-device UI steps (Trade tab -> long-press -> Close).", parameters: { type: "object", properties: { symbol: { type: "string" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "get_mt5_positions", description: "Read currently visible MT5 positions from the phone screen (symbol, side, volume, P/L). Call read_screen first if empty.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "get_systems_status", description: "Get overall server system status.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "wait_and_verify", description: "Wait a moment before verifying state (use after actions that take time).", parameters: { type: "object", properties: { delay_ms: { type: "number" } } } } },
  { type: "function", function: { name: "assert_text_visible", description: "Verify that text is visible on the last screen state.", parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } } },
  { type: "function", function: { name: "get_frit_manual", description: "Get a full reference of every real tool FRIT has, grouped by category, plus how to use the phone's installed-apps list correctly. Call this only if you're unsure what capabilities you have — don't call it for every task.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "delegate_subtasks", description: "Fan OUT independent subtasks to parallel subagent models (different providers, zero extra agent turns). Use for: researching several angles at once, drafting + summarizing while you keep driving the phone, comparing options. Args: subtasks = [{task, kind}] where kind is lookup/extract/draft/summarize/research/analyze/code. Max 4 per call. Results come back merged in one response.", parameters: { type: "object", properties: { subtasks: { type: "array", items: { type: "object", properties: { task: { type: "string" }, kind: { type: "string" } }, required: ["task"] } } }, required: ["subtasks"] } } },
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
    case "get_market_data": return { ok: true, data: await fetchMarketPrices([args.symbol || "BTCUSD"]) };
    case "get_market_news": {
      const sym = args.symbol || "market";
      const q = args.query || `${sym} market fundamental news ${new Date().getFullYear()}`;
      return { ok: true, data: await webSearch(q) };
    }
    case "analyze_market": return { ok: true, data: await mtfStrategy.analyze(args.symbol, { interval: args.interval, balance: args.balance, riskPercent: args.risk_percent }) };
    case "run_code": return { ok: true, data: await runSandbox({ language: args.language, code: args.code, stdin: args.stdin || "", timeout_ms: args.timeout_ms || 15000 }) };
    case "get_frit_manual": return { ok: true, data: buildFritManual() };
    // Fan-out: run independent subtasks in PARALLEL across providers so the
    // primary brain isn't the bottleneck. Each subtask gets the cheapest
    // capable model on a DIFFERENT provider (spreads rate-limit + cost load):
    //   - fast lookup/extract  -> Groq gpt-oss-20b (1000 tok/s, $0.075/M)
    //   - drafting/summarize   -> Mistral Small (inside your 1M/mo free tier)
    //   - research/heavy lift  -> Zen Big Pickle (free pool)
    // Local 1.7B can also take subtasks via execute_local_action when offline.
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
const SUBAGENT_POOL = [
  { slot: "speed", model: "groq:openai/gpt-oss-20b", kinds: ["lookup", "extract", "classify", "quick"] },
  { slot: "draft", model: "mistral-small-latest", kinds: ["draft", "summarize", "rewrite", "plan"] },
  // Heavy slot follows the Qwen migration: 3.8-27B leads research/code
  // (SWE-Pro 61.7, OSWorld-Verified 84.3), DeepSeek direct if keyed.
    { slot: "heavy", model: (typeof HAS_GROQ !== "undefined" && HAS_GROQ) ? "groq:moonshotai/kimi-k2-instruct-0905" : ((typeof HAS_DEEPSEEK !== "undefined" && HAS_DEEPSEEK) ? `deepseek:${DEEPSEEK_MODEL}` : (typeof MISTRAL_MODEL !== "undefined" && MISTRAL_MODEL ? MISTRAL_MODEL : "groq:openai/gpt-oss-120b")), kinds: ["research", "compare", "analyze", "code"] },
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
      out => ({ index: i, model, ok: true, result: String(out || "").slice(0, 2500) }),
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
function buildAutomationSystemPrompt({ deviceState, memory, ledger = [], goal = "", tradeMemory = "", verification = null, lastFailure = "" }) {
  const deviceStateText = buildDeviceStateBlock(deviceState);
  const memoryText = buildMemoryBlock(summarizeMemory(memory, 6, 700));
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
    "You are FRIT, an autonomous Android AI Agent. You operate the phone exactly like a human: observing the screen, planning, acting, and verifying.",
    "CRITICAL RULE: YOU MUST BE AGENTIC AND PERSISTENT.",
    "",
    "# FRIT Mobile Agent Execution Mindset:",
    "- UI Asynchrony: Android UIs do not refresh instantly. After executing a structural tap or typing text, always assume an animation or network lag of 300-800ms.",
    "- Flaky Element Matching: Resource IDs change between app updates, and text labels may contain leading/trailing whitespaces. Always use fuzzy substring matching if an exact match fails.",
    "- Coordination Safety: Never issue raw coordinates (tap_coordinates) unless element-based text anchors (tap_button) are entirely absent from the structured screen dump. Bounding boxes shift based on device display scaling and DPI variations.",
    "- Recovery: If an execution path blocks or fields are missing, do not hallucinate success. Tap go_back, re-examine the screen text structure, or call take_screenshot to confirm the visual layer.",
    "",
    "1. OBSERVE: Use 'read_screen' or 'read_screen_structured' to see what's on screen.",
    "2. ANALYZE: If you don't see what you need, ANALYZE why. Maybe the app isn't open? Maybe you need to scroll?",
    "3. ACT: Decide on ONE next step (tap, type, scroll, go_back, press_home).",
    "4. VERIFY: Immediately call 'read_screen' again to see the result of your action. Did it work? If not, self-correct.",
    "",
    "HARD RULES — VIOLATING THESE IS A FAILURE:",
    "- NEVER write a fake action as plain text or a markdown code block (e.g. 'Action: ```python get_market_data(...)```'). That is not a real tool call and does NOTHING. If you need a tool, you MUST use the actual function-calling mechanism provided to you — never describe, narrate, or pretend to call a tool in prose.",
    "- If you genuinely cannot call a tool (none fits), say so directly in one sentence. Do not paste code for the user to run manually as a substitute for calling 'run_code' yourself — you have 'run_code', use it.",
    "- Only call DEVICE-CONTROL tools (open_app, tap, type, scroll, go_back, press_home) when the user's message clearly asks for a phone action. For greetings or small talk with no request in them, reply in plain conversational text with ZERO tool calls — do not invent a phone task out of a greeting like 'hi'.",
    "- This restriction does NOT apply to server-side analysis/data tools (analyze_market, get_market_data, search_web, run_code, get_weather). If the user asks a question those tools can answer — e.g. 'what's your analysis on XAUUSD', 'search X', 'what's the weather' — CALL the relevant tool immediately. A question is still a request; don't treat 'they didn't say an imperative command' as a reason to skip the tool and answer from memory instead.",
    "- For 'open_app': the 'app_name' argument must be ONLY the literal app name (e.g. 'WhatsApp', 'Messenger') — never a sentence, instruction, or task description. Open the app first, THEN use separate tool calls (read_screen, tap, type) to carry out the actual task once it's open.",
    "- SETTINGS TOGGLES (bluetooth/wifi/data/airplane/location/battery): step 1 call 'execute_local_action' (e.g. 'open bluetooth') — it lands directly on the right page. Step 2 'read_screen', then 'tap_button' the toggle. Step 3 'read_screen' to confirm it flipped. Step 4 call 'return_to_frit'. Never navigate Settings menus manually.",
    "- DIVISION OF LABOR: the phone's local engine owns launching apps and system shortcuts (open_app, execute_local_action). You own analysis, decisions, and every tap/type INSIDE an app. Never navigate to an app manually; launch it, then act on the screen text the launch returns.",
    "- If the device state below lists 'Installed apps', ONLY target names from that list with open_app — do not guess an app exists if it isn't listed. If it's not there, tell the user instead of trying anyway.",
    "- UNFAMILIAR APP UI (Opay, Facebook, MT5, any app you haven't driven in THIS session): BEFORE tapping blindly, spend ONE 'search_web' call on the exact flow — e.g. 'Opay Android app how to transfer money steps 2026', 'Facebook Android app create post steps'. Combine that walkthrough with the live screen text and NEVER second-guess: screen text always wins over the article when they disagree. Skip the search only for apps/flows you already completed successfully in this session.",
    "- PARALLELIZE with 'delegate_subtasks': independent research angles, per-option comparisons, or draft-while-you-drive work goes there (up to 4 at once across Groq/Mistral/Go) instead of burning sequential agent turns.",
    "- PAST FEEDBACK IS BINDING: user memory may contain 'feedback_negative: task=[...] bad_reply=[...]'. If the current goal matches such a task, you MUST use a different approach than the recorded bad reply — repeating it is a failure. 'feedback_positive' entries mark the approach to reuse.",
    "- You only have the tools explicitly provided to you in this request (open_app, read_screen, tap_button, type_text, run_code, search_web, get_market_data, analyze_market, send_whatsapp, make_call, etc.). Never assume a capability exists beyond that list — e.g. there is no generic 'send_message' or 'call_contact' tool, use the exact tool names you were given.",
    "",
    "TIPS FOR FULL AUTONOMY:",
    "- CHOOSE THE RIGHT TOOL: match the problem to the cheapest correct tool. Code/webpages/files/charts/math/backtesting → 'run_code' (server-side sandbox). Facts/news/research → 'search_web'. Weather → 'get_weather'. Market prices/analysis → 'get_market_data'/'analyze_market'. Phone app control → device tools (open_app, tap, type).",
    "- NEVER write or run code on the phone. Do not open IDE apps (PyCharm, etc.) on the device to build code — the phone is only for UI actions on OTHER apps. Code always goes through 'run_code' on the server, and any file it produces is returned to the user automatically.",
    "- If the user asks for something that can be DONE server-side (a webpage, a chart, a report, a calculation, a search), do it with the server-side tool — never invent a phone workflow for it.",
    "- To open any app: If not visible, press_home -> click search bar or use 'open_app'.",
    "- To find a specific button: Use 'read_screen_structured' to get exact coordinates if text-matching fails.",
    "- If a tool fails: Don't give up. Try a different approach (e.g., tap_coordinates instead of tap_button).",
    "- Run Code: Use 'run_code' for complex logic, math, or data processing. Don't guess calculations.",
    "- Browse: Use 'search_web' to find information.",
    "- Market/trading news & fundamentals: ALWAYS call 'get_market_news' or 'search_web' to retrieve current live 2025/2026 market news. NEVER cite outdated news from 2024 or earlier memory!",
     "- Market/trading tasks: analysis happens HERE on the server, NOT on the phone. Actually CALL the 'analyze_market' or 'get_market_data' tool (a real function call) and read the returned direction/entry/SL/TP — do not narrate calling it. Only use the phone (open_app MetaTrader5, tap, type) to EXECUTE an order after the analysis is complete.",
     "- TRADING NUMBERS DISCIPLINE: report the engine's decision/entry_zone/scenario/pullback_health fields VERBATIM. NEVER invent MACD, Bollinger, ADX/DI, or session values the tools did not return — if you want an indicator the engine lacks, compute it with run_code from real candles, never from memory. When decision is WAIT_PULLBACK, present ONLY the pullback-zone entry (limit-style); never substitute a market entry. When a healthy pullback exists, scenario_2 IS the trade — scenario_1 immediates apply only when the engine is aligned.",
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
const frameBuffer = [];
const FRAME_BUFFER_SIZE = 10;

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
      strategy: ["/enhanced/analyze", "/market/strategy", "/strategy/status", "/systems/status"],
      memory: ["/memory/trade"],
      screen: ["/screen/frame", "/screen/analyze-frame", "/screen/status"],
      transcribe: ["/transcribe"],
      tools: ["/tools/search"],
    utility: ["/weather"],
    sandbox: ["/sandbox/run"],
  },
});
});

app.get("/health", (_req, res) => {
  res.json({
    status: "active",
    models: MODELS,
    twelve_data: !!TWELVE_DATA_KEY,
    mt5_bridge: !!MT5_BRIDGE_URL,
    sandbox_url: SANDBOX_URL,
    sandbox_auth: !!SANDBOX_AUTH,
    frame_buffer: frameBuffer.length,
    cache_entries: _cache.size,
    uptime: Math.floor(process.uptime()) + "s",
  });
});


app.post("/screen/frame", requireAuth, (req, res) => {
  const { frameData, timestamp, width, height, current_app, screen_text, current_activity } = req.body || {};
  if (!frameData) return res.status(400).json({ error: "frameData required" });
  frameBuffer.push({ data: frameData, timestamp: timestamp || Date.now(), width, height });
  if (frameBuffer.length > FRAME_BUFFER_SIZE) frameBuffer.shift();
  res.json({ status: "received", buffered: frameBuffer.length });
});

app.post("/screen/analyze-frame", requireAuth, async (req, res) => {
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
  if (!frameBuffer.length) return res.status(400).json({ error: "No frames in buffer. Android app must send frames first via /screen/frame." });
  const frame = frameIndex >= 0 && frameIndex < frameBuffer.length ? frameBuffer[frameIndex] : frameBuffer.at(-1);
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
      frameIndex: frameIndex >= 0 ? frameIndex : frameBuffer.length - 1,
    });
  } catch (err) {
    res.status(500).json({ error: "Frame analysis failed", details: err.message });
  }
});

app.get("/screen/status", requireAuth, (_req, res) => {
  res.json({
    buffered_frames: frameBuffer.length,
    max_buffer: FRAME_BUFFER_SIZE,
    oldest_frame_ts: frameBuffer[0]?.timestamp || null,
    newest_frame_ts: frameBuffer.at(-1)?.timestamp || null,
  });
});

app.post("/screen/clear", requireAuth, (_req, res) => {
  frameBuffer.length = 0;
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
    res.json(await fetchMarketPrices(symbols.map(s => String(s).toUpperCase())));
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
app.post("/trade", requireAuth, async (req, res) => {
  const { symbol, risk_percent = 1, balance, reason = "", interval = "1h" } = req.body || {};
  if (!symbol) return res.status(400).json({ error: "symbol required" });

  try {
    const result = await mtfStrategy.run(symbol, { interval, balance: balance || 1000, riskPercent: risk_percent });

    if (["NO_TRADE", "WAIT", "WAIT_PULLBACK", "COOLDOWN", "DATA_UNAVAILABLE", "DATA_RATE_LIMITED", "ERROR"].includes(result.decision)) {
      return res.status(200).json({ status: "blocked", ...result });
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
      rr: result.rr,
      confidence: result.confidence,
      regime: result.regime,
      structure: result.structure_30m,
      entry_ctx: result.entry_ctx,
      guards: result.guards,
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

app.get("/weather", async (req, res) => {
  try {
    const city = String(req.query.city || "Lagos");
    const result = await getWeather(city);
    res.json({ city, result });
  } catch (err) {
    res.status(500).json({ error: "Weather failed", details: err.message });
  }
});

app.post("/transcribe", requireAuth, async (req, res) => {
  try {
    const { audio_base64, mime_type = "audio/webm" } = req.body || {};
    if (!audio_base64) return res.status(400).json({ error: "audio_base64 required" });
    const text = await mistralTranscribe(audio_base64, mime_type);
    if (!text) return res.status(500).json({ error: "Transcription failed" });
    res.json({ text, model_used: MODELS.voxtral });
  } catch (err) {
    console.error("[/transcribe]", err.message);
    res.status(500).json({ error: "Transcription failed", details: err.message });
  }
});

app.post("/sandbox/run", requireAuth, async (req, res) => {
  try {
    const result = await runSandbox(req.body);
    res.json(result);
  } catch (err) {
    console.error("[/sandbox/run]", err.message);
    res.status(500).json({ error: "Sandbox execution failed", details: err.message });
  }
});

// ==================== WEB SEARCH ====================
// /tools/search — raw web-search endpoint used by the agent brain (search_web tool)
// and by any client that wants real-time info without going through the LLM.
app.post("/tools/search", requireAuth, async (req, res) => {
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

// ====== ERROR HANDLER ======
app.use((err, _req, res, _next) => {
  console.error("Unhandled Error!", err);
  res.status(500).json({ error: "Internal server error", details: err?.message || "Unknown error" });
});

// ====== START ======
app.listen(PORT, () => {
  console.log(`
FRIT - engine.js Trading Engine & Mistral AI Orchestrator
Port : ${String(PORT).padEnd(5)}
Single trading engine (server/src/strategy/engine.js):
 - 30M primary + 4H confirmation (EMA 9/21 + RSI + DI, ADX percentile gate)
 - News filter (FF calendar — 30 min advisory)
 - Multi-timeframe confirmation (4H trend regime)
Single Mistral ecosystem:
 - Mistral Medium — chat + agentic + vision
 - Voxtral Mini — STT
Infrastructure:
 - Lot size engine (per-pair pip math)
 - MT5 bridge (live) or paper mode
 - Code sandbox (Python/JS)
 - Android automation (State Machine: Verify-Act)
 - Screen frame ingestion -> vision analysis
  `);
});