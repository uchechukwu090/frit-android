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
import { FritSystems } from "./frit_systems.js";
import { analyzeSMC_CRT } from "./smc_crt_strategy.js";
import { MTFStrategyEngine } from "./strategy/engine.js";
import { TradeTaskScheduler } from "./strategy/scheduler.js";
import { PositionMonitor } from "./strategy/position_monitor.js";

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
// Model id on OpenCode Zen — confirm the exact free-tier id at
// https://opencode.ai/zen/v1/models before relying on this default, since
// Zen's free listings are labeled "free for a limited time" and can change.
const ZEN_MODEL = process.env.OPENCODE_ZEN_MODEL || "deepseek-v4-flash-free";
const MISTRAL_MODEL = process.env.MISTRAL_MODEL || "mistral-large-latest";
const MISTRAL_FAST_MODEL = process.env.MISTRAL_FAST_MODEL || "mistral-small-latest";
const TWELVE_DATA_KEY = process.env.TWELVE_DATA_KEY || "";
const SANDBOX_URL = process.env.SANDBOX_URL || "https://sandbox-rexv.onrender.com";
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
// Model resolution order: explicit per-role env override -> provider priority.
// Per your request, OpenCode Zen (DeepSeek V4 Flash Free — confirmed native
// tool calling, 1M context) is now the primary brain for agent/tools roles,
// with Groq Llama 3.3 70B as the first fallback (also strong tool calling,
// solid TPM), then Gemini 3.6 Flash, then Mistral Large last (weakest
// observed tool-calling reliability of the four).
const MODELS = {
  vision: process.env.VISION_MODEL || (GEMINI_API_KEY ? "gemini-3.6-flash" : "pixtral-large-latest"),
  agent: process.env.AGENT_MODEL || (HAS_ZEN ? `zen:${ZEN_MODEL}` : (HAS_GROQ ? "groq:llama-3.3-70b-versatile" : (GEMINI_API_KEY ? "gemini-3.6-flash" : MISTRAL_MODEL))),
  conversation: process.env.CONVERSATION_MODEL || (HAS_GROQ ? "groq:llama-3.1-8b-instant" : (HAS_ZEN ? `zen:${ZEN_MODEL}` : (GEMINI_API_KEY ? "gemini-3.6-flash" : MISTRAL_MODEL))),
  tools: process.env.TOOLS_MODEL || (HAS_ZEN ? `zen:${ZEN_MODEL}` : (HAS_GROQ ? "groq:llama-3.3-70b-versatile" : (GEMINI_API_KEY ? "gemini-3.6-flash" : MISTRAL_MODEL))),
  coding: process.env.CODING_MODEL || (HAS_GROQ ? "groq:qwen/qwen3.6-27b" : "codestral-latest"),
  fast: process.env.FAST_MODEL || (HAS_GROQ ? "groq:llama-3.1-8b-instant" : (GEMINI_API_KEY ? "gemini-3.6-flash" : MISTRAL_FAST_MODEL)),
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

// Runtime fallback chains: if the primary model for a role fails (e.g. rate
// limit exhausted, or fewer/no models registered), walk to the next provider
// instead of throwing "brain disconnected" up to the client.
// Priority order, by observed tool-calling reliability + free-tier strength:
//   1. OpenCode Zen (DeepSeek V4 Flash Free — native tool calling, 1M ctx)
//   2. Groq Llama 3.3 70B (strong tool calling, generous TPM)
//   3. Gemini 3.6 Flash (solid tool calling, smaller free-tier RPM/RPD)
//   4. Mistral Large (kept last — weaker observed tool-calling reliability
//      on multi-field function args, per production logs)
function buildFallbackChain(primary) {
  const chain = [primary];
  if (HAS_ZEN) chain.push(`zen:${ZEN_MODEL}`);
  if (HAS_GROQ) chain.push("groq:llama-3.3-70b-versatile");
  if (GEMINI_API_KEY) chain.push("gemini-3.6-flash");
  if (MISTRAL_MODEL) chain.push(MISTRAL_MODEL);
  return [...new Set(chain)];
}

const FALLBACK_CHAINS = {
  agent: buildFallbackChain(MODELS.agent),
  tools: buildFallbackChain(MODELS.tools),
  conversation: buildFallbackChain(MODELS.conversation),
  fast: buildFallbackChain(MODELS.fast),
};

// Same signature as mistralChat, but walks a role's fallback chain on failure
// (e.g. Groq rate limit exhausted) instead of bubbling the error straight up.
async function chatWithFallback(role, { messages, tools = null, tool_choice = "auto", temperature, max_tokens }) {
  const chain = FALLBACK_CHAINS[role] || [MODELS[role] || MODELS.conversation];
  const attempts = [];
  for (let i = 0; i < chain.length; i++) {
    const model = chain[i];
    try {
      const out = await mistralChat({ model, messages, tools, tool_choice, temperature, max_tokens });
      if (i > 0) console.warn(`[chatWithFallback] role=${role} recovered on fallback model ${model} (primary failed)`);
      return out;
    } catch (err) {
      attempts.push(`${model}: ${err.message}`);
      console.warn(`[chatWithFallback] role=${role} model=${model} failed: ${err.message}`);
    }
  }
  // Every model in the chain failed — surface a clear, distinguishable error
  // (rather than just the last provider's raw message) so the client/UI can
  // show something better than a generic "disconnected" for this case.
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
// "zen:..."   -> OpenCode Zen (OpenAI-compatible /chat/completions, native tool calling)
// "gemini-*"  -> Gemini native generateContent (vision; tools unsupported here)
// otherwise    -> Mistral (OpenAI-compatible chat completions, supports tool calls)
function resolveProvider(model) {
  if (typeof model === "string" && model.startsWith("groq:")) return "groq";
  if (typeof model === "string" && model.startsWith("zen:")) return "zen";
  if (typeof model === "string" && model.startsWith("gemini")) return "gemini";
  return "mistral";
}

async function mistralChat({ model, messages, tools = null, temperature = 0.3, max_tokens = 1600, tool_choice = "auto", retries = 3 }) {
  const provider = resolveProvider(model);
  if (provider === "gemini") return geminiChat({ model, messages, temperature, max_tokens });

  const base = provider === "groq" ? "https://api.groq.com/openai/v1/chat/completions"
    : provider === "zen" ? "https://opencode.ai/zen/v1/chat/completions"
    : `${MISTRAL_BASE}/chat/completions`;
  const apiKey = provider === "groq" ? GROQ_API_KEY : provider === "zen" ? ZEN_API_KEY : MISTRAL_API_KEY;
  const cleanModel = (provider === "groq" || provider === "zen") ? model.slice(model.indexOf(":") + 1) : model;

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
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
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

  // ---- 2. Verification of the last action batch (independent second model) ----
  let verification = null;
  if (toolResults && deviceState) {
    verification = await verifyLastStep(session, ledger, toolResults, deviceState);
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
  const deviceStateForPrompt = deviceState || (session.last_device_state ? safeJsonParse(session.last_device_state, {}) : {});
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
  // Groq rate limit (e.g. llama-3.3-70b-versatile) doesn't kill the whole turn.
  const modelRole = modelToUse === MODELS.agent ? "agent"
    : modelToUse === MODELS.fast ? "fast"
    : modelToUse === MODELS.tools ? "tools"
    : "conversation";

  for (let iter = 0; iter <= MAX_SERVER_TOOL_LOOP; iter++) {
    const out = await chatWithFallback(modelRole, { messages: fullMessages, tools: AGENT_TOOLS });
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
    for (const tc of rawAndroidCalls) {
      if (tc.function.name === "open_app") {
        const appName = (tc.function.arguments || {}).app_name || "";
        if (appName.length > MAX_APP_NAME_LEN || looksLikeSentence(appName)) {
          badAppCalls.push(tc);
          continue;
        }
        // Ground against the real installed-apps list (if the device sent
        // one) so a near-miss like "Messenger" vs "Messages" self-corrects
        // instead of bouncing back to the model or failing on-device.
        const resolved = resolveInstalledApp(appName, deviceStateForPrompt);
        if (resolved && resolved !== appName) {
          tc.function.arguments = { ...tc.function.arguments, app_name: resolved };
        }
      }
      androidCalls.push(tc);
    }

    if (badAppCalls.length) {
      for (const tc of badAppCalls) {
        const content = JSON.stringify({
          ok: false,
          error: "Invalid app_name — it must be ONLY the literal app name (e.g. 'Messenger', 'WhatsApp'), not a sentence or task description. Call open_app again with just the app name, then use separate tool calls (read_screen, tap, type) once it's open to carry out the task.",
        });
        fullMessages.push({ role: "tool", tool_call_id: tc.id, name: tc.function.name, content });
      }
      if (iter < MAX_SERVER_TOOL_LOOP) continue; // let the model self-correct
    }

    if (serverCalls.length && iter < MAX_SERVER_TOOL_LOOP) {
      for (const tc of serverCalls) {
        const name = tc.function.name;
        const args = tc.function.arguments || {};
        let result;
        try {
          result = await runLocalTool(name, args, { deviceState: deviceStateForPrompt });
        } catch (err) {
          result = { ok: false, error: err.message };
        }
        // run_code can produce real files (charts, CSVs, backtest reports).
        // Collect them for the client response, but strip the raw base64
        // out of what goes back into the model's own context — feeding
        // megabytes of encoded image data into every subsequent LLM call
        // would blow the token budget for no benefit (the model can't see
        // images this way anyway; it only needs to know a file was made).
        if (name === "run_code" && Array.isArray(result?.data?.artifacts) && result.data.artifacts.length) {
          for (const art of result.data.artifacts) turnArtifacts.push(art);
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
      continue; // feed results back to the brain
    }

    androidPending = androidCalls;
    break;
  }

  const done = androidPending.length === 0;

  // A ledger only represents a real task if something in it is actually
  // active/pending/attention. A plain chat turn (e.g. "hi") produces zero
  // Android tool calls too, but there's no task to "complete" — so we must
  // not force the ledger to "done"/"completed" in that case.
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
app.post("/agent/start", requireAuth, async (req, res) => {
  const { goal, device_state, memory } = req.body;
  if (!goal) return res.status(400).json({ error: "Goal required" });

  const sessionId = `sess_${Date.now()}`;

  try {
    // Initial Plan / Ledger creation
    const planOut = await chatWithFallback("fast", {
      messages: [
        { role: "system", content: "Break the user's goal into an ordered JSON list of subtasks: [{\"seq\": 1, \"description\": \"...\", \"status\": \"pending\"}]" },
        { role: "user", content: goal }
      ]
    });

    let ledger = [];
    try {
      const parsed = JSON.parse(planOut.choices[0].message.content.match(/\[.*\]/s)[0]);
      ledger = (Array.isArray(parsed) ? parsed : []).map((t, i) => ({
        seq: t.seq ?? i + 1,
        description: String(t.description || t.task || goal),
        status: t.status || "pending",
      }));
    } catch (e) {
      ledger = [{ seq: 1, description: goal, status: "pending" }];
    }

    const safeMemory = Array.isArray(memory) ? memory : [];
    db.prepare("INSERT INTO agent_sessions (id, goal, task_ledger, last_device_state, memory) VALUES (?, ?, ?, ?, ?)").run(
      sessionId, goal, JSON.stringify(ledger), JSON.stringify(device_state || {}), JSON.stringify(safeMemory)
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

// STT via Mistral Voxtral (fallback: Groq Whisper, then OpenAI Whisper)
async function mistralTranscribe(audioBase64, mimeType = "audio/wav") {
  const base64Data = audioBase64.includes(",") ? audioBase64.split(",")[1] : audioBase64;
  const audioBuffer = Buffer.from(base64Data, "base64");
  // Normalize to a bare mime type (strip any codec suffix like ";codecs=opus")
  // and fall back to wav if the client didn't send one Mistral will accept.
  const cleanMime = (mimeType || "audio/wav").split(";")[0].trim() || "audio/wav";

  // Try Mistral Voxtral first
  try {
    const res = await fetch("https://api.mistral.ai/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${MISTRAL_API_KEY}` },
      body: JSON.stringify({
        model: MODELS.voxtral,
        messages: [
          { role: "system", content: "Transcribe the user's speech accurately. Return only the transcribed text, no explanations." },
          { role: "user", content: [{ type: "audio_url", audio_url: { url: `data:${cleanMime};base64,${base64Data}` } }] },
        ],
        max_tokens: 800,
      }),
    });
    if (res.ok) {
      const data = await res.json();
      const text = data?.choices?.[0]?.message?.content?.trim();
      if (text) return text;
    } else {
      const errBody = await res.text().catch(() => "");
      console.warn(`[MistralTranscribe] Mistral Voxtral failed: ${res.status} mime=${cleanMime} body=${errBody.slice(0, 300)}`);
    }
  } catch (e) { console.warn("[MistralTranscribe] Mistral error:", e.message); }

  // Fallback: Groq Whisper
  if (process.env.GROQ_API_KEY) {
    try {
      const form = new FormData();
      const ext = cleanMime.split("/")[1] || "wav";
      form.append("file", audioBuffer, { filename: `audio.${ext}`, contentType: cleanMime });
      form.append("model", "whisper-large-v3-turbo");
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
async function runSandbox(args = {}) {
  try {
    const localRes = await fetch("http://127.0.0.1:8790/sandbox/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(args),
      signal: AbortSignal.timeout(4000),
    });
    if (localRes.ok) {
      const data = await localRes.json();
      return data;
    }
  } catch (_) {}
  const res = await fetch(`${SANDBOX_URL}/sandbox/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.details || data?.error || "Sandbox failed");
  return data;
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

const BINANCE_SYM = {
  BTC: "BTCUSD", ETH: "ETHUSD", SOL: "SOLUSD", BNB: "BNBUSD",
  XRP: "XRPUSD", DOGE: "DOGEUSD", ADA: "ADAUSD",
  BTCUSD: "BTCUSD", ETHUSD: "ETHUSD", SOLUSD: "SOLUSD", BNBUSD: "BNBUSD",
  XRPUSD: "XRPUSD", DOGEUSD: "DOGEUSD", ADAUSD: "ADAUSD",
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

  if (BINANCE_SYM[sym]) {
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

  if (TWELVE_DATA_KEY && TD_SYMBOLS[sym]) {
    try {
      const res = await fetch(`https://api.twelvedata.com/price?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&apikey=${TWELVE_DATA_KEY}`);
      const data = await res.json();
      if (data.price) result = { price: parseFloat(data.price), source: "twelvedata" };
    } catch (e) { console.error("[TwelveData spot]", sym, e.message); }
  }

  if (!result && COINGECKO_IDS[sym]) {
    try {
      const res = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${COINGECKO_IDS[sym]}&vs_currencies=usd&include_24hr_change=true`);
      const data = await res.json();
      const id = COINGECKO_IDS[sym];
      if (data[id]) result = { price: data[id].usd, change24h: data[id].usd_24h_change, source: "coingecko" };
    } catch (e) { console.error("[CoinGecko spot]", sym, e.message); }
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

// ======================== TECHNICAL INDICATORS ========================
function calcATR(candles, period = 14) {
  if (!candles || candles.length < period + 1) return 0;
  const trs = [];
  for (let i = 1; i < candles.length; i++) {
    const c = candles[i];
    const p = candles[i - 1];
    trs.push(Math.max(c.high - c.low, Math.abs(c.high - p.close), Math.abs(c.low - p.close)));
  }
  const recent = trs.slice(-period);
  return recent.reduce((a, b) => a + b, 0) / recent.length;
}

function calcSR(candles) {
  const window = candles.slice(-50);
  const swings = findSwings(window, 3);
  const currentPrice = candles.at(-1)?.close ?? 0;
  const support = swings.lows.map(s => s.price).filter(p => p < currentPrice).sort((a, b) => b - a)[0] ?? Math.min(...window.map(c => c.low));
  const resistance = swings.highs.map(s => s.price).filter(p => p > currentPrice).sort((a, b) => a - b)[0] ?? Math.max(...window.map(c => c.high));
  return { support, resistance };
}

function findSwings(candles, lookback = 2) {
  const highs = [];
  const lows = [];
  for (let i = lookback; i < candles.length - lookback; i++) {
    const cur = candles[i];
    let isHigh = true;
    let isLow = true;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (candles[j].high >= cur.high) isHigh = false;
      if (candles[j].low <= cur.low) isLow = false;
    }
    if (isHigh) highs.push({ index: i, price: cur.high, time: cur.time });
    if (isLow) lows.push({ index: i, price: cur.low, time: cur.time });
  }
  return { highs, lows };
}

function detectCandlePattern(candles) {
  if (!candles || candles.length < 2) return [];
  const a = candles[candles.length - 2];
  const b = candles[candles.length - 1];
  const patterns = [];
  const aBear = a.close < a.open;
  const aBull = a.close > a.open;
  const bBull = b.close > b.open;
  const bBear = b.close < b.open;
  if (aBear && bBull && b.open <= a.close && b.close >= a.open) patterns.push("bullish_engulfing");
  if (aBull && bBear && b.open >= a.close && b.close <= a.open) patterns.push("bearish_engulfing");
  const body = Math.abs(b.close - b.open);
  const upperWick = b.high - Math.max(b.open, b.close);
  const lowerWick = Math.min(b.open, b.close) - b.low;
  if (body > 0) {
    if (lowerWick > body * 2 && upperWick < body) patterns.push("pinbar_bullish");
    if (upperWick > body * 2 && lowerWick < body) patterns.push("pinbar_bearish");
    if (body < (b.high - b.low) * 0.35) patterns.push("indecision");
  }
  return patterns;
}

// ======================== VOLUME PROFILE (Market Auction) ========================
function calcAuction(candles, buckets = 40) {
  if (!candles || candles.length < 10) return null;
  const window = candles.slice(-100);
  const totalRawVol = window.reduce((s, c) => s + (c.volume || 0), 0);
  const hasRealVol = totalRawVol > window.length * 2;
  const hi = Math.max(...window.map(c => c.high));
  const lo = Math.min(...window.map(c => c.low));
  if (hi === lo) return null;
  const bucketSize = (hi - lo) / buckets;
  const vap = new Array(buckets).fill(0);
  for (const c of window) {
    const rangeProxy = c.high - c.low || bucketSize;
    const bodySize = Math.abs(c.close - c.open) || rangeProxy * 0.3;
    const vol = hasRealVol ? (c.volume > 0 ? c.volume : rangeProxy) : rangeProxy * (1 + bodySize / rangeProxy);
    const candleRange = c.high - c.low || bucketSize;
    for (let b = 0; b < buckets; b++) {
      const bLo = lo + b * bucketSize;
      const bHi = bLo + bucketSize;
      const overlap = Math.max(0, Math.min(c.high, bHi) - Math.max(c.low, bLo));
      vap[b] += vol * (overlap / candleRange);
    }
  }
  const pocIdx = vap.indexOf(Math.max(...vap));
  const poc = lo + (pocIdx + 0.5) * bucketSize;
  const totalVol = vap.reduce((a, b) => a + b, 0);
  const target = totalVol * 0.7;
  let lo_idx = pocIdx, hi_idx = pocIdx, accumulated = vap[pocIdx];
  while (accumulated < target) {
    const addLo = lo_idx > 0 ? vap[lo_idx - 1] : 0;
    const addHi = hi_idx < buckets - 1 ? vap[hi_idx + 1] : 0;
    if (addLo === 0 && addHi === 0) break;
    if (addHi >= addLo) { hi_idx++; accumulated += addHi; }
    else { lo_idx--; accumulated += addLo; }
  }
  const vah = lo + (hi_idx + 1) * bucketSize;
  const val = lo + lo_idx * bucketSize;
  const sorted = vap.map((v, i) => ({ v, price: lo + (i + 0.5) * bucketSize })).sort((a, b) => b.v - a.v);
  return {
    poc, vah, val,
    hvn: sorted.slice(0, 3).map(x => x.price),
    lvn: sorted.slice(-5).map(x => x.price),
    range_hi: hi,
    range_lo: lo,
    volume_mode: hasRealVol ? "real" : "proxy",
  };
}

function auctionSignal(price, auction) {
  if (!auction) return { position: "unknown", bias: "neutral", note: "" };
  const { poc, vah, val } = auction;
  if (price > vah) return { position: "above_value", bias: "bullish", note: "Price above Value Area — buyers in control." };
  if (price < val) return { position: "below_value", bias: "bearish", note: "Price below Value Area — sellers in control." };
  if (price > poc) return { position: "inside_value_upper", bias: "mild_bullish", note: "Inside Value Area above POC — mean reversion risk, watch VAH." };
  return { position: "inside_value_lower", bias: "mild_bearish", note: "Inside Value Area below POC — mean reversion risk, watch VAL." };
}

// ==================== VOLATILITY ====================
function analyzeVolatility(candles, price) {
  const atr = calcATR(candles, 14);
  let regime = "normal";
  if (atr / (price || 1) < 0.0015) regime = "compressed";
  else if (atr / (price || 1) > 0.005) regime = "expanding";
  return { atr, regime };
}

// ==================== SCORING + TRADE PLAN ====================
function scoreSetup({ structure, price, support, resistance, patterns, auction, auctionSig, mtf, smcCrt, volatility }) {
  let bull = 0, bear = 0;
  const atr = volatility?.atr || 1;

  if (smcCrt?.structure) {
    const st = smcCrt.structure;
    if (st.trend === "uptrend") { bull += 4; bear -= 1; }
    else if (st.trend === "downtrend") { bear += 4; bull -= 1; }
    else if (st.trend === "potential_reversal_up") { bull += 2; }
    else if (st.trend === "potential_reversal_down") { bear += 2; }
  }
  if (smcCrt) {
    const isBuy = smcCrt.signal === "buy";
    const isSell = smcCrt.signal === "sell";
    if (isBuy) bull += Math.round(smcCrt.confidence / 8);
    if (isSell) bear += Math.round(smcCrt.confidence / 8);
    if (smcCrt.entry) {
      if (isBuy) { bull += 3; bear -= 1; }
      if (isSell) { bear += 3; bull -= 1; }
    }
    if (smcCrt.order_blocks?.bullish?.length > 0) bull += 1;
    if (smcCrt.order_blocks?.bearish?.length > 0) bear += 1;
    if (smcCrt.fvgs?.bullish?.length > 0) bull += 1;
    if (smcCrt.fvgs?.bearish?.length > 0) bear += 1;
    if (smcCrt.choch?.direction === "bullish") { bull += 3; if (isSell) bear -= 2; }
    if (smcCrt.choch?.direction === "bearish") { bear += 3; if (isBuy) bull -= 2; }
    const hasBullSweep = smcCrt.sweeps?.some(s => s.type === "bullish_sweep");
    const hasBearSweep = smcCrt.sweeps?.some(s => s.type === "bearish_sweep");
    if (hasBullSweep && isBuy) bull += 2;
    if (hasBearSweep && isSell) bear += 2;
    if (hasBullSweep && isSell) bear += 1;
    if (hasBearSweep && isBuy) bull += 1;
    if (smcCrt.crt) {
      if (isBuy) { bull += 3; bear -= 2; }
      if (isSell) { bear += 3; bull -= 2; }
    }
  }

  if (auction && auctionSig) {
    if (auctionSig.bias === "bullish") { bull += 3; bear -= 1; }
    else if (auctionSig.bias === "bearish") { bear += 3; bull -= 1; }
    else if (auctionSig.bias === "mild_bullish") bull += 1;
    else if (auctionSig.bias === "mild_bearish") bear += 1;
    if (auction.poc && Math.abs(price - auction.poc) / ((auction.vah - auction.val) || 1) < 0.1) {
      bull += 1;
      bear += 1;
    }
  }

  if (price > support && (price - support) / (price || 1) < 0.003) bull += 2;
  if (price < resistance && (resistance - price) / (price || 1) < 0.003) bear += 2;

  const levels = [support, resistance];
  if (auction) levels.push(auction.poc, auction.vah, auction.val);
  const nearLevel = (p) => levels.some(lvl => lvl && Math.abs(p - lvl) <= atr * 0.5);
  if (nearLevel(price)) {
    if (patterns.includes("bullish_engulfing") || patterns.includes("pinbar_bullish")) bull += 2;
    if (patterns.includes("bearish_engulfing") || patterns.includes("pinbar_bearish")) bear += 2;
    if (patterns.includes("indecision")) { bull -= 0.5; bear -= 0.5; }
  }

  if (volatility?.regime === "compressed") { bull -= 0.5; bear -= 0.5; }

  let mtfNote = "4H data unavailable";
  if (mtf) {
    if (mtf.trend === "up") { bull += 2; bear -= 1; mtfNote = "4H trend UP - favors longs"; }
    else if (mtf.trend === "down") { bear += 2; bull -= 1; mtfNote = "4H trend DOWN - favors shorts"; }
    else mtfNote = "4H trend neutral";
  }

  let bias = "neutral";
  const diff = bull - bear;
  if (diff >= 2) bias = "bullish";
  if (diff <= -2) bias = "bearish";
  const confidence = Math.max(5, Math.min(95, Math.round(50 + diff * 6)));

  return { bull_score: bull, bear_score: bear, bias, confidence, mtf_note: mtfNote };
}

function buildTradePlan({ bias, price, support, resistance, atr, dp }) {
  if (!price || !atr) return { entry_zone: null, invalidation: null, tp1: null, tp2: null, risk_state: "unknown" };
  if (bias === "bullish") {
    const entry1 = price - atr * 0.15;
    const entry2 = price + atr * 0.15;
    const invalidation = support > 0 ? support - atr * 0.25 : price - atr * 1.2;
    const tp1 = resistance > 0 ? resistance : price + atr * 1.2;
    const tp2 = resistance > 0 ? resistance + atr * 0.8 : price + atr * 2.2;
    return {
      entry_zone: `${entry1.toFixed(dp)} - ${entry2.toFixed(dp)}`,
      invalidation: invalidation.toFixed(dp),
      tp1: tp1.toFixed(dp),
      tp2: tp2.toFixed(dp),
      risk_state: "acceptable",
    };
  }
  if (bias === "bearish") {
    const entry1 = price - atr * 0.15;
    const entry2 = price + atr * 0.15;
    const invalidation = resistance > 0 ? resistance + atr * 0.25 : price + atr * 1.2;
    const tp1 = support > 0 ? support : price - atr * 1.2;
    const tp2 = support > 0 ? support - atr * 0.8 : price - atr * 2.2;
    return {
      entry_zone: `${entry1.toFixed(dp)} - ${entry2.toFixed(dp)}`,
      invalidation: invalidation.toFixed(dp),
      tp1: tp1.toFixed(dp),
      tp2: tp2.toFixed(dp),
      risk_state: "acceptable",
    };
  }
  return { entry_zone: null, invalidation: null, tp1: null, tp2: null, risk_state: "no_trade" };
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
    return {
      blocked: true,
      reason: `High-impact news within 30 min: ${nearby.map(e => `${e.currency} ${e.title}`).join(", ")}`,
      events: nearby,
    };
  }
  return { blocked: false };
}

// ===================== MTF CONFIRMATION =====================
async function getMTFBias(symbol) {
  const sym = String(symbol || "").toUpperCase();
  const ck = `mtf:${sym}`;
  const cached = cacheGet(ck);
  if (cached) return cached;
  try {
    const candles4h = await fetchCandles(sym, "4h", 100);
    if (!candles4h || candles4h.length < 50) return { trend: "neutral", structure: null };
    const swings = findSwings(candles4h);
    const recentHighs = swings.highs.slice(-3);
    const recentLows = swings.lows.slice(-3);
    const hh = recentHighs.length >= 2 && recentHighs.at(-1).price > recentHighs.at(-2).price;
    const hl = recentLows.length >= 2 && recentLows.at(-1).price > recentLows.at(-2).price;
    const lh = recentHighs.length >= 2 && recentHighs.at(-1).price < recentHighs.at(-2).price;
    const ll = recentLows.length >= 2 && recentLows.at(-1).price < recentLows.at(-2).price;
    let trend = "neutral";
    if (hh && hl) trend = "up";
    else if (lh && ll) trend = "down";
    const result = { trend, price: candles4h.at(-1)?.close ?? 0 };
    cacheSet(ck, result, 15 * 60 * 1000);
    return result;
  } catch (err) {
    console.warn("[MTF]", err.message);
    return { trend: "neutral", structure: null };
  }
}

// ===================== MAIN ANALYSIS =====================
async function analyzeSymbol(symbol, interval = "1h", customSize = null) {
  const sym = String(symbol || "").toUpperCase();
  const iv = normalizeInterval(interval);
  const ck = `analysis:${sym}:${iv}:${customSize || "auto"}`;
  const cached = cacheGet(ck);
  if (cached) return cached;

  const [candles, spot, newsCheck, mtf] = await Promise.all([
    fetchCandles(sym, iv, customSize),
    fetchSpotPrice(sym),
    checkNewsFilter(sym),
    getMTFBias(sym),
  ]);

  if (newsCheck.blocked) {
    return {
      symbol: sym,
      direction: "NEUTRAL",
      strength: "NEWS_BLACKOUT",
      interval: iv,
      news_filter: newsCheck,
      trade_plan: { entry_zone: null, invalidation: null, tp1: null, tp2: null, risk_state: "no_trade" },
      concise_signal: { direction: "STAND DOWN", entry: "N/A", sl: "N/A", tp: "N/A", ai_opinion: `News blackout active. ${newsCheck.reason}` },
      ai_opinion: `STAND DOWN — ${newsCheck.reason}`,
    };
  }
  if (!candles && !spot) return { symbol: sym, error: `No data for ${sym}` };
  const price = spot?.price ?? candles?.at(-1)?.close ?? 0;
  if (!candles || candles.length < 30) return { symbol: sym, price, source: spot?.source, analysis: "Insufficient candle data", candleCount: candles?.length ?? 0 };

  const { support, resistance } = calcSR(candles);
  const volatility = analyzeVolatility(candles, price);
  const patterns = detectCandlePattern(candles);
  const auction = calcAuction(candles);
  const auctionSig = auctionSignal(price, auction);
  const smcCrt = analyzeSMC_CRT(candles, price, volatility.atr, auction);
  const score = scoreSetup({ price, support, resistance, patterns, auction, auctionSig, mtf, smcCrt, volatility });

  let direction = "NEUTRAL";
  if (score.bias === "bullish") direction = "BULLISH";
  else if (score.bias === "bearish") direction = "BEARISH";
  let strength = "WEAK";
  if (score.confidence >= 75) strength = "STRONG";
  else if (score.confidence >= 60) strength = "MODERATE";
  const isCrypto = CRYPTO_SET.has(sym);
  const dp = isCrypto || sym === "XAUUSD" ? 2 : 5;

  let trade_plan;
  const smcEntry = smcCrt?.entry;
  if (smcEntry) {
    trade_plan = {
      entry_zone: `${smcEntry.price.toFixed(dp)}`,
      invalidation: smcEntry.invalidation.toFixed(dp),
      tp1: smcEntry.tp1.toFixed(dp),
      tp2: smcEntry.tp2.toFixed(dp),
      risk_state: "acceptable",
      method: smcEntry.type,
      reason: smcEntry.reason,
    };
  } else {
    trade_plan = buildTradePlan({ bias: score.bias, price, support, resistance, atr: volatility.atr, dp });
    trade_plan.method = "none";
  }

  const auctionNote = auctionSig.note ? `Auction: ${auctionSig.note}` : "";
  const smcReasons = smcCrt?.reasons?.length > 0 ? `SMC+CRT: ${smcCrt.reasons.join(";")}` : "";
  const aiOpinion = direction === "NEUTRAL"
    ? `No clear edge. ${smcReasons} ${auctionNote}`.trim()
    : `${direction} ${strength} | ${smcReasons} | ${auctionNote}`.trim();

  const sc = smcCrt?.structure || {};
  const result = {
    symbol: sym,
    price,
    direction,
    strength,
    interval: iv,
    candleCount: candles.length,
    source: spot?.source ?? "binance",
    support: +support.toFixed(dp),
    resistance: +resistance.toFixed(dp),
    confidence: score.confidence,
    structure: {
      trend: sc.trend || "ranging",
      last_swing_high: smcCrt?.last_swing_high ? +smcCrt.last_swing_high.toFixed(dp) : null,
      last_swing_low: smcCrt?.last_swing_low ? +smcCrt.last_swing_low.toFixed(dp) : null,
    },
    volatility: { atr: +volatility.atr.toFixed(dp), regime: volatility.regime },
    patterns,
    trade_plan,
    concise_signal: {
      direction,
      entry: trade_plan.entry_zone || "N/A",
      sl: trade_plan.invalidation || "N/A",
      tp: trade_plan.tp1 || "N/A",
      ai_opinion: aiOpinion,
    },
    auction: auction ? {
      poc: +auction.poc.toFixed(dp),
      vah: +auction.vah.toFixed(dp),
      val: +auction.val.toFixed(dp),
      position: auctionSig.position,
      bias: auctionSig.bias,
      note: auctionSig.note,
      hvn: auction.hvn.map(p => +p.toFixed(dp)),
      lvn: auction.lvn.map(p => +p.toFixed(dp)),
    } : null,
    mtf: { trend: mtf?.trend || "unknown", note: score.mtf_note },
    news_filter: { blocked: false },
    ai_opinion: aiOpinion,
    smc_crt: smcCrt ? {
      signal: smcCrt.signal,
      confidence: smcCrt.confidence,
      structure: sc,
      order_blocks: smcCrt.order_blocks,
      fvgs: smcCrt.fvgs,
      sweeps: smcCrt.sweeps,
      choch: smcCrt.choch,
      crt: smcCrt.crt,
      volume_profile: smcCrt.volume_profile,
      entry: smcCrt.entry ? {
        type: smcCrt.entry.type,
        price: +smcCrt.entry.price.toFixed(dp),
        invalidation: +smcCrt.entry.invalidation.toFixed(dp),
        tp1: +smcCrt.entry.tp1.toFixed(dp),
        tp2: +smcCrt.entry.tp2.toFixed(dp),
        reason: smcCrt.entry.reason,
      } : null,
      reasons: smcCrt.reasons,
    } : null,
    summary: `${sym} @${price.toFixed(dp)} | ${direction}(${strength}) | Conf:${score.confidence} | Structure:${sc.trend || "?"}(4H:${mtf?.trend || "?"}) | SMC:${smcCrt?.signal || "none"}(${smcCrt?.confidence || 0}) | VP:${auctionSig.position.replace("_", " ")} POC:${auction?.poc.toFixed(dp) || "?"} VAH:${auction?.vah.toFixed(dp) || "?"} VAL:${auction?.val.toFixed(dp) || "?"} | S:${support.toFixed(dp)} R:${resistance.toFixed(dp)} [${candles.length} ${iv}]`,
  };
  cacheSet(ck, result, 60000);
  return result;
}

// =================== WEB SEARCH + WEATHER ===================
async function webSearch(query) {
  try {
    const res = await fetch(`https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`);
    const d = await res.json();
    return d.AbstractText || d.Answer || d.RelatedTopics?.[0]?.Text || "No result found.";
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

// ==================== GSRI RISK OVERLAY ====================
const GSR1_LOCAL_PATH = process.env.GSR1_SNAPSHOT_PATH || join(__dirname, "gsri_snapshot.json");
const GSR1_REMOTE_URL = process.env.GSR1_REMOTE_URL || "";
const GSR1_ALPHA = 0.6;
const GSR1_MIN_SCALE = 0.2;
const GSR1_REMOTE_TTL = 5 * 60 * 1000;
let _gsriRemoteCache = null;

async function getGsriSnapshot() {
  try {
    const raw = readFileSync(GSR1_LOCAL_PATH, "utf8");
    const data = JSON.parse(raw);
    const snap = Array.isArray(data) ? data.at(-1) : data;
    if (snap && typeof snap === "object") return snap;
  } catch { /* absent or malformed */ }
  if (GSR1_REMOTE_URL) {
    const now = Date.now();
    if (_gsriRemoteCache && now - _gsriRemoteCache.ts < GSR1_REMOTE_TTL) return _gsriRemoteCache.snap;
    try {
      const r = await fetch(GSR1_REMOTE_URL, { signal: AbortSignal.timeout(6000) });
      if (r.ok) {
        const data = await r.json();
        const snap = Array.isArray(data) ? data.at(-1) : data;
        if (snap) { _gsriRemoteCache = { snap, ts: now }; return snap; }
      }
    } catch (e) { console.warn("[GSRI] Remote fetch failed:", e.message); }
  }
  return { Risk_Score: 0.8, Alert: 1, source: "fallback" };
}

function gsriLotScale(riskScore) {
  return Math.max(GSR1_MIN_SCALE, 1.0 - GSR1_ALPHA * Number(riskScore));
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

// ==================== ENHANCED SYSTEMS ====================
const frit = new FritSystems({
  fetchCandles, analyzeSymbol, cacheGet, cacheSet,
  checkNewsFilter, getGsriSnapshot, gsriLotScale, calculateLotSize,
  sendToMT5Bridge, addTradeMemory, getTradeMemory,
});

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
    if (pos.paper_trade_id) {
      try {
        const closed = frit.paperLogger.resolve(pos.paper_trade_id, pos.outcome === "win" ? "win" : "loss", pos.close_price);
        if (closed) console.log(`[PositionMonitor] paper trade ${pos.paper_trade_id} closed as ${pos.outcome}`);
      } catch (e) { console.warn("[PositionMonitor] paperLogger.resolve failed:", e.message); }
    }
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

if (process.env.AUTO_SCANNER === "true") {
  frit.startScanner(2 * 60 * 1000);
  console.log("[FRIT] Autonomous scanner started");
}

// ==================== ANDROID AGENT TOOLS ====================
// Contract: tools in SERVER_SIDE_TOOLS execute HERE (inside runAgentStep).
// Everything else in AGENT_TOOLS is returned to the Android app as a
// pending_action and executed by IntegrationCoordinator.executeEdgeTool().
// run_command / write_file / run_sandbox_code are deliberately NOT advertised:
// they stay gated behind DEV_TOOLS_ENABLED=false on the client.
const SERVER_SIDE_TOOLS = new Set(["search_web", "get_weather", "get_market_data",
  "analyze_market", "run_code", "wait_and_verify", "assert_text_visible", "get_frit_manual"
]);

const AGENT_TOOLS = [
  // ---- Phone/UI control (executed on Android) ----
  { type: "function", function: { name: "open_app", description: "Launch any installed app by name or package. Always verify afterwards with read_screen.", parameters: { type: "object", properties: { app_name: { type: "string" }, package: { type: "string" } } } } },
  { type: "function", function: { name: "read_screen", description: "Read visible text from the screen. Use this frequently to observe state and verify the result of every action.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "read_screen_structured", description: "Return exact coordinates of UI elements. Essential for precise clicking.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "tap_button", description: "Tap a button by its visible label/text.", parameters: { type: "object", properties: { label: { type: "string" } }, required: ["label"] } } },
  { type: "function", function: { name: "tap_element", description: "Tap a UI element by its text label (same as tap_button).", parameters: { type: "object", properties: { label: { type: "string" } }, required: ["label"] } } },
  { type: "function", function: { name: "tap_coordinates", description: "Tap specific x/y coordinates. Use when text-based tapping fails.", parameters: { type: "object", properties: { x: { type: "number" }, y: { type: "number" } }, required: ["x", "y"] } } },
  { type: "function", function: { name: "type_text", description: "Type text into the focused input field.", parameters: { type: "object", properties: { value: { type: "string" }, field: { type: "string" } }, required: ["value"] } } },
  { type: "function", function: { name: "scroll", description: "Scroll the UI in a direction.", parameters: { type: "object", properties: { direction: { type: "string", enum: ["up", "down", "left", "right"] } }, required: ["direction"] } } },
  { type: "function", function: { name: "go_back", description: "Press the Android back button.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "press_back", description: "Press the Android back button (alias of go_back).", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "press_home", description: "Go to the home screen.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "get_current_app", description: "Return the name of the currently focused app.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "get_current_activity", description: "Return the current Android activity/class.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "take_screenshot", description: "Capture the current screen for later analysis.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "analyze_screenshot", description: "Send the last screenshot to a vision model for UI analysis.", parameters: { type: "object", properties: { prompt: { type: "string" } } } } },
  { type: "function", function: { name: "set_volume", description: "Set media/alarm volume level (0-100).", parameters: { type: "object", properties: { level: { type: "number" } }, required: ["level"] } } },

  // ---- Communication & daily-life tasks (general-purpose) ----
  { type: "function", function: { name: "send_whatsapp", description: "Open a WhatsApp chat with a contact (by name) and draft a message. Then verify and send via UI.", parameters: { type: "object", properties: { contact_name: { type: "string" }, message: { type: "string" } }, required: ["contact_name", "message"] } } },
  { type: "function", function: { name: "send_sms", description: "Open the SMS composer to a contact with a message. Then verify and send via UI.", parameters: { type: "object", properties: { contact_name: { type: "string" }, message: { type: "string" } }, required: ["contact_name", "message"] } } },
  { type: "function", function: { name: "make_call", description: "Place a phone call to a contact.", parameters: { type: "object", properties: { contact_name: { type: "string" }, phone_number: { type: "string" } }, required: ["contact_name"] } } },
  { type: "function", function: { name: "set_alarm", description: "Set an alarm at a given time.", parameters: { type: "object", properties: { time: { type: "string" }, label: { type: "string" } }, required: ["time"] } } },
  { type: "function", function: { name: "set_timer", description: "Start a countdown timer.", parameters: { type: "object", properties: { duration: { type: "string" } }, required: ["duration"] } } },
  { type: "function", function: { name: "play_music", description: "Play music or media by query.", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
  { type: "function", function: { name: "navigate_to", description: "Open navigation/GPS to a destination.", parameters: { type: "object", properties: { destination: { type: "string" } }, required: ["destination"] } } },
  { type: "function", function: { name: "take_photo", description: "Open the camera app for a photo.", parameters: { type: "object", properties: { front_camera: { type: "boolean" } } } } },

  // ---- Server-side tools (execute on the server) ----
  { type: "function", function: { name: "run_code", description: "Execute Python/JS code for math, data parsing, backtesting, or logic tasks. Runs in a real sandbox — if your code writes a file (e.g. matplotlib chart to 'chart.png', a CSV, an HTML report) into its working directory, that file is captured and returned to the user as a visual/downloadable artifact. Prefer producing a file for anything the user would want to see or keep (charts, tables, reports) rather than just printing numbers.", parameters: { type: "object", properties: { language: { type: "string", enum: ["python", "javascript"] }, code: { type: "string" }, stdin: { type: "string" } }, required: ["language", "code"] } } },
  { type: "function", function: { name: "search_web", description: "Browse the internet for real-time information.", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
  { type: "function", function: { name: "get_weather", description: "Get current weather for a city.", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } },
  { type: "function", function: { name: "get_market_data", description: "Fetch live spot prices for one or more symbols (e.g. XAUUSD, BTCUSD).", parameters: { type: "object", properties: { symbol: { type: "string" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "analyze_market", description: "Run the FRIT SMC+CRT market analysis on a symbol (returns direction, entry, SL, TP, confidence).", parameters: { type: "object", properties: { symbol: { type: "string" }, interval: { type: "string" }, outputsize: { type: "number" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "get_market_quote", description: "Fetch a quick market quote for a symbol.", parameters: { type: "object", properties: { symbol: { type: "string" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "get_acp_status", description: "Get the Automated Conviction Proxy for a symbol (direction, confidence, paper-trade win rate, crash regime).", parameters: { type: "object", properties: { symbol: { type: "string" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "get_gsri_status", description: "Get the GSRI risk snapshot.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "get_systems_status", description: "Get overall server system status.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "wait_and_verify", description: "Wait a moment before verifying state (use after actions that take time).", parameters: { type: "object", properties: { delay_ms: { type: "number" } } } } },
  { type: "function", function: { name: "assert_text_visible", description: "Verify that text is visible on the last screen state.", parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } } },
  { type: "function", function: { name: "get_frit_manual", description: "Get a full reference of every real tool FRIT has, grouped by category, plus how to use the phone's installed-apps list correctly. Call this only if you're unsure what capabilities you have — don't call it for every task.", parameters: { type: "object", properties: {} } } },
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
    "- 'open_app' only works reliably for apps present in the device state's 'Installed apps' list — check that list before assuming an app exists.",
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
    case "analyze_market": return { ok: true, data: await analyzeSymbol(args.symbol, args.interval, args.outputsize) };
    case "run_code": return { ok: true, data: await runSandbox({ language: args.language, code: args.code, stdin: args.stdin || "", timeout_ms: args.timeout_ms || 8000 }) };
    case "get_frit_manual": return { ok: true, data: buildFritManual() };
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
    "1. OBSERVE: Use 'read_screen' or 'read_screen_structured' to see what's on screen.",
    "2. ANALYZE: If you don't see what you need, ANALYZE why. Maybe the app isn't open? Maybe you need to scroll?",
    "3. ACT: Decide on ONE next step (tap, type, scroll, go_back, press_home).",
    "4. VERIFY: Immediately call 'read_screen' again to see the result of your action. Did it work? If not, self-correct.",
    "",
    "HARD RULES — VIOLATING THESE IS A FAILURE:",
    "- NEVER write a fake action as plain text or a markdown code block (e.g. 'Action: ```python get_market_data(...)```'). That is not a real tool call and does NOTHING. If you need a tool, you MUST use the actual function-calling mechanism provided to you — never describe, narrate, or pretend to call a tool in prose.",
    "- If you genuinely cannot call a tool (none fits), say so directly in one sentence. Do not paste code for the user to run manually as a substitute for calling 'run_code' yourself — you have 'run_code', use it.",
    "- Only call device-control or task tools when the user's message clearly asks for an action. For greetings, small talk, or questions that need no device interaction, reply in plain conversational text with ZERO tool calls. Do not invent a task out of a greeting like 'hi'.",
    "- For 'open_app': the 'app_name' argument must be ONLY the literal app name (e.g. 'WhatsApp', 'Messenger') — never a sentence, instruction, or task description. Open the app first, THEN use separate tool calls (read_screen, tap, type) to carry out the actual task once it's open.",
    "- If the device state below lists 'Installed apps', ONLY target names from that list with open_app — do not guess an app exists if it isn't listed. If it's not there, tell the user instead of trying anyway.",
    "- You only have the tools explicitly provided to you in this request (open_app, read_screen, tap_button, type_text, run_code, search_web, get_market_data, analyze_market, send_whatsapp, make_call, etc.). Never assume a capability exists beyond that list — e.g. there is no generic 'send_message' or 'call_contact' tool, use the exact tool names you were given.",
    "",
    "TIPS FOR FULL AUTONOMY:",
    "- To open any app: If not visible, press_home -> click search bar or use 'open_app'.",
    "- To find a specific button: Use 'read_screen_structured' to get exact coordinates if text-matching fails.",
    "- If a tool fails: Don't give up. Try a different approach (e.g., tap_coordinates instead of tap_button).",
    "- Run Code: Use 'run_code' for complex logic, math, or data processing. Don't guess calculations.",
    "- Browse: Use 'search_web' to find information.",
    "- Market/trading tasks: analysis happens HERE on the server, NOT on the phone. Actually CALL the 'analyze_market' or 'get_market_data' tool (a real function call) and read the returned direction/entry/SL/TP — do not narrate calling it. Only use the phone (open_app MetaTrader5, tap, type) to EXECUTE an order after the analysis is complete.",
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
  if (d.battery_level !== undefined) parts.push(`Battery: ${d.battery_level}%`);
  if (d.keyboard_open !== undefined) parts.push(`Keyboard open: ${d.keyboard_open}`);
  if (Array.isArray(d.installed_apps) && d.installed_apps.length) {
    // Ground open_app in reality: only these names are guaranteed to exist.
    // Capped to keep prompt size sane on large phones (150+ apps is common).
    const names = d.installed_apps.map(a => (typeof a === "string" ? a : a.name)).filter(Boolean);
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
  const names = list.map(a => (typeof a === "string" ? a : a.name)).filter(Boolean);
  const target = requestedName.trim().toLowerCase();
  const exact = names.find(n => n.toLowerCase() === target);
  if (exact) return exact;
  const contains = names.find(n => n.toLowerCase().includes(target) || target.includes(n.toLowerCase()));
  return contains || null;
}

// ==================== SCREEN FRAME INGESTION ====================
const frameBuffer = [];
const FRAME_BUFFER_SIZE = 10;

// ==================== ROUTES ====================
app.get("/", (_req, res) => {
  res.json({
    name: "FRIT SMC+CRT Trading Engine & Mistral AI Orchestrator",
    description: "Pure SMC+CRT + Volume Profile + GSRI — no indicators. Single Mistral API ecosystem.",
    status: "online",
    endpoints: {
      core: ["/health", "/agent/start", "/agent/resume", "/agent/status"],
      market: ["/market/quote", "/market/batch", "/market/analyze", "/trade", "/gsri/status"],
      positions: ["/positions", "/positions/status", "/positions/resolve"],
      scanner: ["/scanner/scan", "/scanner/control", "/scanner/status", "/smc/status"],
      memory: ["/memory/trade"],
      screen: ["/screen/frame", "/screen/analyze-frame", "/screen/status"],
      transcribe: ["/transcribe"],
      tools: ["/tools/search", "/acp/status"],
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
      const out = await mistralChat({
        model: MODELS.vision,
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
    const out = await mistralChat({
      model: MODELS.vision,
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

// ==================== TRADE ENDPOINT ====================
app.post("/trade", requireAuth, async (req, res) => {
  const { symbol, action, risk_percent = 1, balance, reason = "", interval = "1h", pipeline = "original" } = req.body || {};
  if (!symbol || !action) return res.status(400).json({ error: "symbol and action required" });
  if (!["buy", "sell"].includes(action)) return res.status(400).json({ error: "action must be buy or sell" });

  if (pipeline === "enhanced") {
    try {
      const strategy = String(req.body?.strategy || "mtf").toLowerCase();
      const result = strategy === "smc_crt"
        ? await frit.analyze(symbol, interval, { balance: balance || 1000, riskPercent: risk_percent })
        : await mtfStrategy.run(symbol, { interval, balance: balance || 1000, riskPercent: risk_percent });

      if (["NO_TRADE", "WAIT", "COOLDOWN", "DATA_UNAVAILABLE", "DATA_RATE_LIMITED", "ERROR"].includes(result.decision)) {
        return res.status(200).json({ status: "blocked", ...result });
      }
      const tradeResult = await sendToMT5Bridge({
        symbol: symbol.toUpperCase(),
        action: result.decision === "BUY" ? "buy" : "sell",
        lotSize: result.lot_size,
        entry: result.entry,
        sl: result.sl,
        tp: result.tp,
        reason: reason || `MTF v1 pipeline: conf=${result.confidence}% regime=${result.regime?.trend} struct=${result.structure?.trend} zone=${result.entry_ctx?.zone}`,
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
            source: strategy === "smc_crt" ? "smc_crt_pipeline" : "mtf_v1",
            reason: reason || `conf=${result.confidence}%`,
          });
        } catch (e) { console.warn("[PositionMonitor] /trade register failed:", e.message); }
      }

      return res.json({
        status: "submitted",
        pipeline: strategy === "smc_crt" ? "smc_crt" : "mtf_v1",
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
        structure: result.structure,
        entry_ctx: result.entry_ctx,
        guards: result.guards,
        mt5_result: tradeResult,
        position_id: position?.id ?? null,
      });
    } catch (err) {
      console.error("[/trade enhanced]", err.message);
      return res.status(500).json({ error: "Enhanced trade failed", details: err.message });
    }
  }

  if (!reason) return res.status(400).json({ error: "reason required — AI must justify every trade" });

  try {
    const gsriSnap = await getGsriSnapshot();
    const gsriScore = parseFloat(gsriSnap?.Risk_Score ?? 0.8);
    const gsriAlert = parseInt(gsriSnap?.Alert ?? 1);
    const gsriScale = gsriLotScale(gsriScore);
    const gsriDate = gsriSnap?.Date ?? "unknown";
    if (gsriAlert === 1) {
      return res.status(200).json({
        status: "blocked_by_gsri",
        reason: `GSRI Alert active — Risk_Score=${gsriScore.toFixed(3)}, date=${gsriDate}. New entries blocked during elevated systemic risk.`,
        gsri: { score: gsriScore, alert: gsriAlert, scale: 0, date: gsriDate, source: gsriSnap?.source || "file" },
      });
    }

    const analysis = await analyzeSymbol(symbol, interval);
    if (analysis.news_filter?.blocked) return res.status(200).json({ status: "blocked", reason: analysis.ai_opinion });

    const entry = parseFloat(analysis.trade_plan?.entry_zone?.split("-")[0]) || analysis.price;
    const sl = parseFloat(analysis.trade_plan?.invalidation) || 0;
    const tp = parseFloat(analysis.trade_plan?.tp1) || 0;
    if (!sl) return res.status(400).json({ error: "Could not determine stop loss from analysis" });

    const accountBalance = balance || 1000;
    const rawLot = calculateLotSize({ symbol, balance: accountBalance, riskPercent: risk_percent, entry, stopLoss: sl });
    const lotSize = parseFloat((rawLot * gsriScale).toFixed(2));

    const tradeResult = await sendToMT5Bridge({
      symbol: symbol.toUpperCase(),
      action,
      lotSize,
      entry,
      sl,
      tp,
      reason,
    });

    addTradeMemory(symbol, { direction: action, pattern: analysis.patterns?.join(", ") || "none", outcome: "pending", note: reason });

    // Additive: watch this position until TP/SL is hit.
    let position = null;
    if (tradeResult.status !== "failed" && sl && tp) {
      try {
        position = positionMonitor.register({
          symbol: symbol.toUpperCase(),
          action,
          lotSize,
          entry,
          sl,
          tp,
          source: "trade",
          reason,
        });
      } catch (e) { console.warn("[PositionMonitor] /trade register failed:", e.message); }
    }

    res.json({
      status: "submitted",
      symbol: symbol.toUpperCase(),
      action,
      lotSize,
      raw_lot: rawLot,
      entry,
      sl,
      tp,
      risk_percent,
      balance_used: accountBalance,
      reason,
      mt5_result: tradeResult,
      analysis_confidence: analysis.confidence,
      mtf_note: analysis.mtf?.note,
      gsri: { score: gsriScore, alert: gsriAlert, scale: gsriScale, date: gsriDate },
      position_id: position?.id ?? null,
    });
  } catch (err) {
    console.error("[/trade]", err.message);
    res.status(500).json({ error: "Trade failed", details: err.message });
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

app.get("/gsri/status", requireAuth, async (req, res) => {
  try {
    const snap = await getGsriSnapshot();
    const score = parseFloat(snap?.Risk_Score ?? 0.8);
    res.json({
      snapshot: snap,
      lot_scale: gsriLotScale(score),
      entries_allowed: parseInt(snap?.Alert ?? 1) === 0,
    });
  } catch (err) {
    res.status(500).json({ error: "GSRI status failed", details: err.message });
  }
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

app.post("/transcribe", async (req, res) => {
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

// ==================== WEB SEARCH & CONVICTION PROXY ====================
// /tools/search — raw web-search endpoint used by the agent brain (search_web tool)
// and by any client that wants real-time info without going through the LLM.
app.post("/tools/search", requireAuth, async (req, res) => {
  const { query } = req.body || {};
  if (!query) return res.status(400).json({ error: "query required" });
  try {
    const data = await webSearch(String(query));
    res.json({ ok: true, query, data });
  } catch (err) {
    console.error("[/tools/search]", err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// /acp/status — Automated Conviction Proxy: a lightweight composite of every
// signal the server already computes for a symbol (engine analysis + paper
// track record + crash regime). Lets the brain/UI gauge conviction cheaply.
app.get("/acp/status", requireAuth, async (req, res) => {
  const sym = String(req.query.symbol || "XAUUSD").toUpperCase();
  try {
    const analysis = await analyzeSymbol(sym, "1h", null);
    const stats = frit.paperLogger.getStats(sym);
    const gsri = frit.crashGSRI.getCached() || {};
    const smc = frit.scanner.getLatestScan()?.results?.[sym] || null;
    res.json({
      symbol: sym,
      direction: analysis.direction || "UNKNOWN",
      confidence: analysis.confidence ?? null,
      price: analysis.price ?? null,
      strength: analysis.strength ?? null,
      smc_signal: analysis.smc_crt?.signal ?? (smc ? smc.signal : null),
      smc_confidence: smc ? smc.confidence : null,
      paper_trades: stats ? { wins: stats.wins, losses: stats.losses, pending: stats.pending, win_rate: stats.win_rate } : null,
      crash_regime: gsri.phase || "unknown",
      crash_risk_score: gsri.risk_score ?? null,
      timestamp: Date.now(),
    });
  } catch (err) {
    console.error("[/acp/status]", err.message);
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ==================== ENHANCED PIPELINE ROUTES ====================
app.post("/enhanced/analyze", requireAuth, async (req, res) => {
  const { symbol, interval = "1h", balance, riskPercent, strategy } = req.body || {};
  if (!symbol) return res.status(400).json({ error: "symbol required" });
  try {
    const useMtf = String(strategy || "mtf").toLowerCase() !== "smc_crt";
    const result = useMtf
      ? await mtfStrategy.analyze(symbol, { interval, balance, riskPercent })
      : await frit.analyze(symbol, interval, { balance, riskPercent });
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

app.post("/scanner/scan", requireAuth, async (req, res) => {
  const { interval = "1h" } = req.body || {};
  try {
    const result = await frit.scan(interval);
    res.json(result);
  } catch (err) {
    console.error("[/scanner/scan]", err.message);
    res.status(500).json({ error: "Scan failed", details: err.message });
  }
});

app.post("/scanner/control", requireAuth, (req, res) => {
  const { action, interval_ms } = req.body || {};
  if (action === "start") {
    const result = frit.startScanner(interval_ms || 120000);
    res.json(result);
  } else if (action === "stop") {
    const result = frit.stopScanner();
    res.json(result);
  } else {
    res.status(400).json({ error: "action must be 'start' or 'stop'" });
  }
});

app.get("/scanner/status", requireAuth, (_req, res) => {
  res.json(frit.getStatus());
});

app.get("/crash-gsri/status", requireAuth, async (_req, res) => {
  try {
    const metrics = await frit.crashGSRI.compute(fetchCandles);
    res.json(metrics);
  } catch (err) {
    console.error("[/crash-gsri/status]", err.message);
    res.status(500).json({ error: "Crash GSRI failed", details: err.message });
  }
});

app.post("/crash-gsri/recompute", requireAuth, async (_req, res) => {
  try {
    frit.crashGSRI.lastComputeTime = 0;
    frit.crashGSRI.lastResult = null;
    const metrics = await frit.crashGSRI.compute(fetchCandles);
    res.json(metrics);
  } catch (err) {
    console.error("[/crash-gsri/recompute]", err.message);
    res.status(500).json({ error: "Crash GSRI recompute failed", details: err.message });
  }
});

app.get("/smc/status", requireAuth, (req, res) => {
  const symbol = String(req.query.symbol || "").toUpperCase();
  if (!symbol) return res.status(400).json({ error: "symbol query param required" });
  const scan = frit.scanner.getLatestScan();
  const symbolResult = scan?.results?.[symbol] || null;
  const paperStats = frit.paperLogger.getStats(symbol);
  res.json({ symbol, smc_scan: symbolResult, paper_trade_stats: paperStats });
});

app.get("/paper-trades", requireAuth, (req, res) => {
  const symbol = String(req.query.symbol || "").toUpperCase();
  const limit = parseInt(req.query.limit) || 20;
  if (symbol) {
    res.json({ symbol, trades: frit.paperLogger.getTrades(symbol, limit) });
  } else {
    const all = {};
    for (const sym of frit.paperLogger.trades.keys()) {
      all[sym] = frit.paperLogger.getTrades(sym, 10);
    }
    res.json({ trades: all });
  }
});

app.post("/paper-trades/resolve", requireAuth, (req, res) => {
  const { trade_id, outcome, close_price } = req.body || {};
  if (!trade_id || !outcome) return res.status(400).json({ error: "trade_id and outcome required" });
  if (!["win", "loss", "breakeven"].includes(outcome)) return res.status(400).json({ error: "outcome must be win/loss/breakeven" });
  const result = frit.paperLogger.resolve(trade_id, outcome, close_price);
  if (result) res.json({ ok: true, trade: result });
  else res.status(404).json({ error: "Trade not found" });
});

app.get("/systems/status", requireAuth, (_req, res) => {
  res.json(frit.getStatus());
});

app.get("/pipeline/history", requireAuth, (req, res) => {
  const limit = parseInt(req.query.limit) || 20;
  res.json({ history: frit.pipeline.getHistory(limit) });
});

// ====== ERROR HANDLER ======
app.use((err, _req, res, _next) => {
  console.error("Unhandled Error!", err);
  res.status(500).json({ error: "Internal server error", details: err?.message || "Unknown error" });
});

// ====== START ======
app.listen(PORT, () => {
  console.log(`
FRIT - SMC+CRT Trading Engine & Mistral AI Orchestrator
Port : ${String(PORT).padEnd(5)}
Pure SMC+CRT (no indicator lag):
 - Candle Range Theory (CRT)
 - Smart Money Concepts (OB, FVG, CHoCH, Liquidity)
 - Market Structure (HH/HL, LH/LL, BOS)
 - Volume Profile (POC, VAH, VAL)
 - GSRI systemic risk overlay
 - News filter (FF calendar — 30 min blackout)
 - Multi-timeframe confirmation (4H swing structure)
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