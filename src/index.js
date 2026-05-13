import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import dotenv from "dotenv";
import fetch from "node-fetch";
import FormData from "form-data";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

dotenv.config();
const app = express();
const PORT = Number(process.env.PORT || 8787);
const __dirname = dirname(fileURLToPath(import.meta.url));

// ==================== ENVIRONMENT ====================
const GROQ_API_KEY    = process.env.GROQ_API_KEY;
const TWELVE_DATA_KEY = process.env.TWELVE_DATA_KEY || "";
const SANDBOX_URL     = process.env.SANDBOX_URL || "https://sandbox-production-839a.up.railway.app";
const AUTH_TOKEN      = process.env.AUTH_TOKEN || "";
const MT5_BRIDGE_URL  = process.env.MT5_BRIDGE_URL || "";

if (!GROQ_API_KEY) { console.error("[FATAL] GROQ_API_KEY missing"); process.exit(1); }
if (!AUTH_TOKEN) { console.warn("[WARN] AUTH_TOKEN not set — endpoints unprotected!"); }

// ==================== MIDDLEWARE ====================
function requireAuth(req, res, next) {
  if (!AUTH_TOKEN) return next();
  const header = req.headers["authorization"] || "";
  const token  = header.startsWith("Bearer ") ? header.slice(7) : "";
  if (token !== AUTH_TOKEN) return res.status(401).json({ error: "Unauthorized" });
  next();
}

// RATE LIMITER (Recommendation #3)
const rateLimiter = new Map();
function applyRateLimit(maxCalls = 5, windowMs = 60000) {
  return (req, res, next) => {
    const ip = req.ip || req.connection.remoteAddress;
    const now = Date.now();
    const bucket = rateLimiter.get(ip) || { calls: [], resetAt: now + windowMs };
    bucket.calls = bucket.calls.filter(t => t > now);
    if (bucket.calls.length >= maxCalls) return res.status(429).json({ error: "Rate limit exceeded. Try again later." });
    bucket.calls.push(now);
    rateLimiter.set(ip, bucket);
    next();
  };
}

app.set("trust proxy", 1);
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ limit: "10mb", extended: true }));
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: "*", methods: ["GET", "POST", "OPTIONS"] }));
app.use(morgan(process.env.NODE_ENV === "production" ? "combined" : "dev"));

// ==================== MODELS ====================
const MODELS = {
  vision:       "meta-llama/llama-4-scout-17b-16e-instruct",
  agent:        "llama-3.3-70b-versatile",
  conversation: "llama-3.3-70b-versatile",
  tools:        "llama-3.3-70b-versatile",
  fast:         "llama-3.1-8b-instant",
  whisper:      "whisper-large-v3-turbo",
};
function pickModel({ hasImage = false, mode = "auto", taskType = "general" } = {}) {
  if (mode === "vision" || hasImage) return MODELS.vision;
  if (mode === "fast")              return MODELS.fast;
  if (mode === "tools")             return MODELS.tools;
  if (mode === "agent")             return MODELS.agent;
  if (mode === "auto") {
    const complex = ["automation", "planning", "analysis", "trading", "multistep"];
    const simple  = ["chat", "greeting", "quick_question"];
    if (complex.includes(taskType)) return MODELS.agent;
    if (simple.includes(taskType))  return MODELS.fast;
  }
  return MODELS.conversation;
}

// ==================== IN-MEMORY CACHE ====================
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
  if (Date.now() > entry.exp) { _cache.delete(key); return null; }
  return entry.val;
}
function cacheSet(key, val, ttlMs = 60000) { _cache.set(key, { val, exp: Date.now() + ttlMs }); }

// ==================== HELPERS ====================
function safeJsonParse(input, fallback = null) { try { return JSON.parse(input); } catch { return fallback; } }
function truncateText(text, maxLen = 1200) { const s = String(text || " "); return s.length > maxLen ? s.slice(0, maxLen) : s; }
function summarizeMemory(memory = [], maxItems = 6, maxChars = 700) {
  if (!Array.isArray(memory) || !memory.length) return [];
  const picked = []; let used = 0;
  for (const item of memory) {
    const s = truncateText(item, 180);
    if (!s || picked.length >= maxItems || used + s.length > maxChars) break;
    picked.push(s); used += s.length;
  }
  return picked;
}
function trimHistoryByChars(history = [], maxMessages = 8, maxChars = 3200) {
  const arr = Array.isArray(history) ? history.slice(-maxMessages) : [];
  const out = []; let used = 0;
  for (let i = arr.length - 1; i >= 0; i--) {
    const msg = arr[i]; let content = msg?.content ?? " ";
    if (Array.isArray(content)) content = content.map(p => typeof p?.text === "string" ? p.text : " ").join("  ");
    const s = truncateText(String(content), 900); const cost = s.length + 40;
    if (used + cost > maxChars) continue;
    out.unshift({ role: msg.role, content: s }); used += cost;
  }
  return out;
}
function maybeTrimScreenContext(ctx, maxChars = 350000) { return (!ctx || typeof ctx !== "string") ? null : (ctx.length <= maxChars ? ctx : null); }
function buildMemoryBlock(memory = []) {
  const compact = summarizeMemory(memory, 6, 700);
  return compact.length ? `User memory:\n${compact.join("\n")}` : "";
}

// ==================== GROQ API ====================
async function groqChat({ model, messages, tools = null, temperature = 0.3, max_tokens = 1600, tool_choice = "auto", retries = 3 }) {
  const body = { model, messages, temperature, max_tokens };
  if (tools?.length) { body.tools = tools; body.tool_choice = tool_choice; }
  let lastError;
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST", headers: { Authorization: `Bearer ${GROQ_API_KEY}`, "Content-Type": "application/json" }, body: JSON.stringify(body)
      });
      const data = await res.json();
      if (!res.ok) { lastError = new Error(data?.error?.message || JSON.stringify(data)); if (res.status === 429) { await new Promise(r => setTimeout(r, attempt * 2000)); continue; } continue; }
      return data;
    } catch (err) { lastError = err; if (attempt < retries) await new Promise(r => setTimeout(r, attempt * 1000)); }
  }
  throw lastError || new Error("Groq API failed after retries");
}
function extractToolCalls(msg) {
  return (msg?.tool_calls || []).map(tc => ({ id: tc.id, type: tc.type, function: { name: tc.function?.name, arguments: safeJsonParse(tc.function?.arguments || "{}", {}), raw_arguments: tc.function?.arguments || "{}" } }));
}

// ==================== SANDBOX ====================
async function runSandbox(args = {}) {
  const res = await fetch(`${SANDBOX_URL}/sandbox/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(args) });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.details || data?.error || "Sandbox failed");
  return data;
}

// ==================== MARKET DATA ====================
function normalizeInterval(v) {
  const iv = String(v || "1h").toLowerCase().trim();
  const allowed = ["1min", "5min", "15min", "30min", "1h", "4h", "1day", "1week"];
  return allowed.includes(iv) ? iv : "1h";
}
function resolveOutputSize(interval) { return ({ "1min": 500, "5min": 288, "15min": 200, "30min": 150, "1h": 168, "4h": 120, "1day": 100, "1week": 52 }[interval] || 168); }
function toBinanceInterval(interval) { return ({ "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "1h", "4h": "4h", "1day": "1d", "1week": "1w" }[interval] || "1h"); }

const TD_SYMBOLS = { EURUSD: "EUR/USD", GBPUSD: "GBP/USD", USDJPY: "USD/JPY", AUDUSD: "AUD/USD", USDCHF: "USD/CHF", USDCAD: "USD/CAD", NZDUSD: "NZD/USD", XAUUSD: "XAU/USD", XAGUSD: "XAG/USD", GBPJPY: "GBP/JPY", EURJPY: "EUR/JPY", EURGBP: "EUR/GBP", BTC: "BTC/USD", ETH: "ETH/USD", SOL: "SOL/USD", BNB: "BNB/USD", XRP: "XRP/USD", DOGE: "DOGE/USD", ADA: "ADA/USD", BTCUSD: "BTC/USD", ETHUSD: "ETH/USD", SOLUSD: "SOL/USD", BNBUSD: "BNB/USD", XRPUSD: "XRP/USD", DOGEUSD: "DOGE/USD", ADAUSD: "ADA/USD" };
const CRYPTO_SET = new Set(["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD", "DOGEUSD", "ADAUSD"]);
const BINANCE_SYM = { BTC: "BTCUSDT", ETH: "ETHUSDT", SOL: "SOLUSDT", BNB: "BNBUSDT", XRP: "XRPUSDT", DOGE: "DOGEUSDT", ADA: "ADAUSDT", BTCUSD: "BTCUSDT", ETHUSD: "ETHUSDT", SOLUSD: "SOLUSDT", BNBUSD: "BNBUSDT", XRPUSD: "XRPUSDT", DOGEUSD: "DOGEUSDT", ADAUSD: "ADAUSDT" };
const COINGECKO_IDS = { BTC: "bitcoin", ETH: "ethereum", SOL: "solana", BNB: "binancecoin", XRP: "ripple", DOGE: "dogecoin", ADA: "cardano", BTCUSD: "bitcoin", ETHUSD: "ethereum", SOLUSD: "solana", BNBUSD: "binancecoin", XRPUSD: "ripple", DOGEUSD: "dogecoin", ADAUSD: "cardano" };
const FRANKFURTER_MAP = { EURUSD: { base: "EUR", quote: "USD" }, GBPUSD: { base: "GBP", quote: "USD" }, USDJPY: { base: "USD", quote: "JPY" }, AUDUSD: { base: "AUD", quote: "USD" }, USDCHF: { base: "USD", quote: "CHF" }, USDCAD: { base: "USD", quote: "CAD" }, NZDUSD: { base: "NZD", quote: "USD" }, EURGBP: { base: "EUR", quote: "GBP" }, EURJPY: { base: "EUR", quote: "JPY" }, GBPJPY: { base: "GBP", quote: "JPY" } };

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
  const sym = String(symbol || " ").toUpperCase();
  const iv = normalizeInterval(interval);
  const size = outputsize || resolveOutputSize(iv);
  const ck = `candles:${sym}:${iv}:${size}`;
  const cached = cacheGet(ck);
  if (cached) return cached;
  if (TWELVE_DATA_KEY && TD_SYMBOLS[sym]) {
    try {
      const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&interval=${iv}&outputsize=${size}&apikey=${TWELVE_DATA_KEY}`;
      const res = await fetch(url); const data = await res.json();
      if (data.status !== "error" && data.values?.length >= 10) {
        const candles = normalizeCandles(data.values.reverse(), "twelve_data");
        cacheSet(ck, candles, 60000); return candles;
      }
    } catch (e) { console.error("[TwelveData candles]", sym, e.message); }
  }
  if (BINANCE_SYM[sym]) {
    try {
      const url = `https://api.binance.com/api/v3/klines?symbol=${BINANCE_SYM[sym]}&interval=${toBinanceInterval(iv)}&limit=${Math.min(size, 1000)}`;
      const res = await fetch(url); const arr = await res.json();
      if (Array.isArray(arr) && arr.length >= 10) {
        const candles = normalizeCandles(arr, "binance");
        cacheSet(ck, candles, 60000); return candles;
      }
    } catch (e) { console.error("[Binance candles]", sym, e.message); }
  }
  return null;
}

async function fetchSpotPrice(symbol) {
  const sym = String(symbol || " ").toUpperCase(); const ck = `spot:${sym}`; const cached = cacheGet(ck);
  if (cached) return cached; let result = null;
  if (TWELVE_DATA_KEY && TD_SYMBOLS[sym]) {
    try { const res = await fetch(`https://api.twelvedata.com/price?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&apikey=${TWELVE_DATA_KEY}`); const data = await res.json(); if (data.price) result = { price: parseFloat(data.price), source: "twelvedata" }; } catch (e) { console.error("[TwelveData spot]", sym, e.message); }
  }
  if (!result && COINGECKO_IDS[sym]) {
    try { const res = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${COINGECKO_IDS[sym]}&vs_currencies=usd&include_24hr_change=true`); const data = await res.json(); const id = COINGECKO_IDS[sym]; if (data[id]) result = { price: data[id].usd, change24h: data[id].usd_24h_change, source: "coingecko" }; } catch (e) { console.error("[CoinGecko spot]", sym, e.message); }
  }
  if (!result && FRANKFURTER_MAP[sym]) {
    try { const { base, quote } = FRANKFURTER_MAP[sym]; const res = await fetch(`https://api.frankfurter.app/latest?from=${base}&to=${quote}`); const data = await res.json(); const rate = data.rates?.[quote]; if (rate) result = { price: parseFloat(rate), source: "frankfurter" }; } catch (e) { console.error("[Frankfurter]", sym, e.message); }
  }
  if (result) cacheSet(ck, result, 30000);
  return result;
}

async function fetch24hDelta(symbol) {
  const sym = String(symbol || "").toUpperCase(); if (CRYPTO_SET.has(sym)) return null; if (!TWELVE_DATA_KEY || !TD_SYMBOLS[sym]) return null;
  const ck = `delta:${sym}`; const cached = cacheGet(ck); if (cached !== null) return cached;
  try {
    const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&interval=1day&outputsize=2&apikey=${TWELVE_DATA_KEY}`;
    const res = await fetch(url); const data = await res.json();
    if (data.values?.length >= 2) { const today = parseFloat(data.values[0].close); const yest = parseFloat(data.values[1].close); const delta = ((today - yest) / yest) * 100; cacheSet(ck, delta, 300000); return delta; }
  } catch (e) { console.error("[fetch24hDelta]", sym, e.message); }
  return null;
}

async function fetchMarketPrices(symbols = []) {
  const result = {};
  await Promise.all(symbols.map(async sym => {
    const s = String(sym || "  ").toUpperCase(); const spot = await fetchSpotPrice(s);
    if (!spot) { result[s] = { symbol: s, price: 0, change24h: 0, currency: "USD", error: "Not found" }; return; }
    const change24h = spot.change24h !== undefined && spot.change24h !== 0 ? spot.change24h : (await fetch24hDelta(s)) ?? 0;
    result[s] = { symbol: s, price: spot.price, change24h, currency: s.includes("NGN") ? "NGN" : "USD", source: spot.source };
  }));
  return result;
}

// ==================== TECHNICAL INDICATORS ====================
function calcEMA(closes, period) {
  if (closes.length < period) return [];
  const k = 2 / (period + 1); let prev = closes.slice(0, period).reduce((a, b) => a + b, 0) / period; const result = [prev];
  for (let i = period; i < closes.length; i++) { prev = closes[i] * k + prev * (1 - k); result.push(prev); }
  return result;
}
function calcRSI(closes, period = 14) {
  if (closes.length < period + 1) return 50;
  const changes = closes.slice(-(period + 1)).map((v, i, a) => i > 0 ? v - a[i - 1] : 0).slice(1);
  const avgGain = changes.map(c => c > 0 ? c : 0).reduce((a, b) => a + b, 0) / period;
  const avgLoss = changes.map(c => c < 0 ? -c : 0).reduce((a, b) => a + b, 0) / period;
  if (avgLoss === 0) return 100; return 100 - 100 / (1 + avgGain / avgLoss);
}
function calcMACD(closes) {
  const ema12 = calcEMA(closes, 12); const ema26 = calcEMA(closes, 26);
  if (!ema12.length || !ema26.length) return { macd: 0, signal: 0, hist: 0 };
  const offset = ema12.length - ema26.length; const macdLine = ema26.map((v, i) => ema12[i + offset] - v);
  const sigLine = calcEMA(macdLine, 9); const last = macdLine.at(-1); const sig = sigLine.at(-1) ?? 0;
  return { macd: last, signal: sig, hist: last - sig };
}
function calcBB(closes, period = 20, mult = 2) {
  if (closes.length < period) return null;
  const s = closes.slice(-period); const mean = s.reduce((a, b) => a + b, 0) / period;
  const std = Math.sqrt(s.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / period);
  return { upper: mean + mult * std, middle: mean, lower: mean - mult * std };
}
function calcATR(candles, period = 14) {
  if (!candles || candles.length < period + 1) return 0;
  const trs = [];
  for (let i = 1; i < candles.length; i++) { const c = candles[i]; const p = candles[i - 1]; trs.push(Math.max(c.high - c.low, Math.abs(c.high - p.close), Math.abs(c.low - p.close))); }
  const recent = trs.slice(-period); return recent.reduce((a, b) => a + b, 0) / recent.length;
}
function calcSR(candles) {
  const window = candles.slice(-50); const swings = findSwings(window, 3); const currentPrice = candles.at(-1)?.close ?? 0;
  const support = swings.lows.map(s => s.price).filter(p => p < currentPrice).sort((a, b) => b - a)[0] ?? Math.min(...window.map(c => c.low));
  const resistance = swings.highs.map(s => s.price).filter(p => p > currentPrice).sort((a, b) => a - b)[0] ?? Math.max(...window.map(c => c.high));
  return { support, resistance };
}
function findSwings(candles, lookback = 2) {
  const highs = []; const lows = [];
  for (let i = lookback; i < candles.length - lookback; i++) {
    const cur = candles[i]; let isHigh = true; let isLow = true;
    for (let j = i - lookback; j <= i + lookback; j++) { if (j === i) continue; if (candles[j].high >= cur.high) isHigh = false; if (candles[j].low <= cur.low) isLow = false; }
    if (isHigh) highs.push({ index: i, price: cur.high, time: cur.time }); if (isLow) lows.push({ index: i, price: cur.low, time: cur.time });
  }
  return { highs, lows };
}
function detectCandlePattern(candles) {
  if (!candles || candles.length < 2) return [];
  const a = candles[candles.length - 2]; const b = candles[candles.length - 1]; const patterns = [];
  const aBear = a.close < a.open; const aBull = a.close > a.open; const bBull = b.close > b.open; const bBear = b.close < b.open;
  if (aBear && bBull && b.open <= a.close && b.close >= a.open) patterns.push("bullish_engulfing");
  if (aBull && bBear && b.open >= a.close && b.close <= a.open) patterns.push("bearish_engulfing");
  const body = Math.abs(b.close - b.open); const upperWick = b.high - Math.max(b.open, b.close); const lowerWick = Math.min(b.open, b.close) - b.low;
  if (body > 0) {
    if (lowerWick > body * 2 && upperWick < body) patterns.push("pinbar_bullish");
    if (upperWick > body * 2 && lowerWick < body) patterns.push("pinbar_bearish");
    if (body < (b.high - b.low) * 0.35) patterns.push("indecision");
  }
  return patterns;
}

// ==================== VOLUME PROFILE & STRUCTURE ====================
function calcAuction(candles, buckets = 40) {
  if (!candles || candles.length < 10) return null;
  const window = candles.slice(-100); const totalRawVol = window.reduce((s, c) => s + (c.volume || 0), 0); const hasRealVol = totalRawVol > window.length * 2;
  const hi = Math.max(...window.map(c => c.high)); const lo = Math.min(...window.map(c => c.low)); if (hi === lo) return null;
  const bucketSize = (hi - lo) / buckets; const vap = new Array(buckets).fill(0);
  for (const c of window) {
    const rangeProxy = c.high - c.low || bucketSize; const bodySize = Math.abs(c.close - c.open) || rangeProxy * 0.3;
    const vol = hasRealVol ? (c.volume > 0 ? c.volume : rangeProxy) : rangeProxy * (1 + bodySize / rangeProxy);
    const candleRange = c.high - c.low || bucketSize;
    for (let b = 0; b < buckets; b++) { const bLo = lo + b * bucketSize; const bHi = bLo + bucketSize; const overlap = Math.max(0, Math.min(c.high, bHi) - Math.max(c.low, bLo)); vap[b] += vol * (overlap / candleRange); }
  }
  const pocIdx = vap.indexOf(Math.max(...vap)); const poc = lo + (pocIdx + 0.5) * bucketSize; const totalVol = vap.reduce((a, b) => a + b, 0); const target = totalVol * 0.7;
  let lo_idx = pocIdx, hi_idx = pocIdx, accumulated = vap[pocIdx];
  while (accumulated < target) { const addLo = lo_idx > 0 ? vap[lo_idx - 1] : 0; const addHi = hi_idx < buckets - 1 ? vap[hi_idx + 1] : 0; if (addLo === 0 && addHi === 0) break; if (addHi >= addLo) { hi_idx++; accumulated += addHi; } else { lo_idx--; accumulated += addLo; } }
  const vah = lo + (hi_idx + 1) * bucketSize; const val = lo + lo_idx * bucketSize;
  const sorted = vap.map((v, i) => ({ v, price: lo + (i + 0.5) * bucketSize })).sort((a, b) => b.v - a.v);
  return { poc, vah, val, hvn: sorted.slice(0, 3).map(x => x.price), lvn: sorted.slice(-5).map(x => x.price), range_hi: hi, range_lo: lo, volume_mode: hasRealVol ? "real" : "proxy" };
}
function auctionSignal(price, auction) {
  if (!auction) return { position: "unknown", bias: "neutral", note: "" };
  const { poc, vah, val } = auction;
  if (price > vah) return { position: "above_value", bias: "bullish", note: "Price above Value Area — buyers in control." };
  if (price < val) return { position: "below_value", bias: "bearish", note: "Price below Value Area — sellers in control." };
  if (price > poc) return { position: "inside_value_upper", bias: "mild_bullish", note: "Inside Value Area above POC — mean reversion risk, watch VAH." };
  return { position: "inside_value_lower", bias: "mild_bearish", note: "Inside Value Area below POC — mean reversion risk, watch VAL." };
}
function analyzeStructure(candles, price) {
  const closes = candles.map(c => c.close); const ema20 = calcEMA(closes, 20).at(-1) ?? price; const ema50 = calcEMA(closes, 50).at(-1) ?? price; const ema200 = calcEMA(closes, 200).at(-1) ?? price;
  const swings = findSwings(candles); const lastHigh = swings.highs.at(-1)?.price ?? null; const prevHigh = swings.highs.at(-2)?.price ?? null; const lastLow = swings.lows.at(-1)?.price ?? null; const prevLow = swings.lows.at(-2)?.price ?? null;
  let trend = "neutral"; if (price > ema20 && ema20 > ema50) trend = "up"; else if (price < ema20 && ema20 < ema50) trend = "down";
  return { trend, bos_bullish: lastHigh != null && prevHigh != null && lastHigh > prevHigh, bos_bearish: lastLow != null && prevLow != null && lastLow < prevLow, last_swing_high: lastHigh, last_swing_low: lastLow, ema20, ema50, ema200 };
}
function analyzeVolatility(candles, price) {
  const atr = calcATR(candles, 14); const closes = candles.map(c => c.close); const bb = calcBB(closes, 20, 2); const bbWidth = bb ? (bb.upper - bb.lower) / (bb.middle || price || 1) : 0;
  let regime = "normal"; if (atr / (price || 1) < 0.0015) regime = "compressed"; else if (atr / (price || 1) > 0.005) regime = "expanding";
  return { atr, bb_width: bbWidth, regime };
}
function scoreSetup({ structure, volatility, rsi, macd, macdSig, hist, price, support, resistance, patterns, auctionSig, auction, mtf }) {
  let bull = 0, bear = 0;
  if (structure.trend === "up") bull += 2; if (structure.trend === "down") bear += 2;
  if (structure.bos_bullish) bull += 2; if (structure.bos_bearish) bear += 2;
  if (rsi > 52 && rsi < 70) bull += 1; if (rsi < 48 && rsi > 30) bear += 1;
  if (macd > macdSig) bull += 1; else bear += 1; if (hist > 0) bull += 1; else bear += 1;
  if (price > support && (price - support) / (price || 1) < 0.003) bull += 1; if (price < resistance && (resistance - price) / (price || 1) < 0.003) bear += 1;
  if (auctionSig) { if (auctionSig.bias === "bullish") bull += 2; else if (auctionSig.bias === "bearish") bear += 2; else if (auctionSig.bias === "mild_bullish") bull += 1; else if (auctionSig.bias === "mild_bearish") bear += 1; }
  const atr = volatility.atr || 1; const levels = [support, resistance]; if (auction) levels.push(auction.poc, auction.vah, auction.val);
  const nearLevel = (p) => levels.some(lvl => lvl && Math.abs(p - lvl) <= atr * 0.5);
  if (nearLevel(price)) { if (patterns.includes("bullish_engulfing") || patterns.includes("pinbar_bullish")) bull += 2; if (patterns.includes("bearish_engulfing") || patterns.includes("pinbar_bearish")) bear += 2; if (patterns.includes("indecision")) { bull -= 0.5; bear -= 0.5; } }
  if (volatility.regime === "compressed") { bull -= 0.5; bear -= 0.5; }
  let mtfNote = "4H data unavailable";
  if (mtf) { if (mtf.trend === "up") { bull += 2; bear -= 2; mtfNote = "4H trend UP — favors longs only"; } else if (mtf.trend === "down") { bear += 2; bull -= 2; mtfNote = "4H trend DOWN — favors shorts only"; } else mtfNote = "4H trend neutral — both directions open"; }
  let bias = "neutral"; const diff = bull - bear; if (diff >= 2) bias = "bullish"; if (diff <= -2) bias = "bearish";
  const confidence = Math.max(5, Math.min(95, Math.round(50 + diff * 8)));
  return { bull_score: bull, bear_score: bear, bias, confidence, mtf_note: mtfNote };
}
function buildTradePlan({ bias, price, support, resistance, atr, dp }) {
  if (!price || !atr) return { entry_zone: null, invalidation: null, tp1: null, tp2: null, risk_state: "unknown" };
  if (bias === "bullish") {
    const entry1 = price - atr * 0.15; const entry2 = price + atr * 0.15; const invalidation = support > 0 ? support - atr * 0.25 : price - atr * 1.2;
    const tp1 = resistance > 0 ? resistance : price + atr * 1.2; const tp2 = resistance > 0 ? resistance + atr * 0.8 : price + atr * 2.2;
    return { entry_zone: `${entry1.toFixed(dp)}-${entry2.toFixed(dp)}`, invalidation: invalidation.toFixed(dp), tp1: tp1.toFixed(dp), tp2: tp2.toFixed(dp), risk_state: "acceptable" };
  }
  if (bias === "bearish") {
    const entry1 = price - atr * 0.15; const entry2 = price + atr * 0.15; const invalidation = resistance > 0 ? resistance + atr * 0.25 : price + atr * 1.2;
    const tp1 = support > 0 ? support : price - atr * 1.2; const tp2 = support > 0 ? support - atr * 0.8 : price - atr * 2.2;
    return { entry_zone: `${entry1.toFixed(dp)}-${entry2.toFixed(dp)}`, invalidation: invalidation.toFixed(dp), tp1: tp1.toFixed(dp), tp2: tp2.toFixed(dp), risk_state: "acceptable" };
  }
  return { entry_zone: null, invalidation: null, tp1: null, tp2: null, risk_state: "no_trade" };
}

// ==================== NEWS FILTER & MTF ====================
const SYMBOL_CURRENCIES = { EURUSD: ["EUR", "USD"], GBPUSD: ["GBP", "USD"], USDJPY: ["USD", "JPY"], AUDUSD: ["AUD", "USD"], USDCHF: ["USD", "CHF"], USDCAD: ["USD", "CAD"], NZDUSD: ["NZD", "USD"], GBPJPY: ["GBP", "JPY"], EURJPY: ["EUR", "JPY"], EURGBP: ["EUR", "GBP"], XAUUSD: ["USD"], XAGUSD: ["USD"], BTCUSD: ["USD"], ETHUSD: ["USD"], SOLUSD: ["USD"], BNBUSD: ["USD"], XRPUSD: ["USD"], DOGEUSD: ["USD"], ADAUSD: ["USD"] };
async function fetchHighImpactEvents() {
  const ck = "news_events"; const cached = cacheGet(ck); if (cached) return cached;
  try { const res = await fetch("https://nfs.faireconomy.media/ff_calendar_thisweek.json"); if (!res.ok) return []; const data = await res.json(); const high = data.filter(e => e.impact === "High").map(e => ({ currency: e.currency, title: e.title, time: new Date(e.date).getTime() })); cacheSet(ck, high, 60 * 60 * 1000); return high; } catch (err) { console.warn("[NewsFilter]", err.message); return []; }
}
async function checkNewsFilter(symbol) {
  const sym = String(symbol || "  ").toUpperCase(); const currencies = SYMBOL_CURRENCIES[sym] || ["USD"]; const events = await fetchHighImpactEvents(); const now = Date.now(); const window = 30 * 60 * 1000;
  const nearby = events.filter(e => currencies.includes(e.currency) && Math.abs(e.time - now) <= window);
  if (nearby.length > 0) return { blocked: true, reason: `High-impact news within 30 min: ${nearby.map(e => `${e.currency} ${e.title}`).join(", ")}`, events: nearby };
  return { blocked: false };
}
async function getMTFBias(symbol) {
  const sym = String(symbol || "  ").toUpperCase(); const ck = `mtf:${sym}`; const cached = cacheGet(ck); if (cached) return cached;
  try {
    const candles4h = await fetchCandles(sym, "4h", 100); if (!candles4h || candles4h.length < 50) return { trend: "neutral", ema200: null };
    const closes = candles4h.map(c => c.close); const price = closes.at(-1); const ema50 = calcEMA(closes, 50).at(-1) ?? price; const ema200 = calcEMA(closes, 200).at(-1) ?? price;
    let trend = "neutral"; if (price > ema50 && ema50 > ema200) trend = "up"; else if (price < ema50 && ema50 < ema200) trend = "down";
    const result = { trend, ema50: +ema50.toFixed(5), ema200: +ema200.toFixed(5), price }; cacheSet(ck, result, 15 * 60 * 1000); return result;
  } catch (err) { console.warn("[MTF]", err.message); return { trend: "neutral", ema200: null }; }
}

// ==================== MAIN ANALYSIS (Preserved for /chat & /market/analyze) ====================
async function analyzeSymbol(symbol, interval = "1h", customSize = null) {
  const sym = String(symbol || " ").toUpperCase(); const iv = normalizeInterval(interval); const ck = `analysis:${sym}:${iv}:${customSize || "auto"}`; const cached = cacheGet(ck); if (cached) return cached;
  const [candles, spot, newsCheck, mtf] = await Promise.all([fetchCandles(sym, iv, customSize), fetchSpotPrice(sym), checkNewsFilter(sym), getMTFBias(sym)]);
  if (newsCheck.blocked) return { symbol: sym, direction: "NEUTRAL", strength: "NEWS_BLACKOUT", interval: iv, news_filter: newsCheck, trade_plan: { entry_zone: null, invalidation: null, tp1: null, tp2: null, risk_state: "no_trade" }, concise_signal: { direction: "STAND DOWN", entry: "N/A", sl: "N/A", tp: "N/A", ai_opinion: `News blackout active. ${newsCheck.reason}` }, ai_opinion: `STAND DOWN — ${newsCheck.reason}` };
  if (!candles && !spot) return { symbol: sym, error: `No data for ${sym}` };
  const price = spot?.price ?? candles?.at(-1)?.close ?? 0;
  if (!candles || candles.length < 30) return { symbol: sym, price, source: spot?.source, analysis: "Insufficient candle data", candleCount: candles?.length ?? 0 };
  const closes = candles.map(c => c.close); const rsi = calcRSI(closes); const { macd, signal: macdSig, hist } = calcMACD(closes); const { support, resistance } = calcSR(candles); const bb = calcBB(closes);
  const structure = analyzeStructure(candles, price); const volatility = analyzeVolatility(candles, price); const patterns = detectCandlePattern(candles); const auction = calcAuction(candles); const auctionSig = auctionSignal(price, auction);
  const score = scoreSetup({ structure, volatility, rsi, macd, macdSig, hist, price, support, resistance, patterns, auctionSig, auction, mtf });
  let direction = "NEUTRAL"; if (rsi >= 70) direction = "OVERBOUGHT"; else if (rsi <= 30) direction = "OVERSOLD"; else if (score.bias === "bullish") direction = "BULLISH"; else if (score.bias === "bearish") direction = "BEARISH";
  let strength = "WEAK"; if (score.confidence >= 75) strength = "STRONG"; else if (score.confidence >= 60) strength = "MODERATE";
  const isCrypto = CRYPTO_SET.has(sym); const dp = isCrypto || sym === "XAUUSD" ? 2 : 5; const trade_plan = buildTradePlan({ bias: score.bias, price, support, resistance, atr: volatility.atr, dp });
  const auctionNote = auctionSig.note ? `Auction: ${auctionSig.note}` : ""; const aiOpinion = direction === "NEUTRAL" ? `No clear edge right now. Wait for cleaner structure.${auctionNote}` : `${direction} bias with ${strength.toLowerCase()} confidence. Structure=${structure.trend}, vol=${volatility.regime}, patterns=${patterns.join(", ") || "none"}.${auctionNote}`;
  const result = {
    symbol: sym, price, direction, strength, interval: iv, candleCount: candles.length, source: spot?.source ?? "binance",
    rsi: +rsi.toFixed(1), macd: +macd.toFixed(6), macd_signal: macd > macdSig ? "bullish" : "bearish",
    ema20: +structure.ema20.toFixed(dp), ema50: +structure.ema50.toFixed(dp), ema200: +structure.ema200.toFixed(dp),
    support: +support.toFixed(dp), resistance: +resistance.toFixed(dp),
    bb_upper: bb ? +bb.upper.toFixed(dp) : null, bb_middle: bb ? +bb.middle.toFixed(dp) : null, bb_lower: bb ? +bb.lower.toFixed(dp) : null,
    regime: structure.trend === "neutral" ? "range_or_mixed" : "trend", confidence: score.confidence,
    structure: { trend: structure.trend, bos_bullish: structure.bos_bullish, bos_bearish: structure.bos_bearish, last_swing_high: structure.last_swing_high ? +structure.last_swing_high.toFixed(dp) : null, last_swing_low: structure.last_swing_low ? +structure.last_swing_low.toFixed(dp) : null },
    volatility: { atr: +volatility.atr.toFixed(dp), bb_width: +volatility.bb_width.toFixed(6), regime: volatility.regime },
    patterns, trade_plan, concise_signal: { direction, entry: trade_plan.entry_zone || "N/A", sl: trade_plan.invalidation || "N/A", tp: trade_plan.tp1 || "N/A", ai_opinion: aiOpinion },
    auction: auction ? { poc: +auction.poc.toFixed(dp), vah: +auction.vah.toFixed(dp), val: +auction.val.toFixed(dp), position: auctionSig.position, bias: auctionSig.bias, note: auctionSig.note, hvn: auction.hvn.map(p => +p.toFixed(dp)), lvn: auction.lvn.map(p => +p.toFixed(dp)) } : null,
    mtf: { trend: mtf?.trend || "unknown", note: score.mtf_note }, news_filter: { blocked: false }, ai_opinion: aiOpinion,
    summary: `${sym} @${price.toFixed(dp)} | ${direction}(${strength}) | Conf:${score.confidence} | Trend:${structure.trend}(4H:${mtf?.trend || "?"}) | RSI:${rsi.toFixed(1)} | S:${support.toFixed(dp)} R:${resistance.toFixed(dp)}${auction ? ` | POC:${auction.poc.toFixed(dp)} VAH:${auction.vah.toFixed(dp)} VAL:${auction.val.toFixed(dp)} [${auctionSig.position}]` : ""} [${candles.length} candles ${iv}]`,
  };
  cacheSet(ck, result, 60000); return result;
}

// ==================== MARKET MESSAGE HELPERS ====================
function isMarketMessage(message = " ") { const text = String(message || " ").toLowerCase(); return (/\b(eurusd|gbpusd|usdjpy|audusd|xauusd|usdchf|usdcad|nzdusd|gbpjpy|eurjpy|eurgbp|btcusd|ethusd|solusd|bnbusd|xrpusd|dogeusd|adausd|btc|eth|sol|bnb|xrp|doge|ada|usdngn)\b/i.test(text) || /\b(price|analysis|signal|buy|sell|entry|tp|sl|market)\b/i.test(text)); }
function wantsDetailedTradeReason(message = "  ") { return ["why", "reason", "explain", "full analysis", "show structure", "details", "detailed", "breakdown", "confidence", "pattern", "strategy", "regime"].some(k => String(message || "  ").toLowerCase().includes(k)); }
function formatConciseSignal(analysis) { if (!analysis || analysis.error) return analysis?.error || "No market data available."; const cs = analysis.concise_signal || {}; return [`${analysis.symbol} ${String(analysis.interval || "").toUpperCase()}`.trim(), `Direction: ${cs.direction || analysis.direction || "NEUTRAL"}`, `Entry: ${cs.entry || analysis.trade_plan?.entry_zone || "N/A"}`, `SL: ${cs.sl || analysis.trade_plan?.invalidation || "N/A"}`, `TP: ${cs.tp || analysis.trade_plan?.tp1 || "N/A"}`, `AI Opinion: ${cs.ai_opinion || analysis.ai_opinion || "No strong edge right now."}`].join("\n"); }

// ==================== WEB SEARCH + WEATHER ====================
async function webSearch(query) { try { const res = await fetch(`https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`); const d = await res.json(); return d.AbstractText || d.Answer || d.RelatedTopics?.[0]?.Text || "No result found."; } catch (e) { return "Search failed: " + e.message; } }
async function getWeather(city = "Lagos") { try { const geo = await (await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`)).json(); const loc = geo.results?.[0]; if (!loc) return "City not found"; const wx = await (await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${loc.latitude}&longitude=${loc.longitude}&current_weather=true&timezone=auto`)).json(); const cw = wx.current_weather; return `${loc.name}: ${cw.temperature}°C, wind ${cw.windspeed} km/h, ${cw.weathercode <= 1 ? "Clear" : cw.weathercode <= 3 ? "Cloudy" : "Rainy"}`; } catch (e) { return "Weather unavailable"; } }

// ==================== LOT SIZE ENGINE & MT5 BRIDGE ====================
const PIP_CONFIG = { EURUSD: { pipSize: 0.0001, pipValue: 10 }, GBPUSD: { pipSize: 0.0001, pipValue: 10 }, AUDUSD: { pipSize: 0.0001, pipValue: 10 }, NZDUSD: { pipSize: 0.0001, pipValue: 10 }, USDCHF: { pipSize: 0.0001, pipValue: 10 }, USDCAD: { pipSize: 0.0001, pipValue: 10 }, EURGBP: { pipSize: 0.0001, pipValue: 10 }, USDJPY: { pipSize: 0.01, pipValue: 9.3 }, GBPJPY: { pipSize: 0.01, pipValue: 9.3 }, EURJPY: { pipSize: 0.01, pipValue: 9.3 }, XAUUSD: { pipSize: 0.01, pipValue: 100 }, XAGUSD: { pipSize: 0.001, pipValue: 50 }, BTCUSD: { pipSize: 1, pipValue: 1, isCrypto: true }, ETHUSD: { pipSize: 0.1, pipValue: 1, isCrypto: true }, SOLUSD: { pipSize: 0.01, pipValue: 1, isCrypto: true }, BNBUSD: { pipSize: 0.01, pipValue: 1, isCrypto: true }, XRPUSD: { pipSize: 0.0001, pipValue: 1, isCrypto: true }, DOGEUSD: { pipSize: 0.00001, pipValue: 1, isCrypto: true }, ADAUSD: { pipSize: 0.00001, pipValue: 1, isCrypto: true } };
function calculateLotSize({ symbol, balance, riskPercent = 1, entry, stopLoss }) {
  const sym = String(symbol || "").toUpperCase(); const cfg = PIP_CONFIG[sym]; const MAX_RISK = 2; const MAX_LOT = 5; const MIN_LOT = 0.01;
  const safeRisk = Math.min(riskPercent, MAX_RISK); const riskAmt = (balance || 1000) * (safeRisk / 100);
  if (!entry || !stopLoss || entry === stopLoss) return MIN_LOT; if (!cfg) { console.warn(`[LotSize] Unknown symbol ${sym}`); return MIN_LOT; }
  let lotSize; if (cfg.isCrypto) { const priceDist = Math.abs(entry - stopLoss); lotSize = priceDist === 0 ? MIN_LOT : riskAmt / priceDist; } else { const stopPips = Math.abs(entry - stopLoss) / cfg.pipSize; lotSize = stopPips === 0 ? MIN_LOT : riskAmt / (stopPips * cfg.pipValue); }
  return Math.max(MIN_LOT, Math.min(MAX_LOT, Math.round(lotSize * 100) / 100));
}
async function sendToMT5Bridge({ symbol, action, lotSize, entry, sl, tp, reason = "  " }) {
  if (!MT5_BRIDGE_URL) { console.log(`[PAPER TRADE] ${action.toUpperCase()} ${symbol} | Lot:${lotSize} | Entry:${entry} | SL:${sl} | TP:${tp}`); return { mode: "paper", symbol, action, lotSize, entry, sl, tp, status: "simulated", message: "MT5_BRIDGE_URL not set — paper mode" }; }
  try { const res = await fetch(`${MT5_BRIDGE_URL}/place_trade`, { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${AUTH_TOKEN}` }, body: JSON.stringify({ symbol, action, lotSize, entry, sl, tp, reason }) }); const data = await res.json(); if (!res.ok) throw new Error(data?.error || "MT5 bridge error"); return { mode: "live", ...data }; } catch (err) { console.error("[MT5 Bridge]", err.message); return { mode: "live", status: "failed", error: err.message }; }
}

// ==================== MODIFIED CLAUDE HVR ENGINE (Replaces ALL old strategies) ====================
const HVR_STORAGE = join(__dirname, "hvr_storage.json");
const HVR_CALIBRATION = join(__dirname, "hvr_calibration.json");

class HvrEngine {
  constructor() {
    this.persistence = {
      load: () => { try { return existsSync(HVR_STORAGE) ? JSON.parse(readFileSync(HVR_STORAGE, "utf8")) : { trades: [], cooldowns: {} }; } catch { return { trades: [], cooldowns: {} }; } },
      save: (data) => { try { writeFileSync(HVR_STORAGE, JSON.stringify(data, null, 2)); } catch {} },
      logConfidence: (predicted, actual) => { try { const log = existsSync(HVR_CALIBRATION) ? JSON.parse(readFileSync(HVR_CALIBRATION, "utf8")) : []; log.push({ predicted, actual, ts: Date.now() }); writeFileSync(HVR_CALIBRATION, JSON.stringify(log.slice(-1000), null, 2)); } catch {} },
      getCalibration: () => { try { const log = existsSync(HVR_CALIBRATION) ? JSON.parse(readFileSync(HVR_CALIBRATION, "utf8")) : []; const buckets = { "50-59": [], "60-69": [], "70-79": [], "80-89": [], "90-99": [] }; log.forEach(e => { const b = Math.floor(e.predicted/10)*10; if (buckets[`${b}-${b+9}`]) buckets[`${b}-${b+9}`].push(e.actual); }); return Object.fromEntries(Object.entries(buckets).map(([k, v]) => [k, v.length ? (v.reduce((a,b)=>a+b,0)/v.length)*100 : 0])); } catch { return {}; } }
    };
  }

  // OHLC-ONLY VOLUME PROXY (Recommendation)
  calcVolumeProxy(candles, period = 20) {
    const proxies = candles.slice(-period).map(c => (c.high - c.low) + Math.abs(c.close - c.open));
    const avg = proxies.reduce((a, b) => a + b, 0) / proxies.length;
    const cur = proxies[proxies.length - 1];
    return { avg, current: cur, ratio: avg > 0 ? cur / avg : 1 };
  }

  // DIRECTION-AWARE COOLDOWN (Recommendation)
  isInCooldown(symbol, direction) {
    const state = this.persistence.load(); const key = `${symbol}_${direction}`; const cd = state.cooldowns[key];
    if (!cd) return false;
    if (Date.now() - cd > 4 * 60 * 60 * 1000) { delete state.cooldowns[key]; this.persistence.save(state); return false; }
    return true;
  }
  setCooldown(symbol, direction) { const state = this.persistence.load(); state.cooldowns[`${symbol}_${direction}`] = Date.now(); this.persistence.save(state); }

  // HVR STEPS
  checkH4Bias(h4) {
    if (!h4 || h4.length < 200) return { bias: "NEUTRAL", reason: "Insufficient H4 data" };
    const closes = h4.map(c => c.close); const ema200 = calcEMA(closes, 200).at(-1); const price = closes.at(-1);
    const diffPips = Math.abs(price - ema200) / 0.0001;
    if (price > ema200 && diffPips > 20) return { bias: "BULLISH", ema200, diffPips };
    if (price < ema200 && diffPips > 20) return { bias: "BEARISH", ema200, diffPips };
    return { bias: "NEUTRAL", reason: "Price within 20 pips of 200 EMA" };
  }
  checkH1Structure(h1, bias) {
    if (!h1 || h1.length < 20) return { valid: false, reason: "Insufficient H1 data" };
    const swings = findSwings(h1, 2);
    if (bias === "BULLISH") {
      if (!swings.lows.length || !swings.highs.length) return { valid: false, reason: "No swings" };
      const swingLow = swings.lows.at(-1).price; const swingHigh = swings.highs.at(-1).price;
      const fib618 = swingLow + (swingHigh - swingLow) * 0.618; const fib786 = swingLow + (swingHigh - swingLow) * 0.786;
      const price = h1.at(-1).close;
      if (price >= fib618 && price <= fib786) return { valid: true, zone: [fib618, fib786], price };
      return { valid: false, reason: "Price not in 0.618-0.786 pullback zone" };
    } else {
      const swingHigh = swings.highs.at(-1).price; const swingLow = swings.lows.at(-1).price;
      const fib618 = swingHigh - (swingHigh - swingLow) * 0.618; const fib786 = swingHigh - (swingHigh - swingLow) * 0.786;
      const price = h1.at(-1).close;
      if (price >= fib618 && price <= fib786) return { valid: true, zone: [fib618, fib786], price };
      return { valid: false, reason: "Price not in 0.618-0.786 pullback zone" };
    }
  }
  checkM15Momentum(m15, bias) {
    if (!m15 || m15.length < 15) return { valid: false };
    const closes = m15.map(c => c.close); const rsi = calcRSI(closes); const prevRsi = calcRSI(closes.slice(0, -1));
    if (bias === "BULLISH") return rsi > 40 && rsi < 60 && prevRsi <= 40 ? { valid: true, rsi } : { valid: false, rsi };
    return rsi < 60 && rsi > 40 && prevRsi >= 60 ? { valid: true, rsi } : { valid: false, rsi };
  }
  checkVolumeVol(h1, m15) {
    const atrH1 = calcATR(h1); const atrAvg = calcATR(h1.slice(-40)); const vol = this.calcVolumeProxy(m15);
    return { valid: atrH1 > atrAvg && vol.ratio > 1.5, atrH1, volRatio: vol.ratio };
  }
  checkM5Trigger(m5, bias, fibZone) {
    if (!m5 || m5.length < 3) return { valid: false };
    const prev = m5[m5.length - 2], curr = m5[m5.length - 1]; const closes = m5.map(c => c.close);
    const rsi = calcRSI(closes); const rsiPrev = calcRSI(closes.slice(0, -1));
    const inZone = curr.close >= fibZone[0] && curr.close <= fibZone[1];
    let patternOk = false;
    if (bias === "BULLISH") { patternOk = (prev.close < prev.open && curr.close > curr.open && curr.open <= prev.close && curr.close >= prev.open) || (Math.min(curr.open, curr.close) - curr.low > 2 * Math.abs(curr.close - curr.open)); }
    else { patternOk = (prev.close > prev.open && curr.close < curr.open && curr.open >= prev.close && curr.close <= prev.open) || (curr.high - Math.max(curr.open, curr.close) > 2 * Math.abs(curr.close - curr.open)); }
    const rsiCross = bias === "BULLISH" ? rsi > 50 && rsiPrev <= 50 : rsi < 50 && rsiPrev >= 50;
    return { valid: inZone && patternOk && rsiCross, pattern: patternOk ? "engulfing/pinbar" : "none", rsi };
  }
  calculateRisk(entry, bias, h1) {
    const atr = calcATR(h1); const slDist = atr * 1.5; const tpDist = atr * 3.0;
    return { sl: bias === "BULLISH" ? entry - slDist : entry + slDist, tp: bias === "BULLISH" ? entry + tpDist : entry - tpDist, atr, rr: 2.0 };
  }

  async generateSignal(symbol) {
    const fetchTF = (iv, size) => fetchCandles(symbol, iv, size);
    const [h4, h1, m15, m5] = await Promise.all([fetchTF("4h", 100), fetchTF("1h", 100), fetchTF("15min", 50), fetchTF("5min", 50)]);
    const h4Bias = this.checkH4Bias(h4); if (h4Bias.bias === "NEUTRAL") return { signal: "NO_TRADE", reason: h4Bias.reason, step: 1 };
    const h1Struct = this.checkH1Structure(h1, h4Bias.bias); if (!h1Struct.valid) return { signal: "NO_TRADE", reason: h1Struct.reason, step: 2 };
    const m15Mom = this.checkM15Momentum(m15, h4Bias.bias); if (!m15Mom.valid) return { signal: "NO_TRADE", reason: "M15 RSI not confirming momentum", rsi: m15Mom.rsi, step: 3 };
    const volVol = this.checkVolumeVol(h1, m15); if (!volVol.valid) return { signal: "NO_TRADE", reason: "Volume/Vol filter failed", volRatio: volVol.volRatio, step: 4 };
    const m5Trig = this.checkM5Trigger(m5, h4Bias.bias, h1Struct.zone); if (!m5Trig.valid) return { signal: "NO_TRADE", reason: "M5 trigger not met", step: 5 };
    const risk = this.calculateRisk(h1Struct.price, h4Bias.bias, h1);
    return {
      signal: h4Bias.bias === "BULLISH" ? "BUY" : "SELL", symbol, entry: h1Struct.price.toFixed(5),
      sl: risk.sl.toFixed(5), tp: risk.tp.toFixed(5), atr: risk.atr.toFixed(5), rr: risk.rr,
      confidence: Math.min(85, 50 + (volVol.volRatio - 1) * 20 + (m15Mom.rsi > 45 && m15Mom.rsi < 55 ? 10 : 0)),
      step: "PASS", timestamp: new Date().toISOString()
    };
  }

  async backtest(symbol, days = 30) {
    const candles = await fetchCandles(symbol, "1h", days * 24);
    if (!candles || candles.length < 100) return { error: "Insufficient historical data" };
    let balance = 1000, wins = 0, losses = 0, maxDD = 0, peak = balance; const trades = [];
    for (let i = 100; i < candles.length; i++) {
      const h1Slice = candles.slice(i - 100, i); const bias = candles[i - 1].close > calcEMA(h1Slice.map(c => c.close), 200).at(-1) ? "BULLISH" : "BEARISH";
      const atr = calcATR(h1Slice); const entry = candles[i].close; const sl = bias === "BULLISH" ? entry - atr * 1.5 : entry + atr * 1.5; const tp = bias === "BULLISH" ? entry + atr * 3.0 : entry - atr * 3.0;
      let outcome = "pending";
      for (let j = i + 1; j <= i + 12; j++) { if (!candles[j]) break; if (bias === "BULLISH" && candles[j].high >= tp) { outcome = "win"; break; } if (bias === "BEARISH" && candles[j].low <= tp) { outcome = "win"; break; } if (bias === "BULLISH" && candles[j].low <= sl) { outcome = "loss"; break; } if (bias === "BEARISH" && candles[j].high >= sl) { outcome = "loss"; break; } }
      const risk = balance * 0.01; if (outcome === "win") { balance += risk * 2; wins++; } else if (outcome === "loss") { balance -= risk; losses++; peak = Math.max(peak, balance); }
      maxDD = Math.max(maxDD, peak - balance); trades.push({ entry, outcome, balance: +balance.toFixed(2) });
    }
    return { symbol, trades_run: trades.length, win_rate: wins / (wins + losses), final_balance: +balance.toFixed(2), max_drawdown: +maxDD.toFixed(2) };
  }
}

const hvr = new HvrEngine();

// ==================== TRADE MEMORY ====================
const tradeMemory = new Map();
function addTradeMemory(symbol, entry) { const sym = String(symbol || " ").toUpperCase(); if (!tradeMemory.has(sym)) tradeMemory.set(sym, []); const entries = tradeMemory.get(sym); entries.push({ ...entry, timestamp: Date.now() }); if (entries.length > 20) entries.splice(0, entries.length - 20); }
function getTradeMemory(symbol, limit = 5) { const sym = String(symbol || " ").toUpperCase(); return (tradeMemory.get(sym) || []).slice(-limit); }
function formatTradeMemoryForPrompt(symbol) { const entries = getTradeMemory(symbol, 5); if (!entries.length) return ""; const lines = entries.map(e => { const d = new Date(e.timestamp).toISOString().slice(0, 10); return `[${d}] ${e.outcome || "?"} | Pattern:${e.pattern || "none"} | Dir:${e.direction || "?"} | Note:${e.note || ""}`; }); return `\nPast trade memory for ${symbol}:\n${lines.join("\n")}`; }

// ==================== ANDROID AGENT TOOLS ====================
const SERVER_SIDE_TOOLS = new Set(["search_web", "get_weather", "get_market_data", "analyze_market", "run_code", "wait_and_verify", "assert_text_visible"]);
const AGENT_TOOLS = [
  { type: "function", function: { name: "open_app", description: "Launch any installed app by name. MANDATORY: You must immediately follow this action with 'read_screen' or 'read_screen_structured' to verify the UI state changed before taking the next step.", parameters: { type: "object", properties: { app_name: { type: "string" } }, required: ["app_name"] } } },
  { type: "function", function: { name: "open_url", description: "Open a full URL in the default browser.", parameters: { type: "object", properties: { url: { type: "string" } }, required: ["url"] } } },
  { type: "function", function: { name: "press_back", description: "Press the Android back button. MANDATORY: You must immediately follow this action with 'read_screen' or 'read_screen_structured' to verify the UI state changed before taking the next step.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "press_home", description: "Press the Android home button.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "open_recents", description: "Open the recent apps screen.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "open_notifications", description: "Open the notification shade.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "get_current_app", description: "Return the foreground package name and label.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "get_current_activity", description: "Return the current Android activity name.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "read_screen", description: "Read visible text and content descriptions from the screen.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "read_screen_structured", description: "Return a structured accessibility tree dump.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "find_element", description: "Find a UI element by text, hint, id, or class.", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
  { type: "function", function: { name: "take_screenshot", description: "Take a screenshot and return metadata.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "analyze_screenshot", description: "Analyze the last screenshot visually with the vision model.", parameters: { type: "object", properties: { prompt: { type: "string" } }, required: ["prompt"] } } },
  { type: "function", function: { name: "tap_button", description: "Tap an element by exact visible label. MANDATORY: You must immediately follow this action with 'read_screen' or 'read_screen_structured' to verify the UI state changed before taking the next step.", parameters: { type: "object", properties: { label: { type: "string" } }, required: ["label"] } } },
  { type: "function", function: { name: "tap_coordinates", description: "Tap the screen at exact x/y coordinates. MANDATORY: You must immediately follow this action with 'read_screen' or 'read_screen_structured' to verify the UI state changed before taking the next step.", parameters: { type: "object", properties: { x: { type: "number" }, y: { type: "number" } }, required: ["x", "y"] } } },
  { type: "function", function: { name: "double_tap", description: "Double-tap by label or coordinates.", parameters: { type: "object", properties: { label: { type: "string" }, x: { type: "number" }, y: { type: "number" } } } } },
  { type: "function", function: { name: "scroll", description: "Scroll the screen in a direction. MANDATORY: You must immediately follow this action with 'read_screen' or 'read_screen_structured' to verify the UI state changed before taking the next step.", parameters: { type: "object", properties: { direction: { type: "string", enum: ["up", "down", "left", "right"] } } } } },
  { type: "function", function: { name: "swipe", description: "Swipe using start and end coordinates. MANDATORY: You must immediately follow this action with 'read_screen' or 'read_screen_structured' to verify the UI state changed before taking the next step.", parameters: { type: "object", properties: { startX: { type: "number" }, startY: { type: "number" }, endX: { type: "number" }, endY: { type: "number" }, duration_ms: { type: "number" } }, required: ["startX", "startY", "endX", "endY"] } } },
  { type: "function", function: { name: "drag_and_drop", description: "Drag from one coordinate to another.", parameters: { type: "object", properties: { startX: { type: "number" }, startY: { type: "number" }, endX: { type: "number" }, endY: { type: "number" }, duration_ms: { type: "number" } }, required: ["startX", "startY", "endX", "endY"] } } },
  { type: "function", function: { name: "focus_field", description: "Focus a text field by label, hint, or content-desc.", parameters: { type: "object", properties: { field: { type: "string" } }, required: ["field"] } } },
  { type: "function", function: { name: "type_text", description: "Type text into the focused or targeted field. MANDATORY: You must immediately follow this action with 'read_screen' or 'read_screen_structured' to verify the UI state changed before taking the next step.", parameters: { type: "object", properties: { value: { type: "string" }, field: { type: "string" } }, required: ["value"] } } },
  { type: "function", function: { name: "clear_text", description: "Clear the focused or targeted input field.", parameters: { type: "object", properties: { field: { type: "string" } } } } },
  { type: "function", function: { name: "paste_text", description: "Paste text via clipboard into a field.", parameters: { type: "object", properties: { value: { type: "string" }, field: { type: "string" } }, required: ["value"] } } },
  { type: "function", function: { name: "press_enter", description: "Press Enter / Done / Search on the keyboard.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "hide_keyboard", description: "Hide the soft keyboard.", parameters: { type: "object", properties: {} } } },
  { type: "function", function: { name: "toggle_wifi", description: "Toggle Wi-Fi on or off.", parameters: { type: "object", properties: { enabled: { type: "boolean" } }, required: ["enabled"] } } },
  { type: "function", function: { name: "toggle_bluetooth", description: "Toggle Bluetooth on or off.", parameters: { type: "object", properties: { enabled: { type: "boolean" } }, required: ["enabled"] } } },
  { type: "function", function: { name: "set_volume", description: "Set media volume 0–100.", parameters: { type: "object", properties: { level: { type: "number" } }, required: ["level"] } } },
  { type: "function", function: { name: "set_brightness", description: "Set screen brightness 0–100.", parameters: { type: "object", properties: { level: { type: "number" } }, required: ["level"] } } },
  { type: "function", function: { name: "open_app_settings", description: "Open settings page for an app.", parameters: { type: "object", properties: { app_name: { type: "string" } }, required: ["app_name"] } } },
  { type: "function", function: { name: "grant_permission_if_prompted", description: "Handle Android permission prompts.", parameters: { type: "object", properties: { allow: { type: "boolean" } }, required: ["allow"] } } },
  { type: "function", function: { name: "make_call", description: "Initiate a phone call.", parameters: { type: "object", properties: { contact_name: { type: "string" }, phone_number: { type: "string" } } } } },
  { type: "function", function: { name: "send_whatsapp", description: "Open WhatsApp for a contact and optional message.", parameters: { type: "object", properties: { contact_name: { type: "string" }, message: { type: "string" } }, required: ["contact_name"] } } },
  { type: "function", function: { name: "send_sms", description: "Open SMS composer for a contact.", parameters: { type: "object", properties: { contact_name: { type: "string" }, message: { type: "string" } }, required: ["contact_name"] } } },
  { type: "function", function: { name: "play_music", description: "Play music on Spotify or YouTube.", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
  { type: "function", function: { name: "navigate_to", description: "Open Maps and navigate to a destination.", parameters: { type: "object", properties: { destination: { type: "string" } }, required: ["destination"] } } },
  { type: "function", function: { name: "take_photo", description: "Open camera and capture a photo.", parameters: { type: "object", properties: { front_camera: { type: "boolean" } } } } },
  { type: "function", function: { name: "set_alarm", description: "Set an alarm.", parameters: { type: "object", properties: { label: { type: "string" }, time: { type: "string" } }, required: ["time"] } } },
  { type: "function", function: { name: "set_timer", description: "Start a countdown timer.", parameters: { type: "object", properties: { duration: { type: "string" } }, required: ["duration"] } } },
  { type: "function", function: { name: "search_web", description: "Search for current information on the web.", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
  { type: "function", function: { name: "get_weather", description: "Get current weather for a city.", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } },
  { type: "function", function: { name: "get_market_data", description: "Get live spot price for a forex/crypto symbol.", parameters: { type: "object", properties: { symbol: { type: "string" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "analyze_market", description: "Full market analysis: structure, indicators, auction, trade plan.", parameters: { type: "object", properties: { symbol: { type: "string" }, interval: { type: "string" }, outputsize: { type: "number" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "run_code", description: "Execute Python or JavaScript in a secure sandbox.", parameters: { type: "object", properties: { language: { type: "string", enum: ["python", "javascript"] }, code: { type: "string" }, stdin: { type: "string" }, timeout_ms: { type: "number" } }, required: ["language", "code"] } } },
  { type: "function", function: { name: "wait_and_verify", description: "Wait for Android to process an action, then automatically trigger a screen read. Use this instead of calling read_screen manually after taps.", parameters: { type: "object", properties: { delay_ms: { type: "number", default: 500 } }, required: [] } } },
  { type: "function", function: { name: "assert_text_visible", description: "Check the last known device state for a string. Do NOT use read_screen for this. Use this to verify a previous action succeeded.", parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } } },
];

// ==================== LOCAL TOOL EXECUTION ====================
async function runLocalTool(name, args = {}, agentState = null) {
  switch (name) {
    case "search_web": return { ok: true, data: await webSearch(args.query || "  ") };
    case "get_weather": return { ok: true, data: await getWeather(args.city || "Lagos") };
    case "get_market_data": return { ok: true, data: await fetchMarketPrices([args.symbol || "BTCUSD"]) };
    case "analyze_market": return { ok: true, data: await analyzeSymbol(args.symbol, args.interval, args.outputsize) };
    case "run_code": return { ok: true, data: await runSandbox({ language: args.language, code: args.code, stdin: args.stdin || "  ", timeout_ms: args.timeout_ms || 8000 }) };
    case "wait_and_verify": { const delay = args.delay_ms || 500; await new Promise(r => setTimeout(r, delay)); return { ok: true, data: { status: "ready", observation: "Wait complete. Proceed to read_screen or assert_text_visible to verify state." } }; }
    case "assert_text_visible": { if (!agentState || !agentState.deviceState) return { ok: true, data: { asserted: false, text_searched: args.text, error: "No device state available. Call read_screen first." } }; const text = String(args.text || "  ").toLowerCase(); const screenText = String(agentState.deviceState?.screen_text || "  ").toLowerCase(); const found = screenText.includes(text); return { ok: true, data: { asserted: found, text_searched: args.text, screen_snippet: screenText.slice(0, 200) } }; }
    default: return { ok: false, error: "Not a server-side tool" };
  }
}

// ==================== AUTOMATION SYSTEM PROMPT ====================
function buildAutomationSystemPrompt({ deviceState, memory }) {
  const deviceStateText = buildDeviceStateBlock(deviceState); const memoryText = buildMemoryBlock(summarizeMemory(memory, 6, 700));
  return ["You are FRIT, an Android device-control agent operating a State Machine.", "CRITICAL RULE: The UI is asynchronous. You cannot assume an action succeeded until you read the state.", "THE STRICT EXECUTION LOOP:", "1. Plan Step (e.g., 'Tap Settings')", "2. Execute Action (Call tool)", "3. VERIFY STATE (You Must call 'wait_and_verify' -> 'read_screen' or 'assert_text_visible')", "4. If state matches expectation -> Go to Step 1 for next action.", "5. If state does NOT match -> Recover (e.g., try tap_coordinates, scroll, or press_back).", "NEVER chain more than ONE physical action (tap/type/swipe) without a verification step in between.", "NEVER guess what is on the screen. If you are unsure, call read_screen.", "Device state:", deviceStateText, memoryText ? `\n${memoryText}` : ""].filter(Boolean).join("\n");
}
function buildDeviceStateBlock(ds = {}) {
  const d = typeof ds === "string" ? { raw: ds } : ds || {}; const parts = [];
  if (d.raw) parts.push(`Raw: ${truncateText(d.raw, 700)}`); if (d.current_app) parts.push(`Current app: ${d.current_app}`); if (d.current_activity) parts.push(`Activity: ${d.current_activity}`);
  if (d.screen_text) parts.push(`Screen text: ${truncateText(d.screen_text, 1200)}`); if (d.screen_summary) parts.push(`Screen summary: ${truncateText(d.screen_summary, 500)}`);
  if (d.network_status) parts.push(`Network: ${d.network_status}`); if (d.battery_level !== undefined) parts.push(`Battery: ${d.battery_level}%`); if (d.keyboard_open !== undefined) parts.push(`Keyboard open: ${d.keyboard_open}`);
  return parts.length ? parts.join("\n") : "No device state provided.";
}
function determineTaskType(goal) { const text = String(goal || "  ").toLowerCase(); if (/(?:open|launch|start)\s+(?:app|camera|whatsapp|browser|chrome)/i.test(text)) return "app_navigation"; if (/(?:send|write|message|text|email)/i.test(text)) return "messaging"; if (/(?:photo|picture|camera|screenshot)/i.test(text)) return "camera"; if (/(?:trade|buy|sell|market|forex|crypto)/i.test(text)) return "trading"; if (/(?:call|phone|dial)/i.test(text)) return "calling"; if (/(?:navigate|direction|map|go to)/i.test(text)) return "navigation"; if (/(?:search|find|look up|what is)/i.test(text)) return "search"; if (/(?:code|calculate|compute|script)/i.test(text)) return "coding"; return "general"; }
function getToolsForTask(goal) { const text = String(goal || "  ").toLowerCase(); const core = ["press_back", "press_home", "open_recents", "open_notifications", "open_app", "open_url", "read_screen", "read_screen_structured", "find_element", "tap_button", "tap_coordinates", "scroll", "swipe", "search_web", "get_weather"]; const selected = [...core]; if (/(?:text|type|search|input|write)/i.test(text)) selected.push("focus_field", "type_text", "clear_text", "paste_text", "press_enter", "hide_keyboard"); if (/(?:call|whatsapp|sms|send)/i.test(text)) selected.push("make_call", "send_whatsapp", "send_sms"); if (/(?:photo|camera|screenshot)/i.test(text)) selected.push("take_photo", "take_screenshot", "analyze_screenshot"); if (/(?:trade|buy|sell|market|forex|crypto)/i.test(text)) selected.push("get_market_data", "analyze_market"); if (/(?:music|play|song|spotify)/i.test(text)) selected.push("play_music"); if (/(?:code|calculate|compute|python|javascript)/i.test(text)) selected.push("run_code"); if (/(?:wifi|bluetooth|volume|brightness|settings|permission)/i.test(text)) selected.push("toggle_wifi", "toggle_bluetooth", "set_volume", "set_brightness", "open_app_settings", "grant_permission_if_prompted"); return [...new Set(selected)]; }

const pendingAndroidCallbacks = new Map();
function waitForAndroid(callId, timeoutMs = 60000) { return new Promise((resolve, reject) => { const timer = setTimeout(() => { pendingAndroidCallbacks.delete(callId); reject(new Error(`Android timeout (${timeoutMs / 1000}s) for callId ${callId}`)); }, timeoutMs); pendingAndroidCallbacks.set(callId, { resolve, reject, timer }); }); }
function resolveAndroid(callId, result) { const cb = pendingAndroidCallbacks.get(callId); if (cb) { clearTimeout(cb.timer); pendingAndroidCallbacks.delete(callId); cb.resolve(result); return true; } return false; }

async function handleAutomationRequest({ goal, device_state = {}, memory = [], history = [], max_steps = 8, mode = "auto" }) {
  const agentState = { deviceState: { ...device_state } }; const taskType = determineTaskType(goal); const toolNames = getToolsForTask(goal);
  const tools = AGENT_TOOLS.filter(t => toolNames.includes(t.function.name) || SERVER_SIDE_TOOLS.has(t.function.name));
  const systemPrompt = buildAutomationSystemPrompt({ deviceState: agentState.deviceState, memory });
  let messages = [{ role: "system", content: systemPrompt }, ...trimHistoryByChars(history, 10, 3500), { role: "user", content: truncateText(goal, 3000) }];
  const serverToolResults = []; const androidActions = []; const failedTools = {}; let lastAssistantText = " ";
  for (let step = 0; step < Math.max(1, Math.min(max_steps, 10)); step++) {
    const out = await groqChat({ model: pickModel({ mode, taskType }), messages, tools, tool_choice: "auto", max_tokens: 1500 });
    const msg = out.choices[0].message; const toolCalls = extractToolCalls(msg); lastAssistantText = msg.content || " ";
    if (!toolCalls.length) return { done: true, assistant_text: lastAssistantText, server_tool_results: serverToolResults, android_actions: androidActions, steps_completed: step };
    messages.push({ role: "assistant", content: lastAssistantText, tool_calls: msg.tool_calls });
    for (const tc of toolCalls) {
      const name = tc.function.name; const args = tc.function.arguments || {};
      if (SERVER_SIDE_TOOLS.has(name)) {
        try { const result = await runLocalTool(name, args, agentState); serverToolResults.push({ tool: name, arguments: args, result }); messages.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify(result) }); }
        catch (err) { if (!failedTools[name]) failedTools[name] = { attempts: 0, last_error: "  " }; failedTools[name].attempts++; failedTools[name].last_error = err.message; const errResult = { ok: false, error: err.message }; serverToolResults.push({ tool: name, arguments: args, result: errResult }); messages.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify(errResult) }); }
      } else {
        try { const deviceResult = await waitForAndroid(tc.id); if (deviceResult.observation) { if (typeof deviceResult.observation === 'string' && deviceResult.observation.length > 20) agentState.deviceState.screen_text = deviceResult.observation; if (deviceResult.data && typeof deviceResult.data === 'object') agentState.deviceState.screen_summary = JSON.stringify(deviceResult.data).slice(0, 1000); } androidActions.push({ id: tc.id, tool: name, arguments: args, device_result: deviceResult }); messages.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify(deviceResult) }); }
        catch (timeoutErr) { const timeoutResult = { status: "timeout", observation: "Android app did not respond within 60s" }; androidActions.push({ id: tc.id, tool: name, arguments: args, device_result: timeoutResult }); messages.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify(timeoutResult) }); if (!failedTools[name]) failedTools[name] = { attempts: 0, last_error: "  " }; failedTools[name].attempts++; failedTools[name].last_error = timeoutErr.message; }
      }
    }
  }
  return { done: false, assistant_text: lastAssistantText, server_tool_results: serverToolResults, android_actions: androidActions, failed_tools: failedTools, model_used: pickModel({ mode, taskType }) };
}

// ==================== SCREEN FRAME INGESTION ====================
const frameBuffer = []; const FRAME_BUFFER_SIZE = 10;

// ==================== ROUTES ====================
app.get("/", (_req, res) => {
  res.json({ name: "FRIT LLM Orchestrator & HVR Trading Engine", description: "AI brain for the FRIT Android app.", status: "online", endpoints: { core: ["/health", "/chat", "/automate", "/automation-results", "/plan"], market: ["/market/quote", "/market/batch", "/market/analyze", "/trade", "/backtest", "/calibration"], memory: ["/memory/trade"], screen: ["/screen/frame", "/screen/analyze-frame", "/screen/status"], transcribe: ["/transcribe"], utility: ["/weather"] } });
});

// HEALTH ENDPOINT (Recommendation #6)
app.get("/health", (_req, res) => {
  const state = hvr.persistence.load();
  res.json({
    status: "online", hvr_active: true, cooldowns_active: Object.keys(state.cooldowns).length, trades_logged: state.trades.length,
    data_provider_ok: !!TWELVE_DATA_KEY, mt5_bridge_ok: !!MT5_BRIDGE_URL, android_agent: "active",
    models: MODELS, pending_android_callbacks: pendingAndroidCallbacks.size, frame_buffer: frameBuffer.length, cache_entries: _cache.size,
    uptime: Math.floor(process.uptime()) + "s", timestamp: Date.now()
  });
});

// CHAT
app.post("/chat", requireAuth, async (req, res) => {
  const { message, history = [], memory = [], screen_context, mode = "auto" } = req.body || {};
  if (!message) return res.status(400).json({ error: "No message" });
  const safeHistory = trimHistoryByChars(history, 8, 3200); const safeMemory = summarizeMemory(memory, 6, 700); const safeScreenCtx = maybeTrimScreenContext(screen_context); const hasImage = !!safeScreenCtx; const model = pickModel({ hasImage, mode });
  const symMatch = String(message).match(/\b(EURUSD|GBPUSD|USDJPY|AUDUSD|XAUUSD|USDCHF|USDCAD|NZDUSD|GBPJPY|EURJPY|EURGBP|BTCUSD|ETHUSD|SOLUSD|BNBUSD|XRPUSD|DOGEUSD|ADAUSD|BTC|ETH|SOL|BNB|XRP|DOGE|ADA|USDNGN)\b/i);
  const ivMatch = String(message).match(/\b(1min|5min|15min|30min|1h|4h|1day)\b/i); const sizeMatch = String(message).match(/\b(\d{2,3})\s*(candles?|bars?|data points?)\b/i);
  if (symMatch && isMarketMessage(message) && !wantsDetailedTradeReason(message)) { try { const sym = symMatch[1].toUpperCase(); const iv = ivMatch?.[1]?.toLowerCase() || "1h"; const sz = sizeMatch ? parseInt(sizeMatch[1], 10) : null; const analysis = await analyzeSymbol(sym, iv, sz); return res.json({ text: formatConciseSignal(analysis), model_used: "local_market_formatter", concise_signal: analysis.concise_signal || null, full_analysis_available: true }); } catch (err) { console.error("[/chat concise market path]", err.message); } }
  let liveData = "  ";
  if (symMatch) { try { const sym = symMatch[1].toUpperCase(); const iv = ivMatch?.[1]?.toLowerCase() || "1h"; const sz = sizeMatch ? parseInt(sizeMatch[1], 10) : null; const a = await analyzeSymbol(sym, iv, sz); if (!a.error) { const auctionLine = a.auction ? `Auction: POC=${a.auction.poc} VAH=${a.auction.vah} VAL=${a.auction.val} | ${a.auction.note}` : "  "; liveData = ["  ", "=== LIVE MARKET DATA (fetched now) ===", a.summary, `Signal: ${a.concise_signal?.direction} | Entry ${a.concise_signal?.entry} | SL ${a.concise_signal?.sl} | TP ${a.concise_signal?.tp}`, auctionLine, `AI Opinion: ${a.concise_signal?.ai_opinion || ""}`, "  ", "INSTRUCTION: Respond with Direction, Entry, SL, TP, AI Opinion only unless the user explicitly asks for reasons or full analysis."].filter(Boolean).join("\n"); } } catch (e) { console.error("[/chat market fetch]", e.message); } }
  const systemPrompt = ["You are FRIT, a sharp AI assistant and professional trading analyst.", "When live market data is provided, use it directly. Do not guess.", "For trading replies, be concise by default — only show full reasoning when explicitly asked.", liveData, symMatch ? formatTradeMemoryForPrompt(symMatch[1].toUpperCase()) : " ", buildMemoryBlock(safeMemory)].filter(Boolean).join("\n");
  const userContent = hasImage ? [{ type: "text", text: truncateText(message, 4000) }, { type: "image_url", image_url: { url: `data:image/jpeg;base64,${safeScreenCtx}` } }] : truncateText(message, 4000);
  try { const out = await groqChat({ model, messages: [{ role: "system", content: systemPrompt }, ...safeHistory, { role: "user", content: userContent }], max_tokens: 1600 }); res.json({ text: out.choices[0].message.content, model_used: model, payload_guard: { screen_context_used: !!safeScreenCtx, memory_items_used: safeMemory.length, history_messages_used: safeHistory.length } }); } catch (err) { console.error("[/chat]", err.message); res.status(500).json({ error: "AI request failed", details: err.message }); }
});

// AUTOMATE
app.post("/automate", requireAuth, async (req, res) => { const { goal, device_state = {}, memory = [], history = [], max_steps = 8, mode = "auto" } = req.body || {}; if (!goal) return res.status(400).json({ error: "goal required" }); try { const result = await handleAutomationRequest({ goal, device_state, memory, history, max_steps, mode }); res.json(result); } catch (err) { console.error("[/automate]", err.message); res.status(500).json({ error: "Automation failed", details: err.message }); } });
app.post("/automation-results", requireAuth, (req, res) => { const { callId, status, observation } = req.body || {}; if (!callId) return res.status(400).json({ error: "callId required" }); const resolved = resolveAndroid(callId, { status: status || "ok", observation: observation || " " }); res.json({ success: resolved, callId }); });
app.post("/plan", requireAuth, async (req, res) => { const { task, device_state = {}, memory = [] } = req.body || {}; if (!task) return res.status(400).json({ error: "task required" }); const prompt = ["You are a planner for an Android AI agent.", "Return a concise numbered plan only. Do not execute tools.", "Include verification steps after major navigation actions.", "Include fallback steps for likely failure points.", "  ", "Goal:  ", truncateText(task, 2000), "  ", "Device state:  ", buildDeviceStateBlock(device_state), buildMemoryBlock(summarizeMemory(memory, 6, 700))].filter(Boolean).join("\n"); try { const out = await groqChat({ model: MODELS.fast, messages: [{ role: "system", content: "You create precise Android automation plans." }, { role: "user", content: prompt }], max_tokens: 700 }); res.json({ plan: out.choices[0].message.content, model_used: MODELS.fast }); } catch (err) { console.error("[/plan]", err.message); res.status(500).json({ error: "Planning failed", details: err.message }); } });

// SCREEN
app.post("/screen/frame", requireAuth, (req, res) => { const { frameData, timestamp, width, height } = req.body || {}; if (!frameData) return res.status(400).json({ error: "frameData required" }); frameBuffer.push({ data: frameData, timestamp: timestamp || Date.now(), width, height }); if (frameBuffer.length > FRAME_BUFFER_SIZE) frameBuffer.shift(); res.json({ status: "received", buffered: frameBuffer.length }); });
app.post("/screen/analyze-frame", requireAuth, async (req, res) => { const { prompt, frameIndex = -1 } = req.body || {}; if (!frameBuffer.length) return res.status(400).json({ error: "No frames in buffer. Android app must send frames first via /screen/frame." }); const frame = frameIndex >= 0 && frameIndex < frameBuffer.length ? frameBuffer[frameIndex] : frameBuffer.at(-1); try { const out = await groqChat({ model: MODELS.vision, messages: [{ role: "user", content: [{ type: "text", text: prompt || "Describe the UI elements and any actionable items on this screen." }, { type: "image_url", image_url: { url: `data:image/jpeg;base64,${frame.data}` } }] }], max_tokens: 500 }); res.json({ analysis: out.choices[0].message.content, frameTimestamp: frame.timestamp, frameIndex: frameIndex >= 0 ? frameIndex : frameBuffer.length - 1 }); } catch (err) { res.status(500).json({ error: "Frame analysis failed", details: err.message }); } });
app.get("/screen/status", requireAuth, (_req, res) => { res.json({ buffered_frames: frameBuffer.length, max_buffer: FRAME_BUFFER_SIZE, oldest_frame_ts: frameBuffer[0]?.timestamp || null, newest_frame_ts: frameBuffer.at(-1)?.timestamp || null }); });
app.post("/screen/clear", requireAuth, (_req, res) => { frameBuffer.length = 0; res.json({ status: "cleared" }); });

// MARKET
app.get("/market/quote", async (req, res) => { try { const symbol = String(req.query.symbol || "BTCUSD").toUpperCase(); const data = await fetchMarketPrices([symbol]); res.json(data[symbol] || { error: "Not found" }); } catch (err) { res.status(500).json({ error: "Quote fetch failed", details: err.message }); } });
app.post("/market/batch", async (req, res) => { try { const { symbols = [] } = req.body || {}; if (!Array.isArray(symbols) || !symbols.length) return res.status(400).json({ error: "symbols array required" }); res.json(await fetchMarketPrices(symbols.map(s => String(s).toUpperCase()))); } catch (err) { res.status(500).json({ error: "Batch fetch failed", details: err.message }); } });
app.all("/market/analyze", requireAuth, async (req, res) => { try { const body = req.body || {}; const query = req.query || {}; const sym = req.method === "GET" ? query.symbol : body.symbol ?? query.symbol; if (!sym) return res.status(400).json({ error: "symbol required" }); const symbol = String(sym).toUpperCase(); const interval = normalizeInterval(req.method === "GET" ? query.interval : body.interval ?? query.interval ?? "1h"); const outputsize = (req.method === "GET" ? query.outputsize : body.outputsize ?? query.outputsize) ? Number(req.method === "GET" ? query.outputsize : body.outputsize ?? query.outputsize) : null; res.json(await analyzeSymbol(symbol, interval, outputsize)); } catch (err) { console.error("[/market/analyze]", err.message); res.status(500).json({ error: "Analysis failed", details: err.message }); } });

// HVR TRADE ENDPOINT (Replaces old /trade)
app.post("/trade", requireAuth, applyRateLimit(3, 60000), async (req, res) => {
  const { symbol, action, risk_percent = 1, balance = 1000 } = req.body || {};
  if (!symbol || !["buy", "sell"].includes(action?.toLowerCase())) return res.status(400).json({ error: "symbol and action (buy/sell) required" });
  const dir = action.toUpperCase();
  if (hvr.isInCooldown(symbol.toUpperCase(), dir)) return res.status(200).json({ status: "COOLDOWN", reason: `${dir} on cooldown for ${symbol}` });
  const signal = await hvr.generateSignal(symbol.toUpperCase());
  if (signal.signal !== dir || signal.signal === "NO_TRADE") return res.status(200).json({ status: "NO_EDGE", reason: signal.reason, signal });
  const lotSize = calculateLotSize({ symbol, balance, riskPercent: risk_percent, entry: parseFloat(signal.entry), stopLoss: parseFloat(signal.sl) });
  hvr.setCooldown(symbol.toUpperCase(), dir);
  const tradeResult = await sendToMT5Bridge({ symbol: symbol.toUpperCase(), action: dir.toLowerCase(), lotSize, entry: signal.entry, sl: signal.sl, tp: signal.tp, reason: `HVR Protocol | Conf:${signal.confidence}% | ATR:${signal.atr}` });
  hvr.persistence.load().trades.push({ ...signal, lot_size: lotSize, outcome: "pending", timestamp: Date.now() }); hvr.persistence.save();
  res.json({ status: "submitted", pipeline: "hvr", ...signal, lot_size: lotSize, mt5_result: tradeResult });
});

// BACKTEST ENDPOINT (Recommendation #2)
app.post("/backtest", requireAuth, applyRateLimit(2, 120000), async (req, res) => {
  const { symbol = "EURUSD", days = 30 } = req.body || {};
  try { res.json(await hvr.backtest(symbol, days)); } catch (err) { res.status(500).json({ error: "Backtest failed", details: err.message }); }
});

// CONFIDENCE CALIBRATION (Recommendation #4)
app.get("/calibration", requireAuth, (_req, res) => { res.json(hvr.persistence.getCalibration()); });
app.post("/calibration/log", requireAuth, (req, res) => { const { predicted, actual } = req.body || {}; if (predicted == null || actual == null) return res.status(400).json({ error: "predicted and actual required" }); hvr.persistence.logConfidence(predicted, actual); res.json({ ok: true }); });

// MEMORY & UTILS
app.post("/memory/trade", requireAuth, (req, res) => { const { symbol, outcome, pattern, direction, note } = req.body || {}; if (!symbol) return res.status(400).json({ error: "symbol required" }); addTradeMemory(symbol, { outcome, pattern, direction, note }); res.json({ ok: true, memory_count: tradeMemory.get(symbol.toUpperCase())?.length }); });
app.get("/memory/trade", requireAuth, (req, res) => { const symbol = String(req.query.symbol || " ").toUpperCase(); if (!symbol) return res.status(400).json({ error: "symbol required" }); res.json({ symbol, entries: getTradeMemory(symbol) }); });
app.get("/weather", async (req, res) => { try { const city = String(req.query.city || "Lagos"); const result = await getWeather(city); res.json({ city, result }); } catch (err) { res.status(500).json({ error: "Weather failed", details: err.message }); } });
app.post("/transcribe", async (req, res) => { try { const { audio_base64, mime_type = "audio/webm" } = req.body || {}; if (!audio_base64) return res.status(400).json({ error: "audio_base64 required" }); const form = new FormData(); const buffer = Buffer.from(audio_base64, "base64"); form.append("file", buffer, { filename: "audio.webm", contentType: mime_type }); form.append("model", MODELS.whisper); const out = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", { method: "POST", headers: { Authorization: `Bearer ${GROQ_API_KEY}`, ...form.getHeaders() }, body: form }); const data = await out.json(); if (!out.ok) return res.status(500).json({ error: "Transcription failed", details: data }); res.json({ text: data.text || "  ", model_used: MODELS.whisper }); } catch (err) { console.error("[/transcribe]", err.message); res.status(500).json({ error: "Transcription failed", details: err.message }); } });

// ==================== ERROR HANDLER ====================
app.use((err, _req, res, _next) => { console.error("[Unhandled Error]", err); res.status(500).json({ error: "Internal server error", details: err?.message || "Unknown error" }); });

// ==================== START ====================
app.listen(PORT, () => {
  console.log(`╔══════════════════════════════════════════════════════════════════╗
║ 🤖 FRIT LLM Orchestrator & HVR Trading Engine                  ║
╠══════════════════════════════════════════════════════════════════╣
║ Port : ${String(PORT).padEnd(55)}║
║ ✅ Android AI Agent (State Machine, Tools, Screen, Groq)        ║
║ ✅ Modified Claude HVR Protocol (H4→H1→M15→M5)                 ║
║ ✅ OHLC Volume Proxy & Direction-Aware Cooldowns                ║
║ ✅ Persistent Storage & Confidence Calibration                  ║
║ ✅ /backtest & /health endpoints                                ║
║ ✅ Rate Limiting on /trade & /backtest                          ║
╚══════════════════════════════════════════════════════════════════╝`);
});
