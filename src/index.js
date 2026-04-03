import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import dotenv from "dotenv";
import fetch from "node-fetch";
import FormData from "form-data";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT || 8787);

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const TWELVE_DATA_KEY = process.env.TWELVE_DATA_KEY || "";
const SANDBOX_URL = process.env.SANDBOX_URL || "http://127.0.0.1:8790";

if (!GROQ_API_KEY) {
  console.error("GROQ_API_KEY missing");
  process.exit(1);
}

app.set("trust proxy", 1);
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: "*", methods: ["GET", "POST", "OPTIONS"] }));
app.use(morgan(process.env.NODE_ENV === "production" ? "combined" : "dev"));

/* ──────────────────────────────────────────────────────────────
   MODELS
────────────────────────────────────────────────────────────── */
const MODELS = {
  vision: "meta-llama/llama-4-scout-17b-16e-instruct",
  conversation: "compound-beta",
  tools: "llama-3.3-70b-versatile",
  fast: "llama-3.1-8b-instant",
  whisper: "whisper-large-v3",
};

/* ──────────────────────────────────────────────────────────────
   CACHE
────────────────────────────────────────────────────────────── */
const _cache = new Map();

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

function cacheStats() {
  return { entries: _cache.size };
}

/* ──────────────────────────────────────────────────────────────
   HELPERS
────────────────────────────────────────────────────────────── */
function pickModel({ hasImage = false, mode = "auto" } = {}) {
  if (mode === "vision" || hasImage) return MODELS.vision;
  if (mode === "fast") return MODELS.fast;
  if (mode === "tools") return MODELS.tools;
  return MODELS.conversation;
}

function safeJsonParse(input, fallback = null) {
  try {
    return JSON.parse(input);
  } catch {
    return fallback;
  }
}

function truncateText(text, maxLen = 1200) {
  const s = String(text || "");
  return s.length > maxLen ? s.slice(0, maxLen) : s;
}

function summarizeMemory(memory = [], maxItems = 6, maxChars = 700) {
  if (!Array.isArray(memory) || memory.length === 0) return [];
  const picked = [];
  let used = 0;
  for (const item of memory) {
    const s = truncateText(item, 180);
    if (!s) continue;
    if (picked.length >= maxItems) break;
    if (used + s.length > maxChars) break;
    picked.push(s);
    used += s.length;
  }
  return picked;
}

function trimHistoryByChars(history = [], maxMessages = 8, maxChars = 3200) {
  const arr = Array.isArray(history) ? history.slice(-maxMessages) : [];
  const out = [];
  let used = 0;

  for (let i = arr.length - 1; i >= 0; i--) {
    const msg = arr[i];
    let content = msg?.content ?? "";
    if (Array.isArray(content)) {
      content = content
        .map((p) => (typeof p?.text === "string" ? p.text : ""))
        .join(" ");
    }
    const s = truncateText(String(content), 900);
    const cost = s.length + 40;
    if (used + cost > maxChars) continue;
    out.unshift({ role: msg.role, content: s });
    used += cost;
  }
  return out;
}

function maybeTrimScreenContext(screenContext, maxBase64Chars = 350000) {
  if (!screenContext || typeof screenContext !== "string") return null;
  if (screenContext.length <= maxBase64Chars) return screenContext;
  return null;
}

async function runSandbox(args = {}) {
  const res = await fetch(`${SANDBOX_URL}/sandbox/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.details || data?.error || "Sandbox failed");
  return data;
}

async function groqChat({
  model,
  messages,
  tools = null,
  temperature = 0.3,
  max_tokens = 1600,
  tool_choice = "auto",
}) {
  const body = { model, messages, temperature, max_tokens };
  if (tools?.length) {
    body.tools = tools;
    body.tool_choice = tool_choice;
  }

  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await res.json();

  if (!res.ok) {
    console.error("[Groq Error]", model, JSON.stringify(data?.error || data));
    throw new Error(data?.error?.message || JSON.stringify(data));
  }

  return data;
}

function extractToolCalls(choiceMessage) {
  const toolCalls = choiceMessage?.tool_calls || [];
  return toolCalls.map((tc) => ({
    id: tc.id,
    type: tc.type,
    function: {
      name: tc.function?.name,
      arguments: safeJsonParse(tc.function?.arguments || "{}", {}),
      raw_arguments: tc.function?.arguments || "{}",
    },
  }));
}

function resolveOutputSize(interval) {
  return (
    {
      "1min": 500,
      "5min": 288,
      "15min": 200,
      "30min": 150,
      "1h": 168,
      "4h": 120,
      "1day": 100,
      "1week": 52,
    }[interval] || 168
  );
}

function toBinanceInterval(interval) {
  return (
    {
      "1min": "1m",
      "5min": "5m",
      "15min": "15m",
      "30min": "30m",
      "1h": "1h",
      "4h": "4h",
      "1day": "1d",
      "1week": "1w",
    }[interval] || "1h"
  );
}

function normalizeInterval(v) {
  const iv = String(v || "1h").toLowerCase().trim();
  const allowed = ["1min", "5min", "15min", "30min", "1h", "4h", "1day", "1week"];
  return allowed.includes(iv) ? iv : "1h";
}

/* ──────────────────────────────────────────────────────────────
   MARKET MESSAGE HELPERS (NEW)
────────────────────────────────────────────────────────────── */
function isMarketMessage(message = "") {
  const text = String(message || "").toLowerCase();
  return (
    /\b(eurusd|gbpusd|usdjpy|audusd|xauusd|usdchf|usdcad|nzdusd|gbpjpy|eurjpy|eurgbp|btcusd|ethusd|solusd|bnbusd|xrpusd|dogeusd|adausd|btc|eth|sol|bnb|xrp|doge|ada|usdngn)\b/i.test(text) ||
    /\b(price|analysis|signal|buy|sell|entry|tp|sl|market)\b/i.test(text)
  );
}

function wantsDetailedTradeReason(message = "") {
  const text = String(message || "").toLowerCase();
  return [
    "why",
    "reason",
    "explain",
    "show full analysis",
    "full analysis",
    "show structure",
    "details",
    "detailed",
    "breakdown",
    "confidence",
    "pattern",
    "strategy",
    "structure",
    "regime",
  ].some((k) => text.includes(k));
}

function formatConciseSignal(analysis) {
  if (!analysis || analysis.error) {
    return analysis?.error || "No market data available.";
  }
  const cs = analysis.concise_signal || {};
  const direction = cs.direction || analysis.direction || "NEUTRAL";
  const entry = cs.entry || analysis.trade_plan?.entry_zone || "N/A";
  const sl = cs.sl || analysis.trade_plan?.invalidation || "N/A";
  const tp = cs.tp || analysis.trade_plan?.tp1 || "N/A";
  const opinion = cs.ai_opinion || analysis.ai_opinion || "No strong edge right now.";
  return [
    `${analysis.symbol} ${String(analysis.interval || "").toUpperCase()}`.trim(),
    `Direction: ${direction}`,
    `Entry: ${entry}`,
    `SL: ${sl}`,
    `TP: ${tp}`,
    `AI Opinion: ${opinion}`,
  ].join("\n");
}

/* ──────────────────────────────────────────────────────────────
   MARKET DATA
────────────────────────────────────────────────────────────── */
const TD_SYMBOLS = {
  EURUSD: "EUR/USD",
  GBPUSD: "GBP/USD",
  USDJPY: "USD/JPY",
  AUDUSD: "AUD/USD",
  USDCHF: "USD/CHF",
  USDCAD: "USD/CAD",
  NZDUSD: "NZD/USD",
  XAUUSD: "XAU/USD",
  XAGUSD: "XAG/USD",
  GBPJPY: "GBP/JPY",
  EURJPY: "EUR/JPY",
  EURGBP: "EUR/GBP",
  BTC: "BTC/USD",
  ETH: "ETH/USD",
  SOL: "SOL/USD",
  BNB: "BNB/USD",
  XRP: "XRP/USD",
  DOGE: "DOGE/USD",
  ADA: "ADA/USD",
  BTCUSD: "BTC/USD",
  ETHUSD: "ETH/USD",
  SOLUSD: "SOL/USD",
  BNBUSD: "BNB/USD",
  XRPUSD: "XRP/USD",
  DOGEUSD: "DOGE/USD",
  ADAUSD: "ADA/USD",
};

const CRYPTO_SET = new Set([
  "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA",
  "BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD", "DOGEUSD", "ADAUSD",
]);

const COINGECKO_IDS = {
  BTC: "bitcoin",
  ETH: "ethereum",
  SOL: "solana",
  BNB: "binancecoin",
  XRP: "ripple",
  DOGE: "dogecoin",
  ADA: "cardano",
  BTCUSD: "bitcoin",
  ETHUSD: "ethereum",
  SOLUSD: "solana",
  BNBUSD: "binancecoin",
  XRPUSD: "ripple",
  DOGEUSD: "dogecoin",
  ADAUSD: "cardano",
};

const BINANCE_SYM = {
  BTC: "BTCUSDT",
  ETH: "ETHUSDT",
  SOL: "SOLUSDT",
  BNB: "BNBUSDT",
  XRP: "XRPUSDT",
  DOGE: "DOGEUSDT",
  ADA: "ADAUSDT",
  BTCUSD: "BTCUSDT",
  ETHUSD: "ETHUSDT",
  SOLUSD: "SOLUSDT",
  BNBUSD: "BNBUSDT",
  XRPUSD: "XRPUSDT",
  DOGEUSD: "DOGEUSDT",
  ADAUSD: "ADAUSDT",
};

const FRANKFURTER_MAP = {
  EURUSD: { base: "EUR", quote: "USD" },
  GBPUSD: { base: "GBP", quote: "USD" },
  USDJPY: { base: "USD", quote: "JPY" },
  AUDUSD: { base: "AUD", quote: "USD" },
  USDCHF: { base: "USD", quote: "CHF" },
  USDCAD: { base: "USD", quote: "CAD" },
  NZDUSD: { base: "NZD", quote: "USD" },
  EURGBP: { base: "EUR", quote: "GBP" },
  EURJPY: { base: "EUR", quote: "JPY" },
  GBPJPY: { base: "GBP", quote: "JPY" },
};

async function fetchCandles(symbol, interval = "1h", outputsize = null) {
  const sym = String(symbol || "").toUpperCase();
  const iv = normalizeInterval(interval);
  const size = outputsize || resolveOutputSize(iv);
  const ck = `candles:${sym}:${iv}:${size}`;

  const cached = cacheGet(ck);
  if (cached) return cached;

  if (TWELVE_DATA_KEY && TD_SYMBOLS[sym]) {
    try {
      const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(
        TD_SYMBOLS[sym]
      )}&interval=${iv}&outputsize=${size}&apikey=${TWELVE_DATA_KEY}`;

      const res = await fetch(url);
      const data = await res.json();

      if (data.status !== "error" && data.values?.length >= 10) {
        const candles = data.values.reverse().map((v) => ({
          time: new Date(v.datetime).getTime(),
          open: parseFloat(v.open),
          high: parseFloat(v.high),
          low: parseFloat(v.low),
          close: parseFloat(v.close),
          volume: parseFloat(v.volume || 0),
        }));
        cacheSet(ck, candles, 60000);
        return candles;
      } else {
        console.error("[TwelveData candles]", sym, data?.message || "Unknown error");
      }
    } catch (e) {
      console.error("[TwelveData candles fetch]", sym, e.message);
    }
  }

  if (BINANCE_SYM[sym]) {
    try {
      const url = `https://api.binance.com/api/v3/klines?symbol=${BINANCE_SYM[sym]}&interval=${toBinanceInterval(
        iv
      )}&limit=${Math.min(size, 1000)}`;

      const res = await fetch(url);
      const arr = await res.json();

      if (Array.isArray(arr) && arr.length >= 10) {
        const candles = arr.map((k) => ({
          time: k[0],
          open: parseFloat(k[1]),
          high: parseFloat(k[2]),
          low: parseFloat(k[3]),
          close: parseFloat(k[4]),
          volume: parseFloat(k[5]),
        }));
        cacheSet(ck, candles, 60000);
        return candles;
      }
    } catch (e) {
      console.error("[Binance candles]", sym, e.message);
    }
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
      const res = await fetch(
        `https://api.twelvedata.com/price?symbol=${encodeURIComponent(
          TD_SYMBOLS[sym]
        )}&apikey=${TWELVE_DATA_KEY}`
      );
      const data = await res.json();
      if (data.price) result = { price: parseFloat(data.price), source: "twelvedata" };
    } catch (e) {
      console.error("[TwelveData spot]", sym, e.message);
    }
  }

  if (!result && COINGECKO_IDS[sym]) {
    try {
      const res = await fetch(
        `https://api.coingecko.com/api/v3/simple/price?ids=${COINGECKO_IDS[sym]}&vs_currencies=usd&include_24hr_change=true`
      );
      const data = await res.json();
      const id = COINGECKO_IDS[sym];
      if (data[id]) {
        result = {
          price: data[id].usd,
          change24h: data[id].usd_24h_change,
          source: "coingecko",
        };
      }
    } catch (e) {
      console.error("[CoinGecko spot]", sym, e.message);
    }
  }

  if (!result && FRANKFURTER_MAP[sym]) {
    try {
      const { base, quote } = FRANKFURTER_MAP[sym];
      const res = await fetch(`https://api.frankfurter.app/latest?from=${base}&to=${quote}`);
      const data = await res.json();
      const rate = data.rates?.[quote];
      if (rate) result = { price: parseFloat(rate), source: "frankfurter" };
    } catch (e) {
      console.error("[Frankfurter spot]", sym, e.message);
    }
  }

  if (!result && sym === "USDNGN") {
    try {
      const res = await fetch("https://open.er-api.com/v6/latest/USD");
      const data = await res.json();
      if (data.rates?.NGN) result = { price: data.rates.NGN, source: "er-api" };
    } catch (e) {
      console.error("[ER-API USDNGN]", e.message);
    }
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
  if (cached !== null) return cached;

  try {
    const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(
      TD_SYMBOLS[sym]
    )}&interval=1day&outputsize=2&apikey=${TWELVE_DATA_KEY}`;

    const res = await fetch(url);
    const data = await res.json();

    if (data.values?.length >= 2) {
      const today = parseFloat(data.values[0].close);
      const yesterday = parseFloat(data.values[1].close);
      const delta = ((today - yesterday) / yesterday) * 100;
      cacheSet(ck, delta, 300000);
      return delta;
    }
  } catch (e) {
    console.error("[fetch24hDelta]", sym, e.message);
  }

  return null;
}

async function fetchMarketPrices(symbols = []) {
  const result = {};
  await Promise.all(
    symbols.map(async (sym) => {
      const s = String(sym || "").toUpperCase();
      const spot = await fetchSpotPrice(s);
      if (!spot) {
        result[s] = {
          symbol: s,
          price: 0,
          change24h: 0,
          currency: "USD",
          error: "Not found",
        };
        return;
      }

      const change24h =
        spot.change24h !== undefined && spot.change24h !== 0
          ? spot.change24h
          : (await fetch24hDelta(s)) ?? 0;

      result[s] = {
        symbol: s,
        price: spot.price,
        change24h,
        currency: s.includes("NGN") ? "NGN" : "USD",
        source: spot.source,
      };
    })
  );
  return result;
}

/* ──────────────────────────────────────────────────────────────
   ANALYSIS ENGINE
────────────────────────────────────────────────────────────── */
function calcEMA(closes, period) {
  if (closes.length < period) return [];
  const k = 2 / (period + 1);
  let prev = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;
  const result = [prev];
  for (let i = period; i < closes.length; i++) {
    prev = closes[i] * k + prev * (1 - k);
    result.push(prev);
  }
  return result;
}

function calcRSI(closes, period = 14) {
  if (closes.length < period + 1) return 50;
  const changes = closes
    .slice(-(period + 1))
    .map((v, i, a) => (i > 0 ? v - a[i - 1] : 0))
    .slice(1);
  const avgGain = changes.map((c) => (c > 0 ? c : 0)).reduce((a, b) => a + b, 0) / period;
  const avgLoss = changes.map((c) => (c < 0 ? -c : 0)).reduce((a, b) => a + b, 0) / period;
  if (avgLoss === 0) return 100;
  return 100 - 100 / (1 + avgGain / avgLoss);
}

function calcMACD(closes) {
  const ema12 = calcEMA(closes, 12);
  const ema26 = calcEMA(closes, 26);
  if (!ema12.length || !ema26.length) return { macd: 0, signal: 0, hist: 0 };
  const offset = ema12.length - ema26.length;
  const macdLine = ema26.map((v, i) => ema12[i + offset] - v);
  const signalLine = calcEMA(macdLine, 9);
  const last = macdLine.at(-1);
  const sig = signalLine.at(-1) ?? 0;
  return { macd: last, signal: sig, hist: last - sig };
}

function calcSR(candles) {
  const r = candles.slice(-20);
  return {
    support: Math.min(...r.map((c) => c.low)),
    resistance: Math.max(...r.map((c) => c.high)),
  };
}

function calcBB(closes, period = 20, mult = 2) {
  if (closes.length < period) return null;
  const s = closes.slice(-period);
  const mean = s.reduce((a, b) => a + b, 0) / period;
  const std = Math.sqrt(s.map((v) => (v - mean) ** 2).reduce((a, b) => a + b, 0) / period);
  return {
    upper: mean + mult * std,
    middle: mean,
    lower: mean - mult * std,
  };
}

function calcATR(candles, period = 14) {
  if (!candles || candles.length < period + 1) return 0;
  const trs = [];
  for (let i = 1; i < candles.length; i++) {
    const c = candles[i];
    const p = candles[i - 1];
    const tr = Math.max(
      c.high - c.low,
      Math.abs(c.high - p.close),
      Math.abs(c.low - p.close)
    );
    trs.push(tr);
  }
  const recent = trs.slice(-period);
  return recent.reduce((a, b) => a + b, 0) / recent.length;
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

  const aBull = a.close > a.open;
  const aBear = a.close < a.open;
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

function analyzeStructure(candles, price) {
  const closes = candles.map((c) => c.close);
  const ema20 = calcEMA(closes, 20).at(-1) ?? price;
  const ema50 = calcEMA(closes, 50).at(-1) ?? price;
  const ema200 = calcEMA(closes, 200).at(-1) ?? price;
  const swings = findSwings(candles);
  const lastHigh = swings.highs.at(-1)?.price ?? null;
  const prevHigh = swings.highs.at(-2)?.price ?? null;
  const lastLow = swings.lows.at(-1)?.price ?? null;
  const prevLow = swings.lows.at(-2)?.price ?? null;

  let trend = "neutral";
  if (price > ema20 && ema20 > ema50) trend = "up";
  else if (price < ema20 && ema20 < ema50) trend = "down";

  const bosBull = lastHigh != null && prevHigh != null && lastHigh > prevHigh;
  const bosBear = lastLow != null && prevLow != null && lastLow < prevLow;

  return {
    trend,
    bos_bullish: bosBull,
    bos_bearish: bosBear,
    last_swing_high: lastHigh,
    last_swing_low: lastLow,
    ema20,
    ema50,
    ema200,
  };
}

function analyzeVolatility(candles, price) {
  const atr = calcATR(candles, 14);
  const closes = candles.map((c) => c.close);
  const bb = calcBB(closes, 20, 2);
  const bbWidth = bb ? (bb.upper - bb.lower) / (bb.middle || price || 1) : 0;
  let regime = "normal";
  if (atr / (price || 1) < 0.0015) regime = "compressed";
  else if (atr / (price || 1) > 0.005) regime = "expanding";

  return {
    atr,
    bb_width: bbWidth,
    regime,
  };
}

function scoreSetup({ structure, volatility, rsi, macd, macdSig, hist, price, support, resistance, patterns }) {
  let bull = 0;
  let bear = 0;

  if (structure.trend === "up") bull += 2;
  if (structure.trend === "down") bear += 2;

  if (structure.bos_bullish) bull += 2;
  if (structure.bos_bearish) bear += 2;

  if (rsi > 52 && rsi < 70) bull += 1;
  if (rsi < 48 && rsi > 30) bear += 1;

  if (macd > macdSig) bull += 1;
  else bear += 1;

  if (hist > 0) bull += 1;
  else bear += 1;

  if (price > support && (price - support) / (price || 1) < 0.003) bull += 1;
  if (price < resistance && (resistance - price) / (price || 1) < 0.003) bear += 1;

  if (patterns.includes("bullish_engulfing") || patterns.includes("pinbar_bullish")) bull += 1;
  if (patterns.includes("bearish_engulfing") || patterns.includes("pinbar_bearish")) bear += 1;

  if (volatility.regime === "compressed") {
    bull -= 0.5;
    bear -= 0.5;
  }

  let bias = "neutral";
  const diff = bull - bear;
  if (diff >= 2) bias = "bullish";
  else if (diff <= -2) bias = "bearish";

  let confidence = 50 + diff * 8;
  confidence = Math.max(5, Math.min(95, Math.round(confidence)));

  return { bull_score: bull, bear_score: bear, bias, confidence };
}

function buildTradePlan({ bias, price, support, resistance, atr, dp }) {
  if (!price || !atr) {
    return {
      entry_zone: null,
      invalidation: null,
      tp1: null,
      tp2: null,
      risk_state: "unknown",
    };
  }

  if (bias === "bullish") {
    const entry1 = price - atr * 0.15;
    const entry2 = price + atr * 0.15;
    const invalidation = support > 0 ? support - atr * 0.25 : price - atr * 1.2;
    const tp1 = resistance > 0 ? resistance : price + atr * 1.2;
    const tp2 = resistance > 0 ? resistance + atr * 0.8 : price + atr * 2.2;
    return {
      entry_zone: `${entry1.toFixed(dp)}-${entry2.toFixed(dp)}`,
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
      entry_zone: `${entry1.toFixed(dp)}-${entry2.toFixed(dp)}`,
      invalidation: invalidation.toFixed(dp),
      tp1: tp1.toFixed(dp),
      tp2: tp2.toFixed(dp),
      risk_state: "acceptable",
    };
  }

  return {
    entry_zone: null,
    invalidation: null,
    tp1: null,
    tp2: null,
    risk_state: "no_trade",
  };
}

async function analyzeSymbol(symbol, interval = "1h", customSize = null) {
  const sym = String(symbol || "").toUpperCase();
  const iv = normalizeInterval(interval);
  const ck = `analysis:${sym}:${iv}:${customSize || "auto"}`;

  const cached = cacheGet(ck);
  if (cached) return cached;

  const [candles, spot] = await Promise.all([
    fetchCandles(sym, iv, customSize),
    fetchSpotPrice(sym),
  ]);

  if (!candles && !spot) {
    return { symbol: sym, error: `No data for ${sym}` };
  }

  const price = spot?.price ?? candles?.at(-1)?.close ?? 0;
  if (!candles || candles.length < 30) {
    return {
      symbol: sym,
      price,
      source: spot?.source,
      analysis: "Insufficient candle data",
      candleCount: candles?.length ?? 0,
    };
  }

  const closes = candles.map((c) => c.close);
  const rsi = calcRSI(closes);
  const { macd, signal: macdSig, hist } = calcMACD(closes);
  const { support, resistance } = calcSR(candles);
  const bb = calcBB(closes);
  const structure = analyzeStructure(candles, price);
  const volatility = analyzeVolatility(candles, price);
  const patterns = detectCandlePattern(candles);
  const score = scoreSetup({
    structure,
    volatility,
    rsi,
    macd,
    macdSig,
    hist,
    price,
    support,
    resistance,
    patterns,
  });

  let direction = "NEUTRAL";
  if (rsi >= 70) direction = "OVERBOUGHT";
  else if (rsi <= 30) direction = "OVERSOLD";
  else if (score.bias === "bullish") direction = "BULLISH";
  else if (score.bias === "bearish") direction = "BEARISH";

  let strength = "WEAK";
  if (score.confidence >= 75) strength = "STRONG";
  else if (score.confidence >= 60) strength = "MODERATE";

  const isCrypto = CRYPTO_SET.has(sym);
  const dp = isCrypto || sym === "XAUUSD" ? 2 : 5;

  const recentCandles = candles
    .slice(-20)
    .map((c) => `O:${c.open.toFixed(dp)} H:${c.high.toFixed(dp)} L:${c.low.toFixed(dp)} C:${c.close.toFixed(dp)}`)
    .join(" | ");

  const trade_plan = buildTradePlan({
    bias: score.bias,
    price,
    support,
    resistance,
    atr: volatility.atr,
    dp,
  });

  const aiOpinion =
    direction === "NEUTRAL"
      ? "No clear edge right now. Better to wait for cleaner structure or stronger confirmation."
      : `${direction} bias with ${strength.toLowerCase()} confidence. Structure=${structure.trend}, volatility=${volatility.regime}, patterns=${patterns.join(", ") || "none"}.`;

  const concise_signal = {
    direction,
    entry: trade_plan.entry_zone || "N/A",
    sl: trade_plan.invalidation || "N/A",
    tp: trade_plan.tp1 || "N/A",
    ai_opinion: aiOpinion,
  };

  const result = {
    symbol: sym,
    price,
    direction,
    strength,
    interval: iv,
    candleCount: candles.length,
    source: spot?.source ?? "binance",

    rsi: +rsi.toFixed(1),
    macd: +macd.toFixed(6),
    macd_signal: macd > macdSig ? "bullish" : "bearish",
    ema20: +structure.ema20.toFixed(dp),
    ema50: +structure.ema50.toFixed(dp),
    ema200: +structure.ema200.toFixed(dp),

    support: +support.toFixed(dp),
    resistance: +resistance.toFixed(dp),
    bb_upper: bb ? +bb.upper.toFixed(dp) : null,
    bb_middle: bb ? +bb.middle.toFixed(dp) : null,
    bb_lower: bb ? +bb.lower.toFixed(dp) : null,

    regime: structure.trend === "neutral" ? "range_or_mixed" : "trend",
    confidence: score.confidence,
    structure: {
      trend: structure.trend,
      bos_bullish: structure.bos_bullish,
      bos_bearish: structure.bos_bearish,
      last_swing_high: structure.last_swing_high ? +structure.last_swing_high.toFixed(dp) : null,
      last_swing_low: structure.last_swing_low ? +structure.last_swing_low.toFixed(dp) : null,
    },
    volatility: {
      atr: +volatility.atr.toFixed(dp),
      bb_width: +volatility.bb_width.toFixed(6),
      regime: volatility.regime,
    },
    patterns,
    trade_plan,
    concise_signal,
    recent_candles: recentCandles,
    ai_opinion: aiOpinion,
    summary: `${sym} @${price.toFixed(dp)} | ${direction}(${strength}) | Confidence:${score.confidence} | Trend:${structure.trend} | RSI:${rsi.toFixed(
      1
    )} | S:${support.toFixed(dp)} R:${resistance.toFixed(dp)} [${candles.length} candles ${iv}]`,
  };

  cacheSet(ck, result, 60000);
  return result;
}

/* ──────────────────────────────────────────────────────────────
   WEB SEARCH / WEATHER
────────────────────────────────────────────────────────────── */
async function webSearch(query) {
  try {
    const res = await fetch(
      `https://api.duckduckgo.com/?q=${encodeURIComponent(
        query
      )}&format=json&no_html=1&skip_disambig=1`
    );
    const d = await res.json();
    return d.AbstractText || d.Answer || d.RelatedTopics?.[0]?.Text || "No result found.";
  } catch (e) {
    console.error("[webSearch]", e.message);
    return "Search failed: " + e.message;
  }
}

async function getWeather(city = "Lagos") {
  try {
    const geo = await (
      await fetch(
        `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`
      )
    ).json();

    const loc = geo.results?.[0];
    if (!loc) return "City not found";

    const weather = await (
      await fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${loc.latitude}&longitude=${loc.longitude}&current_weather=true&timezone=auto`
      )
    ).json();

    const cw = weather.current_weather;
    return `${loc.name}: ${cw.temperature}C, wind ${cw.windspeed}km/h, ${
      cw.weathercode <= 1 ? "Clear" : cw.weathercode <= 3 ? "Cloudy" : "Rainy"
    }`;
  } catch (e) {
    console.error("[getWeather]", e.message);
    return "Weather unavailable";
  }
}

/* ──────────────────────────────────────────────────────────────
   TOOL DEFINITIONS
────────────────────────────────────────────────────────────── */
const AGENT_TOOLS = [
  {
    type: "function",
    function: {
      name: "open_app",
      description: "Launch any installed app by name.",
      parameters: { type: "object", properties: { app_name: { type: "string" } }, required: ["app_name"] },
    },
  },
  {
    type: "function",
    function: {
      name: "open_url",
      description: "Open a full URL in browser.",
      parameters: { type: "object", properties: { url: { type: "string" } }, required: ["url"] },
    },
  },
  {
    type: "function",
    function: {
      name: "get_current_app",
      description: "Return the foreground package/app name and label.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "get_current_activity",
      description: "Return current Android activity/screen if available.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "press_back",
      description: "Press Android back button.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "press_home",
      description: "Press Android home button.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "open_recents",
      description: "Open recent apps screen.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "open_notifications",
      description: "Open notification shade.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "read_screen",
      description: "Read visible text, labels, and content descriptions from current screen.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "read_screen_structured",
      description: "Return structured UI dump.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "find_element",
      description: "Find element by text, hint, id, class or semantic description.",
      parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
    },
  },
  {
    type: "function",
    function: {
      name: "take_screenshot",
      description: "Capture a screenshot and return reference/metadata.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "analyze_screenshot",
      description: "Analyze screenshot visually for custom UI or icons.",
      parameters: { type: "object", properties: { prompt: { type: "string" } }, required: ["prompt"] },
    },
  },
  {
    type: "function",
    function: {
      name: "tap_button",
      description: "Tap an element by exact visible label or content description.",
      parameters: { type: "object", properties: { label: { type: "string" } }, required: ["label"] },
    },
  },
  {
    type: "function",
    function: {
      name: "tap_coordinates",
      description: "Tap screen by exact x/y coordinates.",
      parameters: { type: "object", properties: { x: { type: "number" }, y: { type: "number" } }, required: ["x", "y"] },
    },
  },
  {
    type: "function",
    function: {
      name: "double_tap",
      description: "Double tap by label or coordinates.",
      parameters: { type: "object", properties: { label: { type: "string" }, x: { type: "number" }, y: { type: "number" } } },
    },
  },
  {
    type: "function",
    function: {
      name: "long_press",
      description: "Long press by label or coordinates.",
      parameters: { type: "object", properties: { label: { type: "string" }, x: { type: "number" }, y: { type: "number" }, duration_ms: { type: "number" } } },
    },
  },
  {
    type: "function",
    function: {
      name: "scroll",
      description: "Scroll the current screen in a direction.",
      parameters: { type: "object", properties: { direction: { type: "string", enum: ["up", "down", "left", "right"] } } },
    },
  },
  {
    type: "function",
    function: {
      name: "swipe",
      description: "Swipe using exact coordinates.",
      parameters: {
        type: "object",
        properties: {
          startX: { type: "number" },
          startY: { type: "number" },
          endX: { type: "number" },
          endY: { type: "number" },
          duration_ms: { type: "number" },
        },
        required: ["startX", "startY", "endX", "endY"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "drag_and_drop",
      description: "Drag from one coordinate to another.",
      parameters: {
        type: "object",
        properties: {
          startX: { type: "number" },
          startY: { type: "number" },
          endX: { type: "number" },
          endY: { type: "number" },
          duration_ms: { type: "number" },
        },
        required: ["startX", "startY", "endX", "endY"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "focus_field",
      description: "Focus a text field by label, hint, content-desc, or query.",
      parameters: { type: "object", properties: { field: { type: "string" } }, required: ["field"] },
    },
  },
  {
    type: "function",
    function: {
      name: "type_text",
      description: "Type text into current or targeted field.",
      parameters: { type: "object", properties: { value: { type: "string" }, field: { type: "string" } }, required: ["value"] },
    },
  },
  {
    type: "function",
    function: {
      name: "clear_text",
      description: "Clear current or targeted input field.",
      parameters: { type: "object", properties: { field: { type: "string" } } },
    },
  },
  {
    type: "function",
    function: {
      name: "paste_text",
      description: "Paste text via clipboard into current or targeted field.",
      parameters: { type: "object", properties: { value: { type: "string" }, field: { type: "string" } }, required: ["value"] },
    },
  },
  {
    type: "function",
    function: {
      name: "press_enter",
      description: "Press Enter / Done / Search on keyboard.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "hide_keyboard",
      description: "Hide the soft keyboard.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "toggle_wifi",
      description: "Toggle Wi-Fi on or off if Android side supports it.",
      parameters: { type: "object", properties: { enabled: { type: "boolean" } }, required: ["enabled"] },
    },
  },
  {
    type: "function",
    function: {
      name: "toggle_bluetooth",
      description: "Toggle Bluetooth on or off if Android side supports it.",
      parameters: { type: "object", properties: { enabled: { type: "boolean" } }, required: ["enabled"] },
    },
  },
  {
    type: "function",
    function: {
      name: "set_volume",
      description: "Set media volume level 0-100.",
      parameters: { type: "object", properties: { level: { type: "number" } }, required: ["level"] },
    },
  },
  {
    type: "function",
    function: {
      name: "set_brightness",
      description: "Set screen brightness level 0-100.",
      parameters: { type: "object", properties: { level: { type: "number" } }, required: ["level"] },
    },
  },
  {
    type: "function",
    function: {
      name: "open_app_settings",
      description: "Open Android settings page for a specific app.",
      parameters: { type: "object", properties: { app_name: { type: "string" } }, required: ["app_name"] },
    },
  },
  {
    type: "function",
    function: {
      name: "grant_permission_if_prompted",
      description: "Handle common Android permission prompts.",
      parameters: { type: "object", properties: { allow: { type: "boolean" } }, required: ["allow"] },
    },
  },
  {
    type: "function",
    function: {
      name: "make_call",
      description: "Call a contact or phone number.",
      parameters: { type: "object", properties: { contact_name: { type: "string" }, phone_number: { type: "string" } } },
    },
  },
  {
    type: "function",
    function: {
      name: "send_whatsapp",
      description: "Open WhatsApp for a contact/message workflow.",
      parameters: { type: "object", properties: { contact_name: { type: "string" }, message: { type: "string" } }, required: ["contact_name"] },
    },
  },
  {
    type: "function",
    function: {
      name: "send_sms",
      description: "Open SMS composer for a contact.",
      parameters: { type: "object", properties: { contact_name: { type: "string" }, message: { type: "string" } }, required: ["contact_name"] },
    },
  },
  {
    type: "function",
    function: {
      name: "set_alarm",
      description: "Set an alarm at a given time.",
      parameters: { type: "object", properties: { label: { type: "string" }, time: { type: "string" } }, required: ["time"] },
    },
  },
  {
    type: "function",
    function: {
      name: "set_timer",
      description: "Start a countdown timer.",
      parameters: { type: "object", properties: { duration: { type: "string" } }, required: ["duration"] },
    },
  },
  {
    type: "function",
    function: {
      name: "play_music",
      description: "Play music on Spotify, YouTube, or supported music app.",
      parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
    },
  },
  {
    type: "function",
    function: {
      name: "navigate_to",
      description: "Open Maps and navigate to destination.",
      parameters: { type: "object", properties: { destination: { type: "string" } }, required: ["destination"] },
    },
  },
  {
    type: "function",
    function: {
      name: "take_photo",
      description: "Open camera and take photo.",
      parameters: { type: "object", properties: { front_camera: { type: "boolean" } } },
    },
  },
  {
    type: "function",
    function: {
      name: "search_web",
      description: "Search current web information.",
      parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
    },
  },
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get current weather for city.",
      parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    },
  },
  {
    type: "function",
    function: {
      name: "get_market_data",
      description: "Get live spot price for forex/crypto symbol.",
      parameters: { type: "object", properties: { symbol: { type: "string" } }, required: ["symbol"] },
    },
  },
  {
    type: "function",
    function: {
      name: "analyze_market",
      description: "Get market analysis with structure, volatility, indicators, patterns, and trade plan.",
      parameters: {
        type: "object",
        properties: {
          symbol: { type: "string" },
          interval: { type: "string" },
          outputsize: { type: "number" },
        },
        required: ["symbol"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "place_trade",
      description: "Place a buy or sell trade on MT5 or broker app via Android side integration.",
      parameters: {
        type: "object",
        properties: {
          symbol: { type: "string" },
          action: { type: "string", enum: ["buy", "sell"] },
          volume: { type: "number" },
          sl: { type: "number" },
          tp: { type: "number" },
        },
        required: ["symbol", "action", "volume"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_code",
      description: "Run Python or JavaScript code in a controlled sandbox. Use for calculations, data processing, or any task that benefits from code execution.",
      parameters: {
        type: "object",
        properties: {
          language: { type: "string", enum: ["python", "javascript"] },
          code: { type: "string" },
          stdin: { type: "string" },
          timeout_ms: { type: "number" },
        },
        required: ["language", "code"],
      },
    },
  },
];

/* ──────────────────────────────────────────────────────────────
   LOCAL TOOL EXECUTION
────────────────────────────────────────────────────────────── */
async function runLocalTool(name, args = {}) {
  switch (name) {
    case "search_web":
      return { ok: true, data: await webSearch(args.query || "") };

    case "get_weather":
      return { ok: true, data: await getWeather(args.city || "Lagos") };

    case "get_market_data": {
      const data = await fetchMarketPrices([args.symbol || "BTCUSD"]);
      return { ok: true, data };
    }

    case "analyze_market": {
      const analysis = await analyzeSymbol(args.symbol, args.interval, args.outputsize);
      return { ok: true, data: analysis };
    }

    case "run_code": {
      const result = await runSandbox({
        language: args.language,
        code: args.code,
        stdin: args.stdin || "",
        timeout_ms: args.timeout_ms || 8000,
      });
      return { ok: true, data: result };
    }

    default:
      return { ok: false, error: "Not a server-side tool" };
  }
}

function isServerSideTool(name) {
  return ["search_web", "get_weather", "get_market_data", "analyze_market", "run_code"].includes(name);
}

/* ──────────────────────────────────────────────────────────────
   PROMPT BUILDERS
────────────────────────────────────────────────────────────── */
function buildDeviceStateBlock(deviceState = {}) {
  const ds = typeof deviceState === "string" ? { raw: deviceState } : deviceState || {};
  const parts = [];

  if (ds.raw) parts.push(`Raw device state: ${truncateText(ds.raw, 700)}`);
  if (ds.current_app) parts.push(`Current app: ${ds.current_app}`);
  if (ds.current_activity) parts.push(`Current activity: ${ds.current_activity}`);
  if (ds.screen_text) parts.push(`Visible screen text: ${truncateText(ds.screen_text, 1200)}`);
  if (ds.screen_summary) parts.push(`Screen summary: ${truncateText(ds.screen_summary, 500)}`);
  if (ds.network_status) parts.push(`Network: ${ds.network_status}`);
  if (ds.battery_level !== undefined) parts.push(`Battery: ${ds.battery_level}%`);
  if (ds.keyboard_open !== undefined) parts.push(`Keyboard open: ${ds.keyboard_open}`);
  if (ds.notifications_count !== undefined) parts.push(`Notifications count: ${ds.notifications_count}`);

  return parts.length ? parts.join("\n") : "No device state provided.";
}

function buildMemoryBlock(memory = []) {
  const compactMemory = summarizeMemory(memory, 6, 700);
  if (!compactMemory.length) return "";
  return `User memory:\n${compactMemory.join("\n")}`;
}

function buildAutomationSystemPrompt({ deviceState, memory }) {
  return [
    "You are FRIT, an Android device-control agent.",
    "Your job is to plan safe, precise, verifiable actions.",
    "Never assume the UI changed successfully after an action; verification is mandatory.",
    "After risky actions or navigation, prefer read_screen or read_screen_structured.",
    "If element text is missing, use find_element, screenshot analysis, scrolling, or coordinate tools.",
    "Prefer deterministic actions first; use screenshot analysis only when accessibility text is insufficient.",
    "When blocked, recover using back, home, reopen app, scroll, or alternate navigation.",
    "Never invent tool results.",
    "For Android-side tools, return explicit tool calls.",
    "For server-side tools, you may call them directly.",
    "",
    "Device state:",
    buildDeviceStateBlock(deviceState),
    "",
    buildMemoryBlock(memory),
  ]
    .filter(Boolean)
    .join("\n");
}

/* ──────────────────────────────────────────────────────────────
   ROUTES
────────────────────────────────────────────────────────────── */
app.get("/", (_req, res) => {
  res.json({
    name: "FRIT Advanced AI Server",
    status: "online",
    model: MODELS.conversation,
    cache: cacheStats(),
    endpoints: [
      "/health",
      "/chat",
      "/plan",
      "/automate",
      "/market/analyze",
      "/market/batch",
      "/market/quote",
      "/transcribe",
      "/weather",
    ],
  });
});

app.get("/health", (_req, res) => {
  res.json({
    status: "active",
    models: MODELS,
    twelve_data: !!TWELVE_DATA_KEY,
    cache: cacheStats(),
    uptime: Math.floor(process.uptime()) + "s",
  });
});

/* ──────────────────────────────────────────────────────────────
   CHAT
────────────────────────────────────────────────────────────── */
app.post("/chat", async (req, res) => {
  const { message, history = [], memory = [], screen_context, mode = "auto" } = req.body || {};

  if (!message) {
    return res.status(400).json({ error: "No message" });
  }

  const safeHistory = trimHistoryByChars(history, 8, 3200);
  const safeMemory = summarizeMemory(memory, 6, 700);
  const safeScreenContext = maybeTrimScreenContext(screen_context);
  const hasImage = !!safeScreenContext;
  const model = pickModel({ hasImage, mode });

  const symMatch = String(message).match(
    /\b(EURUSD|GBPUSD|USDJPY|AUDUSD|XAUUSD|USDCHF|USDCAD|NZDUSD|GBPJPY|EURJPY|EURGBP|BTCUSD|ETHUSD|SOLUSD|BNBUSD|XRPUSD|DOGEUSD|ADAUSD|BTC|ETH|SOL|BNB|XRP|DOGE|ADA|USDNGN)\b/i
  );
  const ivMatch = String(message).match(/\b(1min|5min|15min|30min|1h|4h|1day)\b/i);
  const sizeMatch = String(message).match(/\b(\d{2,3})\s*(candles?|bars?|data points?)\b/i);

  // FAST PATH: return concise market signal without calling AI model
  if (symMatch && isMarketMessage(message) && !wantsDetailedTradeReason(message)) {
    try {
      const sym = symMatch[1].toUpperCase();
      const iv = ivMatch?.[1]?.toLowerCase() || "1h";
      const sz = sizeMatch ? parseInt(sizeMatch[1], 10) : null;
      const analysis = await analyzeSymbol(sym, iv, sz);

      return res.json({
        text: formatConciseSignal(analysis),
        model_used: "local_market_formatter",
        concise_signal: analysis.concise_signal || null,
        full_analysis_available: true,
      });
    } catch (err) {
      console.error("[/chat concise market path]", err.message);
      // fall through to normal AI path if fast path fails
    }
  }

  let liveData = "";

  if (symMatch) {
    try {
      const sym = symMatch[1].toUpperCase();
      const iv = ivMatch?.[1]?.toLowerCase() || "1h";
      const sz = sizeMatch ? parseInt(sizeMatch[1], 10) : null;
      const a = await analyzeSymbol(sym, iv, sz);

      if (!a.error) {
        liveData = [
          "",
          "=== LIVE MARKET DATA (fetched now) ===",
          a.summary,
          `Concise Signal: ${a.concise_signal?.direction || a.direction} | Entry ${a.concise_signal?.entry || "N/A"} | SL ${a.concise_signal?.sl || "N/A"} | TP ${a.concise_signal?.tp || "N/A"}`,
          `AI Opinion: ${a.concise_signal?.ai_opinion || a.ai_opinion || ""}`,
          "",
          "IMPORTANT:",
          "For market requests, default to ONLY these fields unless the user explicitly asks for reasons/details:",
          "- Direction",
          "- Entry",
          "- SL",
          "- TP",
          "- AI opinion",
          "Only explain structure, confidence, patterns, or deeper reasons if the user asks why or requests full analysis.",
          "",
        ].join("\n");
      }
    } catch (e) {
      console.error("[/chat market fetch]", e.message);
    }
  }

  const systemPrompt = [
    "You are FRIT, a sharp professional AI assistant and trading analyst.",
    "You can browse the web for current information.",
    "When live market data is provided, use it directly.",
    "For trading replies, be concise by default.",
    "Only show full reasoning when the user explicitly asks for it.",
    liveData,
    buildMemoryBlock(safeMemory),
  ]
    .filter(Boolean)
    .join("\n");

  const userContent = hasImage
    ? [
        { type: "text", text: truncateText(message, 4000) },
        { type: "image_url", image_url: { url: `data:image/jpeg;base64,${safeScreenContext}` } },
      ]
    : truncateText(message, 4000);

  try {
    const out = await groqChat({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        ...safeHistory,
        { role: "user", content: userContent },
      ],
      max_tokens: 1600,
    });

    res.json({
      text: out.choices[0].message.content,
      model_used: model,
      payload_guard: {
        screen_context_used: !!safeScreenContext,
        memory_items_used: safeMemory.length,
        history_messages_used: safeHistory.length,
      },
    });
  } catch (err) {
    console.error("[/chat]", err.message);
    res.status(500).json({
      error: "AI request failed",
      details: err.message,
    });
  }
});

/* ──────────────────────────────────────────────────────────────
   PLAN
────────────────────────────────────────────────────────────── */
app.post("/plan", async (req, res) => {
  const { task, device_state = {}, memory = [] } = req.body || {};
  if (!task) return res.status(400).json({ error: "No task" });

  const prompt = [
    "You are a planner for an advanced Android AI agent.",
    "Return a concise numbered plan only.",
    "Do not execute tools.",
    "Include verification steps after major actions.",
    "If the task could fail due to UI ambiguity, include fallback steps.",
    "",
    "Goal:",
    truncateText(task, 2000),
    "",
    "Device state:",
    buildDeviceStateBlock(device_state),
    "",
    buildMemoryBlock(memory),
  ]
    .filter(Boolean)
    .join("\n");

  try {
    const out = await groqChat({
      model: MODELS.fast,
      messages: [
        { role: "system", content: "You create precise phone automation plans." },
        { role: "user", content: prompt },
      ],
      max_tokens: 700,
    });

    res.json({
      plan: out.choices[0].message.content,
      model_used: MODELS.fast,
    });
  } catch (err) {
    console.error("[/plan]", err.message);
    res.status(500).json({
      error: "Planning failed",
      details: err.message,
    });
  }
});

/* ──────────────────────────────────────────────────────────────
   AUTOMATE
────────────────────────────────────────────────────────────── */
app.post("/automate", async (req, res) => {
  const {
    goal,
    device_state = {},
    memory = [],
    history = [],
    max_steps = 4,
    mode = "tools",
  } = req.body || {};

  if (!goal) return res.status(400).json({ error: "No goal" });

  const systemPrompt = buildAutomationSystemPrompt({
    deviceState: device_state,
    memory,
  });

  let messages = [
    { role: "system", content: systemPrompt },
    ...trimHistoryByChars(history, 10, 3500),
    { role: "user", content: truncateText(goal, 3000) },
  ];

  const executedServerTools = [];
  const pendingAndroidActions = [];

  try {
    for (let step = 0; step < Math.max(1, Math.min(max_steps, 8)); step++) {
      const out = await groqChat({
        model: pickModel({ mode }),
        messages,
        tools: AGENT_TOOLS,
        tool_choice: "auto",
        max_tokens: 1200,
      });

      const msg = out.choices[0].message;
      const toolCalls = extractToolCalls(msg);

      if (!toolCalls.length) {
        return res.json({
          done: pendingAndroidActions.length === 0,
          assistant_text: msg.content || "",
          pending_android_actions: pendingAndroidActions,
          server_tool_results: executedServerTools,
          model_used: pickModel({ mode }),
        });
      }

      messages.push({
        role: "assistant",
        content: msg.content || "",
        tool_calls: msg.tool_calls,
      });

      for (const tc of toolCalls) {
        const toolName = tc.function.name;
        const args = tc.function.arguments || {};

        if (isServerSideTool(toolName)) {
          const result = await runLocalTool(toolName, args);
          executedServerTools.push({ tool: toolName, arguments: args, result });

          messages.push({
            role: "tool",
            tool_call_id: tc.id,
            content: JSON.stringify(result),
          });
        } else {
          pendingAndroidActions.push({
            id: tc.id,
            tool: toolName,
            arguments: args,
            requires_android_execution: true,
          });

          messages.push({
            role: "tool",
            tool_call_id: tc.id,
            content: JSON.stringify({
              ok: false,
              pending_android_execution: true,
              tool: toolName,
              arguments: args,
              note: "Execute on device, then call /automate again with updated device_state/history.",
            }),
          });
        }
      }

      if (pendingAndroidActions.length > 0) break;
    }

    return res.json({
      done: false,
      assistant_text:
        "Android actions required. Execute returned actions on device, then send updated device_state and history back to continue.",
      pending_android_actions: pendingAndroidActions,
      server_tool_results: executedServerTools,
      model_used: pickModel({ mode }),
    });
  } catch (err) {
    console.error("[/automate]", err.message);
    res.status(500).json({
      error: "Automation failed",
      details: err.message,
    });
  }
});

/* ──────────────────────────────────────────────────────────────
   MARKET ROUTES
────────────────────────────────────────────────────────────── */
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
    if (!Array.isArray(symbols) || symbols.length === 0) {
      return res.status(400).json({ error: "symbols array required" });
    }
    const normalized = symbols.map((s) => String(s).toUpperCase());
    const data = await fetchMarketPrices(normalized);
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: "Batch fetch failed", details: err.message });
  }
});

app.all("/market/analyze", async (req, res) => {
  try {
    const body = req.body || {};
    const query = req.query || {};

    const symbolRaw = req.method === "GET" ? query.symbol : (body.symbol ?? query.symbol);
    if (!symbolRaw) {
      return res.status(400).json({ error: "Symbol is required" });
    }

    const symbol = String(symbolRaw).toUpperCase();
    const interval = normalizeInterval(req.method === "GET" ? query.interval : (body.interval ?? query.interval ?? "1h"));
    const outputsizeRaw = req.method === "GET" ? query.outputsize : (body.outputsize ?? query.outputsize);
    const outputsize = outputsizeRaw ? Number(outputsizeRaw) : null;

    const data = await analyzeSymbol(symbol, interval, outputsize);
    res.json(data);
  } catch (err) {
    console.error("[/market/analyze]", err.message);
    res.status(500).json({ error: "Analysis failed", details: err.message });
  }
});

/* ──────────────────────────────────────────────────────────────
   WEATHER ROUTE
────────────────────────────────────────────────────────────── */
app.get("/weather", async (req, res) => {
  try {
    const city = String(req.query.city || "Lagos");
    const result = await getWeather(city);
    res.json({ city, result });
  } catch (err) {
    res.status(500).json({ error: "Weather failed", details: err.message });
  }
});

/* ──────────────────────────────────────────────────────────────
   TRANSCRIBE
────────────────────────────────────────────────────────────── */
app.post("/transcribe", async (req, res) => {
  try {
    const { audio_base64, mime_type = "audio/webm" } = req.body || {};
    if (!audio_base64) {
      return res.status(400).json({ error: "audio_base64 required" });
    }

    const form = new FormData();
    const buffer = Buffer.from(audio_base64, "base64");

    form.append("file", buffer, {
      filename: "audio.webm",
      contentType: mime_type,
    });
    form.append("model", MODELS.whisper);

    const out = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`,
        ...form.getHeaders(),
      },
      body: form,
    });

    const data = await out.json();
    if (!out.ok) {
      console.error("[/transcribe]", JSON.stringify(data));
      return res.status(500).json({
        error: "Transcription failed",
        details: data,
      });
    }

    res.json({
      text: data.text || "",
      model_used: MODELS.whisper,
    });
  } catch (err) {
    console.error("[/transcribe]", err.message);
    res.status(500).json({
      error: "Transcription failed",
      details: err.message,
    });
  }
});

/* ──────────────────────────────────────────────────────────────
   ERROR FALLBACK
────────────────────────────────────────────────────────────── */
app.use((err, _req, res, _next) => {
  console.error("[Unhandled Error]", err);
  res.status(500).json({
    error: "Internal server error",
    details: err?.message || "Unknown error",
  });
});

/* ──────────────────────────────────────────────────────────────
   START
────────────────────────────────────────────────────────────── */
app.listen(PORT, () => {
  console.log(`FRIT advanced server listening on :${PORT}`);
});