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

app.set("trust proxy", 1);
if (!GROQ_API_KEY) { console.error("GROQ_API_KEY missing"); process.exit(1); }

// compound-beta = Groq browsing model (built-in web search, NO custom tools array)
// llama-3.3-70b = /automate custom tool calling (AGENT_TOOLS)
const MODELS = {
  vision:       "meta-llama/llama-4-scout-17b-16e-instruct",
  conversation: "compound-beta",
  tools:        "llama-3.3-70b-versatile",
  fast:         "llama-3.1-8b-instant",
  whisper:      "whisper-large-v3",
};

app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: "*", methods: ["GET","POST","OPTIONS"] }));
app.use(morgan(process.env.NODE_ENV === "production" ? "combined" : "dev"));

async function groqChat({ model, messages, tools = null, temperature = 0.4, max_tokens = 1500 }) {
  const body = { model, messages, temperature, max_tokens };
  if (tools?.length) body.tools = tools;
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { "Authorization": `Bearer ${GROQ_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  const data = await res.json();
  if (!res.ok) throw new Error(JSON.stringify(data));
  return data;
}

function pickModel(hasImage = false, mode = "auto") {
  if (mode === "vision" || hasImage) return MODELS.vision;
  if (mode === "fast") return MODELS.fast;
  return MODELS.conversation;
}

// ═══════════════════════════════════════════════════════════════
// MARKET DATA SOURCES
// Forex  → Twelve Data (primary) → Frankfurter (free fallback)
// Crypto → Twelve Data (primary) → Binance candles (fallback)
// ═══════════════════════════════════════════════════════════════

// Twelve Data symbol map — covers forex + crypto + MT5-style variants
const TD_SYMBOLS = {
  EURUSD:"EUR/USD", GBPUSD:"GBP/USD", USDJPY:"USD/JPY", AUDUSD:"AUD/USD",
  USDCHF:"USD/CHF", USDCAD:"USD/CAD", NZDUSD:"NZD/USD", XAUUSD:"XAU/USD",
  XAGUSD:"XAG/USD", GBPJPY:"GBP/JPY", EURJPY:"EUR/JPY", EURGBP:"EUR/GBP",
  BTC:"BTC/USD",    ETH:"ETH/USD",    SOL:"SOL/USD",    BNB:"BNB/USD",
  XRP:"XRP/USD",    DOGE:"DOGE/USD",  ADA:"ADA/USD",
  BTCUSD:"BTC/USD", ETHUSD:"ETH/USD", SOLUSD:"SOL/USD",
  BNBUSD:"BNB/USD", XRPUSD:"XRP/USD", DOGEUSD:"DOGE/USD", ADAUSD:"ADA/USD",
};

// Crypto set — used for decimal precision + CoinGecko/Binance routing
const CRYPTO_SET = new Set([
  "BTC","ETH","SOL","BNB","XRP","DOGE","ADA",
  "BTCUSD","ETHUSD","SOLUSD","BNBUSD","XRPUSD","DOGEUSD","ADAUSD"
]);

const COINGECKO_IDS = {
  BTC:"bitcoin", ETH:"ethereum", SOL:"solana", BNB:"binancecoin",
  XRP:"ripple",  DOGE:"dogecoin", ADA:"cardano",
  BTCUSD:"bitcoin", ETHUSD:"ethereum", SOLUSD:"solana",
  BNBUSD:"binancecoin", XRPUSD:"ripple", DOGEUSD:"dogecoin", ADAUSD:"cardano",
};

// Binance symbol map for crypto candle fallback
const BINANCE_SYM = {
  BTC:"BTCUSDT",   ETH:"ETHUSDT",   SOL:"SOLUSDT",  BNB:"BNBUSDT",
  XRP:"XRPUSDT",   DOGE:"DOGEUSDT", ADA:"ADAUSDT",
  BTCUSD:"BTCUSDT",ETHUSD:"ETHUSDT",SOLUSD:"SOLUSDT",
  BNBUSD:"BNBUSDT",XRPUSD:"XRPUSDT",DOGEUSD:"DOGEUSDT",ADAUSD:"ADAUSDT",
};

const FRANKFURTER_MAP = {
  EURUSD:{base:"EUR",quote:"USD"}, GBPUSD:{base:"GBP",quote:"USD"},
  USDJPY:{base:"USD",quote:"JPY"}, AUDUSD:{base:"AUD",quote:"USD"},
  USDCHF:{base:"USD",quote:"CHF"}, USDCAD:{base:"USD",quote:"CAD"},
  NZDUSD:{base:"NZD",quote:"USD"}, EURGBP:{base:"EUR",quote:"GBP"},
  EURJPY:{base:"EUR",quote:"JPY"}, GBPJPY:{base:"GBP",quote:"JPY"},
};

// Scale outputsize to interval so we always get ~1 week of data minimum
function resolveOutputSize(interval) {
  const map = {
    "1min":500, "5min":288, "15min":200, "30min":150,
    "1h":168, "4h":120, "1day":100, "1week":52,
  };
  return map[interval] || 168;
}

// Binance interval format mapping
function toBinanceInterval(interval) {
  const map = {
    "1min":"1m","5min":"5m","15min":"15m","30min":"30m",
    "1h":"1h","4h":"4h","1day":"1d","1week":"1w",
  };
  return map[interval] || "1h";
}

// ── fetchCandles: Twelve Data first, Binance fallback for crypto ──
async function fetchCandles(symbol, interval = "1h", outputsize = null) {
  const sym = symbol.toUpperCase();
  const size = outputsize || resolveOutputSize(interval);

  // 1. Twelve Data (forex + crypto)
  if (TWELVE_DATA_KEY && TD_SYMBOLS[sym]) {
    try {
      const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&interval=${interval}&outputsize=${size}&apikey=${TWELVE_DATA_KEY}`;
      const res = await fetch(url);
      const data = await res.json();
      if (data.status !== "error" && data.values?.length >= 10) {
        return data.values.reverse().map(v => ({
          time: new Date(v.datetime).getTime(),
          open: parseFloat(v.open), high: parseFloat(v.high),
          low: parseFloat(v.low),   close: parseFloat(v.close),
          volume: parseFloat(v.volume || 0)
        }));
      }
    } catch {}
  }

  // 2. Binance fallback for crypto
  if (BINANCE_SYM[sym]) {
    try {
      const url = `https://api.binance.com/api/v3/klines?symbol=${BINANCE_SYM[sym]}&interval=${toBinanceInterval(interval)}&limit=${Math.min(size, 1000)}`;
      const res = await fetch(url);
      const arr = await res.json();
      if (Array.isArray(arr) && arr.length >= 10) {
        return arr.map(k => ({
          time: k[0], open: parseFloat(k[1]), high: parseFloat(k[2]),
          low: parseFloat(k[3]), close: parseFloat(k[4]), volume: parseFloat(k[5])
        }));
      }
    } catch {}
  }

  return null;
}

// ── Spot price: Twelve Data → CoinGecko → Frankfurter → er-api ──
async function fetchSpotPrice(symbol) {
  const sym = symbol.toUpperCase();

  if (TWELVE_DATA_KEY && TD_SYMBOLS[sym]) {
    try {
      const res = await fetch(`https://api.twelvedata.com/price?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&apikey=${TWELVE_DATA_KEY}`);
      const data = await res.json();
      if (data.price) return { price: parseFloat(data.price), source: "twelvedata" };
    } catch {}
  }

  if (COINGECKO_IDS[sym]) {
    try {
      const res = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${COINGECKO_IDS[sym]}&vs_currencies=usd&include_24hr_change=true`);
      const data = await res.json();
      const id = COINGECKO_IDS[sym];
      if (data[id]) return { price: data[id].usd, change24h: data[id].usd_24h_change, source: "coingecko" };
    } catch {}
  }

  if (FRANKFURTER_MAP[sym]) {
    try {
      const { base, quote } = FRANKFURTER_MAP[sym];
      const res = await fetch(`https://api.frankfurter.app/latest?from=${base}&to=${quote}`);
      const data = await res.json();
      const rate = data.rates?.[quote];
      if (rate) return { price: parseFloat(rate), source: "frankfurter" };
    } catch {}
  }

  if (sym === "USDNGN") {
    try {
      const res = await fetch("https://open.er-api.com/v6/latest/USD");
      const data = await res.json();
      if (data.rates?.NGN) return { price: data.rates.NGN, source: "er-api" };
    } catch {}
  }

  return null;
}

async function fetch24hDelta(sym) {
  if (!CRYPTO_SET.has(sym) && TWELVE_DATA_KEY && TD_SYMBOLS[sym]) {
    try {
      const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&interval=1day&outputsize=2&apikey=${TWELVE_DATA_KEY}`;
      const res = await fetch(url);
      const data = await res.json();
      if (data.values?.length >= 2) {
        const today = parseFloat(data.values[0].close);
        const yesterday = parseFloat(data.values[1].close);
        return ((today - yesterday) / yesterday) * 100;
      }
    } catch {}
  }
  return null;
}

async function fetchMarketPrices(symbols) {
  const result = {};
  await Promise.all(symbols.map(async (sym) => {
    const s = sym.toUpperCase();
    const spot = await fetchSpotPrice(s);
    if (!spot) { result[sym] = { symbol:sym, price:0, change24h:0, currency:"USD", error:"Not found" }; return; }
    let change24h = (spot.change24h !== undefined && spot.change24h !== 0) ? spot.change24h : (await fetch24hDelta(s) ?? 0);
    result[sym] = { symbol:sym, price:spot.price, change24h, currency:sym.includes("NGN")?"NGN":"USD", source:spot.source };
  }));
  return result;
}

// ── Technical Indicators ──────────────────────────────────────────
function calcEMA(closes, period) {
  if (closes.length < period) return [];
  const k = 2 / (period + 1);
  const result = [];
  let prev = closes.slice(0, period).reduce((a,b) => a+b, 0) / period;
  result.push(prev);
  for (let i = period; i < closes.length; i++) { prev = closes[i]*k + prev*(1-k); result.push(prev); }
  return result;
}
function calcRSI(closes, period = 14) {
  if (closes.length < period+1) return 50;
  const changes = closes.slice(-(period+1)).map((v,i,a) => i>0 ? v-a[i-1] : 0).slice(1);
  const avgGain = changes.map(c=>c>0?c:0).reduce((a,b)=>a+b,0)/period;
  const avgLoss = changes.map(c=>c<0?-c:0).reduce((a,b)=>a+b,0)/period;
  if (avgLoss===0) return 100;
  return 100-(100/(1+avgGain/avgLoss));
}
function calcMACD(closes) {
  const ema12=calcEMA(closes,12), ema26=calcEMA(closes,26);
  if (!ema12.length||!ema26.length) return {macd:0,signal:0,hist:0};
  const offset=ema12.length-ema26.length;
  const macdLine=ema26.map((v,i)=>ema12[i+offset]-v);
  const signalLine=calcEMA(macdLine,9);
  const last=macdLine.at(-1), sig=signalLine.at(-1)??0;
  return {macd:last,signal:sig,hist:last-sig};
}
function calcSR(candles) {
  const r=candles.slice(-20);
  return {support:Math.min(...r.map(c=>c.low)),resistance:Math.max(...r.map(c=>c.high))};
}
function calcBB(closes, period=20, mult=2) {
  if (closes.length < period) return null;
  const slice = closes.slice(-period);
  const mean = slice.reduce((a,b)=>a+b,0)/period;
  const std = Math.sqrt(slice.map(v=>(v-mean)**2).reduce((a,b)=>a+b,0)/period);
  return {upper:mean+mult*std, middle:mean, lower:mean-mult*std};
}

// ── Core analysis — fetches real data, sends to AI ──────────────
// This is the key fix: candles are fetched from Twelve Data/Binance
// based on user-requested interval, then FULL data is sent to AI
async function analyzeSymbol(symbol, interval = "1h", customSize = null) {
  const sym = symbol.toUpperCase();
  const candles = await fetchCandles(sym, interval, customSize);
  const spot    = await fetchSpotPrice(sym);

  if (!candles && !spot) return { symbol:sym, error:`No data for ${sym} — check Twelve Data key or symbol` };

  const price = spot?.price ?? candles?.at(-1)?.close ?? 0;
  if (!candles || candles.length < 30) {
    return { symbol:sym, price, source:spot?.source, analysis:"Insufficient candle data", candleCount:candles?.length??0 };
  }

  const closes = candles.map(c=>c.close);
  const rsi = calcRSI(closes);
  const {macd,signal:macdSig,hist} = calcMACD(closes);
  const ema20 = calcEMA(closes,20).at(-1)??price;
  const ema50 = calcEMA(closes,50).at(-1)??price;
  const ema200= calcEMA(closes,200).at(-1)??price;
  const {support,resistance} = calcSR(candles);
  const bb = calcBB(closes);

  let bull=0, bear=0;
  if (rsi>50&&rsi<70) bull++; else if (rsi<50&&rsi>30) bear++;
  if (macd>macdSig)  bull++; else bear++;
  if (price>ema20)   bull++; else bear++;
  if (ema20>ema50)   bull++; else bear++;
  if (hist>0)        bull++; else bear++;

  const direction = rsi>=70?"OVERBOUGHT":rsi<=30?"OVERSOLD":bull>=4?"BULLISH":bear>=4?"BEARISH":"NEUTRAL";
  const strength  = (bull===5||bear===5)?"STRONG":(bull>=3||bear>=3)?"MODERATE":"WEAK";

  // Correct decimal precision: crypto=2dp, gold=2dp, forex=5dp
  const isCrypto = CRYPTO_SET.has(sym);
  const dp = (isCrypto || sym==="XAUUSD") ? 2 : 5;

  // Recent 20 candles summary for AI context
  const recentCandles = candles.slice(-20).map(c=>
    `O:${c.open.toFixed(dp)} H:${c.high.toFixed(dp)} L:${c.low.toFixed(dp)} C:${c.close.toFixed(dp)}`
  ).join(" | ");

  return {
    symbol:sym, price, direction, strength, interval,
    candleCount:candles.length, source:spot?.source??"binance",
    rsi:+rsi.toFixed(1), macd:+macd.toFixed(6),
    macd_signal:macd>macdSig?"bullish":"bearish",
    ema20:+ema20.toFixed(dp), ema50:+ema50.toFixed(dp), ema200:+ema200.toFixed(dp),
    support:+support.toFixed(dp), resistance:+resistance.toFixed(dp),
    bb_upper:bb?+bb.upper.toFixed(dp):null,
    bb_middle:bb?+bb.middle.toFixed(dp):null,
    bb_lower:bb?+bb.lower.toFixed(dp):null,
    recent_candles: recentCandles,
    summary:`${sym} @${price.toFixed(dp)} | ${direction}(${strength}) RSI:${rsi.toFixed(1)} MACD:${macd>macdSig?"UP":"DOWN"} EMA20:${ema20.toFixed(dp)} EMA50:${ema50.toFixed(dp)} S:${support.toFixed(dp)} R:${resistance.toFixed(dp)} [${candles.length} candles, ${interval}]`
  };
}

// ── Web Search & Weather ──────────────────────────────────────────
async function webSearch(query) {
  try {
    const res = await fetch(`https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`);
    const d = await res.json();
    return d.AbstractText||d.Answer||d.RelatedTopics?.[0]?.Text||"No result found.";
  } catch(e) { return "Search failed: "+e.message; }
}
async function getWeather(city="Lagos") {
  try {
    const g = await (await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`)).json();
    const loc = g.results?.[0]; if(!loc) return "City not found";
    const w = await (await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${loc.latitude}&longitude=${loc.longitude}&current_weather=true&timezone=auto`)).json();
    const cw = w.current_weather;
    return `${loc.name}: ${cw.temperature}C, wind ${cw.windspeed}km/h, ${cw.weathercode<=1?"Clear":cw.weathercode<=3?"Cloudy":"Rainy"}`;
  } catch(e) { return "Weather unavailable"; }
}

// ═══════════════════════════════════════════════════════════════
// AGENTIC TOOLS — custom tools for llama-3.3-70b (/automate only)
// ═══════════════════════════════════════════════════════════════
const AGENT_TOOLS = [
  {type:"function",function:{name:"open_app",description:"Open an app on Android",parameters:{type:"object",properties:{app_name:{type:"string"}},required:["app_name"]}}},
  {type:"function",function:{name:"make_call",description:"Call a contact or number",parameters:{type:"object",properties:{contact_name:{type:"string"},phone_number:{type:"string"}}}}},
  {type:"function",function:{name:"send_whatsapp",description:"Open WhatsApp and compose message",parameters:{type:"object",properties:{contact_name:{type:"string"},message:{type:"string"}},required:["message"]}}},
  {type:"function",function:{name:"send_sms",description:"Send SMS to a contact",parameters:{type:"object",properties:{contact_name:{type:"string"},message:{type:"string"}},required:["message"]}}},
  {type:"function",function:{name:"get_market_data",description:"Get live spot price. Accepts BTC, ETH, BTCUSD, EURUSD, XAUUSD etc.",parameters:{type:"object",properties:{symbol:{type:"string"}},required:["symbol"]}}},
  {type:"function",function:{name:"analyze_market",description:"Full technical analysis with RSI MACD EMA Bollinger Bands support/resistance. Works for ALL symbols: BTCUSD ETHUSD EURUSD GBPUSD XAUUSD etc.",parameters:{type:"object",properties:{symbol:{type:"string"},interval:{type:"string",description:"1min 5min 15min 30min 1h 4h 1day. Default 1h"},outputsize:{type:"number",description:"Number of candles. Default auto-scaled by interval"}},required:["symbol"]}}},
  {type:"function",function:{name:"place_trade",description:"Place buy or sell trade on MT5. Works for any symbol.",parameters:{type:"object",properties:{symbol:{type:"string"},action:{type:"string",enum:["buy","sell"]},volume:{type:"number"},sl:{type:"number"},tp:{type:"number"}},required:["symbol","action","volume"]}}},
  {type:"function",function:{name:"search_web",description:"Search internet for news, market sentiment, or any current info",parameters:{type:"object",properties:{query:{type:"string"}},required:["query"]}}},
  {type:"function",function:{name:"read_screen",description:"Read all visible text from current phone screen",parameters:{type:"object",properties:{}}}},
  {type:"function",function:{name:"tap_button",description:"Tap a button on screen by text label",parameters:{type:"object",properties:{label:{type:"string"}},required:["label"]}}},
  {type:"function",function:{name:"type_text",description:"Type text into a field on screen",parameters:{type:"object",properties:{field:{type:"string"},value:{type:"string"}},required:["value"]}}},
  {type:"function",function:{name:"set_alarm",description:"Set an alarm",parameters:{type:"object",properties:{label:{type:"string"},time:{type:"string"}},required:["label","time"]}}},
  {type:"function",function:{name:"set_timer",description:"Start a countdown timer",parameters:{type:"object",properties:{duration:{type:"string"}},required:["duration"]}}},
  {type:"function",function:{name:"play_music",description:"Play music on Spotify or YouTube",parameters:{type:"object",properties:{query:{type:"string"}},required:["query"]}}},
  {type:"function",function:{name:"navigate_to",description:"Open Google Maps navigation",parameters:{type:"object",properties:{destination:{type:"string"}},required:["destination"]}}},
  {type:"function",function:{name:"get_weather",description:"Get current weather for a city",parameters:{type:"object",properties:{city:{type:"string"}},required:["city"]}}},
];

// ═══════════════════════════════════════════════════════════════
// ROUTES
// ═══════════════════════════════════════════════════════════════
app.get("/", (_req, res) => res.json({ name:"FRIT Server", status:"online", model:MODELS.conversation,
  endpoints:["/health","/chat","/automate","/market/analyze","/market/batch","/market/quote","/transcribe","/weather"] }));

app.get("/health", (_req, res) => res.json({ status:"active", engine:"FRIT-Core",
  models:MODELS, twelve_data:!!TWELVE_DATA_KEY, uptime:Math.floor(process.uptime())+"s" }));

// ── /chat: compound-beta (browsing) + live data injected into prompt ─
app.post("/chat", async (req, res) => {
  const { message, history=[], memory=[], screen_context, mode="auto" } = req.body||{};
  if (!message) return res.status(400).json({ error:"No message provided" });

  const hasImage = !!screen_context;
  const model = pickModel(hasImage, mode);

  // Detect market symbol in message and pre-fetch live data
  const symMatch = message.match(/\b(EURUSD|GBPUSD|USDJPY|AUDUSD|XAUUSD|USDCHF|USDCAD|NZDUSD|GBPJPY|EURJPY|EURGBP|BTCUSD|ETHUSD|SOLUSD|BNBUSD|XRPUSD|DOGEUSD|ADAUSD|BTC|ETH|SOL|BNB|XRP|DOGE|ADA|USDNGN)\b/i);
  const intervalMatch = message.match(/\b(1min|5min|15min|30min|1h|4h|1day)\b/i);
  const sizeMatch = message.match(/\b(\d{2,3})\s*(candles?|bars?|data points?)\b/i);

  let liveDataBlock = "";
  if (symMatch) {
    try {
      const sym = symMatch[1].toUpperCase();
      const interval = intervalMatch?.[1]?.toLowerCase() || "1h";
      const customSize = sizeMatch ? parseInt(sizeMatch[1]) : null;
      const analysis = await analyzeSymbol(sym, interval, customSize);
      if (!analysis.error) {
        liveDataBlock = `\n\n=== LIVE MARKET DATA (fetched now) ===\n${analysis.summary}\nRSI:${analysis.rsi} | MACD:${analysis.macd_signal} | EMA20:${analysis.ema20} | EMA50:${analysis.ema50} | EMA200:${analysis.ema200}\nBollinger: U:${analysis.bb_upper} M:${analysis.bb_middle} L:${analysis.bb_lower}\nSupport:${analysis.support} | Resistance:${analysis.resistance}\nLast 20 candles (${analysis.interval}): ${analysis.recent_candles}\nData source: ${analysis.source} | ${analysis.candleCount} candles loaded\n===\nUSE THIS DATA. Do NOT say data is unavailable.`;
      }
    } catch {}
  }

  const systemPrompt = [
    "You are FRIT — a sharp professional AI assistant and trading analyst on a custom Android system.",
    "You can browse the web for current news and market sentiment.",
    "When live market data is provided below, USE IT to give specific entry, SL, TP levels with exact price numbers.",
    "Never say you lack live data when data has been injected into this prompt.",
    "The user is a forex/crypto trader in Lagos, Nigeria. Be concise and decisive.",
    liveDataBlock,
    memory.length ? `User memory:\n${memory.join("\n")}` : ""
  ].filter(Boolean).join("\n\n");

  const userContent = hasImage
    ? [{type:"text",text:message},{type:"image_url",image_url:{url:`data:image/jpeg;base64,${screen_context}`}}]
    : message;

  try {
    const result = await groqChat({ model, messages:[{role:"system",content:systemPrompt},...history,{role:"user",content:userContent}], max_tokens:1500 });
    res.json({ text:result.choices[0].message.content, model_used:model });
  } catch(err) { res.status(500).json({ error:"AI request failed", details:err.message }); }
});

// ── /automate: llama-3.3-70b + AGENT_TOOLS (custom tool calling) ──

app.post("/automate", async (req, res) => {
  const { task, context="", screen_text="", steps_done="" } = req.body||{};
  if (!task) return res.status(400).json({ error:"No task provided" });
  const spLines = [
    "You are FRIT, an AI that controls an Android phone step by step",
    "You receive: the task, what is on screen NOW, and steps done so far",
    "Decide ONLY the NEXT action - do not plan the whole sequence upfront",
    "",
    "RULES",
    "1 Read current screen content before deciding what to do next",
    "2 If screen is empty or wrong app - call open_app first",
    "3 After open_app Android waits 3 seconds and reads screen - you see result next call",
    "4 WhatsApp: open_app then tap_button(person name) then type_text(message) then tap_button(Send)",
    "5 MT5: analyze_market then place_trade if signal strong or moderate or user said force",
    "6 Reading messages: screen_text IS the live screen - read and summarise it",
    "7 When task complete set done true and explain in summary",
    "8 Never repeat actions already in steps_done",
    "9 All symbols supported BTCUSD ETHUSD EURUSD GBPUSD XAUUSD",
    "10 Website: open_app(chrome) then tap address bar then type url then fill in login fields",
    "",
    "Current screen: " + (screen_text || "Not available - enable accessibility service"),
    "Steps done: " + (steps_done || "None - first step"),
    "Context: " + (context || "Forex/crypto trader Lagos Nigeria")
  ];
  const sp = spLines.join("\n");
  const messages = [{role:"system",content:sp},{role:"user",content:"Task: "+task}];
  const allActions = [];
  let finalText = "", isDone = false;
  for (let i=0; i<4; i++) {
    let result;
    try { result = await groqChat({model:MODELS.tools,messages,tools:AGENT_TOOLS,max_tokens:900}); }
    catch(err) { return res.status(500).json({error:"AI failed",details:err.message}); }
    const choice = result.choices[0];
    if (choice.finish_reason !== "tool_calls" || !choice.message.tool_calls?.length) {
      finalText = choice.message.content||"";
      isDone = /done|complete|finish|sent|placed|opened|navigating/i.test(finalText);
      break;
    }
    messages.push({role:"assistant",content:choice.message.content||null,tool_calls:choice.message.tool_calls});
    const toolResults = [];
    for (const tc of choice.message.tool_calls) {
      const name = tc.function.name;
      let args = {}; try { args = JSON.parse(tc.function.arguments); } catch {}
      let toolResult = "";
      if (name==="analyze_market") {
        const sym = (args.symbol||"").toUpperCase();
        const analysis = await analyzeSymbol(sym, args.interval||"1h", args.outputsize||null);
        if (!analysis.error) {
          const aR = await groqChat({model:MODELS.fast,max_tokens:600,temperature:0.3,
            messages:[{role:"user",content:"Analyst "+sym+" RSI "+analysis.rsi+" Dir "+analysis.direction+" "+analysis.strength+" S "+analysis.support+" R "+analysis.resistance+" Candles "+analysis.recent_candles+" Give direction confidence entry SL TP exact"}]});
          analysis.ai_opinion = aR.choices[0].message.content;
        }
        toolResult = JSON.stringify(analysis);
        allActions.push({tool:name,result:analysis,server_side:true});
      } else if (name==="get_market_data") {
        const sym = (args.symbol||"").toUpperCase();
        const data = await fetchMarketPrices([sym]);
        toolResult = JSON.stringify(data[sym]);
        allActions.push({tool:name,result:data[sym],server_side:true});
      } else if (name==="search_web") {
        toolResult = await webSearch(args.query);
        allActions.push({tool:name,result:toolResult,server_side:true});
      } else if (name==="get_weather") {
        toolResult = await getWeather(args.city);
        allActions.push({tool:name,result:toolResult,server_side:true});
      } else {
        toolResult = "Queued for Android: "+name;
        allActions.push({tool:name,args,action:"android_execute"});
      }
      toolResults.push({role:"tool",tool_call_id:tc.id,content:toolResult});
    }
    messages.push(...toolResults);
  }
  res.json({type:"tool_result",actions:allActions,summary:finalText,done:isDone,model_used:MODELS.tools});
});
// ── /market/analyze ───────────────────────────────────────────────
app.post("/market/analyze", async (req, res) => {
  const { symbol, interval="1h", user_context="", outputsize=null } = req.body||{};
  if (!symbol) return res.status(400).json({ error:"symbol required" });
  try {
    const analysis = await analyzeSymbol(symbol.toUpperCase(), interval, outputsize);
    if (analysis.error) return res.json(analysis);
    const aiRes = await groqChat({model:MODELS.tools, max_tokens:600, temperature:0.3,
      messages:[{role:"user",content:`Professional forex/crypto analyst.\nSymbol:${analysis.symbol} Interval:${interval} Candles:${analysis.candleCount} Source:${analysis.source}\nRSI:${analysis.rsi} MACD:${analysis.macd_signal} EMA20:${analysis.ema20} EMA50:${analysis.ema50} EMA200:${analysis.ema200}\nBB: U:${analysis.bb_upper} M:${analysis.bb_middle} L:${analysis.bb_lower}\nS:${analysis.support} R:${analysis.resistance} Direction:${analysis.direction}(${analysis.strength})\nRecent candles: ${analysis.recent_candles}\nUser: ${user_context||"Lagos trader"}\nGive: direction, confidence%, exact entry, SL, TP.`}]});
    res.json({...analysis, ai_opinion:aiRes.choices[0].message.content});
  } catch(err) { res.status(500).json({error:err.message}); }
});

// ── /analyze ──────────────────────────────────────────────────────
app.post("/analyze", async (req, res) => {
  const { prompt, context="" } = req.body||{};
  if (!prompt) return res.status(400).json({ error:"prompt required" });
  try {
    const result = await groqChat({model:MODELS.conversation,
      messages:[{role:"system",content:"You are FRIT intelligence core. Provide deep expert analysis. Browse the web for current data when needed."},
                {role:"user",content:context?`Context:\n${context}\n\nTask:\n${prompt}`:prompt}], max_tokens:2048, temperature:0.3});
    res.json({text:result.choices[0].message.content, model_used:MODELS.conversation});
  } catch(err) { res.status(500).json({error:err.message}); }
});

// ── /market/quote ─────────────────────────────────────────────────
app.get("/market/quote", async (req, res) => {
  const { symbol } = req.query;
  if (!symbol) return res.status(400).json({ error:"symbol required" });
  const spot = await fetchSpotPrice(symbol.toUpperCase());
  if (!spot) return res.status(404).json({ error:`Symbol not found: ${symbol}` });
  res.json({symbol, ...spot});
});

// ── /market/batch ─────────────────────────────────────────────────
app.post("/market/batch", async (req, res) => {
  const { symbols=["EURUSD","GBPUSD","XAUUSD","BTC","USDNGN"] } = req.body;
  try { res.json(await fetchMarketPrices(symbols)); }
  catch(err) { res.status(500).json({error:err.message}); }
});

// ── /transcribe ───────────────────────────────────────────────────
app.post("/transcribe", async (req, res) => {
  const { audio_base64, language="en" } = req.body||{};
  if (!audio_base64) return res.status(400).json({ error:"audio_base64 required" });
  try {
    const audioBuffer = Buffer.from(audio_base64, "base64");
    if (audioBuffer.length<1000) return res.status(400).json({ error:"Audio too short" });
    const form = new FormData();
    form.append("file", audioBuffer, {filename:"audio.wav",contentType:"audio/wav"});
    form.append("model", MODELS.whisper);
    form.append("language", language);
    form.append("response_format", "json");
    const whisperRes = await fetch("https://api.groq.com/openai/v1/audio/transcriptions",
      {method:"POST",headers:{"Authorization":`Bearer ${GROQ_API_KEY}`,...form.getHeaders()},body:form});
    const result = await whisperRes.json();
    if (!whisperRes.ok) throw new Error(JSON.stringify(result));
    res.json({text:result.text||""});
  } catch(err) { res.status(500).json({error:"Transcription failed",details:err.message}); }
});

// ── /weather ──────────────────────────────────────────────────────
app.get("/weather", async (req, res) => {
  res.json({result: await getWeather(req.query.city||"Lagos")});
});

// ── /reminder ─────────────────────────────────────────────────────
app.post("/reminder", (req, res) => {
  const { text, delay_seconds } = req.body||{};
  if (!text) return res.status(400).json({ error:"text required" });
  res.json({scheduled:true, task:text, in_seconds:delay_seconds||0});
});

// ── START ─────────────────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
  console.log(`
  FRIT SERVER RUNNING
  Port:        ${PORT}
  Chat model:  ${MODELS.conversation} (web browsing enabled)
  Tools model: ${MODELS.tools} (custom tool calling)
  Twelve Data: ${TWELVE_DATA_KEY?"Connected":"MISSING - add TWELVE_DATA_KEY"}
  Data flow:   Forex -> TwelveData -> Frankfurter fallback
               Crypto -> TwelveData -> Binance fallback
  `);
});

