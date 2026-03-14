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
if (!GROQ_API_KEY) { console.error("❌ GROQ_API_KEY missing"); process.exit(1); }

// ── MODELS ────────────────────────────────────────────────────────
const MODELS = {
  vision:       "meta-llama/llama-4-scout-17b-16e-instruct",
  conversation: "llama-3.3-70b-versatile",
  tools:        "llama-3.3-70b-versatile",
  fast:         "llama-3.1-8b-instant",
  whisper:      "whisper-large-v3",
};

// ── MIDDLEWARE ────────────────────────────────────────────────────
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: "*", methods: ["GET","POST","OPTIONS"] }));
app.use(morgan(process.env.NODE_ENV === "production" ? "combined" : "dev"));

// ── GROQ HELPER ───────────────────────────────────────────────────
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

function pickModel(message = "", hasImage = false, mode = "auto") {
  if (mode === "vision" || hasImage) return MODELS.vision;
  if (mode === "tools")              return MODELS.tools;
  if (mode === "fast")               return MODELS.fast;
  return MODELS.conversation;
}


// ═══════════════════════════════════════════════════════════════
// MARKET DATA — Twelve Data (forex+crypto OHLC) + CoinGecko fallback
// ═══════════════════════════════════════════════════════════════
const TD_SYMBOLS = {
  EURUSD:"EUR/USD", GBPUSD:"GBP/USD", USDJPY:"USD/JPY", AUDUSD:"AUD/USD",
  USDCHF:"USD/CHF", USDCAD:"USD/CAD", NZDUSD:"NZD/USD", XAUUSD:"XAU/USD",
  XAGUSD:"XAG/USD", GBPJPY:"GBP/JPY", EURJPY:"EUR/JPY", EURGBP:"EUR/GBP",
  BTC:"BTC/USD", ETH:"ETH/USD", SOL:"SOL/USD", BNB:"BNB/USD",
  XRP:"XRP/USD", DOGE:"DOGE/USD", ADA:"ADA/USD"
};
const CRYPTO_IDS = {
  BTC:"bitcoin", ETH:"ethereum", SOL:"solana", BNB:"binancecoin",
  XRP:"ripple", DOGE:"dogecoin", ADA:"cardano"
};

async function fetchCandles(symbol, interval = "1h", outputsize = 100) {
  if (!TWELVE_DATA_KEY) return null;
  const tdSym = TD_SYMBOLS[symbol.toUpperCase()] || symbol;
  try {
    const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(tdSym)}&interval=${interval}&outputsize=${outputsize}&apikey=${TWELVE_DATA_KEY}`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.status === "error" || !data.values) return null;
    return data.values.reverse().map(v => ({
      time: new Date(v.datetime).getTime(),
      open: parseFloat(v.open), high: parseFloat(v.high),
      low: parseFloat(v.low), close: parseFloat(v.close),
      volume: parseFloat(v.volume || 0)
    }));
  } catch { return null; }
}

async function fetchSpotPrice(symbol) {
  const sym = symbol.toUpperCase();
  if (TWELVE_DATA_KEY && TD_SYMBOLS[sym]) {
    try {
      const res = await fetch(`https://api.twelvedata.com/price?symbol=${encodeURIComponent(TD_SYMBOLS[sym])}&apikey=${TWELVE_DATA_KEY}`);
      const data = await res.json();
      if (data.price) return { price: parseFloat(data.price), source: "twelvedata" };
    } catch {}
  }
  if (CRYPTO_IDS[sym]) {
    try {
      const res = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${CRYPTO_IDS[sym]}&vs_currencies=usd&include_24hr_change=true`);
      const data = await res.json();
      const id = CRYPTO_IDS[sym];
      if (data[id]) return { price: data[id].usd, change24h: data[id].usd_24h_change, source: "coingecko" };
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
  return 100 - (100/(1+avgGain/avgLoss));
}
function calcMACD(closes) {
  const ema12 = calcEMA(closes, 12), ema26 = calcEMA(closes, 26);
  if (!ema12.length || !ema26.length) return { macd:0, signal:0, hist:0 };
  const offset = ema12.length - ema26.length;
  const macdLine = ema26.map((v,i) => ema12[i+offset]-v);
  const signalLine = calcEMA(macdLine, 9);
  const last = macdLine.at(-1), sig = signalLine.at(-1)??0;
  return { macd: last, signal: sig, hist: last-sig };
}
function calcSR(candles) {
  const r = candles.slice(-20);
  return { support: Math.min(...r.map(c=>c.low)), resistance: Math.max(...r.map(c=>c.high)) };
}

async function analyzeSymbol(symbol, interval = "1h") {
  const candles = await fetchCandles(symbol, interval);
  const spot = await fetchSpotPrice(symbol);
  if (!candles && !spot) return { symbol, error: "No data available" };
  const price = spot?.price ?? candles?.at(-1)?.close ?? 0;
  if (!candles || candles.length < 30) return { symbol, price, analysis: "Insufficient candle history", source: spot?.source };
  const closes = candles.map(c=>c.close);
  const rsi = calcRSI(closes);
  const { macd, signal, hist } = calcMACD(closes);
  const ema20 = calcEMA(closes,20).at(-1)??price, ema50 = calcEMA(closes,50).at(-1)??price;
  const { support, resistance } = calcSR(candles);
  let bull=0, bear=0;
  if (rsi>50&&rsi<70) bull++; else if (rsi<50&&rsi>30) bear++;
  if (macd>signal) bull++; else bear++;
  if (price>ema20) bull++; else bear++;
  if (ema20>ema50) bull++; else bear++;
  if (hist>0) bull++; else bear++;
  const direction = rsi>=70?"OVERBOUGHT":rsi<=30?"OVERSOLD":bull>=4?"BULLISH":bear>=4?"BEARISH":"NEUTRAL";
  const strength = (bull===5||bear===5)?"STRONG":(bull>=3||bear>=3)?"MODERATE":"WEAK";
  const isForex = !CRYPTO_IDS[symbol.toUpperCase()];
  const dp = isForex && symbol!=="XAUUSD" ? 5 : 2;
  return { symbol, price, direction, strength, rsi: +rsi.toFixed(1), macd: +macd.toFixed(6),
    macd_signal: macd>signal?"bullish":"bearish", ema20: +ema20.toFixed(dp), ema50: +ema50.toFixed(dp),
    support: +support.toFixed(dp), resistance: +resistance.toFixed(dp), interval,
    summary: `${symbol} @${price.toFixed(dp)} | ${direction}(${strength}) RSI:${rsi.toFixed(1)} MACD:${macd>signal?"▲":"▼"} EMA20:${ema20.toFixed(dp)} EMA50:${ema50.toFixed(dp)} S:${support.toFixed(dp)} R:${resistance.toFixed(dp)}`
  };
}

async function fetchMarketPrices(symbols) {
  const result = {};
  await Promise.all(symbols.map(async (sym) => {
    const spot = await fetchSpotPrice(sym);
    result[sym] = spot ? { symbol:sym, price:spot.price, change24h:spot.change24h??0, currency:sym.includes("NGN")?"NGN":"USD" }
                       : { symbol:sym, price:0, change24h:0, currency:"USD", error:"Not found" };
  }));
  return result;
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
    return `${loc.name}: ${cw.temperature}°C, wind ${cw.windspeed}km/h, ${cw.weathercode<=1?"Clear":cw.weathercode<=3?"Cloudy":"Rainy"}`;
  } catch(e) { return "Weather unavailable"; }
}


// ═══════════════════════════════════════════════════════════════
// AGENTIC TOOLS — 16 tools for full phone + market control
// ═══════════════════════════════════════════════════════════════
const AGENT_TOOLS = [
  { type:"function", function:{ name:"open_app", description:"Open an app on Android", parameters:{ type:"object", properties:{ app_name:{type:"string"} }, required:["app_name"] } } },
  { type:"function", function:{ name:"make_call", description:"Call a contact or number", parameters:{ type:"object", properties:{ contact_name:{type:"string"}, phone_number:{type:"string"} } } } },
  { type:"function", function:{ name:"send_whatsapp", description:"Open WhatsApp and compose message to contact", parameters:{ type:"object", properties:{ contact_name:{type:"string"}, message:{type:"string"} }, required:["message"] } } },
  { type:"function", function:{ name:"send_sms", description:"Send SMS to a contact", parameters:{ type:"object", properties:{ contact_name:{type:"string"}, message:{type:"string"} }, required:["message"] } } },
  { type:"function", function:{ name:"get_market_data", description:"Get live price for a symbol (crypto or forex like EURUSD, GBPUSD, XAUUSD, BTC)", parameters:{ type:"object", properties:{ symbol:{type:"string"} }, required:["symbol"] } } },
  { type:"function", function:{ name:"analyze_market", description:"Full technical analysis: RSI, MACD, EMA, support/resistance with AI trading opinion", parameters:{ type:"object", properties:{ symbol:{type:"string"}, interval:{type:"string",description:"1min,5min,15min,30min,1h,4h,1day. Default 1h"} }, required:["symbol"] } } },
  { type:"function", function:{ name:"place_trade", description:"Place buy or sell trade on MT5", parameters:{ type:"object", properties:{ symbol:{type:"string"}, action:{type:"string",enum:["buy","sell"]}, volume:{type:"number"}, sl:{type:"number"}, tp:{type:"number"} }, required:["symbol","action","volume"] } } },
  { type:"function", function:{ name:"search_web", description:"Search the internet for current information", parameters:{ type:"object", properties:{ query:{type:"string"} }, required:["query"] } } },
  { type:"function", function:{ name:"read_screen", description:"Read all visible text from the current phone screen — use this after opening an app to see what loaded", parameters:{ type:"object", properties:{} } } },
  { type:"function", function:{ name:"tap_button", description:"Tap a button or element on screen by its text label", parameters:{ type:"object", properties:{ label:{type:"string"} }, required:["label"] } } },
  { type:"function", function:{ name:"type_text", description:"Type text into a field currently on screen", parameters:{ type:"object", properties:{ field:{type:"string",description:"hint/label of the field"}, value:{type:"string"} }, required:["value"] } } },
  { type:"function", function:{ name:"set_alarm", description:"Set an alarm", parameters:{ type:"object", properties:{ label:{type:"string"}, time:{type:"string"} }, required:["label","time"] } } },
  { type:"function", function:{ name:"set_timer", description:"Start a countdown timer", parameters:{ type:"object", properties:{ duration:{type:"string",description:"e.g. '10 minutes'"} }, required:["duration"] } } },
  { type:"function", function:{ name:"play_music", description:"Play music on Spotify or YouTube", parameters:{ type:"object", properties:{ query:{type:"string"} }, required:["query"] } } },
  { type:"function", function:{ name:"navigate_to", description:"Open Google Maps navigation", parameters:{ type:"object", properties:{ destination:{type:"string"} }, required:["destination"] } } },
  { type:"function", function:{ name:"get_weather", description:"Get current weather for a city", parameters:{ type:"object", properties:{ city:{type:"string"} }, required:["city"] } } },
];


// ═══════════════════════════════════════════════════════════════
// ROUTES
// ═══════════════════════════════════════════════════════════════
app.get("/", (_req, res) => res.json({ name:"FRIT Server", status:"online",
  endpoints:["/health","/chat","/automate","/market/analyze","/market/batch","/market/quote","/transcribe","/weather","/health"] }));

app.get("/health", (_req, res) => res.json({ status:"active", engine:"FRIT-Core",
  models:MODELS, twelve_data:!!TWELVE_DATA_KEY, uptime:Math.floor(process.uptime())+"s" }));

// ── /chat ─────────────────────────────────────────────────────────
app.post("/chat", async (req, res) => {
  const { message, history=[], memory=[], screen_context, mode="auto" } = req.body||{};
  if (!message) return res.status(400).json({ error:"No message provided" });
  const hasImage = !!screen_context;
  const model = pickModel(message, hasImage, mode);
  const systemPrompt = [
    "You are FRIT — a sharp, professional AI assistant on a custom Android system.",
    "You have full phone control: calls, WhatsApp, SMS, alarms, timers, music, navigation, market analysis, MT5 trading, notes, settings, any app.",
    "When given market data or technical analysis, give a clear trading opinion: direction, entry, SL, TP.",
    "The user is a forex/crypto trader based in Lagos, Nigeria. Be concise and decisive.",
    memory.length ? `User memory:\n${memory.join("\n")}` : ""
  ].filter(Boolean).join("\n\n");
  const userContent = hasImage
    ? [{ type:"text", text:message },{ type:"image_url", image_url:{ url:`data:image/jpeg;base64,${screen_context}` } }]
    : message;
  try {
    const result = await groqChat({ model, messages:[{role:"system",content:systemPrompt},...history,{role:"user",content:userContent}], max_tokens:1500 });
    res.json({ text:result.choices[0].message.content, model_used:model });
  } catch(err) { res.status(500).json({ error:"AI request failed", details:err.message }); }
});


// ── /automate — AGENTIC LOOP (multi-step, follows through till done) ──────────
app.post("/automate", async (req, res) => {
  const { task, context="", screen_text="" } = req.body||{};
  if (!task) return res.status(400).json({ error:"No task provided" });

  const systemPrompt = `You are FRIT's agentic engine. Complete tasks end-to-end on an Android phone.
Rules:
1. NEVER stop after one step. Chain tools until the task is fully done.
2. After open_app → ALWAYS call read_screen to see what loaded.
3. After read_screen → use tap_button or type_text to interact with what you see.
4. For trading: call analyze_market FIRST (unless user says "force"), then place_trade if signal is STRONG.
5. For WhatsApp "send message to X": open_app(whatsapp) → read_screen → tap_button(search/X's name) → type_text(message) → tap_button(send).
6. Current screen: ${screen_text||"Unknown — call read_screen first"}
7. User context: forex/crypto trader in Lagos Nigeria. Preferred pairs: EURUSD, GBPUSD, XAUUSD.`;

  const messages = [
    { role:"system", content:systemPrompt },
    { role:"user", content:`Task: ${task}\nContext: ${context}` }
  ];
  const allActions = [];
  let finalText = "";

  // Agentic loop — up to 6 iterations
  for (let i=0; i<6; i++) {
    let result;
    try { result = await groqChat({ model:MODELS.tools, messages, tools:AGENT_TOOLS, max_tokens:1200 }); }
    catch(err) { return res.status(500).json({ error:"AI failed", details:err.message }); }
    const choice = result.choices[0];
    if (choice.finish_reason !== "tool_calls" || !choice.message.tool_calls?.length) {
      finalText = choice.message.content||"";
      break;
    }
    messages.push({ role:"assistant", content:choice.message.content||null, tool_calls:choice.message.tool_calls });
    const toolResults = [];
    for (const tc of choice.message.tool_calls) {
      const name = tc.function.name;
      let args = {}; try { args = JSON.parse(tc.function.arguments); } catch {}
      let toolResult = "";
      if (name==="analyze_market") {
        const analysis = await analyzeSymbol(args.symbol, args.interval||"1h");
        // Also get AI trading opinion
        const aiRes = await groqChat({ model:MODELS.conversation, max_tokens:400, temperature:0.3,
          messages:[{ role:"user", content:`Technical data: ${JSON.stringify(analysis)}\nGive: direction, entry, SL, TP with specific numbers. Be brief.` }] });
        analysis.ai_opinion = aiRes.choices[0].message.content;
        toolResult = JSON.stringify(analysis);
        allActions.push({ tool:name, result:analysis, server_side:true });
      } else if (name==="get_market_data") {
        const data = await fetchMarketPrices([args.symbol.toUpperCase()]);
        toolResult = JSON.stringify(data[args.symbol.toUpperCase()]);
        allActions.push({ tool:name, result:data[args.symbol.toUpperCase()], server_side:true });
      } else if (name==="search_web") {
        toolResult = await webSearch(args.query);
        allActions.push({ tool:name, result:toolResult, server_side:true });
      } else if (name==="get_weather") {
        toolResult = await getWeather(args.city);
        allActions.push({ tool:name, result:toolResult, server_side:true });
      } else {
        // Android-side: send to device for execution
        toolResult = `Queued for Android: ${name}`;
        allActions.push({ tool:name, args, action:"android_execute" });
      }
      toolResults.push({ role:"tool", tool_call_id:tc.id, content:toolResult });
    }
    messages.push(...toolResults);
  }
  res.json({ type:"tool_result", actions:allActions, summary:finalText, model_used:MODELS.tools });
});


// ── /market/analyze — technical analysis + AI opinion ────────────
app.post("/market/analyze", async (req, res) => {
  const { symbol, interval="1h", user_context="" } = req.body||{};
  if (!symbol) return res.status(400).json({ error:"symbol required" });
  try {
    const analysis = await analyzeSymbol(symbol.toUpperCase(), interval);
    if (analysis.error) return res.json(analysis);
    const aiRes = await groqChat({ model:MODELS.conversation, max_tokens:500, temperature:0.3,
      messages:[{ role:"user", content:`You are a professional forex/crypto analyst.\nData: ${JSON.stringify(analysis)}\nUser: ${user_context||"Lagos-based trader"}\nGive clear trading recommendation: direction, confidence %, entry, stop loss, take profit. Use specific price numbers.` }] });
    res.json({ ...analysis, ai_opinion: aiRes.choices[0].message.content });
  } catch(err) { res.status(500).json({ error:err.message }); }
});

// ── /analyze — deep reasoning ─────────────────────────────────────
app.post("/analyze", async (req, res) => {
  const { prompt, context="" } = req.body||{};
  if (!prompt) return res.status(400).json({ error:"prompt required" });
  try {
    const result = await groqChat({ model:MODELS.conversation,
      messages:[{role:"system",content:"You are FRIT intelligence core. Provide deep expert analysis."},
                {role:"user",content:context?`Context:\n${context}\n\nTask:\n${prompt}`:prompt}], max_tokens:2048, temperature:0.3 });
    res.json({ text:result.choices[0].message.content, model_used:MODELS.conversation });
  } catch(err) { res.status(500).json({ error:err.message }); }
});

// ── /market/quote ─────────────────────────────────────────────────
app.get("/market/quote", async (req, res) => {
  const { symbol } = req.query;
  if (!symbol) return res.status(400).json({ error:"symbol required" });
  const spot = await fetchSpotPrice(symbol.toUpperCase());
  if (!spot) return res.status(404).json({ error:"Symbol not found" });
  res.json({ symbol, ...spot });
});

// ── /market/batch ─────────────────────────────────────────────────
app.post("/market/batch", async (req, res) => {
  const { symbols=["EURUSD","GBPUSD","XAUUSD","BTC","USDNGN"] } = req.body;
  try { res.json(await fetchMarketPrices(symbols)); }
  catch(err) { res.status(500).json({ error:err.message }); }
});

// ── /transcribe — Whisper STT ─────────────────────────────────────
app.post("/transcribe", async (req, res) => {
  const { audio_base64, language="en" } = req.body||{};
  if (!audio_base64) return res.status(400).json({ error:"audio_base64 required" });
  try {
    const audioBuffer = Buffer.from(audio_base64, "base64");
    if (!audioBuffer||audioBuffer.length<1000) return res.status(400).json({ error:"Audio too short or empty" });
    const form = new FormData();
    form.append("file", audioBuffer, { filename:"audio.wav", contentType:"audio/wav" });
    form.append("model", MODELS.whisper);
    form.append("language", language);
    form.append("response_format", "json");
    const whisperRes = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
      method:"POST", headers:{ "Authorization":`Bearer ${GROQ_API_KEY}`, ...form.getHeaders() }, body:form });
    const result = await whisperRes.json();
    if (!whisperRes.ok) throw new Error(JSON.stringify(result));
    res.json({ text:result.text||"" });
  } catch(err) { res.status(500).json({ error:"Transcription failed", details:err.message }); }
});

// ── /weather ──────────────────────────────────────────────────────
app.get("/weather", async (req, res) => {
  const { city="Lagos" } = req.query;
  res.json({ result: await getWeather(city) });
});

// ── /reminder ─────────────────────────────────────────────────────
app.post("/reminder", (req, res) => {
  const { text, delay_seconds } = req.body||{};
  if (!text) return res.status(400).json({ error:"text required" });
  res.json({ scheduled:true, task:text, in_seconds:delay_seconds||0 });
});

// ── START ─────────────────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
  console.log(`
  🚀 FRIT SERVER
  ─────────────────────────────────────────────────────
  Port:         ${PORT}
  Twelve Data:  ${TWELVE_DATA_KEY?"✅ Connected (forex+crypto OHLC)":"⚠️  Not set — add TWELVE_DATA_KEY env var (free at twelvedata.com)"}
  ─────────────────────────────────────────────────────
  Models:
    Vision/Screen:   ${MODELS.vision}
    Conversation:    ${MODELS.conversation}
    Tools/Agents:    ${MODELS.tools}
    Whisper STT:     ${MODELS.whisper}
  ─────────────────────────────────────────────────────
  Endpoints:
    POST /chat              AI chat (smart routing)
    POST /automate          Agentic task loop (multi-step)
    POST /market/analyze    Technical analysis + AI opinion
    GET  /market/quote      Live price (?symbol=EURUSD)
    POST /market/batch      Multi-symbol prices
    POST /transcribe        Whisper STT
    GET  /weather           Weather (?city=Lagos)
    GET  /health            Status
  `);
});
