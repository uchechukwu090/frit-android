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

// Production: trust Railway/Render/Fly reverse proxy for correct IP and HTTPS
app.set("trust proxy", 1);

if (!GROQ_API_KEY) {
  console.error("❌ ERROR: GROQ_API_KEY is missing. Set it as an environment variable.");
  process.exit(1);
}

// ─── MODEL REGISTRY ─────────────────────────────────────────────────────────
// Updated March 2026: llama3-8b-8192 and llama3-groq-70b tool-use are deprecated
const MODELS = {
  vision:    "meta-llama/llama-4-scout-17b-16e-instruct",  // vision + general
  reasoning: "llama-3.3-70b-versatile",                    // deep analysis
  fast:      "llama-3.1-8b-instant",                       // quick chat (replaces llama3-8b-8192)
  tools:     "llama-3.3-70b-versatile",                    // tool calling (70b versatile supports tools)
  whisper:   "whisper-large-v3",                           // speech-to-text
};

// ─── MIDDLEWARE ──────────────────────────────────────────────────────────────
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));
app.use(helmet({ contentSecurityPolicy: false }));

// CORS: allow the Android app (any origin) and future web dashboards
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "*").split(",").map(s => s.trim());
app.use(cors({
  origin: (origin, callback) => {
    if (!origin || ALLOWED_ORIGINS.includes("*") || ALLOWED_ORIGINS.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error("Not allowed by CORS"));
    }
  },
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"]
}));

app.use(morgan(process.env.NODE_ENV === "production" ? "combined" : "dev"));

// ─── HELPERS ─────────────────────────────────────────────────────────────────
async function groqChat({ model, messages, tools = null, temperature = 0.5, max_tokens = 1024 }) {
  const body = { model, messages, temperature, max_tokens };
  if (tools) body.tools = tools;

  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${GROQ_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });

  const data = await res.json();
  if (!res.ok) throw new Error(JSON.stringify(data));
  return data;
}

function pickModel(message = "", hasImage = false, mode = "auto") {
  if (mode === "tools")  return MODELS.tools;
  if (mode === "reason") return MODELS.reasoning;
  if (mode === "fast")   return MODELS.fast;
  if (hasImage)          return MODELS.vision;
  if (message.length > 300 || /analyz|explain|why|compare|research|market|chart|plan|trade/i.test(message))
    return MODELS.reasoning;
  return MODELS.fast;
}

// ─── / root ──────────────────────────────────────────────────────────────────
app.get("/", (_req, res) => {
  res.json({
    name: "FRIT Server",
    status: "online",
    endpoints: ["/health", "/chat", "/automate", "/analyze", "/transcribe", "/market/quote", "/market/batch", "/weather", "/reminder"]
  });
});

// ─── /health ─────────────────────────────────────────────────────────────────
app.get("/health", (_req, res) => {
  res.json({
    status: "active",
    engine: "FRIT-Core",
    env: process.env.NODE_ENV || "development",
    models: MODELS,
    uptime: Math.floor(process.uptime()) + "s"
  });
});

// ─── /chat  (smart model routing) ────────────────────────────────────────────
app.post("/chat", async (req, res) => {
  const { message, history = [], memory = [], screen_context, mode = "auto" } = req.body || {};
  if (!message) return res.status(400).json({ error: "No message provided" });

  const hasImage = !!screen_context;
  const model = pickModel(message, hasImage, mode);

  const systemPrompt = [
    "You are FRIT — a sharp, professional AI assistant running on a custom Android system.",
    "Be concise but complete. Never refuse reasonable requests.",
    "You have full access to the user's phone: calls, WhatsApp, SMS, alarms, timers, music, navigation, notes, settings, and any app.",
    "When the user says something like 'call mum', 'set alarm for 7am', 'play Afrobeats', 'navigate to Lekki', 'remind me in 10 minutes' — execute it directly via the automation system.",
    "You also have live market data, web search, and weather.",
    "Always respond in the user's conversational tone. Be helpful like a personal assistant, not a chatbot.",
    memory.length ? `User context & memory:\n${memory.join("\n")}` : ""
  ].filter(Boolean).join("\n\n");

  const userContent = hasImage
    ? [{ type: "text", text: message }, { type: "image_url", image_url: { url: `data:image/jpeg;base64,${screen_context}` } }]
    : message;

  const messages = [
    { role: "system", content: systemPrompt },
    ...history,
    { role: "user", content: userContent }
  ];

  try {
    const result = await groqChat({ model, messages, max_tokens: 1500 });
    res.json({ text: result.choices[0].message.content, model_used: model });
  } catch (err) {
    console.error("Chat error:", err.message);
    res.status(500).json({ error: "AI request failed", details: err.message });
  }
});

// ─── /automate  (tool-calling + action dispatch) ─────────────────────────────
app.post("/automate", async (req, res) => {
  const { task, context = "" } = req.body || {};
  if (!task) return res.status(400).json({ error: "No task provided" });

  const tools = [
    { type: "function", function: { name: "search_web", description: "Search the internet for current info, news, facts", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
    { type: "function", function: { name: "open_app", description: "Open an app on Android (e.g. WhatsApp, Chrome, MT5, Instagram)", parameters: { type: "object", properties: { app_name: { type: "string" } }, required: ["app_name"] } } },
    { type: "function", function: { name: "make_call", description: "Call a contact or phone number", parameters: { type: "object", properties: { contact_name: { type: "string" }, phone_number: { type: "string" } }, required: [] } } },
    { type: "function", function: { name: "send_whatsapp", description: "Send a WhatsApp message to a contact", parameters: { type: "object", properties: { contact_name: { type: "string" }, message: { type: "string" } }, required: ["message"] } } },
    { type: "function", function: { name: "send_sms", description: "Send an SMS text message", parameters: { type: "object", properties: { contact_name: { type: "string" }, message: { type: "string" } }, required: ["message"] } } },
    { type: "function", function: { name: "get_market_data", description: "Fetch live crypto/forex prices", parameters: { type: "object", properties: { symbols: { type: "array", items: { type: "string" } } }, required: ["symbols"] } } },
    { type: "function", function: { name: "get_weather", description: "Get current weather for a city", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } },
    { type: "function", function: { name: "set_alarm", description: "Set an alarm or reminder", parameters: { type: "object", properties: { label: { type: "string" }, time: { type: "string", description: "e.g. '7:30 AM' or 'in 30 minutes'" } }, required: ["label"] } } },
    { type: "function", function: { name: "set_timer", description: "Start a countdown timer", parameters: { type: "object", properties: { duration: { type: "string", description: "e.g. '10 minutes', '30 seconds'" } }, required: ["duration"] } } },
    { type: "function", function: { name: "play_music", description: "Play music on Spotify or YouTube", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
    { type: "function", function: { name: "navigate_to", description: "Open Google Maps navigation to a location", parameters: { type: "object", properties: { destination: { type: "string" } }, required: ["destination"] } } },
    { type: "function", function: { name: "schedule_task", description: "Schedule a background recurring task", parameters: { type: "object", properties: { task_description: { type: "string" }, delay_seconds: { type: "number" } }, required: ["task_description"] } } },
    { type: "function", function: { name: "take_note", description: "Save a note or memo", parameters: { type: "object", properties: { title: { type: "string" }, content: { type: "string" } }, required: ["content"] } } },
    { type: "function", function: { name: "open_settings", description: "Open phone settings (WiFi, Bluetooth, etc.)", parameters: { type: "object", properties: { section: { type: "string", description: "wifi, bluetooth, battery, display, sound, all" } }, required: [] } } }
  ];

  const messages = [
    { role: "system", content: `You are FRIT — an agentic AI assistant with full control over the user's Android phone and PC.
You can call, WhatsApp, SMS, set alarms/timers, play music, navigate, search, take notes, open any app, check weather, get market data, and automate background tasks.
Rules:
1. ALWAYS call the most appropriate tool. Never say "I can't do that" if a tool exists for it.
2. For "call mum" → use make_call with contact_name="mum". For "WhatsApp dad" → use send_whatsapp.
3. For "set alarm 7am" → use set_alarm. For "remind me in 20 min" → use set_timer.
4. For "play Burna Boy" → use play_music. For "navigate to Victoria Island" → use navigate_to.
5. Call ALL tools needed. If the task needs two tools, call both.` },
    { role: "user", content: `Task: ${task}\nContext: ${context}` }
  ];

  try {
    const result = await groqChat({ model: MODELS.tools, messages, tools, max_tokens: 1000 });
    const choice = result.choices[0];

    if (choice.finish_reason === "tool_calls" && choice.message.tool_calls) {
      const calls = choice.message.tool_calls.map(tc => ({
        name: tc.function.name,
        args: JSON.parse(tc.function.arguments)
      }));

      const toolResults = await Promise.all(calls.map(async (call) => {
        if (call.name === "get_market_data") {
          const prices = await fetchMarketPrices(call.args.symbols);
          return { tool: call.name, result: prices };
        }
        if (call.name === "search_web") {
          const searchResult = await webSearch(call.args.query);
          return { tool: call.name, result: searchResult };
        }
        if (call.name === "get_weather") {
          try {
            const city = call.args.city || "Lagos";
            const geoRes = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`);
            const geoData = await geoRes.json();
            const loc = geoData.results?.[0];
            if (!loc) return { tool: call.name, result: "City not found" };
            const wRes = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${loc.latitude}&longitude=${loc.longitude}&current_weather=true&timezone=auto`);
            const wData = await wRes.json();
            const cw = wData.current_weather;
            return { tool: call.name, result: `${loc.name}: ${cw.temperature}°C, wind ${cw.windspeed}km/h` };
          } catch(e) { return { tool: call.name, result: "Weather unavailable" }; }
        }
        // All other tools → send to Android to execute locally
        return { tool: call.name, args: call.args, action: "android_execute" };
      }));

      return res.json({ type: "tool_result", actions: toolResults, model_used: MODELS.tools });
    }

    res.json({ type: "text", text: choice.message.content, model_used: MODELS.tools });
  } catch (err) {
    console.error("Automation error:", err.message);
    res.status(500).json({ error: "Automation failed", details: err.message });
  }
});

// ─── /analyze  (deep reasoning) ──────────────────────────────────────────────
app.post("/analyze", async (req, res) => {
  const { prompt, context = "" } = req.body || {};
  if (!prompt) return res.status(400).json({ error: "prompt required" });

  const messages = [
    { role: "system", content: "You are FRIT intelligence core. Provide deep, expert-level analysis. Be thorough and precise." },
    { role: "user", content: context ? `Context:\n${context}\n\nTask:\n${prompt}` : prompt }
  ];

  try {
    const result = await groqChat({ model: MODELS.reasoning, messages, max_tokens: 2048, temperature: 0.3 });
    res.json({ text: result.choices[0].message.content, model_used: MODELS.reasoning });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── /market/quote  ───────────────────────────────────────────────────────────
app.get("/market/quote", async (req, res) => {
  const { symbol } = req.query;
  if (!symbol) return res.status(400).json({ error: "symbol required" });
  try {
    const data = await fetchMarketPrices([symbol.toUpperCase()]);
    const item = data[symbol.toUpperCase()];
    if (!item) return res.status(404).json({ error: "Symbol not found" });
    res.json(item);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── /market/batch  ──────────────────────────────────────────────────────────
app.post("/market/batch", async (req, res) => {
  const { symbols = ["BTC", "ETH", "USDNGN"] } = req.body;
  try {
    const data = await fetchMarketPrices(symbols);
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── /transcribe  (Whisper STT) ───────────────────────────────────────────────
app.post("/transcribe", async (req, res) => {
  const { audio_base64, language = "en" } = req.body || {};
  if (!audio_base64) return res.status(400).json({ error: "audio_base64 required" });

  try {
    const audioBuffer = Buffer.from(audio_base64, "base64");
    if (!audioBuffer || audioBuffer.length === 0) {
      return res.status(400).json({ error: "Empty audio buffer received" });
    }

    // CRITICAL FIX: Use the `form-data` npm package (not Node built-in FormData)
    // Node's built-in Blob+FormData causes "zero-length or detached ArrayBuffer"
    // when used with node-fetch. The form-data package handles Buffers correctly.
    const form = new FormData();
    form.append("file", audioBuffer, { filename: "audio.wav", contentType: "audio/wav" });
    form.append("model", MODELS.whisper);
    form.append("language", language);
    form.append("response_format", "json");

    const whisperRes = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${GROQ_API_KEY}`,
        ...form.getHeaders()
      },
      body: form
    });

    const result = await whisperRes.json();
    if (!whisperRes.ok) throw new Error(JSON.stringify(result));
    res.json({ text: result.text || "" });
  } catch (err) {
    console.error("Transcription error:", err.message);
    res.status(500).json({ error: "Transcription failed", details: err.message });
  }
});

// ─── MARKET DATA HELPER ───────────────────────────────────────────────────────
const CRYPTO_IDS = {
  BTC: "bitcoin", ETH: "ethereum", SOL: "solana", BNB: "binancecoin",
  XRP: "ripple", DOGE: "dogecoin", ADA: "cardano"
};

async function fetchMarketPrices(symbols) {
  const result = {};
  const cryptoSymbols = symbols.filter(s => CRYPTO_IDS[s]);
  const wantsNGN = symbols.some(s => s === "USDNGN" || s === "NGN");

  if (cryptoSymbols.length > 0) {
    try {
      const ids = cryptoSymbols.map(s => CRYPTO_IDS[s]).join(",");
      const geckoRes = await fetch(
        `https://api.coingecko.com/api/v3/simple/price?ids=${ids}&vs_currencies=usd&include_24hr_change=true`,
        { headers: { "Accept": "application/json" } }
      );
      if (geckoRes.ok) {
        const data = await geckoRes.json();
        for (const sym of cryptoSymbols) {
          const id = CRYPTO_IDS[sym];
          if (data[id]) {
            result[sym] = {
              symbol: sym,
              price: data[id].usd,
              change24h: parseFloat((data[id].usd_24h_change || 0).toFixed(2)),
              currency: "USD"
            };
          }
        }
      }
    } catch (e) { console.error("CoinGecko error:", e.message); }
  }

  if (wantsNGN) {
    try {
      const fxRes = await fetch("https://open.er-api.com/v6/latest/USD");
      if (fxRes.ok) {
        const data = await fxRes.json();
        const ngnRate = data.rates?.NGN;
        if (ngnRate) {
          result["USDNGN"] = { symbol: "USDNGN", price: ngnRate, change24h: 0, currency: "NGN" };
          result["NGN"] = result["USDNGN"];
        }
      }
    } catch (e) { console.error("FX error:", e.message); }
  }

  for (const sym of symbols) {
    if (!result[sym]) result[sym] = { symbol: sym, price: 0, change24h: 0, currency: "USD", error: "Not found" };
  }

  return result;
}

// ─── /weather ────────────────────────────────────────────────────────────────
app.get("/weather", async (req, res) => {
  const { city = "Lagos" } = req.query;
  try {
    const geoRes = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`);
    const geoData = await geoRes.json();
    const loc = geoData.results?.[0];
    if (!loc) return res.status(404).json({ error: "City not found" });
    const wRes = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${loc.latitude}&longitude=${loc.longitude}&current_weather=true&hourly=precipitation_probability&timezone=auto`);
    const wData = await wRes.json();
    const cw = wData.current_weather;
    const rain = wData.hourly?.precipitation_probability?.[0] ?? 0;
    res.json({
      city: loc.name, country: loc.country,
      temp_c: cw.temperature, wind_kmh: cw.windspeed,
      condition: cw.weathercode <= 1 ? "Clear" : cw.weathercode <= 3 ? "Cloudy" : cw.weathercode <= 67 ? "Rainy" : "Stormy",
      rain_chance: rain + "%"
    });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ─── /reminder  ──────────────────────────────────────────────────────────────
// Server-side: just acknowledges. Actual alarm is set on Android via AutomationEngine.
app.post("/reminder", (req, res) => {
  const { text, delay_seconds } = req.body || {};
  if (!text) return res.status(400).json({ error: "text required" });
  res.json({ scheduled: true, task: text, in_seconds: delay_seconds || 0 });
});

// ─── WEB SEARCH HELPER ───────────────────────────────────────────────────────
async function webSearch(query) {
  try {
    const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`;
    const res = await fetch(url);
    const data = await res.json();
    return data.AbstractText || data.Answer || data.RelatedTopics?.[0]?.Text || "No result found.";
  } catch (e) {
    return "Search failed: " + e.message;
  }
}

// ─── START ────────────────────────────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
  const isProd = process.env.NODE_ENV === "production";
  console.log(`
  🚀 FRIT SERVER — ${isProd ? "PRODUCTION" : "DEVELOPMENT"}
  ─────────────────────────────────────────────────────────
  Port:     ${PORT}
  Env:      ${process.env.NODE_ENV || "development"}
  ─────────────────────────────────────────────────────────
  ENDPOINTS
    POST /chat          Smart AI routing
    POST /automate      Agentic tool-calling
    POST /analyze       Deep reasoning (70b)
    GET  /market/quote  Live price (?symbol=BTC)
    POST /market/batch  Multiple symbols
    POST /transcribe    Whisper speech-to-text
    GET  /health        Status check
  ─────────────────────────────────────────────────────────
  MODELS
    Vision:    ${MODELS.vision}
    Reasoning: ${MODELS.reasoning}
    Fast:      ${MODELS.fast}
    Whisper:   ${MODELS.whisper}
  ─────────────────────────────────────────────────────────
  ${isProd ? "Deployed — set GROQ_API_KEY in your platform env vars" : "Use your PC IP on phone, not 'localhost'"}
  `);
});
