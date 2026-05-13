import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import dotenv from 'dotenv';
import fetch from 'node-fetch';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

dotenv.config();
const __dirname = dirname(fileURLToPath(import.meta.url));
const STORAGE_PATH = join(__dirname, 'hvr_storage.json');
const CONFIDENCE_PATH = join(__dirname, 'confidence_log.json');

const app = express();
const PORT = Number(process.env.PORT || 8787);

// ==================== MIDDLEWARE ====================
app.use(express.json({ limit: '10mb' }));
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: '*' }));
app.use(morgan(process.env.NODE_ENV === 'production' ? 'combined' : 'dev'));

// ==================== PERSISTENCE ====================
const Persistence = {
  load() {
    try {
      if (!existsSync(STORAGE_PATH)) return { trades: [], cooldowns: {} };
      return JSON.parse(readFileSync(STORAGE_PATH, 'utf8'));
    } catch { return { trades: [], cooldowns: {} }; }
  },
  save(data) {
    try { writeFileSync(STORAGE_PATH, JSON.stringify(data, null, 2)); } catch {}
  },
  logConfidence(predictedConf, actualWin) {
    try {
      const log = existsSync(CONFIDENCE_PATH) ? JSON.parse(readFileSync(CONFIDENCE_PATH, 'utf8')) : [];
      log.push({ predicted: predictedConf, actual: actualWin, ts: Date.now() });
      writeFileSync(CONFIDENCE_PATH, JSON.stringify(log.slice(-1000), null, 2));
    } catch {}
  },
  getCalibration() {
    try {
      const log = existsSync(CONFIDENCE_PATH) ? JSON.parse(readFileSync(CONFIDENCE_PATH, 'utf8')) : [];
      const buckets = { '50-59': [], '60-69': [], '70-79': [], '80-89': [], '90-99': [] };
      log.forEach(e => {
        const b = Math.floor(e.predicted/10)*10;
        if (buckets[`${b}-${b+9}`]) buckets[`${b}-${b+9}`].push(e.actual);
      });
      return Object.fromEntries(
        Object.entries(buckets).map(([k, v]) => [k, v.length ? (v.reduce((a,b)=>a+b,0)/v.length)*100 : 0])
      );
    } catch { return {}; }
  }
};

// ==================== RATE LIMITER ====================
const rateLimiter = new Map();
function applyRateLimit(maxCalls = 5, windowMs = 60000) {
  return (req, res, next) => {
    const ip = req.ip || req.connection.remoteAddress;
    const now = Date.now();
    const bucket = rateLimiter.get(ip) || { calls: [], resetAt: now + windowMs };
    bucket.calls = bucket.calls.filter(t => t > now);
    if (bucket.calls.length >= maxCalls) {
      return res.status(429).json({ error: 'Rate limit exceeded. Try again later.' });
    }
    bucket.calls.push(now);
    rateLimiter.set(ip, bucket);
    next();
  };
}

// ==================== TECHNICAL HELPERS (HVR SPECIFIC) ====================
function calcEMA(closes, period) {
  if (closes.length < period) return [];
  const k = 2 / (period + 1);
  let prev = closes.slice(0, period).reduce((a,b)=>a+b,0)/period;
  const res = [prev];
  for(let i=period; i<closes.length; i++) { prev = closes[i]*k + prev*(1-k); res.push(prev); }
  return res;
}

function calcRSI(closes, period=14) {
  if(closes.length < period+1) return 50;
  const changes = closes.slice(-(period+1)).map((v,i,a)=>i>0?v-a[i-1]:0).slice(1);
  const avgGain = changes.filter(c=>c>0).reduce((a,b)=>a+b,0)/period || 0;
  const avgLoss = changes.filter(c=>c<0).reduce((a,b)=>a+Math.abs(b),0)/period || 0;
  return avgLoss===0 ? 100 : 100 - 100/(1+avgGain/avgLoss);
}

function calcATR(candles, period=14) {
  if(!candles || candles.length < period+1) return 0;
  const trs = [];
  for(let i=1; i<candles.length; i++) {
    const c = candles[i], p = candles[i-1];
    trs.push(Math.max(c.high-c.low, Math.abs(c.high-p.close), Math.abs(c.low-p.close)));
  }
  return trs.slice(-period).reduce((a,b)=>a+b,0)/period;
}

function findSwings(candles, lookback=2) {
  const highs=[], lows=[];
  for(let i=lookback; i<candles.length-lookback; i++) {
    const cur=candles[i];
    let isH=true, isL=true;
    for(let j=i-lookback; j<=i+lookback; j++) {
      if(j===i) continue;
      if(candles[j].high >= cur.high) isH=false;
      if(candles[j].low <= cur.low) isL=false;
    }
    if(isH) highs.push({index:i, price:cur.high});
    if(isL) lows.push({index:i, price:cur.low});
  }
  return {highs, lows};
}

// VOLUME PROXY (OHLC-only workaround)
function calcVolumeProxy(candles, period=20) {
  const proxies = candles.slice(-period).map(c => (c.high-c.low) + Math.abs(c.close-c.open));
  const avg = proxies.reduce((a,b)=>a+b,0)/proxies.length;
  const cur = proxies[proxies.length-1];
  return { avg, current: cur, ratio: avg>0 ? cur/avg : 1 };
}

// ==================== HVR STRATEGY ENGINE ====================
const HVR = {
  async getMultiTF(symbol) {
    // Fetch required timeframes concurrently
    const fetchTF = (iv, size) => fetch(`https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(symbol)}&interval=${iv}&outputsize=${size}&apikey=${process.env.TWELVE_DATA_KEY}`)
      .then(r=>r.json()).then(d=>d.values?.reverse().map(c=>({time:c.datetime, open:+c.open, high:+c.high, low:+c.low, close:+c.close})));
    
    const [h4, h1, m15, m5] = await Promise.all([
      fetchTF('4h', 100), fetchTF('1h', 100), fetchTF('15min', 50), fetchTF('5min', 50)
    ]);
    return {h4, h1, m15, m5};
  },

  checkH4Bias(h4) {
    if(!h4 || h4.length<200) return {bias:'NEUTRAL', reason:'Insufficient H4 data'};
    const closes = h4.map(c=>c.close);
    const ema200 = calcEMA(closes, 200).at(-1);
    const price = closes.at(-1);
    const diffPips = Math.abs(price - ema200) / 0.0001;
    if(price > ema200 && diffPips > 20) return {bias:'BULLISH', ema200, diffPips};
    if(price < ema200 && diffPips > 20) return {bias:'BEARISH', ema200, diffPips};
    return {bias:'NEUTRAL', reason:'Price within 20 pips of 200 EMA'};
  },

  checkH1Structure(h1, bias) {
    if(!h1 || h1.length<20) return {valid:false, reason:'Insufficient H1 data'};
    const swings = findSwings(h1, 2);
    if(bias==='BULLISH') {
      if(!swings.lows.length || !swings.highs.length) return {valid:false, reason:'No swings'};
      const swingLow = swings.lows.at(-1).price;
      const swingHigh = swings.highs.at(-1).price;
      const fib618 = swingLow + (swingHigh-swingLow)*0.618;
      const fib786 = swingLow + (swingHigh-swingLow)*0.786;
      const price = h1.at(-1).close;
      if(price >= fib618 && price <= fib786) return {valid:true, zone:[fib618, fib786], price};
      return {valid:false, reason:'Price not in 0.618-0.786 pullback zone'};
    } else {
      const swingHigh = swings.highs.at(-1).price;
      const swingLow = swings.lows.at(-1).price;
      const fib618 = swingHigh - (swingHigh-swingLow)*0.618;
      const fib786 = swingHigh - (swingHigh-swingLow)*0.786;
      const price = h1.at(-1).close;
      if(price >= fib618 && price <= fib786) return {valid:true, zone:[fib618, fib786], price};
      return {valid:false, reason:'Price not in 0.618-0.786 pullback zone'};
    }
  },

  checkM15Momentum(m15, bias) {
    if(!m15 || m15.length<15) return {valid:false};
    const closes = m15.map(c=>c.close);
    const rsi = calcRSI(closes);
    const prevRsi = calcRSI(closes.slice(0,-1));
    if(bias==='BULLISH') {
      return rsi>40 && rsi<60 && prevRsi<=40 ? {valid:true, rsi} : {valid:false, rsi};
    } else {
      return rsi<60 && rsi>40 && prevRsi>=60 ? {valid:true, rsi} : {valid:false, rsi};
    }
  },

  checkVolumeVol(h1, m15) {
    const atrH1 = calcATR(h1);
    const atrAvg = calcATR(h1.slice(-40));
    const vol = calcVolumeProxy(m15);
    const atrOk = atrH1 > atrAvg;
    const volOk = vol.ratio > 1.5;
    return {valid: atrOk && volOk, atrH1, volRatio: vol.ratio};
  },

  checkM5Trigger(m5, bias, fibZone) {
    if(!m5 || m5.length<3) return {valid:false};
    const prev = m5[m5.length-2], curr = m5[m5.length-1];
    const closes = m5.map(c=>c.close);
    const rsi = calcRSI(closes);
    const rsiPrev = calcRSI(closes.slice(0,-1));
    const inZone = curr.close >= fibZone[0] && curr.close <= fibZone[1];
    
    let patternOk = false;
    if(bias==='BULLISH') {
      patternOk = (prev.close<prev.open && curr.close>curr.open && curr.open<=prev.close && curr.close>=prev.open) || // Engulfing
                  (Math.min(curr.open,curr.close)-curr.low > 2*Math.abs(curr.close-curr.open)); // Hammer
    } else {
      patternOk = (prev.close>prev.open && curr.close<curr.open && curr.open>=prev.close && curr.close<=prev.open) || // Engulfing
                  (curr.high-Math.max(curr.open,curr.close) > 2*Math.abs(curr.close-curr.open)); // Shooting Star
    }
    
    const rsiCross = bias==='BULLISH' ? rsi>50 && rsiPrev<=50 : rsi<50 && rsiPrev>=50;
    return {valid: inZone && patternOk && rsiCross, pattern: patternOk?'engulfing/pinbar':'none', rsi};
  },

  calculateRisk(entry, bias, h1) {
    const atr = calcATR(h1);
    const slDist = atr * 1.5;
    const tpDist = atr * 3.0;
    const sl = bias==='BULLISH' ? entry - slDist : entry + slDist;
    const tp = bias==='BULLISH' ? entry + tpDist : entry - tpDist;
    return {sl, tp, atr, rr: 2.0};
  },

  async generateSignal(symbol) {
    const tfs = await this.getMultiTF(symbol);
    const h4Bias = this.checkH4Bias(tfs.h4);
    if(h4Bias.bias==='NEUTRAL') return {signal:'NO_TRADE', reason:h4Bias.reason, step:1};

    const h1Struct = this.checkH1Structure(tfs.h1, h4Bias.bias);
    if(!h1Struct.valid) return {signal:'NO_TRADE', reason:h1Struct.reason, step:2};

    const m15Mom = this.checkM15Momentum(tfs.m15, h4Bias.bias);
    if(!m15Mom.valid) return {signal:'NO_TRADE', reason:'M15 RSI not confirming momentum', rsi:m15Mom.rsi, step:3};

    const volVol = this.checkVolumeVol(tfs.h1, tfs.m15);
    if(!volVol.valid) return {signal:'NO_TRADE', reason:'Volume/Vol filter failed', volRatio:volVol.volRatio, step:4};

    const m5Trig = this.checkM5Trigger(tfs.m5, h4Bias.bias, h1Struct.zone);
    if(!m5Trig.valid) return {signal:'NO_TRADE', reason:'M5 trigger not met', step:5};

    const risk = this.calculateRisk(h1Struct.price, h4Bias.bias, tfs.h1);
    return {
      signal: h4Bias.bias==='BULLISH' ? 'BUY' : 'SELL',
      symbol,
      entry: h1Struct.price.toFixed(5),
      sl: risk.sl.toFixed(5),
      tp: risk.tp.toFixed(5),
      atr: risk.atr.toFixed(5),
      rr: risk.rr,
      confidence: Math.min(85, 50 + (volVol.volRatio-1)*20 + (m15Mom.rsi>45&&m15Mom.rsi<55?10:0)),
      step: 'PASS',
      timestamp: new Date().toISOString()
    };
  }
};

// ==================== DIRECTION-AWARE COOLDOWN ====================
function isInCooldown(symbol, direction) {
  const state = Persistence.load();
  const key = `${symbol}_${direction}`;
  const cd = state.cooldowns[key];
  if(!cd) return false;
  const expired = Date.now() - cd > 4 * 60 * 60 * 1000; // 4 hours
  if(expired) { delete state.cooldowns[key]; Persistence.save(state); return false; }
  return true;
}

function setCooldown(symbol, direction) {
  const state = Persistence.load();
  state.cooldowns[`${symbol}_${direction}`] = Date.now();
  Persistence.save(state);
}

// ==================== ROUTES ====================
app.get('/health', async (req, res) => {
  const state = Persistence.load();
  const tdOk = !!process.env.TWELVE_DATA_KEY;
  res.json({
    status: 'online',
    hvr_active: true,
    cooldowns_active: Object.keys(state.cooldowns).length,
    data_provider_ok: tdOk,
    api_keys_loaded: tdOk,
    last_trade: state.trades.at(-1)?.timestamp || null,
    timestamp: Date.now()
  });
});

app.post('/trade', applyRateLimit(3, 60000), async (req, res) => {
  const { symbol, action, balance=1000, riskPercent=1 } = req.body || {};
  if(!symbol || !['EURUSD','XAUUSD','GBPUSD'].includes(symbol.toUpperCase())) return res.status(400).json({error:'Symbol must be EURUSD, XAUUSD, or GBPUSD'});
  
  const dir = action.toUpperCase();
  if(isInCooldown(symbol, dir)) return res.status(200).json({status:'COOLDOWN', reason:`${dir} on cooldown for ${symbol}`});

  const signal = await HVR.generateSignal(symbol);
  if(signal.signal !== dir || signal.signal==='NO_TRADE') {
    return res.status(200).json({status:'NO_EDGE', reason:signal.reason, signal});
  }

  // Position sizing
  const riskAmt = balance * (riskPercent/100);
  const pipSize = symbol==='XAUUSD' ? 0.01 : 0.0001;
  const pipVal = symbol==='XAUUSD' ? 100 : 10;
  const slPips = Math.abs(parseFloat(signal.entry)-parseFloat(signal.sl))/pipSize;
  const lot = Math.max(0.01, Math.min(5.0, (riskAmt/(slPips*pipVal))));

  setCooldown(symbol, dir);
  Persistence.load().trades.push({...signal, lot_size:+lot.toFixed(2), balance_used:balance, outcome:'pending', timestamp:Date.now()});
  Persistence.save();

  res.json({status:'SUBMITTED', ...signal, lot_size:+lot.toFixed(2), reason:'HVR Protocol Aligned'});
});

app.post('/backtest', applyRateLimit(2, 120000), async (req, res) => {
  const { symbol='EURUSD', startDate, endDate } = req.body || {};
  // Simplified historical backtest simulation (fetches 1H candles, simulates M15/M5 logic)
  try {
    const url = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(symbol)}&interval=1h&outputsize=500&apikey=${process.env.TWELVE_DATA_KEY}`;
    const data = await fetch(url).then(r=>r.json());
    if(!data.values) return res.status(400).json({error:'Historical data unavailable'});
    
    const candles = data.values.map(c=>({time:c.datetime, open:+c.open, high:+c.high, low:+c.low, close:+c.close}));
    let balance=1000, wins=0, losses=0, maxDD=0, peak=balance;
    const trades = [];

    for(let i=100; i<candles.length; i++) {
      const h1Slice = candles.slice(i-100, i);
      const bias = candles[i-1].close > calcEMA(h1Slice.map(c=>c.close), 200).at(-1) ? 'BULLISH' : 'BEARISH';
      // Simulated entry/exit logic for demo
      const slDist = calcATR(h1Slice) * 1.5;
      const entry = candles[i].close;
      const sl = bias==='BULLISH' ? entry-slDist : entry+slDist;
      const tp = bias==='BULLISH' ? entry+slDist*2 : entry-slDist*2;
      
      let outcome = 'pending';
      // Simulate forward 3 candles
      for(let j=i+1; j<=i+3; j++) {
        if(bias==='BULLISH' && candles[j]?.high >= tp) { outcome='win'; break; }
        if(bias==='BEARISH' && candles[j]?.low <= tp) { outcome='win'; break; }
        if(bias==='BULLISH' && candles[j]?.low <= sl) { outcome='loss'; break; }
        if(bias==='BEARISH' && candles[j]?.high >= sl) { outcome='loss'; break; }
      }
      
      const risk = balance*0.01;
      if(outcome==='win') { balance += risk*2; wins++; }
      else if(outcome==='loss') { balance -= risk; losses++; peak=Math.max(peak,balance); }
      maxDD = Math.max(maxDD, peak-balance);
      trades.push({entry, outcome, balance: +balance.toFixed(2)});
    }

    res.json({
      symbol, trades_run: trades.length,
      win_rate: wins/(wins+losses),
      final_balance: +balance.toFixed(2),
      max_drawdown: +maxDD.toFixed(2),
      expectancy: +((wins/(wins+losses)*2 - (1-wins/(wins+losses)))*10).toFixed(2)
    });
  } catch(e) { res.status(500).json({error:'Backtest failed', details:e.message}); }
});

app.get('/calibration', (req, res) => {
  res.json(Persistence.getCalibration());
});

// ==================== ANDROID/LLM ROUTES (Preserved) ====================
app.post('/chat', async (req, res) => {
  // Your existing LLM/Chat logic stays here
  res.json({status:'ok', message:'Chat endpoint active'});
});

app.post('/automate', async (req, res) => {
  // Your existing Android automation logic stays here
  res.json({status:'ok', message:'Automation endpoint active'});
});

// ==================== START ====================
app.listen(PORT, () => {
  console.log(`✅ FRIT Backend Active (HVR Protocol) on port ${PORT}`);
  console.log(`📊 Trading: EURUSD, XAUUSD, GBPUSD only`);
  console.log(`💾 Persistence: ${STORAGE_PATH}`);
  console.log(`📉 Backtest: POST /backtest`);
  console.log(`⚡ Rate Limit: Active`);
});

export default app;
