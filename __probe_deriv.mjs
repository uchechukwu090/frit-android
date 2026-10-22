// Probe the live Deriv API (anonymous - no token needed for these calls)
// to find which symbols actually offer MULTUP/MULTDN.
import WebSocket from "ws";

const WS_URL = "wss://ws.derivws.com/websockets/v3";
let reqId = 0;
const pending = new Map();
let ws;

function connect() {
  return new Promise((resolve, reject) => {
    ws = new WebSocket(`${WS_URL}?app_id=1089`);
    ws.on("open", () => resolve());
    ws.on("error", reject);
    ws.on("message", (d) => {
      let m; try { m = JSON.parse(String(d)); } catch { return; }
      const id = m.req_id;
      if (id != null && pending.has(id)) { pending.get(id)(m); pending.delete(id); }
    });
  });
}
function call(obj, timeout = 20000) {
  return new Promise((resolve) => {
    const id = ++reqId;
    pending.set(id, resolve);
    ws.send(JSON.stringify({ ...obj, req_id: id }));
    setTimeout(() => { if (pending.has(id)) { pending.delete(id); resolve({ error: { message: "timeout" } }); } }, timeout);
  });
}

const TARGETS = ["EURUSD", "XAUUSD", "XAGUSD", "GBPUSD", "USDJPY", "AUDUSD",
  "frxEURUSD", "frxXAUUSD", "frxXAGUSD", "R_100", "1HZ100V", "OTC_DJI"];

await connect();

const basic = await call({ active_symbols: "brief", product_type: "basic" });
const full = await call({ active_symbols: "brief" });
const basicSet = new Set(basic?.active_symbols || []);
const fullSet = new Set(full?.active_symbols || []);
console.log("basic count:", basicSet.size, " full count:", fullSet.size);
console.log("basic has EURUSD?", basicSet.has("EURUSD"), " frxEURUSD?", basicSet.has("frxEURUSD"));
console.log("full  has EURUSD?", fullSet.has("EURUSD"), " frxEURUSD?", fullSet.has("frxEURUSD"));

console.log("\n--- contracts_for: MULT availability ---");
for (const sym of TARGETS) {
  const r = await call({ contracts_for: sym });
  if (r.error) { console.log(`${sym.padEnd(12)} ERROR ${r.error.message}`); continue; }
  const avail = r.contracts_for?.available || [];
  const types = new Set(avail.flatMap(a => [a.contract_type, ...(a.supported_contract_types || [])].filter(Boolean)));
  const hasMult = types.has("MULTUP") && types.has("MULTDN");
  const hasRf = types.has("CALL") && types.has("PUT");
  console.log(`${sym.padEnd(12)} MULT=${hasMult ? "YES" : "no "}  RiseFall=${hasRf ? "YES" : "no "}  types=[${[...types].slice(0, 8).join(",")}]`);
}
ws.close();
process.exit(0);
