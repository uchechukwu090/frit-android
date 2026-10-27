import WebSocket from "ws";

const HOSTS = [
  "wss://ws.binaryws.com/websockets/v3",
  "wss://ws.derivws.com/websockets/v3",
  "wss://ws.deriv.com/websockets/v3",
];

let reqId = 0;
const pending = new Map();
let ws;

function connect(url) {
  return new Promise((resolve, reject) => {
    try { ws = new WebSocket(`${url}?app_id=1089`); } catch (e) { return reject(e); }
    const t = setTimeout(() => reject(new Error("connect timeout")), 15000);
    ws.on("open", () => { clearTimeout(t); resolve(); });
    ws.on("error", (e) => { clearTimeout(t); reject(e); });
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
    try { ws.send(JSON.stringify({ ...obj, req_id: id })); } catch { resolve({ error: { message: "send failed" } }); return; }
    setTimeout(() => { if (pending.has(id)) { pending.delete(id); resolve({ error: { message: "timeout" } }); } }, timeout);
  });
}

let connected = null;
for (const h of HOSTS) {
  try { await connect(h); connected = h; console.log("connected:", h); break; }
  catch (e) { console.log("failed:", h, e.message.slice(0, 60)); }
}
if (!connected) { console.log("NO DERIV REACHABILITY FROM THIS HOST"); process.exit(2); }

const TARGETS = ["EURUSD", "XAUUSD", "XAGUSD", "GBPUSD", "USDJPY", "frxEURUSD", "frxXAUUSD", "frxXAGUSD", "R_100", "1HZ100V"];

const basic = await call({ active_symbols: "brief", product_type: "basic" });
const full = await call({ active_symbols: "brief" });
const basicSet = new Set(basic?.active_symbols || []);
const fullSet = new Set(full?.active_symbols || []);
console.log("basic count:", basicSet.size, "| full count:", fullSet.size);
console.log("basic has EURUSD?", basicSet.has("EURUSD"), "| frxEURUSD?", basicSet.has("frxEURUSD"));
console.log("full  has EURUSD?", fullSet.has("EURUSD"), "| frxEURUSD?", fullSet.has("frxEURUSD"));
console.log("full  has XAUUSD?", fullSet.has("XAUUSD"), "| frxXAUUSD?", fullSet.has("frxXAUUSD"));

console.log("\n--- contracts_for: MULT vs Rise/Fall availability ---");
for (const sym of TARGETS) {
  const r = await call({ contracts_for: sym });
  if (r.error) { console.log(`${sym.padEnd(12)} ERROR ${r.error.message}`); continue; }
  const avail = r.contracts_for?.available || [];
  const types = new Set(avail.flatMap(a => [a.contract_type, ...(a.supported_contract_types || [])].filter(Boolean)));
  console.log(`${sym.padEnd(12)} MULT=${types.has("MULTUP") && types.has("MULTDN") ? "YES" : "no "}` +
    `  RiseFall=${types.has("CALL") && types.has("PUT") ? "YES" : "no "}` +
    `  [${[...types].slice(0, 7).join(",")}]`);
}
try { ws.close(); } catch { }
process.exit(0);
