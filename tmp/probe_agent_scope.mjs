// Probe: does POST /creative-suite/design/agent reference variables that are
// declared inside the SIBLING /creative-suite/design handler?
// It extracts both handlers from src/index.js and runs them with stubs.
import { readFileSync } from "node:fs";

const src = readFileSync(new URL("../src/index.js", import.meta.url), "utf8");

function extractHandler(marker) {
  const at = src.indexOf(marker);
  if (at < 0) throw new Error("marker not found: " + marker);
  const arrow = src.indexOf("=>", at);
  const open = src.indexOf("{", arrow);
  let depth = 0, i = open;
  for (; i < src.length; i++) {
    const ch = src[i];
    if (ch === "{") depth++;
    else if (ch === "}") { depth--; if (depth === 0) break; }
  }
  return src.slice(arrow, i + 1);
}

const design = extractHandler('app.post("/creative-suite/design",');
const agent = extractHandler('app.post("/creative-suite/design/agent",');

// Is the identifier declared anywhere in the agent handler?
for (const name of ["templateForAgent", "tpl"]) {
  const decl = new RegExp(`\\b(const|let|var)\\s+${name}\\b|\\b${name}\\s*=`).test(agent);
  console.log(`agent handler declares "${name}": ${decl}`);
}

// Run the agent handler with every module-level dependency stubbed EXCEPT
// templateForAgent / tpl (those must come from the handler's own scope).
const stubs = `
  const designAgent = { runDesignAgent: async () => ({ ok: true, render: { image_url: "u" } }) };
  const boundUser = () => "u1";
  const checkEntitlement = () => ({ ok: true });
  const applyBrand = () => ({ brand: null });
  const db = { prepare: () => ({ run: () => {} }) };
  const debitCredits = () => {};
  const todayStr = () => "2026-10-27";
  const templateLibrary = { getGraphicsTemplate: (id) => (id ? { id, name: "Event Flyer", aspect_ratio: "9:16", zones: ["headline"], hint: "x" } : null) };
  const designAgentDeps = {};
  const imageProviders = {};
  const generateImage = async () => ({});
  const chatWithFallback = async () => ({});
  const refs = {};
  const typeLayer = {};
  const PUBLIC_BASE_URL = "";
  const console = { warn: () => {}, error: () => {} };
`;
const body = `return (async () => {\n${stubs}\nconst handler = async (req, res) ${agent};\nconst req = { body: { goal: "a flyer for a church fundraiser", template: "gfx_church" } };\nlet out = null;\nconst res = { status(c) { out = { code: c }; return this; }, json(j) { out = { ...(out||{}), body: j }; } };\nawait handler(req, res);\nreturn out;\n})();`;

try {
  const fn = new Function(body);
  const out = await fn();
  console.log("agent handler response:", JSON.stringify(out));
} catch (e) {
  console.log("agent handler THREW:", e.constructor.name + ": " + e.message);
}
