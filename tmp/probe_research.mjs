// Live probe: does the design agent's inspiration search actually return
// Pinterest/Behance-grade image URLs, and do the reference bytes load?
import { searchInspiration, planInspirationQueries } from "../src/creative/designAgent.js";
import * as refs from "../src/creative/refs.js";

const goal = "church fundraiser flyer with gold accents";
console.log("queries:", planInspirationQueries(goal, { seed: 42 }));

const r = await searchInspiration(goal, { maxRefs: 3, maxQueries: 3, fetchImpl: fetch, seed: 42 });
console.log("refs found:", r.refs.length, "errors:", r.errors);
for (const u of r.refs) console.log("  -", u.slice(0, 120));

let loaded = 0;
for (const u of r.refs) {
  const got = await refs.loadReference(u, { fetchImpl: fetch });
  console.log("load", got ? `OK id=${got.id} ${got.bytes}B ${got.mime}` : "FAILED", u.slice(0, 80));
  if (got) loaded++;
}
console.log(`loaded ${loaded}/${r.refs.length}`);
