// Regression tests for the TWO streaming bugs found while building F-43.
//
// Both of these silently defeated streaming while the unit tests still
// passed, so they are pinned here with tests that fail if the bug returns.
//
// BUG 1 — node-fetch v3's `res.body` is a Node.js PassThrough, NOT a web
//         ReadableStream, so `res.body.getReader()` does not exist. The
//         "obvious" fallback (`await res.json()`) buffers the ENTIRE
//         response, which means the client gets one blob at the end and the
//         stream is pointless.
//
// BUG 2 — `req.on("close")` on an Express request fires as soon as the
//         request BODY has been consumed (express.json() already read it),
//         NOT when the client disconnects. A disconnect handler wired to
//         `req` therefore aborted the upstream fetch instantly, and every
//         "stream" came back as a single chunk. The handler must listen on
//         `res.on("close")`.
import test from "node:test";
import assert from "node:assert/strict";
import { PassThrough, Readable } from "stream";
import express from "express";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { readProviderStream } from "../src/streamProvider.js";

const INDEX_JS = fileURLToPath(new URL("../src/index.js", import.meta.url));
const readIndex = () => readFileSync(INDEX_JS, "utf8");

/**
 * Strip comments so the guards assert about CODE, not prose.
 *
 * This matters: the fix for BUG 2 is documented in a comment that literally
 * contains the string `req.on("close")`, and the fix for BUG 1 is documented
 * in a comment containing `await res.json()`. Asserting on raw source would
 * match the explanations of the bugs, not the bugs.
 */
function stripComments(src) {
  return String(src)
    .replace(/\/\*[\s\S]*?\*\//g, " ")            // block comments
    .replace(/(^|[^:])\/\/[^\n]*/g, "$1");        // line comments (keep "://" in URLs)
}

// ---------------------------------------------------------------------------
// BUG 1: the reader must handle a Node.js stream body
// ---------------------------------------------------------------------------

/** Build a fake node-fetch style response: body is a PassThrough. */
function nodeStyleResponse(chunks, { delayMs = 0 } = {}) {
  const body = new PassThrough();
  const res = { body, ok: true, status: 200 };
  // Push the chunks asynchronously, like a socket would.
  (async () => {
    for (const c of chunks) {
      if (delayMs) await new Promise((r) => setTimeout(r, delayMs));
      body.write(c);
    }
    body.end();
  })();
  return res;
}

/** Build a fake global-fetch style response: body is a web ReadableStream. */
function webStyleResponse(chunks, { delayMs = 0 } = {}) {
  let i = 0;
  const body = new ReadableStream({
    async pull(controller) {
      if (i >= chunks.length) { controller.close(); return; }
      if (delayMs) await new Promise((r) => setTimeout(r, delayMs));
      controller.enqueue(new TextEncoder().encode(chunks[i++]));
    },
  });
  return { body, ok: true, status: 200 };
}

const SSE_CHUNKS = [
  'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
  'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
  "data: [DONE]\n\n",
];

test("BUG 1: a Node.js stream body (node-fetch) yields deltas, not one blob", async () => {
  const res = nodeStyleResponse(SSE_CHUNKS);
  const seen = [];
  for await (const ev of readProviderStream(res)) seen.push(ev);
  const deltas = seen.filter((e) => e.type === "delta");
  assert.equal(deltas.length, 2, "expected 2 separate deltas");
  assert.equal(deltas[0].text, "Hel");
  assert.equal(deltas[1].text, "lo");
  assert.equal(seen.some((e) => e.type === "done"), true);
});

test("BUG 1: a web ReadableStream body (global fetch) yields the same deltas", async () => {
  const res = webStyleResponse(SSE_CHUNKS);
  const seen = [];
  for await (const ev of readProviderStream(res)) seen.push(ev);
  const deltas = seen.filter((e) => e.type === "delta");
  assert.equal(deltas.length, 2);
  assert.equal(deltas.map((d) => d.text).join(""), "Hello");
});

test("BUG 1: the reader never calls getReader() on a Node stream", async () => {
  // A Node PassThrough has no getReader — the old code threw here.
  const res = nodeStyleResponse(SSE_CHUNKS);
  assert.equal(typeof res.body.getReader, "undefined");
  const seen = [];
  for await (const ev of readProviderStream(res)) seen.push(ev);
  assert.ok(seen.length > 0, "reader must not throw on a Node stream");
});

test("BUG 1: the reader tolerates a response with no body", async () => {
  const seen = [];
  for await (const ev of readProviderStream({ ok: true, status: 200, body: null })) seen.push(ev);
  assert.equal(seen.length, 1);
  assert.equal(seen[0].type, "error");
});

test("BUG 1: an aborted signal stops the reader instead of hanging", async () => {
  const ac = new AbortController();
  ac.abort();
  const res = nodeStyleResponse(SSE_CHUNKS);
  const seen = [];
  for await (const ev of readProviderStream(res, ac.signal)) seen.push(ev);
  assert.ok(seen.length >= 1, "an aborted read must terminate");
  assert.equal(seen[seen.length - 1].type, "error");
});

// ---------------------------------------------------------------------------
// BUG 2: the disconnect handler must listen on the RESPONSE
// ---------------------------------------------------------------------------

test("BUG 2: req 'close' fires as soon as the body is read (so it cannot be the disconnect hook)", async () => {
  const app = express();
  app.use(express.json());
  let reqCloseFired = false;
  let resCloseFired = false;
  app.post("/t", (req, res) => {
    req.on("close", () => { reqCloseFired = true; });
    res.on("close", () => { resCloseFired = true; });
    setTimeout(() => { res.json({ ok: true }); }, 300);
  });
  const server = app.listen(0);
  await new Promise((r) => server.once("listening", r));
  const port = server.address().port;

  await fetch(`http://127.0.0.1:${port}/t`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ a: 1 }),
  });
  await new Promise((r) => setTimeout(r, 200));
  server.close();

  // The request 'close' fired — the body had been consumed, NOT a disconnect.
  assert.equal(reqCloseFired, true, "req 'close' fires on body consumption");
  // The response 'close' is the one that means "finished or disconnected".
  assert.equal(resCloseFired, true, "res 'close' is the correct disconnect hook");
});

test("BUG 2: the streaming route aborts on res close, not on req close", async () => {
  // Boots the real pattern: an SSE route whose upstream fetch is aborted
  // when the RESPONSE closes. If the handler were wired to `req`, the abort
  // would fire before the stream ever started.
  const app = express();
  app.use(express.json());
  let abortedAt = null;
  const startedAt = Date.now();

  app.post("/stream", async (req, res) => {
    const ac = new AbortController();
    res.on("close", () => { if (!res.writableEnded) ac.abort(); });
    // A stand-in for the upstream provider fetch.
    const upstream = new Promise((_resolve, reject) => {
      const t = setTimeout(() => reject(new Error("aborted")), 2000);
      ac.signal.addEventListener("abort", () => { clearTimeout(t); reject(new Error("aborted")); });
    });
    res.setHeader("Content-Type", "text/event-stream");
    res.write("data: first\n\n");
    try {
      await upstream;
    } catch (e) {
      abortedAt = Date.now() - startedAt;
      try { res.write(`data: aborted after ${abortedAt}ms\n\n`); } catch { /* gone */ }
    }
    res.end();
  });

  const server = app.listen(0);
  await new Promise((r) => server.once("listening", r));
  const port = server.address().port;

  const r = await fetch(`http://127.0.0.1:${port}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal: "hi" }),
  });
  const text = await r.text();
  server.close();

  // The stream produced its first frame (so the request 'close' did NOT
  // abort the upstream prematurely), and the abort came from res 'close'.
  assert.match(text, /first/);
  assert.match(text, /aborted after \d+ms/);
  assert.ok(abortedAt !== null, "the upstream must have been aborted on res close");
});

// ---------------------------------------------------------------------------
// BUG 2 (cont): pin the ACTUAL wiring in src/index.js.
//
// The behaviour test above proves the Express semantics, but it exercises a
// locally-defined app — it would NOT catch someone re-wiring the real route
// to `req`. These guards read the real route source and fail if the
// anti-pattern comes back. They are deliberately narrow: they assert the two
// specific mistakes, not a general style.
// ---------------------------------------------------------------------------

test("BUG 2 (guard): the real streaming route aborts on res 'close', never req 'close'", () => {
  const src = stripComments(readIndex());
  // Find the streaming route block.
  const start = src.indexOf('app.post("/agent/chat/stream"');
  assert.notEqual(start, -1, "the streaming route must exist");
  const end = src.indexOf("\napp.", start + 10);
  const block = src.slice(start, end > -1 ? end : start + 8000);

  assert.match(block, /res\.on\("close",\s*onClose\)/, "the route must listen on res 'close'");
  assert.doesNotMatch(block, /req\.on\("close"/, "the route must NOT listen on req 'close' (fires on body consumption)");
  assert.doesNotMatch(block, /req\.on\("aborted"/, "the route must NOT listen on req 'aborted'");
});

test("BUG 2 (guard): the abort handler checks res.writableEnded before aborting", () => {
  const src = stripComments(readIndex());
  const start = src.indexOf('app.post("/agent/chat/stream"');
  const block = src.slice(start, start + 8000);
  // Aborting on a normally-finished response would kill the upstream mid-write.
  assert.match(block, /writableEnded/, "the close handler must check res.writableEnded");
});

test("BUG 1 (guard): the provider reader supports BOTH web and Node stream bodies", () => {
  const src = stripComments(readFileSync(fileURLToPath(new URL("../src/streamProvider.js", import.meta.url)), "utf8"));
  assert.match(src, /typeof body\.getReader === "function"/, "must handle a web ReadableStream");
  assert.match(src, /typeof body\.on === "function"/, "must handle a Node.js stream (node-fetch v3)");
  // The old bug: a fallback that buffers the whole body.
  assert.doesNotMatch(src, /await res\.json\(\)/, "must not fall back to buffering the whole body");
});

// ---------------------------------------------------------------------------
// F-50 guard: the shared pre-trade gate must forward the dollar-risk override.
//
// This is a WIRING bug in index.js, which is a 9,000-line module with side
// effects on import and so cannot be unit tested directly. The bug: every
// caller computed the real dollar risk of the order and passed it in as
// `riskUsdOverride`, and `riskGate()` did not destructure it — so it was
// silently dropped and the per-trade cap was priced against a default 0.01
// lot instead of the actual position size.
//
// A source guard is the only thing that can pin this, so that is what it is.
// ---------------------------------------------------------------------------

test("F-50 (guard): riskGate forwards riskUsdOverride to the risk layer", () => {
  const src = readFileSync(fileURLToPath(new URL("../src/index.js", import.meta.url)), "utf8");
  const m = src.match(/function riskGate\(\{([^}]*)\}\)\s*\{([\s\S]{0,400}?)\n\}/);
  assert.ok(m, "riskGate() must exist in index.js");
  const [, params, body] = m;
  assert.match(params, /riskUsdOverride/, "riskGate must accept riskUsdOverride");
  assert.match(body, /riskUsdOverride/, "riskGate must pass riskUsdOverride on to portfolioRisk.check()");
});

