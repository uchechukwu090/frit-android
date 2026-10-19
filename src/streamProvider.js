// ============================================================================
// FRIT STREAMING — provider stream → SSE events (audit: streaming AI chat)
// ----------------------------------------------------------------------------
// Turns a provider's OpenAI-compatible `stream: true` response into the
// Server-Sent Events our clients consume.
//
// Why a module and not a blob inside index.js: the interesting part (partial
// chunks, split events, done markers, provider quirks) is exactly what breaks
// in production, and it is testable in isolation.
//
// The provider contract we rely on:
//   data: {"choices":[{"delta":{"content":"..."}}]}   -> a text delta
//   data: [DONE]                                       -> end of stream
// ============================================================================

import { parseSseChunk, extractDelta, buildSseEvent, isDoneEvent } from "./streaming.js";

/**
 * Read a provider streaming response and yield SSE-ready events.
 *
 * Yields objects of the shape:
 *   { type: "delta", text: "..." }   a chunk of assistant text
 *   { type: "done" }                 the provider finished
 *   { type: "error", message: "..." } something went wrong mid-stream
 *
 * IMPORTANT (why this exists): the server imports `node-fetch` v3, whose
 * `res.body` is a Node.js PassThrough stream — NOT a web ReadableStream — so
 * `res.body.getReader()` does not exist. Reading it that way threw, and the
 * naive `await res.json()` fallback buffered the ENTIRE response, which
 * silently defeated the whole point of streaming. This helper therefore
 * supports BOTH shapes: a web ReadableStream (global fetch/undici) and a
 * Node.js stream (node-fetch).
 *
 * @param {Response} res the fetch response (must be ok and streaming)
 * @param {AbortSignal} [signal] aborts the upstream request
 */
export async function* readProviderStream(res, signal) {
  const body = res?.body;
  if (!body) {
    yield { type: "error", message: "provider returned no body" };
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  const handleText = function* (text) {
    buffer += text;
    const { events, rest } = parseSseChunk(buffer);
    buffer = rest;
    for (const ev of events) {
      if (ev.event && ev.event !== "message" && ev.event !== "data") continue;
      if (isDoneEvent(ev.data)) {
        yield { type: "done" };
        return;
      }
      const d = extractDelta(ev.data);
      if (d) yield { type: "delta", text: d };
    }
  };

  try {
    if (typeof body.getReader === "function") {
      // Web ReadableStream (global fetch / undici).
      const reader = body.getReader();
      try {
        for (;;) {
          if (signal?.aborted) {
            yield { type: "error", message: "aborted" };
            return;
          }
          const { value, done } = await reader.read();
          if (done) break;
          yield* handleText(decoder.decode(value, { stream: true }));
        }
      } finally {
        try { reader.cancel(); } catch { /* already closed */ }
      }
    } else if (typeof body.on === "function") {
      // Node.js stream (node-fetch v3 PassThrough). Read it incrementally so
      // the tokens reach the client as they arrive.
      const queue = [];
      let notify = null;
      let ended = false;
      let failure = null;

      // Wake the consumer exactly once per arrival. The pending promise is
      // cleared BEFORE it is resolved, so a second `push` cannot resolve an
      // already-settled promise (and cannot busy-loop the reader).
      const wake = () => {
        const n = notify;
        notify = null;
        if (n) n();
      };
      const push = (item) => {
        queue.push(item);
        wake();
      };

      body.on("data", (chunk) => push({ kind: "data", chunk }));
      body.on("end", () => { ended = true; push({ kind: "end" }); });
      body.on("error", (err) => { failure = err; ended = true; push({ kind: "end" }); });

      for (;;) {
        if (signal?.aborted) {
          yield { type: "error", message: "aborted" };
          try { body.destroy?.(); } catch { /* ignore */ }
          return;
        }
        if (!queue.length) {
          if (ended) break;
          // Park until the next chunk arrives. The promise is resolved by
          // `push`, not by a self-call (that would spin).
          await new Promise((resolve) => { notify = resolve; });
          continue;
        }
        const item = queue.shift();
        if (item.kind === "end") {
          if (failure) yield { type: "error", message: failure.message || String(failure) };
          break;
        }
        yield* handleText(decoder.decode(item.chunk, { stream: true }));
      }
    } else {
      yield { type: "error", message: "provider returned an unreadable body" };
      return;
    }

    // A final flush: some providers end without a trailing blank line.
    const tail = parseSseChunk(`${buffer}\n\n`);
    for (const ev of tail.events) {
      if (isDoneEvent(ev.data)) {
        yield { type: "done" };
        return;
      }
      const d = extractDelta(ev.data);
      if (d) yield { type: "delta", text: d };
    }
  } catch (err) {
    yield { type: "error", message: err?.message || String(err) };
  }
}

/**
 * Build the SSE frames for a finished (non-streamed) reply, so a client that
 * connects to the stream endpoint always gets the same event vocabulary.
 */
export function sseDone(text, extra = {}) {
  return buildSseEvent({ ...extra, text: String(text || "") }, "done");
}

/**
 * Build an SSE error frame.
 */
export function sseError(message, extra = {}) {
  return buildSseEvent({ ...extra, error: String(message || "stream failed") }, "error");
}

/**
 * Build an SSE delta frame.
 */
export function sseDelta(text) {
  return buildSseEvent({ text: String(text || "") }, "delta");
}
