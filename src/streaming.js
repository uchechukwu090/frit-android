// ============================================================================
// FRIT STREAMING — SSE helpers (audit: "streaming responses for AI chat")
// ----------------------------------------------------------------------------
// The AI chat used to be one blocking request: the user sent a message and
// stared at a spinner until the whole reply had been generated, downloaded
// and parsed. On a phone that is several seconds of silence for a one-line
// answer, and it feels broken.
//
// These helpers are PURE (no network, no Express) so they are unit tested in
// test/streaming.test.mjs:
//   - parseSseChunk  -> turn a raw SSE buffer into { events, rest }
//   - extractDelta   -> pull the text delta out of one provider chunk
//   - buildSseEvent  -> format one event for the wire
// ============================================================================

/**
 * Parse a Server-Sent Events buffer.
 *
 * Providers stream in whatever chunks the socket hands us, so a single read
 * can contain half an event, several whole events, or an event split across
 * two reads. This returns the complete events it could parse plus the
 * unconsumed remainder, which the caller feeds back in on the next read.
 *
 * @param {string} buffer raw text received so far
 * @returns {{events:Array<{event:string,data:string}>, rest:string}}
 */
export function parseSseChunk(buffer) {
  const text = String(buffer || "");
  const events = [];
  // Events are separated by a blank line. Normalise CRLF first so a
  // Windows-ish provider does not produce phantom "\r" in the data.
  const normalised = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const parts = normalised.split("\n\n");
  // The last part is either empty (the buffer ended on a boundary) or an
  // incomplete event that must wait for more bytes.
  const rest = parts.pop() ?? "";
  for (const part of parts) {
    const parsed = parseEventBlock(part);
    if (parsed) events.push(parsed);
  }
  return { events, rest };
}

/** Parse one SSE block ("event: x\ndata: {...}") into {event, data}. */
function parseEventBlock(block) {
  const lines = String(block || "").split("\n").filter((l) => l.length > 0);
  if (!lines.length) return null;
  let event = "message";
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith(":")) continue; // comment / keep-alive
    const idx = line.indexOf(":");
    if (idx === -1) continue;
    const field = line.slice(0, idx).trim();
    let value = line.slice(idx + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (!dataLines.length) return null;
  return { event, data: dataLines.join("\n") };
}

/**
 * Pull the text delta out of one provider chunk.
 *
 * OpenAI-compatible providers (OpenRouter, Mistral, Groq, DeepInfra) all use
 * the same shape:
 *   data: {"choices":[{"delta":{"content":"He"}}]}
 * and finish with:
 *   data: [DONE]
 *
 * Returns null for anything that carries no text (role-only deltas, usage
 * blocks, tool-call deltas, keep-alives) so the caller can skip it.
 *
 * @param {string} data the raw `data:` payload
 * @returns {string|null}
 */
export function extractDelta(data) {
  const raw = String(data || "").trim();
  if (!raw || raw === "[DONE]") return null;
  let obj;
  try {
    obj = JSON.parse(raw);
  } catch {
    return null;
  }
  const choice = obj?.choices?.[0];
  if (!choice) return null;
  // Non-streaming shapes occasionally leak through.
  const content = choice.delta?.content ?? choice.message?.content ?? choice.text;
  if (typeof content === "string" && content.length) return content;
  // Some providers emit an array of content parts.
  if (Array.isArray(content)) {
    const joined = content
      .map((p) => (typeof p === "string" ? p : p?.text || ""))
      .join("");
    return joined.length ? joined : null;
  }
  return null;
}

/**
 * Format one SSE event for the wire.
 * @param {string} data
 * @param {string} [event] defaults to "message"
 */
export function buildSseEvent(data, event = "message") {
  const payload = typeof data === "string" ? data : JSON.stringify(data);
  // A payload containing newlines must be split across multiple `data:` lines
  // per the SSE spec, otherwise the frame is invalid.
  const lines = String(payload).split("\n").map((l) => `data: ${l}`).join("\n");
  return `event: ${event}\n${lines}\n\n`;
}

/**
 * True when the event signals the end of the provider stream.
 */
export function isDoneEvent(data) {
  return String(data || "").trim() === "[DONE]";
}

// ============================================================================
// STREAMED CHAT/TASK ROUTING
// ============================================================================
// The chat router answers "CONVERSATION <reply>" or "TASK". When the router
// runs as a STREAM, the classification arrives in the first few characters —
// so we can decide while the text is still arriving, and stream the reply
// without a second round trip.
//
// "TASK" and "CONVERSATION" share no prefix, so a short buffer is enough to
// classify. Until then the answer is UNKNOWN.

export const ROUTE_UNKNOWN = "unknown";
export const ROUTE_TASK = "task";
export const ROUTE_CONVERSATION = "conversation";

/** How many characters we need before the routing decision is unambiguous. */
export const ROUTE_DECISION_CHARS = 12;

/**
 * Classify a streamed router reply.
 * @param {string} buffer the text received so far
 * @returns {"unknown"|"task"|"conversation"}
 */
export function classifyStreamRoute(buffer) {
  const t = String(buffer || "").trimStart().toUpperCase();
  if (!t) return ROUTE_UNKNOWN;
  if (t.startsWith("TASK")) return ROUTE_TASK;
  // "CONVERSATION" is 12 characters; anything shorter could still become
  // "TASK", so we wait for more.
  if (t.length >= ROUTE_DECISION_CHARS && t.startsWith("CONVERSATION")) return ROUTE_CONVERSATION;
  return ROUTE_UNKNOWN;
}

/**
 * Strip the router's "CONVERSATION " prefix from a reply, returning the
 * actual message. If the prefix is not there yet, the whole buffer is
 * returned unchanged (the caller decides what to do with it).
 */
export function stripConversationPrefix(buffer) {
  const t = String(buffer || "");
  const m = t.match(/^\s*CONVERSATION\b\s*/i);
  return m ? t.slice(m[0].length) : t;
}
