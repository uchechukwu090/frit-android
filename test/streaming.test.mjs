// Streaming (SSE) unit tests — pure parsing helpers, no network.
// Covers the audit item "streaming responses for AI chat".
import test from "node:test";
import assert from "node:assert/strict";
import { parseSseChunk, extractDelta, buildSseEvent, isDoneEvent, classifyStreamRoute, stripConversationPrefix, ROUTE_TASK, ROUTE_CONVERSATION, ROUTE_UNKNOWN } from "../src/streaming.js";

test("parses a single complete event", () => {
  const { events, rest } = parseSseChunk('data: {"text":"Hi"}\n\n');
  assert.equal(events.length, 1);
  assert.equal(events[0].data, '{"text":"Hi"}');
  assert.equal(rest, "");
});

test("parses several events from one buffer", () => {
  const { events } = parseSseChunk('data: one\n\ndata: two\n\n');
  assert.equal(events.length, 2);
  assert.equal(events[0].data, "one");
  assert.equal(events[1].data, "two");
});

test("keeps an incomplete event as the remainder", () => {
  const { events, rest } = parseSseChunk('data: {"text":"Hel');
  assert.equal(events.length, 0);
  assert.equal(rest, 'data: {"text":"Hel');
});

test("reassembles an event split across two reads", () => {
  const first = parseSseChunk('data: {"text":"Hel');
  const second = parseSseChunk(`${first.rest}lo"}\n\n`);
  assert.equal(second.events.length, 1);
  assert.equal(second.events[0].data, '{"text":"Hello"}');
});

test("ignores comments and keep-alives", () => {
  const { events } = parseSseChunk(": keep-alive\n\ndata: real\n\n");
  assert.equal(events.length, 1);
  assert.equal(events[0].data, "real");
});

test("handles CRLF line endings", () => {
  const { events } = parseSseChunk('event: delta\r\ndata: {"text":"x"}\r\n\r\n');
  assert.equal(events.length, 1);
  assert.equal(events[0].event, "delta");
  assert.equal(events[0].data, '{"text":"x"}');
});

test("reads the event name", () => {
  const { events } = parseSseChunk('event: done\ndata: {}\n\n');
  assert.equal(events[0].event, "done");
});

test("joins multi-line data fields", () => {
  const { events } = parseSseChunk("data: line1\ndata: line2\n\n");
  assert.equal(events[0].data, "line1\nline2");
});

test("extracts a content delta", () => {
  const text = extractDelta('{"choices":[{"delta":{"content":"Hello"}}]}');
  assert.equal(text, "Hello");
});

test("extracts a delta with role but no content as null", () => {
  assert.equal(extractDelta('{"choices":[{"delta":{"role":"assistant"}}]}'), null);
});

test("extractDelta ignores non-JSON payloads", () => {
  assert.equal(extractDelta("not json"), null);
  assert.equal(extractDelta(""), null);
  assert.equal(extractDelta("[DONE]"), null);
});

test("extractDelta falls back to a non-streaming message shape", () => {
  assert.equal(extractDelta('{"choices":[{"message":{"content":"whole"}}]}'), "whole");
});

test("extractDelta joins array content parts", () => {
  assert.equal(extractDelta('{"choices":[{"delta":{"content":[{"text":"a"},{"text":"b"}]}}]}'), "ab");
});

test("buildSseEvent formats a frame with the blank-line terminator", () => {
  const frame = buildSseEvent({ text: "hi" }, "delta");
  assert.equal(frame, 'event: delta\ndata: {"text":"hi"}\n\n');
});

test("buildSseEvent splits a payload containing newlines", () => {
  const frame = buildSseEvent("a\nb", "delta");
  assert.ok(frame.includes("data: a\ndata: b\n\n"));
});

test("buildSseEvent defaults to the message event", () => {
  assert.ok(buildSseEvent("x").startsWith("event: message\n"));
});

test("isDoneEvent recognises the terminator", () => {
  assert.equal(isDoneEvent("[DONE]"), true);
  assert.equal(isDoneEvent(" [DONE] "), true);
  assert.equal(isDoneEvent('{"text":"x"}'), false);
});

// --- streamed chat/task routing ---

test("classifyStreamRoute identifies a task once TASK is spelled out", () => {
  assert.equal(classifyStreamRoute("TASK"), ROUTE_TASK);
  assert.equal(classifyStreamRoute("task"), ROUTE_TASK);
  assert.equal(classifyStreamRoute("TASK\n"), ROUTE_TASK);
});

test("classifyStreamRoute waits for the full TASK word", () => {
  // Conservative on purpose: a partial "TAS" is not yet a decision, so the
  // caller keeps buffering rather than committing to the agent loop.
  assert.equal(classifyStreamRoute("T"), ROUTE_UNKNOWN);
  assert.equal(classifyStreamRoute("TA"), ROUTE_UNKNOWN);
  assert.equal(classifyStreamRoute("TAS"), ROUTE_UNKNOWN);
});

test("classifyStreamRoute identifies a conversation once CONVERSATION is complete", () => {
  assert.equal(classifyStreamRoute("CONVERSATION"), ROUTE_CONVERSATION);
  assert.equal(classifyStreamRoute("CONVERSATION Hello there"), ROUTE_CONVERSATION);
  assert.equal(classifyStreamRoute("  conversation hi"), ROUTE_CONVERSATION);
});

test("classifyStreamRoute stays unknown while the prefix is still ambiguous", () => {
  assert.equal(classifyStreamRoute(""), ROUTE_UNKNOWN);
  assert.equal(classifyStreamRoute("C"), ROUTE_UNKNOWN);
  assert.equal(classifyStreamRoute("CONVERSA"), ROUTE_UNKNOWN);
});

test("classifyStreamRoute never mistakes a conversation for a task", () => {
  // "CONVERSATION" starts with C, "TASK" with T — they share no prefix.
  assert.equal(classifyStreamRoute("C"), ROUTE_UNKNOWN);
  assert.equal(classifyStreamRoute("Co"), ROUTE_UNKNOWN);
  assert.equal(classifyStreamRoute("CONVERSATION about tasks"), ROUTE_CONVERSATION);
});

test("stripConversationPrefix removes the router prefix", () => {
  assert.equal(stripConversationPrefix("CONVERSATION Hello there"), "Hello there");
  assert.equal(stripConversationPrefix("conversation  Hello there"), "Hello there");
  assert.equal(stripConversationPrefix("Hello there"), "Hello there");
});
