// ============================================================================
// Reference transport tests — how the agent actually SEES a reference
// ----------------------------------------------------------------------------
// The vision model never gets a Pinterest URL: it gets bytes we fetched
// ourselves (browser-like headers, <=1024px, content-addressed cache id).
// A CDN that refuses us falls back to an origin-side public proxy.
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";

import {
  refId, proxyImageUrl, fetchImageBuffer, shrinkForVision, loadReference,
} from "../src/creative/refs.js";

test("refId: content-addressed, stable and unguessable", () => {
  const a = refId("https://i.pinimg.com/736x/a.jpg");
  assert.equal(a, refId("https://i.pinimg.com/736x/a.jpg"));
  assert.notEqual(a, refId("https://i.pinimg.com/736x/b.jpg"));
  assert.match(a, /^[a-f0-9]{32}$/);
});

test("proxyImageUrl: builds an origin-side proxy URL for a hotlink-protected host", () => {
  const u = proxyImageUrl("https://i.pinimg.com/736x/a.jpg");
  assert.ok(u.startsWith("https://images.weserv.nl/?url="));
  assert.ok(u.includes("i.pinimg.com"));
  assert.equal(proxyImageUrl("not a url"), null);
});

test("fetchImageBuffer: a non-image response is refused (no bytes for the model)", async () => {
  const got = await fetchImageBuffer("https://example.com/page.html", {
    fetchImpl: async () => ({
      ok: true,
      headers: { get: () => "text/html" },
      arrayBuffer: async () => new ArrayBuffer(8),
    }),
  });
  assert.equal(got, null);
});

test("fetchImageBuffer: a failed fetch is a soft null, never a throw", async () => {
  const got = await fetchImageBuffer("https://example.com/a.jpg", {
    fetchImpl: async () => { throw new Error("network down"); },
  });
  assert.equal(got, null);
  const notOk = await fetchImageBuffer("https://example.com/a.jpg", {
    fetchImpl: async () => ({ ok: false, status: 403 }),
  });
  assert.equal(notOk, null);
});

test("fetchImageBuffer: an image response carries the bytes and the mime", async () => {
  const got = await fetchImageBuffer("https://i.pinimg.com/736x/a.jpg", {
    fetchImpl: async () => ({
      ok: true,
      headers: { get: () => "image/jpeg" },
      arrayBuffer: async () => new Uint8Array([1, 2, 3, 4]).buffer,
    }),
  });
  assert.ok(got);
  assert.equal(got.mime, "image/jpeg");
  assert.equal(got.buf.length, 4);
});

test("shrinkForVision: without sharp the original bytes survive", async () => {
  const buf = Buffer.from([1, 2, 3]);
  const out = await shrinkForVision(buf);
  assert.ok(out.buf.length >= 3);
});

test("loadReference: the origin fetch wins and the result is inline + cached", async () => {
  const seen = [];
  const out = await loadReference("https://i.pinimg.com/736x/a.jpg", {
    fetchImpl: async (url) => {
      seen.push(url);
      return {
        ok: true,
        headers: { get: () => "image/jpeg" },
        arrayBuffer: async () => new Uint8Array([1, 2, 3, 4, 5]).buffer,
      };
    },
  });
  assert.ok(out, "a reference must load");
  assert.ok(out.id, "it must be cached under a content id");
  assert.ok(out.dataUri.startsWith("data:image/"), "the vision model reads inline bytes");
  assert.equal(seen.length, 1, "the origin is fetched once");
  assert.equal(seen[0], "https://i.pinimg.com/736x/a.jpg");
});

test("loadReference: a blocked origin falls back to the public proxy", async () => {
  const seen = [];
  const out = await loadReference("https://i.pinimg.com/736x/a.jpg", {
    fetchImpl: async (url) => {
      seen.push(url);
      if (!url.includes("weserv.nl")) return { ok: false, status: 403 };
      return {
        ok: true,
        headers: { get: () => "image/jpeg" },
        arrayBuffer: async () => new Uint8Array([9, 9, 9]).buffer,
      };
    },
  });
  assert.ok(out, "the proxy must rescue the reference");
  assert.equal(seen.length, 2);
  assert.ok(seen[1].includes("weserv.nl"));
});

test("loadReference: when everything fails it returns null, never a broken image", async () => {
  const out = await loadReference("https://i.pinimg.com/736x/a.jpg", {
    fetchImpl: async () => ({ ok: false, status: 403 }),
  });
  assert.equal(out, null);
});
