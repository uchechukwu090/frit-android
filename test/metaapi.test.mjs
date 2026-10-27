// ============================================================================
// FRIT METAAPI RAIL — unit tests for the pure parts (no network)
// ----------------------------------------------------------------------------
// The lot math is the part that decides real money on FBS: a volume that is not
// an exact multiple of the broker's volumeStep is REJECTED by MT5, and a volume
// below minVolume would be refused too. These tests pin that behaviour, plus
// the symbol-candidate ladder and the action mapping.
// ============================================================================

import test from "node:test";
import assert from "node:assert";
import { metaApiConfig, normalizeVolume, symbolCandidates, orderActionType, MetaApiRail } from "../src/strategy/metaapi.js";

test("metaapi config: ready only with both token and account id", () => {
  const none = metaApiConfig({});
  assert.equal(none.ready, false);
  assert.equal(none.region, "new-york");
  assert.equal(none.base, "https://mt-client-api-v1.new-york.agiliumtrade.ai");

  const partial = metaApiConfig({ METAAPI_TOKEN: "t" });
  assert.equal(partial.ready, false);

  const full = metaApiConfig({ METAAPI_TOKEN: "t", METAAPI_ACCOUNT_ID: "acc", METAAPI_REGION: "london" });
  assert.equal(full.ready, true);
  assert.equal(full.base, "https://mt-client-api-v1.london.agiliumtrade.ai");

  const override = metaApiConfig({ METAAPI_TOKEN: "t", METAAPI_ACCOUNT_ID: "acc", METAAPI_BASE_URL: "https://example.test/" });
  assert.equal(override.base, "https://example.test");
});

test("metaapi volume: snaps to the broker step, never below min or above max", () => {
  // 0.0137 Kelly lot on a 0.01-step symbol is refused by MT5 if sent raw.
  assert.equal(normalizeVolume(0.0137, { volumeStep: 0.01, minVolume: 0.01 }), 0.01);
  assert.equal(normalizeVolume(0.017, { volumeStep: 0.01 }), 0.01); // floor, never over-risk
  assert.equal(normalizeVolume(0.02, { volumeStep: 0.01 }), 0.02);
  // 0.08 lot on a symbol that only trades 0.1 steps must become 0.1 (min).
  assert.equal(normalizeVolume(0.08, { volumeStep: 0.1, minVolume: 0.1 }), 0.1);
  assert.equal(normalizeVolume(1.25, { volumeStep: 1, minVolume: 1, maxVolume: 10 }), 1);
  // Broker cap wins over a huge Kelly lot.
  assert.equal(normalizeVolume(999, { volumeStep: 0.01, minVolume: 0.01, maxVolume: 5 }), 5);
  // No spec at all -> retail default step 0.01.
  assert.equal(normalizeVolume(0.057), 0.05);
  // Garbage / zero is "no trade", not 0.01.
  assert.equal(normalizeVolume(0), 0);
  assert.equal(normalizeVolume("abc"), 0);
  assert.equal(normalizeVolume(-1), 0);
});

test("metaapi symbols: candidate ladder tries the broker's own spelling", () => {
  assert.deepEqual(symbolCandidates(""), []);
  const plain = symbolCandidates("eurusd");
  assert.equal(plain[0], "EURUSD");
  assert.ok(plain.includes("EURUSDm"));
  const suffixed = symbolCandidates("XAUUSD", "m");
  assert.equal(suffixed[0], "XAUUSD");
  assert.equal(suffixed[1], "XAUUSDm");
  // No duplicates when the configured suffix is already in the ladder.
  assert.equal(new Set(suffixed).size, suffixed.length);
});

test("metaapi actions: sell/PUT map to ORDER_TYPE_SELL, everything else buys", () => {
  assert.equal(orderActionType("sell"), "ORDER_TYPE_SELL");
  assert.equal(orderActionType("PUT"), "ORDER_TYPE_SELL");
  assert.equal(orderActionType("buy"), "ORDER_TYPE_BUY");
  assert.equal(orderActionType("CALL"), "ORDER_TYPE_BUY");
  assert.equal(orderActionType(null), "ORDER_TYPE_BUY");
});

test("metaapi rail: unconfigured rail refuses to look live", async () => {
  const rail = new MetaApiRail({ ready: false, base: "https://x.test", token: "", accountId: "" });
  assert.equal(rail.ready, false);
  const r = await rail.placeMarketOrder({ symbol: "EURUSD", action: "buy", lotSize: 0.1 });
  assert.equal(r.mode, "paper");
  assert.equal(r.status, "simulated");
});

test("metaapi rail: places a market order with SL/TP and the resolved symbol", async () => {
  const calls = [];
  const fakeFetch = async (url, init) => {
    calls.push({ url, init });
    const body = init.body ? JSON.parse(init.body) : null;
    if (url.endsWith("/specification")) {
      // FBS-style broker: only the suffixed spelling exists.
      if (!url.includes("EURUSDm")) {
        return { ok: false, status: 404, text: async () => JSON.stringify({ message: "not found" }) };
      }
      return { ok: true, status: 200, text: async () => JSON.stringify({ symbol: "EURUSDm", volumeStep: 0.01, minVolume: 0.01, maxVolume: 100 }) };
    }
    if (url.endsWith("/trade")) {
      return { ok: true, status: 200, text: async () => JSON.stringify({ orderId: "42", positionId: "p1", stringCode: "ERR_NO_ERROR", message: null }) };
    }
    void body;
    return { ok: false, status: 404, text: async () => JSON.stringify({ message: "not found" }) };
  };
  const rail = new MetaApiRail(
    metaApiConfig({ METAAPI_TOKEN: "t", METAAPI_ACCOUNT_ID: "acc" }),
    { fetchImpl: fakeFetch, log: () => {} },
  );
  const r = await rail.placeMarketOrder({
    symbol: "EURUSD", action: "buy", lotSize: 0.017, sl: 1.08, tp: 1.1, reason: "scalping bot",
  });
  assert.equal(r.mode, "live");
  assert.equal(r.status, "placed");
  assert.equal(r.brokerSymbol, "EURUSDm");
  assert.equal(r.lotSize, 0.01);
  assert.equal(r.orderId, "42");

  const trade = calls.find((c) => c.url.endsWith("/trade"));
  assert.ok(trade, "a /trade call must be made");
  const sent = JSON.parse(trade.init.body);
  assert.equal(sent.actionType, "ORDER_TYPE_BUY");
  assert.equal(sent.symbol, "EURUSDm");
  assert.equal(sent.volume, 0.01);
  assert.equal(sent.stopLoss, 1.08);
  assert.equal(sent.takeProfit, 1.1);
  assert.ok(sent.clientId, "clientId makes a retry idempotent");
  assert.equal(trade.init.headers["auth-token"], "t");
});

test("metaapi rail: an unlisted symbol degrades to paper, never to a wrong market", async () => {
  const fakeFetch = async () => ({ ok: false, status: 404, text: async () => JSON.stringify({ message: "not found" }) });
  const rail = new MetaApiRail(
    metaApiConfig({ METAAPI_TOKEN: "t", METAAPI_ACCOUNT_ID: "acc" }),
    { fetchImpl: fakeFetch, log: () => {} },
  );
  const r = await rail.placeMarketOrder({ symbol: "NOTREAL", action: "sell", lotSize: 0.1 });
  assert.equal(r.mode, "paper");
  assert.match(r.message, /does not list/);
});

test("metaapi rail: a broker refusal is reported, not swallowed", async () => {
  const fakeFetch = async (url) => {
    if (url.endsWith("/specification")) {
      return { ok: true, status: 200, text: async () => JSON.stringify({ symbol: "EURUSD", volumeStep: 0.01, minVolume: 0.01 }) };
    }
    return { ok: true, status: 200, text: async () => JSON.stringify({ numericCode: 10019, stringCode: "ERR_NO_MONEY", message: "Not enough money" }) };
  };
  const rail = new MetaApiRail(
    metaApiConfig({ METAAPI_TOKEN: "t", METAAPI_ACCOUNT_ID: "acc" }),
    { fetchImpl: fakeFetch, log: () => {} },
  );
  const r = await rail.placeMarketOrder({ symbol: "EURUSD", action: "buy", lotSize: 0.1 });
  assert.equal(r.status, "refused");
  assert.equal(r.code, "ERR_NO_MONEY");
  assert.equal(r.orderId, undefined);
});
