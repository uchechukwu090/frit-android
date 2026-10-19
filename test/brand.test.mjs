// Brand Kit (server-side lock) unit tests — in-memory SQLite, no network.
// Covers audit F-13 ("Brand Kit injection happens only on the client").
import test from "node:test";
import assert from "node:assert/strict";
import Database from "better-sqlite3";
import { normalizeBrand, brandPromptFragment, applyBrandToPrompt, resolveBrandKit } from "../src/creative/brand.js";

function setup() {
  const db = new Database(":memory:");
  db.exec(`
    CREATE TABLE creative_brandkits (
      id TEXT PRIMARY KEY, user_id TEXT, name TEXT, colors TEXT, font TEXT,
      logo_url TEXT, palette TEXT, tone TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
  `);
  return db;
}

const KIT = {
  id: "brand_1",
  user_id: "u1",
  name: "Sunset Co",
  colors: JSON.stringify(["#EBE8E0", "#362F2F"]),
  font: "Editorial Serif",
  logo_url: "",
  palette: "warm",
  tone: "premium",
};

test("normalizeBrand parses the stored JSON colors", () => {
  const brand = normalizeBrand(KIT);
  assert.equal(brand.name, "Sunset Co");
  assert.deepEqual(brand.colors, ["#EBE8E0", "#362F2F"]);
  assert.equal(brand.font, "Editorial Serif");
  assert.equal(brand.palette, "warm");
  assert.equal(brand.tone, "premium");
});

test("normalizeBrand drops malformed colors instead of trusting them", () => {
  const brand = normalizeBrand({ ...KIT, colors: JSON.stringify(["#EBE8E0", "not-a-color", "#GGG", null, 42]) });
  assert.deepEqual(brand.colors, ["#EBE8E0"]);
});

test("normalizeBrand survives a broken JSON colors column", () => {
  const brand = normalizeBrand({ ...KIT, colors: "{not json" });
  assert.deepEqual(brand.colors, []);
});

test("normalizeBrand returns null for an empty kit", () => {
  assert.equal(normalizeBrand(null), null);
  assert.equal(normalizeBrand({ name: "", colors: "[]", font: "", palette: "", tone: "", logo_url: "" }), null);
});

test("brandPromptFragment states the exact hexes, type voice and mood", () => {
  const frag = brandPromptFragment(normalizeBrand(KIT));
  assert.match(frag, /Sunset Co/);
  assert.match(frag, /#EBE8E0/);
  assert.match(frag, /#362F2F/);
  assert.match(frag, /Editorial Serif/);
  assert.match(frag, /warm/);
  assert.match(frag, /premium/);
});

test("brandPromptFragment is empty when there is no brand", () => {
  assert.equal(brandPromptFragment(null), "");
});

test("applyBrandToPrompt appends the lock and leaves a brandless prompt alone", () => {
  const withBrand = applyBrandToPrompt("a red bicycle", normalizeBrand(KIT));
  assert.match(withBrand, /^a red bicycle\./);
  assert.match(withBrand, /Brand lock/);
  assert.equal(applyBrandToPrompt("a red bicycle", null), "a red bicycle");
});

test("resolveBrandKit returns the user's own kit", () => {
  const db = setup();
  db.prepare("INSERT INTO creative_brandkits (id, user_id, name, colors, font, logo_url, palette, tone) VALUES (?,?,?,?,?,?,?,?)")
    .run(KIT.id, KIT.user_id, KIT.name, KIT.colors, KIT.font, KIT.logo_url, KIT.palette, KIT.tone);
  const brand = resolveBrandKit(db, "u1", "brand_1");
  assert.equal(brand.name, "Sunset Co");
});

test("resolveBrandKit refuses another user's kit", () => {
  const db = setup();
  db.prepare("INSERT INTO creative_brandkits (id, user_id, name, colors, font, logo_url, palette, tone) VALUES (?,?,?,?,?,?,?,?)")
    .run(KIT.id, KIT.user_id, KIT.name, KIT.colors, KIT.font, KIT.logo_url, KIT.palette, KIT.tone);
  assert.equal(resolveBrandKit(db, "u2", "brand_1"), null);
});

test("resolveBrandKit ignores a missing id and a missing table", () => {
  const db = setup();
  assert.equal(resolveBrandKit(db, "u1", ""), null);
  assert.equal(resolveBrandKit(db, "u1", null), null);
  const empty = new Database(":memory:");
  assert.equal(resolveBrandKit(empty, "u1", "brand_1"), null);
});
