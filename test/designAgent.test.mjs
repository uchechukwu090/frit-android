// ============================================================================
// Week 6 — design agent tests
// ----------------------------------------------------------------------------
// Hermetic: no network, no keys. The pipeline takes injected deps, so every
// stage is a fake here and the assertions are about BEHAVIOUR (what prompt got
// sent, whether a failed verification retried, what the client receives).
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";

import {
  asImageUrl,
  synthesizeBrief,
  verifyRender,
  planInspirationQueries,
  extractImageUrls,
  looksLikeUsableImage,
  rankInspirationUrls,
  normalizeHex,
  normalizeBrief,
  defaultBrief,
  briefToPrompt,
  judgeVerification,
  describeReference,
  runDesignAgent,
} from "../src/creative/designAgent.js";

// --------------------------------------------------------------- research ---

test("planInspirationQueries: site-filters the named inspiration hosts", () => {
  const qs = planInspirationQueries("cold brew coffee shop flyer", { sources: ["pinterest", "behance", "dribbble"], maxQueries: 3 });
  assert.equal(qs.length, 3);
  assert.match(qs[0], /^site:pinterest\.com /);
  assert.match(qs[1], /^site:behance\.com /);
  assert.match(qs[2], /^site:dribbble\.com /);
});

test("planInspirationQueries: empty goal yields no queries, and it always dedupes", () => {
  assert.deepEqual(planInspirationQueries("   "), []);
  const qs = planInspirationQueries("poster", { sources: ["pinterest", "pinterest"], maxQueries: 4 });
  assert.equal(new Set(qs).size, qs.length);
});

test("extractImageUrls: reads Bing murl (entity-encoded and raw), og:image, and absolutises", () => {
  const html = `
    <a m="{&quot;murl&quot;:&quot;https://i.pinimg.com/736x/aa/bb.jpg&quot;,&quot;turl&quot;:&quot;x&quot;}"></a>
    <script>{"murl":"https://mir-s3-cdn-cf.behance.net/project.jpg","turl":"y"}</script>
    <meta property="og:image" content="/fallback/page.png">
    <img src="https://example.com/thing.webp">
  `;
  const urls = extractImageUrls(html, "https://www.bing.com/images/search");
  assert.ok(urls.some((u) => u.includes("i.pinimg.com/736x/aa/bb.jpg")));
  assert.ok(urls.some((u) => u.includes("behance.net/project.jpg")));
  assert.ok(urls.includes("https://www.bing.com/fallback/page.png"));
  assert.ok(urls.some((u) => u.endsWith("thing.webp")));
});

test("looksLikeUsableImage: rejects icons/avatars/svg/data URIs, accepts design CDNs", () => {
  assert.equal(looksLikeUsableImage("https://i.pinimg.com/736x/aa.jpg"), true);
  assert.equal(looksLikeUsableImage("https://cdn.example.com/logo-icon.png"), false);
  assert.equal(looksLikeUsableImage("https://cdn.example.com/avatar/12.jpg"), false);
  assert.equal(looksLikeUsableImage("https://cdn.example.com/thing.svg"), false);
  assert.equal(looksLikeUsableImage("data:image/png;base64,AAA"), false);
  assert.equal(looksLikeUsableImage("not a url"), false);
});

test("rankInspirationUrls: dedupes query strings, prefers the inspiration hosts, caps", () => {
  const urls = [
    "https://random.example.com/a.jpg",
    "https://i.pinimg.com/736x/x.jpg?w=600",
    "https://i.pinimg.com/736x/x.jpg?w=1200",
    "https://mir-s3-cdn-cf.behance.net/p.jpg",
    "https://random.example.com/avatar.jpg",
  ];
  const out = rankInspirationUrls(urls, { max: 2 });
  assert.equal(out.length, 2);
  assert.ok(out[0].includes("pinimg") || out[0].includes("behance"));
  assert.ok(!out.some((u) => u.includes("avatar")));
});

// ----------------------------------------------------------------- brief ---

test("normalizeHex: expands short hex, upper-cases, rejects junk", () => {
  assert.equal(normalizeHex("#abc"), "#AABBCC");
  assert.equal(normalizeHex("0e3b2e"), "#0E3B2E");
  assert.equal(normalizeHex("nope", "#123456"), "#123456");
});

test("normalizeBrief: fills every field even when the model returns nothing", () => {
  const b = normalizeBrief({}, { headline: "COLD BREW" });
  assert.equal(b.copy.headline, "COLD BREW");
  assert.ok(b.palette.length >= 3);
  assert.ok(b.typography.some((t) => t.role === "headline"));
  assert.ok(b.elements.length >= 2);
  assert.equal(b.canvas.aspect_ratio, "1:1");
  assert.ok(b.negative_prompt.length > 10);
});

test("normalizeBrief: user copy wins, sizes clamp, palette shares normalise", () => {
  const b = normalizeBrief({
    copy: { headline: "MODEL GUESS", body: "model body", cta: "Buy" },
    typography: [{ role: "headline", font: "Grotesk", size_px: 999, weight: 812, tracking: -9, case: "upper" }],
    palette: [{ hex: "#112233", role: "dominant", share_pct: 5 }, { hex: "#445566", role: "accent", share_pct: 5 }],
  }, { headline: "REAL HEADLINE", body: "real body" });

  assert.equal(b.copy.headline, "REAL HEADLINE");
  assert.equal(b.copy.body, "real body");
  assert.equal(b.copy.cta, "Buy");
  const h = b.typography.find((t) => t.role === "headline");
  assert.equal(h.size_px, 160, "size clamped to the headline range");
  assert.equal(h.weight, 800, "weight snapped to a real font weight");
  assert.equal(h.tracking, -4, "tracking clamped");
  assert.deepEqual(b.palette.map((p) => p.hex), ["#112233", "#445566"]);
  assert.equal(b.palette.reduce((n, p) => n + p.share_pct, 0), 100);
});

test("normalizeBrief: invalid/duplicate hexes are dropped, not silently kept", () => {
  const b = normalizeBrief({ palette: [{ hex: "#zzzzzz" }, { hex: "#00FF00" }, { hex: "#00ff00" }, { hex: "00ff00" }] }, {});
  assert.equal(b.palette.filter((p) => p.hex === "#00FF00").length, 1);
  assert.ok(b.palette.every((p) => /^#[0-9A-F]{6}$/.test(p.hex)));
});

test("briefToPrompt: spells out arrangement, alignment, colours, type sizes, elements, background, exact copy", () => {
  const b = normalizeBrief({
    canvas: { aspect_ratio: "4:5", usage: "instagram post" },
    layout: { grid: "modular", margins_pct: 6, alignment: "center", focal: "center top" },
    palette: [{ hex: "#123456", role: "dominant", share_pct: 60 }, { hex: "#FFFFFF", role: "secondary", share_pct: 30 }, { hex: "#FF9900", role: "accent", share_pct: 10 }],
    typography: [{ role: "headline", font: "Grotesk", size_px: 96, weight: 800, tracking: -1, case: "UPPER" }],
    elements: [{ type: "icon", description: "coffee bean line icon", placement: "top right", size_pct: 8 }],
    background: { description: "paper texture", color_hex: "#F4EFE4" },
    copy: { headline: "COLD BREW", subhead: "now pouring", body: "small batch", cta: "Order now" },
  }, { headline: "COLD BREW", cta: "Order now" });

  const p = briefToPrompt(b);
  for (const fact of ["4:5", "modular", "center", "#123456", "60%", "Grotesk", "96px", "coffee bean line icon", "paper texture", '"COLD BREW"', '"Order now"', "AVOID"]) {
    assert.ok(p.includes(fact), `prompt is missing ${fact}`);
  }
});

// ---------------------------------------------------------- verification ---

test("judgeVerification: threshold decides, explicit pass is honoured, junk scores 0", () => {
  assert.equal(judgeVerification({ match_score: 0.9 }).pass, true);
  assert.equal(judgeVerification({ match_score: 0.41, issues: ["headline misspelled"] }).pass, false);
  assert.equal(judgeVerification({ match_score: 0.95, pass: false }).pass, false);
  const junk = judgeVerification(null);
  assert.equal(junk.score, 0);
  assert.equal(junk.pass, false);
  assert.equal(judgeVerification({ match_score: 7 }).score, 1, "scores clamp");
});

test("describeReference: parses the vision reply into design DNA", async () => {
  const seen = [];
  const ref = "data:image/png;base64,AAAA";
  const out = await describeReference(ref, "coffee flyer", {
    chat: async (messages) => {
      seen.push(messages);
      return '```json\n{"palette":[{"hex":"#0E3B2E","role":"dominant"}],"layout":{"alignment":"left"},"mood":"warm"}\n```';
    },
  });
  assert.equal(out.dna.mood, "warm");
  assert.equal(out.ref, ref);
  assert.ok(JSON.stringify(seen[0]).includes("coffee flyer"));
});

// ---------------------------------------------------------------- pipeline ---

const fakeBrief = {
  layout: { grid: "rule-of-thirds", margins_pct: 8, alignment: "left", focal: "left third", reading_order: "headline → cta" },
  copy: { headline: "COLD BREW", cta: "Order now" },
  palette: [{ hex: "#0E3B2E", role: "dominant", share_pct: 60 }, { hex: "#F4EFE4", role: "secondary", share_pct: 30 }, { hex: "#D4A574", role: "accent", share_pct: 10 }],
};

function baseDeps(over = {}) {
  const prompts = [];
  return {
    prompts,
    deps: {
      searchImages: async () => ({
        queries: ["site:pinterest.com coffee"],
        refs: ["https://i.pinimg.com/736x/ref.jpg"],
        errors: [],
      }),
      describeReference: async (ref) => ({ ref, dna: { mood: "warm", layout: { alignment: "left" } } }),
      synthesizeBrief: async () => fakeBrief,
      generate: async (prompt) => {
        prompts.push(prompt);
        return { image_url: "https://cdn.example.com/out.png", model: "test-image" };
      },
      verifyRender: async () => ({ score: 0.9, pass: true, issues: [], checked: true }),
      ...over,
    },
  };
}

test("verifyRender: retries a prose reply, then reports an unusable one honestly", async () => {
  let calls = 0;
  const img = `data:image/png;base64,${"A".repeat(80)}`;
  const chat = async () => (++calls === 1 ? "Looks good to me!" : '{"match_score":0.81,"pass":true,"issues":[]}');
  const v = await verifyRender(img, fakeBrief, { chat, threshold: 0.72 });
  assert.equal(v.pass, true);
  assert.equal(v.score, 0.81);
  assert.equal(calls, 2);

  const junk = await verifyRender(img, fakeBrief, { chat: async () => "still prose", threshold: 0.72 });
  assert.equal(junk.pass, false);
  assert.match(junk.issues[0], /not usable JSON/);
});

test("asImageUrl: accepts data URIs and http URLs, wraps bare base64, rejects junk", () => {
  assert.equal(asImageUrl("https://cdn.example.com/a.png"), "https://cdn.example.com/a.png");
  assert.equal(asImageUrl("data:image/png;base64,AAA"), "data:image/png;base64,AAA");
  const b64 = "A".repeat(400);
  assert.equal(asImageUrl(b64), `data:image/png;base64,${b64}`);
  assert.equal(asImageUrl("not an image"), null);
  assert.equal(asImageUrl(""), null);
});

test("synthesizeBrief: retries once when the model answers with prose instead of JSON", async () => {
  const calls = [];
  const out = await synthesizeBrief("coffee flyer", { headline: "COLD BREW" }, [], {
    chat: async (messages) => {
      calls.push(messages);
      return calls.length === 1
        ? "Sure! Here are some design ideas: warm colors and a big headline."
        : '{"layout":{"alignment":"center"},"copy":{"headline":"COLD BREW"}}';
    },
  });
  assert.equal(calls.length, 2);
  assert.equal(out.copy.headline, "COLD BREW");
  assert.equal(out.layout.alignment, "center");
});

test("runDesignAgent: full run returns brief, render and a passing verification", async () => {
  const { deps, prompts } = baseDeps();
  const out = await runDesignAgent({ goal: "coffee shop flyer", headline: "COLD BREW" }, deps);

  assert.equal(out.ok, true);
  assert.equal(out.brief.copy.headline, "COLD BREW");
  assert.equal(out.render.image_url, "https://cdn.example.com/out.png");
  assert.equal(out.verification.pass, true);
  assert.equal(out.attempts, 1);
  assert.equal(out.inspiration.read, 1);
  assert.ok(prompts[0].includes("#0E3B2E"), "the render prompt carries the brief palette");
  assert.deepEqual(out.stages.map((s) => s.stage), ["research", "references", "brief", "render"]);
});

test("runDesignAgent: inspiration failure is tolerated and surfaced as a warning", async () => {
  const { deps, prompts } = baseDeps({
    searchImages: async () => { throw new Error("bing blocked"); },
  });
  const out = await runDesignAgent({ goal: "coffee shop flyer" }, deps);
  assert.equal(out.ok, true);
  assert.equal(out.inspiration.refs.length, 0);
  assert.ok(out.warnings.some((w) => /inspiration skipped/.test(w)));
  assert.equal(prompts.length, 1);
});

test("runDesignAgent: a failed verification retries ONCE with the issues as corrections", async () => {
  let checks = 0;
  const { deps, prompts } = baseDeps({
    verifyRender: async () => {
      checks++;
      return checks === 1
        ? { score: 0.3, pass: false, issues: ["headline is misspelled", "palette drifted"], checked: true }
        : { score: 0.88, pass: true, issues: [], checked: true };
    },
  });
  const out = await runDesignAgent({ goal: "coffee flyer" }, deps);
  assert.equal(out.attempts, 2);
  assert.equal(out.verification.pass, true);
  assert.match(prompts[1], /CORRECTIONS FROM REVIEW: headline is misspelled; palette drifted/);
});

test("runDesignAgent: verify:false skips review entirely (single render)", async () => {
  const { deps, prompts } = baseDeps({ verifyRender: async () => { throw new Error("must not be called"); } });
  const out = await runDesignAgent({ goal: "coffee flyer", verify: false }, deps);
  assert.equal(prompts.length, 1);
  assert.equal(out.verification.checked, false);
  assert.equal(out.attempts, 1);
});

test("runDesignAgent: unusable references do not break the run", async () => {
  const { deps } = baseDeps({
    describeReference: async () => { throw new Error("vision refused"); },
  });
  const out = await runDesignAgent({ goal: "coffee flyer" }, deps);
  assert.equal(out.ok, true);
  assert.equal(out.inspiration.read, 0);
  assert.ok(out.warnings.some((w) => /reference unreadable/.test(w)));
});

test("runDesignAgent: requires a goal and a generator", async () => {
  await assert.rejects(() => runDesignAgent({}, baseDeps().deps), /goal required/);
  const { deps } = baseDeps();
  delete deps.generate;
  await assert.rejects(() => runDesignAgent({ goal: "flyer" }, deps), /generate dependency missing/);
});

test("defaultBrief is a valid starting point on its own", () => {
  const b = defaultBrief({ headline: "SALE" });
  assert.equal(b.copy.headline, "SALE");
  assert.ok(briefToPrompt(b).includes("SALE"));
});
