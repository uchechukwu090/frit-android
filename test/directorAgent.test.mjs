// ============================================================================
// Director agent tests — the same three stages, for motion
// ============================================================================
import test from "node:test";
import assert from "node:assert/strict";

import {
  planLookQueries, normalizeLook, readLookReference, synthesizeShotPlan,
  enforceShotVariety, verifyShotFrame, runDirectorAgent, normalizeShot,
} from "../src/creative/directorAgent.js";

// --------------------------------------------------------------- research ---

test("planLookQueries: uses cinematography vocabulary, not layout vocabulary", () => {
  const qs = planLookQueries("a Lagos birthday party film", { seed: 5 });
  assert.ok(qs.length >= 2);
  assert.match(qs[0], /site:pinterest\.com/);
  assert.ok(qs.every((q) => !/typography|font|palette/.test(q)));
  assert.deepEqual(planLookQueries("   "), []);
});

test("normalizeLook: keeps only known look axes", () => {
  const look = normalizeLook({ look: { grade_temp: "warm", grade_contrast: "high", lighting: "neon", palette_temp: "lava", framing: "close-up", lens_character: "anamorphic", depth: "shallow", texture: "filmic", composition: "rule-of-thirds" }, palette: [{ hex: "#112233" }], mood: "moody" });
  assert.equal(look.look.grade_temp, "warm");
  assert.equal(look.look.lighting, "neon");
  assert.equal(look.look.palette_temp, "neutral", "an unknown temperature falls back");
  assert.equal(look.look.framing, "close");
  assert.equal(look.palette[0].hex, "#112233");
});

test("readLookReference: parses the vision reply into a look", async () => {
  const out = await readLookReference("data:image/png;base64,AAAA", "a music video", {
    chat: async () => '{"look":{"grade_temp":"cool","lighting":"neon"},"palette":[{"hex":"#001122"}],"mood":"moody"}',
  });
  assert.equal(out.look.grade_temp, "cool");
  assert.equal(out.look.lighting, "neon");
  assert.equal(out.palette[0].hex, "#001122");
});

// ------------------------------------------------------------------ brief ---

test("normalizeShot: clamps the duration and fills every field", () => {
  const s = normalizeShot({ prompt: "a dancer" }, 0);
  assert.equal(s.duration_s, 4);
  assert.equal(s.move, "Static");
  assert.equal(s.aperture, "f/2.8");
  assert.equal(normalizeShot({ duration_s: 99 }, 0).duration_s, 8);
  assert.equal(normalizeShot({ duration_s: 1 }, 0).duration_s, 3);
  assert.equal(normalizeShot({ duration_s: 0 }, 0).duration_s, 4, "zero is invalid -> the default");
});

test("synthesizeShotPlan: parses the shot list and retries a prose reply", async () => {
  let calls = 0;
  const shots = await synthesizeShotPlan("a birthday film", 3, [{ look: { grade_temp: "warm" } }], {
    chat: async () => {
      calls++;
      return calls === 1
        ? "Sure! Here are three lovely shots..."
        : '{"shots":[{"prompt":"a candle being lit","move":"Dolly In","angle":"Close-Up"}]}';
    },
  });
  assert.equal(calls, 2);
  assert.equal(shots.length, 1);
  assert.equal(shots[0].move, "Dolly In");
});

test("enforceShotVariety: two shots cannot share a move AND an angle", () => {
  const { shots, changed } = enforceShotVariety([
    { index: 1, move: "Static", angle: "Eye Level" },
    { index: 2, move: "Static", angle: "Eye Level" },
    { index: 3, move: "Dolly In", angle: "Low Angle" },
  ]);
  assert.equal(changed, 1);
  assert.notEqual(`${shots[1].move}|${shots[1].angle}`, `${shots[0].move}|${shots[0].angle}`);
  assert.equal(shots[2].move, "Dolly In", "a distinct shot is left alone");
});

// ------------------------------------------------------------ verification ---

test("verifyShotFrame: scores the frame and retries a prose reply", async () => {
  let calls = 0;
  const v = await verifyShotFrame("data:image/png;base64,AAA", { prompt: "a dancer", angle: "Close-Up" }, {
    chat: async () => (++calls === 1 ? "Looks great!" : '{"match_score":0.84,"pass":true,"issues":[]}'),
  });
  assert.equal(calls, 2);
  assert.equal(v.pass, true);
  assert.equal(v.score, 0.84);
});

test("verifyShotFrame: an unusable reply is reported honestly", async () => {
  const v = await verifyShotFrame("data:image/png;base64,AAA", { prompt: "x" }, { chat: async () => "prose" });
  assert.equal(v.pass, false);
  assert.equal(v.score, 0);
});

// -------------------------------------------------------------- pipeline ---

function baseDeps(over = {}) {
  return {
    deps: {
      research: async () => ({ refs: ["https://i.pinimg.com/736x/a.jpg"] }),
      readLook: async (ref) => ({ ref, look: { grade_temp: "warm" }, palette: [{ hex: "#112233" }] }),
      planFilm: async () => ([
        { index: 1, prompt: "shot one", move: "Static", angle: "Eye Level" },
        { index: 2, prompt: "shot two", move: "Dolly In", angle: "Low Angle" },
      ]),
      renderShot: async (shot) => ({ ok: true, kind: "image", url: `https://cdn.example.com/${shot.index}.png` }),
      framesOf: async (r) => [r.url],
      verifyFrame: async () => ({ score: 0.9, pass: true, issues: [], checked: true }),
      ...over,
    },
  };
}

test("runDirectorAgent: research -> brief -> render -> verify, all four stages", async () => {
  const { deps } = baseDeps();
  const out = await runDirectorAgent({ brief: "a birthday film", num_shots: 2 }, deps);
  assert.equal(out.ok, true);
  assert.equal(out.shots.length, 2);
  assert.equal(out.shots[0].verification.pass, true);
  assert.deepEqual(out.stages.map((s) => s.stage), ["research", "brief", "render"]);
  assert.equal(out.looks.length, 1);
});

test("runDirectorAgent: a failed frame is RE-RENDERED with the reviewer's complaints", async () => {
  const prompts = [];
  let checks = 0;
  const { deps } = baseDeps({
    planFilm: async () => ([{ index: 1, prompt: "shot one" }]),
    renderShot: async (shot, { correction }) => {
      prompts.push(correction);
      return { ok: true, kind: "image", url: `https://cdn.example.com/${shot.index}.png` };
    },
    verifyFrame: async () => {
      checks++;
      return checks === 1
        ? { score: 0.3, pass: false, issues: ["wrong framing", "colour drifted"], checked: true }
        : { score: 0.9, pass: true, issues: [], checked: true };
    },
  });
  const out = await runDirectorAgent({ brief: "a birthday film", num_shots: 1 }, deps);
  assert.equal(out.shots[0].attempts, 2);
  assert.equal(out.shots[0].verification.pass, true);
  assert.match(prompts[1], /wrong framing; colour drifted/);
});

test("runDirectorAgent: a research failure is a warning, never a dead end", async () => {
  const { deps } = baseDeps({ research: async () => { throw new Error("bing blocked"); } });
  const out = await runDirectorAgent({ brief: "a birthday film", num_shots: 1 }, deps);
  assert.equal(out.ok, true);
  assert.ok(out.warnings.some((w) => /look research skipped/.test(w)));
  assert.equal(out.looks.length, 0);
});

test("runDirectorAgent: a render failure is surfaced, never swallowed", async () => {
  const { deps } = baseDeps({ renderShot: async () => ({ ok: false, error: "provider 500" }) });
  const out = await runDirectorAgent({ brief: "a birthday film", num_shots: 1 }, deps);
  assert.equal(out.ok, false);
  assert.equal(out.shots[0].ok, false);
  assert.ok(out.warnings.some((w) => /provider 500/.test(w)));
});

test("runDirectorAgent: verify:false skips the frame check entirely", async () => {
  const { deps } = baseDeps({ verifyFrame: async () => { throw new Error("must not be called"); } });
  const out = await runDirectorAgent({ brief: "a birthday film", num_shots: 1, verify: false }, deps);
  assert.equal(out.ok, true);
  assert.equal(out.shots[0].verification, undefined);
});

test("runDirectorAgent: a brief is required", async () => {
  await assert.rejects(() => runDirectorAgent({}, baseDeps().deps), /brief required/);
});
