/*
 * Tests for the Cloudflare Worker's normalization layer.
 *
 * Run with:  node --test worker.test.js
 *
 * No dependencies and no network: everything under test is pure. The values asserted
 * here are not opinions - the canvas and frame-grid rules are mirrored from the pinned
 * ComfyUI build's comfy_extras/nodes_minimax_h3.py, so these tests are what stops the
 * Worker drifting away from what the model will actually accept.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  adaptCanvas,
  alignFrameCount,
  isLegalFrameCount,
  durationToFrames,
  framesToDuration,
  buildPrompt,
  resolveMode,
  normalizeRequest,
  normalizeAsset,
  validateR2Key,
  loadWorkflowTemplate,
  applySettings,
  buildWorkflowForSettings,
  describeCapabilities,
  ASPECT_RATIOS,
  QUALITY_PRESETS,
  MODES
} from "./worker.js";

function expectHttpError(fn, status, messageMatch) {
  try {
    fn();
  } catch (error) {
    assert.equal(error.name, "HttpError", `expected HttpError, got ${error}`);
    assert.equal(error.status, status, `expected status ${status}, got ${error.status}: ${error.message}`);
    if (messageMatch) {
      assert.match(error.message, messageMatch);
    }
    return error;
  }
  assert.fail("expected an HttpError but none was thrown");
}

const base = { prompt: "A cinematic ocean scene" };

/* -- frame grid ---------------------------------------------------------------------- */

test("alignFrameCount snaps up to the model's 17k+5 grid", () => {
  assert.equal(alignFrameCount(1), 5);
  assert.equal(alignFrameCount(5), 5);
  assert.equal(alignFrameCount(6), 22);
  assert.equal(alignFrameCount(22), 22);
  assert.equal(alignFrameCount(120), 124);
  assert.equal(alignFrameCount(124), 124);
});

test("every aligned value satisfies frames % 17 == 5", () => {
  for (let n = 1; n < 400; n++) {
    assert.equal(alignFrameCount(n) % 17, 5, `n=${n}`);
  }
});

test("duration 5s resolves to 124 frames", () => {
  assert.equal(durationToFrames(5), 124);
  assert.equal(framesToDuration(124).toFixed(4), "5.1667");
});

test("isLegalFrameCount rejects off-grid and out-of-range values", () => {
  assert.equal(isLegalFrameCount(124), true);
  assert.equal(isLegalFrameCount(120), false);
  assert.equal(isLegalFrameCount(4), false);
  assert.equal(isLegalFrameCount(124.5), false);
  assert.equal(isLegalFrameCount(100000), false);
});

/* -- canvas -------------------------------------------------------------------------- */

test("hd 16:9 lands on the node's own 1344x768 default", () => {
  assert.deepEqual(adaptCanvas(16 / 9, 768), { width: 1344, height: 768 });
});

test("standard 16:9 is 1024x576", () => {
  assert.deepEqual(adaptCanvas(16 / 9, 576), { width: 1024, height: 576 });
});

test("every quality x aspect combination is 32-aligned and within the area cap", () => {
  for (const [qualityName, preset] of Object.entries(QUALITY_PRESETS)) {
    for (const [ratioName, ratio] of Object.entries(ASPECT_RATIOS)) {
      const { width, height } = adaptCanvas(ratio, preset.shortEdge);
      const label = `${qualityName} ${ratioName} -> ${width}x${height}`;
      assert.equal(width % 32, 0, label);
      assert.equal(height % 32, 0, label);
      assert.ok(width * height <= 768 * 1344, `${label} exceeds the area cap`);
      assert.ok(width >= 32 && height >= 32, label);
    }
  }
});

test("portrait ratios are the transpose of their landscape counterparts", () => {
  const landscape = adaptCanvas(ASPECT_RATIOS["16:9"], 768);
  const portrait = adaptCanvas(ASPECT_RATIOS["9:16"], 768);
  assert.deepEqual(portrait, { width: landscape.height, height: landscape.width });
});

/* -- prompt composition -------------------------------------------------------------- */

test("buildPrompt returns the user prompt unchanged when nothing else is supplied", () => {
  assert.equal(buildPrompt({ prompt: "A woman near the ocean." }), "A woman near the ocean.");
});

test("buildPrompt appends only the fields that were supplied", () => {
  const composed = buildPrompt({
    prompt: "A woman standing near the ocean.",
    camera: "slow dolly forward",
    lighting: "warm sunset"
  });

  assert.equal(
    composed,
    "A woman standing near the ocean.\n\nCamera: slow dolly forward.\nLighting: warm sunset."
  );
  assert.ok(!composed.includes("Shot:"), "absent fields must not produce empty labels");
});

test("buildPrompt never rewrites the user's own text", () => {
  const original = "A woman standing near the ocean.";
  const composed = buildPrompt({ prompt: original, style: "cinematic realism" });
  assert.ok(composed.startsWith(original));
});

test("buildPrompt is deterministic and does not double up periods", () => {
  const input = { prompt: "Ocean.", camera: "slow dolly forward." };
  assert.equal(buildPrompt(input), buildPrompt(input));
  assert.ok(!buildPrompt(input).includes(".."));
});

/* -- mode routing -------------------------------------------------------------------- */

test("mode defaults to text_to_video", () => {
  assert.equal(resolveMode({}), "text_to_video");
});

test("mode is inferred from supplied keyframes", () => {
  assert.equal(resolveMode({ first_frame: { url: "https://x/a.png" } }), "first_frame_to_video");
  assert.equal(resolveMode({ last_frame: { url: "https://x/b.png" } }), "last_frame_to_video");
  assert.equal(
    resolveMode({ first_frame: { url: "https://x/a.png" }, last_frame: { url: "https://x/b.png" } }),
    "first_last_frame_to_video"
  );
});

test("reference inputs infer the reference mode", () => {
  assert.equal(resolveMode({ reference_images: [{ url: "https://x/r.png" }] }), "reference");
});

test("an unknown mode is a 400", () => {
  expectHttpError(() => resolveMode({ mode: "teleport" }), 400, /Unknown mode/);
});

test("unimplemented modes return 501 rather than silently degrading", () => {
  for (const mode of ["last_frame_to_video", "first_last_frame_to_video", "reference", "regenerate_2k"]) {
    const error = expectHttpError(() => normalizeRequest({ ...base, mode }), 501);
    assert.match(error.message, new RegExp(mode));
  }
});

/* -- presets and precedence ---------------------------------------------------------- */

test("defaults produce the documented standard 16:9 settings", () => {
  const s = normalizeRequest({ ...base, seed: 51 });
  assert.equal(s.width, 1024);
  assert.equal(s.height, 576);
  assert.equal(s.frames, 124);
  assert.equal(s.steps, 20);
  assert.equal(s.fps, 24);
  assert.equal(s.seed, 51);
  assert.equal(s.mode, "text_to_video");
});

test("fast uses 14 steps, hd uses the 768 canvas", () => {
  assert.equal(normalizeRequest({ ...base, quality: "fast" }).steps, 14);

  const hd = normalizeRequest({ ...base, quality: "hd" });
  assert.equal(hd.width, 1344);
  assert.equal(hd.height, 768);
  assert.equal(hd.steps, 20);
});

test("aspect_ratio selects the canvas within a quality tier", () => {
  const portrait = normalizeRequest({ ...base, quality: "hd", aspect_ratio: "9:16" });
  assert.equal(portrait.width, 768);
  assert.equal(portrait.height, 1344);

  const square = normalizeRequest({ ...base, quality: "hd", aspect_ratio: "1:1" });
  assert.equal(square.width, 768);
  assert.equal(square.height, 768);
});

test("explicit width/height beat the quality preset", () => {
  const s = normalizeRequest({ ...base, quality: "hd", width: 1024, height: 576 });
  assert.equal(s.width, 1024);
  assert.equal(s.height, 576);
  assert.equal(s.resolvedFrom.canvas, "explicit");
});

test("explicit steps beat the quality preset", () => {
  assert.equal(normalizeRequest({ ...base, quality: "fast", steps: 30 }).steps, 30);
});

test("explicit frames beat duration when they agree", () => {
  const s = normalizeRequest({ ...base, duration: 5, frames: 124 });
  assert.equal(s.frames, 124);
  assert.equal(s.resolvedFrom.frames, "explicit");
});

/* -- conflicting combinations are rejected, never silently resolved ------------------- */

test("width/height contradicting aspect_ratio is a 400", () => {
  expectHttpError(
    () => normalizeRequest({ ...base, aspect_ratio: "1:1", width: 1024, height: 576 }),
    400,
    /contradicts aspect_ratio/
  );
});

test("frames contradicting duration is a 400", () => {
  expectHttpError(() => normalizeRequest({ ...base, duration: 10, frames: 124 }), 400, /contradicts duration/);
});

test("width without height is a 400", () => {
  expectHttpError(() => normalizeRequest({ ...base, width: 1024 }), 400, /together/);
});

test("off-grid frames are rejected with the nearest legal values", () => {
  const error = expectHttpError(() => normalizeRequest({ ...base, frames: 120 }), 400, /17 == 5/);
  assert.match(error.message, /124/);
});

test("non-multiple-of-32 dimensions are rejected", () => {
  expectHttpError(() => normalizeRequest({ ...base, width: 1000, height: 576 }), 400, /multiples of 32/);
});

test("an oversized canvas is rejected", () => {
  expectHttpError(() => normalizeRequest({ ...base, width: 1344, height: 1344 }), 400, /maximum area/);
});

test("unknown quality and aspect_ratio are 400s", () => {
  expectHttpError(() => normalizeRequest({ ...base, quality: "ultra" }), 400, /Unknown quality/);
  expectHttpError(() => normalizeRequest({ ...base, aspect_ratio: "21:9" }), 400, /Unsupported aspect_ratio/);
});

test("a missing prompt is a 400", () => {
  expectHttpError(() => normalizeRequest({}), 400, /prompt is required/);
  expectHttpError(() => normalizeRequest({ prompt: "   " }), 400, /prompt is required/);
});

test("steps out of range is a 400", () => {
  expectHttpError(() => normalizeRequest({ ...base, steps: 0 }), 400, /between 1 and 100/);
  expectHttpError(() => normalizeRequest({ ...base, steps: 101 }), 400, /between 1 and 100/);
});

/* -- backward compatibility ---------------------------------------------------------- */

test("the existing raw request shape still works unchanged", () => {
  const s = normalizeRequest({
    backend: "h3-blackwell",
    prompt: "A cinematic ocean scene",
    width: 1024,
    height: 576,
    frames: 124,
    steps: 20,
    seed: 51
  });

  assert.equal(s.width, 1024);
  assert.equal(s.height, 576);
  assert.equal(s.frames, 124);
  assert.equal(s.steps, 20);
  assert.equal(s.seed, 51);
  assert.equal(s.mode, "text_to_video");
  assert.equal(s.prompt, "A cinematic ocean scene", "no structured fields means no appended lines");
});

test("a raw request produces the same graph the old buildWorkflow did", () => {
  const s = normalizeRequest({ prompt: "x", width: 1024, height: 576, frames: 124, steps: 20, seed: 51 });
  const wf = buildWorkflowForSettings(s);

  assert.equal(wf.unet.class_type, "UNETLoader");
  assert.equal(wf.cond.class_type, "MiniMaxH3ImageToVideo");
  assert.equal(wf.cond.inputs.width, 1024);
  assert.equal(wf.cond.inputs.height, 576);
  assert.equal(wf.cond.inputs.length, 124);
  assert.equal(wf.sigmas.inputs.steps, 20);
  assert.equal(wf.noise.inputs.noise_seed, 51);
  assert.equal(wf.video.inputs.fps, 24);
  assert.equal(wf.save.class_type, "SaveVideo");
  assert.ok(!("first_frame" in wf.cond.inputs), "text-to-video must not wire a keyframe");
});

/* -- workflow templates -------------------------------------------------------------- */

test("templates exist for every implemented mode and 501 otherwise", () => {
  for (const [mode, spec] of Object.entries(MODES)) {
    if (spec.implemented) {
      assert.ok(loadWorkflowTemplate(mode), mode);
    } else {
      expectHttpError(() => loadWorkflowTemplate(mode), 501);
    }
  }
});

test("the first-frame template wires LoadImage into cond.first_frame", () => {
  const wf = loadWorkflowTemplate("first_frame_to_video");
  assert.equal(wf.first_frame_image.class_type, "LoadImage");
  assert.deepEqual(wf.cond.inputs.first_frame, ["first_frame_image", 0]);
});

test("every node reference in every template points at a real node", () => {
  for (const mode of Object.keys(MODES).filter((m) => MODES[m].implemented)) {
    const wf = loadWorkflowTemplate(mode);
    for (const [nodeName, node] of Object.entries(wf)) {
      for (const [inputName, value] of Object.entries(node.inputs)) {
        if (Array.isArray(value) && value.length === 2 && typeof value[0] === "string") {
          assert.ok(
            Object.prototype.hasOwnProperty.call(wf, value[0]),
            `${mode}: ${nodeName}.${inputName} -> unknown node '${value[0]}'`
          );
        }
      }
    }
  }
});

test("applySettings is pure and does not mutate the template", () => {
  const template = loadWorkflowTemplate("text_to_video");
  const before = JSON.stringify(template);
  applySettings(template, normalizeRequest({ ...base, seed: 7 }));
  assert.equal(JSON.stringify(template), before);
});

test("templates never reference numeric ComfyUI node ids", () => {
  const wf = loadWorkflowTemplate("text_to_video");
  for (const key of Object.keys(wf)) {
    assert.ok(Number.isNaN(Number(key)), `node key '${key}' looks like a raw ComfyUI id`);
  }
});

/* -- assets and R2 keys -------------------------------------------------------------- */

test("an https keyframe URL is accepted", () => {
  assert.deepEqual(normalizeAsset({ url: "https://example.com/a.png" }, "first_frame"), {
    kind: "url",
    value: "https://example.com/a.png"
  });
});

test("a non-https keyframe URL is a 400", () => {
  expectHttpError(() => normalizeAsset({ url: "http://example.com/a.png" }, "first_frame"), 400, /https/);
});

test("supplying two asset sources at once is a 400", () => {
  expectHttpError(
    () => normalizeAsset({ url: "https://x/a.png", base64: "abc" }, "first_frame"),
    400,
    /exactly one/
  );
});

test("r2_key is validated but not yet deliverable", () => {
  expectHttpError(() => normalizeAsset({ r2_key: "inputs/u/j/a.png" }, "first_frame"), 501, /R2 binding/);
});

test("validateR2Key rejects traversal, absolute paths and odd characters", () => {
  assert.equal(validateR2Key("inputs/user/job/first.png"), "inputs/user/job/first.png");

  for (const bad of [
    "",
    "/inputs/a.png",
    "inputs/../secrets.png",
    "inputs/a\\b.png",
    "outputs/a.png",
    "inputs//a.png",
    "inputs/a b.png",
    "inputs/a*.png"
  ]) {
    expectHttpError(() => validateR2Key(bad), 400, /r2_key|inputs\//);
  }
});

/* -- capabilities -------------------------------------------------------------------- */

test("capabilities advertises real dimensions and marks unavailable modes", () => {
  const caps = describeCapabilities();

  assert.equal(caps.fps, 24);
  assert.equal(caps.qualities.standard.dimensions["16:9"], "1024x576");
  assert.equal(caps.qualities.hd.dimensions["16:9"], "1344x768");
  assert.equal(caps.modes.text_to_video.available, true);
  assert.equal(caps.modes.reference.available, false);
  assert.ok(caps.modes.reference.reason.includes("20.97 GB"));
});
