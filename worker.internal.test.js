/*
 * Tests for the internal RunPod -> Cloudflare surface: job-scoped tokens, R2 key naming,
 * keyframe wiring and the public event protocol.
 *
 * Run with:  node --test
 *
 * The token tests are the security boundary of this whole design. Everything RunPod is
 * allowed to do - report progress, upload a video, read an input asset - is gated on one
 * of these, so "wrong job", "wrong purpose" and "expired" each get their own test.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  signJobToken,
  verifyJobToken,
  outputKey,
  inputKey,
  normalizeAsset,
  normalizeRequest,
  loadWorkflowTemplate,
  publicEvent,
  workerConstants
} from "./worker.js";

const { MODES, TOKEN_PURPOSES } = workerConstants();

const SECRET = "test-secret-value";

function expectHttpError(fn, status, messageMatch) {
  try {
    fn();
  } catch (error) {
    assert.equal(error.name, "HttpError", `expected HttpError, got ${error}`);
    assert.equal(error.status, status, `expected ${status}, got ${error.status}: ${error.message}`);
    if (messageMatch) assert.match(error.message, messageMatch);
    return error;
  }
  assert.fail("expected an HttpError but none was thrown");
}

/* -- job-scoped tokens --------------------------------------------------------------- */

test("a minted token verifies for its own job and purpose", async () => {
  const token = await signJobToken(SECRET, "job-1", TOKEN_PURPOSES.progress);
  const payload = await verifyJobToken(SECRET, token, "job-1", TOKEN_PURPOSES.progress);
  assert.equal(payload.jid, "job-1");
  assert.equal(payload.pur, "progress");
});

test("a token for one job is rejected for another", async () => {
  const token = await signJobToken(SECRET, "job-1", TOKEN_PURPOSES.progress);
  await assert.rejects(
    () => verifyJobToken(SECRET, token, "job-2", TOKEN_PURPOSES.progress),
    (e) => e.status === 403
  );
});

test("a progress token cannot upload output", async () => {
  const token = await signJobToken(SECRET, "job-1", TOKEN_PURPOSES.progress);
  await assert.rejects(
    () => verifyJobToken(SECRET, token, "job-1", TOKEN_PURPOSES.output),
    (e) => e.status === 403
  );
});

test("an expired token is rejected", async () => {
  const token = await signJobToken(SECRET, "job-1", TOKEN_PURPOSES.progress, -1);
  await assert.rejects(
    () => verifyJobToken(SECRET, token, "job-1", TOKEN_PURPOSES.progress),
    (e) => e.status === 401 && /expired/i.test(e.message)
  );
});

test("a token signed with a different secret is rejected", async () => {
  const token = await signJobToken("other-secret", "job-1", TOKEN_PURPOSES.progress);
  await assert.rejects(
    () => verifyJobToken(SECRET, token, "job-1", TOKEN_PURPOSES.progress),
    (e) => e.status === 401
  );
});

test("a tampered payload is rejected", async () => {
  const token = await signJobToken(SECRET, "job-1", TOKEN_PURPOSES.progress);
  const signature = token.split(".")[1];
  const forged = Buffer.from(
    JSON.stringify({ jid: "job-9", pur: "output-upload", exp: 9e9 })
  ).toString("base64url");
  await assert.rejects(
    () => verifyJobToken(SECRET, forged + "." + signature, "job-9", TOKEN_PURPOSES.output),
    (e) => e.status === 401
  );
});

test("missing and malformed tokens are 401s", async () => {
  for (const bad of ["", null, "nodot", undefined]) {
    await assert.rejects(
      () => verifyJobToken(SECRET, bad, "job-1", TOKEN_PURPOSES.progress),
      (e) => e.status === 401
    );
  }
});

test("signing without a secret is a 500, not a silently unsigned token", async () => {
  await assert.rejects(
    () => signJobToken("", "job-1", TOKEN_PURPOSES.progress),
    (e) => e.status === 500
  );
});

/* -- R2 key naming ------------------------------------------------------------------- */

test("output and input keys are deterministic and namespaced by job", () => {
  assert.equal(outputKey("job-1"), "outputs/job-1/video.mp4");
  assert.equal(inputKey("job-1", "first-frame", ".png"), "inputs/job-1/first-frame.png");
});

test("job and asset ids that could escape the namespace are rejected", () => {
  for (const bad of ["../evil", "a/b", "", "with space", "x".repeat(300)]) {
    expectHttpError(() => outputKey(bad), 400);
  }
  for (const bad of ["../evil", "a/b", "", "with space"]) {
    expectHttpError(() => inputKey("job-1", bad, ".png"), 400);
  }
});

/* -- keyframe wiring ----------------------------------------------------------------- */

test("r2_key resolves to an asset id rather than being passed through", () => {
  assert.deepEqual(normalizeAsset({ r2_key: "inputs/job-1/first-frame.png" }, "first_frame"), {
    kind: "r2",
    value: "first-frame"
  });
});

test("asset_id is accepted directly", () => {
  assert.deepEqual(normalizeAsset({ asset_id: "first-frame" }, "first_frame"), {
    kind: "r2",
    value: "first-frame"
  });
});

test("last-frame and first+last modes are available", () => {
  for (const mode of ["last_frame_to_video", "first_last_frame_to_video"]) {
    assert.equal(MODES[mode].implemented, true, mode);
    assert.ok(loadWorkflowTemplate(mode));
  }
});

test("the first+last template wires two independent loaders", () => {
  const wf = loadWorkflowTemplate("first_last_frame_to_video");
  assert.deepEqual(wf.cond.inputs.first_frame, ["first_frame_image", 0]);
  assert.deepEqual(wf.cond.inputs.last_frame, ["last_frame_image", 0]);
  assert.ok(wf.first_frame_image && wf.last_frame_image);
});

test("a mode requiring a keyframe rejects a request without one", () => {
  expectHttpError(
    () => normalizeRequest({ prompt: "x", mode: "last_frame_to_video" }),
    400,
    /requires last_frame/
  );
});

test("reference and 2k remain deferred", () => {
  for (const mode of ["reference", "regenerate_2k"]) {
    assert.equal(MODES[mode].implemented, false, mode);
  }
  expectHttpError(() => normalizeRequest({ prompt: "x", mode: "reference" }), 501, /20.97 GB/);
});

/* -- public event protocol ----------------------------------------------------------- */

test("sampling state becomes a progress event", () => {
  assert.deepEqual(
    publicEvent({ jobId: "j", phase: "sampling", step: 7, steps: 20, percent: 35 }),
    { type: "progress", jobId: "j", phase: "sampling", step: 7, steps: 20, percent: 35 }
  );
});

test("decoding state becomes a progress event without step fields", () => {
  const event = publicEvent({ jobId: "j", phase: "decoding", percent: 90 });
  assert.equal(event.type, "progress");
  assert.equal(event.phase, "decoding");
  assert.equal(event.step, undefined);
});

test("completion carries a playable route, not a bucket URL", () => {
  const event = publicEvent({
    jobId: "j",
    phase: "completed",
    video: { key: "outputs/j/video.mp4" }
  });
  assert.equal(event.type, "completed");
  assert.equal(event.video.url, "/jobs/j/video");
  assert.ok(!JSON.stringify(event).includes("r2.cloudflarestorage"), "no raw bucket URL");
});

test("failure carries a stable error shape", () => {
  const event = publicEvent({ jobId: "j", phase: "failed", error: { code: "x", message: "y" } });
  assert.equal(event.type, "failed");
  assert.deepEqual(event.error, { code: "x", message: "y" });
});

test("cancellation is its own event type", () => {
  assert.deepEqual(publicEvent({ jobId: "j", phase: "cancelled" }), {
    type: "cancelled",
    jobId: "j"
  });
});

test("an unknown state defaults to queued rather than throwing", () => {
  assert.equal(publicEvent({ jobId: "j" }).phase, "queued");
});

test("internal fields never reach a public event", () => {
  const event = publicEvent({
    jobId: "j",
    phase: "sampling",
    step: 1,
    steps: 20,
    comfyLog: "got prompt\nRequested to load MiniMaxH3"
  });
  assert.equal(event.comfyLog, undefined, "raw ComfyUI text must not be forwarded");
});
