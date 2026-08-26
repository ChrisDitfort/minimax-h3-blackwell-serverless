/*
 * Tests that drive the Worker's actual fetch handler, not just its pure functions.
 *
 * Run with:  node --test
 *
 * These exist because a "url is not defined" ReferenceError reached production: every
 * pure function was covered, but nothing ever called `export default.fetch`, so a plain
 * scoping mistake in the route wiring was invisible. Anything reachable by an HTTP verb
 * needs a test that goes in through the front door.
 */

import test from "node:test";
import assert from "node:assert/strict";

import worker from "./worker.js";

/** Minimal env: real secret, stub Durable Object, stub R2. */
function makeEnv(overrides = {}) {
  const doCalls = [];
  const bucket = new Map();

  return {
    JOB_TOKEN_SECRET: "test-secret",
    JOB_TOKEN_TTL_SECONDS: "3600",
    RUNPOD_ENDPOINT_ID: "endpoint-ada",
    RUNPOD_BLACKWELL_ENDPOINT_ID: "endpoint-blackwell",
    RUNPOD_API_KEY: "runpod-key",

    JOB_CHANNEL: {
      idFromName: (name) => ({ name }),
      get: () => ({
        fetch: async (request) => {
          doCalls.push(await request.json().catch(() => null));
          return new Response(null, { status: 204 });
        }
      })
    },

    H3_OUTPUTS: {
      put: async (key, body) => {
        bucket.set(key, body);
        return { size: 123 };
      },
      get: async (key) => (bucket.has(key) ? { body: "x", httpEtag: '"e"', size: 123, writeHttpMetadata() {} } : null)
    },

    __doCalls: doCalls,
    __bucket: bucket,
    ...overrides
  };
}

/** Replace global fetch so no request ever leaves the test. */
function stubRunPod(response = { id: "runpod-job-1", status: "IN_QUEUE" }) {
  const seen = [];
  const original = globalThis.fetch;

  globalThis.fetch = async (input, init) => {
    seen.push({ url: String(input), init });
    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });
  };

  return {
    seen,
    restore() {
      globalThis.fetch = original;
    }
  };
}

function post(path, body) {
  return new Request(`https://worker.example${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

const LEGACY_BODY = {
  backend: "h3-blackwell",
  prompt: "A cinematic ocean scene",
  width: 1024,
  height: 576,
  frames: 124,
  steps: 20,
  seed: 51
};

test("POST /generate succeeds for the legacy raw body", async () => {
  const runpod = stubRunPod();
  try {
    const response = await worker.fetch(post("/generate", LEGACY_BODY), makeEnv());
    const body = await response.json();

    assert.equal(response.status, 202, `expected 202, got ${response.status}: ${JSON.stringify(body)}`);
    assert.equal(body.error, undefined, `unexpected error: ${body.error}`);
    assert.equal(body.id, "runpod-job-1");
    assert.ok(body.jobId, "a Worker-side job id must be returned");
    assert.deepEqual(body.settings, {
      width: 1024,
      height: 576,
      frames: 124,
      fps: 24,
      durationSeconds: 124 / 24,
      steps: 20
    });
  } finally {
    runpod.restore();
  }
});

test("POST /generate returns realtime and video routes", async () => {
  const runpod = stubRunPod();
  try {
    const body = await (await worker.fetch(post("/generate", LEGACY_BODY), makeEnv())).json();
    assert.match(body.routes.events, /^\/ws\/jobs\//);
    assert.match(body.routes.video, /^\/jobs\/.+\/video$/);
    assert.match(body.routes.status, /^\/status\/h3-blackwell\/runpod-job-1\?jobId=/);
    assert.match(body.routes.cancel, /^\/cancel\/h3-blackwell\/runpod-job-1\?jobId=/);
  } finally {
    runpod.restore();
  }
});

test("POST /generate hands RunPod job-scoped callback blocks", async () => {
  const runpod = stubRunPod();
  try {
    await worker.fetch(post("/generate", LEGACY_BODY), makeEnv());

    const submission = runpod.seen.find((r) => r.url.includes("/run"));
    assert.ok(submission, "no RunPod submission was made");
    const input = JSON.parse(submission.init.body).input;

    assert.ok(input.workflow, "the ComfyUI workflow must still be sent");
    assert.ok(input.progress?.url.includes("/internal/jobs/"), "progress URL missing");
    assert.ok(input.progress?.token, "progress token missing");
    assert.ok(input.output?.url.includes("/internal/jobs/"), "output URL missing");
    assert.ok(input.output?.token, "output token missing");
    assert.notEqual(input.progress.token, input.output.token, "tokens must be purpose-scoped");
  } finally {
    runpod.restore();
  }
});

test("POST /generate seeds the realtime channel", async () => {
  const runpod = stubRunPod();
  const env = makeEnv();
  try {
    await worker.fetch(post("/generate", LEGACY_BODY), env);
    assert.equal(env.__doCalls.length, 1, "the job channel should be seeded once");
    assert.equal(env.__doCalls[0].phase, "queued");
  } finally {
    runpod.restore();
  }
});

test("POST /generate still works with no JOB_TOKEN_SECRET, just without callbacks", async () => {
  const runpod = stubRunPod();
  const env = makeEnv({ JOB_TOKEN_SECRET: undefined });
  try {
    const response = await worker.fetch(post("/generate", LEGACY_BODY), env);
    const body = await response.json();
    assert.equal(response.status, 202, JSON.stringify(body));

    const input = JSON.parse(runpod.seen[0].init.body).input;
    assert.ok(input.workflow);
    assert.equal(input.progress, undefined, "no secret means no callback block");
  } finally {
    runpod.restore();
  }
});

test("POST /generate rejects a bad request without reaching RunPod", async () => {
  const runpod = stubRunPod();
  try {
    const response = await worker.fetch(post("/generate", { prompt: "" }), makeEnv());
    assert.equal(response.status, 400);
    assert.equal(runpod.seen.length, 0, "an invalid request must not be forwarded");
  } finally {
    runpod.restore();
  }
});

test("GET /health and /capabilities respond through the real handler", async () => {
  for (const path of ["/health", "/capabilities"]) {
    const response = await worker.fetch(new Request(`https://worker.example${path}`), makeEnv());
    assert.equal(response.status, 200, path);
    await response.json();
  }
});

test("internal routes reject an unauthenticated call through the real handler", async () => {
  const response = await worker.fetch(
    post("/internal/jobs/job-1/progress", { phase: "sampling" }),
    makeEnv()
  );
  assert.equal(response.status, 401);
});

test("an unknown route is a 404, not a crash", async () => {
  const response = await worker.fetch(new Request("https://worker.example/nope"), makeEnv());
  assert.equal(response.status, 404);
});

test("every route reachable by a verb responds without a ReferenceError", async () => {
  /*
   * The regression guard. A scoping mistake surfaces as a 500 whose body is
   * "<name> is not defined"; any such response fails this test regardless of route.
   */
  const runpod = stubRunPod();
  const env = makeEnv();
  const requests = [
    new Request("https://worker.example/health"),
    new Request("https://worker.example/capabilities"),
    post("/generate", LEGACY_BODY),
    new Request("https://worker.example/status/h3-blackwell/j1"),
    new Request("https://worker.example/status/h3-blackwell/j1?jobId=w1"),
    new Request("https://worker.example/cancel/h3-blackwell/j1", { method: "POST" }),
    new Request("https://worker.example/cancel/h3-blackwell/j1?jobId=w1", { method: "POST" }),
    new Request("https://worker.example/jobs/w1/video"),
    post("/internal/jobs/w1/progress", { phase: "sampling" }),
    new Request("https://worker.example/internal/jobs/w1/assets/first-frame")
  ];

  try {
    for (const request of requests) {
      const response = await worker.fetch(request, env);
      const text = await response.text();
      assert.ok(
        !/is not defined/.test(text),
        `${request.method} ${new URL(request.url).pathname} -> ReferenceError: ${text.slice(0, 200)}`
      );
    }
  } finally {
    runpod.restore();
  }
});
