import test from "node:test";
import assert from "node:assert/strict";

import worker from "./worker.js";

const BUILD = Object.freeze({
  sourceCommit: "470b907ae37abff249476c618fb829556b2286eb",
  imageTag: "cached-models-21",
  buildId: "36212956644-1"
});

const CAPABILITIES = Object.freeze({
  modes: {
    create: { available: true, family: "fl2va" },
    animate: { available: true, family: "fl2va" },
    references: { available: true, family: "ref2va" },
    remix: { available: true, family: "ref2va" }
  },
  quality: {
    draft: { dimensions: { "16:9": "512x288" }, steps: 20 },
    standard: { dimensions: { "16:9": "1024x576" }, steps: 20 },
    hd: { dimensions: { "16:9": "1248x704" }, steps: 20 },
    "2k": { dimensions: { "16:9": "2048x1152" }, steps: 20 }
  },
  aspectRatios: ["16:9"],
  duration: { minSeconds: 0.2083, maxSeconds: 149.6667, fps: 24 },
  references: {
    roles: {
      image: ["general", "character"],
      video: ["general", "source"],
      audio: ["general"]
    },
    fidelity: ["standard", "high"],
    maxImages: 9,
    maxVideos: 3,
    maxAudio: 3,
    maxVideoSoundtracks: 3,
    maxTotal: 12
  },
  byFamily: {
    fl2va: ["quality", "turbo", "turboFast"],
    ref2va: ["quality", "turboFast"]
  },
  worker: { build: BUILD }
});

function makeEnv() {
  let state = {};
  const updates = [];

  return {
    JOB_TOKEN_SECRET: "canonical-test-secret",
    JOB_TOKEN_TTL_SECONDS: "3600",
    RUNPOD_ENDPOINT_ID: "endpoint-legacy",
    RUNPOD_BLACKWELL_ENDPOINT_ID: "endpoint-blackwell",
    RUNPOD_API_KEY: "runpod-test-key",
    CF_VERSION_METADATA: { id: "zero-traffic-test-version" },
    JOB_CHANNEL: {
      idFromName: (name) => ({ name }),
      get: () => ({
        fetch: async (request) => {
          const path = new URL(request.url).pathname;
          if (path.endsWith("/update")) {
            const update = await request.json();
            updates.push(update);
            state = { ...state, ...update };
            return new Response(null, { status: 204 });
          }
          return new Response(JSON.stringify(state), {
            headers: { "Content-Type": "application/json" }
          });
        }
      })
    },
    H3_OUTPUTS: {
      delete: async () => {},
      head: async () => null
    },
    __state: () => state,
    __updates: updates
  };
}

function stubRunPod(capabilities = CAPABILITIES) {
  const original = globalThis.fetch;
  const seen = [];

  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    const body = init.body ? JSON.parse(init.body) : null;
    seen.push({ url, body });

    if (url.endsWith("/run") && body?.input?.mode === "capabilities") {
      return Response.json({ id: "capability-job", status: "IN_QUEUE" });
    }
    if (url.endsWith("/status/capability-job")) {
      return Response.json({
        id: "capability-job",
        status: "COMPLETED",
        output: { capabilities }
      });
    }
    if (url.endsWith("/run")) {
      return Response.json({ id: "generation-job", status: "IN_QUEUE" });
    }
    throw new Error(`Unexpected RunPod test request: ${new URL(url).pathname}`);
  };

  return {
    seen,
    restore() {
      globalThis.fetch = original;
    }
  };
}

function canonicalCreate() {
  return new Request("https://worker.example/canonical/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input: {
        mode: "create",
        prompt: "A non-sensitive contract probe",
        quality: "draft",
        aspectRatio: "16:9",
        duration: 5,
        seed: 1701,
        generationMode: "quality",
        privacy: { mode: "standard" }
      },
      assets: []
    })
  });
}

test("health pins the approved immutable cached-models-2 identity", async () => {
  const response = await worker.fetch(new Request("https://worker.example/health"), makeEnv());
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.deepEqual(body.canonical.expectedBuild, BUILD);
  assert.equal(body.version, "zero-traffic-test-version");
  assert.equal(body.features.canonicalMultimodal, true);
});

test("canonical capabilities refresh from RunPod and then use the authoritative cache", async () => {
  const env = makeEnv();
  const runpod = stubRunPod();
  try {
    const refreshed = await worker.fetch(
      new Request("https://worker.example/capabilities?refresh=1"),
      env
    );
    const firstBody = await refreshed.json();

    assert.equal(refreshed.status, 200);
    assert.deepEqual(firstBody.worker.build, BUILD);
    assert.equal(refreshed.headers.get("X-Privora-Capability-Cache"), "refreshed");
    assert.deepEqual(runpod.seen[0].body, { input: { mode: "capabilities" } });
    assert.equal(runpod.seen.length, 2);
    assert.deepEqual(env.__state().capabilities.worker.build, BUILD);

    const cached = await worker.fetch(new Request("https://worker.example/capabilities"), env);
    assert.equal(cached.status, 200);
    assert.equal(cached.headers.get("X-Privora-Capability-Cache"), "hit");
    assert.equal(runpod.seen.length, 2, "a fresh authoritative cache must avoid another probe");
  } finally {
    runpod.restore();
  }
});

test("canonical create transports product fields without constructing a workflow", async () => {
  const env = makeEnv();
  const runpod = stubRunPod();
  try {
    assert.equal(
      (await worker.fetch(new Request("https://worker.example/capabilities?refresh=1"), env)).status,
      200
    );

    const response = await worker.fetch(canonicalCreate(), env);
    const body = await response.json();
    const generation = runpod.seen.find((call) => call.body?.input?.mode === "create");

    assert.equal(response.status, 202, JSON.stringify(body));
    assert.ok(generation, "canonical input was not submitted to RunPod");
    assert.equal(generation.body.input.generationMode, "quality");
    assert.equal(generation.body.input.quality, "draft");
    assert.equal(generation.body.input.duration, 5);
    assert.equal(generation.body.input.aspectRatio, "16:9");
    for (const forbidden of ["workflow", "width", "height", "frames", "steps"]) {
      assert.equal(generation.body.input[forbidden], undefined, `${forbidden} must remain RunPod-owned`);
    }
    assert.deepEqual(body.worker.build, BUILD);
  } finally {
    runpod.restore();
  }
});

test("canonical generation refuses the previous multimodal-3 build", async () => {
  const env = makeEnv();
  const capabilities = {
    ...CAPABILITIES,
    worker: {
      build: {
        sourceCommit: "2fafcf35c95e9826bb11cb2120f8732ef178abd5",
        imageTag: "multimodal-3",
        buildId: "33254392713-1"
      }
    }
  };
  const runpod = stubRunPod(capabilities);
  try {
    assert.equal(
      (await worker.fetch(new Request("https://worker.example/capabilities?refresh=1"), env)).status,
      200
    );

    const response = await worker.fetch(canonicalCreate(), env);
    const body = await response.json();

    assert.equal(response.status, 503);
    assert.equal(body.code, "DEPLOYMENT_ACTION_REQUIRED");
    assert.deepEqual(body.details.expected, BUILD);
    assert.equal(body.details.actual.imageTag, "multimodal-3");
    assert.equal(runpod.seen.filter((call) => call.body?.input?.mode === "create").length, 0);
  } finally {
    runpod.restore();
  }
});


test("status persists terminal results and serves them after RunPod records expire", async () => {
  const env = makeEnv();
  let expired = false;
  const original = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.includes("/status/generation-job")) {
      if (expired) {
        return new Response(JSON.stringify({ error: "Record not found" }), { status: 404 });
      }
      return Response.json({
        id: "generation-job",
        status: "COMPLETED",
        executionTime: 123456,
        delayTime: 2000,
        output: { generation: { quality: "draft" } }
      });
    }
    throw new Error(`Unexpected test request: ${url}`);
  };
  try {
    const first = await worker.fetch(
      new Request("https://worker.example/status/generation-job?jobId=job-terminal-1"),
      env
    );
    assert.equal(first.status, 200);
    const firstBody = await first.json();
    assert.equal(firstBody.persisted, undefined);
    assert.equal(firstBody.executionTime, 123456);
    assert.ok(env.__state().terminalResult, "terminal snapshot should be persisted");
    assert.equal(env.__state().terminalResult.executionTime, 123456);

    expired = true;
    const second = await worker.fetch(
      new Request("https://worker.example/status/generation-job?jobId=job-terminal-1"),
      env
    );
    assert.equal(second.status, 200);
    const secondBody = await second.json();
    assert.equal(secondBody.persisted, true);
    assert.equal(secondBody.executionTime, 123456);
    assert.equal(secondBody.productStatus, "completed");
  } finally {
    globalThis.fetch = original;
  }
});
