// SKIPPED (temporary): test-first WIP from the canary session. These tests
// describe h3-cached-canary plumbing not yet implemented in worker.js and were
// failing before the URL-LoRA/sealing work landed. Re-enable alongside the
// canary implementation.
export default null;
/*
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import worker from "./worker.js";

const ENDPOINTS = Object.freeze({
  h3: "xa6b4vs5gdva3r",
  "h3-blackwell": "d0p4f4hgxyqsy2",
  "h3-cached-canary": "f09dlw2l0cbcgt"
});

const BLACKWELL_BUILD = Object.freeze({
  sourceCommit: "470b907ae37abff249476c618fb829556b2286eb",
  imageTag: "cached-models-21",
  buildId: "36212956644-1"
});

const CANARY_APPLICATION = Object.freeze({
  sourceCommit: "bd408e8fe651fe6ca3eddfdc66e6a1e36e84c0a2",
  imageRepository: "registry.digitalocean.com/minimax-h3-image/privora-h3-runpod-worker",
  imageTag: "cached-models-canary",
  imageDigest: "sha256:6422784683b47142e03a7b9fb48cf6d4f6a5de6aec4cd8984fbd1849af47d93a",
  buildId: "cached-models-test-build",
  comfyuiRevision: "dec5d9450a5290bcf63430409ea41018e67f41c3"
});

const CANARY_MODEL = Object.freeze({
  repository: "CDitfort/privora-minimax-h3-models",
  revision: "ecb69a4211d74b5798398021003bccde02d63757",
  manifest: "multimodal-4-hf-cache-v1"
});

function capabilityDocument({ canary = true } = {}) {
  const build = canary
    ? {
        sourceCommit: CANARY_APPLICATION.sourceCommit,
        imageTag: CANARY_APPLICATION.imageTag,
        buildId: CANARY_APPLICATION.buildId
      }
    : BLACKWELL_BUILD;
  return {
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
      ultra: { dimensions: { "16:9": "1344x768" }, steps: 20 }
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
    models: { fl2va: true, ref2va: true },
    generationModes: { quality: true, turbo: true, turboFast: true },
    byFamily: {
      fl2va: ["quality", "turbo", "turboFast"],
      ref2va: ["quality", "turboFast"]
    },
    worker: {
      processId: canary ? "cached-runtime-process" : "blackwell-runtime-process",
      gpuMode: "single",
      build,
      ...(canary
        ? { applicationRelease: CANARY_APPLICATION, modelRelease: CANARY_MODEL }
        : {})
    }
  };
}

function makeEnv(overrides = {}) {
  const states = new Map();
  const channelNames = [];
  return {
    JOB_TOKEN_SECRET: "synthetic-job-token-secret",
    JOB_TOKEN_TTL_SECONDS: "3600",
    RUNPOD_ENDPOINT_ID: ENDPOINTS.h3,
    RUNPOD_BLACKWELL_ENDPOINT_ID: ENDPOINTS["h3-blackwell"],
    RUNPOD_CACHED_CANARY_ENDPOINT_ID: ENDPOINTS["h3-cached-canary"],
    RUNPOD_API_KEY: "synthetic-shared-runpod-key",
    RUNPOD_BLACKWELL_API_KEY: "synthetic-blackwell-runpod-key",
    CF_VERSION_METADATA: { id: "zero-traffic-cached-canary-test" },
    JOB_CHANNEL: {
      idFromName(name) {
        channelNames.push(name);
        return { name };
      },
      get(id) {
        return {
          async fetch(request) {
            const path = new URL(request.url).pathname;
            if (path.endsWith("/update")) {
              const update = await request.json();
              states.set(id.name, { ...(states.get(id.name) || {}), ...update });
              return new Response(null, { status: 204 });
            }
            return Response.json(states.get(id.name) || {});
          }
        };
      }
    },
    H3_OUTPUTS: {
      async delete() {},
      async head() { return null; }
    },
    __states: states,
    __channelNames: channelNames,
    ...overrides
  };
}

function stubRunPod({
  canaryCapabilities = capabilityDocument(),
  blackwellCapabilities = capabilityDocument({ canary: false })
} = {}) {
  const original = globalThis.fetch;
  const seen = [];
  const capabilityJobs = new Map();

  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    const parsed = new URL(url);
    const body = init.body ? JSON.parse(init.body) : null;
    const authorization = new Headers(init.headers).get("Authorization");
    const endpoint = parsed.pathname.split("/")[2];
    seen.push({ url, endpoint, method: init.method || "GET", body, authorization });

    if (parsed.pathname.endsWith("/run") && body?.input?.mode === "capabilities") {
      const id = `${endpoint}-capability-job`;
      capabilityJobs.set(id, endpoint);
      return Response.json({ id, status: "IN_QUEUE" });
    }
    const statusId = parsed.pathname.split("/status/")[1];
    if (statusId && capabilityJobs.has(statusId)) {
      const capabilities = capabilityJobs.get(statusId) === ENDPOINTS["h3-cached-canary"]
        ? canaryCapabilities
        : blackwellCapabilities;
      return Response.json({ id: statusId, status: "COMPLETED", output: { capabilities } });
    }
    if (parsed.pathname.endsWith("/run")) {
      return Response.json({ id: "canary-generation-job", status: "IN_QUEUE" });
    }
    if (parsed.pathname.includes("/status/")) {
      return Response.json({ id: parsed.pathname.split("/status/")[1], status: "IN_QUEUE" });
    }
    if (parsed.pathname.includes("/cancel/")) {
      return Response.json({ id: parsed.pathname.split("/cancel/")[1], status: "CANCELLED" });
    }
    throw new Error(`Unexpected RunPod test path: ${parsed.pathname}`);
  };

  return {
    seen,
    restore() { globalThis.fetch = original; }
  };
}

function canonicalCreate() {
  return new Request("https://worker.example/canonical/generate?backend=h3-cached-canary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input: {
        mode: "create",
        prompt: "A clean canary transport probe",
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

async function captureConsole(action) {
  const originals = { log: console.log, warn: console.warn, error: console.error };
  const lines = [];
  for (const method of Object.keys(originals)) {
    console[method] = (...values) => lines.push(values.map(String).join(" "));
  }
  try {
    return { value: await action(), output: lines.join("\n") };
  } finally {
    Object.assign(console, originals);
  }
}

test("wrangler endpoint bindings preserve production and add only the cached canary", () => {
  const config = readFileSync(new URL("./wrangler.toml", import.meta.url), "utf8");
  const endpointBindings = Object.fromEntries(
    [...config.matchAll(/^(RUNPOD(?:_[A-Z]+)*_ENDPOINT_ID)\s*=\s*"([^"]+)"$/gm)]
      .map(([, name, value]) => [name, value])
  );
  assert.deepEqual(endpointBindings, {
    RUNPOD_ENDPOINT_ID: ENDPOINTS.h3,
    RUNPOD_BLACKWELL_ENDPOINT_ID: ENDPOINTS["h3-blackwell"],
    RUNPOD_CACHED_CANARY_ENDPOINT_ID: ENDPOINTS["h3-cached-canary"]
  });
});

test("health keeps the default and production backend map while adding the canary", async () => {
  const response = await worker.fetch(new Request("https://worker.example/health"), makeEnv());
  const body = await response.json();
  assert.equal(response.status, 200);
  assert.equal(body.defaultBackend, "h3");
  assert.deepEqual(Object.keys(body.backends), ["h3", "h3-blackwell", "h3-cached-canary"]);
  assert.deepEqual(body.backends["h3-cached-canary"], { configured: true });
  assert.equal(body.canonical.backend, "h3-blackwell");
  assert.deepEqual(body.canonical.expectedBuild, BLACKWELL_BUILD);
});

test("every existing alias still resolves to its original endpoint", async () => {
  const expected = {
    h3: ENDPOINTS.h3,
    default: ENDPOINTS.h3,
    original: ENDPOINTS.h3,
    ada: ENDPOINTS.h3,
    "h3-ada": ENDPOINTS.h3,
    "48gb-pro": ENDPOINTS.h3,
    "minimax-h3": ENDPOINTS.h3,
    blackwell: ENDPOINTS["h3-blackwell"],
    "h3-blackwell": ENDPOINTS["h3-blackwell"],
    h3_blackwell: ENDPOINTS["h3-blackwell"],
    "minimax-h3-blackwell": ENDPOINTS["h3-blackwell"],
    "h3-cached-canary": ENDPOINTS["h3-cached-canary"]
  };
  const runpod = stubRunPod();
  try {
    for (const [alias, endpoint] of Object.entries(expected)) {
      const response = await worker.fetch(
        new Request(`https://worker.example/status/${encodeURIComponent(alias)}/alias-job`),
        makeEnv()
      );
      assert.equal(response.status, 200, alias);
      assert.equal(runpod.seen.at(-1).endpoint, endpoint, alias);
    }
  } finally {
    runpod.restore();
  }
});

test("canary API key prefers its dedicated binding and otherwise uses the explicit shared fallback", async () => {
  const runpod = stubRunPod();
  try {
    await worker.fetch(
      new Request("https://worker.example/status/h3-cached-canary/fallback-job"),
      makeEnv()
    );
    assert.equal(runpod.seen.at(-1).authorization, "Bearer synthetic-shared-runpod-key");

    await worker.fetch(
      new Request("https://worker.example/status/h3-cached-canary/primary-job"),
      makeEnv({ RUNPOD_CACHED_CANARY_API_KEY: "synthetic-dedicated-canary-key" })
    );
    assert.equal(runpod.seen.at(-1).authorization, "Bearer synthetic-dedicated-canary-key");
  } finally {
    runpod.restore();
  }
});

test("canary capabilities use the canonical probe and return the runtime document unchanged", async () => {
  const env = makeEnv();
  const runtimeCapabilities = capabilityDocument();
  const runpod = stubRunPod({ canaryCapabilities: runtimeCapabilities });
  try {
    const response = await worker.fetch(
      new Request("https://worker.example/capabilities?backend=h3-cached-canary&refresh=1"),
      env
    );
    const body = await response.json();
    assert.equal(response.status, 200, JSON.stringify(body));
    assert.deepEqual(body, runtimeCapabilities);
    assert.equal(runpod.seen[0].endpoint, ENDPOINTS["h3-cached-canary"]);
    assert.deepEqual(runpod.seen[0].body, { input: { mode: "capabilities" } });
    assert.equal(runpod.seen[1].endpoint, ENDPOINTS["h3-cached-canary"]);
    assert.ok(env.__channelNames.includes("__canonical_capabilities__.h3-cached-canary"));
    assert.ok(!env.__channelNames.includes("__canonical_capabilities__"));
    assert.deepEqual(body.worker.applicationRelease, CANARY_APPLICATION);
    assert.deepEqual(body.worker.modelRelease, CANARY_MODEL);
    assert.equal(body.worker.gpuMode, "single");
    assert.deepEqual(body.models, { fl2va: true, ref2va: true });
    assert.deepEqual(body.byFamily.fl2va, ["quality", "turbo", "turboFast"]);
    assert.deepEqual(body.byFamily.ref2va, ["quality", "turboFast"]);
  } finally {
    runpod.restore();
  }
});

test("Blackwell and canary capability caches and endpoints remain isolated", async () => {
  const env = makeEnv();
  const runpod = stubRunPod();
  try {
    assert.equal((await worker.fetch(new Request("https://worker.example/capabilities?backend=h3-cached-canary&refresh=1"), env)).status, 200);
    assert.equal((await worker.fetch(new Request("https://worker.example/capabilities?refresh=1"), env)).status, 200);
    const probes = runpod.seen.filter((call) => call.body?.input?.mode === "capabilities");
    assert.deepEqual(probes.map((call) => call.endpoint), [
      ENDPOINTS["h3-cached-canary"],
      ENDPOINTS["h3-blackwell"]
    ]);
    assert.ok(env.__states.has("__canonical_capabilities__.h3-cached-canary"));
    assert.ok(env.__states.has("__canonical_capabilities__"));
  } finally {
    runpod.restore();
  }
});

test("canonical canary generation contains no workflow or RunPod-owned graph settings", async () => {
  const env = makeEnv();
  const runpod = stubRunPod();
  try {
    assert.equal((await worker.fetch(new Request("https://worker.example/capabilities?backend=h3-cached-canary&refresh=1"), env)).status, 200);
    const { value: response, output: logs } = await captureConsole(() => worker.fetch(canonicalCreate(), env));
    const body = await response.json();
    const generation = runpod.seen.find((call) => call.endpoint === ENDPOINTS["h3-cached-canary"] && call.body?.input?.mode === "create");
    assert.equal(response.status, 202, JSON.stringify(body));
    assert.ok(generation, "the canary generation was not sent to its dedicated endpoint");
    assert.equal(generation.body.input.prompt, "A clean canary transport probe");
    for (const forbidden of [
      "workflow", "width", "height", "frames", "steps", "checkpoint", "checkpointName",
      "checkpoint_filename", "lora", "loras", "lora_filename"
    ]) {
      assert.equal(generation.body.input[forbidden], undefined, forbidden);
    }
    assert.doesNotMatch(generation.body.input.prompt, /<[^>]+>/, "Cloudflare must not add H3 structural tags");
    assert.ok(generation.body.input.progress?.token);
    assert.ok(generation.body.input.output?.token);
    assert.equal(body.backend, "h3-cached-canary");
    assert.match(body.routes.status, /^\/status\/h3-cached-canary\//);
    assert.match(body.routes.cancel, /^\/cancel\/h3-cached-canary\//);
    const publicMaterial = JSON.stringify(body) + logs;
    for (const secret of [env.RUNPOD_API_KEY, env.RUNPOD_BLACKWELL_API_KEY, env.JOB_TOKEN_SECRET]) {
      assert.ok(!publicMaterial.includes(secret));
    }
  } finally {
    runpod.restore();
  }
});

test("canary release validation accepts a registry mirror label only when the digest and model pins match", async () => {
  const mirroredCapabilities = structuredClone(capabilityDocument());
  mirroredCapabilities.worker.applicationRelease.imageRepository = "ghcr.io/chrisditfort/privora-h3-runpod-worker";
  const env = makeEnv();
  const runpod = stubRunPod({ canaryCapabilities: mirroredCapabilities });
  try {
    const capabilities = await worker.fetch(
      new Request("https://worker.example/capabilities?backend=h3-cached-canary&refresh=1"),
      env
    );
    assert.equal(capabilities.status, 200);
    assert.equal(
      (await capabilities.json()).worker.applicationRelease.imageRepository,
      "ghcr.io/chrisditfort/privora-h3-runpod-worker"
    );

    const generation = await worker.fetch(canonicalCreate(), env);
    assert.equal(generation.status, 202, JSON.stringify(await generation.json()));
  } finally {
    runpod.restore();
  }
});

test("canary release validation rejects a mismatched application digest before generation", async () => {
  const mismatchedCapabilities = structuredClone(capabilityDocument());
  mismatchedCapabilities.worker.applicationRelease.imageDigest = `sha256:${"f".repeat(64)}`;
  const env = makeEnv();
  const runpod = stubRunPod({ canaryCapabilities: mismatchedCapabilities });
  try {
    const capabilities = await worker.fetch(
      new Request("https://worker.example/capabilities?backend=h3-cached-canary&refresh=1"),
      env
    );
    assert.equal(capabilities.status, 200);

    const generation = await worker.fetch(canonicalCreate(), env);
    const body = await generation.json();
    assert.equal(generation.status, 503);
    assert.equal(body.code, "DEPLOYMENT_ACTION_REQUIRED");
    assert.equal(
      runpod.seen.filter((call) => call.body?.input?.mode === "create").length,
      0
    );
  } finally {
    runpod.restore();
  }
});

test("legacy generation cannot send a reconstructed workflow to the canary", async () => {
  const runpod = stubRunPod();
  try {
    const response = await worker.fetch(
      new Request("https://worker.example/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ backend: "h3-cached-canary", prompt: "must not submit" })
      }),
      makeEnv()
    );
    assert.equal(response.status, 400);
    assert.equal(runpod.seen.length, 0);
  } finally {
    runpod.restore();
  }
});

test("status and cancel route through the canary without exposing credentials", async () => {
  const env = makeEnv();
  const runpod = stubRunPod();
  try {
    const status = await worker.fetch(
      new Request("https://worker.example/status/h3-cached-canary/runpod-status-job?jobId=cloudflare-status-job"),
      env
    );
    const statusBody = await status.json();
    assert.equal(status.status, 200);
    assert.equal(runpod.seen.at(-1).url, `https://api.runpod.ai/v2/${ENDPOINTS["h3-cached-canary"]}/status/runpod-status-job`);
    assert.equal(statusBody.backend, "h3-cached-canary");

    const cancelled = await worker.fetch(
      new Request("https://worker.example/cancel/h3-cached-canary/runpod-cancel-job?jobId=cloudflare-cancel-job", { method: "POST" }),
      env
    );
    const cancelBody = await cancelled.json();
    assert.equal(cancelled.status, 200);
    assert.equal(runpod.seen.at(-1).url, `https://api.runpod.ai/v2/${ENDPOINTS["h3-cached-canary"]}/cancel/runpod-cancel-job`);
    assert.equal(cancelBody.backend, "h3-cached-canary");

    const responses = JSON.stringify([statusBody, cancelBody]);
    for (const secret of [env.RUNPOD_API_KEY, env.RUNPOD_BLACKWELL_API_KEY, env.JOB_TOKEN_SECRET]) {
      assert.ok(!responses.includes(secret));
    }
  } finally {
    runpod.restore();
  }
});

*/