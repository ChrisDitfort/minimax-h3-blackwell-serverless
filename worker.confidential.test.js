/*
 * Confidential Generation, driven through the Worker's real fetch handler.
 *
 * Run with:  node --test
 *
 * Three properties are worth more than the rest, and each has its own section below:
 *
 *   1. A confidential request either works or is refused. It is never quietly downgraded
 *      to storing a plaintext MP4.
 *   2. The encryption key reaches RunPod and nothing else. Not R2, not the Durable Object,
 *      not a status response, not a log, not object metadata.
 *   3. Standard generation is exactly what it was.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import worker, {
  normalizePrivacy,
  outputKey,
  describeArtifactForStatus,
  parseContainerPrefix,
  signJobToken,
  workerConstants,
  HttpError
} from "./worker.js";

const JOB = "3f1b6a5c-9d4e-4b2a-8c7f-0e1d2c3b4a59";
const KEY_BYTES = new Uint8Array(32).map((_, i) => (i * 7 + 3) & 0xff);
const KEY = toBase64Url(KEY_BYTES);
const SALT = toBase64Url(new Uint8Array(16).fill(9));

const VECTOR = JSON.parse(readFileSync("./tests/vectors/confidential_container.json", "utf8"));

function toBase64Url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

const BASE_REQUEST = {
  backend: "h3-blackwell",
  prompt: "A cinematic ocean scene",
  quality: "standard",
  aspect_ratio: "16:9",
  duration: 5
};

const CONFIDENTIAL = {
  ...BASE_REQUEST,
  privacyMode: "confidential",
  encryption: {
    algorithm: "AES-256-GCM",
    key: KEY,
    keyId: "customer-key-1",
    kdf: { name: "argon2id", salt: SALT, parameters: { memorySize: 65536, iterations: 3 } }
  }
};

/** Env with stub R2 and a Durable Object that actually keeps state. */
function makeEnv(overrides = {}) {
  const bucket = new Map();
  const puts = [];
  let state = {};

  return {
    JOB_TOKEN_SECRET: "test-secret",
    JOB_TOKEN_TTL_SECONDS: "3600",
    RUNPOD_ENDPOINT_ID: "e-ada",
    RUNPOD_BLACKWELL_ENDPOINT_ID: "e-bw",
    RUNPOD_API_KEY: "runpod-key",

    JOB_CHANNEL: {
      idFromName: (name) => ({ name }),
      get: () => ({
        fetch: async (request) => {
          const path = new URL(request.url).pathname;
          if (path.endsWith("/update")) {
            state = { ...state, ...(await request.json()) };
            return new Response(null, { status: 204 });
          }
          return new Response(JSON.stringify(state), {
            headers: { "Content-Type": "application/json" }
          });
        }
      })
    },

    H3_OUTPUTS: {
      put: async (key, body, options) => {
        const bytes = body instanceof ReadableStream ? await drain(body) : new Uint8Array(body);
        puts.push({ key, options, bytes });
        bucket.set(key, { bytes, options });
        return { size: bytes.byteLength };
      },
      get: async (key) => {
        if (!bucket.has(key)) return null;
        const { bytes, options } = bucket.get(key);
        return {
          body: bytes,
          size: bytes.byteLength,
          httpEtag: '"etag"',
          customMetadata: options?.customMetadata || {},
          writeHttpMetadata(headers) {
            if (options?.httpMetadata?.contentType) {
              headers.set("Content-Type", options.httpMetadata.contentType);
            }
          }
        };
      },
      delete: async (key) => {
        for (const one of Array.isArray(key) ? key : [key]) bucket.delete(one);
      },
      list: async ({ prefix = "" } = {}) => ({
        objects: [...bucket.keys()].filter((k) => k.startsWith(prefix)).map((key) => ({ key })),
        truncated: false
      })
    },

    __puts: puts,
    __bucket: bucket,
    __state: () => state,
    ...overrides
  };
}

async function drain(stream) {
  const reader = stream.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.byteLength;
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

/** Replace global fetch so no request leaves the test, and record what RunPod was sent. */
function stubRunPod(response = { id: "runpod-1", status: "IN_QUEUE" }) {
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
    get submittedInput() {
      return JSON.parse(seen[0].init.body).input;
    },
    restore() {
      globalThis.fetch = original;
    }
  };
}

/** Capture console output for the duration of one call. */
async function captureConsole(run) {
  const lines = [];
  const original = { log: console.log, warn: console.warn, error: console.error };
  for (const level of ["log", "warn", "error"]) {
    console[level] = (...args) => lines.push(args.map(String).join(" "));
  }
  try {
    return { result: await run(), output: lines.join("\n") };
  } finally {
    Object.assign(console, original);
  }
}

function post(path, body) {
  return new Request(`https://worker.example${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

/* ====================================================================================
 * Request validation
 * ================================================================================== */

test("a valid 256-bit key is accepted", () => {
  const privacy = normalizePrivacy(CONFIDENTIAL);
  assert.equal(privacy.mode, "confidential");
  assert.equal(privacy.encryption.key, KEY);
  assert.equal(privacy.encryption.algorithm, "AES-256-GCM");
  assert.equal(privacy.encryption.kdf.name, "argon2id");
});

test("no privacyMode means standard, exactly as before", () => {
  const privacy = normalizePrivacy(BASE_REQUEST);
  assert.equal(privacy.mode, "standard");
  assert.equal(privacy.encryption, null);
  assert.equal(privacy.expiresAt, null);
});

test("privacy_mode is accepted alongside privacyMode", () => {
  assert.equal(normalizePrivacy({ privacy_mode: "confidential", encryption: { key: KEY } }).mode,
    "confidential");
});

test("a wrong-size key is rejected with its length, never its contents", () => {
  for (const size of [16, 24, 31, 33, 64]) {
    const key = toBase64Url(new Uint8Array(size).fill(1));
    try {
      normalizePrivacy({ ...CONFIDENTIAL, encryption: { key } });
      assert.fail(`expected a rejection for ${size} bytes`);
    } catch (error) {
      assert.equal(error.status, 400);
      assert.match(error.message, new RegExp(`got ${size}`));
      assert.ok(!error.message.includes(key), "the rejected key must not be echoed back");
    }
  }
});

test("malformed base64 is rejected", () => {
  for (const key of ["not base64!", "***", "a b c", "%%%%"]) {
    assert.throws(
      () => normalizePrivacy({ ...CONFIDENTIAL, encryption: { key } }),
      (error) => error.status === 400 && /base64url/.test(error.message)
    );
  }
});

test("a missing encryption block in confidential mode is rejected", () => {
  assert.throws(
    () => normalizePrivacy({ ...BASE_REQUEST, privacyMode: "confidential" }),
    (error) => error.status === 400 && /requires an encryption block/.test(error.message)
  );
});

test("a missing key in confidential mode is rejected", () => {
  assert.throws(
    () => normalizePrivacy({ ...BASE_REQUEST, privacyMode: "confidential", encryption: {} }),
    (error) => error.status === 400 && /encryption\.key is required/.test(error.message)
  );
});

test("an unsupported algorithm is rejected", () => {
  assert.throws(
    () =>
      normalizePrivacy({
        ...BASE_REQUEST,
        privacyMode: "confidential",
        encryption: { algorithm: "AES-128-CBC", key: KEY }
      }),
    (error) => error.status === 400 && /Unsupported encryption algorithm/.test(error.message)
  );
});

test("an unknown privacy mode is a 400 naming the supported ones", () => {
  assert.throws(
    () => normalizePrivacy({ ...BASE_REQUEST, privacyMode: "super-secret" }),
    (error) =>
      error.status === 400 &&
      /Unknown privacyMode/.test(error.message) &&
      /confidential/.test(error.message)
  );
});

test("declared-but-unimplemented modes are a 501 with a reason, not a silent downgrade", () => {
  for (const mode of ["private", "ephemeral"]) {
    assert.throws(
      () => normalizePrivacy({ ...BASE_REQUEST, privacyMode: mode }),
      (error) => error.status === 501 && error.message.length > 60,
      `expected an explained 501 for ${mode}`
    );
  }
});

test("standard mode with an encryption block is rejected rather than ignored", () => {
  assert.throws(
    () => normalizePrivacy({ ...BASE_REQUEST, privacyMode: "standard", encryption: { key: KEY } }),
    (error) => error.status === 400 && /does not encrypt/.test(error.message)
  );
});

test("keyId must not be the key", () => {
  assert.throws(
    () => normalizePrivacy({ ...CONFIDENTIAL, encryption: { key: KEY, keyId: KEY } }),
    (error) => error.status === 400 && /must not be the key itself/.test(error.message)
  );
});

test("keyId is length- and charset-limited", () => {
  for (const keyId of ["x".repeat(129), "has spaces", "quote\"inside"]) {
    assert.throws(
      () => normalizePrivacy({ ...CONFIDENTIAL, encryption: { key: KEY, keyId } }),
      (error) => error.status === 400
    );
  }
});

test("kdf metadata is validated", () => {
  const cases = [
    [{ name: "md5", salt: SALT }, /Unsupported encryption\.kdf\.name/],
    [{ name: "argon2id" }, /salt is required/],
    [{ name: "argon2id", salt: "!!!" }, /must be base64url/],
    [{ name: "argon2id", salt: toBase64Url(new Uint8Array(4)) }, /8-64 bytes/],
    [{ name: "argon2id", salt: SALT, parameters: [] }, /parameters must be an object/],
    [{ name: "argon2id", salt: SALT, parameters: { m: { nested: 1 } } }, /number, string or boolean/]
  ];
  for (const [kdf, pattern] of cases) {
    assert.throws(
      () => normalizePrivacy({ ...CONFIDENTIAL, encryption: { key: KEY, kdf } }),
      (error) => error.status === 400 && pattern.test(error.message),
      `expected ${pattern} for ${JSON.stringify(kdf)}`
    );
  }
});

test("a well-formed kdf survives validation unchanged", () => {
  const privacy = normalizePrivacy(CONFIDENTIAL);
  assert.deepEqual(privacy.encryption.kdf, {
    name: "argon2id",
    salt: SALT,
    parameters: { memorySize: 65536, iterations: 3 }
  });
});

test("retention is validated and resolved to an absolute expiry", () => {
  const privacy = normalizePrivacy({ ...BASE_REQUEST, retentionSeconds: 3600 });
  const delta = (new Date(privacy.expiresAt).getTime() - Date.now()) / 1000;
  assert.ok(Math.abs(delta - 3600) < 5, `expected ~3600s, got ${delta}`);

  for (const body of [
    { retentionSeconds: 5 },
    { retentionSeconds: 100 * 24 * 3600 },
    { expiresAt: "not-a-date" },
    { retentionSeconds: 3600, expiresAt: new Date().toISOString() }
  ]) {
    assert.throws(
      () => normalizePrivacy({ ...BASE_REQUEST, ...body }),
      (error) => error.status === 400
    );
  }
});

/* ====================================================================================
 * Submission: where the key goes, and where it does not
 * ================================================================================== */

test("POST /generate forwards the key to RunPod and nowhere else", async () => {
  const runpod = stubRunPod();
  const env = makeEnv();

  try {
    const { result: response, output } = await captureConsole(() =>
      worker.fetch(post("/generate", CONFIDENTIAL), env)
    );
    const body = await response.json();

    assert.equal(response.status, 202);

    // To RunPod: yes. That is the only place the plaintext exists, so it is the only
    // place that can encrypt.
    const input = runpod.submittedInput;
    assert.equal(input.privacy.mode, "confidential");
    assert.equal(input.encryption.key, KEY);
    assert.equal(input.encryption.algorithm, "AES-256-GCM");
    assert.equal(input.encryption.keyId, "customer-key-1");

    // To the client: never.
    const serialized = JSON.stringify(body);
    assert.ok(!serialized.includes(KEY), "the response must not contain the key");
    assert.equal(body.privacyMode, "confidential");
    assert.equal(body.encryption.algorithm, "AES-256-GCM");
    assert.equal(body.encryption.keyId, "customer-key-1");
    assert.equal(body.encryption.key, undefined);

    // To the Durable Object: never.
    assert.ok(!JSON.stringify(env.__state()).includes(KEY), "the job channel must not hold the key");

    // To the log: never.
    assert.ok(!output.includes(KEY), "the key must not be logged");
    assert.match(output, /privacy_mode=confidential/);
    assert.match(output, /encryption=AES-256-GCM/);
  } finally {
    runpod.restore();
  }
});

test("the generation log names the prompt's length and digest, not the prompt", async () => {
  const runpod = stubRunPod();
  try {
    const { output } = await captureConsole(() =>
      worker.fetch(post("/generate", CONFIDENTIAL), makeEnv())
    );
    assert.ok(!output.includes("cinematic ocean"), "the prompt must not be logged");
    assert.match(output, /prompt_chars=23/);
    assert.match(output, /prompt_sha256=[0-9a-f]{16}/);
  } finally {
    runpod.restore();
  }
});

test("confidential mode is refused when there is no authenticated upload path", async () => {
  const runpod = stubRunPod();
  const env = makeEnv({ JOB_TOKEN_SECRET: undefined });

  try {
    const response = await worker.fetch(post("/generate", CONFIDENTIAL), env);
    const body = await response.json();

    assert.equal(response.status, 503, "fail closed rather than store plaintext");
    assert.match(body.error, /Confidential generation is unavailable/);
    assert.equal(runpod.seen.length, 0, "no job may be submitted");
  } finally {
    runpod.restore();
  }
});

test("standard generation still works when JOB_TOKEN_SECRET is missing", async () => {
  const runpod = stubRunPod();
  try {
    const response = await worker.fetch(
      post("/generate", BASE_REQUEST),
      makeEnv({ JOB_TOKEN_SECRET: undefined })
    );
    assert.equal(response.status, 202, "standard mode must not be affected by the new check");
  } finally {
    runpod.restore();
  }
});

test("a standard submission carries no encryption block at all", async () => {
  const runpod = stubRunPod();
  try {
    await worker.fetch(post("/generate", BASE_REQUEST), makeEnv());
    const input = runpod.submittedInput;
    assert.equal(input.encryption, undefined);
    assert.equal(input.privacy.mode, "standard");
    assert.ok(input.workflow, "the workflow is unchanged");
  } finally {
    runpod.restore();
  }
});

/* ====================================================================================
 * Upload: the fail-closed gate
 * ================================================================================== */

/** A container the Worker should accept for `jobId`. */
function containerFor(jobId, { version = 1, suite = 1, magic = "CGEN" } = {}) {
  const header = new TextEncoder().encode(
    JSON.stringify({
      v: 1,
      alg: "AES-256-GCM",
      artifactId: jobId,
      contentType: "video/mp4",
      plaintextBytes: 64,
      createdAt: "2026-01-01T00:00:00Z",
      kdf: { name: "argon2id", salt: SALT, parameters: { memorySize: 65536 } },
      keyId: "customer-key-1"
    })
  );

  const body = new Uint8Array(8 + header.length + 12 + 64 + 16);
  body.set(new TextEncoder().encode(magic), 0);
  body[4] = version;
  body[5] = suite;
  body[6] = (header.length >> 8) & 0xff;
  body[7] = header.length & 0xff;
  body.set(header, 8);
  // Nonce, payload and tag are opaque here: the Worker never decrypts, it only frames.
  body.fill(0xab, 8 + header.length);
  return body;
}

async function upload(env, jobId, body, { mode = "confidential", contentType } = {}) {
  const token = await signJobToken(
    env.JOB_TOKEN_SECRET,
    jobId,
    workerConstants().TOKEN_PURPOSES.output,
    3600,
    { pm: mode }
  );
  return worker.fetch(
    new Request(`https://worker.example/internal/jobs/${jobId}/output`, {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": contentType || (mode === "confidential" ? "application/octet-stream" : "video/mp4")
      },
      body,
      duplex: "half"
    }),
    env
  );
}

test("a confidential upload of real ciphertext is stored under the .enc key", async () => {
  const env = makeEnv();
  const response = await upload(env, JOB, containerFor(JOB));
  const body = await response.json();

  assert.equal(response.status, 201);
  assert.equal(body.key, outputKey(JOB, true));
  assert.equal(body.key, `outputs/${JOB}/artifact.enc`);
  assert.equal(body.encrypted, true);
  assert.equal(body.contentType, "application/octet-stream");

  const put = env.__puts[0];
  assert.equal(put.options.httpMetadata.contentType, "application/octet-stream");
  assert.equal(put.options.customMetadata.encrypted, "true");
  assert.equal(put.options.customMetadata.privacyMode, "confidential");
  assert.equal(put.options.customMetadata.algorithm, "AES-256-GCM");
  assert.equal(put.options.customMetadata.originalContentType, "video/mp4");
});

test("REFUSED: a plaintext MP4 offered for a confidential job", async () => {
  const env = makeEnv();
  const mp4 = new Uint8Array([0, 0, 0, 0x18, 0x66, 0x74, 0x79, 0x70, 0x6d, 0x70, 0x34, 0x32]);

  const response = await upload(env, JOB, mp4);
  const body = await response.json();

  assert.equal(response.status, 422);
  assert.match(body.error, /not an encrypted container/);
  assert.equal(env.__puts.length, 0, "nothing may be written to R2");
  assert.equal(env.__bucket.size, 0);
});

test("REFUSED: ciphertext whose authenticated header names a different generation", async () => {
  const env = makeEnv();
  const other = "99999999-8888-7777-6666-555555555555";

  const response = await upload(env, JOB, containerFor(other));
  const body = await response.json();

  assert.equal(response.status, 422);
  assert.match(body.error, /names generation/);
  assert.equal(env.__puts.length, 0);
});

test("REFUSED: a container from a future format version or unknown suite", async () => {
  for (const options of [{ version: 2 }, { suite: 9 }, { magic: "XXXX" }]) {
    const env = makeEnv();
    const response = await upload(env, JOB, containerFor(JOB, options));
    assert.equal(response.status, 422, JSON.stringify(options));
    assert.equal(env.__puts.length, 0);
  }
});

test("the uploader cannot choose the privacy mode - only the signed token can", async () => {
  const env = makeEnv();

  // A token minted for a confidential job. The uploader labels the body video/mp4 and
  // sends plaintext anyway; the Content-Type is not what is trusted.
  const response = await upload(env, JOB, new Uint8Array(64).fill(1), {
    mode: "confidential",
    contentType: "video/mp4"
  });

  assert.equal(response.status, 422);
  assert.equal(env.__puts.length, 0);
});

test("a token with no privacy claim is treated as standard, so old jobs keep working", async () => {
  const env = makeEnv();
  const token = await signJobToken(
    env.JOB_TOKEN_SECRET,
    JOB,
    workerConstants().TOKEN_PURPOSES.output,
    3600
  );

  const response = await worker.fetch(
    new Request(`https://worker.example/internal/jobs/${JOB}/output`, {
      method: "PUT",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "video/mp4" },
      body: new Uint8Array(32).fill(7),
      duplex: "half"
    }),
    env
  );

  assert.equal(response.status, 201);
  assert.equal((await response.json()).key, outputKey(JOB));
});

test("a standard upload is stored byte for byte, exactly as before", async () => {
  const env = makeEnv();
  const payload = new Uint8Array(256).map((_, i) => i & 0xff);

  const response = await upload(env, JOB, payload, { mode: "standard" });
  assert.equal(response.status, 201);

  const put = env.__puts[0];
  assert.equal(put.key, outputKey(JOB));
  assert.equal(put.options.httpMetadata.contentType, "video/mp4");
  assert.deepEqual([...put.bytes], [...payload], "the body must not be altered");
});

test("the whole ciphertext survives the header peek", async () => {
  const env = makeEnv();
  const container = containerFor(JOB);

  await upload(env, JOB, container);

  assert.deepEqual(
    [...env.__puts[0].bytes],
    [...container],
    "peeking at the header must not consume or reorder the body"
  );
});

test("R2 metadata never contains key material", async () => {
  const env = makeEnv();
  await upload(env, JOB, containerFor(JOB));
  const serialized = JSON.stringify(env.__puts[0].options);
  assert.ok(!serialized.includes(KEY));
  assert.ok(!/\bkey\b\s*:\s*"[A-Za-z0-9_-]{40,}"/.test(serialized));
});

test("the object path contains no prompt and no user content", async () => {
  const env = makeEnv();
  await upload(env, JOB, containerFor(JOB));
  assert.equal(env.__puts[0].key, `outputs/${JOB}/artifact.enc`);
  assert.ok(!env.__puts[0].key.includes("ocean"));
});

/* ====================================================================================
 * Retrieval
 * ================================================================================== */

test("a confidential artefact is served as ciphertext, labelled and uncacheable", async () => {
  const env = makeEnv();
  await upload(env, JOB, containerFor(JOB));

  const response = await worker.fetch(
    new Request(`https://worker.example/jobs/${JOB}/artifact`),
    env
  );

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Content-Type"), "application/octet-stream");
  assert.equal(response.headers.get("X-Artifact-Encrypted"), "true");
  assert.equal(response.headers.get("X-Privacy-Mode"), "confidential");
  assert.equal(response.headers.get("X-Artifact-Algorithm"), "AES-256-GCM");
  assert.equal(response.headers.get("X-Artifact-Original-Content-Type"), "video/mp4");
  assert.equal(response.headers.get("Cache-Control"), "private, no-store");
  assert.equal(response.headers.get("X-Content-Type-Options"), "nosniff");
  assert.match(response.headers.get("Content-Disposition"), /attachment/);
  assert.match(response.headers.get("Access-Control-Expose-Headers"), /X-Artifact-Algorithm/);

  const bytes = new Uint8Array(await response.arrayBuffer());
  assert.deepEqual([...bytes.subarray(0, 4)], [...new TextEncoder().encode("CGEN")]);
});

test("/video and /artifact are the same route", async () => {
  const env = makeEnv();
  await upload(env, JOB, containerFor(JOB));

  const viaVideo = await worker.fetch(new Request(`https://worker.example/jobs/${JOB}/video`), env);
  const viaArtifact = await worker.fetch(
    new Request(`https://worker.example/jobs/${JOB}/artifact`),
    env
  );

  assert.equal(viaVideo.status, viaArtifact.status);
  assert.equal(
    viaVideo.headers.get("Content-Type"),
    viaArtifact.headers.get("Content-Type"),
    "an old client using /video must not receive ciphertext labelled as video/mp4"
  );
});

test("a standard artefact still streams as a cacheable, seekable MP4", async () => {
  const env = makeEnv();
  await upload(env, JOB, new Uint8Array(1024).fill(3), { mode: "standard" });

  const response = await worker.fetch(new Request(`https://worker.example/jobs/${JOB}/video`), env);

  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Content-Type"), "video/mp4");
  assert.equal(response.headers.get("X-Artifact-Encrypted"), "false");
  assert.equal(response.headers.get("Cache-Control"), "private, max-age=3600, immutable");
  assert.equal(response.headers.get("Accept-Ranges"), "bytes");
});

test("a missing artefact is a clean 404 for both names", async () => {
  const env = makeEnv();
  for (const path of ["video", "artifact"]) {
    const response = await worker.fetch(
      new Request(`https://worker.example/jobs/${JOB}/${path}`),
      env
    );
    assert.equal(response.status, 404);
  }
});

/* ====================================================================================
 * Status
 * ================================================================================== */

test("status describes a confidential artefact fully and never returns the key", () => {
  const state = {
    jobId: JOB,
    privacyMode: "confidential",
    artifact: {
      privacyMode: "confidential",
      encrypted: true,
      encryptionVersion: 1,
      algorithm: "AES-256-GCM",
      contentType: "application/octet-stream",
      originalContentType: "video/mp4",
      kdf: { name: "argon2id", salt: SALT, parameters: { memorySize: 65536 } },
      keyId: "customer-key-1",
      key: outputKey(JOB, true),
      size: 2048,
      deleted: false
    }
  };

  const out = describeArtifactForStatus(JOB, state, { size: 2048 });

  assert.equal(out.privacyMode, "confidential");
  assert.equal(out.artifact.encrypted, true);
  assert.equal(out.artifact.encryptionVersion, 1);
  assert.equal(out.artifact.algorithm, "AES-256-GCM");
  assert.equal(out.artifact.contentType, "application/octet-stream");
  assert.equal(out.artifact.originalContentType, "video/mp4");
  assert.equal(out.artifact.kdf.name, "argon2id");
  assert.equal(out.artifact.keyId, "customer-key-1");

  // Enough to derive the key again from the passphrase; not enough to derive it otherwise.
  assert.ok(out.artifact.kdf.salt, "the salt must be returned - without it the key is lost");
  assert.ok(!JSON.stringify(out).includes(KEY));
  assert.equal(out.artifact.encryptionKey, undefined);
  assert.equal(out.encryption, undefined);
});

test("status keeps the legacy video block for existing clients", () => {
  const out = describeArtifactForStatus(JOB, { jobId: JOB }, { size: 99 });
  assert.equal(out.video.url, `/jobs/${JOB}/video`);
  assert.equal(out.video.deleted, false);
  assert.equal(out.privacyMode, "standard");
});

test("a job recorded before privacy modes existed reads as standard", () => {
  // Exactly the shape the Durable Object held before this release.
  const legacy = { jobId: JOB, phase: "completed", video: { key: outputKey(JOB), size: 2196233 } };
  const out = describeArtifactForStatus(JOB, legacy, { size: 2196233 });

  assert.equal(out.privacyMode, "standard");
  assert.equal(out.artifact.encrypted, false);
  assert.equal(out.artifact.contentType, "video/mp4");
  assert.equal(out.video.deleted, false);
});

test("a deleted artefact is reported as deleted, not as a URL that would 404", () => {
  const out = describeArtifactForStatus(
    JOB,
    { jobId: JOB, privacyMode: "confidential", video: { deleted: true }, artifact: { deleted: true } },
    { size: 10 }
  );
  assert.deepEqual(out.video, { deleted: true });
  assert.equal(out.artifact.deleted, true);
  assert.equal(out.artifact.url, undefined);
});

test("a job with no artefact yet still reports its privacy mode", () => {
  const out = describeArtifactForStatus(JOB, { jobId: JOB, privacyMode: "confidential" }, undefined);
  assert.equal(out.privacyMode, "confidential");
  assert.equal(out.artifact, undefined);
  assert.equal(out.video, undefined);
});

test("an expiry recorded at submission is reported", () => {
  const expiresAt = new Date(Date.now() + 3600_000).toISOString();
  const out = describeArtifactForStatus(JOB, { jobId: JOB, expiresAt }, undefined);
  assert.equal(out.expiresAt, expiresAt);
});

/* ====================================================================================
 * Deletion
 * ================================================================================== */

test("a confidential artefact can be deleted through the API", async () => {
  const env = makeEnv();
  await upload(env, JOB, containerFor(JOB));
  assert.equal(env.__bucket.size, 1);

  const response = await worker.fetch(
    new Request(`https://worker.example/jobs/${JOB}/artifact`, { method: "DELETE" }),
    env
  );

  assert.equal(response.status, 200);
  assert.equal((await response.json()).deleted, true);
  assert.equal(env.__bucket.size, 0);
});

test("both modes are deleted by the same code path", async () => {
  for (const mode of ["standard", "confidential"]) {
    const env = makeEnv();
    await upload(env, JOB, mode === "confidential" ? containerFor(JOB) : new Uint8Array(16), { mode });

    const response = await worker.fetch(
      new Request(`https://worker.example/jobs/${JOB}/video`, { method: "DELETE" }),
      env
    );
    const body = await response.json();

    assert.equal(response.status, 200, mode);
    assert.equal(body.removed, 1, mode);
    assert.equal(env.__bucket.size, 0, mode);
  }
});

test("DELETE /jobs/:id also removes the generation's input keyframes", async () => {
  const env = makeEnv();
  await upload(env, JOB, containerFor(JOB));
  env.__bucket.set(`inputs/${JOB}/first-frame.png`, { bytes: new Uint8Array(4) });
  env.__bucket.set(`inputs/${JOB}/last-frame.png`, { bytes: new Uint8Array(4) });
  // Another generation's input must be untouched.
  env.__bucket.set("inputs/other-job/first-frame.png", { bytes: new Uint8Array(4) });

  const response = await worker.fetch(
    new Request(`https://worker.example/jobs/${JOB}`, { method: "DELETE" }),
    env
  );
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.equal(body.scope, "generation");
  assert.equal(body.removed, 3, "artefact plus two keyframes");
  assert.deepEqual([...env.__bucket.keys()], ["inputs/other-job/first-frame.png"]);
});

test("deleting a confidential artefact twice is safe", async () => {
  const env = makeEnv();
  await upload(env, JOB, containerFor(JOB));

  const request = () =>
    worker.fetch(new Request(`https://worker.example/jobs/${JOB}`, { method: "DELETE" }), env);

  assert.equal((await request()).status, 200);
  const second = await request();
  assert.equal(second.status, 200);
  assert.equal((await second.json()).removed, 0);
});

/* ====================================================================================
 * Container parsing conformance
 * ================================================================================== */

test("the Worker parses the Python-generated conformance vector", () => {
  const container = Uint8Array.from(Buffer.from(VECTOR.container_hex, "hex"));
  const parsed = parseContainerPrefix(container);

  assert.equal(parsed.version, 1);
  assert.equal(parsed.algorithm, "AES-256-GCM");
  assert.equal(parsed.headerLength, VECTOR.header_length);
  assert.equal(parsed.nonceOffset, VECTOR.offsets.nonce);
  assert.equal(parsed.ciphertextOffset, VECTOR.offsets.ciphertext);
  assert.deepEqual(parsed.header, VECTOR.header);
});

test("the parser rejects anything that is not a container", () => {
  const cases = {
    "an MP4": Uint8Array.from([0, 0, 0, 0x18, 0x66, 0x74, 0x79, 0x70, 0x6d, 0x70, 0x34, 0x32]),
    empty: new Uint8Array(0),
    "magic only": new TextEncoder().encode("CGEN"),
    "zero header length": Uint8Array.from([67, 71, 69, 78, 1, 1, 0, 0])
  };
  for (const [name, bytes] of Object.entries(cases)) {
    assert.throws(() => parseContainerPrefix(bytes), undefined, name);
  }
});

/* ====================================================================================
 * Capabilities
 * ================================================================================== */

test("capabilities advertises the privacy modes and their availability", async () => {
  const response = await worker.fetch(
    new Request("https://worker.example/capabilities"),
    makeEnv()
  );
  const body = await response.json();

  assert.equal(body.privacyModes.standard.available, true);
  assert.equal(body.privacyModes.confidential.available, true);
  assert.equal(body.privacyModes.private.available, false);
  assert.ok(body.privacyModes.private.reason.length > 40);
  assert.deepEqual(body.encryption.algorithms, ["AES-256-GCM"]);
  assert.ok(body.encryption.kdfs.includes("argon2id"));
  assert.equal(body.encryption.keyBytes, 32);
});
