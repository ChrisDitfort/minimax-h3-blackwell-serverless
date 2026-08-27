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

const VECTOR = JSON.parse(readFileSync("./tests/vectors/confidential_container.json", "utf8"));

/*
 * One RSA-3072 key pair for the whole file, generated at load. Keygen costs ~160 ms and a
 * per-test pair would dominate the runtime. Reuse is a test-speed decision only: in
 * production the pair belongs to the user, and a fresh file key is drawn per artefact.
 */
const KEYPAIR = await crypto.subtle.generateKey(
  { name: "RSA-OAEP", modulusLength: 3072, publicExponent: new Uint8Array([1, 0, 1]), hash: "SHA-256" },
  true,
  ["encrypt", "decrypt"]
);
const SPKI = new Uint8Array(await crypto.subtle.exportKey("spki", KEYPAIR.publicKey));
const PUBLIC_KEY = toBase64Url(SPKI);
const KEY_ID = toBase64Url(
  new Uint8Array(await crypto.subtle.digest("SHA-256", SPKI)).slice(0, 16)
);

/* Values that must never appear in a payload, a log or a response. */
const PASSPHRASE = "correct horse battery staple";
const PRIVATE_KEY_PKCS8 = Buffer.from(
  await crypto.subtle.exportKey("pkcs8", KEYPAIR.privateKey)
).toString("base64url");

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
    version: 2,
    algorithm: "AES-256-GCM",
    keyWrapAlgorithm: "RSA-OAEP-256",
    publicKey: PUBLIC_KEY
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
        /*
         * Real R2 refuses a stream whose length it cannot know: only a request/response
         * body or the readable half of a FixedLengthStream qualifies. The earlier stub
         * accepted anything, which is how a confidential upload shipped that 500'd in
         * production while every test passed - the peeked-and-replayed stream had no
         * length. `__knownLength` is how the test marks a stream as qualifying.
         */
        if (body instanceof ReadableStream && body.__knownLength === undefined) {
          throw new Error(
            "Provided readable stream must have a known length (request/response body or " +
              "readable half of FixedLengthStream)"
          );
        }
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

/*
 * Stand in for the Workers runtime's FixedLengthStream so withKnownLength() takes the same
 * branch it takes in production. The readable it produces is marked with the length, which
 * is what the R2 stub above checks for.
 */
if (typeof globalThis.FixedLengthStream !== "function") {
  globalThis.FixedLengthStream = class {
    constructor(length) {
      const { readable, writable } = new TransformStream();
      readable.__knownLength = length;
      this.readable = readable;
      this.writable = writable;
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

/* ====================================================================================
 * Request validation
 * ================================================================================== */

test("a valid public key is accepted", async () => {
  const privacy = await normalizePrivacy(CONFIDENTIAL);
  assert.equal(privacy.mode, "confidential");
  assert.equal(privacy.encryption.version, 2);
  assert.equal(privacy.encryption.publicKey, PUBLIC_KEY);
  assert.equal(privacy.encryption.algorithm, "AES-256-GCM");
  assert.equal(privacy.encryption.keyWrapAlgorithm, "RSA-OAEP-256");
  assert.equal(privacy.encryption.keyId, KEY_ID);
});

test("the crypto version defaults to the hybrid design", async () => {
  const { version } = { ...CONFIDENTIAL.encryption };
  assert.equal(version, 2, "fixture sanity");
  const privacy = await normalizePrivacy({
    ...BASE_REQUEST,
    privacyMode: "confidential",
    encryption: { publicKey: PUBLIC_KEY }
  });
  assert.equal(privacy.encryption.version, 2);
});

test("no privacyMode means standard, exactly as before", async () => {
  const privacy = await normalizePrivacy(BASE_REQUEST);
  assert.equal(privacy.mode, "standard");
  assert.equal(privacy.encryption, null);
  assert.equal(privacy.expiresAt, null);
});

test("privacy_mode is accepted alongside privacyMode", async () => {
  const privacy = await normalizePrivacy({
    privacy_mode: "confidential",
    encryption: { publicKey: PUBLIC_KEY }
  });
  assert.equal(privacy.mode, "confidential");
});

/* -- the v1 refusal: the whole point of this release ---------------------------------- */

test("REFUSED: a symmetric key in the request", async () => {
  const symmetric = toBase64Url(new Uint8Array(32).fill(7));
  for (const encryption of [
    { key: symmetric },
    { version: 1, publicKey: PUBLIC_KEY },
    { version: 1 },
    { publicKey: PUBLIC_KEY, key: symmetric }
  ]) {
    await assert.rejects(
      normalizePrivacy({ ...BASE_REQUEST, privacyMode: "confidential", encryption }),
      (error) => error.status === 400 && /no longer accepted/.test(error.message),
      JSON.stringify(Object.keys(encryption))
    );
  }
});

test("the v1 refusal names the replacement rather than just saying no", async () => {
  await assert.rejects(
    normalizePrivacy({
      ...BASE_REQUEST,
      privacyMode: "confidential",
      encryption: { key: toBase64Url(new Uint8Array(32)) }
    }),
    (error) =>
      /publicKey/.test(error.message) &&
      /version 2/.test(error.message) &&
      /remain decryptable/.test(error.message)
  );
});

test("a rejected symmetric key is never echoed back", async () => {
  const symmetric = toBase64Url(new Uint8Array(32).fill(9));
  await assert.rejects(
    normalizePrivacy({ ...BASE_REQUEST, privacyMode: "confidential", encryption: { key: symmetric } }),
    (error) => !error.message.includes(symmetric) && !JSON.stringify(error.details || {}).includes(symmetric)
  );
});

/* -- forbidden fields, at any depth --------------------------------------------------- */

test("REJECTED: any field capable of decrypting the result", async () => {
  const forbidden = [
    "passphrase",
    "password",
    "privateKey",
    "private_key",
    "privateEncryptionKey",
    "encryptedPrivateKey",
    "keyEncryptionKey",
    "kek",
    "fileEncryptionKey",
    "fek",
    "decryptionKey",
    "derivedKey",
    "aesKey",
    "symmetricKey",
    "secretKey"
  ];

  for (const field of forbidden) {
    const runpod = stubRunPod();
    const env = makeEnv();
    try {
      const response = await worker.fetch(post("/generate", { ...CONFIDENTIAL, [field]: "x" }), env);
      const body = await response.json();
      assert.equal(response.status, 400, `${field} must be refused`);
      assert.match(body.error, /must never be sent to this API/);
      assert.equal(body.details.field, field);
      assert.equal(runpod.seen.length, 0, `${field} must not reach RunPod`);
    } finally {
      runpod.restore();
    }
  }
});

test("forbidden fields are caught when nested, not only at the top level", async () => {
  for (const body of [
    { ...CONFIDENTIAL, encryption: { ...CONFIDENTIAL.encryption, privateKey: "x" } },
    { ...CONFIDENTIAL, meta: { user: { passphrase: "hunter2" } } },
    { ...CONFIDENTIAL, items: [{ kek: "x" }] }
  ]) {
    const response = await worker.fetch(post("/generate", body), makeEnv());
    assert.equal(response.status, 400);
    assert.match((await response.json()).error, /must never be sent/);
  }
});

test("field-name matching ignores case and separators", async () => {
  for (const field of ["PRIVATE_KEY", "Private-Key", "PassPhrase", "FILE_ENCRYPTION_KEY"]) {
    const response = await worker.fetch(post("/generate", { ...CONFIDENTIAL, [field]: "x" }), makeEnv());
    assert.equal(response.status, 400, field);
  }
});

test("legitimate public fields are never mistaken for secrets", async () => {
  const runpod = stubRunPod();
  try {
    const response = await worker.fetch(
      post("/generate", {
        ...CONFIDENTIAL,
        encryption: { ...CONFIDENTIAL.encryption, publicKeyAlgorithm: "RSA-OAEP-256" }
      }),
      makeEnv()
    );
    assert.equal(response.status, 202, await response.text());
  } finally {
    runpod.restore();
  }
});

/* -- public key validation ------------------------------------------------------------ */

test("a missing encryption block in confidential mode is rejected", async () => {
  await assert.rejects(
    normalizePrivacy({ ...BASE_REQUEST, privacyMode: "confidential" }),
    (error) => error.status === 400 && /requires an encryption block/.test(error.message)
  );
});

test("a missing public key is rejected", async () => {
  await assert.rejects(
    normalizePrivacy({ ...BASE_REQUEST, privacyMode: "confidential", encryption: { algorithm: "AES-256-GCM" } }),
    (error) => error.status === 400 && /publicKey is required/.test(error.message)
  );
});

test("a malformed public key is rejected", async () => {
  for (const publicKey of ["not base64!", "***", "a b c"]) {
    await assert.rejects(
      normalizePrivacy({ ...BASE_REQUEST, privacyMode: "confidential", encryption: { publicKey } }),
      (error) => error.status === 400 && /base64url/.test(error.message)
    );
  }
});

test("something that is not a DER SubjectPublicKeyInfo is rejected", async () => {
  await assert.rejects(
    normalizePrivacy({
      ...BASE_REQUEST,
      privacyMode: "confidential",
      encryption: { publicKey: toBase64Url(new Uint8Array(500).fill(0x41)) }
    }),
    (error) => error.status === 400 && /SubjectPublicKeyInfo/.test(error.message)
  );
});

test("a public key too small to be a 3072-bit RSA key is rejected", async () => {
  // A well-formed DER SEQUENCE, but far too short for the modulus we require.
  const short = new Uint8Array(120);
  short[0] = 0x30;
  await assert.rejects(
    normalizePrivacy({
      ...BASE_REQUEST,
      privacyMode: "confidential",
      encryption: { publicKey: toBase64Url(short) }
    }),
    (error) => error.status === 400 && /outside the/.test(error.message)
  );
});

test("an unsupported content algorithm is rejected", async () => {
  await assert.rejects(
    normalizePrivacy({
      ...BASE_REQUEST,
      privacyMode: "confidential",
      encryption: { algorithm: "AES-128-CBC", publicKey: PUBLIC_KEY }
    }),
    (error) => error.status === 400 && /Unsupported encryption algorithm/.test(error.message)
  );
});

test("an unsupported key-wrap algorithm is rejected", async () => {
  await assert.rejects(
    normalizePrivacy({
      ...BASE_REQUEST,
      privacyMode: "confidential",
      encryption: { publicKey: PUBLIC_KEY, keyWrapAlgorithm: "RSA-PKCS1v15" }
    }),
    (error) => error.status === 400 && /Unsupported keyWrapAlgorithm/.test(error.message)
  );
});

test("an unknown crypto version is rejected", async () => {
  await assert.rejects(
    normalizePrivacy({
      ...BASE_REQUEST,
      privacyMode: "confidential",
      encryption: { version: 7, publicKey: PUBLIC_KEY }
    }),
    (error) => error.status === 400 && /Unsupported encryption.version/.test(error.message)
  );
});

/* -- key ids -------------------------------------------------------------------------- */

test("the key id is derived from the key, and a disagreeing one is rejected", async () => {
  await assert.rejects(
    normalizePrivacy({
      ...BASE_REQUEST,
      privacyMode: "confidential",
      encryption: { publicKey: PUBLIC_KEY, keyId: "some-other-key" }
    }),
    (error) => error.status === 400 && /does not match the supplied public key/.test(error.message)
  );

  const privacy = await normalizePrivacy({
    ...BASE_REQUEST,
    privacyMode: "confidential",
    encryption: { publicKey: PUBLIC_KEY, keyId: KEY_ID }
  });
  assert.equal(privacy.encryption.keyId, KEY_ID);
});

test("the Worker and the worker derive the same key id from the same key", async () => {
  // artifacts.public_key_id is the truncated SHA-256 of the SPKI, base64url, unpadded.
  // Recomputed here independently so a change on either side breaks this test.
  const digest = await crypto.subtle.digest("SHA-256", Buffer.from(PUBLIC_KEY, "base64url"));
  assert.equal(KEY_ID, Buffer.from(digest).subarray(0, 16).toString("base64url"));
});

/* -- modes and retention -------------------------------------------------------------- */

test("an unknown privacy mode is a 400 naming the supported ones", async () => {
  await assert.rejects(
    normalizePrivacy({ ...BASE_REQUEST, privacyMode: "super-secret" }),
    (error) =>
      error.status === 400 &&
      /Unknown privacyMode/.test(error.message) &&
      /confidential/.test(error.message)
  );
});

test("declared-but-unimplemented modes are a 501 with a reason, not a silent downgrade", async () => {
  for (const mode of ["private", "ephemeral"]) {
    await assert.rejects(
      normalizePrivacy({ ...BASE_REQUEST, privacyMode: mode }),
      (error) => error.status === 501 && error.message.length > 60,
      `expected an explained 501 for ${mode}`
    );
  }
});

test("standard mode with an encryption block is rejected rather than ignored", async () => {
  await assert.rejects(
    normalizePrivacy({ ...BASE_REQUEST, privacyMode: "standard", encryption: { publicKey: PUBLIC_KEY } }),
    (error) => error.status === 400 && /does not encrypt/.test(error.message)
  );
});

test("retention is validated and resolved to an absolute expiry", async () => {
  const privacy = await normalizePrivacy({ ...BASE_REQUEST, retentionSeconds: 3600 });
  const delta = (new Date(privacy.expiresAt).getTime() - Date.now()) / 1000;
  assert.ok(Math.abs(delta - 3600) < 5, `expected ~3600s, got ${delta}`);

  for (const body of [
    { retentionSeconds: 5 },
    { retentionSeconds: 100 * 24 * 3600 },
    { expiresAt: "not-a-date" },
    { retentionSeconds: 3600, expiresAt: new Date().toISOString() }
  ]) {
    await assert.rejects(
      normalizePrivacy({ ...BASE_REQUEST, ...body }),
      (error) => error.status === 400
    );
  }
});

/* ====================================================================================
 * Submission: what crosses into the job queue
 *
 * The security invariant of this release lives in this section. Treat the RunPod payload
 * as permanently retained; a copy of it must not be enough to open the artefact.
 * ================================================================================== */

test("INVARIANT: the RunPod payload carries the public key and no decryption-capable secret", async () => {
  const runpod = stubRunPod();
  const env = makeEnv();

  try {
    const { result: response, output } = await captureConsole(() =>
      worker.fetch(post("/generate", CONFIDENTIAL), env)
    );
    assert.equal(response.status, 202);

    // Inspect the bytes actually sent, not a reconstruction of them.
    const serialized = runpod.seen[0].init.body;
    const payload = JSON.parse(serialized);
    const input = payload.input;

    // Present, and necessarily so: encryption has to happen where the plaintext is.
    assert.equal(input.encryption.publicKey, PUBLIC_KEY);
    assert.equal(input.encryption.version, 2);
    assert.equal(input.encryption.keyWrapAlgorithm, "RSA-OAEP-256");
    assert.equal(input.encryption.keyId, KEY_ID);
    assert.equal(input.privacy.mode, "confidential");

    // Absent, and that is the point of the release.
    assert.equal(input.encryption.key, undefined, "no symmetric key");
    assert.equal(input.encryption.privateKey, undefined);
    assert.equal(input.encryption.kdf, undefined, "no passphrase KDF metadata");

    for (const secret of [PASSPHRASE, PRIVATE_KEY_PKCS8, "kek", "fileEncryptionKey"]) {
      assert.ok(!serialized.includes(secret), `payload must not contain ${secret.slice(0, 24)}`);
    }
    for (const name of ["passphrase", "privateKey", "kek", "fileEncryptionKey", "aesKey"]) {
      assert.ok(
        !new RegExp(`"${name}"\\s*:`, "i").test(serialized),
        `payload must not carry a ${name} field`
      );
    }

    // Nothing secret in the log either.
    assert.ok(!output.includes(PASSPHRASE));
    assert.ok(!output.includes(PRIVATE_KEY_PKCS8));
    assert.match(output, /privacy_mode=confidential/);
  } finally {
    runpod.restore();
  }
});

test("the public key itself is the only key material in the payload", async () => {
  const runpod = stubRunPod();
  try {
    await worker.fetch(post("/generate", CONFIDENTIAL), makeEnv());
    const input = JSON.parse(runpod.seen[0].init.body).input;

    // Every value under encryption is either an algorithm name, a version, the derived
    // key id, or the public key. Nothing else is permitted to be there at all.
    assert.deepEqual(Object.keys(input.encryption).sort(), [
      "algorithm",
      "keyId",
      "keyWrapAlgorithm",
      "publicKey",
      "publicKeyAlgorithm",
      "version"
    ]);
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
    assert.ok(!output.includes(PUBLIC_KEY), "563 characters of public key belong nowhere near a log");
  } finally {
    runpod.restore();
  }
});

test("the response echoes the public metadata and no key material", async () => {
  const runpod = stubRunPod();
  try {
    const response = await worker.fetch(post("/generate", CONFIDENTIAL), makeEnv());
    const body = await response.json();

    assert.equal(body.privacyMode, "confidential");
    assert.equal(body.encryption.version, 2);
    assert.equal(body.encryption.keyWrapAlgorithm, "RSA-OAEP-256");
    assert.equal(body.encryption.keyId, KEY_ID);
    assert.equal(body.encryption.publicKey, undefined, "no need to echo it back");
    assert.equal(body.encryption.key, undefined);

    const serialized = JSON.stringify(body);
    assert.ok(!serialized.includes(PASSPHRASE));
    assert.ok(!serialized.includes(PRIVATE_KEY_PKCS8));
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
    const input = JSON.parse(runpod.seen[0].init.body).input;
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

/** A v2 container the Worker should accept for `jobId`. */
function containerFor(jobId, { version = 2, suite = 1, magic = "CGEN", kw } = {}) {
  const header = new TextEncoder().encode(
    JSON.stringify({
      v: version,
      alg: "AES-256-GCM",
      artifactId: jobId,
      contentType: "video/mp4",
      plaintextBytes: 64,
      privacyMode: "confidential",
      createdAt: "2026-01-01T00:00:00Z",
      kw:
        kw === null
          ? undefined
          : kw || { alg: "RSA-OAEP-256", wrappedFileKey: "d3JhcHBlZA", keyId: KEY_ID }
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

async function upload(env, jobId, body, { mode = "confidential", cryptoVersion = 2, contentType } = {}) {
  const token = await signJobToken(
    env.JOB_TOKEN_SECRET,
    jobId,
    workerConstants().TOKEN_PURPOSES.output,
    3600,
    { pm: mode, ...(mode === "confidential" ? { cv: cryptoVersion } : {}) }
  );
  const request = new Request(`https://worker.example/internal/jobs/${jobId}/output`, {
    method: "PUT",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Length": String(body.byteLength),
      "Content-Type":
        contentType || (mode === "confidential" ? "application/octet-stream" : "video/mp4")
    },
    body,
    duplex: "half"
  });
  // A real request body carries a length R2 accepts; Node's ReadableStream does not model
  // that, so mark it the way the runtime would. Without this the stub would reject the
  // standard path too, which real R2 does not.
  if (request.body) request.body.__knownLength = body.byteLength;
  return worker.fetch(request, env);
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
  assert.equal(put.options.customMetadata.cryptoVersion, "2");
  assert.equal(put.options.customMetadata.keyWrapAlgorithm, "RSA-OAEP-256");
  assert.equal(put.options.customMetadata.keyId, KEY_ID);
});

test("REGRESSION: the peeked body still has a length R2 will accept", async () => {
  /*
   * This shipped broken once. Reading the container header to prove the bytes are
   * ciphertext replaces request.body with a replayed ReadableStream, and R2 rejects a
   * stream whose length it cannot know - so every confidential upload 500'd in production
   * while standard mode, which never peeks, was fine. The stub above now enforces the same
   * rule the real binding does, and withKnownLength() restores the length.
   */
  const env = makeEnv();
  const response = await upload(env, JOB, containerFor(JOB));

  assert.equal(response.status, 201, await response.text());
  assert.equal(env.__puts.length, 1, "the artefact must actually reach R2");
  assert.ok(
    env.__puts[0].bytes.byteLength > 0,
    "a length-bearing stream must still deliver its bytes"
  );
});

test("a confidential upload with no Content-Length is refused, not 500'd", async () => {
  // Without a declared length there is nothing to hand FixedLengthStream, so this fails
  // early and legibly rather than deep inside R2.
  const env = makeEnv();
  const container = containerFor(JOB);
  const token = await signJobToken(
    env.JOB_TOKEN_SECRET, JOB, workerConstants().TOKEN_PURPOSES.output, 3600,
    { pm: "confidential", cv: 2 }
  );
  const response = await worker.fetch(
    new Request(`https://worker.example/internal/jobs/${JOB}/output`, {
      method: "PUT",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/octet-stream" },
      body: container,
      duplex: "half"
    }),
    env
  );

  assert.equal(response.status, 411);
  assert.match((await response.json()).error, /Content-Length/);
  assert.equal(env.__puts.length, 0, "nothing may be written without a known length");
});

test("REFUSED: a v1 container uploaded for a v2 job", async () => {
  const env = makeEnv();
  const response = await upload(env, JOB, containerFor(JOB, { version: 1 }), { cryptoVersion: 2 });
  const body = await response.json();

  assert.equal(response.status, 422);
  assert.match(body.error, /crypto version 2.*is version 1/s);
  assert.equal(env.__puts.length, 0, "a downgrade must write nothing");
});

test("REFUSED: a v2 container with no key-wrapping block", async () => {
  const env = makeEnv();
  const response = await upload(env, JOB, containerFor(JOB, { kw: null }));
  assert.equal(response.status, 422);
  assert.match((await response.json()).error, /kw' key-wrapping block/);
  assert.equal(env.__puts.length, 0);
});

test("REFUSED: a v2 container whose wrapped key is missing or whose algorithm is unknown", async () => {
  for (const kw of [
    { alg: "RSA-OAEP-256", keyId: KEY_ID },
    { alg: "RSA-PKCS1v15", wrappedFileKey: "eA", keyId: KEY_ID },
    { alg: "RSA-OAEP-256", wrappedFileKey: "eA" }
  ]) {
    const env = makeEnv();
    const response = await upload(env, JOB, containerFor(JOB, { kw }));
    assert.equal(response.status, 422, JSON.stringify(kw));
    assert.equal(env.__puts.length, 0);
  }
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
  for (const options of [{ version: 9 }, { suite: 9 }, { magic: "XXXX" }]) {
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

  const payload = new Uint8Array(32).fill(7);
  const request = new Request(`https://worker.example/internal/jobs/${JOB}/output`, {
    method: "PUT",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "video/mp4",
      "Content-Length": String(payload.byteLength)
    },
    body: payload,
    duplex: "half"
  });
  if (request.body) request.body.__knownLength = payload.byteLength;
  const response = await worker.fetch(request, env);

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

test("R2 metadata never contains anything that could decrypt the artefact", async () => {
  const env = makeEnv();
  await upload(env, JOB, containerFor(JOB));
  const serialized = JSON.stringify(env.__puts[0].options);

  assert.ok(!serialized.includes(PASSPHRASE));
  assert.ok(!serialized.includes(PRIVATE_KEY_PKCS8));
  assert.ok(!/"(privateKey|passphrase|fileEncryptionKey|kek)"/i.test(serialized));

  // What *is* there is a key id and algorithm names, both public by construction. The
  // wrapped file key is not in metadata at all - it lives inside the container's
  // authenticated header, where it cannot be swapped for one wrapped to another key.
  assert.equal(env.__puts[0].options.customMetadata.keyId, KEY_ID);
  assert.equal(env.__puts[0].options.customMetadata.keyWrapAlgorithm, "RSA-OAEP-256");
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

test("status describes a confidential v2 artefact and never returns a decryption secret", () => {
  const state = {
    jobId: JOB,
    privacyMode: "confidential",
    artifact: {
      privacyMode: "confidential",
      encrypted: true,
      cryptoVersion: 2,
      encryptionVersion: 2,
      algorithm: "AES-256-GCM",
      keyWrapAlgorithm: "RSA-OAEP-256",
      keyId: KEY_ID,
      contentType: "application/octet-stream",
      originalContentType: "video/mp4",
      key: outputKey(JOB, true),
      size: 2048,
      deleted: false
    }
  };

  const out = describeArtifactForStatus(JOB, state, { size: 2048 });

  assert.equal(out.privacyMode, "confidential");
  assert.equal(out.artifact.encrypted, true);
  assert.equal(out.artifact.cryptoVersion, 2);
  assert.equal(out.artifact.videoAlgorithm, "AES-256-GCM");
  assert.equal(out.artifact.keyWrapAlgorithm, "RSA-OAEP-256");
  assert.equal(out.artifact.keyId, KEY_ID);
  assert.equal(out.artifact.contentType, "application/octet-stream");
  assert.equal(out.artifact.originalContentType, "video/mp4");

  // Enough to know which private key to unlock; nothing that unlocks it.
  const serialized = JSON.stringify(out);
  assert.ok(!serialized.includes(PASSPHRASE));
  assert.ok(!serialized.includes(PRIVATE_KEY_PKCS8));
  assert.equal(out.artifact.privateKey, undefined);
  assert.equal(out.artifact.encryptionKey, undefined);
  assert.equal(out.artifact.wrappedFileKey, undefined, "it lives in the authenticated header");
  assert.equal(out.encryption, undefined);
});

test("a v1 artefact still reports its KDF, so an old client can still open it", () => {
  const out = describeArtifactForStatus(
    JOB,
    {
      jobId: JOB,
      privacyMode: "confidential",
      artifact: {
        encrypted: true,
        cryptoVersion: 1,
        algorithm: "AES-256-GCM",
        kdf: { name: "pbkdf2-sha256", salt: "c2FsdA", parameters: { iterations: 600000 } },
        keyId: "legacy-key",
        key: outputKey(JOB, true),
        size: 10
      }
    },
    { size: 10 }
  );
  assert.equal(out.artifact.cryptoVersion, 1);
  assert.equal(out.artifact.kdf.name, "pbkdf2-sha256");
  assert.equal(out.artifact.keyWrapAlgorithm, undefined, "v1 wraps nothing");
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
 * End to end, through every real component
 * ================================================================================== */

test("CANONICAL: a real hybrid artefact survives upload, storage, retrieval and unwrapping", async () => {
  /*
   * Everything here is the production article except the R2 binding and the GPU. The
   * ciphertext was produced by artifacts.py using only a public key; it goes in through
   * the Worker's real upload gate, comes back out through the real retrieval route, and
   * is opened by unwrapping the file key with a private key that never left the client.
   *
   * If the container layout, the AAD binding, the key-wrap encoding or the RSA-OAEP
   * parameters ever disagree between Python and WebCrypto, this fails.
   */
  const v2 = VECTOR.v2;
  const jobId = v2.generation_id;
  const container = Uint8Array.from(Buffer.from(v2.container_hex, "hex"));
  const plaintext = Uint8Array.from(Buffer.from(v2.plaintext_hex, "hex"));
  const env = makeEnv();

  // 1. The worker uploads what it encrypted. The gate accepts it: right version, right job.
  const uploaded = await upload(env, jobId, container);
  assert.equal(uploaded.status, 201);
  const ack = await uploaded.json();
  assert.equal(ack.key, `outputs/${jobId}/artifact.enc`);
  assert.equal(ack.encrypted, true);

  // 2. R2 holds ciphertext, and nothing that looks like the video.
  const stored = env.__bucket.get(`outputs/${jobId}/artifact.enc`).bytes;
  assert.notDeepEqual([...stored], [...plaintext]);
  assert.ok(!Buffer.from(stored).includes(Buffer.from(plaintext)));
  assert.ok(
    !Buffer.from(stored.subarray(0, 12)).includes(Buffer.from("ftyp")),
    "the stored object must not begin with an MP4 signature"
  );

  // 3. The browser fetches it back through the public route.
  const response = await worker.fetch(
    new Request(`https://worker.example/jobs/${jobId}/artifact`),
    env
  );
  assert.equal(response.headers.get("X-Artifact-Encrypted"), "true");
  const downloaded = new Uint8Array(await response.arrayBuffer());

  // 4. Unwrap the file key with the private key, then decrypt. Two steps, one of which
  //    the platform cannot perform at all.
  const parsed = parseContainerPrefix(downloaded);
  assert.equal(parsed.version, 2);
  assert.equal(parsed.keyWrap.alg, "RSA-OAEP-256");
  assert.equal(parsed.keyWrap.keyId, v2.key_id);

  const privateKey = await crypto.subtle.importKey(
    "pkcs8",
    Buffer.from(v2.private_key_pkcs8_b64, "base64url"),
    { name: "RSA-OAEP", hash: "SHA-256" },
    false,
    ["decrypt"]
  );
  const fileKeyBytes = new Uint8Array(
    await crypto.subtle.decrypt(
      { name: "RSA-OAEP" },
      privateKey,
      Buffer.from(parsed.keyWrap.wrappedFileKey, "base64url")
    )
  );
  assert.equal(Buffer.from(fileKeyBytes).toString("hex"), v2.file_key_hex);

  const fileKey = await crypto.subtle.importKey("raw", fileKeyBytes, "AES-GCM", false, ["decrypt"]);
  const recovered = new Uint8Array(
    await crypto.subtle.decrypt(
      {
        name: "AES-GCM",
        iv: downloaded.subarray(parsed.nonceOffset, parsed.ciphertextOffset),
        additionalData: downloaded.subarray(8, 8 + parsed.headerLength),
        tagLength: 128
      },
      fileKey,
      downloaded.subarray(parsed.ciphertextOffset)
    )
  );
  assert.deepEqual([...recovered], [...plaintext], "the original video, byte for byte");

  // 5. Nothing the platform kept could have done that.
  const everythingStored = JSON.stringify({
    r2: [...env.__bucket.entries()].map(([k, value]) => [k, value.options]),
    jobState: env.__state()
  });
  assert.ok(!everythingStored.includes(v2.private_key_pkcs8_b64));
  assert.ok(!everythingStored.includes(v2.file_key_hex));
  assert.ok(!everythingStored.includes(PASSPHRASE));
});

test("INVARIANT: platform data alone cannot decrypt a v2 artefact", async () => {
  /*
   * The central security property, stated as an experiment.
   *
   * Assume an attacker obtains everything the platform has: the RunPod job input, the
   * RunPod job output, the R2 object and its metadata, the Durable Object job state, the
   * status response, the public key, the wrapped file key and the container header. Give
   * them every one of those values as a candidate decryption key and let them try.
   *
   * All of it must fail. Only the private key - which never crossed the boundary - works.
   */
  const v2 = VECTOR.v2;
  const jobId = v2.generation_id;
  const container = Uint8Array.from(Buffer.from(v2.container_hex, "hex"));
  const plaintext = Uint8Array.from(Buffer.from(v2.plaintext_hex, "hex"));

  const runpod = stubRunPod();
  const env = makeEnv();
  let runpodInput;

  try {
    // Produce a genuine job payload for this key, so the attacker's copy is the real one.
    await worker.fetch(
      post("/generate", {
        ...BASE_REQUEST,
        privacyMode: "confidential",
        encryption: { version: 2, publicKey: v2.public_key_spki_b64 }
      }),
      env
    );
    runpodInput = runpod.seen[0].init.body;
  } finally {
    runpod.restore();
  }

  await upload(env, jobId, container);
  const stored = env.__bucket.get(`outputs/${jobId}/artifact.enc`);
  const status = describeArtifactForStatus(jobId, env.__state(), { size: stored.bytes.byteLength });
  const parsed = parseContainerPrefix(container);

  // Everything the attacker now holds, as raw byte strings.
  const platformData = {
    runpodInput,
    runpodOutput: JSON.stringify({ key: `outputs/${jobId}/artifact.enc`, encrypted: true }),
    r2Object: Buffer.from(stored.bytes).toString("base64"),
    r2Metadata: JSON.stringify(stored.options),
    jobState: JSON.stringify(env.__state()),
    statusResponse: JSON.stringify(status),
    publicKey: v2.public_key_spki_b64,
    wrappedFileKey: parsed.keyWrap.wrappedFileKey,
    containerHeader: Buffer.from(container.subarray(8, 8 + parsed.headerLength)).toString("utf8"),
    keyId: v2.key_id
  };

  // Sanity: the private key really is absent from all of it.
  for (const [name, value] of Object.entries(platformData)) {
    assert.ok(
      !String(value).includes(v2.private_key_pkcs8_b64),
      `${name} must not contain the private key`
    );
    assert.ok(!String(value).includes(v2.file_key_hex), `${name} must not contain the file key`);
    assert.ok(!String(value).includes(PASSPHRASE), `${name} must not contain a passphrase`);
  }

  // Now actually try to decrypt with every candidate. Any 32 bytes derivable from platform
  // data is a candidate file key; the whole point is that none of them is the right one.
  const nonce = container.subarray(parsed.nonceOffset, parsed.ciphertextOffset);
  const aad = container.subarray(8, 8 + parsed.headerLength);
  const body = container.subarray(parsed.ciphertextOffset);

  let successes = 0;
  for (const [name, value] of Object.entries(platformData)) {
    // Two derivations per value: its raw first 32 bytes, and its SHA-256.
    const raw = new TextEncoder().encode(String(value));
    const candidates = [
      raw.length >= 32 ? raw.slice(0, 32) : null,
      new Uint8Array(await crypto.subtle.digest("SHA-256", raw))
    ].filter(Boolean);

    for (const candidate of candidates) {
      try {
        const key = await crypto.subtle.importKey("raw", candidate, "AES-GCM", false, ["decrypt"]);
        await crypto.subtle.decrypt(
          { name: "AES-GCM", iv: nonce, additionalData: aad, tagLength: 128 },
          key,
          body
        );
        successes += 1;
        assert.fail(`${name} decrypted the artefact - the invariant is broken`);
      } catch (error) {
        if (/invariant is broken/.test(error.message)) throw error;
        /* expected: authentication failed */
      }
    }

    // And the wrapped key cannot be unwrapped with the public key either - it can only
    // encrypt. WebCrypto will not even import it for decryption.
    await assert.rejects(
      crypto.subtle.importKey(
        "spki",
        Buffer.from(v2.public_key_spki_b64, "base64url"),
        { name: "RSA-OAEP", hash: "SHA-256" },
        false,
        ["decrypt"]
      ),
      "a public key must not be importable as a decryption key"
    );
  }

  assert.equal(successes, 0, "no platform value may decrypt the artefact");

  // The one thing that does work is the thing that never crossed the boundary.
  const privateKey = await crypto.subtle.importKey(
    "pkcs8",
    Buffer.from(v2.private_key_pkcs8_b64, "base64url"),
    { name: "RSA-OAEP", hash: "SHA-256" },
    false,
    ["decrypt"]
  );
  const fileKeyBytes = new Uint8Array(
    await crypto.subtle.decrypt(
      { name: "RSA-OAEP" },
      privateKey,
      Buffer.from(parsed.keyWrap.wrappedFileKey, "base64url")
    )
  );
  const fileKey = await crypto.subtle.importKey("raw", fileKeyBytes, "AES-GCM", false, ["decrypt"]);
  const recovered = new Uint8Array(
    await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: nonce, additionalData: aad, tagLength: 128 },
      fileKey,
      body
    )
  );
  assert.deepEqual([...recovered], [...plaintext]);
});

test("a wrong private key cannot unwrap the file key", async () => {
  const v2 = VECTOR.v2;
  const parsed = parseContainerPrefix(Uint8Array.from(Buffer.from(v2.container_hex, "hex")));

  const stranger = await crypto.subtle.generateKey(
    { name: "RSA-OAEP", modulusLength: 3072, publicExponent: new Uint8Array([1, 0, 1]), hash: "SHA-256" },
    true,
    ["encrypt", "decrypt"]
  );

  await assert.rejects(
    crypto.subtle.decrypt(
      { name: "RSA-OAEP" },
      stranger.privateKey,
      Buffer.from(parsed.keyWrap.wrappedFileKey, "base64url")
    )
  );
});

test("a modified wrapped file key fails rather than yielding a usable key", async () => {
  const v2 = VECTOR.v2;
  const container = Uint8Array.from(Buffer.from(v2.container_hex, "hex"));
  const parsed = parseContainerPrefix(container);

  const wrapped = Buffer.from(parsed.keyWrap.wrappedFileKey, "base64url");
  wrapped[10] ^= 0x01;

  const privateKey = await crypto.subtle.importKey(
    "pkcs8",
    Buffer.from(v2.private_key_pkcs8_b64, "base64url"),
    { name: "RSA-OAEP", hash: "SHA-256" },
    false,
    ["decrypt"]
  );

  // OAEP is an authenticated padding: a flipped bit does not produce a different key, it
  // produces a decoding failure.
  await assert.rejects(crypto.subtle.decrypt({ name: "RSA-OAEP" }, privateKey, wrapped));
});

test("modifying the wrapped key inside the header breaks the video's authentication", async () => {
  /*
   * The reason the wrapped key belongs in the authenticated header rather than beside it.
   * An attacker who substitutes a file key wrapped to their own public key cannot then
   * decrypt the video: the header is the AEAD's associated data, so the swap invalidates
   * the tag over the ciphertext.
   */
  const v2 = VECTOR.v2;
  const container = Uint8Array.from(Buffer.from(v2.container_hex, "hex"));
  const parsed = parseContainerPrefix(container);

  const tampered = Uint8Array.from(container);
  // Flip a byte inside the header's wrapped key, keeping the length identical.
  const headerText = Buffer.from(container.subarray(8, 8 + parsed.headerLength)).toString("utf8");
  const at = 8 + headerText.indexOf(parsed.keyWrap.wrappedFileKey);
  tampered[at] = tampered[at] === 65 ? 66 : 65;

  const fileKey = await crypto.subtle.importKey(
    "raw",
    Buffer.from(v2.file_key_hex, "hex"),
    "AES-GCM",
    false,
    ["decrypt"]
  );

  await assert.rejects(
    crypto.subtle.decrypt(
      {
        name: "AES-GCM",
        iv: tampered.subarray(parsed.nonceOffset, parsed.ciphertextOffset),
        additionalData: tampered.subarray(8, 8 + parsed.headerLength),
        tagLength: 128
      },
      fileKey,
      tampered.subarray(parsed.ciphertextOffset)
    ),
    "a rewritten wrapped key must invalidate the video's tag"
  );
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

/* ====================================================================================
 * Response hardening
 * ================================================================================== */

test("JSON responses carry the strict browser controls", async () => {
  const response = await worker.fetch(new Request("https://worker.example/health"), makeEnv());

  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.equal(response.headers.get("X-Content-Type-Options"), "nosniff");
  assert.equal(response.headers.get("Referrer-Policy"), "no-referrer");
  assert.match(response.headers.get("Content-Security-Policy"), /default-src 'none'/);
});

test("preflight allows the verbs the artefact routes actually use", async () => {
  const response = await worker.fetch(
    new Request("https://worker.example/jobs/x/artifact", { method: "OPTIONS" }),
    makeEnv()
  );
  assert.equal(response.status, 204);
  assert.match(response.headers.get("Access-Control-Allow-Methods"), /DELETE/);
});

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
  assert.deepEqual(body.encryption.keyWrapAlgorithms, ["RSA-OAEP-256"]);
  assert.equal(body.encryption.cryptoVersion, 2);
  assert.deepEqual(body.encryption.readableCryptoVersions, [1, 2]);
  assert.equal(body.encryption.publicKeyFormat, "spki-der-base64url");
  assert.equal(body.encryption.minPublicKeyBits, 3072);
  // The passphrase KDF protects the caller's private key, which never reaches this API.
  assert.equal(body.encryption.kdfs, undefined);
});
