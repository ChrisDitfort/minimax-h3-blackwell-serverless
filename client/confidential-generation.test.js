/*
 * Browser client tests, run under Node's WebCrypto - the same API the browser exposes.
 *
 * Run with:  node --test
 *
 * The test that matters most is the first one. It takes a container that Python produced,
 * derives the key from a passphrase using this module, and decrypts the video. If the
 * container format, the canonical header serialization, the AAD binding or the KDF
 * parameters ever drift between the two languages, that single test fails - which is the
 * only thing standing between a format change and a user who can no longer open a video
 * they generated last month.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  DECRYPT_FAILURE_MESSAGE,
  DecryptionError,
  decryptArtifact,
  deriveKey,
  fetchAndDecrypt,
  fromBase64Url,
  generate,
  newKdfParams,
  parseContainer,
  toBase64Url,
  wipe
} from "./confidential-generation.js";

const VECTOR = JSON.parse(
  readFileSync(new URL("../tests/vectors/confidential_container.json", import.meta.url), "utf8")
);

const CONTAINER = Uint8Array.from(Buffer.from(VECTOR.container_hex, "hex"));
const PLAINTEXT = Uint8Array.from(Buffer.from(VECTOR.plaintext_hex, "hex"));
const KEY = Uint8Array.from(Buffer.from(VECTOR.key_hex, "hex"));

async function bytesOf(blob) {
  return new Uint8Array(await blob.arrayBuffer());
}

/* ====================================================================================
 * Cross-language: Python encrypted this, the browser opens it
 * ================================================================================== */

test("a passphrase alone recovers a video that Python encrypted", async () => {
  // Exactly what a user does: type the passphrase, get the video. Nothing else is needed -
  // no key, no server-side secret, no state the platform holds.
  const key = await deriveKey(VECTOR.kdf.passphrase, VECTOR.kdf);
  assert.equal(
    Buffer.from(key).toString("hex"),
    VECTOR.kdf.derived_key_hex,
    "PBKDF2 must agree byte for byte across implementations"
  );

  const { blob, header } = await decryptArtifact(CONTAINER, key, {
    expectArtifactId: VECTOR.generation_id
  });

  assert.deepEqual([...(await bytesOf(blob))], [...PLAINTEXT]);
  assert.equal(blob.type, "video/mp4");
  assert.equal(header.artifactId, VECTOR.generation_id);
  assert.equal(header.alg, "AES-256-GCM");
});

test("the parsed layout matches the offsets Python recorded", () => {
  const container = parseContainer(CONTAINER);
  assert.equal(container.version, 1);
  assert.equal(container.algorithm, "AES-256-GCM");
  assert.deepEqual(container.header, VECTOR.header);
  assert.equal(container.additionalData.byteLength, VECTOR.header_length);
  assert.equal(container.nonce.byteLength, 12);
  assert.equal(
    container.ciphertextWithTag.byteLength,
    PLAINTEXT.byteLength + 16,
    "GCM is length preserving; the extra 16 bytes are the tag"
  );
});

/* ====================================================================================
 * Failure is opaque and total
 * ================================================================================== */

test("a wrong key fails with the generic message and nothing else", async () => {
  const wrong = new Uint8Array(32).fill(0xaa);
  await assert.rejects(
    decryptArtifact(CONTAINER, wrong),
    (error) =>
      error instanceof DecryptionError &&
      error.message === DECRYPT_FAILURE_MESSAGE &&
      !/tag|GCM|authentication|OperationError/i.test(error.message)
  );
});

test("a wrong passphrase is indistinguishable from a wrong key", async () => {
  const key = await deriveKey("not the passphrase", VECTOR.kdf);
  await assert.rejects(decryptArtifact(CONTAINER, key), (error) => error.message === DECRYPT_FAILURE_MESSAGE);
});

test("modified ciphertext fails authentication", async () => {
  const tampered = Uint8Array.from(CONTAINER);
  tampered[tampered.length - 40] ^= 0x01;
  await assert.rejects(decryptArtifact(tampered, KEY), DecryptionError);
});

test("a modified tag fails", async () => {
  const tampered = Uint8Array.from(CONTAINER);
  tampered[tampered.length - 1] ^= 0x80;
  await assert.rejects(decryptArtifact(tampered, KEY), DecryptionError);
});

test("a rewritten header fails, because the header is authenticated", async () => {
  const tampered = Uint8Array.from(CONTAINER);
  const text = Buffer.from(tampered).toString("latin1");
  const at = text.indexOf(VECTOR.generation_id);
  assert.ok(at > 0, "the artefact id should be in the header");

  // Same length, so the framing stays valid and only the authenticated bytes change.
  const forged = Buffer.from("99999999-2222-3333-4444-555555555555", "latin1");
  tampered.set(forged, at);

  // The header now reads differently and parses fine...
  assert.equal(parseContainer(tampered).header.artifactId, "99999999-2222-3333-4444-555555555555");
  // ...but the ciphertext will not open under it.
  await assert.rejects(decryptArtifact(tampered, KEY), DecryptionError);
});

test("a truncated download is rejected rather than half-played", async () => {
  await assert.rejects(
    decryptArtifact(CONTAINER.subarray(0, CONTAINER.length - 8), KEY),
    DecryptionError
  );
});

test("a plaintext MP4 is not mistaken for an artefact", () => {
  const mp4 = Uint8Array.from([0, 0, 0, 0x18, 0x66, 0x74, 0x79, 0x70, 0x6d, 0x70, 0x34, 0x32]);
  assert.throws(() => parseContainer(mp4), (error) => /not an encrypted artefact/.test(error.message));
});

test("a future container version says so instead of failing obscurely", () => {
  const future = Uint8Array.from(CONTAINER);
  future[4] = 2;
  assert.throws(() => parseContainer(future), (error) => /Update the page/.test(error.message));
});

test("an artefact belonging to another generation is refused after it decrypts", async () => {
  await assert.rejects(
    decryptArtifact(CONTAINER, KEY, { expectArtifactId: "some-other-job" }),
    (error) => /different generation/.test(error.message)
  );
});

/* ====================================================================================
 * Key derivation
 * ================================================================================== */

test("the same passphrase and salt always give the same key", async () => {
  const kdf = newKdfParams({ parameters: { iterations: 100000 } });
  const a = await deriveKey("a passphrase", kdf);
  const b = await deriveKey("a passphrase", kdf);
  assert.deepEqual([...a], [...b]);
  assert.equal(a.length, 32);
});

test("a different salt gives a different key from the same passphrase", async () => {
  const first = await deriveKey("same passphrase", newKdfParams({ parameters: { iterations: 100000 } }));
  const second = await deriveKey("same passphrase", newKdfParams({ parameters: { iterations: 100000 } }));
  assert.notDeepEqual([...first], [...second]);
});

test("every generation gets a fresh random salt", () => {
  const salts = new Set(Array.from({ length: 50 }, () => newKdfParams().salt));
  assert.equal(salts.size, 50);
  assert.equal(fromBase64Url(newKdfParams().salt).length, 16);
});

test("the default KDF cost is not weakened by accident", () => {
  assert.equal(newKdfParams().name, "pbkdf2-sha256");
  assert.ok(
    newKdfParams().parameters.iterations >= 600000,
    "600k is OWASP's floor for PBKDF2-HMAC-SHA256"
  );
});

test("a too-cheap PBKDF2 cost is refused", async () => {
  await assert.rejects(
    deriveKey("passphrase", { name: "pbkdf2-sha256", salt: toBase64Url(new Uint8Array(16)), parameters: { iterations: 1000 } }),
    /at least 100000/
  );
});

test("an empty passphrase is refused", async () => {
  await assert.rejects(deriveKey("", newKdfParams()), /passphrase is required/);
});

test("argon2id is supported but needs an implementation supplied", async () => {
  const kdf = { name: "argon2id", salt: toBase64Url(new Uint8Array(16)), parameters: {} };

  await assert.rejects(deriveKey("passphrase", kdf), /no Argon2id implementation was supplied/);

  // With one, it is used - and the parameters recorded in the artefact are what it gets.
  let seen = null;
  const key = await deriveKey("passphrase", { ...kdf, parameters: { memorySize: 65536, iterations: 3 } }, {
    argon2id: async (options) => {
      seen = options;
      return new Uint8Array(32).fill(4);
    }
  });

  assert.equal(key.length, 32);
  assert.equal(seen.memorySize, 65536);
  assert.equal(seen.iterations, 3);
  assert.equal(seen.hashLength, 32);
});

test("an unknown KDF is refused rather than silently defaulted", async () => {
  await assert.rejects(
    deriveKey("passphrase", { name: "md5", salt: toBase64Url(new Uint8Array(16)) }),
    /Unsupported KDF/
  );
});

test("wipe zeroes the buffer it is given", () => {
  const key = new Uint8Array(32).fill(9);
  wipe(key);
  assert.deepEqual([...key], new Array(32).fill(0));
});

/* ====================================================================================
 * Submission
 * ================================================================================== */

test("generate() sends the derived key and never the passphrase", async () => {
  let sent = null;
  const fetchImpl = async (url, init) => {
    sent = { url, body: JSON.parse(init.body) };
    return new Response(JSON.stringify({ jobId: "j1", privacyMode: "confidential" }), {
      status: 202,
      headers: { "Content-Type": "application/json" }
    });
  };

  const kdf = newKdfParams();
  const key = await deriveKey("the user's passphrase", kdf);
  await generate("https://api.example/", { prompt: "hello" }, { key, kdf, keyId: "kid", fetchImpl });

  assert.equal(sent.url, "https://api.example/generate");
  assert.equal(sent.body.privacyMode, "confidential");
  assert.equal(sent.body.encryption.key, toBase64Url(key));
  assert.equal(sent.body.encryption.keyId, "kid");
  assert.deepEqual(sent.body.encryption.kdf, kdf);

  const serialized = JSON.stringify(sent.body);
  assert.ok(!serialized.includes("the user's passphrase"), "the passphrase must never be sent");
});

test("generate() without a key produces a plain standard request", async () => {
  let sent = null;
  const fetchImpl = async (url, init) => {
    sent = JSON.parse(init.body);
    return new Response("{}", { status: 202, headers: { "Content-Type": "application/json" } });
  };

  await generate("https://api.example", { prompt: "hello" }, { fetchImpl });
  assert.equal(sent.privacyMode, undefined);
  assert.equal(sent.encryption, undefined);
  assert.equal(sent.prompt, "hello");
});

test("generate() refuses a key of the wrong size before sending anything", async () => {
  let called = false;
  await assert.rejects(
    generate("https://api.example", {}, {
      key: new Uint8Array(16),
      fetchImpl: async () => {
        called = true;
      }
    }),
    /must be 32 raw bytes/
  );
  assert.equal(called, false);
});

/* ====================================================================================
 * Retrieval
 * ================================================================================== */

test("fetchAndDecrypt downloads ciphertext and returns a playable blob", async () => {
  const fetchImpl = async () =>
    new Response(CONTAINER, {
      status: 200,
      headers: { "X-Artifact-Encrypted": "true", "Content-Type": "application/octet-stream" }
    });

  const { blob, encrypted, header } = await fetchAndDecrypt(
    "https://api.example",
    VECTOR.generation_id,
    KEY,
    { fetchImpl }
  );

  assert.equal(encrypted, true);
  assert.equal(header.artifactId, VECTOR.generation_id);
  assert.deepEqual([...(await bytesOf(blob))], [...PLAINTEXT]);
});

test("fetchAndDecrypt passes a standard artefact straight through", async () => {
  const fetchImpl = async () =>
    new Response(PLAINTEXT, {
      status: 200,
      headers: { "X-Artifact-Encrypted": "false", "Content-Type": "video/mp4" }
    });

  const { blob, encrypted } = await fetchAndDecrypt("https://api.example", "j1", null, { fetchImpl });
  assert.equal(encrypted, false);
  assert.deepEqual([...(await bytesOf(blob))], [...PLAINTEXT]);
});

test("a deleted artefact produces a message a user can act on", async () => {
  const fetchImpl = async () => new Response("{}", { status: 404 });
  await assert.rejects(
    fetchAndDecrypt("https://api.example", "j1", KEY, { fetchImpl }),
    /no longer available/
  );
});

test("the artefact route is the one requested", async () => {
  let requested = null;
  const fetchImpl = async (url) => {
    requested = url;
    return new Response(CONTAINER, { status: 200, headers: { "X-Artifact-Encrypted": "true" } });
  };
  await fetchAndDecrypt("https://api.example/", VECTOR.generation_id, KEY, { fetchImpl });
  assert.equal(requested, `https://api.example/jobs/${VECTOR.generation_id}/artifact`);
});

/* ====================================================================================
 * base64url
 * ================================================================================== */

test("base64url round-trips, including bytes that need URL-safe characters", () => {
  for (const bytes of [
    new Uint8Array(0),
    Uint8Array.from([0xfb, 0xff, 0xfe]),
    Uint8Array.from({ length: 32 }, (_, i) => (i * 251) & 0xff)
  ]) {
    const encoded = toBase64Url(bytes);
    assert.ok(!/[+/=]/.test(encoded), `${encoded} must be URL-safe and unpadded`);
    assert.deepEqual([...fromBase64Url(encoded)], [...bytes]);
  }
});
