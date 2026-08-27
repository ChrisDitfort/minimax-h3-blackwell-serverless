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
  changeBundlePassphrase,
  createKeyBundle,
  decryptArtifact,
  deriveFileKeyV1,
  deriveKey,
  deriveKeyId,
  exportKeyBundle,
  fetchAndDecrypt,
  fromBase64Url,
  generate,
  importKeyBundle,
  newKdfParams,
  parseContainer,
  toBase64Url,
  unlockKeyBundle,
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
  const key = await deriveFileKeyV1(VECTOR.kdf.passphrase, VECTOR.kdf);
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
  // 2 is the hybrid design and is understood; 9 is genuinely from the future.
  future[4] = 9;
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

test("generate() with no bundle produces a plain standard request", async () => {
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

/* ====================================================================================
 * Crypto v2: the key pair, and the passphrase that protects it
 * ================================================================================== */

const V2 = VECTOR.v2;
const V2_CONTAINER = Uint8Array.from(Buffer.from(V2.container_hex, "hex"));
const V2_PLAINTEXT = Uint8Array.from(Buffer.from(V2.plaintext_hex, "hex"));

/* One bundle for the whole file: RSA keygen plus 600k PBKDF2 iterations is ~1s. */
const BUNDLE_PASSPHRASE = "a passphrase the platform never sees";
const BUNDLE = await createKeyBundle(BUNDLE_PASSPHRASE, {
  kdf: newKdfParams({ parameters: { iterations: 100000 } })
});

test("CROSS-LANGUAGE: a container Python wrapped with a public key opens here", async () => {
  /*
   * The conformance test that matters. Python produced this container knowing nothing but
   * a public key; opening it here exercises the container layout, the AAD binding, the
   * base64url encoding of the wrapped key and the RSA-OAEP parameters, in that order. If
   * any of the four disagrees between the two languages, this is where it shows.
   */
  const privateKey = await vectorPrivateKey();

  const { blob, header, cryptoVersion } = await decryptArtifact(
    V2_CONTAINER,
    { privateKey, keyId: V2.key_id },
    { expectArtifactId: V2.generation_id }
  );

  assert.equal(cryptoVersion, 2);
  assert.deepEqual([...(await bytesOf(blob))], [...V2_PLAINTEXT]);
  assert.equal(blob.type, "video/mp4");
  assert.equal(header.artifactId, V2.generation_id);
  assert.equal(header.kw.alg, "RSA-OAEP-256");
  assert.equal(header.kw.keyId, V2.key_id);
});

test("CANONICAL: passphrase to key pair to wrapped file key and back to the video", async () => {
  /*
   * The whole client-side journey with nothing stubbed: a passphrase produces a key pair,
   * the public half encrypts an artefact the way the worker would, and only the passphrase
   * gets it back.
   */
  const passphrase = "the user's own passphrase";
  const bundle = await createKeyBundle(passphrase, {
    kdf: newKdfParams({ parameters: { iterations: 100000 } })
  });

  // What the inference worker does, in the same order and with the same primitives.
  const video = new Uint8Array([0, 0, 0, 0x18, 0x66, 0x74, 0x79, 0x70, ...crypto.getRandomValues(new Uint8Array(2048))]);
  const container = await sealForPublicKey(bundle.publicKey, bundle.keyId, video, "gen-local");

  // What the browser does when the user comes back for it.
  const unlocked = await unlockKeyBundle(bundle, passphrase);
  const { blob } = await decryptArtifact(container, unlocked, { expectArtifactId: "gen-local" });
  assert.deepEqual([...(await bytesOf(blob))], [...video]);

  // And what happens when they get it wrong.
  await assert.rejects(unlockKeyBundle(bundle, "close but no"), DecryptionError);
});

/** Build a v2 container the way the inference worker does. Used to test the read path. */
async function sealForPublicKey(publicKeyB64, keyId, plaintext, artifactId) {
  const fileKey = crypto.getRandomValues(new Uint8Array(32));
  const encryptor = await crypto.subtle.importKey(
    "spki", fromBase64Url(publicKeyB64), { name: "RSA-OAEP", hash: "SHA-256" }, false, ["encrypt"]
  );
  const wrapped = new Uint8Array(
    await crypto.subtle.encrypt({ name: "RSA-OAEP" }, encryptor, fileKey)
  );

  const header = new TextEncoder().encode(
    JSON.stringify({
      alg: "AES-256-GCM",
      artifactId,
      contentType: "video/mp4",
      createdAt: "2026-01-01T00:00:00Z",
      kw: { alg: "RSA-OAEP-256", keyId, wrappedFileKey: toBase64Url(wrapped) },
      plaintextBytes: plaintext.length,
      privacyMode: "confidential",
      v: 2
    })
  );

  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const key = await crypto.subtle.importKey("raw", fileKey, "AES-GCM", false, ["encrypt"]);
  const sealed = new Uint8Array(
    await crypto.subtle.encrypt(
      { name: "AES-GCM", iv: nonce, additionalData: header, tagLength: 128 },
      key,
      plaintext
    )
  );

  const out = new Uint8Array(8 + header.length + 12 + sealed.length);
  out.set(new TextEncoder().encode("CGEN"), 0);
  out[4] = 2;
  out[5] = 1;
  out[6] = (header.length >> 8) & 0xff;
  out[7] = header.length & 0xff;
  out.set(header, 8);
  out.set(nonce, 8 + header.length);
  out.set(sealed, 8 + header.length + 12);
  return out;
}

test("a bundle's public key round-trips and names itself correctly", async () => {
  assert.equal(BUNDLE.v, 1);
  assert.equal(BUNDLE.publicKeyAlgorithm, "RSA-OAEP-256");
  assert.equal(BUNDLE.keyId, await deriveKeyId(fromBase64Url(BUNDLE.publicKey)));

  const spki = fromBase64Url(BUNDLE.publicKey);
  const publicKey = await crypto.subtle.importKey(
    "spki", spki, { name: "RSA-OAEP", hash: "SHA-256" }, false, ["encrypt"]
  );
  assert.equal(publicKey.type, "public");
});

test("the public key can encrypt and cannot decrypt", async () => {
  const spki = fromBase64Url(BUNDLE.publicKey);

  const encryptor = await crypto.subtle.importKey(
    "spki", spki, { name: "RSA-OAEP", hash: "SHA-256" }, false, ["encrypt"]
  );
  const wrapped = await crypto.subtle.encrypt({ name: "RSA-OAEP" }, encryptor, new Uint8Array(32));
  assert.ok(wrapped.byteLength >= 384, "a 3072-bit wrap is 384 bytes");

  // WebCrypto will not even hand back a key usable for decryption from a public key.
  await assert.rejects(
    crypto.subtle.importKey("spki", spki, { name: "RSA-OAEP", hash: "SHA-256" }, false, ["decrypt"]),
    "a public key must not be importable for decryption"
  );
});

test("the private key unwraps what the public key wrapped", async () => {
  const fileKey = crypto.getRandomValues(new Uint8Array(32));

  const encryptor = await crypto.subtle.importKey(
    "spki", fromBase64Url(BUNDLE.publicKey), { name: "RSA-OAEP", hash: "SHA-256" }, false, ["encrypt"]
  );
  const wrapped = await crypto.subtle.encrypt({ name: "RSA-OAEP" }, encryptor, fileKey);

  const { privateKey } = await unlockKeyBundle(BUNDLE, BUNDLE_PASSPHRASE);
  const recovered = new Uint8Array(
    await crypto.subtle.decrypt({ name: "RSA-OAEP" }, privateKey, wrapped)
  );
  assert.deepEqual([...recovered], [...fileKey]);
});

/* -- passphrase protection ------------------------------------------------------------ */

test("the correct passphrase unlocks the private key", async () => {
  const unlocked = await unlockKeyBundle(BUNDLE, BUNDLE_PASSPHRASE);
  assert.equal(unlocked.privateKey.type, "private");
  assert.equal(unlocked.keyId, BUNDLE.keyId);
  assert.equal(unlocked.privateKey.extractable, false, "an unlocked key must not be exportable");
});

test("a wrong passphrase fails authentication", async () => {
  await assert.rejects(
    unlockKeyBundle(BUNDLE, "not the passphrase"),
    (error) => error instanceof DecryptionError && /Check the passphrase/.test(error.message)
  );
});

test("the bundle contains no usable secret", () => {
  const serialized = exportKeyBundle(BUNDLE);
  assert.ok(!serialized.includes(BUNDLE_PASSPHRASE));
  // Everything in it is either public or ciphertext.
  assert.deepEqual(Object.keys(JSON.parse(serialized)).sort(), [
    "createdAt",
    "encryptedPrivateKey",
    "kdf",
    "keyId",
    "privateKeyNonce",
    "publicKey",
    "publicKeyAlgorithm",
    "v"
  ]);
});

test("swapping the public key in a stored bundle breaks the unlock", async () => {
  /*
   * Why the public half is bound into the private key's authentication tag. An attacker
   * who edits a bundle at rest to substitute their own public key would otherwise have
   * every future video encrypted to a key they hold, silently.
   */
  const other = await createKeyBundle("someone else", {
    kdf: newKdfParams({ parameters: { iterations: 100000 } })
  });
  const tampered = { ...BUNDLE, publicKey: other.publicKey };

  await assert.rejects(unlockKeyBundle(tampered, BUNDLE_PASSPHRASE), DecryptionError);
});

test("editing the key id or KDF parameters also breaks the unlock", async () => {
  for (const tampered of [
    { ...BUNDLE, keyId: "someone-elses-key" },
    { ...BUNDLE, kdf: { ...BUNDLE.kdf, parameters: { iterations: 100001 } } }
  ]) {
    await assert.rejects(unlockKeyBundle(tampered, BUNDLE_PASSPHRASE), DecryptionError);
  }
});

test("a malformed bundle is rejected before any crypto happens", async () => {
  for (const bad of [null, {}, { v: 99 }, { ...BUNDLE, encryptedPrivateKey: "" }]) {
    await assert.rejects(unlockKeyBundle(bad, BUNDLE_PASSPHRASE), /bundle/i);
  }
  assert.throws(() => importKeyBundle("not json"), /not a valid key bundle/);
});

test("the passphrase can be changed without the private key leaving the module", async () => {
  const rotated = await changeBundlePassphrase(BUNDLE, BUNDLE_PASSPHRASE, "a new passphrase", {
    kdf: newKdfParams({ parameters: { iterations: 100000 } })
  });

  // Same key pair, new protection.
  assert.equal(rotated.keyId, BUNDLE.keyId);
  assert.equal(rotated.publicKey, BUNDLE.publicKey);
  assert.notEqual(rotated.encryptedPrivateKey, BUNDLE.encryptedPrivateKey);
  assert.notEqual(rotated.kdf.salt, BUNDLE.kdf.salt, "a new salt, or the two are linkable");

  const unlocked = await unlockKeyBundle(rotated, "a new passphrase");
  assert.equal(unlocked.privateKey.type, "private");
  await assert.rejects(unlockKeyBundle(rotated, BUNDLE_PASSPHRASE), DecryptionError);
});

test("changing the passphrase requires the old one", async () => {
  await assert.rejects(
    changeBundlePassphrase(BUNDLE, "wrong", "new one"),
    DecryptionError
  );
});

/* -- wrong keys and tampering --------------------------------------------------------- */

test("a different private key cannot open the artefact", async () => {
  const stranger = await unlockKeyBundle(BUNDLE, BUNDLE_PASSPHRASE);
  await assert.rejects(
    decryptArtifact(V2_CONTAINER, stranger),
    (error) => error instanceof DecryptionError
  );
});

test("a mismatched key id is caught before the attempt", async () => {
  const stranger = await unlockKeyBundle(BUNDLE, BUNDLE_PASSPHRASE);
  await assert.rejects(
    decryptArtifact(V2_CONTAINER, { privateKey: stranger.privateKey, keyId: BUNDLE.keyId }),
    /encrypted to a different key/
  );
});

test("modified v2 ciphertext fails authentication", async () => {
  const privateKey = await vectorPrivateKey();
  const tampered = Uint8Array.from(V2_CONTAINER);
  tampered[tampered.length - 40] ^= 0x01;
  await assert.rejects(decryptArtifact(tampered, { privateKey }), DecryptionError);
});

test("a modified wrapped file key fails rather than yielding a different key", async () => {
  const privateKey = await vectorPrivateKey();
  const container = parseContainer(V2_CONTAINER);
  const tampered = Uint8Array.from(V2_CONTAINER);

  const headerText = Buffer.from(container.additionalData).toString("utf8");
  const at = 8 + headerText.indexOf(container.keyWrap.wrappedFileKey) + 4;
  tampered[at] = tampered[at] === 65 ? 66 : 65;

  await assert.rejects(decryptArtifact(tampered, { privateKey }), DecryptionError);
});

test("a v2 container with no wrapped key is refused", () => {
  const header = new TextEncoder().encode(
    JSON.stringify({ v: 2, alg: "AES-256-GCM", artifactId: "x", contentType: "video/mp4" })
  );
  const body = new Uint8Array(8 + header.length + 12 + 16 + 16);
  body.set(new TextEncoder().encode("CGEN"), 0);
  body[4] = 2;
  body[5] = 1;
  body[6] = (header.length >> 8) & 0xff;
  body[7] = header.length & 0xff;
  body.set(header, 8);

  assert.throws(() => parseContainer(body), /missing its wrapped file key/);
});

test("a v2 artefact handed a raw symmetric key says what is actually needed", async () => {
  await assert.rejects(
    decryptArtifact(V2_CONTAINER, new Uint8Array(32)),
    /needs your private key/
  );
});

/* -- submission ----------------------------------------------------------------------- */

test("generate() sends the public key and nothing that could decrypt", async () => {
  let sent = null;
  const fetchImpl = async (url, init) => {
    sent = { url, body: JSON.parse(init.body), raw: init.body };
    return new Response(JSON.stringify({ jobId: "j1", privacyMode: "confidential" }), {
      status: 202,
      headers: { "Content-Type": "application/json" }
    });
  };

  await generate("https://api.example/", { prompt: "hello" }, { bundle: BUNDLE, fetchImpl });

  assert.equal(sent.url, "https://api.example/generate");
  assert.equal(sent.body.privacyMode, "confidential");
  assert.equal(sent.body.encryption.version, 2);
  assert.equal(sent.body.encryption.publicKey, BUNDLE.publicKey);
  assert.equal(sent.body.encryption.keyWrapAlgorithm, "RSA-OAEP-256");
  assert.equal(sent.body.encryption.keyId, BUNDLE.keyId);

  // The three things that must never be there.
  assert.equal(sent.body.encryption.key, undefined, "no symmetric key");
  assert.equal(sent.body.encryption.privateKey, undefined);
  assert.ok(!sent.raw.includes(BUNDLE_PASSPHRASE), "the passphrase must never be sent");
  assert.ok(
    !sent.raw.includes(BUNDLE.encryptedPrivateKey),
    "not even the encrypted private key needs to go"
  );
});

test("generate() without a bundle produces a plain standard request", async () => {
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

test("fetchAndDecrypt opens a v2 artefact with an unlocked bundle", async () => {
  const privateKey = await vectorPrivateKey();
  const fetchImpl = async () =>
    new Response(V2_CONTAINER, {
      status: 200,
      headers: { "X-Artifact-Encrypted": "true", "Content-Type": "application/octet-stream" }
    });

  const { blob, encrypted, cryptoVersion } = await fetchAndDecrypt(
    "https://api.example",
    V2.generation_id,
    { privateKey },
    { fetchImpl }
  );

  assert.equal(encrypted, true);
  assert.equal(cryptoVersion, 2);
  assert.deepEqual([...(await bytesOf(blob))], [...V2_PLAINTEXT]);
});

async function vectorPrivateKey() {
  return crypto.subtle.importKey(
    "pkcs8",
    Buffer.from(V2.private_key_pkcs8_b64, "base64url"),
    { name: "RSA-OAEP", hash: "SHA-256" },
    false,
    ["decrypt"]
  );
}
