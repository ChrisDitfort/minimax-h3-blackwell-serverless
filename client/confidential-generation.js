/*
 * Confidential Generation - browser client.
 *
 * Zero dependencies, WebCrypto only. Drop it into any frontend; it is a plain ES module
 * and imports nothing.
 *
 * WHAT LIVES WHERE, AND WHY
 * -------------------------
 * There are four distinct keys in this design and conflating any two of them is how these
 * systems go wrong. Named explicitly throughout:
 *
 *     passphrase              typed by the user. Never leaves this file.
 *        |  KDF
 *        v
 *     keyEncryptionKey (KEK)  256 bits, derived. Never leaves this file. Never stored.
 *        |  AES-256-GCM
 *        v
 *     privateKey              RSA. Stored only in its encrypted form. Never sent anywhere.
 *
 *     publicKey               RSA. Safe to send; it can encrypt and cannot decrypt.
 *        |  sent with the generation request
 *        v
 *     fileEncryptionKey (FEK) 256 bits, random, generated *by the inference worker*, one
 *                             per video, wrapped to publicKey. Never seen by this file
 *                             except transiently after unwrapping it for playback.
 *
 * The property that follows: a complete copy of everything the platform stores - the
 * encrypted video, the wrapped file key, the job payload, the status JSON - is not enough
 * to decrypt anything. The private key is required, and it is only ever here.
 *
 * This is not zero-knowledge inference. The model needs the plaintext prompt and produces
 * plaintext frames inside the inference environment. What is protected is the *stored*
 * artefact. See docs/confidential-generation.md.
 *
 * WHICH KDF, AND WHY THE DEFAULT IS PBKDF2
 * ----------------------------------------
 * Argon2id is the better password KDF, and the bundle records which one was used, so an
 * artefact encrypted either way stays openable. It costs a WebAssembly dependency, because
 * no browser ships Argon2id natively and this repository has no frontend build step. The
 * default is therefore PBKDF2-HMAC-SHA256 at 600,000 iterations - OWASP's current floor -
 * which WebCrypto implements natively.
 *
 *     import { argon2id } from "hash-wasm";
 *     const bundle = await createKeyBundle(passphrase, {
 *       kdf: newKdfParams({ name: "argon2id", parameters: { memorySize: 65536, iterations: 3 } }),
 *       argon2id
 *     });
 */

/* ------------------------------------------------------------------------------------
 * Container format - mirrors artifacts.py. See docs/confidential-generation.md.
 * ---------------------------------------------------------------------------------- */

const MAGIC = "CGEN";

/** v1: the file key came from the passphrase. v2: it is random and wrapped to a public key. */
export const CONTAINER_V1_SYMMETRIC = 1;
export const CONTAINER_V2_HYBRID = 2;
const SUPPORTED_CONTAINER_VERSIONS = new Set([CONTAINER_V1_SYMMETRIC, CONTAINER_V2_HYBRID]);

const SUITES = { 1: "AES-256-GCM" };
const KEY_WRAP_RSA_OAEP_256 = "RSA-OAEP-256";
const PREAMBLE_BYTES = 8;
const MAX_HEADER_BYTES = 8192;
const NONCE_BYTES = 12;
const TAG_BYTES = 16;
const KEY_BYTES = 32;

export const BUNDLE_VERSION = 1;
export const DEFAULT_MODULUS_LENGTH = 3072;

/*
 * The message a user sees when decryption fails, whatever the reason.
 *
 * A wrong passphrase, a truncated download, modified ciphertext and a swapped object are
 * the same event to AES-GCM - authentication failed - and telling them apart for the user
 * would mean telling an attacker apart too. One message, no detail.
 */
export const DECRYPT_FAILURE_MESSAGE = "Unable to decrypt this video. Check the passphrase.";

export class DecryptionError extends Error {
  constructor(message = DECRYPT_FAILURE_MESSAGE) {
    super(message);
    this.name = "DecryptionError";
  }
}

/* ------------------------------------------------------------------------------------
 * Key derivation
 * ---------------------------------------------------------------------------------- */

export const DEFAULT_KDF = Object.freeze({
  name: "pbkdf2-sha256",
  parameters: Object.freeze({ iterations: 600000 })
});

/**
 * Fresh KDF parameters: a random salt plus the chosen cost.
 *
 * The salt is not secret. It exists so that two users with the same passphrase do not get
 * the same key, and so a precomputed table cannot be reused. It is stored in the key
 * bundle, which is exactly what lets the same passphrase reproduce the KEK later.
 */
export function newKdfParams(overrides = {}) {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  return {
    name: overrides.name || DEFAULT_KDF.name,
    salt: toBase64Url(salt),
    parameters: { ...DEFAULT_KDF.parameters, ...(overrides.parameters || {}) }
  };
}

/**
 * Derive the key-encryption key that protects the private key.
 *
 * Named for its role rather than its mechanism: this key encrypts another key. It never
 * touches a video, never leaves the browser and is never stored - it is recomputed from
 * the passphrase each time the bundle is unlocked.
 */
export async function deriveKeyEncryptionKey(passphrase, kdf = DEFAULT_KDF, options = {}) {
  return derive256(passphrase, kdf, options);
}

/**
 * Derive a v1 file key from a passphrase.
 *
 * Deprecated, and kept only so artefacts written under the old symmetric design remain
 * openable. The computation is identical to deriveKeyEncryptionKey; the difference is
 * entirely one of role, which is the point of giving it a different name.
 */
export async function deriveFileKeyV1(passphrase, kdf = DEFAULT_KDF, options = {}) {
  return derive256(passphrase, kdf, options);
}

/** Backwards-compatible alias for callers written against the v1 module. */
export const deriveKey = deriveFileKeyV1;

async function derive256(passphrase, kdf, options) {
  if (typeof passphrase !== "string" || passphrase.length === 0) {
    throw new Error("A passphrase is required.");
  }

  const name = String(kdf.name || DEFAULT_KDF.name).toLowerCase();
  const salt = fromBase64Url(kdf.salt);
  if (salt.length < 8) {
    throw new Error("KDF salt is too short.");
  }

  if (name === "pbkdf2-sha256" || name === "pbkdf2-sha512") {
    const hash = name.endsWith("512") ? "SHA-512" : "SHA-256";
    const iterations = Number(kdf.parameters?.iterations ?? DEFAULT_KDF.parameters.iterations);
    if (!Number.isFinite(iterations) || iterations < 100000) {
      throw new Error("PBKDF2 iterations must be at least 100000.");
    }

    const material = await crypto.subtle.importKey(
      "raw",
      new TextEncoder().encode(passphrase),
      "PBKDF2",
      false,
      ["deriveBits"]
    );
    const bits = await crypto.subtle.deriveBits(
      { name: "PBKDF2", salt, iterations, hash },
      material,
      KEY_BYTES * 8
    );
    return new Uint8Array(bits);
  }

  if (name === "argon2id") {
    if (typeof options.argon2id !== "function") {
      throw new Error(
        "This key was derived with Argon2id, but no Argon2id implementation was supplied. " +
          "Pass one as options.argon2id - for example hash-wasm's argon2id. No browser " +
          "provides it natively."
      );
    }
    const parameters = kdf.parameters || {};
    const raw = await options.argon2id({
      password: passphrase,
      salt,
      parallelism: Number(parameters.parallelism ?? 1),
      iterations: Number(parameters.iterations ?? 3),
      memorySize: Number(parameters.memorySize ?? parameters.memory ?? 65536),
      hashLength: KEY_BYTES,
      outputType: "binary"
    });
    const key = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
    if (key.length !== KEY_BYTES) {
      throw new Error(`Argon2id returned ${key.length} bytes, expected ${KEY_BYTES}.`);
    }
    return key;
  }

  throw new Error(`Unsupported KDF '${name}'.`);
}

/**
 * Overwrite a key buffer.
 *
 * Worth doing and worth being honest about: JavaScript engines copy and move memory
 * freely, and neither V8 nor the browser gives a page any way to find or erase copies the
 * runtime made. This zeroes the one buffer you hold. It does not guarantee the key is gone
 * from the process.
 */
export function wipe(bytes) {
  if (bytes instanceof Uint8Array) bytes.fill(0);
}

/* ------------------------------------------------------------------------------------
 * The key bundle: a public key plus a passphrase-encrypted private key
 * ---------------------------------------------------------------------------------- */

/**
 * Create a key pair and protect its private half with a passphrase.
 *
 * Returns a bundle that is safe to persist anywhere the user finds convenient - a file, a
 * synced note, `localStorage`, or a future account-scoped store on the platform. It
 * contains no usable secret: the private key inside it is AES-256-GCM ciphertext under a
 * key derived from a passphrase that is not in the bundle.
 *
 * The pair is generated extractable, because the private key has to be exported once in
 * order to be encrypted. After that it is only ever imported non-extractable, so an
 * unlocked bundle cannot be turned back into exportable key material by page script.
 */
export async function createKeyBundle(passphrase, options = {}) {
  if (typeof passphrase !== "string" || passphrase.length === 0) {
    throw new Error("A passphrase is required.");
  }

  const modulusLength = options.modulusLength || DEFAULT_MODULUS_LENGTH;
  if (modulusLength < 3072) {
    throw new Error("Use an RSA modulus of at least 3072 bits.");
  }

  const pair = await crypto.subtle.generateKey(
    {
      name: "RSA-OAEP",
      modulusLength,
      publicExponent: new Uint8Array([1, 0, 1]),
      hash: "SHA-256"
    },
    true,
    ["encrypt", "decrypt"]
  );

  const spki = new Uint8Array(await crypto.subtle.exportKey("spki", pair.publicKey));
  const pkcs8 = new Uint8Array(await crypto.subtle.exportKey("pkcs8", pair.privateKey));

  const kdf = options.kdf || newKdfParams();
  const keyId = await deriveKeyId(spki);

  const skeleton = {
    v: BUNDLE_VERSION,
    keyId,
    publicKey: toBase64Url(spki),
    publicKeyAlgorithm: KEY_WRAP_RSA_OAEP_256,
    kdf,
    createdAt: new Date().toISOString().replace(/\.\d+Z$/, "Z")
  };

  const sealed = await sealPrivateKey(pkcs8, passphrase, skeleton, options);
  wipe(pkcs8);

  return { ...skeleton, ...sealed };
}

/**
 * Unlock a bundle: derive the KEK, decrypt the private key, import it for decryption only.
 *
 * The returned private key is a non-extractable `CryptoKey`. It can unwrap file keys and
 * nothing else - it cannot be exported, and there is no code path here that would.
 */
export async function unlockKeyBundle(bundle, passphrase, options = {}) {
  validateBundle(bundle);

  const kek = await deriveKeyEncryptionKey(passphrase, bundle.kdf, options);
  let pkcs8;
  try {
    const kekKey = await crypto.subtle.importKey("raw", kek, "AES-GCM", false, ["decrypt"]);
    pkcs8 = new Uint8Array(
      await crypto.subtle.decrypt(
        {
          name: "AES-GCM",
          iv: fromBase64Url(bundle.privateKeyNonce),
          // The public half, the key id and the KDF parameters are all authenticated, so
          // nobody can swap a different public key into a stored bundle and have future
          // videos silently encrypted to a key they control.
          additionalData: bundleAdditionalData(bundle),
          tagLength: TAG_BYTES * 8
        },
        kekKey,
        fromBase64Url(bundle.encryptedPrivateKey)
      )
    );
  } catch {
    throw new DecryptionError("Unable to unlock this key. Check the passphrase.");
  } finally {
    wipe(kek);
  }

  try {
    const privateKey = await crypto.subtle.importKey(
      "pkcs8",
      pkcs8,
      { name: "RSA-OAEP", hash: "SHA-256" },
      false,
      ["decrypt"]
    );
    return { privateKey, publicKey: bundle.publicKey, keyId: bundle.keyId, kdf: bundle.kdf };
  } finally {
    wipe(pkcs8);
  }
}

/**
 * Re-encrypt a bundle's private key under a new passphrase.
 *
 * A legitimate operation that never weakens the design: the old passphrase is required,
 * the private key never leaves this function in plaintext, and no copy is retained. This
 * is not a recovery mechanism - losing a passphrase with no bundle re-encrypted under a
 * known one is still unrecoverable, by design.
 */
export async function changeBundlePassphrase(bundle, oldPassphrase, newPassphrase, options = {}) {
  validateBundle(bundle);

  const kek = await deriveKeyEncryptionKey(oldPassphrase, bundle.kdf, options);
  let pkcs8;
  try {
    const kekKey = await crypto.subtle.importKey("raw", kek, "AES-GCM", false, ["decrypt"]);
    pkcs8 = new Uint8Array(
      await crypto.subtle.decrypt(
        {
          name: "AES-GCM",
          iv: fromBase64Url(bundle.privateKeyNonce),
          additionalData: bundleAdditionalData(bundle),
          tagLength: TAG_BYTES * 8
        },
        kekKey,
        fromBase64Url(bundle.encryptedPrivateKey)
      )
    );
  } catch {
    throw new DecryptionError("Unable to unlock this key. Check the passphrase.");
  } finally {
    wipe(kek);
  }

  try {
    // A new salt as well as a new passphrase: reusing the salt would leak that the two
    // passphrases protect the same key to anyone holding both versions of the bundle.
    const skeleton = { ...bundle, kdf: options.kdf || newKdfParams({ name: bundle.kdf.name }) };
    delete skeleton.encryptedPrivateKey;
    delete skeleton.privateKeyNonce;
    const sealed = await sealPrivateKey(pkcs8, newPassphrase, skeleton, options);
    return { ...skeleton, ...sealed };
  } finally {
    wipe(pkcs8);
  }
}

async function sealPrivateKey(pkcs8, passphrase, skeleton, options) {
  const kek = await deriveKeyEncryptionKey(passphrase, skeleton.kdf, options);
  try {
    const kekKey = await crypto.subtle.importKey("raw", kek, "AES-GCM", false, ["encrypt"]);
    const nonce = crypto.getRandomValues(new Uint8Array(NONCE_BYTES));
    const sealed = new Uint8Array(
      await crypto.subtle.encrypt(
        {
          name: "AES-GCM",
          iv: nonce,
          additionalData: bundleAdditionalData(skeleton),
          tagLength: TAG_BYTES * 8
        },
        kekKey,
        pkcs8
      )
    );
    return {
      encryptedPrivateKey: toBase64Url(sealed),
      privateKeyNonce: toBase64Url(nonce)
    };
  } finally {
    wipe(kek);
  }
}

/** The bundle fields bound into the private key's authentication tag. */
function bundleAdditionalData(bundle) {
  return new TextEncoder().encode(
    JSON.stringify({
      v: bundle.v,
      keyId: bundle.keyId,
      publicKey: bundle.publicKey,
      publicKeyAlgorithm: bundle.publicKeyAlgorithm,
      kdf: bundle.kdf
    })
  );
}

function validateBundle(bundle) {
  if (!bundle || typeof bundle !== "object") {
    throw new Error("A key bundle is required.");
  }
  if (bundle.v !== BUNDLE_VERSION) {
    throw new Error(`Unsupported key bundle version ${bundle.v}.`);
  }
  for (const field of [
    "keyId",
    "publicKey",
    "publicKeyAlgorithm",
    "encryptedPrivateKey",
    "privateKeyNonce"
  ]) {
    if (typeof bundle[field] !== "string" || !bundle[field]) {
      throw new Error(`Key bundle is missing '${field}'.`);
    }
  }
  if (!bundle.kdf || typeof bundle.kdf !== "object") {
    throw new Error("Key bundle is missing its KDF parameters.");
  }
}

/** Serialize a bundle for storage. Contains no usable secret; safe to write to a file. */
export function exportKeyBundle(bundle) {
  validateBundle(bundle);
  return JSON.stringify(bundle, null, 2);
}

export function importKeyBundle(text) {
  let bundle;
  try {
    bundle = typeof text === "string" ? JSON.parse(text) : text;
  } catch {
    throw new Error("That is not a valid key bundle.");
  }
  validateBundle(bundle);
  return bundle;
}

/** A key's stable name: the truncated SHA-256 of its SPKI. Mirrors artifacts.public_key_id. */
export async function deriveKeyId(spki) {
  const digest = await crypto.subtle.digest("SHA-256", spki);
  return toBase64Url(new Uint8Array(digest).slice(0, 16));
}

/* ------------------------------------------------------------------------------------
 * Generation
 * ---------------------------------------------------------------------------------- */

/**
 * Submit a generation. With a bundle it is confidential; without one it is standard.
 *
 * Neither the passphrase nor the private key is a parameter of this function, and that is
 * deliberate: the signature should make it impossible to pass one by accident.
 */
export async function generate(apiBase, request, { bundle, publicKey, keyId, fetchImpl = fetch } = {}) {
  const body = { ...request };

  const encoded = bundle ? bundle.publicKey : publicKey;
  if (encoded) {
    body.privacyMode = "confidential";
    body.encryption = {
      version: CONTAINER_V2_HYBRID,
      algorithm: "AES-256-GCM",
      keyWrapAlgorithm: KEY_WRAP_RSA_OAEP_256,
      publicKey: encoded,
      keyId: bundle ? bundle.keyId : keyId
    };
    if (!body.encryption.keyId) delete body.encryption.keyId;
  }

  const response = await fetchImpl(`${trimEnd(apiBase)}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload?.error || `Generation failed with HTTP ${response.status}`);
  }
  return payload;
}

/* ------------------------------------------------------------------------------------
 * Retrieval and decryption
 * ---------------------------------------------------------------------------------- */

/**
 * Split a container into its parts. Does not decrypt, and does not authenticate anything:
 * until decryption succeeds, treat every field here as unverified.
 */
export function parseContainer(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);

  if (bytes.length < PREAMBLE_BYTES) {
    throw new DecryptionError("This file is too small to be an encrypted artefact.");
  }
  if (String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !== MAGIC) {
    throw new DecryptionError("This file is not an encrypted artefact.");
  }

  const version = bytes[4];
  const suite = bytes[5];
  if (!SUPPORTED_CONTAINER_VERSIONS.has(version)) {
    throw new DecryptionError(
      `This artefact uses container version ${version}; this client understands ` +
        `${[...SUPPORTED_CONTAINER_VERSIONS].join(" and ")}. Update the page.`
    );
  }
  if (!SUITES[suite]) {
    throw new DecryptionError(`This artefact uses an unknown cipher suite (${suite}).`);
  }

  const headerLength = (bytes[6] << 8) | bytes[7];
  if (headerLength === 0 || headerLength > MAX_HEADER_BYTES) {
    throw new DecryptionError("This artefact's header is malformed.");
  }

  const headerEnd = PREAMBLE_BYTES + headerLength;
  const ciphertextStart = headerEnd + NONCE_BYTES;
  if (bytes.length < ciphertextStart + TAG_BYTES) {
    throw new DecryptionError("This artefact is incomplete - the download may have been cut short.");
  }

  let header;
  try {
    header = JSON.parse(new TextDecoder().decode(bytes.subarray(PREAMBLE_BYTES, headerEnd)));
  } catch {
    throw new DecryptionError("This artefact's header is malformed.");
  }

  const keyWrap = version === CONTAINER_V2_HYBRID ? header.kw : null;
  if (version === CONTAINER_V2_HYBRID) {
    if (!keyWrap || typeof keyWrap !== "object" || !keyWrap.wrappedFileKey) {
      throw new DecryptionError("This artefact is missing its wrapped file key.");
    }
    if (keyWrap.alg !== KEY_WRAP_RSA_OAEP_256) {
      throw new DecryptionError(`This artefact uses an unknown key-wrap algorithm (${keyWrap.alg}).`);
    }
  }

  return {
    version,
    suite,
    algorithm: SUITES[suite],
    header,
    keyWrap,
    // The header bytes exactly as they appear on the wire: they are the AEAD's associated
    // data, so they must be authenticated byte for byte, not re-serialized from `header`.
    additionalData: bytes.subarray(PREAMBLE_BYTES, headerEnd),
    nonce: bytes.subarray(headerEnd, ciphertextStart),
    // AES-GCM in WebCrypto expects ciphertext and tag concatenated, which is exactly how
    // the format lays them out. No reassembly.
    ciphertextWithTag: bytes.subarray(ciphertextStart)
  };
}

/**
 * Decrypt a container. Returns a Blob typed as the original media, plus its header.
 *
 * `opener` is whichever key the artefact's version calls for:
 *
 *   v2 - an unlocked bundle, or its `privateKey`. The wrapped file key is unwrapped with
 *        it, and the file key then decrypts the video.
 *   v1 - the raw 32-byte file key, derived from the passphrase by deriveFileKeyV1.
 *
 * Pass `expectArtifactId` whenever the caller knows which generation it asked for. The
 * header is authenticated, so a mismatch means a different artefact was served - a real
 * substitution, not a corrupted download.
 */
export async function decryptArtifact(buffer, opener, { expectArtifactId } = {}) {
  const container = parseContainer(buffer);

  let fileKeyBytes = null;
  let ownsFileKey = false;

  try {
    if (container.version === CONTAINER_V2_HYBRID) {
      const privateKey = opener?.privateKey ?? opener;
      if (!privateKey || typeof privateKey !== "object" || privateKey.type !== "private") {
        throw new DecryptionError(
          "This video needs your private key. Unlock your key bundle with your passphrase first."
        );
      }

      if (opener?.keyId && container.keyWrap.keyId && opener.keyId !== container.keyWrap.keyId) {
        throw new DecryptionError(
          "This video was encrypted to a different key than the one you unlocked."
        );
      }

      try {
        fileKeyBytes = new Uint8Array(
          await crypto.subtle.decrypt(
            { name: "RSA-OAEP" },
            privateKey,
            fromBase64Url(container.keyWrap.wrappedFileKey)
          )
        );
        ownsFileKey = true;
      } catch {
        throw new DecryptionError();
      }
    } else {
      fileKeyBytes = opener instanceof Uint8Array ? opener : opener?.fileKey;
      if (!(fileKeyBytes instanceof Uint8Array) || fileKeyBytes.length !== KEY_BYTES) {
        throw new DecryptionError("The decryption key is the wrong size.");
      }
    }

    const fileKey = await crypto.subtle.importKey("raw", fileKeyBytes, "AES-GCM", false, ["decrypt"]);

    let plaintext;
    try {
      plaintext = await crypto.subtle.decrypt(
        {
          name: "AES-GCM",
          iv: container.nonce,
          additionalData: container.additionalData,
          tagLength: TAG_BYTES * 8
        },
        fileKey,
        container.ciphertextWithTag
      );
    } catch {
      throw new DecryptionError();
    }

    // Only now is the header trustworthy: it was the AEAD's associated data, so reaching
    // this line proves nobody altered it - including the wrapped key inside it.
    if (
      expectArtifactId &&
      container.header.artifactId &&
      container.header.artifactId !== expectArtifactId
    ) {
      throw new DecryptionError(
        "This artefact decrypted correctly but belongs to a different generation."
      );
    }

    return {
      blob: new Blob([plaintext], { type: container.header.contentType || "video/mp4" }),
      header: container.header,
      cryptoVersion: container.version
    };
  } finally {
    // The file key existed for one decryption. Only wipe the copy this function made -
    // a v1 caller's key belongs to the caller.
    if (ownsFileKey) wipe(fileKeyBytes);
  }
}

/**
 * Fetch a job's artefact and decrypt it. The whole file is downloaded first.
 *
 * That is a real limitation and the right trade for version 1 of the format: a single GCM
 * tag over the whole file cannot be verified until the last byte arrives, so playback
 * cannot start early. For a five-second H3 clip - a couple of megabytes - it is not
 * noticeable. The format is versioned precisely so a chunked variant can lift this later
 * without breaking anything already encrypted.
 */
export async function fetchAndDecrypt(apiBase, jobId, opener, { fetchImpl = fetch, signal } = {}) {
  const response = await fetchImpl(`${trimEnd(apiBase)}/jobs/${encodeURIComponent(jobId)}/artifact`, {
    signal
  });

  if (!response.ok) {
    throw new Error(
      response.status === 404
        ? "This video is no longer available. It may have been deleted."
        : `Could not download the artefact (HTTP ${response.status}).`
    );
  }

  if (response.headers.get("X-Artifact-Encrypted") === "false") {
    // A standard artefact. Return it as-is rather than failing: a caller that asked for a
    // video should get the video.
    return { blob: await response.blob(), header: null, encrypted: false };
  }

  const buffer = await response.arrayBuffer();
  const { blob, header, cryptoVersion } = await decryptArtifact(buffer, opener, {
    expectArtifactId: jobId
  });
  return { blob, header, cryptoVersion, encrypted: true };
}

/**
 * Attach a decrypted blob to a `<video>` element and hand back a cleanup function.
 *
 * Object URLs keep their blob alive until revoked, so a page that creates one per playback
 * and never revokes leaks the decrypted video for as long as the tab is open.
 */
export function attachToVideo(videoElement, blob) {
  const url = URL.createObjectURL(blob);
  videoElement.src = url;
  return () => {
    URL.revokeObjectURL(url);
    if (videoElement.src === url) videoElement.removeAttribute("src");
  };
}

/* ------------------------------------------------------------------------------------
 * base64url
 * ---------------------------------------------------------------------------------- */

export function toBase64Url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function fromBase64Url(value) {
  const padded = String(value).replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded + "=".repeat((4 - (padded.length % 4)) % 4));
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

function trimEnd(base) {
  return String(base).replace(/\/+$/, "");
}
