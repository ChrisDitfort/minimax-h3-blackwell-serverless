/*
 * Confidential Generation - browser client.
 *
 * Zero dependencies, WebCrypto only. Drop it into any frontend; it is a plain ES module
 * and imports nothing.
 *
 * The passphrase never leaves this file. What crosses the network is a key derived from
 * it, and that key is used once, inside the inference container, to encrypt the finished
 * video before it reaches storage. Decryption happens here, in the browser, from the
 * passphrase the user re-enters.
 *
 *     const kdf = newKdfParams();                          // random salt, chosen params
 *     const key = await deriveKey(passphrase, kdf);        // 32 bytes, in memory only
 *     const job = await generate(API, { prompt, ... }, { key, kdf });
 *     ...
 *     const { blob } = await fetchAndDecrypt(API, job.jobId, key);
 *     video.src = URL.createObjectURL(blob);
 *
 * WHICH KDF, AND WHY THE DEFAULT IS PBKDF2
 * ----------------------------------------
 * Argon2id is the better password KDF and the format supports it: `kdf.name` is recorded
 * in the artefact's authenticated header, and the backend never runs a KDF at all - it
 * only stores the parameters so the same passphrase derives the same key tomorrow. So
 * using Argon2id costs the server nothing.
 *
 * It costs the *client* a WebAssembly dependency, because no browser ships Argon2id
 * natively. This repository has no frontend build step and no package.json, so the
 * default here is PBKDF2-HMAC-SHA256 at 600,000 iterations - OWASP's current floor -
 * which WebCrypto implements natively.
 *
 * To use Argon2id instead, pass an implementation:
 *
 *     import { argon2id } from "hash-wasm";
 *     const key = await deriveKey(passphrase, kdf, { argon2id });
 *
 * with `kdf.name === "argon2id"`. Nothing else changes, and artefacts encrypted either
 * way remain decryptable, because each one records how its own key was derived.
 */

/* ------------------------------------------------------------------------------------
 * Container format - mirrors artifacts.py. See docs/confidential-generation.md.
 * ---------------------------------------------------------------------------------- */

const MAGIC = "CGEN";
const FORMAT_VERSION = 1;
const SUITES = { 1: "AES-256-GCM" };
const PREAMBLE_BYTES = 8;
const MAX_HEADER_BYTES = 8192;
const NONCE_BYTES = 12;
const TAG_BYTES = 16;
const KEY_BYTES = 32;

/*
 * The message a user sees when decryption fails, whatever the reason.
 *
 * A wrong passphrase, a truncated download, a modified ciphertext and a swapped object
 * are the same event to AES-GCM - authentication failed - and telling them apart for the
 * user would mean telling an attacker apart too. One message, no detail.
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
 * Fresh KDF parameters for a new generation: a random salt plus the chosen cost.
 *
 * The salt is not secret. It exists so that two users with the same passphrase do not get
 * the same key, and so that a precomputed table cannot be reused across artefacts. It is
 * stored with the artefact and returned by the status API, which is exactly what lets the
 * same passphrase reproduce the key later.
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
 * Derive a 256-bit key from a passphrase.
 *
 * Returns raw bytes rather than a `CryptoKey` because the key has two jobs: it is sent to
 * the backend once (so it must be exportable) and used locally to decrypt. Call
 * `wipe(key)` when finished - see the honesty note on that function.
 */
export async function deriveKey(passphrase, kdf = DEFAULT_KDF, options = {}) {
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
        "This artefact was encrypted with an Argon2id-derived key, but no Argon2id " +
          "implementation was supplied. Pass one as options.argon2id - for example " +
          "hash-wasm's argon2id. No browser provides it natively."
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
 * runtime made. This zeroes the one buffer you hold. It does not guarantee the key is
 * gone from the process.
 */
export function wipe(bytes) {
  if (bytes instanceof Uint8Array) bytes.fill(0);
}

/* ------------------------------------------------------------------------------------
 * Generation
 * ---------------------------------------------------------------------------------- */

/**
 * Submit a generation. With `key` and `kdf`, it is confidential; without them, standard.
 *
 * The passphrase is not a parameter of this function on purpose. It is not sent, and the
 * signature should not suggest it could be.
 */
export async function generate(apiBase, request, { key, kdf, keyId, fetchImpl = fetch } = {}) {
  const body = { ...request };

  if (key) {
    if (!(key instanceof Uint8Array) || key.length !== KEY_BYTES) {
      throw new Error(`Encryption key must be ${KEY_BYTES} raw bytes.`);
    }
    body.privacyMode = "confidential";
    body.encryption = {
      algorithm: "AES-256-GCM",
      key: toBase64Url(key),
      ...(keyId ? { keyId } : {}),
      ...(kdf ? { kdf } : {})
    };
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
 * until `decryptArtifact` succeeds, treat every field here as unverified.
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
  if (version !== FORMAT_VERSION) {
    throw new DecryptionError(
      `This artefact uses container version ${version}; this client understands ${FORMAT_VERSION}. ` +
        "Update the page."
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

  return {
    version,
    suite,
    algorithm: SUITES[suite],
    header,
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
 * Pass `expectArtifactId` whenever the caller knows which generation it asked for. The
 * header is authenticated, so a mismatch means the platform served a different artefact -
 * a real substitution, not a corrupted download - and that is worth refusing.
 */
export async function decryptArtifact(buffer, key, { expectArtifactId } = {}) {
  if (!(key instanceof Uint8Array) || key.length !== KEY_BYTES) {
    throw new DecryptionError("The decryption key is the wrong size.");
  }

  const container = parseContainer(buffer);

  const cryptoKey = await crypto.subtle.importKey("raw", key, "AES-GCM", false, ["decrypt"]);

  let plaintext;
  try {
    plaintext = await crypto.subtle.decrypt(
      {
        name: "AES-GCM",
        iv: container.nonce,
        additionalData: container.additionalData,
        tagLength: TAG_BYTES * 8
      },
      cryptoKey,
      container.ciphertextWithTag
    );
  } catch {
    // Deliberately opaque. See DECRYPT_FAILURE_MESSAGE.
    throw new DecryptionError();
  }

  // Only now is the header trustworthy: it was the AEAD's associated data, so reaching
  // this line proves nobody altered it.
  if (expectArtifactId && container.header.artifactId && container.header.artifactId !== expectArtifactId) {
    throw new DecryptionError(
      "This artefact decrypted correctly but belongs to a different generation."
    );
  }

  return {
    blob: new Blob([plaintext], { type: container.header.contentType || "video/mp4" }),
    header: container.header
  };
}

/**
 * Fetch a job's artefact and decrypt it. The whole file is downloaded first.
 *
 * That is a real limitation and it is the right trade for version 1: a single GCM tag over
 * the whole file cannot be verified until the last byte arrives, so playback cannot start
 * early. For a five-second H3 clip - a couple of megabytes - it is not noticeable. The
 * format is versioned precisely so a chunked variant can lift this later without breaking
 * anything already encrypted.
 */
export async function fetchAndDecrypt(apiBase, jobId, key, { fetchImpl = fetch, signal } = {}) {
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
  const { blob, header } = await decryptArtifact(buffer, key, { expectArtifactId: jobId });
  return { blob, header, encrypted: true };
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
