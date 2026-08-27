# Confidential Generation

Optional per-request privacy mode. The generated video is encrypted **inside the inference
container**, with a file key the worker generates and wraps to a public key the caller
supplied, before it is handed to persistent storage. R2 receives ciphertext. The browser
decrypts for playback.

Standard generation is unchanged. A request with no `privacyMode` behaves exactly as it did
before this existed.

**Crypto v2 (hybrid) is current.** v1 (symmetric) artefacts remain readable; new v1 jobs are
refused. See §14.

---

## 1. What this is, precisely

> Confidential Generation uses hybrid encryption. The inference worker receives an
> encryption-only public key and generates a unique random file-encryption key for each
> video. The video is encrypted with authenticated symmetric encryption, and the file key is
> wrapped using the user's public key before persistent storage. The user's passphrase,
> private key and password-derived key are not sent to the inference job. Plaintext prompts
> and generated media necessarily exist transiently within the trusted inference environment
> while generation and encryption occur.

### What it is not

This is **not zero-knowledge**, and nothing in this system should be described that way. The
model needs the plaintext prompt to generate, and it produces plaintext frames. Both exist,
in the clear, inside the inference process while it runs — as does the file key, for the
moment it is in use.

Language to avoid, because it would be untrue here:

| Do not say | Why it is false |
|---|---|
| "zero knowledge" | The inference environment sees the prompt and the video. |
| "end-to-end encrypted AI inference" | There is no end-to-end path that excludes the model. |
| "the AI never sees your prompt" | It cannot generate without it. |
| "the server never sees plaintext" | The GPU worker does, for the duration of the job. |
| "RunPod can never access plaintext during inference" | It runs the inference. |
| "impossible for infrastructure to access" | A compromised inference process could read it. |

The accurate claim is narrower and defensible:

> **A complete copy of the RunPod job input, the RunPod job output, Cloudflare application
> state and R2 storage is not enough to decrypt a Confidential Generation v2 artefact
> without the user's passphrase or an unlocked private key.**

---

## 2. Why v2 exists

v1 sent an AES-256 key in the generation request. That key travelled through the job queue,
which retains job records, producing this property:

```
RunPod retained job payload          R2
    AES file key            +    encrypted video
                     |
                     v
             decryption possible
```

v2 removes it. The request carries a public key instead, and the file key is created inside
the worker and never leaves it in usable form:

```
RunPod retained job payload          R2 / platform storage
    PUBLIC key only          +   encrypted video
                                  wrapped file key
                                  public metadata
                     |
                     v
          still cannot decrypt anything

  user passphrase -> KEK -> private key -> unwrap file key -> decrypt video
```

---

## 3. Architecture

### Standard generation

```
Browser  →  Worker  →  RunPod  →  H3  →  MP4  →  R2  →  Browser
```

### Confidential generation (v2)

```
BROWSER
   |  passphrase ──KDF──▶ keyEncryptionKey ──AES-256-GCM──▶ encrypted privateKey
   |  (none of those three ever leave this box)
   |
   |  publicKey only
   v
CLOUDFLARE WORKER
   |  validates; rejects any field that could decrypt; forwards once; stores nothing
   v
RUNPOD JOB QUEUE            ← treat as permanently retained
   |  prompt, settings, publicKey, upload authorisation
   v
RUNPOD INFERENCE WORKER (trusted plaintext boundary)
   |  H3 inference
   |  plaintext MP4, transiently, on ephemeral local disk
   |  fileEncryptionKey ← CSPRNG, 256 bits, fresh per artefact
   |  AES-256-GCM encrypt
   |  RSA-OAEP-256 wrap fileEncryptionKey to publicKey
   |  destroy fileEncryptionKey, delete plaintext
   v
encrypted container ── PUT ──▶ Worker ──stream──▶ R2
                                 |  verifies it really is a v2 container for this job
                                 v
                              BROWSER
                                 |  passphrase → KEK → private key
                                 |  unwrap fileEncryptionKey → AES-GCM decrypt
                                 v
                              playback
```

### The five keys, named

Conflating any two of these is how such systems go wrong, so the code names each explicitly.

| Name | What it is | Lives where |
|---|---|---|
| `passphrase` | typed by the user | browser only, transiently |
| `keyEncryptionKey` (KEK) | 256-bit, derived from the passphrase | browser only, transiently |
| `privateKey` | RSA-3072, unwraps file keys | browser; stored **encrypted** under the KEK |
| `publicKey` | RSA-3072, wraps file keys | sent with the request; not secret |
| `fileEncryptionKey` (FEK) | 256-bit, random, one per artefact | inference worker, transiently; persisted **wrapped only** |

---

## 4. Cryptographic primitives, and why

| Purpose | Primitive |
|---|---|
| Content encryption | AES-256-GCM, 96-bit random nonce, 128-bit tag |
| Key wrapping | RSA-OAEP, SHA-256, MGF1-SHA-256, empty label, ≥3072-bit modulus |
| Private-key protection | AES-256-GCM under a passphrase-derived key |
| Passphrase KDF | PBKDF2-HMAC-SHA256, 600,000 iterations (Argon2id supported) |
| Public key encoding | SPKI DER, base64url |
| Private key encoding | PKCS#8 DER, encrypted, base64url |

### Why RSA-OAEP and not ECIES or HPKE

The priority order given was security, browser interoperability, Python interoperability,
maturity, maintainability, auditability. RSA-OAEP wins on the last four and ties on the
first.

**HPKE (RFC 9180)** is the most modern answer and was rejected on availability: `cryptography`
does not implement it and no browser exposes it through WebCrypto. Adopting it would mean a
new dependency on *both* sides, one of them WebAssembly in a repository with no frontend
build step.

**X25519 or P-256 ECIES** is available natively on both sides and has far smaller keys. It
was rejected on composition risk. An ECIES wrap is not one operation; it is generate an
ephemeral key pair, do the agreement, run a KDF with correct domain separation, encrypt
under the derived key, and serialise the ephemeral public key alongside the result — five
or six steps that must all be right, written by us, sitting inside the most
security-critical function in the system. RSA-OAEP is a single library call on each side:

```python
public_key.encrypt(file_key, padding.OAEP(mgf=MGF1(SHA256()), algorithm=SHA256(), label=None))
```

```js
crypto.subtle.decrypt({ name: "RSA-OAEP" }, privateKey, wrapped)
```

Those defaults match, which is why the two interoperate with nothing to negotiate. This was
verified before any of it was built, not assumed.

The cost is key size — a 3072-bit SPKI is 422 bytes, about 563 base64url characters, and a
wrap is 384 bytes — and key generation, which is ~130 ms. Both are irrelevant here: the
public key travels once per job in a payload that already contains a workflow graph, and key
generation happens **once per user**, not per video.

### Why 3072 bits

3072-bit RSA is the NIST SP 800-57 128-bit-security level and is what "modern RSA" means. Be
precise about what that implies: the wrap sits at the 128-bit security level while the
content encryption is AES-256, so 128 bits is the binding constraint. That is considered
adequate well beyond 2030.

It is enforced as a **floor, not a fixed size**. A client that prefers 4096 simply generates
4096 and nothing in the backend changes; the modulus is read from the key itself. Keys below
3072 bits are refused.

---

## 5. The container format

Same framing as v1, so one parser reads both. All integers big-endian:

```
offset            size          field
0                 4             MAGIC = "CGEN"
4                 1             CONTAINER VERSION  (1 = symmetric, 2 = hybrid)
5                 1             SUITE = 1  (AES-256-GCM)
6                 2             HEADER_LEN (uint16, max 8192)
8                 HEADER_LEN    HEADER — canonical UTF-8 JSON, also the AEAD's
                                additional authenticated data
8 + HEADER_LEN    12            NONCE
20 + HEADER_LEN   ...           CIPHERTEXT ‖ TAG (tag is the last 16 bytes)
```

A v2 header:

```json
{
  "v": 2,
  "alg": "AES-256-GCM",
  "artifactId": "5f7c…",
  "contentType": "video/mp4",
  "plaintextBytes": 2196233,
  "privacyMode": "confidential",
  "createdAt": "2026-08-28T12:00:00Z",
  "kw": {
    "alg": "RSA-OAEP-256",
    "wrappedFileKey": "<base64url, 384 bytes wrapped>",
    "keyId": "<truncated SHA-256 of the SPKI>"
  }
}
```

### What is cryptographically bound

The header **is** the associated data, so every field above is authenticated. Concretely,
none of these can be altered without the file key: the container version, the cipher name,
the generation id, the plaintext media type and length, the privacy mode, the key-wrap
algorithm, the key id, **and the wrapped file key itself**.

That last one is the reason the wrapped key lives *in* the header rather than beside it. An
attacker who substitutes a file key wrapped to their own public key does not get a readable
video — the swap invalidates the tag over the ciphertext. The container is bound to one key
pair and one generation, not merely accompanied by claims about them.

Nothing unstable is authenticated. Everything in the header is fixed at creation, so a
legitimate artefact can always be reopened.

### What is *not* in a v2 container

No KDF metadata. The passphrase KDF protects the caller's **private key**, which is
account-scoped and never reaches the platform. Repeating it inside every video would imply
the platform had a role in deriving it. It lives in the key bundle instead (§6).

### Versioning

Two independent fields — container version for framing, `SUITE` for the cipher — so a future
XChaCha20-Poly1305 or chunked-streaming variant is additive. Nothing decides how to decrypt
from a file extension; the magic bytes and version byte do.

---

## 6. The key bundle

The client holds a bundle. It is safe to store anywhere — a file, a synced note,
`localStorage`, a future account-scoped store — because the private key inside it is
ciphertext.

```json
{
  "v": 1,
  "keyId": "<truncated SHA-256 of the SPKI>",
  "publicKey": "<base64url SPKI>",
  "publicKeyAlgorithm": "RSA-OAEP-256",
  "encryptedPrivateKey": "<base64url AES-256-GCM ciphertext‖tag over PKCS#8>",
  "privateKeyNonce": "<base64url 12 bytes>",
  "kdf": { "name": "pbkdf2-sha256", "salt": "<base64url 16 bytes>", "parameters": { "iterations": 600000 } },
  "createdAt": "2026-08-28T12:00:00Z"
}
```

The public key, key id, algorithm and KDF parameters are bound into the private key's
authentication tag. Without that, someone able to edit a bundle at rest could substitute
their own public key and have every future video silently encrypted to a key they hold. With
it, that edit makes the unlock fail.

`changeBundlePassphrase()` re-seals under a new passphrase **and a new salt** — reusing the
salt would tell anyone holding both versions that they protect the same key.

### Key scope: account-wide, but there are no accounts

The default preference is a user-scoped key pair with a unique random FEK per video, and
that is what this implements — with the caveat that **this system has no user identity**.
There is nowhere to attach an account-scoped key to.

So the client owns the bundle, and **the platform stores no private key material at all,
not even encrypted**. That is strictly stronger than storing it server-side and costs
nothing today. The trade is cross-device recovery: moving to a new browser means moving the
bundle, which is why the reference UI has an export button and says plainly what losing it
means.

The migration path is already open: the bundle format is defined, and an account-scoped
store would simply persist this same JSON per user. That is a change to where a blob lives,
not to the cryptography.

Per-generation key pairs were considered and rejected: they would compartmentalise better
but require the client to retain one encrypted private key per video, turning "remember one
passphrase" into "never lose any of N key blobs" — a worse failure mode for the same threat
model, given the FEK is already unique per video.

---

## 7. Trust boundaries

### Inside — plaintext exists here

| Where | What | For how long |
|---|---|---|
| Browser tab | passphrase, KEK, private key, decrypted video | while in use |
| RunPod GPU worker process | prompt, model state, decoded frames, **the FEK** | the job |
| RunPod container filesystem | the MP4 from `SaveVideo` | seconds, until encryption completes |

### Outside — ciphertext and public metadata only

| Where | What it holds |
|---|---|
| RunPod job input | prompt, settings, **public key**, upload authorisation |
| RunPod job output | generation id, status, artefact location, algorithm names, timings |
| Cloudflare R2 | the encrypted container (which contains the wrapped FEK) |
| R2 object metadata | privacy mode, crypto version, algorithms, key id, media type, expiry |
| Durable Object | the same non-secret description, plus progress |
| Cloudflare logs | generation id, privacy mode, dimensions, algorithm, prompt length and digest |

### What RunPod can still see

Stated plainly, because it is the honest half of the claim. RunPod's control plane may
retain the plaintext prompt, the generation settings, the public key, job metadata,
encrypted-artefact metadata and the output JSON. The inference worker additionally sees,
transiently, the model's intermediate state, the plaintext video and the random FEK while it
is in use.

What RunPod does **not** receive and cannot retain through the job payload: the passphrase,
the KEK, the private key, or any long-lived key capable of decrypting a stored artefact.

### Threats addressed

- R2 bucket compromise, accidental public exposure, storage backups, leaked object URLs — all
  yield ciphertext.
- **Retained RunPod job payloads** — now the point of the design: the payload holds a public
  key, which cannot decrypt.
- Database/job-state leakage — no key material is stored there.
- Logs — no key, no passphrase, no prompt text, no video bytes.
- Long-lived plaintext on the worker — destroyed as soon as a container exists, and again in
  a `finally` on every failure path.
- Ciphertext or metadata modification — authentication fails.
- A substituted artefact, or a substituted *wrapped key* — both detected.
- Nonce and key reuse across retries — both drawn fresh per attempt.

### Threats not addressed

- **Malicious code running inside the inference process.** It holds the plaintext and the FEK
  by construction. Nothing here constrains it.
- Compromised GPU firmware or hypervisor.
- A malicious RunPod host administrator with access to container memory or disk.
- A compromised browser or a malicious page — that is where the private key lives.
- A weak passphrase. The KDF raises the cost of guessing; it cannot fix low entropy.
- Making plaintext unnecessary for inference. It is not.
- A user who loses the passphrase *and* the bundle. See §12.

---

## 8. Fail-closed guarantees

| Situation | Outcome |
|---|---|
| Confidential with no `encryption` block | 400 |
| **A symmetric `encryption.key`, or `version: 1`** | **400 naming the v2 replacement** |
| Any field that could decrypt (`passphrase`, `privateKey`, `kek`, `fek`, …) | **400, at any depth** |
| Public key not SPKI, wrong type, or under 3072 bits | 400 |
| `keyId` disagreeing with the supplied public key | 400 |
| Unsupported algorithm, key-wrap algorithm, or version | 400 |
| Unknown privacy mode | 400 |
| `private` / `ephemeral` | 501 with a reason |
| `standard` **with** an `encryption` block | 400 |
| `JOB_TOKEN_SECRET` unset | 503; the job is never submitted |
| FEK generation or wrapping fails | job fails, nothing uploaded |
| Encryption fails | job fails, nothing uploaded, plaintext deleted |
| Upload fails | job fails, plaintext and ciphertext both deleted |
| Plaintext MP4 uploaded for a confidential job | **422, nothing written** |
| **A v1 container uploaded for a v2 job** | **422, nothing written** |
| A v2 container with no/invalid `kw` block | **422, nothing written** |
| Container header naming a different generation | **422, nothing written** |

### How the upload gate is enforced

The job's privacy mode **and crypto version** are signed into the output token:

```
token payload = { jid: <job id>, pur: "output-upload", exp: …, pm: "confidential", cv: 2 }
```

When the upload arrives, the Worker does not ask the uploader what kind of job this was and
does not consult a Durable Object that could be stale. It reads both claims out of the
HMAC-signed token the uploader had to present. RunPod cannot change them without the signing
secret, which never leaves Cloudflare.

It then peeks at the header only — never the payload, so memory stays flat — and requires a
container of exactly that version whose authenticated header names exactly that job. There
is no override, query parameter or header that relaxes this.

---

## 9. Lifecycles

### Plaintext video

1. `SaveVideo` writes the MP4. Unavoidable: video tooling writes files.
2. The protector creates a private scratch directory (`mkdtemp`, mode `0700`, unpredictable
   name) and streams the MP4 into a container in 4 MB chunks under a fresh `uuid4` name.
3. The moment a complete container exists, the plaintext is overwritten with random bytes
   and unlinked.
4. The container is uploaded.
5. `finally`: the scratch directory is removed — success, failure or exception.

`H3_KEEP_OUTPUTS=1` is a debugging convenience for standard mode and is **deliberately
ignored** for confidential artefacts. There is a test for that, and another asserting
standard mode still honours it.

An artefact whose path resolves outside the worker's own output directories is **refused
before encryption** rather than encrypted and shredded — the path comes from ComfyUI's
history entry, which is trustworthy in normal operation and exactly the wrong thing to trust
when the next step is an unconditional delete.

### File-encryption key

```
secrets.token_bytes(32) → AES-256-GCM encrypt → RSA-OAEP wrap → destroy()
```

Never logged, never written to disk, never returned, never in R2 metadata, never in the job
output. Destroyed in a `finally`, so it is gone before the upload starts — on the failure
paths as well as the happy one. Its only persistent form is `wrappedFileKey` inside the
authenticated header.

### Private key

Generated extractable exactly once, because it must be exported to be encrypted. Thereafter
imported **non-extractable**, so an unlocked bundle cannot be turned back into exportable key
material by page script. Never transmitted.

### On overwriting before deletion

`H3_OVERWRITE_PLAINTEXT=1` overwrites before unlinking. Be clear about what that is worth:
occasionally useful on plain ext4, close to nothing on overlayfs, copy-on-write filesystems
or SSDs with wear levelling — which is where these containers actually run. **Guaranteed
secure erasure of underlying host storage is not achievable and is not claimed.** The
controls that matter are the short lifetime and the destruction of the worker.

---

## 10. Retries and idempotency

RunPod can retry a job. Every attempt draws a **fresh FEK and a fresh nonce**, both from a
CSPRNG. Neither is derived from the generation id, the seed, the job id or a timestamp — a
deterministic object path makes "reuse it, the job is the same" an easy mistake, and it is
the one fatal misuse of GCM. A test encrypts the same plaintext under the same generation id
sixteen times and asserts sixteen distinct file keys, nonces and ciphertexts.

R2 `put` is atomic, so a retry replaces the object wholesale; there is no partial artefact to
mistake for a complete one, and a truncated container fails authentication anyway.

---

## 11. Storage

```
inputs/{jobId}/{assetId}.{png|jpg|webp}
outputs/{jobId}/video.mp4        standard
outputs/{jobId}/artifact.enc     confidential — ciphertext
```

R2 holds the encrypted container (including the wrapped FEK), plus non-secret metadata:
privacy mode, crypto version, content algorithm, key-wrap algorithm, key id, original media
type, and an advisory expiry.

R2 never holds: a plaintext MP4, a passphrase, a KEK, a plaintext private key, or a plaintext
FEK. Object names are opaque and contain no prompt.

### Retrieval

| | Standard | Confidential |
|---|---|---|
| Content-Type | `video/mp4` | `application/octet-stream` |
| Cache-Control | `private, max-age=3600, immutable` | `private, no-store` |
| Disposition | inline | `attachment` + `nosniff` |
| Crypto metadata | — | `X-Artifact-*`, CORS-exposed |
| Range | supported | supported |

---

## 12. Recovery: there is none

If the passphrase is lost and no bundle was exported under a known one, the encrypted private
key cannot be unlocked, so the wrapped file key cannot be unwrapped, so the video cannot be
decrypted. Nothing the platform stores changes that.

This is a **security property, not a defect**, and it is what makes §1's claim true. There is
deliberately no master key, no admin override, no recovery copy of a plaintext private key,
no server-side passphrase copy and no escrow. Adding any of those would recreate exactly the
capability this design removes, and would need separate threat modelling and explicit
approval.

---

## 13. Retention

`retentionSeconds` or `expiresAt` may be supplied, are validated, resolve to an absolute
timestamp, are recorded in object metadata and reported by `/status` — and are **advisory**.
Enforcement today is the prefix-scoped R2 lifecycle policy (`scripts/set_r2_lifecycle.sh`),
which is age-based and cannot express a per-object lifetime.

A Worker-internal pseudo-scheduler is deliberately not implemented: a timer that fires only
while requests happen to arrive is worse than no promise at all. The documented path is a
Cloudflare Cron Trigger that lists `outputs/`, reads each object's `expiresAt`, and calls the
same `deleteGeneration()` the API uses.

---

## 14. Migration and compatibility

| | v1 (symmetric) | v2 (hybrid) |
|---|---|---|
| Request carries | a 256-bit AES key | an RSA public key |
| File key origin | derived from the passphrase | random, generated in the worker |
| Readable | **yes** | yes |
| Creatable | **no** | yes |

- **Existing v1 artefacts remain decryptable.** The parser accepts both versions, the browser
  module dispatches on the version byte, and the v1 conformance vector still round-trips.
- **New v1 jobs are refused** with a 400 naming the replacement. A compatibility path would
  preserve the property v2 exists to remove, so there is no flag to re-enable it.
- **Standard mode is untouched** and does not go through any of this code.
- Status records written before privacy modes existed still read as `standard` /
  `encrypted: false`.

There is no data migration. Nothing is re-encrypted, and no stored object changes.

### Deletion

`DELETE /jobs/:id/artifact` removes the artefact; `DELETE /jobs/:id` also removes that
generation's input keyframes. One code path serves both privacy modes and both crypto
versions. The wrapped FEK is inside the container, so deleting the object deletes it.

Nothing deletes a key bundle: it is client-side, account-scoped in intent, and other videos
depend on it. If key pairs later become per-generation, that changes — and the decision is
recorded in §6 so the reasoning is not lost.

---

## 15. Answers to the questions people actually ask

**Does the RunPod job payload contain any value capable of decrypting a stored v2 video?**
No. It contains a public key, which can only encrypt.

**Can the public key decrypt a stored video?**
No. It has no private exponent. WebCrypto will not even import it with a `decrypt` usage.

**Does RunPod receive the user's passphrase?** No.
**Does RunPod receive the user's private key?** No.
**Does RunPod receive the KEK?** No.

**Where is the per-video AES FEK generated?**
Inside the RunPod inference worker, from a CSPRNG, fresh for every artefact and every retry.

**Does the plaintext FEK ever reach R2?**
No. Only its wrapped form, inside the container's authenticated header.

**Does the plaintext FEK ever appear in RunPod output?**
No. The job result carries filenames, sizes, the storage key, algorithm names and timings.

**What exactly is stored in R2?**
The encrypted container — magic, version, suite, authenticated header (including the wrapped
FEK), nonce, ciphertext, tag — plus non-secret object metadata: privacy mode, crypto version,
algorithms, key id, original media type, advisory expiry.

**What exactly is in the RunPod input?**
The workflow graph (which contains the prompt), generation settings, `privacy.mode`, the
`encryption` block (version, algorithm, key-wrap algorithm, public key, key id), and the
job-scoped callback URLs and tokens.

**What exactly is returned in the RunPod output?**
Generation id, status, artefact filename and size, storage key, privacy mode, crypto version,
algorithm names, and phase timings.

**What plaintext exists during generation?**
The prompt, the model's intermediate state, the decoded frames, the MP4 on the container
filesystem for a few seconds, and the FEK while it is in use.

**When is the plaintext MP4 deleted?**
The moment a complete container exists — before the upload starts — and again in a `finally`
on every failure path.

**What happens if encryption fails?** The job fails, nothing is uploaded, the plaintext is
deleted. There is no plaintext fallback.

**What happens if key wrapping fails?** The job fails before any ciphertext is written.
Wrapping happens first precisely so this costs nothing.

**What happens if upload fails?** The job fails; plaintext and ciphertext are both deleted.
A retry re-encrypts under a fresh FEK and nonce.

**Can somebody with the full R2 bucket and RunPod job JSON decrypt a video?**
No. This is tested directly, from both languages: every value from the job input, job output,
stored object, metadata, job state, status response, public key, wrapped key and container
header is turned into a candidate key and none of them decrypts.

**What additional information is required to decrypt it?**
The user's passphrase plus their key bundle — or an already-unlocked private key.

**What happens if the user loses the passphrase?** The video is unrecoverable. See §12.

**Can existing Confidential v1 videos still be decrypted?** Yes. See §14.

**Can Standard videos still stream normally?** Yes — same key, same content type, same cache
headers, same Range support.

---

## 16. Performance

Measured on the development machine. The asymmetric operation applies only to a 32-byte file
key, so it disappears next to everything else.

| Operation | Median |
|---|---|
| RSA-OAEP-3072 wrap (worker) | **0.05 ms** |
| `protect()` — 2 MB, a 5 s clip | 13.6 ms |
| `protect()` — 10 MB | 34.9 ms |
| `protect()` — 50 MB | 152 ms |
| Container overhead | 789 bytes fixed |
| RSA-3072 keygen (browser, **once per user**) | 129 ms |
| `createKeyBundle` incl. 600k PBKDF2 | 311 ms |
| Unlock private key (PBKDF2-dominated) | 236 ms |
| Unwrap file key (browser) | 1.65 ms |
| AES-GCM decrypt 2 MB (browser) | 2 ms |
| AES-GCM decrypt 50 MB (browser) | 62 ms |

`protect()` at 2 MB is unchanged from v1's 13.6 ms — the wrap is 0.4% of it. Against a
measured H3 generation of 20–40 s, encrypting a five-second clip is roughly **0.03–0.07%** of
job wall time. Ciphertext is the same length as the video; AES-GCM is length preserving.

The `[perf]` line carries `privacy=` and `encryption_ms=`.

---

## 17. Streaming

Version 1 of the format downloads the whole ciphertext before playback: a single GCM tag
covers the entire file, so nothing can be authenticated until the last byte arrives, and
playing unauthenticated plaintext would defeat the point. At 2 MB and 2 ms of decryption this
is not noticeable; it is a real limit for long videos.

The format is versioned so this can be lifted without breaking any existing artefact. A
chunked variant would be a new `SUITE`, with the chunk index and total bound into each
chunk's associated data so a reordered, duplicated or dropped chunk fails authentication.
**Deliberately not built yet** — chunked AEAD protocols are easy to get subtly wrong, and
correctness for the current use case matters more.

---

## 18. Secrets in memory

Perfect erasure is not achievable in Python, JavaScript or V8, and this document will not
claim it.

What is done: keys are request-scoped; `EphemeralKey` wraps bytes in a type whose `__repr__`,
`__str__` and `__format__` are redacted, so an f-string, traceback or debug dump cannot print
one by accident; `destroy()` zeroes the backing `bytearray` and `wipe()` does the same in the
browser; unlocked private keys are non-extractable; nothing is serialised or logged.

What is not: CPython may already have copied the bytes into an intermediate `bytes`, a freed
buffer, the page cache or swap, and offers no way to find or erase those. JavaScript engines
move memory freely. These narrow the window; they do not close it.

---

## 19. Recommended next hardening work

1. **Authenticate the public API.** There is no caller identity, so there is no per-user
   authorization on retrieval or deletion. The artefact is ciphertext, but its existence and
   size are readable by anyone holding the job id. This is now the largest gap.
2. **An account-scoped bundle store**, so a user can reach their videos from a second device
   without moving a file by hand. The format is ready; it needs identity first, which is why
   it follows item 1.
3. **Argon2id by default**, once a frontend build step can carry the WASM. No backend change
   is required — the KDF is entirely client-side.
4. **Per-object retention enforcement** via a Cron Trigger, which also unblocks `private` and
   `ephemeral`.
5. **Reduce prompt retention in the job queue.** The prompt still travels in plaintext because
   the model needs it; RunPod's own job-retention controls are the lever.
6. **Chunked AEAD** if long videos become a product requirement.
