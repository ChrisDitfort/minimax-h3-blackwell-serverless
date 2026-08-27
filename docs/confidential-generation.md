# Confidential Generation

Optional per-request privacy mode. The generated video is encrypted **inside the inference
container**, with a key the caller derived, before it is handed to persistent storage. R2
receives ciphertext. The browser decrypts for playback.

Standard generation is unchanged. A request with no `privacyMode` behaves exactly as it did
before this existed.

---

## 1. What this is, precisely

> Confidential Generation encrypts generated media before persistent storage using
> encryption material derived from credentials controlled by the user. Persistent storage
> receives encrypted media rather than the plaintext video. Plaintext prompts and generated
> media necessarily exist transiently within the trusted inference environment while the AI
> model performs generation and before the resulting media is encrypted.

### What it is not

This is **not zero-knowledge**, and nothing in this system should be described that way. The
model needs the plaintext prompt to generate, and it produces plaintext frames. Both exist,
in the clear, inside the inference process while it runs.

Language to avoid, because it would be untrue here:

| Do not say | Why it is false |
|---|---|
| "zero knowledge" | The inference environment sees the prompt and the video. |
| "end-to-end encrypted AI inference" | There is no end-to-end path that excludes the model. |
| "the AI never sees your prompt" | It cannot generate without it. |
| "the server never sees plaintext" | The GPU worker does, for the duration of the job. |
| "impossible for infrastructure to access" | A compromised inference process could read it. |

The accurate claim is narrower and defensible: **plaintext exists transiently only where
inference requires it; persistent storage contains ciphertext only.**

---

## 2. Architecture

### Standard generation

```
Browser  →  Worker  →  RunPod  →  H3  →  MP4  →  R2  →  Browser
```

### Confidential generation

```
Browser
   |  passphrase → Argon2id / PBKDF2 → 256-bit key      (the passphrase stays here)
   |
   |  derived encryption key
   v
Cloudflare Worker
   |  validates; forwards once; stores nothing
   |
   v
RunPod (trusted inference boundary)
   |  prompt plaintext
   |  AI inference
   |  plaintext MP4, transiently, on ephemeral local disk
   |  AES-256-GCM encryption
   |  plaintext destroyed
   v
Encrypted artefact  ── PUT ──▶  Worker  ── stream ──▶  R2
                                  |  verifies it really is ciphertext
                                  |  ciphertext only
                                  v
                               Browser
                                  |  passphrase → KDF → key
                                  |  local decryption
                                  v
                            Video playback
```

The bytes are already ciphertext when they leave RunPod. The Worker streams them through
without buffering and without the ability to decrypt them: it does not hold the key at that
point, and never did after the submission request finished.

---

## 3. Trust boundaries

Stated explicitly, including the parts that are uncomfortable.

### Inside the trust boundary — plaintext exists here

| Where | What | For how long |
|---|---|---|
| RunPod GPU worker process | the prompt, model activations, decoded frames | the job |
| RunPod container filesystem | the MP4 written by ComfyUI's `SaveVideo` | from write until encryption completes — seconds |
| Browser tab | the passphrase, the derived key, the decrypted video | while the tab is open |

### Outside the trust boundary — ciphertext only

| Where | What it holds |
|---|---|
| Cloudflare R2 | the encrypted container. No key, no passphrase, no plaintext. |
| R2 object metadata | privacy mode, algorithm, format version, original media type, advisory expiry. |
| Durable Object (job state) | the same non-secret description, plus progress. |
| Cloudflare logs | generation id, privacy mode, dimensions, algorithm, prompt length and digest. |

### Threats this addresses

- **R2 bucket compromise** — the attacker gets ciphertext and cannot decrypt it.
- **An accidentally public object** — likewise.
- **Storage backups** — they contain ciphertext.
- **A leaked object URL** — it serves ciphertext.
- **Database / job-state leakage** — no key material is stored there.
- **Logs containing sensitive data** — no key, no passphrase, no prompt text, no video bytes.
- **Long-lived plaintext on the worker** — destroyed as soon as a complete container exists,
  and again in a `finally` on every failure path.
- **Ciphertext modification** — AES-GCM authentication fails; the browser refuses it.
- **A substituted artefact** — the header is authenticated and names the generation, so an
  object from a different job is detected, not merely mis-played.
- **Nonce reuse** — a fresh 96-bit nonce is drawn per encryption, never derived from the
  job id.
- **Abandoned or failed jobs** — cleanup runs in `finally`; nothing is uploaded on failure.

### Threats this does **not** address

These are real. They are not solved here, and pretending otherwise would be the failure.

- **Malicious code running inside the inference process.** It holds the plaintext and the
  key by construction. Nothing in this design constrains it.
- **Compromised GPU firmware or hypervisor.**
- **A malicious RunPod host administrator** with access to container memory or disk.
- **The key's presence in the RunPod job payload.** The Worker must send the key to the
  worker that will encrypt, and RunPod holds the job record for its retention window
  (`policy.ttl`, currently 30 minutes). During that window the key exists in a third-party
  system. This is the single largest residual exposure and is called out again in §11.
- **Making plaintext unnecessary for inference.** It is not.
- **Homomorphic video generation.** Not a thing at this scale.
- **A user who loses the passphrase.** By design, see §10.

---

## 4. The encrypted container format

Self-describing, so a file recovered on its own — from a backup, a download folder, an
object listing — can be identified and decrypted with nothing but the passphrase.

All integers big-endian:

```
offset            size          field
0                 4             MAGIC = "CGEN"
4                 1             FORMAT_VERSION = 1
5                 1             SUITE = 1  (AES-256-GCM, 96-bit nonce, 128-bit tag)
6                 2             HEADER_LEN (uint16, max 8192)
8                 HEADER_LEN    HEADER — canonical UTF-8 JSON, also the AEAD's
                                additional authenticated data
8 + HEADER_LEN    12            NONCE
20 + HEADER_LEN   ...           CIPHERTEXT ‖ TAG (the tag is the last 16 bytes)
```

Fixed overhead is 36 bytes plus the header — about 320 bytes in practice. AES-GCM is length
preserving, so the ciphertext is exactly as long as the video.

### The header

```json
{
  "v": 1,
  "alg": "AES-256-GCM",
  "artifactId": "5f7c…",
  "contentType": "video/mp4",
  "plaintextBytes": 2196233,
  "createdAt": "2026-08-27T12:00:00Z",
  "kdf": { "name": "argon2id", "salt": "…", "parameters": { "memorySize": 65536 } },
  "keyId": "customer-key-1"
}
```

Serialized canonically — sorted keys, no whitespace — because it must be reproducible byte
for byte.

### Two choices worth explaining

**The header is the AAD.** Nobody can rewrite the recorded algorithm, KDF parameters,
content type or artefact id without the key: doing so breaks authentication. A client that
compares `artifactId` against the generation it asked for therefore detects a *substituted*
object, not merely a corrupted one. That check runs in two places — the Worker refuses an
upload whose header names a different job, and the browser refuses a download that does.

**The tag trails the ciphertext.** That is exactly the byte range WebCrypto's
`crypto.subtle.decrypt` expects for AES-GCM, so a browser decrypts by slicing from the end
of the nonce to end-of-file with no reassembly. It also makes encryption a single forward
pass with no seeking, which is what allows constant-memory streaming.

### Versioning

Two independent version fields. `FORMAT_VERSION` covers the framing; `SUITE` covers the
cipher. A future XChaCha20-Poly1305 (`SUITE = 2`) or chunked-streaming variant
(`SUITE = 3`) can be added without invalidating a single existing artefact. Nothing anywhere
uses the file extension to decide how to decrypt: the magic bytes and these two fields do.

### Conformance

`tests/vectors/confidential_container.json` pins one container's exact bytes, produced from
one passphrase through one KDF. Three implementations are held to it:

| Implementation | Must |
|---|---|
| `artifacts.py` | reproduce those bytes exactly |
| `worker.js` | parse the framing and header |
| `client/confidential-generation.js` | derive the key from the passphrase and decrypt it |

Regenerate with `python scripts/make_container_vector.py > tests/vectors/confidential_container.json`.

---

## 5. Key management

```
user passphrase
      │  never leaves the browser
      ▼
   browser
      │  Argon2id (preferred) or PBKDF2-HMAC-SHA256 (default), random 16-byte salt
      ▼
256-bit key ──▶ POST /generate ──▶ Worker ──▶ RunPod ──▶ AES-256-GCM ──▶ ciphertext
```

### Where the key is, and is not

| Location | Key present? |
|---|---|
| Browser memory | yes, while the tab is open |
| `POST /generate` request body | yes, once |
| RunPod job payload | yes, for RunPod's job retention window |
| RunPod worker memory | yes, during the job |
| R2 object bytes | **no** |
| R2 object metadata | **no** |
| Durable Object job state | **no** |
| KV / D1 / Analytics | **no** — none are used by this feature |
| `/status`, `/generate` responses | **no** |
| Cloudflare or RunPod logs | **no** |
| Exception messages | **no** — errors describe shape, never content |
| URLs, query strings, cache keys | **no** |

The Worker holds the key for the duration of one request and forwards it once. It is never
written to any store. After the response is sent it is unreachable, and there is no code
path that could retrieve it later — see §11.

### The KDF choice

Argon2id is the better password KDF, and the format supports it: `kdf.name` is recorded in
the artefact's authenticated header. **The backend never runs a KDF** — it only records
which one was used, so supporting Argon2id costs the server nothing.

It costs the *client* a WebAssembly dependency, because no browser ships Argon2id natively.
This repository has no frontend build step and no `package.json`, so the shipped default is
**PBKDF2-HMAC-SHA256 at 600,000 iterations** (OWASP's current floor), which WebCrypto
implements natively.

To use Argon2id, pass an implementation:

```js
import { argon2id } from "hash-wasm";
const kdf = newKdfParams({ name: "argon2id", parameters: { memorySize: 65536, iterations: 3 } });
const key = await deriveKey(passphrase, kdf, { argon2id });
```

Artefacts encrypted either way remain readable, because each one records how its own key
was derived.

### The salt

Random, 16 bytes, per generation. **Not secret.** It is stored inside the artefact *and*
returned by `/status`, which is exactly what lets the same passphrase reproduce the key
tomorrow. Losing the salt would make an artefact permanently undecryptable, which is why
`encryption.kdf.salt` is validated strictly at submission.

---

## 6. Fail-closed guarantees

Confidential generation either works or is refused. It is never silently downgraded.

| Situation | Outcome |
|---|---|
| `privacyMode: confidential` with no `encryption` | 400 |
| Key that does not decode to exactly 32 bytes | 400, reporting the length, never the value |
| Unsupported algorithm or KDF | 400 |
| Unknown privacy mode | 400 naming the supported ones |
| `private` / `ephemeral` | 501 with the reason — see §9 |
| `privacyMode: standard` **with** an `encryption` block | 400 — ignoring it would leave the caller believing they were encrypted |
| `JOB_TOKEN_SECRET` unset (no authenticated upload path) | 503; the job is never submitted |
| Handler receives a confidential job with no `output` endpoint | job fails; there is no inline fallback |
| Encryption raises | job fails, nothing uploaded, plaintext deleted |
| Upload fails | job fails, plaintext and ciphertext both deleted |
| Plaintext MP4 uploaded for a confidential job | **422, nothing written to R2** |
| Container header names a different generation | **422, nothing written to R2** |

### How the last two are enforced

The job's privacy mode is **signed into the output token** the Worker mints at submission:

```
token payload = { jid: <job id>, pur: "output-upload", exp: …, pm: "confidential" }
```

When the upload arrives, the Worker does not ask the uploader what kind of job this was, and
does not look it up in a Durable Object that could be unavailable or stale. It reads the mode
out of the HMAC-signed token the uploader had to present. RunPod cannot change it without
the signing secret, which never leaves Cloudflare.

If that mode encrypts, the Worker peeks at the first bytes of the body — the header only,
never the payload — and requires them to parse as a container whose authenticated header
names this exact job. There is deliberately no override, no query parameter and no header
that relaxes this.

---

## 7. Plaintext lifecycle on RunPod

1. ComfyUI's `SaveVideo` writes the MP4 to the output directory. This is unavoidable: the
   video tooling writes files.
2. `ConfidentialProtector.protect()` creates a private scratch directory —
   `mkdtemp`, mode `0700`, unpredictable name — and streams the MP4 into a container in
   4 MB chunks under a fresh `uuid4` filename. Names are never reused.
3. The moment a complete container exists, the plaintext is shredded: overwritten with
   random bytes (see the caveat below), then unlinked.
4. The container is uploaded.
5. `finally`: the scratch directory is removed, whether the upload succeeded, failed or
   raised.

`H3_KEEP_OUTPUTS=1` is a debugging convenience for standard mode. It is **deliberately
ignored** for confidential artefacts — "keep the outputs for debugging" must never mean
"keep the plaintext of an artefact the caller asked us to encrypt". There is a test for
that.

### On overwriting before deletion

`H3_OVERWRITE_PLAINTEXT=1` (the default) overwrites a file with random bytes before
unlinking. Be clear about what that is worth: it is occasionally useful on a plain ext4
volume and worth close to nothing on overlayfs, copy-on-write filesystems, or SSDs with wear
levelling — which is where these containers actually run. **Guaranteed secure erasure on
container storage is not achievable.** The controls that matter are the short lifetime and
the destruction of the worker.

---

## 8. Retries and idempotency

RunPod Serverless can retry a job. Every retry:

- draws a **fresh nonce**. The nonce is never derived from the generation id, so a
  deterministic object path cannot cause nonce reuse. `test_every_encryption_uses_a_fresh_nonce`
  encrypts the same plaintext under the same key 32 times and asserts 32 distinct nonces.
- writes to the same deterministic key, `outputs/<jobId>/artifact.enc`. R2 `put` is atomic:
  there is no partial object, and a reader never sees a half-written artefact.
- produces a container whose trailing tag covers the whole file, so a truncated artefact
  fails authentication rather than playing as a corrupt video.

An upload that succeeded but whose status update failed is safe to repeat: the second
upload replaces the object with an equally valid one, and deletion is idempotent.

---

## 9. Privacy modes

One registry, mirrored in `artifacts.py` and `worker.js`, with a test asserting they agree.
Nothing else in either codebase compares privacy-mode strings — code asks the table what a
mode *does*.

| Mode | Status | Semantics |
|---|---|---|
| `standard` | implemented | Current behaviour. Stored as-is, streamed normally. |
| `confidential` | implemented | Client-derived encryption, ciphertext-only persistent storage. |
| `private` | **501** | Unencrypted storage, restricted logging, short default retention. |
| `ephemeral` | **501** | Confidential, plus server-side deletion as soon as delivery succeeds. |

The two unimplemented modes are declared rather than omitted, following the pattern this
Worker already uses for the `reference` and `regenerate_2k` generation modes. A request for
one gets an honest 501 with a reason, never a silent downgrade to something weaker.

**Why `private` is not implemented:** it needs a retention enforcer to mean anything.
Today's retention is a prefix-wide R2 lifecycle rule, which cannot express a per-job
lifetime, so `private` would promise a shorter life than the platform can deliver.

**Why `ephemeral` is not implemented:** it needs a reliable definition of "delivery
succeeded". A ranged or aborted GET is not a delivery, and deleting on the first byte read
would destroy the artefact mid-download.

---

## 10. Recovery: there is none

If the passphrase is lost, the video is unrecoverable. Nothing the platform stores can
decrypt it, and no support process can.

This is a **security property, not a defect**. It is what makes the claim in §1 true. The
reference UI states it plainly before the user commits:

> Your video will be encrypted before persistent storage. You will need this passphrase to
> decrypt it later. We cannot recover a lost passphrase.

---

## 11. Answers to the questions people actually ask

**Does R2 ever receive plaintext for confidential generations?**
No. Encryption happens in the inference container before upload; the Worker independently
verifies the bytes are a container and refuses anything else with a 422.

**Can the encryption key be recovered from platform storage?**
No. It is not in R2 objects, R2 metadata, Durable Object state, KV, D1, Analytics, logs or
any API response. It is only ever in flight.

**Can the Worker recover an old key after the request has completed?**
No. It is a local variable in one request, forwarded once and never written. There is no
code path that could read it back, because there is nowhere it was put.

**Can a confidential video be decrypted using only data stored by the platform?**
No. The platform stores ciphertext, the algorithm, the format version, the KDF name, the
salt and the parameters. Those let a user who knows the passphrase reproduce the key. They
do not let anyone else.

**What plaintext exists during inference?**
The prompt, the model's intermediate state, the decoded frames, and the MP4 on the
container filesystem for the seconds between `SaveVideo` and encryption.

**What happens to plaintext files after generation?**
Overwritten (best effort) and deleted the moment a complete container exists, and again in
a `finally` on every failure path. `H3_KEEP_OUTPUTS` cannot preserve them.

**What happens if encryption fails?**
The job fails. Nothing is uploaded. The plaintext is deleted. There is no fallback that
stores the video unencrypted.

**What happens if upload fails?**
The job fails. Plaintext and ciphertext are both deleted. The job is retryable, and a retry
re-encrypts under a fresh nonce.

**Can existing standard videos still be streamed normally?**
Yes. Nothing about the standard path changed — same key, same content type, same cache
headers, same Range support.

**Can encrypted videos be deleted through the API?**
Yes, by the same code path standard ones use. `DELETE /jobs/:id/artifact` removes the
artefact; `DELETE /jobs/:id` also removes that generation's input keyframes. Both are
idempotent.

**What happens if the user loses their passphrase?**
The video is unrecoverable. See §10.

**Where is the largest residual risk?**
The key is present in the RunPod job payload for RunPod's job retention window. That is
inherent to encrypting where the plaintext lives, and it is the first thing to improve —
see §14.

---

## 12. Retention

`retentionSeconds` or `expiresAt` may be supplied at submission. Both are validated, both
resolve to an absolute ISO-8601 `expiresAt`, and that value is recorded in the artefact's
R2 metadata and reported by `/status`.

**It is advisory.** Nothing in the Worker deletes on a timer. Enforcement today is the
bucket's prefix-scoped R2 lifecycle rule (`scripts/set_r2_lifecycle.sh`, currently 7 days
for `inputs/` and 30 for `outputs/`), which is age-based and cannot express a per-object
lifetime.

A Worker-internal pseudo-scheduler is deliberately **not** implemented: a timer that fires
only while requests happen to arrive is worse than no promise at all.

The implementation path when per-object expiry is needed: a Cloudflare Cron Trigger that
lists `outputs/`, reads each object's `expiresAt` metadata, and deletes what has passed —
calling the same `deleteGeneration()` abstraction the API uses. That is also what would make
`private` and `ephemeral` implementable.

---

## 13. Streaming

Version 1 downloads the whole ciphertext before playback can start. A single GCM tag covers
the entire file, so nothing can be authenticated until the last byte arrives — and playing
unauthenticated plaintext would defeat the point.

For a five-second H3 clip, around 2 MB, this is not noticeable. It is a real limit for long
videos.

The format is versioned so this can be lifted without breaking any existing artefact. A
chunked variant would be a new `SUITE`:

```
encrypted header
chunk 0 ‖ tag 0
chunk 1 ‖ tag 1
chunk 2 ‖ tag 2
...
```

with the chunk index and total bound into each chunk's associated data, so a reordered,
duplicated or dropped chunk fails authentication. **This is deliberately not built yet.**
Chunked AEAD protocols are easy to get subtly wrong, and correctness for the current
five-second use case matters more than a streaming capability nobody has asked for.

---

## 14. Recommended next security improvements

1. **Shorten the key's exposure in RunPod's job store.** The largest residual risk. Options:
   reduce `policy.ttl` for confidential jobs, or hand RunPod a one-time fetch URL for the
   key instead of the key itself, so the payload holds a token the worker redeems once.
2. **Authenticate the public API.** There is no caller identity today, so there is no
   per-user authorization on `/jobs/:id/artifact` — the artefact is ciphertext, but its
   existence and size are readable by anyone with the job id. Adding auth would also give
   `outputKey()` a user namespace.
3. **Argon2id by default**, once the frontend has a build step that can carry the WASM.
4. **Per-object retention enforcement** via a Cron Trigger, which also unblocks `private`
   and `ephemeral`.
5. **Chunked AEAD** if long videos become a product requirement.

---

## 15. Secrets in memory

Perfect erasure is not achievable in Python, JavaScript or V8, and this document will not
claim it.

What is done:

- Keys are scoped to a request and released when it ends.
- `EphemeralKey` wraps the bytes in a type whose `__repr__`, `__str__` and `__format__` are
  redacted, so an f-string, a traceback or a debug dump cannot print it by accident.
- `destroy()` zeroes the backing `bytearray`; `wipe()` does the same in the browser.
- Keys are never serialized, never logged, never copied into a store.

What is not:

- CPython may already have copied the bytes into an intermediate `bytes`, a freed buffer,
  the OS page cache or a swap file. The interpreter offers no way to find or erase those.
- JavaScript engines move and copy memory freely; zeroing one `Uint8Array` does not reach
  the copies the runtime made.

These narrow the window. They do not close it.
