# Technical privacy statement — Confidential Generation

*Draft wording for product documentation. Every sentence here is meant to survive scrutiny
from someone who reads the code. If a claim cannot be defended from
[docs/confidential-generation.md](./confidential-generation.md), it does not belong here.*

---

## The statement

> **Confidential Generation** encrypts generated media before persistent storage using
> encryption material derived from credentials controlled by the user. Persistent storage
> receives encrypted media rather than the plaintext video.
>
> Plaintext prompts and generated media necessarily exist transiently within the trusted
> inference environment while the AI model performs generation and before the resulting
> media is encrypted.

---

## Longer form, for a docs page

Confidential Generation is an optional per-request mode. When it is enabled:

- Your browser derives a 256-bit encryption key from a passphrase you choose. The
  passphrase itself is never transmitted.
- The derived key is sent with your generation request and used **once**, inside the
  environment that runs the model, to encrypt the finished video with **AES-256-GCM**.
- Only the encrypted result is stored. The plaintext video is deleted from the inference
  environment as soon as encryption completes.
- Your browser downloads the encrypted file and decrypts it locally when you enter your
  passphrase again. Playback does not require sending the passphrase anywhere.
- We do not store your passphrase or your encryption key. **If you lose your passphrase,
  the video cannot be recovered — by you or by us.** That is a deliberate property of the
  design, not a limitation we intend to remove.

### What this protects against

If our storage were compromised, exposed, copied into a backup, or if an object link
leaked, the content is ciphertext. It cannot be viewed without the key, and the key is not
something we hold.

### What this does not protect against

Generating a video requires the model to read your prompt and produce the frames. During
that time — seconds, inside the environment running the model — the prompt and the video
exist unencrypted. Encryption happens immediately afterwards, before anything is stored.

Confidential Generation therefore protects **stored** media. It does not, and cannot, make
generation itself invisible to the system performing it.

---

## Terminology

Use these:

| Term | Meaning |
|---|---|
| standard | Default generation and storage. |
| confidential | User-controlled encryption; ciphertext-only persistent storage. |
| encrypted artifact | The stored ciphertext container. |
| client-derived encryption key | The key the browser derives from the passphrase. |
| ephemeral plaintext | Plaintext that exists only for the duration of inference. |
| trusted inference boundary | The environment where the model runs and encryption happens. |

Never use these — each one is false for this system:

| Term | Why not |
|---|---|
| zero knowledge | The inference environment sees the prompt and the video. |
| end-to-end encrypted AI inference | There is no end-to-end path that excludes the model. |
| "the AI never sees your prompt" | It cannot generate without it. |
| "the server never sees plaintext" | The GPU worker does, for the duration of the job. |
| "impossible for infrastructure to access" | A compromised inference process could read it. |

---

## Support answers

**"Can you recover my video? I forgot the passphrase."**
No. We do not store your passphrase or your key, and the stored file cannot be decrypted
without them. This is what Confidential Generation means.

**"Can your staff watch my confidential videos?"**
Not from storage — what is stored is ciphertext and we do not hold the key. While a video
is being generated it exists unencrypted inside the environment running the model, which is
inherent to generating it at all.

**"Is this end-to-end encrypted?"**
Not in the sense that term usually carries. The model has to read your prompt and produce
the video. What is encrypted end to end is the *stored artifact*: from the moment it is
created until you decrypt it in your browser, nothing in between can read it.

**"Where is my key stored?"**
Nowhere. It exists in your browser, in the request that starts the generation, and in the
environment that encrypts the video. It is not written to storage, metadata, logs or any API
response.
