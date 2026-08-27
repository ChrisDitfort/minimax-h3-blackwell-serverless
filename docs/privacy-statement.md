# Technical privacy statement — Confidential Generation

*Draft wording for product documentation. Every sentence here is meant to survive scrutiny
from someone who reads the code. If a claim cannot be defended from
[docs/confidential-generation.md](./confidential-generation.md), it does not belong here.*

---

## The statement

> **Confidential Generation** uses hybrid encryption. The inference worker receives an
> encryption-only public key and generates a unique random file-encryption key for each
> video. The video is encrypted with authenticated symmetric encryption, and the file key is
> wrapped using the user's public key before persistent storage. The user's passphrase,
> private key and password-derived key are not sent to the inference job.
>
> Plaintext prompts and generated media necessarily exist transiently within the trusted
> inference environment while generation and encryption occur.

---

## Longer form, for a docs page

Confidential Generation is an optional per-request mode. When it is enabled:

- Your browser holds an encryption key pair. The private half is encrypted with a key
  derived from a passphrase you choose; the passphrase itself is never transmitted, and
  neither is the private key.
- Only the **public** half is sent with your generation request. A public key can encrypt
  and cannot decrypt, so a copy of that request is of no use to anyone.
- The environment that runs the model generates a fresh random key for each video, encrypts
  the finished video with **AES-256-GCM**, and locks that key to your public key. It then
  discards it.
- Only the encrypted result is stored. The plaintext video is deleted from the inference
  environment as soon as encryption completes.
- Your browser downloads the encrypted file, unlocks its private key with your passphrase,
  and decrypts locally. Playback does not send the passphrase anywhere.
- We do not store your passphrase, your private key, or the key that encrypted any video.
  **If you lose your passphrase, the video cannot be recovered — by you or by us.** That is
  a deliberate property of the design, not a limitation we intend to remove.

### What this protects against

If our storage were compromised, exposed, copied into a backup, or if an object link
leaked, the content is ciphertext. The same is true of the job records our inference
provider retains: they contain your public key, which cannot decrypt anything.

Put precisely: a complete copy of everything we and our providers store is not enough to
recover a confidential video. Your passphrase is required, and we do not have it.

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
| key-encryption key | Derived from the passphrase. Unlocks the private key; never touches a video. |
| file-encryption key | Random, one per video, generated where the video is. |
| ephemeral plaintext | Plaintext that exists only for the duration of inference. |
| trusted inference boundary | The environment where the model runs and encryption happens. |
| hybrid encryption | A random per-video key encrypts the media; your public key locks that key. |
| public encryption key | The half that can encrypt but not decrypt. Safe to send. |
| wrapped file key | The per-video key, locked to your public key. Useless without your private key. |

Never use these — each one is false for this system:

| Term | Why not |
|---|---|
| zero knowledge | The inference environment sees the prompt and the video. |
| end-to-end encrypted AI inference | There is no end-to-end path that excludes the model. |
| "the AI never sees your prompt" | It cannot generate without it. |
| "the server never sees plaintext" | The GPU worker does, for the duration of the job. |
| "impossible for infrastructure to access" | A compromised inference process could read it. |
| "the inference provider never sees plaintext" | It runs the model; it sees the prompt and the frames. |

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
Your private key is stored only in encrypted form, and only where you keep it — we never
receive it. Your public key travels with each generation request, which is safe because it
cannot decrypt. The per-video key exists for a few seconds inside the environment that
encrypts the video and is then discarded; its only stored form is locked to your public key.

**"You send my key to a third party?"**
We send your *public* key to the service that runs the model, because encryption has to
happen where the video is created. A public key cannot decrypt. Your passphrase, your
private key and the key derived from your passphrase are never sent anywhere.
