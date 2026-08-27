/*
 * MiniMax H3 - Cloudflare Worker
 *
 * Public API in front of two RunPod Serverless endpoints (Ada "h3" and Blackwell
 * "h3-blackwell"). This Worker owns the public contract and builds the ComfyUI graph;
 * the RunPod handler stays a thin executor that runs whatever workflow it is handed.
 *
 * Normalized request (preferred):
 *
 *   { "backend": "h3-blackwell", "mode": "text_to_video",
 *     "prompt": "...", "quality": "standard", "duration": 5,
 *     "aspect_ratio": "16:9", "seed": 51 }
 *
 * Raw request (still supported, unchanged):
 *
 *   { "backend": "h3-blackwell", "prompt": "...",
 *     "width": 1024, "height": 576, "frames": 124, "steps": 20, "seed": 51 }
 *
 * Every dimension and frame-count rule below is taken from the pinned ComfyUI build's
 * own comfy_extras/nodes_minimax_h3.py, not invented here:
 *
 *   CANVAS_MULTIPLE = 32, BASE_SHORT_EDGE = 768, MAX_PIXELS = 768 * 1344, FPS = 24
 *   align_frame_count(n): smallest n' >= n with n' % 17 == 5
 *   adapt_canvas(w, h):   short edge to 768, area capped, each axis rounded to 32
 */

const DEFAULT_BACKEND = "h3";

const BACKENDS = Object.freeze({
  h3: Object.freeze({
    endpointIdEnv: "RUNPOD_ENDPOINT_ID",
    apiKeyEnv: "RUNPOD_API_KEY",
    fallbackApiKeyEnv: null
  }),

  "h3-blackwell": Object.freeze({
    endpointIdEnv: "RUNPOD_BLACKWELL_ENDPOINT_ID",
    apiKeyEnv: "RUNPOD_BLACKWELL_API_KEY",

    // This lets both endpoints share RUNPOD_API_KEY
    // when they are in the same RunPod account.
    fallbackApiKeyEnv: "RUNPOD_API_KEY"
  })
});

const BACKEND_ALIASES = Object.freeze({
  h3: "h3",
  default: "h3",
  original: "h3",
  ada: "h3",
  "h3-ada": "h3",
  "48gb-pro": "h3",
  "minimax-h3": "h3",

  blackwell: "h3-blackwell",
  "h3-blackwell": "h3-blackwell",
  h3_blackwell: "h3-blackwell",
  "minimax-h3-blackwell": "h3-blackwell"
});

const BINARY_FIELD_NAMES = new Set([
  "data",
  "base64",
  "image_base64",
  "video_base64",
  "audio_base64",
  "file_data"
]);

/* ------------------------------------------------------------------------------------
 * H3 model constants (mirrored from comfy_extras/nodes_minimax_h3.py)
 * ---------------------------------------------------------------------------------- */

const FPS = 24;
const CANVAS_MULTIPLE = 32;
const BASE_SHORT_EDGE = 768;
const MAX_PIXELS = 768 * 1344;

// The model's temporal grid: legal frame counts satisfy n % 17 == 5.
const FRAME_GRID_MODULUS = 17;
const FRAME_GRID_REMAINDER = 5;
const MIN_FRAMES = 5;
const MAX_FRAMES = 3600;

/*
 * Quality tiers. Only the short edge and step count vary; the actual pixel dimensions
 * are derived per aspect ratio through the model's own canvas rule, so every tier is
 * guaranteed 32-aligned and within the area cap.
 *
 * "hd" uses BASE_SHORT_EDGE (768), which is the canvas the H3 nodes default to and the
 * resolution the model was trained around - not an upscale or a second-stage pipeline.
 */
const QUALITY_PRESETS = Object.freeze({
  fast: Object.freeze({ shortEdge: 576, steps: 14 }),
  standard: Object.freeze({ shortEdge: 576, steps: 20 }),
  hd: Object.freeze({ shortEdge: BASE_SHORT_EDGE, steps: 20 })
});

const DEFAULT_QUALITY = "standard";

const ASPECT_RATIOS = Object.freeze({
  "16:9": 16 / 9,
  "9:16": 9 / 16,
  "1:1": 1,
  "4:3": 4 / 3,
  "3:4": 3 / 4
});

const DEFAULT_ASPECT_RATIO = "16:9";
const DEFAULT_DURATION_SECONDS = 5;

/*
 * Generation modes. `implemented` gates what this release will actually run - an
 * unimplemented mode returns 501 rather than silently degrading to text-to-video,
 * which would hand back a video that quietly ignored the caller's keyframes.
 */
const MODES = Object.freeze({
  text_to_video: Object.freeze({ implemented: true, firstFrame: false, lastFrame: false }),
  first_frame_to_video: Object.freeze({ implemented: true, firstFrame: true, lastFrame: false }),

  last_frame_to_video: Object.freeze({ implemented: true, firstFrame: false, lastFrame: true }),
  first_last_frame_to_video: Object.freeze({
    implemented: true,
    firstFrame: true,
    lastFrame: true
  }),

  reference: Object.freeze({
    implemented: false,
    firstFrame: false,
    lastFrame: false,
    reason:
      "Ref2VA needs minimax_h3_ref2va_pruned_int8_convrot.safetensors (20.97 GB), which " +
      "is deliberately excluded from the image. See README 'Ref2VA status'."
  }),

  regenerate_2k: Object.freeze({
    implemented: false,
    firstFrame: false,
    lastFrame: false,
    reason:
      "No second-stage upscale model is installed and none has been measured. See " +
      "README '2K status'."
  })
});

const DEFAULT_MODE = "text_to_video";

/* ------------------------------------------------------------------------------------
 * Privacy modes
 *
 * The registry, mirrored in artifacts.py. Nothing else in this file compares privacy-mode
 * strings: code asks this table what a mode *does*. That is what stops a future mode from
 * needing edits scattered through the request path, and it is why `private` and
 * `ephemeral` can be declared here honestly - as a 501 with a reason, exactly like the
 * unimplemented generation MODES above - instead of being silently treated as `standard`.
 * ---------------------------------------------------------------------------------- */

const PRIVACY_MODES = Object.freeze({
  standard: Object.freeze({
    implemented: true,
    encrypts: false,
    restrictsPromptLogging: false,
    description: "Current behaviour. The MP4 is stored as-is and streamed back normally."
  }),

  confidential: Object.freeze({
    implemented: true,
    encrypts: true,
    restrictsPromptLogging: true,
    description:
      "The artefact is encrypted inside the inference environment with a key the caller " +
      "derived. Persistent storage receives ciphertext only; the browser decrypts for " +
      "playback."
  }),

  private: Object.freeze({
    implemented: false,
    encrypts: false,
    restrictsPromptLogging: true,
    description: "Unencrypted storage with restricted logging and a short default retention.",
    reason:
      "Needs a retention enforcer to be meaningful. Today's retention is a prefix-wide R2 " +
      "lifecycle rule, which cannot express a per-job lifetime, so 'private' would promise " +
      "a shorter life than the platform can deliver."
  }),

  ephemeral: Object.freeze({
    implemented: false,
    encrypts: true,
    restrictsPromptLogging: true,
    description:
      "Confidential, plus the server-side copy is deleted as soon as delivery succeeds.",
    reason:
      "Needs a reliable definition of 'delivery succeeded'. A ranged or aborted GET is not " +
      "a delivery, and deleting on the first byte read would destroy the artefact " +
      "mid-download."
  })
});

const DEFAULT_PRIVACY_MODE = "standard";

/*
 * Encrypted container framing, mirrored from artifacts.py. The Worker only ever *reads*
 * this - it parses enough of an upload to prove the bytes really are ciphertext before
 * they are allowed into R2. tests/vectors/confidential_container.json pins the layout so
 * the two implementations cannot drift apart unnoticed.
 */
const CONTAINER_MAGIC = "CGEN";
const CONTAINER_VERSION = 1;
const CONTAINER_SUITES = Object.freeze({ 1: "AES-256-GCM" });
const CONTAINER_PREAMBLE_BYTES = 8;
const CONTAINER_MAX_HEADER_BYTES = 8192;
const CONTAINER_NONCE_BYTES = 12;
const CONTAINER_TAG_BYTES = 16;

const ENCRYPTED_CONTENT_TYPE = "application/octet-stream";

/*
 * Supported client-side key derivation functions.
 *
 * The Worker never runs a KDF - derivation happens in the browser and only the derived key
 * is ever transmitted. This list exists so that unrecognised metadata is rejected at the
 * door rather than stored and echoed back to a client that would not know what to do with
 * it. Adding a KDF here costs nothing on the backend, which is the point: the choice
 * belongs to the client.
 */
const SUPPORTED_KDFS = Object.freeze({
  argon2id: { saltMin: 8, saltMax: 64 },
  scrypt: { saltMin: 8, saltMax: 64 },
  "pbkdf2-sha256": { saltMin: 8, saltMax: 64 },
  "pbkdf2-sha512": { saltMin: 8, saltMax: 64 }
});

const SUPPORTED_ENCRYPTION_ALGORITHMS = Object.freeze(["AES-256-GCM"]);
const ENCRYPTION_KEY_BYTES = 32;

// Retention bounds for the advisory expiresAt recorded on an artefact. One minute is the
// shortest that is not simply a mistake; 90 days is past any lifecycle rule we ship.
const MIN_RETENTION_SECONDS = 60;
const MAX_RETENTION_SECONDS = 90 * 24 * 3600;

// Model filenames, matching what models.tsv bakes into the RunPod image.
const MODEL_FILES = Object.freeze({
  unet: "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
  clip: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
  videoVae: "minimax_h3_video_vae_fp16.safetensors",
  audioVae: "minimax_h3_audio_vae_fp32.safetensors"
});

class HttpError extends Error {
  constructor(status, message, details) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.details = details;
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const headers = corsHeaders();

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers });
    }

    try {
      const segments = splitPath(url.pathname);

      if (request.method === "POST" && isExactRoute(segments, "generate")) {
        return await generateVideo(request, env, headers, url);
      }

      if (request.method === "GET" && segments[0] === "status") {
        const { backend, jobId } = parseJobRoute(segments, url, "status");
        return await getJobStatus(backend, jobId, env, headers, url);
      }

      if (request.method === "POST" && segments[0] === "cancel") {
        const { backend, jobId } = parseJobRoute(segments, url, "cancel");
        return await cancelJob(backend, jobId, env, headers, url);
      }

      /*
       * GET /capabilities
       *
       * Lets a client discover the presets, aspect ratios and modes this deployment
       * actually supports, instead of hardcoding them and finding out via a 400.
       */
      if (request.method === "GET" && isExactRoute(segments, "capabilities")) {
        return json(describeCapabilities(), 200, headers);
      }

      // POST /jobs/:jobId/assets  - upload an input keyframe
      if (
        request.method === "POST" &&
        segments.length === 3 &&
        segments[0] === "jobs" &&
        segments[2] === "assets"
      ) {
        return await uploadAsset(request, env, headers, segments[1], url);
      }

      /*
       * GET /jobs/:jobId/video  - stream the finished artefact out of R2
       * GET /jobs/:jobId/artifact  - the same route under a mode-neutral name
       *
       * `video` is kept because callers already use it and it still works for both modes.
       * `artifact` exists because a confidential artefact is not a video until the browser
       * has decrypted it, and an API should not say otherwise.
       */
      if (
        request.method === "GET" &&
        segments.length === 3 &&
        segments[0] === "jobs" &&
        (segments[2] === "video" || segments[2] === "artifact")
      ) {
        return await streamVideo(request, env, segments[1]);
      }

      // DELETE /jobs/:jobId/video|artifact  - remove that job's output from R2
      if (
        request.method === "DELETE" &&
        segments.length === 3 &&
        segments[0] === "jobs" &&
        (segments[2] === "video" || segments[2] === "artifact")
      ) {
        return await deleteGeneration(env, headers, segments[1], "output");
      }

      // DELETE /jobs/:jobId  - the whole generation: artefact plus its input keyframes
      if (request.method === "DELETE" && segments.length === 2 && segments[0] === "jobs") {
        return await deleteGeneration(env, headers, segments[1], "generation");
      }

      // GET /ws/jobs/:jobId  - realtime progress
      if (
        request.method === "GET" &&
        segments.length === 3 &&
        segments[0] === "ws" &&
        segments[1] === "jobs"
      ) {
        return await openJobSocket(request, env, segments[2]);
      }

      // --- internal, RunPod -> Cloudflare. All HMAC-gated, all job-scoped. ---

      if (
        segments[0] === "internal" &&
        segments[1] === "jobs" &&
        segments.length >= 4
      ) {
        const jobId = segments[2];

        if (request.method === "POST" && segments[3] === "progress" && segments.length === 4) {
          return await receiveProgress(request, env, headers, jobId);
        }

        if (request.method === "PUT" && segments[3] === "output" && segments.length === 4) {
          return await receiveOutput(request, env, headers, jobId);
        }

        if (request.method === "GET" && segments[3] === "assets" && segments.length === 5) {
          return await serveAsset(request, env, jobId, segments[4]);
        }
      }

      if (request.method === "GET" && isExactRoute(segments, "health")) {
        return json(
          {
            ok: true,
            service: "minimax-h3-backend",
            defaultBackend: DEFAULT_BACKEND,

            backends: {
              h3: { configured: isBackendConfigured("h3", env) },
              "h3-blackwell": { configured: isBackendConfigured("h3-blackwell", env) }
            },

            routes: routeDescriptions()
          },
          200,
          headers
        );
      }

      return json(
        {
          error: "Not found",
          routes: routeDescriptions(),
          backends: ["h3", "h3-blackwell"]
        },
        404,
        headers
      );
    } catch (error) {
      const status = error instanceof HttpError ? error.status : 500;
      const response = { error: error?.message || String(error) };

      if (error instanceof HttpError && error.details !== undefined) {
        response.details = error.details;
      }

      return json(response, status, headers);
    }
  }
};

/* ------------------------------------------------------------------------------------
 * Settings normalization
 *
 * One place that turns a request body - normalized or raw - into a fully resolved
 * settings object. Nothing downstream re-parses the request.
 * ---------------------------------------------------------------------------------- */

/**
 * The model's canvas rule, mirrored from adapt_canvas() in nodes_minimax_h3.py:
 * put the short edge at `shortEdge`, cap total area, then round each axis to 32.
 */
export function adaptCanvas(ratio, shortEdge) {
  let nomW;
  let nomH;

  if (ratio >= 1) {
    nomW = shortEdge * ratio;
    nomH = shortEdge;
  } else {
    nomW = shortEdge;
    nomH = shortEdge / ratio;
  }

  if (nomW * nomH > MAX_PIXELS) {
    const scale = Math.sqrt(MAX_PIXELS / (nomW * nomH));
    nomW *= scale;
    nomH *= scale;
  }

  return {
    width: Math.max(CANVAS_MULTIPLE, Math.round(nomW / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    height: Math.max(CANVAS_MULTIPLE, Math.round(nomH / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
  };
}

/** Smallest legal frame count >= n. H3 only accepts n % 17 == 5. */
export function alignFrameCount(n) {
  let frames = Math.max(MIN_FRAMES, Math.trunc(n));
  while (frames % FRAME_GRID_MODULUS !== FRAME_GRID_REMAINDER) {
    frames += 1;
  }
  return frames;
}

export function isLegalFrameCount(n) {
  return (
    Number.isInteger(n) &&
    n >= MIN_FRAMES &&
    n <= MAX_FRAMES &&
    n % FRAME_GRID_MODULUS === FRAME_GRID_REMAINDER
  );
}

/** Seconds -> legal frame count. The single place duration is converted. */
export function durationToFrames(seconds) {
  return alignFrameCount(Math.round(seconds * FPS));
}

export function framesToDuration(frames) {
  return frames / FPS;
}

/**
 * Compose the final H3 prompt from the user's text plus optional structured fields.
 *
 * The user's prompt is never rewritten - the structured lines are appended after it, and
 * absent fields contribute nothing at all (no empty labels).
 */
export function buildPrompt(input) {
  const base = String(input.prompt ?? "").trim();

  const sections = [
    ["Camera", input.camera],
    ["Shot", input.shot],
    ["Lighting", input.lighting],
    ["Style", input.style],
    ["Motion", input.motion],
    ["Audio", input.audio_prompt]
  ];

  const lines = [];
  for (const [label, value] of sections) {
    const text = String(value ?? "").trim();
    if (text) {
      lines.push(`${label}: ${trimTrailingPeriod(text)}.`);
    }
  }

  if (lines.length === 0) {
    return base;
  }
  return `${base}\n\n${lines.join("\n")}`;
}

function trimTrailingPeriod(text) {
  return text.endsWith(".") ? text.slice(0, -1) : text;
}

/**
 * Resolve the generation mode. An explicit `mode` always wins; otherwise it is inferred
 * from which keyframes were supplied, so existing raw callers keep working untouched.
 */
export function resolveMode(body) {
  if (hasNonEmptyValue(body.mode)) {
    const mode = String(body.mode).trim().toLowerCase();
    if (!Object.prototype.hasOwnProperty.call(MODES, mode)) {
      throw new HttpError(
        400,
        `Unknown mode '${mode}'. Supported: ${Object.keys(MODES).join(", ")}`
      );
    }
    return mode;
  }

  const hasFirst = isAssetReference(body.first_frame);
  const hasLast = isAssetReference(body.last_frame);

  if (hasFirst && hasLast) return "first_last_frame_to_video";
  if (hasFirst) return "first_frame_to_video";
  if (hasLast) return "last_frame_to_video";

  if (
    isNonEmptyArray(body.reference_images) ||
    isNonEmptyArray(body.reference_videos) ||
    isNonEmptyArray(body.reference_audio)
  ) {
    return "reference";
  }

  return DEFAULT_MODE;
}

function isAssetReference(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isNonEmptyArray(value) {
  return Array.isArray(value) && value.length > 0;
}

/**
 * Turn a request body into fully resolved settings.
 *
 * Precedence is explicit and never silent: raw width/height/frames/steps beat the
 * quality preset, but a raw value that *contradicts* a normalized one (aspect_ratio vs
 * width/height, duration vs frames) is a 400 rather than a coin toss.
 */
export function normalizeRequest(body) {
  const mode = resolveMode(body);
  const modeSpec = MODES[mode];

  if (!modeSpec.implemented) {
    throw new HttpError(501, `Mode '${mode}' is not available yet. ${modeSpec.reason}`, {
      mode,
      implementedModes: Object.keys(MODES).filter((m) => MODES[m].implemented)
    });
  }

  const prompt = buildPrompt(body);
  if (!String(body.prompt ?? "").trim()) {
    throw new HttpError(400, "prompt is required");
  }

  // --- quality tier -----------------------------------------------------------------
  const qualityGiven = hasNonEmptyValue(body.quality);
  const quality = qualityGiven ? String(body.quality).trim().toLowerCase() : DEFAULT_QUALITY;

  if (!Object.prototype.hasOwnProperty.call(QUALITY_PRESETS, quality)) {
    throw new HttpError(
      400,
      `Unknown quality '${quality}'. Supported: ${Object.keys(QUALITY_PRESETS).join(", ")}`
    );
  }
  const preset = QUALITY_PRESETS[quality];

  // --- canvas -----------------------------------------------------------------------
  const widthGiven = hasNonEmptyValue(body.width);
  const heightGiven = hasNonEmptyValue(body.height);
  const aspectGiven = hasNonEmptyValue(body.aspect_ratio);

  if (widthGiven !== heightGiven) {
    throw new HttpError(400, "width and height must be supplied together");
  }

  const aspectRatio = aspectGiven
    ? String(body.aspect_ratio).trim()
    : DEFAULT_ASPECT_RATIO;

  if (aspectGiven && !Object.prototype.hasOwnProperty.call(ASPECT_RATIOS, aspectRatio)) {
    throw new HttpError(
      400,
      `Unsupported aspect_ratio '${aspectRatio}'. Supported: ${Object.keys(ASPECT_RATIOS).join(", ")}`
    );
  }

  let width;
  let height;
  let canvasSource;

  if (widthGiven) {
    width = toInteger(body.width, 0);
    height = toInteger(body.height, 0);
    canvasSource = "explicit";

    if (aspectGiven) {
      const requested = ASPECT_RATIOS[aspectRatio];
      const actual = width / height;
      // 32-alignment means the realised ratio is never exact; allow a little slack.
      if (Math.abs(actual - requested) > 0.05) {
        throw new HttpError(
          400,
          `width/height (${width}x${height}, ratio ${actual.toFixed(3)}) contradicts ` +
            `aspect_ratio '${aspectRatio}' (${requested.toFixed(3)}). Supply one or the other.`
        );
      }
    }
  } else {
    const canvas = adaptCanvas(ASPECT_RATIOS[aspectRatio], preset.shortEdge);
    width = canvas.width;
    height = canvas.height;
    canvasSource = aspectGiven || qualityGiven ? "preset" : "default";
  }

  validateCanvas(width, height);

  // --- frames -----------------------------------------------------------------------
  const framesGiven = hasNonEmptyValue(body.frames);
  const durationGiven = hasNonEmptyValue(body.duration);

  let frames;
  let framesSource;

  if (framesGiven) {
    frames = toInteger(body.frames, 0);
    framesSource = "explicit";

    if (!isLegalFrameCount(frames)) {
      throw new HttpError(
        400,
        `frames must satisfy frames % 17 == 5 (H3's temporal grid), between ${MIN_FRAMES} ` +
          `and ${MAX_FRAMES}. Nearest legal values to ${frames}: ` +
          `${nearestLegalFrames(frames).join(" or ")}.`
      );
    }

    if (durationGiven) {
      const fromDuration = durationToFrames(toNumber(body.duration, 0));
      if (fromDuration !== frames) {
        throw new HttpError(
          400,
          `frames (${frames}) contradicts duration (${body.duration}s -> ${fromDuration} ` +
            "frames). Supply one or the other."
        );
      }
    }
  } else {
    const seconds = durationGiven ? toNumber(body.duration, 0) : DEFAULT_DURATION_SECONDS;
    if (!(seconds > 0)) {
      throw new HttpError(400, "duration must be a positive number of seconds");
    }
    frames = durationToFrames(seconds);
    framesSource = durationGiven ? "duration" : "default";

    if (frames > MAX_FRAMES) {
      throw new HttpError(
        400,
        `duration ${seconds}s exceeds the maximum of ${framesToDuration(MAX_FRAMES).toFixed(1)}s`
      );
    }
  }

  // --- steps ------------------------------------------------------------------------
  const stepsGiven = hasNonEmptyValue(body.steps);
  const steps = stepsGiven ? toInteger(body.steps, preset.steps) : preset.steps;

  if (steps <= 0 || steps > 100) {
    throw new HttpError(400, "steps must be between 1 and 100");
  }

  // --- seed -------------------------------------------------------------------------
  const seed = hasNonEmptyValue(body.seed)
    ? toInteger(body.seed, 0)
    : Math.floor(Math.random() * 2147483647);

  if (seed < 0) {
    throw new HttpError(400, "seed must be non-negative");
  }

  // --- keyframes --------------------------------------------------------------------
  const firstFrame = modeSpec.firstFrame ? normalizeAsset(body.first_frame, "first_frame") : null;
  const lastFrame = modeSpec.lastFrame ? normalizeAsset(body.last_frame, "last_frame") : null;

  if (modeSpec.firstFrame && !firstFrame) {
    throw new HttpError(400, `mode '${mode}' requires first_frame`);
  }
  if (modeSpec.lastFrame && !lastFrame) {
    throw new HttpError(400, `mode '${mode}' requires last_frame`);
  }

  return {
    mode,
    quality,
    aspectRatio,
    prompt,
    userPrompt: String(body.prompt).trim(),
    width,
    height,
    frames,
    fps: FPS,
    durationSeconds: framesToDuration(frames),
    steps,
    seed,
    audio: body.audio === undefined ? true : Boolean(body.audio),
    firstFrame,
    lastFrame,
    resolvedFrom: { canvas: canvasSource, frames: framesSource, steps: stepsGiven ? "explicit" : "preset" }
  };
}

/* ------------------------------------------------------------------------------------
 * Privacy and encryption validation
 *
 * Everything a malformed confidential request could do wrong is caught here, before a job
 * is submitted. The rule throughout: reject rather than reinterpret. A request that thinks
 * it is being encrypted and is not is the single worst outcome this endpoint can produce,
 * so every ambiguity is a 400.
 * ---------------------------------------------------------------------------------- */

/**
 * Resolve `privacyMode` and validate any `encryption` block that came with it.
 *
 * Returns `{ mode, spec, encryption, expiresAt }`. `encryption` is null in standard mode
 * and, in confidential mode, holds the raw key alongside its non-secret metadata. The
 * caller must treat `encryption.key` as request-scoped: forward it once, never store it.
 */
export function normalizePrivacy(body) {
  const requested = firstPresent(body.privacyMode, body.privacy_mode);
  const mode = requested ? String(requested).trim().toLowerCase() : DEFAULT_PRIVACY_MODE;
  const spec = PRIVACY_MODES[mode];

  if (!spec) {
    throw new HttpError(
      400,
      `Unknown privacyMode '${mode}'. Supported: ${Object.keys(PRIVACY_MODES).join(", ")}`
    );
  }

  if (!spec.implemented) {
    throw new HttpError(501, `privacyMode '${mode}' is not available yet. ${spec.reason}`, {
      privacyMode: mode,
      implementedPrivacyModes: Object.keys(PRIVACY_MODES).filter((m) => PRIVACY_MODES[m].implemented)
    });
  }

  const block = body.encryption;
  const hasBlock = block !== undefined && block !== null;

  if (hasBlock && (typeof block !== "object" || Array.isArray(block))) {
    throw new HttpError(400, "encryption must be an object");
  }

  if (!spec.encrypts) {
    // Not ignored: a caller who sent a key believes their artefact is being encrypted.
    if (hasBlock) {
      throw new HttpError(
        400,
        `privacyMode '${mode}' does not encrypt, but an encryption block was supplied. ` +
          "Use privacyMode 'confidential', or remove the block."
      );
    }
    return { mode, spec, encryption: null, expiresAt: resolveExpiry(body) };
  }

  if (!hasBlock) {
    throw new HttpError(
      400,
      "privacyMode 'confidential' requires an encryption block with a client-derived key. " +
        "See docs/confidential-generation.md."
    );
  }

  const algorithm = hasNonEmptyValue(block.algorithm)
    ? String(block.algorithm).trim()
    : SUPPORTED_ENCRYPTION_ALGORITHMS[0];

  if (!SUPPORTED_ENCRYPTION_ALGORITHMS.includes(algorithm)) {
    throw new HttpError(
      400,
      `Unsupported encryption algorithm '${algorithm}'. Supported: ` +
        SUPPORTED_ENCRYPTION_ALGORITHMS.join(", ")
    );
  }

  const key = validateEncryptionKey(block.key);
  const keyId = validateKeyId(firstPresent(block.keyId, block.key_id), key);
  const kdf = validateKdf(block.kdf);

  return {
    mode,
    spec,
    encryption: { algorithm, key, keyId, kdf },
    expiresAt: resolveExpiry(body)
  };
}

/**
 * Check a base64url-encoded 256-bit key without ever putting it in an error message.
 *
 * Every rejection describes the shape of what arrived - "decoded to 16 bytes" - and never
 * its content, because a 400 body is exactly the kind of thing that ends up in a log.
 */
function validateEncryptionKey(value) {
  if (!hasNonEmptyValue(value)) {
    throw new HttpError(400, "encryption.key is required in confidential mode");
  }
  if (typeof value !== "string") {
    throw new HttpError(400, "encryption.key must be a base64url-encoded string");
  }

  const text = value.trim();
  if (!/^[A-Za-z0-9_\-+/]+={0,2}$/.test(text)) {
    throw new HttpError(400, "encryption.key is not valid base64url");
  }

  let bytes;
  try {
    bytes = b64urlDecodeToBytes(text);
  } catch {
    throw new HttpError(400, "encryption.key is not valid base64url");
  }

  if (bytes.length !== ENCRYPTION_KEY_BYTES) {
    throw new HttpError(
      400,
      `encryption.key must decode to ${ENCRYPTION_KEY_BYTES} bytes (256 bits), got ${bytes.length}`
    );
  }

  return text;
}

/**
 * `keyId` is an opaque client label, and its whole job is to be safe to show.
 *
 * The one check worth making is that the client has not simply pasted the key into it:
 * that mistake would take a value we go to lengths never to persist and put it into
 * metadata, status responses and logs.
 */
function validateKeyId(value, key) {
  if (!hasNonEmptyValue(value)) return null;

  const keyId = String(value).trim();
  if (keyId.length > 128) {
    throw new HttpError(400, "encryption.keyId must be 128 characters or fewer");
  }
  if (!/^[A-Za-z0-9._:\-]+$/.test(keyId)) {
    throw new HttpError(400, "encryption.keyId may contain only [A-Za-z0-9._:-]");
  }
  if (key && keyId === key) {
    throw new HttpError(
      400,
      "encryption.keyId must not be the key itself. keyId is stored and returned; the key " +
        "never is."
    );
  }
  return keyId;
}

/**
 * Validate the KDF description the client will need to reconstruct its own key later.
 *
 * None of this is secret and none of it is executed here - it is recorded so that a user
 * who returns tomorrow with the same passphrase derives the same key. Which is exactly why
 * it has to be well-formed: a corrupted salt makes an artefact permanently undecryptable.
 */
function validateKdf(value) {
  if (value === undefined || value === null) return null;

  if (typeof value !== "object" || Array.isArray(value)) {
    throw new HttpError(400, "encryption.kdf must be an object");
  }

  const name = String(value.name ?? "").trim().toLowerCase();
  const spec = SUPPORTED_KDFS[name];
  if (!spec) {
    throw new HttpError(
      400,
      `Unsupported encryption.kdf.name '${name || "(missing)"}'. Supported: ` +
        Object.keys(SUPPORTED_KDFS).join(", ")
    );
  }

  if (!hasNonEmptyValue(value.salt)) {
    throw new HttpError(400, "encryption.kdf.salt is required");
  }
  if (typeof value.salt !== "string" || !/^[A-Za-z0-9_\-+/]+={0,2}$/.test(value.salt.trim())) {
    throw new HttpError(400, "encryption.kdf.salt must be base64url");
  }

  let salt;
  try {
    salt = b64urlDecodeToBytes(value.salt.trim());
  } catch {
    throw new HttpError(400, "encryption.kdf.salt must be base64url");
  }
  if (salt.length < spec.saltMin || salt.length > spec.saltMax) {
    throw new HttpError(
      400,
      `encryption.kdf.salt must decode to ${spec.saltMin}-${spec.saltMax} bytes, got ${salt.length}`
    );
  }

  const parameters = value.parameters ?? {};
  if (typeof parameters !== "object" || Array.isArray(parameters)) {
    throw new HttpError(400, "encryption.kdf.parameters must be an object");
  }

  // Flat, small and scalar. These end up inside the artefact's authenticated header, whose
  // size is capped; an unbounded object here would fail much later, at encryption time.
  const entries = Object.entries(parameters);
  if (entries.length > 12) {
    throw new HttpError(400, "encryption.kdf.parameters may hold at most 12 values");
  }
  const cleaned = {};
  for (const [name_, raw] of entries) {
    if (!/^[A-Za-z0-9_]{1,32}$/.test(name_)) {
      throw new HttpError(400, `encryption.kdf.parameters key '${name_}' is not a simple name`);
    }
    if (typeof raw === "number") {
      if (!Number.isFinite(raw)) {
        throw new HttpError(400, `encryption.kdf.parameters.${name_} must be finite`);
      }
      cleaned[name_] = raw;
    } else if (typeof raw === "string") {
      if (raw.length > 64) {
        throw new HttpError(400, `encryption.kdf.parameters.${name_} is too long`);
      }
      cleaned[name_] = raw;
    } else if (typeof raw === "boolean") {
      cleaned[name_] = raw;
    } else {
      throw new HttpError(
        400,
        `encryption.kdf.parameters.${name_} must be a number, string or boolean`
      );
    }
  }

  return { name, salt: value.salt.trim(), parameters: cleaned };
}

/**
 * Resolve an advisory expiry from `retentionSeconds` or `expiresAt`.
 *
 * Advisory is the honest word. Nothing in this Worker deletes on a timer: enforcement is
 * the bucket's prefix-scoped R2 lifecycle rule (scripts/set_r2_lifecycle.sh), which is
 * age-based and cannot express a per-object lifetime. Recording the intent means the API
 * can report it and a future scheduled sweep has something to act on. See "Retention" in
 * docs/confidential-generation.md - a Worker-internal pseudo-scheduler is deliberately not
 * implemented, because a timer that only fires while requests happen to arrive is worse
 * than no promise at all.
 */
function resolveExpiry(body) {
  const hasSeconds = hasNonEmptyValue(body.retentionSeconds);
  const hasAt = hasNonEmptyValue(body.expiresAt);

  if (hasSeconds && hasAt) {
    throw new HttpError(400, "Supply retentionSeconds or expiresAt, not both");
  }
  if (!hasSeconds && !hasAt) return null;

  if (hasSeconds) {
    const seconds = toInteger(body.retentionSeconds, NaN);
    if (!Number.isFinite(seconds) || seconds < MIN_RETENTION_SECONDS || seconds > MAX_RETENTION_SECONDS) {
      throw new HttpError(
        400,
        `retentionSeconds must be between ${MIN_RETENTION_SECONDS} and ${MAX_RETENTION_SECONDS}`
      );
    }
    return new Date(Date.now() + seconds * 1000).toISOString();
  }

  const at = new Date(String(body.expiresAt));
  if (Number.isNaN(at.getTime())) {
    throw new HttpError(400, "expiresAt must be an ISO-8601 timestamp");
  }
  const delta = (at.getTime() - Date.now()) / 1000;
  if (delta < MIN_RETENTION_SECONDS || delta > MAX_RETENTION_SECONDS) {
    throw new HttpError(
      400,
      `expiresAt must be between ${MIN_RETENTION_SECONDS} seconds and ` +
        `${MAX_RETENTION_SECONDS} seconds from now`
    );
  }
  return at.toISOString();
}

function firstPresent(...values) {
  for (const value of values) {
    if (hasNonEmptyValue(value)) return value;
  }
  return undefined;
}

function nearestLegalFrames(n) {
  const up = alignFrameCount(n);
  let down = up;
  while (down - FRAME_GRID_MODULUS >= MIN_FRAMES) {
    down -= FRAME_GRID_MODULUS;
    if (down <= n) break;
  }
  return down === up ? [up] : [down, up];
}

function validateCanvas(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new HttpError(400, "width and height must be positive integers");
  }
  if (width % CANVAS_MULTIPLE !== 0 || height % CANVAS_MULTIPLE !== 0) {
    throw new HttpError(400, `width and height must be multiples of ${CANVAS_MULTIPLE}`);
  }
  if (width * height > MAX_PIXELS) {
    throw new HttpError(
      400,
      `requested canvas ${width}x${height} exceeds the H3 maximum area of ${MAX_PIXELS} pixels`
    );
  }
}

/*
 * Keyframe assets.
 *
 * `url` is passed through to the RunPod handler's existing image_url path, which already
 * validates scheme, host, redirects, size and content. `r2_key` is accepted and validated
 * here but cannot be delivered yet - the Worker has no R2 binding wired up, so returning
 * 501 is honest where silently ignoring it would produce a video with no keyframe.
 */
export function normalizeAsset(value, fieldName) {
  if (!isAssetReference(value)) {
    return null;
  }

  const hasUrl = hasNonEmptyValue(value.url);
  const hasKey = hasNonEmptyValue(value.r2_key);
  const hasAssetId = hasNonEmptyValue(value.asset_id);
  const hasBase64 = hasNonEmptyValue(value.base64);

  if ([hasUrl, hasKey || hasAssetId, hasBase64].filter(Boolean).length > 1) {
    throw new HttpError(
      400,
      `${fieldName} must specify exactly one of url, r2_key/asset_id or base64`
    );
  }

  if (hasAssetId) {
    return { kind: "r2", value: String(value.asset_id).trim() };
  }

  if (hasKey) {
    // Validated, then reduced to its asset id: RunPod is handed a route, never a key, so
    // it can never address an arbitrary object in the bucket.
    const key = validateR2Key(String(value.r2_key), fieldName);
    const base = key.split("/").pop() || "";
    const assetId = base.replace(/\.[A-Za-z0-9]+$/, "");
    if (!assetId) {
      throw new HttpError(400, `${fieldName}.r2_key has no usable asset name`);
    }
    return { kind: "r2", value: assetId };
  }

  if (hasBase64) {
    return { kind: "base64", value: String(value.base64) };
  }

  if (hasUrl) {
    const url = String(value.url).trim();
    if (!/^https:\/\//i.test(url)) {
      throw new HttpError(400, `${fieldName}.url must be an https URL`);
    }
    return { kind: "url", value: url };
  }

  throw new HttpError(400, `${fieldName} must specify url, r2_key or base64`);
}

/**
 * Validate a client-supplied R2 key. Never trusted verbatim: no traversal, no absolute
 * paths, no wildcards, and it must sit under a known prefix.
 */
export function validateR2Key(key, fieldName = "r2_key") {
  const value = String(key || "").trim();

  if (!value) {
    throw new HttpError(400, `${fieldName} must not be empty`);
  }
  if (value.length > 1024) {
    throw new HttpError(400, `${fieldName} is too long`);
  }
  if (value.startsWith("/") || value.includes("..") || value.includes("\\")) {
    throw new HttpError(400, `${fieldName} must be a relative key without '..' or '\\'`);
  }
  if (!/^[A-Za-z0-9._\-/]+$/.test(value)) {
    throw new HttpError(400, `${fieldName} contains characters outside [A-Za-z0-9._-/]`);
  }
  if (value.includes("//")) {
    throw new HttpError(400, `${fieldName} must not contain empty path segments`);
  }
  if (!value.startsWith("inputs/")) {
    throw new HttpError(400, `${fieldName} must start with 'inputs/'`);
  }
  return value;
}

/* ------------------------------------------------------------------------------------
 * Workflow templates and routing
 *
 * One template per mode rather than a single graph with every branch in it. Node keys are
 * internal names, never exposed in the public API.
 * ---------------------------------------------------------------------------------- */

/** Nodes shared by every FL2VA mode. */
function baseFl2vaTemplate() {
  return {
    unet: {
      class_type: "UNETLoader",
      inputs: { unet_name: MODEL_FILES.unet, weight_dtype: "default" }
    },
    clip: {
      class_type: "CLIPLoader",
      inputs: { clip_name: MODEL_FILES.clip, type: "minimax", device: "default" }
    },
    vae_video: {
      class_type: "VAELoader",
      inputs: { vae_name: MODEL_FILES.videoVae }
    },
    vae_audio: {
      class_type: "VAELoader",
      inputs: { vae_name: MODEL_FILES.audioVae }
    },
    cond: {
      class_type: "MiniMaxH3ImageToVideo",
      inputs: {
        clip: ["clip", 0],
        vae: ["vae_video", 0],
        prompt: "",
        width: 0,
        height: 0,
        length: 0
      }
    },
    noise: { class_type: "RandomNoise", inputs: { noise_seed: 0 } },
    guider: {
      class_type: "BasicGuider",
      inputs: { model: ["unet", 0], conditioning: ["cond", 0] }
    },
    sampler_select: {
      class_type: "KSamplerSelect",
      inputs: { sampler_name: "res_multistep" }
    },
    sigmas: {
      class_type: "BasicScheduler",
      inputs: { model: ["unet", 0], scheduler: "simple", steps: 0, denoise: 1.0 }
    },
    sample: {
      class_type: "SamplerCustomAdvanced",
      inputs: {
        noise: ["noise", 0],
        guider: ["guider", 0],
        sampler: ["sampler_select", 0],
        sigmas: ["sigmas", 0],
        latent_image: ["cond", 1]
      }
    },
    decode_video: {
      class_type: "VAEDecode",
      inputs: { samples: ["sample", 0], vae: ["vae_video", 0] }
    },
    decode_audio: {
      class_type: "VAEDecodeAudio",
      inputs: { samples: ["sample", 0], vae: ["vae_audio", 0] }
    },
    video: {
      class_type: "CreateVideo",
      inputs: { images: ["decode_video", 0], fps: FPS, audio: ["decode_audio", 0] }
    },
    save: {
      class_type: "SaveVideo",
      inputs: {
        video: ["video", 0],
        filename_prefix: "video/H3_private",
        format: "auto",
        codec: "auto"
      }
    }
  };
}

/**
 * Return the graph template for a mode. Keyframe modes add LoadImage nodes wired into
 * MiniMaxH3ImageToVideo's optional first_frame / last_frame inputs - the exact input
 * names from the node's own schema.
 */
export function loadWorkflowTemplate(mode) {
  const spec = MODES[mode];
  if (!spec) {
    throw new HttpError(400, `Unknown mode '${mode}'`);
  }
  if (!spec.implemented) {
    throw new HttpError(501, `Mode '${mode}' is not available yet. ${spec.reason}`);
  }

  const template = baseFl2vaTemplate();

  if (spec.firstFrame) {
    template.first_frame_image = {
      class_type: "LoadImage",
      inputs: { image: "placeholder.png" }
    };
    template.cond.inputs.first_frame = ["first_frame_image", 0];
  }

  if (spec.lastFrame) {
    template.last_frame_image = {
      class_type: "LoadImage",
      inputs: { image: "placeholder.png" }
    };
    template.cond.inputs.last_frame = ["last_frame_image", 0];
  }

  return template;
}

/** Stamp resolved settings onto a template. Pure: returns a new graph. */
export function applySettings(template, settings) {
  const workflow = JSON.parse(JSON.stringify(template));

  workflow.cond.inputs.prompt = settings.prompt;
  workflow.cond.inputs.width = settings.width;
  workflow.cond.inputs.height = settings.height;
  workflow.cond.inputs.length = settings.frames;
  workflow.noise.inputs.noise_seed = settings.seed;
  workflow.sigmas.inputs.steps = settings.steps;
  workflow.video.inputs.fps = settings.fps;

  return workflow;
}

export function buildWorkflowForSettings(settings) {
  return applySettings(loadWorkflowTemplate(settings.mode), settings);
}

function describeCapabilities() {
  const qualities = {};
  for (const [name, preset] of Object.entries(QUALITY_PRESETS)) {
    const dimensions = {};
    for (const [ratioName, ratio] of Object.entries(ASPECT_RATIOS)) {
      const canvas = adaptCanvas(ratio, preset.shortEdge);
      dimensions[ratioName] = `${canvas.width}x${canvas.height}`;
    }
    qualities[name] = { steps: preset.steps, dimensions };
  }

  return {
    fps: FPS,
    maxPixels: MAX_PIXELS,
    canvasMultiple: CANVAS_MULTIPLE,
    frameGrid: `frames % ${FRAME_GRID_MODULUS} == ${FRAME_GRID_REMAINDER}`,
    defaultQuality: DEFAULT_QUALITY,
    defaultAspectRatio: DEFAULT_ASPECT_RATIO,
    defaultDurationSeconds: DEFAULT_DURATION_SECONDS,
    qualities,
    aspectRatios: Object.keys(ASPECT_RATIOS),
    modes: Object.fromEntries(
      Object.entries(MODES).map(([name, spec]) => [
        name,
        spec.implemented ? { available: true } : { available: false, reason: spec.reason }
      ])
    )
  };
}

/* ------------------------------------------------------------------------------------
 * Routes
 * ---------------------------------------------------------------------------------- */

// `url` is needed for the callback origin, so it is passed in rather than re-parsed.
async function generateVideo(request, env, headers, url) {
  const body = await readJsonObject(request);
  const backend = resolveRequestedBackend(body);
  const config = getRunPodConfig(backend, env);

  const settings = normalizeRequest(body);
  const privacy = normalizePrivacy(body);
  const workflow = buildWorkflowForSettings(settings);

  /*
   * Our own job id, minted before submission because the callback tokens have to be
   * inside the payload we are about to send. RunPod's id is kept separately so the
   * existing /status/:id and /cancel/:id routes keep working untouched.
   */
  const jobId = crypto.randomUUID();
  const callbacks = env.JOB_TOKEN_SECRET
    ? await buildCallbackBlocks(env, jobId, url.origin, settings, privacy)
    : null;

  if (!callbacks) {
    // Worth saying out loud rather than silently degrading to a job with no realtime
    // channel and no R2 output.
    console.warn("JOB_TOKEN_SECRET is unset: progress callbacks and R2 output are disabled");
  }

  /*
   * Fail closed. Without callbacks the artefact comes back inline in RunPod's job result,
   * which means a copy of it living in a store we do not control - and, in confidential
   * mode, a promise we cannot keep. A 503 is the honest answer; running the job anyway and
   * hoping is not.
   */
  if (privacy.spec.encrypts && !callbacks) {
    throw new HttpError(
      503,
      "Confidential generation is unavailable on this deployment: JOB_TOKEN_SECRET is not " +
        "configured, so there is no authenticated path for the encrypted artefact to " +
        "reach storage. Standard generation is unaffected."
    );
  }

  await logGeneration({ jobId, backend, settings, privacy });

  const runpodUrl = `https://api.runpod.ai/v2/${config.endpointId}/run`;

  const response = await fetch(runpodUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey}`
    },
    body: JSON.stringify({
      input: buildRunPodInput(backend, workflow, settings, callbacks, privacy),
      policy: { executionTimeout: 600000, ttl: 1800000 }
    })
  });

  const data = await safeJson(response);

  if (!response.ok) {
    return runPodErrorResponse({
      operation: "submission",
      backend,
      upstreamStatus: response.status,
      data,
      headers
    });
  }

  if (!data || typeof data !== "object" || !data.id) {
    return json(
      {
        error: "RunPod submission returned no job ID",
        backend,
        details: sanitizeRunPodResult(data)
      },
      502,
      headers
    );
  }

  const encodedBackend = encodeURIComponent(backend);
  const encodedJobId = encodeURIComponent(data.id);

  // Seed the realtime channel so a client connecting before the worker starts still gets
  // a sensible first frame instead of an empty state.
  if (callbacks) {
    try {
      await pushJobState(env, jobId, {
        status: data.status ?? "IN_QUEUE",
        phase: "queued",
        percent: 0,
        runpodId: data.id,
        backend,
        // Non-secret, and recorded now so /status can describe the job's privacy before
        // there is any artefact to describe. Never the key: see redactJobState().
        privacyMode: privacy.mode,
        ...(privacy.expiresAt ? { expiresAt: privacy.expiresAt } : {})
      });
    } catch (error) {
      console.warn(`Could not seed job channel: ${error?.message || error}`);
    }
  }

  return json(
    {
      backend,
      id: data.id,
      jobId,
      status: data.status ?? null,
      seed: settings.seed,
      mode: settings.mode,
      quality: settings.quality,
      aspectRatio: settings.aspectRatio,

      /*
       * Echoed so a client can confirm what it got rather than what it asked for, and so
       * `standard` callers see the field appear without having to change anything. The
       * key is not here, is not anywhere in this response, and is not stored.
       */
      privacyMode: privacy.mode,
      ...(privacy.encryption
        ? {
            encryption: {
              algorithm: privacy.encryption.algorithm,
              ...(privacy.encryption.keyId ? { keyId: privacy.encryption.keyId } : {}),
              ...(privacy.encryption.kdf ? { kdf: privacy.encryption.kdf } : {})
            }
          }
        : {}),
      ...(privacy.expiresAt ? { expiresAt: privacy.expiresAt } : {}),

      // Preserved shape: width/height/frames/fps/durationSeconds/steps as before.
      settings: {
        width: settings.width,
        height: settings.height,
        frames: settings.frames,
        fps: settings.fps,
        durationSeconds: settings.durationSeconds,
        steps: settings.steps
      },

      resolvedFrom: settings.resolvedFrom,

      /*
       * status and cancel carry ?jobId= so those routes can also reach the realtime
       * channel, which is keyed on our job id rather than RunPod's. Both work without it
       * exactly as before - the parameter only adds the video route and the cancelled
       * event.
       */
      routes: {
        status: `/status/${encodedBackend}/${encodedJobId}?jobId=${encodeURIComponent(jobId)}`,
        cancel: `/cancel/${encodedBackend}/${encodedJobId}?jobId=${encodeURIComponent(jobId)}`,
        events: `/ws/jobs/${encodeURIComponent(jobId)}`,
        video: `/jobs/${encodeURIComponent(jobId)}/video`,
        deleteVideo: `/jobs/${encodeURIComponent(jobId)}/video`,
        // Mode-neutral names. `video` still works and still means the same thing; these
        // are what a client should use once it stops assuming the artefact is a playable
        // MP4, which a confidential artefact is not until the browser decrypts it.
        artifact: `/jobs/${encodeURIComponent(jobId)}/artifact`,
        deleteGeneration: `/jobs/${encodeURIComponent(jobId)}`
      }
    },
    202,
    headers
  );
}

/**
 * Mint the job-scoped blocks RunPod needs to call back in.
 *
 * Each token is bound to this job and one purpose, so the progress token cannot upload a
 * video and the output token cannot read another job's assets. RunPod holds nothing
 * permanent - the tokens expire on their own.
 */
async function buildCallbackBlocks(env, jobId, origin, settings, privacy) {
  const secret = env.JOB_TOKEN_SECRET;
  const ttl = Number(env.JOB_TOKEN_TTL_SECONDS || 3600);
  const base = String(env.PUBLIC_BASE_URL || origin).replace(/\/+$/, "");

  /*
   * The job's privacy mode is signed into its output token.
   *
   * This is what makes a downgrade impossible rather than merely unlikely. When the upload
   * arrives, the Worker does not ask the uploader what kind of job this was, and does not
   * look it up in a Durable Object that might be unavailable or stale - it reads the mode
   * out of the HMAC-signed token the uploader had to present. RunPod cannot change it
   * without the signing secret, which never leaves Cloudflare.
   */
  const blocks = {
    progress: {
      url: `${base}/internal/jobs/${jobId}/progress`,
      token: await signJobToken(secret, jobId, TOKEN_PURPOSES.progress, ttl),
      jobId
    },
    output: {
      url: `${base}/internal/jobs/${jobId}/output`,
      token: await signJobToken(secret, jobId, TOKEN_PURPOSES.output, ttl, {
        pm: privacy?.mode ?? DEFAULT_PRIVACY_MODE,
        ...(privacy?.expiresAt ? { exa: privacy.expiresAt } : {})
      })
    }
  };

  const assets = {};
  for (const role of ["firstFrame", "lastFrame"]) {
    const asset = settings[role];
    if (!asset) continue;
    const key = role === "firstFrame" ? "first_frame" : "last_frame";

    if (asset.kind === "r2") {
      assets[key] = {
        url: `${base}/internal/jobs/${jobId}/assets/${asset.value}`,
        token: await signJobToken(secret, jobId, TOKEN_PURPOSES.asset, ttl)
      };
    } else if (asset.kind === "url") {
      assets[key] = { url: asset.value };
    } else if (asset.kind === "base64") {
      assets[key] = { base64: asset.value };
    }
  }

  if (Object.keys(assets).length > 0) {
    blocks.assets = assets;
  }

  return blocks;
}

async function getJobStatus(backend, jobId, env, headers, url) {
  const config = getRunPodConfig(backend, env);
  const runpodUrl = `https://api.runpod.ai/v2/${config.endpointId}/status/${encodeURIComponent(jobId)}`;

  const response = await fetch(runpodUrl, {
    method: "GET",
    headers: { Authorization: `Bearer ${config.apiKey}` }
  });

  const data = await safeJson(response);

  if (!response.ok) {
    return runPodErrorResponse({
      operation: "status",
      backend,
      upstreamStatus: response.status,
      data,
      headers
    });
  }

  const result = addBackendToResult(backend, sanitizeRunPodResult(data));

  /*
   * Surface a playable route when the handler reported an R2-backed video. `?jobId=` is
   * the Worker-side id from /generate; without it we cannot name the object, because the
   * R2 key is keyed on our id rather than RunPod's.
   *
   * Note the base64 path is unchanged: sanitizeRunPodResult still strips it, exactly as
   * before. This adds a way to get the video, it does not remove one.
   */
  const workerJobId = url?.searchParams.get("jobId");
  if (workerJobId) {
    try {
      // One read of the job channel now answers three questions - privacy mode, artefact
      // description, deletion - where it previously answered only the last.
      const state = await readJobState(env, workerJobId);
      Object.assign(result, describeArtifactForStatus(workerJobId, state, result?.output?.video));
    } catch {
      /* an unusable job id simply means no artefact is advertised */
    }
  }

  return json(result, 200, headers);
}

/**
 * Build the privacy half of a status response from stored job state.
 *
 * Split out because it is pure and worth testing directly, and because §34-style migration
 * lives here: a job recorded before privacy modes existed has no `privacyMode` and no
 * `artifact`, and is reported as `standard` / `encrypted: false`. Old status records stay
 * valid and old clients keep seeing the `video` block they already read.
 *
 * The key is not a field this function could return. It is not in `state` - nothing ever
 * writes it there - and there is no branch below that looks for one.
 */
export function describeArtifactForStatus(jobId, state, runpodVideo) {
  const encoded = encodeURIComponent(jobId);
  const stored = (state && state.artifact) || null;
  const privacyMode = state?.privacyMode || stored?.privacyMode || DEFAULT_PRIVACY_MODE;
  const deleted = Boolean(state?.video?.deleted || stored?.deleted);

  // Present if the handler reported one, or if the upload endpoint recorded one. Either is
  // enough to say an artefact exists; neither on its own is enough to say it does not.
  const exists = Boolean(stored || (runpodVideo && typeof runpodVideo === "object"));
  const out = { privacyMode };

  if (state?.expiresAt || stored?.expiresAt) {
    out.expiresAt = stored?.expiresAt || state.expiresAt;
  }

  if (!exists) return out;

  if (deleted) {
    out.video = { deleted: true };
    out.artifact = { deleted: true, privacyMode, encrypted: Boolean(stored?.encrypted) };
    return out;
  }

  const encrypted = Boolean(stored?.encrypted);
  const size = stored?.size ?? runpodVideo?.size;
  const key = stored?.key ?? outputKey(jobId, encrypted);

  // Kept for every existing client that reads `video`. For a confidential job the URL is
  // the same one - it just serves ciphertext, and `artifact` below says so.
  out.video = { url: `/jobs/${encoded}/video`, key, size, deleted: false };

  out.artifact = {
    url: `/jobs/${encoded}/artifact`,
    key,
    size,
    deleted: false,
    privacyMode,
    encrypted,
    contentType: stored?.contentType || (encrypted ? ENCRYPTED_CONTENT_TYPE : "video/mp4"),
    originalContentType: stored?.originalContentType || "video/mp4"
  };

  if (encrypted) {
    out.artifact.encryptionVersion = stored?.encryptionVersion ?? CONTAINER_VERSION;
    out.artifact.algorithm = stored?.algorithm || CONTAINER_SUITES[1];
    // Everything the browser needs to derive the same key from the same passphrase - and
    // nothing that would let anyone else do it.
    if (stored?.kdf) out.artifact.kdf = stored.kdf;
    if (stored?.keyId) out.artifact.keyId = stored.keyId;
  }

  return out;
}

async function cancelJob(backend, jobId, env, headers, url) {
  const config = getRunPodConfig(backend, env);
  const runpodUrl = `https://api.runpod.ai/v2/${config.endpointId}/cancel/${encodeURIComponent(jobId)}`;

  const response = await fetch(runpodUrl, {
    method: "POST",
    headers: { Authorization: `Bearer ${config.apiKey}` }
  });

  const data = await safeJson(response);

  if (!response.ok) {
    return runPodErrorResponse({
      operation: "cancel",
      backend,
      upstreamStatus: response.status,
      data,
      headers
    });
  }

  /*
   * Tell the realtime channel too, so a browser watching the socket sees `cancelled`
   * instead of the progress simply stopping with no explanation. Cancellation itself is
   * unchanged - this is additive, and a DO failure must not turn a successful cancel into
   * an error response.
   */
  const workerJobId = url?.searchParams.get("jobId");
  if (workerJobId) {
    try {
      await pushJobState(env, workerJobId, {
        status: "CANCELLED",
        phase: "cancelled",
        runpodId: jobId
      });
    } catch (error) {
      console.warn(`Cancelled job ${jobId} but could not update its channel: ${error?.message}`);
    }
  }

  return json(addBackendToResult(backend, sanitizeRunPodResult(data)), 200, headers);
}

/*
 * The RunPod handler's contract is unchanged: it receives {workflow} and executes it.
 * Keyframes ride along in the fields handler.py already understands (image_url /
 * image_base64), so no image rebuild is needed for the modes enabled here.
 */
function buildRunPodInput(backend, workflow, settings, callbacks, privacy) {
  switch (backend) {
    case "h3":
    case "h3-blackwell": {
      const input = { workflow };

      /*
       * The one place the encryption key is ever written to an outbound payload.
       *
       * It has to be here: encryption must happen where the plaintext is, and that is
       * inside the inference container. The key is request-scoped and travels no further -
       * it is not stored in R2, KV, D1, a Durable Object or a log, and no response the API
       * produces contains it. What it does mean is that the key is present in the job
       * payload RunPod holds for that job's retention window, which is a real exposure and
       * is documented as one in docs/confidential-generation.md.
       */
      if (privacy) {
        input.privacy = { mode: privacy.mode };
        if (privacy.encryption) {
          input.encryption = {
            algorithm: privacy.encryption.algorithm,
            key: privacy.encryption.key,
            ...(privacy.encryption.keyId ? { keyId: privacy.encryption.keyId } : {}),
            ...(privacy.encryption.kdf ? { kdf: privacy.encryption.kdf } : {})
          };
        }
      }

      if (callbacks) {
        input.progress = callbacks.progress;
        input.output = callbacks.output;
        if (callbacks.assets) {
          // Tell the handler which loader node each keyframe belongs to, so it never has
          // to infer it from graph shape.
          input.assets = {};
          for (const [role, asset] of Object.entries(callbacks.assets)) {
            input.assets[role] = {
              ...asset,
              node_id: role === "first_frame" ? "first_frame_image" : "last_frame_image"
            };
          }
        }
      }

      return input;
    }

    default:
      throw new HttpError(400, `Unsupported backend: ${backend}`);
  }
}

function getRunPodConfig(backend, env) {
  const definition = BACKENDS[backend];

  if (!definition) {
    throw new HttpError(400, `Unsupported backend: ${backend}`);
  }

  const endpointId = String(env[definition.endpointIdEnv] || "").trim();
  let apiKey = String(env[definition.apiKeyEnv] || "").trim();

  if (!apiKey && definition.fallbackApiKeyEnv) {
    apiKey = String(env[definition.fallbackApiKeyEnv] || "").trim();
  }

  if (!endpointId) {
    throw new HttpError(500, `Missing Cloudflare variable: ${definition.endpointIdEnv}`);
  }

  if (!apiKey) {
    const acceptedNames = [definition.apiKeyEnv, definition.fallbackApiKeyEnv]
      .filter(Boolean)
      .join(" or ");
    throw new HttpError(500, `Missing Cloudflare secret: ${acceptedNames}`);
  }

  return { backend, endpointId, apiKey };
}

function isBackendConfigured(backend, env) {
  const definition = BACKENDS[backend];
  if (!definition) return false;

  const endpointId = String(env[definition.endpointIdEnv] || "").trim();
  const primaryApiKey = String(env[definition.apiKeyEnv] || "").trim();
  const fallbackApiKey = definition.fallbackApiKeyEnv
    ? String(env[definition.fallbackApiKeyEnv] || "").trim()
    : "";

  return Boolean(endpointId && (primaryApiKey || fallbackApiKey));
}

function resolveRequestedBackend(body) {
  const hasBackend = hasNonEmptyValue(body.backend);
  const hasModel = hasNonEmptyValue(body.model);

  const fromBackend = hasBackend ? normalizeBackend(body.backend) : null;
  const fromModel = hasModel ? normalizeBackend(body.model) : null;

  if (fromBackend && fromModel && fromBackend !== fromModel) {
    throw new HttpError(400, "backend and model select different RunPod endpoints");
  }

  return fromBackend || fromModel || DEFAULT_BACKEND;
}

function normalizeBackend(value) {
  const key = String(value ?? DEFAULT_BACKEND).trim().toLowerCase();
  const backend = BACKEND_ALIASES[key];

  if (!backend) {
    throw new HttpError(400, `Unknown backend '${key}'. Use 'h3' or 'h3-blackwell'.`);
  }

  return backend;
}

function parseJobRoute(segments, url, routeName) {
  if (segments.length === 2) {
    const selectedBackend =
      url.searchParams.get("backend") ?? url.searchParams.get("model") ?? DEFAULT_BACKEND;

    return {
      backend: normalizeBackend(selectedBackend),
      jobId: validateJobId(segments[1])
    };
  }

  if (segments.length === 3) {
    return {
      backend: normalizeBackend(segments[1]),
      jobId: validateJobId(segments[2])
    };
  }

  throw new HttpError(400, `Use /${routeName}/:jobId or /${routeName}/:backend/:jobId`);
}

function validateJobId(value) {
  const jobId = String(value || "").trim();

  if (!jobId) {
    throw new HttpError(400, "Missing job ID");
  }
  if (jobId.length > 256) {
    throw new HttpError(400, "Job ID is too long");
  }

  return jobId;
}

function splitPath(pathname) {
  return pathname
    .split("/")
    .filter(Boolean)
    .map((segment) => {
      try {
        return decodeURIComponent(segment);
      } catch {
        throw new HttpError(400, "Invalid URL encoding");
      }
    });
}

function isExactRoute(segments, name) {
  return segments.length === 1 && segments[0] === name;
}

function hasNonEmptyValue(value) {
  return value !== undefined && value !== null && String(value).trim() !== "";
}

async function readJsonObject(request) {
  let body;

  try {
    body = await request.json();
  } catch {
    throw new HttpError(400, "Request body must be valid JSON");
  }

  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw new HttpError(400, "Request body must be a JSON object");
  }

  return body;
}

function runPodErrorResponse({ operation, backend, upstreamStatus, data, headers }) {
  let status = 502;

  if (upstreamStatus === 429) {
    status = 429;
  } else if (upstreamStatus === 404 && (operation === "status" || operation === "cancel")) {
    status = 404;
  }

  return json(
    {
      error: `RunPod ${operation} failed`,
      backend,
      upstreamStatus,
      details: sanitizeRunPodResult(data)
    },
    status,
    headers
  );
}

function addBackendToResult(backend, result) {
  if (result && typeof result === "object" && !Array.isArray(result)) {
    return { backend, ...result };
  }
  return { backend, result };
}

function routeDescriptions() {
  return [
    "GET /health",
    "GET /capabilities",
    "POST /jobs/:jobId/assets",
    "GET /jobs/:jobId/artifact (alias: /video)",
    "DELETE /jobs/:jobId/artifact (alias: /video)",
    "DELETE /jobs/:jobId",
    "GET /ws/jobs/:jobId",
    "POST /generate",
    "GET /status/:backend/:jobId",
    "POST /cancel/:backend/:jobId",
    "GET /status/:jobId (legacy; defaults to h3)",
    "POST /cancel/:jobId (legacy; defaults to h3)"
  ];
}

/* ------------------------------------------------------------------------------------
 * Redaction and safe logging
 *
 * Two separate jobs, deliberately not merged:
 *
 *   redactSecrets()  cleans a *structure* - anything about to be logged, or echoed back
 *                    from an upstream service. Structural rather than a regex over a
 *                    serialized blob, so it can tell an R2 object key from a crypto key.
 *
 *   logGeneration()  emits an explicit, hand-picked set of fields. Nothing here ever
 *                    serializes a request body and hopes redaction catches everything;
 *                    that is backwards, and one new field would be enough to break it.
 * ---------------------------------------------------------------------------------- */

/** Field names whose value is secret wherever it appears. */
const SECRET_FIELDS = new Set([
  "encryptionkey",
  "encryption_key",
  "passphrase",
  "password",
  "secret",
  "authorization",
  "cf-access-client-id",
  "cf-access-client-secret",
  "cf_access_client_secret",
  "x-api-key",
  "api_key",
  "apikey",
  "token",
  "job_token_secret",
  "runpod_api_key",
  "dek",
  "data_key",
  "wrapped_key"
]);

/*
 * `key` is only a secret in a cryptographic context. Everywhere else in this Worker it is
 * an R2 object key - returned by the upload endpoint, shown in /status, and exactly what a
 * support question needs. Redacting it by name would destroy a useful field to protect one
 * that lives somewhere specific.
 */
const CRYPTO_PARENTS = new Set(["encryption", "crypto", "kdf", "confidential", "privacy"]);

/** Never redacted: opaque client-chosen labels whose whole purpose is to be visible. */
const NEVER_REDACT = new Set(["keyid", "key_id", "r2_key", "storage_key", "object_key"]);

const REDACTED = "[redacted]";

export function redactSecrets(value, parent = "", depth = 0) {
  if (depth > 12) return "[omitted: nesting too deep]";
  if (value === null || value === undefined) return value;

  if (Array.isArray(value)) {
    return value.map((item) => redactSecrets(item, parent, depth + 1));
  }

  if (typeof value === "object") {
    const out = {};
    for (const [name, child] of Object.entries(value)) {
      const lowered = String(name).trim().toLowerCase();
      if (NEVER_REDACT.has(lowered)) {
        out[name] = child;
      } else if (SECRET_FIELDS.has(lowered) || (lowered === "key" && CRYPTO_PARENTS.has(parent.toLowerCase()))) {
        out[name] = REDACTED;
      } else {
        out[name] = redactSecrets(child, name, depth + 1);
      }
    }
    return out;
  }

  return value;
}

/**
 * One structured line per submitted generation.
 *
 * Every field is chosen by hand and every one of them is non-secret. The prompt is not
 * here in any mode - only its length and a truncated digest, which is enough to correlate
 * two log lines about the same generation and not enough to reconstruct the text. That is
 * a correlation handle, not a privacy guarantee: prompts are low entropy, and anyone with
 * a candidate list can confirm a match by hashing it.
 */
async function logGeneration({ jobId, backend, settings, privacy }) {
  const fields = [
    `generation_id=${jobId}`,
    `privacy_mode=${privacy.mode}`,
    `backend=${backend}`,
    `mode=${settings.mode}`,
    `width=${settings.width}`,
    `height=${settings.height}`,
    `frames=${settings.frames}`,
    `steps=${settings.steps}`,
    `encryption=${privacy.encryption ? privacy.encryption.algorithm : "none"}`,
    `prompt_chars=${settings.userPrompt.length}`
  ];
  if (privacy.encryption?.keyId) fields.push(`key_id=${privacy.encryption.keyId}`);
  if (privacy.encryption?.kdf) fields.push(`kdf=${privacy.encryption.kdf.name}`);
  if (privacy.expiresAt) fields.push(`expires_at=${privacy.expiresAt}`);
  fields.push(`prompt_sha256=${await promptDigest(settings.userPrompt)}`);

  console.log(fields.join(" "));
}

/** First 16 hex characters of the prompt's SHA-256. See the caveat on logGeneration. */
async function promptDigest(prompt) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(prompt));
  return [...new Uint8Array(digest)]
    .slice(0, 8)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function corsHeaders() {
  return {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers":
      "Content-Type, Authorization, CF-Access-Client-Id, CF-Access-Client-Secret",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Max-Age": "86400",
    "Cache-Control": "no-store"
  };
}

/**
 * Reduce an upstream RunPod response to the fields this API promises, with binary payloads
 * stripped and any secret redacted.
 *
 * The redaction pass matters even though RunPod's status response does not normally echo a
 * job's input: RunPod holds that input, error paths are the least predictable part of any
 * upstream, and this is the one function every upstream body flows through. An encryption
 * key that somehow appeared in an upstream payload must not be able to reach a client by
 * being copied through here.
 */
function sanitizeRunPodResult(data) {
  return redactSecrets(sanitizeRunPodShape(data));
}

function sanitizeRunPodShape(data) {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return sanitizePayload(data, "result");
  }

  const result = {
    id: data.id ?? null,
    status: data.status ?? null,
    delayTime: data.delayTime ?? null,
    executionTime: data.executionTime ?? null,
    workerId: data.workerId ?? data.worker_id ?? null
  };

  if (data.error !== undefined) {
    result.error = sanitizePayload(data.error, "error");
  }

  if (data.errors !== undefined) {
    result.errors = sanitizePayload(data.errors, "errors");
  }

  if (data.output !== undefined) {
    result.output = sanitizePayload(data.output, "output");
  }

  for (const key of ["message", "detail", "code", "raw"]) {
    if (data[key] !== undefined) {
      result[key] = sanitizePayload(data[key], key);
    }
  }

  return result;
}

function sanitizePayload(value, key = "", depth = 0) {
  if (value === null || value === undefined) {
    return value;
  }

  if (depth > 12) {
    return "[omitted: nesting too deep]";
  }

  if (typeof value === "string") {
    if (shouldOmitBinaryString(key, value)) {
      return {
        omitted: true,
        reason: "binary/base64 payload removed",
        characterCount: value.length
      };
    }

    if (value.length > 10000) {
      return value.slice(0, 4000) + "...[truncated]";
    }

    return value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return value;
  }

  if (Array.isArray(value)) {
    const limit = 100;
    const sanitized = value.slice(0, limit).map((item) => sanitizePayload(item, key, depth + 1));

    if (value.length > limit) {
      sanitized.push({ omittedItems: value.length - limit });
    }

    return sanitized;
  }

  if (typeof value === "object") {
    const entries = Object.entries(value);
    const output = {};
    const limit = 100;

    for (const [childKey, childValue] of entries.slice(0, limit)) {
      output[childKey] = sanitizePayload(childValue, childKey, depth + 1);
    }

    if (entries.length > limit) {
      output.__omittedKeys = entries.length - limit;
    }

    return output;
  }

  return String(value);
}

function shouldOmitBinaryString(key, value) {
  if (isHttpUrl(value)) {
    return false;
  }

  const lowerKey = String(key).trim().toLowerCase();

  if (value.startsWith("data:") && value.includes(";base64,")) {
    return true;
  }

  if (BINARY_FIELD_NAMES.has(lowerKey) && value.length > 1024) {
    return true;
  }

  return value.length > 100000 && looksLikeBase64(value);
}

function isHttpUrl(value) {
  return /^https?:\/\//i.test(String(value).trim());
}

function looksLikeBase64(value) {
  const sample = String(value).slice(0, 4096).replace(/\s/g, "");
  return sample.length >= 1024 && /^[A-Za-z0-9+/=]+$/.test(sample);
}

function toInteger(value, fallback) {
  if (value === undefined || value === null || value === "") {
    return fallback;
  }

  const number = Number(value);
  return Number.isFinite(number) ? Math.trunc(number) : fallback;
}

function toNumber(value, fallback) {
  if (value === undefined || value === null || value === "") {
    return fallback;
  }

  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

async function safeJson(response) {
  const text = await response.text();

  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text);
  } catch {
    return { raw: text.slice(0, 2000) };
  }
}

function json(data, status, headers) {
  return new Response(JSON.stringify(data, null, 2), { status, headers });
}

/* ------------------------------------------------------------------------------------
 * Asset and video routes
 * ---------------------------------------------------------------------------------- */

async function uploadAsset(request, env, headers, jobId, url) {
  const bucket = requireBucket(env);
  const contentType = (request.headers.get("Content-Type") || "").split(";")[0].trim();

  const extension = ASSET_CONTENT_TYPES[contentType];
  if (!extension) {
    throw new HttpError(
      415,
      `Unsupported Content-Type '${contentType || "(none)"}'. Allowed: ` +
        Object.keys(ASSET_CONTENT_TYPES).join(", ")
    );
  }

  const declared = Number(request.headers.get("Content-Length") || 0);
  if (declared > MAX_ASSET_BYTES) {
    throw new HttpError(413, `Asset exceeds the ${MAX_ASSET_BYTES}-byte limit`);
  }

  const assetId = url.searchParams.get("id") || "first-frame";
  const key = inputKey(jobId, assetId, extension);

  // Read once so the true size is known - Content-Length is a claim, not a fact.
  const body = await request.arrayBuffer();
  if (body.byteLength > MAX_ASSET_BYTES) {
    throw new HttpError(413, `Asset exceeds the ${MAX_ASSET_BYTES}-byte limit`);
  }

  await bucket.put(key, body, { httpMetadata: { contentType } });

  return json(
    { asset: { id: assetId, key, contentType, size: body.byteLength } },
    201,
    headers
  );
}

/**
 * Stream the finished MP4 out of R2, with Range support so browsers can seek.
 *
 * The object body is handed to the Response directly and never read into a buffer, so
 * Worker memory does not scale with video size.
 */
/**
 * Serve a job's artefact from R2, with Range support so browsers can seek.
 *
 * Standard artefacts are streamed exactly as before: `video/mp4`, cacheable, seekable.
 *
 * Confidential artefacts are ciphertext and are labelled as such - `application/octet-
 * stream`, `no-store`, `nosniff`, and an attachment disposition - because nothing here can
 * play them and pretending otherwise would only produce a broken video element. The
 * cryptographic facts a client needs travel in `X-Artifact-*` headers, exposed to
 * cross-origin readers; the key is not among them and never will be.
 *
 * Which of the two it is comes from the object's own stored metadata, not from its name
 * and not from a lookup that could be stale.
 */
async function streamVideo(request, env, jobId) {
  const bucket = requireBucket(env);
  const range = request.headers.get("Range");
  const options = range ? { range: request.headers } : undefined;

  let object = null;
  let encrypted = false;

  // Standard first, so the common path costs exactly one R2 read as it always has.
  for (const [index, key] of outputKeyCandidates(jobId).entries()) {
    object = await bucket.get(key, options);
    if (object) {
      encrypted = index === 1;
      break;
    }
  }

  if (!object) {
    return new Response(JSON.stringify({ error: "No artefact found for this job" }), {
      status: 404,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" }
    });
  }

  const stored = object.customMetadata || {};
  // Metadata wins; the key name is the fallback for objects written before it existed.
  if (stored.encrypted !== undefined) {
    encrypted = stored.encrypted === "true";
  }

  const responseHeaders = new Headers();
  object.writeHttpMetadata(responseHeaders);
  responseHeaders.set("etag", object.httpEtag);
  responseHeaders.set("Accept-Ranges", "bytes");
  responseHeaders.set("Access-Control-Allow-Origin", "*");
  responseHeaders.set("X-Privacy-Mode", stored.privacyMode || (encrypted ? "confidential" : "standard"));

  if (encrypted) {
    responseHeaders.set("Content-Type", ENCRYPTED_CONTENT_TYPE);
    responseHeaders.set("X-Artifact-Encrypted", "true");
    responseHeaders.set("X-Artifact-Encryption-Version", stored.encryptionVersion || String(CONTAINER_VERSION));
    responseHeaders.set("X-Artifact-Algorithm", stored.algorithm || CONTAINER_SUITES[1]);
    responseHeaders.set("X-Artifact-Original-Content-Type", stored.originalContentType || "video/mp4");
    responseHeaders.set("Content-Disposition", 'attachment; filename="artifact.enc"');
    responseHeaders.set("X-Content-Type-Options", "nosniff");
    // Ciphertext is useless to a cache and its presence there is one more copy to lose
    // control of. The browser decrypts into memory; there is nothing to re-fetch.
    responseHeaders.set("Cache-Control", "private, no-store");
    responseHeaders.set(
      "Access-Control-Expose-Headers",
      "Content-Length, Content-Range, ETag, X-Privacy-Mode, X-Artifact-Encrypted, " +
        "X-Artifact-Encryption-Version, X-Artifact-Algorithm, X-Artifact-Original-Content-Type"
    );
  } else {
    responseHeaders.set("Content-Type", "video/mp4");
    responseHeaders.set("X-Artifact-Encrypted", "false");
    // Immutable: a job's video never changes once written.
    responseHeaders.set("Cache-Control", "private, max-age=3600, immutable");
    responseHeaders.set("Access-Control-Expose-Headers", "Content-Length, Content-Range, ETag");
  }

  // Gate on the *request* header, not object.range: R2 populates object.range even for a
  // full read, and answering an unconditional GET with 206 confuses some players.
  if (range && object.range && object.size !== undefined) {
    const offset = object.range.offset ?? 0;
    const length = object.range.length ?? object.size - offset;
    const end = offset + length - 1;
    responseHeaders.set("Content-Range", `bytes ${offset}-${end}/${object.size}`);
    return new Response(object.body, { status: 206, headers: responseHeaders });
  }

  return new Response(object.body, { status: 200, headers: responseHeaders });
}

/**
 * Delete one job's output from R2.
 *
 * Scoped by construction: the key is derived from the job id through outputKey(), so there
 * is no way to name an arbitrary object. There is deliberately no generic "delete this key"
 * primitive anywhere in this Worker - the only deletable object is the one this job owns.
 *
 * Idempotent. R2's delete does not fail on a missing key, and the job is marked deleted
 * either way, so repeated calls return the same answer. That matters because a retry after
 * a dropped connection must not look like an error.
 *
 * The generation itself stays COMPLETED. Removing an artefact afterwards is not a
 * retrospective failure of the job that produced it.
 */
/**
 * Delete everything one generation owns.
 *
 * `scope` is "output" for DELETE /jobs/:id/video and /artifact - the artefact only, which
 * is what those routes have always meant - or "generation" for DELETE /jobs/:id, which
 * also removes the input keyframes that generation uploaded.
 *
 * Deliberately one function for both privacy modes. A confidential artefact is deleted by
 * exactly the same code as a standard one, because "we forgot to handle deletion for the
 * encrypted kind" is precisely the bug a second code path invites.
 */
async function deleteGeneration(env, headers, jobId, scope = "output") {
  const bucket = requireBucket(env);
  const segment = safeJobSegment(jobId); // throws 400 on a malformed id, before touching R2

  const prefixes =
    scope === "generation" ? [`outputs/${segment}/`, `inputs/${segment}/`] : [`outputs/${segment}/`];

  const found = new Set();
  for (const prefix of prefixes) {
    for (const key of await listAllKeys(bucket, prefix)) {
      found.add(key);
    }
  }

  // The artefact's two possible names are deleted whether or not the listing mentioned
  // them, so a listing that is momentarily behind cannot leave ciphertext in the bucket.
  // They are not counted as removed - naming a key is not evidence an object was there.
  const doomed = new Set(found);
  for (const key of outputKeyCandidates(jobId)) doomed.add(key);

  const all = [...doomed];
  if (all.length) {
    // R2's delete accepts an array and does not fail on keys that are already gone, which
    // is what makes a repeated call safe rather than merely tolerated.
    await bucket.delete(all.length === 1 ? all[0] : all);
  }

  // Record it so /status stops advertising a URL that would now 404.
  try {
    await pushJobState(env, jobId, {
      video: { key: outputKey(jobId), deleted: true },
      artifact: { deleted: true }
    });
  } catch (error) {
    // The objects are gone either way; a channel that cannot be updated must not turn a
    // successful deletion into a failure the caller will retry forever.
    console.warn(`Deleted ${found.size} object(s) but could not update the channel for ` +
      `job ${segment}: ${error?.message || error}`);
  }

  return json({ id: jobId, deleted: true, scope, removed: found.size }, 200, headers);
}

/** Every key under a prefix, following R2's cursor. */
async function listAllKeys(bucket, prefix) {
  if (typeof bucket.list !== "function") {
    throw new HttpError(500, "R2 binding does not support list(); cannot delete safely");
  }

  const keys = [];
  let cursor;
  do {
    const page = await bucket.list({ prefix, cursor, limit: 1000 });
    for (const object of page?.objects || []) keys.push(object.key);
    cursor = page?.truncated ? page.cursor : undefined;
  } while (cursor);

  return keys;
}

async function openJobSocket(request, env, jobId) {
  const stub = jobChannel(env, jobId);
  return stub.fetch(new Request("https://do/ws", { headers: request.headers }));
}

/* ------------------------------------------------------------------------------------
 * Internal routes (RunPod -> Cloudflare)
 * ---------------------------------------------------------------------------------- */

async function receiveProgress(request, env, headers, jobId) {
  await verifyJobToken(env.JOB_TOKEN_SECRET, bearerToken(request), jobId, TOKEN_PURPOSES.progress);

  const body = await readJsonObject(request);

  // Only the fields the public protocol defines are persisted - RunPod cannot inject
  // arbitrary keys into what clients receive.
  const update = {};
  for (const field of ["phase", "step", "steps", "percent", "error"]) {
    if (body[field] !== undefined) update[field] = body[field];
  }
  if (update.phase === "completed" || update.phase === "failed") {
    update.status = update.phase === "completed" ? "COMPLETED" : "FAILED";
  }
  if (body.video && typeof body.video === "object") {
    update.video = { key: outputKey(jobId), size: body.video.size };
  }

  await pushJobState(env, jobId, update);
  return json({ ok: true }, 200, headers);
}

/**
 * Receive a finished artefact from RunPod and stream it into R2.
 *
 * For a confidential job this is the last gate before persistent storage, and it is the
 * one that has to hold even if the inference worker is buggy or replaced. Three things are
 * checked, in this order:
 *
 *   1. the token is valid for this job and this purpose (unchanged);
 *   2. the job's privacy mode comes from the *signed token*, not from the uploader;
 *   3. if that mode encrypts, the first bytes of the body must parse as a confidential
 *      container whose authenticated header names this exact job.
 *
 * A plaintext MP4 offered for a confidential job is rejected with a 422 and never written.
 * There is deliberately no override, no query parameter and no header that relaxes this.
 */
async function receiveOutput(request, env, headers, jobId) {
  const claims = await verifyJobToken(
    env.JOB_TOKEN_SECRET,
    bearerToken(request),
    jobId,
    TOKEN_PURPOSES.output
  );

  const bucket = requireBucket(env);
  const mode = PRIVACY_MODES[claims.pm] ? claims.pm : DEFAULT_PRIVACY_MODE;
  const spec = PRIVACY_MODES[mode];

  if (!request.body) {
    throw new HttpError(400, "Expected a raw artefact body");
  }

  let body = request.body;
  let artifact = { privacyMode: mode, encrypted: false, contentType: "video/mp4" };

  if (spec.encrypts) {
    // Peek only as far as the header. The payload itself is never buffered, so Worker
    // memory stays flat regardless of video size.
    const peeked = await peekStream(body, CONTAINER_PREAMBLE_BYTES + CONTAINER_MAX_HEADER_BYTES);
    body = peeked.stream;

    let container;
    try {
      container = parseContainerPrefix(peeked.prefix);
    } catch (error) {
      throw new HttpError(
        422,
        `Refusing to store this artefact: job ${jobId} is confidential, and the uploaded ` +
          `bytes are not an encrypted container (${error.message}). Nothing was written.`
      );
    }

    /*
     * The header is the AEAD's associated data, so an attacker cannot rewrite artifactId
     * without breaking authentication. Checking it here means a ciphertext produced for
     * another generation cannot be filed under this one.
     */
    if (container.header.artifactId && container.header.artifactId !== jobId) {
      throw new HttpError(
        422,
        `Refusing to store this artefact: its authenticated header names generation ` +
          `'${container.header.artifactId}', not '${jobId}'.`
      );
    }

    artifact = {
      privacyMode: mode,
      encrypted: true,
      encryptionVersion: container.version,
      algorithm: container.algorithm,
      contentType: ENCRYPTED_CONTENT_TYPE,
      originalContentType: container.header.contentType || "application/octet-stream",
      ...(container.header.kdf ? { kdf: container.header.kdf } : {}),
      ...(container.header.keyId ? { keyId: container.header.keyId } : {})
    };
  }

  const key = outputKey(jobId, artifact.encrypted);
  const expiresAt = typeof claims.exa === "string" ? claims.exa : undefined;

  // The stream goes straight into R2: the artefact is never buffered in Worker memory, and
  // the key comes from us, never from the uploader.
  const object = await bucket.put(key, body, {
    httpMetadata: { contentType: artifact.contentType },
    // Non-secret, and the reason an object recovered on its own is still identifiable.
    // No key material appears here - there is no branch of this function that could put
    // it here, because the Worker never holds the key at upload time.
    customMetadata: {
      privacyMode: artifact.privacyMode,
      encrypted: String(artifact.encrypted),
      ...(artifact.encrypted
        ? {
            encryptionVersion: String(artifact.encryptionVersion),
            algorithm: artifact.algorithm,
            originalContentType: artifact.originalContentType
          }
        : {}),
      ...(expiresAt ? { expiresAt } : {})
    }
  });

  const declaredSize = Number(request.headers.get("Content-Length") || 0);
  const size = object?.size ?? (declaredSize > 0 ? declaredSize : undefined);

  await pushJobState(env, jobId, {
    privacyMode: artifact.privacyMode,
    video: { key, size },
    artifact: { ...artifact, key, size, deleted: false, ...(expiresAt ? { expiresAt } : {}) }
  });

  return json(
    {
      key,
      size,
      url: `/jobs/${jobId}/artifact`,
      privacyMode: artifact.privacyMode,
      encrypted: artifact.encrypted,
      contentType: artifact.contentType
    },
    201,
    headers
  );
}

/**
 * Read the first `limit` bytes of a stream without consuming it.
 *
 * Returns the bytes seen so far plus a stream that replays them followed by the rest, so
 * the caller can inspect a header and still hand the whole body to R2 unbuffered.
 */
async function peekStream(stream, limit) {
  const reader = stream.getReader();
  const chunks = [];
  let seen = 0;

  while (seen < limit) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    seen += value.byteLength;
  }

  const prefix = new Uint8Array(seen);
  let offset = 0;
  for (const chunk of chunks) {
    prefix.set(chunk, offset);
    offset += chunk.byteLength;
  }

  const replay = new ReadableStream({
    async pull(controller) {
      if (chunks.length) {
        controller.enqueue(chunks.shift());
        return;
      }
      const { value, done } = await reader.read();
      if (done) {
        controller.close();
        reader.releaseLock();
        return;
      }
      controller.enqueue(value);
    },
    cancel(reason) {
      return reader.cancel(reason);
    }
  });

  return { prefix, stream: replay };
}

/**
 * Parse the framing and authenticated header of a confidential container.
 *
 * Mirrors artifacts.py::parse_container_prefix; tests/vectors/confidential_container.json
 * pins the layout both sides must agree on. Throws for anything that is not a container,
 * which is how "are these bytes really ciphertext?" is answered cheaply.
 */
export function parseContainerPrefix(data) {
  if (data.byteLength < CONTAINER_PREAMBLE_BYTES) {
    throw new Error(
      `too short: ${data.byteLength} bytes, need at least ${CONTAINER_PREAMBLE_BYTES}`
    );
  }

  const magic = String.fromCharCode(data[0], data[1], data[2], data[3]);
  if (magic !== CONTAINER_MAGIC) {
    throw new Error("missing container magic - these bytes are not ciphertext");
  }

  const version = data[4];
  const suite = data[5];
  if (version !== CONTAINER_VERSION) {
    throw new Error(`unsupported container version ${version}`);
  }
  if (!CONTAINER_SUITES[suite]) {
    throw new Error(`unsupported cipher suite ${suite}`);
  }

  const headerLength = (data[6] << 8) | data[7];
  if (headerLength === 0 || headerLength > CONTAINER_MAX_HEADER_BYTES) {
    throw new Error(`illegal header length ${headerLength}`);
  }

  const headerEnd = CONTAINER_PREAMBLE_BYTES + headerLength;
  if (data.byteLength < headerEnd) {
    throw new Error(`truncated header: need ${headerEnd} bytes, have ${data.byteLength}`);
  }

  let header;
  try {
    header = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(
        data.subarray(CONTAINER_PREAMBLE_BYTES, headerEnd)
      )
    );
  } catch (error) {
    throw new Error(`header is not valid JSON: ${error.message}`);
  }

  if (!header || typeof header !== "object" || Array.isArray(header)) {
    throw new Error("header must be a JSON object");
  }

  return {
    version,
    suite,
    algorithm: CONTAINER_SUITES[suite],
    header,
    headerLength,
    nonceOffset: headerEnd,
    ciphertextOffset: headerEnd + CONTAINER_NONCE_BYTES
  };
}

async function serveAsset(request, env, jobId, assetId) {
  await verifyJobToken(env.JOB_TOKEN_SECRET, bearerToken(request), jobId, TOKEN_PURPOSES.asset);

  const bucket = requireBucket(env);

  // Try each allowed extension rather than trusting a caller-supplied filename.
  for (const extension of Object.values(ASSET_CONTENT_TYPES)) {
    const object = await bucket.get(inputKey(jobId, assetId, extension));
    if (object) {
      const responseHeaders = new Headers();
      object.writeHttpMetadata(responseHeaders);
      responseHeaders.set("Cache-Control", "no-store");
      return new Response(object.body, { status: 200, headers: responseHeaders });
    }
  }

  throw new HttpError(404, `No asset '${assetId}' for job ${jobId}`);
}

/* ------------------------------------------------------------------------------------
 * Job-scoped tokens
 *
 * RunPod needs to call back into this Worker to report progress, upload the finished MP4
 * and fetch input assets. Those endpoints must not be open to anyone who guesses the URL,
 * and RunPod must not hold a permanent credential.
 *
 * So each job is issued short-lived HMAC tokens that bind three things: the job id, the
 * purpose, and an expiry. A progress token cannot upload output; an output token for job A
 * cannot write to job B; and every token stops working on its own. The signing secret
 * lives only in Cloudflare and is never sent anywhere.
 * ---------------------------------------------------------------------------------- */

const TOKEN_PURPOSES = Object.freeze({
  progress: "progress",
  output: "output-upload",
  asset: "asset-download"
});

const TOKEN_DEFAULT_TTL_SECONDS = 3600;

function b64urlEncode(bytes) {
  let binary = "";
  for (const byte of new Uint8Array(bytes)) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlDecodeToBytes(value) {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded + "=".repeat((4 - (padded.length % 4)) % 4));
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

async function hmacKey(secret) {
  return crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
}

/**
 * Mint a token binding jobId + purpose + expiry, plus any extra signed claims.
 *
 * `claims` are authenticated the same way everything else in the payload is: they ride
 * inside the signed body, so the holder can read them but cannot change them. That is how
 * a job's privacy mode reaches the upload endpoint without a storage lookup.
 */
export async function signJobToken(
  secret,
  jobId,
  purpose,
  ttlSeconds = TOKEN_DEFAULT_TTL_SECONDS,
  claims = null
) {
  if (!secret) {
    throw new HttpError(500, "Missing Cloudflare secret: JOB_TOKEN_SECRET");
  }
  const payload = {
    jid: jobId,
    pur: purpose,
    exp: Math.floor(Date.now() / 1000) + ttlSeconds,
    ...(claims || {})
  };
  const body = b64urlEncode(new TextEncoder().encode(JSON.stringify(payload)));
  const signature = await crypto.subtle.sign("HMAC", await hmacKey(secret), new TextEncoder().encode(body));
  return `${body}.${b64urlEncode(signature)}`;
}

/**
 * Verify a token for an exact job and purpose. Returns the payload or throws 401/403.
 *
 * Signature is checked before anything in the payload is trusted, and compared with
 * `crypto.subtle.verify` rather than string equality so it is not timing-sensitive.
 */
export async function verifyJobToken(secret, token, jobId, purpose) {
  if (!secret) {
    throw new HttpError(500, "Missing Cloudflare secret: JOB_TOKEN_SECRET");
  }
  if (!token || typeof token !== "string" || !token.includes(".")) {
    throw new HttpError(401, "Missing or malformed job token");
  }

  const [body, signature] = token.split(".", 2);

  let valid = false;
  try {
    valid = await crypto.subtle.verify(
      "HMAC",
      await crypto.subtle.importKey(
        "raw",
        new TextEncoder().encode(secret),
        { name: "HMAC", hash: "SHA-256" },
        false,
        ["verify"]
      ),
      b64urlDecodeToBytes(signature),
      new TextEncoder().encode(body)
    );
  } catch {
    valid = false;
  }

  if (!valid) {
    throw new HttpError(401, "Invalid job token signature");
  }

  let payload;
  try {
    payload = JSON.parse(new TextDecoder().decode(b64urlDecodeToBytes(body)));
  } catch {
    throw new HttpError(401, "Malformed job token payload");
  }

  if (payload.exp && Math.floor(Date.now() / 1000) > payload.exp) {
    throw new HttpError(401, "Job token has expired");
  }
  if (payload.jid !== jobId) {
    throw new HttpError(403, "Job token is for a different job");
  }
  if (payload.pur !== purpose) {
    throw new HttpError(403, `Job token is for '${payload.pur}', not '${purpose}'`);
  }

  return payload;
}

function bearerToken(request) {
  const header = request.headers.get("Authorization") || "";
  return header.startsWith("Bearer ") ? header.slice(7).trim() : "";
}

/* ------------------------------------------------------------------------------------
 * R2 object naming
 *
 * Deterministic and entirely Worker-owned. RunPod never chooses a key - it uploads to a
 * route that already knows where the object belongs, which is what stops a compromised or
 * buggy worker writing anywhere in the bucket.
 *
 * There is no user namespace because this system has no user identity to namespace by;
 * inventing one here would be fiction. If auth is added later, these two helpers are the
 * only places that need to change.
 * ---------------------------------------------------------------------------------- */

/**
 * Where a job's finished artefact lives.
 *
 * Two names, one prefix. A confidential artefact is not an MP4 and must not be called one:
 * anyone looking at the bucket during an incident should be able to tell ciphertext from
 * video at a glance, and nothing should be able to hand `artifact.enc` to a `<video>` tag
 * by mistake. The prefix stays `outputs/` because the bucket's R2 lifecycle rules are
 * prefix-scoped - moving it would silently drop those artefacts out of retention.
 *
 * Neither name contains the prompt, the caller or anything else about the content.
 */
export function outputKey(jobId, encrypted = false) {
  return `outputs/${safeJobSegment(jobId)}/${encrypted ? "artifact.enc" : "video.mp4"}`;
}

/** Every name a job's artefact could have. Ordered so the common case is tried first. */
export function outputKeyCandidates(jobId) {
  return [outputKey(jobId, false), outputKey(jobId, true)];
}

export function inputKey(jobId, assetId, extension) {
  return `inputs/${safeJobSegment(jobId)}/${safeAssetId(assetId)}${extension}`;
}

function safeJobSegment(jobId) {
  const value = String(jobId || "").trim();
  if (!value || !/^[A-Za-z0-9._-]+$/.test(value) || value.length > 256) {
    throw new HttpError(400, "Invalid job id");
  }
  return value;
}

function safeAssetId(assetId) {
  const value = String(assetId || "").trim();
  if (!value || !/^[A-Za-z0-9._-]+$/.test(value) || value.length > 128) {
    throw new HttpError(400, "Invalid asset id");
  }
  return value;
}

const ASSET_CONTENT_TYPES = Object.freeze({
  "image/png": ".png",
  "image/jpeg": ".jpg",
  "image/webp": ".webp"
});

const MAX_ASSET_BYTES = 32 * 1024 * 1024;

/* ------------------------------------------------------------------------------------
 * Durable Object: one per job, the realtime source of truth
 * ---------------------------------------------------------------------------------- */

export class JobChannel {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async currentState() {
    return (await this.state.storage.get("state")) || null;
  }

  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname.endsWith("/ws")) {
      if (request.headers.get("Upgrade") !== "websocket") {
        return new Response("Expected websocket upgrade", { status: 426 });
      }

      const pair = new WebSocketPair();
      const [client, server] = Object.values(pair);

      // Hibernation: the DO can be evicted between events and woken by the runtime, so a
      // long generation does not hold an instance alive doing nothing.
      this.state.acceptWebSocket(server);

      // Replay immediately so a client that connects late - or reconnects - is never
      // left staring at an empty screen waiting for the next event.
      const snapshot = await this.currentState();
      if (snapshot) {
        try {
          server.send(JSON.stringify({ type: "state", ...snapshot }));
        } catch {
          /* a socket that dies during replay is not an error worth failing on */
        }
      }

      return new Response(null, { status: 101, webSocket: client });
    }

    if (url.pathname.endsWith("/update") && request.method === "POST") {
      const update = await request.json();
      const previous = (await this.currentState()) || {};

      const next = {
        ...previous,
        ...update,
        jobId: update.jobId ?? previous.jobId,
        createdAt: previous.createdAt ?? new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      await this.state.storage.put("state", next);
      this.broadcast(next);
      return new Response(null, { status: 204 });
    }

    if (url.pathname.endsWith("/state")) {
      return new Response(JSON.stringify((await this.currentState()) || {}), {
        headers: { "Content-Type": "application/json" }
      });
    }

    return new Response("Not found", { status: 404 });
  }

  broadcast(state) {
    const message = JSON.stringify(publicEvent(state));
    for (const socket of this.state.getWebSockets()) {
      try {
        socket.send(message);
      } catch {
        /* the runtime reaps dead sockets; one bad peer must not stop the others */
      }
    }
  }

  // Clients are listeners, not commanders: nothing they send can change job state.
  async webSocketMessage() {}

  async webSocketClose(socket) {
    try {
      socket.close();
    } catch {
      /* already closed */
    }
  }
}

/** Map internal job state onto the stable public event shape. */
export function publicEvent(state) {
  const phase = state.phase || "queued";

  if (phase === "completed") {
    let video;
    if (state.video?.deleted) {
      // Deleted outputs are reported as such rather than as a URL that would 404.
      video = { deleted: true };
    } else if (state.video) {
      video = { url: `/jobs/${state.jobId}/video`, key: state.video.key, deleted: false };
    }
    return { type: "completed", jobId: state.jobId, video };
  }
  if (phase === "failed") {
    return {
      type: "failed",
      jobId: state.jobId,
      error: state.error || { code: "unknown", message: "Generation failed" }
    };
  }
  if (phase === "cancelled") {
    return { type: "cancelled", jobId: state.jobId };
  }

  const event = { type: "progress", jobId: state.jobId, phase };
  // Step counters belong to sampling only. State is merged, so without this the decoding
  // event would inherit the last sampler step and render as "decoding, step 12/20".
  if (phase === "sampling") {
    if (state.step !== undefined) event.step = state.step;
    if (state.steps !== undefined) event.steps = state.steps;
  }
  if (state.percent !== undefined) event.percent = state.percent;
  return event;
}

function jobChannel(env, jobId) {
  if (!env.JOB_CHANNEL) {
    throw new HttpError(500, "Missing Durable Object binding: JOB_CHANNEL");
  }
  return env.JOB_CHANNEL.get(env.JOB_CHANNEL.idFromName(safeJobSegment(jobId)));
}

/**
 * Read a job's recorded state.
 *
 * The Durable Object is the record of intent - it knows a video was deleted, where a
 * missing R2 object could equally mean the upload never happened. Any failure answers
 * `null`, so a channel problem degrades to "describe nothing extra" rather than hiding an
 * artefact that is really there.
 */
async function readJobState(env, jobId) {
  try {
    const response = await jobChannel(env, jobId).fetch(new Request("https://do/state"));
    const state = await response.json();
    return state && typeof state === "object" ? state : null;
  } catch {
    return null;
  }
}

async function pushJobState(env, jobId, update) {
  const stub = jobChannel(env, jobId);
  await stub.fetch(new Request("https://do/update", {
    method: "POST",
    body: JSON.stringify({ jobId, ...update }),
    headers: { "Content-Type": "application/json" }
  }));
}

function requireBucket(env) {
  // H3_OUTPUTS is the binding name already present on the deployed Worker; it is not a
  // name chosen here, and wrangler.toml matches it.
  if (!env.H3_OUTPUTS) {
    throw new HttpError(500, "Missing R2 binding: H3_OUTPUTS");
  }
  return env.H3_OUTPUTS;
}

// Exported for tests. Cloudflare only uses the default export and JobChannel above.
/*
 * Only functions and the Durable Object class may be named exports here: the Workers
 * runtime builds an export map from this module and rejects anything that is not a
 * function or an ExportedHandler ("Incorrect type for map entry"). Exporting a plain
 * constant stops the Worker booting at all, so the tables below are handed to tests
 * through a function instead.
 */
export function workerConstants() {
  return {
    MODES,
    QUALITY_PRESETS,
    ASPECT_RATIOS,
    MODEL_FILES,
    TOKEN_PURPOSES,
    ASSET_CONTENT_TYPES,
    MAX_ASSET_BYTES
  };
}

export { HttpError, describeCapabilities };
