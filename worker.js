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
        return await generateVideo(request, env, headers);
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

      // GET /jobs/:jobId/video  - stream the finished MP4 out of R2
      if (
        request.method === "GET" &&
        segments.length === 3 &&
        segments[0] === "jobs" &&
        segments[2] === "video"
      ) {
        return await streamVideo(request, env, segments[1]);
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

async function generateVideo(request, env, headers) {
  const body = await readJsonObject(request);
  const backend = resolveRequestedBackend(body);
  const config = getRunPodConfig(backend, env);

  const settings = normalizeRequest(body);
  const workflow = buildWorkflowForSettings(settings);

  /*
   * Our own job id, minted before submission because the callback tokens have to be
   * inside the payload we are about to send. RunPod's id is kept separately so the
   * existing /status/:id and /cancel/:id routes keep working untouched.
   */
  const jobId = crypto.randomUUID();
  const callbacks = env.JOB_TOKEN_SECRET
    ? await buildCallbackBlocks(env, jobId, url.origin, settings)
    : null;

  if (!callbacks) {
    // Worth saying out loud rather than silently degrading to a job with no realtime
    // channel and no R2 output.
    console.warn("JOB_TOKEN_SECRET is unset: progress callbacks and R2 output are disabled");
  }

  const runpodUrl = `https://api.runpod.ai/v2/${config.endpointId}/run`;

  const response = await fetch(runpodUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey}`
    },
    body: JSON.stringify({
      input: buildRunPodInput(backend, workflow, settings, callbacks),
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
        backend
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
        video: `/jobs/${encodeURIComponent(jobId)}/video`
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
async function buildCallbackBlocks(env, jobId, origin, settings) {
  const secret = env.JOB_TOKEN_SECRET;
  const ttl = Number(env.JOB_TOKEN_TTL_SECONDS || 3600);
  const base = String(env.PUBLIC_BASE_URL || origin).replace(/\/+$/, "");

  const blocks = {
    progress: {
      url: `${base}/internal/jobs/${jobId}/progress`,
      token: await signJobToken(secret, jobId, TOKEN_PURPOSES.progress, ttl),
      jobId
    },
    output: {
      url: `${base}/internal/jobs/${jobId}/output`,
      token: await signJobToken(secret, jobId, TOKEN_PURPOSES.output, ttl)
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
  const video = result?.output?.video;
  if (workerJobId && video && typeof video === "object") {
    try {
      result.video = {
        url: `/jobs/${encodeURIComponent(workerJobId)}/video`,
        key: outputKey(workerJobId),
        size: video.size
      };
    } catch {
      /* an unusable job id simply means no video route is advertised */
    }
  }

  return json(result, 200, headers);
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
function buildRunPodInput(backend, workflow, settings, callbacks) {
  switch (backend) {
    case "h3":
    case "h3-blackwell": {
      const input = { workflow };

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
    "POST /generate",
    "GET /status/:backend/:jobId",
    "POST /cancel/:backend/:jobId",
    "GET /status/:jobId (legacy; defaults to h3)",
    "POST /cancel/:jobId (legacy; defaults to h3)"
  ];
}

function corsHeaders() {
  return {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers":
      "Content-Type, Authorization, CF-Access-Client-Id, CF-Access-Client-Secret",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Max-Age": "86400",
    "Cache-Control": "no-store"
  };
}

function sanitizeRunPodResult(data) {
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
async function streamVideo(request, env, jobId) {
  const bucket = requireBucket(env);
  const key = outputKey(jobId);
  const range = request.headers.get("Range");

  const object = await bucket.get(key, range ? { range: request.headers } : undefined);

  if (!object) {
    return new Response(JSON.stringify({ error: "Video not found for this job" }), {
      status: 404,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" }
    });
  }

  const responseHeaders = new Headers();
  object.writeHttpMetadata(responseHeaders);
  responseHeaders.set("Content-Type", "video/mp4");
  responseHeaders.set("etag", object.httpEtag);
  responseHeaders.set("Accept-Ranges", "bytes");
  responseHeaders.set("Access-Control-Allow-Origin", "*");
  // Immutable: a job's video never changes once written.
  responseHeaders.set("Cache-Control", "private, max-age=3600, immutable");

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

async function receiveOutput(request, env, headers, jobId) {
  await verifyJobToken(env.JOB_TOKEN_SECRET, bearerToken(request), jobId, TOKEN_PURPOSES.output);

  const bucket = requireBucket(env);
  const key = outputKey(jobId);

  if (!request.body) {
    throw new HttpError(400, "Expected a raw video/mp4 request body");
  }

  // The stream goes straight into R2: the video is never buffered in Worker memory, and
  // the key comes from us, never from the uploader.
  const object = await bucket.put(key, request.body, {
    httpMetadata: { contentType: "video/mp4" }
  });

  const declaredSize = Number(request.headers.get("Content-Length") || 0);
  const size = object?.size ?? (declaredSize > 0 ? declaredSize : undefined);

  await pushJobState(env, jobId, { video: { key, size } });

  return json({ key, size, url: `/jobs/${jobId}/video` }, 201, headers);
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

/** Mint a token binding jobId + purpose + expiry. */
export async function signJobToken(secret, jobId, purpose, ttlSeconds = TOKEN_DEFAULT_TTL_SECONDS) {
  if (!secret) {
    throw new HttpError(500, "Missing Cloudflare secret: JOB_TOKEN_SECRET");
  }
  const payload = { jid: jobId, pur: purpose, exp: Math.floor(Date.now() / 1000) + ttlSeconds };
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

export function outputKey(jobId) {
  return `outputs/${safeJobSegment(jobId)}/video.mp4`;
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
    return {
      type: "completed",
      jobId: state.jobId,
      video: state.video ? { url: `/jobs/${state.jobId}/video`, key: state.video.key } : undefined
    };
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
