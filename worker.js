var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// canonical.js
var CANONICAL_FIELDS = Object.freeze([
  "mode",
  "prompt",
  "quality",
  "aspectRatio",
  "duration",
  "seed",
  "generationMode",
  "firstFrame",
  "lastFrame",
  "references",
  "referenceFidelity",
  "camera",
  "style",
  "privacy",
  "encryption",
  "progress",
  "output"
]);
var EXPECTED_H3_BUILD = Object.freeze({
  sourceCommit: "470b907ae37abff249476c618fb829556b2286eb",
  imageTag: "cached-models-21",
  buildId: "36212956644-1"
});
// Rollback (cached-models-19): "668f146b1a6a6ac23d0fc99ef13a200a35786b77" / "cached-models-19" / "35968520283-1",
// digest sha256:df68e86bbf0250b863e5d8bf491acee99dab2ca99436077469fbec0e2ab8b15c.
// Rollback (cached-models-18): "dbb960222fa6942a7c8b3701c4b7fb5b2ec676c3" / "cached-models-18" / "35722315664-1",
// digest sha256:f502e6d89373af6657da2313b82bbc31939b1486e2753cc70da8f637e98a00c4.
var EXPECTED_CACHED_CANARY_RELEASE = Object.freeze({
  applicationRelease: Object.freeze({
    imageRepository: "registry.digitalocean.com/minimax-h3-image/privora-h3-runpod-worker",
    imageDigest: "sha256:6422784683b47142e03a7b9fb48cf6d4f6a5de6aec4cd8984fbd1849af47d93a"
  }),
  modelRelease: Object.freeze({
    repository: "CDitfort/privora-minimax-h3-models",
    revision: "ecb69a4211d74b5798398021003bccde02d63757",
    manifest: "multimodal-4-hf-cache-v1"
  }),
  gpuMode: "single",
  models: Object.freeze({ fl2va: true, ref2va: true }),
  byFamily: Object.freeze({
    fl2va: Object.freeze(["quality", "turbo", "turboFast"]),
    ref2va: Object.freeze(["quality", "turboFast"])
  })
});
var MEDIA_TYPES = Object.freeze({
  image: Object.freeze({
    maxBytes: 32 * 1024 * 1024,
    contentTypes: Object.freeze({ "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp" })
  }),
  video: Object.freeze({
    maxBytes: 200 * 1024 * 1024,
    contentTypes: Object.freeze({ "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm" })
  }),
  audio: Object.freeze({
    maxBytes: 64 * 1024 * 1024,
    contentTypes: Object.freeze({
      "audio/wav": ".wav",
      "audio/x-wav": ".wav",
      "audio/mpeg": ".mp3",
      "audio/flac": ".flac",
      "audio/ogg": ".ogg",
      "audio/mp4": ".m4a",
      "audio/x-m4a": ".m4a"
    })
  })
});
var MODE_FAMILY = Object.freeze({ create: "fl2va", animate: "fl2va", references: "ref2va", remix: "ref2va" });
var CAMERA = Object.freeze({
  shot: /* @__PURE__ */ new Set(["extreme_wide", "wide", "medium_wide", "medium", "medium_close", "close", "extreme_close"]),
  movement: /* @__PURE__ */ new Set(["static", "pan", "tilt", "dolly", "truck", "orbit", "crane", "handheld", "zoom", "push_in", "pull_out"]),
  strength: /* @__PURE__ */ new Set(["subtle", "moderate", "strong"]),
  speed: /* @__PURE__ */ new Set(["slow", "medium", "fast"])
});
var STYLE = Object.freeze({
  visual: /* @__PURE__ */ new Set(["cinematic", "documentary", "animation", "anime", "photoreal", "vintage", "noir"]),
  lighting: /* @__PURE__ */ new Set(["natural", "golden_hour", "high_key", "low_key", "neon", "candlelit", "overcast"]),
  motion: /* @__PURE__ */ new Set(["natural", "slow_motion", "timelapse", "hyperlapse"])
});
var CanonicalError = class extends Error {
  static {
    __name(this, "CanonicalError");
  }
  constructor(status, code, message, details) {
    super(message);
    this.name = "CanonicalError";
    this.status = status;
    this.code = code;
    if (details !== void 0) this.details = details;
  }
};
function fail(status, code, message, details) {
  throw new CanonicalError(status, code, message, details);
}
__name(fail, "fail");
function isObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
__name(isObject, "isObject");
function exactKeys(value, allowed, path) {
  if (!isObject(value)) fail(400, "INVALID_REQUEST", `${path} must be an object.`);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) fail(400, "UNKNOWN_FIELD", `Unknown field '${path}.${key}'.`, { field: `${path}.${key}` });
  }
}
__name(exactKeys, "exactKeys");
function requireString(value, field, max = 2e3) {
  const out = typeof value === "string" ? value.trim() : "";
  if (!out) fail(400, field === "prompt" ? "MISSING_PROMPT" : "INVALID_REQUEST", `${field} is required.`);
  if (out.length > max) fail(400, "INVALID_REQUEST", `${field} is too long.`);
  return out;
}
__name(requireString, "requireString");
function validateSemanticObject(value, allowed, path) {
  if (value === void 0) return;
  exactKeys(value, new Set(Object.keys(allowed)), path);
  for (const [key, raw] of Object.entries(value)) {
    if (raw === void 0 || raw === null || raw === "") continue;
    if (typeof raw !== "string" || !allowed[key].has(raw)) {
      fail(400, "INVALID_REQUEST", `Invalid ${path}.${key}.`);
    }
  }
}
__name(validateSemanticObject, "validateSemanticObject");
function capabilityDocument(value) {
  if (isObject(value?.capabilities)) return value.capabilities;
  return isObject(value) ? value : {};
}
__name(capabilityDocument, "capabilityDocument");
function isCanonicalCapabilityDocument(value) {
  const caps = capabilityDocument(value);
  const build = caps.worker?.build;
  return isObject(caps.modes) && ["create", "animate", "references", "remix"].every((mode) => isObject(caps.modes[mode])) && isObject(caps.quality) && ["draft", "standard", "hd", "2k"].every((quality) => isObject(caps.quality[quality])) && Array.isArray(caps.aspectRatios) && caps.aspectRatios.length > 0 && isObject(caps.references) && isObject(caps.references.roles) && ["image", "video", "audio"].every((type) => Array.isArray(caps.references.roles[type])) && isObject(caps.byFamily) && Array.isArray(caps.byFamily.fl2va) && Array.isArray(caps.byFamily.ref2va) && isObject(build) && ["sourceCommit", "imageTag", "buildId"].every((key) => typeof build[key] === "string" && build[key].length > 0);
}
__name(isCanonicalCapabilityDocument, "isCanonicalCapabilityDocument");
function buildMatchesExpected(capabilities) {
  const actual = capabilityDocument(capabilities)?.worker?.build || {};
  return Object.entries(EXPECTED_H3_BUILD).every(([key, value]) => actual[key] === value);
}
__name(buildMatchesExpected, "buildMatchesExpected");
function buildIdentity(capabilities) {
  const build = capabilityDocument(capabilities)?.worker?.build || {};
  return {
    sourceCommit: typeof build.sourceCommit === "string" ? build.sourceCommit : null,
    imageTag: typeof build.imageTag === "string" ? build.imageTag : null,
    buildId: typeof build.buildId === "string" ? build.buildId : null
  };
}
__name(buildIdentity, "buildIdentity");
function stringOrNull(value) {
  return typeof value === "string" && value.length > 0 ? value : null;
}
__name(stringOrNull, "stringOrNull");
function cachedCanaryReleaseIdentity(capabilities) {
  const caps = capabilityDocument(capabilities);
  const worker = isObject(caps.worker) ? caps.worker : {};
  const applicationRelease = isObject(worker.applicationRelease) ? worker.applicationRelease : {};
  const modelRelease = isObject(worker.modelRelease) ? worker.modelRelease : {};
  return {
    applicationRelease: {
      imageRepository: stringOrNull(applicationRelease.imageRepository),
      imageDigest: stringOrNull(applicationRelease.imageDigest)
    },
    modelRelease: {
      repository: stringOrNull(modelRelease.repository),
      revision: stringOrNull(modelRelease.revision),
      manifest: stringOrNull(modelRelease.manifest)
    },
    gpuMode: stringOrNull(worker.gpuMode),
    models: {
      fl2va: caps.models?.fl2va === true,
      ref2va: caps.models?.ref2va === true
    },
    byFamily: {
      fl2va: Array.isArray(caps.byFamily?.fl2va) ? [...caps.byFamily.fl2va] : [],
      ref2va: Array.isArray(caps.byFamily?.ref2va) ? [...caps.byFamily.ref2va] : []
    }
  };
}
__name(cachedCanaryReleaseIdentity, "cachedCanaryReleaseIdentity");
function cachedCanaryReleaseMatchesExpected(capabilities) {
  const actual = cachedCanaryReleaseIdentity(capabilities);
  const expected = EXPECTED_CACHED_CANARY_RELEASE;
  // Registry mirrors may give the same immutable application bytes a different
  // repository label. Preserve that runtime-reported label, but pin the release
  // gate to the digest plus the exact cached-model and capability identities.
  return actual.applicationRelease.imageDigest === expected.applicationRelease.imageDigest && actual.modelRelease.repository === expected.modelRelease.repository && actual.modelRelease.revision === expected.modelRelease.revision && actual.modelRelease.manifest === expected.modelRelease.manifest && actual.gpuMode === expected.gpuMode && actual.models.fl2va === expected.models.fl2va && actual.models.ref2va === expected.models.ref2va && actual.byFamily.fl2va.length === expected.byFamily.fl2va.length && actual.byFamily.fl2va.every((mode, index) => mode === expected.byFamily.fl2va[index]) && actual.byFamily.ref2va.length === expected.byFamily.ref2va.length && actual.byFamily.ref2va.every((mode, index) => mode === expected.byFamily.ref2va[index]);
}
__name(cachedCanaryReleaseMatchesExpected, "cachedCanaryReleaseMatchesExpected");
function validateMediaHandle(value, path, roles, expectedType) {
  exactKeys(value, /* @__PURE__ */ new Set(["id", "type", "role", "soundtrack"]), path);
  const type = String(value.type || "");
  if (!MEDIA_TYPES[type] || expectedType && type !== expectedType) {
    fail(400, "INVALID_REFERENCE_TYPE", `${path}.type is invalid.`);
  }
  const id = requireString(value.id, `${path}.id`, 128);
  if (!/^[A-Za-z0-9._-]+$/.test(id)) fail(400, "INVALID_REFERENCE_TYPE", `${path}.id is invalid.`);
  const validRoles = Array.isArray(roles?.[type]) ? roles[type] : [];
  const role = String(value.role || "general");
  if (!validRoles.includes(role)) fail(400, "INVALID_REFERENCE_ROLE", `${path}.role is invalid for ${type}.`);
  if (value.soundtrack !== void 0) {
    if (type !== "video") fail(400, "INVALID_REFERENCE_TYPE", `${path}.soundtrack is only valid on video references.`);
    validateMediaHandle(value.soundtrack, `${path}.soundtrack`, roles, "audio");
    if (value.soundtrack.soundtrack !== void 0) fail(400, "INVALID_REFERENCE_TYPE", "Nested soundtracks are not supported.");
  }
  return { id, type, role };
}
__name(validateMediaHandle, "validateMediaHandle");
function countReferences(input) {
  const counts = { image: 0, video: 0, audio: 0, soundtracks: 0, total: 0 };
  const add = /* @__PURE__ */ __name((ref) => {
    if (!ref) return;
    counts[ref.type] += 1;
    counts.total += 1;
    if (ref.soundtrack) {
      counts.audio += 1;
      counts.soundtracks += 1;
      counts.total += 1;
    }
  }, "add");
  add(input.firstFrame);
  add(input.lastFrame);
  for (const ref of input.references || []) add(ref);
  return counts;
}
__name(countReferences, "countReferences");
function validateCanonicalInput(rawInput, rawCapabilities) {
  exactKeys(rawInput, new Set(CANONICAL_FIELDS), "input");
  const caps = capabilityDocument(rawCapabilities);
  const input = { ...rawInput };
  const mode = String(input.mode || "").trim();
  if (!MODE_FAMILY[mode] || caps.modes?.[mode]?.available !== true) {
    fail(400, "UNSUPPORTED_MODE", `Mode '${mode || "(missing)"}' is unavailable.`);
  }
  input.mode = mode;
  input.prompt = requireString(input.prompt, "prompt");
  const quality = String(input.quality || "standard").trim();
  if (!isObject(caps.quality?.[quality])) fail(400, "INVALID_QUALITY", `Quality '${quality}' is unavailable.`);
  input.quality = quality;
  const aspectRatio = String(input.aspectRatio || "16:9").trim();
  if (!Array.isArray(caps.aspectRatios) || !caps.aspectRatios.includes(aspectRatio)) {
    fail(400, "INVALID_ASPECT_RATIO", `Aspect ratio '${aspectRatio}' is unavailable.`);
  }
  input.aspectRatio = aspectRatio;
  const duration = Number(input.duration ?? 5);
  const minDuration = Number(caps.duration?.minSeconds);
  const maxDuration = Number(caps.duration?.maxSeconds);
  if (!Number.isFinite(duration) || duration <= 0 || Number.isFinite(minDuration) && duration < minDuration || Number.isFinite(maxDuration) && duration > maxDuration) {
    fail(400, "INVALID_DURATION", "duration is outside the worker capability range.");
  }
  input.duration = duration;
  if (input.seed !== void 0) {
    if (!Number.isSafeInteger(input.seed) || input.seed < 0) fail(400, "INVALID_SEED", "seed must be a non-negative safe integer.");
  }
  const family = MODE_FAMILY[mode];
  const generationMode = String(input.generationMode || "quality");
  if (!Array.isArray(caps.byFamily?.[family]) || !caps.byFamily[family].includes(generationMode)) {
    fail(400, "INVALID_STEPS", `Generation mode '${generationMode}' is unavailable for ${family}.`);
  }
  input.generationMode = generationMode;
  validateSemanticObject(input.camera, CAMERA, "input.camera");
  validateSemanticObject(input.style, STYLE, "input.style");
  if (input.privacy !== void 0) exactKeys(input.privacy, /* @__PURE__ */ new Set(["mode"]), "input.privacy");
  const privacyMode = String(input.privacy?.mode || "standard");
  if (!["standard", "confidential"].includes(privacyMode)) fail(400, "INVALID_REQUEST", "privacy.mode is invalid.");
  input.privacy = { mode: privacyMode };
  const roles = caps.references?.roles || {};
  if (input.firstFrame !== void 0) validateMediaHandle(input.firstFrame, "input.firstFrame", roles, "image");
  if (input.lastFrame !== void 0) validateMediaHandle(input.lastFrame, "input.lastFrame", roles, "image");
  if (input.references !== void 0 && !Array.isArray(input.references)) fail(400, "INVALID_REFERENCE_TYPE", "references must be an array.");
  for (const [index, ref] of (input.references || []).entries()) validateMediaHandle(ref, `input.references[${index}]`, roles);
  if (mode === "create" && (input.firstFrame || input.lastFrame || (input.references || []).length)) {
    fail(400, "INVALID_REFERENCE_COUNT", "Create does not accept frames or references.");
  }
  if (mode === "animate" && !input.firstFrame && !input.lastFrame) fail(400, "MISSING_FRAME", "Animate requires a start frame, an end frame, or both.");
  if (mode === "animate" && (input.references || []).length) fail(400, "INVALID_REFERENCE_COUNT", "Animate does not accept references.");
  if ((mode === "references" || mode === "remix") && (input.firstFrame || input.lastFrame)) {
    fail(400, "INVALID_REFERENCE_TYPE", `${mode} does not accept keyframes.`);
  }
  if (mode === "references" && !(input.references || []).length) fail(400, "INVALID_REFERENCE_COUNT", "References mode requires at least one reference.");
  if (mode === "remix" && !(input.references || []).some((ref) => ref.type === "video" && ref.role === "source")) {
    fail(400, "INVALID_REFERENCE_ROLE", "Remix requires a source video.");
  }
  if (input.referenceFidelity !== void 0) {
    if (!["references", "remix"].includes(mode) || !Array.isArray(caps.references?.fidelity) || !caps.references.fidelity.includes(input.referenceFidelity)) {
      fail(400, "INVALID_REQUEST", "referenceFidelity is unavailable for this mode.");
    }
  }
  const counts = countReferences(input);
  const limits = caps.references || {};
  if (counts.image > Number(limits.maxImages ?? 9) || counts.video > Number(limits.maxVideos ?? 3) || counts.audio - counts.soundtracks > Number(limits.maxAudio ?? 3) || counts.soundtracks > Number(limits.maxVideoSoundtracks ?? 3) || counts.total > Number(limits.maxTotal ?? 12)) {
    fail(400, "INVALID_REFERENCE_COUNT", "Reference media exceeds the product capability limits.", { counts });
  }
  const hasMedia = counts.total > 0;
  if (privacyMode === "confidential" && hasMedia) {
    fail(409, "CONFIDENTIAL_REFERENCES_UNAVAILABLE", "Confidential media references are disabled pending a non-persistent relay.");
  }
  if (privacyMode === "confidential" && mode !== "create") {
    fail(409, "CONFIDENTIAL_REFERENCES_UNAVAILABLE", "This Confidential mode is disabled pending a non-persistent reference relay.");
  }
  if (privacyMode === "standard" && input.encryption !== void 0) fail(400, "INVALID_REQUEST", "Standard generation must not include encryption.");
  if (privacyMode === "confidential" && !isObject(input.encryption)) fail(400, "CONFIDENTIAL_ENCRYPTION_REQUIRED", "Confidential generation requires encryption v2.");
  if (input.loras !== void 0) {
    if (!Array.isArray(input.loras)) fail(400, "INVALID_REQUEST", "loras must be an array.");
    const maxLoras = Number(caps.userLoras?.max ?? 2);
    if (input.loras.length > maxLoras) fail(400, "INVALID_LORA", `At most ${maxLoras} LoRAs may be attached.`);
    // URL-addressed LoRAs: the GPU downloads, verifies and purges them per job.
    // No LoRA bytes or object keys exist on the Worker or in storage at all.
    for (const [index, entry] of input.loras.entries()) {
      if (!isObject(entry)) fail(400, "INVALID_LORA", `loras[${index}] must be an object.`);
      const keys = Object.keys(entry).sort();
      if (keys.some((k) => !["sha256", "strength", "url"].includes(k))) {
        fail(400, "INVALID_LORA", `loras[${index}] has unknown fields.`);
      }
      const url = String(entry.url || "").trim();
      if (!/^https:\/\//.test(url) || url.length > 2048 || /\s/.test(url)) {
        fail(400, "INVALID_LORA", `loras[${index}].url must be an https download URL.`);
      }
      if (entry.sha256 !== void 0 && !/^[0-9a-f]{64}$/.test(String(entry.sha256))) {
        fail(400, "INVALID_LORA", `loras[${index}].sha256 must be a 64-character hex digest.`);
      }
      const strength = entry.strength === void 0 ? 1 : Number(entry.strength);
      if (!Number.isFinite(strength) || strength < 0 || strength > 1) fail(400, "INVALID_LORA", `loras[${index}].strength must be between 0 and 1.`);
      entry.url = url;
      entry.sha256 = entry.sha256 === void 0 ? void 0 : String(entry.sha256).toLowerCase();
      entry.strength = strength;
    }
    if (privacyMode === "confidential" && input.loras.length) {
      fail(409, "CONFIDENTIAL_LORAS_UNAVAILABLE", "Confidential generation cannot attach user LoRAs pending a non-persistent relay.");
    }
  }
  delete input.progress;
  delete input.output;
  return { input, family, counts, privacyMode };
}
__name(validateCanonicalInput, "validateCanonicalInput");
function safeSegment(value, name, max = 128) {
  const text = String(value || "").trim();
  if (!text || text.length > max || !/^[A-Za-z0-9._-]+$/.test(text)) fail(400, "INVALID_ASSET", `${name} is invalid.`);
  return text;
}
__name(safeSegment, "safeSegment");
function validateObjectKey(key, descriptor = null) {
  const text = String(key || "");
  const isReferenceKey = /^references\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\.[A-Za-z0-9]+$/.test(text);
  // User LoRA library objects: owner-scoped, digest-pinned, safetensors only.
  const isLoraKey = /^loras\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\.safetensors$/.test(text);
  if ((!isReferenceKey && !isLoraKey) || text.includes("..")) {
    fail(400, "INVALID_ASSET", "Reference object key is invalid.");
  }
  if (descriptor && isReferenceKey) {
    const expectedPrefix = `references/${safeSegment(descriptor.accountScope, "accountScope")}/${safeSegment(descriptor.uploadSessionId, "uploadSessionId")}/${safeSegment(descriptor.assetId, "assetId")}`;
    if (!text.startsWith(`${expectedPrefix}.`)) fail(403, "ASSET_SCOPE_MISMATCH", "Reference object is outside its job scope.");
  }
  return text;
}
__name(validateObjectKey, "validateObjectKey");
function createReferenceObject({ accountScope, uploadSessionId, type, contentType, declaredSize }) {
  const spec = MEDIA_TYPES[type];
  const extension = spec?.contentTypes?.[contentType];
  if (!spec || !extension) fail(415, "INVALID_REFERENCE_TYPE", "Unsupported reference content type.");
  const size = Number(declaredSize);
  if (!Number.isInteger(size) || size <= 0 || size > spec.maxBytes) fail(413, "REFERENCE_TOO_LARGE", `${type} exceeds its upload limit.`);
  const scope = safeSegment(accountScope, "accountScope");
  const session = safeSegment(uploadSessionId, "uploadSessionId");
  const assetId = crypto.randomUUID();
  return {
    assetId,
    objectKey: `references/${scope}/${session}/${assetId}${extension}`,
    type,
    contentType,
    declaredSize: size
  };
}
__name(createReferenceObject, "createReferenceObject");
function descriptorMap(descriptors) {
  const map = /* @__PURE__ */ new Map();
  for (const descriptor of descriptors || []) {
    if (!isObject(descriptor)) fail(400, "INVALID_ASSET", "Asset descriptor is invalid.");
    exactKeys(descriptor, /* @__PURE__ */ new Set([
      "assetId",
      "objectKey",
      "type",
      "contentType",
      "size",
      "status",
      "accountScope",
      "uploadSessionId",
      "expiresAt"
    ]), "asset");
    const assetId = safeSegment(descriptor.assetId, "assetId");
    if (map.has(assetId)) fail(400, "INVALID_ASSET", "Duplicate asset descriptor.");
    if (descriptor.status !== "uploaded") fail(409, "ASSET_NOT_READY", `Asset '${assetId}' is not uploaded.`);
    if (!MEDIA_TYPES[descriptor.type]) fail(400, "INVALID_ASSET", "Asset type is invalid.");
    validateObjectKey(descriptor.objectKey, descriptor);
    map.set(assetId, descriptor);
  }
  return map;
}
__name(descriptorMap, "descriptorMap");
async function resolveReferenceAssets(input, descriptors, signGet) {
  const assets = descriptorMap(descriptors);
  const usedKeys = [];
  const resolveOne = /* @__PURE__ */ __name(async (ref, path) => {
    if (!ref) return void 0;
    const descriptor = assets.get(ref.id);
    if (!descriptor) fail(403, "ASSET_OWNERSHIP_INVALID", `${path} is not an owned uploaded asset.`);
    if (descriptor.type !== ref.type) fail(400, "INVALID_REFERENCE_TYPE", `${path} type does not match the uploaded asset.`);
    const objectKey = validateObjectKey(descriptor.objectKey, descriptor);
    const url = await signGet(objectKey);
    const out = { type: ref.type, role: ref.role || "general", id: ref.id, url };
    usedKeys.push(objectKey);
    if (ref.soundtrack) out.soundtrack = await resolveOne(ref.soundtrack, `${path}.soundtrack`);
    return out;
  }, "resolveOne");
  const resolved = { ...input };
  if (input.firstFrame) resolved.firstFrame = await resolveOne(input.firstFrame, "firstFrame");
  if (input.lastFrame) resolved.lastFrame = await resolveOne(input.lastFrame, "lastFrame");
  if (Array.isArray(input.references)) {
    resolved.references = [];
    for (const [index, ref] of input.references.entries()) resolved.references.push(await resolveOne(ref, `references[${index}]`));
  }
  return { input: resolved, objectKeys: [...new Set(usedKeys)] };
}
__name(resolveReferenceAssets, "resolveReferenceAssets");
function encodeRfc3986(value) {
  return encodeURIComponent(value).replace(/[!'()*]/g, (character) => `%${character.charCodeAt(0).toString(16).toUpperCase()}`);
}
__name(encodeRfc3986, "encodeRfc3986");
function hex(bytes) {
  return Array.from(new Uint8Array(bytes)).map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
__name(hex, "hex");
async function sha256(value) {
  const bytes = typeof value === "string" ? new TextEncoder().encode(value) : value;
  return crypto.subtle.digest("SHA-256", bytes);
}
__name(sha256, "sha256");
async function hmac(key, value) {
  const rawKey = typeof key === "string" ? new TextEncoder().encode(key) : key;
  const imported = await crypto.subtle.importKey("raw", rawKey, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return crypto.subtle.sign("HMAC", imported, new TextEncoder().encode(value));
}
__name(hmac, "hmac");
async function presignR2({ accountId, accessKeyId, secretAccessKey, bucket, objectKey, method, expiresSeconds, contentType, contentLength, now = /* @__PURE__ */ new Date() }) {
  for (const [name, value] of Object.entries({ accountId, accessKeyId, secretAccessKey, bucket })) {
    if (!String(value || "").trim()) fail(503, "R2_SIGNING_NOT_CONFIGURED", `R2 ${name} is not configured.`);
  }
  const expires = Math.max(60, Math.min(604800, Math.trunc(Number(expiresSeconds) || 1200)));
  const key = validateObjectKey(objectKey);
  const amzDate = now.toISOString().replace(/[:-]|\.\d{3}/g, "");
  const dateStamp = amzDate.slice(0, 8);
  const host = `${accountId}.r2.cloudflarestorage.com`;
  const canonicalUri = `/${encodeRfc3986(bucket)}/${key.split("/").map(encodeRfc3986).join("/")}`;
  const credentialScope = `${dateStamp}/auto/s3/aws4_request`;
  const headers = { host };
  if (contentType) headers["content-type"] = String(contentType).trim().toLowerCase();
  if (Number.isInteger(contentLength) && contentLength > 0) headers["content-length"] = String(contentLength);
  const signedHeaders = Object.keys(headers).sort().join(";");
  const canonicalHeaders = Object.keys(headers).sort().map((name) => `${name}:${headers[name]}
`).join("");
  const query = {
    "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
    "X-Amz-Credential": `${accessKeyId}/${credentialScope}`,
    "X-Amz-Date": amzDate,
    "X-Amz-Expires": String(expires),
    "X-Amz-SignedHeaders": signedHeaders
  };
  const canonicalQuery = Object.entries(query).sort(([a], [b]) => a.localeCompare(b)).map(([name, value]) => `${encodeRfc3986(name)}=${encodeRfc3986(value)}`).join("&");
  const canonicalRequest = [String(method || "GET").toUpperCase(), canonicalUri, canonicalQuery, canonicalHeaders, signedHeaders, "UNSIGNED-PAYLOAD"].join("\n");
  const stringToSign = ["AWS4-HMAC-SHA256", amzDate, credentialScope, hex(await sha256(canonicalRequest))].join("\n");
  const dateKey = await hmac(`AWS4${secretAccessKey}`, dateStamp);
  const regionKey = await hmac(dateKey, "auto");
  const serviceKey = await hmac(regionKey, "s3");
  const signingKey = await hmac(serviceKey, "aws4_request");
  const signature = hex(await hmac(signingKey, stringToSign));
  return {
    url: `https://${host}${canonicalUri}?${canonicalQuery}&X-Amz-Signature=${signature}`,
    expiresAt: new Date(now.getTime() + expires * 1e3).toISOString(),
    signedHeaders
  };
}
__name(presignR2, "presignR2");

// worker.js
var __defProp2 = Object.defineProperty;
var __name2 = /* @__PURE__ */ __name((target, value) => __defProp2(target, "name", { value, configurable: true }), "__name");
var DEFAULT_BACKEND = "h3";
var BACKENDS = Object.freeze({
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
  }),
  "h3-cached-canary": Object.freeze({
    endpointIdEnv: "RUNPOD_CACHED_CANARY_ENDPOINT_ID",
    apiKeyEnv: "RUNPOD_CACHED_CANARY_API_KEY",
    // This endpoint is in the same RunPod account. Prefer an isolated key when
    // present, while retaining the already-tested account-key fallback.
    fallbackApiKeyEnv: "RUNPOD_API_KEY"
  })
});
var BACKEND_ALIASES = Object.freeze({
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
  "minimax-h3-blackwell": "h3-blackwell",
  "h3-cached-canary": "h3-cached-canary"
});
var BINARY_FIELD_NAMES = /* @__PURE__ */ new Set([
  "data",
  "base64",
  "image_base64",
  "video_base64",
  "audio_base64",
  "file_data"
]);
var FPS = 24;
var CANVAS_MULTIPLE = 32;
var BASE_SHORT_EDGE = 768;
var MAX_PIXELS = 768 * 1344;
var FRAME_GRID_MODULUS = 17;
var FRAME_GRID_REMAINDER = 5;
var MIN_FRAMES = 5;
var MAX_FRAMES = 3600;
var QUALITY_PRESETS = Object.freeze({
  fast: Object.freeze({ shortEdge: 576, steps: 14 }),
  standard: Object.freeze({ shortEdge: 576, steps: 20 }),
  hd: Object.freeze({ shortEdge: BASE_SHORT_EDGE, steps: 20 })
});
var DEFAULT_QUALITY = "standard";
var ASPECT_RATIOS = Object.freeze({
  "16:9": 16 / 9,
  "9:16": 9 / 16,
  "1:1": 1,
  "4:3": 4 / 3,
  "3:4": 3 / 4
});
var DEFAULT_ASPECT_RATIO = "16:9";
var DEFAULT_DURATION_SECONDS = 5;
var MODES = Object.freeze({
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
    reason: "Ref2VA needs minimax_h3_ref2va_pruned_int8_convrot.safetensors (20.97 GB), which is deliberately excluded from the image. See README 'Ref2VA status'."
  }),
  regenerate_2k: Object.freeze({
    implemented: false,
    firstFrame: false,
    lastFrame: false,
    reason: "No second-stage upscale model is installed and none has been measured. See README '2K status'."
  })
});
var DEFAULT_MODE = "text_to_video";
var PRIVACY_MODES = Object.freeze({
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
    description: "The artefact is encrypted inside the inference environment with a key the caller derived. Persistent storage receives ciphertext only; the browser decrypts for playback."
  }),
  private: Object.freeze({
    implemented: false,
    encrypts: false,
    restrictsPromptLogging: true,
    description: "Unencrypted storage with restricted logging and a short default retention.",
    reason: "Needs a retention enforcer to be meaningful. Today's retention is a prefix-wide R2 lifecycle rule, which cannot express a per-job lifetime, so 'private' would promise a shorter life than the platform can deliver."
  }),
  ephemeral: Object.freeze({
    implemented: false,
    encrypts: true,
    restrictsPromptLogging: true,
    description: "Confidential, plus the server-side copy is deleted as soon as delivery succeeds.",
    reason: "Needs a reliable definition of 'delivery succeeded'. A ranged or aborted GET is not a delivery, and deleting on the first byte read would destroy the artefact mid-download."
  })
});
var DEFAULT_PRIVACY_MODE = "standard";
var CONTAINER_MAGIC = "CGEN";
var CONTAINER_V1_SYMMETRIC = 1;
var CONTAINER_V2_HYBRID = 2;
var SUPPORTED_CONTAINER_VERSIONS = /* @__PURE__ */ new Set([CONTAINER_V1_SYMMETRIC, CONTAINER_V2_HYBRID]);
var CONTAINER_VERSION = CONTAINER_V2_HYBRID;
var CONTAINER_SUITES = Object.freeze({ 1: "AES-256-GCM" });
var CONTAINER_PREAMBLE_BYTES = 8;
var CONTAINER_MAX_HEADER_BYTES = 8192;
var CONTAINER_NONCE_BYTES = 12;
var ENCRYPTED_CONTENT_TYPE = "application/octet-stream";
var SUPPORTED_ENCRYPTION_ALGORITHMS = Object.freeze(["AES-256-GCM"]);
var SUPPORTED_KEY_WRAP_ALGORITHMS = Object.freeze(["RSA-OAEP-256"]);
var MIN_RSA_MODULUS_BITS = 3072;
var MIN_SPKI_BYTES = 380;
var MAX_SPKI_BYTES = 1200;
var FORBIDDEN_REQUEST_FIELDS = /* @__PURE__ */ new Set([
  "passphrase",
  "password",
  "privatekey",
  "privateencryptionkey",
  "encryptedprivatekey",
  "keyencryptionkey",
  "kek",
  "fileencryptionkey",
  "fek",
  "decryptionkey",
  "derivedkey",
  "aeskey",
  "symmetrickey",
  "secretkey"
]);
var MIN_RETENTION_SECONDS = 60;
var MAX_RETENTION_SECONDS = 90 * 24 * 3600;
var MODEL_FILES = Object.freeze({
  unet: "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
  clip: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
  videoVae: "minimax_h3_video_vae_fp16.safetensors",
  audioVae: "minimax_h3_audio_vae_fp32.safetensors"
});
var HttpError = class extends Error {
  static {
    __name(this, "HttpError");
  }
  static {
    __name2(this, "HttpError");
  }
  constructor(status, message, details) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.details = details;
  }
};
var worker_default = {
  async fetch(request, env) {
    const url = new URL(request.url);
    const headers = corsHeaders();
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers });
    }
    try {
      const segments = splitPath(url.pathname);
      if (request.method === "POST" && segments.length === 2 && segments[0] === "canonical" && segments[1] === "generate") {
        return await generateCanonicalVideo(request, env, headers, url);
      }
      if (request.method === "POST" && segments.length === 3 && segments[0] === "canonical" && segments[1] === "assets" && segments[2] === "authorize") {
        return await authorizeCanonicalAsset(request, env, headers);
      }
      if (request.method === "POST" && segments.length === 3 && segments[0] === "canonical" && segments[1] === "assets" && segments[2] === "confirm") {
        return await confirmCanonicalAsset(request, env, headers);
      }
      if (request.method === "POST" && segments.length === 3 && segments[0] === "canonical" && segments[1] === "assets" && segments[2] === "delete") {
        return await deleteCanonicalAsset(request, env, headers);
      }
      if (request.method === "POST" && segments.length === 3 && segments[0] === "canonical" && segments[1] === "loras" && segments[2] === "authorize") {
        return await authorizeCanonicalLora(request, env, headers);
      }
      if (request.method === "POST" && segments.length === 3 && segments[0] === "canonical" && segments[1] === "loras" && segments[2] === "confirm") {
        return await confirmCanonicalLora(request, env, headers);
      }
      if (request.method === "POST" && segments.length === 3 && segments[0] === "canonical" && segments[1] === "loras" && segments[2] === "delete") {
        return await deleteCanonicalLora(request, env, headers);
      }
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
      if (request.method === "GET" && isExactRoute(segments, "capabilities")) {
        const backend = canonicalBackendFromUrl(url);
        return await serveCanonicalCapabilities(env, headers, url.searchParams.get("refresh") === "1", backend);
      }
      if (request.method === "GET" && segments.length === 2 && segments[0] === "capabilities" && segments[1] === "legacy") {
        return json(describeCapabilities(), 200, headers);
      }
      if (request.method === "POST" && segments.length === 3 && segments[0] === "jobs" && segments[2] === "assets") {
        return await uploadAsset(request, env, headers, segments[1], url);
      }
      if (request.method === "GET" && segments.length === 3 && segments[0] === "jobs" && (segments[2] === "video" || segments[2] === "artifact")) {
        return await streamVideo(request, env, segments[1]);
      }
      if (request.method === "DELETE" && segments.length === 3 && segments[0] === "jobs" && (segments[2] === "video" || segments[2] === "artifact")) {
        return await deleteGeneration(env, headers, segments[1], "output");
      }
      if (request.method === "DELETE" && segments.length === 2 && segments[0] === "jobs") {
        return await deleteGeneration(env, headers, segments[1], "generation");
      }
      if (request.method === "GET" && segments.length === 3 && segments[0] === "ws" && segments[1] === "jobs") {
        return await openJobSocket(request, env, segments[2]);
      }
      if (segments[0] === "internal" && segments[1] === "jobs" && segments.length >= 4) {
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
        const capabilityState = await readJobState(env, CAPABILITY_CACHE_JOB_ID);
        return json(
          {
            ok: true,
            service: "minimax-h3-backend",
            version: env.CF_VERSION_METADATA?.id || null,
            defaultBackend: DEFAULT_BACKEND,
            backends: {
              h3: { configured: isBackendConfigured("h3", env) },
              "h3-blackwell": { configured: isBackendConfigured("h3-blackwell", env) },
              "h3-cached-canary": { configured: isBackendConfigured("h3-cached-canary", env) }
            },
            features: {
              statusProgress: true,
              canonicalMultimodal: true,
              directR2Presigning: Boolean(env.R2_ACCOUNT_ID && env.R2_ACCESS_KEY_ID && env.R2_SECRET_ACCESS_KEY),
              confidentialMultimodalReferences: false
            },
            canonical: {
              backend: CANONICAL_BACKEND,
              expectedBuild: EXPECTED_H3_BUILD,
              cachedBuild: capabilityState?.capabilities ? buildIdentity(capabilityState.capabilities) : capabilityState?.lastRejectedBuild || null,
              capabilityFetchedAt: capabilityState?.fetchedAt || null,
              capabilityRejectedAt: capabilityState?.lastRejectedAt || null,
              r2BucketConfigured: Boolean(env.H3_OUTPUTS),
              r2SigningConfigured: Boolean(env.R2_ACCOUNT_ID && env.R2_ACCESS_KEY_ID && env.R2_SECRET_ACCESS_KEY)
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
          backends: ["h3", "h3-blackwell", "h3-cached-canary"]
        },
        404,
        headers
      );
    } catch (error) {
      const isSafeError = error instanceof HttpError || error instanceof CanonicalError;
      const status = isSafeError ? error.status : 500;
      const response = { error: isSafeError ? error.message : "Internal error" };
      if (isSafeError && error.code) response.code = error.code;
      if (isSafeError && error.details !== void 0) {
        response.details = error.details;
      }
      return json(response, status, headers);
    }
  },
  // Every 2 minutes: capture terminal status for in-flight jobs nobody is
  // polling, so RunPod's ~30-minute record expiry can never lose a result.
  async scheduled(event, env) {
    try {
      await sweepActiveJobs(env);
    } catch (error) {
      console.warn(`active job sweep failed: ${errorCodeForLog(error)}`);
    }
  }
};
function adaptCanvas(ratio, shortEdge) {
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
__name(adaptCanvas, "adaptCanvas");
__name2(adaptCanvas, "adaptCanvas");
function alignFrameCount(n) {
  let frames = Math.max(MIN_FRAMES, Math.trunc(n));
  while (frames % FRAME_GRID_MODULUS !== FRAME_GRID_REMAINDER) {
    frames += 1;
  }
  return frames;
}
__name(alignFrameCount, "alignFrameCount");
__name2(alignFrameCount, "alignFrameCount");
function isLegalFrameCount(n) {
  return Number.isInteger(n) && n >= MIN_FRAMES && n <= MAX_FRAMES && n % FRAME_GRID_MODULUS === FRAME_GRID_REMAINDER;
}
__name(isLegalFrameCount, "isLegalFrameCount");
__name2(isLegalFrameCount, "isLegalFrameCount");
function durationToFrames(seconds) {
  return alignFrameCount(Math.round(seconds * FPS));
}
__name(durationToFrames, "durationToFrames");
__name2(durationToFrames, "durationToFrames");
function framesToDuration(frames) {
  return frames / FPS;
}
__name(framesToDuration, "framesToDuration");
__name2(framesToDuration, "framesToDuration");
function buildPrompt(input) {
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
  return `${base}

${lines.join("\n")}`;
}
__name(buildPrompt, "buildPrompt");
__name2(buildPrompt, "buildPrompt");
function trimTrailingPeriod(text) {
  return text.endsWith(".") ? text.slice(0, -1) : text;
}
__name(trimTrailingPeriod, "trimTrailingPeriod");
__name2(trimTrailingPeriod, "trimTrailingPeriod");
function resolveMode(body) {
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
  if (isNonEmptyArray(body.reference_images) || isNonEmptyArray(body.reference_videos) || isNonEmptyArray(body.reference_audio)) {
    return "reference";
  }
  return DEFAULT_MODE;
}
__name(resolveMode, "resolveMode");
__name2(resolveMode, "resolveMode");
function isAssetReference(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
__name(isAssetReference, "isAssetReference");
__name2(isAssetReference, "isAssetReference");
function isNonEmptyArray(value) {
  return Array.isArray(value) && value.length > 0;
}
__name(isNonEmptyArray, "isNonEmptyArray");
__name2(isNonEmptyArray, "isNonEmptyArray");
function normalizeRequest(body) {
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
  const qualityGiven = hasNonEmptyValue(body.quality);
  const quality = qualityGiven ? String(body.quality).trim().toLowerCase() : DEFAULT_QUALITY;
  if (!Object.prototype.hasOwnProperty.call(QUALITY_PRESETS, quality)) {
    throw new HttpError(
      400,
      `Unknown quality '${quality}'. Supported: ${Object.keys(QUALITY_PRESETS).join(", ")}`
    );
  }
  const preset = QUALITY_PRESETS[quality];
  const widthGiven = hasNonEmptyValue(body.width);
  const heightGiven = hasNonEmptyValue(body.height);
  const aspectGiven = hasNonEmptyValue(body.aspect_ratio);
  if (widthGiven !== heightGiven) {
    throw new HttpError(400, "width and height must be supplied together");
  }
  const aspectRatio = aspectGiven ? String(body.aspect_ratio).trim() : DEFAULT_ASPECT_RATIO;
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
      if (Math.abs(actual - requested) > 0.05) {
        throw new HttpError(
          400,
          `width/height (${width}x${height}, ratio ${actual.toFixed(3)}) contradicts aspect_ratio '${aspectRatio}' (${requested.toFixed(3)}). Supply one or the other.`
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
        `frames must satisfy frames % 17 == 5 (H3's temporal grid), between ${MIN_FRAMES} and ${MAX_FRAMES}. Nearest legal values to ${frames}: ${nearestLegalFrames(frames).join(" or ")}.`
      );
    }
    if (durationGiven) {
      const fromDuration = durationToFrames(toNumber(body.duration, 0));
      if (fromDuration !== frames) {
        throw new HttpError(
          400,
          `frames (${frames}) contradicts duration (${body.duration}s -> ${fromDuration} frames). Supply one or the other.`
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
  const stepsGiven = hasNonEmptyValue(body.steps);
  const steps = stepsGiven ? toInteger(body.steps, preset.steps) : preset.steps;
  if (steps <= 0 || steps > 100) {
    throw new HttpError(400, "steps must be between 1 and 100");
  }
  const seed = hasNonEmptyValue(body.seed) ? toInteger(body.seed, 0) : Math.floor(Math.random() * 2147483647);
  if (seed < 0) {
    throw new HttpError(400, "seed must be non-negative");
  }
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
    audio: body.audio === void 0 ? true : Boolean(body.audio),
    firstFrame,
    lastFrame,
    resolvedFrom: { canvas: canvasSource, frames: framesSource, steps: stepsGiven ? "explicit" : "preset" }
  };
}
__name(normalizeRequest, "normalizeRequest");
__name2(normalizeRequest, "normalizeRequest");
async function normalizePrivacy(body) {
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
  const hasBlock = block !== void 0 && block !== null;
  if (hasBlock && (typeof block !== "object" || Array.isArray(block))) {
    throw new HttpError(400, "encryption must be an object");
  }
  if (!spec.encrypts) {
    if (hasBlock) {
      throw new HttpError(
        400,
        `privacyMode '${mode}' does not encrypt, but an encryption block was supplied. Use privacyMode 'confidential', or remove the block.`
      );
    }
    return { mode, spec, encryption: null, expiresAt: resolveExpiry(body) };
  }
  if (!hasBlock) {
    throw new HttpError(
      400,
      `privacyMode 'confidential' requires an encryption block carrying your public key: { "version": 2, "publicKey": "<base64url SPKI>" }. See docs/confidential-generation.md.`
    );
  }
  const algorithm = hasNonEmptyValue(block.algorithm) ? String(block.algorithm).trim() : SUPPORTED_ENCRYPTION_ALGORITHMS[0];
  if (!SUPPORTED_ENCRYPTION_ALGORITHMS.includes(algorithm)) {
    throw new HttpError(
      400,
      `Unsupported encryption algorithm '${algorithm}'. Supported: ` + SUPPORTED_ENCRYPTION_ALGORITHMS.join(", ")
    );
  }
  const version = hasNonEmptyValue(block.version) ? toInteger(block.version, 0) : CONTAINER_V2_HYBRID;
  if (version === CONTAINER_V1_SYMMETRIC || hasNonEmptyValue(block.key)) {
    throw new HttpError(
      400,
      "Confidential generation v1 (a symmetric key in the request) is no longer accepted: that key would travel through the job queue and could later decrypt the stored video. Send encryption.version 2 with a publicKey instead. Existing v1 artefacts remain decryptable.",
      { cryptoVersion: CONTAINER_V2_HYBRID, migration: "docs/confidential-generation.md" }
    );
  }
  if (version !== CONTAINER_V2_HYBRID) {
    throw new HttpError(
      400,
      `Unsupported encryption.version ${version}. This deployment creates version ${CONTAINER_V2_HYBRID}.`
    );
  }
  const keyWrapAlgorithm = hasNonEmptyValue(firstPresent(block.keyWrapAlgorithm, block.key_wrap_algorithm)) ? String(firstPresent(block.keyWrapAlgorithm, block.key_wrap_algorithm)).trim() : SUPPORTED_KEY_WRAP_ALGORITHMS[0];
  if (!SUPPORTED_KEY_WRAP_ALGORITHMS.includes(keyWrapAlgorithm)) {
    throw new HttpError(
      400,
      `Unsupported keyWrapAlgorithm '${keyWrapAlgorithm}'. Supported: ` + SUPPORTED_KEY_WRAP_ALGORITHMS.join(", ")
    );
  }
  const publicKeyAlgorithm = hasNonEmptyValue(firstPresent(block.publicKeyAlgorithm, block.public_key_algorithm)) ? String(firstPresent(block.publicKeyAlgorithm, block.public_key_algorithm)).trim() : keyWrapAlgorithm;
  if (publicKeyAlgorithm !== keyWrapAlgorithm) {
    throw new HttpError(
      400,
      `encryption.publicKeyAlgorithm '${publicKeyAlgorithm}' does not match keyWrapAlgorithm '${keyWrapAlgorithm}'.`
    );
  }
  const publicKey = validatePublicKey(firstPresent(block.publicKey, block.public_key));
  const keyId = await deriveKeyId(publicKey.spki);
  const claimedKeyId = firstPresent(block.keyId, block.key_id);
  if (hasNonEmptyValue(claimedKeyId) && String(claimedKeyId).trim() !== keyId) {
    throw new HttpError(
      400,
      `encryption.keyId does not match the supplied public key (expected '${keyId}').`
    );
  }
  return {
    mode,
    spec,
    encryption: {
      version: CONTAINER_V2_HYBRID,
      algorithm,
      keyWrapAlgorithm,
      publicKeyAlgorithm,
      publicKey: publicKey.encoded,
      publicKeyBits: publicKey.modulusBits,
      keyId
    },
    expiresAt: resolveExpiry(body)
  };
}
__name(normalizePrivacy, "normalizePrivacy");
__name2(normalizePrivacy, "normalizePrivacy");
function rejectSecretFields(value, path = "", depth = 0) {
  if (depth > 12 || value === null || typeof value !== "object") return;
  if (Array.isArray(value)) {
    for (const [index, item] of value.entries()) {
      rejectSecretFields(item, `${path}[${index}]`, depth + 1);
    }
    return;
  }
  for (const [name, child] of Object.entries(value)) {
    const normalized = String(name).toLowerCase().replace(/[_-]/g, "");
    if (FORBIDDEN_REQUEST_FIELDS.has(normalized)) {
      const where = path ? `${path}.${name}` : name;
      throw new HttpError(
        400,
        `Request field '${where}' must never be sent to this API. Confidential generation is public-key based: the inference worker receives an encryption-only public key, and any value that could decrypt the result stays in your browser.`,
        { field: where }
      );
    }
    rejectSecretFields(child, path ? `${path}.${name}` : name, depth + 1);
  }
}
__name(rejectSecretFields, "rejectSecretFields");
__name2(rejectSecretFields, "rejectSecretFields");
function validatePublicKey(value) {
  if (!hasNonEmptyValue(value)) {
    throw new HttpError(
      400,
      "encryption.publicKey is required in confidential mode (base64url SPKI)"
    );
  }
  if (typeof value !== "string") {
    throw new HttpError(400, "encryption.publicKey must be a base64url-encoded string");
  }
  const text = value.trim();
  if (!/^[A-Za-z0-9_\-+/]+={0,2}$/.test(text)) {
    throw new HttpError(400, "encryption.publicKey is not valid base64url");
  }
  let spki;
  try {
    spki = b64urlDecodeToBytes(text);
  } catch {
    throw new HttpError(400, "encryption.publicKey is not valid base64url");
  }
  if (spki[0] !== 48) {
    throw new HttpError(400, "encryption.publicKey is not a DER SubjectPublicKeyInfo");
  }
  if (spki.length < MIN_SPKI_BYTES || spki.length > MAX_SPKI_BYTES) {
    throw new HttpError(
      400,
      `encryption.publicKey is ${spki.length} bytes, outside the ${MIN_SPKI_BYTES}-${MAX_SPKI_BYTES} range for an RSA key of ${MIN_RSA_MODULUS_BITS} bits or more`
    );
  }
  return { encoded: text, spki, modulusBits: null };
}
__name(validatePublicKey, "validatePublicKey");
__name2(validatePublicKey, "validatePublicKey");
async function deriveKeyId(spki) {
  const digest = await crypto.subtle.digest("SHA-256", spki);
  return b64urlEncode(new Uint8Array(digest).slice(0, 16));
}
__name(deriveKeyId, "deriveKeyId");
__name2(deriveKeyId, "deriveKeyId");
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
    return new Date(Date.now() + seconds * 1e3).toISOString();
  }
  const at = new Date(String(body.expiresAt));
  if (Number.isNaN(at.getTime())) {
    throw new HttpError(400, "expiresAt must be an ISO-8601 timestamp");
  }
  const delta = (at.getTime() - Date.now()) / 1e3;
  if (delta < MIN_RETENTION_SECONDS || delta > MAX_RETENTION_SECONDS) {
    throw new HttpError(
      400,
      `expiresAt must be between ${MIN_RETENTION_SECONDS} seconds and ${MAX_RETENTION_SECONDS} seconds from now`
    );
  }
  return at.toISOString();
}
__name(resolveExpiry, "resolveExpiry");
__name2(resolveExpiry, "resolveExpiry");
function firstPresent(...values) {
  for (const value of values) {
    if (hasNonEmptyValue(value)) return value;
  }
  return void 0;
}
__name(firstPresent, "firstPresent");
__name2(firstPresent, "firstPresent");
function nearestLegalFrames(n) {
  const up = alignFrameCount(n);
  let down = up;
  while (down - FRAME_GRID_MODULUS >= MIN_FRAMES) {
    down -= FRAME_GRID_MODULUS;
    if (down <= n) break;
  }
  return down === up ? [up] : [down, up];
}
__name(nearestLegalFrames, "nearestLegalFrames");
__name2(nearestLegalFrames, "nearestLegalFrames");
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
__name(validateCanvas, "validateCanvas");
__name2(validateCanvas, "validateCanvas");
function normalizeAsset(value, fieldName) {
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
__name(normalizeAsset, "normalizeAsset");
__name2(normalizeAsset, "normalizeAsset");
function validateR2Key(key, fieldName = "r2_key") {
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
__name(validateR2Key, "validateR2Key");
__name2(validateR2Key, "validateR2Key");
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
      inputs: { model: ["unet", 0], scheduler: "simple", steps: 0, denoise: 1 }
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
__name(baseFl2vaTemplate, "baseFl2vaTemplate");
__name2(baseFl2vaTemplate, "baseFl2vaTemplate");
function loadWorkflowTemplate(mode) {
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
__name(loadWorkflowTemplate, "loadWorkflowTemplate");
__name2(loadWorkflowTemplate, "loadWorkflowTemplate");
function applySettings(template, settings) {
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
__name(applySettings, "applySettings");
__name2(applySettings, "applySettings");
function buildWorkflowForSettings(settings) {
  return applySettings(loadWorkflowTemplate(settings.mode), settings);
}
__name(buildWorkflowForSettings, "buildWorkflowForSettings");
__name2(buildWorkflowForSettings, "buildWorkflowForSettings");
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
    statusProgress: {
      available: true,
      fields: ["phase", "percent", "step", "steps", "updatedAt"]
    },
    qualities,
    aspectRatios: Object.keys(ASPECT_RATIOS),
    modes: Object.fromEntries(
      Object.entries(MODES).map(([name, spec]) => [
        name,
        spec.implemented ? { available: true } : { available: false, reason: spec.reason }
      ])
    ),
    /*
     * Privacy is discoverable for the same reason generation modes are: a client should be
     * able to find out that `ephemeral` is not available yet, and why, without having to
     * send a request and read a 501.
     */
    defaultPrivacyMode: DEFAULT_PRIVACY_MODE,
    privacyModes: Object.fromEntries(
      Object.entries(PRIVACY_MODES).map(([name, spec]) => [
        name,
        spec.implemented ? { available: true, encrypts: spec.encrypts, description: spec.description } : { available: false, encrypts: spec.encrypts, description: spec.description, reason: spec.reason }
      ])
    ),
    encryption: {
      /*
       * Hybrid: the caller supplies an encryption-only public key, and the inference
       * worker generates a fresh symmetric file key per artefact and wraps it to that key.
       * Nothing capable of decrypting the result is ever sent.
       */
      cryptoVersion: CONTAINER_V2_HYBRID,
      readableCryptoVersions: [CONTAINER_V1_SYMMETRIC, CONTAINER_V2_HYBRID],
      algorithms: [...SUPPORTED_ENCRYPTION_ALGORITHMS],
      keyWrapAlgorithms: [...SUPPORTED_KEY_WRAP_ALGORITHMS],
      publicKeyFormat: "spki-der-base64url",
      minPublicKeyBits: MIN_RSA_MODULUS_BITS,
      /*
       * The passphrase KDF is deliberately absent. It protects the caller's private key,
       * which never reaches this API in any form, so the platform neither runs it nor
       * records its parameters. That is a client-side concern by design.
       */
      retentionSeconds: { min: MIN_RETENTION_SECONDS, max: MAX_RETENTION_SECONDS }
    }
  };
}
__name(describeCapabilities, "describeCapabilities");
__name2(describeCapabilities, "describeCapabilities");
var CANONICAL_BACKEND = "h3-blackwell";
var CACHED_CANARY_BACKEND = "h3-cached-canary";
var CAPABILITY_CACHE_JOB_ID = "__canonical_capabilities__";
// 30 min: a cache miss blocks on a live RunPod probe (~11-21s), which surfaced
// as multi-second /api/models loads on the dashboard. Build changes are rare,
// deliberate events — refresh with ?refresh=1 after any cutover.
var CAPABILITY_CACHE_TTL_MS = 30 * 60 * 1e3;
var CAPABILITY_REFRESH_TIMEOUT_MS = 12 * 1e3;
var CAPABILITY_PENDING_MAX_AGE_MS = 30 * 60 * 1e3;
var CANONICAL_OUTER_FIELDS = /* @__PURE__ */ new Set(["input", "assets"]);
function canonicalBackendFromUrl(url) {
  const backendValue = url.searchParams.get("backend");
  const modelValue = url.searchParams.get("model");
  const hasBackend = hasNonEmptyValue(backendValue);
  const hasModel = hasNonEmptyValue(modelValue);
  if (!hasBackend && !hasModel) return CANONICAL_BACKEND;
  const backend = resolveRequestedBackend({
    ...hasBackend ? { backend: backendValue } : {},
    ...hasModel ? { model: modelValue } : {}
  });
  if (backend !== CANONICAL_BACKEND && backend !== CACHED_CANARY_BACKEND) {
    throw new CanonicalError(400, "UNSUPPORTED_CANONICAL_BACKEND", `Canonical routes support '${CANONICAL_BACKEND}' and '${CACHED_CANARY_BACKEND}'.`);
  }
  return backend;
}
__name(canonicalBackendFromUrl, "canonicalBackendFromUrl");
__name2(canonicalBackendFromUrl, "canonicalBackendFromUrl");
function capabilityCacheJobId(backend) {
  return backend === CANONICAL_BACKEND ? CAPABILITY_CACHE_JOB_ID : `${CAPABILITY_CACHE_JOB_ID}.${backend}`;
}
__name(capabilityCacheJobId, "capabilityCacheJobId");
__name2(capabilityCacheJobId, "capabilityCacheJobId");
var ASSET_AUTHORIZE_FIELDS = /* @__PURE__ */ new Set([
  "accountScope",
  "uploadSessionId",
  "type",
  "contentType",
  "declaredSize"
]);
var ASSET_DESCRIPTOR_FIELDS = /* @__PURE__ */ new Set([
  "assetId",
  "objectKey",
  "type",
  "contentType",
  "size",
  "status",
  "accountScope",
  "uploadSessionId",
  "expiresAt"
]);
function rejectUnknownObjectFields(value, allowed, path) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new CanonicalError(400, "INVALID_REQUEST", `${path} must be an object.`);
  }
  for (const name of Object.keys(value)) {
    if (!allowed.has(name)) {
      throw new CanonicalError(400, "UNKNOWN_FIELD", `Unknown field '${path}.${name}'.`, { field: `${path}.${name}` });
    }
  }
}
__name(rejectUnknownObjectFields, "rejectUnknownObjectFields");
__name2(rejectUnknownObjectFields, "rejectUnknownObjectFields");
function canonicalR2Config(env) {
  return {
    accountId: String(env.R2_ACCOUNT_ID || "").trim(),
    accessKeyId: String(env.R2_ACCESS_KEY_ID || "").trim(),
    secretAccessKey: String(env.R2_SECRET_ACCESS_KEY || "").trim(),
    bucket: String(env.R2_BUCKET_NAME || "minimax-h3-private-output").trim(),
    putTtl: Math.max(600, Math.min(1800, toInteger(env.R2_PRESIGN_PUT_TTL_SECONDS, 1200))),
    getTtl: Math.max(600, Math.min(1800, toInteger(env.R2_PRESIGN_GET_TTL_SECONDS, 1800)))
  };
}
__name(canonicalR2Config, "canonicalR2Config");
__name2(canonicalR2Config, "canonicalR2Config");
async function signReferencePut(env, asset) {
  const config = canonicalR2Config(env);
  return presignR2({
    ...config,
    objectKey: asset.objectKey,
    method: "PUT",
    expiresSeconds: config.putTtl,
    contentType: asset.contentType,
    contentLength: asset.declaredSize
  });
}
__name(signReferencePut, "signReferencePut");
__name2(signReferencePut, "signReferencePut");
async function signReferenceGet(env, objectKey) {
  const config = canonicalR2Config(env);
  const signed = await presignR2({
    ...config,
    objectKey,
    method: "GET",
    expiresSeconds: config.getTtl
  });
  return signed.url;
}
__name(signReferenceGet, "signReferenceGet");
__name2(signReferenceGet, "signReferenceGet");
async function authorizeCanonicalAsset(request, env, headers) {
  const body = await readJsonObject(request);
  rejectSecretFields(body);
  rejectUnknownObjectFields(body, ASSET_AUTHORIZE_FIELDS, "request");
  const asset = createReferenceObject(body);
  const signed = await signReferencePut(env, asset);
  return json({
    asset: {
      assetId: asset.assetId,
      objectKey: asset.objectKey,
      type: asset.type,
      contentType: asset.contentType,
      declaredSize: asset.declaredSize,
      status: "authorized",
      uploadSessionId: String(body.uploadSessionId),
      accountScope: String(body.accountScope)
    },
    upload: {
      url: signed.url,
      method: "PUT",
      headers: { "Content-Type": asset.contentType },
      expiresAt: signed.expiresAt
    }
  }, 201, headers);
}
__name(authorizeCanonicalAsset, "authorizeCanonicalAsset");
__name2(authorizeCanonicalAsset, "authorizeCanonicalAsset");
var LORA_KEY_RE = /^loras\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\.safetensors$/;
async function authorizeCanonicalLora(request, env, headers) {
  const body = await readJsonObject(request);
  rejectSecretFields(body);
  rejectUnknownObjectFields(body, /* @__PURE__ */ new Set(["accountScope", "loraId", "sha256", "bytes"]), "request");
  const accountScope = safeSegment(body.accountScope, "accountScope");
  const loraId = safeSegment(body.loraId, "loraId");
  const sha256 = String(body.sha256 || "").trim().toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(sha256)) fail(400, "INVALID_LORA", "sha256 must be a 64-character hex digest.");
  const bytes = Number(body.bytes);
  if (!Number.isInteger(bytes) || bytes <= 0 || bytes > 300 * 1024 * 1024) {
    fail(400, "INVALID_LORA", "bytes must be a positive integer up to 300 MiB.");
  }
  const objectKey = `loras/${accountScope}/${loraId}/${sha256}.safetensors`;
  const config = canonicalR2Config(env);
  const signed = await presignR2({
    ...config,
    objectKey,
    method: "PUT",
    expiresSeconds: config.putTtl,
    contentType: "application/octet-stream",
    contentLength: bytes
  });
  return json({
    lora: { loraId, objectKey, sha256, bytes, status: "authorized" },
    upload: {
      url: signed.url,
      method: "PUT",
      headers: { "Content-Type": "application/octet-stream" },
      expiresAt: signed.expiresAt
    }
  }, 201, headers);
}
__name(authorizeCanonicalLora, "authorizeCanonicalLora");
__name2(authorizeCanonicalLora, "authorizeCanonicalLora");
async function confirmCanonicalLora(request, env, headers) {
  const body = await readJsonObject(request);
  rejectSecretFields(body);
  rejectUnknownObjectFields(body, /* @__PURE__ */ new Set(["objectKey"]), "request");
  const objectKey = String(body.objectKey || "");
  if (!LORA_KEY_RE.test(objectKey) || objectKey.includes("..")) {
    fail(400, "INVALID_LORA", "LoRA object key is invalid.");
  }
  const config = canonicalR2Config(env);
  const signed = await presignR2({ ...config, objectKey, method: "HEAD", expiresSeconds: config.getTtl });
  const resp = await fetch(signed.url, { method: "HEAD" });
  if (!resp.ok) fail(404, "LORA_NOT_FOUND", "The LoRA object is not present in storage.");
  return json({
    lora: { objectKey, status: "ready", sizeBytes: Number(resp.headers.get("content-length") || 0) }
  }, 200, headers);
}
__name(confirmCanonicalLora, "confirmCanonicalLora");
__name2(confirmCanonicalLora, "confirmCanonicalLora");
async function deleteCanonicalLora(request, env, headers) {
  const body = await readJsonObject(request);
  rejectSecretFields(body);
  rejectUnknownObjectFields(body, /* @__PURE__ */ new Set(["objectKey"]), "request");
  const objectKey = String(body.objectKey || "");
  if (!LORA_KEY_RE.test(objectKey) || objectKey.includes("..")) {
    fail(400, "INVALID_LORA", "LoRA object key is invalid.");
  }
  const config = canonicalR2Config(env);
  const signed = await presignR2({ ...config, objectKey, method: "DELETE", expiresSeconds: 300 });
  const resp = await fetch(signed.url, { method: "DELETE" });
  if (!resp.ok && resp.status !== 404) fail(502, "LORA_DELETE_FAILED", "Storage rejected the delete.");
  return json({ lora: { objectKey, status: "deleted" } }, 200, headers);
}
__name(deleteCanonicalLora, "deleteCanonicalLora");
__name2(deleteCanonicalLora, "deleteCanonicalLora");
function canonicalAssetDescriptor(body, path = "request") {
  rejectUnknownObjectFields(body, ASSET_DESCRIPTOR_FIELDS, path);
  const descriptor = {
    assetId: String(body.assetId || "").trim(),
    objectKey: String(body.objectKey || "").trim(),
    type: String(body.type || "").trim(),
    contentType: String(body.contentType || "").trim().toLowerCase(),
    size: Number(body.size),
    status: String(body.status || "").trim(),
    accountScope: String(body.accountScope || "").trim(),
    uploadSessionId: String(body.uploadSessionId || "").trim(),
    expiresAt: body.expiresAt
  };
  validateObjectKey(descriptor.objectKey, descriptor);
  const typeSpec = MEDIA_TYPES[descriptor.type];
  if (!typeSpec || !typeSpec.contentTypes[descriptor.contentType]) {
    throw new CanonicalError(415, "INVALID_REFERENCE_TYPE", "Asset content type is invalid.");
  }
  return descriptor;
}
__name(canonicalAssetDescriptor, "canonicalAssetDescriptor");
__name2(canonicalAssetDescriptor, "canonicalAssetDescriptor");
function r2ObjectContentType(object) {
  return String(object?.httpMetadata?.contentType || object?.httpMetadata?.content_type || "").split(";")[0].trim().toLowerCase();
}
__name(r2ObjectContentType, "r2ObjectContentType");
__name2(r2ObjectContentType, "r2ObjectContentType");
async function verifyCanonicalAsset(env, descriptor, { deleteInvalid = false } = {}) {
  const bucket = requireBucket(env);
  if (typeof bucket.head !== "function") throw new CanonicalError(500, "R2_HEAD_UNAVAILABLE", "R2 metadata verification is unavailable.");
  const object = await bucket.head(descriptor.objectKey);
  if (!object) throw new CanonicalError(404, "ASSET_NOT_FOUND", "Uploaded reference asset was not found.");
  const actualSize = Number(object.size || 0);
  const actualType = r2ObjectContentType(object);
  const spec = MEDIA_TYPES[descriptor.type];
  const valid = actualSize > 0 && actualSize <= spec.maxBytes && actualType === descriptor.contentType;
  if (!valid) {
    if (deleteInvalid) {
      try {
        await bucket.delete(descriptor.objectKey);
      } catch {
      }
    }
    throw new CanonicalError(422, "ASSET_METADATA_MISMATCH", "Uploaded asset metadata does not match its authorization.");
  }
  if (Number.isFinite(descriptor.size) && descriptor.size > 0 && actualSize !== descriptor.size) {
    if (deleteInvalid) {
      try {
        await bucket.delete(descriptor.objectKey);
      } catch {
      }
    }
    throw new CanonicalError(422, "ASSET_SIZE_MISMATCH", "Uploaded asset size does not match its authorization.");
  }
  return { size: actualSize, contentType: actualType, etag: object.httpEtag || object.etag || null };
}
__name(verifyCanonicalAsset, "verifyCanonicalAsset");
__name2(verifyCanonicalAsset, "verifyCanonicalAsset");
async function confirmCanonicalAsset(request, env, headers) {
  const body = await readJsonObject(request);
  rejectSecretFields(body);
  const descriptor = canonicalAssetDescriptor(body);
  const metadata = await verifyCanonicalAsset(env, descriptor, { deleteInvalid: true });
  return json({
    asset: {
      assetId: descriptor.assetId,
      objectKey: descriptor.objectKey,
      type: descriptor.type,
      contentType: metadata.contentType,
      size: metadata.size,
      status: "uploaded",
      etag: metadata.etag
    }
  }, 200, headers);
}
__name(confirmCanonicalAsset, "confirmCanonicalAsset");
__name2(confirmCanonicalAsset, "confirmCanonicalAsset");
async function deleteCanonicalAsset(request, env, headers) {
  const body = await readJsonObject(request);
  rejectSecretFields(body);
  const descriptor = canonicalAssetDescriptor(body);
  await requireBucket(env).delete(descriptor.objectKey);
  return json({ assetId: descriptor.assetId, deleted: true }, 200, headers);
}
__name(deleteCanonicalAsset, "deleteCanonicalAsset");
__name2(deleteCanonicalAsset, "deleteCanonicalAsset");
function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
__name(sleep, "sleep");
__name2(sleep, "sleep");
async function refreshCanonicalCapabilities(env, backend = CANONICAL_BACKEND) {
  const config = getRunPodConfig(backend, env);
  const base = `https://api.runpod.ai/v2/${config.endpointId}`;
  const cacheJobId = capabilityCacheJobId(backend);
  const previous = await readJobState(env, cacheJobId);
  const pendingAge = previous?.pendingSubmittedAt ? Date.now() - Date.parse(previous.pendingSubmittedAt) : Infinity;
  let runpodId = previous?.pendingRunpodId && Number.isFinite(pendingAge) && pendingAge < CAPABILITY_PENDING_MAX_AGE_MS ? previous.pendingRunpodId : null;
  if (!runpodId) {
    const submission = await fetch(`${base}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${config.apiKey}` },
      body: JSON.stringify({ input: { mode: "capabilities" } })
    });
    const accepted = await safeJson(submission);
    if (!submission.ok || !accepted?.id) {
      throw new CanonicalError(502, "CAPABILITY_PROBE_FAILED", "RunPod capability probe was not accepted.");
    }
    runpodId = accepted.id;
    await pushJobState(env, cacheJobId, {
      pendingRunpodId: runpodId,
      pendingSubmittedAt: (/* @__PURE__ */ new Date()).toISOString(),
      backend
    });
  }
  const deadline = Date.now() + CAPABILITY_REFRESH_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const response = await fetch(`${base}/status/${encodeURIComponent(runpodId)}`, {
      headers: { Authorization: `Bearer ${config.apiKey}` }
    });
    const data = await safeJson(response);
    if (!response.ok) throw new CanonicalError(502, "CAPABILITY_PROBE_FAILED", "RunPod capability status failed.");
    const status = String(data.status || "").toUpperCase();
    if (status === "COMPLETED") {
      if (data.output?.error) throw new CanonicalError(502, "CAPABILITY_PROBE_FAILED", "RunPod capability handler returned an error.");
      const capabilities = data.output?.capabilities;
      if (!isCanonicalCapabilityDocument(capabilities)) {
        await pushJobState(env, cacheJobId, {
          capabilities: null,
          fetchedAt: null,
          pendingRunpodId: null,
          pendingSubmittedAt: null,
          lastRejectedBuild: buildIdentity(capabilities),
          lastRejectedAt: (/* @__PURE__ */ new Date()).toISOString()
        });
        throw new CanonicalError(502, "CAPABILITY_PROBE_INVALID", "RunPod returned no canonical capability document.");
      }
      const record = {
        capabilities,
        fetchedAt: (/* @__PURE__ */ new Date()).toISOString(),
        runpodId,
        backend,
        pendingRunpodId: null,
        pendingSubmittedAt: null
      };
      await pushJobState(env, cacheJobId, record);
      return { ...record, cache: "refreshed" };
    }
    if (["FAILED", "CANCELLED", "TIMED_OUT"].includes(status)) {
      await pushJobState(env, cacheJobId, { pendingRunpodId: null, pendingSubmittedAt: null });
      throw new CanonicalError(502, "CAPABILITY_PROBE_FAILED", `RunPod capability probe ended as ${status}.`);
    }
    await sleep(300);
  }
  throw new CanonicalError(503, "CAPABILITY_PROBE_PENDING", "RunPod capability discovery is still pending.");
}
__name(refreshCanonicalCapabilities, "refreshCanonicalCapabilities");
__name2(refreshCanonicalCapabilities, "refreshCanonicalCapabilities");
async function getCanonicalCapabilities(env, forceRefresh = false, backend = CANONICAL_BACKEND) {
  const cached = await readJobState(env, capabilityCacheJobId(backend));
  const age = cached?.fetchedAt ? Date.now() - Date.parse(cached.fetchedAt) : Infinity;
  const rejectedAge = cached?.lastRejectedAt ? Date.now() - Date.parse(cached.lastRejectedAt) : Infinity;
  const cachedIsCanonical = isCanonicalCapabilityDocument(cached?.capabilities);
  if (!forceRefresh && cachedIsCanonical && Number.isFinite(age) && age < CAPABILITY_CACHE_TTL_MS) {
    return { capabilities: cached.capabilities, fetchedAt: cached.fetchedAt, runpodId: cached.runpodId || null, cache: "hit" };
  }
  if (!forceRefresh && !cachedIsCanonical && Number.isFinite(rejectedAge) && rejectedAge < CAPABILITY_CACHE_TTL_MS) {
    throw new CanonicalError(503, "DEPLOYMENT_ACTION_REQUIRED", backend === CANONICAL_BACKEND ? "The Blackwell endpoint did not report canonical multimodal-4 capabilities." : "The cached-model canary endpoint did not report canonical capabilities.", {
      expected: backend === CANONICAL_BACKEND ? EXPECTED_H3_BUILD : EXPECTED_CACHED_CANARY_RELEASE,
      actual: cached?.lastRejectedBuild || null
    });
  }
  try {
    return await refreshCanonicalCapabilities(env, backend);
  } catch (error) {
    if (cachedIsCanonical) {
      return { capabilities: cached.capabilities, fetchedAt: cached.fetchedAt, runpodId: cached.runpodId || null, cache: "stale" };
    }
    throw error;
  }
}
__name(getCanonicalCapabilities, "getCanonicalCapabilities");
__name2(getCanonicalCapabilities, "getCanonicalCapabilities");
async function serveCanonicalCapabilities(env, headers, forceRefresh, backend = CANONICAL_BACKEND) {
  const result = await getCanonicalCapabilities(env, forceRefresh, backend);
  return json(result.capabilities, 200, {
    ...headers,
    "X-Privora-Capability-Cache": result.cache,
    "X-Privora-Capability-Fetched-At": result.fetchedAt || ""
  });
}
__name(serveCanonicalCapabilities, "serveCanonicalCapabilities");
__name2(serveCanonicalCapabilities, "serveCanonicalCapabilities");
async function cleanupReferenceObjects(env, objectKeys) {
  const keys = [...new Set((objectKeys || []).map((key) => validateObjectKey(key)))];
  if (!keys.length) return 0;
  await requireBucket(env).delete(keys.length === 1 ? keys[0] : keys);
  return keys.length;
}
__name(cleanupReferenceObjects, "cleanupReferenceObjects");
__name2(cleanupReferenceObjects, "cleanupReferenceObjects");
async function generateCanonicalVideo(request, env, headers, url) {
  const backend = canonicalBackendFromUrl(url);
  const body = await readJsonObject(request);
  rejectSecretFields(body);
  rejectUnknownObjectFields(body, CANONICAL_OUTER_FIELDS, "request");
  if (!Array.isArray(body.assets || [])) throw new CanonicalError(400, "INVALID_ASSET", "assets must be an array.");
  for (const [index, descriptor] of (body.assets || []).entries()) {
    rejectUnknownObjectFields(descriptor, ASSET_DESCRIPTOR_FIELDS, `assets[${index}]`);
  }
  const capabilityResult = await getCanonicalCapabilities(env, false, backend);
  const capabilities = capabilityResult.capabilities;
  if (backend === CANONICAL_BACKEND && !buildMatchesExpected(capabilities)) {
    throw new CanonicalError(503, "DEPLOYMENT_ACTION_REQUIRED", "The Blackwell endpoint is not running the required multimodal-4 build.", {
      expected: EXPECTED_H3_BUILD,
      actual: buildIdentity(capabilities)
    });
  }
  if (backend === CACHED_CANARY_BACKEND && !cachedCanaryReleaseMatchesExpected(capabilities)) {
    throw new CanonicalError(503, "DEPLOYMENT_ACTION_REQUIRED", "The cached-model canary endpoint is not running the required pinned release.", {
      expected: EXPECTED_CACHED_CANARY_RELEASE,
      actual: cachedCanaryReleaseIdentity(capabilities)
    });
  }
  const validated = validateCanonicalInput(body.input, capabilities);
  const privacy = await normalizePrivacy({
    privacyMode: validated.privacyMode,
    encryption: validated.input.encryption
  });
  const descriptors = (body.assets || []).map((item) => canonicalAssetDescriptor(item, "asset"));
  for (const descriptor of descriptors) await verifyCanonicalAsset(env, descriptor);
  const resolved = await resolveReferenceAssets(
    validated.input,
    descriptors,
    (objectKey) => signReferenceGet(env, objectKey)
  );
  const jobId = crypto.randomUUID();
  // User LoRAs are URL-addressed (HuggingFace/Civitai style). The GPU downloads
  // each one inside the job, applies it, and purges it with the job directory;
  // the Worker neither stores nor proxies LoRA bytes.
  const lorasWithUrls = (validated.input.loras || []).map((entry) => ({
    url: entry.url,
    ...(entry.sha256 ? { sha256: entry.sha256 } : {}),
    strength: entry.strength
  }));
  if (!env.JOB_TOKEN_SECRET) {
    throw new CanonicalError(503, "CALLBACKS_NOT_CONFIGURED", "Canonical generation requires authenticated progress and output callbacks.");
  }
  const callbacks = await buildCallbackBlocks(env, jobId, url.origin, {}, privacy);
  function b64uEncode(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
__name(b64uEncode, "b64uEncode");
__name2(b64uEncode, "b64uEncode");
async function importJobPayloadKey(spkiB64Url) {
  const pad = "=".repeat((4 - spkiB64Url.length % 4) % 4);
  const der = atob(spkiB64Url.replace(/-/g, "+").replace(/_/g, "/") + pad);
  const bytes = new Uint8Array(der.length);
  for (let i = 0; i < der.length; i++) bytes[i] = der.charCodeAt(i);
  return crypto.subtle.importKey("spki", bytes, { name: "RSA-OAEP", hash: "SHA-256" }, false, ["encrypt"]);
}
__name(importJobPayloadKey, "importJobPayloadKey");
__name2(importJobPayloadKey, "importJobPayloadKey");
// Seal a RunPod job input so only ciphertext rests in RunPod's job records.
// The Worker holds the public half (env var); only the GPU worker's private
// key (RunPod endpoint env) can open the envelope. When the public key is not
// configured the input is sent unchanged, so key rollout is per-deployment.
async function sealJobInput(env, input) {
  const publicKey = String(env.JOB_PAYLOAD_PUBLIC_KEY || "").trim();
  if (!publicKey) return input;
  try {
    const key = await importJobPayloadKey(publicKey);
    const dataKey = await crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, true, ["encrypt"]);
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const plaintext = new TextEncoder().encode(JSON.stringify(input));
    const ct = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, dataKey, plaintext);
    const wk = await crypto.subtle.encrypt({ name: "RSA-OAEP" }, key, dataKey);
    return { pv_envelope: { v: 1, alg: "RSA-OAEP-256+AES-256-GCM", wk: b64uEncode(wk), iv: b64uEncode(iv), ct: b64uEncode(ct) } };
  } catch (error) {
    throw new HttpError(503, "Job payload sealing is configured but failed. Generation refused rather than sending plaintext.");
  }
}
__name(sealJobInput, "sealJobInput");
__name2(sealJobInput, "sealJobInput");
const canonicalInput = {
    ...resolved.input,
    ...lorasWithUrls.length ? { loras: lorasWithUrls } : {},
    privacy: { mode: privacy.mode },
    ...privacy.encryption ? { encryption: {
      version: privacy.encryption.version,
      algorithm: privacy.encryption.algorithm,
      keyWrapAlgorithm: privacy.encryption.keyWrapAlgorithm,
      publicKeyAlgorithm: privacy.encryption.publicKeyAlgorithm,
      publicKey: privacy.encryption.publicKey,
      keyId: privacy.encryption.keyId
    } } : {},
    progress: callbacks.progress,
    output: callbacks.output
  };
  if (Object.prototype.hasOwnProperty.call(canonicalInput, "workflow")) {
    throw new CanonicalError(500, "CANONICAL_WORKFLOW_REGRESSION", "Canonical input must not contain workflow.");
  }
  const config = getRunPodConfig(backend, env);
  // Upscale tiers re-sample at the enlarged resolution after the latent upscale;
  // HD at long durations paces ~37s/step with sage3 off (quadratic attention in
  // frames), so 15s HD needs ~13-15 min. Both get the extended budget.
  const isLongRunningTier = ["2k", "4k", "hd"].includes(validated.input.quality);
  const response = await fetch(`https://api.runpod.ai/v2/${config.endpointId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${config.apiKey}` },
    body: JSON.stringify({
      input: canonicalInput,
      policy: { executionTimeout: isLongRunningTier ? 15e5 : 6e5, ttl: 18e5 }
    })
  });
  const data = await safeJson(response);
  if (!response.ok || !data?.id) {
    try {
      await cleanupReferenceObjects(env, resolved.objectKeys);
    } catch {
    }
    if (!response.ok) return runPodErrorResponse({ operation: "canonical submission", backend, upstreamStatus: response.status, data, headers });
    throw new CanonicalError(502, "UPSTREAM_INVALID_RESPONSE", "RunPod submission returned no job ID.");
  }
  const state = {
    status: data.status || "IN_QUEUE",
    phase: "queued",
    percent: 0,
    runpodId: data.id,
    backend,
    canonical: true,
    mode: validated.input.mode,
    family: validated.family,
    quality: validated.input.quality,
    generationMode: validated.input.generationMode,
    privacyMode: privacy.mode,
    referenceObjectKeys: resolved.objectKeys,
    workerBuild: buildIdentity(capabilities),
    referenceCounts: validated.counts
  };
  await pushJobState(env, jobId, state);
  console.log([
    `generation_id=${jobId}`,
    "canonical=true",
    `backend=${backend}`,
    `mode=${validated.input.mode}`,
    `family=${validated.family}`,
    `quality=${validated.input.quality}`,
    `generation_mode=${validated.input.generationMode}`,
    `privacy_mode=${privacy.mode}`,
    `reference_count=${validated.counts.total}`
  ].join(" "));
  const encodedRunPodId = encodeURIComponent(data.id);
  const encodedJobId = encodeURIComponent(jobId);
  return json({
    canonical: true,
    backend,
    id: data.id,
    jobId,
    status: data.status || "IN_QUEUE",
    mode: validated.input.mode,
    family: validated.family,
    quality: validated.input.quality,
    generationMode: validated.input.generationMode,
    aspectRatio: validated.input.aspectRatio,
    requestedDuration: validated.input.duration,
    seed: validated.input.seed ?? null,
    privacyMode: privacy.mode,
    worker: { build: buildIdentity(capabilities) },
    referenceCounts: validated.counts,
    routes: {
      status: `/status/${encodeURIComponent(backend)}/${encodedRunPodId}?jobId=${encodedJobId}`,
      cancel: `/cancel/${encodeURIComponent(backend)}/${encodedRunPodId}?jobId=${encodedJobId}`,
      events: `/ws/jobs/${encodedJobId}`,
      artifact: `/jobs/${encodedJobId}/artifact`,
      deleteGeneration: `/jobs/${encodedJobId}`
    }
  }, 202, headers);
}
__name(generateCanonicalVideo, "generateCanonicalVideo");
__name2(generateCanonicalVideo, "generateCanonicalVideo");
async function generateVideo(request, env, headers, url) {
  const body = await readJsonObject(request);
  rejectSecretFields(body);
  const backend = resolveRequestedBackend(body);
  if (backend === CACHED_CANARY_BACKEND) {
    throw new HttpError(400, "h3-cached-canary accepts only the canonical /canonical/generate route");
  }
  const config = getRunPodConfig(backend, env);
  const settings = normalizeRequest(body);
  const privacy = await normalizePrivacy(body);
  const workflow = buildWorkflowForSettings(settings);
  const jobId = crypto.randomUUID();
  const callbacks = env.JOB_TOKEN_SECRET ? await buildCallbackBlocks(env, jobId, url.origin, settings, privacy) : null;
  if (!callbacks) {
    console.warn("JOB_TOKEN_SECRET is unset: progress callbacks and R2 output are disabled");
  }
  if (privacy.spec.encrypts && !callbacks) {
    throw new HttpError(
      503,
      "Confidential generation is unavailable on this deployment: JOB_TOKEN_SECRET is not configured, so there is no authenticated path for the encrypted artefact to reach storage. Standard generation is unaffected."
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
      policy: { executionTimeout: 6e5, ttl: 18e5 }
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
        ...privacy.expiresAt ? { expiresAt: privacy.expiresAt } : {}
      });
    } catch (error) {
      console.warn(`Could not seed job channel: ${errorCodeForLog(error)}`);
    }
    try {
      await addToActiveJobsIndex(env, jobId, data.id, backend);
    } catch (error) {
      console.warn(`Could not index job for terminal capture: ${errorCodeForLog(error)}`);
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
      ...privacy.encryption ? {
        encryption: {
          version: privacy.encryption.version,
          algorithm: privacy.encryption.algorithm,
          keyWrapAlgorithm: privacy.encryption.keyWrapAlgorithm,
          // The derived id, not the key: it is what a client matches an artefact
          // against to know which private key to unlock.
          keyId: privacy.encryption.keyId
        }
      } : {},
      ...privacy.expiresAt ? { expiresAt: privacy.expiresAt } : {},
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
__name(generateVideo, "generateVideo");
__name2(generateVideo, "generateVideo");
async function buildCallbackBlocks(env, jobId, origin, settings, privacy) {
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
      // Repeated here, not only in the progress block: the handler stamps this id into the
      // encrypted container's authenticated header, and the upload endpoint checks it
      // matches. Deriving it from a *different* block would make the two agree only by
      // coincidence.
      jobId,
      token: await signJobToken(secret, jobId, TOKEN_PURPOSES.output, ttl, {
        pm: privacy?.mode ?? DEFAULT_PRIVACY_MODE,
        // Signed too, so a v2 job cannot have a v1 container filed against it. The
        // uploader can read this claim; it cannot change it.
        ...privacy?.encryption ? { cv: privacy.encryption.version } : {},
        ...privacy?.expiresAt ? { exa: privacy.expiresAt } : {}
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
__name(buildCallbackBlocks, "buildCallbackBlocks");
__name2(buildCallbackBlocks, "buildCallbackBlocks");
var ACTIVE_JOBS_INDEX_ID = "active-jobs-index";
var ACTIVE_JOB_MAX_AGE_MS = 3 * 60 * 60 * 1000;
async function addToActiveJobsIndex(env, workerJobId, runpodId, backend) {
  const index = (await readJobState(env, ACTIVE_JOBS_INDEX_ID)) || {};
  const jobs = index.jobs && typeof index.jobs === "object" ? index.jobs : {};
  jobs[workerJobId] = { runpodId, backend, submittedAt: (/* @__PURE__ */ new Date()).toISOString() };
  await pushJobState(env, ACTIVE_JOBS_INDEX_ID, { jobs, updatedAt: (/* @__PURE__ */ new Date()).toISOString() });
}
__name(addToActiveJobsIndex, "addToActiveJobsIndex");
__name2(addToActiveJobsIndex, "addToActiveJobsIndex");
async function removeFromActiveJobsIndex(env, workerJobId) {
  const index = (await readJobState(env, ACTIVE_JOBS_INDEX_ID)) || {};
  const jobs = index.jobs && typeof index.jobs === "object" ? index.jobs : {};
  if (!(workerJobId in jobs)) return;
  delete jobs[workerJobId];
  await pushJobState(env, ACTIVE_JOBS_INDEX_ID, { jobs, updatedAt: (/* @__PURE__ */ new Date()).toISOString() });
}
__name(removeFromActiveJobsIndex, "removeFromActiveJobsIndex");
__name2(removeFromActiveJobsIndex, "removeFromActiveJobsIndex");
async function persistTerminalResult(env, workerJobId, result) {
  await pushJobState(env, workerJobId, { terminalResult: result, terminalSavedAt: (/* @__PURE__ */ new Date()).toISOString() });
  try {
    await removeFromActiveJobsIndex(env, workerJobId);
  } catch {
  }
}
__name(persistTerminalResult, "persistTerminalResult");
__name2(persistTerminalResult, "persistTerminalResult");
// The cron sweep guarantees a terminal snapshot is taken even when no client
// ever polls while RunPod still has the record: without it, a finished job
// nobody watched loses its executionTime (runtime billing) and media refs.
async function sweepActiveJobs(env) {
  const index = await readJobState(env, ACTIVE_JOBS_INDEX_ID);
  const jobs = index?.jobs && typeof index.jobs === "object" ? index.jobs : {};
  const now = Date.now();
  let changed = false;
  for (const workerJobId of Object.keys(jobs)) {
    const entry = jobs[workerJobId];
    if (!entry?.runpodId || !entry?.backend) {
      delete jobs[workerJobId];
      changed = true;
      continue;
    }
    if (!Date.parse(entry.submittedAt || 0) || now - Date.parse(entry.submittedAt) > ACTIVE_JOB_MAX_AGE_MS) {
      delete jobs[workerJobId];
      changed = true;
      continue;
    }
    try {
      const config = getRunPodConfig(entry.backend, env);
      const response = await fetch(`https://api.runpod.ai/v2/${config.endpointId}/status/${encodeURIComponent(entry.runpodId)}`, {
        method: "GET",
        headers: { Authorization: `Bearer ${config.apiKey}` }
      });
      if (!response.ok) continue;
      const data = await safeJson(response);
      const normalized = normalizeRunPodJobStatus(data);
      if (!["completed", "failed", "cancelled"].includes(normalized.status)) continue;
      const result = addBackendToResult(entry.backend, sanitizeRunPodResult(data));
      result.productStatus = normalized.status;
      if (normalized.errorCode) result.errorCode = normalized.errorCode;
      if (normalized.timedOut) result.timedOut = true;
      const state = await readJobState(env, workerJobId);
      Object.assign(
        result,
        describeArtifactForStatus(workerJobId, state, result?.output?.video),
        describeProgressForStatus(state)
      );
      if (state?.referenceObjectKeys?.length) {
        await cleanupReferenceObjects(env, state.referenceObjectKeys);
        await pushJobState(env, workerJobId, { referenceObjectKeys: [], referencesDeletedAt: (/* @__PURE__ */ new Date()).toISOString() });
      }
      await persistTerminalResult(env, workerJobId, result);
      delete jobs[workerJobId];
      changed = true;
    } catch (error) {
      console.warn(`active job sweep skipped ${workerJobId}: ${errorCodeForLog(error)}`);
    }
  }
  if (changed) {
    await pushJobState(env, ACTIVE_JOBS_INDEX_ID, { jobs, updatedAt: (/* @__PURE__ */ new Date()).toISOString() });
  }
}
__name(sweepActiveJobs, "sweepActiveJobs");
__name2(sweepActiveJobs, "sweepActiveJobs");
async function getJobStatus(backend, jobId, env, headers, url) {
  const config = getRunPodConfig(backend, env);
  const workerJobId = url?.searchParams.get("jobId");
  const runpodUrl = `https://api.runpod.ai/v2/${config.endpointId}/status/${encodeURIComponent(jobId)}`;
  const response = await fetch(runpodUrl, {
    method: "GET",
    headers: { Authorization: `Bearer ${config.apiKey}` }
  });
  const data = await safeJson(response);
  if (!response.ok) {
    // RunPod purges completed job records after ~30 minutes. If this worker ever
    // observed a terminal status (poll or cron sweep), the persisted snapshot -
    // including executionTime, which settles runtime billing - outlives the
    // upstream record and is served unchanged.
    if (workerJobId) {
      try {
        const state = await readJobState(env, workerJobId);
        if (state?.terminalResult) {
          return json(
            Object.assign({}, state.terminalResult, { persisted: true, persistedAt: state.terminalSavedAt || null }),
            200,
            headers
          );
        }
      } catch {
      }
    }
    return runPodErrorResponse({
      operation: "status",
      backend,
      upstreamStatus: response.status,
      data,
      headers
    });
  }
  const normalized = normalizeRunPodJobStatus(data);
  const result = addBackendToResult(backend, sanitizeRunPodResult(data));
  result.productStatus = normalized.status;
  if (normalized.errorCode) result.errorCode = normalized.errorCode;
  if (normalized.timedOut) result.timedOut = true;
  if (workerJobId) {
    try {
      const state = await readJobState(env, workerJobId);
      Object.assign(
        result,
        describeArtifactForStatus(workerJobId, state, result?.output?.video),
        describeProgressForStatus(state)
      );
      if (["completed", "failed", "cancelled"].includes(normalized.status)) {
        if (state?.referenceObjectKeys?.length) {
          await cleanupReferenceObjects(env, state.referenceObjectKeys);
          await pushJobState(env, workerJobId, { referenceObjectKeys: [], referencesDeletedAt: (/* @__PURE__ */ new Date()).toISOString() });
        }
        await persistTerminalResult(env, workerJobId, result);
      }
    } catch {
    }
  }
  return json(result, 200, headers);
}
__name(getJobStatus, "getJobStatus");
__name2(getJobStatus, "getJobStatus");
function normalizeRunPodJobStatus(data) {
  const upstream = String(data?.status || "").toUpperCase();
  const outputError = data?.output && typeof data.output === "object" && !Array.isArray(data.output) ? data.output.error : null;
  if (upstream === "COMPLETED" && outputError) {
    return { status: "failed", errorCode: classifyRunPodError(outputError), timedOut: false };
  }
  if (upstream === "COMPLETED") return { status: "completed", errorCode: null, timedOut: false };
  if (upstream === "IN_QUEUE") return { status: "queued", errorCode: null, timedOut: false };
  if (["IN_PROGRESS", "RUNNING"].includes(upstream)) return { status: "running", errorCode: null, timedOut: false };
  if (upstream === "CANCELLED") return { status: "cancelled", errorCode: "cancelled", timedOut: false };
  if (upstream === "TIMED_OUT") return { status: "failed", errorCode: "timed_out", timedOut: true };
  if (upstream === "FAILED") {
    return { status: "failed", errorCode: classifyRunPodError(data?.error || data?.output?.error), timedOut: false };
  }
  return { status: "running", errorCode: null, timedOut: false };
}
__name(normalizeRunPodJobStatus, "normalizeRunPodJobStatus");
__name2(normalizeRunPodJobStatus, "normalizeRunPodJobStatus");
function describeArtifactForStatus(jobId, state, runpodVideo) {
  const encoded = encodeURIComponent(jobId);
  const stored = state && state.artifact || null;
  const privacyMode = state?.privacyMode || stored?.privacyMode || DEFAULT_PRIVACY_MODE;
  const deleted = Boolean(state?.video?.deleted || stored?.deleted);
  const exists = Boolean(stored || runpodVideo && typeof runpodVideo === "object");
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
    const cryptoVersion = stored?.cryptoVersion ?? stored?.encryptionVersion ?? CONTAINER_VERSION;
    out.artifact.cryptoVersion = cryptoVersion;
    out.artifact.encryptionVersion = cryptoVersion;
    out.artifact.videoAlgorithm = stored?.algorithm || CONTAINER_SUITES[1];
    out.artifact.algorithm = out.artifact.videoAlgorithm;
    if (stored?.keyWrapAlgorithm) out.artifact.keyWrapAlgorithm = stored.keyWrapAlgorithm;
    if (stored?.keyId) out.artifact.keyId = stored.keyId;
    if (stored?.kdf) out.artifact.kdf = stored.kdf;
  }
  return out;
}
__name(describeArtifactForStatus, "describeArtifactForStatus");
__name2(describeArtifactForStatus, "describeArtifactForStatus");
function describeProgressForStatus(state) {
  if (!state || typeof state !== "object") return {};
  const event = publicEvent(state);
  const rawPhase = event?.phase || (event?.type !== "progress" ? event?.type : state.phase) || "queued";
  const phase = String(rawPhase).toLowerCase().replace(/[^a-z0-9_-]/g, "").slice(0, 40) || "queued";
  const progress = { phase };
  for (const field of ["step", "steps"]) {
    const value = Number(event?.[field]);
    if (Number.isInteger(value) && value >= 0) progress[field] = value;
  }
  const percent = Number(event?.percent);
  if (Number.isFinite(percent)) progress.percent = Math.max(0, Math.min(100, percent));
  if (phase === "completed" && progress.percent === void 0) progress.percent = 100;
  if (typeof state.updatedAt === "string") progress.updatedAt = state.updatedAt;
  return { progress };
}
__name(describeProgressForStatus, "describeProgressForStatus");
__name2(describeProgressForStatus, "describeProgressForStatus");
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
  const workerJobId = url?.searchParams.get("jobId");
  if (workerJobId) {
    try {
      const state = await readJobState(env, workerJobId);
      if (state?.referenceObjectKeys?.length) await cleanupReferenceObjects(env, state.referenceObjectKeys);
      await pushJobState(env, workerJobId, {
        status: "CANCELLED",
        phase: "cancelled",
        runpodId: jobId,
        referenceObjectKeys: [],
        referencesDeletedAt: (/* @__PURE__ */ new Date()).toISOString()
      });
    } catch (error) {
      console.warn(`Cancelled job ${jobId} but could not update its channel: ${errorCodeForLog(error)}`);
    }
  }
  return json(addBackendToResult(backend, sanitizeRunPodResult(data)), 200, headers);
}
__name(cancelJob, "cancelJob");
__name2(cancelJob, "cancelJob");
function buildRunPodInput(backend, workflow, settings, callbacks, privacy) {
  switch (backend) {
    case "h3":
    case "h3-blackwell": {
      const input = { workflow };
      if (privacy) {
        input.privacy = { mode: privacy.mode };
        if (privacy.encryption) {
          input.encryption = {
            version: privacy.encryption.version,
            algorithm: privacy.encryption.algorithm,
            keyWrapAlgorithm: privacy.encryption.keyWrapAlgorithm,
            publicKeyAlgorithm: privacy.encryption.publicKeyAlgorithm,
            publicKey: privacy.encryption.publicKey,
            keyId: privacy.encryption.keyId
          };
        }
      }
      if (callbacks) {
        input.progress = callbacks.progress;
        input.output = callbacks.output;
        if (callbacks.assets) {
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
__name(buildRunPodInput, "buildRunPodInput");
__name2(buildRunPodInput, "buildRunPodInput");
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
    const acceptedNames = [definition.apiKeyEnv, definition.fallbackApiKeyEnv].filter(Boolean).join(" or ");
    throw new HttpError(500, `Missing Cloudflare secret: ${acceptedNames}`);
  }
  return { backend, endpointId, apiKey };
}
__name(getRunPodConfig, "getRunPodConfig");
__name2(getRunPodConfig, "getRunPodConfig");
function isBackendConfigured(backend, env) {
  const definition = BACKENDS[backend];
  if (!definition) return false;
  const endpointId = String(env[definition.endpointIdEnv] || "").trim();
  const primaryApiKey = String(env[definition.apiKeyEnv] || "").trim();
  const fallbackApiKey = definition.fallbackApiKeyEnv ? String(env[definition.fallbackApiKeyEnv] || "").trim() : "";
  return Boolean(endpointId && (primaryApiKey || fallbackApiKey));
}
__name(isBackendConfigured, "isBackendConfigured");
__name2(isBackendConfigured, "isBackendConfigured");
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
__name(resolveRequestedBackend, "resolveRequestedBackend");
__name2(resolveRequestedBackend, "resolveRequestedBackend");
function normalizeBackend(value) {
  const key = String(value ?? DEFAULT_BACKEND).trim().toLowerCase();
  const backend = BACKEND_ALIASES[key];
  if (!backend) {
    throw new HttpError(400, `Unknown backend '${key}'. Use 'h3', 'h3-blackwell', or 'h3-cached-canary'.`);
  }
  return backend;
}
__name(normalizeBackend, "normalizeBackend");
__name2(normalizeBackend, "normalizeBackend");
function parseJobRoute(segments, url, routeName) {
  if (segments.length === 2) {
    const selectedBackend = url.searchParams.get("backend") ?? url.searchParams.get("model") ?? DEFAULT_BACKEND;
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
__name(parseJobRoute, "parseJobRoute");
__name2(parseJobRoute, "parseJobRoute");
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
__name(validateJobId, "validateJobId");
__name2(validateJobId, "validateJobId");
function splitPath(pathname) {
  return pathname.split("/").filter(Boolean).map((segment) => {
    try {
      return decodeURIComponent(segment);
    } catch {
      throw new HttpError(400, "Invalid URL encoding");
    }
  });
}
__name(splitPath, "splitPath");
__name2(splitPath, "splitPath");
function isExactRoute(segments, name) {
  return segments.length === 1 && segments[0] === name;
}
__name(isExactRoute, "isExactRoute");
__name2(isExactRoute, "isExactRoute");
function hasNonEmptyValue(value) {
  return value !== void 0 && value !== null && String(value).trim() !== "";
}
__name(hasNonEmptyValue, "hasNonEmptyValue");
__name2(hasNonEmptyValue, "hasNonEmptyValue");
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
__name(readJsonObject, "readJsonObject");
__name2(readJsonObject, "readJsonObject");
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
__name(runPodErrorResponse, "runPodErrorResponse");
__name2(runPodErrorResponse, "runPodErrorResponse");
function addBackendToResult(backend, result) {
  if (result && typeof result === "object" && !Array.isArray(result)) {
    return { backend, ...result };
  }
  return { backend, result };
}
__name(addBackendToResult, "addBackendToResult");
__name2(addBackendToResult, "addBackendToResult");
function routeDescriptions() {
  return [
    "GET /health",
    "GET /capabilities (optional backend=h3-cached-canary)",
    "GET /capabilities/legacy",
    "POST /canonical/generate (optional backend=h3-cached-canary)",
    "POST /canonical/assets/authorize",
    "POST /canonical/assets/confirm",
    "POST /canonical/assets/delete",
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
__name(routeDescriptions, "routeDescriptions");
__name2(routeDescriptions, "routeDescriptions");
const SECRET_FIELDS = new Set([
  "encryptionkey",
  "encryption_key",
  "passphrase",
  "password",
  "secret",
  "authorization",
  "cf-access-client-id",
  "cf_access_client_id",
  "cloudflare_access_client_id",
  "cf-access-client-secret",
  "cf_access_client_secret",
  "cloudflare_access_client_secret",
  "x-api-key",
  "api_key",
  "apikey",
  "token",
  "job_token_secret",
  "runpod_api_key",
  "dek",
  "data_key",
  "datakey",
  "wrapped_key",
  "wrappedkey",
  "h3_key_wrap_key",
  // Crypto v2. A private key, the key that protects it, and the per-video file key are
  // the three values whose exposure would undo the whole design.
  "privatekey",
  "private_key",
  "privateencryptionkey",
  "private_encryption_key",
  "encryptedprivatekey",
  "encrypted_private_key",
  "keyencryptionkey",
  "key_encryption_key",
  "kek",
  "fileencryptionkey",
  "file_encryption_key",
  "fek",
  "decryptionkey",
  "decryption_key",
  "derivedkey",
  "derived_key",
  "aeskey",
  "aes_key",
  "symmetrickey",
  "symmetric_key",
  "secretkey",
  "secret_key"
]);
const CRYPTO_PARENTS = new Set(["encryption", "crypto", "kdf", "confidential", "privacy"]);
const NEVER_REDACT = new Set([
  "keyid",
  "key_id",
  "r2_key",
  "storage_key",
  "object_key",
  "publickey",
  "public_key",
  "publickeyalgorithm",
  "public_key_algorithm",
  "keywrapalgorithm",
  "key_wrap_algorithm",
  "wrappedfilekey",
  "wrapped_file_key",
  "cryptoversion",
  "crypto_version",
  "salt"
]);
var REDACTED = "[redacted]";
var DOWNSTREAM_CONTENT_FIELDS = /* @__PURE__ */ new Set([
  "prompt",
  "negativeprompt",
  "inputprompt",
  "workflow",
  "input",
  "request",
  "requestbody",
  "responsebody",
  "body",
  "payload",
  "generationinput"
]);
function redactSecrets(value, parent = "", depth = 0) {
  if (depth > 12) return "[omitted: nesting too deep]";
  if (value === null || value === void 0) return value;
  if (Array.isArray(value)) {
    return value.map((item) => redactSecrets(item, parent, depth + 1));
  }
  if (typeof value === "object") {
    const out = {};
    for (const [name, child] of Object.entries(value)) {
      const lowered = String(name).trim().toLowerCase();
      if (NEVER_REDACT.has(lowered)) {
        out[name] = child;
      } else if (SECRET_FIELDS.has(lowered) || lowered === "key" && CRYPTO_PARENTS.has(parent.toLowerCase())) {
        out[name] = REDACTED;
      } else {
        out[name] = redactSecrets(child, name, depth + 1);
      }
    }
    return out;
  }
  return value;
}
__name(redactSecrets, "redactSecrets");
__name2(redactSecrets, "redactSecrets");
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
    `encryption=${privacy.encryption ? privacy.encryption.algorithm : "none"}`
  ];
  if (privacy.encryption?.keyId) fields.push(`key_id=${privacy.encryption.keyId}`);
  if (privacy.encryption?.keyWrapAlgorithm) {
    fields.push(`key_wrap=${privacy.encryption.keyWrapAlgorithm}`);
    fields.push(`crypto_version=${privacy.encryption.version}`);
  }
  if (privacy.expiresAt) fields.push(`expires_at=${privacy.expiresAt}`);
  console.log(fields.join(" "));
}
__name(logGeneration, "logGeneration");
__name2(logGeneration, "logGeneration");
function corsHeaders() {
  return {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, CF-Access-Client-Id, CF-Access-Client-Secret",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Max-Age": "86400",
    // Every JSON response here is a job's state or an error; none of it is cacheable and
    // some of it describes a confidential generation.
    "Cache-Control": "no-store",
    /*
     * Hardening for the JSON surface. These responses are never a document, so the
     * strictest possible policy costs nothing: nothing may be loaded, framed or executed
     * if a browser is ever persuaded to render one as HTML.
     */
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; sandbox",
    "Referrer-Policy": "no-referrer"
  };
}
__name(corsHeaders, "corsHeaders");
__name2(corsHeaders, "corsHeaders");
function sanitizeRunPodResult(data) {
  return redactSecrets(sanitizeRunPodShape(data));
}
__name(sanitizeRunPodResult, "sanitizeRunPodResult");
__name2(sanitizeRunPodResult, "sanitizeRunPodResult");
function sanitizeRunPodShape(data) {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return { code: "upstream_response_unavailable" };
  }
  const result = {
    id: data.id ?? null,
    status: data.status ?? null,
    delayTime: data.delayTime ?? null,
    executionTime: data.executionTime ?? null,
    workerId: data.workerId ?? data.worker_id ?? null
  };
  if (data.error !== void 0) {
    result.error = sanitizeProgressError(data.error);
  }
  if (data.errors !== void 0) {
    result.errors = Array.isArray(data.errors) ? data.errors.slice(0, 20).map((item) => sanitizeProgressError(item)) : [sanitizeProgressError(data.errors)];
  }
  if (data.output !== void 0) {
    result.output = sanitizePayload(data.output, "output");
  }
  if (data.code !== void 0) {
    result.code = normalizedErrorCode(data.code);
  }
  return result;
}
__name(sanitizeRunPodShape, "sanitizeRunPodShape");
__name2(sanitizeRunPodShape, "sanitizeRunPodShape");
function normalizedErrorCode(value) {
  const code = String(value || "upstream_error").toLowerCase().replace(/[^a-z0-9_.-]/g, "_").slice(0, 80);
  return code || "upstream_error";
}
__name(normalizedErrorCode, "normalizedErrorCode");
__name2(normalizedErrorCode, "normalizedErrorCode");
function classifyRunPodError(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const code = normalizedErrorCode(value.code || value.error_code || value.type || value.name || "generation_failed");
    if (!(/* @__PURE__ */ new Set(["error", "upstream_error", "runtime_error", "worker_failed", "generation_failed"])).has(code)) return code;
    const detail = [value.message, value.reason, value.detail, value.error].find((item) => typeof item === "string");
    if (detail) {
      const classified = classifyRunPodError(detail);
      if (classified !== "generation_failed") return classified;
    }
    return code;
  }
  const message = String(value || "").toLowerCase();
  if (/out[ _-]?of[ _-]?memory|\boom\b|cuda[^\n]{0,80}memory|memory allocation/.test(message)) return "out_of_memory";
  if (/(?:model|checkpoint|weights?|lora|vae)[^\n]{0,120}(?:load|missing|not found|failed)|(?:load|missing|not found|failed)[^\n]{0,120}(?:model|checkpoint|weights?|lora|vae)/.test(message)) return "model_load_failed";
  if (/(?:reference|media)[^\n]{0,120}(?:download|fetch|probe|preprocess|decode|invalid)|(?:ffmpeg|ffprobe)[^\n]{0,120}(?:failed|error)/.test(message)) return "reference_preprocessing_failed";
  if (/timed?[ _-]?out|timeout/.test(message)) return "timed_out";
  if (/(?:worker|container|process)[^\n]{0,120}(?:exit|crash|failed|terminated)|exit[ _-]?code/.test(message)) return "worker_process_failed";
  return "generation_failed";
}
__name(classifyRunPodError, "classifyRunPodError");
__name2(classifyRunPodError, "classifyRunPodError");
function sanitizeProgressError(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return { code: normalizedErrorCode(value.code || value.error_code || value.type || value.name) };
  }
  return { code: classifyRunPodError(value) };
}
__name(sanitizeProgressError, "sanitizeProgressError");
__name2(sanitizeProgressError, "sanitizeProgressError");
function errorCodeForLog(error) {
  return normalizedErrorCode(error?.code || error?.name || "internal_error");
}
__name(errorCodeForLog, "errorCodeForLog");
__name2(errorCodeForLog, "errorCodeForLog");
function sanitizePayload(value, key = "", depth = 0) {
  if (value === null || value === void 0) {
    return value;
  }
  if (depth > 12) {
    return "[omitted: nesting too deep]";
  }
  if (typeof value === "string") {
    if (isCredentialUrl(value)) {
      return "[redacted-url]";
    }
    if (shouldOmitBinaryString(key, value)) {
      return {
        omitted: true,
        reason: "binary/base64 payload removed",
        characterCount: value.length
      };
    }
    if (value.length > 1e4) {
      return value.slice(0, 4e3) + "...[truncated]";
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
      const normalized = String(childKey).toLowerCase().replace(/[^a-z0-9]/g, "");
      if (DOWNSTREAM_CONTENT_FIELDS.has(normalized)) {
        output[childKey] = REDACTED;
      } else if (["error", "errors", "message", "detail", "raw"].includes(normalized)) {
        output[childKey] = sanitizeProgressError(childValue);
      } else {
        output[childKey] = sanitizePayload(childValue, childKey, depth + 1);
      }
    }
    if (entries.length > limit) {
      output.__omittedKeys = entries.length - limit;
    }
    return output;
  }
  return String(value);
}
__name(sanitizePayload, "sanitizePayload");
__name2(sanitizePayload, "sanitizePayload");
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
  return value.length > 1e5 && looksLikeBase64(value);
}
__name(shouldOmitBinaryString, "shouldOmitBinaryString");
__name2(shouldOmitBinaryString, "shouldOmitBinaryString");
function isHttpUrl(value) {
  return /^https?:\/\//i.test(String(value).trim());
}
__name(isHttpUrl, "isHttpUrl");
__name2(isHttpUrl, "isHttpUrl");
function isCredentialUrl(value) {
  if (!isHttpUrl(value)) return false;
  try {
    const url = new URL(String(value));
    const names = [...url.searchParams.keys()].map((name) => name.toLowerCase());
    return names.some((name) => name.startsWith("x-amz-") || ["token", "sig", "signature", "key"].includes(name));
  } catch {
    return true;
  }
}
__name(isCredentialUrl, "isCredentialUrl");
__name2(isCredentialUrl, "isCredentialUrl");
function looksLikeBase64(value) {
  const sample = String(value).slice(0, 4096).replace(/\s/g, "");
  return sample.length >= 1024 && /^[A-Za-z0-9+/=]+$/.test(sample);
}
__name(looksLikeBase64, "looksLikeBase64");
__name2(looksLikeBase64, "looksLikeBase64");
function toInteger(value, fallback) {
  if (value === void 0 || value === null || value === "") {
    return fallback;
  }
  const number = Number(value);
  return Number.isFinite(number) ? Math.trunc(number) : fallback;
}
__name(toInteger, "toInteger");
__name2(toInteger, "toInteger");
function toNumber(value, fallback) {
  if (value === void 0 || value === null || value === "") {
    return fallback;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
__name(toNumber, "toNumber");
__name2(toNumber, "toNumber");
async function safeJson(response) {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text.slice(0, 2e3) };
  }
}
__name(safeJson, "safeJson");
__name2(safeJson, "safeJson");
function json(data, status, headers) {
  return new Response(JSON.stringify(data, null, 2), { status, headers });
}
__name(json, "json");
__name2(json, "json");
async function uploadAsset(request, env, headers, jobId, url) {
  const bucket = requireBucket(env);
  const contentType = (request.headers.get("Content-Type") || "").split(";")[0].trim();
  const extension = ASSET_CONTENT_TYPES[contentType];
  if (!extension) {
    throw new HttpError(
      415,
      `Unsupported Content-Type '${contentType || "(none)"}'. Allowed: ` + Object.keys(ASSET_CONTENT_TYPES).join(", ")
    );
  }
  const declared = Number(request.headers.get("Content-Length") || 0);
  if (declared > MAX_ASSET_BYTES) {
    throw new HttpError(413, `Asset exceeds the ${MAX_ASSET_BYTES}-byte limit`);
  }
  const assetId = url.searchParams.get("id") || "first-frame";
  const key = inputKey(jobId, assetId, extension);
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
__name(uploadAsset, "uploadAsset");
__name2(uploadAsset, "uploadAsset");
async function streamVideo(request, env, jobId) {
  const bucket = requireBucket(env);
  const range = request.headers.get("Range");
  const options = range ? { range: request.headers } : void 0;
  let object = null;
  let encrypted = false;
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
  if (stored.encrypted !== void 0) {
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
    responseHeaders.set("Cache-Control", "private, no-store");
    responseHeaders.set(
      "Access-Control-Expose-Headers",
      "Content-Length, Content-Range, ETag, X-Privacy-Mode, X-Artifact-Encrypted, X-Artifact-Encryption-Version, X-Artifact-Algorithm, X-Artifact-Original-Content-Type"
    );
  } else {
    responseHeaders.set("Content-Type", "video/mp4");
    responseHeaders.set("X-Artifact-Encrypted", "false");
    responseHeaders.set("Cache-Control", "private, max-age=3600, immutable");
    responseHeaders.set("Access-Control-Expose-Headers", "Content-Length, Content-Range, ETag");
  }
  if (range && object.range && object.size !== void 0) {
    const offset = object.range.offset ?? 0;
    const length = object.range.length ?? object.size - offset;
    const end = offset + length - 1;
    responseHeaders.set("Content-Range", `bytes ${offset}-${end}/${object.size}`);
    return new Response(object.body, { status: 206, headers: responseHeaders });
  }
  return new Response(object.body, { status: 200, headers: responseHeaders });
}
__name(streamVideo, "streamVideo");
__name2(streamVideo, "streamVideo");
async function deleteGeneration(env, headers, jobId, scope = "output") {
  const bucket = requireBucket(env);
  const segment = safeJobSegment(jobId);
  const prefixes = scope === "generation" ? [`outputs/${segment}/`, `inputs/${segment}/`] : [`outputs/${segment}/`];
  const found = /* @__PURE__ */ new Set();
  for (const prefix of prefixes) {
    for (const key of await listAllKeys(bucket, prefix)) {
      found.add(key);
    }
  }
  const doomed = new Set(found);
  for (const key of outputKeyCandidates(jobId)) doomed.add(key);
  if (scope === "generation") {
    const state = await readJobState(env, jobId);
    for (const key of state?.referenceObjectKeys || []) doomed.add(validateObjectKey(key));
  }
  const all = [...doomed];
  if (all.length) {
    await bucket.delete(all.length === 1 ? all[0] : all);
  }
  try {
    await pushJobState(env, jobId, {
      video: { key: outputKey(jobId), deleted: true },
      artifact: { deleted: true }
    });
  } catch (error) {
    console.warn(`Deleted ${found.size} object(s) but could not update the channel for job ${segment}: ${errorCodeForLog(error)}`);
  }
  return json({ id: jobId, deleted: true, scope, removed: found.size }, 200, headers);
}
__name(deleteGeneration, "deleteGeneration");
__name2(deleteGeneration, "deleteGeneration");
async function listAllKeys(bucket, prefix) {
  if (typeof bucket.list !== "function") {
    throw new HttpError(500, "R2 binding does not support list(); cannot delete safely");
  }
  const keys = [];
  let cursor;
  do {
    const page = await bucket.list({ prefix, cursor, limit: 1e3 });
    for (const object of page?.objects || []) keys.push(object.key);
    cursor = page?.truncated ? page.cursor : void 0;
  } while (cursor);
  return keys;
}
__name(listAllKeys, "listAllKeys");
__name2(listAllKeys, "listAllKeys");
async function openJobSocket(request, env, jobId) {
  const stub = jobChannel(env, jobId);
  return stub.fetch(new Request("https://do/ws", { headers: request.headers }));
}
__name(openJobSocket, "openJobSocket");
__name2(openJobSocket, "openJobSocket");
async function receiveProgress(request, env, headers, jobId) {
  await verifyJobToken(env.JOB_TOKEN_SECRET, bearerToken(request), jobId, TOKEN_PURPOSES.progress);
  const body = await readJsonObject(request);
  const update = {};
  for (const field of ["phase", "step", "steps", "percent"]) {
    if (body[field] !== void 0) update[field] = body[field];
  }
  if (body.error !== void 0) update.error = sanitizeProgressError(body.error);
  if (["completed", "failed", "cancelled"].includes(update.phase)) {
    update.status = update.phase === "completed" ? "COMPLETED" : update.phase === "cancelled" ? "CANCELLED" : "FAILED";
  }
  if (body.video && typeof body.video === "object") {
    update.video = { key: outputKey(jobId), size: body.video.size };
  }
  await pushJobState(env, jobId, update);
  if (["completed", "failed", "cancelled"].includes(update.phase)) {
    const state = await readJobState(env, jobId);
    if (state?.referenceObjectKeys?.length) {
      await cleanupReferenceObjects(env, state.referenceObjectKeys);
      await pushJobState(env, jobId, { referenceObjectKeys: [], referencesDeletedAt: (/* @__PURE__ */ new Date()).toISOString() });
    }
  }
  return json({ ok: true }, 200, headers);
}
__name(receiveProgress, "receiveProgress");
__name2(receiveProgress, "receiveProgress");
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
  const declaredSize = Number(request.headers.get("Content-Length") || 0);
  if (spec.encrypts) {
    if (!(declaredSize > 0)) {
      throw new HttpError(
        411,
        "A confidential artefact upload must declare Content-Length: the container header has to be read before the body can be stored, and R2 needs a known length."
      );
    }
    const peeked = await peekStream(body, CONTAINER_PREAMBLE_BYTES + CONTAINER_MAX_HEADER_BYTES);
    body = withKnownLength(peeked.stream, declaredSize);
    let container;
    try {
      container = parseContainerPrefix(peeked.prefix);
    } catch (error) {
      throw new HttpError(
        422,
        `Refusing to store this artefact: job ${jobId} is confidential, and the uploaded bytes are not an encrypted container (${error.message}). Nothing was written.`
      );
    }
    const requiredVersion = Number.isInteger(claims.cv) ? claims.cv : container.version;
    if (container.version !== requiredVersion) {
      throw new HttpError(
        422,
        `Refusing to store this artefact: job ${jobId} was created with confidential crypto version ${requiredVersion}, but the uploaded container is version ${container.version}. Nothing was written.`
      );
    }
    if (container.header.artifactId && container.header.artifactId !== jobId) {
      throw new HttpError(
        422,
        `Refusing to store this artefact: its authenticated header names generation '${container.header.artifactId}', not '${jobId}'.`
      );
    }
    artifact = {
      privacyMode: mode,
      encrypted: true,
      cryptoVersion: container.version,
      // Kept under its original name as well, so existing clients reading
      // `encryptionVersion` are unaffected.
      encryptionVersion: container.version,
      algorithm: container.algorithm,
      contentType: ENCRYPTED_CONTENT_TYPE,
      originalContentType: container.header.contentType || "application/octet-stream",
      ...container.keyWrap ? { keyWrapAlgorithm: container.keyWrap.alg, keyId: container.keyWrap.keyId } : {},
      // v1 artefacts carry their KDF and key id at the top of the header.
      ...container.header.kdf ? { kdf: container.header.kdf } : {},
      ...!container.keyWrap && container.header.keyId ? { keyId: container.header.keyId } : {}
    };
  }
  const key = outputKey(jobId, artifact.encrypted);
  const expiresAt = typeof claims.exa === "string" ? claims.exa : void 0;
  const object = await bucket.put(key, body, {
    httpMetadata: { contentType: artifact.contentType },
    // Non-secret, and the reason an object recovered on its own is still identifiable.
    // No key material appears here - there is no branch of this function that could put
    // it here, because the Worker never holds the key at upload time.
    customMetadata: {
      privacyMode: artifact.privacyMode,
      encrypted: String(artifact.encrypted),
      ...artifact.encrypted ? {
        cryptoVersion: String(artifact.cryptoVersion),
        encryptionVersion: String(artifact.encryptionVersion),
        algorithm: artifact.algorithm,
        originalContentType: artifact.originalContentType,
        // Non-secret by construction: an algorithm name and a key's public
        // fingerprint. Neither helps anyone decrypt anything.
        ...artifact.keyWrapAlgorithm ? { keyWrapAlgorithm: artifact.keyWrapAlgorithm } : {},
        ...artifact.keyId ? { keyId: artifact.keyId } : {}
      } : {},
      ...expiresAt ? { expiresAt } : {}
    }
  });
  const size = object?.size ?? (declaredSize > 0 ? declaredSize : void 0);
  await pushJobState(env, jobId, {
    privacyMode: artifact.privacyMode,
    video: { key, size },
    artifact: { ...artifact, key, size, deleted: false, ...expiresAt ? { expiresAt } : {} }
  });
  const state = await readJobState(env, jobId);
  if (state?.referenceObjectKeys?.length) {
    await cleanupReferenceObjects(env, state.referenceObjectKeys);
    await pushJobState(env, jobId, { referenceObjectKeys: [], referencesDeletedAt: (/* @__PURE__ */ new Date()).toISOString() });
  }
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
__name(receiveOutput, "receiveOutput");
__name2(receiveOutput, "receiveOutput");
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
__name(peekStream, "peekStream");
__name2(peekStream, "peekStream");
function withKnownLength(stream, length) {
  if (typeof FixedLengthStream !== "function") {
    return stream;
  }
  const fixed = new FixedLengthStream(length);
  stream.pipeTo(fixed.writable).catch(() => {
  });
  return fixed.readable;
}
__name(withKnownLength, "withKnownLength");
__name2(withKnownLength, "withKnownLength");
function parseContainerPrefix(data) {
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
  if (!SUPPORTED_CONTAINER_VERSIONS.has(version)) {
    throw new Error(`unsupported container version ${version}`);
  }
  if (!CONTAINER_SUITES[suite]) {
    throw new Error(`unsupported cipher suite ${suite}`);
  }
  const headerLength = data[6] << 8 | data[7];
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
  let keyWrap = null;
  if (version === CONTAINER_V2_HYBRID) {
    keyWrap = header.kw;
    if (!keyWrap || typeof keyWrap !== "object" || Array.isArray(keyWrap)) {
      throw new Error("a v2 container must carry a 'kw' key-wrapping block");
    }
    if (!SUPPORTED_KEY_WRAP_ALGORITHMS.includes(keyWrap.alg)) {
      throw new Error(`unsupported key-wrap algorithm ${JSON.stringify(keyWrap.alg)}`);
    }
    if (typeof keyWrap.wrappedFileKey !== "string" || !keyWrap.wrappedFileKey) {
      throw new Error("a v2 container must carry a wrapped file key");
    }
    if (typeof keyWrap.keyId !== "string" || !keyWrap.keyId) {
      throw new Error("a v2 container must name the key it was wrapped to");
    }
  }
  return {
    version,
    suite,
    algorithm: CONTAINER_SUITES[suite],
    header,
    keyWrap,
    headerLength,
    nonceOffset: headerEnd,
    ciphertextOffset: headerEnd + CONTAINER_NONCE_BYTES
  };
}
__name(parseContainerPrefix, "parseContainerPrefix");
__name2(parseContainerPrefix, "parseContainerPrefix");
async function serveAsset(request, env, jobId, assetId) {
  await verifyJobToken(env.JOB_TOKEN_SECRET, bearerToken(request), jobId, TOKEN_PURPOSES.asset);
  const bucket = requireBucket(env);
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
__name(serveAsset, "serveAsset");
__name2(serveAsset, "serveAsset");
var TOKEN_PURPOSES = Object.freeze({
  progress: "progress",
  output: "output-upload",
  asset: "asset-download"
});
var TOKEN_DEFAULT_TTL_SECONDS = 3600;
function b64urlEncode(bytes) {
  let binary = "";
  for (const byte of new Uint8Array(bytes)) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
__name(b64urlEncode, "b64urlEncode");
__name2(b64urlEncode, "b64urlEncode");
function b64urlDecodeToBytes(value) {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded + "=".repeat((4 - padded.length % 4) % 4));
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}
__name(b64urlDecodeToBytes, "b64urlDecodeToBytes");
__name2(b64urlDecodeToBytes, "b64urlDecodeToBytes");
async function hmacKey(secret) {
  return crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
}
__name(hmacKey, "hmacKey");
__name2(hmacKey, "hmacKey");
async function signJobToken(secret, jobId, purpose, ttlSeconds = TOKEN_DEFAULT_TTL_SECONDS, claims = null) {
  if (!secret) {
    throw new HttpError(500, "Missing Cloudflare secret: JOB_TOKEN_SECRET");
  }
  const payload = {
    jid: jobId,
    pur: purpose,
    exp: Math.floor(Date.now() / 1e3) + ttlSeconds,
    ...claims || {}
  };
  const body = b64urlEncode(new TextEncoder().encode(JSON.stringify(payload)));
  const signature = await crypto.subtle.sign("HMAC", await hmacKey(secret), new TextEncoder().encode(body));
  return `${body}.${b64urlEncode(signature)}`;
}
__name(signJobToken, "signJobToken");
__name2(signJobToken, "signJobToken");
async function verifyJobToken(secret, token, jobId, purpose) {
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
  if (payload.exp && Math.floor(Date.now() / 1e3) > payload.exp) {
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
__name(verifyJobToken, "verifyJobToken");
__name2(verifyJobToken, "verifyJobToken");
function bearerToken(request) {
  const header = request.headers.get("Authorization") || "";
  return header.startsWith("Bearer ") ? header.slice(7).trim() : "";
}
__name(bearerToken, "bearerToken");
__name2(bearerToken, "bearerToken");
function outputKey(jobId, encrypted = false) {
  return `outputs/${safeJobSegment(jobId)}/${encrypted ? "artifact.enc" : "video.mp4"}`;
}
__name(outputKey, "outputKey");
__name2(outputKey, "outputKey");
function outputKeyCandidates(jobId) {
  return [outputKey(jobId, false), outputKey(jobId, true)];
}
__name(outputKeyCandidates, "outputKeyCandidates");
__name2(outputKeyCandidates, "outputKeyCandidates");
function inputKey(jobId, assetId, extension) {
  return `inputs/${safeJobSegment(jobId)}/${safeAssetId(assetId)}${extension}`;
}
__name(inputKey, "inputKey");
__name2(inputKey, "inputKey");
function safeJobSegment(jobId) {
  const value = String(jobId || "").trim();
  if (!value || !/^[A-Za-z0-9._-]+$/.test(value) || value.length > 256) {
    throw new HttpError(400, "Invalid job id");
  }
  return value;
}
__name(safeJobSegment, "safeJobSegment");
__name2(safeJobSegment, "safeJobSegment");
function safeAssetId(assetId) {
  const value = String(assetId || "").trim();
  if (!value || !/^[A-Za-z0-9._-]+$/.test(value) || value.length > 128) {
    throw new HttpError(400, "Invalid asset id");
  }
  return value;
}
__name(safeAssetId, "safeAssetId");
__name2(safeAssetId, "safeAssetId");
var ASSET_CONTENT_TYPES = Object.freeze({
  "image/png": ".png",
  "image/jpeg": ".jpg",
  "image/webp": ".webp"
});
var MAX_ASSET_BYTES = 32 * 1024 * 1024;
var JobChannel = class {
  static {
    __name(this, "JobChannel");
  }
  static {
    __name2(this, "JobChannel");
  }
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }
  async currentState() {
    return await this.state.storage.get("state") || null;
  }
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname.endsWith("/ws")) {
      if (request.headers.get("Upgrade") !== "websocket") {
        return new Response("Expected websocket upgrade", { status: 426 });
      }
      const pair = new WebSocketPair();
      const [client, server] = Object.values(pair);
      this.state.acceptWebSocket(server);
      const snapshot = await this.currentState();
      if (snapshot) {
        try {
          server.send(JSON.stringify({ type: "state", ...snapshot }));
        } catch {
        }
      }
      return new Response(null, { status: 101, webSocket: client });
    }
    if (url.pathname.endsWith("/update") && request.method === "POST") {
      const update = await request.json();
      const previous = await this.currentState() || {};
      const next = {
        ...previous,
        ...update,
        jobId: update.jobId ?? previous.jobId,
        createdAt: previous.createdAt ?? (/* @__PURE__ */ new Date()).toISOString(),
        updatedAt: (/* @__PURE__ */ new Date()).toISOString()
      };
      await this.state.storage.put("state", next);
      this.broadcast(next);
      return new Response(null, { status: 204 });
    }
    if (url.pathname.endsWith("/state")) {
      return new Response(JSON.stringify(await this.currentState() || {}), {
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
      }
    }
  }
  // Clients are listeners, not commanders: nothing they send can change job state.
  async webSocketMessage() {
  }
  async webSocketClose(socket) {
    try {
      socket.close();
    } catch {
    }
  }
};
function publicEvent(state) {
  const phase = state.phase || "queued";
  if (phase === "completed") {
    let video;
    if (state.video?.deleted) {
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
  if (phase === "sampling") {
    if (state.step !== void 0) event.step = state.step;
    if (state.steps !== void 0) event.steps = state.steps;
  }
  if (state.percent !== void 0) event.percent = state.percent;
  return event;
}
__name(publicEvent, "publicEvent");
__name2(publicEvent, "publicEvent");
function jobChannel(env, jobId) {
  if (!env.JOB_CHANNEL) {
    throw new HttpError(500, "Missing Durable Object binding: JOB_CHANNEL");
  }
  return env.JOB_CHANNEL.get(env.JOB_CHANNEL.idFromName(safeJobSegment(jobId)));
}
__name(jobChannel, "jobChannel");
__name2(jobChannel, "jobChannel");
async function readJobState(env, jobId) {
  try {
    const response = await jobChannel(env, jobId).fetch(new Request("https://do/state"));
    const state = await response.json();
    return state && typeof state === "object" ? state : null;
  } catch {
    return null;
  }
}
__name(readJobState, "readJobState");
__name2(readJobState, "readJobState");
async function pushJobState(env, jobId, update) {
  const stub = jobChannel(env, jobId);
  await stub.fetch(new Request("https://do/update", {
    method: "POST",
    body: JSON.stringify({ jobId, ...update }),
    headers: { "Content-Type": "application/json" }
  }));
}
__name(pushJobState, "pushJobState");
__name2(pushJobState, "pushJobState");
function requireBucket(env) {
  if (!env.H3_OUTPUTS) {
    throw new HttpError(500, "Missing R2 binding: H3_OUTPUTS");
  }
  return env.H3_OUTPUTS;
}
__name(requireBucket, "requireBucket");
__name2(requireBucket, "requireBucket");
function workerConstants() {
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
__name(workerConstants, "workerConstants");
__name2(workerConstants, "workerConstants");
export {
  HttpError,
  JobChannel,
  adaptCanvas,
  alignFrameCount,
  applySettings,
  buildPrompt,
  buildWorkflowForSettings,
  classifyRunPodError,
  worker_default as default,
  describeArtifactForStatus,
  describeCapabilities,
  describeProgressForStatus,
  durationToFrames,
  framesToDuration,
  inputKey,
  isLegalFrameCount,
  loadWorkflowTemplate,
  normalizeAsset,
  normalizePrivacy,
  normalizeRequest,
  normalizeRunPodJobStatus,
  outputKey,
  outputKeyCandidates,
  parseContainerPrefix,
  publicEvent,
  redactSecrets,
  rejectSecretFields,
  resolveMode,
  sanitizeProgressError,
  sanitizeRunPodResult,
  signJobToken,
  validateR2Key,
  verifyJobToken,
  workerConstants
};
//# sourceMappingURL=worker.js.map
