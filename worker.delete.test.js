/*
 * Tests for DELETE /jobs/:id/video, driven through the real fetch handler.
 *
 * Run with:  node --test
 *
 * The property that matters most is scoping: the key is derived from the job id inside the
 * Worker, so there is no request a caller can construct that names an arbitrary R2 object.
 * The second is idempotence - a retry after a dropped connection must not read as an error.
 */

import test from "node:test";
import assert from "node:assert/strict";

import worker, { outputKey, outputKeyCandidates, publicEvent } from "./worker.js";

const JOB = "08ad4ace-d847-46b4-a1d6-a919d7e3a0c9";

/** Env with an R2 stub that records every delete, and a Durable Object that keeps state. */
function makeEnv({ withVideo = true } = {}) {
  const bucket = new Map();
  const deletes = [];
  let state = withVideo
    ? { jobId: JOB, phase: "completed", video: { key: outputKey(JOB), size: 2196233 } }
    : { jobId: JOB, phase: "completed" };

  if (withVideo) {
    bucket.set(outputKey(JOB), Buffer.alloc(2196233));
  }

  return {
    JOB_TOKEN_SECRET: "test-secret",
    RUNPOD_ENDPOINT_ID: "e-ada",
    RUNPOD_BLACKWELL_ENDPOINT_ID: "e-bw",
    RUNPOD_API_KEY: "k",

    JOB_CHANNEL: {
      idFromName: (name) => ({ name }),
      get: () => ({
        fetch: async (request) => {
          const path = new URL(request.url).pathname;
          if (path.endsWith("/update")) {
            const update = await request.json();
            state = { ...state, ...update };
            return new Response(null, { status: 204 });
          }
          return new Response(JSON.stringify(state), {
            headers: { "Content-Type": "application/json" }
          });
        }
      })
    },

    H3_OUTPUTS: {
      delete: async (key) => {
        // R2's delete takes a key or an array of them, and does not error on keys that are
        // already gone. The stub models both, because deletion now names several
        // candidates at once and idempotence depends on that being tolerated.
        for (const one of Array.isArray(key) ? key : [key]) {
          deletes.push(one);
          bucket.delete(one);
        }
      },
      list: async ({ prefix = "", cursor, limit = 1000 } = {}) => {
        const all = [...bucket.keys()].filter((key) => key.startsWith(prefix)).sort();
        const start = cursor ? Number(cursor) : 0;
        const page = all.slice(start, start + limit);
        const next = start + page.length;
        return {
          objects: page.map((key) => ({ key })),
          truncated: next < all.length,
          cursor: next < all.length ? String(next) : undefined
        };
      },
      get: async (key) => {
        if (!bucket.has(key)) return null;
        const body = bucket.get(key);
        return {
          body,
          size: body.length,
          httpEtag: '"e"',
          writeHttpMetadata() {}
        };
      },
      put: async (key, body) => {
        bucket.set(key, body);
        return { size: 1 };
      }
    },

    __deletes: deletes,
    __bucket: bucket,
    __state: () => state
  };
}

function del(jobId) {
  return new Request(`https://worker.example/jobs/${jobId}/video`, { method: "DELETE" });
}

function get(jobId, headers = {}) {
  return new Request(`https://worker.example/jobs/${jobId}/video`, { headers });
}

/* -- happy path ---------------------------------------------------------------------- */

test("deleting a job's video removes exactly that object", async () => {
  const env = makeEnv();
  const response = await worker.fetch(del(JOB), env);
  const body = await response.json();

  assert.equal(response.status, 200);
  // Subset rather than deepEqual: the response is additive (scope, removed) and a strict
  // shape assertion here would fail every time a non-breaking field is added.
  assert.equal(body.id, JOB);
  assert.equal(body.deleted, true);
  assert.equal(body.removed, 1, "one object actually existed");
  // Both possible artefact names are deleted - standard and confidential - and every one
  // of them is derived from the job id. That scoping is the property under test.
  assert.deepEqual(env.__deletes.sort(), outputKeyCandidates(JOB).sort());
  assert.equal(env.__bucket.has(outputKey(JOB)), false);
});

test("deletion is recorded so status stops advertising a URL", async () => {
  const env = makeEnv();
  await worker.fetch(del(JOB), env);
  assert.equal(env.__state().video.deleted, true);
});

/* -- idempotence --------------------------------------------------------------------- */

test("repeated deletion succeeds identically", async () => {
  const env = makeEnv();

  const first = await worker.fetch(del(JOB), env);
  const second = await worker.fetch(del(JOB), env);
  const third = await worker.fetch(del(JOB), env);

  for (const response of [first, second, third]) {
    assert.equal(response.status, 200);
  }
  const body = await third.json();
  assert.equal(body.id, JOB);
  assert.equal(body.deleted, true);
  assert.equal(body.removed, 0, "nothing was left to remove by the third call");
});

test("deleting a job that never had a video still succeeds", async () => {
  const env = makeEnv({ withVideo: false });
  const response = await worker.fetch(del(JOB), env);

  assert.equal(response.status, 200, "a missing object is not an error condition");
  const body = await response.json();
  assert.equal(body.id, JOB);
  assert.equal(body.deleted, true);
  assert.equal(body.removed, 0);
});

test("deleting an entirely unknown job id succeeds without touching anything else", async () => {
  const env = makeEnv();
  const other = "11111111-2222-3333-4444-555555555555";

  const response = await worker.fetch(del(other), env);
  assert.equal(response.status, 200);
  assert.deepEqual(env.__deletes.sort(), outputKeyCandidates(other).sort());
  assert.equal(env.__bucket.has(outputKey(JOB)), true, "the real job's video is untouched");
});

/* -- scoping: the whole security property --------------------------------------------- */

test("a job id that could escape its namespace is rejected before R2 is touched", async () => {
  for (const bad of ["../evil", "a/b", "with space", "x".repeat(300)]) {
    const env = makeEnv();
    const response = await worker.fetch(
      new Request(`https://worker.example/jobs/${encodeURIComponent(bad)}/video`, {
        method: "DELETE"
      }),
      env
    );
    assert.equal(response.status, 400, `expected 400 for ${bad}`);
    assert.deepEqual(env.__deletes, [], "nothing may be deleted for a malformed id");
  }
});

test("deleting one job cannot reach another job's object", async () => {
  const env = makeEnv();
  const other = "99999999-8888-7777-6666-555555555555";
  env.__bucket.set(outputKey(other), Buffer.alloc(10));

  await worker.fetch(del(JOB), env);

  assert.equal(env.__bucket.has(outputKey(other)), true);
  // Every key named is inside this job's own namespace; nothing can address another's.
  assert.ok(env.__deletes.every((key) => key.startsWith(`outputs/${JOB}/`)), env.__deletes.join());
});

test("there is no route that deletes an arbitrary R2 key", async () => {
  const env = makeEnv();
  for (const path of [
    "/r2/outputs/other/video.mp4",
    "/objects/outputs/other/video.mp4",
    "/internal/objects/outputs/other/video.mp4"
  ]) {
    const response = await worker.fetch(
      new Request(`https://worker.example${path}`, { method: "DELETE" }),
      env
    );
    assert.equal(response.status, 404, `${path} must not exist`);
  }
  assert.deepEqual(env.__deletes, []);
});

/* -- retrieval after deletion --------------------------------------------------------- */

test("GET video after deletion is a 404, not a truncated stream", async () => {
  const env = makeEnv();
  await worker.fetch(del(JOB), env);

  const response = await worker.fetch(get(JOB), env);
  assert.equal(response.status, 404);
});

test("a Range request after deletion is also a 404", async () => {
  const env = makeEnv();
  await worker.fetch(del(JOB), env);

  const response = await worker.fetch(get(JOB, { Range: "bytes=0-1023" }), env);
  assert.equal(response.status, 404, "a ranged read must not resurrect a deleted object");
});

test("GET video before deletion still streams", async () => {
  const env = makeEnv();
  const response = await worker.fetch(get(JOB), env);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Content-Type"), "video/mp4");
});

/* -- status semantics ----------------------------------------------------------------- */

test("a completed job stays completed after its output is deleted", () => {
  const before = publicEvent({
    jobId: JOB,
    phase: "completed",
    video: { key: outputKey(JOB) }
  });
  assert.equal(before.type, "completed");
  assert.equal(before.video.url, `/jobs/${JOB}/video`);
  assert.equal(before.video.deleted, false);

  const after = publicEvent({
    jobId: JOB,
    phase: "completed",
    video: { key: outputKey(JOB), deleted: true }
  });
  assert.equal(after.type, "completed", "deleting an artefact is not a failed generation");
  assert.deepEqual(after.video, { deleted: true });
  assert.equal(after.video.url, undefined, "no playable URL may be advertised");
});

test("the completed event never leaks a bucket URL", () => {
  const event = publicEvent({ jobId: JOB, phase: "completed", video: { key: outputKey(JOB) } });
  assert.ok(!JSON.stringify(event).includes("r2.cloudflarestorage"));
});
