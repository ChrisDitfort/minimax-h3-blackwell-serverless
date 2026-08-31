/*
 * Local-only helper: mint job-scoped tokens for exercising the internal routes against
 * `wrangler dev`. Reads the secret from argv so nothing is hardcoded.
 *
 *   node mint-tokens.mjs <secret> <jobId>
 *
 * Prints, one per line: progress token, output token, asset token.
 */
import { signJobToken, workerConstants } from "./worker.js";

const [secret, jobId] = process.argv.slice(2);
if (!secret || !jobId) {
  console.error("usage: node mint-tokens.mjs <secret> <jobId>");
  process.exit(1);
}

const { TOKEN_PURPOSES } = workerConstants();
for (const purpose of [TOKEN_PURPOSES.progress, TOKEN_PURPOSES.output, TOKEN_PURPOSES.asset]) {
  console.log(await signJobToken(secret, jobId, purpose));
}
