#!/bin/sh
# Apply an R2 retention policy to the H3 output bucket.
#
# NOT run automatically, and deliberately not baked into any deploy: how long a generated
# video should live is a product decision with user-visible consequences, not something to
# infer from a repository. Read the numbers below, change them if they are wrong for your
# product, then run this once.
#
#   sh scripts/set_r2_lifecycle.sh          # show what would be applied
#   sh scripts/set_r2_lifecycle.sh --apply  # actually apply it
#
# Requires an authenticated wrangler (`npx wrangler login`).
#
# This is separate from DELETE /jobs/:id/video. That endpoint handles deliberate removal;
# this handles outputs nobody ever came back for.

set -e

ACCOUNT=899c4111fb42559764b1bd1118d8cf79
BUCKET=minimax-h3-private-output
CFG="${WRANGLER_CONFIG:-$HOME/.wrangler/config/default.toml}"

# --- the policy ------------------------------------------------------------------------
#
# inputs/  7 days. Keyframes are only needed while the job that references them runs, which
#          is under two minutes. A week is already generous and exists purely so a support
#          question can still be answered.
#
# outputs/ 30 days. Conservative on purpose: long enough that a user who bookmarked a link
#          is unlikely to be surprised, short enough that storage does not grow without
#          bound. At ~2 MB per clip, 30 days of 100 videos/day is roughly 6 GB.
#
# Both are prefix-scoped, so nothing outside these two namespaces is ever touched.
INPUT_DAYS=7
OUTPUT_DAYS=30

PAYLOAD=$(cat <<JSON
{"rules":[
  {"id":"Default Multipart Abort Rule","enabled":true,"conditions":{},
   "abortMultipartUploadsTransition":{"condition":{"type":"Age","maxAge":604800}}},
  {"id":"expire-inputs","enabled":true,
   "conditions":{"prefix":"inputs/"},
   "deleteObjectsTransition":{"condition":{"type":"Age","maxAge":$((INPUT_DAYS * 86400))}}},
  {"id":"expire-outputs","enabled":true,
   "conditions":{"prefix":"outputs/"},
   "deleteObjectsTransition":{"condition":{"type":"Age","maxAge":$((OUTPUT_DAYS * 86400))}}}
]}
JSON
)

echo "Bucket:  $BUCKET"
echo "inputs/  expire after ${INPUT_DAYS} days"
echo "outputs/ expire after ${OUTPUT_DAYS} days"
echo
echo "The existing multipart-abort rule is restated because this API replaces the whole"
echo "rule set; omitting it would silently drop it."
echo

if [ "$1" != "--apply" ]; then
  echo "Dry run. Re-run with --apply to write this policy."
  exit 0
fi

TOKEN=$(sed -n 's/^oauth_token *= *"\([^"]*\)".*/\1/p' "$CFG")
if [ -z "$TOKEN" ]; then
  echo "No wrangler OAuth token found at $CFG - run 'npx wrangler login' first." >&2
  exit 1
fi

curl -s -X PUT \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  --data "$PAYLOAD" \
  "https://api.cloudflare.com/client/v4/accounts/${ACCOUNT}/r2/buckets/${BUCKET}/lifecycle" \
  | python -c "
import json,sys
d=json.load(sys.stdin)
print('applied' if d.get('success') else 'FAILED')
for e in (d.get('errors') or []):
    print(' ', e.get('code'), '-', e.get('message'))
"
