#!/usr/bin/env bash
# Run a test command; if it fails, re-emit its failure lines as GitHub error
# annotations before failing the step.
#
# Workflow logs on GitHub need a signed-in account to read, even for a public
# repository, but annotations are returned by the public API. Turning each
# failure into an annotation means anyone -- or any tool -- can see *why* CI
# failed without signing in.
#
#   .github/annotate.sh python tests/test_visualiser.py
set -o pipefail
log=$(mktemp)
"$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    # the lines that say what failed: our FAIL markers, ctest's failed-test
    # list, assertion and Python errors -- capped so a flood stays readable
    grep -E 'FAIL|\(Failed\)|Failed|Assertion|Error|error:|Traceback|max abs err|rel\. to scale' "$log" \
        | grep -vE '^ *[0-9]+ (checks|comparisons|tensors)' \
        | head -40 \
        | while IFS= read -r line; do
              clean=$(printf '%s' "$line" | sed 's/%/%25/g' | cut -c1-300)
              echo "::error title=$1 $2::$clean"
          done
fi
rm -f "$log"
exit "$status"
