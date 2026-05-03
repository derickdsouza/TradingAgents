#!/usr/bin/env bash
# Check upstream/main for new activity and refresh .claude/upstream-status.json.
#
# Usage: .claude/check-upstream.sh           (just runs the check)
#        .claude/check-upstream.sh --quiet   (no stdout, only exit code)
#
# Exit codes:
#   0 = check ran successfully
#   1 = git fetch failed (network / auth)
#   2 = repo not in expected state (no upstream remote, no local-patches branch)
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "error: not inside a git repo" >&2
    exit 2
}
cd "$REPO_ROOT"

QUIET=0
[[ "${1-}" == "--quiet" ]] && QUIET=1
say() { [[ $QUIET -eq 0 ]] && echo "$@"; return 0; }

# Sanity checks
git remote get-url upstream >/dev/null 2>&1 || {
    echo "error: 'upstream' remote not configured. Run: git remote add upstream https://github.com/TauricResearch/TradingAgents.git" >&2
    exit 2
}
git rev-parse --verify local-patches >/dev/null 2>&1 || {
    echo "error: 'local-patches' branch missing" >&2
    exit 2
}

PRE_FETCH_SHA=$(git rev-parse upstream/main 2>/dev/null || echo "none")

say "Fetching upstream..."
git fetch upstream --quiet || { echo "error: git fetch upstream failed" >&2; exit 1; }

POST_FETCH_SHA=$(git rev-parse upstream/main)
LOCAL_UNIQUE_COUNT=$(git rev-list --count "${POST_FETCH_SHA}...local-patches" --right-only 2>/dev/null || echo 0)
UPSTREAM_AHEAD_COUNT=$(git rev-list --count "${POST_FETCH_SHA}...local-patches" --left-only 2>/dev/null || echo 0)

# git cherry: lines starting with - are upstream-equivalent (safe to drop on rebase)
CHERRY_OUT=$(git cherry -v upstream/main local-patches || true)
DROPPABLE=$(echo "$CHERRY_OUT" | awk '/^-/ {print $2}' | tr '\n' ',' | sed 's/,$//')
KEEP=$(echo "$CHERRY_OUT" | awk '/^\+/ {print $2}' | tr '\n' ',' | sed 's/,$//')
DROPPABLE_COUNT=$(echo "$CHERRY_OUT" | awk '/^-/' | wc -l | tr -d ' ')

NEW_UPSTREAM_LOG=""
if [[ "$PRE_FETCH_SHA" != "$POST_FETCH_SHA" && "$PRE_FETCH_SHA" != "none" ]]; then
    NEW_UPSTREAM_LOG=$(git log --oneline "${PRE_FETCH_SHA}..${POST_FETCH_SHA}" | sed 's/"/\\"/g' | tr '\n' '|' | sed 's/|$//')
fi

NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
STATUS_FILE="$REPO_ROOT/.claude/upstream-status.json"

cat > "$STATUS_FILE" <<JSON
{
  "last_check": "$NOW",
  "upstream_main_sha": "$POST_FETCH_SHA",
  "previous_upstream_main_sha": "$PRE_FETCH_SHA",
  "local_patches_unique_count": ${LOCAL_UNIQUE_COUNT:-0},
  "local_patches_unique": "$KEEP",
  "local_patches_droppable_count": $DROPPABLE_COUNT,
  "local_patches_droppable": "$DROPPABLE",
  "upstream_commits_ahead_of_branch_point": ${UPSTREAM_AHEAD_COUNT:-0},
  "new_upstream_commits_since_last_check": "$NEW_UPSTREAM_LOG"
}
JSON

say ""
say "=== upstream check @ $NOW ==="
say "upstream/main: $POST_FETCH_SHA"
if [[ "$PRE_FETCH_SHA" != "$POST_FETCH_SHA" && "$PRE_FETCH_SHA" != "none" ]]; then
    say ""
    say "New upstream commits since last check:"
    git log --oneline "${PRE_FETCH_SHA}..${POST_FETCH_SHA}" | sed 's/^/  /'
fi
say ""
say "local-patches stack:"
say "$CHERRY_OUT" | sed 's/^/  /'
say ""
if [[ $DROPPABLE_COUNT -gt 0 ]]; then
    say "⚠️  $DROPPABLE_COUNT patch(es) appear in upstream — drop on next rebase: $DROPPABLE"
else
    say "✓ all $LOCAL_UNIQUE_COUNT local patches still unique to your stack"
fi
if [[ ${UPSTREAM_AHEAD_COUNT:-0} -gt 0 ]]; then
    say "ℹ  upstream is $UPSTREAM_AHEAD_COUNT commit(s) ahead of your branch point — consider rebasing"
fi
say ""
say "Logged to: $STATUS_FILE"
