# Local Fork Workflow — TradingAgents

This repo is a personal fork of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).
The goal is to keep a stack of focused local patches on top of upstream `main`,
rebasing periodically and dropping any patch that upstream eventually adopts.

This file documents the workflow so it survives between Claude sessions, and
also serves as the assistant briefing for further changes. It is local-only —
listed in `.git/info/exclude` so it never gets pushed to either remote.

---

## Remote topology

```
upstream  https://github.com/TauricResearch/TradingAgents.git    # the canonical repo (read-only for us)
origin    https://github.com/derickdsouza/TradingAgents.git      # personal fork (push here)
```

Branches:

- `main` — tracks `upstream/main`. Never commit here directly.
- `local-patches` — tracks `origin/local-patches`. All personal changes live as a stack of focused commits on top of `upstream/main`.

---

## Patch stack (current)

The `local-patches` branch is composed of small, single-purpose commits so each
one can be evaluated against upstream independently. As of this writing:

| # | Commit  | Theme                                                                                                  |
|---|---------|--------------------------------------------------------------------------------------------------------|
| 1 | feat(cli)         | Non-interactive runs, auto-save reports, `--horizon` flag                                    |
| 2 | feat(horizon)     | Config-driven trading horizon (`swing` / `position` / `long-term`) plumbed through agents    |
| 3 | feat(indicators)  | Institutional volume (OBV/RVOL/breakout_20/MFI/ADX) + VSA + Guppy + Minervini SEPA suite     |
| 4 | feat(llm)         | `glm-anthropic` provider for z.ai's Anthropic-compatible endpoint                             |

When adding new local features, prefer **a new focused commit on `local-patches`**
over amending an existing one — the rebase tooling below works better with small,
single-purpose commits.

---

## Session start protocol (assistant: read this first)

**On every new session in this repo, do this before anything else:**

1. Read `.claude/upstream-status.json`. Compute `now - last_check`.
2. **If `last_check` is more than 7 days old (or the file is missing)** → run `.claude/check-upstream.sh`. The script does the fetch, refreshes the log, and prints a human summary. Then surface the result to the user — at minimum, mention:
   - how many days since the last check
   - new upstream commits since last check (if any)
   - any patches that became upstream-equivalent and should be dropped on next rebase (`local_patches_droppable_count > 0`)
3. **If `last_check` is fresh (≤ 7 days)** → just note the cached status from the file in one line ("upstream last checked N days ago, K patches still unique") and continue with the user's task.

**Do not run the script if the user is mid-task** — finish their request first, then mention upstream status if it warrants attention. The check is informational, never blocking.

If the script is missing or fails, mention it once and move on — do not retry.

The status file format:
```json
{
  "last_check":                 "ISO-8601 UTC",
  "upstream_main_sha":           "current upstream/main",
  "previous_upstream_main_sha":  "what it was last check",
  "local_patches_unique_count":  N,
  "local_patches_unique":        "comma-separated SHAs",
  "local_patches_droppable_count": M,
  "local_patches_droppable":     "comma-separated SHAs upstream now has equivalents for",
  "upstream_commits_ahead_of_branch_point": K,
  "new_upstream_commits_since_last_check":  "oneline log, pipe-separated"
}
```

To force a check manually at any time:
```bash
.claude/check-upstream.sh
```

---

## Daily / weekly: pull upstream updates and rebase

```bash
git checkout main
git fetch upstream
git merge --ff-only upstream/main      # main mirrors upstream exactly
git push origin main                    # keep fork's main in sync (optional but tidy)

git checkout local-patches
git rebase upstream/main                # replay our patches on top of new upstream
# resolve any conflicts → git add ... → git rebase --continue
git push --force-with-lease origin local-patches
```

`--force-with-lease` is the safe variant of `--force`: it refuses to push if the
remote moved since we last fetched.

If a rebase conflict is in a file upstream completely rewrote, the right move
is usually:
1. Drop the patch (`git rebase` will let you `git rebase --skip` or edit the
   todo list to delete the commit).
2. Reapply the *intent* of the patch on top, in a fresh commit.
3. Don't fight to preserve the original SHA.

---

## Detect upstream-merged patches (so we can drop them)

Three complementary tools — run any/all of these after fetching upstream.

### 1. `git cherry` — patch-id equivalence

```bash
git cherry -v upstream/main local-patches
```

- `+ <sha> <subject>` → still unique to our stack, keep
- `- <sha> <subject>` → upstream has an equivalent patch (by content, not SHA), **drop on next rebase**

This is the everyday command. Run it before rebasing to spot patches that have
landed upstream.

### 2. `git range-diff` — semantic before/after

After a rebase, compare your old vs new patch stack:

```bash
git range-diff upstream/main...local-patches@{1} upstream/main...local-patches
```

Shows commits as `=` (unchanged), `!` (modified), `<` (dropped), `>` (added).
Useful for sanity-checking that a rebase didn't quietly lose anything.

### 3. `git log --left-right --cherry-pick`

```bash
git log --left-right --cherry-pick --no-merges upstream/main...local-patches
```

Lists only commits unique to each side; equivalents are hidden. Higher-signal
log view than plain `git log`.

---

## Sending pieces upstream

When a patch is generally useful, opening a PR back to upstream is the cleanest
outcome — once merged, that commit drops from our stack permanently.

```bash
git checkout -b feature-X-upstream upstream/main
git cherry-pick <sha-of-our-commit>
git push origin feature-X-upstream
gh pr create --repo TauricResearch/TradingAgents --base main \
  --head derickdsouza:feature-X-upstream
```

Good upstream candidates:
- generic bug fixes
- new providers / model catalog entries
- regional benchmark resolution (likely useful to anyone trading non-US tickers)
- structured-output managers, if not already done

Personal-style preferences (e.g. `--horizon` defaulting to `swing`) usually
shouldn't be upstreamed — keep those local.

---

## Conventions for new patches

Each commit on `local-patches` should:

1. **Touch one theme.** The 4-commit groups above are the model — split if a
   change spans two.
2. **Have a body explaining *why*** (one paragraph), not just what. Why-first
   commits make rebase conflicts easier to resolve a year from now when the
   "what" is no longer obvious from the diff.
3. **Stand on its own.** A patch that only makes sense after another patch
   should be folded in or follow it as a clearly-labelled follow-up. Never
   mid-stack patches that depend on later ones.
4. **Not modify upstream code "casually".** Reformatting an unrelated file or
   silently fixing a different bug while adding a feature creates rebase pain.

---

## Files that should never leave the local checkout

The following are in `.git/info/exclude` (local, not in upstream's `.gitignore`):

- `CLAUDE.md` — this file
- `reports/` — generated analysis outputs
- `.claude/` — assistant scratch dir: upstream-check script, last-check log, future hooks

If you add new local-only artefacts, append them to `.git/info/exclude` rather
than `.gitignore` (which would create a noisy diff against upstream).

---

## Mirroring to the venv site-packages

This project's venv lives at `tradingagents/lib/python3.13/site-packages/` and
imports from there, **not** from the working tree. After every edit to
`tradingagents/...` source files, mirror the changed file:

```bash
cp tradingagents/<path>.py tradingagents/lib/python3.13/site-packages/tradingagents/<path>.py
```

(This is a non-editable install quirk — fix once at install time with
`pip install -e .` if convenient. Until then, mirror manually.)

---

## Project-specific context for the assistant

- **`get_stock_stats_indicators_window`** is the canonical entry point for
  technical indicators. Custom (non-stockstats) indicators are listed in
  `_CUSTOM_INDICATORS` in `tradingagents/dataflows/y_finance.py` and computed
  by `_compute_custom_indicator`.
- **`_resolve_benchmark(symbol)`** picks the regional index for Minervini's
  RS-line check: `^CRSLDX` (Nifty 500) for `.NS`/`.BO`, `^FTSE`/`^HSI`/`^N225`/
  `^GSPTSE`/`^AXJO` for other suffixes, `SPY` otherwise.
- **`HORIZONS`** in `tradingagents/agents/utils/agent_utils.py` is the source of
  truth for swing/position/long-term lookbacks and prompt fragments.
- **CLI flags** added on top of upstream: `--horizon`, `--report-name`,
  `--skip-save-report`, `--display-report`. Default depth is `deep`; default
  horizon is `swing`.
- **Indian tickers** use the `.NS` (NSE) or `.BO` (BSE) suffix in yfinance and
  benchmark to Nifty 500 automatically.

When extending an indicator: register the description in `best_ind_params`,
add the key to `_CUSTOM_INDICATORS` (if it's not native to stockstats), advertise
it in `tradingagents/agents/analysts/market_analyst.py`'s system message, and
mirror both files into the venv.
