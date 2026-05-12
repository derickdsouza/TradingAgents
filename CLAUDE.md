# Local Fork Workflow — TradingAgents

This repo is a personal fork of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).
The goal is to keep a stack of focused local patches on top of upstream `main`,
rebasing periodically and dropping any patch that upstream eventually adopts.

This file documents the workflow so it survives between Claude sessions, and
also serves as the assistant briefing for further changes. It is **tracked on
the `meta` branch of the fork** (separate from `local-patches`), and listed in
`.git/info/exclude` so it stays invisible on every other branch.

---

## Remote topology

```
upstream  https://github.com/TauricResearch/TradingAgents.git    # the canonical repo (read-only for us)
origin    https://github.com/derickdsouza/TradingAgents.git      # personal fork (push here)
```

Branches:

- `main` — tracks `upstream/main`. Never commit here directly. Used as the rebase base for `local-patches`.
- `local-patches` — tracks `origin/local-patches`. All personal **code** changes live as a stack of focused commits on top of `upstream/main`. Rebased onto upstream periodically.
- `meta` — tracks `origin/meta`. Carries durable **non-code** files (this `CLAUDE.md`, `.claude/check-upstream.sh`). Sits on top of `upstream/main` as a tiny one-commit branch. **Never rebased** — it's reference material, not a patch stack.

The split exists because `CLAUDE.md` and the upstream-check script need to sync across machines (so they go on `meta`, on the fork) but shouldn't tangle with the patch stack rebase loop (so they're not on `local-patches`). Per-machine state files (`.claude/upstream-status.json`, `.claude/settings.local.json`) stay strictly local via `.git/info/exclude`.

**Restore on a new machine after cloning the fork:**
```bash
git checkout meta -- CLAUDE.md .claude/check-upstream.sh
chmod +x .claude/check-upstream.sh
```
Then continue work on `local-patches` as normal — the meta files will be present in the working tree but excluded from the index of every other branch.

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

## Preferred local seams

Fork-specific behavior should accumulate behind named, stable seams instead of
sprawling across upstream-owned modules. Edits inside a seam don't conflict
with upstream rewrites; edits inside upstream files do. Each seam below is a
place to add new local behavior without touching upstream prose.

| Seam | Purpose | Location |
|---|---|---|
| **Decision contract validators** | Enforce semantic trading invariants on Trader/Portfolio Manager structured output (e.g. stop below entry on a long Hold). Caught in code, not prompts. | planned: `tradingagents/contracts/` |
| **Evidence ledger** | Compact structured facts (key levels, regime, news, dates) threaded through agent state so prose reports don't carry the load. | planned: `tradingagents/state/evidence.py` |
| **Prompt overlays** | Fork-specific prompt fragments layered onto upstream prompt scaffolds at the call site. | planned: `tradingagents/prompts/overlays/` |
| **Manager scorecards** | Explicit, weighted reasoning surfaces for Research Manager and Portfolio Manager, aligned with the evidence ledger. | planned: `tradingagents/agents/managers/scorecard.py` |
| **Outcome memory policy** | Horizon- and region-aware reflection: resolve trade outcomes against the right benchmark and holding period. | planned: `tradingagents/memory/outcome_policy.py` |
| **Evaluation harness** | Regression tests over report-quality invariants and decision contracts. Runs locally and in CI. | planned: `tests/eval/` |
| **Vendor routing** | Region-specific data providers (yfinance for US, NSE/BSE handling for `.NS`/`.BO`). | existing: `tradingagents/dataflows/y_finance.py` |
| **Report renderers** | Markdown/PDF output shape, including stop-suppression logic and trade-setup formatting. | existing: `tradingagents/reports/` (and CLI) |

When tracked work for any "planned" seam lands, update its row from `planned`
to the actual path so this table stays load-bearing rather than aspirational.

---

## Direct edit vs. local adapter

Default to a local seam. Edit upstream-owned files directly only when **all**
of these hold:

1. The change is genuinely upstream-shaped (a bug, a clearer comment, a
   provider/model registry entry that any user would want).
2. No existing seam covers it, and adding one would be over-engineering.
3. The edit fits in a one-paragraph commit body — meaning a future reader can
   reapply the intent on top of an upstream rewrite without archaeology.

If any of those fail, prefer one of:

- Hook a small call from upstream code into a fork-owned module (a 1–3 line
  edit upstream + the real logic in a local file). The 1–3 lines are cheap to
  re-apply on rebase; the local file is rebase-safe.
- Add the behavior to an existing seam.
- Stand up a new seam (and add a row to the table above).

If a deep upstream edit is genuinely required, factor it into its own focused
commit on `local-patches` so a future rebase can drop or reshape it without
unrelated collateral.

---

## Minimum verification

After rebasing onto a new `upstream/main`, or after editing any seam:

```bash
./tradingagents/bin/python -m pytest -q
```

Once the evaluation harness exists, also run its subset when the change
touches decision contracts, report shape, or outcome resolution:

```bash
./tradingagents/bin/python -m pytest -q tests/eval
```

Mirror any edited `tradingagents/<path>.py` into the venv site-packages copy
(see "Mirroring to the venv site-packages" below) before running pytest, or
the test will exercise the stale install rather than your edit.

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

---

## Agent skills

Per-repo setup for the engineering skills (`triage`, `to-issues`, `to-prd`, `qa`,
`improve-codebase-architecture`, `diagnose`, `tdd`, `grill-with-docs`). This
block overrides the global defaults in `~/.claude/CLAUDE.md`.

### Issue tracker

Beads (`bd`). Issues, tasks, and PRDs all live in the local Beads workspace.
See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical role names used verbatim as native `bd` labels (`needs-triage`,
`needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context. `CONTEXT.md` and `docs/adr/` will live at the repo root once
materialized — neither exists yet. Skills should proceed silently when absent.
See `docs/agents/domain.md`.
