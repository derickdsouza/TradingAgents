# Issue tracker: Beads (`bd`)

Issues, tasks, and PRDs for this repo live in the local Beads workspace. Use
the `bd` CLI for all operations. Beads is also the project's task-tracker — do
**not** use TodoWrite, TaskCreate, or markdown TODO lists.

Run `bd prime` once per session for the full command surface.

## Conventions

- **Create an issue**: `bd create --title "<summary>" --description "<why + what>" --type task|bug|feature|epic --priority 2`
  - Priority is `0–4` or `P0–P4` (`0` = critical, `4` = backlog). Never use
    words like "high"/"medium"/"low".
  - Use a heredoc for multi-line descriptions.
- **Read an issue**: `bd show <id>` — full view including dependencies, notes,
  labels, and design.
- **List issues**:
  - `bd ready` — issues with no open blockers, ready to work.
  - `bd list --status=open` — all open issues.
  - `bd list --status=in_progress` — currently claimed work.
- **Claim work**: `bd update <id> --claim`
- **Update fields inline**: `bd update <id> --title/--description/--notes/--design`
- **Apply / remove labels**: `bd update <id> --add-label "<name>"` /
  `--remove-label "<name>"`. Triage state lives on native labels — see
  `triage-labels.md` for the canonical role strings.
- **Add dependencies**: `bd dep add <id> <depends-on-id>` (issue depends on
  depends-on, i.e. depends-on blocks issue).
- **Close**: `bd close <id>` (or `bd close <id1> <id2> ...` for batch close).
- **Persistent knowledge**: `bd remember "<insight>"` for cross-session notes;
  search with `bd memories <keyword>`.

## When a skill says "publish to the issue tracker"

Run `bd create` with an appropriate `--type` and `--priority`. Capture the
rationale in `--description`. Add `--blocked-by` or `--blocks` flags (or
follow up with `bd dep add`) when there is an explicit dependency.

## When a skill says "fetch the relevant ticket"

Run `bd show <id>`. If you don't have an id, narrow with `bd list --status=open`
or `bd ready` and pick by title/description.

## When a skill says "apply the <role> label"

Apply it as a native `bd` label with the exact role string from
`triage-labels.md`:

```bash
bd update <id> --add-label ready-for-agent
```

Move the bead's status to match the role's intent where appropriate (e.g.
`bd update <id> --status=blocked` for a `needs-info` role, `bd close <id>`
with a `wontfix` label for the wontfix role).

## Session close

After meaningful work, close completed issues and push code:

```bash
bd close <id1> <id2> ...
git status
git add <files>
git commit -m "..."
git push
```

This fork does **not** have a Dolt remote configured for the Beads workspace
itself — `bd dolt push` is a no-op here and should not be run. Beads state is
local-machine-only; code lives on `origin/local-patches` (see top-level
`CLAUDE.md` for the fork workflow).

Work is not complete until `git push` succeeds.
