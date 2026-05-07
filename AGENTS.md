# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## Clarity

This project uses `clarity` to visualize code changes, provide design feedback, and guide refactoring.

### When to Use Clarity

1. **After making changes** - Run `clarity` to visualize your changes, understand impact, and prepare context for developer review.
    - **Always run `clarity show` when you modify 3 or more files** to ensure the developer can review the full scope of changes
2. **Discussing design** - Use `clarity` to visualize architecture and dependencies for specific files, directories, or commits when discussing design decisions with the developer.
3. **Refactoring verification** - After implementing design changes, run `clarity` to verify the resulting structure aligns with the discussed design.

### How to Use Clarity

**For developer review (visualize):**
- Generate and render graphs for the developer to review
- For CLI agents, default to DOT output (`clarity show` or `clarity show -f dot`)
- For CLI agents, generate a URL with `clarity show -u`, then open that URL in the system browser with the platform command:
  - macOS: `open "<url>"`
  - Linux: `xdg-open "<url>"`
  - Windows (cmd): `start "" "<url>"`
  - Windows (PowerShell): `Start-Process "<url>"`
- Use `clarity show -f mermaid` if your environment supports Mermaid rendering (desktop apps, IDEs)
- Use `clarity show` or `clarity show -f dot` if your environment supports Graphviz rendering or has dot tools installed (supports SVG, PNG, etc.)
- Do not assume `clarity show -u` auto-opens a browser in CLI environments; always open the generated URL explicitly
- Choose the visualization method that works best for your coding environment

**For agent verification (feedback and analysis):**
- Run `clarity show` and read the dot/mermaid output directly
- Parse the graph structure to verify dependencies and relationships
- No visualization needed - the text output contains all structural information
- Use this during refactoring iterations to confirm progress

### Quick Reference

```bash
clarity show                    # Visualize uncommitted changes (most common)
clarity show -c HEAD            # Visualize changes in last commit
clarity show -i <files/dirs>    # Build graph from specific files or directories (comma-separated)
clarity show -w <file1,file2>   # Find all paths between two or more files (comma-separated)
clarity show -f mermaid         # Generate output in mermaid format (default 'dot' Graphviz format)
clarity show -u                 # Generate visualization URL
```

For full reference, use `clarity show -h`
