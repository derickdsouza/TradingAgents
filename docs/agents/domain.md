# Domain Docs

How the engineering skills should consume this repo's domain documentation
when exploring the codebase.

## Layout: single-context

One `CONTEXT.md` + one `docs/adr/` at the repo root, when they exist:

```
/
├── CONTEXT.md              ← not yet present
├── docs/
│   ├── adr/                ← not yet present
│   └── agents/             ← this directory
└── tradingagents/
    └── ...
```

Neither `CONTEXT.md` nor `docs/adr/` is materialized yet. The producer skill
(`/grill-with-docs`) creates them lazily when terms or decisions actually get
resolved.

## Before exploring, read these (when they exist)

- **`CONTEXT.md`** at the repo root
- **`docs/adr/`** — read ADRs that touch the area you're about to work in

If either is absent, **proceed silently**. Don't flag the absence; don't
suggest creating these files upfront.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor
proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`.
Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either
you're inventing language the project doesn't use (reconsider), or there's
a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than
silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_

## Fork context

This repo is a personal fork of `TauricResearch/TradingAgents` carrying a
stack of focused local patches on top of upstream `main` (see the top-level
`CLAUDE.md`). When ADRs and `CONTEXT.md` are eventually authored here, they
should describe **the fork's seams and conventions** rather than upstream's
architecture — anything truly upstream-shaped belongs in a PR to upstream,
not in this repo's domain docs.
