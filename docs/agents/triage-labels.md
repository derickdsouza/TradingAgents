# Triage Labels

The skills speak in terms of five canonical triage roles. This repo uses the
canonical names verbatim as native `bd` labels.

| Role in mattpocock/skills | Native `bd` label | Meaning                                  |
| ------------------------- | ----------------- | ---------------------------------------- |
| `needs-triage`            | `needs-triage`    | Maintainer needs to evaluate this issue  |
| `needs-info`              | `needs-info`      | Waiting on reporter for more information |
| `ready-for-agent`         | `ready-for-agent` | Fully specified, ready for an AFK agent  |
| `ready-for-human`         | `ready-for-human` | Requires human implementation            |
| `wontfix`                 | `wontfix`         | Will not be actioned                     |

Apply the labels with `bd update <id> --add-label <name>`; remove with
`--remove-label <name>`. When a skill mentions a role (e.g. "apply the
AFK-ready triage label"), use the corresponding label string from this table.

Pair label changes with the appropriate native lifecycle state where it
clarifies intent (e.g. `--status=blocked` alongside `needs-info`,
`bd close` alongside `wontfix`).
