# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual
labels in this repo's **GitHub Issues** (`Zenithon-AI-Development/athenak`). All five labels exist and
map 1:1.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), apply the corresponding GitHub
label with `gh issue edit <number> --add-label "<label>"` (and `--remove-label "<old-label>"` to clear
the previous triage state). Create a missing label first with `gh label create`.

GitHub also carries default labels (`bug`, `enhancement`, `documentation`, `duplicate`,
`good first issue`, `help wanted`, `invalid`, `question`); leave those untouched when changing triage
state. Edit the right-hand column if your vocabulary ever changes.
