# Issue tracker: GitHub Issues

Issues and PRDs for this repo live in **GitHub Issues**, in the **`Zenithon-AI-Development/athenak`**
repository (<https://github.com/Zenithon-AI-Development/athenak/issues>). All operations go through the
**`gh` CLI** — there is no MCP server for this tracker.

## Authentication

`gh` is **not** globally authenticated. A `GH_TOKEN` lives in the untracked repo-root `.env`. Load it
before any `gh` call:

```bash
set -a && . ./.env && set +a
```

## Conventions

- **Default repo**: `Zenithon-AI-Development/athenak`. Pass `-R Zenithon-AI-Development/athenak` when the
  working directory might not be the repo.
- **Issue identifiers** are plain numbers like `#123`.
- **Create an issue**: `gh issue create -R Zenithon-AI-Development/athenak --title "..." --body "..."`.
  Add `--label`, `--assignee`, `--milestone` when known. Pass the body via a heredoc or `--body-file`
  so markdown formatting survives.
- **Read an issue**: `gh issue view <number> -R Zenithon-AI-Development/athenak` (add `--comments` for
  discussion history).
- **List issues**: `gh issue list -R Zenithon-AI-Development/athenak` filtered by `--state`, `--label`,
  `--assignee`, or `--search` as needed.
- **Comment**: `gh issue comment <number> -R Zenithon-AI-Development/athenak --body "..."`.
- **Apply / change labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`. Create
  a missing label first with `gh label create "<name>" --description "..." --color "<hex>"`.
- **Close / reopen**: `gh issue close <number> --reason completed|"not planned"` / `gh issue reopen <number>`.
- **PRDs**: write the PRD as a parent issue with implementation work as linked sub-issues (reference the
  parent `#N` in each child's body) — see `to-prd` / `to-issues`.

Pass markdown content directly (real newlines, not literal `\n`).

## AFK agent loop

The autonomous (Ralph) loop reads **GitHub Issues** directly and sequences work by the work-tag in the
issue title (letters before numbers; an `[CI]` tag takes precedence). Keep new issue titles tagged so
the loop can order them.

## When a skill says "publish to the issue tracker"

Create a GitHub issue in `Zenithon-AI-Development/athenak` with `gh issue create`, applying the
appropriate triage label (see `triage-labels.md`).

## When a skill says "fetch the relevant ticket"

Call `gh issue view <number> -R Zenithon-AI-Development/athenak --comments`.
