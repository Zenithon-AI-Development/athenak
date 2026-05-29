# CLAUDE.md

Guidance for AI agents working in the AthenaK repository.

## Agent skills

These engineering skills (from `mattpocock/skills`) read per-repo config from `docs/agents/`.
Re-run `/setup-matt-pocock-skills` only to switch issue trackers or restart from scratch.

### Issue tracker

Issues are tracked in **GitHub Issues** — repo **`Zenithon-AI-Development/athenak`** — via the
`gh` CLI (`GH_TOKEN` lives in the untracked repo-root `.env`; load it with `set -a && . ./.env && set +a`).
See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles map 1:1 to GitHub labels in the `Zenithon-AI-Development/athenak` repo. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Document output format

When a skill produces a **standalone documentation deliverable** — a handoff doc, a
next-step / session summary, an architecture or audit report — write it as a
**self-contained rich HTML file**, not Markdown. Use **Tailwind via CDN** for styling and
**Mermaid via CDN** only when a diagram communicates better than prose; no other scripts.
Use `<code>` for paths/commands/identifiers, and reference existing artifacts by path or
link rather than duplicating them. After writing, open it (`xdg-open <path>` on Linux,
`open` on macOS, `start` on Windows) and report the absolute path. (`improve-codebase-architecture`
already follows this — see its `HTML-REPORT.md` for the scaffold and styling conventions.)

These stay **Markdown** — do *not* convert them:

- Repo-committed structured docs read by other skills: `CONTEXT.md`, `docs/adr/*.md`.
- Content published to the issue tracker (GitHub issues, comments).
- Skill definitions (`SKILL.md`) and other config files.
