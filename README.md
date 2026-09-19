# Deep Research Agent — Claude Code Upgrade Plan

## Files

- `CLAUDE.md` — permanent project instructions for Claude Code.
- `docs/phases/PHASE-01-AUDIT.md` ... `PHASE-19-FINAL-VALIDATION.md` — one document per implementation phase.

## Workflow

Use one Claude Code session per phase:

1. Open the repository.
2. Ensure `CLAUDE.md` is present at repository root.
3. Open only the current phase file.
4. Ask Claude Code to inspect relevant files and produce a plan.
5. Review the plan.
6. Approve implementation.
7. Run focused tests.
8. Ask Claude Code to update the phase file.
9. Review the diff.
10. Commit the phase.
11. Start a fresh Claude Code session for the next phase.

## Important

Do not ask Claude Code to "build the whole architecture" in one prompt.

The architecture is already defined. Claude Code's job is to inspect the actual codebase and implement one controlled increment at a time.

LangSmith, automated evaluation, and evaluation datasets are intentionally excluded from these phases.
