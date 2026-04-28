# Agent file

The purpose of this tile is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in the project that surprises you, please alert the developer working with you and indicate that this is the case in the AgentMD file to help prevent future agents from having the same issue.

# Workflow rules

- **Always run `just ci` after any code change.**
- **Always ask before committing a `fix:` or `feat:` commit** — these trigger automated releases (patch and minor respectively).
- **NEVER commit `feat!:` or `BREAKING CHANGE:` without explicit user authorization** — these trigger a major version bump.
- Every new example should include:
  1. Ad-hoc classes in `src/anypinn/catalog/`
  2. example directory under `examples/`
  3. A CLI scaffold template under `src/anypinn/cli/scaffold/<name>/`
