# Auto-persist granted Bash permissions

`persist_granted_perms.py` is a **PostToolUse hook** (matcher `Bash`) wired in
`.claude/settings.local.json`. It makes permission grants stick: the first time
you approve a Bash command, the hook saves a rule so future sessions run it
without asking again.

## How it works

PostToolUse fires *after* a tool runs — meaning the command was permitted
(either by an existing rule, or because you just approved it). For each
sub-command the hook:

1. **Skips** it if an existing allow rule already covers it (no allowlist bloat).
2. **Generalizes** a fresh grant to a command *prefix*:
   - `git push origin main`  →  `Bash(git push *)`  (subcommand tools keep 2 tokens)
   - `../../vcp_env/bin/python scan_stocks.py`  →  `Bash(../../vcp_env/bin/python *)`
3. **Never persists** destructive / network commands (still run when approved,
   just never auto-allowlisted) — see `DANGER_PROGRAMS` / `DANGER_GIT_SUBCMDS`
   in the script (`rm`, `sudo`, `curl`, `wget`, `ssh`, `bash`/`sh`,
   `git push|reset|clean|rebase|checkout`, …).
4. Appends the rule to `.claude/settings.local.json` (atomic write + flock) and
   records it in `.claude/permission-grants.log` (timestamped audit trail).

Robustness guards: quote-aware sub-command splitting (won't break
`python3 -c "...; ..."` or `jq '... | ...'`), a heredoc guard (won't parse a
`<<EOF` body as commands), a sane-program-token check, and a
shell-metacharacter gate — so a mis-parse can never write a junk rule.

## Tuning

- Edit `DANGER_PROGRAMS`, `DANGER_GIT_SUBCMDS`, `SUBCOMMAND_TOOLS` at the top of
  the script.
- Scope is **Bash only** (matcher in settings). Add matchers to widen it.
- To pause it: set `"disableAllHooks": true` in settings, or remove the
  `hooks.PostToolUse` block. Review/disable via the `/hooks` menu.
- The script honors `PERM_HOOK_SETTINGS` / `PERM_HOOK_LOG` / `PERM_HOOK_LOCK`
  env overrides (used only for testing against a throwaway file).

## Note

`settings.local.json` is gitignored (personal); this script and README are not,
so the tooling can be shared while each person keeps their own grants.
