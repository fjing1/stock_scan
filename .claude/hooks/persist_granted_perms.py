#!/usr/bin/env python3
"""PostToolUse hook: persist interactively-granted Bash permissions.

Wired from `.claude/settings.local.json` as a PostToolUse hook with matcher
"Bash". Because PostToolUse only fires AFTER a tool ran, reaching this script
means the command was permitted — either by an existing allow rule, or by the
user approving it interactively this session. We persist only the *fresh*
grants (those not already covered by a rule) so future sessions reuse them
without re-prompting.

Behavior (configured via the user's choices, 2026-06-08):
  - Scope: Bash only.
  - Granularity: generalize each (sub)command to a command *prefix* pattern
        git push origin main          -> Bash(git push *)
        ../../vcp_env/bin/python a.py  -> Bash(../../vcp_env/bin/python *)
  - Safety: destructive / network commands (see DANGER_*) are NEVER persisted,
    matching the user's standing rule that rm/curl/git push/etc. keep prompting.
  - Idempotent: a command already covered by an existing allow rule is skipped,
    so the allowlist does not bloat.

Every decision is appended to `.claude/permission-grants.log` (the audit trail).
The script never raises into the session: any error exits 0 silently.
"""
import sys
import os
import json
import re
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLAUDE_DIR = os.path.dirname(SCRIPT_DIR)                       # .../.claude
# Paths are overridable via env vars purely so the script can be unit-tested
# against a throwaway settings file; in normal operation the defaults apply.
SETTINGS = os.environ.get("PERM_HOOK_SETTINGS", os.path.join(CLAUDE_DIR, "settings.local.json"))
LOG = os.environ.get("PERM_HOOK_LOG", os.path.join(CLAUDE_DIR, "permission-grants.log"))
LOCK = os.environ.get("PERM_HOOK_LOCK", os.path.join(CLAUDE_DIR, ".perm-hook.lock"))

# Programs whose grants must never be auto-persisted (still run when approved,
# just never added to the allowlist). Edit freely.
DANGER_PROGRAMS = {
    "rm", "rmdir", "sudo", "su", "doas", "chmod", "chown", "chgrp",
    "dd", "mkfs", "fdisk", "parted", "diskutil", "shred", "truncate",
    "kill", "killall", "pkill", "shutdown", "reboot", "halt", "poweroff",
    "curl", "wget", "nc", "ncat", "netcat", "ssh", "scp", "sftp", "telnet",
    "ftp", "rsync", "eval", "exec", "source", "mv", "ln", "mkfifo",
    "crontab", "launchctl", "defaults", "git-push",
    "bash", "sh", "zsh", "dash", "ksh", "fish",   # arbitrary script exec / curl|bash
}
# git subcommands that are irreversible / push to remotes -> never persist.
DANGER_GIT_SUBCMDS = {
    "push", "reset", "clean", "rebase", "filter-branch", "filter-repo",
    "gc", "prune", "update-ref", "reflog", "checkout", "restore", "branch",
}
# Tools that take a subcommand -> generalize on the first TWO tokens.
SUBCOMMAND_TOOLS = {
    "git", "npm", "npx", "pnpm", "yarn", "bun", "docker", "docker-compose",
    "kubectl", "cargo", "go", "gh", "brew", "pip", "pip3", "conda", "poetry",
    "make", "terraform", "systemctl", "apt", "apt-get", "dnf", "yum",
}

_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def log(msg):
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG, "a") as fh:
            fh.write(f"{ts}  {msg}\n")
    except Exception:
        pass


def split_subcommands(command):
    """Split a shell line into its top-level sub-commands on && || ; | & and
    newlines, WITHOUT splitting inside single/double quotes. This matters for
    things like  python3 -c "a; b | c"  and  jq '... | ...'  whose operators
    are data, not shell control."""
    parts, buf = [], []
    i, n, quote = 0, len(command), None
    while i < n:
        c = command[i]
        if quote:
            buf.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1]); i += 2; continue
            if c == quote:
                quote = None
            i += 1; continue
        if c in ("'", '"'):
            quote = c; buf.append(c); i += 1; continue
        if c == "\\" and i + 1 < n:
            buf.append(c); buf.append(command[i + 1]); i += 2; continue
        if command[i:i + 2] in ("&&", "||"):
            parts.append("".join(buf)); buf = []; i += 2; continue
        # Note: split on `&&`/`||` but NOT a lone `&` — a bare & would break
        # redirections like `2>&1`. (`2>&1` then just rides along as an arg.)
        if c in (";", "|", "\n"):
            parts.append("".join(buf)); buf = []; i += 1; continue
        buf.append(c); i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


# A program token we're willing to turn into an allow rule: plain command names
# or paths. Anything with shell/quote/paren junk is rejected so a mis-parse can
# never produce a garbage rule like  Bash(print('valid *).
_SANE_PROGRAM = re.compile(r"^[A-Za-z0-9_./+-]+$")


def generalize(subcmd):
    """Return (rule_pattern, program_basename, git_subcmd_or_None) for one
    sub-command, or (None, None, None) if it can't be generalized safely.

    rule_pattern is the inside of Bash(...): a prefix followed by ' *'."""
    toks = subcmd.split()
    if not toks:
        return None, None, None

    env_tokens = []
    i = 0
    while i < len(toks) and _ENV_ASSIGN.match(toks[i]):
        env_tokens.append(toks[i])
        i += 1
    if i >= len(toks):
        return None, None, None   # only env assignments, nothing to run

    program = toks[i]
    if not _SANE_PROGRAM.match(program):
        return None, None, None   # mis-parsed / odd token -> never make a rule
    base = os.path.basename(program)
    rest = toks[i + 1:]

    git_sub = None
    prefix_tokens = env_tokens + [program]
    if base in SUBCOMMAND_TOOLS:
        sub = next((t for t in rest if not t.startswith("-")), None)
        if sub is not None:
            prefix_tokens.append(sub)
            if base == "git":
                git_sub = sub

    pattern = " ".join(prefix_tokens) + " *"
    # Final gate: a real command prefix never contains shell metacharacters.
    # If any leaked in (subshell `(cd ..`, an arg like `build)`, stray quote),
    # refuse to make a rule — let it keep prompting rather than persist junk.
    if re.search(r"""[()\[\]{}'"`;|&<>]""", pattern[:-2]):
        return None, base, git_sub
    # Sanity: the generalized pattern must actually cover the original command.
    if not covered(subcmd, pattern):
        return None, base, git_sub
    return pattern, base, git_sub


def covered(command, pattern):
    """Does an allow-rule inner pattern cover this command string?
    Mirrors Claude Code glob semantics closely enough for dedup: '*' is a
    wildcard, and a trailing ' *' also matches the bare prefix."""
    if pattern == command:
        return True
    if "*" in pattern:
        rx = "^" + re.escape(pattern).replace("\\*", ".*") + "$"
        if re.match(rx, command):
            return True
        if pattern.endswith(" *") and command == pattern[:-2]:
            return True
    return False


def is_dangerous(base, git_sub):
    if base in DANGER_PROGRAMS:
        return True
    if base == "git" and git_sub in DANGER_GIT_SUBCMDS:
        return True
    return False


def existing_bash_patterns(allow):
    """Yield the inner pattern of each Bash(...) allow rule."""
    for rule in allow:
        if isinstance(rule, str) and rule.startswith("Bash(") and rule.endswith(")"):
            yield rule[len("Bash("):-1]


def main():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except Exception:
        return 0
    if data.get("tool_name") != "Bash":
        return 0
    command = (data.get("tool_input") or {}).get("command")
    if not command or not isinstance(command, str):
        return 0
    # A heredoc (`cmd <<'EOF' ... EOF`) feeds a script body as DATA. Don't parse
    # those body lines as shell commands — keep only through the heredoc-start
    # line and drop the rest.
    if "<<" in command:
        lines = command.split("\n")
        cut = next((i for i, ln in enumerate(lines) if "<<" in ln), None)
        if cut is not None:
            command = "\n".join(lines[:cut + 1])

    try:
        with open(SETTINGS) as fh:
            settings = json.load(fh)
    except Exception:
        return 0   # missing/malformed -> never clobber
    allow = settings.setdefault("permissions", {}).setdefault("allow", [])
    if not isinstance(allow, list):
        return 0
    existing = list(existing_bash_patterns(allow))

    new_rules = []
    for sub in split_subcommands(command):
        # Already permitted by some rule? Then it wasn't a fresh grant -> skip.
        if any(covered(sub, pat) for pat in existing):
            continue
        pattern, base, git_sub = generalize(sub)
        if pattern is None:
            continue   # not a clean, generalizable command -> silently skip
        if is_dangerous(base, git_sub):
            log(f"SKIP (danger-listed '{base}{'/' + git_sub if git_sub else ''}'): {sub!r}")
            continue
        rule = f"Bash({pattern})"
        if rule not in new_rules and rule not in allow:
            new_rules.append(rule)

    if not new_rules:
        return 0

    # Lock + re-read + append + atomic replace, to coexist with Claude Code's
    # own writes to this file.
    import fcntl
    try:
        lockfh = open(LOCK, "w")
        fcntl.flock(lockfh, fcntl.LOCK_EX)
    except Exception:
        lockfh = None
    try:
        try:
            with open(SETTINGS) as fh:
                settings = json.load(fh)
        except Exception:
            return 0
        allow = settings.setdefault("permissions", {}).setdefault("allow", [])
        if not isinstance(allow, list):
            return 0
        existing = set(existing_bash_patterns(allow))
        added = []
        for rule in new_rules:
            pat = rule[len("Bash("):-1]
            if rule in allow:
                continue
            # Re-check coverage in case Claude Code added a covering rule between reads.
            if any(covered(pat, ep) or covered(pat[:-2] if pat.endswith(" *") else pat, ep)
                   for ep in existing):
                continue
            allow.append(rule)
            existing.add(pat)
            added.append(rule)
        if not added:
            return 0
        tmp = SETTINGS + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(settings, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, SETTINGS)
        for rule in added:
            log(f"SAVED {rule}   (from: {command!r})")
        msg = "🔓 Auto-saved permission(s): " + ", ".join(added)
        print(json.dumps({"systemMessage": msg, "suppressOutput": True}))
    finally:
        if lockfh is not None:
            try:
                fcntl.flock(lockfh, fcntl.LOCK_UN)
                lockfh.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception:
        sys.exit(0)
