#!/bin/sh
# secret scan — scaffolded by quinn-suite skills/ci-contract (ADR 035) for sample-aws-physical-ai-toolchain. Regenerate from
# quinn-suite (`ci_contract.py scaffold --force-templates`), never hand-edit (qs-so4ox).
# The pre-commit secret scan (.pre-commit-config.yaml) and the secrets lane's scan step, a plain
# grep so no scanner binary ever runs on the box or the self-hosted runner: the tools on
# ~/.config/quinn/blocked-scanners trip GuardDuty and Epoxy isolates the instance (incident
# 2026-09-30, prototype-chat chat-oef6). It reads the staged copy of every added or changed file
# and refuses AWS access keys, private key headers, 12-digit account ids and private IPv4
# addresses. Each hit is printed as file:line and rule, never the matched value. With a base ref
# (CI's secrets lane: `secret-scan-staged.sh origin/main`) it reads HEAD's copy of the files
# changed since that ref.
#
# Exemptions: .claude/secret-scan-allow lists repo-relative files or directories, one per line then
# `# reason`, exempt from every rule; .claude/account-id-allow (the CI account-id grep's list) also
# exempts paths from the account-id rule. AWS's documented example keys (ending in EXAMPLE) and
# the placeholder ids (twelve zeros, 1234567890 then 12, and 123456 twice) always pass; this file
# spells them so, never as 12-digit runs, so a repo's own account-id grep passes it (qs-hwlgz). A
# private address written as a network block of /28 or wider (a VPC or subnet CIDR such as
# 10.0.0.0/16) passes, so a terraform file need not be wholly exempt (qs-t5is0); a /29 to /32 names
# a host or two and is refused like a bare address.
root=$(git rev-parse --show-toplevel) || exit 2
base=$1

# The checks need a grep whose -I skips binary input and whose -E knows \b (GNU, BSD). BusyBox
# accepts -I but ignores it, so a binary blob reads as text: refuse it rather than misjudge.
if ! printf 'a b\n' | grep -qE '\bb\b' 2>/dev/null || printf 'a\000b\n' | grep -IqE 'b' 2>/dev/null; then
  echo "secret-scan: this grep ignores -I or lacks \\b (BusyBox?); install GNU grep" >&2
  exit 2
fi

listed() { # listed <allow file> <path>: is <path> or a parent directory on the list?
  [ -r "$root/$1" ] || return 1
  sed 's/#.*//; s/^[[:space:]]*//; s|[[:space:]/]*$||' "$root/$1" | {
    while IFS= read -r p; do
      [ -n "$p" ] || continue
      case "$2" in "$p" | "$p"/*) exit 0 ;; esac
    done
    exit 1
  }
}

# The staged blob is named :0:<path>, never :<path>: git reads :0:foo as stage 0 of foo, so a file
# staged as 0:foo would scan the clean foo instead (qs-u2xk1m).
content() {
  if [ -n "$base" ]; then git show "HEAD:$1"; else git show ":0:$1"; fi
}

# readable <path>: does git hold a blob for <path>? A path git show cannot read would scan as empty
# and pass, so the loop refuses it instead (qs-u2xk1m).
readable() {
  if [ -n "$base" ]; then git cat-file -e "HEAD:$1" 2>/dev/null; else git cat-file -e ":0:$1" 2>/dev/null; fi
}

# gitlink <path>: is <path> a submodule commit (mode 160000)? It has no blob to read (qs-di2q7).
gitlink() {
  if [ -n "$base" ]; then entry=$(git -C "$root" --literal-pathspecs ls-tree HEAD -- "$1")
  else entry=$(git -C "$root" --literal-pathspecs ls-files --stage -- "$1"); fi
  case "$entry" in 160000\ *) return 0 ;; esac
  return 1
}

# check <path> <rule name> <pattern> <allowed values>: print path:line for each match of
# <pattern> that is not wholly an <allowed values> match. Exits 0 when it printed a hit. grep's \b
# counts an underscore as a word character, so KEY_<key> or role_<id> would pass: tr turns each
# underscore into a space first, which keeps the matches non-overlapping and the line numbers, so
# an allowed value cannot hide a real one beside it (qs-o60izg). LC_ALL=C makes tr bytewise: in a
# UTF-8 locale BSD tr stops at the first invalid byte, cutting off the rest of a Latin-1 file.
check() {
  content "$1" | LC_ALL=C tr _ ' ' | grep -IonE -e "$3" | grep -vE "^[0-9]+:($4)\$" | cut -d: -f1 | sort -un | {
    hit=1
    while IFS= read -r n; do
      printf 'secret-scan: %s:%s: %s\n' "$1" "$n" "$2" # dash's echo expands a \c in the path
      hit=0
    done
    exit $hit
  }
}

# -z prints every path raw: without it git C-quotes a name holding a non-ASCII byte, a quote, a
# backslash or a tab, and git show cannot read the quoted form (qs-o60izg, qs-u2xk1m). tr turns
# the NULs into newlines, so a name holding a newline would split into two paths that may both
# exist and scan clean: any newline in the raw list is refused instead.
names() {
  if [ -n "$base" ]; then git diff -z --name-only --diff-filter=ACMRT "$base"...HEAD
  else git diff --cached -z --name-only --diff-filter=ACMRT; fi
}
names >/dev/null || exit 2
if [ "$(names | tr -cd '\n' | wc -c)" -ne 0 ]; then
  echo "secret-scan: a changed path holds a newline, so it cannot be scanned; rename it" >&2
  exit 2
fi
files=$(names | tr '\000' '\n')

found=0
# The here-doc keeps the loop in this shell so `found` survives it.
while IFS= read -r f; do
  [ -n "$f" ] || continue
  listed .claude/secret-scan-allow "$f" && continue
  gitlink "$f" && continue
  if ! readable "$f"; then
    printf 'secret-scan: %s: git cannot read this path, so it cannot be scanned\n' "$f" >&2
    exit 2
  fi
  check "$f" "AWS access key" '\b(AKIA|ASIA)[A-Z0-9]{16}\b' '(AKIA|ASIA)[A-Z0-9]*EXAMPLE' && found=1
  check "$f" "private key header" '-----BEGIN ([A-Z0-9]+ )*PRIVATE KEY-----' 'NONE' && found=1
  # Under .beads only the account-id rule is skipped: bd metadata UUIDs can end in 12 decimal
  # digits, but a key pasted into a bead file is still refused (qs-geu140).
  case "$f" in .beads/*) beads=1 ;; *) beads= ;; esac
  if [ -z "$beads" ] && ! listed .claude/account-id-allow "$f"; then
    # Not after a dot, letter or digit: the decimal part of a float is no account id (qs-1vhzr).
    # grep -E has no lookbehind, so the match carries that one leading character. A UUID whose last
    # group is all digits (8-4-4-4-12 hex) is no account id either (qs-vqhusu): the match takes the
    # whole UUID instead, and the allowed values pass it.
    check "$f" "12-digit account id" '([0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}|^|[^.[:alnum:]_])[0-9]{12}\b' \
      '[^0-9]?(0{12}|1234567890(12)|(123456){2})|[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9]{12}' && found=1
  fi
  # The optional /nn makes grep take the whole CIDR, so only a bare address or a narrow block is a hit.
  check "$f" "private IP address" '\b(10\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])|192\.168)\.[0-9]{1,3}\.[0-9]{1,3}(/[0-9]{1,2})?\b' '[0-9.]+/([0-9]|1[0-9]|2[0-8])' && found=1
done <<EOF
$files
EOF

if [ "$found" = 1 ]; then
  echo "secret-scan: refused. Remove the value, or list a fixture path in .claude/secret-scan-allow with a reason."
  exit 1
fi
exit 0
