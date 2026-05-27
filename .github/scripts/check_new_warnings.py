#!/usr/bin/env python3
"""Fail if a compiler warning lands on a line this branch added or changed.

AthenaK's existing tree is not warning-clean under ``-Wall -Wextra`` (hundreds of
pre-existing warnings, plus some from Kokkos headers), so a blanket ``-Werror`` would
break the build on the very first PR. This script implements the acceptance criterion
"new/changed code is warning-clean" precisely instead: it builds with ``-Wall -Wextra``
(warnings, not errors), then cross-references the compiler's warning lines against the
exact post-image line numbers this branch introduced, and exits non-zero only when a
warning falls on changed code.

Usage:
    check_new_warnings.py --build-log build.log [--base-ref origin/main]
                          [--repo-root <path>]

The build log is the combined stdout+stderr of the ``-Wall -Wextra`` build. The base ref
defaults to ``origin/main``; the diff is taken against the merge-base (``base...HEAD``).
"""

import argparse
import os
import re
import subprocess
import sys

# GCC/Clang warning line: "path:line:col: warning: message [-Wflag]" (col optional).
WARNING_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?:\d+:)?\s*warning:\s*(?P<msg>.*)$")
# Unified-diff hunk header with --unified=0: "@@ -a,b +c,d @@" (the ",d" parts optional).
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")
# A diff "+++ b/<path>" file header (skip "+++ /dev/null" deletions).
PLUS_FILE_RE = re.compile(r"^\+\+\+ b/(?P<path>.*)$")

SOURCE_SUFFIXES = (".cpp", ".hpp", ".c", ".h")


def run_git(args, repo_root):
    """Return stdout of a git command, or None if it fails."""
    try:
        out = subprocess.run(
            ["git"] + args,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"git {' '.join(args)} failed:\n{exc.stderr}\n")
        return None


def changed_lines(base_ref, repo_root):
    """Map each changed source file (repo-relative) to the set of lines it added.

    Uses ``git diff --unified=0 base...HEAD`` so only lines introduced on this branch
    (post-image line numbers, matching what the compiler reports) are collected.
    """
    diff = run_git(
        ["diff", "--unified=0", f"{base_ref}...HEAD", "--", "*.cpp", "*.hpp",
         "*.c", "*.h"],
        repo_root,
    )
    if diff is None:
        return None
    result = {}
    current = None
    for line in diff.splitlines():
        mfile = PLUS_FILE_RE.match(line)
        if mfile:
            current = mfile.group("path")
            result.setdefault(current, set())
            continue
        mhunk = HUNK_RE.match(line)
        if mhunk and current is not None:
            start = int(mhunk.group("start"))
            count = mhunk.group("count")
            count = 1 if count is None else int(count)
            for offset in range(count):
                result[current].add(start + offset)
    # Drop files with no added lines (pure deletions) for a tidy report.
    return {f: lines for f, lines in result.items() if lines}


def normalize(path, repo_root):
    """Return a repo-relative, normalized path or None if it is outside the repo."""
    norm = os.path.normpath(path)
    if os.path.isabs(norm):
        norm = os.path.normpath(os.path.relpath(norm, repo_root))
    # Resolve relative compiler paths (e.g. "../../../src/x.cpp") to the src-rooted form.
    if norm.startswith("..") and "src/" in norm:
        norm = norm[norm.index("src/"):]
    return norm


def parse_warnings(build_log, repo_root):
    """Yield (relpath, line_number, message) for every warning in the build log."""
    with open(build_log, "r", errors="replace") as handle:
        for raw in handle:
            match = WARNING_RE.match(raw.rstrip("\n"))
            if not match:
                continue
            rel = normalize(match.group("path"), repo_root)
            if rel is None or not rel.endswith(SOURCE_SUFFIXES):
                continue
            yield rel, int(match.group("line")), match.group("msg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-log", required=True, help="combined build stdout+stderr")
    parser.add_argument("--base-ref", default="origin/main", help="PR base ref to diff")
    parser.add_argument("--repo-root", default=None, help="repository root (auto-detect)")
    opts = parser.parse_args()

    repo_root = opts.repo_root
    if repo_root is None:
        top = run_git(["rev-parse", "--show-toplevel"], ".")
        repo_root = top.strip() if top else os.getcwd()
    repo_root = os.path.abspath(repo_root)

    if not os.path.exists(opts.build_log):
        sys.stderr.write(f"ERROR: build log not found: {opts.build_log}\n")
        return 2

    changed = changed_lines(opts.base_ref, repo_root)
    if changed is None:
        sys.stderr.write(
            "ERROR: could not compute the diff; is the base ref fetched?\n"
        )
        return 2

    # De-duplicate identical warnings (a header included by many TUs repeats).
    offenders = set()
    for rel, lineno, msg in parse_warnings(opts.build_log, repo_root):
        if lineno in changed.get(rel, ()):  # warning on a line this branch changed
            offenders.add((rel, lineno, msg))

    if offenders:
        sys.stderr.write(
            "FAIL: compiler warning(s) on lines added/changed by this branch.\n"
            "New and changed code must be warning-clean (-Wall -Wextra).\n\n"
        )
        for rel, lineno, msg in sorted(offenders):
            sys.stderr.write(f"  {rel}:{lineno}: warning: {msg}\n")
        sys.stderr.write(f"\n{len(offenders)} warning(s) on changed lines.\n")
        return 1

    print("OK: no compiler warnings on lines added/changed by this branch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
