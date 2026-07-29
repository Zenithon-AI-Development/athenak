"""
Repo-hygiene checks for generated files (#218).

The test harness appends to tst/test_log.txt on every run.  While that file was
*tracked*, every local run dirtied it, a routine `git add -u` swept it into the
commit, and once it crossed GitHub's 100 MB blob limit the push was rejected --
naming a file the author never touched.  Tracked __pycache__/*.pyc files showed
up dirty on every run for the same reason.

These tests pin the fix: run output and Python bytecode must never be tracked,
and must be gitignored so a full suite run leaves `git status` clean.
"""

# Modules
import os
import subprocess

# Repo root: this file lives at <root>/tst/test_suite/style/
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)


def git(*args):
    """Run a git command at the repo root; return (returncode, stdout)."""
    result = subprocess.run(
        ["git", "-C", REPO_ROOT] + list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode, result.stdout


def tracked_files():
    """All paths tracked by git, relative to the repo root."""
    _, out = git("ls-files")
    return out.splitlines()


def test_log_file_untracked():
    """tst/test_log.txt is harness run output and must not be tracked."""
    assert "tst/test_log.txt" not in tracked_files(), (
        "tst/test_log.txt is tracked by git; it grows on every test run and "
        "will exceed GitHub's 100 MB push limit (#218). "
        "Untrack it with `git rm --cached tst/test_log.txt`."
    )


def test_log_file_gitignored():
    """tst/test_log.txt must be gitignored so runs leave git status clean."""
    code, _ = git("check-ignore", "-q", "tst/test_log.txt")
    assert code == 0, (
        "tst/test_log.txt is not gitignored; every suite run dirties the "
        "working tree (#218). Add it to .gitignore."
    )


def test_no_tracked_bytecode():
    """No compiled Python bytecode may be tracked."""
    bad = [
        path
        for path in tracked_files()
        if path.endswith(".pyc") or "__pycache__/" in path
    ]
    assert bad == [], (
        "Tracked Python bytecode found (machine-local, dirty on every run, "
        "#218). Untrack with `git rm --cached`: " + ", ".join(bad)
    )


def test_pycache_gitignored():
    """__pycache__/ contents must be gitignored, wherever they appear."""
    for probe in (
        "tst/test_suite/__pycache__/x.cpython-313.pyc",
        "vis/python/__pycache__/x.cpython-313.pyc",
    ):
        code, _ = git("check-ignore", "-q", probe)
        assert code == 0, (
            f"{probe} is not covered by .gitignore; pytest/imports will dirty "
            "the working tree on every run (#218). Ignore __pycache__/."
        )


def test_harness_log_path_unchanged():
    """The harness must still write its log where the wrappers expect it.

    run_test_suite.py removes ../tst/test_log.txt at startup and
    testutils logs to the same path; untracking the file (#218) must not
    move it.
    """
    import test_suite.testutils as testutils

    expected_tail = os.path.join("tst", "test_log.txt")
    assert testutils.LOG_FILE_PATH.endswith(expected_tail), (
        "testutils.LOG_FILE_PATH moved away from tst/test_log.txt: "
        f"{testutils.LOG_FILE_PATH}"
    )
    # The harness appends to this path exactly like run_command does; the
    # append must succeed and land the file where the wrappers expect it.
    # (run_test_suite.py deletes the log at startup, so it may not exist yet.)
    with open(testutils.LOG_FILE_PATH, "a") as log_file:
        log_file.write("repo-hygiene probe (#218)\n")
    assert os.path.exists(testutils.LOG_FILE_PATH), (
        "harness log append did not create "
        f"{testutils.LOG_FILE_PATH}"
    )
