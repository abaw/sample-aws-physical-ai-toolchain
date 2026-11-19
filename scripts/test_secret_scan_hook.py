"""
Checks the root pre-commit config runs the staged secret scan, and that the scan refuses a
staged AWS access key and passes a clean file. Stdlib only:
  pytest scripts/test_secret_scan_hook.py
or, with no pytest installed (the fast lane):
  python3 scripts/test_secret_scan_hook.py
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "secret-scan-staged.sh"
# Run from a git hook, GIT_DIR or GIT_INDEX_FILE would point the temp repo at this one.
ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def scan_staged(name, text):
    with tempfile.TemporaryDirectory() as tmp:
        git = ["git", "-C", tmp, "-c", "core.hooksPath=/dev/null"]
        subprocess.run([*git, "init", "-q"], check=True, env=ENV)
        (Path(tmp) / name).write_text(text)
        subprocess.run([*git, "add", name], check=True, env=ENV)
        return subprocess.run(["sh", str(SCRIPT)], cwd=tmp, env=ENV, capture_output=True, text=True)


def test_precommit_config_runs_secret_scan():
    config = (ROOT / ".pre-commit-config.yaml").read_text()
    assert re.search(r"^\s*-?\s*id:\s*secret-scan\b", config, re.MULTILINE), config
    assert re.search(r"^\s*entry:\s*sh scripts/secret-scan-staged\.sh\b", config, re.MULTILINE), config
    # Without it pre-commit passes file names, which the script reads as a base ref.
    assert re.search(r"^\s*pass_filenames:\s*false\b", config, re.MULTILINE), config
    assert SCRIPT.is_file()


def test_scan_refuses_staged_access_key():
    # Built at runtime so this file holds no key-shaped literal for the scan to refuse.
    key = "AKIA" + "Q" * 16
    result = scan_staged("leak.txt", f"aws_access_key_id = {key}\n")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "leak.txt:1: AWS access key" in result.stdout
    assert key not in result.stdout


def test_scan_passes_clean_file():
    result = scan_staged("clean.txt", "nothing secret here\n")
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} passed")
