"""Shell checkout regressions; never execute installers or change global Git config."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _git(cwd, *args):
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_ATTR_NOSYSTEM="1")
    return subprocess.run(
        ["git", "-C", str(cwd), *args], env=env, check=True,
        capture_output=True, timeout=30,
    ).stdout


@pytest.fixture(scope="module")
def tracked_shells():
    if not shutil.which("git"):
        pytest.skip("Git is required to test tracked shell checkout portability")
    paths = _git(ROOT, "ls-files", "-z", "--", "*.sh").decode().split("\0")
    paths = [path for path in paths if path]
    assert paths, "No tracked shell scripts found"
    return paths


def _assert_lf(root, paths):
    for path in paths:
        contents = (root / path).read_bytes()
        assert b"\r\n" not in contents, f"CRLF found in {path}"
        assert b"\r" not in contents, f"Bare CR found in {path}"


@pytest.fixture(scope="module")
def autocrlf_checkout(tmp_path_factory, tracked_shells):
    scratch = tmp_path_factory.mktemp("shell-checkout")
    source = scratch / "source"
    source.mkdir()
    _git(scratch, "-c", "init.templateDir=", "init", "--quiet", str(source))
    _git(source, "config", "core.autocrlf", "false")
    _git(source, "config", "core.attributesFile", os.devnull)

    # Include attributes at every applicable directory, including a new unstaged
    # root policy, so this also works while developing the portability fix.
    attributes = {ROOT / ".gitattributes"}
    for path in tracked_shells:
        dest = source / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest)
        for parent in (ROOT / path).parents:
            if parent == ROOT:
                break
            attributes.add(parent / ".gitattributes")
    for path in attributes:
        if path.is_file():
            dest = source / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)

    _git(source, "add", "--all")
    _git(source, "-c", "user.name=EdgeForge portability test",
         "-c", "user.email=portability@example.invalid",
         "commit", "--quiet", "--no-gpg-sign", "-m", "Shell checkout fixture")

    checkout = scratch / "checkout"
    _git(scratch, "clone", "--quiet", "--no-local", "--no-checkout",
         str(source), str(checkout))
    _git(checkout, "config", "core.autocrlf", "true")
    _git(checkout, "config", "core.attributesFile", os.devnull)
    _git(checkout, "checkout", "--quiet", "--force", "HEAD")
    assert _git(checkout, "config", "--get", "core.autocrlf").strip() == b"true"
    return checkout


def test_tracked_shell_scripts_use_lf(tracked_shells):
    _assert_lf(ROOT, tracked_shells)


def test_shell_scripts_stay_lf_with_autocrlf_true(autocrlf_checkout, tracked_shells):
    _assert_lf(autocrlf_checkout, tracked_shells)


def test_autocrlf_shell_checkout_passes_bash_syntax(autocrlf_checkout, tracked_shells):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required for shell syntax validation")
    for path in tracked_shells:
        result = subprocess.run(
            [bash, "-n", path], cwd=autocrlf_checkout,
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"{path}: {result.stdout}{result.stderr}"
