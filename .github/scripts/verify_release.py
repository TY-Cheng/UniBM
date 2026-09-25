"""Verify an existing release bundle; never rebuild or change its distributions."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import urlopen


ROOT = Path("release-files")


def github(endpoint: str) -> dict:
    """Read repository metadata using the runner's existing GitHub authentication."""
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))


def verify_files(directory: Path, manifest: dict) -> None:
    """Require the exact distribution inventory and hashes recorded before upload."""
    expected = manifest["artifacts"]
    version = manifest["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Expected a stable three-part release version")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != set(expected):
        raise ValueError("Distribution inventory differs from the release manifest")
    for name, digest in expected.items():
        if not re.fullmatch(rf"unibm-{re.escape(version)}(?:-[\w.-]+\.whl|\.tar\.gz)", name):
            raise ValueError(f"Unexpected distribution filename: {name}")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"SHA256 mismatch: {name}")
    if (
        f"unibm-{version}-py3-none-any.whl" not in actual
        or f"unibm-{version}.tar.gz" not in actual
    ):
        raise ValueError("The portable wheel and source distribution are required")
    if not any(name.endswith(".whl") and "-py3-none-any" not in name for name in actual):
        raise ValueError("Native wheels are required")


def prepare(tag: str) -> None:
    """Download a tagged bundle and require successful CI at its exact source commit."""
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise ValueError("Expected an existing tag in vMAJOR.MINOR.PATCH format")
    repository = os.environ["GITHUB_REPOSITORY"]
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            tag,
            "--dir",
            str(ROOT),
            "--pattern",
            "release-manifest.json",
            "--pattern",
            "SHA256SUMS",
        ],
        check=True,
    )
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            tag,
            "--dir",
            str(ROOT / "dist"),
            "--pattern",
            "*.whl",
            "--pattern",
            "*.tar.gz",
        ],
        check=True,
    )
    manifest = json.loads((ROOT / "release-manifest.json").read_text())
    commit = github(f"repos/{repository}/commits/{tag}")["sha"]
    if manifest["version"] != tag[1:] or manifest["commit"] != commit:
        raise ValueError("Tag, source commit and manifest version must agree")
    verify_files(ROOT / "dist", manifest)
    sums = "".join(f"{digest}  {name}\n" for name, digest in sorted(manifest["artifacts"].items()))
    if (ROOT / "SHA256SUMS").read_text() != sums:
        raise ValueError("SHA256SUMS differs from the release manifest")
    for workflow in ("ci", "wheels", "docs"):
        run = github(
            f"repos/{repository}/actions/runs/{int(manifest['validation_runs'][workflow])}"
        )
        if (
            run["head_sha"] != commit
            or run["conclusion"] != "success"
            or run["status"] != "completed"
            or run["event"] != "push"
            or run["path"] != f".github/workflows/{workflow}.yml"
        ):
            raise ValueError(f"{workflow} has not passed for the tagged source commit")
    source = github(f"repos/{repository}/contents/tests/test_unibm_package_smoke.py?ref={commit}")
    (ROOT / "smoke.py").write_bytes(base64.b64decode(source["content"]))
    print(f"Verified {len(manifest['artifacts'])} distributions from {commit}")


def verify_index(index: str) -> None:
    """Check the uploaded inventory and select a registry-hosted Linux smoke-test wheel."""
    if index not in {"https://pypi.org", "https://test.pypi.org"}:
        raise ValueError("Expected PyPI or TestPyPI")
    manifest = json.loads((ROOT / "release-manifest.json").read_text())
    url = f"{index}/pypi/unibm/{manifest['version']}/json"
    for attempt in range(6):
        try:
            with urlopen(url, timeout=30) as response:
                files = json.load(response)["urls"]
            if {item["filename"]: item["digests"]["sha256"] for item in files} == manifest[
                "artifacts"
            ]:
                break
        except HTTPError as error:
            if error.code != 404:
                raise
        if attempt == 5:
            raise ValueError(f"Registry files do not match the verified bundle: {index}")
        time.sleep(10)
    wheel = next(
        item
        for item in files
        if "-cp314-cp314-manylinux" in item["filename"] and item["filename"].endswith("x86_64.whl")
    )
    (ROOT / "wheel-url.txt").write_text(wheel["url"])
    print(f"Verified all {len(files)} registry hashes at {index}")


if __name__ == "__main__":
    mode, value = sys.argv[1:]
    if mode == "prepare":
        prepare(value)
    elif mode == "verify-index":
        verify_index(value)
    else:
        raise ValueError(f"Unknown verification mode: {mode}")
