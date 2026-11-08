"""Generate checksums, a minimal CycloneDX SBOM, and local build provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(dist: Path, output: Path, source_commit: str) -> None:
    if not source_commit.strip():
        raise ValueError("A source commit is required for release provenance.")
    artifacts = sorted(path for path in dist.iterdir() if path.is_file())
    if not artifacts:
        raise ValueError("The distribution directory contains no artifacts.")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {path.name: _sha256(path) for path in artifacts}
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8"
    )
    components = sorted(
        (
            {
                "type": "library",
                "name": distribution.metadata["Name"],
                "version": distribution.version,
                "purl": f"pkg:pypi/{distribution.metadata['Name']}@{distribution.version}",
            }
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        ),
        key=lambda component: component["name"].lower(),
    )
    (output / "sbom.cdx.json").write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "version": 1,
                "components": components,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "artifacts": hashes,
                "builder": "scripts/release_evidence.py",
                "generated_at": datetime.now(UTC).isoformat(),
                "python": platform.python_version(),
                "source_commit": source_commit,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path, default=Path("release-evidence"))
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    generate(args.dist, args.output, args.source_commit)


if __name__ == "__main__":
    main()
