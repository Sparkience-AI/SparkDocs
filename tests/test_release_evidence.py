import json
from pathlib import Path

import pytest

from scripts.release_evidence import generate


def test_release_evidence_contains_hash_sbom_and_source_commit(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    output = tmp_path / "evidence"
    dist.mkdir()
    (dist / "sparkdocs.whl").write_bytes(b"wheel")

    generate(dist, output, "abc123")

    assert "sparkdocs.whl" in (output / "SHA256SUMS").read_text(encoding="utf-8")
    sbom = json.loads((output / "sbom.cdx.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert sbom["bomFormat"] == "CycloneDX"
    assert provenance["source_commit"] == "abc123"


def test_release_evidence_requires_artifacts_and_commit(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    with pytest.raises(ValueError, match="source commit"):
        generate(dist, tmp_path / "out", "")
    with pytest.raises(ValueError, match="no artifacts"):
        generate(dist, tmp_path / "out", "abc")
