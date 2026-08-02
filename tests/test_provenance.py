# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from pathlib import Path

from flash_rwkv.provenance import IMPORTED_SOURCE_FAMILIES

ROOT = Path(__file__).parents[1]


def test_imported_source_families_have_fixed_revision_license_and_paths() -> None:
    names = {family.name for family in IMPORTED_SOURCE_FAMILIES}
    assert names == {
        "albatross-fused-infer",
        "flashkda-chunk",
        "rwkv-lm-pretrain",
        "vllm-recurrent",
    }
    paths: list[str] = []
    for family in IMPORTED_SOURCE_FAMILIES:
        assert re.fullmatch(r"[0-9a-f]{40}", family.revision)
        assert family.repository.startswith("https://github.com/")
        assert family.license in {"Apache-2.0", "MIT"}
        assert family.paths
        for relative_path in family.paths:
            source = ROOT / relative_path
            assert source.is_file(), relative_path
            contents = source.read_text(encoding="utf-8")
            assert f"SPDX-License-Identifier: {family.license}" in contents
            assert family.revision in contents
            paths.append(relative_path)
    assert len(paths) == len(set(paths))


def test_notice_covers_every_imported_source_family() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for family in IMPORTED_SOURCE_FAMILIES:
        assert family.revision in notice
        for relative_path in family.paths:
            assert f"- {relative_path}" in notice
