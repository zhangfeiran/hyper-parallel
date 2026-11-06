# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Maintain the committed checksum list for the branch reproduction workflow."""

import argparse
import hashlib
from pathlib import Path

from audit_publication import audit, source_files

ROOT = Path(__file__).resolve().parent


def checksum_text(root: Path) -> str:
    """Compute checksums for the reviewed source and data allowlist.

    Args:
        root: Reproduction source directory.
    """
    return "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n" for path in source_files(root)
    )


def main() -> None:
    """Verify committed checksums, refreshing them only when explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refresh checksums after reviewed source changes")
    args = parser.parse_args()
    findings = audit(ROOT)
    if findings:
        raise RuntimeError(f"Publication scan failed: {findings}")
    expected = checksum_text(ROOT)
    path = ROOT / "SHA256SUMS"
    if args.refresh:
        path.write_text(expected, encoding="utf-8")
    elif path.read_text(encoding="utf-8") != expected:
        raise RuntimeError("Committed checksum list is stale; review changes before running --refresh")
    print("Publication scan and committed file checksums: PASS")


if __name__ == "__main__":
    main()
