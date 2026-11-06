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
"""List every missing or mismatched reference dependency without installing packages."""

import json
import sys
from importlib import metadata
from pathlib import Path


def main() -> None:
    """Report all dependency differences before any NPU framework import."""
    missing, mismatched, matched = [], [], []
    lock = Path(__file__).resolve().parent / "requirements-no-torch.lock"
    for line in lock.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append({"name": name, "expected": expected})
            continue
        if actual != expected:
            mismatched.append({"name": name, "expected": expected, "actual": actual})
        else:
            matched.append(name)
    print(
        json.dumps(
            {
                "status": "FAIL" if missing or mismatched else "PASS",
                "missing": missing,
                "mismatched": mismatched,
                "matched_count": len(matched),
                "packages_installed_or_modified": False,
            },
            indent=2,
        )
    )
    if missing or mismatched:
        sys.exit(2)


if __name__ == "__main__":
    main()
