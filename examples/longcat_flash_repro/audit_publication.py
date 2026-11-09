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
"""Check reproducibility sources for workstation identity and credential values."""

import argparse
import json
import re
from pathlib import Path

SOURCE_FILES = {
    ".gitignore",
    "PUBLICATION.md",
    "LICENSE",
    "THIRD_PARTY.md",
    "README.zh-CN.md",
    "INTEGRATION.zh-CN.md",
    "manifest.json",
    "requirements-no-torch.lock",
    "run.sh",
    "launch.py",
    "runtime_probe.py",
    "check_deps.py",
    "audit_publication.py",
    "checksums.py",
    "diagnose.py",
    "diagnose_probe.py",
    "diagnose_reports.py",
}
SOURCE_DIRECTORIES = {"assets", "templates", "reference", "validation"}
TEXT_SUFFIXES = {".py", ".sh", ".json", ".jsonl", ".yaml", ".md", ".txt", ".lock"}
PATTERNS = {
    "workstation_home_or_cache_path": re.compile(r"/(?:home|cache|Users)/[^\s\"<>]+"),
    "literal_network_address": re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])"),
    "credential_value": re.compile(
        r"(?i)(?:password|api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*[\"']" + r"[A-Za-z0-9_+/=-]{16,}"
    ),
    "credential_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b|\bAKIA[A-Z0-9]{16}\b"),
}
IDENTITY_FIELDS = {
    "hostname",
    "host_name",
    "device_serial",
    "serial_number",
    "sdid",
    "superpod_id",
    "pid",
    "launcher_pid",
}


def source_files(root: Path) -> list[Path]:
    """List the publication allowlist, excluding generated runtime directories.

    Args:
        root: Source root or owning launcher identity to inspect.
    """
    paths = [root / name for name in SOURCE_FILES if (root / name).is_file()]
    for name in SOURCE_DIRECTORIES:
        paths.extend(
            path
            for path in (root / name).rglob("*")
            if path.is_file() and path.suffix in TEXT_SUFFIXES and "__pycache__" not in path.parts
        )
    return sorted(paths)


def identity_fields(value: object, location: str = "") -> list[str]:
    """Return JSON fields containing concrete host or process identities.

    Args:
        value: Complete nested value to inspect.
        location: Relative field location in the structured artifact.
    """
    findings = []
    if isinstance(value, dict):
        for key, item in value.items():
            field = f"{location}.{key}"
            if key.lower() in IDENTITY_FIELDS and item not in (None, ""):
                if not isinstance(item, str) or not item.startswith("<"):
                    findings.append(field)
            findings.extend(identity_fields(item, field))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(identity_fields(item, f"{location}[{index}]"))
    return findings


def audit(root: Path) -> list[dict[str, object]]:
    """Report file-relative findings without reproducing sensitive values.

    Args:
        root: Source root or owning launcher identity to inspect.
    """
    findings = []
    for path in source_files(root):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            findings.append({"file": relative, "reason": "symlink"})
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({"file": relative, "line": text.count("\n", 0, match.start()) + 1, "reason": label})
        if path.suffix == ".json":
            for field in identity_fields(json.loads(text)):
                findings.append({"file": relative, "field": field, "reason": "concrete_identity"})
    return findings


def main() -> None:
    """Run a publication scan without reading devices or importing frameworks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    findings = audit(args.root.resolve())
    print(json.dumps({"status": "FAIL" if findings else "PASS", "findings": findings}, indent=2))
    if findings:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
