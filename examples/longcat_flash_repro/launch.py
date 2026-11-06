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
"""Run the Git branch reproduction workflow with an existing compiled environment."""

import argparse
import ast
import hashlib
import io
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent


def save(path: Path, value: dict) -> None:
    """Write a structured receipt outside all timed intervals.

    Args:
        path: Artifact path to read or write.
        value: Complete nested value to inspect.
    """
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def check_files() -> None:
    """Verify committed reproduction sources and data before launching anything."""
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Reproduction source checksum mismatch: {name}")


def check_source(repo: Path, manifest: dict) -> None:
    """Require the exact source used by the reference experiment.

    Args:
        repo: Editable model checkout to validate.
        manifest: Recorded model baseline and source fingerprints.
    """
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", manifest["required_commit"], current], cwd=repo, check=False
    )
    if ancestry.returncode:
        raise RuntimeError(
            f"Checkout must contain model baseline {manifest['required_commit']}; HEAD is {current}. "
            + "Update the checkout explicitly before replay; this script never changes Git state."
        )
    failures = []
    for name, expected in manifest["source_sha256"].items():
        path = repo / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    if failures:
        raise RuntimeError(f"Source files differ from the reference: {failures[:12]}")


def host_architecture(machine: str = "") -> str:
    """Normalize the supported Linux host CPU architectures.

    Args:
        machine: Optional architecture value for validation or tests.
    """
    value = (machine or platform.machine()).lower()
    normalized = {"amd64": "x86_64", "arm64": "aarch64"}.get(value, value)
    if normalized not in {"aarch64", "x86_64"}:
        raise RuntimeError(f"Unsupported host CPU architecture: {value}")
    return normalized


def verify_elf_architecture(path: Path, architecture: str) -> None:
    """Reject a selected native host library built for another CPU architecture.

    Args:
        path: Selected native shared library.
        architecture: Normalized architecture of the current Python host.
    """
    expected = {"x86_64": 62, "aarch64": 183}[host_architecture(architecture)]
    with path.open("rb") as stream:
        header = stream.read(20)
    if len(header) != 20 or header[:4] != b"\x7fELF" or header[4:6] != bytes((2, 1)):
        raise RuntimeError(f"Expected an ELF64 little-endian host library: {path.name}")
    machine = int.from_bytes(header[18:20], "little")
    if machine != expected:
        raise RuntimeError(f"Native ELF architecture mismatch: {path.name}; host={architecture}, e_machine={machine}")


def render_python(text: str, replacements: dict) -> str:
    """Replace path string literals without interpreting them as shell or Python code.

    Args:
        text: Template content to render.
        replacements: Caller-provided path substitutions.
    """
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                value = None
            if isinstance(value, str):
                for key, replacement in replacements.items():
                    value = value.replace(key, replacement)
                token = tokenize.TokenInfo(token.type, repr(value), token.start, token.end, token.line)
        tokens.append(token)
    rendered = tokenize.untokenize(tokens)
    if re.search(r"__(REPO|OUTPUT|ASSETS|NATIVE_ROOT|PACKAGE)__", rendered):
        raise RuntimeError("An unresolved path token remains in a generated Python script")
    compile(rendered, "<rendered>", "exec")
    return rendered


def render_text(text: str, replacements: dict) -> str:
    """Replace YAML path values using JSON quoting, which is valid YAML.

    Args:
        text: Template content to render.
        replacements: Caller-provided path substitutions.
    """
    for key, replacement in replacements.items():
        # Templates use unquoted path scalars; quote complete token-bearing values.
        text = re.sub(
            r"(?m)^([^\n]*?: )([^\n]*" + re.escape(key) + r"[^\n]*)$",
            lambda match: match.group(1) + json.dumps(match.group(2).replace(key, replacement)),
            text,
        )
    return text


def validate_devices(value: Optional[str], count: int, option: str) -> str:
    """Validate an explicit device assignment without selecting host defaults.

    Args:
        value: Complete nested value to inspect.
        count: Required number of unique logical device IDs.
        option: CLI option reported in validation errors.
    """
    if value is None:
        raise ValueError(f"{option} requires {count} reserved logical NPU IDs")
    parts = value.split(",")
    if len(parts) != count or any(not part.strip().isdigit() for part in parts):
        raise ValueError(f"{option} requires exactly {count} nonnegative integer IDs")
    devices = [int(part.strip()) for part in parts]
    if len(set(devices)) != count:
        raise ValueError(f"{option} must not repeat device IDs")
    return ",".join(map(str, devices))


def precision_command(output: Path, worker: str, report: str) -> list[str]:
    """Return the unchanged original-width component precision command.

    Args:
        output: Worker output or fresh report directory.
        worker: Precision worker filename.
        report: Precision report directory name.
    """
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(output / worker),
        "--backend",
        "native",
        "--ep",
        "4",
        "--real",
        "32",
        "--zero",
        "16",
        "--tokens",
        "2048",
        "--initial-token-capacity",
        "24576",
        "--output",
        str(output / report),
    ]


def write_case_scripts(args: argparse.Namespace, repo: Path, output: Path) -> None:
    """Write quoted launch commands using only caller-supplied device assignments.

    Args:
        args: Parsed CLI settings or original call arguments.
        repo: Editable model checkout to validate.
        output: Worker output or fresh report directory.
    """
    prefix = "set -eo pipefail\ncd -- " + shlex.quote(str(repo)) + "\n"
    scripts = {
        "run_trial.sh": (args.component_devices, precision_command(output, "trial.py", "precision-t2048")),
        "run_edges.sh": (args.component_devices, precision_command(output, "edges.py", "precision-edges")),
        "run_formal_component.sh": (args.component_devices, [sys.executable, str(output / "component/controller.py")]),
        "run_formal_trainer.sh": (args.trainer_devices, [sys.executable, str(output / "trainer/controller.py")]),
    }
    for name, (devices, command) in scripts.items():
        if devices is None:
            content = "echo 'No trainer devices were configured; pass --trainer-devices' >&2\nexit 2\n"
        else:
            content = prefix + f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(devices)}\n" + shlex.join(command) + "\n"
        (output / name).write_text(content, encoding="utf-8")
    commands = [
        f"bash {shlex.quote(str(output / script))} > {shlex.quote(str(output / log))} 2>&1"
        for script, log in [("run_trial.sh", "precision-t2048.log"), ("run_edges.sh", "precision-edges.log")]
    ]
    (output / "run_precision.sh").write_text("set -eo pipefail\n" + "\n".join(commands) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace, repo: Path, native_root: Path) -> Path:
    """Generate a fresh report tree without touching the customer checkout.

    Args:
        args: Parsed CLI settings or original call arguments.
        repo: Editable model checkout to validate.
        native_root: Selected compiled payload root.
    """
    output = args.out.resolve()
    if output.is_relative_to(repo) or output.is_relative_to(ROOT):
        raise RuntimeError("Write runtime reports outside the source checkout and reproduction directory")
    if output.exists():
        raise RuntimeError(f"Output already exists; choose a new directory: {output}")
    output.mkdir(parents=True)
    replacements = {
        "__REPO__": str(repo),
        "__OUTPUT__": str(output),
        "__ASSETS__": str(ROOT / "assets"),
        "__NATIVE_ROOT__": str(native_root),
        "__PACKAGE__": str(ROOT),
    }
    for template in (ROOT / "templates").rglob("*"):
        if not template.is_file() or template.suffix not in {".py", ".yaml"}:
            continue
        relative = template.relative_to(ROOT / "templates")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = template.read_text(encoding="utf-8")
        target.write_text(
            render_python(text, replacements) if template.suffix == ".py" else render_text(text, replacements),
            encoding="utf-8",
        )
    write_case_scripts(args, repo, output)
    subprocess.run(
        [sys.executable, str(output / "freeze.py")],
        check=True,
        env=dict(
            os.environ,
            HP_REPRO_NATIVE_ROOT=str(native_root),
            HP_REPRO_COMPONENT_DEVICES=args.component_devices,
            HP_REPRO_TRAINER_DEVICES=args.trainer_devices or "",
        ),
    )
    save(
        output / "portable-launch.json",
        {
            "repo": str(repo),
            "python": sys.executable,
            "native_root": str(native_root),
            "package": str(ROOT),
            "phase": args.phase,
            "no_external_wait_launcher": True,
            "prepare_only": args.prepare_only,
            "component_devices": args.component_devices,
            "trainer_devices": args.trainer_devices,
        },
    )
    return output


def check_device_inventory(args: argparse.Namespace) -> None:
    """Reject unavailable device assignments before starting process groups.

    Args:
        args: Explicit component and optional Trainer device assignments.
    """
    raw = subprocess.check_output(["npu-smi", "info", "-m"], text=True)
    available = {int(logical) for logical in re.findall(r"^\s*\d+\s+\d+\s+(\d+)\s+\d+\s+Ascend910", raw, re.M)}
    if not available:
        raise RuntimeError("Unable to read the Ascend logical NPU inventory")
    selected = set(map(int, args.component_devices.split(",")))
    if args.trainer_devices:
        selected.update(map(int, args.trainer_devices.split(",")))
    missing = selected - available
    if missing:
        raise RuntimeError(
            f"Selected logical NPU IDs are unavailable: {sorted(missing)}; "
            f"host exposes {len(available)} logical NPUs. "
            "The full Trainer experiment requires 16; the component requires 4."
        )


def check_idle() -> None:
    """Fail immediately on occupied devices; scheduling stays with the customer."""
    result = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, check=True)
    if "Process id" not in result.stdout or "No running processes found" not in result.stdout:
        raise RuntimeError("Could not establish NPU process availability")
    pattern = r"^\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|\s*[^|]+\|\s*\d+\s*\|"
    if re.search(pattern, result.stdout, re.M):
        raise RuntimeError(
            "The node has active NPU processes. Reserve idle devices before launching; no wait/retry is used."
        )


def parse_arguments() -> argparse.Namespace:
    """Parse the caller environment and explicit device assignments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="A new absolute report directory")
    parser.add_argument("--python", help="Consumed by run.sh; uses this Python for every child")
    parser.add_argument("--cann-env", type=Path, required=True)
    parser.add_argument("--native-env", type=Path)
    parser.add_argument("--resolved-native-env", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--native-root", type=Path, help="Defaults to REPO/build/native/payload")
    parser.add_argument("--phase", choices=["all", "precision", "component", "trainer"], default="all")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Render/freeze only; no dependency install or NPU work"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Check existing dependencies and imports; no NPU work"
    )
    parser.add_argument("--component-devices", help="Four explicitly reserved logical NPU IDs, comma separated")
    parser.add_argument("--trainer-devices", help="Sixteen explicitly reserved logical NPU IDs, comma separated")
    args = parser.parse_args()
    if args.trainer_devices is not None:
        args.trainer_devices = validate_devices(args.trainer_devices, 16, "--trainer-devices")
    if not args.check_only:
        args.component_devices = validate_devices(args.component_devices, 4, "--component-devices")
        if args.phase in ("all", "trainer"):
            args.trainer_devices = validate_devices(args.trainer_devices, 16, "--trainer-devices")
            if not set(args.component_devices.split(",")).issubset(args.trainer_devices.split(",")):
                parser.error("--component-devices must be a subset of --trainer-devices")
    return args


def runtime_preflight(args: argparse.Namespace, repo: Path, native_root: Path, env: dict) -> str:
    """Check dependencies and native selection without running device kernels.

    Args:
        args: Parsed CLI settings or original call arguments.
        repo: Editable model checkout to validate.
        native_root: Selected compiled payload root.
        env: Child process environment.
    """
    audit = subprocess.run(
        [sys.executable, str(ROOT / "check_deps.py")],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if audit.returncode:
        print(audit.stdout, flush=True)
        print(audit.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(
            "Dependency audit failed; existing packages were preserved. "
            + "Install only the listed differences in the existing environment, preserving Torch/PTA."
        )
    probe = subprocess.check_output(
        [sys.executable, str(ROOT / "runtime_probe.py"), str(repo), str(native_root)],
        cwd=tempfile.gettempdir(),
        env=env,
        text=True,
    )
    print(probe, flush=True)
    if args.check_only:
        return probe
    if args.phase in ("all", "trainer"):
        parent = args.out.resolve().parent
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < 120 * 1024**3:
            raise RuntimeError("Full Trainer replay needs at least 120 GiB free for its shared checkpoint and reports")
    check_device_inventory(args)
    check_idle()
    return probe


def execute_experiment(args: argparse.Namespace, repo: Path, manifest: dict, output: Path, env: dict) -> None:
    """Run precision before freezing and timing the selected experiment.

    Args:
        args: Caller configuration and explicit devices.
        repo: Editable model checkout.
        manifest: Validated model baseline fingerprints.
        output: Fresh experiment output directory.
        env: Environment inherited by every experiment worker.
    """
    os.environ.update(env)
    subprocess.run(["bash", str(output / "run_precision.sh")], check=True, env=env)
    subprocess.run([sys.executable, str(output / "verify_module.py")], check=True, env=env)
    # Match the reference run: seal the loaded payload after precision, before formal timing.
    check_source(repo, manifest)
    subprocess.run([sys.executable, str(output / "freeze.py")], check=True, env=env)
    if args.phase == "all":
        subprocess.run([sys.executable, str(output / "pipeline.py")], check=True, env=env)
    elif args.phase in ("component", "trainer"):
        subprocess.run(["bash", str(output / f"run_formal_{args.phase}.sh")], check=True, env=env)
    if args.phase == "all":
        verdict = json.loads((output / "results.json").read_text(encoding="utf-8"))
        print(verdict["precision"]["status"], verdict["performance_verdict"], flush=True)
        if verdict["precision"]["status"] != "PRECISION_PASS" or verdict["performance_verdict"] != "PERFORMANCE_PASS":
            raise RuntimeError("Experiment completed with a failing verdict; all evidence is retained")
    print(f"COMPLETE {output}", flush=True)


def main() -> None:
    """Validate branch sources and dispatch the selected experiment phase."""
    args = parse_arguments()
    check_files()
    repo = args.repo.resolve()
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    check_source(repo, manifest)
    if args.out.exists() and not args.check_only:
        raise RuntimeError(f"Output already exists; choose a new directory: {args.out}")
    native_root = (args.native_root or repo / "build/native/payload").resolve()
    if not native_root.is_dir():
        raise RuntimeError(f"Compiled native root not found: {native_root}")
    if args.out.resolve().is_relative_to(repo) or args.out.resolve().is_relative_to(ROOT):
        raise RuntimeError("Write runtime reports outside the source checkout and reproduction directory")
    if not args.out.is_absolute():
        raise RuntimeError("--out must be an absolute path")
    env = os.environ.copy()
    env.update(
        HP_VALIDATION_REPO=str(repo),
        HP_REPRO_NATIVE_ROOT=str(native_root),
        HP_REPRO_DEPENDENCY_MODE="existing",
        HP_REPRO_COMPONENT_DEVICES=args.component_devices or "",
        HP_REPRO_TRAINER_DEVICES=args.trainer_devices or "",
        HCCL_DETERMINISTIC="true",
    )
    probe = None if args.prepare_only else runtime_preflight(args, repo, native_root, env)
    if args.check_only:
        return
    output = prepare(args, repo, native_root)
    if args.prepare_only:
        print(f"PREPARED_ONLY {output}", flush=True)
        return
    save(output / "runtime-preflight.json", json.loads(probe))
    execute_experiment(args, repo, manifest, output, env)


if __name__ == "__main__":
    main()
