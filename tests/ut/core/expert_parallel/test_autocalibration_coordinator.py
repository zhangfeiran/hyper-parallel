# Copyright 2026 Huawei Technologies Co., Ltd
"""Keep completed collection separate from calibration quality and runtime failures."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[4]


def load_coordinator() -> ModuleType:
    """Load the actual standalone coordinator without importing framework modules."""
    spec = importlib.util.spec_from_file_location('replica_coordinator_gate', ROOT / 'scripts/replica_autocalibration.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('rejected_phase', ['fit', 'publish'])
def test_quality_rejection_keeps_collection_and_never_reports_validated(rejected_phase: str) -> None:
    """A fitter quality exit stops publication with a distinct, machine-readable result."""
    coordinator = load_coordinator()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / 'shape.json'
        manifest.write_text(json.dumps(dict(ep_size=4)))
        output = root / 'output'
        phases = []

        def run(command: list[str], log: Path, timeout: int, peers: list[Path] | None = None) -> int:
            phase = log.stem
            phases.append(phase)
            if phase in ('collect', 'verify'):
                (output / phase).mkdir()
                for rank in range(4):
                    (output / phase / f'status-rank{rank}.json').write_text('{}')
            return 20 if phase == rejected_phase else 0

        environment = dict(NNODES='1', NODE_RANK='0', NUM_CARDS='4', MASTER_ADDR='127.0.0.1')
        with patch.dict(os.environ, environment), patch.object(coordinator, '_run', side_effect=run), \
                patch.object(signal, 'signal'), patch.object(sys, 'argv', [str(ROOT / 'scripts/replica_autocalibration.py'),
                    '--manifest', str(manifest), '--output', str(output)]):
            with pytest.raises(SystemExit) as error:
                coordinator.main()
        assert error.value.code == 20
        outcome = json.loads((output / 'outcome.json').read_text())
        assert outcome['status'] == 'rejected'
        assert outcome['collection_complete'] is True
        assert outcome['rejected_phase'] == rejected_phase
        assert not (output / 'models/catalog.json').exists()
        marker = json.loads((output / f'{rejected_phase}-done.json').read_text())
        assert marker['worker_exit_status'] == 20
        assert marker['quality_rejected'] is True
        assert phases[-1] == rejected_phase


def test_runtime_failure_is_not_converted_to_quality_rejection() -> None:
    """A real worker failure must still stop the job and retain its nonzero status."""
    coordinator = load_coordinator()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / 'shape.json'
        manifest.write_text(json.dumps(dict(ep_size=4)))
        output = root / 'output'
        environment = dict(NNODES='1', NODE_RANK='0', NUM_CARDS='4', MASTER_ADDR='127.0.0.1')
        with patch.dict(os.environ, environment), patch.object(coordinator, '_run', return_value=1), \
                patch.object(signal, 'signal'), patch.object(sys, 'argv', ['coordinator', '--manifest',
                                                                          str(manifest), '--output', str(output)]):
            with pytest.raises(RuntimeError, match='stage failed'):
                coordinator.main()
        assert not (output / 'outcome.json').exists()
        assert json.loads((output / 'nodes/node_0/terminal.json').read_text())['exit_status'] == 1
