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
"""Small fixed-plan graph controls; no framework import or automatic model fitting."""

from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


GRAPH_FILES = tuple('hyper_parallel/core/multicore/modules/mega_moe/' + name
                    for name in ('module.py', 'plan.py', 'function.py', 'workspace.py', 'inactive_tasks.py'))


def weight_submission_summary(events: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count actual Host matrix submissions by lease phase without claiming service time."""
    phases = {event['lease_id']: event['metadata']['phase'] for event in events if event['kind'] == 'LeaseBegin'}
    puts = {phase: dict(count=0, bytes=0, generation_reuse_hits=0) for phase in ('forward', 'recompute', 'backward')}
    for event in events:
        phase = phases.get(event['lease_id'])
        if event['kind'] == 'WeightPrefetchSubmit':
            if phase not in puts:
                raise RuntimeError('Weight submission has no recorded lease phase')
            puts[phase]['count'] += 1
            puts[phase]['bytes'] += event['metadata']['bytes']
        if event['kind'] == 'WeightGenerationReuse' and event['metadata'].get('reuse_hit'):
            if phase not in puts:
                raise RuntimeError('Weight generation reuse has no recorded lease phase')
            puts[phase]['generation_reuse_hits'] += 1
    return puts


def schedule_cases(phases: str, order: str) -> list[dict[str, Any]]:
    """Capture five matched layouts in adjacent original/suffix graph controls."""
    if phases != 'schedule' or order != 'AB':
        raise ValueError('Schedule capture requires PHASES=schedule and ORDER=AB')
    cases = []
    workloads = (('push', 'hot', 1), ('push', 'hot', 6), ('push', 'middle', 6),
                 ('pull', 'hot', 1), ('pull', 'hot', 6))
    for mode, pattern, budget in workloads:
        pair = f'{mode}-{pattern}-b{budget}'
        for graph in ('original', 'suffix'):
            arguments = ['--ep-size', '64', '--backend', mode, '--budget', str(budget), '--warmup', '3',
                         '--fixture', f'@BUNDLE@/schedule-fixtures/{pair}.json',
                         '--skew', 'moderate' if pattern == 'middle' else 'strong',
                         '--force-residency-miss', '--output', '@OUTPUT@']
            arguments += ['--timing-warmup', '6', '--measure-iterations', '8']
            cases.append(dict(phase='schedule', label=f'schedule-{pair}-{graph}', pair=pair, graph=graph,
                              worker='tests.torch.expert_parallel._replica_cluster_causal', arguments=arguments,
                              planner='fixed', host=1, metadata=0, cost=0, requires_d2_gate=False,
                              scope=('Forced-normal-prefetch-miss fixed-plan diagnostics; no latency acceptance '
                                     'or automatic calibration installation'), diagnostic_pull=mode == 'pull'))
    return cases


def prepare_controls(repository: Path, bundle: Path, original: str = 'fc3c5040e',
                     control_source: Path | None = None) -> dict[str, Any]:
    """Save immutable source deltas and provenance without modifying the active repository."""
    if (bundle / 'READY').exists():
        raise ValueError('Do not modify an already released bundle')
    versions = {}
    for variant, revision in (('original', original), ('suffix', 'HEAD')):
        commit = subprocess.run(['git', '-C', str(repository), 'rev-parse', revision],
                                check=True, capture_output=True, text=True).stdout.strip()
        hashes = {}
        for relative in GRAPH_FILES:
            raw = subprocess.run(['git', '-C', str(repository), 'show', f'{commit}:{relative}'],
                                 check=True, capture_output=True).stdout
            if variant == 'original' and control_source is not None:
                control_raw = (control_source / relative).read_bytes()
                if control_raw != raw:
                    raise ValueError(f'CONTROL_SOURCE does not match the original graph: {relative}')
            destination = bundle / 'schedule-controls' / variant / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            hashes[relative] = hashlib.sha256(raw).hexdigest()
        versions[variant] = dict(source_commit=commit, files=hashes)
    path = bundle / 'schedule-controls/manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(versions, indent=2) + '\n')
    planner = 'hyper_parallel/core/expert_parallel/hot_replica/planner.py'
    raw = subprocess.run(['git', '-C', str(repository), 'show', f'HEAD:{planner}'],
                         check=True, capture_output=True).stdout
    destination = bundle / 'controls/planner-fixed.py'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return versions


def install_graph(source: Path, bundle: Path, variant: str,
                  control_source: Path | None = None) -> dict[str, Any]:
    """Change only the node-local clone between completed fresh-process stages."""
    if (source.resolve().is_relative_to(bundle.resolve()) or (source / '.git').exists()
            or control_source is not None and source.resolve().is_relative_to(control_source.resolve())
            or variant not in ('original', 'suffix')):
        raise ValueError('Graph controls require a node-local source clone and a declared variant')
    manifest = json.loads((bundle / 'schedule-controls/manifest.json').read_text())[variant]
    if set(manifest['files']) != set(GRAPH_FILES):
        raise ValueError('Graph source manifest does not match the bounded control file set')
    origin = control_source if variant == 'original' and control_source is not None else (
        bundle / 'schedule-controls' / variant)
    payloads = {relative: (origin / relative).read_bytes() for relative in GRAPH_FILES}
    for relative, raw in payloads.items():
        if hashlib.sha256(raw).hexdigest() != manifest['files'][relative]:
            raise ValueError(f'Graph control checksum mismatch: {variant}/{relative}')
    for relative, raw in payloads.items():
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return manifest | dict(control_source=str(origin))


def prepare_fixtures(parent: Path, bundle: Path) -> list[dict[str, Any]]:
    """Keep actual Push layouts and explicitly derive legal nonzero Pull diagnostics."""
    if (bundle / 'READY').exists():
        raise ValueError('Do not modify an already released bundle')
    output = bundle / 'schedule-fixtures'
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for mode, pattern, budget in (('push', 'hot', 1), ('push', 'hot', 6), ('push', 'middle', 6),
                                  ('pull', 'hot', 1), ('pull', 'hot', 6)):
        suffix = '-fixed' if budget == 6 and pattern == 'hot' else ''
        original = parent / f'push-b{budget}-{pattern}{suffix}.json'
        raw = original.read_bytes()
        fixture = deepcopy(json.loads(raw))
        model = fixture['model_configuration']
        if (model['ep_size'], model['hidden_size'], model['intermediate_size'], model['num_experts'],
                model['top_k'], fixture['route_spec']['tokens']) != (64, 5120, 2304, 384, 6, 4096):
            raise ValueError(f'Fixture is outside the declared calibration shape: {original}')
        if mode == 'pull':
            fixture['parent_original_decision'] = fixture['original_decision']
            options = dict(fixture['original_decision']['options'])
            options.update(target_load=None, receive_capacity=None, capacity_limit=64 * 4096 * 6)
            fixture['original_decision'] = dict(model=None, options=options)
            fixture['model_configuration']['backend'] = 'pull'
            fixture['observed_execution_signature'] = None
            fixture['scope'] = 'Explicit legal nonzero Pull diagnostic derived from Push; not observed Pull routing'
            fixture['diagnostic_derivation'] = dict(parent_file=original.name,
                parent_sha256=hashlib.sha256(raw).hexdigest(), plan_unchanged=True, production_policy_observed=False)
        destination = output / f'{mode}-{pattern}-b{budget}.json'
        destination.write_text(json.dumps(fixture, indent=2) + '\n')
        receipts.append(dict(file=destination.name, diagnostic_pull=mode == 'pull',
                             plan_sha256=fixture['plan_sha256'],
                             sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    (output / 'manifest.json').write_text(json.dumps(receipts, indent=2) + '\n')
    return receipts


def verify_pair(root: Path, pair: str, world: int) -> dict[str, Any]:
    """Require every rank's fixed inputs, plan and training hashes to match across graphs."""
    groups: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
    for rank in range(world):
        rows = [json.loads((root / f'schedule-{pair}-{variant}' / f'rank{rank}.json').read_text())
                for variant in ('original', 'suffix')]
        fields = ('fixed_route_sha256', 'actual_plan_sha256', 'tensor_sha256', 'native_fingerprint',
                  'ep_group_global_ranks', 'fixture_file_sha256')
        if any(rows[0][field] != rows[1][field] for field in fields):
            raise RuntimeError(f'Graph control changed fixed inputs, plan or training bytes: {pair}/rank{rank}')
        if any(row['rank'] != rank or row['terminal_status'] != 0 or not row['tensor_hashes_exact'] for row in rows):
            raise RuntimeError(f'Graph control did not finish its numerical gate: {pair}/rank{rank}')
        if not all(row['args'].get('force_residency_miss') for row in rows):
            raise RuntimeError(f'Graph control omitted the declared weight lifecycle: {pair}/rank{rank}')
        for variant, row in zip(('original', 'suffix'), rows):
            timing = row['pure_timing']
            samples = timing['samples']
            if (timing['observations'] or timing['profiler'] or timing['warmup'] != 6
                    or timing['iterations'] != 8 or [sample['sample_index'] for sample in samples] != list(range(8))
                    or any(not math.isfinite(sample['local_step_ms']) or sample['local_step_ms'] <= 0
                           for sample in samples)):
                raise RuntimeError(f'Uninstrumented timing window is incomplete: {pair}/{variant}/rank{rank}')
            if row['observations']['dropped_records'] or row['observations']['host_dropped']:
                raise RuntimeError(f'Graph control dropped observations: {pair}/rank{rank}')
            groups.setdefault((variant, tuple(row['ep_group_global_ranks'])), []).append(row)
    lifecycle = {}
    for (variant, members), rows in groups.items():
        if sorted(row['rank'] for row in rows) != list(members):
            raise RuntimeError(f'Incomplete EP group in weight lifecycle gate: {pair}/{members}')
        edges = rows[0]['fixed_plan_replay']['transfers']
        args = rows[0]['args']
        expected_bytes = edges * 3 * args['hidden'] * args['intermediate'] * 2
        actual = {phase: {field: sum(row['observations']['weight_submissions'][phase][field] for row in rows)
                          for field in ('count', 'bytes', 'generation_reuse_hits')}
                  for phase in ('forward', 'recompute', 'backward')}
        if edges < 1 or any(actual[phase]['count'] != 2 * edges or actual[phase]['bytes'] != expected_bytes
                            for phase in ('forward', 'recompute')):
            raise RuntimeError(f'Actual F/R matrix submissions do not match the forced-miss plan: {pair}/{variant}')
        if actual['backward']['count'] or actual['backward']['bytes']:
            raise RuntimeError(f'Backward unexpectedly submitted weight matrices: {pair}/{variant}')
        if any(row['observations']['weight_submissions']['backward']['generation_reuse_hits'] < 1 for row in rows):
            raise RuntimeError(f'R-to-B saved-generation reuse was not observed: {pair}/{variant}')
        lifecycle[f'{variant}-ep{members[0]}'] = actual
    return dict(pair=pair, ranks=world, training_hashes_exact=True, fixed_plan_exact=True,
                lifecycle='Normal prefetch forced miss; R-to-B saved-generation reuse retained',
                actual_weight_submissions=lifecycle,
                byte_scope='BF16 runtime.put submitted matrix bytes, not measured wire traffic or completion')


def main() -> None:
    """Prepare the small matrix before bundle checksums and READY are published."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--parent-fixtures', type=Path, required=True)
    parser.add_argument('--original', default='fc3c5040e')
    parser.add_argument('--control-source', type=Path)
    args = parser.parse_args()
    controls = prepare_controls(args.repository, args.bundle, args.original, args.control_source)
    fixtures = prepare_fixtures(args.parent_fixtures, args.bundle)
    print(json.dumps(dict(controls=controls, fixtures=fixtures, stages=10,
                          calibration_installed=False), indent=2))


if __name__ == '__main__':
    main()
