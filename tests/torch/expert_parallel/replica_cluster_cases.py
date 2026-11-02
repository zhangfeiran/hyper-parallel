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
"""Framework-free stage definitions for coordinated replica cluster validation."""

from typing import Any

PHASES = ('fsdp', 'timing', 'diagnostic', 'causal', 'gate')
PREFIX = 'tests.torch.expert_parallel.'


def cluster_cases(phases: str, order: str = 'AB') -> list[dict[str, Any]]:
    """Create one deterministic table shared by every node, without allocating devices."""
    selected = phases.split(',')
    if not selected or len(set(selected)) != len(selected) or any(item not in PHASES for item in selected):
        raise ValueError(f'PHASES must contain distinct names from {PHASES}')
    if order not in ('AB', 'ABBA'):
        raise ValueError('ORDER must be AB or ABBA')
    cases = []

    def add(phase: str, label: str, worker: str, arguments: list[str], **changes: Any) -> None:
        if phase in selected:
            cases.append(dict(phase=phase, label=label, worker=PREFIX + worker, arguments=arguments,
                              planner='fixed', host=1, metadata=0, cost=1, requires_d2_gate=False) | changes)

    small = ['--ep-size', '64', '--source', '@SOURCE@', '--hidden', '512', '--intermediate', '256',
             '--tokens', '512', '--gas', '4', '--output', '@OUTPUT@', '--bootstrap-base', '@BOOTSTRAP@']
    for mode in ('push', 'pull'):
        add('gate', f'gate-d2-{mode}', '_replica_cluster_fsdp', small + [
            '--feature', 'metadata', '--mode', mode, '--policy', 'reshard'])
    for mode, policy in (('push', 'reshard'), ('push', 'retain'), ('pull', 'reshard')):
        add('fsdp', f'fsdp-c-{mode}-{policy}', '_replica_cluster_fsdp', small + [
            '--feature', 'residency', '--mode', mode, '--policy', policy])

    pure = ['--ep-size', '64', '--hidden', '5120', '--intermediate', '2304', '--tokens', '4096',
            '--top-k', '6', '--warmup', '6', '--iterations', '24', '--initial-capacity-factor', '1.25',
            '--route-file', '@BUNDLE@/routes.json', '--result-dir', '@OUTPUT@']
    add('timing', 'timing-native-b0', '_replica_cost_worker', pure + ['--backend', 'native', '--budget', '0'])
    controls = ('old-a', 'fixed-b', 'fixed-b2', 'old-a2') if order == 'ABBA' else ('old-a', 'fixed-b')
    for control in controls:
        for mode in ('push', 'pull'):
            for budget in (0, 1, 6):
                add('timing', f'timing-{control}-{mode}-b{budget}', '_replica_cost_worker',
                    pure + ['--backend', mode, '--budget', str(budget)], planner=control.split('-')[0])
    for mode, budget in (('push', 6), ('pull', 1)):
        args = pure + ['--backend', mode, '--budget', str(budget)]
        add('timing', f'timing-d1off-{mode}-b{budget}', '_replica_cost_worker', args, host=0)
        add('timing', f'timing-d2on-{mode}-b{budget}', '_replica_cost_worker', args,
            metadata=1, requires_d2_gate=True)
    add('timing', 'timing-cmiss-push-b6', '_replica_cost_worker',
        pure + ['--backend', 'push', '--budget', '6', '--force-residency-miss'])
    for variant in ('old', 'fixed'):
        add('diagnostic', f'diagnostic-{variant}', '_benchmark_replica_critical_path',
            ['--ep-size', '64', '--experts', '384', '--hidden', '5120', '--intermediate', '2304',
             '--top-k', '6', '--tokens', '4096', '--hot-fraction', '0.06', '--repeats', '8',
             '--observations-only', '--output', '@OUTPUT@'], planner=variant)
    for name, budget, fixture in (('old-b6', 6, 'push-b6-hot.json'),
                                 ('fixed-b6', 6, 'push-b6-hot-fixed.json'), ('b1', 1, 'push-b1-hot.json')):
        add('causal', f'causal-push-hot-{name}', '_replica_cluster_causal',
            ['--ep-size', '64', '--budget', str(budget), '--backend', 'push', '--warmup', '3',
             '--fixture', '@BUNDLE@/causal-fixtures/' + fixture, '--skew', 'strong', '--output', '@OUTPUT@',
             '--outer-capture', '--outer-ranks', '0', '16'], cost=0)
    for pattern, skew in (('middle', 'moderate'), ('hot', 'strong')):
        for budget in (0, 1):
            add('causal', f'causal-pull-{pattern}-b{budget}', '_replica_cluster_causal',
                ['--ep-size', '64', '--budget', str(budget), '--backend', 'pull', '--warmup', '3',
                 '--fixture', f'@BUNDLE@/causal-fixtures/push-b1-{pattern}.json', '--skew', skew,
                 '--dynamic-plan', '--output', '@OUTPUT@'])
    priority = {'fsdp': 0, 'timing': 1, 'diagnostic': 2, 'causal': 3, 'gate': 4}
    return sorted(cases, key=lambda case: 5 if case['requires_d2_gate'] else priority[case['phase']])
