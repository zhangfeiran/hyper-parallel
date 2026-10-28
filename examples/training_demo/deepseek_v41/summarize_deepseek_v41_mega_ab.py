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
"""Audit three Mega configurations and report matched per-module timing ratios."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from examples.training_demo.deepseek_v41.summarize_deepseek_v41_cluster import (
    _compare, _summarize_case, _timing_stats,
)


def module_statistics(records: list[dict[str, Any]], warmup: int) -> dict[str, Any]:
    """Sum within each rank before taking rank maxima for each measured step."""
    rows = [record['steps_detail'][warmup:] for record in records]
    if not rows or not rows[0] or any(len(rank_rows) != len(rows[0]) for rank_rows in rows):
        raise ValueError('Missing or unequal measured windows')
    expected = {'moe': 8, 'mhc': 16, 'gate': 8}
    first = rows[0][0]['moe_timing']['module_groups']
    names = {group: sorted(first[group]) for group in expected}
    for group, count in expected.items():
        if len(names[group]) != count:
            raise ValueError(f'Expected {count} {group} modules, got {len(names[group])}')
    for rank_rows in rows:
        for row in rank_rows:
            timing = row['moe_timing']
            if timing['incomplete_intervals']:
                raise ValueError(f'Incomplete module intervals at step {row["step"]}')
            for group in expected:
                if sorted(timing['module_groups'][group]) != names[group]:
                    raise ValueError(f'Inconsistent {group} module names')
                for phases in timing['module_groups'][group].values():
                    for phase in ('forward', 'recompute_forward', 'backward_to_input_grad'):
                        value = phases[phase]
                        if (not math.isfinite(value['current_stream_ms']) or value['current_stream_ms'] < 0
                                or value['calls'] != 1):
                            raise ValueError(f'Invalid {group}/{phase} interval: {value}')
    output = {}
    phases = {'forward': ('forward',), 'recompute_forward': ('recompute_forward',),
              'backward_to_input_grad': ('backward_to_input_grad',),
              'forward_plus_backward': ('forward', 'backward_to_input_grad')}
    for group in expected:
        output[group] = {}
        for label, components in phases.items():
            values = [max(sum(rank_rows[index]['moe_timing']['module_groups'][group][name][phase]
                              ['current_stream_ms'] for name in names[group] for phase in components)
                          for rank_rows in rows) for index in range(len(rows[0]))]
            output[group][label] = _timing_stats(values)
    return output


def summarize(run_root: Path, nodes: int, cards_per_node: int, steps: int,
              warmup: int) -> dict[str, Any]:
    """Audit each case and retain eligible completed pairs if another case fails."""
    cases, fingerprints = {}, {}
    for label, mode in (('native', 'off'), ('moe_only', 'moe'), ('push', 'on')):
        try:
            case, signature = _summarize_case(
                run_root / 'packed', run_root / 'packed/nodes', nodes, cards_per_node,
                steps, warmup, label)
            records = [json.loads((run_root / f'packed/{label}/rank{rank}/result.json').read_text())
                       for rank in range(nodes * cards_per_node)]
            if any(record.get('mega_kernels') != mode for record in records):
                raise ValueError(f'Wrong Mega mode in {label}')
            case['module_timing'] = module_statistics(records, warmup)
            cases[label], fingerprints[label] = case, signature
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
            cases[label] = {'status': 'incomplete_or_failed', 'reason': f'{type(error).__name__}: {error}'}
    complete = all(case['status'] == 'complete' for case in cases.values())
    report = {'complete': complete, 'purpose': 'instrumented_three_mega_configurations',
              'scope': '8 text layers, all-off then MegaMoe-only then all-on',
              'cases': cases, 'comparisons': {}, 'module_speed_ratios': {}, 'speedup_eligible': False,
              'timing_boundary': 'Current-stream spans include waits; MoE includes Gate. Groups overlap. '
                                 'Backward may include replay; do not add replay again. '
                                 'mHC uses matched shifted post/predict/pre/RMSNorm boundaries.'}
    pairs = (('off_vs_moe_only', 'native', 'moe_only'),
             ('off_vs_all_on', 'native', 'push'),
             ('moe_only_vs_all_on', 'moe_only', 'push'))
    for name, baseline, candidate in pairs:
        if any(cases[label]['status'] != 'complete' for label in (baseline, candidate)):
            continue
        comparison = _compare({'native': cases[baseline], 'push': cases[candidate]},
                              {'native': fingerprints[baseline], 'push': fingerprints[candidate]}, 'push')
        comparison.update(baseline_case=baseline, candidate_case=candidate,
                          whole_step_mean_speed_ratio=(cases[baseline]['mean_seconds'] /
                                                       cases[candidate]['mean_seconds']),
                          whole_step_median_speed_ratio=cases[baseline]['median_seconds'] /
                          cases[candidate]['median_seconds'])
        report['comparisons'][name] = comparison
        if not comparison['initialization_match'] or not comparison['actual_token_rows_match']:
            continue
        ratios = {}
        for group in ('moe', 'mhc', 'gate'):
            ratios[group] = {}
            for phase, before in cases[baseline]['module_timing'][group].items():
                after = cases[candidate]['module_timing'][group][phase]
                ratios[group][phase] = {
                    'baseline_case': baseline, 'candidate_case': candidate,
                    'baseline_mean_ms': before['mean_ms'], 'candidate_mean_ms': after['mean_ms'],
                    'baseline_median_ms': before['median_ms'], 'candidate_median_ms': after['median_ms'],
                    'mean_speed_ratio': before['mean_ms'] / after['mean_ms'] if after['mean_ms'] > 0 else None,
                    'median_speed_ratio': before['median_ms'] / after['median_ms'] if after['median_ms'] > 0 else None,
                    'eligible': comparison['performance_eligible'],
                }
        report['module_speed_ratios'][name] = ratios
    report['speedup_eligible'] = complete and all(
        comparison['performance_eligible'] for comparison in report['comparisons'].values())
    return report


def main() -> None:
    """Write complete or failed evidence and a compact module ratio table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--cards-per-node', type=int, default=16)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--warmup-steps', type=int, default=10)
    args = parser.parse_args()
    report = summarize(args.run_root, args.nodes, args.cards_per_node, args.steps, args.warmup_steps)
    (args.run_root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    fields = ['comparison', 'module', 'phase', 'baseline_case', 'candidate_case',
              'baseline_mean_ms', 'candidate_mean_ms', 'baseline_median_ms', 'candidate_median_ms',
              'mean_speed_ratio', 'median_speed_ratio', 'eligible']
    with (args.run_root / 'mega_module_speedups.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for comparison, groups in report['module_speed_ratios'].items():
            for group, phases in groups.items():
                for phase, values in phases.items():
                    writer.writerow({'comparison': comparison, 'module': group, 'phase': phase, **values})
    print(json.dumps({'complete': report['complete'], 'speedup_eligible': report['speedup_eligible'],
                      'module_speed_ratios': report['module_speed_ratios']}))


if __name__ == '__main__':
    main()
