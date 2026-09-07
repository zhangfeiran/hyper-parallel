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
"""Analyze gradient waits in sixteen-layer EP8 Torch-NPU traces through exact device flows."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from statistics import mean, median
from typing import Any


def _nanoseconds(value: str | int | float) -> int:
    """Preserve decimal microsecond timestamps without epoch-sized floating point."""
    return int(Decimal(str(value)) * 1000)


def _contains(scope: dict[str, Any], host: dict[str, Any]) -> bool:
    """Require the same CPU thread as well as interval containment."""
    return ((scope['pid'], scope['tid']) == (host['pid'], host['tid'])
            and scope['start_ns'] <= host['start_ns'] <= scope['end_ns'])


def _communication_events(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    native: dict[str, Any],
    scopes: dict[str, dict[str, Any]],
    *,
    receive_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate a P2P wait from its eventual compute-stream join through exact scopes."""
    barriers = [device for device, host in joined if _contains(scopes['wait'], host)
                and device['name'] == 'EVENT_WAIT']
    streams = {(event['pid'], event['tid']) for event in barriers}
    if len(streams) != 1:
        raise ValueError('Non-unique P2P wait stream')
    stream = next(iter(streams))
    caller = (native['pid'], native['tid'])
    independent = stream != caller
    if independent and not receive_only:
        raise ValueError('Gradient send lost its native producer dependency')
    records = [device for device, host in joined if _contains(scopes['launch'], host)
               and (device['pid'], device['tid']) == stream and device['name'] == 'EVENT_RECORD']
    caller_barriers = ([device for device, host in joined
                        if _contains(scopes['reduction'], host) and not _contains(scopes['wait'], host)
                        and (device['pid'], device['tid']) == caller and device['name'] == 'EVENT_WAIT']
                       if independent else barriers)
    if not records or not caller_barriers:
        raise ValueError('Missing P2P ready record or caller-stream join')
    return records, barriers, caller_barriers


def _validate_stream_order(row: dict[str, Any]) -> None:
    """Require native production and P2P completion before owner consumption."""
    if ((not row['independent_receive'] and row['ready_ns'] < row['native_end_ns'])
            or row['p2p_wait_start_ns'] < row['ready_ns']
            or row['wait_start_ns'] < row['native_end_ns']
            or row['wait_end_ns'] < row['p2p_wait_end_ns']):
        raise ValueError('Invalid producer/communication/consumer stream order')
    if row['owner_start_ns'] is not None and row['owner_start_ns'] < row['wait_end_ns']:
        raise ValueError('Owner starts before P2P completion')


def _rank_timeline(path: Path, rank: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize one host's clock and join its device events to launch/wait scopes."""
    raw_roots = list(path.parent.glob(path.stem + '_*_ascend_pt'))
    if len(raw_roots) != 1:
        raise ValueError(f'Expected one raw artifact root for {path}: {raw_roots}')
    starts = list(raw_roots[0].glob('PROF_*/host/start_info'))
    if len(starts) != 1:
        raise ValueError(f'Expected one CANN clock calibration: {starts}')
    clock = json.loads(starts[0].read_text())
    offset = _nanoseconds(clock['collectionTimeBegin']) - int(clock['clockMonotonicRaw'])
    events = json.loads(path.read_text())
    for event in events:
        if 'ts' in event:
            event['start_ns'] = _nanoseconds(event['ts']) - offset
            event['end_ns'] = event['start_ns'] + _nanoseconds(event.get('dur', 0))
    flow_starts = {event['id']: event for event in events if event.get('cat') == 'async_npu' and event['ph'] == 's'}
    flow_finishes = {}
    for event in events:
        if event.get('cat') == 'async_npu' and event['ph'] == 'f':
            key = (event['pid'], event['tid'], event['start_ns'])
            if key in flow_finishes:
                raise ValueError(f'Duplicate device flow endpoint: {key}')
            flow_finishes[key] = event
    joined = []
    for event in events:
        if event.get('ph') != 'X' or not event.get('args', {}).get('Task Type'):
            continue
        finish = flow_finishes.get((event['pid'], event['tid'], event['start_ns']))
        if finish:
            joined.append((event, flow_starts[finish['id']]))

    def scopes(name: str) -> list[dict[str, Any]]:
        """Sort only complete CPU scopes carrying the exact requested annotation."""
        return sorted((event for event in events if event.get('ph') == 'X' and event.get('name') == name),
                      key=lambda event: event['start_ns'])

    reductions = scopes('megamoe::replica.ExpertReplicaManager.reduce_replica_gradients')
    launches = scopes('megamoe::replica._launch_transfers')
    waits = scopes('megamoe::replica._wait_handles')
    backwards = scopes('megamoe::function._launch_backward')
    gradients = [batch for batch in payload['batches'] if batch['phase'] == 'gradient_return']
    if not (len(reductions) == len(backwards) == len(gradients) == 16):
        raise ValueError(f'Expected 16 complete gradient calls on rank {rank}')
    rows = []
    for index, (reduction, backward, gradient) in enumerate(zip(reductions, backwards, gradients)):
        native = [device for device, host in joined if _contains(backward, host)
                  and device['name'] == 'aclnnHyperMegaMoeGrad_HyperMegaMoeGrad_HyperMegaMoeGrad']
        batch = [device for device, host in joined if _contains(reduction, host)
                 and device['name'] == 'batch_send_recvAicpuKernel']
        launch = [scope for scope in launches if _contains(reduction, scope)]
        wait = [scope for scope in waits if _contains(reduction, scope)]
        if not all(len(group) == 1 for group in (native, batch, launch, wait)):
            raise ValueError(f'Non-unique native/P2P launch/wait association: {rank} {index}')
        native, batch, launch, wait = native[0], batch[0], launch[0], wait[0]
        records, barriers, caller_barriers = _communication_events(
            joined, native, {'reduction': reduction, 'launch': launch, 'wait': wait},
            receive_only=bool(gradient['transfers']) and all(
                entry['operation'] == 'receive' for entry in gradient['transfers']
            ),
        )
        additions = [device for device, host in joined if _contains(reduction, host)
                     and device['args']['Task Type'].startswith('KERNEL')
                     and device['name'] != 'batch_send_recvAicpuKernel']
        row = {
            'backward_index': index, 'layer': 15 - index,
            'native_start_ns': native['start_ns'], 'native_end_ns': native['end_ns'],
            'batch_control_start_ns': batch['start_ns'], 'batch_control_end_ns': batch['end_ns'],
            'ready_ns': max(event['end_ns'] for event in records),
            'wait_start_ns': min(event['start_ns'] for event in caller_barriers),
            'wait_end_ns': max(event['end_ns'] for event in caller_barriers),
            'wait_task_count': len(caller_barriers), 'ready_record_count': len(records),
            'p2p_stream_id': barriers[0]['tid'], 'caller_stream_id': native['tid'],
            'independent_receive': barriers[0]['tid'] != native['tid'],
            'p2p_wait_start_ns': min(event['start_ns'] for event in barriers),
            'p2p_wait_end_ns': max(event['end_ns'] for event in barriers),
            'owner_start_ns': min((event['start_ns'] for event in additions), default=None),
            'owner_end_ns': max((event['end_ns'] for event in additions), default=None),
            'owner_kernel_names': [event['name'] for event in additions],
            'send_to': sorted({entry['peer_rank'] for entry in gradient['transfers'] if entry['operation'] == 'send'}),
            'receive_from': sorted({entry['peer_rank'] for entry in gradient['transfers']
                                    if entry['operation'] == 'receive'}),
        }
        _validate_stream_order(row)
        rows.append(row)
    return {'rank': rank, 'wall_offset_ns': offset, 'trace_events': len(events), 'rows': rows}


def main() -> None:
    """Print validated per-layer timings and neighborhood-readiness bounds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    run = json.loads(args.result.read_text())
    traces = run['diagnostic_trace']['per_rank']
    if run['configuration']['layers'] != 16 or [item['rank'] for item in traces] != list(range(8)):
        raise ValueError('analysis requires sixteen layers and all eight ranks in order')
    per_rank = [_rank_timeline(Path(item['path']), item['rank'], run['diagnostic_p2p']['per_rank'][item['rank']])
                for item in traces]
    for rank_data in per_rank:
        for row in rank_data['rows']:
            neighbors = sorted(set(row['send_to'] + row['receive_from']))
            neighbor_ready = max(per_rank[peer]['rows'][row['backward_index']]['ready_ns'] for peer in neighbors)
            row['last_neighbor_ready_ns'] = neighbor_ready
            row['late_neighbor_ns'] = max(0, neighbor_ready - row['wait_start_ns'])
            row['after_neighbors_ready_ns'] = row['wait_end_ns'] - max(row['wait_start_ns'], neighbor_ready)
            if row['after_neighbors_ready_ns'] < -10000:
                raise ValueError(f'Wait completes before peer readiness: {rank_data["rank"]} {row["layer"]}')
    origin = min(rank_data['rows'][0]['native_start_ns'] for rank_data in per_rank)
    summaries = []
    for rank_data in per_rank:
        rows = rank_data['rows']
        summaries.append({
            'rank': rank_data['rank'],
            'native_duration_ms': sum(row['native_end_ns'] - row['native_start_ns'] for row in rows) / 1e6,
            'ready_after_native_ms': sum(row['ready_ns'] - row['native_end_ns'] for row in rows) / 1e6,
            'wait_ms': sum(row['wait_end_ns'] - row['wait_start_ns'] for row in rows) / 1e6,
            'independent_receive_calls': sum(row['independent_receive'] for row in rows),
            'p2p_stream_wait_ms': sum(row['p2p_wait_end_ns'] - row['p2p_wait_start_ns'] for row in rows) / 1e6,
            'late_neighbor_ms': sum(row['late_neighbor_ns'] for row in rows) / 1e6,
            'after_neighbors_ready_ms': sum(row['after_neighbors_ready_ns'] for row in rows) / 1e6,
            'owner_kernels_ms': sum(row['owner_end_ns'] - row['owner_start_ns'] for row in rows
                                    if row['owner_start_ns'] is not None) / 1e6,
        })
        for row in rows:
            for key, value in list(row.items()):
                if (key.endswith('_ns') and key not in ('late_neighbor_ns', 'after_neighbors_ready_ns')
                        and value is not None):
                    row[key] -= origin
    spreads = [max(rank_data['rows'][i]['ready_ns'] for rank_data in per_rank)
               - min(rank_data['rows'][i]['ready_ns'] for rank_data in per_rank) for i in range(16)]
    print(json.dumps({'origin_monotonic_ns': origin, 'summary': summaries, 'per_rank': per_rank,
                      'ready_spread_median_ms': median(spreads) / 1e6,
                      'ready_spread_mean_ms': mean(spreads) / 1e6,
                      'ready_spread_max_ms': max(spreads) / 1e6}, indent=2))


if __name__ == '__main__':
    main()
