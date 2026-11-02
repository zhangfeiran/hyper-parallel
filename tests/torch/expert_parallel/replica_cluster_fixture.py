"""Load actual saved layouts without replacing them with a load-only solution."""

from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.cost import ExpertReplicaCostModel
from hyper_parallel.core.expert_parallel.hot_replica.plan import ExpertExecutionPlan
from hyper_parallel.core.expert_parallel.hot_replica.planner import _build_expert_replica_plan


def plan_digest(plan: ExpertExecutionPlan) -> str:
    """Hash the immutable config/counts/slots/dispatch, independent of payload paths."""
    return hashlib.sha256(json.dumps(asdict(plan), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def local_routes(fixture: dict[str, Any], ep_rank: int) -> tuple[tuple[int, ...], ...]:
    """Reproduce exactly the saved deterministic TopK IDs on one source rank."""
    config, route = fixture['plan']['config'], fixture['route_spec']
    if not 0 <= ep_rank < config['ep_size']:
        raise ValueError('ep_rank is outside the fixed plan')
    tokens, top_k, experts = route['tokens'], route['top_k'], config['num_experts']
    fraction = route['hot_fraction'] / (2 if route['pattern'] == 'middle' else 1)
    count = min(tokens, math.ceil(tokens * fraction))
    result = tuple(tuple(range(top_k)) if index < count else
                   tuple((index * top_k + position) % experts for position in range(top_k))
                   for index in range(tokens))
    histogram = [0] * experts
    for row in result:
        if len(set(row)) != top_k:
            raise ValueError('Fixed route contains duplicate TopK IDs')
        for expert in row:
            histogram[expert] += 1
    if histogram != fixture['plan']['logical_counts'][ep_rank]:
        raise ValueError('Reconstructed token IDs do not match the actual saved source histogram')
    return result


def load_fixture(path: Path, verify_original_planner: bool = True) -> tuple[dict[str, Any], ExpertExecutionPlan]:
    """Validate the full original plan and optional exact original-policy replay.

    Execution must use the returned plan itself. The original cost model is
    provenance and a CPU oracle; it does not calibrate a changed native payload.
    """
    fixture = json.loads(path.read_text())
    if fixture['schema_version'] != 1:
        raise ValueError('Unsupported actual-plan fixture schema')
    saved = fixture['plan']
    plan = ExpertExecutionPlan(ExpertReplicaConfig(**saved['config']), saved['logical_counts'],
                               saved['slot_to_logical'], saved['dispatch_counts'])
    plan.validate()
    if plan_digest(plan) != fixture['plan_sha256']:
        raise ValueError('Actual-plan fixture checksum mismatch')
    for rank in range(plan.config.ep_size):
        local_routes(fixture, rank)
    if verify_original_planner:
        record = fixture['original_decision']
        options = dict(record['options'])
        options['_rank_nodes'] = tuple(options['_rank_nodes'])
        replay = _build_expert_replica_plan(plan.logical_counts, plan.config.replica_slots_per_rank,
                                           cost_model=ExpertReplicaCostModel(**record['model']), **options)
        if replay.slot_to_logical != plan.slot_to_logical or replay.dispatch_counts != plan.dispatch_counts:
            raise ValueError('Original policy did not reproduce the actual plan')
    return fixture, plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    verified = []
    for path in sorted(args.directory.glob('push-*.json')):
        fixture, plan = load_fixture(path)
        verified.append(dict(file=path.name, plan_sha256=plan_digest(plan), transfers=len(plan.transfers),
                             max_slots=max(plan.replica_slots_used), peak_rows=max(plan.destination_loads)))
    print(json.dumps(verified, indent=2))
