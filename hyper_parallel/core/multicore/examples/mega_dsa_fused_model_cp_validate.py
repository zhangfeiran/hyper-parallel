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
"""Real BF16 CP attention against independent global stock LI/SFA/KL model execution."""

from __future__ import annotations

import argparse
import copy
import json
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _metadata
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import _loaded_library_identity
from hyper_parallel.core.multicore.examples.mega_dsa_local_training_oracle import projection_vjp_metrics
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import build_model_fixture, model_inputs
from hyper_parallel.core.multicore.examples.mega_dsa_model_validate import _collect, _StockReference
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import collective_validation_check
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout, CannDsaSelection
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import FusedDsaCpForwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.fused_model_boundary import FusedDsaCpAttention
from hyper_parallel.core.multicore.modules.mega_dsa.local_training import FusedDsaLocalCpTrainingProbe
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta, DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.model_boundary import CannDsaReferenceAttention
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import DsaWorkspaceSpec, MegaDsaWorkspace
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def model_fixture() -> nn.Module:
    """Construct the real HP projection boundary with Native-supported 64 index heads."""
    model = build_model_fixture(dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(712)
        model.indexer.wq_b = nn.Linear(64, 64*128, bias=False)
        model.indexer.weights_proj = nn.Linear(32, 64, bias=False)
    model.num_index_heads = 64
    model.config.index_n_heads = 64
    return model.bfloat16()


def _complete_indices(meta):
    indices = torch.full((meta.global_valid_queries, 2048), -1, dtype=torch.int32)
    for token in range(meta.global_valid_queries):
        start = meta.global_cu_seqlens[meta.sequence_position(token)[0]]
        indices[token, :token-start+1] = torch.arange(start, token+1, dtype=torch.int32)
    return indices


class _IndependentStockReference(_StockReference):
    """Stock SFA/KL with a fresh stock indexer selection, independent of the CP candidate."""

    def indexer(self, index_query: torch.Tensor, index_key: torch.Tensor,
                merge_weight: torch.Tensor) -> CannDsaSelection:
        """Compute and explicitly validate stock selection outside the measured training path."""
        indices = torch.ops.npu.npu_lightning_indexer(
            index_query.contiguous(), index_key[:, None].contiguous(), merge_weight.contiguous(),
            actual_seq_lengths_query=self.layout.length_tensor, actual_seq_lengths_key=self.layout.length_tensor,
            layout_query="TND", layout_key="TND", sparse_count=2048, sparse_mode=3, return_value=True)[0]
        self.last_selection = self.prepare_selection(self.layout.sequence_to_global_indices(indices).cpu())
        return self.last_selection


def _configure(model, mode):
    model.train(mode != "eval")
    model.freeze_dsa = mode == "freeze"
    model.dsa_loss_coeff = 0. if mode == "zero" else .3


def _metric(actual, expected):
    if actual is None or expected is None:
        return {"passed": actual is None and expected is None, "availability_only": True}
    return projection_vjp_metrics(actual, expected)


def _compare(peers, reference, meta, selection):
    fields = {name: None for name in reference}
    for name, expected in reference.items():
        values = [peer["values"][name] for peer in peers if peer["values"][name] is not None]
        if name in ("output", "hidden"):
            if values:
                value = torch.zeros_like(expected)
                for peer in peers:
                    observed = peer["values"][name]
                    if observed is not None:
                        value[:, list(peer["ids"])] = observed
                fields[name] = value
        elif values:
            fields[name] = sum(values)
    metrics = {name: _metric(fields[name], expected) for name, expected in reference.items()}
    observed_indices = torch.full_like(selection, -1)
    for peer in peers:
        observed_indices[list(peer["ids"])] = peer["indices"]
    exact_selection = all(sorted(row[row>=0].tolist()) == sorted(other[other>=0].tolist())
                          for row, other in zip(observed_indices, selection))
    return {"passed": exact_selection and all(item["passed"] for item in metrics.values()),
            "independent_stock_model": metrics, "selected_sets_equal": exact_selection,
            "global_tokens": meta.global_valid_queries}


def _sgd(model, gradients, peers, reference_model, reference_gradients):
    metrics = {}
    reference_parameters = dict(reference_model.named_parameters())
    for (name, parameter), gradient in zip(model.named_parameters(), gradients[1:]):
        active = any(peer["values"][name] is not None for peer in peers)
        if active:
            reduced = (torch.zeros_like(parameter, dtype=torch.float32) if gradient is None
                       else gradient.detach().float().contiguous())
            dist.all_reduce(reduced)
            parameter.grad = reduced.to(parameter.dtype)
        else:
            parameter.grad = None
    for parameter, gradient in zip(reference_model.parameters(), reference_gradients[1:]):
        parameter.grad = gradient
    torch.optim.SGD(model.parameters(), lr=.01, foreach=False).step()
    torch.optim.SGD(reference_model.parameters(), lr=.01, foreach=False).step()
    torch.npu.synchronize()
    for name, parameter in model.named_parameters():
        metrics[name] = _metric(parameter, reference_parameters[name])
    return {"passed": all(value["passed"] for value in metrics.values()), "learning_rate": .01,
            "updated_parameters": metrics}


def _case(report, workspace, backend, meta, template, mode, replay):
    device = backend.layout.device
    model_source, reference_source = (copy.deepcopy(template).to(device) for _ in range(2))
    for source in (model_source, reference_source):
        _configure(source, mode)
    model = FusedDsaCpAttention(module=model_source)
    reference_model = CannDsaReferenceAttention(module=reference_source)
    enabled = mode not in ("freeze", "zero", "eval")
    probe = FusedDsaLocalCpTrainingProbe(backend, normalization=DsaLossNormalization(13),
                                        loss_coeff=model.dsa_loss_coeff, mode="lm_plus_kl" if enabled else "lm_only",
                                        fused_kl=True, sparse_main_return=True, sparse_index_return=True)
    queries = backend.prepare_local_queries()
    global_meta = DsaBatchMeta.packed((3, 10))
    hidden_cpu, embeddings_cpu = model_inputs(global_meta, dtype=torch.bfloat16)
    ids = list(meta.kv_global_ids)
    hidden = hidden_cpu[:, ids].to(device).requires_grad_()
    embeddings = tuple(value[:, ids].to(device) for value in embeddings_cpu)
    reference_hidden = hidden_cpu.to(device).requires_grad_()
    reference_embeddings = tuple(value.to(device) for value in embeddings_cpu)
    reference_layout = CannDsaLayout(global_meta, device)
    reference_backend = _IndependentStockReference(reference_layout, attention_scale=model.scaling,
                                                    indices=_complete_indices(global_meta))
    auxiliary = 0. if mode == "aux_zero" else -2.5
    set_aux_loss_scale(torch.tensor(auxiliary, device=device))
    invocation = workspace.prepare(replace(meta, invocation=workspace.transport_epoch+1))
    captured = []
    original = backend.forward_local
    def _capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result
    def _forward(value):
        return model(value, position_embeddings=embeddings, dsa_training=probe,
                     dsa_invocation=invocation, dsa_query_layout=queries)[0]
    with patch.object(backend, "forward_local", side_effect=_capture):
        output = checkpoint(_forward, hidden, use_reentrant=False) if replay == "checkpoint" else _forward(hidden)
        objective = output.float().sum()*0 if mode == "kl_only" else output.float().square().sum()/(13*32)
        gradients = torch.autograd.grad(objective, (hidden, *model.parameters()), allow_unused=True,
                                        retain_graph=replay == "retained")
        if replay == "retained":
            _forward(hidden.detach())
            repeated = torch.autograd.grad(objective, (hidden, *model.parameters()), allow_unused=True)
            for actual, expected in zip(repeated, gradients):
                if expected is None:
                    if actual is not None:
                        raise RuntimeError("retained model backward changed gradient isolation")
                else:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference_output = reference_model(reference_hidden, position_embeddings=reference_embeddings,
                                       dsa_reference=reference_backend)[0]
    reference_objective = (reference_output.float().sum()*0 if mode == "kl_only"
                           else reference_output.float().square().sum()/(13*32))
    reference_gradients = torch.autograd.grad(reference_objective, (reference_hidden, *reference_model.parameters()),
                                             allow_unused=True)
    torch.npu.synchronize()
    loss = captured[0].kl_loss
    if loss is not None:
        loss = loss*probe.loss_coeff*probe.normalization.local_sum_scale
    values = _collect(model, hidden, output, loss, gradients=gradients)
    reference_values = _collect(reference_model, reference_hidden, reference_output, reference_backend.last_loss,
                                gradients=reference_gradients)
    payload = {"ids": ids, "values": values, "indices": captured[0].global_indices[:, 0].cpu()}
    peers = [None]*dist.get_world_size()
    dist.all_gather_object(peers, payload)
    proof = collective_validation_check(partial(_compare, peers, reference_values, global_meta,
                                                reference_backend.last_selection.to_global_indices().cpu()))
    optimizer = _sgd(model, gradients, peers, reference_model, reference_gradients)
    case = {"pattern": meta.layout_id, "mode": mode, "replay": replay, **proof, "sgd": optimizer,
            "native_phase_counts": [len(result.phase_traces) for result in captured]}
    case["passed"] = case["passed"] and optimizer["passed"]
    report["cases"].append(case)


def main() -> None:
    """Archive complete real model results against stock, retaining every numerical mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "running", "cases": [], "scope": "real HP BF16 attention CP versus global stock model",
              "full_model_acceptance": False, "independent_fp32_model_acceptance": False,
              "performance_acceptance": False}
    try:
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("hccl", timeout=timedelta(minutes=10))
        size, rank = dist.get_world_size(), dist.get_rank()
        report.update(rank=rank, cp_size=size)
        template = model_fixture()
        for pattern in ("strided", "empty"):
            root = SharedShmemRoot(torch.device("npu", int(os.environ.get("LOCAL_RANK", "0"))),
                                   root_group=dist.group.WORLD)
            meta = _metadata((3, 10), pattern, size, rank)
            meta = replace(meta, q_global_ids=meta.kv_global_ids)
            workspace = MegaDsaWorkspace(root, "real-model", DsaWorkspaceSpec(13, 13, 13, size))
            try:
                workspace.bind()
                meta = replace(meta, heap_generation=root.generation)
                backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=32,
                                                 attention_scale=template.scaling, schedule=MixedSfaSchedule(7))
                cases = [("joint", replay) for replay in ("plain", "retained", "checkpoint")]
                cases += [(mode, "plain") for mode in ("kl_only", "freeze", "zero", "eval", "aux_zero")]
                for mode, replay in cases:
                    collective_validation_check(partial(_case, report, workspace, backend, meta, template,
                                                        mode, replay))
            finally:
                workspace.close()
                root.close()
        accepted = all(case["passed"] for case in report["cases"])
        report.update(status="passed" if accepted else "numerical_mismatch",
                      loaded_library_sha256=_loaded_library_identity())
        if not accepted:
            raise RuntimeError("real model numerical mismatches recorded; inspect per-field metrics")
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        if report["status"] == "running":
            report["status"] = "error"
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
