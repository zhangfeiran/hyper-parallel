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
"""Single-NPU P0 measurement against the FP32 oracle; no performance benchmark."""

import argparse
import json
import traceback
from pathlib import Path

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_backend_probe import (
    probe_environment,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaReference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    selected_kl_reference,
    sparse_attention_reference,
)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.detach().float().cpu().flatten().double()
    expected = expected.detach().float().cpu().flatten().double()
    difference = actual - expected
    actual_norm, expected_norm = actual.norm(), expected.norm()
    cosine = (actual @ expected) / (actual_norm * expected_norm).clamp_min(1e-30)
    return {
        "finite": bool(actual.isfinite().all() and expected.isfinite().all()),
        "relative_l2": float(difference.norm() / expected_norm.clamp_min(1e-30)),
        "cosine": float(cosine),
        "max_abs": float(difference.abs().max()),
        "reference_norm": float(expected_norm),
    }


def run_validation(report: dict) -> None:
    """Measure one packed BF16 case and all seven attention/indexer input gradients.

    The fixture covers complete legal histories (K=2048 exceeds both sequence
    lengths). Native and oracle use the same final selection; this does not
    establish Top-K equivalence for long contexts or ties. Metric thresholds
    must be calibrated separately before declaring BF16 numerical acceptance.
    """
    report["stage"] = "device_init"
    # The NPU extension is optional and loaded only for this explicitly requested device validator.
    __import__("torch_npu")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    report["device"] = torch.npu.get_device_name(0)
    report["environment"] = probe_environment()
    meta = DsaBatchMeta.packed((3, 5))
    layout = CannDsaLayout(meta, device)
    backend = CannDsaReference(layout, attention_scale=192**-0.5)
    generator = torch.Generator().manual_seed(20261007)
    shapes = ((8, 32, 512), (8, 512), (8, 32, 64), (8, 64), (8, 8, 128), (8, 128), (8, 8))
    cpu = tuple((torch.randn(shape, generator=generator) * 0.1).bfloat16() for shape in shapes)
    main_inputs = tuple(tensor.to(device).requires_grad_() for tensor in cpu[:4])
    index = tuple(tensor.to(device).requires_grad_() for tensor in cpu[4:])
    reference_main = tuple(tensor.float().requires_grad_() for tensor in cpu[:4])
    reference_index = tuple(tensor.float().requires_grad_() for tensor in cpu[4:])
    report["stage"] = "indexer"
    indices = backend.indexer(*index)
    torch.npu.synchronize()
    report["indices_first_slots"] = indices[:, :8].cpu().tolist()
    report["stage"] = "attention_forward"
    output, stats = backend.attention(*main_inputs, indices)
    torch.npu.synchronize()
    reference_output, reference_stats = sparse_attention_reference(
        *reference_main, indices.cpu(), meta, attention_scale=backend.attention_scale)
    report["output"] = _metrics(output, reference_output)
    report["lse"] = _metrics(stats.lse[0], reference_stats.lse)
    report["stage"] = "attention_backward"
    output.float().square().mean().backward()
    torch.npu.synchronize()
    reference_grad = torch.autograd.grad(reference_output.square().mean(), reference_main)
    native_grad = tuple(tensor.grad.detach().cpu().clone() for tensor in main_inputs)
    report["attention_gradients"] = {
        name: _metrics(actual, expected) for name, actual, expected in
        zip(("q_nope", "compressed_kv", "q_rope", "k_rope"), native_grad, reference_grad)
    }
    report["stage"] = "selected_kl"
    normalization = DsaLossNormalization(meta.global_valid_queries)
    loss = backend.kl_loss(*index, main_inputs, indices, stats, normalization=normalization, loss_coeff=0.3)
    (loss * 7).backward()
    torch.npu.synchronize()
    reference_loss = selected_kl_reference(
        *reference_index, *reference_main, indices.cpu(), meta, attention_scale=backend.attention_scale,
        normalization=normalization, loss_coeff=0.3)
    reference_index_grad = torch.autograd.grad(reference_loss * 7, reference_index)
    report["kl_loss"] = _metrics(loss, reference_loss)
    report["indexer_gradients"] = {
        name: _metrics(tensor.grad, expected) for name, tensor, expected in
        zip(("index_q", "index_k", "merge_weight"), index, reference_index_grad)
    }
    report["kl_main_gradient_isolation"] = all(torch.equal(tensor.grad.cpu(), before)
                                               for tensor, before in zip(main_inputs, native_grad))
    report["stage"] = "complete"
    report["status"] = "measured"
    report["numerical_acceptance"] = "pending primitive-specific BF16 threshold calibration"


def main() -> None:
    """Write a complete measurement or error report and preserve process failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "running", "cp_size": 1, "sequence_lengths": [3, 5], "sparse_count": 2048,
              "performance_measurement": False}
    try:
        run_validation(report)
    except Exception as error:
        report["status"] = "error"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
