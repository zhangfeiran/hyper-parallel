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
"""Check matched Mega boundaries, recompute observers and failure eligibility."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType

from examples.training_demo.deepseek_v41 import summarize_deepseek_v41_mega_ab as mega_summary
from examples.training_demo.deepseek_v41.deepseek_v41_mega_ab import ModuleTiming, ReferenceMhc, select_kernels
from examples.training_demo.deepseek_v41.summarize_deepseek_v41_mega_ab import module_statistics, summarize
from hyper_parallel.core.activation_memory import ckpt_wrapper
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc import DeepseekV41HyperMegaMhc
from tests.ut.auto_models.models.deepseek_v41.test_moe_timing import CpuEvent
from tests.ut.auto_models.models.deepseek_v41.test_host_recompute import _tiny_model


class TestMegaAb(unittest.TestCase):
    """Observers preserve real model gradients and compare the same mHC protocol."""

    def test_recipe_preserves_unrelated_replacements(self) -> None:
        """Switch only the requested Mega mHC/Gate replacements."""
        original = {'plan_overrides': [
            {'match': 'mhc', 'replace_module': {'_target_': 'example.DeepseekV41HyperMegaMhc'}},
            {'match': 'gate', 'replace_module': {'_target_': 'example.DeepseekV41MegaGate'}},
            {'match': 'other', 'replace_module': {'_target_': 'example.Other'}},
        ]}
        recipe = deepcopy(original)
        select_kernels(recipe, True)
        self.assertEqual(recipe, original)
        select_kernels(recipe, False)
        self.assertEqual(len(recipe['plan_overrides']), 2)
        self.assertTrue(recipe['plan_overrides'][0]['replace_module']['_target_'].endswith('.ReferenceMhc'))
        self.assertEqual(recipe['plan_overrides'][1], original['plan_overrides'][2])

    def test_real_decoder_recompute_and_all_group_gradients(self) -> None:
        """Exercise tuple-output mHC/Gate and nested model replay for 40 steps."""
        with TemporaryDirectory() as directory, patch.object(DefaultDeviceType, 'get_device_type', return_value='cpu'):
            model = _tiny_model(directory)
            for layer in model.model.layers:
                layer.attn_hc = ReferenceMhc(module=layer.attn_hc)
                layer.ffn_hc = ReferenceMhc(module=layer.ffn_hc)
            golden = deepcopy(model)
            for layer in golden.model.layers:
                layer.attn_hc = DeepseekV41HyperMegaMhc(module=layer.attn_hc)
                layer.ffn_hc = DeepseekV41HyperMegaMhc(module=layer.ffn_hc)
            # The real wrapper sets the same recompute context as distributed training.
            for index, layer in enumerate(model.model.layers):
                model.model.layers[index] = ckpt_wrapper(layer)
            timing = ModuleTiming(model, event_factory=CpuEvent)
            self.addCleanup(timing.close)
            inputs = torch.arange(16).view(1,16).remainder(31) + 1
            for step in range(40):
                model.zero_grad(set_to_none=True)
                model(input_ids=inputs, use_cache=False).logits.float().square().mean().backward()
                record = timing.snapshot()
                self.assertEqual(record['incomplete_intervals'], [])
                self.assertEqual({key:len(value) for key,value in record['module_groups'].items()},
                                 {'moe':4, 'mhc':8, 'gate':4})
                for modules in record['module_groups'].values():
                    for phases in modules.values():
                        self.assertEqual(set(phases), {'forward', 'recompute_forward', 'backward_to_input_grad'})
                        self.assertTrue(all(interval['calls'] == 1 for interval in phases.values()))
                if step == 0:
                    golden(input_ids=inputs, use_cache=False).logits.float().square().mean().backward()
                    for parameter, other in zip(model.parameters(), golden.parameters()):
                        if parameter.grad is not None:
                            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-6)
                timing.clear()
                self.assertFalse(timing.grad_hooks)

    def test_incomplete_run_has_no_ratios(self) -> None:
        """Keep missing or failed jobs out of the module speedup table."""
        with TemporaryDirectory() as directory:
            report = summarize(Path(directory), 8, 16, 40, 10)
        self.assertFalse(report['complete'])
        self.assertFalse(report['speedup_eligible'])
        self.assertEqual(report['module_speed_ratios'], {})

    def test_statistics_take_rank_max_after_module_sum(self) -> None:
        """Reject a sum of unrelated per-module rank maxima."""
        records=[]
        for rank in range(2):
            groups={}
            for group,count in (('moe',8),('mhc',16),('gate',8)):
                groups[group]={str(index):{phase:{'current_stream_ms':10.0 if index % 2 == rank else 1.0,
                                                  'calls':1,'completed_calls':1}
                                           for phase in ('forward','recompute_forward','backward_to_input_grad')}
                               for index in range(count)}
            records.append({'steps_detail':[{'step':1,'moe_timing':{
                'incomplete_intervals':[],'module_groups':groups}}]})
        result=module_statistics(records,0)
        self.assertEqual(result['gate']['forward']['mean_ms'],44)
        self.assertEqual(result['mhc']['forward_plus_backward']['mean_ms'],176)

    def test_completed_first_pair_survives_all_on_failure(self) -> None:
        """Keep the MegaMoe comparison available when the final case aborts."""
        signature = [{'initialization': {'seed': 42}, 'rows': [(1, 2, {'layer': 4096})]}]
        stats = {'moe': {'forward': {'mean_ms': 20.0, 'median_ms': 18.0}}}
        def case_result(_root: Path, _nodes: Path, _count: int, _cards: int, _steps: int,
                        _warmup: int, label: str) -> tuple[dict, list]:
            """Supply audited case records except for the injected final failure."""
            if label == 'push':
                raise RuntimeError('Injected all-on failure')
            return {'status': 'complete', 'mean_seconds': 2.0 if label == 'native' else 1.0,
                    'median_seconds': 2.0 if label == 'native' else 1.0,
                    'hbm_eligible': True, 'moe_timing': {'eligible': False}}, signature

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for label, mode in (('native', 'off'), ('moe_only', 'moe')):
                folder = root / 'packed' / label / 'rank0'
                folder.mkdir(parents=True)
                (folder / 'result.json').write_text(json.dumps({'mega_kernels': mode}))
            with patch.object(mega_summary, '_summarize_case', side_effect=case_result), \
                    patch.object(mega_summary, 'module_statistics', return_value={
                        **stats, 'mhc': stats['moe'], 'gate': stats['moe']}):
                report = summarize(root, 1, 1, 40, 10)
        self.assertFalse(report['complete'])
        self.assertEqual(set(report['comparisons']), {'off_vs_moe_only'})
        self.assertTrue(report['comparisons']['off_vs_moe_only']['performance_eligible'])
        self.assertEqual(report['comparisons']['off_vs_moe_only']['whole_step_mean_speed_ratio'], 2.0)
        self.assertTrue(report['module_speed_ratios']['off_vs_moe_only']['gate']['forward']['eligible'])
