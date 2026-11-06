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
"""Validate the full DSV replica recipe and its actual model adapter boundary."""

from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
import yaml
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from examples.training_demo.deepseek_v41 import benchmark_deepseek_v41_vlm as benchmark
from examples.training_demo.deepseek_v41 import deepseek_v41_replica_e2e as runner
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import deepseek_v41_ep_compute_fn

from tests.common.mark_utils import arg_mark


class TestReplicaRecipe(unittest.TestCase):
    """Exercise parsed recipes through real factories without distributed setup."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_budget_reaches_adapter_and_keeps_safe_training_policy(self) -> None:
        """Feature: Replica training recipe.
        Description: Resolve nine recipes and invoke their actual replica adapters on meta parameters.
        Expectation: B reaches native policy or Mega executor while B0 preserves omitted defaults.
        """
        with TemporaryDirectory() as directory:
            work = Path(directory)
            cases = product((("owner_ep", "push"), ("megamoe", "push"), ("megamoe", "pull")), (0, 1, 6))
            for (backend, mode), budget in cases:
                with self.subTest(backend=backend, mode=mode, budget=budget):
                    args = SimpleNamespace(
                        model_dir=directory, engram_assets=str(work / "assets.json"), experts=384,
                        modality="text", packing_mode="packed", steps=3, warmup_steps=1,
                        empty_cache_steps=1, random_init_seed=42, activation_checkpoint="full",
                        ep_size=64, edp_shard_size=2, data=str(work / "input.jsonl"),
                        validation_crop=True, num_hidden_layers=8, vision_num_hidden_layers=1,
                        full_model=False, engram_storage_backend="host", backend=backend,
                        dispatch_mode=mode, capacity_factor=None, replica_slots_per_rank=budget,
                        replica_transport="p2p" if backend == "owner_ep" else "shmem_signal_kernel_gradient_bands",
                    )
                    config = benchmark._recipe(args, work, 128)
                    recipe = yaml.safe_load((work / "recipe.yaml").read_text())
                    self.assertEqual(config.fsdp_config.edp_shard_size, 2)
                    self.assertEqual(recipe["fsdp_config"]["dp_shard_size"], 128)
                    self.assertEqual(recipe["activation_checkpoint"]["selection"]["layer_count"], 8)
                    self.assertEqual(recipe["model"]["engram_storage_backend"], "host")
                    self.assertFalse(recipe["checkpoint"]["save_ckpt"])
                    with torch.device("meta"):
                        source = DeepseekV4Experts(SimpleNamespace(
                            num_local_experts=384, hidden_size=16, intermediate_size=24,
                            hidden_act="silu", swiglu_limit=10.0))
                    if backend == "megamoe":
                        options = next(entry["replace"]["implementation"] for entry in recipe["plan_overrides"]
                                       if entry.get("match") == "model.layers.*.mlp.experts")
                        options = {key: value for key, value in options.items() if key != "_target_"}
                        if budget == 0:
                            self.assertNotIn("replica_slots_per_rank", options)
                            self.assertNotIn("replica_transport", options)
                        adapter = DeepseekV41MegaMoeExperts(module=source, **options)
                        with patch("hyper_parallel.core.multicore.MegaMoeExperts") as executor:
                            adapter.configure("ep-group", 64, 6)
                        self.assertEqual(executor.call_args.kwargs["replica_slots_per_rank"], budget)
                        if budget:
                            self.assertEqual(executor.call_args.kwargs["replica_transport"], args.replica_transport)
                    else:
                        options = next(entry["replace"]["implementation"] for entry in recipe["plan_overrides"]
                                       if entry.get("match") == "model.layers.*.mlp.forward")
                        options = {key: value for key, value in options.items() if key != "_target_"}
                        block = SimpleNamespace(gate=MagicMock(), experts=source, shared_experts=MagicMock(),
                                                is_hash=False, forward=lambda value: value)
                        mesh = MagicMock()
                        mesh.__getitem__.return_value.size.return_value = 64
                        compute = deepseek_v41_ep_compute_fn(
                            module=block, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=mesh, **options)
                        self.assertTrue(callable(compute))
                        if budget:
                            self.assertEqual(source.native_replica_policy["budget"], budget)
                        else:
                            self.assertNotIn("replica_slots_per_rank", options)
                            self.assertFalse(hasattr(source, "native_replica_policy"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_cli_validates_native_transport_and_budget(self) -> None:
        """Feature: Replica command parsing.
        Description: Reject unsupported native transports and invalid caps before recipe creation.
        Expectation: Native uses P2P; valid nonnegative caps are accepted for both dispatch modes.
        """
        base = ['--model-dir', '/unused', '--engram-assets', '/unused', '--data', '/unused',
                '--output', '/unused', '--backend', 'owner_ep']
        with patch.dict('os.environ', {'WORLD_SIZE': '128'}):
            for options in (['--replica-slots-per-rank', '-1'],
                            ['--replica-slots-per-rank', '1', '--replica-transport',
                             'shmem_signal_kernel_gradient_bands']):
                with self.subTest(options=options), self.assertRaises(SystemExit):
                    benchmark._arguments([*base, *options])
            self.assertEqual(benchmark._arguments([*base, '--replica-slots-per-rank', '6']).replica_slots_per_rank, 6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_pure_timing_registers_no_route_hooks(self) -> None:
        """Feature: Uninstrumented execution.
        Description: Disable route diagnostics on a real MoE-shaped module boundary.
        Expectation: The module has no diagnostic hooks and no route records are produced.
        """
        model = torch.nn.Module()
        model.layer = torch.nn.Module()
        model.layer.mlp = torch.nn.Identity()
        model.layer.mlp.gate = torch.nn.Identity()
        diagnostics = benchmark._RouteDiagnostics(model, enabled=False)
        self.addCleanup(diagnostics.close)
        self.assertEqual(diagnostics.hooks, [])
        model.layer.mlp(torch.ones(2, 4))
        self.assertEqual(diagnostics.token_rows, {})


class TestReplicaMatrix(unittest.TestCase):
    """Check stage selection and generated commands before cluster scheduling."""

    def test_no_table_load_controls_leave_heap_automatic(self) -> None:
        """Ordinary performance controls must not inherit a diagnostic fixed heap."""
        environment = dict(
            HYPER_ROOT='/source', RUN_ROOT='/results', MASTER_PORT_BASE='28600',
            SHMEM_HOST='127.0.0.1', MASTER_ADDR='127.0.0.1', NNODES='8', NODE_RANK='0',
            NUM_CARDS='16', TRAIN_ITERS='194', WARMUP_STEPS='5', MODEL_DIR='/model',
            ENGRAM='/engram.json', DATA='/data.jsonl', NATIVE_LIB='/native',
            DSV41_COMPLETE_GUEST_SEARCH_E2E='1')
        with patch.dict('os.environ', environment, clear=True):
            for backend, budget in (('native', 0), ('push', 0), ('push', 2), ('push', 6),
                                    ('pull', 0), ('pull', 2), ('pull', 6)):
                with self.subTest(backend=backend, budget=budget):
                    command = runner.command_for(runner.Stage(backend, budget), 0)
                    self.assertFalse(any(value.startswith('HYPER_PARALLEL_SHMEM_HEAP_SIZE=')
                                         for value in command))

    def test_schedule_matrix_has_three_baselines_and_four_adjacent_graph_pairs(self) -> None:
        """The whole-model schedule comparison preserves B0 and compares every changed cap."""
        matrix = runner.stages(schedule_comparison=True)
        self.assertEqual([stage.name for stage in matrix[:3]], ['native_b0', 'push_b0', 'pull_b0'])
        self.assertEqual(len(matrix), 11)
        self.assertEqual(len({stage.name for stage in matrix}), 11)
        for old, new in zip(matrix[3::2], matrix[4::2]):
            self.assertEqual((old.backend, old.budget), (new.backend, new.budget))
            self.assertEqual((old.graph, new.graph), ('original', 'suffix'))
            self.assertIn(old.budget, (1, 6))
        selected = runner.stages('push_b6_original,push_b6_suffix', schedule_comparison=True)
        self.assertEqual([stage.name for stage in selected], ['push_b6_original', 'push_b6_suffix'])
        with self.assertRaises(ValueError):
            runner.stages('push_b1', schedule_comparison=True)

    def test_complete_guest_arm_requires_both_stock_controls(self) -> None:
        """Do not launch the physical B2 candidate without matched B2 and B6."""
        selection = 'push_b0,push_b6,push_b2,push_b2_complete,native_b0'
        with self.assertRaises(ValueError):
            runner.stages(selection)
        with self.assertRaisesRegex(ValueError, 'requires stock Push B2 and B6'):
            runner.stages('push_b2,push_b2_complete', complete_guest_comparison=True)
        with self.assertRaisesRegex(ValueError, 'requires the ordinary no-table schedule'):
            runner.stages(selection, calibration_comparison=True, complete_guest_comparison=True)
        with self.assertRaisesRegex(ValueError, 'requires the same complete layout'):
            runner.stages('push_b2,push_b6,push_b2_complete_repack', complete_guest_comparison=True)
        matrix = runner.stages(selection, complete_guest_comparison=True)
        self.assertEqual(tuple(stage.name for stage in matrix), tuple(selection.split(',')))
        self.assertEqual((matrix[2].budget, matrix[3].budget), (2, 2))
        combined = runner.stages(selection + ',push_b2_complete_repack', complete_guest_comparison=True)
        self.assertEqual(combined[-1].budget, 2)
        with self.assertRaisesRegex(ValueError, 'requires stock Push B6'):
            runner.stages('push_b0,push_b6_repack', complete_guest_comparison=True)
        b6_pair = runner.stages('push_b6,push_b6_repack', complete_guest_comparison=True)
        self.assertEqual([(stage.budget, stage.policy) for stage in b6_pair], [(6, None), (6, 'repack')])
        with self.assertRaisesRegex(ValueError, 'requires stock Pull B2'):
            runner.stages('pull_b0,pull_b2_heap2g', complete_guest_comparison=True)
        heap_pair = runner.stages('pull_b2,pull_b2_heap2g', complete_guest_comparison=True)
        self.assertEqual([(stage.budget, stage.policy) for stage in heap_pair], [(2, None), (2, 'heap2g')])

    def test_selected_commands_parse_into_actual_training_recipe(self) -> None:
        """Feature: Cluster stage reachability.
        Description: Parse default stages and explicitly selected B2 commands through the real recipe resolver.
        Expectation: Defaults remain nine stages; B2 reaches the requested backend and safe training policy.
        """
        with TemporaryDirectory() as directory:
            environment = {
                'HYPER_ROOT': directory, 'RUN_ROOT': directory, 'MASTER_PORT_BASE': '28600',
                'SHMEM_HOST': '127.0.0.1', 'MASTER_ADDR': '127.0.0.1', 'NNODES': '8', 'NODE_RANK': '0',
                'NUM_CARDS': '16', 'TRAIN_ITERS': '20', 'WARMUP_STEPS': '5', 'MODEL_DIR': directory,
                'ENGRAM': str(Path(directory) / 'engram.json'), 'DATA': str(Path(directory) / 'data.jsonl'),
                'NATIVE_LIB': directory, 'WORLD_SIZE': '128',
                'REPLICA_TRANSPORT': 'shmem_signal_kernel_gradient_bands',
            }
            with patch.dict('os.environ', environment):
                selected = runner.stages()
                self.assertEqual(len(selected), 9)
                selected += runner.stages('native_b2,push_b2,pull_b2')
                for index, stage in enumerate(selected):
                    with self.subTest(stage=stage.name):
                        command = runner.command_for(stage, index)
                        argv = command[command.index('--full-model'):]
                        argv[0] = '--validation-crop'
                        args = benchmark._arguments(argv)
                        config = benchmark._recipe(args, Path(directory), 128)
                        self.assertEqual(args.replica_slots_per_rank, stage.budget)
                        self.assertEqual(args.backend, 'owner_ep' if stage.backend == 'native' else 'megamoe')
                        self.assertFalse(config.profiler.enable)
                        self.assertFalse(args.moe_timing)
                        self.assertTrue(args.disable_route_diagnostics)
                        self.assertFalse(args.record_input_identity)
                        self.assertIn('DSV41_CPU_PLACEMENT=numa', command)
                        self.assertEqual(config.fsdp_config.edp_shard_size, 2)
                self.assertEqual([stage.name for stage in runner.stages('push_b1,pull_b6')], ['push_b1', 'pull_b6'])
                for value in ('', 'push_b7', 'push_b1,push_b1', ' push_b1'):
                    with self.subTest(selection=value), self.assertRaises(ValueError):
                        runner.stages(value)
