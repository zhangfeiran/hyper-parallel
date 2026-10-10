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
"""Owner-run coverage, Q-only permutation and leased native CP submission contracts."""

import gc
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from weakref import ref

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import fused_cp
from hyper_parallel.core.multicore.modules.mega_dsa.cp_reference import DsaCpLayout
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_backward import FusedDsaCpBackwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.selected_requests import SELECTED_REQUESTS_MAGIC
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from tests.ut.core.multicore.shmem.test_consumer import SharedRootFixture


class TestQueryGather(unittest.TestCase):
    """Query-owned fields must all use Q storage ordering, including their owner gradients."""

    def test_query_fields_never_use_kv_order(self):
        """Exercise different Q/K storage permutations with an independent global-row oracle."""
        meta = DsaBatchMeta((0, 3), (2, 0, 1), (1, 2, 0), (0, 0, 0), (1, 2, 0))
        layout = DsaCpLayout(meta, "cpu")
        global_values = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        tensors = tuple((global_values[list(meta.q_global_ids)] * factor).requires_grad_() for factor in (1, 3))
        output = layout.gather_query_fields(tensors, ((2,), (2,)))
        for actual, factor in zip(output, (1, 3)):
            torch.testing.assert_close(actual, global_values * factor)
        coefficients = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        sum((value * coefficients).sum() for value in output).backward()
        for tensor in tensors:
            torch.testing.assert_close(tensor.grad, coefficients[list(meta.q_global_ids)])

    def test_runs_cover_global_ids_and_original_owner_offsets(self):
        """Expand runs independently and require exact owner/offset pairs for every destination row."""
        for owners, offsets in (((0, 0, 1, 1), (0, 1, 0, 1)),
                                ((0, 1, 0, 1), (1, 1, 0, 0)),
                                ((0, 0, 0, 0), (3, 2, 1, 0))):
            ids = tuple(index for index, owner in enumerate(owners) if owner == 0)
            meta = DsaBatchMeta((0, 4), ids, ids, owners, offsets, cp_ranks=(0, 1), root_pes=(0, 1))
            actual = {}
            for peer, source, destination, count in fused_cp.cp_owner_runs(meta):
                for step in range(count):
                    self.assertNotIn(destination + step, actual)
                    actual[destination + step] = (peer, source + step)
            self.assertEqual(actual, dict(enumerate(zip(owners, offsets))))


class TestFusedCpSubmission(SharedRootFixture):
    """Native transport cannot outlive ownership or inherit stale epochs from another adapter."""

    def setUp(self) -> None:
        """Bind one real workspace over mocked CPU SHMEM storage."""
        super().setUp()
        self.workspace = MegaDsaWorkspace(self.root, "cp-probe", DsaWorkspaceSpec(3, 3, 3, 1))
        self.consumers.append(self.workspace.consumer)
        self.addCleanup(self.workspace.close)
        self.workspace.bind()
        self.meta = replace(DsaBatchMeta.packed((3,)), q_global_ids=(2, 0, 1), kv_global_ids=(1, 2, 0),
                            heap_generation=self.root.generation)
        self.invocation = self.workspace.prepare(self.meta)

    def _backend(self):
        return fused_cp.FusedDsaCpForwardProbe(self.workspace, self.invocation, heads=32,
                                              attention_scale=0.1, schedule=MixedSfaSchedule(7))

    def _local_states(self, ids):
        query = torch.tensor(ids, dtype=torch.bfloat16)
        key = torch.tensor(self.meta.kv_global_ids, dtype=torch.bfloat16)
        main = (query[:, None, None].expand(-1, 32, 512).clone(),
                key[:, None].expand(-1, 512).clone(),
                query[:, None, None].expand(-1, 32, 64).clone(), key[:, None].expand(-1, 64).clone())
        index = (query[:, None, None].expand(-1, 64, 128).clone(),
                 key[:, None].expand(-1, 128).clone(), query[:, None].expand(-1, 64).clone())
        return main, index

    def test_local_forward_uses_real_tiles_and_owns_delayed_outputs(self):
        """No Q collective, fake empty row or reused saved buffers can satisfy local submission."""
        for ids in ((2, 0), ()):
            with self.subTest(ids=ids):
                backend = self._backend()
                backend.prepare_local_queries(ids)
                main, index = self._local_states(ids)
                epochs = []
                original_where = torch.where

                def _translate(*args):
                    if not ids:
                        raise AssertionError("empty query outputs require no device index arithmetic")
                    return original_where(*args)

                def _execute(*args):
                    self.assertIs(self.root.lease_owner, self.workspace.consumer)
                    self.assertEqual(len(args), 29)
                    self.assertEqual(args[0].shape[0], len(ids))
                    self.assertEqual(args[9].shape, (6, 20, 64))
                    self.assertEqual(args[10].numel(), 47226880 if ids else 0)
                    expected = torch.tensor(sorted(ids), dtype=torch.bfloat16)
                    for field in (0, 2, 4):
                        torch.testing.assert_close(args[field][:, 0, 0], expected)
                    torch.testing.assert_close(args[26], torch.tensor([len(ids)], dtype=torch.int32))
                    torch.testing.assert_close(args[27], torch.tensor(sorted(ids), dtype=torch.int64))
                    torch.testing.assert_close(args[28], torch.tensor(sorted(ids), dtype=torch.int64))
                    torch.testing.assert_close(self.workspace.buffers["source_compressed"][:, 0],
                                               torch.arange(3, dtype=torch.bfloat16))
                    epochs.append(int(args[19][2]))
                    args[13].fill_(-1)
                    args[13][:, 0, 0] = 0
                    args[14].fill_(0)
                    args[15].copy_(args[2])
                    args[16].fill_(0)
                    args[17].fill_(1)

                with patch.object(fused_cp, "_load_native"), \
                        patch.object(fused_cp.torch, "where", side_effect=_translate), \
                        patch.object(backend.layout, "gather_query_fields", side_effect=AssertionError("Q gather")), \
                        patch.object(torch.ops.hyper_parallel, "dsa_local_cp_forward_version", return_value=1,
                                     create=True), \
                        patch.object(torch.ops.hyper_parallel, "dsa_local_cp_forward_out", side_effect=_execute,
                                     create=True):
                    first = backend.forward_local(self.invocation, main, index)
                    later = self.workspace.prepare(replace(self.meta, invocation=1, layer=1, microbatch=1))
                    second = backend.forward_local(later, main, index)
                self.assertEqual(epochs[1], epochs[0] + 1)
                self.assertEqual(int(first._runtime_buffers[0][2]), epochs[0])
                self.assertEqual(int(second._runtime_buffers[0][2]), epochs[1])
                self.assertIsNot(first._runtime_buffers[0], second._runtime_buffers[0])
                self.assertEqual(first._runtime_buffers[1].numel(), 47226880 if ids else 0)
                self.assertEqual(first.saved.query_scope, "local")
                self.assertIsNone(first.kl_loss)
                first.saved.validate_versions()
                torch.testing.assert_close(first.output[:, 0, 0], torch.tensor(ids, dtype=torch.bfloat16))
                self.assertIsNot(first.saved.states[1], second.saved.states[1])
                self.assertIsNot(first.saved.indices, second.saved.indices)
                self.assertIsNone(self.root.lease_owner)
                saved_layout = backend.local_query_layout
                self.assertIs(first.saved.local_query_layout, saved_layout)
                self.assertIs(second.saved.local_query_layout, saved_layout)
                backend.prepare_local_queries(())
                first.saved.validate_versions()
                second.saved.validate_versions()
                self.assertIsNot(backend.local_query_layout, saved_layout)
                self.assertEqual(first.saved.local_query_layout.batch_meta.q_global_ids, ids)
                with patch.object(FusedDsaCpBackwardProbe, "check_native_support") as support:
                    with self.assertRaisesRegex(ValueError, "local backward adapter"):
                        FusedDsaCpBackwardProbe(backend).backward(first, torch.zeros_like(first.output))
                    support.assert_not_called()

    def test_local_saved_layout_rejects_mutation_and_wrong_tile(self):
        """Delayed consumers must reject changed native positions, lengths and row permutations."""
        backend = self._backend()
        backend.prepare_local_queries((2, 0))
        with patch.object(fused_cp, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_local_cp_forward_version", return_value=1, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_local_cp_forward_out", create=True):
            result = backend.forward_local(self.invocation, *self._local_states((2, 0)))
        saved = result.saved
        saved.validate_versions()
        with self.assertRaisesRegex(ValueError, "forward query layout"):
            replace(saved, local_query_layout=None).validate_versions()
        backend.prepare_local_queries(())
        with self.assertRaisesRegex(ValueError, "forward query layout"):
            replace(saved, local_query_layout=backend.local_query_layout).validate_versions()
        for name in ("positions", "query_lengths", "key_lengths", "restore_order", "global_ids_tensor"):
            for replacement in (False, True):
                layout = fused_cp.DsaLocalQueryLayout(saved.local_query_layout.batch_meta, "cpu")
                candidate = replace(saved, local_query_layout=layout)
                candidate.validate_versions()
                tensor = getattr(layout, name)
                if replacement:
                    setattr(layout, name, tensor.clone())
                else:
                    tensor.add_(0)
                with self.subTest(name=name, replacement=replacement), \
                        self.assertRaisesRegex(ValueError, "metadata was modified"):
                    candidate.validate_versions()

    def test_local_forward_rejects_changed_metadata_before_loading_or_epoch(self):
        """In-place edits and same-version replacements both invalidate native addresses."""
        for name in ("query_lengths", "positions", "global_ids_tensor", "selected_rows"):
            for replacement in (False, True):
                with self.subTest(name=name, replacement=replacement):
                    backend = self._backend()
                    backend.prepare_local_queries((2, 0))
                    owner = backend if name == "selected_rows" else backend.local_query_layout
                    tensor = getattr(owner, name)
                    if replacement:
                        setattr(owner, name, tensor.clone())
                    else:
                        tensor.add_(0)
                    epoch = self.workspace.transport_epoch
                    with patch.object(fused_cp, "_load_native") as load:
                        with self.assertRaises(ValueError):
                            backend.forward_local(self.invocation, *self._local_states((2, 0)))
                        load.assert_not_called()
                    self.assertEqual(self.workspace.transport_epoch, epoch)

    def test_local_forward_rejects_shape_dtype_and_grad_before_loading(self):
        """Admission distinguishes local Q from complete owner KV without relying on native guards."""
        backend = self._backend()
        backend.prepare_local_queries((2, 0))
        for field, mode in ((0, "dtype"), (1, "rows"), (6, "rows"), (2, "grad")):
            main, index = self._local_states((2, 0))
            values = list((*main, *index))
            values[field] = (values[field].float() if mode == "dtype" else values[field][:1]
                             if mode == "rows" else values[field].requires_grad_())
            with self.subTest(field=field, mode=mode), patch.object(fused_cp, "_load_native") as load:
                with self.assertRaises(ValueError):
                    backend.forward_local(self.invocation, tuple(values[:4]), tuple(values[4:]))
                load.assert_not_called()

    def test_explicit_prepared_layout_survives_reprepare_without_accepting_foreign_layout(self):
        """Checkpoint can replay an admitted old tile, but metadata equivalence cannot forge admission."""
        backend = self._backend()
        layout = backend.prepare_local_queries((2, 0))
        backend.prepare_local_queries(())
        seen = []

        def _execute(*args):
            self.assertIs(args[26], layout.query_lengths)
            self.assertIs(args[27], layout.positions)
            seen.append(args[0].shape[0])

        with patch.object(fused_cp, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_local_cp_forward_version", return_value=1, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_local_cp_forward_out", new=_execute, create=True):
            result = backend.forward_local(self.invocation, *self._local_states((2, 0)), query_layout=layout)
        self.assertEqual(seen, [2])
        self.assertIs(result.saved.local_query_layout, layout)
        foreign = fused_cp.DsaLocalQueryLayout(layout.batch_meta, "cpu")
        with patch.object(fused_cp, "_load_native") as load:
            with self.assertRaisesRegex(ValueError, "cold prepared query layout"):
                backend.forward_local(self.invocation, *self._local_states((2, 0)), query_layout=foreign)
            layout.positions.add_(0)
            with self.assertRaisesRegex(ValueError, "metadata was modified"):
                backend.forward_local(self.invocation, *self._local_states((2, 0)), query_layout=layout)
            load.assert_not_called()

    def test_backend_does_not_keep_retired_query_metadata_alive(self):
        """Long-running preparation must release cold tile tensors once no saved consumer remains."""
        backend = self._backend()
        old = backend.prepare_local_queries((2, 0))
        weak = ref(old)
        backend.prepare_local_queries(())
        self.assertIs(weak(), old)
        del old
        gc.collect()
        self.assertIsNone(weak())

    def test_epoch_requires_lease_and_survives_adapter_recreation(self):
        with self.assertRaisesRegex(RuntimeError, "active invocation"):
            self.workspace.next_transport_epoch(self.invocation)
        with self.workspace.lease(self.invocation, direction="forward"):
            self.assertEqual(self.workspace.next_transport_epoch(self.invocation), 1)
        self._backend()
        with self.workspace.lease(self.invocation, direction="backward"):
            self.assertEqual(self.workspace.next_transport_epoch(self.invocation), 2)
            foreign = self.workspace.prepare(replace(self.meta, invocation=1))
            with self.assertRaisesRegex(RuntimeError, "active invocation"):
                self.workspace.next_transport_epoch(foreign)

    def test_local_order_changes_reject_before_collective_or_native_dispatch(self):
        """Global protocol signatures do not substitute for this rank's prepared storage permutation."""
        backend = self._backend()
        for field in ("q_global_ids", "kv_global_ids"):
            changed = self.workspace.prepare(replace(self.meta, **{field: (0, 1, 2)}))
            with self.subTest(field=field), patch.object(fused_cp, "_load_native") as load:
                with self.assertRaisesRegex(ValueError, "local Q/KV storage orders"):
                    backend.forward(changed, (), ())
                load.assert_not_called()

    def test_query_gather_publication_and_native_launch_share_lease(self):
        backend = self._backend()
        query_ids = torch.tensor(self.meta.q_global_ids, dtype=torch.bfloat16)
        key_ids = torch.tensor(self.meta.kv_global_ids, dtype=torch.bfloat16)
        main = (query_ids[:, None, None].expand(3, 32, 512).clone(),
                key_ids[:, None].expand(3, 512).clone(), torch.zeros(3, 32, 64, dtype=torch.bfloat16),
                key_ids[:, None].expand(3, 64).clone())
        index = (torch.zeros(3, 64, 128, dtype=torch.bfloat16),
                 key_ids[:, None].expand(3, 128).clone(), torch.zeros(3, 64, dtype=torch.bfloat16))
        epochs = []

        def _execute(*args):
            self.assertIs(self.root.lease_owner, self.workspace.consumer)
            self.assertTrue(all(tensor.is_contiguous() for tensor in args[:11]))
            self.assertIs(args[17], self.workspace.arena)
            metadata = args[18]
            epochs.append(int(metadata[2]))
            torch.testing.assert_close(self.workspace.buffers["source_compressed"][:, 0],
                                       torch.arange(3, dtype=torch.bfloat16))
            for position in (1, 3, 5):
                args[position].fill_(1)
            args[12].fill_(-1)
            args[12][:, 0, 0] = 0
            args[13].fill_(0)
            args[14].copy_(args[2])
            args[15].fill_(0)
            args[16].fill_(1)

        with patch.object(fused_cp, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_forward_version", return_value=2, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_cp_forward_out", side_effect=_execute, create=True):
            first = backend.forward(self.invocation, main, index)
            second = self._backend().forward(self.invocation, main, index)
        self.assertEqual(epochs, [1, 2])
        self.assertIsNone(self.root.lease_owner)
        torch.testing.assert_close(first.output[:, 0, 0], query_ids)
        self.assertNotEqual(first.transport_trace.data_ptr(), second.transport_trace.data_ptr())
        self.assertNotEqual(first.global_keys[0].data_ptr(), second.global_keys[0].data_ptr())

    def test_transport_evidence_rejects_incomplete_reads_and_unordered_progress(self):
        """Reject corrupted ready/ACK, byte counts and intervals independently of native execution."""
        backend = self._backend()
        phases = []
        for epoch in (1, 2, 3):
            trace = torch.zeros(20, 64, dtype=torch.int64)
            for group in range(7):
                tasks = tuple(range(group, 20, 7))
                trace[group, 0] = tasks[-1]
                for offset in (0, 16, 32):
                    trace[group, offset + 1:offset + 4] = torch.tensor(
                        [len(tasks), sum(task + 1 for task in tasks), tasks[-1]])
                    trace[group, offset + 6] = epoch
                    if epoch == 1:
                        trace[group, offset + 7:offset + 10] = torch.tensor([1, 150, 300])
            trace[7, [6, 21, 22, 24, 38]] = epoch
            trace[7, 25] = 21
            phases.append(trace)
        transport = torch.tensor([5, 5, 1, 1, 768, 3456, 0, 0, 100, 120, 200, 250,
                                  0, 21] + [0] * 18, dtype=torch.int64)
        empty = torch.empty(0)
        result = fused_cp.FusedCpForwardResult(empty, empty, empty, empty, empty, (), (), transport, 5)
        evidence = backend.validate_trace(result, tuple(phases), transport, require_overlap=True)
        self.assertEqual(len(evidence["compute_members_overlapping_transfer"]), 21)
        for word, value in ((0, 4), (1, 4), (2, 0), (3, 0), (4, 767), (5, 3455),
                            (6, 1), (7, 1), (8, 0), (9, 99), (10, 119), (11, 200),
                            (12, 1), (13, 20), (14, 1), (31, 1)):
            broken = transport.clone()
            broken[word] = value
            with self.subTest(word=word), self.assertRaises(ValueError):
                backend.validate_trace(result, tuple(phases), broken, require_overlap=True)
        phases[0][0, 7] = 0
        with self.assertRaisesRegex(ValueError, "compute interval"):
            backend.validate_trace(result, tuple(phases), transport, require_overlap=True)
        phases[0][0, 7] = 1
        no_overlap = transport.clone()
        no_overlap[10:12] = torch.tensor([400, 450])
        backend.validate_trace(result, tuple(phases), no_overlap, require_overlap=False)
        with self.assertRaisesRegex(ValueError, "transfer overlap"):
            backend.validate_trace(result, tuple(phases), no_overlap, require_overlap=True)

    def test_selected_transport_accounts_only_for_admitted_keys_and_published_counts(self):
        """Device counts determine main bytes, and descriptor publication precedes the main pull."""
        backend = self._backend()
        phases = []
        for epoch in range(1, 10):
            trace = torch.zeros(20, 64, dtype=torch.int64)
            for group in range(7):
                tasks = tuple(range(group, 20, 7))
                trace[group, 0] = tasks[-1]
                for offset in (0, 16, 32):
                    trace[group, offset + 1:offset + 4] = torch.tensor(
                        [len(tasks), sum(task + 1 for task in tasks), tasks[-1]])
                    trace[group, offset + 6] = epoch
                    if epoch == 1:
                        trace[group, offset + 7:offset + 10] = torch.tensor([1, 150, 180])
            trace[7, [6, 21, 22, 24, 38]] = epoch
            trace[7, 25] = 21
            phases.append(trace)
        indices = torch.tensor([[[0, -1]], [[0, -1]], [[2, -1]]], dtype=torch.int32)
        requests = torch.tensor([[0, 0, 0, 1], [0, 2, 2, 1], [-1, -1, -1, -1]], dtype=torch.int64)
        membership = torch.tensor([2, 0, 1, 0, 0, 0, 0, 0], dtype=torch.int32)
        counts = torch.tensor([SELECTED_REQUESTS_MAGIC, 5, 2, 2, 3, 0, 0, 0, 190, 195] + [0] * 6)
        transport = torch.tensor([5, 5, 1, 1, 768, 2304, 0, 0, 100, 120, 200, 250] + [0] * 20)
        empty = torch.empty(0)
        result = fused_cp.FusedCpForwardResult(empty, empty, empty, empty, empty, (), (), transport, 5,
                                              saved=SimpleNamespace(batch_meta=self.meta, query_scope="global"),
                                              kl_loss=torch.tensor(1.),
                                              selected_requests=(requests, counts, membership))
        snapshots = (requests, counts, membership, indices)
        evidence = backend.validate_trace(result, tuple(phases), transport, require_overlap=False,
                                           selected_snapshots=snapshots)
        self.assertEqual(evidence["main_bytes"], 2304)
        self.assertEqual(evidence["selected_requests"]["request_count"], 2)
        self.assertEqual(evidence["compute_members_overlapping_transfer"], [])
        with self.assertRaisesRegex(ValueError, "explicit CPU"):
            backend.validate_trace(result, tuple(phases), transport, require_overlap=False)
        for word, value in ((5, 3456), (13, 21), (10, 194)):
            broken = transport.clone()
            broken[word] = value
            with self.subTest(word=word), self.assertRaises(ValueError):
                backend.validate_trace(result, tuple(phases), broken, require_overlap=False,
                                       selected_snapshots=snapshots)


class TestLocalTraceScope(SharedRootFixture):
    """Local evidence must describe the saved query tile in a mixed-length batch."""

    def test_saved_tile_controls_long_history_and_selected_membership(self):
        """An unqueried long sequence needs no LD, while a queried one must prove its merge."""
        tokens = 2116
        workspace = MegaDsaWorkspace(self.root, "local-trace", DsaWorkspaceSpec(tokens, tokens, tokens, 1))
        self.consumers.append(workspace.consumer)
        self.addCleanup(workspace.close)
        workspace.bind()
        meta = replace(DsaBatchMeta.packed((3, 2113)), heap_generation=self.root.generation)
        backend = fused_cp.FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=32,
                                                 attention_scale=0.1, schedule=MixedSfaSchedule(7))
        phases = []
        for epoch in range(1, 7):
            trace = torch.zeros(20, 64, dtype=torch.int64)
            for group in range(7):
                tasks = tuple(range(group, 20, 7))
                trace[group, 0] = tasks[-1]
                for offset in (0, 16, 32):
                    trace[group, offset + 1:offset + 4] = torch.tensor(
                        [len(tasks), sum(task + 1 for task in tasks), tasks[-1]])
                    trace[group, offset + 6] = epoch
                    if epoch == 1:
                        trace[group, offset + 7:offset + 10] = torch.tensor([1, 150, 180])
            trace[7, [6, 21, 22, 24, 38]] = epoch
            trace[7, 25] = 21
            phases.append(trace)
        for ids in ((2, 0), (), (3,)):
            with self.subTest(query_ids=ids):
                indices = torch.zeros((len(ids), 1, 1), dtype=torch.int32)
                requests = torch.full((tokens, 4), -1, dtype=torch.int64)
                membership = torch.zeros((tokens + 7) // 8 * 8, dtype=torch.int32)
                unique = int(bool(ids))
                if ids:
                    key = 3 if ids == (3,) else 0
                    requests[0] = torch.tensor([0, key, key, 1])
                    membership[key] = len(ids)
                counts = torch.tensor([SELECTED_REQUESTS_MAGIC, 5, unique, unique, len(ids), 0, 0, 0,
                                       190, 195] + [0] * 6)
                transport = torch.tensor([5, 5, 1, 1, tokens * 256, unique * 1152, 0, 0,
                                          100, 120, 200, 250] + [0] * 20)
                empty = torch.empty(0)
                saved = SimpleNamespace(batch_meta=replace(meta, q_global_ids=ids), query_scope="local")
                result = fused_cp.FusedCpForwardResult(empty, empty, empty, empty, empty, (), (), transport, 5,
                                                      saved=saved, selected_requests=(requests, counts, membership))
                snapshots = (requests, counts, membership, indices)
                backend.prepare_local_queries(())
                if ids == (3,):
                    with self.assertRaisesRegex(ValueError, "cross-partition LD merge"):
                        backend.validate_trace(result, tuple(phases), transport, require_overlap=False,
                                               selected_snapshots=snapshots)
                else:
                    evidence = backend.validate_trace(result, tuple(phases), transport, require_overlap=False,
                                                       selected_snapshots=snapshots)
                    self.assertEqual(evidence["compute"]["indexer"]["ld_partitions"], 0)
                    self.assertEqual(evidence["selected_requests"]["selected_slots"], len(ids))
