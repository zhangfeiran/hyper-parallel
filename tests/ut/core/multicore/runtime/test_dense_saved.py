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
"""验证保存缓存的准入、流交接和所有者协议，不将 CPU 证据当作 NPU 验收。"""

import gc
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.runtime.dense_saved import DenseSavedPool, shared_dense_saved
from tests.common.mark_utils import arg_mark


class _Owner:
    """可弱登记的执行对象。"""


class TestDenseSaved(unittest.TestCase):
    """原生引用查询使用受控替身，真实引用语义由原生产物集成验证。"""

    def setUp(self) -> None:
        """创建受控原生查询和独立缓存。"""
        self.query = Mock(return_value=False)
        self.pool = DenseSavedPool(torch.device("cpu"), self.query)

    def tearDown(self) -> None:
        """释放本用例缓存及残留别名。"""
        self.pool.close()
        gc.collect()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_leaf_layout_and_retained_slots_force_private_fallback(self):
        """Feature: 内部保存值借用。
        Description: 保留两次借用，再申请第三次，并分别写入各值。
        Expectation: 逻辑值不重叠、不暴露基准张量，活跃缓存触发私有回退。
        """
        shapes = {3: (7, 262), 4: (7, 131)}
        first, _ = self.pool.acquire(shapes, None)
        first[3].fill_(3)
        first[4].fill_(4)
        second, _ = self.pool.acquire(shapes, None)
        third, _ = self.pool.acquire(shapes, None)
        for value_id, tensor in first.items():
            self.assertIsNone(tensor._base)
            self.assertEqual(tensor.data_ptr() % 64, 0)
            self.assertNotEqual(tensor.data_ptr(), second[value_id].data_ptr())
            self.assertNotEqual(tensor.data_ptr(), third[value_id].data_ptr())
        torch.testing.assert_close(first[3], torch.full_like(first[3], 3))
        torch.testing.assert_close(first[4], torch.full_like(first[4], 4))
        self.assertEqual(self.pool.statistics()["cached_invocations"], 2)
        self.assertEqual(self.pool.statistics()["private_fallbacks"], 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_only_native_exclusive_answer_admits_reuse_and_growth(self):
        """Feature: 独占准入与高水位增长。
        Description: 释放叶张量后由原生查询允许复用，再申请更大形状。
        Expectation: 先复用同一地址，容量不足时增长，统计与实际行为一致。
        """
        first, _ = self.pool.acquire({3: (7, 131)}, None)
        pointer = first[3].data_ptr()
        del first
        self.query.return_value = True
        reused, _ = self.pool.acquire({3: (7, 131)}, None)
        self.assertEqual(reused[3].data_ptr(), pointer)
        del reused
        grown, _ = self.pool.acquire({3: (129, 131)}, None)
        self.assertEqual(tuple(grown[3].shape), (129, 131))
        self.assertEqual(self.pool.statistics()["allocations"], 2)
        self.assertEqual(self.pool.statistics()["reuses"], 1)
        self.query.assert_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_stream_handoff_and_expired_hook_generation(self):
        """Feature: 消费流交接与保存钩子代次。
        Description: 在新流借用已释放槽位，再使用旧代次的消费令牌。
        Expectation: 新流等待原流事件，旧代次不能修改新调用的流记录。
        """
        stream_a, stream_b, stream_c = Mock(), Mock(), Mock()
        event = Mock()
        api = Mock()
        api.Event.return_value = event
        with patch.object(torch, "get_device_module", return_value=api):
            first, old_use = self.pool.acquire({3: (7, 131)}, stream_a)
            del first
            self.query.return_value = True
            second, use = self.pool.acquire({3: (7, 131)}, stream_b)
            event.record.assert_called_once_with(stream_a)
            stream_b.wait_event.assert_called_once_with(event)
            old_use.consume((), stream_c)
            self.assertEqual(self.pool.statistics()["stream_transfers"], 1)
            use.consume(tuple(second.values()), stream_c)
            self.assertEqual(self.pool.statistics()["stream_transfers"], 2)
            stream_c.wait_event.assert_called_once_with(event)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_close_preserves_borrowed_storage_and_rejects_new_acquisition(self):
        """Feature: 保存值关闭生命周期。
        Description: 关闭缓存时仍持有保存张量和消费令牌。
        Expectation: 缓存字节释放，张量仍可读，关闭后拒绝新借用。
        """
        values, use = self.pool.acquire({3: (7, 131)}, None)
        values[3].fill_(5)
        self.pool.close()
        use.consume(tuple(values.values()), None)
        torch.testing.assert_close(values[3], torch.full_like(values[3], 5))
        self.assertEqual(self.pool.statistics()["cached_bytes"], 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.pool.acquire({3: (7, 131)}, None)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_peer_owners_share_cache_until_last_close(self):
        """Feature: 设备级有界缓存所有权。
        Description: 两个执行对象共享缓存，依次关闭并重新登记所有者。
        Expectation: 第一次关闭保留缓存，最后关闭释放缓存，新所有者得到新池。
        """
        owners = [_Owner(), _Owner(), _Owner()]
        pools = [shared_dense_saved(torch.device("cpu"), self.query, owner) for owner in owners[:2]]
        self.assertIs(pools[0], pools[1])
        values, _ = pools[0].acquire({3: (7, 131)}, None)
        pools[0].release_owner(owners[0])
        self.assertGreater(pools[0].statistics()["cached_bytes"], 0)
        pools[0].release_owner(owners[1])
        self.assertEqual(pools[0].statistics()["cached_bytes"], 0)
        fresh = shared_dense_saved(torch.device("cpu"), self.query, owners[2])
        self.assertIsNot(fresh, pools[0])
        self.assertEqual(tuple(values[3].shape), (7, 131))
        fresh.release_owner(owners[2])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_shape_bounds_and_private_policy(self):
        """Feature: 缓存参数准入。
        Description: 提交非法槽位上限、负维度及禁止缓存的策略。
        Expectation: 非法值在分配前拒绝，私有策略没有持久槽位。
        """
        for bound in (-1, True, 1.5):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                DenseSavedPool(torch.device("cpu"), self.query, bound)
        with self.assertRaises(ValueError):
            self.pool.acquire({3: (-1, 131)}, None)
        private = DenseSavedPool(torch.device("cpu"), self.query, max_cached_invocations=0)
        values, _ = private.acquire({3: (7, 131)}, None)
        self.assertEqual(private.statistics()["cached_bytes"], 0)
        self.assertEqual(tuple(values[3].shape), (7, 131))
        private.close()
