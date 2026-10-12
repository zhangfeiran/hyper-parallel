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
"""Stage snapshots preserve ownership, integer identity and independent failure attribution."""

import unittest

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_gradient_stage_trace import (
    GradientStageTrace,
    compare_stage_fields,
)


class TestGradientStageTrace(unittest.TestCase):
    """No hardware or collectives are needed to verify paired snapshot semantics."""

    def test_owned_snapshots_pair_reverse_backward_and_distinguish_integer_fields(self):
        """Later buffer overwrites cannot alter the recorded original native stage."""
        trace = GradientStageTrace()
        for phase in ('isolated', 'delayed'):
            for microbatch in ((0, 1) if phase == 'isolated' else (1, 0)):
                value = torch.tensor([float(microbatch)], requires_grad=True)
                ids = torch.tensor([microbatch], dtype=torch.int32)
                trace.record(phase, microbatch, 'input', {'value': value, 'ids': ids, 'inactive': None})
                output = value.detach()+float(phase == 'delayed' and microbatch == 1)
                trace.record(phase, microbatch, 'output', {'value': output})
                with torch.no_grad():
                    value.add_(100)
                    ids.add_(100)
                    output.add_(100)
        report = trace.diagnose()['stages']
        for comparison in report['input']:
            self.assertTrue(all(field['bit_exact'] for field in comparison.values()))
            self.assertIsNone(comparison['ids']['metrics'])
            self.assertIsNone(comparison['inactive']['metrics'])
        self.assertTrue(report['output'][0]['value']['bit_exact'])
        self.assertFalse(report['output'][1]['value']['bit_exact'])
        self.assertEqual(report['output'][1]['value']['metrics']['max_abs'], 1.)

    def test_missing_duplicate_or_incompatible_stage_records_are_rejected(self):
        """Incomplete traces and schema/dtype mismatches must not silently become passing evidence."""
        trace = GradientStageTrace()
        value = {'value': torch.ones(2)}
        trace.record('isolated', 0, 'input', value)
        with self.assertRaisesRegex(RuntimeError, 'recorded twice'):
            trace.record('isolated', 0, 'input', value)
        with self.assertRaisesRegex(RuntimeError, 'complete'):
            trace.diagnose()
        with self.assertRaisesRegex(ValueError, 'phase'):
            trace.record('other', 0, 'input', value)
        with self.assertRaisesRegex(ValueError, 'schema'):
            compare_stage_fields(value, {'other': torch.ones(2)})
        with self.assertRaisesRegex(ValueError, 'shape and dtype'):
            compare_stage_fields(value, {'value': torch.ones(2, dtype=torch.float64)})
