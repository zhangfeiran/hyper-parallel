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
"""Lazy registration for the HF-native complete LongCat Flash architecture."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_parallel():
    """Load the model-owned EP bindings only when requested."""
    from . import parallel  # pylint: disable=C0415

    return parallel


def _sharding_rules():
    """Resolve parameter naming rules without eager model/backend imports."""
    from .parallel import build_parameter_sharding_rules  # pylint: disable=C0415

    return build_parameter_sharding_rules()


def _materialization(model):
    """Register derived rotary state before meta materialization.

    Args:
        model: Complete model whose state is managed.
    """
    from .materialization import register_materialization  # pylint: disable=C0415

    register_materialization(model)


def _load_validation():
    """Load independent validation declarations on demand."""
    from .validation import get_validation_spec  # pylint: disable=C0415

    return get_validation_spec()


MODEL_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="LongcatFlashForCausalLM", model_type="longcat_flash",
    expert_parallel=_load_parallel, sharding_rules=_sharding_rules,
    materialization=_materialization, validation=_load_validation,
)
register_model_adapter(MODEL_ADAPTER_SPEC)
