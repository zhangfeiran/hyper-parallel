# DeepSeek-V4.1 MegaMoe adaptation

The optional `DeepseekV41MegaMoeExperts` replacement executes routed experts with the generic MegaMoe component.
The original learned router, visual routing bias, routing probabilities and shared experts retain their semantics.
Native training recipes remain unchanged and continue to provide the reference implementation.

## Recipes

- Text: [train_deepseek_v41_megamoe_online.yaml](train_deepseek_v41_megamoe_online.yaml).
- VLM: [train_deepseek_v41_megamoe_vlm_online.yaml](train_deepseek_v41_megamoe_vlm_online.yaml).

Both recipes derive their settings from the corresponding native online example and enable the expert replacement plus
`deepseek_v41_megamoe_compute_fn`. They preserve the base recipe's MegaGate/MegaMhc, checkpoint, precision, data and parallel
ownership declarations. Default topology is 16 ranks, EP16, TP=CP=PP=1; dense FSDP16 and expert FSDP1. The recipes use FP32
initialization/main parameters and BF16 expert execution with FP32 gradient reduction.

The replacement's `local_num_tokens: 4096` is an upper bound on actual expert input rows. It configures the generic
`max_local_num_tokens` interface, executes real rows without padding, and permits unequal and empty EP sources. Increase
this bound when the encoded VLM batch can exceed it. The native data packing contract is unchanged.

`dispatch_mode: push` selects PUT dispatch. Set this field to `pull` for GET dispatch. Both modes use PUT combine.
Push defaults to `initial_capacity_factor=1.25` and `capacity_growth_factor=1.25`, with managed lossless growth on overflow.
Pull must omit explicit capacity factors. These recipes have no expert replicas or Host Engram dependency.

After preparing local model/tokenizer, Engram and dataset assets with the existing native example tools, run text training:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=16 \
  --module examples.training_demo.train_text \
  examples/training_demo/deepseek_v41/train_deepseek_v41_megamoe_online.yaml \
  --model.config_path=/path/to/local/model \
  --model.engram_assets_path=/path/to/local/engram.json \
  --dataset.model_assets.tokenizer.pretrained_model_name_or_path=/path/to/local/model \
  --dataset.data_path=/path/to/local/text.jsonl
```

For VLM, use the existing VLM Trainer entry point and the VLM MegaMoe recipe, retaining the native example's model, media
and dataset overrides. Model assets are local-only; the MegaMoe path does not download weights or data. The recipes retain
the native depth-preserving dimension crop. They do not demonstrate the original full-model memory footprint.

## Parameter and parallel contracts

The adapter keeps `gate_up_proj` and `down_proj` checkpoint names. Their final two dimensions are transposed into the native
MegaMoe layout, with reversible `make_transforms()` conversion. EP partitions the expert axis; FSDP owns parameters and
optimizer state. Each expert forward reads the current unsharded DTensor/local tensors and passes them as external weights
to a parameterless executor, so FSDP2 all-gather or checkpoint loading cannot leave a cached stale weight pointer.

The adapter supports ordered EP subgroups, including expert data parallelism outside EP. It requires learned routing,
SiLU with a positive finite SwiGLU limit, and TP=CP=PP=1. The routed and shared expert limits must match. Hash routing,
expert tensor parallelism and graph capture are outside this adapter's supported scope.

The generic `ModelRuntimeResources` preparation hook groups compatible serial expert modules for shared temporary storage.
Normal cleanup remains owned by the base multicore/process-group lifecycle; explicit `close_runtime()` is available after
all backward work completes. Do not close a live model or independently reorder layers on one rank.

## Validation

```bash
pytest -q tests/ut/core/multicore/modules/mega_moe \
  tests/ut/auto_models/models/deepseek_v41/test_megamoe.py \
  tests/ut/auto_models/test_runtime_resources.py

pytest -s tests/torch/multicore/test_mega_moe.py::test_mega_moe_dynamic_tokens
pytest -s tests/torch/multicore/test_deepseek_v41_megamoe.py
```

Distributed launchers run push and pull separately on eight ranks. Coverage includes unequal/zero source lengths, changing
schedules, positive clamp saturation, external expert weights, output, input gradients, expert gradients and router-weight
gradients. CPU tests also check replacement pairing, meta initialization, checkpoint-layout round trips and runtime hooks.
NPU execution requires a fresh multicore payload built for the active Torch/CANN environment and idle devices.

Host Engram, hot replicas, PanGu job wrappers, packing experiments and profiler reports are separate work. Operator parity
and cropped execution must not be reported as full-depth training acceptance or an end-to-end speedup claim.
