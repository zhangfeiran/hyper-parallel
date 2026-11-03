# LongCat Flash

This integration uses the complete Hugging Face `LongcatFlashForCausalLM` architecture. Every logical
block retains two MLA attention sublayers, two dense MLPs, and the shortcut MoE whose output joins the
second sublayer's residual. The reference model executes before any optional replacement is installed.

The semantic reference is
[Transformers LongCat Flash](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/models/longcat_flash/modeling_longcat_flash.py).
The adapter accepts the router's two-value and three-value return contracts. It keeps full-pool FP32
softmax, selection-only correction bias, and the original scaled probabilities without Top-K renormalization.

## Expert execution

IDs below `n_routed_experts` select parameterized SwiGLU experts. The remaining IDs select identity
experts. Each selected identity contributes its routing probability multiplied by the input token.
Real and identity contributions are accumulated in expert order, retaining the reference's per-contribution
cast. Identity weights and the input bypass remain differentiable. Storage-sharing identity input views and
an ordered gather backward preserve HF's per-expert BF16 input-gradient accumulation without expanding
identity tokens. `swiglu_precision: hf` is the model default: the optional generic MegaMoE switch
`swiglu_hf_bf16=True` rounds the SiLU intermediate in forward and the two multiply branches in backward.
The generic MegaMoE default remains fused. `swiglu_precision: fused` explicitly selects that arithmetic
for both reference and candidate; validation reports distinguish the two modes.

`LongcatFlashMegaMoeExperts` compacts active real assignments into a K=1 `MegaMoeExperts` executor.
The `megamoe` backend requires BF16 NPU inputs and supports push and pull. The explicit `native` backend
uses NPU permute/unpermute, grouped GEMM and differentiable EP collectives on Ascend. Its HF mode
uses separate SiLU and multiply to retain BF16 cast boundaries; fused mode uses `npu_swiglu`.
On CPU it uses the portable expert reference. The explicit `eager` backend retains the per-expert
PyTorch reference on every device. Empty sources still participate when
other ranks have real assignments. A globally identity-only call leaves expert gradients unused.

Compaction materializes ordinary `[R, H]` scratch for the real assignments. It avoids identity traffic and
identity FFN work, but does not implement indexed reads from an unexpanded source. It is a correctness
and integration baseline; throughput and HBM improvements need paired measurements.

## Parameter and checkpoint layouts

EOI is the default execution layout and uses the master's contiguous EOI gradient path. External FSDP
parameters are supplied afresh at every call. The model retains HF's full Gate/Up tensor, including the
unused rows allocated for identity experts; those rows are never executed as FFNs.

For EP, a reversible conversion interleaves each owner's real and unused Gate/Up rows. The Down tensor
is sharded over real experts. Both expert cardinalities must be divisible by EP size. A checkpoint marker
rejects loading a packed layout from another EP size. Export to the HF layout before changing EP degree;
HF export restores all original tensor names, rows, and values and removes the packing marker.

The adapter rebuilds non-persistent RoPE frequencies after meta materialization and preserves persistent
router correction biases. Compatible MegaMoE executors share transient resources before the first layer.

## Model replacements and training

The recipes retain HF RMSNorm and use a model-owned dense MLP adapter that keeps the generic packed
checkpoint layout while preserving HF's separate BF16 Gate/Up and activation boundaries. The MLA adapter
reuses generic MLA with LongCat's Q/KV LoRA scales and interleaved rotary convention. An explicit eager
attention request retains HF attention and separate latent projections; other NPU attention modes use
the fused attention interface. Dense MLP boundaries are declared explicitly because LongCat stores
them in a `ModuleList`.

- [MegaMoE recipe](recipes/train.yaml)
- [Same-topology native expert control](recipes/train_native.yaml)
- [Original-width one-logical-layer scratch recipe](recipes/train_one_layer.yaml)
- [Local validation manifest](validation.yaml)

The one-layer recipe uses 16 ranks with EP16, dense FSDP16 and expert FSDP1. Its model config only
changes `num_layers` and disables the inference cache: hidden size 6144, expert intermediate size 2048,
dense intermediate size 12288, real/identity expert counts 512/256, Top-K 12, vocabulary 131072, 64 heads,
Q-LoRA rank 1536 and KV-LoRA rank 512 retain the original values. A logical layer still contains both
attention and dense sublayers plus shortcut MoE. Sequence length is 2048 and global batch is 16.
It initializes scratch parameters and requires local tokenizer/data paths; it never downloads weights.

Native GMM and literal HF per-expert `F.linear` are separate numerical references. The HF SwiGLU cast
option aligns activation semantics, while different GEMM implementations can retain BF16 rounding
differences. Record their precision verdicts separately and declare the primary reference before
performance acceptance. Keep HF RMSNorm in both recipes: fused RMSNorm can change the BF16 weight
gradient's multiply/reduction boundaries even when forward outputs are close.

The full-depth pretrained recipes start at 256 ranks with EP64 and expert FSDP4. Their global batch is divisible by the
resolved data-parallel micro-batch. This is a topology declaration, not a measured memory-capacity result.
Retaining HF's unused Gate/Up rows adds parameter and optimizer storage; account for those rows when
sizing a full checkpoint job. No example downloads weights, tokenizers, or datasets implicitly.

Replace the local asset paths and launch the selected recipe through the standard `scripts/train_lm.py`
entry. The precision policy is FP32 initialization/main parameters, BF16 compute parameters, FP32 gradient
reduction, and `cast_forward_inputs: false`. Initial optimized recipes require TP=CP=PP=1. Expert parallelism
and FSDP compose through the existing framework; unvalidated parallel axes fail explicitly.

## Validation

Run CPU contracts with:

```bash
pytest -q tests/ut/models/longcat_flash tests/ut/core/multicore/modules/mega_moe/test_assignments.py
python -m hyper_parallel.tools.model_integration inspect --manifest hyper_parallel/models/longcat_flash/validation.yaml
python -m hyper_parallel.tools.model_integration check --manifest hyper_parallel/models/longcat_flash/validation.yaml
python -m hyper_parallel.tools.model_integration parity --manifest hyper_parallel/models/longcat_flash/validation.yaml --device cpu --dtype float32
```

The portable validation builders preserve all 28 logical blocks and routing cardinalities 512/256/12
while reducing parameter widths for CPU contract checks. They are diagnostics, not original-width
accelerator acceptance. Use the one-layer recipe for depth-only cropping on NPU. The validation manifest's
generated handoff covers the declared structural/portable expert checks; measured accelerator evidence
remains separate from full pretrained training acceptance.

The distributed worker uses the actual recipe replacements and sharding planner. A reduced two-layer case
is a smoke test. Set `HP_LONGCAT_FULL_DEPTH=1` to preserve the released depth and routing cardinalities.
Set `HP_LONGCAT_REFERENCE=same_dense` to compare against HF experts with identical non-MoE replacements;
the default `hf` control retains the unmodified HF model. Reports record this distinction.

Activate a matching CANN/native payload and acquire the shared NPU launcher lock before running the
NPU launcher `tests/torch/models/longcat_flash/test_longcat_flash.py`. The parent launcher imports no
framework. Both native and MegaMoE training recipes retain HF RMSNorm on every device.

Acceptance must name its oracle: native GMM agreement does not establish literal HF `F.linear` gradient
parity. Functional Trainer smoke, finite optimizer state, and portable parity do not substitute for the
declared accelerator precision/performance gates. The assignment path skips
empty-index `index_copy` calls and retains a zero-sized autograd dependency so empty source ranks still
serve remote expert backward work. The distributed lifecycle test covers that case and globally
identity-only calls independently of the learned-routing precision test.

For full-depth CPU/Gloo coverage through the recipe and planner:

```bash
HP_LONGCAT_BACKEND=native HP_LONGCAT_DEVICE=cpu HP_LONGCAT_FULL_DEPTH=1 OMP_NUM_THREADS=1 \
  torchrun --standalone --nproc-per-node=2 -m pytest -q -s \
  tests/torch/models/longcat_flash/_test_longcat_flash.py
```

`_test_longcat_empty_routes.py` is the separate fixed-route lifecycle diagnostic. Its boundary tolerances
are independent of the complete-model learned-routing gradient gate; passing it cannot turn a failed
complete-model precision result into a pass.
