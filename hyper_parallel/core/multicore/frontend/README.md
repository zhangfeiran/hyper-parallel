# Python AST frontend and Gate compatibility plans

This implements the semantic frontend foundation from the 2026-10-07 AST design.
The implementation covers typed source capture, identity-based primitive schemas,
ProgramIR, CPU reference interpretation and source-mapped Gate WorkerPipeline host
plans. Device materialization and execution remain a separate milestone.

## Use the frontend

```python
import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml


@mc.program
def activate(value: ml.Tensor[ml.fp32, ("T", "E")]):
    return ml.sqrt(ml.softplus(value))


ir = activate.lower()
print(ir.dump())
```

The runnable [Gate Route example](examples/gate_route.py) follows the design's
softplus, sqrt, detached selection bias, top-k, gather, normalization and scale
order. Run it from an editable checkout:

```bash
pip install --no-deps -e .
python -m hyper_parallel.core.multicore.frontend.examples.gate_route
python -m pytest -q tests/ut/core/multicore/frontend
```

`Program.lower(k=3, scale=2.5)` produces immutable semantic IR.
`Program.explain(...)` shows schedule intent and a deterministic JSON IR dump,
including storage dtype, symbolic shapes, logical accesses and original source
locations. `Program.interpret(...)` runs registered references on contiguous CPU
Torch tensors and unifies symbolic dimensions across inputs and outputs.

Interpretation preserves Torch autograd through the reference functions. This
verifies logits gradients and detached bias for the Gate example; it does not
provide a native backward recipe.

## Language and registration contracts

- Tensor annotations retain bf16, fp16, fp32, int32, int64 and boolean identity,
  static/symbolic shapes, and a contiguous logical layout.
- Constexpr parameters accept exact scalar types, including optional scalar
  unions. Runtime tensors cannot be constexpr values. Static integers, strings,
  tuples and floats are bounded and validated.
- Ordinary named parameters, assignment, annotated assignment, tuple unpacking,
  tuple returns, scalar arithmetic/comparisons and constexpr branches are supported.
- `@mc.helper` registers typed helper source for inlining. Errors retain the
  helper definition location and caller chain. Recursive calls and nesting beyond
  32 helpers are rejected.
- `mc.static_range` supports one to three integer bounds and at most 1024 total
  unrolled iterations across a compilation, including helpers and nested loops.
- Symbols resolve by local lexical scope and exact registered primitive identity.
  Aliases work; a same-name callable does not become a primitive. Only the DSL
  module allows attribute lookup. Captured globals and closure values must satisfy
  the static contract when used as constants.
- The compiler does not execute the captured function body, use eval/exec, or
  invoke arbitrary functions/properties. Registered inference and reference hooks
  are trusted extension code; reference hooks execute only in the interpreter.
- Unsupported syntax is rejected with file, line and column, including inside
  branches removed by specialization. Data-dependent control flow, while loops,
  arbitrary subscripting, mutable containers and in-place tensor writes are rejected.

`from_source(source, signature, constants, symbols=...)` captures exactly one
function when inspect source is unavailable. Types can be supplied by the explicit
signature; primitive/helper symbols must be supplied explicitly. String annotations
are parsed with the same restricted resolver, without evaluation.

`PrimitiveRegistry` keys schemas by logical namespace and version, independently
of native task numbers. Each schema supplies an inspect signature, type/shape
inference, logical access inference and an optional reference. Calls normalize
keyword/default arguments before inference; duplicate schema registration fails.
The default access contract reads tensor arguments and writes newly allocated
results. A custom access hook can declare read/write/reduce/atomic accesses to
logical tensor arguments. Physical buffer aliasing and communication effects await
buffer/runtime integration.

## Gate WorkerPipeline host plans

A supported text Route program with `schedule=mc.WorkerPipeline()` now provides
`Program.plan(signature, topology, **constants)`:

```python
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route

plan = _route.plan({"T": 33, "E": 4}, mc.HardwareSpec(48), k=3, scale=2.5)
print(plan.explain())
print(plan.export_manifest())
normal_runtime_bytes = plan.forward.normal
profiled_runtime_bytes = plan.forward.profiled
```

The signature supplies exactly the symbolic input dimensions; a program with
static input shapes needs no shape signature. Hardware topology supplies the
available AIV workers independently of computation. Planning does not probe or
read devices. `from_source(..., schedule=mc.WorkerPipeline())` supports the same
plan API for explicit source.

The compiler proves that the graph uses the canonical Route schemas and complete
dataflow: FP32 logits and detached bias; sqrt after softplus; selection using
scores plus bias; unsorted last-axis top-k; gathering original scores; row sums,
`1.0e-20` epsilon and division for k greater than one; scale and int64 indices.
Changing these contracts, adding extra operations or declaring mutating effects
causes an explicit rejection. Matching only a familiar primitive name is not
sufficient. The backend emits the pinned forward/backward templates after these
checks.

The Gate semantic IR has 11 operations for k greater than one and 8 for k equal to
one. Both lower to the existing ten-stage forward descriptor sequence. For k equal
to one, ReduceSum/AddEpsilon/Div descriptors remain present, with an explicit
`retained_legacy_k1_stage` reason. Backward selects the original eleven-stage or
two-stage descriptor template, retaining the separate CANN postprocessing and
optional direct-logits gradient addition as external call metadata.

`KernelPlan` includes normal/profiled images, logical/native stage identities,
source spans, native binding order and the five saved-state names. The schedule
keeps each launched worker's ordered stages and row interval, including native
empty tail workers. Images retain the fixed 48-worker slot capacity and their
bytes are independent of token count/available workers. `schedule.simulate()`
enumerates the ordered descriptor visits for each launched worker; it does not
simulate device instructions, streams or CANN postprocessing.

Family contracts are documented in the
[compatibility baseline](../runtime/baselines/README.md). CPU tests reproduce all
six Gate wire images byte for byte against snapshots from the original builder,
and compile extracted original C++ declarations to check Python structure sizes
and field offsets for MoE, MHC and Gate.

## Current implementation boundary

`TaskDAG` still records selection metadata; MoE/MHC schedule lowering is pending.
The default MegaMoE call path, workers, runtime ABI and backward code are retained.
The component package loads MegaMoE/profiler business exports on first access,
allowing this frontend to import without torch_npu or a native payload.

Gate plans currently report `native_status=unbound` and
`device_tiling=required_from_legacy_host`. Native host tiling must still validate
UB/workspace requirements and materialize the plan. No native library, tensor
pointer, stream or workspace is bound by `Program.plan()`. A build-provided
`NativeManifest` can be checked against the selected family ABI/source contract;
this metadata guard does not inspect ELF contents or prove a device build. The
native build does not yet emit this frontend manifest.

The remaining Gate integration includes the legacy native payload, host tiling,
materialization, autograd/saved-state resource adapter, vision masking and device
validation. Full MoE/MHC plan snapshots and lowering remain pending, as do ragged
tensors, tensor lists, runtime/shape scalars, buffer planning, cache and generated
Ascend workers. CPU plan parity is not an NPU correctness or performance result.
