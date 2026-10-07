# Python AST frontend: first implementation

This implements the semantic frontend foundation from the 2026-10-07 AST design.
The first milestone covers typed source capture, identity-based primitive schemas,
ProgramIR and CPU reference interpretation. Native family integration remains a
separate milestone.

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

## Current implementation boundary

`WorkerPipeline` and `TaskDAG` currently record schedule selection metadata.
There is no schedule emitter, device plan, native launcher, cache or generated
Ascend code in this milestone. The Gate semantic IR has 11 operations for k greater
than 1 and 8 for k equal to 1; those counts are not native stage descriptors. In
particular, the existing Gate ten-stage forward sequence must be preserved by a
future compatibility lowering even when constexpr removes normalization here.

The default MegaMoE call path, workers, runtime ABI and backward code are retained.
The component package loads MegaMoE/profiler business exports on first access,
allowing this frontend to import without torch_npu or a native payload.

The next stages need the P0 family ABI manifests and fixed plan snapshots, followed
by Gate WorkerPipeline lowering, MegaMoE's complete legacy finalize/RATR path and
backward adapter, and MHC fork/ring-buffer contracts. Ragged tensors, tensor lists,
runtime/shape scalars, buffer planning and native backward recipes are not yet
implemented. No NPU correctness or performance result is claimed by these CPU tests.
