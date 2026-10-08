# Python AST frontend and compatibility plans

This implements the semantic frontend foundation from the 2026-10-07 AST design.
The implementation covers typed source capture, identity-based primitive schemas,
ProgramIR, CPU reference interpretation, Gate WorkerPipeline plans and MoE/MHC TaskDAG
plans. Device bindings preserve the existing family-specific numerical kernels
and explicit backward recipes.

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

`Program.plan()` remains CPU-only and reports `native_status=unbound` until
explicit materialization. The isolated Gate builder exports the fixed Gate
revision and verifies its source hashes, plus the pinned ops-nn LinearIndex
source. It builds only Route/RouteGrad into a separate vendor, without replacing
the current MoE runtime or building SHMEM. CANN 9.2 or later is required by the
pinned tiling API (`TensorShape` and `TensorDataType`); CANN 9.1 does not provide
these types. The supported device targets are ascend910b and ascend910_93.

Build in an editable checkout with the fixed Gate commit available in its Git
object database and a clean, locked ops-nn checkout:

```bash
source /path/to/cann/set_env.sh
python -m hyper_parallel.core.multicore._build.build_gate \
    --ops-nn-source /path/to/ops-nn --soc ascend910b --jobs 16
source build/native/gate/payload/set_env.bash
# Start a fresh Python process after activation.
```

The builder checks required ACLNN symbols, host ELF hardening and device binary
presence, then emits `manifest.json`. Its build fingerprint covers build inputs,
CANN/compiler/framework identity and hashes of all payload artifacts. Binding
checks the family/source/schema identities, fingerprint and every artifact hash
before loading the adapter. CANN, Torch and torch_npu identities must match the
build. The compatibility payload uses an isolated custom OPP process: do not
activate another custom vendor in the same process, since framework ACLNN and
tiling caches are process-wide.

```python
import torch
import torch_npu
import hyper_parallel.core.multicore.frontend as mc
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route

torch.npu.set_device(0)
topology = mc.HardwareSpec(torch.npu.get_device_properties(0).vector_core_num)
plan = _route.plan({"T": 49, "E": 64}, topology, k=3, scale=2.5)
route = plan.materialize("npu:0")
logits = torch.randn(49, 64, device="npu:0", requires_grad=True)
bias = torch.zeros(64, device="npu:0", requires_grad=True)
weights, indices = route(logits, bias, profile=True)
(weights.square().sum() + logits.square().sum()).backward()
records = [call.records() for call in route.take_profiles()]
```

The native host computes UB/workspace tiling. Materialization checks that the
plan's launched row workers match the actual device topology. The 48 descriptor
slots are a capacity, not a hardware core-count claim: Ascend910B3 has 40 AIV
workers. Original backward tiling launches only nonempty row workers, while the
forward launches the full row-worker count, including empty tails.

Forward state belongs to each autograd invocation; correction bias is detached,
indices are nondifferentiable, and double backward is unsupported. RouteGrad
uses the original 11/2-stage template and CANN LinearIndex/scatter/sqrt/softplus
postprocessing. Direct logits losses compose through Torch autograd. Normal and
profiled descriptors remain separate; profiling buffers are private to each
call, and reading records synchronizes only that call's completion event.
Materialized descriptors use an initialization event and current-stream
allocator recording for reuse across streams. Callers must establish input
producer/consumer stream dependencies, following ordinary Torch stream rules.

Run single-card acceptance after activation:

```bash
python -m pytest -s tests/torch/multicore/test_ast_gate.py
```

Acceptance compares selected expert sets exactly and FP32 weights/logits
gradients against independent CPU and NPU Torch references at `rtol=2e-4`,
`atol=2e-5`. It covers k=1/k>1, small/tail/batched rows, direct logits gradients,
projection autograd, overlapping forwards, multiple streams, noncontiguous
incoming gradients and actual stage-record coverage. Results are written to
`build/native/gate/acceptance.json` (override with `HP_AST_GATE_RESULT`).

## MoE TaskDAG integration

The [MoE region](examples/moe_region.py) declares dispatch, grouped matmul,
packed gate/up SwiGLU, grouped matmul and combine under
`mc.TaskDAG(policy="moe_ratr_v1")`. The compiler verifies canonical primitive
identities, BF16 storage, complete operand/group/metadata dataflow, packed layout,
clamp specialization and preserved numerical order. It derives the forward DAG
edges from ProgramIR, then uses the original native task nodes and complete
`build_config_for_rank` path. Termination tasks, queue revision, dynamic group
scratch, ready/completion protocols and profiling layouts retain their original
contracts. The original autograd and replica gradient-return recipe supplies
backward, including W13 overlap and its no-replica fallback.

Models select the region through the existing module constructor:

```python
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.frontend.examples.moe_region import moe_region

experts = MegaMoeExperts(
    local_num_tokens=128, hidden_size=512, intermediate_size=128,
    num_experts=4, top_k=2, ep_size=2, ep_group=ep_group,
    dispatch_mode="push", swiglu_limit=10.0, program=moe_region,
).to(device="npu", dtype=torch.bfloat16)
output = experts(hidden_states, topk_ids, topk_weights,
                 tokens_per_expert=tokens_per_expert)
output.backward(grad_output)
experts.close()
```

Router/permutation, histogram handling, caller-owned weights, capacity growth,
hot-replica planning and SHMEM/workspace ownership continue through that module.
Program identity participates in static rank agreement and shared-resource
compatibility. Modules must have the same program to share execution resources.
Changing runtime counts or route skew does not specialize the semantic program.
Omitting `program` retains the original module entry.

For CPU planning, call `moe_region.plan(spec, limit=spec.swiglu_limit)` with an
existing `MegaMoeSpec` describing rank-local shapes, physical cores and guest
slots. This emits full normal/profiled forward/backward images, fixed-stride
worker queues, stage source spans, physical bindings and a provenance manifest;
it requires no NPU import or device access. `plan.materialize(device)` returns
the existing `MegaMoePlan` resource object. Model execution should use
`MegaMoeExperts` so the module also owns routing and distributed lifetimes.

`ml.RaggedTensor[dtype, (capacity, width)]` separates storage capacity from valid
rows. `ml.TensorList[dtype, matrix_shape]` describes homogeneous expert matrices;
the dtype-only form binds matrix shapes through the native specification.
`ml.RouteMetadata` identifies runtime cumulative INT64 expert counts. The CPU
interpreter accepts `ml.RaggedTensor(storage, valid_rows)` and
`ml.RouteMetadata(group_list)` for already local, expert-major rows. Its dispatch
and combine references validate local grouping and preserve row order; they do
not simulate distributed communication or token permutation. Native execution
binds the module's actual routing buffers and dynamic group lists.

The normal MoE build now emits `lib/frontend_manifest.json`, sealing the fixed
MoE ABI/native source baseline, build inputs, toolchain/framework identity and
all MoE/private SHMEM payload files. AST materialization checks the seal,
artifact set and CANN/Torch/C++ ABI identity before loading resources. Build and
activate the MoE payload in a fresh process:

```bash
bash hyper_parallel/core/multicore/build.sh --soc-list ascend910b --jobs 16
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
python -m pytest -s tests/torch/multicore/test_ast_moe.py
```

The sealed MoE vendor must be first in `ASCEND_CUSTOM_OPP_PATH`. Use separate
processes for Gate and MoE custom OPP activation because CANN/framework operator
caches are process-wide. Older MoE builds need rebuilding to provide the seal
before selecting the AST program.

Vision masking, runtime/shape scalars, general buffer planning,
compilation caching and generated Ascend workers remain future stages. Device
acceptance establishes the tested topology and shapes; it does not establish
performance or full-model training equivalence. Ascend910_93 still requires
device validation.

## MegaMHC shifted TaskDAG integration

The [MHC boundary](examples/mhc_boundary.py) declares post, mapping, input mix
and RMSNorm under `mc.TaskDAG(policy="shifted_mhc_v1")`. It returns updated
residual, next pre/post/residual mixes and the current block input. InputMix
uses **previous** pre coefficients; the mapping branch predicts the next layer's
coefficients. The compiler checks canonical primitive identity, all nine typed
operands, ordered five outputs, fixed four streams, twenty Sinkhorn iterations
and equal NormCast/RMSNorm epsilon before deriving the semantic fork from SSA.

Mapping expands to NormCast, Projection and Mapping. The six native stages retain
the original AIV gap-filling order and AIC projection overlap. NormCast waits
for Projection before reusing an X-cast ring slot. The backward recipe retains
output initialization, RMSNormGrad, previous-A/mapping preparation, AIC Phi/RMS
and previous-X/post gradients. Each AIC macro waits for all of its corresponding
AIV producers. `plan.explain()` exposes stage source locations, worker queues,
ring ownership, macro joins and seven distinct saved cache buffers; the legacy
TensorSpec placeholder does not imply physical buffer aliasing.

```python
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec

plan = mhc_boundary.plan(MhcSpec(2593, 128, num_cube_cores=20))
print(plan.explain())
# Compilation and complete normal/profiled serialization require only CPU Torch.
```

Native forward/backward workers and adapters come from the design's exact MHC
revision `979e2a9ac913413e361f4fc2dd9987766af8ddb4`. MHC task IDs remain local
to that family; the MoE task enum and native headers are not extended. The
builder exports verified sources into an isolated directory, checks and applies
their locked patches, and records all actual build inputs and payload artifacts.
The CANN 9.2 wrapper removes one unused obsolete SDK include from exported common
headers. It does not alter the pinned worker's numerical implementation.

```bash
source /path/to/cann-9.2/set_env.sh
# Prepare the normal MoE dependencies/private SDK with the existing build first.
python -m hyper_parallel.core.multicore._build.build_mhc \
    --ops-nn-source build/native/deps/ops_nn/src \
    --ops-mhc-source /path/to/clean/ops-transformer-mhc
unset ASCEND_CUSTOM_OPP_PATH
source build/native/mhc/payload/set_env.bash
python -m pytest -s tests/torch/multicore/test_ast_mhc.py
```

The MHC dependency checkout must identify
`58b4a6bdeb29feeb0070dd266106bd4e130bb72b` with its locked tree and archive
hash, including a compatible hash from the original lock. Git LFS smudging can
change archive bytes: preserve the committed pointer bytes when materializing
this source checkout. The builder currently consumes the existing private SDK
record at `build/native/work/multicore/shmem/sdk.json` for generic worker header
dependencies. It does not initialize SHMEM or link a SHMEM runtime into MHC.
Use a fresh process with exactly the activated MHC vendor in
`ASCEND_CUSTOM_OPP_PATH`; Gate/MoE/MHC operator caches cannot share a process.

The model entry point preserves the original parameter names, shapes and dtypes:

```python
from hyper_parallel.core.multicore import HyperMegaMhc

layer = HyperMegaMhc(128, device="npu:0")
updated, next_pre, next_post, next_matrix, block_input = layer(
    previous_output, residual, previous_pre, previous_post, previous_matrix,
    profile=True,
)
loss = block_input.float().square().mean() + next_pre.square().mean()
loss.backward()
records = [record for call in layer.take_profiles() for record in call.records()]
layer.close()
```

Phi/alpha/bias stay FP32 and norm weight stays BF16. A compatible custom region
can be supplied through `program=`. Model descriptors are cached per flattened
shape/device/backward contract; inputs and parameters bind anew on every call.
Each invocation owns fresh event counters, saved caches and optional profile
storage, permitting multiple pending forwards and checkpoint recomputation.
Closing a binding rejects future forwards; pending autograd contexts retain
its descriptors. Initialization events and allocator stream recording make
cross-stream reuse safe after callers establish their input producer ordering.
Native double backward is unsupported. Profiling reports actual task cycles,
source locations and direction, and rejects dropped/corrupted records.

Native backward requires T at least the physical AIV count and H <=5760; H must
be divisible by 128. Use `need_backward=False` for smaller inference inputs.
Epsilons must stay finite and positive after FP32 conversion. Planning rejects
both forward/backward event boundaries that lack padding for the original
8-element atomic counter write; it does not enlarge or silently change that ABI.
For example, T=10817 with a 32-row tile is rejected, while a larger safe tile
can be selected explicitly. General automatic buffer planning, generated workers
and unified schema-derived bindings are P5 work.

Committed CPU fixtures contain sixteen complete original forward/backward images
covering tails, ring wrap and 32/64/96-row tiles. Reproduce them with:

```bash
python -m tests.ut.core.multicore.backends.fixtures.capture_mhc_snapshots
python -m pytest -q tests/ut/core/multicore/backends/test_mhc.py
```

The single-card device suite checks all five outputs and all nine gradients
against the independent pinned Torch oracle using its original precision gate
(relative L2 <=0.02, cosine >=0.999). It also checks streams, multiple pending
forwards, missing output gradients, closing before backward, checkpoint/SGD,
and profile coverage. The CPU primitive references expose the native BF16
mixed-input cache boundary explicitly; the independent acceptance oracle retains
its original mathematical reference. Device admission applies to the validated
shapes/topology, without a performance or full-model convergence claim.

## P5 shared schema and source emission

`Program.compile(...)` uses the same backend emission entry for Gate, MoE and
MHC. Existing `plan(...)`, materialization and model call signatures retain their
family-specific behavior. Compilation currently emits sources and complete
schedule artifacts; its manifest explicitly reports `status="source_only"`.
It does not claim that a generated source bundle is a compiled device binary.

```python
from pathlib import Path
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
from hyper_parallel.core.multicore.runtime.cache import EmissionCache

emission = _route.compile({"T": 41, "E": 16}, k=3, scale=2.5)
path = EmissionCache(Path("build/native/frontend-cache")).store(emission)
print(emission.export_manifest())
```

The bundle includes Python/C++ wire declarations, Python/C++ typed call wrappers,
the selected native binding schema, normal/profiled forward/backward images,
optional MoE no-replica backward, the static plan and its semantic source map.
The following identities have different lifetimes:

- `definition_key` covers normalized semantic IR, constexprs, family ABI/native
  source closure and actual generator inputs. Runtime row counts and tensor
  pointers do not enter this key.
- `plan_key` adds static shapes/capacities, topology/rank and serialized schedule
  images. Dynamic routing counts and process-group objects remain outside it.
- `artifact_key` also covers exact generated files and source locations. This
  prevents a bundle from reusing the wrong source map when native semantics match.

`EmissionCache` stores source-only bundles atomically and verifies metadata and
all file hashes before reuse. It rejects missing, escaping, corrupted or extra
files. Invocation-owned pointers, epochs, caches, counters and workspace leases
continue through existing family resources; the disk cache never stores them.

Wire declarations derive from `runtime/baselines/families.json`. Common enums,
six native call signatures and preserved native field representations live in
`runtime/native_calls.json`. The generated Python ctypes structures and native
C++ types assert every size/offset from those shared contracts. MoE keeps its
completion/protocol fields and dynamic scratch; Gate/MHC keep their original
reserved header fields and family-local task IDs. The original native unsigned
representation of `DynamicData.dynamic_max_seq_len` is recorded explicitly,
preserving its existing C++ contract alongside the original signed ctypes view.

Regenerate or check the checked-in outputs with:

```bash
python -m hyper_parallel.core.multicore.backends.schema
python -m hyper_parallel.core.multicore.backends.schema --check
python -m pytest -q tests/ut/core/multicore/backends/test_codegen.py
```

The live scheduler consumes generated MoE Python types. MoE workers include
generated C++ declarations; Gate/MHC builders install corresponding declarations
in verified isolated source exports. The MoE manifest still checks all pinned
native sources. Its sole header adaptation must equal the deterministic
transformation of the verified original Git object; unrelated native drift is
rejected. The transformation replaces wire declarations, formats C++ and
shortens descriptor-reader local names while preserving accessor behavior.

All six runtime forward/backward calls use the generated Python launch wrappers.
They validate the actual loaded dispatcher schema once, including argument names,
types, result order and mutable alias annotations. Existing autograd, stream,
SHMEM, tiling and allocator ownership remains with the family implementation.
Generated C++ forwarding wrappers are instantiated in CPU tests for argument
order and const/mutable ownership; existing native Torch adapters still enqueue
the family operators.

This delivers the first P5 increment. Automatic worker switches/context factories,
replacement of family native host adapters, general buffer planning and compiled
binary/materialized-device caches remain subsequent work. New arbitrary primitive
combinations still need supported numerical implementations and lowering recipes.
