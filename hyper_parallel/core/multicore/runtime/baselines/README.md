# Fixed family compatibility baseline

These are host compatibility contracts extracted from the design's fixed sources.
`runtime_abi_version=1` is the host manifest's family-schema revision. It is not
written into a legacy header and does not replace MegaMoE's communication
`protocol_version`.

## Source identity

| Source | Fixed revision |
| --- | --- |
| MegaMoE | `befaa01069e997d511987feec82464c5f4549b95` |
| MegaMHC | `979e2a9ac913413e361f4fc2dd9987766af8ddb4` |
| MegaGate | `890a00ac71226594a7c69c4980ad7ac50bd8896c` |
| little-kernel reference | `d9c29909d7fcd1202d3f676e84c9c3d456308ca2` |

[families.json](families.json) contains family-local task mappings, structure
sizes/fields/offsets, scratch policy, dependency locks and SHA256 source closures.
Logical task identities carry a namespace and version; native numeric IDs are
preserved inside each family. For example, native ID 107 refers to MoE SHMEM Get,
MHC Post and Gate Route in their respective binaries.

All three pinned TaskDesc layouts are 576 bytes with extra_value_0 through
extra_value_2. Their RuntimeHeader layouts are 64 bytes, but MegaMoE uses byte
positions 32 and 36 for completion_event/protocol_version, while MHC/Gate keep
padding at those positions. MegaMoE computes grouped-list scratch capacity from
local experts and cache-line alignment; MHC/Gate retain 512 int64 entries.

The schema hash includes family, host ABI revision, numeric mappings, field
layouts, constants and scratch policy. The source fingerprint separately includes
the source revision, dependency locks and runtime/family/build/profiler/patch
source hashes. Build metadata must also supply an actual build fingerprint. The
future build integration must include CANN/toolchain/framework C++ ABI and build
options in that fingerprint; no build fingerprint is synthesized from source by
this baseline.

## Dependency transplant order

1. Keep the current MegaMoE scheduler/header, dynamic grouped-list sizing, pull
   protocol, completion handshake and hot-replica lifecycle as the runtime baseline.
2. Both locked ops-nn and ops-transformer source tags are v9.1.0.pre, but their
   adapter patch hashes differ by family. Compare each adapter semantically;
   preserve current MoE clamped SwiGLU behavior and grouped-matmul scratch fixes.
3. MHC additionally locks ops_transformer_mhc at
   `58b4a6bdeb29feeb0070dd266106bd4e130bb72b`. Move its numerical adapters with their
   dependency closure and cache/ring contracts, preserving isolated vendor paths.
4. Gate Route/RouteGrad need their own native family workers and host tiling. Keep
   the external CANN postprocessing, optional direct gradient, saved caches and
   disabled/profiled plan separation when adding the autograd adapter.
5. Rebuild affected native host/device artifacts after any numeric ID or field
   layout change. A legacy Gate image must never be submitted to a MoE binary.
   Build-produced family/schema/source metadata must be verified before binding.

The capture tool verifies every patch's actual bytes against its declared lock
hash. The initial host milestone did not transplant workers. Gate and MHC now have
isolated native builders; MoE reuses its current builder. Their artifact-bound
manifests retain separate family ABI and dependency identities.

## Independent snapshots and reproduction

The [capture tool](../../../../../tests/ut/core/multicore/backends/fixtures/capture_baselines.py)
reads the exact Git objects above. It executes trusted fixed-revision config,
serializer, profiler layout and Gate builder definitions in an isolated namespace,
without importing torch_npu. The DSL compiler does not use this execution path.

```bash
python tests/ut/core/multicore/backends/fixtures/capture_baselines.py
python -m pytest -q tests/ut/core/multicore/backends
```

Reproduction requires these Git objects locally. Normal tests use committed
snapshots and require no network or source Git objects. Fixtures contain original
C++ layout declarations and six immutable Gate runtime images: forward,
backward k greater than one and backward k equal to one, each with profiling
disabled/enabled. Metadata records image sizes and SHA256 digests.

CPU checks compile the extracted original C++ declarations and compare structure
sizes and every field offset to Python schemas. The new serializer is compared
byte for byte to the original builder snapshots. Row scheduling tests include
small token counts, tails, available worker counts and empty launched workers.

This completes host schema/source evidence and Gate snapshot/emission coverage.
The isolated Gate builder now emits artifact-bound manifests, and the Gate
runtime checks all payload hashes before loading. It exports pinned Gate sources
without modifying them, retaining original host tiling and CANN postprocessing;
its separate CMake wrappers exclude MoE/SHMEM from the native build. The active
Gate dependency is ops-nn LinearIndex; the full original dependency lock remains
part of the baseline source fingerprint. The wrappers and actual build inputs
are also recorded in the native build fingerprint.

Single-card Gate forward/backward acceptance is available through
`tests/torch/multicore/test_ast_gate.py`; see the
[前端原生绑定指南](../../frontend/README.md#当前实现边界)
for build, activation, stream and validation contracts. MoE now emits complete
current rank-local plans; MHC has sixteen independently captured fixed-revision
normal/profiled forward/backward snapshots and an isolated builder. Reproduce
MHC snapshots through `python -m tests.ut.core.multicore.backends.fixtures.capture_mhc_snapshots`.
Vision masking, other-device and performance validation remain outstanding.
`NativeManifest` alone is a metadata guard; family artifact admission additionally
checks the complete build-produced payload before loading.
