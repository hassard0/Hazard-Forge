# Slice FPX1 — Deterministic Fixed-Point Physics: Q16.16 INTEGRATOR + integer broadphase (Beachhead, Phase 11 #1) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The BEACHHEAD of FLAGSHIP #6:
> a DETERMINISTIC FIXED-POINT physics solver — the first physics in the engine that is BIT-IDENTICAL CPU↔Vulkan↔Metal
> AND frame-to-frame / run-to-run reproducible (unlike the existing FLOAT `engine/physics/` solver, explicitly held to
> the lesser "visually identical at rest" bar). This is the missing primitive that makes the engine's ALREADY-SHIPPED
> netcode (snapshot replication + client prediction) actually lockstep-correct: send inputs only, both peers
> re-simulate to identical state — a capability UE5's Chaos (float, non-bit-deterministic) cannot provide. This first
> slice builds the integer core: a Q16.16 fixed-point integrator (gravity + a floor clamp) + an integer broadphase
> grid cell quantizer, proven GPU==CPU BIT-EXACT, with an integer golden that is cross-backend BIT-IDENTICAL. Pure
> integer/fixed-point — NO collision response yet, NO new RHI. Namespace `hf::sim::fpx`. Branch:
> `slice-fpx-integrate`. Grounded by a read-only Plan scout @ master `ef06328` (2026-06-17). See [[hazard-forge-project]].

**Goal:** Add `engine/sim/fpx.h` (the pure-CPU fixed-point physics core: `fx` Q16.16 scalar + `fxmul`, `FxVec3`,
`FxBody`, `FxWorld`, `IntegrateStep` CPU reference, `BroadphaseCell`/`CellId`, `FxISqrt` for length) +
`shaders/fpx_integrate.comp.hlsl` (one thread per body, the integrator copied VERBATIM) + a `--fpx-shot` (Vulkan) /
`--fpx` (Metal) showcase that drops a deterministic grid of bodies, integrates K fixed steps, reads back the body
SSBO, proves it BIT-EXACT vs the CPU reference, and bakes an integer body-position debug-viz golden. Make-safe: a NEW
header + NEW shader + NEW showcase + NEW golden; the existing FLOAT `engine/physics/` is UNTOUCHED (fpx is additive +
parallel). The cross-backend bit-identity is guaranteed by the GPU consuming host-snapped Q16.16 integers + doing ZERO
float.

## Why this is the right beachhead (the VT1/MC1 integer template, + de-risked)
The existing physics is decisively float: `engine/physics/body.h:21-32` (`RigidBody` = `math::Vec3 position` etc.,
float), `engine/physics/world.cpp:72-209` (`Step(float dt)`, `std::sqrt`-based length), and the design spec concedes
"cross-platform need only be visually identical at the settled rest state" (`2026-06-14-physics-design.md:76-77`). A
fixed-point solver is a genuinely-new, distinct flagship. The hardest part (integer sqrt + a Q-format) is ALREADY
proven in-engine: `engine/render/mc.h:461-472` (`ISqrt`, integer binary sqrt), `mc.h:457` (`kFixed` Q-format),
`engine/audio/mixer.h:6-12` (the Q15/int32-accumulator "no float anywhere" discipline), `engine/render/swraster.h:151`
(int64 fixed-point intermediates + host-snap). This slice is the integrator — the direct analog of VT1's marking /
MC1's classification: an order-independent per-body integer update proven GPU==CPU bit-exact (each body is independent
in FPX1 — no inter-body coupling until FPX3's collision response, so the GPU write is trivially race-free).

## The integer/fixed-point core (the cross-backend crux)
- **`using fx = int32_t;`** a Q16.16 fixed-point scalar (`kFrac = 16`, `kOne = 1 << 16`). `fx fxmul(fx a, fx b) {
  return (fx)(((int64_t)a * b) >> kFrac); }` — **int64 intermediate** (the swraster `SwEdge` / mc `d2` pattern,
  `swraster.h:151`), then shift. Document the Q16.16 range: positions cap at ±32768 world units; all products use
  int64 → no overflow within the bound. (Truncating `>>` is an arithmetic right shift on int64 — deterministic; pin
  it identically CPU↔HLSL/MSL.)
- **`struct FxVec3 { fx x, y, z; };`** the integer twin of `math::Vec3` — pure add/sub/`fxmul`-by-scalar. NO
  `std::sqrt`: `fx FxLength(const FxVec3&)` uses `FxISqrt` (the `mc.h::ISqrt` floor-integer sqrt copied verbatim, on
  the int64 sum-of-squares in Q-format). (FxLength is provided for FPX2/FPX3; FPX1 only needs add/`fxmul`.)
- **`struct FxBody { FxVec3 pos, vel; fx invMass; uint32_t flags; };`** (`flags` bit 0 = dynamic; `invMass==0` ⇒
  static/kinematic). Orientation is deferred to FPX4 (FPX1 is translational, like MC1 classified before geometry).
- **`struct FxWorld { FxVec3 gravity; fx groundY; std::vector<FxBody> bodies; };`**
- **`void IntegrateStep(FxWorld& w, fx dt)`** — the deterministic semi-implicit-Euler integrator (the VERBATIM math
  the shader copies): for each body, if dynamic (`flags & 1`): `vel += gravity * dt` (component-wise `fxmul`); `pos +=
  vel * dt`; then a single non-penetration FLOOR clamp: `if (pos.y < groundY) { pos.y = groundY; if (vel.y < 0)
  vel.y = 0; }` (pure integer compare/min — no float, no restitution yet). Fixed iteration order, no RNG, no clock.
  Each body independent → order-independent → two-run bit-identical + GPU-thread-race-safe.
- **`int3 BroadphaseCell(const FxVec3& p, fx cellSize)`** = integer `(p.x / cellSize, p.y / cellSize, p.z /
  cellSize)` (truncating integer divide, the `vt.h:154` `VtPageId` floor-quantize analog — use a deterministic floor
  for negatives, e.g. the swraster `FloorDiv`, copied verbatim CPU↔GPU); **`uint CellId(int3 cell, int3 gridDim)`** =
  the flat linearization (the `mc.h:114` cell-id pattern). (FPX1 exercises + unit-tests these; FPX2 builds the
  broadphase PAIR list on them. The beachhead's GPU pass is the integrator; the broadphase cell math is a tested
  header helper.)

## Reuse map (file:line)
- **The integer-sqrt + Q-format groundwork (copy):** `engine/render/mc.h:461-472` (`ISqrt`), `mc.h:457` (`kFixed`),
  `mc.h:618` (the truncating integer-divide pattern); `engine/render/swraster.h:90,151` (`kSub` fixed-point + int64
  intermediates + `FloorDiv`); `engine/audio/mixer.h:6-12` (the Q-format no-float discipline).
- **The float solver ALGORITHM to mirror in fixed-point (READ, do NOT modify):** `engine/physics/world.cpp:72-209`
  (`Step`: integrate → clamp); `engine/physics/body.h:21-32`. fpx is a fixed-point reimplementation, NOT an edit.
- **The VT1/MC1 beachhead template (structure + four proofs):** `engine/render/vt.h` (namespace + CPU-reference +
  host-snap + GPU-mirror + GPU==CPU/disabled/determinism/known proofs); `shaders/mc_classify.comp.hlsl:22-24,36-38,75`
  (the 3-SSBO + enabled-flag + no-op shader template, seam discipline).
- **The compute + readback surface (NO new RHI):** `BufferUsage::Storage` (`rhi.h:166`), `CreateComputePipeline`
  (`rhi.h:477`), `BindStorageBuffer`/`ComputePushConstants`/`DispatchCompute` (`rhi.h:414,416,426`), `ReadBuffer`
  (`rhi.h:616`), `ComputeToComputeBarrier` (`rhi.h:434`).
- **The integer-readback debug-viz golden (CPU-colored from integers → bit-identical):** the swraster host-snap
  `pos >> kFrac → pixel` (`swraster.h:132-139`); `meshlet.h:79` `hashColor`.
- **The lockstep "beyond UE5" hook (FPX5 target, NOT this slice):** `engine/net/snapshot.h:99`
  (`Replicator::Capture(... const physics::World&)`).
- **Showcase + registration patterns:** the MC1 `--mc-classify-shot`/`--mc-classify` showcase + introspect +
  `verify.ps1` shapes.

## Design decisions (locked)

1. **`engine/sim/fpx.h` (NEW dir `engine/sim/`, namespace `hf::sim::fpx`, pure CPU, header-only, 0 above-seam backend
   symbols, mirrors `vt.h`/`mc.h`).** The Q16.16 `fx`/`fxmul`, `FxVec3`, `FxISqrt`/`FxLength`, `FxBody`, `FxWorld`,
   `IntegrateStep`, `BroadphaseCell`/`CellId`/`FloorDiv`. NO `<cmath>` on the bit-exact path (FxISqrt, not std::sqrt).
   Register the new `engine/sim/` dir in CMake (the `hf_engine`/`hf_core` lib + the test include path) the way
   `engine/render/` headers are picked up — header-only, so just an include path if needed (mirror how mc.h/vt.h are
   consumed; they're header-only under engine/render — confirm whether engine/sim needs a CMake add or is header-only-
   reachable; prefer header-only + add the include dir if the tests/showcase can't find it).
2. **`shaders/fpx_integrate.comp.hlsl` (NEW).** ONE thread per body (`i < bodyCount`). Reads `gBodies` (b0, the
   Q16.16 `FxBody` array: pos.xyz, vel.xyz, invMass, flags — std430-packed ints), `gParams` (b1: gravity.xyz, dt,
   groundY, bodyCount, steps, integrateEnabled). Runs `steps` iterations of `IntegrateStep`'s body (the `fxmul` +
   integrate + floor-clamp copied VERBATIM) on its own body, writes `gBodies[i]` back. `integrateEnabled=0` → write
   the input back unchanged (the disabled no-op). `ComputePipelineDesc{ storageBufferCount=2, threadsPerGroupX=64 }`.
   NO atomics (each body independent, disjoint writes). Only `[[vk::binding]]` + `HF_MSL_GEN` above-seam. Plain integer
   (int + int64 via the fxmul `>>` on int64; if int64 trips glslc, apply the swraster Vulkan-SPIR-V-only + Metal-CPU-
   path convention — but `fxmul`'s int64 is a single mul+shift; PREFER it compiles; report). NO `--msl-version 20200`.
3. **Showcase `--fpx-shot <out>` (Vulkan, main.cpp) AND `--fpx` (Metal, visual_test.mm — WIRE BOTH; confirm
   visual_test.mm + `#include "sim/fpx.h"`).** A deterministic `FxWorld`: gravity = `(0, -9.8*kOne approximated in
   Q16.16, 0)` (host-precompute the Q16.16 constant), `groundY = 0`, a fixed M-body grid (e.g. an 8×8 grid of dynamic
   bodies at staggered heights `y = (8 + (i%5))*kOne`, x/z spread on a grid). dt = `kOne/60` (Q16.16 1/60). Upload
   `gBodies` + `gParams{steps=K}` (e.g. K=120), dispatch `fpx_integrate` (one dispatch, K steps in the per-thread
   loop), `ReadBuffer` `gBodies`. CPU-run `IntegrateStep` K times over the SAME world → the reference. Golden = a
   PURE-INTEGER side-view debug-viz: project each body's integer `(pos.x >> kFrac, pos.y >> kFrac)` to a pixel
   (mapped into the image with a fixed integer transform), splat a small `hashColor(bodyIndex)` dot, draw the
   `groundY` line → `tests/golden/metal/fpx.png` (the bodies fallen/settled near the ground; CPU-colored from the
   integer read-back → identical both backends by construction).
4. **PROOFS (fail loudly; exact print lines):**
   - **(1) GPU==CPU bit-exact (make-or-break):** `memcmp(gpuBodies, cpuBodies) == 0` over all bodies after K steps
     (Q16.16 ints, NO tolerance). Print `fpx GPU==CPU bodies: <N> bodies BIT-EXACT (<K> steps)`.
   - **(2) disabled-path no-op:** `integrateEnabled=false` → `gBodies` byte-identical to the upload. Print `fpx
     disabled: bodies UNCHANGED (no-op)`.
   - **(3) determinism:** two K-step runs → byte-identical body array. Print `fpx determinism: two runs
     BYTE-IDENTICAL`.
   - **(4) hand-checked closed form:** a single body at `pos.y = H*kOne`, `vel=0`, gravity `g`, dt `1/60`, after K
     steps → assert `pos.y` equals the hand-computed Q16.16 value (the semi-implicit-Euler closed form `y_k = y_0 +
     Σ`, clamped at `groundY`). Print `fpx hand-check: body.y = <q16.16> == closed-form OK`.
   - **(5) broadphase bijection:** `BroadphaseCell`/`CellId`/`FloorDiv` over known positions → known cells (unit-test
     + a printed `fpx broadphase: <C> cells, FloorDiv neg OK`).
   - **(6) {stats}:** `fpx: {bodies:<N>, steps:<K>, settled:<m>/<N>}` (settled = bodies at `pos.y == groundY`).
   - **Golden discipline: ONLY `tests/golden/metal/fpx.png`; do NOT commit it — the CONTROLLER bakes on the Mac.**
     Existing 105 image goldens UNTOUCHED.
5. **Determinism / cross-backend.** The integrator is pure integer/fixed-point on host-snapped Q16.16 inputs (the GPU
   does ZERO float); each body is independent (disjoint writes, no atomics); the golden is CPU-colored from the
   integer read-back → bit-identical Vulkan/Metal AND the GPU==CPU memcmp holds. Run under the Vulkan sync-validation
   gate → the upload→dispatch→readback barriers SYNC-HAZARD-free.
6. **Tests `tests/fpx_test.cpp` (pure CPU, NEW):** `fxmul` round-trip + known products (incl negatives, the int64
   intermediate); `FxISqrt` matches `mc::ISqrt` semantics on known squares; `IntegrateStep` one-step + K-step closed
   form (a body falls the exact Q16.16 distance, clamps at `groundY`, `vel.y` zeroed on contact); `integrateEnabled`-
   off modeled → unchanged; determinism (two runs); `BroadphaseCell`/`CellId`/`FloorDiv` (incl negative coords →
   correct floor); overflow-bound assertion within the documented range. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-fixedpoint-physics-integrate` (features) + `--fpx-shot` (showcases).

## RHI seam additions (summary)
- **None.** `BufferUsage::Storage` + `CreateComputePipeline`/`BindStorageBuffer`/`DispatchCompute`/`ReadBuffer` — the
  VT1/MC1/froxel precedent. New non-backend code adds ZERO above-seam backend symbols. `engine/rhi/rhi.h` +
  `rhi_factory` (dispatch baseline 2) + the backend dirs UNCHANGED. The existing `engine/physics/` UNTOUCHED. Report
  the seam.

## Out of scope (YAGNI — FPX2 and beyond)
Broadphase PAIR generation (FPX2 — FPX1 only assigns cells), collision RESPONSE / sequential-impulse or PBD (FPX3, the
make-or-break — sphere–plane first), fixed-point ORIENTATION / quaternion + box-SAT (FPX4), the lockstep replica==
authority proof via `net::Replicator` (FPX5, the "beyond UE5" headline), the float instanced-lit render of the
settled pile (FPX6, the only float slice). Friction/contact-manifold physical accuracy (claim DETERMINISM +
cross-platform BIT-IDENTITY, NOT "more physically correct than Chaos"). ONE fixed-point integrator + integer
broadphase cell math with the GPU==CPU bit-exact proof + disabled no-op + determinism + the hand-checked closed form
and the integer body-position golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 91) + the new `fpx_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--fpx-shot` on Vulkan: a coherent side-view of bodies fallen/settled near the ground line
   (deterministic dot positions); `fpx GPU==CPU bodies: <N> bodies BIT-EXACT` + `disabled` + `determinism` +
   `hand-check` + `broadphase` + the `{...}` line. Run under the Vulkan-validation gate → ZERO VUID in the OUTPUT
   (gate on the output grep, not the exit code — the layer may teardown-crash per the MC ops lesson).
3. Metal: `visual_test --fpx` → new golden `tests/golden/metal/fpx.png`; two runs DIFF 0.0000 (gate on compare.sh EXIT
   CODE). The GPU==CPU + determinism proofs also pass on Metal (integer math). **Confirm visual_test.mm in the diff;
   confirm fpx_integrate.comp MSL-generates + the int64 `fxmul` lowers on Metal (single mul+shift — should need no
   MSL-2.2; if int64 trips glslc, apply the Vulkan-only + Metal-CPU-path convention + report).** This is an INTEGER
   golden → a strict cross-backend pixel compare must show ZERO differing pixels.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `fpx.png` added; the other 105
   byte-identical. `git diff master --stat -- tests/golden` = ONLY `fpx.png` (metal) + the 2-line introspect json — NO
   loose `tests/golden/fpx.png`, NO other golden changed. `engine/physics/` UNCHANGED.
5. Introspect JSON rebaked exactly `+deterministic-fixedpoint-physics-integrate` + `--fpx-shot`; introspect test
   updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated to include the new `fpx` image
   golden in the Mac round-trip loop AND `--fpx-shot` in the `$vkShots` validation gate. (If `--fpx-shot` tips
   `main()`'s stack on the debug build, the `/STACK:16777216` link option is already in place from MC6.)
