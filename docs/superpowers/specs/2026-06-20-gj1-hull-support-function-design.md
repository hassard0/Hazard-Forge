# Slice GJ1 — General Convex-Hull Contacts: THE HULL + SUPPORT FUNCTION (the beachhead) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice (the BEACHHEAD) of FLAGSHIP #22
> (DETERMINISTIC GENERAL CONVEX-HULL CONTACTS via integer GJK + EPA, `hf::sim::gjk`). The whole convex/friction/
> persist contact stack (#19-21) is BOX-ONLY (box-box SAT). This flagship generalizes the ONE box-only component
> — the narrowphase — to arbitrary convex polyhedra. GJ1 builds the single primitive GJK and EPA are both built
> on: the **support function** — the world-space vertex of a convex hull farthest along a direction — plus the
> Minkowski-difference (configuration-space-obstacle) support that is GJK/EPA's only geometry call. NEW header
> `engine/sim/gjk.h`, namespace `hf::sim::gjk`, `#include "sim/convex.h"` READ-ONLY (which transitively gives the
> fpx Q16.16 toolbox). convex.h / fric.h / persist.h / fpx.h stay BYTE-FROZEN. Branch: `slice-gj1`. See
> [[hazard-forge-gjk-roadmap]], [[hazard-forge-docs-style]].

**Goal:** Create `engine/sim/gjk.h` with `kMaxHullVerts` + `FxHull` (a fixed-cap local-space vertex array) +
`SupportLocal(hull, dir)` (the local-space farthest vertex — argmax of `convex::FxDot`, fixed scan, lowest-index
tie-break) + `Support(hull, body, dir)` (the world-space support, via `fpx::FxRotate`) + `SupportMinkowski(hullA,
bodyA, hullB, bodyB, dir)` (the CSO support `Support(A,dir) − Support(B,−dir)`) + a `HullMeasure` summary. Add the
int64 GPU shader `shaders/gjk_support.comp.hlsl` (Vulkan-only) that copies the CPU support path VERBATIM, plus the
showcase `--gjk-support-shot` (Vulkan) / `--gjk-support` (Metal). Bake the integer golden `gjk_support`.

## Design call: the support function is GJK/EPA's only geometry primitive — get it bit-exact first

A convex hull's support in direction `d` is the vertex maximizing `dot(vertex, d)`. Everything in GJK (simplex
evolution) and EPA (polytope expansion) reduces to repeated support queries on the **Minkowski difference**
A⊖B, whose support is `Support_A(d) − Support_B(−d)`. GJ1 nails this primitive — deterministic, bit-exact
CPU↔Vulkan↔Metal — before any simplex/polytope logic is built on it. The box had no support function (the box
was implicit in `convex::BoxAxes`/`ProjectedRadius`); this is genuinely new code.

- **`constexpr uint32_t kMaxHullVerts = 20;`** — the fixed compile-time cap (GPU buffers are fixed-size; 20 verts
  covers tetra/octa/cube/wedge/dodecahedron — the documented ceiling, identical CPU/GPU). Pick 20 unless the
  implementer finds a sibling constant precedent; document the choice.
- **`struct FxHull { convex::FxVec3 verts[kMaxHullVerts]; uint32_t count; };`** — a convex hull as its
  LOCAL-space vertices (body-relative, the hull is immutable/shared like a box's half-extents). `count` ≤
  `kMaxHullVerts`. (Faces are NOT needed for GJ1 — the support function is purely over vertices; faces arrive in
  GJ3/GJ4.)
- **`convex::FxVec3 SupportLocal(const FxHull& hull, convex::FxVec3 dir);`** — the local-space vertex with the
  maximum `convex::FxDot(hull.verts[i], dir)` over `i ∈ [0, count)`, scanned in FIXED index order with a
  **strict-greater** update so ties keep the LOWEST index (the `convex.h` min-pen tie-break idiom, convex.h:28).
  Deterministic, pure integer (the argmax compare is int32; the `FxDot` it ranks is int64).
- **`convex::FxVec3 Support(const FxHull& hull, const fpx::FxBody& body, convex::FxVec3 dir);`** — the WORLD-space
  support: rotate `dir` into the body's local frame (by the conjugate of `body.orient`), call `SupportLocal`,
  then map that local vertex to world (`fpx::FxRotate(body.orient, v) + body.pos`). Rotating the direction in
  (one vector) rather than every vertex out is fewer ops and identical result. **Decide and document** the exact
  conjugate-rotation helper (grep `fpx.h` for an existing inverse/conjugate-rotate; if none, rotate by the
  conjugate quaternion built inline — do NOT modify fpx.h). The result must be bit-identical on CPU and the GPU
  shader.
- **`convex::FxVec3 SupportMinkowski(const FxHull& hullA, const fpx::FxBody& bodyA, const FxHull& hullB, const
  fpx::FxBody& bodyB, convex::FxVec3 dir);`** — `convex::FxSub(Support(hullA, bodyA, dir), Support(hullB, bodyB,
  convex::FxNeg/negate(dir)))` (negate `dir` by component negation — grep for an existing negate helper or negate
  inline). The single geometry call GJK/EPA make.
- **`struct HullMeasure { uint32_t hulls; uint32_t dirs; convex::FxVec3 supportSum; convex::fx extentSum; };`** (or
  similar) — a deterministic summary the showcase asserts: over a fixed hull set × a fixed direction set, the sum
  of returned support vertices + the sum of support extents (`FxDot(support, dir)`). The exact fields are the
  implementer's call; it must be a pure function of the inputs.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **convex.h (read-only — REUSE, do NOT redefine):** `convex::FxVec3` + `convex::FxDot` (convex.h:54) + `FxSub`/
  `FxAdd`/`FxScale` + `convex::fx` (the Q16.16 scalar). `FxHull`/`Support*` go in the NEW `gjk.h`, NOT convex.h.
- **fpx.h (read-only):** `fpx::FxBody` (fpx.h:116 — pos/orient/vel/...), `fpx::FxRotate` (fpx.h:440 — rotate a
  vector by a quaternion; the support's world transform), the Q16.16 ops. Grep for a conjugate/inverse-rotate +
  a vector-negate helper; if absent, express inline WITHOUT modifying fpx.h.
- **The proof-tier convention (convex.h:16-23 — READ IT):** int64 world-scale products → `shaders/gjk_support.comp.hlsl`
  is **VULKAN-SPIR-V-ONLY** (in the Vulkan compile list, NOT in `metal_headless` `hf_gen_msl`); the Metal
  `--gjk-support` runs the CPU `Support` path → byte-identical BY CONSTRUCTION (the `convex_sat.comp` /
  `fpx_solve.comp` split), while the Vulkan side carries the **GPU==CPU memcmp** proof. The shader copies the CPU
  `SupportLocal`/`Support` body VERBATIM so the GPU exercises the exact integer ops.
- **The showcase + shader precedent:** `--convex-sat-shot` (Vulkan, `samples/hello_triangle/main.cpp`) /
  `--convex-sat` (Metal, `metal_headless/visual_test.mm`) + `shaders/convex_sat.comp.hlsl` — the one-thread-per-pair
  int64 compute proof with the GPU==CPU memcmp + the integer diagnostic render. Mirror its structure for
  `--gjk-support`: one GPU thread per (hull, direction) query, write the support vertex into an output buffer,
  memcmp vs the CPU `Support`.
- **Registration:** `scripts/verify.ps1` (the `@{Name; Flag}` Mac golden table ~:158-177 — append `gjk_support`;
  add `--gjk-support-shot` to `$vkShots`), `metal_headless/CMakeLists.txt` (`hf_gen_msl` :41-48 — **do NOT** add
  `gjk_support.comp` there; it is Vulkan-only), `engine/editor/introspect.cpp` + `tests/introspect_test.cpp`
  (**controller rebakes the JSON golden — do NOT**), a NEW `tests/gjk_test.cpp` (+ register it in the test
  CMake like `persist_test`/`convex_test`).

## Design decisions (locked)
1. **NEW header `engine/sim/gjk.h`** (namespace `hf::sim::gjk`, `#include "sim/convex.h"` read-only): `kMaxHullVerts`,
   `FxHull`, `SupportLocal`, `Support`, `SupportMinkowski`, `HullMeasure` + a `MeasureSupport` helper. convex.h /
   fric.h / persist.h / fpx.h BYTE-FROZEN. A small set of canonical test hulls (unit tetra / octa / cube / wedge)
   may be provided as helper builders in `gjk.h` (e.g. `MakeTetra()`/`MakeBox(he)`) OR in the test — implementer's
   call; they must be deterministic integer constructions.
2. **New shader `shaders/gjk_support.comp.hlsl` (int64, VULKAN-ONLY)** — copies the CPU `Support` path verbatim;
   one thread per (hull, direction) query; writes the support vertex (+ its extent dot) to an SSBO. NOT added to
   `hf_gen_msl` (Metal runs the CPU path). NO new RHI (rides the existing compute-dispatch + SSBO binding the
   `convex_sat` shot uses — confirm the seam is unchanged).
3. **Showcase `--gjk-support-shot <out>` (Vulkan) AND `--gjk-support` (Metal) — WIRE BOTH.** Build a fixed scene:
   the canonical hull set at fixed poses (a couple rotated, to exercise `FxRotate`) × a fixed direction sweep
   (e.g. the 6 axes + a few diagonals). Vulkan dispatches `gjk_support.comp` and memcmps the GPU support buffer
   vs the CPU `Support`; Metal runs the CPU `Support`. BOTH render an integer diagnostic (the hulls' support
   points marked per direction — mirror `convex_sat`'s diagnostic). Golden = `tests/golden/metal/gjk_support.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU:** `gjk-support: {hulls:<H>, dirs:<D>, queries:<Q>} GPU==CPU BIT-EXACT` — the GPU support
     buffer == the CPU `Support` buffer byte-for-byte; assert.
   - **(2) determinism:** two runs identical. `gjk-support determinism: two runs BYTE-IDENTICAL`.
   - **(3) correctness:** the returned support is a genuine maximum — for each query, NO hull vertex has a
     strictly greater `FxDot(vert, dir)` than the returned one (a CPU brute-force check over all verts). Print
     `gjk-support correct: {maximalAll:true}`; assert. Also assert the Minkowski support equals
     `Support_A − Support_B(−dir)` for a sample pair.
   - **Golden discipline: ONLY `tests/golden/metal/gjk_support.png`; do NOT commit it.** Existing 203 image
     goldens UNTOUCHED.
5. **Cross-backend bar (int64 → strict):** Vulkan GPU==CPU bit-exact; Metal CPU-ref byte-identical by
   construction → the golden is bit-identical cross-backend; cross-vendor ZERO differing pixels.
6. **Tests — NEW `tests/gjk_test.cpp`:** `SupportLocal` returns the maximal vertex (vs brute force) over the
   canonical hulls + a direction sweep, ties resolve to the lowest index; `Support` matches a hand-computed
   world support for a rotated body; `SupportMinkowski` == `Support_A − Support_B(−dir)`; `MeasureSupport` is a
   pure function (two calls equal). Clean under `windows-msvc-asan`. Register `gjk_test` in the test CMake.
7. **Introspect.** Add exactly `deterministic-hull-support` (features) + `--gjk-support-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** The shot rides the existing compute-dispatch + SSBO path (the `convex_sat` shot's seam). `rhi.h` +
  backend dirs UNCHANGED. `engine/sim/convex.h` + `fric.h` + `persist.h` + `fpx.h` + ALL other sim headers +
  ALL existing shaders + `engine/physics/`/`nav/`/`anim/` UNCHANGED. NEW files only: `engine/sim/gjk.h`,
  `shaders/gjk_support.comp.hlsl`, `tests/gjk_test.cpp` (+ the showcase/introspect/verify edits). Report the seam:
  one new Vulkan-only shader, no RHI change, no frozen-file edit.

## Out of scope (YAGNI — later slices)
GJK simplex evolution (GJ2), EPA (GJ3), the hull world step + manifold + hull inertia (GJ4), lockstep (GJ5), lit
render (GJ6). GJ1 claims ONLY: a deterministic, bit-exact (CPU↔Vulkan↔Metal) convex-hull support function + its
Minkowski-difference support, with the integer golden + the three proofs. NOTE: convex polyhedra only (no curved
shapes — their support is not a vertex-argmax); the hull vertex cap is `kMaxHullVerts`.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing + the new `gjk_test`). Clean under `windows-msvc-asan`
   (build+run `gjk_test` + `introspect_test`).
2. **proofs + visual:** `--gjk-support-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID. **VERIFY the diagnostic image shows the canonical hulls with their per-direction support points
   marked (coherent, no garbage/NaN).**
3. Metal: `visual_test --gjk-support` → new golden `tests/golden/metal/gjk_support.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `gjk_support.comp` is NOT in `hf_gen_msl` (Vulkan-only).**
   Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `gjk_support.png` added; the
   other 203 byte-identical. `git diff master --stat -- tests/golden` = ONLY `gjk_support.png` (metal) + the
   introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-hull-support` + `--gjk-support-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + `engine/sim/convex.h`/`fric.h`/`persist.h`/`fpx.h` + ALL other sim headers + ALL
   existing shaders + `engine/physics/`/`nav/`/`anim/` byte-unchanged; ONE new Vulkan-only shader, no RHI change).
   `scripts/verify.ps1` updated: `gjk_support` golden in the Mac loop + `--gjk-support-shot` in `$vkShots`.
   `gjk_support.comp` NOT in `hf_gen_msl`.
