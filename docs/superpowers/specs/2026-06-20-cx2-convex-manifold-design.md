# Slice CX2 — Deterministic Convex Contacts: THE CONTACT MANIFOLD (clip the incident face → 1-4 pts) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of FLAGSHIP #19
> (DETERMINISTIC CONVEX RIGID-BODY CONTACTS, `hf::sim::convex`). CX1 gave the box-box SAT min-penetration axis +
> depth (or separation). CX2 turns that axis into the CONTACT MANIFOLD: the actual set of 1-4 contact POINTS (with
> per-point penetration depth) where the two boxes touch — the data CX3's angular impulse needs. Deterministic
> integer Sutherland-Hodgman face clipping, fixed vertex order. INTEGER-bit-exact. int64 → the
> `convex_manifold.comp` shader is Vulkan-only + a Metal CPU reference (the fpx_solve.comp / convex_sat.comp split).
> CX1's `convex.h` code + `convex_sat.comp` are BYTE-FROZEN (CX2 is additive). Branch: `slice-cx2`. See
> [[hazard-forge-convex-roadmap]].

**Goal:** Extend `engine/sim/convex.h` (additive — CX1 byte-unchanged) with a deterministic box-box CONTACT MANIFOLD:
`ContactManifold` (1-4 points + depths + count) + `BuildManifold(bodyA, boxA, bodyB, boxB, satResult)` → the
manifold (or empty if separated), plus a small measure. Add the new int64 shader `shaders/convex_manifold.comp.hlsl`
+ `--convex-manifold-shot` (Vulkan) / `--convex-manifold` (Metal). Bake the integer golden `convex_manifold`.
**NO new RHI.**

## Design call: reference-face / incident-face clipping in Q16.16 → 1-4 contact points

CX1's `SatResult{overlap, axisIndex, penetration, axis}` already tells us HOW the boxes overlap. CX2 builds the
manifold from it. The SAT axisIndex is 0..14 (the CX1 fixed order: 0..2 = A's face normals, 3..5 = B's face
normals, 6..14 = the 9 edge-edge crosses). Two cases:

### Case 1 — FACE contact (axisIndex 0..5): reference/incident face clip (the Sutherland-Hodgman core)
- **Reference box** = the box that OWNS the min-pen axis: A if `axisIndex ∈ 0..2`, B if `axisIndex ∈ 3..5`. The
  **reference normal** `n` = the world face axis for that index, SIGNED so it points from the reference box toward
  the incident box (CX1 already signs `SatResult.axis` toward B; re-derive the outward reference-face normal
  consistently: for an A-face the outward normal toward B is `+axA[i]` flipped to point at B; for a B-face the
  outward normal toward A is `+axB[j]` flipped to point at A — use `FxDot(axis, t)` with `t = b.pos − a.pos` to fix
  the sign, EXACTLY the CX1 sign rule).
- **Reference face** = the reference box's face whose outward normal is `n`: its center is `refBox.pos + n·H` where
  `H` = the reference half-extent along that local axis; its 4 corner vertices are
  `center ± u·hu ± v·hv` where `u,v` = the OTHER two world face axes of the reference box and `hu,hv` their
  half-extents (a FIXED corner order: `(+u+v), (−u+v), (−u−v), (+u−v)` — CCW, pinned).
- **Incident face** = the face of the INCIDENT box (the other box) whose outward world normal is MOST ANTI-PARALLEL
  to `n` — i.e. the incident box face axis `±axInc[k]` minimizing `FxDot(axInc_signed, n)` (most negative). FIXED
  tie-break: lowest local-axis index, then `+` before `−`. Its 4 corner vertices in the SAME fixed corner order.
- **Clip:** Sutherland-Hodgman clip the incident face's 4-vertex polygon against the 4 SIDE planes of the reference
  face (the planes through the reference-face edges, with inward normals `±u, ±v` in a FIXED plane order). Each
  side plane keeps the part of the polygon on its inner side; a crossing edge emits the intersection point
  (`fxdiv` the integer parametric `t`, int64). The Sutherland-Hodgman vertex/plane iteration order is PINNED so
  the output vertex list is bit-reproducible.
- **Keep penetrating points:** for each clipped polygon vertex, its penetration depth `d = FxDot(n, refFaceCenter −
  vertex)` (≥ 0 means below/inside the reference face). KEEP the vertices with `d ≥ 0`; their world contact point is
  the vertex projected onto/under the reference face (use the vertex position itself — document the choice), depth =
  `d`. This yields 0..8 candidate points.
- **Reduce to ≤ 4 (the deterministic cap):** ALWAYS keep the DEEPEST point (max `d`; tie → lowest clip-order
  index). Then keep up to 3 MORE in FIXED clip order until 4 total. (HONEST SIMPLIFICATION: a production solver
  keeps the deepest + the 3 that maximize the contact-patch area; CX2 keeps deepest + first-3-in-clip-order — a
  deterministic first cut; the area-maximizing reduction is a deferred refinement. Documented in the header + the
  caveat list.)

### Case 2 — EDGE-EDGE contact (axisIndex 6..14): the single closest-point contact
- The min-pen axis is `FxCross(axA[i], axB[j])` (i = (axisIndex−6)/3, j = (axisIndex−6)%3). The manifold is ONE
  contact point = the closest point between the two infinite edges' nearest segments. Compute the closest points of
  edge A_i (through A's center along `axA[i]`, length `2·hA[i]`) and edge B_j, the standard integer
  closest-point-of-two-segments (clamp the parametric `s,t` to the half-extents; `fxdiv`, int64). The contact point
  = the midpoint of the two closest points; depth = `SatResult.penetration`. count = 1. (Edge-edge box contacts ARE
  a single point — this is exact, not a simplification.)

**Output:** `ContactManifold { uint32 count; FxVec3 points[4]; fx depths[4]; FxVec3 normal; }` — `normal` = `n` (the
reference normal, or the signed edge-cross axis), `points`/`depths` filled 0..4, the rest zero. A SEPARATED pair
(`!satResult.overlap`) → `{count = 0}`. PURE INTEGER, FIXED order → bit-identical CPU↔Vulkan↔Metal.

**THE int64 REALITY (the CX1/FPX3 lesson):** the clip `fxdiv`/`FxDot`/`FxCross` Q16.16 products are int64. DXC
compiles int64 (Vulkan); glslc cannot. So `convex_manifold.comp` is **VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl)**; the
Metal `--convex-manifold` runs the CPU `BuildManifold` — byte-identical to the Vulkan GPU result BY CONSTRUCTION
(the convex_sat.comp convention), while the Vulkan side carries the GPU==CPU memcmp proof. Document this.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **CX1 `engine/sim/convex.h` (read it; do NOT edit CX1's code — APPEND only):** `FxDot`, `FxCross`, `FxBox
  {halfExtents}`, `BoxAxes(body, axesOut[3])`, `ProjectedRadius`, `SatResult{overlap, axisIndex, penetration,
  axis}`, `BoxSat`, `MeasureSat`, `kEdgeEps`, `SatPair{bodyA,boxA,bodyB,boxB}`. CX2 calls `BoxSat` to get the axis,
  then builds the manifold. The new code is APPENDED after `MeasureSat` (CX1's lines byte-frozen).
- **fpx Q16.16 + body toolbox (`engine/sim/fpx.h`, read-only):** `FxVec3`/`FxBody{pos, orient}`, `FxRotate`,
  `FxSub`/`FxAdd`/`FxScale`, `FxDot`/`FxNormalize`/`FxLength`, `fxmul`/`fxdiv`, `kOne`/`kFrac`. **DO NOT modify
  fpx.h.** (NOTE: FPX2 `BuildPairs` is available as the broadphase, but the CX2 SHOWCASE reuses the SAME fixed
  deterministic box-pair scene as CX1 — see decision 2 — so BuildPairs is optional; the manifold runs per-pair over
  the overlapping pairs.)
- **The new-shader showcase precedent (`samples/hello_triangle/main.cpp`):** study CX1's `--convex-sat-shot` (the
  per-pair compute + the GPU==CPU memcmp + the standalone arg-parse + the `SatResultGpu` 6×int32 pack) — CX2
  mirrors it with a `ManifoldGpu` pack (count + 4 points + 4 depths + normal, a fixed int32 layout). Confirm
  `convex_manifold` NOT in hf_gen_msl (grep 0).
- **Metal showcase (`metal_headless/visual_test.mm`):** mirror CX1's `--convex-sat` block — `--convex-manifold
  <out.png>` runs the CPU `BuildManifold` over the same scene + the same 2D top-down render.
- **Registration:** `scripts/verify.ps1` (the Mac golden loop + `$vkShots`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden — do NOT**), append to `tests/convex_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/convex.h`** (CX1 byte-frozen): `ContactManifold`, `BuildManifold(bodyA, boxA, bodyB,
   boxB, satResult)` (the face-clip + edge-edge cases above), `MeasureManifold(pairs)` → total contact points +
   count of pairs with a manifold (deterministic). FIXED corner order, FIXED clip plane order, FIXED reduction
   (deepest + clip-order). **NEW shader** `convex_manifold.comp.hlsl` (int64, Vulkan-only, one thread per box pair —
   runs `BoxSat` then `BuildManifold`, copies BOTH bodies VERBATIM). NOT in hf_gen_msl; Metal runs the CPU path.
2. **Showcase `--convex-manifold-shot <out>` (Vulkan) AND `--convex-manifold` (Metal) — WIRE BOTH** (standalone
   arg-parse). The SCENE: REUSE the SAME fixed deterministic box-pair array as CX1's `--convex-sat` (separated /
   deep face-face / edge-edge / touching). Vulkan: the GPU `convex_manifold.comp` → **memcmp the GPU
   `ManifoldGpu[]` vs the CPU `BuildManifold`** (NO tolerance — the make-or-break). Metal: the CPU reference.
   Render a PURE-INTEGER 2D top-down view: the box footprints (as CX1) PLUS each contact point as a small
   filled marker (e.g. a 3×3 dot) colored by depth, the manifold normal as a short segment. Golden =
   `tests/golden/metal/convex_manifold.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU `ManifoldGpu[]` == the CPU `BuildManifold` byte-for-byte. Print
     `convex-manifold: {pairs:<N>, withContact:<M>, points:<P>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `convex-manifold determinism: two runs BYTE-IDENTICAL`.
   - **(3) manifold correct:** every overlapping pair (per CX1 `BoxSat`) yields `count ≥ 1`; every separated pair
     yields `count == 0`; every contact point lies inside BOTH boxes' slabs within an integer epsilon (i.e. the
     point's projection onto each box axis is within `halfExtent + eps`) AND every depth ≥ 0. Print
     `convex-manifold correct: {allOverlapHaveContact:true, pointsInsideBoxes:true, depthNonNeg:true}`; assert all.
   - **(4) face-contact control:** a deep face-face pair (boxes stacked flat) yields a 4-POINT manifold (a full
     face patch). Print `convex-manifold control: {faceFace:4pts}`; assert the count for that known pair is 4.
   - **Golden discipline: ONLY `tests/golden/metal/convex_manifold.png`; do NOT commit it.** Existing 184 image
     goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests — APPEND to `tests/convex_test.cpp` (pure CPU):** a deep face-face stack → 4-point manifold, all depths
   ≥ 0, all points inside the slabs; a separated pair → count 0; an edge-edge overlap → count 1 at the expected
   midpoint; a touching face pair → ≥ 1 point with ~0 depth; the manifold is byte-identical across two runs; the
   reduction keeps the deepest point. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-convex-manifold` (features) + `--convex-manifold-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED.
  `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` + `vehicle.h` +
  `active.h` + `boids.h` + `engine/nav/` + `engine/anim/` + `engine/physics/` + **CX1's convex.h code +
  convex_sat.comp** + all EXISTING shaders UNCHANGED. The ONLY new shader is `convex_manifold.comp.hlsl` (int64,
  Vulkan-only, NOT in hf_gen_msl). Report the seam empty (only the convex.h APPEND + the new shader + the
  showcase/test/introspect are new/changed; CX1's convex.h lines byte-frozen).

## Out of scope (YAGNI — later CX slices)
The angular impulse / inertia-tensor solve (CX3 — CX2 only produces the contact points, NOT the velocity response;
`FxMat3` stays introduced-but-CX3-used), the full convex step (CX4), lockstep (CX5), the lit 3D render (CX6 — CX2's
render is the 2D manifold diagnostic). Arbitrary convex hulls / GJK / EPA (BOXES only). The area-maximizing 4-point
reduction (CX2 keeps deepest + clip-order — documented simplification). CX2 claims ONLY: a deterministic integer
box-box contact manifold (1-4 points + depths + normal, or empty), bit-identical CPU↔Vulkan↔Metal, with the
integer golden + the four proofs. NOTE (honest): boxes only; the 4-point cap is deepest+clip-order not
area-maximizing; int64 → Vulkan-GPU + Metal-CPU-ref (the FPX3 proof strength, not MSL-native strict-zero).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 105 incl. CX1's `convex_test` + the appended CX2 cases).
   Clean under `windows-msvc-asan` (build+run `convex_test` + `introspect_test`).
2. **proofs + visual:** `--convex-manifold-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   the box pairs WITH contact-point markers on the overlapping pairs (a coherent manifold diagnostic).**
3. Metal: `visual_test --convex-manifold` → new golden `tests/golden/metal/convex_manifold.png`; two runs DIFF
   0.0000. **Confirm `visual_test.mm` in the diff; confirm `convex_manifold.comp` NOT in `hf_gen_msl`.** Cross-vendor
   STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `convex_manifold.png` added; the
   other 184 byte-identical. `git diff master --stat -- tests/golden` = ONLY `convex_manifold.png` (metal) + the
   introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-convex-manifold` + `--convex-manifold-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`joint.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/
   `fract.h`/`vehicle.h`/`active.h`/`boids.h` + `engine/nav/` + `engine/anim/` + `engine/physics/` + **CX1's
   convex.h code + convex_sat.comp** + ALL existing shaders byte-unchanged). `scripts/verify.ps1` updated:
   `convex_manifold` golden in the Mac loop + `--convex-manifold-shot` in `$vkShots`. **The ONLY new shader is
   `convex_manifold.comp.hlsl` (int64, NOT in `hf_gen_msl`).**
