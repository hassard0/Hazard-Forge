# Slice BP6 — Deterministic Integer Broadphase: THE LIT 3D RENDER CAPSTONE (the money-shot) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice
> of FLAGSHIP #23 (DETERMINISTIC INTEGER BROADPHASE, `hf::sim::broad`). BP1-BP5 built the grid, the candidate-pair
> generator (proven equivalent to all-pairs), the broadphase-driven box + hull world steps (proven byte-identical
> to all-pairs at scale), and lockstep/rollback through the broadphase. BP6 is the money-shot: it renders the
> bit-exact **broadphase-driven settled hull pile** as a LIT 3D scene — a heap of mixed convex polyhedra that the
> O(n²) all-pairs path could never have stepped, settled by the broadphase-accelerated solver and drawn as true
> polyhedra under directional light. The render is the ONE FLOAT crossing (outside the bit-exact integer loop),
> so its bar is the FLOAT visresolve in-band metric, NOT strict-zero. PURE REUSE: the settled world is a
> `gjk::HullWorld`, so BP6 delegates VERBATIM to the FROZEN GJ6 render bridge `gjk::HullToRenderInstances` (the
> existing instanced-lit pipeline). ZERO new render shader, ZERO new RHI. APPEND to `engine/sim/broad.h` (BP1-BP5
> + gjk.h/convex.h/fric.h/persist.h/fpx.h/grain.h BYTE-FROZEN). Branch: `slice-bp6`. See
> [[hazard-forge-broad-roadmap]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/broad.h` (additive — BP1-BP5 byte-unchanged) with `BroadToRenderInstances(const
gjk::HullWorld& world)` — a one-line delegate to the frozen `gjk::HullToRenderInstances` (render-only, a PURE
FUNCTION — two calls byte-equal). Add the showcase `--broad-render-shot <out>` (Vulkan) / `--broad-render`
(Metal) — both build the BP4 mixed-hull scene, CPU-settle via `StepHullWorldBPN` (the broadphase-driven step),
and draw the settled pile LIT 3D through the EXISTING instanced-lit pipeline. Bake the float golden `broad_render`.

## Design call: render the broadphase-settled pile; the sim stays bit-exact

The bit-exact settled `gjk::HullWorld` is BP4's deterministic Q16.16 result (via the broadphase-driven step). BP6
adds ONLY a render bridge: it maps that frozen world to FLOAT geometry for display. The sim is NOT mutated; the
provenance contract (two `BroadToRenderInstances` calls on the same world → byte-equal output) proves the render
is a pure function of the deterministic sim.
- **`BroadToRenderInstances(const gjk::HullWorld& world)`** = `return gjk::HullToRenderInstances(world);` (the
  FC6/GJ6 idiom — render-only, OUTSIDE the bit-exact loop). MATTE (the GJ6 material) to dodge the iridescence
  trap. Provided in `broad::` for namespace symmetry; the actual mesh comes from the byte-unchanged GJ6 helper.
- **The scene = the broadphase-settled pile.** Build the BP4 mixed-hull scene (a static floor + ~48-64 dynamic
  tetra/octa/wedge/box), CPU-settle via `broad::StepHullWorldBPN` to the converged rest, draw
  `BroadToRenderInstances(world)` lit. The money-shot: a heap of mixed convex polyhedra at rest — stepped by the
  broadphase-accelerated solver (the scaling payoff), lit in 3D.

> NOTE: BP6 settles ON THE CPU (the bit-exact `StepHullWorldBPN`) and renders the result — the render is a normal
> draw, NOT a heavy compute dispatch, so NO TDR concern (unlike BP3/BP4's GPU step shots).

## Reuse map (file:line)
- **BP1-BP5 `engine/sim/broad.h` (APPEND after BP5's `RunBroadRollback`):** `gjk::HullWorld`, `StepHullWorldBPN`,
  `BroadStepConfig`. BP1-BP5 frozen.
- **gjk.h GJ6 render bridge (read-only — REUSE verbatim):** `gjk::HullToRenderInstances` (the settled-hull →
  float world-space triangle mesh, matte), `gjk::HullRenderMesh`/`HullRenderMeshEqual`. The GJ6
  `--gjk-render-shot`/`--gjk-render` showcases are the precedent (the lit instanced 3D draw + the float
  visresolve-bar + the provenance proof). Mirror for `--broad-render`.
- **Registration:** `scripts/verify.ps1` (append `broad_render` to the Mac loop as a plain entry like `gjk_render`
  — NO special threshold; the committed golden is the Mac bake, verify re-renders + compares at 0.0000 same-backend
  — + `--broad-render-shot` to `$vkShots`), `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden**), append to `tests/broad_test.cpp`. (No new shader → nothing for `hf_gen_msl`.)

## Design decisions (locked)
1. **APPEND to `engine/sim/broad.h`** (BP1-BP5 byte-frozen): `BroadToRenderInstances` (the one-line GJ6 delegate).
   Render-only float, OUTSIDE the bit-exact loop. **NO new shader, NO new RHI** (reuse the instanced-lit pipeline).
2. **Showcase `--broad-render-shot <out>` (Vulkan) AND `--broad-render` (Metal) — WIRE BOTH (grep your own
   visual_test.mm for `--broad-render` before reporting DONE).** BOTH build the BP4 scene, CPU-settle via
   `StepHullWorldBPN`, draw `BroadToRenderInstances(world)` LIT 3D (matte, directional light) through the EXISTING
   pipeline. Golden = `tests/golden/metal/broad_render.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly):**
   - (1) provenance: `broad-render: {hulls:<H>, tris:<T>} provenance two-calls BYTE-EQUAL`.
   - (2) non-trivial: `broad-render: {dynamic:<D>, restedPile:true}` (the dynamic hulls rest above the floor).
   - (3) determinism: the Vulkan render path exits 0 + writes the image; Metal two-run DIFF 0.0000 (controller).
   - Golden discipline: ONLY `tests/golden/metal/broad_render.png`; do NOT commit it. Existing 214 goldens
     UNTOUCHED.
4. **Cross-backend bar:** the COMMITTED golden is the Mac-Metal bake; verify.ps1 re-renders on the Mac + compares
   vs it at 0.0000 (same-backend determinism — the gate). The CONTROLLER measures the Windows-Vulkan vs Mac-Metal
   cross-vendor visresolve mean as a DIAGNOSTIC — a FLOAT render is in-band (~20-55, the GJ6/FR6 lineage), NOT
   strict-zero cross-vendor.
5. **Tests — APPEND to `tests/broad_test.cpp` (pure CPU):** `BroadToRenderInstances` provenance (two calls
   byte-equal); the hull/triangle counts correct; a render of the settled world vs a perturbed world differ.
   Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-broadphase-render` (features) + `--broad-render-shot` (showcases) +
   update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None — and NO new shader.** Render reuses the existing instanced-lit pipeline. `engine/sim/broad.h`
  APPEND-only (BP1-BP5 frozen); gjk.h/convex.h/etc + ALL other sim headers + ALL existing shaders UNCHANGED.
  Report the seam: NO shaders/ change, no RHI change, no frozen-file edit, broad.h append-only.

## Out of scope (YAGNI)
A general convex-hull triangulator (BP6 reuses the GJ6 canonical-hull meshes). BP6 claims ONLY: the bit-exact
broadphase-settled hull pile renders as a coherent LIT 3D scene, the render is a PURE FUNCTION of the
deterministic sim (provenance byte-equal), the Metal bake is per-backend deterministic (two runs 0.0000) +
cross-vendor in-band (float visresolve). This is the FINAL slice → flagship #23 COMPLETE.

## Verification gate
1. `ctest --preset windows-msvc-debug -R "broad|introspect"` green. Clean under `windows-msvc-asan` (separate
   build + test).
2. **proofs + visual:** `--broad-render-shot` on Vulkan: the proofs + exit 0 under the conan validation layer →
   ZERO VUID. VERIFY a coherent LIT 3D pile of TRUE matte polyhedra (no iridescence, no garbage/NaN).
3. Metal: `visual_test --broad-render` → `tests/golden/metal/broad_render.png`; two runs DIFF 0.0000. Confirm
   `visual_test.mm` in the diff; NO shader added. Cross-vendor = FLOAT visresolve in-band (~20-55).
4. **Render-invariance:** ONLY `broad_render.png` added; the other 214 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+deterministic-broadphase-render` + `--broad-render-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + BP1-BP5 broad.h code + gjk.h/convex.h/fric.h/persist.h/fpx.h/grain.h + ALL other sim
   headers + ALL existing shaders byte-unchanged; broad.h APPEND-only; NO shaders/ change). `broad_render` in the
   Mac loop + `--broad-render-shot` in `$vkShots`.
