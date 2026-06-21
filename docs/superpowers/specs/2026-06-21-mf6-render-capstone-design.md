# Slice MF6 — Hull Narrowphase Hardening: THE LIT 3D RENDER CAPSTONE (the money-shot) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice
> of FLAGSHIP #25 (DETERMINISTIC HULL NARROWPHASE HARDENING, `hf::sim::manifold`). MF1-MF5 built the face
> topology, the multi-point manifold, its GPU twin, the FULL-inertia hardened step (a flat-dropped hull SETTLES TO
> REST where the frozen single-point step teeters), and the lockstep/rollback netcode. MF6 is the money-shot: it
> renders the bit-exact hardened-SETTLED stack as a LIT 3D scene — a STACK OF RESTING CONVEX POLYHEDRA (the thing
> the single-point manifold can't hold) drawn as true lit polyhedra under directional light. The render is the ONE
> FLOAT crossing (outside the bit-exact integer loop), so its bar is the FLOAT visresolve in-band metric, NOT
> strict-zero. PURE REUSE: the settled world is a `gjk::HullWorld`, so MF6 delegates VERBATIM to the FROZEN
> BP6/GJ6/CD6 render bridge `gjk::HullToRenderInstances` (the existing instanced-lit pipeline). ZERO new render
> shader, ZERO new RHI. APPEND to `engine/sim/manifold.h` (MF1-MF5 + gjk/broad/ccd/convex/fpx/etc BYTE-FROZEN).
> Branch: `slice-mf6`. See [[hazard-forge-manifold-roadmap]], [[hazard-forge-gjk-roadmap]],
> [[hazard-forge-docs-style]], [[hazard-forge-metal-showcase-gate]].

**Goal:** Extend `engine/sim/manifold.h` (additive — MF1-MF5 byte-unchanged) with `HardenedHullToRenderInstances(
const gjk::HullWorld& world)` — a one-line delegate to the frozen `gjk::HullToRenderInstances` (render-only, a PURE
FUNCTION — two calls byte-equal). Add the showcase `--mf6-render-shot <out>` (Vulkan) / `--mf6-render` (Metal) —
both build a small hardened-stack scene, CPU-settle via `StepHullWorldHardenedN` (the stack settles into resting
polyhedra), and draw the world LIT 3D through the EXISTING instanced-lit pipeline. Bake the float golden
`mf6_render`. **NO new shader, NO new RHI.** This is the FINAL slice → FLAGSHIP #25 COMPLETE.

## Design call: render the hardened-settled stack; the sim stays bit-exact

The bit-exact settled `gjk::HullWorld` is MF4's deterministic Q16.16 result (via `StepHullWorldHardenedN`). MF6
adds ONLY a render bridge: it maps that frozen world to FLOAT geometry for display. The sim is NOT mutated; the
provenance contract (two `HardenedHullToRenderInstances` calls on the same world → byte-equal output) proves the
render is a pure function of the deterministic sim.
- **`HardenedHullToRenderInstances(const gjk::HullWorld& world)`** = `return gjk::HullToRenderInstances(world);`
  (the BP6/GJ6/CD6 idiom — render-only, OUTSIDE the bit-exact loop). MATTE (the GJ6 material) to dodge the
  iridescence trap (the documented GF6/FR6/CX6 lesson — keep metallic 0, roughness 1). Provided in `manifold::`
  for namespace symmetry; the actual mesh comes from the byte-unchanged GJ6 helper.
- **The scene = the hardened-SETTLED stack.** Build a small deterministic stack (a few canonical hulls dropped to
  rest on a static support — the MF4 hardened-stack scene, or a 3-4 hull mixed stack), CPU-settle via
  `manifold::StepHullWorldHardenedN` to the converged rest (the resting polyhedra), draw
  `HardenedHullToRenderInstances(world)` lit. The money-shot: a stable stack of resting polyhedra — the
  deterministic stabilization payoff — lit in 3D, the visual finale of the flagship.

> NOTE: MF6 settles ON THE CPU (the bit-exact `StepHullWorldHardenedN`) and renders the result — a normal draw,
> NOT a heavy compute dispatch, so NO TDR concern.

## Reuse map (file:line)
- **MF1-MF5 `engine/sim/manifold.h` (APPEND after `RunHullRollbackHardened`):** `StepHullWorldHardenedN`,
  `StepHullWorldHardened`. MF1-MF5 frozen.
- **gjk.h BP6/GJ6 render bridge (read-only — REUSE verbatim):** `gjk::HullToRenderInstances` (the settled-hull →
  float world-space triangle mesh, matte), `gjk::HullRenderMesh`/`HullRenderMeshEqual`. The GJ6 `--gjk-render` /
  BP6 `--broad-render` / CD6 `--ccd-render` showcases are the precedent (the lit instanced 3D draw + the float
  visresolve-bar + the provenance proof). Mirror for `--mf6-render`. Do NOT modify gjk.h/etc — BYTE-FROZEN.
- **Registration:** `scripts/verify.ps1` (append `mf6_render` to the Mac loop as a plain entry like `ccd_render` —
  NO special threshold — + `--mf6-render-shot` to `$vkShots`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to `tests/manifold_test.cpp`. (No
  new shader → nothing for `hf_gen_msl`.)

## Design decisions (locked)
1. **APPEND to `engine/sim/manifold.h`** (MF1-MF5 byte-frozen): `HardenedHullToRenderInstances` (the one-line GJ6
   delegate). Render-only float, OUTSIDE the bit-exact loop. **NO new shader, NO new RHI** (reuse the instanced-lit
   pipeline).
2. **Showcase `--mf6-render-shot <out>` (Vulkan) AND `--mf6-render` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--mf6-render` BEFORE reporting DONE).** BOTH build the hardened-stack scene, CPU-settle
   via `StepHullWorldHardenedN`, draw `HardenedHullToRenderInstances(world)` LIT 3D (matte, directional light)
   through the EXISTING pipeline. Golden = `tests/golden/metal/mf6_render.png` (Mac-baked by the CONTROLLER — DO
   NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) provenance:** `mf6-render: {hulls:<H>, tris:<T>} provenance two-calls BYTE-EQUAL`.
   - **(2) the headline:** `mf6-render: {stackSettled:true, maxAngVel:<v>}` — the rendered stack is the settled
     hardened world (`maxAngVel` below a fixed band — the resting polyhedra, not a still-tumbling pile).
   - **(3) sim-unmutated:** `mf6-render: sim byte-unmutated by render (bodiesEqual:true)` — `HullBodiesEqual` of the
     world before vs after the render call is true (the render is a pure read).
   - **(4) determinism:** the Vulkan render path exits 0 + writes the image; Metal two-run DIFF 0.0000 (controller).
   - Golden discipline: ONLY `tests/golden/metal/mf6_render.png`; do NOT commit it. Existing 226 goldens UNTOUCHED.
4. **Cross-backend bar:** the COMMITTED golden is the Mac-Metal bake; verify.ps1 re-renders on the Mac + compares
   vs it at 0.0000 (same-backend determinism — the gate). The CONTROLLER measures the Windows-Vulkan vs Mac-Metal
   cross-vendor visresolve mean as a DIAGNOSTIC — a FLOAT render is in-band (~20-55, the GJ6/BP6/CD6 lineage), NOT
   strict-zero cross-vendor.
5. **Tests — APPEND to `tests/manifold_test.cpp` (pure CPU):** `HardenedHullToRenderInstances` provenance (two
   calls byte-equal via `gjk::HullRenderMeshEqual`); the hull/triangle counts correct; the render of the settled
   (resting) world vs a perturbed world differ; the sim world is byte-unmutated by the render call. Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `manifold-render` (features) + `--mf6-render-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None — and NO new shader.** Render reuses the existing instanced-lit pipeline. `engine/sim/manifold.h`
  APPEND-only (MF1-MF5 frozen); gjk.h/broad.h/etc + ALL other sim headers + ALL existing shaders UNCHANGED. Report
  the seam: NO shaders/ change, no RHI change, no frozen-file edit, manifold.h append-only.

## Out of scope (YAGNI)
A general convex-hull triangulator (MF6 reuses the GJ6 canonical-hull meshes). MF6 claims ONLY: the bit-exact
hardened-settled stack renders as a coherent LIT 3D scene (resting polyhedra), the render is a PURE FUNCTION of the
deterministic sim (provenance byte-equal + sim byte-unmutated), the Metal bake is per-backend deterministic (two
runs 0.0000) + cross-vendor in-band (float visresolve). This is the FINAL slice → FLAGSHIP #25 COMPLETE.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "manifold|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--mf6-render-shot` on Vulkan: the proofs + exit 0 under the conan validation layer → ZERO
   VUID. VERIFY a coherent LIT 3D scene — a STACK OF RESTING POLYHEDRA on the support under directional light, no
   iridescence, no garbage/NaN.
3. Metal: `visual_test --mf6-render` → `tests/golden/metal/mf6_render.png`; two runs DIFF 0.0000. Confirm
   `visual_test.mm` in the diff; NO shader added. Cross-vendor = FLOAT visresolve in-band (~20-55).
4. **Render-invariance:** ONLY `mf6_render.png` added; the other 226 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+manifold-render` + `--mf6-render-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + MF1-MF5 manifold.h code + gjk.h/broad.h/convex.h/fpx.h/etc + ALL other sim headers +
   ALL existing shaders byte-unchanged; manifold.h APPEND-only; NO shaders/ change). `mf6_render` in the Mac loop +
   `--mf6-render-shot` in `$vkShots`. **FLAGSHIP #25 COMPLETE → consolidation #59 (full verify.ps1 + ARCHITECTURE
   manifold section + golden/ctest count bumps).**
