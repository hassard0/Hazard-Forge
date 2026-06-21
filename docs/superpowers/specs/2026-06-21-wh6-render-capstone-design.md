# Slice WH6 — Warm-Started Hull Contacts: THE LIT 3D RENDER CAPSTONE (the money-shot) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice
> of FLAGSHIP #26 (WARM-STARTED HULL CONTACTS + ROBUST DETERMINISTIC STACKING, `hf::sim::warmhull`). WH1-WH5 built
> the feature ID, the cache, the warm solver, the sleeping-island STABLE STACK (a deterministic N=4 hull tower the
> frozen #25 step topples), and the lockstep/rollback netcode. WH6 is the money-shot: it renders the bit-exact
> warm+sleep-SETTLED tower as a LIT 3D scene — a **stable, resting stack of convex polyhedra** (the thing the
> single-point/non-accumulated #25 step cannot hold) drawn as true lit polyhedra under directional light. The
> render is the ONE FLOAT crossing (outside the bit-exact integer loop), so its bar is the FLOAT visresolve in-band
> metric, NOT strict-zero. PURE REUSE: the settled world is a `gjk::HullWorld`, so WH6 delegates VERBATIM to the
> FROZEN BP6/CD6/MF6 render bridge `gjk::HullToRenderInstances` (the existing instanced-lit pipeline). ZERO new
> render shader, ZERO new RHI. APPEND to `engine/sim/warmhull.h` (WH1-WH5 + manifold/gjk/persist/etc BYTE-FROZEN).
> Branch: `slice-wh6`. See [[hazard-forge-warmhull-roadmap]], [[hazard-forge-manifold-roadmap]],
> [[hazard-forge-docs-style]], [[hazard-forge-metal-showcase-gate]].

**Goal:** Extend `engine/sim/warmhull.h` (additive — WH1-WH5 byte-unchanged) with `WarmHullToRenderInstances(const
gjk::HullWorld& world)` — a one-line delegate to the frozen `gjk::HullToRenderInstances` (render-only, a PURE
FUNCTION — two calls byte-equal). Add the showcase `--wh6-render-shot <out>` (Vulkan) / `--wh6-render` (Metal) —
both build the WH4 tower scene, CPU-settle via `StepWarmSleepHullWorldN` (the tower settles into a stable asleep
stack), and draw the world LIT 3D through the EXISTING instanced-lit pipeline. Bake the float golden `wh6_render`.
**NO new shader, NO new RHI.** This is the FINAL slice → FLAGSHIP #26 COMPLETE.

## Design call: render the warm+sleep-settled stable tower; the sim stays bit-exact

The bit-exact settled `gjk::HullWorld` is WH4's deterministic Q16.16 result (via `StepWarmSleepHullWorldN`). WH6
adds ONLY a render bridge: it maps that frozen world to FLOAT geometry for display. The sim is NOT mutated; the
provenance contract (two `WarmHullToRenderInstances` calls on the same world → byte-equal output) proves the render
is a pure function of the deterministic sim.
- **`WarmHullToRenderInstances(const gjk::HullWorld& world)`** = `return gjk::HullToRenderInstances(world);` (the
  BP6/CD6/MF6 idiom — render-only, OUTSIDE the bit-exact loop). MATTE (the GJ6 material) to dodge the iridescence
  trap (metallic 0, roughness 1 — the documented GF6/FR6/MF6 lesson).
- **The scene = the warm+sleep-SETTLED stable tower.** Build the WH4 N-tower (the same N that reliably holds —
  N=4), CPU-settle via `warmhull::StepWarmSleepHullWorldN` to the converged ASLEEP rest (the stable stack of
  resting polyhedra), draw `WarmHullToRenderInstances(world)` lit. The money-shot: a deterministic stable tower of
  convex polyhedra at rest — the closed-#25-gap payoff — lit in 3D, the visual finale of the flagship.

> NOTE: WH6 settles ON THE CPU (the bit-exact `StepWarmSleepHullWorldN`) and renders the result — a normal draw,
> NOT a heavy compute dispatch, so NO TDR concern.

## Reuse map (file:line)
- **WH1-WH5 `engine/sim/warmhull.h` (APPEND after `RunWarmHullRollback`):** `StepWarmSleepHullWorldN`,
  `HullCache`, `HullSleepState`, `HullSleepConfig`, `MeasureHullSleep`. WH1-WH5 frozen.
- **gjk.h BP6/CD6/MF6 render bridge (read-only — REUSE verbatim):** `gjk::HullToRenderInstances` (the settled-hull
  → float world-space triangle mesh, matte), `gjk::HullRenderMesh`/`HullRenderMeshEqual`. The MF6 `--mf6-render`
  showcase is the direct precedent (the lit instanced 3D draw + the float visresolve-bar + the provenance proof).
  Mirror for `--wh6-render`. Do NOT modify gjk.h/etc — BYTE-FROZEN.
- **Registration:** `scripts/verify.ps1` (append `wh6_render` to the Mac loop as a plain entry like `mf6_render`
  — NO special threshold — + `--wh6-render-shot` to `$vkShots`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to `tests/warmhull_test.cpp`. (No
  new shader → nothing for `hf_gen_msl`.)

## Design decisions (locked)
1. **APPEND to `engine/sim/warmhull.h`** (WH1-WH5 byte-frozen): `WarmHullToRenderInstances` (the one-line MF6
   delegate). Render-only float, OUTSIDE the bit-exact loop. **NO new shader, NO new RHI** (reuse the instanced-lit
   pipeline).
2. **Showcase `--wh6-render-shot <out>` (Vulkan) AND `--wh6-render` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--wh6-render` BEFORE reporting DONE).** BOTH build the WH4 tower scene, CPU-settle via
   `StepWarmSleepHullWorldN` (the stable asleep tower), draw `WarmHullToRenderInstances(world)` LIT 3D (matte,
   directional light) through the EXISTING pipeline. Golden = `tests/golden/metal/wh6_render.png` (Mac-baked by the
   CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) provenance:** `wh6-render: {hulls:<H>, tris:<T>} provenance two-calls BYTE-EQUAL`.
   - **(2) the headline:** `wh6-render: {tower:<N>, asleep:<A>, stacked:true}` — the rendered world is the
     warm+sleep-settled tower (the N bodies asleep + still stacked within the rest band — the stable stack, not a
     toppled pile).
   - **(3) sim-unmutated:** `wh6-render: sim byte-unmutated by render (bodiesEqual:true)` — `gjk::HullBodiesEqual`
     of the world before vs after the render call is true (the render is a pure read).
   - **(4) determinism:** the Vulkan render path exits 0 + writes the image; Metal two-run DIFF 0.0000 (controller).
   - Golden discipline: ONLY `tests/golden/metal/wh6_render.png`; do NOT commit it. Existing 232 goldens UNTOUCHED.
4. **Cross-backend bar:** the COMMITTED golden is the Mac-Metal bake; verify.ps1 re-renders on the Mac + compares
   vs it at 0.0000 (same-backend determinism — the gate). The CONTROLLER measures the Windows-Vulkan vs Mac-Metal
   cross-vendor visresolve mean as a DIAGNOSTIC — a FLOAT render is in-band (~20-55, the MF6/CD6 lineage), NOT
   strict-zero cross-vendor.
5. **Tests — APPEND to `tests/warmhull_test.cpp` (pure CPU):** `WarmHullToRenderInstances` provenance (two calls
   byte-equal via `gjk::HullRenderMeshEqual`); the hull/triangle counts correct; the render of the settled (asleep)
   tower vs a perturbed world differ; the sim world is byte-unmutated by the render call. Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `warmhull-render` (features) + `--wh6-render-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None — and NO new shader.** Render reuses the existing instanced-lit pipeline. `engine/sim/warmhull.h`
  APPEND-only (WH1-WH5 frozen); gjk.h/manifold.h/persist.h + ALL other sim headers + ALL existing shaders
  UNCHANGED. Report the seam: NO shaders/ change, no RHI change, no frozen-file edit, warmhull.h append-only.

## Out of scope (YAGNI)
A general convex-hull triangulator (WH6 reuses the GJ6 canonical-hull meshes). WH6 claims ONLY: the bit-exact
warm+sleep-settled stable tower renders as a coherent LIT 3D scene (resting polyhedra), the render is a PURE
FUNCTION of the deterministic sim (provenance byte-equal + sim byte-unmutated), the Metal bake is per-backend
deterministic (two runs 0.0000) + cross-vendor in-band (float visresolve). The WH4 within-band-settle caveat is
inherited (a leaning-but-stable tower). This is the FINAL slice → FLAGSHIP #26 COMPLETE.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "warmhull|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--wh6-render-shot` on Vulkan: the proofs + exit 0 under the conan validation layer → ZERO
   VUID. VERIFY a coherent LIT 3D scene — a STACK OF RESTING POLYHEDRA (the settled tower) on the support under
   directional light, no iridescence, no garbage/NaN.
3. Metal: `visual_test --wh6-render` → `tests/golden/metal/wh6_render.png`; two runs DIFF 0.0000. Confirm
   `visual_test.mm` in the diff; NO shader added. Cross-vendor = FLOAT visresolve in-band (~20-55).
4. **Render-invariance:** ONLY `wh6_render.png` added; the other 232 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+warmhull-render` + `--wh6-render-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + WH1-WH5 warmhull.h code + gjk.h/manifold.h/persist.h + ALL other sim headers + ALL
   existing shaders byte-unchanged; warmhull.h APPEND-only; NO shaders/ change). `wh6_render` in the Mac loop +
   `--wh6-render-shot` in `$vkShots`. **FLAGSHIP #26 COMPLETE → consolidation #60 (full verify.ps1 + ARCHITECTURE
   warmhull section + golden/ctest count bumps).**
