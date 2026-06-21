# Slice CD6 — Deterministic Integer CCD: THE LIT 3D RENDER CAPSTONE (the money-shot) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice
> of FLAGSHIP #24 (DETERMINISTIC INTEGER CCD, `hf::sim::ccd`). CD1-CD5 built the TOI, the swept broadphase, the
> substepped CCD step, the bullet-wall beat, and the lockstep netcode. CD6 is the money-shot: it renders the
> bit-exact CCD-settled bullet-wall world as a LIT 3D scene — the **projectile arrested at the thin wall** (the
> thing a discrete solver would tunnel through), drawn as true lit polyhedra under directional light. The render
> is the ONE FLOAT crossing (outside the bit-exact integer loop), so its bar is the FLOAT visresolve in-band
> metric, NOT strict-zero. PURE REUSE: the settled world is a `gjk::HullWorld`, so CD6 delegates VERBATIM to the
> FROZEN BP6/GJ6 render bridge `gjk::HullToRenderInstances` (the existing instanced-lit pipeline). ZERO new render
> shader, ZERO new RHI. APPEND to `engine/sim/ccd.h` (CD1-CD5 + broad.h/gjk.h/convex.h/fpx.h/fric.h/persist.h/
> grain.h BYTE-FROZEN). Branch: `slice-cd6`. See [[hazard-forge-ccd-roadmap]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/ccd.h` (additive — CD1-CD5 byte-unchanged) with `CcdToRenderInstances(const
gjk::HullWorld& world)` — a one-line delegate to the frozen `gjk::HullToRenderInstances` (render-only, a PURE
FUNCTION — two calls byte-equal). Add the showcase `--ccd-render-shot <out>` (Vulkan) / `--ccd-render` (Metal) —
both build the CD4 bullet-wall scene, CPU-settle via `StepHullWorldCCDN` (the projectile arrests at the wall),
and draw the world LIT 3D through the EXISTING instanced-lit pipeline. Bake the float golden `ccd_render`.

## Design call: render the CCD-settled bullet-wall; the sim stays bit-exact

The bit-exact settled `gjk::HullWorld` is CD3/CD4's deterministic Q16.16 result (via the CCD step). CD6 adds ONLY
a render bridge: it maps that frozen world to FLOAT geometry for display. The sim is NOT mutated; the provenance
contract (two `CcdToRenderInstances` calls on the same world → byte-equal output) proves the render is a pure
function of the deterministic sim.
- **`CcdToRenderInstances(const gjk::HullWorld& world)`** = `return gjk::HullToRenderInstances(world);` (the
  BP6/GJ6 idiom — render-only, OUTSIDE the bit-exact loop). MATTE (the GJ6 material) to dodge the iridescence
  trap. Provided in `ccd::` for namespace symmetry; the actual mesh comes from the byte-unchanged GJ6 helper.
- **The scene = the CCD-settled bullet-wall.** Build the CD4 `MakeBulletWallScene` (a thin static wall + a fast
  projectile + slow hulls), CPU-settle via `ccd::StepHullWorldCCDN` to the converged rest (the projectile
  arrested at the wall's near face), draw `CcdToRenderInstances(world)` lit. The money-shot: the bullet stopped
  at the wall — the deterministic continuous-contact payoff — lit in 3D.

> NOTE: CD6 settles ON THE CPU (the bit-exact `StepHullWorldCCDN`) and renders the result — a normal draw, NOT a
> heavy compute dispatch, so NO TDR concern.

## Reuse map (file:line)
- **CD1-CD5 `engine/sim/ccd.h` (APPEND after CD5's `RunCcdRollback`):** `gjk::HullWorld`, `StepHullWorldCCDN`,
  `CcdStepConfig`, `MakeBulletWallScene`, `MeasureBullet`. CD1-CD5 frozen.
- **gjk.h BP6/GJ6 render bridge (read-only — REUSE verbatim):** `gjk::HullToRenderInstances` (the settled-hull →
  float world-space triangle mesh, matte), `gjk::HullRenderMesh`/`HullRenderMeshEqual`. The BP6 `--broad-render`
  / GJ6 `--gjk-render` showcases are the precedent (the lit instanced 3D draw + the float visresolve-bar + the
  provenance proof). Mirror for `--ccd-render`. Do NOT modify gjk.h/broad.h/etc — BYTE-FROZEN.
- **Registration:** `scripts/verify.ps1` (append `ccd_render` to the Mac loop as a plain entry like `gjk_render`
  — NO special threshold — + `--ccd-render-shot` to `$vkShots`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to `tests/ccd_test.cpp`. (No new
  shader → nothing for `hf_gen_msl`.)

## Design decisions (locked)
1. **APPEND to `engine/sim/ccd.h`** (CD1-CD5 byte-frozen): `CcdToRenderInstances` (the one-line GJ6 delegate).
   Render-only float, OUTSIDE the bit-exact loop. **NO new shader, NO new RHI** (reuse the instanced-lit pipeline).
2. **Showcase `--ccd-render-shot <out>` (Vulkan) AND `--ccd-render` (Metal) — WIRE BOTH (grep your own
   visual_test.mm for `--ccd-render` before reporting DONE).** BOTH build the CD4 bullet-wall scene, CPU-settle
   via `StepHullWorldCCDN`, draw `CcdToRenderInstances(world)` LIT 3D (matte, directional light) through the
   EXISTING pipeline. Golden = `tests/golden/metal/ccd_render.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly):**
   - (1) provenance: `ccd-render: {hulls:<H>, tris:<T>} provenance two-calls BYTE-EQUAL`.
   - (2) the headline: `ccd-render: {projectile:arrested, tunneled:false}` (the settled projectile is on the
     approach side of the wall, not tunneled).
   - (3) determinism: the Vulkan render path exits 0 + writes the image; Metal two-run DIFF 0.0000 (controller).
   - Golden discipline: ONLY `tests/golden/metal/ccd_render.png`; do NOT commit it. Existing 220 goldens UNTOUCHED.
4. **Cross-backend bar:** the COMMITTED golden is the Mac-Metal bake; verify.ps1 re-renders on the Mac + compares
   vs it at 0.0000 (same-backend determinism — the gate). The CONTROLLER measures the Windows-Vulkan vs Mac-Metal
   cross-vendor visresolve mean as a DIAGNOSTIC — a FLOAT render is in-band (~20-55, the GJ6/BP6 lineage), NOT
   strict-zero cross-vendor.
5. **Tests — APPEND to `tests/ccd_test.cpp` (pure CPU):** `CcdToRenderInstances` provenance (two calls
   byte-equal); the hull/triangle counts correct; a render of the settled (arrested) world vs a perturbed world
   differ. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-ccd-render` (features) + `--ccd-render-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None — and NO new shader.** Render reuses the existing instanced-lit pipeline. `engine/sim/ccd.h` APPEND-only
  (CD1-CD5 frozen); gjk.h/broad.h/etc + ALL other sim headers + ALL existing shaders UNCHANGED. Report the seam:
  NO shaders/ change, no RHI change, no frozen-file edit, ccd.h append-only.

## Out of scope (YAGNI)
A general convex-hull triangulator (CD6 reuses the GJ6 canonical-hull meshes). CD6 claims ONLY: the bit-exact
CCD-settled bullet-wall world renders as a coherent LIT 3D scene (the projectile arrested at the wall), the
render is a PURE FUNCTION of the deterministic sim (provenance byte-equal), the Metal bake is per-backend
deterministic (two runs 0.0000) + cross-vendor in-band (float visresolve). This is the FINAL slice → flagship #24
COMPLETE.

## Verification gate
1. `ctest --preset windows-msvc-debug -R "ccd|introspect"` green. Clean under `windows-msvc-asan` (separate build
   + test).
2. **proofs + visual:** `--ccd-render-shot` on Vulkan: the proofs + exit 0 under the conan validation layer →
   ZERO VUID. VERIFY a coherent LIT 3D scene — the matte projectile arrested at the thin wall under directional
   light, no iridescence, no garbage/NaN.
3. Metal: `visual_test --ccd-render` → `tests/golden/metal/ccd_render.png`; two runs DIFF 0.0000. Confirm
   `visual_test.mm` in the diff; NO shader added. Cross-vendor = FLOAT visresolve in-band (~20-55).
4. **Render-invariance:** ONLY `ccd_render.png` added; the other 220 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+deterministic-ccd-render` + `--ccd-render-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + CD1-CD5 ccd.h code + gjk.h/broad.h/convex.h/fpx.h/etc + ALL other sim headers + ALL
   existing shaders byte-unchanged; ccd.h APPEND-only; NO shaders/ change). `ccd_render` in the Mac loop +
   `--ccd-render-shot` in `$vkShots`.
