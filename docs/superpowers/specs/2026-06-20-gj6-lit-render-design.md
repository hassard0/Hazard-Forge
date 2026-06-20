# Slice GJ6 — General Convex-Hull Contacts: THE LIT 3D RENDER CAPSTONE (the money-shot) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of FLAGSHIP #22
> (DETERMINISTIC GENERAL CONVEX-HULL CONTACTS via integer GJK + EPA, `hf::sim::gjk`). GJ1-GJ5 built the
> narrowphase trio, the general-hull world step, and the lockstep/rollback netcode — arbitrary convex polyhedra
> settle bit-exact CPU↔Vulkan↔Metal and are lockstep-replayable. GJ6 is the money-shot: it takes the bit-exact
> settled hull world (a `HullWorld`) and draws it as a LIT 3D scene — a **pile of mixed convex polyhedra**
> (tetrahedra, octahedra, wedges, boxes) at rest under directional light, **as true polyhedra** (not box proxies).
> The render is the ONE FLOAT crossing of the whole flagship (outside the bit-exact integer loop), so its bar is
> the FLOAT visresolve cross-vendor in-band metric, NOT strict-zero. APPEND to `engine/sim/gjk.h` (GJ1-GJ5 +
> convex.h/fric.h/persist.h/fpx.h BYTE-FROZEN). Branch: `slice-gj6`. See [[hazard-forge-gjk-roadmap]],
> [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/gjk.h` (additive — GJ1-GJ5 byte-unchanged) with per-hull FACE triangulations for the
canonical hulls + `HullToRenderMesh` / `HullToRenderInstances` (the bit-exact settled hull world → a float
world-space triangle mesh, render-only, a PURE FUNCTION of the world — two calls byte-equal). Add the showcase
`--gjk-render-shot <out>` (Vulkan) / `--gjk-render` (Metal) — both build the GJ4/GJ5 settle scene, run
`StepHullWorld`/`RunHullLockstep` to the converged world, and draw it LIT 3D through the EXISTING instanced-lit
pipeline. Bake the float golden `gjk_render`. **NO new shader, NO new RHI.**

## Design call: render the settled hulls as TRUE polyhedra; the sim stays bit-exact integer

The bit-exact settled `HullWorld` is GJ1-GJ5's deterministic Q16.16 result. GJ6 adds ONLY a render bridge: it maps
that frozen world to FLOAT geometry for display. The sim is NOT mutated; the provenance contract (two
`HullToRenderInstances` calls on the same world produce byte-equal output) proves the render is a pure function
of the deterministic sim.
- **Per-hull face triangulation (the GJ6-new geometry).** GJ1's `FxHull` is VERTS-ONLY. To draw a hull as a true
  polyhedron we need its triangles. Provide, for each canonical hull (`MakeTetra`/`MakeBox`/`MakeOcta`/
  `MakeWedge`), a FIXED triangle index list (the known face triangulation — tetra 4 tris, box 12, octa 8, wedge 8)
  with a consistent OUTWARD winding (for face normals / back-face culling). Pack as a small `HullMesh { verts;
  tris; }` or a per-builder index table. (A general convex-hull-from-points triangulation is YAGNI here — the
  scene uses the canonical hulls; document that GJ6 renders the canonical-hull meshes, a general triangulator is
  a future refinement.) The triangulation is host-side float, render-only — it does NOT touch the integer
  collision hull (`FxHull` verts) the sim uses.
- **`HullToRenderInstances(const HullWorld& world)` → a render payload** — the FC6/CX6 idiom, adapted. Because
  different hull types have different meshes (unlike the box capstones' single shared cube mesh), the cleanest
  render-only form is a **world-space triangle soup**: for each body, transform its `HullMesh` triangles by the
  body's FLOAT transform (`fpx::FxBodyTransform` → `FxToFloat`), compute per-triangle (or per-vertex) FLOAT
  normals for lighting, tag each with a per-hull-type matte color (static floor cool-grey; dynamic hulls warm
  matte amber / per-type accent), and accumulate into ONE float vertex buffer (positions + normals + colors)
  drawn as a single lit mesh — OR a small set of per-type instanced draws if that fits the existing pipeline
  more cleanly (the implementer picks the form that REUSES the existing instanced-lit pipeline with the least
  new plumbing; document the choice). **MATTE (metallic 0 / roughness 1)** to DODGE the documented GF6/FR6/JT6/
  PS6 IRIDESCENCE TRAP. A PURE FUNCTION of the bit-exact `HullWorld` (two calls byte-equal — the provenance
  contract the showcase asserts).
- **The scene = the settled pile.** Build the GJ4 settle scene (floor + tetra + octa + wedge + static box) and
  step it to the converged rest (optionally via the GJ5 command stream for a more dynamic final pose); draw
  `HullToRenderInstances(world)` lit. The money-shot: a pile of mixed convex polyhedra at rest, lit, in 3D — the
  visible payoff of the deterministic general-hull moat.

> NOTE (TDR, [[hazard-forge-gpu-tdr-chunking]]): if the showcase steps the sim ON THE GPU before rendering, apply
> the GJ4 chunked-dispatch pattern; but GJ6 can simply run the settle ON THE CPU (the bit-exact reference) and
> render the result — the render itself is a normal draw, not a heavy compute dispatch, so no TDR concern. Prefer
> the CPU-settle-then-render path (simpler, and the render is the point of this slice).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **GJ1-GJ5 `engine/sim/gjk.h` (read it; APPEND only after `RunHullRollback`, before the namespace close):**
  `FxHull`, the canonical builders, `HullWorld`, `StepHullWorld`, `RunHullLockstep`, `fpx::FxBodyTransform`/
  `FxToFloat`. GJ1-GJ5 byte-frozen.
- **The render precedent (read-only — REUSE the pipeline):** `convex::ConvexToRenderInstances` (convex.h:~1156)
  + the `--convex-render-shot` / `--fric-render` / `--persist-render` showcases (the lit instanced 3D draw, the
  matte material, the float visresolve-bar, the provenance proof). GJ6 mirrors these but feeds per-hull
  polyhedron meshes instead of the shared cube. Reuse `math::FromTRS`/`fpx::FxToFloat` (already available via
  fpx.h) — do NOT re-implement.
- **Registration:** `scripts/verify.ps1` (append `gjk_render` to the Mac golden loop EXACTLY like `fric_render`/
  `convex_render` — a plain `@{ Name = 'gjk_render'; Flag = '--gjk-render' }`, NO special threshold; the committed
  golden is the Mac bake, verify re-renders on the Mac + compares vs it at 0.0000 same-backend — + add
  `--gjk-render-shot` to `$vkShots`), `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/gjk_test.cpp`. (No new shader → nothing for `hf_gen_msl`.)

## Design decisions (locked)
1. **APPEND to `engine/sim/gjk.h`** (GJ1-GJ5 byte-frozen): the per-hull face triangulations (a `HullMesh` +
   builders, or index tables for the 4 canonical hulls) + `HullToRenderInstances` (the world→float-render-mesh
   bridge). Render-only float, OUTSIDE the bit-exact loop — the integer sim is NOT touched. **NO new shader, NO
   new RHI** (reuse the instanced-lit pipeline).
2. **Showcase `--gjk-render-shot <out>` (Vulkan) AND `--gjk-render` (Metal) — WIRE BOTH** (standalone arg-parse).
   BOTH build the settle scene, CPU-settle to the converged world, draw `HullToRenderInstances(world)` LIT 3D
   (matte; per-type colors; directional light) through the EXISTING pipeline. Golden =
   `tests/golden/metal/gjk_render.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance:** two `HullToRenderInstances(world)` calls produce BYTE-EQUAL output (the render is a pure
     function of the bit-exact sim). Print `gjk-render: {hulls:<H>, tris:<T>} provenance two-calls BYTE-EQUAL`;
     assert.
   - **(2) non-trivial scene:** the rendered world is the settled pile (assert the hull count + a non-trivial
     triangle count + e.g. the dynamic hulls rest above the floor). Print `gjk-render: {dynamic:<D>,
     restedPile:true}`.
   - **(3) determinism:** the Vulkan render path runs cleanly + writes the image (exit 0); the Metal side bakes
     two runs DIFF 0.0000 (per-backend render determinism).
   - **Golden discipline: ONLY `tests/golden/metal/gjk_render.png`; do NOT commit it.** Existing 208 image
     goldens UNTOUCHED.
4. **Cross-backend bar:** the COMMITTED golden is the Mac-Metal bake; verify.ps1 re-renders on the Mac + compares
   vs it at 0.0000 (same-backend determinism — the Metal two-run DIFF 0.0000 IS the gate, exactly like every
   other render golden). SEPARATELY, the CONTROLLER measures the Windows-Vulkan vs Mac-Metal cross-vendor
   visresolve mean as a DIAGNOSTIC — a FLOAT render is in-band (~20-55, the FR6/JT6/PS6 lineage), NOT strict-zero
   cross-vendor (expected for float; the integer-determinism headline is GJ1-GJ5's, not GJ6's).
5. **Tests — APPEND to `tests/gjk_test.cpp` (pure CPU):** `HullToRenderInstances` over the settled world → the
   provenance contract (two calls byte-equal — compare the float arrays element-by-element / a deterministic
   hash); the hull/triangle counts are correct (the canonical-hull mesh tri counts); a render of the settled
   world vs a perturbed world differ (the render reflects the sim state). Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-hull-render` (features) + `--gjk-render-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None — and NO new shader.** Render reuses the existing instanced-lit pipeline. `rhi.h` + backend dirs
  UNCHANGED. `engine/sim/gjk.h` APPEND-only (GJ1-GJ5 frozen); convex.h/fric.h/persist.h/fpx.h + ALL other sim
  headers + ALL existing shaders + `engine/physics/`/`nav/`/`anim/` UNCHANGED. Report the seam: NO shaders/
  change, no RHI change, no frozen-file edit, gjk.h append-only (only the render bridge + the
  showcase/test/introspect are new/changed).

## Out of scope (YAGNI)
A general convex-hull-from-points triangulator (GJ6 renders the canonical-hull face meshes — documented). Soft
shadows / post on the capstone (the existing instanced-lit pipeline as-is). GJ6 claims ONLY: the bit-exact
settled hull world renders as a coherent LIT 3D scene (the pile of mixed convex polyhedra), the render is a PURE
FUNCTION of the deterministic sim (provenance byte-equal), the Metal bake is per-backend deterministic (two runs
0.0000) + cross-vendor in-band (float visresolve). This is the FINAL slice → flagship #22 COMPLETE.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 108 incl. the appended GJ6 `gjk_test` cases). Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--gjk-render-shot` on Vulkan: the proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID. **VERIFY the image shows a coherent LIT 3D pile of TRUE polyhedra (a tetra/octa/wedge at rest on a
   floor under directional light) — NO iridescence (matte), NO garbage/NaN, recognizably non-box shapes.**
3. Metal: `visual_test --gjk-render` → new golden `tests/golden/metal/gjk_render.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm NO shader added.** Cross-vendor = FLOAT visresolve in-band
   (~20-55, NOT strict-zero).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `gjk_render.png` added; the other
   208 byte-identical. `git diff master --stat -- tests/golden` = ONLY `gjk_render.png` (metal) + the introspect
   json (controller rebake).
5. Introspect: exactly `+deterministic-hull-render` + `--gjk-render-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + GJ1-GJ5 gjk.h code + convex.h/fric.h/persist.h/fpx.h + ALL other sim headers + ALL
   existing shaders byte-unchanged; gjk.h APPEND-only; NO shaders/ change). `scripts/verify.ps1` updated:
   `gjk_render` in the Mac loop (plain entry) + `--gjk-render-shot` in `$vkShots`.
