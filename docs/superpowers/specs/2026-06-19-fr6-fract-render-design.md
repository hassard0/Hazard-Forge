# Slice FR6 — Deterministic Fracture/Destruction: LIT 3D RENDER CAPSTONE (the money-shot) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of FLAGSHIP #14
> (DETERMINISTIC RIGID-BODY FRACTURE / DESTRUCTION, `hf::sim::fract`) — **COMPLETES THE FLAGSHIP (the 14th)**.
> The money-shot: render the bit-exact settled rubble as a LIT 3D scene — the shattered object's chunks scattered
> as lit stone spheres around the held anchor. The FR1-FR5 sim stays strict-integer/bit-exact; here — and ONLY
> here — we cross to FLOAT to build the per-body render transforms (the documented FLOAT visresolve-bar, the
> FPX6/CG6/GF6 precedent). NO new shader, NO new RHI — reuse the EXISTING instanced-lit pipeline VERBATIM. The
> DIRECT TWIN of `fpx::FxBodyTransform`-based render (FPX6) over the fracture world. Branch: `slice-fr6`. See
> [[hazard-forge-fract-roadmap]].

**Goal:** Extend `engine/sim/fract.h` (additive — FR1-FR5 byte-unchanged) with `FractToRenderInstances(world,
…)` — per-body model matrices built DIRECTLY from the bit-exact settled `fpx::FxWorld` via `fpx::FxBodyTransform`
(REUSED VERBATIM), grouped so the showcase can colour the anchor vs the dislodged chunks. Add `--fract-render-shot`
(Vulkan) / `--fract-render` (Metal) — a lit 3D render through the EXISTING instanced-lit pipeline. Bake the FLOAT
golden `fract_render`. **NO new shader, NO new RHI.**

## Design call: the FPX6 render bridge over the fracture world, FLOAT visresolve-bar
FR6 is the FPX6/CG6/GF6 capstone: build the per-instance model matrices DIRECTLY from the settled `fpx::FxWorld`
(the FR4/FR5 output) and draw them through the SAME instanced-lit-sphere pipeline FPX6/GR6/CG6 use. The ONLY
float crossing of the whole flagship is the transform build (`fpx::FxBodyTransform`: `translate(pos/kOne) ·
quatToMat(normalize(orient/kOne)) · scale(radius/kOne)` — fpx.h:627, REUSED VERBATIM). The FR1-FR5 sim above is
untouched and strict-integer. Bar: the **FLOAT visresolve-bar** (NOT the integer zero-diff bar) — Metal two-run
determinism DIFF 0.0000 + provenance (every transform derives from the bit-exact body state) + visual parity +
cross-vendor mean ~46-55 (the engine float-render baseline: FPX6=27, CL6=29, NAV6=45, FL6=40, GR6=46, CP6=51,
GF6=46). Pure deterministic host float (no RNG, no clock).

## The render bridge + the anchor/chunk colour distinction
`FractToRenderInstances(world, …)` returns one `math::Mat4` per body via `fpx::FxBodyTransform(b)`, in body
index order (the FR4 spawn order: the bodies carry their dynamic/static flag). The showcase splits the draw by
the body's `kFlagDynamic` flag (or a parallel piece/anchor id array the bridge fills) so it can colour:
- the **anchor** (static) fragments a held-stone colour, and
- the **dislodged** (dynamic) chunks a slightly warmer/brighter rubble colour (or per-piece colours),
as TWO (or more) instanced draws through the SAME existing instanced-lit pipeline (the per-draw material/albedo
mechanism the GR6/CG6/GF6 showcases use — **NO new shader, NO new per-instance attribute, NO new RHI**). **THE
GF6 LESSON (heed it):** the lit pipeline has **no per-draw albedo uniform** — distinguish materials via per-draw
solid albedo **textures** (bound through the existing `CreateTexture`/`BindMaterial`), and use **roughness 1.0
(matte)** for the stone so it does NOT mirror the sky IBL into an iridescent sheen. Use a warm/neutral STONE
palette (the rubble reads as broken rock), not the GF6 sand/cyan. The result is a coherent lit 3D destroyed
object — a held base with scattered fallen chunks.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FPX6 render bridge to REUSE VERBATIM (`engine/sim/fpx.h:619-650`):** `FxToFloat` (:620), `FxBodyTransform`
  (:627 — the per-body model matrix from the bit-exact integer state, the ONE float crossing). FR6's
  `FractToRenderInstances` calls `FxBodyTransform` per body. DO NOT modify fpx.h.
- **The CG6/GF6 lit render showcase to COPY (the render path + the GF6 colour lesson):** the `--cgrain-render` /
  `--cgf-render` showcases in `samples/hello_triangle/main.cpp` (Vulkan) + `metal_headless/visual_test.mm`
  (Metal) — the EXISTING instanced-lit pipeline setup, the camera, the lighting, the per-draw solid-albedo-
  TEXTURE material trick (warm matte for sand / cyan for fluid), and the matte (roughness 1.0) discipline that
  killed GF6's iridescence. FR6 copies this path and issues the anchor draw + the chunk draw with STONE albedo
  textures. **Reuse the lit pipeline VERBATIM — NO new shader/RHI.**
- **The FR4/FR5 fracture world (this branch's `fract.h`, read-only):** `SpawnFractWorld`/`StepFractureSteps` (or
  `RunFractLockstep`) to produce the settled `fpx::FxWorld`; `FractStepConfig`; `fpx::FxWorld`/`FxBody`/
  `kFlagDynamic`. **DO NOT modify fpx.h or FR1-FR5 code** — FR6 is additive.
- **Showcase + registration:** FR1-FR5's `--fract-*-shot` plumbing — **standalone arg-parse loop** (the FR1 C1061
  lesson). `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the
  introspect JSON golden**), `tests/fract_test.cpp`.

## Design decisions (locked)
1. **`FractToRenderInstances(world, …)` — per-body `FxBodyTransform` matrices** (body order; the caller splits by
   `kFlagDynamic` for the anchor/chunk colouring, or the bridge fills a parallel `isDynamic[]`/`pieceId[]`).
   Empty world → empty output (the no-op). Pure deterministic host float, render-only, NO sim mutation. The FPX6/
   CG6 twin.
2. **Showcase `--fract-render-shot <out>` (Vulkan) AND `--fract-render` (Metal) — WIRE BOTH** (standalone
   arg-parse). Settle the FR4 fracture scene (`SpawnFractWorld` → K `StepFracture` ticks — reuse the FR4 scene),
   then render: the anchor (held stone) + the dislodged chunks (warmer rubble), two instanced draws through the
   existing lit pipeline, a lit 3D side view of the destroyed object. Golden = `tests/golden/metal/fract_render.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) instance provenance / count:** the instance count == `world.bodies.size()`, every transform derived
     from the settled `FxWorld`. Print `fract-render: {bodies:<N>, anchor:<A>, dynamic:<D>, instances:<N>} from
     bit-exact rubble`.
   - **(2) determinism:** two renders → byte-identical (Metal). Print `fract-render determinism: two runs
     BYTE-IDENTICAL`.
   - **(3) provenance check:** rebuilding instances from the same settled world → identical matrices (the float
     transform is a pure function of the integer body state). Print `fract-render provenance: instances ==
     rebuild`.
   - **Golden discipline: ONLY `tests/golden/metal/fract_render.png`; do NOT commit it.** Existing 158 image
     goldens UNTOUCHED.
4. **Cross-backend bar (FLOAT visresolve-bar, NOT integer):** Metal two-run DIFF 0.0000 (gate on `compare.sh`) +
   provenance + visual parity + cross-vendor mean ~46-55 (the engine float-render baseline — document the actual
   mean; NOT held to the integer zero-diff bar — the FPX6/GR6/CP6/GF6 precedent).
5. **Tests `tests/fract_test.cpp` additions (pure CPU):** `FractToRenderInstances` — instance count ==
   bodies.size(); a known body's matrix == `fpx::FxBodyTransform(b)` (spot-check translation == `pos/kOne`);
   empty world → empty; the anchor/dynamic split count matches the world's static/dynamic bodies. Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-fract-render` (features) + `--fract-render-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`.

## RHI seam additions (summary)
- **None.** Reuse the EXISTING instanced-lit pipeline + dispatch + the offscreen render path (the FPX6/GR6/CG6/GF6
  surface). `rhi.h` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple.h`
  + `couple_grain.h` + `couple_gf.h` + `engine/physics/` + all existing shaders UNCHANGED. FR1-FR5 `fract.h` code
  + shaders UNCHANGED (FR6 additive — only the render bridge + the showcase). **NO new shader.** Report the seam
  empty.

## Out of scope (YAGNI)
Per-instance colour attributes / a new material (reuse the GR6/GF6 per-draw albedo texture). Fragment surface
geometry / real shard meshes (FR6 renders the bounding-sphere bodies as spheres — the FR4 sphere-bound
simplification carried through; a real shard mesh is a far-future refinement). Motion blur / debris trails /
dust. FR6 claims ONLY: a deterministic lit 3D render of the bit-exact settled rubble, Metal-deterministic +
provenance + cross-vendor visual parity, with the float golden + the three proofs. **FR6 COMPLETES FLAGSHIP #14
— the FOURTEENTH flagship.**

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 100) + the new `fract_test` render cases. Clean under
   `windows-msvc-asan` (build+run `fract_test` + `introspect_test`).
2. **proofs + visual:** `--fract-render-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image is a
   coherent lit 3D destroyed-object scene — stone chunks scattered around the held anchor, lit/shaded, matte (NOT
   iridescent) (pixel-check; the GF6 lesson).**
3. Metal: `visual_test --fract-render` → new golden `tests/golden/metal/fract_render.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (FR6 reuses the
   lit pipeline — `hf_gen_msl` UNCHANGED).** Cross-vendor = the FLOAT visresolve-bar (document the mean ~46-55; NOT
   integer zero).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fract_render.png` added; the
   other 158 byte-identical. `git diff master --stat -- tests/golden` = ONLY `fract_render.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fract-render` + `--fract-render-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h` +
   `engine/physics/` + FR1-FR5 `fract.h`/shaders byte-unchanged). `scripts/verify.ps1` updated: `fract_render`
   golden in the Mac loop + `--fract-render-shot` in `$vkShots`. **NO new entry in `hf_gen_msl`.**
