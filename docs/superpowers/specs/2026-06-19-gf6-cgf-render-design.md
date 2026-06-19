# Slice GF6 — Deterministic Grain↔Fluid Coupling: LIT 3D RENDER CAPSTONE (the money-shot) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of FLAGSHIP #13
> (DETERMINISTIC TWO-WAY GRAIN↔FLUID COUPLING, `hf::sim::cgf`) — **COMPLETES THE FLAGSHIP**. The money-shot:
> render the bit-exact coupled state as a LIT 3D scene — a sand bed of small lit spheres with cyan fluid droplets
> pooled around/through it (WET SAND / MUD). The GF1-GF5 sim stays strict-integer/bit-exact; here — and ONLY here
> — we cross to FLOAT to build the per-instance render transforms (the documented FLOAT visresolve-bar, the
> FPX6/CL6/NAV6/FL6/GR6/CP6/CG6 precedent). NO new shader, NO new RHI — reuse the EXISTING instanced-lit pipeline
> VERBATIM. The DIRECT TWIN of `couple_grain.h::CGrainToRenderInstances` (CG6) over the grain+fluid sim. Branch:
> `slice-gf6`. See [[hazard-forge-couple-gf-roadmap]].

**Goal:** Extend `engine/sim/couple_gf.h` (additive — GF1-GF5 byte-unchanged) with `CGFToRenderInstances(world,
grainRadius, fluidRadius)` — a COMBINED instance set built DIRECTLY from the bit-exact `CGFWorld` state (grains
via `grain::GrainToRenderInstances` + fluid via `fluid::FluidToRenderInstances`). Add `--cgf-render-shot`
(Vulkan) / `--cgf-render` (Metal) — a lit 3D render through the EXISTING instanced-lit pipeline. Bake the FLOAT
golden `cgf_render`. **NO new shader, NO new RHI.**

## Design call: the CG6 render bridge over two particle pools, FLOAT visresolve-bar
GF6 is the GF-arc's `CGrainToRenderInstances` (CG6): build the per-instance model matrices DIRECTLY from the
settled `CGFWorld` (the GF4/GF5 output), one small sphere per grain + one small sphere (droplet) per fluid
particle, and draw them through the SAME instanced-lit pipeline the GR6/FL6/CG6 showcases use. The ONLY float
crossing of the whole flagship is the transform build (`pos/kOne` → float translate · scale(radius)); the
GF1-GF5 sim above is untouched and strict-integer. Bar: the **FLOAT visresolve-bar** (NOT the integer zero-diff
bar) — Metal two-run determinism DIFF 0.0000 + provenance (every transform derives from the bit-exact settled
state) + visual parity + cross-vendor mean ~30-60 (the engine float-render baseline: FPX6=27, CL6=29, NAV6=45,
FL6=40, GR6=46, CP6=51). Pure deterministic host float (no RNG, no clock).

## The render bridge + the two-pool color distinction
`CGFToRenderInstances(world, grainRadius, fluidRadius)` returns the grain instance matrices FOLLOWED by the fluid
instance matrices (grains first, like CG6 puts bodies first), so the caller can split the combined set into the
two draws:
```
out.reserve(grains.size() + fluid.size());
auto g = grain::GrainToRenderInstances(world.grains, grainRadius);   // GR6 bridge VERBATIM (small sand spheres)
auto f = fluid::FluidToRenderInstances(world.fluid,  fluidRadius);   // FL6 bridge VERBATIM (small droplet spheres)
out.insert(out.end(), g.begin(), g.end());
out.insert(out.end(), f.begin(), f.end());
```
**Distinguish sand from fluid by COLOR**, exactly the way the GR6 (sand-warm) + FL6 (cyan) showcases color their
instanced draws: render the grain instances with the GR6 sand material color and the fluid instances with the
FL6 cyan/water material color, as TWO instanced draws through the SAME existing instanced-lit pipeline (the
pipeline's per-draw material/color uniform — NO new shader, NO per-instance color attribute, NO new RHI). If the
GR6/FL6 showcases set color via a per-draw uniform, GF6 does the same twice (sand draw, then fluid draw). The
result reads as WET SAND: warm sand grains with cyan fluid pooled around/through them, lit in 3D.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CG6 render bridge to MIRROR (`engine/sim/couple_grain.h:739-749`):** `CGrainToRenderInstances` (combined
  instance set, bodies-then-grains). GF6's `CGFToRenderInstances` is the SAME shape with grains-then-fluid, both
  small spheres.
- **The two render bridges to REUSE VERBATIM:** `grain::GrainToRenderInstances` (`grain.h`, the GR6 bridge —
  translate(pos/kOne)·scale(grainRadius)) + `fluid::FluidToRenderInstances` (`fluid.h:985`, the FL6 bridge). Read
  both for the exact transform form.
- **The GR6 + FL6 + CG6 lit showcases to COPY (the render path + coloring):** the `--gr6/--grain-render`,
  `--fl6/--fluid-render`, `--cgrain-render` showcases in `samples/hello_triangle/main.cpp` (Vulkan) +
  `metal_headless/visual_test.mm` (Metal) — the EXISTING instanced-lit pipeline setup, the camera, the lighting,
  and HOW each sets its per-draw material color. GF6 wires `--cgf-render-shot` / `--cgf-render` by copying this
  path and issuing the sand draw + the cyan fluid draw. **Reuse the lit pipeline VERBATIM — NO new shader/RHI.**
- **The GF1-GF5 world (this branch's `couple_gf.h`, read-only):** `CGFWorld`, `StepCGFSteps` (settle the scene
  before rendering), `MeasureCGFState`. `grain::GrainParticle`/`fluid::FluidParticle`. **DO NOT modify
  grain.h/fluid.h/fpx.h/cloth.h/couple.h/couple_grain.h/rhi.h/engine/physics or GF1-GF5 code** — GF6 is additive.
- **Showcase + registration:** GF1-GF5's `--cgf-*-shot` plumbing; `scripts/verify.ps1`, `engine/editor/
  introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden**), `tests/cgf_test.cpp`.

## Design decisions (locked)
1. **`CGFToRenderInstances(world, grainRadius, fluidRadius)` — grains-then-fluid combined matrices.** Reuses the
   GR6 + FL6 bridges VERBATIM. Empty world → empty output (the no-op). Pure deterministic host float, render-only,
   NO sim mutation. The CG6 twin.
2. **Showcase `--cgf-render-shot <out>` (Vulkan) AND `--cgf-render` (Metal) — WIRE BOTH.** Settle the GF4
   wet-sand scene (a `StepCGFSteps` run — reuse the GF4 scene constants), then render: the sand grains (GR6 warm)
   + the fluid droplets (FL6 cyan) as two instanced draws through the existing lit pipeline, lit 3D side view.
   Golden = `tests/golden/metal/cgf_render.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) instance provenance / count:** the instance count == `grains.size() + fluid.size()`, every transform
     derived from the settled `CGFWorld`. Print `cgf-render: {grains:<G>, fluid:<F>, instances:<N>} from bit-exact
     coupled state`.
   - **(2) determinism:** two renders → byte-identical (Metal). Print `cgf-render determinism: two runs
     BYTE-IDENTICAL`.
   - **(3) provenance check:** rebuilding instances from the same settled state → identical matrices (the float
     transform is a pure function of the integer state). Print `cgf-render provenance: instances == rebuild`.
   - **Golden discipline: ONLY `tests/golden/metal/cgf_render.png`; do NOT commit it.** Existing 152 image
     goldens UNTOUCHED.
4. **Cross-backend bar (FLOAT visresolve-bar, NOT integer):** Metal two-run DIFF 0.0000 (gate on `compare.sh`)
   + provenance + visual parity + cross-vendor mean ~30-60 (the engine float-render baseline — document the
   actual mean; it is NOT held to the integer zero-diff bar — the FPX6/GR6/CP6/CG6 precedent).
5. **Tests `tests/cgf_test.cpp` additions (pure CPU):** `CGFToRenderInstances` — instance count == grains+fluid;
   the first G matrices are the GR6 grain transforms, the next F are the FL6 fluid transforms (spot-check a known
   grain/fluid pos → its translate·scale matrix); empty world → empty. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-cgf-render` (features) + `--cgf-render-shot` (showcases). **REBAKE
   `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (`git diff master --
   tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the EXISTING instanced-lit pipeline + dispatch + the offscreen render path (the GR6/FL6/CG6
  surface). `rhi.h` + backend dirs UNCHANGED. `engine/sim/grain.h` + `fluid.h` + `fpx.h` + `cloth.h` + `couple.h`
  + `couple_grain.h` + `engine/physics/` UNCHANGED. GF1-GF5 cgf code + shaders UNCHANGED (GF6 additive — only the
  render bridge + the showcase). **NO new shader.** Report the seam empty.

## Out of scope (YAGNI)
Per-instance color attributes / a new material (reuse the GR6/FL6 per-draw color), fluid surface reconstruction /
metaballs / screen-space fluid (GF6 renders droplet spheres, the FL6 precedent), refraction/caustics. GF6 claims
ONLY: a deterministic lit 3D render of the bit-exact coupled grain+fluid state, Metal-deterministic + provenance
+ cross-vendor visual parity, with the float golden + the three proofs. **GF6 COMPLETES FLAGSHIP #13.**

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 99) + the new `cgf_test` render cases. Clean under
   `windows-msvc-asan` (build+run `cgf_test` + `introspect_test`).
2. **proofs + visual:** `--cgf-render-shot` on Vulkan: the 3 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image is a
   coherent lit 3D wet-sand scene — warm sand grains + cyan fluid droplets, lit/shaded (pixel-check; the
   GR6/CG6 lesson).**
3. Metal: `visual_test --cgf-render` → new golden `tests/golden/metal/cgf_render.png`; two runs DIFF 0.0000 (gate
   on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (GF6 reuses the lit
   pipeline — `hf_gen_msl` UNCHANGED).** Cross-vendor = the FLOAT visresolve-bar (document the mean ~30-60; NOT
   integer zero).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgf_render.png` added; the other
   152 byte-identical. `git diff master --stat -- tests/golden` = ONLY `cgf_render.png` (metal) + the introspect
   json.
5. Introspect JSON rebaked exactly `+deterministic-cgf-render` + `--cgf-render-shot`; introspect test updated.
   (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/grain.h` + `fluid.h` + `fpx.h` + `cloth.h` + `couple.h` +
   `couple_grain.h` + `engine/physics/` + GF1-GF5 cgf code/shaders byte-unchanged). `scripts/verify.ps1` updated:
   `cgf_render` golden in the Mac loop + `--cgf-render-shot` in `$vkShots`. **NO new entry in `hf_gen_msl`.**
