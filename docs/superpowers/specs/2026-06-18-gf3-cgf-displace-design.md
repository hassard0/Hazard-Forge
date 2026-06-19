# Slice GF3 — Deterministic Grain↔Fluid Coupling: CONTACT REACTION / DISPLACEMENT (grain→fluid) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #13 (DETERMINISTIC
> TWO-WAY GRAIN↔FLUID COUPLING, `hf::sim::cgf`). The Newton's-3rd-law HALF of GF2: the grains now push BACK on
> the fluid — each fluid particle inside a grain's exclusion radius is projected out to the grain surface (the
> sand DISPLACES the fluid — the fluid sits ON / seeps AROUND the bed, not inside the grain volumes) and
> receives the equal-opposite drag impulse. Completes the two-way exchange (GF2 fluid→grain, GF3 grain→fluid).
> JACOBI, per-fluid-particle, `[numthreads(64,1,1)]` → NO TDR. The CG3/CP3 twin — the projection is the
> `grain.h` `CollideGrainSphere` sphere-projection with each GRAIN as the sphere. Branch: `slice-gf3`. See
> [[hazard-forge-couple-gf-roadmap]].

**Goal:** Extend `engine/sim/couple_gf.h` (additive — GF1/GF2 byte-unchanged) with `ApplyGrainsToFluid(world,
neighbors)` (the per-fluid-particle projection-out-of-grains + drag reaction, the CG3/GR3 mold over the GF1
`fgNeighbors` grain list) + a `MeasureFluidGrainPenetration` helper. Add `shaders/cgf_displace.comp.hlsl`
(int64 → **Vulkan-only** + Metal CPU reference). Add `--cgf-displace-shot` (Vulkan) / `--cgf-displace` (Metal).
Bake the integer golden `cgf_displace`. NO new RHI.

## Design call: per-FLUID-particle Jacobi over the GF1 grain neighbours (race-free) + the INTEGER bar
GF2's buoyancy was per-GRAIN (over its fluid neighbours). GF3's displacement is the mirror: **per-FLUID-particle
over its grain neighbours** — one thread per FLUID particle, each fluid particle iterates its GF1 `fgNeighbors`
grain list (fixed order), and for each grain that contains it, accumulates the positional push (snap to the
grain surface) into a SEPARATE `dp[]` (Jacobi) + applies the drag-reaction velocity impulse. Each fluid
particle writes ONLY its own `dp`/`vel` → per-fluid-disjoint, race-free, NO atomics, `[numthreads(64,1,1)]`
multi-thread, NO TDR (the GR3/CG3 win; the EXACT shape of `grain.h`'s `CollideGrainSpheres`). int64
(`FxLength`/`FxNormalize`/`fxmul`) → `cgf_displace.comp` Vulkan-only + the Metal showcase runs the CPU
`ApplyGrainsToFluid` (byte-identical by construction, the GR3/CG3 split). Bar: strict INTEGER (Vulkan GPU ==
Metal CPU-ref == golden, ZERO differing pixels). The GF1 cross-query passes stay int32 MSL-native; the GF2
`cgf_buoyancy` is unchanged.

## The displacement + drag-reaction model (Newton's 3rd law to GF2 — reuse the grain.h sphere projection)
For each fluid particle `p` (NOT static), over each of its GF1 grain neighbours `g` (`fgNeighbors[fgStart[p]
.. fgStart[p+1])`, fixed order), with `d = p.pos − g.pos`, `dist = FxLength(d)`, `surf = g.radius` (the grain
exclusion radius; fluid particles are points):
```
if (dist < surf) {                                      // the fluid particle is inside the grain
    // POSITIONAL DISPLACEMENT — snap the fluid particle to the grain surface (the sand parts the fluid). This
    // is grain.h::CollideGrainSphere's snap (sphere = the grain): snap to surf along the outward normal.
    n   = FxNormalize(d);                               // outward normal (dist==0 -> +Y fallback)
    Δp += FxAdd(g.pos, FxScale(n, surf)) − p.pos;       // into the Jacobi dp[] (the GR3/CG3 surface push)
    // DRAG REACTION — the grain imparts momentum to the fluid (the equal-opposite of GF2's grain drag):
    p.vel += fxmul(kDragReaction, (g.vel − p.vel)) · dt;   // per axis (toward the grain velocity)
}
```
Then apply `p.pos += Δp` for all fluid particles (Jacobi). Static fluid particles (boundary) → Δp 0.
`kDragReaction` is a host-snapped Q16.16 constant (the GF2 `kDrag` partner). The sand DISPLACES the fluid (the
fluid can't occupy the grain volumes → it sits ON / seeps AROUND the bed) and drags it along. The projection
is the GR3 `CollideGrainSphere` with each grain as the sphere + the drag-reaction term + the Jacobi `dp[]`.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CG3 displacement to MIRROR (`engine/sim/couple_grain.h`):** `ApplyBodyToGrains` (the per-particle
  project-out-of-sphere + drag reaction, Jacobi dp[]) — GF3's `ApplyGrainsToFluid` is the SAME shape, per-FLUID
  over its grain neighbours, each GRAIN the sphere.
- **The grain sphere projection to REUSE (`engine/sim/grain.h`):** `CollideGrainSphere` (the per-particle
  project-out-of-a-sphere — snap to the surface along the outward normal, int64 `FxLength`/`FxNormalize`,
  `dist==0` +Y fallback). GF3's positional push IS this projection with the grain as the sphere (surf =
  `g.radius`). Read it to get the exact normal/snap formula; compute the Δp form for the Jacobi dp[].
- **The GF1/GF2 world + neighbours (this branch's `couple_gf.h`, read-only):** `CGFWorld`, `BuildCGFNeighbors` →
  `CGFNeighbors{..., fgStart, fgNeighbors}` (GF3 iterates `fgNeighbors` = each fluid particle's grain
  neighbours), `fluid::FluidParticle` (pos, vel, `kFlagStatic`), `grain::GrainParticle` (pos, vel, radius).
  `fpx::FxLength`/`FxNormalize`/`fxmul`/`FxAdd`/`FxSub`/`FxScale`. DO NOT modify grain.h/fluid.h/fpx.h/couple.h/
  couple_grain.h or GF1/GF2 code.
- **The int64 Jacobi SHADER mold (`shaders/cgrain_displace.comp.hlsl` CG3 / `grain_collide.comp.hlsl` GR3):** the
  per-particle int64 Jacobi projection — `cgf_displace.comp` is the SAME `[numthreads(64,1,1)]` structure (one
  thread per FLUID particle), Vulkan-only (NOT in `hf_gen_msl`). The GF1 cross-query passes stay int32
  MSL-native.
- **Showcase + registration:** GF1/GF2's `--cgf-*-shot` plumbing; `scripts/verify.ps1`, `engine/editor/
  introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden** — the GR2/CP2 lesson),
  `tests/cgf_test.cpp`.

## Design decisions (locked)
1. **`ApplyGrainsToFluid(world, neighbors)` (per-fluid Jacobi, the CG3/GR3 mold).** For each fluid particle
   (skip static), over each of its `fgNeighbors` grains (fixed order): if inside (`dist < g.radius`), accumulate
   the surface-snap push into `dp[]` + apply the drag-reaction velocity impulse. Apply `pos += dp` for all
   after (Jacobi). int64. `cgf_displace.comp` copies this body VERBATIM (one thread per fluid particle). The
   push formula == GR3 `CollideGrainSphere` (the grain as the sphere). Deterministic.
2. **Showcase `--cgf-displace-shot <out>` (Vulkan) AND `--cgf-displace` (Metal) — WIRE BOTH.** The GF1/GF2
   scene (a grain bed + a fluid block overlapping it), but run the displacement: `ApplyGrainsToFluid` parts the
   fluid out of the grain volumes (the fluid sits on / around the sand). Vulkan: the GPU pass → **memcmp vs
   the CPU `ApplyGrainsToFluid` reference**. Metal: the CPU reference. Color the grains + fluid to a side/top
   view (the fluid displaced out of the bed). Golden = `tests/golden/metal/cgf_displace.png` (Mac-baked by the
   CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU fluid array after the displacement == the CPU reference byte-for-byte.
     Print `cgf-displace: {grains:<G>, fluid:<F>, displaced:<D>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `cgf-displace determinism: two runs BYTE-IDENTICAL`.
   - **(3) displacement / no-penetration:** no fluid particle ends inside a grain — print `cgf-displace
     no-penetration: {penBefore:<Pb>, penAfter:<Pa>} (fluid parted from sand)` with `Pa < Pb` (the summed
     fluid-into-grain penetration relieved; Jacobi single-projection so the residual is deterministic-but-nonzero
     if a fluid particle is inside multiple grains — the FL4/GR3/CG3 caveat shape). Assert `displaced > 0`.
   - **(4) no-op:** the fluid clear of the grains (or zero grains) → the fluid is unchanged. Print `cgf-displace
     clear: fluid unchanged (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/cgf_displace.png`; do NOT commit it.** Existing 149 image
     goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests `tests/cgf_test.cpp` additions (pure CPU):** `ApplyGrainsToFluid` — a fluid particle inside a grain
   → snapped to the grain surface (`|p−g.pos| == g.radius` within an LSB epsilon) + the drag-reaction velocity
   toward the grain; a fluid particle outside → untouched; a static fluid particle → untouched; two grains →
   the fixed-order projection; `MeasureFluidGrainPenetration` on a known overlap. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-cgf-displace` (features) + `--cgf-displace-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (the GR2/CP2
   lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the GF2/CG3 surface).
  `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/grain.h` + `fluid.h` + `fpx.h` + `cloth.h` +
  `couple.h` + `couple_grain.h` + `engine/physics/` UNCHANGED. GF1/GF2 cgf code + shaders UNCHANGED (GF3
  additive). Report the seam empty.

## Out of scope (YAGNI — later GF slices)
The full coupled step (GF4 — GF3 is the grain→fluid pass in isolation), lockstep (GF5), the lit render (GF6).
Surface tension, capillarity, multiple grains overlapping (single-projection residual documented). GF3 claims
ONLY: a deterministic grain→fluid displacement + drag reaction that parts the fluid out of the sand,
bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 99) + the new `cgf_test` displace cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--cgf-displace-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows the fluid PARTED out of the grain bed (the fluid sits on/around the sand — pixel-check; the
   NAV6/CL6 lesson).**
3. Metal: `visual_test --cgf-displace` → new golden `tests/golden/metal/cgf_displace.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm `cgf_displace.comp` is
   correctly NOT MSL-generated (int64); the GF1 cross-query passes still ARE.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgf_displace.png` added; the
   other 149 byte-identical (re-run `--cgf-query/buoyancy-shot` → still bit-exact). `git diff master --stat --
   tests/golden` = ONLY `cgf_displace.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cgf-displace` + `--cgf-displace-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/grain.h` + `fluid.h` + `fpx.h` + `cloth.h` + `couple.h` +
   `couple_grain.h` + `engine/physics/` + GF1/GF2 cgf code/shaders byte-unchanged). `scripts/verify.ps1`
   updated: `cgf_displace` golden in the Mac loop + `--cgf-displace-shot` in `$vkShots`. `cgf_displace.comp`
   NOT in `hf_gen_msl`.
