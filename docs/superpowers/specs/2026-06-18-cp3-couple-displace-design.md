# Slice CP3 — Deterministic Rigid↔Fluid Coupling: FLUID REACTION / DISPLACEMENT (body→fluid) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #11 (DETERMINISTIC
> TWO-WAY RIGID↔FLUID COUPLING, `hf::sim::couple`). The Newton's-3rd-law HALF of CP2: the body now pushes
> BACK on the fluid — each fluid particle inside a body is projected out to the body surface (the body
> DISPLACES the fluid, a parting/wake) and receives the equal-opposite drag impulse (the body imparts its
> momentum to the fluid). Completes the two-way exchange (CP2 fluid→body, CP3 body→fluid). JACOBI,
> per-particle, `[numthreads(64,1,1)]` → NO TDR (the FL4/GR3 win). Branch: `slice-cp3`. See [[hazard-forge-couple-roadmap]].

**Goal:** Extend `engine/sim/couple.h` (additive — CP1/CP2 byte-unchanged) with `ApplyBodyToFluid(world)`
(the per-fluid-particle projection-out-of-body + drag reaction, the GR3 `CollideGrainSpheres` mold over the
body set) + a `MeasureFluidPenetration` helper. Add `shaders/couple_displace.comp.hlsl` (int64 → **Vulkan-only**
+ Metal CPU reference). Add `--couple-displace-shot` (Vulkan) / `--couple-displace` (Metal). Bake the integer
golden `couple_displace`. NO new RHI.

## Design call: per-PARTICLE Jacobi over the (tiny) body set (race-free) + the INTEGER bar (strict zero-diff)
CP2's body-force reduction was per-BODY (over its gathered particles). CP3's fluid displacement is the
mirror: **per-PARTICLE over the body set** — one thread per FLUID PARTICLE, each particle iterates the tiny
body list (fixed order), and for each body that contains it, accumulates the positional push (snap to the
body surface) into a SEPARATE `dp[]` (Jacobi) + applies the drag-reaction velocity impulse. Each particle
writes ONLY its own `dp`/`vel` → per-particle-disjoint, race-free, NO atomics, `[numthreads(64,1,1)]`
multi-thread, NO TDR (the FL4/GR3 win; this is the EXACT shape of GR3 `CollideGrainSpheres`, which iterates
per-particle per-sphere). int64 (`FxLength`/`FxNormalize`/`fxmul`) → `couple_displace.comp` Vulkan-only + the
Metal showcase runs the CPU `ApplyBodyToFluid` (byte-identical by construction, the FL4/GR3 split). Bar:
strict INTEGER (Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels). (CP3 does NOT need the CP1
per-body query — it re-tests the tiny body set per particle, the GR3 collide pattern; CP1's query is the
body-side link CP2/CP4 use.)

## The displacement + drag-reaction model (Newton's 3rd law to CP2)
For each fluid particle `p` (NOT static), over each dynamic body `b` (fixed order), with `d = p.pos − b.pos`,
`dist = FxLength(d)`:
```
if (dist < b.radius) {                                  // the particle is inside the body
    // POSITIONAL DISPLACEMENT — snap the particle to the body surface (the body parts the fluid):
    n   = FxNormalize(d);                               // outward normal (dist==0 -> +Y fallback)
    Δp += FxAdd(b.pos, FxScale(n, b.radius)) − p.pos;   // into the Jacobi dp[] (the GR3 CollideParticleSphere push)
    // DRAG REACTION — the body imparts momentum to the fluid (the equal-opposite of CP2's body drag):
    p.vel += fxmul(kDragReaction, (b.vel − p.vel)) · dt;   // per axis (toward the body velocity)
}
```
Then apply `p.pos += Δp` for all particles (Jacobi). Static particles (boundary) → Δp 0. `kDragReaction` is a
host-snapped Q16.16 constant (the CP2 `kDrag` partner). The body DISPLACES the fluid (a cavity/wake where the
body is) and drags the surrounding fluid along. This is the GR3 `CollideGrainSpheres` projection with the
body as the sphere + the drag-reaction term.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The GR3 collide to MIRROR (`engine/sim/grain.h`):** `CollideGrainSphere` (the per-particle project-out-of-
  a-sphere — snap to `sphereR + grainR` along the outward normal, the int64 `FxLength`/`FxNormalize`,
  `dist==0` +Y fallback) + `CollideGrainSpheres` (the per-particle, per-sphere fixed-order loop) — CP3's
  `ApplyBodyToFluid` is the SAME shape with `fpx::FxBody` as the sphere (project to `b.radius`, the fluid
  particle is a point) + the drag-reaction velocity term + the Jacobi `dp[]` double-buffer.
- **The fluid particle (`engine/sim/fluid.h`, read-only):** `FluidParticle` (pos, vel, flags — `kFlagStatic`).
  The CP1/CP2 `CoupleWorld` (this branch). DO NOT modify fpx.h/fluid.h/cloth.h/grain.h or CP1/CP2 code.
- **The Q16.16 toolbox (read-only, `engine/sim/fpx.h`):** `FxLength`/`FxNormalize`/`fxmul`/`FxAdd`/`FxSub`/
  `FxScale`. `FxBody` (pos, vel, radius, flags).
- **The int64 Jacobi SHADER mold (`shaders/grain_collide.comp.hlsl` / `grain_contact_dp.comp.hlsl`):** the
  per-particle int64 Jacobi accumulate — `couple_displace.comp` is the SAME `[numthreads(64,1,1)]` structure,
  Vulkan-only (NOT in `hf_gen_msl`). The body-side CP2 `couple_buoyancy.comp` is unchanged.
- **Showcase + registration:** CP1/CP2's `--couple-*-shot` plumbing; `scripts/verify.ps1`,
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden** — the
  GR2 lesson), `tests/couple_test.cpp`.

## Design decisions (locked)
1. **`ApplyBodyToFluid(world)` (per-particle Jacobi, the GR3 mold).** For each fluid particle (skip static),
   over each dynamic body (fixed order): if inside (`dist < b.radius`), accumulate the surface-snap push into
   `dp[]` + apply the drag-reaction velocity impulse. Apply `pos += dp` for all after (Jacobi). int64.
   `couple_displace.comp` copies this body VERBATIM (one thread per particle). Deterministic.
2. **Showcase `--couple-displace-shot <out>` (Vulkan) AND `--couple-displace` (Metal) — WIRE BOTH.** The CP2
   scene (a fluid pool + a body), but run the displacement: a body submerged in / dropped through the pool →
   `ApplyBodyToFluid` parts the fluid around it (a visible cavity/wake). Vulkan: the GPU pass → **memcmp vs
   the CPU `ApplyBodyToFluid` reference**. Metal: the CPU reference. Color the fluid + body to a side view
   (the cavity around the submerged body visible). Golden = `tests/golden/metal/couple_displace.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU fluid array after the displacement == the CPU reference byte-for-byte.
     Print `couple-displace: {bodies:<B>, particles:<N>, displaced:<D>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `couple-displace determinism: two runs BYTE-IDENTICAL`.
   - **(3) displacement / no-penetration:** no fluid particle ends inside a body — print `couple-displace
     no-penetration: {penBefore:<Pb>, penAfter:<Pa>} (fluid parted)` with `Pa < Pb` (the summed
     particle-into-body penetration relieved; Jacobi single-projection so the residual is deterministic-but-
     nonzero if a particle is inside multiple bodies — the FL4/GR3 caveat shape). Assert `displaced > 0` (the
     body did part the fluid).
   - **(4) no-op:** a body clear of the fluid (or zero bodies) → the fluid is unchanged. Print
     `couple-displace clear: fluid unchanged (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/couple_displace.png`; do NOT commit it.** Existing 137
     image goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests `tests/couple_test.cpp` additions (pure CPU):** `ApplyBodyToFluid` — a particle inside a body →
   snapped to the body surface (`|p−b.pos| == b.radius` within an LSB epsilon) + the drag-reaction velocity
   toward the body; a particle outside → untouched; a static particle → untouched; two bodies → the fixed-order
   projection; `MeasureFluidPenetration` on a known overlap. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-couple-displace` (features) + `--couple-displace-shot`
   (showcases). **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`
   (the GR2 lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the CP2/GR3 surface).
  `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `fluid.h` + `cloth.h` + `grain.h` +
  `engine/physics/` UNCHANGED. CP1/CP2 couple code + shaders UNCHANGED (CP3 additive). Report the seam empty.

## Out of scope (YAGNI — later CP slices)
The full coupled step (CP4 — CP3 is the body→fluid pass in isolation), lockstep (CP5), the lit render (CP6).
Two-way density coupling, surface tension, splash/foam, multiple bodies overlapping (single-projection
residual documented). CP3 claims ONLY: a deterministic body→fluid displacement + drag reaction that parts the
fluid around a submerged body, bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 97) + the new `couple_test` displace cases. Clean
   under `windows-msvc-asan`.
2. **proofs + visual:** `--couple-displace-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows the fluid PARTED around the submerged body (a cavity/wake — pixel-check; the NAV6/CL6 lesson).**
3. Metal: `visual_test --couple-displace` → new golden `tests/golden/metal/couple_displace.png`; two runs
   DIFF 0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm
   `couple_displace.comp` is correctly NOT MSL-generated (int64); the CP1 query passes still ARE.**
   Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `couple_displace.png` added;
   the other 137 byte-identical (re-run `--couple-query/buoyancy-shot` → still bit-exact). `git diff master
   --stat -- tests/golden` = ONLY `couple_displace.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-couple-displace` + `--couple-displace-shot`; introspect
   test updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h` + `fluid.h` + `cloth.h` + `grain.h` +
   `engine/physics/` + CP1/CP2 couple code/shaders byte-unchanged). `scripts/verify.ps1` updated:
   `couple_displace` golden in the Mac loop + `--couple-displace-shot` in `$vkShots`. `couple_displace.comp`
   NOT in `hf_gen_msl`.
