# Slice CG3 — Deterministic Rigid↔Grain Coupling: GRAIN REACTION / DISPLACEMENT (body→grain) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #12 (DETERMINISTIC
> TWO-WAY RIGID↔GRAIN COUPLING, `hf::sim::cgrain`). The Newton's-3rd-law HALF of CG2: the body now pushes BACK
> on the sand — each grain inside a body is projected out to the body surface (the body PARTS the sand, a
> cavity) and receives the equal-opposite drag impulse (the body imparts momentum to the grains). Completes
> the two-way exchange (CG2 grain→body, CG3 body→grain). JACOBI, per-grain, `[numthreads(64,1,1)]` → NO TDR.
> The CP3 twin — and the per-grain projection is LITERALLY `grain.h`'s `CollideGrainSphere` /
> `GrainSphereFromBody` (the grain-out-of-`fpx::FxBody` bridge ALREADY exists). Branch: `slice-cg3`. See
> [[hazard-forge-couple-grain-roadmap]].

**Goal:** Extend `engine/sim/couple_grain.h` (additive — CG1/CG2 byte-unchanged) with `ApplyBodyToGrains(world)`
(the per-grain projection-out-of-body + drag reaction, the GR3/CP3 mold) + a `MeasureGrainBodyPenetration`
helper. Add `shaders/cgrain_displace.comp.hlsl` (int64 → **Vulkan-only** + Metal CPU reference). Add
`--cgrain-displace-shot` (Vulkan) / `--cgrain-displace` (Metal). Bake the integer golden `cgrain_displace`.
NO new RHI.

## Design call: per-GRAIN Jacobi over the (tiny) body set (race-free) + the INTEGER bar (strict zero-diff)
CG2's body-force reduction was per-BODY (over its gathered grains). CG3's grain displacement is the mirror:
**per-GRAIN over the body set** — one thread per GRAIN, each grain iterates the tiny body list (fixed order),
and for each body that contains it, accumulates the positional push (snap to the body surface) into a SEPARATE
`dp[]` (Jacobi) + applies the drag-reaction velocity impulse. Each grain writes ONLY its own `dp`/`vel` →
per-grain-disjoint, race-free, NO atomics, `[numthreads(64,1,1)]` multi-thread, NO TDR (the GR3/CP3 win; this
is the EXACT shape of `grain.h`'s `CollideGrainSpheres`, which iterates per-grain per-sphere). int64
(`FxLength`/`FxNormalize`/`fxmul`) → `cgrain_displace.comp` Vulkan-only + the Metal showcase runs the CPU
`ApplyBodyToGrains` (byte-identical by construction, the GR3/CP3 split). Bar: strict INTEGER (Vulkan GPU ==
Metal CPU-ref == golden, ZERO differing pixels).

## The displacement + drag-reaction model (Newton's 3rd law to CG2 — reuse the grain.h sphere projection)
For each grain `g` (NOT static), over each dynamic body `b` (fixed order), with `d = g.pos − b.pos`, `dist =
FxLength(d)`, `surf = b.radius + g.radius`:
```
if (dist < surf) {                                      // the grain is inside the body
    // POSITIONAL DISPLACEMENT — snap the grain to the body surface (the body parts the sand). This is
    // EXACTLY grain.h::CollideGrainSphere(g, GrainSphereFromBody(b)): snap to surf along the outward normal.
    n   = FxNormalize(d);                               // outward normal (dist==0 -> +Y fallback)
    Δp += FxAdd(b.pos, FxScale(n, surf)) − g.pos;       // into the Jacobi dp[] (the GR3/CP3 surface push)
    // DRAG REACTION — the body imparts momentum to the grain (the equal-opposite of CG2's body drag):
    g.vel += fxmul(kDragReaction, (b.vel − g.vel)) · dt;   // per axis (toward the body velocity)
}
```
Then apply `g.pos += Δp` for all grains (Jacobi). Static grains (boundary) → Δp 0. `kDragReaction` is a
host-snapped Q16.16 constant (the CG2 `kDrag` partner). The body DISPLACES the sand (a cavity where the body
is) and drags the surrounding grains along. The projection is the GR3 `CollideGrainSphere` with the body as
the sphere + the drag-reaction term + the Jacobi `dp[]` double-buffer.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CP3 fluid-displacement to MIRROR (`engine/sim/couple.h`):** `ApplyBodyToFluid` (the per-particle
  project-out-of-body + drag reaction, Jacobi dp[]) — CG3's `ApplyBodyToGrains` is the SAME shape with
  `grain::GrainParticle` instead of `fluid::FluidParticle`.
- **The grain sphere projection to REUSE DIRECTLY (`engine/sim/grain.h`):** `CollideGrainSphere(GrainParticle&,
  const GrainSphereCollider&)` (the per-grain project-out-of-a-sphere — snap to `sphereR + grainR` along the
  outward normal, int64 `FxLength`/`FxNormalize`, `dist==0` +Y fallback) + `GrainSphereFromBody(const
  fpx::FxBody&)` (makes a `GrainSphereCollider` from a body — `grain.h:480`). The grain-out-of-FxBody bridge
  ALREADY exists; CG3's positional push IS this projection (compute the Δp form for the Jacobi dp[]). Read it
  to get the exact surf/normal/snap formula.
- **The CG1/CG2 world + body (this branch's `couple_grain.h` + `fpx.h`/`grain.h`, read-only):** `CGrainWorld`,
  `GrainParticle` (pos, vel, flags — `grain::kFlagStatic`), `fpx::FxBody` (pos, vel, radius, flags),
  `FxLength`/`FxNormalize`/`fxmul`/`FxAdd`/`FxSub`/`FxScale`. DO NOT modify fpx.h/grain.h/fluid.h/cloth.h/
  couple.h or CG1/CG2 code.
- **The int64 Jacobi SHADER mold (`shaders/couple_displace.comp.hlsl` — CP3 / `grain_collide.comp.hlsl` — GR3):**
  the per-particle int64 Jacobi projection — `cgrain_displace.comp` is the SAME `[numthreads(64,1,1)]`
  structure (one thread per grain), Vulkan-only (NOT in `hf_gen_msl`). The CG1 query passes stay int32
  MSL-native; the CG2 `cgrain_support` is unchanged.
- **Showcase + registration:** CG1/CG2's `--cgrain-*-shot` plumbing; `scripts/verify.ps1`,
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden** — the
  GR2/CP2 lesson), `tests/cgrain_test.cpp`.

## Design decisions (locked)
1. **`ApplyBodyToGrains(world)` (per-grain Jacobi, the GR3/CP3 mold).** For each grain (skip static), over each
   dynamic body (fixed order): if inside (`dist < b.radius + g.radius`), accumulate the surface-snap push into
   `dp[]` + apply the drag-reaction velocity impulse. Apply `pos += dp` for all after (Jacobi). int64.
   `cgrain_displace.comp` copies this body VERBATIM (one thread per grain). The push formula == GR3
   `CollideGrainSphere` (body as the sphere). Deterministic.
2. **Showcase `--cgrain-displace-shot <out>` (Vulkan) AND `--cgrain-displace` (Metal) — WIRE BOTH.** The CG2
   dense-bed scene, but run the displacement: a body submerged/buried in the bed → `ApplyBodyToGrains` parts
   the sand around it (a visible cavity). Vulkan: the GPU pass → **memcmp vs the CPU `ApplyBodyToGrains`
   reference**. Metal: the CPU reference. Color the grains + body to a side/top view (the cavity around the
   buried body). Golden = `tests/golden/metal/cgrain_displace.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU grain array after the displacement == the CPU reference byte-for-byte.
     Print `cgrain-displace: {bodies:<B>, grains:<N>, displaced:<D>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `cgrain-displace determinism: two runs BYTE-IDENTICAL`.
   - **(3) displacement / no-penetration:** no grain ends inside a body — print `cgrain-displace
     no-penetration: {penBefore:<Pb>, penAfter:<Pa>} (sand parted)` with `Pa < Pb` (the summed grain-into-body
     penetration relieved; Jacobi single-projection so the residual is deterministic-but-nonzero if a grain is
     inside multiple bodies — the FL4/GR3/CP3 caveat shape). Assert `displaced > 0` (the body did part the sand).
   - **(4) no-op:** a body clear of the bed (or zero bodies) → the grains are unchanged. Print `cgrain-displace
     clear: sand unchanged (no-op)`.
   - **Golden discipline: ONLY `tests/golden/metal/cgrain_displace.png`; do NOT commit it.** Existing 143 image
     goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests `tests/cgrain_test.cpp` additions (pure CPU):** `ApplyBodyToGrains` — a grain inside a body →
   snapped to the body surface (`|g−b.pos| == b.radius + g.radius` within an LSB epsilon) + the drag-reaction
   velocity toward the body; a grain outside → untouched; a static grain → untouched; two bodies → the
   fixed-order projection; `MeasureGrainBodyPenetration` on a known overlap. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-cgrain-displace` (features) + `--cgrain-displace-shot`
   (showcases). **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`
   (the GR2/CP2 lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + barrier + read-back path (the CG2/GR3 surface).
  `rhi.h` + `rhi_factory` + backend dirs UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` +
  `couple.h` + `engine/physics/` UNCHANGED. CG1/CG2 cgrain code + shaders UNCHANGED (CG3 additive). Report the
  seam empty.

## Out of scope (YAGNI — later CG slices)
The full coupled step (CG4 — CG3 is the body→grain pass in isolation), lockstep (CG5), the lit render (CG6).
Two-way friction coupling, cohesion, multiple bodies overlapping (single-projection residual documented). CG3
claims ONLY: a deterministic body→grain displacement + drag reaction that parts the sand around a buried body,
bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 98) + the new `cgrain_test` displace cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--cgrain-displace-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows the sand PARTED around the buried body (a cavity — pixel-check; the NAV6/CL6 lesson).**
3. Metal: `visual_test --cgrain-displace` → new golden `tests/golden/metal/cgrain_displace.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm `cgrain_displace.comp`
   is correctly NOT MSL-generated (int64); the CG1 query passes still ARE.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgrain_displace.png` added;
   the other 143 byte-identical (re-run `--cgrain-query/support-shot` → still bit-exact). `git diff master
   --stat -- tests/golden` = ONLY `cgrain_displace.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cgrain-displace` + `--cgrain-displace-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple.h` +
   `engine/physics/` + CG1/CG2 cgrain code/shaders byte-unchanged). `scripts/verify.ps1` updated:
   `cgrain_displace` golden in the Mac loop + `--cgrain-displace-shot` in `$vkShots`. `cgrain_displace.comp`
   NOT in `hf_gen_msl`.
