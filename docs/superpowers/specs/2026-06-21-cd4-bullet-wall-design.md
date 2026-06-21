# Slice CD4 — Deterministic Integer CCD: A BULLET THROUGH A THIN WALL STOPS (the new-physics beat) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice (THE
> HEADLINE BEAT) of FLAGSHIP #24 (DETERMINISTIC INTEGER CCD, `hf::sim::ccd`). CD1-CD3 built the time-of-impact,
> the swept broadphase, and the substepped CCD world step (proven byte-identical + no-tunnel vs the discrete
> step). CD4 is the money-shot demonstration: a **fast projectile fired at a thin static wall is ARRESTED at the
> wall face** — the per-tick travel is many times the wall thickness, so a discrete solver tunnels it straight
> through, but the CCD step stops it exactly at impact, deterministically and bit-identically across backends.
> This is the GJ4/BP4-style "new capability" beat — the visible, falsifiable payoff of the whole flagship: the
> single most recognizable "the float engines have this and you don't" gap, closed deterministically. CD4 REUSES
> CD3's `StepHullWorldCCD` + `ccd_step.comp` ENTIRELY — it adds only the dedicated bullet-wall scene + the
> impact-tick measurement + the side-by-side discrete control. **NO new shader.** APPEND to `engine/sim/ccd.h`
> (CD1-CD3 + broad.h/gjk.h/convex.h/fpx.h/fric.h/persist.h/grain.h BYTE-FROZEN). Branch: `slice-cd4`. See
> [[hazard-forge-ccd-roadmap]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/ccd.h` (additive — CD1-CD3 byte-unchanged) with `MakeBulletWallScene` (a thin static
wall + a fast dynamic projectile aimed at it) + `BulletMeasure` / `MeasureBullet` (`tunneled`, `impactTick`, the
arrested contact pose). Add the showcase `--ccd-bullet-shot` (Vulkan) / `--ccd-bullet` (Metal) — both step the
scene with `StepHullWorldCCD` (the projectile stops at the wall) AND, as a CONTROL, with the discrete
`broad::StepHullWorldBP` (the projectile tunnels through). Bake the integer golden `ccd_bullet`. **NO new shader,
NO new RHI** — reuse CD3's `ccd_step.comp` over the bullet-wall world.

## Design call: the dedicated headline scene + the discrete control; reuse CD3 verbatim

- **`MakeBulletWallScene() → gjk::HullWorld`** — a deterministic scene: a THIN static wall (a box hull with a
  small thickness along the projectile's travel axis, large in the other two) + a fast DYNAMIC projectile (a small
  box/octa hull) positioned on the approach side with a velocity whose per-tick travel (`|vel|·dt`) is MANY times
  the wall thickness (e.g. 5-10×) — so a discrete tick steps the projectile from before the wall to past it in
  one move (a guaranteed tunnel for the discrete solver). Optionally a floor + a couple of slow bodies for
  context. All integer, fixed.
- **`StepHullWorldCCD` (CD3) over this scene** arrests the projectile at the wall's near face (the substep
  advances it to the TOI, the resolve stops it). `MeasureBullet(world, wall index, projectile index)` reports
  `tunneled` (is the projectile past the wall's far face?), `impactTick`, and the arrested contact pose. The CCD
  result: `tunneled = false`.
- **The discrete control:** `broad::StepHullWorldBP` (BP4) over the IDENTICAL scene steps the projectile straight
  through (the discrete narrowphase never samples the in-between pose) → `tunneled = true`. The two final worlds
  DIFFER. This is the falsifiable headline.
- **GPU/CPU + render:** the GPU path runs the CCD scene through `ccd_step.comp` (CD3's shader, chunked
  1-tick/dispatch) and memcmps the final world vs the CPU `StepHullWorldCCDN`. The render shows the CCD final
  world (the projectile arrested at the wall) — optionally with a faint marker/ghost of the discrete tunnel-through
  position for contrast. PURE reuse of CD3's machinery.

## Reuse map (file:line)
- **CD1-CD3 `engine/sim/ccd.h` (APPEND after CD3's `MeasureCcd`):** `StepHullWorldCCD`/`StepHullWorldCCDN`,
  `CcdStepConfig`, `CcdMeasure`. The CD3 `ccd_step.comp` shader (REUSED — no new shader). CD1-CD3 frozen.
- **broad.h (read-only — REUSE):** `broad::StepHullWorldBP` (the discrete control), `gjk::HullWorld`/`FxHull`/the
  canonical hull builders (`MakeBox`/`MakeOcta`), `gjk::MeasureHullStack`. Do NOT modify broad.h/gjk.h/etc —
  BYTE-FROZEN.
- **The showcase precedent:** CD3's `--ccd-step-shot`/`--ccd-step` (the chunked GPU step + GPU==CPU memcmp + the
  side-view render + the no-tunnel comparison). Mirror for `--ccd-bullet`, swapping in `MakeBulletWallScene`.
- **Registration:** `scripts/verify.ps1` (append `ccd_bullet` + `--ccd-bullet-shot` to `$vkShots`),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to
  `tests/ccd_test.cpp`. (NO new shader → nothing for `hf_gen_msl`; `ccd_step.comp` is already registered.)

## Design decisions (locked)
1. **APPEND to `engine/sim/ccd.h`** (CD1-CD3 byte-frozen): `MakeBulletWallScene`, `BulletMeasure`,
   `MeasureBullet`. **NO new shader, NO new RHI** — reuse CD3's `ccd_step.comp` over the bullet-wall world.
2. **Showcase `--ccd-bullet-shot <out>` (Vulkan) AND `--ccd-bullet` (Metal) — WIRE BOTH (grep your own
   visual_test.mm for `--ccd-bullet` before reporting DONE).** Build `MakeBulletWallScene`, step it with
   `StepHullWorldCCD` (the GPU via the reused `ccd_step.comp` chunked 1-tick/dispatch on Vulkan; the CPU on Metal)
   + the discrete `broad::StepHullWorldBP` control. Vulkan memcmps the GPU CCD final world vs the CPU
   `StepHullWorldCCDN`. BOTH render the CCD final world (projectile arrested at the wall). Golden =
   `tests/golden/metal/ccd_bullet.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU:** `ccd-bullet: {tunneled:false, impactTick:<T>} GPU==CPU BIT-EXACT` — the GPU CCD final world
     == the CPU `StepHullWorldCCDN` byte-for-byte, and the projectile did NOT tunnel; assert.
   - **(2) determinism:** `ccd-bullet determinism: two runs BYTE-IDENTICAL`.
   - **(3) THE DISCRETE CONTROL:** `ccd-bullet control: {ccdTunneled:false, discreteTunneled:true}` — the IDENTICAL
     scene under the discrete `broad::StepHullWorldBP` TUNNELS the projectile (tunneled=true), while the CCD step
     does not (tunneled=false); assert both. **CCD provably stops the bullet the discrete solver lets pass.**
   - **Golden discipline: ONLY `tests/golden/metal/ccd_bullet.png`; do NOT commit it.** Existing 218 goldens
     UNTOUCHED.
4. **Cross-backend bar (int64 → strict):** Vulkan GPU==CPU bit-exact (~3× clean — reuses CD3's TDR-safe
   1-tick/dispatch chunking); Metal CPU-ref byte-identical; cross-vendor ZERO differing pixels.
5. **Tests — APPEND to `tests/ccd_test.cpp`:** `MakeBulletWallScene` builds the expected wall + fast projectile
   (the per-tick travel ≫ wall thickness); `StepHullWorldCCDN` over it → `tunneled=false`, projectile arrested on
   the approach side; the discrete `broad::StepHullWorldBP` over the SAME scene → `tunneled=true` (the control);
   deterministic (two runs byte-equal). Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-ccd-bullet` (features) + `--ccd-bullet-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None — and NO new shader.** Reuses CD3's `ccd_step.comp`. `engine/sim/ccd.h` APPEND-only (CD1-CD3 frozen);
  broad.h/gjk.h/convex.h/fpx.h/etc + ALL other sim headers + ALL existing shaders (incl. `ccd_step.comp`)
  UNCHANGED. Report the seam: NO new shader, no RHI change, no frozen-file edit, ccd.h append-only.

## Out of scope (YAGNI — later slices)
Lockstep (CD5), lit render (CD6). CD4 claims ONLY: a deterministic, bit-exact (CPU↔Vulkan↔Metal) demonstration
that the CCD step arrests a fast projectile at a thin wall the discrete step tunnels through, with the integer
golden + the three proofs. CAVEATS: the projectile speed is chosen below the `maxSubsteps` budget (a projectile
fast enough to exceed the budget can still tunnel — a documented deterministic limit); inherited CD1-CD3
within-band.

## Verification gate
1. `ctest --preset windows-msvc-debug -R "ccd|broad|introspect"` green. Clean under `windows-msvc-asan` (separate
   build + test).
2. **proofs + visual:** `--ccd-bullet-shot` on Vulkan: the 3 proofs (incl. the discrete control) + exit 0 under
   the conan validation layer → ZERO VUID. **~3 runs all GPU==CPU (reuses CD3's 1-tick/dispatch — TDR-safe). VERIFY
   the image shows the projectile ARRESTED at the wall face (not through it).**
3. Metal: `visual_test --ccd-bullet` → `tests/golden/metal/ccd_bullet.png`; two runs DIFF 0.0000. Confirm
   `visual_test.mm` in the diff; NO new shader added. Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `ccd_bullet.png` added; the other 218 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+deterministic-ccd-bullet` + `--ccd-bullet-shot`; introspect test updated.
6. Seam grep clean (`rhi.h` + CD1-CD3 ccd.h code + broad.h/gjk.h/convex.h/fpx.h/etc + ALL other sim headers + ALL
   existing shaders byte-unchanged; ccd.h APPEND-only; NO new shader, no RHI change). `ccd_bullet` in the Mac loop
   + `--ccd-bullet-shot` in `$vkShots`.
