# Slice GF5 — Deterministic Grain↔Fluid Coupling: LOCKSTEP + ROLLBACK (the netcode headline) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #13 (DETERMINISTIC
> TWO-WAY GRAIN↔FLUID COUPLING, `hf::sim::cgf`). THE NETCODE HEADLINE, PURE CPU: a peer fed the input stream
> ALONE re-derives the EXACT coupled mud state — BOTH the grain pool AND the fluid pool — bit-for-bit, and a
> rollback re-sim corrects a misprediction EXACTLY. This is the FPX5/CL5/FL5/GR5/CP5/CG5 lockstep harness applied
> to the two-particle-pool coupled world — the first time a wet-sand / mud simulation is shown to be
> deterministically replayable across platforms (UE5's float Chaos/Niagara cannot). NO new shader, NO new RHI —
> pure CPU over the GF4 `StepCGF`. Branch: `slice-gf5`. See [[hazard-forge-couple-gf-roadmap]].

**Goal:** Extend `engine/sim/couple_gf.h` (additive — GF1/GF2/GF3/GF4 byte-unchanged) with the lockstep/rollback
harness over `CGFWorld`: `CGFCommand` (target a grain OR a fluid particle), `ApplyCGFCommand`, `SimCGFTick`,
`CGFSnapshot`/`SnapshotCGF`/`RestoreCGF`, `RunCGFLockstep`, `RunCGFRollback`. Add `--cgf-lockstep-shot` (Vulkan)
/ `--cgf-lockstep` (Metal). Bake the integer golden `cgf_lockstep`. **NO new shader, NO new RHI.**

## Design call: the CG5 harness VERBATIM over two particle pools (kernel threaded through)
GF5 is the GF-arc's `RunCGrainLockstep` (CG5): the lockstep/rollback machinery is pure CPU — snapshot both pools,
re-run `StepCGF` from inputs alone, memcmp authority vs replica across BOTH pools. The ONE difference from CG5 is
that `StepCGF` takes a `FluidKernel` argument (GF4's per-step kernel), so `SimCGFTick`/`RunCGFLockstep`/
`RunCGFRollback` thread the kernel through (built once by the caller from the scene, constant across ticks). Bar:
strict INTEGER — `StepCGF` is already integer-bit-exact cross-platform (GF4 proved it), so a peer/platform fed
the same init + stream produces a byte-identical final state. The two-run / cross-vendor proof is the SAME shape
as GF4 plus the lockstep authority==replica + rollback==authority memcmps.

## The harness (mirror CG5 `couple_grain.h` EXACTLY, two particle pools + the kernel)
- **`CGFCommand{tick, kind, target, arg}`** — the per-tick input. Two kinds (the grain↔fluid analogue of CG5's
  body-shove / grain-wind): `kCmdGrainWind = 0` (`arg` added to the target GRAIN's velocity — a sand gust) and
  `kCmdFluidPush = 1` (`arg` added to the target FLUID particle's velocity — a fluid jet). A
  `std::vector<CGFCommand>` is the STREAM, processed in ARRAY ORDER per tick (the deterministic-order contract).
- **`ApplyCGFCommand(world, c)`** — pure integer: add `arg` to the target grain's or fluid particle's velocity.
  Out-of-range target → no-op (deterministic); unknown kind → no-op; STATIC grains (`grain::kFlagStatic`) and
  STATIC fluid (`fluid::kFlagStatic`) are NEVER mutated (they hold). The `ApplyCGrainCommand` twin.
- **`SimCGFTick(world, kernel, stream, tick, dt, iters)`** — (1) apply ALL commands with `.tick == tick` in
  ARRAY ORDER; (2) `StepCGF(world, kernel, dt, iters)` one step. The `SimCGrainTick` twin (+ kernel).
- **`CGFSnapshot{grains, fluid}` / `SnapshotCGF(world)` / `RestoreCGF(world, snap)`** — deep-copy BOTH pools
  (std::vector value copy). The `CGrainSnapshot`/`SnapshotCGrain`/`RestoreCGrain` twin (grains + fluid instead of
  bodies + grains). Bit-exact round-trip.
- **`RunCGFLockstep(init, kernel, stream, ticks, dt, iters)`** — run `ticks` `SimCGFTick`s from a COPY of `init`
  applying the stream → the final coupled state. authority = RunCGFLockstep(...); replica = RunCGFLockstep(...)
  from the SAME init + stream (inputs ONLY) → BIT-IDENTICAL by determinism (memcmp BOTH pools). The
  `RunCGrainLockstep` twin.
- **`RunCGFRollback(init, kernel, authStream, mispredictStream, ticks, mispredictTick, dt, iters)`** — (1)
  advance 0..mispredictTick with authStream; (2) snapshot at mispredictTick; (2b) speculatively advance ≤3 ticks
  with the MISPREDICTED stream (the diverging client prediction); (3) RestoreCGF + re-sim mispredictTick..ticks
  with the CORRECT authStream → the corrected state. Asserts `== RunCGFLockstep(init, authStream, ticks)` (the
  rollback corrected EXACTLY, BOTH pools) AND that the mispredicted-before-rollback state DIFFERED (a real
  divergence fixed). The `RunCGrainRollback` twin.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CG5 harness to MIRROR (`engine/sim/couple_grain.h:609-717`):** `kCmd*` constants, `CGrainCommand`,
  `ApplyCGrainCommand`, `SimCGrainTick`, `CGrainSnapshot`, `SnapshotCGrain`, `RestoreCGrain`, `RunCGrainLockstep`,
  `RunCGrainRollback`. GF5 is the SAME shapes with `grains`+`fluid` (two particle pools) and the kernel threaded
  through every Sim/Run entry. Read it top-to-bottom and translate body→grain-target, grain→fluid-target.
- **The GF4 coupled step (this branch's `couple_gf.h`, read-only — call VERBATIM):** `StepCGF(world, kernel, dt,
  iters)`, `CGFWorld` (grains, fluid, gravity, dt, groundY, h), `fluid::FluidKernel`. GF5 only ADDS the harness;
  it does NOT touch `StepCGF` or GF1/GF2/GF3/GF4 code. `grain::kFlagStatic`, `fluid::kFlagStatic`. **DO NOT modify
  grain.h/fluid.h/fpx.h/cloth.h/couple.h/couple_grain.h/engine/physics or GF1-GF4 code** — GF5 is the additive
  sibling.
- **The CG5 showcase mold (the `--cgrain-lockstep-shot` / `--cgrain-lockstep` plumbing):** GF5 wires
  `--cgf-lockstep-shot` (Vulkan) + `--cgf-lockstep` (Metal) the SAME way — a PURE-CPU showcase (NO GPU dispatch
  needed; the proof is authority==replica + rollback==authority memcmps, plus a render of the final coupled
  state). Render the final lockstep state with the SAME camera/coloring as the GF4 `cgf_step` showcase (reuse
  that render path verbatim — a wet-sand side view).
- **Showcase + registration:** GF1-GF4's `--cgf-*-shot` plumbing; `scripts/verify.ps1`, `engine/editor/
  introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect JSON golden** — the GR2/CP2/GF3 lesson),
  `tests/cgf_test.cpp`.

## Design decisions (locked)
1. **The 6 harness functions above, mirroring CG5 verbatim over two particle pools + the threaded kernel.** Pure
   CPU, pure integer. NO new shader, NO new RHI. `StepCGF` and GF1-GF4 stay byte-frozen.
2. **Two command kinds:** `kCmdGrainWind` (grain velocity) + `kCmdFluidPush` (fluid velocity). Enough to drive a
   meaningful divergence in BOTH pools for the rollback proof.
3. **Showcase `--cgf-lockstep-shot <out>` (Vulkan) AND `--cgf-lockstep` (Metal) — WIRE BOTH (PURE CPU).** The GF4
   wet-sand scene, plus a command stream (a sand gust + a fluid jet at a few ticks). Run `RunCGFLockstep` twice
   (authority + replica) and `RunCGFRollback` once; assert authority==replica AND rollback==authority (BOTH pools
   memcmp) AND mispredicted≠authority. Render the final coupled state (reuse the GF4 `cgf_step` render path).
   Golden = `tests/golden/metal/cgf_lockstep.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) lockstep authority==replica:** two independent runs from the same init+stream → identical BOTH pools.
     Print `cgf-lockstep: {grains:<G>, fluid:<F>, ticks:<T>} authority==replica BIT-IDENTICAL`.
   - **(2) rollback==authority:** `RunCGFRollback` final state == `RunCGFLockstep(authStream)` final state, BOTH
     pools. Print `cgf-lockstep rollback: corrected==authority BIT-EXACT (BOTH pools)`.
   - **(3) the misprediction was real:** the speculative (pre-rollback) state DIFFERED from authority. Print
     `cgf-lockstep mispredict: diverged before rollback (real divergence fixed)`.
   - **(4) determinism:** two renders of the final state → byte-identical. Print `cgf-lockstep determinism: two
     runs BYTE-IDENTICAL`.
   - **Golden discipline: ONLY `tests/golden/metal/cgf_lockstep.png`; do NOT commit it.** Existing 151 image
     goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** the final lockstep render is Vulkan == Metal CPU-ref == golden, ZERO
   differing pixels (the GF4 render is already integer-bit-exact; the lockstep state feeding it is integer).
6. **Tests `tests/cgf_test.cpp` additions (pure CPU):** `ApplyCGFCommand` (grain-wind adds to a grain vel;
   fluid-push adds to a fluid vel; out-of-range/unknown/static → no-op); `SnapshotCGF`/`RestoreCGF` bit-exact
   round-trip on BOTH pools; `RunCGFLockstep` authority==replica on a tiny scene; `RunCGFRollback` ==
   `RunCGFLockstep(authStream)` AND mispredicted≠authority. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-cgf-lockstep` (features) + `--cgf-lockstep-shot` (showcases).
   **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp` (`git diff master
   -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None.** PURE CPU — no compute, no dispatch. `rhi.h` + backend dirs UNCHANGED. `engine/sim/grain.h` +
  `fluid.h` + `fpx.h` + `cloth.h` + `couple.h` + `couple_grain.h` + `engine/physics/` UNCHANGED. GF1-GF4 cgf code
  + shaders UNCHANGED (GF5 additive — only the harness + the showcase). **NO new shader.** Report the seam empty.

## Out of scope (YAGNI — later GF slice)
The lit 3D render capstone (GF6 — GF5's render reuses the GF4 `cgf_step` path as-is). Network transport,
delta-compression, input prediction models (the harness proves bit-exact replay + rollback; the transport layer
is the existing net stack's job, out of scope here). GF5 claims ONLY: a deterministic lockstep + rollback over
the coupled grain+fluid world, bit-identical BOTH pools across two runs and (by `StepCGF`'s proven bit-exactness)
across platforms, with the integer golden + the four proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 99) + the new `cgf_test` lockstep cases. Clean under
   `windows-msvc-asan` (build+run `cgf_test` + `introspect_test`).
2. **proofs + visual:** `--cgf-lockstep-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the rendered
   final state shows the coherent wet-sand scene (the GF4 render — pixel-check; the GF3/GF4 lesson).**
3. Metal: `visual_test --cgf-lockstep` → new golden `tests/golden/metal/cgf_lockstep.png`; two runs DIFF 0.0000
   (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader was added (GF5
   is pure CPU — `hf_gen_msl` UNCHANGED).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgf_lockstep.png` added; the
   other 151 byte-identical. `git diff master --stat -- tests/golden` = ONLY `cgf_lockstep.png` (metal) + the
   introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cgf-lockstep` + `--cgf-lockstep-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/grain.h` + `fluid.h` + `fpx.h` + `cloth.h` + `couple.h` +
   `couple_grain.h` + `engine/physics/` + GF1-GF4 cgf code/shaders byte-unchanged). `scripts/verify.ps1` updated:
   `cgf_lockstep` golden in the Mac loop + `--cgf-lockstep-shot` in `$vkShots`. **NO new entry in `hf_gen_msl`.**
