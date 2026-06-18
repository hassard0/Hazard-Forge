# Slice FL5 — Deterministic GPU Fluid: LOCKSTEP + ROLLBACK (the beyond-UE5 headline) (Phase 14 #5) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #9
> (DETERMINISTIC GPU FLUID via Position-Based Fluids, `hf::sim::fluid`, header `engine/sim/fluid.h`). The
> HEADLINE: prove the fixed-point fluid is true cross-platform LOCKSTEP + ROLLBACK — a peer fed the INPUT
> command stream alone re-derives the authority's exact fluid state bit-for-bit, and a mispredicted input
> is corrected by rollback-to-snapshot + re-sim. This extends the FPX5/CL5 lockstep determinism headline
> from rigid bodies and cloth to FLUID — a deterministic, rollback-able, bit-identical-cross-platform
> fluid that UE5's float Niagara categorically cannot do. PURE CPU harness over the bit-exact FL1–FL4
> fluid — NO new shader, NO new RHI. ZERO new RHI. Branch: `slice-fl5`. See [[hazard-forge-fluid-roadmap]].

**Goal:** Extend `engine/sim/fluid.h` with the netcode harness (the direct analog of `fpx.h`'s and
`cloth.h`'s `*Command`/`*Tick`/`Snapshot*`/`Restore*`/`Run*Lockstep`/`Run*Rollback`): `FluidCommand{tick,
kind, target, FxVec3 arg}` (e.g. a wind/emit impulse to a region of particles) + `ApplyFluidCommand`,
`SimFluidTick` (apply the tick's commands in deterministic order → `StepFluid`), `SnapshotFluid`/
`RestoreFluid` (deep-copy the integer particle array — the rollback primitive), `RunFluidLockstep(init,
stream, ticks)` (the peer entry point), `RunFluidRollback(init, authStream, mispredictStream, ticks,
mispredictTick)` (run to mispredictTick saving a snapshot, advance with the WRONG input, then
`RestoreFluid` + re-sim with the CORRECT input). Add `--fluid-lockstep-shot` (Vulkan) / `--fluid-lockstep`
(Metal) — BOTH run the IDENTICAL CPU harness (NO GPU dispatch). The `fluid_lockstep` integer golden (the
converged fluid state → strict ZERO-DIFFERING-PIXEL cross-backend), and `tests/fluid_test.cpp` additions.
Reuse FL1–FL4 verbatim (StepFluid) — FL5 is additive (FL1–FL4 pipelines + goldens stay byte-identical;
pure-CPU, NO new shader/RHI).

## Design call: PURE CPU, the FPX5/CL5 twin (no new shader/RHI)
Lockstep/rollback is a CPU harness over the bit-exact fluid step — exactly like FPX5 and CL5 (both added
NO shader and NO RHI). The fluid step (`StepFluid`) is already proven bit-exact CPU↔Vulkan↔Metal (FL1–FL4),
so running it from a command stream is deterministic by construction. `--fluid-lockstep-shot` (Vulkan) and
`--fluid-lockstep` (Metal) run the IDENTICAL CPU harness (no GPU dispatch on either) → the golden is the
CPU-colored converged-state read-back, byte-identical cross-backend. **THE HEADLINE made concrete: same
inputs → byte-identical fluid state on Vulkan-Windows AND Metal-Mac (a zero-differing-pixel cross-platform
compare), which UE5's float Niagara cannot guarantee — completing the deterministic-sim trilogy's
lockstep story (rigid `fpx` + cloth + fluid).** Use a scene with enough ticks + a VISIBLE command (a wind
impulse that demonstrably moves the fluid) so the converged state is NON-TRIVIAL (the FL4 lesson: a static
near-resting block is a weak demo — give the fluid something to do).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The netcode harness to MIRROR (read it carefully):** `engine/sim/cloth.h` — `ClothCommand`/
  `ApplyClothCommand`/`SimClothTick`/`SnapshotCloth`/`RestoreCloth`/`RunClothLockstep`/`RunClothRollback`
  (the CL5 harness, itself the `fpx.h:531-601` twin). FL5's harness is the SAME shape over `FluidParticle`/
  `StepFluid`. The CL5 `RunClothRollback` (advance to mispredictTick saving a snapshot, speculate with the
  WRONG stream, `RestoreCloth` + re-sim the authoritative stream) is the EXACT structural template.
- **The bit-exact fluid step to drive:** `engine/sim/fluid.h::StepFluid` (FL4 — the full PBF loop).
  `SimFluidTick` applies the tick's commands then calls `StepFluid`.
- **The pure-CPU showcase template:** CL5's `--cloth-lockstep-shot` / `RunClothLockstepShowcase` (BOTH
  backends run the IDENTICAL CPU harness, no GPU dispatch; the 5 proof lines; the converged-state golden).
  Mirror it for fluid.
- **RHI compute envelope:** NONE used (pure CPU). `rhi.h` UNCHANGED.
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--fluid-lockstep-shot` standalone arg
  branch — main.cpp has `/bigobj`; NO new shader so no DXC entry), `metal_headless/{visual_test.mm,
  CMakeLists.txt}` (`RunFluidLockstepShowcase` + `--fluid-lockstep` — pure CPU, NO `hf_gen_msl` entry),
  `engine/editor/introspect.cpp` (+`deterministic-fluid-lockstep` feature + `--fluid-lockstep-shot`
  showcase) + `tests/introspect_test.cpp` + the JSON golden, `scripts/verify.ps1` (`fluid_lockstep`
  golden in the Mac loop + `--fluid-lockstep-shot` in `$vkShots`).

## Design decisions (locked)
1. **The command model.** `FluidCommand{ uint32 tick; uint32 kind; uint32 target; FxVec3 arg; }` (kind:
   e.g. `kCmdWind` = add a wind impulse to all/region particles' velocity; `kCmdEmit`/`kCmdPush` = push a
   region). `ApplyFluidCommand(world, cmd)` mutates the fluid state deterministically. Commands at the same
   tick applied in a FIXED order (ascending index). Keep the set small (a wind/push impulse is enough to
   prove lockstep/rollback AND give the fluid visible motion).
2. **The harness (the FPX5/CL5 twin).** `SimFluidTick(world, cmds_for_tick, params)`: apply the tick's
   commands (fixed order) → `StepFluid` (one step). `SnapshotFluid(world)` = deep copy the particle array;
   `RestoreFluid(world, snap)` = overwrite. `RunFluidLockstep(init, stream, ticks)`: run `ticks`
   `SimFluidTick`s feeding the per-tick commands → the final fluid state. `RunFluidRollback(init,
   authStream, mispredictStream, ticks, mispredictTick)`: run to `mispredictTick` taking a snapshot,
   advance with the MISPREDICTED command, then `RestoreFluid` + re-sim from the snapshot with the CORRECT
   command to `ticks` → must equal the pure-authority run; the pre-rollback mispredicted state must DIFFER
   (the FPX5/CL5 positive+negative control).
3. **Showcase `--fluid-lockstep-shot <out>` (Vulkan, main.cpp) AND `--fluid-lockstep` (Metal,
   visual_test.mm — WIRE BOTH; BOTH run the IDENTICAL CPU harness, NO GPU dispatch; confirm visual_test.mm
   + `#include "sim/fluid.h"`).** A deterministic scene: a fluid block (e.g. a few hundred particles), an
   authStream of a few wind/push commands over K ticks (enough to move the fluid VISIBLY), a
   mispredictStream (a wrong push at some tick), a mispredictTick. Run the lockstep replica (inputs-only) +
   the rollback. CPU-color the converged fluid state → `tests/golden/metal/fluid_lockstep.png` (baked on
   the Mac by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines — the 5 FPX5/CL5-style proofs):**
   - **(1) lockstep (the headline):** the replica fed INPUTS ONLY re-derives the authority's EXACT fluid
     state. Print `fluid-lockstep: replica==authority <N> particles BIT-EXACT (<K> ticks, inputs-only)`.
   - **(2) rollback:** rollback-to-snapshot + re-sim with the corrected input == authority BIT-EXACT, AND
     the pre-rollback mispredicted state DIFFERED. Print `fluid-lockstep rollback: corrected to authority
     BIT-EXACT (mispredict@tick<m> diverged then converged)`.
   - **(3) determinism:** two runs BYTE-IDENTICAL. Print `fluid-lockstep determinism: two runs
     BYTE-IDENTICAL`.
   - **(4) snapshot:** `SnapshotFluid` → `RestoreFluid` round-trip == original BIT-EXACT. Print
     `fluid-lockstep snapshot: round-trip BIT-EXACT`.
   - **(5) stats:** `fluid-lockstep: {particles:<N>, ticks:<K>, commands:<C>, mispredict-tick:<m>}`.
   - **Golden discipline: ONLY `tests/golden/metal/fluid_lockstep.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 127 image goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict).** BOTH backends run the IDENTICAL CPU harness over the bit-exact
   fluid → the converged state is byte-identical; the golden is the CPU-colored integer read-back; the
   controller's cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare (the cross-platform fluid
   lockstep proof made concrete — Vulkan-Windows converged == Metal-Mac converged). Any nonzero
   cross-backend diff is a real bug.
6. **Tests `tests/fluid_test.cpp` additions (pure CPU):** `ApplyFluidCommand` (wind changes velocity;
   OOB target no-op); `SnapshotFluid`/`RestoreFluid` round-trip == original; `SimFluidTick` determinism +
   deterministic command order; `RunFluidLockstep` replica==authority; `RunFluidRollback` positive
   (converges to authority) + negative (mispredicted state differs) control. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-fluid-lockstep` (features) + `--fluid-lockstep-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None — pure CPU.** NO new shader, NO new RHI (a CPU harness over the existing bit-exact fluid step).
  `rhi.h` + `rhi_factory` (baseline 2) + backend dirs + ALL fluid shaders + `engine/sim/fpx.h` +
  `engine/sim/cloth.h` + `engine/physics/` UNCHANGED. Report the seam.

## Out of scope (YAGNI — flagship's last slice is FL6)
The float lit-3D render (FL6). A real network transport (FL5 proves the determinism that MAKES lockstep
possible — the wire is the existing net layer's concern). FL5 is ONLY the lockstep/rollback CPU harness +
its bit-exact golden + the 5 proofs. No float, no new shader/RHI.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 95) + the new `fluid_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--fluid-lockstep-shot` on Vulkan: the 5 proofs (lockstep replica==authority,
   rollback positive+negative, determinism, snapshot round-trip, stats); the converged fluid state image.
   Run under the Vulkan-validation gate → ZERO VUID (set BOTH `VK_LAYER_PATH` to the conan
   `...\.conan2\p\...\layers` dir AND `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer
   LOADED = zero "not found"). (Pure CPU — no GPU dispatch.)
3. Metal: `visual_test --fluid-lockstep` → new golden `tests/golden/metal/fluid_lockstep.png`; two runs
   DIFF 0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm NO new shader
   (NO `hf_gen_msl` entry) — pure CPU both backends.** Cross-backend = STRICT ZERO-DIFFERING-PIXEL
   (Vulkan-Windows converged == Metal-Mac converged — the cross-platform fluid lockstep proof), NOT the
   float baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fluid_lockstep.png` added;
   the other 127 byte-identical (FL1–FL4 + all existing untouched). `git diff master --stat -- tests/golden`
   = ONLY `fluid_lockstep.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fluid-lockstep` + `--fluid-lockstep-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` + ALL shaders UNCHANGED — pure CPU, NO new shader/RHI). `scripts/verify.ps1`
   updated: `fluid_lockstep` golden in the Mac loop + `--fluid-lockstep-shot` in `$vkShots`. FL1–FL4 +
   `engine/sim/fpx.h` + `engine/sim/cloth.h` + `engine/physics/` UNTOUCHED.
