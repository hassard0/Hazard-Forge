# Slice CL5 — Deterministic GPU Cloth: LOCKSTEP + ROLLBACK (the beyond-UE5 headline) (Phase 13 #5) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #8
> (DETERMINISTIC GPU CLOTH, `hf::sim::cloth`, header `engine/sim/cloth.h`). The HEADLINE: prove the
> fixed-point cloth is true cross-platform LOCKSTEP + ROLLBACK — a peer fed the INPUT command stream
> alone re-derives the authority's exact cloth state bit-for-bit, and a mispredicted input is corrected
> by rollback-to-snapshot + re-sim. This extends the FPX5 lockstep/rollback determinism headline from
> rigid to DEFORMABLE bodies — a deterministic, rollback-able, bit-identical-cross-platform cloth that
> UE5's float Chaos Cloth/NvCloth cannot provide. PURE CPU harness over the bit-exact CL1–CL4 cloth — NO
> new shader, NO new RHI. ZERO new RHI. Branch: `slice-cl5`. See [[hazard-forge-cloth-roadmap]].

**Goal:** Extend `engine/sim/cloth.h` with the netcode harness (the direct analog of `fpx.h`'s
`FxCommand`/`ApplyCommand`/`SimTick`/`SnapshotWorld`/`RestoreWorld`/`RunLockstep`/`RunRollback`, ~`fpx.h:
531-601`): `ClothCommand{tick, kind, particle/region, arg}` (e.g. wind impulse, pin/unpin) +
`ApplyClothCommand`, `SimClothTick` (apply the tick's commands in deterministic order → `StepClothCollide`),
`SnapshotCloth`/`RestoreCloth` (deep-copy the integer particle array — the rollback primitive),
`RunClothLockstep(init, stream, ticks)` (the peer entry point), `RunClothRollback(init, authStream,
mispredictStream, ticks, mispredictTick)` (run to mispredictTick saving a snapshot, advance with the WRONG
input, then `RestoreCloth` + re-sim with the CORRECT input). Add `--cloth-lockstep-shot` (Vulkan) /
`--cloth-lockstep` (Metal) — BOTH run the IDENTICAL CPU harness (NO GPU dispatch). The `cloth_lockstep`
integer golden (the converged cloth state → strict ZERO-DIFFERING-PIXEL cross-backend), and
`tests/cloth_test.cpp` additions. Reuse CL1–CL4 verbatim (StepCloth/StepClothCollide) — CL5 is additive
(CL1–CL4 pipelines + goldens stay byte-identical; pure-CPU, NO new shader/RHI).

## Design call: PURE CPU, the FPX5 twin (no new shader/RHI)
Lockstep/rollback is a CPU harness over the bit-exact cloth step — exactly like FPX5 (which added NO shader
and NO RHI). The cloth step (`StepClothCollide`) is already proven bit-exact CPU↔Vulkan↔Metal (CL1–CL4), so
running it from a command stream is deterministic by construction. `--cloth-lockstep-shot` (Vulkan) and
`--cloth-lockstep` (Metal) run the IDENTICAL CPU harness (no GPU dispatch on either) → the golden is the
CPU-colored converged-state read-back, byte-identical cross-backend. **THE HEADLINE made concrete: same
inputs → byte-identical cloth state on Vulkan-Windows AND Metal-Mac (a zero-differing-pixel cross-platform
compare), which UE5's float Chaos Cloth cannot guarantee.** This composes the cloth flagship with the FPX5
netcode story (the engine's existing differentiator) — now for deformable bodies.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The netcode harness to MIRROR (read it carefully):** `engine/sim/fpx.h` — `FxCommand{tick,kind,bodyId,
  arg}` + `kCmdImpulse`/`kCmdSetAngVel`; `ApplyCommand`; `SimTick` (apply the tick's commands in
  deterministic order → BuildPairs → StepWorld → IntegrateOrientation); `SnapshotWorld`/`RestoreWorld`
  (deep copy = the rollback primitive); `RunLockstep(init, stream, ticks)`; `RunRollback(init, authStream,
  mispredictStream, ticks, mispredictTick)` (~`fpx.h:531-601`). CL5's harness is the SAME shape over
  `ClothParticle`/`StepClothCollide`.
- **The bit-exact cloth step to drive:** `engine/sim/cloth.h::StepClothCollide` (CL4 — integrate + K
  constraint passes + collision). `SimClothTick` applies the tick's commands then calls `StepClothCollide`.
- **The pure-CPU showcase template:** FPX5's `--fpx-lockstep-shot` / `RunFpxLockstepShowcase` (BOTH
  backends run the IDENTICAL CPU harness, no GPU dispatch; the proof lines; the converged-state golden).
  This is the EXACT structural template — mirror it for cloth.
- **The integer-golden discipline:** CL3/CL4 (ReadBuffer-free here — it's CPU; CPU-color the converged
  particle array directly, strict zero-diff). FPX5's `fpx_lockstep` golden.
- **RHI compute envelope:** NONE used (pure CPU). `rhi.h` UNCHANGED.
- **Wiring:** `samples/hello_triangle/{main.cpp,CMakeLists.txt}` (`--cloth-lockstep-shot` standalone arg
  branch — note main.cpp now needs `/bigobj`, already added in CL4 — NO new shader so no DXC entry),
  `metal_headless/{visual_test.mm,CMakeLists.txt}` (`RunClothLockstepShowcase` + `--cloth-lockstep` —
  pure CPU, NO `hf_gen_msl` entry), `engine/editor/introspect.cpp` (+`deterministic-cloth-lockstep`
  feature + `--cloth-lockstep-shot` showcase) + `tests/introspect_test.cpp` + the JSON golden,
  `scripts/verify.ps1` (`cloth_lockstep` golden in the Mac loop + `--cloth-lockstep-shot` in `$vkShots`).

## Design decisions (locked)
1. **The command model.** `ClothCommand{ uint32 tick; uint32 kind; uint32 target; FxVec3 arg; }` (kind:
   e.g. `kCmdWind` = apply a wind impulse to all/region particles' velocity; `kCmdUnpin`/`kCmdPin` = toggle
   a particle's PINNED flag). `ApplyClothCommand(world, cmd)` mutates the cloth state deterministically.
   Commands at the same tick applied in a FIXED order (ascending index). Keep the set small (wind + pin
   toggle is enough to prove lockstep/rollback).
2. **The harness (the FPX5 twin).** `SimClothTick(world, cmds_for_tick, params)`: apply the tick's commands
   (fixed order) → `StepClothCollide` (one step). `SnapshotCloth(world)` = deep copy the particle array +
   any sim state; `RestoreCloth(world, snap)` = overwrite. `RunClothLockstep(init, stream, ticks)`: run
   `ticks` `SimClothTick`s feeding the per-tick commands → the final cloth state. `RunClothRollback(init,
   authStream, mispredictStream, ticks, mispredictTick)`: run to `mispredictTick` taking a snapshot, advance
   with the MISPREDICTED command, then `RestoreCloth` + re-sim from the snapshot with the CORRECT command to
   `ticks` → must equal the pure-authority run; the pre-rollback mispredicted state must DIFFER (the FPX5
   positive+negative control). Keep ticks/iters under any practical bound (pure CPU, no TDR — but keep it
   quick).
3. **Showcase `--cloth-lockstep-shot <out>` (Vulkan, main.cpp) AND `--cloth-lockstep` (Metal,
   visual_test.mm — WIRE BOTH; BOTH run the IDENTICAL CPU harness, NO GPU dispatch; confirm visual_test.mm
   + `#include "sim/cloth.h"`).** A deterministic scene: a small cloth (e.g. 16×16, top corners pinned), an
   authStream of a few wind/pin commands over K ticks, a mispredictStream (a wrong wind at some tick), a
   mispredictTick. Run the lockstep replica (inputs-only) + the rollback. CPU-color the converged cloth
   state → `tests/golden/metal/cloth_lockstep.png` (baked on the Mac by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines — the 5 FPX5-style proofs):**
   - **(1) lockstep (the headline):** the replica fed INPUTS ONLY (the command stream, NOT full state)
     re-derives the authority's EXACT cloth state. Print `cloth-lockstep: replica==authority <N> particles
     BIT-EXACT (<K> ticks, inputs-only)`.
   - **(2) rollback:** rollback-to-snapshot + re-sim with the corrected input == authority BIT-EXACT, AND
     the pre-rollback mispredicted state DIFFERED (positive + negative control). Print `cloth-lockstep
     rollback: corrected to authority BIT-EXACT (mispredict@tick<m> diverged then converged)`.
   - **(3) determinism:** two runs BYTE-IDENTICAL. Print `cloth-lockstep determinism: two runs
     BYTE-IDENTICAL`.
   - **(4) snapshot:** `SnapshotCloth` → `RestoreCloth` round-trip == original BIT-EXACT. Print
     `cloth-lockstep snapshot: round-trip BIT-EXACT`.
   - **(5) stats:** `cloth-lockstep: {particles:<N>, ticks:<K>, commands:<C>, mispredict-tick:<m>}`.
   - **Golden discipline: ONLY `tests/golden/metal/cloth_lockstep.png`; do NOT commit it — the CONTROLLER
     bakes on the Mac.** Existing 121 image goldens UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict).** BOTH backends run the IDENTICAL CPU harness over the bit-exact
   cloth → the converged state is byte-identical; the golden is the CPU-colored integer read-back; the
   controller's cross-backend check is the STRICT ZERO-DIFFERING-PIXEL compare (the cross-platform lockstep
   proof made concrete — Vulkan-Windows converged == Metal-Mac converged). Any nonzero cross-backend diff
   is a real bug.
6. **Tests `tests/cloth_test.cpp` additions (pure CPU):** `ApplyClothCommand` (wind changes velocity;
   pin/unpin toggles the flag; OOB target no-op); `SnapshotCloth`/`RestoreCloth` round-trip == original;
   `SimClothTick` determinism + deterministic command order; `RunClothLockstep` replica==authority;
   `RunClothRollback` positive (converges to authority) + negative (mispredicted state differs) control.
   Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-cloth-lockstep` (features) + `--cloth-lockstep-shot`
   (showcases). Rebake the JSON golden; update `introspect_test.cpp`.

## RHI seam additions (summary)
- **None — pure CPU.** NO new shader, NO new RHI (a CPU harness over the existing bit-exact cloth step).
  `rhi.h` + `rhi_factory` (baseline 2) + backend dirs + ALL cloth shaders + `engine/sim/fpx.h` +
  `engine/physics/` UNCHANGED. Report the seam.

## Out of scope (YAGNI — flagship's last slice is CL6)
The float lit-3D render (CL6). A real network transport (CL5 proves the determinism that MAKES lockstep
possible — the wire is the existing net layer's concern, as FPX5 framed it). Self-collision, dynamic
colliders, friction (out of the whole flagship). CL5 is ONLY the lockstep/rollback CPU harness + its
bit-exact golden + the 5 proofs. No float, no new shader/RHI.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 94) + the new `cloth_test` cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--cloth-lockstep-shot` on Vulkan: the 5 proofs (lockstep replica==authority,
   rollback positive+negative, determinism, snapshot round-trip, stats); the converged cloth state image.
   Run under the Vulkan-validation gate → ZERO VUID in the OUTPUT (set BOTH `VK_LAYER_PATH` to the conan
   `...\.conan2\p\...\layers` dir AND `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; confirm the layer
   LOADED = zero "not found"). (Pure CPU — no GPU dispatch, so no TDR concern.)
3. Metal: `visual_test --cloth-lockstep` → new golden `tests/golden/metal/cloth_lockstep.png`; two runs
   DIFF 0.0000 (gate on compare.sh EXIT CODE). **Confirm visual_test.mm in the diff; confirm NO new shader
   (NO `hf_gen_msl` entry) — pure CPU both backends.** Cross-backend = STRICT ZERO-DIFFERING-PIXEL
   (Vulkan-Windows converged == Metal-Mac converged — the cross-platform lockstep proof), NOT the float
   baseline.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cloth_lockstep.png` added;
   the other 121 byte-identical (CL1–CL4 + all existing untouched). `git diff master --stat --
   tests/golden` = ONLY `cloth_lockstep.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cloth-lockstep` + `--cloth-lockstep-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` + ALL shaders UNCHANGED — pure CPU, NO new shader/RHI). `scripts/verify.ps1`
   updated: `cloth_lockstep` golden in the Mac loop + `--cloth-lockstep-shot` in `$vkShots`. CL1–CL4 +
   `engine/sim/fpx.h` + `engine/physics/` UNTOUCHED.
