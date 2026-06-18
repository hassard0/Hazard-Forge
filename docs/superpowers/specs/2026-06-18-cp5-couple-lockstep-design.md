# Slice CP5 — Deterministic Rigid↔Fluid Coupling: LOCKSTEP + ROLLBACK (the multi-body netcode HEADLINE) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #11 (DETERMINISTIC
> TWO-WAY RIGID↔FLUID COUPLING, `hf::sim::couple`). Proves the bit-exact CP4 coupled step is true cross-platform
> **LOCKSTEP + ROLLBACK** — and it is the arc's FIRST **MULTI-BODY** lockstep: a peer fed the input command
> stream ALONE re-derives the authority's exact COUPLED state — BOTH the rigid bodies AND the fluid — bit-for-
> bit, and a mispredicted input is corrected by rolling back to a snapshot + re-simulating. **PURE CPU, 0
> backend symbols, NO new shader / RHI** — a determinism PROPERTY of the existing bit-exact `StepCouple`. The
> direct composition of the FPX5 (rigid) + FL5 (fluid) harnesses over `CoupleWorld`. The trilogy's netcode
> story now spans a COUPLED system: shove the barrel, and two peers re-simulate the bob AND the splash bit-
> for-bit. Branch: `slice-cp5`. See [[hazard-forge-couple-roadmap]].

**Goal:** Extend `engine/sim/couple.h` (additive — CP1–CP4 byte-unchanged) with the lockstep/rollback harness:
`CoupleCommand` + `ApplyCoupleCommand` + `SimCoupleTick` + `SnapshotCouple`/`RestoreCouple` (the FIRST
multi-body snapshot — bodies AND fluid) + `RunCoupleLockstep` + `RunCoupleRollback`. Add `--couple-lockstep-shot`
(Vulkan) / `--couple-lockstep` (Metal) — BOTH run the IDENTICAL CPU harness. Bake the integer golden
`couple_lockstep` (the converged coupled state). NO new shader, NO new RHI.

## Design call: a determinism PROPERTY (pure CPU) — strict zero-diff cross-platform
Lockstep is the proof that the bit-exact `StepCouple` (CP4, itself bit-identical Vulkan/Metal) is replayable
from inputs alone. Both `--couple-lockstep-shot` (Vulkan side) and `--couple-lockstep` (Metal side) run the
SAME pure-CPU harness → the converged coupled state (bodies + fluid) is byte-identical on Vulkan-Windows AND
Metal-Mac, which IS the cross-platform-lockstep evidence. Bar: strict INTEGER (zero-differing-pixel, the
FL5/GR5 bar). NO `<cmath>`, NO RNG, NO clock.

**The MULTI-BODY twist (the new thing vs FL5/GR5):** the world has TWO heterogeneous body sets — the rigid
`std::vector<fpx::FxBody> bodies` AND the `std::vector<fluid::FluidParticle> particles`. `SnapshotCouple` must
deep-copy BOTH; `RunCoupleLockstep`'s `replica == authority` must memcmp BOTH; a `CoupleCommand` can target a
rigid body OR a fluid particle. This is the first lockstep over a *coupled multi-material* system — strictly
more than FL5 (fluid alone) or FPX5 (rigid alone).

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FL5 / GR5 lockstep harness to MIRROR (`engine/sim/fluid.h:817-925`, `engine/sim/grain.h` GR5):**
  `FluidCommand`/`ApplyFluidCommand`/`SimFluidTick`/`SnapshotFluid`/`RestoreFluid`/`RunFluidLockstep`/
  `RunFluidRollback` — the SAME shape over `CoupleWorld` + `StepCouple`. `SnapshotCouple` returns a
  `CoupleSnapshot{ std::vector<fpx::FxBody> bodies; std::vector<fluid::FluidParticle> particles; }` (deep-copy
  BOTH vectors); `RestoreCouple` restores both. The command stream is processed in ARRAY ORDER per tick.
- **The CP4 step `SimCoupleTick` wraps (this branch's `engine/sim/couple.h`):** `StepCouple(world, dt, iters)`
  — the bit-exact coupled tick. `CoupleWorld` (bodies + particles + config). DO NOT modify CP1–CP4 functions —
  CP5 is additive.
- **The showcase + golden discipline:** FL5's `--fluid-lockstep-shot` / GR5's `--grain-lockstep-shot` (BOTH
  run the identical CPU harness, color the converged state to a side view) — mirror for the couple pair.
  `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect
  JSON golden** — the GR2 lesson), `tests/couple_test.cpp`.

## Design decisions (locked)
1. **`CoupleCommand` + `ApplyCoupleCommand` (the multi-body input event).** `CoupleCommand{tick, kind, target,
   arg}`; kinds: `kCmdBodyShove` (add `arg` to rigid body `target`'s velocity — the "shove the barrel"
   headline), `kCmdBodyMove` (add `arg` to body `target`'s position), `kCmdFluidWind` (add `arg` to fluid
   particle `target`'s velocity). Integer adds; out-of-range target / unknown kind → no-op; static bodies/
   particles never mutated. The deterministic per-tick INPUT (NOT full state). A `std::vector<CoupleCommand>`
   is the stream, processed in ARRAY ORDER per tick.
2. **`SimCoupleTick`.** Apply ALL stream commands with `.tick == tick` in array order, THEN `StepCouple` one
   step. Pure integer, fixed order → bit-identical on every peer/platform.
3. **`SnapshotCouple`/`RestoreCouple` (the FIRST multi-body snapshot).** Deep copy / restore of BOTH the
   `bodies` AND the `particles` vectors (a `CoupleSnapshot` struct); bit-exact round-trip.
4. **`RunCoupleLockstep`** (ticks from a copy of init + the stream → the final coupled state) **+
   `RunCoupleRollback`** (snapshot at mispredictTick → speculate with the WRONG input → restore + re-sim the
   CORRECT authStream → the corrected state). The FL5/GR5 twins, over `CoupleWorld`.
5. **Showcase `--couple-lockstep-shot <out>` (Vulkan) AND `--couple-lockstep` (Metal) — WIRE BOTH, IDENTICAL
   CPU harness.** The CP4 coupled scene (a basin pool + a body) + a command stream that SHOVES the body
   (`kCmdBodyShove`) at chosen ticks so it bobs/moves to a NON-TRIVIAL converged state (the FL4/FL5 lesson —
   a stream that visibly displaces the body AND perturbs the fluid). Color the converged coupled state
   (bodies + fluid) to a side view. Golden = `tests/golden/metal/couple_lockstep.png` (Mac-baked by the
   CONTROLLER — DO NOT commit).
6. **PROOFS (fail loudly; exact lines — the FL5/GR5 set, multi-body):**
   - **(1) lockstep (inputs-only, BOTH bodies+fluid):** `RunCoupleLockstep` authority == replica byte-for-byte
     across BOTH the bodies AND the fluid (a peer fed the stream ALONE re-derives the exact coupled state).
     Print `couple-lockstep: replica==authority {bodies:<B>, particles:<N>} BIT-EXACT (<T> ticks, inputs-only)`.
   - **(2) rollback:** `RunCoupleRollback` corrected == the lockstep authority byte-for-byte (bodies+fluid),
     AND the mispredicted-before-rollback state DIFFERED. Print `couple-lockstep rollback: corrected to
     authority BIT-EXACT (mispredict@tick<M> diverged then converged)`.
   - **(3) determinism + snapshot round-trip:** two lockstep runs identical; `RestoreCouple(SnapshotCouple(w))`
     == w (bodies+fluid). Print `couple-lockstep determinism: two runs BYTE-IDENTICAL + snapshot round-trip exact`.
   - **(4) motion (non-trivial):** the command stream displaced the body (and perturbed the fluid) by a
     non-trivial amount. Print `couple-lockstep motion: shoved the body by <D>`.
   - **Golden discipline: ONLY `tests/golden/metal/couple_lockstep.png`; do NOT commit it.** Existing 139
     image goldens UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict):** the Vulkan-Windows converged state == the Metal-Mac converged
   state == the golden, ZERO differing pixels (the same pure-CPU harness on both → bit-identical).
8. **Tests `tests/couple_test.cpp` additions (pure CPU):** `ApplyCoupleCommand` (body-shove adds to body vel,
   fluid-wind adds to particle vel, body-move adds to body pos, static unmoved, out-of-range no-op);
   `SnapshotCouple`/`RestoreCouple` round-trip (bodies AND fluid); `RunCoupleLockstep` two runs identical;
   `RunCoupleRollback` corrected == authority AND mispredicted ≠ authority. Clean under `windows-msvc-asan`.
9. **Introspect.** Add exactly `deterministic-couple-lockstep` (features) + `--couple-lockstep-shot`
   (showcases). **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`
   (the GR2 lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None — PURE CPU.** NO new shader, NO new RHI, NO GPU pass. `rhi.h` + `rhi_factory` + backend dirs + ALL
  shaders UNCHANGED. `engine/sim/fpx.h` + `fluid.h` + `cloth.h` + `grain.h` + `engine/physics/` UNCHANGED.
  CP1–CP4 couple code + shaders UNCHANGED (CP5 appends the CPU harness). Report the seam is empty (incl. shaders).

## Out of scope (YAGNI — last CP slice)
The lit 3D render (CP6). Delta-compressed snapshots, partial-state sync, input delay/buffering, real transport,
body-body lockstep beyond the single coupled body. CP5 claims ONLY: the bit-exact coupled sim is lockstep +
rollback replayable from inputs alone — across BOTH the rigid bodies AND the fluid — bit-identical
CPU↔Vulkan↔Metal, with the converged-state integer golden + the four FL5/GR5 proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 97) + the new `couple_test` lockstep cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--couple-lockstep-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows a coherent converged coupled state (body + fluid) visibly shoved by the stream (pixel-check;
   the NAV6/CL6 lesson).**
3. Metal: `visual_test --couple-lockstep` → new golden `tests/golden/metal/couple_lockstep.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (the
   harness is pure CPU).** Cross-vendor Vulkan-vs-Metal STRICT ZERO (the same CPU harness on both).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `couple_lockstep.png` added;
   the other 139 byte-identical (re-run `--couple-query/buoyancy/displace/step-shot` → still bit-exact). `git
   diff master --stat -- tests/golden` = ONLY `couple_lockstep.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-couple-lockstep` + `--couple-lockstep-shot`; introspect
   test updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` + ALL shaders UNCHANGED — PURE CPU, no new shader/RHI; `engine/sim/fpx.h` +
   `fluid.h` + `cloth.h` + `grain.h` + `engine/physics/` + CP1–CP4 couple code/shaders byte-unchanged).
   `scripts/verify.ps1` updated: `couple_lockstep` golden in the Mac loop + `--couple-lockstep-shot` in
   `$vkShots`.
