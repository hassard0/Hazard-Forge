# Slice CG5 — Deterministic Rigid↔Grain Coupling: LOCKSTEP + ROLLBACK (the multi-body netcode HEADLINE) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #12 (DETERMINISTIC
> TWO-WAY RIGID↔GRAIN COUPLING, `hf::sim::cgrain`). Proves the bit-exact CG4 coupled step is true cross-platform
> **LOCKSTEP + ROLLBACK** — a MULTI-BODY lockstep over a COUPLED rigid+granular system: a peer fed the input
> command stream ALONE re-derives the authority's exact coupled state — BOTH the rigid bodies AND the sand —
> bit-for-bit, and a mispredicted input is corrected by rolling back to a snapshot + re-simulating. **PURE CPU,
> 0 backend symbols, NO new shader / RHI** — a determinism PROPERTY of the existing bit-exact `StepCGrain`. The
> direct composition of the CP5 (coupled) + GR5 (grain) harnesses over `CGrainWorld`. Shove the body, and two
> peers re-simulate the sink AND the sand-spray bit-for-bit. Branch: `slice-cg5`. See [[hazard-forge-couple-grain-roadmap]].

**Goal:** Extend `engine/sim/couple_grain.h` (additive — CG1–CG4 byte-unchanged) with the lockstep/rollback
harness: `CGrainCommand` + `ApplyCGrainCommand` + `SimCGrainTick` + `SnapshotCGrain`/`RestoreCGrain` (the
multi-body snapshot — bodies AND grains) + `RunCGrainLockstep` + `RunCGrainRollback`. Add `--cgrain-lockstep-shot`
(Vulkan) / `--cgrain-lockstep` (Metal) — BOTH run the IDENTICAL CPU harness. Bake the integer golden
`cgrain_lockstep` (the converged coupled state). NO new shader, NO new RHI.

## Design call: a determinism PROPERTY (pure CPU) — strict zero-diff cross-platform — and KEEP IT FAST
Lockstep is the proof that the bit-exact `StepCGrain` (CG4, itself bit-identical Vulkan/Metal) is replayable
from inputs alone. Both `--cgrain-lockstep-shot` (Vulkan side) and `--cgrain-lockstep` (Metal side) run the
SAME pure-CPU harness → the converged coupled state (bodies + grains) is byte-identical on Vulkan-Windows AND
Metal-Mac, the cross-platform-lockstep evidence. Bar: strict INTEGER (zero-differing-pixel, the CP5/GR5 bar).
NO `<cmath>`, NO RNG, NO clock. **PERFORMANCE NOTE (critical): `StepCGrain` is HEAVY (the CG4 sim ran ~79s
for 2925 grains × 300 steps). The lockstep harness runs the sim MULTIPLE times (authority + replica +
rollback speculation + re-sim), so use a SMALL/FAST scene — a MODEST grain bed (a few hundred grains) and a
MODEST tick count (≈30–60 ticks, NOT 300) — so the whole `--cgrain-lockstep-shot` (which runs 3–4 sim
sequences) finishes in a couple of minutes, not an hour. The lockstep PROOF needs determinism + a non-trivial
converged state, NOT a long settle.**

## The MULTI-BODY twist (the CP5 lesson)
The world has TWO heterogeneous body sets — the rigid `std::vector<fpx::FxBody> bodies` AND the
`std::vector<grain::GrainParticle> grains`. `SnapshotCGrain` must deep-copy BOTH; `RunCGrainLockstep`'s
`replica == authority` must memcmp BOTH; a `CGrainCommand` can target a rigid body OR a grain. This is a
lockstep over a coupled rigid+granular system — the GR5 (grain) + CP5 (coupled-fluid) harnesses fused.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The CP5 / GR5 lockstep harness to MIRROR near-verbatim (`engine/sim/couple.h` CP5 + `engine/sim/grain.h`
  GR5):** `CoupleCommand`/`ApplyCoupleCommand`/`SimCoupleTick`/`SnapshotCouple`/`RestoreCouple`/
  `RunCoupleLockstep`/`RunCoupleRollback` (CP5, the SAME shape over a bodies+particles world) — CG5 is the
  SAME with `grain::GrainParticle` instead of `fluid::FluidParticle` and `StepCGrain` as the per-tick step.
  `SnapshotCGrain` returns a `CGrainSnapshot{ std::vector<fpx::FxBody> bodies; std::vector<grain::GrainParticle>
  grains; }` (deep-copy BOTH); `RestoreCGrain` restores both. The stream is processed in ARRAY ORDER per tick.
- **The CG4 step `SimCGrainTick` wraps (this branch's `engine/sim/couple_grain.h`):** `StepCGrain(world, dt,
  iters)` — the bit-exact coupled tick. `CGrainWorld` (bodies + grains + config). DO NOT modify CG1–CG4
  functions — CG5 is additive.
- **The showcase + golden discipline:** CP5's `--couple-lockstep-shot` / GR5's `--grain-lockstep-shot` (BOTH
  run the identical CPU harness, color the converged state to a side view) — mirror for the cgrain pair.
  `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect
  JSON golden** — the GR2/CP2 lesson), `tests/cgrain_test.cpp`.

## Design decisions (locked)
1. **`CGrainCommand` + `ApplyCGrainCommand` (the multi-body input event).** `CGrainCommand{tick, kind, target,
   arg}`; kinds: `kCmdBodyShove` (add `arg` to rigid body `target`'s velocity — the "shove the body" headline),
   `kCmdBodyMove` (add `arg` to body `target`'s position), `kCmdGrainWind` (add `arg` to grain `target`'s
   velocity). Integer adds; out-of-range target / unknown kind → no-op; static bodies/grains never mutated.
   A `std::vector<CGrainCommand>` is the stream, processed in ARRAY ORDER per tick.
2. **`SimCGrainTick`.** Apply ALL stream commands with `.tick == tick` in array order, THEN `StepCGrain` one
   step. Pure integer, fixed order → bit-identical on every peer/platform.
3. **`SnapshotCGrain`/`RestoreCGrain` (the multi-body snapshot).** Deep copy / restore of BOTH the `bodies`
   AND the `grains` vectors (a `CGrainSnapshot` struct); bit-exact round-trip.
4. **`RunCGrainLockstep`** (ticks from a copy of init + the stream → the final coupled state) **+
   `RunCGrainRollback`** (snapshot at mispredictTick → speculate with the WRONG input → restore + re-sim the
   CORRECT authStream → the corrected state). The CP5/GR5 twins, over `CGrainWorld`.
5. **Showcase `--cgrain-lockstep-shot <out>` (Vulkan) AND `--cgrain-lockstep` (Metal) — WIRE BOTH, IDENTICAL
   CPU harness.** A SMALL/FAST coupled scene (a modest grain bed + a body, per the performance note) + a
   command stream that SHOVES the body (`kCmdBodyShove`) at chosen ticks so it moves to a NON-TRIVIAL converged
   state (the FL5/CP5 lesson — a stream that visibly displaces the body AND perturbs the sand). Color the
   converged coupled state (bodies + grains) to a side view. Golden = `tests/golden/metal/cgrain_lockstep.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
6. **PROOFS (fail loudly; exact lines — the CP5/GR5 set, multi-body):**
   - **(1) lockstep (inputs-only, BOTH bodies+grains):** `RunCGrainLockstep` authority == replica byte-for-byte
     across BOTH the bodies AND the grains. Print `cgrain-lockstep: replica==authority {bodies:<B>, grains:<N>}
     BIT-EXACT (<T> ticks, inputs-only)`.
   - **(2) rollback:** `RunCGrainRollback` corrected == the lockstep authority byte-for-byte (bodies+grains),
     AND the mispredicted-before-rollback state DIFFERED. Print `cgrain-lockstep rollback: corrected to
     authority BIT-EXACT (mispredict@tick<M> diverged then converged)`.
   - **(3) determinism + snapshot round-trip:** two lockstep runs identical; `RestoreCGrain(SnapshotCGrain(w))`
     == w (bodies+grains). Print `cgrain-lockstep determinism: two runs BYTE-IDENTICAL + snapshot round-trip exact`.
   - **(4) motion (non-trivial):** the command stream displaced the body (and perturbed the sand) by a
     non-trivial amount. Print `cgrain-lockstep motion: shoved the body by <D>`.
   - **Golden discipline: ONLY `tests/golden/metal/cgrain_lockstep.png`; do NOT commit it.** Existing 145
     image goldens UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict):** the Vulkan-Windows converged state == the Metal-Mac converged
   state == the golden, ZERO differing pixels (the same pure-CPU harness on both → bit-identical).
8. **Tests `tests/cgrain_test.cpp` additions (pure CPU, FAST — small scene/few ticks):** `ApplyCGrainCommand`
   (body-shove adds to body vel, grain-wind adds to grain vel, body-move adds to body pos, static unmoved,
   out-of-range no-op); `SnapshotCGrain`/`RestoreCGrain` round-trip (bodies AND grains); `RunCGrainLockstep`
   two runs identical; `RunCGrainRollback` corrected == authority AND mispredicted ≠ authority. Use a tiny bed
   + few ticks so the test stays fast. Clean under `windows-msvc-asan`.
9. **Introspect.** Add exactly `deterministic-cgrain-lockstep` (features) + `--cgrain-lockstep-shot`
   (showcases). **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`
   (the GR2/CP2 lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None — PURE CPU.** NO new shader, NO new RHI, NO GPU pass. `rhi.h` + `rhi_factory` + backend dirs + ALL
  shaders UNCHANGED. `engine/sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple.h` + `engine/physics/`
  UNCHANGED. CG1–CG4 cgrain code + shaders UNCHANGED (CG5 appends the CPU harness). Report the seam is empty
  (incl. shaders).

## Out of scope (YAGNI — last CG slice before the render)
The lit 3D render (CG6). Delta-compressed snapshots, partial-state sync, input delay/buffering, real
transport. CG5 claims ONLY: the bit-exact coupled sim is lockstep + rollback replayable from inputs alone —
across BOTH the rigid bodies AND the sand — bit-identical CPU↔Vulkan↔Metal, with the converged-state integer
golden + the four CP5/GR5 proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 98) + the new `cgrain_test` lockstep cases (FAST — a
   small scene). Clean under `windows-msvc-asan`.
2. **proofs + visual:** `--cgrain-lockstep-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows a coherent converged coupled state (body + sand) visibly shoved by the stream (pixel-check; the
   NAV6/CL6 lesson).**
3. Metal: `visual_test --cgrain-lockstep` → new golden `tests/golden/metal/cgrain_lockstep.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (the
   harness is pure CPU).** Cross-vendor Vulkan-vs-Metal STRICT ZERO (the same CPU harness on both).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `cgrain_lockstep.png` added;
   the other 145 byte-identical (re-run `--cgrain-query/support/displace/step-shot` → still bit-exact). `git
   diff master --stat -- tests/golden` = ONLY `cgrain_lockstep.png` (metal) + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-cgrain-lockstep` + `--cgrain-lockstep-shot`; introspect
   test updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` + ALL shaders UNCHANGED — PURE CPU, no new shader/RHI; `engine/sim/fpx.h` +
   `grain.h` + `fluid.h` + `cloth.h` + `couple.h` + `engine/physics/` + CG1–CG4 cgrain code/shaders
   byte-unchanged). `scripts/verify.ps1` updated: `cgrain_lockstep` golden in the Mac loop +
   `--cgrain-lockstep-shot` in `$vkShots`.
