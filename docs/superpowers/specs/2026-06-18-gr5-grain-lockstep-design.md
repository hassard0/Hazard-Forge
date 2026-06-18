# Slice GR5 — Deterministic GPU Granular/Sand: LOCKSTEP + ROLLBACK (the netcode HEADLINE, pure CPU) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #10
> (DETERMINISTIC GPU GRANULAR / SAND, `hf::sim::grain`). Proves the bit-exact GR1–GR4 granular sim (WITH
> friction) is true cross-platform **LOCKSTEP + ROLLBACK**: a peer fed the INPUT command stream ALONE
> re-derives the authority's exact grain state bit-for-bit, and a mispredicted input is corrected by rolling
> back to a snapshot + re-simulating the authoritative stream. **PURE CPU, 0 backend symbols, NO new shader /
> RHI** — a determinism PROPERTY of the existing bit-exact `StepGrainFriction`. The direct twin of the FL5 /
> CL5 / FPX5 harness. This is the trilogy's **4th netcode headline** (rigid → cloth → fluid → grain): a
> deterministic, rollback-able, bit-identical-cross-platform GRANULAR sim — UE5 has no such thing. Branch:
> `slice-gr5`. See [[hazard-forge-grain-roadmap]].

**Goal:** Extend `engine/sim/grain.h` (additive — GR1–GR4 byte-unchanged) with the lockstep/rollback netcode
harness: `GrainCommand` + `ApplyGrainCommand` + `SimGrainTick` + `SnapshotGrain`/`RestoreGrain` +
`RunGrainLockstep` + `RunGrainRollback` (the FL5 shape over `GrainParticle`/`StepGrainFriction`). Add
`--grain-lockstep-shot` (Vulkan) / `--grain-lockstep` (Metal) — BOTH run the IDENTICAL CPU harness. Bake the
integer golden `grain_lockstep` (the converged state). NO new shader, NO new RHI.

## Design call: a determinism PROPERTY (pure CPU) — strict zero-diff cross-platform
Lockstep is not a new GPU pass — it is the proof that the bit-exact `StepGrainFriction` (GR4, which itself is
bit-identical Vulkan/Metal) is replayable from inputs alone. Both `--grain-lockstep-shot` (Vulkan side) and
`--grain-lockstep` (Metal side) run the SAME pure-CPU harness → the converged grain state is byte-identical
on Vulkan-Windows AND Metal-Mac, which IS the cross-platform-lockstep evidence. Bar: strict INTEGER (the
cross-platform compare is zero-differing-pixel, the FL5/CL5 bar). NO `<cmath>`, NO RNG, NO clock.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **The FL5 lockstep harness to MIRROR near-verbatim (`engine/sim/fluid.h:817-920+`):** `kCmdWind`/`kCmdPush`
  (`fluid.h:839-840`), `FluidCommand` (`fluid.h:842-847`), `ApplyFluidCommand` (`fluid.h:853-866`, add to
  vel / add to pos, skip static, out-of-range/unknown no-op), `SimFluidTick` (`fluid.h:874-881`, apply the
  tick's commands in ARRAY ORDER then one sim step), `SnapshotFluid`/`RestoreFluid` (`fluid.h:885-894`, the
  value-copy snapshot/restore), `RunFluidLockstep` (`fluid.h:901-910`, ticks from a copy of init + stream →
  final state), `RunFluidRollback` (`fluid.h:912+`, snapshot at mispredictTick → speculate with the WRONG
  input → restore + re-sim the CORRECT stream → corrected == lockstep authority, and mispredicted differed).
  GR5 is the SAME shape with `Grain` types and `StepGrainFriction` (GR4) as the per-tick step.
- **The GR4 sim step `SimGrainTick` wraps (`engine/sim/grain.h`):** `StepGrainFriction(grains, spheres,
  gravity, dt, groundY, hSearch, mu, iters)` — the bit-exact per-tick granular step (predict → neighbours →
  K iters {normal + friction} → velocity → collide). `kFlagStatic`/`GrainParticle`/`GrainSphereCollider`.
  DO NOT modify the GR1–GR4 functions — GR5 is additive.
- **The showcase + golden discipline:** FL5's `--fluid-lockstep-shot` / `--fluid-lockstep` (BOTH run the
  identical CPU harness, color the converged state to a side view) — mirror for the grain pair.
  `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**REBAKE the introspect
  JSON golden** — the GR2 lesson), `tests/grain_test.cpp`.

## Design decisions (locked)
1. **`GrainCommand` + `ApplyGrainCommand` (the FL5 input event).** `GrainCommand{tick, kind, target, arg}`;
   `kCmdWind` adds `arg` to the target grain's velocity, `kCmdPush` adds `arg` to its position (integer adds,
   skip static, out-of-range/unknown → no-op). The deterministic per-tick INPUT a netcode layer puts on the
   wire (NOT full state). A `std::vector<GrainCommand>` is the stream, processed in ARRAY ORDER per tick.
2. **`SimGrainTick`.** Apply ALL stream commands with `.tick == tick` in array order, THEN `StepGrainFriction`
   one step (GR4 — the full friction sim). Pure integer, fixed order → bit-identical on every peer/platform.
3. **`SnapshotGrain`/`RestoreGrain` (value-copy).** Deep copy / restore of the full integer grain array (the
   rollback primitive); bit-exact round-trip.
4. **`RunGrainLockstep`** (ticks from a copy of init + the stream → final state) **+ `RunGrainRollback`**
   (snapshot at mispredictTick → speculate with the WRONG input → restore + re-sim the CORRECT authStream →
   the corrected state). The FL5 twins.
5. **Showcase `--grain-lockstep-shot <out>` (Vulkan) AND `--grain-lockstep` (Metal) — WIRE BOTH, IDENTICAL
   CPU harness.** Use the GR4 friction scene (the staggered block + μ + a collider-free or simple ground) so
   the lockstep proves the FULL granular sim (incl. friction) is replayable. A command stream
   (wind/push on a few grains at chosen ticks) perturbs the pile to a NON-TRIVIAL converged state (the FL4/FL5
   lesson — use a stream that visibly DISPLACES the grains so the proof is non-degenerate). Color the
   converged state to a BGRA8 side view. Golden = `tests/golden/metal/grain_lockstep.png` (Mac-baked by the
   CONTROLLER — DO NOT commit).
6. **PROOFS (fail loudly; exact lines — the FL5/CL5 set):**
   - **(1) lockstep (inputs-only):** `RunGrainLockstep` authority == replica byte-for-byte (a peer fed the
     stream ALONE re-derives the exact state). Print `grain-lockstep: replica==authority <N> grains BIT-EXACT
     (<T> ticks, inputs-only)`.
   - **(2) rollback:** `RunGrainRollback` corrected == the lockstep authority byte-for-byte, AND the
     mispredicted-before-rollback state DIFFERED (a real divergence was fixed). Print `grain-lockstep
     rollback: corrected to authority BIT-EXACT (mispredict@tick<M> diverged then converged)`.
   - **(3) determinism + snapshot round-trip:** two lockstep runs identical; `RestoreGrain(SnapshotGrain(p))`
     == p. Print `grain-lockstep determinism: two runs BYTE-IDENTICAL + snapshot round-trip exact`.
   - **(4) motion (non-trivial, the FL5 lesson):** the command stream displaced the grains by a non-trivial
     amount (the converged state ≠ the no-command baseline). Print `grain-lockstep motion: displaced the pile
     by <D>`.
   - **Golden discipline: ONLY `tests/golden/metal/grain_lockstep.png`; do NOT commit it.** Existing 133
     image goldens UNTOUCHED.
7. **Cross-backend bar (INTEGER, strict):** the Vulkan-Windows converged state == the Metal-Mac converged
   state == the golden, ZERO differing pixels (the same pure-CPU harness on both → bit-identical).
8. **Tests `tests/grain_test.cpp` additions (pure CPU):** `ApplyGrainCommand` (wind adds to vel, push adds to
   pos, static unmoved, out-of-range no-op); `SnapshotGrain`/`RestoreGrain` round-trip; `RunGrainLockstep`
   two runs identical; `RunGrainRollback` corrected == authority AND mispredicted ≠ authority. Clean under
   `windows-msvc-asan`.
9. **Introspect.** Add exactly `deterministic-grain-lockstep` (features) + `--grain-lockstep-shot`
   (showcases). **REBAKE `tests/golden/introspect/default_scene.json`** + update `tests/introspect_test.cpp`
   (the GR2 lesson — `git diff master -- tests/golden/` MUST include `default_scene.json`).

## RHI seam additions (summary)
- **None — PURE CPU.** NO new shader, NO new RHI, NO GPU pass. `rhi.h` + `rhi_factory` + backend dirs +
  ALL shaders UNCHANGED. `engine/sim/fpx.h` + `cloth.h` + `fluid.h` + `engine/physics/` UNCHANGED. GR1–GR4
  grain code + shaders UNCHANGED (GR5 appends the CPU harness). Report the seam is empty (incl. shaders).

## Out of scope (YAGNI — last GR slice)
The lit 3D render (GR6). Delta-compressed snapshots, partial-state sync, input delay/buffering, real
transport. GR5 claims ONLY: the bit-exact granular sim is lockstep + rollback replayable from inputs alone,
bit-identical CPU↔Vulkan↔Metal, with the converged-state integer golden + the four FL5/CL5 proofs.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 96) + the new `grain_test` lockstep cases. Clean
   under `windows-msvc-asan`.
2. **proofs + visual:** `--grain-lockstep-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation
   gate → ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the
   image shows a coherent converged grain state displaced by the command stream (pixel-check; the NAV6/CL6
   lesson).**
3. Metal: `visual_test --grain-lockstep` → new golden `tests/golden/metal/grain_lockstep.png`; two runs DIFF
   0.0000 (gate on `compare.sh` EXIT CODE). **Confirm `visual_test.mm` in the diff; confirm NO new shader (the
   harness is pure CPU).** Cross-vendor Vulkan-vs-Metal STRICT ZERO (the same CPU harness on both).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `grain_lockstep.png` added;
   the other 133 byte-identical (re-run `--grain-integrate/neighbors/contact/friction-shot` → still
   bit-exact). `git diff master --stat -- tests/golden` = ONLY `grain_lockstep.png` (metal) + the introspect
   json.
5. Introspect JSON rebaked exactly `+deterministic-grain-lockstep` + `--grain-lockstep-shot`; introspect test
   updated. (`git diff master -- tests/golden/` MUST include `default_scene.json`.)
6. Seam grep clean (`rhi.h` + ALL shaders UNCHANGED — PURE CPU, no new shader/RHI; `engine/sim/fpx.h` +
   `cloth.h` + `fluid.h` + `engine/physics/` + GR1–GR4 grain code/shaders byte-unchanged). `scripts/verify.ps1`
   updated: `grain_lockstep` golden in the Mac loop + `--grain-lockstep-shot` in `$vkShots`.
