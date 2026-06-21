# Slice MF5 — Hull Narrowphase Hardening: LOCKSTEP + ROLLBACK over the hardened stack (the netcode headline) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of
> FLAGSHIP #25 (DETERMINISTIC HULL NARROWPHASE HARDENING, `hf::sim::manifold`). MF1-MF4 built the face topology,
> the multi-point manifold, its GPU twin, and the FULL-inertia hardened step (a flat-dropped hull SETTLES TO REST
> where the frozen single-point step teeters — bit-identical CPU/Vulkan/Metal). MF5 proves the hardened sim is
> **lockstep- and rollback-replayable**: two peers fed only an input-command stream re-derive the entire hardened
> world — including the multi-point manifolds + the full-inertia response — byte-identical, and a rollback re-sims
> from a snapshot bit-for-bit. THE HEADLINE: the deterministic STABLE STACK is now NETCODE-replayable — a settling
> pile of polyhedra that two peers reproduce exactly and a client can roll back and re-sim, which mainstream float
> engines cannot do (their stabilization + contact solve is non-deterministic and DISABLED in lockstep). Because
> `StepHullWorldHardened` (MF4) is a pure deterministic integer function and its only mutable state is the body
> vector (faces + inertia are pure functions of the immutable `hulls`, RECOMPUTED each tick — never snapshotted),
> this falls out by retargeting the FROZEN GJ5/BP5/CD5 lockstep harness over it — PURE CPU, no shader, no RHI →
> both backends run the IDENTICAL harness → the golden is bit-identical BY CONSTRUCTION (cross-vendor 0 px).
> APPEND to `engine/sim/manifold.h` (MF1-MF4 + gjk/broad/ccd/convex/fpx/etc BYTE-FROZEN). Branch: `slice-mf5`. See
> [[hazard-forge-manifold-roadmap]], [[hazard-forge-gjk-roadmap]], [[hazard-forge-docs-style]],
> [[hazard-forge-metal-showcase-gate]].

**Goal:** Extend `engine/sim/manifold.h` (additive — MF1-MF4 byte-unchanged) with `SimHullTickHardened`
(`gjk::ApplyHullCommands` + `StepHullWorldHardened`) + `RunHullLockstepHardened` + `RunHullRollbackHardened`,
reusing the FROZEN command/snapshot/equality machinery (`convex::ConvexCommand`, `gjk::HullSnapshot`/`SnapshotHull`/
`RestoreHull`/`HullBodiesEqual`/`ApplyHullCommands`). Add the showcase `--mf5-lockstep-shot` (Vulkan) /
`--mf5-lockstep` (Metal) — both run the SAME pure-CPU harness over a hardened-stack scene + a command stream, and
render the converged authority world. Bake the integer golden `mf5_lockstep`. **NO new shader, NO new RHI.**

## Design call: the pure-CPU lockstep harness over `StepHullWorldHardened` (the GJ5/BP5/CD5 twin)

`StepHullWorldHardened` (MF4) is a fully deterministic integer tick; its mutable replayable state is the body
vector (the `hulls` are immutable; the faces, the full inertia tensors, and the per-pair multi-point manifolds are
RE-DERIVED each tick from the bodies/hulls — NOT state to snapshot, exactly why the lockstep holds through the
hardened solve). So MF5 is the direct GJ5/CD5 twin — the SAME harness with the step swapped to
`StepHullWorldHardened`.
- **Commands — REUSE frozen `convex::ConvexCommand` + `gjk::ApplyHullCommands`** (gjk.h:1315). `SimHullTickHardened(
  world, cfg, commands, tick)` = `gjk::ApplyHullCommands(world, commands, tick)` then `StepHullWorldHardened(world,
  cfg)` (the `gjk::SimHullTick` shape, gjk.h:1335, with the step swapped). A command can NUDGE the stack (an
  add-impulse / set-angVel), making the settling pile the lockstep scene.
- **Snapshot/restore/equality — REUSE frozen `gjk::HullSnapshot`/`SnapshotHull`/`RestoreHull`/`HullBodiesEqual`**
  (gjk.h:1345-1371, bodies only). NO new snapshot type (the faces/inertia/manifolds are recomputed, not stored).
- **`RunHullLockstepHardened(world0, cfg, commands, ticks, outIdentical)`** → two peers (authority + replica) BOTH
  from `world0`, BOTH run `SimHullTickHardened` for `ticks` with the SAME command stream → `*outIdentical =
  HullBodiesEqual(...)`; return the converged authority. (Mirror `gjk::RunHullLockstep`, gjk.h:1379, step swapped.)
- **`RunHullRollbackHardened(world0, cfg, authStream, mispredictStream, ticks, rollbackAt, outCorrectedEqAuthority,
  outMispredictDiverged)`** → the `gjk::RunHullRollback` control flow (gjk.h:1402) over `SimHullTickHardened`:
  advance to `rollbackAt`, `SnapshotHull`, speculatively mispredict (≤3 ticks, diverges), `RestoreHull`, re-sim the
  correct stream; set corrected==authority + mispredict-diverged flags. cfg + streams CONSTANT.

**The golden scene:** the MF4 hardened-stack scene (the flat-rest box-on-box, or a small settling stack) + a
deterministic command stream (an impulse nudges the stack; it re-settles; a later command perturbs). Render the
converged authority world (the MF4-style PURE-INTEGER 2D side-view — strict-zero cross-vendor by construction).
PURE CPU (no GPU dispatch → no TDR concern).

## Reuse map (file:line)
- **MF4 `engine/sim/manifold.h` (APPEND after `StepHullWorldHardenedN`):** `StepHullWorldHardened`/`…N`,
  `FxHullInertiaBodyFull`, `HullContactMulti`. MF1-MF4 byte-frozen.
- **gjk.h GJ5 machinery (read-only — REUSE verbatim):** `gjk::SimHullTick`/`RunHullLockstep`/`RunHullRollback`
  (gjk.h:1335/1379/1402 — the SHAPE to mirror with `StepHullWorld`→`StepHullWorldHardened`), `gjk::ApplyHullCommands`
  (gjk.h:1315), `gjk::HullSnapshot`/`SnapshotHull`/`RestoreHull`/`HullBodiesEqual` (gjk.h:1345-1371),
  `convex::ConvexCommand`/`kConvexCmdAddImpulse`/`kConvexCmdSetAngVel`. Do NOT modify gjk.h/etc — BYTE-FROZEN.
- **The showcase precedent:** GJ5/BP5/CD5 `--gjk-lockstep`/`--broad-lockstep`/`--ccd-lockstep` (the pure-CPU
  harness + the 4 proofs + the converged-world render, NO GPU dispatch). Mirror for `--mf5-lockstep`. The MF4
  `--mf4-stack` 2D-integer side-view render is the render base (strict-zero cross-vendor).
- **Registration:** `scripts/verify.ps1` (append `mf5_lockstep` + `--mf5-lockstep-shot` to `$vkShots`),
  `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON golden**), append to
  `tests/manifold_test.cpp`. (No shader → nothing for `hf_gen_msl`.)

## Design decisions (locked)
1. **APPEND to `engine/sim/manifold.h`** (MF1-MF4 byte-frozen): `SimHullTickHardened`, `RunHullLockstepHardened`,
   `RunHullRollbackHardened`. Pure integer, FIXED command + peer order. **NO new shader, NO new RHI** (the seam
   incl. shaders — EMPTY).
2. **Showcase `--mf5-lockstep-shot <out>` (Vulkan) AND `--mf5-lockstep` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--mf5-lockstep` BEFORE reporting DONE).** BOTH run the IDENTICAL pure-CPU
   `RunHullLockstepHardened` + `RunHullRollbackHardened` over the hardened-stack scene + a fixed command stream —
   NO GPU dispatch. Render the converged authority world. Golden = `tests/golden/metal/mf5_lockstep.png` (Mac-baked
   by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1)** `mf5-lockstep: {bodies:<N>, ticks:<K>, commands:<C>} authority==replica BIT-IDENTICAL`
   - **(2)** `mf5-lockstep determinism: two runs BYTE-IDENTICAL`
   - **(3)** `mf5-lockstep rollback: corrected==authority BIT-EXACT`
   - **(4)** `mf5-lockstep mispredict: diverged before rollback (real divergence corrected)`
   - **Golden discipline: ONLY `tests/golden/metal/mf5_lockstep.png`; do NOT commit it.** Existing 225 goldens
     UNTOUCHED.
4. **Cross-backend bar (PURE CPU → strict):** both backends run the identical harness → bit-identical BY
   CONSTRUCTION; cross-vendor ZERO differing pixels (the strict-zero integer render, MF4 lineage).
5. **Tests — APPEND to `tests/manifold_test.cpp` (pure CPU):** `RunHullLockstepHardened` authority==replica;
   `RunHullRollbackHardened` corrected==authority AND mispredict diverged; two runs byte-identical; a command stream
   moved the stack non-trivially (the hardened step + re-manifold exercised — not a frozen no-op). Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `manifold-lockstep` (features) + `--mf5-lockstep-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None — and NO new shader.** Pure CPU. `engine/sim/manifold.h` APPEND-only (MF1-MF4 frozen); gjk.h/convex.h/etc
  + ALL other sim headers + ALL existing shaders UNCHANGED. Report the seam empty (only the manifold.h APPEND + the
  showcase/test/introspect are new/changed; NO shaders/ change at all).

## Out of scope (YAGNI — MF6)
The lit/settled-stack render capstone polish (MF6). Real network transport. MF5 claims ONLY: the hardened sim is
lockstep-deterministic (two peers converge from inputs alone, re-deriving the manifolds + inertia each tick) and
rollback-replayable, bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs. CAVEATS inherited:
the MF4 within-band settle (the Gauss-Seidel + linear de-pen residual); canonical hulls only.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "manifold|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--mf5-lockstep-shot` on Vulkan: the 4 proofs + exit 0. VERIFY a coherent converged world
   (the settled stack). (PURE CPU — no GPU compute, no TDR/VUID risk from this slice.)
3. Metal: `visual_test --mf5-lockstep` → `tests/golden/metal/mf5_lockstep.png`; two runs DIFF 0.0000. **Confirm
   `--mf5-lockstep` is wired in `visual_test.mm` (grep it) BEFORE the Mac bake** — NO shader added (pure CPU).
   Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `mf5_lockstep.png` added; the other 225 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+manifold-lockstep` + `--mf5-lockstep-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + MF1-MF4 manifold.h code + gjk.h/convex.h + ALL other sim headers + ALL existing
   shaders byte-unchanged; manifold.h APPEND-only; NO shaders/ change). `mf5_lockstep` in the Mac loop +
   `--mf5-lockstep-shot` in `$vkShots`.
