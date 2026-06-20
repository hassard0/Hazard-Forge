# Slice FC5 — Deterministic Contact Friction: LOCKSTEP + ROLLBACK (the netcode headline) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #20
> (DETERMINISTIC TANGENTIAL CONTACT FRICTION, `hf::sim::fric`) — THE NETCODE HEADLINE. FC1-FC4 built the
> deterministic friction-locked world step. FC5 proves it is **lockstep- and rollback-replayable**: two peers fed
> only an input-command stream converge BYTE-IDENTICAL, and a rollback re-sims from a snapshot bit-for-bit. The moat
> sentence: **friction contacts that are lockstep-replayable — boxes that grip, slide, and stack with deterministic
> Coulomb friction, re-derived bit-for-bit on two machines from inputs alone.** PURE CPU (NO GPU shader, NO new
> RHI) → both backends run the IDENTICAL `StepFrictionWorld` harness → bit-identical golden BY CONSTRUCTION
> (cross-vendor 0 px). FC1-FC4's `fric.h` code + CX1-CX6's `convex.h` are BYTE-FROZEN (FC5 is additive). Branch:
> `slice-fc5`. See [[hazard-forge-fric-roadmap]].

**Goal:** Extend `engine/sim/fric.h` (additive — FC1-FC4 + convex.h byte-unchanged) with the lockstep harness over
`StepFrictionWorld`, MAXIMALLY REUSING CX5's frozen command + snapshot machinery (the friction world IS a
`convex::ConvexWorld`): `SimFricTick` (apply commands → `StepFrictionWorld`) + `RunFricLockstep` (two peers
converge) + `RunFricRollback` (re-sim from a snapshot). Add `--fric-lockstep-shot` (Vulkan) / `--fric-lockstep`
(Metal) — both run the SAME pure-CPU harness. Bake the integer golden `fric_lockstep`. **NO new shader, NO new RHI.**

## Design call: the pure-CPU lockstep harness over `StepFrictionWorld` (the CX5 twin)

`StepFrictionWorld` (FC4) is fully deterministic (fixed orders, integer math). The friction world is a
`convex::ConvexWorld` — the SAME type CX5's lockstep machinery already operates on. So FC5 REUSES CX5's frozen
infrastructure verbatim and only swaps the per-tick step:
- **Commands + snapshot — REUSE the frozen CX5 types (`convex.h`):** `convex::ConvexCommand` (tick/kind/bodyId/arg)
  + `convex::kConvexCmdAddImpulse`/`kConvexCmdSetAngVel` + `convex::ApplyConvexCommands(world, commands, tick)` +
  `convex::ConvexSnapshot` + `convex::SnapshotConvex`/`RestoreConvex` + `convex::ConvexBodiesEqual`. These work on
  `ConvexWorld` (the friction world's type) — FC5 does NOT redefine them.
- **`SimFricTick(world, cfg, commands, tick)`** — `convex::ApplyConvexCommands(world, commands, tick)` then
  `StepFrictionWorld(world, cfg)` (the FC4 friction tick). ONE deterministic friction tick with its inputs.
- **`RunFricLockstep(world0, cfg, commands, ticks, outIdentical)`** → two independent peers (authority + replica)
  BOTH start from `world0`, BOTH run `SimFricTick` for `ticks` with the SAME command stream; set `*outIdentical` to
  whether the two final body vectors are byte-identical (`convex::ConvexBodiesEqual`) + return the converged
  authority world (for the golden). The make-or-break.
- **`RunFricRollback(world0, cfg, authStream, mispredictStream, ticks, rollbackAt, outCorrectedEqAuthority,
  outMispredictDiverged)`** → run the authority to `ticks`. Separately: advance a peer to `rollbackAt` with
  `authStream`, `convex::SnapshotConvex`, speculatively advance a few ticks with the MISPREDICTED stream (now
  diverged), then `convex::RestoreConvex` the snapshot and re-sim the CORRECT `authStream` to `ticks`; set
  `*outCorrectedEqAuthority` (corrected == authority byte-for-byte) and `*outMispredictDiverged` (the speculative
  intermediate genuinely differed from the authority at the same tick — a REAL divergence corrected). This is the
  `convex::RunConvexRollback` control flow with `SimFricTick` swapped for `SimConvexTick`.

**The golden scene:** the FC4 friction stack (or ramp) scene + a small deterministic command stream (a couple of
impulse/angVel perturbations at fixed early ticks that knock the friction world, which then re-settles under
friction). Render the converged authority world (the FC4 2D side-view). Both backends produce the identical image
BY CONSTRUCTION. PURE CPU.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **FC4 `engine/sim/fric.h` (read it; APPEND only after `MeasureFrictionStack`):** `StepFrictionWorld`,
  `FrictionStepConfig`. FC1-FC4 lines byte-frozen.
- **convex.h CX5 machinery (read-only — REUSE, do NOT redefine):** `convex::ConvexCommand`,
  `convex::kConvexCmdAddImpulse`/`kConvexCmdSetAngVel`, `convex::ApplyConvexCommands`, `convex::ConvexSnapshot`,
  `convex::SnapshotConvex`/`RestoreConvex`, `convex::ConvexBodiesEqual`, `convex::ConvexWorld`. (CX5 added these ~
  `convex.h:980-1128`.) FC5's `SimFricTick`/`RunFricLockstep`/`RunFricRollback` mirror `convex::SimConvexTick`/
  `RunConvexLockstep`/`RunConvexRollback` with `StepFrictionWorld` swapped for `StepConvexWorld`.
- **The showcase precedent:** CX5's `--convex-lockstep-shot`/`--convex-lockstep` (the pure-CPU harness + the 4
  proofs + the converged-world render, NO GPU dispatch) in `samples/hello_triangle/main.cpp` +
  `metal_headless/visual_test.mm`. Mirror them for `--fric-lockstep`.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/fric_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/fric.h`** (FC1-FC4 byte-frozen): `SimFricTick`, `RunFricLockstep`, `RunFricRollback`
   (reusing the frozen `convex::` command/snapshot helpers). Pure integer, FIXED command + peer order. **NO new
   shader, NO new RHI** (the seam incl. shaders — EMPTY; this is a pure-CPU slice).
2. **Showcase `--fric-lockstep-shot <out>` (Vulkan-side binary) AND `--fric-lockstep` (Metal-side binary) — WIRE
   BOTH** (standalone arg-parse). BOTH run the IDENTICAL pure-CPU `RunFricLockstep` + `RunFricRollback` over the FC4
   friction scene + a fixed command stream — NO GPU dispatch (the CX5 pure-CPU pattern). Render the converged
   authority world as the FC4-style 2D side-view. Golden = `tests/golden/metal/fric_lockstep.png` (Mac-baked by the
   CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) lockstep:** authority == replica byte-for-byte. Print `fric-lockstep: {bodies:<N>, ticks:<K>,
     commands:<C>} authority==replica BIT-IDENTICAL`; assert.
   - **(2) determinism:** two runs → identical. Print `fric-lockstep determinism: two runs BYTE-IDENTICAL`.
   - **(3) rollback:** the corrected re-sim == the authority byte-for-byte. Print `fric-lockstep rollback:
     corrected==authority BIT-EXACT`; assert.
   - **(4) mispredict real:** the mispredicted intermediate genuinely DIVERGED from the authority before the
     rollback. Print `fric-lockstep mispredict: diverged before rollback (real divergence corrected)`; assert the
     divergence was non-zero.
   - **Golden discipline: ONLY `tests/golden/metal/fric_lockstep.png`; do NOT commit it.** Existing 194 image
     goldens UNTOUCHED.
4. **Cross-backend bar (PURE CPU → strict):** both backends run the identical harness → the golden is bit-identical
   BY CONSTRUCTION; cross-vendor ZERO differing pixels.
5. **Tests — APPEND to `tests/fric_test.cpp` (pure CPU):** `RunFricLockstep` over the friction scene + a command
   stream → authority==replica; `RunFricRollback` → corrected==authority AND the mispredict diverged; two runs
   byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-friction-lockstep` (features) + `--fric-lockstep-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None — and NO new shader.** Pure CPU. `rhi.h` + backend dirs UNCHANGED. `engine/sim/convex.h` + `fpx.h` +
  **FC1-FC4's fric.h code + ALL fric shaders (fric_basis/points/solve/step.comp)** + all other sim headers +
  `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing shaders UNCHANGED. `fric.h` APPEND-only. Report
  the seam empty (only the fric.h APPEND + the showcase/test/introspect are new/changed; NO shaders/ change at all).

## Out of scope (YAGNI — FC6)
The lit 3D render capstone (FC6 — FC5's render is the 2D converged-state side-view). Real network transport. FC5
claims ONLY: the friction contact sim is lockstep-deterministic (two peers converge from inputs alone) and
rollback-replayable (a snapshot re-sim is bit-exact), bit-identical CPU↔Vulkan↔Metal, with the integer golden + the
four proofs. NOTE: boxes only; the same within-band friction caveats as FC3/FC4.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 106 incl. FC1-FC4's `fric_test` + the appended FC5 cases).
   Clean under `windows-msvc-asan` (build+run `fric_test` + `introspect_test`).
2. **proofs + visual:** `--fric-lockstep-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID. **VERIFY the image shows a coherent converged friction world (the settled stack/ramp).**
3. Metal: `visual_test --fric-lockstep` → new golden `tests/golden/metal/fric_lockstep.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm NO shader added (pure CPU).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fric_lockstep.png` added; the other
   194 byte-identical. `git diff master --stat -- tests/golden` = ONLY `fric_lockstep.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-friction-lockstep` + `--fric-lockstep-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fpx.h` + **FC1-FC4's fric.h code + ALL fric shaders**
   + ALL other sim headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing shaders
   byte-unchanged; **NO shaders/ change at all — pure CPU**). `scripts/verify.ps1` updated: `fric_lockstep` golden
   in the Mac loop + `--fric-lockstep-shot` in `$vkShots`.
