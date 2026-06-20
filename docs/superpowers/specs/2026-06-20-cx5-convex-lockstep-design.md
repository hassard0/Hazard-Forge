# Slice CX5 — Deterministic Convex Contacts: LOCKSTEP + ROLLBACK (the netcode headline) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIFTH slice of FLAGSHIP #19
> (DETERMINISTIC CONVEX RIGID-BODY CONTACTS, `hf::sim::convex`). CX1-CX4 built the deterministic box-box contact
> sim (SAT → manifold → angular impulse → settling stack). CX5 is the NETCODE HEADLINE — the FR5/CP5/CG5/GF5/BD5/
> AC5/VH5/JT5 twin: two peers fed only an input-command stream converge BYTE-IDENTICAL, and a rollback re-sims from
> a snapshot bit-for-bit. The moat sentence: **angular-impulse convex rigid-body contacts that are
> lockstep-replayable — UE5's float Chaos can never guarantee this.** PURE CPU (NO GPU shader, NO new RHI) → both
> backends run the IDENTICAL `StepConvexWorld` harness → bit-identical golden BY CONSTRUCTION (cross-vendor 0 px).
> CX1-CX4's `convex.h` code is BYTE-FROZEN (CX5 is additive). Branch: `slice-cx5`. See [[hazard-forge-convex-roadmap]].

**Goal:** Extend `engine/sim/convex.h` (additive — CX1-CX4 byte-unchanged) with the lockstep harness: a
`ConvexCommand` (a per-tick perturbation input — add-impulse / set-angVel on a body) + a `ConvexSnapshot` (the world
state at a tick) + `SnapshotConvex`/`RestoreConvex` + `SimConvexTick` (apply this tick's commands → `StepConvexWorld`)
+ `RunConvexLockstep` (two peers converge) + `RunConvexRollback` (re-sim from a snapshot). Add `--convex-lockstep-shot`
(Vulkan) / `--convex-lockstep` (Metal) — both run the SAME pure-CPU harness. Bake the integer golden
`convex_lockstep`. **NO new shader, NO new RHI.**

## Design call: the pure-CPU lockstep harness over `StepConvexWorld`

The convex step is already fully deterministic (CX4: fixed orders, integer math). Lockstep falls out: feed two
independent `ConvexWorld` peers the SAME initial world + the SAME per-tick command stream, step both K ticks, and
they are byte-identical. Rollback: snapshot the world at tick T, mis-simulate forward (a wrong/late command), then
restore the snapshot and re-sim with the CORRECT command stream — bit-identical to the authority.
- **`ConvexCommand { uint32_t tick; uint32_t kind; uint32_t bodyId; FxVec3 arg; }`** — applied BEFORE the step on
  its `tick`. Kinds (convex-local, FIXED): `kConvexCmdAddImpulse = 0` (`body.vel += arg·body.invMass` — an impulse;
  statics unaffected since invMass==0) and `kConvexCmdSetAngVel = 1` (`body.angVel = arg` — a spin input). Pure
  integer.
- **`ApplyConvexCommands(world, commands, tick)`** — apply, in the commands' FIXED array order, every command whose
  `tick == tick` (a body's invMass==0 → the impulse is a no-op by construction). Deterministic.
- **`SimConvexTick(world, cfg, commands, tick)`** — `ApplyConvexCommands(world, commands, tick)` then
  `StepConvexWorld(world, cfg)`. ONE deterministic tick with its inputs.
- **`ConvexSnapshot { std::vector<FxBody> bodies; uint32_t tick; }`** — the world's mutable state at a tick (the
  `boxes` are immutable/shared, so the snapshot is just the bodies + the tick). `SnapshotConvex(world, tick)` /
  `RestoreConvex(world, snapshot)` (memberwise copy — restores vel/pos/orient/angVel exactly).
- **`RunConvexLockstep(world0, cfg, commands, ticks)`** → two independent peers (authority + replica) BOTH start
  from `world0`, BOTH run `SimConvexTick` for `ticks` with the SAME command stream; return whether the two final
  body vectors are byte-identical (`std::memcmp`) + the final authority world (for the golden). The make-or-break.
- **`RunConvexRollback(world0, cfg, commands, ticks, rollbackAt, mispredict)`** → run the authority to `ticks`.
  Separately: run a peer to `rollbackAt`, `SnapshotConvex`, apply a MISPREDICTED command (e.g. an extra/wrong
  impulse) + step one tick (now diverged from authority), then `RestoreConvex` the snapshot and re-sim the CORRECT
  stream to `ticks`; return whether the corrected final world == the authority final world byte-for-byte AND whether
  the mispredicted intermediate genuinely diverged (so the test proves a REAL divergence was corrected, not a no-op).

**The golden scene:** reuse the CX4 settling-stack scene (static floor + 3 dynamic boxes) + a small deterministic
command stream (a couple of impulse/angVel perturbations at fixed early ticks that knock the stack, then it
re-settles). Render the CONVERGED authority world (the same 2D side-view as CX4) — both backends produce the
identical image BY CONSTRUCTION. PURE CPU.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **CX1-CX4 `engine/sim/convex.h` (read it; APPEND only after `MeasureStack`):** `ConvexWorld`, `ConvexStepConfig`,
  `StepConvexWorld`/`StepConvexWorldN`, `FxBody`, `MeasureStack`. CX5 wraps `StepConvexWorld` with the command
  application + snapshot/restore. CX1-CX4 byte-frozen.
- **The lockstep precedent (`engine/sim/fract.h` FR5 / `engine/sim/boids.h` BD5 / `engine/sim/couple.h` CP5):** the
  EXACT command + snapshot + RunLockstep + RunRollback shape (the `kCmd*` vocab, the `Snapshot{...}` struct, the
  authority-vs-replica memcmp, the rollback-corrects-a-mispredict harness). Mirror it for `ConvexWorld`. The
  `fpx.h` command vocab (`kCmdAddImpulse`/`kCmdSetAngVel`, `ApplyCommand`, fpx.h:507-527) is the naming model —
  CX5 defines CONVEX-LOCAL kinds over `ConvexWorld` (NOT the fpx `FxWorld` ones).
- **The showcase precedents (`samples/hello_triangle/main.cpp` + `metal_headless/visual_test.mm`):** FR5/BD5's
  `--fract-lockstep`/`--boids-lockstep` (the pure-CPU harness + the 4 proofs + the converged-world render, NO GPU
  dispatch). Mirror them for `--convex-lockstep`.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), append to `tests/convex_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/convex.h`** (CX1-CX4 byte-frozen): `ConvexCommand`, `kConvexCmdAddImpulse`/
   `kConvexCmdSetAngVel`, `ApplyConvexCommands`, `SimConvexTick`, `ConvexSnapshot`, `SnapshotConvex`/
   `RestoreConvex`, `RunConvexLockstep`, `RunConvexRollback`. Pure integer, FIXED command order. **NO new shader, NO
   new RHI** (the seam includes shaders — EMPTY; this is a pure-CPU slice).
2. **Showcase `--convex-lockstep-shot <out>` (Vulkan-side binary) AND `--convex-lockstep` (Metal-side binary) —
   WIRE BOTH** (standalone arg-parse). BOTH run the IDENTICAL pure-CPU `RunConvexLockstep` + `RunConvexRollback`
   over the CX4 stack scene + a fixed command stream — NO GPU dispatch (the FR5/BD5 pure-CPU pattern). Render the
   converged authority world as the CX4-style 2D side-view. Golden = `tests/golden/metal/convex_lockstep.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) lockstep:** authority == replica byte-for-byte. Print `convex-lockstep: {bodies:<N>, ticks:<K>,
     commands:<C>} authority==replica BIT-IDENTICAL`; assert.
   - **(2) determinism:** two runs → identical. Print `convex-lockstep determinism: two runs BYTE-IDENTICAL`.
   - **(3) rollback:** the corrected re-sim == the authority byte-for-byte. Print `convex-lockstep rollback:
     corrected==authority BIT-EXACT`; assert.
   - **(4) mispredict real:** the mispredicted intermediate genuinely DIVERGED from the authority before the
     rollback (proving the rollback corrected a real divergence, not a no-op). Print `convex-lockstep mispredict:
     diverged before rollback (real divergence corrected)`; assert the divergence was non-zero.
   - **Golden discipline: ONLY `tests/golden/metal/convex_lockstep.png`; do NOT commit it.** Existing 187 image
     goldens UNTOUCHED.
4. **Cross-backend bar (PURE CPU → strict):** both backends run the identical harness → the golden is bit-identical
   BY CONSTRUCTION; cross-vendor ZERO differing pixels.
5. **Tests — APPEND to `tests/convex_test.cpp` (pure CPU):** `RunConvexLockstep` over the stack scene + a command
   stream → authority==replica; `RunConvexRollback` → corrected==authority AND the mispredict diverged;
   `ApplyConvexCommands` applies an impulse to a dynamic body + is a no-op on a static body; `SnapshotConvex`/
   `RestoreConvex` round-trips the world exactly; two runs byte-identical. Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-convex-lockstep` (features) + `--convex-lockstep-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None — and NO new shader.** Pure CPU. `rhi.h` + backend dirs UNCHANGED. `engine/sim/fpx.h` + all sibling sim
  headers + **CX1-CX4's convex.h code + convex_sat.comp + convex_manifold.comp + convex_solve.comp +
  convex_step.comp** + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing shaders UNCHANGED.
  `convex.h` APPEND-only. Report the seam empty (only the convex.h APPEND + the showcase/test/introspect are
  new/changed; NO shaders/ change at all).

## Out of scope (YAGNI — CX6)
The lit 3D render capstone (CX6 — CX5's render is the 2D converged-stack side-view). Real network transport /
delay / packet loss (CX5 is the DETERMINISM PROOF — lockstep convergence + rollback re-sim; the transport layer is
out of scope, as in every prior #5 slice). Prediction smoothing / interpolation. CX5 claims ONLY: the convex
contact sim is lockstep-deterministic (two peers converge from inputs alone) and rollback-replayable (a snapshot
re-sim is bit-exact), bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs. NOTE: boxes only;
the same within-band stack-stability caveat as CX4 (the command stream knocks + re-settles within K).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 105 incl. CX1-CX4's `convex_test` + the appended CX5 cases).
   Clean under `windows-msvc-asan` (build+run `convex_test` + `introspect_test`).
2. **proofs + visual:** `--convex-lockstep-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID (the harness is pure CPU but the binary still inits Vulkan for other paths; confirm the layer LOADED
   + zero errors). **VERIFY the image shows a coherent settled stack (the converged authority world).**
3. Metal: `visual_test --convex-lockstep` → new golden `tests/golden/metal/convex_lockstep.png`; two runs DIFF
   0.0000. **Confirm `visual_test.mm` in the diff; confirm NO shader added (pure CPU).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `convex_lockstep.png` added; the
   other 187 byte-identical. `git diff master --stat -- tests/golden` = ONLY `convex_lockstep.png` (metal) + the
   introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-convex-lockstep` + `--convex-lockstep-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h` + all sibling sim headers + **CX1-CX4's convex.h code +
   ALL convex shaders** + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing shaders byte-unchanged;
   **NO shaders/ change at all — pure CPU**). `scripts/verify.ps1` updated: `convex_lockstep` golden in the Mac loop
   + `--convex-lockstep-shot` in `$vkShots`.
