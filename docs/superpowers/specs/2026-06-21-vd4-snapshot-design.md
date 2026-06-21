# Slice VD4 — Deterministic Gameplay/Netcode: THE HETEROGENEOUS SNAPSHOT/RESTORE + EQUALITY — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FOURTH slice of FLAGSHIP
> #27 (DETERMINISTIC GAMEPLAY / NETCODE PRODUCT LAYER, `hf::game::verdict`). VD1-VD3 built the entity world + command
> bus, the gameplay systems, and the one composed `StepWorld` tick (gameplay + the frozen physics sim). VD4 builds
> the make-or-break primitive for rollback netcode: ONE snapshot/restore that captures the ENTIRE heterogeneous
> world — the entities + every component pool + the embedded sim's TRIPLE (bodies + cache + sleep) — under a single
> restore point, plus a byte-equality over all of it. This is the prerequisite for VD5's world-level
> lockstep/rollback. THE CRUX (the reason a typical engine can't do this): a correct snapshot must capture ALL
> mutable state, or a rolled-back peer resumes with stale entity / impulse-cache / sleep state and diverges (the
> WH5 TRIPLE lesson, generalized to a heterogeneous world). VD4 proves COMPLETENESS: advance → snapshot → advance →
> restore → re-advance is byte-identical (nothing escapes the snapshot). PURE CPU integer (strict 0px). APPEND to
> `engine/game/verdict.h` (VD1-VD3 + warmhull/ecs BYTE-FROZEN). Branch: `slice-vd4`. See
> [[hazard-forge-verdict-roadmap]], [[hazard-forge-warmhull-roadmap]], [[hazard-forge-docs-style]],
> [[hazard-forge-metal-showcase-gate]].

**Goal:** Extend `engine/game/verdict.h` (additive — VD1-VD3 byte-unchanged) with `VerdictSnapshot` (the TRIPLE-plus:
`tick` + `nextId` + a deep copy of `order[]` + every component pool serialized IN `order[]` SEQUENCE + the embedded
sim's `warmhull::WarmHullSnapshot`), `SnapshotWorld(VerdictWorld&) -> VerdictSnapshot`, `RestoreWorld(VerdictWorld&,
const VerdictSnapshot&)`, and `VerdictStatesEqual(a, b)` (entity set + every component pool in `order[]` order +
`warmhull::WarmHullStatesEqual` for the sim TRIPLE). Add the showcase `--vd4-snap-shot` (Vulkan) / `--vd4-snap`
(Metal) — both run the completeness scenario (advance → snapshot → advance → restore → re-advance) and render the
restored world. Bake the integer golden `vd4_snap`. **NO new render RHI, NO new shader, NO new compute.**

## Design call: snapshot the heterogeneous world deterministically; the sim third delegates to WH5

The world's mutable state is THREE kinds: (1) the entity bookkeeping (`order[]`, `nextId`, the `EntityId→handle`
mapping), (2) the component pools (`Transform2D`/`Health`/`BodyRef`/`Velocity2D`/`Pickup`/`Score`), and (3) the
embedded sim TRIPLE (`sim.bodies` + `cache` + `sleep`). VD4 snapshots all three deterministically:
- **`VerdictSnapshot`** = `{ uint32 tick; EntityId nextId; std::vector<EntityId> order; <per-component-type a
  std::vector of {EntityId, component-value} entries, serialized in order[] SEQUENCE>; warmhull::WarmHullSnapshot
  simSnap; }`. The component pools are captured IN `order[]` SEQUENCE (NOT the ECS dense order — the determinism
  contract: the snapshot/restore/equality are all order[]-keyed, so the non-deterministic ECS pool layout never
  leaks). For each live entity (in `order[]`) record which components it has + their values.
- **`SnapshotWorld(world)`** — deep-copy `order[]` + `nextId` + `tick`; for each entity in `order[]`, capture its
  components (the value of each `has<T>` component); delegate the sim third to `warmhull::SnapshotWarmHull(sim,
  cache, sleep, tick)` (frozen). A pure read.
- **`RestoreWorld(world, snap)`** — REBUILD the world from the snapshot: clear the registry/handle/order, then for
  each `EntityId` in `snap.order` re-create an entity (handle churn is irrelevant — the `EntityId` is the pinned
  identity, re-mapped) and re-add its captured components; restore `nextId`/`tick`; restore the sim third via
  `warmhull::RestoreWarmHull`. The restored world is byte-equal (by `VerdictStatesEqual`) to the snapshotted one.
  (The `ecs::Entity` handles may differ post-restore — that is fine; the determinism contract is over `EntityId` +
  `order[]` + component values + the sim, NOT the opaque handles.)
- **`VerdictStatesEqual(a, b)`** = `a.order == b.order` (same id sequence) AND `a.nextId == b.nextId` AND every live
  entity's components are equal (compared in `order[]` order, byte-for-byte) AND `warmhull::WarmHullStatesEqual` on
  the sim TRIPLE. The make-or-break comparison VD5 builds on.

> NOTE: because the determinism contract is over `EntityId`/`order[]`/component-values (NOT the `ecs::Entity`
> handles or the pool dense indices), restore can freely re-allocate registry handles — what matters is that the
> SAME `EntityId`s map to the SAME component values in the SAME `order[]`. Pure CPU, no GPU dispatch.

## Reuse map (file:line)
- **VD1-VD3 `engine/game/verdict.h` (APPEND after `StepWorldN`):** `VerdictWorld`, `EntityId`, `order[]`,
  `Transform2D`/`Health`/`BodyRef`/`Velocity2D`/`Pickup`/`Score`, `SpawnEntity`/`DespawnEntity`, `StepWorld`/
  `StepWorldN`, `MeasureVerdict`. VD1-VD3 byte-frozen.
- **warmhull.h (read-only — the sim TRIPLE snapshot, REUSE verbatim):** `warmhull::WarmHullSnapshot`,
  `warmhull::SnapshotWarmHull` (`warmhull.h:1157`), `warmhull::RestoreWarmHull` (`warmhull.h:1170`),
  `warmhull::WarmHullStatesEqual` (`warmhull.h:1209` — the TRIPLE byte-compare). The WH5 PS5 TRIPLE lesson is the
  template; VD4 wraps it inside the heterogeneous snapshot.
- **ecs.h (read-only):** `ecs::Registry::create`/`add`/`get`/`has`/`valid`/`destroy` (rebuild the pools on restore).
- **The pure-CPU showcase precedent:** VD3 `--vd3-world` (the composed world render). Mirror for `--vd4-snap`.
- **Registration:** `scripts/verify.ps1` (`vd4_snap` + `--vd4-snap-shot`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden — verify it stages as ` M `**), append to
  `tests/verdict_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/game/verdict.h`** (VD1-VD3 byte-frozen): `VerdictSnapshot`, `SnapshotWorld`, `RestoreWorld`,
   `VerdictStatesEqual` (+ the per-component capture/restore helpers). **warmhull.h/ecs.h and ALL sim headers
   BYTE-UNCHANGED.** All snapshot data is integer; the serialization/equality are `order[]`-keyed (NOT ECS dense
   order).
2. **Showcase `--vd4-snap-shot <out>` (Vulkan) AND `--vd4-snap` (Metal) — WIRE BOTH (grep your own `visual_test.mm`
   for `--vd4-snap` BEFORE reporting DONE).** BOTH run the COMPLETENESS scenario: build a gameplay+physics world,
   advance N ticks, `SnapshotWorld`, advance M more (diverging), `RestoreWorld`, re-advance M with the same inputs →
   assert `VerdictStatesEqual` with the never-diverged reference; render the restored world. Golden =
   `tests/golden/metal/vd4_snap.png` (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines):**
   - **(1) round-trip:** `vd4-snap: snapshot->restore round-trip BIT-EXACT` — `VerdictStatesEqual(w,
     RestoreWorld(snapshot(w)))` over the whole world (entities + components + sim).
   - **(2) THE COMPLETENESS CRUX:** `vd4-snap: completeness {advance:<N>, restore@:<R>, readvance:<M>} == reference
     BIT-EXACT` — advance N → snapshot → advance M (diverge) → restore → re-advance M equals a reference world that
     advanced N+M straight through with the same inputs. **Proves NO hidden mutable state escapes the snapshot —
     the entity bookkeeping, every component pool, AND the sim TRIPLE are all captured.**
   - **(3) the necessity of the sim third:** `vd4-snap: sim-third necessary (omit -> diverge, include -> ==)` — a
     restore that OMITS the `warmhull::WarmHullSnapshot` (bodies+cache+sleep) DIVERGES; including it matches (the
     WH5 TRIPLE lesson, proven at the world level).
   - Golden discipline: ONLY `tests/golden/metal/vd4_snap.png`; do NOT commit it. Existing 236 goldens UNTOUCHED.
4. **Cross-backend bar:** the NUMERIC proofs are pure integer → strict and backend-independent. The golden IMAGE is
   the pure-integer restored-world render → strict-zero cross-vendor.
5. **Tests — APPEND to `tests/verdict_test.cpp` (pure CPU):** snapshot→restore round-trip bit-exact; the
   completeness scenario (advance→snap→advance→restore→re-advance == reference); a bodies/cache/sleep-OMITTING
   restore diverges (the sim-third necessity); `VerdictStatesEqual` compares in `order[]` order (order-stable);
   restore re-derives the same `EntityId`→component-value mapping even though the `ecs::Entity` handles differ; a
   despawn-then-restore preserves the retired-id contract. Clean under `windows-msvc-asan`.
6. **Introspect.** Add EXACTLY `verdict-snapshot` (features) + `--vd4-snap-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does (and verifies it stages).**

## RHI seam additions (summary)
- **None — NO new shader, NO new compute, NO new render RHI.** Pure-CPU integer snapshot/restore; the sim third
  delegates to the frozen `warmhull` snapshot; the render reuses the existing draw. `engine/game/verdict.h`
  APPEND-only (VD1-VD3 frozen); warmhull.h/ecs.h + ALL sim headers + ALL shaders UNCHANGED. Report the seam:
  verdict.h APPEND-only; NO rhi.h change, NO shader, NO frozen-file edit.

## Out of scope (YAGNI — later slices)
The world-level lockstep/rollback (VD5 — VD4 provides the snapshot/restore/equality that VD5's rollback control
flow uses). The render capstone (VD6). A delta/compressed snapshot (the snapshot is a full value copy — a delta
encoder is a future optimization, out of scope). Network serialization of the snapshot (VD4 is the in-memory
restore point; wire format is deferred). VD4 claims ONLY: a deterministic snapshot/restore of the heterogeneous
world (entities + every component pool + the sim TRIPLE) under one restore point, with the completeness proof + the
sim-third-necessity proof + byte-equality, bit-identical CPU/Vulkan/Metal, + the integer golden + the three proofs.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "verdict|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test).
2. **proofs + visual:** `--vd4-snap-shot` on Vulkan: the 3 proof lines (incl. the completeness crux) + exit 0 under
   the conan validation layer → ZERO VUID. VERIFY the restored-world render is coherent, no garbage/NaN.
3. Metal: `visual_test --vd4-snap` → `tests/golden/metal/vd4_snap.png`; two runs DIFF 0.0000. **Confirm `--vd4-snap`
   wired in `visual_test.mm` (grep it) BEFORE the Mac bake** — NO shader added. Cross-vendor STRICT ZERO.
4. **Render-invariance:** ONLY `vd4_snap.png` added; the other 236 byte-identical (+ controller introspect rebake,
   verified staged as ` M `).
5. Introspect: exactly `+verdict-snapshot` + `--vd4-snap-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + VD1-VD3 verdict.h code + warmhull.h/ecs.h + ALL sim headers + ALL shaders
   byte-unchanged; verdict.h APPEND-only; NO shader change). `vd4_snap` in the Mac loop + `--vd4-snap-shot` in
   `$vkShots`.
