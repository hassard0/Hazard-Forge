# Slice BQ — Networking / Replication Snapshot Layer (Phase 4 #17, depth pivot #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A big-ticket
> UE5-parity gap (replication), made headlessly verifiable via DETERMINISTIC snapshot/delta round-trip —
> NO actual sockets (transport is a future slice, like the windowed-Metal / WASAPI paths).

**Goal:** A snapshot-based state replication foundation. An AUTHORITY simulates the deterministic
roll-game (Slice AX) and emits, each tick, a SNAPSHOT of replicated state; a REPLICA reconstructs the
state by applying snapshots (and deltas between them). The proof is that the replica's reconstructed state
EQUALS the authority's after applying the stream — plus delta-vs-full byte savings — all deterministic
and unit-tested. A `--net-shot` renders the REPLICA's reconstructed scene; golden-verified.

## Why no sockets (deterministic, headlessly verifiable)

Real transport (UDP/TCP, loss, reordering) is nondeterministic and a separate concern. The REPLICATION
CORE — snapshot serialization, delta compression, applying snapshots to a replica, and interpolation — is
pure, deterministic logic that we verify by feeding the authority's snapshot byte stream directly into the
replica (an in-process "perfect channel"). Same authority sim → same snapshots → replica reconstructs
bit-identical state. Transport/jitter-buffer/reliability are explicitly OUT OF SCOPE (future slice).

## Design decisions (locked)

1. **Replicated state + snapshot (engine/net/snapshot.{h,cpp}, pure CPU, no backend symbols).** Namespace
   `hf::net`.
   - `struct RepEntity { uint32_t id; math::Vec3 position; math::Quat orientation; uint32_t flags; };`
     (the minimal replicated per-entity state; the roll-game's player + pickups map to these).
   - `struct Snapshot { uint32_t tick; std::vector<RepEntity> entities; };`
   - `std::vector<uint8_t> Serialize(const Snapshot&)` / `Snapshot Deserialize(span<const uint8_t>)` —
     deterministic, fixed-endian (little), fixed field order. (Use a fixed-point or raw-IEEE encoding —
     since the authority + replica are the SAME build, raw float bytes round-trip exactly; document.)
   - `std::vector<uint8_t> DeltaEncode(const Snapshot& prev, const Snapshot& curr)` — encode ONLY entities
     whose `RepEntity` changed vs `prev` (by id), plus add/remove markers; a per-entity changed-field
     bitmask is optional (document the granularity: per-entity or per-field). `Snapshot DeltaApply(const
     Snapshot& prev, span<const uint8_t> delta)` — reconstruct `curr` from `prev` + delta. Invariant:
     `DeltaApply(prev, DeltaEncode(prev, curr)) == curr`.

2. **Replicator (authority + replica).** `class Replicator`:
   - Authority side: `Snapshot Capture(tick, const game::GameState&, const physics::World&)` — read the
     roll-game's player body + pickups into a `Snapshot`.
   - Channel: the authority serializes a snapshot (full keyframe every N ticks, delta otherwise) into
     bytes; the replica consumes them.
   - Replica side: `void Receive(span<const uint8_t> packet)` (apply full or delta) → maintains the
     replica's latest `Snapshot`; `const Snapshot& State() const`.
   - `bool Matches(const Snapshot& authority) const` — replica state == authority state (exact).
   - **Interpolation:** `Snapshot Interpolate(const Snapshot& a, const Snapshot& b, float alpha)` —
     per-entity lerp position + slerp/nlerp orientation (reuse the anim/math Quat slerp). Deterministic.

3. **Showcase `--net-shot <out>` (Vulkan) / `--net` (Metal).** Run the AX roll-game as the AUTHORITY for
   the fixed scripted track; each tick `Capture` → serialize (keyframe every N + deltas) → the REPLICA
   `Receive`s. At a FIXED tick (documented), assert `replica.State() == authority snapshot` and RENDER THE
   REPLICA'S reconstructed scene (player + remaining pickups from the replica's RepEntities, on the ground
   + lit/shadowed) → capture. Print `net: {ticks:T, snapshots:S, fullBytes:F, deltaBytes:D, savings:P%,
   replicaMatch:true}`. New golden `tests/golden/metal/net.png` (Metal two runs DIFF 0.0000). Existing 41
   image goldens UNTOUCHED. (The replica's render should look like the game scene at that tick — visually
   confirms the replica reconstructs a renderable world.)

4. **Tests `tests/replication_test.cpp` (pure CPU, no GPU):**
   - **Snapshot round-trip:** `Deserialize(Serialize(s)) == s` (incl. entity count, ids, transforms,
     flags) for a hand-built snapshot.
   - **Delta correctness:** `DeltaApply(prev, DeltaEncode(prev, curr)) == curr` for (a) no change, (b)
     one entity moved, (c) entity added, (d) entity removed; and `DeltaEncode` of an unchanged snapshot is
     SMALLER than a full serialize (compression actually compresses).
   - **Replica==authority over a stream:** feed a sequence of authority snapshots (keyframe + deltas)
     through the Replicator; after each Receive, `replica.State() == authority[i]` exactly; running twice
     is deterministic.
   - **Interpolation:** `Interpolate(a, b, 0) == a`, `== b at 1`, midpoint lerps position halfway +
     orientation normalized.
   - **Roll-game capture:** `Capture` of a known GameState yields the expected RepEntities (player +
     pickups, right ids/positions).
   - Clean under `windows-msvc-asan`.

5. **Introspect.** Add exactly `state-replication` (features) + `--net-shot` (showcases).

## RHI seam additions (summary)
- **None.** Pure CPU serialization/replication; the `--net-shot` render reuses the existing lit/shadowed
  scene path (like the roll-game showcase). New files (`engine/net/snapshot.{h,cpp}`,
  `tests/replication_test.cpp`) add ZERO backend symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Actual sockets / UDP / TCP / transport, packet loss / reordering / reliability / ack-nack, a jitter
buffer, client-side prediction + server reconciliation, lag compensation, interest management / relevancy,
bit-packing/quantization beyond the simple delta, encryption, RPCs, authority migration, lock-step. One
deterministic in-process snapshot+delta+replica+interpolation core, proven replica==authority, golden +
state-asserted.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 41) + new `replication_test` (round-trip, delta
   correctness + compression, replica==authority stream, interpolation, roll-game capture). Clean under
   `windows-msvc-asan`.
2. `--net-shot` on Windows/Vulkan: controller visual review — the REPLICA's reconstructed scene (player +
   remaining pickup at the fixed tick) renders coherently (matching the game scene), lit + shadowed; the
   `net: {... replicaMatch:true, savings:P%}` line is deterministic (two runs → byte-identical capture).
   Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --net` → new golden `tests/golden/metal/net.png`; two runs DIFF 0.0000; the net
   stat line (incl. replicaMatch:true + byte counts) matches Vulkan.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `net.png` added; the
   other 41 byte-identical.
5. Introspect JSON rebaked exactly `+state-replication` + `--net-shot`; introspect test updated; no other
   drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `net` image
   golden in the Mac round-trip loop.
