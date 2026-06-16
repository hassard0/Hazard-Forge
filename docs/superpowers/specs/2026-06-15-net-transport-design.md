# Slice BU — Networking Transport + Client Jitter-Buffer / Interpolation (Phase 4 #20) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Extends BQ
> (replication) with the hard part of real networking — an imperfect channel + client-side smoothing —
> made deterministic via a SEEDED channel model (still NO actual sockets; real transport = future).

**Goal:** Make the replica robust to an IMPERFECT channel. Add a deterministic simulated transport
(configurable latency + seeded packet loss + reordering) and a client JITTER BUFFER that holds received
snapshots and renders an INTERPOLATED view at a fixed interpolation delay (the standard technique that
hides latency/loss with smooth motion). Proven deterministic; the interpolated client view stays within
bounds of the authority. A `--netsim-shot` renders the client's interpolated view under a lossy/laggy
channel; golden-verified.

## Why a seeded channel (deterministic, headlessly verifiable)

Real packet loss/jitter is nondeterministic. Model the channel deterministically: a `SimChannel` queues
each sent packet with a delivery tick = sendTick + latency, DROPS it if a SEEDED LCG draw < dropRate, and
applies a seeded reorder. Same seed + same packet stream → same delivered/dropped/reordered sequence →
the client's reconstructed + interpolated state is bit-stable. (Real UDP/TCP transport is a future slice;
this verifies the loss/lag-HANDLING logic, which is the substance.)

## Design decisions (locked)

1. **Simulated transport (engine/net/transport.{h,cpp}, pure CPU, no backend symbols).** Namespace
   `hf::net`. Reuses BQ's `Snapshot`/`Serialize`.
   - `struct ChannelConfig { int latencyTicks; uint32_t lossSeed; float dropRate /*0..1*/;
     float reorderRate; };`
   - `class SimChannel` — `Send(tick, bytes)` enqueues `{deliverTick = tick + latency (+ a seeded jitter),
     bytes}` UNLESS a seeded LCG draw `< dropRate` (dropped, counted); a seeded draw `< reorderRate`
     perturbs `deliverTick` to reorder. `std::vector<Packet> Deliver(tick)` returns all packets whose
     `deliverTick <= tick`, in `deliverTick` order (ties by send order). Counters: delivered, dropped,
     reordered. Pure integer LCG seeded by `lossSeed` (NO real RNG, NO clock) → deterministic.

2. **Client jitter buffer + interpolation (engine/net/client.{h,cpp} or in transport).**
   - `class ClientView` — receives delivered packets (deserialized `Snapshot`s), keeps the most recent K
     in a buffer keyed by tick. `Snapshot RenderState(float renderTick)` returns the snapshot at
     `renderTick = latestReceivedTick - interpDelay`, INTERPOLATED between the two buffered snapshots that
     bracket `renderTick` (reuse BQ `Interpolate` — lerp position + nlerp orientation by the fractional
     alpha). If a snapshot is missing (dropped), interpolate across the GAP (extrapolate/hold per a
     documented policy — e.g. hold-last or extrapolate; pick + document). Deterministic.
   - The `interpDelay` (in ticks) is fixed/configurable; document the default (e.g. 2 ticks so a single
     drop is covered by the next snapshot).

3. **Showcase `--netsim-shot <out>` (Vulkan) / `--netsim` (Metal).** Authority = the AX roll-game over the
   scripted track; each tick `Capture` → serialize → `SimChannel.Send` (with a FIXED ChannelConfig:
   latency e.g. 3 ticks, dropRate e.g. 0.15, a fixed lossSeed, reorderRate e.g. 0.1). Each tick the client
   `Deliver`s + buffers, and at a FIXED render tick renders its INTERPOLATED `RenderState` (player +
   pickups, lit+shadowed). Print `netsim: {latency:3, lossRate:0.15, delivered:D, dropped:X, reordered:R,
   interpDelay:2, renderTick:..., converged:true}` where `converged` asserts the interpolated client state
   is within a documented tolerance of the authority's true state at that tick (the smoothing hides the
   loss). New golden `tests/golden/metal/netsim.png` (Metal two runs DIFF 0.0000). Existing 44 image
   goldens UNTOUCHED.

4. **Tests `tests/net_transport_test.cpp` (pure CPU, no GPU):**
   - **Channel determinism:** the same (config, send stream) yields identical delivered/dropped/reordered
     sequences across two runs; delivered+dropped == sent; latency respected (a packet sent at T delivers
     at >= T+latency).
   - **Loss/reorder:** with dropRate>0 some packets are dropped (counted); with reorderRate>0 the delivery
     order differs from send order at least once for a long stream; with dropRate=0/reorderRate=0/latency=L
     it's a pure FIFO delay.
   - **Jitter buffer + interpolation:** with a clean channel (no loss), `RenderState(renderTick)` equals
     the authority snapshot at `renderTick` (within the interp alpha); a single dropped snapshot is hidden
     (the bracketing snapshots interpolate across the gap; the result stays within tolerance of the
     authority's true value, NOT a visible pop); the documented gap policy holds at the buffer edge.
   - **Convergence/bounds:** over the full lossy stream, the client's interpolated state stays within the
     documented tolerance of the authority (no unbounded drift); deterministic.
   - Clean under `windows-msvc-asan`.

5. **Introspect.** Add exactly `network-transport-sim` (features) + `--netsim-shot` (showcases).

## RHI seam additions (summary)
- **None.** Pure CPU transport + interpolation; the `--netsim-shot` render reuses the lit/shadowed scene
  path (like `--net-shot`). New files (`engine/net/transport.{h,cpp}`, `tests/net_transport_test.cpp`) add
  ZERO backend symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Actual sockets/UDP/TCP, client-side PREDICTION + server reconciliation (a future slice — this slice is the
channel + interpolation/smoothing half), ack/nack reliability + retransmission, congestion control,
bit-level quantization, delta-on-top-of-transport (BQ already has delta; this models the channel over
whole snapshots for clarity — note it), lag compensation/rewind on the server, multiple clients. One
deterministic seeded channel + a client jitter buffer + interpolation, golden + bounds-asserted.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 44) + new `net_transport_test` (channel
   determinism, loss/reorder, jitter-buffer interpolation incl. a hidden drop, convergence/bounds). Clean
   under `windows-msvc-asan`.
2. `--netsim-shot` on Windows/Vulkan: controller visual review — the client's interpolated scene (player +
   remaining pickup at the render tick) renders coherently despite the lossy channel; the `netsim: {...
   converged:true}` line is deterministic (two runs → byte-identical capture). Run under the AT
   Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --netsim` → new golden `tests/golden/metal/netsim.png`; two runs DIFF 0.0000; the
   netsim stat line (delivered/dropped/reordered/converged) matches Vulkan.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `netsim.png` added;
   the other 44 byte-identical.
5. Introspect JSON rebaked exactly `+network-transport-sim` + `--netsim-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `netsim` image
   golden in the Mac round-trip loop.
