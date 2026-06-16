#pragma once
// Slice BU — Networking Transport + Client Jitter-Buffer / Interpolation (Phase 4 #20). Pure CPU,
// deterministic. Extends BQ (engine/net/snapshot) with the HARD part of real networking — an
// IMPERFECT channel (latency + packet loss + reordering) + a client JITTER BUFFER that holds received
// snapshots and renders an INTERPOLATED view at a fixed interpolation delay (the standard technique
// that hides latency/loss with smooth motion). Still NO real sockets/UDP/TCP — the channel is modeled
// DETERMINISTICALLY via a SEEDED integer LCG (NO real RNG, NO clock), so the same seed + same packet
// stream yields the same delivered/dropped/reordered sequence and the client's reconstructed +
// interpolated state is bit-stable. (Real transport is a future slice.)
//
// HARD RULE: this module is PURE CPU above engine/math + engine/net (BQ Snapshot/Serialize/
// Deserialize/Interpolate). It has ZERO RHI / graphics-backend symbols (no vk*/MTL*/mtl::/
// Backend::Metal). It is compiled into BOTH hf_core (ASan-scoped, unit-tested in
// tests/net_transport_test.cpp) and hf_engine (the live --netsim-shot showcase that renders the
// client's interpolated scene through the existing lit/shadowed path).
//
// Determinism: the channel's drop/reorder/jitter decisions are drawn from a pure integer LCG seeded by
// `lossSeed` + the sender's monotonically increasing packet sequence number. Latency is a FIXED tick
// offset. Same config + same Send order => identical delivery. Client interpolation reuses BQ
// `Interpolate` (lerp position + nlerp/slerp orientation by the fractional alpha) — also deterministic.
#include <cstdint>
#include <map>
#include <span>
#include <vector>

#include "net/snapshot.h"

namespace hf::net {

// --- The simulated channel's configuration (all knobs deterministic). ----------------------------
// latencyTicks : the BASE one-way delay, in ticks, applied to every delivered packet (deliverTick =
//                sendTick + latencyTicks, plus an optional seeded jitter — see jitterTicks).
// lossSeed     : the LCG seed. Same seed + same Send stream => identical drop/reorder/jitter draws.
// dropRate     : probability [0..1] a sent packet is DROPPED (never delivered). 0 => lossless.
// reorderRate  : probability [0..1] a (non-dropped) packet is REORDERED — its deliverTick is perturbed
//                so it can arrive out of send order. 0 => in-order (subject only to the fixed latency).
// reorderSpread: the extra seeded delay (in ticks) added to a reordered packet's deliverTick, drawn in
//                [2 .. 2+reorderSpread]. The minimum of 2 guarantees a reordered packet falls BEHIND
//                the next in-stream packet (which arrives at +latency, i.e. +1 relative to this one's
//                send tick), so delivery order genuinely differs from send order — not merely a tie.
//                Documented, deterministic.
struct ChannelConfig {
    int      latencyTicks  = 0;
    uint32_t lossSeed      = 0;
    float    dropRate      = 0.0f;
    float    reorderRate   = 0.0f;
    int      reorderSpread = 3;  // reordered packets get +[2 .. 2+reorderSpread] ticks (deterministic)
};

// A queued packet inside the channel: the tick it should be DELIVERED at, the original SEND tick + a
// monotonic sequence number (for stable tie-breaking + diagnostics), and the opaque serialized bytes.
struct Packet {
    int                  deliverTick = 0;  // tick at/after which Deliver(tick) hands this to the client
    int                  sendTick    = 0;  // the tick Send() was called (diagnostic / latency check)
    uint32_t             seq         = 0;  // monotonic send-order index (stable tie-break)
    std::vector<uint8_t> bytes;            // serialized Snapshot (BQ Serialize)
};

// --- SimChannel: a deterministic, seeded, socket-free network channel. ---------------------------
// Send(tick, bytes) enqueues the packet with deliverTick = tick + latency (+ a seeded jitter on a
// reordered packet) UNLESS a seeded LCG draw < dropRate (then it is DROPPED + counted). Deliver(tick)
// returns every still-queued packet whose deliverTick <= tick, in deliverTick order (ties broken by
// send sequence), and removes them from the queue. Counters track delivered/dropped/reordered. The LCG
// is seeded by lossSeed and advanced per Send, so the SAME config + SAME Send stream is bit-identical.
class SimChannel {
public:
    explicit SimChannel(const ChannelConfig& cfg) : cfg_(cfg), lcg_(cfg.lossSeed) {}

    // Enqueue (or drop) one packet sent at `tick`. The bytes are an opaque serialized Snapshot. Returns
    // true if the packet was queued for delivery, false if it was dropped by the seeded loss model.
    bool Send(int tick, std::vector<uint8_t> bytes);

    // Hand the client every queued packet whose deliverTick <= `tick`, in deliverTick order (ties by
    // send sequence). Removes them from the queue. (Packets with deliverTick > tick stay queued.)
    std::vector<Packet> Deliver(int tick);

    // True once no packets remain queued (all delivered or the queue was drained).
    bool Empty() const { return queue_.empty(); }

    // Counters (for the showcase stat line + tests).
    uint32_t Sent()      const { return sent_; }       // total Send() calls
    uint32_t Delivered() const { return delivered_; }  // packets actually queued for delivery
    uint32_t Dropped()   const { return dropped_; }    // packets dropped by the loss model
    uint32_t Reordered() const { return reordered_; }  // packets whose deliverTick was perturbed

private:
    // Pure integer LCG (Numerical Recipes constants). Deterministic; NO real RNG, NO clock.
    uint32_t NextRand() {
        lcg_ = lcg_ * 1664525u + 1013904223u;
        return lcg_;
    }
    // A deterministic draw in [0,1): the top 24 bits of the LCG over 2^24.
    float NextUnit() { return (float)(NextRand() >> 8) / (float)(1u << 24); }

    ChannelConfig cfg_;
    uint32_t      lcg_;
    // The in-flight queue, ordered by (deliverTick, seq) so Deliver pops the front in delivery order.
    // A std::map keyed by (deliverTick, seq) keeps it sorted + deterministic without a manual re-sort.
    std::map<std::pair<int, uint32_t>, Packet> queue_;
    uint32_t nextSeq_   = 0;
    uint32_t sent_      = 0;
    uint32_t delivered_ = 0;
    uint32_t dropped_   = 0;
    uint32_t reordered_ = 0;
};

// --- ClientView: the jitter buffer + interpolated render state. ----------------------------------
// Receives delivered packets (deserialized Snapshots) into a buffer keyed by tick. RenderState renders
// at renderTick = latestReceivedTick - interpDelay, INTERPOLATED between the two buffered snapshots
// that bracket renderTick (reuse BQ Interpolate: lerp position + nlerp orientation by the fractional
// alpha). The fixed interpDelay (default 2 ticks) keeps the render point ~2 ticks behind the newest
// snapshot so a single dropped snapshot is bracketed by its neighbors and interpolated ACROSS (hidden,
// no pop). Gap policy is documented on RenderState.
class ClientView {
public:
    explicit ClientView(int interpDelay = 2) : interpDelay_(interpDelay) {}

    // Buffer a delivered packet's deserialized Snapshot (keyed by its tick). Out-of-order / duplicate
    // ticks are fine: the newest copy wins and latestReceivedTick tracks the max tick ever seen.
    void Receive(std::span<const uint8_t> bytes);
    void ReceiveSnapshot(const Snapshot& snap);

    // True once at least one snapshot has been buffered.
    bool HasData() const { return !buffer_.empty(); }

    // The newest snapshot tick ever received (the leading edge of the buffer).
    int LatestReceivedTick() const { return latestTick_; }

    // The tick the client RENDERS at: latestReceivedTick - interpDelay (clamped >= 0).
    float RenderTick() const;

    // The interpolated snapshot at `renderTick`. Policy:
    //   * renderTick between two buffered ticks lo<=renderTick<=hi: Interpolate(lo, hi, alpha) where
    //     alpha = (renderTick-lo)/(hi-lo). lo/hi are the nearest buffered ticks bracketing renderTick —
    //     so a single DROPPED snapshot (a one-tick hole) is spanned by its neighbors and interpolated
    //     ACROSS the gap (the smoothing HIDES the drop; no visible pop). With interpDelay>=2 the render
    //     point trails the leading edge by enough that the next snapshot fills the hole before we reach
    //     it.
    //   * renderTick at/below the oldest buffered tick (buffer edge / startup): HOLD the oldest
    //     snapshot (clamp). renderTick at/above the newest: HOLD the newest (we never EXTRAPOLATE past
    //     the last received snapshot — hold-last is the documented edge policy; bounded, never drifts).
    //   * empty buffer: returns a default (empty) Snapshot.
    Snapshot RenderState(float renderTick) const;

    // Convenience: RenderState at the client's own RenderTick().
    Snapshot RenderStateNow() const { return RenderState(RenderTick()); }

    int InterpDelay() const { return interpDelay_; }
    size_t BufferSize() const { return buffer_.size(); }

private:
    int interpDelay_;
    int latestTick_ = -1;
    std::map<int, Snapshot> buffer_;  // tick -> snapshot, ordered (lower_bound brackets renderTick)
};

}  // namespace hf::net
