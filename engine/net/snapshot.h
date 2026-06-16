#pragma once
// Slice BQ — Networking / Replication Snapshot Layer (Phase 4 #17). Pure CPU, deterministic state
// replication: an AUTHORITY serializes a per-tick SNAPSHOT of replicated entity state into bytes; a
// REPLICA reconstructs bit-identical state by applying those bytes (full keyframes + deltas between
// them). NO sockets / UDP / TCP — transport is a future slice. The authority's byte stream is fed
// directly into the replica via an in-process "perfect channel", so the same authority sim yields the
// same snapshots and the replica reconstructs identical state.
//
// HARD RULE: this module is PURE CPU above engine/math + engine/game + engine/physics. It has ZERO
// RHI / graphics-backend symbols (no vk*/MTL*/mtl::/Backend::Metal). It is compiled into BOTH hf_core
// (ASan-scoped, unit-tested in tests/replication_test.cpp) and hf_engine (the live --net-shot
// showcase that renders the replica's reconstructed scene through the existing lit/shadowed path).
//
// Determinism / wire format: fixed little-endian byte order, fixed field order. Floats are written as
// their raw IEEE-754 bit pattern (memcpy of the 4 bytes, LE) — since the authority and replica are the
// SAME build, this round-trips a float bit-exactly. (A future cross-machine transport slice can swap in
// fixed-point quantization; the snapshot/delta API stays the same.)
#include <cstdint>
#include <span>
#include <vector>

#include "math/math.h"

namespace hf::game { struct GameState; }
namespace hf::physics { struct World; }

namespace hf::net {

// The minimal replicated per-entity state. The roll-game's player + pickups map onto these: the player
// is one RepEntity (its physics body position+orientation), each uncollected pickup is one RepEntity.
// `flags` carries a small per-entity bitfield (e.g. kFlagPlayer / kFlagPickup) so a renderer can tint
// the right material without a side table.
struct RepEntity {
    uint32_t   id = 0;
    math::Vec3 position;
    math::Quat orientation = math::Quat::Identity();
    uint32_t   flags = 0;
};

// Per-entity flag bits (documented; stable wire values).
constexpr uint32_t kFlagPlayer = 1u << 0;  // the dynamic player body
constexpr uint32_t kFlagPickup = 1u << 1;  // a collectible pickup

// A full snapshot of the replicated world at one simulation tick. Entities are stored in a fixed,
// ascending-id order by the capture/serialize path so the wire bytes are deterministic.
struct Snapshot {
    uint32_t                tick = 0;
    std::vector<RepEntity>  entities;
};

// --- Equality (exact; used by the tests + Replicator::Matches) ----------------------------------
// RepEntity compares all four fields bit-exactly (raw float bits, no epsilon): the replica must
// reconstruct the authority EXACTLY, not approximately. Snapshot compares tick + entity list in order.
bool operator==(const RepEntity& a, const RepEntity& b);
inline bool operator!=(const RepEntity& a, const RepEntity& b) { return !(a == b); }
bool operator==(const Snapshot& a, const Snapshot& b);
inline bool operator!=(const Snapshot& a, const Snapshot& b) { return !(a == b); }

// --- Serialization (full snapshot <-> bytes) ----------------------------------------------------
// Deterministic, fixed little-endian, fixed field order. Layout:
//   u32 tick | u32 entityCount | entityCount * { u32 id; f32 px,py,pz; f32 qx,qy,qz,qw; u32 flags }
// (each entity is a fixed 40 bytes; header 8 bytes). Deserialize is the exact inverse.
std::vector<uint8_t> Serialize(const Snapshot& s);
Snapshot             Deserialize(std::span<const uint8_t> bytes);

// --- Delta compression (per-entity granularity) -------------------------------------------------
// DeltaEncode emits ONLY the entities that differ from `prev` (by id): entities ADDED or CHANGED are
// written in full (id+transform+flags), entities REMOVED are written as id-only remove markers. The
// granularity is PER-ENTITY (a changed entity is re-sent whole) — documented choice; a per-field
// bitmask is an explicit YAGNI for this slice. Layout:
//   u32 tick | u32 changedCount | u32 removedCount
//     | changedCount * { u32 id; f32 px,py,pz; f32 qx,qy,qz,qw; u32 flags }   (added OR changed)
//     | removedCount * { u32 id }                                              (present in prev, gone)
// Invariant: DeltaApply(prev, DeltaEncode(prev, curr)) == curr. An unchanged snapshot encodes to just
// the 12-byte header (changed=0, removed=0), which is strictly smaller than a full Serialize — the
// compression actually compresses.
std::vector<uint8_t> DeltaEncode(const Snapshot& prev, const Snapshot& curr);
Snapshot             DeltaApply(const Snapshot& prev, std::span<const uint8_t> delta);

// --- Interpolation (client-side smoothing between two received snapshots) ------------------------
// Per matched-id entity: lerp position, slerp/nlerp orientation (reuses math::Slerp). `alpha` is
// clamped to [0,1]. Entities present in only one of (a,b) are carried through unchanged at their own
// transform (no fade). The result tick is b.tick. Deterministic.
Snapshot Interpolate(const Snapshot& a, const Snapshot& b, float alpha);

// --- The Replicator: authority capture + channel + replica reconstruction ------------------------
// Authority side: Capture(tick, GameState, World) reads the roll-game's player body + uncollected
// pickups into a Snapshot (ascending id, player id 0, pickups id 1..N in pickup order). The channel
// serializes that snapshot — a FULL keyframe every `keyframeInterval` ticks (and the first packet),
// a per-entity DELTA against the last sent snapshot otherwise.
//
// Replica side: Receive(packet) applies a full or delta packet (auto-detected by a 1-byte tag) and
// updates the replica's latest Snapshot, exposed via State(). Matches(authority) is an exact compare.
class Replicator {
public:
    explicit Replicator(uint32_t keyframeInterval = 8) : keyframeInterval_(keyframeInterval) {}

    // --- Authority: build a Snapshot from the live roll-game state. -------------------------------
    static Snapshot Capture(uint32_t tick, const game::GameState& gs, const physics::World& world);

    // --- Channel (authority): serialize `snap` into a packet (full keyframe or delta vs the last
    // authority snapshot). Tracks the keyframe cadence + byte-savings counters. The first call always
    // emits a full keyframe. The returned bytes are what the replica's Receive consumes. ------------
    std::vector<uint8_t> Send(const Snapshot& snap);

    // --- Channel (replica): apply a packet (full or delta) -> updates State(). --------------------
    void Receive(std::span<const uint8_t> packet);

    const Snapshot& State() const { return replicaState_; }

    // Exact replica == authority compare.
    bool Matches(const Snapshot& authority) const { return replicaState_ == authority; }

    // --- Byte-savings + cadence counters (for the --net-shot stat line + tests). ------------------
    uint32_t SnapshotsSent()  const { return snapshotsSent_; }
    uint32_t KeyframesSent()  const { return keyframesSent_; }
    uint64_t FullBytes()      const { return fullBytes_; }    // bytes if EVERY snapshot were a full keyframe
    uint64_t DeltaBytes()     const { return deltaBytes_; }   // bytes actually sent (keyframe + deltas)

private:
    uint32_t keyframeInterval_;
    bool     haveAuthorityPrev_ = false;
    Snapshot authorityPrev_;     // last snapshot the authority serialized (delta base)
    Snapshot replicaState_;      // the replica's reconstructed latest snapshot

    uint32_t snapshotsSent_ = 0;
    uint32_t keyframesSent_ = 0;
    uint64_t fullBytes_  = 0;
    uint64_t deltaBytes_ = 0;
};

// Packet tag bytes (first byte of a Send() packet). FULL carries a serialized Snapshot; DELTA carries
// a DeltaEncode payload that the replica applies against its current State().
constexpr uint8_t kPacketFull  = 0xF0;
constexpr uint8_t kPacketDelta = 0xD0;

} // namespace hf::net
