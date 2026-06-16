// Slice BQ — replication snapshot layer implementation. See snapshot.h for the contract. Pure CPU
// above engine/math + engine/game + engine/physics; ZERO RHI / backend symbols. Deterministic:
// fixed little-endian byte order, fixed field order, raw-IEEE float bytes (same build round-trips
// bit-exactly). Compiled into BOTH hf_core (ASan, unit-tested) and hf_engine (the --net-shot showcase).
#include "net/snapshot.h"

#include "game/roll_game.h"
#include "physics/body.h"
#include "physics/world.h"

#include <algorithm>
#include <cstring>

namespace hf::net {

// --- Equality ------------------------------------------------------------------------------------
// Bit-exact comparison: the replica must reconstruct the authority EXACTLY (raw float bits, no
// epsilon). We compare the raw bytes of each float so two NaNs / signed zeros behave like the wire.
namespace {
bool BitsEqual(float a, float b) {
    uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    return ua == ub;
}
}  // namespace

bool operator==(const RepEntity& a, const RepEntity& b) {
    return a.id == b.id && a.flags == b.flags &&
           BitsEqual(a.position.x, b.position.x) &&
           BitsEqual(a.position.y, b.position.y) &&
           BitsEqual(a.position.z, b.position.z) &&
           BitsEqual(a.orientation.x, b.orientation.x) &&
           BitsEqual(a.orientation.y, b.orientation.y) &&
           BitsEqual(a.orientation.z, b.orientation.z) &&
           BitsEqual(a.orientation.w, b.orientation.w);
}

bool operator==(const Snapshot& a, const Snapshot& b) {
    if (a.tick != b.tick || a.entities.size() != b.entities.size()) return false;
    for (size_t i = 0; i < a.entities.size(); ++i)
        if (a.entities[i] != b.entities[i]) return false;
    return true;
}

// --- Little-endian primitive read/write helpers --------------------------------------------------
namespace {
void PutU32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back((uint8_t)(v & 0xFF));
    out.push_back((uint8_t)((v >> 8) & 0xFF));
    out.push_back((uint8_t)((v >> 16) & 0xFF));
    out.push_back((uint8_t)((v >> 24) & 0xFF));
}
void PutF32(std::vector<uint8_t>& out, float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);  // raw IEEE-754 bit pattern
    PutU32(out, u);
}
// A tiny bounds-checked cursor over the input span. On underrun it clamps to zero-filled reads so a
// truncated/garbage packet yields a well-defined (empty-ish) result rather than UB.
struct Reader {
    std::span<const uint8_t> b;
    size_t pos = 0;
    uint32_t U32() {
        uint32_t v = 0;
        if (pos + 4 <= b.size()) {
            v = (uint32_t)b[pos] | ((uint32_t)b[pos + 1] << 8) |
                ((uint32_t)b[pos + 2] << 16) | ((uint32_t)b[pos + 3] << 24);
        }
        pos += 4;
        return v;
    }
    float F32() {
        uint32_t u = U32();
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
    bool Ok() const { return pos <= b.size(); }
};

// Write / read one RepEntity's full record (id + transform + flags): 40 bytes, fixed order.
void PutEntity(std::vector<uint8_t>& out, const RepEntity& e) {
    PutU32(out, e.id);
    PutF32(out, e.position.x);
    PutF32(out, e.position.y);
    PutF32(out, e.position.z);
    PutF32(out, e.orientation.x);
    PutF32(out, e.orientation.y);
    PutF32(out, e.orientation.z);
    PutF32(out, e.orientation.w);
    PutU32(out, e.flags);
}
RepEntity GetEntity(Reader& r) {
    RepEntity e;
    e.id = r.U32();
    e.position.x = r.F32();
    e.position.y = r.F32();
    e.position.z = r.F32();
    e.orientation.x = r.F32();
    e.orientation.y = r.F32();
    e.orientation.z = r.F32();
    e.orientation.w = r.F32();
    e.flags = r.U32();
    return e;
}
}  // namespace

// --- Serialize / Deserialize (full snapshot) -----------------------------------------------------
std::vector<uint8_t> Serialize(const Snapshot& s) {
    std::vector<uint8_t> out;
    out.reserve(8 + s.entities.size() * 40);
    PutU32(out, s.tick);
    PutU32(out, (uint32_t)s.entities.size());
    for (const RepEntity& e : s.entities) PutEntity(out, e);
    return out;
}

Snapshot Deserialize(std::span<const uint8_t> bytes) {
    Reader r{bytes};
    Snapshot s;
    s.tick = r.U32();
    uint32_t count = r.U32();
    s.entities.reserve(count);
    for (uint32_t i = 0; i < count; ++i) s.entities.push_back(GetEntity(r));
    return s;
}

// --- Delta encode / apply (per-entity granularity) -----------------------------------------------
std::vector<uint8_t> DeltaEncode(const Snapshot& prev, const Snapshot& curr) {
    // Index prev by id for O(1) lookup. (ids are unique per snapshot by construction.)
    // Changed/added: every curr entity that is absent from prev OR differs from its prev twin.
    // Removed: every prev id absent from curr.
    std::vector<RepEntity> changed;
    std::vector<uint32_t>  removed;

    auto findIn = [](const Snapshot& s, uint32_t id) -> const RepEntity* {
        for (const RepEntity& e : s.entities) if (e.id == id) return &e;
        return nullptr;
    };

    for (const RepEntity& e : curr.entities) {
        const RepEntity* p = findIn(prev, e.id);
        if (!p || *p != e) changed.push_back(e);
    }
    for (const RepEntity& e : prev.entities) {
        if (!findIn(curr, e.id)) removed.push_back(e.id);
    }

    std::vector<uint8_t> out;
    out.reserve(12 + changed.size() * 40 + removed.size() * 4);
    PutU32(out, curr.tick);
    PutU32(out, (uint32_t)changed.size());
    PutU32(out, (uint32_t)removed.size());
    for (const RepEntity& e : changed) PutEntity(out, e);
    for (uint32_t id : removed) PutU32(out, id);
    return out;
}

Snapshot DeltaApply(const Snapshot& prev, std::span<const uint8_t> delta) {
    Reader r{delta};
    Snapshot curr;
    curr.tick = r.U32();
    uint32_t changedCount = r.U32();
    uint32_t removedCount = r.U32();

    std::vector<RepEntity> changed;
    changed.reserve(changedCount);
    for (uint32_t i = 0; i < changedCount; ++i) changed.push_back(GetEntity(r));
    std::vector<uint32_t> removed;
    removed.reserve(removedCount);
    for (uint32_t i = 0; i < removedCount; ++i) removed.push_back(r.U32());

    auto isRemoved = [&](uint32_t id) {
        return std::find(removed.begin(), removed.end(), id) != removed.end();
    };
    auto changedFor = [&](uint32_t id) -> const RepEntity* {
        for (const RepEntity& e : changed) if (e.id == id) return &e;
        return nullptr;
    };

    // Start from prev's entities (in order), dropping removed ones + applying in-place changes; then
    // append any genuinely NEW entities (changed entities whose id was absent from prev) in delta
    // order. This preserves prev's ordering for surviving entities and appends additions — matching
    // the authority's stable-id capture order.
    for (const RepEntity& e : prev.entities) {
        if (isRemoved(e.id)) continue;
        if (const RepEntity* c = changedFor(e.id)) curr.entities.push_back(*c);
        else curr.entities.push_back(e);
    }
    auto findIn = [](const Snapshot& s, uint32_t id) -> const RepEntity* {
        for (const RepEntity& e : s.entities) if (e.id == id) return &e;
        return nullptr;
    };
    for (const RepEntity& e : changed) {
        if (!findIn(prev, e.id)) curr.entities.push_back(e);  // a genuinely added entity
    }
    return curr;
}

// --- Interpolation -------------------------------------------------------------------------------
Snapshot Interpolate(const Snapshot& a, const Snapshot& b, float alpha) {
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    Snapshot out;
    // Tick follows alpha so the endpoints equal the inputs exactly: Interpolate(a,b,0)==a (a.tick),
    // Interpolate(a,b,1)==b (b.tick). Rounded linear blend in between.
    out.tick = (uint32_t)((float)a.tick + ((float)b.tick - (float)a.tick) * alpha + 0.5f);
    out.entities.reserve(b.entities.size());

    auto findIn = [](const Snapshot& s, uint32_t id) -> const RepEntity* {
        for (const RepEntity& e : s.entities) if (e.id == id) return &e;
        return nullptr;
    };

    // For each entity in b: if it also exists in a (by id), lerp position + slerp orientation;
    // otherwise carry b's entity through unchanged (newly-appeared entity, no fade).
    for (const RepEntity& eb : b.entities) {
        const RepEntity* ea = findIn(a, eb.id);
        if (!ea) { out.entities.push_back(eb); continue; }
        RepEntity e;
        e.id = eb.id;
        e.flags = eb.flags;
        e.position = ea->position + (eb.position - ea->position) * alpha;
        e.orientation = math::Slerp(ea->orientation, eb.orientation, alpha);
        out.entities.push_back(e);
    }
    // Carry through entities present only in a (disappeared in b) at their own transform.
    for (const RepEntity& ea : a.entities) {
        if (!findIn(b, ea.id)) out.entities.push_back(ea);
    }
    return out;
}

// --- Replicator: authority capture --------------------------------------------------------------
Snapshot Replicator::Capture(uint32_t tick, const game::GameState& gs, const physics::World& world) {
    Snapshot snap;
    snap.tick = tick;

    // Entity 0: the player body (position + orientation from the rigid body).
    if (gs.playerBodyIndex >= 0 && (size_t)gs.playerBodyIndex < world.bodies.size()) {
        const physics::RigidBody& body = world.bodies[(size_t)gs.playerBodyIndex];
        RepEntity player;
        player.id = 0;
        player.position = body.position;
        player.orientation = body.orientation;
        player.flags = kFlagPlayer;
        snap.entities.push_back(player);
    }

    // Entities 1..N: the UNCOLLECTED pickups, each keeping a STABLE id == 1 + its pickup index (so a
    // collected pickup drops out as a delta "remove" while the survivors keep their ids). Pickups are
    // axis-aligned (identity orientation).
    for (size_t i = 0; i < gs.pickups.size(); ++i) {
        const game::Pickup& p = gs.pickups[i];
        if (p.collected) continue;
        RepEntity e;
        e.id = (uint32_t)(1 + i);
        e.position = p.pos;
        e.orientation = math::Quat::Identity();
        e.flags = kFlagPickup;
        snap.entities.push_back(e);
    }
    return snap;
}

// --- Replicator: channel (authority Send) --------------------------------------------------------
std::vector<uint8_t> Replicator::Send(const Snapshot& snap) {
    // Decide: full keyframe on the first packet, every `keyframeInterval` snapshots, or when the
    // keyframe interval is degenerate (0 => always full). Otherwise a per-entity delta vs the last
    // authority snapshot.
    const bool firstPacket = !haveAuthorityPrev_;
    const bool cadenceKeyframe =
        (keyframeInterval_ == 0) || (snapshotsSent_ % keyframeInterval_ == 0);
    const bool sendFull = firstPacket || cadenceKeyframe;

    std::vector<uint8_t> packet;
    if (sendFull) {
        packet.push_back(kPacketFull);
        std::vector<uint8_t> body = Serialize(snap);
        packet.insert(packet.end(), body.begin(), body.end());
        ++keyframesSent_;
    } else {
        packet.push_back(kPacketDelta);
        std::vector<uint8_t> body = DeltaEncode(authorityPrev_, snap);
        packet.insert(packet.end(), body.begin(), body.end());
    }

    // Byte accounting: fullBytes_ is what we'd have sent if EVERY snapshot were a full keyframe
    // (the 1-byte tag + a full Serialize); deltaBytes_ is what we ACTUALLY sent.
    fullBytes_ += 1 + Serialize(snap).size();
    deltaBytes_ += packet.size();

    authorityPrev_ = snap;
    haveAuthorityPrev_ = true;
    ++snapshotsSent_;
    return packet;
}

// --- Replicator: channel (replica Receive) -------------------------------------------------------
void Replicator::Receive(std::span<const uint8_t> packet) {
    if (packet.empty()) return;
    uint8_t tag = packet[0];
    std::span<const uint8_t> body = packet.subspan(1);
    if (tag == kPacketFull) {
        replicaState_ = Deserialize(body);
    } else if (tag == kPacketDelta) {
        replicaState_ = DeltaApply(replicaState_, body);
    }
    // Unknown tag: ignore (keep prior state). A future transport layer validates upstream.
}

}  // namespace hf::net
