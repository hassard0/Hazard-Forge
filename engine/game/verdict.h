#pragma once
// Slice VD1 — Deterministic Gameplay / Netcode: THE ENTITY WORLD + THE INPUT-COMMAND BUS (the beachhead of
// FLAGSHIP #27: DETERMINISTIC GAMEPLAY / NETCODE PRODUCT LAYER, hf::game::verdict). The engine has a complete
// deterministic Q16.16 simulation suite (every sim bit-identical CPU/Vulkan/Metal + lockstep/rollback-replayable)
// and an ECS (engine/ecs/ecs.h) — but the ECS has NO determinism contract: ecs::Registry::create() recycles
// indices via a free-list + bumps generations, and view<> (ecs.h:248) drives off the SMALLEST pool in dense
// (insertion) order, neither a pinned cross-peer/cross-platform wire order. VD1 LAYERS a determinism contract ON
// TOP of the raw ECS, WITHOUT modifying it:
//
//   * EntityId = a monotonic uint32 (start 1, NEVER recycled within a match) — the PINNED sim identity. It is a
//     pure function of the command stream (a kCmdSpawn at tick T in array order -> nextId++), so it is identical
//     on every peer + survives rollback. The ecs::Entity handle (whose free-list/generation churn is
//     non-deterministic-by-design) is an implementation detail behind VerdictWorld::handle[EntityId].
//   * order[] = the live-entity id list in spawn order — the pinned iteration sequence (VD2's OrderedView uses it;
//     VD1 only MAINTAINS it).
//   * Command generalizes convex::ConvexCommand {tick,kind,bodyId,arg} to {tick,kind,target,arg} carrying gameplay
//     verbs (kCmdSpawn/kCmdDespawn/kCmdMove/kCmdAbility) UNION the lowered sim verbs (kCmdImpulse/kCmdSetAngVel,
//     which LowerToHullCommands maps straight to convex::ConvexCommand for gjk::ApplyHullCommands).
//   * ApplyCommands(world, commands, tick) applies only the commands whose .tick == tick, in FIXED array order,
//     BEFORE any step (the convex::ApplyConvexCommands contract, convex.h:1000): a command whose target is dead /
//     out-of-range is a deterministic NO-OP (never a crash, never undefined). Pure integer, fixed order ->
//     bit-identical on every peer/platform.
//
// VD1 is PURE CPU INTEGER (the strictest determinism tier, strict 0px). It is the DATA + the BUS only — it does
// NOT yet run gameplay systems (VD2) or step the physics (VD3 — VD1 carries the sim fields but does NOT step
// them). ALL component/world state is Q16.16 integer (NO float anywhere in the world state).
//
// Header-only, namespace hf::game::verdict. #includes sim/warmhull.h + ecs/ecs.h READ-ONLY/BYTE-FROZEN; verdict.h
// is a brand-new additive sibling that NEVER edits a frozen header or a shader. NO new render RHI, NO new shader,
// NO new compute.

#include <cstdint>
#include <cstdio>              // DX5: std::snprintf (the 16-hex digest)
#include <cstdlib>             // DX5: std::strtoll / std::free (ParseReplay over json.h)
#include <cstring>             // DX5: std::strcmp (ParseReplay member lookup)
#include <string>              // DX5: std::string (DigestSnapshot / Serialize / ParseReplay)
#include <unordered_map>
#include <vector>

#include "ecs/ecs.h"           // read-only: ecs::Registry / ecs::Entity / create/get/has/remove (the sparse-set
                               // store VD1 layers the determinism contract over — NOT modified)
#include "json/json.h"         // DX5 ONLY: the vendored single-header JSON parser (json_weak/inline C TU) for
                               // ParseReplay — the scene_io.cpp parser, reused; VD1-VD4 do NOT use it
#include "sim/warmhull.h"      // read-only: warmhull::HullCache / warmhull::HullSleepState (the embedded sim
                               // subsystem fields VD1 carries but does NOT step — VD3 steps them); transitively
                               // gjk::HullWorld + convex::ConvexCommand + the fpx Q16.16 toolbox (FxVec3/FxQuat)

namespace hf::game {
namespace verdict {

// The frozen sim namespaces (all under hf::sim) — alias them locally so the rest of the header reads like the sim
// headers (fpx::/gjk::/convex::/warmhull::) without re-qualifying hf::sim:: everywhere.
namespace fpx      = hf::sim::fpx;
namespace gjk      = hf::sim::gjk;
namespace convex   = hf::sim::convex;
namespace warmhull = hf::sim::warmhull;
namespace ecs      = hf::ecs;

// Pull the frozen helpers into this namespace (REUSE, do NOT redefine).
using fpx::fx;
using fpx::kOne;
using fpx::kFrac;
using fpx::FxVec3;
using fpx::FxQuat;

// ----- EntityId: the PINNED, monotonic gameplay identity (start 1, NEVER recycled within a match) ------------
// A pure function of the command stream — a kCmdSpawn at tick T in array order allocates nextId++. The 0 value is
// reserved as "no entity" (kCmdSpawn never returns it). Distinct from the ecs::Entity index (which recycles).
using EntityId = uint32_t;
inline constexpr EntityId kNoEntity = 0u;

// ----- Integer-only components (Q16.16 ONLY — NO float anywhere in world state) ------------------------------
// Transform2D — the pinned 2D-plane pose (a Q16.16 position + a Q16.16 quaternion orientation). The render
// projects pos>>kFrac to a pixel; the orient is carried for VD2/VD3 (kept integer so the whole world state is
// bit-reproducible).
struct Transform2D {
    FxVec3 pos;             // Q16.16 world position
    FxQuat orient;          // Q16.16 orientation quaternion (default identity {0,0,0,kOne})
};

// Health — an integer hit-point counter (a gameplay component; kCmdAbility writes it).
struct Health {
    int32_t hp = 0;
};

// BodyRef — the index of this entity's body in the embedded gjk::HullWorld (the sim<->gameplay bridge;
// LowerToHullCommands reads it to target a convex::ConvexCommand). kNoBody == not yet bound to the sim.
struct BodyRef {
    uint32_t simBodyIndex = 0u;
};
inline constexpr uint32_t kNoBody = ~0u;

// ----- Command: the generalization of convex::ConvexCommand {tick,kind,bodyId,arg} -> {tick,kind,target,arg} ---
// One stream drives the whole world. kind = one of the kCmd* verbs below. target is an EntityId (NOT a body
// index — the sim verbs are LOWERED to a body index by LowerToHullCommands via the target's BodyRef). arg is the
// Q16.16 payload (a move delta / impulse delta-velocity / angular velocity / an ability scalar).
struct Command {
    uint32_t tick   = 0;   // the tick this command fires at (applied BEFORE the step, the ApplyConvexCommands rule)
    uint32_t kind   = 0;   // one of the kCmd* verbs
    EntityId target = kNoEntity;  // the gameplay target (an EntityId; kCmdSpawn ignores it)
    FxVec3   arg;          // the Q16.16 payload
};

// The verb set: gameplay verbs (kCmdSpawn..kCmdAbility) UNION the sim verbs (kCmdImpulse/kCmdSetAngVel, lowered to
// convex::ConvexCommand). FIXED numbering — the wire contract.
inline constexpr uint32_t kCmdSpawn     = 0u;   // allocate a NEW entity (nextId++) + append to order[] (target ignored)
inline constexpr uint32_t kCmdDespawn   = 1u;   // remove the target from order[] + free its ecs handle (id NEVER recycled)
inline constexpr uint32_t kCmdMove      = 2u;   // target.Transform2D.pos += arg (a gameplay move delta)
inline constexpr uint32_t kCmdAbility   = 3u;   // target.Health.hp += arg.x>>kFrac (a gameplay ability — an integer hp delta)
inline constexpr uint32_t kCmdImpulse   = 4u;   // SIM: lowered -> convex::kConvexCmdAddImpulse on the target's BodyRef
inline constexpr uint32_t kCmdSetAngVel = 5u;   // SIM: lowered -> convex::kConvexCmdSetAngVel on the target's BodyRef

// ----- VerdictWorld: the deterministic entity world ----------------------------------------------------------
// The raw ecs::Registry is the STORE; the determinism contract rides on top: `handle` maps the pinned EntityId to
// its (recyclable, non-deterministic) ecs::Entity handle; `order` is the spawn-order live-id list; `nextId` is the
// monotonic id allocator (NEVER decremented, NEVER recycled). `sim`/`cache`/`sleep` are the embedded Q16.16 sim
// subsystem (VD1 carries them but does NOT step them — VD3). `tick` is the world clock.
struct VerdictWorld {
    ecs::Registry reg;                                  // the sparse-set store (read-only contract over it)
    std::unordered_map<EntityId, ecs::Entity> handle;   // EntityId -> the live ecs handle (impl detail)
    std::vector<EntityId> order;                        // the live-entity id list in spawn order (the pinned seq)
    EntityId nextId = 1u;                               // the monotonic id allocator (NEVER recycled)

    gjk::HullWorld sim;                                 // the embedded Q16.16 sim (VD1 carries; VD3 steps)
    warmhull::HullCache cache;                          // the warm-start cache (VD1 carries; VD3 steps)
    std::vector<warmhull::HullSleepState> sleep;        // per-body sleep state (VD1 carries; VD3 steps)

    uint32_t tick = 0;                                  // the world clock
};

// ----- IsLive(world, id): is `id` a currently-live entity (in order[] / handle map + a valid ecs handle)? -----
inline bool IsLive(const VerdictWorld& world, EntityId id) {
    if (id == kNoEntity) return false;
    auto it = world.handle.find(id);
    if (it == world.handle.end()) return false;
    return world.reg.valid(it->second);
}

// ----- SpawnEntity(world, components...) -> EntityId: deterministic monotonic id alloc -----------------------
// Allocate the next monotonic EntityId (nextId++), create a fresh ecs::Entity handle, register it in handle[],
// push the id onto order[] (spawn order), and add the given components to the ecs pools. The EntityId is a pure
// function of how many spawns have happened — independent of the ecs::Registry free-list recycle, so a
// spawn->despawn->spawn re-derives the SAME next id on every peer. Returns the new EntityId.
inline EntityId SpawnEntity(VerdictWorld& world, const Transform2D& xf, const Health& health,
                            const BodyRef& body) {
    const EntityId id = world.nextId++;            // the pinned, never-recycled allocation
    const ecs::Entity e = world.reg.create();      // the recyclable ecs handle (impl detail behind `id`)
    world.handle[id] = e;
    world.order.push_back(id);
    world.reg.add<Transform2D>(e, xf);
    world.reg.add<Health>(e, health);
    world.reg.add<BodyRef>(e, body);
    return id;
}

// Convenience: spawn with just a transform (Health 0, BodyRef unbound) — the common gameplay spawn.
inline EntityId SpawnEntity(VerdictWorld& world, const Transform2D& xf) {
    return SpawnEntity(world, xf, Health{0}, BodyRef{kNoBody});
}

// ----- DespawnEntity(world, id): remove from order[] + free the ecs handle, but NEVER recycle the id ----------
// Erase `id` from order[] (preserving the spawn order of the survivors), destroy its ecs::Entity (which the ecs
// free-list MAY recycle — that churn is invisible to the pinned EntityId contract), and drop it from handle[].
// `nextId` is UNCHANGED — the next spawn still allocates a fresh monotonic id, so a despawned id is NEVER reused
// within the match (the pinned-identity guarantee). A dead/unknown id is a deterministic no-op.
inline void DespawnEntity(VerdictWorld& world, EntityId id) {
    auto it = world.handle.find(id);
    if (it == world.handle.end()) return;          // unknown -> no-op (deterministic)
    world.reg.destroy(it->second);                 // free the ecs handle (the free-list may recycle the INDEX)
    world.handle.erase(it);
    for (size_t i = 0; i < world.order.size(); ++i) {
        if (world.order[i] == id) {
            world.order.erase(world.order.begin() + (std::ptrdiff_t)i);   // keep the survivors' spawn order
            break;
        }
    }
    // nextId is NOT touched — the id is retired, NEVER recycled.
}

// ----- LowerToHullCommands(commands, tick): the sim-verb subset -> std::vector<convex::ConvexCommand> ---------
// For every command of this tick (in FIXED array order) whose kind is a SIM verb (kCmdImpulse/kCmdSetAngVel),
// emit the matching convex::ConvexCommand with bodyId = the target entity's BodyRef.simBodyIndex (resolved
// against the live world), mapping kCmdImpulse->convex::kConvexCmdAddImpulse and
// kCmdSetAngVel->convex::kConvexCmdSetAngVel. A dead/unbound target is SKIPPED (no convex command emitted — the
// deterministic guard; gjk::ApplyHullCommands would no-op an out-of-range bodyId anyway, but a dead target has no
// body to lower to). Gameplay verbs (kCmdSpawn/Despawn/Move/Ability) produce NO convex command. The result is the
// frozen sim command stream gjk::ApplyHullCommands consumes — the bus does NOT diverge from the sim contract.
inline std::vector<convex::ConvexCommand> LowerToHullCommands(const VerdictWorld& world,
                                                              const std::vector<Command>& commands,
                                                              uint32_t tick) {
    std::vector<convex::ConvexCommand> out;
    for (size_t c = 0; c < commands.size(); ++c) {
        const Command& cmd = commands[c];
        if (cmd.tick != tick) continue;
        if (cmd.kind != kCmdImpulse && cmd.kind != kCmdSetAngVel) continue;  // not a sim verb -> not lowered
        // Resolve the target's body index (a dead/unbound target -> skip, the deterministic guard).
        auto it = world.handle.find(cmd.target);
        if (it == world.handle.end()) continue;
        const ecs::Entity e = it->second;
        if (!world.reg.valid(e) || !world.reg.has<BodyRef>(e)) continue;
        const uint32_t bodyId = world.reg.get<BodyRef>(e).simBodyIndex;
        if (bodyId == kNoBody) continue;            // unbound -> nothing to drive in the sim
        const uint32_t kind = (cmd.kind == kCmdImpulse) ? convex::kConvexCmdAddImpulse
                                                        : convex::kConvexCmdSetAngVel;
        out.push_back(convex::ConvexCommand{tick, kind, bodyId, cmd.arg});
    }
    return out;
}

// ----- ApplyCommands(world, commands, tick): the unified input-command bus apply -----------------------------
// Apply, in the commands' FIXED array order, every command whose .tick == `tick`, BEFORE any step (the
// convex::ApplyConvexCommands contract). kCmdSpawn allocates a monotonic id + appends to order[]; kCmdDespawn
// removes from order[] + frees the ecs handle (id NEVER recycled); kCmdMove/kCmdAbility write component state; the
// sim verbs (kCmdImpulse/kCmdSetAngVel) are LOWERED to convex::ConvexCommand (LowerToHullCommands) — VD1 lowers
// them but does NOT step the sim (VD3), so they are a no-op on the world STATE here (the lowering is proven by the
// LowerToHullCommands == hand-written-stream proof). A command whose `target` is DEAD / out-of-range is a
// deterministic NO-OP (never a crash, never undefined). Pure integer, fixed order -> bit-identical every peer.
inline void ApplyCommands(VerdictWorld& world, const std::vector<Command>& commands, uint32_t tick) {
    for (size_t c = 0; c < commands.size(); ++c) {
        const Command& cmd = commands[c];
        if (cmd.tick != tick) continue;            // tick-filtered (only this tick's commands)

        if (cmd.kind == kCmdSpawn) {
            // Spawn a fresh entity AT arg (Health 0, BodyRef unbound). The pos rides in arg.
            SpawnEntity(world, Transform2D{cmd.arg, FxQuat{0, 0, 0, kOne}});
            continue;
        }
        if (cmd.kind == kCmdDespawn) {
            DespawnEntity(world, cmd.target);      // dead/unknown -> no-op inside DespawnEntity
            continue;
        }

        // Verbs below all TARGET a live entity. A dead/out-of-range target is a deterministic NO-OP.
        auto it = world.handle.find(cmd.target);
        if (it == world.handle.end()) continue;    // out-of-range -> no-op
        const ecs::Entity e = it->second;
        if (!world.reg.valid(e)) continue;         // dead handle -> no-op

        if (cmd.kind == kCmdMove) {
            if (world.reg.has<Transform2D>(e)) {
                Transform2D& xf = world.reg.get<Transform2D>(e);
                xf.pos = fpx::FxAdd(xf.pos, cmd.arg);   // integer move delta
            }
        } else if (cmd.kind == kCmdAbility) {
            if (world.reg.has<Health>(e)) {
                Health& h = world.reg.get<Health>(e);
                h.hp += (int32_t)(cmd.arg.x >> kFrac);  // an integer hp delta (the Q16.16 arg.x un-fixed)
            }
        }
        // kCmdImpulse / kCmdSetAngVel: lowered to the sim by LowerToHullCommands; VD1 does NOT step the sim, so
        // there is no world-STATE mutation here (VD3 wires LowerToHullCommands -> gjk::ApplyHullCommands).
    }
}

// ----- VerdictMeasure / MeasureVerdict: the deterministic world-state summary (the proofs read this) ----------
// A compact, byte-comparable digest of the determinism-relevant world state: the live-entity count, nextId, and a
// fixed-order fold over order[] + the live Transform2D positions (so two runs of a fixed script can be compared
// byte-for-byte, and a single integer captures the whole world). Pure integer, fixed order.
struct VerdictMeasure {
    uint32_t entities = 0;   // order.size() (live entity count)
    EntityId nextId   = 1u;  // the monotonic allocator value
    uint64_t orderHash = 0;  // a fixed-order fold over order[] (the id sequence)
    uint64_t stateHash = 0;  // a fixed-order fold over the live Transform2D positions + Health
};

inline VerdictMeasure MeasureVerdict(const VerdictWorld& world) {
    VerdictMeasure m;
    m.entities = (uint32_t)world.order.size();
    m.nextId   = world.nextId;
    // A deterministic FNV-1a-style fold (fixed shifts/xors, NO products that could overflow surprisingly) over the
    // pinned order[] sequence + each live entity's position/health, in order[] order.
    uint64_t oh = 1469598103934665603ull;
    uint64_t sh = 1469598103934665603ull;
    auto mix = [](uint64_t h, uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
        return h;
    };
    for (size_t i = 0; i < world.order.size(); ++i) {
        const EntityId id = world.order[i];
        oh = mix(oh, (uint64_t)id);
        auto it = world.handle.find(id);
        if (it == world.handle.end()) continue;
        const ecs::Entity e = it->second;
        if (!world.reg.valid(e)) continue;
        if (world.reg.has<Transform2D>(e)) {
            const Transform2D& xf = world.reg.get<Transform2D>(e);
            sh = mix(sh, (uint64_t)(uint32_t)xf.pos.x);
            sh = mix(sh, (uint64_t)(uint32_t)xf.pos.y);
            sh = mix(sh, (uint64_t)(uint32_t)xf.pos.z);
        }
        if (world.reg.has<Health>(e)) {
            sh = mix(sh, (uint64_t)(uint32_t)world.reg.get<Health>(e).hp);
        }
    }
    m.orderHash = oh;
    m.stateHash = sh;
    return m;
}

// ----- VerdictMeasuresEqual(a, b): byte-for-byte equality of two world summaries (the two-run proof) ----------
inline bool VerdictMeasuresEqual(const VerdictMeasure& a, const VerdictMeasure& b) {
    return a.entities == b.entities && a.nextId == b.nextId && a.orderHash == b.orderHash &&
           a.stateHash == b.stateHash;
}

// =================================================================================================
// Slice VD2 — THE SYSTEM SCHEDULE + THE GAMEPLAY TICK (APPEND-ONLY; VD1 above is BYTE-FROZEN).
//
// VD1 built the deterministic entity world (a pinned monotonic EntityId, a fixed order[], integer-only
// components) + the unified input-command bus (ApplyCommands). VD2 adds the part that makes it a GAME:
// deterministic gameplay SYSTEMS running on a PINNED schedule over a PINNED iteration order.
//
// THE DETERMINISM CRUX VD2 FIXES: the raw ecs::Registry::view<>() (ecs.h:248 View::Resolve) drives off
// the SMALLEST matching pool in dense (insertion) order, which renumbers under entity add/remove churn —
// so two peers iterating that way could visit entities in a DIFFERENT sequence and diverge when a system
// mutates state while iterating. VD2's OrderedView/ForEachOrdered iterate STRICTLY in world.order[]
// (spawn order — the VD1 pinned live-id list), so every system visits every entity in the IDENTICAL
// sequence on every peer/platform. The gameplay rules are PURE INTEGER (Q16.16, NO float, NO sqrt — a
// distance² compare). The schedule order is a FIXED, hand-ordered, documented sequence. PURE CPU.
// =================================================================================================

// ----- VD2 integer-only gameplay components (Q16.16 / integer ONLY — NO float in any world state) ------------
// Velocity2D — a per-entity Q16.16 velocity DELTA the movement system integrates into Transform2D.pos
// each tick (pos = FxAdd(pos, vel)). Integer; default zero.
struct Velocity2D {
    FxVec3 vel;             // Q16.16 per-tick position delta
};

// Pickup — marks an entity as a collectible worth `value` score points (an integer gameplay scalar).
struct Pickup {
    int32_t value = 0;
};

// Score — an integer score accumulator (the player carries one; SystemCollect bumps it by pickup.value).
struct Score {
    int32_t points = 0;
};

// ----- ForEachOrdered<Ts...>(world, fn): the DETERMINISM CONTRACT — iterate live entities in order[] ----------
// Walk world.order[] (the VD1 spawn-order live-id list) IN SEQUENCE; for each id, resolve its ecs::Entity
// handle and invoke fn(id, e) IFF reg.valid(e) && (reg.has<Ts>(e) && ...). The iteration order is
// EXACTLY order[] — identical on every peer/platform, independent of pool sizes / add-remove churn. This
// is a THIN range/callback over order[] + reg.get<> (NOT a new container). It is the deterministic
// alternative to ecs::Registry::view<>() (whose smallest-pool/insertion order renumbers under churn) —
// and it does NOT modify ecs.h. `fn` is invoked as fn(EntityId, ecs::Entity).
template <typename... Ts, typename Fn>
inline void ForEachOrdered(VerdictWorld& world, Fn&& fn) {
    for (size_t i = 0; i < world.order.size(); ++i) {
        const EntityId id = world.order[i];
        auto it = world.handle.find(id);
        if (it == world.handle.end()) continue;
        const ecs::Entity e = it->second;
        if (!world.reg.valid(e)) continue;
        if (!(world.reg.has<Ts>(e) && ...)) continue;   // require ALL of Ts...
        fn(id, e);
    }
}

// ----- OrderedView<Ts...>(world): collect the order[]-sequence ids matching ALL of Ts... ----------------------
// A small materialized view over the SAME contract as ForEachOrdered: returns the live entity ids that
// have ALL of Ts..., IN order[] SEQUENCE. Handy where the proof wants the explicit id list to compare
// against order[] (and to contrast with a raw reg.view<> that yields a DIFFERENT, churn-renumbered
// order). Pure read; the order is order[] by construction.
template <typename... Ts>
inline std::vector<EntityId> OrderedView(VerdictWorld& world) {
    std::vector<EntityId> ids;
    ForEachOrdered<Ts...>(world, [&](EntityId id, ecs::Entity) { ids.push_back(id); });
    return ids;
}

// ----- The hazard region: a fixed integer AABB in the XY plane (SystemDamage's deterministic hazard) ---------
// A simple pinned, integer world-space band; an entity whose Transform2D.pos is inside it takes -1 hp per
// tick. Pure integer position test — NO float. Centred low-left so the player crosses it on its way to the
// pickups in the showcase scene.
struct HazardRegion {
    fx minX, minY, maxX, maxY;   // Q16.16 inclusive AABB bounds in the XY plane
};

inline bool InHazard(const HazardRegion& hz, const FxVec3& p) {
    return p.x >= hz.minX && p.x <= hz.maxX && p.y >= hz.minY && p.y <= hz.maxY;
}

// ----- SystemMovement(world): integrate each entity's Transform2D by its Velocity2D (pure Q16.16) ------------
// For each entity (in order[] sequence) with BOTH Transform2D and Velocity2D: pos = FxAdd(pos, vel). The
// integer move rule — deterministic, no float. Entities without a Velocity2D are unaffected.
inline void SystemMovement(VerdictWorld& world) {
    ForEachOrdered<Transform2D, Velocity2D>(world, [&](EntityId, ecs::Entity e) {
        Transform2D& xf = world.reg.get<Transform2D>(e);
        const Velocity2D& v = world.reg.get<Velocity2D>(e);
        xf.pos = fpx::FxAdd(xf.pos, v.vel);
    });
}

// ----- SystemDamage(world, hazard): deterministic integer hazard decay over Health -----------------------
// For each entity (in order[] sequence) with BOTH Transform2D and Health: if its position is inside the
// fixed hazard AABB (an integer position test), apply -1 hp (clamped at 0). Deterministic, pure integer.
inline void SystemDamage(VerdictWorld& world, const HazardRegion& hazard) {
    ForEachOrdered<Transform2D, Health>(world, [&](EntityId, ecs::Entity e) {
        const Transform2D& xf = world.reg.get<Transform2D>(e);
        if (!InHazard(hazard, xf.pos)) return;
        Health& h = world.reg.get<Health>(e);
        if (h.hp > 0) h.hp -= 1;   // deterministic integer hazard decay
    });
}

// ----- FxDist2(a, b): the Q16.16 squared distance between two points, in int64 (NO float, NO sqrt) -----------
// Each component delta is Q16.16; its square is Q32.32; the sum is the squared distance in Q32.32 held in
// an int64 (never overflows for in-world coordinates). Compared against a Q32.32 squared-radius formed the
// SAME way — a pure-integer overlap test with NO sqrt.
inline int64_t FxDist2(const FxVec3& a, const FxVec3& b) {
    const FxVec3 d = fpx::FxSub(a, b);
    return (int64_t)d.x * (int64_t)d.x + (int64_t)d.y * (int64_t)d.y + (int64_t)d.z * (int64_t)d.z;
}

// ----- SystemCollect(world, player, radius): integer overlap collect (Q16.16 distance², NO sqrt) -------------
// For the given player entity (which must carry a Transform2D + a Score): test every Pickup entity (in
// order[] sequence) for overlap with the player by a Q16.16 distance² compare against `radius`² — NO
// float, NO sqrt. On overlap: bump the player's Score by the pickup's value and DESPAWN the pickup. A
// dead/unbound player is a deterministic no-op. (Despawn mutates order[]/handle[]; we collect the hits in
// order[] sequence FIRST, then despawn — so the iteration is not disturbed mid-walk.)
inline void SystemCollect(VerdictWorld& world, EntityId player, fx radius) {
    auto pit = world.handle.find(player);
    if (pit == world.handle.end()) return;                 // unknown player -> no-op
    const ecs::Entity pe = pit->second;
    if (!world.reg.valid(pe) || !world.reg.has<Transform2D>(pe) || !world.reg.has<Score>(pe)) return;
    const FxVec3 pp = world.reg.get<Transform2D>(pe).pos;
    const int64_t r2 = (int64_t)radius * (int64_t)radius;  // Q32.32 squared radius (NO sqrt)

    // Collect overlapping pickups in order[] sequence, then despawn them (deterministic, fixed order).
    std::vector<EntityId> hits;
    ForEachOrdered<Transform2D, Pickup>(world, [&](EntityId id, ecs::Entity e) {
        if (id == player) return;
        if (FxDist2(world.reg.get<Transform2D>(e).pos, pp) <= r2) hits.push_back(id);
    });
    Score& sc = world.reg.get<Score>(pe);
    for (size_t i = 0; i < hits.size(); ++i) {
        auto hit = world.handle.find(hits[i]);
        if (hit == world.handle.end() || !world.reg.valid(hit->second)) continue;
        sc.points += world.reg.get<Pickup>(hit->second).value;   // bump score by the pickup value
        DespawnEntity(world, hits[i]);                            // remove the collected pickup
    }
}

// ----- StepGameplay(world, commands, tick, hazard, player, collectRadius): the FIXED gameplay schedule -------
// ONE deterministic gameplay tick. The SCHEDULE ORDER IS FIXED AND DOCUMENTED (later systems see earlier
// systems' mutations — a single deterministic pass, the Gauss-Seidel-in-fixed-order discipline the sims
// use):
//   1. ApplyCommands(world, commands, tick)   — the VD1 input-command bus (BEFORE the systems)
//   2. SystemMovement(world)                  — integrate Transform2D by Velocity2D
//   3. SystemDamage(world, hazard)            — hazard decay over Health
//   4. SystemCollect(world, player, radius)   — overlap-collect pickups (sees the moved player)
//   5. world.tick++
// Swapping any two systems changes the result (e.g. collect-before-move would test the pre-move player
// position) — order MATTERS and is PINNED. Pure integer; bit-identical CPU/Vulkan/Metal by construction.
inline void StepGameplay(VerdictWorld& world, const std::vector<Command>& commands, uint32_t tick,
                         const HazardRegion& hazard, EntityId player, fx collectRadius) {
    ApplyCommands(world, commands, tick);   // 1. the command bus (VD1)
    SystemMovement(world);                   // 2. movement
    SystemDamage(world, hazard);             // 3. hazard damage
    SystemCollect(world, player, collectRadius);  // 4. collect (sees the moved player)
    ++world.tick;                            // 5. advance the world clock
}

// ----- A SpawnEntity overload that also attaches a Velocity2D (the VD2 movement spawn) -----------------------
// Convenience for the VD2 showcase/tests: spawn at `xf` with `health`, an unbound BodyRef, AND a
// Velocity2D. Reuses the VD1 SpawnEntity (id alloc + order[] + Transform2D/Health/BodyRef), then adds the
// velocity component to the ecs pool. Returns the new EntityId.
inline EntityId SpawnMover(VerdictWorld& world, const Transform2D& xf, const Health& health,
                           const Velocity2D& vel) {
    const EntityId id = SpawnEntity(world, xf, health, BodyRef{kNoBody});
    world.reg.add<Velocity2D>(world.handle.at(id), vel);
    return id;
}

// =================================================================================================
// Slice VD3 — COMPOSING THE PHYSICS SUBSYSTEM — ONE WORLD TICK (APPEND-ONLY; VD1/VD2 above are
// BYTE-FROZEN). The 3rd slice of FLAGSHIP #27 (DETERMINISTIC GAMEPLAY / NETCODE, hf::game::verdict).
//
// VD1 built the deterministic entity world + the command bus; VD2 the gameplay SYSTEMS on a pinned
// schedule. VD3 makes it a *physics game*: ONE deterministic StepWorld tick that runs the gameplay
// systems AND drives the embedded Q16.16 physics sim (the FROZEN warmhull warm+sleep hull world from
// flagship #26) in a single pinned order, bridged by a deterministic bidirectional BodyRef sync.
//
// THE COMPOSITION (the embedded sim is NEVER modified — proof #2 below is the guard):
//   * SyncComponentsToBodies(world) — for each order[]-bound entity, PUSH its gameplay-driven
//     Velocity2D into world.sim.bodies[simBodyIndex].vel (gameplay -> sim). FIXED order, integer.
//   * The sim commands (kCmdImpulse/kCmdSetAngVel) are LOWERED by LowerToHullCommands (VD1) and
//     applied by gjk::ApplyHullCommands (frozen) BEFORE the step — the ApplyHullCommands-before-step
//     contract every sim's lockstep tick uses.
//   * warmhull::StepWarmSleepHullWorld(sim, cache, sleep, cfg) — the FROZEN WH4 step, called verbatim.
//   * SyncBodiesToComponents(world) — for each order[]-bound entity, READ its settled body
//     pos/orient back into its Transform2D (sim -> gameplay). FIXED order, integer.
//
// PURE CPU INTEGER (Q16.16, the strictest determinism tier, strict 0px): NO new render RHI, NO new
// shader, NO new compute. warmhull.h / gjk.h / ALL sim headers + ecs.h are BYTE-UNCHANGED; the
// warmhull step is CALLED verbatim, never re-implemented. VD3 composes EXACTLY ONE sim subsystem.
// =================================================================================================

// ----- BindBody(world, id, simBodyIndex): bind a gameplay entity to a sim body --------------------
// Set the entity's BodyRef.simBodyIndex (adding the BodyRef component if absent). After binding, the
// two syncs (below) push the entity's velocity into world.sim.bodies[simBodyIndex] and read the
// body's settled pose back. simBodyIndex is a STABLE index into world.sim.bodies (sim bodies are NOT
// recycled within a match). A dead/unknown id is a deterministic no-op. Pure integer.
inline void BindBody(VerdictWorld& world, EntityId id, uint32_t simBodyIndex) {
    auto it = world.handle.find(id);
    if (it == world.handle.end()) return;          // unknown -> no-op (deterministic)
    const ecs::Entity e = it->second;
    if (!world.reg.valid(e)) return;               // dead handle -> no-op
    if (world.reg.has<BodyRef>(e)) world.reg.get<BodyRef>(e).simBodyIndex = simBodyIndex;
    else                          world.reg.add<BodyRef>(e, BodyRef{simBodyIndex});
}

// ----- SyncComponentsToBodies(world): the deterministic gameplay -> sim PUSH --------------------
// For each entity (in order[] sequence) with a BOUND BodyRef (simBodyIndex != kNoBody AND in range)
// AND a Velocity2D, WRITE that gameplay-driven velocity into world.sim.bodies[simBodyIndex].vel.
// Unbound (kNoBody) / out-of-range BodyRefs + entities with no Velocity2D are deterministic no-ops
// (a gameplay-only entity has no body to drive). FIXED order[] order, pure integer write -> a pure
// function (calling it twice on the same world yields byte-equal sim bodies — proof #3).
inline void SyncComponentsToBodies(VerdictWorld& world) {
    const size_t nBodies = world.sim.bodies.size();
    for (size_t i = 0; i < world.order.size(); ++i) {
        const EntityId id = world.order[i];
        auto it = world.handle.find(id);
        if (it == world.handle.end()) continue;
        const ecs::Entity e = it->second;
        if (!world.reg.valid(e)) continue;
        if (!world.reg.has<BodyRef>(e)) continue;
        const uint32_t bi = world.reg.get<BodyRef>(e).simBodyIndex;
        if (bi == kNoBody || (size_t)bi >= nBodies) continue;   // unbound / out-of-range -> no-op
        if (!world.reg.has<Velocity2D>(e)) continue;            // no gameplay velocity -> no-op
        world.sim.bodies[(size_t)bi].vel = world.reg.get<Velocity2D>(e).vel;   // gameplay drives the body
    }
}

// ----- SyncBodiesToComponents(world): the deterministic sim -> gameplay READ-BACK ----------------
// For each entity (in order[] sequence) with a BOUND BodyRef (simBodyIndex != kNoBody AND in range)
// AND a Transform2D, READ the settled body's pos/orient back into that Transform2D. Unbound /
// out-of-range BodyRefs + entities with no Transform2D are deterministic no-ops. FIXED order[]
// order, pure integer copy -> a pure function (proof #3): a body-bound entity's Transform2D tracks
// its bound body's settled pos.
inline void SyncBodiesToComponents(VerdictWorld& world) {
    const size_t nBodies = world.sim.bodies.size();
    for (size_t i = 0; i < world.order.size(); ++i) {
        const EntityId id = world.order[i];
        auto it = world.handle.find(id);
        if (it == world.handle.end()) continue;
        const ecs::Entity e = it->second;
        if (!world.reg.valid(e)) continue;
        if (!world.reg.has<BodyRef>(e)) continue;
        const uint32_t bi = world.reg.get<BodyRef>(e).simBodyIndex;
        if (bi == kNoBody || (size_t)bi >= nBodies) continue;   // unbound / out-of-range -> no-op
        if (!world.reg.has<Transform2D>(e)) continue;           // no transform to write -> no-op
        Transform2D& xf = world.reg.get<Transform2D>(e);
        xf.pos    = world.sim.bodies[(size_t)bi].pos;           // read the settled body pose back
        xf.orient = world.sim.bodies[(size_t)bi].orient;
    }
}

// ----- StepWorld(world, commands, tick, cfg): the FIXED pinned composed world tick ----------------
// ONE deterministic world tick composing the gameplay systems AND the embedded warm+sleep hull sim.
// THE SCHEDULE ORDER IS FIXED AND DOCUMENTED — commands + tick++ applied EXACTLY ONCE (this factors
// StepGameplay's body so it does NOT double-apply commands or double-increment tick):
//   1. ApplyCommands(world, commands, tick)              — the VD1 command bus (BEFORE the systems)
//   2. SystemMovement / SystemDamage / SystemCollect     — the VD2 gameplay systems (gameplay FIRST)
//   3. SyncComponentsToBodies(world)                     — gameplay -> sim PUSH (velocities)
//   4. LowerToHullCommands -> gjk::ApplyHullCommands     — the lowered sim commands, BEFORE the step
//   5. warmhull::StepWarmSleepHullWorld(sim,cache,sleep) — the FROZEN WH4 step, VERBATIM
//   6. SyncBodiesToComponents(world)                     — sim -> gameplay READ-BACK (settled pose)
//   7. ++world.tick                                      — advance the world clock (ONCE)
// Gameplay runs BEFORE the sim so a collected pickup / moved player is reflected before the sim reads
// bodies; the sync halves are pinned (swapping them changes the result). The embedded sim is NEVER
// perturbed (the sync only reads/writes body fields; the step is verbatim — proof #2). Pure integer,
// FIXED order -> bit-identical CPU/Vulkan/Metal by construction.
inline void StepWorld(VerdictWorld& world, const std::vector<Command>& commands, uint32_t tick,
                      const HazardRegion& hazard, EntityId player, fx collectRadius,
                      const warmhull::HullSleepConfig& cfg) {
    ApplyCommands(world, commands, tick);            // 1. the command bus (VD1) — applied ONCE
    SystemMovement(world);                            // 2a. movement (VD2)
    SystemDamage(world, hazard);                      // 2b. hazard damage (VD2)
    SystemCollect(world, player, collectRadius);      // 2c. collect (sees the moved player) (VD2)
    SyncComponentsToBodies(world);                    // 3. gameplay -> sim PUSH
    const std::vector<convex::ConvexCommand> hullCmds = LowerToHullCommands(world, commands, tick);
    gjk::ApplyHullCommands(world.sim, hullCmds, tick);  // 4. the lowered sim commands, BEFORE the step
    warmhull::StepWarmSleepHullWorld(world.sim, world.cache, world.sleep, cfg);  // 5. FROZEN WH4 step
    SyncBodiesToComponents(world);                    // 6. sim -> gameplay READ-BACK
    ++world.tick;                                     // 7. advance the world clock (ONCE)
}

// ----- StepWorldN(world, commands, tick0, ..., ticks): run `ticks` composed world ticks ----------
// Advance the composed world `ticks` steps from tick0 (each StepWorld applies the commands of its own
// tick + steps the sim once). The gameplay state + the sim TRIPLE (bodies/cache/sleep) carry across.
inline void StepWorldN(VerdictWorld& world, const std::vector<Command>& commands, uint32_t tick0,
                       const HazardRegion& hazard, EntityId player, fx collectRadius,
                       const warmhull::HullSleepConfig& cfg, uint32_t ticks) {
    for (uint32_t t = 0; t < ticks; ++t)
        StepWorld(world, commands, tick0 + t, hazard, player, collectRadius, cfg);
}

// =================================================================================================
// Slice VD4 — THE HETEROGENEOUS SNAPSHOT/RESTORE + EQUALITY (APPEND-ONLY; VD1/VD2/VD3 above are
// BYTE-FROZEN). The 4th slice of FLAGSHIP #27 (DETERMINISTIC GAMEPLAY / NETCODE, hf::game::verdict).
//
// VD1-VD3 built the entity world + command bus, the gameplay systems, and the one composed StepWorld
// tick (gameplay + the frozen physics sim). VD4 builds the make-or-break primitive for rollback
// netcode: ONE snapshot/restore that captures the ENTIRE heterogeneous world — the entities + every
// component pool + the embedded sim's TRIPLE (bodies + cache + sleep) — under a single restore point,
// plus a byte-equality over all of it. This is the prerequisite for VD5's world-level
// lockstep/rollback.
//
// THE DETERMINISM CONTRACT IS order[]-KEYED. The component pools are captured / restored / compared
// IN order[] SEQUENCE (NOT the ECS dense / insertion order, which renumbers under add/remove churn
// and is non-deterministic). The EntityId is the pinned identity; the ecs::Entity handle may differ
// after a restore — that is FINE; everything is keyed off EntityId + order[] + component values + the
// sim TRIPLE, NEVER off the opaque ecs handle.
//
// THE CRUX (COMPLETENESS): a correct snapshot must capture ALL mutable state, or a rolled-back peer
// resumes with stale entity / impulse-cache / sleep state and DIVERGES. VD4 proves COMPLETENESS:
// advance N -> snapshot -> advance M (diverge) -> restore -> re-advance M is byte-identical to a
// reference world that advanced N+M straight (nothing escapes the snapshot). A restore that OMITS the
// warmhull::WarmHullSnapshot must DIVERGE (the WH5 TRIPLE lesson, proven at the world level).
//
// PURE CPU INTEGER (strict 0px): NO new render RHI, NO new shader, NO new compute. The sim third uses
// warmhull::SnapshotWarmHull / RestoreWarmHull / WarmHullStatesEqual VERBATIM. warmhull.h / ecs.h +
// ALL sim headers + ALL shaders BYTE-UNCHANGED; verdict.h is APPEND-ONLY.
// =================================================================================================

// ----- Per-component capture entry: {EntityId, component-value}, recorded IN order[] SEQUENCE -------
// One std::vector<CompEntry<T>> per component type carries every live entity (in order[] order) that
// HAS that component, paired with the component value. The EntityId is the pinned key (the ecs handle
// is NOT stored — it is re-allocated on restore). All component types are POD integer.
template <typename T>
struct CompEntry {
    EntityId id;
    T        value;
};

// ----- VerdictSnapshot: the FULL heterogeneous restore point (entities + EVERY component pool + sim) -
// The TRIPLE-plus: the entity bookkeeping (tick + nextId + a deep copy of order[]), every component
// pool serialized IN order[] SEQUENCE (one CompEntry vector per type — capture EVERY component type
// the world uses: Transform2D / Health / BodyRef / Velocity2D / Pickup / Score), and the embedded
// sim's warmhull::WarmHullSnapshot (bodies + cache + sleep). A full value copy (a delta encoder is a
// future optimization, out of scope). The ecs::Entity handles are deliberately NOT captured — restore
// re-allocates them; the contract is order[]-keyed.
struct VerdictSnapshot {
    uint32_t                       tick   = 0;   // the world clock at snapshot time
    EntityId                       nextId = 1u;  // the monotonic id allocator (NEVER recycled)
    std::vector<EntityId>          order;        // a deep copy of the pinned spawn-order live-id list

    // Every component pool, serialized in order[] sequence (NOT the ECS dense order).
    std::vector<CompEntry<Transform2D>> transforms;
    std::vector<CompEntry<Health>>      healths;
    std::vector<CompEntry<BodyRef>>     bodyRefs;
    std::vector<CompEntry<Velocity2D>>  velocities;
    std::vector<CompEntry<Pickup>>      pickups;
    std::vector<CompEntry<Score>>       scores;

    warmhull::WarmHullSnapshot     simSnap;      // the sim third (bodies + cache + sleep), VERBATIM WH5
};

// ----- SnapshotWorld(world) -> VerdictSnapshot: deep-copy the ENTIRE heterogeneous world (pure read) -
// Deep-copies order[] / nextId / tick; for each entity in order[] (the pinned sequence), captures each
// component it has<T>() into that type's CompEntry vector (so the serialization is order[]-keyed, NOT
// ECS-dense-keyed); delegates the sim third to warmhull::SnapshotWarmHull(sim, cache, sleep, tick).
// A pure read — `world` is unchanged. The crux: capture EVERY component type, or the completeness
// proof diverges on the one that escaped.
inline VerdictSnapshot SnapshotWorld(const VerdictWorld& world) {
    VerdictSnapshot snap;
    snap.tick   = world.tick;
    snap.nextId = world.nextId;
    snap.order  = world.order;   // deep copy of the pinned id sequence

    for (size_t i = 0; i < world.order.size(); ++i) {
        const EntityId id = world.order[i];
        auto it = world.handle.find(id);
        if (it == world.handle.end()) continue;
        const ecs::Entity e = it->second;
        if (!world.reg.valid(e)) continue;
        // Capture every component type the world uses, IN order[] sequence.
        if (world.reg.has<Transform2D>(e)) snap.transforms.push_back({id, world.reg.get<Transform2D>(e)});
        if (world.reg.has<Health>(e))      snap.healths.push_back({id, world.reg.get<Health>(e)});
        if (world.reg.has<BodyRef>(e))     snap.bodyRefs.push_back({id, world.reg.get<BodyRef>(e)});
        if (world.reg.has<Velocity2D>(e))  snap.velocities.push_back({id, world.reg.get<Velocity2D>(e)});
        if (world.reg.has<Pickup>(e))      snap.pickups.push_back({id, world.reg.get<Pickup>(e)});
        if (world.reg.has<Score>(e))       snap.scores.push_back({id, world.reg.get<Score>(e)});
    }

    // The sim third — delegate to the frozen WH5 snapshot (bodies + cache + sleep), VERBATIM.
    snap.simSnap = warmhull::SnapshotWarmHull(world.sim, world.cache, world.sleep, world.tick);
    return snap;
}

// ----- RestoreWorld(world, snap): REBUILD the world from the snapshot (handle churn irrelevant) ------
// REBUILD: clear reg / handle / order, then for each EntityId in snap.order re-create a FRESH
// ecs::Entity (reg.create() — the new handle may differ from the original; that is FINE) + map it in
// handle[]. Re-add each captured component (in the CompEntry vectors, which are in order[] sequence).
// Restore nextId / tick, and restore the sim third via warmhull::RestoreWarmHull (bodies + cache +
// sleep). After this, VerdictStatesEqual(restored, snapshotted) holds (the contract is over EntityId +
// order[] + component values + the sim TRIPLE, NOT the opaque handles).
inline void RestoreWorld(VerdictWorld& world, const VerdictSnapshot& snap) {
    // Clear the entity bookkeeping (a fresh registry drops every pool + free-list — the cleanest
    // rebuild; the handle churn is invisible to the order[]-keyed contract).
    world.reg = ecs::Registry{};
    world.handle.clear();
    world.order.clear();

    // Re-create one fresh handle per EntityId, in order[] sequence, mapping id -> the new handle.
    for (size_t i = 0; i < snap.order.size(); ++i) {
        const EntityId id = snap.order[i];
        const ecs::Entity e = world.reg.create();   // a fresh handle (may differ — irrelevant)
        world.handle[id] = e;
        world.order.push_back(id);
    }

    // Re-add every captured component to its owning entity's new handle. The CompEntry vectors are in
    // order[] sequence; we re-add by EntityId -> handle (a defensive guard skips an unknown/dead id).
    auto reAdd = [&](EntityId id, auto&& addFn) {
        auto it = world.handle.find(id);
        if (it == world.handle.end()) return;
        const ecs::Entity e = it->second;
        if (!world.reg.valid(e)) return;
        addFn(e);
    };
    for (const auto& c : snap.transforms) reAdd(c.id, [&](ecs::Entity e) { world.reg.add<Transform2D>(e, c.value); });
    for (const auto& c : snap.healths)    reAdd(c.id, [&](ecs::Entity e) { world.reg.add<Health>(e, c.value); });
    for (const auto& c : snap.bodyRefs)   reAdd(c.id, [&](ecs::Entity e) { world.reg.add<BodyRef>(e, c.value); });
    for (const auto& c : snap.velocities) reAdd(c.id, [&](ecs::Entity e) { world.reg.add<Velocity2D>(e, c.value); });
    for (const auto& c : snap.pickups)    reAdd(c.id, [&](ecs::Entity e) { world.reg.add<Pickup>(e, c.value); });
    for (const auto& c : snap.scores)     reAdd(c.id, [&](ecs::Entity e) { world.reg.add<Score>(e, c.value); });

    world.nextId = snap.nextId;
    world.tick   = snap.tick;

    // Restore the sim third — delegate to the frozen WH5 restore (bodies + cache + sleep), VERBATIM.
    // (world.sim.hulls are immutable/shared geometry — RestoreWarmHull leaves them untouched.)
    warmhull::RestoreWarmHull(world.sim, world.cache, world.sleep, snap.simSnap);
}

// ----- The per-component value equality (field-wise — NO memcmp, padding-safe for the POD components) -
inline bool TransformEq(const Transform2D& a, const Transform2D& b) {
    return a.pos.x == b.pos.x && a.pos.y == b.pos.y && a.pos.z == b.pos.z &&
           a.orient.x == b.orient.x && a.orient.y == b.orient.y && a.orient.z == b.orient.z &&
           a.orient.w == b.orient.w;
}
inline bool HealthEq(const Health& a, const Health& b)       { return a.hp == b.hp; }
inline bool BodyRefEq(const BodyRef& a, const BodyRef& b)    { return a.simBodyIndex == b.simBodyIndex; }
inline bool Velocity2DEq(const Velocity2D& a, const Velocity2D& b) {
    return a.vel.x == b.vel.x && a.vel.y == b.vel.y && a.vel.z == b.vel.z;
}
inline bool PickupEq(const Pickup& a, const Pickup& b)       { return a.value == b.value; }
inline bool ScoreEq(const Score& a, const Score& b)          { return a.points == b.points; }

// ----- CompEntriesEqual(a, b, eq): two CompEntry vectors are equal iff same {id, value} sequence ----
// Both vectors are built in order[] sequence (by SnapshotWorld), so a POSITIONAL compare is exact: the
// SAME EntityId at the SAME position with an equal component value. (The order[]-keyed contract.)
template <typename T, typename Eq>
inline bool CompEntriesEqual(const std::vector<CompEntry<T>>& a, const std::vector<CompEntry<T>>& b, Eq eq) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].id != b[i].id) return false;
        if (!eq(a[i].value, b[i].value)) return false;
    }
    return true;
}

// ----- VerdictSnapshotsEqual(a, b): byte-for-byte equality of two heterogeneous snapshots ------------
// Equal iff: a.order == b.order (the same pinned id sequence) AND a.nextId == b.nextId AND every
// component pool is equal in order[] sequence (CompEntriesEqual over each type) AND the sim TRIPLE
// matches (warmhull::WarmHullStatesEqual over bodies + cache + sleep). This is the make-or-break
// comparison VD5's rollback builds on. (tick is part of the snapshot but the equality is over the
// determinism-relevant state — order[] + components + sim; we compare it too for completeness.)
inline bool VerdictSnapshotsEqual(const VerdictSnapshot& a, const VerdictSnapshot& b) {
    if (a.order  != b.order)  return false;
    if (a.nextId != b.nextId) return false;
    if (a.tick   != b.tick)   return false;
    if (!CompEntriesEqual(a.transforms, b.transforms, TransformEq))   return false;
    if (!CompEntriesEqual(a.healths,    b.healths,    HealthEq))      return false;
    if (!CompEntriesEqual(a.bodyRefs,   b.bodyRefs,   BodyRefEq))     return false;
    if (!CompEntriesEqual(a.velocities, b.velocities, Velocity2DEq))  return false;
    if (!CompEntriesEqual(a.pickups,    b.pickups,    PickupEq))      return false;
    if (!CompEntriesEqual(a.scores,     b.scores,     ScoreEq))       return false;
    return warmhull::WarmHullStatesEqual(a.simSnap.bodies, a.simSnap.cache, a.simSnap.sleep,
                                         b.simSnap.bodies, b.simSnap.cache, b.simSnap.sleep);
}

// ----- VerdictStatesEqual(a, b): two LIVE VerdictWorlds are equal over the whole heterogeneous state -
// The make-or-break world-level comparison: snapshot BOTH worlds (which serializes the components in
// order[] sequence + bundles the sim TRIPLE) and compare the snapshots (VerdictSnapshotsEqual). Two
// worlds are equal iff the SAME order[] + nextId + every component pool (in order[] order) + the sim
// TRIPLE (warmhull::WarmHullStatesEqual) — independent of the ecs::Entity handles (which may differ
// after a restore). Note: this compares tick too (snapshots carry it); a caller comparing worlds at
// the SAME logical tick (the completeness / round-trip proofs) is unaffected.
inline bool VerdictStatesEqual(const VerdictWorld& a, const VerdictWorld& b) {
    return VerdictSnapshotsEqual(SnapshotWorld(a), SnapshotWorld(b));
}

// =================================================================================================
// Slice VD5 — WHOLE-WORLD LOCKSTEP + ROLLBACK — THE NETCODE HEADLINE (APPEND-ONLY; VD1/VD2/VD3/VD4
// above are BYTE-FROZEN). The 5th slice of FLAGSHIP #27 (DETERMINISTIC GAMEPLAY / NETCODE,
// hf::game::verdict; the warmhull::WH5 / persist::PS5 / gjk::GJ5 twin at the WHOLE-WORLD level).
//
// VD1-VD4 built the deterministic entity world + command bus, the gameplay systems, the one composed
// StepWorld tick (gameplay + the frozen physics sim), and the heterogeneous snapshot/restore/equality.
// VD5 is THE PRODUCT CLAIM: the ENTIRE game world — entities + gameplay rules + the embedded physics
// sim — is LOCKSTEP- and ROLLBACK-replayable from an input stream ALONE, bit-identical
// CPU/Vulkan/Metal. Two peers fed only commands re-derive the WHOLE world byte-for-byte; a rollback
// re-sims from a VD4 snapshot bit-for-bit, and a misprediction that diverges across BOTH gameplay (a
// spawned/despawned entity / a score) AND physics (a perturbed body) is corrected. This falls out by
// retargeting the WH5 lockstep/rollback harness over StepWorld + the VD4 snapshot/restore — PURE CPU.
//
// THE NON-COPYABLE CONSTRAINT (the ONE structural difference vs WH5): VerdictWorld holds an
// ecs::Registry whose pools are unique_ptr — it is NON-COPYABLE. So the two peers CANNOT be cloned by
// `VerdictWorld b = a;`. They are cloned via RestoreWorld(world0Snap) into a FRESH world — the VD4
// determinism-faithful clone (the restored peers are VerdictStatesEqual to world0). Snapshots
// (VerdictSnapshot) ARE copyable and cross the harness boundaries; a fresh world is materialized from
// one by RestoreWorld. The harness therefore takes an initial VerdictSnapshot (world0Snap), NOT a world.
//
// PURE CPU INTEGER (strict 0px): NO new render RHI, NO new shader, NO new compute. StepWorld + the VD4
// SnapshotWorld/RestoreWorld/VerdictStatesEqual are reused VERBATIM. warmhull.h/ecs.h + ALL sim headers
// + ALL shaders BYTE-UNCHANGED; verdict.h is APPEND-ONLY (VD1-VD4 code byte-frozen).
// =================================================================================================

// ----- VerdictParams: the StepWorld scene knobs bundled (so the harness signatures stay clean) -------
// StepWorld takes {hazard, player, collectRadius, cfg}; these are CONSTANT across a match (the same on
// every peer + NOT snapshotted — only the world state + the command stream are replayable). Bundling
// them keeps RunVerdictLockstep/RunVerdictRollback to (world0Snap, params, streams, ticks, ...).
//
// `hulls` is the IMMUTABLE per-body hull geometry (world.sim.hulls). It is CONSTANT scene data — NOT
// replayable state — and therefore (like the WarmHullSnapshot's `hulls` lesson) is NOT carried in the
// VerdictSnapshot (SnapshotWorld captures bodies/cache/sleep, the warmhull::SnapshotWarmHull does NOT
// copy hulls). The harness clones a peer via RestoreWorld(world0Snap) into a FRESH VerdictWorld whose
// sim.hulls START EMPTY, so the harness MUST seed `params.hulls` into each fresh peer's sim.hulls BEFORE
// the restore, or the embedded sim step indexes an empty hulls[] (the non-copyable-clone subtlety).
struct VerdictParams {
    HazardRegion              hazard;          // the fixed integer hazard AABB (SystemDamage)
    EntityId                  player = kNoEntity;  // the collect/score player entity
    fx                        collectRadius = 0;   // the Q16.16 collect overlap radius
    warmhull::HullSleepConfig cfg;             // the frozen WH4 warm+sleep step config
    std::vector<gjk::FxHull>  hulls;           // the IMMUTABLE per-body hull geometry (seeded into each peer)
};

// ----- ClonePeer(snap, params) -> VerdictWorld: the determinism-faithful non-copyable clone --------
// Materialize a fresh peer from a copyable snapshot: seed the immutable hull geometry (NOT in the
// snapshot — constant scene data), then RestoreWorld the bodies/components/cache/sleep. The result is
// VerdictStatesEqual to the world the snapshot was taken from (the snapshot-faithful clone — the
// non-copyable-world replacement for `VerdictWorld b = a;`).
inline VerdictWorld ClonePeer(const VerdictSnapshot& snap, const VerdictParams& params) {
    VerdictWorld w;
    w.sim.hulls = params.hulls;   // seed the immutable geometry BEFORE the restore (it is NOT in the snapshot)
    RestoreWorld(w, snap);        // restore bodies/components/cache/sleep into the seeded world
    return w;
}

// ----- SimVerdictTick(world, params, commands, tick): ONE deterministic composed world tick + inputs --
// = StepWorld(world, commands, tick, params.hazard, params.player, params.collectRadius, params.cfg) —
// the VD3 composed tick (the gameplay verbs + the lowered sim verbs in one bus). The WH5 SimWarmHullTick
// analog at the WHOLE-WORLD level. Pure integer, fixed order -> bit-identical on every peer/platform.
inline void SimVerdictTick(VerdictWorld& world, const VerdictParams& params,
                           const std::vector<Command>& commands, uint32_t tick) {
    StepWorld(world, commands, tick, params.hazard, params.player, params.collectRadius, params.cfg);
}

// ----- RunVerdictLockstep(world0Snap, params, commands, ticks, outIdentical) -> the authority snapshot -
// THE peer entry point (the warmhull::RunWarmHullLockstep control flow over SimVerdictTick, with the
// NON-COPYABLE world cloned via RestoreWorld instead of a copy). RestoreWorld(world0Snap) into BOTH a
// fresh `authority` and a fresh `replica` VerdictWorld (the VD4 determinism-faithful clone — NOT a copy);
// BOTH run SimVerdictTick for `ticks` with the SAME command stream (INPUTS ONLY — no state shared) ->
// BIT-IDENTICAL by determinism, each re-deriving the entities + every component pool + the sim TRIPLE
// every tick. Sets *outIdentical (if non-null) to whether the two final WHOLE worlds are byte-identical
// (VerdictStatesEqual — the make-or-break lockstep proof) + returns the converged AUTHORITY's snapshot
// (SnapshotWorld(authority) — copyable, so the caller renders it WITHOUT copying the non-copyable world).
// The peer step order is PINNED.
inline VerdictSnapshot RunVerdictLockstep(const VerdictSnapshot& world0Snap, const VerdictParams& params,
                                          const std::vector<Command>& commands, uint32_t ticks,
                                          bool* outIdentical = nullptr) {
    VerdictWorld authority = ClonePeer(world0Snap, params);   // the VD4 clone (NOT a copy; hulls seeded)
    VerdictWorld replica   = ClonePeer(world0Snap, params);   // the second peer fed the SAME inputs
    for (uint32_t t = 0; t < ticks; ++t) {
        SimVerdictTick(authority, params, commands, t);
        SimVerdictTick(replica,   params, commands, t);
    }
    if (outIdentical)
        *outIdentical = VerdictStatesEqual(authority, replica);   // the WHOLE world (entities+components+sim)
    return SnapshotWorld(authority);   // a copyable snapshot the caller restores+renders (world is non-copyable)
}

// ----- RunVerdictRollback(world0Snap, params, authStream, mispredictStream, ticks, rollbackAt, ...) ---
// The rollback harness (the warmhull::RunWarmHullRollback control flow over SimVerdictTick + the VD4
// heterogeneous snapshot). (1) RestoreWorld(world0Snap) + advance 0..rollbackAt with authStream; (2)
// SnapshotWorld AT rollbackAt (the VD4 restore point — entities + EVERY component pool + the sim TRIPLE);
// (2b) speculatively advance a few ticks (<=3) with the MISPREDICTED stream (a WRONG stream that diverges
// across BOTH gameplay [a mis-spawn/despawn or a different score] AND physics [a wrong kCmdImpulse] — the
// client prediction that diverges), capturing that diverged snapshot; (3) ROLLBACK — RestoreWorld back to
// the snapshot + re-sim rollbackAt..ticks with the CORRECT authStream -> the corrected world. Returns the
// corrected authority's snapshot; sets *outCorrectedEqAuthority (if non-null) to whether the corrected
// world == RunVerdictLockstep(world0Snap, params, authStream, ticks) over the WHOLE world, and
// *outMispredictDiverged (if non-null) to whether the speculative pre-rollback world DIFFERED from the
// authority-at-that-tick (proving a REAL divergence was corrected). THE HEADLINE: the divergence spans
// BOTH gameplay AND physics — see the showcase/test, which assert it component-wise. params + the streams
// are CONSTANT, NOT snapshotted.
inline VerdictSnapshot RunVerdictRollback(const VerdictSnapshot& world0Snap, const VerdictParams& params,
                                          const std::vector<Command>& authStream,
                                          const std::vector<Command>& mispredictStream,
                                          uint32_t ticks, uint32_t rollbackAt,
                                          bool* outCorrectedEqAuthority = nullptr,
                                          bool* outMispredictDiverged = nullptr) {
    VerdictWorld w = ClonePeer(world0Snap, params);   // the VD4 clone (NOT a copy; hulls seeded)
    // (1) advance 0..rollbackAt with the authoritative stream.
    for (uint32_t t = 0; t < rollbackAt; ++t)
        SimVerdictTick(w, params, authStream, t);
    // (2) SAVE the VD4 snapshot at rollbackAt (the rollback restore point — entities + components + sim).
    const VerdictSnapshot snap = SnapshotWorld(w);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (the wrong prediction that
    // diverges across gameplay AND physics). Bounded to the remaining ticks (<=3). Capture the diverged snap.
    uint32_t specTicks = ticks - rollbackAt;
    if (specTicks > 3u) specTicks = 3u;
    for (uint32_t s = 0; s < specTicks; ++s)
        SimVerdictTick(w, params, mispredictStream, rollbackAt + s);
    const VerdictSnapshot specSnap = SnapshotWorld(w);   // the diverged pre-rollback world (the "real divergence")
    // (3) ROLLBACK: restore the snapshot + re-sim rollbackAt..ticks with the authStream.
    RestoreWorld(w, snap);
    for (uint32_t t = rollbackAt; t < ticks; ++t)
        SimVerdictTick(w, params, authStream, t);

    if (outCorrectedEqAuthority || outMispredictDiverged) {
        // The authority advanced the SAME number of speculative ticks (rollbackAt + specTicks) with the
        // CORRECT stream — the apples-to-apples comparison point for the misprediction-diverged proof.
        VerdictWorld authAtSpec = ClonePeer(world0Snap, params);
        for (uint32_t t = 0; t < rollbackAt + specTicks; ++t)
            SimVerdictTick(authAtSpec, params, authStream, t);
        if (outMispredictDiverged) {
            VerdictWorld specWorld = ClonePeer(specSnap, params);
            *outMispredictDiverged = !VerdictStatesEqual(specWorld, authAtSpec);
        }
        if (outCorrectedEqAuthority) {
            // The corrected world == a full lockstep authority over the WHOLE world (snapshot-level compare).
            const VerdictSnapshot authFinal = RunVerdictLockstep(world0Snap, params, authStream, ticks);
            *outCorrectedEqAuthority = VerdictSnapshotsEqual(SnapshotWorld(w), authFinal);
        }
    }
    return SnapshotWorld(w);   // the corrected authority's snapshot (copyable — the caller restores+renders)
}

// =================================================================================================
// Slice VD6 — THE PLAYABLE LIT 3D CAPSTONE — THE MONEY-SHOT (APPEND-ONLY; VD1/VD2/VD3/VD4/VD5 above are
// BYTE-FROZEN). The 6th and FINAL slice of FLAGSHIP #27 (DETERMINISTIC GAMEPLAY / NETCODE PRODUCT LAYER,
// hf::game::verdict; the gjk::GJ6 / warmhull::WH6 / ccd::CD6 render-capstone twin at the WHOLE-WORLD level).
//
// VD1-VD5 built the deterministic entity world, the gameplay systems, the composed StepWorld tick, the
// heterogeneous snapshot, and the whole-world lockstep/rollback. VD6 is the artifact the whole flagship was
// built to produce: it RENDERS the bit-exact deterministic game — a player rolling through a hull-stack arena,
// collecting pickups, the stack settling — as a LIT 3D scene, whose ENTIRE world (entities + gameplay +
// physics) is the bit-exact VD1-VD5 deterministic world and is lockstep-replayable.
//
// THE RENDER IS THE ONE FLOAT CROSSING (outside the bit-exact integer loop): VerdictToRenderInstances maps the
// frozen world to FLOAT geometry for display via the FROZEN gjk::HullToRenderInstances (the WH6/CD6/MF6/GJ6
// delegate — the sim bodies render through it verbatim, MATTE) PLUS gameplay-entity overlays (a small marker
// mesh per non-body entity at its Transform2D.pos, pickups tinted by type, the player tinted by Score). It is a
// PURE FUNCTION of `world` (two calls byte-equal via gjk::HullRenderMeshEqual); the integer world is NOT mutated
// (the provenance + world-unmutated proofs guarantee this). NO new shader, NO new RHI: the soup feeds the
// EXISTING instanced-lit pipeline (the WH6/GJ6 render path REUSED VERBATIM). gjk.h/warmhull.h + ALL sim headers
// + ecs.h + ALL shaders BYTE-UNCHANGED; verdict.h is APPEND-ONLY (VD1-VD5 code byte-frozen). This is the FINAL
// slice -> FLAGSHIP #27 COMPLETE.
// =================================================================================================

// ----- EntityIsBodyBound(world, id): does `id` carry a BOUND BodyRef (a sim body renders its hull)? ----------
// A body-bound entity's hull is already drawn by gjk::HullToRenderInstances(world.sim) (the sim-body soup); its
// gameplay overlay would double-draw at the same place, so VerdictToRenderInstances overlays ONLY the NON-body
// entities (pickups, the player marker, gameplay-only spawns). A dead/unknown id reports false (skipped).
inline bool EntityIsBodyBound(const VerdictWorld& world, EntityId id) {
    auto it = world.handle.find(id);
    if (it == world.handle.end()) return false;
    const ecs::Entity e = it->second;
    if (!world.reg.valid(e)) return false;
    if (!world.reg.has<BodyRef>(e)) return false;
    const uint32_t bi = world.reg.get<BodyRef>(e).simBodyIndex;
    return bi != kNoBody && (size_t)bi < world.sim.bodies.size();
}

// ----- AppendMarkerCube(out, cx, cy, cz, half, color): emit a small flat-shaded cube into the soup -----------
// A pure host-float helper (render-only, OUTSIDE the bit-exact loop) that appends a 12-triangle axis-aligned
// cube (centred at (cx,cy,cz), half-extent `half`) into the gjk::HullRenderMesh soup, each face flat-shaded with
// an OUTWARD normal + the given matte color — the SAME HullRenderVertex layout gjk::HullToRenderMesh emits, so a
// marker draws through the EXISTING lit pipeline VERBATIM. The 8 corners (sign sweep ix*4+iy*2+iz) + 6 quad
// faces with explicit outward normals (no centroid flip needed — axis-aligned). Deterministic: a pure function
// of its float args (same inputs -> byte-equal verts), so two VerdictToRenderInstances calls are byte-equal.
inline void AppendMarkerCube(gjk::HullRenderMesh& out, float cx, float cy, float cz, float half,
                             const float color[3]) {
    float c[8][3];
    for (int i = 0; i < 8; ++i) {
        c[i][0] = cx + ((i & 4) ? half : -half);
        c[i][1] = cy + ((i & 2) ? half : -half);
        c[i][2] = cz + ((i & 1) ? half : -half);
    }
    // 6 faces: {4 corner indices, outward normal}. Winding is CCW as seen from outside.
    struct Face { uint32_t a, b, d, e; float n[3]; };
    const Face faces[6] = {
        {0, 1, 3, 2, {-1, 0, 0}},   // -x
        {4, 6, 7, 5, { 1, 0, 0}},   // +x
        {0, 4, 5, 1, { 0,-1, 0}},   // -y
        {2, 3, 7, 6, { 0, 1, 0}},   // +y
        {0, 2, 6, 4, { 0, 0,-1}},   // -z
        {1, 5, 7, 3, { 0, 0, 1}},   // +z
    };
    auto emit = [&](uint32_t idx, const float n[3]) {
        gjk::HullRenderVertex hv;
        hv.pos[0] = c[idx][0]; hv.pos[1] = c[idx][1]; hv.pos[2] = c[idx][2];
        hv.normal[0] = n[0]; hv.normal[1] = n[1]; hv.normal[2] = n[2];
        hv.color[0] = color[0]; hv.color[1] = color[1]; hv.color[2] = color[2];
        out.verts.push_back(hv);
    };
    for (const Face& f : faces) {
        // Quad (a,b,d,e) -> two triangles (a,b,d) + (a,d,e).
        emit(f.a, f.n); emit(f.b, f.n); emit(f.d, f.n);
        emit(f.a, f.n); emit(f.d, f.n); emit(f.e, f.n);
    }
}

// ----- VerdictToRenderInstances(world): the spec-named entry — the render payload for the VD6 game showcase. --
// The render of the bit-exact deterministic game: the sim bodies via the FROZEN gjk::HullToRenderInstances(
// world.sim) (the WH6/CD6/MF6/GJ6 delegate, MATTE) PLUS a gameplay-entity overlay — a small marker cube per
// LIVE NON-body entity (in order[] sequence) at its Transform2D.pos>>kFrac (a Pickup tinted warm by its value;
// the player — the entity carrying a Score — tinted by its Score; other gameplay entities a cool neutral).
// Body-bound entities are NOT overlaid (their hull is already in the sim-body soup). A PURE FUNCTION of `world`
// (two calls byte-equal — gjk::HullRenderMeshEqual; render-only float, OUTSIDE the bit-exact loop). The integer
// sim/world is UNTOUCHED (a pure read — VerdictStatesEqual(before, after) holds). NO new shader, NO new RHI.
inline gjk::HullRenderMesh VerdictToRenderInstances(const VerdictWorld& world) {
    // (1) the sim bodies — the FROZEN render bridge (matte), verbatim.
    gjk::HullRenderMesh out = gjk::HullToRenderInstances(world.sim);

    // (2) the gameplay-entity overlays — a marker cube per LIVE NON-body entity, IN order[] SEQUENCE (the pinned
    // deterministic iteration order, so the overlay verts append in a fixed order -> a pure function of `world`).
    const float kMarkerHalf = 0.35f;   // a small distinct marker (render-only float; smaller than a unit hull)
    for (size_t i = 0; i < world.order.size(); ++i) {
        const EntityId id = world.order[i];
        if (!IsLive(world, id)) continue;
        if (EntityIsBodyBound(world, id)) continue;   // its hull is already in the sim-body soup
        auto it = world.handle.find(id);
        if (it == world.handle.end()) continue;
        const ecs::Entity e = it->second;
        if (!world.reg.has<Transform2D>(e)) continue;
        const Transform2D& xf = world.reg.get<Transform2D>(e);
        const float px = (float)(xf.pos.x >> kFrac);
        const float py = (float)(xf.pos.y >> kFrac);
        const float pz = (float)(xf.pos.z >> kFrac);

        float color[3];
        if (world.reg.has<Score>(e)) {
            // The player: tint by Score (more points -> warmer/brighter green-gold; the score-driven highlight).
            const int32_t s = world.reg.get<Score>(e).points;
            const float t = (s > 8) ? 1.0f : (s < 0 ? 0.0f : (float)s / 8.0f);   // clamp 0..1
            color[0] = 0.85f; color[1] = 0.45f + 0.40f * t; color[2] = 0.12f;     // amber -> gold
        } else if (world.reg.has<Pickup>(e)) {
            // A pickup: tint warm by its value (a brighter teal-green for the higher-value collectibles).
            const int32_t v = world.reg.get<Pickup>(e).value;
            const float t = (v > 5) ? 1.0f : (v < 0 ? 0.0f : (float)v / 5.0f);
            color[0] = 0.15f; color[1] = 0.55f + 0.30f * t; color[2] = 0.45f;
        } else {
            color[0] = 0.45f; color[1] = 0.48f; color[2] = 0.52f;   // a cool neutral gameplay marker
        }
        AppendMarkerCube(out, px, py, pz, kMarkerHalf, color);
    }

    out.triangles = (uint32_t)(out.verts.size() / 3);
    return out;
}

// =================================================================================================
// Slice DX5 — THE RECORD->REPLAY->ASSERT-DETERMINISM HARNESS (APPEND-ONLY; VD1-VD6 above are
// BYTE-FROZEN). The 5th slice of FLAGSHIP #31 (THE AGENT EXPERIENCE / AX PRODUCT, slice-tag DX) —
// productizing the lockstep MOAT (flagship #27) as a testable agent artifact.
//
// VD1-VD6 built the deterministic entity world, the gameplay systems, the composed StepWorld tick,
// the heterogeneous snapshot, the whole-world lockstep/rollback, and the lit render. DX5 turns that
// determinism into a PRODUCT GUARANTEE an agent codes against: a command stream is RECORDED against
// the canonical gameplay world, the engine REPLAYS it (RunVerdictLockstep), and the final world is
// digested + asserted bit-identical against a committed golden replay file. A float engine cannot
// ship this: a non-deterministic sim has no reproducible "replay -> bit-identical" guarantee.
//
// THE DESIGN: a FIXED canonical scene builder (BuildCanonicalReplay) that record + verify (and the
// test) ALL call, so the replay file need only carry {stream, ticks, finalDigest} — the initial
// state is IMPLIED by the canonical builder (serializing arbitrary initial worlds is a future
// extension, out of scope). The digest is over the FINAL world (the moat's payload: two machines
// that replay the same stream produce the same final world -> the same digest). DigestSnapshot
// walks the snapshot in the order[]-KEYED canonical sequence the determinism contract already uses,
// so it is a pure function of the deterministic final state — stable under ECS handle churn,
// backend-identical. Tamper-detect (the falsifiable beat): flip one command's arg -> the replayed
// final digest DIFFERS from finalDigest (VerifyReplay returns false).
//
// PURE CPU INTEGER + ONE 16-hex digest. The FNV-1a is a verdict-LOCAL mirror of introspect.cpp's
// contentHash (same constants) — NO render/introspect include is pulled in. SerializeReplay/
// ParseReplay are deterministic JSON {ticks, stream:[{tick,kind,target,arg}], finalDigest} over the
// FROZEN Command wire format (serialized, NEVER changed). NO new RHI, NO new shader, NO Metal
// golden; verdict.h is APPEND-ONLY (VD1-VD6 byte-frozen).
// =================================================================================================

// ----- DigestFnv: the verdict-LOCAL FNV-1a 64-bit (the introspect.cpp contentHash constants) -------
// A pure-integer running mix — NO render/introspect include (the no-cross-dependency contract). The
// SAME 64-bit FNV-1a offset/prime as engine/editor/introspect.cpp's AgentFnv (1469598103934665603 /
// 1099511628211), so the digest discipline matches the engine's other deterministic key/hash sites.
struct DigestFnv {
    uint64_t h = 1469598103934665603ull;
    void mix(uint64_t v) {
        // Fold the 8 bytes of v LSB-first (a fixed byte order -> endianness-independent + stable).
        for (int b = 0; b < 8; ++b) {
            h ^= (uint64_t)(uint8_t)(v >> (b * 8));
            h *= 1099511628211ull;
        }
    }
    void mix32(uint32_t v) { mix((uint64_t)v); }
    void sep() {                                  // an unambiguous separator (the introspect AgentFnv idiom)
        h ^= 0x1Full;
        h *= 1099511628211ull;
    }
};

// ----- Fold a Q16.16 FxVec3 / FxQuat / FxBody into a DigestFnv (fixed field order) ------------------
inline void DigestVec3(DigestFnv& d, const FxVec3& v) {
    d.mix32((uint32_t)v.x); d.mix32((uint32_t)v.y); d.mix32((uint32_t)v.z);
}
inline void DigestQuat(DigestFnv& d, const FxQuat& q) {
    d.mix32((uint32_t)q.x); d.mix32((uint32_t)q.y); d.mix32((uint32_t)q.z); d.mix32((uint32_t)q.w);
}
inline void DigestBody(DigestFnv& d, const fpx::FxBody& b) {
    DigestVec3(d, b.pos); DigestVec3(d, b.vel);
    d.mix32((uint32_t)b.invMass); d.mix32(b.flags); d.mix32((uint32_t)b.radius);
    DigestQuat(d, b.orient); DigestVec3(d, b.angVel);
}

// ----- DigestSnapshot(s) -> 16-hex FNV-1a over a CANONICAL byte-encoding of the WHOLE snapshot -------
// A compact, deterministic, backend-agnostic fingerprint of the final world. It walks the snapshot in
// the order[]-KEYED canonical sequence the determinism contract uses (NOT the ECS dense order): the
// header (tick, nextId, order[]), THEN every component pool IN order[] sequence (each CompEntry vector
// is already serialized in order[] sequence by SnapshotWorld, with a separator after each pool so the
// pools cannot alias), THEN the sim TRIPLE (bodies + cache entries [key + impulse] + per-body sleep
// state). Two snapshots that are VerdictSnapshotsEqual hash IDENTICALLY (the digest is a pure function
// of the determinism-relevant state); a restored world (whose ecs handles churn) digests identically
// because nothing here reads the opaque handle. Returns the 16-hex-digit lowercase string.
inline std::string DigestSnapshot(const VerdictSnapshot& s) {
    DigestFnv d;
    // (1) header.
    d.mix32(s.tick);
    d.mix32(s.nextId);
    d.mix32((uint32_t)s.order.size());
    for (size_t i = 0; i < s.order.size(); ++i) d.mix32((uint32_t)s.order[i]);
    d.sep();

    // (2) every component pool, IN order[] sequence (the SnapshotWorld serialization order). Each
    // entry folds the EntityId key then the component fields; a separator follows each pool.
    for (const auto& c : s.transforms) { d.mix32((uint32_t)c.id); DigestVec3(d, c.value.pos); DigestQuat(d, c.value.orient); }
    d.sep();
    for (const auto& c : s.healths)    { d.mix32((uint32_t)c.id); d.mix32((uint32_t)c.value.hp); }
    d.sep();
    for (const auto& c : s.bodyRefs)   { d.mix32((uint32_t)c.id); d.mix32(c.value.simBodyIndex); }
    d.sep();
    for (const auto& c : s.velocities) { d.mix32((uint32_t)c.id); DigestVec3(d, c.value.vel); }
    d.sep();
    for (const auto& c : s.pickups)    { d.mix32((uint32_t)c.id); d.mix32((uint32_t)c.value.value); }
    d.sep();
    for (const auto& c : s.scores)     { d.mix32((uint32_t)c.id); d.mix32((uint32_t)c.value.points); }
    d.sep();

    // (3) the sim TRIPLE (bodies + cache + sleep), in fixed vector order.
    d.mix32((uint32_t)s.simSnap.bodies.size());
    for (const auto& b : s.simSnap.bodies) DigestBody(d, b);
    d.sep();
    d.mix32((uint32_t)s.simSnap.cache.entries.size());
    for (const auto& e : s.simSnap.cache.entries) {
        d.mix32(e.key.bodyA); d.mix32(e.key.bodyB); d.mix32(e.key.refFaceId); d.mix32(e.key.incVertId);
        d.mix32((uint32_t)e.normalImpulse);
    }
    d.sep();
    d.mix32((uint32_t)s.simSnap.sleep.size());
    for (const auto& sl : s.simSnap.sleep) {
        d.mix32((uint32_t)sl.energy); d.mix32(sl.lowEnergyTicks); d.mix32(sl.asleep ? 1u : 0u);
    }

    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)d.h);
    return std::string(buf);
}

// ----- ReplayFile: the deterministic replay regression artifact --------------------------------------
// {ticks, stream, finalDigest}. The initial world is IMPLIED by BuildCanonicalReplay (both record +
// verify build the identical world0 from it), so only the input stream + tick count + the expected
// final-world digest are serialized. The Command wire format {tick,kind,target,arg} is the FROZEN VD1
// layout — serialized verbatim, NEVER changed.
struct ReplayFile {
    uint32_t             ticks = 0;
    std::vector<Command> stream;
    std::string          finalDigest;
};

// ----- SerializeReplay(rf) -> deterministic JSON text (LF, the house integer style) ------------------
// {ticks, stream:[{tick,kind,target,arg:[x,y,z]}], finalDigest}. arg is the Q16.16 FxVec3 emitted as
// INTEGERS (the raw fixed-point words — NO float, so the round-trip is exact). FIXED key order +
// 2-space indent -> byte-identical run-to-run (the introspect/scene_io house style). NO BOM, LF only.
inline std::string SerializeReplay(const ReplayFile& rf) {
    std::string o;
    o.reserve(256 + rf.stream.size() * 64);
    o += "{\n";
    o += "  \"ticks\": " + std::to_string(rf.ticks) + ",\n";
    o += "  \"stream\": [";
    for (size_t i = 0; i < rf.stream.size(); ++i) {
        const Command& c = rf.stream[i];
        o += (i ? ",\n" : "\n");
        o += "    { \"tick\": " + std::to_string(c.tick);
        o += ", \"kind\": " + std::to_string(c.kind);
        o += ", \"target\": " + std::to_string((uint32_t)c.target);
        o += ", \"arg\": [" + std::to_string((int64_t)(int32_t)c.arg.x) + ", " +
             std::to_string((int64_t)(int32_t)c.arg.y) + ", " +
             std::to_string((int64_t)(int32_t)c.arg.z) + "] }";
    }
    o += (rf.stream.empty() ? "],\n" : "\n  ],\n");
    o += "  \"finalDigest\": \"" + rf.finalDigest + "\"\n";
    o += "}\n";
    return o;
}

// ----- ParseReplay(text) -> ReplayFile: the inverse (the vendored json.h parser) --------------------
// Tolerant of arbitrary key order / whitespace (the parser handles it); reads ticks, the stream array
// of {tick,kind,target,arg:[x,y,z]} (arg parsed as INTEGER fixed-point words), and finalDigest. A
// malformed / missing field yields a default (0 / empty) — VerifyReplay then simply fails the digest
// compare (the deterministic, non-crashing posture; never a crash). Uses the vendored single-header
// json.h (a json_weak / inline C TU — safe to include in a header). SerializeReplay -> ParseReplay is
// a byte-faithful round-trip of {ticks, stream, finalDigest}.
namespace replay_detail {

inline const json_value_s* Member(const json_object_s* obj, const char* key) {
    if (!obj) return nullptr;
    for (const json_object_element_s* el = obj->start; el; el = el->next)
        if (el->name && std::strcmp(el->name->string, key) == 0) return el->value;
    return nullptr;
}
inline int64_t NumberOf(const json_value_s* v, int64_t fallback) {
    if (!v || v->type != json_type_number) return fallback;
    return (int64_t)std::strtoll(static_cast<const json_number_s*>(v->payload)->number, nullptr, 10);
}

}  // namespace replay_detail

inline ReplayFile ParseReplay(const std::string& text) {
    ReplayFile rf;
    json_value_s* root = json_parse(text.data(), text.size());
    if (!root) return rf;                       // malformed -> empty (VerifyReplay then fails the compare)
    struct FreeGuard { void* p; ~FreeGuard() { std::free(p); } } guard{root};
    if (root->type != json_type_object) return rf;
    const json_object_s* obj = static_cast<const json_object_s*>(root->payload);

    rf.ticks = (uint32_t)replay_detail::NumberOf(replay_detail::Member(obj, "ticks"), 0);

    const json_value_s* fd = replay_detail::Member(obj, "finalDigest");
    if (fd && fd->type == json_type_string) {
        const json_string_s* s = static_cast<const json_string_s*>(fd->payload);
        rf.finalDigest.assign(s->string, s->string_size);
    }

    const json_value_s* st = replay_detail::Member(obj, "stream");
    if (st && st->type == json_type_array) {
        const json_array_s* arr = static_cast<const json_array_s*>(st->payload);
        for (const json_array_element_s* el = arr->start; el; el = el->next) {
            if (el->value->type != json_type_object) continue;
            const json_object_s* co = static_cast<const json_object_s*>(el->value->payload);
            Command c;
            c.tick   = (uint32_t)replay_detail::NumberOf(replay_detail::Member(co, "tick"), 0);
            c.kind   = (uint32_t)replay_detail::NumberOf(replay_detail::Member(co, "kind"), 0);
            c.target = (EntityId)replay_detail::NumberOf(replay_detail::Member(co, "target"), 0);
            const json_value_s* ag = replay_detail::Member(co, "arg");
            if (ag && ag->type == json_type_array) {
                const json_array_s* aa = static_cast<const json_array_s*>(ag->payload);
                fx comp[3] = {0, 0, 0};
                int i = 0;
                for (const json_array_element_s* ae = aa->start; ae && i < 3; ae = ae->next, ++i)
                    comp[i] = (fx)(int32_t)replay_detail::NumberOf(ae->value, 0);
                c.arg = FxVec3{comp[0], comp[1], comp[2]};
            }
            rf.stream.push_back(c);
        }
    }
    return rf;
}

// ----- BuildCanonicalReplay(): the FIXED canonical replay scene ------------------------------------
// A VD3/VD5-style composed gameplay+physics world0 + VerdictParams + a FIXED command stream + tick
// count. Record AND verify (AND the test) build the IDENTICAL world0 from this, so the replay file
// needs only {stream, ticks, finalDigest}. Returns the pieces by out-params (VerdictWorld is
// non-copyable, so world0 is filled in place). The scene MIRRORS the VD5 --vd5-net showcase: a static
// support body, a dynamic player body, a small hull stack, plus gameplay entities (player Health+Score,
// two pickups, a mover) so EVERY component type is exercised by the digest.
struct CanonicalReplay {
    VerdictParams        params;
    std::vector<Command> stream;
    uint32_t             ticks = 0;
    EntityId             player = kNoEntity;
};

inline CanonicalReplay BuildCanonicalReplay(VerdictWorld& world0) {
    auto fi = [](int v) -> fx { return (fx)((int64_t)v * (int64_t)kOne); };
    auto V  = [&](int x, int y, int z) { return FxVec3{fi(x), fi(y), fi(z)}; };
    const FxQuat kIdentity{0, 0, 0, kOne};

    // The deterministic warm+sleep config (== the VD3/VD4/VD5 lineage; angDamp OFF).
    const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
    convex::ConvexStepConfig stepCfg;
    stepCfg.gravity     = FxVec3{0, kGravY, 0};
    stepCfg.dt          = kOne / 60;
    stepCfg.solveIters  = 8;
    stepCfg.restitution = 0;
    stepCfg.slop        = kOne / 64;
    stepCfg.beta        = (fx)((int64_t)2 * kOne / 10);
    stepCfg.linDamp     = (fx)((int64_t)95 * kOne / 100);
    stepCfg.angDamp     = kOne;
    stepCfg.posIters    = 4;
    warmhull::HullSleepConfig cfg;
    cfg.warm           = stepCfg;
    cfg.sleepThreshold = kOne;
    cfg.wakeThreshold  = (fx)(2 * (int)kOne);
    cfg.sleepTicks     = 30;

    const HazardRegion hazard{V(-9,-9,0).x, V(-9,-9,0).y, V(-9,-9,0).x, V(-9,-9,0).y};  // empty band
    const fx collectR = kOne;
    const int stackN = 2;

    // The composed sim scene: body 0 = static support; body 1 = the player's dynamic hull; bodies
    // 2..1+stackN = a small stack of dynamic boxes resting above.
    world0 = VerdictWorld{};
    {
        gjk::HullWorld sim;
        { fpx::FxBody b; b.pos={0,0,0}; b.orient={0,0,0,kOne}; b.invMass=0; b.flags=0u; b.vel={0,0,0}; b.angVel={0,0,0}; sim.bodies.push_back(b); }
        sim.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));   // 0 static support
        { fpx::FxBody b; b.pos={0, fi(3), 0}; b.orient={0,0,0,kOne}; b.invMass=kOne; b.flags=fpx::kFlagDynamic; b.vel={0,0,0}; b.angVel={0,0,0}; sim.bodies.push_back(b); }
        sim.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));   // 1 dynamic PLAYER body
        for (int k = 0; k < stackN; ++k) {
            fpx::FxBody b; b.pos={0, fi(5 + 2 * k), 0}; b.orient={0,0,0,kOne};
            b.invMass=kOne; b.flags=fpx::kFlagDynamic; b.vel={0,0,0}; b.angVel={0,0,0};
            sim.bodies.push_back(b);
            sim.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));   // 2..1+stackN stacked boxes
        }
        world0.sim = sim;
    }

    // Gameplay entities: support (body 0), player (body 1, Health+Score), a pickup near the player (a
    // collect fires + despawns it + bumps Score), a second pickup, a mover with a Velocity2D, then the
    // stacked boxes — so EVERY component type is in the digest.
    const EntityId support = SpawnEntity(world0, Transform2D{V(0,0,0), kIdentity});
    BindBody(world0, support, 0u);
    const EntityId player = SpawnEntity(world0, Transform2D{V(0,3,0), kIdentity}, Health{77}, BodyRef{kNoBody});
    world0.reg.add<Score>(world0.handle.at(player), Score{0});
    BindBody(world0, player, 1u);
    const EntityId pk0 = SpawnEntity(world0, Transform2D{V(0,3,0), kIdentity});
    world0.reg.add<Pickup>(world0.handle.at(pk0), Pickup{5});
    const EntityId pk1 = SpawnEntity(world0, Transform2D{V(5,5,0), kIdentity});
    world0.reg.add<Pickup>(world0.handle.at(pk1), Pickup{3});
    const EntityId mover = SpawnMover(world0, Transform2D{V(4,4,0), kIdentity}, Health{10}, Velocity2D{V(0,0,0)});
    for (int k = 0; k < stackN; ++k) {
        const EntityId se = SpawnEntity(world0, Transform2D{V(0, 5 + 2 * k, 0), kIdentity});
        BindBody(world0, se, (uint32_t)(2 + k));
    }
    (void)support; (void)pk0; (void)pk1; (void)mover;

    CanonicalReplay cr;
    cr.player = player;
    cr.ticks  = 24u;
    cr.params = VerdictParams{hazard, player, collectR, cfg, world0.sim.hulls};
    // The FIXED command stream: a kCmdImpulse nudges the player body, a kCmdSpawn adds an entity.
    cr.stream = {
        Command{0u, kCmdImpulse, player, V(1,0,0)},        // nudge the player body
        Command{2u, kCmdSpawn,   kNoEntity, V(-3,2,0)},    // a deterministic gameplay spawn
    };
    return cr;
}

// ----- VerifyReplay(rf): rebuild the canonical world0 -> lockstep over rf.stream -> compare digest ---
// The product guarantee: rebuild the FIXED canonical world0 (BuildCanonicalReplay), RunVerdictLockstep
// over rf.stream for rf.ticks, DigestSnapshot the converged authority, and return whether it EQUALS
// rf.finalDigest. A changed input (a tampered command) OR a changed engine BOTH move the digest -> the
// regression guard. (The lockstep also re-derives a replica from inputs alone; the authority==replica
// equality is the moat, asserted separately by the driver via RunVerdictLockstep's outIdentical.)
inline bool VerifyReplay(const ReplayFile& rf) {
    VerdictWorld world0;
    const CanonicalReplay cr = BuildCanonicalReplay(world0);
    const VerdictSnapshot world0Snap = SnapshotWorld(world0);
    const VerdictSnapshot finalSnap = RunVerdictLockstep(world0Snap, cr.params, rf.stream, rf.ticks);
    return DigestSnapshot(finalSnap) == rf.finalDigest;
}

// =================================================================================================
// Slice DX6 — THE DETERMINISM-STRESS FUZZER — THE CAPSTONE (APPEND-ONLY; VD1-VD6 + the DX5 additions
// above are BYTE-FROZEN). The 6th and FINAL slice of FLAGSHIP #31 (THE AGENT EXPERIENCE / AX PRODUCT,
// hf::game::verdict). DX1-DX5 shipped the contract + read + author + reload + replay surfaces. DX6 is
// the marketable CAPSTONE: a DETERMINISM-STRESS HARNESS that, for the FIXED canonical command stream,
// asserts that replaying through ANY rollback/snapshot point yields the bit-identical final world as a
// straight lockstep run — a bounded, seeded, reproducible fuzzer that stress-tests the moat itself and
// emits a golden "stress report" (every snapshot point -> corrected==authority). It is the
// deterministic, golden-able equivalent of a fuzzer: a regression that breaks determinism at ANY
// rollback boundary moves a `corrected` bit and breaks the golden.
//
// THE DESIGN: sweep `rollbackAt` over 0..ticks INCLUSIVE (ticks+1 points), each calling the FROZEN
// RunVerdictRollback (the VD5 per-point oracle: its outCorrectedEqAuthority + outMispredictDiverged
// out-params). For a deterministic engine EVERY point's `corrected` is true (the moat guarantee:
// snapshot+mispredict+rollback ANYWHERE recovers the authority bit-exactly). The `diverged` flag makes
// the fuzzer NON-VACUOUS — it records whether the injected misprediction actually perturbed the world
// at that point; with a REAL (perturbed) mispredict stream early points diverge, the last point
// (rollbackAt==ticks, zero speculative ticks) cannot diverge -> allDiverged < points is EXPECTED. The
// "fuzzer fires" control: mispredict==auth -> diverged false at EVERY point (the flag is meaningful,
// not always-true), while corrected stays true in BOTH.
//
// PURE CPU: NO new render RHI, NO new shader, NO new compute. RunVerdictRollback + the VD4/VD5 machinery
// are reused VERBATIM. warmhull.h/ecs.h + ALL sim headers + ALL shaders BYTE-UNCHANGED; verdict.h is
// APPEND-ONLY (VD1-VD6 + DX5 code byte-frozen). This is the FINAL slice -> FLAGSHIP #31 COMPLETE.
// =================================================================================================

// ----- StressPoint: one rollback/snapshot point's verdict ----------------------------------------
// `rollbackAt` is the snapshot tick swept; `corrected` is whether RunVerdictRollback recovered the
// authority bit-exactly from that point (MUST be true at EVERY point); `diverged` is whether the
// injected misprediction actually perturbed the speculative world at that point (the non-vacuous flag).
struct StressPoint {
    uint32_t rollbackAt = 0;
    bool     corrected  = false;
    bool     diverged   = false;
};

// ----- StressReport: the full per-point PASS matrix (the golden artifact) --------------------------
// `points` == ticks+1 (the inclusive sweep); `allCorrected` == count of points where corrected==true
// (MUST equal points for a deterministic engine — the headline); `allDiverged` == count where
// diverged==true (EXPECTED < points: the last point has zero speculative ticks so cannot diverge).
struct StressReport {
    uint32_t                 points       = 0;
    uint32_t                 allCorrected = 0;
    uint32_t                 allDiverged  = 0;
    std::vector<StressPoint> perPoint;
};

// ----- RunDeterminismStress(world0Snap, params, authStream, mispredictStream, ticks) -> StressReport -
// THE capstone sweep: for rollbackAt in [0, ticks] (ticks+1 points), call the FROZEN VD5
// RunVerdictRollback(world0Snap, params, authStream, mispredictStream, ticks, rollbackAt, &corrected,
// &diverged) and record a StressPoint. Tallies allCorrected / allDiverged. Pure CPU, deterministic:
// the sweep order + the fixed streams -> a byte-identical report run-to-run + cross-backend (verdict.h
// is render-symbol-free). A determinism regression anywhere in the rollback machinery flips a
// `corrected` bit -> the report bytes change -> the golden breaks (the fuzzer's regression value).
inline StressReport RunDeterminismStress(const VerdictSnapshot& world0Snap, const VerdictParams& params,
                                         const std::vector<Command>& authStream,
                                         const std::vector<Command>& mispredictStream, uint32_t ticks) {
    StressReport report;
    report.points = ticks + 1u;
    report.perPoint.reserve(report.points);
    for (uint32_t rollbackAt = 0; rollbackAt <= ticks; ++rollbackAt) {
        bool corrected = false, diverged = false;
        (void)RunVerdictRollback(world0Snap, params, authStream, mispredictStream, ticks, rollbackAt,
                                 &corrected, &diverged);
        StressPoint p;
        p.rollbackAt = rollbackAt;
        p.corrected  = corrected;
        p.diverged   = diverged;
        if (corrected) ++report.allCorrected;
        if (diverged)  ++report.allDiverged;
        report.perPoint.push_back(p);
    }
    return report;
}

// ----- PerturbCanonicalStream(authStream) -> a REAL deterministic mispredict stream ----------------
// The mispredict stream = the canonical auth stream with ONE command perturbed at a FIXED index (the
// player nudge impulse, command 0) — its arg.x is flipped (negated minus kOne, the DX5 tamper convention)
// AND its kind is left a sim verb so the divergence spans physics. A real, deterministic divergence that
// the harness's `diverged` flag detects. The empty-stream case returns the input unchanged (no perturb).
inline std::vector<Command> PerturbCanonicalStream(const std::vector<Command>& authStream) {
    std::vector<Command> mispredict = authStream;
    if (!mispredict.empty())
        mispredict[0].arg.x = -mispredict[0].arg.x - kOne;   // a physics-divergent flip at a fixed index
    return mispredict;
}

// ----- SerializeStressReport(r) -> the deterministic stress_report.json (LF-clean, no BOM) ----------
// The golden PASS matrix: {points, allCorrected, allDiverged, perPoint:[{rollbackAt, corrected,
// diverged}]} — a pure function of the report (same report -> byte-equal text). NO filesystem paths
// (invocation-independent). Two-space indent, LF line endings, booleans as true/false. Backend-agnostic
// (only integer/bool fields) -> byte-identical on Vulkan and Metal.
inline std::string SerializeStressReport(const StressReport& r) {
    std::string s;
    s += "{\n";
    s += "  \"points\": " + std::to_string(r.points) + ",\n";
    s += "  \"allCorrected\": " + std::to_string(r.allCorrected) + ",\n";
    s += "  \"allDiverged\": " + std::to_string(r.allDiverged) + ",\n";
    s += "  \"perPoint\": [\n";
    for (size_t i = 0; i < r.perPoint.size(); ++i) {
        const StressPoint& p = r.perPoint[i];
        s += "    { \"rollbackAt\": " + std::to_string(p.rollbackAt) +
             ", \"corrected\": " + (p.corrected ? "true" : "false") +
             ", \"diverged\": "  + (p.diverged  ? "true" : "false") + " }";
        s += (i + 1 < r.perPoint.size()) ? ",\n" : "\n";
    }
    s += "  ]\n";
    s += "}\n";
    return s;
}

// =================================================================================================
// Issue #40 — DETERMINISTIC COLLISION-EVENT API — OnHit(actorA, actorB, point, impulse) (APPEND-ONLY;
// VD1-VD6 + DX5/DX6 above are BYTE-FROZEN). A derived, per-tick gameplay OUTPUT layered on the existing
// determinism contract — NOT a new slice of any flagship, NOT a new field on VerdictWorld, and NOT a
// change to StepWorld.
//
// THE PROBLEM: gameplay samples that want "did entity A hit entity B, and how hard?" today re-roll their
// own AABB / distance checks on top of the sim — duplicating (and risking diverging from) the contacts
// the deterministic sim ALREADY solves bit-exactly. warmhull::StepWarmSleepHullWorld REWRITES
// world.cache to EXACTLY this tick's solved contact manifold (one CachedHullContact per contact POINT,
// keyed by an order-normalized HullContactKey{bodyA<bodyB, refFaceId, incVertId} + the solved
// normalImpulse). That cache is part of the snapshot/lockstep state (the persist/VD4 lessons proved it is
// bit-identical cross-platform + replayable), so a pure read of it after StepWorld is ALSO bit-identical.
//
// THE DESIGN: CollectHitEvents(world) is a PURE FUNCTION of the POST-StepWorld state (world.cache +
// world.order + world.sim.bodies). It resolves each cache contact's two GLOBAL sim-body indices back to
// the gameplay EntityIds bound to them (via world.order + BodyRef), aggregates per ENTITY-PAIR (summing
// normalImpulse over the manifold's up-to-4 points -> one HitEvent per colliding pair = the collision
// strength), and returns the events in a FIXED canonical (a,b)-ascending order. Gameplay calls StepWorld
// then CollectHitEvents — the OnHit list, with NO bespoke overlap test.
//
// HONEST SCOPE: HitEvent.point is the deterministic MIDPOINT of the two bodies' positions (an integer
// Q16.16 average), NOT a per-contact manifold point — precise contact points are re-derivable via the
// manifold query later; the midpoint is the cheap, deterministic, good-enough "where" for a gameplay
// event. A contact between two STATIC-only / unbound bodies (no gameplay entity on one side) is SKIPPED
// (no entity to fire OnHit on). This adds NO field to VerdictWorld and modifies NO existing function ->
// the VD4 snapshot / VD5 lockstep determinism is untouched (HitEvents are a derived per-tick OUTPUT,
// recomputed each call, NEVER part of the snapshot). PURE CPU INTEGER (strict 0px): NO new render RHI,
// NO new shader, NO new compute. All sim headers + ecs.h BYTE-UNCHANGED; verdict.h is APPEND-ONLY.
// =================================================================================================

// ----- HitEvent: a deterministic collision event (the per-pair OnHit gameplay reads after StepWorld) -
// The two colliding entities in canonical order (a < b by EntityId — a FIXED total order), the contact
// LOCATION (the deterministic midpoint of the two bodies' positions, Q16.16), and the summed
// normalImpulse over the pair's contact points (Q16.16 — the collision STRENGTH). All integer.
struct HitEvent {
    EntityId a = kNoEntity;   // the two colliding entities (a < b by EntityId, deterministic order)
    EntityId b = kNoEntity;
    FxVec3   point{};         // contact location = deterministic midpoint of the two bodies' positions (Q16.16)
    fx       impulse = 0;     // summed normalImpulse over the pair's contact points (Q16.16) — collision strength
};

// ----- HitEventsEqual / HitEventVecsEqual: field-wise equality (the two-run BYTE-IDENTICAL proof) -----
// Padding-safe field-wise compares (the POD HitEvent has padding between EntityId/FxVec3/fx, so NO
// memcmp). Two HitEvent vectors are equal iff the SAME sequence of {a, b, point, impulse}.
inline bool HitEventsEqual(const HitEvent& x, const HitEvent& y) {
    return x.a == y.a && x.b == y.b &&
           x.point.x == y.point.x && x.point.y == y.point.y && x.point.z == y.point.z &&
           x.impulse == y.impulse;
}
inline bool HitEventVecsEqual(const std::vector<HitEvent>& a, const std::vector<HitEvent>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!HitEventsEqual(a[i], b[i])) return false;
    return true;
}

// ----- CollectHitEvents(world) -> the per-pair OnHit list, canonical (a,b)-ascending -----------------
// A PURE FUNCTION of the POST-StepWorld state (world.cache + world.order + world.sim.bodies). Steps:
//   (1) Build a FIXED-order reverse map sim-body-index -> EntityId by scanning world.order (skip an
//       entity with no bound BodyRef / simBodyIndex==kNoBody / out-of-range). FIRST entity to claim a
//       body index wins (a 1:1 binding is the contract; a double-bind is a caller bug, resolved
//       deterministically by order[] position).
//   (2) Iterate world.cache.entries in FIXED order. Resolve key.bodyA / key.bodyB -> EntityIds via the
//       map; SKIP a contact where EITHER side is unmapped (e.g. a static-only support with no gameplay
//       entity). Aggregate per ENTITY-PAIR (canonical a<b by EntityId): SUM normalImpulse across that
//       pair's contact points (a manifold has up to 4 points -> ONE HitEvent per pair).
//   (3) point = the deterministic midpoint of the two bodies' positions: (posA+posB) averaged via the
//       Q16.16 helpers (FxAdd then a >>1 per component — an exact integer average, NO float). Honestly
//       the inter-body midpoint, NOT a per-contact manifold point.
//   (4) Return the events sorted by (a, b) ascending (a FIXED deterministic total order) -> canonical.
// Bit-identical cross-platform because world.cache + world.sim.bodies are (the persist/VD lessons).
inline std::vector<HitEvent> CollectHitEvents(const VerdictWorld& world) {
    const size_t nBodies = world.sim.bodies.size();

    // (1) the FIXED-order reverse map sim-body-index -> EntityId (FIRST claimant wins).
    std::vector<EntityId> bodyToEntity(nBodies, kNoEntity);
    for (size_t i = 0; i < world.order.size(); ++i) {
        const EntityId id = world.order[i];
        auto it = world.handle.find(id);
        if (it == world.handle.end()) continue;
        const ecs::Entity e = it->second;
        if (!world.reg.valid(e)) continue;
        if (!world.reg.has<BodyRef>(e)) continue;
        const uint32_t bi = world.reg.get<BodyRef>(e).simBodyIndex;
        if (bi == kNoBody || (size_t)bi >= nBodies) continue;   // unbound / out-of-range -> not a sim body
        if (bodyToEntity[(size_t)bi] == kNoEntity) bodyToEntity[(size_t)bi] = id;  // first wins (deterministic)
    }

    // (2) aggregate normalImpulse per ENTITY-PAIR over the cache, in FIXED cache order. The pair count is
    // tiny (a handful of colliding bodies); a flat vector + a fixed linear scan is the deterministic
    // baseline (the warmhull::HullCache shape — no hash-order nondeterminism).
    struct PairAccum { EntityId a; EntityId b; uint32_t bodyA; uint32_t bodyB; fx impulse; };
    std::vector<PairAccum> pairs;
    for (size_t c = 0; c < world.cache.entries.size(); ++c) {
        const warmhull::CachedHullContact& ce = world.cache.entries[c];
        const uint32_t bA = ce.key.bodyA, bB = ce.key.bodyB;
        if ((size_t)bA >= nBodies || (size_t)bB >= nBodies) continue;   // defensive (the key is sim-internal)
        const EntityId eA = bodyToEntity[(size_t)bA];
        const EntityId eB = bodyToEntity[(size_t)bB];
        if (eA == kNoEntity || eB == kNoEntity) continue;   // a static-only / unbound side -> no gameplay event
        if (eA == eB) continue;                              // a body cannot hit itself (defensive)
        // Canonical entity order (a < b by EntityId); carry the matching sim-body index for the midpoint.
        EntityId pa = eA, pb = eB; uint32_t pba = bA, pbb = bB;
        if (pb < pa) { EntityId te = pa; pa = pb; pb = te; uint32_t tb = pba; pba = pbb; pbb = tb; }
        bool found = false;
        for (size_t p = 0; p < pairs.size(); ++p) {
            if (pairs[p].a == pa && pairs[p].b == pb) {
                pairs[p].impulse = (fx)((int64_t)pairs[p].impulse + (int64_t)ce.normalImpulse);  // sum the manifold
                found = true;
                break;
            }
        }
        if (!found) pairs.push_back(PairAccum{pa, pb, pba, pbb, ce.normalImpulse});
    }

    // (3) one HitEvent per pair: point = the integer midpoint of the two bodies' positions (NO float).
    std::vector<HitEvent> events;
    events.reserve(pairs.size());
    for (size_t p = 0; p < pairs.size(); ++p) {
        const FxVec3& posA = world.sim.bodies[(size_t)pairs[p].bodyA].pos;
        const FxVec3& posB = world.sim.bodies[(size_t)pairs[p].bodyB].pos;
        const FxVec3 sum = fpx::FxAdd(posA, posB);
        HitEvent ev;
        ev.a       = pairs[p].a;
        ev.b       = pairs[p].b;
        ev.point   = FxVec3{(fx)(sum.x >> 1), (fx)(sum.y >> 1), (fx)(sum.z >> 1)};  // exact integer average
        ev.impulse = pairs[p].impulse;
        events.push_back(ev);
    }

    // (4) canonical total order: sort by (a, b) ascending. A self-contained insertion sort (the event
    // count is tiny — a handful of colliding pairs) keeps this header dependency-free (NO <algorithm>
    // added to the frozen include block) AND deterministic (a stable, fixed-comparator pass).
    auto less = [](const HitEvent& x, const HitEvent& y) {
        if (x.a != y.a) return x.a < y.a;
        return x.b < y.b;
    };
    for (size_t i = 1; i < events.size(); ++i) {
        HitEvent key = events[i];
        size_t j = i;
        while (j > 0 && less(key, events[j - 1])) { events[j] = events[j - 1]; --j; }
        events[j] = key;
    }
    return events;
}

}  // namespace verdict
}  // namespace hf::game
