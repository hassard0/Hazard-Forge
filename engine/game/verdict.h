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
#include <unordered_map>
#include <vector>

#include "ecs/ecs.h"           // read-only: ecs::Registry / ecs::Entity / create/get/has/remove (the sparse-set
                               // store VD1 layers the determinism contract over — NOT modified)
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

}  // namespace verdict
}  // namespace hf::game
