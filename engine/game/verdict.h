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

}  // namespace verdict
}  // namespace hf::game
