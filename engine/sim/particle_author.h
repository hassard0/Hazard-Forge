#pragma once
// Slice PA1 — PARTICLE-SYSTEM AUTHORING VIA THE FLOW VM (Track-S S11 of docs/SUPERIORITY_ROADMAP.md):
// Niagara-class emitter/behavior graphs, DETERMINISTIC. FLAGSHIP #19 (deterministic GPU particles,
// engine/sim/particles.h PT1-PT6) ships a bit-exact CPU/Vulkan/Metal + lockstep-replayable particle system —
// but configured ONLY from hand-written C++ structs (EmitterConfig / ForceField literals). PA1 wires the flow
// VM (engine/flow/flow.h — the deterministic node-graph execution VM, per-tick StepGraph with stateful nodes)
// as the particle AUTHORING system: a flow graph's per-tick OUTPUT REGISTERS drive the emitter + force-field
// parameters. Data-driven, bit-exact, lockstep-replayable particle EFFECTS authored as GRAPHS, not code —
// UE5's Niagara emitter graphs are float/non-deterministic (module order + float accumulation + timing), so
// two machines running the same Niagara system routinely diverge; a graph-authored effect whose entire
// evolution (graph state AND particle pool) is a pure integer function of (graph, bindings, input stream) is
// a capability UE5 structurally lacks.
//
// THE COMPOSITION (one deterministic seam, everything else reused VERBATIM):
//   StepAuthoredEffect(effect, pool, dt, tick, inputs):
//     (1) regs = flow::StepGraph(effect.graph, effect.state, inputs, tick)   — the per-tick VM tick
//     (2) resolve = COPY effect.baseCfg / baseFields / baseGravity / baseDragK, then for each ParamBinding
//         (in ARRAY ORDER — the deterministic contract; a later binding to the same param wins) overwrite
//         the bound parameter with the bound node's output register                     — the AUTHORING seam
//     (3) particles::StepParticles(pool, resolved...)                       — the bit-exact PT4 tick VERBATIM
// ZERO bindings -> step (2) copies the base config unchanged -> StepAuthoredEffect == StepParticles with the
// base config BIT-IDENTICAL (the identity control the test pins).
//
// THE UNIT CONTRACT (flow regs are int32; particle params are Q16.16 fx — the EXACT conversion, pinned):
//   * Q16.16 params (kParamEmitterX/Y/Z, kParamLifetime, kParamSpeed, kParamFieldStrength, kParamFieldX/Y/Z,
//     kParamGravityY, kParamDragK): the register IS the Q16.16 value — reg 65536 == 1.0 world units. A graph
//     computes world-unit values directly in Q16.16 via integer arithmetic (e.g. kMul by a Q16.16 constant).
//   * kParamSpawnPerTick: the register IS the plain integer COUNT (particles.h::Emit's documented dual
//     convention — a value < kOne is read as the raw count), NOT Q16.16. Negative -> Emit clamps to 0.
// Both are pure int32 reinterpretations (fx is int32) — NO scaling, NO rounding, NO float, ever.
//
// LOCKSTEP (the PT5 mold over the COMPOSED state): the snapshot is the pool (particles + freeList +
// spawnCursor + tick + cfg — the PT5 crux) PLUS the flow GraphState (the graph's persistent register file —
// counters/delays/latches ARE sim state: the snapshot-completeness control proves omitting it diverges).
// Commands write flow INPUT channels (the player-drives-the-effect story: a gameplay input flows through the
// AUTHORED graph into spawn rate / emitter position / field strength). A peer fed only (authored asset,
// command stream) re-derives BOTH the graph state and the particle pool bit-for-bit; rollback re-simulates
// from a snapshot bit-exact. PURE CPU — NO GPU dispatch, NO new shader, NO new RHI: the underlying PT passes
// keep their own GPU/CPU split untouched (this header only COMPOSES the existing bit-exact CPU references).
//
// REUSE MAP: particles.h (EmitterConfig/ForceField/ParticlePool/StepParticles/SnapshotParticles/
// RestoreParticles/ParticleStatesEqual — READ-ONLY, byte-untouched), flow.h (Graph/GraphState/StepGraph/
// MakeState/SerializeGraph — READ-ONLY, byte-untouched), editor/flow_edit_ops.h (AddFlowNode/ConnectFlow —
// the AUTHORING ops the showcase fixture is built with: the pulsing-fountain graph below is constructed
// EXCLUSIVELY through the editor edit-ops, and its SerializeGraph digest is pinned by the test — the
// "authored, not hardcoded" evidence), net/session.h (DigestBytes via flow.h). All header-only pure CPU.
// NOTE the one layering addition: sim/ includes editor/flow_edit_ops.h — deliberate, because PA1 IS the
// authoring bridge (the edit-ops are themselves pure-CPU header-only over flow.h alone, no ImGui/RHI).

#include <cstdint>
#include <cstring>   // std::memcmp — not strictly needed (ParticleStatesEqual handles the pool) but kept
                     // hoisted like particles.h for libstdc++ namespace hygiene
#include <vector>

#include "sim/particles.h"          // read-only: the bit-exact PT1-PT6 particle system (the driven layer)
#include "flow/flow.h"              // read-only: the deterministic node-graph VM (the driving layer)
#include "editor/flow_edit_ops.h"   // read-only: AddFlowNode/ConnectFlow — the graph AUTHORING ops

namespace hf::sim {
namespace pauthor {

// Reuse the particle/flow vocabulary verbatim (NO new fixed-point or graph primitives).
using particles::fx;
using particles::FxVec3;
inline constexpr int kFrac = particles::kFrac;
inline constexpr fx  kOne  = particles::kOne;

// ----- ParamKind: the drivable particle parameters (FIXED enum numbering — the wire/asset contract) -------
// The useful per-tick-drivable set of the PT4 StepParticles inputs. Emitter: spawn count + origin + lifetime
// + speed. Fields: per-field strength + center (fieldIndex selects which base field). Environment: gravity Y
// + linear drag. Collision geometry (groundY/radius/restitution/spheres) is deliberately NOT drivable in v1
// (static world, like the PT scenes). Units per the header contract above.
enum ParamKind : uint32_t {
    kParamSpawnPerTick  = 0,   // emitter ratePerTick — reg IS the plain integer count (Emit's raw-count convention)
    kParamEmitterX      = 1,   // emitter origin.x — reg IS the Q16.16 value
    kParamEmitterY      = 2,   // emitter origin.y — Q16.16
    kParamEmitterZ      = 3,   // emitter origin.z — Q16.16
    kParamLifetime      = 4,   // per-spawn lifetime — Q16.16 seconds
    kParamSpeed         = 5,   // per-spawn initial speed scale — Q16.16
    kParamFieldStrength = 6,   // baseFields[fieldIndex].strength — Q16.16 (signed)
    kParamFieldX        = 7,   // baseFields[fieldIndex].center.x — Q16.16
    kParamFieldY        = 8,   // baseFields[fieldIndex].center.y — Q16.16
    kParamFieldZ        = 9,   // baseFields[fieldIndex].center.z — Q16.16
    kParamGravityY      = 10,  // gravity.y — Q16.16 (world units / s^2)
    kParamDragK         = 11,  // linear drag coefficient — Q16.16
};

// ----- ParamBinding: ONE wire from a flow node's output register to a particle parameter ------------------
// `node` is the flow NodeId whose per-tick output register drives the parameter. `fieldIndex` selects which
// base force field for the kParamField* kinds (ignored otherwise). An out-of-range node or fieldIndex is a
// deterministic NO-OP at resolve time (never UB) — the "no edge" discipline.
struct ParamBinding {
    flow::NodeId node       = 0;                  // the flow node whose output register drives the param
    uint32_t     param      = kParamSpawnPerTick; // ParamKind
    uint32_t     fieldIndex = 0;                  // which baseFields entry (kParamField* only)
};

// ----- AuthoredEffect: the authored particle-effect ASSET + its live graph state --------------------------
// The STATIC authored artifact (graph + bindings + base scene) plus the PERSISTENT per-effect graph state.
// baseCfg/baseFields/baseGravity/baseDragK are the tick-0 defaults every unbound parameter keeps; the static
// collision world (groundY/radius/restitution/spheres) completes the StepParticles argument set so
// StepAuthoredEffect is the ONE composition point. The graph state is SIM STATE (snapshotted by the lockstep).
struct AuthoredEffect {
    flow::Graph                                 graph;        // the authored node graph (built via edit-ops)
    flow::GraphState                            state;        // the persistent per-tick register file (SIM STATE)
    std::vector<ParamBinding>                   bindings;     // outputs -> parameters (ARRAY ORDER resolved)
    particles::EmitterConfig                    baseCfg{};    // defaults for every unbound emitter param
    std::vector<particles::ForceField>          baseFields;   // defaults for every unbound field param
    FxVec3                                      baseGravity{};// default gravity (kParamGravityY overrides .y)
    fx                                          baseDragK = 0;// default linear drag (kParamDragK overrides)
    fx                                          groundY = 0;  // static collision world (NOT drivable in v1)
    fx                                          radius = particles::kParticleRadius;
    fx                                          restitution = particles::kParticleRestitution;
    std::vector<particles::ParticleSphereCollider> spheres;   // static sphere colliders (NOT drivable in v1)
};

// ----- ResolvedParams: the per-tick parameter set after applying the bindings (a reporting value) ---------
// COPIES of the base config with the bound registers written over them — what StepParticles actually ran
// with this tick. Exposed (via StepAuthoredEffect's optional out) so tests/showcases can assert the pulse /
// oscillation / ramp structure without re-deriving the graph.
struct ResolvedParams {
    particles::EmitterConfig           cfg{};
    std::vector<particles::ForceField> fields;
    FxVec3                             gravity{};
    fx                                 dragK = 0;
};

// ----- ResolveBindings: base config + bound registers -> the per-tick ResolvedParams ----------------------
// Pure function of (effect's static asset, regs). Bindings applied in ARRAY ORDER (deterministic; a later
// binding to the same parameter wins — the documented conflict rule). An out-of-range node reads NOTHING
// (the binding is a deterministic no-op); an out-of-range fieldIndex likewise. Pure int32 reinterpretation
// per the unit contract (reg IS the Q16.16 value; kParamSpawnPerTick reg IS the raw count).
inline ResolvedParams ResolveBindings(const AuthoredEffect& e, const std::vector<flow::Reg>& regs) {
    ResolvedParams r;
    r.cfg     = e.baseCfg;
    r.fields  = e.baseFields;
    r.gravity = e.baseGravity;
    r.dragK   = e.baseDragK;
    for (const ParamBinding& b : e.bindings) {                     // ARRAY ORDER (the deterministic contract)
        if (static_cast<std::size_t>(b.node) >= regs.size()) continue;  // out-of-range node -> no-op
        const fx v = (fx)regs[static_cast<std::size_t>(b.node)];        // reg IS the value (int32 == fx)
        switch (b.param) {
            case kParamSpawnPerTick:  r.cfg.ratePerTick = v;  break;   // raw integer count (Emit convention)
            case kParamEmitterX:      r.cfg.origin.x    = v;  break;
            case kParamEmitterY:      r.cfg.origin.y    = v;  break;
            case kParamEmitterZ:      r.cfg.origin.z    = v;  break;
            case kParamLifetime:      r.cfg.lifetime    = v;  break;
            case kParamSpeed:         r.cfg.speed       = v;  break;
            case kParamFieldStrength:
                if ((std::size_t)b.fieldIndex < r.fields.size()) r.fields[(std::size_t)b.fieldIndex].strength = v;
                break;
            case kParamFieldX:
                if ((std::size_t)b.fieldIndex < r.fields.size()) r.fields[(std::size_t)b.fieldIndex].center.x = v;
                break;
            case kParamFieldY:
                if ((std::size_t)b.fieldIndex < r.fields.size()) r.fields[(std::size_t)b.fieldIndex].center.y = v;
                break;
            case kParamFieldZ:
                if ((std::size_t)b.fieldIndex < r.fields.size()) r.fields[(std::size_t)b.fieldIndex].center.z = v;
                break;
            case kParamGravityY:      r.gravity.y = v;  break;
            case kParamDragK:         r.dragK     = v;  break;
            default: break;                                            // unknown param kind -> deterministic no-op
        }
    }
    return r;
}

// ----- StepAuthoredEffect: ONE deterministic authored-effect tick (THE composition point) -----------------
// (1) StepGraph (mutates effect.state — the per-tick VM tick over the external inputs);
// (2) ResolveBindings (base copies + bound registers — the authoring seam);
// (3) particles::StepParticles with the resolved params (the bit-exact PT4 tick VERBATIM — emit -> forces+
//     integrate -> collide -> recycle -> ++tick).
// Returns StepParticles' contact count. `outResolved` (optional) receives the per-tick resolved params for
// structure assertions. ZERO bindings -> the resolved params ARE the base config -> bit-identical to calling
// StepParticles(base...) directly (the identity control). Pure integer, fixed order.
inline int StepAuthoredEffect(AuthoredEffect& e, particles::ParticlePool& pool, fx dt, uint32_t tick,
                              const std::vector<flow::Reg>& inputs, ResolvedParams* outResolved = nullptr) {
    const std::vector<flow::Reg> regs = flow::StepGraph(e.graph, e.state, inputs, tick);   // (1) the VM tick
    ResolvedParams r = ResolveBindings(e, regs);                                           // (2) the seam
    if (outResolved) *outResolved = r;
    return particles::StepParticles(pool, r.cfg,                                           // (3) the PT4 tick
                                    r.fields.empty() ? nullptr : r.fields.data(), (uint32_t)r.fields.size(),
                                    r.gravity, r.dragK, dt, e.groundY, e.radius, e.restitution,
                                    e.spheres.empty() ? nullptr : e.spheres.data(), (uint32_t)e.spheres.size());
}

// ===== PA1 LOCKSTEP + ROLLBACK (the PT5 mold over the COMPOSED graph+pool state) ==========================

// ----- AuthoredCommand: ONE per-tick input on the wire — write a flow INPUT channel -----------------------
// The deterministic input event a netcode layer would ship: on tick `tick`, input channel `channel` (a
// kInput node's constArg index) carries `value`. The per-tick input vector is rebuilt EVERY tick from the
// command stream (unset channels read 0 — a command is a one-tick impulse, matching kInput's semantics).
// The player-drives-the-effect story: the value flows through the AUTHORED graph into the bound parameters.
struct AuthoredCommand {
    uint32_t  tick    = 0;   // the tick this input applies on
    uint32_t  channel = 0;   // the kInput channel index (a node's constArg)
    flow::Reg value   = 0;   // the input value (int32; Q16.16 or raw per what the graph does with it)
};

// SimAuthoredTick: apply all commands whose .tick == pool.tick (ARRAY ORDER; later same-channel wins) into a
// fresh zeroed input vector of `inputChannels` entries, then StepAuthoredEffect. The commands gate on
// pool.tick BEFORE StepParticles' ++tick (the PT5 SimParticleTick convention). Pure integer, fixed order.
inline void SimAuthoredTick(AuthoredEffect& e, particles::ParticlePool& pool, fx dt,
                            const AuthoredCommand* cmds, uint32_t cmdCount, uint32_t inputChannels) {
    std::vector<flow::Reg> inputs((std::size_t)inputChannels, flow::Reg{0});
    for (uint32_t c = 0; c < cmdCount; ++c)
        if (cmds[c].tick == pool.tick && (std::size_t)cmds[c].channel < inputs.size())
            inputs[(std::size_t)cmds[c].channel] = cmds[c].value;
    StepAuthoredEffect(e, pool, dt, pool.tick, inputs);
}

// ----- AuthoredSnapshot: THE SNAPSHOT CRUX — the pool AND the flow graph state ----------------------------
// The particle snapshot (particles + freeList + spawnCursor + tick + cfg — the complete PT5 pool state) PLUS
// the flow GraphState (the persistent register file: counters, delays, latches — the authored effect's
// "where in the pulse cycle am I"). A snapshot that omits the graph state restores the pool but resumes the
// pulse/oscillation/ramp from the WRONG phase -> divergence (the completeness control proves it).
struct AuthoredSnapshot {
    particles::ParticleSnapshot pool;       // the full PT5 pool snapshot (deep copies)
    flow::GraphState            flowState;  // the graph's persistent register file (SIM STATE)
};

// SnapshotAuthored / RestoreAuthored: deep-copy capture/restore of the COMPOSED state. The emitter cfg
// travels inside the particle snapshot (SnapshotParticles' cfg slot) even though PA1 re-derives cfg from the
// graph each tick — keeping the PT5 shape so RestoreParticles round-trips bit-exact.
inline AuthoredSnapshot SnapshotAuthored(const AuthoredEffect& e, const particles::ParticlePool& pool) {
    AuthoredSnapshot s;
    s.pool      = particles::SnapshotParticles(pool, e.baseCfg);
    s.flowState = flow::SnapshotState(e.state);
    return s;
}
inline void RestoreAuthored(AuthoredEffect& e, particles::ParticlePool& pool, const AuthoredSnapshot& s) {
    particles::EmitterConfig scratch;                        // cfg is re-derived per tick; restore to scratch
    particles::RestoreParticles(pool, scratch, s.pool);
    flow::RestoreState(e.state, s.flowState);
}

// AuthoredStatesEqual: bit-exact compare of two COMPOSED states — the pool (ParticleStatesEqual: particles
// memcmp + freeList + spawnCursor + tick) AND the flow register file (exact vector compare). The lockstep/
// rollback proofs compare with THIS (both halves are sim state).
inline bool AuthoredStatesEqual(const particles::ParticlePool& pa, const flow::GraphState& sa,
                                const particles::ParticlePool& pb, const flow::GraphState& sb) {
    if (!particles::ParticleStatesEqual(pa, pb)) return false;
    if (sa.prev != sb.prev) return false;
    return true;
}

// RunAuthoredLockstep: THE peer entry point (the PT5 RunParticleLockstep twin over the composed state).
// Clone a peer from (asset, init): copy the STATIC authored asset (graph/bindings/base scene — the shipped
// artifact), restore the DYNAMIC state (pool + graph state) from init, run T SimAuthoredTicks applying the
// command stream. Two peers fed the SAME (asset, init, stream) — inputs ONLY, no state shared — re-derive
// BIT-IDENTICAL (pool AND graph state) by determinism. Returns the final AuthoredSnapshot.
inline AuthoredSnapshot RunAuthoredLockstep(const AuthoredEffect& asset, const AuthoredSnapshot& init,
                                            const AuthoredCommand* stream, uint32_t streamCount,
                                            uint32_t T, fx dt, uint32_t inputChannels) {
    AuthoredEffect e = asset;                                 // clone the static asset (graph/bindings/base)
    particles::ParticlePool pool;
    RestoreAuthored(e, pool, init);                           // clone the dynamic state (pool + graph state)
    for (uint32_t t = 0; t < T; ++t)
        SimAuthoredTick(e, pool, dt, stream, streamCount, inputChannels);
    return SnapshotAuthored(e, pool);
}

// RunAuthoredRollback: the rollback harness (the PT5 RunParticleRollback twin over the composed state).
// (1) advance 0..rollbackAt with the authoritative stream; (2) SAVE the composed snapshot; (2b) speculate
// <=3 ticks with the MISPREDICTED stream (the divergent client prediction); (3) restore + re-simulate
// rollbackAt..T with the CORRECT stream. The proof asserts the result == RunAuthoredLockstep(auth, T) AND
// that the mispredicted state DIFFERED (a real divergence was corrected).
inline AuthoredSnapshot RunAuthoredRollback(const AuthoredEffect& asset, const AuthoredSnapshot& init,
                                            const AuthoredCommand* authStream, uint32_t authCount,
                                            const AuthoredCommand* mispredictStream, uint32_t mispredictCount,
                                            uint32_t T, uint32_t rollbackAt, fx dt, uint32_t inputChannels) {
    AuthoredEffect e = asset;
    particles::ParticlePool pool;
    RestoreAuthored(e, pool, init);
    for (uint32_t t = 0; t < rollbackAt; ++t)                          // (1) authoritative prefix
        SimAuthoredTick(e, pool, dt, authStream, authCount, inputChannels);
    const AuthoredSnapshot snap = SnapshotAuthored(e, pool);           // (2) the restore point
    uint32_t specTicks = (T > rollbackAt) ? (T - rollbackAt) : 0u;     // (2b) bounded speculation (<=3)
    if (specTicks > 3u) specTicks = 3u;
    for (uint32_t s = 0; s < specTicks; ++s)
        SimAuthoredTick(e, pool, dt, mispredictStream, mispredictCount, inputChannels);
    RestoreAuthored(e, pool, snap);                                    // (3) ROLLBACK + correct re-sim
    for (uint32_t t = rollbackAt; t < T; ++t)
        SimAuthoredTick(e, pool, dt, authStream, authCount, inputChannels);
    return SnapshotAuthored(e, pool);
}

// ===== Digest currency (the pinned-golden values) ==========================================================

// DigestAuthoredGraph: FNV-1a-64 over flow::SerializeGraph's hand-LE byte encoding of the STATIC graph — the
// "authored, not hardcoded" pin: the test asserts the edit-ops-built showcase graph's digest equals a pinned
// value, so any drift in the authoring calls (or the edit-ops) is caught.
inline uint64_t DigestAuthoredGraph(const flow::Graph& g) {
    const std::vector<uint8_t> bytes = flow::SerializeGraph(g);
    return hf::net::DigestBytes(bytes.data(), bytes.size());
}

// DigestAuthored: FNV-1a-64 over a HAND little-endian serialization of the COMPOSED dynamic state — the
// particle slot array (48-byte no-padding int32 records — the memcmp currency, folded field-by-field LE),
// the free-list, spawnCursor, tick, then the flow register file. Hand-LE field by field (the replay.h /
// DigestEvents discipline: NEVER hash a padded host struct) -> byte-stable cross-platform (MSVC == clang).
inline uint64_t DigestAuthored(const particles::ParticlePool& pool, const flow::GraphState& st) {
    std::vector<uint8_t> buf;
    buf.reserve(pool.particles.size() * 48u + pool.freeList.size() * 4u + st.prev.size() * 4u + 16u);
    auto putU32 = [&](uint32_t v) {
        buf.push_back((uint8_t)( v        & 0xFFu));
        buf.push_back((uint8_t)((v >> 8)  & 0xFFu));
        buf.push_back((uint8_t)((v >> 16) & 0xFFu));
        buf.push_back((uint8_t)((v >> 24) & 0xFFu));
    };
    putU32((uint32_t)pool.particles.size());
    for (const particles::FxParticle& p : pool.particles) {
        putU32((uint32_t)p.pos.x); putU32((uint32_t)p.pos.y); putU32((uint32_t)p.pos.z);
        putU32((uint32_t)p.vel.x); putU32((uint32_t)p.vel.y); putU32((uint32_t)p.vel.z);
        putU32((uint32_t)p.age);   putU32((uint32_t)p.lifetime);
        putU32(p.seed);            putU32(p.flags);
        putU32((uint32_t)p.rsv0);  putU32((uint32_t)p.rsv1);
    }
    putU32((uint32_t)pool.freeList.size());
    for (const uint32_t f : pool.freeList) putU32(f);
    putU32(pool.spawnCursor);
    putU32(pool.tick);
    putU32((uint32_t)st.prev.size());
    for (const flow::Reg r : st.prev) putU32((uint32_t)r);
    return hf::net::DigestBytes(buf.data(), buf.size());
}

// ===== THE SHOWCASE FIXTURE — the PULSING FOUNTAIN, authored via the edit-ops (FIXED forever) =============
// A Niagara-class effect built EXCLUSIVELY with editor::AddFlowNode / editor::ConnectFlow (the authoring
// proof — the test pins DigestAuthoredGraph of the result):
//   * a PULSING spawn burst: a self-resetting cycle counter (Delay feedback + Select reset — period
//     kPulsePeriod) routes kSelect between the base rate and the burst rate: cycle==0 (every 6th tick,
//     ticks 5, 11, 17, ...) -> kBurstSpawn, else kBaseSpawn; PLUS a kInput channel-0 additive boost (the
//     player input: commands add extra spawns through the AUTHORED graph);
//   * a sine-ish OSCILLATING emitter X: a second cycle counter (period kOscPeriod) folded to a TRIANGLE wave
//     (Min(c, P-c)), centered and scaled by a Q16.16 constant -> emitter X sweeps -1.0 .. +1.0 world units;
//   * a RAMPING vortex strength: a kCounter accumulating kRampStep Q16.16 per tick drives field 0's strength
//     (the effect spins up over time — late-tick tangential velocities exceed early ones).
// All integer, all deterministic, all driven through ParamBindings — NOT hand-written EmitterConfig code.

inline constexpr int      kPulsePeriod = 6;      // burst every 6 ticks (cycle==0 on ticks 5, 11, 17, ...)
inline constexpr int      kBaseSpawn   = 1;      // base spawn count per tick (a trickle)
inline constexpr int      kBurstSpawn  = 12;     // burst spawn count on the pulse tick (the puff)
inline constexpr int      kOscPeriod   = 16;     // emitter-X triangle-wave period (ticks)
inline constexpr flow::Reg kOscScale   = 16384;  // 0.25 in Q16.16: (tri-4) in [-4,4] * 0.25 -> [-1.0, +1.0]
inline constexpr flow::Reg kRampStep   = 2048;   // vortex-strength ramp: +1/32 (Q16.16) per tick

// The flow NodeIds the bindings reference (returned by MakePulsingFountainGraph's AddFlowNode calls, pinned
// by construction order — asserted against the pinned graph digest).
struct PulsingFountainNodes {
    flow::NodeId spawnOut = 0;   // -> kParamSpawnPerTick (raw count)
    flow::NodeId emitXOut = 0;   // -> kParamEmitterX (Q16.16)
    flow::NodeId rampOut  = 0;   // -> kParamFieldStrength field 0 (Q16.16)
};

// MakePulsingFountainGraph: build the showcase graph VIA THE EDIT-OPS (AddFlowNode/ConnectFlow ONLY — the
// authoring story; no Node{} literals). Returns the graph + the three output NodeIds the bindings use.
inline flow::Graph MakePulsingFountainGraph(PulsingFountainNodes& out) {
    using namespace hf::editor;   // AddFlowNode / ConnectFlow — the flow-editor authoring ops
    flow::Graph g;

    // --- shared constants -------------------------------------------------------------------------------
    const flow::NodeId nOne  = AddFlowNode(g, flow::kConst, 1);              // 1
    const flow::NodeId nZero = AddFlowNode(g, flow::kConst, 0);              // 0

    // --- the PULSE cycle counter (period kPulsePeriod): cycle = (prev+1 == N) ? 0 : prev+1 ---------------
    // Delay feedback (Delay's `a` is a STATE read, not a topo edge -> the loop is legal), Select reset on
    // the exact wrap tick via (inc - N) as the != 0 predicate.
    const flow::NodeId nPulseN    = AddFlowNode(g, flow::kConst, kPulsePeriod);   // N = 6
    const flow::NodeId nPulsePrev = AddFlowNode(g, flow::kDelay);                 // prev cycle (a wired below)
    const flow::NodeId nPulseInc  = AddFlowNode(g, flow::kAdd);                   // inc = prev + 1
    ConnectFlow(g, nPulsePrev, nPulseInc, 0);   // a = prev
    ConnectFlow(g, nOne,       nPulseInc, 1);   // b = 1
    const flow::NodeId nPulseDiff = AddFlowNode(g, flow::kSub);                   // diff = inc - N (0 on wrap)
    ConnectFlow(g, nPulseInc, nPulseDiff, 0);
    ConnectFlow(g, nPulseN,   nPulseDiff, 1);
    const flow::NodeId nPulseCyc  = AddFlowNode(g, flow::kSelect);                // cycle = diff!=0 ? inc : 0
    ConnectFlow(g, nPulseInc,  nPulseCyc, 0);   // a = inc
    ConnectFlow(g, nZero,      nPulseCyc, 1);   // b = 0
    ConnectFlow(g, nPulseDiff, nPulseCyc, 2);   // c = diff (predicate)
    ConnectFlow(g, nPulseCyc,  nPulsePrev, 0);  // the Delay reads cycle's PREVIOUS tick (state read, no cycle)

    // --- the spawn select + the player-input boost -------------------------------------------------------
    const flow::NodeId nBase   = AddFlowNode(g, flow::kConst, kBaseSpawn);        // 1/tick trickle
    const flow::NodeId nBurst  = AddFlowNode(g, flow::kConst, kBurstSpawn);       // 12 on the pulse tick
    const flow::NodeId nSpawnSel = AddFlowNode(g, flow::kSelect);                 // cycle!=0 ? base : BURST
    ConnectFlow(g, nBase,     nSpawnSel, 0);    // a = base   (cycle != 0 — the off-beat ticks)
    ConnectFlow(g, nBurst,    nSpawnSel, 1);    // b = burst  (cycle == 0 — the pulse tick)
    ConnectFlow(g, nPulseCyc, nSpawnSel, 2);    // c = cycle
    const flow::NodeId nBoost  = AddFlowNode(g, flow::kInput, 0);                 // input channel 0 (the player)
    const flow::NodeId nSpawn  = AddFlowNode(g, flow::kAdd);                      // spawn = select + boost
    ConnectFlow(g, nSpawnSel, nSpawn, 0);
    ConnectFlow(g, nBoost,    nSpawn, 1);

    // --- the OSC cycle counter (period kOscPeriod) -> triangle -> centered -> Q16.16 scale ---------------
    const flow::NodeId nOscP    = AddFlowNode(g, flow::kConst, kOscPeriod);       // P = 16
    const flow::NodeId nOscPrev = AddFlowNode(g, flow::kDelay);                   // prev osc (a wired below)
    const flow::NodeId nOscInc  = AddFlowNode(g, flow::kAdd);                     // oinc = prev + 1
    ConnectFlow(g, nOscPrev, nOscInc, 0);
    ConnectFlow(g, nOne,     nOscInc, 1);
    const flow::NodeId nOscDiff = AddFlowNode(g, flow::kSub);                     // odiff = oinc - P
    ConnectFlow(g, nOscInc, nOscDiff, 0);
    ConnectFlow(g, nOscP,   nOscDiff, 1);
    const flow::NodeId nOscCyc  = AddFlowNode(g, flow::kSelect);                  // ocycle = odiff!=0 ? oinc : 0
    ConnectFlow(g, nOscInc,  nOscCyc, 0);
    ConnectFlow(g, nZero,    nOscCyc, 1);
    ConnectFlow(g, nOscDiff, nOscCyc, 2);
    ConnectFlow(g, nOscCyc,  nOscPrev, 0);      // the Delay feedback
    const flow::NodeId nOscInv  = AddFlowNode(g, flow::kSub);                     // P - ocycle
    ConnectFlow(g, nOscP,   nOscInv, 0);
    ConnectFlow(g, nOscCyc, nOscInv, 1);
    const flow::NodeId nTri     = AddFlowNode(g, flow::kMin);                     // tri = Min(o, P-o) in 0..8
    ConnectFlow(g, nOscCyc, nTri, 0);
    ConnectFlow(g, nOscInv, nTri, 1);
    const flow::NodeId nHalf    = AddFlowNode(g, flow::kConst, kOscPeriod / 4);   // 4 (the half-amplitude)
    const flow::NodeId nCent    = AddFlowNode(g, flow::kSub);                     // centered = tri - 4 in [-4,4]
    ConnectFlow(g, nTri,  nCent, 0);
    ConnectFlow(g, nHalf, nCent, 1);
    const flow::NodeId nScale   = AddFlowNode(g, flow::kConst, kOscScale);        // 0.25 Q16.16
    const flow::NodeId nEmitX   = AddFlowNode(g, flow::kMul);                     // emitterX = centered * scale
    ConnectFlow(g, nCent,  nEmitX, 0);
    ConnectFlow(g, nScale, nEmitX, 1);

    // --- the RAMP: vortex strength accumulates kRampStep per tick ----------------------------------------
    const flow::NodeId nRamp = AddFlowNode(g, flow::kCounter, kRampStep);         // prev[self] + step each tick

    out.spawnOut = nSpawn;
    out.emitXOut = nEmitX;
    out.rampOut  = nRamp;
    return g;
}

// MakePulsingFountainEffect: the COMPLETE showcase asset — the edit-ops-built graph, the three bindings, and
// the base scene (a fountain over a ground plane with ONE vortex field whose strength the ramp drives).
// FIXED forever (the test pins the graph digest, the per-tick alive-count trace digest, and the final
// composed-state digest; the --pa1-fountain showcases run THIS EXACT asset on both backends).
inline AuthoredEffect MakePulsingFountainEffect() {
    AuthoredEffect e;
    PulsingFountainNodes nodes;
    e.graph = MakePulsingFountainGraph(nodes);
    e.state = flow::MakeState(e.graph);

    e.bindings = {
        ParamBinding{nodes.spawnOut, kParamSpawnPerTick,  0u},   // the pulse (+ player boost) -> spawn count
        ParamBinding{nodes.emitXOut, kParamEmitterX,      0u},   // the triangle wave -> emitter X (Q16.16)
        ParamBinding{nodes.rampOut,  kParamFieldStrength, 0u},   // the ramp -> vortex strength (Q16.16)
    };

    e.baseCfg.origin      = FxVec3{0, kOne * 3, 0};   // the fountain mouth, 3 units up (X driven per tick)
    e.baseCfg.ratePerTick = (fx)kBaseSpawn;           // overridden every tick by the spawn binding
    e.baseCfg.lifetime    = kOne;                     // 1.0 s (60 ticks at dt=1/60) — a steady churn
    e.baseCfg.speed       = kOne * 2;                 // initial speed 2.0
    e.baseCfg.emitterId   = 1u;

    e.baseFields.resize(1);                           // ONE vortex, strength ramped by the binding
    e.baseFields[0].kind     = particles::kFieldVortex;
    e.baseFields[0].center   = FxVec3{0, kOne, 0};
    e.baseFields[0].axis     = FxVec3{0, kOne, 0};
    e.baseFields[0].strength = 0;                     // tick-0 default; the ramp binding overrides per tick
    e.baseFields[0].radius   = kOne * 5;

    // gravity -9.8 host-snapped to Q16.16 (the PT scene rounding), drag 1/50, ground at -2.
    e.baseGravity = FxVec3{0, (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5)), 0};
    e.baseDragK   = kOne / 50;
    e.groundY     = -kOne * 2;
    e.radius      = particles::kParticleRadius;
    e.restitution = particles::kParticleRestitution;
    // no sphere colliders (the pulse/oscillation/ramp is the story; the ground catches the spray)
    return e;
}

// The showcase run constants (FIXED — the pinned digests depend on them).
inline constexpr uint32_t kShowcaseCapacity = 320;   // pool capacity (steady alive ~180 -> headroom for bursts)
inline constexpr uint32_t kShowcaseSteps    = 128;   // ticks: >2 lifetimes of churn; last burst tick 125 is 3
                                                     // ticks old at capture (the puff visibly mid-flight)
inline constexpr uint32_t kShowcaseChannels = 1;     // ONE input channel (the spawn boost)

}  // namespace pauthor
}  // namespace hf::sim
