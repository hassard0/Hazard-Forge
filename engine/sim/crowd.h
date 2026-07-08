#pragma once
// Slice CR1 — DETERMINISTIC CROWD SIMULATION AT 10k+ AGENTS (the Mass-class archetype crowd; the parity++
// audit's CROWD-SCALE gap). boids.h (FLAGSHIP #18) ships a deterministic flocking sim, but its golden proves
// only ~256 agents and BD1's SteerSeparation is BRUTE-FORCE all-pairs O(N²) — it does NOT scale. UE5's Mass
// framework runs 10k-100k agents; CR1 proves Hazard Forge's crowd at 10,000+ agents with O(N) GRID neighbor
// queries (the BD2 spatial-hash reused VERBATIM), deterministic + lockstep-replayable (UE5 Mass is float /
// non-deterministic — it cannot replay a 10k-agent crowd bit-for-bit; CR1 can). PURE CPU strict-integer
// (Q16.16), NO new shader — DELIBERATELY pure to avoid render-path risk while proving the SCALE headline.
//
// THE Mass "ARCHETYPE" IDEA: behaviour is parameterized by a small fixed set of TYPES (pedestrian / runner /
// wanderer), not a per-agent branch. Each agent carries an archetype INDEX + a GOAL index; its maxSpeed /
// separation radius / goal weight / body radius come from CrowdArchetype[archetype]. This is exactly the
// data-oriented "archetype" of an ECS crowd: the fast inner loop reads a tiny shared parameter block by type.
//
// THE O(N) NEIGHBOUR CRUX (the whole point — vs BD1's O(N²)): separation queries ONLY the agent's 3x3(x3)
// neighbour cells via the BD2 uniform spatial-hash grid (boids.h::MakeBoidsGrid / BuildBoidsCellTable /
// BuildBoidsNeighborList, COMPOSED READ-ONLY — boids.h is byte-UNTOUCHED). The grid CELL SIZE == the MAX
// separation radius over all archetypes, so an agent's stencil (its cell + the 26 neighbours) covers every
// cell that can hold a within-radius neighbour (|dx| < maxSepRadius <= cellSize => at most one cell away per
// axis — the BD2 invariant). Each agent's separation then filters that candidate list by ITS OWN archetype
// separation radius (a per-axis box test). Grid cost is O(N) at fixed density; the neighbour list is rebuilt
// each tick over the FROZEN positions (the boids Jacobi discipline).
//
// THE CROSS-BACKEND / DETERMINISM CONTRACT (the boids Jacobi mold): every agent computes its steering force
// from a FROZEN snapshot of ALL agents, THEN integrates — so the per-agent update is INDEPENDENT of order
// (two runs byte-identical; a GPU one-thread-per-agent port would be race-free BY CONSTRUCTION). Separation
// is a SUM of raw integer away-deltas Σ(pos_i - pos_j) over the neighbour set; integer addition is
// associative + commutative (even under two's-complement wrap), so the GRID separation sum == the ALL-PAIRS
// separation sum BIT-FOR-BIT regardless of iteration order — that is CR1's KEY correctness pin (the O(N) grid
// gives the SAME answer as the O(N²) brute force). All Q16.16, fixed accumulation order, NO transcendentals
// (un-normalized seek + per-axis box clamps, the BD1 discipline — no FxNormalize/FxISqrt on the hot path).
//
// REUSE MAP (read-only, byte-frozen): boids.h — Agent / FxVec3 vocabulary, ClampAxisVec, MakeBoidsGrid /
// BuildBoidsCellTable / BuildBoidsNeighborList / BoidsNeighborAccept (the BD2 grid engine), the BD5 lockstep
// mold (RunBoidsLockstep/RunBoidsRollback shape). fpx.h (via boids.h) — fx/kOne/fxmul/FxVec3/FxAdd/FxSub/
// FxScale. NOTHING in boids.h or fpx.h is modified.
//
// HONEST CAVEATS (the GR4/BD1 shape): agents are POINTS with a soft separation PUSH (not a hard
// non-penetration contact — dense clusters at a goal can overlap; a Jacobi single-projection residual, real
// behaviour, pinned not papered over). Un-normalized goal-seek carries MOMENTUM, so a lone agent's approach
// is monotone while far but can plateau/oscillate within a step of the goal (pinned as measured). The
// headline is DETERMINISM + SCALE + lockstep-replayability, not perfect flow.

#include <cstdint>
#include <cstring>     // std::memcmp (lockstep/rollback bit-exact compares) — hoisted to file scope
#include <vector>

#include "sim/boids.h"   // read-only: Agent, ClampAxisVec, the BD2 grid engine, the fpx Q16.16 vocabulary

namespace hf::sim {
namespace crowd {

// Re-export the boids/fpx Q16.16 vocabulary (the crowd shares the fixed-point format + agent shape exactly).
using fx       = boids::fx;
using FxVec3   = boids::FxVec3;
using Agent    = boids::Agent;          // a point with a Q16.16 pos + vel (the SAME struct boids/grid consume)
inline constexpr int kFrac = boids::kFrac;
inline constexpr fx  kOne  = boids::kOne;
using boids::fxmul;
using boids::ClampAxisVec;              // the per-axis box clamp (no radial sqrt)
using fpx::FxAdd;                       // re-export the Q16.16 vector ops (so callers use crowd::FxAdd etc.)
using fpx::FxSub;
using fpx::FxScale;

// ----- CrowdArchetype: the per-TYPE behaviour parameters (the Mass "archetype") --------------------------
// A small fixed set of these (e.g. 3: pedestrian / runner / wanderer) is shared by index. maxSpeed caps the
// per-axis speed; sepRadius is the neighbour box half-width for THIS type's separation; goalWeight scales the
// (goal - pos) seek pull; radius is the agent's body/render radius (viz + a spacing hint). All Q16.16.
struct CrowdArchetype {
    fx maxSpeed  = 0;   // Q16.16 per-AXIS speed clamp (the type's top speed)
    fx sepRadius = 0;   // Q16.16 separation neighbour box half-width (this type's personal space)
    fx goalWeight= 0;   // Q16.16 gain on the (goal - pos) un-normalized seek pull
    fx radius    = 0;   // Q16.16 body/render radius (viz + spacing hint; not a hard collider)
};

// ----- CrowdConfig: the host-fixed sim tuning (golden-stable; NO RNG) ------------------------------------
// A fixed archetype table + a fixed set of GOAL points (agents seek goals[goal[i]]) + the global separation
// gain + the global per-axis force clamp. cellSize (the grid cell) == the MAX archetype sepRadius (computed
// by MaxSepRadius / MakeCrowdConfig) so the 27-cell stencil covers every within-radius neighbour.
struct CrowdConfig {
    std::vector<CrowdArchetype> archetypes;   // the type table (indexed by Crowd::archetype[i])
    std::vector<FxVec3>         goals;         // the goal points (indexed by Crowd::goal[i])
    fx                          sepGain  = 0;  // Q16.16 gain on the accumulated separation away-sum
    fx                          maxForce = 0;  // Q16.16 per-AXIS force clamp (the axis-box magnitude limit)
    fx                          cellSize = 0;  // Q16.16 grid cell (== max archetype sepRadius); set by helper
};

// MaxSepRadius(cfg): the maximum separation radius over all archetypes (the grid cell size). >= any agent's
// per-archetype sepRadius, so a stencil built at this cell size is a SUPERSET of every agent's candidates.
inline fx MaxSepRadius(const CrowdConfig& cfg) {
    fx m = 0;
    for (const CrowdArchetype& a : cfg.archetypes) if (a.sepRadius > m) m = a.sepRadius;
    if (m <= 0) m = kOne;   // degenerate (no archetypes / zero radii) -> a 1-unit cell (deterministic)
    return m;
}

// FinalizeCrowdConfig(cfg): set cfg.cellSize = MaxSepRadius(cfg). Call once after populating archetypes.
inline void FinalizeCrowdConfig(CrowdConfig& cfg) { cfg.cellSize = MaxSepRadius(cfg); }

// ----- Crowd: the parallel agent arrays (pos/vel in a boids::Agent vector so the BD2 grid consumes it) ----
// agents[i] carries pos+vel (the exact struct MakeBoidsGrid/BuildBoidsCellTable/BuildBoidsNeighborList read);
// archetype[i] is the agent's TYPE index (into cfg.archetypes); goal[i] is its GOAL index (into cfg.goals).
// The three arrays are index-parallel (same length). Keeping pos/vel as boids::Agent is what lets CR1 COMPOSE
// the byte-frozen BD2 grid engine with ZERO copy of boids.h.
struct Crowd {
    std::vector<Agent>    agents;      // pos + vel (the BD2 grid input)
    std::vector<uint8_t>  archetype;   // per-agent archetype index (into cfg.archetypes)
    std::vector<uint32_t> goal;        // per-agent goal index (into cfg.goals)
};

// CrowdSize(c): the agent count (agents.size()).
inline uint32_t CrowdSize(const Crowd& c) { return (uint32_t)c.agents.size(); }

// ----- CrowdBoxAccept(a, b, radius): the per-axis box neighbour test (== boids::BoidsNeighborAccept) --------
// Accept b as a separation neighbour of a iff |a.axis - b.axis| < radius on EVERY axis. PURE INT32 (subtract +
// abs + compare per axis; NO products, NO int64, NO sqrt) — the SAME predicate BuildBoidsNeighborList uses, so
// a stencil built at cell==maxSepRadius and re-filtered here at a per-archetype radius <= maxSepRadius yields
// EXACTLY the all-pairs accepted set. (Delegates to the byte-frozen boids predicate.)
inline bool CrowdBoxAccept(const FxVec3& a, const FxVec3& b, fx radius) {
    return boids::BoidsNeighborAccept(a, b, radius);
}

// ----- SeparationAllPairs(i, agents, archIdx, cfg): the O(N²) brute-force separation (the REFERENCE) --------
// Σ (pos_i - pos_j) over EVERY other agent j within agent i's archetype box radius, scaled by sepGain. This is
// the BD1 SteerSeparation shape (all-pairs), used ONLY as the correctness reference the O(N) grid path must
// match bit-for-bit. Fixed ascending-j order; raw integer away-deltas; integer add is associative/commutative
// so the ORDER vs the grid's stencil order does NOT change the sum.
inline FxVec3 SeparationAllPairs(int i, const std::vector<Agent>& agents,
                                 const std::vector<uint8_t>& archIdx, const CrowdConfig& cfg) {
    const Agent& a = agents[(size_t)i];
    const fx r = cfg.archetypes[archIdx[(size_t)i]].sepRadius;
    FxVec3 sum{0, 0, 0};
    const int n = (int)agents.size();
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        const Agent& o = agents[(size_t)j];
        if (CrowdBoxAccept(a.pos, o.pos, r)) {
            sum.x += a.pos.x - o.pos.x;
            sum.y += a.pos.y - o.pos.y;
            sum.z += a.pos.z - o.pos.z;
        }
    }
    return FxScale(sum, cfg.sepGain);
}

// ----- SeparationGrid(i, agents, list, archIdx, cfg): the O(N) grid separation (the PRODUCTION path) --------
// Σ (pos_i - pos_j) over the BD2 neighbour slice list.neighbors[neighborStart[i]..[i+1]) — the candidates in
// agent i's 27-cell stencil (built at cell==maxSepRadius) — re-filtered by i's OWN archetype box radius, scaled
// by sepGain. Because maxSepRadius >= the per-archetype radius, the stencil is a SUPERSET of every accepted
// neighbour, so the re-filtered set == the all-pairs accepted set -> SeparationGrid == SeparationAllPairs
// BIT-FOR-BIT (the correctness pin). The neighbour slice is in the BD2 fixed stencil order; the sum is
// order-independent (integer add). NO int64, NO sqrt.
inline FxVec3 SeparationGrid(int i, const std::vector<Agent>& agents, const boids::BoidsNeighborList& list,
                             const std::vector<uint8_t>& archIdx, const CrowdConfig& cfg) {
    const Agent& a = agents[(size_t)i];
    const fx r = cfg.archetypes[archIdx[(size_t)i]].sepRadius;
    FxVec3 sum{0, 0, 0};
    const uint32_t s0 = list.neighborStart[(size_t)i];
    const uint32_t s1 = list.neighborStart[(size_t)i + 1u];
    for (uint32_t s = s0; s < s1; ++s) {
        const Agent& o = agents[(size_t)list.neighbors[(size_t)s]];
        if (CrowdBoxAccept(a.pos, o.pos, r)) {
            sum.x += a.pos.x - o.pos.x;
            sum.y += a.pos.y - o.pos.y;
            sum.z += a.pos.z - o.pos.z;
        }
    }
    return FxScale(sum, cfg.sepGain);
}

// ----- GoalSeek(a, arch, cfg, goalIdx): the per-archetype un-normalized goal pull (the BD1 SteerSeek shape) --
// desired = goals[goalIdx] - pos; force = FxScale(desired, arch.goalWeight). Un-normalized "arrive" (pulls
// harder when far, eases in when close — integer-cheap, NO FxNormalize). Out-of-range goalIdx -> zero (the
// deterministic guard).
inline FxVec3 GoalSeek(const Agent& a, const CrowdArchetype& arch, const CrowdConfig& cfg, uint32_t goalIdx) {
    if (goalIdx >= (uint32_t)cfg.goals.size()) return FxVec3{0, 0, 0};
    return FxScale(FxSub(cfg.goals[(size_t)goalIdx], a.pos), arch.goalWeight);
}

// ----- StepCrowd(c, cfg, dt): one deterministic O(N) crowd tick (JACOBI, grid separation) ----------------
// (1) FREEZE the current positions (the Jacobi input every agent's force reads). (2) Rebuild the BD2 grid +
//     cell table + neighbour list on the frozen positions at cell==cfg.cellSize (== maxSepRadius) — O(N).
// (3) Per agent in FIXED index order: force = grid-separation + per-archetype goal-seek; per-axis clamp to
//     ±maxForce; vel += force*dt clamp ±(archetype maxSpeed); pos += vel*dt. Fixed op order + frozen snapshot
//     -> two runs bit-identical AND order-independent (a GPU one-thread-per-agent port would be race-free).
inline void StepCrowd(Crowd& c, const CrowdConfig& cfg, fx dt) {
    const std::vector<Agent> prev = c.agents;   // frozen snapshot (the Jacobi input)
    const fx cell = cfg.cellSize > 0 ? cfg.cellSize : MaxSepRadius(cfg);
    // Rebuild the BD2 grid engine on the frozen positions (byte-frozen boids.h functions, composed read-only).
    const boids::BoidsGrid grid = boids::MakeBoidsGrid(prev, cell);
    const boids::BoidsCellTable table = boids::BuildBoidsCellTable(prev, grid);
    const boids::BoidsNeighborList list = boids::BuildBoidsNeighborList(prev, grid, table, cell);
    const int n = (int)c.agents.size();
    for (int i = 0; i < n; ++i) {
        const Agent& a = prev[(size_t)i];                              // read the FROZEN state
        const CrowdArchetype& arch = cfg.archetypes[c.archetype[(size_t)i]];
        // (a) O(N) grid separation (bit-exact == the all-pairs reference).
        FxVec3 force = SeparationGrid(i, prev, list, c.archetype, cfg);
        // (b) per-archetype goal-seek toward this agent's goal point.
        force = FxAdd(force, GoalSeek(a, arch, cfg, c.goal[(size_t)i]));
        // (c) per-axis clamp the force (the axis-box magnitude limit).
        force = ClampAxisVec(force, cfg.maxForce);
        // (d) integrate velocity: vel += force*dt; clamp to the archetype's per-axis maxSpeed.
        FxVec3 vel = FxAdd(a.vel, FxScale(force, dt));
        vel = ClampAxisVec(vel, arch.maxSpeed);
        // (e) integrate position: pos += vel*dt.
        const FxVec3 pos = FxAdd(a.pos, FxScale(vel, dt));
        c.agents[(size_t)i].vel = vel;
        c.agents[(size_t)i].pos = pos;
    }
}

// ----- StepCrowdAllPairs(c, cfg, dt): the O(N²) REFERENCE tick (brute-force separation) ------------------
// Byte-identical to StepCrowd EXCEPT separation is SeparationAllPairs (no grid). Used ONLY by the correctness
// pin: StepCrowd(c) == StepCrowdAllPairs(c) for the SAME input proves the O(N) grid gives the O(N²) answer.
inline void StepCrowdAllPairs(Crowd& c, const CrowdConfig& cfg, fx dt) {
    const std::vector<Agent> prev = c.agents;
    const int n = (int)c.agents.size();
    for (int i = 0; i < n; ++i) {
        const Agent& a = prev[(size_t)i];
        const CrowdArchetype& arch = cfg.archetypes[c.archetype[(size_t)i]];
        FxVec3 force = SeparationAllPairs(i, prev, c.archetype, cfg);
        force = FxAdd(force, GoalSeek(a, arch, cfg, c.goal[(size_t)i]));
        force = ClampAxisVec(force, cfg.maxForce);
        FxVec3 vel = FxAdd(a.vel, FxScale(force, dt));
        vel = ClampAxisVec(vel, arch.maxSpeed);
        const FxVec3 pos = FxAdd(a.pos, FxScale(vel, dt));
        c.agents[(size_t)i].vel = vel;
        c.agents[(size_t)i].pos = pos;
    }
}

// StepCrowdSteps(c, cfg, dt, steps): run `steps` StepCrowd ticks (the scale/showcase settle loop).
inline void StepCrowdSteps(Crowd& c, const CrowdConfig& cfg, fx dt, int steps) {
    for (int s = 0; s < steps; ++s) StepCrowd(c, cfg, dt);
}

// ----- DigestCrowd(c): the deterministic full-state FNV-1a digest (pos.xyz, vel.xyz, archetype, goal) -----
// A 64-bit FNV-1a over EVERY agent's complete replayable state, in agent-index order. Pure integer -> the
// digest is bit-identical on every compiler/vendor (MSVC == clang). The scale-proof + lockstep currency.
inline uint64_t Fnv1aWord(uint64_t h, uint32_t w) {
    h ^= (uint64_t)w;
    h *= 1099511628211ull;
    return h;
}
inline uint64_t DigestCrowd(const Crowd& c) {
    uint64_t h = 14695981039346656037ull;
    const uint32_t n = CrowdSize(c);
    h = Fnv1aWord(h, n);
    for (uint32_t i = 0; i < n; ++i) {
        const Agent& a = c.agents[(size_t)i];
        h = Fnv1aWord(h, (uint32_t)a.pos.x); h = Fnv1aWord(h, (uint32_t)a.pos.y); h = Fnv1aWord(h, (uint32_t)a.pos.z);
        h = Fnv1aWord(h, (uint32_t)a.vel.x); h = Fnv1aWord(h, (uint32_t)a.vel.y); h = Fnv1aWord(h, (uint32_t)a.vel.z);
        h = Fnv1aWord(h, (uint32_t)c.archetype[(size_t)i]);
        h = Fnv1aWord(h, c.goal[(size_t)i]);
    }
    return h;
}

// ----- CrowdStats: the deterministic integer crowd statistics (Q16.16 / integer L1) ----------------------
// meanToGoal = mean L1 distance |pos - goals[goal[i]]| (the "they sought their goals" stat — DROPS as the
//   crowd flows to its goals). minSep = the minimum L1 pairwise separation (the "they didn't all collapse"
//   stat — O(N²), so ONLY meaningful/used for the SMALL correctness/goal tests, not the 10k scale run).
//   meanSpeed = mean L1 speed. All integer L1 (|dx|+|dy|+|dz|, NO sqrt) -> bit-exact + cross-platform.
struct CrowdStats {
    fx meanToGoal = 0;   // mean L1 distance to each agent's goal
    fx meanSpeed  = 0;   // mean L1 speed
    fx minSep     = 0;   // minimum L1 pairwise separation (O(N²) — small-N only)
};

inline fx CrowdAbs(fx v) { return v < 0 ? -v : v; }
inline fx CrowdL1(const FxVec3& v) { return CrowdAbs(v.x) + CrowdAbs(v.y) + CrowdAbs(v.z); }

// MeasureCrowd(c, cfg, withMinSep): the crowd stats. withMinSep gates the O(N²) min-separation scan (pass
// false for the 10k scale run — the pairwise scan is O(N²) and only wanted for small correctness/goal tests).
inline CrowdStats MeasureCrowd(const Crowd& c, const CrowdConfig& cfg, bool withMinSep) {
    CrowdStats s;
    const int n = (int)c.agents.size();
    if (n == 0) return s;
    int64_t goalSum = 0, speedSum = 0;
    for (int i = 0; i < n; ++i) {
        speedSum += CrowdL1(c.agents[(size_t)i].vel);
        const uint32_t g = c.goal[(size_t)i];
        if (g < (uint32_t)cfg.goals.size())
            goalSum += CrowdL1(FxSub(c.agents[(size_t)i].pos, cfg.goals[(size_t)g]));
    }
    s.meanToGoal = (fx)(goalSum / n);
    s.meanSpeed  = (fx)(speedSum / n);
    if (withMinSep) {
        int64_t minSep = -1;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                const int64_t d = CrowdL1(FxSub(c.agents[(size_t)i].pos, c.agents[(size_t)j].pos));
                if (minSep < 0 || d < minSep) minSep = d;
            }
        s.minSep = (fx)(minSep < 0 ? 0 : minSep);
    }
    return s;
}

// ===== LOCKSTEP + ROLLBACK (the netcode headline: a 10k-agent crowd two peers replay bit-for-bit) =========
// The BD5 mold: commands = GOAL CHANGES (redirect an agent to a new goal — the "the crowd is re-routed"
// input); the snapshot is the FULL mutable crowd (agents pos+vel, archetype, goal — a deep copy). Two peers
// fed ONLY the command stream re-derive the exact crowd trajectory bit-for-bit; a MISPREDICTED goal-change is
// corrected by rolling back to a snapshot + re-simulating with the authoritative stream. PURE CPU.

// ----- CrowdCommand: ONE goal-redirect input event (the "re-route this agent" perturbation) ---------------
struct CrowdCommand {
    uint32_t tick    = 0;   // the tick this redirect fires at
    uint32_t agent   = 0;   // the agent index whose goal changes
    uint32_t newGoal = 0;   // the new goal index (into cfg.goals)
};

// ApplyCrowdCommand(c, cmd): set agent cmd.agent's goal to cmd.newGoal (in-range guard).
inline void ApplyCrowdCommand(Crowd& c, const CrowdCommand& cmd) {
    if (cmd.agent < CrowdSize(c)) c.goal[(size_t)cmd.agent] = cmd.newGoal;
}

// ----- CrowdSnapshot: the captured mutable crowd state (a deep copy of ALL replayable state) --------------
// The full Crowd: agents (pos+vel), archetype, goal. archetype is immutable during a run but is INCLUDED so
// the snapshot is COMPLETE — a peer restoring it re-derives the trajectory exactly (the completeness contract:
// dropping velocity OR archetype from the restore diverges, proven by crowd_test).
struct CrowdSnapshot { Crowd crowd; };
inline CrowdSnapshot SnapshotCrowd(const Crowd& c) { return CrowdSnapshot{c}; }
inline void RestoreCrowd(Crowd& c, const CrowdSnapshot& snap) { c = snap.crowd; }

// ----- SimCrowdTick(c, cfg, commands, tick, dt): apply this tick's redirects, then StepCrowd -------------
inline void SimCrowdTick(Crowd& c, const CrowdConfig& cfg, const std::vector<CrowdCommand>& commands,
                         int tick, fx dt) {
    for (size_t k = 0; k < commands.size(); ++k)
        if ((int)commands[k].tick == tick) ApplyCrowdCommand(c, commands[k]);
    StepCrowd(c, cfg, dt);
}

// ----- RunCrowdLockstep: authority + replica from the SAME inputs, bit-identical every tick ---------------
// Run `ticks` SimCrowdTicks from a COPY of `initial`, applying the command stream. authority + replica step
// from the SAME init + stream -> BIT-IDENTICAL by determinism; asserts authority == replica every tick
// (memcmp the agent vector + goal/archetype). Returns the converged authority crowd.
inline Crowd RunCrowdLockstep(const CrowdConfig& cfg, const Crowd& initial,
                              const std::vector<CrowdCommand>& commands, int ticks, fx dt) {
    Crowd authority = initial;
    Crowd replica   = initial;
    for (int t = 0; t < ticks; ++t) {
        SimCrowdTick(authority, cfg, commands, t, dt);
        SimCrowdTick(replica,   cfg, commands, t, dt);
        const bool same = authority.agents.size() == replica.agents.size() &&
            std::memcmp(authority.agents.data(), replica.agents.data(),
                        authority.agents.size() * sizeof(Agent)) == 0 &&
            authority.goal == replica.goal && authority.archetype == replica.archetype;
        if (!same) return authority;   // the lockstep invariant broke (unreachable for a deterministic sim)
    }
    return authority;
}

// ----- RunCrowdRollback: snapshot -> mispredict diverges -> rollback -> corrected == authority ------------
// (1) advance 0..divergeTick with authorityCmds; (2) SAVE a snapshot; (2b) speculatively advance a few ticks
// with the MISPREDICTED stream (a wrong redirect); (3) ROLLBACK to the snapshot + re-sim divergeTick..ticks
// with the CORRECT authorityCmds. The caller asserts the result == RunCrowdLockstep(...authorityCmds...) AND
// that the speculative state DIFFERED (a real divergence was corrected).
inline Crowd RunCrowdRollback(const CrowdConfig& cfg, const Crowd& initial,
                              const std::vector<CrowdCommand>& authorityCmds,
                              const std::vector<CrowdCommand>& mispredictCmds, int divergeTick, int ticks,
                              fx dt) {
    Crowd c = initial;
    for (int t = 0; t < divergeTick; ++t) SimCrowdTick(c, cfg, authorityCmds, t, dt);
    const CrowdSnapshot snap = SnapshotCrowd(c);
    int specTicks = ticks - divergeTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s) SimCrowdTick(c, cfg, mispredictCmds, divergeTick + s, dt);
    RestoreCrowd(c, snap);
    for (int t = divergeTick; t < ticks; ++t) SimCrowdTick(c, cfg, authorityCmds, t, dt);
    return c;
}

// ===================== The shared showcase scenario (--cr1-crowd-shot, both backends) ====================
// The AN2 header-shared-scenario pattern: RunCrowdShotScenario + RenderCrowdShot are the ONE implementation
// both backends call — strict-zero cross-backend BY CONSTRUCTION (pure-integer scenario; the raster consumes
// only integers). A TOP-DOWN density plot: thousands of agents as dots coloured by archetype, flowing along
// several streams toward a few converging goal points.

inline constexpr int kShotAgents = 3600;   // a representative count that renders CLEARLY (the 10k test proves
                                           // scale; the shot subsamples to 3600 for a legible density plot)
inline constexpr int kShotTicks  = 140;    // ticks the shot crowd is settled for
inline constexpr fx  kShotDt      = boids::kOne / 4;   // 0.25s Q16.16 tick

// MakeShotConfig(): the 3-archetype / 3-goal shot tuning (pedestrian / runner / wanderer). All Q16.16 binary
// fractions. Goals sit on the +x side; agents start in three lanes on the -x side and stream toward the goals.
inline CrowdConfig MakeShotConfig() {
    CrowdConfig cfg;
    // archetype 0 pedestrian: moderate speed, roomy personal space, gentle goal pull.
    cfg.archetypes.push_back(CrowdArchetype{ boids::kOne * 4, boids::kOne * 3 / 2,  boids::kOne / 4, boids::kOne / 2 });
    // archetype 1 runner: fast, tighter spacing, strong goal pull.
    cfg.archetypes.push_back(CrowdArchetype{ boids::kOne * 7, boids::kOne,          boids::kOne / 2, boids::kOne / 3 });
    // archetype 2 wanderer: slow, wide spacing, weak goal pull (lingers/mills).
    cfg.archetypes.push_back(CrowdArchetype{ boids::kOne * 2, boids::kOne * 2,      boids::kOne / 8, boids::kOne / 2 });
    // 3 goal points on the +x side (the streams converge here).
    cfg.goals.push_back(FxVec3{ boids::kOne * 60, boids::kOne * 18, 0 });
    cfg.goals.push_back(FxVec3{ boids::kOne * 64, 0,                0 });
    cfg.goals.push_back(FxVec3{ boids::kOne * 60, -boids::kOne * 18, 0 });
    cfg.sepGain  = boids::kOne / 2;    // 0.5
    cfg.maxForce = boids::kOne * 10;
    FinalizeCrowdConfig(cfg);
    return cfg;
}

// MakeShotCrowd(cfg): lay out kShotAgents agents in a deterministic grid on the -x side, assigning archetype
// + goal by a fixed integer pattern (NO RNG) so three lanes stream to the three goals. Pure integer layout.
inline Crowd MakeShotCrowd(const CrowdConfig& cfg) {
    Crowd c;
    const int cols = 60, rows = (kShotAgents + cols - 1) / cols;   // 60 x 60 = 3600
    c.agents.reserve((size_t)kShotAgents);
    c.archetype.reserve((size_t)kShotAgents);
    c.goal.reserve((size_t)kShotAgents);
    int made = 0;
    for (int r = 0; r < rows && made < kShotAgents; ++r) {
        for (int col = 0; col < cols && made < kShotAgents; ++col, ++made) {
            // A grid on the -x side: x in [-60,-18] world units, y in [-30,+30]. 0.7-unit spacing.
            const fx x = (fx)(-boids::kOne * 60 + (fx)((int64_t)col * boids::kOne * 7 / 10));
            const fx y = (fx)(-boids::kOne * 30 + (fx)((int64_t)r   * boids::kOne * 7 / 10));
            Agent a; a.pos = FxVec3{x, y, 0}; a.vel = FxVec3{0, 0, 0};
            c.agents.push_back(a);
            c.archetype.push_back((uint8_t)(made % 3));      // 3 archetypes interleaved
            c.goal.push_back((uint32_t)(r % 3));             // 3 lanes by row -> 3 goals (streams)
        }
    }
    (void)cfg;
    return c;
}

// The recorded shot result (positions + digest for the stat line; the raster reads the crowd directly).
struct CrowdShotRun {
    Crowd    crowd;                // the settled crowd (rendered)
    int32_t  agents      = 0;
    int32_t  archetypes  = 0;
    int32_t  ticks       = 0;
    int32_t  goalCount   = 0;
    fx       meanToGoal0 = 0;      // mean L1 to goal BEFORE the run (the "they started far" stat)
    fx       meanToGoalN = 0;      // mean L1 to goal AFTER  the run (DROPS -> they flowed to the goals)
    uint64_t digest      = 0;      // the full-state digest (the two-run comparison currency)
};

// RunCrowdShotScenario(): the pure function both backends call — build the config + crowd, settle kShotTicks,
// record the digest + the before/after goal distance. Deterministic (NO RNG, NO clock).
inline CrowdShotRun RunCrowdShotScenario() {
    CrowdShotRun run;
    const CrowdConfig cfg = MakeShotConfig();
    Crowd c = MakeShotCrowd(cfg);
    run.agents     = (int32_t)CrowdSize(c);
    run.archetypes = (int32_t)cfg.archetypes.size();
    run.ticks      = kShotTicks;
    run.goalCount  = (int32_t)cfg.goals.size();
    run.meanToGoal0 = MeasureCrowd(c, cfg, false).meanToGoal;
    StepCrowdSteps(c, cfg, kShotDt, kShotTicks);
    run.meanToGoalN = MeasureCrowd(c, cfg, false).meanToGoal;
    run.crowd  = c;
    run.digest = DigestCrowd(c);
    return run;
}

// RenderCrowdShot(run, bgra, outW, outH): the PURE-INTEGER top-down density raster both backends call —
// strict-zero cross-backend BY CONSTRUCTION. Each agent is a dot coloured by archetype (pedestrian teal /
// runner amber / wanderer violet); the goal points are white crosses. 640x480 BGRA8, integer projection only.
inline void RenderCrowdShot(const CrowdShotRun& run, std::vector<uint8_t>& bgra, uint32_t& outW, uint32_t& outH) {
    const int W = 640, H = 480;
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {   // deep slate ground
        bgra[p * 4 + 0] = 18; bgra[p * 4 + 1] = 16; bgra[p * 4 + 2] = 14; bgra[p * 4 + 3] = 255;
    }
    auto putPx = [&](int ix, int iy, uint8_t r, uint8_t g, uint8_t b) {
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) return;
        uint8_t* d = &bgra[((size_t)iy * W + ix) * 4];
        d[0] = b; d[1] = g; d[2] = r; d[3] = 255;
    };
    // World (x,y) -> screen. World x in ~[-60,+66], y in ~[-32,+32]. Center x=64 world -> px ~ 320; 4 px/unit.
    const int kScale = 4, kCX = 300, kCY = 240;
    auto project = [&](const FxVec3& pos, int& sx, int& sy) {
        sx = kCX + (int)(((int64_t)pos.x * kScale) >> kFrac);
        sy = kCY - (int)(((int64_t)pos.y * kScale) >> kFrac);
    };
    // Archetype palette (r,g,b): 0 pedestrian teal, 1 runner amber, 2 wanderer violet.
    const uint8_t pal[3][3] = {{ 60, 200, 190 }, { 240, 170, 60 }, { 170, 90, 220 }};
    auto disc = [&](int cx, int cy, int rr, uint8_t r, uint8_t g, uint8_t b) {
        for (int dy = -rr; dy <= rr; ++dy)
            for (int dx = -rr; dx <= rr; ++dx)
                if (dx * dx + dy * dy <= rr * rr) putPx(cx + dx, cy + dy, r, g, b);
    };
    // Draw the agents (a 1px dot each; additive-ish overwrite gives a density read where they pile up).
    const uint32_t n = CrowdSize(run.crowd);
    for (uint32_t i = 0; i < n; ++i) {
        int sx, sy; project(run.crowd.agents[(size_t)i].pos, sx, sy);
        const uint8_t* c = pal[run.crowd.archetype[(size_t)i] % 3];
        putPx(sx, sy, c[0], c[1], c[2]);
    }
    // Draw the goal points as white crosses (drawn last, on top).
    // (goals are not in run; re-derive from MakeShotConfig — deterministic, same values.)
    const CrowdConfig cfg = MakeShotConfig();
    for (const FxVec3& gp : cfg.goals) {
        int gx, gy; project(gp, gx, gy);
        disc(gx, gy, 2, 255, 255, 255);
        for (int t = -6; t <= 6; ++t) { putPx(gx + t, gy, 255, 255, 255); putPx(gx, gy + t, 255, 255, 255); }
    }
}

}  // namespace crowd
}  // namespace hf::sim
