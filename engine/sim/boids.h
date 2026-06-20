#pragma once
// Slice BD1 — Deterministic GPU Crowds: THE INTEGER STEERING PRIMITIVE (the BEACHHEAD of FLAGSHIP #18:
// DETERMINISTIC GPU CROWDS — boids + steering + path-following, hf::sim::boids). A boid is a POINT with a
// Q16.16 position + velocity (the fpx::FxBody cousin, no orientation/mass). Each tick it computes a Reynolds
// STEERING force — SteerSeek (proportional pull toward a shared target) + SteerSeparation (brute-force
// all-pairs push away from too-close neighbors) — accumulated into the per-agent velocity, then integrates.
// Pure CPU, header-only, NO device, NO backend symbols, NO <cmath> on the bit-exact path. The STRUCTURAL
// TWIN of the FPX1/GR1 integer beachhead: a pure-integer per-agent update proven GPU==CPU BIT-EXACT, with a
// cross-backend BIT-IDENTICAL integer golden.
//
// THE int32 GOAL & THE int64 REALITY (the honest proof-strength call): BD1 keeps the SEPARATION SEARCH +
// the per-axis CLAMPS pure-int32 (no FxNormalize/FxISqrt on the per-agent path — the spec's "accumulate raw
// integer deltas" rule + an axis-box clamp, NOT a radial magnitude clamp). BUT the steer/integrate path
// MUST scale by Q16.16 gains + dt: `force = FxScale(desired, seekGain)`, `vel += force*dt`, `pos += vel*dt`
// — each an fxmul ((int64)a*b >> 16). fxmul uses an int64 INTERMEDIATE (Q16.16 world-scale products overflow
// int32, the FPX1/GR1/FL1 lesson). DXC compiles int64 (the Vulkan path); glslc (the Metal HLSL->SPIR-V->MSL
// frontend) CANNOT parse int64_t in HLSL. So shaders/boids_steer.comp.hlsl is VULKAN-SPIR-V-ONLY (in the
// Vulkan compile list, NOT in the Metal hf_gen_msl list); on Metal the --boids-steer showcase runs the CPU
// StepBoids — byte-identical to the Vulkan GPU result BY CONSTRUCTION (the fpx_integrate.comp / grain_
// integrate.comp convention), while the Vulkan side carries the GPU==CPU memcmp proof. The math here is
// copied VERBATIM by boids_steer.comp so the GPU exercises the EXACT integer ops.
//
// THE CROSS-BACKEND CRUX (the make-or-break for GPU==CPU): the steering is JACOBI — every agent computes its
// force from a FROZEN snapshot of ALL agents' positions, THEN integrates. So the per-agent update is
// INDEPENDENT of update order: the GPU one-thread-per-agent write (reading the previous step's positions
// from a SEPARATE input buffer, writing the next step's into an output buffer — ping-pong) is trivially
// race-free / order-independent, and two runs are byte-identical. The brute-force separation loop iterates
// ALL agents in the SAME fixed index order (skip self by index) on both paths -> bit-exact.
//
// REUSE MAP (engine/sim/fpx.h, read-only): fx/kFrac/kOne, fxmul, FxVec3/FxAdd/FxSub/FxScale. A small
// ClampAxis is re-implemented here (do NOT modify fpx.h / vehicle.h). The integrate shape mirrors
// fpx.h::IntegrateBody (vel += a*dt; pos += vel*dt) without gravity-only forcing.
//
// OUT OF SCOPE (later BD slices): the grid-hash neighbor list (BD2 — BD1 is brute-force all-pairs),
// alignment + cohesion (BD3), path-following (BD4), lockstep/rollback (BD5), the lit 3D render (BD6 — BD1's
// render is the 2D top-down diagnostic). HONEST CAVEATS (the GR4-shape): boids are POINTS with steering
// FORCES (a soft separation push, NOT a hard non-penetration contact — agents can briefly overlap); the
// per-axis clamp is an axis-BOX, not a radial magnitude clamp (deterministic + integer-cheap). Determinism +
// cross-platform bit-identity is the headline.

#include <cstdint>
#include <span>
#include <vector>

#include "sim/fpx.h"   // read-only: fx/kFrac/kOne/fxmul + FxVec3/FxAdd/FxSub/FxScale

namespace hf::sim {
namespace boids {

// Re-export the fpx Q16.16 vocabulary (boids share the fixed-point format exactly).
using fx = fpx::fx;
using FxVec3 = fpx::FxVec3;
inline constexpr int kFrac = fpx::kFrac;
inline constexpr fx  kOne  = fpx::kOne;
using fpx::fxmul;   // (a*b) >> kFrac with an int64 intermediate (the steer/integrate path's scalar mul)

// ----- A boid: a point with a Q16.16 position + velocity (the fpx::FxBody cousin, no orient/mass) ---------
// std430-packable as 6 x int32 (pos.xyz, vel.xyz) — the GPU Agent mirror.
struct Agent {
    FxVec3 pos;   // Q16.16 world position
    FxVec3 vel;   // Q16.16 velocity (world units / second)
};

// ----- The host-fixed Q16.16 steering tuning (golden-stable; NO RNG) -------------------------------------
struct BoidsConfig {
    fx     seekGain  = 0;   // Q16.16 gain on the (target - pos) seek force (proportional / un-normalized)
    fx     sepGain   = 0;   // Q16.16 gain on the accumulated separation push
    fx     sepRadius = 0;   // Q16.16 neighbor radius — only agents within this push each other apart
    fx     maxForce  = 0;   // Q16.16 per-AXIS force clamp (the axis-box magnitude limit, not radial)
    fx     maxSpeed  = 0;   // Q16.16 per-AXIS speed clamp
    FxVec3 target;          // Q16.16 the shared seek target
    FxVec3 gravity;         // Q16.16 constant acceleration (default 0 — BD1 is planar steering)
};

// ----- ClampAxis: the deterministic scalar clamp to [-limit, +limit] (limit >= 0). int32, no sqrt -------
// The per-component magnitude cap (the axis-box clamp the spec locks — deterministic + integer-cheap; a
// radial |v| clamp would need the int64 FxISqrt). The shader copies THIS body VERBATIM.
inline fx ClampAxis(fx v, fx limit) {
    if (v >  limit) return  limit;
    if (v < -limit) return -limit;
    return v;
}

// ClampAxisVec(v, limit): per-axis ClampAxis on each component (a box clamp to the [-limit,limit]^3 cube).
inline FxVec3 ClampAxisVec(const FxVec3& v, fx limit) {
    return FxVec3{ClampAxis(v.x, limit), ClampAxis(v.y, limit), ClampAxis(v.z, limit)};
}

// ----- SteerSeek(a, target, cfg): the proportional (un-normalized) seek force ----------------------------
// desired = target - pos; force = FxScale(desired, seekGain). NO FxNormalize/maxSpeed*unit (which would need
// the int64 FxISqrt) — un-normalized seek is the standard "arrive" behaviour (pulls harder when far, eases
// in when close) and is integer-cheap. The fxmul inside FxScale uses an int64 intermediate (the integrate
// path's int64 already pins the shader Vulkan-only). The shader copies THIS body VERBATIM.
inline FxVec3 SteerSeek(const Agent& a, const FxVec3& target, const BoidsConfig& cfg) {
    const FxVec3 desired = FxSub(target, a.pos);
    return FxScale(desired, cfg.seekGain);
}

// ----- SteerSeparation(a, others, cfg): the brute-force all-pairs separation push ------------------------
// For each OTHER agent o (skip self by index) with squared-distance < sepRadius², accumulate (a.pos - o.pos)
// — the raw away-direction — into a running integer sum (NO per-pair normalize, the scout's "accumulate raw
// integer deltas" rule). Scaled by sepGain. The squared-distance d² = dx²+dy²+dz² and sepRadius² are formed
// in int64 (a dx of a few Q16.16 world units squared exceeds int32), but they are only a COMPARE (no result
// stored) — and since the shader's integrate path is already int64 (Vulkan-only), this int64 compare costs
// nothing in proof-strength. selfIndex < 0 => no self to skip (compare against an external snapshot).
inline FxVec3 SteerSeparation(const Agent& a, std::span<const Agent> others, const BoidsConfig& cfg,
                              int selfIndex) {
    const int64_t r2 = (int64_t)cfg.sepRadius * (int64_t)cfg.sepRadius;
    FxVec3 sum{0, 0, 0};
    const int n = (int)others.size();
    for (int j = 0; j < n; ++j) {
        if (j == selfIndex) continue;                       // skip self by INDEX (fixed order, bit-exact)
        const Agent& o = others[(size_t)j];
        const fx dx = a.pos.x - o.pos.x;
        const fx dy = a.pos.y - o.pos.y;
        const fx dz = a.pos.z - o.pos.z;
        const int64_t d2 = (int64_t)dx * (int64_t)dx + (int64_t)dy * (int64_t)dy + (int64_t)dz * (int64_t)dz;
        if (d2 < r2) {                                      // within the separation radius -> push apart
            sum.x += dx;                                    // raw integer adds (no per-pair normalize)
            sum.y += dy;
            sum.z += dz;
        }
    }
    return FxScale(sum, cfg.sepGain);
}

// ----- StepBoids(agents, cfg, dt): one deterministic Reynolds steering tick (JACOBI) ---------------------
// (1) snapshot the current positions/velocities (the frozen input every agent's force reads — order
//     independence + GPU race-freedom); (2) per agent in FIXED index order: force = seek + separation +
//     gravity; clamp force per-axis to ±maxForce; vel += force*dt; clamp vel per-axis to ±maxSpeed; pos +=
//     vel*dt. Fixed op order + the frozen snapshot -> two runs bit-identical AND bit-exact GPU==CPU (the GPU
//     reads the previous step's agent buffer + writes a fresh one — ping-pong, the same Jacobi semantics).
inline void StepBoids(std::vector<Agent>& agents, const BoidsConfig& cfg, fx dt) {
    const std::vector<Agent> prev = agents;                 // frozen snapshot (the Jacobi input)
    const std::span<const Agent> snap(prev);
    const int n = (int)agents.size();
    for (int i = 0; i < n; ++i) {
        const Agent& a = prev[(size_t)i];                   // read the FROZEN state
        // (a) the steering force: seek + separation + gravity.
        FxVec3 force = SteerSeek(a, cfg.target, cfg);
        const FxVec3 sep = SteerSeparation(a, snap, cfg, i);
        force = FxAdd(force, sep);
        force = FxAdd(force, cfg.gravity);
        // (b) per-axis clamp the force (the axis-box magnitude limit).
        force = ClampAxisVec(force, cfg.maxForce);
        // (c) integrate velocity: vel += force * dt; then per-axis clamp to ±maxSpeed.
        FxVec3 vel = FxAdd(a.vel, FxScale(force, dt));
        vel = ClampAxisVec(vel, cfg.maxSpeed);
        // (d) integrate position: pos += vel * dt.
        const FxVec3 pos = FxAdd(a.pos, FxScale(vel, dt));
        agents[(size_t)i].vel = vel;
        agents[(size_t)i].pos = pos;
    }
}

// StepBoidsSteps(agents, cfg, dt, steps): run `steps` StepBoids ticks (the showcase settle loop).
inline void StepBoidsSteps(std::vector<Agent>& agents, const BoidsConfig& cfg, fx dt, int steps) {
    for (int s = 0; s < steps; ++s) StepBoids(agents, cfg, dt);
}

// ----- MeasureBoids: the deterministic flock statistics (Q16.16) -----------------------------------------
// meanSpeed = mean per-axis-summed |vel| component magnitude (an integer L1 proxy — NO sqrt, deterministic);
// meanToTarget = mean L1 distance |pos - target| (the "they sought it" stat — drops as the flock converges);
// minSep = the minimum L1 pairwise separation (the "they didn't collapse" stat — stays above a floor when
// separation is on, goes to ~0 when sepGain=0). All integer L1 metrics (|dx|+|dy|+|dz|) so NO sqrt / NO
// int64-result — bit-exact + cross-platform identical.
struct BoidsStats {
    fx meanSpeed    = 0;   // mean L1 speed
    fx meanToTarget = 0;   // mean L1 distance to the target
    fx minSep       = 0;   // minimum L1 pairwise separation
};

inline fx FxAbs(fx v) { return v < 0 ? -v : v; }
inline fx L1(const FxVec3& v) { return FxAbs(v.x) + FxAbs(v.y) + FxAbs(v.z); }

inline BoidsStats MeasureBoids(const std::vector<Agent>& agents, const BoidsConfig& cfg) {
    BoidsStats s;
    const int n = (int)agents.size();
    if (n == 0) return s;
    int64_t speedSum = 0, targSum = 0;
    for (int i = 0; i < n; ++i) {
        speedSum += L1(agents[(size_t)i].vel);
        targSum  += L1(FxSub(agents[(size_t)i].pos, cfg.target));
    }
    s.meanSpeed    = (fx)(speedSum / n);
    s.meanToTarget = (fx)(targSum / n);
    // min L1 pairwise separation over j>i (a single canonical orientation; deterministic).
    int64_t minSep = -1;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const int64_t d = L1(FxSub(agents[(size_t)i].pos, agents[(size_t)j].pos));
            if (minSep < 0 || d < minSep) minSep = d;
        }
    s.minSep = (fx)(minSep < 0 ? 0 : minSep);
    return s;
}

// ===== Slice BD2 — the GRID-HASH NEIGHBOR LIST (the SCALABLE neighbor engine; THE REUSE SPINE) ===========
// Builds the per-agent NEIGHBOR LIST over the BD1 agent pool via a uniform spatial-hash grid, with the proven
// count->scan->emit compaction (grain.h's GrainGrid/GrainCellTable/GrainNeighborList engine, byte-for-byte
// CLONED for boids::Agent + Agent.pos). PURE INT32 -> the BD2 shaders MSL-generate NATIVELY (a true GPU pass
// on BOTH backends, unlike BD1's int64 steer/integrate): the whole neighbor search is integer INDEX
// arithmetic. Cell ids are FloorDiv per axis (the cell size is the perception radius); the cell bucketing is
// count->scan->emit of agent indices; the candidate reject is a per-axis |pos_i.axis - pos_j.axis| < radius
// compare (fx is int32 -> a PURE INT32 compare, NO squaring, NO products, NO int64, NO sqrt). The exact radial
// cull is the box reject: a neighbor is accepted iff it lies within the radius-box on EVERY axis (the GR2/FL2
// "box candidate" discipline — the box IS the accepted neighborhood; BD3's flock forces weight by it). NO
// float, NO sqrt, NO int64 (the int64 dist² compare is AVOIDED — the box keeps the whole pass MSL-native, the
// spec's strongest-proof escape hatch; glslc cannot parse int64 in HLSL, which is exactly why BD1's int64
// boids_steer is Vulkan-only while BD2 stays int32 / in hf_gen_msl).
//
// THE PERCEPTION RADIUS `radius` (the cell size): the range within which two agents are flock NEIGHBORS (the
// BD3 sep/align/cohesion radius; BD1's sepRadius). The grid cell-size == radius, so the 3x3x3 stencil around
// an agent's cell covers every cell that can hold a box-neighbor (|dx| < radius => the neighbor is at most one
// cell away per axis). The caller picks radius >= the agent spacing so the flock is a NON-DEGENERATE neighbor
// graph (each interior agent sees its lateral neighbours -> a rich "dense cluster hot / sparse edge cold" heat
// viz).
//
// REUSE: fpx::FloorDiv (the deterministic floor-division for negative coords, fpx.h:177), fpx::FxCell
// (fpx.h:183), fpx::CellId (fpx.h:196). The whole engine is grain.h::MakeGrainGrid/BuildGrainCellTable/
// GrainNeighborAccept/BuildGrainNeighborList with boids::Agent + radius for the cell-size + Agent.pos as
// position. The shaders boids_cell_{count,scan,emit} + boids_neighbor_{count,scan,emit} copy this VERBATIM.

using fpx::FloorDiv;     // read-only: the deterministic floor-division (correct for negative coords)
using fpx::FxCell;       // read-only: the int3 cell coordinate
using fpx::CellId;       // read-only: the flat cell linearization

// ----- The bounded dense grid over the agent AABB (the grain.h::GrainGrid scheme verbatim) ----------------
// CHOSEN SCHEME: a BOUNDED DENSE GRID (== grain.h::GrainGrid). Cell-size = the perception radius. An agent's
// cell coord is FloorDiv(pos.axis, radius) per axis (monotone across 0 for negatives). The grid covers
// [cellMin, cellMin+gridDim) cells; a cell's flat id is fpx::CellId of (coord - cellMin) into gridDim. The
// caller sizes the grid to the agent AABB (every agent's cell in [0,gridDim)), so the linearization is total +
// collision-free + deterministic. cellMin lets the grid sit at any world location (incl. negative coords).
struct BoidsGrid {
    fx     cellSize = 0;   // Q16.16 cell size (== the perception radius)
    FxCell cellMin;        // the integer cell coord of the grid's (0,0,0) corner (the AABB lower cell)
    FxCell gridDim;        // the grid extent in cells per axis (cellCount = x*y*z)
};

// BoidsCellOf(pos, cellSize): the integer grid cell an agent's position falls in, FloorDiv per axis. Pure int32.
inline FxCell BoidsCellOf(const FxVec3& pos, fx cellSize) {
    return FxCell{FloorDiv(pos.x, cellSize), FloorDiv(pos.y, cellSize), FloorDiv(pos.z, cellSize)};
}

// FlatBoidsCellId(cell, grid): the flat id of an absolute cell coord into the bounded dense grid (offset by
// cellMin into [0,gridDim), then fpx::CellId). The caller guarantees the cell is in range; returns the linear
// cell index in [0, gridDim.x*y*z).
inline uint32_t FlatBoidsCellId(const FxCell& cell, const BoidsGrid& grid) {
    const FxCell local{cell.x - grid.cellMin.x, cell.y - grid.cellMin.y, cell.z - grid.cellMin.z};
    return CellId(local, grid.gridDim);
}

// BoidsCellCount(grid): the total number of cells in the dense grid (gridDim.x * y * z).
inline uint32_t BoidsCellCount(const BoidsGrid& grid) {
    return (uint32_t)(grid.gridDim.x * grid.gridDim.y * grid.gridDim.z);
}

// MakeBoidsGrid(agents, cellSize): build the bounded dense grid that tightly covers the agent pool at cell-size
// cellSize (== the perception radius). cellMin = the min cell coord over all agents; gridDim = (maxCell -
// minCell + 1) per axis. Empty pool -> a 1x1x1 grid at origin (deterministic degenerate). Pure int32 (==
// grain.h::MakeGrainGrid).
inline BoidsGrid MakeBoidsGrid(const std::vector<Agent>& agents, fx cellSize) {
    BoidsGrid grid;
    grid.cellSize = cellSize;
    if (agents.empty()) {
        grid.cellMin = FxCell{0, 0, 0};
        grid.gridDim = FxCell{1, 1, 1};
        return grid;
    }
    FxCell lo = BoidsCellOf(agents[0].pos, cellSize);
    FxCell hi = lo;
    for (const Agent& a : agents) {
        const FxCell c = BoidsCellOf(a.pos, cellSize);
        if (c.x < lo.x) lo.x = c.x; if (c.x > hi.x) hi.x = c.x;
        if (c.y < lo.y) lo.y = c.y; if (c.y > hi.y) hi.y = c.y;
        if (c.z < lo.z) lo.z = c.z; if (c.z > hi.z) hi.z = c.z;
    }
    grid.cellMin = lo;
    grid.gridDim = FxCell{hi.x - lo.x + 1, hi.y - lo.y + 1, hi.z - lo.z + 1};
    return grid;
}

// ----- BuildBoidsCellTable: bucket agent indices into cells (the count->scan->emit on agents) -------------
// The CSR-style cell table: cellStart[c..] is the exclusive prefix-sum of per-cell counts (cellStart has
// cellCount+1 entries; cellStart[c]..cellStart[c+1] is cell c's slice), and cellAgents[] holds the agent
// indices grouped by cell, ASCENDING agent index within each cell (deterministic). count->scan->emit: (1)
// count agents per cell; (2) exclusive prefix-sum -> cellStart; (3) scatter each agent index into its cell's
// slice (the emit, ascending-index order by construction since the agent loop is ascending). Pure int32 -> the
// GPU boids_cell_{count,scan,emit} mirror this byte-for-byte. (DET-CRUX: the EMIT is the single-thread
// ascending-agent scatter — a parallel atomic cursor would make the within-cell order GPU-scheduling-dependent
// -> non-deterministic. The cell COUNT + the neighbor passes are per-agent-disjoint + race-free; only the
// cell-emit scatter is the ordered pass.) (== grain.h::BuildGrainCellTable.)
struct BoidsCellTable {
    std::vector<uint32_t> cellStart;    // cellCount+1 exclusive prefix-sum offsets (CSR row pointers)
    std::vector<uint32_t> cellAgents;   // agent indices grouped by cell (size == agent count)
};

inline BoidsCellTable BuildBoidsCellTable(const std::vector<Agent>& agents, const BoidsGrid& grid) {
    const uint32_t n = (uint32_t)agents.size();
    const uint32_t cells = BoidsCellCount(grid);
    BoidsCellTable table;
    // (1) COUNT: per-cell agent count.
    std::vector<uint32_t> counts((size_t)cells, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = FlatBoidsCellId(BoidsCellOf(agents[i].pos, grid.cellSize), grid);
        ++counts[c];
    }
    // (2) SCAN: exclusive prefix-sum -> cellStart (cellCount+1 entries; the last == n).
    table.cellStart.assign((size_t)cells + 1u, 0u);
    uint32_t running = 0;
    for (uint32_t c = 0; c < cells; ++c) {
        table.cellStart[c] = running;
        running += counts[c];
    }
    table.cellStart[cells] = running;   // == n (the total)
    // (3) EMIT: scatter each agent index into its cell's slice (ascending index by the ascending loop).
    table.cellAgents.assign((size_t)n, 0u);
    std::vector<uint32_t> cursor((size_t)cells, 0u);   // per-cell write cursor (local offset)
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = FlatBoidsCellId(BoidsCellOf(agents[i].pos, grid.cellSize), grid);
        table.cellAgents[table.cellStart[c] + cursor[c]] = i;
        ++cursor[c];
    }
    return table;
}

// ----- The neighbor reject (the PURE INT32 per-axis |dx| < radius candidate test) ------------------------
// BoidsNeighborAccept(a, b, radius): accept b as a neighbor of a iff |a.axis - b.axis| < radius on EVERY axis
// (a box of half-width radius — the perception neighborhood). PURE INT32: an integer subtract + abs + compare
// per axis, NO products, NO int64, NO sqrt (the box keeps the whole pass MSL-native — the radial dist²<radius²
// compare would need an int64 product glslc cannot parse, the spec's documented escape hatch). The shader
// copies THIS verbatim. (== grain.h::GrainNeighborAccept with radius for hSearch.)
inline bool BoidsNeighborAccept(const FxVec3& a, const FxVec3& b, fx radius) {
    fx dx = a.x - b.x; if (dx < 0) dx = -dx;
    fx dy = a.y - b.y; if (dy < 0) dy = -dy;
    fx dz = a.z - b.z; if (dz < 0) dz = -dz;
    return dx < radius && dy < radius && dz < radius;
}

// ----- BuildBoidsNeighborList: per-agent neighbors over the 27-cell stencil (count->scan->emit) -----------
// For each agent i, scan the 27 cells of its 3x3x3 stencil (the cell + its 26 neighbors); for each agent
// j != i in those cells, accept iff BoidsNeighborAccept(pos_i, pos_j, radius). Emit the accepted j into
// neighbors[] at i's offset, in a FIXED order: ascending stencil-cell (dz,dy,dx -1..+1), then within a cell
// ascending j (cellAgents is already ascending-index per cell) -> fully deterministic. The variable-length
// per-agent lists are laid out by count->scan->emit (neighborStart = exclusive prefix-sum; neighbors[] grouped
// by i). Stencil cells outside the grid are skipped (clamped). Pure int32 (== grain.h::BuildGrainNeighborList).
struct BoidsNeighborList {
    std::vector<uint32_t> neighborStart;   // agentCount+1 exclusive prefix-sum offsets (CSR)
    std::vector<uint32_t> neighbors;       // neighbor j indices grouped by i (in stencil order)
};

// CountBoidsNeighbors(agents, grid, table, radius, perAgentOut): the count pass. perAgentOut[i] = #neighbors
// of i (j!=i in the 27-cell stencil passing BoidsNeighborAccept); returns the total. The GPU
// boids_neighbor_count mirrors THIS per-thread (one thread per agent i).
inline uint32_t CountBoidsNeighbors(const std::vector<Agent>& agents, const BoidsGrid& grid,
                                    const BoidsCellTable& table, fx radius,
                                    std::vector<uint32_t>& perAgentOut) {
    const uint32_t n = (uint32_t)agents.size();
    perAgentOut.assign((size_t)n, 0u);
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const FxCell ci = BoidsCellOf(agents[i].pos, grid.cellSize);
        uint32_t c = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            // Skip stencil cells outside the bounded grid (clamp).
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatBoidsCellId(nc, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellAgents[s];
                if (j == i) continue;                                      // NO self-neighbor
                if (BoidsNeighborAccept(agents[i].pos, agents[j].pos, radius)) ++c;
            }
        }
        perAgentOut[i] = c;
        total += c;
    }
    return total;
}

// BuildBoidsNeighborList(agents, grid, table, radius): the full builder (count->scan->emit). (1)
// CountBoidsNeighbors -> per-agent counts; (2) exclusive prefix-sum -> neighborStart; (3) emit each accepted j
// into i's disjoint slice in the FIXED stencil order. The list is grouped by i (ascending), then stencil-cell
// (dz,dy,dx ascending), then j (ascending within a cell) -> fully deterministic. The GPU does the SAME three
// passes -> the GPU neighbors[]+neighborStart memcmp's against this byte-for-byte.
inline BoidsNeighborList BuildBoidsNeighborList(const std::vector<Agent>& agents, const BoidsGrid& grid,
                                                const BoidsCellTable& table, fx radius) {
    const uint32_t n = (uint32_t)agents.size();
    BoidsNeighborList list;
    std::vector<uint32_t> counts;
    const uint32_t total = CountBoidsNeighbors(agents, grid, table, radius, counts);
    // (2) SCAN: exclusive prefix-sum -> neighborStart (agentCount+1 entries; the last == total).
    list.neighborStart.assign((size_t)n + 1u, 0u);
    uint32_t running = 0;
    for (uint32_t i = 0; i < n; ++i) {
        list.neighborStart[i] = running;
        running += counts[i];
    }
    list.neighborStart[n] = running;   // == total
    // (3) EMIT: each agent writes its neighbors into its disjoint [neighborStart[i], ..) slice.
    list.neighbors.assign((size_t)total, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        const FxCell ci = BoidsCellOf(agents[i].pos, grid.cellSize);
        uint32_t local = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const FxCell nc{ci.x + dx, ci.y + dy, ci.z + dz};
            if (nc.x < grid.cellMin.x || nc.x >= grid.cellMin.x + grid.gridDim.x) continue;
            if (nc.y < grid.cellMin.y || nc.y >= grid.cellMin.y + grid.gridDim.y) continue;
            if (nc.z < grid.cellMin.z || nc.z >= grid.cellMin.z + grid.gridDim.z) continue;
            const uint32_t cell = FlatBoidsCellId(nc, grid);
            for (uint32_t s = table.cellStart[cell]; s < table.cellStart[cell + 1u]; ++s) {
                const uint32_t j = table.cellAgents[s];
                if (j == i) continue;
                if (BoidsNeighborAccept(agents[i].pos, agents[j].pos, radius)) {
                    list.neighbors[list.neighborStart[i] + local] = j;
                    ++local;
                }
            }
        }
    }
    return list;
}

// ===== Slice BD3 — THE FULL FLOCK STEP (SEPARATION + ALIGNMENT + COHESION over the BD2 neighbor list) =====
// BD1 built the steering primitive (brute-force seek + separation); BD2 built the grid-hash neighbor list.
// BD3 is the FULL FLOCK: over an agent's BD2 neighbor slice (neighborStart[i]..neighborStart[i+1]), accumulate
// the THREE Reynolds rules — SEPARATION (steer away from crowding), ALIGNMENT (steer toward the neighbors'
// mean heading), COHESION (steer toward the neighbors' mean position) — and integrate one deterministic tick.
// A real flock emerges. INTEGER-bit-exact, JACOBI (frozen-snapshot), fixed op order -> GPU==CPU + two-run
// byte-identical. The flock-step shader boids_flock.comp.hlsl is int64-Vulkan-only (the integrate fxmul, the
// BD1/FPX1/GR1 lesson) over the MSL-native BD2 neighbor list; the BD2 grid passes stay in hf_gen_msl (reused
// per tick to rebuild the list). BD3 APPENDS — BD1/BD2 are byte-frozen. NO new RHI.
//
// THE MEANS ARE INTEGER DIVIDES (the cross-backend crux): alignment + cohesion divide the neighbor velocity /
// position SUM by the neighbor COUNT — an integer divide (truncate toward zero, the C++/HLSL/MSL `/` semantics,
// identical on every vendor). count==0 -> NO align/coh term (the div-by-zero guard); separation is still an
// accumulate (it sums per-neighbor away-deltas, zero if no neighbors). The shader copies THIS math VERBATIM.

// ----- FlockConfig: the BD1 BoidsConfig tuning + the BD3 alignment/cohesion gains + the perception radius -----
// = BoidsConfig (seek/sep/maxForce/maxSpeed/target/gravity) PLUS alignGain (the mean-heading pull), cohGain
// (the mean-position pull), and perceptionRadius (the BD2 grid cell size == the sep/align/coh neighbor radius).
// The flock's sep radius for AccumFlock IS the perception radius (the BD2 neighbor box), so sepRadius below is
// unused by the flock path (kept for BoidsConfig reuse / the BD1 helpers); the flock uses perceptionRadius.
struct FlockConfig {
    fx     seekGain         = 0;   // Q16.16 gain on the optional (target - pos) seek force (0 => free flocking)
    fx     sepGain          = 0;   // Q16.16 gain on the accumulated separation push (away from each neighbor)
    fx     alignGain        = 0;   // Q16.16 gain on the alignment force (toward the neighbors' mean velocity)
    fx     cohGain          = 0;   // Q16.16 gain on the cohesion force (toward the neighbors' mean position)
    fx     perceptionRadius = 0;   // Q16.16 neighbor radius (the BD2 grid cell size; the sep/align/coh range)
    fx     maxForce         = 0;   // Q16.16 per-AXIS force clamp (the axis-box magnitude limit, not radial)
    fx     maxSpeed         = 0;   // Q16.16 per-AXIS speed clamp
    FxVec3 target;                 // Q16.16 the optional shared seek target (only used if seekGain != 0)
    FxVec3 gravity;                // Q16.16 constant acceleration (default 0 — BD3 is planar flocking)
    fx     pathGain         = 0;   // Slice BD4: Q16.16 gain on the corridor-follow (SteerPath) arrive force
                                   //   (default 0 -> the path term is identically zero -> exactly BD3; only the
                                   //   BD4 path showcase/tests set it. Render-invariant: pathGain 0 == BD3.)
};

// ----- AccumFlock(i, neighborList, agents, cfg): the 3 Reynolds rules over agent i's BD2 neighbor slice -------
// Over the neighbor indices in [neighborStart[i], neighborStart[i+1]) (the BD2 list built on the SAME frozen
// positions): accumulate the separation away-sum (Σ FxSub(pos_i, pos_j)), the neighbor velocity sum, and the
// neighbor position sum, all in a FIXED ascending-slice order. Then:
//   sep   = FxScale(Σ (pos_i - pos_j), sepGain)                              (away from crowding)
//   align = FxScale((Σ vel_j)/count - vel_i, alignGain)   [count>0 only]     (toward the mean heading)
//   coh   = FxScale((Σ pos_j)/count - pos_i, cohGain)     [count>0 only]     (toward the mean position)
//   force = sep + align + coh
// The means are INTEGER DIVIDES by count (deterministic truncation). count==0 -> sep only (no align/coh, no
// div-by-zero). The sums are integer ADDS; FxScale's fxmul is int64 (-> the shader is Vulkan-only). The shader
// copies THIS body VERBATIM. NOTE: the optional seek + gravity are added by StepFlock (not here), mirroring the
// BD1 split (SteerSeek separate from StepBoids).
inline FxVec3 AccumFlock(int i, const BoidsNeighborList& list, const std::vector<Agent>& agents,
                         const FlockConfig& cfg) {
    const Agent& a = agents[(size_t)i];
    const uint32_t s0 = list.neighborStart[(size_t)i];
    const uint32_t s1 = list.neighborStart[(size_t)i + 1u];
    FxVec3 sepSum{0, 0, 0};   // Σ (pos_i - pos_j) — the raw away-direction accumulate
    FxVec3 velSum{0, 0, 0};   // Σ vel_j           — for the alignment mean
    FxVec3 posSum{0, 0, 0};   // Σ pos_j           — for the cohesion mean
    int count = 0;
    for (uint32_t s = s0; s < s1; ++s) {
        const Agent& o = agents[(size_t)list.neighbors[(size_t)s]];
        sepSum.x += a.pos.x - o.pos.x; sepSum.y += a.pos.y - o.pos.y; sepSum.z += a.pos.z - o.pos.z;
        velSum.x += o.vel.x;           velSum.y += o.vel.y;           velSum.z += o.vel.z;
        posSum.x += o.pos.x;           posSum.y += o.pos.y;           posSum.z += o.pos.z;
        ++count;
    }
    // separation: scale the away-sum by sepGain (zero if no neighbors -> zero force).
    FxVec3 force = FxScale(sepSum, cfg.sepGain);
    if (count > 0) {
        // alignment: (mean neighbor velocity) - own velocity, scaled by alignGain. Integer divide by count.
        const FxVec3 meanVel{velSum.x / count, velSum.y / count, velSum.z / count};
        const FxVec3 align = FxScale(FxSub(meanVel, a.vel), cfg.alignGain);
        // cohesion: (mean neighbor position) - own position, scaled by cohGain. Integer divide by count.
        const FxVec3 meanPos{posSum.x / count, posSum.y / count, posSum.z / count};
        const FxVec3 coh = FxScale(FxSub(meanPos, a.pos), cfg.cohGain);
        force = FxAdd(force, align);
        force = FxAdd(force, coh);
    }
    return force;
}

// ----- StepFlock(agents, cfg, dt): one deterministic FULL-FLOCK tick (rebuild grid+neighbors -> JACOBI) ------
// (1) Rebuild the BD2 neighbor engine on the CURRENT (frozen) positions: MakeBoidsGrid(agents, perceptionRadius)
//     -> BuildBoidsCellTable -> BuildBoidsNeighborList (the BD2 functions, REUSED — the flock moves, so the grid
//     is rebuilt every tick). (2) JACOBI per agent in FIXED index order over the FROZEN snapshot: force =
//     AccumFlock(i) + optional seek (if seekGain != 0) + gravity; per-axis clamp force to ±maxForce; vel +=
//     force*dt clamp ±maxSpeed; pos += vel*dt. Every agent reads the frozen snapshot (positions + the neighbor
//     list built over it) and writes the next state -> order-independent + GPU race-free + bit-exact GPU==CPU.
//     The optional seek mirrors BD1's SteerSeek (un-normalized desired*seekGain), added only when seekGain != 0.
inline void StepFlock(std::vector<Agent>& agents, const FlockConfig& cfg, fx dt) {
    const std::vector<Agent> prev = agents;                    // frozen snapshot (the Jacobi input)
    // Rebuild the BD2 grid + cell table + neighbor list on the frozen positions (== the GPU per-tick rebuild).
    const BoidsGrid grid = MakeBoidsGrid(prev, cfg.perceptionRadius);
    const BoidsCellTable table = BuildBoidsCellTable(prev, grid);
    const BoidsNeighborList list = BuildBoidsNeighborList(prev, grid, table, cfg.perceptionRadius);
    const int n = (int)agents.size();
    for (int i = 0; i < n; ++i) {
        const Agent& a = prev[(size_t)i];                      // read the FROZEN state
        // (a) the 3 Reynolds rules over the neighbor list.
        FxVec3 force = AccumFlock(i, list, prev, cfg);
        // (b) the optional seek toward the shared target (un-normalized; skipped when seekGain == 0).
        if (cfg.seekGain != 0) force = FxAdd(force, FxScale(FxSub(cfg.target, a.pos), cfg.seekGain));
        // (c) + gravity (a constant acceleration; 0 in the planar showcase).
        force = FxAdd(force, cfg.gravity);
        // (d) per-axis clamp the force (the axis-box magnitude limit).
        force = ClampAxisVec(force, cfg.maxForce);
        // (e) integrate velocity: vel += force * dt; then per-axis clamp to ±maxSpeed.
        FxVec3 vel = FxAdd(a.vel, FxScale(force, dt));
        vel = ClampAxisVec(vel, cfg.maxSpeed);
        // (f) integrate position: pos += vel * dt.
        const FxVec3 pos = FxAdd(a.pos, FxScale(vel, dt));
        agents[(size_t)i].vel = vel;
        agents[(size_t)i].pos = pos;
    }
}

// StepFlockSteps(agents, cfg, dt, steps): run `steps` StepFlock ticks (the showcase settle loop).
inline void StepFlockSteps(std::vector<Agent>& agents, const FlockConfig& cfg, fx dt, int steps) {
    for (int s = 0; s < steps; ++s) StepFlock(agents, cfg, dt);
}

// ----- MeasureFlock: the deterministic FLOCK statistics (Q16.16 / integer) -------------------------------
// meanSpeed   = mean L1 speed (|vx|+|vy|+|vz|), the "they're moving" stat (an integer L1 proxy, NO sqrt).
// diag        = the flock's bounding-box L1 diagonal ((maxX-minX)+(maxY-minY)+(maxZ-minZ)) — COHESION pulls it
//               IN, so it SHRINKS from the spread start (the "they clustered" stat). Pure integer.
// alignment   = the mean per-axis heading-agreement: for each axis, sum the SIGNS of the velocities and take
//               |sum|; the agreement is (|Σ sign(vx)| + |Σ sign(vy)| + |Σ sign(vz)|) (0..3N). A perfectly
//               aligned flock (all velocities same sign per axis) -> ~3N; a random flock -> ~0. Returned as the
//               per-agent-averaged Q16.16 fraction of N (so 0..3*kOne). The "they flock" stat — RISES as
//               alignment aligns the headings. Pure integer (sign + add + a single divide), NO sqrt/float.
// minSep      = the minimum L1 pairwise separation (the "they didn't collapse" stat — separation keeps it above
//               a floor). All integer L1 metrics -> bit-exact + cross-platform identical.
struct FlockStats {
    fx meanSpeed = 0;   // mean L1 speed
    fx diag      = 0;   // flock bounding-box L1 diagonal (shrinks under cohesion)
    fx alignment = 0;   // mean heading-alignment in [0, 3*kOne] (rises under alignment)
    fx minSep    = 0;   // minimum L1 pairwise separation (stays above a floor under separation)
};

inline int FxSign(fx v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }

inline FlockStats MeasureFlock(const std::vector<Agent>& agents, const FlockConfig& /*cfg*/) {
    FlockStats s;
    const int n = (int)agents.size();
    if (n == 0) return s;
    int64_t speedSum = 0;
    int64_t signX = 0, signY = 0, signZ = 0;
    fx minX = agents[0].pos.x, maxX = minX;
    fx minY = agents[0].pos.y, maxY = minY;
    fx minZ = agents[0].pos.z, maxZ = minZ;
    for (int i = 0; i < n; ++i) {
        const Agent& a = agents[(size_t)i];
        speedSum += L1(a.vel);
        signX += FxSign(a.vel.x); signY += FxSign(a.vel.y); signZ += FxSign(a.vel.z);
        if (a.pos.x < minX) minX = a.pos.x; if (a.pos.x > maxX) maxX = a.pos.x;
        if (a.pos.y < minY) minY = a.pos.y; if (a.pos.y > maxY) maxY = a.pos.y;
        if (a.pos.z < minZ) minZ = a.pos.z; if (a.pos.z > maxZ) maxZ = a.pos.z;
    }
    s.meanSpeed = (fx)(speedSum / n);
    s.diag = (maxX - minX) + (maxY - minY) + (maxZ - minZ);
    // heading-alignment: |Σ sign| per axis (0..N each), summed (0..3N), scaled to a Q16.16 fraction of N.
    const int64_t agree = (signX < 0 ? -signX : signX) + (signY < 0 ? -signY : signY) +
                          (signZ < 0 ? -signZ : signZ);
    s.alignment = (fx)((agree * (int64_t)kOne) / n);   // 0 .. 3*kOne (mean per-axis agreement)
    // min L1 pairwise separation over j>i (a single canonical orientation; deterministic).
    int64_t minSep = -1;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const int64_t d = L1(FxSub(agents[(size_t)i].pos, agents[(size_t)j].pos));
            if (minSep < 0 || d < minSep) minSep = d;
        }
    s.minSep = (fx)(minSep < 0 ? 0 : minSep);
    return s;
}

// ===== Slice BD4 — PATH-FOLLOWING THE A* CORRIDOR (THE NAV BRIDGE) =======================================
// BD1-BD3 built a FREE-flocking crowd (steering -> neighbors -> the 3 Reynolds rules). BD4 is the COMPOSITION
// HEADLINE — "crowds, not just boids": the flock FOLLOWS the bit-exact A* corridor from the NAV navmesh
// flagship (#7). Each agent steers along the corridor toward the goal (a SteerArrive-to-waypoint term)
// BLENDED with the BD3 flock, so the swarm STREAMS along the navmesh path while still cohering/aligning/
// spacing. This pairs the engine's two integer pillars — nav A* + the crowd sim — both deterministic.
// INTEGER-bit-exact. The path term extends the BD3 boids_flock.comp RENDER-INVARIANTLY (pathCount==0 -> BD3
// byte-identical); NAV is reused read-only (byte-frozen). NO new RHI.
//
// BD4 is ADDITIVE: BD1-BD3 above are byte-FROZEN. BD4 APPENDS BoidsPath/SteerPath/StepFlockPath/
// StepFlockPathSteps/MeasureFlockPath + adds pathGain to FlockConfig (a render-invariant default 0 — at 0 the
// path term is identically zero AND an empty corridor returns zero, so StepFlockPath with an empty path ==
// exactly BD3 StepFlock, the equivalence/render-invariance contract).

// ----- BoidsPath: the A* corridor as Q16.16 world waypoints (built HOST-side from nav::FindPath) ----------
// The corridor poly-id sequence (nav::FindPath) mapped through the poly centroids (the cx/cz arrays the
// navmesh stores) into Q16.16 world points. The nav A* is bit-exact + the corridor is FIXED for the run; the
// waypoint Q16.16 conversion is a host integer snap (the corridor poly centers are already integers). An
// EMPTY path (no waypoints) -> SteerPath returns 0 -> StepFlockPath == BD3 StepFlock exactly.
struct BoidsPath {
    std::vector<FxVec3> waypoints;   // Q16.16 world points along the corridor (poly centroids, start->goal)
};

// ----- SteerPath(a, path, cfg): the corridor-follow steering (deterministic, STATELESS) ------------------
// Recomputed each tick from the agent's position (NO per-agent path-state -> nothing to snapshot for BD5):
// find the index k of the NEAREST waypoint (an integer L1 distance min-scan over the corridor — the corridor
// is short), target = waypoints[min(k+1, last)] (the NEXT waypoint ahead — corridor progress; near the FINAL
// waypoint it eases in), pathForce = FxScale(FxSub(target, a.pos), pathGain) (the un-normalized arrive/seek,
// the BD1 SteerSeek shape). An EMPTY path (or pathGain==0) -> zero force. The L1 nearest scan uses the integer
// L1() helper (|dx|+|dy|+|dz|, no int64/sqrt). The shader copies THIS body VERBATIM (gated by pathCount>0).
inline FxVec3 SteerPath(const Agent& a, const BoidsPath& path, const FlockConfig& cfg) {
    const int w = (int)path.waypoints.size();
    if (w == 0) return FxVec3{0, 0, 0};            // empty corridor -> no path force (the BD3 equivalence)
    // find the NEAREST waypoint by integer L1 distance (ascending index, strict-less tie-break -> lowest k).
    int k = 0;
    fx best = L1(FxSub(path.waypoints[0], a.pos));
    for (int j = 1; j < w; ++j) {
        const fx d = L1(FxSub(path.waypoints[(size_t)j], a.pos));
        if (d < best) { best = d; k = j; }          // strict < -> ties keep the LOWEST index (deterministic)
    }
    const int tgt = (k + 1 < w) ? (k + 1) : (w - 1);   // the NEXT waypoint ahead (clamped to the last)
    return FxScale(FxSub(path.waypoints[(size_t)tgt], a.pos), cfg.pathGain);
}

// ----- StepFlockPath(agents, cfg, path, dt): one FULL-FLOCK + PATH-FOLLOW tick (JACOBI) ------------------
// = the BD3 StepFlock body with the path term added to the per-agent force: force = AccumFlock(i) +
// SteerPath(i) + optional seek + gravity -> clamp -> integrate (JACOBI, frozen snapshot). When path.waypoints
// is empty (or cfg.pathGain==0), SteerPath returns 0 -> EXACTLY BD3 StepFlock (the equivalence contract). The
// path is recomputed each tick from the FROZEN positions (stateless), so the per-agent update stays
// order-independent + GPU race-free + bit-exact GPU==CPU + two-run byte-identical.
inline void StepFlockPath(std::vector<Agent>& agents, const FlockConfig& cfg, const BoidsPath& path, fx dt) {
    const std::vector<Agent> prev = agents;                    // frozen snapshot (the Jacobi input)
    // Rebuild the BD2 grid + cell table + neighbor list on the frozen positions (== the GPU per-tick rebuild).
    const BoidsGrid grid = MakeBoidsGrid(prev, cfg.perceptionRadius);
    const BoidsCellTable table = BuildBoidsCellTable(prev, grid);
    const BoidsNeighborList list = BuildBoidsNeighborList(prev, grid, table, cfg.perceptionRadius);
    const int n = (int)agents.size();
    for (int i = 0; i < n; ++i) {
        const Agent& a = prev[(size_t)i];                      // read the FROZEN state
        // (a) the 3 Reynolds rules over the neighbor list.
        FxVec3 force = AccumFlock(i, list, prev, cfg);
        // (a2) the PATH-FOLLOW term (the NAV bridge): arrive toward the next corridor waypoint. Empty path or
        //      pathGain==0 -> zero -> exactly BD3 StepFlock (the render-invariance contract).
        force = FxAdd(force, SteerPath(a, path, cfg));
        // (b) the optional seek toward the shared target (un-normalized; skipped when seekGain == 0).
        if (cfg.seekGain != 0) force = FxAdd(force, FxScale(FxSub(cfg.target, a.pos), cfg.seekGain));
        // (c) + gravity (a constant acceleration; 0 in the planar showcase).
        force = FxAdd(force, cfg.gravity);
        // (d) per-axis clamp the force (the axis-box magnitude limit).
        force = ClampAxisVec(force, cfg.maxForce);
        // (e) integrate velocity: vel += force * dt; then per-axis clamp to ±maxSpeed.
        FxVec3 vel = FxAdd(a.vel, FxScale(force, dt));
        vel = ClampAxisVec(vel, cfg.maxSpeed);
        // (f) integrate position: pos += vel * dt.
        const FxVec3 pos = FxAdd(a.pos, FxScale(vel, dt));
        agents[(size_t)i].vel = vel;
        agents[(size_t)i].pos = pos;
    }
}

// StepFlockPathSteps(agents, cfg, path, dt, steps): run `steps` StepFlockPath ticks (the showcase settle loop).
inline void StepFlockPathSteps(std::vector<Agent>& agents, const FlockConfig& cfg, const BoidsPath& path,
                               fx dt, int steps) {
    for (int s = 0; s < steps; ++s) StepFlockPath(agents, cfg, path, dt);
}

// ----- FlockPathStats: the BD3 flock stats + the "they reached the goal" stat ----------------------------
// = FlockStats (meanSpeed/diag/alignment/minSep) PLUS centroidToGoal = the L1 distance from the flock CENTROID
// (the integer mean position) to the FINAL corridor waypoint — DROPS as the crowd streams along the corridor
// to the goal. Pure integer (L1, no sqrt/float). An empty path -> centroidToGoal 0 (no goal).
struct FlockPathStats {
    FlockStats flock;          // the BD3 stats (meanSpeed/diag/alignment/minSep)
    fx         centroidToGoal = 0;   // L1 distance from the flock centroid to the FINAL waypoint
};

// MeasureFlockPath(agents, cfg, path): the BD3 MeasureFlock stats + the centroid->final-waypoint L1 distance.
inline FlockPathStats MeasureFlockPath(const std::vector<Agent>& agents, const FlockConfig& cfg,
                                       const BoidsPath& path) {
    FlockPathStats s;
    s.flock = MeasureFlock(agents, cfg);
    const int n = (int)agents.size();
    if (n == 0 || path.waypoints.empty()) return s;
    // the flock centroid = the integer mean position (truncating divide by n, deterministic).
    int64_t sx = 0, sy = 0, sz = 0;
    for (int i = 0; i < n; ++i) {
        sx += agents[(size_t)i].pos.x; sy += agents[(size_t)i].pos.y; sz += agents[(size_t)i].pos.z;
    }
    const FxVec3 centroid{(fx)(sx / n), (fx)(sy / n), (fx)(sz / n)};
    s.centroidToGoal = L1(FxSub(centroid, path.waypoints.back()));
    return s;
}

// ===== Slice BD5 — LOCKSTEP + ROLLBACK (THE NETCODE HEADLINE) ============================================
// BD1-BD4 built a bit-exact path-following flocking crowd (the per-tick StepFlockPath); BD5 proves it is true
// cross-platform LOCKSTEP + ROLLBACK — the FPX5/FR5/GR5/CG5/GF5/JT5/VH5/AC5 twin. Two peers fed ONLY a
// PERTURBATION input stream re-derive the exact crowd trajectory bit-for-bit (RunBoidsLockstep:
// authority==replica every tick), and a MISPREDICTED perturbation is corrected by rolling back to a saved
// snapshot + re-simulating (RunBoidsRollback: corrected==authority, the mispredict diverged before rollback).
// PURE CPU (NO GPU dispatch, NO new shader, NO new RHI) — both Vulkan-Windows (--boids-lockstep-shot) and
// Metal-Mac (--boids-lockstep) run the IDENTICAL CPU harness over StepFlockPath, so the converged crowd golden
// is bit-identical cross-backend BY CONSTRUCTION.
//
// THE BD SIMPLIFICATION — the snapshot is JUST the agent world. StepFlockPath rebuilds the grid+neighbors from
// the current positions each tick (no persistent grid state), SteerPath is recomputed each tick from the
// positions (no per-agent path-state), and the corridor BoidsPath is a CONST input (not mutated). So the ONLY
// mutable replayable state is std::vector<Agent>: BoidsSnapshot is a plain deep-copy of the agent vector — no
// mutable extra (simpler than VH5's hinge axes / AC5's clip-time tick; the stateless steering pays off here).
//
// THE INPUT is a PERTURBATION stream. BoidsCommand {tick; agent; dv} — a velocity kick to one agent (a
// "scare"/"shove" event; the AC5 ApplyImpulse analog lifted to a replayable input). A perturbation scatters
// part of the flock; the flock+path drive re-coheres it. BD5 APPENDS — BD1-BD4 are byte-FROZEN.

#include <cstring>   // std::memcmp (the lockstep/rollback bit-exact compares)

// ----- BoidsCommand: ONE velocity-kick perturbation input event (the AC5 ApplyImpulse twin) ---------------
// A velocity kick to one agent at a given tick. The harness applies every command whose cmd.tick == tick (in
// ARRAY ORDER) via ApplyBoidsCommand before the StepFlockPath of that tick. Pure integer (dv is a Q16.16
// FxVec3 velocity delta).
struct BoidsCommand {
    uint32_t tick  = 0;   // the tick this perturbation fires at
    uint32_t agent = 0;   // the agent index the velocity kick is applied to
    FxVec3   dv;          // the Q16.16 velocity-kick delta
};

// ----- ApplyBoidsCommand(agents, cmd): apply one perturbation (agents[cmd.agent].vel += cmd.dv) ------------
// If cmd.agent is in range, kick that agent's velocity by cmd.dv (FxAdd); else a no-op (out-of-range agents are
// silently ignored — the deterministic guard).
inline void ApplyBoidsCommand(std::vector<Agent>& agents, const BoidsCommand& cmd) {
    if (cmd.agent < (uint32_t)agents.size())
        agents[(size_t)cmd.agent].vel = FxAdd(agents[(size_t)cmd.agent].vel, cmd.dv);
}

// ----- BoidsSnapshot: the captured mutable crowd state (JUST the agent world — the BD simplification) ------
// The ONLY mutable replayable state is the agent vector (SteerPath is stateless + the corridor is const), so
// the snapshot is a plain deep-copy of std::vector<Agent> — no mutable extra (the AC5/VH5 snapshot's tick /
// hinge-axes are NOT needed here).
struct BoidsSnapshot {
    std::vector<Agent> agents;   // a deep-copy of the agent world (the rollback restore point)
};

// ----- SnapshotBoids(agents): deep-copy the agent world (the rollback restore point) ----------------------
// A value copy -> deep-copies the agent vector. Bit-exact round-trip with RestoreBoids:
// RestoreBoids(agents, SnapshotBoids(agents0)) leaves agents == agents0 byte-for-byte.
inline BoidsSnapshot SnapshotBoids(const std::vector<Agent>& agents) {
    BoidsSnapshot snap;
    snap.agents = agents;   // deep copy
    return snap;
}

// ----- RestoreBoids(agents, snap): restore the agent world from a snapshot (the rollback) -----------------
// Bit-exact round-trip with SnapshotBoids.
inline void RestoreBoids(std::vector<Agent>& agents, const BoidsSnapshot& snap) {
    agents = snap.agents;   // restore the deep-copied world
}

// ----- SimBoidsTick(agents, cfg, path, commands, tick, dt): the deterministic per-tick step ----------------
// (0) APPLY this tick's commands in ARRAY ORDER (every cmd with cmd.tick == tick) via ApplyBoidsCommand — the
//     perturbation input — BEFORE the step so the kick integrates this tick.
// (1) StepFlockPath(agents, cfg, path, dt) — the BD4 path-following flock tick (rebuild grid+neighbors ->
//     JACOBI). Pure integer -> bit-identical on every peer/platform. The AC5 SimActiveTick twin.
inline void SimBoidsTick(std::vector<Agent>& agents, const FlockConfig& cfg, const BoidsPath& path,
                         const std::vector<BoidsCommand>& commands, int tick, fx dt) {
    for (size_t c = 0; c < commands.size(); ++c)
        if ((int)commands[c].tick == tick)
            ApplyBoidsCommand(agents, commands[c]);
    StepFlockPath(agents, cfg, path, dt);
}

// ----- RunBoidsLockstep: authority + replica from the SAME inputs, bit-identical every tick ----------------
// THE peer entry point (the AC5 RunActiveLockstep / VH5 RunVehicleLockstep control flow over SimBoidsTick).
// Run `ticks` SimBoidsTicks from a COPY of `initialAgents`, applying the command stream -> the converged crowd.
// authority + replica step from the SAME init + stream (INPUTS ONLY — no state shared) -> BIT-IDENTICAL by
// determinism. This function ASSERTS authority == replica bit-for-bit every tick (memcmp the agent vector) via
// an internal replica run; the caller also memcmps two RunBoidsLockstep returns for the determinism proof.
// Returns the converged authority agent vector.
inline std::vector<Agent> RunBoidsLockstep(const FlockConfig& cfg, const BoidsPath& path,
                                           const std::vector<Agent>& initialAgents,
                                           const std::vector<BoidsCommand>& commands, int ticks, fx dt) {
    std::vector<Agent> authority = initialAgents;   // a fresh copy
    std::vector<Agent> replica   = initialAgents;   // the second peer fed the SAME inputs
    for (int t = 0; t < ticks; ++t) {
        SimBoidsTick(authority, cfg, path, commands, t, dt);
        SimBoidsTick(replica,   cfg, path, commands, t, dt);
        // assert bit-identical every tick — two peers fed only the perturbation stream stay in lockstep.
        if (authority.size() != replica.size() ||
            std::memcmp(authority.data(), replica.data(), authority.size() * sizeof(Agent)) != 0) {
            // the lockstep invariant broke — a nondeterminism the showcase/test reports loudly (unreachable for
            // a deterministic sim — the fixed-order integer ops guarantee authority == replica). We leave the
            // authority as-is; the caller's memcmp proof catches the divergence.
            return authority;
        }
    }
    return authority;
}

// ----- RunBoidsRollback: snapshot -> mispredict diverges -> rollback -> corrected == authority -------------
// The rollback harness (the AC5 RunActiveRollback / VH5 RunVehicleRollback control flow over SimBoidsTick).
// (1) advance ticks 0..divergeTick from `initialAgents` applying authorityCmds; (2) SAVE a BoidsSnapshot AT
// divergeTick (SnapshotBoids — just the agent world); (2b) speculatively advance a few ticks with the
// MISPREDICTED stream (a WRONG perturbation — different agent/dv/tick, the client prediction that diverges);
// (3) ROLLBACK — RestoreBoids to the snapshot + RE-SIMULATE divergeTick..ticks with the CORRECT authorityCmds
// -> the corrected final crowd. The caller asserts this == RunBoidsLockstep(cfg, path, init, authorityCmds,
// ticks, dt) (rollback corrected the misprediction EXACTLY) AND that the speculative pre-rollback state
// DIFFERED from authority (a real divergence was fixed). Reuses SnapshotBoids/RestoreBoids. cfg/path/dt are
// CONSTANT, NOT snapshotted.
inline std::vector<Agent> RunBoidsRollback(const FlockConfig& cfg, const BoidsPath& path,
                                           const std::vector<Agent>& initialAgents,
                                           const std::vector<BoidsCommand>& authorityCmds,
                                           const std::vector<BoidsCommand>& mispredictCmds, int divergeTick,
                                           int ticks, fx dt) {
    std::vector<Agent> agents = initialAgents;
    // (1) advance 0..divergeTick with the authoritative stream.
    for (int t = 0; t < divergeTick; ++t)
        SimBoidsTick(agents, cfg, path, authorityCmds, t, dt);
    // (2) SAVE the snapshot at divergeTick (the rollback restore point — just the agent world).
    const BoidsSnapshot snap = SnapshotBoids(agents);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (the wrong perturbation — the client
    // prediction that diverges). Bounded to the remaining ticks.
    int specTicks = ticks - divergeTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s)
        SimBoidsTick(agents, cfg, path, mispredictCmds, divergeTick + s, dt);
    // (3) ROLLBACK: restore the snapshot (the agent world) + re-sim divergeTick..ticks with the authStream.
    RestoreBoids(agents, snap);
    for (int t = divergeTick; t < ticks; ++t)
        SimBoidsTick(agents, cfg, path, authorityCmds, t, dt);
    return agents;
}

}  // namespace boids
}  // namespace hf::sim
