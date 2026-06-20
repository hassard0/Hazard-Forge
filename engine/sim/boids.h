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

}  // namespace boids
}  // namespace hf::sim
