// Slice PA1 — PARTICLE-SYSTEM AUTHORING VIA THE FLOW VM (engine/sim/particle_author.h, Track-S S11):
// the flow VM (engine/flow/flow.h) drives the deterministic particle system (engine/sim/particles.h) —
// a flow graph's per-tick output registers overwrite bound emitter/force-field parameters, so particle
// effects are AUTHORED as graphs (via the editor edit-ops), not hand-written C++ structs. Pure CPU
// (hf_core), ASan-eligible like the other sim tests.
//
// PREMISE (verified before this slice): NO flow->particles bridge existed anywhere in the tree (grep for
// particle_author / pauthor / AuthoredEffect / ParamBinding: zero hits; engine/sim/* had no flow.h include).
// particles.h and flow.h are byte-UNTOUCHED by PA1 (this test + particles_test + flow_test all stay green).
//
// What this test PINS (the PA1 contracts):
//   (b) IDENTITY: an effect with ZERO bindings == StepParticles with the base config BIT-IDENTICAL.
//   (c) THE AUTHORING PROOF: the edit-ops-built pulsing-fountain graph's SerializeGraph digest is pinned
//       (authored, not hardcoded); over kShowcaseSteps ticks the resolved params show the PULSE (spawn
//       count == kBurstSpawn exactly on ticks t%6==5, kBaseSpawn otherwise), the OSCILLATION (emitter X is
//       a period-16 triangle wave hitting exactly +-1.0), and the RAMP (vortex strength == (t+1)*kRampStep
//       exactly; late-tick tangential speed exceeds early — both metrics pinned); the per-tick alive-count
//       trace digest and the final composed-state digest are pinned.
//   (d) BINDING EXACTNESS: a kConst-driven binding sets the parameter EXACTLY (the fx value seen by the
//       emitter is the register bit-for-bit — pinned).
//   (e) DETERMINISM + LOCKSTEP: two runs bit-identical; a replica peer fed ONLY (asset, init, command
//       stream) re-derives the COMPOSED state (graph state + pool) bit-for-bit; rollback corrects a real
//       misprediction to the authority; the snapshot-completeness control (omit the flow GraphState) DIVERGES.
// All pins are pure-integer digests/values — identical MSVC vs clang (the flow_test/particles_test bar).
#include "sim/particle_author.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace pt = hf::sim::particles;
namespace pa = hf::sim::pauthor;
using pa::fx;
using pa::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Bit-exact compare of two composed snapshots (pool bytes + free-list + cursor + tick + flow register file).
static bool SnapEqual(const pa::AuthoredSnapshot& a, const pa::AuthoredSnapshot& b) {
    if (a.pool.particles.size() != b.pool.particles.size()) return false;
    if (!a.pool.particles.empty() &&
        std::memcmp(a.pool.particles.data(), b.pool.particles.data(),
                    a.pool.particles.size() * sizeof(pt::FxParticle)) != 0) return false;
    if (a.pool.freeList != b.pool.freeList) return false;
    if (a.pool.spawnCursor != b.pool.spawnCursor) return false;
    if (a.pool.tick != b.pool.tick) return false;
    if (a.flowState.prev != b.flowState.prev) return false;
    return true;
}

int main() {
    HF_TEST_MAIN_INIT();

    const fx kDt = kOne / 60;
    const std::vector<hf::flow::Reg> kNoInput = {0};   // one zeroed input channel (the boost idle)

    // ================= (b) IDENTITY: ZERO bindings == StepParticles with the base config ==================
    {
        pa::AuthoredEffect e = pa::MakePulsingFountainEffect();
        e.bindings.clear();                                  // ZERO bindings — the identity control
        pt::ParticlePool authored = pt::InitParticlePool(pa::kShowcaseCapacity);
        pt::ParticlePool direct   = pt::InitParticlePool(pa::kShowcaseCapacity);
        const int K = 100;
        for (int t = 0; t < K; ++t) {
            pa::StepAuthoredEffect(e, authored, kDt, authored.tick, kNoInput);
            pt::StepParticles(direct, e.baseCfg, e.baseFields.data(), (uint32_t)e.baseFields.size(),
                              e.baseGravity, e.baseDragK, kDt, e.groundY, e.radius, e.restitution,
                              nullptr, 0u);
        }
        check(std::memcmp(authored.particles.data(), direct.particles.data(),
                          authored.particles.size() * sizeof(pt::FxParticle)) == 0 &&
              authored.freeList == direct.freeList && authored.spawnCursor == direct.spawnCursor &&
              authored.tick == direct.tick,
              "PA1 identity: ZERO bindings == StepParticles(baseCfg) BIT-IDENTICAL (the identity control)");
    }

    // ================= (c) THE AUTHORING PROOF: the edit-ops-built pulsing fountain =======================
    {
        pa::AuthoredEffect e = pa::MakePulsingFountainEffect();

        // --- the pinned STATIC graph digest: the graph was AUTHORED via AddFlowNode/ConnectFlow — any
        // drift in the authoring calls (or the edit-ops themselves) changes SerializeGraph's bytes. ---
        const uint64_t graphDigest = pa::DigestAuthoredGraph(e.graph);
        std::printf("pa1: pulsing-fountain graph nodes=%u bindings=%u digest=0x%016llx\n",
                    (unsigned)e.graph.nodes.size(), (unsigned)e.bindings.size(),
                    (unsigned long long)graphDigest);
        const uint64_t kPinnedGraphDigest = 0xca55eabb27042ba8ull;  // PINNED (MSVC == clang)
        check(graphDigest == kPinnedGraphDigest,
              "PA1 authoring: edit-ops-built graph digest == pinned (authored, not hardcoded)");
        check(e.graph.nodes.size() == 24, "PA1 authoring: the fixture graph has the fixed 24 nodes");

        // --- run the showcase: capture the per-tick resolved params + alive counts + the ramp metric ---
        pt::ParticlePool pool = pt::InitParticlePool(pa::kShowcaseCapacity);
        std::vector<uint32_t> aliveTrace;
        aliveTrace.reserve(pa::kShowcaseSteps);
        bool pulseOk = true, oscOk = true, rampExactOk = true, oscVaries = false;
        fx oscMin = 0, oscMax = 0;
        std::vector<fx> oscTrace;
        oscTrace.reserve(pa::kShowcaseSteps);
        int64_t earlyTangential = -1, lateTangential = -1;
        const uint32_t kEarlyTick = 24, kLateTick = 120;
        for (uint32_t t = 0; t < pa::kShowcaseSteps; ++t) {
            pa::ResolvedParams rp;
            pa::StepAuthoredEffect(e, pool, kDt, pool.tick, kNoInput, &rp);
            // PULSE: spawn count == kBurstSpawn exactly on ticks t%6==5, kBaseSpawn otherwise (input 0).
            const fx expectedSpawn = (t % (uint32_t)pa::kPulsePeriod == (uint32_t)(pa::kPulsePeriod - 1))
                                         ? (fx)pa::kBurstSpawn : (fx)pa::kBaseSpawn;
            if (rp.cfg.ratePerTick != expectedSpawn) pulseOk = false;
            // OSC: emitter X is a triangle wave: ocycle=(t+1)%16, tri=min(o,16-o), x=(tri-4)*kOscScale.
            const int o = (int)((t + 1u) % (uint32_t)pa::kOscPeriod);
            const int tri = (o < pa::kOscPeriod - o) ? o : (pa::kOscPeriod - o);
            const fx expectedX = (fx)((tri - pa::kOscPeriod / 4) * pa::kOscScale);
            if (rp.cfg.origin.x != expectedX) oscOk = false;
            if (rp.cfg.origin.x < oscMin) oscMin = rp.cfg.origin.x;
            if (rp.cfg.origin.x > oscMax) oscMax = rp.cfg.origin.x;
            if (rp.cfg.origin.x != e.baseCfg.origin.x) oscVaries = true;
            oscTrace.push_back(rp.cfg.origin.x);
            // RAMP: vortex strength == (t+1)*kRampStep EXACTLY (the kCounter accumulation).
            if (rp.fields.size() != 1 || rp.fields[0].strength != (fx)((int64_t)(t + 1) * pa::kRampStep))
                rampExactOk = false;
            aliveTrace.push_back(pt::CountAlive(pool));
            // The ramp EFFECT metric: total tangential speed sum |vel.x|+|vel.z| over alive particles.
            if (t == kEarlyTick || t == kLateTick) {
                int64_t sum = 0;
                for (const pt::FxParticle& p : pool.particles) {
                    if (!(p.flags & pt::kFlagAlive)) continue;
                    sum += (p.vel.x < 0 ? -(int64_t)p.vel.x : (int64_t)p.vel.x);
                    sum += (p.vel.z < 0 ? -(int64_t)p.vel.z : (int64_t)p.vel.z);
                }
                if (t == kEarlyTick) earlyTangential = sum; else lateTangential = sum;
            }
        }
        check(pulseOk, "PA1 pulse: spawn count == kBurstSpawn exactly on ticks t%6==5, kBaseSpawn otherwise");
        check(oscOk, "PA1 oscillation: emitter X == the exact period-16 triangle wave every tick");
        check(oscVaries && oscMin == -kOne && oscMax == kOne,
              "PA1 oscillation: emitter X sweeps exactly [-1.0, +1.0] Q16.16 (varies, hits both extremes)");
        // Periodicity: origin.x(t) == origin.x(t+16) across the run.
        {
            bool periodOk = true;
            for (uint32_t t = 0; t + (uint32_t)pa::kOscPeriod < pa::kShowcaseSteps; ++t)
                if (oscTrace[t] != oscTrace[t + (uint32_t)pa::kOscPeriod]) { periodOk = false; break; }
            check(periodOk, "PA1 oscillation: emitter X has exact period kOscPeriod (16 ticks)");
        }
        check(rampExactOk, "PA1 ramp: vortex strength == (t+1)*kRampStep exactly every tick (the kCounter)");
        std::printf("pa1: ramp tangential-speed metric early(t=%u)=%lld late(t=%u)=%lld\n",
                    kEarlyTick, (long long)earlyTangential, kLateTick, (long long)lateTangential);
        check(earlyTangential >= 0 && lateTangential > earlyTangential,
              "PA1 ramp: late-tick tangential speed exceeds early (the ramping vortex bites)");
        // The pinned ramp metrics (pure integer — identical MSVC/clang).
        const int64_t kPinnedEarlyTangential = 4603439;    // PINNED (MSVC == clang)
        const int64_t kPinnedLateTangential  = 15945193;   // PINNED (MSVC == clang) — ~3.5x early: the ramp bites
        check(earlyTangential == kPinnedEarlyTangential && lateTangential == kPinnedLateTangential,
              "PA1 ramp: early/late tangential metrics == pinned values");

        // --- the pinned alive-count-per-tick trace digest (the pulse structure as a byte-stable value) ---
        std::vector<uint8_t> traceBytes;
        traceBytes.reserve(aliveTrace.size() * 4u);
        for (const uint32_t a : aliveTrace) {
            traceBytes.push_back((uint8_t)(a & 0xFFu));
            traceBytes.push_back((uint8_t)((a >> 8) & 0xFFu));
            traceBytes.push_back((uint8_t)((a >> 16) & 0xFFu));
            traceBytes.push_back((uint8_t)((a >> 24) & 0xFFu));
        }
        const uint64_t aliveTraceDigest = hf::net::DigestBytes(traceBytes.data(), traceBytes.size());
        const uint64_t finalDigest = pa::DigestAuthored(pool, e.state);
        std::printf("pa1: alive-trace digest=0x%016llx final composed digest=0x%016llx alive=%u\n",
                    (unsigned long long)aliveTraceDigest, (unsigned long long)finalDigest,
                    pt::CountAlive(pool));
        const uint64_t kPinnedAliveTraceDigest = 0x1ffeca8ce3436077ull;  // PINNED (MSVC == clang)
        const uint64_t kPinnedFinalDigest      = 0x745c02a3574c687full;  // PINNED (MSVC == clang)
        check(aliveTraceDigest == kPinnedAliveTraceDigest,
              "PA1 authoring: alive-count-per-tick trace digest == pinned (the pulse structure)");
        check(finalDigest == kPinnedFinalDigest,
              "PA1 authoring: final composed-state digest == pinned (pool + graph state)");
        check(pt::CountAlive(pool) > 0, "PA1 authoring: the fountain is alive at capture (non-degenerate)");
    }

    // ================= (d) BINDING EXACTNESS: a kConst register IS the parameter, bit-for-bit =============
    {
        // A minimal edit-ops graph: one kConst carrying an arbitrary Q16.16 value.
        const fx kExactY = (fx)123456;                       // 1.8838... in Q16.16 — an arbitrary exact value
        pa::AuthoredEffect e;
        const hf::flow::NodeId nc = hf::editor::AddFlowNode(e.graph, hf::flow::kConst, (hf::flow::Reg)kExactY);
        e.state = hf::flow::MakeState(e.graph);
        e.bindings = { pa::ParamBinding{nc, pa::kParamEmitterY, 0u} };
        e.baseCfg.origin      = pa::FxVec3{0, 0, 0};
        e.baseCfg.ratePerTick = (fx)1;
        e.baseCfg.lifetime    = kOne * 100;
        e.baseCfg.speed       = 0;                           // zero initial speed — the spawn pos is untouched
        e.baseCfg.emitterId   = 1u;
        e.baseGravity = pa::FxVec3{0, 0, 0};                 // zero gravity/drag — pos.y stays the origin...
        e.baseDragK   = 0;
        e.groundY     = -kOne * 100;                         // ...well above the far-away ground
        pt::ParticlePool pool = pt::InitParticlePool(4);
        pa::ResolvedParams rp;
        pa::StepAuthoredEffect(e, pool, kDt, pool.tick, {}, &rp);
        check(rp.cfg.origin.y == kExactY,
              "PA1 binding exactness: the resolved emitter origin.y IS the kConst register (123456) bit-for-bit");
        check((pool.particles[0].flags & pt::kFlagAlive) && pool.particles[0].pos.y == kExactY,
              "PA1 binding exactness: the spawned particle's pos.y == 123456 exactly (the emitter saw the value)");

        // A raw-count binding: kConst 7 -> kParamSpawnPerTick spawns exactly 7 on tick 0.
        pa::AuthoredEffect e2;
        const hf::flow::NodeId n7 = hf::editor::AddFlowNode(e2.graph, hf::flow::kConst, 7);
        e2.state = hf::flow::MakeState(e2.graph);
        e2.bindings = { pa::ParamBinding{n7, pa::kParamSpawnPerTick, 0u} };
        e2.baseCfg = e.baseCfg;
        e2.baseGravity = e.baseGravity; e2.baseDragK = 0; e2.groundY = e.groundY;
        pt::ParticlePool pool2 = pt::InitParticlePool(16);
        pa::StepAuthoredEffect(e2, pool2, kDt, pool2.tick, {});
        check(pt::CountAlive(pool2) == 7,
              "PA1 binding exactness: kConst 7 -> kParamSpawnPerTick spawns exactly 7 (the raw-count convention)");
    }

    // ================= (e) DETERMINISM + LOCKSTEP + ROLLBACK + the completeness control ===================
    {
        const pa::AuthoredEffect asset = pa::MakePulsingFountainEffect();   // the static authored artifact
        pt::ParticlePool pool0 = pt::InitParticlePool(pa::kShowcaseCapacity);
        const pa::AuthoredSnapshot init = pa::SnapshotAuthored(asset, pool0);

        // The scripted command stream: the player boosts the spawn through the AUTHORED graph's kInput.
        const std::vector<pa::AuthoredCommand> authStream = {
            pa::AuthoredCommand{30,  0u, 6},    // +6 spawns on tick 30 (a player-triggered puff)
            pa::AuthoredCommand{70,  0u, 10},   // +10 on tick 70
            pa::AuthoredCommand{100, 0u, 3},    // +3 on tick 100
        };
        const uint32_t cc = (uint32_t)authStream.size();
        const uint32_t T = 200, rollbackAt = 64;
        std::vector<pa::AuthoredCommand> mispredictStream = authStream;
        mispredictStream.push_back(pa::AuthoredCommand{rollbackAt, 0u, 25});   // the WRONG predicted input

        const pa::AuthoredSnapshot authority = pa::RunAuthoredLockstep(
            asset, init, authStream.data(), cc, T, kDt, pa::kShowcaseChannels);
        const pa::AuthoredSnapshot replica = pa::RunAuthoredLockstep(
            asset, init, authStream.data(), cc, T, kDt, pa::kShowcaseChannels);
        check(SnapEqual(authority, replica),
              "PA1 lockstep: replica == authority BIT-EXACT (inputs-only re-derivation of graph state + pool)");

        pt::ParticlePool authorityPool;
        authorityPool.particles   = authority.pool.particles;
        authorityPool.freeList    = authority.pool.freeList;
        authorityPool.spawnCursor = authority.pool.spawnCursor;
        authorityPool.tick        = authority.pool.tick;
        check(pt::CountAlive(authorityPool) > 0,
              "PA1 lockstep: the authority pool is alive (non-degenerate)");

        const uint64_t lockstepDigest = pa::DigestAuthored(authorityPool, authority.flowState);
        std::printf("pa1: lockstep final composed digest=0x%016llx (T=%u, commands=%u)\n",
                    (unsigned long long)lockstepDigest, T, cc);
        const uint64_t kPinnedLockstepDigest = 0x2f2c7ba5d6ee3fccull;   // PINNED (MSVC == clang)
        check(lockstepDigest == kPinnedLockstepDigest,
              "PA1 lockstep: final composed digest == pinned (MSVC == clang)");

        // Rollback: corrected == authority AND the mispredicted state DIFFERED (real divergence fixed).
        const pa::AuthoredSnapshot rolledBack = pa::RunAuthoredRollback(
            asset, init, authStream.data(), cc, mispredictStream.data(), (uint32_t)mispredictStream.size(),
            T, rollbackAt, kDt, pa::kShowcaseChannels);
        const pa::AuthoredSnapshot mispredicted = pa::RunAuthoredLockstep(
            asset, init, mispredictStream.data(), (uint32_t)mispredictStream.size(), T, kDt,
            pa::kShowcaseChannels);
        check(SnapEqual(rolledBack, authority),
              "PA1 rollback: corrected == authority BIT-EXACT");
        check(!SnapEqual(mispredicted, authority),
              "PA1 rollback: the mispredicted state DIFFERED from authority (a real divergence was fixed)");

        // THE SNAPSHOT-COMPLETENESS CONTROL: restoring the pool but OMITTING the flow GraphState (a stale
        // fresh MakeState) resumes the pulse/osc/ramp from the WRONG phase -> DIVERGES from the full restore.
        {
            const uint32_t kMid = 80, kTail = 40;
            const pa::AuthoredSnapshot atMid = pa::RunAuthoredLockstep(
                asset, init, authStream.data(), cc, kMid, kDt, pa::kShowcaseChannels);
            // full restore + kTail ticks
            pa::AuthoredEffect eFull = asset;
            pt::ParticlePool poolFull;
            pa::RestoreAuthored(eFull, poolFull, atMid);
            for (uint32_t t = 0; t < kTail; ++t)
                pa::SimAuthoredTick(eFull, poolFull, kDt, authStream.data(), cc, pa::kShowcaseChannels);
            // INCOMPLETE restore: pool restored, flow state STALE (fresh zero state — the naive-snapshot bug)
            pa::AuthoredEffect eBad = asset;
            pt::ParticlePool poolBad;
            pa::RestoreAuthored(eBad, poolBad, atMid);
            eBad.state = hf::flow::MakeState(eBad.graph);     // OMIT the graph state (reset to zeros)
            for (uint32_t t = 0; t < kTail; ++t)
                pa::SimAuthoredTick(eBad, poolBad, kDt, authStream.data(), cc, pa::kShowcaseChannels);
            check(!pa::AuthoredStatesEqual(poolFull, eFull.state, poolBad, eBad.state),
                  "PA1 snapshot-completeness: omit the flow GraphState -> DIVERGES (graph state IS sim state)");
        }
    }

    if (g_fail == 0) std::printf("particle_author_test: ALL CHECKS PASSED\n");
    else std::printf("particle_author_test: %d CHECK(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
