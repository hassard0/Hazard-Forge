// Slice MM1 — DETERMINISTIC MOTION MATCHING (engine/anim/motion_match.h, Track-S S4 of
// docs/SUPERIORITY_ROADMAP.md, the long-queued flagship #33): a Q16.16 pose FEATURE DATABASE
// extracted from animation clips (the ONE documented float->integer quantize boundary) + an integer
// weighted-L1 nearest-neighbor search with the (cost, index) tie-break + a search-interval
// controller + CL5/FPX5-mold lockstep/rollback — replayable motion-matched locomotion (UE5's motion
// matching is float/non-replayable). Namespace hf::anim::mm, header-only, pure CPU.
//
// What this test PINS:
//   * DB BUILD DETERMINISM: two builds byte-identical; DigestMotionDatabase PINNED (must be
//     identical under MSVC and local clang — the cross-compiler proof; the fixture clips use exact
//     binary-fraction keys so the float sampling is EXACT); frame counts hand-derived (2.0 s @ 32
//     fps = ticks 0..64, minus the 30-tick lookahead margin -> 35 db frames/clip, 70 total).
//   * ANALYTIC FEATURES: walk root velocity == (0, 98304) exactly (1.5 u/s in Q16.16); walk
//     trajectory offsets == 30720/61440/92160 (1.5 * {10,20,30}/32); idle foot-bob velocity ==
//     +-4096 (0.0625 u/s) — hand-derived integers, not run-derived.
//   * QUERY: an exact-match query returns THAT frame at cost 0; a perturbed query (+4096 on the
//     left-foot-x dim) returns the nearest at the analytic cost 4096 * wFootPos = 268435456;
//     TIE-BREAK: two equidistant frames -> the LOWER index (a constructed synthetic case AND the
//     organic walk duplicate — the walk loop's ticks 0 and 32 are feature-identical).
//   * THE BEHAVIOR: idle->walk->idle under a scripted velocity stream — zero velocity holds idle,
//     forward velocity at tick 60 switches to walk AT tick 60 (a search tick), reversing at tick 150
//     switches back AT tick 150; switches == 2; the full selection-trace digest PINNED.
//   * SEARCH-INTERVAL SEMANTICS: a velocity change at tick 63 (mid-interval) does NOT switch until
//     the next search tick 70 (pinned).
//   * LOCKSTEP: replica == authority BIT-EXACT from inputs alone; rollback corrects a real
//     misprediction; SNAPSHOT COMPLETENESS controls — omitting the cursor OR the held input from a
//     snapshot diverges the replay (cursor and input ARE sim state).
//
// Pure C++ (hf_core), ASan-eligible like the other anim/sim tests.
#include "anim/motion_match.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace mm = hf::anim::mm;
using mm::fx;
using mm::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// The shared fixture: the 3-joint rig + {idle, walk} clips + the default 32-fps config (identical
// bits in the test and BOTH showcase harnesses — the header's fixture builders).
static mm::MotionDatabase BuildFixtureDb() {
    const anim::Skeleton sk = mm::MakeMMTestRig();
    const std::vector<anim::Animation> clips{mm::MakeMMIdleClip(), mm::MakeMMWalkClip()};
    return mm::BuildMotionDatabase(sk, clips, mm::MMConfig{});
}

// Resimulate [fromTick, toTicks) from `start` and return the (clip, cursor) trace digest — the
// snapshot-completeness harness (a corrupted snapshot must diverge THIS digest).
static uint64_t ResimTraceDigest(const mm::MotionDatabase& db, const mm::MMParams& prm,
                                 mm::MMState start, int fromTick, int toTick,
                                 const std::vector<mm::MMCommand>& stream) {
    std::vector<mm::MMTraceEntry> trace;
    for (int t = fromTick; t < toTick; ++t) {
        mm::SimMMTick(start, db, prm, stream, (uint32_t)t);
        trace.push_back(mm::MMTraceEntry{start.clip, start.cursor});
    }
    return mm::DigestMMTrace(trace);
}

int main() {
    HF_TEST_MAIN_INIT();

    const int kWalkVelZ = 98304;   // 1.5 u/s in Q16.16 (exact: 1.5 * 65536)

    // ================= (a) DB BUILD: determinism + pinned digest + hand-derived structure ============
    {
        const mm::MotionDatabase db  = BuildFixtureDb();
        const mm::MotionDatabase db2 = BuildFixtureDb();

        // Structure: 2.0 s @ 32 fps -> sample ticks 0..64; lookahead margin 30 -> 35 frames/clip.
        check(db.clips.size() == 2, "db: two clips");
        check(db.clips[0].firstFrame == 0 && db.clips[0].frameCount == 35,
              "db: idle range [0, 35) (65 sample ticks minus the 30-tick lookahead margin)");
        check(db.clips[1].firstFrame == 35 && db.clips[1].frameCount == 35,
              "db: walk range [35, 70)");
        check(db.frames.size() == 70, "db: 70 frames total");

        // Two builds byte-identical (MotionFrame is a padding-free int32 POD).
        check(db2.frames.size() == db.frames.size() &&
              std::memcmp(db.frames.data(), db2.frames.data(),
                          db.frames.size() * sizeof(mm::MotionFrame)) == 0 &&
              db2.clips.size() == db.clips.size() &&
              std::memcmp(db.clips.data(), db2.clips.data(),
                          db.clips.size() * sizeof(mm::MotionClipRange)) == 0,
              "db: two builds byte-identical");

        // ANALYTIC feature pins (hand-derived, not run-derived).
        const mm::MotionFrame& w0 = db.frames[35];   // walk tick 0
        check(w0.clip == 1 && w0.tick == 0, "db: frame 35 is walk tick 0");
        check(w0.feature[mm::kFRootVel + 0] == 0 && w0.feature[mm::kFRootVel + 1] == kWalkVelZ,
              "db analytic: walk root velocity == (0, 98304) exactly (1.5 u/s)");
        check(w0.feature[mm::kFTraj + 1] == 30720 && w0.feature[mm::kFTraj + 3] == 61440 &&
              w0.feature[mm::kFTraj + 5] == 92160,
              "db analytic: walk trajectory z offsets == 1.5 * {10,20,30}/32 exactly");
        const mm::MotionFrame& i0 = db.frames[0];    // idle tick 0
        check(i0.clip == 0 && i0.tick == 0, "db: frame 0 is idle tick 0");
        check(i0.feature[mm::kFRootVel + 0] == 0 && i0.feature[mm::kFRootVel + 1] == 0 &&
              i0.feature[mm::kFTraj + 0] == 0 && i0.feature[mm::kFTraj + 5] == 0,
              "db analytic: idle root velocity + trajectory == 0");
        check(i0.feature[mm::kFFootPos + 0] == -16384 && i0.feature[mm::kFFootPos + 3] == 16384,
              "db analytic: idle foot x == -+0.25 (root-relative rest offsets)");
        check(i0.feature[mm::kFFootVel + 1] == 4096 && i0.feature[mm::kFFootVel + 4] == -4096,
              "db analytic: idle foot-bob y velocity == +-4096 (0.0625 u/s, antiphase)");

        // The pinned digest (the golden currency — identical MSVC + clang).
        const uint64_t d = mm::DigestMotionDatabase(db);
        std::printf("MM1 pin: DigestMotionDatabase = 0x%016llx (frames %d)\n",
                    (unsigned long long)d, (int)db.frames.size());
        check(d == 0xeeb343f0795a497dull, "MM1 pin: db digest == the pinned value");
        check(mm::DigestMotionDatabase(db2) == d, "db: second build digest identical");
    }

    // ================= (b) QUERY: exact match, perturbed nearest, tie-breaks ==========================
    {
        const mm::MotionDatabase db = BuildFixtureDb();

        // Exact-match query: idle frame 7's feature row -> THAT frame at cost 0.
        {
            mm::MatchResult r = mm::MatchPose(db, db.frames[7].feature);
            check(r.frameIndex == 7 && r.cost == 0 && r.clip == 0 && r.tick == 7,
                  "query: an exact-match query returns THAT frame at cost 0");
        }
        // Perturbed query: +4096 on the left-foot-x dim (constant -0.25 across ALL frames, so frame
        // 7 stays the unique nearest). Analytic cost = 4096 * wFootPos(kOne) = 268435456.
        {
            fx q[mm::kFeatureDim];
            std::memcpy(q, db.frames[7].feature, sizeof(q));
            q[mm::kFFootPos + 0] += 4096;
            mm::MatchResult r = mm::MatchPose(db, q);
            check(r.frameIndex == 7, "query: the perturbed query still returns the nearest frame 7");
            check(r.cost == 268435456ll,
                  "query: perturbed cost == 4096 * wFootPos == 268435456 (analytic)");
        }
        // TIE-BREAK (constructed): a synthetic 2-frame db whose frames are equidistant from the
        // query (+kOne vs -kOne on one dim) -> the LOWER index wins.
        {
            mm::MotionDatabase tiny;
            tiny.clips.push_back(mm::MotionClipRange{0, 2});
            for (int k = 0; k < mm::kFeatureDim; ++k) tiny.weight[k] = kOne;
            mm::MotionFrame fa; fa.clip = 0; fa.tick = 0; fa.feature[mm::kFFootPos + 0] = +kOne;
            mm::MotionFrame fb; fb.clip = 0; fb.tick = 1; fb.feature[mm::kFFootPos + 0] = -kOne;
            tiny.frames = {fa, fb};
            fx q[mm::kFeatureDim] = {};                       // equidistant (kOne each way)
            mm::MatchResult r = mm::MatchPose(tiny, q);
            const int64_t costA = mm::MatchCost(tiny, fa, q);
            const int64_t costB = mm::MatchCost(tiny, fb, q);
            check(costA == costB && costA > 0, "tie-break: the two frames really are equidistant");
            check(r.frameIndex == 0 && r.tick == 0,
                  "tie-break: equal cost -> the LOWER index wins (the (cost,index) total order)");
        }
        // TIE-BREAK (organic): the walk loop has a 32-tick foot period, so walk ticks 0 and 32 are
        // feature-IDENTICAL rows; querying that row returns the FIRST occurrence (frame 35, tick 0).
        {
            check(std::memcmp(db.frames[35].feature, db.frames[67].feature,
                              sizeof(fx) * mm::kFeatureDim) == 0,
                  "tie-break organic: walk ticks 0 and 32 are feature-identical (the loop period)");
            mm::MatchResult r = mm::MatchPose(db, db.frames[67].feature);
            check(r.frameIndex == 35 && r.tick == 0,
                  "tie-break organic: the duplicate row resolves to the first occurrence");
        }
        // Empty database: MatchPose returns frameIndex -1 (deterministic no-result).
        {
            mm::MotionDatabase empty;
            fx q[mm::kFeatureDim] = {};
            check(mm::MatchPose(empty, q).frameIndex == -1, "query: empty db -> frameIndex -1");
        }
    }

    // ================= (c) THE BEHAVIOR: idle -> walk -> idle under the scripted stick ================
    // Zero velocity for ticks [0,60), forward 1.5 u/s for [60,150), zero again for [150,240).
    // 60 and 150 are search ticks (N=10) -> the switches land EXACTLY there. Trace digest pinned.
    {
        const mm::MotionDatabase db = BuildFixtureDb();
        const mm::MMParams prm;                                  // searchInterval = 10
        const std::vector<mm::MMCommand> stream{
            mm::MMCommand{60,  mm::kMMCmdSetVelocity, 0, kWalkVelZ},
            mm::MMCommand{150, mm::kMMCmdSetVelocity, 0, 0},
        };
        std::vector<mm::MMTraceEntry> trace;
        const mm::MMState fin = mm::RunMMLockstep(db, prm, mm::MMState{}, stream, 240, &trace);

        check(trace.size() == 240, "behavior: 240 trace entries");
        // Idle phase: every selection in [0,60) is the idle clip.
        bool idleHeld = true;
        for (int t = 0; t < 60; ++t) if (trace[(size_t)t].clip != 0) idleHeld = false;
        check(idleHeld, "behavior: zero velocity holds the IDLE clip for all of [0,60)");
        // The switch to walk lands exactly at tick 60.
        check(trace[59].clip == 0 && trace[60].clip == 1,
              "behavior: forward velocity switches to WALK exactly at search tick 60");
        bool walkHeld = true;
        for (int t = 60; t < 150; ++t) if (trace[(size_t)t].clip != 1) walkHeld = false;
        check(walkHeld, "behavior: the WALK clip holds for all of [60,150)");
        // Reversing lands exactly at tick 150.
        check(trace[149].clip == 1 && trace[150].clip == 0,
              "behavior: reversing to zero switches back to IDLE exactly at search tick 150");
        bool idleHeld2 = true;
        for (int t = 150; t < 240; ++t) if (trace[(size_t)t].clip != 0) idleHeld2 = false;
        check(idleHeld2, "behavior: IDLE holds for all of [150,240)");

        check(fin.switches == 2 && fin.lastSwitchTick == 150,
              "behavior: exactly 2 clip switches, the last at tick 150");
        check(fin.searches == 24, "behavior: 240 ticks / N=10 -> 24 searches");

        const uint64_t td = mm::DigestMMTrace(trace);
        const uint64_t sd = mm::MMStateDigest(fin);
        std::printf("MM1 pin: behavior trace digest = 0x%016llx  final state digest = 0x%016llx  "
                    "(switches %u at ticks 60/%d)\n",
                    (unsigned long long)td, (unsigned long long)sd, fin.switches,
                    fin.lastSwitchTick);
        check(td == 0xb37aa47b7bc7d69bull, "MM1 pin: behavior trace digest == the pinned value");
        check(sd == 0xb58696a56b71ec9aull, "MM1 pin: final state digest == the pinned value");

        // Root-motion coherence: MMRootVelocity reports the walk speed during walk, zero during idle.
        mm::MMState probe;
        fx vx = -1, vz = -1;
        std::vector<mm::MMTraceEntry> t2;
        mm::MMState mid = mm::RunMMLockstep(db, prm, mm::MMState{}, stream, 100, &t2);
        mm::MMRootVelocity(db, mid, vx, vz);
        check(vx == 0 && vz == kWalkVelZ,
              "behavior: MMRootVelocity during the walk phase == (0, 98304)");
        probe = mm::RunMMLockstep(db, prm, mm::MMState{}, stream, 40, &t2);
        mm::MMRootVelocity(db, probe, vx, vz);
        check(vx == 0 && vz == 0, "behavior: MMRootVelocity during the idle phase == (0, 0)");
    }

    // ================= (d) SEARCH-INTERVAL SEMANTICS: mid-interval input waits for the search =========
    {
        const mm::MotionDatabase db = BuildFixtureDb();
        const mm::MMParams prm;
        const std::vector<mm::MMCommand> stream{
            mm::MMCommand{63, mm::kMMCmdSetVelocity, 0, kWalkVelZ},   // mid-interval (63 % 10 != 0)
        };
        std::vector<mm::MMTraceEntry> trace;
        const mm::MMState fin = mm::RunMMLockstep(db, prm, mm::MMState{}, stream, 100, &trace);
        bool heldUntil70 = true;
        for (int t = 0; t < 70; ++t) if (trace[(size_t)t].clip != 0) heldUntil70 = false;
        check(heldUntil70,
              "interval: a velocity change at tick 63 does NOT switch before the next search tick");
        check(trace[70].clip == 1 && fin.lastSwitchTick == 70 && fin.switches == 1,
              "interval: the switch lands EXACTLY at search tick 70 (pinned)");
        // Clip changes only ever happen on search ticks (t % 10 == 0) — sweep the whole trace.
        bool onlyOnSearch = true;
        for (size_t t = 1; t < trace.size(); ++t)
            if (trace[t].clip != trace[t - 1].clip && (t % 10u) != 0u) onlyOnSearch = false;
        check(onlyOnSearch, "interval: selections change ONLY on search ticks");
    }

    // ================= (e) LOCKSTEP: replica == authority; rollback; snapshot completeness ============
    {
        const mm::MotionDatabase db = BuildFixtureDb();
        const mm::MMParams prm;
        const std::vector<mm::MMCommand> authStream{
            mm::MMCommand{60,  mm::kMMCmdSetVelocity, 0, kWalkVelZ},
            mm::MMCommand{150, mm::kMMCmdSetVelocity, 0, 0},
        };
        const int ticks = 240, mispredictTick = 100;

        std::vector<mm::MMTraceEntry> traceA, traceB;
        const mm::MMState authority = mm::RunMMLockstep(db, prm, mm::MMState{}, authStream, ticks,
                                                        &traceA);
        const mm::MMState replica = mm::RunMMLockstep(db, prm, mm::MMState{}, authStream, ticks,
                                                      &traceB);
        check(std::memcmp(&authority, &replica, sizeof(mm::MMState)) == 0 &&
              mm::DigestMMTrace(traceA) == mm::DigestMMTrace(traceB),
              "lockstep: replica == authority BIT-EXACT (inputs-only re-sim, full trace)");

        // Rollback: the client mispredicts a stop at tick 100 (a real divergence — the search at 100
        // would switch to idle); rollback restores + re-sims to authority BIT-EXACT.
        std::vector<mm::MMCommand> mispredictStream = authStream;
        mispredictStream.push_back(mm::MMCommand{(uint32_t)mispredictTick,
                                                 mm::kMMCmdSetVelocity, 0, 0});
        const mm::MMState rolledBack = mm::RunMMRollback(db, prm, mm::MMState{}, authStream,
                                                         mispredictStream, ticks, mispredictTick);
        check(std::memcmp(&rolledBack, &authority, sizeof(mm::MMState)) == 0,
              "rollback: corrected to authority BIT-EXACT (positive control)");
        const mm::MMState mispredicted = mm::RunMMLockstep(db, prm, mm::MMState{}, mispredictStream,
                                                           ticks);
        check(std::memcmp(&mispredicted, &authority, sizeof(mm::MMState)) != 0,
              "rollback: the mispredicted run DIFFERS from authority (the divergence was real)");

        // SNAPSHOT COMPLETENESS: advance to tick 95 (mid-walk), snapshot, resim [95,240) three ways.
        mm::MMState at95;
        for (int t = 0; t < 95; ++t) mm::SimMMTick(at95, db, prm, authStream, (uint32_t)t);
        const uint64_t goodTail = ResimTraceDigest(db, prm, at95, 95, ticks, authStream);
        // (i) The complete snapshot reproduces the authority tail.
        std::vector<mm::MMTraceEntry> authTail(traceA.begin() + 95, traceA.end());
        check(goodTail == mm::DigestMMTrace(authTail),
              "snapshot: the COMPLETE snapshot resim reproduces the authority tail bit-for-bit");
        // (ii) Omitting the CURSOR diverges (cursor IS sim state).
        mm::MMState noCursor = at95;
        noCursor.cursor = (noCursor.cursor + 17) % 35;
        check(ResimTraceDigest(db, prm, noCursor, 95, ticks, authStream) != goodTail,
              "snapshot: omitting/corrupting the CURSOR diverges the replay (completeness control)");
        // (iii) Omitting the held INPUT diverges (the held stick IS sim state).
        mm::MMState noInput = at95;
        noInput.input = mm::MMInput{};
        check(ResimTraceDigest(db, prm, noInput, 95, ticks, authStream) != goodTail,
              "snapshot: omitting the held INPUT diverges the replay (completeness control)");

        const uint64_t d = mm::MMStateDigest(authority);
        std::printf("MM1 pin: lockstep authority state digest = 0x%016llx\n", (unsigned long long)d);
        check(d == 0xb58696a56b71ec9aull,
              "MM1 pin: lockstep authority digest == the pinned value (== the behavior run)");
    }

    if (g_fail == 0) std::printf("motion_match_test: ALL PASS\n");
    else std::printf("motion_match_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
