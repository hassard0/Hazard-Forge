// Slice LA1 — LANDSCAPE AUTHORING (engine/terrain/terrain_author.h, hf::terrain::author): heightmap
// brush sculpting + splat-layer painting + spline-carved roads over the procedural terrain, every op
// pure integer, recorded/reversible via the TERRAIN-LOCAL history (the ED5 capture-before/apply/record
// recipe; edit_history.h byte-UNTOUCHED — see the header's enrollment-choice note). What this test PINS:
//   (a) SCULPT — the raise brush on flat terrain: center + flat-core + smoothstep-ramp heights EXACT
//       integers at sampled radii (designed-exact: fall kOne/2 at d = 3r/4 etc.); falloff monotone;
//       every outside-radius texel BIT-untouched. Flatten hits the target EXACTLY across the whole
//       flat core (d <= r/2) and leaves the outside bit-untouched. Smooth: K recorded passes -> pinned
//       digest; outside untouched.
//   (b) PAINT — the largest-remainder renorm: single-stamp exact weights ({0,255,0,0} in the core;
//       {128,127,0,0} at the half-falloff ring; restamp -> {1,254,0,0}); EVERY texel of a multi-paint
//       proc terrain sums to EXACTLY 255 (asserted globally); pinned painted-terrain digest.
//   (c) ROAD — a straight 2-point spline (the SP1 exact-lerp identity) over flat terrain: the carved
//       bed EXACTLY the spline height along/around the centerline, the road layer painted 255, the
//       shoulder blend MONOTONE and continuous at both band edges (smoothstep endpoints exact), the
//       off-band texels BIT-untouched, roadTexels pinned; plus the carve over the bumpy showcase
//       terrain pinned by digest (the staircase-on-slopes profile is the pinned real profile —
//       nearest-polyline-point height, documented honesty).
//   (d) UNDO — every Recorded op kind: op -> Undo -> DigestTerrain returns the pre-op digest BIT-EXACT
//       (the rect-memento proof); Redo re-applies to the post-op digest; branch-kill truncation on
//       record-after-undo; bitwise no-ops record nothing.
//   (e) DETERMINISM — the showcase scenario digest + raster digest pinned + two-run identical; every
//       pinned digest printed for the MSVC == clang cross-compiler comparison
//       (`clang++ -std=c++20 -I engine -I third_party tests/terrain_author_test.cpp` runs standalone).
// Pure C++ (hf_core), pure integer authoring, ASan-eligible; ZERO backend symbols. heightmap.h /
// procterrain.h / terrain_stream.h / spline.h / edit_history.h are composed READ-ONLY (byte-untouched).
#include "terrain/terrain_author.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace au = hf::terrain::author;
using au::fx;
using au::kOne;
using au::kFrac;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// ==== THE PINNED DIGESTS (MSVC == clang; bake once, then locked) ====================================
static const uint64_t kPinSeedDigest     = 0x0be3526b0c971ba8ull;  // the 128x128 fBm-seeded terrain
static const uint64_t kPinSmoothDigest   = 0x391004c59500b292ull;  // (a) after 4 recorded smooth passes
static const uint64_t kPinPaintDigest    = 0x048fc24a10d9c2c4ull;  // (b) the multi-paint proc terrain
static const uint64_t kPinRoadDigest     = 0x29c836ae634581ceull;  // (c) the bumpy-terrain carve
static const uint64_t kPinScenarioDigest = 0x7368c12f1f50e17cull;  // (e) the full showcase terrain
static const uint64_t kPinRasterDigest   = 0xf8ab0a02fd5196eaull;  // (e) the 512x512 BGRA raster
static const uint32_t kPinRoadTexels     = 365u;                   // (c) straight-road bed texel count
static const uint32_t kPinShowRoadTexels = 993u;                   // (e) showcase road bed texel count

int main() {
    HF_TEST_MAIN_INIT();

    // ================= (a) SCULPT — the raise brush on flat terrain ================================
    {
        au::AuthoredTerrain t = au::MakeFlatTerrain(64, 64, 0);
        const fx r = 8 * kOne, st = 4 * kOne;
        au::ApplyRaiseBrush(t, 32 * kOne, 32 * kOne, r, st);
        auto H = [&](int x, int z) { return t.heightQ[(size_t)z * 64 + (size_t)x]; };
        // The designed-exact falloff profile (center at texel (32,32), r = 8, flat core d <= 4):
        check(H(32, 32) == st, "(a) raise center == strength exactly (flat core)");
        check(H(36, 32) == st, "(a) raise at d = r/2 == strength exactly (core edge)");
        // d = 6 -> t = (8-6)/4 = kOne/2 -> smoothstep = kOne/2 -> exactly strength/2.
        check(H(38, 32) == 2 * kOne, "(a) raise at d = 6 == strength/2 exactly (smoothstep midpoint)");
        // d = 7 -> t = kOne/4 -> smoothstep = 10240/65536 -> fxmul(4*kOne, 10240) = 40960.
        check(H(39, 32) == 40960, "(a) raise at d = 7 == 40960 exactly (smoothstep quarter)");
        // Monotone non-increasing along the +x ray.
        bool mono = true;
        for (int x = 32; x < 41; ++x)
            if (H(x + 1, 32) > H(x, 32)) mono = false;
        check(mono, "(a) falloff monotone non-increasing");
        // EVERY texel strictly outside the radius is bit-untouched (== 0).
        bool outside = true;
        for (int z = 0; z < 64 && outside; ++z)
            for (int x = 0; x < 64; ++x) {
                const int64_t dx = x - 32, dz = z - 32;
                if (dx * dx + dz * dz > 64 && H(x, z) != 0) { outside = false; break; }
            }
        check(outside, "(a) raise: outside-radius texels bit-untouched");

        // FLATTEN over the bumpy proc terrain: the whole flat core hits the target EXACTLY.
        au::AuthoredTerrain b = au::MakeProcTerrain(64, 64, 16 * kOne, 4, 99u, 8 * kOne);
        const au::AuthoredTerrain before = b;
        const fx target = 5 * kOne;
        au::ApplyFlattenBrush(b, 32 * kOne, 32 * kOne, 12 * kOne, target);
        bool coreExact = true, flatOutside = true;
        for (int z = 0; z < 64; ++z)
            for (int x = 0; x < 64; ++x) {
                const int64_t dx = x - 32, dz = z - 32;
                const int64_t d2 = dx * dx + dz * dz;
                const fx hv = b.heightQ[(size_t)z * 64 + (size_t)x];
                if (d2 * 4 <= 144 && hv != target) coreExact = false;       // d <= 6 (r/2)
                if (d2 > 144 && hv != before.heightQ[(size_t)z * 64 + (size_t)x])
                    flatOutside = false;                                     // d > 12 (r)
            }
        check(coreExact, "(a) flatten hits the target EXACTLY across the flat core");
        check(flatOutside, "(a) flatten: outside-radius texels bit-untouched");

        // SMOOTH: 4 recorded passes over the bumpy terrain -> pinned after-K digest.
        au::AuthoredTerrain sm = au::MakeProcTerrain(64, 64, 16 * kOne, 4, 99u, 8 * kOne);
        au::TerrainHistory smHist;
        for (int k = 0; k < 4; ++k)
            au::RecordedSmoothBrush(smHist, sm, 32 * kOne, 32 * kOne, 10 * kOne);
        const uint64_t smDig = au::DigestTerrain(sm);
        std::printf("la1 pin: smooth-after-4 digest %016llx\n", (unsigned long long)smDig);
        check(smDig == kPinSmoothDigest, "(a) smooth after-4 digest pinned");
        check(smHist.commands.size() == 4, "(a) smooth passes each recorded");
    }

    // ================= (b) PAINT — the largest-remainder renorm =====================================
    {
        au::AuthoredTerrain t = au::MakeFlatTerrain(64, 64, 0);
        au::ApplyPaintBrush(t, 1, 32 * kOne, 32 * kOne, 8 * kOne, kOne);
        auto S = [&](int x, int z, int l) { return t.splat[((size_t)z * 64 + (size_t)x) * 4 + l]; };
        // Flat core at full strength: add = 255 -> the layer takes everything.
        check(S(32, 32, 0) == 0 && S(32, 32, 1) == 255 && S(32, 32, 2) == 0 && S(32, 32, 3) == 0,
              "(b) paint core -> {0,255,0,0} exactly");
        // The half-falloff ring (d = 6): add = 127 -> {128,127,0,0} (grass keeps floor(255*128/255)).
        check(S(38, 32, 0) == 128 && S(38, 32, 1) == 127,
              "(b) paint half-ring -> {128,127,0,0} exactly");
        // Restamp: layer1 127+127 = 254, rem 1 -> grass floor(128*1/128) = 1 -> {1,254,0,0}.
        au::ApplyPaintBrush(t, 1, 32 * kOne, 32 * kOne, 8 * kOne, kOne);
        check(S(38, 32, 0) == 1 && S(38, 32, 1) == 254, "(b) paint restamp -> {1,254,0,0} exactly");

        // Multi-paint over the proc terrain: EVERY texel sums to exactly 255 (the global invariant).
        au::AuthoredTerrain m = au::MakeProcTerrain(64, 64, 16 * kOne, 4, 7u, 8 * kOne);
        au::ApplyPaintBrush(m, 1, 20 * kOne, 20 * kOne, 12 * kOne, kOne);
        au::ApplyPaintBrush(m, 2, 26 * kOne, 22 * kOne, 9 * kOne, kOne / 2);
        au::ApplyPaintBrush(m, 3, 40 * kOne, 40 * kOne, 10 * kOne, kOne / 3);
        au::ApplyPaintBrush(m, 2, 22 * kOne, 24 * kOne, 7 * kOne, kOne);
        bool sum255 = true;
        for (size_t i = 0; i < 64 * 64; ++i) {
            int sum = 0;
            for (int l = 0; l < 4; ++l) sum += m.splat[i * 4 + l];
            if (sum != 255) sum255 = false;
        }
        check(sum255, "(b) splat weights renorm to EXACTLY 255 on every texel (global)");
        const uint64_t pDig = au::DigestTerrain(m);
        std::printf("la1 pin: paint digest %016llx\n", (unsigned long long)pDig);
        check(pDig == kPinPaintDigest, "(b) painted-region digest pinned");
    }

    // ================= (c) ROAD — the spline carve (SP1 composition) ================================
    {
        // A straight 2-point spline (the SP1 mirrored-phantom EXACT-LERP identity): along z = 32 at
        // height 4 wu, over flat terrain at 8 wu. Every polyline sample sits exactly on the line.
        au::AuthoredTerrain t = au::MakeFlatTerrain(64, 64, 8 * kOne);
        spline::Spline sp;
        sp.points = {au::FxVec3{8 * kOne, 4 * kOne, 32 * kOne},
                     au::FxVec3{56 * kOne, 4 * kOne, 32 * kOne}};
        const spline::ArcTable tab = spline::BuildArcTable(sp);
        const fx width = 6 * kOne, shoulder = 4 * kOne;   // halfW = 3
        const uint32_t roadTexels = au::ApplyCarveRoad(t, sp, tab, width, shoulder);
        auto H = [&](int x, int z) { return t.heightQ[(size_t)z * 64 + (size_t)x]; };
        auto RD = [&](int x, int z) { return t.splat[((size_t)z * 64 + (size_t)x) * 4 + au::kRoadLayer]; };
        // The bed: exactly the spline height + road layer 255, along the centerline and the full bed
        // half-width (d = |z-32| <= 3 for x within the run).
        bool bedFlat = true, bedPaint = true;
        for (int x = 8; x <= 56; ++x)
            for (int z = 29; z <= 35; ++z) {
                if (H(x, z) != 4 * kOne) bedFlat = false;
                if (RD(x, z) != 255) bedPaint = false;
            }
        check(bedFlat, "(c) road bed EXACTLY flat at the spline height (centerline + half-width)");
        check(bedPaint, "(c) road layer painted 255 across the bed");
        // The shoulder blends MONOTONICALLY from the bed (4 wu) back to the terrain (8 wu), continuous
        // at both edges: d = 7 (t = kOne) -> smoothstep(kOne) = kOne -> exactly the before-height.
        bool shoulderMono = true;
        for (int z = 32; z <= 38; ++z)
            if (H(32, z + 1) < H(32, z)) shoulderMono = false;
        check(shoulderMono, "(c) shoulder blend monotone across the band");
        check(H(32, 39) == 8 * kOne, "(c) shoulder outer edge continuous (smoothstep(kOne) exact)");
        check(H(32, 35) == 4 * kOne && H(32, 36) > 4 * kOne,
              "(c) shoulder inner edge continuous with the bed");
        // Off-band texels BIT-untouched (height 8 wu, grass 255, road 0) — beyond halfW+shoulder+cap.
        bool offRoad = true;
        for (int x = 0; x < 64 && offRoad; ++x)
            for (int z = 0; z < 64; ++z) {
                if (z >= 24 && z <= 40 && x >= 0 && x <= 64) continue;   // the band + end caps
                if (H(x, z) != 8 * kOne || RD(x, z) != 0 ||
                    t.splat[((size_t)z * 64 + (size_t)x) * 4 + 0] != 255) { offRoad = false; break; }
            }
        check(offRoad, "(c) off-road texels bit-untouched");
        std::printf("la1 pin: straight-road bed texels %u\n", roadTexels);
        check(roadTexels == kPinRoadTexels, "(c) straight-road bed texel count pinned");

        // The carve over the BUMPY showcase-style terrain: pinned digest (the nearest-polyline-point
        // bed height staircases on slopes — the pinned real profile, documented honesty).
        au::AuthoredTerrain b = au::MakeProcTerrain(64, 64, 16 * kOne, 4, 42u, 8 * kOne);
        spline::Spline sp2;
        sp2.points = {au::FxVec3{4 * kOne, 2 * kOne, 6 * kOne},
                      au::FxVec3{22 * kOne, 3 * kOne, 24 * kOne},
                      au::FxVec3{40 * kOne, 4 * kOne, 38 * kOne},
                      au::FxVec3{58 * kOne, 5 * kOne, 58 * kOne}};
        const spline::ArcTable tab2 = spline::BuildArcTable(sp2);
        au::ApplyCarveRoad(b, sp2, tab2, 5 * kOne, 4 * kOne);
        const uint64_t rDig = au::DigestTerrain(b);
        std::printf("la1 pin: bumpy-road digest %016llx\n", (unsigned long long)rDig);
        check(rDig == kPinRoadDigest, "(c) bumpy-terrain carve digest pinned");
    }

    // ================= (d) UNDO — the rect-memento proof, every op kind =============================
    {
        au::AuthoredTerrain t = au::MakeProcTerrain(64, 64, 16 * kOne, 4, 5u, 8 * kOne);
        au::TerrainHistory hist;
        spline::Spline sp;
        sp.points = {au::FxVec3{6 * kOne, 3 * kOne, 10 * kOne},
                     au::FxVec3{30 * kOne, 4 * kOne, 30 * kOne},
                     au::FxVec3{56 * kOne, 5 * kOne, 50 * kOne}};
        const spline::ArcTable tab = spline::BuildArcTable(sp);
        // Each Recorded op: pre-digest -> op -> post-digest -> Undo == pre BIT-EXACT -> Redo == post.
        struct Step { const char* name; int kind; };
        const Step steps[] = {{"raise", 0}, {"flatten", 1}, {"smooth", 2}, {"paint", 3}, {"carve", 4}};
        for (const Step& s : steps) {
            const uint64_t pre = au::DigestTerrain(t);
            switch (s.kind) {
                case 0: au::RecordedRaiseBrush(hist, t, 20 * kOne, 20 * kOne, 9 * kOne, 3 * kOne); break;
                case 1: au::RecordedFlattenBrush(hist, t, 44 * kOne, 20 * kOne, 8 * kOne, 5 * kOne); break;
                case 2: au::RecordedSmoothBrush(hist, t, 32 * kOne, 44 * kOne, 8 * kOne); break;
                case 3: au::RecordedPaintBrush(hist, t, 1, 20 * kOne, 20 * kOne, 9 * kOne, kOne); break;
                case 4: au::RecordedCarveRoad(hist, t, sp, tab, 5 * kOne, 4 * kOne); break;
            }
            const uint64_t post = au::DigestTerrain(t);
            check(post != pre, "(d) op changed the terrain");
            check(au::Undo(hist, t), "(d) Undo available");
            check(au::DigestTerrain(t) == pre, "(d) Undo returns the pre-op digest BIT-EXACT");
            check(au::Redo(hist, t), "(d) Redo available");
            check(au::DigestTerrain(t) == post, "(d) Redo re-applies to the post-op digest");
            (void)s;
        }
        check(hist.commands.size() == 5 && hist.cursor == 5, "(d) five commands recorded");
        // Undo-all -> the seed digest; the whole stack round-trips.
        const uint64_t authored = au::DigestTerrain(t);
        while (au::Undo(hist, t)) {}
        check(au::DigestTerrain(t) == au::DigestTerrain(au::MakeProcTerrain(64, 64, 16 * kOne, 4, 5u,
                                                                            8 * kOne)),
              "(d) undo-all returns the seed terrain BIT-EXACT");
        while (au::Redo(hist, t)) {}
        check(au::DigestTerrain(t) == authored, "(d) redo-all returns the authored terrain BIT-EXACT");
        // Branch-kill: undo 2, record a new op -> the redo tail is truncated.
        au::Undo(hist, t);
        au::Undo(hist, t);
        au::RecordedRaiseBrush(hist, t, 10 * kOne, 50 * kOne, 6 * kOne, 2 * kOne);
        check(hist.commands.size() == 4 && hist.cursor == 4, "(d) record-after-undo kills the redo tail");
        check(!au::Redo(hist, t), "(d) no redo after branch-kill");
        // Bitwise no-op records nothing (strength 0 raise).
        const size_t nCmd = hist.commands.size();
        au::RecordedRaiseBrush(hist, t, 20 * kOne, 20 * kOne, 6 * kOne, 0);
        check(hist.commands.size() == nCmd, "(d) a bitwise no-op op records nothing");
    }

    // ================= (e) DETERMINISM — the showcase scenario + raster, pinned ======================
    {
        const au::LandscapeShotRun run  = au::RunLandscapeShotScenario();
        const au::LandscapeShotRun run2 = au::RunLandscapeShotScenario();
        check(run.digest == run2.digest && run.seedDigest == run2.seedDigest,
              "(e) two-run scenario digests identical");
        check(run.undoRedoOk, "(e) scenario undo-all/redo-all round-trip BIT-EXACT");
        check(run.splatSum255, "(e) scenario splat weights sum 255 everywhere");
        check(run.flattenExact, "(e) scenario plateau flat-core exact");
        std::printf("la1 pin: seed digest %016llx\n", (unsigned long long)run.seedDigest);
        std::printf("la1 pin: scenario digest %016llx (ops:%u roadTexels:%u)\n",
                    (unsigned long long)run.digest, run.ops, run.roadTexels);
        check(run.seedDigest == kPinSeedDigest, "(e) seed digest pinned");
        check(run.digest == kPinScenarioDigest, "(e) scenario digest pinned");
        check(run.ops == 8u, "(e) scenario records 8 ops");
        check(run.roadTexels == kPinShowRoadTexels, "(e) showcase road texel count pinned");
        std::vector<uint8_t> img;
        uint32_t w = 0, h = 0;
        au::RenderLandscapeShot(run, img, w, h);
        const uint64_t rasterDig = hf::net::DigestBytes(img.data(), img.size());
        std::printf("la1 pin: raster digest %016llx (%ux%u)\n", (unsigned long long)rasterDig, w, h);
        check(w == 512 && h == 512, "(e) raster 512x512");
        check(rasterDig == kPinRasterDigest, "(e) raster digest pinned");
    }

    if (g_fail == 0) std::printf("terrain_author_test: ALL PASS\n");
    else std::printf("terrain_author_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
