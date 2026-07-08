// Slice SP1 — FIRST-CLASS DETERMINISTIC SPLINES (engine/spline/spline.h), the deterministic integer
// spline core + its three consumers. Pure CPU (header-only, no device, no backend symbols). Namespace
// hf::spline. What this test PINS (the contracts the showcase golden + every future spline consumer
// build on):
//   (a) EVAL — uniform Catmull-Rom integer Horner: fixed (segment,t) samples pinned EXACT on the fixed
//       6-point showcase S-curve; knot interpolation exact (Eval(i,0)==points[i], Eval(i,kOne)==
//       points[i+1] — Catmull-Rom interpolates its knots, bit-asserted); the 2-POINT STRAIGHT-LINE
//       IDENTITY (the mirrored-phantom convention makes Eval an EXACT integer lerp, c2=c3=0); the
//       CLOSED-LOOP WRAP (Eval(last,kOne)==Eval(0,0)==points[0] bit-exact).
//   (b) TANGENT — analytic derivative vs central finite difference within a pinned LSB band; on the
//       straight line the tangent is EXACTLY (B-A), constant.
//   (c) ARC-LENGTH — the fixed-K (16/segment) chord table: total pinned, cum monotonic, table digest
//       pinned, EvalByDistance(total/2) lands mid-curve at a pinned position, ArcToParam monotonic.
//   (d) SCATTER — ScatterAlongSpline at spacing kOne: pinned count + pinned positions/yaw quaternions,
//       consecutive chord distances pinned (spacing +- the chord-table quantization), the scatter digest
//       pinned, and COMPOSITION with pcg::PruneOverlaps (shuffle-invariance: reversed input -> identical
//       survivors — the canonical-order guarantee is pcg's, we just feed it).
//   (e) SWEEP — the road strip: vert/index counts + digest pinned, no degenerate index triples, width
//       exact (|right-left| == widthQ within a pinned LSB band) and perpendicular to the tangent
//       (|dot| within a pinned band) at EVERY sample, and the COLLINEAR IDENTITY (collinear control
//       points -> an exactly straight strip with exact +Y normals). The float mesh bridge
//       (StripToMeshVertices -> scene::Vertex) is shape-checked only (render-only, not bit-pinned).
//   (f) CAMERA — CameraAlongSpline endpoints land exactly on the end control points; forward is unit
//       within +-2 LSB. Plus the SHOWCASE SCENARIO digest pinned + two-run identical (the stat line's
//       digest), and every pinned digest printed for the MSVC==clang cross-compiler comparison
//       (`clang++ -std=c++20 -I engine -I third_party tests/spline_test.cpp` runs standalone).
//
// STATELESS — NO LOCKSTEP: splines are pure functions of their control points (no evolving state, no
// snapshot, no command stream), so there is deliberately NO lockstep/rollback harness here; the netcode
// story is inherited by consumers (seq.h-driven camera distance, pcg-seeded scatter) — documented in
// spline.h. Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests. pcg.h/seq.h/fpx.h
// are composed READ-ONLY (byte-untouched).
#include "spline/spline.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace sl = hf::spline;
using sl::fx;
using sl::kOne;
using sl::kFrac;
using sl::FxVec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static bool VecEq(const FxVec3& a, const FxVec3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

int main() {
    HF_TEST_MAIN_INIT();

    const sl::Spline sp = sl::MakeShowcaseSpline();

    // ================= (a) EVAL — pinned samples on the fixed 6-point S-curve =========================
    {
        check(sl::SegmentCount(sp) == 5, "SP1 eval: the open 6-point spline has 5 segments");
        // PINNED on first run (MSVC == clang — the cross-compiler bar).
        const int  pinSeg[4] = {0, 1, 3, 4};
        const fx   pinT[4]   = {kOne / 4, kOne / 2, 3 * kOne / 4, kOne / 3};
        const FxVec3 pinP[4] = {
            {-361984, 0, -100864}, {-167936, 0, 0}, {283136, 0, -23040}, {376224, 0, -16993}};
        bool evalPin = true;
        for (int i = 0; i < 4; ++i)
            if (!VecEq(sl::Eval(sp, pinSeg[i], pinT[i]), pinP[i])) evalPin = false;
        check(evalPin, "SP1 eval: fixed (segment,t) samples match the pinned exact positions");

        // Knot interpolation: Catmull-Rom interpolates its knots — EXACT at t=0 and t=kOne.
        bool knots = true;
        for (int i = 0; i < 5; ++i) {
            if (!VecEq(sl::Eval(sp, i, 0), sp.points[(size_t)i])) knots = false;
            if (!VecEq(sl::Eval(sp, i, kOne), sp.points[(size_t)i + 1])) knots = false;
        }
        check(knots, "SP1 eval: Eval(knot) == the control point exactly (t=0 and t=kOne, every segment)");
    }

    // ================= (a) the 2-point STRAIGHT-LINE identity (exact integer lerp) =====================
    {
        sl::Spline line;
        line.points = {FxVec3{0, 0, 0}, FxVec3{4 * kOne, 2 * kOne, 8 * kOne}};
        const FxVec3 A = line.points[0], B = line.points[1];
        bool lerpExact = true;
        for (fx t = 0; t <= kOne; t += kOne / 16) {
            const FxVec3 got = sl::Eval(line, 0, t);
            const FxVec3 want{A.x + (fx)(((int64_t)(B.x - A.x) * t) >> kFrac),
                              A.y + (fx)(((int64_t)(B.y - A.y) * t) >> kFrac),
                              A.z + (fx)(((int64_t)(B.z - A.z) * t) >> kFrac)};
            if (!VecEq(got, want)) lerpExact = false;
        }
        check(lerpExact, "SP1 eval: 2-point spline == the EXACT integer lerp (mirrored phantoms, c2=c3=0)");

        // (b) On the straight line the analytic tangent is EXACTLY (B-A), constant across t.
        bool tanExact = true;
        for (fx t = 0; t <= kOne; t += kOne / 8)
            if (!VecEq(sl::Tangent(line, 0, t), FxVec3{B.x - A.x, B.y - A.y, B.z - A.z})) tanExact = false;
        check(tanExact, "SP1 tangent: straight-line tangent is exactly (B-A), constant");
    }

    // ================= (a) the CLOSED-LOOP wrap (bit-exact continuity) =================================
    {
        sl::Spline loop;
        loop.closed = true;
        loop.points = {FxVec3{-2 * kOne, 0, -2 * kOne}, FxVec3{2 * kOne, 0, -2 * kOne},
                       FxVec3{2 * kOne, 0, 2 * kOne},   FxVec3{-2 * kOne, 0, 2 * kOne}};
        check(sl::SegmentCount(loop) == 4, "SP1 eval: the closed 4-point spline has 4 segments");
        check(VecEq(sl::Eval(loop, 3, kOne), sl::Eval(loop, 0, 0)) &&
                  VecEq(sl::Eval(loop, 3, kOne), loop.points[0]),
              "SP1 eval: closed loop wraps bit-exact (Eval(last,kOne) == Eval(0,0) == points[0])");
        bool knots = true;
        for (int i = 0; i < 4; ++i)
            if (!VecEq(sl::Eval(loop, i, 0), loop.points[(size_t)i])) knots = false;
        check(knots, "SP1 eval: closed-loop knots interpolate exactly");
    }

    // ================= (b) TANGENT — analytic vs central finite difference (pinned band) ===============
    {
        // FD = (Eval(t+h) - Eval(t-h)) * 128 with h = kOne/256 (so 1/(2h) == 128, an exact integer
        // scale). Truncation is O(h^2 * |c3|) — observed max error 133 LSB on the showcase curve;
        // pinned band 256.
        const fx h = kOne / 256;
        int maxErr = 0;
        for (int seg = 0; seg < sl::SegmentCount(sp); ++seg) {
            for (fx t = h; t <= kOne - h; t += kOne / 16) {
                const FxVec3 an = sl::Tangent(sp, seg, t);
                const FxVec3 fd = sl::FxScale(sl::FxSub(sl::Eval(sp, seg, t + h), sl::Eval(sp, seg, t - h)),
                                              128 * kOne);
                int ex = an.x - fd.x; if (ex < 0) ex = -ex;
                int ey = an.y - fd.y; if (ey < 0) ey = -ey;
                int ez = an.z - fd.z; if (ez < 0) ez = -ez;
                if (ex > maxErr) maxErr = ex;
                if (ey > maxErr) maxErr = ey;
                if (ez > maxErr) maxErr = ez;
            }
        }
        std::printf("tangent-fd: maxErr=%d LSB (band 256)\n", maxErr);
        check(maxErr <= 256, "SP1 tangent: analytic vs finite-difference within the pinned 256-LSB band");
        // A pinned exact sample of the analytic derivative (mid segment 1).
        check(VecEq(sl::Tangent(sp, 1, kOne / 2), FxVec3{204800, 0, -409600}),
              "SP1 tangent: pinned exact analytic tangent at (seg 1, t=0.5)");
    }

    // ================= (c) ARC-LENGTH — the fixed-K chord table ========================================
    const sl::ArcTable tab = sl::BuildArcTable(sp);
    {
        check((int)tab.cum.size() == 5 * sl::kArcSamplesPerSeg + 1,
              "SP1 arc: table has segCount*16 + 1 entries");
        check(sl::ArcTotal(tab) == 1727971, "SP1 arc: pinned total polyline length (1727971 = ~26.37 wu)");
        bool mono = true;
        for (size_t i = 0; i + 1 < tab.cum.size(); ++i)
            if (tab.cum[i + 1] < tab.cum[i]) mono = false;
        check(mono && tab.cum[0] == 0, "SP1 arc: cumulative table is monotonic from 0");
        const uint64_t cumDigest = hf::net::DigestBytes(tab.cum.data(), tab.cum.size() * sizeof(fx));
        std::printf("arc: cumDigest=%016llx (pinned f9eb996033189a0b)\n", (unsigned long long)cumDigest);
        check(cumDigest == 0xf9eb996033189a0bull, "SP1 arc: pinned chord-table digest (MSVC == clang)");
        // EvalByDistance(total/2) lands mid-curve — pinned exact position + a mid-parameter sanity.
        check(VecEq(sl::EvalByDistance(sp, tab, sl::ArcTotal(tab) / 2), FxVec3{21332, 0, 5246}),
              "SP1 arc: EvalByDistance(total/2) lands at the pinned mid-curve position");
        const sl::SplineParam mid = sl::ArcToParam(tab, sl::ArcTotal(tab) / 2);
        check(mid.seg == 2, "SP1 arc: the arc midpoint lands in the middle segment (seg 2 of 5)");
        // ArcToParam monotonic: increasing distance -> lexicographically non-decreasing (seg, t).
        bool paramMono = true;
        sl::SplineParam prev = sl::ArcToParam(tab, 0);
        for (fx s = 0; s <= sl::ArcTotal(tab); s += kOne / 4) {
            const sl::SplineParam p = sl::ArcToParam(tab, s);
            if (p.seg < prev.seg || (p.seg == prev.seg && p.t < prev.t)) paramMono = false;
            prev = p;
        }
        check(paramMono, "SP1 arc: ArcToParam is monotonic in distance");
        // Endpoints: s=0 -> the first control point; s=total -> the last (both exact knots).
        check(VecEq(sl::EvalByDistance(sp, tab, 0), sp.points.front()) &&
                  VecEq(sl::EvalByDistance(sp, tab, sl::ArcTotal(tab)), sp.points.back()),
              "SP1 arc: distance endpoints land exactly on the end control points");
    }

    // ================= (d) SCATTER — pinned instances + the pcg composition ============================
    {
        const pcg::PcgStream stream{77u, 1u};
        const std::vector<pcg::PcgInstance> inst =
            sl::ScatterAlongSpline(sp, tab, kOne, 0, 0, stream);   // spacing 1 wu, no offset, no jitter
        check(inst.size() == 27, "SP1 scatter: pinned count (total/spacing + 1 == 27)");
        // Pinned positions + yaw quaternions (first three instances; scale is kOne by contract).
        const FxVec3 pinPos[3] = {{-393216, 0, -196608}, {-372000, 0, -134744}, {-353690, 0, -71693}};
        const fx pinYawY[3] = {12394, 9687, 8985};
        const fx pinYawW[3] = {64353, 64815, 64917};
        bool pinned = inst.size() >= 3;
        for (int i = 0; i < 3 && pinned; ++i) {
            if (!VecEq(inst[(size_t)i].pos, pinPos[i])) pinned = false;
            if (inst[(size_t)i].orient.y != pinYawY[i] || inst[(size_t)i].orient.w != pinYawW[i] ||
                inst[(size_t)i].orient.x != 0 || inst[(size_t)i].orient.z != 0 ||
                inst[(size_t)i].scale != kOne)
                pinned = false;
        }
        check(pinned, "SP1 scatter: pinned positions + yaw-from-tangent quaternions (first 3 instances)");
        // Spacing honored: consecutive chord distances pinned (spacing +- the chord-table quantization;
        // chords cut curvature so min < spacing; the polyline table slightly under-measures so max can
        // exceed spacing by a few hundred LSB). Pinned EXACT min/max.
        fx minD = 0x7FFFFFFF, maxD = -0x7FFFFFFF;
        for (size_t i = 0; i + 1 < inst.size(); ++i) {
            const fx d = sl::FxLength(sl::FxSub(inst[i + 1].pos, inst[i].pos));
            if (d < minD) minD = d;
            if (d > maxD) maxD = d;
        }
        std::printf("scatter: chord min=%d max=%d (spacing=%d, pinned 52700/65827)\n", minD, maxD, kOne);
        check(minD == 52700 && maxD == 65827,
              "SP1 scatter: consecutive arc spacing pinned (spacing +- table quantization, exact min/max)");
        // The scatter digest (pos + yaw fields, field-serialized) — pinned MSVC == clang.
        std::vector<fx> sbuf;
        for (const pcg::PcgInstance& in : inst) {
            sbuf.push_back(in.pos.x); sbuf.push_back(in.pos.y); sbuf.push_back(in.pos.z);
            sbuf.push_back(in.orient.y); sbuf.push_back(in.orient.w);
        }
        const uint64_t sDigest = hf::net::DigestBytes(sbuf.data(), sbuf.size() * sizeof(fx));
        std::printf("scatter: digest=%016llx (pinned c47ce5b6134669dd)\n", (unsigned long long)sDigest);
        check(sDigest == 0xc47ce5b6134669ddull, "SP1 scatter: pinned scatter digest");
        // COMPOSITION with pcg::PruneOverlaps — shuffle-invariance: reversed input -> IDENTICAL survivors
        // (pcg's canonical-order guarantee; scatter output is the PcgInstance currency, unmodified pcg).
        const std::vector<pcg::PcgInstance> jittered =
            sl::ScatterAlongSpline(sp, tab, kOne / 2, 0, kOne / 4, pcg::PcgStream{77u, 9u});
        std::vector<pcg::PcgInstance> reversed(jittered.rbegin(), jittered.rend());
        const std::vector<pcg::PcgInstance> keepA = pcg::PruneOverlaps(jittered, kOne / 4);
        const std::vector<pcg::PcgInstance> keepB = pcg::PruneOverlaps(reversed, kOne / 4);
        bool same = keepA.size() == keepB.size() && !keepA.empty() && keepA.size() < jittered.size();
        for (size_t i = 0; same && i < keepA.size(); ++i)
            if (!VecEq(keepA[i].pos, keepB[i].pos)) same = false;
        check(same, "SP1 scatter: composes with pcg::PruneOverlaps (shuffle-invariant survivors, some pruned)");
    }

    // ================= (e) SWEEP — the road strip ======================================================
    {
        const sl::SweptStrip strip = sl::SweepStrip(sp, sl::kShotRoadWidth, sl::kShotSegsPerSpan);
        // 5 segments * 8 + 1 = 41 samples -> 82 verts; 40 spans * 6 = 240 indices.
        check(strip.positions.size() == 82 && strip.normals.size() == 82 && strip.tangents.size() == 82,
              "SP1 sweep: pinned vertex count (41 samples -> 82 verts)");
        check(strip.indices.size() == 240, "SP1 sweep: pinned index count (40 spans -> 240)");
        // The strip digest (positions + indices, field-serialized) — pinned MSVC == clang.
        std::vector<fx> vbuf;
        for (const FxVec3& p : strip.positions) { vbuf.push_back(p.x); vbuf.push_back(p.y); vbuf.push_back(p.z); }
        for (uint32_t ix : strip.indices) vbuf.push_back((fx)ix);
        const uint64_t vDigest = hf::net::DigestBytes(vbuf.data(), vbuf.size() * sizeof(fx));
        std::printf("sweep: digest=%016llx (pinned 0515501b575a049c)\n", (unsigned long long)vDigest);
        check(vDigest == 0x0515501b575a049cull, "SP1 sweep: pinned strip digest");
        // No degenerate triangles: every index triple is distinct (and in vertex range).
        bool okTris = true;
        for (size_t t = 0; t + 2 < strip.indices.size(); t += 3) {
            const uint32_t a = strip.indices[t], b = strip.indices[t + 1], c = strip.indices[t + 2];
            if (a == b || b == c || a == c) okTris = false;
            if (a >= strip.positions.size() || b >= strip.positions.size() || c >= strip.positions.size())
                okTris = false;
        }
        check(okTris, "SP1 sweep: no degenerate index triples, all indices in range");
        // Width exact + perpendicular at EVERY sample: |right-left| == widthQ within a pinned 8-LSB band
        // (observed 3), and (right-left) . tangentDir within a pinned 16-LSB band (observed 2).
        fx maxLenErr = 0;
        int64_t maxDot = 0;
        const size_t S = strip.positions.size() / 2;
        for (size_t k = 0; k < S; ++k) {
            const FxVec3 d = sl::FxSub(strip.positions[k * 2 + 1], strip.positions[k * 2]);
            fx err = sl::FxLength(d) - sl::kShotRoadWidth;
            if (err < 0) err = -err;
            if (err > maxLenErr) maxLenErr = err;
            const FxVec3& tn = strip.tangents[k * 2];
            int64_t dot = ((int64_t)d.x * tn.x + (int64_t)d.y * tn.y + (int64_t)d.z * tn.z) >> kFrac;
            if (dot < 0) dot = -dot;
            if (dot > maxDot) maxDot = dot;
        }
        std::printf("sweep: maxLenErr=%d maxAbsDot=%lld (bands 8/16)\n", maxLenErr, (long long)maxDot);
        check(maxLenErr <= 8, "SP1 sweep: strip width exact within the pinned 8-LSB band at every sample");
        check(maxDot <= 16, "SP1 sweep: strip cross-section perpendicular to the tangent (pinned 16-LSB band)");
        // COLLINEAR IDENTITY: collinear control points -> an exactly straight strip with exact +Y normals.
        sl::Spline straight;
        straight.points = {FxVec3{0, 0, 0}, FxVec3{2 * kOne, 0, 0}, FxVec3{4 * kOne, 0, 0},
                           FxVec3{6 * kOne, 0, 0}};
        const sl::SweptStrip flat = sl::SweepStrip(straight, kOne, 4);
        const fx halfW = kOne >> 1;
        bool flatExact = !flat.positions.empty();
        for (size_t k = 0; k < flat.positions.size() / 2; ++k) {
            const FxVec3& L = flat.positions[k * 2];
            const FxVec3& R = flat.positions[k * 2 + 1];
            if (L.z != -halfW || R.z != halfW || L.y != 0 || R.y != 0) flatExact = false;
            if (!VecEq(flat.normals[k * 2], FxVec3{0, kOne, 0})) flatExact = false;
            if (!VecEq(flat.tangents[k * 2], FxVec3{kOne, 0, 0})) flatExact = false;
        }
        check(flatExact, "SP1 sweep: collinear control points give an exactly straight strip (+Y normals)");
        // The FLOAT mesh bridge (render-only): shape-check — same vert count, rails uv 0/1, layout usable
        // with scene::MeshVertexLayout (stride 56 == sizeof(scene::Vertex)).
        const std::vector<scene::Vertex> mesh = sl::StripToMeshVertices(strip, 0.5f, 0.5f, 0.5f);
        check(mesh.size() == strip.positions.size() && sizeof(scene::Vertex) == 56 &&
                  mesh[0].uv[0] == 0.0f && mesh[1].uv[0] == 1.0f,
              "SP1 sweep: StripToMeshVertices emits the engine mesh layout (56B verts, rail uv 0/1)");
    }

    // ================= (f) CAMERA + the shared showcase scenario =======================================
    {
        const sl::SplineCamera c0 = sl::CameraAlongSpline(sp, tab, 0);
        const sl::SplineCamera c1 = sl::CameraAlongSpline(sp, tab, sl::ArcTotal(tab));
        check(VecEq(c0.pos, sp.points.front()) && VecEq(c1.pos, sp.points.back()),
              "SP1 camera: rail endpoints land exactly on the end control points");
        bool unitFwd = true;
        for (int i = 0; i <= 8; ++i) {
            const fx s = (fx)(((int64_t)i * sl::ArcTotal(tab)) / 8);
            const fx len = sl::FxLength(sl::CameraAlongSpline(sp, tab, s).forward);
            if (len < kOne - 2 || len > kOne + 2) unitFwd = false;
        }
        check(unitFwd, "SP1 camera: forward is the unit tangent (+-2 LSB) along the whole rail");

        // The shared showcase scenario (the golden's digest): pinned + two-run identical (stateless —
        // a pure function of the control points; no lockstep harness by design, see spline.h).
        const sl::SplineShotRun run  = sl::RunSplineShotScenario();
        const sl::SplineShotRun run2 = sl::RunSplineShotScenario();
        std::printf("shot: digest=%016llx (pinned 24fa72580c7c2cef) points=%zu samples=%zu instances=%zu "
                    "stripTris=%zu\n",
                    (unsigned long long)run.digest, run.spline.points.size(),
                    run.strip.positions.size() / 2, run.postsL.size() + run.postsR.size(),
                    run.strip.indices.size() / 3);
        check(run.digest == run2.digest, "SP1 shot: two-run identical (stateless pure function)");
        check(run.digest == 0x24fa72580c7c2cefull, "SP1 shot: pinned showcase scenario digest");
        check(run.spline.points.size() == 6 && run.strip.positions.size() / 2 == 41 &&
                  run.postsL.size() + run.postsR.size() == 53 && run.strip.indices.size() / 3 == 80,
              "SP1 shot: pinned stat-line counts {points:6, samples:41, instances:53, stripTris:80}");
        // The shared raster itself: two renders byte-identical (the strict-zero golden's substrate).
        std::vector<uint8_t> imgA, imgB;
        uint32_t wA = 0, hA = 0, wB = 0, hB = 0;
        sl::RenderSplineShot(run, imgA, wA, hA);
        sl::RenderSplineShot(run2, imgB, wB, hB);
        check(wA == wB && hA == hB && imgA == imgB && wA == 680 && hA == 400,
              "SP1 shot: the shared integer raster is two-run byte-identical (680x400)");
    }

    if (g_fail == 0) std::printf("spline_test: ALL PASS\n");
    else std::printf("spline_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
