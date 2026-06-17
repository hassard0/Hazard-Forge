// Slice DI — DDGI Slice 2: Probe Radiance Capture. Pure CPU math: the capture-loop face count, the
// probe×face flat-slot indexing, the reuse of cubemap::FaceView/FaceProj from the probe center, the
// ProbeRadiance layout descriptor, and the deterministic debug-viz face average. No device, ASan-eligible
// (links hf_core). Exercises the EXACT math the --probecapture-shot showcase (samples/hello_triangle/
// main.cpp) + the Metal --probecapture showcase consume (engine/render/probe_capture.h), so the GPU
// capture loop's probe×face slot layout + the 6-face projection agree with this CPU reference.
//
// Properties pinned (per the spec §5):
//   * CaptureFaceCount == probeCount*6; EXACTLY 0 at dimX/dimY/dimZ == 0 (the probeCount=0 no-op).
//   * ProbeFaceIndex(p,f) round-trips (ProbeFaceFromIndex) + covers all probe×face slots without overlap.
//   * FaceView/FaceProj reuse: the 6 cubemap::FaceView(f, probeCenter) look down the 6 axes from the
//     probe center; the +Y/-Y up-vector convention is re-asserted (the classic cubemap pole flip).
//   * ProbeRadiance: faceBytes/probeBytes match faceSize²·8·{1,6} (RGBA16F, 6 faces).
//   * FaceAverage: hand cases (constant face → that constant; mixed → the mean) + determinism.
#include "render/probe_capture.h"
#include "render/cubemap.h"
#include "render/probe_gi.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace probecap = hf::render::probecap;
namespace cubemap  = hf::render::cubemap;
using hf::math::Mat4;
using hf::math::Vec3;
using hf::math::Vec4;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

int main() {
    // ====================================================================================
    // CaptureFaceCount == probeCount*6; 0 at dimX/dimY/dimZ == 0
    // ====================================================================================
    {
        probecap::ProbeGrid grid;          // the SMALL DI capture grid: 2x2x2 = 8 probes
        grid.dimX = 2; grid.dimY = 2; grid.dimZ = 2; grid.spacing = 1.0f;
        check(grid.probeCount() == 8, "DI capture grid: 2x2x2 == 8 probes");
        check(probecap::CaptureFaceCount(grid) == 8 * 6,
              "CaptureFaceCount: 8 probes -> 48 face renders (probeCount*6)");

        // A few more shapes.
        probecap::ProbeGrid g111; g111.dimX = 1; g111.dimY = 1; g111.dimZ = 1;
        check(probecap::CaptureFaceCount(g111) == 6, "CaptureFaceCount: 1 probe -> 6 faces");
        probecap::ProbeGrid g321; g321.dimX = 3; g321.dimY = 2; g321.dimZ = 1;
        check(probecap::CaptureFaceCount(g321) == 6 * 6, "CaptureFaceCount: 6 probes -> 36 faces");

        // The probeCount=0 no-op: each zero dim -> probeCount 0 -> CaptureFaceCount 0.
        probecap::ProbeGrid gx; gx.dimX = 0; gx.dimY = 2; gx.dimZ = 2;
        probecap::ProbeGrid gy; gy.dimX = 2; gy.dimY = 0; gy.dimZ = 2;
        probecap::ProbeGrid gz; gz.dimX = 2; gz.dimY = 2; gz.dimZ = 0;
        check(gx.probeCount() == 0 && probecap::CaptureFaceCount(gx) == 0,
              "Disabled: dimX==0 -> probeCount 0 -> CaptureFaceCount 0 (capture loop skipped)");
        check(gy.probeCount() == 0 && probecap::CaptureFaceCount(gy) == 0,
              "Disabled: dimY==0 -> probeCount 0 -> CaptureFaceCount 0");
        check(gz.probeCount() == 0 && probecap::CaptureFaceCount(gz) == 0,
              "Disabled: dimZ==0 -> probeCount 0 -> CaptureFaceCount 0");
    }

    // ====================================================================================
    // ProbeFaceIndex — round-trip + full coverage without overlap
    // ====================================================================================
    {
        const int kProbes = 8;
        const int kSlots = kProbes * probecap::kFaces;   // 48
        check(probecap::kFaces == 6, "kFaces == 6 (the cube faces)");

        // Round-trip: ProbeFaceIndex(p,f) -> ProbeFaceFromIndex -> (p,f).
        bool roundTrip = true;
        for (int p = 0; p < kProbes; ++p)
            for (int f = 0; f < probecap::kFaces; ++f) {
                int slot = probecap::ProbeFaceIndex(p, f);
                int rp, rf;
                probecap::ProbeFaceFromIndex(slot, rp, rf);
                if (rp != p || rf != f) roundTrip = false;
            }
        check(roundTrip, "ProbeFaceIndex: round-trips (probe,face) via ProbeFaceFromIndex");

        // Coverage: every (p,f) maps to a UNIQUE slot in [0, kSlots) with no gaps/overlap.
        std::vector<int> hit(kSlots, 0);
        bool inRange = true;
        for (int p = 0; p < kProbes; ++p)
            for (int f = 0; f < probecap::kFaces; ++f) {
                int slot = probecap::ProbeFaceIndex(p, f);
                if (slot < 0 || slot >= kSlots) { inRange = false; continue; }
                hit[slot]++;
            }
        check(inRange, "ProbeFaceIndex: every slot is in [0, probeCount*6)");
        bool noOverlap = true, fullCover = true;
        for (int s = 0; s < kSlots; ++s) {
            if (hit[s] > 1) noOverlap = false;
            if (hit[s] == 0) fullCover = false;
        }
        check(noOverlap, "ProbeFaceIndex: no two (probe,face) share a slot (no overlap)");
        check(fullCover, "ProbeFaceIndex: covers every slot exactly once (no gaps)");

        // Probe-major, face-minor stride: probe p's block starts at p*6, faces are contiguous.
        check(probecap::ProbeFaceIndex(0, 0) == 0 && probecap::ProbeFaceIndex(0, 5) == 5 &&
                  probecap::ProbeFaceIndex(1, 0) == 6 && probecap::ProbeFaceIndex(7, 5) == 47,
              "ProbeFaceIndex: probe-major face-minor (probe p block = [p*6, p*6+6))");
    }

    // ====================================================================================
    // FaceView/FaceProj reuse — the 6 axes from the probe center, +Y/-Y convention
    // ====================================================================================
    {
        // Probe NOT at the origin (the DI case: probes are scattered on the lattice). Each face's view
        // must look from probeCenter down that face's axis, so probeCenter + FaceDir(f) projects to the
        // face center (UV 0.5,0.5) and the view transforms probeCenter to the view-space origin.
        const Vec3 probeCenter{2.0f, -1.0f, 3.0f};
        const float zN = 0.05f, zF = 60.0f;

        bool dirOk = true, originOk = true;
        for (int f = 0; f < probecap::kFaces; ++f) {
            Mat4 view = cubemap::FaceView(f, probeCenter);
            // The probe center maps to the view-space origin (the camera eye).
            Vec3 eyeVS = hf::math::MulPoint(view, probeCenter);
            if (!approx(eyeVS.x, 0.0f, 1e-4f) || !approx(eyeVS.y, 0.0f, 1e-4f) ||
                !approx(eyeVS.z, 0.0f, 1e-4f)) originOk = false;
            // A point one unit down the face axis from the probe is straight ahead (view -Z, RH).
            Vec3 ahead = probeCenter + cubemap::FaceDir(f);
            Vec3 aheadVS = hf::math::MulPoint(view, ahead);
            // Straight ahead in RH view space: x≈0, y≈0, z≈-1 (looking down -Z).
            if (!approx(aheadVS.x, 0.0f, 1e-4f) || !approx(aheadVS.y, 0.0f, 1e-4f) ||
                !approx(aheadVS.z, -1.0f, 1e-4f)) dirOk = false;
        }
        check(originOk, "FaceView: the probe center maps to the view-space origin for all 6 faces");
        check(dirOk, "FaceView: probeCenter+FaceDir(f) is straight ahead (view -Z) for all 6 faces");

        // The +Y/-Y up-vector convention (the classic cubemap pole-flip pitfall): the vertical faces use
        // ±Z up, NOT ±Y (which would be parallel to the look axis). Re-assert the DD convention here.
        check(approx(cubemap::FaceUp(2).z, 1.0f) && approx(cubemap::FaceUp(2).y, 0.0f),
              "FaceUp: +Y face (2) up == +Z (not ±Y — pole would be degenerate)");
        check(approx(cubemap::FaceUp(3).z, -1.0f) && approx(cubemap::FaceUp(3).y, 0.0f),
              "FaceUp: -Y face (3) up == -Z");
        check(approx(cubemap::FaceUp(0).y, -1.0f) && approx(cubemap::FaceUp(4).y, -1.0f),
              "FaceUp: lateral faces (+X,+Z) up == -Y");

        // ProbeFaceViewProj == cubemap::FaceViewProj at the probe center (the thin reuse is exact).
        bool vpOk = true;
        for (int f = 0; f < probecap::kFaces; ++f) {
            Mat4 a = probecap::ProbeFaceViewProj(f, probeCenter, zN, zF);
            Mat4 b = cubemap::FaceViewProj(f, probeCenter, zN, zF);
            if (std::memcmp(&a, &b, sizeof(Mat4)) != 0) vpOk = false;
        }
        check(vpOk, "ProbeFaceViewProj == cubemap::FaceViewProj at the probe center (exact reuse)");

        // The +Y face center reconstructs to the +Y direction via the hardware cube lookup (capture and
        // sampling agree): a direction straight down +Y selects face 2 at UV (0.5,0.5).
        int selFace; hf::math::Vec2 selUV;
        cubemap::DirToFaceUV(Vec3{0, 1, 0}, selFace, selUV);
        check(selFace == 2 && approx(selUV.x, 0.5f) && approx(selUV.y, 0.5f),
              "DirToFaceUV: +Y dir selects face 2 at UV (0.5,0.5) — capture/sample agree on the pole");
    }

    // ====================================================================================
    // ProbeRadiance — the read-back layout descriptor (RGBA16F, 6 faces)
    // ====================================================================================
    {
        probecap::ProbeRadiance pr;
        pr.faceSize = 256;
        check(pr.faceCount == 6, "ProbeRadiance: 6 faces");
        check(pr.faceBytes() == (long)256 * 256 * 8, "ProbeRadiance: faceBytes == faceSize^2 * 8 (RGBA16F)");
        check(pr.probeBytes() == (long)256 * 256 * 8 * 6, "ProbeRadiance: probeBytes == 6 * faceBytes");
        // A zero-size descriptor is benign (0 bytes).
        probecap::ProbeRadiance z;
        check(z.faceBytes() == 0 && z.probeBytes() == 0, "ProbeRadiance: faceSize 0 -> 0 bytes");
    }

    // ====================================================================================
    // FaceAverage — hand cases + determinism
    // ====================================================================================
    {
        // A constant face averages to that constant (4 texels, all (0.25,0.5,0.75,1.0)).
        std::vector<float> constFace;
        for (int i = 0; i < 4; ++i) { constFace.push_back(0.25f); constFace.push_back(0.5f);
                                      constFace.push_back(0.75f); constFace.push_back(1.0f); }
        Vec4 avg = probecap::FaceAverage(constFace.data(), 4);
        check(approx(avg.x, 0.25f) && approx(avg.y, 0.5f) && approx(avg.z, 0.75f) && approx(avg.w, 1.0f),
              "FaceAverage: a constant face averages to that constant");

        // A mixed face averages to the per-channel mean. Two texels: (1,0,0,0) and (0,1,0,0) -> (.5,.5,0,0).
        float mixed[8] = {1, 0, 0, 0,  0, 1, 0, 0};
        Vec4 m = probecap::FaceAverage(mixed, 2);
        check(approx(m.x, 0.5f) && approx(m.y, 0.5f) && approx(m.z, 0.0f) && approx(m.w, 0.0f),
              "FaceAverage: a mixed face averages to the per-channel mean");

        // texelCount<=0 / null -> the zero vector (an empty face is black).
        Vec4 empty = probecap::FaceAverage(mixed, 0);
        check(approx(empty.x, 0.0f) && approx(empty.y, 0.0f) && approx(empty.z, 0.0f) && approx(empty.w, 0.0f),
              "FaceAverage: texelCount 0 -> black (zero vector)");
        Vec4 nullAvg = probecap::FaceAverage(nullptr, 4);
        check(approx(nullAvg.x, 0.0f) && approx(nullAvg.w, 0.0f),
              "FaceAverage: null data -> black (no UB)");

        // Determinism: a second call over the same data is byte-identical.
        Vec4 a = probecap::FaceAverage(constFace.data(), 4);
        Vec4 b = probecap::FaceAverage(constFace.data(), 4);
        check(std::memcmp(&a, &b, sizeof(Vec4)) == 0,
              "FaceAverage: deterministic (byte-identical across calls)");
    }

    if (g_fail == 0) { std::printf("probe_capture_test OK\n"); return 0; }
    std::printf("probe_capture_test: %d failures\n", g_fail);
    return 1;
}
