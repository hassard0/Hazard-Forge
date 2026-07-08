#pragma once

// Slice SK1 — the SHOWCASE half: a self-contained authored UsdSkel asset + the pure-CPU shot scenario both
// backends run (Vulkan --sk1-import-shot / Metal --sk1-import) and the shared strict-zero stick-figure raster.
// Kept OUT of usd_skel.h so the importer stays device-free + retarget-free; this header composes the imported
// rig with the EXISTING anim + retarget (AN2) stack:
//   ImportUsdSkel(asset) -> anim::Skeleton + anim::Animation
//     -> SampleLocalPose (float) -> retarget::QuantizePose (the ONE Q16.16 boundary)
//     -> retarget::ForwardKinematics (INTEGER) -> model-space joints  [strict-zero cross-backend BY CONSTRUCTION]
//     -> retarget::BuildRetargetMap/Retarget (self-retarget) -> proves imported rigs feed AN2.
// The clip is sampled AT keyframe times only (frames 0/12/24 -> 0/0.5/1.0 s) so the pre-quantization float
// pose is the stored keyframe value EXACTLY (no slerp) — the AN1/AN2 discipline. NO new shader / RHI.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "asset/usd_skel.h"     // ImportUsdSkel / UsdSkelImport / DigestUsdSkel
#include "anim/animation.h"     // SampleLocalPose
#include "anim/retarget.h"      // QuantizePose / ForwardKinematics / FxJointModel / BuildRetargetMap / Retarget / DigestModel

namespace hf::asset {

// A minimal but complete authored UsdSkel rig: a 3-joint chain (Root -> Hip -> Knee), a 6-vertex / 2-quad
// skinned strip (2 influences per vertex, weights summing to 1), and a Z-bend clip on Hip+Knee at 3 frames.
// Bind/rest transforms are translation-only (identity rotation, unit scale) so the import digest is exactly
// representable -> cross-compiler exact. USD conventions: matrix translation in the LAST ROW; quat = (w,x,y,z).
inline const char* Sk1UsdaAsset() {
    return
        "#usda 1.0\n"
        "(\n"
        "    upAxis = \"Y\"\n"
        "    timeCodesPerSecond = 24\n"
        ")\n"
        "def SkelRoot \"Character\"\n"
        "{\n"
        "    def Skeleton \"Rig\"\n"
        "    {\n"
        "        uniform token[] joints = [\"Root\", \"Root/Hip\", \"Root/Hip/Knee\"]\n"
        "        uniform matrix4d[] bindTransforms = [\n"
        "            ( (1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1) ),\n"
        "            ( (1,0,0,0),(0,1,0,0),(0,0,1,0),(0,1,0,1) ),\n"
        "            ( (1,0,0,0),(0,1,0,0),(0,0,1,0),(0,2,0,1) )\n"
        "        ]\n"
        "        uniform matrix4d[] restTransforms = [\n"
        "            ( (1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1) ),\n"
        "            ( (1,0,0,0),(0,1,0,0),(0,0,1,0),(0,1,0,1) ),\n"
        "            ( (1,0,0,0),(0,1,0,0),(0,0,1,0),(0,1,0,1) )\n"
        "        ]\n"
        "    }\n"
        "    def SkelAnimation \"Motion\"\n"
        "    {\n"
        "        uniform token[] joints = [\"Root/Hip\", \"Root/Hip/Knee\"]\n"
        "        quatf[] rotations.timeSamples = {\n"
        "            0: [(1,0,0,0), (1,0,0,0)],\n"
        "            12: [(0.9238795,0,0,0.3826834), (0.9238795,0,0,0.3826834)],\n"
        "            24: [(0.7071068,0,0,0.7071068), (0.7071068,0,0,0.7071068)]\n"
        "        }\n"
        "        float3[] translations.timeSamples = {\n"
        "            0: [(0,1,0), (0,1,0)],\n"
        "            24: [(0,1,0), (0,1,0)]\n"
        "        }\n"
        "    }\n"
        "    def Mesh \"Body\"\n"
        "    {\n"
        "        point3f[] points = [(-0.5,0,0),(0.5,0,0),(-0.5,1,0),(0.5,1,0),(-0.5,2,0),(0.5,2,0)]\n"
        "        int[] faceVertexCounts = [4, 4]\n"
        "        int[] faceVertexIndices = [0,1,3,2, 2,3,5,4]\n"
        "        int[] primvars:skel:jointIndices = [0,1, 0,1, 1,2, 1,2, 2,1, 2,1]\n"
        "        float[] primvars:skel:jointWeights = [0.8,0.2, 0.8,0.2, 0.6,0.4, 0.6,0.4, 0.9,0.1, 0.9,0.1]\n"
        "    }\n"
        "}\n";
}

namespace sk1 {

namespace rtn = hf::anim::retarget;
namespace mm  = hf::anim::mm;

inline constexpr int kShotFrames = 3;                       // clip keyframe times (seconds)
inline const float* ShotTimes() { static const float t[kShotFrames] = {0.0f, 0.5f, 1.0f}; return t; }

struct Sk1ShotFrame {
    std::vector<rtn::FxJointModel> model;                   // imported rig posed at this clip time (model space)
};

struct Sk1ShotRun {
    UsdSkelImport             import;
    std::vector<Sk1ShotFrame> frames;
    int32_t  bones     = 0;
    int32_t  verts     = 0;
    int32_t  clips     = 0;
    int32_t  keyframes = 0;                                 // total keyframes across all channels of clip 0
    uint64_t importDigest   = 0;                            // DigestUsdSkel (skeleton+skin+clips)
    uint64_t poseDigest     = 0;                            // integer model-space palette over the frames
    uint64_t retargetDigest = 0;                            // self-retarget compose proof (imported rig -> AN2)
    uint64_t digest         = 0;                            // combined two-run comparison currency
};

// RunSk1ShotScenario: import the authored rig, then for each keyframe time sample the clip (float), snap to
// Q16.16 at the ONE boundary, and integer-FK the imported skeleton into model-space joints. Also runs a
// self-retarget (BuildRetargetMap over the imported skeleton) as the compose-with-AN2 proof.
inline Sk1ShotRun RunSk1ShotScenario() {
    Sk1ShotRun run;
    run.import = ImportUsdSkel(Sk1UsdaAsset(), std::strlen(Sk1UsdaAsset()));
    run.bones = (int32_t)run.import.skeleton.joints.size();
    run.verts = (int32_t)run.import.vertices.size();
    run.clips = (int32_t)run.import.animations.size();
    run.importDigest = DigestUsdSkel(run.import);

    uint64_t hp = 14695981039346656037ull;
    if (run.clips > 0) {
        const anim::Animation& clip = run.import.animations[0];
        for (const anim::Channel& c : clip.channels) run.keyframes += (int32_t)c.times.size();
        const float* times = ShotTimes();
        run.frames.reserve(kShotFrames);
        for (int f = 0; f < kShotFrames; ++f) {
            const std::vector<anim::JointPose> pose = anim::SampleLocalPose(run.import.skeleton, clip, times[f]);
            const std::vector<rtn::FxJointPose> poseQ = rtn::QuantizePose(pose);
            Sk1ShotFrame fr;
            fr.model = rtn::ForwardKinematics(run.import.skeleton, poseQ);
            const uint64_t dm = rtn::DigestModel(fr.model);
            hp = mm::detail::Fnv1a64Word(hp, (uint32_t)(dm & 0xffffffffu));
            hp = mm::detail::Fnv1a64Word(hp, (uint32_t)(dm >> 32));
            run.frames.push_back(std::move(fr));
        }
    }
    run.poseDigest = hp;

    // Compose proof: the imported skeleton drives AN2 retargeting (self-retarget -> reproduces the motion).
    uint64_t hr = 14695981039346656037ull;
    if (run.clips > 0 && run.bones > 0) {
        const std::string root = run.import.jointNames.empty() ? std::string("Root") : run.import.jointNames[0];
        const std::string foot = run.import.jointNames.back();
        const rtn::RetargetMap map = rtn::BuildRetargetMap(run.import.skeleton, run.import.jointNames,
                                                           run.import.skeleton, run.import.jointNames, root, foot);
        const float* times = ShotTimes();
        for (int f = 0; f < kShotFrames; ++f) {
            const std::vector<anim::JointPose> pose = anim::SampleLocalPose(run.import.skeleton,
                                                                            run.import.animations[0], times[f]);
            const std::vector<rtn::FxJointPose> tgt = rtn::Retarget(map, pose);
            const std::vector<rtn::FxJointModel> g = rtn::ForwardKinematics(run.import.skeleton, tgt);
            const uint64_t dm = rtn::DigestModel(g);
            hr = mm::detail::Fnv1a64Word(hr, (uint32_t)(dm & 0xffffffffu));
            hr = mm::detail::Fnv1a64Word(hr, (uint32_t)(dm >> 32));
        }
    }
    run.retargetDigest = hr;

    uint64_t h = 14695981039346656037ull;
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.importDigest & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.importDigest >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.poseDigest & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.poseDigest >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.retargetDigest & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.retargetDigest >> 32));
    run.digest = h;
    return run;
}

// RenderSk1Shot: the SHARED pure-integer raster — strict-zero cross-backend BY CONSTRUCTION. The imported rig
// drawn as an overlaid posed stick figure at each keyframe (bones = line segments between model-space joints);
// later frames brighter so the bend progression reads. 420x360 BGRA8, integer DDA only.
inline void RenderSk1Shot(const Sk1ShotRun& run, std::vector<uint8_t>& bgra, uint32_t& outW, uint32_t& outH) {
    const int W = 420, H = 360;
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {            // deep slate ground
        bgra[p * 4 + 0] = 24; bgra[p * 4 + 1] = 18; bgra[p * 4 + 2] = 14; bgra[p * 4 + 3] = 255;
    }
    auto putPx = [&](int ix, int iy, uint8_t r, uint8_t g, uint8_t b) {
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) return;
        uint8_t* d = &bgra[((size_t)iy * W + ix) * 4];
        d[0] = b; d[1] = g; d[2] = r; d[3] = 255;
    };
    auto line = [&](int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
        const int dx = x1 - x0, dy = y1 - y0;
        int steps = dx >= 0 ? dx : -dx;
        const int ady = dy >= 0 ? dy : -dy;
        if (ady > steps) steps = ady;
        if (steps == 0) { putPx(x0, y0, r, g, b); return; }
        for (int i = 0; i <= steps; ++i) {
            const int px = x0 + (int)(((int64_t)dx * i * 2 + steps) / (2 * (int64_t)steps));
            const int py = y0 + (int)(((int64_t)dy * i * 2 + steps) / (2 * (int64_t)steps));
            putPx(px, py, r, g, b);
        }
    };
    auto disc = [&](int cx, int cy, int rr, uint8_t r, uint8_t g, uint8_t b) {
        for (int dy = -rr; dy <= rr; ++dy)
            for (int dx = -rr; dx <= rr; ++dx)
                if (dx * dx + dy * dy <= rr * rr) putPx(cx + dx, cy + dy, r, g, b);
    };
    const int kScale = 90, kBaseY = 300, kCx = 150;
    auto project = [&](const rtn::FxV3& p, int frame, int& sx, int& sy) {
        sx = kCx + (int)(((int64_t)p.x * kScale) >> rtn::kFrac) + frame * 60;   // fan the frames rightward
        sy = kBaseY - (int)(((int64_t)p.y * kScale) >> rtn::kFrac);
    };
    const int nF = (int)run.frames.size();
    auto shade = [&](int frame, uint8_t base) -> uint8_t {
        const int lo = 90;
        const int v = lo + (base - lo) * (frame + 1) / (nF > 0 ? nF : 1);
        return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    };
    for (int f = 0; f < nF; ++f) {
        const uint8_t r = shade(f, 120), g = shade(f, 200), b = shade(f, 232);   // cyan rig
        const std::vector<rtn::FxJointModel>& gmod = run.frames[(size_t)f].model;
        for (size_t j = 0; j < gmod.size() && j < run.import.skeleton.joints.size(); ++j) {
            int sx, sy; project(gmod[j].pos, f, sx, sy);
            const int parent = run.import.skeleton.joints[j].parent;
            if (parent >= 0 && (size_t)parent < gmod.size()) {
                int px, py; project(gmod[(size_t)parent].pos, f, px, py);
                line(px, py, sx, sy, r, g, b);
            }
            disc(sx, sy, 3, r, g, b);
        }
    }
}

}  // namespace sk1
}  // namespace hf::asset
