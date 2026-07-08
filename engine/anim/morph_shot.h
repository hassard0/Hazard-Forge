#pragma once

// Slice MP1 — the SHOWCASE half: a self-contained glTF-style authored morph asset (a face grid with a
// "smile" + a "blink" blend shape + a weight animation track) + the pure-CPU shot scenario both backends run
// (Vulkan --mp1-morph-shot / Metal --mp1-morph) and the shared strict-zero deformed-grid raster. Kept OUT of
// morph.h so the core stays dependency-light; this header composes the imported MorphSet with the EXISTING
// anim FK (retarget.h, AN2/SK1) to PROVE the morph->skin order:
//   BuildMorphSet(asset) -> MorphSet ; BuildWeightTrack(asset) -> WeightTrack
//     -> SampleMorphWeightsFx(track,t) (LINEAR weight sampling, quantized at the ONE boundary)
//     -> ApplyMorphFx(baseQ, targets, weightsQ) (INTEGER Q16.16 blend)          [strict-zero cross-backend]
//     -> SkinPointFx(joint, deformed) (retarget FK palette rotate+translate)    [morph BEFORE skin]
// The blend is integer, so the shared raster + all digests are bit-identical cross-backend BY CONSTRUCTION.
// NO new shader / RHI. Every authored delta/weight is an exact binary fraction (multiples of 1/8) so the
// QuantizeFx inputs are cross-compiler exact and the float/integer blend paths agree.

#include <cstdint>
#include <string>
#include <vector>

#include "anim/morph.h"      // MorphSet/WeightTrack/GltfMorphAsset/BuildMorphSet/BuildWeightTrack/ApplyMorphFx/...
#include "anim/retarget.h"   // FxJointModel/FxQuat/FxQuatRotate/kFrac (the integer FK palette for the compose proof)

namespace hf::anim {
namespace morph {
namespace mp1 {

namespace rtn = hf::anim::retarget;

inline constexpr int kGrid = 7;                 // 7x7 face grid = 49 vertices
inline constexpr int kShotFrames = 5;           // sample the weight track at 5 times across the clip

// ShotTimes: the 5 sample times (seconds) across the [0,1]s clip — neutral -> full smile+blink.
inline const float* ShotTimes() { static const float t[kShotFrames] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}; return t; }

// Mp1MorphAsset: a glTF-style authored morph description (the SK1 authored-literal discipline — no .glb). A
// 7x7 face grid in the XY plane (positions multiples of 1/2), two blend shapes:
//   * "smile" — the two mouth rows (row 1,2) rise at the corners (delta.y = |col-3| * 1/8, exact).
//   * "blink" — the two eye rows (row 4,5) drop at the eye columns ({1,2},{4,5}) (delta.y = -1/4, exact).
// A `weights` animation track (2 targets, 3 keys): 0s [0,0] -> 0.5s [1/2,0] -> 1s [1,1].
inline GltfMorphAsset Mp1MorphAsset() {
    GltfMorphAsset a;
    const int G = kGrid;
    const size_t vCount = (size_t)G * G;
    a.basePositions.resize(vCount * 3);
    std::vector<float> smile(vCount * 3, 0.0f);
    std::vector<float> blink(vCount * 3, 0.0f);
    for (int row = 0; row < G; ++row) {
        for (int col = 0; col < G; ++col) {
            const size_t v = (size_t)row * G + col;
            a.basePositions[v * 3 + 0] = (float)(col - 3) * 0.5f;   // x in [-1.5, 1.5], step 1/2 (exact)
            a.basePositions[v * 3 + 1] = (float)(row - 3) * 0.5f;   // y in [-1.5, 1.5]
            a.basePositions[v * 3 + 2] = 0.0f;
            // smile: mouth rows 1,2 rise at the corners (|col-3| eighths of a unit — exact binary fraction).
            if (row == 1 || row == 2) {
                const int dc = col - 3;
                smile[v * 3 + 1] = (float)(dc < 0 ? -dc : dc) * 0.125f;
            }
            // blink: eye rows 4,5 at eye columns {1,2}(left) and {4,5}(right) drop 1/4 unit.
            if ((row == 4 || row == 5) && (col == 1 || col == 2 || col == 4 || col == 5)) {
                blink[v * 3 + 1] = -0.25f;
            }
        }
    }
    a.targetDeltas = {smile, blink};
    a.targetNames  = {"smile", "blink"};
    a.defaultWeights = {0.0f, 0.0f};
    a.weightTimes  = {0.0f, 0.5f, 1.0f};
    a.weightValues = {0.0f, 0.0f,   1.0f, 0.0f,   1.0f, 1.0f};   // K*targetCount (glTF weights output)
    return a;
}

// SkinPointFx: skin a (morphed) vertex by one joint's model transform — rotate by the joint orientation
// then translate by the joint position (a rigid palette transform). The morph->skin COMPOSE primitive.
inline rtn::FxV3 SkinPointFx(const rtn::FxJointModel& joint, const rtn::FxV3& p) {
    const rtn::FxV3 r = rtn::FxQuatRotate(joint.rot, p);
    return rtn::FxV3{joint.pos.x + r.x, joint.pos.y + r.y, joint.pos.z + r.z};
}

// The compose-witness joint: a head bone rotated 180 deg about Z (exact-unit quat (0,0,1,0)) + translated
// (+2,0,0). 180-about-Z maps (x,y,z)->(-x,-y,z) EXACTLY (integer), so morph->skin vs skin->morph differ in a
// PINNED, exactly-representable way (proves the order matters whenever the bone rotates).
inline rtn::FxJointModel ComposeJoint() {
    rtn::FxJointModel j;
    j.rot = rtn::FxQuat{0, 0, rtn::kOne, 0};                       // 180 deg about Z
    j.pos = rtn::FxV3{2 * rtn::kOne, 0, 0};                        // +2 on X
    return j;
}
// The compose-witness base vertex + its morph delta (target 0 == "smile" scaled arbitrarily for the witness):
// a dedicated (1,0,0) vertex with a (0, 1/2, 0) delta at full weight -> deformed (1, 1/2, 0).
inline rtn::FxV3 ComposeBaseQ()  { return rtn::FxV3{rtn::kOne, 0, 0}; }
inline rtn::FxV3 ComposeDeltaQ() { return rtn::FxV3{0, rtn::kOne / 2, 0}; }

struct Mp1ShotFrame {
    std::vector<morph::fx> weightsQ;              // sampled weights (Q16.16) at this frame
    std::vector<morph::FxV3> deformed;            // morph-blended grid (Q16.16)
};

struct Mp1ShotRun {
    MorphSet                  morphSet;
    WeightTrack               track;
    std::vector<Mp1ShotFrame> frames;
    int32_t  verts      = 0;
    int32_t  targets    = 0;
    int32_t  weightKeys = 0;
    uint64_t importDigest  = 0;                   // DigestMorphSet folded with DigestWeightTrack
    uint64_t blendDigest   = 0;                   // the deformed grids over the frames
    uint64_t composeDigest = 0;                   // the morph->skin composed witness over the frames
    uint64_t digest        = 0;                   // combined two-run comparison currency
    // compose-order witness at FULL weight (target 0 weight 1): morph-then-skin vs skin-then-morph.
    morph::FxV3 morphThenSkin{};
    morph::FxV3 skinThenMorph{};
};

// RunMp1ShotScenario: import the authored morph asset, then for each sample time compute the weights (LINEAR,
// quantized) and the integer morph-blended grid; also compose a rigged witness vertex morph-THEN-skin at
// each frame's target-0 weight, and record the full-weight order witness. All integer -> strict-zero.
inline Mp1ShotRun RunMp1ShotScenario() {
    Mp1ShotRun run;
    const GltfMorphAsset asset = Mp1MorphAsset();
    run.morphSet = BuildMorphSet(asset);
    run.track    = BuildWeightTrack(asset);
    run.verts      = (int32_t)run.morphSet.basePositions.size();
    run.targets    = (int32_t)run.morphSet.targets.size();
    run.weightKeys = (int32_t)run.track.times.size();
    run.importDigest = DigestMorphSet(run.morphSet) ^ DigestWeightTrack(run.track);

    const std::vector<morph::FxV3> baseQ = QuantizeVerts(run.morphSet.basePositions);
    const rtn::FxJointModel joint = ComposeJoint();

    uint64_t hb = 14695981039346656037ull;   // blend digest
    uint64_t hc = 14695981039346656037ull;   // compose digest
    const float* times = ShotTimes();
    run.frames.reserve((size_t)kShotFrames);
    for (int f = 0; f < kShotFrames; ++f) {
        Mp1ShotFrame fr;
        fr.weightsQ = SampleMorphWeightsFx(run.track, times[f]);
        fr.deformed = ApplyMorphFx(baseQ, run.morphSet.targets, fr.weightsQ);
        const uint64_t db = DigestVertsFx(fr.deformed);
        hb = mm::detail::Fnv1a64Word(hb, (uint32_t)(db & 0xffffffffu));
        hb = mm::detail::Fnv1a64Word(hb, (uint32_t)(db >> 32));
        // compose the witness vertex morph-THEN-skin at this frame's target-0 weight (proves the pipeline).
        const morph::fx w0 = fr.weightsQ.empty() ? 0 : fr.weightsQ[0];
        const rtn::FxV3 base = ComposeBaseQ();
        const rtn::FxV3 delta = ComposeDeltaQ();
        const rtn::FxV3 morphed{base.x + fxmul(w0, delta.x), base.y + fxmul(w0, delta.y),
                                base.z + fxmul(w0, delta.z)};
        const rtn::FxV3 composed = SkinPointFx(joint, morphed);
        hc = mm::detail::Fnv1a64Word(hc, (uint32_t)composed.x);
        hc = mm::detail::Fnv1a64Word(hc, (uint32_t)composed.y);
        hc = mm::detail::Fnv1a64Word(hc, (uint32_t)composed.z);
        run.frames.push_back(std::move(fr));
    }
    run.blendDigest = hb;
    run.composeDigest = hc;

    // The FULL-weight order witness: morph-then-skin (correct) vs skin-then-morph (wrong) at weight 1.
    {
        const rtn::FxV3 base = ComposeBaseQ();
        const rtn::FxV3 delta = ComposeDeltaQ();
        const rtn::FxV3 morphed{base.x + delta.x, base.y + delta.y, base.z + delta.z};
        run.morphThenSkin = SkinPointFx(joint, morphed);
        const rtn::FxV3 skinnedBase = SkinPointFx(joint, base);
        run.skinThenMorph = rtn::FxV3{skinnedBase.x + delta.x, skinnedBase.y + delta.y, skinnedBase.z + delta.z};
    }

    uint64_t h = 14695981039346656037ull;
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.importDigest & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.importDigest >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.blendDigest & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.blendDigest >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.composeDigest & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(run.composeDigest >> 32));
    run.digest = h;
    return run;
}

// RenderMp1Shot: the SHARED pure-integer raster — strict-zero cross-backend BY CONSTRUCTION. The deformed
// face grid drawn at each of the 5 sample settings side by side (wireframe edges between adjacent grid verts
// + a dot per vertex; brighter left-to-right so the neutral->smile->blink progression reads), plus a bottom
// WEIGHT-TIMELINE strip: two colored polylines (smile amber, blink cyan) rising from 0 to 1 across the
// frames. 700x420 BGRA8, integer DDA only.
inline void RenderMp1Shot(const Mp1ShotRun& run, std::vector<uint8_t>& bgra, uint32_t& outW, uint32_t& outH) {
    const int W = 700, H = 420;
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {   // deep slate ground
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
    const int G = kGrid;
    const int kScale = 40;                        // px per unit
    const int panelW = W / kShotFrames;           // 140
    const int gridBaseY = 190;                    // grid vertical center
    auto project = [&](const morph::FxV3& p, int panel, int& sx, int& sy) {
        const int cx = panel * panelW + panelW / 2;
        sx = cx + (int)(((int64_t)p.x * kScale) >> kFrac);
        sy = gridBaseY - (int)(((int64_t)p.y * kScale) >> kFrac);
    };
    const int nF = (int)run.frames.size();
    auto shade = [&](int frame, uint8_t base) -> uint8_t {
        const int lo = 110;
        const int v = lo + (base - lo) * (frame + 1) / (nF > 0 ? nF : 1);
        return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    };
    for (int f = 0; f < nF; ++f) {
        const std::vector<morph::FxV3>& g = run.frames[(size_t)f].deformed;
        const uint8_t r = shade(f, 236), gg = shade(f, 214), b = shade(f, 150);   // warm face
        for (int row = 0; row < G; ++row) {
            for (int col = 0; col < G; ++col) {
                const size_t v = (size_t)row * G + col;
                if (v >= g.size()) continue;
                int sx, sy; project(g[v], f, sx, sy);
                if (col + 1 < G) {                 // edge to right neighbor
                    int nx, ny; project(g[v + 1], f, nx, ny);
                    line(sx, sy, nx, ny, r, gg, b);
                }
                if (row + 1 < G) {                 // edge to up neighbor
                    int nx, ny; project(g[v + (size_t)G], f, nx, ny);
                    line(sx, sy, nx, ny, r, gg, b);
                }
            }
        }
        for (size_t v = 0; v < g.size(); ++v) { int sx, sy; project(g[v], f, sx, sy); disc(sx, sy, 1, r, gg, b); }
    }
    // Weight-timeline strip (bottom): two polylines over the frames, smile amber + blink cyan.
    const int stripTop = 340, stripBot = 404, stripH = stripBot - stripTop;
    for (int x = 30; x < W - 30; ++x) { putPx(x, stripBot, 70, 78, 90); putPx(x, stripTop, 46, 52, 62); }
    auto weightAt = [&](int frame, int target) -> morph::fx {
        const std::vector<morph::fx>& w = run.frames[(size_t)frame].weightsQ;
        return (target < (int)w.size()) ? w[(size_t)target] : 0;
    };
    auto stripX = [&](int frame) { return 40 + frame * ((W - 80) / (nF > 1 ? nF - 1 : 1)); };
    auto stripY = [&](morph::fx wq) {   // Q16.16 weight (0..kOne) -> strip Y (bottom==0, top==1)
        int64_t up = ((int64_t)wq * stripH) >> kFrac;
        if (up < 0) up = 0; if (up > stripH) up = stripH;
        return stripBot - (int)up;
    };
    for (int t = 0; t < run.targets && t < 2; ++t) {
        const uint8_t r = t == 0 ? 236 : 96, g = t == 0 ? 190 : 200, b = t == 0 ? 96 : 232;   // smile amber / blink cyan
        for (int f = 0; f + 1 < nF; ++f) {
            const int x0 = stripX(f), y0 = stripY(weightAt(f, t));
            const int x1 = stripX(f + 1), y1 = stripY(weightAt(f + 1, t));
            line(x0, y0, x1, y1, r, g, b);
            disc(x0, y0, 2, r, g, b);
        }
        disc(stripX(nF - 1), stripY(weightAt(nF - 1, t)), 2, r, g, b);
    }
}

}  // namespace mp1
}  // namespace morph
}  // namespace hf::anim
