#pragma once
// Slice DI — DDGI Slice 2: Probe Radiance Capture — pure CPU (header-only, no device, no backend
// symbols). Namespace hf::render::probecap. The SECOND slice of the GLOBAL ILLUMINATION (DDGI) flagship
// arc (after DH's probe ray-trace). Mirrors cubemap.h (the per-face cube projection math) + probe_gi.h
// (the world-space probe lattice + the probeCount=0 disabled-path idiom): a tiny shared-math header
// ABOVE the RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only mentions of "vk"/"MTL"
// anywhere in this slice's above-seam files are seam-discipline doc comments). The showcase capture loop
// (samples/hello_triangle/main.cpp, metal_headless/visual_test.mm) AND the unit test
// (tests/probe_capture_test.cpp) consume the SAME CaptureFaceCount / ProbeFaceIndex / FaceAverage and the
// SAME cubemap::FaceView/FaceProj — so the GPU capture loop and the CPU test agree on the probe×face slot
// layout and on the 6-face projection.
//
// THE TECHNIQUE (the DDGI radiance-capture data layer): for each probe in a (small) world-space lattice,
// RENDER the lit scene into the 6 faces of a cubemap from the probe center — each face the scene viewed
// through a 90°-FOV square frustum down one of the ±X/±Y/±Z axes (cubemap.h::FaceView/FaceProj) — and
// read the faces back into a per-probe radiance store. This is the raw radiance the SH-encode slice (DJ)
// will convolve into per-probe irradiance. To keep render counts tractable the DI capture grid is SMALL
// (probe_gi.h ProbeGrid{dimX=2,dimY=2,dimZ=2} = 8 probes → 8×6 = 48 face renders, vs DH's 256-probe
// ray-trace grid which is UNTOUCHED). Production DDGI captures incrementally (a few probes/frame); this
// slice captures the small grid in one shot for determinism. The captured scene EXCLUDES any reflective /
// probe-debug surfaces (no recursion) — see the showcase.
//
// THE CAPTURE-CORRECTNESS PROOF (the DD pattern, backend-portable): because each face uses the IDENTICAL
// scene + bake pipeline + push constants + viewport a direct render would, the captured face equals a
// standalone render with that face's FaceView/FaceProj. The showcase renders probe-0's face 0 DIRECTLY
// into a 2D RT and asserts it is BYTE-IDENTICAL (SHA / memcmp) to ReadCubemapFace(probe-0, face 0) — the
// per-probe analog of DD's single-cube capture==direct proof. THE probeCount=0 NO-OP PROOF: dimX==0 →
// probeCount()==0 → CaptureFaceCount()==0 → the capture loop body never runs → the per-probe radiance
// store stays at its cleared value (a byte-identical skip-loop no-op, the probe_gi.h dispatch-0 analog).
//
// CONVENTIONS (must match cubemap.h + probe_gi.h EXACTLY):
//   * The probe grid + its probePos/flatIndex/probeCount come from probe_gi.h (re-used unchanged). The DI
//     capture grid is a SMALL grid; DH's default 256-probe grid is a separate instance.
//   * The 6 faces use cubemap::FaceView(face, probeCenter) + cubemap::FaceProj(zNear,zFar) — the SAME
//     6-axis projection + the SAME +Y/-Y up-vector convention DD's capture uses (here CENTERED on the
//     probe rather than a single fixed probe center). The face order is +X,-X,+Y,-Y,+Z,-Z (cubemap.h).
//   * The radiance store is a FLAT per-probe×face block: slot = ProbeFaceIndex(probe,face) =
//     probe*kFaces + face (probe-major, face-minor) — covering every probe×face exactly once with no
//     overlap. The actual face PIXELS live in the read-back buffer; this header only describes the layout
//     + the indexing + the small debug-viz average.
//
// CROSS-BACKEND FP NOTE (from DH): FaceAverage accumulates the face's texels in a FIXED order and folds
// each texel with an explicit std::fma (a SINGLE correctly-rounded multiply-add — the CPU/Vulkan/Metal
// all agree to the bit, vs a plain a+b*c which Metal contracts to fma and the others do not, a 1-ULP
// divergence). It uses NO transcendentals. The face renders themselves go through the shared lit/bake
// pipeline (already cross-backend-stable, as DD proved capture==direct on both backends).
//
// Pure, deterministic functions: no RNG, no time, no device.

#include "math/math.h"
#include "render/probe_gi.h"   // SHARE the probe grid (ProbeGrid / probePos / probeCount / flatIndex)
#include "render/cubemap.h"    // SHARE the 6-face cube projection (FaceView / FaceProj / FaceViewProj)

#include <cmath>
#include <cstdint>

namespace hf::render::probecap {

// Re-export the SHARED probe grid + cube-face count so the capture loop + the tests write
// probecap::ProbeGrid / probecap::kFaces and the reuse is explicit at the call site (these are the SAME
// types/values the DH ray-trace + the DD capture use — DI does not redefine them).
using probegi::ProbeGrid;
inline constexpr int kFaces = cubemap::kFaces;   // 6 — the cube faces, +X,-X,+Y,-Y,+Z,-Z

// --- The total face-render count (the capture-loop trip count). ------------------------------------
// CaptureFaceCount(grid) = grid.probeCount()*kFaces = the number of (probe,face) renders the capture
// loop performs. EXACTLY 0 when probeCount<=0 (dimX/dimY/dimZ == 0 → probeCount == 0 → 0 face renders →
// the capture loop body never runs → the radiance store is untouched == its cleared value). This is the
// byte-identical no-op the probeCount=0 proof rests on (the probe_gi.h ProbeDispatchGroups(...) → 0
// analog). Pure / deterministic.
inline int CaptureFaceCount(const ProbeGrid& grid) {
    int n = grid.probeCount();
    return (n <= 0) ? 0 : n * kFaces;
}

// --- The probe×face slot index into the flat radiance store. ---------------------------------------
// ProbeFaceIndex(probe, face) = probe*kFaces + face (probe-major, face-minor). The flat radiance store is
// a ProbeFace[probeCount*kFaces] block; probe p face f lives at slot p*kFaces+f. This covers every
// probe×face pair in [0, probeCount*kFaces) exactly once with NO overlap (the round-trip is exact, hand-
// checked by the test). Pure / deterministic.
inline int ProbeFaceIndex(int probe, int face) { return probe * kFaces + face; }

// The inverse of ProbeFaceIndex: recover (probe,face) from a flat slot. slot = probe*kFaces+face →
// probe = slot/kFaces, face = slot%kFaces. Round-trips ProbeFaceIndex for every (probe,face).
inline void ProbeFaceFromIndex(int slot, int& outProbe, int& outFace) {
    outProbe = slot / kFaces;
    outFace  = slot % kFaces;
}

// The per-face view-projection from the probe center (the camera the face's scene render uses). A thin
// reuse of cubemap::FaceViewProj at the probe's world position — the SAME 6-axis projection DD's capture
// uses, here centered on the probe. The capture loop + the test both call this for the face render math.
inline math::Mat4 ProbeFaceViewProj(int face, const math::Vec3& probeCenter, float zNear, float zFar) {
    return cubemap::FaceViewProj(face, probeCenter, zNear, zFar);
}

// --- The per-probe captured-radiance record (the read-back face data DESCRIPTOR). ------------------
// ProbeRadiance documents the LAYOUT of one probe's captured cubemap in the host/SSBO radiance store:
// kFaces faces, each faceSize×faceSize, RGBA16F (the HDR cube format the capture writes), face f at
// ProbeFaceIndex(p,f). The actual texels live in the read-back buffer (ReadCubemapFace fills them per
// face); this struct is the size/format/layout contract the next slice (SH encode) consumes. It carries
// NO pixels itself (header-only, pure) — it is the descriptor, mirroring how probe_gi.h's ProbeRayHits
// documents the per-probe ray block stride without owning the SSBO.
struct ProbeRadiance {
    int   faceSize = 0;       // per-face edge length (square); the captured cube's `size`
    int   faceCount = kFaces; // always 6 (the cube faces)
    // RGBA16F = 8 bytes/texel (4 channels × 2 bytes). One face = faceSize*faceSize*8 bytes; one probe's
    // full cube = kFaces*that. Documented (not allocated here) — the read-back buffer holds the texels.
    static constexpr int kBytesPerTexel = 8;   // RGBA16F
    long faceBytes() const { return (long)faceSize * (long)faceSize * kBytesPerTexel; }
    long probeBytes() const { return faceBytes() * (long)faceCount; }
};

// --- The debug-viz face-average radiance (deterministic). ------------------------------------------
// FaceAverage(rgba, texelCount) folds `texelCount` RGBA float4 texels of one captured face into a single
// average radiance (the per-probe/per-face debug-viz value: a colored sphere/swatch at the probe). The
// accumulation is in a FIXED texel order and each fold is an explicit std::fma(texel, w, acc) with the
// reciprocal weight w = 1/texelCount precomputed once — a SINGLE correctly-rounded multiply-add per
// texel, so the CPU result matches a GPU reduction folding the SAME order with mad() to the bit (the DH
// cross-backend discipline). `rgba` is a flat float[texelCount*4] (r,g,b,a interleaved). texelCount<=0 →
// the zero vector (an empty face averages to black). Pure / deterministic.
inline math::Vec4 FaceAverage(const float* rgba, int texelCount) {
    if (texelCount <= 0 || rgba == nullptr) return math::Vec4{0.0f, 0.0f, 0.0f, 0.0f};
    const float w = 1.0f / (float)texelCount;   // the reciprocal weight, precomputed ONCE (host float32)
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
    for (int i = 0; i < texelCount; ++i) {
        // acc += texel * w as a single fma (correctly-rounded multiply-add; bit-stable across backends).
        r = std::fma(rgba[i * 4 + 0], w, r);
        g = std::fma(rgba[i * 4 + 1], w, g);
        b = std::fma(rgba[i * 4 + 2], w, b);
        a = std::fma(rgba[i * 4 + 3], w, a);
    }
    return math::Vec4{r, g, b, a};
}

}  // namespace hf::render::probecap
