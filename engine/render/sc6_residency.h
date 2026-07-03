#pragma once
// Slice SC6 — TEXTURE RESIDENCY: VT PAGE FEEDBACK DRIVING A STREAMING BUDGET ON REAL CONTENT
// (docs/GAP_CLOSING_ROADMAP.md Tier 2, the last Tier-2 slice). Pure CPU (header-only, no device, no
// backend symbols), namespace hf::render::sc6 — the same composition-header pattern as sc3_stack.h /
// sc5_foliage.h. vt.h and scene/streaming.h are byte-UNTOUCHED: SC6 composes vt.h's proven page-table
// math (VtTexture / PageId / SelectMipLevel / SnapRequest — the VT1 feedback convention) with the
// budget + hysteresis STREAMING DISCIPLINE proven in scene/streaming.h (StreamingWorld: per-frame
// load/unload budgets, a hysteresis band so boundary flicker does not thrash, deterministic
// nearest-first ordering), applied to TEXTURE PAGES instead of scene cells.
//
// WHY: the virtual-texturing page machinery (VT1-VT5) and the distance-streaming budget (Slice BD)
// had each been proven in ISOLATION — the page math on fixed synthetic request sets, the budget on
// procedural scene cells. They had never been CONNECTED, and no camera path over real content had
// ever driven page residency under a budget. SC6 wires them end to end:
//
//   camera path over a textured scene  ->  per-frame VT page FEEDBACK (the vt.h convention:
//   UV footprint + texel density -> SelectMipLevel -> SnapRequest page coords)  ->  a page
//   RESIDENCY MANAGER (StepResidency) that loads/evicts under a per-frame budget with
//   K-frame hysteresis  ->  a pinned, deterministic residency TRACE.
//
// THE RESIDENCY MANAGER (StepResidency — the streaming.h discipline for pages):
//   * per-frame budgets: at most loadBudgetPerFrame pages load and evictBudgetPerFrame pages evict
//     per frame (the StreamingWorld loadBudgetPerFrame/unloadBudgetPerFrame analog);
//   * HYSTERESIS: a resident page only becomes evictable after it has gone UN-REQUESTED for
//     hysteresisFrames consecutive frames (the time-domain analog of streaming.h's loadRadius <
//     unloadRadius spatial band) — a page flickering in/out of view between consecutive frames is
//     never evicted, so it never thrashes;
//   * POOL CAP: the resident set NEVER exceeds poolCapacity (the finite physical tile pool — the
//     VtTilePool analog); when the pool is full, loads WAIT until hysteresis frees pages;
//   * DETERMINISTIC ORDERING: requested pages load in (mip asc, tex asc, py, px) order — finest mip
//     first, the ascending-pageId priority vt.h's AllocatePhysicalTiles established; evictions go
//     oldest-un-requested-first (ascending lastRequestedFrame), ties broken by ascending global page
//     id. Evictions are processed BEFORE loads each frame (freed slots are usable the same frame).
//   The step is a pure function of (prior state, this frame's feedback, config) — bit-stable.
//
// THE FEEDBACK (two tiers, both from the camera math on the CPU — the vt.h VT1 convention where the
// host computes integer (mip,px,py) page coords via SnapRequest/SelectMipLevel):
//   * SYNTHETIC (always-on, asset-free): a textured ground plane fly-over. Per frame, one ray per
//     SCREEN TILE through a fixed analytic camera onto the plane -> the hit UV; the texel density is
//     the UV footprint between adjacent tile rays scaled to texels-per-pixel -> SelectMipLevel ->
//     SnapRequest. This tier feeds the pinned trace digest and the --sc6-residency-shot heatmap.
//   * MESH (the Sponza tier, asset-gated in the test): per visible TRIANGLE of a real instanced
//     scene, the clip-space screen area vs the UV area in texels gives the density (the standard
//     object-space VT feedback approximation) -> SelectMipLevel -> the triangle's wrapped UV AABB
//     marks pages in ITS texture's page space (each real texture gets its own VtTexture page
//     pyramid inside one global page id space). No z-buffer: all frustum-visible triangles feed
//     (conservative over-request — the honest, documented v1 fidelity bound).
//
// THE DETERMINISM CRUX (MSVC == clang-x64 == clang-arm64, the strict-zero showcase bar): the feedback
// path is float (camera math) but uses ONLY IEEE-exactly-rounded ops (+ - * / sqrt floor fabs) and
// NO libm transcendentals (the perspective tan is a HOST-BAKED constant), and every multiply-add is
// SPLIT INTO SEPARATE STATEMENTS so FP_CONTRACT=on (clang's default; fuses only within one
// expression) can never fuse an a*b+c into an fma — the sc5_foliage.h lesson, taken one step
// further so the pin holds on arm64 too. Everything AFTER the float->int page quantization is pure
// integer, and the pinned artifact (the residency trace digest) is integers only.
//
// Shared by THREE call sites (the established composition discipline):
//   1. tests/sc6_residency_test.cpp — StepResidency semantics + the synthetic-tier pins + the
//      budget/hysteresis/cap proofs (+ the Sponza tier, skip-gated on the fetched asset).
//   2. samples/hello_triangle/main.cpp (--sc6-residency-shot, Vulkan BMP) — the residency HEATMAP
//      (page grid colored by state at the fixed mid-path frame) + the pinned stat line.
//   3. metal_headless/visual_test.mm (--sc6-residency-shot, Metal PNG) — the IDENTICAL pure-CPU
//      pipeline + heatmap (strict-zero cross-backend BY CONSTRUCTION).
//
// SEAM DISCIPLINE: ZERO backend symbols. render/vt.h + scene/streaming.h byte-UNTOUCHED.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "math/math.h"
#include "net/session.h"   // hf::net::DigestBytes (FNV-1a-64, the pinned-golden currency)
#include "render/vt.h"     // VtTexture / PageId / SelectMipLevel / SampleRequest / SnapRequest
#include "scene/vertex.h"  // scene::Vertex (pos + uv) for the mesh-feedback tier

namespace hf::render::sc6 {

// ============================ The page SPACE: N real textures, one global id space ==================
// Each real texture gets its own vt::VtTexture mip pyramid; global page id = base[tex] + local
// vt::PageId. base is the host prefix-sum over per-texture pageCount() (the same prefix-sum shape
// vt.h uses across mips). The synthetic tier is simply textureCount == 1.
struct Sc6PageSpace {
    std::vector<vt::VtTexture> textures;
    std::vector<int>           base;        // [textureCount] prefix offsets into the global id space
    int totalPages = 0;
};

// Build a page space of `textureCount` identical `proto` pyramids (the real-content case: every
// texture mapped into the same page-space shape; per-texture heterogeneity is a future refinement).
inline Sc6PageSpace MakeSc6PageSpace(int textureCount, const vt::VtTexture& proto) {
    Sc6PageSpace s;
    s.textures.assign((size_t)textureCount, proto);
    s.base.resize((size_t)textureCount);
    int off = 0;
    for (int t = 0; t < textureCount; ++t) {
        s.base[(size_t)t] = off;
        off += proto.pageCount();
    }
    s.totalPages = off;
    return s;
}

// One page REQUEST from the feedback pass: integer page coords in texture `tex`'s pyramid (the
// SnapRequest convention). The deterministic load-priority key is (mip asc, tex asc, py, px) —
// finest mip first (the vt.h ascending-pageId priority), then texture, then row-major page order.
struct Sc6PageRequest {
    int32_t tex = 0;
    int32_t mip = 0;
    int32_t px = 0;
    int32_t py = 0;
};

inline bool Sc6RequestLess(const Sc6PageRequest& a, const Sc6PageRequest& b) {
    if (a.mip != b.mip) return a.mip < b.mip;
    if (a.tex != b.tex) return a.tex < b.tex;
    if (a.py != b.py)   return a.py < b.py;
    return a.px < b.px;
}
inline bool Sc6RequestEq(const Sc6PageRequest& a, const Sc6PageRequest& b) {
    return a.tex == b.tex && a.mip == b.mip && a.px == b.px && a.py == b.py;
}

// The GLOBAL page id of a request (base[tex] + the vt.h flat PageId within that texture's pyramid).
inline int32_t Sc6GlobalPageId(const Sc6PageRequest& r, const Sc6PageSpace& space) {
    const vt::VtTexture& vtx = space.textures[(size_t)r.tex];
    return (int32_t)(space.base[(size_t)r.tex] + vt::PageId(r.mip, r.px, r.py, vtx));
}

// ============================ The RESIDENCY MANAGER (the streaming.h discipline for pages) =========

struct Sc6ResidencyConfig {
    int loadBudgetPerFrame  = 8;    // max pages that finish loading per frame (streaming.h analog)
    int evictBudgetPerFrame = 8;    // max pages that finish evicting per frame
    int hysteresisFrames    = 6;    // K: a page must be un-requested K consecutive frames before it is
                                    // evictable (clamped to >= 1; K=1 == "no hysteresis" baseline)
    int poolCapacity        = 64;   // hard cap on the resident set (the finite physical pool) —
                                    // chosen BELOW the canonical run's free peak (67) so the pool
                                    // provably sits FULL (peak == cap) and loads stall on hysteresis
};

// One resident page's bookkeeping. The resident array is kept sorted by ascending globalId (the
// assertable set + the deterministic eviction tie-break).
struct Sc6ResidentPage {
    int32_t globalId = 0;
    int32_t lastRequestedFrame = 0;  // the most recent frame this page appeared in the feedback
    int32_t loadedFrame = 0;         // the frame it was loaded (trace/debug)
};

struct Sc6ResidencyState {
    std::vector<Sc6ResidentPage> resident;  // sorted by globalId ascending
    int32_t frame = -1;                     // the last stepped frame (first StepResidency -> 0)
    int64_t totalLoads = 0;
    int64_t totalEvicts = 0;
    int32_t peakResident = 0;
};

// The per-frame step result (the trace record).
struct Sc6StepResult {
    std::vector<int32_t> loaded;   // global ids, in the deterministic load order
    std::vector<int32_t> evicted;  // global ids, in the deterministic evict order
    int32_t requested = 0;         // deduped request count this frame
    int32_t missingAfter = 0;      // requested pages still NOT resident after this frame's loads
    int32_t residentCount = 0;     // resident set size at end of frame
};

// THE STEP. Advance one residency frame under `cfg` for this frame's feedback `requests`:
//   1. frame += 1; dedup+sort the requests by the (mip, tex, py, px) priority key;
//   2. touch: every REQUESTED page that is already resident gets lastRequestedFrame = frame;
//   3. EVICT (before loads — freed slots are usable this frame): candidates are resident pages with
//      frame - lastRequestedFrame >= max(1, hysteresisFrames); order oldest-un-requested-first
//      (ascending lastRequestedFrame, ties by ascending globalId); up to evictBudgetPerFrame;
//   4. LOAD: walk the sorted deduped requests; each page not yet resident loads while both
//      loads < loadBudgetPerFrame AND resident.size() < poolCapacity; loaded pages get
//      lastRequestedFrame = loadedFrame = frame;
//   5. report {loaded, evicted, requested, missingAfter, residentCount} + update the aggregates.
// Pure function of (state, requests, cfg) -> bit-stable, integer end to end.
inline Sc6StepResult StepResidency(Sc6ResidencyState& st,
                                   std::span<const Sc6PageRequest> requests,
                                   const Sc6PageSpace& space,
                                   const Sc6ResidencyConfig& cfg) {
    Sc6StepResult out;
    st.frame += 1;
    const int32_t frame = st.frame;

    // --- 1. Dedup + sort the requests by the deterministic priority key. ---
    std::vector<Sc6PageRequest> req(requests.begin(), requests.end());
    std::sort(req.begin(), req.end(), Sc6RequestLess);
    req.erase(std::unique(req.begin(), req.end(), Sc6RequestEq), req.end());
    out.requested = (int32_t)req.size();

    // The requested global ids (sorted ascending) for the resident-touch + missing accounting.
    std::vector<int32_t> reqIds(req.size());
    for (size_t i = 0; i < req.size(); ++i) reqIds[i] = Sc6GlobalPageId(req[i], space);
    std::vector<int32_t> reqSorted = reqIds;
    std::sort(reqSorted.begin(), reqSorted.end());

    // --- 2. Touch requested resident pages. ---
    auto isRequested = [&](int32_t id) {
        return std::binary_search(reqSorted.begin(), reqSorted.end(), id);
    };
    for (Sc6ResidentPage& p : st.resident)
        if (isRequested(p.globalId)) p.lastRequestedFrame = frame;

    // --- 3. Evictions (hysteresis + budget; BEFORE loads). ---
    const int32_t kHyst = cfg.hysteresisFrames < 1 ? 1 : cfg.hysteresisFrames;
    {
        // Candidate indices into st.resident, ordered oldest-un-requested-first, globalId tie-break.
        std::vector<int32_t> cand;
        for (int32_t i = 0; i < (int32_t)st.resident.size(); ++i) {
            const Sc6ResidentPage& p = st.resident[(size_t)i];
            if (frame - p.lastRequestedFrame >= kHyst) cand.push_back(i);
        }
        std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) {
            const Sc6ResidentPage& pa = st.resident[(size_t)a];
            const Sc6ResidentPage& pb = st.resident[(size_t)b];
            if (pa.lastRequestedFrame != pb.lastRequestedFrame)
                return pa.lastRequestedFrame < pb.lastRequestedFrame;
            return pa.globalId < pb.globalId;  // resident[] is globalId-sorted, but be explicit
        });
        if ((int)cand.size() > cfg.evictBudgetPerFrame) cand.resize((size_t)cfg.evictBudgetPerFrame);
        // Remove the evicted indices (collect ids first, then erase by flag to keep it O(n)).
        if (!cand.empty()) {
            std::vector<uint8_t> kill(st.resident.size(), 0u);
            for (int32_t i : cand) {
                kill[(size_t)i] = 1u;
                out.evicted.push_back(st.resident[(size_t)i].globalId);
            }
            std::vector<Sc6ResidentPage> keep;
            keep.reserve(st.resident.size() - cand.size());
            for (size_t i = 0; i < st.resident.size(); ++i)
                if (!kill[i]) keep.push_back(st.resident[i]);
            st.resident.swap(keep);
            st.totalEvicts += (int64_t)cand.size();
        }
    }

    // --- 4. Loads (priority order + budget + pool cap). ---
    {
        auto isResident = [&](int32_t id) {
            auto it = std::lower_bound(st.resident.begin(), st.resident.end(), id,
                                       [](const Sc6ResidentPage& p, int32_t v) { return p.globalId < v; });
            return it != st.resident.end() && it->globalId == id;
        };
        std::vector<Sc6ResidentPage> newlyLoaded;
        auto isNewlyLoaded = [&](int32_t id) {
            for (const Sc6ResidentPage& p : newlyLoaded)
                if (p.globalId == id) return true;
            return false;
        };
        int loads = 0;
        for (size_t i = 0; i < req.size(); ++i) {
            const int32_t id = reqIds[i];
            if (isResident(id) || isNewlyLoaded(id)) continue;
            if (loads >= cfg.loadBudgetPerFrame) break;  // requests are priority-sorted: stop here
            if ((int)(st.resident.size() + newlyLoaded.size()) >= cfg.poolCapacity) break;
            newlyLoaded.push_back(Sc6ResidentPage{id, frame, frame});
            out.loaded.push_back(id);
            ++loads;
        }
        if (!newlyLoaded.empty()) {
            st.resident.insert(st.resident.end(), newlyLoaded.begin(), newlyLoaded.end());
            std::sort(st.resident.begin(), st.resident.end(),
                      [](const Sc6ResidentPage& a, const Sc6ResidentPage& b) {
                          return a.globalId < b.globalId;
                      });
            st.totalLoads += (int64_t)newlyLoaded.size();
        }
        // Missing accounting (requested pages still not resident).
        int32_t missing = 0;
        for (int32_t id : reqSorted)
            if (!isResident(id)) ++missing;
        out.missingAfter = missing;
    }

    out.residentCount = (int32_t)st.resident.size();
    if (out.residentCount > st.peakResident) st.peakResident = out.residentCount;
    return out;
}

// ============================ The residency TRACE (the pinned artifact) ============================
// A deterministic integer byte stream folded per frame: {frame, loadedCount, loaded ids...,
// evictedCount, evicted ids..., residentCount, missingAfter} — all little-endian u32/i32. The final
// FNV-1a-64 over the stream is THE pin (integers only -> bit-stable everywhere).
struct Sc6Trace {
    std::vector<unsigned char> stream;

    void PutU32(uint32_t v) {
        stream.push_back((unsigned char)(v & 0xFFu));
        stream.push_back((unsigned char)((v >> 8) & 0xFFu));
        stream.push_back((unsigned char)((v >> 16) & 0xFFu));
        stream.push_back((unsigned char)((v >> 24) & 0xFFu));
    }
    void Fold(int32_t frame, const Sc6StepResult& r) {
        PutU32((uint32_t)frame);
        PutU32((uint32_t)r.loaded.size());
        for (int32_t id : r.loaded) PutU32((uint32_t)id);
        PutU32((uint32_t)r.evicted.size());
        for (int32_t id : r.evicted) PutU32((uint32_t)id);
        PutU32((uint32_t)r.residentCount);
        PutU32((uint32_t)r.missingAfter);
    }
    uint64_t Digest() const { return net::DigestBytes(stream.data(), stream.size()); }
};

// Aggregate run stats (the assertable/pinnable summary).
struct Sc6RunStats {
    int32_t pages = 0;            // page-space size
    int32_t peakResident = 0;
    int64_t totalLoads = 0;
    int64_t totalEvicts = 0;
    int32_t convergedAt = -1;     // first frame f with missingAfter==0 for f AND every later frame;
                                  // -1 when the run never converges (missing on the final frame)
    int32_t maxLoadedInFrame = 0;
    bool    capRespected = true;  // resident set <= poolCapacity on EVERY frame
    uint64_t traceDigest = 0;
};

// Optional per-page state capture at one fixed frame (the heatmap's input): flags at END of the
// captured frame, all sized space.totalPages.
struct Sc6FrameCapture {
    std::vector<uint8_t> resident;       // 1 = resident at end of the captured frame
    std::vector<uint8_t> loadedNow;      // 1 = loaded during the captured frame
    std::vector<uint8_t> evictedNow;     // 1 = evicted during the captured frame
    std::vector<uint8_t> requestedNow;   // 1 = in the captured frame's (deduped) feedback
    std::vector<uint8_t> everRequested;  // 1 = requested on any frame <= the captured frame
};

// Run a full residency trace over per-frame feedback. Pure composition: StepResidency per frame,
// fold the trace, gather the stats, optionally capture page states at `captureFrame`.
inline Sc6RunStats Sc6RunTrace(const std::vector<std::vector<Sc6PageRequest>>& frames,
                               const Sc6PageSpace& space, const Sc6ResidencyConfig& cfg,
                               int captureFrame = -1, Sc6FrameCapture* capture = nullptr) {
    Sc6ResidencyState st;
    Sc6Trace trace;
    Sc6RunStats stats;
    stats.pages = space.totalPages;

    std::vector<uint8_t> everReq((size_t)space.totalPages, 0u);
    int32_t lastMissingFrame = -1;

    for (int32_t f = 0; f < (int32_t)frames.size(); ++f) {
        const std::vector<Sc6PageRequest>& req = frames[(size_t)f];
        Sc6StepResult r = StepResidency(st, std::span<const Sc6PageRequest>(req.data(), req.size()),
                                        space, cfg);
        trace.Fold(f, r);
        if ((int32_t)r.loaded.size() > stats.maxLoadedInFrame)
            stats.maxLoadedInFrame = (int32_t)r.loaded.size();
        if (r.residentCount > cfg.poolCapacity) stats.capRespected = false;
        if (r.missingAfter > 0) lastMissingFrame = f;

        for (const Sc6PageRequest& rq : req) {
            const int32_t id = Sc6GlobalPageId(rq, space);
            if (id >= 0 && id < space.totalPages) everReq[(size_t)id] = 1u;
        }

        if (capture && f == captureFrame) {
            capture->resident.assign((size_t)space.totalPages, 0u);
            capture->loadedNow.assign((size_t)space.totalPages, 0u);
            capture->evictedNow.assign((size_t)space.totalPages, 0u);
            capture->requestedNow.assign((size_t)space.totalPages, 0u);
            for (const Sc6ResidentPage& p : st.resident)
                capture->resident[(size_t)p.globalId] = 1u;
            for (int32_t id : r.loaded)  capture->loadedNow[(size_t)id] = 1u;
            for (int32_t id : r.evicted) capture->evictedNow[(size_t)id] = 1u;
            for (const Sc6PageRequest& rq : req) {
                const int32_t id = Sc6GlobalPageId(rq, space);
                if (id >= 0 && id < space.totalPages) capture->requestedNow[(size_t)id] = 1u;
            }
            capture->everRequested = everReq;
        }
    }

    stats.peakResident = st.peakResident;
    stats.totalLoads = st.totalLoads;
    stats.totalEvicts = st.totalEvicts;
    stats.convergedAt = (lastMissingFrame + 1 < (int32_t)frames.size()) ? lastMissingFrame + 1 : -1;
    if (frames.empty()) stats.convergedAt = -1;
    stats.traceDigest = trace.Digest();
    return stats;
}

// ============================ SYNTHETIC feedback: the textured ground-plane FLY-OVER ================
// A fixed analytic camera flies forward/down over a textured plane. One ray per screen tile; the hit
// UV + the texel footprint between adjacent tile rays give (u, v, density) -> vt::SelectMipLevel ->
// vt::SnapRequest (the VT1 convention, verbatim). EVERY multiply-add is split into separate
// statements (the FP-contraction guard — see the header banner) and there are NO transcendentals.
struct Sc6SyntheticConfig {
    vt::VtTexture texture;        // default: the VT1 canonical 4-mip / 16-vpps0 / 340-page pyramid
    int   pathFrames = 72;        // camera-in-motion frames
    int   holdFrames = 24;        // frames held at the final pose (the convergence tail)
    int   tilesX = 64, tilesY = 36;
    float viewportW = 1280.0f, viewportH = 720.0f;
    float planeHalf = 16.0f;      // plane spans [-planeHalf, +planeHalf]^2 in XZ; UV in [0,1]

    // Camera path endpoints (a low diagonal strafe across the plane — chosen so the live working
    // set churns hard: ~25-50 distinct pages/frame across all 4 mips, entering AND leaving view).
    // The eye slides start->end over pathFrames; the analytic view basis is fixed: forward F,
    // right R, up U — chosen constants, no normalize/no trig.
    float eyeStart[3] = {-9.0f, 2.6f, 13.0f};
    float eyeEnd[3]   = {9.0f, 1.5f, -12.0f};
    float F[3] = {0.0f, -0.55f, -1.0f};
    float R[3] = {0.85f, 0.0f, 0.0f};
    float U[3] = {0.0f, 0.48f, 0.0f};

    int totalFrames() const { return pathFrames + holdFrames; }
};

inline Sc6SyntheticConfig Sc6DefaultSynthetic() {
    Sc6SyntheticConfig c;
    c.texture.mipLevels = 4;
    c.texture.pageSize = 128;
    c.texture.virtualPagesPerSideMip0 = 32;   // 1024+256+64+16 = 1360 pages (a 4096²-texel pyramid)
    return c;
}

// Intersect the tile ray (sx, sy in [-1,1]) with the y=0 plane and return the plane UV; false = the
// ray misses (sky or outside the plane). Split-statement float math only.
inline bool Sc6TileRayUv(const Sc6SyntheticConfig& cfg, float ex, float ey, float ez,
                         float sx, float sy, float& u, float& v) {
    float rx = cfg.R[0] * sx;
    float uy = cfg.U[1] * sy;
    float dirX = cfg.F[0] + rx;
    float dirY = cfg.F[1] + uy;
    float dirZ = cfg.F[2];      // R/U have zero z in the canonical basis; keep general form simple
    float uz = cfg.U[2] * sy;
    dirZ = dirZ + uz;
    float rz = cfg.R[2] * sx;
    dirZ = dirZ + rz;
    if (dirY >= -1e-5f) return false;          // parallel/upward: sky
    float t = (0.0f - ey) / dirY;
    float mx = dirX * t;
    float mz = dirZ * t;
    float hx = ex + mx;
    float hz = ez + mz;
    float span = cfg.planeHalf * 2.0f;
    float ox = hx + cfg.planeHalf;
    float oz = hz + cfg.planeHalf;
    u = ox / span;
    v = oz / span;
    return u >= 0.0f && u < 1.0f && v >= 0.0f && v < 1.0f;
}

// Per-frame synthetic feedback: one request per screen tile that hits the plane, at the mip the
// tile's texel density selects. Density = texels spanned per PIXEL, estimated by forward-differencing
// the hit UV against the next tile in x and in y (max of the two axes; no sqrt needed).
inline std::vector<Sc6PageRequest> Sc6GroundPlaneFeedback(int frame, const Sc6SyntheticConfig& cfg) {
    std::vector<Sc6PageRequest> out;
    const int fClamped = frame < cfg.pathFrames ? frame : cfg.pathFrames - 1;
    const float denom = (float)(cfg.pathFrames - 1);
    const float t = (float)fClamped / denom;

    float dxe = cfg.eyeEnd[0] - cfg.eyeStart[0];
    float dye = cfg.eyeEnd[1] - cfg.eyeStart[1];
    float dze = cfg.eyeEnd[2] - cfg.eyeStart[2];
    float mxe = dxe * t;
    float mye = dye * t;
    float mze = dze * t;
    float ex = cfg.eyeStart[0] + mxe;
    float ey = cfg.eyeStart[1] + mye;
    float ez = cfg.eyeStart[2] + mze;

    const float texels = (float)(cfg.texture.pagesPerSide(0) * cfg.texture.pageSize);
    const float pixPerTileX = cfg.viewportW / (float)cfg.tilesX;
    const float pixPerTileY = cfg.viewportH / (float)cfg.tilesY;
    const float stepX = 2.0f / (float)cfg.tilesX;
    const float stepY = 2.0f / (float)cfg.tilesY;

    for (int ty = 0; ty < cfg.tilesY; ++ty) {
        for (int tx = 0; tx < cfg.tilesX; ++tx) {
            float cxs = (float)tx + 0.5f;
            float cys = (float)ty + 0.5f;
            float sxn = cxs * stepX;
            float syn = cys * stepY;
            float sx = sxn - 1.0f;
            float sy = 1.0f - syn;   // top row looks UP the view
            float u0, v0;
            if (!Sc6TileRayUv(cfg, ex, ey, ez, sx, sy, u0, v0)) continue;

            // Forward-difference footprints (the adjacent tile rays; misses fall back to this hit).
            float sxr = sx + stepX;
            float syd = sy - stepY;
            float u1, v1, u2, v2;
            bool okX = Sc6TileRayUv(cfg, ex, ey, ez, sxr, sy, u1, v1);
            bool okY = Sc6TileRayUv(cfg, ex, ey, ez, sx, syd, u2, v2);
            if (!okX) { u1 = u0; v1 = v0; }
            if (!okY) { u2 = u0; v2 = v0; }
            float dux = u1 - u0;
            float dvx = v1 - v0;
            float duy = u2 - u0;
            float dvy = v2 - v0;
            float adux = std::fabs(dux);
            float advx = std::fabs(dvx);
            float aduy = std::fabs(duy);
            float advy = std::fabs(dvy);
            float spanUx = adux > advx ? adux : advx;
            float spanUy = aduy > advy ? aduy : advy;
            float texX = spanUx * texels;
            float texY = spanUy * texels;
            float densX = texX / pixPerTileX;
            float densY = texY / pixPerTileY;
            float density = densX > densY ? densX : densY;

            const int mip = vt::SelectMipLevel(density, cfg.texture);
            const vt::SnappedRequest sn = vt::SnapRequest(vt::SampleRequest{u0, v0, mip}, cfg.texture);
            out.push_back(Sc6PageRequest{0, sn.mip, sn.px, sn.py});
        }
    }
    return out;
}

// Build the whole synthetic per-frame feedback sequence (the Sc6RunTrace input).
inline std::vector<std::vector<Sc6PageRequest>> Sc6SyntheticFeedback(const Sc6SyntheticConfig& cfg) {
    std::vector<std::vector<Sc6PageRequest>> frames;
    frames.reserve((size_t)cfg.totalFrames());
    for (int f = 0; f < cfg.totalFrames(); ++f)
        frames.push_back(Sc6GroundPlaneFeedback(f, cfg));
    return frames;
}

// ============================ MESH feedback: real instanced content (the Sponza tier) ==============
// Object-space VT feedback: per visible triangle, the clip-space SCREEN area vs the UV area in
// TEXELS gives the density (texels per pixel), SelectMipLevel picks the mip, and the triangle's
// wrapped UV AABB marks pages in its texture's pyramid. Conservative: no z-buffer (all
// frustum-visible triangles feed), AABB (not exact) UV coverage, whole-mip mark when the UV span
// wraps a full tile period. Split-statement float discipline throughout.

struct Sc6MeshView {
    std::span<const scene::Vertex> verts;
    std::span<const uint32_t>      indices;
};
struct Sc6MeshInstance {
    uint32_t   meshIndex = 0;
    int32_t    tex = -1;          // texture slot in the page space; -1 = untextured (skipped)
    math::Mat4 world = math::Mat4::Identity();
};

// Split-statement column-major mat4 * vec4(p,1) (clip-space transform; no fma-fusable expressions).
inline void Sc6TransformPoint(const float* m /*[16] column-major*/, const float* p /*[3]*/,
                              float& cx, float& cy, float& cz, float& cw) {
    float ax = m[0] * p[0];  float bx = m[4] * p[1];  float gx = m[8] * p[2];
    float sx0 = ax + bx;     float sx1 = sx0 + gx;    cx = sx1 + m[12];
    float ay = m[1] * p[0];  float by = m[5] * p[1];  float gy = m[9] * p[2];
    float sy0 = ay + by;     float sy1 = sy0 + gy;    cy = sy1 + m[13];
    float az = m[2] * p[0];  float bz = m[6] * p[1];  float gz = m[10] * p[2];
    float sz0 = az + bz;     float sz1 = sz0 + gz;    cz = sz1 + m[14];
    float aw = m[3] * p[0];  float bw = m[7] * p[1];  float gw = m[11] * p[2];
    float sw0 = aw + bw;     float sw1 = sw0 + gw;    cw = sw1 + m[15];
}

// Split-statement column-major mat4 * mat4 (viewProj composition without math::operator*'s
// single-expression dot products).
inline void Sc6MulMat4(const float* a, const float* b, float* out /*[16]*/) {
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            float t0 = a[0 * 4 + r] * b[c * 4 + 0];
            float t1 = a[1 * 4 + r] * b[c * 4 + 1];
            float t2 = a[2 * 4 + r] * b[c * 4 + 2];
            float t3 = a[3 * 4 + r] * b[c * 4 + 3];
            float s0 = t0 + t1;
            float s1 = s0 + t2;
            out[c * 4 + r] = s1 + t3;
        }
    }
}

// Split-statement LookAt (the math::Mat4::LookAt convention: row0=s, row1=u, row2=-f) built with
// exactly-rounded ops only (sqrt is IEEE-exact; no trig).
inline void Sc6LookAt(const float* eye, const float* center, float* out /*[16] column-major*/) {
    // f = normalize(center - eye)  (up is fixed +Y — the engine's camera convention here)
    float fx = center[0] - eye[0];
    float fy = center[1] - eye[1];
    float fz = center[2] - eye[2];
    float fx2 = fx * fx;
    float fy2 = fy * fy;
    float fz2 = fz * fz;
    float fs0 = fx2 + fy2;
    float fs1 = fs0 + fz2;
    float flen = std::sqrt(fs1);
    fx = fx / flen; fy = fy / flen; fz = fz / flen;
    // s = normalize(cross(f, up=+Y)) = normalize({-fz, 0, fx})
    float sx = -fz;
    float sz = fx;
    float sx2 = sx * sx;
    float sz2 = sz * sz;
    float ss = sx2 + sz2;
    float slen = std::sqrt(ss);
    sx = sx / slen; sz = sz / slen;
    // u = cross(s, f) = {sz*fy*(-1)... } computed componentwise: u = cross({sx,0,sz}, {fx,fy,fz})
    float ux0 = 0.0f * fz;   float ux1 = sz * fy;   float ux = ux0 - ux1;
    float uy0 = sz * fx;     float uy1 = sx * fz;   float uy = uy0 - uy1;
    float uz0 = sx * fy;     float uz1 = 0.0f * fx; float uz = uz0 - uz1;
    for (int i = 0; i < 16; ++i) out[i] = 0.0f;
    out[15] = 1.0f;
    out[0] = sx;  out[4] = 0.0f; out[8]  = sz;
    out[1] = ux;  out[5] = uy;   out[9]  = uz;
    out[2] = -fx; out[6] = -fy;  out[10] = -fz;
    // translation: -dot(s,eye), -dot(u,eye), dot(f,eye) — split statements
    float d0 = sx * eye[0];  float d1 = sz * eye[2];  float ds = d0 + d1;
    out[12] = -ds;
    float e0 = ux * eye[0];  float e1 = uy * eye[1];  float e2 = uz * eye[2];
    float es0 = e0 + e1;     float es = es0 + e2;
    out[13] = -es;
    float g0 = fx * eye[0];  float g1 = fy * eye[1];  float g2 = fz * eye[2];
    float gs0 = g0 + g1;     float gs = gs0 + g2;
    out[14] = gs;
}

// Split-statement perspective with a HOST-BAKED tan(fovY/2) constant (the math::Mat4::Perspective
// element layout, Vulkan Y-flip included) — NO transcendental at runtime.
inline void Sc6Perspective(float tanHalfFovY, float aspect, float zNear, float zFar,
                           float* out /*[16]*/) {
    for (int i = 0; i < 16; ++i) out[i] = 0.0f;
    float at = aspect * tanHalfFovY;
    out[0] = 1.0f / at;
    out[5] = -1.0f / tanHalfFovY;
    float nf = zNear - zFar;
    out[10] = zFar / nf;
    out[11] = -1.0f;
    float nz = zNear * zFar;
    out[14] = nz / nf;
}

// Mark the wrapped UV-AABB pages of one triangle at `mip` into `out`. A span >= one full UV period
// marks the whole mip row/column (the wrap-heavy floor-tile case, documented conservative).
inline void Sc6MarkUvAabb(float u0, float u1, float v0, float v1, int tex, int mip,
                          const vt::VtTexture& vtx, std::vector<Sc6PageRequest>& out) {
    const int pps = vtx.pagesPerSide(mip);
    const float spanU = u1 - u0;
    const float spanV = v1 - v0;
    int pxFirst = 0, pxCount = pps;
    if (spanU < 1.0f) {
        float wu = u0 - std::floor(u0);
        float fu = wu * (float)pps;
        pxFirst = (int)std::floor(fu);
        float su = spanU * (float)pps;
        pxCount = (int)std::floor(su) + 2;   // conservative: start partial + end partial
        if (pxCount > pps) pxCount = pps;
    }
    int pyFirst = 0, pyCount = pps;
    if (spanV < 1.0f) {
        float wv = v0 - std::floor(v0);
        float fv = wv * (float)pps;
        pyFirst = (int)std::floor(fv);
        float sv = spanV * (float)pps;
        pyCount = (int)std::floor(sv) + 2;
        if (pyCount > pps) pyCount = pps;
    }
    if (pxFirst < 0) pxFirst = 0;
    if (pxFirst > pps - 1) pxFirst = pps - 1;
    if (pyFirst < 0) pyFirst = 0;
    if (pyFirst > pps - 1) pyFirst = pps - 1;
    for (int j = 0; j < pyCount; ++j) {
        const int py = (pyFirst + j) % pps;
        for (int i = 0; i < pxCount; ++i) {
            const int px = (pxFirst + i) % pps;
            out.push_back(Sc6PageRequest{tex, mip, px, py});
        }
    }
}

// Per-frame mesh feedback over the instanced scene from `viewProj` (column-major, the engine
// convention). Requests are RAW (duplicates expected); StepResidency dedups deterministically.
inline std::vector<Sc6PageRequest> Sc6MeshFeedback(std::span<const Sc6MeshView> meshes,
                                                   std::span<const Sc6MeshInstance> instances,
                                                   const float* viewProj /*[16]*/,
                                                   float viewportW, float viewportH,
                                                   const Sc6PageSpace& space) {
    std::vector<Sc6PageRequest> out;
    const float kWEps = 0.05f;   // conservative near-clip reject (behind/straddling the camera)

    for (const Sc6MeshInstance& inst : instances) {
        if (inst.tex < 0 || inst.tex >= (int32_t)space.textures.size()) continue;
        if (inst.meshIndex >= meshes.size()) continue;
        const Sc6MeshView& mesh = meshes[inst.meshIndex];
        const vt::VtTexture& vtx = space.textures[(size_t)inst.tex];
        const float texelsW = (float)(vtx.pagesPerSide(0) * vtx.pageSize);

        float wvp[16];
        Sc6MulMat4(viewProj, inst.world.m, wvp);

        const size_t triCount = mesh.indices.size() / 3;
        for (size_t tIdx = 0; tIdx < triCount; ++tIdx) {
            const uint32_t i0 = mesh.indices[3 * tIdx + 0];
            const uint32_t i1 = mesh.indices[3 * tIdx + 1];
            const uint32_t i2 = mesh.indices[3 * tIdx + 2];
            if (i0 >= mesh.verts.size() || i1 >= mesh.verts.size() || i2 >= mesh.verts.size())
                continue;
            const scene::Vertex& a = mesh.verts[i0];
            const scene::Vertex& b = mesh.verts[i1];
            const scene::Vertex& c = mesh.verts[i2];

            float ax, ay, az, aw, bx, by, bz, bw, cx, cy, cz, cw;
            Sc6TransformPoint(wvp, a.pos, ax, ay, az, aw);
            Sc6TransformPoint(wvp, b.pos, bx, by, bz, bw);
            Sc6TransformPoint(wvp, c.pos, cx, cy, cz, cw);
            if (aw <= kWEps || bw <= kWEps || cw <= kWEps) continue;   // conservative near reject

            float nax = ax / aw; float nay = ay / aw;
            float nbx = bx / bw; float nby = by / bw;
            float ncx = cx / cw; float ncy = cy / cw;
            // NDC AABB frustum reject (all three verts beyond one clip edge).
            if ((nax < -1.0f && nbx < -1.0f && ncx < -1.0f) ||
                (nax >  1.0f && nbx >  1.0f && ncx >  1.0f) ||
                (nay < -1.0f && nby < -1.0f && ncy < -1.0f) ||
                (nay >  1.0f && nby >  1.0f && ncy >  1.0f)) continue;

            // Screen-space area in pixels² (0.5 * |2D cross|), split statements.
            float hw = viewportW * 0.5f;
            float hh = viewportH * 0.5f;
            float sax = nax * hw; float say = nay * hh;
            float sbx = nbx * hw; float sby = nby * hh;
            float scx = ncx * hw; float scy = ncy * hh;
            float e1x = sbx - sax; float e1y = sby - say;
            float e2x = scx - sax; float e2y = scy - say;
            float cr0 = e1x * e2y;
            float cr1 = e1y * e2x;
            float cr = cr0 - cr1;
            float screenArea2 = std::fabs(cr);       // 2*area
            if (screenArea2 <= 1e-6f) continue;      // degenerate / edge-on: no reliable footprint

            // UV area in texels² (same 2D cross over the UVs scaled by the texture size).
            float u1e = b.uv[0] - a.uv[0]; float v1e = b.uv[1] - a.uv[1];
            float u2e = c.uv[0] - a.uv[0]; float v2e = c.uv[1] - a.uv[1];
            float uc0 = u1e * v2e;
            float uc1 = v1e * u2e;
            float uc = uc0 - uc1;
            float uvArea2 = std::fabs(uc);
            float t0 = uvArea2 * texelsW;
            float uvTexArea2 = t0 * texelsW;         // 2*area in texels²
            if (uvTexArea2 <= 0.0f) continue;        // no UV footprint: nothing to page in

            float ratio = uvTexArea2 / screenArea2;
            float density = std::sqrt(ratio);        // texels per pixel (IEEE-exact sqrt)
            const int mip = vt::SelectMipLevel(density, vtx);

            float uMin = a.uv[0]; float uMax = a.uv[0];
            if (b.uv[0] < uMin) uMin = b.uv[0];
            if (b.uv[0] > uMax) uMax = b.uv[0];
            if (c.uv[0] < uMin) uMin = c.uv[0];
            if (c.uv[0] > uMax) uMax = c.uv[0];
            float vMin = a.uv[1]; float vMax = a.uv[1];
            if (b.uv[1] < vMin) vMin = b.uv[1];
            if (b.uv[1] > vMax) vMax = b.uv[1];
            if (c.uv[1] < vMin) vMin = c.uv[1];
            if (c.uv[1] > vMax) vMax = c.uv[1];
            Sc6MarkUvAabb(uMin, uMax, vMin, vMax, inst.tex, mip, vtx, out);
        }
    }
    return out;
}

// The fixed mid-path frame the showcase heatmap captures (also the flicker set's source frame) —
// chosen where the trace is at its liveliest (a full-budget load batch + evictions + a pending page).
inline constexpr int kSc6ShotFrame = 54;

// ============================ The HYSTERESIS THRASH CONTRAST (the load-bearing-K proof) ============
// A page set flickering in/out of view every other frame (the worst case the hysteresis band
// exists for): feedback alternates {R, {}, R, {}, ...} where R is the canonical synthetic scenario's
// kSc6ShotFrame request set. With K >= 2 the one-frame gap never reaches the eviction threshold, so
// every page loads EXACTLY ONCE (zero thrash); with the K=1 baseline (evict as soon as one frame
// un-requested) every gap evicts and every re-appearance reloads. Budgets are unthrottled here so
// the contrast is purely the hysteresis (the budget proof is separate).
struct Sc6ThrashStats {
    int64_t loadsHyst = 0, evictsHyst = 0;    // K = the canonical hysteresisFrames
    int64_t loadsBase = 0, evictsBase = 0;    // K = 1 (no hysteresis)
    int32_t flickerPages = 0;                 // |R| (deduped)
};

inline Sc6ThrashStats Sc6ThrashContrast(const Sc6SyntheticConfig& scfg, int hystFrames,
                                        int flickerFrames = 32) {
    std::vector<Sc6PageRequest> r = Sc6GroundPlaneFeedback(kSc6ShotFrame, scfg);
    const Sc6PageSpace space = MakeSc6PageSpace(1, scfg.texture);
    std::sort(r.begin(), r.end(), Sc6RequestLess);
    r.erase(std::unique(r.begin(), r.end(), Sc6RequestEq), r.end());
    std::vector<std::vector<Sc6PageRequest>> frames;
    frames.reserve((size_t)flickerFrames);
    for (int f = 0; f < flickerFrames; ++f)
        frames.push_back((f % 2 == 0) ? r : std::vector<Sc6PageRequest>{});

    Sc6ResidencyConfig unthrottled;
    unthrottled.loadBudgetPerFrame = 1 << 20;
    unthrottled.evictBudgetPerFrame = 1 << 20;
    unthrottled.poolCapacity = space.totalPages;

    Sc6ThrashStats out;
    {
        Sc6ResidencyConfig cfg = unthrottled;
        cfg.hysteresisFrames = hystFrames;
        Sc6RunStats s = Sc6RunTrace(frames, space, cfg);
        out.loadsHyst = s.totalLoads;
        out.evictsHyst = s.totalEvicts;
    }
    {
        Sc6ResidencyConfig cfg = unthrottled;
        cfg.hysteresisFrames = 1;
        Sc6RunStats s = Sc6RunTrace(frames, space, cfg);
        out.loadsBase = s.totalLoads;
        out.evictsBase = s.totalEvicts;
    }
    out.flickerPages = (int32_t)r.size();
    return out;
}

// ============================ The residency HEATMAP (the shared showcase pixels) ===================
// The per-mip page-grid panel layout of the VT1 feedback viz (mip 0 widest, coarser mips
// proportionally smaller, a small gutter between panels), colored by RESIDENCY STATE at the captured
// frame instead of by feedback membership. Pure integer coloring from the integer capture flags ->
// BGRA bytes identical Vulkan/Metal/anywhere BY CONSTRUCTION (one source of truth: both showcases
// call THIS). State priority (highest first): evicted-this-frame (red), loaded-this-frame (bright
// lime), resident (green), requested-but-not-yet-resident (amber), previously-requested-now-cold
// (slate blue), never-requested (dark). Grid lines dim the cell by 5/8 (integer).
inline constexpr uint32_t kSc6HeatW = 1024;
inline constexpr uint32_t kSc6HeatH = 256;

inline void Sc6DrawHeatmap(const Sc6FrameCapture& cap, const vt::VtTexture& vtx,
                           std::vector<uint8_t>& bgra) {
    const uint32_t imgW = kSc6HeatW, imgH = kSc6HeatH;
    bgra.assign((size_t)imgW * imgH * 4, 0);
    for (size_t p = 0; p < (size_t)imgW * imgH; ++p) {   // deep navy background
        bgra[p * 4 + 0] = 10; bgra[p * 4 + 1] = 8; bgra[p * 4 + 2] = 4; bgra[p * 4 + 3] = 255;
    }
    const int kGutter = 8;
    const int kMaxPanel = (int)imgH - 2 * kGutter;
    int x = kGutter;
    for (int mip = 0; mip < vtx.mipLevels; ++mip) {
        const int pps = vtx.pagesPerSide(mip);
        int side = kMaxPanel * pps / vtx.virtualPagesPerSideMip0;
        if (side < pps) side = pps;
        int y0 = ((int)imgH - side) / 2;
        for (int sy = 0; sy < side; ++sy) {
            int py = sy * pps / side; if (py >= pps) py = pps - 1;
            int iy = y0 + sy; if (iy < 0 || iy >= (int)imgH) continue;
            for (int sx = 0; sx < side; ++sx) {
                int px = sx * pps / side; if (px >= pps) px = pps - 1;
                int ix = x + sx; if (ix < 0 || ix >= (int)imgW) continue;
                const size_t pageId = (size_t)vt::PageId(mip, px, py, vtx);
                uint32_t b, g, r;
                if (pageId < cap.evictedNow.size() && cap.evictedNow[pageId]) {
                    b = 40; g = 40; r = 210;                     // evicted this frame: red
                } else if (pageId < cap.loadedNow.size() && cap.loadedNow[pageId]) {
                    b = 70; g = 235; r = 190;                    // loaded this frame: bright lime
                } else if (pageId < cap.resident.size() && cap.resident[pageId]) {
                    b = 80; g = 185; r = 60;                     // steady resident: green
                } else if (pageId < cap.requestedNow.size() && cap.requestedNow[pageId]) {
                    b = 40; g = 150; r = 235;                    // requested, awaiting budget: amber
                } else if (pageId < cap.everRequested.size() && cap.everRequested[pageId]) {
                    b = 120; g = 66; r = 44;                     // previously hot, now cold: slate
                } else {
                    b = 34; g = 34; r = 34;                      // never requested: dark
                }
                const bool gridLine = ((sx * pps) % side) < pps || ((sy * pps) % side) < pps;
                if (gridLine) { b = (b * 5u) >> 3; g = (g * 5u) >> 3; r = (r * 5u) >> 3; }
                uint8_t* dst = &bgra[((size_t)iy * imgW + ix) * 4];
                dst[0] = (uint8_t)b; dst[1] = (uint8_t)g; dst[2] = (uint8_t)r; dst[3] = 255;
            }
        }
        x += side + kGutter;
    }
}

// ============================ THE CANONICAL SHOWCASE SCENARIO + PINS ================================
// One source of truth for the --sc6-residency-shot (Vulkan + Metal) and the test's synthetic tier:
// the default synthetic fly-over under the default residency config. The PINS below are the run's
// invariant artifacts, asserted at all three call sites (integers only -> cross-platform).

inline Sc6ResidencyConfig Sc6DefaultResidency() { return Sc6ResidencyConfig{}; }

// PINNED artifacts of the canonical synthetic run (from the verified run; asserted at all three
// call sites — a change here is a residency-behavior change and must be justified). VERIFIED
// bit-identical MSVC x64 == clang x64 AND under a forced `clang -mfma -ffp-contract=fast` stress
// build (the split-statement discipline holds even beyond the standard contraction rules), so these
// are cross-platform pins including arm64. peakResident == poolCapacity: the pool provably sits
// FULL under the cap (the cap is load-bearing, not decorative). convergedAt = 66 < pathFrames = 72:
// the 8-page budget visibly LAGS the moving camera for most of the flight and catches up before the
// hold tail.
inline constexpr int32_t  kSc6ExpectedPages        = 1360;
inline constexpr int32_t  kSc6ExpectedPeakResident = 64;
inline constexpr int64_t  kSc6ExpectedLoads        = 294;
inline constexpr int64_t  kSc6ExpectedEvicts       = 279;
inline constexpr int32_t  kSc6ExpectedConvergedAt  = 66;
inline constexpr uint64_t kSc6ExpectedTraceDigest  = 0xe57f2491f1aad4dcull;

// The full canonical showcase computation, shared VERBATIM by the Vulkan --sc6-residency-shot, the
// Metal --sc6-residency-shot and the test's synthetic tier (ONE source of truth — the sc5_foliage
// Sc5DefaultConfig discipline): the budgeted run (twice, for the determinism proof), the unthrottled
// contrast run (the budget-is-load-bearing proof), the hysteresis thrash contrast, and the
// kSc6ShotFrame page-state capture for the heatmap.
struct Sc6CanonicalRun {
    Sc6SyntheticConfig  scfg;
    Sc6ResidencyConfig  rcfg;
    Sc6PageSpace        space;
    Sc6RunStats         stats;        // the canonical budgeted run
    Sc6RunStats         statsRepeat;  // an independent second run (two-run determinism)
    Sc6RunStats         unbounded;    // unthrottled budgets + full-pool cap (the contrast)
    Sc6ThrashStats      thrash;
    Sc6FrameCapture     capture;      // page states at kSc6ShotFrame (the heatmap input)
};

inline Sc6CanonicalRun Sc6RunCanonicalShowcase() {
    Sc6CanonicalRun run;
    run.scfg = Sc6DefaultSynthetic();
    run.rcfg = Sc6DefaultResidency();
    run.space = MakeSc6PageSpace(1, run.scfg.texture);

    const std::vector<std::vector<Sc6PageRequest>> frames = Sc6SyntheticFeedback(run.scfg);
    run.stats = Sc6RunTrace(frames, run.space, run.rcfg, kSc6ShotFrame, &run.capture);
    run.statsRepeat = Sc6RunTrace(frames, run.space, run.rcfg);

    Sc6ResidencyConfig unb = run.rcfg;
    unb.loadBudgetPerFrame = 1 << 20;
    unb.evictBudgetPerFrame = 1 << 20;
    unb.poolCapacity = run.space.totalPages;
    run.unbounded = Sc6RunTrace(frames, run.space, unb);

    run.thrash = Sc6ThrashContrast(run.scfg, run.rcfg.hysteresisFrames);
    return run;
}

}  // namespace hf::render::sc6
