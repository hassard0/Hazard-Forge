#pragma once
// Slice UF1 — DETERMINISTIC SDF TEXT (signed-distance-field glyph generation + proportional text
// layout/shaping). The parity++ UI-text gap: engine/ui/text.h is a FIXED 8x8 monospace ASCII bitmap
// (0x20..0x7E) — no scalable glyphs, no proportional advances, no kerning/shaping. UF1 adds a
// deterministic SDF glyph pipeline so UI text SCALES crisply (the SDF win: one distance field renders
// sharp at any output resolution — no 2x blockiness the 8x8 bitmap suffers) and lays out with real
// per-glyph ADVANCES + pair KERNING.
//
// PURE CPU: this module has ZERO RHI / graphics-backend symbols (no vk*/MTL*/mtl::/Backend::Metal),
// NO clock, NO RNG, NO runtime transcendentals — every distance is INTEGER (point-to-segment in
// fixed-point + an exact integer sqrt, the NAV/convex discipline), so the SDF bytes + layout + atlas
// are bit-identical on every compiler/backend. It composes engine/ui/text.h READ-ONLY (constants
// only) — text.h/widget.h are byte-UNTOUCHED.
//
// GLYPH SOURCE — HONEST SCOPE: the glyphs are HAND-AUTHORED VECTOR STROKE OUTLINES (centerline
// segments + quadratic arcs on a 64-unit em grid, with a fixed stroke radius), NOT a TTF/FreeType
// font file. Bringing in TTF/FreeType is out of scope: an external loader is nondeterministic-ish
// (version/hinting-dependent) and not self-contained. So UF1 ships a genuine SUBSET — uppercase A-Z,
// digits 0-9, space, '.', '-', '!' (~40 glyphs) — enough to prove the SDF+layout+shaping pipeline.
// The SDF is the signed distance to the STROKED skeleton (the Minkowski sum of the centerline with a
// disc of radius kStrokeEm): a "monoline" stroke font, not a filled-outline font. Unicode / bidi /
// complex shaping / ligatures are NOT done (Latin subset only). A GPU SDF-text shader (alpha =
// smoothstep(0.5-w, 0.5+w, sdf)) is the natural FUTURE capstone; this slice stays shader-free and
// proves the field with a strict-zero INTEGER coverage raster.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "ui/text.h"  // READ-ONLY: kGlyphFirst/kGlyphLast constants (the 8x8 bitmap we improve upon)

namespace hf::ui::sdf {

// --- Fixed-point / grid constants ---------------------------------------------------------------
inline constexpr int kEm       = 64;   // glyphs authored on a 0..64 em grid (y-up: 0 = baseline bot)
inline constexpr int kSub      = 64;   // work units per em unit -> distances at 1/64-em precision
inline constexpr int kEmWork   = kEm * kSub;          // 4096: full em box in work units
inline constexpr int kStrokeEm = 3;    // stroke RADIUS in em units (stroke width ~6em ~= 0.094 em)
inline constexpr int kStrokeW  = kStrokeEm * kSub;    // stroke radius in work units (192)
inline constexpr int kQuadSub  = 8;    // quadratic-Bezier flattening: fixed subdivisions (det.)

// --- FNV-1a digest (self-contained; no anim dependency) -----------------------------------------
namespace detail {
inline constexpr uint64_t kFnvSeed = 14695981039346656037ull;
inline uint64_t Fnv(uint64_t h, uint32_t w) { h ^= w; h *= 1099511628211ull; return h; }
inline uint64_t Fnv64(uint64_t h, uint64_t v) {
    h = Fnv(h, (uint32_t)(v & 0xffffffffu));
    h = Fnv(h, (uint32_t)(v >> 32));
    return h;
}
// Exact integer floor(sqrt(v)) — deterministic, NOT a transcendental (bisection, overflow-free).
inline int64_t IsqrtFloor(int64_t v) {
    if (v <= 0) return 0;
    int64_t lo = 1, hi = 3037000499ll;  // floor(sqrt(INT64_MAX))
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo + 1) / 2;
        if (mid <= v / mid) lo = mid; else hi = mid - 1;
    }
    return lo;
}
}  // namespace detail

// --- Authored vector glyphs ---------------------------------------------------------------------
// A stroke segment on the em grid: a LINE (a->b) or a quadratic-Bezier arc (a, control c, b). Curved
// glyphs (O/C/D/S/digits...) use quads (flattened deterministically); straight glyphs use lines.
struct Seg {
    int kind;                 // 0 = line, 1 = quadratic
    int ax, ay, bx, by;       // endpoints (em units, y-up)
    int cx, cy;               // quad control (em units); unused for lines
};
struct Glyph {
    char ch;
    int  advance;             // proportional advance width (em units)
    std::vector<Seg> segs;    // stroke centerlines
};

namespace detail {
inline Seg L(int ax, int ay, int bx, int by) { return Seg{0, ax, ay, bx, by, 0, 0}; }
inline Seg Q(int ax, int ay, int cx, int cy, int bx, int by) { return Seg{1, ax, ay, bx, by, cx, cy}; }
}  // namespace detail

// The hand-authored glyph table. Guides (em, y-up): BOT=8 TOP=56 MIDY=32, L=14 R=50 MIDX=32.
inline const std::vector<Glyph>& GlyphSet() {
    using detail::L;
    using detail::Q;
    static const std::vector<Glyph> g = [] {
        const int BOT = 8, TOP = 56, MY = 32, LX = 14, RX = 50, MX = 32;
        std::vector<Glyph> v;
        auto add = [&](char ch, int adv, std::vector<Seg> s) { v.push_back(Glyph{ch, adv, std::move(s)}); };

        add(' ', 32, {});
        add('A', 56, {L(LX, BOT, MX, TOP), L(MX, TOP, RX, BOT), L(21, 28, 43, 28)});
        add('B', 56, {L(LX, BOT, LX, TOP), Q(LX, TOP, 58, 44, LX, MY), Q(LX, MY, 58, 20, LX, BOT)});
        add('C', 56, {Q(48, 52, 12, 54, 12, MY), Q(12, MY, 12, 10, 48, 12)});
        add('D', 56, {L(LX, BOT, LX, TOP), Q(LX, TOP, 58, MY, LX, BOT)});
        add('E', 52, {L(LX, BOT, LX, TOP), L(LX, TOP, RX, TOP), L(LX, MY, 44, MY), L(LX, BOT, RX, BOT)});
        add('F', 50, {L(LX, BOT, LX, TOP), L(LX, TOP, RX, TOP), L(LX, MY, 44, MY)});
        add('G', 58, {Q(48, 52, 12, 54, 12, MY), Q(12, MY, 12, 10, 48, 12),
                      L(48, 12, 48, 26), L(36, 26, 48, 26)});
        add('H', 56, {L(LX, BOT, LX, TOP), L(RX, BOT, RX, TOP), L(LX, MY, RX, MY)});
        add('I', 28, {L(MX, BOT, MX, TOP), L(24, TOP, 40, TOP), L(24, BOT, 40, BOT)});
        add('J', 46, {L(40, TOP, 40, 20), Q(40, 20, 40, 8, 26, 8), Q(26, 8, 14, 8, 14, 20)});
        add('K', 54, {L(LX, BOT, LX, TOP), L(RX, TOP, 18, MY), L(18, MY, RX, BOT)});
        add('L', 48, {L(LX, BOT, LX, TOP), L(LX, BOT, RX, BOT)});
        add('M', 64, {L(LX, BOT, LX, TOP), L(LX, TOP, MX, 30), L(MX, 30, RX, TOP), L(RX, TOP, RX, BOT)});
        add('N', 58, {L(LX, BOT, LX, TOP), L(LX, TOP, RX, BOT), L(RX, BOT, RX, TOP)});
        add('O', 58, {Q(52, MY, 52, 56, MX, 56), Q(MX, 56, 12, 56, 12, MY),
                      Q(12, MY, 12, 8, MX, 8), Q(MX, 8, 52, 8, 52, MY)});
        add('P', 54, {L(LX, BOT, LX, TOP), Q(LX, TOP, 58, 44, LX, MY)});
        add('Q', 60, {Q(52, MY, 52, 56, MX, 56), Q(MX, 56, 12, 56, 12, MY),
                      Q(12, MY, 12, 8, MX, 8), Q(MX, 8, 52, 8, 52, MY), L(36, 22, 54, 6)});
        add('R', 56, {L(LX, BOT, LX, TOP), Q(LX, TOP, 58, 44, LX, MY), L(30, MY, RX, BOT)});
        add('S', 54, {Q(48, 50, 16, 58, 16, 40), Q(16, 40, 48, 32, 48, 24), Q(48, 24, 48, 8, 16, 14)});
        add('T', 52, {L(LX, TOP, RX, TOP), L(MX, TOP, MX, BOT)});
        add('U', 56, {L(LX, TOP, LX, 20), Q(LX, 20, LX, 8, MX, 8), Q(MX, 8, RX, 8, RX, 20), L(RX, 20, RX, TOP)});
        add('V', 56, {L(LX, TOP, MX, BOT), L(MX, TOP, RX, BOT)});
        add('W', 64, {L(LX, TOP, 24, BOT), L(24, BOT, MX, 30), L(MX, 30, 40, BOT), L(40, BOT, RX, TOP)});
        add('X', 56, {L(LX, BOT, RX, TOP), L(LX, TOP, RX, BOT)});
        add('Y', 56, {L(LX, TOP, MX, MY), L(RX, TOP, MX, MY), L(MX, MY, MX, BOT)});
        add('Z', 54, {L(LX, TOP, RX, TOP), L(RX, TOP, LX, BOT), L(LX, BOT, RX, BOT)});
        add('0', 56, {Q(52, MY, 52, 56, MX, 56), Q(MX, 56, 12, 56, 12, MY),
                      Q(12, MY, 12, 8, MX, 8), Q(MX, 8, 52, 8, 52, MY), L(22, 20, 42, 44)});
        add('1', 44, {L(22, 48, MX, TOP), L(MX, TOP, MX, BOT), L(22, BOT, 42, BOT)});
        add('2', 54, {Q(16, 46, 20, 58, 34, 56), Q(34, 56, 50, 54, 46, 38), L(46, 38, 16, 8), L(16, 8, 48, 8)});
        add('3', 54, {Q(18, 52, 52, 56, 34, 34), Q(34, 34, 52, 12, 16, 12)});
        add('4', 54, {L(40, TOP, 14, 22), L(14, 22, RX, 22), L(40, TOP, 40, BOT)});
        add('5', 54, {L(48, TOP, 16, TOP), L(16, TOP, 16, 36), Q(16, 36, 50, 40, 48, 22), Q(48, 22, 46, 8, 16, 12)});
        add('6', 54, {Q(46, 50, 16, 56, 14, 30), Q(14, 30, 14, 8, MX, 8),
                      Q(MX, 8, 50, 8, 50, 22), Q(50, 22, 50, 34, 30, 32)});
        add('7', 52, {L(LX, TOP, RX, TOP), L(RX, TOP, 26, BOT)});
        add('8', 54, {Q(MX, 34, 16, 34, 16, 46), Q(16, 46, 16, 56, MX, 56), Q(MX, 56, 48, 56, 48, 46),
                      Q(48, 46, 48, 34, MX, 34), Q(MX, 34, 14, 34, 14, 20), Q(14, 20, 14, 8, MX, 8),
                      Q(MX, 8, 50, 8, 50, 20), Q(50, 20, 50, 34, MX, 34)});
        add('9', 54, {Q(18, 14, 48, 8, 50, 34), Q(50, 34, 50, 56, MX, 56),
                      Q(MX, 56, 14, 56, 14, 42), Q(14, 42, 14, 32, 34, 34)});
        add('.', 28, {L(30, 10, 34, 10)});
        add('-', 48, {L(18, MY, 46, MY)});
        add('!', 28, {L(MX, TOP, MX, 20), L(MX, 10, MX, 12)});
        return v;
    }();
    return g;
}

// Index of char `c` in GlyphSet(), or -1 if unsupported.
inline int GlyphIndex(char c) {
    const std::vector<Glyph>& g = GlyphSet();
    for (size_t i = 0; i < g.size(); ++i)
        if (g[i].ch == c) return (int)i;
    return -1;
}

// --- Flattened stroke skeleton (work units) -----------------------------------------------------
// One flattened line segment in WORK units (em*kSub). Quads are pre-subdivided into kQuadSub pieces.
struct FlatSeg { int64_t ax, ay, bx, by; };

inline std::vector<FlatSeg> FlattenGlyph(const Glyph& gl) {
    std::vector<FlatSeg> out;
    auto emToWork = [](int64_t e) { return e * kSub; };
    for (const Seg& s : gl.segs) {
        if (s.kind == 0) {
            out.push_back(FlatSeg{emToWork(s.ax), emToWork(s.ay), emToWork(s.bx), emToWork(s.by)});
        } else {
            const int N = kQuadSub;
            int64_t px = 0, py = 0;
            for (int i = 0; i <= N; ++i) {
                const int64_t u = N - i;
                // B(t) in em, scaled to work with rounding: ((u^2 a + 2 u i c + i^2 b) * kSub)/N^2
                const int64_t nx = ((u * u * s.ax + 2 * u * i * s.cx + (int64_t)i * i * s.bx) * kSub);
                const int64_t ny = ((u * u * s.ay + 2 * u * i * s.cy + (int64_t)i * i * s.by) * kSub);
                const int64_t den = (int64_t)N * N;
                const int64_t qx = (nx + den / 2) / den;
                const int64_t qy = (ny + den / 2) / den;
                if (i > 0) out.push_back(FlatSeg{px, py, qx, qy});
                px = qx; py = qy;
            }
        }
    }
    return out;
}

// Signed distance (work units) from point (px,py) to the STROKED glyph: min point-to-segment
// distance minus the stroke radius. Negative INSIDE the stroke, positive OUTSIDE (standard SDF).
inline int32_t SignedDistWork(const std::vector<FlatSeg>& segs, int64_t px, int64_t py) {
    int64_t best2 = -1;
    for (const FlatSeg& s : segs) {
        const int64_t abx = s.bx - s.ax, aby = s.by - s.ay;
        const int64_t apx = px - s.ax, apy = py - s.ay;
        const int64_t den = abx * abx + aby * aby;
        int64_t cxp, cyp;
        if (den == 0) {
            cxp = s.ax; cyp = s.ay;
        } else {
            int64_t num = apx * abx + apy * aby;
            if (num <= 0) { cxp = s.ax; cyp = s.ay; }
            else if (num >= den) { cxp = s.bx; cyp = s.by; }
            else { cxp = s.ax + (abx * num + den / 2) / den; cyp = s.ay + (aby * num + den / 2) / den; }
        }
        const int64_t dx = px - cxp, dy = py - cyp;
        const int64_t d2 = dx * dx + dy * dy;
        if (best2 < 0 || d2 < best2) best2 = d2;
    }
    if (best2 < 0) return kEmWork;  // empty glyph (space): everywhere "outside"
    const int64_t dist = detail::IsqrtFloor(best2);
    return (int32_t)(dist - kStrokeW);
}

// --- Glyph SDF bitmap ---------------------------------------------------------------------------
// A square size x size signed-distance field covering the full 0..kEm em box. sd[ty*size+tx] holds
// the signed distance in WORK units (negative inside). Row 0 is the TOP of the glyph (y flipped).
struct GlyphSDF {
    char ch = 0;
    int  w = 0, h = 0;        // == size
    int  size = 0;
    std::vector<int32_t> sd;  // w*h
};

inline GlyphSDF GenerateGlyphSDF(const Glyph& gl, int size) {
    GlyphSDF g;
    g.ch = gl.ch; g.w = size; g.h = size; g.size = size;
    g.sd.resize((size_t)size * size);
    const std::vector<FlatSeg> segs = FlattenGlyph(gl);
    for (int ty = 0; ty < size; ++ty) {
        // texel-center em (y-up): row 0 -> near top
        const int64_t emY = ((int64_t)(2 * (size - 1 - ty) + 1) * kEmWork) / (2 * size);
        for (int tx = 0; tx < size; ++tx) {
            const int64_t emX = ((int64_t)(2 * tx + 1) * kEmWork) / (2 * size);
            g.sd[(size_t)ty * size + tx] = SignedDistWork(segs, emX, emY);
        }
    }
    return g;
}
inline GlyphSDF GenerateGlyphSDF(char c, int size) {
    const int gi = GlyphIndex(c);
    static const Glyph blank{0, 32, {}};
    return GenerateGlyphSDF(gi < 0 ? blank : GlyphSet()[(size_t)gi], size);
}

// --- SDF sampling -> coverage (shader-free, integer) --------------------------------------------
// coverage(sd): the SDF->alpha threshold. Fully inside (sd <= -band) -> 255, fully outside (>= band)
// -> 0, an integer SMOOTHSTEP across the +/-band transition (the GPU shader's smoothstep(0.5-w,
// 0.5+w) analog). `bandWork` is the AA half-width in work units — set to ~one OUTPUT texel so the
// transition stays ~1-2 texels crisp at ANY output resolution (the SDF scale win).
inline int SampleCoverage(int32_t sd, int bandWork) {
    if (bandWork <= 0) return sd < 0 ? 255 : 0;
    if (sd <= -bandWork) return 255;
    if (sd >= bandWork) return 0;
    const long lin = (long)(bandWork - sd) * 255 / (2L * bandWork);  // 0..255 linear
    const long t = lin < 0 ? 0 : (lin > 255 ? 255 : lin);
    const long s = (t * t * (3 * 255 - 2 * t)) / (255L * 255L);      // integer smoothstep
    return (int)(s < 0 ? 0 : (s > 255 ? 255 : s));
}

// Bilinear sample of a GlyphSDF at normalized (u,v) given in Q16 [0..65536] -> signed dist (work).
inline int32_t SampleSDF(const GlyphSDF& g, int32_t uQ16, int32_t vQ16) {
    if (g.w <= 0 || g.h <= 0) return kEmWork;
    auto clampq = [](int32_t q) { return q < 0 ? 0 : (q > 65536 ? 65536 : q); };
    uQ16 = clampq(uQ16); vQ16 = clampq(vQ16);
    const int64_t fx = (int64_t)uQ16 * (g.w - 1);   // Q16 grid x
    const int64_t fy = (int64_t)vQ16 * (g.h - 1);
    const int x0 = (int)(fx >> 16), y0 = (int)(fy >> 16);
    const int x1 = x0 + 1 < g.w ? x0 + 1 : x0;
    const int y1 = y0 + 1 < g.h ? y0 + 1 : y0;
    const int64_t tx = fx & 0xffff, tyf = fy & 0xffff;
    auto at = [&](int x, int y) -> int64_t { return g.sd[(size_t)y * g.w + x]; };
    const int64_t top = at(x0, y0) + (((at(x1, y0) - at(x0, y0)) * tx) >> 16);
    const int64_t bot = at(x0, y1) + (((at(x1, y1) - at(x0, y1)) * tx) >> 16);
    return (int32_t)(top + (((bot - top) * tyf) >> 16));
}

// Render a coverage bitmap (0..255) of size outW x outH by SAMPLING the SDF `g` (bilinear). The SAME
// field renders crisp at any (outW,outH) — the scale win over the 8x8 bitmap. band == one out texel.
inline std::vector<uint8_t> RenderCoverage(const GlyphSDF& g, int outW, int outH) {
    std::vector<uint8_t> cov((size_t)outW * outH, 0);
    const int band = kEmWork / (outW > 0 ? outW : 1);
    for (int oy = 0; oy < outH; ++oy) {
        const int32_t vQ = (int32_t)(((int64_t)(2 * oy + 1) << 16) / (2 * outH));
        for (int ox = 0; ox < outW; ++ox) {
            const int32_t uQ = (int32_t)(((int64_t)(2 * ox + 1) << 16) / (2 * outW));
            cov[(size_t)oy * outW + ox] = (uint8_t)SampleCoverage(SampleSDF(g, uQ, vQ), band);
        }
    }
    return cov;
}

// Digest of a GlyphSDF (all distances + dims).
inline uint64_t DigestGlyphSDF(const GlyphSDF& g) {
    uint64_t h = detail::kFnvSeed;
    h = detail::Fnv(h, (uint32_t)g.ch);
    h = detail::Fnv(h, (uint32_t)g.w);
    h = detail::Fnv(h, (uint32_t)g.h);
    for (int32_t d : g.sd) h = detail::Fnv(h, (uint32_t)d);
    return h;
}

// --- Text layout / shaping ----------------------------------------------------------------------
// A pinned kerning table: pair (a,b) -> delta in EM units (negative pulls closer). Proves the shaping
// capability on a few classic pairs.
struct KernPair { char a, b; int deltaEm; };
inline const std::vector<KernPair>& KernTable() {
    static const std::vector<KernPair> k = {
        {'A', 'V', -10}, {'V', 'A', -10}, {'A', 'Y', -10}, {'Y', 'A', -10},
        {'A', 'W', -8},  {'W', 'A', -8},  {'A', 'T', -8},  {'T', 'A', -8},
        {'L', 'T', -6},  {'L', 'Y', -8},  {'P', 'A', -8},  {'F', 'A', -8},
        {'T', 'O', -6},
    };
    return k;
}
inline int KernEm(char a, char b) {
    for (const KernPair& k : KernTable())
        if (k.a == a && k.b == b) return k.deltaEm;
    return 0;
}

// One positioned glyph: `glyph` indexes GlyphSet(); penX/penY are the top-left pen position in output
// PIXELS (penY is the top of the line box). Spaces advance but are NOT emitted.
struct Positioned {
    int  glyph;   // index into GlyphSet()
    char ch;
    int  penX, penY;
    int  advancePx;
};
struct LayoutResult {
    std::vector<Positioned> glyphs;
    int lines = 1;
    int width = 0;   // max line extent in px
    int height = 0;  // total block height in px
};

// Lay `s` out at pixel height `sizePx`, `trackingPx` extra spacing per glyph, wrapping when the pen
// would exceed `maxWidthPx` (0 = no wrap). `applyKern` toggles the pair-kerning table. All-integer,
// deterministic. Advances are PROPORTIONAL (glyph.advance * sizePx / kEm), not monospace.
inline LayoutResult LayoutText(const std::string& s, int sizePx, int trackingPx, int maxWidthPx,
                               bool applyKern) {
    LayoutResult r;
    const int lineH = sizePx * 5 / 4;
    r.height = lineH;
    int penX = 0, penY = 0, lineStart = 1;
    char prev = 0;
    auto advPx = [&](int advEm) { return advEm * sizePx / kEm; };
    for (char c : s) {
        if (c == '\n') {
            if (penX > r.width) r.width = penX;
            penX = 0; penY += lineH; r.lines++; r.height = penY + lineH; prev = 0; lineStart = 1;
            continue;
        }
        const int gi = GlyphIndex(c);
        const int advEm = gi < 0 ? 32 : GlyphSet()[(size_t)gi].advance;
        const int gAdv = advPx(advEm);
        int kern = 0;
        if (applyKern && !lineStart && prev) kern = advPx(KernEm(prev, c)) ;
        // wrap BEFORE placing (never wrap the first glyph on a line)
        if (maxWidthPx > 0 && !lineStart && penX + kern + gAdv > maxWidthPx) {
            if (penX > r.width) r.width = penX;
            penX = 0; penY += lineH; r.lines++; r.height = penY + lineH; prev = 0; lineStart = 1;
            kern = 0;
        }
        penX += kern;
        if (c != ' ' && gi >= 0)
            r.glyphs.push_back(Positioned{gi, c, penX, penY, gAdv});
        penX += gAdv + trackingPx;
        prev = c; lineStart = 0;
    }
    if (penX > r.width) r.width = penX;
    return r;
}

inline uint64_t DigestLayout(const LayoutResult& r) {
    uint64_t h = detail::kFnvSeed;
    h = detail::Fnv(h, (uint32_t)r.lines);
    h = detail::Fnv(h, (uint32_t)r.width);
    h = detail::Fnv(h, (uint32_t)r.height);
    for (const Positioned& p : r.glyphs) {
        h = detail::Fnv(h, (uint32_t)p.ch);
        h = detail::Fnv(h, (uint32_t)p.penX);
        h = detail::Fnv(h, (uint32_t)p.penY);
        h = detail::Fnv(h, (uint32_t)p.advancePx);
    }
    return h;
}

// --- Glyph atlas (deterministic shelf packing) --------------------------------------------------
struct AtlasRect { char ch; int x, y, w, h; };
struct Atlas {
    int w = 0, h = 0;
    std::vector<AtlasRect> rects;
    std::vector<GlyphSDF>  cells;   // parallel to rects: the per-glyph SDF at cell resolution
    std::vector<uint8_t>   cov;     // w*h coverage thumbnail (glyphs blitted at native cell res)
};

// Pack every glyph's `cellSize x cellSize` SDF into an atlas via left-to-right shelf/row packing,
// wrapping to a new shelf when the row would exceed `maxWidth`. Uniform square cells => a real
// deterministic shelf packer with NO overlaps (pinned rects).
inline Atlas BuildAtlas(int cellSize, int maxWidth) {
    Atlas a;
    const std::vector<Glyph>& gs = GlyphSet();
    int curX = 0, curY = 0, shelfH = 0, usedW = 0;
    for (const Glyph& gl : gs) {
        if (curX > 0 && curX + cellSize > maxWidth) { curY += shelfH; curX = 0; shelfH = 0; }
        a.rects.push_back(AtlasRect{gl.ch, curX, curY, cellSize, cellSize});
        a.cells.push_back(GenerateGlyphSDF(gl, cellSize));
        curX += cellSize;
        if (cellSize > shelfH) shelfH = cellSize;
        if (curX > usedW) usedW = curX;
    }
    a.w = usedW;
    a.h = curY + shelfH;
    a.cov.assign((size_t)a.w * a.h, 0);
    for (size_t i = 0; i < a.rects.size(); ++i) {
        const AtlasRect& rc = a.rects[i];
        const std::vector<uint8_t> c = RenderCoverage(a.cells[i], rc.w, rc.h);
        for (int y = 0; y < rc.h; ++y)
            for (int x = 0; x < rc.w; ++x)
                a.cov[(size_t)(rc.y + y) * a.w + (rc.x + x)] = c[(size_t)y * rc.w + x];
    }
    return a;
}

inline uint64_t DigestAtlas(const Atlas& a) {
    uint64_t h = detail::kFnvSeed;
    h = detail::Fnv(h, (uint32_t)a.w);
    h = detail::Fnv(h, (uint32_t)a.h);
    for (const AtlasRect& r : a.rects) {
        h = detail::Fnv(h, (uint32_t)r.ch);
        h = detail::Fnv(h, (uint32_t)r.x);
        h = detail::Fnv(h, (uint32_t)r.y);
        h = detail::Fnv(h, (uint32_t)r.w);
        h = detail::Fnv(h, (uint32_t)r.h);
    }
    for (const GlyphSDF& c : a.cells) h = detail::Fnv64(h, DigestGlyphSDF(c));
    return h;
}

// ================================================================================================
// The SHARED showcase scenario both backends run (Vulkan --uf1-text-shot / Metal --uf1-text). PURE
// CPU, strict-zero INTEGER coverage raster (NO shader). Composes: a packed atlas, one glyph's raw SDF
// (the distance ramp inset), and two proportionally-laid-out + kerned text lines rendered crisp.
// ================================================================================================
struct Uf1Run {
    Atlas atlas;
    GlyphSDF ramp;                 // one showcase glyph at high res (raw SDF, for the ramp inset)
    LayoutResult line1, line2;     // the two laid-out text lines (kerned)
    std::string s1, s2;
    int bigSize = 0;               // render size of the text lines
    int atlasCell = 0;
    int glyphCount = 0;
    int stringLen = 0;
    uint64_t sdfDigest = 0, atlasDigest = 0, layoutDigest = 0, digest = 0;
};

inline Uf1Run RunUf1TextScenario() {
    Uf1Run r;
    r.atlasCell  = 24;
    r.bigSize    = 44;
    r.s1 = "HAZARD FORGE";
    r.s2 = "0123456789 -!.";
    r.glyphCount = (int)GlyphSet().size();
    r.stringLen  = (int)(r.s1.size() + r.s2.size());

    r.atlas = BuildAtlas(r.atlasCell, /*maxWidth=*/8 * r.atlasCell);
    r.ramp  = GenerateGlyphSDF('G', 48);                                 // the distance-ramp inset
    r.line1 = LayoutText(r.s1, r.bigSize, /*tracking=*/1, /*maxWidth=*/0, /*kern=*/true);
    r.line2 = LayoutText(r.s2, r.bigSize, /*tracking=*/1, /*maxWidth=*/0, /*kern=*/true);

    r.atlasDigest  = DigestAtlas(r.atlas);
    r.sdfDigest    = DigestGlyphSDF(r.ramp);
    uint64_t hl = detail::kFnvSeed;
    hl = detail::Fnv64(hl, DigestLayout(r.line1));
    hl = detail::Fnv64(hl, DigestLayout(r.line2));
    r.layoutDigest = hl;

    uint64_t h = detail::kFnvSeed;
    h = detail::Fnv64(h, r.atlasDigest);
    h = detail::Fnv64(h, r.sdfDigest);
    h = detail::Fnv64(h, r.layoutDigest);
    h = detail::Fnv(h, (uint32_t)r.glyphCount);
    h = detail::Fnv(h, (uint32_t)r.stringLen);
    r.digest = h;
    return r;
}

// RenderUf1Shot: the SHARED pure-integer raster (strict-zero cross-backend BY CONSTRUCTION). BGRA8.
// Top: "HAZARD FORGE" + a digit/punct row rendered crisp from per-glyph SDFs (warm amber ink on
// slate). Right inset: the raw SDF of 'G' as a signed distance ramp (blue inside / red outside).
// Bottom inset: the packed glyph atlas thumbnail (cyan). All integer; NO shader.
inline void RenderUf1Shot(const Uf1Run& run, std::vector<uint8_t>& bgra, uint32_t& outW, uint32_t& outH) {
    const int W = 760, H = 420;
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {
        bgra[p * 4 + 0] = 22; bgra[p * 4 + 1] = 17; bgra[p * 4 + 2] = 13; bgra[p * 4 + 3] = 255;
    }
    auto blend = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b, int a) {
        if (x < 0 || x >= W || y < 0 || y >= H || a <= 0) return;
        uint8_t* d = &bgra[((size_t)y * W + x) * 4];
        if (a >= 255) { d[0] = b; d[1] = g; d[2] = r; d[3] = 255; return; }
        d[0] = (uint8_t)((b * a + d[0] * (255 - a)) / 255);
        d[1] = (uint8_t)((g * a + d[1] * (255 - a)) / 255);
        d[2] = (uint8_t)((r * a + d[2] * (255 - a)) / 255);
    };
    // --- The two text lines, rendered crisp from per-glyph SDFs ---
    auto drawLine = [&](const LayoutResult& lr, int ox, int oy, uint8_t cr, uint8_t cg, uint8_t cb) {
        for (const Positioned& p : lr.glyphs) {
            const GlyphSDF g = GenerateGlyphSDF(GlyphSet()[(size_t)p.glyph], run.bigSize);
            const std::vector<uint8_t> cov = RenderCoverage(g, run.bigSize, run.bigSize);
            for (int y = 0; y < run.bigSize; ++y)
                for (int x = 0; x < run.bigSize; ++x) {
                    const int a = cov[(size_t)y * run.bigSize + x];
                    if (a) blend(ox + p.penX + x, oy + p.penY + y, cr, cg, cb, a);
                }
        }
    };
    drawLine(run.line1, 24, 30, 240, 190, 90);    // amber
    drawLine(run.line2, 24, 128, 150, 210, 235);  // cyan

    // --- Raw SDF ramp inset (one glyph): blue inside (neg), red outside (pos), white at the 0 edge ---
    {
        const int ix = 560, iy = 210, sz = run.ramp.size, scale = 3;
        for (int y = 0; y < sz; ++y)
            for (int x = 0; x < sz; ++x) {
                const int32_t sd = run.ramp.sd[(size_t)y * sz + x];
                // map sd in [-kStrokeW, +2*kStrokeW] to a diverging ramp
                uint8_t r, g, b;
                if (sd < 0) {           // inside: blue, brighter toward core
                    const int t = (int)((int64_t)(-sd) * 255 / (kStrokeW + 1));
                    r = 40; g = (uint8_t)(80 + t / 3); b = (uint8_t)(160 + (t > 95 ? 95 : t));
                } else {                // outside: red fading to background
                    const int t = (int)((int64_t)sd * 255 / (2 * kStrokeW + 1));
                    const int u = 255 - (t > 255 ? 255 : t);
                    r = (uint8_t)(120 + u / 3); g = (uint8_t)(30 + u / 6); b = 30;
                }
                if (sd > -18 && sd < 18) { r = 235; g = 235; b = 235; }  // the zero contour
                for (int sy = 0; sy < scale; ++sy)
                    for (int sx = 0; sx < scale; ++sx)
                        blend(ix + x * scale + sx, iy + y * scale + sy, r, g, b, 255);
            }
    }
    // --- Packed atlas thumbnail (cyan coverage) ---
    {
        const int ax = 560, ay = 40;
        for (int y = 0; y < run.atlas.h && y < 150; ++y)
            for (int x = 0; x < run.atlas.w && x < 190; ++x) {
                const int a = run.atlas.cov[(size_t)y * run.atlas.w + x];
                if (a) blend(ax + x, ay + y, 120, 220, 210, a);
            }
    }
}

}  // namespace hf::ui::sdf
