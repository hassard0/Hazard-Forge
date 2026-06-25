#pragma once

// USDA (ASCII USD) geometry parser (issue #15, ask #3 — "Asset import: support FBX, OBJ and USD alongside
// glTF"). A clean-room, dependency-free importer for Pixar's Universal Scene Description text format (.usda /
// .usd), the interchange standard every modern DCC + film/game pipeline (Maya, Houdini, Blender, Omniverse,
// Unreal's USD stage) now ships. Like obj_loader.h / fbx_loader.h, the PURE parse (text -> UsdMesh) is split
// from any device/RHI dependency so it is unit-testable standalone and golden-hashable: re-importing the same
// USDA produces a BYTE-IDENTICAL mesh across machines/compilers (the engine's moat), which a pinned
// net::DigestBytes test proves — unlike a float-/machine-dependent USD-SDK import. A thin device wrapper
// (UsdMesh -> scene::Mesh, computing smooth normals + tangents) would layer on top elsewhere, exactly like
// obj_loader's LoadObjMesh.
//
// SELF-CONTAINED on purpose: <cstddef>/<cstdint>/<cstring>/<string>/<vector> + net/session.h (DigestBytes)
// only. NO RHI/scene/<map>/<unordered_*>/<algorithm>/<cmath>/<cstdlib> — a small hand tokenizer parses the
// floats/ints out of the bracketed/parenthesized arrays (no strtod/strtol locale or rounding surprises;
// every digit is folded the SAME way on every compiler, so the decoded mesh is bit-identical).
//
// THE USDA FORMAT (a text format):
//   #usda 1.0
//   def Mesh "cube" {
//       point3f[] points = [(-0.5,-0.5,-0.5), (0.5,-0.5,-0.5), ...]   # vertex positions (x,y,z) floats
//       int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]                   # vertices per face (3=tri,4=quad,...)
//       int[] faceVertexIndices = [0,1,2,3, 4,5,6,7, ...]            # flat index list; face i eats counts[i]
//   }
// Each polygon is fan-triangulated: a face with corners [i0,i1,...,ik] emits (i0,i1,i2),(i0,i2,i3),...
//
// Scope (v1): the FIRST `def Mesh`'s points + faceVertexCounts + faceVertexIndices only. Transforms,
// materials, nested prims, normals/uvs, multiple meshes, and binary/crate USD are documented follow-ups.
// Unknown constructs are skipped (forward-compatible, not fatal). Malformed input -> ok=false (bounds-checked).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "net/session.h"   // hf::net::DigestBytes (the pinned state-digest currency)

namespace hf::asset {

struct UsdMesh {
    std::vector<float>    positions;   // xyz triples (3 floats/vertex), straight from point3f[] points
    std::vector<uint32_t> indices;     // triangle list (3 per tri), fan-triangulated from the faces
    bool                  ok = false;  // true iff a Mesh with points + faceVertexCounts + faceVertexIndices decoded
};

namespace usd_detail {

// ---------------------------------------------------------------------------------------------------------
// A tiny hand tokenizer over the USDA text. The two array kinds we read are:
//   point3f[] points = [ (f,f,f), (f,f,f), ... ]   -> floats (we keep every numeric token in order; the
//                                                     parens/commas are pure separators)
//   int[]     ...     = [ i, i, i, ... ]           -> ints
// So the parse reduces to: find the `= [` after a keyword, then scan numeric tokens until the matching `]`.
// We hand-parse each numeric token (sign, integer part, '.', fraction, e/E exponent) so there is NO strtod
// dependency and the float bits are produced identically on every compiler. The values are small authored
// constants (cube coordinates), well inside the exactly-representable range.
// ---------------------------------------------------------------------------------------------------------

inline bool IsSpace(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
inline bool IsDigit(char c) { return c >= '0' && c <= '9'; }

// Skip whitespace AND `#`-to-end-of-line comments, starting at i. Advances i past them.
inline void SkipWsAndComments(const char* t, std::size_t n, std::size_t& i) {
    for (;;) {
        while (i < n && IsSpace(t[i])) ++i;
        if (i < n && t[i] == '#') { while (i < n && t[i] != '\n') ++i; continue; }
        break;
    }
}

// Parse a numeric token (a decimal float; ints are a special case) beginning at t[i]. On success sets
// `out` and advances i past the token; returns true. A token is [sign] digits [.digits] [(e|E)[sign]digits].
// Pure-integer mantissa/exponent assembly then a single scaled divide/multiply — deterministic across
// compilers (no strtod). Returns false if no numeric token starts at i.
inline bool ParseNumber(const char* t, std::size_t n, std::size_t& i, double& out) {
    std::size_t j = i;
    bool neg = false;
    if (j < n && (t[j] == '+' || t[j] == '-')) { neg = (t[j] == '-'); ++j; }
    // need at least one digit or a leading '.digit'
    bool any = false;
    // integer part
    double mant = 0.0;
    while (j < n && IsDigit(t[j])) { mant = mant * 10.0 + (double)(t[j] - '0'); ++j; any = true; }
    // fractional part
    if (j < n && t[j] == '.') {
        ++j;
        double scale = 1.0;
        while (j < n && IsDigit(t[j])) { mant = mant * 10.0 + (double)(t[j] - '0'); scale *= 10.0; ++j; any = true; }
        if (any) mant /= scale;   // divide AFTER assembling all digits (one rounding, compiler-stable)
        else return false;        // a bare '.' with no digits -> not a number
    }
    if (!any) return false;
    // exponent
    if (j < n && (t[j] == 'e' || t[j] == 'E')) {
        std::size_t k = j + 1;
        bool eneg = false;
        if (k < n && (t[k] == '+' || t[k] == '-')) { eneg = (t[k] == '-'); ++k; }
        if (k < n && IsDigit(t[k])) {
            int e = 0;
            while (k < n && IsDigit(t[k])) { e = e * 10 + (t[k] - '0'); ++k; }
            double p = 1.0;
            for (int x = 0; x < e; ++x) p *= 10.0;
            mant = eneg ? (mant / p) : (mant * p);
            j = k;
        }
        // a stray 'e' with no exponent digits: leave j before it (token ends at the integer/fraction)
    }
    out = neg ? -mant : mant;
    i = j;
    return true;
}

// Find the keyword `kw` (e.g. "points", "faceVertexCounts") as a whole identifier within [start, defEnd),
// then locate its `= [ ... ]` array and scan every numeric token into `out` (as double). Returns false if
// the keyword's array is absent or malformed (no matching ']'). `out` is appended to (caller clears).
inline bool ScanArray(const char* t, std::size_t n, std::size_t start, std::size_t defEnd,
                      const char* kw, std::vector<double>& out) {
    const std::size_t kwLen = std::strlen(kw);
    std::size_t i = start;
    while (i + kwLen <= defEnd) {
        // match `kw` as a token: preceded by a non-identifier char (or start) and followed by ws/'='/'['
        if (std::memcmp(t + i, kw, kwLen) == 0) {
            const char before = (i == 0) ? ' ' : t[i - 1];
            const char after  = (i + kwLen < n) ? t[i + kwLen] : ' ';
            const bool wordBefore = (before == '_' || (before >= 'a' && before <= 'z') ||
                                     (before >= 'A' && before <= 'Z') || IsDigit(before));
            const bool wordAfter  = (after == '_' || (after >= 'a' && after <= 'z') ||
                                     (after >= 'A' && after <= 'Z') || IsDigit(after));
            if (!wordBefore && !wordAfter) {
                // found the identifier; advance to the '=' then the '['
                std::size_t j = i + kwLen;
                SkipWsAndComments(t, n, j);
                if (j < defEnd && t[j] == '=') {
                    ++j;
                    SkipWsAndComments(t, n, j);
                    if (j < defEnd && t[j] == '[') {
                        ++j;
                        for (;;) {
                            SkipWsAndComments(t, n, j);
                            if (j >= defEnd) return false;      // unterminated array
                            if (t[j] == ']') { ++j; return true; }
                            // separators inside the array: '(' ')' ','
                            if (t[j] == '(' || t[j] == ')' || t[j] == ',') { ++j; continue; }
                            double v;
                            std::size_t before2 = j;
                            if (ParseNumber(t, n, j, v)) { out.push_back(v); }
                            else { ++j; }                       // skip an unexpected char, keep scanning
                            if (j == before2) ++j;              // guarantee forward progress
                        }
                    }
                }
                // not an array assignment for this occurrence; keep searching past it
                i = i + kwLen;
                continue;
            }
        }
        ++i;
    }
    return false;
}

// Find the byte range of the FIRST `def Mesh` prim body: returns [bodyStart, bodyEnd) spanning the braces'
// interior, and true on success. Scans for the token `def`, then `Mesh`, then the opening `{`, and matches
// the closing `}` by brace depth. Returns false if no `def Mesh { ... }` is present.
inline bool FindFirstMeshBody(const char* t, std::size_t n, std::size_t& bodyStart, std::size_t& bodyEnd) {
    std::size_t i = 0;
    while (i < n) {
        SkipWsAndComments(t, n, i);
        if (i + 3 <= n && std::memcmp(t + i, "def", 3) == 0 &&
            (i + 3 == n || IsSpace(t[i + 3]) || t[i + 3] == '"')) {
            std::size_t j = i + 3;
            SkipWsAndComments(t, n, j);
            // optional specifier could be a prim type token; we want "Mesh" specifically
            if (j + 4 <= n && std::memcmp(t + j, "Mesh", 4) == 0 &&
                (j + 4 == n || IsSpace(t[j + 4]) || t[j + 4] == '"' || t[j + 4] == '{')) {
                // advance to the '{' that opens this prim
                std::size_t k = j + 4;
                while (k < n && t[k] != '{' && t[k] != '}') ++k;
                if (k < n && t[k] == '{') {
                    bodyStart = k + 1;
                    int depth = 1;
                    std::size_t m = bodyStart;
                    while (m < n && depth > 0) {
                        if (t[m] == '{') ++depth;
                        else if (t[m] == '}') --depth;
                        ++m;
                    }
                    if (depth == 0) { bodyEnd = m - 1; return true; }   // m-1 points at the closing '}'
                    return false;                                       // unbalanced braces
                }
            }
            i = j;   // a `def` of some other type — keep scanning
            continue;
        }
        ++i;
    }
    return false;
}

}  // namespace usd_detail

// Parse USDA text -> UsdMesh (the FIRST def Mesh's points + triangulated faces). Deterministic, pure CPU.
// ok=false on a missing Mesh, missing/empty points or face arrays, or an out-of-range index.
inline UsdMesh ParseUsda(const char* text, std::size_t n) {
    using namespace usd_detail;
    UsdMesh out;
    if (!text || n == 0) return out;

    std::size_t bodyStart = 0, bodyEnd = 0;
    if (!FindFirstMeshBody(text, n, bodyStart, bodyEnd)) return out;

    std::vector<double> pts, counts, idx;
    const bool hasPts    = ScanArray(text, n, bodyStart, bodyEnd, "points", pts);
    const bool hasCounts = ScanArray(text, n, bodyStart, bodyEnd, "faceVertexCounts", counts);
    const bool hasIdx    = ScanArray(text, n, bodyStart, bodyEnd, "faceVertexIndices", idx);
    if (!hasPts || !hasCounts || !hasIdx) return out;
    if (pts.empty() || (pts.size() % 3) != 0) return out;
    if (counts.empty() || idx.empty()) return out;

    // positions: the parsed doubles -> float (the authored constants are exactly representable).
    out.positions.resize(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i) out.positions[i] = (float)pts[i];

    const uint32_t vertCount = (uint32_t)(pts.size() / 3);

    // Walk the faces: face f consumes counts[f] consecutive entries of idx; fan-triangulate each.
    std::size_t cursor = 0;   // running offset into idx
    for (std::size_t f = 0; f < counts.size(); ++f) {
        const double cd = counts[f];
        if (cd < 3.0) {                                   // a degenerate (point/edge) face — skip its indices
            if (cd > 0.0) cursor += (std::size_t)(cd + 0.5);
            continue;
        }
        const std::size_t fc = (std::size_t)(cd + 0.5);   // vertices in this face (rounded; authored ints)
        if (cursor + fc > idx.size()) { out.positions.clear(); out.indices.clear(); return out; }  // truncated
        // gather + range-check the face's corners
        std::vector<uint32_t> face;
        face.reserve(fc);
        for (std::size_t c = 0; c < fc; ++c) {
            const double vd = idx[cursor + c];
            if (vd < 0.0) { out.positions.clear(); out.indices.clear(); return out; }
            const uint32_t vi = (uint32_t)(vd + 0.5);
            if (vi >= vertCount) { out.positions.clear(); out.indices.clear(); return out; }
            face.push_back(vi);
        }
        cursor += fc;
        for (std::size_t k = 1; k + 1 < face.size(); ++k) {
            out.indices.push_back(face[0]);
            out.indices.push_back(face[k]);
            out.indices.push_back(face[k + 1]);
        }
    }

    out.ok = !out.indices.empty();
    if (!out.ok) { out.positions.clear(); out.indices.clear(); }
    return out;
}

inline UsdMesh ParseUsda(const std::string& text) { return ParseUsda(text.data(), text.size()); }

}  // namespace hf::asset
