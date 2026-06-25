#pragma once

// Wavefront OBJ geometry parser (issue #15, ask #1) — a dependency-free importer for the format every
// academic + indie pipeline still ships (the McGuire archive — Sponza/Bistro/Conference —, Stanford scans,
// every Blender export). The PURE parse (text -> ObjMesh) is split out of any device/RHI dependency so it
// is unit-testable standalone and golden-hashable: the issue's headline value-prop is "your asset pipeline
// is reproducible" — re-importing the same OBJ produces a BYTE-IDENTICAL mesh across machines/compilers,
// which this parser delivers and a pinned-hash test proves (unlike UE5's machine-dependent FP import).
//
// Self-contained: <cstddef>/<cstdint>/<cstdlib>/<cstring>/<string>/<vector> only. NO RHI/scene/<map>/
// <unordered_*>/std::hash. A thin device wrapper (LoadObjMesh -> scene::Mesh, computing smooth normals when
// absent + tangents + a neutral tint, exactly like the glTF BuildPrimitive) layers on top elsewhere.
//
// Scope (v1): geometry only — v / vt / vn / f, polygons fan-triangulated, 1-based + negative (relative)
// indices, the v, v/vt, v/vt/vn and v//vn corner forms. MTL materials, multi-submesh `usemtl` grouping,
// smoothing groups, and the w components are NOT handled yet (documented follow-ups). Unknown lines are
// skipped (forward-compatible, not fatal).

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace hf::asset {

// One deduplicated OBJ vertex — exactly the data the format carries. (color/tangent are render-derived and
// added by the device wrapper, not the file.) uv/normal default to 0 when the corner omits vt/vn.
struct ObjVertex {
    float pos[3]    = {0, 0, 0};
    float uv[2]     = {0, 0};
    float normal[3] = {0, 0, 0};
};

struct ObjMesh {
    std::vector<ObjVertex> vertices;   // deduplicated unique (v,vt,vn) corners
    std::vector<uint32_t>  indices;    // triangle list (3 per tri), indexing `vertices`
    uint32_t positions = 0;            // raw `v` count parsed (diagnostics)
    uint32_t faces     = 0;            // raw `f` count parsed (pre-triangulation)
};

namespace obj_detail {

// Resolve a 1-based-or-negative OBJ index against the current count. 1 => 0, -1 => count-1. 0/out-of-range
// => -1 (skip — malformed). `count` is the number parsed SO FAR (OBJ negatives are relative to that).
inline int32_t ResolveIndex(long raw, std::size_t count) {
    if (raw > 0) { long i = raw - 1; return (i < (long)count) ? (int32_t)i : -1; }
    if (raw < 0) { long i = (long)count + raw; return (i >= 0) ? (int32_t)i : -1; }
    return -1;   // 0 is not a valid OBJ index
}

// A parsed face corner: indices into the v / vt / vn arrays (already resolved to 0-based; -1 = absent).
struct Corner { int32_t v, vt, vn; };

// Pack a (v,vt,vn) corner into a uint64 key for dedup (v in the high bits; vt/vn never realistically
// exceed ~21 bits for these assets, but we mix all three so distinct corners get distinct keys).
inline uint64_t CornerKey(const Corner& c) {
    return (uint64_t)(uint32_t)c.v * 2654435761ull ^
           ((uint64_t)(uint32_t)c.vt << 32) ^ (uint64_t)(uint32_t)c.vn;
}

}  // namespace obj_detail

// Parse OBJ text -> ObjMesh. Deterministic + locale-independent-stable (uses the default C locale's '.'
// decimal point, as the engine + tests do). Pure CPU; no allocation beyond the output vectors + a dedup map.
inline ObjMesh ParseObj(const char* text, std::size_t len) {
    using obj_detail::Corner;
    ObjMesh out;
    std::vector<float> px, py, pz;           // positions
    std::vector<float> tu, tv;               // texcoords
    std::vector<float> nx, ny, nz;           // normals

    // dedup: sorted vector<(key, outIndex)> with binary-search insert (deterministic, no hash container).
    std::vector<std::pair<uint64_t, uint32_t>> dedup;

    auto emitCorner = [&](const Corner& c) -> uint32_t {
        const uint64_t key = obj_detail::CornerKey(c);
        std::size_t lo = 0, hi = dedup.size();
        while (lo < hi) { std::size_t mid = (lo + hi) / 2; if (dedup[mid].first < key) lo = mid + 1; else hi = mid; }
        if (lo < dedup.size() && dedup[lo].first == key) return dedup[lo].second;
        ObjVertex v;
        if (c.v >= 0)  { v.pos[0] = px[c.v];  v.pos[1] = py[c.v];  v.pos[2] = pz[c.v]; }
        if (c.vt >= 0) { v.uv[0] = tu[c.vt];  v.uv[1] = tv[c.vt]; }
        if (c.vn >= 0) { v.normal[0] = nx[c.vn]; v.normal[1] = ny[c.vn]; v.normal[2] = nz[c.vn]; }
        const uint32_t idx = (uint32_t)out.vertices.size();
        out.vertices.push_back(v);
        dedup.insert(dedup.begin() + (std::ptrdiff_t)lo, {key, idx});
        return idx;
    };

    // Parse one face corner token "v", "v/vt", "v/vt/vn", "v//vn" -> resolved Corner.
    auto parseCorner = [&](const char* s, const char* e) -> Corner {
        Corner c{-1, -1, -1};
        char* end = nullptr;
        long v = std::strtol(s, &end, 10);
        c.v = obj_detail::ResolveIndex(v, px.size());
        if (end < e && *end == '/') {
            const char* s2 = end + 1;
            if (s2 < e && *s2 != '/') { long vt = std::strtol(s2, &end, 10); c.vt = obj_detail::ResolveIndex(vt, tu.size()); }
            else end = (char*)s2;
            if (end < e && *end == '/') { long vn = std::strtol(end + 1, &end, 10); c.vn = obj_detail::ResolveIndex(vn, nx.size()); }
        }
        return c;
    };

    std::size_t i = 0;
    while (i < len) {
        std::size_t ls = i;
        while (i < len && text[i] != '\n') ++i;
        std::size_t le = i;                       // [ls,le) is the line (no newline)
        if (i < len) ++i;                         // step past '\n'
        if (le > ls && text[le - 1] == '\r') --le; // strip CR
        const char* p = text + ls;
        const char* e = text + le;
        while (p < e && (*p == ' ' || *p == '\t')) ++p;
        if (p >= e || *p == '#') continue;        // blank / comment

        if (p[0] == 'v' && (p + 1 < e) && (p[1] == ' ' || p[1] == '\t')) {
            char* end = nullptr;
            float x = std::strtof(p + 1, &end); float y = std::strtof(end, &end); float z = std::strtof(end, &end);
            px.push_back(x); py.push_back(y); pz.push_back(z); ++out.positions;
        } else if (p[0] == 'v' && p[1] == 't') {
            char* end = nullptr;
            float u = std::strtof(p + 2, &end); float v = std::strtof(end, &end);
            tu.push_back(u); tv.push_back(v);
        } else if (p[0] == 'v' && p[1] == 'n') {
            char* end = nullptr;
            float x = std::strtof(p + 2, &end); float y = std::strtof(end, &end); float z = std::strtof(end, &end);
            nx.push_back(x); ny.push_back(y); nz.push_back(z);
        } else if (p[0] == 'f' && (p + 1 < e) && (p[1] == ' ' || p[1] == '\t')) {
            // Collect the face's corners, then fan-triangulate (v0, vk, vk+1).
            std::vector<uint32_t> face;
            const char* q = p + 1;
            while (q < e) {
                while (q < e && (*q == ' ' || *q == '\t')) ++q;
                if (q >= e) break;
                const char* ts = q;
                while (q < e && *q != ' ' && *q != '\t') ++q;
                Corner c = parseCorner(ts, q);
                if (c.v >= 0) face.push_back(emitCorner(c));
            }
            if (face.size() >= 3) {
                ++out.faces;
                for (std::size_t k = 1; k + 1 < face.size(); ++k) {
                    out.indices.push_back(face[0]); out.indices.push_back(face[k]); out.indices.push_back(face[k + 1]);
                }
            }
        }
        // all other keywords (vt w, usemtl, mtllib, g, o, s, ...) are skipped in v1
    }
    return out;
}

// Convenience: parse from a std::string.
inline ObjMesh ParseObj(const std::string& text) { return ParseObj(text.data(), text.size()); }

}  // namespace hf::asset
