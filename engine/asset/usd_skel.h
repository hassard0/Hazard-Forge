#pragma once

// Slice SK1 — DETERMINISTIC SKELETAL-ANIMATION ASSET IMPORT (USD / UsdSkel).
// ==========================================================================================================
// PREMISE (the parity++ asset-import gap). The engine has THREE mesh importers:
//   * gltf_loader.h  — the RICH one: LoadSkinnedGltfModel ALREADY imports skin + skeleton + inverse-bind +
//                      JOINTS_0/WEIGHTS_0 + animation clips (via cgltf, and it uploads to an rhi device).
//   * fbx_loader.h   — binary FBX, GEOMETRY ONLY (skeleton/skin/anim are documented follow-ups).
//   * usd_loader.h   — USDA text, GEOMETRY ONLY (the FIRST `def Mesh`'s points/faces; skeleton/skin/anim
//                      are documented follow-ups).
// So the glTF path is the ONLY skeletal importer, and it is device-coupled. SK1 closes the highest-value
// ADJACENT gap: a PURE-CPU, dependency-free, byte-identical **USD skeletal importer** (UsdSkel) — the
// interchange schema Maya/Houdini/Blender/Omniverse + Unreal's USD stage all speak. It is the cleanest
// deterministic source after glTF (plain text + numeric arrays, NO proprietary binary, NO strtod), and it
// composes with the EXISTING anim::Skeleton / anim::Animation / skinned-vertex / retarget stack: an imported
// USD rig can be posed by SampleAnimation, skinned, retargeted, and blended with zero new anim code.
//
// HONESTY: this is NOT "add skeletal import to the engine" — glTF already does that. SK1's real contribution
// is a SECOND, format-diverse, DEVICE-FREE skeletal importer (USD), matching the clean-room usd_loader.h /
// fbx_loader.h discipline (UsdSkelImport + a pinned net::DigestBytes, cross-compiler exact BY CONSTRUCTION).
//
// THE UsdSkel FORMAT (text; the subset SK1 reads):
//   def SkelRoot "Char" {
//     def Skeleton "Rig" {
//       uniform token[]     joints         = ["Root", "Root/Hip", "Root/Hip/Knee"]   # parent = path prefix
//       uniform matrix4d[]  bindTransforms = [ (...16...), ... ]   # WORLD bind, row-major -> inverseBind
//       uniform matrix4d[]  restTransforms = [ (...16...), ... ]   # LOCAL rest -> decomposed to TRS
//     }
//     def SkelAnimation "Motion" {
//       uniform token[]  joints                   = ["Root/Hip", ...]          # the animated subset
//       quatf[]          rotations.timeSamples    = { 0: [(w,x,y,z),...], ... } # USD quat = (real,i,j,k)
//       float3[]         translations.timeSamples = { 0: [(x,y,z),...], ... }
//       float3[]         scales.timeSamples       = { ... }
//     }
//     def Mesh "Body" {
//       point3f[] points ...  int[] faceVertexCounts ...  int[] faceVertexIndices ...   # (via ParseUsda)
//       int[]   primvars:skel:jointIndices = [...]   # elementSize influences per vertex, into Skeleton order
//       float[] primvars:skel:jointWeights = [...]
//     }
//   }
//
// DETERMINISM. Numeric tokens are hand-parsed by usd_loader.h's ParseNumber (no strtod). USD matrix4d is
// row-major/row-vector; math::Mat4 is column-major/column-vector, and the two conventions CANCEL so the 16
// authored doubles copy STRAIGHT into m[] (element_eng(r,c)==element_usd(c,r)==d[c*4+r]==m[c*4+r]). inverseBind
// = bindWorld.Inverse() (translation matrices -> exact). rest TRS = DecomposeTRS(restLocal). USD quats are
// reordered (w,x,y,z)->(x,y,z,w). Skin weights are truncated to the top-4 influences + renormalized to sum 1
// (the standard). Joints are topologically reordered (parent<child) so anim.h's single-pass FK is valid.
//
// SCOPE (v1) — documented deferrals: USDA TEXT only (no binary/crate USD); the FIRST SkelRoot's first
// Skeleton + first SkelAnimation + first Mesh; LINEAR sample interpolation (USD timeSamples carry no tangents
// -> no CUBICSPLINE); NO blend shapes / morph targets; NO skel:animationSource relationship resolution
// (proximity-matched); DecomposeTRS assumes TRS-only rest matrices (no shear); non-uniform timeCodesPerSecond
// honored, default 24. Malformed input -> ok=false (bounds-checked, forward-compatible skips).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "anim/animation.h"     // hf::anim::Animation / Channel (the animation-clip target type)
#include "anim/skeleton.h"      // hf::anim::Skeleton / Joint     (the skeleton target type)
#include "math/math.h"          // Mat4 (Inverse) / Quat / Vec3   (bind/rest transform math)
#include "scene/vertex.h"       // scene::SkinnedVertex           (the skinned-mesh vertex target type)
#include "net/session.h"        // hf::net::DigestBytes           (the pinned state-digest currency)
#include "asset/usd_loader.h"   // usd_detail::{ParseNumber,SkipWsAndComments,IsSpace,IsDigit,ScanArray} + ParseUsda

namespace hf::asset {

// The imported rig: an anim::Skeleton (topologically sorted), the per-joint USD path names, the skinned mesh
// (SkinnedVertex quads + triangle indices), and every animation clip — everything the existing anim + skinning
// + retarget stack consumes. `ok` is true iff a Skeleton with joints + bind/rest decoded.
struct UsdSkelImport {
    anim::Skeleton                    skeleton;             // joints in topological order (parent < child)
    std::vector<std::string>          jointNames;           // per skeleton joint: the USD joint path (skel order)
    std::vector<scene::SkinnedVertex> vertices;             // skinned mesh (pos/uv/joints/weights); normal 0, tangent x
    std::vector<uint32_t>             indices;              // triangle list (from ParseUsda)
    std::vector<anim::Animation>      animations;           // clips, channels remapped to skeleton joint indices
    float                             timeCodesPerSecond = 24.0f;
    bool                              ok = false;
};

namespace usdskel_detail {

using usd_detail::IsSpace;
using usd_detail::IsDigit;
using usd_detail::SkipWsAndComments;
using usd_detail::ParseNumber;
using usd_detail::ScanArray;

inline bool IsIdentChar(char c) {
    return c == '_' || c == ':' || c == '.' ||
           (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || IsDigit(c);
}

// Find the body [bodyStart, bodyEnd) of the FIRST `def <typeKw> "..." { ... }` prim (braces interior).
// Scans for the token `def`, then the exact type keyword, then brace-matches. Returns false if absent.
inline bool FindFirstPrimBody(const char* t, std::size_t n, const char* typeKw,
                              std::size_t& bodyStart, std::size_t& bodyEnd) {
    const std::size_t kwLen = std::strlen(typeKw);
    std::size_t i = 0;
    while (i < n) {
        SkipWsAndComments(t, n, i);
        if (i + 3 <= n && std::memcmp(t + i, "def", 3) == 0 &&
            (i + 3 == n || IsSpace(t[i + 3]))) {
            std::size_t j = i + 3;
            SkipWsAndComments(t, n, j);
            if (j + kwLen <= n && std::memcmp(t + j, typeKw, kwLen) == 0 &&
                (j + kwLen == n || IsSpace(t[j + kwLen]) || t[j + kwLen] == '"' || t[j + kwLen] == '{')) {
                std::size_t k = j + kwLen;
                while (k < n && t[k] != '{' && t[k] != '}') ++k;   // advance to this prim's opening brace
                if (k < n && t[k] == '{') {
                    bodyStart = k + 1;
                    int depth = 1;
                    std::size_t m = bodyStart;
                    while (m < n && depth > 0) {
                        if (t[m] == '{') ++depth;
                        else if (t[m] == '}') --depth;
                        ++m;
                    }
                    if (depth == 0) { bodyEnd = m - 1; return true; }
                    return false;
                }
            }
            i = j;   // a `def` of another type — keep scanning
            continue;
        }
        ++i;
    }
    return false;
}

// Match `kw` as a whole identifier at t[i], bounded by non-identifier chars on both sides.
inline bool MatchKeyword(const char* t, std::size_t n, std::size_t i, const char* kw, std::size_t kwLen) {
    if (i + kwLen > n || std::memcmp(t + i, kw, kwLen) != 0) return false;
    const char before = (i == 0) ? ' ' : t[i - 1];
    const char after  = (i + kwLen < n) ? t[i + kwLen] : ' ';
    return !IsIdentChar(before) && !IsIdentChar(after);
}

// Scan `<kw> = [ "a", "b/c", ... ]` -> out (the quoted string tokens). Returns false if absent/malformed.
inline bool ScanTokenArray(const char* t, std::size_t n, std::size_t start, std::size_t end,
                           const char* kw, std::vector<std::string>& out) {
    const std::size_t kwLen = std::strlen(kw);
    for (std::size_t i = start; i + kwLen <= end; ++i) {
        if (!MatchKeyword(t, n, i, kw, kwLen)) continue;
        std::size_t j = i + kwLen;
        SkipWsAndComments(t, n, j);
        if (j >= end || t[j] != '=') { i = i + kwLen; continue; }
        ++j; SkipWsAndComments(t, n, j);
        if (j >= end || t[j] != '[') { i = i + kwLen; continue; }
        ++j;
        for (;;) {
            SkipWsAndComments(t, n, j);
            if (j >= end) return false;               // unterminated
            if (t[j] == ']') return true;
            if (t[j] == ',') { ++j; continue; }
            if (t[j] == '"') {
                ++j;
                std::string s;
                while (j < end && t[j] != '"') { s.push_back(t[j]); ++j; }
                if (j >= end) return false;            // unterminated string
                ++j;                                   // past closing quote
                out.push_back(std::move(s));
                continue;
            }
            ++j;                                       // skip stray char
        }
    }
    return false;
}

// One time-sample stream: sorted sample times (frames) + a flat value array per time (each `values[k]` holds
// all the numeric tokens between the k-th `[ ... ]`). The caller reshapes by per-joint stride.
struct TimeSamples {
    std::vector<double>              times;   // frame numbers (USD time codes)
    std::vector<std::vector<double>> values;  // parallel: values[k] = flat numbers of the k-th sample
    bool ok = false;
};

// Scan `<attr>.timeSamples = { t0: [..], t1: [..], ... }` (parens/commas inside a sample are separators).
inline TimeSamples ScanTimeSamples(const char* t, std::size_t n, std::size_t start, std::size_t end,
                                   const char* attr) {
    TimeSamples ts;
    std::string kwStr = std::string(attr) + ".timeSamples";
    const char* kw = kwStr.c_str();
    const std::size_t kwLen = kwStr.size();
    for (std::size_t i = start; i + kwLen <= end; ++i) {
        if (!MatchKeyword(t, n, i, kw, kwLen)) continue;
        std::size_t j = i + kwLen;
        SkipWsAndComments(t, n, j);
        if (j >= end || t[j] != '=') return ts;
        ++j; SkipWsAndComments(t, n, j);
        if (j >= end || t[j] != '{') return ts;
        ++j;
        for (;;) {
            SkipWsAndComments(t, n, j);
            if (j >= end) return ts;                  // unterminated
            if (t[j] == '}') { ts.ok = true; return ts; }
            if (t[j] == ',') { ++j; continue; }
            double tcode = 0.0;
            if (!ParseNumber(t, n, j, tcode)) { ++j; continue; }   // expect a time code
            SkipWsAndComments(t, n, j);
            if (j >= end || t[j] != ':') return ts;
            ++j; SkipWsAndComments(t, n, j);
            if (j >= end || t[j] != '[') return ts;
            ++j;
            std::vector<double> sample;
            for (;;) {
                SkipWsAndComments(t, n, j);
                if (j >= end) return ts;
                if (t[j] == ']') { ++j; break; }
                if (t[j] == '(' || t[j] == ')' || t[j] == ',') { ++j; continue; }
                double v; std::size_t before = j;
                if (ParseNumber(t, n, j, v)) sample.push_back(v);
                else ++j;
                if (j == before) ++j;
            }
            ts.times.push_back(tcode);
            ts.values.push_back(std::move(sample));
        }
    }
    return ts;
}

// Scan a scalar `<kw> = <number>` in [start,end). Returns false if absent.
inline bool ScanScalar(const char* t, std::size_t n, std::size_t start, std::size_t end,
                       const char* kw, double& out) {
    const std::size_t kwLen = std::strlen(kw);
    for (std::size_t i = start; i + kwLen <= end; ++i) {
        if (!MatchKeyword(t, n, i, kw, kwLen)) continue;
        std::size_t j = i + kwLen;
        SkipWsAndComments(t, n, j);
        if (j >= end || t[j] != '=') { i = i + kwLen; continue; }
        ++j; SkipWsAndComments(t, n, j);
        double v;
        if (ParseNumber(t, n, j, v)) { out = v; return true; }
    }
    return false;
}

// Read a USD row-major matrix4d (16 doubles) straight into a column-major math::Mat4 (see banner: the
// row/column-major + row/column-vector conventions cancel, so m[k] == d[k]).
inline math::Mat4 Mat4FromUsd(const double* d) {
    math::Mat4 m;
    for (int k = 0; k < 16; ++k) m.m[k] = (float)d[k];
    return m;
}

// DecomposeTRS: split a LOCAL transform matrix into translation + rotation + scale (assumes no shear; the
// documented v1 scope). Deterministic: column lengths for scale, normalized-column trace method for the
// quaternion. Identity/translation matrices decompose EXACTLY (sqrt(1)==1, trace==3 -> w==1, xyz==0).
inline void DecomposeTRS(const math::Mat4& M, math::Vec3& t, math::Quat& r, math::Vec3& s) {
    t = math::Vec3{M.m[12], M.m[13], M.m[14]};
    // basis columns (element(row,col) == m[col*4+row])
    const float c0x = M.m[0], c0y = M.m[1], c0z = M.m[2];
    const float c1x = M.m[4], c1y = M.m[5], c1z = M.m[6];
    const float c2x = M.m[8], c2y = M.m[9], c2z = M.m[10];
    const float sx = std::sqrt(c0x*c0x + c0y*c0y + c0z*c0z);
    const float sy = std::sqrt(c1x*c1x + c1y*c1y + c1z*c1z);
    const float sz = std::sqrt(c2x*c2x + c2y*c2y + c2z*c2z);
    s = math::Vec3{sx, sy, sz};
    const float ix = sx > 0.0f ? 1.0f / sx : 0.0f;
    const float iy = sy > 0.0f ? 1.0f / sy : 0.0f;
    const float iz = sz > 0.0f ? 1.0f / sz : 0.0f;
    // normalized rotation basis; a(row,col)
    const float a00 = c0x*ix, a10 = c0y*ix, a20 = c0z*ix;   // col0
    const float a01 = c1x*iy, a11 = c1y*iy, a21 = c1z*iy;   // col1
    const float a02 = c2x*iz, a12 = c2y*iz, a22 = c2z*iz;   // col2
    const float trace = a00 + a11 + a22;
    math::Quat q;
    if (trace > 0.0f) {
        const float S = std::sqrt(trace + 1.0f) * 2.0f;      // S = 4w
        q.w = 0.25f * S;
        q.x = (a21 - a12) / S;
        q.y = (a02 - a20) / S;
        q.z = (a10 - a01) / S;
    } else if (a00 > a11 && a00 > a22) {
        const float S = std::sqrt(1.0f + a00 - a11 - a22) * 2.0f;   // S = 4x
        q.w = (a21 - a12) / S;
        q.x = 0.25f * S;
        q.y = (a01 + a10) / S;
        q.z = (a02 + a20) / S;
    } else if (a11 > a22) {
        const float S = std::sqrt(1.0f + a11 - a00 - a22) * 2.0f;   // S = 4y
        q.w = (a02 - a20) / S;
        q.x = (a01 + a10) / S;
        q.y = 0.25f * S;
        q.z = (a12 + a21) / S;
    } else {
        const float S = std::sqrt(1.0f + a22 - a00 - a11) * 2.0f;   // S = 4z
        q.w = (a10 - a01) / S;
        q.x = (a02 + a20) / S;
        q.y = (a12 + a21) / S;
        q.z = 0.25f * S;
    }
    r = math::Normalize(q);
}

// The parent joint path of a USD joint path: the substring up to the last '/', or "" for a root.
inline std::string ParentPath(const std::string& p) {
    const std::size_t slash = p.find_last_of('/');
    return slash == std::string::npos ? std::string() : p.substr(0, slash);
}

}  // namespace usdskel_detail

// NormalizeInfluences: reduce a vertex's (index,weight) influence list to the skinned-vertex 4-quad — take
// the TOP-4 by weight (stable: ties keep the lower influence slot), renormalize the kept weights to sum 1.0,
// pad unused slots with (index 0, weight 0). The standard >4-influence handling; exposed for direct testing.
inline void NormalizeInfluences(const int32_t* idx, const float* wt, std::size_t count,
                                float outJoints[4], float outWeights[4]) {
    // stable selection of the 4 largest weights
    int sel[4] = {-1, -1, -1, -1};
    for (std::size_t k = 0; k < count; ++k) {
        for (int slot = 0; slot < 4; ++slot) {
            if (sel[slot] < 0) { sel[slot] = (int)k; break; }
            if (wt[k] > wt[sel[slot]]) {                     // strictly greater -> displace (ties keep earlier)
                for (int m = 3; m > slot; --m) sel[m] = sel[m - 1];
                sel[slot] = (int)k;
                break;
            }
        }
    }
    float sum = 0.0f;
    for (int slot = 0; slot < 4; ++slot) if (sel[slot] >= 0) sum += wt[sel[slot]];
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int slot = 0; slot < 4; ++slot) {
        if (sel[slot] >= 0) {
            outJoints[slot]  = (float)idx[sel[slot]];
            outWeights[slot] = wt[sel[slot]] * inv;
        } else {
            outJoints[slot] = 0.0f; outWeights[slot] = 0.0f;
        }
    }
}

// ImportUsdSkel: parse UsdSkel text -> a device-free UsdSkelImport (skeleton + skin + clips). Deterministic,
// pure CPU. ok=false on a missing/malformed Skeleton. See the banner for the format subset + scope.
inline UsdSkelImport ImportUsdSkel(const char* text, std::size_t n) {
    using namespace usdskel_detail;
    UsdSkelImport out;
    if (!text || n == 0) return out;

    // --- stage metadata: timeCodesPerSecond (default 24) --------------------------------------------------
    {
        double tcps = 24.0;
        // scan only the leading metadata region (before the first `def`) to avoid matching prim attributes
        std::size_t defAt = 0; bool found = false;
        for (std::size_t i = 0; i + 3 <= n; ++i)
            if (std::memcmp(text + i, "def", 3) == 0) { defAt = i; found = true; break; }
        const std::size_t metaEnd = found ? defAt : n;
        if (ScanScalar(text, n, 0, metaEnd, "timeCodesPerSecond", tcps) && tcps > 0.0)
            out.timeCodesPerSecond = (float)tcps;
    }

    // --- Skeleton: joints (paths) + bindTransforms + restTransforms --------------------------------------
    std::size_t sBeg = 0, sEnd = 0;
    if (!FindFirstPrimBody(text, n, "Skeleton", sBeg, sEnd)) return out;

    std::vector<std::string> jointPaths;
    if (!ScanTokenArray(text, n, sBeg, sEnd, "joints", jointPaths) || jointPaths.empty()) return out;
    const std::size_t nJ = jointPaths.size();

    std::vector<double> bindD, restD;
    const bool hasBind = ScanArray(text, n, sBeg, sEnd, "bindTransforms", bindD);
    const bool hasRest = ScanArray(text, n, sBeg, sEnd, "restTransforms", restD);
    if (!hasRest || restD.size() != nJ * 16) return out;             // rest is required (local TRS source)
    const bool bindOk = hasBind && bindD.size() == nJ * 16;

    // parent index of each joint (authored order) by path-prefix
    std::vector<int> parentAuthored(nJ, -1);
    for (std::size_t j = 0; j < nJ; ++j) {
        const std::string pp = ParentPath(jointPaths[j]);
        if (pp.empty()) continue;
        for (std::size_t p = 0; p < nJ; ++p)
            if (p != j && jointPaths[p] == pp) { parentAuthored[j] = (int)p; break; }
    }

    // topological reorder: emit a joint once its parent has been emitted (stable authored-order tiebreak)
    std::vector<int> oldToNew(nJ, -1), newToOld;
    newToOld.reserve(nJ);
    {
        std::vector<char> placed(nJ, 0);
        bool progress = true;
        while (newToOld.size() < nJ && progress) {
            progress = false;
            for (std::size_t j = 0; j < nJ; ++j) {
                if (placed[j]) continue;
                const int pa = parentAuthored[j];
                if (pa < 0 || placed[pa]) {
                    oldToNew[j] = (int)newToOld.size();
                    newToOld.push_back((int)j);
                    placed[j] = 1;
                    progress = true;
                }
            }
        }
        if (newToOld.size() != nJ) return out;   // cyclic / dangling parent -> malformed
    }

    out.skeleton.joints.resize(nJ);
    out.jointNames.resize(nJ);
    for (std::size_t ni = 0; ni < nJ; ++ni) {
        const int oj = newToOld[ni];
        out.jointNames[ni] = jointPaths[oj];
        anim::Joint& J = out.skeleton.joints[ni];
        J.parent = parentAuthored[oj] < 0 ? -1 : oldToNew[parentAuthored[oj]];
        // rest LOCAL TRS
        math::Mat4 restM = Mat4FromUsd(&restD[(std::size_t)oj * 16]);
        DecomposeTRS(restM, J.t, J.r, J.s);
        // inverse-bind: inverse of the WORLD bind transform (fall back to identity if bind absent)
        if (bindOk) J.inverseBind = Mat4FromUsd(&bindD[(std::size_t)oj * 16]).Inverse();
        else        J.inverseBind = math::Mat4::Identity();
    }

    // --- Skinned mesh: geometry (ParseUsda) + skin primvars ----------------------------------------------
    {
        const UsdMesh mesh = ParseUsda(text, n);
        std::size_t mBeg = 0, mEnd = 0;
        const bool haveMeshBody = FindFirstPrimBody(text, n, "Mesh", mBeg, mEnd);
        if (mesh.ok && haveMeshBody) {
            out.indices = mesh.indices;
            const std::size_t vCount = mesh.positions.size() / 3;
            std::vector<double> jiD, jwD;
            const bool hasJI = ScanArray(text, n, mBeg, mEnd, "primvars:skel:jointIndices", jiD);
            const bool hasJW = ScanArray(text, n, mBeg, mEnd, "primvars:skel:jointWeights", jwD);
            std::size_t elem = 0;
            if (hasJI && hasJW && jiD.size() == jwD.size() && vCount > 0 && (jiD.size() % vCount) == 0)
                elem = jiD.size() / vCount;
            out.vertices.resize(vCount);
            for (std::size_t v = 0; v < vCount; ++v) {
                scene::SkinnedVertex sv{};
                sv.pos[0] = mesh.positions[v * 3 + 0];
                sv.pos[1] = mesh.positions[v * 3 + 1];
                sv.pos[2] = mesh.positions[v * 3 + 2];
                sv.color[0] = sv.color[1] = sv.color[2] = 1.0f;
                sv.tangent[0] = 1.0f;                       // default tangent (UsdSkel carries no vertex TBN here)
                if (elem > 0) {
                    std::vector<int32_t> vi(elem);
                    std::vector<float>   vw(elem);
                    for (std::size_t e = 0; e < elem; ++e) {
                        const int rawJoint = (int)(jiD[v * elem + e] + 0.5);   // authored USD joint index
                        vi[e] = (rawJoint >= 0 && (std::size_t)rawJoint < nJ) ? oldToNew[rawJoint] : 0;
                        vw[e] = (float)jwD[v * elem + e];
                    }
                    NormalizeInfluences(vi.data(), vw.data(), elem, sv.joints, sv.weights);
                } else {
                    sv.joints[0] = 0.0f; sv.weights[0] = 1.0f;   // rigid bind to root if no skin data
                }
                out.vertices[v] = sv;
            }
        }
    }

    // --- SkelAnimation: rotations/translations/scales timeSamples -> anim::Animation clip ----------------
    {
        std::size_t aBeg = 0, aEnd = 0;
        if (FindFirstPrimBody(text, n, "SkelAnimation", aBeg, aEnd)) {
            std::vector<std::string> animJoints;
            ScanTokenArray(text, n, aBeg, aEnd, "joints", animJoints);
            if (!animJoints.empty()) {
                // map each anim joint -> skeleton index by path
                std::vector<int> animToSkel(animJoints.size(), -1);
                for (std::size_t aj = 0; aj < animJoints.size(); ++aj)
                    for (std::size_t sk = 0; sk < nJ; ++sk)
                        if (out.jointNames[sk] == animJoints[aj]) { animToSkel[aj] = (int)sk; break; }

                const float invTcps = out.timeCodesPerSecond > 0.0f ? 1.0f / out.timeCodesPerSecond : 1.0f;
                anim::Animation clip;
                clip.name = "SkelAnimation";
                float duration = 0.0f;

                // Build channels for one attribute stream (stride 3 = T/S, 4 = R). USD quat order (w,x,y,z)
                // is reordered to (x,y,z,w) when `isRotation`.
                auto buildChannels = [&](const char* attr, anim::Channel::Path path, int stride, bool isRotation) {
                    TimeSamples ts = ScanTimeSamples(text, n, aBeg, aEnd, attr);
                    if (!ts.ok || ts.times.empty()) return;
                    const std::size_t nAj = animJoints.size();
                    for (std::size_t aj = 0; aj < nAj; ++aj) {
                        if (animToSkel[aj] < 0) continue;
                        anim::Channel ch;
                        ch.jointIndex = animToSkel[aj];
                        ch.path = path;
                        ch.interp = anim::Channel::Interp::Linear;
                        for (std::size_t k = 0; k < ts.times.size(); ++k) {
                            const std::vector<double>& sv = ts.values[k];
                            if (sv.size() < (aj + 1) * (std::size_t)stride) continue;   // malformed sample row
                            const float tSec = (float)ts.times[k] * invTcps;
                            ch.times.push_back(tSec);
                            if (tSec > duration) duration = tSec;
                            const double* base = &sv[aj * stride];
                            if (isRotation) {   // (w,x,y,z) -> (x,y,z,w)
                                ch.values.push_back((float)base[1]);
                                ch.values.push_back((float)base[2]);
                                ch.values.push_back((float)base[3]);
                                ch.values.push_back((float)base[0]);
                            } else {
                                for (int c = 0; c < stride; ++c) ch.values.push_back((float)base[c]);
                            }
                        }
                        if (!ch.times.empty()) clip.channels.push_back(std::move(ch));
                    }
                };
                buildChannels("rotations",    anim::Channel::Path::Rotation,    4, true);
                buildChannels("translations", anim::Channel::Path::Translation, 3, false);
                buildChannels("scales",       anim::Channel::Path::Scale,       3, false);
                clip.duration = duration;
                if (!clip.channels.empty()) out.animations.push_back(std::move(clip));
            }
        }
    }

    out.ok = !out.skeleton.joints.empty();
    return out;
}

inline UsdSkelImport ImportUsdSkel(const std::string& text) { return ImportUsdSkel(text.data(), text.size()); }

// DigestUsdSkel: FNV-1a-64 over the imported rig — skeleton (parent + inverseBind + TRS), joint names, skinned
// vertices (pos + joints + weights), triangle indices, and every clip (channel target/path/interp + keyframe
// times + values). The cross-platform moat currency; the authored fixture's bind/rest are translation-only so
// every folded float is exactly representable -> cross-compiler exact BY CONSTRUCTION.
inline uint64_t DigestUsdSkel(const UsdSkelImport& s) {
    std::vector<uint8_t> buf;
    auto putU32 = [&](uint32_t v) { const uint8_t* p = (const uint8_t*)&v; buf.insert(buf.end(), p, p + 4); };
    auto putF   = [&](float v)    { const uint8_t* p = (const uint8_t*)&v; buf.insert(buf.end(), p, p + 4); };
    auto putStr = [&](const std::string& s2) { buf.insert(buf.end(), s2.begin(), s2.end()); putU32(0xffffffffu); };

    putU32((uint32_t)s.skeleton.joints.size());
    for (std::size_t j = 0; j < s.skeleton.joints.size(); ++j) {
        const anim::Joint& J = s.skeleton.joints[j];
        putU32((uint32_t)(int32_t)J.parent);
        putStr(s.jointNames[j]);
        for (int k = 0; k < 16; ++k) putF(J.inverseBind.m[k]);
        putF(J.t.x); putF(J.t.y); putF(J.t.z);
        putF(J.r.x); putF(J.r.y); putF(J.r.z); putF(J.r.w);
        putF(J.s.x); putF(J.s.y); putF(J.s.z);
    }
    putU32((uint32_t)s.vertices.size());
    for (const scene::SkinnedVertex& v : s.vertices) {
        putF(v.pos[0]); putF(v.pos[1]); putF(v.pos[2]);
        for (int k = 0; k < 4; ++k) putF(v.joints[k]);
        for (int k = 0; k < 4; ++k) putF(v.weights[k]);
    }
    putU32((uint32_t)s.indices.size());
    for (uint32_t i : s.indices) putU32(i);
    putU32((uint32_t)s.animations.size());
    for (const anim::Animation& a : s.animations) {
        putStr(a.name); putF(a.duration); putU32((uint32_t)a.channels.size());
        for (const anim::Channel& c : a.channels) {
            putU32((uint32_t)(int32_t)c.jointIndex);
            putU32((uint32_t)c.path); putU32((uint32_t)c.interp);
            putU32((uint32_t)c.times.size());
            for (float t : c.times) putF(t);
            for (float x : c.values) putF(x);
        }
    }
    return net::DigestBytes(buf.data(), buf.size());
}

}  // namespace hf::asset
