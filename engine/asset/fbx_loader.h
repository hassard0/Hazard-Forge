#pragma once

// Binary FBX ("Kaydara FBX Binary") geometry parser (issue #15, ask #2) — a clean-room, dependency-free
// importer for Autodesk's interchange format that drives most DCC/game pipelines (Maya, 3ds Max, the
// FBX SDK exporters every studio ships). Like obj_loader.h, the PURE parse (raw bytes -> FbxMesh) is split
// from any device/RHI dependency so it is unit-testable standalone and golden-hashable: re-importing the
// same FBX produces a BYTE-IDENTICAL mesh across machines/compilers (the engine's moat), which a pinned
// net::DigestBytes test proves — unlike the FBX SDK's machine/float-dependent import.
//
// SELF-CONTAINED on purpose: <cstddef>/<cstdint>/<cstring>/<string>/<vector> + net/session.h (DigestBytes)
// only. NO RHI/scene/<map>/<unordered_*>/<zlib.h>. A full clean-room INFLATE (DEFLATE, RFC 1951, + the
// 2-byte zlib header RFC 1950) is implemented here because most binary FBX exporters zlib-COMPRESS their
// geometry arrays (encoding==1) — the assimp box.fbx fixture happens to store raw (encoding==0), so the
// raw path is the one the golden exercises, but the inflate path is real and round-trip-tested so any
// production FBX loads. A thin device wrapper (FbxMesh -> scene::Mesh, computing smooth normals + tangents)
// would layer on top elsewhere, exactly like obj_loader's LoadObjMesh.
//
// FORMAT (the binary container, all little-endian):
//   * Header: 23-byte magic "Kaydara FBX Binary  \x00\x1a\x00" + uint32 version.
//   * A tree of NODE RECORDS. For version < 7500 a record is: uint32 endOffset, uint32 numProperties,
//     uint32 propertyListLen, uint8 nameLen, char name[nameLen], propertyList, nested children, then a
//     13-byte NULL record terminates a sibling list. Version >= 7500 widens endOffset/numProperties/
//     propertyListLen to uint64 (so the null terminator is 25 bytes) — both are handled via the version.
//   * Properties carry a 1-char type code: Y=int16 C=bool I=int32 F=float32 D=float64 L=int64 R=raw S=string;
//     ARRAY types f/d/l/i/b (float/double/int64/int32/bool): uint32 arrayLength, uint32 encoding
//     (0=raw, 1=zlib-DEFLATE), uint32 compressedLength, then the (possibly compressed) payload.
//   * Mesh geometry lives in `Geometry` nodes: `Vertices` (a double array of xyz triples) and
//     `PolygonVertexIndex` (an int32 array; each polygon's LAST corner is stored as the bitwise-NOT of the
//     index, i.e. a NEGATIVE value (-idx-1), delimiting the polygon). Polygons are fan-triangulated.
//
// Scope (v1): the first Geometry's positions + triangulated indices only. Normals/UVs (LayerElementNormal/
// LayerElementUV), multi-mesh scenes, materials, the connection graph, and ASCII FBX are documented
// follow-ups. Unknown nodes/property codes are skipped (forward-compatible, not fatal).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "net/session.h"   // hf::net::DigestBytes (the pinned state-digest currency)

namespace hf::asset {

struct FbxMesh {
    std::vector<float>    positions;   // xyz triples (3 floats/vertex), straight from the Vertices double array
    std::vector<uint32_t> indices;     // triangle list (3 per tri), fan-triangulated from PolygonVertexIndex
    uint32_t version = 0;              // the FBX version word (diagnostics; 7400 = the box.fbx fixture)
    bool     ok      = false;          // true iff a Geometry with Vertices + PolygonVertexIndex was decoded
};

namespace fbx_detail {

// ===========================================================================================================
// Clean-room INFLATE (DEFLATE decompressor, RFC 1951) + the zlib stream wrapper (RFC 1950).
// A canonical-Huffman bit reader; no allocation beyond the output vector. Returns false on any malformed
// input (truncation, bad block type, invalid distance). This is the path for FBX arrays with encoding==1.
// ===========================================================================================================

// LSB-first bit reader over a byte span (DEFLATE packs bits LSB-first within each byte).
struct BitReader {
    const uint8_t* p;
    std::size_t    n;
    std::size_t    byte = 0;   // next byte to consume
    uint32_t       bitbuf = 0; // accumulated bits (LSB = next to emit)
    uint32_t       bitcnt = 0; // bits currently in bitbuf
    bool           bad = false;

    BitReader(const uint8_t* data, std::size_t len) : p(data), n(len) {}

    // Pull `count` bits (count <= 24), LSB-first. Sets `bad` and returns 0 on underflow.
    uint32_t bits(uint32_t count) {
        while (bitcnt < count) {
            if (byte >= n) { bad = true; return 0; }
            bitbuf |= (uint32_t)p[byte++] << bitcnt;
            bitcnt += 8;
        }
        uint32_t v = bitbuf & ((1u << count) - 1u);
        bitbuf >>= count;
        bitcnt -= count;
        return v;
    }

    void align() { bitbuf = 0; bitcnt = 0; }   // discard to the next byte boundary (for stored blocks)
};

// A canonical-Huffman decode table built from a list of code lengths (the DEFLATE convention).
struct Huff {
    // count[len] = number of codes of that length; symbol[] = symbols sorted by (len, symbol).
    uint16_t count[16] = {0};
    std::vector<uint16_t> symbol;

    void build(const uint8_t* lengths, int count_n) {
        for (int i = 0; i < 16; ++i) count[i] = 0;
        for (int i = 0; i < count_n; ++i) count[lengths[i]]++;
        count[0] = 0;
        uint16_t offs[16] = {0};
        for (int i = 1; i < 16; ++i) offs[i] = offs[i - 1] + count[i - 1];
        symbol.assign((std::size_t)count_n, 0);
        for (int i = 0; i < count_n; ++i)
            if (lengths[i]) symbol[offs[lengths[i]]++] = (uint16_t)i;
    }

    // Decode one symbol from the bit stream. Returns -1 on a malformed code.
    int decode(BitReader& br) const {
        int code = 0, first = 0, index = 0;
        for (int len = 1; len <= 15; ++len) {
            code |= (int)br.bits(1);
            if (br.bad) return -1;
            int cnt = count[len];
            if (code - first < cnt) return symbol[(std::size_t)(index + (code - first))];
            index += cnt;
            first += cnt;
            first <<= 1;
            code <<= 1;
        }
        return -1;
    }
};

// The fixed (block type 1) literal/length and distance tables, plus the length/distance extra-bit tables.
inline const Huff& FixedLit() {
    static Huff h = [] {
        uint8_t lens[288];
        for (int i = 0; i < 144; ++i) lens[i] = 8;
        for (int i = 144; i < 256; ++i) lens[i] = 9;
        for (int i = 256; i < 280; ++i) lens[i] = 7;
        for (int i = 280; i < 288; ++i) lens[i] = 8;
        Huff t; t.build(lens, 288); return t;
    }();
    return h;
}
inline const Huff& FixedDist() {
    static Huff h = [] {
        uint8_t lens[30];
        for (int i = 0; i < 30; ++i) lens[i] = 5;
        Huff t; t.build(lens, 30); return t;
    }();
    return h;
}

// Inflate a single DEFLATE stream (no zlib header) into `out`. Returns false on malformed input.
inline bool InflateRaw(const uint8_t* src, std::size_t srcLen, std::vector<uint8_t>& out) {
    static const uint16_t lenBase[29] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
    static const uint8_t  lenExtra[29] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
    static const uint16_t distBase[30] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
    static const uint8_t  distExtra[30] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
    static const uint8_t  clOrder[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};

    BitReader br(src, srcLen);
    bool finalBlock = false;
    while (!finalBlock) {
        finalBlock = br.bits(1) != 0;
        uint32_t type = br.bits(2);
        if (br.bad) return false;

        if (type == 0) {                          // stored (uncompressed)
            br.align();
            if (br.byte + 4 > br.n) return false;
            uint32_t len = (uint32_t)src[br.byte] | ((uint32_t)src[br.byte + 1] << 8);
            br.byte += 4;                         // skip LEN + NLEN
            if (br.byte + len > br.n) return false;
            out.insert(out.end(), src + br.byte, src + br.byte + len);
            br.byte += len;
            continue;
        }
        if (type == 3) return false;              // reserved/invalid

        Huff dynLit, dynDist;
        const Huff* lit;
        const Huff* dist;
        if (type == 1) {                          // fixed Huffman
            lit = &FixedLit(); dist = &FixedDist();
        } else {                                  // type == 2: dynamic Huffman
            uint32_t hlit  = br.bits(5) + 257;
            uint32_t hdist = br.bits(5) + 1;
            uint32_t hclen = br.bits(4) + 4;
            if (br.bad || hlit > 288 || hdist > 30) return false;
            uint8_t clLens[19] = {0};
            for (uint32_t i = 0; i < hclen; ++i) clLens[clOrder[i]] = (uint8_t)br.bits(3);
            if (br.bad) return false;
            Huff clHuff; clHuff.build(clLens, 19);
            uint8_t lengths[288 + 30] = {0};
            uint32_t i = 0;
            while (i < hlit + hdist) {
                int sym = clHuff.decode(br);
                if (sym < 0) return false;
                if (sym < 16) { lengths[i++] = (uint8_t)sym; }
                else if (sym == 16) {
                    if (i == 0) return false;
                    uint32_t rep = br.bits(2) + 3; uint8_t prev = lengths[i - 1];
                    while (rep-- && i < hlit + hdist) lengths[i++] = prev;
                } else if (sym == 17) {
                    uint32_t rep = br.bits(3) + 3;  while (rep-- && i < hlit + hdist) lengths[i++] = 0;
                } else {  // sym == 18
                    uint32_t rep = br.bits(7) + 11; while (rep-- && i < hlit + hdist) lengths[i++] = 0;
                }
                if (br.bad) return false;
            }
            dynLit.build(lengths, (int)hlit);
            dynDist.build(lengths + hlit, (int)hdist);
            lit = &dynLit; dist = &dynDist;
        }

        for (;;) {                                // decode symbols until end-of-block (256)
            int sym = lit->decode(br);
            if (sym < 0) return false;
            if (sym == 256) break;
            if (sym < 256) { out.push_back((uint8_t)sym); continue; }
            sym -= 257;
            if (sym >= 29) return false;
            uint32_t length = lenBase[sym] + br.bits(lenExtra[sym]);
            int dsym = dist->decode(br);
            if (dsym < 0 || dsym >= 30) return false;
            uint32_t distance = distBase[dsym] + br.bits(distExtra[dsym]);
            if (br.bad || distance == 0 || distance > out.size()) return false;
            std::size_t from = out.size() - distance;
            for (uint32_t k = 0; k < length; ++k) out.push_back(out[from + k]);   // LZ77 back-copy (may overlap)
        }
    }
    return true;
}

// Inflate a zlib stream (RFC 1950): a 2-byte header (CMF/FLG, CM must be 8=deflate) then a DEFLATE body
// (the trailing 4-byte Adler-32 is not validated — FBX integrity is covered by our own pinned digest).
inline bool InflateZlib(const uint8_t* src, std::size_t srcLen, std::vector<uint8_t>& out) {
    if (srcLen < 2) return false;
    uint8_t cmf = src[0], flg = src[1];
    if ((cmf & 0x0F) != 8) return false;                  // CM must be 8 (deflate)
    if (((uint32_t)cmf * 256 + flg) % 31 != 0) return false;  // header checksum
    std::size_t off = 2;
    if (flg & 0x20) off += 4;                             // FDICT present -> skip the 4-byte dictionary id
    if (off > srcLen) return false;
    return InflateRaw(src + off, srcLen - off, out);
}

// ===========================================================================================================
// The binary FBX byte reader + node-tree walker.
// ===========================================================================================================

struct Reader {
    const uint8_t* p;
    std::size_t    n;
    std::size_t    pos = 0;
    bool           bad = false;

    Reader(const uint8_t* data, std::size_t len) : p(data), n(len) {}

    bool need(std::size_t k) const { return pos + k <= n; }

    uint8_t  u8()  { if (!need(1)) { bad = true; return 0; } return p[pos++]; }
    uint16_t u16() { if (!need(2)) { bad = true; return 0; } uint16_t v; std::memcpy(&v, p + pos, 2); pos += 2; return v; }
    uint32_t u32() { if (!need(4)) { bad = true; return 0; } uint32_t v; std::memcpy(&v, p + pos, 4); pos += 4; return v; }
    uint64_t u64() { if (!need(8)) { bad = true; return 0; } uint64_t v; std::memcpy(&v, p + pos, 8); pos += 8; return v; }
};

// Decode a typed-array property body into the int32 (i) or double (d) values we care about; on encoding==1
// the payload is zlib-inflated first. `code` is one of f/d/l/i/b. Only the i and d arrays are surfaced (the
// two Geometry arrays); the rest are still consumed so the cursor stays aligned. Returns false on malformed.
inline bool ReadArrayProp(Reader& r, char code, std::vector<int32_t>& outI, std::vector<double>& outD) {
    uint32_t arrayLen = r.u32();
    uint32_t encoding = r.u32();
    uint32_t compLen  = r.u32();
    if (r.bad || !r.need(compLen)) { r.bad = true; return false; }

    // element width by array code
    std::size_t elemSize = 0;
    switch (code) {
        case 'f': case 'i': elemSize = 4; break;   // float32 / int32
        case 'd': case 'l': elemSize = 8; break;   // double  / int64
        case 'b':           elemSize = 1; break;   // bool (1 byte)
        default: return false;
    }

    const uint8_t* data = r.p + r.pos;
    std::vector<uint8_t> inflated;
    if (encoding == 1) {
        if (!InflateZlib(data, compLen, inflated)) { r.pos += compLen; return false; }
        data = inflated.data();
        if (inflated.size() != (std::size_t)arrayLen * elemSize) { r.pos += compLen; return false; }
    } else {
        if (compLen != (uint32_t)((std::size_t)arrayLen * elemSize)) { r.pos += compLen; return false; }
    }
    r.pos += compLen;   // advance past the on-disk payload regardless of encoding

    if (code == 'i') {
        outI.resize(arrayLen);
        for (uint32_t k = 0; k < arrayLen; ++k) { int32_t v; std::memcpy(&v, data + (std::size_t)k * 4, 4); outI[k] = v; }
    } else if (code == 'd') {
        outD.resize(arrayLen);
        for (uint32_t k = 0; k < arrayLen; ++k) { double v; std::memcpy(&v, data + (std::size_t)k * 8, 8); outD[k] = v; }
    }
    return true;
}

// Skip a single scalar/string/raw property body (we only read the i/d arrays inside the Geometry node).
inline void SkipScalarProp(Reader& r, char code) {
    switch (code) {
        case 'Y': r.pos += 2; break;                       // int16
        case 'C': r.pos += 1; break;                       // bool
        case 'I': r.pos += 4; break;                       // int32
        case 'F': r.pos += 4; break;                       // float32
        case 'D': r.pos += 8; break;                       // float64
        case 'L': r.pos += 8; break;                       // int64
        case 'R': { uint32_t len = r.u32(); r.pos += len; break; }   // raw bytes
        case 'S': { uint32_t len = r.u32(); r.pos += len; break; }   // string
        default:  r.bad = true; break;                     // unknown code -> can't keep alignment
    }
    if (r.pos > r.n) r.bad = true;
}

// One walk of the node tree. When we enter a `Geometry` node we capture its Vertices (d) + PolygonVertexIndex
// (i) child arrays into `mesh` and set `found`. Records use 32-bit fields for version < 7500 else 64-bit.
// Returns false on a structural error; `pos` is advanced to the sibling end on success.
inline bool WalkNode(Reader& r, bool wide, FbxMesh& mesh, bool& found,
                     std::vector<int32_t>& geomIdx, std::vector<double>& geomVtx, bool inGeometry) {
    std::size_t recStart = r.pos;
    uint64_t endOffset    = wide ? r.u64() : r.u32();
    uint64_t numProps     = wide ? r.u64() : r.u32();
    uint64_t /*propLen*/  __propLen = wide ? r.u64() : r.u32();
    (void)__propLen;
    uint8_t  nameLen      = r.u8();
    if (r.bad) return false;

    // The null record (all-zero header) terminates a sibling list.
    if (endOffset == 0 && numProps == 0 && nameLen == 0) return true;

    if (!r.need(nameLen)) { r.bad = true; return false; }
    std::string name((const char*)(r.p + r.pos), nameLen);
    r.pos += nameLen;

    const bool isGeometry = (name == "Geometry");
    const bool wantArrays = inGeometry;   // collect arrays only when our parent is the (first) Geometry
    std::vector<int32_t> localI;
    std::vector<double>  localD;
    char arrayCode = 0;

    // --- property list ---
    for (uint64_t i = 0; i < numProps; ++i) {
        char code = (char)r.u8();
        if (r.bad) return false;
        if (code == 'f' || code == 'd' || code == 'l' || code == 'i' || code == 'b') {
            std::vector<int32_t> tmpI; std::vector<double> tmpD;
            if (!ReadArrayProp(r, code, tmpI, tmpD)) { if (r.bad) return false; }
            else { arrayCode = code; if (!tmpI.empty()) localI.swap(tmpI); if (!tmpD.empty()) localD.swap(tmpD); }
        } else {
            SkipScalarProp(r, code);
            if (r.bad) return false;
        }
    }

    // Surface the two Geometry arrays to the parent walk.
    if (wantArrays) {
        if (name == "Vertices" && arrayCode == 'd' && geomVtx.empty()) geomVtx = std::move(localD);
        else if (name == "PolygonVertexIndex" && arrayCode == 'i' && geomIdx.empty()) geomIdx = std::move(localI);
    }

    // --- nested children (present iff the cursor hasn't reached endOffset) ---
    // For a Geometry node, recurse with inGeometry=true so its Vertices/PolygonVertexIndex are captured.
    const bool childInGeom = isGeometry && !found;   // only the FIRST Geometry
    std::vector<int32_t> myIdx; std::vector<double> myVtx;
    if (r.pos < endOffset) {
        while (r.pos < endOffset) {
            // A null record (13 or 25 bytes of zero) closes the child list.
            std::size_t childStart = r.pos;
            if (!WalkNode(r, wide, mesh, found, childInGeom ? myIdx : geomIdx,
                          childInGeom ? myVtx : geomVtx, childInGeom || inGeometry)) {
                if (r.bad) return false;
            }
            if (r.pos <= childStart) { r.bad = true; return false; }   // no forward progress -> bail
            // Detect the terminating null record: WalkNode returns with pos advanced by the null size.
            if (childStart + (wide ? 25 : 13) == r.pos) {
                // Could be a real 0-length node OR the terminator; the terminator is all-zero. Peek back.
                bool allZero = true;
                for (std::size_t b = childStart; b < r.pos; ++b) if (r.p[b] != 0) { allZero = false; break; }
                if (allZero) break;
            }
        }
    }

    if (childInGeom) {
        if (!myVtx.empty() && !myIdx.empty()) {
            geomVtx = std::move(myVtx);
            geomIdx = std::move(myIdx);
            found = true;
        }
    }

    // Jump to the record end (skips any unparsed trailing bytes / child padding deterministically).
    if (endOffset >= recStart && endOffset <= r.n) r.pos = (std::size_t)endOffset;
    else { r.bad = true; return false; }
    return true;
}

}  // namespace fbx_detail

// Parse a binary FBX byte buffer -> FbxMesh (first Geometry's positions + triangulated indices).
// Deterministic, pure CPU. ok=false on a bad magic, an unsupported/missing Geometry, or a structural error.
inline FbxMesh ParseFbx(const uint8_t* bytes, std::size_t n) {
    using namespace fbx_detail;
    FbxMesh out;

    static const uint8_t kMagic[23] = {
        'K','a','y','d','a','r','a',' ','F','B','X',' ','B','i','n','a','r','y',' ',' ',0x00,0x1A,0x00
    };
    if (n < 27) return out;
    if (std::memcmp(bytes, kMagic, 23) != 0) return out;

    uint32_t version;
    std::memcpy(&version, bytes + 23, 4);
    out.version = version;
    const bool wide = (version >= 7500);   // 64-bit record header fields

    Reader r(bytes, n);
    r.pos = 27;

    std::vector<int32_t> geomIdx;
    std::vector<double>  geomVtx;
    bool found = false;

    // Walk top-level sibling records until the buffer's footer/null region.
    while (r.pos + (wide ? 25 : 13) <= n && !found) {
        std::size_t before = r.pos;
        if (!WalkNode(r, wide, out, found, geomIdx, geomVtx, /*inGeometry*/ false)) break;
        if (r.pos <= before) break;
        // top-level null record terminates the list
        bool allZero = true;
        for (std::size_t b = before; b < before + (wide ? 25 : 13) && b < n; ++b)
            if (bytes[b] != 0) { allZero = false; break; }
        if (allZero && (r.pos - before) == (std::size_t)(wide ? 25 : 13)) break;
    }

    if (!found || geomVtx.empty() || geomIdx.empty()) return out;

    // positions: the double xyz triples -> float
    out.positions.resize(geomVtx.size());
    for (std::size_t i = 0; i < geomVtx.size(); ++i) out.positions[i] = (float)geomVtx[i];

    // triangulate: each polygon's last corner is encoded as ~idx (negative); fan-triangulate.
    std::vector<uint32_t> poly;
    const uint32_t vertCount = (uint32_t)(geomVtx.size() / 3);
    for (int32_t raw : geomIdx) {
        bool last = raw < 0;
        uint32_t idx = last ? (uint32_t)(~raw) : (uint32_t)raw;
        if (idx >= vertCount) { out.positions.clear(); out.indices.clear(); return out; }  // corrupt index
        poly.push_back(idx);
        if (last) {
            for (std::size_t k = 1; k + 1 < poly.size(); ++k) {
                out.indices.push_back(poly[0]);
                out.indices.push_back(poly[k]);
                out.indices.push_back(poly[k + 1]);
            }
            poly.clear();
        }
    }

    out.ok = !out.indices.empty();
    if (!out.ok) { out.positions.clear(); out.indices.clear(); }
    return out;
}

inline FbxMesh ParseFbx(const std::vector<uint8_t>& bytes) { return ParseFbx(bytes.data(), bytes.size()); }

}  // namespace hf::asset
