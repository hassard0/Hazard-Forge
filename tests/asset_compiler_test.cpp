// Unit test for the content-addressed CacheKey (engine/asset/asset_compiler.h, Slice ASSET-S1, issue #16).
//
// Proves the beachhead's headline: a cache key is a pure function of (raw content bytes, integer compile
// params, asset kind) — reproducible, content/params/kind load-bearing, and INDEPENDENT of any pointer,
// clock, or file mtime. The golden is a hard-pinned net::DigestBytes over the key derivation, proven
// identical Windows/MSVC + Mac/clang via a standalone clang compile (NO render-bake, the cheapest proof):
//   clang++ -std=c++20 -I engine -I tests tests/asset_compiler_test.cpp
// Pure hf_core (no device/RHI/float-render), self-contained scaffold (copied from obj_loader_test.cpp).

#include "asset/asset_compiler.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"

using namespace hf::asset;

static int g_fail = 0;
static void check(bool c, const char* what) {
    if (!c) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else    { std::printf("PASS: %s\n", what); }
}

int main() {
    HF_TEST_MAIN_INIT();

    const char* raw = ShowcaseRawBytes();
    const std::size_t len = ShowcaseRawLen();
    const CompileParams params = ShowcaseParams();

    // The showcase key — MakeKey(Mesh, HashRawAsset(fixture), HashParams(showcase)).
    const CacheKey key = MakeKey((uint32_t)AssetKind::Mesh, HashRawAsset(raw, len), HashParams(params));
    std::printf("asset-s1: showcase key = 0x%016llx\n", (unsigned long long)key);

    // (1) PINNED KEY — the make-or-break cross-platform anchor (identical MSVC + clang).
    const CacheKey kPinned = 0x7fb6a48b4b99f1b7ULL;
    check(key == kPinned,
          "asset-s1: MakeKey(Mesh, HashRawAsset(fixture), HashParams(showcase)) == pinned uint64 (cross-platform)");

    // (2) REPRODUCIBLE — recomputing from the same inputs -> identical key (content-addressed).
    const CacheKey key2 = MakeKey((uint32_t)AssetKind::Mesh, HashRawAsset(raw, len), HashParams(params));
    check(key2 == key, "asset-s1: same (bytes, params, kind) -> the SAME key (reproducible / content-addressed)");

    // (3) CONTENT LOAD-BEARING — copy the fixture bytes, flip one byte, re-key -> a DIFFERENT key.
    {
        std::vector<uint8_t> bytes(raw, raw + len);
        bytes[0] ^= 0x01;   // flip one content bit
        const CacheKey kFlip = MakeKey((uint32_t)AssetKind::Mesh, HashRawAsset(bytes.data(), bytes.size()),
                                       HashParams(params));
        check(kFlip != key, "asset-s1: a single flipped content byte -> a DIFFERENT key (content is load-bearing)");
    }

    // (4) PARAMS LOAD-BEARING — clone the showcase params, change one field, re-key -> DIFFERENT.
    {
        CompileParams p2 = params;
        p2.scale += 1;   // smallest possible change to a single field
        const CacheKey kP = MakeKey((uint32_t)AssetKind::Mesh, HashRawAsset(raw, len), HashParams(p2));
        check(kP != key, "asset-s1: a changed compile param -> a DIFFERENT key (params are load-bearing)");
    }

    // (5) POINTER-INDEPENDENT — a SECOND buffer with the SAME bytes at a different address -> the SAME key.
    {
        std::vector<uint8_t> copy(raw, raw + len);
        check(copy.data() != (const uint8_t*)raw, "asset-s1: the copy is at a different address (sanity)");
        const CacheKey kCopy = MakeKey((uint32_t)AssetKind::Mesh, HashRawAsset(copy.data(), copy.size()),
                                       HashParams(params));
        check(kCopy == key,
              "asset-s1: the key is independent of the buffer's address (two copies of the bytes -> the same key)");
    }

    // (6) KIND LOAD-BEARING — same content/params, a different AssetKind -> a DIFFERENT key.
    {
        const CacheKey kTex = MakeKey((uint32_t)AssetKind::Texture, HashRawAsset(raw, len), HashParams(params));
        check(kTex != key, "asset-s1: a different AssetKind -> a DIFFERENT key (kind is load-bearing)");
    }

    if (g_fail == 0) std::printf("asset_compiler_test: ALL PASS\n");
    else             std::printf("asset_compiler_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
