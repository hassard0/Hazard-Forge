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
namespace net = hf::net;   // S5: the build is a net::Session lockstep replay (InputRing/RunLockstep/DigestTrace)

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

    // =====================================================================================================
    // Slice ASSET-S2 — the deterministic compiled-artifact format. The OBJ fixture -> a canonical Q16.16
    // blob whose pinned DigestBytes is bit-identical MSVC / Windows-clang / Mac-clang.
    // =====================================================================================================

    // The compiled artifact over the SAME fixture + showcase params (scale = 2.0 in Q16.16).
    const std::vector<uint8_t> blob = CompileObj(raw, len, params);
    const CacheKey artifact = DigestArtifact(blob);
    std::printf("asset-s2: compiled-artifact digest = 0x%016llx  (%zu bytes)\n",
                (unsigned long long)artifact, blob.size());

    // (S2-1) S1 INVARIANT re-assertion — the showcase key is STILL the pinned uint64 (S2 is additive).
    check(key == 0x7fb6a48b4b99f1b7ULL,
          "asset-s2: S1 showcase key STILL == 0x7fb6a48b4b99f1b7 (S2 additive, S1 unchanged)");

    // (S2-2) HEADER — the blob decodes to magic == kCompiledMeshMagic, version == 1.
    {
        CompiledMesh out;
        const bool ok = DecodeCompiledMesh(blob, out);
        check(ok && out.magic == kCompiledMeshMagic && out.version == kCompiledMeshVersion,
              "asset-s2: CompileObj(fixture, showcase) magic/version == kCompiledMeshMagic / 1");
    }

    // (S2-3) PINNED ARTIFACT DIGEST — the cross-compiler anchor (identical MSVC + clang).
    const CacheKey kPinnedArtifact = 0xf7ee13c169dc0464ULL;
    check(artifact == kPinnedArtifact,
          "asset-s2: the compiled-artifact digest == pinned uint64 (Q16.16 blob, byte-stable cross-platform)");

    // (S2-4) ROUND-TRIP — DecodeCompiledMesh recovers vertexCount/indexCount/verts/indices; re-encode matches.
    {
        CompiledMesh out;
        const bool ok = DecodeCompiledMesh(blob, out);
        bool recoded = ok && out.vertexCount == 3 && out.indexCount == 3 &&
                       out.verts.size() == 24 && out.indices.size() == 3;
        // re-encode the recovered params + raw fixture must reproduce the SAME blob (canonical).
        if (recoded) {
            const std::vector<uint8_t> blob2 = CompileObj(raw, len, out.params);
            recoded = (blob2 == blob);
        }
        check(recoded,
              "asset-s2: DecodeCompiledMesh(blob) round-trips — vertexCount/indexCount/verts/indices recovered");
    }

    // (S2-5) PARAM LOAD-BEARING — change `scale` -> a DIFFERENT artifact digest.
    {
        CompileParams p2 = params;
        p2.scale = 3 * 65536;   // 3.0 instead of 2.0
        const CacheKey a2 = DigestArtifact(CompileObj(raw, len, p2));
        check(a2 != artifact, "asset-s2: a changed compile param (scale) changes the artifact digest (params are load-bearing)");
    }

    // (S2-6) CONTENT LOAD-BEARING — a different OBJ text -> a DIFFERENT artifact digest.
    {
        const char* raw2 = "v 0 0 0\nv 1 0 0\nv 0 2 0\nf 1 2 3\n";   // moved one vertex
        std::size_t len2 = 0; while (raw2[len2] != '\0') ++len2;
        const CacheKey a2 = DigestArtifact(CompileObj(raw2, len2, params));
        check(a2 != artifact, "asset-s2: different raw bytes -> a different artifact digest (content is load-bearing)");
    }

    // (S2-7) PURE-INTEGER EXACTNESS — the fixture vertex (1,0,0) with scale=2.0 decodes to EXACTLY 131072.
    // pos.x = FxMul(FxQuantize(1.0), 2*65536) = FxMul(65536, 131072) = (65536*131072)>>16 = 131072 (= 2.0).
    {
        CompiledMesh out;
        const bool ok = DecodeCompiledMesh(blob, out);
        // The fixture parses (0,0,0),(1,0,0),(0,1,0) in order; vertex 1 is (1,0,0) -> verts[8] is its pos.x.
        bool exact = ok && out.verts.size() >= 9 && out.verts[8] == 131072;
        check(exact,
              "asset-s2: the artifact is pure-integer Q16.16 — the showcase positions decode to exact integer values");
    }

    // =====================================================================================================
    // Slice ASSET-S3 — the content-addressed cache. A key->artifact store with miss/hit GetOrCompile, hand-LE
    // serialize round-trip, and a store digest INDEPENDENT of insertion order (the content-address property).
    // =====================================================================================================

    const AssetCache showcaseCache = MakeShowcaseCache();
    const CacheKey cacheDigest = DigestCache(showcaseCache);
    std::printf("asset-s3: cache digest = 0x%016llx  (%zu entries)\n",
                (unsigned long long)cacheDigest, showcaseCache.entries.size());

    // (S3-1) PRIOR INVARIANT — S1 key + S2 artifact digest STILL pinned (S3 is additive).
    check(key == 0x7fb6a48b4b99f1b7ULL && artifact == 0xf7ee13c169dc0464ULL,
          "asset-s3: S1 key 0x7fb6a48b4b99f1b7 + S2 artifact 0xf7ee13c169dc0464 UNCHANGED (S3 additive)");

    // (S3-2) PINNED CACHE DIGEST — the cross-platform anchor (identical MSVC + clang).
    const CacheKey kPinnedCache = 0x029174f13e64c9f1ULL;
    check(cacheDigest == kPinnedCache,
          "asset-s3: DigestCache(MakeShowcaseCache()) == pinned uint64 (content-addressed store, byte-stable)");

    // (S3-3) MISS THEN HIT — a fresh cache; first GetOrCompile is a MISS, the second of the same asset a HIT.
    {
        AssetCache c;
        const CompileResult r1 = GetOrCompile(c, raw, len, params);
        const CompileResult r2 = GetOrCompile(c, raw, len, params);
        check(r1.wasHit == false && r2.wasHit == true,
              "asset-s3: cold GetOrCompile is a MISS, a second GetOrCompile of the same asset is a HIT");
    }

    // (S3-4) NO WASHIT LEAK — the HIT blob is byte-identical to a cold CompileObj of the same inputs.
    {
        AssetCache c;
        GetOrCompile(c, raw, len, params);                       // prime
        const CompileResult hit = GetOrCompile(c, raw, len, params);
        const std::vector<uint8_t> cold = CompileObj(raw, len, params);
        check(hit.wasHit == true && hit.blob == cold,
              "asset-s3: the HIT blob is byte-identical to a cold CompileObj of the same inputs (no wasHit leak)");
    }

    // (S3-5) ROUND-TRIP — DeserializeCache(SerializeCache(c)) succeeds, same DigestCache + same entries.
    {
        AssetCache out;
        const bool ok = DeserializeCache(SerializeCache(showcaseCache), out);
        bool same = ok && DigestCache(out) == cacheDigest &&
                    out.entries.size() == showcaseCache.entries.size();
        if (same) {
            for (std::size_t i = 0; i < out.entries.size(); ++i) {
                if (out.entries[i].key != showcaseCache.entries[i].key ||
                    out.entries[i].blob != showcaseCache.entries[i].blob) { same = false; break; }
            }
        }
        check(same, "asset-s3: DeserializeCache(SerializeCache(c)) round-trips — same DigestCache");
    }

    // (S3-6) ORDER-INDEPENDENT — insert A,B,C vs C,B,A -> the SAME digest (sorted-by-key store).
    {
        AssetCache x;
        GetOrCompile(x, ShowcaseRawBytes(),  ShowcaseRawLen(),  params);
        GetOrCompile(x, ShowcaseRawBytesB(), ShowcaseRawLenB(), params);
        GetOrCompile(x, ShowcaseRawBytesC(), ShowcaseRawLenC(), params);
        AssetCache y;
        GetOrCompile(y, ShowcaseRawBytesC(), ShowcaseRawLenC(), params);
        GetOrCompile(y, ShowcaseRawBytesB(), ShowcaseRawLenB(), params);
        GetOrCompile(y, ShowcaseRawBytes(),  ShowcaseRawLen(),  params);
        check(DigestCache(x) == DigestCache(y),
              "asset-s3: the cache digest is INDEPENDENT of insertion order (insert A,B,C vs C,B,A -> same digest)");
    }

    // (S3-7) CLEAN MISS — Lookup of an absent key returns nullptr.
    {
        // A key that is not any of the three fixtures' (flip a bit on the showcase key).
        const CacheKey absent = key ^ 0xDEADBEEFULL;
        check(Lookup(showcaseCache, absent) == nullptr,
              "asset-s3: Lookup of an absent key returns nullptr (clean miss)");
    }

    // =====================================================================================================
    // Slice ASSET-S4 — dependency graph + deterministic incremental rebuild. A fixed 6-node graph; when one
    // input changes, recompile ONLY that node + its transitive dependents, in a deterministic topo order.
    // =====================================================================================================

    const DepGraph graph = MakeShowcaseGraph();

    // The rebuild plan for a single mesh-0 change (the headline: incremental, byte-stable order).
    const RebuildResult rb0 = Rebuild(graph, std::vector<NodeId>{ 0 });
    std::printf("asset-s4: rebuild(change mesh 0) digest = 0x%016llx  recompiled = [",
                (unsigned long long)rb0.digest);
    for (std::size_t i = 0; i < rb0.recompiled.size(); ++i)
        std::printf("%s%u", i ? ", " : "", rb0.recompiled[i]);
    std::printf("]\n");

    // (S4-1) PRIOR INVARIANT — S1 key + S2 artifact + S3 cache digests STILL pinned (S4 is additive).
    check(key == 0x7fb6a48b4b99f1b7ULL && artifact == 0xf7ee13c169dc0464ULL &&
          cacheDigest == 0x029174f13e64c9f1ULL,
          "asset-s4: S1 0x7fb6a48b4b99f1b7 + S2 0xf7ee13c169dc0464 + S3 0x029174f13e64c9f1 UNCHANGED (S4 additive)");

    // (S4-2) INVALIDATION (mesh 0) — InvalidationSet(g, 0) == {0, 2} exactly (mesh + the scene depending on it).
    {
        const std::vector<NodeId> inv = InvalidationSet(graph, 0);
        const bool ok = (inv == std::vector<NodeId>{ 0, 2 });
        check(ok, "asset-s4: InvalidationSet(change mesh 0) == {0, 2} exactly (mesh + the scene depending on it)");
    }

    // (S4-3) INVALIDATION (shared mesh 1) — InvalidationSet(g, 1) == {1, 2, 5} exactly (hits both scenes).
    {
        const std::vector<NodeId> inv = InvalidationSet(graph, 1);
        const bool ok = (inv == std::vector<NodeId>{ 1, 2, 5 });
        check(ok, "asset-s4: InvalidationSet(change mesh 1) == {1, 2, 5} exactly (shared mesh hits both scenes)");
    }

    // (S4-4) INCREMENTALITY — recompiling for one mesh change touches 2 nodes, NOT all 6.
    check(rb0.recompiled.size() == 2,
          "asset-s4: incrementality — recompiling for one mesh change touches 2 nodes, NOT all 6");

    // (S4-5) TOPOLOGICAL — in RebuildOrder(g, InvalidationSet(g, 1)) every node appears AFTER its in-subset deps.
    {
        const std::vector<NodeId> inv = InvalidationSet(graph, 1);
        const OrderResult ord = RebuildOrder(graph, inv);
        bool topo = ord.ok;
        // for each emitted node, every in-subset dependency must appear EARLIER in the order.
        for (std::size_t i = 0; topo && i < ord.order.size(); ++i) {
            const std::vector<NodeId>* deps = Dependencies(graph, ord.order[i]);
            if (!deps) continue;
            for (std::size_t d = 0; d < deps->size(); ++d) {
                const NodeId dep = (*deps)[d];
                // is `dep` in the subset? if so it must appear before index i.
                bool inSubset = false, before = false;
                for (std::size_t k = 0; k < inv.size(); ++k) if (inv[k] == dep) { inSubset = true; break; }
                if (!inSubset) continue;
                for (std::size_t j = 0; j < i; ++j) if (ord.order[j] == dep) { before = true; break; }
                if (!before) { topo = false; break; }
            }
        }
        check(topo,
              "asset-s4: RebuildOrder is topological — every dependency precedes its dependent (and is deterministic)");
    }

    // (S4-6) PINNED REBUILD DIGEST — Rebuild(g, {0}).digest == the hard-pinned uint64 (byte-stable plan).
    const CacheKey kPinnedRebuild = 0x0808b56e0322d8c1ULL;
    check(rb0.digest == kPinnedRebuild,
          "asset-s4: Rebuild(change mesh 0) digest == pinned uint64 (the recompile plan is byte-stable)");

    // (S4-7) CYCLE — a hand-built graph with a cycle -> ok == false (a deterministic error, no hang).
    {
        DepGraph cyc;
        AddDep(cyc, 10, 11);   // 10 depends on 11
        AddDep(cyc, 11, 10);   // 11 depends on 10 -> a cycle
        const OrderResult ord = RebuildOrder(cyc, std::vector<NodeId>{ 10, 11 });
        const RebuildResult rr = Rebuild(cyc, std::vector<NodeId>{ 10 });
        check(ord.ok == false && rr.ok == false,
              "asset-s4: a cyclic graph -> ok == false (deterministic error, no hang)");
    }

    // (S4-8) LEAF TEXTURE — InvalidationSet(g, 3) == {3, 4, 5}; an unrelated mesh (0) is NOT in the set.
    {
        const std::vector<NodeId> inv = InvalidationSet(graph, 3);
        bool ok = (inv == std::vector<NodeId>{ 3, 4, 5 });
        bool meshAbsent = true;
        for (std::size_t i = 0; i < inv.size(); ++i) if (inv[i] == 0) { meshAbsent = false; break; }
        check(ok && meshAbsent,
              "asset-s4: changing a leaf texture (3) invalidates {3,4,5}, an unrelated mesh (0) stays a cache hit");
    }

    // =====================================================================================================
    // Slice ASSET-S5 — Manifest / batch compile, the build AS a net::Session lockstep replay. CompileSet
    // produces a deterministic manifest (node -> artifact digest, sorted-unique); StepBuild is the
    // net::Session StepFn so RunLockstep / DigestTrace replay the build to the SAME manifest + per-tick trace.
    // =====================================================================================================

    // The plain batch manifest over the three showcase jobs (a fresh cache).
    AssetCache batchCache;
    const Manifest batchManifest = CompileSet(MakeShowcaseJobs(), batchCache);
    const CacheKey manifestDigest = ManifestDigest(batchManifest);
    std::printf("asset-s5: manifest digest = 0x%016llx  (%zu entries)\n",
                (unsigned long long)manifestDigest, batchManifest.entries.size());

    // The build AS a net::Session lockstep replay: an InputRing<CompileJob> with job i on tick i, a step
    // lambda capturing a cache, and digest = ManifestDigest. RunLockstep drives the build to the final
    // manifest; DigestTrace records the per-tick manifest digest (the manifest grows one entry per tick).
    net::InputRing<CompileJob> ring;
    {
        const std::vector<CompileJob> jobs = MakeShowcaseJobs();
        for (uint32_t i = 0; i < (uint32_t)jobs.size(); ++i) ring.AddInput(i, jobs[i]);  // job i on tick i
    }
    const uint32_t kTicks = 3;
    AssetCache lockstepCache;
    auto step = [&lockstepCache](Manifest& w, const std::vector<CompileJob>& js, uint32_t tick) {
        StepBuild(lockstepCache, w, js, tick);
    };
    auto digest = [](const Manifest& m) { return ManifestDigest(m); };

    const uint64_t lockstepFinal = net::RunLockstep(Manifest{}, ring, kTicks, step, digest);

    const std::vector<uint64_t> trace = net::DigestTrace(Manifest{}, ring, kTicks, step, digest);
    std::vector<uint8_t> traceBytes;
    for (std::size_t i = 0; i < trace.size(); ++i) PutU64(traceBytes, trace[i]);
    const uint64_t traceDigest = net::DigestBytes(traceBytes.data(), traceBytes.size());
    std::printf("asset-s5: build trace digest = 0x%016llx  (%zu ticks)\n",
                (unsigned long long)traceDigest, trace.size());

    // (S5-1) PRIOR INVARIANT — S1 key + S2 artifact + S3 cache + S4 rebuild digests STILL pinned (S5 additive).
    check(key == 0x7fb6a48b4b99f1b7ULL && artifact == 0xf7ee13c169dc0464ULL &&
          cacheDigest == 0x029174f13e64c9f1ULL && rb0.digest == 0x0808b56e0322d8c1ULL,
          "asset-s5: S1 0x7fb6a48b4b99f1b7 + S2 0xf7ee13c169dc0464 + S3 0x029174f13e64c9f1 + S4 0x0808b56e0322d8c1 UNCHANGED (S5 additive)");

    // (S5-2) PINNED MANIFEST — ManifestDigest(CompileSet(showcase)) == the hard-pinned uint64 (build output byte-stable).
    const CacheKey kPinnedManifest = 0x37ca56d9da205682ULL;
    check(manifestDigest == kPinnedManifest,
          "asset-s5: ManifestDigest(CompileSet(showcase)) == pinned uint64 (the build output is byte-stable)");

    // (S5-3) BUILD == LOCKSTEP — RunLockstep final digest == the CompileSet ManifestDigest (same manifest as the batch).
    check(lockstepFinal == manifestDigest,
          "asset-s5: the build is a lockstep replay — RunLockstep final digest == the CompileSet ManifestDigest");

    // (S5-4) PINNED TRACE — DigestBytes of the per-tick DigestTrace vector == the hard-pinned uint64 (length == 3 ticks).
    const CacheKey kPinnedTrace = 0xcfbe58567fb8fa07ULL;
    check(traceDigest == kPinnedTrace && trace.size() == 3,
          "asset-s5: DigestTrace digest == pinned uint64 (the per-tick build trace is replayable, 3 ticks)");

    // (S5-5) ORDER-INDEPENDENT — CompileSet(jobs) and CompileSet(reordered jobs) -> the SAME manifest digest.
    {
        AssetCache c1, c2;
        const CacheKey d1 = ManifestDigest(CompileSet(MakeShowcaseJobs(), c1));
        const CacheKey d2 = ManifestDigest(CompileSet(MakeShowcaseJobsReordered(), c2));
        check(d1 == d2,
              "asset-s5: order-independent — CompileSet(jobs) and CompileSet(reordered) -> the SAME manifest digest");
    }

    // (S5-6) DETERMINISTIC — two RunLockstep calls over the same ring -> the IDENTICAL final digest.
    {
        AssetCache ca, cb;
        auto stepA = [&ca](Manifest& w, const std::vector<CompileJob>& js, uint32_t tick) { StepBuild(ca, w, js, tick); };
        auto stepB = [&cb](Manifest& w, const std::vector<CompileJob>& js, uint32_t tick) { StepBuild(cb, w, js, tick); };
        const uint64_t a = net::RunLockstep(Manifest{}, ring, kTicks, stepA, digest);
        const uint64_t b = net::RunLockstep(Manifest{}, ring, kTicks, stepB, digest);
        check(a == b,
              "asset-s5: two RunLockstep runs over the same ring yield the IDENTICAL final digest (deterministic build)");
    }

    if (g_fail == 0) std::printf("asset_compiler_test: ALL PASS\n");
    else             std::printf("asset_compiler_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
