// Slice VR1 — VFX RENDERER VARIETY: ribbon trails + beams + mesh emitters + particle lights
// (engine/vfx/vfx_render.h) over the deterministic particle sim (PT1-PT6) + the PA1 authored fountain.
// Pure CPU, links hf_core, ASan-eligible. Existing particles_test / particle_author_test /
// manylight_test are UNTOUCHED — this file only ADDS coverage.
//
// PINNED VALUES: captured from the MSVC x64 release build and verified IDENTICAL under local
// clang-on-Windows. Everything except kPinnedAssignDigest is PURE INTEGER (byte-stable on every
// platform); kPinnedAssignDigest is the ML1 clustered assignment (float pow/log slice math) and is
// pinned PER-TOOLCHAIN-FAMILY (the documented ML1 honesty convention — the Metal stat line prints the
// Mac's value so a last-ULP libm divergence surfaces explicitly).
#include "vfx/vfx_render.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace vr = hf::vfx::render;
namespace pt = hf::sim::particles;
namespace cl = hf::render::clustered;
using vr::fx;
using vr::FxVec3;
using vr::kOne;
using vr::kFrac;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// ---- THE PINS (captured MSVC x64 release; verified identical under local clang) ----
static const uint64_t kPinnedTrailsDigest   = 0xe0b10b97f2411e3full;  // (a) 10-tick trail history
static const uint64_t kPinnedRibbonDigest   = 0x3736e7ad815458d1ull;  // (a) the ribbon strip
static const uint64_t kPinnedKinkDigest     = 0xe40aaf8b23d9d886ull;  // (a) the direction-reversal worst case
static const uint64_t kPinnedBeamDigest     = 0xf22f57ebc0896457ull;  // (b) the jittered showcase beam
static const uint64_t kPinnedInstDigest     = 0xf00e9a6e4df05fd9ull;  // (c) the mesh-emitter instances
static const uint64_t kPinnedLightDigest    = 0x3bf12b498330f980ull;  // (d) the brightest-N light list
static const uint64_t kPinnedShotDigest     = 0x4c5b2968f2263079ull;  // (e) the combined showcase digest
static const uint32_t kPinnedShotAlive      = 170u;                   // (e) alive particles at tick 96
static const uint32_t kPinnedShotTrails     = 158u;                   // (e) drawable trails (>=2 samples —
                                                                      //     the final-tick spawns have 1)
static const uint32_t kPinnedShotRibbonTris = 2148u;                  // (e) ribbon triangle count
static const uint32_t kPinnedShotInst       = 170u;                   // (e) mesh instances == alive
static const uint32_t kPinnedShotLights     = 64u;                    // (e) min(64, alive)
static const uint64_t kPinnedAssignDigest   = 0x6379470253967084ull;  // (d/e) ML1 seam — TOOLCHAIN-FAMILY pin
static const uint32_t kPinnedAssignMax      = 64u;                    // (d/e) densest cluster (the plume core
                                                                      //     overlaps ALL 64 light spheres)
static const uint64_t kPinnedAssignTotal    = 1494ull;                // (d/e) total assignments

// The small deterministic PT1 trail fixture: a 32-slot fountain stepped N ticks with trail recording.
static void RunTrailFixture(pt::ParticlePool& pool, vr::TrailHistory& th, int ticks) {
    pool = pt::InitParticlePool(32);
    th = vr::InitTrailHistory(32, 6);
    pt::EmitterConfig cfg;
    cfg.origin = FxVec3{0, 3 * kOne, 0};
    cfg.ratePerTick = 2;
    cfg.lifetime = kOne / 2;          // 0.5 s = ~30 ticks; nothing dies inside the 10-tick fixture
    cfg.speed = 2 * kOne;
    cfg.emitterId = 7u;
    const FxVec3 g{0, (fx)(-9.8 * (double)kOne - 0.5), 0};
    for (int t = 0; t < ticks; ++t) {
        pt::StepEmitIntegrate(pool, cfg, g, kOne / 50, kOne / 60);
        vr::UpdateTrails(pool, th);
    }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= (a) RIBBON TRAILS =================
    {
        pt::ParticlePool pool;
        vr::TrailHistory th;
        RunTrailFixture(pool, th, 10);

        // Two runs are byte-identical (the deterministic-fold property — the netcode replay argument).
        pt::ParticlePool pool2;
        vr::TrailHistory th2;
        RunTrailFixture(pool2, th2, 10);
        check(vr::DigestTrails(th) == vr::DigestTrails(th2), "two trail runs -> identical digest");

        // Ring shape: the oldest tracked slot has a saturated ring (K=6 < 10 ticks alive), the newest 1.
        check(th.count[0] == 6u, "slot 0 (first spawn) ring saturated at K");
        check(pt::CountAlive(pool) == 20u, "fixture: 2/tick x 10 ticks alive");

        // Drawable trails: 20 alive, but the 2 spawned on the FINAL tick have only 1 recorded sample
        // (count < 2 -> not drawable yet) -> exactly 18 ribbons.
        const vr::RibbonSet rib = vr::BuildRibbons(th, kOne / 6);
        check(rib.trailCount == 18u, "every alive particle with >=2 samples has a trail (18 of 20)");
        check(rib.geom.indices.size() % 3u == 0u, "ribbon indices are whole triangles");

        // No degenerate INDEX triples (the SweepStrip winding discipline).
        bool degen = false;
        for (size_t i = 0; i + 2 < rib.geom.indices.size(); i += 3) {
            const uint32_t a = rib.geom.indices[i], b = rib.geom.indices[i + 1],
                           c = rib.geom.indices[i + 2];
            if (a == b || b == c || a == c) degen = true;
            if (a >= rib.geom.positions.size() || b >= rib.geom.positions.size() ||
                c >= rib.geom.positions.size()) degen = true;
        }
        check(!degen, "no degenerate/out-of-range index triples");

        // TAPER MONOTONIC: per trail, the rail-to-rail width is non-decreasing tail -> head (side is
        // unit to +-1 LSB, so allow an 8-LSB integer-sqrt slack), and the head is strictly wider than
        // the tail (the taper is real).
        bool monotonic = true, tapers = true;
        for (uint32_t t = 0; t < rib.trailCount; ++t) {
            const uint32_t base = rib.trailFirstVert[t], m = rib.trailSampleCount[t];
            fx prevW = -1;
            for (uint32_t j = 0; j < m; ++j) {
                const FxVec3 d = pt::FxSub(rib.geom.positions[base + 2 * j + 1],
                                           rib.geom.positions[base + 2 * j]);
                const fx w = pt::FxLength(d);
                if (prevW >= 0 && w + 8 < prevW) monotonic = false;
                prevW = w;
            }
            const FxVec3 d0 = pt::FxSub(rib.geom.positions[base + 1], rib.geom.positions[base]);
            if (prevW <= pt::FxLength(d0)) tapers = false;
        }
        check(monotonic, "taper is monotonic non-decreasing tail -> head");
        check(tapers, "head width strictly exceeds tail width");

        // DEAD TRAIL EXPIRES (pinned behavior): a 3-tick-lifetime particle's trail is GONE the tick
        // after death, and the recycled slot's NEW particle starts a FRESH 1-sample trail.
        {
            pt::ParticlePool p2 = pt::InitParticlePool(4);
            vr::TrailHistory h2 = vr::InitTrailHistory(4, 6);
            pt::EmitterConfig short1;
            short1.origin = FxVec3{0, kOne, 0};
            short1.ratePerTick = 1;
            short1.lifetime = 3 * (kOne / 60);   // dies on the 3rd integrate
            short1.speed = kOne;
            short1.emitterId = 9u;
            pt::EmitterConfig none = short1; none.ratePerTick = 0;
            pt::StepEmitIntegrate(p2, short1, FxVec3{0, -kOne, 0}, 0, kOne / 60);  // spawn + tick 1
            vr::UpdateTrails(p2, h2);
            check(h2.count[0] == 1u, "trail starts recording at spawn");
            pt::StepEmitIntegrate(p2, none, FxVec3{0, -kOne, 0}, 0, kOne / 60);    // tick 2
            vr::UpdateTrails(p2, h2);
            pt::StepEmitIntegrate(p2, none, FxVec3{0, -kOne, 0}, 0, kOne / 60);    // tick 3: age>=lifetime
            vr::UpdateTrails(p2, h2);
            check(pt::CountAlive(p2) == 0u, "the short particle died");
            check(h2.count[0] == 0u && h2.seed[0] == 0u, "dead particle's trail EXPIRED (pinned)");
            check(vr::BuildRibbons(h2, kOne / 6).trailCount == 0u, "no ribbon for the dead slot");
            // Slot reuse: the LIFO free-list hands slot 0 back; the seed guard starts a FRESH trail.
            pt::StepEmitIntegrate(p2, short1, FxVec3{0, -kOne, 0}, 0, kOne / 60);
            vr::UpdateTrails(p2, h2);
            check(h2.count[0] == 1u && h2.seed[0] != 0u, "recycled slot starts a fresh 1-sample trail");
        }

        // THE KINK WORST CASE (honesty, pinned): a trail that doubles back (+X then -X) flips the side
        // vector sign between adjacent samples -> the rails CROSS (a visible kink/twist). Deterministic,
        // digest-pinned; camera-facing ribbons (VR2+) would hide it.
        {
            vr::TrailHistory hk = vr::InitTrailHistory(1, 4);
            hk.seed[0] = 1u; hk.count[0] = 4u; hk.head[0] = 0u;   // ring order == oldest->newest
            hk.ring[0] = FxVec3{0, kOne, 0};
            hk.ring[1] = FxVec3{2 * kOne, kOne, 0};   // moving +X
            hk.ring[2] = FxVec3{kOne, kOne, 0};       // REVERSAL: moving -X
            hk.ring[3] = FxVec3{0, kOne, 0};
            const vr::RibbonSet kink = vr::BuildRibbons(hk, kOne / 4);
            check(kink.trailCount == 1u, "kink fixture builds one trail");
            // Sample 0's side is +Z (dir +X), sample 1's side is -Z (dir -X): the left rail's z flips.
            const fx z0 = kink.geom.positions[0].z;   // left of sample 0
            const fx z1 = kink.geom.positions[2].z;   // left of sample 1
            check((z0 < 0) != (z1 < 0), "direction reversal flips the side vector (the KINK, documented)");
            std::printf("vfx kink worst-case digest: 0x%016llx\n",
                        (unsigned long long)vr::DigestStrip(kink.geom));
            check(vr::DigestStrip(kink.geom) == kPinnedKinkDigest, "kink worst-case digest pinned");
        }

        std::printf("vfx trails digest: 0x%016llx  ribbon digest: 0x%016llx\n",
                    (unsigned long long)vr::DigestTrails(th),
                    (unsigned long long)vr::DigestStrip(rib.geom));
        check(vr::DigestTrails(th) == kPinnedTrailsDigest, "trail-history digest pinned");
        check(vr::DigestStrip(rib.geom) == kPinnedRibbonDigest, "ribbon digest pinned");
    }

    // ================= (b) BEAMS =================
    {
        const FxVec3 a = vr::ShotBeamA(), b = vr::ShotBeamB();

        // Segment count EXACT: segments+1 samples x 2 rails; segments x 6 indices.
        const vr::StripGeom beam = vr::BuildBeam(a, b, vr::kShotBeamW, vr::kShotBeamSegs,
                                                 vr::kShotBeamAmp, vr::kShotBeamSeed, 96u);
        check(beam.positions.size() == (size_t)(vr::kShotBeamSegs + 1) * 2u,
              "beam sample count exact ((segments+1)*2 rail verts)");
        check(beam.indices.size() == (size_t)vr::kShotBeamSegs * 6u,
              "beam index count exact (segments*6)");

        // Deterministic per (endpoints, seed, tick): same inputs byte-identical; a different tick or
        // seed produces a DIFFERENT jitter stream.
        const vr::StripGeom beam2 = vr::BuildBeam(a, b, vr::kShotBeamW, vr::kShotBeamSegs,
                                                  vr::kShotBeamAmp, vr::kShotBeamSeed, 96u);
        check(beam.positions.size() == beam2.positions.size() &&
                  std::memcmp(beam.positions.data(), beam2.positions.data(),
                              beam.positions.size() * sizeof(FxVec3)) == 0,
              "beam is a pure function of (endpoints, params, seed, tick)");
        const vr::StripGeom beamT = vr::BuildBeam(a, b, vr::kShotBeamW, vr::kShotBeamSegs,
                                                  vr::kShotBeamAmp, vr::kShotBeamSeed, 97u);
        check(std::memcmp(beam.positions.data(), beamT.positions.data(),
                          beam.positions.size() * sizeof(FxVec3)) != 0,
              "a different tick re-rolls the jitter (the animated-lightning look)");

        // JITTER == 0 IS THE EXACT STRAIGHT STRIP (identity): independent of seed/tick, and every
        // centerline point ((left+right)/2) equals the exact integer lerp.
        const vr::StripGeom s1 = vr::BuildBeam(a, b, vr::kShotBeamW, vr::kShotBeamSegs, 0, 123u, 9u);
        const vr::StripGeom s2 = vr::BuildBeam(a, b, vr::kShotBeamW, vr::kShotBeamSegs, 0, 999u, 77u);
        check(std::memcmp(s1.positions.data(), s2.positions.data(),
                          s1.positions.size() * sizeof(FxVec3)) == 0 &&
                  s1.indices == s2.indices,
              "jitter=0 -> the straight strip is seed/tick-INDEPENDENT (identity)");
        bool lerpExact = true;
        const FxVec3 delta = pt::FxSub(b, a);
        for (int j = 0; j <= vr::kShotBeamSegs; ++j) {
            const FxVec3& L = s1.positions[(size_t)(2 * j)];
            const FxVec3& R = s1.positions[(size_t)(2 * j + 1)];
            const FxVec3 mid{(fx)(((int64_t)L.x + R.x) / 2), (fx)(((int64_t)L.y + R.y) / 2),
                             (fx)(((int64_t)L.z + R.z) / 2)};
            const FxVec3 want{a.x + (fx)(((int64_t)delta.x * j) / vr::kShotBeamSegs),
                              a.y + (fx)(((int64_t)delta.y * j) / vr::kShotBeamSegs),
                              a.z + (fx)(((int64_t)delta.z * j) / vr::kShotBeamSegs)};
            if (mid.x != want.x || mid.y != want.y || mid.z != want.z) lerpExact = false;
        }
        check(lerpExact, "jitter=0 centerline == the exact integer lerp on every sample");

        // Endpoints are ANCHORED even with jitter on (only interior points wobble).
        check(std::memcmp(&beam.positions[0], &s1.positions[0], 2 * sizeof(FxVec3)) == 0 &&
                  std::memcmp(&beam.positions[beam.positions.size() - 2],
                              &s1.positions[s1.positions.size() - 2], 2 * sizeof(FxVec3)) == 0,
              "beam endpoints anchored (never jittered)");

        std::printf("vfx beam digest: 0x%016llx\n", (unsigned long long)vr::DigestStrip(beam));
        check(vr::DigestStrip(beam) == kPinnedBeamDigest, "beam digest pinned");
    }

    // ================= (c) MESH EMITTERS =================
    {
        pt::ParticlePool pool;
        vr::TrailHistory th;
        RunTrailFixture(pool, th, 10);

        const std::vector<vr::FxInstance> inst = vr::BuildMeshInstances(pool);
        check((uint32_t)inst.size() == pt::CountAlive(pool), "instance count == alive count EXACTLY");

        // Every orientation is a unit quaternion to +-LSB (|q|^2 within 8 LSB of kOne in Q16.16) and
        // FxRotate(q, +Z) lands on the normalized velocity within the integer-sqrt band.
        bool unit = true, aims = true;
        size_t k = 0;
        for (uint32_t i = 0; i < (uint32_t)pool.particles.size(); ++i) {
            const pt::FxParticle& p = pool.particles[i];
            if (!(p.flags & pt::kFlagAlive)) continue;
            const hf::sim::fpx::FxQuat q = inst[k].orient;
            const int64_t n2 = ((int64_t)q.x * q.x + (int64_t)q.y * q.y +
                                (int64_t)q.z * q.z + (int64_t)q.w * q.w) >> kFrac;
            if (n2 < kOne - 8 || n2 > kOne + 8) unit = false;
            const FxVec3 fwd = hf::sim::fpx::FxRotate(q, FxVec3{0, 0, kOne});
            const FxVec3 dir = pt::FxNormalize(p.vel);
            if ((fwd.x - dir.x > 1024) || (dir.x - fwd.x > 1024) ||
                (fwd.y - dir.y > 1024) || (dir.y - fwd.y > 1024) ||
                (fwd.z - dir.z > 1024) || (dir.z - fwd.z > 1024)) aims = false;
            ++k;
        }
        check(unit, "every instance quaternion is unit to +-8 LSB");
        check(aims, "FxRotate(orient, +Z) tracks the velocity direction (integer half-angle yaw+pitch)");

        // Deterministic edge cases: zero velocity -> identity; straight up -> a pure X-pitch quat.
        const hf::sim::fpx::FxQuat qid = vr::OrientFromVelocity(FxVec3{0, 0, 0});
        check(qid.x == 0 && qid.y == 0 && qid.z == 0 && qid.w == kOne, "zero velocity -> identity");
        const hf::sim::fpx::FxQuat qup = vr::OrientFromVelocity(FxVec3{0, 5 * kOne, 0});
        check(qup.y == 0 && qup.z == 0 && qup.x < 0, "straight-up velocity -> a pure negative-X pitch");

        std::printf("vfx mesh-instance digest: 0x%016llx\n",
                    (unsigned long long)vr::DigestInstances(inst));
        check(vr::DigestInstances(inst) == kPinnedInstDigest, "mesh-instance digest pinned");
    }

    // ================= (d) PARTICLE LIGHTS + THE ML1 COMPOSITION =================
    {
        pt::ParticlePool pool;
        vr::TrailHistory th;
        RunTrailFixture(pool, th, 10);
        const uint32_t alive = pt::CountAlive(pool);   // 20 < 64

        // Exactly min(N, alive) entries, both below and above the cap.
        const std::vector<vr::ParticleLightFx> lights = vr::BuildParticleLights(pool, 64u);
        check((uint32_t)lights.size() == alive, "alive < cap -> exactly `alive` lights");
        const std::vector<vr::ParticleLightFx> capped = vr::BuildParticleLights(pool, 8u);
        check(capped.size() == 8u, "alive > cap -> exactly cap lights");

        // Brightest-first order (remaining life descending) with the lowest-slot tie-break: particles
        // spawned the SAME tick share a brightness -> they must appear in ascending slot order.
        bool ordered = true;
        for (size_t i = 0; i + 1 < lights.size(); ++i) {
            if (lights[i].bright < lights[i + 1].bright) ordered = false;
            if (lights[i].bright == lights[i + 1].bright && lights[i].slot >= lights[i + 1].slot)
                ordered = false;
        }
        check(ordered, "brightest-then-lowest-slot total order (the pinned tie-break)");
        // The fixture spawns 2/tick with one lifetime: the two brightest are this tick's spawns, and
        // the tie between them resolves to the lower slot first.
        check(lights.size() >= 2 && lights[0].bright == lights[1].bright &&
                  lights[0].slot < lights[1].slot,
              "same-tick spawns tie on brightness and resolve by slot");

        std::printf("vfx light digest: 0x%016llx\n", (unsigned long long)vr::DigestLights(lights));
        check(vr::DigestLights(lights) == kPinnedLightDigest, "light-selection digest pinned");

        // THE ML1 COMPOSITION PROOF: the particle lights feed render::clustered::BuildClusters (the
        // EXISTING ML1 machinery, byte-untouched) and the assignment digest is pinned (via the showcase
        // scenario below, which uses the fixed camera). Structural check here: every listed light index
        // is in range and overlaps its cluster AABB.
        const vr::VfxShotRun run = vr::RunVfxShotScenario();
        check(run.assignStats.totalAssignments > 0, "particle lights land in clusters (ML1 seam live)");
        std::printf("vfx ml1 assignment: {lights:%u, maxPerCluster:%u, totalAssignments:%llu, "
                    "assignDigest:0x%016llx}\n",
                    (uint32_t)run.lights.size(), run.assignStats.maxPerCluster,
                    (unsigned long long)run.assignStats.totalAssignments,
                    (unsigned long long)run.assignDigest);
        check(run.assignDigest == kPinnedAssignDigest,
              "ML1 assignment digest pinned (toolchain-family pin — the ML1 convention)");
        check(run.assignStats.maxPerCluster == kPinnedAssignMax, "maxPerCluster pinned");
        check(run.assignStats.totalAssignments == kPinnedAssignTotal, "totalAssignments pinned");
    }

    // ================= (e) THE SHOWCASE SCENARIO (the shared-scenario witness) =================
    {
        const vr::VfxShotRun r1 = vr::RunVfxShotScenario();
        const vr::VfxShotRun r2 = vr::RunVfxShotScenario();
        check(r1.digest == r2.digest && r1.trailsDigest == r2.trailsDigest &&
                  r1.beamDigest == r2.beamDigest && r1.instDigest == r2.instDigest &&
                  r1.lightDigest == r2.lightDigest,
              "showcase scenario two-run identical (every generator digest)");

        std::printf("vfx shot: {particles:%u, trails:%u, ribbonTris:%u, beamSegs:%d, meshInst:%u, "
                    "lights:%u, digest:0x%016llx}\n",
                    r1.aliveCount, r1.ribbons.trailCount,
                    (uint32_t)(r1.ribbons.geom.indices.size() / 3u), vr::kShotBeamSegs,
                    (uint32_t)r1.instances.size(), (uint32_t)r1.lights.size(),
                    (unsigned long long)r1.digest);
        check(r1.aliveCount == kPinnedShotAlive, "showcase alive count pinned");
        check(r1.ribbons.trailCount == kPinnedShotTrails, "showcase trail count pinned");
        check((uint32_t)(r1.ribbons.geom.indices.size() / 3u) == kPinnedShotRibbonTris,
              "showcase ribbon triangle count pinned");
        check((uint32_t)r1.instances.size() == kPinnedShotInst, "showcase mesh instances == alive");
        check((uint32_t)r1.lights.size() == kPinnedShotLights, "showcase lights == min(64, alive)");
        check(r1.digest == kPinnedShotDigest, "showcase combined integer digest pinned");

        // The shared raster is a pure function of the run (two renders byte-identical).
        std::vector<uint8_t> imgA, imgB;
        uint32_t w = 0, h = 0, w2 = 0, h2 = 0;
        vr::RenderVfxShot(r1, imgA, w, h);
        vr::RenderVfxShot(r2, imgB, w2, h2);
        check(w == (uint32_t)vr::kShotImgW && h == (uint32_t)vr::kShotImgH, "raster dimensions");
        check(w == w2 && h == h2 && imgA == imgB, "two raster renders byte-identical (strict-zero seed)");
    }

    if (g_fail == 0) std::printf("vfx_render_test: all checks passed\n");
    else std::printf("vfx_render_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
