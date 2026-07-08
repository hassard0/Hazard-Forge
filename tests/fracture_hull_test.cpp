// Slice DH1 — DETERMINISTIC CONVEX-CELL FRACTURE HULLS + DEBRIS + DUST (engine/sim/fracture_hull.h,
// hf::sim::fhull). Pure CPU, header-only. This test PINS the DH1 contracts and, critically, treats the
// composed fracture flagship (fract.h) + convex stack (gjk.h/manifold.h) + particles.h as READ-ONLY: DH1 only
// reads their output types, so their goldens stay byte-identical (verified separately by the controller).
//
// WHAT THIS PINS:
//   (A) HULL: the exact integer 3D convex hull builder — a cube point cloud -> its 8 corners (the cube
//       exactly), a tetra/octa point cloud -> 4/6 verts, degenerate (collinear/coplanar) -> rejected; a solid
//       3x3x3 cube CELL -> an 8-vert / 6-face box hull with EXACT volume + inertia; hull volume <= AABB volume
//       (the tightness win).
//   (B) COLLISION: a dynamic convex shard rests STABLE on a static floor on a FLAT FACE (contact normal
//       axis-aligned ~ (0,1,0), NOT radial; near-identity resting pose).
//   (C) SHATTER SETTLE: shatter a block -> the shards settle interlocked (rest + face-dominant contacts +
//       a full-scene digest pinned) — with an HONEST exact-vs-fallback hull count.
//   (D) DEBRIS + DUST: debris count deterministic; dust spawns one mote per severed bond at the fracture
//       surface (count + positions + digest pinned).
//   (E) LOCKSTEP: a peer re-derives the shattered settle bit-for-bit from inputs alone; rollback corrects a
//       mispredicted impact to authority EXACTLY; snapshot round-trip.
//
// Pure C++ (hf_core), ASan-eligible. Digests pinned identical MSVC + local clang.
#include "sim/fracture_hull.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"

using namespace hf;
namespace fhull = hf::sim::fhull;
namespace fract = hf::sim::fract;
namespace fpx   = hf::sim::fpx;
namespace convex = hf::sim::convex;
namespace gjk   = hf::sim::gjk;
namespace particles = hf::sim::particles;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

using fpx::fx;
using fpx::kOne;
using fpx::FxVec3;

// Build a single-cell fracture field (nx*ny*nz, one seed at the center) -> ExtractFragments gives ONE fragment
// whose cell vertex set is the WHOLE lattice (a solid box). The clean identity fixture.
static void MakeSolidBoxFragment(int nx, int ny, int nz, fract::FractField& field, fract::FractCells& cells,
                                 fract::FractFragments& frags) {
    field.nx = nx; field.ny = ny; field.nz = nz;
    std::vector<fract::FractSeed> seeds = {{nx / 2, ny / 2, nz / 2}};
    fract::ClassifyFractCells(field, seeds, cells);
    fract::ExtractFragments(field, cells, 1, frags);
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= (A) the exact integer convex-hull builder =================
    {
        // cube corners + interior/face samples -> exactly the 8 corners.
        std::vector<FxVec3> cube;
        for (int x = 0; x <= 2; ++x) for (int y = 0; y <= 2; ++y) for (int z = 0; z <= 2; ++z)
            cube.push_back(FxVec3{x, y, z});   // 27 lattice points (a solid 3x3x3 block)
        const fhull::HullBuild hb = fhull::BuildConvexHull(cube);
        check(hb.ok, "hull: cube build ok");
        check(hb.vertIdx.size() == 8, "hull: cube -> 8 hull vertices (the cube exactly)");
        check(fhull::CountDistinctFacePlanes(cube, hb.tris) == 6, "hull: cube -> 6 distinct face planes");
        // volume: 2x2x2 = 8 cells -> 6*vol = 48.
        check(fhull::HullLatticeVolume6(cube, hb.tris) == 48, "hull: cube 6*volume == 48 (exact)");

        // tetra corners + centroid -> 4 verts.
        std::vector<FxVec3> tet = {{0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {0, 0, 4}, {1, 1, 1}};
        const fhull::HullBuild ht = fhull::BuildConvexHull(tet);
        check(ht.ok && ht.vertIdx.size() == 4, "hull: tetra+interior -> 4 hull vertices");

        // octahedron poles + center -> 6 verts (a NON-box exact hull).
        std::vector<FxVec3> oct = {{2, 0, 0}, {-2, 0, 0}, {0, 2, 0}, {0, -2, 0}, {0, 0, 2}, {0, 0, -2}, {0, 0, 0}};
        const fhull::HullBuild ho = fhull::BuildConvexHull(oct);
        check(ho.ok && ho.vertIdx.size() == 6, "hull: octa -> 6 hull vertices (non-box)");
        FxVec3 dummy[8];
        check(!fhull::IsAxisBox(oct, ho.vertIdx, dummy), "hull: octa is NOT an axis box");

        // degenerate: collinear -> rejected; coplanar (flat) -> rejected.
        std::vector<FxVec3> line = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
        check(!fhull::BuildConvexHull(line).ok, "hull: collinear rejected");
        std::vector<FxVec3> flat = {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {2, 2, 0}, {1, 1, 0}};
        check(!fhull::BuildConvexHull(flat).ok, "hull: coplanar rejected (degenerate -> AABB fallback)");
    }

    // ================= (A') a solid box CELL -> exact box hull + exact inertia + tightness =================
    {
        fract::FractField field; fract::FractCells cells; fract::FractFragments frags;
        MakeSolidBoxFragment(3, 3, 3, field, cells, frags);   // 27 samples, one fragment
        check(frags.fragments.size() == 1, "cell: one fragment for a 1-seed field");
        const fhull::CellHull ch = fhull::BuildCellHull(field, cells, frags, 0u, kOne);
        check(ch.exact, "cell: 3x3x3 box cell -> EXACT hull (not fallback)");
        check(ch.isBox, "cell: box cell recognized as an axis box");
        check(ch.hullVerts == 8, "cell: box cell -> 8 hull verts");
        check(ch.faceCount == 6, "cell: box cell -> 6 faces");
        // 2x2x2 world box (half-extent kOne with cellSize kOne) -> volume 8*kOne; hull == AABB (tightness =).
        check(ch.worldVolume == (fx)(8 * (int)kOne), "cell: box world volume == 8 (exact)");
        check(ch.worldVolume <= ch.aabbVolume, "cell: hull volume <= AABB volume (tightness)");
        check(ch.worldVolume == ch.aabbVolume, "cell: box hull volume == AABB volume (solid box, equality)");

        // EXACT inertia: the collision-path hull inertia equals the analytic box inertia (the gjk identity).
        const fx invMass = kOne / 8;   // some mass
        const FxVec3 gjkDiag = gjk::FxHullInvInertiaBody(ch.hull, invMass);
        convex::FxBox box; box.halfExtents = FxVec3{kOne, kOne, kOne};   // the recognized 2x2x2 box
        const FxVec3 boxDiag = convex::FxBoxInvInertiaBody(box, invMass);
        check(gjkDiag.x == boxDiag.x && gjkDiag.y == boxDiag.y && gjkDiag.z == boxDiag.z,
              "cell: box hull inertia == analytic box inertia (EXACT, gjk identity)");
        // the WH full-convex inertia (manifold) is symmetric with ~zero products of inertia for the box.
        const convex::FxMat3 invI = fhull::CellHullInvInertiaBody(ch, invMass);
        check(invI.m[0] > 0 && invI.m[4] > 0 && invI.m[8] > 0, "cell: box full inertia positive diagonal");
        auto absfx = [](fx v) { return v < 0 ? -v : v; };
        check(absfx(invI.m[1]) <= kOne / 64 && absfx(invI.m[2]) <= kOne / 64 && absfx(invI.m[5]) <= kOne / 64,
              "cell: box full inertia products-of-inertia ~ 0 (axis-aligned box)");
    }

    // ================= (B) a shard rests stable on a static floor on a FLAT FACE =================
    {
        gjk::HullWorld world;
        // static floor box hull.
        fpx::FxBody floor; floor.pos = FxVec3{0, 0, 0}; floor.invMass = 0; floor.flags = 0;
        floor.orient = fpx::FxQuat{0, 0, 0, kOne};
        world.bodies.push_back(floor);
        world.hulls.push_back(gjk::MakeBox((fx)(10 * (int)kOne), kOne, (fx)(10 * (int)kOne)));
        // a dynamic box dropped just above the floor top (y = groundY + halfExtent + a small gap).
        fpx::FxBody b; b.pos = FxVec3{0, (fx)(3 * (int)kOne), 0}; b.invMass = kOne; b.flags = fpx::kFlagDynamic;
        b.orient = fpx::FxQuat{0, 0, 0, kOne};
        world.bodies.push_back(b);
        world.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));

        convex::ConvexStepConfig cfg;
        cfg.gravity = FxVec3{0, (fx)((int64_t)(-98 * (int)kOne) / 10), 0};   // -9.8
        cfg.dt = kOne / 60; cfg.solveIters = 12; cfg.restitution = 0; cfg.slop = kOne / 64;
        cfg.beta = (fx)((int64_t)5 * kOne / 10); cfg.posIters = 4;
        cfg.linDamp = (fx)((int64_t)99 * kOne / 100); cfg.angDamp = (fx)((int64_t)9 * kOne / 10);
        fhull::StepFractureHullN(world, cfg, 240u);

        const fpx::FxBody& rb = world.bodies[1];
        // rests near the floor top (y ~ groundY(1) + halfExtent(1) = 2; allow a band).
        check(rb.pos.y > (fx)(1 * (int)kOne) && rb.pos.y < (fx)(3 * (int)kOne), "collide: shard rests on floor top");
        check(fpx::FxLength(rb.vel) < kOne, "collide: shard at rest (low speed)");
        // near-identity orientation (rested flat, did not tumble).
        auto absfx = [](fx v) { return v < 0 ? -v : v; };
        check(absfx(rb.orient.x) < kOne / 8 && absfx(rb.orient.z) < kOne / 8, "collide: shard rests near-flat");
        // FACE-to-face contact normal with the floor (axis-aligned ~ (0, +-1, 0), NOT radial).
        const FxVec3 nrm = fhull::FaceToFaceContactNormal(world, 1u, 0u);
        const fx ax = absfx(nrm.x), ay = absfx(nrm.y), az = absfx(nrm.z);
        check(ay > ax && ay > az, "collide: floor contact normal is the +-Y FACE normal (not radial)");
        const fx tot = ax + ay + az;
        check(tot > 0 && (int64_t)ay * 4 >= (int64_t)tot * 3, "collide: contact normal face-dominant (>=3/4 on Y)");
    }

    // ================= a reusable shatter scene builder =================
    auto buildShatterScene = [](fhull::FractHullConfig& scfg, fract::FractBonds& bonds,
                                std::vector<uint8_t>& severed) -> fhull::FractHullScene {
        static fract::FractField field; field.nx = 8; field.ny = 8; field.nz = 6;
        static const std::vector<fract::FractSeed> seeds = {
            {1, 1, 1}, {6, 1, 4}, {1, 6, 1}, {6, 6, 4}, {3, 3, 2}, {5, 2, 1},
            {2, 5, 3}, {4, 6, 2}, {6, 3, 3}, {3, 5, 4},
        };
        static fract::FractCells cells; fract::ClassifyFractCells(field, seeds, cells);
        static fract::FractFragments frags; fract::ExtractFragments(field, cells, (int)seeds.size(), frags);
        fract::BuildFractBonds(field, cells, frags, bonds);
        fract::BreakImpact hardImpact{0u, (fx)(800 * (int)kOne)};
        fract::ApplyImpactBreak(bonds, frags, hardImpact, 4, severed);
        std::vector<uint32_t> clusters;
        fract::CountFractPieces(frags, bonds, severed, &clusters);

        scfg.worldCellSize = kOne / 2;
        scfg.gravity = FxVec3{0, (fx)((int64_t)(-98 * (int)kOne) / 10), 0};
        scfg.groundY = 0;
        scfg.impactDir = FxVec3{kOne / 2, -kOne, 0};
        scfg.impactSpeed = (fx)(3 * (int)kOne);
        scfg.floorHalfExtents = FxVec3{(fx)(12 * (int)kOne), kOne, (fx)(12 * (int)kOne)};
        return fhull::SpawnFractureHullScene(field, cells, frags, bonds, severed, clusters, hardImpact, scfg);
    };

    // ================= (C) SHATTER SETTLE (+ honest exact-vs-fallback report) =================
    convex::ConvexStepConfig scfg;
    scfg.gravity = FxVec3{0, (fx)((int64_t)(-98 * (int)kOne) / 10), 0};
    scfg.dt = kOne / 60; scfg.solveIters = 10; scfg.restitution = 0; scfg.slop = kOne / 64;
    scfg.beta = (fx)((int64_t)4 * kOne / 10); scfg.posIters = 4;
    scfg.linDamp = (fx)((int64_t)99 * kOne / 100); scfg.angDamp = (fx)((int64_t)9 * kOne / 10);
    const uint32_t kSettleTicks = 200u;

    fhull::FractHullConfig cfg0; fract::FractBonds bonds0; std::vector<uint8_t> severed0;
    const fhull::FractHullScene scene0 = buildShatterScene(cfg0, bonds0, severed0);
    check(scene0.world.bodies.size() >= 2, "shatter: scene has fragment bodies + a floor");
    std::printf("dh1 shatter: fragments=%u exactHulls=%u debris=%u floorIndex=%u\n",
                (uint32_t)scene0.cells.size(), scene0.exactHulls, scene0.debrisCount, scene0.floorIndex);
    check(scene0.exactHulls >= 1u, "shatter: at least one fragment got the EXACT cell hull (not fallback)");

    gjk::HullWorld settled = scene0.world;
    fhull::StepFractureHullN(settled, scfg, kSettleTicks);
    // determinism: a second full settle is byte-identical.
    {
        gjk::HullWorld settled2 = scene0.world;
        fhull::StepFractureHullN(settled2, scfg, kSettleTicks);
        check(convex::ConvexBodiesEqual(settled.bodies, settled2.bodies), "shatter: two settles BYTE-IDENTICAL");
    }
    const fhull::FractHullState st = fhull::MeasureFractureHull(scene0, settled, kOne);
    std::printf("dh1 settle: dynamic=%u rested=%u faceRest=%u maxSpeed=%d maxPen=%d minY=%d\n",
                st.dynamic, st.rested, st.faceRestPairs, (int)st.maxSpeed, (int)st.maxPenetration,
                (int)st.minDynamicY);
    check(st.dynamic >= 1u, "shatter: >=1 dynamic shard");
    check(st.maxSpeed < (fx)(2 * (int)kOne), "shatter: shards settle (bounded speed)");
    check(st.faceRestPairs >= 1u, "shatter: >=1 shard rests on a FACE-dominant contact (interlocked)");

    const uint64_t sceneDigest = fhull::SceneBodiesFnv64(settled);
    std::printf("dh1 scene digest = 0x%016llx\n", (unsigned long long)sceneDigest);
    // PIN the full-scene settled digest (baked from the MSVC run; asserted identical MSVC + clang).
    const uint64_t kExpectSceneDigest = 0x2ebb8c28912b74acull;   // baked; identical MSVC + clang
    check(sceneDigest == kExpectSceneDigest, "shatter: full-scene digest matches the pinned value");

    // ================= (D) DEBRIS + DUST =================
    {
        // dust: one mote per SEVERED bond, at the fracture-surface midpoint (world units).
        uint32_t severedCount = 0;
        for (uint8_t s : severed0) if (s) ++severedCount;
        const particles::ParticlePool dust =
            fhull::SpawnFractureDust(bonds0, severed0, cfg0, (fx)(2 * (int)kOne), 256u);
        const uint32_t alive = particles::CountAlive(dust);
        std::printf("dh1 dust: severedBonds=%u aliveMotes=%u\n", severedCount, alive);
        check(alive == severedCount, "dust: one alive mote per severed bond");
        // placement: the k-th alive mote is at the k-th severed bond midpoint * cellSize.
        bool placedOk = true;
        uint32_t moteSlot = 0;
        for (size_t bi = 0; bi < bonds0.bonds.size(); ++bi) {
            if (bi >= severed0.size() || !severed0[bi]) continue;
            const FxVec3 want = FxVec3{fpx::fxmul(bonds0.bonds[bi].midpoint.x, cfg0.worldCellSize),
                                       fpx::fxmul(bonds0.bonds[bi].midpoint.y, cfg0.worldCellSize),
                                       fpx::fxmul(bonds0.bonds[bi].midpoint.z, cfg0.worldCellSize)};
            const particles::FxParticle& p = dust.particles[(size_t)moteSlot];   // ascending slots (LIFO from 0)
            if (!(p.pos.x == want.x && p.pos.y == want.y && p.pos.z == want.z)) placedOk = false;
            ++moteSlot;
        }
        check(placedOk, "dust: motes placed at the severed-bond fracture-surface midpoints");
        const uint64_t dustDigest = fhull::DustFnv64(dust);
        std::printf("dh1 dust digest = 0x%016llx\n", (unsigned long long)dustDigest);
        const uint64_t kExpectDustDigest = 0x383c010806829b2bull;   // baked; identical MSVC + clang
        check(dustDigest == kExpectDustDigest, "dust: pool digest matches the pinned value");

        // debris: deterministic count (a small-shard classification over the fragments).
        std::printf("dh1 debris count = %u\n", scene0.debrisCount);
        const uint32_t kExpectDebris = 1u;   // baked; identical MSVC + clang
        check(scene0.debrisCount == kExpectDebris, "debris: count matches the pinned value");
    }

    // ================= (E) LOCKSTEP + ROLLBACK =================
    {
        // pick the first dynamic shard to kick.
        uint32_t kick = 0xFFFFFFFFu;
        for (uint32_t i = 0; i < (uint32_t)scene0.world.bodies.size(); ++i)
            if ((i != scene0.floorIndex) && (scene0.world.bodies[i].flags & fpx::kFlagDynamic)) { kick = i; break; }
        check(kick != 0xFFFFFFFFu, "lockstep: found a dynamic shard to kick");

        const std::vector<convex::ConvexCommand> authStream = {
            convex::ConvexCommand{2u, convex::kConvexCmdAddImpulse, kick, FxVec3{(fx)(1500 * (int)kOne), 0, 0}},
            convex::ConvexCommand{5u, convex::kConvexCmdSetAngVel, kick, FxVec3{0, kOne, 0}},
        };
        bool identical = false;
        const gjk::HullWorld authority =
            fhull::RunFractureHullLockstep(scene0.world, scfg, authStream, 60u, &identical);
        check(identical, "lockstep: authority == replica BIT-IDENTICAL from inputs alone");

        std::vector<convex::ConvexCommand> mispredict = authStream;
        mispredict.push_back(convex::ConvexCommand{20u, convex::kConvexCmdAddImpulse, kick,
                             FxVec3{0, 0, (fx)(3000 * (int)kOne)}});
        bool corrected = false, diverged = false;
        const gjk::HullWorld rolled =
            fhull::RunFractureHullRollback(scene0.world, scfg, authStream, mispredict, 60u, 20u,
                                           &corrected, &diverged);
        check(corrected, "rollback: corrected == authority BIT-EXACT");
        check(diverged, "rollback: mispredict diverged before rollback (a real divergence fixed)");
        check(convex::ConvexBodiesEqual(rolled.bodies, authority.bodies), "rollback: rolled == authority bodies");
    }

    if (g_fail == 0) std::printf("fracture_hull_test: ALL PASS\n");
    else std::printf("fracture_hull_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
