// Slice S — deterministic impulse-based rigid-body solver. Pure C++ (engine/math + stdlib only).
#include "physics/world.h"

#include <algorithm>
#include <cmath>

namespace hf::physics {
namespace {

using math::Vec3;

// Apply the body-space diagonal inverse inertia to a WORLD-space vector. The collider inertia used
// here (solid sphere) is rotation-invariant — invInertiaDiag is identical on every axis — so the
// body->world rotation cancels and we can apply the diagonal directly in world space. (For a box
// this would need the full R * Iinv * R^T transform; the showcase + tests only use spheres, and box
// angular response is intentionally approximate.)
inline Vec3 ApplyInvInertia(const RigidBody& b, const Vec3& w) {
    return {w.x * b.invInertiaDiag.x, w.y * b.invInertiaDiag.y, w.z * b.invInertiaDiag.z};
}

struct Contact {
    int a;              // body index
    int b;              // body index, or -1 for the static ground plane
    Vec3 normal;        // unit, points from A's surface toward B (ground: +Y)
    float penetration;  // > 0 means overlapping by this depth
    Vec3 point;         // world-space contact point (on the shared surface)
};

// Velocity of body `bi` (or zero for the ground) at world point p, including the angular term.
inline Vec3 PointVelocity(const std::vector<RigidBody>& bodies, int bi, const Vec3& p) {
    if (bi < 0) return {0, 0, 0};
    const RigidBody& b = bodies[bi];
    Vec3 r = p - b.position;
    return b.linVel + math::cross(b.angVel, r);
}

// Apply an impulse `j` (world space) at world point p to body `bi` (ground bi<0 is immovable).
inline void ApplyImpulse(std::vector<RigidBody>& bodies, int bi, const Vec3& j, const Vec3& p) {
    if (bi < 0) return;
    RigidBody& b = bodies[bi];
    if (b.invMass == 0.0f) return;
    b.linVel = b.linVel + j * b.invMass;
    Vec3 r = p - b.position;
    b.angVel = b.angVel + ApplyInvInertia(b, math::cross(r, j));
}

// Effective inverse mass along direction `dir` at the contact (linear + angular), summed over both
// bodies. dir must be unit.
inline float EffectiveMass(const std::vector<RigidBody>& bodies, const Contact& c, const Vec3& dir) {
    float k = 0.0f;
    const RigidBody& A = bodies[c.a];
    k += A.invMass;
    {
        Vec3 rA = c.point - A.position;
        Vec3 rn = math::cross(rA, dir);
        Vec3 t = ApplyInvInertia(A, rn);
        k += math::dot(math::cross(t, rA), dir);
    }
    if (c.b >= 0) {
        const RigidBody& B = bodies[c.b];
        k += B.invMass;
        Vec3 rB = c.point - B.position;
        Vec3 rn = math::cross(rB, dir);
        Vec3 t = ApplyInvInertia(B, rn);
        k += math::dot(math::cross(t, rB), dir);
    }
    return k;
}

} // namespace

void World::Step(float dt) {
    const int n = (int)bodies.size();

    // (a) Integrate velocities: gravity on dynamic bodies only.
    for (int i = 0; i < n; ++i) {
        RigidBody& b = bodies[i];
        if (b.invMass > 0.0f) b.linVel = b.linVel + gravity * dt;
    }

    // (b) Detect contacts in a FIXED order: ground contacts (body index ascending) first, then
    // sphere-sphere pairs (i ascending, j>i ascending). Deterministic, no RNG.
    std::vector<Contact> contacts;
    contacts.reserve((size_t)n + (size_t)n * n / 2);

    for (int i = 0; i < n; ++i) {
        const RigidBody& b = bodies[i];
        if (b.shape != Shape::Sphere) continue;          // ground collision: spheres only
        float pen = b.radius - (b.position.y - groundY);
        if (pen > 0.0f) {
            Contact c;
            c.a = i; c.b = -1;
            // Convention: normal points from A toward B. B is the ground (below the sphere), so the
            // A->B normal is -Y; the resolved impulse on A is -jn*normal = +Y*jn (pushes the sphere
            // UP, out of the plane). This matches the sphere-sphere A->B convention exactly.
            c.normal = {0.0f, -1.0f, 0.0f};
            c.penetration = pen;
            // Contact point on the plane directly below the sphere center.
            c.point = {b.position.x, groundY, b.position.z};
            contacts.push_back(c);
        }
    }

    for (int i = 0; i < n; ++i) {
        const RigidBody& A = bodies[i];
        if (A.shape != Shape::Sphere) continue;
        for (int j = i + 1; j < n; ++j) {
            const RigidBody& B = bodies[j];
            if (B.shape != Shape::Sphere) continue;
            if (A.invMass == 0.0f && B.invMass == 0.0f) continue;  // static vs static: skip
            Vec3 d = B.position - A.position;
            float dist = math::length(d);
            float rsum = A.radius + B.radius;
            if (dist < rsum && dist > 1e-6f) {
                Contact c;
                c.a = i; c.b = j;
                c.normal = d / dist;                     // points A -> B
                c.penetration = rsum - dist;
                // Contact point: midpoint of the overlap along the normal.
                c.point = A.position + c.normal * (A.radius - 0.5f * c.penetration);
                contacts.push_back(c);
            }
        }
    }

    // (c) Resolve with sequential impulse: kIter passes, fixed order, per-contact accumulated
    // normal impulse so friction can Coulomb-clamp against it.
    const int kIter = solverIterations;
    std::vector<float> normalImpulse(contacts.size(), 0.0f);

    for (int it = 0; it < kIter; ++it) {
        for (size_t ci = 0; ci < contacts.size(); ++ci) {
            Contact& c = contacts[ci];
            // --- Normal impulse ---
            Vec3 vA = PointVelocity(bodies, c.a, c.point);
            Vec3 vB = PointVelocity(bodies, c.b, c.point);
            Vec3 rv = vB - vA;                            // relative velocity B - A
            float vn = math::dot(rv, c.normal);          // <0 means approaching
            float kn = EffectiveMass(bodies, c, c.normal);
            if (kn <= 0.0f) continue;

            // Restitution: only above the slop speed, else 0 (resting stacks don't jitter). Use the
            // smaller of the two bodies' restitution (a resting body shouldn't gain energy).
            float e = 0.0f;
            float restA = bodies[c.a].restitution;
            float restB = (c.b >= 0) ? bodies[c.b].restitution : restA;
            float restitution = std::min(restA, restB);
            if (-vn > restitutionSlop) e = restitution;

            // jn drives relative normal velocity to -e*vn (separating). Impulse on B is +jn*normal.
            float jn = -(1.0f + e) * vn / kn;
            // Accumulate + clamp >= 0 (the constraint can only push, never pull/stick).
            float old = normalImpulse[ci];
            float newImp = std::max(old + jn, 0.0f);
            jn = newImp - old;
            normalImpulse[ci] = newImp;

            Vec3 Pn = c.normal * jn;
            ApplyImpulse(bodies, c.b, Pn, c.point);
            ApplyImpulse(bodies, c.a, -Pn, c.point);

            // --- Friction impulse (Coulomb-clamped to mu * accumulated normal) ---
            vA = PointVelocity(bodies, c.a, c.point);
            vB = PointVelocity(bodies, c.b, c.point);
            rv = vB - vA;
            Vec3 vt = rv - c.normal * math::dot(rv, c.normal);   // tangential rel velocity
            float vtLen = math::length(vt);
            if (vtLen > 1e-6f) {
                Vec3 tdir = vt / vtLen;
                float kt = EffectiveMass(bodies, c, tdir);
                if (kt > 0.0f) {
                    float jt = -vtLen / kt;                       // kill tangential velocity
                    float muA = bodies[c.a].friction;
                    float muB = (c.b >= 0) ? bodies[c.b].friction : muA;
                    float mu = std::min(muA, muB);
                    float maxF = mu * normalImpulse[ci];
                    jt = std::max(-maxF, std::min(jt, maxF));
                    Vec3 Pt = tdir * jt;
                    ApplyImpulse(bodies, c.b, Pt, c.point);
                    ApplyImpulse(bodies, c.a, -Pt, c.point);
                }
            }
        }
    }

    // (d) Positional correction (Baumgarte): push bodies apart along the normal by a fraction of the
    // penetration beyond the slop, split by inverse mass. Position-only — injects NO velocity.
    for (size_t ci = 0; ci < contacts.size(); ++ci) {
        Contact& c = contacts[ci];
        float corr = std::max(c.penetration - penetrationSlop, 0.0f) * baumgarte;
        if (corr <= 0.0f) continue;
        float invA = bodies[c.a].invMass;
        float invB = (c.b >= 0) ? bodies[c.b].invMass : 0.0f;
        float invSum = invA + invB;
        if (invSum <= 0.0f) continue;
        Vec3 push = c.normal * (corr / invSum);
        // Normal points A->B, so move A back (-) and B forward (+).
        bodies[c.a].position = bodies[c.a].position - push * invA;
        if (c.b >= 0) bodies[c.b].position = bodies[c.b].position + push * invB;
    }

    // (e) Integrate positions (semi-implicit Euler) + orientation from angular velocity.
    for (int i = 0; i < n; ++i) {
        RigidBody& b = bodies[i];
        if (b.invMass == 0.0f) continue;                 // static bodies never move
        b.position = b.position + b.linVel * dt;
        b.orientation = math::IntegrateOrientation(b.orientation, b.angVel, dt);
    }
}

float World::KineticEnergy() const {
    float e = 0.0f;
    for (const auto& b : bodies) {
        if (b.invMass <= 0.0f) continue;
        float mass = 1.0f / b.invMass;
        e += 0.5f * mass * math::dot(b.linVel, b.linVel);
        // Angular term (diagonal inertia, world ~ body for spheres).
        float Ix = (b.invInertiaDiag.x > 0) ? 1.0f / b.invInertiaDiag.x : 0.0f;
        float Iy = (b.invInertiaDiag.y > 0) ? 1.0f / b.invInertiaDiag.y : 0.0f;
        float Iz = (b.invInertiaDiag.z > 0) ? 1.0f / b.invInertiaDiag.z : 0.0f;
        e += 0.5f * (Ix * b.angVel.x * b.angVel.x +
                     Iy * b.angVel.y * b.angVel.y +
                     Iz * b.angVel.z * b.angVel.z);
    }
    return e;
}

} // namespace hf::physics
