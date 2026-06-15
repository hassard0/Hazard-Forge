#pragma once
// Slice S — minimal impulse-based rigid bodies. Pure C++ (engine/math + stdlib only); NO RHI or
// graphics-backend symbols. Compiled into hf_core (ASan-scoped, unit-tested) and hf_engine.
#include "math/math.h"

namespace hf::physics {

enum class Shape { Sphere, Box };

// A single rigid body. invMass == 0 marks a STATIC (infinite-mass) body that never moves. The
// inertia is stored as the body-space diagonal of the INVERSE inertia tensor (invInertiaDiag); for
// the rotation-symmetric sphere this is the same in every axis, and a zero vector disables angular
// response. Linear/angular velocity are in world space; orientation maps body->world.
struct RigidBody {
    math::Vec3 position;                              // center of mass, world space
    math::Quat orientation = math::Quat::Identity();  // body -> world
    math::Vec3 linVel;                                // world-space linear velocity
    math::Vec3 angVel;                                // world-space angular velocity
    float invMass = 0.0f;                             // 0 == static / infinite mass
    math::Vec3 invInertiaDiag;                        // body-space inverse inertia (diagonal)
    Shape shape = Shape::Sphere;
    float radius = 0.5f;                              // sphere collider radius
    math::Vec3 halfExtents{0.5f, 0.5f, 0.5f};         // box collider half-extents
    float restitution = 0.2f;                         // bounciness (small so piles settle)
    float friction = 0.5f;                            // Coulomb friction coefficient

    // Model matrix for rendering: the engine sphere/cube meshes are unit primitives (sphere r=0.5,
    // cube ±0.5), so the render scale maps that to the collider size (2*radius / 2*halfExtents).
    math::Mat4 Transform() const {
        math::Vec3 s = (shape == Shape::Sphere)
            ? math::Vec3{2.0f * radius, 2.0f * radius, 2.0f * radius}
            : math::Vec3{2.0f * halfExtents.x, 2.0f * halfExtents.y, 2.0f * halfExtents.z};
        return math::FromTRS(position, orientation, s);
    }
};

// Construct a DYNAMIC solid-sphere body. Mass = density * (4/3)pi r^3; solid-sphere inertia
// I = 2/5 m r^2 (uniform on all axes), so invInertiaDiag = 1/I on each axis. RNG/clock-free.
inline RigidBody MakeDynamicSphere(const math::Vec3& pos, float radius, float density = 1.0f) {
    RigidBody b;
    b.shape = Shape::Sphere;
    b.position = pos;
    b.radius = radius;
    float mass = density * (4.0f / 3.0f) * 3.14159265358979323846f * radius * radius * radius;
    b.invMass = (mass > 0.0f) ? 1.0f / mass : 0.0f;
    float I = (2.0f / 5.0f) * mass * radius * radius;   // solid sphere
    float invI = (I > 0.0f) ? 1.0f / I : 0.0f;
    b.invInertiaDiag = {invI, invI, invI};
    return b;
}

// Construct a STATIC sphere (invMass 0, no angular response). Used for fixed obstacles.
inline RigidBody MakeStaticSphere(const math::Vec3& pos, float radius) {
    RigidBody b;
    b.shape = Shape::Sphere;
    b.position = pos;
    b.radius = radius;
    b.invMass = 0.0f;
    b.invInertiaDiag = {0.0f, 0.0f, 0.0f};
    return b;
}

} // namespace hf::physics
