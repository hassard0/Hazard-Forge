#pragma once
// Slice S — deterministic impulse-based rigid-body world. Pure C++ (engine/math + stdlib only);
// NO RHI or graphics-backend symbols. Compiled into hf_core (ASan-scoped) and hf_engine.
#include <vector>
#include "math/math.h"
#include "physics/body.h"

namespace hf::physics {

// A single static ground plane (y = groundY, normal +Y) plus a vector of rigid bodies. Step()
// advances the simulation one fixed timestep with a deterministic, fixed-iteration-order solver
// (no RNG, no clock): two runs with identical inputs on the same build are bit-identical.
struct World {
    math::Vec3 gravity{0.0f, -9.81f, 0.0f};
    float groundY = 0.0f;                 // static infinite plane, normal +Y
    std::vector<RigidBody> bodies;

    // Solver tuning (fixed; documented in the design spec).
    int   solverIterations = 8;
    float restitutionSlop  = 1.0f;        // approach speed (m/s) below which restitution is zeroed
    float baumgarte        = 0.2f;        // positional-correction fraction
    float penetrationSlop  = 0.005f;      // allowed penetration before correction kicks in

    void Step(float dt);

    // Convenience: total kinetic energy (used by tests to bound "no explosion").
    float KineticEnergy() const;
};

} // namespace hf::physics
