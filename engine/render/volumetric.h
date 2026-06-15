#pragma once
// Volumetric fog / light-shaft math — pure CPU (header-only, no device, no backend symbols).
// Shared by the --volumetric-shot showcase reasoning AND tests/volumetric_test.cpp so the unit test
// exercises the SAME phase function + world-ray reconstruction the in-shader ray-march uses.
//
// The volumetric pass (Slice AJ) ray-marches the view ray, sampling the directional shadow map at
// each step; lit steps add in-scattering weighted by the Henyey-Greenstein phase function and a fog
// density, attenuated by Beer-Lambert extinction. The pure pieces below are the HG phase and the
// camera-basis world-ray reconstruction (identical to sky.frag / volumetric.frag.hlsl).

#include "math/math.h"
#include <algorithm>
#include <cmath>

namespace hf::render::volumetric {

constexpr float kPi = 3.14159265358979323846f;

// Henyey-Greenstein phase function. Describes the angular distribution of single-scattered light:
// `cosTheta` is the cosine between the photon travel direction and the view (scattering) direction;
// `g` in (-1,1) is the asymmetry parameter (g>0 forward-scattering, g<0 back, g=0 isotropic).
//   HG(cosTheta, g) = (1 - g^2) / (4*pi * (1 + g^2 - 2*g*cosTheta)^1.5)
// Normalised so that integrating over the sphere yields 1; at g=0 it is the isotropic 1/(4*pi).
inline float HenyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    denom = std::max(denom, 1e-6f);
    return (1.0f - g2) / (4.0f * kPi * denom * std::sqrt(denom));
}

// Reconstruct the (un-normalised) world-space view ray for a pixel from the camera basis, EXACTLY
// like sky.frag / volumetric.frag.hlsl: the returned vector has a unit camFwd projection, so a point
// at view-linear-depth `t` along the ray is `viewPos + rayU * t`. `u,v` are the screen UV in [0,1];
// `tanHalfFovY` and `aspect` are the projection params. Y uses (-v*2+1) to match the V-down UV.
inline math::Vec3 WorldRayUnnormalized(
    float u, float v, const math::Vec3& camFwd, const math::Vec3& camRight,
    const math::Vec3& camUp, float tanHalfFovY, float aspect) {
    float ndcx = u * 2.0f - 1.0f;
    float ndcy = v * 2.0f - 1.0f;
    return math::Vec3{
        camFwd.x + camRight.x * ndcx * tanHalfFovY * aspect + camUp.x * (-ndcy) * tanHalfFovY,
        camFwd.y + camRight.y * ndcx * tanHalfFovY * aspect + camUp.y * (-ndcy) * tanHalfFovY,
        camFwd.z + camRight.z * ndcx * tanHalfFovY * aspect + camUp.z * (-ndcy) * tanHalfFovY};
}

// Beer-Lambert transmittance after travelling distance `dist` through a medium of extinction
// coefficient `sigma`. Monotonically decreasing from 1 (dist=0) toward 0.
inline float Transmittance(float dist, float sigma) {
    return std::exp(-sigma * std::max(dist, 0.0f));
}

} // namespace hf::render::volumetric
