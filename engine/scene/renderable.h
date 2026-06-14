#pragma once
#include "scene/mesh.h"
#include "scene/transform.h"
#include "rhi/rhi.h"
namespace hf::scene {
// metallic/roughness are the per-object PBR material, pushed (after the model matrix) into the
// lit pipeline's push constant. Defaults: dielectric, medium roughness.
// normalMap is an optional tangent-space normal map (RGBA8, 0..1-encoded). When null the renderer
// binds a flat default normal (0.5,0.5,1) so the lit shader's TBN perturbation is a no-op.
struct Renderable { Mesh* mesh; rhi::ITexture* texture; Transform transform;
                    float metallic = 0.0f; float roughness = 0.5f;
                    rhi::ITexture* normalMap = nullptr; };
} // namespace
