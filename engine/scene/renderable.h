#pragma once
#include "scene/mesh.h"
#include "scene/transform.h"
#include "rhi/rhi.h"
namespace hf::scene {
// metallic/roughness are the per-object PBR material, pushed (after the model matrix) into the
// lit pipeline's push constant. Defaults: dielectric, medium roughness.
struct Renderable { Mesh* mesh; rhi::ITexture* texture; Transform transform;
                    float metallic = 0.0f; float roughness = 0.5f; };
} // namespace
