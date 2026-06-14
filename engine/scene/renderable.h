#pragma once
#include "scene/mesh.h"
#include "scene/transform.h"
#include "rhi/rhi.h"
namespace hf::scene {
struct Renderable { Mesh* mesh; rhi::ITexture* texture; Transform transform; };
} // namespace
