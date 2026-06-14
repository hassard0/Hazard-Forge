#pragma once
#include <memory>
#include "rhi/rhi.h"

namespace hf::hal { class Window; }

namespace hf::rhi {

// Create a backend device bound to the given window.
std::unique_ptr<IRHIDevice> CreateDevice(Backend backend, hf::hal::Window& window);

} // namespace hf::rhi
