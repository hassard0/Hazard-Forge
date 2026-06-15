// Camera is header-only (engine/runtime/camera.h) — pure inline math. This TU exists so the camera
// has a compiled object in both hf_core and hf_engine (and a stable place for any future non-inline
// helpers). Intentionally minimal.
#include "runtime/camera.h"

namespace hf::runtime {

// Anchor symbol so the translation unit is never empty across toolchains.
const Camera kDefaultCamera{};

} // namespace hf::runtime
