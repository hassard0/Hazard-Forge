// FixedTimestep is header-only (engine/runtime/clock.h) — pure inline accumulator. This TU anchors
// a compiled object for the class in both hf_core and hf_engine. Intentionally minimal.
#include "runtime/clock.h"

namespace hf::runtime {

// Anchor symbol so the translation unit is never empty across toolchains.
FixedTimestep MakeDefaultTimestep() { return FixedTimestep{}; }

} // namespace hf::runtime
