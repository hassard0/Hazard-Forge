#pragma once
// Hazard Forge ImGui config. Passed to ImGui via -DIMGUI_USER_CONFIG.
//
// Force 32-bit indices so ImDrawData index buffers can be bound directly through the RHI, which
// binds VK_INDEX_TYPE_UINT32 / MTLIndexTypeUInt32. (ImGui defaults to 16-bit ImDrawIdx.)
#define ImDrawIdx unsigned int
