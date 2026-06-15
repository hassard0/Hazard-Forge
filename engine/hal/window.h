#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "runtime/input_state.h"

struct SDL_Window;
struct VkInstance_T;  using VkInstance = VkInstance_T*;
struct VkSurfaceKHR_T; using VkSurfaceKHR = VkSurfaceKHR_T*;

namespace hf::hal {

struct WindowConfig {
    std::string title = "Hazard Forge";
    int width = 1280;
    int height = 720;
};

// Owns an SDL3 window with the Vulkan flag set. Pumps OS events.
class Window {
public:
    explicit Window(const WindowConfig& cfg);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Process pending OS events. Returns false when the user requested quit.
    bool PumpEvents();

    // True for exactly one PumpEvents cycle after a size/resize event.
    bool ConsumeResized();

    // Input snapshot for the frame most recently pumped: level keyboard/mouse-button state plus the
    // relative mouse-motion and wheel ACCUMULATED since the previous PumpEvents. Backend-agnostic
    // (runtime::InputState carries no SDL types); the fly-camera controller consumes it directly.
    const runtime::InputState& Input() const { return input_; }

    // Toggle relative ("captured") mouse mode for mouse-look. While on, the OS cursor is hidden and
    // motion is reported as deltas. Reflected in Input().relativeMouse.
    void SetRelativeMouse(bool enabled);

    // Vulkan instance extensions SDL requires for this window's surface.
    std::vector<const char*> RequiredVulkanInstanceExtensions() const;

    // Create a Vulkan surface for this window on the given instance.
    VkSurfaceKHR CreateVulkanSurface(VkInstance instance) const;

#ifdef __APPLE__
    // Create (once) an SDL Metal view over this window and return its CAMetalLayer as a void*.
    // Caller (MetalDevice) casts to CAMetalLayer*. Returns the same layer on repeat calls.
    // Kept as void* so this header stays free of Metal/QuartzCore types.
    void* CreateMetalLayer();
#endif

    int FramebufferWidth() const;
    int FramebufferHeight() const;
    SDL_Window* Raw() const { return window_; }

private:
    SDL_Window* window_ = nullptr;
    bool resized_ = false;
    runtime::InputState input_;  // filled from SDL each PumpEvents (no SDL types escape this header)
#ifdef __APPLE__
    void* metalView_ = nullptr;  // SDL_MetalView (opaque); created lazily by CreateMetalLayer()
#endif
};

} // namespace hf::hal
