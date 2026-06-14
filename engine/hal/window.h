#pragma once
#include <cstdint>
#include <string>
#include <vector>

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

    // Vulkan instance extensions SDL requires for this window's surface.
    std::vector<const char*> RequiredVulkanInstanceExtensions() const;

    // Create a Vulkan surface for this window on the given instance.
    VkSurfaceKHR CreateVulkanSurface(VkInstance instance) const;

    int FramebufferWidth() const;
    int FramebufferHeight() const;
    SDL_Window* Raw() const { return window_; }

private:
    SDL_Window* window_ = nullptr;
    bool resized_ = false;
};

} // namespace hf::hal
