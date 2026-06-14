#include "hal/window.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <stdexcept>

namespace hf::hal {

Window::Window(const WindowConfig& cfg) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
    }
    window_ = SDL_CreateWindow(cfg.title.c_str(), cfg.width, cfg.height,
                               SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window_) {
        throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
    }
}

Window::~Window() {
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

bool Window::PumpEvents() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT) return false;
        if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED ||
            e.type == SDL_EVENT_WINDOW_RESIZED) {
            resized_ = true;
        }
    }
    return true;
}

bool Window::ConsumeResized() {
    bool r = resized_;
    resized_ = false;
    return r;
}

std::vector<const char*> Window::RequiredVulkanInstanceExtensions() const {
    Uint32 count = 0;
    const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);
    if (!names) {
        throw std::runtime_error(std::string("SDL_Vulkan_GetInstanceExtensions failed: ") +
                                 SDL_GetError());
    }
    return std::vector<const char*>(names, names + count);
}

VkSurfaceKHR Window::CreateVulkanSurface(VkInstance instance) const {
    VkSurfaceKHR surface = nullptr;
    if (!SDL_Vulkan_CreateSurface(window_, instance, nullptr, &surface)) {
        throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface failed: ") +
                                 SDL_GetError());
    }
    return surface;
}

int Window::FramebufferWidth() const {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    return w;
}

int Window::FramebufferHeight() const {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    return h;
}

} // namespace hf::hal
