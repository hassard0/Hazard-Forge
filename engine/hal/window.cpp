#include "hal/window.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#ifdef __APPLE__
#include <SDL3/SDL_metal.h>
#endif
#include <stdexcept>

namespace hf::hal {

Window::Window(const WindowConfig& cfg) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
    }
#ifdef __APPLE__
    // Metal-backed window on Apple; the Vulkan flag would force the Vulkan loader.
    const SDL_WindowFlags gpuFlag = SDL_WINDOW_METAL;
#else
    const SDL_WindowFlags gpuFlag = SDL_WINDOW_VULKAN;
#endif
    window_ = SDL_CreateWindow(cfg.title.c_str(), cfg.width, cfg.height,
                               gpuFlag | SDL_WINDOW_RESIZABLE);
    if (!window_) {
        std::string err = std::string("SDL_CreateWindow failed: ") + SDL_GetError();
        SDL_Quit();  // constructor failed: dtor won't run, so undo SDL_Init here
        throw std::runtime_error(err);
    }
}

Window::~Window() {
#ifdef __APPLE__
    if (metalView_) SDL_Metal_DestroyView(static_cast<SDL_MetalView>(metalView_));
#endif
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

#ifdef __APPLE__
void* Window::CreateMetalLayer() {
    if (!metalView_) {
        metalView_ = SDL_Metal_CreateView(window_);
        if (!metalView_) {
            throw std::runtime_error(std::string("SDL_Metal_CreateView failed: ") +
                                     SDL_GetError());
        }
    }
    // Returns a CAMetalLayer* (as void*); MetalDevice casts it back.
    void* layer = SDL_Metal_GetLayer(static_cast<SDL_MetalView>(metalView_));
    if (!layer) {
        throw std::runtime_error(std::string("SDL_Metal_GetLayer failed: ") + SDL_GetError());
    }
    return layer;
}
#endif

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
