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

// Map an SDL scancode to our backend-agnostic runtime::Key, or Key::Count if unmapped.
static runtime::Key MapScancode(SDL_Scancode sc) {
    using K = runtime::Key;
    switch (sc) {
        case SDL_SCANCODE_W: return K::W;
        case SDL_SCANCODE_A: return K::A;
        case SDL_SCANCODE_S: return K::S;
        case SDL_SCANCODE_D: return K::D;
        case SDL_SCANCODE_Q: return K::Q;
        case SDL_SCANCODE_E: return K::E;
        case SDL_SCANCODE_SPACE: return K::Space;
        case SDL_SCANCODE_LCTRL:
        case SDL_SCANCODE_RCTRL: return K::Ctrl;
        case SDL_SCANCODE_LSHIFT:
        case SDL_SCANCODE_RSHIFT: return K::Shift;
        case SDL_SCANCODE_ESCAPE: return K::Esc;
        case SDL_SCANCODE_P: return K::P;   // editor: play/pause toggle
        case SDL_SCANCODE_O: return K::O;   // editor: single step
        case SDL_SCANCODE_G: return K::G;   // editor: gizmo Translate
        case SDL_SCANCODE_R: return K::R;   // editor: gizmo Rotate
        case SDL_SCANCODE_T: return K::T;   // editor: gizmo Scale
        default: return K::Count;
    }
}

bool Window::PumpEvents() {
    // Deltas/wheel are "since last pump": reset at the start of each pump and accumulate below.
    input_.mouseDx = 0.0f;
    input_.mouseDy = 0.0f;
    input_.wheel = 0.0f;

    bool quit = false;
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT) quit = true;
        if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED ||
            e.type == SDL_EVENT_WINDOW_RESIZED) {
            resized_ = true;
        }
        if (e.type == SDL_EVENT_MOUSE_MOTION) {
            input_.mouseDx += e.motion.xrel;
            input_.mouseDy += e.motion.yrel;
        }
        if (e.type == SDL_EVENT_MOUSE_WHEEL) {
            input_.wheel += e.wheel.y;
        }
        if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN || e.type == SDL_EVENT_MOUSE_BUTTON_UP) {
            bool down = (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
            if (e.button.button == SDL_BUTTON_LEFT)   input_.mouseButtons[0] = down;
            if (e.button.button == SDL_BUTTON_RIGHT)  input_.mouseButtons[1] = down;
            if (e.button.button == SDL_BUTTON_MIDDLE) input_.mouseButtons[2] = down;
        }
    }

    // Level keyboard state (independent of event ordering) into the snapshot.
    int numKeys = 0;
    const bool* ks = SDL_GetKeyboardState(&numKeys);
    for (int i = 0; i < static_cast<int>(runtime::Key::Count); ++i)
        input_.keyDown[i] = false;
    if (ks) {
        for (int sc = 0; sc < numKeys; ++sc) {
            if (!ks[sc]) continue;
            runtime::Key k = MapScancode(static_cast<SDL_Scancode>(sc));
            if (k != runtime::Key::Count)
                input_.keyDown[static_cast<int>(k)] = true;
        }
    }

    return !quit;
}

void Window::SetRelativeMouse(bool enabled) {
    SDL_SetWindowRelativeMouseMode(window_, enabled);
    input_.relativeMouse = enabled;
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
