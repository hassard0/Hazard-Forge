# Slice AN — Mac windowed Metal interactive path

Date: 2026-06-15
Branch: `slice-mac-window`
Status: design self-approved (autonomy delegate), implementation in progress

## Goal

Bring the LIVE interactive `--fly` viewport to macOS/Metal, completing cross-platform
parity for the editor. On Windows the interactive viewport runs on Vulkan
(`hello_triangle.exe --fly`). On the Mac the engine has so far only rendered HEADLESS
(offscreen `MTLTexture` -> PNG golden, via the `metal_headless/visual_test` target).
The one remaining gap (noted across the project memory) is a real on-screen Metal window
that pumps input, flies the camera, and presents to a `CAMetalLayer` each frame.

## Hard constraints

- RHI seam: Metal symbols ONLY in `engine/rhi_metal/`.
- The 22 image goldens + the JSON golden stay byte-untouched.
- Do NOT alter the headless `metal_headless` / `visual_test` golden path or any rendering shader.
- Additive only.
- A live GUI window CANNOT be verified over SSH (no Mac display session). The deliverable
  is a clean BUILD + LINK on the Mac; the window-opens-and-flies check is the user's, by hand
  on the Mac Mini's display.

## Window-path decision: native Cocoa/AppKit (NOT SDL)

Two options were on the table:

- (a) Build the full `hello_triangle` SDL sample for macOS with the Metal backend.
- (b) A minimal native Cocoa/AppKit window (`NSWindow` + `CAMetalLayer` view), no SDL,
      driving the same engine RHI/scene/camera/runtime loop.

**Chosen: (b) native Cocoa.** The decision is FORCED by the Mac environment, verified live
over SSH on the Mac Mini (M4, macOS 15.6, Command Line Tools only):

```
pkg-config --exists sdl3        -> NO
ls /opt/homebrew/lib/libSDL3*   -> no matches
xcodebuild -version             -> CLT only, no Xcode
which cmake ninja clang glslc spirv-cross -> all present
```

There is no SDL3 on the Mac (no library, no pkg-config, no conan). The whole windowed
path in `engine/hal/window.cpp` and `engine/rhi_metal/metal_device_windowed.mm` is
SDL-based (`SDL_Metal_CreateView` / `SDL_Metal_GetLayer`), so option (a) cannot build there
without installing SDL3 — which the project deliberately avoids on the Mac (the
`metal_headless` target is conan-free / SDL-free by design). Additionally the
`hello_triangle` sample links `hf_editor` (Vulkan/ImGui-only, gated `NOT APPLE`), uses the
DXC `hf_compile_shaders` toolchain (not available on Apple), and hardcodes
`Backend::Vulkan` with a SPIR-V shader path Metal explicitly rejects. So option (a) would be
a large, fragile rewrite.

Option (b) mirrors the proven `metal_headless` pattern exactly: a standalone, conan-free,
SDL-free CMake target (`mac_window/`) that:
- generates MSL from the SHARED HLSL via the SAME `glslc -> spirv-cross` chain (identical
  shader bytes as the headless golden path — no shader edits),
- builds the REAL Metal RHI classes,
- opens a native `NSWindow` whose `contentView` is layer-backed by a `CAMetalLayer`,
- drives the SAME backend-agnostic `runtime::Camera` + `FlyCameraController` +
  `FixedTimestep` + the SAME Slice-F scene (`RunSceneShowcase` construction) through the
  SAME `render::RenderGraph`,
- maps AppKit `NSEvent` (keyDown/keyUp/mouseDragged/scrollWheel) into the SAME
  `runtime::InputState` the SDL HAL fills on Windows.

## RHI: windowed Metal device WITHOUT hal::Window

The existing windowed ctor `MetalDevice(hal::Window&)` (in `metal_device_windowed.mm`) and
the `rhi_factory` path both pull in `hal/window.h` (SDL). The Cocoa target has no SDL, so it
cannot use that ctor.

Addition (engine/rhi_metal/, seam-clean): a new ctor `MetalDevice(void* caMetalLayer, w, h)`
plus a Metal-free factory `CreateMetalDeviceWindowedLayer(void* layer, w, h)` declared in a
new header `metal_windowed.h` (mirrors `metal_offscreen.h`'s `CreateMetalDeviceHeadless`).
The `mac_window` entry hands it the `CAMetalLayer*` (as `void*`) it created on the NSView.
This keeps ALL Metal/Obj-C in `engine/rhi_metal/`; the new entry only passes a `void*`.

The windowed present path itself is ALREADY complete in `metal_device.mm`:
`BeginFrame()` acquires `[layer nextDrawable]`, `EndFrame()` does `[cmd presentDrawable:]` +
an `addCompletedHandler` that signals the in-flight semaphore, then `commit`. The
`RenderGraph`'s `Swapchain` pass already calls `BeginFrame`/`EndFrame`, so
`graph.Execute()` per frame presents automatically. No present logic is invented here.

## Render loop (mac_window)

```
init: gen MSL, create windowed MetalDevice from the CAMetalLayer, build the Slice-F scene
      + render graph (shadow -> scene RT -> post-to-swapchain), Camera + FlyCameraController.
each frame (CVDisplayLink / NSTimer driven, or a tight runloop):
  pump NSEvent -> fill runtime::InputState (WASD/QE/Space/Ctrl/Shift, right-drag look, wheel)
  dt -> FixedTimestep::Tick (the fly controller is frame-rate independent here; we advance the
        camera by the real dt once per frame, matching --fly's camera-update cadence)
  controller.Update(camera, input, dt)
  recompute fd.vp / camera basis from the camera
  graph.Execute(device)   // shadow + scene RT + post pass -> BeginFrame/EndFrame present
  ESC -> quit
```

Resize: on `windowDidResize`, update the `CAMetalLayer.drawableSize` and call
`device->Swapchain().Recreate(w,h)`. (First cut keeps the offscreen render-target + shadow
map at their initial size for simplicity; the post pass samples the RT and blits to the
resized drawable. A full RT-resize is a follow-up; the window still presents correctly.)

## Files

- `engine/rhi_metal/metal_windowed.h` (NEW) — Metal-free `CreateMetalDeviceWindowedLayer`.
- `engine/rhi_metal/metal_device_windowed.mm` (EDIT, additive) — new `MetalDevice(void*,w,h)`
  ctor + the `CreateMetalDeviceWindowedLayer` factory. The existing `MetalDevice(Window&)`
  ctor + `CreateMetalDevice(Window&)` factory are guarded so they only compile when
  `hal/window.h` (SDL) is available (`HF_HAVE_SDL_WINDOW`), so this file can ALSO build in the
  SDL-free `mac_window` target.
- `engine/rhi_metal/metal_device.h` (EDIT, additive) — declare the new ctor + a
  `ResizeLayer` helper note; no behavior change to existing ctors.
- `mac_window/CMakeLists.txt` (NEW) — standalone conan-free/SDL-free target `mac_window`,
  same MSL-gen + RHI source list as `metal_headless`, + AppKit/Cocoa frameworks.
- `mac_window/main.mm` (NEW) — the NSWindow + CAMetalLayer + NSEvent input + interactive loop.

## Verification (honest)

- Mac: `cmake -S mac_window -B build-mac-window -G Ninja && cmake --build build-mac-window`
  must COMPILE + LINK and produce the `mac_window` binary. THIS is the achievable bar over SSH.
- The live window opening / rendering / flying is NOT verifiable over SSH (no display session).
  The user runs the binary on the Mac Mini's physical display to confirm.
- Windows/Vulkan untouched: `ctest --preset windows-msvc-debug` 22/22; goldens byte-identical
  (`git diff master --stat -- tests/golden` empty); seam audit (no Metal symbols outside
  `engine/rhi_metal/`).
