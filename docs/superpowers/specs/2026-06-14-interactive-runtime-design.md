# Slice AA — Interactive Runtime: update loop + camera + input

Status: implementing on branch `slice-interactive`.

## Goal
Turn the static headless showcases into a live, navigable scene with a real game loop and a
flyable camera — WITHOUT breaking the headless/golden verification everything else depends on.
First slice of the EDITOR TRACK.

## Non-goals / hard constraints
- DO NOT modify the existing showcase camera blocks or their fixed cameras (13 goldens depend on
  them: scene_shadow, skinning, pbr_helmet, instanced, ibl_helmet, physics, transparency, bloom,
  scene_import, debug_viz, anim_blend, ssao, capstone). `git diff master --stat` for those code
  paths stays empty; all 13 stay DIFF 0.0000.
- RHI seam stays clean: `grep -rnE "vk[A-Z]|MTL|Metal"` over the protected dirs (incl. new
  `engine/runtime`) MUST equal 12 (master baseline). Add ZERO. SDL stays only in `engine/hal/`.

## New modules (backend-agnostic, compiled into BOTH hf_core and hf_engine)
### `engine/runtime/camera.{h,cpp}` (hf_core)
`class Camera`:
- State: `Vec3 position; float yaw, pitch; float fovY, aspect, znear, zfar;`
- Convention: **yaw** rotates around world +Y; **pitch** around the camera right axis. At
  `yaw=0, pitch=0` forward = `(0,0,-1)` (RH, looking down -Z — matches `LookAt`/the showcases).
  - `Forward() = ( cos(pitch)*sin(yaw), sin(pitch), -cos(pitch)*cos(yaw) )`
  - `Right()   = normalize(cross(Forward(), worldUp))` with worldUp = (0,1,0)
  - `Up()      = cross(Right(), Forward())`
  - Pitch clamped to ±(pi/2 - epsilon) to avoid gimbal flip.
- `Mat4 View() const` = `Mat4::LookAt(position, position + Forward(), {0,1,0})`.
- `Mat4 Proj() const` = `Mat4::Perspective(fovY, aspect, znear, zfar)` (already bakes the Vulkan
  Y-flip). Metal callers apply their existing `FlipProjY` on top, exactly as the showcases do.
- `Mat4 ViewProj() const = Proj() * View()`.
- `struct CameraBasis { Vec3 forward, right, up, position; float tanHalfFovY, aspect; }`
  returned by `Basis()` so the sample copies fwd/right/up/viewPos/skyParams straight into FrameData
  (centralizes the per-showcase duplication). This is a *helper*, not a renderer dependency.

This is pure math; unit-tested without any window/GPU.

### `engine/runtime/clock.{h,cpp}` (hf_core)
`class FixedTimestep`:
- ctor `FixedTimestep(float stepSeconds = 1.0f/120.0f, float maxStepsPerTick = 8)`.
- `int Tick(float realDtSeconds)` — accumulates realDt, returns how many fixed steps to run,
  leaving the remainder in the accumulator (canonical "Fix Your Timestep"). Clamps the number of
  steps to `maxStepsPerTick` to avoid the spiral of death (and discards the excess so we don't
  perpetually fall behind). `float Step() const` returns the fixed step size. `float Alpha()` returns
  the interpolation fraction (`accumulator / step`). Deterministic given the dt sequence.

Unit-tested without a window: 0.05s at 1/120 -> 6 steps, remainder ~ 0.05 - 6/120.

### `engine/runtime/input_state.h` (hf_core, pure struct — NO SDL)
`struct InputState`:
- `bool keyDown[Key::Count]` indexed by a small `enum Key` (W,A,S,D,Q,E,Space,Ctrl,Shift,Esc...).
- `float mouseDx, mouseDy;` accumulated relative mouse delta since the previous pump.
- `bool mouseButtons[3];` (L,R,M). `float wheel;` accumulated wheel since previous pump.
- `bool relativeMouse;` whether mouse-look (relative) mode is active.
The controller (hf_core) consumes this by const ref. The HAL fills it from SDL. Keeping the struct
in runtime/ (not hal/) lets hf_core depend on it with zero SDL.

### `engine/runtime/fly_camera_controller.{h,cpp}` (hf_core, pure C++)
`class FlyCameraController`:
- `void Update(Camera& cam, const InputState& in, float dt)`.
- WASD = move along forward/right; Space/E = up, Ctrl/Q = down (world up). Shift = sprint x4.
- mouse delta -> yaw/pitch when relative-mouse (look) is active; sensitivity configurable.
- wheel adjusts move speed (clamped to a sane range).
- Pure: takes InputState by const ref, no SDL, no GPU. Unit-testable with synthetic input.

## HAL input (SDL, hf_engine only — `engine/hal/`)
Extend `hf::hal::Window`:
- `const runtime::InputState& Input() const;` returns the current frame snapshot.
- `PumpEvents()` now also: reads `SDL_GetKeyboardState` into `keyDown[]`, accumulates relative mouse
  motion (`SDL_EVENT_MOUSE_MOTION` xrel/yrel) into mouseDx/dy, mouse-button up/down into
  mouseButtons[], wheel into `wheel`; ESC sets keyDown[Esc]. Deltas/wheel reset at the START of each
  pump so each snapshot is "since last pump".
- `void SetRelativeMouse(bool)` toggles `SDL_SetWindowRelativeMouseMode` for mouse-look.
- All SDL types stay inside window.cpp. The InputState lives in runtime/ (pure).

## Sample entry points (`samples/hello_triangle/main.cpp`)
A new self-contained block builds a scene IDENTICAL in construction to the capstone scene
(same meshes/models/physics/fox-blend/transforms/light/bloom) but driven by a `runtime::Camera`
instead of the fixed hand-built capstone camera. To avoid ANY risk to the capstone golden, the
capstone block is left byte-for-byte untouched; the new block is a parallel path. Shared helpers
(`FrameData`, `WriteBMP`, `MakeCheckerboard`) are reused. The scene-build + render-graph code is
factored into a single lambda `runInteractiveScene(...)` used by both new entries:

- `--camera-shot <yaw,pitch,x,y,z> <out.bmp>`: HEADLESS. Set the Camera to the scripted pose,
  render ONE frame, capture to BMP via the existing offscreen capture path, exit. NEW golden.
- `--fly`: open the window, build the same scene, run the real-time loop:
  pump events -> update InputState -> FixedTimestep accumulator advances animation/physics time
  (deterministic per dt) -> FlyCameraController updates the Camera from input -> rebuild FrameData
  from the Camera -> render the graph straight to the swapchain (bloom composite -> present). ESC
  quits. This is the live editor-viewport foundation.

`--fly` reuses the capstone scene's render graph but the final composite writes the swapchain and
the frame is PRESENTED (no capture). A `--fly-dry-run` hidden flag feeds N synthetic InputState
frames through one loop iteration each and exits 0, so CI can exercise the loop logic headlessly
without a GUI window.

## Metal (`metal_headless/visual_test.mm`)
Add `--camera <yaw,pitch,x,y,z> <out.png>`: construct the SAME capstone scene, set the
`runtime::Camera` to the scripted pose, apply `FlipProjY(cam.Proj())`, render one frame, write PNG.
This golden-verifies the Camera math on Metal (two-run DIFF 0.0000 -> `tests/golden/metal/camera_pose.png`).
The INTERACTIVE windowed `--fly` loop is Windows/Vulkan only this slice — windowed Metal is a known
separate gap (SDL Metal view + present loop) and is explicitly NOT attempted here.

## Verification split (honest)
- **Headlessly verifiable:** runtime unit tests (camera math, controller delta, fixed-timestep step
  count); the `--camera-shot` Vulkan PNG (visually inspected) and the Metal `camera_pose` golden
  (two-run DIFF 0.0000); `--fly` BUILDS and a synthetic-input dry-run drives one loop iteration
  without crashing; all 13 existing goldens stay DIFF 0.0000.
- **Manual only:** live keyboard/mouse navigation in the `--fly` window (WASD + mouse-look). This is
  confirmed by the user; the agent does NOT claim to have verified live input.

## Build wiring
- `engine/CMakeLists.txt`: add `runtime/camera.cpp runtime/clock.cpp runtime/fly_camera_controller.cpp`
  to BOTH `HF_ENGINE_COMMON_SOURCES` (live build) and `hf_core` (ASan/test build).
  `engine/hal/input` logic lives inside `window.cpp` (already in COMMON, NOT in hf_core), so no SDL
  leaks into hf_core.
- `tests/CMakeLists.txt`: `hf_add_pure_test(runtime_test)` (links hf_core, ASan-eligible).
