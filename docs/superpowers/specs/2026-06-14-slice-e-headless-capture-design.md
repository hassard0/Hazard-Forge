# Hazard Forge — Slice E: Headless Frame Capture (GPU Readback)

**Date:** 2026-06-14
**Status:** Self-approved (autonomous session)
**Branch:** `slice-e-headless-capture`

## Goal & motivation

Make the engine able to **render a frame and save it to an image file without a visible desktop** —
GPU image readback. This is:
1. **A real engine capability** — offscreen capture / readback underpins editor thumbnails,
   automated render tests, golden-image regression testing, and screenshot tooling.
2. **The agentic-development enabler** the user explicitly asked for: deterministic, headless,
   machine-readable verification. An AI agent (or CI) can render and inspect output with no GPU
   desktop session, no screen-scraping, no locked-session fragility.
3. **The verification unblock** — the dev machine's session locked, so desktop screen capture
   stopped working. This replaces it with something strictly better and permanent.

**Definition of Done:** `hello_triangle.exe --shot out.bmp` renders one frame of the lit cube,
reads the rendered image back from the GPU, writes a valid 32-bit BMP, and exits 0 — with NO
dependence on an unlocked/visible desktop. The controller opens the BMP and confirms the lit cube
is correct (this also retroactively verifies Slice D's lighting).

## Bold decisions (architect-of-record)

1. **BMP, no image library.** The swapchain image is BGRA8; a 32-bit BMP stores BGRA bottom-up
   natively. So we write the readback bytes almost verbatim — ~40 lines, zero dependencies, no
   conan churn. (PNG/asset export is a later concern.)
2. **Reuse the swapchain image as the capture source.** Rather than build a full offscreen
   render-target abstraction (a bigger RHI surface), add `TRANSFER_SRC` usage to swapchain images
   and copy the just-rendered image to a host buffer in `EndFrame` when capture is armed. Minimal
   new machinery, reuses the entire existing render path (pipeline, lighting, textures).
3. **`--shot` skips present.** In capture mode the frame is acquired, rendered, copied back, and
   the process exits — present is skipped (robust on a locked/non-presentable window). The window
   is still created (needed for the Vulkan surface/swapchain) but never needs to be visible.
4. **Synchronous readback.** `EndFrame` waits idle when capturing, then maps the host buffer. Fine
   for a one-shot capture / test path; not on the steady-state present path.

## RHI seam extensions (additive)

- `IRHIDevice::CaptureNextFrame()` — arms capture for the next `EndFrame`.
- `IRHIDevice::GetCapturedPixels(std::vector<uint8_t>& outBGRA, uint32_t& w, uint32_t& h) -> bool` —
  returns the most recent captured frame (BGRA8, tightly packed, top-row-first) and clears it;
  false if nothing captured.

Hard rule holds: no `vk*` in `engine/rhi/` (`std::vector`/`uint32_t` are fine — the rule is about
Vulkan symbols, not all C++).

## Vulkan implementation

- **Swapchain:** add `VK_IMAGE_USAGE_TRANSFER_SRC_BIT` to the swapchain image usage flags
  (SwapchainBuilder `add_image_usage_flags`).
- **Device:** `bool captureArmed_`, plus storage for the last capture (`std::vector<uint8_t>
  capturedBGRA_`, `uint32_t capW_, capH_`). `CaptureNextFrame()` sets the flag.
- **EndFrame (when armed):** after recording ends and before/instead of present:
  1. transition the rendered swapchain image `COLOR_ATTACHMENT_OPTIMAL → TRANSFER_SRC_OPTIMAL`
     (sync2: src COLOR_ATTACHMENT_OUTPUT/COLOR_ATTACHMENT_WRITE → dst COPY/TRANSFER_READ).
  2. `vkCmdCopyImageToBuffer` into a host-visible staging buffer (VMA, size = w*h*4).
  3. end + submit (`vkQueueSubmit2`), then `vkQueueWaitIdle`.
  4. map staging → copy into `capturedBGRA_` (row-major, top-first; swapchain rows are already
     top-first), set capW_/capH_, free staging, clear `captureArmed_`.
  5. **Skip present** in capture mode (return without `vkQueuePresentKHR`); still advance frame
     index.
- `GetCapturedPixels` moves `capturedBGRA_`/dims out and resets them.

> Note: copying directly from the swapchain image requires it to be in a copyable layout; the
> normal path leaves it in PRESENT_SRC. The capture path transitions COLOR→TRANSFER_SRC instead of
> COLOR→PRESENT. Keep the two paths cleanly separated by the armed flag.

## Sample (`samples/hello_triangle/main.cpp`)

- Parse argv: if `--shot <path>` present, run **capture mode**; else the normal interactive loop.
- Capture mode: create window+device+pipeline+texture+buffers as usual; set frame uniforms; render
  ONE frame with a fixed model rotation (e.g. `t = 0.6f` so the cube shows three faces clearly);
  `device->CaptureNextFrame()` before `BeginFrame`; record the lit cube; `EndFrame`;
  `GetCapturedPixels`; write a 32-bit BMP to `<path>` (BGRA bottom-up: BMP is bottom-up, captured
  is top-first, so write rows in reverse); exit 0. Print the path + dimensions.
- BMP writer: a small free function `WriteBMP(path, bgra, w, h)` in the sample (BITMAPFILEHEADER +
  BITMAPINFOHEADER, 32bpp BI_RGB, rows bottom-up).

## Testing

- `rhi_smoke` + `math_test` unchanged, still pass.
- New `capture_smoke` is NOT added as a GPU CTest (capture needs the full swapchain frame loop;
  rhi_smoke already covers device init). Instead the controller runs `--shot` and inspects the BMP
  — that IS the test for this slice.
- Manual/agentic gate: `hello_triangle.exe --shot out.bmp` exits 0 and produces a valid non-empty
  BMP showing the lit cube.

## Out of scope
Full offscreen render-target abstraction, PNG/JPEG export, multi-frame capture / video, depth
readback, async readback. One armed swapchain readback → one BMP.

## Risk
- Window creation on a locked session: window creation (not presentation) generally works; capture
  skips present, so a non-presentable window shouldn't block. If `vkAcquireNextImageKHR` fails on a
  locked session, fall back / report — the controller will find out at run time.
