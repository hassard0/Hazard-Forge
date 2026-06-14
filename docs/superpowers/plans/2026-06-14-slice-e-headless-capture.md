# Slice E: Headless Frame Capture — Implementation Plan

> subagent-driven. Toolchain proven. Env facts below.

**Goal:** `hello_triangle.exe --shot out.bmp` renders one lit-cube frame, reads it back from the
GPU, writes a 32-bit BMP, exits 0 — no visible desktop needed. Verifies on a locked session.

**Builds on:** Slices A–D on `master`. Additive.

---

## Environment (proven — trust)
- VS BuildTools x64 dev shell: `& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64`, then cmake in that session.
- Conan installed. `cmake --preset windows-msvc-debug` then `cmake --build --preset windows-msvc-debug`.
- DO NOT run the interactive `hello_triangle.exe` with no args (infinite loop). You MAY run `hello_triangle.exe --shot <tmp>.bmp` (it renders one frame and EXITS) to self-verify — do so and confirm exit 0 + a non-empty BMP.
- Gates: ctest (rhi_smoke + math_test) still pass; plus the `--shot` self-check.
- READ first: engine/rhi/rhi.h, engine/rhi_vulkan/{vulkan_device.h,.cpp, vulkan_swapchain.h,.cpp, vulkan_command_buffer.cpp}, samples/hello_triangle/main.cpp. The EndFrame present/transition code and the swapchain Build() are the spots you touch.

## Existing facts
- Vulkan 1.3 sync2. EndFrame currently: transition COLOR_ATTACHMENT→PRESENT_SRC, vkEndCommandBuffer, vkQueueSubmit2 (signals fence + renderFinished sem), vkQueuePresentKHR, advance frameIndex_. There's a TransitionImage sync2 helper. The swapchain exposes image(i)/view(i)/extent()/imageCount() and is built by vkb::SwapchainBuilder in VulkanSwapchain::Build with add_image_usage_flags(COLOR_ATTACHMENT_BIT).
- acquiredImage_ holds the current swapchain image index during a frame.
- HARD RULE: no vulkan symbols in engine/rhi/. CaptureNextFrame/GetCapturedPixels use std::vector/uint32_t only.
- Adding pure-virtuals makes VulkanDevice abstract — implement all together, build once.

---

## Task E1: RHI seam
**File:** engine/rhi/rhi.h. Add to `IRHIDevice`:
```cpp
    // Headless capture: arm before BeginFrame; after EndFrame, retrieve via GetCapturedPixels.
    virtual void CaptureNextFrame() = 0;
    // Returns the last captured frame as tightly-packed BGRA8 (top row first); false if none.
    virtual bool GetCapturedPixels(std::vector<uint8_t>& outBGRA,
                                   uint32_t& width, uint32_t& height) = 0;
```
(`<vector>` and `<cstdint>` are already included by rhi.h.)

## Task E2: Swapchain — TRANSFER_SRC usage
**File:** engine/rhi_vulkan/vulkan_swapchain.cpp. In Build(), add `TRANSFER_SRC` to the usage:
change `.add_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)` to also include
`VK_IMAGE_USAGE_TRANSFER_SRC_BIT` (call add_image_usage_flags twice or OR them). This lets us copy
the rendered image.

## Task E3: Device capture path
**Files:** vulkan_device.h, vulkan_device.cpp.
- [ ] Members: `bool captureArmed_ = false; std::vector<uint8_t> capturedBGRA_; uint32_t capW_ = 0, capH_ = 0;`
- [ ] `void CaptureNextFrame() override { captureArmed_ = true; }`
- [ ] `bool GetCapturedPixels(std::vector<uint8_t>& out, uint32_t& w, uint32_t& h) override`:
  if `capturedBGRA_.empty()` return false; `out = std::move(capturedBGRA_); w = capW_; h = capH_;`
  `capturedBGRA_.clear(); capW_ = capH_ = 0;` return true.
- [ ] **EndFrame branch when `captureArmed_`:** instead of the normal COLOR→PRESENT transition +
  submit + present, do:
  1. transition the current swapchain image (`swapchain_->image(acquiredImage_)`)
     COLOR_ATTACHMENT_OPTIMAL → TRANSFER_SRC_OPTIMAL via the sync2 helper
     (src COLOR_ATTACHMENT_OUTPUT/COLOR_ATTACHMENT_WRITE → dst COPY_BIT/TRANSFER_READ, aspect COLOR).
  2. Create a host-visible VMA staging buffer of size `extent.width*extent.height*4`
     (`TRANSFER_DST`, HOST_ACCESS_RANDOM + MAPPED, HOST_COHERENT required).
  3. `vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region)`
     with `region.imageSubresource={COLOR,0,0,1}`, `imageExtent={w,h,1}`, bufferOffset 0,
     bufferRowLength 0, bufferImageHeight 0.
  4. `vkEndCommandBuffer`; submit via `vkQueueSubmit2` (wait on `imageAvailable` like the normal
     path, signal the in-flight fence; renderFinished sem can be signaled but is unused — simplest:
     reuse the normal submit's wait semaphore so acquire is respected). Then `vkQueueWaitIdle(graphicsQueue_)`.
  5. Map staging (`info.pMappedData`), copy `w*h*4` bytes into `capturedBGRA_` (resize first), set
     capW_/capH_. Destroy staging buffer.
  6. **Do NOT present.** Advance `frameIndex_` and clear `captureArmed_`. Return.
  Keep the normal (non-capture) path exactly as-is.
- [ ] Ensure `#include <vector>` / `<cstdint>` present in vulkan_device.h as needed.

> Note the submit in capture mode must still wait on `imageAvailable` (acquired in BeginFrame) so
> the copy happens after rendering. Use the same per-frame fence so the next BeginFrame's
> vkWaitForFences stays correct. Since you skip present, the renderFinished semaphore is left
> unsignaled-and-unwaited — that's fine (do not signal a semaphore nobody waits on if validation
> complains; simplest is to not include renderFinished in the capture submit).

## Task E4: Sample `--shot` mode + BMP writer
**File:** samples/hello_triangle/main.cpp.
- [ ] Add `int main(int argc, char** argv)`; scan for `--shot <path>`.
- [ ] BMP writer (free function):
```cpp
#include <cstdint>
#include <cstdio>
#include <vector>
static bool WriteBMP(const char* path, const std::vector<uint8_t>& bgra,
                     uint32_t w, uint32_t h) {
    // 32bpp BI_RGB, bottom-up. Captured data is top-row-first BGRA; BMP wants bottom-up.
    uint32_t imgSize = w * h * 4;
    uint32_t fileSize = 54 + imgSize;
    uint8_t fh[14] = {0}; uint8_t ih[40] = {0};
    fh[0]='B'; fh[1]='M';
    fh[2]=fileSize; fh[3]=fileSize>>8; fh[4]=fileSize>>16; fh[5]=fileSize>>24;
    fh[10]=54;
    ih[0]=40;
    ih[4]=w; ih[5]=w>>8; ih[6]=w>>16; ih[7]=w>>24;
    ih[8]=h; ih[9]=h>>8; ih[10]=h>>16; ih[11]=h>>24;
    ih[12]=1; ih[14]=32;            // planes=1, bpp=32
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fwrite(fh, 1, 14, f); std::fwrite(ih, 1, 40, f);
    for (int y = (int)h - 1; y >= 0; --y)
        std::fwrite(&bgra[(size_t)y * w * 4], 1, w * 4, f);
    std::fclose(f);
    return true;
}
```
- [ ] Capture mode flow: build window/device/pipeline/texture/vertex+index buffers exactly like the
  normal path. Set frame uniforms (fixed camera + light). Use a FIXED `t = 0.6f` for the model
  rotation (shows three faces). Then:
  `device->CaptureNextFrame(); auto fc = device->BeginFrame();` — if `fc.cmd` is null, retry once
  (a fresh swapchain may report out-of-date on first acquire). Record BeginRenderPass/BindPipeline/
  PushConstants(model)/BindTexture/BindVertexBuffer/BindIndexBuffer/DrawIndexed(36)/EndRenderPass.
  `device->EndFrame(fc);`
  `std::vector<uint8_t> px; uint32_t w,h; if (device->GetCapturedPixels(px,w,h)) WriteBMP(path,px,w,h);`
  `device->WaitIdle();` print `"wrote <path> (WxH)"`, return 0. On failure print to stderr, return 1.
- [ ] Keep the interactive loop for the no-arg case unchanged. Title unchanged.

## Verify
- [ ] Build clean.
- [ ] `ctest --preset windows-msvc-debug` — rhi_smoke + math_test pass.
- [ ] Run `hello_triangle.exe --shot %TEMP%\hf_e_selftest.bmp` — exits 0, prints the path+dims, and
  the BMP file exists and is > 54 bytes (header + pixels). (This is allowed — it renders one frame
  and exits; it does NOT loop.)
- [ ] grep: no vulkan symbols in engine/rhi/.

## Commit (footer: Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>)
`feat(rhi+vulkan): headless frame capture — GPU readback to BMP via --shot`
No build/ artifacts.

## DoD
- [ ] `--shot out.bmp` produces a valid non-empty 32-bit BMP and exits 0, with no dependence on a
  visible desktop.
- [ ] ctest green; seam rule held.
