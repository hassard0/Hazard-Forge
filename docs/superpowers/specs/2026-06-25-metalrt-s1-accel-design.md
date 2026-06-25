# Slice METAL-RT S1 — RHI acceleration-structure factory on Metal (issues #42/#35, beachhead)

The beachhead of REAL Metal hardware ray tracing in the RHI. The scout verdict is GO: the hard part is
already proven — `shaders/rt_query.metal` (native MSL `intersection_query`, fx int64 via MSL `long`,
sidestepping the glslc int64→MSL blocker) already builds an `MTLAccelerationStructure` headless on the M4
and matches the CPU integer reference BYTE-EQUAL (`--rt2-query-hw`, `metal_headless/visual_test.mm:25556-25734`).
That code lives in the SHOWCASE, doing raw `MTL*` calls. S1 lifts it into `engine/rhi_metal/` so Metal
implements the existing (Vulkan-only-today) `IAccelStructure` RHI seam — turning `SupportsHardwareRayQuery()`
true on Apple and routing the RT through `device->CreateBlas/CreateTlas` + `cmd->BindAccelStructure`.

Determinism is preserved EXACTLY (the engine's brand): the float HW BVH is a candidate generator only
(every AABB inflated `kRtAabbMargin = 1/64` so the driver's float overlap is a strict superset of every
true fx hit), the kernel never commits/narrows, and correctness is the integer `(t, primIndex)` Q16.16
reduction — bit-identical CPU/Vulkan/Metal. So the golden stays a **strict memcmp byte-equal** to the CPU
reference, not a tolerance.

## Execution model (READ THIS — Metal compiles ONLY on the M4)
The implementer (you) WRITES the `engine/rhi_metal/` Metal/Obj-C++ code on Windows by **lifting from the
proven RT2b showcase + mirroring the Vulkan reference** — it will NOT compile on Windows (no Metal SDK),
that is EXPECTED. Commit the draft to the branch; the CONTROLLER does the Mac compile/test round-trips
(scp → ssh build `metal_headless` → read MSL/Metal errors → fix → re-test) and the byte-equal validation.
So: be accurate, cite the proven sources, but do NOT block on compiling.

## Sources to lift from (READ these)
- `metal_headless/visual_test.mm:25556-25734` — `RunRt2QueryHwShowcase`: the PROVEN headless accel-build +
  dispatch + memcmp-vs-CPU. Lines :25652-25670 = the primitive accel build
  (`MTLPrimitiveAccelerationStructureDescriptor` of margin-inflated bboxes:
  `accelerationStructureSizesWithDescriptor` → `newAccelerationStructureWithSize` →
  `accelerationStructureCommandEncoder` → `buildAccelerationStructure` → `commit`/`waitUntilCompleted`);
  :25708-25709 = `setAccelerationStructure:atBufferIndex:` + `useResource:`; :25615 = `dev.supportsRaytracing`
  gate; :25624 = the `kRtAabbMargin` inflation.
- `engine/rhi_vulkan/vulkan_accel.cpp` (+ `vulkan_accel.h`) — the reference BLAS/TLAS two-level shape the
  Metal side must match (BLAS @ ~68, TLAS @ ~165, `kRtAabbMargin` @ vulkan_accel.h:23).
- `engine/rhi/rhi.h:250-277` (the seam: `IAccelStructure`, `AccelGeometry`/`BlasDesc`/`TlasInstance`/
  `TlasDesc`), `:558-560` (`CreateBlas`/`CreateTlas`/`SupportsHardwareRayQuery` virtuals), `:400`
  (`BindAccelStructure`). `:202` (`ComputePipelineDesc::accelStructureBinding`).
- `shaders/rt_query.metal` — the native MSL kernel (reusable; note it uses `primitive_acceleration_structure`
  at buffer(3), :180). `engine/rhi_metal/metal_device.{h,mm}` (MetalDevice `final : public IRHIDevice`,
  device.h:24; CreateBuffer/CreateComputePipeline at device.mm:72/76 — add overrides nearby) and
  `engine/rhi_metal/metal_command_buffer.{h,mm}` (MetalCommandBuffer @ command_buffer.h:13).

## What to build
1. **NEW `engine/rhi_metal/metal_accel.h` / `metal_accel.mm`** — `class MetalAccelStructure final : public
   rhi::IAccelStructure` holding `id<MTLAccelerationStructure> as_` (+ the retained input buffers: the
   bbox buffer for a BLAS, the instance-descriptor buffer + the child BLAS pointers for a TLAS). Expose
   `id<MTLAccelerationStructure> Handle() const`.
2. **`MetalDevice::CreateBlas(const BlasDesc&)` override** — for each `AabbProcedural` geom, append an
   `MTLAxisAlignedBoundingBox{lo, hi}` (host-side, ALREADY inflated by the caller as the Vulkan path is —
   confirm the caller inflates; if not, mirror `kRtAabbMargin` here exactly as the showcase does at
   :25624). Build one `MTLAccelerationStructureBoundingBoxGeometryDescriptor` over the bbox buffer →
   `MTLPrimitiveAccelerationStructureDescriptor` → size → `newAccelerationStructureWithSize` → encode build
   on a command buffer → `waitUntilCompleted`. (Lift :25652-25670 verbatim; this is the proven path.)
   Return a `MetalAccelStructure`.
3. **`MetalDevice::CreateTlas(const TlasDesc&)` override** — build an `MTLInstanceAccelerationStructureDescriptor`:
   one `MTLAccelerationStructureInstanceDescriptor` per `TlasInstance` (its `accelerationStructureIndex`
   into the `instancedAccelerationStructures` array of the child BLAS handles, the row-major `transform[12]`
   copied into `MTLPackedFloat4x3`, `userID = instanceId`). Build + wait. Return a `MetalAccelStructure`
   wrapping the instance AS (retaining the child BLAS handles so they outlive the TLAS).
   - **CRUX (the one real Metal subtlety):** `rt_query.metal` currently declares
     `primitive_acceleration_structure` (a BLAS), but a TLAS is an `instance_acceleration_structure`. To
     drive the proper two-level RHI shape through the kernel, the kernel's accel param must become
     `instance_acceleration_structure` and the hit read its `primitive_id` (and the instance's `user_id`
     if needed). **For the S1 BEACHHEAD, keep it minimal:** a SINGLE BLAS (the N-box geom, exactly RT2b)
     wrapped in a 1-instance identity TLAS; the `primitive_id` indexing the showcase relies on
     (rt_query.metal:225) is unchanged. The kernel change is `primitive_acceleration_structure` →
     `instance_acceleration_structure` + (if the intersector needs it) `instancing` tag on the
     `intersection_query`. If, on the Mac, the instance-AS path proves fiddly, the FALLBACK that still
     proves the seam is: `CreateTlas` over a 1-instance identity transform may return the child BLAS's
     primitive-AS handle directly (a degenerate "TLAS == its single BLAS") so the existing
     `primitive_acceleration_structure` kernel binds unchanged — document this as the S1 simplification,
     with the true instance-AS landing in S2. The CONTROLLER decides which on the Mac based on what compiles.
4. **`MetalDevice::SupportsHardwareRayQuery() const` override** — `return supportsRaytracing_;` cached from
   `[mtlDevice supportsRaytracing]` at device creation (M4 → true; M1/M2 → false → callers keep the CPU path).
5. **`MetalCommandBuffer::BindAccelStructure(IAccelStructure& tlas, uint32_t slot)` override** — downcast to
   `MetalAccelStructure`, on the active compute encoder `setAccelerationStructure:atBufferIndex:slot` +
   `useResource:usage:MTLResourceUsageRead` for the AS and its child BLAS handles (lift :25708-25709). Store
   it so a dispatch in the same encoder sees it (mirror how BindTexture/BindBuffer stash state).
6. **`MetalComputePipeline` / `metal_shader.mm`** — ensure the RT kernel compiles at `MTLLanguageVersion2_4`
   (intersection_query requires it; metal_shader.mm:12 notes the pin "may be needed"). If `accelStructureBinding`
   on `ComputePipelineDesc` needs a reserved slot, honor it (the showcase hardcodes buffer(3); map the seam
   slot to that).
7. **CMake** — add `metal_accel.mm` to the `metal_headless` (and any `hf_engine` Metal) target sources.
8. **The proof showcase** — add `--rt2-query-rhi <out.bmp>` in `visual_test.mm` (clone `RunRt2QueryHwShowcase`)
   that builds the accel via `device->CreateBlas` + `device->CreateTlas`, binds via `cmd->BindAccelStructure`,
   dispatches the (possibly instance-AS-adapted) `rt_query` kernel, and asserts `memcmp(hwImage, cpuImage)
   == 0` (byte-equal to `rtrace::RenderScene`) + two-run byte-identical — the SAME proofs as `--rt2-query-hw`,
   but THROUGH THE RHI SEAM. (Keep `--rt2-query-hw` as-is for comparison.)

## Proof / golden
- `--rt2-query-rhi` on the M4: `SupportsHardwareRayQuery()` true, accel built + bound via the seam, HW image
  `memcmp == 0` to the CPU `rtrace::RenderScene`, two runs byte-identical. AND the produced image equals the
  committed `tests/golden/metal/rt2_query.png` (the existing RT2 Metal golden — same scene) byte-for-byte.
- No NEW golden needed if `--rt2-query-rhi` reproduces the existing `rt2_query` scene; otherwise register a
  `rt2_query_rhi` golden + bake it. (Controller decides on the Mac.)
- Determinism is STRICT (memcmp 0.0000) — this is an integer-reconciled RT slice, NOT a float visresolve slice.

## Constraints (HARD)
- ADDITIVE to the RHI: only Metal-side OVERRIDES of the existing `rhi.h` virtuals (the seam is already
  defined + inert). Do NOT change `rhi.h` (the seam exists) except — if genuinely required — an additive
  defaulted member; prefer NOT to. Do NOT touch the Vulkan backend, the CPU `rtrace.h` reference (frozen),
  or any existing golden. The Metal accel code is NEW files + overrides; existing Metal showcases unaffected.
- The change must keep every EXISTING Metal golden byte-identical (the additions are inert until
  `--rt2-query-rhi` invokes them).
- Branch `fix-metalrt-s1`. Commit the draft even though it can't compile on Windows (note that in the
  report). Do NOT merge. Commit message via temp file + `git commit -F`.
- COMPLETION (implementer): the code is written + committed to the branch, accurately lifted from the cited
  proven sources, with the seam overrides + the `--rt2-query-rhi` showcase + CMake wiring. REPORT: the
  commit hash, every file added/changed, the exact lift points used, the CreateTlas approach chosen (true
  instance AS vs the degenerate-single-BLAS S1 simplification) + WHY, and any spots you're unsure compile
  on Metal (flag them for the controller's Mac pass). Do NOT claim it builds — you can't compile it.
  (The CONTROLLER then: push branch → Mac clone → `cmake -S metal_headless` build → iterate MSL/Metal
  compile fixes → run `--rt2-query-rhi` → confirm memcmp byte-equal CPU + the `rt2_query.png` golden +
  two-run 0.0000 → confirm existing Metal goldens unaffected → ff-merge → advance to S2.)
