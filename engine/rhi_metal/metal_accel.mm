// Slice METAL-RT S1 — Hardware Ray Tracing: the Metal acceleration-structure backend behind the RT1
// IAccelStructure seam (MetalDevice::CreateBlas / CreateTlas + MetalAccelStructure). The Metal twin of
// rhi_vulkan/vulkan_accel.cpp. Builds a procedural-AABB BLAS (each AABB inflated by kRtAabbMargin per the
// determinism contract) and a TLAS over the child BLAS(es), SYNCHRONOUSLY (one-shot command buffer +
// waitUntilCompleted — the showcase build pattern), then stores the id<MTLAccelerationStructure> for the
// inline-ray-query kernel.
//
// LIFTED VERBATIM from the PROVEN headless RT2b showcase metal_headless/visual_test.mm:25652-25670
// (RunRt2QueryHwShowcase): MTLAccelerationStructureBoundingBoxGeometryDescriptor ->
// MTLPrimitiveAccelerationStructureDescriptor -> accelerationStructureSizesWithDescriptor ->
// newAccelerationStructureWithSize -> accelerationStructureCommandEncoder -> buildAccelerationStructure ->
// commit/waitUntilCompleted. The margin inflation mirrors visual_test.mm:25624 + vulkan_accel.cpp:79-80.
//
// SEAM: reached ONLY by the new --rt2-query-rhi showcase; every existing path is byte-for-byte unaffected
// (CreateBlas/CreateTlas were defaulted-no-op and no existing Metal pipeline binds an accel structure).
//
// ===== CreateTlas approach (S1): the DEGENERATE-SINGLE-BLAS simplification. =====
// rt_query.metal declares `primitive_acceleration_structure accel [[buffer(3)]]` (a BLAS) and indexes the
// parallel gPrims[] info buffer via get_candidate_primitive_id() (rt_query.metal:224). A TRUE TLAS is an
// `instance_acceleration_structure`, which would force rt_query.metal -> instance_acceleration_structure +
// an `instancing`-tagged intersection_query and a per-instance/per-geometry primitive_id read — a Metal API
// change that is fiddly to get right BLIND (no Metal SDK on Windows). Per the spec CRUX/fallback (lines
// 59-71), for the S1 BEACHHEAD a 1-instance IDENTITY TLAS over a single BLAS returns/shares the child BLAS's
// PRIMITIVE-AS handle directly (a "TLAS == its single BLAS"), so the existing primitive_acceleration_structure
// kernel + its proven (t,primIndex) proof bind UNCHANGED. The true instance-AS lands in S2 (the controller
// can promote it on the Mac if instance_acceleration_structure compiles cleanly). The TLAS object still
// retains the child BLAS handle + exposes it via ChildHandles() so BindAccelStructure's useResource:
// discipline is fully exercised.
//
// ===== Slice RT7 (Track-R R9): the REAL MULTI-INSTANCE TLAS. =====
// CreateTlas now BRANCHES on the instance count: a 1-instance TlasDesc keeps the S1 degenerate path above
// BYTE-FOR-BYTE (every pre-RT7 caller — rt2-query-rhi/rt3/rt4/rt6 — passes exactly one identity instance,
// so their goldens are untouched, and the returned handle stays a primitive AS the existing
// primitive_acceleration_structure kernels bind unchanged). N >= 2 instances build the TRUE
// MTLInstanceAccelerationStructureDescriptor path: one MTLAccelerationStructureUserIDInstanceDescriptor
// per TlasInstance (row-major transform[12] -> the column-major MTLPackedFloat4x3, mask 0xFF,
// userID = instanceId — matching the Vulkan VkAccelerationStructureInstanceKHR::instanceCustomIndex
// semantics, vulkan_accel.cpp:177-192) over the deduped child-BLAS array. The result is an INSTANCE AS:
// it binds to an `instance_acceleration_structure` kernel (shaders/rt_instanced.metal) — NOT to the
// primitive_acceleration_structure kernels the 1-instance path serves.
#include "rhi_metal/metal_accel.h"
#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_buffer.h"
#include "rhi_metal/metal_common.h"

#include <vector>

namespace hf::rhi::mtl {

// --- CreateBlas: a procedural-AABB bottom-level (primitive) acceleration structure ------------------
std::unique_ptr<IAccelStructure> MetalDevice::CreateBlas(const BlasDesc& desc) {
    if (!supportsRaytracing_) return nullptr;

    // 1) Build the float MTLAxisAlignedBoundingBox array — each AABB INFLATED by kRtAabbMargin so the
    //    driver's float BVH overlap is a strict SUPERSET of every true fx hit (the candidate-completeness
    //    guarantee). Triangle geoms are out of scope (Tier B / RT6) — skipped. Mirrors vulkan_accel.cpp:76-82
    //    and the showcase's bbox build at visual_test.mm:25636-25641.
    std::vector<MTLAxisAlignedBoundingBox> boxes;
    boxes.reserve(desc.geoms.size());
    for (const AccelGeometry& g : desc.geoms) {
        if (g.kind != AccelGeometry::Kind::AabbProcedural) continue;
        MTLAxisAlignedBoundingBox bb;
        bb.min.x = g.lo.x - kRtAabbMargin; bb.min.y = g.lo.y - kRtAabbMargin; bb.min.z = g.lo.z - kRtAabbMargin;
        bb.max.x = g.hi.x + kRtAabbMargin; bb.max.y = g.hi.y + kRtAabbMargin; bb.max.z = g.hi.z + kRtAabbMargin;
        boxes.push_back(bb);
    }
    if (boxes.empty()) return nullptr;
    const uint32_t primCount = (uint32_t)boxes.size();

    // 2) Upload the bbox data to a shared buffer (Apple Silicon: CPU+GPU visible).
    id<MTLBuffer> bboxBuf = [device_ newBufferWithBytes:boxes.data()
                                                 length:boxes.size() * sizeof(MTLAxisAlignedBoundingBox)
                                                options:MTLResourceStorageModeShared];
    if (!bboxBuf) return nullptr;

    // 3) Describe the AABB geometry -> primitive AS descriptor -> sizes (LIFTED visual_test.mm:25652-25659).
    MTLAccelerationStructureBoundingBoxGeometryDescriptor* geo =
        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
    geo.boundingBoxBuffer = bboxBuf;
    geo.boundingBoxCount = primCount;
    MTLPrimitiveAccelerationStructureDescriptor* asDesc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    asDesc.geometryDescriptors = @[geo];
    MTLAccelerationStructureSizes sizes = [device_ accelerationStructureSizesWithDescriptor:asDesc];

    // 4) Allocate the AS + a transient scratch buffer (LIFTED visual_test.mm:25660-25662).
    id<MTLAccelerationStructure> accel =
        [device_ newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratch = [device_ newBufferWithLength:sizes.buildScratchBufferSize
                                                 options:MTLResourceStorageModePrivate];
    if (!accel || !scratch) return nullptr;

    // 5) Encode + submit the build, then WAIT (synchronous load path — LIFTED visual_test.mm:25663-25670).
    {
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> enc = [cb accelerationStructureCommandEncoder];
        [enc buildAccelerationStructure:accel descriptor:asDesc scratchBuffer:scratch scratchBufferOffset:0];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) return nullptr;  // accel build failed
    }

    auto out = std::make_unique<MetalAccelStructure>();
    out->SetHandle(accel);
    out->RetainBuffer(bboxBuf);  // the BLAS references the bbox buffer; keep it alive for the AS's lifetime.
    return out;
}

// --- CreateTlas: 1 instance -> the S1 degenerate single-BLAS "TLAS == its single BLAS";
//     N >= 2 instances -> the RT7 REAL instance acceleration structure (see file header) ----------------
std::unique_ptr<IAccelStructure> MetalDevice::CreateTlas(const TlasDesc& desc) {
    if (!supportsRaytracing_) return nullptr;
    if (desc.instances.empty()) return nullptr;

    // ===== RT7: N >= 2 instances -> the TRUE two-level instance AS =====
    if (desc.instances.size() >= 2) {
        // 1) One MTLAccelerationStructureUserIDInstanceDescriptor per TlasInstance. The child BLAS handles
        //    are DEDUPED (in first-appearance order) into the instancedAccelerationStructures array;
        //    accelerationStructureIndex points each instance at its BLAS. The row-major 3x4 transform[12]
        //    (rhi.h TlasInstance, row r = transform[r*4 .. r*4+3]) transposes into the COLUMN-major
        //    MTLPackedFloat4x3 (columns[c] = {transform[0*4+c], transform[1*4+c], transform[2*4+c]};
        //    columns[3] = the translation). userID = instanceId (the Vulkan instanceCustomIndex twin,
        //    surfaced to the kernel via get_candidate_user_instance_id()); mask = 0xFF (matches the
        //    Vulkan inst.mask and the kernels' all-pass 0xFF query mask).
        std::vector<id<MTLAccelerationStructure>> children;
        std::vector<MTLAccelerationStructureUserIDInstanceDescriptor> idescs;
        idescs.reserve(desc.instances.size());
        for (const TlasInstance& ti : desc.instances) {
            if (!ti.blas) continue;
            id<MTLAccelerationStructure> h = static_cast<MetalAccelStructure*>(ti.blas)->Handle();
            if (!h) continue;
            uint32_t childIdx = 0;
            while (childIdx < children.size() && children[childIdx] != h) ++childIdx;
            if (childIdx == children.size()) children.push_back(h);

            MTLAccelerationStructureUserIDInstanceDescriptor d{};
            for (int c = 0; c < 4; ++c)
                d.transformationMatrix.columns[c] = MTLPackedFloat3Make(
                    ti.transform[0 * 4 + c], ti.transform[1 * 4 + c], ti.transform[2 * 4 + c]);
            d.options = MTLAccelerationStructureInstanceOptionNone;
            d.mask = 0xFFu;
            d.intersectionFunctionTableOffset = 0;
            d.accelerationStructureIndex = childIdx;
            d.userID = ti.instanceId;
            idescs.push_back(d);
        }
        if (idescs.empty()) return nullptr;

        // 2) Upload the instance descriptors to a shared buffer (the CreateBlas bbox-upload idiom).
        id<MTLBuffer> instBuf = [device_
            newBufferWithBytes:idescs.data()
                        length:idescs.size() * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor)
                       options:MTLResourceStorageModeShared];
        if (!instBuf) return nullptr;

        // 3) The instance-AS descriptor over the child BLAS array + the descriptor buffer.
        NSMutableArray<id<MTLAccelerationStructure>>* childArray =
            [NSMutableArray arrayWithCapacity:children.size()];
        for (id<MTLAccelerationStructure> c : children) [childArray addObject:c];

        MTLInstanceAccelerationStructureDescriptor* asDesc =
            [MTLInstanceAccelerationStructureDescriptor descriptor];
        asDesc.instancedAccelerationStructures = childArray;
        asDesc.instanceCount = (NSUInteger)idescs.size();
        asDesc.instanceDescriptorBuffer = instBuf;
        asDesc.instanceDescriptorBufferOffset = 0;
        asDesc.instanceDescriptorStride = sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
        asDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;

        // 4) Size + allocate + synchronous build (LIFTED from CreateBlas steps 3-5 above).
        MTLAccelerationStructureSizes sizes = [device_ accelerationStructureSizesWithDescriptor:asDesc];
        id<MTLAccelerationStructure> accel =
            [device_ newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        id<MTLBuffer> scratch = [device_ newBufferWithLength:sizes.buildScratchBufferSize
                                                     options:MTLResourceStorageModePrivate];
        if (!accel || !scratch) return nullptr;
        {
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLAccelerationStructureCommandEncoder> enc = [cb accelerationStructureCommandEncoder];
            [enc buildAccelerationStructure:accel descriptor:asDesc scratchBuffer:scratch scratchBufferOffset:0];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) return nullptr;  // tlas build failed
        }

        // 5) The TLAS retains its instance-descriptor buffer AND every child BLAS handle (so they outlive
        //    it; BindAccelStructure useResource:'s the TLAS + each child for the GPU traversal).
        auto out = std::make_unique<MetalAccelStructure>();
        out->SetHandle(accel);
        out->RetainBuffer(instBuf);
        for (id<MTLAccelerationStructure> c : children) out->AddChild(c);
        return out;
    }

    // S1 SIMPLIFICATION: a 1-instance identity TLAS shares the single child BLAS's primitive-AS handle, so
    // the existing primitive_acceleration_structure rt_query.metal kernel binds UNCHANGED. We still build a
    // distinct MetalAccelStructure object that RETAINS the child BLAS handle + records it as a child, so
    // BindAccelStructure useResource:'s both the (shared) AS handle and the child — fully exercising the
    // seam's resource-residency discipline. (The true MTLInstanceAccelerationStructureDescriptor path —
    // one MTLAccelerationStructureInstanceDescriptor per TlasInstance with transform[12] -> MTLPackedFloat4x3
    // + userID=instanceId over instancedAccelerationStructures — lands in S2; the controller may promote it
    // on the Mac if instance_acceleration_structure compiles. See vulkan_accel.cpp:165-255 for the shape.)
    const TlasInstance& first = desc.instances[0];
    if (!first.blas) return nullptr;
    auto* childBlas = static_cast<MetalAccelStructure*>(first.blas);
    id<MTLAccelerationStructure> childHandle = childBlas->Handle();
    if (!childHandle) return nullptr;

    auto out = std::make_unique<MetalAccelStructure>();
    out->SetHandle(childHandle);    // share the child BLAS's primitive AS as the bound structure.
    out->AddChild(childHandle);     // record it as a child so BindAccelStructure useResource:'s it.
    return out;
}

// --- SupportsHardwareRayQuery: the cached [mtlDevice supportsRaytracing] (set at device creation) ----
bool MetalDevice::SupportsHardwareRayQuery() const {
    return supportsRaytracing_;
}

} // namespace hf::rhi::mtl
