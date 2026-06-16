// Slice CL — Clustered Light Culling (Forward+): the LIGHT-ASSIGNMENT compute pass. ONE thread per
// cluster (dispatched as one workgroup of HF_CLUSTER_THREADS, looping if there are more clusters than
// threads). Each thread computes its cluster's VIEW-space AABB (the EXACT mirror of
// hf::render::cluster::ClusterViewAABB) and tests EVERY light's view-space sphere against it
// (SphereAABBIntersect), writing the cluster's overlapping light indices into a fixed
// MAX_LIGHTS_PER_CLUSTER slot array in ASCENDING light index (a DETERMINISTIC ORDERED fill — NO
// atomics — so the per-cluster list order is IDENTICAL to the CPU AssignLights reference). It also
// accumulates the global assigned-light total (the showcase asserts == the CPU reference total).
//
// CONSERVATIVE / NO-DROP: the cap is sized (kMaxLightsPerCluster=64) so no cluster overflows for the
// showcase; if a cluster would overflow we record an overflow flag (the host asserts it stays 0) and
// stop appending — we NEVER silently drop a contributing light without surfacing it.
//
// The math (exponential Z slices, NDC-corner unprojection via invProj, sphere/AABB closest-point
// test) is the verbatim mirror of engine/render/cluster.h. A mismatch shows up as a light wrongly
// excluded from a cluster it contributes to -> the clustered shade diverges from brute-force (caught
// by the showcase's byte-identical assertion).
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3)):
//   b0 gLights      : the point lights (view-space pos+radius, color+intensity), READ.
//   b1 gClusterList : per-cluster {count, idx[MAX]} fixed-slot arrays, WRITE (ordered).
//   b2 gTotals      : [0]=assignedTotal, [1]=overflow flag, WRITE (atomics, diagnostics only).
//   b3 gParams      : grid dims + zNear/zFar + light count + invProj (the cluster math inputs), READ.

#define HF_MAX_LIGHTS_PER_CLUSTER 96
#define HF_CLUSTER_THREADS        256

struct GpuLight {
    float4 posRadius;   // xyz = VIEW-space position, w = radius
    float4 color;       // rgb = color, w = intensity
};

// One cluster's fixed-cap ordered light list: count + MAX slots. Matches the C++ upload struct
// (std430): uint count + uint pad[3] (16B align) + uint idx[96] = 16 + 384 = 400 bytes.
struct ClusterList {
    uint count;
    uint pad0, pad1, pad2;
    uint idx[HF_MAX_LIGHTS_PER_CLUSTER];
};

// Cluster-grid + camera params (std430). dims(dimX,dimY,dimZ,lightCount) + range(zNear,zFar,_,_) +
// invProj (column-major float4x4).
struct Params {
    uint4    dims;       // x=dimX, y=dimY, z=dimZ, w=lightCount
    float4   range;      // x=zNear, y=zFar, z/w unused
    float4x4 invProj;    // proj.Inverse(), column-major (matches engine Mat4)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<GpuLight>    gLights      : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<ClusterList> gClusterList : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>        gTotals      : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<Params>      gParams      : register(u3);

// SliceZ(k): positive view-distance of exponential slice boundary k. Mirrors cluster.h::SliceZ.
float SliceZ(uint k, uint dimZ, float zNear, float zFar) {
    return zNear * pow(zFar / zNear, (float)k / (float)dimZ);
}

// Unproject NDC (nx,ny) at positive view-distance vz to view space, via invProj. Mirrors
// cluster.h::UnprojectToView: form the NDC ray (z=0), then scale to the plane z = -vz.
float3 UnprojectToView(float4x4 invProj, float nx, float ny, float vz) {
    float4 h = mul(invProj, float4(nx, ny, 0.0, 1.0));
    float3 vDir = h.xyz / h.w;             // a view-space point on the ray through the eye
    float  s = (abs(vDir.z) > 1e-9) ? (-vz / vDir.z) : 0.0;
    return float3(vDir.x * s, vDir.y * s, -vz);
}

// The view-space AABB of cluster (cx,cy,cz). Mirrors cluster.h::ClusterViewAABB EXACTLY.
void ClusterViewAABB(uint cx, uint cy, uint cz, uint dimX, uint dimY, uint dimZ,
                     float zNear, float zFar, float4x4 invProj,
                     out float3 outMin, out float3 outMax) {
    float zNearK = SliceZ(cz,     dimZ, zNear, zFar);
    float zFarK  = SliceZ(cz + 1, dimZ, zNear, zFar);
    float nx0 = ((float)cx       / (float)dimX) * 2.0 - 1.0;
    float nx1 = ((float)(cx + 1) / (float)dimX) * 2.0 - 1.0;
    float ny0 = ((float)cy       / (float)dimY) * 2.0 - 1.0;
    float ny1 = ((float)(cy + 1) / (float)dimY) * 2.0 - 1.0;

    float minX = 1e30, minY = 1e30, maxX = -1e30, maxY = -1e30;
    float zs[2]  = {zNearK, zFarK};
    float nxs[2] = {nx0, nx1};
    float nys[2] = {ny0, ny1};
    [unroll] for (int zi = 0; zi < 2; ++zi)
        [unroll] for (int xi = 0; xi < 2; ++xi)
            [unroll] for (int yi = 0; yi < 2; ++yi) {
                float3 v = UnprojectToView(invProj, nxs[xi], nys[yi], zs[zi]);
                minX = min(minX, v.x); maxX = max(maxX, v.x);
                minY = min(minY, v.y); maxY = max(maxY, v.y);
            }
    outMin = float3(minX, minY, -zFarK);
    outMax = float3(maxX, maxY, -zNearK);
}

// Sphere/AABB closest-point test. Mirrors cluster.h::SphereAABBIntersect.
bool SphereAABBIntersect(float3 c, float radius, float3 mn, float3 mx) {
    float dx = c.x < mn.x ? mn.x - c.x : (c.x > mx.x ? c.x - mx.x : 0.0);
    float dy = c.y < mn.y ? mn.y - c.y : (c.y > mx.y ? c.y - mx.y : 0.0);
    float dz = c.z < mn.z ? mn.z - c.z : (c.z > mx.z ? c.z - mx.z : 0.0);
    float d2 = dx * dx + dy * dy + dz * dz;
    return d2 <= radius * radius;
}

[numthreads(HF_CLUSTER_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint dimX = gParams[0].dims.x;
    uint dimY = gParams[0].dims.y;
    uint dimZ = gParams[0].dims.z;
    uint lightCount = gParams[0].dims.w;
    float zNear = gParams[0].range.x;
    float zFar  = gParams[0].range.y;
    float4x4 invProj = gParams[0].invProj;

    uint clusterCount = dimX * dimY * dimZ;

    // One thread per cluster; loop in case clusterCount > thread count (grid-stride over the single
    // workgroup's threads — DispatchCompute is called with enough groups to cover all clusters).
    uint c = gid.x;
    if (c >= clusterCount) return;

    // Decode (cx,cy,cz) from the flat index: idx = cx + cy*dimX + cz*(dimX*dimY).
    uint cz  = c / (dimX * dimY);
    uint rem = c % (dimX * dimY);
    uint cy  = rem / dimX;
    uint cx  = rem % dimX;

    float3 mn, mx;
    ClusterViewAABB(cx, cy, cz, dimX, dimY, dimZ, zNear, zFar, invProj, mn, mx);

    // Ordered fill: iterate lights in ASCENDING index, append the overlapping ones (no atomics ->
    // identical order to the CPU AssignLights reference).
    uint count = 0;
    [loop] for (uint li = 0; li < lightCount; ++li) {
        float4 pr = gLights[li].posRadius;
        if (SphereAABBIntersect(pr.xyz, pr.w, mn, mx)) {
            if (count < HF_MAX_LIGHTS_PER_CLUSTER) {
                gClusterList[c].idx[count] = li;
                count += 1u;
            } else {
                uint dummy;
                InterlockedAdd(gTotals[1], 1u, dummy);   // overflow (host asserts == 0)
            }
        }
    }
    gClusterList[c].count = count;
    gClusterList[c].pad0 = 0u; gClusterList[c].pad1 = 0u; gClusterList[c].pad2 = 0u;

    uint dummy2;
    InterlockedAdd(gTotals[0], count, dummy2);           // global assigned total (== CPU ref)
}
